# SPDX-License-Identifier: Apache-2.0
"""
Form finding by the force density method, and the density subspaces it moves in.

The equilibrium is linear in the coordinates once the force densities are fixed,
so `jax-fdm` differentiates it by tracing the solve. Connectivity is topology,
known before any force density is chosen, and is built once on the host.
"""

import abc
from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax_fdm import DTYPE_INT_NP
from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumParametersState
from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.equilibrium import LoadState
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int
from scipy.linalg import qr

from normax.config import FormFindingConfig
from normax.config import check_start_fields
from normax.structures import DesignShape
from normax.structures import Structure
from normax.structures import compute_member_lengths
from normax.structures import read_drawn_shape
from normax.symmetry import SignGuard
from normax.symmetry import guard_signs
from normax.symmetry import permute_members
from normax.symmetry import shift_densities
from normax.symmetry import sketch_lens

# The shape parametrizations a run config may name, in the order they are
# reported: found by equilibrium, written as heights, or not moved at all.
SHAPE_PARAMETRIZATIONS = ("fdm", "heights", "fixed")


class PlanBasis(NamedTuple):
    """
    A basis of the force densities that hold the drawn plan.

    Attributes
    ----------
    columns :
        One column per coefficient, spanning the null space of the horizontal
        balance.
    independents :
        Member indices whose densities are the coefficients, or None when the
        columns are orthonormal and a coefficient is a projection.

    Notes
    -----
    Any density vector in the span keeps the drawn plan in horizontal
    equilibrium under vertical loads, so no bound on a coefficient is a bound
    on funicularity. The two read-back conventions live here: an orthonormal basis
    reads a density vector as `Bᵀ q`, a pivoted one reads off the independent
    densities, never `Bᵀ q`.
    """

    columns: Float[np.ndarray, "members coefficients"]
    independents: Int[np.ndarray, "coefficients"] | None

    @property
    def width(self) -> int:
        """
        Number of coefficients.
        """
        return int(self.columns.shape[1])

    def densities(
        self,
        xi: Float[Array, "coefficients"],
    ) -> Float[Array, "members"]:
        """
        Expand density coefficients into the density of every member.

        Parameters
        ----------
        xi :
            Coefficients along the basis columns.

        Returns
        -------
        q :
            Force density of every member, inside the span by construction.
        """
        return jnp.asarray(self.columns) @ xi

    def coefficients(
        self,
        q: Float[np.ndarray, "members"],
    ) -> Float[np.ndarray, "coefficients"]:
        """
        Read a density vector back as coefficients of the subspace.

        Parameters
        ----------
        q :
            Force density of every member.

        Returns
        -------
        xi :
            The coefficients whose expansion reproduces the densities exactly
            inside the span, and the nearest expressible ones outside it.
        """
        if self.independents is None:
            return self.columns.T @ np.asarray(q)

        return np.asarray(q)[self.independents]


class CoefficientBounds(NamedTuple):
    """
    The limits a run config states, for a finder to box its own coefficients by.

    Attributes
    ----------
    density_box :
        Box on the force densities, or None where the config names none.
    height_max :
        Height no free node may rise above, or None.
    height_min :
        Height no free node may hang below, or None.

    Notes
    -----
    One container rather than three arguments because a finder is handed every
    limit and picks the ones its coefficients are in: the same `rise_max` a
    density parametrization can only hold as a constraint row is a plain box
    bound where the coefficients are the heights themselves.
    """

    density_box: tuple[float, float] | None
    height_max: float | None
    height_min: float | None


class AbstractFormFinder(eqx.Module):
    """
    A parametrization of the shapes a structure may take in equilibrium.

    Attributes
    ----------
    basis :
        The held-plan subspace the force densities move in, or None when every
        density is a coefficient of its own.

    Notes
    -----
    Built from the structure it is to shape, and from nothing else that varies.
    Concrete form finders differ in which quantities they treat as independent,
    not in the mechanics they encode. The basis is the finder's own: a search
    moves its coefficients, and the finder alone knows how they expand.
    """

    basis: eqx.AbstractVar[PlanBasis | None]

    @abc.abstractmethod
    def __call__(
        self,
        q: Float[Array, "members"],
        loads: Float[Array, "nodes 3"],
    ) -> DesignShape:
        """
        Find the shape that carries a load case at given force densities.

        Parameters
        ----------
        q :
            Force density of every member. Negative in compression.
        loads :
            Force applied at every node.

        Returns
        -------
        shape :
            The geometry at equilibrium, and its member lengths.
        """

    @abc.abstractmethod
    def count_shape_coefficients(self) -> int:
        """
        How many coefficients a call's parameters are expanded from.
        """

    def expand_shape_coefficients(
        self,
        coefficients: Float[Array, "coefficients"],
    ) -> Float[Array, "shape_parameters"]:
        """
        What a call takes, at given coefficients.

        Parameters
        ----------
        coefficients :
            The basis coefficients, or the parameters where there is no basis.

        Returns
        -------
        parameters :
            What `__call__` is handed, in this finder's own space.
        """
        if self.basis is None:
            return coefficients

        return self.basis.densities(coefficients)

    def read_shape_coefficients(
        self,
        parameters: Float[np.ndarray, "shape_parameters"],
    ) -> Float[np.ndarray, "coefficients"]:
        """
        The coefficients a set of parameters reads back as, on the host.

        Parameters
        ----------
        parameters :
            What `__call__` is handed, in this finder's own space.

        Returns
        -------
        coefficients :
            The parameters themselves, or their coefficients in the basis.
        """
        if self.basis is None:
            return np.asarray(parameters)

        return self.basis.coefficients(parameters)

    def bound_coefficients(
        self,
        limits: CoefficientBounds,
    ) -> list[tuple[float | None, float | None]]:
        """
        One bound pair per coefficient a call expands from.

        Parameters
        ----------
        limits :
            Every limit the run config states, of which this finder takes the
            ones its coefficients are in.

        Returns
        -------
        boxes :
            The density box on every coefficient, or nothing where the config
            names none.

        Notes
        -----
        The finder is asked rather than told, because only it knows what its
        coefficients mean. A density box belongs on densities alone, and the
        height limits cannot be a box here — a density does not say what
        height it reaches without a solve, so they reach a form-found shape as
        constraint rows instead.
        """
        boxed = limits.density_box or (None, None)

        return [boxed] * self.count_shape_coefficients()

    def read_sign_guard(self, guard: SignGuard | None) -> SignGuard | None:
        """
        The sign guard that applies to this finder's coefficients.

        Parameters
        ----------
        guard :
            The guard a run config asked for, or None for none.

        Returns
        -------
        guard :
            The guard itself, whose rows are linear in the force densities.

        Notes
        -----
        A guard keeps a density off zero, where the force density system turns
        singular. A finder that is not called with densities has no such sheet
        to stay on and answers None, so the rows are never built against a
        quantity they were not written for.
        """
        return guard


def build_equilibrium_graph(structure: Structure) -> EquilibriumStructure:
    """
    The connectivity the force density method solves on.

    Parameters
    ----------
    structure :
        The structure to read nodes, edges and supports from.

    Returns
    -------
    graph :
        The connectivity matrices and the free-fixed node partition.
    """
    num_nodes = structure.num_nodes

    nodes = np.arange(num_nodes, dtype=DTYPE_INT_NP)
    edges = np.asarray(structure.edges, dtype=DTYPE_INT_NP)

    supports = np.zeros(num_nodes, dtype=DTYPE_INT_NP)
    supports[np.asarray(structure.supports)] = 1

    return EquilibriumStructure(nodes, edges, supports)


def solve_equilibrium(
    q: Float[Array, "members"],
    xyz_fixed: Float[Array, "supports 3"],
    graph: EquilibriumStructure,
    loads: Float[Array, "nodes 3"],
) -> EquilibriumState:
    """
    The geometry that carries the loads at a given set of force densities.

    Parameters
    ----------
    q :
        Force density of every member. Negative in compression.
    xyz_fixed :
        Position of every supported node, in the order `graph.indices_fixed`
        gives them.
    graph :
        The connectivity, from `build_equilibrium_graph`.
    loads :
        Force applied at every node.

    Returns
    -------
    state :
        Node positions, member lengths, member forces and nodal residuals.

    Notes
    -----
    One linear force density step, so the loads stay fixed in direction and
    magnitude rather than following the shape.
    """
    load_state = LoadState(nodes=loads, edges=0.0, faces=0.0)
    params = EquilibriumParametersState(q=q, xyz_fixed=xyz_fixed, loads=load_state)

    return EquilibriumModel(tmax=1)(params, graph)


class FdmFormFinder(AbstractFormFinder):
    """
    The force density method, as a block of the design pipeline.

    Attributes
    ----------
    xyz_fixed :
        Position of every supported node, which the shape is hung from.
    graph :
        The connectivity the method solves on.
    basis :
        The held-plan subspace the densities move in, or None.
    num_members :
        How many members the structure has, the width without a basis.

    Notes
    -----
    The block differentiates by tracing its own solve and carries no rule of
    its own. Only the supported positions survive a solve, so they are all the
    block keeps of the starting geometry.
    """

    xyz_fixed: Float[Array, "supports 3"]
    graph: EquilibriumStructure
    basis: PlanBasis | None
    num_members: int = eqx.field(static=True)

    def __init__(self, structure: Structure, basis: PlanBasis | None = None) -> None:
        """
        Build a form finder on a structure's connectivity.

        Parameters
        ----------
        structure :
            The structure supplying the topology and the supported nodes.
        basis :
            The held-plan subspace the densities move in, or None to move every
            density freely.
        """
        graph = build_equilibrium_graph(structure)

        self.xyz_fixed = structure.nodes[graph.indices_fixed]
        self.graph = graph
        self.basis = basis
        self.num_members = int(structure.num_edges)

    def count_shape_coefficients(self) -> int:
        """
        The basis width, or the member count where every density moves freely.
        """
        if self.basis is None:
            return self.num_members

        return self.basis.width

    def __call__(
        self,
        q: Float[Array, "members"],
        loads: Float[Array, "nodes 3"],
    ) -> DesignShape:
        """
        Find the shape that carries a load case at given force densities.

        Parameters
        ----------
        q :
            Force density of every member. Negative in compression.
        loads :
            Force applied at every node.

        Returns
        -------
        shape :
            The geometry at equilibrium, and its member lengths.
        """
        state = solve_equilibrium(q, self.xyz_fixed, self.graph, loads)

        return DesignShape(state.xyz, state.lengths[:, 0])


class HeightsFormFinder(AbstractFormFinder):
    """
    Free heights: the coefficients are the free nodes' height, in the drawn plan.

    Attributes
    ----------
    xyz :
        The drawn geometry, whose plan and supports every shape keeps.
    edges :
        The two node indices spanned by every member.
    nodes_free :
        Indices of the nodes whose height a call writes.
    width :
        How many heights a call takes.
    basis :
        None: a height is its own coefficient, and no subspace holds them.

    Notes
    -----
    Not funicular: the loads are accepted and ignored, so the frame analysis
    downstream sees whatever bending the heights raise. Holding the plan by
    never moving it bounds every member length below by its plan projection,
    except where two nodes share a plan position -- a Vierendeel vertical --
    which a height crossing can still collapse. The length floor walls that
    off.

    The rival the shipped finder is measured against: the same members, loads,
    analysis and check, differing only in whether the geometry answers to
    equilibrium or is written down.
    """

    xyz: Float[Array, "nodes 3"]
    edges: Int[np.ndarray, "members 2"]
    nodes_free: Int[np.ndarray, "nodes_free"]
    width: int = eqx.field(static=True)
    basis: PlanBasis | None

    def __init__(self, structure: Structure) -> None:
        """
        Build a heights finder on a drawn structure.

        Parameters
        ----------
        structure :
            The structure supplying the plan, the members and the supports.
        """
        nodes_free = select_free_nodes(structure)

        self.xyz = jnp.asarray(structure.nodes)
        self.edges = np.asarray(structure.edges)
        self.nodes_free = nodes_free
        self.width = int(nodes_free.size)
        self.basis = None

    def count_shape_coefficients(self) -> int:
        """
        How many heights a call takes, one per free node.
        """
        return self.width

    def read_shape_coefficients(
        self,
        parameters: Float[np.ndarray, "shape_parameters"],
    ) -> Float[np.ndarray, "nodes_free"]:
        """
        The heights this finder starts from, which are the drawn ones.

        Parameters
        ----------
        parameters :
            Accepted and ignored: a fitted density says nothing about where a
            written geometry leaves from.

        Returns
        -------
        heights :
            Height of every free node as drawn.
        """
        return np.asarray(self.xyz)[self.nodes_free, 2]

    def bound_coefficients(
        self,
        limits: CoefficientBounds,
    ) -> list[tuple[float | None, float | None]]:
        """
        The sag floor and the rise ceiling, as a box on every height.

        Parameters
        ----------
        limits :
            Every limit the run config states. The density box among them is
            ignored, being a box on a quantity this finder never sees.

        Returns
        -------
        boxes :
            One `(sag_min, rise_max)` pair per free node, either end open where
            the config names no limit.

        Notes
        -----
        Here the coefficients *are* the heights, so the same limits a
        form-found shape can only be held to by constraint rows are held
        natively by the inner solver — no penalty, no multiplier, and no
        iterate outside them. The rows are emitted as well and are then
        redundant rather than wrong: a limit is stated once in the file and
        every parametrization answers to it, which is the point.

        Applying the *density* box here would be a disaster rather than a
        nuisance: a compression box is negative on both ends, so every node
        would be driven under the ground plane.
        """
        boxed = (limits.height_min, limits.height_max)

        return [boxed] * self.width

    def read_sign_guard(self, guard: SignGuard | None) -> SignGuard | None:
        """
        None: there are no densities here to keep off zero.
        """
        return None

    def __call__(
        self,
        heights: Float[Array, "nodes_free"],
        loads: Float[Array, "nodes 3"],
    ) -> DesignShape:
        """
        The drawn geometry with the free nodes lifted to the given heights.

        Parameters
        ----------
        heights :
            Height of every free node.
        loads :
            Accepted and ignored.

        Returns
        -------
        shape :
            The geometry, and its member lengths.
        """
        xyz = self.xyz.at[self.nodes_free, 2].set(heights)
        lengths = compute_member_lengths(xyz, self.edges)

        return DesignShape(xyz, lengths)


class FixedFormFinder(AbstractFormFinder):
    """
    Sizing only: the shape is the drawn geometry, whatever it is called with.

    Attributes
    ----------
    shape :
        The drawn geometry and its member lengths, measured once.
    basis :
        None: there are no coefficients for a subspace to hold.

    Notes
    -----
    A search over this finder moves the diameters alone, and every "off"
    behavior follows from its coefficient count being zero rather than from a
    branch anywhere downstream: the variable vector is all diameters, the
    coefficient half of the box is empty, and the geometry is a captured
    constant, so the mass differentiates through the sections and never
    through the lengths.

    The shape it hands on is a real geometry that never moves, not a stand-in
    for one -- which is what makes neutralizing this slot safe where a null
    analysis returning zero forces would not be.
    """

    shape: DesignShape
    basis: PlanBasis | None

    def __init__(self, structure: Structure) -> None:
        """
        Build a fixed finder on a structure.

        Parameters
        ----------
        structure :
            The structure supplying the geometry and the members.
        """
        self.shape = read_drawn_shape(structure)
        self.basis = None

    def count_shape_coefficients(self) -> int:
        """
        Zero: the drawn shape takes no coefficients.
        """
        return 0

    def read_shape_coefficients(
        self,
        parameters: Float[np.ndarray, "shape_parameters"],
    ) -> Float[np.ndarray, "0"]:
        """
        Nothing: a shape that never moves leaves from no coefficients.

        Parameters
        ----------
        parameters :
            Accepted and ignored.

        Returns
        -------
        coefficients :
            An empty vector, so a start is its diameters alone.
        """
        return np.zeros(0)

    def bound_coefficients(
        self,
        limits: CoefficientBounds,
    ) -> list[tuple[float | None, float | None]]:
        """
        No pairs, there being no coefficients to bound.

        Parameters
        ----------
        limits :
            Accepted and ignored. A shape that never moves answers to a height
            limit through its rows, which it either satisfies as drawn or
            cannot satisfy at all.

        Returns
        -------
        boxes :
            An empty list.
        """
        return []

    def read_sign_guard(self, guard: SignGuard | None) -> SignGuard | None:
        """
        None: there are no densities here to keep off zero.
        """
        return None

    def __call__(
        self,
        coefficients: Float[Array, "0"],
        loads: Float[Array, "nodes 3"],
    ) -> DesignShape:
        """
        The drawn geometry as it stands.

        Parameters
        ----------
        coefficients :
            Accepted and ignored, an empty vector.
        loads :
            Accepted and ignored.

        Returns
        -------
        shape :
            The drawn geometry, and its member lengths.
        """
        return self.shape


def build_form_finder(
    structure: Structure,
    basis: PlanBasis | None,
    config: FormFindingConfig,
) -> AbstractFormFinder:
    """
    The shape parametrization a run config asks for.

    Parameters
    ----------
    structure :
        The structure the block is built on.
    basis :
        The held-plan subspace the densities move in, or None. Read by the
        form-found parametrization alone.
    config :
        The parametrization, and where its densities start.

    Returns
    -------
    formfinder :
        The block filling the pipeline's first slot.

    Raises
    ------
    ValueError
        If the parametrization is not one this module knows.

    Notes
    -----
    The one place every shipping parametrization is named, as `build_analyzer`
    and `build_sizer` are for the two crossed blocks, and the one place a
    misspelled word is refused.
    """
    named = config.shape_parametrization
    if named == "fdm":
        return FdmFormFinder(structure, basis)
    if named == "heights":
        return HeightsFormFinder(structure)
    if named == "fixed":
        return FixedFormFinder(structure)

    known = ", ".join(SHAPE_PARAMETRIZATIONS)
    raise ValueError(f"shape parametrization must be one of {known}, got {named!r}")


def select_free_nodes(structure: Structure) -> Int[np.ndarray, "nodes_free"]:
    """
    Indices of the unsupported nodes, in ascending order.

    Parameters
    ----------
    structure :
        The structure to read the supports from.

    Returns
    -------
    nodes_free :
        Every node index that is not a support.
    """
    every = np.arange(structure.num_nodes)

    return np.setdiff1d(every, np.asarray(structure.supports))


def assemble_balance_rows(
    structure: Structure,
    xyz: Float[np.ndarray, "nodes 3"],
    axes: tuple[int, ...],
) -> Float[np.ndarray, "equations members"]:
    """
    Coefficient of every force density in the free nodes' balance, per axis.

    Parameters
    ----------
    structure :
        The structure supplying the members and the supports.
    xyz :
        The geometry the balance is written at.
    axes :
        Coordinate axes to write a balance row for, in row-block order.

    Returns
    -------
    balance :
        One row per free node and axis; the residual there is this matrix
        times the densities, minus the applied load.
    """
    edges = np.asarray(structure.edges)
    nodes = np.asarray(xyz)
    nodes_free = select_free_nodes(structure)
    num_edges = edges.shape[0]

    incidence = np.zeros((num_edges, nodes.shape[0]))
    incidence[np.arange(num_edges), edges[:, 0]] = 1.0
    incidence[np.arange(num_edges), edges[:, 1]] = -1.0

    blocks = [(incidence.T * (incidence @ nodes[:, axis]))[nodes_free] for axis in axes]

    return np.concatenate(blocks, axis=0)


def assemble_mirror_rows(
    structure: Structure,
    nodes_mirrored: Int[np.ndarray, "nodes"],
) -> Float[np.ndarray, "members members"]:
    """
    Rows demanding every density equal that of its mirrored member.

    Parameters
    ----------
    structure :
        The structure supplying the members the mirror permutes.
    nodes_mirrored :
        Mirror image of every node index.

    Returns
    -------
    rows :
        One row per member, zero exactly when the densities are symmetric.
    """
    members_mirrored = permute_members(np.asarray(nodes_mirrored), structure)
    rows = np.eye(structure.num_edges)
    rows[np.arange(structure.num_edges), members_mirrored] -= 1.0

    return rows


# The two conventions a held-plan basis is built in, by the name a run config uses.
BASIS_CONVENTIONS = ("pivoted", "svd")


def build_plan_basis(
    structure: Structure,
    nodes_mirrored: Int[np.ndarray, "nodes"] | None,
    convention: str,
) -> PlanBasis:
    """
    The subspace of force densities holding the drawn plan, in two conventions.

    Parameters
    ----------
    structure :
        The structure whose drawn plan is to be held.
    nodes_mirrored :
        Mirror image of every node index, or None to ask for no symmetry.
        When given the span shrinks to the densities equal on mirrored members.
    convention :
        `pivoted` for coefficients that are the densities of members QR pivoting
        elects independent, `svd` for projections on an orthonormal basis.

    Returns
    -------
    basis :
        The columns, and the independent members where the basis is pivoted.

    Raises
    ------
    ValueError
        If the convention is not one named.

    Notes
    -----
    Both conventions span the identical subspace, so switching prices the
    coefficients and never the reachable designs. The pivoted one is thrust
    network analysis's independent-edges construction: each coefficient is the
    density of one member, and every dependent density a fixed linear function
    of them, at the price of columns that are not orthonormal.
    """
    if convention not in BASIS_CONVENTIONS:
        raise ValueError(
            f"basis must be one of {BASIS_CONVENTIONS}, got {convention!r}"
        )
    balance = assemble_balance_rows(structure, structure.nodes, (0, 1))
    if nodes_mirrored is not None:
        symmetry = assemble_mirror_rows(structure, nodes_mirrored)
        balance = np.concatenate([balance, symmetry], axis=0)

    if convention == "svd":
        _, singulars, rows = np.linalg.svd(balance)
        tolerance = singulars.max() * max(balance.shape) * np.finfo(float).eps
        rank = int(np.sum(singulars > tolerance))

        return PlanBasis(rows[rank:].T, None)

    _, triangular, permutation = qr(balance, pivoting=True)
    diagonal = np.abs(np.diag(triangular))
    tolerance = diagonal.max() * max(balance.shape) * np.finfo(float).eps
    rank = int(np.sum(diagonal > tolerance))

    dependents = permutation[:rank]
    independents = np.sort(permutation[rank:])

    held = balance[:, dependents]
    thrown = -balance[:, independents]
    transfer, _, _, _ = np.linalg.lstsq(held, thrown, rcond=None)

    columns = np.zeros((balance.shape[1], independents.size))
    columns[independents, np.arange(independents.size)] = 1.0
    columns[dependents] = transfer

    return PlanBasis(columns, independents)


class DensityFit(NamedTuple):
    """
    Force densities that put a drawn geometry in equilibrium with its loads.

    Attributes
    ----------
    q :
        Force density of every member, from a least-squares fit of the balance.
    self_stresses :
        Basis of the density directions that leave the drawn geometry
        balanced, one column per state of self-stress.
    gap :
        Largest balance violation the fit leaves.
    """

    q: Float[np.ndarray, "members"]
    self_stresses: Float[np.ndarray, "members stresses"]
    gap: float


def fit_densities(
    structure: Structure,
    xyz: Float[np.ndarray, "nodes 3"],
    loads: Float[np.ndarray, "nodes 3"],
    basis: PlanBasis | None = None,
) -> DensityFit:
    """
    Fit force densities to a drawn geometry, the balance being linear in them.

    Parameters
    ----------
    structure :
        The structure supplying the topology and the supports.
    xyz :
        The drawn geometry to be equilibrated.
    loads :
        Force applied at every node.
    basis :
        Subspace to restrict the fit to, or None to fit every density freely.

    Returns
    -------
    fit :
        The fitted densities, the self-stress directions, and the largest
        balance violation left.

    Notes
    -----
    A start generator: sketch the shape wanted, read off the densities that make
    it funicular, and begin a search there. A topology with more members than
    balance rows reaches every sketch, and the surplus returns as states of
    self-stress — directions to trade member signs along without moving a node.
    """
    balance = assemble_balance_rows(structure, xyz, (0, 1, 2))
    nodes_free = select_free_nodes(structure)
    columns = [np.asarray(loads)[nodes_free, axis] for axis in (0, 1, 2)]
    applied = np.concatenate(columns)

    span = np.eye(structure.num_edges) if basis is None else basis.columns
    restricted = balance @ span

    coefficients, _, rank, _ = np.linalg.lstsq(restricted, applied, rcond=None)
    _, _, rows = np.linalg.svd(restricted)
    q = span @ coefficients
    self_stresses = span @ rows[rank:].T
    gap = float(np.abs(balance @ q - applied).max())

    return DensityFit(q, self_stresses, gap)


class SignGuardSpec(NamedTuple):
    """
    Which members a start must sign, and by how much.

    Attributes
    ----------
    signs :
        Sign each guarded member must carry, positive for tension.
    members :
        Indices of the guarded members.
    margin_fraction :
        Margin each must clear, as a share of the guarded members' median
        density. Zero or less asks for no guard.
    """

    signs: Float[np.ndarray, "guarded"]
    members: Int[np.ndarray, "guarded"]
    margin_fraction: float


class AbstractDensityInitializer(eqx.Module):
    """
    What generates the force densities a search starts from.

    Notes
    -----
    A concrete initializer states only how the densities are fitted; signing
    them is shared. Where the fit leaves a self-stress, the densities are
    shifted along it until every guarded member clears the margin. Where it
    leaves none, the fit must already clear the margin, since no shift could
    repair it.

    **The densities are all that comes back.** Scaling the guard the descent
    holds is a read off these densities and the spec the caller already has, so
    it belongs with the rest of what a design is held to and
    `normax.design.build_design_constraints` does it. What cannot leave is the
    provisional guard below: `shift_densities` needs a margin to size its
    slide, so signing the fit and scaling a guard against the fit are the same
    step.
    """

    @abc.abstractmethod
    def fit_start(
        self,
        structure: Structure,
        loads: Float[np.ndarray, "nodes 3"],
        basis: PlanBasis | None,
    ) -> DensityFit:
        """
        The densities before any sign is imposed, and their self-stresses.

        Parameters
        ----------
        structure :
            The structure as drawn.
        loads :
            Force applied at every node in the case the shape answers to.
        basis :
            The held-plan subspace the search moves in, or None.

        Returns
        -------
        fit :
            The densities, the self-stress directions, and the balance gap.
        """

    def __call__(
        self,
        structure: Structure,
        loads: Float[np.ndarray, "nodes 3"],
        basis: PlanBasis | None,
        guarded: SignGuardSpec | None,
    ) -> Float[np.ndarray, "members"]:
        """
        The start densities, signed where a guard asks them to be.

        Parameters
        ----------
        structure :
            The structure as drawn.
        loads :
            Force applied at every node in the case the shape answers to.
        basis :
            The held-plan subspace the search moves in, or None.
        guarded :
            Which members to sign and by how much, or None for no guard.

        Returns
        -------
        density_start :
            Force density of every member at the start.

        Raises
        ------
        ValueError
            If the fit has no self-stress and a guarded member misses its sign.
        """
        fit = self.fit_start(structure, loads, basis)
        if guarded is None:
            return fit.q

        fraction = max(guarded.margin_fraction, 0.0)
        guard = guard_signs(fit.q, guarded.signs, guarded.members, fraction)
        density_start = fit.q
        if fit.self_stresses.shape[1] > 0:
            density_start = shift_densities(fit.q, fit.self_stresses[:, 0], guard)
        elif guarded.margin_fraction > 0.0:
            signed = guard.signs * density_start[guard.members]
            if signed.min() < guard.margin:
                raise ValueError(
                    f"a guarded member misses its sign by {guard.margin:.4f}: "
                    f"worst signed density {signed.min():.4f}"
                )

        return density_start


class UniformDensityInitializer(AbstractDensityInitializer):
    """
    One force density in every member.

    Attributes
    ----------
    force_density :
        Force density every member starts at. Negative in compression.
    """

    force_density: float

    def __init__(self, described: dict[str, float | bool]):
        """
        Read the one density a file described.

        Parameters
        ----------
        described :
            What the file gave the start, naming `force_density` alone.
        """
        check_start_fields(described, ("force_density",))
        self.force_density = float(described["force_density"])

    def fit_start(
        self,
        structure: Structure,
        loads: Float[np.ndarray, "nodes 3"],
        basis: PlanBasis | None,
    ) -> DensityFit:
        """
        Every member at the one density; no fit, so no self-stress.
        """
        q = np.full(structure.num_edges, self.force_density)
        no_self_stress = np.zeros((structure.num_edges, 0))
        fit = DensityFit(q, no_self_stress, 0.0)

        return fit


class LensShapeInitializer(AbstractDensityInitializer):
    """
    The densities that make a lens sketched over the drawn plan funicular.

    Attributes
    ----------
    sag :
        Depth the sketch hangs the bottom chord to at midspan.
    rise :
        Height the sketch arches the top chord to at midspan.
    held_plan :
        Whether the fit is restricted to the basis, or made in the full member
        space and read into it afterwards.

    Notes
    -----
    Offered a sketch off the funicular manifold, the free least squares may
    abandon a chord and return a singular vertical stiffness; the restricted
    fit keeps plan balance exact, and its self-stress is the split between the
    hanging deck and the arching top chord.
    """

    sag: float
    rise: float
    held_plan: bool

    def __init__(self, described: dict[str, float | bool]):
        """
        Read the lens a file sketched.

        Parameters
        ----------
        described :
            What the file gave the start, naming `sag`, `rise` and `held_plan`.
        """
        check_start_fields(described, ("sag", "rise", "held_plan"))
        self.sag = float(described["sag"])
        self.rise = float(described["rise"])
        self.held_plan = bool(described["held_plan"])

    def fit_start(
        self,
        structure: Structure,
        loads: Float[np.ndarray, "nodes 3"],
        basis: PlanBasis | None,
    ) -> DensityFit:
        """
        Sketch the lens, and fit the densities that balance the loads on it.

        Raises
        ------
        ValueError
            If the fit is to be held to a basis and there is none.
        """
        if self.held_plan and basis is None:
            raise ValueError(
                "held_plan asks for a basis the form finding does not build"
            )
        lens = sketch_lens(structure, self.sag, self.rise)
        restricted = basis if self.held_plan else None

        return fit_densities(structure, lens, loads, restricted)


class DrawnShapeInitializer(AbstractDensityInitializer):
    """
    The densities that make the drawn geometry itself funicular.

    Notes
    -----
    A drawn cap under its own pressure is already compression-only, so the fit
    is the start and no sign needs shifting.
    """

    def __init__(self, described: dict[str, float | bool]):
        """
        Take the start a file wrote, which for a drawn fit is nothing.

        Parameters
        ----------
        described :
            What the file gave the start, which must name no field at all.
        """
        check_start_fields(described, ())

    def fit_start(
        self,
        structure: Structure,
        loads: Float[np.ndarray, "nodes 3"],
        basis: PlanBasis | None,
    ) -> DensityFit:
        """
        Fit the densities that balance the loads on the drawn nodes.

        Notes
        -----
        The fit is held to the basis wherever there is one, since a basis is
        built to hold this same drawn plan and the search moves nowhere else.
        Fitting freely would be right only by luck: the densities would have
        to land in the span unasked, and where they did not, the start
        reported would not be the start descended from.
        """
        xyz = np.asarray(structure.nodes)

        return fit_densities(structure, xyz, loads, basis)
