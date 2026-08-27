# Copyright 2026 Rafael Pastrana
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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

from normax.structures import Structure
from normax.symmetry import SignGuard
from normax.symmetry import guard_signs
from normax.symmetry import permute_members
from normax.symmetry import shift_densities
from normax.symmetry import sketch_lens


class FormFoundShape(NamedTuple):
    """
    The geometry a form finder settles on, and what its members measure there.

    Attributes
    ----------
    xyz :
        Position of every node at equilibrium.
    lengths :
        Length of every member.

    Notes
    -----
    The handoff downstream is a geometry — no prestress and no member forces.
    A frame analysis finds its own axial forces, and that they agree with the
    form finder's is a prediction that gets tested rather than an input.
    """

    xyz: Float[Array, "nodes 3"]
    lengths: Float[Array, "members"]


class PlanBasis(NamedTuple):
    """
    A basis of the force densities that hold the drawn plan.

    Attributes
    ----------
    columns :
        One column per coordinate, spanning the null space of the horizontal
        balance.
    independents :
        Member indices whose densities are the coordinates, or None when the
        columns are orthonormal and a coordinate is a projection.

    Notes
    -----
    Any density vector in the span keeps the drawn plan in horizontal
    equilibrium under vertical loads, so no bound on a coordinate is a bound on
    funicularity. The two read-back conventions live here: an orthonormal basis
    reads a density vector as `Bᵀ q`, a pivoted one reads off the independent
    densities, never `Bᵀ q`.
    """

    columns: Float[np.ndarray, "members coordinates"]
    independents: Int[np.ndarray, "coordinates"] | None

    @property
    def width(self) -> int:
        """
        Number of coordinates.
        """
        return int(self.columns.shape[1])

    def densities(
        self,
        xi: Float[Array, "coordinates"],
    ) -> Float[Array, "members"]:
        """
        Expand a coordinate into the density of every member.

        Parameters
        ----------
        xi :
            Coordinate along the basis columns.

        Returns
        -------
        q :
            Force density of every member, inside the span by construction.
        """
        return jnp.asarray(self.columns) @ xi

    def coordinates(
        self,
        q: Float[np.ndarray, "members"],
    ) -> Float[np.ndarray, "coordinates"]:
        """
        Read a density vector back as a coordinate of the subspace.

        Parameters
        ----------
        q :
            Force density of every member.

        Returns
        -------
        xi :
            The coordinate whose expansion reproduces the densities exactly
            inside the span, and the nearest expressible ones outside it.
        """
        if self.independents is None:
            return self.columns.T @ np.asarray(q)

        return np.asarray(q)[self.independents]


class AbstractFormFinder(eqx.Module):
    """
    A parametrization of the shapes a structure may take in equilibrium.

    Attributes
    ----------
    basis :
        The held-plan subspace the force densities move in, or None when every
        density is a coordinate of its own.

    Notes
    -----
    Built from the structure it is to shape, and from nothing else that varies.
    Concrete form finders differ in which quantities they treat as independent,
    not in the mechanics they encode. The basis is the finder's own: a search
    moves its coordinates, and the finder alone knows how they expand.
    """

    basis: eqx.AbstractVar[PlanBasis | None]

    @abc.abstractmethod
    def __call__(
        self,
        q: Float[Array, "members"],
        loads: Float[Array, "nodes 3"],
    ) -> FormFoundShape:
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
    def count_coordinates(self) -> int:
        """
        How many coordinates a call expands from.
        """

    def expand_coordinates(
        self,
        coordinates: Float[Array, "coordinates"],
    ) -> Float[Array, "members"]:
        """
        The force density of every member at given coordinates.

        Parameters
        ----------
        coordinates :
            The basis coordinates, or the densities where there is no basis.

        Returns
        -------
        q :
            Force density of every member, as a call takes it.
        """
        if self.basis is None:
            return coordinates

        return self.basis.densities(coordinates)

    def read_coordinates(
        self,
        q: Float[np.ndarray, "members"],
    ) -> Float[np.ndarray, "coordinates"]:
        """
        The coordinates a set of force densities reads back as, on the host.

        Parameters
        ----------
        q :
            Force density of every member.

        Returns
        -------
        coordinates :
            The densities themselves, or their coordinates in the basis.
        """
        if self.basis is None:
            return np.asarray(q)

        return self.basis.coordinates(q)


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

    def count_coordinates(self) -> int:
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
    ) -> FormFoundShape:
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

        return FormFoundShape(state.xyz, state.lengths[:, 0])


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
        `pivoted` for coordinates that are the densities of members QR pivoting
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
    coordinates and never the reachable designs. The pivoted one is thrust
    network analysis's independent-edges construction: each coordinate is the
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

    coordinates, _, rank, _ = np.linalg.lstsq(restricted, applied, rcond=None)
    _, _, rows = np.linalg.svd(restricted)
    q = span @ coordinates
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


class DensityStart(NamedTuple):
    """
    Where a search starts: the densities, and the guard that keeps their signs.

    Attributes
    ----------
    q :
        Force density of every member at the start.
    guard :
        The sign guard scaled at those densities, or None where none was asked.
    """

    q: Float[np.ndarray, "members"]
    guard: SignGuard | None


class AbstractDensityInitializer(eqx.Module):
    """
    What generates the force densities a search starts from.

    Notes
    -----
    A concrete initializer states only how the densities are fitted; signing
    them is shared. Where the fit leaves a self-stress, the densities are
    shifted along it until every guarded member clears the margin, and the
    guard is rescaled at the shifted densities. Where it leaves none, the fit
    must already clear the margin, since no shift could repair it. A margin of
    zero or less still signs the start but hands the descent no guard rows.
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
    ) -> DensityStart:
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
        started :
            The densities, and the guard scaled at them, or None where the
            margin asks for no rows.

        Raises
        ------
        ValueError
            If the fit has no self-stress and a guarded member misses its sign.
        """
        fit = self.fit_start(structure, loads, basis)
        if guarded is None:
            return DensityStart(fit.q, None)

        fraction = max(guarded.margin_fraction, 0.0)
        guard = guard_signs(fit.q, guarded.signs, guarded.members, fraction)
        q = fit.q
        if fit.self_stresses.shape[1] > 0:
            q = shift_densities(fit.q, fit.self_stresses[:, 0], guard)
        elif guarded.margin_fraction > 0.0:
            signed = guard.signs * q[guard.members]
            if signed.min() < guard.margin:
                raise ValueError(
                    f"a guarded member misses its sign by {guard.margin:.4f}: "
                    f"worst signed density {signed.min():.4f}"
                )
        if guarded.margin_fraction <= 0.0:
            return DensityStart(q, None)

        rescaled = guard_signs(q, guarded.signs, guarded.members, fraction)
        started = DensityStart(q, rescaled)

        return started


class UniformDensityInitializer(AbstractDensityInitializer):
    """
    One force density in every member.

    Attributes
    ----------
    force_density :
        Force density every member starts at. Negative in compression.
    """

    force_density: float

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

    def fit_start(
        self,
        structure: Structure,
        loads: Float[np.ndarray, "nodes 3"],
        basis: PlanBasis | None,
    ) -> DensityFit:
        """
        Fit the densities that balance the loads on the drawn nodes.
        """
        return fit_densities(structure, np.asarray(structure.nodes), loads)


def build_density_initializer(
    described: float | str | dict[str, dict[str, float | bool]],
) -> AbstractDensityInitializer:
    """
    The initializer a run config's `force_density` value names.

    Parameters
    ----------
    described :
        A number for one density in every member, `drawn` to fit the drawn
        geometry, or `{lens: {sag, rise, held_plan}}` to fit a lens sketch.

    Returns
    -------
    initializer :
        The generator of the start densities.

    Raises
    ------
    ValueError
        If the value is none of those forms.
    """
    if isinstance(described, bool):
        raise ValueError("force_density cannot be a boolean")
    if isinstance(described, (int, float)):
        return UniformDensityInitializer(float(described))
    if described == "drawn":
        return DrawnShapeInitializer()
    if isinstance(described, dict) and set(described) == {"lens"}:
        lens = described["lens"]
        return LensShapeInitializer(
            float(lens["sag"]), float(lens["rise"]), bool(lens["held_plan"])
        )

    raise ValueError(
        f"force_density must be a number, 'drawn' or {{lens: ...}}, got {described!r}"
    )
