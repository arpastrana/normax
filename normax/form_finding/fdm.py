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
The force density method, the form finder this package ships with.

Maps force densities to the geometry that carries the applied loads in pure
tension or compression. The equilibrium is linear in the coordinates once the
force densities are fixed, so `jax-fdm` differentiates it by tracing the solve.

The split is along the line that separates a shape from a number. Connectivity
is topology, known before any force density is chosen, and is built once on the
host by `equilibrium_graph`. Only the force densities enter the traced call.
"""

from typing import NamedTuple

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

from normax.form_finding import AbstractFormFinder
from normax.form_finding import FormFoundShape
from normax.structures import Structure


def equilibrium_graph(structure: Structure) -> EquilibriumStructure:
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

    Notes
    -----
    Built once on the host, outside any traced call. The support indices become
    the per-node flags `jax-fdm` partitions on, so a node absent from them is
    free in all three coordinates.
    """
    num_nodes = structure.num_nodes

    nodes = np.arange(num_nodes, dtype=DTYPE_INT_NP)
    edges = np.asarray(structure.edges, dtype=DTYPE_INT_NP)

    supports = np.zeros(num_nodes, dtype=DTYPE_INT_NP)
    supports[np.asarray(structure.supports)] = 1

    return EquilibriumStructure(nodes, edges, supports)


def equilibrium_state(
    q: Float[Array, "edges"],
    xyz_fixed: Float[Array, "supports 3"],
    graph: EquilibriumStructure,
    loads: Float[Array, "nodes 3"],
) -> EquilibriumState:
    """
    The geometry that carries the loads at a given set of force densities.

    Parameters
    ----------
    q :
        Force density of every edge. Negative in compression.
    xyz_fixed :
        Position of every supported node, in the order `graph.indices_fixed`
        gives them.
    graph :
        The connectivity, from `equilibrium_graph`.
    loads :
        Force applied at every node.

    Returns
    -------
    state :
        Node positions, edge lengths, edge forces, nodal residuals and loads.

    Notes
    -----
    One linear force density step, so the loads stay fixed in direction and
    magnitude rather than following the shape. Each edge carries the product of
    its force density and its length, and that state balances the applied loads
    at every free node to solver precision, which is the claim the analysis
    stage is measured against.

    **The supported positions are all this reads of a starting geometry**, which
    is why they arrive as an array rather than as the structure holding them:
    every free node's position is solved for, and the topology it is solved on
    is already in the graph. Their order is the graph's rather than the
    structure's, the two being free to differ.

    The load case is an argument and never a property of the structure, so the
    case a shape answers to is named by the caller.
    """
    load_state = LoadState(nodes=loads, edges=0.0, faces=0.0)
    params = EquilibriumParametersState(q=q, xyz_fixed=xyz_fixed, loads=load_state)

    return EquilibriumModel(tmax=1)(params, graph)


def positions_vertical(
    q: Float[Array, "edges"],
    xyz: Float[Array, "nodes 3"],
    graph: EquilibriumStructure,
    loads: Float[Array, "nodes 3"],
) -> Float[Array, "nodes 3"]:
    """
    The geometry that carries the vertical loads with the plan held fixed.

    Parameters
    ----------
    q :
        Force density of every edge. Negative in compression.
    xyz :
        Starting position of every node, supplying the plan that is held and
        the supported heights.
    graph :
        The connectivity, from `equilibrium_graph`.
    loads :
        Force applied at every node.

    Returns
    -------
    xyz :
        Position of every node, with the two horizontal coordinates as given.

    Notes
    -----
    The classical form-finding problem: the plan is chosen and only the heights
    are solved for. Because the horizontal coordinates cannot move, **no edge can
    shorten past its own projection**, which is what makes this a hard bound on
    member length rather than a penalty on one.

    **It is not a design space, and the reason is algebraic rather than
    numerical.** Only vertical equilibrium is imposed here. Horizontal
    equilibrium of the axial forces at a node reads
    `q_before (x_before - x) + q_after (x_after - x) = 0`, so on an evenly spaced
    plan it collapses to `q_after = q_before`: **the only force densities that
    leave such a shape funicular are uniform ones.** Every other choice buys its
    fixed plan by handing the horizontal thrust to structure that is not being
    designed, and the shape then carries the design load in bending rather than
    axially.

    So this holds the plan exactly, and a member can never shorten past its own
    projection, but the funicular part of what it can reach is a single
    parameter — the scale of a uniform force density. Use it to hold a plan while
    sweeping that one parameter, not to give an optimizer twenty.
    """
    xyz = jnp.asarray(xyz)

    free = graph.indices_free
    fixed = graph.indices_fixed

    stiffness = graph.connectivity_free.T @ (q[:, None] * graph.connectivity_free)
    coupling = graph.connectivity_free.T @ (q[:, None] * graph.connectivity_fixed)

    applied = jnp.asarray(loads)[free, 2]
    held = coupling @ xyz[fixed, 2]

    heights = jnp.linalg.solve(stiffness, applied - held)

    return xyz.at[free, 2].set(heights)


def _balance_rows(
    xyz: Float[np.ndarray, "nodes 3"],
    edges: Int[np.ndarray, "edges 2"],
    nodes_free: Int[np.ndarray, "nodes_free"],
    axes: tuple[int, ...],
) -> Float[np.ndarray, "equations edges"]:
    """
    Coefficient of every force density in the nodal balance, per axis.

    Parameters
    ----------
    xyz :
        The geometry the balance is written at.
    edges :
        The two node indices spanned by every edge.
    nodes_free :
        Indices of the nodes whose balance is written.
    axes :
        Coordinate axes to write a balance row for, in row-block order.

    Returns
    -------
    balance :
        One row per free node and axis; the residual there is this matrix
        times the densities, minus the applied load.
    """
    num_edges = edges.shape[0]

    incidence = np.zeros((num_edges, xyz.shape[0]))
    incidence[np.arange(num_edges), edges[:, 0]] = 1.0
    incidence[np.arange(num_edges), edges[:, 1]] = -1.0

    blocks = [(incidence.T * (incidence @ xyz[:, axis]))[nodes_free] for axis in axes]

    return np.concatenate(blocks, axis=0)


def _nodes_free(structure: Structure) -> Int[np.ndarray, "nodes_free"]:
    """
    Indices of the unsupported nodes, in ascending order.
    """
    every = np.arange(structure.num_nodes)

    return np.setdiff1d(every, np.asarray(structure.supports))


def plan_equilibrium(structure: Structure) -> Float[np.ndarray, "equations edges"]:
    """
    Horizontal balance of the axial forces at every free node, linear in `q`.

    Parameters
    ----------
    structure :
        The structure whose starting plan is to be held.

    Returns
    -------
    balance :
        One row per free node and horizontal axis. A force density vector in
        its null space keeps the held plan in horizontal equilibrium under
        purely vertical loads.

    Notes
    -----
    The coefficients read only the plan, which a held-plan search never moves,
    so the matrix is a constant of the topology and is built once on the host.
    """
    nodes = np.asarray(structure.nodes)
    edges = np.asarray(structure.edges)

    return _balance_rows(nodes, edges, _nodes_free(structure), (0, 1))


def _mirror_rows(
    structure: Structure,
    nodes_mirrored: Int[np.ndarray, "nodes"],
) -> Float[np.ndarray, "edges edges"]:
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
    edges = np.asarray(structure.edges)
    ordered = np.sort(edges, axis=1)
    reflected = np.sort(nodes_mirrored[edges], axis=1)

    lookup = {tuple(pair): index for index, pair in enumerate(ordered.tolist())}
    targets = []
    for pair in reflected.tolist():
        target = lookup.get(tuple(pair))
        if target is None:
            raise ValueError(f"the mirror maps edge {pair} onto no member")
        targets.append(target)

    rows = np.eye(edges.shape[0])
    rows[np.arange(edges.shape[0]), targets] -= 1.0

    return rows


def density_basis(
    structure: Structure,
    nodes_mirrored: Int[np.ndarray, "nodes"] | None = None,
) -> Float[np.ndarray, "edges independents"]:
    """
    Orthonormal basis of the force densities that hold the starting plan.

    Parameters
    ----------
    structure :
        The structure whose starting plan is to be held.
    nodes_mirrored :
        Mirror image of every node index, or None to ask for no symmetry.
        When given, the basis spans only the densities equal on mirrored
        members, and the search shrinks accordingly.

    Returns
    -------
    basis :
        One orthonormal column per independent edge, spanning the null space
        of `plan_equilibrium` — intersected, when a mirror is given, with the
        densities that mirror onto themselves.

    Notes
    -----
    The width of the basis is the count of independent edges: members minus
    the rank of the horizontal balance. A chain gives one, which is the
    degeneracy `positions_vertical` warns about; a triangulated topology gives
    more, its members accumulating faster than its balance rows. Any `q` in
    the span of this basis, put through `positions_vertical`, yields a shape
    in full equilibrium — horizontal included.

    Symmetry arrives as extra rows rather than as orbit bookkeeping: the
    stacked system asks the density vector to balance the plan and to survive
    the edge permutation the node mirror induces, and its null space is what
    is returned. A mirror that fails to map members onto members raises.
    """
    balance = plan_equilibrium(structure)
    if nodes_mirrored is not None:
        symmetry = _mirror_rows(structure, np.asarray(nodes_mirrored))
        balance = np.concatenate([balance, symmetry], axis=0)

    _, singulars, rows = np.linalg.svd(balance)
    tolerance = singulars.max() * max(balance.shape) * np.finfo(float).eps
    rank = int(np.sum(singulars > tolerance))

    return rows[rank:].T


class PivotedBasis(NamedTuple):
    """
    A held-plan basis whose coordinates are the densities of named members.

    Attributes
    ----------
    basis :
        One column per independent edge, with identity rows on the
        independent members and the transfer on the dependent ones.
    independents :
        Edge indices whose densities are the coordinates, ascending.
    dependents :
        Edge indices the transfer fills in, in the pivot order QR chose.
    """

    basis: Float[np.ndarray, "edges independents"]
    independents: Int[np.ndarray, "independents"]
    dependents: Int[np.ndarray, "dependents"]


def pivoted_basis(
    structure: Structure,
    nodes_mirrored: Int[np.ndarray, "nodes"] | None = None,
) -> PivotedBasis:
    """
    The held-plan subspace in member coordinates, pivoted the TNA way.

    Parameters
    ----------
    structure :
        The structure whose starting plan is to be held.
    nodes_mirrored :
        Mirror image of every node index, or None to ask for no symmetry.
        When given, each coordinate drives its mirrored member too, and the
        count of coordinates shrinks accordingly.

    Returns
    -------
    pivot :
        The basis, and the edges elected independent and dependent.

    Notes
    -----
    The same subspace `density_basis` spans, in different coordinates: each
    one is the density of one actual member, so a bound, a start or a report
    reads member by member. QR with column pivoting elects the
    best-conditioned dependent block — thrust network analysis's
    independent-edges construction — and every dependent density becomes a
    fixed linear function of the independent ones. The columns are not
    orthonormal: legibility is bought at the price of the transfer's
    conditioning, which the pivoting keeps as mild as the topology allows.
    """
    balance = plan_equilibrium(structure)
    if nodes_mirrored is not None:
        symmetry = _mirror_rows(structure, np.asarray(nodes_mirrored))
        balance = np.concatenate([balance, symmetry], axis=0)

    _, triangular, permutation = qr(balance, pivoting=True)
    diagonal = np.abs(np.diag(triangular))
    tolerance = diagonal.max() * max(balance.shape) * np.finfo(float).eps
    rank = int(np.sum(diagonal > tolerance))

    dependents = permutation[:rank]
    independents = np.sort(permutation[rank:])

    held = balance[:, dependents]
    thrown = -balance[:, independents]
    transfer, _, _, _ = np.linalg.lstsq(held, thrown, rcond=None)

    basis = np.zeros((balance.shape[1], independents.size))
    basis[independents, np.arange(independents.size)] = 1.0
    basis[dependents] = transfer

    return PivotedBasis(basis, independents, dependents)


class DensityFit(NamedTuple):
    """
    Force densities that put a drawn geometry in equilibrium with its loads.

    Attributes
    ----------
    q :
        Force density of every edge, from a least-squares fit of the balance.
    self_stresses :
        Orthonormal basis of the density directions that leave the drawn
        geometry balanced, one column per state of self-stress.
    gap :
        Largest balance violation the fit leaves, near zero when the drawn
        geometry is exactly reachable.
    """

    q: Float[np.ndarray, "edges"]
    self_stresses: Float[np.ndarray, "edges stresses"]
    gap: float


def fit_densities(
    structure: Structure,
    xyz: Float[np.ndarray, "nodes 3"],
    loads: Float[np.ndarray, "nodes 3"],
    basis: Float[np.ndarray, "edges independents"] | None = None,
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
        Columns to restrict the fit to, or None to fit every density freely.
        Hand a held-plan basis here and the fit is the nearest funicular
        member of that subspace, plan balance kept exactly by construction.

    Returns
    -------
    fit :
        The fitted densities, the self-stress directions, and the largest
        balance violation left.

    Notes
    -----
    A start generator: sketch the shape wanted, read off the densities that
    make it funicular, and begin a search there. Whether the sketch is exactly
    reachable is reported by the gap rather than assumed. A topology with more
    members than balance rows reaches every sketch, and the surplus returns as
    states of self-stress — directions to trade member signs along without
    moving a node. Under a basis the self-stress columns are combinations of
    its columns, orthonormal only when the basis is.
    """
    nodes = np.asarray(xyz)
    edges = np.asarray(structure.edges)
    nodes_free = _nodes_free(structure)

    balance = _balance_rows(nodes, edges, nodes_free, (0, 1, 2))
    columns = [np.asarray(loads)[nodes_free, axis] for axis in (0, 1, 2)]
    applied = np.concatenate(columns)

    span = np.eye(edges.shape[0]) if basis is None else np.asarray(basis)
    restricted = balance @ span

    coordinates, _, rank, _ = np.linalg.lstsq(restricted, applied, rcond=None)
    _, _, rows = np.linalg.svd(restricted)
    q = span @ coordinates
    self_stresses = span @ rows[rank:].T
    gap = float(np.abs(balance @ q - applied).max())

    return DensityFit(q, self_stresses, gap)


def equilibrium_gap(
    structure: Structure,
    xyz: Float[np.ndarray, "nodes 3"],
    q: Float[np.ndarray, "edges"],
    loads: Float[np.ndarray, "nodes 3"],
) -> float:
    """
    Largest nodal balance violation of a geometry at given force densities.

    Parameters
    ----------
    structure :
        The structure supplying the topology and the supports.
    xyz :
        The geometry the balance is measured at.
    q :
        Force density of every edge.
    loads :
        Force applied at every node.

    Returns
    -------
    gap :
        Largest residual force component over the free nodes and all three
        axes, in the units of the loads.

    Notes
    -----
    The measurement `fit_densities` reports about its own answer, offered for
    a `q` chosen elsewhere — a solved shape, a shifted fit, a hand guess. Host
    arithmetic, for reporting rather than for tracing.
    """
    nodes = np.asarray(xyz)
    edges = np.asarray(structure.edges)
    nodes_free = _nodes_free(structure)

    balance = _balance_rows(nodes, edges, nodes_free, (0, 1, 2))
    columns = [np.asarray(loads)[nodes_free, axis] for axis in (0, 1, 2)]
    applied = np.concatenate(columns)

    return float(np.abs(balance @ np.asarray(q) - applied).max())


class FdmFormFinder(AbstractFormFinder):
    """
    The force density method, as a block of the design pipeline.

    Attributes
    ----------
    xyz_fixed :
        Position of every supported node, which the shape is hung from.
    graph :
        The connectivity the method solves on.

    Notes
    -----
    Building the block builds the connectivity matrices and the free-fixed
    partition, which is everything the method can settle before a force density
    is chosen. Both are read on the host, so a block rebuilt inside a trace is
    what stops the stage being jitted.

    **The structure itself is not kept, only the two things the method reads of
    it.** The graph holds the topology already, so a block that also held the
    structure would carry it twice; what the graph cannot hold is a coordinate,
    and of those only the supported ones survive a solve. The supports do not
    move, so the slice is taken once here rather than per call.

    The block differentiates by tracing its own solve and carries no rule of its
    own, the equilibrium being linear in the coordinates once the force
    densities are fixed.
    """

    xyz_fixed: Float[Array, "supports 3"]
    graph: EquilibriumStructure

    def __init__(self, structure: Structure) -> None:
        """
        Build a form finder on a structure's connectivity.

        Parameters
        ----------
        structure :
            The structure supplying the topology and the supported nodes.
        """
        graph = equilibrium_graph(structure)

        self.xyz_fixed = structure.nodes[graph.indices_fixed]
        self.graph = graph

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
        state = equilibrium_state(q, self.xyz_fixed, self.graph, loads)
        shape = FormFoundShape(state.xyz, state.lengths[:, 0])

        return shape
