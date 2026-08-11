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
Form finding with the force density method, the first stage of the pipeline.

Maps force densities to the geometry that carries the applied loads in pure
tension or compression. The equilibrium is linear in the coordinates once the
force densities are fixed, so `jax-fdm` differentiates it by tracing the solve.

The two functions split along the line that separates a shape from a number.
Connectivity is topology, known before any force density is chosen, and is built
once on the host. Only the force densities enter the traced call.
"""

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
    num_nodes = structure.nodes.shape[0]

    nodes = np.arange(num_nodes, dtype=DTYPE_INT_NP)
    edges = np.asarray(structure.edges, dtype=DTYPE_INT_NP)

    supports = np.zeros(num_nodes, dtype=DTYPE_INT_NP)
    supports[np.asarray(structure.supports)] = 1

    return EquilibriumStructure(nodes, edges, supports)


def equilibrium_state(
    q: Float[Array, "edges"],
    structure: Structure,
    graph: EquilibriumStructure,
) -> EquilibriumState:
    """
    The geometry that carries the loads at a given set of force densities.

    Parameters
    ----------
    q :
        Force density of every edge. Negative in compression.
    structure :
        The structure supplying the support positions and the nodal loads.
    graph :
        The connectivity, from `graph`.

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

    The supported nodes hold the positions they were given, so the starting
    geometry matters only there. Everywhere else it is discarded.
    """
    xyz_fixed = structure.nodes[graph.indices_fixed]
    loads = LoadState(nodes=structure.loads, edges=0.0, faces=0.0)
    params = EquilibriumParametersState(q=q, xyz_fixed=xyz_fixed, loads=loads)

    return EquilibriumModel(tmax=1)(params, graph)


def node_positions(
    q: Float[Array, "edges"],
    structure: Structure,
    graph: EquilibriumStructure,
) -> Float[Array, "nodes 3"]:
    """
    The geometry that carries the loads, as coordinates alone.

    Parameters
    ----------
    q :
        Force density of every edge. Negative in compression.
    structure :
        The structure supplying the support positions and the nodal loads.
    graph :
        The connectivity, from `graph`.

    Returns
    -------
    xyz :
        Position of every node at equilibrium.

    Notes
    -----
    Equilibrium in all three coordinates, so the shape is funicular: every edge
    carries its force along its own axis and the nodal loads balance exactly.
    The plan is a result rather than an input, and it moves when the force
    densities stop being uniform.
    """
    return equilibrium_state(q, structure, graph).xyz


def positions_vertical(
    q: Float[Array, "edges"],
    structure: Structure,
    graph: EquilibriumStructure,
) -> Float[Array, "nodes 3"]:
    """
    The geometry that carries the vertical loads with the plan held fixed.

    Parameters
    ----------
    q :
        Force density of every edge. Negative in compression.
    structure :
        The structure supplying the plan, the support positions and the loads.
    graph :
        The connectivity, from `graph`.

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
    xyz = jnp.asarray(structure.nodes)

    free = graph.indices_free
    fixed = graph.indices_fixed

    stiffness = graph.connectivity_free.T @ (q[:, None] * graph.connectivity_free)
    coupling = graph.connectivity_free.T @ (q[:, None] * graph.connectivity_fixed)

    loads = jnp.asarray(structure.loads)[free, 2]
    held = coupling @ xyz[fixed, 2]

    heights = jnp.linalg.solve(stiffness, loads - held)

    return xyz.at[free, 2].set(heights)
