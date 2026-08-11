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
Generators of the structures the pipeline form-finds and sizes.
"""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int


class Structure(NamedTuple):
    """
    The topology, the starting geometry and the loads of a bar structure.

    Attributes
    ----------
    nodes :
        Starting position of every node, an initial guess for form finding.
    edges :
        The two node indices spanned by every edge.
    supports :
        Indices of the nodes whose position is fixed.
    loads :
        Force applied at every node. Zero at the supports.

    Notes
    -----
    A pytree, so it crosses a jit boundary as four array leaves rather than as
    one opaque object that would have to be hashed. The geometry and the loads
    are then traced and differentiable, and the two index arrays are traced but
    never indexed with, every consumer that needs them concrete — the backends
    preparing a solver — reading them on the host.
    """

    nodes: Float[Array, "nodes 3"]
    edges: Int[Array, "edges 2"]
    supports: Int[Array, "supports"]
    loads: Float[Array, "nodes 3"]


def cable_2d(
    num_edges: int = 10,
    span: float = 10.0,
    sag: float = 0.0,
    load: float = 1.0,
) -> Structure:
    """
    A 2D cable hanging between two pinned supports, in the XZ plane.

    Parameters
    ----------
    num_edges :
        Number of segments the cable is discretized into.
    span :
        Horizontal distance between the two supports.
    sag :
        Depth of the parabola the starting geometry sags along.
    load :
        Magnitude of the downward point load applied at every free node.

    Returns
    -------
    structure :
        The cable.

    Notes
    -----
    A cable carries tension, so the force densities on its edges are positive.
    """
    nodes, edges, supports = _parabola_chain(num_edges, span, -sag)
    loads = _loads_vertical(nodes.shape[0], supports, load)

    return _structure_on_device(nodes, edges, supports, loads)


def arch_2d(
    num_edges: int = 10,
    span: float = 10.0,
    rise: float = 0.0,
    load: float = 1.0,
) -> Structure:
    """
    A 2D funicular arch spanning between two pinned supports, in the XZ plane.

    Parameters
    ----------
    num_edges :
        Number of segments the arch is discretized into.
    span :
        Horizontal distance between the two supports.
    rise :
        Height of the parabola the starting geometry rises along.
    load :
        Magnitude of the downward point load applied at every free node.

    Returns
    -------
    structure :
        The arch.

    Notes
    -----
    An arch carries compression, so the force densities on its edges are
    negative. The topology it shares with a cable of the same span is the
    mirror of that sign, not a different structure.
    """
    nodes, edges, supports = _parabola_chain(num_edges, span, rise)
    loads = _loads_vertical(nodes.shape[0], supports, load)

    return _structure_on_device(nodes, edges, supports, loads)


def gridshell_3d(
    num_rings: int = 4,
    num_spokes: int = 12,
    radius: float = 5.0,
    rise: float = 2.0,
    load: float = 1.0,
) -> Structure:
    """
    A gridshell on a spherical cap, supported along its circular boundary.

    The grid is polar: one node at the apex, then a node per spoke on every
    ring. Radial edges run from the apex outwards, hoop edges close each ring.

    Parameters
    ----------
    num_rings :
        Number of rings between the apex and the boundary, boundary included.
    num_spokes :
        Number of spokes radiating from the apex.
    radius :
        Radius of the circular plan of the cap.
    rise :
        Height of the apex above the plane of the boundary.
    load :
        Magnitude of the downward point load applied at every free node.

    Returns
    -------
    structure :
        The gridshell.

    Notes
    -----
    The cap is a sphere of radius `(radius ** 2 + rise ** 2) / (2 * rise)`,
    which is never smaller than the plan radius, so the cap is well defined
    from a shallow dome up to a hemisphere and beyond.
    """
    if num_rings < 1:
        raise ValueError(f"num_rings must be at least 1, got {num_rings}")
    if num_spokes < 3:
        raise ValueError(f"num_spokes must be at least 3, got {num_spokes}")
    if radius <= 0.0:
        raise ValueError(f"radius must be positive, got {radius}")
    if rise <= 0.0:
        raise ValueError(f"rise must be positive, got {rise}")

    radius_sphere = (radius**2 + rise**2) / (2.0 * rise)

    rhos = radius * np.arange(1, num_rings + 1) / num_rings
    thetas = 2.0 * np.pi * np.arange(num_spokes) / num_spokes

    xs = np.outer(rhos, np.cos(thetas))
    ys = np.outer(rhos, np.sin(thetas))
    zs = np.sqrt(radius_sphere**2 - rhos**2) - (radius_sphere - rise)
    zs = np.repeat(zs[:, None], num_spokes, axis=1)

    apex = np.array([[0.0, 0.0, rise]])
    ring = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1)
    nodes = np.concatenate([apex, ring], axis=0)

    # Node index of spoke k on ring j, with the apex at index 0.
    indices = 1 + np.arange(num_rings * num_spokes).reshape(num_rings, num_spokes)
    starts = np.concatenate([np.zeros((1, num_spokes), dtype=int), indices[:-1]])

    neighbors = np.roll(indices, -1, axis=1)

    edges_radial = np.stack([starts.ravel(), indices.ravel()], axis=1)
    edges_hoop = np.stack([indices.ravel(), neighbors.ravel()], axis=1)
    edges = np.concatenate([edges_radial, edges_hoop], axis=0)

    supports = indices[-1]
    loads = _loads_vertical(nodes.shape[0], supports, load)

    return _structure_on_device(nodes, edges, supports, loads)


def loads_uniform(
    structure: Structure,
    load: float,
) -> Float[Array, "nodes 3"]:
    """
    A downward point load of the same size on every free node.

    Parameters
    ----------
    structure :
        The structure to load.
    load :
        Magnitude of the downward point load.

    Returns
    -------
    loads :
        Force applied at every node.

    Notes
    -----
    The load case a funicular structure is form-found under, so the geometry carries
    it in pure tension or compression and the members see no bending. Every
    other load case is a departure from it, and the bending that appears is what a
    frame analysis exists to report.
    """
    return _nodal_loads(structure, jnp.ones(structure.nodes.shape[0]) * load)


def loads_half_span(
    structure: Structure,
    load: float,
    *,
    axis: int = 0,
    factor: float = 0.0,
) -> Float[Array, "nodes 3"]:
    """
    A downward point load on one half of the span and a fraction on the other.

    Parameters
    ----------
    structure :
        The structure to load.
    load :
        Magnitude of the downward point load on the loaded half.
    axis :
        Index of the global axis the span is measured along.
    factor :
        Fraction of that load carried by the other half.

    Returns
    -------
    loads :
        Force applied at every node.

    Raises
    ------
    ValueError
        If the axis is not 0, 1 or 2.

    Notes
    -----
    The load case that decides an arch. A funicular shape carries its design load
    axially and nothing else, so any redistribution of that load has to be
    carried in bending, and the members most affected are the ones the symmetric
    load case left slenderest.

    The split reads the starting geometry rather than the form-found one, which
    keeps it a property of the structure rather than of a particular set of
    force densities. A node exactly at midspan counts as belonging to the
    loaded half.
    """
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1 or 2, got {axis}")

    along = structure.nodes[:, axis]
    middle = 0.5 * (jnp.min(along) + jnp.max(along))

    return _nodal_loads(structure, jnp.where(along <= middle, load, load * factor))


def loads_point(
    structure: Structure,
    load: float,
    *,
    node: int,
) -> Float[Array, "nodes 3"]:
    """
    A single downward point load at one node.

    Parameters
    ----------
    structure :
        The structure to load.
    load :
        Magnitude of the downward point load.
    node :
        Index of the node carrying it.

    Returns
    -------
    loads :
        Force applied at every node.

    Notes
    -----
    Adds to any other load case, being an array like the rest, so a concentrated
    load on top of a distributed one is a sum and needs no separate generator.
    A load placed on a support is discarded, since the support carries it
    straight to ground.
    """
    magnitudes = jnp.zeros(structure.nodes.shape[0]).at[node].set(load)

    return _nodal_loads(structure, magnitudes)


def crown_node(structure: Structure) -> int:
    """
    Index of the highest node of a structure.

    Parameters
    ----------
    structure :
        The structure to search.

    Returns
    -------
    crown :
        Index of the node highest above the ground plane.

    Notes
    -----
    Read from the starting geometry, so it is a static Python integer and may
    index a load case or select a member without being traced.
    """
    return int(jnp.argmax(structure.nodes[:, 2]))


def _nodal_loads(
    structure: Structure,
    magnitudes: Float[Array, "nodes"],
) -> Float[Array, "nodes 3"]:
    """
    Downward forces of given magnitudes, zeroed at the supports.

    Parameters
    ----------
    structure :
        The structure supplying the node count and the supported nodes.
    magnitudes :
        Size of the downward force at every node.

    Returns
    -------
    loads :
        Force applied at every node.
    """
    vertical = jnp.zeros((structure.nodes.shape[0], 3)).at[:, 2].set(-magnitudes)

    return vertical.at[structure.supports, :].set(0.0)


def _parabola_chain(
    num_edges: int,
    span: float,
    offset: float,
) -> tuple[
    Float[np.ndarray, "nodes 3"],
    Int[np.ndarray, "edges 2"],
    Int[np.ndarray, "supports"],
]:
    """
    A chain of edges along a parabola in the XZ plane, pinned at both ends.

    Parameters
    ----------
    num_edges :
        Number of segments the chain is discretized into.
    span :
        Horizontal distance between the two supports.
    offset :
        Height of the parabola at midspan. Negative offsets sag.

    Returns
    -------
    nodes :
        Position of every node.
    edges :
        The two node indices spanned by every edge.
    supports :
        Indices of the two end nodes.
    """
    if num_edges < 1:
        raise ValueError(f"num_edges must be at least 1, got {num_edges}")
    if span <= 0.0:
        raise ValueError(f"span must be positive, got {span}")

    ss = np.linspace(0.0, 1.0, num_edges + 1)
    xs = span * ss
    ys = np.zeros_like(ss)
    zs = 4.0 * offset * ss * (1.0 - ss)
    nodes = np.stack([xs, ys, zs], axis=1)

    starts = np.arange(num_edges)
    edges = np.stack([starts, starts + 1], axis=1)
    supports = np.array([0, num_edges])

    return nodes, edges, supports


def _loads_vertical(
    num_nodes: int,
    supports: Int[np.ndarray, "supports"],
    load: float,
) -> Float[np.ndarray, "nodes 3"]:
    """
    A downward point load on every node that is not a support.

    Parameters
    ----------
    num_nodes :
        Number of nodes in the structure.
    supports :
        Indices of the nodes whose position is fixed.
    load :
        Magnitude of the downward point load.

    Returns
    -------
    loads :
        Force applied at every node.
    """
    loads = np.zeros((num_nodes, 3))
    loads[:, 2] = -load
    loads[supports, :] = 0.0

    return loads


def _structure_on_device(
    nodes: Float[np.ndarray, "nodes 3"],
    edges: Int[np.ndarray, "edges 2"],
    supports: Int[np.ndarray, "supports"],
    loads: Float[np.ndarray, "nodes 3"],
) -> Structure:
    """
    Move host-side arrays onto the device, once, at the container boundary.

    Parameters
    ----------
    nodes :
        Position of every node.
    edges :
        The two node indices spanned by every edge.
    supports :
        Indices of the nodes whose position is fixed.
    loads :
        Force applied at every node.

    Returns
    -------
    structure :
        The structure.
    """
    return Structure(
        nodes=jnp.asarray(nodes),
        edges=jnp.asarray(edges),
        supports=jnp.asarray(supports),
        loads=jnp.asarray(loads),
    )
