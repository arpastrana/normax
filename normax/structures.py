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

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int


@dataclass(frozen=True)
class Structure:
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
    """

    nodes: Float[Array, "nodes 3"]
    edges: Int[Array, "edges 2"]
    supports: Int[Array, "supports"]
    loads: Float[Array, "nodes 3"]


def cable(
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
    nodes, edges, supports = _parabola(num_edges, span, -sag)
    loads = _loads_vertical(nodes.shape[0], supports, load)

    return _structure(nodes, edges, supports, loads)


def arch(
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
    nodes, edges, supports = _parabola(num_edges, span, rise)
    loads = _loads_vertical(nodes.shape[0], supports, load)

    return _structure(nodes, edges, supports, loads)


def gridshell(
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

    return _structure(nodes, edges, supports, loads)


def _parabola(
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


def _structure(
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
