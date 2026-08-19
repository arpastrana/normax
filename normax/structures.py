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
    The topology and the starting geometry of a bar structure.

    Attributes
    ----------
    nodes :
        Starting position of every node, an initial guess for form finding.
    edges :
        The two node indices spanned by every edge.
    supports :
        Indices of the nodes whose position is fixed.

    Notes
    -----
    A pytree, so it crosses a jit boundary as three array leaves rather than as
    one opaque object that would have to be hashed. The geometry is then traced
    and differentiable, and the two index arrays are traced but never indexed
    with, every consumer that needs them concrete — the backends preparing a
    solver — reading them on the host.

    **What the structure does not carry is a load.** A structure is asked to
    survive several load cases and is shaped by one of them, so no single case
    belongs to it; `normax.loads.LoadCases` holds them, and the generators in
    `normax.loads` build them from a structure without attaching them to it.
    """

    nodes: Float[Array, "nodes 3"]
    edges: Int[Array, "edges 2"]
    supports: Int[Array, "supports"]

    @property
    def num_edges(self) -> int:
        """
        Number of edges, which is the number of members once sized.
        """
        return int(self.edges.shape[0])

    @property
    def num_nodes(self) -> int:
        """
        Number of nodes, the supported ones included.
        """
        return int(self.nodes.shape[0])

    def crown_node(self) -> int:
        """
        Index of the highest node of the starting geometry.

        Returns
        -------
        crown :
            Index of the node highest above the ground plane.

        Notes
        -----
        Read from the starting geometry, so it is a static Python integer and
        may index a load case or select a member without being traced. A crown
        of the form-found shape would be a traced value and could do neither.
        """
        return int(jnp.argmax(self.nodes[:, 2]))


def build_arch_2d(
    num_edges: int = 10,
    span: float = 10.0,
    rise: float = 0.0,
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

    Returns
    -------
    structure :
        The arch.
    """
    if num_edges < 1:
        raise ValueError(f"num_edges must be at least 1, got {num_edges}")
    if span <= 0.0:
        raise ValueError(f"span must be positive, got {span}")

    ss = np.linspace(0.0, 1.0, num_edges + 1)
    xs = span * ss
    ys = np.zeros_like(ss)
    zs = 4.0 * rise * ss * (1.0 - ss)
    nodes = np.stack([xs, ys, zs], axis=1)

    starts = np.arange(num_edges)
    edges = np.stack([starts, starts + 1], axis=1)
    supports = np.array([0, num_edges])

    return build_structure(nodes, edges, supports)


def build_warren_2d(
    num_bays: int = 8,
    span: float = 10.0,
    depth: float = 1.0,
) -> Structure:
    """
    A 2D Warren truss spanning between two pinned supports, in the XZ plane.

    The bottom chord runs along the span at height zero; the top chord floats
    one depth above it, offset by half a bay. Edges come in families, in this
    order: the `num_bays` bottom-chord members, the `num_bays - 1` top-chord
    members, the `num_bays` rising diagonals, and the `num_bays` falling ones.

    Parameters
    ----------
    num_bays :
        Number of bottom-chord segments the span is divided into.
    span :
        Horizontal distance between the two supports.
    depth :
        Height of the top chord above the bottom chord.

    Returns
    -------
    structure :
        The truss.

    Notes
    -----
    Both bottom-chord ends are supported, matching the pinned-support policy
    of the pipeline. The family ordering is part of the contract: a consumer
    that constrains the chords by sign slices them off the front.
    """
    if num_bays < 2:
        raise ValueError(f"num_bays must be at least 2, got {num_bays}")
    if span <= 0.0:
        raise ValueError(f"span must be positive, got {span}")
    if depth <= 0.0:
        raise ValueError(f"depth must be positive, got {depth}")

    bay = span / num_bays

    xs_bottom = bay * np.arange(num_bays + 1)
    xs_top = bay / 2.0 + bay * np.arange(num_bays)
    zeros_bottom = np.zeros(num_bays + 1)
    bottom = np.stack([xs_bottom, zeros_bottom, zeros_bottom], axis=1)
    top = np.stack([xs_top, np.zeros(num_bays), np.full(num_bays, depth)], axis=1)
    nodes = np.concatenate([bottom, top], axis=0)

    lower = np.arange(num_bays + 1)
    upper = num_bays + 1 + np.arange(num_bays)

    edges_bottom = np.stack([lower[:-1], lower[1:]], axis=1)
    edges_top = np.stack([upper[:-1], upper[1:]], axis=1)
    edges_rising = np.stack([lower[:-1], upper], axis=1)
    edges_falling = np.stack([upper, lower[1:]], axis=1)
    families = [edges_bottom, edges_top, edges_rising, edges_falling]
    edges = np.concatenate(families, axis=0)

    supports = np.array([0, num_bays])

    return build_structure(nodes, edges, supports)


def build_vierendeel_2d(
    num_bays: int = 8,
    span: float = 10.0,
    depth: float = 1.0,
) -> Structure:
    """
    A 2D Vierendeel truss on four pinned supports, in the XZ plane.

    The bottom chord runs along the span at height zero; the top chord runs
    directly above it, one depth up, and verticals join the two at every
    interior node — no diagonals, which is what makes it a Vierendeel. Edges
    come in families, in this order: the `num_bays` bottom-chord members, the
    `num_bays` top-chord members, and the `num_bays - 1` verticals.

    Parameters
    ----------
    num_bays :
        Number of bottom-chord segments the span is divided into.
    span :
        Horizontal distance between the two supports.
    depth :
        Height of the top chord above the bottom chord.

    Returns
    -------
    structure :
        The truss.

    Notes
    -----
    Both chords spring at supports — four pinned nodes, not two. A floating
    top chord reached only through verticals has its held-plan densities
    forced to zero, since verticals project to nothing in the plan balance.
    There are no end posts: they would join two supports and carry nothing in
    any model here. The family ordering is part of the contract, as in
    `build_warren_2d`.
    """
    if num_bays < 2:
        raise ValueError(f"num_bays must be at least 2, got {num_bays}")
    if span <= 0.0:
        raise ValueError(f"span must be positive, got {span}")
    if depth <= 0.0:
        raise ValueError(f"depth must be positive, got {depth}")

    bay = span / num_bays

    xs = bay * np.arange(num_bays + 1)
    zeros = np.zeros(num_bays + 1)
    bottom = np.stack([xs, zeros, zeros], axis=1)
    top = np.stack([xs, zeros, np.full(num_bays + 1, depth)], axis=1)
    nodes = np.concatenate([bottom, top], axis=0)

    lower = np.arange(num_bays + 1)
    upper = num_bays + 1 + np.arange(num_bays + 1)

    edges_bottom = np.stack([lower[:-1], lower[1:]], axis=1)
    edges_top = np.stack([upper[:-1], upper[1:]], axis=1)
    edges_vertical = np.stack([lower[1:-1], upper[1:-1]], axis=1)
    families = [edges_bottom, edges_top, edges_vertical]
    edges = np.concatenate(families, axis=0)

    supports = np.array([0, num_bays, num_bays + 1, 2 * num_bays + 1])

    return build_structure(nodes, edges, supports)


def build_gridshell_3d(
    num_rings: int = 4,
    num_spokes: int = 12,
    radius: float = 5.0,
    rise: float = 2.0,
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

    return build_structure(nodes, edges, supports)


def member_lengths(
    xyz: Float[Array, "nodes 3"],
    edges: Int[Array, "edges 2"],
) -> Float[Array, "edges"]:
    """
    Length of every member at a given geometry.

    Parameters
    ----------
    xyz :
        Position of every node.
    edges :
        The two node indices spanned by every edge.

    Returns
    -------
    lengths :
        Distance between the two nodes of every edge.

    Notes
    -----
    **Geometry, and no stage's product.** A member length is the distance
    between two nodes; a stage reporting one would be reporting arithmetic the
    way a code check reporting a mass would be. What it needs beyond the
    coordinates is the connectivity, so the block holding a view of that answers
    the question and this is the arithmetic it answers it with.

    A free function rather than a method, because the blocks that call it hold a
    connectivity without holding the structure it came from.

    Differentiable in the coordinates, which is what carries the shape's
    influence on both the mass and the slenderness.
    """
    spans = xyz[edges[:, 1]] - xyz[edges[:, 0]]

    return jnp.linalg.norm(spans, axis=1)


def build_structure(
    nodes: Float[np.ndarray, "nodes 3"],
    edges: Int[np.ndarray, "edges 2"],
    supports: Int[np.ndarray, "supports"],
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

    Returns
    -------
    structure :
        The structure.
    """
    nodes = jnp.asarray(nodes)
    edges = jnp.asarray(edges)
    supports = jnp.asarray(supports)

    return Structure(nodes, edges, supports)
