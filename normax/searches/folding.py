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
A structure's mirror, and the folded coordinates a symmetry buys.
"""

from typing import NamedTuple

import numpy as np
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from normax.searches.config import TaskConfig
from normax.structures import Structure


class MirrorFolding(NamedTuple):
    """
    Pattern matrices folding the mirror into the searched variables.

    Attributes
    ----------
    diameters :
        One column per mirror orbit of members, or None to size every member
        independently.
    heights :
        One column per mirror orbit of free nodes, or None to move every
        height independently.

    Notes
    -----
    The symmetric switch folds the whole problem, not just the density
    basis: a pattern variable is the shared value of its orbit, expanding is
    one matmul, and every search then searches a mirror-symmetric design
    space. With the switch off both matrices are None and the searches run on
    the full variables, untouched.
    """

    diameters: Float[Array, "edges patterns_diameter"] | None
    heights: Float[Array, "nodes_free patterns_height"] | None


class SignShift(NamedTuple):
    """
    A fit shifted along its self-stress until the chords carry their signs.

    Attributes
    ----------
    q :
        The shifted densities, each chord clearing its sign margin.
    window :
        Interval of shifts that sign the chords, after capping.
    shift :
        The shift taken, the feasible one nearest zero, stepped inside.
    """

    q: Float[np.ndarray, "edges"]
    window: tuple[float, float]
    shift: float


class ChordSigns(NamedTuple):
    """
    The sign each chord density must keep, entering the slack as linear rows.

    Attributes
    ----------
    signs :
        Sign each chord member must carry, positive for tension.
    chords :
        Indices of the chord members the signs speak about.
    margin :
        Density each chord must clear beyond zero, in its own sign.
    scale :
        Density the rows are normalized by, putting them at the utilization
        rows' scale.

    Notes
    -----
    A guard for trusses whose held-plan subspace touches degenerate states:
    a chord density crossing zero switches off that chord's chain, the
    vertical stiffness the form finder solves turns singular, and the frame
    analysis is handed a non-finite geometry. The rows are exactly linear in
    the searched coordinates, so the quadratic subproblem holds every trial
    point on the signed sheet of the manifold rather than merely the answer.
    """

    signs: Float[np.ndarray, "chords"]
    chords: Int[np.ndarray, "chords"]
    margin: float
    scale: float


def folding_matrix(
    mirrors: Int[np.ndarray, "items"],
) -> Float[np.ndarray, "items patterns"]:
    """
    One column per mirror orbit, carrying each of its members at one.

    Parameters
    ----------
    mirrors :
        The item the mirror carries each item onto.

    Returns
    -------
    spread :
        Matrix expanding one value per orbit into a full, symmetric vector.
    """
    columns = []
    seen = set()
    for index, partner in enumerate(mirrors.tolist()):
        if index in seen:
            continue
        column = np.zeros(mirrors.size)
        column[index] = 1.0
        column[partner] = 1.0
        columns.append(column)
        seen.add(index)
        seen.add(partner)

    return np.stack(columns, axis=1)


def orbit_matrix(
    mappings: tuple[Int[np.ndarray, "items"], ...],
) -> Float[np.ndarray, "items patterns"]:
    """
    One column per orbit of the group several permutations generate.

    Parameters
    ----------
    mappings :
        The item each permutation carries each item onto, one array per
        generator. A single generator that is an involution reproduces
        `folding_matrix` exactly, column for column.

    Returns
    -------
    spread :
        Matrix expanding one value per orbit into a full vector that every
        generator leaves unchanged.

    Notes
    -----
    Orbits come from union-find over every generator at once, so the group
    they generate is folded rather than each generator separately — a mirror
    and a one-spoke rotation together give the whole dihedral group, not two
    reflections. Columns are ordered by their smallest member, which keeps the
    pattern order stable as generators are added or dropped.

    Folding is a restriction of the search, not a symmetrisation of the
    answer: a pattern variable *is* the shared value of its orbit, so the
    design cannot break the symmetry however unsymmetric the loading is.
    """
    size = int(mappings[0].size)
    parent = list(range(size))

    def root_of(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for mapping in mappings:
        for index, image in enumerate(mapping.tolist()):
            left = root_of(index)
            right = root_of(int(image))
            if left != right:
                parent[max(left, right)] = min(left, right)

    orbits: dict[int, list[int]] = {}
    for index in range(size):
        orbits.setdefault(root_of(index), []).append(index)

    columns = []
    for root in sorted(orbits):
        column = np.zeros(size)
        column[orbits[root]] = 1.0
        columns.append(column)

    return np.stack(columns, axis=1)


def folded_seed(
    values: Float[np.ndarray, "items"],
    spread: Float[Array, "items patterns"] | None,
) -> Float[np.ndarray, "patterns"]:
    """
    Fold a full seed vector into one value per mirror orbit.

    Parameters
    ----------
    values :
        The full seed, one value per item.
    spread :
        The orbit columns, or None to keep the seed as it is.

    Returns
    -------
    seed :
        The largest value of each orbit — an envelope, so a folded diameter
        seed still covers both members it now sizes at once.
    """
    if spread is None:
        return values

    columns = np.asarray(spread).T
    folded = [float(values[column > 0.0].max()) for column in columns]

    return np.asarray(folded)


def unfolded_values(
    values: Float[np.ndarray, "patterns"],
    spread: Float[Array, "items patterns"] | None,
) -> Float[np.ndarray, "items"]:
    """
    Expand pattern values back into one value per item.

    Parameters
    ----------
    values :
        One value per mirror orbit.
    spread :
        The orbit columns, or None when the values are already full.

    Returns
    -------
    expanded :
        The full vector, orbit members carrying their shared value.
    """
    if spread is None:
        return values

    return np.asarray(spread) @ values


def pattern_count(spread: Float[Array, "items patterns"] | None, full: int) -> int:
    """
    How many variables a folded block searches.

    Parameters
    ----------
    spread :
        The orbit columns, or None when the block is not folded.
    full :
        The unfolded count.

    Returns
    -------
    count :
        One per orbit when folded, the full count otherwise.
    """
    if spread is None:
        return full

    return int(spread.shape[1])


def lens_geometry(
    structure: Structure,
    span: float,
    num_bays: int,
    sag: float,
    rise: float,
) -> Float[np.ndarray, "nodes 3"]:
    """
    The drawn truss with each chord bent into a parabola, the plan held.

    Parameters
    ----------
    structure :
        The truss as drawn.
    span :
        Horizontal distance between the two supports.
    num_bays :
        Number of bottom-chord segments, splitting the nodes into chords.
    sag :
        Depth the bottom chord hangs to at midspan.
    rise :
        Height the top chord arches to at midspan, above its drawn line.

    Returns
    -------
    xyz :
        The sketch, every horizontal coordinate as drawn.
    """
    xyz = np.asarray(structure.nodes).copy()
    shape = 4.0 * (xyz[:, 0] / span) * (1.0 - xyz[:, 0] / span)

    bottom = slice(0, num_bays + 1)
    top = slice(num_bays + 1, None)
    xyz[bottom, 2] -= sag * shape[bottom]
    xyz[top, 2] += rise * shape[top]

    return xyz


def mirrored_edges(
    nodes_mirrored: Int[np.ndarray, "nodes"],
    structure: Structure,
) -> Int[np.ndarray, "edges"]:
    """
    Index of every member's mirror image about midspan.

    Parameters
    ----------
    nodes_mirrored :
        The node the mirror carries each node onto.
    structure :
        The truss supplying the members the mirror permutes.

    Returns
    -------
    edges_mirrored :
        The member the mirror carries each member onto.
    """
    return permuted_members(nodes_mirrored, structure)


def permuted_members(
    nodes_permuted: Int[np.ndarray, "nodes"],
    structure: Structure,
) -> Int[np.ndarray, "edges"]:
    """
    Index of every member's image under a permutation of the nodes.

    Parameters
    ----------
    nodes_permuted :
        The node the permutation carries each node onto.
    structure :
        The structure supplying the members the permutation acts on.

    Returns
    -------
    edges_permuted :
        The member the permutation carries each member onto.

    Raises
    ------
    KeyError
        If some member's image is not itself a member, which means the
        permutation is not a symmetry of the structure.

    Notes
    -----
    Members are matched unordered, so a permutation that reverses a member
    still finds it. Nothing here assumes the permutation is an involution: a
    rotation is looked up the same way a reflection is.
    """
    edges = np.asarray(structure.edges)
    ordered = np.sort(edges, axis=1)
    moved = np.sort(nodes_permuted[edges], axis=1)

    lookup = {tuple(pair): index for index, pair in enumerate(ordered.tolist())}
    targets = [lookup[tuple(pair)] for pair in moved.tolist()]

    return np.asarray(targets)


class FoldingMaps(NamedTuple):
    """
    The permutations a run folds each kind of variable by.

    Attributes
    ----------
    nodes_mirrored :
        The node the mirror carries each node onto. Restricts the density
        basis, which is folded by the mirror alone whatever else is.
    nodes_folded :
        The node permutations the free heights are folded by, the mirror
        first.
    members_folded :
        The member permutations the diameters are folded by, the mirror
        first.

    Notes
    -----
    Three entries rather than one because the three kinds of variable need not
    fold by the same group, and on a polar grid they deliberately do not: a
    section may be folded as far as fabrication wants, while folding the
    geometry changes what a comparison between the searches even means.
    """

    nodes_mirrored: Int[np.ndarray, "nodes"]
    nodes_folded: tuple[Int[np.ndarray, "nodes"], ...]
    members_folded: tuple[Int[np.ndarray, "edges"], ...]


def folding_maps(
    profile: "StructureProfile",
    config: TaskConfig,
    structure: Structure,
) -> FoldingMaps:
    """
    Every permutation a run folds its variables by, gathered once.

    Parameters
    ----------
    profile :
        The structural family, read for the mirror and for whichever
        rotations it offers.
    config :
        The run description, which the profile reads to decide which
        rotations are wanted.
    structure :
        The structure the permutations act on.

    Returns
    -------
    folding :
        The mirror, the height permutations and the member permutations. The
        mirror leads both tuples, so a caller wanting it alone reads the first
        entry.
    """
    nodes_mirrored = profile.mirrored_nodes(config)

    heights = [nodes_mirrored]
    if profile.heights_rotated is not None:
        turned = profile.heights_rotated(config)
        if turned is not None:
            heights.append(turned)

    sections = [nodes_mirrored]
    if profile.sections_rotated is not None:
        turned = profile.sections_rotated(config)
        if turned is not None:
            sections.append(turned)

    members = tuple(permuted_members(nodes, structure) for nodes in sections)

    return FoldingMaps(nodes_mirrored, tuple(heights), members)


def signed_shift(
    q: Float[np.ndarray, "edges"],
    mode: Float[np.ndarray, "edges"],
    signs: Float[np.ndarray, "chords"],
    chords: Int[np.ndarray, "chords"],
    margin: float,
) -> SignShift:
    """
    Shift densities along a self-stress until every chord carries its sign.

    Parameters
    ----------
    q :
        The fitted densities to shift.
    mode :
        The self-stress direction to shift along.
    signs :
        Sign each chord member must carry, positive for tension.
    chords :
        Indices of the chord members the signs speak about.
    margin :
        Density each chord must clear beyond zero, in its own sign.

    Returns
    -------
    shifted :
        The signed densities, the feasible window, and the shift taken.

    Notes
    -----
    Each chord member asks its sign of the shift as one linear inequality, so
    the feasible set is an interval and is intersected exactly. Of the
    feasible shifts the one nearest zero is taken, stepped a twentieth of the
    window inside it. Members off the chords are left free on purpose: a
    hanger in one shape is a post in another, and a sign pinned here would
    fight the physics later.
    """
    values = signs * q[chords]
    slopes = signs * mode[chords]

    cap = 20.0 * float(np.abs(q).max())
    lower, upper = -cap, cap
    for value, slope in zip(values, slopes):
        if slope > 1e-12:
            lower = max(lower, (margin - value) / slope)
        elif slope < -1e-12:
            upper = min(upper, (margin - value) / slope)
        elif value < margin:
            raise ValueError("a chord ignores the self-stress and misses its sign")
    if lower > upper:
        raise ValueError("no self-stress shift signs both chords at once")

    inset = 0.05 * (upper - lower)
    shift = float(np.clip(0.0, lower + inset, upper - inset))

    return SignShift(q + shift * mode, (lower, upper), shift)
