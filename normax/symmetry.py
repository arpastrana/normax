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
Symmetries of a structure, and what a search may fold or sign by them.

Arrays in, arrays out. Nothing here reads a run config: a mirror is a
node permutation, a folding is the orbits several permutations generate, and a
sign guard is a set of members with the sign each must keep.
"""

from typing import NamedTuple

import numpy as np
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from normax.structures import Structure


class SignGuard(NamedTuple):
    """
    The sign a set of force densities must keep through a descent.

    Attributes
    ----------
    signs :
        Sign each guarded member must carry, positive for tension.
    members :
        Indices of the guarded members.
    margin :
        Density each member must clear beyond zero, in its own sign.
    scale :
        Density the rows are normalized by.

    Notes
    -----
    A density crossing zero switches off its member's chain and the form
    finder's vertical stiffness turns singular, so the guard keeps every trial
    point on the signed sheet where a funicular geometry exists. Linear in the
    densities, so the rows are exact.
    """

    signs: Float[np.ndarray, "guarded"]
    members: Int[np.ndarray, "guarded"]
    margin: float
    scale: float


def build_orbit_matrix(
    mappings: tuple[Int[np.ndarray, "items"], ...],
) -> Float[np.ndarray, "items patterns"]:
    """
    One column per orbit of the group several permutations generate.

    Parameters
    ----------
    mappings :
        The item each permutation carries each item onto, one array per
        generator.

    Returns
    -------
    spread :
        Matrix expanding one value per orbit into a full vector every
        generator leaves unchanged.

    Notes
    -----
    Union-find over every generator at once folds the group they generate, so
    a mirror and a one-spoke rotation together give the whole dihedral group.
    Columns are ordered by their smallest member.
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


def permute_members(
    nodes_permuted: Int[np.ndarray, "nodes"],
    structure: Structure,
) -> Int[np.ndarray, "members"]:
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
    members_permuted :
        The member the permutation carries each member onto.

    Raises
    ------
    ValueError
        If some member's image is not itself a member, so the permutation is
        not a symmetry of the structure.
    """
    edges = np.asarray(structure.edges)
    ordered = np.sort(edges, axis=1)
    moved = np.sort(nodes_permuted[edges], axis=1)

    lookup = {tuple(pair): index for index, pair in enumerate(ordered.tolist())}
    targets = []
    for pair in moved.tolist():
        target = lookup.get(tuple(pair))
        if target is None:
            raise ValueError(f"the permutation maps edge {pair} onto no member")
        targets.append(target)

    return np.asarray(targets)


def fold_values(
    values: Float[np.ndarray, "items"],
    spread: Float[Array, "items patterns"] | None,
) -> Float[np.ndarray, "patterns"]:
    """
    Fold a full vector into one value per orbit, the largest of each.

    Parameters
    ----------
    values :
        One value per item.
    spread :
        The orbit columns, or None to leave the vector as it is.

    Returns
    -------
    folded :
        An envelope, so a folded diameter still covers every member it sizes.
    """
    if spread is None:
        return np.asarray(values)

    columns = np.asarray(spread).T
    folded = [float(np.asarray(values)[column > 0.0].max()) for column in columns]

    return np.asarray(folded)


def unfold_values(
    values: Float[np.ndarray, "patterns"],
    spread: Float[Array, "items patterns"] | None,
) -> Float[np.ndarray, "items"]:
    """
    Expand one value per orbit back into one value per item.

    Parameters
    ----------
    values :
        One value per orbit.
    spread :
        The orbit columns, or None when the values are already full.

    Returns
    -------
    unfolded :
        The full vector, orbit members carrying their shared value.
    """
    if spread is None:
        return np.asarray(values)

    return np.asarray(spread) @ np.asarray(values)


def sketch_lens(
    structure: Structure,
    sag: float,
    rise: float,
) -> Float[np.ndarray, "nodes 3"]:
    """
    A drawn truss with each chord bent into a parabola, the plan held.

    Parameters
    ----------
    structure :
        The truss as drawn, its bottom chord the lowest nodes.
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
    along = xyz[:, 0]
    span = along.max() - along.min()
    fraction = (along - along.min()) / span
    parabola = 4.0 * fraction * (1.0 - fraction)

    bottom = np.isclose(xyz[:, 2], xyz[:, 2].min())
    xyz[bottom, 2] -= sag * parabola[bottom]
    xyz[~bottom, 2] += rise * parabola[~bottom]

    return xyz


def guard_signs(
    q: Float[np.ndarray, "members"],
    signs: Float[np.ndarray, "guarded"],
    members: Int[np.ndarray, "guarded"],
    margin_fraction: float,
) -> SignGuard:
    """
    The sign guard a set of densities scales, at a margin stated as a share.

    Parameters
    ----------
    q :
        The densities the margin is scaled by.
    signs :
        Sign each guarded member must carry, positive for tension.
    members :
        Indices of the guarded members.
    margin_fraction :
        Margin each member must clear, as a share of the guarded members'
        median density.

    Returns
    -------
    guard :
        Signs, members, and the margin and scale read off the densities.
    """
    scale = float(np.median(np.abs(np.asarray(q)[members])))

    return SignGuard(signs, members, margin_fraction * scale, scale)


def shift_densities(
    q: Float[np.ndarray, "members"],
    mode: Float[np.ndarray, "members"],
    guard: SignGuard,
) -> Float[np.ndarray, "members"]:
    """
    Shift densities along a self-stress until every guarded member is signed.

    Parameters
    ----------
    q :
        The fitted densities to shift.
    mode :
        The self-stress direction to shift along.
    guard :
        The signs to reach, and the margin to clear them by.

    Returns
    -------
    shifted :
        The signed densities.

    Raises
    ------
    ValueError
        If no shift along the mode signs every guarded member at once.

    Notes
    -----
    Each guarded member asks its sign of the shift as one linear inequality, so
    the feasible set is an interval; the shift nearest zero is taken, stepped a
    twentieth of the window inside it. Unguarded members stay free.
    """
    values = guard.signs * np.asarray(q)[guard.members]
    slopes = guard.signs * np.asarray(mode)[guard.members]

    cap = 20.0 * float(np.abs(q).max())
    lower, upper = -cap, cap
    for value, slope in zip(values, slopes, strict=True):
        if slope > 1e-12:
            lower = max(lower, (guard.margin - value) / slope)
        elif slope < -1e-12:
            upper = min(upper, (guard.margin - value) / slope)
        elif value < guard.margin:
            raise ValueError("a guarded member ignores the self-stress and misses")
    if lower > upper:
        raise ValueError("no self-stress shift signs every guarded member at once")

    inset = 0.05 * (upper - lower)
    shift = float(np.clip(0.0, lower + inset, upper - inset))

    return np.asarray(q) + shift * np.asarray(mode)


def build_member_spread(
    structure: Structure,
    nodes_permuted: tuple[Int[np.ndarray, "nodes"] | None, ...],
) -> Float[np.ndarray, "members patterns"] | None:
    """
    The orbit columns folding the diameters by several node permutations.

    Parameters
    ----------
    structure :
        The structure the permutations act on.
    nodes_permuted :
        The node each permutation carries each node onto, None entries
        standing for a symmetry the run does not ask for.

    Returns
    -------
    spread :
        One column per orbit of the members, or None when no permutation is
        given and every member is sized on its own.
    """
    offered = [nodes for nodes in nodes_permuted if nodes is not None]
    if not offered:
        return None

    generators = tuple(permute_members(nodes, structure) for nodes in offered)

    return build_orbit_matrix(generators)
