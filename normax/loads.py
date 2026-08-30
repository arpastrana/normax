# SPDX-License-Identifier: Apache-2.0
"""
The load cases a structure is shaped by and checked against.

Every pattern is a function of a structure and one magnitude, reading the
pattern's own geometry — a deck off the lowest free nodes, a tributary area
off a polar plan — from the structure alone, so a case can be named in a file.
"""

from collections.abc import Sequence
from typing import NamedTuple
from typing import TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from jaxtyping import Bool
from jaxtyping import Float

from normax.config import LoadCaseConfig
from normax.structures import Structure

# Any container whose fields take a leading load case axis, at either rank.
LoadCaseAxis = TypeVar("LoadCaseAxis", bound=tuple)


class LoadCases(NamedTuple):
    """
    What a structure is shaped by, and what it is then checked against.

    Attributes
    ----------
    formfinding :
        Force applied at every node in the load case the shape answers to.
    analysis :
        Force applied at every node in every load case the members carry.

    Notes
    -----
    A funicular shape carries exactly one load case axially; every other case
    is a departure from it, and the bending that appears is the reason a frame
    analysis sits between the form finder and the check.
    """

    formfinding: Float[Array, "nodes 3"]
    analysis: Float[Array, "load_cases nodes 3"]


def distribute_loads(
    structure: Structure,
    magnitudes: Float[np.ndarray, "nodes"],
    total: float,
) -> Float[Array, "nodes 3"]:
    """
    Downward forces in a given pattern, zeroed at the supports, carrying a total.

    Parameters
    ----------
    structure :
        The structure supplying the node count and the supported nodes.
    magnitudes :
        Relative size of the downward force at every node.
    total :
        Total downward force the free nodes carry once rescaled.

    Returns
    -------
    loads :
        Force applied at every node.
    """
    pattern = np.asarray(magnitudes, dtype=np.float64).copy()
    pattern[np.asarray(structure.supports)] = 0.0
    scaled = pattern * (total / pattern.sum())

    return jnp.zeros((structure.num_nodes, 3)).at[:, 2].set(-jnp.asarray(scaled))


def mask_free_nodes(structure: Structure) -> Bool[np.ndarray, "nodes"]:
    """
    Whether each node is free to move.

    Parameters
    ----------
    structure :
        The structure supplying the supports.

    Returns
    -------
    free :
        True at every node that is not a support.
    """
    free = np.ones(structure.num_nodes, dtype=bool)
    free[np.asarray(structure.supports)] = False

    return free


def mask_deck_nodes(structure: Structure) -> Bool[np.ndarray, "nodes"]:
    """
    Whether each node sits on the deck, the lowest free nodes of a truss.

    Parameters
    ----------
    structure :
        The truss as drawn.

    Returns
    -------
    deck :
        True at every free node drawn at the lowest free height.

    Raises
    ------
    ValueError
        If no free node sits at the lowest free height, which is a structure
        without a deck.
    """
    heights = np.asarray(structure.nodes)[:, 2]
    free = mask_free_nodes(structure)
    deck = free & np.isclose(heights, heights[free].min())
    if not deck.any():
        raise ValueError("no free node sits at the lowest height: there is no deck")

    return deck


def mask_half_span(structure: Structure, mirrored: bool) -> Bool[np.ndarray, "nodes"]:
    """
    Whether each node sits on the near half of the span, or the far one.

    Parameters
    ----------
    structure :
        The structure, its span measured along the first axis.
    mirrored :
        Whether to pick the far half instead of the near one.

    Returns
    -------
    half :
        True on the picked half, midspan included either way.
    """
    along = np.asarray(structure.nodes)[:, 0]
    middle = 0.5 * (along.min() + along.max())

    return along >= middle if mirrored else along <= middle


def create_load_uniform(structure: Structure, total: float) -> Float[Array, "nodes 3"]:
    """
    A total shared equally over every free node.
    """
    return distribute_loads(structure, np.ones(structure.num_nodes), total)


def create_load_half_span(
    structure: Structure,
    total: float,
    factor: float = 0.0,
    mirrored: bool = False,
) -> Float[Array, "nodes 3"]:
    """
    A total on one half of the span, the other half keeping a fraction.

    Parameters
    ----------
    structure :
        The structure to load.
    total :
        Total downward force the case carries.
    factor :
        Fraction of the nodal load the other half keeps, before rescaling.
    mirrored :
        Whether to load the far half instead of the near one.

    Returns
    -------
    loads :
        Force applied at every node.
    """
    pattern = np.where(mask_half_span(structure, mirrored), 1.0, factor)

    return distribute_loads(structure, pattern, total)


def create_load_deck(structure: Structure, total: float) -> Float[Array, "nodes 3"]:
    """
    A total shared equally over the deck nodes of a truss.
    """
    return distribute_loads(structure, mask_deck_nodes(structure).astype(float), total)


def create_load_deck_half(
    structure: Structure,
    total: float,
    factor: float = 0.0,
    mirrored: bool = False,
) -> Float[Array, "nodes 3"]:
    """
    A total on one half of the deck, the other half keeping a fraction.

    Parameters
    ----------
    structure :
        The truss to load.
    total :
        Total downward force the case carries.
    factor :
        Fraction of the nodal load the other half keeps, before rescaling.
    mirrored :
        Whether to load the far half instead of the near one.

    Returns
    -------
    loads :
        Force applied at every node.
    """
    halved = np.where(mask_half_span(structure, mirrored), 1.0, factor)
    pattern = np.where(mask_deck_nodes(structure), halved, 0.0)

    return distribute_loads(structure, pattern, total)


def create_load_deck_point(
    structure: Structure, total: float
) -> Float[Array, "nodes 3"]:
    """
    A total concentrated on the deck node nearest midspan.
    """
    along = np.asarray(structure.nodes)[:, 0]
    middle = 0.5 * (along.min() + along.max())
    distance = np.where(mask_deck_nodes(structure), np.abs(along - middle), np.inf)
    pattern = np.zeros(structure.num_nodes)
    pattern[int(np.argmin(distance))] = 1.0

    return distribute_loads(structure, pattern, total)


class PolarPlan(NamedTuple):
    """
    A polar grid read off its drawn plan.

    Attributes
    ----------
    ring :
        Index of the ring every node sits on, the apex counting as ring zero
        of radius zero where there is one.
    spoke :
        Index of the spoke every node sits on, zero at the apex.
    radii :
        Plan radius of every ring, ascending.
    num_spokes :
        Number of spokes, read off the outermost ring.
    """

    ring: np.ndarray
    spoke: np.ndarray
    radii: np.ndarray
    num_spokes: int


def read_polar_plan(structure: Structure) -> PolarPlan:
    """
    Read the rings and spokes of a polar grid off its drawn plan.

    Parameters
    ----------
    structure :
        The cap as drawn, its plan centered on the origin.

    Returns
    -------
    plan :
        Which ring and spoke every node sits on.
    """
    plan = np.asarray(structure.nodes)[:, :2]
    rho = np.linalg.norm(plan, axis=1)
    radii = np.unique(np.round(rho, 6))
    ring = np.searchsorted(radii, np.round(rho, 6))

    num_spokes = int(np.sum(ring == ring.max()))
    angle = np.arctan2(plan[:, 1], plan[:, 0])
    spoke = np.rint(angle / (2.0 * np.pi / num_spokes)).astype(int) % num_spokes
    spoke[np.isclose(rho, 0.0)] = 0

    return PolarPlan(ring, spoke, radii, num_spokes)


def compute_tributary_areas(structure: Structure) -> Float[np.ndarray, "nodes"]:
    """
    Plan area every node of a polar cap carries.

    Parameters
    ----------
    structure :
        The cap as drawn.

    Returns
    -------
    areas :
        Plan area of every node.

    Notes
    -----
    Each ring owns the annulus reaching halfway to its neighbors, split evenly
    between its spokes; an apex owns the disc inside the first boundary, and an
    open crown carries nothing, so the first ring then owns only the annulus
    outside itself. The areas sum to the plan exactly, which makes the
    supports' share readable as the pressure's total minus the applied total.
    """
    plan = read_polar_plan(structure)
    radii = plan.radii
    outermost = radii[-1]

    inner = np.concatenate([[radii[0]], 0.5 * (radii[:-1] + radii[1:])])
    outer = np.concatenate([0.5 * (radii[:-1] + radii[1:]), [outermost]])
    if np.isclose(radii[0], 0.0):
        inner[0] = 0.0
    ring_areas = np.pi * (outer**2 - inner**2)

    areas = ring_areas[plan.ring] / plan.num_spokes
    if np.isclose(radii[0], 0.0):
        areas[plan.ring == 0] = ring_areas[0]

    return areas


def create_load_tributary(
    structure: Structure, pressure: float
) -> Float[Array, "nodes 3"]:
    """
    A plan pressure resolved onto the nodes by tributary area.

    Parameters
    ----------
    structure :
        The cap to load.
    pressure :
        Downward force per unit of plan area.

    Returns
    -------
    loads :
        Force applied at every node, the supports' share going to ground.
    """
    areas = compute_tributary_areas(structure)
    carried = float(np.sum(areas[mask_free_nodes(structure)]))

    return distribute_loads(structure, areas, pressure * carried)


def create_load_sector(
    structure: Structure,
    pressure: float,
    center: int,
    spokes: int,
    factor: float,
) -> Float[Array, "nodes 3"]:
    """
    A plan pressure kept whole over one sector and graded outside it.

    Parameters
    ----------
    structure :
        The cap to load.
    pressure :
        Downward force per unit of plan area.
    center :
        Spoke the sector is centered on.
    spokes :
        How many spokes the sector covers, odd so it centers on a spoke.
    factor :
        Fraction of the pressure the plan outside the sector keeps.

    Returns
    -------
    loads :
        Force applied at every node, rescaled to the uniform pressure's total
        so no case wins by carrying less.

    Notes
    -----
    The trusses' half-span construction read onto a disc: a drift grades the
    roof rather than spotlighting a slice of it. A crown node sits on every
    sector's axis and is always inside.
    """
    plan = read_polar_plan(structure)
    reach = spokes // 2
    offset = (plan.spoke - center + reach) % plan.num_spokes
    crown = (plan.ring == 0) & np.isclose(plan.radii[0], 0.0)
    inside = (offset <= 2 * reach) | crown

    areas = compute_tributary_areas(structure)
    carried = float(np.sum(areas[mask_free_nodes(structure)]))
    pattern = np.where(inside, areas, factor * areas)

    return distribute_loads(structure, pattern, pressure * carried)


LOAD_PATTERNS = {
    "uniform": create_load_uniform,
    "half_span": create_load_half_span,
    "deck": create_load_deck,
    "deck_half": create_load_deck_half,
    "deck_point": create_load_deck_point,
    "tributary": create_load_tributary,
    "sector": create_load_sector,
}


def build_load_cases(
    structure: Structure,
    described: Sequence[LoadCaseConfig],
) -> LoadCases:
    """
    The load case a structure is shaped by, and every case it is checked against.

    Parameters
    ----------
    structure :
        The structure to load.
    described :
        The cases to build, the first of which shapes the structure.

    Returns
    -------
    loads :
        The form-finding case and the checked cases.

    Raises
    ------
    ValueError
        If a case names a pattern that does not exist.
    """
    applied = []
    for load_case in described:
        if load_case.name not in LOAD_PATTERNS:
            known = ", ".join(sorted(LOAD_PATTERNS))
            raise ValueError(f"unknown load case {load_case.name!r}, expected {known}")
        pattern = LOAD_PATTERNS[load_case.name]
        applied.append(pattern(structure, load_case.magnitude, **load_case.options))

    return assemble_load_cases(applied)


def number_load_cases(load_cases: tuple[LoadCaseConfig, ...]) -> tuple[str, ...]:
    """
    A short name per load case: `LC1`, `LC2`, and so on in the order given.

    Parameters
    ----------
    load_cases :
        The cases as described.

    Returns
    -------
    numbered :
        One name per case, in order.

    Notes
    -----
    What a figure's axis is labeled with. The pattern and its options are what
    a case *is*, and they do not fit under a bar -- two sector cases differing
    only in their center node come out as one unreadable string repeated -- so
    the drawing carries the number and `label_load_cases` names it in the run's
    report, where there is a line to spend on each.
    """
    return tuple(f"LC{order + 1}" for order in range(len(load_cases)))


def label_load_cases(load_cases: tuple[LoadCaseConfig, ...]) -> tuple[str, ...]:
    """
    A label per load case, the pattern's name and whatever options it took.

    Parameters
    ----------
    load_cases :
        The cases as described.

    Returns
    -------
    labels :
        One label per case, in order.
    """
    labels = []
    for load_case in load_cases:
        options = " ".join(f"{key}={value}" for key, value in load_case.options.items())
        labels.append(f"{load_case.name} {options}".strip())

    return tuple(labels)


def assemble_load_cases(
    load_cases: Sequence[Float[Array, "nodes 3"]],
) -> LoadCases:
    """
    Stack checked cases, the first of them shaping the structure.

    Parameters
    ----------
    load_cases :
        Force applied at every node, one entry per checked load case.

    Returns
    -------
    loads :
        The checked cases stacked along a leading axis, and the first of them
        again as the case the shape answers to.
    """
    return LoadCases(load_cases[0], jnp.stack(list(load_cases)))


def stack_load_cases(per_case: Sequence[LoadCaseAxis]) -> LoadCaseAxis:
    """
    Several load cases of one container, stacked into one container.

    Parameters
    ----------
    per_case :
        The container of every load case, in order.

    Returns
    -------
    stacked :
        The same container, every field carrying a leading load case axis.
    """
    return jax.tree.map(lambda *cases: jnp.stack(cases), *per_case)


def select_load_case(stacked: LoadCaseAxis, load_case: int) -> LoadCaseAxis:
    """
    One load case of a stacked container, without the axis.

    Parameters
    ----------
    stacked :
        A container whose every field carries a leading load case axis.
    load_case :
        Index of the load case to read.

    Returns
    -------
    selected :
        The same container, for that load case alone.
    """
    return jax.tree.map(lambda field: field[load_case], stacked)


def count_load_cases(stacked: LoadCaseAxis) -> int:
    """
    How many load cases a stacked container carries.

    Parameters
    ----------
    stacked :
        A container whose every field carries a leading load case axis.

    Returns
    -------
    count :
        Number of load cases, a static Python integer.
    """
    leaves = jax.tree.leaves(stacked)

    return int(leaves[0].shape[0])
