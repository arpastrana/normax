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
What each search may move, and the programs that turn it into a design.
"""

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from jaxtyping import Float

from normax.design import Design
from normax.design import design_envelope
from normax.form_finding import FormFoundShape
from normax.optimization import augmented_penalty
from normax.searches.config import DescentConfig
from normax.searches.config import TaskConfig
from normax.searches.folding import ChordSigns
from normax.searches.folding import folded_seed
from normax.searches.folding import pattern_count
from normax.searches.problem import DesignProblem
from normax.searches.problem import SearchMaps
from normax.searches.problem import StartPoint
from normax.searches.settings import REPAIR_PASSES
from normax.searches.settings import SEARCH_DRAWN
from normax.searches.settings import SEARCH_FORMFOUND
from normax.searches.settings import SEARCH_HEIGHTS
from normax.structures import member_lengths


class HeightTruss(NamedTuple):
    """
    The ceiling and the floor the shaped searches keep their heights inside.

    Attributes
    ----------
    ceiling :
        Height no vertex may rise above, or None to leave the rise free.
    floor :
        Height no vertex may hang under, or None to leave the sag free.

    Notes
    -----
    Each limit travels the way a search can carry it: a box bound where the
    heights are variables, one normalized inequality row per free node where
    they are outputs of the form finder. Neither is a box around the truss —
    either side may be off on its own.
    """

    ceiling: float | None
    floor: float | None


def height_scale(limits: HeightTruss) -> float:
    """
    Length the sag rows are normalized by, putting them at the utilization scale.

    Parameters
    ----------
    limits :
        The ceiling and the floor the shape is held between.

    Returns
    -------
    scale :
        The floor's own depth where it has one, and the ceiling otherwise.

    Notes
    -----
    A floor at zero is a real limit — no vertex below the plane of the
    supports — but it is its own distance from zero, so it can normalize
    nothing. The ceiling stands in, being the one other length the run states
    about heights. Where the floor is nonzero this is exactly the depth the
    rows were always divided by, so no truss's descent path moves.
    """
    if limits.floor:
        return abs(limits.floor)
    if limits.ceiling:
        return abs(limits.ceiling)

    return 1.0


def truss_heights(config: TaskConfig) -> HeightTruss:
    """
    The height limits a truss run keeps its vertices inside.

    Parameters
    ----------
    config :
        The run description, read for the switches, the factors and the depth.

    Returns
    -------
    limits :
        The ceiling above and the floor below, None where a side is off.
    """
    return height_truss(config.descent, config.structure.depth)


def shell_heights(config: TaskConfig) -> HeightTruss:
    """
    The height limits a shell run keeps its vertices inside.

    Parameters
    ----------
    config :
        The run description, read for the switches, the factors and the plan
        radius.

    Returns
    -------
    limits :
        The ceiling above and the floor below, None where a side is off.

    Notes
    -----
    **The plan radius is the reference, not the drawn rise.** A shell's height
    limits are the room it has to shelter, which is stated against what it
    spans rather than against how high it happens to have been drawn: the
    radius is half the span, so a `rise_factor` of one is a ceiling at half
    the span and stays that whatever rise the generator is given. A truss
    scales its limits by its drawn depth instead, having no span-like length
    of its own that a height should be read against.

    A `sag_factor` of zero puts the floor on the plane of the supports, which
    is the useful setting here: a shell that dips below its own supports is
    not sheltering anything, and forbidding it outright is cheaper than
    pricing it.
    """
    return height_truss(config.descent, config.structure.radius)


def height_truss(budget: DescentConfig, reference: float) -> HeightTruss:
    """
    The height limits a run keeps its vertices inside, read from the budgets.

    Parameters
    ----------
    budget :
        The budgets, read for the switches and the factors.
    reference :
        The drawn length both limits are multiples of.

    Returns
    -------
    limits :
        The ceiling above and the floor below, None where a side is off.
    """
    depth = reference
    if budget.limit_rise:
        ceiling = budget.rise_factor * depth
    else:
        ceiling = None
    if budget.limit_sag:
        floor = -budget.sag_factor * depth
    else:
        floor = None

    return HeightTruss(ceiling, floor)


def limit_label(limit: float | None, factor: float) -> str:
    """
    One height limit spelled for a report entry.

    Parameters
    ----------
    limit :
        The limit's height, or None when that side is free.
    factor :
        The limit as a multiple of the drawn depth.

    Returns
    -------
    label :
        The limit in millimeters and as its multiple, or `off`.
    """
    if limit is None:
        return "off"

    return f"{limit:.0f} mm, {factor:g}x the drawn depth"


def envelope_diameters(
    problem: DesignProblem,
    xyz: Float[Array, "nodes 3"],
    floor: float,
) -> Float[np.ndarray, "edges"]:
    """
    The frozen-seed envelope sections at one geometry, floored.

    Parameters
    ----------
    problem :
        The prepared truss.
    xyz :
        The geometry to seed a search at.
    floor :
        Smallest diameter any member may take.

    Returns
    -------
    diameters :
        One diameter per member, satisfying every case at the seed forces.

    Notes
    -----
    Frozen-seed on purpose: this is the classical design office move — analyze
    at a guess, size to the forces — and how infeasible it turns out to be
    once the frame is re-analyzed at these very sections is one of the
    numbers the experiments exist to print.
    """
    lengths = member_lengths(xyz, problem.structure.edges)
    seed = problem.diameters_seed
    forces = problem.pipeline.analyzer(xyz, seed, problem.loads.analysis)
    sizes = problem.pipeline.sizer(forces, lengths)
    design = Design(FormFoundShape(xyz, lengths), forces, sizes)
    sized = design_envelope(design, None)

    diameters = np.asarray(sized.sizes.sections.diameter)

    return np.maximum(diameters, floor)


def augmented_search(
    weigh: Callable[[Float[Array, "variables"]], Float[Array, ""]],
    slack: Callable[[Float[Array, "variables"]], Float[Array, "constraints"]],
) -> object:
    """
    A search's mass and rows as one scalar, compiled with its gradient.

    Parameters
    ----------
    weigh :
        The mass at a variable vector, untraced and unjitted.
    slack :
        How far above zero every inequality row sits, untraced and unjitted.

    Returns
    -------
    augmented :
        Value and gradient of the augmented objective, taking the multipliers,
        the penalty and the reference mass beside the variables.

    Notes
    -----
    Built from the two maps before either is compiled, because the whole saving
    is that the aggregation happens inside one traced program: a jitted `slack`
    handed to an outer loop that penalizes it afterwards is differentiated row
    by row again.

    The mass is divided through by a reference so that the penalty and the
    objective share a scale. Without it the penalty parameter would have to be
    quoted in tonnes and would mean something different on every structure.
    """

    def augmented(x, multipliers, penalty, reference):
        mass = weigh(x)
        rows = slack(x)
        penalized = augmented_penalty(rows, multipliers, penalty)

        return mass / reference + penalized

    return jax.jit(jax.value_and_grad(augmented))


def formfound_maps(
    problem: DesignProblem,
    limits: HeightTruss,
    length_floor: float,
    chord_signs: ChordSigns | None,
) -> SearchMaps:
    """
    The end-to-end search's compiled maps, over coordinates and diameters.

    Parameters
    ----------
    problem :
        The prepared truss.
    limits :
        The ceiling and the floor no vertex may leave.
    length_floor :
        Smallest length any member may keep, entering as inequality rows for
        the members whose held plan projection is under it.
    chord_signs :
        Signs the chord densities must keep, or None when the subspace has
        no degenerate states worth guarding.

    Returns
    -------
    maps :
        The mass with its gradient, the slack, the slack's Jacobian, and a
        repair that grows the diameters of a landing that missed feasibility.

    Notes
    -----
    **The repair grows diameters and moves no coordinate.** Only the
    utilization rows answer to a section; the height limits and the chord
    signs are functions of the geometry alone, so a landing that missed one of
    those is beyond repair and stays refused. Each pass grows a folded
    diameter by the square root of the worst utilization over the members it
    serves, which under-grows nothing — resistance rises at least as fast as
    the square of the diameter — and the passes repeat because a fatter member
    is stiffer and draws more force.

    The variable vector is the basis coordinates followed by every diameter,
    so the analysis runs at the search's own geometry and sections: the whole
    `∂N/∂ξ` and `∂N/∂d` feedback rides inside the gradient. Every geometry
    the search can reach holds the plan by construction — the coordinates
    span the null space of the horizontal balance, so no bound on them is a
    bound on funicularity.

    Here a height is an output of the form finder rather than a variable, so
    both height limits enter as one inequality row per free node — normalized
    by the limit, so they sit at the utilization rows' scale — where the
    free-heights search carries the same limits as plain box bounds. The
    chord signs enter the same way, one linear row per chord member, and so
    does the length floor: the signed funicular tends to keep members long
    on its own, but that is a tendency, and the floor makes it a constraint
    on the same members the free-heights search guards.
    """
    formfinder = problem.pipeline.formfinder
    analyzer = problem.pipeline.analyzer
    sizer = problem.pipeline.sizer
    family = sizer.family
    width = int(formfinder.basis.shape[1])
    spread = problem.folding.diameters
    held_cases = problem.loads.analysis[problem.cases_held]

    plan = np.asarray(problem.structure.nodes)[:, :2]
    edges = np.asarray(problem.structure.edges)
    spans_plan = np.linalg.norm(plan[edges[:, 1]] - plan[edges[:, 0]], axis=1)
    collapsible = np.flatnonzero(spans_plan < length_floor)

    def sized_members(x: Float[Array, "variables"]) -> Float[Array, "edges"]:
        if spread is None:
            return x[width:]
        return spread @ x[width:]

    def weigh(x: Float[Array, "variables"]) -> Float[Array, ""]:
        shape = formfinder(x[:width], problem.loads.formfinding)
        sections = family(sized_members(x))
        mass = jnp.sum(sections.area * shape.lengths) * family.material.density

        return mass

    def slack(x: Float[Array, "variables"]) -> Float[Array, "constraints"]:
        diameters = sized_members(x)
        shape = formfinder(x[:width], problem.loads.formfinding)
        forces = analyzer(shape.xyz, diameters, held_cases)
        used = sizer.compute_utilization(diameters, forces, shape.lengths)
        rows = [1.0 - used.ravel()]
        if limits.ceiling is not None:
            heights = shape.xyz[problem.nodes_free, 2]
            rows.append((limits.ceiling - heights) / limits.ceiling)
        if limits.floor is not None:
            heights = shape.xyz[problem.nodes_free, 2]
            rows.append((heights - limits.floor) / height_scale(limits))
        if collapsible.size:
            exposed = shape.lengths[collapsible]
            rows.append((exposed - length_floor) / length_floor)
        if chord_signs is not None:
            q = formfinder.member_densities(x[:width])
            signed = chord_signs.signs * q[chord_signs.chords]
            rows.append((signed - chord_signs.margin) / chord_signs.scale)

        return jnp.concatenate(rows)

    def grown(
        shape: FormFoundShape,
        folded: Float[Array, "patterns"],
    ) -> Float[Array, "patterns"]:
        diameters = spread @ folded if spread is not None else folded
        forces = analyzer(shape.xyz, diameters, held_cases)
        used = sizer.compute_utilization(diameters, forces, shape.lengths)
        worst = jnp.max(used, axis=0)
        if spread is None:
            demanded = worst
        else:
            masked = jnp.where(spread.T > 0.0, worst[None, :], 0.0)
            demanded = jnp.max(masked, axis=1)

        return folded * jnp.sqrt(jnp.maximum(demanded, 1.0))

    def repair(x: Float[Array, "variables"]) -> Float[Array, "variables"]:
        held = jnp.asarray(x)
        coordinates = held[:width]
        folded = held[width:]
        # The coordinates never move, so the shape is found once for them all.
        shape = formfinder(coordinates, problem.loads.formfinding)
        for _ in range(REPAIR_PASSES):
            folded = grown(shape, folded)

        return jnp.concatenate([coordinates, folded])

    maps = SearchMaps(
        jax.jit(jax.value_and_grad(weigh)),
        jax.jit(slack),
        jax.jit(jax.jacfwd(slack)),
        repair,
        augmented_search(weigh, slack),
    )

    return maps


def heights_maps(problem: DesignProblem, length_floor: float) -> SearchMaps:
    """
    The free-heights search's compiled maps, over heights and diameters.

    Parameters
    ----------
    problem :
        The prepared truss, supplying the free nodes whose height moves.
    length_floor :
        Smallest length any member may keep, entering as inequality rows.

    Returns
    -------
    maps :
        The mass with its gradient, the slack, and the slack's Jacobian.

    Notes
    -----
    The pipeline minus its first block: the variable vector is the height of
    every free node followed by every diameter, the geometry is written down
    rather than form-found, and the same T2 and T3 run on it. The plan is
    held by never moving it, so no member can shorten past its own horizontal
    projection — but a member joining nodes of equal plan position, a
    Vierendeel vertical, can still be collapsed by a height crossing, which
    hands the analysis a singular frame. The length floor walls that off,
    and its rows exist only for the members whose held projection is under
    the floor: everywhere else the plan already enforces them, so trusses
    without such members run without the rows, untouched. Nothing here keeps
    an iterate funicular, and the heights answer to the analysis alone. The
    rise ceiling, when asked for, is the driver's business: heights are
    variables here, so it arrives as a box bound rather than as constraint
    rows.
    """
    analyzer = problem.pipeline.analyzer
    sizer = problem.pipeline.sizer
    family = sizer.family
    spread_heights = problem.folding.heights
    spread_diameters = problem.folding.diameters
    count = pattern_count(spread_heights, int(problem.nodes_free.shape[0]))
    held_cases = problem.loads.analysis[problem.cases_held]

    plan = np.asarray(problem.structure.nodes)[:, :2]
    edges = np.asarray(problem.structure.edges)
    spans_plan = np.linalg.norm(plan[edges[:, 1]] - plan[edges[:, 0]], axis=1)
    collapsible = np.flatnonzero(spans_plan < length_floor)

    def free_heights(x: Float[Array, "variables"]) -> Float[Array, "nodes_free"]:
        if spread_heights is None:
            return x[:count]
        return spread_heights @ x[:count]

    def sized_members(x: Float[Array, "variables"]) -> Float[Array, "edges"]:
        if spread_diameters is None:
            return x[count:]
        return spread_diameters @ x[count:]

    def written_shape(heights: Float[Array, "nodes_free"]) -> FormFoundShape:
        xyz = problem.structure.nodes.at[problem.nodes_free, 2].set(heights)
        lengths = member_lengths(xyz, problem.structure.edges)

        return FormFoundShape(xyz, lengths)

    def weigh(x: Float[Array, "variables"]) -> Float[Array, ""]:
        shape = written_shape(free_heights(x))
        sections = family(sized_members(x))

        return jnp.sum(sections.area * shape.lengths) * family.material.density

    def slack(x: Float[Array, "variables"]) -> Float[Array, "constraints"]:
        shape = written_shape(free_heights(x))
        diameters = sized_members(x)
        forces = analyzer(shape.xyz, diameters, held_cases)
        used = sizer.compute_utilization(diameters, forces, shape.lengths)
        rows = [1.0 - used.ravel()]
        if collapsible.size:
            exposed = shape.lengths[collapsible]
            rows.append((exposed - length_floor) / length_floor)

        return jnp.concatenate(rows)

    maps = SearchMaps(
        jax.jit(jax.value_and_grad(weigh)),
        jax.jit(slack),
        jax.jit(jax.jacfwd(slack)),
        None,
        augmented_search(weigh, slack),
    )

    return maps


def drawn_maps(problem: DesignProblem) -> SearchMaps:
    """
    The sizing-only search's compiled maps, over the diameters alone.

    Parameters
    ----------
    problem :
        The prepared truss, supplying the drawn geometry that never moves.

    Returns
    -------
    maps :
        The mass with its gradient, the slack, and the slack's Jacobian.
    """
    analyzer = problem.pipeline.analyzer
    sizer = problem.pipeline.sizer
    family = sizer.family
    xyz = problem.structure.nodes
    lengths = member_lengths(xyz, problem.structure.edges)
    spread = problem.folding.diameters
    held_cases = problem.loads.analysis[problem.cases_held]

    def sized_members(x: Float[Array, "variables"]) -> Float[Array, "edges"]:
        if spread is None:
            return x
        return spread @ x

    def weigh(x: Float[Array, "variables"]) -> Float[Array, ""]:
        sections = family(sized_members(x))

        return jnp.sum(sections.area * lengths) * family.material.density

    def slack(x: Float[Array, "variables"]) -> Float[Array, "constraints"]:
        diameters = sized_members(x)
        forces = analyzer(xyz, diameters, held_cases)
        used = sizer.compute_utilization(diameters, forces, lengths)

        return 1.0 - used.ravel()

    maps = SearchMaps(
        jax.jit(jax.value_and_grad(weigh)),
        jax.jit(slack),
        jax.jit(jax.jacfwd(slack)),
        None,
        augmented_search(weigh, slack),
    )

    return maps


def search_maps(
    problem: DesignProblem,
    limits: HeightTruss,
    length_floor: float,
    chord_signs: ChordSigns | None = None,
) -> dict[str, SearchMaps]:
    """
    Every search's compiled maps, keyed by search.

    Parameters
    ----------
    problem :
        The prepared truss.
    limits :
        The ceiling and the floor no vertex may leave.
    length_floor :
        Smallest length the shaped searches may draw any member at.
    chord_signs :
        Signs the end-to-end chord densities must keep, or None for none.

    Returns
    -------
    maps :
        The three searches' maps, in the shared search names.
    """
    maps = {
        SEARCH_FORMFOUND: formfound_maps(problem, limits, length_floor, chord_signs),
        SEARCH_HEIGHTS: heights_maps(problem, length_floor),
        SEARCH_DRAWN: drawn_maps(problem),
    }

    return maps


def search_starts(
    problem: DesignProblem,
    start: StartPoint,
    shape_xyz: Float[Array, "nodes 3"],
    floor: float,
) -> dict[str, Float[np.ndarray, "variables"]]:
    """
    Every search's starting variable vector, keyed by search.

    Parameters
    ----------
    problem :
        The prepared truss.
    start :
        The signed lens fit both shaped searches leave from.
    shape_xyz :
        The form-found lens geometry, sizing the shaped searches' seed.
    floor :
        Smallest diameter any member may take.

    Returns
    -------
    starts :
        The variable vectors, the two shaped searches matched to one geometry.
    """
    spread_diameters = problem.folding.diameters
    spread_heights = problem.folding.heights

    sized_found = envelope_diameters(problem, shape_xyz, floor)
    sized_drawn = envelope_diameters(problem, problem.structure.nodes, floor)
    d_found = folded_seed(sized_found, spread_diameters)
    d_drawn = folded_seed(sized_drawn, spread_diameters)

    x_found = np.concatenate([start.xi, d_found])
    z_full = np.asarray(start.lens)[np.asarray(problem.nodes_free), 2]
    z_start = folded_seed(z_full, spread_heights)
    x_heights = np.concatenate([z_start, d_found])

    starts = {
        SEARCH_FORMFOUND: x_found,
        SEARCH_HEIGHTS: x_heights,
        SEARCH_DRAWN: d_drawn,
    }

    return starts


def search_boxes(
    problem: DesignProblem,
    floor: float,
    limits: HeightTruss,
) -> dict[str, list[tuple[float | None, float | None]]]:
    """
    Every search's bound pairs, keyed by search.

    Parameters
    ----------
    problem :
        The prepared truss, supplying the variable counts.
    floor :
        Smallest diameter any member may take.
    limits :
        The ceiling and the floor boxing the free-heights variables.

    Returns
    -------
    boxes :
        One bound pair per variable, per search.
    """
    width = int(problem.pipeline.formfinder.basis.shape[1])
    count = pattern_count(problem.folding.heights, int(problem.nodes_free.shape[0]))
    members = pattern_count(problem.folding.diameters, problem.structure.num_edges)

    boxes = {
        SEARCH_FORMFOUND: [(None, None)] * width + [(floor, None)] * members,
        SEARCH_HEIGHTS: [(limits.floor, limits.ceiling)] * count
        + [(floor, None)] * members,
        SEARCH_DRAWN: [(floor, None)] * members,
    }

    return boxes


def search_variables(problem: DesignProblem) -> dict[str, int]:
    """
    Every search's variable count, keyed by search.

    Parameters
    ----------
    problem :
        The prepared truss, supplying the counts.

    Returns
    -------
    variables :
        Geometry variables plus diameters, per search.
    """
    width = int(problem.pipeline.formfinder.basis.shape[1])
    count = pattern_count(problem.folding.heights, int(problem.nodes_free.shape[0]))
    members = pattern_count(problem.folding.diameters, problem.structure.num_edges)

    variables = {
        SEARCH_FORMFOUND: width + members,
        SEARCH_HEIGHTS: count + members,
        SEARCH_DRAWN: members,
    }

    return variables
