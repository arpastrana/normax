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
The Warren truss designed end to end, against two searches without a form finder.

Three constrained searches over the same members, the same load cases, the
same analysis and the same code check, differing only in how — and whether —
the geometry moves. The end-to-end route moves the held-plan subspace
coordinates of experiment 16 together with every member diameter: the form
finder turns the coordinates into a geometry, the frame analysis into member
forces, the EN 1993-1-1 check into utilizations. The free-heights route drops
the form finder and hands the optimizer the height of every free node
directly, driving the same T2 and T3 alone. The sizing-only route holds the
truss as drawn and moves the diameters alone. All three run the same SLSQP
under hard `U <= 1` per member and load case, analytic Jacobians throughout.

**On this truss the two shaped routes span the same geometries.** Experiment
16 counted every held-plan geometry funicular-reachable — sixteen independent
edges against fifteen free heights plus one self-stress — so unlike the arch
of experiment 15, where the heights were a strict superset, here the density
route gives nothing away. Whether the two parametrizations also *land* on the
same design is what the run measures: any gap between them is landscape and
conditioning, never reach.

Four load cases, all on the bottom chord: the uniform deck the shape is
form-found under, the two half-span cases that swap the diagonals between
tension and compression — those three of equal total — and a fraction of that
total concentrated at the midspan deck node.
One diameter per member has to satisfy all four at once, so the envelope is
a KKT condition rather than a reconciliation.

**The truss is once statically indeterminate, and it shows twice.** The
thrust-vs-tie split of the funicular fit is not what the elastic frame
carries: internal forces depend on the stiffness distribution, so the
funicular `q L` and the analyzed axial force disagree where the arch is
determinate and they agree to machine precision. And the frozen-seed
envelope that seeds each search is measurably infeasible once the frame is
re-analyzed at its own sections — the `∂N/∂d` coupling the arch priced at a
tenth of a percent is orders larger here. Both are measured and reported;
the simultaneous formulation holds the coupling inside the gradient, so
neither survives to the answers.

The YAML's `limit_rise` switch puts a lid on how tall either shaped route may
grow: no vertex above `rise_factor` times the drawn depth. The free-heights
route carries the lid as a box bound on its own variables; the end-to-end
route, whose heights are outputs of the form finder, carries it as one
normalized inequality row per free node. The sag stays free either way — it
is a ceiling, not a box — and the sizing-only route never notices it.

The descents restart until quiet: SLSQP is rerun from its own answer until a
round no longer moves, each restart refreshing the quadratic model. The two
shaped routes leave from the same matched start — the signed lens, written
once as basis coordinates and once as heights. The report compares the routes
by shape, by count of variables, by mass across all load cases, and by member
utilization, family by family.

Run with `uv run --group pipeline python experiments/18_warren_optimize.py
[warren_optimize.yaml]`.
"""

import sys
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int
from scipy.optimize import minimize

from normax.analysis.smax import SmaxAnalyzer
from normax.design import Design
from normax.design import StructuralDesignPipeline
from normax.design import design_envelope
from normax.form_finding import FormFoundShape
from normax.form_finding.fdm import FdmFormFinder
from normax.form_finding.fdm import density_basis
from normax.form_finding.fdm import fit_densities
from normax.form_finding.fdm import pivoted_basis
from normax.loads import LoadCases
from normax.loads import assemble_load_cases
from normax.loads import loads_point
from normax.materials import Steel355
from normax.reporting import Report
from normax.reporting import ReportColumn
from normax.reporting import ToleranceCheck
from normax.reporting import checks_passed
from normax.sizing.ec3 import Ec3Sizer
from normax.sizing.ec3 import thinnest_family
from normax.structures import Structure
from normax.structures import build_warren_2d
from normax.structures import member_lengths
from normax.visualization import DescentTrace
from normax.visualization import UtilizationForm
from normax.visualization import figure_mass_descent
from normax.visualization import figure_utilization

CASE_NAMES = (
    "LC1 uniform deck",
    "LC2 half span",
    "LC3 half span mirrored",
    "LC4 midspan point",
)

# Relative steps the central difference sweeps, and the worst scaled error the
# directional derivative may show at its plateau.
GRADIENT_STEPS = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
TOLERANCE_GRADIENT = 1e-6

# Worst constraint violation an answer may show — SLSQP holds its constraints
# to its own ftol, measured orders below this headroom.
TOLERANCE_FEASIBILITY = 1e-6

# How exactly the signed lens densities live in the searched basis, and how
# exactly the full form-finding solve reproduces the drawn lens from them.
TOLERANCE_PROJECTION = 1e-9
TOLERANCE_SHAPE = 1e-8

# A member is counted fully stressed above this envelope utilization, and
# counted at the floor within this distance of the bound.
ACTIVE_UTILIZATION = 0.999
FLOOR_SLACK = 1e-6

FIGURES = Path(__file__).resolve().parent.parent / "figures"

# Both routes compile a gradient and a Jacobian program; the persistent cache
# keeps reruns from paying the compilations again.
COMPILATION_CACHE = Path(__file__).resolve().parent.parent / ".jax_cache"
COMPILATION_CACHE.mkdir(exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(COMPILATION_CACHE))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

# The fixture every pinned tolerance was measured at, so code rather than file.
GRADE = Steel355()
SECTION_CLASS = 3

ROUTE_FORMFOUND = "end to end"
ROUTE_HEIGHTS = "free heights"
ROUTE_DRAWN = "sizing only"
ROUTE_ORDER = (ROUTE_FORMFOUND, ROUTE_HEIGHTS, ROUTE_DRAWN)


class TrussConfig(NamedTuple):
    """
    The truss to build.

    Attributes
    ----------
    num_bays :
        Number of bottom-chord segments the span is divided into. Even, so
        that a bottom-chord node sits exactly at midspan for the point case.
    span :
        Horizontal distance between the two supports.
    depth :
        Height of the top chord above the bottom chord, as drawn.
    """

    num_bays: int
    span: float
    depth: float


class LoadConfig(NamedTuple):
    """
    The load every case carries, however it sits.

    Attributes
    ----------
    total :
        Total downward force of every distributed case.
    half_factor :
        Fraction of the deck load the unloaded half keeps in the asymmetric
        cases, before the case is rescaled back to the shared total.
    point_factor :
        Fraction of the total the midspan point case concentrates. The one
        case exempt from the shared total: a lone wheel is not the whole
        deck, and at the full total it governs nearly every member.
    """

    total: float
    half_factor: float
    point_factor: float


class SketchConfig(NamedTuple):
    """
    The lens the end-to-end route starts from.

    Attributes
    ----------
    sag_lens :
        Depth the sketch hangs its bottom chord to at midspan.
    rise_lens :
        Height the sketch arches its top chord to at midspan.
    """

    sag_lens: float
    rise_lens: float


class SubspaceConfig(NamedTuple):
    """
    Which held-plan basis the geometry variables span.

    Attributes
    ----------
    symmetric :
        Whether the search runs on the mirror-symmetric basis.
    basis :
        Which coordinates span the subspace: `svd` for the orthonormal
        null-space basis, `pivoted` for the member-named independent-edge
        basis QR pivoting elects. The two span the identical subspace, so
        switching prices the coordinates, never the reachable designs.
    margin_fraction :
        Sign margin the starting chords must clear, as a share of their
        median density.
    """

    symmetric: bool
    basis: str
    margin_fraction: float


class AnalysisConfig(NamedTuple):
    """
    What the frame is analyzed with, before either search has spoken.

    Attributes
    ----------
    diameter :
        Outer diameter every member is seeded with.
    """

    diameter: float


class DescentConfig(NamedTuple):
    """
    The budgets both constrained searches share.

    Attributes
    ----------
    iterations :
        Most iterations to spend in each SLSQP round.
    rounds :
        Most restarts, each rerun from the previous round's answer.
    tolerance :
        Convergence tolerance of the constrained solver.
    diameter_floor :
        Smallest diameter any member may take, as a bound rather than a
        constraint, so the fully-stressed condition stays readable off the
        constraint activities alone.
    limit_rise :
        Whether any vertex is kept under the rise ceiling.
    rise_factor :
        The ceiling, as a multiple of the drawn depth. The sag stays free:
        this is a lid on how tall the truss may grow, not a box around it.
    """

    iterations: int
    rounds: int
    tolerance: float
    diameter_floor: float
    limit_rise: bool
    rise_factor: float


class TaskConfig(NamedTuple):
    """
    Everything a run is described by.

    Attributes
    ----------
    structure :
        The truss to build.
    loads :
        The load every case carries.
    sketch :
        The lens the end-to-end route starts from.
    subspace :
        Which held-plan basis the geometry variables span.
    analysis :
        What the frame is seeded with.
    descent :
        The budgets both searches share.
    """

    structure: TrussConfig
    loads: LoadConfig
    sketch: SketchConfig
    subspace: SubspaceConfig
    analysis: AnalysisConfig
    descent: DescentConfig


def parse_config(text: str) -> TaskConfig:
    """
    The truss and the budgets a run is described by.

    Parameters
    ----------
    text :
        Text of the file describing the run.

    Returns
    -------
    config :
        The truss, and the settings its two routes are compared under.

    Raises
    ------
    TypeError
        If the text names a field that does not exist, or omits one that does.

    Notes
    -----
    No container carries a default, so a file missing a field is refused
    rather than quietly completed.
    """
    document = yaml.safe_load(text)

    config = TaskConfig(
        structure=TrussConfig(**document["structure"]),
        loads=LoadConfig(**document["loads"]),
        sketch=SketchConfig(**document["sketch"]),
        subspace=SubspaceConfig(**document["subspace"]),
        analysis=AnalysisConfig(**document["analysis"]),
        descent=DescentConfig(**document["descent"]),
    )
    if config.subspace.basis not in ("svd", "pivoted"):
        raise ValueError(f"basis must be svd or pivoted, got {config.subspace.basis}")

    return config


class WarrenProblem(NamedTuple):
    """
    The prepared truss, its blocks, and the subspace the geometry moves in.

    Attributes
    ----------
    structure :
        The truss the blocks were built against, supplying the drawn geometry
        the sizing-only route holds.
    pipeline :
        The three blocks, each already bound to the truss on the host.
    loads :
        The case the shape answers to, and the cases both routes are checked
        against.
    basis :
        Held-plan density basis the end-to-end geometry coordinates span.
    independents :
        Edge indices the pivoted coordinates read back as, or None when the
        basis is the orthonormal one and coordinates are projections.
    nodes_free :
        Indices of the nodes whose height the free-heights route moves.
    diameters_seed :
        Outer diameter the frame is analyzed at before any search sizes it.
    """

    structure: Structure
    pipeline: StructuralDesignPipeline
    loads: LoadCases
    basis: Float[Array, "edges independents"]
    independents: Int[np.ndarray, "independents"] | None
    nodes_free: Int[Array, "nodes_free"]
    diameters_seed: Float[Array, "edges"]


class StartPoint(NamedTuple):
    """
    Where the end-to-end route leaves from, and how exactly it was matched.

    Attributes
    ----------
    q :
        The signed lens densities, chords carrying their signs.
    xi :
        The same densities as coordinates of the searched basis.
    lens :
        The sketch the densities were fitted to.
    projection :
        How much of the signed densities the basis fails to express.
    """

    q: Float[np.ndarray, "edges"]
    xi: Float[np.ndarray, "independents"]
    lens: Float[np.ndarray, "nodes 3"]
    projection: float


class RouteMaps(NamedTuple):
    """
    The compiled maps a constrained descent calls, over one variable vector.

    Attributes
    ----------
    weigh :
        The mass and its gradient together.
    slack :
        How far under one every member's utilization sits, per load case.
    jacobian :
        The slack's derivative in every variable, by forward mode — the
        variables are the short axis against members times cases.
    """

    weigh: object
    slack: object
    jacobian: object


class RouteAnswer(NamedTuple):
    """
    What one constrained descent arrived at, and the road there.

    Attributes
    ----------
    variables :
        The variable vector the solver stopped on.
    masses :
        Objective at every iterate, the start included, across all rounds.
    iterations :
        Iterations spent over every round.
    converged :
        Whether the last round reported clean convergence.
    """

    variables: Float[np.ndarray, "variables"]
    masses: Float[np.ndarray, "steps"]
    iterations: int
    converged: bool


class RouteRead(NamedTuple):
    """
    One answer read back as a design.

    Attributes
    ----------
    mass :
        Mass of the frame analyzed at its own sections.
    xyz :
        Position of every node of the answer.
    rise :
        Height of the highest node.
    sag :
        Height of the lowest node, negative below the supports.
    diameters :
        Outer diameter of every member.
    utilization :
        Worst utilization of every member over the load cases.
    governing :
        Index of the load case working each member hardest.
    active :
        Count of members whose envelope utilization sits at one.
    floored :
        Count of members resting on the diameter floor.
    mirror :
        How far the diameters depart from their own reflection.
    """

    mass: float
    xyz: Float[np.ndarray, "nodes 3"]
    rise: float
    sag: float
    diameters: Float[np.ndarray, "edges"]
    utilization: Float[np.ndarray, "edges"]
    governing: Int[np.ndarray, "edges"]
    active: int
    floored: int
    mirror: float


def build_load_cases(
    structure: Structure,
    weight: LoadConfig,
    num_bays: int,
) -> LoadCases:
    """
    Four cases of equal total, every one on the bottom chord alone.

    Parameters
    ----------
    structure :
        The truss to load.
    weight :
        The total and the asymmetry factor.
    num_bays :
        Number of bottom-chord segments, locating the interior deck nodes.

    Returns
    -------
    loads :
        The uniform deck the shape answers to, the two half-span cases, and
        a fraction of the total concentrated at the midspan deck node.

    Notes
    -----
    The arch experiments' load family, moved onto the deck: the top chord
    carries nothing directly, matching a bridge whose traffic runs on the
    bottom chord. The three distributed cases are rescaled to the shared
    total so none wins by simply carrying less; the point case carries its
    own fraction of it.
    """
    if num_bays % 2:
        raise ValueError(f"num_bays must be even for a midspan node, got {num_bays}")

    interior = np.arange(1, num_bays)
    along = np.asarray(structure.nodes)[interior, 0]
    middle = 0.5 * float(np.asarray(structure.nodes)[num_bays, 0])

    def deck_case(weights: Float[np.ndarray, "interior"]) -> Float[Array, "nodes 3"]:
        scaled = weights * (weight.total / float(weights.sum()))
        cases = [
            loads_point(structure, float(load), node=int(node))
            for node, load in zip(interior, scaled)
        ]

        return jnp.sum(jnp.stack(cases), axis=0)

    uniform = deck_case(np.ones(interior.size))
    near = deck_case(np.where(along <= middle, 1.0, weight.half_factor))
    far = deck_case(np.where(along >= middle, 1.0, weight.half_factor))
    concentrated = weight.total * weight.point_factor
    point = loads_point(structure, concentrated, node=num_bays // 2)
    cases = [uniform, near, far, point]

    return assemble_load_cases(cases)


def mirrored_nodes(num_bays: int) -> Int[np.ndarray, "nodes"]:
    """
    Mirror image of every node index about midspan, chord by chord.
    """
    bottom = num_bays - np.arange(num_bays + 1)
    top = 2 * num_bays - np.arange(num_bays)

    return np.concatenate([bottom, top])


def mirrored_edges(num_bays: int, structure: Structure) -> Int[np.ndarray, "edges"]:
    """
    Index of every member's mirror image about midspan.
    """
    edges = np.asarray(structure.edges)
    ordered = np.sort(edges, axis=1)
    reflected = np.sort(mirrored_nodes(num_bays)[edges], axis=1)

    lookup = {tuple(pair): index for index, pair in enumerate(ordered.tolist())}
    targets = [lookup[tuple(pair)] for pair in reflected.tolist()]

    return np.asarray(targets)


def warren_problem(config: TaskConfig) -> WarrenProblem:
    """
    The truss, its prepared blocks, and the searched basis.

    Parameters
    ----------
    config :
        The run description.

    Returns
    -------
    problem :
        Everything both routes read, gathered once on the host.
    """
    drawn = config.structure
    structure = build_warren_2d(drawn.num_bays, drawn.span, drawn.depth)
    loads = build_load_cases(structure, config.loads, drawn.num_bays)

    mirror = mirrored_nodes(drawn.num_bays) if config.subspace.symmetric else None
    if config.subspace.basis == "pivoted":
        pivot = pivoted_basis(structure, mirror)
        basis = pivot.basis
        independents = np.asarray(pivot.independents)
    else:
        basis = density_basis(structure, mirror)
        independents = None

    family = thinnest_family(GRADE, SECTION_CLASS)
    blocks = StructuralDesignPipeline(
        FdmFormFinder(structure),
        SmaxAnalyzer(structure, family(config.analysis.diameter)),
        Ec3Sizer(structure, family),
    )

    everyone = np.arange(structure.num_nodes)
    frees = np.setdiff1d(everyone, np.asarray(structure.supports))
    nodes_free = jnp.asarray(frees)

    diameters_seed = jnp.full(structure.num_edges, config.analysis.diameter)

    problem = WarrenProblem(
        structure,
        blocks,
        loads,
        jnp.asarray(basis),
        independents,
        nodes_free,
        diameters_seed,
    )

    return problem


def signed_start(problem: WarrenProblem, config: TaskConfig) -> StartPoint:
    """
    The lens fit of experiment 16, signed and written in the searched basis.

    Parameters
    ----------
    problem :
        The prepared truss.
    config :
        The run description, supplying the sketch and the sign margin.

    Returns
    -------
    start :
        The signed densities, their coordinates, and the projection gap.

    Notes
    -----
    Fitted in the full edge space and then projected, rather than fitted in
    the basis directly: the restricted fit's self-stress direction rides at
    the least-squares rank cutoff and is unreliably detected, while the free
    fit's is orders below it. The projection costs nothing measurable — the
    signed densities hold the plan, so they already live in the basis's span,
    and the gap is reported rather than assumed.
    """
    bays = config.structure.num_bays
    span = config.structure.span

    xyz = np.asarray(problem.structure.nodes).copy()
    shape = 4.0 * (xyz[:, 0] / span) * (1.0 - xyz[:, 0] / span)
    lens = xyz.copy()
    bottom = slice(0, bays + 1)
    top = slice(bays + 1, None)
    lens[bottom, 2] -= config.sketch.sag_lens * shape[bottom]
    lens[top, 2] += config.sketch.rise_lens * shape[top]

    fit = fit_densities(problem.structure, lens, problem.loads.formfinding)
    mode = fit.self_stresses[:, 0]

    signs = np.concatenate([np.ones(bays), -np.ones(bays - 1)])
    chords = np.arange(2 * bays - 1)
    margin_fraction = config.subspace.margin_fraction
    margin = margin_fraction * float(np.median(np.abs(fit.q[:bays])))

    values = signs * fit.q[chords]
    slopes = signs * mode[chords]

    cap = 20.0 * float(np.abs(fit.q).max())
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
    q = fit.q + shift * mode

    basis = np.asarray(problem.basis)
    if problem.independents is None:
        xi = basis.T @ q
    else:
        xi = q[problem.independents]
    rebuilt = basis @ xi
    projection = float(np.linalg.norm(rebuilt - q) / np.linalg.norm(q))

    return StartPoint(q, xi, lens, projection)


def envelope_diameters(
    problem: WarrenProblem,
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
    numbers this experiment exists to print.
    """
    lengths = member_lengths(xyz, problem.structure.edges)
    seed = problem.diameters_seed
    forces = problem.pipeline.analyzer(xyz, seed, problem.loads.analysis)
    sizes = problem.pipeline.sizer(forces, lengths)
    design = Design(FormFoundShape(xyz, lengths), forces, sizes)
    sized = design_envelope(design, None)

    diameters = np.asarray(sized.sizes.sections.diameter)

    return np.maximum(diameters, floor)


def formfound_maps(problem: WarrenProblem, ceiling: float | None) -> RouteMaps:
    """
    The end-to-end route's compiled maps, over coordinates and diameters.

    Parameters
    ----------
    problem :
        The prepared truss.
    ceiling :
        Height no vertex may rise above, or None to leave the rise free.

    Returns
    -------
    maps :
        The mass with its gradient, the slack, and the slack's Jacobian.

    Notes
    -----
    The variable vector is the basis coordinates followed by every diameter,
    so the analysis runs at the search's own geometry and sections: the whole
    `∂N/∂ξ` and `∂N/∂d` feedback rides inside the gradient. Every geometry
    the search can reach holds the plan by construction — the coordinates
    span the null space of the horizontal balance, so no bound on them is a
    bound on funicularity.

    Here a height is an output of the form finder rather than a variable, so
    the rise ceiling enters as one inequality row per free node — normalized
    by the ceiling, so it sits at the utilization rows' scale — where the
    free-heights route can carry the same wall as a plain box bound.
    """
    formfinder = problem.pipeline.formfinder
    analyzer = problem.pipeline.analyzer
    sizer = problem.pipeline.sizer
    family = sizer.family
    width = int(problem.basis.shape[1])

    def weigh(x: Float[Array, "variables"]) -> Float[Array, ""]:
        q = problem.basis @ x[:width]
        shape = formfinder(q, problem.loads.formfinding)
        sections = family(x[width:])
        mass = jnp.sum(sections.area * shape.lengths) * family.material.density

        return mass

    def slack(x: Float[Array, "variables"]) -> Float[Array, "constraints"]:
        q = problem.basis @ x[:width]
        diameters = x[width:]
        shape = formfinder(q, problem.loads.formfinding)
        forces = analyzer(shape.xyz, diameters, problem.loads.analysis)
        used = sizer.compute_utilization(diameters, forces, shape.lengths)
        rows = [1.0 - used.ravel()]
        if ceiling is not None:
            heights = shape.xyz[problem.nodes_free, 2]
            rows.append((ceiling - heights) / ceiling)

        return jnp.concatenate(rows)

    maps = RouteMaps(
        jax.jit(jax.value_and_grad(weigh)),
        jax.jit(slack),
        jax.jit(jax.jacfwd(slack)),
    )

    return maps


def heights_maps(problem: WarrenProblem) -> RouteMaps:
    """
    The free-heights route's compiled maps, over heights and diameters.

    Parameters
    ----------
    problem :
        The prepared truss, supplying the free nodes whose height moves.

    Returns
    -------
    maps :
        The mass with its gradient, the slack, and the slack's Jacobian.

    Notes
    -----
    The pipeline minus its first block: the variable vector is the height of
    every free node followed by every diameter, the geometry is written down
    rather than form-found, and the same T2 and T3 run on it. The plan is
    held by never moving it, so no member can shorten past its own
    projection — but nothing here keeps an iterate funicular, and the
    heights answer to the analysis alone. The rise ceiling, when asked for,
    is the driver's business: heights are variables here, so it arrives as
    a box bound rather than as constraint rows.
    """
    analyzer = problem.pipeline.analyzer
    sizer = problem.pipeline.sizer
    family = sizer.family
    count = int(problem.nodes_free.shape[0])

    def written_shape(heights: Float[Array, "nodes_free"]) -> FormFoundShape:
        xyz = problem.structure.nodes.at[problem.nodes_free, 2].set(heights)
        lengths = member_lengths(xyz, problem.structure.edges)

        return FormFoundShape(xyz, lengths)

    def weigh(x: Float[Array, "variables"]) -> Float[Array, ""]:
        shape = written_shape(x[:count])
        sections = family(x[count:])

        return jnp.sum(sections.area * shape.lengths) * family.material.density

    def slack(x: Float[Array, "variables"]) -> Float[Array, "constraints"]:
        shape = written_shape(x[:count])
        diameters = x[count:]
        forces = analyzer(shape.xyz, diameters, problem.loads.analysis)
        used = sizer.compute_utilization(diameters, forces, shape.lengths)

        return 1.0 - used.ravel()

    maps = RouteMaps(
        jax.jit(jax.value_and_grad(weigh)),
        jax.jit(slack),
        jax.jit(jax.jacfwd(slack)),
    )

    return maps


def drawn_maps(problem: WarrenProblem) -> RouteMaps:
    """
    The sizing-only route's compiled maps, over the diameters alone.

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

    def weigh(diameters: Float[Array, "edges"]) -> Float[Array, ""]:
        sections = family(diameters)

        return jnp.sum(sections.area * lengths) * family.material.density

    def slack(diameters: Float[Array, "edges"]) -> Float[Array, "constraints"]:
        forces = analyzer(xyz, diameters, problem.loads.analysis)
        used = sizer.compute_utilization(diameters, forces, lengths)

        return 1.0 - used.ravel()

    maps = RouteMaps(
        jax.jit(jax.value_and_grad(weigh)),
        jax.jit(slack),
        jax.jit(jax.jacfwd(slack)),
    )

    return maps


def descend_route(
    maps: RouteMaps,
    start: Float[np.ndarray, "variables"],
    boxes: list[tuple[float | None, float | None]],
    budget: DescentConfig,
) -> RouteAnswer:
    """
    SLSQP under hard `U <= 1`, restarted from its own answer until quiet.

    Parameters
    ----------
    maps :
        The route's compiled maps.
    start :
        The variable vector to leave from.
    boxes :
        One bound pair per variable.
    budget :
        Iterations per round, rounds, and the solver tolerance.

    Returns
    -------
    answer :
        The variables, the mass at every iterate, and how the solver ended.

    Notes
    -----
    Each restart hands SLSQP a fresh quadratic model at the previous answer,
    which is what moves it off the slow tail of a single long run; the loop
    stops the first time a round barely moves. The mass trajectory is read
    through the compiled objective at every iterate, one cheap extra
    evaluation against the figure it buys.
    """

    def objective(x):
        value, slope = maps.weigh(jnp.asarray(x))

        return float(value), np.asarray(slope, dtype=np.float64)

    def feasible(x):
        return np.asarray(maps.slack(jnp.asarray(x)), dtype=np.float64)

    def feasible_jacobian(x):
        return np.asarray(maps.jacobian(jnp.asarray(x)), dtype=np.float64)

    masses = [objective(start)[0]]

    def track(x):
        masses.append(objective(x)[0])

    held = {"type": "ineq", "fun": feasible, "jac": feasible_jacobian}
    options = {"maxiter": budget.iterations, "ftol": budget.tolerance}

    x = np.asarray(start, dtype=np.float64)
    spent = 0
    converged = False
    for _ in range(budget.rounds):
        found = minimize(
            objective,
            x,
            jac=True,
            method="SLSQP",
            bounds=boxes,
            constraints=[held],
            callback=track,
            options=options,
        )
        x = np.asarray(found.x)
        spent += int(found.nit)
        converged = found.status == 0
        if found.nit <= 1:
            break

    return RouteAnswer(x, np.asarray(masses), spent, converged)


def read_answer(
    problem: WarrenProblem,
    xyz: Float[Array, "nodes 3"],
    diameters: Float[np.ndarray, "edges"],
    budget: DescentConfig,
) -> RouteRead:
    """
    One answer read back as a design, at its own geometry and sections.

    Parameters
    ----------
    problem :
        The prepared truss.
    xyz :
        The answer's geometry.
    diameters :
        The answer's sections.
    budget :
        The budgets, read for the diameter floor.

    Returns
    -------
    read :
        The mass, the shape extremes, and the utilization member by member.
    """
    family = problem.pipeline.sizer.family
    num_bays = (problem.structure.num_nodes - 1) // 2

    lengths = member_lengths(xyz, problem.structure.edges)
    sized = jnp.asarray(diameters)
    forces = problem.pipeline.analyzer(xyz, sized, problem.loads.analysis)
    used = problem.pipeline.sizer.compute_utilization(sized, forces, lengths)

    sections = family(sized)
    mass = float(jnp.sum(sections.area * lengths) * family.material.density)

    utilization = np.asarray(jnp.max(used, axis=0))
    governing = np.asarray(jnp.argmax(used, axis=0))
    active = int(np.sum(utilization > ACTIVE_UTILIZATION))
    floored = int(np.sum(diameters < budget.diameter_floor + FLOOR_SLACK))

    reflected = diameters[mirrored_edges(num_bays, problem.structure)]
    mirror = float(np.max(np.abs(diameters - reflected)) / np.max(diameters))

    read = RouteRead(
        mass,
        np.asarray(xyz),
        float(jnp.max(jnp.asarray(xyz)[:, 2])),
        float(jnp.min(jnp.asarray(xyz)[:, 2])),
        diameters,
        utilization,
        governing,
        active,
        floored,
        mirror,
    )

    return read


def report_gradient(
    report: Report,
    maps: RouteMaps,
    start: Float[np.ndarray, "variables"],
    label: str,
) -> float:
    """
    One route's gradient against a directional central difference.

    Parameters
    ----------
    report :
        Where the sweep is written.
    maps :
        The route's compiled maps.
    start :
        The point the derivative is taken at.
    label :
        Name of the route, for the heading.

    Returns
    -------
    best :
        The smallest scaled disagreement over the swept steps.

    Notes
    -----
    One seeded random direction rather than the coordinate axes: the probe
    moves the geometry variables and the diameters at once, which is exactly
    the mixture SLSQP steps through — for the end-to-end route the form
    finder and the frame analysis inside the same derivative, for the
    free-heights route the coordinates straight into the analysis.
    """
    generator = np.random.default_rng(2026)
    drawn = generator.normal(size=start.shape[0])
    direction = drawn / np.linalg.norm(drawn)

    point = jnp.asarray(start)
    _, slope = maps.weigh(point)
    exact = float(jnp.sum(slope * jnp.asarray(direction)))
    magnitude = float(np.linalg.norm(start))

    columns = (
        ReportColumn("relative step", ".0e"),
        ReportColumn("central difference", ".9e"),
        ReportColumn("scaled error", ".2e"),
    )
    rows = []
    best = float("inf")
    for relative in GRADIENT_STEPS:
        step = magnitude * relative
        forward, _ = maps.weigh(point + step * jnp.asarray(direction))
        backward, _ = maps.weigh(point - step * jnp.asarray(direction))
        quotient = (float(forward) - float(backward)) / (2.0 * step)
        scaled = abs(exact - quotient) / abs(exact)
        best = min(best, scaled)
        rows.append((relative, quotient, scaled))

    entries = (
        ("exact directional derivative", f"{exact:.9e}"),
        ("best scaled error", f"{best:.2e} ({TOLERANCE_GRADIENT:.0e})"),
    )

    report.write_heading(f"The {label} gradient, checked at the start")
    report.write_table(columns, rows)
    report.write_entries(entries)

    return best


def report_routes(
    report: Report,
    reads: dict[str, RouteRead],
    answers: dict[str, RouteAnswer],
    variables: dict[str, int],
) -> None:
    """
    The routes side by side, by every measure the comparison makes.

    Parameters
    ----------
    report :
        Where the table is written.
    reads :
        Each route's answer read back, keyed by route.
    answers :
        Each route's descent record, keyed by route.
    variables :
        Each route's variable count, keyed by route.
    """
    columns = (
        ReportColumn("route", align="<"),
        ReportColumn("variables"),
        ReportColumn("iterations"),
        ReportColumn("mass [t]", ".6f"),
        ReportColumn("rise [mm]", ".0f"),
        ReportColumn("sag [mm]", ".0f"),
        ReportColumn("max U", ".9f"),
        ReportColumn("fully stressed"),
        ReportColumn("at floor"),
    )
    rows = []
    for route in ROUTE_ORDER:
        read = reads[route]
        rows.append(
            (
                route,
                variables[route],
                answers[route].iterations,
                read.mass,
                read.rise,
                read.sag,
                float(read.utilization.max()),
                read.active,
                read.floored,
            )
        )

    report.write_heading("The routes, side by side")
    report.write_table(columns, rows)


def report_families(
    report: Report,
    problem: WarrenProblem,
    reads: dict[str, RouteRead],
) -> None:
    """
    Sections and utilizations family by family, for every route.

    Parameters
    ----------
    report :
        Where the table is written.
    problem :
        The prepared truss, supplying the family slices.
    reads :
        Each route's answer read back, keyed by route.
    """
    bays = (problem.structure.num_nodes - 1) // 2
    families = (
        ("bottom chord", slice(0, bays)),
        ("top chord", slice(bays, 2 * bays - 1)),
        ("rising diagonals", slice(2 * bays - 1, 3 * bays - 1)),
        ("falling diagonals", slice(3 * bays - 1, 4 * bays - 1)),
    )

    columns = (
        ReportColumn("route", align="<"),
        ReportColumn("family", align="<"),
        ReportColumn("d min [mm]", ".1f"),
        ReportColumn("d max [mm]", ".1f"),
        ReportColumn("U min", ".3f"),
        ReportColumn("U max", ".3f"),
    )
    rows = []
    for route in ROUTE_ORDER:
        read = reads[route]
        for name, members in families:
            rows.append(
                (
                    route,
                    name,
                    float(read.diameters[members].min()),
                    float(read.diameters[members].max()),
                    float(read.utilization[members].min()),
                    float(read.utilization[members].max()),
                )
            )

    report.write_heading("Sections and utilization, family by family")
    report.write_table(columns, rows)


def report_governing(report: Report, reads: dict[str, RouteRead]) -> None:
    """
    How many members each load case governs, per route.

    Parameters
    ----------
    report :
        Where the table is written.
    reads :
        Each route's answer read back, keyed by route.
    """
    columns = [ReportColumn("route", align="<")]
    for name in CASE_NAMES:
        columns.append(ReportColumn(name))

    rows = []
    for route in ROUTE_ORDER:
        counts = np.bincount(reads[route].governing, minlength=len(CASE_NAMES))
        rows.append((route, *[int(count) for count in counts]))

    report.write_heading("Members governed, case by case")
    report.write_table(tuple(columns), rows)


def force_agreement(
    problem: WarrenProblem,
    start: StartPoint,
    xyz: Float[Array, "nodes 3"],
) -> float:
    """
    How far the elastic axial forces sit from the funicular prediction.

    Parameters
    ----------
    problem :
        The prepared truss.
    start :
        The signed lens fit, supplying the funicular densities.
    xyz :
        The lens geometry the frame is analyzed at.

    Returns
    -------
    disagreement :
        Worst `|N - q L|` under the shaping case, scaled by the largest
        funicular force.

    Notes
    -----
    On the determinate arch this number sat at solver precision. The truss is
    once statically indeterminate, so the elastic frame chooses its own
    thrust-vs-tie split and the disagreement is structural rather than
    numerical — the reason T1 hands T2 geometry only, never member forces.
    """
    lengths = member_lengths(xyz, problem.structure.edges)
    seed = problem.diameters_seed
    forces = problem.pipeline.analyzer(xyz, seed, problem.loads.analysis)

    funicular = start.q * np.asarray(lengths)
    elastic = np.asarray(forces.axial_force)[0]
    disagreement = float(np.abs(elastic - funicular).max() / np.abs(funicular).max())

    return disagreement


def write_figures(
    problem: WarrenProblem,
    reads: dict[str, RouteRead],
    answers: dict[str, RouteAnswer],
) -> None:
    """
    The three final designs, and the descents that reached them.

    Parameters
    ----------
    problem :
        The prepared truss, supplying the drawn geometry to outline.
    reads :
        Each route's answer read back, keyed by route.
    answers :
        Each route's descent record, keyed by route.
    """
    FIGURES.mkdir(exist_ok=True)

    forms = []
    for route in ROUTE_ORDER:
        read = reads[route]
        title = f"{route} — {read.mass:.4f} t"
        drawn = UtilizationForm(
            title, read.xyz, read.diameters, read.utilization, read.governing
        )
        forms.append(drawn)
    designs = figure_utilization(
        problem.structure.edges, forms, CASE_NAMES, reference=problem.structure.nodes
    )
    designs.savefig(FIGURES / "18_warren_designs.png", dpi=200, bbox_inches="tight")

    traces = (
        DescentTrace(f"{ROUTE_FORMFOUND} (ξ and d)", answers[ROUTE_FORMFOUND].masses),
        DescentTrace(f"{ROUTE_HEIGHTS} (z and d)", answers[ROUTE_HEIGHTS].masses),
        DescentTrace(f"{ROUTE_DRAWN} (d)", answers[ROUTE_DRAWN].masses),
    )
    descents = figure_mass_descent(traces)
    descents.savefig(FIGURES / "18_warren_descent.png", dpi=200, bbox_inches="tight")


def main(path: Path) -> None:
    """
    Run both routes, write the report, and save the figures.

    Parameters
    ----------
    path :
        The YAML file describing the run.
    """
    report = Report()
    report.write_banner("Warren truss — three routes to a design")

    config = parse_config(path.read_text())
    budget = config.descent
    problem = warren_problem(config)
    width = int(problem.basis.shape[1])
    members = problem.structure.num_edges
    count = int(problem.nodes_free.shape[0])

    start = signed_start(problem, config)
    shape = problem.pipeline.formfinder(jnp.asarray(start.q), problem.loads.formfinding)
    reproduction = float(jnp.max(jnp.abs(shape.xyz - jnp.asarray(start.lens))))
    disagreement = force_agreement(problem, start, shape.xyz)

    if budget.limit_rise:
        ceiling = budget.rise_factor * config.structure.depth
    else:
        ceiling = None

    maps_found = formfound_maps(problem, ceiling)
    maps_heights = heights_maps(problem)
    maps_drawn = drawn_maps(problem)

    floor = budget.diameter_floor
    d_found = envelope_diameters(problem, shape.xyz, floor)
    d_drawn = envelope_diameters(problem, problem.structure.nodes, floor)

    x_found = np.concatenate([start.xi, d_found])
    z_start = np.asarray(start.lens)[np.asarray(problem.nodes_free), 2]
    x_heights = np.concatenate([z_start, d_found])
    opening_found = float(np.min(np.asarray(maps_found.slack(jnp.asarray(x_found)))))
    opening_drawn = float(np.min(np.asarray(maps_drawn.slack(jnp.asarray(d_drawn)))))

    if ceiling is None:
        lidded = "off"
    else:
        lidded = f"{ceiling:.0f} mm, {budget.rise_factor:g}x the drawn depth"

    report.write_heading("The start, and what the indeterminacy does to it")
    entries = (
        (
            "searched basis",
            f"{'symmetric' if config.subspace.symmetric else 'full'}"
            f" {config.subspace.basis}",
        ),
        ("rise ceiling", lidded),
        ("geometry variables, end to end", f"{width}"),
        ("geometry variables, free heights", f"{count}"),
        ("projection gap", f"{start.projection:.2e} of |q|"),
        ("lens reproduction [mm]", f"{reproduction:.2e}"),
        ("elastic vs funicular, LC1", f"{disagreement:.1%} of the largest force"),
        ("seed envelope infeasibility, lens", f"{-opening_found:.1%}"),
        ("seed envelope infeasibility, drawn", f"{-opening_drawn:.1%}"),
    )
    report.write_entries(entries)

    best_found = report_gradient(report, maps_found, x_found, ROUTE_FORMFOUND)
    best_heights = report_gradient(report, maps_heights, x_heights, ROUTE_HEIGHTS)
    best_error = max(best_found, best_heights)

    report.write_heading("Descending the three routes")
    boxes_found = [(None, None)] * width + [(floor, None)] * members
    answer_found = descend_route(maps_found, x_found, boxes_found, budget)
    report.write_line(
        f"{ROUTE_FORMFOUND}: {answer_found.masses[-1]:.6f} t in "
        f"{answer_found.iterations} iterations"
    )

    boxes_heights = [(None, ceiling)] * count + [(floor, None)] * members
    answer_heights = descend_route(maps_heights, x_heights, boxes_heights, budget)
    report.write_line(
        f"{ROUTE_HEIGHTS}: {answer_heights.masses[-1]:.6f} t in "
        f"{answer_heights.iterations} iterations"
    )

    boxes_drawn = [(floor, None)] * members
    answer_drawn = descend_route(maps_drawn, d_drawn, boxes_drawn, budget)
    report.write_line(
        f"{ROUTE_DRAWN}: {answer_drawn.masses[-1]:.6f} t in "
        f"{answer_drawn.iterations} iterations"
    )

    q_final = problem.basis @ jnp.asarray(answer_found.variables[:width])
    shape_final = problem.pipeline.formfinder(q_final, problem.loads.formfinding)

    z_final = jnp.asarray(answer_heights.variables[:count])
    xyz_heights = problem.structure.nodes.at[problem.nodes_free, 2].set(z_final)

    read_found = read_answer(
        problem, shape_final.xyz, answer_found.variables[width:], budget
    )
    read_heights = read_answer(
        problem, xyz_heights, answer_heights.variables[count:], budget
    )
    read_drawn = read_answer(
        problem, problem.structure.nodes, answer_drawn.variables, budget
    )
    reads = {
        ROUTE_FORMFOUND: read_found,
        ROUTE_HEIGHTS: read_heights,
        ROUTE_DRAWN: read_drawn,
    }
    answers = {
        ROUTE_FORMFOUND: answer_found,
        ROUTE_HEIGHTS: answer_heights,
        ROUTE_DRAWN: answer_drawn,
    }
    variables = {
        ROUTE_FORMFOUND: width + members,
        ROUTE_HEIGHTS: count + members,
        ROUTE_DRAWN: members,
    }

    report_routes(report, reads, answers, variables)
    report_families(report, problem, reads)
    report_governing(report, reads)

    saving = 1.0 - read_found.mass / read_drawn.mass
    depth_found = read_found.rise - read_found.sag
    routes_gap = read_heights.mass / read_found.mass - 1.0
    shapes_gap = float(np.abs(read_found.xyz[:, 2] - read_heights.xyz[:, 2]).max())
    entries = (
        ("mass, end to end", f"{read_found.mass:.6f} t"),
        ("mass, free heights", f"{read_heights.mass:.6f} t"),
        ("mass, sizing only", f"{read_drawn.mass:.6f} t"),
        ("the geometry bought", f"{saving:.1%}"),
        ("free heights vs end to end", f"{routes_gap:+.2%}"),
        ("the shaped answers differ by [mm]", f"{shapes_gap:.0f}"),
        (
            "depth at the answer [mm]",
            f"{depth_found:.0f}, drawn at {config.structure.depth:.0f}",
        ),
        ("rise ceiling", lidded),
        ("diameter mirror gap, end to end", f"{read_found.mirror:.2e}"),
        ("diameter mirror gap, free heights", f"{read_heights.mirror:.2e}"),
        ("diameter mirror gap, sizing only", f"{read_drawn.mirror:.2e}"),
    )
    report.write_heading("Summary")
    report.write_entries(entries)

    write_figures(problem, reads, answers)
    report.write_heading(f"figures written to {FIGURES}")

    checks = [
        ToleranceCheck("gradient scaled error", best_error, TOLERANCE_GRADIENT),
        ToleranceCheck("projection gap", start.projection, TOLERANCE_PROJECTION),
        ToleranceCheck("lens reproduction [mm]", reproduction, TOLERANCE_SHAPE),
    ]
    for route in ROUTE_ORDER:
        violation = max(0.0, float(reads[route].utilization.max()) - 1.0)
        checks.append(
            ToleranceCheck(
                f"{route} constraint violation", violation, TOLERANCE_FEASIBILITY
            )
        )
    if ceiling is not None:
        for route in (ROUTE_FORMFOUND, ROUTE_HEIGHTS):
            overrise = max(0.0, (reads[route].rise - ceiling) / ceiling)
            checks.append(
                ToleranceCheck(
                    f"{route} rise violation", overrise, TOLERANCE_FEASIBILITY
                )
            )
    converged = all(answers[route].converged for route in ROUTE_ORDER)
    lighter = read_found.mass < read_drawn.mass
    lighter = lighter and read_heights.mass < read_drawn.mass
    passed = checks_passed(checks) and converged and lighter

    report.write_checks(tuple(checks))
    report.write_verdict(passed)


if __name__ == "__main__":
    jnp.set_printoptions(precision=12)
    described = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    main(described or Path(__file__).with_name("warren_optimize.yaml"))
