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
A form finder as a shape generator, against node coordinates searched directly.

Two parametrizations of the same design space, feeding the same analysis, the
same code check and the same descent. The direct route hands the optimizer the
height of every free node — nineteen variables — and holds the plan by simply
never moving it. The physics-informed route hands it one variable, a uniform
force density, and lets the form finder turn it into a geometry; on an evenly
spaced plan a uniform density is the only choice that keeps the projection
fixed while staying funicular, so the fixed plan is maintained by equilibrium
rather than by decree. One variable against nineteen is not an accident: the
funicular subspace of a held plan is exactly one parameter wide on a chain,
so the single density is that whole subspace and nothing less.

Six things are reported.

    starts      the matched starting designs, one geometry entering two routes
    gradient    the direct route's gradient against central differences
    descents    both routes from every start, and what each arrived at
    quality     the bending-to-axial ratio every iterate passed through
    coupling    the same descents with the analysis re-sectioned between rounds
    variables   the density and the diameters as one constrained search

**The two routes bracket a trade every reparametrization makes.** The height
space contains the funicular family as a curve, so the space's optimum can
only be equal or better — but whether a local descent finds it is the actual
question, and it is answered start by start rather than argued. What the
prior buys is measured alongside: every iterate of the density route is
funicular under the shaping case by construction, while the height route is
free to wander through bending-dominated shapes on its way, and does.

**Both routes carry their length floor by construction.** The plan never
moves, so no member can shorten past its own projection and the collapse
mode of experiment 03 does not exist here — neither route needs the penalty,
and the objective is the enveloped mass alone.

**The height box excludes hanging shapes on purpose.** Below zero the members
turn to tension, buckling disappears from the check, and the search leaves for
a different structure — a cable, not an arch. The density route is held in
compression by its own bounds, so the height route is held above the ground
plane to compare like with like.

**Two more starts have no matched pair on purpose.** A flat line and random
heights lie off the funicular manifold, where no force density can start, so
they are descended by the height route alone — the no-prior scenario, begun
from no prior. The flat start carries the shaping case in pure bending, a
straight beam's axial force being identically zero, which is as far from
funicular as the box allows.

**The coupling is closed by rounds, never inside the objective.** Every
descent above analyzes at the seed sections for its whole search: a line
search compares values of one function, and a seed that moved inside it
would hand the search a different function at every trial.
`normax.design.optimize_staggered` is the driver that closes the loop — a
bounded descent at held sections, the sections settled to what the check
demanded where it stopped, and the descent rerun there until settling no
longer moves them. Both routes take it from every start, and what the
frozen seed cost each answer is a printed column rather than a caveat.

**The third formulation makes the diameters variables outright.** The mass
is strictly increasing in every diameter, so an unconstrained descent would
only shrink them: the standard has to enter as constraints rather than as a
solver. SLSQP moves the one density and every diameter together under
`U <= 1` per member and load case — analytic Jacobians throughout, and the
`∂N/∂d` feedback inside the gradient rather than closed between rounds. No
sizer solves and no envelope reconciles: one diameter per member satisfies
every case at once, and utilization at the answer is a KKT condition rather
than a construction.

The run is described by `parametrization.yaml` beside this file — the arch,
the load, and every search budget. The tolerances stay here in the script,
being the experiment's assertions rather than its settings.

Run with `uv run --group pipeline python experiments/15_shape_parametrization.py
[parametrization.yaml]`.
"""

import sys
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import equinox as eqx
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
from normax.design import DesignParameters
from normax.design import StructuralDesignPipeline
from normax.design import compute_mass
from normax.design import design_envelope
from normax.design import governing_load_case
from normax.design import optimize_staggered
from normax.form_finding import FormFoundShape
from normax.form_finding.fdm import FdmFormFinder
from normax.loads import LoadCases
from normax.loads import assemble_load_cases
from normax.loads import loads_half_span
from normax.loads import loads_uniform
from normax.materials import Steel355
from normax.optimization import SearchResult
from normax.optimization import Trajectory
from normax.optimization import annealing_schedule
from normax.optimization import optimize_annealed
from normax.reporting import Report
from normax.reporting import ReportColumn
from normax.reporting import ToleranceCheck
from normax.reporting import checks_passed
from normax.sizing.ec3 import Ec3Sizer
from normax.sizing.ec3 import thinnest_family
from normax.structures import Structure
from normax.structures import build_arch_2d
from normax.structures import member_lengths
from normax.visualization import Form
from normax.visualization import RouteTrace
from normax.visualization import StartSpread
from normax.visualization import figure_load_cases
from normax.visualization import figure_parametrization

# The arch and every search budget, unless another file is named on the
# command line. Units are millimeters and newtons.
CONFIG = Path(__file__).with_name("parametrization.yaml")

CASE_NAMES = (
    "LC1 uniform",
    "LC2 half span",
    "LC3 half span mirrored",
)

# Relative steps the central difference sweeps, and the worst scaled error the
# directional derivative may show at its plateau.
GRADIENT_STEPS = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
TOLERANCE_GRADIENT = 1e-6

# Ceiling on the bending-to-axial ratio of every density-route iterate, which
# is the funicular claim: the route cannot leave the axial-dominant manifold.
# Not zero, because the frame deforms elastically at the seed stiffness before
# carrying anything, and that bending grows as the arch shallows: measured at
# 2.4e-3 along the matched descent and 6.1e-3 along the shallow-start one.
TOLERANCE_BENDING = 1e-2

# Largest relative spread of the density route's answers over the starts: a
# one-variable search should land on the same design from anywhere, within
# the optimizer's own convergence slack — measured at 1.8e-4 over the starts.
TOLERANCE_SPREAD = 1e-3

# Worst deviation of any per-case utilization from exactly one, the invariant
# the fully-stressed sizing map exists to hold.
TOLERANCE_UTILIZATION = 1e-9

# Worst constraint violation the simultaneous answers may show — SLSQP holds
# its constraints to its own ftol, measured orders below this headroom.
TOLERANCE_FEASIBILITY = 1e-6

FIGURES = Path(__file__).resolve().parent.parent / "figures"

# Every staggered run compiles its own gradient program, so the persistent
# cache is what keeps eight of them from paying eight compilations.
COMPILATION_CACHE = Path(__file__).resolve().parent.parent / ".jax_cache"
COMPILATION_CACHE.mkdir(exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(COMPILATION_CACHE))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

# The fixture every pinned tolerance was measured at, so code rather than file.
GRADE = Steel355()
SECTION_CLASS = 3

# The route names, keys wherever a frozen answer meets its staggered one.
ROUTE_DENSITY = "one density"
ROUTE_HEIGHTS = "free heights"

# The reads the reports make, compiled once each.
governing_compiled = eqx.filter_jit(governing_load_case)


class ArchConfig(NamedTuple):
    """
    The arch to build.

    Attributes
    ----------
    num_edges :
        Number of members the arch is discretized into.
    span :
        Horizontal distance between the two supports.
    rise :
        Height of the parabola the starting geometry rises along.
    """

    num_edges: int
    span: float
    rise: float


class LoadConfig(NamedTuple):
    """
    The load every case carries, however it sits.

    Attributes
    ----------
    total :
        Total downward force of every case.
    half_factor :
        Fraction of the spread the unloaded half keeps in the asymmetric
        cases, before the case is rescaled back to the shared total.
    """

    total: float
    half_factor: float


class AnalysisConfig(NamedTuple):
    """
    What the frame is analyzed with, before the check has spoken.

    Attributes
    ----------
    diameter :
        Outer diameter every member is analyzed at.
    """

    diameter: float


class SearchConfig(NamedTuple):
    """
    The budgets every bounded descent shares.

    Attributes
    ----------
    beta_start :
        Sharpness of the first annealing round.
    beta_stop :
        Sharpness of the last round, and the one the staggered rounds hold.
    rounds :
        Number of annealing rounds.
    iterations :
        Most iterations to spend in each round.
    density_decades :
        How far the force density may move either side of the funicular
        value, keeping it away from zero, where the system is singular.
    """

    beta_start: float
    beta_stop: float
    rounds: int
    iterations: int
    density_decades: float


class StartConfig(NamedTuple):
    """
    Where the descents leave from.

    Attributes
    ----------
    rise_fractions :
        Multiples of the funicular rise the matched starts are taken at.
    random_seed :
        What makes the random start the same run after run.
    ceiling_fraction :
        Multiple of the rise the random heights are drawn under.
    """

    rise_fractions: tuple[float, ...]
    random_seed: int
    ceiling_fraction: float


class SimultaneousConfig(NamedTuple):
    """
    The budgets of the constrained search over density and diameters.

    Attributes
    ----------
    iterations :
        Most iterations to spend.
    tolerance :
        Convergence tolerance of the constrained solver.
    diameter_floor :
        Smallest diameter any member may take, as a bound rather than a
        constraint — a bound never needs a multiplier, so the fully-stressed
        condition stays readable off the constraint activities alone.
    """

    iterations: int
    tolerance: float
    diameter_floor: float


class TaskConfig(NamedTuple):
    """
    Everything a run is described by.

    Attributes
    ----------
    structure :
        The arch to build.
    loads :
        The load every case carries.
    analysis :
        What the frame is analyzed with.
    search :
        The budgets every bounded descent shares.
    starts :
        Where the descents leave from.
    simultaneous :
        The budgets of the constrained search.
    """

    structure: ArchConfig
    loads: LoadConfig
    analysis: AnalysisConfig
    search: SearchConfig
    starts: StartConfig
    simultaneous: SimultaneousConfig


def parse_config(text: str) -> TaskConfig:
    """
    The arch and the budgets a run is described by.

    Parameters
    ----------
    text :
        Text of the file describing the run.

    Returns
    -------
    config :
        The arch, and the settings its parametrizations are compared under.

    Raises
    ------
    TypeError
        If the text names a field that does not exist, or omits one that does.

    Notes
    -----
    No container carries a default, so a file missing a field is refused
    rather than quietly completed. The file is the description of the run,
    and half a description is not one.
    """
    document = yaml.safe_load(text)
    drawn = dict(document["starts"])
    fractions = tuple(drawn.pop("rise_fractions"))
    starts = StartConfig(rise_fractions=fractions, **drawn)

    config = TaskConfig(
        structure=ArchConfig(**document["structure"]),
        loads=LoadConfig(**document["loads"]),
        analysis=AnalysisConfig(**document["analysis"]),
        search=SearchConfig(**document["search"]),
        starts=starts,
        simultaneous=SimultaneousConfig(**document["simultaneous"]),
    )

    return config


class ArchProblem(NamedTuple):
    """
    The prepared arch, the cases it answers to, and the funicular force density.

    Attributes
    ----------
    structure :
        The arch the blocks were built against.
    pipeline :
        The three blocks, each already bound to the arch on the host.
    loads :
        The case the shape answers to, and the cases it is checked against.
    q :
        Force densities that reach the target rise under the funicular case.
    nodes_free :
        Indices of the nodes whose height the direct route moves.
    diameters_seed :
        Outer diameter the frame is analyzed at until a settle says otherwise.
    bounds :
        The box the force density may move in.
    heights_box :
        The box the heights may move in — floored at the ground plane, which
        excludes hanging tension shapes, a different structure rather than a
        worse arch.
    """

    structure: Structure
    pipeline: StructuralDesignPipeline
    loads: LoadCases
    q: Float[Array, "edges"]
    nodes_free: Int[Array, "nodes_free"]
    diameters_seed: Float[Array, "edges"]
    bounds: tuple[float, float]
    heights_box: tuple[float, float]


class RouteStart(NamedTuple):
    """
    One starting geometry, written in both routes' variables.

    Attributes
    ----------
    label :
        Name of the start, by its multiple of the funicular rise.
    density :
        The uniform force density whose form-found shape is this geometry.
    heights :
        The height of every free node of that same geometry.
    """

    label: str
    density: Float[Array, "1"]
    heights: Float[Array, "nodes_free"]


class HeightStart(NamedTuple):
    """
    One starting geometry only the height route can take.

    Attributes
    ----------
    label :
        Name of the start.
    heights :
        The height of every free node.
    """

    label: str
    heights: Float[Array, "nodes_free"]


class RouteRun(NamedTuple):
    """
    One descent, and the bending ratio replayed along it.

    Attributes
    ----------
    start :
        Name of the start the descent left from.
    found :
        The answer, and every iterate on the way to it.
    bending :
        Largest bending-to-axial ratio of any member at every iterate, under
        the load case the shape answers to.
    """

    start: str
    found: SearchResult
    bending: Float[np.ndarray, "steps"]


class StaggeredRun(NamedTuple):
    """
    One descent with the coupling closed, or the word that it never closed.

    Attributes
    ----------
    route :
        Which parametrization descended.
    start :
        Name of the start the descent left from.
    found :
        The answer at settled sections, or None where the settling residual
        survived every round the driver was given.
    """

    route: str
    start: str
    found: SearchResult | None


class FinalRead(NamedTuple):
    """
    The design a descent arrived at, read back against the true largest case.

    Attributes
    ----------
    mass :
        Mass at the sections the unsmoothed envelope demands.
    rise :
        Height of the crown of the final geometry.
    mirror :
        How far the diameters depart from their own reflection.
    deviation :
        Worst departure of any per-case utilization from exactly one.
    xyz :
        Position of every node of the final geometry.
    diameters :
        Reconciled outer diameter of every member.
    governing :
        Index of the load case working each member hardest.
    """

    mass: float
    rise: float
    mirror: float
    deviation: float
    xyz: Float[np.ndarray, "nodes 3"]
    diameters: Float[np.ndarray, "edges"]
    governing: Int[np.ndarray, "edges"]


class RouteOutcomes(NamedTuple):
    """
    Everything one route's descents produced, in start order.

    Attributes
    ----------
    runs :
        The descents, each with its bending ratio replayed.
    finals :
        The answers read back at the frozen seed sections.
    """

    runs: tuple[RouteRun, ...]
    finals: tuple[FinalRead, ...]


# What a design builder is: a route's variables and the analysis diameters
# in — None analyzing at the seed — and the by-case design out.
DesignBuilder = Callable[
    [ArchProblem, Float[Array, "variables"], Float[Array, "edges"] | None],
    Design,
]

# What a bending measure is: a route's variables in, the worst ratio out.
BendingMeasure = Callable[[ArchProblem, Float[Array, "variables"]], Float[Array, ""]]


def mirror_gap(values: Float[np.ndarray, "edges"]) -> float:
    """
    How far a per-member quantity departs from its own reflection.

    The arch is a chain built left to right, so reversing the array reflects
    the design about midspan. Scaled by the largest entry, so the number reads
    the same whatever the quantity is.
    """
    values = np.asarray(values)
    scale = float(np.max(np.abs(values)))
    departure = float(np.max(np.abs(values - values[::-1])))

    return departure / scale if scale > 0.0 else 0.0


def build_load_cases(structure: Structure, weight: LoadConfig) -> LoadCases:
    """
    Three cases of equal total: funicular, half span, and its mirror.
    """
    spread = weight.total / (structure.num_edges - 1)

    uniform = loads_uniform(structure, spread)

    half = loads_half_span(structure, spread, factor=weight.half_factor)
    half = half * (weight.total / abs(float(jnp.sum(half[:, 2]))))

    mirrored = loads_half_span(
        structure, spread, factor=weight.half_factor, mirrored=True
    )
    mirrored = mirrored * (weight.total / abs(float(jnp.sum(mirrored[:, 2]))))

    cases = [uniform, half, mirrored]

    return assemble_load_cases(cases)


def arch_problem(config: TaskConfig) -> ArchProblem:
    """
    The arch, its prepared blocks, and the `q` that reaches the rise.

    The blocks are built here, on the host, because preparing the analysis
    reads support flags in Python, which a tracer cannot follow. The free node
    indices are read off the supports once and shipped as an array, so the
    direct route's scatter is a device operation rather than a rebuild.
    """
    arch = config.structure
    structure = build_arch_2d(arch.num_edges, arch.span, arch.rise)
    loads = build_load_cases(structure, config.loads)
    formfinder = FdmFormFinder(structure)

    trial = jnp.full(structure.num_edges, -1.0)
    shape = formfinder(trial, loads.formfinding)
    reached = jnp.max(shape.xyz[:, 2])

    family = thinnest_family(GRADE, SECTION_CLASS)
    blocks = StructuralDesignPipeline(
        formfinder,
        SmaxAnalyzer(structure, family(config.analysis.diameter)),
        Ec3Sizer(structure, family),
    )

    funicular = trial * reached / arch.rise
    everyone = np.arange(structure.num_nodes)
    frees = np.setdiff1d(everyone, np.asarray(structure.supports))
    nodes_free = jnp.asarray(frees)

    diameters_seed = jnp.full(structure.num_edges, config.analysis.diameter)
    decades = config.search.density_decades
    box = (float(funicular[0]) * decades, float(funicular[0]) / decades)
    heights_box = (0.0, arch.span)

    problem = ArchProblem(
        structure,
        blocks,
        loads,
        funicular,
        nodes_free,
        diameters_seed,
        box,
        heights_box,
    )

    return problem


@eqx.filter_jit
def density_design(
    problem: ArchProblem,
    density: Float[Array, "1"],
    diameters: Float[Array, "edges"] | None = None,
) -> Design:
    """
    The by-case design at one uniform force density, through the form finder.

    Parameters
    ----------
    problem :
        The prepared arch and its blocks.
    density :
        The one force density every member shares.
    diameters :
        Outer diameter the frame is analyzed with, or None for the seed.

    Returns
    -------
    design :
        The shape, the forces, and what each load case demands on its own.
    """
    densities = jnp.broadcast_to(density, (problem.structure.num_edges,))
    seed = problem.diameters_seed if diameters is None else diameters
    params = DesignParameters(densities, seed)

    return problem.pipeline(params, problem.loads)


@eqx.filter_jit
def heights_design(
    problem: ArchProblem,
    heights: Float[Array, "nodes_free"],
    diameters: Float[Array, "edges"] | None = None,
) -> Design:
    """
    The by-case design of the geometry the heights describe, no form finder.

    Parameters
    ----------
    problem :
        The prepared arch and its blocks.
    heights :
        The height of every free node, the plan being held where it started.
    diameters :
        Outer diameter the frame is analyzed with, or None for the seed.

    Returns
    -------
    design :
        The shape, the forces, and what each load case demands on its own.

    Notes
    -----
    The pipeline minus its first block: the geometry is written down rather
    than found, and the same analysis and check run on it. What disappears
    with the form finder is the guarantee that the shaping case is carried
    axially — here bending is whatever the heights imply.
    """
    xyz = problem.structure.nodes.at[problem.nodes_free, 2].set(heights)
    lengths = member_lengths(xyz, problem.structure.edges)
    shape = FormFoundShape(xyz, lengths)

    seed = problem.diameters_seed if diameters is None else diameters
    forces = problem.pipeline.analyzer(shape.xyz, seed, problem.loads.analysis)
    sizes = problem.pipeline.sizer(forces, shape.lengths)

    return Design(shape, forces, sizes)


def bending_ratio(design: Design) -> Float[Array, ""]:
    """
    The largest end moment against the axial couple, under the shaping case.

    Parameters
    ----------
    design :
        A design still carrying its load case axis.

    Returns
    -------
    ratio :
        The worst `|M| / (|N| L)` of any member under the first load case.

    Notes
    -----
    Dimensionless, and near zero exactly where a shape carries its shaping
    case axially. The first case is the one the funicular route answers to by
    construction, so this is the measure on which the two routes must differ
    if the physics prior is doing anything at all.

    The denominator is walled at one newton-millimeter. The wall binds only
    where a member carries no axial force at all — the flat start's straight
    beam — and turns the pure-bending limit's infinite ratio into a finite,
    plottable one.
    """
    moments = jnp.abs(design.forces.moment_major[0])
    axial = jnp.abs(design.forces.axial_force[0])
    couples = jnp.maximum(axial * design.shape.lengths, 1.0)
    ratio = jnp.max(moments / couples[:, None])

    return ratio


@eqx.filter_jit
def density_bending(
    problem: ArchProblem,
    density: Float[Array, "1"],
) -> Float[Array, ""]:
    """
    The bending ratio at one uniform force density.
    """
    design = density_design(problem, density)

    return bending_ratio(design)


@eqx.filter_jit
def heights_bending(
    problem: ArchProblem,
    heights: Float[Array, "nodes_free"],
) -> Float[Array, ""]:
    """
    The bending ratio at one set of free heights.
    """
    design = heights_design(problem, heights)

    return bending_ratio(design)


def matched_starts(
    problem: ArchProblem,
    config: TaskConfig,
) -> tuple[RouteStart, ...]:
    """
    The same starting geometries, written in both routes' variables.

    Parameters
    ----------
    problem :
        The prepared arch, supplying the funicular force density.
    config :
        The run description, supplying the rise multiples.

    Returns
    -------
    starts :
        One start per rise multiple, exactly shared by the two routes.

    Notes
    -----
    The force density system is linear in the coordinates, so scaling a
    uniform density by `1 / f` scales every free height by `f` and moves the
    plan not at all. A start is therefore matched exactly rather than
    approximately: the density route's form-found shape and the height
    route's written-down one are the same geometry, which `report_starts`
    measures rather than assumes.
    """
    shape = problem.pipeline.formfinder(problem.q, problem.loads.formfinding)
    reference = shape.xyz[problem.nodes_free, 2]

    starts = []
    for fraction in config.starts.rise_fractions:
        density = problem.q[:1] / fraction
        heights = fraction * reference
        starts.append(RouteStart(f"{fraction:g}x rise", density, heights))

    return tuple(starts)


def unmatched_starts(
    problem: ArchProblem,
    config: TaskConfig,
) -> tuple[HeightStart, ...]:
    """
    Starts off the funicular manifold, which no force density can express.

    Parameters
    ----------
    problem :
        The prepared arch, supplying the free node count.
    config :
        The run description, supplying the random draw and its ceiling.

    Returns
    -------
    starts :
        A flat line on the ground plane, and reproducibly random heights.

    Notes
    -----
    The no-prior scenario. The matched starts all lie on the funicular curve,
    so a descent from them borrows the prior once even in the height route;
    these two owe it nothing. The flat line is the blank page — the shaping
    case carried in pure bending — and the random heights are the adversarial
    page. Both are descended by the height route alone, the density route
    having no variable that reaches them.
    """
    count = int(problem.nodes_free.shape[0])
    flat = jnp.zeros(count)

    ceiling = config.starts.ceiling_fraction * config.structure.rise
    generator = np.random.default_rng(config.starts.random_seed)
    drawn = generator.uniform(0.0, ceiling, size=count)
    random = jnp.asarray(drawn)

    starts = (HeightStart("flat line", flat), HeightStart("random", random))

    return starts


def density_descent(
    problem: ArchProblem,
    search: SearchConfig,
    start: Float[Array, "1"],
) -> SearchResult:
    """
    Descend on the single force density, annealed over the shared schedule.
    """

    def objective(density, beta):
        design = density_design(problem, density)
        envelope = design_envelope(design, beta)

        return compute_mass(envelope)

    schedule = annealing_schedule(search.beta_start, search.beta_stop, search.rounds)
    found = optimize_annealed(
        objective, start, schedule, bounds=problem.bounds, iterations=search.iterations
    )

    return found


def heights_descent(
    problem: ArchProblem,
    search: SearchConfig,
    start: Float[Array, "nodes_free"],
) -> SearchResult:
    """
    Descend on the free heights, annealed over the shared schedule.
    """

    def objective(heights, beta):
        design = heights_design(problem, heights)
        envelope = design_envelope(design, beta)

        return compute_mass(envelope)

    schedule = annealing_schedule(search.beta_start, search.beta_stop, search.rounds)
    found = optimize_annealed(
        objective,
        start,
        schedule,
        bounds=problem.heights_box,
        iterations=search.iterations,
    )

    return found


def staggered_objective(
    problem: ArchProblem,
    builder: DesignBuilder,
    sharpness: Float[Array, ""],
) -> Callable[[DesignParameters], tuple[Float[Array, ""], Design]]:
    """
    One route's enveloped mass as a function of whole design parameters.

    Parameters
    ----------
    problem :
        The prepared arch.
    builder :
        The route's design builder.
    sharpness :
        Envelope sharpness held for every round, the schedule's last.

    Returns
    -------
    weigh_design :
        The mass and the design behind it, as the staggered driver requires.

    Notes
    -----
    The staggered driver reads the container's first field as nothing more
    than the variables a descent moves, so the height route's heights ride
    where force densities usually do. The envelope holds the final sharpness
    throughout: the rounds close a coupling rather than anneal a smoothing,
    and settling needs the reconciled one-diameter-per-member it produces.
    """

    def weigh_design(params: DesignParameters) -> tuple[Float[Array, ""], Design]:
        design = builder(problem, params.force_densities, params.diameters)
        envelope = design_envelope(design, sharpness)

        return compute_mass(envelope), envelope

    return weigh_design


def run_staggered(
    report: Report,
    problem: ArchProblem,
    search: SearchConfig,
    starts: tuple[RouteStart, ...],
    extras: tuple[HeightStart, ...],
) -> tuple[StaggeredRun, ...]:
    """
    Both routes again from every start, the coupling closed between rounds.

    Parameters
    ----------
    report :
        Where each run's one-line result is written as it lands.
    problem :
        The prepared arch.
    search :
        The budgets every bounded descent shares.
    starts :
        The matched starts, each taken by both routes.
    extras :
        The starts off the funicular manifold, taken by the height route.

    Returns
    -------
    runs :
        One entry per descent, in the order they ran.

    Notes
    -----
    A run whose coupling never closes is reported and carried as None rather
    than allowed to kill the experiment: the driver raises where the sizes
    and the stiffnesses chase each other, and that a start does so is a
    finding about the start.
    """
    sharpness = jnp.asarray(search.beta_stop)
    weigh_density = staggered_objective(problem, density_design, sharpness)
    weigh_heights = staggered_objective(problem, heights_design, sharpness)

    jobs = []
    for start in starts:
        jobs.append(
            (ROUTE_DENSITY, start.label, weigh_density, start.density, problem.bounds)
        )
        jobs.append(
            (
                ROUTE_HEIGHTS,
                start.label,
                weigh_heights,
                start.heights,
                problem.heights_box,
            )
        )
    for extra in extras:
        jobs.append(
            (
                ROUTE_HEIGHTS,
                extra.label,
                weigh_heights,
                extra.heights,
                problem.heights_box,
            )
        )

    runs = []
    for route, label, weighed, start, bounds in jobs:
        seeded = DesignParameters(start, problem.diameters_seed)
        try:
            found = optimize_staggered(
                weighed, seeded, bounds=bounds, iterations=search.iterations
            )
        except ValueError as stalled:
            report.write_line(f"{route}, {label}: {stalled}")
            runs.append(StaggeredRun(route, label, None))
            continue

        runs.append(StaggeredRun(route, label, found))
        report.write_line(f"{route}, {label}: {float(found.value):.9f} t staggered")

    return tuple(runs)


def trajectory_bending(
    measure: BendingMeasure,
    problem: ArchProblem,
    walked: Trajectory,
) -> Float[np.ndarray, "steps"]:
    """
    The bending ratio at every iterate, replayed through one compiled read.

    Parameters
    ----------
    measure :
        The route's compiled bending read.
    problem :
        The prepared arch.
    walked :
        The iterates a descent recorded.

    Returns
    -------
    ratios :
        The worst bending-to-axial ratio at every iterate, in walk order.
    """
    ratios = []
    for step in range(len(walked.mass)):
        ratio = measure(problem, walked.q[step])
        ratios.append(float(ratio))

    return np.asarray(ratios)


def report_starts(
    report: Report,
    problem: ArchProblem,
    starts: tuple[RouteStart, ...],
) -> None:
    """
    The matched starts measured: one geometry, two sets of variables.
    """
    columns = (
        ReportColumn("start", align="<"),
        ReportColumn("rise [mm]", ".1f"),
        ReportColumn("geometry gap [mm]", ".2e"),
        ReportColumn("mass, one density [t]", ".9f"),
        ReportColumn("mass, free heights [t]", ".9f"),
    )
    rows = []
    for start in starts:
        densities = jnp.broadcast_to(start.density, (problem.structure.num_edges,))
        found = problem.pipeline.formfinder(densities, problem.loads.formfinding)
        written = problem.structure.nodes.at[problem.nodes_free, 2].set(start.heights)
        gap = float(jnp.max(jnp.abs(found.xyz - written)))
        rise = float(jnp.max(written[:, 2]))

        weighed = design_envelope(density_design(problem, start.density), None)
        direct = design_envelope(heights_design(problem, start.heights), None)
        rows.append(
            (
                start.label,
                rise,
                gap,
                float(compute_mass(weighed)),
                float(compute_mass(direct)),
            )
        )

    report.write_line("Two routes to one geometry, started matched")
    report.write_table(columns, rows)


def report_gradient(
    report: Report,
    problem: ArchProblem,
    search: SearchConfig,
    heights: Float[Array, "nodes_free"],
) -> float:
    """
    The direct route's gradient against a directional central difference.

    Parameters
    ----------
    report :
        Where the sweep is written.
    problem :
        The prepared arch.
    search :
        The budgets, supplying the sharpness the descents differentiate at.
    heights :
        The point the derivative is taken at, the matched start.

    Returns
    -------
    best :
        The smallest scaled disagreement over the swept steps.

    Notes
    -----
    The density route's gradient was validated in experiment 03 against the
    whole uniform sweep, and the single density is that family restricted, so
    only the new differentiation path — coordinates straight into the
    analysis, no form finder — is checked here. One direction rather than
    nineteen, the direction being the start itself, so the probe moves every
    variable at once.
    """
    sharpness = jnp.asarray(search.beta_stop)

    def objective(probed: Float[Array, "nodes_free"]) -> Float[Array, ""]:
        design = heights_design(problem, probed)
        envelope = design_envelope(design, sharpness)

        return compute_mass(envelope)

    weighed = eqx.filter_jit(objective)
    sloped = eqx.filter_jit(jax.grad(objective))

    direction = heights / jnp.linalg.norm(heights)
    exact = float(jnp.sum(sloped(heights) * direction))
    magnitude = float(jnp.linalg.norm(heights))

    columns = (
        ReportColumn("relative step", ".0e"),
        ReportColumn("central difference", ".9e"),
        ReportColumn("scaled error", ".2e"),
    )
    rows = []
    best = float("inf")
    for relative in GRADIENT_STEPS:
        step = magnitude * relative
        forward = float(weighed(heights + step * direction))
        backward = float(weighed(heights - step * direction))
        quotient = (forward - backward) / (2.0 * step)
        scaled = abs(exact - quotient) / abs(exact)
        best = min(best, scaled)
        rows.append((relative, quotient, scaled))

    entries = (
        ("exact directional derivative", f"{exact:.9e}"),
        ("best scaled error", f"{best:.2e} ({TOLERANCE_GRADIENT:.0e})"),
    )

    report.write_heading("The direct route's gradient, checked along the start")
    report.write_table(columns, rows)
    report.write_entries(entries)

    return best


def run_routes(
    report: Report,
    problem: ArchProblem,
    search: SearchConfig,
    starts: tuple[RouteStart, ...],
) -> tuple[tuple[RouteRun, ...], tuple[RouteRun, ...]]:
    """
    Both descents from every start, each with its bending ratio replayed.

    Parameters
    ----------
    report :
        Where each run's one-line result is written as it lands.
    problem :
        The prepared arch.
    search :
        The budgets every bounded descent shares.
    starts :
        The matched starts, each descended from twice.

    Returns
    -------
    runs :
        The density route's runs and the height route's runs, start by start.
    """
    runs_density = []
    runs_heights = []
    for start in starts:
        found = density_descent(problem, search, start.density)
        walked = trajectory_bending(density_bending, problem, found.trajectory)
        runs_density.append(RouteRun(start.label, found, walked))
        report.write_line(
            f"one density, {start.label}: {float(found.value):.9f} t "
            f"in {len(walked)} iterates"
        )

        found = heights_descent(problem, search, start.heights)
        walked = trajectory_bending(heights_bending, problem, found.trajectory)
        runs_heights.append(RouteRun(start.label, found, walked))
        report.write_line(
            f"free heights, {start.label}: {float(found.value):.9f} t "
            f"in {len(walked)} iterates"
        )

    return tuple(runs_density), tuple(runs_heights)


def run_unmatched(
    report: Report,
    problem: ArchProblem,
    search: SearchConfig,
    starts: tuple[HeightStart, ...],
) -> tuple[RouteRun, ...]:
    """
    The height route alone, descended from the starts only it can take.

    Parameters
    ----------
    report :
        Where each run's one-line result is written as it lands.
    problem :
        The prepared arch.
    search :
        The budgets every bounded descent shares.
    starts :
        The starts off the funicular manifold.

    Returns
    -------
    runs :
        One run per start, in start order.
    """
    runs = []
    for start in starts:
        opening = design_envelope(heights_design(problem, start.heights), None)
        began = float(compute_mass(opening))

        found = heights_descent(problem, search, start.heights)
        walked = trajectory_bending(heights_bending, problem, found.trajectory)
        runs.append(RouteRun(start.label, found, walked))
        report.write_line(
            f"free heights, {start.label}: {began:.9f} t to "
            f"{float(found.value):.9f} t in {len(walked)} iterates"
        )

    return tuple(runs)


def read_final(
    problem: ArchProblem,
    builder: DesignBuilder,
    answer: Float[Array, "variables"],
    diameters: Float[Array, "edges"] | None = None,
) -> FinalRead:
    """
    One answer read back against the true largest of the load cases.

    The diameters are the sections the answer is analyzed at — None for the
    frozen seed, or the settled sections a staggered descent closed on.
    """
    by_case = builder(problem, answer, diameters)
    sized = design_envelope(by_case, None)

    mass = float(compute_mass(sized))
    rise = float(jnp.max(sized.shape.xyz[:, 2]))
    diameters = np.asarray(sized.sizes.sections.diameter)
    deviation = float(jnp.max(jnp.abs(by_case.sizes.utilization - 1.0)))
    governing = np.asarray(governing_compiled(by_case.sizes.sections.diameter))
    positions = np.asarray(sized.shape.xyz)

    final = FinalRead(
        mass, rise, mirror_gap(diameters), deviation, positions, diameters, governing
    )

    return final


def read_finals(
    problem: ArchProblem,
    builder: DesignBuilder,
    runs: tuple[RouteRun, ...],
) -> tuple[FinalRead, ...]:
    """
    Every run's answer read back, in run order.
    """
    finals = []
    for run in runs:
        answer = run.found.trajectory.q[-1]
        finals.append(read_final(problem, builder, answer, None))

    return tuple(finals)


def report_descents(
    report: Report,
    density: RouteOutcomes,
    heights: RouteOutcomes,
) -> None:
    """
    What both routes arrived at from every start, side by side.

    One row per run rather than per matched pair, because two of the starts
    have no pair: the density rows first, then every height row, the
    unmatched ones last. The final ratio column is the last iterate of the
    replayed bending series, which is the answer's own funicularity.
    """
    columns = (
        ReportColumn("start", align="<"),
        ReportColumn("route", align="<"),
        ReportColumn("variables"),
        ReportColumn("iterates"),
        ReportColumn("mass [t]", ".9f"),
        ReportColumn("rise [mm]", ".1f"),
        ReportColumn("mirror gap", ".2e"),
        ReportColumn("peak |M|/(|N|L)", ".2e"),
        ReportColumn("final |M|/(|N|L)", ".2e"),
    )
    rows = []
    labeled = (
        (ROUTE_DENSITY, density.runs, density.finals),
        (ROUTE_HEIGHTS, heights.runs, heights.finals),
    )
    for route, runs, finals in labeled:
        for run, final in zip(runs, finals):
            rows.append(
                (
                    run.start,
                    route,
                    int(run.found.trajectory.q.shape[1]),
                    len(run.bending),
                    final.mass,
                    final.rise,
                    final.mirror,
                    float(np.max(run.bending)),
                    float(run.bending[-1]),
                )
            )

    report.write_heading("The descents, both routes from every start")
    report.write_table(columns, rows)


def report_staggered(
    report: Report,
    problem: ArchProblem,
    staggered: tuple[StaggeredRun, ...],
    frozen: dict[tuple[str, str], FinalRead],
) -> dict[tuple[str, str], FinalRead]:
    """
    The frozen-seed answers against the coupling closed, run by run.

    Parameters
    ----------
    report :
        Where the comparison is written.
    problem :
        The prepared arch.
    staggered :
        Every staggered run, the never-closed ones included.
    frozen :
        The frozen-seed finals, keyed by route and start.

    Returns
    -------
    closed :
        The closed runs' answers read back at their settled sections, keyed
        like the frozen finals — for the utilization invariant and for the
        spread panel's hollow markers.

    Notes
    -----
    Each staggered answer is re-read at the sections its last settling pass
    demanded, so the mass in its column belongs to a frame analyzed at its
    own sections — the number the frozen column only approximates. What the
    column pair prices is the `∂N/∂d` feedback every descent above froze.
    """
    columns = (
        ReportColumn("start", align="<"),
        ReportColumn("route", align="<"),
        ReportColumn("mass, frozen [t]", ".9f"),
        ReportColumn("mass, staggered [t]", ".9f"),
        ReportColumn("coupling moved", "+.4%"),
        ReportColumn("rise [mm]", ".1f"),
    )
    rows = []
    closed = {}
    shifts = []
    for run in staggered:
        if run.found is None:
            continue
        builder = density_design if run.route == ROUTE_DENSITY else heights_design
        answer = run.found.trajectory.q[-1]
        settled = run.found.aux.sizes.sections.diameter
        final = read_final(problem, builder, answer, settled)

        before = frozen[(run.route, run.start)]
        moved = final.mass / before.mass - 1.0
        rows.append((run.start, run.route, before.mass, final.mass, moved, final.rise))
        closed[(run.route, run.start)] = final
        shifts.append(abs(moved))

    report.write_heading("The coupling closed, against the frozen seed")
    report.write_table(columns, rows)

    if shifts:
        entries = (("largest coupling shift", f"{max(shifts):.4%}"),)
        report.write_entries(entries)

    return closed


class ConstrainedMaps(NamedTuple):
    """
    The compiled maps a constrained descent calls, over one variable vector.

    Attributes
    ----------
    weigh :
        The mass and its gradient together.
    slack :
        How far under one every member's utilization sits, per load case.
    jacobian :
        The slack's derivative in every variable.
    """

    weigh: Callable[
        [Float[Array, "variables"]],
        tuple[Float[Array, ""], Float[Array, "variables"]],
    ]
    slack: Callable[[Float[Array, "variables"]], Float[Array, "constraints"]]
    jacobian: Callable[
        [Float[Array, "variables"]], Float[Array, "constraints variables"]
    ]


class SimultaneousAnswer(NamedTuple):
    """
    What the constrained search over density and diameters arrived at.

    Attributes
    ----------
    start :
        Name of the start the search left from.
    mass :
        Mass at the answer, of a frame analyzed at its own sections.
    rise :
        Height of the crown of the final geometry.
    worked :
        Largest utilization of any member under any load case.
    spent :
        Iterations and evaluations the solver reported.
    """

    start: str
    mass: float
    rise: float
    worked: float
    spent: str


def constrained_maps(problem: ArchProblem) -> ConstrainedMaps:
    """
    Compile the objective and the constraints the simultaneous solver calls.

    Parameters
    ----------
    problem :
        The prepared arch.

    Returns
    -------
    maps :
        The mass with its gradient, the slack, and the slack's Jacobian.

    Notes
    -----
    The variable vector is the one force density followed by every diameter,
    so the analysis runs at the search's own sections and the `∂N/∂d`
    feedback rides inside the gradient — nothing is frozen and nothing is
    settled. The check enters as `compute_utilization`, sizes the caller
    owns, so there is no sizing solve and no envelope: one diameter per
    member has to satisfy every case at once.
    """
    family = problem.pipeline.sizer.family
    members = problem.structure.num_edges

    def weigh(x: Float[Array, "variables"]) -> Float[Array, ""]:
        densities = jnp.broadcast_to(x[:1], (members,))
        shape = problem.pipeline.formfinder(densities, problem.loads.formfinding)
        sections = family(x[1:])
        mass = jnp.sum(sections.area * shape.lengths) * family.material.density

        return mass

    def slack(x: Float[Array, "variables"]) -> Float[Array, "constraints"]:
        densities = jnp.broadcast_to(x[:1], (members,))
        diameters = x[1:]
        shape = problem.pipeline.formfinder(densities, problem.loads.formfinding)
        forces = problem.pipeline.analyzer(shape.xyz, diameters, problem.loads.analysis)
        used = problem.pipeline.sizer.compute_utilization(
            diameters, forces, shape.lengths
        )

        return 1.0 - used.ravel()

    # Forward mode for the Jacobian: twenty-one columns against sixty rows.
    maps = ConstrainedMaps(
        jax.jit(jax.value_and_grad(weigh)),
        jax.jit(slack),
        jax.jit(jax.jacfwd(slack)),
    )

    return maps


def simultaneous_descent(
    maps: ConstrainedMaps,
    problem: ArchProblem,
    searched: SimultaneousConfig,
    start: Float[Array, "variables"],
) -> tuple[Float[np.ndarray, "variables"], str]:
    """
    SLSQP over the density and the diameters at once, under hard `U <= 1`.

    Parameters
    ----------
    maps :
        The compiled objective and constraints.
    problem :
        The prepared arch, supplying the density box.
    searched :
        The budgets of the constrained search.
    start :
        The variable vector to start from.

    Returns
    -------
    answer :
        The variables the solver stopped on, and what it spent getting there.
    """

    def objective(x):
        value, slope = maps.weigh(jnp.asarray(x))

        return float(value), np.asarray(slope, dtype=np.float64)

    def feasible(x):
        return np.asarray(maps.slack(jnp.asarray(x)), dtype=np.float64)

    def feasible_jacobian(x):
        return np.asarray(maps.jacobian(jnp.asarray(x)), dtype=np.float64)

    held = {"type": "ineq", "fun": feasible, "jac": feasible_jacobian}
    boxes = [problem.bounds] + [
        (searched.diameter_floor, None)
    ] * problem.structure.num_edges

    found = minimize(
        objective,
        np.asarray(start, dtype=np.float64),
        jac=True,
        method="SLSQP",
        bounds=boxes,
        constraints=[held],
        options={"maxiter": searched.iterations, "ftol": searched.tolerance},
    )
    spent = f"{found.nit} iterations, {found.nfev} evaluations"

    return np.asarray(found.x), spent


def report_simultaneous(
    report: Report,
    problem: ArchProblem,
    searched: SimultaneousConfig,
    starts: tuple[RouteStart, ...],
    closed: dict[tuple[str, str], FinalRead],
) -> tuple[SimultaneousAnswer, ...]:
    """
    The constrained search from every matched start, against the staggered.

    Parameters
    ----------
    report :
        Where the comparison is written.
    problem :
        The prepared arch.
    searched :
        The budgets of the constrained search.
    starts :
        The matched starts — the only ones a density variable can take.
    closed :
        The staggered finals, the baseline this formulation is priced against.

    Returns
    -------
    answers :
        One answer per start, in start order.

    Notes
    -----
    Each search starts at the start's own geometry with the sections the
    check demanded of it there, so a run measures the formulation rather
    than the luck of an arbitrary diameter guess. The comparison column is
    the staggered answer — a self-consistent frame either way, so what the
    pair prices is holding the coupling inside the gradient against closing
    it between rounds.
    """
    maps = constrained_maps(problem)

    columns = (
        ReportColumn("start", align="<"),
        ReportColumn("mass [t]", ".9f"),
        ReportColumn("rise [mm]", ".1f"),
        ReportColumn("max utilization", ".9f"),
        ReportColumn("vs staggered", "+.4%"),
        ReportColumn("spent", align="<"),
    )
    rows = []
    answers = []
    for start in starts:
        opening = design_envelope(density_design(problem, start.density), None)
        sized = opening.sizes.sections.diameter
        begin = jnp.concatenate([start.density, sized])

        variables, spent = simultaneous_descent(maps, problem, searched, begin)
        answer = jnp.asarray(variables)

        members = problem.structure.num_edges
        densities = jnp.broadcast_to(answer[:1], (members,))
        shape = problem.pipeline.formfinder(densities, problem.loads.formfinding)
        forces = problem.pipeline.analyzer(
            shape.xyz, answer[1:], problem.loads.analysis
        )
        used = problem.pipeline.sizer.compute_utilization(
            answer[1:], forces, shape.lengths
        )
        weighed, _ = maps.weigh(answer)

        mass = float(weighed)
        rise = float(jnp.max(shape.xyz[:, 2]))
        worked = float(jnp.max(used))
        base = closed.get((ROUTE_DENSITY, start.label))
        gained = mass / base.mass - 1.0 if base is not None else float("nan")

        rows.append((start.label, mass, rise, worked, gained, spent))
        answers.append(SimultaneousAnswer(start.label, mass, rise, worked, spent))

    report.write_heading("Density and diameters as one constrained search")
    report.write_table(columns, rows)

    return tuple(answers)


def write_figures(
    problem: ArchProblem,
    density: RouteOutcomes,
    heights: RouteOutcomes,
    closed: dict[tuple[str, str], FinalRead],
    constrained: float,
) -> None:
    """
    The comparison figure, and the final forms member by member.

    The spread panel carries every coupling — filled markers at the frozen
    seed, hollow ones at the settled sections, and the simultaneous answer
    as one dashed level, it being the same mass from every start — while
    the descent and bending panels stay with the frozen matched-start runs:
    a staggered trajectory concatenates rounds, and its seams belong to the
    driver rather than to either parametrization.
    """
    FIGURES.mkdir(exist_ok=True)

    matched_q, matched_z = density.runs[0], heights.runs[0]
    traces = (
        RouteTrace(
            "one force density",
            np.asarray(matched_q.found.trajectory.mass),
            matched_q.bending,
        ),
        RouteTrace(
            "free heights",
            np.asarray(matched_z.found.trajectory.mass),
            matched_z.bending,
        ),
    )
    # A start a route never took carries NaN in that route's slot, which
    # the figure simply does not draw.
    labels = tuple(run.start for run in heights.runs)
    unpaired = len(heights.runs) - len(density.runs)
    padded = [final.mass for final in density.finals] + [float("nan")] * unpaired
    spread = StartSpread(
        labels,
        np.asarray(padded),
        np.asarray([final.mass for final in heights.finals]),
    )

    def settled_mass(route: str, start: str) -> float:
        final = closed.get((route, start))

        return final.mass if final is not None else float("nan")

    settled = StartSpread(
        labels,
        np.asarray([settled_mass(ROUTE_DENSITY, label) for label in labels]),
        np.asarray([settled_mass(ROUTE_HEIGHTS, label) for label in labels]),
    )
    comparison = figure_parametrization(traces, spread, settled, constrained)
    comparison.savefig(FIGURES / "15_parametrization.png", dpi=200)

    reference = problem.pipeline.formfinder(problem.q, problem.loads.formfinding)
    forms = [
        Form(
            f"One density, {matched_q.start}, {density.finals[0].mass:.4f} t",
            density.finals[0].xyz,
            density.finals[0].diameters,
            density.finals[0].governing,
        )
    ]
    for run, final in zip(heights.runs, heights.finals):
        title = f"Free heights, {run.start}, {final.mass:.4f} t"
        forms.append(Form(title, final.xyz, final.diameters, final.governing))

    cases = figure_load_cases(
        problem.structure.edges, forms, CASE_NAMES, reference=reference.xyz
    )
    cases.savefig(FIGURES / "15_forms.png", dpi=200)


def main(config_path: Path) -> None:
    """
    Descend every parametrization the file describes, and compare the routes.

    Parameters
    ----------
    config_path :
        File naming the arch and the budgets its searches spend.
    """
    report = Report()
    config = parse_config(config_path.read_text())
    problem = arch_problem(config)
    starts = matched_starts(problem, config)
    search = config.search

    report_starts(report, problem, starts)
    best_error = report_gradient(report, problem, search, starts[0].heights)

    report.write_heading("Descending both routes from every start")
    runs_density, runs_matched = run_routes(report, problem, search, starts)
    extras = unmatched_starts(problem, config)
    runs_heights = runs_matched + run_unmatched(report, problem, search, extras)
    density = RouteOutcomes(
        runs_density, read_finals(problem, density_design, runs_density)
    )
    heights = RouteOutcomes(
        runs_heights, read_finals(problem, heights_design, runs_heights)
    )
    report_descents(report, density, heights)

    report.write_heading("Closing the analysis and sizing coupling, per round")
    staggered = run_staggered(report, problem, search, starts, extras)
    frozen = {}
    for run, final in zip(density.runs, density.finals):
        frozen[(ROUTE_DENSITY, run.start)] = final
    for run, final in zip(heights.runs, heights.finals):
        frozen[(ROUTE_HEIGHTS, run.start)] = final
    closed = report_staggered(report, problem, staggered, frozen)

    answers = report_simultaneous(report, problem, config.simultaneous, starts, closed)

    finals_density = density.finals
    finals_heights = heights.finals
    masses_density = [final.mass for final in finals_density]
    masses_heights = [final.mass for final in finals_heights]
    spread_density = max(masses_density) / min(masses_density) - 1.0
    spread_heights = max(masses_heights) / min(masses_heights) - 1.0
    peak_density = max(float(np.max(run.bending)) for run in runs_density)
    read_back = finals_density + finals_heights + tuple(closed.values())
    deviations = [final.deviation for final in read_back]
    worst_deviation = max(deviations)

    # The like-for-like bending comparison is the matched trio alone: the
    # flat start's pure-bending opening would swamp it with its own premise.
    matched = len(config.starts.rise_fractions)
    peak_matched = max(float(np.max(run.bending)) for run in runs_heights[:matched])
    bent_more = peak_matched / peak_density

    best_simultaneous = min(answer.mass for answer in answers)
    worst_worked = max(answer.worked for answer in answers)

    matched_above = finals_heights[0].mass / finals_density[0].mass - 1.0
    flat_final, random_final = finals_heights[matched], finals_heights[matched + 1]
    entries = (
        ("mass, one density, matched start", f"{finals_density[0].mass:.9f} t"),
        ("mass, free heights, matched start", f"{finals_heights[0].mass:.9f} t"),
        ("matched start, free heights end", f"{matched_above:+.2%} of one density"),
        ("mass, free heights, flat line", f"{flat_final.mass:.9f} t"),
        ("mass, free heights, random", f"{random_final.mass:.9f} t"),
        ("best one density over starts", f"{min(masses_density):.9f} t"),
        ("best free heights over starts", f"{min(masses_heights):.9f} t"),
        ("density route spread over starts", f"{spread_density:.2e}"),
        ("height route spread over starts", f"{spread_heights:.2e}"),
        ("peak bending ratio, one density", f"{peak_density:.2e}"),
        ("peak bending, matched free heights", f"{peak_matched:.2e}"),
        ("the matched iterates bent", f"{bent_more:.0f}x more"),
        ("worst utilization deviation", f"{worst_deviation:.2e}"),
        ("mass, simultaneous, matched start", f"{answers[0].mass:.9f} t"),
        ("best simultaneous over starts", f"{best_simultaneous:.9f} t"),
        ("worst simultaneous utilization", f"{worst_worked:.9f}"),
    )

    report.write_heading("Summary")
    report.write_entries(entries)

    write_figures(problem, density, heights, closed, best_simultaneous)

    checked_gradient = ToleranceCheck(
        "scaled directional gradient error", best_error, TOLERANCE_GRADIENT
    )
    checked_bending = ToleranceCheck(
        "density route bending ratio", peak_density, TOLERANCE_BENDING
    )
    checked_spread = ToleranceCheck(
        "density route spread over starts", spread_density, TOLERANCE_SPREAD
    )
    checked_utilization = ToleranceCheck(
        "worst utilization deviation", worst_deviation, TOLERANCE_UTILIZATION
    )
    violation = max(0.0, worst_worked - 1.0)
    checked_feasible = ToleranceCheck(
        "simultaneous constraint violation", violation, TOLERANCE_FEASIBILITY
    )
    checks = (
        checked_gradient,
        checked_bending,
        checked_spread,
        checked_utilization,
        checked_feasible,
    )

    # The superset claim is about the space, not about any one descent: the
    # height space contains the funicular curve, so its best answer over the
    # starts must reach at least as low. Which starts get there is the finding.
    beats_single = min(masses_heights) < min(masses_density)

    report.write_checks(checks)
    passed = checks_passed(checks) and beats_single

    report.write_verdict(passed)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else CONFIG)
