# SPDX-License-Identifier: Apache-2.0
"""
Sizes as a solver's answer, and sizes as an optimizer's variables.

The same cross-section check is used two ways on the same arch. Nested, the
sizer bisects `U(d) = 1` inside the pipeline and the implicit function theorem
carries the derivative; the analysis runs at seed sections, so the answer is
settled to self-consistency afterwards by forward passes. Simultaneous, the
diameters join the decision variables, the check becomes the inequality
constraint `U <= 1`, and a constrained optimizer finds the fully-stressed
state as active constraints — self-consistent by construction, no implicit
function anywhere, only the check's explicit hand-derived partials.

The bridge between the two is first-order optimality: at the constrained
optimum the active constraints sit at `U = 1` — the fully-stressed condition
re-derived as a KKT condition — and the multipliers are, member by member,
what the implicit function theorem prices a unit of utilization at.

A penalty is the stated fallback (`normax.optimization.penalized_mass`
through `minimize_bounded`), not the primary: the fully-stressed claim is an
equality to machine precision, and a penalty holds it only in a limit.

Blueprints is LGPL-2.1, experiment-only, waived 2026-08-15.

Run with `uv run --group pipeline python
experiments/13_simultaneous_sizing.py`.
"""

import time
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Float
from scipy.optimize import minimize

from normax.analysis.smax import SmaxAnalyzer
from normax.config import SizingConfig
from normax.design import Design
from normax.design import DesignParameters
from normax.design import StructuralDesignPipeline
from normax.design import compute_mass
from normax.form_finding import FdmFormFinder
from normax.loads import LoadCases
from normax.loads import assemble_load_cases
from normax.loads import create_load_uniform
from normax.materials import Steel355
from normax.optimization.nested import design_envelope
from normax.optimization.nested import minimize_bounded
from normax.optimization.nested import settle_diameters
from normax.optimization.nested import value_and_gradient
from normax.reporting import Report
from normax.reporting import ReportColumn
from normax.reporting import ToleranceCheck
from normax.reporting import verify_checks
from normax.sections import TubeCatalog
from normax.sizing.blueprint import DIAMETER_MINIMUM
from normax.structures import build_arch_2d
from normax.tesseract import build_sizer

TITLE = "Sizes as a solver's answer, and sizes as an optimizer's variables."

SPAN = 10_000.0
RISE = 3_000.0
TOTAL_LOAD = 180_000.0
NUM_EDGES = 10

# The cross-section check, across the boundary: the in-process sizer this
# experiment used was dissolved into the Tesseract backend.
SIZING = SizingConfig(3, "blueprint", False, False)

SEED = 100.0
RATIO = 50.0

# How far the joint search may move each force density off the funicular one.
SPREAD_UP = 2.0
SPREAD_DOWN = 0.5

DESCENT_ITERATIONS = 100
SOLVER_ITERATIONS = 200
SOLVER_TOLERANCE = 1e-12

# The two routes land on the same sizes to the optimizer's own tolerance.
TOLERANCE_AGREEMENT = 1e-5
TOLERANCE_FEASIBLE = 1e-6

# Measured 2.1e-2 on this arch: the analyzer's force-redistribution feedback,
# which the diagonal quotient ignores — a physical share, not a solver error.
TOLERANCE_REDISTRIBUTION = 5e-2

SIZES_COLUMNS = (
    ReportColumn("member", align="<"),
    ReportColumn("nested, settled [mm]", ".6f"),
    ReportColumn("simultaneous [mm]", ".6f"),
    ReportColumn("gap", ".2e"),
)

KKT_COLUMNS = (
    ReportColumn("member", align="<"),
    ReportColumn("max U", ".9f"),
    ReportColumn("multiplier [t]", ".6e"),
    ReportColumn("diagonal [t]", ".6e"),
    ReportColumn("gap", ".2e"),
)

ROUTE_COLUMNS = (
    ReportColumn("route", align="<"),
    ReportColumn("mass [t]", ".9f"),
    ReportColumn("evaluations", ""),
    ReportColumn("wall clock [s]", ".3f"),
)


class ArchProblem(NamedTuple):
    """
    Everything the two formulations share: one arch, one check, one load case.

    Attributes
    ----------
    pipeline :
        The three blocks, with the blueprints check as the sizer.
    params :
        Funicular force densities, and the seed analysis diameters.
    loads :
        The one assembled load case.
    """

    pipeline: StructuralDesignPipeline
    params: DesignParameters
    loads: LoadCases


class SolverState(NamedTuple):
    """
    What the KKT bridge reads off a finished constrained solve.

    Attributes
    ----------
    weigh :
        The mass as a traced function of the diameters.
    slack :
        The compiled constraint slack, one minus the utilization.
    slack_jacobian :
        The compiled Jacobian of that slack.
    """

    weigh: Callable
    slack: Callable
    slack_jacobian: Callable


class RouteAnswer(NamedTuple):
    """
    What one formulation reported: sizes, a mass, and what they cost to find.

    Attributes
    ----------
    diameters :
        One diameter per member, self-consistent with the analysis.
    mass :
        Total mass at those sizes.
    evaluations :
        How many objective or constraint evaluations the route spent.
    elapsed :
        Wall-clock seconds of the solve, compilation excluded.
    """

    diameters: Float[np.ndarray, "members"]
    mass: float
    evaluations: str
    elapsed: float


def arch_problem() -> ArchProblem:
    """
    The funicular arch, its blueprint sizer, and its one load case.
    """
    structure = build_arch_2d(num_edges=NUM_EDGES, span=SPAN, rise=RISE)
    grade = Steel355()
    catalog = TubeCatalog(RATIO, grade)
    pipeline = StructuralDesignPipeline(
        FdmFormFinder(structure),
        SmaxAnalyzer(structure, catalog(SEED)),
        build_sizer(structure, catalog, SIZING),
    )

    load_case = create_load_uniform(structure, TOTAL_LOAD / (NUM_EDGES - 1))
    loads = assemble_load_cases([load_case])

    trial = jnp.full(NUM_EDGES, -1.0)
    shape = pipeline.formfinder(trial, load_case)
    reached = jnp.max(shape.xyz[:, 2])
    params = DesignParameters(trial * reached / RISE, jnp.full(NUM_EDGES, SEED))

    return ArchProblem(pipeline, params, loads)


def design_objective(problem: ArchProblem) -> Callable:
    """
    The mass of a whole design, handing the design back beside it.
    """

    def objective(params: DesignParameters):
        # The nested route asks the check for a size rather than a verdict, so
        # it calls the sizer itself; calling the pipeline would run the held
        # check and echo the diameters back unsized.
        pipeline = problem.pipeline
        loads = problem.loads
        shape = pipeline.formfinder(params.coordinates, loads.formfinding)
        forces = pipeline.analyzer(shape.xyz, params.diameters, loads.analysis)
        sizes = pipeline.sizer(forces, shape.lengths)
        sized = design_envelope(Design(shape, forces, sizes))
        mass = compute_mass(sized)

        return mass, sized

    return objective


def nested_fixed(problem: ArchProblem) -> RouteAnswer:
    """
    The nested route at fixed force densities: size, then settle.

    Notes
    -----
    One sizer pass answers at the seed stiffness; the settle iterates the
    analysis at the sections just demanded until the two agree. Forward
    passes only — this is the fixed point the simultaneous route reaches as
    a constraint instead.
    """
    objective = design_objective(problem)
    compiled = jax.jit(objective)
    compiled(problem.params)

    started = time.perf_counter()
    settled = settle_diameters(objective, problem.params)
    resized = DesignParameters(problem.params.coordinates, settled)
    mass, _ = compiled(resized)
    elapsed = time.perf_counter() - started

    return RouteAnswer(np.asarray(settled), float(mass), "settle passes", elapsed)


def simultaneous_fixed(problem: ArchProblem) -> tuple[RouteAnswer, SolverState]:
    """
    The simultaneous route at fixed force densities: SLSQP over the diameters.
    """
    pipeline = problem.pipeline
    sizer = pipeline.sizer
    shape = pipeline.formfinder(problem.params.coordinates, problem.loads.formfinding)
    density = sizer.catalog.material.density

    def weigh(diameters):
        sections = sizer.catalog(diameters)

        return jnp.sum(sections.area * shape.lengths) * density

    def slack(diameters):
        forces = pipeline.analyzer(shape.xyz, diameters, problem.loads.analysis)
        used = sizer.compute_utilization(diameters, forces, shape.lengths)

        return 1.0 - used.ravel()

    compute_mass_and_gradient = jax.jit(jax.value_and_grad(weigh))
    slack_compiled = jax.jit(slack)
    slack_jacobian = jax.jit(jax.jacrev(slack))
    start = np.full(NUM_EDGES, SEED)
    compute_mass_and_gradient(jnp.asarray(start))
    slack_compiled(jnp.asarray(start))
    slack_jacobian(jnp.asarray(start))

    def objective(x):
        value, slope = compute_mass_and_gradient(jnp.asarray(x))

        return float(value), np.asarray(slope, dtype=np.float64)

    def feasible(x):
        return np.asarray(slack_compiled(jnp.asarray(x)), dtype=np.float64)

    def feasible_jacobian(x):
        return np.asarray(slack_jacobian(jnp.asarray(x)), dtype=np.float64)

    held = {"type": "ineq", "fun": feasible, "jac": feasible_jacobian}
    bounds = [(DIAMETER_MINIMUM, None)] * NUM_EDGES

    started = time.perf_counter()
    found = minimize(
        objective,
        start,
        jac=True,
        method="SLSQP",
        bounds=bounds,
        constraints=[held],
        options={"maxiter": SOLVER_ITERATIONS, "ftol": SOLVER_TOLERANCE},
    )
    elapsed = time.perf_counter() - started

    spent = f"{found.nit} it, {found.nfev} f"
    answer = RouteAnswer(np.asarray(found.x), float(found.fun), spent, elapsed)
    state = SolverState(weigh, slack_compiled, slack_jacobian)

    return answer, state


def kkt_bridge(report: Report, answer: RouteAnswer, state: SolverState) -> float:
    """
    The multipliers of the active constraints, two ways, and their gap.

    Notes
    -----
    Stationarity in the diameters gives the exact multipliers from the full
    constraint Jacobian; the diagonal quotient `-(dm/dd) / (dU/dd)` ignores
    the force redistribution an analysis feeds back, so the gap between the
    columns measures that feedback. The diagonal is what the implicit
    function theorem prices a member's utilization at — the nested route's
    adjoint, resurfacing as a multiplier.
    """
    diameters = jnp.asarray(answer.diameters)
    mass_slope = np.asarray(jax.grad(state.weigh)(diameters))
    jacobian = -np.asarray(state.slack_jacobian(diameters))
    used = 1.0 - np.asarray(state.slack(diameters))

    exact = np.linalg.lstsq(jacobian.T, -mass_slope, rcond=None)[0]
    diagonal = -mass_slope / np.diag(jacobian)
    gaps = np.abs(exact - diagonal) / np.max(np.abs(exact))

    rows = [
        (f"{index}", used[index], exact[index], diagonal[index], gaps[index])
        for index in range(NUM_EDGES)
    ]
    report.write_heading("The KKT bridge: multipliers against the implicit rule")
    report.write_table(KKT_COLUMNS, rows)
    report.write_note(
        "The diagonal quotient is the implicit function theorem's price of a "
        "unit of utilization; the gap is the analyzer's force redistribution."
    )

    return float(np.max(gaps))


def joint_bounds(problem: ArchProblem) -> list[tuple[float, float | None]]:
    """
    Box bounds for the joint search: force densities boxed, diameters floored.
    """
    funicular = float(problem.params.coordinates[0])
    lower = SPREAD_UP * funicular
    upper = SPREAD_DOWN * funicular
    force_box: list[tuple[float, float | None]] = [(lower, upper)] * NUM_EDGES
    size_box: list[tuple[float, float | None]] = [(DIAMETER_MINIMUM, None)] * NUM_EDGES

    return force_box + size_box


def nested_joint(problem: ArchProblem) -> RouteAnswer:
    """
    The nested route over the force densities: descend, then settle.

    Notes
    -----
    The descent's gradient is taken at the frozen seed sections, so it omits
    how the sizes react to the shape — the recorded coupling shortcut. The
    settle at the answer prices what that omission cost the forward value.
    """
    objective = design_objective(problem)
    seeds = problem.params.diameters

    def compute_mass(force_densities):
        return objective(DesignParameters(force_densities, seeds))

    compiled = value_and_gradient(compute_mass, has_aux=True)
    compiled(problem.params.coordinates)
    funicular = float(problem.params.coordinates[0])

    started = time.perf_counter()
    found = minimize_bounded(
        compute_mass,
        problem.params.coordinates,
        bounds=(SPREAD_UP * funicular, SPREAD_DOWN * funicular),
        iterations=DESCENT_ITERATIONS,
        has_aux=True,
        gradient=compiled,
    )
    answer = found.trajectory.q[-1]
    settled = settle_diameters(objective, DesignParameters(answer, seeds))
    mass, _ = objective(DesignParameters(answer, settled))
    elapsed = time.perf_counter() - started

    steps = found.trajectory.mass.shape[0] - 1
    spent = f"{steps} it + settle"

    return RouteAnswer(np.asarray(settled), float(mass), spent, elapsed)


def simultaneous_joint(problem: ArchProblem) -> RouteAnswer:
    """
    The simultaneous route over shape and sizes at once: SLSQP on (q, d).

    Notes
    -----
    Nothing is nested and nothing is frozen: the constraint Jacobian carries
    the check's partials, the analyzer's stiffness feedback and the form
    finder's geometry in one reverse pass, so the coupling the nested route
    omits is priced on every iteration rather than settled at the end.
    """
    pipeline = problem.pipeline
    sizer = pipeline.sizer
    density = sizer.catalog.material.density

    def split(x):
        return x[:NUM_EDGES], x[NUM_EDGES:]

    def weigh(x):
        force_densities, diameters = split(x)
        shape = pipeline.formfinder(force_densities, problem.loads.formfinding)
        sections = sizer.catalog(diameters)

        return jnp.sum(sections.area * shape.lengths) * density

    def slack(x):
        force_densities, diameters = split(x)
        shape = pipeline.formfinder(force_densities, problem.loads.formfinding)
        forces = pipeline.analyzer(shape.xyz, diameters, problem.loads.analysis)
        used = sizer.compute_utilization(diameters, forces, shape.lengths)

        return 1.0 - used.ravel()

    compute_mass_and_gradient = jax.jit(jax.value_and_grad(weigh))
    slack_compiled = jax.jit(slack)
    slack_jacobian = jax.jit(jax.jacrev(slack))
    seeded = jnp.concatenate([problem.params.coordinates, problem.params.diameters])
    start = np.asarray(seeded)
    compute_mass_and_gradient(seeded)
    slack_compiled(seeded)
    slack_jacobian(seeded)

    def objective(x):
        value, slope = compute_mass_and_gradient(jnp.asarray(x))

        return float(value), np.asarray(slope, dtype=np.float64)

    def feasible(x):
        return np.asarray(slack_compiled(jnp.asarray(x)), dtype=np.float64)

    def feasible_jacobian(x):
        return np.asarray(slack_jacobian(jnp.asarray(x)), dtype=np.float64)

    held = {"type": "ineq", "fun": feasible, "jac": feasible_jacobian}

    started = time.perf_counter()
    found = minimize(
        objective,
        start,
        jac=True,
        method="SLSQP",
        bounds=joint_bounds(problem),
        constraints=[held],
        options={"maxiter": SOLVER_ITERATIONS, "ftol": SOLVER_TOLERANCE},
    )
    elapsed = time.perf_counter() - started

    _, sized = split(jnp.asarray(found.x))
    spent = f"{found.nit} it, {found.nfev} f"

    return RouteAnswer(np.asarray(sized), float(found.fun), spent, elapsed)


def report_sizes(report: Report, nested: RouteAnswer, direct: RouteAnswer) -> float:
    """
    Both routes' sizes at fixed force densities, and the worst gap.
    """
    scale = float(np.max(np.abs(nested.diameters)))
    gaps = np.abs(direct.diameters - nested.diameters) / scale
    rows = [
        (f"{index}", nested.diameters[index], direct.diameters[index], gaps[index])
        for index in range(NUM_EDGES)
    ]

    report.write_heading("Fixed shape: the solver's sizes against the optimizer's")
    report.write_table(SIZES_COLUMNS, rows)

    return float(np.max(gaps))


def report_routes(report: Report, labeled: list[tuple[str, RouteAnswer]]) -> None:
    """
    Every route's mass and cost, one row each.
    """
    rows = [
        (label, answer.mass, answer.evaluations, answer.elapsed)
        for label, answer in labeled
    ]

    report.write_heading("What each route found, and what it spent")
    report.write_table(ROUTE_COLUMNS, rows)


def main(verbose: bool = True) -> None:
    """
    Run both formulations at a fixed shape, then jointly over the shape too.
    """
    report = Report(verbose)
    report.write_line(TITLE)

    problem = arch_problem()
    entries = (
        ("members", f"{NUM_EDGES}"),
        ("d/t", f"{RATIO:.1f}"),
        ("solver", "SLSQP, analytic Jacobians from the hand partials"),
        ("fallback", "penalized_mass through minimize_bounded, stated not run"),
    )
    report.write_heading("The arch, and the two formulations")
    report.write_entries(entries)

    nested_answer = nested_fixed(problem)
    direct_answer, state = simultaneous_fixed(problem)
    worst_gap = report_sizes(report, nested_answer, direct_answer)
    worst_multiplier = kkt_bridge(report, direct_answer, state)

    slackness = np.asarray(state.slack(jnp.asarray(direct_answer.diameters)))
    worst_violation = float(np.max(-slackness))

    joint_nested = nested_joint(problem)
    joint_direct = simultaneous_joint(problem)

    labeled = [
        ("nested, fixed shape", nested_answer),
        ("simultaneous, fixed shape", direct_answer),
        ("nested descent over the shape", joint_nested),
        ("simultaneous over shape and sizes", joint_direct),
    ]
    report_routes(report, labeled)
    report.write_note(
        "The joint rows optimize the shape too, under the same bounds; the "
        "nested one differentiates at frozen seed sections and settles after, "
        "the simultaneous one carries the size-shape coupling in every step."
    )

    checks = (
        ToleranceCheck("route agreement on the sizes", worst_gap, TOLERANCE_AGREEMENT),
        ToleranceCheck("constraint violation", worst_violation, TOLERANCE_FEASIBLE),
        ToleranceCheck(
            "redistribution share", worst_multiplier, TOLERANCE_REDISTRIBUTION
        ),
    )
    report.write_heading("Summary")
    report.write_checks(checks)
    report.write_verdict(verify_checks(checks))


if __name__ == "__main__":
    jnp.set_printoptions(precision=12)
    main()
