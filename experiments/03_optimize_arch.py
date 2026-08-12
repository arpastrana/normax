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
Searching an arch for the lightest shape EN 1993-1-1 will accept.

Twenty circular hollow members, three load cases, and one variable per member:
the force density that decides the shape. The sizes are never searched over —
they are solved for inside the objective at every iterate, so what the optimizer
sees is an unconstrained scalar and what it gets back is a gradient that has
crossed form finding, a frame analysis and the design standard.

Six things are reported.

    cases       what each load case demands, and which of them governs where
    smoothing   what the envelope over cases costs at each sharpness
    sweep       the mass along the uniform force densities, and its gradient
    descent     the same mass with one variable per member
    compliance  each design read back against the standard, unsmoothed
    floor       the same descent again with a shortest member it may not pass

**The sweep is the validation and the descent is the result.** A single
variable can only scale the funicular shape, and along that family the mass has
an interior minimum, which is the tension the whole project is built around:
raising the arch drops the thrust but lengthens the members, and the buckling
term eventually wins. The gradient is checked against that curve at every
sample. Twenty variables are not restricted to that family and end below its
minimum.

**The descent is bounded by its box rather than by the physics, and that is a
finding rather than a defect.** Nothing in a member check penalises a shape for
being a bad arch: every member is fully stressed and adequate, whatever the form
does between them. What would penalise it is global stability, and that is
deliberately outside the pipeline — reported here beside the answer, never
inside it.

**Left to itself the search collapses members rather than improving the form.** A
vanishing member is free, its mass being an area times a length, and it is also
unbucklable, its buckling length being that same length; so the standard has no
objection to one and the objective actively rewards it. The second descent adds
a floor on the shortest member and the two are reported side by side, which is
the only way to see how much of the first one is design and how much is
collapse.

**Holding the plan instead would not do it.** That bounds every member by its own
projection, but it leaves horizontal equilibrium unimposed, and on an evenly
spaced plan that equilibrium admits only a uniform force density — so the
funicular part of a held plan is the one-variable sweep above. See
`normax.formfinding.positions_vertical` and the roadmap's note on thrust network
analysis, which is the construction that fixes a plan and keeps equilibrium.

Run with `uv run --group pipeline python experiments/03_optimize_arch.py`.
"""

from pathlib import Path
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from jaxtyping import Float

from normax.analysis.smax import prepare_model
from normax.ec3.actions import MemberActions
from normax.ec3.material import SteelGrade
from normax.ec3.section import TubeCatalogue
from normax.formfinding import equilibrium_graph
from normax.formfinding import equilibrium_state
from normax.optimization import Trajectory
from normax.optimization import annealing_schedule
from normax.optimization import optimize_annealed
from normax.optimization import penalized_mass
from normax.optimization import shortest_member
from normax.pipeline import Design
from normax.pipeline import Envelope
from normax.pipeline import ProblemSetup
from normax.pipeline import Unsmoothed
from normax.pipeline import design_envelope
from normax.pipeline import frame_stability
from normax.pipeline import governing_load_case
from normax.pipeline import unsmoothed_design
from normax.reporting import ColumnSpec
from normax.reporting import ReportWriter
from normax.reporting import ToleranceCheck
from normax.reporting import checks_passed
from normax.structures import arch_2d
from normax.structures import crown_node
from normax.structures import loads_half_span
from normax.structures import loads_point
from normax.structures import loads_uniform
from normax.visualization import Descent
from normax.visualization import Form
from normax.visualization import GradientCheck
from normax.visualization import MassSweep
from normax.visualization import figure_load_cases
from normax.visualization import figure_optimization

# A 10 m arch rising 3 m over twenty members, carrying 180 kN. Units are
# millimetres and newtons.
SPAN = 10_000.0
RISE = 3_000.0
TOTAL_LOAD = 180_000.0
NUM_EDGES = 20

# The arch lies in the XZ plane, so it has no thickness along Y.
NORMAL = 1

# The diameter the frame is analysed with before the check has spoken.
SEED = 100.0

# Every case carries the same total, so the three differ in where the load sits
# and in nothing else. A case with a smaller total would be lighter for a reason
# that has nothing to do with the shape.
HALF_FACTOR = 0.5
POINT_SHARE = 0.25
CASE_NAMES = ("LC1 uniform", "LC2 half span", "LC3 crown point")

# Sharpnesses to report the cost of the smoothing at, and the schedule the
# descent anneals over. Arrays rather than floats so that a sharpness is traced
# and changing it does not compile the objective again.
SHARPNESSES = jnp.asarray([10.0, 25.0, 50.0, 100.0, 250.0, 500.0])
BETA_START = 10.0
BETA_STOP = jnp.asarray(500.0)
ROUNDS = 5

# Enough that neither descent stops on its limit. The floored one needs far more
# than the unconstrained one: a penalty that bites makes every step smaller, and
# a run that halts at its cap reports a bound rather than an optimum.
ITERATIONS = 60

# The force densities may move a decade either side of the funicular value. The
# bound exists to keep them away from zero, where the force density system is
# singular, and not to express a design intent.
DECADES = 10.0

# Multiples of the funicular force density the sweep samples.
SCALES = np.linspace(0.4, 2.4, 21)

# Relative steps the central difference is swept over, and the one it plateaus
# at. Not P3's 1e-5: three load cases make the mass four times larger and the
# arithmetic behind it three times longer, so cancellation dominates a decade
# sooner and the trough moves. The experiment prints the sweep rather than
# asserting the choice.
STEPS = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
STEP = 1e-4

# The floor at that step, measured at 4.4e-8 over the whole sweep, so this is
# the noise of the reference rather than an error in the gradient. Pinned with
# headroom: a gradient that was actually wrong would miss by a thousand times
# more than the gap between these two numbers.
TOLERANCE_GRADIENT = 2e-7

# Passes of the staggered analysis and check, to measure what one pass costs.
PASSES = 6

# Shortest member the design may have, as a fraction of the nominal bay. Nothing
# in a member check objects to a vanishing member and two things reward one, so
# without a floor the search collapses edges instead of improving the form.
FLOOR = 0.6 * SPAN / NUM_EDGES
FLOOR_BETA = 50.0
FLOOR_WEIGHT = 50.0

FIGURES = Path(__file__).resolve().parent.parent / "figures"

STEEL = SteelGrade()
SECTION_CLASS = 3
CATALOGUE = TubeCatalogue.at_class_limit(STEEL.f_y, SECTION_CLASS)

# The reads the reports make, compiled. Left eager each one costs an XLA
# compilation per primitive, which is most of what reporting a design costs.
unsmoothed_compiled = eqx.filter_jit(unsmoothed_design)
governing_compiled = eqx.filter_jit(governing_load_case)
stability_compiled = eqx.filter_jit(frame_stability)
shortest_compiled = eqx.filter_jit(shortest_member)


class ArchProblem(NamedTuple):
    """
    The prepared arch, the cases it answers to, and the funicular force density.

    Attributes
    ----------
    problem :
        The prepared problem the three stages read, built once on the host.
    load_cases :
        Nodal loads of every load case, stacked along the leading axis.
    q :
        Force densities that reach the target rise under the funicular case.
    """

    problem: ProblemSetup
    load_cases: Float[Array, "cases nodes 3"]
    q: Float[Array, "edges"]

    @property
    def bounds(self) -> tuple[float, float]:
        """
        The box the force densities may move in, a decade either side.
        """
        box = (float(self.q[0]) * DECADES, float(self.q[0]) / DECADES)

        return box


class SweepReport(NamedTuple):
    """
    The mass along the uniform force densities, and the gradient against it.

    Attributes
    ----------
    masses :
        Mass at every sampled multiple of the funicular force density.
    exact :
        Directional derivative of the mass at each sample.
    numeric :
        Central difference of the same quantity.
    best :
        Index of the lightest sample.
    worst :
        Worst scaled disagreement between the two derivatives.
    """

    masses: Float[np.ndarray, "samples"]
    exact: Float[np.ndarray, "samples"]
    numeric: Float[np.ndarray, "samples"]
    best: int
    worst: float

    @property
    def interior(self) -> bool:
        """
        Whether the lightest sample is interior rather than on an end.
        """
        return 0 < self.best < len(self.masses) - 1


class DescentRun(NamedTuple):
    """
    One descent, and the box it was run in.

    Attributes
    ----------
    label :
        What distinguishes this descent from the other.
    walked :
        Force densities, sharpnesses and objective values, step by step.
    floor :
        Shortest member the design may have, or zero where none was imposed.
    """

    label: str
    walked: Trajectory
    floor: float


class FinalReport(NamedTuple):
    """
    The design a descent arrived at, read back against the standard.

    Attributes
    ----------
    envelope :
        The enveloped design at the force densities the descent ended on.
    sized :
        The same design sized against the true largest of the load cases.
    decided :
        Index of the load case that governs each member.
    stagger :
        Fraction of the mass one staggered re-analysis moves.
    alpha_cr :
        Weakest critical load factor over the load cases.
    lengths :
        Length of every member of the finished design.
    """

    envelope: Envelope
    sized: Unsmoothed
    decided: Float[np.ndarray, "edges"]
    stagger: float
    alpha_cr: float
    lengths: Float[np.ndarray, "edges"]


def arch_problem() -> ArchProblem:
    """
    The arch, both prepared topologies, and the `q` that reaches the rise.

    Neither the form-finding connectivity nor the analysis model depends on a
    force density, so both are built here and passed to everything below. The
    analysis model is the expensive one and the reason the objective can be
    compiled at all: preparing it reads support flags in Python, which a tracer
    cannot follow.
    """
    structure = arch_2d(
        num_edges=NUM_EDGES,
        span=SPAN,
        rise=RISE,
        load=TOTAL_LOAD / (NUM_EDGES - 1),
    )
    graph_fdm = equilibrium_graph(structure)
    model = prepare_model(structure, STEEL, CATALOGUE, normal=NORMAL)

    trial = jnp.full(NUM_EDGES, -1.0)
    state = equilibrium_state(trial, structure, graph_fdm)
    reached = jnp.max(state.xyz[:, 2])

    problem = ProblemSetup(structure, graph_fdm, model, STEEL, CATALOGUE)
    load_cases = build_load_cases(structure)
    setup = ArchProblem(problem, load_cases, trial * reached / RISE)

    return setup


def build_load_cases(structure) -> Float[Array, "cases nodes 3"]:
    """
    Three cases of equal total: funicular, half span, and a crown point load.
    """
    spread = TOTAL_LOAD / (NUM_EDGES - 1)

    uniform = loads_uniform(structure, spread)

    half = loads_half_span(structure, spread, factor=HALF_FACTOR)
    half = half * (TOTAL_LOAD / abs(float(jnp.sum(half[:, 2]))))

    crown = crown_node(structure)
    spread_share = loads_uniform(structure, spread * (1.0 - POINT_SHARE))
    at_crown = loads_point(structure, TOTAL_LOAD * POINT_SHARE, node=crown)
    point = spread_share + at_crown

    cases = [uniform, half, point]

    return jnp.stack(cases)


@eqx.filter_jit
def build(setup: ArchProblem, q, beta, diameters=None) -> Envelope:
    """
    The enveloped design at one set of force densities.

    Compiled, which is what makes the sweeps below affordable as well as the
    descents: every caller here passes the same prepared topologies, so one trace
    serves all of them. The sharpness is an argument rather than a constant so
    that annealing it does not compile again.
    """
    seed = jnp.full(NUM_EDGES, SEED) if diameters is None else diameters
    envelope = design_envelope(
        q,
        seed,
        setup.problem,
        setup.load_cases,
        beta=beta,
        section_class=SECTION_CLASS,
    )

    return envelope


def design_under(envelope: Envelope, sized: Unsmoothed, load_case: int = 0) -> Design:
    """
    A finished design: unsmoothed sizes, actions of one load case.
    """
    actions = MemberActions(
        envelope.axial_force[load_case],
        envelope.moment_major[load_case],
        envelope.moment_minor[load_case],
        envelope.moment_factor_major[load_case],
        envelope.moment_factor_minor[load_case],
    )

    design = Design(
        envelope.xyz,
        envelope.lengths,
        actions,
        envelope.buckling_length,
        sized.diameters,
        sized.utilization[load_case],
        sized.mass,
    )

    return design


def report_load_cases(report: ReportWriter, setup: ArchProblem) -> None:
    """
    What each case demands of each member at the starting shape.
    """
    result = build(setup, setup.q, BETA_STOP)
    decided = np.asarray(governing_compiled(result))

    case_columns = [
        ColumnSpec(f"d {name.split()[0]} [mm]", ".2f") for name in CASE_NAMES
    ]
    columns = (
        ColumnSpec("member"),
        *case_columns,
        ColumnSpec("governs", align="<"),
    )
    rows = []
    for member in range(NUM_EDGES):
        sizes = [
            float(result.required[load_case, member])
            for load_case in range(len(CASE_NAMES))
        ]
        rows.append((member, *sizes, CASE_NAMES[decided[member]]))

    entries = [
        (name, f"governs {int(np.sum(decided == index))} of {NUM_EDGES}")
        for index, name in enumerate(CASE_NAMES)
    ]

    report.write_line("The three load cases, each carrying 180 kN")
    report.write_table(columns, rows)
    report.write_entries(entries)


def report_smoothing(report: ReportWriter, setup: ArchProblem) -> None:
    """
    What the envelope gives away at each sharpness, against the true largest.
    """
    columns = (
        ColumnSpec("beta", ".0f"),
        ColumnSpec("mass [t]", ".9f"),
        ColumnSpec("excess", ".4%"),
        ColumnSpec("bound", ".4%"),
        ColumnSpec("max utilization", ".15f"),
    )
    rows = []
    for beta in SHARPNESSES:
        result = build(setup, setup.q, beta)
        exact = unsmoothed_compiled(result, setup.problem, section_class=SECTION_CLASS)
        smoothed = float(result.mass)
        excess = smoothed / float(exact.mass) - 1.0
        cases = setup.load_cases.shape[0]
        bound = float(cases ** (2.0 / float(beta))) - 1.0
        utilization = float(jnp.max(exact.utilization))
        rows.append((float(beta), smoothed, excess, bound, utilization))

    report.write_heading("What the envelope costs, and the bound on it")
    report.write_table(columns, rows)


def report_sweep(report: ReportWriter, setup: ArchProblem) -> SweepReport:
    """
    The mass along the uniform force densities, and the gradient against it.
    """

    def objective(scaled):
        envelope = build(setup, scaled, BETA_STOP)

        return envelope.mass

    slopes = [
        float(jnp.sum(jax.grad(objective)(setup.q * scale) * setup.q))
        for scale in SCALES
    ]
    largest = max(abs(slope) for slope in slopes)

    def difference_at(scale: float, relative: float) -> float:
        step = abs(scale) * relative
        forward = float(objective(setup.q * (scale + step)))
        backward = float(objective(setup.q * (scale - step)))

        return (forward - backward) / (2.0 * step)

    sampled = (0, len(SCALES) // 2, len(SCALES) - 1)
    step_columns = (
        ColumnSpec("relative step", ".0e"),
        ColumnSpec("worst scaled error", ".2e"),
    )
    step_rows = []
    for relative in STEPS:
        errors = []
        for index in sampled:
            quotient = difference_at(SCALES[index], relative)
            errors.append(abs(slopes[index] - quotient) / largest)
        step_rows.append((relative, max(errors)))

    trough = f"{STEP:.0e}, so that is where the check is made"
    entries = (("the trough is at", trough),)

    report.write_heading("The central difference plateaus before it is trusted")
    report.write_table(step_columns, step_rows)
    report.write_entries(entries)

    masses = []
    numeric = []
    rows = []
    for scale, slope in zip(SCALES, slopes):
        mass = float(objective(setup.q * scale))
        quotient = difference_at(scale, STEP)
        masses.append(mass)
        numeric.append(quotient)
        scaled = abs(slope - quotient) / largest
        rows.append((scale, float(setup.q[0]) * scale, mass, slope, scaled))

    columns = (
        ColumnSpec("scale", ".2f"),
        ColumnSpec("q [N/mm]", ".3f"),
        ColumnSpec("mass [t]", ".9f"),
        ColumnSpec("d/dk", ".6e"),
        ColumnSpec("scaled", ".2e"),
    )

    report.write_heading("The mass along the uniform force densities")
    report.write_table(columns, rows)

    best = int(np.argmin(masses))
    worst = max(abs(a - b) / largest for a, b in zip(slopes, numeric))
    interior = 0 < best < len(SCALES) - 1
    entries = (
        ("best uniform scale", f"{SCALES[best]:.2f}, mass {masses[best]:.9f} t"),
        ("interior minimum", f"{interior}"),
        ("worst scaled gradient error", f"{worst:.2e}"),
    )

    report.write_entries(entries)

    sweep = SweepReport(
        np.asarray(masses), np.asarray(slopes), np.asarray(numeric), best, worst
    )

    return sweep


def report_descent(
    report: ReportWriter, setup: ArchProblem, floor: float
) -> DescentRun:
    """
    The same mass with one force density per member, annealed.
    """
    schedule = annealing_schedule(BETA_START, float(BETA_STOP), ROUNDS)
    label = f"a {floor:.0f} mm floor" if floor else "no length floor"

    def objective(x, beta):
        result = build(setup, x, beta)
        if not floor:
            return result.mass

        penalized = penalized_mass(
            result.mass,
            result.lengths,
            floor,
            beta=FLOOR_BETA,
            weight=FLOOR_WEIGHT,
        )

        return penalized

    walked = optimize_annealed(
        objective, setup.q, schedule, bounds=setup.bounds, iterations=ITERATIONS
    )

    columns = (
        ColumnSpec("step"),
        ColumnSpec("beta", ".1f"),
        ColumnSpec("objective [t]", ".9f"),
    )
    rows = [
        (step, float(beta), float(total))
        for step, (beta, total) in enumerate(zip(walked.beta, walked.mass))
    ]
    heading = f"Descending with one variable per member, {label}, bounds {setup.bounds}"

    report.write_heading(heading)
    report.write_table(columns, rows)

    run = DescentRun(label, walked, floor)

    return run


def report_stagger(
    report: ReportWriter,
    setup: ArchProblem,
    q: Float[Array, "edges"],
    result: Envelope,
) -> float:
    """
    What the objective's single analysis pass costs, by repeating it.

    One analysis, then size, then re-analyse at those sizes. The first pass is
    what the objective uses; the rest measure how much that one-shot costs.
    """
    diameters = jnp.full(NUM_EDGES, SEED)
    relaxed = result
    rows = []

    for step in range(PASSES):
        relaxed = build(setup, q, BETA_STOP, diameters)
        shift = jnp.abs(relaxed.diameters - diameters) / relaxed.diameters
        rows.append((step, float(jnp.max(shift)), float(relaxed.mass)))
        diameters = relaxed.diameters

    stagger = abs(float(result.mass) - float(relaxed.mass)) / float(relaxed.mass)
    columns = (
        ColumnSpec("pass"),
        ColumnSpec("move", ".3e"),
        ColumnSpec("mass [t]", ".9f"),
    )
    entries = (("one pass costs", f"{stagger:.3%} of the mass"),)

    report.write_heading("Staggered re-analysis")
    report.write_table(columns, rows)
    report.write_entries(entries)

    return stagger


def report_final(
    report: ReportWriter,
    setup: ArchProblem,
    run: DescentRun,
) -> FinalReport:
    """
    The design the descent arrived at, read back against the standard.
    """
    q = run.walked.q[-1]
    result = build(setup, q, BETA_STOP)
    sized = unsmoothed_compiled(result, setup.problem, section_class=SECTION_CLASS)
    decided = np.asarray(governing_compiled(result))
    lengths = np.asarray(result.lengths)

    lower, upper = setup.bounds
    at_lower = int(jnp.sum(jnp.abs(q - lower) < 1e-6))
    at_upper = int(jnp.sum(jnp.abs(q - upper) < 1e-6))
    shortest, longest = lengths.min(), lengths.max()

    thinnest = float(jnp.min(sized.diameters))
    thickest = float(jnp.max(sized.diameters))
    smoothed = float(shortest_compiled(result.lengths, FLOOR_BETA))
    stubs = int(np.sum(lengths < FLOOR))
    governing = [
        (name, f"governs {int(np.sum(decided == index))} of {NUM_EDGES}")
        for index, name in enumerate(CASE_NAMES)
    ]
    entries = (
        ("mass, enveloped", f"{float(result.mass):.9f} t"),
        ("mass, unsmoothed", f"{float(sized.mass):.9f} t"),
        ("worst utilization", f"{float(jnp.max(sized.utilization)):.15f}"),
        ("diameters", f"{thinnest:.2f} .. {thickest:.2f} mm"),
        ("rise", f"{float(jnp.max(result.xyz[:, 2])):.1f} mm"),
        ("developed length", f"{float(jnp.sum(result.lengths)):.1f} mm"),
        (
            "member length",
            f"{shortest:.1f} .. {longest:.1f} mm (ratio {longest / shortest:.1f})",
        ),
        ("shortest, smoothed", f"{smoothed:.1f} mm"),
        (f"members under {FLOOR:.0f} mm", f"{stubs} of {NUM_EDGES}"),
        ("force densities at bounds", f"{at_lower} lower, {at_upper} upper"),
        *governing,
    )

    report.write_heading(f"The design the descent arrived at, {run.label}")
    report.write_entries(entries)

    stagger = report_stagger(report, setup, q, result)

    # A critical load factor belongs to a load case, and the case the shape was
    # found under is not the one that sized it, so quoting only that one would
    # flatter the design.
    frame = design_under(result, sized)
    factors = []
    for name, loads in zip(CASE_NAMES, setup.load_cases):
        checked = stability_compiled(frame, setup.problem, num_modes=1, loads=loads)
        verdict = "adequate" if bool(checked.adequate) else "inadequate"
        factors.append((name, float(checked.factors[0]), verdict))

    columns = (
        ColumnSpec("load case", align="<"),
        ColumnSpec("alpha_cr", ".4f"),
        ColumnSpec("verdict", align="<"),
    )

    report.write_heading("Critical load factor by case")
    report.write_table(columns, factors)

    weakest = min(factor for _, factor, _ in factors)
    final = FinalReport(result, sized, decided, stagger, weakest, lengths)

    return final


def report_floor(
    report: ReportWriter,
    loose: FinalReport,
    held: FinalReport,
    funicular_mass: float,
) -> None:
    """
    What a length floor costs, and what it buys, side by side.
    """
    loose_ratio = loose.lengths.max() / loose.lengths.min()
    held_ratio = held.lengths.max() / held.lengths.min()
    loose_stubs = float(np.sum(loose.lengths < FLOOR))
    held_stubs = float(np.sum(held.lengths < FLOOR))
    rows = (
        ("mass [t]", float(loose.sized.mass), float(held.sized.mass)),
        ("shortest member [mm]", loose.lengths.min(), held.lengths.min()),
        ("longest member [mm]", loose.lengths.max(), held.lengths.max()),
        ("length ratio", loose_ratio, held_ratio),
        ("members under floor", loose_stubs, held_stubs),
        ("alpha_cr, weakest", loose.alpha_cr, held.alpha_cr),
    )
    columns = (
        ColumnSpec("quantity", align="<"),
        ColumnSpec("unconstrained", ".4f"),
        ColumnSpec("floored", ".4f"),
    )
    kept = 1.0 - float(held.sized.mass) / funicular_mass
    entries = (
        ("the floored design", f"is {kept:.1%} lighter than the funicular arch"),
    )

    report.write_heading("What a length floor costs, and what it buys")
    report.write_table(columns, rows)
    report.write_entries(entries)


def write_figures(
    setup: ArchProblem,
    sweep: SweepReport,
    runs: tuple[DescentRun, DescentRun],
    finals: tuple[FinalReport, FinalReport],
) -> None:
    """
    The optimization curves, and the three forms compared member by member.
    """
    loose, held = finals
    funicular = int(np.argmin(np.abs(SCALES - 1.0)))

    FIGURES.mkdir(exist_ok=True)
    descents = []
    for run in runs:
        masses = np.asarray(run.walked.mass)
        sharpnesses = np.asarray(run.walked.beta)
        descents.append(Descent(run.label, masses, sharpnesses))
    swept = MassSweep(SCALES, sweep.masses, funicular)
    checked = GradientCheck(sweep.exact, sweep.numeric)
    optimization = figure_optimization(swept, checked, descents)
    optimization.savefig(FIGURES / "03_optimization.png", dpi=200)

    single = build(setup, setup.q * SCALES[sweep.best], BETA_STOP)
    single_sized = unsmoothed_compiled(
        single, setup.problem, section_class=SECTION_CLASS
    )
    single_decided = np.asarray(governing_compiled(single))
    title_single = f"Best single $q$, {sweep.masses[sweep.best]:.4f} t"
    title_loose = f"Per member, no floor, {float(loose.sized.mass):.4f} t"
    title_held = f"Per member, {FLOOR:.0f} mm floor, {float(held.sized.mass):.4f} t"
    forms = (
        Form(title_single, single.xyz, single_sized.diameters, single_decided),
        Form(title_loose, loose.envelope.xyz, loose.sized.diameters, loose.decided),
        Form(title_held, held.envelope.xyz, held.sized.diameters, held.decided),
    )
    cases = figure_load_cases(setup.problem.structure.edges, forms, CASE_NAMES)
    cases.savefig(FIGURES / "03_load_cases.png", dpi=200)


def main(verbose: bool = True) -> None:
    """
    Sweep the funicular family, then descend on one variable per member.
    """
    report = ReportWriter(verbose)
    setup = arch_problem()

    report_load_cases(report, setup)
    report_smoothing(report, setup)
    sweep = report_sweep(report, setup)

    loose_run = report_descent(report, setup, floor=0.0)
    loose = report_final(report, setup, loose_run)

    held_run = report_descent(report, setup, floor=FLOOR)
    held = report_final(report, setup, held_run)

    # The funicular design, which is where the descent starts and the only
    # reference either reduction means anything against.
    funicular = int(np.argmin(np.abs(SCALES - 1.0)))
    reduction = 1.0 - float(loose.sized.mass) / sweep.masses[funicular]
    against_best = 1.0 - float(loose.sized.mass) / sweep.masses[sweep.best]

    runs = (loose_run, held_run)
    finals = (loose, held)
    write_figures(setup, sweep, runs, finals)

    worst_utilization = float(jnp.max(loose.sized.utilization))
    entries = (
        ("interior minimum in the uniform sweep", f"{sweep.interior}"),
        (
            "worst scaled gradient error",
            f"{sweep.worst:.2e} ({TOLERANCE_GRADIENT:.0e})",
        ),
        ("worst utilization of the final design", f"{worst_utilization:.12f}"),
        ("lighter than the funicular arch", f"{reduction:.1%}"),
        ("lighter than the best uniform arch", f"{against_best:.1%}"),
    )

    report.write_heading("Summary")
    report.write_entries(entries)

    report_floor(report, loose, held, sweep.masses[funicular])

    not_the_answer = (
        ("one staggered pass costs", f"{loose.stagger:.2%} of the mass"),
        ("critical load factor of the design", f"{loose.alpha_cr:.4f}"),
    )

    report.write_heading("What the answer is not")
    report.write_entries(not_the_answer)
    report.write_note(
        """
        The descent spent the stability margin the starting arch had, and nothing
        in a member check was ever going to stop it. Global stability is outside
        the pipeline by design; this is what that costs.
        """
    )

    checks = (ToleranceCheck("scaled gradient error", sweep.worst, TOLERANCE_GRADIENT),)
    adequate = worst_utilization < 1.0 + 1e-9
    beats_uniform = float(loose.sized.mass) < sweep.masses[sweep.best]
    passed = checks_passed(checks) and sweep.interior and adequate and beats_uniform

    report.write_verdict(passed)


if __name__ == "__main__":
    main()
