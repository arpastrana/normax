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
deliberately outside the pipeline — verifying a finished design against it is
future work.

**Left to itself the search collapses members rather than improving the form.** A
vanishing member is free, its mass being an area times a length, and it is also
unbucklable, its buckling length being that same length; so the standard has no
objection to one and the objective actively rewards it. The second descent adds
a floor on the shortest member and the two are reported side by side, which is
the only way to see how much of the first one is design and how much is
collapse.

**The two asymmetric cases are a mirrored pair, and that is what makes the
answer readable.** A single half-span case leaves one half of the arch light and
biases the search towards it, so an asymmetric optimum says nothing about
whether the asymmetry is structural or an artefact of the loading. Loading each
half in turn makes the whole set symmetric about midspan, and a symmetric
problem started from a symmetric shape should stay symmetric at every iterate.
The departure from that is measured rather than assumed: `mirror_gap` reports
how far the design differs from its own reflection.

**Holding the plan instead would not do it.** That bounds every member by its own
projection, but it leaves horizontal equilibrium unimposed, and on an evenly
spaced plan that equilibrium admits only a uniform force density — so the
funicular part of a held plan is the one-variable sweep above. See
`normax.form_finding.positions_vertical` and the roadmap's note on thrust network
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

from normax.analysis import SmaxAnalyzer
from normax.design import Design
from normax.design import DesignParameters
from normax.design import StructuralDesignPipeline
from normax.design import compute_mass
from normax.design import design_envelope
from normax.design import governing_load_case
from normax.form_finding import FdmFormFinder
from normax.form_finding import build_equilibrium_graph
from normax.form_finding import solve_equilibrium
from normax.loads import LoadCases
from normax.loads import assemble_load_cases
from normax.loads import create_loads_half_span
from normax.loads import create_loads_uniform
from normax.materials import Steel355
from normax.optimization import Trajectory
from normax.optimization import annealing_schedule
from normax.optimization import optimize_annealed
from normax.optimization import penalized_mass
from normax.optimization import shortest_member
from normax.reporting import Report
from normax.reporting import ReportColumn
from normax.reporting import ToleranceCheck
from normax.reporting import verify_checks
from normax.sizing import Ec3Sizer
from normax.sizing import build_section_family
from normax.structures import Structure
from normax.structures import build_arch_2d
from normax.visualization import Descent
from normax.visualization import Form
from normax.visualization import GradientCheck
from normax.visualization import MassSweep
from normax.visualization import figure_load_cases
from normax.visualization import figure_optimization

# A 10 m arch rising 3 m over twenty members, carrying 180 kN. Units are
# millimeters and newtons.
SPAN = 10_000.0
RISE = 3_000.0
TOTAL_LOAD = 180_000.0
NUM_EDGES = 20

# The diameter the frame is analyzed with before the check has spoken.
SEED = 100.0

# Every case carries the same total, so they differ in where the load sits and
# in nothing else. A case with a smaller total would be lighter for a reason
# that has nothing to do with the shape.
HALF_FACTOR = 0.5
CASE_NAMES = (
    "LC1 uniform",
    "LC2 half span",
    "LC3 half span mirrored",
)

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
# at. Not P3's 1e-5: several load cases make the mass larger and the
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

FIGURES = Path(__file__).resolve().parents[2] / "figures"

GRADE = Steel355()
SECTION_CLASS = 3

# The reads the reports make, compiled. Left eager each one costs an XLA
# compilation per primitive, which is most of what reporting a design costs.
governing_compiled = eqx.filter_jit(governing_load_case)
shortest_compiled = eqx.filter_jit(shortest_member)


class ArchProblem(NamedTuple):
    """
    The prepared arch, the cases it answers to, and the funicular force density.

    Attributes
    ----------
    structure :
        The arch the blocks were built against, read for its connectivity alone.
    pipeline :
        The three blocks, each already bound to the arch on the host.
    loads :
        The case the shape answers to, and the cases it is checked against.
    q :
        Force densities that reach the target rise under the funicular case.
    """

    structure: Structure
    pipeline: StructuralDesignPipeline
    loads: LoadCases
    q: Float[Array, "edges"]

    @property
    def analyzer(self) -> SmaxAnalyzer:
        """
        The analysis block, which the stability check reads its assembly from.
        """
        return self.pipeline.analyzer

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
    lengths :
        Length of every member of the finished design.
    mirror :
        How far the diameters depart from their own reflection.
    """

    envelope: Design
    sized: Design
    decided: Float[np.ndarray, "edges"]
    stagger: float
    lengths: Float[np.ndarray, "edges"]
    mirror: float


def mirror_gap(values: Float[np.ndarray, "edges"]) -> float:
    """
    How far a per-member quantity departs from its own reflection.

    The arch is a chain built left to right, so member `k` mirrors member
    `-1 - k` and reversing the array reflects the design about midspan. Scaled
    by the largest entry, so the number reads the same whatever the quantity is.
    """
    values = np.asarray(values)
    scale = float(np.max(np.abs(values)))
    departure = float(np.max(np.abs(values - values[::-1])))

    return departure / scale if scale > 0.0 else 0.0


def arch_problem() -> ArchProblem:
    """
    The arch, both prepared topologies, and the `q` that reaches the rise.

    Neither the form-finding connectivity nor the analysis model depends on a
    force density, so both are built here and passed to everything below. The
    analysis model is the expensive one and the reason the objective can be
    compiled at all: preparing it reads support flags in Python, which a tracer
    cannot follow.
    """
    structure = build_arch_2d(num_edges=NUM_EDGES, span=SPAN, rise=RISE)
    graph_fdm = build_equilibrium_graph(structure)
    loads = build_load_cases(structure)

    trial = jnp.full(NUM_EDGES, -1.0)
    state = solve_equilibrium(
        trial,
        structure.nodes[graph_fdm.indices_fixed],
        graph_fdm,
        loads.formfinding,
    )
    reached = jnp.max(state.xyz[:, 2])

    # The analysis is configured with one tube, drawn from the sizer's family;
    # the check chooses within that family.
    family = build_section_family(GRADE, SECTION_CLASS)
    sizer = Ec3Sizer(structure, family)
    blocks = StructuralDesignPipeline(
        FdmFormFinder(structure),
        SmaxAnalyzer(structure, family(SEED)),
        sizer,
    )
    funicular = trial * reached / RISE
    setup = ArchProblem(structure, blocks, loads, funicular)

    return setup


def build_load_cases(structure: Structure) -> LoadCases:
    """
    Three cases of equal total: funicular, half span, and its mirror.
    """
    spread = TOTAL_LOAD / (NUM_EDGES - 1)

    uniform = create_loads_uniform(structure, spread)

    half = create_loads_half_span(structure, spread, factor=HALF_FACTOR)
    half = half * (TOTAL_LOAD / abs(float(jnp.sum(half[:, 2]))))

    mirrored = create_loads_half_span(
        structure,
        spread,
        factor=HALF_FACTOR,
        mirrored=True,
    )
    mirrored = mirrored * (TOTAL_LOAD / abs(float(jnp.sum(mirrored[:, 2]))))

    cases = [uniform, half, mirrored]

    return assemble_load_cases(cases)


@eqx.filter_jit
def build_by_case(setup: ArchProblem, q, diameters=None) -> Design:
    """
    The design every load case demands on its own, before they are reconciled.

    What the pipeline itself returns. Only the reports that read a case apart
    from the others need it — which case governs a member is an `argmax` over
    this axis, and the axis is gone once the envelope has collapsed it.
    """
    seed = jnp.full(NUM_EDGES, SEED) if diameters is None else diameters
    params = DesignParameters(q, seed)
    design = setup.pipeline(params, setup.loads)

    return design


@eqx.filter_jit
def build(setup: ArchProblem, q, beta, diameters=None) -> Design:
    """
    The enveloped design at one set of force densities.

    Compiled, which is what makes the sweeps below affordable as well as the
    descents: every caller here passes the same prepared topologies, so one trace
    serves all of them. The sharpness is an argument rather than a constant so
    that annealing it does not compile again.
    """
    design = build_by_case(setup, q, diameters)
    envelope = design_envelope(design, beta)

    return envelope


def report_load_cases(report: Report, setup: ArchProblem) -> None:
    """
    What each case demands of each member at the starting shape.
    """
    by_case = build_by_case(setup, setup.q)
    required = by_case.sizes.sections.diameter
    decided = np.asarray(governing_compiled(required))

    case_columns = [
        ReportColumn(f"d {name.split()[0]} [mm]", ".2f") for name in CASE_NAMES
    ]
    columns = (
        ReportColumn("member"),
        *case_columns,
        ReportColumn("governs", align="<"),
    )
    rows = []
    for member in range(NUM_EDGES):
        sizes = [
            float(required[load_case, member]) for load_case in range(len(CASE_NAMES))
        ]
        rows.append((member, *sizes, CASE_NAMES[decided[member]]))

    entries = [
        (name, f"governs {int(np.sum(decided == index))} of {NUM_EDGES}")
        for index, name in enumerate(CASE_NAMES)
    ]

    report.write_line(f"The {len(CASE_NAMES)} load cases, each carrying 180 kN")
    report.write_table(columns, rows)
    report.write_entries(entries)


def report_smoothing(report: Report, setup: ArchProblem) -> None:
    """
    What the envelope gives away at each sharpness, against the true largest.
    """
    columns = (
        ReportColumn("beta", ".0f"),
        ReportColumn("mass [t]", ".9f"),
        ReportColumn("excess", ".4%"),
        ReportColumn("bound", ".4%"),
        ReportColumn("max utilization", ".15f"),
    )
    rows = []
    for beta in SHARPNESSES:
        result = build(setup, setup.q, beta)
        exact = build(setup, setup.q, None)
        smoothed = float(compute_mass(result))
        excess = smoothed / float(compute_mass(exact)) - 1.0
        cases = setup.loads.analysis.shape[0]
        bound = float(cases ** (2.0 / float(beta))) - 1.0
        utilization = float(jnp.max(exact.sizes.utilization))
        rows.append((float(beta), smoothed, excess, bound, utilization))

    report.write_heading("What the envelope costs, and the bound on it")
    report.write_table(columns, rows)


def report_sweep(report: Report, setup: ArchProblem) -> SweepReport:
    """
    The mass along the uniform force densities, and the gradient against it.
    """

    def objective(scaled):
        envelope = build(setup, scaled, BETA_STOP)

        return compute_mass(envelope)

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
        ReportColumn("relative step", ".0e"),
        ReportColumn("worst scaled error", ".2e"),
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
        ReportColumn("scale", ".2f"),
        ReportColumn("q [N/mm]", ".3f"),
        ReportColumn("mass [t]", ".9f"),
        ReportColumn("d/dk", ".6e"),
        ReportColumn("scaled", ".2e"),
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


def report_descent(report: Report, setup: ArchProblem, floor: float) -> DescentRun:
    """
    The same mass with one force density per member, annealed.
    """
    schedule = annealing_schedule(BETA_START, float(BETA_STOP), ROUNDS)
    label = f"a {floor:.0f} mm floor" if floor else "no length floor"

    def objective(x, beta):
        result = build(setup, x, beta)
        if not floor:
            return compute_mass(result)

        penalized = penalized_mass(
            compute_mass(result),
            result.shape.lengths,
            floor,
            beta=FLOOR_BETA,
            weight=FLOOR_WEIGHT,
        )

        return penalized

    searched = optimize_annealed(
        objective, setup.q, schedule, bounds=setup.bounds, iterations=ITERATIONS
    )
    walked = searched.trajectory

    columns = (
        ReportColumn("step"),
        ReportColumn("beta", ".1f"),
        ReportColumn("objective [t]", ".9f"),
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
    report: Report,
    setup: ArchProblem,
    q: Float[Array, "edges"],
    result: Design,
) -> float:
    """
    What the objective's single analysis pass costs, by repeating it.

    One analysis, then size, then re-analyze at those sizes. The first pass is
    what the objective uses; the rest measure how much that one-shot costs.
    """
    diameters = jnp.full(NUM_EDGES, SEED)
    relaxed = result
    rows = []

    for step in range(PASSES):
        relaxed = build(setup, q, BETA_STOP, diameters)
        required = relaxed.sizes.sections.diameter
        shift = jnp.abs(required - diameters) / required
        rows.append((step, float(jnp.max(shift)), float(compute_mass(relaxed))))
        diameters = required

    settled = float(compute_mass(relaxed))
    stagger = abs(float(compute_mass(result)) - settled) / settled
    columns = (
        ReportColumn("pass"),
        ReportColumn("move", ".3e"),
        ReportColumn("mass [t]", ".9f"),
    )
    entries = (("one pass costs", f"{stagger:.3%} of the mass"),)

    report.write_heading("Staggered re-analysis")
    report.write_table(columns, rows)
    report.write_entries(entries)

    return stagger


def report_final(
    report: Report,
    setup: ArchProblem,
    run: DescentRun,
) -> FinalReport:
    """
    The design the descent arrived at, read back against the standard.
    """
    q = run.walked.q[-1]
    result = build(setup, q, BETA_STOP)
    sized = build(setup, q, None)
    by_case = build_by_case(setup, q)
    decided = np.asarray(governing_compiled(by_case.sizes.sections.diameter))
    lengths = np.asarray(result.shape.lengths)

    lower, upper = setup.bounds
    at_lower = int(jnp.sum(jnp.abs(q - lower) < 1e-6))
    at_upper = int(jnp.sum(jnp.abs(q - upper) < 1e-6))
    shortest, longest = lengths.min(), lengths.max()

    diameters = sized.sizes.sections.diameter
    thinnest = float(jnp.min(diameters))
    thickest = float(jnp.max(diameters))
    smoothed = float(shortest_compiled(result.shape.lengths, FLOOR_BETA))
    stubs = int(np.sum(lengths < FLOOR))
    governing = [
        (name, f"governs {int(np.sum(decided == index))} of {NUM_EDGES}")
        for index, name in enumerate(CASE_NAMES)
    ]
    entries = (
        ("mass, enveloped", f"{float(compute_mass(result)):.9f} t"),
        ("mass, unsmoothed", f"{float(compute_mass(sized)):.9f} t"),
        ("worst utilization", f"{float(jnp.max(sized.sizes.utilization)):.15f}"),
        ("diameters", f"{thinnest:.2f} .. {thickest:.2f} mm"),
        ("rise", f"{float(jnp.max(result.shape.xyz[:, 2])):.1f} mm"),
        ("developed length", f"{float(jnp.sum(result.shape.lengths)):.1f} mm"),
        (
            "member length",
            f"{shortest:.1f} .. {longest:.1f} mm (ratio {longest / shortest:.1f})",
        ),
        ("shortest, smoothed", f"{smoothed:.1f} mm"),
        (f"members under {FLOOR:.0f} mm", f"{stubs} of {NUM_EDGES}"),
        ("force densities at bounds", f"{at_lower} lower, {at_upper} upper"),
        ("mirror gap, diameters", f"{mirror_gap(np.asarray(diameters)):.2e}"),
        ("mirror gap, force densities", f"{mirror_gap(np.asarray(q)):.2e}"),
        *governing,
    )

    report.write_heading(f"The design the descent arrived at, {run.label}")
    report.write_entries(entries)

    stagger = report_stagger(report, setup, q, result)

    mirror = mirror_gap(np.asarray(diameters))
    final = FinalReport(result, sized, decided, stagger, lengths, mirror)

    return final


def report_floor(
    report: Report,
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
        (
            "mass [t]",
            float(compute_mass(loose.sized)),
            float(compute_mass(held.sized)),
        ),
        ("shortest member [mm]", loose.lengths.min(), held.lengths.min()),
        ("longest member [mm]", loose.lengths.max(), held.lengths.max()),
        ("length ratio", loose_ratio, held_ratio),
        ("members under floor", loose_stubs, held_stubs),
    )
    columns = (
        ReportColumn("quantity", align="<"),
        ReportColumn("unconstrained", ".4f"),
        ReportColumn("floored", ".4f"),
    )
    kept = 1.0 - float(compute_mass(held.sized)) / funicular_mass
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

    starting = build(setup, setup.q, BETA_STOP)
    single_q = setup.q * SCALES[sweep.best]
    single = build(setup, single_q, BETA_STOP)
    single_sized = build(setup, single_q, None)
    single_by_case = build_by_case(setup, single_q)
    single_required = single_by_case.sizes.sections.diameter
    single_decided = np.asarray(governing_compiled(single_required))
    title_single = f"Best single $q$, {sweep.masses[sweep.best]:.4f} t"
    mass_loose = float(compute_mass(loose.sized))
    mass_held = float(compute_mass(held.sized))
    title_loose = f"Per member, no floor, {mass_loose:.4f} t"
    title_held = f"Per member, {FLOOR:.0f} mm floor, {mass_held:.4f} t"
    diameters_single = single_sized.sizes.sections.diameter
    diameters_loose = loose.sized.sizes.sections.diameter
    diameters_held = held.sized.sizes.sections.diameter
    forms = (
        Form(title_single, single.shape.xyz, diameters_single, single_decided),
        Form(title_loose, loose.envelope.shape.xyz, diameters_loose, loose.decided),
        Form(title_held, held.envelope.shape.xyz, diameters_held, held.decided),
    )
    edges = setup.structure.edges
    cases = figure_load_cases(edges, forms, CASE_NAMES, reference=starting.shape.xyz)
    cases.savefig(FIGURES / "03_load_cases.png", dpi=200)


def main(verbose: bool = True) -> None:
    """
    Sweep the funicular family, then descend on one variable per member.
    """
    report = Report(verbose)
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
    mass_loose = float(compute_mass(loose.sized))
    reduction = 1.0 - mass_loose / sweep.masses[funicular]
    against_best = 1.0 - mass_loose / sweep.masses[sweep.best]

    runs = (loose_run, held_run)
    finals = (loose, held)
    write_figures(setup, sweep, runs, finals)

    worst_utilization = float(jnp.max(loose.sized.sizes.utilization))
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

    not_the_answer = (("one staggered pass costs", f"{loose.stagger:.2%} of the mass"),)

    report.write_heading("What the answer is not")
    report.write_entries(not_the_answer)
    report.write_note(
        """
        Nothing in a member check restrains the stability of the whole frame,
        and nothing here measures it. Global stability is outside the pipeline
        by design, and verifying a finished design against it is future work.
        """
    )

    checks = (ToleranceCheck("scaled gradient error", sweep.worst, TOLERANCE_GRADIENT),)
    adequate = worst_utilization < 1.0 + 1e-9
    beats_uniform = mass_loose < sweep.masses[sweep.best]
    passed = verify_checks(checks) and sweep.interior and adequate and beats_uniform

    report.write_verdict(passed)


if __name__ == "__main__":
    main()
