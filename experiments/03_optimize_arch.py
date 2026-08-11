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

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from normax.analysis.smax import prepare_model
from normax.ec3.actions import MemberActions
from normax.ec3.material import SteelGrade
from normax.ec3.section import TubeCatalogue
from normax.formfinding import equilibrium_graph
from normax.formfinding import equilibrium_state
from normax.optimization import annealing_schedule
from normax.optimization import optimize_annealed
from normax.optimization import penalized_mass
from normax.optimization import shortest_member
from normax.pipeline import Design
from normax.pipeline import ProblemSetup
from normax.pipeline import design_envelope
from normax.pipeline import frame_stability
from normax.pipeline import governing_load_case
from normax.pipeline import unsmoothed_design
from normax.structures import arch_2d
from normax.structures import crown_node
from normax.structures import loads_half_span
from normax.structures import loads_point
from normax.structures import loads_uniform
from normax.visualization import Descent
from normax.visualization import Form
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
CATALOGUE = TubeCatalogue.at_class_limit(STEEL.f_y, 3)
SECTION_CLASS = 3


def setup():
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
    reached = jnp.max(equilibrium_state(trial, structure, graph_fdm).xyz[:, 2])

    problem = ProblemSetup(structure, graph_fdm, model, STEEL, CATALOGUE)

    return problem, trial * reached / RISE


def build_load_cases(structure):
    """
    Three cases of equal total: funicular, half span, and a crown point load.
    """
    spread = TOTAL_LOAD / (NUM_EDGES - 1)

    uniform = loads_uniform(structure, spread)

    half = loads_half_span(structure, spread, factor=HALF_FACTOR)
    half = half * (TOTAL_LOAD / abs(float(jnp.sum(half[:, 2]))))

    point = loads_uniform(structure, spread * (1.0 - POINT_SHARE)) + loads_point(
        structure, TOTAL_LOAD * POINT_SHARE, node=crown_node(structure)
    )

    return jnp.stack([uniform, half, point])


@eqx.filter_jit
def build(q, problem, load_cases, beta, diameters=None):
    """
    The enveloped design at one set of force densities.

    Compiled, which is what makes the sweeps below affordable as well as the
    descents: every caller here passes the same prepared topologies, so one trace
    serves all of them. The sharpness is an argument rather than a constant so
    that annealing it does not compile again.
    """
    seed = jnp.full(NUM_EDGES, SEED) if diameters is None else diameters

    return design_envelope(
        q,
        seed,
        problem,
        load_cases,
        beta=beta,
        section_class=SECTION_CLASS,
    )


def report_load_cases(problem, load_cases, q):
    """
    What each case demands of each member at the starting shape.
    """
    result = build(q, problem, load_cases, BETA_STOP)
    decided = np.asarray(governing_load_case(result))

    print("The three load cases, each carrying 180 kN")
    print(f"  {'member':>7} {'d LC1':>9} {'d LC2':>9} {'d LC3':>9} {'governs':>16}")
    for member in range(NUM_EDGES):
        sizes = "".join(
            f" {float(result.required[load_case, member]):>9.2f}"
            for load_case in range(3)
        )
        print(f"  {member:>7}{sizes} {CASE_NAMES[decided[member]]:>16}")

    counts = [int(np.sum(decided == load_case)) for load_case in range(3)]
    for name, count in zip(CASE_NAMES, counts):
        print(f"  {name:>16} governs {count:>3} of {NUM_EDGES}")

    return result, decided


def report_smoothing(problem, load_cases, q):
    """
    What the envelope gives away at each sharpness, against the true largest.
    """
    print("\nWhat the envelope costs, and the bound on it")
    print(f"  {'beta':>8} {'mass [t]':>14} {'excess':>10} {'bound':>10} {'max u':>18}")

    for beta in SHARPNESSES:
        result = build(q, problem, load_cases, beta)
        exact = unsmoothed_design(result, problem, section_class=SECTION_CLASS)

        excess = float(result.mass) / float(exact.mass) - 1.0
        bound = float(load_cases.shape[0] ** (2.0 / float(beta))) - 1.0
        print(
            f"  {float(beta):>8.0f} {float(result.mass):>14.9f} {excess:>10.4%}"
            f" {bound:>10.4%} {float(jnp.max(exact.utilization)):>18.15f}"
        )


def report_sweep(problem, load_cases, q):
    """
    The mass along the uniform force densities, and the gradient against it.
    """

    def objective(scaled):
        return build(scaled, problem, load_cases, BETA_STOP).mass

    masses = []
    exact = []
    numeric = []

    print("\nThe mass along the uniform force densities")
    print(
        f"  {'scale':>7} {'q [N/mm]':>12} {'mass [t]':>13} {'d/dk':>14} {'scaled':>10}"
    )

    slopes = [float(jnp.sum(jax.grad(objective)(q * scale) * q)) for scale in SCALES]
    largest = max(abs(slope) for slope in slopes)

    sampled = (0, len(SCALES) // 2, len(SCALES) - 1)
    print(f"\n  {'relative step':>14} {'worst scaled error':>20}")
    for relative in STEPS:
        error = 0.0
        for index in sampled:
            scale = SCALES[index]
            step = abs(scale) * relative
            forward = float(objective(q * (scale + step)))
            backward = float(objective(q * (scale - step)))
            difference = (forward - backward) / (2.0 * step)
            error = max(error, abs(slopes[index] - difference) / largest)
        print(f"  {relative:>14.0e} {error:>20.2e}")
    print(f"  the trough is at {STEP:.0e}, so that is where the check is made\n")

    for scale, slope in zip(SCALES, slopes):
        step = abs(scale) * STEP
        forward = float(objective(q * (scale + step)))
        backward = float(objective(q * (scale - step)))
        difference = (forward - backward) / (2.0 * step)

        masses.append(float(objective(q * scale)))
        exact.append(slope)
        numeric.append(difference)

        scaled = abs(slope - difference) / largest
        print(
            f"  {scale:>7.2f} {float(q[0]) * scale:>12.3f} {masses[-1]:>13.9f}"
            f" {slope:>14.6e} {scaled:>10.2e}"
        )

    best = int(np.argmin(masses))
    interior = 0 < best < len(SCALES) - 1
    worst = max(abs(a - b) / largest for a, b in zip(exact, numeric))

    print(f"  best uniform scale {SCALES[best]:.2f}, mass {masses[best]:.9f} t")
    print(f"  interior minimum   {interior}")
    print(f"  worst scaled gradient error {worst:.2e}")

    return np.asarray(masses), np.asarray(exact), np.asarray(numeric), best, worst


def report_descent(problem, load_cases, q, *, floor):
    """
    The same mass with one force density per member, annealed.
    """
    bounds = (float(q[0]) * DECADES, float(q[0]) / DECADES)
    schedule = annealing_schedule(BETA_START, float(BETA_STOP), ROUNDS)

    kind = f"a {floor:.0f} mm floor" if floor else "no length floor"
    print(f"\nDescending with one variable per member, {kind}, bounds {bounds}")

    def objective(x, beta):
        result = build(x, problem, load_cases, beta)
        if not floor:
            return result.mass

        return penalized_mass(
            result.mass,
            result.lengths,
            floor,
            beta=FLOOR_BETA,
            weight=FLOOR_WEIGHT,
        )

    walked = optimize_annealed(
        objective, q, schedule, bounds=bounds, iterations=ITERATIONS
    )

    print(f"  {'step':>6} {'beta':>8} {'objective [t]':>16}")
    for step, (beta, total) in enumerate(zip(walked.beta, walked.mass)):
        print(f"  {step:>6} {float(beta):>8.1f} {float(total):>16.9f}")

    return walked, bounds


def report_final(problem, load_cases, walked, bounds, label):
    """
    The design the descent arrived at, read back against the standard.
    """
    q = walked.q[-1]
    result = build(q, problem, load_cases, BETA_STOP)
    exact = unsmoothed_design(result, problem, section_class=SECTION_CLASS)
    decided = np.asarray(governing_load_case(result))

    lower = int(jnp.sum(jnp.abs(q - bounds[0]) < 1e-6))
    upper = int(jnp.sum(jnp.abs(q - bounds[1]) < 1e-6))

    lengths = np.asarray(result.lengths)

    print(f"\nThe design the descent arrived at, {label}")
    print(f"  mass, enveloped      {float(result.mass):.9f} t")
    print(f"  mass, unsmoothed     {float(exact.mass):.9f} t")
    print(f"  worst utilization    {float(jnp.max(exact.utilization)):.15f}")
    print(
        f"  diameters            {float(jnp.min(exact.diameters)):.2f}"
        f" .. {float(jnp.max(exact.diameters)):.2f} mm"
    )
    print(f"  rise                 {float(jnp.max(result.xyz[:, 2])):.1f} mm")
    print(f"  developed length     {float(jnp.sum(result.lengths)):.1f} mm")
    print(
        f"  member length        {lengths.min():.1f} .. {lengths.max():.1f} mm,"
        f" ratio {lengths.max() / lengths.min():.1f}"
    )
    smoothed = float(shortest_member(result.lengths, FLOOR_BETA))
    print(f"  shortest, smoothed   {smoothed:.1f} mm")
    stubs = int(np.sum(lengths < FLOOR))
    print(f"  members under {FLOOR:.0f} mm   {stubs} of {NUM_EDGES}")
    print(f"  force densities at the lower bound {lower}, at the upper {upper}")

    counts = [int(np.sum(decided == load_case)) for load_case in range(3)]
    for name, count in zip(CASE_NAMES, counts):
        print(f"  {name:>16} governs {count:>3} of {NUM_EDGES}")

    relaxed = result
    diameters = jnp.full(NUM_EDGES, SEED)
    print(f"\n  {'pass':>6} {'relative move':>15} {'mass [t]':>14}")
    for step in range(PASSES):
        relaxed = build(q, problem, load_cases, BETA_STOP, diameters)
        move = float(
            jnp.max(jnp.abs(relaxed.diameters - diameters) / relaxed.diameters)
        )
        print(f"  {step:>6} {move:>15.3e} {float(relaxed.mass):>14.9f}")
        diameters = relaxed.diameters

    stagger = abs(float(result.mass) - float(relaxed.mass)) / float(relaxed.mass)
    print(f"  one pass costs {stagger:.3%} of the mass")

    single = Design(
        result.xyz,
        result.lengths,
        MemberActions(
            result.axial_force[0],
            result.moment_major[0],
            result.moment_minor[0],
            result.moment_factor_major[0],
            result.moment_factor_minor[0],
        ),
        result.buckling_length,
        exact.diameters,
        exact.utilization[0],
        exact.mass,
    )
    # A critical load factor belongs to a load case, and the case the shape was
    # found under is not the one that sized it, so quoting only that one would
    # flatter the design.
    print(f"\n  {'case':>16} {'alpha_cr':>10} {'adequate':>10}")
    weakest = None
    for name, load_case in zip(CASE_NAMES, load_cases):
        factors = frame_stability(single, problem, num_modes=1, loads=load_case)
        alpha_cr = float(factors.factors[0])
        weakest = alpha_cr if weakest is None else min(weakest, alpha_cr)
        print(f"  {name:>16} {alpha_cr:>10.4f} {str(bool(factors.adequate)):>10}")

    return result, exact, decided, stagger, weakest, lengths


def main():
    problem, q = setup()
    load_cases = build_load_cases(problem.structure)

    start, _ = report_load_cases(problem, load_cases, q)
    report_smoothing(problem, load_cases, q)
    masses, exact, numeric, best, worst = report_sweep(problem, load_cases, q)
    walked, bounds = report_descent(problem, load_cases, q, floor=0.0)
    final, sized, decided_final, stagger, alpha_cr, lengths = report_final(
        problem, load_cases, walked, bounds, "no length floor"
    )

    # The funicular design, which is where the descent starts and the only
    # reference either reduction means anything against.
    funicular = int(np.argmin(np.abs(SCALES - 1.0)))
    reduction = 1.0 - float(sized.mass) / masses[funicular]
    against_best = 1.0 - float(sized.mass) / masses[best]

    floored, _ = report_descent(structure, graph_fdm, model, cases, q, floor=FLOOR)
    held, held_sized, decided_held, _, held_alpha, held_lengths = report_final(
        structure, graph_fdm, model, cases, floored, bounds, f"a {FLOOR:.0f} mm floor"
    )

    # The best the single force density can do, which is the design the twenty
    # variables have to beat and the one the figures compare against.
    single = build(q * SCALES[best], structure, graph_fdm, model, cases, BETA_STOP)
    single_sized = unsmoothed_design(
        single, STEEL, CATALOGUE, section_class=SECTION_CLASS
    )
    decided_single = np.asarray(governing_case(single))

    FIGURES.mkdir(exist_ok=True)
    figure_optimization(
        SCALES,
        masses,
        exact,
        numeric,
        (
            Descent(
                "no length floor",
                np.asarray(walked.mass),
                np.asarray(walked.beta),
            ),
            Descent(
                f"{FLOOR:.0f} mm floor",
                np.asarray(floored.mass),
                np.asarray(floored.beta),
            ),
        ),
    ).savefig(FIGURES / "03_optimization.png", dpi=200)
    figure_load_cases(
        problem.structure.edges,
        (
            Form(
                f"Best single $q$, {masses[best]:.4f} t",
                single.xyz,
                single_sized.diameters,
                decided_single,
            ),
            Form(
                f"Per member, no floor, {float(sized.mass):.4f} t",
                final.xyz,
                sized.diameters,
                decided_final,
            ),
            Form(
                f"Per member, {FLOOR:.0f} mm floor, {float(held_sized.mass):.4f} t",
                held.xyz,
                held_sized.diameters,
                decided_held,
            ),
        ),
        CASE_NAMES,
    ).savefig(FIGURES / "03_load_cases.png", dpi=200)

    print("\nSummary")
    print(f"  interior minimum in the uniform sweep   {0 < best < len(SCALES) - 1}")
    print(
        f"  worst scaled gradient error             {worst:.2e}"
        f"  (target {TOLERANCE_GRADIENT:.0e})"
    )
    print(
        f"  worst utilization of the final design   "
        f"{float(jnp.max(sized.utilization)):.12f}"
    )
    print(f"  lighter than the funicular arch         {reduction:.1%}")
    print(f"  lighter than the best uniform arch      {against_best:.1%}")

    print("\nWhat a length floor costs, and what it buys")
    print(f"  {'':>22} {'unconstrained':>16} {'floored':>16}")
    rows = (
        ("mass [t]", float(sized.mass), float(held_sized.mass)),
        ("shortest member [mm]", lengths.min(), held_lengths.min()),
        ("longest member [mm]", lengths.max(), held_lengths.max()),
        (
            "length ratio",
            lengths.max() / lengths.min(),
            held_lengths.max() / held_lengths.min(),
        ),
        (
            "members under floor",
            int(np.sum(lengths < FLOOR)),
            int(np.sum(held_lengths < FLOOR)),
        ),
        ("alpha_cr, weakest", alpha_cr, held_alpha),
    )
    for name, loose, tight in rows:
        print(f"  {name:>22} {loose:>16.4f} {tight:>16.4f}")
    kept = 1.0 - float(held_sized.mass) / masses[funicular]
    print(f"  the floored design is {kept:.1%} lighter than the funicular arch")

    print("\nWhat the answer is not")
    print(f"  one staggered pass costs                {stagger:.2%} of the mass")
    print(f"  critical load factor of the design      {alpha_cr:.4f}")
    print("  the descent spent the stability margin the starting arch had, and")
    print("  nothing in a member check was ever going to stop it. Global")
    print("  stability is outside the pipeline by design; this is what that costs.")

    passed = (
        0 < best < len(SCALES) - 1
        and worst < TOLERANCE_GRADIENT
        and float(jnp.max(sized.utilization)) < 1.0 + 1e-9
        and float(sized.mass) < masses[best]
    )
    print(f"\n{'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
