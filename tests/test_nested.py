# SPDX-License-Identifier: Apache-2.0
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from normax.analysis.smax import SmaxAnalyzer
from normax.design import DesignParameters
from normax.design import StructuralDesignPipeline
from normax.design import compute_mass
from normax.form_finding import FdmFormFinder
from normax.form_finding import build_equilibrium_graph
from normax.form_finding import solve_equilibrium
from normax.loads import assemble_load_cases
from normax.loads import create_load_half_span
from normax.loads import create_load_uniform
from normax.materials import Steel355
from normax.optimization.nested import annealing_schedule
from normax.optimization.nested import design_envelope
from normax.optimization.nested import diameter_envelope
from normax.optimization.nested import governing_load_case
from normax.optimization.nested import minimize_bounded
from normax.optimization.nested import optimize_annealed
from normax.optimization.nested import optimize_staggered
from normax.optimization.nested import penalized_mass
from normax.optimization.nested import settle_diameters
from normax.optimization.nested import shortest_member
from normax.optimization.nested import size_design
from normax.optimization.nested import value_and_gradient
from normax.sections import build_section_catalog
from normax.sizing.ec3 import Ec3Sizer
from normax.structures import build_arch_2d

# A bowl whose minimum is known exactly, so the driver is tested against
# arithmetic rather than against the pipeline it usually drives.
CENTER = jnp.asarray([-2.0, -3.0, -1.5])
START = jnp.asarray([-8.0, -8.0, -8.0])
BOUNDS = (-10.0, -0.1)


def bowl(q, beta=1.0):
    return beta * jnp.sum((q - CENTER) ** 2)


def wall(q, beta=1.0):
    # Minimized outside the box, so the answer sits on the bound and the bound
    # is what has to hold it there.
    return beta * jnp.sum((q + 50.0) ** 2)


def bowl_with_design(q, beta=1.0):
    # Stands in for the pipeline: what the objective computed on the way to its
    # value, which a caller wants back rather than recomputed.
    return bowl(q, beta), {"q": q, "residual": q - CENTER}


# --------------------------------------------------------------------------- #
# The annealing schedule
# --------------------------------------------------------------------------- #
def test_the_schedule_starts_and_ends_where_it_is_told():
    schedule = annealing_schedule(10.0, 500.0, 5)

    assert float(schedule[0]) == pytest.approx(10.0)
    assert float(schedule[-1]) == pytest.approx(500.0)
    assert schedule.shape == (5,)


def test_the_schedule_rises_by_a_constant_ratio():
    # Geometric, because what the envelope gives away falls as the reciprocal of
    # the sharpness, so equal ratios buy equal fractions of what is left.
    schedule = np.asarray(annealing_schedule(10.0, 500.0, 6))
    ratios = schedule[1:] / schedule[:-1]

    assert np.allclose(ratios, ratios[0])


def test_a_schedule_of_one_round_is_just_its_start():
    assert np.asarray(annealing_schedule(7.0, 7.0, 1)) == pytest.approx(7.0)


@pytest.mark.parametrize("start,stop", [(0.0, 100.0), (10.0, -1.0), (-5.0, 5.0)])
def test_the_schedule_refuses_a_sharpness_that_is_not_positive(start, stop):
    with pytest.raises(ValueError, match="positive"):
        annealing_schedule(start, stop, 4)


def test_the_schedule_refuses_to_have_no_rounds():
    with pytest.raises(ValueError, match="rounds"):
        annealing_schedule(10.0, 500.0, 0)


# --------------------------------------------------------------------------- #
# One descent
# --------------------------------------------------------------------------- #
def test_the_descent_finds_a_minimum_it_can_reach():
    walked = minimize_bounded(bowl, START, bounds=BOUNDS, iterations=50).trajectory

    assert np.allclose(np.asarray(walked.q[-1]), np.asarray(CENTER), atol=1e-6)


def test_the_descent_never_goes_uphill():
    walked = minimize_bounded(bowl, START, bounds=BOUNDS, iterations=50).trajectory
    masses = np.asarray(walked.mass)

    assert np.all(np.diff(masses) <= 1e-12)


def test_the_descent_records_the_starting_point_first():
    # The trajectory is what a figure plots, so it has to begin where the search
    # began rather than after its first step.
    walked = minimize_bounded(bowl, START, bounds=BOUNDS, iterations=5).trajectory

    assert np.allclose(np.asarray(walked.q[0]), np.asarray(START))
    assert float(walked.mass[0]) == pytest.approx(float(bowl(START)))


def test_the_recorded_mass_belongs_to_the_recorded_iterate():
    walked = minimize_bounded(bowl, START, bounds=BOUNDS, iterations=50).trajectory

    for step, mass in zip(walked.q, walked.mass):
        assert float(mass) == pytest.approx(float(bowl(step)), rel=1e-12)


def test_the_bounds_hold_every_iterate():
    walked = minimize_bounded(wall, START, bounds=BOUNDS, iterations=30).trajectory
    visited = np.asarray(walked.q)

    assert np.all(visited >= BOUNDS[0] - 1e-12)
    assert np.all(visited <= BOUNDS[1] + 1e-12)


def test_a_minimum_outside_the_box_lands_on_the_bound():
    walked = minimize_bounded(wall, START, bounds=BOUNDS, iterations=30).trajectory

    assert np.allclose(np.asarray(walked.q[-1]), BOUNDS[0])


def test_spending_no_iterations_leaves_the_starting_point():
    # L-BFGS-B takes a step before it honours a limit of zero, and that step is
    # a clipped trial point rather than an improvement, so the driver refuses to
    # report it as an answer.
    walked = minimize_bounded(bowl, START, bounds=BOUNDS, iterations=0).trajectory

    assert np.allclose(np.asarray(walked.q[-1]), np.asarray(START))
    assert walked.q.shape[0] == 1


def test_the_last_iterate_is_the_answer_the_search_returned():
    # The point reported last and the point kept as best differ whenever the
    # search stops part-way through a line search, and callers read the last.
    walked = minimize_bounded(bowl, START, bounds=BOUNDS, iterations=2).trajectory
    masses = np.asarray(walked.mass)

    assert float(masses[-1]) == min(masses)


# --------------------------------------------------------------------------- #
# What the objective computed, carried out of the search
# --------------------------------------------------------------------------- #
def test_the_search_reports_the_value_the_trajectory_ends_on():
    found = minimize_bounded(bowl, START, bounds=BOUNDS, iterations=50)

    assert float(found.value) == float(found.trajectory.mass[-1])


def test_an_objective_that_computes_nothing_else_carries_nothing():
    found = minimize_bounded(bowl, START, bounds=BOUNDS, iterations=50)

    assert found.aux is None


def test_the_aux_belongs_to_the_answer_and_not_to_some_other_iterate():
    # The point the search answers with is not always the last one it evaluated,
    # and a design carried out of the wrong evaluation would be a design of a
    # structure nobody asked about.
    found = minimize_bounded(
        bowl_with_design, START, bounds=BOUNDS, iterations=50, has_aux=True
    )
    answer = np.asarray(found.trajectory.q[-1])

    assert np.allclose(np.asarray(found.aux["q"]), answer)
    assert np.allclose(np.asarray(found.aux["residual"]), answer - np.asarray(CENTER))


def test_the_aux_survives_a_search_that_goes_nowhere():
    found = minimize_bounded(
        bowl_with_design, START, bounds=BOUNDS, iterations=0, has_aux=True
    )

    assert np.allclose(np.asarray(found.aux["q"]), np.asarray(START))


def test_carrying_a_design_costs_at_most_one_extra_evaluation():
    # The design comes out of the evaluations the search already made, save for
    # the answer when it is not the point evaluated last, which costs one call.
    def count_evaluations(objective, has_aux):
        calls = []
        compiled = value_and_gradient(objective, has_aux=has_aux)

        def counted(q):
            calls.append(q)
            return compiled(q)

        minimize_bounded(
            objective,
            START,
            bounds=BOUNDS,
            iterations=20,
            has_aux=has_aux,
            gradient=counted,
        )

        return len(calls)

    plain = count_evaluations(bowl, False)
    carried = count_evaluations(bowl_with_design, True)

    assert carried <= plain + 1


def test_carrying_a_design_leaves_the_search_where_it_was():
    plain = minimize_bounded(bowl, START, bounds=BOUNDS, iterations=50)
    carried = minimize_bounded(
        bowl_with_design, START, bounds=BOUNDS, iterations=50, has_aux=True
    )

    assert np.allclose(np.asarray(carried.trajectory.q), np.asarray(plain.trajectory.q))
    assert np.allclose(
        np.asarray(carried.trajectory.mass), np.asarray(plain.trajectory.mass)
    )


def test_the_annealed_search_carries_the_last_round_out():
    schedule = annealing_schedule(1.0, 4.0, 3)
    found = optimize_annealed(
        bowl_with_design, START, schedule, bounds=BOUNDS, iterations=10, has_aux=True
    )

    assert np.allclose(np.asarray(found.aux["q"]), np.asarray(found.trajectory.q[-1]))


# --------------------------------------------------------------------------- #
# The annealed descent
# --------------------------------------------------------------------------- #
def test_the_annealed_descent_reports_every_round():
    schedule = annealing_schedule(1.0, 4.0, 3)
    found = optimize_annealed(bowl, START, schedule, bounds=BOUNDS, iterations=10)
    walked = found.trajectory

    assert set(np.unique(np.asarray(walked.beta))) == set(
        float(sharpness) for sharpness in schedule
    )


def test_each_round_starts_where_the_last_one_stopped():
    # Warm starting is the whole point of a schedule: a sharper round refines a
    # design rather than rediscovering it.
    schedule = annealing_schedule(1.0, 4.0, 3)
    found = optimize_annealed(bowl, START, schedule, bounds=BOUNDS, iterations=10)
    walked = found.trajectory

    betas = np.asarray(walked.beta)
    visited = np.asarray(walked.q)
    boundaries = np.flatnonzero(np.diff(betas))

    for edge in boundaries:
        assert np.allclose(visited[edge], visited[edge + 1])


def test_the_annealed_descent_reaches_the_same_minimum():
    schedule = annealing_schedule(1.0, 4.0, 3)
    found = optimize_annealed(bowl, START, schedule, bounds=BOUNDS, iterations=25)
    walked = found.trajectory

    assert np.allclose(np.asarray(walked.q[-1]), np.asarray(CENTER), atol=1e-6)


def test_the_sharpness_reaches_the_objective():
    # A round has to be taken under its own sharpness, or annealing does nothing
    # at all and the schedule is decoration. Read off the objective rather than
    # inside it: the sharpness arrives traced, so only its effect is concrete.
    found = optimize_annealed(
        bowl, START, annealing_schedule(2.0, 8.0, 3), bounds=BOUNDS, iterations=3
    )
    walked = found.trajectory

    assert sorted(set(np.asarray(walked.beta).tolist())) == pytest.approx(
        [2.0, 4.0, 8.0]
    )
    for q, beta, mass in zip(walked.q, walked.beta, walked.mass):
        assert float(mass) == pytest.approx(float(bowl(q, beta)), rel=1e-12)


def test_the_schedule_traces_the_objective_once():
    # The sharpness parameterizes one compiled program rather than selecting
    # between one per round. Capturing it again would compile the pipeline as
    # many times as the schedule is long, which is most of what a descent costs.
    traces = 0

    def counted(q, beta):
        nonlocal traces
        traces += 1
        return bowl(q, beta)

    optimize_annealed(
        counted, START, annealing_schedule(2.0, 8.0, 5), bounds=BOUNDS, iterations=3
    )

    assert traces == 1


def test_a_schedule_of_plain_floats_also_traces_once():
    # A float leaf is static under `eqx.filter_jit`, so a sequence that is not
    # converted to an array compiles a program per round without saying so.
    traces = 0

    def counted(q, beta):
        nonlocal traces
        traces += 1
        return bowl(q, beta)

    optimize_annealed(counted, START, [2.0, 4.0, 8.0], bounds=BOUNDS, iterations=3)

    assert traces == 1


# --------------------------------------------------------------------------- #
# The length floor
# --------------------------------------------------------------------------- #
LENGTHS = jnp.asarray([500.0, 700.0, 300.0, 900.0])


def test_the_smooth_minimum_never_overstates_the_shortest():
    # Understating is the safe direction for a floor: a constraint built on it
    # bites slightly early rather than slightly late.
    for beta in (5.0, 20.0, 100.0):
        assert float(shortest_member(LENGTHS, beta)) <= float(jnp.min(LENGTHS))


def test_the_smooth_minimum_approaches_the_shortest_as_it_sharpens():
    blunt = float(shortest_member(LENGTHS, 5.0))
    sharp = float(shortest_member(LENGTHS, 200.0))

    assert blunt < sharp <= float(jnp.min(LENGTHS))
    assert sharp == pytest.approx(float(jnp.min(LENGTHS)), rel=1e-6)


def test_the_smooth_minimum_respects_its_bound():
    # It falls below the true shortest by at most the member count raised to the
    # reciprocal of the sharpness, the same bound the envelope carries.
    for beta in (5.0, 20.0, 100.0):
        ratio = float(jnp.min(LENGTHS)) / float(shortest_member(LENGTHS, beta))

        assert 1.0 <= ratio <= float(len(LENGTHS) ** (1.0 / beta)) + 1e-12


def test_the_smooth_minimum_scales_with_the_structure():
    # Taken in the logarithm, so the sharpness means the same thing whatever the
    # structure is measured in.
    assert float(shortest_member(LENGTHS * 1000.0, 20.0)) == pytest.approx(
        1000.0 * float(shortest_member(LENGTHS, 20.0))
    )


def test_a_design_clear_of_the_floor_is_not_penalized():
    mass = jnp.asarray(0.05)

    assert float(penalized_mass(mass, LENGTHS, 200.0, beta=50.0, weight=10.0)) == (
        pytest.approx(float(mass))
    )


def test_a_design_below_the_floor_is_penalized():
    mass = jnp.asarray(0.05)
    inflated = float(penalized_mass(mass, LENGTHS, 600.0, beta=50.0, weight=10.0))

    assert inflated > float(mass)


def test_the_penalty_grows_as_the_shortest_member_shrinks():
    mass = jnp.asarray(0.05)
    lengths = [LENGTHS.at[2].set(value) for value in (280.0, 200.0, 120.0)]
    inflated = [
        float(penalized_mass(mass, load_case, 600.0, beta=50.0, weight=10.0))
        for load_case in lengths
    ]

    assert inflated[0] < inflated[1] < inflated[2]


def test_the_penalty_is_flat_where_it_starts():
    # Squared, so the objective is not kinked at the floor and a search may
    # approach it from either side.
    mass = jnp.asarray(0.05)
    floor = float(shortest_member(LENGTHS, 50.0))

    slope = jax.grad(lambda x: penalized_mass(mass, x, floor, beta=50.0, weight=10.0))(
        LENGTHS
    )

    assert np.allclose(np.asarray(slope), 0.0, atol=1e-9)


def test_the_penalty_is_a_fraction_and_not_a_mass():
    # Multiplicative and reading a ratio, so it needs no mass scale and means
    # the same thing on a structure of any size.
    light = jnp.asarray(0.05)
    heavy = jnp.asarray(5.0)

    ratio_light = (
        float(penalized_mass(light, LENGTHS, 600.0, beta=50.0, weight=10.0)) / 0.05
    )
    ratio_heavy = (
        float(penalized_mass(heavy, LENGTHS, 600.0, beta=50.0, weight=10.0)) / 5.0
    )

    assert ratio_light == pytest.approx(ratio_heavy)


def test_the_penalty_gradient_reaches_only_the_shortest_members():
    mass = jnp.asarray(0.05)

    slope = np.asarray(
        jax.grad(lambda x: penalized_mass(mass, x, 600.0, beta=50.0, weight=10.0))(
            LENGTHS
        )
    )

    assert np.all(np.isfinite(slope))
    assert abs(slope[2]) > abs(slope[3])


# --------------------------------------------------------------------------- #
# The envelope and the staggered coupling, on a real pipeline
# --------------------------------------------------------------------------- #
# A 10 m arch rising 3 m under 180 kN over its free nodes, in millimeters and
# newtons.
SPAN = 10_000.0
RISE = 3_000.0
TOTAL_LOAD = 180_000.0
NUM_EDGES = 10

# The diameter the frame is analyzed with before the check has spoken.
SEED = 100.0

# Sharpness of the envelope in the several-load-case tests.
SHARPNESS = 50.0

# Bounds wide enough that a descent is not pinned by them, and what it may spend.
DENSITY_BOUNDS = (-500.0, -1.0)
STAGGERED_ITERATIONS = 10

# Largest fractional movement in a diameter a settled coupling may still show.
TOLERANCE_SETTLING = 1e-6

# Below this, two masses of the same design disagree only by round-off.
TOLERANCE_ROUNDOFF = 1e-9


@pytest.fixture(scope="module")
def structure():
    return build_arch_2d(num_edges=NUM_EDGES, span=SPAN, rise=RISE)


@pytest.fixture(scope="module")
def force_densities(structure):
    """Force densities reaching the target rise, so the arch is the same one."""
    graph = build_equilibrium_graph(structure)
    trial = jnp.full(NUM_EDGES, -1.0)
    state = solve_equilibrium(
        trial,
        structure.nodes[graph.indices_fixed],
        graph,
        create_load_uniform(structure, TOTAL_LOAD),
    )

    return trial * jnp.max(state.xyz[:, 2]) / RISE


@pytest.fixture(scope="module")
def pipeline(structure):
    catalog = build_section_catalog(Steel355(), 3)

    return StructuralDesignPipeline(
        FdmFormFinder(structure),
        SmaxAnalyzer(structure, catalog(SEED)),
        Ec3Sizer(structure, catalog),
    )


@pytest.fixture(scope="module")
def params(force_densities):
    return DesignParameters(force_densities, jnp.full(NUM_EDGES, SEED))


@pytest.fixture(scope="module")
def one_case(structure):
    return assemble_load_cases([create_load_uniform(structure, TOTAL_LOAD)])


@pytest.fixture(scope="module")
def three_cases(structure):
    cases = [
        create_load_uniform(structure, TOTAL_LOAD),
        create_load_half_span(structure, TOTAL_LOAD, factor=0.25),
        create_load_half_span(structure, TOTAL_LOAD, factor=0.25, mirrored=True),
    ]

    return assemble_load_cases(cases)


def mass_objective(pipeline, loads):
    """
    The enveloped mass of a design, and the design that was weighed.
    """

    def objective(params):
        design = size_design(pipeline, params, loads)
        sized = design_envelope(design, SHARPNESS)

        return compute_mass(sized), sized

    return objective


def round_seams(trajectory):
    """
    Where one round hands over to the next, which is a repeated iterate.
    """
    return [
        index
        for index in range(len(trajectory.q) - 1)
        if jnp.array_equal(trajectory.q[index], trajectory.q[index + 1])
    ]


@pytest.fixture(scope="module")
def staggered(pipeline, params, three_cases):
    """One staggered search, counting the traces so every test shares them."""
    traces = []
    weighed = mass_objective(pipeline, three_cases)

    def counted(design_params):
        traces.append(1)

        return weighed(design_params)

    found = optimize_staggered(
        counted,
        params,
        bounds=DENSITY_BOUNDS,
        iterations=STAGGERED_ITERATIONS,
    )

    return found, len(traces)


def test_the_sizing_map_sizes_every_case_on_its_own(pipeline, params, three_cases):
    design = size_design(pipeline, params, three_cases)

    assert design.sizes.sections.diameter.shape == (3, NUM_EDGES)
    assert jnp.max(jnp.abs(jnp.max(design.sizes.utilization, axis=0) - 1.0)) < 1e-9


def test_the_governing_case_demanded_the_largest_section(pipeline, params, three_cases):
    design = size_design(pipeline, params, three_cases)
    governing = governing_load_case(design.sizes.sections.diameter)

    assert np.array_equal(
        np.asarray(governing),
        np.argmax(np.asarray(design.sizes.sections.diameter), axis=0),
    )


def test_repeating_a_load_case_changes_nothing(pipeline, params, one_case):
    once = size_design(pipeline, params, one_case)
    twice = size_design(
        pipeline, params, assemble_load_cases([one_case.analysis[0]] * 2)
    )

    assert jnp.array_equal(
        twice.sizes.sections.diameter[1], once.sizes.sections.diameter[0]
    )
    assert compute_mass(design_envelope(twice)) == compute_mass(design_envelope(once))


def test_the_largest_is_the_default_and_the_envelope_bounds_it(
    pipeline, params, three_cases
):
    design = size_design(pipeline, params, three_cases)
    largest = design_envelope(design)
    enveloped = design_envelope(design, SHARPNESS)

    assert jnp.array_equal(
        largest.sizes.sections.diameter, jnp.max(design.sizes.sections.diameter, axis=0)
    )
    assert jnp.all(enveloped.sizes.sections.diameter >= largest.sizes.sections.diameter)
    assert compute_mass(enveloped) >= compute_mass(largest)


def test_one_load_case_is_never_enveloped(pipeline, params, one_case):
    design = size_design(pipeline, params, one_case)

    for sharpness in (None, 1.0, SHARPNESS):
        covered = design_envelope(design, sharpness)

        assert jnp.array_equal(
            covered.sizes.sections.diameter, design.sizes.sections.diameter[0]
        )


CASES = jnp.asarray([[100.0, 300.0], [200.0, 150.0], [50.0, 220.0]])


def test_the_envelope_covers_every_load_case():
    largest = jnp.max(CASES, axis=0)

    assert jnp.all(diameter_envelope(CASES, 20.0) >= largest * (1.0 - 1e-12))


def test_the_envelope_approaches_the_largest_load_case():
    sharp = diameter_envelope(CASES, 2000.0)

    assert np.asarray(sharp) == pytest.approx(
        np.asarray(jnp.max(CASES, axis=0)), rel=1e-3
    )


def test_the_envelope_respects_its_bound():
    beta = 20.0
    slack = jnp.log(diameter_envelope(CASES, beta)) - jnp.log(jnp.max(CASES, axis=0))

    assert jnp.all(slack >= 0.0)
    assert jnp.all(slack <= jnp.log(CASES.shape[0]) / beta + 1e-12)


def test_the_envelope_tightens_as_it_sharpens():
    values = [
        float(diameter_envelope(CASES, beta)[0]) for beta in (2.0, 5.0, 10.0, 20.0)
    ]

    assert np.all(np.diff(values) < 0.0)


def test_the_envelope_is_differentiable():
    gradient = jax.grad(lambda c: jnp.sum(diameter_envelope(c, 50.0)))(CASES)

    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.all(gradient >= 0.0)


def test_the_enveloped_mass_compiles_under_jit(pipeline, params, three_cases):
    def composed(blocks, design, loads):
        return compute_mass(
            design_envelope(size_design(blocks, design, loads), SHARPNESS)
        )

    traced = eqx.filter_jit(composed)(pipeline, params, three_cases)
    eager = composed(pipeline, params, three_cases)

    assert float(traced) == pytest.approx(float(eager), rel=1e-12)


def test_settling_returns_diameters_the_check_agrees_with(
    pipeline, params, three_cases
):
    weighed = mass_objective(pipeline, three_cases)
    settled = settle_diameters(weighed, params)
    _, design = weighed(DesignParameters(params.force_densities, settled))
    demanded = design.sizes.sections.diameter

    assert float(jnp.max(jnp.abs(demanded / settled - 1.0))) < TOLERANCE_SETTLING


def test_the_staggered_search_closes_its_coupling(staggered, pipeline, three_cases):
    found, _ = staggered
    settled = found.aux.sizes.sections.diameter
    answer = DesignParameters(found.trajectory.q[-1], settled)
    weighed = design_envelope(size_design(pipeline, answer, three_cases), SHARPNESS)
    demanded = weighed.sizes.sections.diameter

    assert float(jnp.max(jnp.abs(demanded / settled - 1.0))) < TOLERANCE_SETTLING


def test_the_staggered_search_traces_the_objective_twice(staggered):
    # Once for the descent, once for the settling passes; neither retraces per round.
    found, traces = staggered

    assert round_seams(found.trajectory)
    assert traces == 2


def test_a_round_starts_where_the_last_one_stopped(staggered):
    # A seam repeats the densities and moves the mass, because only the seed changed.
    found, _ = staggered
    walked = found.trajectory
    seams = round_seams(walked)

    assert seams

    first = seams[0]
    before = float(walked.mass[first])
    after = float(walked.mass[first + 1])

    assert abs(after / before - 1.0) > TOLERANCE_ROUNDOFF


def test_the_staggered_search_reports_the_value_it_ends_on(staggered):
    found, _ = staggered

    assert float(found.value) == float(found.trajectory.mass[-1])


def test_the_staggered_search_refuses_a_coupling_that_has_not_settled(
    pipeline, params, three_cases
):
    weighed = mass_objective(pipeline, three_cases)

    with pytest.raises(ValueError, match="still moving"):
        optimize_staggered(
            weighed,
            params,
            bounds=DENSITY_BOUNDS,
            iterations=STAGGERED_ITERATIONS,
            rounds=1,
        )


def test_the_two_caps_are_refused_separately(pipeline, params, three_cases):
    weighed = mass_objective(pipeline, three_cases)

    with pytest.raises(ValueError, match="passes at fixed force densities"):
        optimize_staggered(
            weighed, params, bounds=DENSITY_BOUNDS, iterations=1, settling_passes=1
        )


def test_the_frozen_seed_misreports_the_mass(staggered, pipeline, three_cases):
    found, _ = staggered
    frozen = DesignParameters(found.trajectory.q[-1], jnp.full(NUM_EDGES, SEED))
    weighed = design_envelope(size_design(pipeline, frozen, three_cases), SHARPNESS)
    reported = float(compute_mass(weighed))

    assert abs(reported / float(found.value) - 1.0) > TOLERANCE_ROUNDOFF
