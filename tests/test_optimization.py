import jax
import jax.numpy as jnp
import numpy as np
import pytest

from normax.optimization import AugmentedBudget
from normax.optimization import ConstrainedMaps
from normax.optimization import annealing_schedule
from normax.optimization import augmented_penalty
from normax.optimization import descend_augmented
from normax.optimization import minimize_bounded
from normax.optimization import optimize_annealed
from normax.optimization import penalized_mass
from normax.optimization import shifted_multipliers
from normax.optimization import shortest_member
from normax.optimization import strayed_point
from normax.optimization import value_and_gradient

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
# The augmented Lagrangian
# --------------------------------------------------------------------------- #
# Minimize the sum of two positives under the hyperbola they must sit above.
# The answer is the corner (1, 1), the row is active there, and matching
# gradients puts its multiplier at exactly one — so the landing, the activity
# and the price of the constraint are all arithmetic rather than a fixture.
CORNER = np.array([1.0, 1.0])
CORNER_START = np.array([3.0, 3.0])
CORNER_BOXES = [(0.1, 10.0), (0.1, 10.0)]
CORNER_BUDGET = AugmentedBudget(
    rounds=25,
    iterations=200,
    settled=100,
    opening=2,
    penalty=1.0,
    growth=10.0,
    ceiling=1.0e8,
    tolerance=1.0e-9,
    quiet=1.0e-12,
)


def summed(x):
    return x[0] + x[1]


def hyperbola(x):
    return jnp.atleast_1d(x[0] * x[1] - 1.0)


def constrained_maps(weigh, slack):
    def augmented(x, multipliers, penalty, reference):
        penalized = augmented_penalty(slack(x), multipliers, penalty)

        return weigh(x) / reference + penalized

    return ConstrainedMaps(
        jax.jit(jax.value_and_grad(augmented)),
        jax.jit(jax.value_and_grad(weigh)),
        jax.jit(slack),
    )


def test_satisfied_rows_at_no_price_cost_nothing():
    slack = jnp.asarray([0.5, 1.0, 2.0])
    resting = jnp.zeros(3)

    assert float(augmented_penalty(slack, resting, 10.0)) == pytest.approx(0.0)


def test_a_violated_row_is_charged_the_square_of_its_violation():
    slack = jnp.asarray([-0.2])
    resting = jnp.zeros(1)
    charged = float(augmented_penalty(slack, resting, 10.0))

    assert charged == pytest.approx(0.5 * 10.0 * 0.2**2)


def test_an_active_row_is_priced_at_its_own_multiplier():
    # The whole point of the shift: at zero slack the derivative in the row is
    # minus its multiplier, whatever the penalty, so first-order optimality of
    # the original problem is recovered without driving the penalty up.
    priced = jax.grad(augmented_penalty)(jnp.zeros(1), jnp.asarray([3.5]), 10.0)

    assert float(priced[0]) == pytest.approx(-3.5)


def test_the_price_of_an_active_row_does_not_move_with_the_penalty():
    steep = jax.grad(augmented_penalty)(jnp.zeros(1), jnp.asarray([3.5]), 1.0e6)

    assert float(steep[0]) == pytest.approx(-3.5)


def test_a_multiplier_never_goes_negative():
    # A row satisfied with room to spare would otherwise be paid to stay away.
    shifted = shifted_multipliers(np.zeros(3), np.array([1.0, 2.0, 3.0]), 10.0, 1e8)

    assert np.all(shifted == 0.0)


def test_a_violated_row_raises_its_multiplier_in_proportion():
    shifted = shifted_multipliers(np.zeros(1), np.array([-0.5]), 10.0, 1e8)

    assert float(shifted[0]) == pytest.approx(5.0)


def test_a_multiplier_is_capped_at_its_ceiling():
    shifted = shifted_multipliers(np.zeros(1), np.array([-1.0]), 1.0e9, 100.0)

    assert float(shifted[0]) == pytest.approx(100.0)


def test_a_point_outside_the_domain_is_worse_than_the_anchor():
    anchor = np.array([1.0, 1.0])
    strayed = np.array([5.0, 4.0])
    value, _ = strayed_point(strayed, anchor, 2.0)

    assert value > 2.0


def test_the_gradient_outside_the_domain_points_back_inside():
    anchor = np.array([1.0, 1.0])
    strayed = np.array([5.0, 4.0])
    _, slope = strayed_point(strayed, anchor, 2.0)
    homeward = anchor - strayed

    assert float(np.dot(-slope, homeward)) > 0.0


def test_the_descent_lands_on_the_constraint_surface():
    maps = constrained_maps(summed, hyperbola)
    answer = descend_augmented(maps, CORNER_START, CORNER_BOXES, CORNER_BUDGET)

    assert np.allclose(answer.variables, CORNER, atol=1e-5)
    assert float(answer.masses[-1]) == pytest.approx(2.0, abs=1e-5)


def test_the_landing_sits_at_the_row_rather_than_inside_it():
    # A plain penalty stops short of the surface; the shift is what does not.
    maps = constrained_maps(summed, hyperbola)
    answer = descend_augmented(maps, CORNER_START, CORNER_BOXES, CORNER_BUDGET)
    rows = np.asarray(maps.slack(jnp.asarray(answer.variables)))

    assert abs(float(rows[0])) < 1e-6


def test_the_descent_reports_the_violation_of_every_round():
    maps = constrained_maps(summed, hyperbola)
    answer = descend_augmented(maps, CORNER_START, CORNER_BOXES, CORNER_BUDGET)

    assert answer.violations.size == answer.masses.size
    assert float(answer.violations[-1]) <= CORNER_BUDGET.tolerance


def test_the_descent_says_so_when_it_stopped_because_it_was_done():
    maps = constrained_maps(summed, hyperbola)
    answer = descend_augmented(maps, CORNER_START, CORNER_BOXES, CORNER_BUDGET)

    assert answer.converged


def test_a_row_the_answer_does_not_need_costs_nothing():
    # The bowl's minimum is deep inside the feasible region, so the row is
    # slack at the answer and the search must land on the bowl and not on it.
    def bowled(x):
        return jnp.sum((x - jnp.asarray([4.0, 4.0])) ** 2)

    maps = constrained_maps(bowled, hyperbola)
    answer = descend_augmented(maps, CORNER_START, CORNER_BOXES, CORNER_BUDGET)

    assert np.allclose(answer.variables, np.array([4.0, 4.0]), atol=1e-5)


def test_the_descent_is_repeatable():
    maps = constrained_maps(summed, hyperbola)
    first = descend_augmented(maps, CORNER_START, CORNER_BOXES, CORNER_BUDGET)
    again = descend_augmented(maps, CORNER_START, CORNER_BOXES, CORNER_BUDGET)

    assert np.array_equal(first.variables, again.variables)


def test_a_trial_point_outside_the_model_is_walked_back_in():
    # Stands in for a geometry whose frame will not factorize: the objective
    # raises there rather than returning a number, and the descent has to
    # answer that without either crashing or accepting the point.
    refused = {"count": 0}
    inside = constrained_maps(summed, hyperbola)

    def walled(x, multipliers, penalty, reference):
        # Concrete, outside the trace, which is where a solver's own
        # factorization check raises from.
        if float(x[0]) < 0.9:
            refused["count"] += 1
            raise ValueError("outside the model's domain")

        return inside.augmented(x, multipliers, penalty, reference)

    maps = inside._replace(augmented=walled)
    answer = descend_augmented(maps, CORNER_START, CORNER_BOXES, CORNER_BUDGET)

    assert refused["count"] > 0
    assert float(answer.variables[0]) >= 0.9
    assert np.isfinite(answer.masses[-1])


def test_a_runtime_error_from_a_compiled_solver_is_caught_too():
    # The commonest way a frame fails is detected inside a compiled program and
    # reported through a host callback, which surfaces as a runtime error and
    # not as anything about a value. Catching the value errors alone left the
    # real failure mode unhandled.
    refused = {"count": 0}
    inside = constrained_maps(summed, hyperbola)

    def failing(x, multipliers, penalty, reference):
        if float(x[0]) < 0.9:
            refused["count"] += 1
            raise RuntimeError("the linear solve returned a non-finite solution")

        return inside.augmented(x, multipliers, penalty, reference)

    maps = inside._replace(augmented=failing)
    answer = descend_augmented(maps, CORNER_START, CORNER_BOXES, CORNER_BUDGET)

    assert refused["count"] > 0
    assert float(answer.variables[0]) >= 0.9
    assert np.isfinite(answer.masses[-1])


def test_a_non_finite_objective_is_treated_as_outside_the_model():
    # A solver that reports a NaN rather than raising would otherwise poison
    # every curvature estimate taken after it.
    def poisoned(x, multipliers, penalty, reference):
        penalized = augmented_penalty(hyperbola(x), multipliers, penalty)
        value = summed(x) / reference + penalized

        return jnp.where(x[0] < 0.9, jnp.nan, value)

    maps = ConstrainedMaps(
        jax.jit(jax.value_and_grad(poisoned)),
        jax.jit(jax.value_and_grad(summed)),
        jax.jit(hyperbola),
    )
    answer = descend_augmented(maps, CORNER_START, CORNER_BOXES, CORNER_BUDGET)

    assert np.all(np.isfinite(answer.variables))
    assert np.isfinite(answer.masses[-1])


@pytest.mark.parametrize(
    "spoiled",
    [
        {"rounds": 0},
        {"iterations": 0},
        {"settled": 0},
        {"penalty": 0.0},
        {"growth": 1.0},
        {"ceiling": 0.5},
    ],
)
def test_the_descent_refuses_a_budget_it_cannot_use(spoiled):
    maps = constrained_maps(summed, hyperbola)
    budget = CORNER_BUDGET._replace(**spoiled)

    with pytest.raises(ValueError):
        descend_augmented(maps, CORNER_START, CORNER_BOXES, budget)
