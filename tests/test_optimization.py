# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from normax.optimization import DEGENERATE_SHARE
from normax.optimization import ConstrainedMaps
from normax.optimization import DescentHistory
from normax.optimization import OptimizationBudget
from normax.optimization import compute_penalty
from normax.optimization import optimize_augmented_lagrangian
from normax.optimization import recoil_point_to_last_good
from normax.optimization import update_multipliers
from normax.visualization import read_round_bounds
from normax.visualization import track_best_feasible

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
CORNER_BUDGET = OptimizationBudget(
    rounds_max=25,
    iterations_warmup=200,
    iterations_after_warmup=100,
    rounds_warmup=2,
    penalty_start=1.0,
    penalty_growth=10.0,
    penalty_cap=1.0e8,
    violation_tol=1.0e-9,
    objective_rtol=1.0e-12,
    trace_iterations=False,
)


def summed(x):
    return x[0] + x[1]


def hyperbola(x):
    return jnp.atleast_1d(x[0] * x[1] - 1.0)


def constrained_maps(weigh, slack):
    def augmented_lagrangian(x, multipliers, penalty, reference):
        penalized = compute_penalty(slack(x), multipliers, penalty)

        return weigh(x) / reference + penalized

    def read_point(x):
        return weigh(x), slack(x)

    def read_shortest(x):
        # No geometry here, so nothing can collapse: a constant clears the guard.
        return jnp.asarray(1.0)

    return ConstrainedMaps(
        jax.jit(jax.value_and_grad(augmented_lagrangian)),
        jax.jit(jax.value_and_grad(weigh)),
        jax.jit(slack),
        jax.jit(read_point),
        jax.jit(read_shortest),
    )


def test_satisfied_rows_at_no_price_cost_nothing():
    slack = jnp.asarray([0.5, 1.0, 2.0])
    resting = jnp.zeros(3)

    assert float(compute_penalty(slack, resting, 10.0)) == pytest.approx(0.0)


def test_a_violated_row_is_charged_the_square_of_its_violation():
    slack = jnp.asarray([-0.2])
    resting = jnp.zeros(1)
    charged = float(compute_penalty(slack, resting, 10.0))

    assert charged == pytest.approx(0.5 * 10.0 * 0.2**2)


def test_an_active_row_is_priced_at_its_own_multiplier():
    # The whole point of the shift: at zero slack the derivative in the row is
    # minus its multiplier, whatever the penalty, so first-order optimality of
    # the original problem is recovered without driving the penalty up.
    priced = jax.grad(compute_penalty)(jnp.zeros(1), jnp.asarray([3.5]), 10.0)

    assert float(priced[0]) == pytest.approx(-3.5)


def test_the_price_of_an_active_row_does_not_move_with_the_penalty():
    steep = jax.grad(compute_penalty)(jnp.zeros(1), jnp.asarray([3.5]), 1.0e6)

    assert float(steep[0]) == pytest.approx(-3.5)


def test_a_multiplier_never_goes_negative():
    # A row satisfied with room to spare would otherwise be paid to stay away.
    shifted = update_multipliers(np.zeros(3), np.array([1.0, 2.0, 3.0]), 10.0, 1e8)

    assert np.all(shifted == 0.0)


def test_a_violated_row_raises_its_multiplier_in_proportion():
    shifted = update_multipliers(np.zeros(1), np.array([-0.5]), 10.0, 1e8)

    assert float(shifted[0]) == pytest.approx(5.0)


def test_a_multiplier_is_capped_at_its_ceiling():
    shifted = update_multipliers(np.zeros(1), np.array([-1.0]), 1.0e9, 100.0)

    assert float(shifted[0]) == pytest.approx(100.0)


def test_a_point_outside_the_domain_is_worse_than_the_anchor():
    last_good = np.array([1.0, 1.0])
    strayed = np.array([5.0, 4.0])
    value, _ = recoil_point_to_last_good(strayed, last_good, 2.0)

    assert value > 2.0


def test_the_gradient_outside_the_domain_points_back_inside():
    last_good = np.array([1.0, 1.0])
    strayed = np.array([5.0, 4.0])
    _, slope = recoil_point_to_last_good(strayed, last_good, 2.0)
    homeward = last_good - strayed

    assert float(np.dot(-slope, homeward)) > 0.0


def test_the_descent_lands_on_the_constraint_surface():
    maps = constrained_maps(summed, hyperbola)
    answer = optimize_augmented_lagrangian(
        maps, CORNER_START, CORNER_BOXES, CORNER_BUDGET
    )

    assert np.allclose(answer.parameters, CORNER, atol=1e-5)
    assert float(answer.rounds.objectives[-1]) == pytest.approx(2.0, abs=1e-5)


def test_the_landing_sits_at_the_row_rather_than_inside_it():
    # A plain penalty stops short of the surface; the shift is what does not.
    maps = constrained_maps(summed, hyperbola)
    answer = optimize_augmented_lagrangian(
        maps, CORNER_START, CORNER_BOXES, CORNER_BUDGET
    )
    rows = np.asarray(maps.slack(jnp.asarray(answer.parameters)))

    assert abs(float(rows[0])) < 1e-6


def test_the_descent_reports_the_violation_of_every_round():
    maps = constrained_maps(summed, hyperbola)
    answer = optimize_augmented_lagrangian(
        maps, CORNER_START, CORNER_BOXES, CORNER_BUDGET
    )

    assert answer.rounds.violations.size == answer.rounds.objectives.size
    assert float(answer.rounds.violations[-1]) <= CORNER_BUDGET.violation_tol


def test_the_descent_says_so_when_it_stopped_because_it_was_done():
    maps = constrained_maps(summed, hyperbola)
    answer = optimize_augmented_lagrangian(
        maps, CORNER_START, CORNER_BOXES, CORNER_BUDGET
    )

    assert answer.converged


def test_a_row_the_answer_does_not_need_costs_nothing():
    # The bowl's minimum is deep inside the feasible region, so the row is
    # slack at the answer and the search must land on the bowl and not on it.
    def bowled(x):
        return jnp.sum((x - jnp.asarray([4.0, 4.0])) ** 2)

    maps = constrained_maps(bowled, hyperbola)
    answer = optimize_augmented_lagrangian(
        maps, CORNER_START, CORNER_BOXES, CORNER_BUDGET
    )

    assert np.allclose(answer.parameters, np.array([4.0, 4.0]), atol=1e-5)


def test_the_descent_is_repeatable():
    maps = constrained_maps(summed, hyperbola)
    first = optimize_augmented_lagrangian(
        maps, CORNER_START, CORNER_BOXES, CORNER_BUDGET
    )
    again = optimize_augmented_lagrangian(
        maps, CORNER_START, CORNER_BOXES, CORNER_BUDGET
    )

    assert np.array_equal(first.parameters, again.parameters)


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

        return inside.augmented_lagrangian(x, multipliers, penalty, reference)

    maps = inside._replace(augmented_lagrangian=walled)
    answer = optimize_augmented_lagrangian(
        maps, CORNER_START, CORNER_BOXES, CORNER_BUDGET
    )

    assert refused["count"] > 0
    assert float(answer.parameters[0]) >= 0.9
    assert np.isfinite(answer.rounds.objectives[-1])


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

        return inside.augmented_lagrangian(x, multipliers, penalty, reference)

    maps = inside._replace(augmented_lagrangian=failing)
    answer = optimize_augmented_lagrangian(
        maps, CORNER_START, CORNER_BOXES, CORNER_BUDGET
    )

    assert refused["count"] > 0
    assert float(answer.parameters[0]) >= 0.9
    assert np.isfinite(answer.rounds.objectives[-1])


def test_a_non_finite_objective_is_treated_as_outside_the_model():
    # A solver that reports a NaN rather than raising would otherwise poison
    # every curvature estimate taken after it.
    def poisoned(x, multipliers, penalty, reference):
        penalized = compute_penalty(hyperbola(x), multipliers, penalty)
        value = summed(x) / reference + penalized

        return jnp.where(x[0] < 0.9, jnp.nan, value)

    maps = constrained_maps(summed, hyperbola)._replace(
        augmented_lagrangian=jax.jit(jax.value_and_grad(poisoned))
    )
    answer = optimize_augmented_lagrangian(
        maps, CORNER_START, CORNER_BOXES, CORNER_BUDGET
    )

    assert np.all(np.isfinite(answer.parameters))
    assert np.isfinite(answer.rounds.objectives[-1])


@pytest.mark.parametrize(
    "spoiled",
    [
        {"rounds_max": 0},
        {"iterations_warmup": 0},
        {"iterations_after_warmup": 0},
        {"penalty_start": 0.0},
        {"penalty_growth": 1.0},
        {"penalty_cap": 0.5},
    ],
)
def test_the_descent_refuses_a_budget_it_cannot_use(spoiled):
    maps = constrained_maps(summed, hyperbola)
    budget = CORNER_BUDGET._replace(**spoiled)

    with pytest.raises(ValueError):
        optimize_augmented_lagrangian(maps, CORNER_START, CORNER_BOXES, budget)


# --------------------------------------------------------------------------- #
# The curve a descent is read as
# --------------------------------------------------------------------------- #
def test_the_descent_carries_the_variables_of_every_round():
    maps = constrained_maps(summed, hyperbola)

    answer = optimize_augmented_lagrangian(
        maps, CORNER_START, CORNER_BOXES, CORNER_BUDGET
    )

    rounds = answer.rounds.objectives.size
    assert answer.rounds.iterates.shape == (rounds, CORNER_START.size)
    assert np.allclose(answer.rounds.iterates[0], CORNER_START)
    assert np.allclose(answer.rounds.iterates[-1], answer.parameters)


def test_the_best_feasible_curve_never_rises_and_reads_only_its_prefix():
    # A search that spends the infeasible region, lands, worsens, then improves.
    objective = np.array([9.0, 4.0, 2.0, 3.0, 1.5])
    violation = np.array([1.0, 1e-2, 1e-8, 1e-8, 0.0])

    best = track_best_feasible(objective, violation, 1e-6)

    # No design has been found while the rows are violated, and a gap says so.
    assert np.isnan(best[0])
    assert np.isnan(best[1])
    # Monotone from the first satisfied round on, so the rise at round 3 is
    # held rather than drawn.
    assert np.array_equal(best[2:], np.array([2.0, 2.0, 1.5]))

    # Prefix-closed: a round is unmoved by every round after it, which is what
    # lets one call on the whole run be sliced into frames.
    for cut in range(1, objective.size + 1):
        walked = track_best_feasible(objective[:cut], violation[:cut], 1e-6)
        assert np.array_equal(np.isnan(walked), np.isnan(best[:cut]))
        assert np.allclose(walked, best[:cut], equal_nan=True)


def test_the_best_feasible_curve_is_all_gap_when_nothing_is_satisfied():
    objective = np.array([9.0, 4.0, 2.0])
    violation = np.array([1.0, 0.5, 1e-3])

    best = track_best_feasible(objective, violation, 1e-6)

    assert np.all(np.isnan(best))


def test_the_finer_timeline_records_every_inner_iteration():
    maps = constrained_maps(summed, hyperbola)
    budget = CORNER_BUDGET._replace(trace_iterations=True)

    answer = optimize_augmented_lagrangian(maps, CORNER_START, CORNER_BOXES, budget)

    walked = answer.iterations
    assert walked is not None
    # A round is one L-BFGS-B descent, so the finer walk is the longer one.
    steps = walked.objectives.size
    assert steps > answer.rounds.objectives.size
    assert walked.iterates.shape == (steps, CORNER_START.size)
    assert walked.violations.shape == (steps,)
    assert np.allclose(walked.iterates[0], CORNER_START)
    assert np.allclose(walked.iterates[-1], answer.parameters)

    # Every point says which round it came out of, the start before them all,
    # and the numbering only ever climbs.
    numbered = walked.round_index
    assert numbered[0] == 0
    assert np.all(np.diff(numbered) >= 0)
    assert numbered[-1] == answer.rounds.objectives.size - 1
    # Every round carries a point, a round that moved nowhere included, so the
    # numbering skips nothing and the join onto the round history is total.
    assert np.array_equal(np.unique(numbered), np.arange(numbered[-1] + 1))


def test_tracing_the_iterations_moves_neither_the_answer_nor_the_rounds():
    maps = constrained_maps(summed, hyperbola)
    budget = CORNER_BUDGET._replace(trace_iterations=True)

    plain = optimize_augmented_lagrangian(
        maps, CORNER_START, CORNER_BOXES, CORNER_BUDGET
    )
    traced = optimize_augmented_lagrangian(maps, CORNER_START, CORNER_BOXES, budget)

    # The walk is measured after the descent lands, so the descent is the same
    # program either way and the record costs the answer nothing.
    assert plain.iterations is None
    assert np.array_equal(plain.parameters, traced.parameters)
    assert np.array_equal(plain.rounds.objectives, traced.rounds.objectives)
    assert np.array_equal(plain.rounds.violations, traced.rounds.violations)
    assert plain.evaluations == traced.evaluations


def test_the_two_resolutions_agree_where_they_meet():
    maps = constrained_maps(summed, hyperbola)
    budget = CORNER_BUDGET._replace(trace_iterations=True)

    answer = optimize_augmented_lagrangian(maps, CORNER_START, CORNER_BOXES, budget)

    # The last point of a round is that round's endpoint, which is what makes
    # the two histories one timeline read at two resolutions.
    walked = answer.iterations
    for index, objective in enumerate(answer.rounds.objectives):
        belongs = np.flatnonzero(walked.round_index == index)
        assert belongs.size > 0
        assert float(walked.objectives[belongs[-1]]) == pytest.approx(
            float(objective), rel=1e-12
        )


def _history(round_index):
    """
    A walk that carries nothing but the round each of its points came out of.
    """
    width = round_index.size
    blank = np.zeros(width)

    return DescentHistory(np.zeros((width, 1)), blank, blank, round_index)


def test_the_round_bounds_are_empty_on_a_walk_recorded_a_round_at_a_time():
    coarse = np.arange(5)

    crossings = read_round_bounds(_history(coarse))

    assert crossings.size == 0


def test_the_round_bounds_mark_the_first_point_of_every_later_round():
    # The start, then two points in round 1, one in round 2, three in round 3.
    fine = np.array([0, 1, 1, 2, 3, 3, 3])

    crossings = read_round_bounds(_history(fine))

    assert np.array_equal(crossings, np.array([1, 3, 4]))


# --------------------------------------------------------------------------- #
# The geometry a trial point stands for
# --------------------------------------------------------------------------- #
def test_a_collapsed_geometry_is_never_handed_to_the_program():
    # A frame solver given a member of no length does not raise, it takes the
    # process down, so the guard has to refuse the point before it is
    # evaluated rather than catch what evaluating it throws.
    maps = constrained_maps(summed, hyperbola)
    seen = []

    def watched(z, *carried):
        seen.append(float(np.asarray(z)[0]))

        return maps.augmented_lagrangian(z, *carried)

    # Shortest member 2.5 at the start, so the guard refuses anything at or
    # under 0.025 of it -- reachable, the box running down to 0.1.
    guarded = maps._replace(
        augmented_lagrangian=watched,
        shortest=lambda z: jnp.asarray(z)[0] - 0.5,
    )

    answer = optimize_augmented_lagrangian(
        guarded, CORNER_START, CORNER_BOXES, CORNER_BUDGET
    )

    assert seen
    assert min(seen) - 0.5 > DEGENERATE_SHARE * 2.5
    # The corner is well clear of the guard, so refusing those points costs
    # the answer nothing.
    assert np.allclose(answer.parameters, CORNER, atol=1e-4)


def test_a_descent_leaving_from_a_collapsed_geometry_is_refused():
    # Nothing to measure a collapse against, so the budget cannot be spent
    # safely and the loop says so rather than walking into the solver.
    maps = constrained_maps(summed, hyperbola)
    resting = maps._replace(shortest=lambda z: jnp.asarray(0.0))

    with pytest.raises(ValueError, match="degenerate"):
        optimize_augmented_lagrangian(
            resting, CORNER_START, CORNER_BOXES, CORNER_BUDGET
        )
