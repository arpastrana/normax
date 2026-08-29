# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from normax.optimization import ConstrainedMaps
from normax.optimization import OptimizationBudget
from normax.optimization import compute_penalty
from normax.optimization import descend_augmented_lagrangian
from normax.optimization import recoil_point_to_last_good
from normax.optimization import update_multipliers

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
)


def summed(x):
    return x[0] + x[1]


def hyperbola(x):
    return jnp.atleast_1d(x[0] * x[1] - 1.0)


def constrained_maps(weigh, slack):
    def augmented_lagrangian(x, multipliers, penalty, reference):
        penalized = compute_penalty(slack(x), multipliers, penalty)

        return weigh(x) / reference + penalized

    return ConstrainedMaps(
        jax.jit(jax.value_and_grad(augmented_lagrangian)),
        jax.jit(jax.value_and_grad(weigh)),
        jax.jit(slack),
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
    answer = descend_augmented_lagrangian(
        maps, CORNER_START, CORNER_BOXES, CORNER_BUDGET
    )

    assert np.allclose(answer.parameters, CORNER, atol=1e-5)
    assert float(answer.objectives[-1]) == pytest.approx(2.0, abs=1e-5)


def test_the_landing_sits_at_the_row_rather_than_inside_it():
    # A plain penalty stops short of the surface; the shift is what does not.
    maps = constrained_maps(summed, hyperbola)
    answer = descend_augmented_lagrangian(
        maps, CORNER_START, CORNER_BOXES, CORNER_BUDGET
    )
    rows = np.asarray(maps.slack(jnp.asarray(answer.parameters)))

    assert abs(float(rows[0])) < 1e-6


def test_the_descent_reports_the_violation_of_every_round():
    maps = constrained_maps(summed, hyperbola)
    answer = descend_augmented_lagrangian(
        maps, CORNER_START, CORNER_BOXES, CORNER_BUDGET
    )

    assert answer.violations.size == answer.objectives.size
    assert float(answer.violations[-1]) <= CORNER_BUDGET.violation_tol


def test_the_descent_says_so_when_it_stopped_because_it_was_done():
    maps = constrained_maps(summed, hyperbola)
    answer = descend_augmented_lagrangian(
        maps, CORNER_START, CORNER_BOXES, CORNER_BUDGET
    )

    assert answer.converged


def test_a_row_the_answer_does_not_need_costs_nothing():
    # The bowl's minimum is deep inside the feasible region, so the row is
    # slack at the answer and the search must land on the bowl and not on it.
    def bowled(x):
        return jnp.sum((x - jnp.asarray([4.0, 4.0])) ** 2)

    maps = constrained_maps(bowled, hyperbola)
    answer = descend_augmented_lagrangian(
        maps, CORNER_START, CORNER_BOXES, CORNER_BUDGET
    )

    assert np.allclose(answer.parameters, np.array([4.0, 4.0]), atol=1e-5)


def test_the_descent_is_repeatable():
    maps = constrained_maps(summed, hyperbola)
    first = descend_augmented_lagrangian(
        maps, CORNER_START, CORNER_BOXES, CORNER_BUDGET
    )
    again = descend_augmented_lagrangian(
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
    answer = descend_augmented_lagrangian(
        maps, CORNER_START, CORNER_BOXES, CORNER_BUDGET
    )

    assert refused["count"] > 0
    assert float(answer.parameters[0]) >= 0.9
    assert np.isfinite(answer.objectives[-1])


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
    answer = descend_augmented_lagrangian(
        maps, CORNER_START, CORNER_BOXES, CORNER_BUDGET
    )

    assert refused["count"] > 0
    assert float(answer.parameters[0]) >= 0.9
    assert np.isfinite(answer.objectives[-1])


def test_a_non_finite_objective_is_treated_as_outside_the_model():
    # A solver that reports a NaN rather than raising would otherwise poison
    # every curvature estimate taken after it.
    def poisoned(x, multipliers, penalty, reference):
        penalized = compute_penalty(hyperbola(x), multipliers, penalty)
        value = summed(x) / reference + penalized

        return jnp.where(x[0] < 0.9, jnp.nan, value)

    maps = ConstrainedMaps(
        jax.jit(jax.value_and_grad(poisoned)),
        jax.jit(jax.value_and_grad(summed)),
        jax.jit(hyperbola),
    )
    answer = descend_augmented_lagrangian(
        maps, CORNER_START, CORNER_BOXES, CORNER_BUDGET
    )

    assert np.all(np.isfinite(answer.parameters))
    assert np.isfinite(answer.objectives[-1])


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
        descend_augmented_lagrangian(maps, CORNER_START, CORNER_BOXES, budget)
