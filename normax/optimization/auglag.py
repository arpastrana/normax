# SPDX-License-Identifier: Apache-2.0
"""
Constrained minimization by an augmented Lagrangian, in box bounds.

The inequality rows are aggregated inside the traced program, so the whole
constraint set costs one reverse pass per gradient — the formulation that lets
a member-by-member, case-by-case check cross a remote boundary as a single
cotangent. Nothing here knows what a design is.
"""

from collections.abc import Callable
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from jaxtyping import Float
from scipy.optimize import minimize

# How much worse than the last evaluable point a point outside the model's
# domain is reported as, so that no line search prefers one.
RECOIL_GROWTH = 1e3

# A round is solved only as accurately as the violation it inherited.
INNER_SHARE = 0.1
INNER_FLOOR = 1e-10

# A round has earned its keep if it took this share off the worst violation.
EARNED_SHARE = 0.25


class OptimizationBudget(NamedTuple):
    """
    What an augmented Lagrangian descent may spend, and when it stops.

    Attributes
    ----------
    rounds_max :
        Most multiplier updates to spend.
    iterations_warmup :
        Most inner iterations in each warmup round.
    iterations_after_warmup :
        Most inner iterations in every round after the warmup ones.
    rounds_warmup :
        How many of the first rounds get the warmup budget of iterations.
    penalty_start :
        Penalty parameter of the first round.
    penalty_growth :
        What the penalty is multiplied by when a round fails to earn its share
        of the violation it inherited.
    penalty_cap :
        Largest penalty the loop may reach, and the largest multiplier with it.
    violation_tol :
        Violation at or under which the rows count as satisfied.
    objective_rtol :
        Relative movement of the objective between rounds that counts as none.

    Notes
    -----
    A small starting penalty decides the answer rather than the speed: it leaves
    the objective in charge of the first rounds, so the search crosses the
    infeasible region and returns to the constraint surface somewhere a method
    confined to feasible points cannot reach.
    """

    rounds_max: int
    iterations_warmup: int
    iterations_after_warmup: int
    rounds_warmup: int
    penalty_start: float
    penalty_growth: float
    penalty_cap: float
    violation_tol: float
    objective_rtol: float


class OptimizationSolution(NamedTuple):
    """
    What an augmented Lagrangian descent arrived at, and the road there.

    Attributes
    ----------
    parameters :
        The design parameters the loop stopped on.
    objectives :
        Objective at the end of every round, the starting value first.
    violations :
        Worst violation over the rows at the end of every round.
    evaluations :
        Objective evaluations spent over every round.
    converged :
        Whether the loop stopped because the rows were satisfied and the
        objective had stopped moving, rather than on its round budget.

    Notes
    -----
    The two columns are read together. An objective falling while the violation
    is still large is the search spending the infeasible region, and only a row
    with the violation under the tolerance is a design.
    """

    parameters: Float[np.ndarray, "variables"]
    objectives: Float[np.ndarray, "rounds"]
    violations: Float[np.ndarray, "rounds"]
    evaluations: int
    converged: bool


class ConstrainedMaps(NamedTuple):
    """
    The three compiled programs a constrained descent calls.

    Attributes
    ----------
    augmented_lagrangian :
        Value and gradient of the augmented objective in the variables, taking
        the multipliers, the penalty and the objective's reference beside them.
    objective :
        Value and gradient of the objective alone.
    slack :
        How far above zero every inequality row sits.

    Notes
    -----
    The multipliers, the penalty and the reference are arguments of the
    augmented program rather than constants captured in it, so one compilation
    covers the whole outer loop.
    """

    augmented_lagrangian: Callable
    objective: Callable
    slack: Callable


def compute_penalty(
    slack: Float[Array, "constraints"],
    multipliers: Float[Array, "constraints"],
    penalty: float | Float[Array, ""],
) -> Float[Array, ""]:
    """
    Shifted quadratic penalty of a set of inequality rows.

    Parameters
    ----------
    slack :
        How far above zero every row sits. A negative entry is a violation.
    multipliers :
        Current estimate of the multiplier of every row, never negative.
    penalty :
        Penalty parameter of the round.

    Returns
    -------
    penalized :
        What the rows add to the objective at this multiplier estimate.

    Notes
    -----
    The shift is what makes this an augmented Lagrangian rather than a penalty:
    first-order optimality is recovered at a finite penalty, so a row that
    governs ends at zero slack rather than a little inside it. That matters for
    a fully-stressed design, whose answer sits on the constraint surface.
    """
    shifted = jnp.minimum(slack - multipliers / penalty, 0.0)

    return 0.5 * penalty * jnp.sum(shifted**2)


def update_multipliers(
    multipliers: Float[np.ndarray, "constraints"],
    slack: Float[np.ndarray, "constraints"],
    penalty: float,
    penalty_cap: float,
) -> Float[np.ndarray, "constraints"]:
    """
    The multiplier estimates a round of the outer loop leaves behind.

    Parameters
    ----------
    multipliers :
        Estimate the round was solved at.
    slack :
        How far above zero every row sits at the round's answer.
    penalty :
        Penalty parameter the round was solved at.
    penalty_cap :
        Largest value any multiplier may take.

    Returns
    -------
    shifted :
        The estimate the next round is solved at.
    """
    raised = multipliers - penalty * slack

    return np.clip(raised, 0.0, penalty_cap)


def recoil_point_to_last_good(
    x: Float[np.ndarray, "variables"],
    last_good: Float[np.ndarray, "variables"],
    held: float,
) -> tuple[float, Float[np.ndarray, "variables"]]:
    """
    A value and a gradient that walk a line search back into the model's domain.

    Parameters
    ----------
    x :
        The trial point that could not be evaluated.
    last_good :
        The last point that could be, which the walk heads back towards.
    held :
        Objective at that last_good.

    Returns
    -------
    strayed :
        The value to report, and the gradient to report with it.

    Notes
    -----
    A distant quadratic centered on the last_good: strictly worse than the last_good
    and pointing home, evaluated only at points the search goes on to reject.
    """
    strayed = np.asarray(x, dtype=np.float64) - last_good
    scale = max(abs(held), 1.0)
    value = RECOIL_GROWTH * scale + 0.5 * float(strayed @ strayed)

    return value, strayed


def measure_violation(
    slack: Callable[[Float[Array, "variables"]], Float[Array, "constraints"]],
    x: Float[np.ndarray, "variables"],
) -> tuple[float, Float[np.ndarray, "constraints"]]:
    """
    How far the worst row falls below zero, and every row with it.

    Parameters
    ----------
    slack :
        How far above zero every inequality row sits.
    x :
        The point to read the rows at.

    Returns
    -------
    read :
        The worst violation, never negative, and the rows themselves.
    """
    rows = np.asarray(slack(jnp.asarray(x)), dtype=np.float64)
    violation = -min(float(rows.min()), 0.0)

    return violation, rows


def optimize_augmented_lagrangian(
    maps: ConstrainedMaps,
    start: Float[np.ndarray, "variables"],
    boxes: list[tuple[float | None, float | None]],
    budget: OptimizationBudget,
) -> OptimizationSolution:
    """
    Minimize under inequality rows by an augmented Lagrangian, in box bounds.

    Parameters
    ----------
    maps :
        The compiled programs.
    start :
        The variable vector to leave from, inside the model's domain.
    boxes :
        One bound pair per variable, held natively by the inner solver.
    budget :
        Rounds, inner iterations, the penalty schedule and the stopping rules.

    Returns
    -------
    answer :
        The variables, the objective and violation of every round, and how it
        ended.

    Raises
    ------
    ValueError
        If a budget is not usable, or the objective at the start is not a
        positive finite number.

    Notes
    -----
    Each round is one L-BFGS-B descent of the augmented objective, solved to
    precision in the warmup rounds and only as far as the inherited violation
    afterwards. A trial point that raises or returns a non-finite number is
    charged `recoil_point_to_last_good`; `RuntimeError` is caught alongside the value
    errors because a solver failing inside a compiled program surfaces through
    a host callback as one. Deterministic: two runs of one budget agree bit
    for bit.
    """
    counts = (
        budget.rounds_max,
        budget.iterations_warmup,
        budget.iterations_after_warmup,
    )
    if min(counts) < 1:
        raise ValueError(f"rounds and iterations must be positive, got {budget}")
    if budget.penalty_start <= 0.0 or budget.penalty_cap < budget.penalty_start:
        raise ValueError(f"the penalty must be positive and under its cap: {budget}")
    if budget.penalty_growth <= 1.0:
        raise ValueError(f"the penalty must grow, got {budget.penalty_growth}")

    x = np.asarray(start, dtype=np.float64)
    reference = abs(float(maps.objective(jnp.asarray(x))[0]))
    if not np.isfinite(reference) or reference == 0.0:
        raise ValueError(f"the objective at the start is not usable: {reference}")

    scale = jnp.asarray(reference)
    violation, rows = measure_violation(maps.slack, x)
    multipliers = np.zeros(rows.size)
    penalty = float(budget.penalty_start)

    resting = jnp.zeros(rows.size)
    opened = maps.augmented_lagrangian(jnp.asarray(x), resting, scale, scale)[0]
    last_good = x.copy()
    held = float(opened)

    def evaluate_augmented_lagrangian(z, carried, charged):
        nonlocal last_good, held
        try:
            value, slope = maps.augmented_lagrangian(
                jnp.asarray(z), carried, charged, scale
            )
            value = float(value)
            slope = np.asarray(slope, dtype=np.float64)
        except (ValueError, FloatingPointError, RuntimeError):
            return recoil_point_to_last_good(z, last_good, held)
        if not np.isfinite(value) or not np.all(np.isfinite(slope)):
            return recoil_point_to_last_good(z, last_good, held)
        last_good = np.asarray(z, dtype=np.float64).copy()
        held = value

        return value, slope

    objectives = [reference]
    violations = [violation]
    spent = 0
    converged = False
    inherited = violation

    for round_index in range(budget.rounds_max):
        carried = jnp.asarray(multipliers)
        charged = jnp.asarray(penalty)
        if round_index < budget.rounds_warmup:
            inner = budget.iterations_warmup
            precision = INNER_FLOOR
        else:
            inner = budget.iterations_after_warmup
            precision = max(INNER_FLOOR, INNER_SHARE * inherited)

        def round_objective(z, carried=carried, charged=charged):
            return evaluate_augmented_lagrangian(z, carried, charged)

        options = {
            "maxiter": inner,
            "maxfun": 3 * inner,
            "ftol": 0.0,
            "gtol": precision,
        }
        found = minimize(
            round_objective,
            x,
            jac=True,
            method="L-BFGS-B",
            bounds=boxes,
            options=options,
        )
        x = np.asarray(found.x, dtype=np.float64)
        spent += int(found.nfev)

        violation, rows = measure_violation(maps.slack, x)
        objective = abs(float(maps.objective(jnp.asarray(x))[0]))
        moved = abs(objective - objectives[-1]) / max(objective, reference)
        objectives.append(objective)
        violations.append(violation)

        multipliers = update_multipliers(multipliers, rows, penalty, budget.penalty_cap)
        if violation > EARNED_SHARE * inherited:
            penalty = min(penalty * budget.penalty_growth, budget.penalty_cap)
        inherited = violation

        if violation <= budget.violation_tol and moved <= budget.objective_rtol:
            converged = True
            break

    answer = OptimizationSolution(
        x, np.asarray(objectives), np.asarray(violations), spent, converged
    )

    return answer
