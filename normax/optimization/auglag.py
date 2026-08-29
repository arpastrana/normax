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
from jaxtyping import Int
from scipy.optimize import minimize
from tqdm.auto import tqdm

# How much worse than the last evaluable point a point outside the model's
# domain is reported as, so that no line search prefers one.
RECOIL_GROWTH = 1e3

# A round is solved only as accurately as the violation it inherited.
INNER_SHARE = 0.1
INNER_FLOOR = 1e-10

# A round has earned its keep if it took this share off the worst violation.
EARNED_SHARE = 0.25

# A geometry whose shortest member falls to this share of the one the search
# started from has collapsed rather than merely broken a length constraint.
# Small on purpose: the guard exists to keep a solver from being handed a
# member of no length, not to keep the search away from short ones. Anything
# larger walls off the region a line search must probe to escape a constrained
# round, and the recoil it charges there poisons the curvature estimate.
DEGENERATE_SHARE = 1e-9


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
    trace_iterations :
        Whether to record every inner iteration and not only every round. The
        walk itself is free, but reading the objective and the rows at each
        point it passed through costs two evaluations per iteration, paid once
        after the descent has landed.

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
    trace_iterations: bool


class DescentHistory(NamedTuple):
    """
    Where a descent went, in the order it went there.

    Attributes
    ----------
    iterates :
        The variable vector at every point, the start first.
    objectives :
        Objective at every point.
    violations :
        Worst violation over the rows at every point.
    round_index :
        Which outer round each point came out of, the start at zero and a
        round's own points numbered from one, so this indexes the round
        history whatever resolution the walk was recorded at.

    Notes
    -----
    One row per point in all four columns, and the last row is the answer. The
    same container holds a walk recorded per round and one recorded per inner
    iteration, which is what lets a reader take the finer of the two without
    knowing which it got.
    """

    iterates: Float[np.ndarray, "steps variables"]
    objectives: Float[np.ndarray, "steps"]
    violations: Float[np.ndarray, "steps"]
    round_index: Int[np.ndarray, "steps"]


class OptimizationSolution(NamedTuple):
    """
    What an augmented Lagrangian descent arrived at, and the road there.

    Attributes
    ----------
    parameters :
        The design parameters the loop stopped on.
    rounds :
        The walk read at the end of every round, always recorded.
    iterations :
        The same walk read at every inner iteration, or None where the budget
        asked for no such record.
    evaluations :
        Objective evaluations spent over every round, the tracing pass aside.
    converged :
        Whether the loop stopped because the rows were satisfied and the
        objective had stopped moving, rather than on its round budget.

    Notes
    -----
    A round is one L-BFGS-B descent, so the coarse walk has a point where the
    multipliers moved and nowhere else: on a converged run that is a handful of
    points, enough to read convergence off and too few to watch a design
    change. The two resolutions answer those two questions, and the fine one is
    absent rather than empty when it was not asked for.
    """

    parameters: Float[np.ndarray, "variables"]
    rounds: DescentHistory
    iterations: DescentHistory | None
    evaluations: int
    converged: bool


class ConstrainedMaps(NamedTuple):
    """
    The compiled programs a constrained descent calls.

    Attributes
    ----------
    augmented_lagrangian :
        Value and gradient of the augmented objective in the variables, taking
        the multipliers, the penalty and the objective's reference beside them.
    objective :
        Value and gradient of the objective alone.
    slack :
        How far above zero every inequality row sits.
    readings :
        Objective and rows together at a point, and no gradient.
    shortest :
        Length of the shortest member of the geometry a point stands for.

    Notes
    -----
    The multipliers, the penalty and the reference are arguments of the
    augmented program rather than constants captured in it, so one compilation
    covers the whole outer loop.

    The descent itself never calls `readings`: it exists for reading a walk
    back afterwards, where a gradient is waste and the objective and the rows
    are wanted at the same point. Asking for them together is what lets the
    two share a form finding rather than each paying for one, and being
    compiled lazily it costs nothing on a run that reads no walk.

    `shortest` is read before every trial point is evaluated, and reads the
    geometry alone rather than the whole pipeline. It is what stands between
    the line search and a collapsed member: a solver handed one does not raise,
    it dies, and a dead process is the one failure a recoil cannot catch.
    """

    augmented_lagrangian: Callable
    objective: Callable
    slack: Callable
    readings: Callable
    shortest: Callable


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
    violation = max(-float(rows.min()), 0.0)

    return violation, rows


def measure_history(
    maps: ConstrainedMaps,
    iterates: list[Float[np.ndarray, "variables"]],
    round_index: Int[np.ndarray, "steps"],
) -> DescentHistory:
    """
    The objective and the violation at every point a descent passed through.

    Parameters
    ----------
    maps :
        The compiled programs, read for the objective and the rows.
    iterates :
        The variable vector at every point, in the order it was reached.
    round_index :
        Which outer round each point came out of.

    Returns
    -------
    history :
        The walk, its objective and violation read at every point.

    Notes
    -----
    Read after the descent has landed rather than inside it, so a line search
    pays nothing for the record and the loop is the same program whether or not
    one was asked for. One forward pass a point through `readings`, which is
    what the finer resolution costs.
    """
    values = []
    gaps = []
    for step in iterates:
        value, rows = maps.readings(jnp.asarray(step))
        values.append(abs(float(value)))
        gaps.append(max(-float(np.asarray(rows).min()), 0.0))

    history = DescentHistory(
        np.stack(iterates),
        np.asarray(values),
        np.asarray(gaps),
        np.asarray(round_index),
    )

    return history


def optimize_augmented_lagrangian(
    maps: ConstrainedMaps,
    start: Float[np.ndarray, "variables"],
    boxes: list[tuple[float | None, float | None]],
    budget: OptimizationBudget,
    progress: bool = False,
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
    progress :
        Whether to draw a progress bar, one step per round, carrying the
        objective and the violation as they stand.

    Returns
    -------
    answer :
        The variables it stopped on, the walk it took to get there at one
        resolution or two, and how it ended.

    Raises
    ------
    ValueError
        If a budget is not usable, if the objective at the start is not a
        positive finite number, or if the geometry at the start has already
        collapsed and so gives no length for the guard to measure against.

    Notes
    -----
    Each round is one L-BFGS-B descent of the augmented objective, solved to
    precision in the warmup rounds and only as far as the inherited violation
    afterwards. A trial point that raises or returns a non-finite number is
    charged `recoil_point_to_last_good`; `RuntimeError` is caught alongside the value
    errors because a solver failing inside a compiled program surfaces through
    a host callback as one. A point whose geometry has collapsed is charged the
    same way and never evaluated at all, since a frame solver handed a member of
    no length does not raise, it takes the process down with it, and that is the
    one failure no `except` reaches. Deterministic: two runs of one budget agree
    bit for bit. A budget that asks for the finer record changes what is read off
    the descent, never the descent, since the walk is measured after it lands.
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

    walking = tqdm(
        total=budget.rounds_max,
        disable=not progress,
        unit="round",
        desc="auglag",
        leave=False,
    )
    counted = 0

    drawn = float(maps.shortest(jnp.asarray(x)))
    if not np.isfinite(drawn) or drawn <= 0.0:
        raise ValueError(f"the geometry at the start is degenerate: {drawn}")
    collapsed = DEGENERATE_SHARE * drawn

    def evaluate_augmented_lagrangian(z, carried, charged):
        nonlocal last_good, held, counted
        counted += 1
        walking.set_postfix(
            objective=f"{objectives[-1]:.6f}",
            violation=f"{violations[-1]:.2e}",
            evaluations=counted,
            refresh=False,
        )
        # update(0) advances nothing and redraws only when tqdm's own interval
        # has passed, so a piped run does not get a line per evaluation.
        walking.update(0)
        reached = float(maps.shortest(jnp.asarray(z)))
        if not np.isfinite(reached) or reached <= collapsed:
            return recoil_point_to_last_good(z, last_good, held)
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

    iterates = [x.copy()]
    objectives = [reference]
    violations = [violation]
    walked = [x.copy()]
    walked_rounds = [0]
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

        def record_step(z, numbered=round_index + 1):
            walked.append(np.asarray(z, dtype=np.float64).copy())
            walked_rounds.append(numbered)

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
            callback=record_step if budget.trace_iterations else None,
        )
        x = np.asarray(found.x, dtype=np.float64)
        spent += int(found.nfev)

        # L-BFGS-B does not call back on the point it returns when the round
        # ends on its budget rather than on its own test, and a round that
        # moves nowhere calls back not at all -- which would leave the walk
        # with no point to carry that round's number.
        if budget.trace_iterations:
            recorded = walked_rounds[-1] == round_index + 1
            if not recorded or not np.array_equal(walked[-1], x):
                record_step(x)

        violation, rows = measure_violation(maps.slack, x)
        objective = abs(float(maps.objective(jnp.asarray(x))[0]))
        moved = abs(objective - objectives[-1]) / max(objective, reference)
        iterates.append(x.copy())
        objectives.append(objective)
        violations.append(violation)

        walking.update(1)
        walking.set_postfix(
            objective=f"{objective:.6f}",
            violation=f"{violation:.2e}",
            evaluations=counted,
        )

        multipliers = update_multipliers(multipliers, rows, penalty, budget.penalty_cap)
        if violation > EARNED_SHARE * inherited:
            penalty = min(penalty * budget.penalty_growth, budget.penalty_cap)
        inherited = violation

        if violation <= budget.violation_tol and moved <= budget.objective_rtol:
            converged = True
            break

    walking.close()

    coarse = DescentHistory(
        np.stack(iterates),
        np.asarray(objectives),
        np.asarray(violations),
        np.arange(len(iterates)),
    )
    fine = None
    if budget.trace_iterations:
        fine = measure_history(maps, walked, np.asarray(walked_rounds))

    answer = OptimizationSolution(x, coarse, fine, spent, converged)

    return answer
