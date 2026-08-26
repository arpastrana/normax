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
Searching the force densities for the lightest structure the standard allows.

The objective is a mass and the variables are force densities, so the optimizer
never sees a diameter: the sizes are solved for inside the objective, at every
iterate, by the fully-stressed map. That is what makes this an unconstrained
scalar minimization over a handful of variables rather than a constrained one
over a size per member, and it is why box bounds are the only constraint here.

Two things are annealed rather than fixed. The envelope over load cases is
smooth, and its sharpness rises geometrically so the design approaches the
smallest adequate one from above instead of chattering between load cases. Nothing
else about the problem is relaxed: the check is exact at every iterate, and the
utilization the final design reports is the standard's own.

The gradient comes from the pipeline and costs one reverse pass whatever the
number of force densities, which is what makes a per-member design variable
affordable. The optimizer itself is L-BFGS-B, because the bounds are boxes and
the objective is smooth between the branches of the standard.

A second search lives here for the formulation where the sizes are variables
too and the check is an inequality per member and load case. It keeps the one
reverse pass by aggregating the rows inside the traced program — an augmented
Lagrangian, whose shift is what still leaves the answer on the constraint
surface where a plain penalty would stop a little inside it. Only the box
bounds reach the inner solver as bounds; everything else is in the objective.
"""

from collections.abc import Callable
from collections.abc import Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import PyTree
from scipy.optimize import minimize

# What an objective returns: a mass, or a mass and whatever was computed with it.
ObjectiveValue = Float[Array, ""] | tuple[Float[Array, ""], PyTree]
ValueAndGradient = Callable[..., tuple[ObjectiveValue, Float[Array, "members"]]]


class Trajectory(NamedTuple):
    """
    Where the optimizer went, in the order it went there.

    Attributes
    ----------
    q :
        Force densities at every iterate, the starting point included.
    mass :
        Objective at every iterate.
    beta :
        Sharpness of the envelope the iterate was taken under.

    Notes
    -----
    One row per iteration and not per function evaluation, so the line search
    inside an iteration is invisible here. The last row is the answer.

    The sharpness column is what makes a trajectory over an annealing schedule
    readable: the objective steps up whenever it rises, because a sharper
    envelope is a smaller number over the same design, and a mass that falls
    across a step is a real improvement rather than an artefact of the
    smoothing.
    """

    q: Float[Array, "steps members"]
    mass: Float[Array, "steps"]
    beta: Float[Array, "steps"]


class SearchResult(NamedTuple):
    """
    The answer a search returned, and everything computed at it.

    Attributes
    ----------
    value :
        Objective at the answer, the last entry of the trajectory's own column.
    aux :
        Whatever the objective returned alongside its value there. None if it
        returned nothing but a value.
    trajectory :
        Where the optimizer went, in the order it went there.

    Notes
    -----
    **The point of the aux field is that the answer is not recomputed.** A search
    over a pipeline ends holding a force density and nothing else, and the design
    behind it — the shape, the forces, the sections, the utilization — is what a
    caller actually wants to report on. Rebuilding it afterwards costs a second
    trace and a second compilation of a program the search already ran hundreds
    of times, which is far more than the forward pass it looks like.

    The objective returns it instead, as `(value, aux)` under `has_aux`, and the
    search carries out the one belonging to the point it answered with. Nothing
    here knows what the aux is: to this module it is a pytree that rode along.
    """

    value: Float[Array, ""]
    aux: PyTree | None
    trajectory: Trajectory


def shortest_member(
    lengths: Float[Array, "members"],
    beta: float | Float[Array, ""],
) -> Float[Array, ""]:
    """
    Smooth minimum of the member lengths.

    Parameters
    ----------
    lengths :
        Length of every member.
    beta :
        Sharpness. The value approaches the true shortest as it grows.

    Returns
    -------
    shortest :
        A length no greater than the shortest member.

    Notes
    -----
    The envelope of `ec3x.sizing` with its sign reversed, and taken in the
    logarithm of the length for the same reason: the sharpness is then
    dimensionless and comparable between structures of different size.

    It never overstates the true shortest, and falls below it by at most the
    member count raised to the reciprocal of the sharpness. Understating is the
    safe direction for a floor, since a constraint built on this one bites
    slightly early rather than slightly late.
    """
    return jnp.exp(-logsumexp(-beta * jnp.log(lengths)) / beta)


def penalized_mass(
    mass: Float[Array, ""],
    lengths: Float[Array, "members"],
    floor: float | Float[Array, ""],
    *,
    beta: float | Float[Array, ""],
    weight: float | Float[Array, ""],
) -> Float[Array, ""]:
    """
    A mass inflated wherever the shortest member falls below a floor.

    Parameters
    ----------
    mass :
        Total mass of the members.
    lengths :
        Length of every member.
    floor :
        Shortest member the design is allowed.
    beta :
        Sharpness of the smooth minimum.
    weight :
        Size of the inflation at a member of zero length.

    Returns
    -------
    penalized :
        The mass, multiplied by one plus the penalty.

    Notes
    -----
    **Why a floor is needed at all.** Nothing in a member check objects to a
    vanishing member, and two things reward one: its mass is the product of an
    area and a length, and its buckling length is its own length, so as it
    shortens it becomes both free and unbucklable. An unconstrained search
    therefore collapses members rather than improving the form, and the mass it
    reports is the collapse rather than a design.

    The penalty is multiplicative and reads a ratio, so it needs no mass scale
    and means the same thing on any structure. It is the square of the
    fractional violation, so it is zero and flat at the floor rather than
    kinked, and a search may approach from either side.

    A penalty rather than a constraint because the objective stays scalar, which
    keeps one reverse pass per gradient. Bounding the length exactly would need
    a constrained method and a Jacobian row per member.
    """
    violation = jnp.maximum(1.0 - shortest_member(lengths, beta) / floor, 0.0)

    return mass * (1.0 + weight * violation**2)


def annealing_schedule(
    start: float,
    stop: float,
    rounds: int,
) -> Float[Array, "rounds"]:
    """
    A geometric schedule of envelope sharpnesses.

    Parameters
    ----------
    start :
        Sharpness of the first round.
    stop :
        Sharpness of the last round.
    rounds :
        Number of rounds, the first and last included.

    Returns
    -------
    schedule :
        The sharpness of every round.

    Raises
    ------
    ValueError
        If either sharpness is not positive, or there is less than one round.

    Notes
    -----
    Geometric rather than linear because what the envelope gives away falls as
    the reciprocal of the sharpness, so equal ratios buy equal fractions of the
    remaining excess and a linear schedule would spend most of its rounds where
    there is nothing left to gain.
    """
    if start <= 0.0 or stop <= 0.0:
        raise ValueError(f"sharpness must be positive, got {start} and {stop}")
    if rounds < 1:
        raise ValueError(f"rounds must be at least one, got {rounds}")

    return jnp.geomspace(start, stop, rounds)


def value_and_gradient(
    objective: Callable[..., ObjectiveValue],
    *,
    has_aux: bool = False,
) -> ValueAndGradient:
    """
    The value and the gradient of an objective together, compiled once.

    Parameters
    ----------
    objective :
        The mass, as a function of the force densities and of anything else it
        is parameterized by. Differentiated in its first argument alone.
    has_aux :
        Whether the objective returns a value and a pytree rather than a value.
        The pytree is carried out untouched and never differentiated.

    Returns
    -------
    value_and_gradient :
        A function of the same arguments returning both, tracing on its first
        call and running the compiled program on every later one.

    Notes
    -----
    **An objective that returns what it computed costs nothing extra.** The
    design behind a mass is already in the program, so returning it alongside
    hands the caller the answer's shape, forces and sections for the price of
    the traffic — where recomputing it afterwards is a second compilation.

    **The compilation boundary, exposed so that a caller can decide when it is
    paid.** `descend` builds one of these if it is not handed one, and a caller
    timing a search wants it built and called once beforehand instead: compiling
    is a fixed cost that belongs to neither the objective nor the optimizer, and
    charging a descent for it measures the tracer.

    Reusing one across several searches is what makes the compilation a cost per
    objective rather than per search, and an objective that takes what varies as
    an argument rather than capturing it is what makes that reuse possible. An
    annealing schedule shares one program across all of its rounds this way: a
    traced sharpness parameterizes a single program instead of selecting between
    one program per round.

    **Every argument must be a JAX type**, an array or a pytree of them. What
    the objective computes with and no optimizer varies — the blocks, their
    assemblies, the load cases — belongs in its closure, where it is a constant
    rather than a traced leaf. Handing one of those in as an argument instead
    traces the index arrays a solver was compiled around, and the first place
    that surfaces is a concretization error inside the assembly.
    """
    return jax.jit(jax.value_and_grad(objective, has_aux=has_aux))


def minimize_bounded(
    objective: Callable[[Float[Array, "members"]], ObjectiveValue],
    q: Float[Array, "members"],
    *,
    bounds: tuple[float, float],
    iterations: int,
    sharpness: float | Float[Array, ""] = 0.0,
    has_aux: bool = False,
    gradient: ValueAndGradient | None = None,
) -> SearchResult:
    """
    Minimize a scalar objective in the force densities, under box bounds.

    Parameters
    ----------
    objective :
        The mass, as a function of the force densities alone.
    q :
        Force density of every member, to start from.
    bounds :
        Smallest and largest value any force density may take.
    iterations :
        Most iterations to spend.
    sharpness :
        Envelope sharpness to stamp on every iterate. Recorded and never used,
        this being generic over any scalar objective; zero says the caller had
        none to give.
    has_aux :
        Whether the objective returns a value and a pytree rather than a value.
        A handed-in gradient must have been built the same way.
    gradient :
        A compiled value and gradient of the objective, from
        `value_and_gradient`. If None, one is built here and this call pays for
        compiling it.

    Returns
    -------
    found :
        The answer, whatever the objective computed there, and the trajectory.

    Notes
    -----
    L-BFGS-B, driven from JAX's value and gradient together, so each iteration
    costs one forward pass and one reverse pass rather than two forward ones.

    **Compiled once and reused for every evaluation.** The objective is traced on
    the first evaluation and every later one runs the compiled program, which is
    where nearly all of the saving in a descent is. The objective must therefore
    be jittable, which for the pipeline means its analysis model is prepared by
    the caller rather than built inside.

    **A caller who is timing the search should compile before it starts**, by
    building `value_and_gradient` and calling it once, then passing it here. Left
    to itself this call traces inside its first evaluation, and the elapsed time
    of the search then includes a fixed cost that has nothing to do with either
    the objective or the optimizer.

    One compilation covers one objective, and a schedule of them covers all its
    rounds provided the sharpness reaches the objective as a traced argument
    rather than a captured constant. `optimize_annealed` hands one compiled
    function down for that reason.

    The value at an iterate the search never evaluated directly is recovered from
    this same compiled function and its gradient discarded, rather than by
    compiling the objective a second time on its own.

    **The bounds are what keep the force densities away from zero**, where the
    force density system is singular and a funicular shape stops existing. They
    are not a design constraint and should sit far from the answer; a bound that
    binds means the search has left the region where the model is meaningful.

    The objective recorded against an iterate is the one computed at it during
    the search, recovered from the values already evaluated rather than by
    calling again. An iterate the search never evaluated directly is recomputed.

    **The last row is the answer the search returned, not the last point it
    reported.** L-BFGS-B keeps the best iterate it found, and the two differ
    whenever it stops on its iteration limit part-way through a line search. It
    also steps once before honouring a limit of zero, so a trajectory that ends
    where it started is the only honest report of having gone nowhere.

    **Only the newest aux is held**, since the answer is not known until the
    search is over and a pytree per evaluation is a design per evaluation. The
    answer is usually the point evaluated last, and when it is not, one call
    recovers what it computed there — still cheaper than tracing the objective
    again, which is what recomputing the answer outside this function costs.
    """
    evaluated: dict[bytes, float] = {}
    newest: dict[bytes, PyTree] = {}
    visited = [np.asarray(q, dtype=np.float64)]

    if gradient is None:
        gradient = value_and_gradient(objective, has_aux=has_aux)

    def evaluate_objective(x: Float[np.ndarray, "members"]):
        computed, slope = gradient(jnp.asarray(x))
        value, aux = computed if has_aux else (computed, None)
        evaluated[x.tobytes()] = float(value)
        newest.clear()
        newest[x.tobytes()] = aux

        return float(value), np.asarray(slope, dtype=np.float64)

    def record_step(x: Float[np.ndarray, "members"]) -> None:
        visited.append(np.array(x, dtype=np.float64))

    if iterations > 0:
        result = minimize(
            evaluate_objective,
            np.asarray(q, dtype=np.float64),
            jac=True,
            method="L-BFGS-B",
            bounds=[bounds] * int(np.size(q)),
            callback=record_step,
            options={"maxiter": iterations},
        )

        if not np.array_equal(visited[-1], result.x):
            visited.append(np.asarray(result.x, dtype=np.float64))

    masses = []
    for step in visited:
        recorded = evaluated.get(step.tobytes())
        if recorded is None:
            recorded = evaluate_objective(step)[0]
        masses.append(recorded)

    answer = visited[-1]
    if answer.tobytes() not in newest:
        evaluate_objective(answer)

    trajectory = Trajectory(
        q=jnp.asarray(np.stack(visited)),
        mass=jnp.asarray(masses),
        beta=jnp.full(len(visited), sharpness),
    )

    return SearchResult(trajectory.mass[-1], newest[answer.tobytes()], trajectory)


def optimize_annealed(
    objective: Callable[
        [Float[Array, "members"], float | Float[Array, ""]], ObjectiveValue
    ],
    q: Float[Array, "members"],
    schedule: Sequence[float] | Float[Array, "rounds"],
    *,
    bounds: tuple[float, float],
    iterations: int = 50,
    has_aux: bool = False,
) -> SearchResult:
    """
    Minimize over an annealing schedule, each round warm-starting the next.

    Parameters
    ----------
    objective :
        The mass, as a function of the force densities and a sharpness.
    q :
        Force density of every member, to start from.
    schedule :
        Sharpness of every round, from `anneal`.
    bounds :
        Smallest and largest value any force density may take.
    iterations :
        Most iterations to spend in each round.
    has_aux :
        Whether the objective returns a value and a pytree rather than a value.

    Returns
    -------
    found :
        The last round's answer, what the objective computed there, and every
        iterate of every round with the sharpness it was taken under.

    Notes
    -----
    Warm-starting is the whole point of a schedule. A blunt envelope is cheap to
    descend and lands near the right design; a sharp one is what makes that
    design the smallest adequate one, and starting it from scratch would spend
    its iterations rediscovering what the blunt round already found.

    The design stays adequate throughout, since the envelope never understates
    what any load case demands. Annealing therefore approaches the answer from the
    safe side, and stopping early leaves a heavier structure rather than an
    inadequate one.

    **Every round shares one compiled objective, built here rather than inside
    the search.** The sharpness is an argument of that program rather than a
    constant captured in it, so a round is the same program as its neighbour at a
    different value and the compilation is paid once for the whole schedule.
    Building it at this level is what keeps that cost outside the descent it pays
    for, and visible to anything timing one.

    The schedule is converted to an array first, so that every round reaches the
    objective at the same dtype and a sequence and an array behave alike here.
    """
    iterates = []
    masses = []
    sharpnesses = []

    compiled = value_and_gradient(objective, has_aux=has_aux)
    schedule = jnp.asarray(schedule)

    for sharpness in schedule:

        def round_objective(x, sharpness=sharpness):
            return objective(x, sharpness)

        found = minimize_bounded(
            round_objective,
            q,
            bounds=bounds,
            iterations=iterations,
            sharpness=sharpness,
            has_aux=has_aux,
            gradient=lambda x, sharpness=sharpness: compiled(x, sharpness),
        )
        walked = found.trajectory
        q = walked.q[-1]

        iterates.append(walked.q)
        masses.append(walked.mass)
        sharpnesses.append(walked.beta)

    trajectory = Trajectory(
        q=jnp.concatenate(iterates),
        mass=jnp.concatenate(masses),
        beta=jnp.concatenate(sharpnesses),
    )

    return SearchResult(found.value, found.aux, trajectory)


def augmented_penalty(
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
    **The shift is what makes this an augmented Lagrangian rather than a
    penalty.** At a stationary point of the sum an active row sits at zero
    slack, so its shifted argument is the multiplier over the penalty and the
    term contributes exactly minus the multiplier times the row's gradient.
    First-order optimality of the original problem is then recovered at a
    finite penalty, where a plain penalty reaches it only in the limit and so
    always stops a little inside the feasible region.

    That matters wherever the answer is known to sit on the constraint
    surface. A fully-stressed design is such an answer: every governing member
    is at utilization one, and a method that stops short of the surface
    reports a heavier structure for a reason that has nothing to do with the
    standard.

    Aggregating the rows here, inside the traced program, is what leaves the
    whole constraint set costing one reverse pass. Handing the rows to a solver
    that builds its own penalty costs a Jacobian row per constraint instead,
    which for a frame checked member by member and case by case is the whole
    expense of a search.
    """
    shifted = jnp.minimum(slack - multipliers / penalty, 0.0)

    return 0.5 * penalty * jnp.sum(shifted**2)


def shifted_multipliers(
    multipliers: Float[np.ndarray, "constraints"],
    slack: Float[np.ndarray, "constraints"],
    penalty: float,
    ceiling: float,
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
    ceiling :
        Largest value any multiplier may take.

    Returns
    -------
    shifted :
        The estimate the next round is solved at.

    Notes
    -----
    A row that is satisfied with room to spare has its multiplier driven to
    zero, and a row that is violated has it raised in proportion to the
    violation. The floor at zero is the sign condition an inequality's
    multiplier must satisfy, and is not a safeguard.

    **The ceiling is a safeguard.** One badly conditioned round can hand back
    an estimate orders of magnitude past anything the answer will support, and
    an unbounded estimate turns the next subproblem into a barrier whose
    minimum sits nowhere near the constraint surface. Capping costs nothing at
    a well-behaved point, where no multiplier approaches it.
    """
    raised = multipliers - penalty * slack

    return np.clip(raised, 0.0, ceiling)


class AugmentedBudget(NamedTuple):
    """
    What an augmented Lagrangian descent is allowed to spend, and when it stops.

    Attributes
    ----------
    rounds :
        Most multiplier updates to spend.
    iterations :
        Most inner iterations in each opening round.
    settled :
        Most inner iterations in every round after the opening ones.
    opening :
        How many rounds count as opening ones.
    penalty :
        Penalty parameter of the first round.
    growth :
        What the penalty is multiplied by when a round fails to earn its share
        of the violation it inherited.
    ceiling :
        Largest penalty the loop may reach, and the largest multiplier with it.
    tolerance :
        Violation at or under which the rows count as satisfied.
    quiet :
        Relative movement of the mass, between consecutive rounds, that the
        loop treats as no movement.

    Notes
    -----
    **A small opening penalty is the setting that decides the answer, not the
    speed.** It leaves the mass in charge of the first rounds, so the search
    crosses the infeasible region rather than skirting it, and comes back to
    the constraint surface somewhere a method confined to feasible points
    cannot reach. Raising it to where feasibility leads from the start turns
    the same machinery into an expensive way of reproducing a worse answer.

    The inner budget falls after the opening rounds because by then the mass is
    nearly decided and the remaining rounds are buying feasibility, which is a
    much shorter walk than the one that found the basin.
    """

    rounds: int
    iterations: int
    settled: int
    opening: int
    penalty: float
    growth: float
    ceiling: float
    tolerance: float
    quiet: float


class AugmentedAnswer(NamedTuple):
    """
    What an augmented Lagrangian descent arrived at, and the road there.

    Attributes
    ----------
    variables :
        The variable vector the loop stopped on.
    masses :
        Objective at the end of every round, the starting value first.
    violations :
        Worst violation over the rows at the end of every round.
    evaluations :
        Objective evaluations spent over every round.
    converged :
        Whether the loop stopped because the rows were satisfied and the mass
        had stopped moving, rather than on its round budget.

    Notes
    -----
    The two columns are read together or not at all. A mass falling while the
    violation is still large is not an improvement — it is the search spending
    the infeasible region — and only the last row, where the violation is under
    the tolerance, is a design.
    """

    variables: Float[np.ndarray, "variables"]
    masses: Float[np.ndarray, "rounds"]
    violations: Float[np.ndarray, "rounds"]
    evaluations: int
    converged: bool


# How much worse than the last point that evaluated a point outside the model's
# domain is reported as, so that no line search prefers one.
RECOIL_GROWTH = 1e3

# A round need only be solved as accurately as the violation it inherited, and
# never more accurately than this.
INNER_SHARE = 0.1
INNER_FLOOR = 1e-10

# A round has earned its keep if it took this share off the worst violation.
EARNED_SHARE = 0.25


def strayed_point(
    x: Float[np.ndarray, "variables"],
    anchor: Float[np.ndarray, "variables"],
    held: float,
) -> tuple[float, Float[np.ndarray, "variables"]]:
    """
    A value and a gradient that walk a line search back into the model's domain.

    Parameters
    ----------
    x :
        The trial point that could not be evaluated.
    anchor :
        The last point that could be, which the walk heads back towards.
    held :
        Objective at that anchor.

    Returns
    -------
    strayed :
        The value to report, and the gradient to report with it.

    Notes
    -----
    A geometry whose frame cannot be factorized, or whose form-finding system
    has gone indefinite, has no objective at all — the truthful reading is not
    a number but a refusal. Reporting a refusal to a line search stops the
    search, so instead the point is charged the value of a distant quadratic
    centred on the anchor: strictly worse than the anchor by construction, and
    with a gradient whose descent direction points home. The search then steps
    back inside on its own, and no accepted iterate ever sits out here.

    **The quadratic is not a barrier and does not shape the answer.** It is
    only ever evaluated at points the search goes on to reject, so it enters no
    accepted step and no curvature estimate built from accepted steps.
    """
    strayed = np.asarray(x, dtype=np.float64) - anchor
    scale = max(abs(held), 1.0)
    value = RECOIL_GROWTH * scale + 0.5 * float(strayed @ strayed)

    return value, strayed


class ConstrainedMaps(NamedTuple):
    """
    The three compiled programs a constrained descent calls.

    Attributes
    ----------
    augmented :
        Value and gradient of the augmented objective, in the variables, taking
        the multipliers, the penalty and the objective's reference value as
        arguments beside them.
    weigh :
        Value and gradient of the objective alone, read for the reference the
        augmented objective is scaled by and for the mass a round reports.
    slack :
        How far above zero every inequality row sits.

    Notes
    -----
    **The multipliers, the penalty and the reference are arguments of the
    augmented program rather than constants captured in it.** A round is then
    the same program as its neighbour at different values, so one compilation
    covers the whole outer loop; capturing them instead retraces a program of
    the size of the constraint set once per round.
    """

    augmented: object
    weigh: object
    slack: object


def worst_violation(
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


def descend_augmented(
    maps: ConstrainedMaps,
    start: Float[np.ndarray, "variables"],
    boxes: list[tuple[float | None, float | None]],
    budget: AugmentedBudget,
) -> AugmentedAnswer:
    """
    Minimize under inequality rows by an augmented Lagrangian, in box bounds.

    Parameters
    ----------
    maps :
        The search's compiled programs.
    start :
        The variable vector to leave from, which must be inside the model's
        domain.
    boxes :
        One bound pair per variable, held natively by the inner solver rather
        than penalized.
    budget :
        Rounds, inner iterations, the penalty schedule, and what counts as
        satisfied and as no longer moving.

    Returns
    -------
    answer :
        The variables, the mass and violation of every round, and how it ended.

    Raises
    ------
    ValueError
        If a budget is not usable, or if the objective at the start is not a
        positive finite number.

    Notes
    -----
    **One reverse pass per gradient, whatever the constraint set.** The rows are
    aggregated inside the traced program by `augmented_penalty`, so the whole
    set costs one adjoint rather than a Jacobian row apiece. On a frame checked
    member by member and case by case that is the difference between a search
    that fits in a working session and one that does not, and it is what lets
    the same objective cross a remote boundary as a single cotangent.

    **The answer lands on the constraint surface.** The shift in the penalty
    recovers first-order optimality at a finite penalty, so a row that governs
    ends at zero slack rather than a little inside it. A short run of a
    constrained method from here confirms it and costs a handful of iterations,
    which is also the cheapest available certificate that the landing is a
    stationary point and not somewhere the outer loop ran out of rounds.

    **The opening rounds are solved to precision and the later ones are not.**
    A round is only ever trying to remove the violation it inherited, so asking
    it for more accuracy than that buys nothing; the opening rounds are the
    exception because they are choosing the basin rather than repairing the
    rows, and there the inner iteration budget is what stops them.

    A trial point can leave the model's domain entirely — a geometry whose
    frame will not factorize, or whose form-finding system has gone indefinite.
    Such a point is charged a distant quadratic centred on the last point that
    evaluated, which no line search prefers and whose descent direction points
    back inside. A non-finite value or gradient is treated identically, since a
    solver that reports a NaN rather than raising would otherwise poison every
    curvature estimate after it.

    **`RuntimeError` is caught alongside the value errors, and has to be.** A
    solver whose failure is detected inside a compiled program reports it
    through a host callback, and the exception that surfaces from one is a
    runtime error rather than anything about a value. Catching only the value
    errors leaves the commonest way for a frame to fail unhandled, which is not
    a crash but a finite and meaningless number carried forward.

    The loop is deterministic. Nothing here is sampled, so a landing is a
    measurement and two runs of one budget agree bit for bit.
    """
    if budget.rounds < 1 or budget.iterations < 1 or budget.settled < 1:
        raise ValueError(f"rounds and iterations must be positive, got {budget}")
    if budget.penalty <= 0.0 or budget.ceiling < budget.penalty:
        raise ValueError(
            f"the penalty must be positive and under its ceiling, {budget}"
        )
    if budget.growth <= 1.0:
        raise ValueError(f"the penalty must grow, got {budget.growth}")

    x = np.asarray(start, dtype=np.float64)
    reference = abs(float(maps.weigh(jnp.asarray(x))[0]))
    if not np.isfinite(reference) or reference == 0.0:
        raise ValueError(f"the objective at the start is not usable: {reference}")

    scale = jnp.asarray(reference)
    violation, rows = worst_violation(maps.slack, x)
    multipliers = np.zeros(rows.size)
    penalty = float(budget.penalty)

    resting = jnp.zeros(rows.size)
    opened = maps.augmented(jnp.asarray(x), resting, scale, scale)[0]
    anchor = x.copy()
    held = float(opened)

    def evaluate_augmented(z, carried, charged):
        nonlocal anchor, held
        try:
            value, slope = maps.augmented(jnp.asarray(z), carried, charged, scale)
            value = float(value)
            slope = np.asarray(slope, dtype=np.float64)
        except (ValueError, FloatingPointError, RuntimeError):
            return strayed_point(z, anchor, held)
        if not np.isfinite(value) or not np.all(np.isfinite(slope)):
            return strayed_point(z, anchor, held)
        anchor = np.asarray(z, dtype=np.float64).copy()
        held = value

        return value, slope

    masses = [reference]
    violations = [violation]
    spent = 0
    converged = False
    inherited = violation

    for round_index in range(budget.rounds):
        carried = jnp.asarray(multipliers)
        charged = jnp.asarray(penalty)
        if round_index < budget.opening:
            inner = budget.iterations
            precision = INNER_FLOOR
        else:
            inner = budget.settled
            precision = max(INNER_FLOOR, INNER_SHARE * inherited)

        def round_objective(z, carried=carried, charged=charged):
            return evaluate_augmented(z, carried, charged)

        found = minimize(
            round_objective,
            x,
            jac=True,
            method="L-BFGS-B",
            bounds=boxes,
            options={
                "maxiter": inner,
                "maxfun": 3 * inner,
                "ftol": 0.0,
                "gtol": precision,
            },
        )
        x = np.asarray(found.x, dtype=np.float64)
        spent += int(found.nfev)

        violation, rows = worst_violation(maps.slack, x)
        mass = abs(float(maps.weigh(jnp.asarray(x))[0]))
        moved = abs(mass - masses[-1]) / max(mass, reference)
        masses.append(mass)
        violations.append(violation)

        multipliers = shifted_multipliers(multipliers, rows, penalty, budget.ceiling)
        if violation > EARNED_SHARE * inherited:
            penalty = min(penalty * budget.growth, budget.ceiling)
        inherited = violation

        if violation <= budget.tolerance and moved <= budget.quiet:
            converged = True
            break

    return AugmentedAnswer(
        x, np.asarray(masses), np.asarray(violations), spent, converged
    )
