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
smallest adequate one from above instead of chattering between cases. Nothing
else about the problem is relaxed: the check is exact at every iterate, and the
utilization the final design reports is the standard's own.

The gradient comes from the pipeline and costs one reverse pass whatever the
number of force densities, which is what makes a per-member design variable
affordable. The optimizer itself is L-BFGS-B, because the bounds are boxes and
the objective is smooth between the branches of the standard.
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
from scipy.optimize import minimize


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


def shortest(
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
    The envelope of `normax.ec3.sizing` with its sign reversed, and taken in the
    logarithm of the length for the same reason: the sharpness is then
    dimensionless and comparable between structures of different size.

    It never overstates the true shortest, and falls below it by at most the
    member count raised to the reciprocal of the sharpness. Understating is the
    safe direction for a floor, since a constraint built on this one bites
    slightly early rather than slightly late.
    """
    return jnp.exp(-logsumexp(-beta * jnp.log(lengths)) / beta)


def penalized(
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
    violation = jnp.maximum(1.0 - shortest(lengths, beta) / floor, 0.0)

    return mass * (1.0 + weight * violation**2)


def anneal(
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


def descend(
    objective: Callable[[Float[Array, "members"]], Float[Array, ""]],
    q: Float[Array, "members"],
    *,
    bounds: tuple[float, float],
    iterations: int,
    sharpness: float | Float[Array, ""] = 0.0,
) -> Trajectory:
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

    Returns
    -------
    trajectory :
        Every iterate, its objective and the starting point.

    Notes
    -----
    L-BFGS-B, driven from JAX's value and gradient together, so each iteration
    costs one forward pass and one reverse pass rather than two forward ones.

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
    """
    evaluated: dict[bytes, float] = {}
    visited = [np.asarray(q, dtype=np.float64)]

    gradient = jax.value_and_grad(objective)

    def evaluate(x: Float[np.ndarray, "members"]):
        value, slope = gradient(jnp.asarray(x))
        evaluated[x.tobytes()] = float(value)

        return float(value), np.asarray(slope, dtype=np.float64)

    def record(x: Float[np.ndarray, "members"]) -> None:
        visited.append(np.array(x, dtype=np.float64))

    if iterations < 1:
        return Trajectory(
            q=jnp.asarray(np.stack(visited)),
            mass=jnp.asarray([float(objective(jnp.asarray(q)))]),
            beta=jnp.full(1, sharpness),
        )

    result = minimize(
        evaluate,
        np.asarray(q, dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        bounds=[bounds] * int(np.size(q)),
        callback=record,
        options={"maxiter": iterations},
    )

    if not np.array_equal(visited[-1], result.x):
        visited.append(np.asarray(result.x, dtype=np.float64))

    masses = [
        evaluated.get(step.tobytes()) or float(objective(jnp.asarray(step)))
        for step in visited
    ]

    return Trajectory(
        q=jnp.asarray(np.stack(visited)),
        mass=jnp.asarray(masses),
        beta=jnp.full(len(visited), sharpness),
    )


def optimize(
    objective: Callable[
        [Float[Array, "members"], float | Float[Array, ""]], Float[Array, ""]
    ],
    q: Float[Array, "members"],
    schedule: Sequence[float] | Float[Array, "rounds"],
    *,
    bounds: tuple[float, float],
    iterations: int = 50,
) -> Trajectory:
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

    Returns
    -------
    trajectory :
        Every iterate of every round, and the sharpness it was taken under.

    Notes
    -----
    Warm-starting is the whole point of a schedule. A blunt envelope is cheap to
    descend and lands near the right design; a sharp one is what makes that
    design the smallest adequate one, and starting it from scratch would spend
    its iterations rediscovering what the blunt round already found.

    The design stays adequate throughout, since the envelope never understates
    what any case demands. Annealing therefore approaches the answer from the
    safe side, and stopping early leaves a heavier structure rather than an
    inadequate one.
    """
    iterates = []
    masses = []
    sharpnesses = []

    for sharpness in schedule:
        walked = descend(
            lambda x, sharpness=sharpness: objective(x, sharpness),
            q,
            bounds=bounds,
            iterations=iterations,
            sharpness=sharpness,
        )
        q = walked.q[-1]

        iterates.append(walked.q)
        masses.append(walked.mass)
        sharpnesses.append(walked.beta)

    return Trajectory(
        q=jnp.concatenate(iterates),
        mass=jnp.concatenate(masses),
        beta=jnp.concatenate(sharpnesses),
    )
