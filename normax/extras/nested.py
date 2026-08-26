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
The nested route: descend the force densities, size the members inside.

The optimizer never sees a diameter. Every iterate is analyzed at frozen seed
diameters and sized by the fully-stressed map, the load cases are reconciled
by a smooth envelope whose sharpness anneals upward, a vanishing member is
priced by a penalized length floor, and the seed is refreshed between rounds
until the frame is analyzed at its own sections. The augmented Lagrangian in
`normax.design` replaced all of it; this is kept as an add-on.
"""

from collections.abc import Callable
from collections.abc import Sequence
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int
from jaxtyping import PyTree
from scipy.optimize import minimize

from normax.design import Design
from normax.design import DesignParameters
from normax.design import StructuralDesignPipeline
from normax.loads import LoadCases
from normax.loads import count_load_cases
from normax.sections import MemberSections
from normax.sizing import MemberSizes

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
    One row per iteration, not per evaluation, and the last row is the answer.
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
        Objective at the answer.
    aux :
        Whatever the objective returned alongside its value there, or None.
    trajectory :
        Where the optimizer went, in the order it went there.

    Notes
    -----
    The aux is the design behind the answer, carried out so it is never
    recomputed — recomputing would trace the pipeline a second time.
    """

    value: Float[Array, ""]
    aux: PyTree | None
    trajectory: Trajectory


def size_design(
    pipeline: StructuralDesignPipeline,
    params: DesignParameters,
    loads: LoadCases,
) -> Design:
    """
    Form-find once, analyze at the given diameters, and size for every case.

    Parameters
    ----------
    pipeline :
        The three blocks.
    params :
        The form finder's coordinates, and the seed diameters the frame is
        analyzed at.
    loads :
        The case the shape answers to, and the cases it is sized for.

    Returns
    -------
    design :
        The shape, the forces, and the section every load case demands of
        every member on its own.

    Notes
    -----
    The sizing map rather than the check: the diameters set the stiffness and
    the sections come back from the standard, one per load case, so the
    coupling between the two is staggered and closed by `settle_diameters`.
    """
    shape = pipeline.formfinder(params.coordinates, loads.formfinding)
    forces = pipeline.analyzer(shape.xyz, params.diameters, loads.analysis)
    sizes = pipeline.sizer(forces, shape.lengths)

    return Design(shape, forces, sizes)


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
    Taken in the logarithm, so the sharpness is dimensionless. It understates
    the shortest by at most the member count to the reciprocal sharpness,
    which is the safe direction for a floor.
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
    Multiplicative and squared in the fractional violation: it needs no mass
    scale, and it is zero and flat at the floor so a search may approach from
    either side. A penalty keeps the objective scalar and the gradient one
    reverse pass.
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
    Geometric because what the envelope gives away falls as the reciprocal of
    the sharpness, so equal ratios buy equal fractions of the excess.
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

    Returns
    -------
    value_and_gradient :
        A function of the same arguments returning both.

    Notes
    -----
    Built and called once before a timed search, so the compilation is not
    charged to the descent. Every argument must be a JAX type; what no
    optimizer varies belongs in the objective's closure.
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
        Envelope sharpness to stamp on every iterate. Recorded, never used;
        zero says the caller had none to give.
    has_aux :
        Whether the objective returns a value and a pytree rather than a value.
    gradient :
        A compiled value and gradient of the objective, or None to build one
        here and pay for compiling it.

    Returns
    -------
    found :
        The answer, whatever the objective computed there, and the trajectory.

    Notes
    -----
    L-BFGS-B on a compiled value and gradient. The last row of the trajectory
    is the answer the solver returned, which differs from the last point it
    reported whenever it stopped inside a line search; a limit of zero
    iterations returns the start. Only the newest aux is held, and one extra
    evaluation recovers the answer's when it was not the point evaluated last.
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
        Sharpness of every round, from `annealing_schedule`.
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
    The sharpness reaches one compiled program as a traced argument, so the
    schedule is compiled once. The schedule is made an array first so a
    sequence of floats is not one program per round.
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


def governing_load_case(
    diameters: Float[Array, "load_cases members"],
) -> Int[Array, "members"]:
    """
    Which load case decided each member's size.

    Parameters
    ----------
    diameters :
        Outer diameter every load case demands of every member on its own.

    Returns
    -------
    governing_load_case :
        Index of the load case working each member hardest.

    Notes
    -----
    An `argmax`, exact because capacity is strictly increasing in the
    diameter. Non-differentiable.
    """
    return jnp.argmax(diameters, axis=0)


def diameter_envelope(
    diameters: Float[Array, "load_cases members"],
    beta: float | Float[Array, ""],
) -> Float[Array, "members"]:
    """
    Smooth envelope of a member's size over several load cases.

    Parameters
    ----------
    diameters :
        Diameter required by each load case, one row per case.
    beta :
        Sharpness. The envelope approaches the true largest as it grows.

    Returns
    -------
    diameter :
        Diameter covering every load case.

    Notes
    -----
    Taken in the logarithm, so the sharpness is dimensionless. It never
    understates the largest and exceeds it by at most the logarithm of the
    case count over the sharpness, so annealing approaches from the safe side.
    """
    logarithms = jnp.log(diameters)
    smoothed = logsumexp(beta * logarithms, axis=0) / beta

    return jnp.exp(smoothed)


def design_envelope(
    design: Design,
    sharpness: float | Float[Array, ""] | None = None,
) -> Design:
    """
    Reconcile the load cases into one section per member.

    Parameters
    ----------
    design :
        A design whose sections carry a load case axis.
    sharpness :
        Sharpness of the envelope, or None for the true largest.

    Returns
    -------
    design :
        The same design with one section per member and the axis collapsed.

    Notes
    -----
    The envelope is scale-equivariant, so a thickness that is a fixed fraction
    of the diameter keeps its ratio. The utilization passes through untouched;
    it still describes the per-case sections. A single load case is returned
    as it stands.
    """
    demanded = design.sizes.sections
    cases = count_load_cases(demanded.diameter)

    def cover_cases(
        field: Float[Array, "load_cases members"],
    ) -> Float[Array, "members"]:
        if cases == 1:
            return field[0]
        if sharpness is None:
            return jnp.max(field, axis=0)

        return diameter_envelope(field, sharpness)

    diameters = cover_cases(demanded.diameter)
    thicknesses = cover_cases(demanded.thickness)
    covering = MemberSections(diameters, thicknesses, demanded.material)
    sizes = MemberSizes(covering, design.sizes.utilization)

    return Design(design.shape, design.forces, sizes)


def settle_diameters(
    objective: Callable[[DesignParameters], ObjectiveValue],
    params: DesignParameters,
    *,
    settling_passes: int = 400,
    settling_tolerance: float = 1e-6,
) -> Float[Array, "members"]:
    """
    The diameters an analysis at these coordinates asks of itself.

    Parameters
    ----------
    objective :
        The mass of a set of design parameters, returning the enveloped
        design it weighed alongside.
    params :
        Coordinates to hold, and the diameters to start the analysis at.
    settling_passes :
        Most analyses to spend before the coupling is called stalled.
    settling_tolerance :
        Largest fractional movement in any diameter that counts as settled.

    Returns
    -------
    settled :
        Diameters the analysis and the check agree on.

    Raises
    ------
    ValueError
        If the diameters are still moving when the passes run out.

    Notes
    -----
    Sizing is a contraction in the diameters, so forward passes reach its
    fixed point without a gradient. The start is restated at its own dtype so
    a weakly typed seed does not compile the pipeline twice.
    """
    weighed = eqx.filter_jit(objective)
    assumed = jnp.asarray(params.diameters, dtype=params.diameters.dtype)
    moved = float("inf")

    for _ in range(settling_passes):
        _, design = weighed(DesignParameters(params.coordinates, assumed))
        demanded = design.sizes.sections.diameter
        moved = float(jnp.max(jnp.abs(demanded / assumed - 1.0)))
        assumed = demanded

        if moved < settling_tolerance:
            return demanded

    raise ValueError(
        f"diameters still moving by {moved:.3e} after {settling_passes} "
        f"passes at fixed force densities, above {settling_tolerance:.3e}"
    )


def optimize_staggered(
    objective: Callable[[DesignParameters], ObjectiveValue],
    params: DesignParameters,
    *,
    bounds: tuple[float, float],
    iterations: int = 50,
    rounds: int = 12,
    settling_passes: int = 400,
    settling_tolerance: float = 1e-6,
) -> SearchResult:
    """
    Minimize in the coordinates, refreshing the analysis diameters per round.

    Parameters
    ----------
    objective :
        The mass of a set of design parameters, returning the enveloped
        design it weighed alongside.
    params :
        Coordinates to start from, and the diameters the first round is
        analyzed with.
    bounds :
        Smallest and largest value any coordinate may take.
    iterations :
        Most iterations to spend in each round.
    rounds :
        Most descents to spend before the coupling is called stalled.
    settling_passes :
        Most analyses one round may spend closing the coupling.
    settling_tolerance :
        Largest fractional movement in any diameter that counts as settled.

    Returns
    -------
    found :
        The last round's answer, the design behind it, and every iterate of
        every round.

    Raises
    ------
    ValueError
        If the coupling has not closed within a round's passes or within the
        round cap.

    Notes
    -----
    Each round is one descent at frozen diameters, so the quasi-Newton model
    sees one fixed function; the seed is refreshed between rounds by
    `settle_diameters`. Both compiled programs take the diameters as an
    argument, so no round retraces the blocks.
    """
    iterates = []
    masses = []
    sharpnesses = []
    residual = float("inf")

    seed_diameters = jnp.asarray(params.diameters, dtype=params.diameters.dtype)
    current = DesignParameters(params.coordinates, seed_diameters)

    def seeded_objective(
        coordinates: Float[Array, "coordinates"],
        diameters: Float[Array, "members"],
    ) -> ObjectiveValue:
        seeded = DesignParameters(coordinates, diameters)

        return objective(seeded)

    compiled = value_and_gradient(seeded_objective, has_aux=True)

    for _ in range(rounds):
        held = current.diameters
        found = minimize_bounded(
            lambda x, seed=held: seeded_objective(x, seed),
            current.coordinates,
            bounds=bounds,
            iterations=iterations,
            has_aux=True,
            gradient=lambda x, seed=held: compiled(x, seed),
        )
        walked = found.trajectory
        iterates.append(walked.q)
        masses.append(walked.mass)
        sharpnesses.append(walked.beta)

        answer = walked.q[-1]
        settled = settle_diameters(
            objective,
            DesignParameters(answer, held),
            settling_passes=settling_passes,
            settling_tolerance=settling_tolerance,
        )
        current = DesignParameters(answer, settled)
        residual = float(jnp.max(jnp.abs(settled / held - 1.0)))

        if residual < settling_tolerance:
            break
    else:
        raise ValueError(
            f"diameters still moving by {residual:.3e} after "
            f"{rounds} rounds, above {settling_tolerance:.3e}"
        )

    trajectory = Trajectory(
        q=jnp.concatenate(iterates),
        mass=jnp.concatenate(masses),
        beta=jnp.concatenate(sharpnesses),
    )

    return SearchResult(found.value, found.aux, trajectory)
