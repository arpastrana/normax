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
Blueprints' EN 1993-1-1 cross-section check, as a differentiable map.

**No JAX anywhere in this module — that is the headline.** The check is
Blueprints' scalar formula objects, the solve is a plain Python bisection, and
the derivative endpoints are literal NumPy arithmetic: the implicit function
theorem at the root, its partials hand-derived. Where the sibling `ec3_check`
delegates its endpoints to `jax.vjp` over a traced map, this Tesseract shows
the other extreme — a library with no derivatives of any kind carrying an
exact adjoint across the boundary, because the boundary asks only for the
numbers, never for the machinery.

The check is cross-section resistance alone: EN 1993-1-1 eq. (6.2) with the
(6.10) and (6.14) resistances. Blueprints implements no §6.3.1 flexural
buckling and no classification, so no buckling length crosses this schema — a
schema field the check would ignore invites the belief that it participates.
The §6.3.2.1 lateral-torsional check it does carry is the one a doubly
symmetric tube never needs.

The schema carries both questions the pipeline asks of a check: what size
these actions demand (`diameter`, `utilization`), and how hard they work the
sizes the caller already owns (`diameter_held` in, `utilization_held` out).
The second is explicit arithmetic with no root find, so its adjoint is the
check's bare partials — the constraint field a simultaneous optimization
holds at or under one, crossing the boundary on every evaluation.

The host arithmetic restates `normax/sizing/blueprint.py`, which is this
module's drift alarm: a value-parity test pins the two at bit-identical.

Blueprints is LGPL-2.1, experiment-only, waived 2026-08-15: never on the
Apache-2.0 submission path.
"""

import hashlib
import math
from typing import Any
from typing import NamedTuple

import numpy as np
from blueprints.codes.eurocode.en_1993_1_1_2005.chapter_6_ultimate_limit_state.formula_6_10 import (  # noqa: E501
    Form6Dot10NcRdClass1And2And3,
)
from blueprints.codes.eurocode.en_1993_1_1_2005.chapter_6_ultimate_limit_state.formula_6_14 import (  # noqa: E501
    Form6Dot14MCRdClass3,
)
from jaxtyping import Float
from jaxtyping import Int
from pydantic import BaseModel
from tesseract_core.runtime import Array
from tesseract_core.runtime import Differentiable
from tesseract_core.runtime import Float64

# Fifty halvings of a bracket at most sqrt(2) + cbrt(2) wide. Measured over
# four thousand random members: the diameter lands within one ulp of what a
# larger budget reaches, and the utilization read back at it is one to 2.8e-15,
# against the 1e-9 the pipeline's invariant asks for.
BISECTION_HALVINGS = 50


class InputSchema(BaseModel):
    """
    Member actions and the family, one load case per call.
    """

    axial_force: Differentiable[Array[(None,), Float64]]
    """Design axial force of every member, in newtons. Tension positive."""

    end_moments_major: Differentiable[Array[(None, 2), Float64]]
    """Major-axis moment at each end of every member, in newton-millimeters."""

    end_moments_minor: Differentiable[Array[(None, 2), Float64]]
    """Minor-axis moment at each end of every member, in newton-millimeters."""

    diameter_held: Differentiable[Array[(None,), Float64]]
    """Outer diameter every member is checked at, in millimeters.

    Read only by the held check: the solve never consults it, so its pull
    through `diameter` and `utilization` is exactly zero.
    """

    f_y: Float64
    """Yield strength, in newtons per square millimeter.

    Not differentiable: the hand-written adjoint covers the actions alone,
    and only what it covers is marked. Stricter than the sibling check, where
    tracing covers every input incidentally.
    """

    gamma_m0: Float64
    """Partial factor for cross-section resistance."""

    ratio: Float64
    """Diameter-to-thickness ratio, fixing the wall."""

    diameter_min: Float64
    """Smallest diameter the section family offers, in millimeters."""


class OutputSchema(BaseModel):
    """
    The sizes the check requires, and how hard they work.
    """

    diameter: Differentiable[Array[(None,), Float64]]
    """Outer diameter of every member, in millimeters, floored at the minimum."""

    utilization: Differentiable[Array[(None,), Float64]]
    """Demand over resistance of every member, at the size just chosen.

    One to machine precision wherever the check decided the size, and below
    one wherever the catalogue minimum did.
    """

    utilization_held: Differentiable[Array[(None,), Float64]]
    """Demand over resistance of every member, at the held size that crossed.

    The constraint a simultaneous optimization holds at or under one.
    """

    clamped: Array[(None,), Float64]
    """Whether the catalogue minimum decided each member's size, as zero or one.

    **Non-differentiable.** A concrete cotangent on this raises `ValueError`,
    so drop it before differentiating; only a symbolic zero is accepted.
    """


class HostFamily(NamedTuple):
    """
    A section family as the host solver needs it: four concrete numbers.

    Attributes
    ----------
    area_coefficient :
        Area per squared diameter of the family's tubes.
    modulus_coefficient :
        Elastic modulus per cubed diameter of the family's tubes.
    f_y :
        Yield strength of the family's grade.
    gamma_m0 :
        Partial factor for cross-section resistance.
    """

    area_coefficient: float
    modulus_coefficient: float
    f_y: float
    gamma_m0: float


class HandPartials(NamedTuple):
    """
    Every closed-form partial the adjoint reads, evaluated at the solved state.

    Attributes
    ----------
    size_axial :
        Derivative of the clamped diameter in the axial force, zero at the clamp.
    size_moment :
        Derivative of the clamped diameter in the demand moment, zero at the clamp.
    check_axial :
        Derivative of the reported utilization in the axial force — exactly zero
        where the check decided the size, since the utilization sits pinned at one.
    check_moment :
        Derivative of the reported utilization in the demand moment.
    held_slope :
        Derivative of the held check in the held diameter, strictly negative
        wherever that size is positive.
    held_axial :
        Derivative of the held check in the axial force.
    held_moment :
        Derivative of the held check in the demand moment.
    winner_major :
        Index of the larger major-axis end moment of every member.
    winner_minor :
        Index of the larger minor-axis end moment of every member.
    sign_major :
        Sign of that winning major-axis end moment.
    sign_minor :
        Sign of that winning minor-axis end moment.
    """

    size_axial: Float[np.ndarray, "members"]
    size_moment: Float[np.ndarray, "members"]
    check_axial: Float[np.ndarray, "members"]
    check_moment: Float[np.ndarray, "members"]
    held_slope: Float[np.ndarray, "members"]
    held_axial: Float[np.ndarray, "members"]
    held_moment: Float[np.ndarray, "members"]
    winner_major: Int[np.ndarray, "members"]
    winner_minor: Int[np.ndarray, "members"]
    sign_major: Float[np.ndarray, "members"]
    sign_minor: Float[np.ndarray, "members"]


def _host_family(inputs: dict[str, Any]) -> HostFamily:
    """
    Reduce the flat schema fields to the coefficients the scalar check reads.

    Parameters
    ----------
    inputs :
        The validated input fields.

    Returns
    -------
    family :
        The family's geometry collapsed to two proportionality constants.

    Raises
    ------
    ValueError
        If the ratio leaves no wall at all — the in-process twin's refusal,
        enforced here too so the boundary never answers for an inverted bore.
    """
    ratio = float(inputs["ratio"])
    if ratio <= 2.0:
        raise ValueError(f"a ratio of {ratio} leaves no wall: need d/t > 2")

    wall = 1.0 / ratio
    bore = 1.0 - 2.0 * wall
    area_coefficient = math.pi * wall * (1.0 - wall)
    second_moment = (math.pi / 64.0) * (1.0 - bore**4)
    modulus_coefficient = 2.0 * second_moment

    return HostFamily(
        area_coefficient,
        modulus_coefficient,
        float(inputs["f_y"]),
        float(inputs["gamma_m0"]),
    )


def _check_scalar(
    diameter: float,
    axial: float,
    moment: float,
    family: HostFamily,
) -> float:
    """
    One member's utilization at one diameter, through Blueprints.

    Parameters
    ----------
    diameter :
        Outer diameter of the trial tube.
    axial :
        Axial force the member carries, negative in compression.
    moment :
        Demand moment the member carries, non-negative.
    family :
        The section family reduced to its host coefficients.

    Returns
    -------
    utilization :
        Demand over resistance, the linear sum of EN 1993-1-1 eq. (6.2).

    Raises
    ------
    ValueError
        If the moment arrives signed rather than reduced.
    """
    if moment < 0.0:
        raise ValueError(f"a moment of {moment} is signed: reduce before checking")

    area = family.area_coefficient * diameter**2
    modulus = family.modulus_coefficient * diameter**3
    squashing = Form6Dot10NcRdClass1And2And3(
        a=area, f_y=family.f_y, gamma_m0=family.gamma_m0
    )
    bending = Form6Dot14MCRdClass3(
        w_el_min=modulus, f_y=family.f_y, gamma_m0=family.gamma_m0
    )

    return abs(axial) / float(squashing) + moment / float(bending)


# Whether the clause's own evaluator can be reached without building the clause.
#
# **A private method of a third-party library is a fast path, never a
# correctness requirement.** It is checked once, here, for existing and for
# agreeing with the public class it belongs to; if a release ever moves it or
# changes it, this backend keeps answering by constructing the clause, slower
# and identical. What must not happen is that a library upgrade quietly changes
# what every size in this repository is solved against.
#
# The clause is a float subclass whose constructor calls this same evaluator and
# wraps the result, so the fast path is the library's own arithmetic without the
# object built around it — and no release can leave the evaluator unusable while
# the class still works.
PROBE_AREA = 1.2e3
PROBE_MODULUS = 5.0e4
PROBE_YIELD = 355.0
PROBE_FACTOR = 1.0


def _evaluator_agrees() -> bool:
    """
    Whether each clause's own evaluator can be called, and matches the class.

    Returns
    -------
    agrees :
        True when both clauses expose an evaluator that returns what
        constructing them returns.
    """
    try:
        squashing = Form6Dot10NcRdClass1And2And3._evaluate(
            PROBE_AREA, PROBE_YIELD, PROBE_FACTOR
        )
        bending = Form6Dot14MCRdClass3._evaluate(
            PROBE_MODULUS, PROBE_YIELD, PROBE_FACTOR
        )
        declared_squashing = float(
            Form6Dot10NcRdClass1And2And3(
                a=PROBE_AREA, f_y=PROBE_YIELD, gamma_m0=PROBE_FACTOR
            )
        )
        declared_bending = float(
            Form6Dot14MCRdClass3(
                w_el_min=PROBE_MODULUS, f_y=PROBE_YIELD, gamma_m0=PROBE_FACTOR
            )
        )
    except (AttributeError, TypeError, ValueError):
        return False

    return squashing == declared_squashing and bending == declared_bending


EVALUATOR_REACHED = _evaluator_agrees()


def _probe_scalar(
    diameter: float,
    axial: float,
    moment: float,
    family: HostFamily,
) -> float:
    """
    The same utilization, evaluated without building the clause objects.

    Parameters
    ----------
    diameter :
        Outer diameter of the trial tube.
    axial :
        Axial force the member carries, negative in compression.
    moment :
        Bending moment the member carries, unsigned.
    family :
        The tube family the trial section is drawn from.

    Returns
    -------
    utilization :
        Demand over resistance, one at the fully stressed diameter.

    Notes
    -----
    **The same library, the same formulas, no object per evaluation.** Each
    clause here is a class whose constructor validates its arguments and then
    computes; a bisection asks for the same formula at fifty diameters per
    member, so the validation is paid fifty times for one answer while the
    arithmetic is paid once. This calls what the constructor calls.

    **What is given up is stated rather than hidden**: the clause's input
    validation does not run during the search. It runs on the answer —
    `_check_scalar` reads every reported utilization through the class itself,
    so any section this would have refused is refused where it is reported.
    The search is ours; the number that leaves is the library's.

    **And the saving is optional.** The evaluator is private to the library, so
    it is checked once at import for existing and for agreeing with its own
    class; where it does not, this falls through to constructing the clause and
    the answer is unchanged.
    """
    if not EVALUATOR_REACHED:
        return _check_scalar(diameter, axial, moment, family)

    area = family.area_coefficient * diameter**2
    modulus = family.modulus_coefficient * diameter**3
    squashing = Form6Dot10NcRdClass1And2And3._evaluate(
        area, family.f_y, family.gamma_m0
    )
    bending = Form6Dot14MCRdClass3._evaluate(modulus, family.f_y, family.gamma_m0)

    return abs(axial) / squashing + moment / bending


def _solve_scalar(axial: float, moment: float, family: HostFamily) -> float:
    """
    The diameter one member's check is exactly satisfied at.

    Parameters
    ----------
    axial :
        Axial force the member carries, negative in compression.
    moment :
        Demand moment the member carries, non-negative.
    family :
        The section family reduced to its host coefficients.

    Returns
    -------
    diameter :
        The root of `U(d) = 1`, or zero for an unloaded member.

    Notes
    -----
    A bisection in log-diameter over Blueprints residuals, on an exact
    bracket; the satisfied end comes back, so the re-read there is one minus
    a rounding error. Restated from `normax/sizing/blueprint.py`.
    """
    if moment < 0.0:
        raise ValueError(f"a moment of {moment} is signed: reduce before solving")

    scale_axial = family.gamma_m0 / (family.area_coefficient * family.f_y)
    scale_moment = family.gamma_m0 / (family.modulus_coefficient * family.f_y)
    demand_axial = abs(axial) * scale_axial
    demand_moment = moment * scale_moment
    if demand_axial == 0.0 and demand_moment == 0.0:
        return 0.0

    lower = max(math.sqrt(demand_axial), math.cbrt(demand_moment))
    upper = math.sqrt(2.0 * demand_axial) + math.cbrt(2.0 * demand_moment)
    low = math.log(lower)
    high = math.log(upper)
    for _ in range(BISECTION_HALVINGS):
        middle = 0.5 * (low + high)
        used = _probe_scalar(math.exp(middle), axial, moment, family)
        if used > 1.0:
            low = middle
        else:
            high = middle

    return math.exp(high)


def _demand_moment(inputs: dict[str, Any]) -> Float[np.ndarray, "members"]:
    """
    Reduce two end moments per axis to the one moment this check reads.

    Parameters
    ----------
    inputs :
        The validated input fields.

    Returns
    -------
    moment :
        The larger end moment in magnitude on each axis, summed over axes.
    """
    end_major = np.asarray(inputs["end_moments_major"], dtype=np.float64)
    end_minor = np.asarray(inputs["end_moments_minor"], dtype=np.float64)
    major = np.max(np.abs(end_major), axis=1)
    minor = np.max(np.abs(end_minor), axis=1)

    return major + minor


class SolvedState(NamedTuple):
    """
    One forward solve, held so the value and the adjoint read the same root.

    Attributes
    ----------
    family :
        The section family reduced to its host coefficients.
    floor :
        Smallest diameter the section family offers.
    axial :
        Axial force every member carries, negative in compression.
    moment :
        Demand moment every member carries, non-negative.
    unclamped :
        The root of each member's check, zero where a member is unloaded.
    diameter :
        The root with the catalogue floor applied.
    """

    family: HostFamily
    floor: float
    axial: Float[np.ndarray, "members"]
    moment: Float[np.ndarray, "members"]
    unclamped: Float[np.ndarray, "members"]
    diameter: Float[np.ndarray, "members"]


# One solved state, remembered between endpoint calls.
#
# **Both endpoints solve the same problem.** The forward pass bisects every
# member, and the reverse rule recomputes that same forward state before it can
# evaluate a single partial — so one gradient pays for two identical searches.
# The bisection is the whole cost of this check, so remembering the last one
# halves it.
#
# Keyed on every input the solve reads, by content: the schema rebuilds its
# payload per call, so nothing survives between them to compare by reference.
#
# **More than one entry, and that is not a guess.** Reverse-mode differentiation
# runs every forward call before any backward call, and one evaluation asks the
# check about more than one point — so a single entry is overwritten by the next
# forward pass before the matching reverse pass arrives, and never hits. The
# room here is for the points one evaluation holds at once; a miss only costs
# the search it would have saved, so being generous is free and being stingy is
# silent.
#
# ⚠ Safe because dispatch is serialized onto one owner thread. A served runtime
# would need a lock of its own.
SOLVED_ROOM = 8

_SOLVED: dict[bytes, SolvedState] = {}

# What the solve reads. `diameter_held` is deliberately absent: the held check
# consults it, the bisection never does.
SOLVE_FIELDS = ("axial_force", "end_moments_major", "end_moments_minor")
SOLVE_SCALARS = ("f_y", "gamma_m0", "ratio", "diameter_min")


def _solve_fingerprint(inputs: dict[str, Any]) -> bytes:
    """
    What identifies the problem a solved state belongs to.

    Parameters
    ----------
    inputs :
        The validated input fields.

    Returns
    -------
    fingerprint :
        A digest of everything the bisection reads.
    """
    digest = hashlib.blake2b(digest_size=32)
    for field in SOLVE_FIELDS:
        value = np.ascontiguousarray(inputs[field], dtype=np.float64)
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    for field in SOLVE_SCALARS:
        digest.update(repr(float(inputs[field])).encode())

    return digest.digest()


def _remembered_state(inputs: dict[str, Any]) -> SolvedState:
    """
    The solved state these inputs describe, searched for only once.

    Parameters
    ----------
    inputs :
        The validated input fields.

    Returns
    -------
    state :
        The sized members, freshly if the last question asked was a different one.
    """
    fingerprint = _solve_fingerprint(inputs)
    held = _SOLVED.get(fingerprint)
    if held is not None:
        return held

    state = _solved_state(inputs)
    if len(_SOLVED) >= SOLVED_ROOM:
        _SOLVED.pop(next(iter(_SOLVED)))
    _SOLVED[fingerprint] = state

    return state


def _solved_state(inputs: dict[str, Any]) -> SolvedState:
    """
    Solve every member once — the one place the bracket and clamp are applied.

    Parameters
    ----------
    inputs :
        The validated input fields.

    Returns
    -------
    state :
        Everything downstream of the bisection, for the value and the adjoint
        alike: linearizing at any other root would be a wrong derivative the
        value-parity tests cannot see.

    Raises
    ------
    ValueError
        If the catalogue floor is not a positive diameter.
    """
    family = _host_family(inputs)
    floor = float(inputs["diameter_min"])
    if floor <= 0.0:
        raise ValueError(f"a floor of {floor} is no diameter: need one above zero")

    axial = np.asarray(inputs["axial_force"], dtype=np.float64)
    moment = _demand_moment(inputs)
    paired = zip(axial, moment, strict=True)
    solved = [_solve_scalar(force, bent, family) for force, bent in paired]
    unclamped = np.asarray(solved, dtype=np.float64)
    diameter = np.maximum(unclamped, floor)

    return SolvedState(family, floor, axial, moment, unclamped, diameter)


def _forward_pass(
    inputs: dict[str, Any],
    *,
    diagnostics: bool,
) -> dict[str, np.ndarray]:
    """
    Size every member, entirely on the host.

    Parameters
    ----------
    inputs :
        The validated input fields.
    diagnostics :
        Whether to report the clamp mask, which is not differentiated and so
        is left out of every gradient endpoint.

    Returns
    -------
    outputs :
        The output fields, the diagnostic included only when asked for.
    """
    state = _remembered_state(inputs)

    tripled = zip(state.diameter, state.axial, state.moment, strict=True)
    used = [
        _check_scalar(size, force, bent, state.family) for size, force, bent in tripled
    ]
    utilization = np.asarray(used, dtype=np.float64)

    held = np.asarray(inputs["diameter_held"], dtype=np.float64)
    held_tripled = zip(held, state.axial, state.moment, strict=True)
    checked = [
        _check_scalar(size, force, bent, state.family)
        for size, force, bent in held_tripled
    ]
    utilization_held = np.asarray(checked, dtype=np.float64)

    outputs = {
        "diameter": state.diameter,
        "utilization": utilization,
        "utilization_held": utilization_held,
    }
    if diagnostics:
        outputs["clamped"] = (state.unclamped < state.floor).astype(np.float64)

    return outputs


def apply(inputs: InputSchema) -> OutputSchema:
    """
    Run the check.

    Parameters
    ----------
    inputs :
        The member actions and the family.

    Returns
    -------
    outputs :
        The required sizes, the utilization and the clamp mask.
    """
    return _forward_pass(inputs.model_dump(), diagnostics=True)


def abstract_eval(abstract_inputs):
    """
    Output shapes and dtypes, without sizing anything.

    Parameters
    ----------
    abstract_inputs :
        The input fields, arrays replaced by their shape and dtype.

    Returns
    -------
    outputs :
        A shape and a dtype for every output field.
    """
    members = abstract_inputs.axial_force.shape[0]

    return {
        "diameter": {"shape": (members,), "dtype": "float64"},
        "utilization": {"shape": (members,), "dtype": "float64"},
        "utilization_held": {"shape": (members,), "dtype": "float64"},
        "clamped": {"shape": (members,), "dtype": "float64"},
    }


def _hand_partials(inputs: dict[str, Any]) -> HandPartials:
    """
    Recompute the forward state and every partial the adjoint reads.

    Parameters
    ----------
    inputs :
        The validated input fields.

    Returns
    -------
    partials :
        The closed-form derivatives at the solved state, and the end-moment
        routing each axis's cotangent follows.

    Notes
    -----
    The implicit function theorem at the root `U(D; N, M) = 1`: with the
    check's partials `U_d`, `U_N` and `U_M` in closed form, the size moves as
    `-U_N/U_d` and `-U_M/U_d` wherever the check decided it, and not at all
    where the catalogue minimum did. The reported utilization is the mirror
    image: pinned at one where the check decided, so its derivative is exactly
    zero there, and the bare partial at the floor where the clamp did.
    """
    state = _remembered_state(inputs)
    family = state.family
    axial = state.axial
    unclamped = state.unclamped
    floor = state.floor
    end_major = np.asarray(inputs["end_moments_major"], dtype=np.float64)
    end_minor = np.asarray(inputs["end_moments_minor"], dtype=np.float64)

    scale_axial = family.gamma_m0 / (family.area_coefficient * family.f_y)
    scale_moment = family.gamma_m0 / (family.modulus_coefficient * family.f_y)
    demand_axial = np.abs(axial) * scale_axial
    demand_moment = state.moment * scale_moment

    rooted = unclamped > 0.0
    safe = np.where(rooted, unclamped, 1.0)
    slope = -(2.0 * demand_axial / safe**3 + 3.0 * demand_moment / safe**4)
    divisor = np.where(rooted, slope, -1.0)

    free = rooted & (unclamped >= floor)
    partial_axial = np.sign(axial) * scale_axial / safe**2
    partial_moment = scale_moment / safe**3
    size_axial = np.where(free, -partial_axial / divisor, 0.0)
    size_moment = np.where(free, -partial_moment / divisor, 0.0)

    bound = unclamped < floor
    check_axial = np.where(bound, np.sign(axial) * scale_axial / floor**2, 0.0)
    check_moment = np.where(bound, scale_moment / floor**3, 0.0)

    # The held check's partials, at the size that crossed rather than a root.
    held = np.asarray(inputs["diameter_held"], dtype=np.float64)
    positive = held > 0.0
    steady = np.where(positive, held, 1.0)
    held_pull = 2.0 * demand_axial / steady**3 + 3.0 * demand_moment / steady**4
    held_slope = np.where(positive, -held_pull, 0.0)
    held_axial = np.where(positive, np.sign(axial) * scale_axial / steady**2, 0.0)
    held_moment = np.where(positive, scale_moment / steady**3, 0.0)

    winner_major = np.argmax(np.abs(end_major), axis=1)
    winner_minor = np.argmax(np.abs(end_minor), axis=1)
    rows = np.arange(axial.shape[0])
    sign_major = np.sign(end_major[rows, winner_major])
    sign_minor = np.sign(end_minor[rows, winner_minor])

    return HandPartials(
        size_axial,
        size_moment,
        check_axial,
        check_moment,
        held_slope,
        held_axial,
        held_moment,
        winner_major,
        winner_minor,
        sign_major,
        sign_minor,
    )


def _refuse_diagnostics(outputs: list[str]) -> None:
    """
    Refuse a derivative of the clamp mask, loudly rather than as a quiet zero.

    Parameters
    ----------
    outputs :
        Names of the output fields a derivative was requested of.

    Raises
    ------
    ValueError
        If the clamp mask is among them.
    """
    if "clamped" in outputs:
        raise ValueError(
            "`clamped` is non-differentiable; drop it before differentiating"
        )


class AdjointState(NamedTuple):
    """
    Everything both derivative endpoints read, built once for either.

    Attributes
    ----------
    partials :
        The closed-form derivatives at the solved state.
    rows :
        Member indices, for routing an end-moment cotangent to its winner.
    axial_pulls :
        Each differentiable output's partial in the axial force, by name.
    moment_pulls :
        Each differentiable output's partial in the demand moment, by name.
    held_pulls :
        Each differentiable output's partial in the held diameter, by name —
        zero for both solve outputs, since the solve never reads it.
    """

    partials: HandPartials
    rows: Int[np.ndarray, "members"]
    axial_pulls: dict[str, Float[np.ndarray, "members"]]
    moment_pulls: dict[str, Float[np.ndarray, "members"]]
    held_pulls: dict[str, Float[np.ndarray, "members"]]


def _adjoint_state(inputs: InputSchema, outputs: list[str]) -> AdjointState:
    """
    The shared preamble of both derivative endpoints.

    Parameters
    ----------
    inputs :
        The member actions and the family.
    outputs :
        Names of the output fields a derivative was requested of.

    Returns
    -------
    state :
        The partials and the per-output pull tables, refusal already applied.

    Raises
    ------
    ValueError
        If the clamp mask is among the requested outputs.
    """
    _refuse_diagnostics(outputs)
    raw = inputs.model_dump()
    partials = _hand_partials(raw)
    rows = np.arange(np.asarray(raw["axial_force"]).shape[0])

    axial_pulls = {
        "diameter": partials.size_axial,
        "utilization": partials.check_axial,
        "utilization_held": partials.held_axial,
    }
    moment_pulls = {
        "diameter": partials.size_moment,
        "utilization": partials.check_moment,
        "utilization_held": partials.held_moment,
    }
    unread = np.zeros(rows.shape[0])
    held_pulls = {
        "diameter": unread,
        "utilization": unread,
        "utilization_held": partials.held_slope,
    }

    return AdjointState(partials, rows, axial_pulls, moment_pulls, held_pulls)


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: list[str],
    vjp_outputs: list[str],
    cotangent_vector: dict[str, Any],
):
    """
    Pull a cotangent on the outputs back to the inputs, by hand.

    Parameters
    ----------
    inputs :
        The member actions and the family.
    vjp_inputs :
        Names of the input fields a derivative is taken with respect to.
    vjp_outputs :
        Names of the output fields a derivative is taken of.
    cotangent_vector :
        Cotangent on each of those outputs.

    Returns
    -------
    cotangents :
        Cotangent on each of the requested inputs.

    Notes
    -----
    What `jax.grad` calls, and the only endpoint it calls — and here it is
    literal arithmetic rather than a delegation: every map is diagonal over
    members, so each requested input accumulates `partial * cotangent` per
    output, and each axis's moment cotangent routes to the larger end with
    that end's sign. No tracer is ever constructed.
    """
    state = _adjoint_state(inputs, vjp_outputs)
    partials = state.partials
    rows = state.rows
    members = rows.shape[0]
    axial_pulls = state.axial_pulls
    moment_pulls = state.moment_pulls

    cotangents = {}
    for name in vjp_inputs:
        if name == "axial_force":
            gathered = np.zeros(members)
            for output in vjp_outputs:
                pulled = np.asarray(cotangent_vector[output], dtype=np.float64)
                gathered = gathered + axial_pulls[output] * pulled
            cotangents[name] = gathered
        elif name == "diameter_held":
            gathered = np.zeros(members)
            for output in vjp_outputs:
                pulled = np.asarray(cotangent_vector[output], dtype=np.float64)
                gathered = gathered + state.held_pulls[output] * pulled
            cotangents[name] = gathered
        elif name == "end_moments_major":
            gathered = np.zeros((members, 2))
            for output in vjp_outputs:
                pulled = np.asarray(cotangent_vector[output], dtype=np.float64)
                routed = moment_pulls[output] * partials.sign_major * pulled
                gathered[rows, partials.winner_major] += routed
            cotangents[name] = gathered
        elif name == "end_moments_minor":
            gathered = np.zeros((members, 2))
            for output in vjp_outputs:
                pulled = np.asarray(cotangent_vector[output], dtype=np.float64)
                routed = moment_pulls[output] * partials.sign_minor * pulled
                gathered[rows, partials.winner_minor] += routed
            cotangents[name] = gathered
        else:
            raise ValueError(f"no hand-written derivative covers `{name}`")

    return cotangents


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: list[str],
    jvp_outputs: list[str],
    tangent_vector: dict[str, Any],
):
    """
    Push a tangent on the inputs forward to the outputs, by hand.

    Parameters
    ----------
    inputs :
        The member actions and the family.
    jvp_inputs :
        Names of the input fields a derivative is taken with respect to.
    jvp_outputs :
        Names of the output fields a derivative is taken of.
    tangent_vector :
        Tangent on each of those inputs.

    Returns
    -------
    tangents :
        Tangent on each of the requested outputs.

    Notes
    -----
    Never reached by `jax.grad`, and provided because it costs the same
    coefficients pushed the other way and cross-checks the reverse rule.
    """
    state = _adjoint_state(inputs, jvp_outputs)
    partials = state.partials
    rows = state.rows
    members = rows.shape[0]
    axial_pulls = state.axial_pulls
    moment_pulls = state.moment_pulls

    moment_dot = np.zeros(members)
    if "end_moments_major" in jvp_inputs:
        seeded = np.asarray(tangent_vector["end_moments_major"], dtype=np.float64)
        winning = seeded[rows, partials.winner_major]
        moment_dot = moment_dot + partials.sign_major * winning
    if "end_moments_minor" in jvp_inputs:
        seeded = np.asarray(tangent_vector["end_moments_minor"], dtype=np.float64)
        winning = seeded[rows, partials.winner_minor]
        moment_dot = moment_dot + partials.sign_minor * winning

    tangents = {}
    for output in jvp_outputs:
        pushed = moment_pulls[output] * moment_dot
        if "axial_force" in jvp_inputs:
            seeded = np.asarray(tangent_vector["axial_force"], dtype=np.float64)
            pushed = pushed + axial_pulls[output] * seeded
        if "diameter_held" in jvp_inputs:
            seeded = np.asarray(tangent_vector["diameter_held"], dtype=np.float64)
            pushed = pushed + state.held_pulls[output] * seeded
        tangents[output] = pushed

    return tangents


def jacobian(
    inputs: InputSchema,
    jac_inputs: set[str],
    jac_outputs: set[str],
) -> dict[str, dict[str, np.ndarray]]:
    """
    Materialize every requested derivative block, by hand.

    Parameters
    ----------
    inputs :
        The member actions and the family.
    jac_inputs :
        Names of the input fields a derivative is taken with respect to.
    jac_outputs :
        Names of the output fields a derivative is taken of.

    Returns
    -------
    blocks :
        One array per (output, input) pair, keyed output first, each shaped
        as the output's shape followed by the input's.

    Raises
    ------
    ValueError
        If the clamp mask is requested, or an input has no derivative rule.

    Notes
    -----
    What a batched `jax.jacrev` or `jax.jacfwd` calls once per Jacobian:
    tesseract-jax routes stacked (co)tangents here whenever this endpoint
    exists, contracting the blocks client-side, so one crossing replaces one
    per constraint row. The caller requests every differentiable output
    whether or not its cotangent is live, and the runtime holds the returned
    keys to exactly the requested sets.

    The blocks are the same per-member pulls the two product endpoints
    contract, written out: every map is diagonal over members, so each block
    is zeros with one entry per member — on the diagonal for the vectors, at
    the winning end column with that end's sign for the moments. No matmul,
    no re-solve, no tracer.

    The name sets arrive unordered and are iterated sorted, which fixes the
    construction order without touching the keyed result.
    """
    outputs = sorted(jac_outputs)
    state = _adjoint_state(inputs, outputs)
    partials = state.partials
    rows = state.rows
    members = rows.shape[0]

    blocks: dict[str, dict[str, np.ndarray]] = {}
    for output in outputs:
        per_input = {}
        for name in sorted(jac_inputs):
            if name == "axial_force":
                block = np.zeros((members, members))
                block[rows, rows] = state.axial_pulls[output]
            elif name == "diameter_held":
                block = np.zeros((members, members))
                block[rows, rows] = state.held_pulls[output]
            elif name == "end_moments_major":
                block = np.zeros((members, members, 2))
                routed = state.moment_pulls[output] * partials.sign_major
                block[rows, rows, partials.winner_major] = routed
            elif name == "end_moments_minor":
                block = np.zeros((members, members, 2))
                routed = state.moment_pulls[output] * partials.sign_minor
                block[rows, rows, partials.winner_minor] = routed
            else:
                raise ValueError(f"no hand-written derivative covers `{name}`")
            per_input[name] = block
        blocks[output] = per_input

    return blocks
