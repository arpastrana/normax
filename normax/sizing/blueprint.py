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
Blueprints' EN 1993-1-1 cross-section check, hosted once and reached two ways.

Blueprints is a scalar Python library whose formula classes subclass `float`
and carry no derivatives. Everything here up to the sizer is plain NumPy on
the host: the check through the library's eq. (6.10) and (6.14), a bisection
to the size that is worked to exactly one, and the adjoint of both by hand —
the implicit function theorem at the root, the bare partials at a held size.
The Tesseract wraps these functions behind its schema, and the in-process
sizer wraps the same functions behind `jax.pure_callback`, so the two answers
are identical by construction rather than by a parity test.

The check is axial force with bending at cross-section level and nothing else:
Blueprints implements no §6.3.1 flexural buckling, so no buckling length is
read, and shear, torsion and their interaction are declined by a measured
decision recorded in the project notes. Blueprints is LGPL-2.1 and is imported
unmodified as a pip package.
"""

import functools
import hashlib
import math
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from blueprints.codes.eurocode.en_1993_1_1_2005.chapter_6_ultimate_limit_state.formula_6_10 import (  # noqa: E501
    Form6Dot10NcRdClass1And2And3,
)
from blueprints.codes.eurocode.en_1993_1_1_2005.chapter_6_ultimate_limit_state.formula_6_14 import (  # noqa: E501
    Form6Dot14MCRdClass3,
)
from jaxtyping import Array
from jaxtyping import Bool
from jaxtyping import Float
from jaxtyping import Int

from normax.analysis import MemberForces
from normax.sections import TubeFamily
from normax.sizing import AbstractMemberSizer
from normax.sizing import MemberSizes
from normax.structures import Structure

# EN 1993-1-1 §6.1, the recommended value.
GAMMA_M0 = 1.0

# The smallest tube the section family offers, the catalogue floor.
DIAMETER_MINIMUM = 21.3

# Halvings of a log-diameter bracket at most sqrt(2) + cbrt(2) wide.
BISECTION_HALVINGS = 50

# How many solved states are remembered between a forward and its reverse pass.
SOLVED_ROOM = 32

# A probe point at which the private evaluator is compared to its public class.
PROBE_AREA = 1.2e3
PROBE_MODULUS = 5.0e4
PROBE_YIELD = 355.0
PROBE_FACTOR = 1.0


class HostFamily(NamedTuple):
    """
    A section family as the host check reads it: concrete numbers only.

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
    floor :
        Smallest diameter the family offers.
    """

    area_coefficient: float
    modulus_coefficient: float
    f_y: float
    gamma_m0: float
    floor: float


def host_family(
    ratio: float,
    f_y: float,
    gamma_m0: float = GAMMA_M0,
    floor: float = DIAMETER_MINIMUM,
) -> HostFamily:
    """
    Reduce a section family to the coefficients the scalar check reads.

    Parameters
    ----------
    ratio :
        Diameter over wall thickness of every tube in the family.
    f_y :
        Yield strength of the family's grade.
    gamma_m0 :
        Partial factor for cross-section resistance.
    floor :
        Smallest diameter the family offers.

    Returns
    -------
    family :
        The family's geometry collapsed to two proportionality constants.

    Raises
    ------
    ValueError
        If the ratio leaves no wall, or the floor is no diameter.

    Notes
    -----
    With the wall a fixed proportion of the diameter, `A = c_A d^2` and
    `W_el = c_W d^3`. The arithmetic mirrors `MemberSections` at a unit
    diameter; Blueprints' own meshed CHS profiles are deliberately not used.
    """
    if ratio <= 2.0:
        raise ValueError(f"a ratio of {ratio} leaves no wall: need d/t > 2")
    if floor <= 0.0:
        raise ValueError(f"a floor of {floor} is no diameter: need one above zero")

    wall = 1.0 / ratio
    bore = 1.0 - 2.0 * wall
    area_coefficient = math.pi * wall * (1.0 - wall)
    second_moment = (math.pi / 64.0) * (1.0 - bore**4)
    modulus_coefficient = 2.0 * second_moment

    return HostFamily(
        area_coefficient, modulus_coefficient, float(f_y), float(gamma_m0), float(floor)
    )


def snapshot_family(family: TubeFamily) -> tuple[float, float]:
    """
    Snapshot a family's ratio and yield strength for the host.

    Parameters
    ----------
    family :
        The section family a sizer is built over.

    Returns
    -------
    ratio :
        The family's wall proportion, as a concrete float.
    f_y :
        The family's yield strength, as a concrete float.

    Raises
    ------
    ValueError
        If the family's ratio leaves no wall at all.

    Notes
    -----
    The two numbers a host check reads off a family, concretized once at
    construction so no material sensitivity flows through a sizer — the
    in-process one and the crossed one snapshot identically.
    """
    ratio = float(family.ratio)
    f_y = float(family.material.f_y)
    host_family(ratio, f_y)

    return ratio, f_y


def _check_scalar(
    diameter: float,
    axial: float,
    moment: float,
    family: HostFamily,
) -> float:
    """
    One member's utilization at one diameter, through Blueprints' classes.

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
        Demand over resistance, the linear sum of EN 1993-1-1 eq. (6.2), with
        eq. (6.10) and eq. (6.14) supplying the resistances.

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


def _evaluator_agrees() -> bool:
    """
    Whether each clause's private evaluator is callable and matches its class.

    Returns
    -------
    agrees :
        True when both evaluators return what constructing the clause returns.

    Notes
    -----
    A fast path and never a correctness requirement: the clause's constructor
    validates and then calls this same evaluator, so bisecting through it skips
    the object per trial. Where a release moves it, the check falls back to
    the class and the answer is unchanged.
    """
    try:
        squashing = Form6Dot10NcRdClass1And2And3._evaluate(
            PROBE_AREA, PROBE_YIELD, PROBE_FACTOR
        )
        bending = Form6Dot14MCRdClass3._evaluate(
            PROBE_MODULUS, PROBE_YIELD, PROBE_FACTOR
        )
        declared_squashing = Form6Dot10NcRdClass1And2And3(
            a=PROBE_AREA, f_y=PROBE_YIELD, gamma_m0=PROBE_FACTOR
        )
        declared_bending = Form6Dot14MCRdClass3(
            w_el_min=PROBE_MODULUS, f_y=PROBE_YIELD, gamma_m0=PROBE_FACTOR
        )
    except (AttributeError, TypeError, ValueError):
        return False

    return squashing == float(declared_squashing) and bending == float(declared_bending)


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
        Demand moment the member carries, non-negative.
    family :
        The section family reduced to its host coefficients.

    Returns
    -------
    utilization :
        Demand over resistance.

    Notes
    -----
    The clause's input validation is skipped during the search and runs on the
    answer instead, since every reported value goes through the class itself.
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


def _demand_scales(family: HostFamily) -> tuple[float, float]:
    """
    The factors turning a force and a moment into diameter-unit demands.
    """
    scale_axial = family.gamma_m0 / (family.area_coefficient * family.f_y)
    scale_moment = family.gamma_m0 / (family.modulus_coefficient * family.f_y)

    return scale_axial, scale_moment


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
    `U(d) = a/d^2 + b/d^3` is strictly decreasing, so the root is unique and
    bisection is safe. The bracket is exact: `max(sqrt(a), cbrt(b))` puts one
    term at one, `sqrt(2a) + cbrt(2b)` puts each at or under one half. The
    satisfied end comes back, so the re-read there is one minus rounding.
    """
    if moment < 0.0:
        raise ValueError(f"a moment of {moment} is signed: reduce before solving")

    scale_axial, scale_moment = _demand_scales(family)
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


def _solve_batch(
    axial: Float[np.ndarray, "*load_cases members"],
    moment: Float[np.ndarray, "*load_cases members"],
    family: HostFamily,
) -> Float[np.ndarray, "*load_cases members"]:
    """
    Every member's exactly-satisfied diameter, one host loop.
    """
    paired = zip(axial.ravel(), moment.ravel(), strict=True)
    solved = [_solve_scalar(force, bent, family) for force, bent in paired]

    return np.asarray(solved, dtype=np.float64).reshape(axial.shape)


def _check_batch(
    diameter: Float[np.ndarray, "*load_cases members"],
    axial: Float[np.ndarray, "*load_cases members"],
    moment: Float[np.ndarray, "*load_cases members"],
    family: HostFamily,
) -> Float[np.ndarray, "*load_cases members"]:
    """
    Every member's utilization at a given diameter, one host loop.
    """
    tripled = zip(diameter.ravel(), axial.ravel(), moment.ravel(), strict=True)
    used = [_check_scalar(size, force, bent, family) for size, force, bent in tripled]

    return np.asarray(used, dtype=np.float64).reshape(diameter.shape)


class HostActions(NamedTuple):
    """
    What every member carries under one or more load cases, on the host.

    Attributes
    ----------
    axial :
        Axial force of every member, tension positive.
    end_major :
        Major-axis moment at each end of every member.
    end_minor :
        Minor-axis moment at each end of every member.
    """

    axial: Float[np.ndarray, "*load_cases members"]
    end_major: Float[np.ndarray, "*load_cases members ends"]
    end_minor: Float[np.ndarray, "*load_cases members ends"]


def host_actions(
    axial: Float[Array, "*load_cases members"],
    end_major: Float[Array, "*load_cases members ends"],
    end_minor: Float[Array, "*load_cases members ends"],
) -> HostActions:
    """
    Bring three arrays of any provenance to the host as contiguous float64.
    """
    return HostActions(
        np.ascontiguousarray(axial, dtype=np.float64),
        np.ascontiguousarray(end_major, dtype=np.float64),
        np.ascontiguousarray(end_minor, dtype=np.float64),
    )


class WinningEnd(NamedTuple):
    """
    Which end carries the larger moment on one axis, and that moment's sign.

    Attributes
    ----------
    winner :
        Index of the end whose moment is larger in magnitude.
    sign :
        Sign of that moment.
    """

    winner: Int[np.ndarray, "*load_cases members"]
    sign: Float[np.ndarray, "*load_cases members"]


def _winning_end(ends: Float[np.ndarray, "*load_cases members ends"]) -> WinningEnd:
    """
    Pick the end whose moment is larger in magnitude on one axis.
    """
    winner = np.argmax(np.abs(ends), axis=-1)
    winning = np.take_along_axis(ends, winner[..., None], axis=-1)[..., 0]

    return WinningEnd(winner, np.sign(winning))


class DemandMoment(NamedTuple):
    """
    The one moment the check reads, and where it came from.

    Attributes
    ----------
    moment :
        The larger end moment in magnitude on each axis, summed over axes.
    major :
        The end that won on the major axis, and its sign.
    minor :
        The end that won on the minor axis, and its sign.
    """

    moment: Float[np.ndarray, "*load_cases members"]
    major: WinningEnd
    minor: WinningEnd


def demand_moment(actions: HostActions) -> DemandMoment:
    """
    Reduce two end moments per axis to the one moment this check reads.

    Parameters
    ----------
    actions :
        What every member carries.

    Returns
    -------
    demand :
        The reduced moment, with the winning end and its sign per axis so
        that a cotangent can be routed back where it came from.

    Notes
    -----
    A modeling choice rather than a clause: the check is read at the worse
    end, the two axes superposed linearly per EN 1993-1-1 eq. (6.2).
    """
    major = _winning_end(actions.end_major)
    minor = _winning_end(actions.end_minor)
    moment = np.max(np.abs(actions.end_major), axis=-1)
    moment = moment + np.max(np.abs(actions.end_minor), axis=-1)

    return DemandMoment(moment, major, minor)


class SolvedState(NamedTuple):
    """
    One forward solve, held so the value and the adjoint read the same root.

    Attributes
    ----------
    family :
        The section family reduced to its host coefficients.
    axial :
        Axial force every member carries, negative in compression.
    demand :
        The reduced moment and its routing.
    unclamped :
        The root of each member's check, zero where a member is unloaded.
    diameter :
        The root with the catalogue floor applied.
    """

    family: HostFamily
    axial: Float[np.ndarray, "*load_cases members"]
    demand: DemandMoment
    unclamped: Float[np.ndarray, "*load_cases members"]
    diameter: Float[np.ndarray, "*load_cases members"]


# Solved states by fingerprint, so a reverse pass never re-bisects.
_SOLVED: dict[bytes, SolvedState] = {}


def _solve_fingerprint(actions: HostActions, family: HostFamily) -> bytes:
    """
    A digest of everything the bisection reads, by content.
    """
    digest = hashlib.blake2b(digest_size=32)
    for value in actions:
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    digest.update(repr(tuple(family)).encode())

    return digest.digest()


def solved_state(actions: HostActions, family: HostFamily) -> SolvedState:
    """
    The solved state these actions describe, searched for only once.

    Parameters
    ----------
    actions :
        What every member carries.
    family :
        The section family reduced to its host coefficients.

    Returns
    -------
    state :
        The sized members, remembered by content so that the reverse pass
        linearizes at the very root the forward pass reported.

    Notes
    -----
    Reverse mode runs every forward call before any backward call, so several
    states are held; a miss only costs the search it would have saved.
    """
    fingerprint = _solve_fingerprint(actions, family)
    held = _SOLVED.get(fingerprint)
    if held is not None:
        return held

    demand = demand_moment(actions)
    unclamped = _solve_batch(actions.axial, demand.moment, family)
    diameter = np.maximum(unclamped, family.floor)
    state = SolvedState(family, actions.axial, demand, unclamped, diameter)
    if len(_SOLVED) >= SOLVED_ROOM:
        _SOLVED.pop(next(iter(_SOLVED)))
    _SOLVED[fingerprint] = state

    return state


class SizedMembers(NamedTuple):
    """
    What the sizing map answers.

    Attributes
    ----------
    diameter :
        Outer diameter of every member, floored at the family's minimum.
    utilization :
        Demand over resistance at that diameter — one where the check decided
        the size, below one where the floor did.
    clamped :
        Whether the floor decided each member's size.
    """

    diameter: Float[np.ndarray, "*load_cases members"]
    utilization: Float[np.ndarray, "*load_cases members"]
    clamped: Bool[np.ndarray, "*load_cases members"]


def size_members(actions: HostActions, family: HostFamily) -> SizedMembers:
    """
    Size every member to the check, entirely on the host.

    Parameters
    ----------
    actions :
        What every member carries.
    family :
        The section family reduced to its host coefficients.

    Returns
    -------
    sized :
        The floored diameters, the utilization re-read at them, and the mask
        of members the floor decided.
    """
    state = solved_state(actions, family)
    used = _check_batch(state.diameter, state.axial, state.demand.moment, family)
    clamped = state.unclamped < family.floor

    return SizedMembers(state.diameter, used, clamped)


def check_members(
    diameter_held: Float[np.ndarray, "*load_cases members"],
    actions: HostActions,
    family: HostFamily,
) -> Float[np.ndarray, "*load_cases members"]:
    """
    Check sizes the caller owns, entirely on the host.

    Parameters
    ----------
    diameter_held :
        Outer diameter every member is checked at.
    actions :
        What every member carries.
    family :
        The section family reduced to its host coefficients.

    Returns
    -------
    utilization_held :
        Demand over resistance of every member at the held size.
    """
    held = np.asarray(diameter_held, dtype=np.float64)
    demand = demand_moment(actions)

    return _check_batch(held, actions.axial, demand.moment, family)


class CheckPartials(NamedTuple):
    """
    The check's closed-form partials at one size, zero where there is no size.

    Attributes
    ----------
    slope :
        Derivative of the check in the diameter, strictly negative when sized.
    axial :
        Derivative of the check in the axial force.
    moment :
        Derivative of the check in the demand moment.
    """

    slope: Float[np.ndarray, "*load_cases members"]
    axial: Float[np.ndarray, "*load_cases members"]
    moment: Float[np.ndarray, "*load_cases members"]


def _check_partials(
    size: Float[np.ndarray, "*load_cases members"],
    axial: Float[np.ndarray, "*load_cases members"],
    moment: Float[np.ndarray, "*load_cases members"],
    family: HostFamily,
) -> CheckPartials:
    """
    The three partials of `U = a/d^2 + b/d^3`, evaluated at a given size.
    """
    scale_axial, scale_moment = _demand_scales(family)
    demand_axial = np.abs(axial) * scale_axial
    demand_moment_units = moment * scale_moment
    positive = size > 0.0
    safe = np.where(positive, size, 1.0)
    pull = 2.0 * demand_axial / safe**3 + 3.0 * demand_moment_units / safe**4
    slope = np.where(positive, -pull, 0.0)
    partial_axial = np.where(positive, np.sign(axial) * scale_axial / safe**2, 0.0)
    partial_moment = np.where(positive, scale_moment / safe**3, 0.0)

    return CheckPartials(slope, partial_axial, partial_moment)


class ActionCotangents(NamedTuple):
    """
    A cotangent pulled back onto what every member carries.

    Attributes
    ----------
    axial :
        Cotangent on the axial force.
    end_major :
        Cotangent on the major-axis end moments.
    end_minor :
        Cotangent on the minor-axis end moments.
    """

    axial: Float[np.ndarray, "*load_cases members"]
    end_major: Float[np.ndarray, "*load_cases members ends"]
    end_minor: Float[np.ndarray, "*load_cases members ends"]


def _route_axis(
    pull: Float[np.ndarray, "*load_cases members"],
    end: WinningEnd,
) -> Float[np.ndarray, "*load_cases members ends"]:
    """
    Place a signed cotangent at the winning end of one axis, zero at the other.
    """
    ends = np.zeros((*pull.shape, 2))
    np.put_along_axis(
        ends, end.winner[..., None], (pull * end.sign)[..., None], axis=-1
    )

    return ends


def _action_cotangents(
    pull_axial: Float[np.ndarray, "*load_cases members"],
    pull_moment: Float[np.ndarray, "*load_cases members"],
    demand: DemandMoment,
) -> ActionCotangents:
    """
    Route a cotangent on the demand moment to the winning end of each axis.
    """
    end_major = _route_axis(pull_moment, demand.major)
    end_minor = _route_axis(pull_moment, demand.minor)

    return ActionCotangents(pull_axial, end_major, end_minor)


class SizeCotangents(NamedTuple):
    """
    A cotangent on the two outputs of the sizing map.

    Attributes
    ----------
    diameter :
        Cotangent on the floored diameter.
    utilization :
        Cotangent on the utilization re-read at it.
    """

    diameter: Float[np.ndarray, "*load_cases members"]
    utilization: Float[np.ndarray, "*load_cases members"]


def size_cotangents(
    actions: HostActions,
    family: HostFamily,
    cotangents: SizeCotangents,
) -> ActionCotangents:
    """
    Pull a cotangent on the sizing map's outputs back to the actions, by hand.

    Parameters
    ----------
    actions :
        What every member carries.
    family :
        The section family reduced to its host coefficients.
    cotangents :
        Cotangent on the diameter and on the utilization.

    Returns
    -------
    pulled :
        Cotangent on the axial force and on both end-moment arrays.

    Notes
    -----
    The implicit function theorem at the root `U(D; N, M) = 1`: the size moves
    as `-U_N/U_d` and `-U_M/U_d` where the check decided it, and not at all
    where the floor did. The reported utilization is the mirror image: pinned
    at one where the check decided, the bare partial at the floor otherwise.
    """
    state = solved_state(actions, family)
    demand = state.demand
    at_root = _check_partials(state.unclamped, state.axial, demand.moment, family)
    at_floor = _check_partials(state.diameter, state.axial, demand.moment, family)

    free = state.unclamped >= family.floor
    divisor = np.where(free, at_root.slope, -1.0)
    size_axial = np.where(free, -at_root.axial / divisor, 0.0)
    size_moment = np.where(free, -at_root.moment / divisor, 0.0)
    check_axial = np.where(free, 0.0, at_floor.axial)
    check_moment = np.where(free, 0.0, at_floor.moment)

    pulled_diameter = np.asarray(cotangents.diameter, dtype=np.float64)
    pulled_used = np.asarray(cotangents.utilization, dtype=np.float64)
    axial = size_axial * pulled_diameter + check_axial * pulled_used
    moment = size_moment * pulled_diameter + check_moment * pulled_used

    return _action_cotangents(axial, moment, demand)


class HeldCotangents(NamedTuple):
    """
    A cotangent on the held check pulled back to everything it reads.

    Attributes
    ----------
    diameter_held :
        Cotangent on the held diameter.
    actions :
        Cotangent on the axial force and on both end-moment arrays.
    """

    diameter_held: Float[np.ndarray, "*load_cases members"]
    actions: ActionCotangents


def check_cotangents(
    diameter_held: Float[np.ndarray, "*load_cases members"],
    actions: HostActions,
    family: HostFamily,
    cotangent: Float[np.ndarray, "*load_cases members"],
) -> HeldCotangents:
    """
    Pull a cotangent on the held check back to the size and the actions.

    Parameters
    ----------
    diameter_held :
        Outer diameter every member was checked at.
    actions :
        What every member carries.
    family :
        The section family reduced to its host coefficients.
    cotangent :
        Cotangent on the held utilization.

    Returns
    -------
    pulled :
        Cotangent on the held diameter, and on the actions.

    Notes
    -----
    Explicit arithmetic with no root find, so the adjoint is the check's bare
    partials at the size that was held.
    """
    held = np.asarray(diameter_held, dtype=np.float64)
    pulled = np.asarray(cotangent, dtype=np.float64)
    demand = demand_moment(actions)
    partials = _check_partials(held, actions.axial, demand.moment, family)
    on_actions = _action_cotangents(
        partials.axial * pulled, partials.moment * pulled, demand
    )

    return HeldCotangents(partials.slope * pulled, on_actions)


def _float_struct(shape: tuple[int, ...]) -> jax.ShapeDtypeStruct:
    """
    The abstract float64 array a host callback promises to return.
    """
    return jax.ShapeDtypeStruct(shape, jnp.float64)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def sized_members(
    family: HostFamily,
    axial: Float[Array, "*load_cases members"],
    end_major: Float[Array, "*load_cases members ends"],
    end_minor: Float[Array, "*load_cases members ends"],
) -> tuple[Float[Array, "*load_cases members"], Float[Array, "*load_cases members"]]:
    """
    The host sizing map, reached from a trace through a callback.

    Parameters
    ----------
    family :
        The section family reduced to its host coefficients, static.
    axial :
        Axial force of every member, tension positive.
    end_major :
        Major-axis moment at each end of every member.
    end_minor :
        Minor-axis moment at each end of every member.

    Returns
    -------
    sized :
        The floored diameter of every member, and the utilization at it; the
        reverse rule is the host's hand adjoint through a second callback.
    """

    def on_host(axial_host, major_host, minor_host):
        actions = host_actions(axial_host, major_host, minor_host)
        sized = size_members(actions, family)

        return sized.diameter, sized.utilization

    struct = _float_struct(jnp.shape(axial))

    return jax.pure_callback(
        on_host, (struct, struct), axial, end_major, end_minor, vmap_method="sequential"
    )


def _sized_forward(family, axial, end_major, end_minor):
    """
    Run the sizing map and keep the actions for the reverse pass.
    """
    sized = sized_members(family, axial, end_major, end_minor)

    return sized, (axial, end_major, end_minor)


def _sized_backward(family, residuals, cotangents):
    """
    Pull the two output cotangents back through the host's hand adjoint.
    """
    axial, end_major, end_minor = residuals
    pulled_diameter, pulled_used = cotangents

    def on_host(axial_host, major_host, minor_host, diameter_host, used_host):
        actions = host_actions(axial_host, major_host, minor_host)
        seeds = SizeCotangents(diameter_host, used_host)

        return tuple(size_cotangents(actions, family, seeds))

    structs = (
        _float_struct(jnp.shape(axial)),
        _float_struct(jnp.shape(end_major)),
        _float_struct(jnp.shape(end_minor)),
    )

    return jax.pure_callback(
        on_host,
        structs,
        axial,
        end_major,
        end_minor,
        pulled_diameter,
        pulled_used,
        vmap_method="sequential",
    )


sized_members.defvjp(_sized_forward, _sized_backward)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def checked_members(
    family: HostFamily,
    diameter: Float[Array, "*load_cases members"],
    axial: Float[Array, "*load_cases members"],
    end_major: Float[Array, "*load_cases members ends"],
    end_minor: Float[Array, "*load_cases members ends"],
) -> Float[Array, "*load_cases members"]:
    """
    The host check at a held size, reached from a trace through a callback.

    Parameters
    ----------
    family :
        The section family reduced to its host coefficients, static.
    diameter :
        Outer diameter every member is checked at.
    axial :
        Axial force of every member, tension positive.
    end_major :
        Major-axis moment at each end of every member.
    end_minor :
        Minor-axis moment at each end of every member.

    Returns
    -------
    utilization :
        Demand over resistance of every member at the held size.
    """

    def on_host(diameter_host, axial_host, major_host, minor_host):
        actions = host_actions(axial_host, major_host, minor_host)

        return check_members(diameter_host, actions, family)

    struct = _float_struct(jnp.shape(diameter))

    return jax.pure_callback(
        on_host,
        struct,
        diameter,
        axial,
        end_major,
        end_minor,
        vmap_method="sequential",
    )


def _checked_forward(family, diameter, axial, end_major, end_minor):
    """
    Run the held check and keep its inputs for the reverse pass.
    """
    used = checked_members(family, diameter, axial, end_major, end_minor)

    return used, (diameter, axial, end_major, end_minor)


def _checked_backward(family, residuals, cotangent):
    """
    Pull the held check's cotangent back through the host's hand adjoint.
    """
    diameter, axial, end_major, end_minor = residuals

    def on_host(diameter_host, axial_host, major_host, minor_host, pulled_host):
        actions = host_actions(axial_host, major_host, minor_host)
        pulled = check_cotangents(diameter_host, actions, family, pulled_host)

        return (pulled.diameter_held, *pulled.actions)

    structs = (
        _float_struct(jnp.shape(diameter)),
        _float_struct(jnp.shape(axial)),
        _float_struct(jnp.shape(end_major)),
        _float_struct(jnp.shape(end_minor)),
    )

    return jax.pure_callback(
        on_host,
        structs,
        diameter,
        axial,
        end_major,
        end_minor,
        cotangent,
        vmap_method="sequential",
    )


checked_members.defvjp(_checked_forward, _checked_backward)


class BlueprintSizer(AbstractMemberSizer):
    """
    Blueprints' cross-section check, in process, as a block of the pipeline.

    Attributes
    ----------
    structure :
        The structure whose members are sized. Read for nothing.
    family :
        The section family every member is drawn from, and its grade.
    ratio :
        The family's wall proportion, snapshotted for the host.
    f_y :
        The family's yield strength, snapshotted for the host.

    Notes
    -----
    Every value is Blueprints' own, behind `jax.pure_callback`, and every
    derivative is the host's hand adjoint behind another. The Tesseract wraps
    the same host functions, so the two agree bit for bit. The ratio and the
    strength are concrete floats, so no material sensitivity flows through
    this block, and the buckling length is accepted and ignored — a
    cross-section check reads no length.
    """

    structure: Structure
    family: TubeFamily
    ratio: float = eqx.field(static=True)
    f_y: float = eqx.field(static=True)

    def __init__(self, structure: Structure, family: TubeFamily) -> None:
        """
        Build a sizer over a section family stated as bare geometry.

        Parameters
        ----------
        structure :
            The structure whose members are sized. Read for nothing.
        family :
            The section family every member is drawn from.

        Raises
        ------
        ValueError
            If the family's ratio leaves no wall at all.
        """
        ratio, f_y = snapshot_family(family)

        self.structure = structure
        self.family = family
        self.ratio = ratio
        self.f_y = f_y

    @property
    def host(self) -> HostFamily:
        """
        The family as the host check reads it.
        """
        return host_family(self.ratio, self.f_y)

    def __call__(
        self,
        forces: MemberForces,
        buckling_length: Float[Array, "members"],
    ) -> MemberSizes:
        """
        Size every member for every load case, each on its own.

        Parameters
        ----------
        forces :
            What every member carries under every load case.
        buckling_length :
            Accepted and ignored: a cross-section check reads no length.

        Returns
        -------
        sizes :
            The section each load case demands, and how hard it is worked —
            one wherever the size was free to move, below one at the floor.
        """
        diameter, used = sized_members(
            self.host, forces.axial_force, forces.moment_major, forces.moment_minor
        )

        return MemberSizes(self.family(diameter), used)

    def compute_utilization(
        self,
        diameters: Float[Array, "members"],
        forces: MemberForces,
        buckling_length: Float[Array, "members"],
    ) -> Float[Array, "load_cases members"]:
        """
        Check sizes the caller owns against Blueprints' cross-section check.

        Parameters
        ----------
        diameters :
            Outer diameter every member was given.
        forces :
            What every member carries under every load case.
        buckling_length :
            Accepted and ignored: a cross-section check reads no length.

        Returns
        -------
        utilization :
            Demand over resistance of every member under every load case.
        """
        spread = jnp.broadcast_to(diameters, jnp.shape(forces.axial_force))

        used = checked_members(
            self.host,
            spread,
            forces.axial_force,
            forces.moment_major,
            forces.moment_minor,
        )

        return used
