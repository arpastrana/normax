# SPDX-License-Identifier: Apache-2.0
"""
Blueprints' Eurocode 3 cross-section check, the host half of the boundary.

Blueprints is a scalar Python library whose formula classes subclass `float`
and carry no derivatives. Everything here is plain NumPy on the host: the
check through the library's eq. (6.10) and (6.14), a bisection to the size
that is worked to exactly one, and the adjoint of both by hand — the implicit
function theorem at the root, the bare partials at a held size. No JAX
anywhere in this module: the sizing Tesseract's blueprint backend maps its
schema onto these functions, and that boundary is the only way a trace
reaches them.

The check is axial force with bending at cross-section level and nothing else:
Blueprints implements no §6.3.1 flexural buckling, so no buckling length is
read, and shear, torsion and their interaction are declined by a measured
decision recorded in the project notes. Blueprints is LGPL-2.1 and is imported
unmodified as a pip package.
"""

import hashlib
import math
from typing import NamedTuple

import numpy as np
from blueprints.codes.eurocode.en_1993_1_1_2005.chapter_6_ultimate_limit_state.formula_6_10 import (  # noqa: E501
    Form6Dot10NcRdClass1And2And3,
)
from blueprints.codes.eurocode.en_1993_1_1_2005.chapter_6_ultimate_limit_state.formula_6_14 import (  # noqa: E501
    Form6Dot14MCRdClass3,
)
from jaxtyping import Bool
from jaxtyping import Float
from jaxtyping import Int

from normax.sections import TubeCatalog

# Eurocode 3 §6.1, the recommended value.
GAMMA_M0 = 1.0

# The smallest tube the section catalog offers, the catalog minimum.
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


class SectionCoefficients(NamedTuple):
    """
    A section catalog as the check reads it: two coefficients and three constants.

    Attributes
    ----------
    area_coefficient :
        Area per squared diameter of the catalog's tubes.
    modulus_coefficient :
        Elastic modulus per cubed diameter of the catalog's tubes.
    f_y :
        Yield strength of the catalog's grade.
    gamma_m0 :
        Partial factor for cross-section resistance.
    diameter_min :
        Smallest diameter the catalog offers.
    """

    area_coefficient: float
    modulus_coefficient: float
    f_y: float
    gamma_m0: float
    diameter_min: float


def coerce_section_coefficients(
    ratio: float,
    f_y: float,
    gamma_m0: float = GAMMA_M0,
    diameter_min: float = DIAMETER_MINIMUM,
) -> SectionCoefficients:
    """
    Reduce a section catalog to the coefficients the scalar check reads.

    Parameters
    ----------
    ratio :
        Diameter over wall thickness of every tube in the catalog.
    f_y :
        Yield strength of the catalog's grade.
    gamma_m0 :
        Partial factor for cross-section resistance.
    diameter_min :
        Smallest diameter the catalog offers.

    Returns
    -------
    catalog :
        The catalog's geometry collapsed to two proportionality constants.

    Raises
    ------
    ValueError
        If the ratio leaves no wall, or the minimum is no diameter.

    Notes
    -----
    With the wall a fixed proportion of the diameter, `A = c_A d^2` and
    `W_el = c_W d^3`. The arithmetic mirrors `MemberSections` at a unit
    diameter; Blueprints' own meshed CHS profiles are deliberately not used.
    """
    if ratio <= 2.0:
        raise ValueError(f"a ratio of {ratio} leaves no wall: need d/t > 2")
    if diameter_min <= 0.0:
        raise ValueError(f"a minimum of {diameter_min} is no diameter: need > 0")

    wall = 1.0 / ratio
    bore = 1.0 - 2.0 * wall
    area_coefficient = math.pi * wall * (1.0 - wall)
    second_moment = (math.pi / 64.0) * (1.0 - bore**4)
    modulus_coefficient = 2.0 * second_moment

    return SectionCoefficients(
        area_coefficient,
        modulus_coefficient,
        float(f_y),
        float(gamma_m0),
        float(diameter_min),
    )


def snapshot_catalog(catalog: TubeCatalog) -> tuple[float, float]:
    """
    Snapshot a catalog's ratio and yield strength for the host.

    Parameters
    ----------
    catalog :
        The section catalog a sizer is built over.

    Returns
    -------
    ratio :
        The catalog's wall proportion, as a concrete float.
    f_y :
        The catalog's yield strength, as a concrete float.

    Raises
    ------
    ValueError
        If the catalog's ratio leaves no wall at all.

    Notes
    -----
    The two numbers a host check reads off a catalog, concretized once at
    construction so no material sensitivity flows through a sizer — the
    in-process one and the crossed one snapshot identically.
    """
    ratio = float(catalog.ratio)
    f_y = float(catalog.material.f_y)
    coerce_section_coefficients(ratio, f_y)

    return ratio, f_y


def _check_scalar(
    diameter: float,
    axial: float,
    moment: float,
    catalog: SectionCoefficients,
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
    catalog :
        The section catalog reduced to its coefficients.

    Returns
    -------
    utilization :
        Demand over resistance, the linear sum of Eurocode 3 eq. (6.2), with
        eq. (6.10) and eq. (6.14) supplying the resistances.

    Raises
    ------
    ValueError
        If the moment arrives signed rather than reduced.
    """
    if moment < 0.0:
        raise ValueError(f"a moment of {moment} is signed: reduce before checking")

    area = catalog.area_coefficient * diameter**2
    modulus = catalog.modulus_coefficient * diameter**3
    squashing = Form6Dot10NcRdClass1And2And3(
        a=area, f_y=catalog.f_y, gamma_m0=catalog.gamma_m0
    )
    bending = Form6Dot14MCRdClass3(
        w_el_min=modulus, f_y=catalog.f_y, gamma_m0=catalog.gamma_m0
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
    catalog: SectionCoefficients,
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
    catalog :
        The section catalog reduced to its coefficients.

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
        return _check_scalar(diameter, axial, moment, catalog)

    area = catalog.area_coefficient * diameter**2
    modulus = catalog.modulus_coefficient * diameter**3
    squashing = Form6Dot10NcRdClass1And2And3._evaluate(
        area, catalog.f_y, catalog.gamma_m0
    )
    bending = Form6Dot14MCRdClass3._evaluate(modulus, catalog.f_y, catalog.gamma_m0)

    return abs(axial) / squashing + moment / bending


def _demand_scales(catalog: SectionCoefficients) -> tuple[float, float]:
    """
    The factors turning a force and a moment into diameter-unit demands.
    """
    scale_axial = catalog.gamma_m0 / (catalog.area_coefficient * catalog.f_y)
    scale_moment = catalog.gamma_m0 / (catalog.modulus_coefficient * catalog.f_y)

    return scale_axial, scale_moment


def _solve_scalar(axial: float, moment: float, catalog: SectionCoefficients) -> float:
    """
    The diameter one member's check is exactly satisfied at.

    Parameters
    ----------
    axial :
        Axial force the member carries, negative in compression.
    moment :
        Demand moment the member carries, non-negative.
    catalog :
        The section catalog reduced to its coefficients.

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

    scale_axial, scale_moment = _demand_scales(catalog)
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
        used = _probe_scalar(math.exp(middle), axial, moment, catalog)
        if used > 1.0:
            low = middle
        else:
            high = middle

    return math.exp(high)


def _solve_batch(
    axial: Float[np.ndarray, "*load_cases members"],
    moment: Float[np.ndarray, "*load_cases members"],
    catalog: SectionCoefficients,
) -> Float[np.ndarray, "*load_cases members"]:
    """
    Every member's exactly-satisfied diameter, one host loop.
    """
    paired = zip(axial.ravel(), moment.ravel(), strict=True)
    solved = [_solve_scalar(force, bent, catalog) for force, bent in paired]

    return np.asarray(solved, dtype=np.float64).reshape(axial.shape)


def _check_batch(
    diameter: Float[np.ndarray, "*load_cases members"],
    axial: Float[np.ndarray, "*load_cases members"],
    moment: Float[np.ndarray, "*load_cases members"],
    catalog: SectionCoefficients,
) -> Float[np.ndarray, "*load_cases members"]:
    """
    Every member's utilization at a given diameter, one host loop.
    """
    tripled = zip(diameter.ravel(), axial.ravel(), moment.ravel(), strict=True)
    used = [_check_scalar(size, force, bent, catalog) for size, force, bent in tripled]

    return np.asarray(used, dtype=np.float64).reshape(diameter.shape)


class MemberActions(NamedTuple):
    """
    What every member carries under one or more load cases, as plain arrays.

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


def coerce_member_actions(
    axial: Float[np.ndarray, "*load_cases members"],
    end_major: Float[np.ndarray, "*load_cases members ends"],
    end_minor: Float[np.ndarray, "*load_cases members ends"],
) -> MemberActions:
    """
    Bring three arrays of any provenance to the host as contiguous float64.
    """
    return MemberActions(
        np.ascontiguousarray(axial, dtype=np.float64),
        np.ascontiguousarray(end_major, dtype=np.float64),
        np.ascontiguousarray(end_minor, dtype=np.float64),
    )


class WinningEnd(NamedTuple):
    """
    Which end carries the larger moment, and where that moment points.

    Attributes
    ----------
    winner :
        Index of the end whose moment vector is larger in magnitude.
    cosine_major :
        Major component of that end's moment over its magnitude.
    cosine_minor :
        Minor component of that end's moment over its magnitude.

    Notes
    -----
    The cosines are the derivative of the magnitude with respect to each
    component, so the adjoint routes a cotangent on the demand by them. Both
    are zero where a member carries no moment at all, the magnitude having no
    derivative at the origin and an unmoved member no demand to route.
    """

    winner: Int[np.ndarray, "*load_cases members"]
    cosine_major: Float[np.ndarray, "*load_cases members"]
    cosine_minor: Float[np.ndarray, "*load_cases members"]


def _read_worse_end(
    major: Float[np.ndarray, "*load_cases members ends"],
    minor: Float[np.ndarray, "*load_cases members ends"],
) -> tuple[Float[np.ndarray, "*load_cases members"], WinningEnd]:
    """
    The larger of the two end moment vectors, and where it points.
    """
    per_end = np.sqrt(major**2 + minor**2)
    winner = np.argmax(per_end, axis=-1)
    take = winner[..., None]

    moment = np.take_along_axis(per_end, take, axis=-1)[..., 0]
    won_major = np.take_along_axis(major, take, axis=-1)[..., 0]
    won_minor = np.take_along_axis(minor, take, axis=-1)[..., 0]

    carried = moment > 0.0
    divisor = np.where(carried, moment, 1.0)
    cosine_major = np.where(carried, won_major / divisor, 0.0)
    cosine_minor = np.where(carried, won_minor / divisor, 0.0)

    return moment, WinningEnd(winner, cosine_major, cosine_minor)


class DemandMoment(NamedTuple):
    """
    The one moment the check reads, and where it came from.

    Attributes
    ----------
    moment :
        Magnitude of the larger of the two end moment vectors.
    worse :
        The end that carried it, and where it points.
    """

    moment: Float[np.ndarray, "*load_cases members"]
    worse: WinningEnd


def reduce_moments(actions: MemberActions) -> DemandMoment:
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
    The check is read at the worse end, and the two components there are
    combined as the **magnitude of the moment vector** rather than summed.

    For a circular hollow section that is the standard's own combination, not
    a relaxation of it. Eurocode 3 6.2.9.2 eq. (6.42) limits the maximum
    longitudinal fibre stress; a moment vector on an axisymmetric section
    bends about the axis perpendicular to itself, so that maximum is set by
    its magnitude. Summing the two components instead adds a peak stress
    occurring at one point of the circumference to one occurring a quarter
    turn away, which no single fibre carries. The same conclusion reaches
    Classes 1 and 2 by 6.2.9.1(6) eq. (6.41), whose exponents are alpha =
    beta = 2 for a circular hollow section; 6.2.9(6) permits taking them as
    unity, which is the linear interaction this reduction used until
    2026-08-28.

    Which end wins is therefore decided by the magnitude, not per axis. The
    two components no longer win separately, so a demand can no longer be
    assembled from moments at two different ends.
    """
    moment, worse = _read_worse_end(actions.end_major, actions.end_minor)

    return DemandMoment(moment, worse)


class SolvedState(NamedTuple):
    """
    One forward solve, held so the value and the adjoint read the same root.

    Attributes
    ----------
    catalog :
        The section catalog reduced to its coefficients.
    axial :
        Axial force every member carries, negative in compression.
    demand :
        The reduced moment and its routing.
    unclamped :
        The root of each member's check, zero where a member is unloaded.
    diameter :
        The root with the catalog minimum applied.
    """

    catalog: SectionCoefficients
    axial: Float[np.ndarray, "*load_cases members"]
    demand: DemandMoment
    unclamped: Float[np.ndarray, "*load_cases members"]
    diameter: Float[np.ndarray, "*load_cases members"]


# Solved states by fingerprint, so a reverse pass never re-bisects.
_SOLVED: dict[bytes, SolvedState] = {}


def _solve_fingerprint(actions: MemberActions, catalog: SectionCoefficients) -> bytes:
    """
    A digest of everything the bisection reads, by content.
    """
    digest = hashlib.blake2b(digest_size=32)
    for value in actions:
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    digest.update(repr(tuple(catalog)).encode())

    return digest.digest()


def solve_state(actions: MemberActions, catalog: SectionCoefficients) -> SolvedState:
    """
    The solved state these actions describe, searched for only once.

    Parameters
    ----------
    actions :
        What every member carries.
    catalog :
        The section catalog reduced to its coefficients.

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
    fingerprint = _solve_fingerprint(actions, catalog)
    held = _SOLVED.get(fingerprint)
    if held is not None:
        return held

    demand = reduce_moments(actions)
    unclamped = _solve_batch(actions.axial, demand.moment, catalog)
    diameter = np.maximum(unclamped, catalog.diameter_min)
    state = SolvedState(catalog, actions.axial, demand, unclamped, diameter)
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
        Outer diameter of every member, floored at the catalog's minimum.
    utilization :
        Demand over resistance at that diameter — one where the check decided
        the size, below one where the minimum did.
    clamped :
        Whether the minimum decided each member's size.
    """

    diameter: Float[np.ndarray, "*load_cases members"]
    utilization: Float[np.ndarray, "*load_cases members"]
    clamped: Bool[np.ndarray, "*load_cases members"]


def size_members(actions: MemberActions, catalog: SectionCoefficients) -> SizedMembers:
    """
    Size every member to the check, entirely on the host.

    Parameters
    ----------
    actions :
        What every member carries.
    catalog :
        The section catalog reduced to its coefficients.

    Returns
    -------
    sized :
        The floored diameters, the utilization re-read at them, and the mask
        of members the minimum decided.
    """
    state = solve_state(actions, catalog)
    used = _check_batch(state.diameter, state.axial, state.demand.moment, catalog)
    clamped = state.unclamped < catalog.diameter_min

    return SizedMembers(state.diameter, used, clamped)


def check_members(
    diameter_held: Float[np.ndarray, "*load_cases members"],
    actions: MemberActions,
    catalog: SectionCoefficients,
) -> Float[np.ndarray, "*load_cases members"]:
    """
    Check sizes the caller owns, entirely on the host.

    Parameters
    ----------
    diameter_held :
        Outer diameter every member is checked at.
    actions :
        What every member carries.
    catalog :
        The section catalog reduced to its coefficients.

    Returns
    -------
    utilization_held :
        Demand over resistance of every member at the held size.
    """
    held = np.asarray(diameter_held, dtype=np.float64)
    demand = reduce_moments(actions)

    return _check_batch(held, actions.axial, demand.moment, catalog)


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
    catalog: SectionCoefficients,
) -> CheckPartials:
    """
    The three partials of `U = a/d^2 + b/d^3`, evaluated at a given size.
    """
    scale_axial, scale_moment = _demand_scales(catalog)
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
    winner: Int[np.ndarray, "*load_cases members"],
    cosine: Float[np.ndarray, "*load_cases members"],
) -> Float[np.ndarray, "*load_cases members ends"]:
    """
    Place one component's share of a cotangent at the winning end.
    """
    ends = np.zeros((*pull.shape, 2))
    np.put_along_axis(ends, winner[..., None], (pull * cosine)[..., None], axis=-1)

    return ends


def _action_cotangents(
    pull_axial: Float[np.ndarray, "*load_cases members"],
    pull_moment: Float[np.ndarray, "*load_cases members"],
    demand: DemandMoment,
) -> ActionCotangents:
    """
    Route a cotangent on the demand moment back to the end that carried it.

    Notes
    -----
    Both components are routed to the one winning end, each by its own
    direction cosine, those being the derivatives of the magnitude the demand
    now is. A member carrying no moment routes zero to both.
    """
    worse = demand.worse
    end_major = _route_axis(pull_moment, worse.winner, worse.cosine_major)
    end_minor = _route_axis(pull_moment, worse.winner, worse.cosine_minor)

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
    actions: MemberActions,
    catalog: SectionCoefficients,
    cotangents: SizeCotangents,
) -> ActionCotangents:
    """
    Pull a cotangent on the sizing map's outputs back to the actions, by hand.

    Parameters
    ----------
    actions :
        What every member carries.
    catalog :
        The section catalog reduced to its coefficients.
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
    where the minimum did. The reported utilization is the mirror image: pinned
    at one where the check decided, the bare partial at the minimum otherwise.
    """
    state = solve_state(actions, catalog)
    demand = state.demand
    at_root = _check_partials(state.unclamped, state.axial, demand.moment, catalog)
    at_minimum = _check_partials(state.diameter, state.axial, demand.moment, catalog)

    free = state.unclamped >= catalog.diameter_min
    divisor = np.where(free, at_root.slope, -1.0)
    size_axial = np.where(free, -at_root.axial / divisor, 0.0)
    size_moment = np.where(free, -at_root.moment / divisor, 0.0)
    check_axial = np.where(free, 0.0, at_minimum.axial)
    check_moment = np.where(free, 0.0, at_minimum.moment)

    pulled_diameter = np.asarray(cotangents.diameter, dtype=np.float64)
    pulled_used = np.asarray(cotangents.utilization, dtype=np.float64)
    axial = size_axial * pulled_diameter + check_axial * pulled_used
    moment = size_moment * pulled_diameter + check_moment * pulled_used

    return _action_cotangents(axial, moment, demand)


class CheckCotangents(NamedTuple):
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
    actions: MemberActions,
    catalog: SectionCoefficients,
    cotangent: Float[np.ndarray, "*load_cases members"],
) -> CheckCotangents:
    """
    Pull a cotangent on the held check back to the size and the actions.

    Parameters
    ----------
    diameter_held :
        Outer diameter every member was checked at.
    actions :
        What every member carries.
    catalog :
        The section catalog reduced to its coefficients.
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
    demand = reduce_moments(actions)
    partials = _check_partials(held, actions.axial, demand.moment, catalog)
    on_actions = _action_cotangents(
        partials.axial * pulled, partials.moment * pulled, demand
    )

    return CheckCotangents(partials.slope * pulled, on_actions)
