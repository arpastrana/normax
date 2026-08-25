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
Blueprints' EN 1993-1-1 cross-section check, as a block of the design pipeline.

Blueprints is a scalar Python library: its formula classes subclass `float`,
carry no derivatives, and cannot be traced. This module hosts them behind
`jax.pure_callback` and carries the derivative itself — a hand-derived rule,
with the implicit function theorem supplying the tangent of the root the host
bisection finds. The library computes every reported value; only the
derivative coefficients are restated here.

The check is **axial force with bending, at cross-section level, and nothing
else**. Two of those boundaries are the library's and one is this module's, and
they are worth keeping apart. Blueprints implements no §6.3.1 flexural buckling
and no cross-section classification, so neither could be had from it. It does
carry §6.3.2.1, the lateral-torsional check, which a doubly symmetric tube never
needs — so the one member clause it offers is the one this repo has no use for. It does
implement shear (§6.2.6), torsion (§6.2.7) and bending with shear (§6.2.8) —
including Eq. 6.18's `A_v` for a circular hollow section and Eq. 6.28's
`V_pl,T,Rd` for a hollow one — and **this module declines all three by choice**.

The choice is measured rather than assumed. EN 1993-1-1 6.2.10 allows shear out
of the bending and axial checks while the design shear stays under half the
plastic shear resistance; `experiments/20_shear_audit.py` reads that fraction off
every converged design in the repo and finds 0.36 at worst, on the drawn
Vierendeel, and 0.16 at worst on any optimized answer. Torsion is identically
zero, a planar frame under in-plane nodal load carrying none. Adding either
clause would move no diameter here. See `docs/shear_design.md` for what adding
them would take, and for the case that would need it.

An honestly different design philosophy from `normax.sizing.ec3`, not a
reimplementation: this sizer never reads a buckling length, and a compressed arch
is sized differently by the two.

Blueprints is LGPL-2.1, experiment-only, waived 2026-08-15: never on the
Apache-2.0 submission path.
"""

import math
from functools import partial
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

from normax.analysis import MemberForces
from normax.sections import TubeFamily
from normax.sizing import AbstractMemberSizer
from normax.sizing import MemberSizes
from normax.structures import Structure

# EN 1993-1-1 §6.1, the recommended value — this module's own statement.
GAMMA_M0 = 1.0

# Restates ec3x's catalogue floor; the cross-repo agreement is a test literal.
DIAMETER_MINIMUM = 21.3

# The bracket ratio is at most sqrt(2) + cbrt(2), so this is far below one ulp.
BISECTION_HALVINGS = 55


class HostFamily(NamedTuple):
    """
    A section family as the host solver needs it: three concrete numbers.

    Attributes
    ----------
    area_coefficient :
        Area per squared diameter of the family's tubes.
    modulus_coefficient :
        Elastic modulus per cubed diameter of the family's tubes.
    f_y :
        Yield strength of the family's grade.
    """

    area_coefficient: float
    modulus_coefficient: float
    f_y: float


def host_family(ratio: float, f_y: float) -> HostFamily:
    """
    Reduce a section family to the coefficients the scalar check reads.

    Parameters
    ----------
    ratio :
        Diameter over wall thickness of every tube in the family.
    f_y :
        Yield strength of the family's grade.

    Returns
    -------
    family :
        The family's geometry collapsed to two proportionality constants.

    Notes
    -----
    With the wall a fixed proportion of the diameter, `A = c_A d^2` and
    `W_el = c_W d^3`, so the whole family is two constants and a strength.
    The arithmetic mirrors `MemberSections` at a unit diameter, expression by
    expression, and a test pins the agreement. Blueprints' own CHS profiles
    are meshed polygons, approximate and slow, and are deliberately not used.
    """
    wall = 1.0 / ratio
    bore = 1.0 - 2.0 * wall
    area_coefficient = math.pi * wall * (1.0 - wall)
    second_moment = (math.pi / 64.0) * (1.0 - bore**4)
    modulus_coefficient = 2.0 * second_moment

    return HostFamily(area_coefficient, modulus_coefficient, f_y)


def _resistance_pair(diameter: float, family: HostFamily) -> tuple[float, float]:
    """
    Both cross-section resistances at one diameter, computed by Blueprints.

    Parameters
    ----------
    diameter :
        Outer diameter of the trial tube.
    family :
        The section family reduced to its host coefficients.

    Returns
    -------
    resistances :
        Design axial resistance and design bending resistance.

    Notes
    -----
    EN 1993-1-1 eq. (6.10) for `N_c,Rd` and eq. (6.14) for the class-3
    `M_c,Rd`, each constructed as a Blueprints formula object. Equation (6.6)
    is the same expression as (6.10) for classes 1 to 3, so tension needs no
    second formula.
    """
    area = family.area_coefficient * diameter**2
    modulus = family.modulus_coefficient * diameter**3
    squashing = Form6Dot10NcRdClass1And2And3(a=area, f_y=family.f_y, gamma_m0=GAMMA_M0)
    bending = Form6Dot14MCRdClass3(w_el_min=modulus, f_y=family.f_y, gamma_m0=GAMMA_M0)

    return float(squashing), float(bending)


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

    squashing, bending = _resistance_pair(diameter, family)

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
    With `a` and `b` the axial and bending demands in diameter units,
    `U(d) = a/d^2 + b/d^3` is strictly decreasing, so the root is unique and
    bisection is unconditionally safe. The bracket is exact: `max(sqrt(a),
    cbrt(b))` puts one term at one, and `sqrt(2a) + cbrt(2b)` puts each term
    at or under one half. Every midpoint is evaluated through Blueprints; the
    bracket alone is arithmetic. The returned end is the satisfied side, so
    the utilization re-read there is one minus a rounding error.
    """
    if moment < 0.0:
        raise ValueError(f"a moment of {moment} is signed: reduce before solving")

    demand_axial = abs(axial) * GAMMA_M0 / (family.area_coefficient * family.f_y)
    demand_moment = moment * GAMMA_M0 / (family.modulus_coefficient * family.f_y)
    if demand_axial == 0.0 and demand_moment == 0.0:
        return 0.0

    lower = max(math.sqrt(demand_axial), math.cbrt(demand_moment))
    upper = math.sqrt(2.0 * demand_axial) + math.cbrt(2.0 * demand_moment)
    low = math.log(lower)
    high = math.log(upper)
    for _ in range(BISECTION_HALVINGS):
        middle = 0.5 * (low + high)
        used = _check_scalar(math.exp(middle), axial, moment, family)
        if used > 1.0:
            low = middle
        else:
            high = middle

    return math.exp(high)


def _solve_batch(
    axial: Float[np.ndarray, "elements"],
    moment: Float[np.ndarray, "elements"],
    family: HostFamily,
) -> Float[np.ndarray, "elements"]:
    """
    Every member's exactly-satisfied diameter, one host loop over flat arrays.

    Parameters
    ----------
    axial :
        Axial force every member carries, negative in compression.
    moment :
        Demand moment every member carries, non-negative.
    family :
        The section family reduced to its host coefficients.

    Returns
    -------
    diameters :
        The root of each member's check, zero where a member is unloaded.
    """
    paired = zip(axial, moment, strict=True)
    solved = [_solve_scalar(force, bent, family) for force, bent in paired]

    return np.asarray(solved, dtype=np.float64)


def _check_batch(
    diameter: Float[np.ndarray, "elements"],
    axial: Float[np.ndarray, "elements"],
    moment: Float[np.ndarray, "elements"],
    family: HostFamily,
) -> Float[np.ndarray, "elements"]:
    """
    Every member's utilization at a given diameter, one host loop over flat arrays.

    Parameters
    ----------
    diameter :
        Outer diameter every member is checked at.
    axial :
        Axial force every member carries, negative in compression.
    moment :
        Demand moment every member carries, non-negative.
    family :
        The section family reduced to its host coefficients.

    Returns
    -------
    utilization :
        Demand over resistance of every member.
    """
    tripled = zip(diameter, axial, moment, strict=True)
    used = [_check_scalar(size, force, bent, family) for size, force, bent in tripled]

    return np.asarray(used, dtype=np.float64)


def _callback_solve(
    ratio: float,
    f_y: float,
    axial_force: Float[Array, "*load_cases members"],
    moment: Float[Array, "*load_cases members"],
) -> Float[Array, "*load_cases members"]:
    """
    Cross to the host, solve every member there, and come back an array.
    """
    family = host_family(ratio, f_y)
    shape = jnp.shape(axial_force)

    def solve_on_host(axial_host, moment_host):
        flat_axial = np.asarray(axial_host, dtype=np.float64).ravel()
        flat_moment = np.asarray(moment_host, dtype=np.float64).ravel()
        solved = _solve_batch(flat_axial, flat_moment, family)

        return solved.reshape(shape)

    struct = jax.ShapeDtypeStruct(shape, jnp.float64)

    return jax.pure_callback(
        solve_on_host,
        struct,
        axial_force,
        moment,
        vmap_method="sequential",
    )


def _callback_check(
    ratio: float,
    f_y: float,
    diameter: Float[Array, "*load_cases members"],
    axial_force: Float[Array, "*load_cases members"],
    moment: Float[Array, "*load_cases members"],
) -> Float[Array, "*load_cases members"]:
    """
    Cross to the host, check every member there, and come back an array.
    """
    family = host_family(ratio, f_y)
    shape = jnp.shape(diameter)

    def check_on_host(diameter_host, axial_host, moment_host):
        flat_diameter = np.asarray(diameter_host, dtype=np.float64).ravel()
        flat_axial = np.asarray(axial_host, dtype=np.float64).ravel()
        flat_moment = np.asarray(moment_host, dtype=np.float64).ravel()
        used = _check_batch(flat_diameter, flat_axial, flat_moment, family)

        return used.reshape(shape)

    struct = jax.ShapeDtypeStruct(shape, jnp.float64)

    return jax.pure_callback(
        check_on_host,
        struct,
        diameter,
        axial_force,
        moment,
        vmap_method="sequential",
    )


@partial(jax.custom_jvp, nondiff_argnums=(0, 1))
def sized_diameter(
    ratio: float,
    f_y: float,
    axial_force: Float[Array, "*load_cases members"],
    moment: Float[Array, "*load_cases members"],
) -> Float[Array, "*load_cases members"]:
    """
    The diameter every member's cross-section check is exactly satisfied at.

    Parameters
    ----------
    ratio :
        Diameter over wall thickness, a concrete number rather than a tracer.
    f_y :
        Yield strength, a concrete number rather than a tracer.
    axial_force :
        Axial force every member carries, negative in compression.
    moment :
        Demand moment every member carries, non-negative.

    Returns
    -------
    diameters :
        The root of each member's check, zero where a member is unloaded, and
        unclamped — the caller applies the catalogue floor.

    Notes
    -----
    The value comes from a host bisection over Blueprints residuals; the
    derivative comes from the implicit function theorem at that root, with
    the residual's partials hand-derived. The family's ratio and strength
    cross as static numbers because the host must hold concrete values to
    construct formula objects — so, unlike the EC3 sizer, no material or
    ratio sensitivity flows through this map.
    """
    return _callback_solve(ratio, f_y, axial_force, moment)


class CheckPartials(NamedTuple):
    """
    The check's hand-derived partials at one size, nan-safe where there is none.

    Attributes
    ----------
    sized :
        Whether each member has a positive size for the partials to hold at.
    slope :
        Derivative of the check in the diameter, strictly negative when sized.
    partial_axial :
        Derivative of the check in the axial force.
    partial_moment :
        Derivative of the check in the demand moment.
    """

    sized: Bool[Array, "*load_cases members"]
    slope: Float[Array, "*load_cases members"]
    partial_axial: Float[Array, "*load_cases members"]
    partial_moment: Float[Array, "*load_cases members"]


def _traced_partials(
    ratio: float,
    f_y: float,
    size: Float[Array, "*load_cases members"],
    axial_force: Float[Array, "*load_cases members"],
    moment: Float[Array, "*load_cases members"],
) -> CheckPartials:
    """
    Every partial both custom rules read, stated once.

    Parameters
    ----------
    ratio :
        Diameter over wall thickness, a concrete number rather than a tracer.
    f_y :
        Yield strength, a concrete number rather than a tracer.
    size :
        Diameter the partials are evaluated at — a root or a caller's own.
    axial_force :
        Axial force every member carries, negative in compression.
    moment :
        Demand moment every member carries, non-negative.

    Returns
    -------
    partials :
        The three closed-form derivatives of `U = a/d^2 + b/d^3`, and the
        mask that makes them safe at a zero size.
    """
    family = host_family(ratio, f_y)
    demand_axial = jnp.abs(axial_force) * GAMMA_M0 / (family.area_coefficient * f_y)
    demand_moment = moment * GAMMA_M0 / (family.modulus_coefficient * f_y)
    sized = size > 0.0
    safe = jnp.where(sized, size, 1.0)
    slope = -(2.0 * demand_axial / safe**3 + 3.0 * demand_moment / safe**4)
    partial_axial = (
        jnp.sign(axial_force) * GAMMA_M0 / (family.area_coefficient * f_y * safe**2)
    )
    partial_moment = GAMMA_M0 / (family.modulus_coefficient * f_y * safe**3)

    return CheckPartials(sized, slope, partial_axial, partial_moment)


@sized_diameter.defjvp
def _sized_jvp(ratio, f_y, primals, tangents):
    """
    The implicit tangent of the sizing map, at the root the host found.

    Notes
    -----
    By the implicit function theorem at `U(D; N, M) = 1`, the tangent is
    `-(dU/dN dN + dU/dM dM) / (dU/dd)`, every partial evaluated in closed
    form at the root. Unloaded members have no root and get a zero tangent.
    """
    axial_force, moment = primals
    axial_dot, moment_dot = tangents
    solved = _callback_solve(ratio, f_y, axial_force, moment)
    partials = _traced_partials(ratio, f_y, solved, axial_force, moment)

    rooted = partials.sized
    divisor = jnp.where(rooted, partials.slope, -1.0)
    quotient_axial = jnp.where(rooted, -partials.partial_axial / divisor, 0.0)
    quotient_moment = jnp.where(rooted, -partials.partial_moment / divisor, 0.0)
    tangent = quotient_axial * axial_dot + quotient_moment * moment_dot

    return solved, tangent


@partial(jax.custom_jvp, nondiff_argnums=(0, 1))
def checked_utilization(
    ratio: float,
    f_y: float,
    diameter: Float[Array, "*load_cases members"],
    axial_force: Float[Array, "*load_cases members"],
    moment: Float[Array, "*load_cases members"],
) -> Float[Array, "*load_cases members"]:
    """
    Every member's utilization at a given diameter, computed by Blueprints.

    Parameters
    ----------
    ratio :
        Diameter over wall thickness, a concrete number rather than a tracer.
    f_y :
        Yield strength, a concrete number rather than a tracer.
    diameter :
        Outer diameter every member is checked at, strictly positive.
    axial_force :
        Axial force every member carries, negative in compression.
    moment :
        Demand moment every member carries, non-negative.

    Returns
    -------
    utilization :
        Demand over resistance under EN 1993-1-1 eq. (6.2), with (6.10) and
        (6.14) supplying the resistances.

    Notes
    -----
    The value is Blueprints' own; the derivative is the check's hand-derived
    partials in the diameter, the force and the moment. This is the explicit
    map a simultaneous optimization constrains — no root find, no implicit
    rule, just a differentiable reading of the code.
    """
    return _callback_check(ratio, f_y, diameter, axial_force, moment)


@checked_utilization.defjvp
def _checked_jvp(ratio, f_y, primals, tangents):
    """
    The hand-derived tangent of the explicit check.
    """
    diameter, axial_force, moment = primals
    diameter_dot, axial_dot, moment_dot = tangents
    used = _callback_check(ratio, f_y, diameter, axial_force, moment)
    partials = _traced_partials(ratio, f_y, diameter, axial_force, moment)

    stretched = partials.slope * diameter_dot
    forced = partials.partial_axial * axial_dot
    bent = partials.partial_moment * moment_dot
    tangent = jnp.where(partials.sized, stretched + forced + bent, 0.0)

    return used, tangent


def demand_moment(forces: MemberForces) -> Float[Array, "*load_cases members"]:
    """
    Reduce two end moments per axis to the one moment this check reads.

    Parameters
    ----------
    forces :
        What every member carries, with both end moments on both axes.

    Returns
    -------
    moment :
        The larger end moment in magnitude on each axis, summed over axes.

    Notes
    -----
    A modeling choice rather than a clause: the check is a cross-section
    check, read at the worse end, and the two axes superpose linearly per
    EN 1993-1-1 eq. (6.2) — conservative for a circular section relative to
    the vector resultant, and stated as this sizer's own reduction.
    """
    major = jnp.max(jnp.abs(forces.moment_major), axis=-1)
    minor = jnp.max(jnp.abs(forces.moment_minor), axis=-1)
    combined = major + minor

    return combined


class BlueprintSizer(AbstractMemberSizer):
    """
    Blueprints' cross-section check, as a block of the design pipeline.

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
    **Cross-section resistance is the whole philosophy.** Blueprints
    implements no member buckling and no classification, so this sizer
    checks EN 1993-1-1 eq. (6.2) with the (6.10) and (6.14) resistances and
    nothing else. The buckling length is accepted and ignored — that is this
    philosophy's statement, not an oversight — and a compressed arch is
    sized thinner here than by the EC3 sizer, which sees buckling.

    **The library runs on the host; the derivative is carried by hand.**
    Every value is computed by Blueprints behind `jax.pure_callback`; the
    sizing map's tangent comes from the implicit function theorem, and the
    check's from its explicit partials. The family's ratio and strength are
    snapshotted to concrete floats at construction, so no material or ratio
    sensitivity flows — unlike the EC3 sizer.

    **One block, two usage modes.** Called, it is a nested fully-stressed
    sizer like any other. Its `compute_utilization` is the same differentiable
    check read at a size the caller owns — the constraint function a
    simultaneous optimization over diameters imposes as `U <= 1`.

    Blueprints is LGPL-2.1, experiment-only, waived 2026-08-15.
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
            The section family every member is drawn from, whose ratio fixes
            the wall proportion and whose grade supplies the material.

        Raises
        ------
        ValueError
            If the family's ratio leaves no wall at all.
        """
        ratio = float(family.ratio)
        if ratio <= 2.0:
            raise ValueError(f"a ratio of {ratio} leaves no wall: need d/t > 2")

        self.structure = structure
        self.family = family
        self.ratio = ratio
        self.f_y = float(family.material.f_y)

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
            one wherever the size was free to move, below one where the
            catalogue minimum bound.
        """
        moment = demand_moment(forces)
        solved = sized_diameter(self.ratio, self.f_y, forces.axial_force, moment)
        clamped = jnp.maximum(solved, DIAMETER_MINIMUM)
        used = checked_utilization(
            self.ratio, self.f_y, clamped, forces.axial_force, moment
        )
        sections = self.family(clamped)

        return MemberSizes(sections, used)

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
            Demand over resistance of every member under every load case —
            the differentiable constraint a simultaneous optimization holds
            at or under one.
        """
        moment = demand_moment(forces)
        spread = jnp.broadcast_to(diameters, jnp.shape(forces.axial_force))
        used = checked_utilization(
            self.ratio, self.f_y, spread, forces.axial_force, moment
        )

        return used
