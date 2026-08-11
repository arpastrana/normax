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
Closed-form derivatives of the sizing map under axial force alone.

An independent oracle, not the rule the map is differentiated with. The sizing
map obtains its partial derivatives by differentiating the check itself and
inverts only the implicit part; everything here is derived on paper instead and
written out in full. Where the two agree, neither a slip in the derivation nor a
mistake in the machinery can be hiding.

Axial force only, and so no moments and no interaction factors. With no moment
the member check of 6.3.3 collapses to the buckling check of 6.3.1 and governs
over the cross-section check for every adequate section, which is what leaves a
single expression to differentiate.

The pieces are the buckling reduction factor and the two properties that carry
the diameter. Every section property is a monomial, so the slenderness varies as
the buckling length over the diameter and nothing else has to be tracked.
"""

import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float

from normax.ec3.material import SteelGrade
from normax.ec3.resistance import SLENDERNESS_OFFSET
from normax.ec3.resistance import buckling_auxiliary
from normax.ec3.resistance import reduction_buckling
from normax.ec3.section import area
from normax.ec3.section import second_moment
from normax.ec3.sizing import Tube


def reduction_buckling_derivative(
    lam: Float[Array, "members"],
    alpha: float | Float[Array, ""],
) -> Float[Array, "members"]:
    """
    Slope of the buckling reduction factor in the slenderness.

    Parameters
    ----------
    lam :
        Non-dimensional slenderness.
    alpha :
        Imperfection factor of the buckling curve.

    Returns
    -------
    slope :
        Derivative of the reduction factor, never positive.

    Notes
    -----
    EN 1993-1-1 6.3.1.2, Eq. 6.49 differentiated. The factor is the reciprocal
    of the auxiliary term plus a square root, so its slope carries the square of
    the uncapped factor.

    Zero below the offset slenderness, where 6.3.1.2(3) caps the factor at one
    and the curve is flat. That flat stretch is a genuine kink in the standard,
    and it is where a member stops caring how long it is.
    """
    slender = jnp.asarray(lam)
    auxiliary = buckling_auxiliary(slender, alpha)
    root = jnp.sqrt(auxiliary**2 - slender**2)

    uncapped = 1.0 / (auxiliary + root)
    slope_auxiliary = 0.5 * (alpha + 2.0 * slender)
    slope = -(uncapped**2) * (
        slope_auxiliary + (auxiliary * slope_auxiliary - slender) / root
    )

    return jnp.where(slender > SLENDERNESS_OFFSET, slope, 0.0)


def slenderness_unit(
    steel: SteelGrade,
    tube: Tube,
) -> Float[Array, ""]:
    """
    Slenderness of a member one long and one across.

    Parameters
    ----------
    steel :
        Material properties and partial factors.
    tube :
        The section family the member is drawn from.

    Returns
    -------
    slenderness :
        Constant of proportionality between the slenderness and the ratio of
        buckling length to diameter.

    Notes
    -----
    The radius of gyration of a tube is a fixed fraction of its diameter, so the
    slenderness is the buckling length over the diameter times a constant fixed
    by the wall proportion and the grade. Collecting that constant once is what
    reduces every derivative below to a rational expression.
    """
    unit_area = area(1.0, tube.ratio)
    unit_inertia = second_moment(1.0, tube.ratio)

    return jnp.sqrt(unit_area * steel.f_y / (jnp.pi**2 * steel.e_mod * unit_inertia))


def utilization_slope(
    diameter: Float[Array, "members"],
    n_ed: Float[Array, "members"],
    l_cr: Float[Array, "members"],
    steel: SteelGrade,
    tube: Tube,
) -> Float[Array, "members"]:
    """
    Slope of the buckling check in the diameter.

    Parameters
    ----------
    diameter :
        Outer diameter.
    n_ed :
        Design axial force, tension positive.
    l_cr :
        Buckling length.
    steel :
        Material properties and partial factors.
    tube :
        The section family the member is drawn from.

    Returns
    -------
    slope :
        Derivative of the utilization, always negative.

    Notes
    -----
    The check is the axial force over a resistance that is the reduction factor
    times the area. The area grows with the square of the diameter and the
    reduction factor grows too, since slenderness falls, so the check falls on
    both counts. Those are the two terms.
    """
    lam = slenderness_unit(steel, tube) * l_cr / diameter
    reduction = reduction_buckling(lam, steel.alpha)
    demand = _buckling_check(diameter, n_ed, l_cr, steel, tube)

    return (demand / diameter) * (
        lam * reduction_buckling_derivative(lam, steel.alpha) / reduction - 2.0
    )


def _buckling_check(
    diameter: Float[Array, "members"],
    n_ed: Float[Array, "members"],
    l_cr: Float[Array, "members"],
    steel: SteelGrade,
    tube: Tube,
) -> Float[Array, "members"]:
    """
    The buckling check written out, without going through the clause modules.

    Returns
    -------
    utilization :
        Axial force over buckling resistance.
    """
    lam = slenderness_unit(steel, tube) * l_cr / diameter
    reduction = reduction_buckling(lam, steel.alpha)
    resistance = reduction * area(diameter, tube.ratio) * steel.f_y / steel.gamma_m1

    return jnp.abs(n_ed) / resistance


def derivative_force(
    diameter: Float[Array, "members"],
    n_ed: Float[Array, "members"],
    l_cr: Float[Array, "members"],
    steel: SteelGrade,
    tube: Tube,
) -> Float[Array, "members"]:
    """
    Sensitivity of the fully-stressed diameter to the axial force.

    Parameters
    ----------
    diameter :
        Outer diameter at which the check is satisfied.
    n_ed :
        Design axial compression, negative.
    l_cr :
        Buckling length.
    steel :
        Material properties and partial factors.
    tube :
        The section family the member is drawn from.

    Returns
    -------
    sensitivity :
        Change in diameter per unit change in axial force, negative because a
        more compressed member is a larger one.

    Notes
    -----
    The implicit function theorem on the check, which stays at one as the force
    moves. The check is proportional to the force, so its derivative there is
    the check over the force, and dividing by the slope in the diameter gives
    the result.
    """
    demand = _buckling_check(diameter, n_ed, l_cr, steel, tube)
    slope_force = -demand / jnp.abs(n_ed)

    return -slope_force / utilization_slope(diameter, n_ed, l_cr, steel, tube)


def derivative_length(
    diameter: Float[Array, "members"],
    n_ed: Float[Array, "members"],
    l_cr: Float[Array, "members"],
    steel: SteelGrade,
    tube: Tube,
) -> Float[Array, "members"]:
    """
    Sensitivity of the fully-stressed diameter to the buckling length.

    Parameters
    ----------
    diameter :
        Outer diameter at which the check is satisfied.
    n_ed :
        Design axial compression, negative.
    l_cr :
        Buckling length.
    steel :
        Material properties and partial factors.
    tube :
        The section family the member is drawn from.

    Returns
    -------
    sensitivity :
        Change in diameter per unit change in buckling length, positive because
        a longer member is a larger one.

    Notes
    -----
    Length reaches the check only through the slenderness, which is
    proportional to it. Below the offset slenderness the reduction factor is
    capped, its slope is zero and so is this: a stocky member does not care how
    long it is.
    """
    lam = slenderness_unit(steel, tube) * l_cr / diameter
    reduction = reduction_buckling(lam, steel.alpha)
    demand = _buckling_check(diameter, n_ed, l_cr, steel, tube)

    slope_length = (
        -demand
        * reduction_buckling_derivative(lam, steel.alpha)
        * lam
        / (reduction * l_cr)
    )

    return -slope_length / utilization_slope(diameter, n_ed, l_cr, steel, tube)


def diameter_tension(
    n_ed: Float[Array, "members"],
    steel: SteelGrade,
    tube: Tube,
) -> Float[Array, "members"]:
    """
    Fully-stressed diameter of a member in tension, in closed form.

    Parameters
    ----------
    n_ed :
        Design axial tension, positive.
    steel :
        Material properties and partial factors.
    tube :
        The section family the member is drawn from.

    Returns
    -------
    diameter :
        Diameter at which the gross section yields exactly.

    Notes
    -----
    EN 1993-1-1 6.2.3, Eq. 6.6 inverted. There is no buckling in tension and no
    length in the answer, and the area is quadratic in the diameter, so the
    root needs no search at all. The catalogue minimum is not applied here.
    """
    unit_area = area(1.0, tube.ratio)

    return jnp.sqrt(jnp.abs(n_ed) * steel.gamma_m0 / (steel.f_y * unit_area))


def derivative_force_tension(
    n_ed: Float[Array, "members"],
    steel: SteelGrade,
    tube: Tube,
) -> Float[Array, "members"]:
    """
    Sensitivity of a tension member's diameter to the axial force.

    Parameters
    ----------
    n_ed :
        Design axial tension, positive.
    steel :
        Material properties and partial factors.
    tube :
        The section family the member is drawn from.

    Returns
    -------
    sensitivity :
        Change in diameter per unit change in axial force.

    Notes
    -----
    The closed form differentiated directly: the diameter goes as the square
    root of the force, so this is half the diameter over the force.
    """
    return diameter_tension(n_ed, steel, tube) / (2.0 * jnp.abs(n_ed))
