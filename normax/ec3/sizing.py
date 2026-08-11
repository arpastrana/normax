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
Implicit fully-stressed sizing map, from member actions to a diameter.

EN 1993-1-1 states resistances and leaves the designer to search for a section
that carries the actions. This module performs that search and, unlike the
standard, carries a derivative through it.

The map returns the diameter at which the utilization is exactly one. The
utilization is the larger of two checks that the standard requires
independently: the member check of 6.3.3, which only applies in compression,
and the cross-section check of 6.2.9, which applies always. Both are strictly
decreasing in the diameter, so their larger is too, the root is unique and
bisection is unconditionally safe.

Nothing here is smoothed. The caps and the switches inside the check are
genuine features of the normative text, and the diameter returned satisfies the
standard exactly rather than a relaxation of it. Differentiability comes from
the implicit function theorem applied at the root, which needs the check to be
differentiable only there, not everywhere.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import logsumexp
from jaxtyping import Array
from jaxtyping import Float

from normax.ec3.actions import MemberActions
from normax.ec3.interaction import governing_equation
from normax.ec3.interaction import moment_factor_linear
from normax.ec3.interaction import utilization_member
from normax.ec3.material import SteelGrade
from normax.ec3.resistance import force_critical
from normax.ec3.resistance import moment_combined
from normax.ec3.resistance import reduction_buckling
from normax.ec3.resistance import slenderness_from_force
from normax.ec3.resistance import utilization_cross_section
from normax.ec3.section import Tube
from normax.ec3.section import TubeCatalogue

# Classes taking plastic section properties, EN 1993-1-1 Table 6.7.
PLASTIC_CLASSES = (1, 2)

# Width of the search interval, in doublings of the analytic lower bound. The
# bound is tight, so this is far more headroom than any member needs.
BRACKET_DOUBLINGS = 40

# Halvings of that interval. At this count the interval is narrower than the
# spacing of the numbers representing it, and the count is a compile-time
# constant, which keeps the forward pass jittable.
BISECTION_HALVINGS = 55

# Floor on the search interval, so that its logarithm stays finite when a member
# carries no action at all. Smaller than any tube, and not a design parameter.
DIAMETER_EPSILON = 1e-3

# Which limit state decided a member's size, reported alongside it as a
# non-differentiable diagnostic.
LIMIT_MINIMUM_SIZE = 0.0
LIMIT_TENSION = 1.0
LIMIT_CROSS_SECTION = 2.0
LIMIT_MAJOR = 3.0
LIMIT_MINOR = 4.0


def is_plastic(cross_section_class: int) -> bool:
    """
    Whether a cross-section class takes plastic section properties.

    Parameters
    ----------
    cross_section_class :
        Class 1, 2, 3 or 4.

    Returns
    -------
    plastic :
        True for Classes 1 and 2.

    Notes
    -----
    EN 1993-1-1 6.3.3, the table of characteristic resistances by class. This
    selects the section modulus, the cross-section clause and the column of
    Table B.1, so it is a build-time choice and never a traced value.
    """
    return cross_section_class in PLASTIC_CLASSES


def _modulus(tube: Tube, *, plastic: bool) -> Float[Array, "members"]:
    """
    Section modulus matching the cross-section class.

    Returns
    -------
    modulus :
        Plastic modulus for Classes 1 and 2, elastic modulus for Class 3.
    """
    if plastic:
        return tube.modulus_plastic

    return tube.modulus_elastic


def utilization_design(
    tube: Tube,
    actions: MemberActions,
    l_cr: Float[Array, "members"],
    steel: SteelGrade,
    *,
    plastic: bool,
    resultant: bool = True,
) -> Float[Array, "members"]:
    """
    Demand over resistance of a given tube.

    Parameters
    ----------
    tube :
        The section carrying the actions.
    actions :
        Design actions on the member.
    l_cr :
        Buckling length.
    steel :
        Material properties and partial factors.
    plastic :
        Whether the section is Class 1 or 2. Static, never a traced value.

    Returns
    -------
    utilization :
        The larger of the member and the cross-section checks. At most one if
        the member is adequate.

    Notes
    -----
    EN 1993-1-1 6.3.3 and 6.2.9, whichever governs. The two are independent
    requirements and neither bounds the other: with an equivalent uniform moment
    factor below one, the member check permits a moment the cross-section check
    refuses.

    The member check is switched off in tension, since 6.3.3 covers bending and
    axial compression only. That switch is a genuine discontinuity in the
    standard rather than an artefact here, and it is the reason a member whose
    axial force changes sign between load cases needs watching.

    Only magnitudes reach the clauses. The member check reads a compression, and
    both checks are indifferent to the sign of either moment.
    """
    member, section = _demands(
        tube, actions, l_cr, steel, plastic=plastic, resultant=resultant
    )

    return jnp.maximum(member, section)


def _demands(
    tube: Tube,
    actions: MemberActions,
    l_cr: Float[Array, "members"],
    steel: SteelGrade,
    *,
    plastic: bool,
    resultant: bool = True,
) -> tuple[Float[Array, "members"], Float[Array, "members"]]:
    """
    The two checks, before either is declared the winner.

    Returns
    -------
    demands :
        The member check of 6.3.3, already switched off in tension, and the
        cross-section check of 6.2.9.

    Notes
    -----
    Shared by the utilization and by the diagnostic that reports which check
    decided a member, so the two cannot disagree about what governed.
    """
    gross = tube.area
    modulus = _modulus(tube, plastic=plastic)

    critical = force_critical(tube.second_moment, l_cr, steel)
    lam = slenderness_from_force(gross, steel, critical)
    reduction = reduction_buckling(lam, steel.alpha)

    axial = jnp.asarray(actions.n_ed)
    member = utilization_member(
        jnp.maximum(-axial, 0.0),
        jnp.abs(actions.m_y_ed),
        jnp.abs(actions.m_z_ed),
        reduction,
        reduction,
        gross * steel.f_y,
        modulus * steel.f_y,
        lam,
        lam,
        actions.c_my,
        actions.c_mz,
        steel.gamma_m1,
        plastic=plastic,
    )

    section = utilization_cross_section(
        actions, gross, modulus, steel, plastic=plastic, resultant=resultant
    )

    return jnp.where(axial < 0.0, member, 0.0), section


def diameter_bracket(
    actions: MemberActions,
    steel: SteelGrade,
    catalogue: TubeCatalogue,
    *,
    plastic: bool,
    resultant: bool = True,
) -> Float[Array, "members"]:
    """
    Smallest diameter that could possibly satisfy the check.

    Parameters
    ----------
    actions :
        Design actions on the member. Neither moment factor is read, both
        bounds below being cross-section conditions.
    steel :
        Material properties and partial factors.
    catalogue :
        The section family the member is drawn from.
    plastic :
        Whether the section is Class 1 or 2. Static, never a traced value.

    Returns
    -------
    diameter :
        Lower bound on the fully-stressed diameter.

    Notes
    -----
    Two necessary conditions, each inverted in closed form. The gross section
    must carry the axial force alone, and the section modulus must carry the
    resultant moment alone; every section property is a monomial in the
    diameter, so both invert exactly. The unit-diameter properties come from the
    section module rather than being restated, so the constants cannot drift.

    The smallest tube the family offers is deliberately absent. That floor is a
    property of the catalogue rather than of the check, and folding it in here
    would let the search stop at a diameter where the check is not satisfied,
    which is where the implicit derivative stops being valid. It is applied
    afterwards instead.
    """
    unit = catalogue.tube(1.0)
    unit_area = unit.area
    unit_modulus = _modulus(unit, plastic=plastic)

    moment = moment_combined(
        actions.m_y_ed, actions.m_z_ed, plastic=plastic, resultant=resultant
    )

    squash = jnp.sqrt(jnp.abs(actions.n_ed) * steel.gamma_m0 / (steel.f_y * unit_area))
    bending = jnp.cbrt(moment * steel.gamma_m0 / (steel.f_y * unit_modulus))

    return jnp.maximum(jnp.maximum(squash, bending), DIAMETER_EPSILON)


def _solve(
    plastic: bool,
    resultant: bool,
    actions: MemberActions,
    l_cr: Float[Array, "members"],
    steel: SteelGrade,
    catalogue: TubeCatalogue,
) -> Float[Array, "members"]:
    """
    Bisect the check in log-diameter.

    Returns
    -------
    diameter :
        Diameter at which the utilization is one, or the family minimum where
        that is already sufficient.

    Notes
    -----
    Working in the logarithm makes the interval a fixed number of doublings
    rather than a fixed length, so the same iteration count resolves a tube of
    any size to the same relative precision.

    The over-sized end of the interval is returned rather than its midpoint. The
    two differ by less than the spacing of the numbers representing them, but
    only one of them is guaranteed to satisfy the check, and a design standard
    should not hand back a member that fails it.

    **The interval is checked at its top before its contents are believed.** The
    lower end brackets by construction, being the larger of two necessary
    conditions, but the upper end is assumed rather than searched for. Were the
    root above it, every midpoint would exceed one, the lower end would climb to
    meet the upper, and the untested top of the interval would come back looking
    like an answer. One evaluation there is the difference between a diameter that
    fails the check and a nan, and only the second is honest.

    Reaching that takes a buckling length of some 1e28 mm, since the two ends are
    twelve orders apart and only buckling can separate the root from the analytic
    bound. Neither tension nor bending can, both inverting exactly.
    """
    shape = jnp.broadcast_shapes(
        *(jnp.shape(action) for action in actions),
        jnp.shape(l_cr),
    )
    lower = diameter_bracket(
        actions, steel, catalogue, plastic=plastic, resultant=resultant
    )
    under = jnp.broadcast_to(jnp.log(lower), shape)
    over = under + BRACKET_DOUBLINGS * jnp.log(2.0)

    def halve(
        _: int,
        bounds: tuple[Float[Array, "members"], Float[Array, "members"]],
    ) -> tuple[Float[Array, "members"], Float[Array, "members"]]:
        small, large = bounds
        middle = 0.5 * (small + large)
        exceeded = (
            utilization_design(
                catalogue.tube(jnp.exp(middle)),
                actions,
                l_cr,
                steel,
                plastic=plastic,
                resultant=resultant,
            )
            > 1.0
        )

        return jnp.where(exceeded, middle, small), jnp.where(exceeded, large, middle)

    ceiling = catalogue.tube(jnp.exp(over))
    bracketed = (
        utilization_design(
            ceiling, actions, l_cr, steel, plastic=plastic, resultant=resultant
        )
        <= 1.0
    )

    under, over = lax.fori_loop(0, BISECTION_HALVINGS, halve, (under, over))

    return jnp.where(bracketed, jnp.exp(over), jnp.nan)


@partial(jax.custom_jvp, nondiff_argnums=(0, 1))
def _diameter(
    plastic: bool,
    resultant: bool,
    actions: MemberActions,
    l_cr: Float[Array, "members"],
    steel: SteelGrade,
    catalogue: TubeCatalogue,
) -> Float[Array, "members"]:
    """
    The sizing map, with the class as the leading static argument.

    Returns
    -------
    diameter :
        Fully-stressed outer diameter.
    """
    return _solve(plastic, resultant, actions, l_cr, steel, catalogue)


@_diameter.defjvp
def _diameter_jvp(
    plastic: bool,
    resultant: bool,
    primals: tuple[MemberActions, Float[Array, "members"], SteelGrade, TubeCatalogue],
    tangents: tuple[MemberActions, Float[Array, "members"], SteelGrade, TubeCatalogue],
) -> tuple[Float[Array, "members"], Float[Array, "members"]]:
    """
    Differentiate the map by the implicit function theorem.

    Returns
    -------
    primal_and_tangent :
        The fully-stressed diameter and its directional derivative.

    Notes
    -----
    The check is one at the root for every choice of actions, so its total
    derivative there vanishes. Splitting that into the diameter and everything
    else and solving for the diameter gives the tangent below: the drift of the
    check at a frozen diameter, divided by the slope of the check in the
    diameter, with a change of sign.

    Only that division is derived by hand. The two partial derivatives come from
    differentiating the check itself, which is explicit and smooth wherever the
    root is, so whichever branch of the standard governs is the branch that gets
    differentiated, with no dispatch written here.

    A tangent rather than an adjoint, so that both modes work: the rule is
    linear in the tangents and the transposition to reverse mode is automatic.
    The bisection is untouched by either, since it depends on the primals alone.

    A member carrying no action at all has a flat check with no root, so the
    slope vanishes and there is nothing to invert. Its size is then decided by
    the catalogue rather than by the standard, and the tangent is zero.
    """
    solved = _solve(plastic, resultant, *primals)
    acting, buckling, grade, family = primals

    def check(size: Float[Array, "members"]) -> Float[Array, ""]:
        demand = utilization_design(
            family.tube(size),
            acting,
            buckling,
            grade,
            plastic=plastic,
            resultant=resultant,
        )
        if jnp.shape(demand) != jnp.shape(size):
            raise ValueError(
                f"the check broadcast to {jnp.shape(demand)} against a diameter "
                f"of {jnp.shape(size)}; actions must share one member axis"
            )

        return jnp.sum(demand)

    def at_root(
        actions: MemberActions,
        l_cr: Float[Array, "members"],
        steel: SteelGrade,
        catalogue: TubeCatalogue,
    ) -> Float[Array, "members"]:
        return utilization_design(
            catalogue.tube(solved),
            actions,
            l_cr,
            steel,
            plastic=plastic,
            resultant=resultant,
        )

    slope = jax.grad(check)(solved)
    _, drift = jax.jvp(at_root, primals, tangents)

    rooted = slope != 0.0
    tangent = jnp.where(rooted, -drift / jnp.where(rooted, slope, 1.0), 0.0)

    return solved, tangent


def diameter_required(
    actions: MemberActions,
    l_cr: Float[Array, "members"],
    steel: SteelGrade,
    catalogue: TubeCatalogue,
    *,
    plastic: bool,
    resultant: bool = True,
) -> Float[Array, "members"]:
    """
    Fully-stressed outer diameter of a member.

    Parameters
    ----------
    actions :
        Design actions on the member.
    l_cr :
        Buckling length.
    steel :
        Material properties and partial factors.
    catalogue :
        The section family the member is drawn from.
    plastic :
        Whether the section is Class 1 or 2. Static, never a traced value.

    Returns
    -------
    diameter :
        Diameter at which EN 1993-1-1 is exactly satisfied, or the family
        minimum where that is already sufficient.

    Notes
    -----
    Differentiable in every argument, including the material properties and the
    diameter-to-thickness ratio, in both forward and reverse mode. Under jit the
    class must be marked static, since it selects a clause rather than scaling a
    number.

    The utilization at the returned diameter is one to machine precision, except
    where the family minimum governs and the member is deliberately
    understressed. The minimum is applied outside the solved map, so where it
    binds the derivative follows it rather than the standard, which is the
    honest answer: the actions have stopped deciding the size.
    """
    solved = _diameter(plastic, resultant, actions, l_cr, steel, catalogue)

    return jnp.maximum(solved, catalogue.diameter_min)


def mass(
    tubes: Tube,
    lengths: Float[Array, "members"],
    steel: SteelGrade,
) -> Float[Array, ""]:
    """
    Total mass of a set of members.

    Parameters
    ----------
    tubes :
        The section of each member.
    lengths :
        Length of each member.
    steel :
        Material properties and partial factors.

    Returns
    -------
    mass :
        Total mass.

    Notes
    -----
    Geometry, not EN 1993-1-1. The objective the sizing map exists to serve.
    """
    return steel.density * jnp.sum(tubes.area * lengths)


def governing_limit_state(
    tube: Tube,
    actions: MemberActions,
    l_cr: Float[Array, "members"],
    steel: SteelGrade,
    catalogue: TubeCatalogue,
    *,
    plastic: bool,
    resultant: bool = True,
) -> Float[Array, "members"]:
    """
    Which limit state decided a member's size.

    Parameters
    ----------
    tube :
        The section the sizing map returned.
    actions :
        Design actions on the member, the same ones it was sized for.
    l_cr :
        Buckling length.
    steel :
        Material properties and partial factors.
    catalogue :
        The section family the member is drawn from, read only for its floor.
    plastic :
        Whether the section is Class 1 or 2. Static, never a traced value.

    Returns
    -------
    governing :
        One of the limit-state codes of this module.

    Notes
    -----
    **Non-differentiable.** A diagnostic reported beside the diameter and never
    differentiated through, so it must be dropped before any gradient is taken.

    The catalogue minimum is reported ahead of everything else, because where it
    binds no clause decided the size at all. Otherwise the two checks are
    compared, and where the member check wins the equation that produced it is
    named. Repeated flips between optimizer steps mean the design is chattering
    across a boundary; flips between tension and the rest are the ones worth
    watching, since the standard is discontinuous there rather than merely
    kinked.
    """
    member, section = _demands(
        tube, actions, l_cr, steel, plastic=plastic, resultant=resultant
    )

    gross = tube.area
    modulus = _modulus(tube, plastic=plastic)
    critical = force_critical(tube.second_moment, l_cr, steel)
    lam = slenderness_from_force(gross, steel, critical)
    reduction = reduction_buckling(lam, steel.alpha)

    equation = governing_equation(
        jnp.maximum(-jnp.asarray(actions.n_ed), 0.0),
        jnp.abs(actions.m_y_ed),
        jnp.abs(actions.m_z_ed),
        reduction,
        reduction,
        gross * steel.f_y,
        modulus * steel.f_y,
        lam,
        lam,
        actions.c_my,
        actions.c_mz,
        steel.gamma_m1,
        plastic=plastic,
    )

    by_member = jnp.where(equation > 0.0, LIMIT_MINOR, LIMIT_MAJOR)
    by_section = jnp.where(
        jnp.asarray(actions.n_ed) >= 0.0, LIMIT_TENSION, LIMIT_CROSS_SECTION
    )
    decided = jnp.where(member > section, by_member, by_section)

    at_minimum = jnp.asarray(tube.diameter) <= catalogue.diameter_min * (1.0 + 1e-12)

    return jnp.where(at_minimum, LIMIT_MINIMUM_SIZE, decided)


def end_moments(
    m_first: Float[Array, "members"],
    m_second: Float[Array, "members"],
) -> tuple[Float[Array, "members"], Float[Array, "members"]]:
    """
    Design moment and equivalent uniform moment factor, from the two end moments.

    Parameters
    ----------
    m_first :
        Bending moment at one end of the member.
    m_second :
        Bending moment at the other end.

    Returns
    -------
    design_and_factor :
        The larger end moment in magnitude, and the equivalent uniform moment
        factor of EN 1993-1-1 Table B.3.

    Notes
    -----
    EN 1993-1-1 Table B.3, first row. Loads applied only at nodes leave the
    moment varying linearly along a member, so that row is exact here rather
    than approximate, and the rest of the table never activates.

    Both moments are read in the bending-diagram convention, in which a uniform
    moment has them equal and of the same sign. The factor is one for a uniform
    moment and falls to its floor under a symmetric reversal.

    A member with no moment at either end is given a factor of one, which is the
    value the factor is never used at, since the moment it multiplies is zero.
    """
    first = jnp.abs(m_first)
    second = jnp.abs(m_second)

    larger = jnp.maximum(first, second)
    smaller = jnp.minimum(first, second)

    bent = larger > 0.0
    ratio = jnp.sign(jnp.asarray(m_first) * jnp.asarray(m_second)) * smaller
    psi = jnp.where(bent, ratio / jnp.where(bent, larger, 1.0), 1.0)

    return larger, moment_factor_linear(psi)


def diameter_envelope(
    diameters: Float[Array, "cases members"],
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
        Diameter covering every case.

    Notes
    -----
    Not EN 1993-1-1. A member must satisfy every load case, so its size is the
    largest any case demands; that largest is not differentiable, and a gradient
    taken through it sees one case at a time and stalls.

    The envelope is taken in the logarithm of the diameter, which makes the
    sharpness dimensionless and so comparable between structures of different
    size. It never understates the largest, and exceeds it by at most the
    logarithm of the number of cases over the sharpness, so annealing the
    sharpness upward drives it onto the true largest from above. Being an upper
    bound is the safe direction: the design stays adequate throughout.
    """
    logarithms = jnp.log(diameters)
    smoothed = logsumexp(beta * logarithms, axis=0) / beta

    return jnp.exp(smoothed)
