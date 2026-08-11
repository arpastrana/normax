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
Cross-section and member resistances, EN 1993-1-1 sections 6.2 and 6.3.

Tension, compression, bending, their combination at cross-section level, shear,
and flexural buckling. Every function takes section properties rather than a
diameter, so the clause layer stays independent of the cross-section that
produced them.

Shear is not part of the design check the sizing map solves. It is here so that
a converged design can be audited afterwards, since excluding shear is only
defensible while the design shear stays below half the plastic shear
resistance.

Out of scope, deliberately. Lateral-torsional buckling, 6.3.2: a circular
hollow section is doubly symmetric and closed, so the reduction factor is one
and flexural buckling governs. Class 4, Eqs. 6.11 and 6.48: the fixed
diameter-to-thickness ratio pins the section at the Class 3 boundary, so
effective properties are never reached. Torsional and flexural-torsional
buckling, Eqs. 6.52 and 6.53: those modes are limited to open sections.
"""

import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float

# EN 1993-1-1 6.1. Nationally determined; these are the values the UK National
# Annex sets in clause NA.2.15.
GAMMA_M0 = 1.0
GAMMA_M1 = 1.0
GAMMA_M2 = 1.25

# EN 1993-1-1 3.2.6.
E_MODULUS = 210000.0

# EN 1993-1-1 Table 6.1. Table 6.2 selects the curve: hollow sections are
# curve a hot finished and curve c cold formed, and a0 or c respectively at
# the 460 grade.
IMPERFECTION_FACTORS = {
    "a0": 0.13,
    "a": 0.21,
    "b": 0.34,
    "c": 0.49,
    "d": 0.76,
}

# EN 1993-1-1 6.3.1.2(3). At or below this slenderness there is no reduction.
SLENDERNESS_OFFSET = 0.2

# EN 1993-1-1 6.2.9.1(5), the exponent of the circular hollow section's
# reduced plastic moment.
MOMENT_EXPONENT = 1.7

# EN 1993-1-1 6.2.8(2) and 6.2.10. Below this fraction of the plastic shear
# resistance, shear may be ignored in the bending and axial checks.
SHEAR_THRESHOLD = 0.5


def n_pl_rd(
    area: Float[Array, "members"],
    f_y: float | Float[Array, ""],
    gamma_m0: float | Float[Array, ""] = GAMMA_M0,
) -> Float[Array, "members"]:
    """
    Design plastic resistance of the gross cross-section in tension.

    Parameters
    ----------
    area :
        Gross cross-sectional area.
    f_y :
        Yield strength.
    gamma_m0 :
        Partial factor for cross-section resistance.

    Returns
    -------
    n_pl_rd :
        Resistance to yielding of the gross cross-section.

    Notes
    -----
    EN 1993-1-1 6.2.3, Eq. 6.6.
    """
    gross = jnp.asarray(area)

    return gross * f_y / gamma_m0


def n_u_rd(
    area_net: Float[Array, "members"],
    f_u: float | Float[Array, ""],
    gamma_m2: float | Float[Array, ""] = GAMMA_M2,
) -> Float[Array, "members"]:
    """
    Design ultimate resistance of the net cross-section in tension.

    Parameters
    ----------
    area_net :
        Net cross-sectional area at holes for fasteners.
    f_u :
        Ultimate tensile strength.
    gamma_m2 :
        Partial factor for resistance to fracture in tension.

    Returns
    -------
    n_u_rd :
        Resistance to fracture across the net section.

    Notes
    -----
    EN 1993-1-1 6.2.3, Eq. 6.7. The nine-tenths factor is part of the equation,
    not a safety allowance applied on top of it.
    """
    net = jnp.asarray(area_net)

    return 0.9 * net * f_u / gamma_m2


def n_t_rd(
    area: Float[Array, "members"],
    area_net: Float[Array, "members"],
    f_y: float | Float[Array, ""],
    f_u: float | Float[Array, ""],
    gamma_m0: float | Float[Array, ""] = GAMMA_M0,
    gamma_m2: float | Float[Array, ""] = GAMMA_M2,
) -> Float[Array, "members"]:
    """
    Design tension resistance.

    Parameters
    ----------
    area :
        Gross cross-sectional area.
    area_net :
        Net cross-sectional area at holes for fasteners.
    f_y :
        Yield strength.
    f_u :
        Ultimate tensile strength.
    gamma_m0 :
        Partial factor for cross-section resistance.
    gamma_m2 :
        Partial factor for resistance to fracture in tension.

    Returns
    -------
    n_t_rd :
        The lesser of gross-section yielding and net-section fracture.

    Notes
    -----
    EN 1993-1-1 6.2.3. With no holes the net area equals the gross area and
    yielding governs, so this collapses to Eq. 6.6.
    """
    yielding = n_pl_rd(area, f_y, gamma_m0)
    fracture = n_u_rd(area_net, f_u, gamma_m2)

    return jnp.minimum(yielding, fracture)


def n_c_rd(
    area: Float[Array, "members"],
    f_y: float | Float[Array, ""],
    gamma_m0: float | Float[Array, ""] = GAMMA_M0,
) -> Float[Array, "members"]:
    """
    Design resistance of the cross-section to uniform compression.

    Parameters
    ----------
    area :
        Gross cross-sectional area.
    f_y :
        Yield strength.
    gamma_m0 :
        Partial factor for cross-section resistance.

    Returns
    -------
    n_c_rd :
        Squash resistance of the cross-section.

    Notes
    -----
    EN 1993-1-1 6.2.4, Eq. 6.10, for Classes 1, 2 and 3. Numerically the same
    expression as Eq. 6.6, under a different clause. This ignores member
    buckling and governs alone only for stocky members.
    """
    gross = jnp.asarray(area)

    return gross * f_y / gamma_m0


def m_pl_rd(
    w_pl: Float[Array, "members"],
    f_y: float | Float[Array, ""],
    gamma_m0: float | Float[Array, ""] = GAMMA_M0,
) -> Float[Array, "members"]:
    """
    Design plastic resistance of the cross-section to bending.

    Parameters
    ----------
    w_pl :
        Plastic section modulus about the bending axis.
    f_y :
        Yield strength.
    gamma_m0 :
        Partial factor for cross-section resistance.

    Returns
    -------
    m_pl_rd :
        Plastic moment resistance.

    Notes
    -----
    EN 1993-1-1 6.2.5, Eq. 6.13, for Classes 1 and 2.
    """
    modulus = jnp.asarray(w_pl)

    return modulus * f_y / gamma_m0


def m_el_rd(
    w_el: Float[Array, "members"],
    f_y: float | Float[Array, ""],
    gamma_m0: float | Float[Array, ""] = GAMMA_M0,
) -> Float[Array, "members"]:
    """
    Design elastic resistance of the cross-section to bending.

    Parameters
    ----------
    w_el :
        Elastic section modulus about the bending axis.
    f_y :
        Yield strength.
    gamma_m0 :
        Partial factor for cross-section resistance.

    Returns
    -------
    m_el_rd :
        Elastic moment resistance.

    Notes
    -----
    EN 1993-1-1 6.2.5, Eq. 6.14, for Class 3.
    """
    modulus = jnp.asarray(w_el)

    return modulus * f_y / gamma_m0


def area_shear(area: Float[Array, "members"]) -> Float[Array, "members"]:
    """
    Shear area of a circular hollow section.

    Parameters
    ----------
    area :
        Gross cross-sectional area.

    Returns
    -------
    area_shear :
        Area mobilised to resist shear.

    Notes
    -----
    EN 1993-1-1 6.2.6(3), the tubular entry of an unnumbered list, for a section
    of uniform thickness. Two over pi is the shear form factor of a thin ring.
    """
    gross = jnp.asarray(area)

    return 2.0 * gross / jnp.pi


def v_pl_rd(
    area_shear: Float[Array, "members"],
    f_y: float | Float[Array, ""],
    gamma_m0: float | Float[Array, ""] = GAMMA_M0,
) -> Float[Array, "members"]:
    """
    Design plastic shear resistance.

    Parameters
    ----------
    area_shear :
        Area mobilised to resist shear.
    f_y :
        Yield strength.
    gamma_m0 :
        Partial factor for cross-section resistance.

    Returns
    -------
    v_pl_rd :
        Plastic shear resistance.

    Notes
    -----
    EN 1993-1-1 6.2.6(2), Eq. 6.18. The shear yield strength is the tensile one
    divided by the square root of three, which is the von Mises criterion in
    pure shear. Written for the shear area rather than the gross area, so it
    holds for any cross-section type.
    """
    shear = jnp.asarray(area_shear)

    return shear * f_y / (jnp.sqrt(3.0) * gamma_m0)


def m_n_rd(
    m_plastic: Float[Array, "members"],
    n: Float[Array, "members"],
) -> Float[Array, "members"]:
    """
    Plastic moment resistance reduced for axial force.

    Parameters
    ----------
    m_plastic :
        Plastic moment resistance without axial force.
    n :
        Design axial force over the plastic resistance to axial force.

    Returns
    -------
    m_n_rd :
        Reduced plastic moment resistance.

    Notes
    -----
    EN 1993-1-1 6.2.9.1(5), for circular hollow sections. The expression
    carries no equation number: it sits unnumbered between the numbered forms
    for I and H sections and those for rectangular hollow sections.

    A circular hollow section is not among the types 6.2.9.1(4) exempts from
    the reduction, so this applies at every axial force. It also collapses the
    biaxial check of 6.2.9.1(6) to a resultant, since the section is
    axisymmetric and both exponents are two.

    The first derivative vanishes as the axial force does, so gradients are
    well behaved at pure bending; the second derivative diverges there.
    """
    ratio = jnp.asarray(n)

    return m_plastic * (1.0 - ratio**MOMENT_EXPONENT)


def moment_resultant(
    m_y_ed: Float[Array, "members"],
    m_z_ed: Float[Array, "members"],
) -> Float[Array, "members"]:
    """
    Resultant of the two bending moments on an axisymmetric section.

    Parameters
    ----------
    m_y_ed :
        Design bending moment about the major axis.
    m_z_ed :
        Design bending moment about the minor axis.

    Returns
    -------
    m_resultant :
        Resultant bending moment.

    Notes
    -----
    EN 1993-1-1 6.2.9.1(6), Eq. 6.41. Both exponents are two for a circular
    hollow section, and its two reduced moment resistances are equal, so the
    interaction is exactly a comparison of this resultant against one of them.

    Cross-section level only. The member check of 6.3.3 keeps the two moments
    separate, since its terms are linear and no source sanctions combining
    them.

    The square root is guarded so that the gradient at the origin is zero rather
    than undefined. Without the guard a member carrying no moment poisons every
    gradient that reaches it, including through a comparison it loses, since the
    undefined value survives being multiplied by zero.
    """
    square = jnp.asarray(m_y_ed) ** 2 + jnp.asarray(m_z_ed) ** 2
    positive = square > 0.0

    return jnp.where(positive, jnp.sqrt(jnp.where(positive, square, 1.0)), 0.0)


def moment_combined(
    m_y_ed: Float[Array, "members"],
    m_z_ed: Float[Array, "members"],
    *,
    plastic: bool,
    resultant: bool = True,
) -> Float[Array, "members"]:
    """
    The two bending moments as the cross-section check reads them.

    Parameters
    ----------
    m_y_ed :
        Design bending moment about the major axis.
    m_z_ed :
        Design bending moment about the minor axis.
    plastic :
        Whether the section is Class 1 or 2. Static, never a traced value.
    resultant :
        Whether to combine the two into a resultant rather than summing them.
        Static, never a traced value.

    Returns
    -------
    moment :
        Resultant on the plastic branch, and either reading on the elastic one.

    Notes
    -----
    EN 1993-1-1 6.2.9. The reading is open only on the elastic branch: for an
    axisymmetric section Eq. 6.41 takes both exponents as two and collapses to a
    resultant exactly, so 6.2.9.1 has nothing to choose and the flag applies to
    6.2.9.2 alone.

    Shared with the analytic lower bound on the diameter, which has to combine
    the moments the way the check does or it stops bounding it.
    """
    if plastic or resultant:
        return moment_resultant(m_y_ed, m_z_ed)

    return jnp.abs(jnp.asarray(m_y_ed)) + jnp.abs(jnp.asarray(m_z_ed))


def utilization_plastic(
    n_ed: Float[Array, "members"],
    m_y_ed: Float[Array, "members"],
    m_z_ed: Float[Array, "members"],
    area: Float[Array, "members"],
    w_pl: Float[Array, "members"],
    f_y: float | Float[Array, ""],
    gamma_m0: float | Float[Array, ""] = GAMMA_M0,
) -> Float[Array, "members"]:
    """
    Cross-section utilization under bending and axial force, Classes 1 and 2.

    Parameters
    ----------
    n_ed :
        Design axial force, of either sign.
    m_y_ed :
        Design bending moment about the major axis.
    m_z_ed :
        Design bending moment about the minor axis.
    area :
        Gross cross-sectional area.
    w_pl :
        Plastic section modulus.
    f_y :
        Yield strength.
    gamma_m0 :
        Partial factor for cross-section resistance.

    Returns
    -------
    utilization :
        Left-hand side of the interaction, at most one if the cross-section is
        adequate.

    Notes
    -----
    EN 1993-1-1 6.2.9.1, Eq. 6.41 with the reduced resistance of 6.2.9.1(5), and
    6.2.4 with it. The clause requires the resultant moment to stay below a
    plastic moment already reduced for axial force. Dividing that requirement
    through by the unreduced plastic moment moves the reduction to the other side
    and turns it into a sum, which is the form returned.

    That rearrangement is exact, not an approximation, and it is what makes the
    expression usable. The quotient form is singular exactly where axial force
    alone exhausts the section, is negative beyond it, and loses precision as it
    is approached; the sum is finite and strictly increasing everywhere. The two
    have the same root, and the factor between them cancels in the implicit
    derivative at that root.

    The sum also recovers the squash check for free. With no moment it is the
    axial ratio raised to the exponent, whose root is an axial ratio of one,
    which is Eq. 6.10. The quotient form loses that check entirely, reporting a
    fully squashed section as unutilized.

    Only the magnitude of the axial force enters, so the clause reads the same
    in tension and in compression. The two moments combine into a resultant,
    exact rather than approximate here: both exponents of Eq. 6.41 are two for a
    circular hollow section and its two reduced resistances are equal.
    """
    combined = moment_combined(m_y_ed, m_z_ed, plastic=True)

    axial = jnp.abs(jnp.asarray(n_ed)) / n_pl_rd(area, f_y, gamma_m0)
    bending = combined / m_pl_rd(w_pl, f_y, gamma_m0)

    return axial**MOMENT_EXPONENT + bending


def utilization_elastic(
    n_ed: Float[Array, "members"],
    m_y_ed: Float[Array, "members"],
    m_z_ed: Float[Array, "members"],
    area: Float[Array, "members"],
    w_el: Float[Array, "members"],
    f_y: float | Float[Array, ""],
    gamma_m0: float | Float[Array, ""] = GAMMA_M0,
    *,
    resultant: bool = True,
) -> Float[Array, "members"]:
    """
    Cross-section utilization under bending and axial force, Class 3.

    Parameters
    ----------
    n_ed :
        Design axial force, of either sign.
    m_y_ed :
        Design bending moment about the major axis.
    m_z_ed :
        Design bending moment about the minor axis.
    area :
        Gross cross-sectional area.
    w_el :
        Elastic section modulus.
    f_y :
        Yield strength.
    gamma_m0 :
        Partial factor for cross-section resistance.
    resultant :
        Whether to combine the two moments into a resultant rather than summing
        them. Static, never a traced value. See the note below: the sources
        disagree and this selects between their readings.

    Returns
    -------
    utilization :
        Greatest longitudinal stress over the design yield strength.

    Notes
    -----
    EN 1993-1-1 6.2.9.2, Eq. 6.42, a limit on the elastic stress of the gross
    section. Dividing that limit through by the design yield strength turns it
    into the sum of the axial and bending ratios, which is what is returned.

    **The two moments may be combined two ways and the sources disagree**, so
    both are implemented and the choice is recorded rather than buried. Under a
    resultant, the peak stress is exact for a circular section: the bending
    stress at perimeter angle theta weights the two moments by the sine and the
    cosine of that angle, so its greatest magnitude around the perimeter is
    their resultant over the elastic modulus. Under a sum, each moment is
    charged at its own peak, which no single point of a circular section
    experiences; that is the safe envelope for a section whose extreme fibre
    sees both, such as the corner of an I or a box, and it is the form
    EN 1993-1-1 writes out for Class 4 in Eq. 6.44.

    The sum is the larger of the two, by at most the square root of two, so it
    is always the conservative reading.

    Unlike the plastic branch this needs no separate axial check, since the
    axial term survives when the moments vanish.
    """
    combined = moment_combined(m_y_ed, m_z_ed, plastic=False, resultant=resultant)

    axial = jnp.abs(jnp.asarray(n_ed)) / n_pl_rd(area, f_y, gamma_m0)

    return axial + combined / m_el_rd(w_el, f_y, gamma_m0)


def utilization_cross_section(
    n_ed: Float[Array, "members"],
    m_y_ed: Float[Array, "members"],
    m_z_ed: Float[Array, "members"],
    area: Float[Array, "members"],
    modulus: Float[Array, "members"],
    f_y: float | Float[Array, ""],
    gamma_m0: float | Float[Array, ""] = GAMMA_M0,
    *,
    plastic: bool,
    resultant: bool = True,
) -> Float[Array, "members"]:
    """
    Cross-section utilization under bending and axial force.

    Parameters
    ----------
    n_ed :
        Design axial force, of either sign.
    m_y_ed :
        Design bending moment about the major axis.
    m_z_ed :
        Design bending moment about the minor axis.
    area :
        Gross cross-sectional area.
    modulus :
        Section modulus about either axis, plastic or elastic to match the class.
    f_y :
        Yield strength.
    gamma_m0 :
        Partial factor for cross-section resistance.
    plastic :
        Whether the section is Class 1 or 2. Static, never a traced value.
    resultant :
        Whether the elastic branch combines the two moments into a resultant
        rather than summing them. Ignored on the plastic branch, where the
        resultant is exact. Static, never a traced value.

    Returns
    -------
    utilization :
        Cross-section demand over resistance.

    Notes
    -----
    EN 1993-1-1 6.2.9. The class selects the clause: 6.2.9.1 and its plastic
    interaction for Classes 1 and 2, 6.2.9.2 and its elastic stress limit for
    Class 3. The class follows from the configured diameter-to-thickness ratio
    and is therefore a build-time choice, not a branch on a traced value.

    Only the elastic branch is ambiguous about how the two moments combine. On
    the plastic branch Eq. 6.41 takes both exponents as two for a circular
    hollow section, which both sources confirm, and the collapse to a resultant
    is then exact algebra rather than an interpretation.
    """
    if plastic:
        return utilization_plastic(n_ed, m_y_ed, m_z_ed, area, modulus, f_y, gamma_m0)

    return utilization_elastic(
        n_ed, m_y_ed, m_z_ed, area, modulus, f_y, gamma_m0, resultant=resultant
    )


def n_cr(
    second_moment: Float[Array, "members"],
    l_cr: Float[Array, "members"],
    e_mod: float | Float[Array, ""] = E_MODULUS,
) -> Float[Array, "members"]:
    """
    Elastic critical force for flexural buckling.

    Parameters
    ----------
    second_moment :
        Second moment of area about the buckling axis.
    l_cr :
        Buckling length.
    e_mod :
        Modulus of elasticity.

    Returns
    -------
    n_cr :
        Euler critical force.

    Notes
    -----
    Classical elastic stability, used explicitly by EN 1993-1-1 6.3.1.3 but not
    itself a numbered equation in the standard.
    """
    inertia = jnp.asarray(second_moment)

    return jnp.pi**2 * e_mod * inertia / l_cr**2


def lambda_1(
    f_y: float | Float[Array, ""],
    e_mod: float | Float[Array, ""] = E_MODULUS,
) -> Float[Array, ""]:
    """
    Reference slenderness.

    Parameters
    ----------
    f_y :
        Yield strength.
    e_mod :
        Modulus of elasticity.

    Returns
    -------
    lambda_1 :
        Slenderness at which the Euler stress equals the yield strength.

    Notes
    -----
    EN 1993-1-1 6.3.1.3. Appears in the code's tables as 93.9 times the
    material factor.
    """
    return jnp.pi * jnp.sqrt(jnp.asarray(e_mod) / f_y)


def slenderness(
    area: Float[Array, "members"],
    f_y: float | Float[Array, ""],
    n_critical: Float[Array, "members"],
) -> Float[Array, "members"]:
    """
    Non-dimensional slenderness for flexural buckling.

    Parameters
    ----------
    area :
        Gross cross-sectional area.
    f_y :
        Yield strength.
    n_critical :
        Elastic critical force.

    Returns
    -------
    slenderness :
        Non-dimensional slenderness, the square root of the squash load over
        the critical load.

    Notes
    -----
    EN 1993-1-1 6.3.1.3, Eq. 6.50, for Classes 1, 2 and 3.
    """
    gross = jnp.asarray(area)

    return jnp.sqrt(gross * f_y / n_critical)


def slenderness_gyration(
    l_cr: Float[Array, "members"],
    radius_gyration: Float[Array, "members"],
    f_y: float | Float[Array, ""],
    e_mod: float | Float[Array, ""] = E_MODULUS,
) -> Float[Array, "members"]:
    """
    Non-dimensional slenderness, from the geometric slenderness.

    Parameters
    ----------
    l_cr :
        Buckling length.
    radius_gyration :
        Radius of gyration about the buckling axis.
    f_y :
        Yield strength.
    e_mod :
        Modulus of elasticity.

    Returns
    -------
    slenderness :
        Non-dimensional slenderness.

    Notes
    -----
    EN 1993-1-1 6.3.1.3, Eq. 6.50, second form. Algebraically identical to the
    first form; it is the one the code's tables are written against.
    """
    geometric = jnp.asarray(l_cr) / radius_gyration

    return geometric / lambda_1(f_y, e_mod)


def phi(
    lam: Float[Array, "members"],
    alpha: float | Float[Array, ""],
) -> Float[Array, "members"]:
    """
    Auxiliary term of the buckling curve.

    Parameters
    ----------
    lam :
        Non-dimensional slenderness.
    alpha :
        Imperfection factor of the buckling curve.

    Returns
    -------
    phi :
        Auxiliary term.

    Notes
    -----
    EN 1993-1-1 6.3.1.2. Carries no equation number: it sits unnumbered
    directly beneath Eq. 6.49. The offset of 0.2 is what makes the reduction
    factor exactly one at that slenderness, for every curve.
    """
    slender = jnp.asarray(lam)

    return 0.5 * (1.0 + alpha * (slender - SLENDERNESS_OFFSET) + slender**2)


def chi(
    lam: Float[Array, "members"],
    alpha: float | Float[Array, ""],
) -> Float[Array, "members"]:
    """
    Reduction factor for flexural buckling.

    Parameters
    ----------
    lam :
        Non-dimensional slenderness.
    alpha :
        Imperfection factor of the buckling curve.

    Returns
    -------
    chi :
        Reduction factor, capped at one.

    Notes
    -----
    EN 1993-1-1 6.3.1.2, Eq. 6.49. The cap implements 6.3.1.2(3): at or below a
    slenderness of 0.2 buckling may be ignored and only section 6.2 applies.
    The reduction factor never exceeds the Euler value, the reciprocal of the
    slenderness squared, and approaches it from below as the member slenders.

    The argument of the square root stays well clear of zero over the whole
    range of the five curves, so it needs no guard. Clipping it would change
    the gradient.
    """
    slender = jnp.asarray(lam)
    auxiliary = phi(slender, alpha)
    reduction = 1.0 / (auxiliary + jnp.sqrt(auxiliary**2 - slender**2))

    return jnp.minimum(reduction, 1.0)


def n_b_rd(
    reduction: Float[Array, "members"],
    area: Float[Array, "members"],
    f_y: float | Float[Array, ""],
    gamma_m1: float | Float[Array, ""] = GAMMA_M1,
) -> Float[Array, "members"]:
    """
    Design buckling resistance of a compression member.

    Parameters
    ----------
    reduction :
        Reduction factor for the relevant buckling mode.
    area :
        Gross cross-sectional area.
    f_y :
        Yield strength.
    gamma_m1 :
        Partial factor for member instability.

    Returns
    -------
    n_b_rd :
        Buckling resistance.

    Notes
    -----
    EN 1993-1-1 6.3.1, Eq. 6.47, for Classes 1, 2 and 3.
    """
    gross = jnp.asarray(area)

    return reduction * gross * f_y / gamma_m1
