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
Bending and axial compression in a member, EN 1993-1-1 6.3.3 and Annex B.

Method 2, the Annex B interaction factors. Two readings of the standard are
taken here, both recorded in docs/clauses.md rather than inferred in code:

Table B.1 lists only I-sections and rectangular hollow sections, and a circular
hollow section matches neither row even though 6.3.3 sends it to that table.
The rectangular row is used, being the only closed-section entry.

The equivalent uniform moment factor comes from the linear row of Table B.3.
That is exact rather than approximate under nodal loading, since a member with
no load along its span carries a linear moment diagram.

Whether the plastic or the elastic column of Table B.1 applies follows from the
cross-section class, which is fixed by the configured diameter-to-thickness
ratio. It is therefore a static choice, never a branch on a traced value.
"""

import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float

from normax.ec3.resistance import GAMMA_M1

# EN 1993-1-1 Table B.3, the lower bound on the linear row.
C_M_MINIMUM = 0.4

# Which of the two interaction equations governs, reported alongside the
# utilization as a non-differentiable diagnostic.
GOVERNING_MAJOR = 0.0
GOVERNING_MINOR = 1.0


def c_m_linear(psi: Float[Array, "members"]) -> Float[Array, "members"]:
    """
    Equivalent uniform moment factor for a linear moment diagram.

    Parameters
    ----------
    psi :
        Ratio of the smaller to the larger end moment, signed, so that a
        uniform moment gives one and a symmetric reversal gives minus one.

    Returns
    -------
    c_m :
        Equivalent uniform moment factor, floored at 0.4.

    Notes
    -----
    EN 1993-1-1 Table B.3, first row. Exact for a member carrying no load
    between its ends.

    The floor is a kink in the slope, the same kind as the cap on the
    interaction factors and on the buckling reduction factor.
    """
    ratio = jnp.asarray(psi)

    return jnp.maximum(0.6 + 0.4 * ratio, C_M_MINIMUM)


def axial_ratio(
    n_ed: Float[Array, "members"],
    chi: Float[Array, "members"],
    n_rk: Float[Array, "members"],
    gamma_m1: float | Float[Array, ""] = GAMMA_M1,
) -> Float[Array, "members"]:
    """
    Axial force over the buckling resistance, as the interaction factors use it.

    Parameters
    ----------
    n_ed :
        Design axial compression.
    chi :
        Reduction factor for flexural buckling about the relevant axis.
    n_rk :
        Characteristic resistance to axial force.
    gamma_m1 :
        Partial factor for member instability.

    Returns
    -------
    ratio :
        Utilization of the axial resistance about that axis.

    Notes
    -----
    EN 1993-1-1 Annex B, Table B.1. The standard writes this ratio out in full
    rather than naming it. Distinct from Annex A's utilization, which carries
    no reduction factor.
    """
    force = jnp.asarray(n_ed)

    return force / (chi * n_rk / gamma_m1)


def _k_axial(
    c_m: Float[Array, "members"],
    slope: Float[Array, "members"],
    cap_slope: float,
    ratio: Float[Array, "members"],
) -> Float[Array, "members"]:
    """
    Shared shape of the diagonal interaction factors.

    Parameters
    ----------
    c_m :
        Equivalent uniform moment factor.
    slope :
        Coefficient multiplying the axial ratio in the unbounded expression.
    cap_slope :
        Coefficient multiplying the axial ratio in the bound.
    ratio :
        Axial force over the buckling resistance about the relevant axis.

    Returns
    -------
    k :
        Interaction factor.

    Notes
    -----
    Both rows of EN 1993-1-1 Table B.1 have the form of a linear rise in the
    axial ratio, bounded by a second linear rise with a smaller coefficient.
    Only the two coefficients change between plastic and elastic.
    """
    axial = jnp.asarray(ratio)
    unbounded = c_m * (1.0 + slope * axial)
    bound = c_m * (1.0 + cap_slope * axial)

    return jnp.minimum(unbounded, bound)


def k_yy(
    c_my: Float[Array, "members"],
    lam_y: Float[Array, "members"],
    ratio_y: Float[Array, "members"],
    *,
    plastic: bool,
) -> Float[Array, "members"]:
    """
    Interaction factor coupling axial force to major-axis bending.

    Parameters
    ----------
    c_my :
        Equivalent uniform moment factor for major-axis bending.
    lam_y :
        Non-dimensional slenderness about the major axis.
    ratio_y :
        Axial force over the major-axis buckling resistance.
    plastic :
        Whether the section is Class 1 or 2. Static, never a traced value.

    Returns
    -------
    k_yy :
        Interaction factor.

    Notes
    -----
    EN 1993-1-1 Annex B, Table B.1, hollow-section row.
    """
    if plastic:
        return _k_axial(c_my, jnp.asarray(lam_y) - 0.2, 0.8, ratio_y)

    return _k_axial(c_my, 0.6 * jnp.asarray(lam_y), 0.6, ratio_y)


def k_zz(
    c_mz: Float[Array, "members"],
    lam_z: Float[Array, "members"],
    ratio_z: Float[Array, "members"],
    *,
    plastic: bool,
) -> Float[Array, "members"]:
    """
    Interaction factor coupling axial force to minor-axis bending.

    Parameters
    ----------
    c_mz :
        Equivalent uniform moment factor for minor-axis bending.
    lam_z :
        Non-dimensional slenderness about the minor axis.
    ratio_z :
        Axial force over the minor-axis buckling resistance.
    plastic :
        Whether the section is Class 1 or 2. Static, never a traced value.

    Returns
    -------
    k_zz :
        Interaction factor.

    Notes
    -----
    EN 1993-1-1 Annex B, Table B.1, hollow-section row — the same expression as
    the major-axis factor. Only I-sections take a different one, with twice the
    slenderness coefficient and a larger bound.
    """
    if plastic:
        return _k_axial(c_mz, jnp.asarray(lam_z) - 0.2, 0.8, ratio_z)

    return _k_axial(c_mz, 0.6 * jnp.asarray(lam_z), 0.6, ratio_z)


def k_yz(
    k_zz_value: Float[Array, "members"],
    *,
    plastic: bool,
) -> Float[Array, "members"]:
    """
    Interaction factor coupling major-axis buckling to minor-axis bending.

    Parameters
    ----------
    k_zz_value :
        The minor-axis diagonal interaction factor.
    plastic :
        Whether the section is Class 1 or 2. Static, never a traced value.

    Returns
    -------
    k_yz :
        Interaction factor.

    Notes
    -----
    EN 1993-1-1 Annex B, Table B.1. Six tenths of the diagonal factor for a
    plastic section, and the diagonal factor itself for an elastic one.
    """
    if plastic:
        return 0.6 * jnp.asarray(k_zz_value)

    return jnp.asarray(k_zz_value)


def k_zy(
    k_yy_value: Float[Array, "members"],
    *,
    plastic: bool,
) -> Float[Array, "members"]:
    """
    Interaction factor coupling minor-axis buckling to major-axis bending.

    Parameters
    ----------
    k_yy_value :
        The major-axis diagonal interaction factor.
    plastic :
        Whether the section is Class 1 or 2. Static, never a traced value.

    Returns
    -------
    k_zy :
        Interaction factor.

    Notes
    -----
    EN 1993-1-1 Annex B, Table B.1. A footnote permits zero for a hollow
    section under compression and major-axis bending alone; that relaxation is
    not taken here.
    """
    if plastic:
        return 0.6 * jnp.asarray(k_yy_value)

    return 0.8 * jnp.asarray(k_yy_value)


def utilization(
    n_ed: Float[Array, "members"],
    m_y_ed: Float[Array, "members"],
    m_z_ed: Float[Array, "members"],
    chi_y: Float[Array, "members"],
    chi_z: Float[Array, "members"],
    n_rk: Float[Array, "members"],
    m_rk: Float[Array, "members"],
    lam_y: Float[Array, "members"],
    lam_z: Float[Array, "members"],
    c_my: Float[Array, "members"],
    c_mz: Float[Array, "members"],
    gamma_m1: float | Float[Array, ""] = GAMMA_M1,
    *,
    plastic: bool,
) -> Float[Array, "members"]:
    """
    Utilization of a member under axial compression and biaxial bending.

    Parameters
    ----------
    n_ed :
        Design axial compression.
    m_y_ed :
        Design bending moment about the major axis.
    m_z_ed :
        Design bending moment about the minor axis.
    chi_y :
        Reduction factor for flexural buckling about the major axis.
    chi_z :
        Reduction factor for flexural buckling about the minor axis.
    n_rk :
        Characteristic resistance to axial force.
    m_rk :
        Characteristic bending resistance, the same about both axes for a
        circular hollow section.
    lam_y :
        Non-dimensional slenderness about the major axis.
    lam_z :
        Non-dimensional slenderness about the minor axis.
    c_my :
        Equivalent uniform moment factor for major-axis bending.
    c_mz :
        Equivalent uniform moment factor for minor-axis bending.
    gamma_m1 :
        Partial factor for member instability.
    plastic :
        Whether the section is Class 1 or 2. Static, never a traced value.

    Returns
    -------
    utilization :
        The larger of the two interaction checks. At most one, if the member
        is adequate.

    Notes
    -----
    EN 1993-1-1 6.3.3, the larger of Eq. 6.61 and Eq. 6.62. The reduction
    factor for lateral-torsional buckling is one throughout: a circular hollow
    section is closed and doubly symmetric, so it is not susceptible.

    The two equations do not coincide. They share the axial term only when the
    two reduction factors are equal, and they weight the two moments
    oppositely, so they agree only when the moments are equal as well. Taking
    the larger is what makes this one check rather than two.
    """
    first, second = _checks(
        n_ed,
        m_y_ed,
        m_z_ed,
        chi_y,
        chi_z,
        n_rk,
        m_rk,
        lam_y,
        lam_z,
        c_my,
        c_mz,
        gamma_m1,
        plastic=plastic,
    )

    return jnp.maximum(first, second)


def interaction_factors(
    n_ed: Float[Array, "members"],
    chi_y: Float[Array, "members"],
    chi_z: Float[Array, "members"],
    n_rk: Float[Array, "members"],
    lam_y: Float[Array, "members"],
    lam_z: Float[Array, "members"],
    c_my: Float[Array, "members"],
    c_mz: Float[Array, "members"],
    gamma_m1: float | Float[Array, ""] = GAMMA_M1,
    *,
    plastic: bool,
) -> tuple[
    Float[Array, "members"],
    Float[Array, "members"],
    Float[Array, "members"],
    Float[Array, "members"],
]:
    """
    All four interaction factors for a hollow section.

    Returns
    -------
    factors :
        The factors coupling each axis to each moment, in the order major to
        major, major to minor, minor to major, minor to minor.

    Notes
    -----
    EN 1993-1-1 Annex B, Table B.1. Separate from the equations that consume
    them, which are EN 1993-1-1 6.3.3 and a different clause: a source that
    publishes its own factors can be checked against `checks` alone.
    """
    ratio_y = axial_ratio(n_ed, chi_y, n_rk, gamma_m1)
    ratio_z = axial_ratio(n_ed, chi_z, n_rk, gamma_m1)

    diagonal_y = k_yy(c_my, lam_y, ratio_y, plastic=plastic)
    diagonal_z = k_zz(c_mz, lam_z, ratio_z, plastic=plastic)

    return (
        diagonal_y,
        k_yz(diagonal_z, plastic=plastic),
        k_zy(diagonal_y, plastic=plastic),
        diagonal_z,
    )


def checks(
    n_ed: Float[Array, "members"],
    m_y_ed: Float[Array, "members"],
    m_z_ed: Float[Array, "members"],
    chi_y: Float[Array, "members"],
    chi_z: Float[Array, "members"],
    n_rk: Float[Array, "members"],
    m_rk: Float[Array, "members"],
    factor_yy: Float[Array, "members"],
    factor_yz: Float[Array, "members"],
    factor_zy: Float[Array, "members"],
    factor_zz: Float[Array, "members"],
    gamma_m1: float | Float[Array, ""] = GAMMA_M1,
) -> tuple[Float[Array, "members"], Float[Array, "members"]]:
    """
    Both interaction equations, from interaction factors given directly.

    Parameters
    ----------
    n_ed :
        Design axial compression.
    m_y_ed :
        Design bending moment about the major axis.
    m_z_ed :
        Design bending moment about the minor axis.
    chi_y :
        Reduction factor for flexural buckling about the major axis.
    chi_z :
        Reduction factor for flexural buckling about the minor axis.
    n_rk :
        Characteristic resistance to axial force.
    m_rk :
        Characteristic bending resistance.
    factor_yy :
        Interaction factor coupling major-axis buckling to major-axis bending.
    factor_yz :
        Interaction factor coupling major-axis buckling to minor-axis bending.
    factor_zy :
        Interaction factor coupling minor-axis buckling to major-axis bending.
    factor_zz :
        Interaction factor coupling minor-axis buckling to minor-axis bending.
    gamma_m1 :
        Partial factor for member instability.

    Returns
    -------
    checks :
        Eq. 6.61 and Eq. 6.62, in that order.

    Notes
    -----
    EN 1993-1-1 6.3.3. The reduction factor for lateral-torsional buckling is
    one throughout, so it does not appear: a circular hollow section is closed
    and doubly symmetric and so is not susceptible.

    Taking the factors as arguments rather than deriving them keeps this clause
    separable from Annex B, so a published check that supplies its own factors
    can be reproduced without also adopting the table they came from.
    """
    ratio_y = axial_ratio(n_ed, chi_y, n_rk, gamma_m1)
    ratio_z = axial_ratio(n_ed, chi_z, n_rk, gamma_m1)

    bending = m_rk / gamma_m1
    major = jnp.asarray(m_y_ed) / bending
    minor = jnp.asarray(m_z_ed) / bending

    first = ratio_y + factor_yy * major + factor_yz * minor
    second = ratio_z + factor_zy * major + factor_zz * minor

    return first, second


def _checks(
    n_ed: Float[Array, "members"],
    m_y_ed: Float[Array, "members"],
    m_z_ed: Float[Array, "members"],
    chi_y: Float[Array, "members"],
    chi_z: Float[Array, "members"],
    n_rk: Float[Array, "members"],
    m_rk: Float[Array, "members"],
    lam_y: Float[Array, "members"],
    lam_z: Float[Array, "members"],
    c_my: Float[Array, "members"],
    c_mz: Float[Array, "members"],
    gamma_m1: float | Float[Array, ""] = GAMMA_M1,
    *,
    plastic: bool,
) -> tuple[Float[Array, "members"], Float[Array, "members"]]:
    """
    Both interaction equations, deriving the factors from Table B.1.

    Returns
    -------
    checks :
        Eq. 6.61 and Eq. 6.62, in that order.

    Notes
    -----
    Shared by the utilization and by the diagnostic that reports which equation
    governs, so the two cannot drift apart.
    """
    factors = interaction_factors(
        n_ed,
        chi_y,
        chi_z,
        n_rk,
        lam_y,
        lam_z,
        c_my,
        c_mz,
        gamma_m1,
        plastic=plastic,
    )

    return checks(n_ed, m_y_ed, m_z_ed, chi_y, chi_z, n_rk, m_rk, *factors, gamma_m1)


def governing_equation(
    n_ed: Float[Array, "members"],
    m_y_ed: Float[Array, "members"],
    m_z_ed: Float[Array, "members"],
    chi_y: Float[Array, "members"],
    chi_z: Float[Array, "members"],
    n_rk: Float[Array, "members"],
    m_rk: Float[Array, "members"],
    lam_y: Float[Array, "members"],
    lam_z: Float[Array, "members"],
    c_my: Float[Array, "members"],
    c_mz: Float[Array, "members"],
    gamma_m1: float | Float[Array, ""] = GAMMA_M1,
    *,
    plastic: bool,
) -> Float[Array, "members"]:
    """
    Which of the two interaction equations governs.

    Returns
    -------
    governing :
        Zero where Eq. 6.61 governs, one where Eq. 6.62 does.

    Notes
    -----
    EN 1993-1-1 6.3.3. **Non-differentiable.** A diagnostic, reported beside the
    utilization and never differentiated through. Repeated flips between
    optimizer steps mean the design is chattering across the boundary where the
    two equations cross, which is where the two moments are equal.
    """
    first, second = _checks(
        n_ed,
        m_y_ed,
        m_z_ed,
        chi_y,
        chi_z,
        n_rk,
        m_rk,
        lam_y,
        lam_z,
        c_my,
        c_mz,
        gamma_m1,
        plastic=plastic,
    )

    return jnp.where(second > first, GOVERNING_MINOR, GOVERNING_MAJOR)


def cap_is_active(
    c_m: Float[Array, "members"],
    lam: Float[Array, "members"],
    ratio: Float[Array, "members"],
    *,
    plastic: bool,
) -> Float[Array, "members"]:
    """
    Whether the bound on an interaction factor is the binding one.

    Parameters
    ----------
    c_m :
        Equivalent uniform moment factor.
    lam :
        Non-dimensional slenderness about the relevant axis.
    ratio :
        Axial force over the buckling resistance about that axis.
    plastic :
        Whether the section is Class 1 or 2. Static, never a traced value.

    Returns
    -------
    active :
        One where the bound governs, zero where it does not.

    Notes
    -----
    EN 1993-1-1 Annex B, Table B.1. **Non-differentiable**, and a kink in the
    slope of the utilization, the same kind as the cap on the buckling
    reduction factor. The bound binds once the slenderness passes the
    coefficient it is compared against, which is 0.8 for a plastic section and
    where 0.6 times the slenderness exceeds 0.6 for an elastic one.
    """
    slenderness = jnp.asarray(lam)
    slope = slenderness - 0.2 if plastic else 0.6 * slenderness
    cap_slope = 0.8 if plastic else 0.6

    unbounded = c_m * (1.0 + slope * ratio)
    bound = c_m * (1.0 + cap_slope * ratio)

    return jnp.where(bound < unbounded, 1.0, 0.0)
