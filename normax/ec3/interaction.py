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

Which column of Table B.1 applies follows from the cross-section class, which is
fixed by the configured diameter-to-thickness ratio. It is therefore a static
choice, never a branch on a traced value.

The class travels as itself all the way to the rows, each of which reads the
column it belongs to. Nothing carries a boolean standing in for a class, so no
call site can name one that its section family does not have.
"""

from typing import NamedTuple

import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float

from normax.ec3.actions import MemberActions
from normax.ec3.classification import is_plastic
from normax.ec3.material import Steel

# EN 1993-1-1 Table B.3, the lower bound on the linear row.
C_M_MINIMUM = 0.4

# Which of the two interaction equations governs, reported alongside the
# utilization as a non-differentiable diagnostic.
GOVERNING_MAJOR = 0.0
GOVERNING_MINOR = 1.0


class CompressionBendingState(NamedTuple):
    """
    What 6.3.3 reads: a compression and two moment magnitudes.

    Attributes
    ----------
    axial_force :
        Design axial compression, non-negative.
    moment_major :
        Magnitude of the design bending moment about the major axis.
    moment_minor :
        Magnitude of the design bending moment about the minor axis.
    moment_factor_major :
        Equivalent uniform moment factor for major-axis bending.
    moment_factor_minor :
        Equivalent uniform moment factor for minor-axis bending.

    Notes
    -----
    **A different type from `MemberActions` on purpose, though the fields line
    up.** 6.3.3 is titled "bending and axial compression" and reads a
    compression, so a tension-positive axial force reaching it yields a negative
    axial ratio, which *subtracts* from Eqs. 6.61 and 6.62 and reports a member
    as safer than it is. Two types with one named constructor between them turn
    that from a silent wrong answer into a checker error.
    """

    axial_force: Float[Array, "members"]
    moment_major: float | Float[Array, "members"] = 0.0
    moment_minor: float | Float[Array, "members"] = 0.0
    moment_factor_major: float | Float[Array, "members"] = 1.0
    moment_factor_minor: float | Float[Array, "members"] = 1.0

    @classmethod
    def from_actions(cls, actions: MemberActions) -> "CompressionBendingState":
        """
        The state 6.3.3 reads, from the actions an analysis produced.

        Parameters
        ----------
        actions :
            Design actions on the member, tension positive.

        Returns
        -------
        state :
            The compression and the two moment magnitudes.

        Notes
        -----
        A member in tension maps to zero compression rather than to a negative
        one, which switches every term of 6.3.3 off instead of reversing its
        sign. The clause does not apply there at all, so a caller must still
        discard the result rather than read a zero as adequacy.
        """
        return cls(
            jnp.maximum(-jnp.asarray(actions.axial_force), 0.0),
            jnp.abs(actions.moment_major),
            jnp.abs(actions.moment_minor),
            actions.moment_factor_major,
            actions.moment_factor_minor,
        )


class MemberResistance(NamedTuple):
    """
    What a member resists, about each axis, before partial factors.

    Attributes
    ----------
    chi_y :
        Reduction factor for flexural buckling about the major axis.
    chi_z :
        Reduction factor for flexural buckling about the minor axis.
    n_rk :
        Characteristic resistance to axial force.
    m_rk :
        Characteristic bending resistance, the same about both axes for a
        circular hollow section.

    Notes
    -----
    The two slendernesses are deliberately absent, carried by
    `MemberSlenderness` instead. 6.3.3 does not read them — only Annex B does,
    to build the interaction factors — and that separability is what lets a
    published check supplying its own factors be reproduced without inventing
    slendernesses for a clause that ignores them.
    """

    chi_y: Float[Array, "members"]
    chi_z: Float[Array, "members"]
    n_rk: Float[Array, "members"]
    m_rk: Float[Array, "members"]


class MemberSlenderness(NamedTuple):
    """
    Non-dimensional slenderness about each buckling axis.

    Attributes
    ----------
    lam_y :
        Non-dimensional slenderness about the major axis.
    lam_z :
        Non-dimensional slenderness about the minor axis.

    Notes
    -----
    Read by Annex B alone, which is why it is a type of its own rather than two
    more fields on the resistance: the equations of 6.3.3 never see it.

    A doubly symmetric closed section buckles alike about both axes, so one
    value serves for both. `about_both_axes` is the constructor for that case,
    so a caller holding a single slenderness never has to write it twice and no
    call site can pass the major-axis value where the minor-axis one belongs.
    """

    lam_y: Float[Array, "members"]
    lam_z: Float[Array, "members"]

    @classmethod
    def about_both_axes(
        cls,
        lam: Float[Array, "members"],
    ) -> "MemberSlenderness":
        """
        One slenderness taken about both axes.

        Parameters
        ----------
        lam :
            Non-dimensional slenderness, the same about every centroidal axis.

        Returns
        -------
        slenderness :
            The same value about the major and the minor axis.

        Notes
        -----
        A circular hollow section has the same second moment about every
        centroidal axis, so its two flexural buckling modes are identical.
        """
        return cls(lam, lam)


class InteractionFactors(NamedTuple):
    """
    The four factors coupling each buckling axis to each bending moment.

    Attributes
    ----------
    yy :
        Major-axis buckling against major-axis bending.
    yz :
        Major-axis buckling against minor-axis bending.
    zy :
        Minor-axis buckling against major-axis bending.
    zz :
        Minor-axis buckling against minor-axis bending.

    Notes
    -----
    EN 1993-1-1 Annex B, Table B.1. Named rather than positional because the two
    off-diagonal factors differ, and swapping them produces a plausible number
    that no check can refuse.
    """

    yy: Float[Array, "members"]
    yz: Float[Array, "members"]
    zy: Float[Array, "members"]
    zz: Float[Array, "members"]


def moment_factor_linear(psi: Float[Array, "members"]) -> Float[Array, "members"]:
    """
    Equivalent uniform moment factor for a linear moment diagram.

    Parameters
    ----------
    psi :
        Ratio of the smaller to the larger end moment, signed, so that a
        uniform moment gives one and a symmetric reversal gives minus one.

    Returns
    -------
    moment_factor_linear :
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
    axial_force: Float[Array, "members"],
    chi: Float[Array, "members"],
    n_rk: Float[Array, "members"],
    steel: Steel,
) -> Float[Array, "members"]:
    """
    Axial force over the buckling resistance, as the interaction factors use it.

    Parameters
    ----------
    axial_force :
        Design axial compression.
    chi :
        Reduction factor for flexural buckling about the relevant axis.
    n_rk :
        Characteristic resistance to axial force.
    steel :
        Material properties and partial factors, read for gamma_M1.

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
    force = jnp.asarray(axial_force)

    return force / (chi * n_rk / steel.gamma_m1)


def _k_axial(
    moment_factor: Float[Array, "members"],
    slope: Float[Array, "members"],
    cap_slope: float,
    ratio: Float[Array, "members"],
) -> Float[Array, "members"]:
    """
    Shared shape of the diagonal interaction factors.

    Parameters
    ----------
    moment_factor_linear :
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
    unbounded = moment_factor * (1.0 + slope * axial)
    bound = moment_factor * (1.0 + cap_slope * axial)

    return jnp.minimum(unbounded, bound)


def k_yy(
    moment_factor_major: Float[Array, "members"],
    lam_y: Float[Array, "members"],
    ratio_y: Float[Array, "members"],
    *,
    section_class: int,
) -> Float[Array, "members"]:
    """
    Interaction factor coupling axial force to major-axis bending.

    Parameters
    ----------
    moment_factor_major :
        Equivalent uniform moment factor for major-axis bending.
    lam_y :
        Non-dimensional slenderness about the major axis.
    ratio_y :
        Axial force over the major-axis buckling resistance.
    section_class :
        Cross-section class, 1, 2 or 3. Static, never a traced value.

    Returns
    -------
    k_yy :
        Interaction factor.

    Notes
    -----
    EN 1993-1-1 Annex B, Table B.1, hollow-section row.
    """
    if is_plastic(section_class):
        return _k_axial(moment_factor_major, jnp.asarray(lam_y) - 0.2, 0.8, ratio_y)

    return _k_axial(moment_factor_major, 0.6 * jnp.asarray(lam_y), 0.6, ratio_y)


def k_zz(
    moment_factor_minor: Float[Array, "members"],
    lam_z: Float[Array, "members"],
    ratio_z: Float[Array, "members"],
    *,
    section_class: int,
) -> Float[Array, "members"]:
    """
    Interaction factor coupling axial force to minor-axis bending.

    Parameters
    ----------
    moment_factor_minor :
        Equivalent uniform moment factor for minor-axis bending.
    lam_z :
        Non-dimensional slenderness about the minor axis.
    ratio_z :
        Axial force over the minor-axis buckling resistance.
    section_class :
        Cross-section class, 1, 2 or 3. Static, never a traced value.

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
    if is_plastic(section_class):
        return _k_axial(moment_factor_minor, jnp.asarray(lam_z) - 0.2, 0.8, ratio_z)

    return _k_axial(moment_factor_minor, 0.6 * jnp.asarray(lam_z), 0.6, ratio_z)


def k_yz(
    k_zz_value: Float[Array, "members"],
    *,
    section_class: int,
) -> Float[Array, "members"]:
    """
    Interaction factor coupling major-axis buckling to minor-axis bending.

    Parameters
    ----------
    k_zz_value :
        The minor-axis diagonal interaction factor.
    section_class :
        Cross-section class, 1, 2 or 3. Static, never a traced value.

    Returns
    -------
    k_yz :
        Interaction factor.

    Notes
    -----
    EN 1993-1-1 Annex B, Table B.1. Six tenths of the diagonal factor for a
    plastic section, and the diagonal factor itself for an elastic one.
    """
    if is_plastic(section_class):
        return 0.6 * jnp.asarray(k_zz_value)

    return jnp.asarray(k_zz_value)


def k_zy(
    k_yy_value: Float[Array, "members"],
    *,
    section_class: int,
) -> Float[Array, "members"]:
    """
    Interaction factor coupling minor-axis buckling to major-axis bending.

    Parameters
    ----------
    k_yy_value :
        The major-axis diagonal interaction factor.
    section_class :
        Cross-section class, 1, 2 or 3. Static, never a traced value.

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
    if is_plastic(section_class):
        return 0.6 * jnp.asarray(k_yy_value)

    return 0.8 * jnp.asarray(k_yy_value)


def utilization_member(
    state: CompressionBendingState,
    resistance: MemberResistance,
    slenderness: MemberSlenderness,
    steel: Steel,
    *,
    section_class: int,
) -> Float[Array, "members"]:
    """
    Utilization of a member under axial compression and biaxial bending.

    Parameters
    ----------
    state :
        Compression and moment magnitudes acting on the member.
    resistance :
        What the member resists about each axis, before partial factors.
    slenderness :
        Non-dimensional slenderness about each buckling axis.
    steel :
        Material properties and partial factors, read for gamma_M1.
    section_class :
        Cross-section class, 1, 2 or 3. Static, never a traced value.

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
    first, second = _derived_checks(
        state, resistance, slenderness, steel, section_class=section_class
    )

    return jnp.maximum(first, second)


def interaction_factors(
    state: CompressionBendingState,
    resistance: MemberResistance,
    slenderness: MemberSlenderness,
    steel: Steel,
    *,
    section_class: int,
) -> InteractionFactors:
    """
    All four interaction factors for a hollow section.

    Parameters
    ----------
    state :
        Compression and moment factors acting on the member. Neither moment
        magnitude is read: Table B.1 scales the moments rather than reading them.
    resistance :
        What the member resists. The bending resistance is not read either.
    slenderness :
        Non-dimensional slenderness about each buckling axis.
    steel :
        Material properties and partial factors, read for gamma_M1.
    section_class :
        Cross-section class, 1, 2 or 3. Static, never a traced value.

    Returns
    -------
    factors :
        The factors coupling each axis to each moment.

    Notes
    -----
    EN 1993-1-1 Annex B, Table B.1. Separate from the equations that consume
    them, which are EN 1993-1-1 6.3.3 and a different clause: a source that
    publishes its own factors can be checked against `checks` alone.
    """
    ratio_y = axial_ratio(state.axial_force, resistance.chi_y, resistance.n_rk, steel)
    ratio_z = axial_ratio(state.axial_force, resistance.chi_z, resistance.n_rk, steel)

    diagonal_y = k_yy(
        state.moment_factor_major,
        slenderness.lam_y,
        ratio_y,
        section_class=section_class,
    )
    diagonal_z = k_zz(
        state.moment_factor_minor,
        slenderness.lam_z,
        ratio_z,
        section_class=section_class,
    )

    return InteractionFactors(
        yy=diagonal_y,
        yz=k_yz(diagonal_z, section_class=section_class),
        zy=k_zy(diagonal_y, section_class=section_class),
        zz=diagonal_z,
    )


def interaction_checks(
    state: CompressionBendingState,
    resistance: MemberResistance,
    factors: InteractionFactors,
    steel: Steel,
) -> tuple[Float[Array, "members"], Float[Array, "members"]]:
    """
    Both interaction equations, from interaction factors given directly.

    Parameters
    ----------
    state :
        Compression and moment magnitudes acting on the member. Neither moment
        factor is read: they reach the check only through the factors.
    resistance :
        What the member resists about each axis, before partial factors.
    factors :
        The four interaction factors of Table B.1.
    steel :
        Material properties and partial factors, read for gamma_M1.

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
    ratio_y = axial_ratio(state.axial_force, resistance.chi_y, resistance.n_rk, steel)
    ratio_z = axial_ratio(state.axial_force, resistance.chi_z, resistance.n_rk, steel)

    bending = resistance.m_rk / steel.gamma_m1
    major = jnp.asarray(state.moment_major) / bending
    minor = jnp.asarray(state.moment_minor) / bending

    first = ratio_y + factors.yy * major + factors.yz * minor
    second = ratio_z + factors.zy * major + factors.zz * minor

    return first, second


def _derived_checks(
    state: CompressionBendingState,
    resistance: MemberResistance,
    slenderness: MemberSlenderness,
    steel: Steel,
    *,
    section_class: int,
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
        state, resistance, slenderness, steel, section_class=section_class
    )

    return interaction_checks(state, resistance, factors, steel)


def governing_equation(
    state: CompressionBendingState,
    resistance: MemberResistance,
    slenderness: MemberSlenderness,
    steel: Steel,
    *,
    section_class: int,
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
    first, second = _derived_checks(
        state, resistance, slenderness, steel, section_class=section_class
    )

    return jnp.where(second > first, GOVERNING_MINOR, GOVERNING_MAJOR)


def cap_is_active(
    moment_factor: Float[Array, "members"],
    lam: Float[Array, "members"],
    ratio: Float[Array, "members"],
    *,
    section_class: int,
) -> Float[Array, "members"]:
    """
    Whether the bound on an interaction factor is the binding one.

    Parameters
    ----------
    moment_factor :
        Equivalent uniform moment factor.
    lam :
        Non-dimensional slenderness about the relevant axis.
    ratio :
        Axial force over the buckling resistance about that axis.
    section_class :
        Cross-section class, 1, 2 or 3. Static, never a traced value.

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
    plastic = is_plastic(section_class)
    slope = slenderness - 0.2 if plastic else 0.6 * slenderness
    cap_slope = 0.8 if plastic else 0.6

    unbounded = moment_factor * (1.0 + slope * ratio)
    bound = moment_factor * (1.0 + cap_slope * ratio)

    return jnp.where(bound < unbounded, 1.0, 0.0)
