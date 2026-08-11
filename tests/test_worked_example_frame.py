import jax.numpy as jnp
import numpy as np
import pytest

from normax.ec3.interaction import CompressionBendingState
from normax.ec3.interaction import InteractionFactors
from normax.ec3.interaction import MemberResistance
from normax.ec3.interaction import MemberSlenderness
from normax.ec3.interaction import interaction_checks
from normax.ec3.interaction import interaction_factors
from normax.ec3.interaction import utilization_member
from normax.ec3.material import SteelGrade

# Simões da Silva, Simões & Gervásio, *Design of Steel Structures* (ECCS).
# Design Example 2, a 47 m single-span pitched-roof portal frame in S355J2,
# HEA 550 columns and IPE 600 rafters, pp. 407-430; and Example 5.2, the same
# rafter checked segment by segment, pp. 397-404.
#
# The member forces below are the book's own, taken as inputs. This is a parity
# check on the member check alone: we are not analysing the frame. T1 and T2 do
# not exist yet, so nothing here computes a force from a geometry -- the point
# is only to confirm that, given the same forces and the same resistances, our
# assembly of Eqs. 6.61 and 6.62 reproduces the published numbers.
#
# Two consequences of that scope, both deliberate:
#
#   - The frame is PLANAR, so the minor-axis moment is zero at every section.
#     Only the uniaxial path through 6.61/6.62 is exercised. The biaxial terms
#     are covered by the property tests in test_interaction.py.
#   - The sections are IPE 600 and HEA 550, not circular. The book supplies its
#     own interaction factors, so they are passed to `checks` directly rather
#     than derived from Table B.1. That is why this validates the equations
#     without also asserting the hollow-section row we read for a CHS.

# The book prints these utilizations to two decimal places, so the honest
# tolerance is the rounding half-width, not a relative one. A relative tolerance
# is the wrong instrument here: our 0.4653 against a printed 0.47 is correct
# rounding, yet it is 1.01% away and would fail a 1% band.
TOLERANCE = 5e-3

# IPE 600, S355, from Table 5.18.
N_PL_RD = 5538.0e3
M_PL_Y_RD = 1246.8e6

# Section 5.4.6.3. Conservatively prismatic, so one pair of reduction factors
# and one pair of interaction factors serve the whole rafter.
CHI_Y = 0.48
CHI_Z = 1.00
FACTOR_YY = 1.03
FACTOR_ZY = 0.62

# Combination 3. Section, axial compression, major-axis moment, and the two
# published results for Eqs. 6.61 and 6.62.
RAFTER = [
    ("C'", 480.3e3, 344.5e6, 0.47, 0.26),
    ("B''", 471.5e3, 700.2e6, 0.76, 0.43),
]


def rafter_checks(axial_force, moment_major):
    return interaction_checks(
        CompressionBendingState(axial_force, moment_major, 0.0),
        MemberResistance(CHI_Y, CHI_Z, N_PL_RD, M_PL_Y_RD),
        InteractionFactors(yy=FACTOR_YY, yz=0.0, zy=FACTOR_ZY, zz=0.0),
        SteelGrade(),
    )


@pytest.mark.parametrize("label, axial_force, moment_major, first, second", RAFTER)
def test_rafter_first_equation_matches_the_book(
    label, axial_force, moment_major, first, second
):
    assert rafter_checks(axial_force, moment_major)[0] == pytest.approx(
        first, abs=TOLERANCE
    ), label


@pytest.mark.parametrize("label, axial_force, moment_major, first, second", RAFTER)
def test_rafter_second_equation_matches_the_book(
    label, axial_force, moment_major, first, second
):
    assert rafter_checks(axial_force, moment_major)[1] == pytest.approx(
        second, abs=TOLERANCE
    ), label


@pytest.mark.parametrize("label, axial_force, moment_major, first, second", RAFTER)
def test_rafter_utilization_takes_the_worse_equation(
    label, axial_force, moment_major, first, second
):
    value = max(rafter_checks(axial_force, moment_major))

    assert value == pytest.approx(max(first, second), abs=TOLERANCE), label


def test_the_first_equation_governs_along_the_whole_rafter():
    # The major axis has the longer buckling length here, so it governs at
    # every section the book checks.
    for _, axial_force, moment_major, _, _ in RAFTER:
        first, second = rafter_checks(axial_force, moment_major)

        assert first > second


def test_the_rafter_is_adequate_as_the_book_concludes():
    for _, axial_force, moment_major, _, _ in RAFTER:
        assert max(rafter_checks(axial_force, moment_major)) < 1.0


def test_the_whole_rafter_evaluates_in_one_call():
    # How the pipeline will call it: one entry per member, not a Python loop.
    forces = jnp.asarray([row[1] for row in RAFTER])
    moments = jnp.asarray([row[2] for row in RAFTER])

    first, second = rafter_checks(forces, moments)

    assert first.shape == (len(RAFTER),)
    assert np.asarray(first) == pytest.approx([row[3] for row in RAFTER], abs=TOLERANCE)
    assert np.asarray(second) == pytest.approx(
        [row[4] for row in RAFTER], abs=TOLERANCE
    )


# ---- Example 5.2, the same rafter checked segment by segment ---- #
#
# Written against buckling resistances rather than characteristic ones, which
# is Eq. 6.62 with the reduction factors already folded in. Passing a reduction
# factor of one and the buckling resistances as the characteristic values
# reproduces it exactly.

SEGMENTS = [
    ("B3X", 632.93e3, 773.29e6, 2378.40e3, 976.96e6, 0.92, 0.99),
    ("XC", 610.93e3, 561.88e6, 4584.99e3, 1092.11e6, 0.99, 0.64),
]


@pytest.mark.parametrize(
    "label, axial_force, moment_major, n_b_rd, m_b_rd, factor_zy, expected",
    SEGMENTS,
)
def test_segment_matches_the_book(
    label, axial_force, moment_major, n_b_rd, m_b_rd, factor_zy, expected
):
    # The book prints 0.62 for segment XC; recomputing from its own inputs
    # gives 0.64. See the errata section of docs/clauses.md -- this asserts the
    # corrected value, so a fixture built on the printed one would fail here.
    _, second = interaction_checks(
        CompressionBendingState(axial_force, moment_major, 0.0),
        MemberResistance(1.0, 1.0, n_b_rd, m_b_rd),
        InteractionFactors(yy=0.0, yz=0.0, zy=factor_zy, zz=0.0),
        SteelGrade(),
    )

    assert second == pytest.approx(expected, abs=TOLERANCE), label


def test_the_governing_segment_is_the_one_the_book_identifies():
    # B3X reaches 0.99 and is the critical segment of the rafter.
    values = [
        float(
            interaction_checks(
                CompressionBendingState(n, m, 0.0),
                MemberResistance(1.0, 1.0, n_b, m_b),
                InteractionFactors(yy=0.0, yz=0.0, zy=k, zz=0.0),
                SteelGrade(),
            )[1]
        )
        for _, n, m, n_b, m_b, k, _ in SEGMENTS
    ]

    assert values[0] > values[1]
    assert max(values) < 1.0


# ---- The equations agree with the full path when the factors do ---- #


def test_supplying_the_factors_agrees_with_deriving_them():
    # `checks` and `utilization` must not drift apart: feeding the factors that
    # Table B.1 would produce has to give what the full path gives.
    state = CompressionBendingState(480.3e3, 344.5e6, 0.0, 0.9, 0.9)
    resistance = MemberResistance(0.48, 1.00, N_PL_RD, M_PL_Y_RD)

    slenderness = MemberSlenderness(0.9, 0.4)

    factors = interaction_factors(
        state, resistance, slenderness, SteelGrade(), section_class=2
    )

    supplied = interaction_checks(state, resistance, factors, SteelGrade())
    derived = utilization_member(
        state, resistance, slenderness, SteelGrade(), section_class=2
    )

    assert max(supplied) == pytest.approx(derived)
