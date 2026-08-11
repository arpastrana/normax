import math

import pytest

from normax.ec3.material import IMPERFECTION_FACTORS
from normax.ec3.material import SteelGrade
from normax.ec3.resistance import buckling_auxiliary
from normax.ec3.resistance import force_critical
from normax.ec3.resistance import reduction_buckling
from normax.ec3.resistance import resistance_buckling
from normax.ec3.resistance import resistance_compression
from normax.ec3.resistance import slenderness_from_force

# Gardner, L. and Nethercot, D. (2011), Designers' Guide to Eurocode 3, 2nd edn,
# ICE Publishing. Worked Examples 6.9 (pp. 78-85), 6.10 (pp. 86-93), 6.8
# (pp. 71-76) and 13.3 (pp. 146-148).
#
# These exist to put more points on the buckling curve. The CHS fixture pins
# exactly one, at a slenderness of 0.63 on curve a; on its own that constrains
# almost nothing, since the property tests fix the shape of the curve but no
# absolute value on it. Together these span curves a, b and c over a
# slenderness range of 0.23 to 1.42.
#
# None of these members is a CHS. That costs nothing: the clause layer takes a
# gross area and a second moment of area, so the section properties tabulated
# in each example feed straight in.
#
# Only the axial buckling chain is exercised. Each example goes on to check
# bending, shear or a bending-plus-axial interaction, none of which this
# package implements.
#
# The guide prints the buckling intermediates to 2 significant figures, so its
# column is asserted at 1%. The closed-form column beside it was recomputed
# independently and is asserted at 0.5%.

TOLERANCE_EXACT = 5e-3
TOLERANCE_GUIDE = 1e-2

MODULUS = 210000.0
GAMMA_M0 = 1.0
GAMMA_M1 = 1.0

CURVE_A = IMPERFECTION_FACTORS["a"]
CURVE_B = IMPERFECTION_FACTORS["b"]
CURVE_C = IMPERFECTION_FACTORS["c"]


def buckling_chain(area, second_moment, length_buckling, f_y, alpha):
    critical = force_critical(second_moment, length_buckling, SteelGrade(e_mod=MODULUS))
    non_dimensional = slenderness_from_force(area, SteelGrade(f_y=f_y), critical)
    reduction = reduction_buckling(non_dimensional, alpha)

    return {
        "n_cr": critical,
        "slenderness": non_dimensional,
        "phi": buckling_auxiliary(non_dimensional, alpha),
        "chi": reduction,
        "n_b_rd": resistance_buckling(
            reduction, area, SteelGrade(f_y=f_y, gamma_m1=GAMMA_M1)
        ),
    }


# ---- Example 6.9: hot-finished 200 x 100 x 16 RHS, S355, curve a ---- #
#
# Primary floor beam, 7.2 m span. Buckling length is the full span about the
# major axis and 2.4 m about the minor, set by the secondary beams.

RHS_AREA = 8300.0
RHS_INERTIA_Y = 36780000.0
RHS_INERTIA_Z = 11470000.0
RHS_YIELD = 355.0
RHS_LENGTH_Y = 7200.0
RHS_LENGTH_Z = 2400.0

RHS_GUIDE = {
    "y": {
        "n_cr": 1470e3,
        "slenderness": 1.42,
        "phi": 1.63,
        "chi": 0.41,
        "n_b_rd": 1209e3,
    },
    "z": {
        "n_cr": 4127e3,
        "slenderness": 0.84,
        "phi": 0.92,
        "chi": 0.77,
        "n_b_rd": 2266e3,
    },
}

RHS_EXACT = {
    "y": {
        "n_cr": 1470502.516843,
        "slenderness": 1.415534,
        "phi": 1.629499,
        "chi": 0.410395,
        "n_b_rd": 1209229.965341,
    },
    "z": {
        "n_cr": 4127242.382101,
        "slenderness": 0.844935,
        "phi": 0.924676,
        "chi": 0.769040,
        "n_b_rd": 2265977.451366,
    },
}


@pytest.fixture(scope="module")
def rhs():
    return {
        "y": buckling_chain(RHS_AREA, RHS_INERTIA_Y, RHS_LENGTH_Y, RHS_YIELD, CURVE_A),
        "z": buckling_chain(RHS_AREA, RHS_INERTIA_Z, RHS_LENGTH_Z, RHS_YIELD, CURVE_A),
    }


@pytest.mark.parametrize("quantity", sorted(RHS_EXACT["y"]))
@pytest.mark.parametrize("axis", ["y", "z"])
def test_rhs_matches_the_closed_form(rhs, axis, quantity):
    assert rhs[axis][quantity] == pytest.approx(
        RHS_EXACT[axis][quantity], rel=TOLERANCE_EXACT
    )


@pytest.mark.parametrize("quantity", sorted(RHS_GUIDE["y"]))
@pytest.mark.parametrize("axis", ["y", "z"])
def test_rhs_matches_the_guide(rhs, axis, quantity):
    assert rhs[axis][quantity] == pytest.approx(
        RHS_GUIDE[axis][quantity], rel=TOLERANCE_GUIDE
    )


def test_rhs_cross_section_resistance():
    resistance = resistance_compression(
        RHS_AREA, SteelGrade(f_y=RHS_YIELD, gamma_m0=GAMMA_M0)
    )

    assert resistance * 1e-3 == pytest.approx(2946.5, rel=TOLERANCE_EXACT)


def test_rhs_major_axis_governs(rhs):
    # The longer buckling length about the major axis wins despite the larger
    # second moment of area.
    assert rhs["y"]["n_b_rd"] < rhs["z"]["n_b_rd"]


# ---- Example 6.10: 305 x 305 x 240 H-section, S275, curves b and c ---- #
#
# Ground-floor column, 4.2 m. Buckling lengths 0.7L in plane and 1.0L out of
# plane. A stocky column: the major-axis reduction factor is close to one, so
# this fixture sits near the cap where the CHS example does not.
#
# f_y is 265, not 275: the flange is 37.7 mm thick, which falls in the
# 16 to 40 mm band of EN 10025-2.

UKC_AREA = 30600.0
UKC_INERTIA_Y = 642.0e6
UKC_INERTIA_Z = 203.1e6
UKC_YIELD = 265.0
UKC_LENGTH_Y = 2940.0
UKC_LENGTH_Z = 4200.0

UKC_GUIDE = {
    "y": {
        "n_cr": 153943e3,
        "slenderness": 0.23,
        "phi": 0.53,
        "chi": 0.99,
        "n_b_rd": 8024e3,
    },
    "z": {
        "n_cr": 23863e3,
        "slenderness": 0.58,
        "phi": 0.76,
        "chi": 0.80,
        "n_b_rd": 6450e3,
    },
}

UKC_EXACT = {
    "y": {
        "n_cr": 153942809.171510,
        "slenderness": 0.229511,
        "phi": 0.531355,
        "chi": 0.989525,
        "n_b_rd": 8024060.677915,
    },
    "z": {
        "n_cr": 23863293.498348,
        "slenderness": 0.582933,
        "phi": 0.763724,
        "chi": 0.795454,
        "n_b_rd": 6450334.989577,
    },
}


@pytest.fixture(scope="module")
def ukc():
    return {
        "y": buckling_chain(UKC_AREA, UKC_INERTIA_Y, UKC_LENGTH_Y, UKC_YIELD, CURVE_B),
        "z": buckling_chain(UKC_AREA, UKC_INERTIA_Z, UKC_LENGTH_Z, UKC_YIELD, CURVE_C),
    }


@pytest.mark.parametrize("quantity", sorted(UKC_EXACT["y"]))
@pytest.mark.parametrize("axis", ["y", "z"])
def test_ukc_matches_the_closed_form(ukc, axis, quantity):
    assert ukc[axis][quantity] == pytest.approx(
        UKC_EXACT[axis][quantity], rel=TOLERANCE_EXACT
    )


@pytest.mark.parametrize("quantity", sorted(UKC_GUIDE["y"]))
@pytest.mark.parametrize("axis", ["y", "z"])
def test_ukc_matches_the_guide(ukc, axis, quantity):
    assert ukc[axis][quantity] == pytest.approx(
        UKC_GUIDE[axis][quantity], rel=TOLERANCE_GUIDE
    )


def test_ukc_cross_section_resistance():
    # The guide later writes this as 8415 kN, which is the same area at 275
    # rather than 265. See the errata in docs/clauses.md.
    resistance = resistance_compression(
        UKC_AREA, SteelGrade(f_y=UKC_YIELD, gamma_m0=GAMMA_M0)
    )

    assert resistance * 1e-3 == pytest.approx(8109.0, rel=TOLERANCE_EXACT)


def test_ukc_uses_two_different_curves(ukc):
    # Table 6.2 sends a rolled H-section with h/b <= 1.2 to curve b about the
    # major axis and curve c about the minor. Different curves at similar
    # slenderness, which no other fixture here covers.
    assert CURVE_B < CURVE_C
    assert ukc["y"]["chi"] > ukc["z"]["chi"]


def test_ukc_major_axis_is_almost_unreduced(ukc):
    assert ukc["y"]["chi"] == pytest.approx(0.99, rel=TOLERANCE_GUIDE)
    assert ukc["y"]["chi"] < 1.0


# ---- Example 13.3: 100 x 50 x 3 plain channel, curve c ---- #
#
# A cold-formed member from the EN 1993-1-3 chapter, pinned over 1.5 m. Its
# critical force comes from torsional-flexural buckling, which this package
# does not implement, so that value is taken from the guide and only the
# chain below it is checked. The two flexural critical forces are computed.
#
# The slenderness of 1.16 is the most slender point in the whole set.

CHANNEL_AREA_EFFECTIVE = 549.0
CHANNEL_INERTIA_Y = 85.41e4
CHANNEL_INERTIA_Z = 13.76e4
CHANNEL_YIELD = 280.0
CHANNEL_LENGTH = 1500.0
CHANNEL_N_CR_TORSIONAL_FLEXURAL = 114e3


def test_channel_flexural_critical_forces():
    critical_y = force_critical(
        CHANNEL_INERTIA_Y, CHANNEL_LENGTH, SteelGrade(e_mod=MODULUS)
    )
    critical_z = force_critical(
        CHANNEL_INERTIA_Z, CHANNEL_LENGTH, SteelGrade(e_mod=MODULUS)
    )

    assert critical_y * 1e-3 == pytest.approx(787.0, rel=TOLERANCE_GUIDE)
    assert critical_z * 1e-3 == pytest.approx(127.0, rel=TOLERANCE_GUIDE)


def test_channel_buckling_chain():
    non_dimensional = slenderness_from_force(
        CHANNEL_AREA_EFFECTIVE,
        SteelGrade(f_y=CHANNEL_YIELD),
        CHANNEL_N_CR_TORSIONAL_FLEXURAL,
    )
    auxiliary = buckling_auxiliary(non_dimensional, CURVE_C)
    reduction = reduction_buckling(non_dimensional, CURVE_C)

    assert non_dimensional == pytest.approx(1.161215, rel=TOLERANCE_EXACT)
    assert auxiliary == pytest.approx(1.409708, rel=TOLERANCE_EXACT)
    assert reduction == pytest.approx(0.452695, rel=TOLERANCE_EXACT)

    assert non_dimensional == pytest.approx(1.16, rel=TOLERANCE_GUIDE)
    assert auxiliary == pytest.approx(1.41, rel=TOLERANCE_GUIDE)
    assert reduction == pytest.approx(0.45, rel=TOLERANCE_GUIDE)


def test_channel_buckling_resistance():
    # The guide prints 69.2 kN, which it reaches by carrying its own rounded
    # reduction factor of 0.45 forward. The unrounded factor gives 69.6 kN.
    reduction = reduction_buckling(1.161215, CURVE_C)
    resistance = resistance_buckling(
        reduction,
        CHANNEL_AREA_EFFECTIVE,
        SteelGrade(f_y=CHANNEL_YIELD, gamma_m1=GAMMA_M1),
    )

    assert resistance * 1e-3 == pytest.approx(69.2, rel=TOLERANCE_GUIDE)
    assert 0.45 * CHANNEL_AREA_EFFECTIVE * CHANNEL_YIELD * 1e-3 == pytest.approx(
        69.2, rel=1e-3
    )


# ---- Example 6.8: the lateral-torsional curve, as an algebra check ---- #
#
# NOT an implementation of 6.3.2. This package has no lateral-torsional
# buckling and needs none: a circular hollow section is closed and doubly
# symmetric, so its reduction factor is one.
#
# Eq. 6.56 of the general case is nevertheless the same function of
# slenderness and imperfection factor as Eq. 6.49, so this example's printed
# values are two more points on curve b at a mid-range slenderness that the
# flexural examples do not reach. The slenderness is fed in as printed,
# because deriving it needs an elastic critical moment this package does not
# compute.

LATERAL_TORSIONAL = [(0.54, 0.70, 0.87), (0.62, 0.76, 0.83)]


@pytest.mark.parametrize("lam, expected_phi, expected_chi", LATERAL_TORSIONAL)
def test_lateral_torsional_curve_shares_the_flexural_algebra(
    lam, expected_phi, expected_chi
):
    assert buckling_auxiliary(lam, CURVE_B) == pytest.approx(
        expected_phi, rel=TOLERANCE_GUIDE
    )
    assert reduction_buckling(lam, CURVE_B) == pytest.approx(
        expected_chi, rel=TOLERANCE_GUIDE
    )


# ---- Coverage of the curve, taken as a whole ---- #

PUBLISHED_POINTS = [
    ("6.7 CHS", CURVE_A, 0.630844, 0.877915),
    ("6.9 RHS major", CURVE_A, 1.415534, 0.410395),
    ("6.9 RHS minor", CURVE_A, 0.844935, 0.769040),
    ("6.10 UKC major", CURVE_B, 0.229511, 0.989525),
    ("6.10 UKC minor", CURVE_C, 0.582933, 0.795454),
    ("13.3 channel", CURVE_C, 1.161215, 0.452695),
    ("6.8 LTB segment BC", CURVE_B, 0.54, 0.866058),
    ("6.8 LTB segment CD", CURVE_B, 0.62, 0.826897),
]


@pytest.mark.parametrize("label, alpha, lam, expected", PUBLISHED_POINTS)
def test_every_published_point(label, alpha, lam, expected):
    assert reduction_buckling(lam, alpha) == pytest.approx(
        expected, rel=TOLERANCE_EXACT
    ), label


def test_the_published_points_span_the_curve():
    slendernesses = [point[2] for point in PUBLISHED_POINTS]
    curves = {point[1] for point in PUBLISHED_POINTS}

    assert min(slendernesses) < 0.25
    assert max(slendernesses) > 1.4
    assert curves == {CURVE_A, CURVE_B, CURVE_C}


def test_published_points_agree_with_a_longhand_evaluation():
    # Independent of the package: Eq. 6.49 written out with the standard
    # library, as a guard against a shared-helper mistake.
    for label, alpha, lam, expected in PUBLISHED_POINTS:
        auxiliary = 0.5 * (1.0 + alpha * (lam - 0.2) + lam**2)
        longhand = 1.0 / (auxiliary + math.sqrt(auxiliary**2 - lam**2))

        assert longhand == pytest.approx(expected, rel=TOLERANCE_EXACT), label
