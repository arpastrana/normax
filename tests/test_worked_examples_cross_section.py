import pytest

from normax.ec3.resistance import resistance_compression
from normax.ec3.resistance import resistance_fracture
from normax.ec3.resistance import resistance_tension
from normax.ec3.resistance import resistance_yielding

# Gardner, L. and Nethercot, D. (2011), Designers' Guide to Eurocode 3, 2nd edn,
# ICE Publishing. Worked Examples 6.1 (tension, p. 38) and 6.2 (compression in
# a cross-section, p. 40).
#
# Neither section is a CHS: 6.1 is a flat bar and 6.2 a 254 x 254 x 73 UKC.
# That is the point. The clause layer takes a gross area, not a diameter, so
# the guide's own section-resistance examples run through it untouched by any
# CHS geometry.

TOLERANCE = 1e-3

NEWTON_TO_KILONEWTON = 1e-3


# ---- Example 6.1: a lap-spliced flat-bar tie in S275 ---- #
#
# 200 x 25 mm bar; at 25 mm thickness EN 10025-2 gives f_y = 265 and f_u = 430.
# Six M20 bolts staggered, so the net area is governed by the staggered path.
# UK NA clause NA.2.15: gamma_M0 = 1.00, gamma_M2 = 1.10.

AREA_GROSS = 5000.0
AREA_NET = 4406.0
YIELD_BAR = 265.0
ULTIMATE_BAR = 430.0
GAMMA_M0_BAR = 1.00
GAMMA_M2_BAR = 1.10


def test_tension_gross_section_yielding():
    resistance = resistance_yielding(AREA_GROSS, YIELD_BAR, GAMMA_M0_BAR)

    assert resistance * NEWTON_TO_KILONEWTON == pytest.approx(1325.0, rel=TOLERANCE)


def test_tension_net_section_fracture():
    resistance = resistance_fracture(AREA_NET, ULTIMATE_BAR, GAMMA_M2_BAR)

    assert resistance * NEWTON_TO_KILONEWTON == pytest.approx(1550.0, rel=TOLERANCE)


def test_tension_resistance_is_the_smaller_of_the_two():
    resistance = resistance_tension(
        AREA_GROSS,
        AREA_NET,
        YIELD_BAR,
        ULTIMATE_BAR,
        GAMMA_M0_BAR,
        GAMMA_M2_BAR,
    )

    assert resistance * NEWTON_TO_KILONEWTON == pytest.approx(1325.0, rel=TOLERANCE)


def test_gross_section_yielding_governs_this_tie():
    gross = resistance_yielding(AREA_GROSS, YIELD_BAR, GAMMA_M0_BAR)
    net = resistance_fracture(AREA_NET, ULTIMATE_BAR, GAMMA_M2_BAR)

    assert gross < net


# ---- Example 6.2: a 254 x 254 x 73 UKC in compression, S355 ---- #
#
# Short member (lam <= 0.2), so 6.2.4 alone governs and no buckling check is
# needed. The section is Class 2 by Table 5.2 sheets 1 and 2, which this
# package does not implement: classification here is CHS-only, Table 5.2
# sheet 3. The guide's own verdict is carried in the test name instead.

AREA_UKC = 9310.0
YIELD_UKC = 355.0
GAMMA_M0_UKC = 1.00


def test_compression_cross_section_resistance():
    resistance = resistance_compression(AREA_UKC, YIELD_UKC, GAMMA_M0_UKC)

    assert resistance * NEWTON_TO_KILONEWTON == pytest.approx(3305.0, rel=TOLERANCE)


def test_compression_and_tension_clauses_agree_on_the_gross_section():
    # Eq. 6.6 and Eq. 6.10 are the same expression under different clauses.
    compression = resistance_compression(AREA_UKC, YIELD_UKC, GAMMA_M0_UKC)
    tension = resistance_yielding(AREA_UKC, YIELD_UKC, GAMMA_M0_UKC)

    assert compression == pytest.approx(tension)
