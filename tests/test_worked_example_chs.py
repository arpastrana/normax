import jax.numpy as jnp
import numpy as np
import pytest

from normax.ec3.classification import class_limits
from normax.ec3.classification import classify
from normax.ec3.classification import epsilon
from normax.ec3.resistance import IMPERFECTION_FACTORS
from normax.ec3.resistance import chi
from normax.ec3.resistance import n_b_rd
from normax.ec3.resistance import n_c_rd
from normax.ec3.resistance import n_cr
from normax.ec3.resistance import phi
from normax.ec3.resistance import slenderness
from normax.ec3.section import area
from normax.ec3.section import modulus_elastic
from normax.ec3.section import modulus_plastic
from normax.ec3.section import second_moment
from normax.ec3.section import thickness

# Gardner, L. and Nethercot, D. (2011), Designers' Guide to Eurocode 3, 2nd edn,
# ICE Publishing. Worked Example 6.7, "buckling resistance of a compression
# member", pp. 61-63. A hot-finished CHS 244.5 x 10 in S355, pinned at both
# ends over 4 m, carrying N_Ed = 2110 kN.
#
# This is the primary fixture: it runs classification, cross-section resistance
# and flexural buckling in one chain.
#
# Note the guide numbers this Example 6.7, not 6.2. docs/clauses.md carried the
# wrong number until it was checked against references/9780727741721.pdf.

DIAMETER = 244.5
THICKNESS = 10.0
RATIO = DIAMETER / THICKNESS
YIELD = 355.0
MODULUS = 210000.0
LENGTH_BUCKLING = 4000.0
ALPHA = IMPERFECTION_FACTORS["a"]
GAMMA_M0 = 1.0
GAMMA_M1 = 1.0

NEWTON_TO_KILONEWTON = 1e-3

# As printed in the guide, rounded to 2 s.f. on the buckling intermediates.
GUIDE = {
    "area": 7370.0,
    "second_moment": 50730000.0,
    "modulus_elastic": 415000.0,
    "modulus_plastic": 550000.0,
    "epsilon": 0.81,
    "ratio": 24.5,
    "n_c_rd": 2616e3,
    "n_cr": 6571e3,
    "slenderness": 0.63,
    "phi": 0.74,
    "chi": 0.88,
    "n_b_rd": 2297e3,
}

# The same quantities in closed form, recomputed independently of this package.
EXACT = {
    "area": 7367.034773,
    "second_moment": 50731473.423122,
    "modulus_elastic": 414981.377694,
    "modulus_plastic": 550235.833333,
    "epsilon": 0.813617,
    "ratio": 24.45,
    "class_limit_1": 33.098592,
    "n_c_rd": 2615297.344297,
    "n_cr": 6571681.900489,
    "slenderness": 0.630844,
    "phi": 0.744221,
    "chi": 0.877915,
    "n_b_rd": 2296007.573913,
}

# The guide rounds intermediates to 2 s.f.: Phi lands 0.57% off its own closed
# form, so the printed column cannot be held to 0.5%.
TOLERANCE_EXACT = 5e-3
TOLERANCE_GUIDE = 1e-2


@pytest.fixture
def chain():
    gross = area(DIAMETER, RATIO)
    inertia = second_moment(DIAMETER, RATIO)
    critical = n_cr(inertia, LENGTH_BUCKLING, MODULUS)
    non_dimensional = slenderness(gross, YIELD, critical)
    reduction = chi(non_dimensional, ALPHA)

    return {
        "area": gross,
        "second_moment": inertia,
        "modulus_elastic": modulus_elastic(DIAMETER, RATIO),
        "modulus_plastic": modulus_plastic(DIAMETER, RATIO),
        "epsilon": epsilon(YIELD),
        "ratio": DIAMETER / thickness(DIAMETER, RATIO),
        "class_limit_1": class_limits(YIELD)[0],
        "n_c_rd": n_c_rd(gross, YIELD, GAMMA_M0),
        "n_cr": critical,
        "slenderness": non_dimensional,
        "phi": phi(non_dimensional, ALPHA),
        "chi": reduction,
        "n_b_rd": n_b_rd(reduction, gross, YIELD, GAMMA_M1),
    }


@pytest.mark.parametrize("quantity", sorted(EXACT))
def test_matches_the_closed_form(chain, quantity):
    assert chain[quantity] == pytest.approx(EXACT[quantity], rel=TOLERANCE_EXACT)


@pytest.mark.parametrize("quantity", sorted(GUIDE))
def test_matches_the_guide(chain, quantity):
    assert chain[quantity] == pytest.approx(GUIDE[quantity], rel=TOLERANCE_GUIDE)


def test_section_properties_come_from_the_figure(chain):
    # Figure 6.21 tabulates all four.
    assert chain["area"] == pytest.approx(7370.0, rel=TOLERANCE_GUIDE)
    assert chain["second_moment"] == pytest.approx(50730000.0, rel=TOLERANCE_GUIDE)
    assert chain["modulus_elastic"] == pytest.approx(415000.0, rel=TOLERANCE_GUIDE)
    assert chain["modulus_plastic"] == pytest.approx(550000.0, rel=TOLERANCE_GUIDE)


def test_section_is_class_one(chain):
    assert classify(RATIO, YIELD) == 1


def test_class_one_limit_is_fifty_epsilon_squared(chain):
    # The guide prints "Limit for Class 1 section = 50 eps^2 = 40.7" on p. 62.
    # 40.7 is 50 eps; 50 eps^2 is 33.10. Table 5.2 sheet 3 (p. 41) and the
    # eps^2 row both give the squared form, so the guide's arithmetic is wrong
    # and the formula is right. The verdict is unaffected either way.
    assert chain["class_limit_1"] == pytest.approx(33.098592, rel=TOLERANCE_EXACT)
    assert chain["ratio"] < chain["class_limit_1"]


def test_buckling_curve_is_curve_a():
    # Table 6.5 of the guide (Table 6.2 of EN 1993-1-1): hollow sections, hot
    # finished, any grade below S460 -> curve a.
    assert ALPHA == 0.21


def test_both_resistances_carry_the_design_force(chain):
    # The guide's verdict: 2616 > 2110 and 2297 > 2110, section acceptable.
    design_force = 2110e3

    assert chain["n_c_rd"] > design_force
    assert chain["n_b_rd"] > design_force


def test_buckling_governs_over_the_cross_section(chain):
    assert chain["n_b_rd"] < chain["n_c_rd"]
    assert chain["chi"] < 1.0


def test_resistances_in_kilonewtons(chain):
    assert chain["n_c_rd"] * NEWTON_TO_KILONEWTON == pytest.approx(
        2616.0, rel=TOLERANCE_GUIDE
    )
    assert chain["n_b_rd"] * NEWTON_TO_KILONEWTON == pytest.approx(
        2297.0, rel=TOLERANCE_GUIDE
    )


def test_chain_is_float64(chain):
    for value in chain.values():
        assert jnp.asarray(value).dtype == jnp.float64


def test_chain_vectorizes_over_members(chain):
    diameters = jnp.full((5,), DIAMETER)

    areas = area(diameters, RATIO)

    assert areas.shape == (5,)
    assert np.asarray(areas) == pytest.approx(float(chain["area"]))
