import jax
import jax.numpy as jnp
import numpy as np
import pytest

from normax.ec3.material import IMPERFECTION_FACTORS
from normax.ec3.material import SteelGrade
from normax.ec3.resistance import SHEAR_THRESHOLD
from normax.ec3.resistance import area_shear
from normax.ec3.resistance import buckling_auxiliary
from normax.ec3.resistance import force_critical
from normax.ec3.resistance import reduction_buckling
from normax.ec3.resistance import resistance_buckling
from normax.ec3.resistance import resistance_compression
from normax.ec3.resistance import resistance_fracture
from normax.ec3.resistance import resistance_shear
from normax.ec3.resistance import resistance_tension
from normax.ec3.resistance import resistance_yielding
from normax.ec3.resistance import slenderness_from_force
from normax.ec3.resistance import slenderness_from_gyration
from normax.ec3.resistance import slenderness_reference

# EN 1993-1-1 Table 6.1, most slender curve first.
CURVES = ["a0", "a", "b", "c", "d"]
ALPHAS = [IMPERFECTION_FACTORS[curve] for curve in CURVES]

# Below 0.2 the chi <= 1 cap binds and every curve is flat at 1.0 (6.3.1.2(3)),
# so the strict properties are only claimed above the offset.
OFFSET = 0.2
BELOW = jnp.linspace(1e-3, OFFSET, 200, endpoint=False)
ABOVE = jnp.linspace(OFFSET, 4.0, 800)
GRID = jnp.linspace(1e-3, 5.0, 1000)


# ---- Table 6.1 ---- #


def test_imperfection_factors_match_table_6_1():
    assert IMPERFECTION_FACTORS == {
        "a0": 0.13,
        "a": 0.21,
        "b": 0.34,
        "c": 0.49,
        "d": 0.76,
    }


# ---- 6.3.1.2, Eq. 6.49 and the Phi below it ---- #


@pytest.mark.parametrize("alpha", ALPHAS)
def test_chi_is_one_at_the_offset(alpha):
    # Phi = 0.5 * (1 + 0 + 0.04) = 0.52 and sqrt(0.52^2 - 0.2^2) = 0.48 exactly,
    # so chi = 1/1.00 for every curve. This pins the -0.2 in Phi.
    assert buckling_auxiliary(OFFSET, alpha) == pytest.approx(0.52, abs=1e-15)
    assert reduction_buckling(OFFSET, alpha) == pytest.approx(1.0, abs=1e-15)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_chi_never_exceeds_one(alpha):
    values = reduction_buckling(GRID, alpha)

    assert jnp.all(values <= 1.0), f"max chi {jnp.max(values)} at alpha {alpha}"


@pytest.mark.parametrize("alpha", ALPHAS)
def test_chi_never_exceeds_euler(alpha):
    # chi = 1/lam^2 is the Euler load over the squash load; buckling resistance
    # may never exceed it.
    euler = 1.0 / GRID**2
    excess = jnp.max(reduction_buckling(GRID, alpha) - euler)

    assert excess <= 0.0, f"chi exceeds 1/lam^2 by {excess} at alpha {alpha}"


@pytest.mark.parametrize("alpha", ALPHAS)
def test_chi_approaches_euler(alpha):
    assert reduction_buckling(1000.0, alpha) * 1000.0**2 == pytest.approx(1.0, rel=1e-3)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_chi_approaches_euler_from_below(alpha):
    slender = jnp.asarray([1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0])
    ratios = reduction_buckling(slender, alpha) * slender**2

    assert jnp.all(ratios < 1.0), f"chi exceeds Euler at alpha {alpha}"
    assert jnp.all(jnp.diff(ratios) > 0.0), f"not monotone at alpha {alpha}"


@pytest.mark.parametrize("alpha", ALPHAS)
def test_chi_is_flat_below_the_offset(alpha):
    # 6.3.1.2(3): below 0.2 buckling may be ignored, so chi is exactly 1.
    assert jnp.all(reduction_buckling(BELOW, alpha) == 1.0), (
        f"chi below 1 under the offset, alpha {alpha}"
    )


@pytest.mark.parametrize("alpha", ALPHAS)
def test_chi_decreases_with_slenderness(alpha):
    steps = jnp.diff(reduction_buckling(ABOVE, alpha))

    assert jnp.all(steps < 0.0), f"chi not strictly decreasing at alpha {alpha}"


@pytest.mark.parametrize("alpha", ALPHAS)
def test_chi_is_positive(alpha):
    assert jnp.all(reduction_buckling(GRID, alpha) > 0.0)


def test_curve_ordering_above_the_offset():
    # More imperfect curves buckle sooner: a0 > a > b > c > d at every
    # slenderness past the offset.
    curves = [reduction_buckling(ABOVE[1:], alpha) for alpha in ALPHAS]

    for slender, stocky, name_a, name_b in zip(curves, curves[1:], CURVES, CURVES[1:]):
        assert jnp.all(slender > stocky), f"curve {name_a} not above curve {name_b}"


def test_curves_coincide_below_the_offset():
    curves = [reduction_buckling(BELOW, alpha) for alpha in ALPHAS]

    for values in curves[1:]:
        assert jnp.all(values == curves[0])


@pytest.mark.parametrize("alpha", ALPHAS)
def test_the_square_root_argument_stays_positive(alpha):
    # Phi^2 - lam^2 never approaches zero over the practical range, so no clip
    # is needed inside the square root. A clip would change the gradient.
    residual = buckling_auxiliary(GRID, alpha) ** 2 - GRID**2

    assert jnp.min(residual) > 0.0, (
        f"sqrt argument {jnp.min(residual)} at alpha {alpha}"
    )


@pytest.mark.parametrize("alpha", ALPHAS)
def test_chi_is_finite_everywhere(alpha):
    assert jnp.all(jnp.isfinite(reduction_buckling(GRID, alpha)))


# ---- 6.3.1.3, Eq. 6.50 ---- #


def test_n_cr_matches_euler():
    second_moment, l_cr, e_mod = 50731473.4, 4000.0, 210000.0
    expected = np.pi**2 * e_mod * second_moment / l_cr**2

    assert force_critical(
        second_moment, l_cr, SteelGrade(e_mod=e_mod)
    ) == pytest.approx(expected)


def test_n_cr_scales_inversely_with_length_squared():
    assert force_critical(1e6, 8000.0, SteelGrade()) == pytest.approx(
        0.25 * force_critical(1e6, 4000.0, SteelGrade())
    )


def test_lambda_1_matches_the_93_9_epsilon_form():
    # lambda_1 = pi sqrt(E/f_y) = 93.9 eps for E = 210000.
    for f_y in (235.0, 275.0, 355.0, 420.0, 460.0):
        assert slenderness_reference(SteelGrade(f_y=f_y)) == pytest.approx(
            93.9 * np.sqrt(235.0 / f_y), rel=1e-3
        )


@pytest.mark.parametrize("l_cr", [2000.0, 4000.0, 8000.0])
def test_slenderness_forms_agree(l_cr):
    # Eq. 6.50 states lam = sqrt(A f_y / N_cr) = (L_cr / i) / lambda_1.
    area, second_moment, f_y = 7367.0348, 50731473.4, 355.0
    radius = np.sqrt(second_moment / area)

    from_force = slenderness_from_force(
        area, SteelGrade(f_y=f_y), force_critical(second_moment, l_cr, SteelGrade())
    )
    from_gyration = slenderness_from_gyration(l_cr, radius, SteelGrade(f_y=f_y))

    assert from_force == pytest.approx(from_gyration)


def test_slenderness_scales_with_length():
    area, second_moment, f_y = 7367.0348, 50731473.4, 355.0
    short = slenderness_from_force(
        area, SteelGrade(f_y=f_y), force_critical(second_moment, 4000.0, SteelGrade())
    )
    long = slenderness_from_force(
        area, SteelGrade(f_y=f_y), force_critical(second_moment, 8000.0, SteelGrade())
    )

    assert long == pytest.approx(2.0 * short)


# ---- 6.2.3, 6.2.4 and 6.3.1 ---- #


def test_n_pl_rd_is_the_squash_load():
    assert resistance_yielding(
        5000.0, SteelGrade(f_y=265.0, gamma_m0=1.0)
    ) == pytest.approx(5000.0 * 265.0)


def test_n_u_rd_carries_the_nine_tenths_factor():
    assert resistance_fracture(
        4406.0, SteelGrade(f_u=430.0, gamma_m2=1.1)
    ) == pytest.approx(0.9 * 4406.0 * 430.0 / 1.1)


def test_n_c_rd_equals_n_pl_rd():
    # Eq. 6.6 and Eq. 6.10 are the same expression under different clauses.
    assert resistance_compression(
        7367.0, SteelGrade(f_y=355.0, gamma_m0=1.0)
    ) == pytest.approx(resistance_yielding(7367.0, SteelGrade(f_y=355.0, gamma_m0=1.0)))


def test_n_t_rd_takes_the_smaller_resistance():
    gross = resistance_yielding(5000.0, SteelGrade(f_y=265.0, gamma_m0=1.0))
    net = resistance_fracture(4406.0, SteelGrade(f_u=430.0, gamma_m2=1.1))

    assert resistance_tension(
        5000.0, 4406.0, SteelGrade(f_y=265.0, f_u=430.0, gamma_m0=1.0, gamma_m2=1.1)
    ) == pytest.approx(min(gross, net))


def test_n_t_rd_is_governed_by_the_net_section_when_holes_are_large():
    # Shrink the net area until fracture governs.
    gross = resistance_yielding(5000.0, SteelGrade(f_y=265.0, gamma_m0=1.0))

    assert (
        resistance_tension(
            5000.0, 2000.0, SteelGrade(f_y=265.0, f_u=430.0, gamma_m0=1.0, gamma_m2=1.1)
        )
        < gross
    )


def test_n_b_rd_reduces_the_cross_section_resistance():
    resistance = resistance_compression(7367.0, SteelGrade(f_y=355.0, gamma_m0=1.0))

    assert resistance_buckling(
        0.877915, 7367.0, SteelGrade(f_y=355.0, gamma_m1=1.0)
    ) == pytest.approx(0.877915 * resistance)


def test_n_b_rd_meets_n_c_rd_for_a_stocky_member():
    # chi = 1 and gamma_M1 = gamma_M0, so buckling stops governing.
    assert resistance_buckling(
        1.0, 7367.0, SteelGrade(f_y=355.0, gamma_m1=1.0)
    ) == pytest.approx(
        resistance_compression(7367.0, SteelGrade(f_y=355.0, gamma_m0=1.0))
    )


def test_partial_factors_divide():
    assert resistance_compression(
        7367.0, SteelGrade(f_y=355.0, gamma_m0=1.1)
    ) == pytest.approx(
        resistance_compression(7367.0, SteelGrade(f_y=355.0, gamma_m0=1.0)) / 1.1
    )


# ---- 6.2.6, shear ---- #
#
# Not part of the design check: the sizing map ignores shear. It is here so the
# converged design can be audited after the fact, because the exclusion is only
# honest while V_Ed stays under half of V_pl,Rd (6.2.10). See docs/clauses.md
# open item 0d.

AREA_CHS = 7367.03


def test_the_shear_area_of_a_tube_is_two_over_pi_of_the_gross_area():
    # 6.2.6(3), "circular hollow section and tubes of uniform thickness".
    assert area_shear(AREA_CHS) == pytest.approx(2.0 * AREA_CHS / np.pi)


def test_the_shear_area_is_about_sixty_four_percent_of_the_gross_area():
    assert float(area_shear(AREA_CHS)) / AREA_CHS == pytest.approx(0.6366, rel=1e-3)


def test_the_shear_resistance_follows_equation_6_18():
    expected = area_shear(AREA_CHS) * (355.0 / np.sqrt(3.0)) / 1.0

    assert resistance_shear(
        area_shear(AREA_CHS), SteelGrade(f_y=355.0, gamma_m0=1.0)
    ) == pytest.approx(float(expected))


def test_the_shear_resistance_of_the_fixture_section():
    # CHS 244.5 x 10, S355. Recomputed rather than quoted: the guide works no
    # shear example on a tube.
    resistance = resistance_shear(
        area_shear(AREA_CHS), SteelGrade(f_y=355.0, gamma_m0=1.0)
    )

    assert float(resistance) / 1e3 == pytest.approx(961.2, rel=1e-3)


def test_the_shear_resistance_divides_by_the_partial_factor():
    assert resistance_shear(
        4690.0, SteelGrade(f_y=355.0, gamma_m0=1.1)
    ) == pytest.approx(
        float(resistance_shear(4690.0, SteelGrade(f_y=355.0, gamma_m0=1.0))) / 1.1
    )


def test_the_shear_yield_stress_is_the_tensile_one_over_root_three():
    # The only physics in 6.18: von Mises puts shear yield at f_y / sqrt(3).
    assert float(
        resistance_shear(1.0, SteelGrade(f_y=355.0, gamma_m0=1.0))
    ) == pytest.approx(355.0 / np.sqrt(3.0))


def test_the_interaction_threshold_is_one_half():
    # 6.2.8(2) for bending and shear, 6.2.10 once axial force is present too.
    assert SHEAR_THRESHOLD == 0.5


def test_the_shear_resistance_grows_with_the_area():
    areas = jnp.linspace(1e3, 2e4, 200)

    assert jnp.all(
        jnp.diff(
            resistance_shear(area_shear(areas), SteelGrade(f_y=355.0, gamma_m0=1.0))
        )
        > 0.0
    )


# ---- JAX plumbing ---- #


def test_resistances_are_float64():
    assert reduction_buckling(0.63, 0.21).dtype == jnp.float64
    assert (
        resistance_compression(7367.0, SteelGrade(f_y=355.0, gamma_m0=1.0)).dtype
        == jnp.float64
    )


def test_chi_vectorizes_over_members():
    slenderness_values = jnp.asarray([0.1, 0.63, 1.5, 3.0])

    values = reduction_buckling(slenderness_values, 0.21)

    assert values.shape == (4,)
    assert np.asarray(values) == pytest.approx(
        [float(reduction_buckling(v, 0.21)) for v in [0.1, 0.63, 1.5, 3.0]]
    )


def test_chi_is_jittable():
    assert jax.jit(reduction_buckling)(0.63, 0.21) == pytest.approx(
        reduction_buckling(0.63, 0.21)
    )


def test_chi_is_differentiable_above_the_offset():
    gradient = jax.grad(reduction_buckling)(0.63, 0.21)

    assert jnp.isfinite(gradient)
    assert gradient < 0.0
