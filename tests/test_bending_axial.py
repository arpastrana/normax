import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from normax.ec3.resistance import MOMENT_EXPONENT
from normax.ec3.resistance import m_el_rd
from normax.ec3.resistance import m_n_rd
from normax.ec3.resistance import m_pl_rd
from normax.ec3.resistance import moment_resultant
from normax.ec3.resistance import n_pl_rd
from normax.ec3.resistance import utilization_cross_section
from normax.ec3.resistance import utilization_elastic
from normax.ec3.resistance import utilization_plastic

# EN 1993-1-1 6.2.9, bending and axial force at cross-section level, for a
# circular hollow section. See docs/clauses.md: the reduced plastic moment for
# a CHS is an unnumbered expression inside 6.2.9.1(5), the Designers' Guide
# omits it entirely, and CHS is not eligible for the 6.2.9.1(4) exemption.

AXIAL_RATIOS = [0.0, 0.05, 0.2, 0.4, 0.5, 0.692, 0.8, 0.95, 1.0]

# CHS 244.5 x 10 in S355, the primary fixture of docs/clauses.md.
YIELD = 355.0
AREA = 7367.03
MODULUS_PLASTIC = 550236.0
MODULUS_ELASTIC = 414981.0


def exact_reduction(n):
    # The closed-form plastic interaction for a circular tube, ECCS p. 226 eq.
    # (3.119), from Lescouarc'h (1977). The codified 1 - n^1.7 approximates it.
    return math.sin(math.pi * (1.0 - n) / 2.0)


# ---- Eqs. 6.13 and 6.14 ---- #


def test_plastic_moment_is_the_modulus_times_the_strength():
    assert m_pl_rd(550236.0, 355.0, 1.0) == pytest.approx(550236.0 * 355.0)


def test_elastic_moment_is_the_modulus_times_the_strength():
    assert m_el_rd(414981.0, 355.0, 1.0) == pytest.approx(414981.0 * 355.0)


def test_the_plastic_moment_exceeds_the_elastic_one():
    # Shape factor of a CHS 244.5 x 10, about 1.33.
    plastic = m_pl_rd(550236.0, 355.0, 1.0)
    elastic = m_el_rd(414981.0, 355.0, 1.0)

    assert plastic / elastic == pytest.approx(1.326, rel=1e-3)


def test_partial_factor_divides():
    assert m_pl_rd(550236.0, 355.0, 1.1) == pytest.approx(
        m_pl_rd(550236.0, 355.0, 1.0) / 1.1
    )


# ---- 6.2.9.1(5), the unnumbered CHS expression ---- #


def test_the_exponent_is_one_point_seven():
    assert MOMENT_EXPONENT == 1.7


@pytest.mark.parametrize("n", AXIAL_RATIOS)
def test_reduced_moment_follows_the_clause(n):
    assert m_n_rd(100.0, n) == pytest.approx(100.0 * (1.0 - n**1.7))


def test_reduced_moment_is_the_plastic_moment_without_axial_force():
    assert m_n_rd(127.4, 0.0) == pytest.approx(127.4)


def test_reduced_moment_vanishes_at_full_axial_force():
    assert m_n_rd(127.4, 1.0) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("n", AXIAL_RATIOS)
def test_reduced_moment_never_exceeds_the_plastic_moment(n):
    assert m_n_rd(127.4, n) <= 127.4 + 1e-12


def test_reduced_moment_strictly_decreases_with_axial_force():
    values = m_n_rd(127.4, jnp.linspace(0.0, 1.0, 500))

    assert jnp.all(jnp.diff(values) < 0.0)


def test_reduced_moment_is_never_negative():
    assert jnp.all(m_n_rd(127.4, jnp.linspace(0.0, 1.0, 500)) >= 0.0)


@pytest.mark.parametrize("n", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_reduced_moment_tracks_the_closed_form_solution(n):
    # Independent of the code: the exact circular-tube interaction. The
    # codified expression stays within 5% of it across the range.
    assert m_n_rd(1.0, n) == pytest.approx(exact_reduction(n), abs=0.05)


def test_the_codified_form_is_not_a_bound_on_the_exact_one():
    # Below the crossover the code is conservative, above it is not. Worth
    # pinning, because "conservative everywhere" is the intuitive assumption
    # and it is wrong.
    below = [n / 1000 for n in range(1, 692)]
    above = [n / 1000 for n in range(693, 1000)]

    assert all(float(m_n_rd(1.0, n)) < exact_reduction(n) for n in below)
    assert all(float(m_n_rd(1.0, n)) > exact_reduction(n) for n in above)


# ---- 6.2.9.1(6), biaxial bending, alpha = beta = 2 ---- #


@pytest.mark.parametrize("m_z", [0.0, 10.0, 40.0, 100.0])
@pytest.mark.parametrize("m_y", [0.0, 10.0, 40.0, 100.0])
def test_resultant_is_the_euclidean_norm(m_y, m_z):
    assert moment_resultant(m_y, m_z) == pytest.approx(math.hypot(m_y, m_z))


@pytest.mark.parametrize("m_z", [0.0, 10.0, 40.0, 100.0])
@pytest.mark.parametrize("m_y", [0.0, 10.0, 40.0, 100.0])
def test_resultant_reproduces_equation_6_41(m_y, m_z):
    # With alpha = beta = 2 and equal resistances about both axes, Eq. 6.41 is
    # exactly a resultant check. This is the identity the CHS collapse rests on.
    reduced = 120.0
    equation = (m_y / reduced) ** 2 + (m_z / reduced) ** 2

    assert moment_resultant(m_y, m_z) / reduced == pytest.approx(math.sqrt(equation))


def test_resultant_is_symmetric():
    assert moment_resultant(30.0, 70.0) == pytest.approx(moment_resultant(70.0, 30.0))


def test_resultant_reduces_to_a_single_moment():
    assert moment_resultant(85.0, 0.0) == pytest.approx(85.0)


def test_resultant_ignores_the_sign_of_either_moment():
    assert moment_resultant(-30.0, 70.0) == pytest.approx(moment_resultant(30.0, 70.0))
    assert moment_resultant(30.0, -70.0) == pytest.approx(moment_resultant(30.0, 70.0))


# ---- CHS is not exempt from the reduction ---- #


def test_the_reduction_applies_at_every_axial_force():
    # 6.2.9.1(4) exempts I/H and rectangular hollow sections at small axial
    # force. CHS appears in neither list, so the reduction always bites.
    small = n_pl_rd(7367.0, 355.0, 1.0) * 0.01
    ratio = small / n_pl_rd(7367.0, 355.0, 1.0)

    assert m_n_rd(127.4, ratio) < 127.4


# ---- 6.2.9 as a utilization ---- #
#
# The clause is two checks, not one. Eq. 6.41 bounds the moment against a
# resistance already reduced for axial force, and 6.2.4 bounds the axial force
# on its own. Reporting only the first loses the second whenever the moment is
# small, so the plastic branch takes the larger of the two.


def test_the_plastic_check_recovers_the_squash_check_without_moment():
    # With no moment the sum is the axial ratio raised to the exponent, so it
    # reaches one exactly at squash. Writing the clause as a quotient instead
    # would report a fully squashed section as completely unutilized, losing
    # 6.2.4 altogether.
    squash = float(n_pl_rd(AREA, YIELD))

    assert utilization_plastic(
        -squash, 0.0, 0.0, AREA, MODULUS_PLASTIC, YIELD
    ) == pytest.approx(1.0, rel=1e-12)
    assert (
        float(utilization_plastic(0.0, 0.0, 0.0, AREA, MODULUS_PLASTIC, YIELD)) == 0.0
    )


def test_the_plastic_check_is_the_axial_ratio_to_the_exponent_without_moment():
    axial = 0.955 * float(n_pl_rd(AREA, YIELD))

    value = utilization_plastic(-axial, 0.0, 0.0, AREA, MODULUS_PLASTIC, YIELD)

    assert value == pytest.approx(0.955**1.7, rel=1e-12)


def test_the_plastic_check_is_the_clause_rearranged_not_a_different_check():
    # The clause bounds the resultant by a reduced plastic moment. Dividing
    # through by the unreduced one gives the sum. Same inequality, so the two
    # cross unity together, which is all the sizing map ever asks of it.
    axial = 0.4 * float(n_pl_rd(AREA, YIELD))
    reduced = float(m_n_rd(m_pl_rd(MODULUS_PLASTIC, YIELD), 0.4))

    for factor in (0.5, 0.9, 1.0, 1.1, 2.0):
        quotient = factor * reduced / reduced
        summed = utilization_plastic(
            -axial, factor * reduced, 0.0, AREA, MODULUS_PLASTIC, YIELD
        )

        assert (float(summed) > 1.0) == (quotient > 1.0)
        assert float(summed) - 1.0 == pytest.approx(
            (1.0 - 0.4**1.7) * (quotient - 1.0), rel=1e-12
        )


def test_the_plastic_check_stays_finite_beyond_the_squash_load():
    # The quotient form is singular at squash and negative beyond it, reporting
    # a section overloaded three times over as safe. The sum is monotone there.
    overloaded = jnp.linspace(0.9, 3.0, 200) * float(n_pl_rd(AREA, YIELD))
    values = utilization_plastic(-overloaded, 40e6, 0.0, AREA, MODULUS_PLASTIC, YIELD)

    assert jnp.all(jnp.isfinite(values))
    assert jnp.all(values > 0.0)
    assert jnp.all(jnp.diff(values) > 0.0)


def test_the_elastic_check_reduces_to_the_axial_ratio_without_moment():
    axial = 0.955 * float(n_pl_rd(AREA, YIELD))

    value = utilization_elastic(-axial, 0.0, 0.0, AREA, MODULUS_ELASTIC, YIELD)

    assert value == pytest.approx(0.955, rel=1e-9)


def test_the_plastic_check_sums_the_axial_and_bending_terms():
    axial = 0.2 * float(n_pl_rd(AREA, YIELD))
    bending = moment_resultant(40e6, 15e6) / m_pl_rd(MODULUS_PLASTIC, YIELD)

    value = utilization_plastic(-axial, 40e6, 15e6, AREA, MODULUS_PLASTIC, YIELD)

    assert value == pytest.approx(0.2**1.7 + float(bending), rel=1e-12)


def test_the_elastic_check_is_equation_6_42_written_longhand():
    # 6.42 is a stress check. Divided through by f_y / gamma_M0 it is the sum
    # of the axial and bending ratios, which is what the function returns.
    stress = 500e3 / AREA + moment_resultant(40e6, 15e6) / MODULUS_ELASTIC

    value = utilization_elastic(-500e3, 40e6, 15e6, AREA, MODULUS_ELASTIC, YIELD)

    assert value == pytest.approx(float(stress) / YIELD, rel=1e-12)


def test_the_elastic_resultant_is_the_greatest_stress_around_the_perimeter():
    # For a circular section the bending stress at perimeter angle theta is
    # (M_y sin - M_z cos) / W_el, whose maximum over theta is the resultant
    # over W_el. The resultant is exact here, not an approximation.
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, 20001)
    around = jnp.abs(40e6 * jnp.sin(theta) - 15e6 * jnp.cos(theta)) / MODULUS_ELASTIC

    assert jnp.max(around) == pytest.approx(
        float(moment_resultant(40e6, 15e6) / MODULUS_ELASTIC), rel=1e-7
    )


@pytest.mark.parametrize("plastic", [True, False])
def test_the_checks_ignore_the_sign_of_the_axial_force(plastic):
    # 6.2.9 covers bending and axial force of either sign; the ratio n is a
    # magnitude. A tension member is checked by the same expression.
    modulus = MODULUS_PLASTIC if plastic else MODULUS_ELASTIC
    tension = utilization_cross_section(
        500e3, 40e6, 15e6, AREA, modulus, YIELD, plastic=plastic
    )
    compression = utilization_cross_section(
        -500e3, 40e6, 15e6, AREA, modulus, YIELD, plastic=plastic
    )

    assert tension == pytest.approx(float(compression), rel=1e-14)


@pytest.mark.parametrize("plastic", [True, False])
def test_the_checks_ignore_the_sign_of_either_moment(plastic):
    modulus = MODULUS_PLASTIC if plastic else MODULUS_ELASTIC
    positive = utilization_cross_section(
        -500e3, 40e6, 15e6, AREA, modulus, YIELD, plastic=plastic
    )
    mixed = utilization_cross_section(
        -500e3, -40e6, 15e6, AREA, modulus, YIELD, plastic=plastic
    )

    assert positive == pytest.approx(float(mixed), rel=1e-14)


@pytest.mark.parametrize("n", [0.0, 0.2, 0.5, 0.9, 0.99, 1.0 - 1e-9])
def test_the_plastic_check_is_unity_on_the_reduced_moment(n):
    # The sum stays conditioned right up to squash, where the quotient form
    # divides by a vanishing resistance and drifts off unity.
    reduced = float(m_n_rd(m_pl_rd(MODULUS_PLASTIC, YIELD), n))
    axial = n * float(n_pl_rd(AREA, YIELD))

    value = utilization_plastic(-axial, reduced, 0.0, AREA, MODULUS_PLASTIC, YIELD)

    assert value == pytest.approx(1.0, abs=1e-12)


def test_the_elastic_check_is_unity_on_the_yield_stress():
    axial = 0.3 * float(n_pl_rd(AREA, YIELD))
    moment = 0.7 * float(m_el_rd(MODULUS_ELASTIC, YIELD))

    value = utilization_elastic(-axial, moment, 0.0, AREA, MODULUS_ELASTIC, YIELD)

    assert value == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("plastic", [True, False])
def test_the_dispatcher_selects_the_branch(plastic):
    modulus = MODULUS_PLASTIC if plastic else MODULUS_ELASTIC
    branch = utilization_plastic if plastic else utilization_elastic

    assert utilization_cross_section(
        -500e3, 40e6, 15e6, AREA, modulus, YIELD, plastic=plastic
    ) == pytest.approx(float(branch(-500e3, 40e6, 15e6, AREA, modulus, YIELD)))


@pytest.mark.parametrize("plastic", [True, False])
def test_the_checks_grow_with_the_axial_force(plastic):
    modulus = MODULUS_PLASTIC if plastic else MODULUS_ELASTIC
    forces = -jnp.linspace(1e3, 0.9 * float(n_pl_rd(AREA, YIELD)), 300)
    values = utilization_cross_section(
        forces, 40e6, 15e6, AREA, modulus, YIELD, plastic=plastic
    )

    assert jnp.all(jnp.diff(values) > 0.0)


@pytest.mark.parametrize("plastic", [True, False])
def test_the_checks_grow_with_the_moment(plastic):
    modulus = MODULUS_PLASTIC if plastic else MODULUS_ELASTIC
    moments = jnp.linspace(0.0, 120e6, 300)
    values = utilization_cross_section(
        -500e3, moments, 0.0, AREA, modulus, YIELD, plastic=plastic
    )

    assert jnp.all(jnp.diff(values) >= 0.0)
    assert values[-1] > values[0]


def test_the_partial_factor_scales_the_elastic_check_exactly():
    # Both terms of 6.42 carry gamma_M0 linearly, so the whole check does.
    unity = utilization_elastic(-500e3, 40e6, 15e6, AREA, MODULUS_ELASTIC, YIELD, 1.0)
    raised = utilization_elastic(-500e3, 40e6, 15e6, AREA, MODULUS_ELASTIC, YIELD, 1.1)

    assert raised == pytest.approx(float(unity) * 1.1, rel=1e-12)


def test_the_partial_factor_raises_the_plastic_check_faster_than_linearly():
    # The factor enters twice: once through the axial ratio and again through
    # the reduced moment, whose own reduction deepens as that ratio grows. So
    # the plastic branch is superlinear in gamma_M0, unlike the elastic one.
    unity = utilization_plastic(-500e3, 40e6, 15e6, AREA, MODULUS_PLASTIC, YIELD, 1.0)
    raised = utilization_plastic(-500e3, 40e6, 15e6, AREA, MODULUS_PLASTIC, YIELD, 1.1)

    assert raised > float(unity) * 1.1


# ---- Gradients ---- #


def test_the_reduced_moment_has_a_finite_gradient_away_from_the_origin():
    for n in (0.05, 0.3, 0.6, 0.9):
        gradient = jax.grad(m_n_rd, argnums=1)(127.4, n)

        assert jnp.isfinite(gradient)
        assert gradient < 0.0


def test_the_first_derivative_stays_finite_at_zero_axial_force():
    # d/dn (1 - n^1.7) = -1.7 n^0.7 -> 0. It is the SECOND derivative that
    # diverges, so first-order gradients are safe at pure bending.
    gradient = jax.grad(m_n_rd, argnums=1)(127.4, 0.0)

    assert jnp.isfinite(gradient)
    assert gradient == pytest.approx(0.0, abs=1e-12)


def test_the_resultant_is_differentiable():
    gradient = jax.grad(moment_resultant)(40.0, 30.0)

    assert jnp.isfinite(gradient)
    assert gradient == pytest.approx(0.8)


def test_the_resultant_has_a_finite_gradient_at_the_origin():
    # An unguarded square root is undefined at zero. A member carrying no
    # moment is the ordinary case in an axial-only check, so leaving it
    # undefined breaks the gradient of every such member.
    gradient = jax.grad(moment_resultant, argnums=(0, 1))(0.0, 0.0)

    assert all(jnp.isfinite(component) for component in gradient)
    assert gradient[0] == pytest.approx(0.0)
    assert gradient[1] == pytest.approx(0.0)


def test_the_gradient_at_the_origin_survives_a_comparison_it_loses():
    # An undefined value is not shielded by losing a maximum: its cotangent is
    # multiplied by zero, and zero times undefined is still undefined. This is
    # the path that reaches the sizing map, so it is pinned separately.
    def governed_by_something_else(m_y, m_z):
        return jnp.maximum(5.0, moment_resultant(m_y, m_z))

    gradient = jax.grad(governed_by_something_else, argnums=(0, 1))(0.0, 0.0)

    assert all(jnp.isfinite(component) for component in gradient)


# ---- JAX plumbing ---- #


def test_values_are_float64():
    assert m_pl_rd(550236.0, 355.0, 1.0).dtype == jnp.float64
    assert m_n_rd(127.4, 0.4).dtype == jnp.float64
    assert moment_resultant(30.0, 40.0).dtype == jnp.float64


def test_vectorizes_over_members():
    ratios = jnp.asarray([0.1, 0.3, 0.6])

    values = m_n_rd(127.4, ratios)

    assert values.shape == (3,)
    assert np.all(np.diff(np.asarray(values)) < 0.0)


def test_is_jittable():
    assert jax.jit(m_n_rd)(127.4, 0.4) == pytest.approx(m_n_rd(127.4, 0.4))
    assert jax.jit(moment_resultant)(30.0, 40.0) == pytest.approx(50.0)
