import jax
import jax.numpy as jnp
import numpy as np
import pytest

from normax.ec3.resistance import n_cr
from normax.ec3.resistance import slenderness
from normax.ec3.section import area
from normax.ec3.section import second_moment
from normax.ec3.stability import ALPHA_CR_AMPLIFIABLE
from normax.ec3.stability import ALPHA_CR_ELASTIC
from normax.ec3.stability import ALPHA_CR_PLASTIC
from normax.ec3.stability import amplification
from normax.ec3.stability import buckling_length
from normax.ec3.stability import critical_force
from normax.ec3.stability import is_adequate
from normax.ec3.stability import resistance_factor
from normax.ec3.stability import slenderness_global
from normax.ec3.stability import utilization

E_MOD = 210_000.0
F_Y = 355.0
RATIO = 59.577_464_788_732_41


# --------------------------------------------------------------------------- #
# The two routes to slenderness are one equation — algebra, so no source needed
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("diameter", [50.0, 100.0, 244.5, 500.0])
@pytest.mark.parametrize("n_ed", [-1e3, -1e5, -2e6])
@pytest.mark.parametrize("l_cr", [500.0, 4000.0, 12000.0])
def test_the_member_route_and_the_global_route_agree(diameter, n_ed, l_cr):
    gross = area(diameter, RATIO)
    inertia = second_moment(diameter, RATIO)

    critical = n_cr(inertia, l_cr, E_MOD)
    by_member = slenderness(gross, F_Y, critical)

    # The same member, described by a load factor instead of a length.
    alpha_cr = critical / abs(n_ed)
    by_global = slenderness_global(resistance_factor(gross, F_Y, n_ed), alpha_cr)

    assert float(by_global) == pytest.approx(float(by_member), rel=1e-14)


@pytest.mark.parametrize("diameter", [80.0, 300.0])
@pytest.mark.parametrize("n_ed", [-5e4, -8e5])
def test_a_buckling_length_survives_a_round_trip_through_a_load_factor(diameter, n_ed):
    inertia = second_moment(diameter, RATIO)
    original = 3500.0

    alpha_cr = n_cr(inertia, original, E_MOD) / abs(n_ed)
    recovered = buckling_length(alpha_cr, n_ed, inertia, E_MOD)

    assert float(recovered) == pytest.approx(original, rel=1e-13)


def test_the_load_factor_scales_the_members_share_of_the_load():
    assert float(critical_force(4.0, -250.0)) == pytest.approx(1000.0, rel=1e-15)
    assert float(critical_force(4.0, 250.0)) == pytest.approx(1000.0, rel=1e-15)


def test_a_stiffer_frame_is_a_less_slender_member():
    gross = area(100.0, RATIO)
    factor = resistance_factor(gross, F_Y, -1e5)

    assert float(slenderness_global(factor, 20.0)) < float(
        slenderness_global(factor, 5.0)
    )


def test_the_routes_agree_elementwise_over_members():
    diameters = jnp.array([60.0, 90.0, 140.0])
    lengths = jnp.array([800.0, 1500.0, 2600.0])
    n_ed = jnp.array([-4e4, -9e4, -3e5])

    gross = area(diameters, RATIO)
    inertia = second_moment(diameters, RATIO)
    critical = n_cr(inertia, lengths, E_MOD)

    by_member = slenderness(gross, F_Y, critical)
    by_global = slenderness_global(
        resistance_factor(gross, F_Y, n_ed), critical / jnp.abs(n_ed)
    )

    assert np.allclose(by_global, by_member, rtol=1e-14)


# --------------------------------------------------------------------------- #
# The check itself
# --------------------------------------------------------------------------- #
def test_the_thresholds_are_the_values_the_spec_records():
    assert ALPHA_CR_ELASTIC == 10.0
    assert ALPHA_CR_PLASTIC == 15.0
    assert ALPHA_CR_AMPLIFIABLE == 3.0


def test_a_frame_exactly_on_the_threshold_is_exactly_utilized():
    assert float(utilization(ALPHA_CR_ELASTIC)) == pytest.approx(1.0, rel=1e-15)
    assert bool(is_adequate(ALPHA_CR_ELASTIC)) is True


@pytest.mark.parametrize("alpha_cr", [10.0, 12.0, 50.0, 1e3])
def test_a_stiff_frame_satisfies_the_clause(alpha_cr):
    assert bool(is_adequate(alpha_cr)) is True
    assert float(utilization(alpha_cr)) <= 1.0


@pytest.mark.parametrize("alpha_cr", [0.129, 1.0, 3.0, 9.999])
def test_a_soft_frame_does_not(alpha_cr):
    assert bool(is_adequate(alpha_cr)) is False
    assert float(utilization(alpha_cr)) > 1.0


def test_the_plastic_threshold_is_harder_to_satisfy():
    assert bool(is_adequate(12.0, ALPHA_CR_ELASTIC)) is True
    assert bool(is_adequate(12.0, ALPHA_CR_PLASTIC)) is False


def test_the_utilization_falls_as_the_frame_stiffens():
    factors = jnp.array([0.5, 1.0, 5.0, 10.0, 100.0])
    used = utilization(factors)

    assert np.all(np.diff(np.asarray(used)) < 0.0)


def test_a_frame_that_has_already_buckled_is_flagged_not_hidden():
    # A factor below one means instability before the design load. The check must
    # report a utilization above one rather than clamp it into looking adequate.
    assert float(utilization(0.1291)) == pytest.approx(77.459, rel=1e-4)
    assert bool(is_adequate(0.1291)) is False


# --------------------------------------------------------------------------- #
# The amplifier
# --------------------------------------------------------------------------- #
def test_the_amplifier_is_one_for_an_infinitely_stiff_frame():
    assert float(amplification(1e12)) == pytest.approx(1.0, rel=1e-11)


def test_the_amplifier_grows_as_the_frame_softens():
    assert float(amplification(10.0)) == pytest.approx(10.0 / 9.0, rel=1e-14)
    assert float(amplification(3.0)) == pytest.approx(1.5, rel=1e-14)
    assert float(amplification(5.0)) < float(amplification(4.0))


def test_the_amplifier_turns_negative_once_the_frame_has_buckled():
    # Arithmetic saying the frame is past its critical load, not a defect. It is
    # returned unclamped so the caller cannot mistake it for a valid amplifier.
    assert float(amplification(0.5)) < 0.0


# --------------------------------------------------------------------------- #
# Differentiability, since these feed reported quantities
# --------------------------------------------------------------------------- #
def test_the_utilization_is_differentiable_in_the_load_factor():
    gradient = jax.grad(lambda alpha: utilization(alpha))(4.0)

    assert np.isfinite(float(gradient))
    assert float(gradient) < 0.0


def test_the_global_slenderness_is_differentiable():
    gradient = jax.grad(slenderness_global, argnums=(0, 1))(9.0, 4.0)

    assert all(np.isfinite(float(value)) for value in gradient)


# --------------------------------------------------------------------------- #
# A member the load never reaches
# --------------------------------------------------------------------------- #
def test_an_unloaded_member_has_no_amplifier():
    # A gridshell's boundary hoops span support to support and carry nothing.
    # The amplifier is a ratio to the load a member carries, so there is none.
    gross = area(jnp.array([100.0, 100.0]), RATIO)
    factor = resistance_factor(gross, F_Y, jnp.array([-1e5, 0.0]))

    assert np.isfinite(float(factor[0]))
    assert np.isnan(float(factor[1]))


def test_an_unloaded_member_has_no_equivalent_buckling_length():
    inertia = second_moment(jnp.array([100.0, 100.0]), RATIO)
    lengths = buckling_length(0.4, jnp.array([-1e5, 0.0]), inertia, E_MOD)

    assert np.isfinite(float(lengths[0]))
    assert np.isnan(float(lengths[1]))


def test_an_unloaded_member_is_not_reported_as_infinitely_slender():
    # Infinity would read as a statement about the member. nan says the question
    # does not apply, and a reduction over the members says so too.
    gross = area(jnp.array([80.0, 80.0]), RATIO)
    slender = slenderness_global(
        resistance_factor(gross, F_Y, jnp.array([-5e4, 0.0])), 0.4
    )

    assert not np.isinf(np.asarray(slender)).any()
    assert np.isnan(float(slender[1]))


def test_a_loaded_member_is_untouched_by_the_guard():
    gross = area(120.0, RATIO)
    inertia = second_moment(120.0, RATIO)
    n_ed = -2.5e5

    critical = n_cr(inertia, 3000.0, E_MOD)
    alpha_cr = critical / abs(n_ed)

    assert float(resistance_factor(gross, F_Y, n_ed)) == pytest.approx(
        float(gross) * F_Y / abs(n_ed), rel=1e-15
    )
    assert float(buckling_length(alpha_cr, n_ed, inertia, E_MOD)) == pytest.approx(
        3000.0, rel=1e-13
    )
