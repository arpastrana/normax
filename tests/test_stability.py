import jax
import jax.numpy as jnp
import numpy as np
import pytest

from normax.ec3.material import Steel
from normax.ec3.resistance import force_critical
from normax.ec3.resistance import slenderness_from_force
from normax.ec3.section import TubeCatalogue
from normax.ec3.stability import ALPHA_CR_AMPLIFIABLE
from normax.ec3.stability import ALPHA_CR_ELASTIC
from normax.ec3.stability import ALPHA_CR_PLASTIC
from normax.ec3.stability import amplification_sway
from normax.ec3.stability import amplifier_resistance
from normax.ec3.stability import buckling_length_global
from normax.ec3.stability import force_critical_global
from normax.ec3.stability import is_adequate
from normax.ec3.stability import slenderness_global
from normax.ec3.stability import utilization_frame

E_MOD = 210_000.0
F_Y = 355.0
RATIO = 59.577_464_788_732_41

# The family every section here is drawn from. Its ratio sits on the Class 3
# limit at this grade, which is what the label states.
FAMILY = TubeCatalogue(RATIO, 3, Steel(f_y=F_Y, e_mod=E_MOD))


# --------------------------------------------------------------------------- #
# The two routes to slenderness are one equation — algebra, so no source needed
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("diameter", [50.0, 100.0, 244.5, 500.0])
@pytest.mark.parametrize("axial_force", [-1e3, -1e5, -2e6])
@pytest.mark.parametrize("buckling_length", [500.0, 4000.0, 12000.0])
def test_the_member_route_and_the_global_route_agree(
    diameter, axial_force, buckling_length
):
    gross = FAMILY(diameter).area
    inertia = FAMILY(diameter).second_moment

    critical = force_critical(inertia, buckling_length, Steel(e_mod=E_MOD))
    by_member = slenderness_from_force(gross, Steel(f_y=F_Y), critical)

    # The same member, described by a load factor instead of a length.
    alpha_cr = critical / abs(axial_force)
    by_global = slenderness_global(
        amplifier_resistance(gross, Steel(f_y=F_Y), axial_force), alpha_cr
    )

    assert float(by_global) == pytest.approx(float(by_member), rel=1e-14)


@pytest.mark.parametrize("diameter", [80.0, 300.0])
@pytest.mark.parametrize("axial_force", [-5e4, -8e5])
def test_a_buckling_length_survives_a_round_trip_through_a_load_factor(
    diameter, axial_force
):
    inertia = FAMILY(diameter).second_moment
    original = 3500.0

    alpha_cr = force_critical(inertia, original, Steel(e_mod=E_MOD)) / abs(axial_force)
    recovered = buckling_length_global(
        alpha_cr, axial_force, inertia, Steel(e_mod=E_MOD)
    )

    assert float(recovered) == pytest.approx(original, rel=1e-13)


def test_the_load_factor_scales_the_members_share_of_the_load():
    assert float(force_critical_global(4.0, -250.0)) == pytest.approx(1000.0, rel=1e-15)
    assert float(force_critical_global(4.0, 250.0)) == pytest.approx(1000.0, rel=1e-15)


def test_a_stiffer_frame_is_a_less_slender_member():
    gross = FAMILY(100.0).area
    factor = amplifier_resistance(gross, Steel(f_y=F_Y), -1e5)

    assert float(slenderness_global(factor, 20.0)) < float(
        slenderness_global(factor, 5.0)
    )


def test_the_routes_agree_elementwise_over_members():
    diameters = jnp.array([60.0, 90.0, 140.0])
    lengths = jnp.array([800.0, 1500.0, 2600.0])
    axial_force = jnp.array([-4e4, -9e4, -3e5])

    gross = FAMILY(diameters).area
    inertia = FAMILY(diameters).second_moment
    critical = force_critical(inertia, lengths, Steel(e_mod=E_MOD))

    by_member = slenderness_from_force(gross, Steel(f_y=F_Y), critical)
    by_global = slenderness_global(
        amplifier_resistance(gross, Steel(f_y=F_Y), axial_force),
        critical / jnp.abs(axial_force),
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
    assert float(utilization_frame(ALPHA_CR_ELASTIC)) == pytest.approx(1.0, rel=1e-15)
    assert bool(is_adequate(ALPHA_CR_ELASTIC)) is True


@pytest.mark.parametrize("alpha_cr", [10.0, 12.0, 50.0, 1e3])
def test_a_stiff_frame_satisfies_the_clause(alpha_cr):
    assert bool(is_adequate(alpha_cr)) is True
    assert float(utilization_frame(alpha_cr)) <= 1.0


@pytest.mark.parametrize("alpha_cr", [0.129, 1.0, 3.0, 9.999])
def test_a_soft_frame_does_not(alpha_cr):
    assert bool(is_adequate(alpha_cr)) is False
    assert float(utilization_frame(alpha_cr)) > 1.0


def test_the_plastic_threshold_is_harder_to_satisfy():
    assert bool(is_adequate(12.0, ALPHA_CR_ELASTIC)) is True
    assert bool(is_adequate(12.0, ALPHA_CR_PLASTIC)) is False


def test_the_utilization_falls_as_the_frame_stiffens():
    factors = jnp.array([0.5, 1.0, 5.0, 10.0, 100.0])
    used = utilization_frame(factors)

    assert np.all(np.diff(np.asarray(used)) < 0.0)


def test_a_frame_that_has_already_buckled_is_flagged_not_hidden():
    # A factor below one means instability before the design load. The check must
    # report a utilization above one rather than clamp it into looking adequate.
    assert float(utilization_frame(0.1291)) == pytest.approx(77.459, rel=1e-4)
    assert bool(is_adequate(0.1291)) is False


# --------------------------------------------------------------------------- #
# The amplifier
# --------------------------------------------------------------------------- #
def test_the_amplifier_is_one_for_an_infinitely_stiff_frame():
    assert float(amplification_sway(1e12)) == pytest.approx(1.0, rel=1e-11)


def test_the_amplifier_grows_as_the_frame_softens():
    assert float(amplification_sway(10.0)) == pytest.approx(10.0 / 9.0, rel=1e-14)
    assert float(amplification_sway(3.0)) == pytest.approx(1.5, rel=1e-14)
    assert float(amplification_sway(5.0)) < float(amplification_sway(4.0))


def test_the_amplifier_turns_negative_once_the_frame_has_buckled():
    # Arithmetic saying the frame is past its critical load, not a defect. It is
    # returned unclamped so the caller cannot mistake it for a valid amplifier.
    assert float(amplification_sway(0.5)) < 0.0


# --------------------------------------------------------------------------- #
# Differentiability, since these feed reported quantities
# --------------------------------------------------------------------------- #
def test_the_utilization_is_differentiable_in_the_load_factor():
    gradient = jax.grad(lambda alpha: utilization_frame(alpha))(4.0)

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
    gross = FAMILY(jnp.array([100.0, 100.0])).area
    factor = amplifier_resistance(gross, Steel(f_y=F_Y), jnp.array([-1e5, 0.0]))

    assert np.isfinite(float(factor[0]))
    assert np.isnan(float(factor[1]))


def test_an_unloaded_member_has_no_equivalent_buckling_length():
    inertia = FAMILY(jnp.array([100.0, 100.0])).second_moment
    lengths = buckling_length_global(
        0.4, jnp.array([-1e5, 0.0]), inertia, Steel(e_mod=E_MOD)
    )

    assert np.isfinite(float(lengths[0]))
    assert np.isnan(float(lengths[1]))


def test_an_unloaded_member_is_not_reported_as_infinitely_slender():
    # Infinity would read as a statement about the member. nan says the question
    # does not apply, and a reduction over the members says so too.
    gross = FAMILY(jnp.array([80.0, 80.0])).area
    slender = slenderness_global(
        amplifier_resistance(gross, Steel(f_y=F_Y), jnp.array([-5e4, 0.0])), 0.4
    )

    assert not np.isinf(np.asarray(slender)).any()
    assert np.isnan(float(slender[1]))


def test_a_loaded_member_is_untouched_by_the_guard():
    gross = FAMILY(120.0).area
    inertia = FAMILY(120.0).second_moment
    axial_force = -2.5e5

    critical = force_critical(inertia, 3000.0, Steel(e_mod=E_MOD))
    alpha_cr = critical / abs(axial_force)

    assert float(
        amplifier_resistance(gross, Steel(f_y=F_Y), axial_force)
    ) == pytest.approx(float(gross) * F_Y / abs(axial_force), rel=1e-15)
    assert float(
        buckling_length_global(alpha_cr, axial_force, inertia, Steel(e_mod=E_MOD))
    ) == pytest.approx(3000.0, rel=1e-13)
