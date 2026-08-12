import jax.numpy as jnp
import numpy as np
import pytest

from normax.ec3.material import DENSITY
from normax.ec3.material import E_MODULUS
from normax.units import to_kilograms_per_cubic_meter
from normax.units import to_meters
from normax.units import to_millimeters
from normax.units import to_newton_meters
from normax.units import to_newton_millimeters
from normax.units import to_newtons_per_square_millimeter
from normax.units import to_pascals
from normax.units import to_tonnes_per_cubic_millimeter

PAIRS = [
    (to_meters, to_millimeters),
    (to_pascals, to_newtons_per_square_millimeter),
    (to_newton_meters, to_newton_millimeters),
    (to_kilograms_per_cubic_meter, to_tonnes_per_cubic_millimeter),
]


@pytest.mark.parametrize("into,out", PAIRS)
@pytest.mark.parametrize("value", [1.0, 21.3, 355.0, 1e-9, 4.2e7])
def test_a_conversion_and_its_inverse_return_the_original(into, out, value):
    assert out(into(value)) == pytest.approx(value, rel=1e-15)
    assert into(out(value)) == pytest.approx(value, rel=1e-15)


@pytest.mark.parametrize("into,out", PAIRS)
def test_a_conversion_maps_zero_to_zero(into, out):
    assert into(0.0) == 0.0
    assert out(0.0) == 0.0


@pytest.mark.parametrize("into,out", PAIRS)
def test_a_conversion_is_odd_about_zero(into, out):
    assert into(-3.5) == -into(3.5)
    assert out(-3.5) == -out(3.5)


@pytest.mark.parametrize("into,out", PAIRS)
def test_a_conversion_acts_elementwise_on_an_array(into, out):
    values = jnp.array([1.0, 21.3, -355.0])
    assert np.allclose(out(into(values)), values, rtol=1e-15)
    assert into(values).shape == values.shape


def test_a_millimeter_is_a_thousandth_of_a_meter():
    assert to_meters(1000.0) == pytest.approx(1.0, rel=1e-15)
    assert to_millimeters(1.0) == pytest.approx(1000.0, rel=1e-15)


def test_the_modulus_of_steel_is_two_hundred_and_ten_gigapascals():
    assert to_pascals(E_MODULUS) == pytest.approx(210e9, rel=1e-15)


def test_a_newton_per_square_millimeter_is_a_megapascal():
    assert to_pascals(1.0) == pytest.approx(1e6, rel=1e-15)


def test_the_density_of_steel_is_seven_thousand_eight_hundred_and_fifty():
    assert to_kilograms_per_cubic_meter(DENSITY) == pytest.approx(7850.0, rel=1e-15)


def test_a_newton_millimeter_is_a_thousandth_of_a_newton_meter():
    assert to_newton_meters(1000.0) == pytest.approx(1.0, rel=1e-15)
    assert to_newton_millimeters(1.0) == pytest.approx(1000.0, rel=1e-15)


def test_a_stress_times_an_area_is_a_force_in_either_system():
    diameter = 244.5
    thickness = 10.0
    f_y = 355.0

    area = np.pi * thickness * (diameter - thickness)
    force = f_y * area

    area_si = np.pi * to_meters(thickness) * to_meters(diameter - thickness)
    force_si = to_pascals(f_y) * area_si

    assert force_si == pytest.approx(force, rel=1e-12)


def test_a_force_times_a_length_is_a_moment_in_either_system():
    force = 4.2e5
    length = 4000.0

    moment = force * length
    moment_si = force * to_meters(length)

    assert to_newton_millimeters(moment_si) == pytest.approx(moment, rel=1e-12)


def test_a_density_times_a_volume_is_a_mass_in_either_system():
    volume = 1.0e7

    mass_tonnes = DENSITY * volume
    mass_kilograms = (
        to_kilograms_per_cubic_meter(DENSITY) * to_meters(1.0) ** 3 * volume
    )

    assert mass_kilograms == pytest.approx(mass_tonnes * 1000.0, rel=1e-12)
