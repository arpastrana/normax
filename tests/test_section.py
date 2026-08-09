import jax
import jax.numpy as jnp
import numpy as np
import pytest

from normax.ec3.section import area
from normax.ec3.section import diameter_inner
from normax.ec3.section import modulus_elastic
from normax.ec3.section import modulus_plastic
from normax.ec3.section import radius_of_gyration
from normax.ec3.section import second_moment
from normax.ec3.section import thickness

DIAMETERS = [50.0, 114.3, 244.5, 508.0]
RATIOS = [10.0, 24.45, 59.58, 90.0]


@pytest.mark.parametrize("r", RATIOS)
@pytest.mark.parametrize("d", DIAMETERS)
def test_thickness_recovers_the_ratio(d, r):
    assert d / thickness(d, r) == pytest.approx(r)


@pytest.mark.parametrize("r", RATIOS)
@pytest.mark.parametrize("d", DIAMETERS)
def test_inner_diameter_is_two_walls_in(d, r):
    assert diameter_inner(d, r) == pytest.approx(d - 2.0 * thickness(d, r))


@pytest.mark.parametrize("r", RATIOS)
@pytest.mark.parametrize("d", DIAMETERS)
def test_area_matches_the_annulus(d, r):
    t = thickness(d, r)

    assert area(d, r) == pytest.approx(np.pi * t * (d - t))


@pytest.mark.parametrize("r", RATIOS)
@pytest.mark.parametrize("d", DIAMETERS)
def test_second_moment_matches_the_annulus(d, r):
    d_inner = diameter_inner(d, r)

    assert second_moment(d, r) == pytest.approx(np.pi / 64.0 * (d**4 - d_inner**4))


@pytest.mark.parametrize("r", RATIOS)
@pytest.mark.parametrize("d", DIAMETERS)
def test_modulus_elastic_is_second_moment_over_the_outer_radius(d, r):
    assert modulus_elastic(d, r) == pytest.approx(2.0 * second_moment(d, r) / d)


@pytest.mark.parametrize("r", RATIOS)
@pytest.mark.parametrize("d", DIAMETERS)
def test_modulus_plastic_matches_the_annulus(d, r):
    d_inner = diameter_inner(d, r)

    assert modulus_plastic(d, r) == pytest.approx((d**3 - d_inner**3) / 6.0)


@pytest.mark.parametrize("r", RATIOS)
@pytest.mark.parametrize("d", DIAMETERS)
def test_radius_of_gyration_matches_its_definition(d, r):
    expected = np.sqrt(second_moment(d, r) / area(d, r))

    assert radius_of_gyration(d, r) == pytest.approx(expected)


@pytest.mark.parametrize("r", RATIOS)
def test_radius_of_gyration_is_proportional_to_diameter(r):
    # i = c * d with c a function of r alone. The whole sizing map leans on
    # this: it makes capacity strictly increasing in d.
    ratios = [float(radius_of_gyration(d, r)) / d for d in DIAMETERS]

    assert ratios == pytest.approx([ratios[0]] * len(ratios))


@pytest.mark.parametrize("r", RATIOS)
def test_area_scales_quadratically(r):
    assert area(2.0 * 244.5, r) == pytest.approx(4.0 * area(244.5, r))


@pytest.mark.parametrize("r", RATIOS)
def test_second_moment_scales_quartically(r):
    assert second_moment(2.0 * 244.5, r) == pytest.approx(
        16.0 * second_moment(244.5, r)
    )


@pytest.mark.parametrize("r", RATIOS)
def test_thin_walls_approach_the_thin_ring(r):
    # A -> pi d t and I -> pi d^3 t / 8 as the wall thins.
    d = 244.5
    t = thickness(d, r)

    assert area(d, r) == pytest.approx(np.pi * d * t, rel=2.0 / r)
    assert second_moment(d, r) == pytest.approx(np.pi * d**3 * t / 8.0, rel=4.0 / r)


def test_properties_are_float64():
    for value in (
        thickness(244.5, 24.45),
        diameter_inner(244.5, 24.45),
        area(244.5, 24.45),
        second_moment(244.5, 24.45),
        radius_of_gyration(244.5, 24.45),
        modulus_elastic(244.5, 24.45),
        modulus_plastic(244.5, 24.45),
    ):
        assert value.dtype == jnp.float64


def test_properties_vectorize_over_members():
    diameters = jnp.asarray(DIAMETERS)

    areas = area(diameters, 24.45)

    assert areas.shape == (len(DIAMETERS),)
    assert np.asarray(areas) == pytest.approx(
        [float(area(d, 24.45)) for d in DIAMETERS]
    )


def test_properties_are_jittable():
    jitted = jax.jit(area)

    assert jitted(244.5, 24.45) == pytest.approx(area(244.5, 24.45))
