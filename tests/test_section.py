import jax
import jax.numpy as jnp
import numpy as np
import pytest

from normax.ec3.section import Tube
from normax.ec3.section import TubeCatalogue

DIAMETERS = [50.0, 114.3, 244.5, 508.0]
RATIOS = [10.0, 24.45, 59.58, 90.0]


def sample(d, r):
    return TubeCatalogue(r).tube(d)


# ---- The two leaves ---- #


@pytest.mark.parametrize("r", RATIOS)
@pytest.mark.parametrize("d", DIAMETERS)
def test_a_tube_carries_the_diameter_it_was_sampled_at(d, r):
    assert float(sample(d, r).diameter) == d


@pytest.mark.parametrize("r", RATIOS)
@pytest.mark.parametrize("d", DIAMETERS)
def test_the_catalogue_walls_the_tube_at_its_ratio(d, r):
    # The round trip is not exact: a wall is a division and reading the ratio
    # back is another. Classification widens its limits for exactly this,
    # which is what CLASS_LIMIT_TOLERANCE exists for.
    assert float(sample(d, r).ratio) == pytest.approx(r, rel=1e-15)


@pytest.mark.parametrize("r", RATIOS)
@pytest.mark.parametrize("d", DIAMETERS)
def test_a_tube_can_be_built_without_a_catalogue(d, r):
    direct = Tube(d, d / r)

    assert float(direct.area) == float(sample(d, r).area)


# ---- The properties, against the annulus formulas ---- #


@pytest.mark.parametrize("r", RATIOS)
@pytest.mark.parametrize("d", DIAMETERS)
def test_inner_diameter_is_two_walls_in(d, r):
    tube = sample(d, r)

    assert tube.diameter_inner == pytest.approx(d - 2.0 * tube.thickness)


@pytest.mark.parametrize("r", RATIOS)
@pytest.mark.parametrize("d", DIAMETERS)
def test_area_matches_the_annulus(d, r):
    tube = sample(d, r)
    t = tube.thickness

    assert tube.area == pytest.approx(np.pi * t * (d - t))


@pytest.mark.parametrize("r", RATIOS)
@pytest.mark.parametrize("d", DIAMETERS)
def test_area_is_the_difference_of_two_circles(d, r):
    # The same area the other way round, which is how a reader would check it.
    tube = sample(d, r)
    expected = np.pi / 4.0 * (d**2 - float(tube.diameter_inner) ** 2)

    assert tube.area == pytest.approx(expected)


@pytest.mark.parametrize("r", RATIOS)
@pytest.mark.parametrize("d", DIAMETERS)
def test_second_moment_matches_the_annulus(d, r):
    tube = sample(d, r)

    assert tube.second_moment == pytest.approx(
        np.pi / 64.0 * (d**4 - float(tube.diameter_inner) ** 4)
    )


@pytest.mark.parametrize("r", RATIOS)
@pytest.mark.parametrize("d", DIAMETERS)
def test_modulus_elastic_is_second_moment_over_the_outer_radius(d, r):
    tube = sample(d, r)

    assert tube.modulus_elastic == pytest.approx(2.0 * tube.second_moment / d)


@pytest.mark.parametrize("r", RATIOS)
@pytest.mark.parametrize("d", DIAMETERS)
def test_modulus_plastic_matches_the_annulus(d, r):
    tube = sample(d, r)

    assert tube.modulus_plastic == pytest.approx(
        (d**3 - float(tube.diameter_inner) ** 3) / 6.0
    )


@pytest.mark.parametrize("r", RATIOS)
@pytest.mark.parametrize("d", DIAMETERS)
def test_radius_of_gyration_matches_its_definition(d, r):
    tube = sample(d, r)

    assert tube.radius_of_gyration == pytest.approx(
        np.sqrt(tube.second_moment / tube.area)
    )


# The shear area is not here. A_v = 2A/pi is EN 1993-1-1 6.2.6(3), a clause
# rather than geometry, so it stays in the resistance layer with the rest.


# ---- The scaling the sizing map rests on ---- #


@pytest.mark.parametrize("r", RATIOS)
def test_radius_of_gyration_is_proportional_to_diameter(r):
    # i = c * d with c a function of r alone. The whole sizing map leans on
    # this: it makes capacity strictly increasing in d. It holds because the
    # catalogue makes the wall proportional to the diameter, not because a
    # tube does.
    ratios = [float(sample(d, r).radius_of_gyration) / d for d in DIAMETERS]

    assert ratios == pytest.approx([ratios[0]] * len(ratios))


@pytest.mark.parametrize("r", RATIOS)
def test_area_scales_quadratically(r):
    assert sample(2.0 * 244.5, r).area == pytest.approx(4.0 * sample(244.5, r).area)


@pytest.mark.parametrize("r", RATIOS)
def test_second_moment_scales_quartically(r):
    assert sample(2.0 * 244.5, r).second_moment == pytest.approx(
        16.0 * sample(244.5, r).second_moment
    )


def test_a_fixed_wall_does_not_scale_that_way():
    # The counterexample the note in the module docstring points at: hold the
    # wall and the area stops being quadratic, so the argument above belongs
    # to the catalogue rather than to the tube.
    held = Tube(2.0 * 244.5, 10.0).area / Tube(244.5, 10.0).area

    assert float(held) < 4.0


@pytest.mark.parametrize("r", RATIOS)
def test_thin_walls_approach_the_thin_ring(r):
    # A -> pi d t and I -> pi d^3 t / 8 as the wall thins.
    d = 244.5
    tube = sample(d, r)
    t = tube.thickness

    assert tube.area == pytest.approx(np.pi * d * t, rel=2.0 / r)
    assert tube.second_moment == pytest.approx(np.pi * d**3 * t / 8.0, rel=4.0 / r)


# ---- Array behaviour ---- #


def test_properties_are_float64():
    tube = sample(244.5, 24.45)

    for value in (
        tube.thickness,
        tube.diameter_inner,
        tube.area,
        tube.second_moment,
        tube.radius_of_gyration,
        tube.modulus_elastic,
        tube.modulus_plastic,
        tube.ratio,
    ):
        assert jnp.asarray(value).dtype == jnp.float64


def test_properties_vectorize_over_members():
    areas = sample(jnp.asarray(DIAMETERS), 24.45).area

    assert areas.shape == (len(DIAMETERS),)
    assert np.asarray(areas) == pytest.approx(
        [float(sample(d, 24.45).area) for d in DIAMETERS]
    )


def test_a_tube_is_two_leaves():
    # Properties are computed, so they are not leaves and cannot carry a
    # cotangent of their own.
    leaves = jax.tree.leaves(sample(244.5, 24.45))

    assert len(leaves) == 2


def test_properties_are_jittable():
    jitted = jax.jit(lambda d, r: TubeCatalogue(r).tube(d).area)

    assert jitted(244.5, 24.45) == pytest.approx(sample(244.5, 24.45).area)


def test_properties_are_differentiable():
    slope = jax.grad(lambda d: TubeCatalogue(24.45).tube(d).area)(244.5)

    # A = pi d^2 (r - 1) / r^2, so dA/dd is twice the area over the diameter.
    assert float(slope) == pytest.approx(2.0 * float(sample(244.5, 24.45).area) / 244.5)
