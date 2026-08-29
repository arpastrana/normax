# SPDX-License-Identifier: Apache-2.0
"""
The tube catalog, checked against the annulus algebra itself.

The right-hand side is the closed form written out here rather than a second
library's tube, which is the stronger test: two implementations can inherit
one shared mistake, the standard's own algebra cannot.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from normax.materials import Steel355
from normax.materials import SteelGrade
from normax.sections import MemberSections
from normax.sections import TubeCatalog
from normax.sections import UniformDiameterInitializer
from normax.sections import build_section_catalog
from normax.structures import build_arch_2d

# The derived properties the container states, all of them closed form.
PROPERTIES = ("ratio", "thickness", "diameter_inner", "area", "second_moment")

RATIO = 59.5934

DIAMETERS = jnp.asarray([21.3, 100.0, 244.5, 508.0])

# EN 1993-1-1 Table 5.2, the class 3 limit for S355: 90 * 235 / 355.
RATIO_CLASS_3_S355 = 59.57746478873239

# The same algebra arranged differently, so a few ulps rather than bit for bit.
TOLERANCE = 1e-14


def build_tube_sections(diameters):
    """
    The tubes at one wall proportion.
    """
    catalog = TubeCatalog(RATIO, Steel355())
    sections = catalog(diameters)

    return sections


def compute_closed_form(diameters, ratio):
    """
    Every derived property of the annulus, from its own algebra.
    """
    thickness = diameters / ratio
    diameter_inner = diameters * (1.0 - 2.0 / ratio)
    area = jnp.pi * diameters**2 * (ratio - 1.0) / ratio**2
    second_moment = (jnp.pi / 64.0) * diameters**4 * (1.0 - (1.0 - 2.0 / ratio) ** 4)

    stated = {
        "ratio": jnp.full_like(diameters, ratio),
        "thickness": thickness,
        "diameter_inner": diameter_inner,
        "area": area,
        "second_moment": second_moment,
    }

    return stated


def test_every_property_agrees_with_the_closed_form():
    sections = build_tube_sections(DIAMETERS)
    stated = compute_closed_form(DIAMETERS, RATIO)

    for name in PROPERTIES:
        left = np.asarray(stated[name])
        right = np.asarray(getattr(sections, name))

        assert right == pytest.approx(left, rel=TOLERANCE), name


def test_the_wall_follows_the_ratio_at_every_diameter():
    sections = build_tube_sections(DIAMETERS)
    proportions = np.asarray(sections.diameter) / np.asarray(sections.thickness)

    assert proportions == pytest.approx(RATIO, rel=TOLERANCE)


def test_the_radius_of_gyration_is_proportional_to_the_diameter():
    # i = sqrt(I / A) = c d, with c a function of the wall proportion alone.
    sections = build_tube_sections(DIAMETERS)
    radii = np.sqrt(np.asarray(sections.second_moment) / np.asarray(sections.area))
    coefficients = radii / np.asarray(DIAMETERS)

    assert coefficients == pytest.approx(coefficients[0], rel=TOLERANCE)


def test_the_sections_carry_no_clause_field():
    assert set(MemberSections._fields) == {"diameter", "thickness", "material"}
    assert set(TubeCatalog._fields) == {"ratio", "material"}


def test_the_sections_are_a_plain_pytree():
    sections = build_tube_sections(DIAMETERS)
    leaves = jax.tree.leaves(sections)

    assert len(leaves) == 2 + len(SteelGrade._fields)


def test_the_load_case_axis_is_variadic():
    stacked = jnp.stack([DIAMETERS, 2.0 * DIAMETERS])
    sections = build_tube_sections(stacked)

    assert sections.area.shape == stacked.shape


def test_the_geometry_is_differentiable_through_the_catalog():
    catalog = TubeCatalog(RATIO, Steel355())

    def total_area(diameter):
        return jnp.sum(catalog(diameter).area)

    gradient = jax.grad(total_area)(DIAMETERS)

    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert bool(jnp.all(gradient > 0.0))


def test_the_class_3_catalog_sits_on_the_table_limit():
    catalog = build_section_catalog(Steel355(), 3)

    assert catalog.ratio == pytest.approx(RATIO_CLASS_3_S355, rel=1e-15)
    assert catalog.ratio == pytest.approx(59.58, abs=5e-3)
    assert catalog.material == Steel355()


def test_the_class_limits_are_the_table_5_2_limits():
    # EN 1993-1-1 Table 5.2 sheet 3: d/t <= k epsilon^2, with epsilon^2 = 235 / f_y.
    epsilon_squared = 235.0 / Steel355().f_y
    limits = {1: 50.0, 2: 70.0, 3: 90.0}

    for section_class, limit in limits.items():
        catalog = build_section_catalog(Steel355(), section_class)
        expected = limit * epsilon_squared

        assert catalog.ratio == pytest.approx(expected, rel=1e-15), section_class


def test_a_class_4_catalog_is_refused():
    with pytest.raises(ValueError, match="section_class"):
        build_section_catalog(Steel355(), 4)


def test_a_uniform_diameter_initializer_seeds_every_member():
    structure = build_arch_2d(num_edges=6)
    initializer = UniformDiameterInitializer({"diameter": 120.0})
    seeded = initializer(structure)

    assert seeded.shape == (structure.num_edges,)
    assert np.all(seeded == 120.0)


def test_a_diameter_start_that_names_the_wrong_field_is_refused():
    with pytest.raises(ValueError, match="diameter"):
        UniformDiameterInitializer({"diameter_min": 120.0})
