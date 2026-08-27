# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ec3x.material import Steel
from ec3x.section import TubeCatalogue

from normax.materials import Steel355
from normax.materials import SteelGrade
from normax.sections import MemberSections
from normax.sections import TubeFamily
from normax.sections import build_section_family

# The derived properties both libraries state. The two must agree bit for bit,
# or the mass normax weighs and the resistance ec3x checks describe two
# different tubes; this file is the drift alarm the sections doc promises.
PROPERTIES = ("ratio", "diameter_inner", "area", "second_moment")

RATIO = 59.5934

DIAMETERS = jnp.asarray([21.3, 100.0, 244.5, 508.0])

# EN 1993-1-1 Table 5.2, the class 3 limit for S355: 90 * 235 / 355.
RATIO_CLASS_3_S355 = 59.57746478873239


def tube_pair(diameters):
    """
    The same tubes, stated by both libraries at one wall proportion.
    """
    catalogue = TubeCatalogue(RATIO, 3, Steel())
    family = TubeFamily(RATIO, Steel355())

    return catalogue(diameters), family(diameters)


def test_every_property_agrees_bitwise_with_ec3x():
    tubes, sections = tube_pair(DIAMETERS)

    for name in PROPERTIES:
        left = np.asarray(getattr(tubes, name))
        right = np.asarray(getattr(sections, name))

        assert np.array_equal(left, right), name


def test_the_family_chooses_the_same_wall_as_the_catalogue():
    tubes, sections = tube_pair(DIAMETERS)

    assert np.array_equal(np.asarray(tubes.thickness), np.asarray(sections.thickness))


def test_the_sections_carry_no_clause_field():
    assert set(MemberSections._fields) == {"diameter", "thickness", "material"}
    assert set(TubeFamily._fields) == {"ratio", "material"}


def test_the_sections_are_a_plain_pytree():
    _, sections = tube_pair(DIAMETERS)
    leaves = jax.tree.leaves(sections)

    assert len(leaves) == 2 + len(SteelGrade._fields)


def test_the_load_case_axis_is_variadic():
    stacked = jnp.stack([DIAMETERS, 2.0 * DIAMETERS])
    _, sections = tube_pair(stacked)

    assert sections.area.shape == stacked.shape


def test_the_geometry_is_differentiable_through_the_family():
    family = TubeFamily(RATIO, Steel355())

    def total_area(diameter):
        return jnp.sum(family(diameter).area)

    gradient = jax.grad(total_area)(DIAMETERS)

    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert bool(jnp.all(gradient > 0.0))


def test_the_class_3_family_sits_on_the_table_limit():
    family = build_section_family(Steel355(), 3)

    assert family.ratio == pytest.approx(RATIO_CLASS_3_S355, rel=1e-15)
    assert family.ratio == pytest.approx(59.58, abs=5e-3)
    assert family.material == Steel355()


def test_the_class_limits_agree_with_ec3x():
    for section_class in (1, 2, 3):
        family = build_section_family(Steel355(), section_class)
        catalogue = TubeCatalogue(family.ratio, section_class, Steel())

        assert bool(catalogue.at_class_limit)


def test_a_class_4_family_is_refused():
    with pytest.raises(ValueError, match="section_class"):
        build_section_family(Steel355(), 4)
