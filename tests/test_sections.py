import jax
import jax.numpy as jnp
import numpy as np
from ec3x.material import Steel
from ec3x.section import TubeCatalogue

from normax.materials import SteelGrade
from normax.sections import MemberSections
from normax.sections import TubeFamily

# The seven derived properties both libraries state. The two must agree bit for
# bit, or the mass normax weighs and the resistance ec3x checks describe two
# different tubes; this file is the drift alarm the sections doc promises.
PROPERTIES = (
    "ratio",
    "diameter_inner",
    "area",
    "second_moment",
    "radius_of_gyration",
    "modulus_elastic",
    "modulus_plastic",
)

RATIO = 59.5934

DIAMETERS = jnp.asarray([21.3, 100.0, 244.5, 508.0])


def tube_pair(diameters):
    """
    The same tubes, stated by both libraries at one wall proportion.
    """
    catalogue = TubeCatalogue(RATIO, 3, Steel())
    family = TubeFamily(RATIO, SteelGrade())

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
    # No static leaf and no registered treedef machinery: every array a design
    # carries is an ordinary leaf, which is what lets the container be a primal
    # of any traced map without ceremony.
    _, sections = tube_pair(DIAMETERS)
    leaves = jax.tree.leaves(sections)

    assert len(leaves) == 2 + len(SteelGrade._fields)


def test_the_load_case_axis_is_variadic():
    stacked = jnp.stack([DIAMETERS, 2.0 * DIAMETERS])
    _, sections = tube_pair(stacked)

    assert sections.area.shape == stacked.shape


def test_the_geometry_is_differentiable_through_the_family():
    family = TubeFamily(RATIO, SteelGrade())

    def total_area(diameter):
        return jnp.sum(family(diameter).area)

    gradient = jax.grad(total_area)(DIAMETERS)

    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert bool(jnp.all(gradient > 0.0))
