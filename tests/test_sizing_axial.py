import jax
import jax.numpy as jnp
import numpy as np
import pytest

from normax.ec3.actions import MemberActions
from normax.ec3.classification import classify_section
from normax.ec3.classification import is_plastic
from normax.ec3.material import IMPERFECTION_FACTORS
from normax.ec3.material import Steel
from normax.ec3.resistance import force_critical
from normax.ec3.resistance import reduction_buckling
from normax.ec3.resistance import resistance_buckling
from normax.ec3.resistance import resistance_yielding
from normax.ec3.resistance import slenderness_from_force
from normax.ec3.section import DIAMETER_MINIMUM
from normax.ec3.section import TubeCatalogue
from normax.ec3.sizing import diameter_bracket
from normax.ec3.sizing import diameter_required
from normax.ec3.sizing import mass_of_tubes
from normax.ec3.sizing import utilization_design

# Step 1 of P2: the fully-stressed map with no bending. Every action here has
# zero moment, so the member check collapses to the buckling check of 6.3.1 and
# the cross-section check to the squash check of 6.2.4. Both have closed forms,
# which is what makes this the fixture that de-risks the machinery.

STEEL = Steel()
CATALOGUE = TubeCatalogue.at_class_limit(STEEL, 3)
SECTION_CLASS = 3

LENGTH = 4000.0

# Spanning three decades of compression. The smallest tube in the family
# already carries about 8.3 kN in squash, so lighter forces are covered by the
# floor rather than by the check and are exercised separately.
FORCES = [-1e4, -1e5, -5e5, -2.11e6, -1e7]
LENGTHS = [2000.0, 4000.0, 12000.0, 40000.0]


def sized(
    axial_force,
    buckling_length=LENGTH,
    *,
    section_class=SECTION_CLASS,
    catalogue=CATALOGUE,
):
    return diameter_required(
        MemberActions(axial_force, 0.0, 0.0, 1.0, 1.0), buckling_length, catalogue
    )


def used(
    d,
    axial_force,
    buckling_length=LENGTH,
    *,
    section_class=SECTION_CLASS,
    catalogue=CATALOGUE,
):
    return utilization_design(
        catalogue(d), MemberActions(axial_force, 0.0, 0.0, 1.0, 1.0), buckling_length
    )


# ---- Defaults ---- #


def test_the_default_steel_is_s355():
    assert STEEL.f_y == 355.0
    assert STEEL.e_mod == 210000.0
    assert STEEL.gamma_m0 == 1.0
    assert STEEL.gamma_m1 == 1.0


def test_the_default_density_is_steel_in_tonnes_per_cubic_millimeter():
    assert STEEL.density == pytest.approx(7.85e-9)


def test_the_default_curve_is_a_for_a_hot_finished_tube():
    assert STEEL.alpha == IMPERFECTION_FACTORS["a"]


def test_the_minimum_diameter_is_the_smallest_standard_tube():
    assert DIAMETER_MINIMUM == pytest.approx(21.3)


@pytest.mark.parametrize("section_class", [1, 2, 3])
def test_the_class_limit_constructor_lands_on_its_own_class(section_class):
    # The ratio is set to the limit, and the limits are inclusive upper bounds,
    # so the section classifies as the class it was built for.
    catalogue = TubeCatalogue.at_class_limit(Steel(), section_class)

    assert int(classify_section(catalogue.ratio, 355.0)) == section_class


def test_the_class_limit_constructor_reproduces_the_documented_ratios():
    assert TubeCatalogue.at_class_limit(Steel(), 3).ratio == pytest.approx(
        59.58, rel=1e-3
    )
    assert TubeCatalogue.at_class_limit(Steel(), 2).ratio == pytest.approx(
        46.34, rel=1e-3
    )


def test_only_the_first_two_classes_are_plastic():
    assert is_plastic(1)
    assert is_plastic(2)
    assert not is_plastic(3)


def test_a_class_four_section_is_refused_rather_than_called_elastic():
    # Reporting Class 4 as elastic would run Class 3's clauses on a shell, which
    # takes effective section properties under 6.2.2.5 instead.
    with pytest.raises(ValueError, match="1993-1-6"):
        is_plastic(4)


# ---- The bracket ---- #


@pytest.mark.parametrize("buckling_length", LENGTHS)
@pytest.mark.parametrize("axial_force", FORCES)
def test_the_bracket_is_never_above_the_root(axial_force, buckling_length):
    lower = diameter_bracket(MemberActions(axial_force, 0.0, 0.0), CATALOGUE)

    assert float(sized(axial_force, buckling_length)) >= float(lower) - 1e-9


@pytest.mark.parametrize("axial_force", FORCES)
def test_the_bracket_is_at_or_beyond_full_utilization(axial_force):
    # The lower bound is the squash diameter, where the axial force alone
    # exhausts the section. Buckling can only make that worse.
    lower = diameter_bracket(MemberActions(axial_force, 0.0, 0.0), CATALOGUE)

    assert float(used(lower, axial_force)) >= 1.0 - 1e-9


def test_the_bracket_does_not_carry_the_catalogue_minimum():
    # The smallest tube on offer is a property of the catalogue, not of the
    # check. Folding it into the search would stop the search at a diameter
    # where the check is unsatisfied, which is where the implicit derivative
    # stops being valid, so it is applied to the answer instead.
    lower = diameter_bracket(MemberActions(0.0, 0.0, 0.0), CATALOGUE)

    assert float(lower) < DIAMETER_MINIMUM
    assert float(sized(0.0)) == pytest.approx(DIAMETER_MINIMUM)


def test_the_bracket_is_the_closed_form_squash_diameter():
    # A f_y / gamma_M0 = |N| inverted, with A quadratic in the diameter.
    unit = float(CATALOGUE(1.0).area)
    expected = float(jnp.sqrt(5e5 * STEEL.gamma_m0 / (STEEL.f_y * unit)))

    assert diameter_bracket(MemberActions(-5e5, 0.0, 0.0), CATALOGUE) == pytest.approx(
        expected, rel=1e-12
    )


# ---- The map ---- #


@pytest.mark.parametrize("buckling_length", LENGTHS)
@pytest.mark.parametrize("axial_force", FORCES)
def test_the_sized_member_is_exactly_fully_stressed(axial_force, buckling_length):
    # Invariant 5 of CLAUDE.md. This is the assertion the whole phase exists to
    # make true, and it is checked against the exact clause functions.
    assert float(
        used(sized(axial_force, buckling_length), axial_force, buckling_length)
    ) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("buckling_length", LENGTHS)
@pytest.mark.parametrize("axial_force", FORCES)
def test_the_sized_member_reproduces_the_buckling_resistance(
    axial_force, buckling_length
):
    # Independent of the residual: rebuild N_b,Rd from 6.47 at the solved
    # diameter and check it carries the force.
    d = sized(axial_force, buckling_length)
    gross = CATALOGUE(d).area
    lam = slenderness_from_force(
        gross,
        Steel(f_y=STEEL.f_y),
        force_critical(CATALOGUE(d).second_moment, buckling_length, Steel()),
    )
    resistance = resistance_buckling(
        reduction_buckling(lam, STEEL.alpha), gross, Steel(f_y=STEEL.f_y)
    )

    assert float(resistance) == pytest.approx(abs(axial_force), rel=1e-9)


@pytest.mark.parametrize("axial_force", FORCES)
def test_a_longer_member_never_needs_a_smaller_tube(axial_force):
    sizes = [float(sized(axial_force, length)) for length in LENGTHS]

    assert np.all(np.diff(sizes) >= 0.0)


def test_a_longer_member_needs_a_larger_tube_once_buckling_bites():
    sizes = [float(sized(-5e5, length)) for length in LENGTHS]

    assert np.all(np.diff(sizes) > 0.0)


def test_length_stops_mattering_for_a_stocky_member():
    # 6.3.1.2(3): at or below a slenderness of 0.2 the reduction factor is
    # capped at one and buckling may be ignored, so 6.2.4 sizes the member and
    # its length drops out of the answer entirely. A 10 MN force needs a tube
    # about 737 mm across, whose slenderness only reaches 0.2 near 3.9 m.
    assert float(sized(-1e7, 500.0)) == pytest.approx(
        float(sized(-1e7, 2000.0)), rel=1e-9
    )


@pytest.mark.parametrize("buckling_length", LENGTHS)
def test_a_larger_force_needs_a_larger_tube(buckling_length):
    sizes = [float(sized(force, buckling_length)) for force in FORCES]

    assert np.all(np.diff(sizes) > 0.0)


def test_the_map_agrees_with_a_dense_scan():
    # Brute force: the smallest diameter on a fine grid whose utilization has
    # fallen to one. Independent of the bisection entirely.
    grid = jnp.linspace(50.0, 400.0, 400001)
    values = np.asarray(used(grid, -5e5))
    crossing = float(grid[int(np.argmax(values <= 1.0))])

    assert float(sized(-5e5)) == pytest.approx(crossing, rel=1e-5)


# ---- Tension ---- #


def test_a_tension_member_is_sized_by_the_gross_section():
    # No buckling in tension, so 6.2.3 governs alone and the diameter follows
    # in closed form from A f_y / gamma_M0 = N.
    unit = float(CATALOGUE(1.0).area)
    expected = float(jnp.sqrt(5e5 * STEEL.gamma_m0 / (STEEL.f_y * unit)))

    assert float(sized(5e5)) == pytest.approx(expected, rel=1e-12)


def test_a_tension_member_does_not_care_how_long_it_is():
    assert float(sized(5e5, 2000.0)) == pytest.approx(float(sized(5e5, 40000.0)))


def test_a_tension_member_is_smaller_than_the_same_force_in_compression():
    assert float(sized(5e5)) < float(sized(-5e5))


@pytest.mark.parametrize("axial_force", [1e4, 1e5, 5e5, 5e6])
def test_a_sized_tension_member_reaches_its_plastic_resistance(axial_force):
    d = sized(axial_force)

    assert float(
        resistance_yielding(CATALOGUE(d).area, Steel(f_y=STEEL.f_y))
    ) == pytest.approx(axial_force, rel=1e-9)


# ---- The minimum size ---- #


def test_a_negligible_force_falls_back_to_the_smallest_tube():
    assert float(sized(-1.0)) == pytest.approx(DIAMETER_MINIMUM)


def test_the_minimum_size_leaves_the_member_understressed():
    # The fully-stressed invariant holds only where the root is interior. Where
    # the floor binds the member is deliberately larger than the check needs.
    assert float(used(sized(-1.0), -1.0)) < 1.0


def test_nothing_is_ever_sized_below_the_floor():
    forces = jnp.asarray([-1e-3, -1.0, -1e3, 0.0, 1.0, 1e3])

    assert jnp.all(
        diameter_required(MemberActions(forces, 0.0, 0.0, 1.0, 1.0), LENGTH, CATALOGUE)
        >= DIAMETER_MINIMUM - 1e-12
    )


# ---- Mass ---- #


def test_mass_is_density_times_volume():
    d = jnp.asarray([100.0, 200.0])
    lengths = jnp.asarray([3000.0, 5000.0])
    volume = float(jnp.sum(CATALOGUE(d).area * lengths))

    assert mass_of_tubes(CATALOGUE(d), lengths) == pytest.approx(
        float(STEEL.density) * volume
    )


def test_mass_of_the_fixture_section_matches_the_tabulated_tube():
    # EN 10210 gives CHS 244.5 x 10 as 57.8 kg/m, so a 4 m length is 231 kg.
    catalogue = TubeCatalogue(24.45, 1, STEEL)
    kilograms = float(mass_of_tubes(catalogue(244.5), 4000.0)) * 1e3

    assert kilograms == pytest.approx(4.0 * 57.8, rel=5e-3)


def test_mass_grows_when_any_member_grows():
    lengths = jnp.asarray([3000.0, 5000.0])
    light = mass_of_tubes(CATALOGUE(jnp.asarray([100.0, 200.0])), lengths)
    heavy = mass_of_tubes(CATALOGUE(jnp.asarray([100.0, 201.0])), lengths)

    assert heavy > light


# ---- The class branches ---- #


@pytest.mark.parametrize("section_class", [2, 3])
def test_both_class_branches_size_a_member(section_class):
    catalogue = TubeCatalogue.at_class_limit(STEEL, section_class)
    d = sized(-5e5, section_class=section_class, catalogue=catalogue)

    assert float(
        used(d, -5e5, section_class=section_class, catalogue=catalogue)
    ) == pytest.approx(1.0, abs=1e-9)


def test_the_two_class_branches_agree_without_bending():
    # Classes 1 to 3 all take the gross area in compression, so with no moment
    # the section modulus never enters and the two branches must coincide.
    thin = TubeCatalogue.at_class_limit(STEEL, 3)
    plastic_size = sized(-5e5, section_class=2, catalogue=thin)
    elastic_size = sized(-5e5, section_class=3, catalogue=thin)

    assert float(plastic_size) == pytest.approx(float(elastic_size), rel=1e-12)


# ---- JAX plumbing ---- #


def test_the_map_vectorizes_over_members():
    forces = jnp.asarray(FORCES)
    lengths = jnp.full_like(forces, LENGTH)

    sizes = diameter_required(
        MemberActions(forces, 0.0, 0.0, 1.0, 1.0), lengths, CATALOGUE
    )

    assert sizes.shape == forces.shape
    assert np.asarray(sizes) == pytest.approx(
        [float(sized(force)) for force in FORCES], rel=1e-9
    )


def test_the_map_broadcasts_a_scalar_force_over_many_lengths():
    lengths = jnp.asarray(LENGTHS)

    sizes = diameter_required(
        MemberActions(-5e5, 0.0, 0.0, 1.0, 1.0), lengths, CATALOGUE
    )

    assert sizes.shape == lengths.shape


def test_the_map_is_jittable():
    # Only the moment combination is static now. The class crosses inside the
    # catalogue, where it is part of the tree structure rather than a leaf, so it
    # neither needs marking nor becomes a tracer.
    jitted = jax.jit(diameter_required, static_argnames=("resultant",))

    assert jitted(
        MemberActions(-5e5), LENGTH, CATALOGUE, resultant=True
    ) == pytest.approx(float(sized(-5e5)))


def test_the_map_is_vmappable():
    forces = jnp.asarray(FORCES)

    def one(force):
        return diameter_required(
            MemberActions(force, 0.0, 0.0, 1.0, 1.0), LENGTH, CATALOGUE
        )

    assert np.asarray(jax.vmap(one)(forces)) == pytest.approx(
        [float(sized(force)) for force in FORCES], rel=1e-9
    )


def test_values_are_float64():
    assert sized(-5e5).dtype == jnp.float64
    assert used(200.0, -5e5).dtype == jnp.float64
