import jax
import jax.numpy as jnp
import numpy as np
import pytest

from normax.ec3.classification import CLASS_LIMIT_FACTORS
from normax.ec3.classification import CLASS_LIMIT_TOLERANCE
from normax.ec3.classification import class_limits
from normax.ec3.classification import classify_section
from normax.ec3.classification import material_factor
from normax.ec3.classification import ratio_at_class_limit
from normax.ec3.classification import section_class_at_ratio
from normax.ec3.section import TubeCatalogue

# EN 1993-1-1 Table 5.2, the epsilon row, rounded to 2 d.p. in the standard.
# (f_y, epsilon, epsilon squared)
TABLE_5_2 = [
    (235.0, 1.00, 1.00),
    (275.0, 0.92, 0.85),
    (355.0, 0.81, 0.66),
    (420.0, 0.75, 0.56),
    (460.0, 0.71, 0.51),
]

GRADES = [row[0] for row in TABLE_5_2]

# The table is printed to 2 d.p., so a tabulated value is within 0.005.
ROUNDING = 5e-3


@pytest.mark.parametrize("f_y, eps, eps_squared", TABLE_5_2)
def test_epsilon_matches_the_table(f_y, eps, eps_squared):
    assert material_factor(f_y) == pytest.approx(eps, abs=ROUNDING)


@pytest.mark.parametrize("f_y, eps, eps_squared", TABLE_5_2)
def test_epsilon_squared_matches_the_table(f_y, eps, eps_squared):
    assert material_factor(f_y) ** 2 == pytest.approx(eps_squared, abs=ROUNDING)


@pytest.mark.parametrize("f_y, eps, eps_squared", TABLE_5_2)
def test_epsilon_is_its_definition(f_y, eps, eps_squared):
    assert material_factor(f_y) == pytest.approx(np.sqrt(235.0 / f_y))


def test_epsilon_is_one_at_the_reference_grade():
    assert material_factor(235.0) == pytest.approx(1.0)


def test_class_limit_factors_are_the_tabulated_ones():
    assert CLASS_LIMIT_FACTORS == (50.0, 70.0, 90.0)


@pytest.mark.parametrize("f_y, eps, eps_squared", TABLE_5_2)
def test_class_limits_are_the_factors_times_epsilon_squared(f_y, eps, eps_squared):
    # The limits are 50, 70 and 90 times epsilon SQUARED (Table 5.2, sheet 3).
    # The tabulated epsilon squared carries 0.005 of rounding into each limit.
    limits = np.asarray(class_limits(f_y))
    expected = [factor * eps_squared for factor in CLASS_LIMIT_FACTORS]

    for limit, target, factor in zip(limits, expected, CLASS_LIMIT_FACTORS):
        assert limit == pytest.approx(target, abs=factor * ROUNDING)


@pytest.mark.parametrize("f_y", GRADES)
def test_class_limits_increase(f_y):
    limits = np.asarray(class_limits(f_y))

    assert np.all(np.diff(limits) > 0.0)


@pytest.mark.parametrize("f_y", GRADES)
def test_class_limits_shrink_with_strength(f_y):
    # Higher grade, lower epsilon, tighter slenderness limits.
    assert np.all(np.asarray(class_limits(f_y)) <= np.asarray(class_limits(235.0)))


@pytest.mark.parametrize("f_y", GRADES)
def test_classify_spans_all_four_classes(f_y):
    lower, middle, upper = np.asarray(class_limits(f_y))

    assert classify_section(0.5 * lower, f_y) == 1
    assert classify_section(0.5 * (lower + middle), f_y) == 2
    assert classify_section(0.5 * (middle + upper), f_y) == 3
    assert classify_section(2.0 * upper, f_y) == 4


@pytest.mark.parametrize("f_y", GRADES)
def test_a_ratio_on_a_limit_takes_the_lower_class(f_y):
    # Table 5.2 states the limits as d/t <= 50 eps^2, so the boundary belongs
    # to the class below it.
    limits = class_limits(f_y)

    for index, limit in enumerate(np.asarray(limits)):
        assert classify_section(limit, f_y) == index + 1


@pytest.mark.parametrize("f_y", GRADES)
def test_the_fixed_ratio_sits_on_the_class_three_boundary(f_y):
    # CLAUDE.md section 3: d/t is pinned at 90 eps^2 so classification is exact
    # by construction and never needs smoothing.
    assert classify_section(90.0 * material_factor(f_y) ** 2, f_y) == 3


def test_the_worked_example_section_is_class_one():
    # CHS 244.5 x 10, S355.
    assert classify_section(244.5 / 10.0, 355.0) == 1


def test_classify_returns_integers():
    assert jnp.issubdtype(classify_section(24.45, 355.0).dtype, jnp.integer)


def test_classify_vectorizes_over_members():
    lower, middle, upper = np.asarray(class_limits(355.0))
    ratios = jnp.asarray(
        [0.5 * lower, 0.5 * (lower + middle), 0.5 * (middle + upper), 2.0 * upper]
    )

    classes = classify_section(ratios, 355.0)

    assert classes.shape == (4,)
    assert np.asarray(classes) == pytest.approx([1, 2, 3, 4])


def test_classify_is_jittable():
    # No Python branch on a traced value, so this must trace cleanly.
    assert jax.jit(classify_section)(24.45, 355.0) == 1


# --------------------------------------------------------------------------- #
# The inclusive bound survives a round trip through a wall thickness
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("f_y", GRADES)
@pytest.mark.parametrize("index", [0, 1, 2])
def test_a_section_built_to_a_limit_classifies_there_from_its_geometry(f_y, index):
    # A member is built by pinning d/t, and its wall follows as d over that
    # ratio. Recovering the ratio from the diameter and the wall returns it only
    # to within rounding, and a strict comparison would scatter members built to
    # one ratio across two classes.
    limit = float(np.asarray(class_limits(f_y))[index])
    diameters = jnp.linspace(21.3, 500.0, 64)
    ratios = diameters / TubeCatalogue(limit).tube_at(diameters).thickness

    assert np.all(np.asarray(classify_section(ratios, f_y)) == index + 1)


@pytest.mark.parametrize("f_y", GRADES)
def test_the_round_trip_stays_inside_the_tolerance(f_y):
    limit = float(np.asarray(class_limits(f_y))[2])
    diameters = jnp.linspace(21.3, 500.0, 64)
    ratios = np.asarray(diameters / TubeCatalogue(limit).tube_at(diameters).thickness)

    assert np.max(np.abs(ratios - limit)) / limit < CLASS_LIMIT_TOLERANCE


def test_the_round_trip_straddles_the_limit_at_the_design_grade():
    # The tolerance is not decoration. Which side of a limit the rounding lands
    # on depends on the grade, and at S355 — the grade every design here uses —
    # it lands on both, which without the tolerance splits one section family
    # across two classes.
    limit = float(np.asarray(class_limits(355.0))[2])
    diameters = jnp.linspace(21.3, 500.0, 64)
    ratios = np.asarray(diameters / TubeCatalogue(limit).tube_at(diameters).thickness)

    assert ratios.max() > limit
    assert ratios.min() < limit


@pytest.mark.parametrize("f_y", GRADES)
def test_the_tolerance_does_not_reach_a_neighbouring_class(f_y):
    # Widening the bound must not swallow any ratio geometry would separate.
    lower, middle, upper = np.asarray(class_limits(f_y))

    assert classify_section(lower * (1.0 + 1e-6), f_y) == 2
    assert classify_section(middle * (1.0 + 1e-6), f_y) == 3
    assert classify_section(upper * (1.0 + 1e-6), f_y) == 4


def test_a_real_thin_walled_tube_is_still_class_four():
    # CHS 508 x 8, S355: d/t = 63.5 against a limit of 59.58. Geometry, not
    # rounding, so the tolerance leaves it alone.
    assert classify_section(508.0 / 8.0, 355.0) == 4


# ---- The class as a clause selector ---- #
#
# `classify_section` answers for a whole set of members and returns an array;
# selecting a clause needs one Python integer, which is what these produce. The
# pair below is what makes a class and a section family unable to disagree.


@pytest.mark.parametrize("f_y", GRADES)
@pytest.mark.parametrize("section_class", [1, 2, 3])
def test_the_class_of_a_ratio_inverts_the_ratio_of_a_class(f_y, section_class):
    ratio = ratio_at_class_limit(f_y, section_class)

    assert section_class_at_ratio(ratio, f_y) == section_class


@pytest.mark.parametrize("f_y", GRADES)
@pytest.mark.parametrize("section_class", [1, 2, 3])
def test_a_catalogue_reports_the_class_it_was_built_at(f_y, section_class):
    # The invariant the structural fix rests on: a class read off the family
    # cannot contradict the wall the family gives a member.
    catalogue = TubeCatalogue.at_class_limit(f_y, section_class)

    assert catalogue.section_class(f_y) == section_class


def test_the_class_of_a_ratio_is_a_python_integer():
    # A traced or array value cannot select a clause under jit.
    value = section_class_at_ratio(55.0, 355.0)

    assert type(value) is int


def test_a_class_four_ratio_is_refused():
    with pytest.raises(ValueError, match="1993-1-6"):
        section_class_at_ratio(508.0 / 8.0, 355.0)


def test_one_ratio_per_family_not_one_per_member():
    with pytest.raises(ValueError, match="one ratio"):
        section_class_at_ratio(jnp.asarray([40.0, 55.0]), 355.0)


def test_the_class_cannot_be_taken_of_a_traced_ratio():
    # The honest failure: the class chooses between clauses, so a derivative
    # with respect to the ratio cannot move it.
    with pytest.raises(Exception, match="[Cc]oncret|[Tt]race"):
        jax.jit(lambda ratio: section_class_at_ratio(ratio, 355.0))(55.0)
