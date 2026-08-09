import jax
import jax.numpy as jnp
import numpy as np
import pytest

from normax.ec3.classification import CLASS_LIMIT_FACTORS
from normax.ec3.classification import class_limits
from normax.ec3.classification import classify
from normax.ec3.classification import epsilon

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
    assert epsilon(f_y) == pytest.approx(eps, abs=ROUNDING)


@pytest.mark.parametrize("f_y, eps, eps_squared", TABLE_5_2)
def test_epsilon_squared_matches_the_table(f_y, eps, eps_squared):
    assert epsilon(f_y) ** 2 == pytest.approx(eps_squared, abs=ROUNDING)


@pytest.mark.parametrize("f_y, eps, eps_squared", TABLE_5_2)
def test_epsilon_is_its_definition(f_y, eps, eps_squared):
    assert epsilon(f_y) == pytest.approx(np.sqrt(235.0 / f_y))


def test_epsilon_is_one_at_the_reference_grade():
    assert epsilon(235.0) == pytest.approx(1.0)


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

    assert classify(0.5 * lower, f_y) == 1
    assert classify(0.5 * (lower + middle), f_y) == 2
    assert classify(0.5 * (middle + upper), f_y) == 3
    assert classify(2.0 * upper, f_y) == 4


@pytest.mark.parametrize("f_y", GRADES)
def test_a_ratio_on_a_limit_takes_the_lower_class(f_y):
    # Table 5.2 states the limits as d/t <= 50 eps^2, so the boundary belongs
    # to the class below it.
    limits = class_limits(f_y)

    for index, limit in enumerate(np.asarray(limits)):
        assert classify(limit, f_y) == index + 1


@pytest.mark.parametrize("f_y", GRADES)
def test_the_fixed_ratio_sits_on_the_class_three_boundary(f_y):
    # CLAUDE.md section 3: d/t is pinned at 90 eps^2 so classification is exact
    # by construction and never needs smoothing.
    assert classify(90.0 * epsilon(f_y) ** 2, f_y) == 3


def test_the_worked_example_section_is_class_one():
    # CHS 244.5 x 10, S355.
    assert classify(244.5 / 10.0, 355.0) == 1


def test_classify_returns_integers():
    assert jnp.issubdtype(classify(24.45, 355.0).dtype, jnp.integer)


def test_classify_vectorizes_over_members():
    lower, middle, upper = np.asarray(class_limits(355.0))
    ratios = jnp.asarray(
        [0.5 * lower, 0.5 * (lower + middle), 0.5 * (middle + upper), 2.0 * upper]
    )

    classes = classify(ratios, 355.0)

    assert classes.shape == (4,)
    assert np.asarray(classes) == pytest.approx([1, 2, 3, 4])


def test_classify_is_jittable():
    # No Python branch on a traced value, so this must trace cleanly.
    assert jax.jit(classify)(24.45, 355.0) == 1
