import jax
import jax.numpy as jnp
import numpy as np
import pytest

from normax.ec3.interaction import C_M_MINIMUM
from normax.ec3.sizing import DIAMETER_MINIMUM
from normax.ec3.sizing import LIMIT_CROSS_SECTION
from normax.ec3.sizing import LIMIT_MAJOR
from normax.ec3.sizing import LIMIT_MINIMUM_SIZE
from normax.ec3.sizing import LIMIT_MINOR
from normax.ec3.sizing import LIMIT_TENSION
from normax.ec3.sizing import Steel
from normax.ec3.sizing import Tube
from normax.ec3.sizing import diameter
from normax.ec3.sizing import end_moments
from normax.ec3.sizing import envelope
from normax.ec3.sizing import governing
from normax.ec3.sizing import is_plastic
from normax.ec3.sizing import utilization

# Step 2 of P2: the unified residual, under axial force and biaxial bending.
# The check is the larger of the member check of 6.3.3, which the standard
# applies only in compression, and the cross-section check of 6.2.9, which it
# applies always. Neither bounds the other.

STEEL = Steel()
LENGTH = 4000.0

ACTIONS = [
    (-5e5, 0.0, 0.0),
    (-5e5, 40e6, 0.0),
    (-5e5, 40e6, 15e6),
    (0.0, 40e6, 15e6),
    (5e5, 40e6, 15e6),
    (-9e5, 80e6, 60e6),
    (-5e4, 5e6, 5e6),
]

BRANCHES = [(2, True), (3, False)]


def tube_for(cross_section_class):
    return Tube.at_class_limit(STEEL.f_y, cross_section_class)


def sized(actions, l_cr=LENGTH, *, c_m=0.9, cross_section_class=3):
    tube = tube_for(cross_section_class)

    return diameter(
        *actions,
        c_m,
        c_m,
        l_cr,
        STEEL,
        tube,
        plastic=is_plastic(cross_section_class),
    )


def used(d, actions, l_cr=LENGTH, *, c_m=0.9, cross_section_class=3):
    tube = tube_for(cross_section_class)

    return utilization(
        d,
        *actions,
        c_m,
        c_m,
        l_cr,
        STEEL,
        tube,
        plastic=is_plastic(cross_section_class),
    )


# ---- The invariant ---- #


@pytest.mark.parametrize("cross_section_class, _plastic", BRANCHES)
@pytest.mark.parametrize("actions", ACTIONS)
def test_every_sized_member_is_exactly_fully_stressed(
    actions, cross_section_class, _plastic
):
    d = sized(actions, cross_section_class=cross_section_class)
    value = used(d, actions, cross_section_class=cross_section_class)

    assert float(value) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("cross_section_class, _plastic", BRANCHES)
@pytest.mark.parametrize("actions", ACTIONS)
def test_no_sized_member_is_ever_overutilized(actions, cross_section_class, _plastic):
    # The search returns the over-sized end of its final interval rather than
    # its midpoint, so the answer satisfies the check rather than merely
    # approaching it. The tolerance is the precision of evaluating the check
    # itself: recomputing it outside the loop it was tested in rounds
    # differently by a few units in the last place.
    d = sized(actions, cross_section_class=cross_section_class)

    assert (
        float(used(d, actions, cross_section_class=cross_section_class)) <= 1.0 + 1e-12
    )


@pytest.mark.parametrize("cross_section_class, _plastic", BRANCHES)
def test_removing_the_moments_reproduces_the_axial_answer(
    cross_section_class, _plastic
):
    with_moment = sized((-5e5, 1e-12, 0.0), cross_section_class=cross_section_class)
    without = sized((-5e5, 0.0, 0.0), cross_section_class=cross_section_class)

    assert float(with_moment) == pytest.approx(float(without), rel=1e-9)


# ---- The two checks ---- #


def test_the_cross_section_check_can_govern_over_the_member_check():
    # A stocky member with a reversed moment diagram. The equivalent uniform
    # moment factor pulls the member demand below the cross-section demand, so
    # 6.2.9 decides the size and the factor stops mattering altogether. Sizing
    # on 6.3.3 alone would hand back a section 6.2.9 refuses, which is why the
    # residual takes the larger of the two.
    stocky = 500.0
    floored = sized((-1e3, 60e6, 0.0), stocky, c_m=C_M_MINIMUM)
    milder = sized((-1e3, 60e6, 0.0), stocky, c_m=0.7)

    assert float(floored) == pytest.approx(float(milder), rel=1e-9)
    assert (
        float(
            governing(
                floored,
                -1e3,
                60e6,
                0.0,
                C_M_MINIMUM,
                C_M_MINIMUM,
                stocky,
                STEEL,
                tube_for(3),
                plastic=False,
            )
        )
        == LIMIT_CROSS_SECTION
    )


def test_a_tension_member_is_sized_by_the_cross_section_alone():
    tube = tube_for(3)
    d = sized((5e5, 40e6, 15e6))
    code = governing(d, 5e5, 40e6, 15e6, 0.9, 0.9, LENGTH, STEEL, tube, plastic=False)

    assert float(code) == LIMIT_TENSION


def test_a_tension_member_does_not_care_how_long_it_is():
    short = sized((5e5, 40e6, 15e6), 2000.0)
    long = sized((5e5, 40e6, 15e6), 40000.0)

    assert float(short) == pytest.approx(float(long), rel=1e-12)


@pytest.mark.parametrize("cross_section_class, _plastic", BRANCHES)
def test_a_compression_member_is_larger_than_the_same_tension_member(
    cross_section_class, _plastic
):
    compression = sized((-5e5, 40e6, 15e6), cross_section_class=cross_section_class)
    tension = sized((5e5, 40e6, 15e6), cross_section_class=cross_section_class)

    assert float(compression) > float(tension)


# ---- The discontinuity at zero axial force ---- #


def test_the_check_jumps_as_the_axial_force_changes_sign():
    # 6.3.3 is titled "bending and axial compression" and simply does not apply
    # in tension, so a member a whisker into tension is held to a weaker
    # requirement than the same member a whisker into compression. The jump is
    # the standard's, not this package's.
    fixed = 300.0
    below = used(fixed, (-1e-9, 30e6, 30e6), c_m=1.0, cross_section_class=2)
    above = used(fixed, (1e-9, 30e6, 30e6), c_m=1.0, cross_section_class=2)

    assert float(below) / float(above) == pytest.approx(1.131, rel=1e-2)


def test_the_jump_is_worse_on_the_elastic_branch():
    # Table B.1's Class 3 column couples the minor-axis moment with the full
    # factor rather than six tenths of it, so the linear sum it compares against
    # the resultant is larger. The bound is the square root of two.
    fixed = 300.0
    below = used(fixed, (-1e-9, 30e6, 30e6), c_m=1.0, cross_section_class=3)
    above = used(fixed, (1e-9, 30e6, 30e6), c_m=1.0, cross_section_class=3)

    assert float(below) / float(above) == pytest.approx(1.414, rel=1e-2)


@pytest.mark.parametrize("cross_section_class, _plastic", BRANCHES)
def test_there_is_no_jump_under_uniaxial_bending(cross_section_class, _plastic):
    # With one moment the linear sum and the resultant coincide, so the
    # cross-section check governs on both sides and the size is continuous.
    below = sized((-1e-9, 30e6, 0.0), c_m=1.0, cross_section_class=cross_section_class)
    above = sized((1e-9, 30e6, 0.0), c_m=1.0, cross_section_class=cross_section_class)

    assert float(below) == pytest.approx(float(above), rel=1e-9)


def test_the_jump_in_mass_is_worth_reporting():
    # What the discontinuity costs in the objective, which is what an optimizer
    # chattering across it would see.
    below = sized((-1e-9, 30e6, 30e6), c_m=1.0, cross_section_class=3)
    above = sized((1e-9, 30e6, 30e6), c_m=1.0, cross_section_class=3)
    areas = (float(below) / float(above)) ** 2

    assert areas > 1.2


# ---- Monotonicity of the unified residual ---- #


@pytest.mark.parametrize("cross_section_class, _plastic", BRANCHES)
@pytest.mark.parametrize("actions", ACTIONS)
def test_the_unified_check_falls_with_the_diameter(
    actions, cross_section_class, _plastic
):
    # The precondition for bisection, on the composition rather than on either
    # check alone. The larger of two strictly decreasing functions is strictly
    # decreasing, but only if both really are.
    diameters = jnp.linspace(60.0, 900.0, 500)
    values = used(diameters, actions, cross_section_class=cross_section_class)

    assert jnp.all(jnp.diff(values) < 0.0)


@pytest.mark.parametrize("cross_section_class, _plastic", BRANCHES)
@pytest.mark.parametrize("actions", ACTIONS)
def test_the_unified_check_crosses_unity_exactly_once(
    actions, cross_section_class, _plastic
):
    diameters = jnp.linspace(60.0, 900.0, 500)
    values = np.asarray(
        used(diameters, actions, cross_section_class=cross_section_class)
    )
    crossings = int(np.sum(np.diff(np.sign(values - 1.0)) != 0))

    assert crossings <= 1


# ---- The diagnostic ---- #


@pytest.mark.parametrize("cross_section_class, _plastic", BRANCHES)
@pytest.mark.parametrize("actions", ACTIONS)
def test_the_diagnostic_names_a_real_limit_state(
    actions, cross_section_class, _plastic
):
    tube = tube_for(cross_section_class)
    plastic = is_plastic(cross_section_class)
    d = sized(actions, cross_section_class=cross_section_class)
    code = governing(d, *actions, 0.9, 0.9, LENGTH, STEEL, tube, plastic=plastic)

    assert float(code) in {
        LIMIT_MINIMUM_SIZE,
        LIMIT_TENSION,
        LIMIT_CROSS_SECTION,
        LIMIT_MAJOR,
        LIMIT_MINOR,
    }


def test_the_diagnostic_reports_the_minimum_size_ahead_of_any_clause():
    tube = tube_for(3)
    d = sized((-1.0, 0.0, 0.0))
    code = governing(d, -1.0, 0.0, 0.0, 0.9, 0.9, LENGTH, STEEL, tube, plastic=False)

    assert float(d) == pytest.approx(DIAMETER_MINIMUM)
    assert float(code) == LIMIT_MINIMUM_SIZE


def test_a_slender_compression_member_is_governed_by_the_member_check():
    tube = tube_for(3)
    d = sized((-5e5, 40e6, 15e6), 12000.0)
    code = governing(d, -5e5, 40e6, 15e6, 0.9, 0.9, 12000.0, STEEL, tube, plastic=False)

    assert float(code) in {LIMIT_MAJOR, LIMIT_MINOR}


def test_the_named_equation_is_the_one_that_actually_wins():
    # Eq. 6.61 weights the two moments the opposite way to 6.62, so the larger
    # moment about the major axis should send the check to 6.61.
    tube = tube_for(2)
    d = sized((-5e5, 80e6, 5e6), 12000.0, cross_section_class=2)
    code = governing(d, -5e5, 80e6, 5e6, 0.9, 0.9, 12000.0, STEEL, tube, plastic=True)

    assert float(code) == LIMIT_MAJOR


# ---- Table B.3, the linear row ---- #


def test_a_uniform_moment_gives_a_factor_of_one():
    design, factor = end_moments(40e6, 40e6)

    assert float(design) == pytest.approx(40e6)
    assert float(factor) == pytest.approx(1.0)


def test_a_symmetric_reversal_floors_the_factor():
    _, factor = end_moments(40e6, -40e6)

    assert float(factor) == pytest.approx(C_M_MINIMUM)


def test_the_design_moment_is_the_larger_end():
    design, _ = end_moments(-40e6, 15e6)

    assert float(design) == pytest.approx(40e6)


def test_the_factor_follows_the_first_row_of_table_b_three():
    design, factor = end_moments(100e6, 50e6)

    assert float(design) == pytest.approx(100e6)
    assert float(factor) == pytest.approx(0.6 + 0.4 * 0.5)


def test_an_unbent_member_is_given_a_factor_of_one():
    design, factor = end_moments(0.0, 0.0)

    assert float(design) == 0.0
    assert float(factor) == pytest.approx(1.0)


def test_the_factor_never_leaves_its_range():
    first = jnp.linspace(-100e6, 100e6, 401)
    _, factor = end_moments(first, 100e6)

    assert jnp.all(factor >= C_M_MINIMUM)
    assert jnp.all(factor <= 1.0)


# ---- The load-case envelope ---- #


def test_the_envelope_covers_every_case():
    # Exactly the largest in the limit, so the comparison carries the rounding
    # of the logarithm and its inverse.
    cases = jnp.asarray([[100.0, 300.0], [200.0, 150.0], [50.0, 220.0]])
    largest = jnp.max(cases, axis=0)

    assert jnp.all(envelope(cases, 20.0) >= largest * (1.0 - 1e-12))


def test_the_envelope_approaches_the_largest_case():
    cases = jnp.asarray([[100.0, 300.0], [200.0, 150.0], [50.0, 220.0]])
    sharp = envelope(cases, 2000.0)

    assert np.asarray(sharp) == pytest.approx(
        np.asarray(jnp.max(cases, axis=0)), rel=1e-3
    )


def test_the_envelope_respects_its_bound():
    # In the logarithm the smooth maximum exceeds the true one by at most the
    # logarithm of the number of cases over the sharpness.
    cases = jnp.asarray([[100.0, 300.0], [200.0, 150.0], [50.0, 220.0]])
    beta = 20.0
    largest = jnp.max(cases, axis=0)
    slack = jnp.log(envelope(cases, beta)) - jnp.log(largest)

    assert jnp.all(slack >= 0.0)
    assert jnp.all(slack <= jnp.log(cases.shape[0]) / beta + 1e-12)


def test_the_envelope_tightens_as_it_sharpens():
    # Past a sharpness of about fifty the smoothing is already below the
    # precision of the numbers, so the annealing range that does any work is
    # the low end.
    cases = jnp.asarray([[100.0, 300.0], [200.0, 150.0], [50.0, 220.0]])
    values = [float(envelope(cases, beta)[0]) for beta in (2.0, 5.0, 10.0, 20.0)]

    assert np.all(np.diff(values) < 0.0)


def test_the_envelope_is_differentiable():
    cases = jnp.asarray([[100.0, 300.0], [200.0, 150.0]])
    gradient = jax.grad(lambda c: jnp.sum(envelope(c, 50.0)))(cases)

    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.all(gradient >= 0.0)


# ---- JAX plumbing ---- #


def test_the_map_vectorizes_over_members():
    forces = jnp.asarray([-5e5, -9e5, 5e5])
    moments = jnp.asarray([40e6, 80e6, 40e6])
    tube = tube_for(3)

    sizes = diameter(forces, moments, 0.0, 0.9, 0.9, LENGTH, STEEL, tube, plastic=False)

    assert sizes.shape == forces.shape
    assert jnp.all(jnp.isfinite(sizes))


def test_the_map_is_jittable_with_moments():
    jitted = jax.jit(diameter, static_argnames=("plastic",))
    tube = tube_for(3)

    assert jitted(
        -5e5, 40e6, 15e6, 0.9, 0.9, LENGTH, STEEL, tube, plastic=False
    ) == pytest.approx(float(sized((-5e5, 40e6, 15e6))))


# ---- The two readings of Eq. 6.42 ---- #
#
# docs/clauses.md records the disagreement: the Designers' Guide says 6.2.9.2
# permits "only a linear interaction of stresses", the ECCS says the stress is
# "evaluated by an elastic stress analysis". Both readings are implemented so
# the gap is a number rather than an argument. Only the elastic branch is
# affected; on the plastic branch Eq. 6.41 takes both exponents as two for a
# circular hollow section and the resultant is exact algebra.


def sized_reading(actions, l_cr=LENGTH, *, resultant, c_m=0.9):
    return diameter(
        *actions,
        c_m,
        c_m,
        l_cr,
        STEEL,
        tube_for(3),
        plastic=False,
        resultant=resultant,
    )


@pytest.mark.parametrize("actions", ACTIONS)
def test_the_linear_sum_never_asks_for_less_than_the_resultant(actions):
    resultant = sized_reading(actions, resultant=True)
    summed = sized_reading(actions, resultant=False)

    assert float(summed) >= float(resultant) - 1e-9


@pytest.mark.parametrize("actions", ACTIONS)
def test_both_readings_are_exactly_fully_stressed(actions):
    # Whichever reading is chosen, the member satisfies that reading exactly.
    for resultant in (True, False):
        d = sized_reading(actions, resultant=resultant)
        value = utilization(
            d,
            *actions,
            0.9,
            0.9,
            LENGTH,
            STEEL,
            tube_for(3),
            plastic=False,
            resultant=resultant,
        )

        assert float(value) == pytest.approx(1.0, abs=1e-9)


def test_the_two_readings_coincide_under_uniaxial_bending():
    # With one moment the sum and the resultant are the same number, so the
    # disagreement cannot show and the reading stops mattering.
    resultant = sized_reading((5e5, 40e6, 0.0), resultant=True)
    summed = sized_reading((5e5, 40e6, 0.0), resultant=False)

    assert float(summed) == pytest.approx(float(resultant), rel=1e-12)


def test_the_gap_is_widest_in_pure_bending():
    # No axial term to dilute it, so the moment term carries the full square
    # root of two and the diameter carries its cube root.
    resultant = sized_reading((0.0, 40e6, 40e6), resultant=True)
    summed = sized_reading((0.0, 40e6, 40e6), resultant=False)

    # The moment term grows by the square root of two, and the modulus it is
    # carried by goes as the cube of the diameter, so the diameter grows by the
    # sixth root of two and the area by its cube root.
    assert float(summed) / float(resultant) == pytest.approx(2.0 ** (1 / 6), rel=1e-3)
    assert (float(summed) / float(resultant)) ** 2 - 1.0 == pytest.approx(
        0.26, abs=0.01
    )


def test_the_reading_cannot_bite_where_the_member_check_governs():
    # Eq. 6.61 already sums the two moments linearly, so where it governs the
    # cross-section reading changes nothing.
    slender = 4000.0
    resultant = sized_reading((-5e5, 40e6, 40e6), slender, resultant=True)
    summed = sized_reading((-5e5, 40e6, 40e6), slender, resultant=False)
    code = governing(
        resultant,
        -5e5,
        40e6,
        40e6,
        0.9,
        0.9,
        slender,
        STEEL,
        tube_for(3),
        plastic=False,
    )

    assert float(code) in {LIMIT_MAJOR, LIMIT_MINOR}
    assert float(summed) == pytest.approx(float(resultant), rel=1e-12)


def test_the_plastic_branch_ignores_the_reading():
    # 6.2.9.1(6) with both exponents at two is exact for a circular hollow
    # section, so there is nothing to choose there.
    kept = diameter(
        -5e5,
        40e6,
        40e6,
        0.9,
        0.9,
        LENGTH,
        STEEL,
        tube_for(2),
        plastic=True,
        resultant=True,
    )
    dropped = diameter(
        -5e5,
        40e6,
        40e6,
        0.9,
        0.9,
        LENGTH,
        STEEL,
        tube_for(2),
        plastic=True,
        resultant=False,
    )

    assert float(kept) == pytest.approx(float(dropped), rel=1e-14)


def test_both_readings_are_differentiable():
    for resultant in (True, False):
        gradient = jax.grad(
            lambda n: diameter(
                n,
                40e6,
                40e6,
                0.9,
                0.9,
                LENGTH,
                STEEL,
                tube_for(3),
                plastic=False,
                resultant=resultant,
            )
        )(-5e5)

        assert jnp.isfinite(gradient)


# --------------------------------------------------------------------------- #
# The top of the search interval is assumed, so it is checked
# --------------------------------------------------------------------------- #
def test_a_root_above_the_search_interval_returns_nan_not_a_diameter():
    # The lower end of the interval brackets by construction; the upper end is
    # assumed. Were the root above it, the untested top would come back looking
    # like an answer, and it would fail the very check it claims to satisfy.
    steel = Steel()
    tube = Tube.at_class_limit(steel.f_y, 3)

    sized = diameter(-1e5, 0.0, 0.0, 1.0, 1.0, 1e30, steel, tube, plastic=False)

    assert jnp.isnan(sized)


def test_the_ceiling_is_out_of_physical_reach():
    # Twelve orders separate the two ends of the interval, so only a buckling
    # length of order 1e28 mm can push the root past it. Anything a structure
    # could have still finds its root.
    steel = Steel()
    tube = Tube.at_class_limit(steel.f_y, 3)

    for l_cr in (1e3, 1e6, 1e12, 1e18, 1e24):
        sized = diameter(-1e5, 0.0, 0.0, 1.0, 1.0, l_cr, steel, tube, plastic=False)
        used = utilization(
            sized, -1e5, 0.0, 0.0, 1.0, 1.0, l_cr, steel, tube, plastic=False
        )

        assert not jnp.isnan(sized)
        assert float(used) == pytest.approx(1.0, abs=1e-9)


def test_the_guard_does_not_fire_on_a_member_carrying_nothing():
    # A flat check has no root but never exceeds one, so the interval is not the
    # reason it has no root and the catalogue decides the size.
    steel = Steel()
    tube = Tube.at_class_limit(steel.f_y, 3)

    sized = diameter(0.0, 0.0, 0.0, 1.0, 1.0, 1000.0, steel, tube, plastic=False)

    assert not jnp.isnan(sized)
    assert float(sized) == pytest.approx(float(tube.diameter_min), rel=1e-12)


def test_a_failed_bracket_poisons_the_gradient_too():
    # Loud in both passes: there is no valid design, so there is no sensitivity
    # of one either.
    steel = Steel()
    tube = Tube.at_class_limit(steel.f_y, 3)

    def size(n_ed):
        return diameter(n_ed, 0.0, 0.0, 1.0, 1.0, 1e30, steel, tube, plastic=False)

    assert jnp.isnan(jax.grad(size)(-1e5))


def test_the_guard_costs_one_evaluation_and_changes_no_answer():
    # Every ordinary case must return exactly what it returned before the guard.
    steel = Steel()
    tube = Tube.at_class_limit(steel.f_y, 3)

    for n_ed, m_ed, l_cr in (
        (-1e4, 0.0, 1000.0),
        (-5e5, 2e6, 4000.0),
        (1e5, 0.0, 4000.0),
        (-2e6, 5e7, 12000.0),
    ):
        sized = diameter(n_ed, m_ed, 0.0, 1.0, 1.0, l_cr, steel, tube, plastic=False)
        used = utilization(
            sized, n_ed, m_ed, 0.0, 1.0, 1.0, l_cr, steel, tube, plastic=False
        )

        assert not jnp.isnan(sized)
        assert float(used) == pytest.approx(1.0, abs=1e-9)
