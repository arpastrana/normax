import jax.numpy as jnp
import numpy as np
import pytest

from normax.ec3.classification import is_plastic
from normax.ec3.classification import material_factor
from normax.ec3.interaction import CompressionBendingState
from normax.ec3.interaction import MemberResistance
from normax.ec3.interaction import MemberSlenderness
from normax.ec3.interaction import utilization_member
from normax.ec3.material import E_MODULUS
from normax.ec3.material import IMPERFECTION_FACTORS
from normax.ec3.material import Steel
from normax.ec3.resistance import force_critical
from normax.ec3.resistance import moment_resultant
from normax.ec3.resistance import reduction_buckling
from normax.ec3.resistance import resistance_bending_elastic
from normax.ec3.resistance import resistance_bending_plastic
from normax.ec3.resistance import resistance_bending_reduced
from normax.ec3.resistance import resistance_yielding
from normax.ec3.resistance import slenderness_from_force
from normax.ec3.section import TubeCatalogue

# The precondition for P2. The fully-stressed sizing map root-finds on
# utilization minus one, and bisection is only unconditionally safe if that
# residual is strictly monotone in the diameter with a unique root.
#
# CLAUDE.md section 4 argues this for the axial-only case: area goes as the
# square of the diameter and the radius of gyration linearly, so capacity is
# strictly increasing. Admitting bending adds terms, so the claim has to be
# re-established rather than assumed. Every term of the interaction has to fall
# as the diameter grows: the section moduli rise, the reduction factor rises,
# and the axial ratios fall, so the interaction factors fall with them.

YIELD = 355.0
LENGTH_BUCKLING = 4000.0
ALPHA = IMPERFECTION_FACTORS["a"]
C_M = 0.9

RATIO_PLASTIC = 70.0 * float(material_factor(YIELD)) ** 2
RATIO_ELASTIC = 90.0 * float(material_factor(YIELD)) ** 2

DIAMETERS = jnp.linspace(60.0, 900.0, 400)


def member_utilization(
    diameter, axial_force, moment_major, moment_minor, *, section_class
):
    ratio = RATIO_PLASTIC if is_plastic(section_class) else RATIO_ELASTIC
    gross = TubeCatalogue(ratio).tube_at(diameter).area
    inertia = TubeCatalogue(ratio).tube_at(diameter).second_moment
    modulus = (
        TubeCatalogue(ratio).tube_at(diameter).modulus_plastic
        if is_plastic(section_class)
        else TubeCatalogue(ratio).tube_at(diameter).modulus_elastic
    )

    non_dimensional = slenderness_from_force(
        gross,
        Steel(f_y=YIELD),
        force_critical(inertia, LENGTH_BUCKLING, Steel(e_mod=E_MODULUS)),
    )
    reduction = reduction_buckling(non_dimensional, ALPHA)

    return utilization_member(
        CompressionBendingState(axial_force, moment_major, moment_minor, C_M, C_M),
        MemberResistance(reduction, reduction, gross * YIELD, modulus * YIELD),
        MemberSlenderness.about_both_axes(non_dimensional),
        Steel(),
        section_class=section_class,
    )


def cross_section_utilization(diameter, axial_force, moment_major, moment_minor):
    ratio = RATIO_PLASTIC
    gross = TubeCatalogue(ratio).tube_at(diameter).area
    plastic_moment = resistance_bending_plastic(
        TubeCatalogue(ratio).tube_at(diameter).modulus_plastic, Steel(f_y=YIELD)
    )
    axial = axial_force / resistance_yielding(gross, Steel(f_y=YIELD))

    return moment_resultant(moment_major, moment_minor) / resistance_bending_reduced(
        plastic_moment, axial
    )


ACTIONS = [
    (500e3, 0.0, 0.0),
    (500e3, 40e6, 0.0),
    (500e3, 40e6, 15e6),
    (0.0, 40e6, 15e6),
    (900e3, 80e6, 60e6),
    (50e3, 5e6, 5e6),
]


# ---- The member check ---- #


@pytest.mark.parametrize("actions", ACTIONS)
@pytest.mark.parametrize("section_class", [2, 3])
def test_member_utilization_strictly_decreases_with_diameter(section_class, actions):
    values = member_utilization(DIAMETERS, *actions, section_class=section_class)
    steps = jnp.diff(values)

    assert jnp.all(steps < 0.0), f"largest step {jnp.max(steps)} at {actions}"


@pytest.mark.parametrize("section_class", [2, 3])
def test_member_utilization_has_exactly_one_root(section_class):
    # Strict monotonicity plus a sign change is what makes bisection safe.
    values = np.asarray(
        member_utilization(DIAMETERS, 500e3, 40e6, 15e6, section_class=section_class)
    )
    crossings = np.sum(np.diff(np.sign(values - 1.0)) != 0)

    assert values[0] > 1.0
    assert values[-1] < 1.0
    assert crossings == 1


@pytest.mark.parametrize("section_class", [2, 3])
def test_member_utilization_vanishes_for_a_large_enough_member(section_class):
    assert (
        member_utilization(5000.0, 500e3, 40e6, 15e6, section_class=section_class)
        < 1e-2
    )


@pytest.mark.parametrize("section_class", [2, 3])
def test_member_utilization_is_finite_across_the_range(section_class):
    assert jnp.all(
        jnp.isfinite(
            member_utilization(
                DIAMETERS, 500e3, 40e6, 15e6, section_class=section_class
            )
        )
    )


# ---- The cross-section check ---- #


@pytest.mark.parametrize(
    "actions", [(500e3, 40e6, 0.0), (500e3, 40e6, 15e6), (200e3, 60e6, 20e6)]
)
def test_cross_section_utilization_strictly_decreases_with_diameter(actions):
    # The plastic moment grows with the cube of the diameter while the axial
    # ratio falls with its square, so the reduced moment grows on both counts.
    diameters = jnp.linspace(150.0, 900.0, 300)
    values = cross_section_utilization(diameters, *actions)

    assert jnp.all(jnp.diff(values) < 0.0)


def test_reduced_moment_grows_with_diameter():
    diameters = jnp.linspace(150.0, 900.0, 300)
    gross = TubeCatalogue(RATIO_PLASTIC).tube_at(diameters).area
    plastic_moment = resistance_bending_plastic(
        TubeCatalogue(RATIO_PLASTIC).tube_at(diameters).modulus_plastic,
        Steel(f_y=YIELD),
    )
    reduced = resistance_bending_reduced(
        plastic_moment, 500e3 / resistance_yielding(gross, Steel(f_y=YIELD))
    )

    assert jnp.all(jnp.diff(reduced) > 0.0)


# ---- The pieces the monotonicity rests on ---- #


def test_the_reduction_factor_grows_with_diameter():
    gross = TubeCatalogue(RATIO_PLASTIC).tube_at(DIAMETERS).area
    inertia = TubeCatalogue(RATIO_PLASTIC).tube_at(DIAMETERS).second_moment
    reduction = reduction_buckling(
        slenderness_from_force(
            gross,
            Steel(f_y=YIELD),
            force_critical(inertia, LENGTH_BUCKLING, Steel(e_mod=E_MODULUS)),
        ),
        ALPHA,
    )

    assert jnp.all(jnp.diff(reduction) >= 0.0)


def test_slenderness_falls_with_diameter():
    gross = TubeCatalogue(RATIO_PLASTIC).tube_at(DIAMETERS).area
    inertia = TubeCatalogue(RATIO_PLASTIC).tube_at(DIAMETERS).second_moment
    non_dimensional = slenderness_from_force(
        gross,
        Steel(f_y=YIELD),
        force_critical(inertia, LENGTH_BUCKLING, Steel(e_mod=E_MODULUS)),
    )

    assert jnp.all(jnp.diff(non_dimensional) < 0.0)


def test_the_elastic_branch_is_the_more_utilized_of_the_two():
    # Class 3 forfeits the shape factor, so at equal diameter and equal actions
    # it must never look better than Class 2.
    plastic = member_utilization(DIAMETERS, 500e3, 40e6, 15e6, section_class=2)
    elastic = member_utilization(DIAMETERS, 500e3, 40e6, 15e6, section_class=3)

    assert jnp.all(elastic > plastic)


def test_the_two_fixed_ratios_bracket_the_class_three_boundary():
    assert RATIO_PLASTIC < RATIO_ELASTIC
    assert RATIO_ELASTIC == pytest.approx(59.58, rel=1e-3)
    assert RATIO_PLASTIC == pytest.approx(46.34, rel=1e-3)


def test_elastic_and_plastic_moduli_differ_by_the_shape_factor():
    plastic = TubeCatalogue(24.45).tube_at(244.5).modulus_plastic
    elastic = TubeCatalogue(24.45).tube_at(244.5).modulus_elastic

    assert resistance_bending_plastic(
        plastic, Steel(f_y=YIELD)
    ) / resistance_bending_elastic(elastic, Steel(f_y=YIELD)) == pytest.approx(
        1.326, rel=1e-3
    )
