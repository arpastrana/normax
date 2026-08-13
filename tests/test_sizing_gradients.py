import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from normax.ec3.actions import MemberActions
from normax.ec3.adjoint import derivative_force
from normax.ec3.adjoint import derivative_force_tension
from normax.ec3.adjoint import derivative_length
from normax.ec3.adjoint import diameter_tension
from normax.ec3.adjoint import reduction_buckling_derivative
from normax.ec3.material import IMPERFECTION_FACTORS
from normax.ec3.material import Steel
from normax.ec3.resistance import reduction_buckling
from normax.ec3.section import TubeCatalogue
from normax.ec3.sizing import diameter_required
from normax.ec3.sizing import mass_of_tubes

# CLAUDE.md invariant 1: every derivative rule ships with a check_grads test.
#
# The sizing map is differentiated by the implicit function theorem, so its
# derivative is only as good as the claim that the check is exactly satisfied at
# the returned diameter. Four oracles have to agree: the forward tangent, the
# reverse transposition of it, the closed forms of adjoint.py, and central
# differences.

STEEL = Steel()
CATALOGUE = TubeCatalogue.at_class_limit(STEEL.f_y, 3)
SECTION_CLASS = 3

LENGTH = 4000.0
FORCE = -5e5

# check_grads perturbs by a small ABSOLUTE step, so an argument of 1e7 is
# perturbed by a relative 1e-11 and its central difference is cancellation
# noise. The map is therefore wrapped so that every argument it sees is of
# order one: forces in kilonewtons, moments in kilonewton-meters, lengths in
# meters. Nothing about the map changes, only the units it is probed in.
KILONEWTON = 1e3
KILONEWTON_METER = 1e6
METER = 1e3


def scaled(force_kn, moment_y_knm, moment_z_knm, length_m, moment_factor=0.9):
    return diameter_required(
        MemberActions(
            force_kn * KILONEWTON,
            moment_y_knm * KILONEWTON_METER,
            moment_z_knm * KILONEWTON_METER,
            moment_factor,
            moment_factor,
        ),
        length_m * METER,
        STEEL,
        CATALOGUE,
        section_class=SECTION_CLASS,
    )


def raw(
    axial_force=FORCE,
    moment_major=0.0,
    moment_minor=0.0,
    buckling_length=LENGTH,
    moment_factor=0.9,
    catalogue=CATALOGUE,
):
    return diameter_required(
        MemberActions(
            axial_force, moment_major, moment_minor, moment_factor, moment_factor
        ),
        buckling_length,
        STEEL,
        catalogue,
        section_class=SECTION_CLASS,
    )


def central(f, x, step):
    return (f(x + step) - f(x - step)) / (2.0 * step)


# ---- The reduction factor's slope ---- #


@pytest.mark.parametrize("curve", ["a0", "a", "b", "c", "d"])
@pytest.mark.parametrize("lam", [0.3, 0.6, 1.0, 1.5, 2.5])
def test_the_closed_form_slope_matches_autodiff(lam, curve):
    alpha = IMPERFECTION_FACTORS[curve]

    assert float(reduction_buckling_derivative(lam, alpha)) == pytest.approx(
        float(jax.grad(reduction_buckling)(lam, alpha)), rel=1e-10
    )


@pytest.mark.parametrize("lam", [0.05, 0.1, 0.19])
def test_the_slope_vanishes_below_the_offset(lam):
    # 6.3.1.2(3) caps the factor at one, so the curve is flat there.
    assert float(reduction_buckling_derivative(lam, 0.21)) == 0.0


def test_the_slope_is_never_positive():
    values = reduction_buckling_derivative(jnp.linspace(0.21, 5.0, 500), 0.21)

    assert jnp.all(values < 0.0)


# ---- The axial map against its closed form ---- #


@pytest.mark.parametrize("buckling_length", [2000.0, 4000.0, 12000.0])
@pytest.mark.parametrize("axial_force", [-1e4, -1e5, -5e5, -2e6])
def test_the_force_sensitivity_matches_the_closed_form(axial_force, buckling_length):
    # The map differentiates the check and inverts the implicit part. This
    # expression was derived on paper instead. They must agree.
    d = raw(axial_force=axial_force, buckling_length=buckling_length)
    automatic = jax.grad(
        lambda force: raw(axial_force=force, buckling_length=buckling_length)
    )(axial_force)
    closed = derivative_force(d, axial_force, buckling_length, STEEL, CATALOGUE)

    assert float(automatic) == pytest.approx(float(closed), rel=1e-9)


@pytest.mark.parametrize("buckling_length", [2000.0, 4000.0, 12000.0])
@pytest.mark.parametrize("axial_force", [-1e4, -1e5, -5e5, -2e6])
def test_the_length_sensitivity_matches_the_closed_form(axial_force, buckling_length):
    d = raw(axial_force=axial_force, buckling_length=buckling_length)
    automatic = jax.grad(
        lambda length: raw(axial_force=axial_force, buckling_length=length)
    )(buckling_length)
    closed = derivative_length(d, axial_force, buckling_length, STEEL, CATALOGUE)

    assert float(automatic) == pytest.approx(float(closed), rel=1e-9)


@pytest.mark.parametrize("axial_force", [1e4, 1e5, 5e5, 5e6])
def test_the_tension_branch_matches_its_closed_form(axial_force):
    assert float(raw(axial_force=axial_force)) == pytest.approx(
        float(diameter_tension(axial_force, STEEL, CATALOGUE)), rel=1e-12
    )

    automatic = jax.grad(lambda force: raw(axial_force=force))(axial_force)

    assert float(automatic) == pytest.approx(
        float(derivative_force_tension(axial_force, STEEL, CATALOGUE)), rel=1e-9
    )


def test_a_tension_member_has_no_length_sensitivity():
    assert (
        float(
            jax.grad(lambda length: raw(axial_force=5e5, buckling_length=length))(
                LENGTH
            )
        )
        == 0.0
    )


def test_a_stocky_member_has_no_length_sensitivity():
    # Below the offset slenderness the reduction factor is capped and length
    # leaves the answer, so the sensitivity is exactly zero rather than small.
    assert (
        float(
            jax.grad(lambda length: raw(axial_force=-1e7, buckling_length=length))(
                500.0
            )
        )
        == 0.0
    )


# ---- Central differences, at engineering scale ---- #


@pytest.mark.parametrize(
    "label, index, value, step",
    [
        ("force", 0, -5e5, 1e-1),
        ("moment_major", 1, 4e7, 1e4),
        ("moment_minor", 2, 1.5e7, 1e4),
        ("length", 3, 4000.0, 1e-3),
    ],
)
def test_every_action_agrees_with_a_central_difference(label, index, value, step):
    # The step is chosen per argument, since one absolute step cannot serve
    # arguments spanning newtons to newton-millimeters.
    base = [-5e5, 4e7, 1.5e7, 4000.0]

    def at(x):
        actions = list(base)
        actions[index] = x

        return raw(
            axial_force=actions[0],
            moment_major=actions[1],
            moment_minor=actions[2],
            buckling_length=actions[3],
        )

    automatic = float(jax.grad(at)(value))
    numeric = float(central(at, value, step))

    assert automatic == pytest.approx(numeric, rel=1e-6), label


# ---- check_grads, on arguments of order one ---- #


def test_check_grads_passes_on_the_scaled_map():
    check_grads(scaled, (-500.0, 40.0, 15.0, 4.0), order=1, modes=("rev",))


def test_check_grads_passes_in_forward_mode():
    # A tangent rule rather than an adjoint, so forward mode is available and
    # reverse mode is its transposition. A custom_vjp would refuse this.
    check_grads(scaled, (-500.0, 40.0, 15.0, 4.0), order=1, modes=("fwd",))


def test_check_grads_passes_on_a_tension_member():
    check_grads(scaled, (500.0, 40.0, 15.0, 4.0), order=1, modes=("rev",))


def test_check_grads_passes_on_the_axial_only_map():
    # The moments are held rather than probed. The check reads their magnitude,
    # so at exactly zero its derivative is one-sided while a central difference
    # straddles the kink and reports nothing. That kink is genuine and is pinned
    # by the two tests below rather than papered over here.
    def axial(force_kn, length_m):
        return scaled(force_kn, 0.0, 0.0, length_m)

    check_grads(axial, (-500.0, 4.0), order=1, modes=("rev",))


def test_the_moment_sensitivity_at_exactly_zero_moment_understates_the_slope():
    # Two corners meet at zero moment and they are not resolved the same way.
    # The member check reads each moment's magnitude and reports a one-sided
    # slope; the cross-section check reads their resultant, which has a cone
    # point there, and reports the symmetric slope of zero. The total therefore
    # sits strictly between nothing and the true right-hand slope.
    #
    # This is confined to exactly zero and is harmless in the pipeline, where a
    # moment is never identically zero, but it is why the axial-only gradcheck
    # above holds the moments rather than probing them.
    def at(moment_major):
        return float(raw(moment_major=moment_major))

    step = 1e2
    one_sided = (at(step) - at(0.0)) / step
    corner = float(jax.grad(lambda m: raw(moment_major=m))(0.0))

    assert 0.0 < corner < one_sided


def test_the_moment_sensitivity_recovers_immediately_off_the_corner():
    # A hair away from zero both checks are ordinary, and the gradient agrees
    # with a central difference again.
    def at(moment_major):
        return raw(moment_major=moment_major)

    automatic = float(jax.grad(at)(1e5))
    numeric = float(central(lambda m: float(at(m)), 1e5, 1e2))

    assert automatic == pytest.approx(numeric, rel=1e-6)


def test_a_central_difference_across_the_corner_reports_nothing():
    # The other half of the same fact, and the reason the moments are held
    # fixed in the axial-only gradcheck above rather than probed.
    def at(moment_major):
        return float(raw(moment_major=moment_major))

    assert (at(1e2) - at(-1e2)) / 2e2 == pytest.approx(0.0, abs=1e-15)


def test_the_sizing_map_is_even_in_the_sign_of_a_moment():
    assert float(raw(moment_major=4e7)) == pytest.approx(
        float(raw(moment_major=-4e7)), rel=1e-12
    )


def test_check_grads_passes_through_the_mass_objective():
    def objective(force_kn, length_m):
        sizes = scaled(force_kn, 40.0, 15.0, length_m)

        return mass_of_tubes(CATALOGUE.tube_at(sizes), length_m * METER, STEEL)

    check_grads(objective, (-500.0, 4.0), order=1, modes=("rev",))


# ---- Forward and reverse are the same derivative ---- #


@pytest.mark.parametrize(
    "actions",
    [
        (-500.0, 40.0, 15.0, 4.0),
        (-500.0, 0.0, 0.0, 4.0),
        (500.0, 40.0, 15.0, 4.0),
        (-900.0, 80.0, 60.0, 12.0),
    ],
)
def test_forward_and_reverse_agree(actions):
    for index in range(4):

        def at(x, index=index):
            probed = list(actions)
            probed[index] = x

            return scaled(*probed)

        forward = float(jax.jacfwd(at)(actions[index]))
        reverse = float(jax.grad(at)(actions[index]))

        assert forward == pytest.approx(reverse, rel=1e-12)


# ---- Gradients through the objective ---- #


def test_the_mass_gradient_is_finite_and_signed():
    forces = jnp.asarray([-5e5, -9e5, 5e5])
    lengths = jnp.asarray([4000.0, 6000.0, 3000.0])

    def objective(axial_force):
        sizes = diameter_required(
            MemberActions(axial_force, 4e7, 1.5e7, 0.9, 0.9),
            lengths,
            STEEL,
            CATALOGUE,
            section_class=SECTION_CLASS,
        )

        return mass_of_tubes(CATALOGUE.tube_at(sizes), lengths, STEEL)

    gradient = jax.grad(objective)(forces)

    assert jnp.all(jnp.isfinite(gradient))
    # More compression means a heavier member, so mass falls as the force rises.
    assert jnp.all(gradient[:2] < 0.0)


def test_the_mass_gradient_survives_a_member_at_the_minimum_size():
    # A member the catalogue floor decides contributes no sensitivity, and must
    # not contribute an undefined one either.
    forces = jnp.asarray([-5e5, -1.0])
    lengths = jnp.asarray([4000.0, 4000.0])

    def objective(axial_force):
        sizes = diameter_required(
            MemberActions(axial_force, 0.0, 0.0, 0.9, 0.9),
            lengths,
            STEEL,
            CATALOGUE,
            section_class=SECTION_CLASS,
        )

        return mass_of_tubes(CATALOGUE.tube_at(sizes), lengths, STEEL)

    gradient = jax.grad(objective)(forces)

    assert jnp.all(jnp.isfinite(gradient))
    assert float(gradient[1]) == 0.0


def test_a_member_with_no_actions_has_no_gradient():
    def objective(axial_force):
        return jnp.sum(
            diameter_required(
                MemberActions(axial_force, 0.0, 0.0, 0.9, 0.9),
                LENGTH,
                STEEL,
                CATALOGUE,
                section_class=SECTION_CLASS,
            )
        )

    gradient = jax.grad(objective)(0.0)

    assert jnp.isfinite(gradient)
    assert float(gradient) == 0.0


# ---- Gradients with respect to the design basis ---- #


def test_the_map_is_differentiable_in_the_yield_strength():
    def at(f_y):
        steel = Steel(f_y, STEEL.e_mod, STEEL.density, STEEL.gamma_m0, STEEL.gamma_m1)

        return diameter_required(
            MemberActions(FORCE, 4e7, 1.5e7, 0.9, 0.9),
            LENGTH,
            steel,
            CATALOGUE,
            section_class=SECTION_CLASS,
        )

    gradient = float(jax.grad(at)(355.0))

    assert np.isfinite(gradient)
    assert gradient < 0.0
    assert gradient == pytest.approx(float(central(at, 355.0, 1e-4)), rel=1e-6)


def test_the_map_is_differentiable_in_the_wall_proportion():
    # The experiment that frees the diameter-to-thickness ratio needs this, and
    # it comes free from differentiating the check rather than the solver.
    def at(ratio):
        return diameter_required(
            MemberActions(FORCE, 4e7, 1.5e7, 0.9, 0.9),
            LENGTH,
            STEEL,
            TubeCatalogue(ratio, CATALOGUE.diameter_min),
            section_class=SECTION_CLASS,
        )

    gradient = float(jax.grad(at)(float(CATALOGUE.ratio)))

    assert np.isfinite(gradient)
    assert gradient == pytest.approx(
        float(central(at, float(CATALOGUE.ratio), 1e-4)), rel=1e-6
    )


# ---- JAX plumbing ---- #


def test_the_gradient_is_jittable():
    def objective(axial_force):
        return raw(axial_force=axial_force)

    jitted = jax.jit(jax.grad(objective))

    assert float(jitted(FORCE)) == pytest.approx(float(jax.grad(objective)(FORCE)))


def test_the_gradient_vmaps_over_members():
    forces = jnp.asarray([-1e5, -5e5, -2e6])

    def one(force):
        return raw(axial_force=force)

    batched = jax.vmap(jax.grad(one))(forces)

    assert batched.shape == forces.shape
    assert np.asarray(batched) == pytest.approx(
        [float(jax.grad(one)(force)) for force in forces], rel=1e-9
    )
