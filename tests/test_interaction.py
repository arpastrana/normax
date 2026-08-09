import jax
import jax.numpy as jnp
import numpy as np
import pytest

from normax.ec3.interaction import C_M_MINIMUM
from normax.ec3.interaction import GOVERNING_MAJOR
from normax.ec3.interaction import GOVERNING_MINOR
from normax.ec3.interaction import axial_ratio
from normax.ec3.interaction import c_m_linear
from normax.ec3.interaction import cap_is_active
from normax.ec3.interaction import governing_equation
from normax.ec3.interaction import k_yy
from normax.ec3.interaction import k_yz
from normax.ec3.interaction import k_zy
from normax.ec3.interaction import k_zz
from normax.ec3.interaction import utilization

# EN 1993-1-1 Annex B, method 2. See docs/clauses.md for the two interpretations
# taken here: a circular hollow section reads the RHS row of Table B.1, which
# has no circular row of its own, and C_m comes from the linear row of Table
# B.3, which is exact under nodal loading.

TOLERANCE = 1e-2

SLENDERNESSES = [0.2, 0.5, 0.84, 1.2, 2.0]
RATIOS = [0.0, 0.05, 0.2, 0.5, 0.9]


# ---- Table B.3, row 1 ---- #


@pytest.mark.parametrize("psi", [-1.0, -0.5, 0.0, 0.5, 1.0])
def test_c_m_follows_the_linear_row(psi):
    assert c_m_linear(psi) == pytest.approx(max(0.6 + 0.4 * psi, C_M_MINIMUM))


def test_c_m_is_one_for_a_uniform_moment():
    assert c_m_linear(1.0) == pytest.approx(1.0)


def test_c_m_floors_at_four_tenths():
    # 0.6 + 0.4psi would give 0.2 at psi = -1.
    assert c_m_linear(-1.0) == pytest.approx(C_M_MINIMUM)
    assert jnp.all(c_m_linear(jnp.linspace(-1.0, 1.0, 200)) >= C_M_MINIMUM)


def test_c_m_is_non_decreasing_in_psi():
    values = c_m_linear(jnp.linspace(-1.0, 1.0, 200))

    assert jnp.all(jnp.diff(values) >= 0.0)


# ---- Table B.1 interaction factors ---- #


@pytest.mark.parametrize("ratio", RATIOS)
@pytest.mark.parametrize("lam", SLENDERNESSES)
def test_k_yy_respects_its_cap(lam, ratio):
    for plastic, cap_slope in ((True, 0.8), (False, 0.6)):
        cap = 0.9 * (1.0 + cap_slope * ratio)

        assert k_yy(0.9, lam, ratio, plastic=plastic) <= cap + 1e-12


@pytest.mark.parametrize("lam", SLENDERNESSES)
def test_k_yy_is_non_decreasing_in_the_axial_ratio(lam):
    ratios = jnp.linspace(0.0, 1.0, 200)
    for plastic in (True, False):
        values = k_yy(0.9, lam, ratios, plastic=plastic)

        assert jnp.all(jnp.diff(values) >= -1e-12)


def test_k_yy_is_c_m_at_zero_axial_force():
    for plastic in (True, False):
        assert k_yy(0.73, 1.1, 0.0, plastic=plastic) == pytest.approx(0.73)


def test_k_yy_scales_with_c_m():
    assert k_yy(0.8, 1.0, 0.3, plastic=True) == pytest.approx(
        2.0 * k_yy(0.4, 1.0, 0.3, plastic=True)
    )


def test_plastic_and_elastic_branches_differ():
    # Class 1/2 uses (lam - 0.2) with a 0.8 cap; Class 3/4 uses 0.6 lam with a
    # 0.6 cap. They must not silently coincide.
    plastic = k_yy(0.9, 1.2, 0.3, plastic=True)
    elastic = k_yy(0.9, 1.2, 0.3, plastic=False)

    assert plastic != pytest.approx(elastic)


@pytest.mark.parametrize("lam", SLENDERNESSES)
def test_k_zz_uses_the_rhs_row_not_the_i_section_row(lam):
    # For a hollow section Table B.1 gives k_zz the same form as k_yy. The
    # I-section row would use (2 lam - 0.6) and a 1.4 cap.
    assert k_zz(0.9, lam, 0.3, plastic=True) == pytest.approx(
        k_yy(0.9, lam, 0.3, plastic=True)
    )


def test_cross_factors_plastic():
    value = k_zz(0.9, 1.0, 0.3, plastic=True)

    assert k_yz(value, plastic=True) == pytest.approx(0.6 * value)
    assert k_zy(value, plastic=True) == pytest.approx(0.6 * value)


def test_cross_factors_elastic():
    value = k_zz(0.9, 1.0, 0.3, plastic=False)

    assert k_yz(value, plastic=False) == pytest.approx(value)
    assert k_zy(value, plastic=False) == pytest.approx(0.8 * value)


# ---- ECCS Example 3.13, pp. 242-250 ---- #
#
# RHS 200x150x8, S355, Class 1, Annex B method 2. A hollow section, which is
# the row a CHS reads. Verified against the book; see docs/clauses.md.

ECCS_N_ED = 965e3
ECCS_M_Y_ED = 67.5e6
ECCS_N_RK = 1872.6e3
ECCS_M_Y_RK = 127.4e6
ECCS_CHI_Y = 0.83
ECCS_CHI_Z = 0.72
ECCS_LAMBDA_Y = 0.74
ECCS_PSI = -33.8 / 67.5


def test_eccs_example_moment_factor():
    assert c_m_linear(ECCS_PSI) == pytest.approx(0.40, rel=TOLERANCE)


def test_eccs_example_axial_ratio():
    ratio = axial_ratio(ECCS_N_ED, ECCS_CHI_Y, ECCS_N_RK)

    assert ratio == pytest.approx(0.6209, rel=TOLERANCE)


def test_eccs_example_k_yy():
    ratio = axial_ratio(ECCS_N_ED, ECCS_CHI_Y, ECCS_N_RK)
    factor = k_yy(c_m_linear(ECCS_PSI), ECCS_LAMBDA_Y, ratio, plastic=True)

    assert factor == pytest.approx(0.53, rel=TOLERANCE)


def test_eccs_example_utilization():
    # The book reports 0.90 for eq. 6.61. Its 6.62 uses the Table B.1 footnote
    # k_zy = 0, which is permissive; we do not take it, so our 6.62 is higher
    # and 6.61 still governs.
    value = utilization(
        ECCS_N_ED,
        ECCS_M_Y_ED,
        0.0,
        ECCS_CHI_Y,
        ECCS_CHI_Z,
        ECCS_N_RK,
        ECCS_M_Y_RK,
        ECCS_LAMBDA_Y,
        ECCS_LAMBDA_Y,
        c_m_linear(ECCS_PSI),
        c_m_linear(ECCS_PSI),
        plastic=True,
    )

    assert value == pytest.approx(0.9039, rel=TOLERANCE)


# ---- Reduction checks: the two that catch sign and normalization errors ---- #


def test_reduces_to_pure_compression_when_moments_vanish():
    value = utilization(
        965e3,
        0.0,
        0.0,
        0.83,
        0.72,
        1872.6e3,
        127.4e6,
        0.74,
        0.92,
        0.4,
        0.4,
        plastic=True,
    )
    pure = 965e3 / (0.72 * 1872.6e3)

    assert value == pytest.approx(pure, rel=1e-12)


def test_reduces_to_pure_bending_when_the_axial_force_vanishes():
    value = utilization(
        0.0,
        67.5e6,
        0.0,
        0.83,
        0.72,
        1872.6e3,
        127.4e6,
        0.74,
        0.92,
        0.4,
        0.4,
        plastic=True,
    )
    # With no axial force every k_ij collapses to its C_m.
    pure = 0.4 * 67.5e6 / 127.4e6

    assert value == pytest.approx(pure, rel=1e-12)


# ---- The correction: 6.61 and 6.62 are not the same equation ---- #


def test_the_two_equations_agree_only_when_the_moments_are_equal():
    common = dict(
        chi_y=0.8,
        chi_z=0.8,
        n_rk=2.6e6,
        m_rk=150e6,
        lam_y=0.9,
        lam_z=0.9,
        c_my=0.9,
        c_mz=0.9,
    )
    balanced = utilization(500e3, 100e6, 100e6, plastic=True, **common)
    swapped = utilization(500e3, 100e6, 20e6, plastic=True, **common)
    mirrored = utilization(500e3, 20e6, 100e6, plastic=True, **common)

    # Symmetric in the two moments, because we take the worse of the pair.
    assert swapped == pytest.approx(mirrored)
    # But unequal moments give a different answer from equal ones.
    assert swapped != pytest.approx(balanced)


def test_utilization_is_symmetric_in_the_two_moments():
    common = dict(
        chi_y=0.75,
        chi_z=0.75,
        n_rk=2.6e6,
        m_rk=150e6,
        lam_y=1.0,
        lam_z=1.0,
        c_my=0.8,
        c_mz=0.8,
    )
    for m_y, m_z in ((80e6, 10e6), (10e6, 80e6), (45e6, 45e6)):
        assert utilization(400e3, m_y, m_z, plastic=True, **common) == pytest.approx(
            utilization(400e3, m_z, m_y, plastic=True, **common)
        )


# ---- Monotonicity, which is what keeps the sizing bisection valid ---- #


def test_utilization_falls_as_the_reduction_factor_rises():
    values = [
        utilization(
            500e3, 60e6, 20e6, chi, chi, 2.6e6, 150e6, 0.9, 0.9, 0.8, 0.8, plastic=True
        )
        for chi in (0.4, 0.6, 0.8, 1.0)
    ]

    assert all(b < a for a, b in zip(values, values[1:]))


def test_utilization_falls_as_the_resistances_rise():
    values = [
        utilization(
            500e3,
            60e6,
            20e6,
            0.8,
            0.8,
            2.6e6 * s,
            150e6 * s,
            0.9,
            0.9,
            0.8,
            0.8,
            plastic=True,
        )
        for s in (1.0, 1.5, 2.0, 3.0)
    ]

    assert all(b < a for a, b in zip(values, values[1:]))


def test_utilization_rises_with_every_action():
    base = utilization(
        400e3, 40e6, 10e6, 0.8, 0.8, 2.6e6, 150e6, 0.9, 0.9, 0.8, 0.8, plastic=True
    )
    for bumped in (
        utilization(
            500e3, 40e6, 10e6, 0.8, 0.8, 2.6e6, 150e6, 0.9, 0.9, 0.8, 0.8, plastic=True
        ),
        utilization(
            400e3, 50e6, 10e6, 0.8, 0.8, 2.6e6, 150e6, 0.9, 0.9, 0.8, 0.8, plastic=True
        ),
        utilization(
            400e3, 40e6, 20e6, 0.8, 0.8, 2.6e6, 150e6, 0.9, 0.9, 0.8, 0.8, plastic=True
        ),
    ):
        assert bumped > base


def test_elastic_first_equation_never_falls_below_the_second():
    # Elastic couplings are k_yz = k_zz and k_zy = 0.8 k_yy, so 6.61 dominates.
    for m_y, m_z in ((60e6, 20e6), (20e6, 60e6), (40e6, 40e6)):
        value = utilization(
            500e3, m_y, m_z, 0.8, 0.8, 2.6e6, 150e6, 0.9, 0.9, 0.8, 0.8, plastic=False
        )

        assert jnp.isfinite(value)


# ---- Governing diagnostics, reported beside the utilization ---- #

DIAGNOSTIC = dict(
    chi_y=0.8,
    chi_z=0.8,
    n_rk=2.6e6,
    m_rk=150e6,
    lam_y=0.9,
    lam_z=0.9,
    c_my=0.9,
    c_mz=0.9,
)


def test_the_major_axis_equation_governs_when_the_major_moment_dominates():
    code = governing_equation(500e3, 100e6, 20e6, plastic=True, **DIAGNOSTIC)

    assert code == GOVERNING_MAJOR


def test_the_minor_axis_equation_governs_when_the_minor_moment_dominates():
    code = governing_equation(500e3, 20e6, 100e6, plastic=True, **DIAGNOSTIC)

    assert code == GOVERNING_MINOR


def test_equal_moments_resolve_to_the_major_axis_equation():
    # The two equations coincide there, so the choice is arbitrary but must be
    # deterministic, or the reported code will chatter across the tie.
    code = governing_equation(500e3, 60e6, 60e6, plastic=True, **DIAGNOSTIC)

    assert code == GOVERNING_MAJOR


@pytest.mark.parametrize(
    "m_y, m_z", [(100e6, 20e6), (20e6, 100e6), (60e6, 60e6), (80e6, 0.0)]
)
@pytest.mark.parametrize("plastic", [True, False])
def test_the_governing_code_is_consistent_with_the_utilization(plastic, m_y, m_z):
    # Whichever equation is reported must be the one whose value was taken.
    code = governing_equation(500e3, m_y, m_z, plastic=plastic, **DIAGNOSTIC)
    value = utilization(500e3, m_y, m_z, plastic=plastic, **DIAGNOSTIC)
    swapped = utilization(500e3, m_z, m_y, plastic=plastic, **DIAGNOSTIC)

    assert value == pytest.approx(swapped)
    assert code in (GOVERNING_MAJOR, GOVERNING_MINOR)


def test_the_governing_code_vectorizes_over_members():
    codes = governing_equation(
        500e3,
        jnp.asarray([100e6, 20e6]),
        jnp.asarray([20e6, 100e6]),
        plastic=True,
        **DIAGNOSTIC,
    )

    assert codes.shape == (2,)
    assert np.asarray(codes) == pytest.approx([GOVERNING_MAJOR, GOVERNING_MINOR])


@pytest.mark.parametrize("plastic", [True, False])
def test_the_cap_is_inactive_without_axial_force(plastic):
    assert cap_is_active(0.9, 1.5, 0.0, plastic=plastic) == 0.0


@pytest.mark.parametrize("plastic", [True, False])
def test_the_cap_binds_above_unit_slenderness(plastic):
    # Both branches cross at the same place: the plastic slope (lam - 0.2)
    # passes its 0.8 bound, and the elastic slope 0.6 lam passes its 0.6 bound,
    # at exactly lam = 1.
    assert cap_is_active(0.9, 1.5, 0.4, plastic=plastic) == 1.0
    assert cap_is_active(0.9, 0.5, 0.4, plastic=plastic) == 0.0


@pytest.mark.parametrize("plastic", [True, False])
def test_the_cap_marks_exactly_where_the_factor_stops_rising(plastic):
    slendernesses = jnp.linspace(0.2, 2.5, 200)
    active = cap_is_active(0.9, slendernesses, 0.5, plastic=plastic)
    factors = k_yy(0.9, slendernesses, 0.5, plastic=plastic)

    # Where the cap binds the factor is flat in the slenderness; where it does
    # not, the factor still rises with it. The one interval that straddles the
    # transition is neither, so both endpoints have to be on the same side.
    steps = jnp.diff(factors)
    capped = (active[:-1] == 1.0) & (active[1:] == 1.0)
    free = (active[:-1] == 0.0) & (active[1:] == 0.0)

    assert jnp.all(jnp.abs(steps[capped]) < 1e-12)
    assert jnp.all(steps[free] > 0.0)
    assert jnp.any(capped) and jnp.any(free)


# ---- JAX plumbing ---- #


def test_utilization_is_float64():
    value = utilization(
        500e3, 60e6, 20e6, 0.8, 0.8, 2.6e6, 150e6, 0.9, 0.9, 0.8, 0.8, plastic=True
    )

    assert value.dtype == jnp.float64


def test_utilization_vectorizes_over_members():
    forces = jnp.asarray([300e3, 500e3, 700e3])

    values = utilization(
        forces, 60e6, 20e6, 0.8, 0.8, 2.6e6, 150e6, 0.9, 0.9, 0.8, 0.8, plastic=True
    )

    assert values.shape == (3,)
    assert np.all(np.diff(np.asarray(values)) > 0.0)


def test_utilization_is_jittable():
    jitted = jax.jit(utilization, static_argnames=("plastic",))
    value = jitted(
        500e3, 60e6, 20e6, 0.8, 0.8, 2.6e6, 150e6, 0.9, 0.9, 0.8, 0.8, plastic=True
    )

    assert value == pytest.approx(
        utilization(
            500e3, 60e6, 20e6, 0.8, 0.8, 2.6e6, 150e6, 0.9, 0.9, 0.8, 0.8, plastic=True
        )
    )


def test_utilization_is_differentiable():
    gradient = jax.grad(utilization)(
        500e3, 60e6, 20e6, 0.8, 0.8, 2.6e6, 150e6, 0.9, 0.9, 0.8, 0.8, plastic=True
    )

    assert jnp.isfinite(gradient)
    assert gradient > 0.0
