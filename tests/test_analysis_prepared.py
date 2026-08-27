# SPDX-License-Identifier: Apache-2.0
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from normax.analysis.smax import compute_member_forces
from normax.analysis.smax import prepare_model
from normax.form_finding import FdmFormFinder
from normax.loads import load_uniform
from normax.materials import Steel355
from normax.materials import SteelGrade
from normax.sections import TubeFamily
from normax.sections import build_section_family
from normax.structures import build_arch_2d

# A 10 m arch of ten members under a 20 kN load at every free node, in the XZ
# plane, at about the size the code check asks for. Units are mm and N.
SPAN = 10_000.0
LOAD = 20_000.0
NUM_EDGES = 10
FORCE_DENSITY = -75.0
DIAMETER = 100.0

# Compiling reassociates the arithmetic. Measured at 1.4e-15 on the axial force
# and 1.2e-12 on the end moments and the gradient, both near-cancellations.
TOLERANCE_JIT = 1e-14
TOLERANCE_JIT_MOMENT = 1e-10

# The one material derivative a linear elastic frame has, against a difference
# quotient of a near-cancellation.
TOLERANCE_GRADIENT = 1e-6


def relative(actual, expected):
    """
    Worst absolute gap over the largest entry of the reference.
    """
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    scale = float(np.max(np.abs(expected)))

    return float(np.max(np.abs(actual - expected))) / (scale if scale > 0.0 else 1.0)


@pytest.fixture(scope="module")
def steel():
    return Steel355()


@pytest.fixture(scope="module")
def catalogue(steel):
    return build_section_family(steel, 3)


@pytest.fixture(scope="module")
def section(catalogue):
    return catalogue(DIAMETER)


@pytest.fixture(scope="module")
def structure():
    return build_arch_2d(num_edges=NUM_EDGES, span=SPAN, rise=SPAN / 3.0)


@pytest.fixture(scope="module")
def applied(structure):
    return load_uniform(structure, LOAD * (NUM_EDGES - 1))


@pytest.fixture(scope="module")
def xyz(structure, applied):
    return FdmFormFinder(structure)(jnp.full(NUM_EDGES, FORCE_DENSITY), applied).xyz


@pytest.fixture(scope="module")
def diameters():
    return jnp.full(NUM_EDGES, DIAMETER)


@pytest.fixture(scope="module")
def model(structure, section):
    return prepare_model(structure, section)


def test_a_model_prepared_from_any_geometry_gives_the_same_forces(
    structure, xyz, section, diameters, applied
):
    # The placeholder geometry differs from the analyzed one by the whole of form
    # finding, so if it reached the assembly the two would not agree at all.
    from_start = prepare_model(structure, section)
    from_found = prepare_model(structure._replace(nodes=xyz), section)

    a = compute_member_forces(from_start, xyz, diameters, section, applied)
    b = compute_member_forces(from_found, xyz, diameters, section, applied)

    assert np.all(np.asarray(a.axial_force) == np.asarray(b.axial_force))
    assert np.all(np.asarray(a.moment_major) == np.asarray(b.moment_major))


def test_a_model_prepared_from_any_material_and_section_gives_the_same_forces(
    structure, xyz, catalogue, section, diameters, applied
):
    # An absurd placeholder: unit strengths, a unit modulus, a unit density and
    # a tube larger than anything the arch uses.
    absurd_family = TubeFamily(
        catalogue.ratio, SteelGrade(f_y=1.0, f_u=1.0, e_mod=1.0, density=1.0)
    )
    absurd = prepare_model(structure._replace(nodes=xyz * 3.0), absurd_family(999.0))
    honest = prepare_model(structure, section)

    a = compute_member_forces(absurd, xyz, diameters, section, applied)
    b = compute_member_forces(honest, xyz, diameters, section, applied)

    assert np.all(np.asarray(a.axial_force) == np.asarray(b.axial_force))
    assert np.all(np.asarray(a.moment_major) == np.asarray(b.moment_major))


def test_a_prepared_model_carries_no_load_case_of_its_own(
    model, xyz, section, diameters, applied
):
    once = compute_member_forces(model, xyz, diameters, section, applied)
    twice = compute_member_forces(model, xyz, diameters, section, applied)
    halved = compute_member_forces(model, xyz, diameters, section, 0.5 * applied)

    assert np.all(np.asarray(once.axial_force) == np.asarray(twice.axial_force))
    assert np.allclose(
        np.asarray(halved.axial_force), 0.5 * np.asarray(once.axial_force)
    )


def test_the_geometry_and_the_diameters_are_live_leaves(
    model, xyz, section, diameters, applied
):
    def by_geometry(coords):
        return jnp.sum(
            compute_member_forces(
                model, coords, diameters, section, applied
            ).axial_force
            ** 2
        )

    def by_size(sizes):
        return jnp.sum(
            compute_member_forces(model, xyz, sizes, section, applied).moment_major ** 2
        )

    for gradient in (jax.grad(by_geometry)(xyz), jax.grad(by_size)(diameters)):
        assert np.all(np.isfinite(np.asarray(gradient)))
        assert float(jnp.max(jnp.abs(gradient))) > 0.0


def test_the_modulus_is_a_live_leaf(model, xyz, steel, catalogue, diameters, applied):
    # Member forces of a uniform-E linear frame are E-independent, so a
    # compliance is what distinguishes an injected modulus from a baked one.
    def compliance(e_mod):
        graded = TubeFamily(catalogue.ratio, steel._replace(e_mod=e_mod))
        member = compute_member_forces(model, xyz, diameters, graded(DIAMETER), applied)

        return jnp.sum(member.moment_major**2) / e_mod

    base = jnp.asarray(steel.e_mod)
    step = base * 1e-5
    exact = float(jax.grad(compliance)(base))
    numeric = float((compliance(base + step) - compliance(base - step)) / (2.0 * step))

    assert exact != 0.0
    assert abs(exact - numeric) / abs(numeric) < TOLERANCE_GRADIENT


def test_the_analysis_and_its_gradient_trace_under_jit(
    model, xyz, section, diameters, applied
):
    def run(coords, sizes):
        member = compute_member_forces(model, coords, sizes, section, applied)

        return member.axial_force, member.moment_major

    def total(coords):
        return jnp.sum(run(coords, diameters)[0] ** 2)

    axial, moments = run(xyz, diameters)
    axial_jit, moments_jit = eqx.filter_jit(run)(xyz, diameters)
    eager = jax.grad(total)(xyz)
    jitted = eqx.filter_jit(jax.grad(total))(xyz)

    assert relative(axial_jit, axial) < TOLERANCE_JIT
    assert relative(moments_jit, moments) < TOLERANCE_JIT_MOMENT
    assert relative(jitted, eager) < TOLERANCE_JIT_MOMENT
