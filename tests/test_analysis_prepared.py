import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from normax.analysis.smax import forces
from normax.analysis.smax import prepare
from normax.ec3.material import SteelGrade
from normax.ec3.sizing import Tube
from normax.formfinding import equilibrium
from normax.formfinding import graph
from normax.structures import arch

# A 10 m arch of ten members under a 20 kN load at every free node, in the XZ
# plane, at about the size the code check asks for. Units are mm and N.
SPAN = 10_000.0
LOAD = 20_000.0
NUM_EDGES = 10
FORCE_DENSITY = -75.0
NORMAL = 1
DIAMETER = 100.0

# Compiling reassociates the arithmetic, so a jitted result differs from an eager
# one in its last bits rather than matching exactly. Measured, scaled by the
# largest entry of the field: 1.4e-15 on the axial force, and 1.2e-12 on the end
# moments and on the gradient, both of which are near-cancellations on a
# funicular arch and so carry the axial scale's precision rather than their own.
TOLERANCE_JIT = 1e-14
TOLERANCE_JIT_MOMENT = 1e-10

# The one material derivative a linear elastic frame has, since member forces of
# a uniform-E frame are E-independent and so cannot tell an injected leaf from a
# baked one. Loose because the moment it reads is a near-cancellation on a
# funicular arch, which is what limits the difference quotient it is checked
# against rather than the derivative itself.
TOLERANCE_GRADIENT = 1e-6


def relative(actual, expected):
    """
    Worst absolute gap over the largest entry of the reference.

    Scaled by the field rather than entrywise, because a pinned base carries an
    end moment of numerically zero and a relative comparison against it would
    report the width of a float as a disagreement.
    """
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    scale = float(np.max(np.abs(expected)))

    return float(np.max(np.abs(actual - expected))) / (scale if scale > 0.0 else 1.0)


@pytest.fixture(scope="module")
def steel():
    return SteelGrade()


@pytest.fixture(scope="module")
def tube(steel):
    return Tube.at_class_limit(steel.f_y, 3)


@pytest.fixture(scope="module")
def structure():
    return arch(num_edges=NUM_EDGES, span=SPAN, rise=SPAN / 3.0, load=LOAD)


@pytest.fixture(scope="module")
def state(structure):
    q = jnp.full(NUM_EDGES, FORCE_DENSITY)

    return equilibrium(q, structure, graph(structure))


@pytest.fixture(scope="module")
def diameters():
    return jnp.full(NUM_EDGES, DIAMETER)


# --------------------------------------------------------------------------- #
# Nothing about the placeholder survives into a result
# --------------------------------------------------------------------------- #
def test_a_model_prepared_from_any_geometry_gives_the_same_forces(
    structure, state, steel, tube, diameters
):
    # The placeholder geometry differs from the analysed one by the whole of form
    # finding, so if it reached the assembly the two would not agree at all.
    from_start = prepare(structure, steel, tube, normal=NORMAL)
    from_found = prepare(
        structure._replace(nodes=state.xyz), steel, tube, normal=NORMAL
    )

    a = forces(from_start, state.xyz, diameters, steel, tube)
    b = forces(from_found, state.xyz, diameters, steel, tube)

    assert np.all(np.asarray(a.n_ed) == np.asarray(b.n_ed))
    assert np.all(np.asarray(a.m_y_ed) == np.asarray(b.m_y_ed))


def test_a_model_prepared_from_any_material_and_section_gives_the_same_forces(
    structure, state, steel, tube, diameters
):
    # An absurd placeholder: a unit modulus, a unit density and a tube whose
    # smallest size is larger than anything the arch uses.
    absurd = prepare(
        structure._replace(nodes=state.xyz * 3.0),
        SteelGrade(f_y=1.0, e_mod=1.0, density=1.0),
        Tube(ratio=tube.ratio, diameter_min=999.0),
        normal=NORMAL,
    )
    honest = prepare(structure, steel, tube, normal=NORMAL)

    a = forces(absurd, state.xyz, diameters, steel, tube)
    b = forces(honest, state.xyz, diameters, steel, tube)

    assert np.all(np.asarray(a.n_ed) == np.asarray(b.n_ed))
    assert np.all(np.asarray(a.m_y_ed) == np.asarray(b.m_y_ed))


def test_the_structures_own_loads_are_the_default_case(
    structure, state, steel, tube, diameters
):
    model = prepare(structure, steel, tube, normal=NORMAL)

    implied = forces(model, state.xyz, diameters, steel, tube)
    named = forces(model, state.xyz, diameters, steel, tube, loads=structure.loads)

    assert np.all(np.asarray(implied.n_ed) == np.asarray(named.n_ed))


# --------------------------------------------------------------------------- #
# Every leaf a derivative might be taken through stays live
# --------------------------------------------------------------------------- #
def test_the_geometry_is_a_live_leaf(structure, state, steel, tube, diameters):
    model = prepare(structure, steel, tube, normal=NORMAL)

    def total(xyz):
        return jnp.sum(forces(model, xyz, diameters, steel, tube).n_ed ** 2)

    gradient = jax.grad(total)(state.xyz)

    assert np.all(np.isfinite(np.asarray(gradient)))
    assert float(jnp.max(jnp.abs(gradient))) > 0.0


def test_the_diameters_are_a_live_leaf(structure, state, steel, tube, diameters):
    model = prepare(structure, steel, tube, normal=NORMAL)

    def total(sizes):
        return jnp.sum(forces(model, state.xyz, sizes, steel, tube).m_y_ed ** 2)

    gradient = jax.grad(total)(diameters)

    assert np.all(np.isfinite(np.asarray(gradient)))
    assert float(jnp.max(jnp.abs(gradient))) > 0.0


def test_the_modulus_is_a_live_leaf(structure, state, steel, tube, diameters):
    # Member forces of a uniform-E linear frame are E-independent, so the axial
    # force cannot distinguish an injected modulus from a baked one. A
    # displacement can, and its derivative is checked against the difference
    # quotient rather than against zero.
    model = prepare(structure, steel, tube, normal=NORMAL)

    def compliance(e_mod):
        member = forces(model, state.xyz, diameters, steel._replace(e_mod=e_mod), tube)

        return jnp.sum(member.m_y_ed**2) / e_mod

    base = jnp.asarray(steel.e_mod)
    step = base * 1e-5
    exact = float(jax.grad(compliance)(base))
    numeric = float((compliance(base + step) - compliance(base - step)) / (2.0 * step))

    assert exact != 0.0
    assert abs(exact - numeric) / abs(numeric) < TOLERANCE_GRADIENT


# --------------------------------------------------------------------------- #
# The stage is jittable, which is what preparing once buys
# --------------------------------------------------------------------------- #
def test_the_analysis_traces_under_jit(structure, state, steel, tube, diameters):
    model = prepare(structure, steel, tube, normal=NORMAL)

    def run(xyz, sizes):
        member = forces(model, xyz, sizes, steel, tube)

        return member.n_ed, member.m_y_ed

    axial, moments = run(state.xyz, diameters)
    axial_jit, moments_jit = eqx.filter_jit(run)(state.xyz, diameters)

    assert relative(axial_jit, axial) < TOLERANCE_JIT
    assert relative(moments_jit, moments) < TOLERANCE_JIT_MOMENT


def test_the_gradient_of_the_analysis_traces_under_jit(
    structure, state, steel, tube, diameters
):
    model = prepare(structure, steel, tube, normal=NORMAL)

    def total(xyz):
        return jnp.sum(forces(model, xyz, diameters, steel, tube).n_ed ** 2)

    eager = jax.grad(total)(state.xyz)
    jitted = eqx.filter_jit(jax.grad(total))(state.xyz)

    assert relative(jitted, eager) < TOLERANCE_JIT_MOMENT
