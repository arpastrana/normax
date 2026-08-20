import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from normax.analysis import member_forces
from normax.analysis import prepare_model
from normax.form_finding import equilibrium_graph
from normax.form_finding import equilibrium_state
from normax.loads import create_loads_uniform
from normax.materials import Steel355
from normax.materials import SteelGrade
from normax.sections import TubeFamily
from normax.sizing import build_section_family
from normax.structures import build_arch_2d

# A 10 m arch of ten members under a 20 kN load at every free node, in the XZ
# plane, at about the size the code check asks for. Units are mm and N.
SPAN = 10_000.0
LOAD = 20_000.0
NUM_EDGES = 10
FORCE_DENSITY = -75.0
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
    return Steel355()


@pytest.fixture(scope="module")
def catalogue(steel):
    # The class-limit wall proportion, as bare geometry: the analysis needs a
    # family and has no use for the class.
    return build_section_family(steel, 3)


@pytest.fixture(scope="module")
def section(catalogue):
    return catalogue(DIAMETER)


@pytest.fixture(scope="module")
def structure():
    return build_arch_2d(num_edges=NUM_EDGES, span=SPAN, rise=SPAN / 3.0)


def funicular(structure):
    """
    The uniform load case the arch is form-found under and analyzed in.
    """
    return create_loads_uniform(structure, LOAD)


@pytest.fixture(scope="module")
def state(structure):
    q = jnp.full(NUM_EDGES, FORCE_DENSITY)

    graph = equilibrium_graph(structure)

    return equilibrium_state(
        q, structure.nodes[graph.indices_fixed], graph, funicular(structure)
    )


@pytest.fixture(scope="module")
def diameters():
    return jnp.full(NUM_EDGES, DIAMETER)


# --------------------------------------------------------------------------- #
# Nothing about the placeholder survives into a result
# --------------------------------------------------------------------------- #
def test_a_model_prepared_from_any_geometry_gives_the_same_forces(
    structure, state, steel, section, diameters
):
    # The placeholder geometry differs from the analyzed one by the whole of form
    # finding, so if it reached the assembly the two would not agree at all.
    from_start = prepare_model(structure, section)
    from_found = prepare_model(structure._replace(nodes=state.xyz), section)

    a = member_forces(from_start, state.xyz, diameters, section, funicular(structure))
    b = member_forces(from_found, state.xyz, diameters, section, funicular(structure))

    assert np.all(np.asarray(a.axial_force) == np.asarray(b.axial_force))
    assert np.all(np.asarray(a.moment_major) == np.asarray(b.moment_major))


def test_a_model_prepared_from_any_material_and_section_gives_the_same_forces(
    structure, state, steel, catalogue, section, diameters
):
    # An absurd placeholder: unit strengths, a unit modulus, a unit density and
    # a tube whose smallest size is larger than anything the arch uses.
    absurd_family = TubeFamily(
        ratio=catalogue.ratio,
        material=SteelGrade(f_y=1.0, f_u=1.0, e_mod=1.0, density=1.0),
    )
    absurd = prepare_model(
        structure._replace(nodes=state.xyz * 3.0), absurd_family(999.0)
    )
    honest = prepare_model(structure, section)

    a = member_forces(absurd, state.xyz, diameters, section, funicular(structure))
    b = member_forces(honest, state.xyz, diameters, section, funicular(structure))

    assert np.all(np.asarray(a.axial_force) == np.asarray(b.axial_force))
    assert np.all(np.asarray(a.moment_major) == np.asarray(b.moment_major))


def test_a_prepared_model_carries_no_load_case_of_its_own(
    structure, state, steel, section, diameters
):
    # Preparing compiles the shape of the nodal channels and nothing else, so a
    # load case is always the caller's and two of them reuse one program.
    model = prepare_model(structure, section)

    applied = funicular(structure)
    once = member_forces(model, state.xyz, diameters, section, applied)
    twice = member_forces(model, state.xyz, diameters, section, applied)
    halved = member_forces(model, state.xyz, diameters, section, 0.5 * applied)

    assert np.all(np.asarray(once.axial_force) == np.asarray(twice.axial_force))
    assert np.allclose(
        np.asarray(halved.axial_force), 0.5 * np.asarray(once.axial_force)
    )


# --------------------------------------------------------------------------- #
# Every leaf a derivative might be taken through stays live
# --------------------------------------------------------------------------- #
def test_the_geometry_is_a_live_leaf(structure, state, steel, section, diameters):
    model = prepare_model(structure, section)

    def total(xyz):
        return jnp.sum(
            member_forces(
                model, xyz, diameters, section, funicular(structure)
            ).axial_force
            ** 2
        )

    gradient = jax.grad(total)(state.xyz)

    assert np.all(np.isfinite(np.asarray(gradient)))
    assert float(jnp.max(jnp.abs(gradient))) > 0.0


def test_the_diameters_are_a_live_leaf(structure, state, steel, section, diameters):
    model = prepare_model(structure, section)

    def total(sizes):
        return jnp.sum(
            member_forces(
                model, state.xyz, sizes, section, funicular(structure)
            ).moment_major
            ** 2
        )

    gradient = jax.grad(total)(diameters)

    assert np.all(np.isfinite(np.asarray(gradient)))
    assert float(jnp.max(jnp.abs(gradient))) > 0.0


def test_the_modulus_is_a_live_leaf(
    structure, state, steel, catalogue, section, diameters
):
    # Member forces of a uniform-E linear frame are E-independent, so the axial
    # force cannot distinguish an injected modulus from a baked one. A
    # displacement can, and its derivative is checked against the difference
    # quotient rather than against zero.
    model = prepare_model(structure, section)
    applied = funicular(structure)

    def compliance(e_mod):
        graded = TubeFamily(catalogue.ratio, steel._replace(e_mod=e_mod))
        member = member_forces(model, state.xyz, diameters, graded(DIAMETER), applied)

        return jnp.sum(member.moment_major**2) / e_mod

    base = jnp.asarray(steel.e_mod)
    step = base * 1e-5
    exact = float(jax.grad(compliance)(base))
    numeric = float((compliance(base + step) - compliance(base - step)) / (2.0 * step))

    assert exact != 0.0
    assert abs(exact - numeric) / abs(numeric) < TOLERANCE_GRADIENT


# --------------------------------------------------------------------------- #
# The stage is jittable, which is what preparing once buys
# --------------------------------------------------------------------------- #
def test_the_analysis_traces_under_jit(structure, state, steel, section, diameters):
    model = prepare_model(structure, section)

    def run(xyz, sizes):
        member = member_forces(model, xyz, sizes, section, funicular(structure))

        return member.axial_force, member.moment_major

    axial, moments = run(state.xyz, diameters)
    axial_jit, moments_jit = eqx.filter_jit(run)(state.xyz, diameters)

    assert relative(axial_jit, axial) < TOLERANCE_JIT
    assert relative(moments_jit, moments) < TOLERANCE_JIT_MOMENT


def test_the_gradient_of_the_analysis_traces_under_jit(
    structure, state, steel, section, diameters
):
    model = prepare_model(structure, section)

    def total(xyz):
        return jnp.sum(
            member_forces(
                model, xyz, diameters, section, funicular(structure)
            ).axial_force
            ** 2
        )

    eager = jax.grad(total)(state.xyz)
    jitted = eqx.filter_jit(jax.grad(total))(state.xyz)

    assert relative(jitted, eager) < TOLERANCE_JIT_MOMENT
