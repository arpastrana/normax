import jax
import jax.numpy as jnp
import numpy as np
import pytest
from smax import PinnedSupport
from smax import Structure as Frame
from smax import compile_structure
from smax import diagnose_mechanisms
from smax import element_forces
from smax import solve

from normax.analysis import normal_axis
from normax.analysis import support_fixities
from normax.analysis.smax import frame_model
from normax.analysis.smax import member_forces
from normax.analysis.smax import prepare_model
from normax.form_finding.fdm import equilibrium_graph
from normax.form_finding.fdm import equilibrium_state
from normax.loads import loads_uniform
from normax.materials import SteelGrade
from normax.sizing.ec3 import Ec3Sizer
from normax.structures import build_arch_2d
from normax.structures import build_gridshell_3d

# A 10 m arch of ten members, rising about a third of its span under a force
# density of 75 N/mm and a 20 kN load at every free node. Units are mm and N.
SPAN = 10_000.0
LOAD = 20_000.0
NUM_EDGES = 10
FORCE_DENSITY = -75.0

# The arch lies in the XZ plane, so it has no thickness along Y. Measured rather
# than declared, and asserted below.
NORMAL = 1

# Close to the 73 to 87 mm the code check asks for on this arch, and a round
# number, so the tolerances below are recorded at a size the design would use.
DIAMETER = 100.0

# Measured, not chosen. The axial forces of the unstressed frame reproduce the
# funicular ones to 2.27e-4, and the largest end moment is 7.58e-4 of the axial
# force times the length.
TOLERANCE_AXIAL = 2.5e-4
TOLERANCE_BENDING = 1.0e-3


@pytest.fixture(scope="module")
def structure():
    return build_arch_2d(num_edges=NUM_EDGES, span=SPAN, rise=SPAN / 3.0)


def funicular(structure, load=LOAD):
    """
    The uniform load case the arch is form-found under.
    """
    return loads_uniform(structure, load)


@pytest.fixture(scope="module")
def steel():
    return SteelGrade()


@pytest.fixture(scope="module")
def catalogue(steel, structure):
    # The class-limit wall proportion, read off a configured sizer as bare
    # geometry: the analysis needs a family and has no use for the class.
    return Ec3Sizer.at_class_limit(structure, steel, 3).family


@pytest.fixture(scope="module")
def section(catalogue):
    return catalogue(DIAMETER)


@pytest.fixture(scope="module")
def model(structure, section):
    return prepare_model(structure, section)


@pytest.fixture(scope="module")
def q():
    return jnp.full(NUM_EDGES, FORCE_DENSITY)


@pytest.fixture(scope="module")
def state(q, structure):
    graph = equilibrium_graph(structure)

    return equilibrium_state(
        q, structure.nodes[graph.indices_fixed], graph, funicular(structure)
    )


@pytest.fixture(scope="module")
def member(model, state, section, structure):
    return member_forces(
        model, state.xyz, jnp.full(NUM_EDGES, DIAMETER), section, funicular(structure)
    )


def deviation(diameter, steel, catalogue, load=LOAD, force_density=FORCE_DENSITY):
    """
    Largest relative gap between the analyzed and the funicular axial force.
    """
    structure = build_arch_2d(num_edges=NUM_EDGES, span=SPAN, rise=SPAN / 3.0)
    applied = funicular(structure, load)
    q = jnp.full(NUM_EDGES, force_density)
    graph = equilibrium_graph(structure)
    state = equilibrium_state(q, structure.nodes[graph.indices_fixed], graph, applied)

    # The grade the caller names, at the wall the family holds.
    graded = catalogue._replace(material=steel)
    section = graded(diameter)

    expected = q * state.lengths[:, 0]
    member = member_forces(
        prepare_model(structure, section),
        state.xyz,
        jnp.full(NUM_EDGES, diameter),
        section,
        applied,
    )

    return float(jnp.max(jnp.abs(member.axial_force - expected) / jnp.abs(expected)))


def span_field(structure, xyz, section):
    """
    The whole span field, to check the assumptions `forces` collapses it under.
    """
    compiled = compile_structure(frame_model(structure, xyz, section))
    loads = funicular(structure)
    response = solve(compiled, loads)

    return element_forces(compiled, response, num_samples=2)


# --------------------------------------------------------------------------- #
# The model the analysis runs on
# --------------------------------------------------------------------------- #
def test_the_arch_is_not_a_mechanism_once_the_plane_is_restrained(
    structure, state, section
):
    model = frame_model(structure, state.xyz, section)

    assert diagnose_mechanisms(model).num_mechanisms == 0


def test_a_planar_arch_on_pinned_supports_alone_is_a_mechanism(
    structure, state, section
):
    model = frame_model(structure, state.xyz, section)
    unrestrained = Frame(
        model.nodes,
        model.elements,
        [PinnedSupport(0), PinnedSupport(NUM_EDGES)],
    )

    # Rotating the whole arch about the line joining its supports strains no
    # member and moves no support. Restraining the out-of-plane translation is
    # what removes that mode, and without it the solve returns nan.
    assert diagnose_mechanisms(unrestrained).num_mechanisms == 1


def test_the_plane_of_the_arch_is_measured_rather_than_declared(structure):
    assert normal_axis(structure) == NORMAL


def test_a_structure_that_fills_space_has_no_normal_axis():
    assert normal_axis(build_gridshell_3d()) is None


def test_a_support_is_pinned_and_never_fixed(structure):
    # Pinned is about the moment a base carries, which is the rotation the
    # in-plane bending happens about. That one stays free. The two rotations out
    # of the plane are held so that a straight structure is not a mechanism, and
    # no in-plane load can excite them.
    flags = support_fixities(structure)
    supports = np.asarray(structure.supports)

    assert np.all(flags[supports, :3] == True)
    assert np.all(flags[supports, 3 + NORMAL] == False)


def test_a_free_node_is_restrained_only_out_of_the_plane(structure):
    flags = support_fixities(structure)
    free = [n for n in range(structure.nodes.shape[0]) if n not in (0, NUM_EDGES)]

    for node in free:
        assert flags[node, NORMAL] == True
        assert np.count_nonzero(flags[node]) == 1


def test_a_planar_support_holds_the_rotations_no_in_plane_load_excites(structure):
    # Pinned and never fixed is a rule about structures that occupy all three
    # dimensions. A planar one is a mechanism without a deviation from it, and a
    # straight planar one is a mechanism even with the normal translation held,
    # so its supports hold the two rotations out of the plane as well.
    flags = support_fixities(structure)
    supports = np.asarray(structure.supports)
    out_of_plane = [3 + axis for axis in (0, 1, 2) if axis != NORMAL]

    assert np.all(flags[np.ix_(supports, out_of_plane)] == True)


def test_a_three_dimensional_structure_restrains_nothing_beyond_its_supports():
    gridshell = build_gridshell_3d()
    flags = support_fixities(gridshell)

    assert np.count_nonzero(flags) == 3 * gridshell.supports.shape[0]


def test_a_structure_held_nowhere_is_refused(structure):
    with pytest.raises(ValueError):
        support_fixities(structure._replace(supports=np.zeros(0, dtype=int)))


# --------------------------------------------------------------------------- #
# What form finding hands over
# --------------------------------------------------------------------------- #
def test_form_finding_balances_the_loads_at_every_free_node(state):
    free = jnp.asarray([n for n in range(NUM_EDGES + 1) if n not in (0, NUM_EDGES)])

    assert float(jnp.max(jnp.abs(state.residuals[free]))) < 1e-8


def test_every_member_carries_its_force_density_times_its_length(q, state):
    assert np.allclose(state.forces[:, 0], q * state.lengths[:, 0], rtol=1e-14)


def test_the_form_found_arch_lies_in_the_plane_it_started_in(state):
    assert float(jnp.max(jnp.abs(state.xyz[:, 1]))) == 0.0


def test_the_arch_rises_rather_than_sags(state):
    assert float(jnp.min(state.xyz[:, 2])) == 0.0
    assert float(jnp.max(state.xyz[:, 2])) > 0.0


# --------------------------------------------------------------------------- #
# The gate: the analysis reproduces the funicular state
# --------------------------------------------------------------------------- #
def test_the_analysis_reproduces_the_funicular_axial_forces(q, state, member):
    funicular = q * state.lengths[:, 0]
    gap = jnp.abs(member.axial_force - funicular) / jnp.abs(funicular)

    assert float(jnp.max(gap)) < TOLERANCE_AXIAL


def test_the_analysis_agrees_with_form_finding_on_the_sign_of_every_member(
    q, state, member
):
    funicular = q * state.lengths[:, 0]

    assert np.all(jnp.sign(member.axial_force) == jnp.sign(funicular))


def test_bending_is_secondary_to_axial_action(q, state, member):
    funicular = q * state.lengths[:, 0]
    peak = jnp.max(jnp.abs(member.moment_major), axis=1)
    ratio = peak / jnp.abs(funicular * state.lengths[:, 0])

    assert float(jnp.max(ratio)) < TOLERANCE_BENDING


def test_the_axial_force_does_not_vary_along_a_member(structure, state, section):
    field = span_field(structure, state.xyz, section)

    assert np.allclose(field.nx[:, 0], field.nx[:, 1], rtol=1e-12)


def test_the_reported_axial_force_is_the_one_the_solver_recovered(
    structure, state, section, member
):
    field = span_field(structure, state.xyz, section)

    assert np.allclose(member.axial_force, field.nx[:, 0], rtol=1e-15)


def test_an_arch_in_a_plane_carries_no_minor_axis_moment(member):
    assert float(jnp.max(jnp.abs(member.moment_minor))) == 0.0


def test_the_end_moments_of_neighbouring_members_agree(member):
    # Continuity at a shared node, and the sign convention that goes with it.
    assert np.allclose(
        member.moment_major[:-1, 1], member.moment_major[1:, 0], rtol=1e-10
    )


def test_the_arch_carries_no_moment_at_a_pinned_base(member):
    assert float(abs(member.moment_major[0, 0])) < 1e-6
    assert float(abs(member.moment_major[-1, 1])) < 1e-6


# --------------------------------------------------------------------------- #
# Why the gap is what it is
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("diameter", [50.0, 100.0, 200.0])
def test_the_gap_is_quadratic_in_the_diameter(diameter, steel, catalogue):
    # A beam chain through a funicular polygon cannot turn a kink on axial force
    # alone, and the bending it needs instead scales as the square of the radius
    # of gyration over the length. So the gap closes as the members thin, and it
    # is a property of their slenderness rather than of the load or the steel.
    reference = deviation(100.0, steel, catalogue)
    scaled = reference * (diameter / 100.0) ** 2

    assert deviation(diameter, steel, catalogue) == pytest.approx(scaled, rel=0.01)


@pytest.mark.parametrize("e_mod", [70_000.0, 400_000.0])
def test_the_gap_does_not_depend_on_the_modulus(e_mod, steel, catalogue):
    softer = deviation(DIAMETER, steel._replace(e_mod=e_mod), catalogue)

    assert softer == pytest.approx(deviation(DIAMETER, steel, catalogue), rel=1e-9)


@pytest.mark.parametrize("scale", [0.1, 10.0])
def test_the_gap_does_not_depend_on_the_scale_of_the_loading(scale, steel, catalogue):
    scaled = deviation(
        DIAMETER,
        steel,
        catalogue,
        load=LOAD * scale,
        force_density=FORCE_DENSITY * scale,
    )

    assert scaled == pytest.approx(deviation(DIAMETER, steel, catalogue), rel=1e-9)


# --------------------------------------------------------------------------- #
# The gradient crosses both stages
# --------------------------------------------------------------------------- #
def test_the_gradient_through_both_stages_matches_central_differences(
    q, structure, model, section
):
    fdm = equilibrium_graph(structure)
    diameters = jnp.full(NUM_EDGES, DIAMETER)
    applied = funicular(structure)

    def objective(q):
        state = equilibrium_state(q, structure.nodes[fdm.indices_fixed], fdm, applied)
        member = member_forces(model, state.xyz, diameters, section, applied)
        return jnp.sum(member.axial_force**2)

    gradient = jax.grad(objective)(q)

    step = 1e-3
    for edge in (0, NUM_EDGES // 2):
        plus = objective(q.at[edge].add(step))
        minus = objective(q.at[edge].add(-step))
        central = (plus - minus) / (2.0 * step)

        assert float(gradient[edge]) == pytest.approx(float(central), rel=1e-7)


def test_the_gradient_through_both_stages_is_finite(q, structure, model, section):
    fdm = equilibrium_graph(structure)
    diameters = jnp.full(NUM_EDGES, DIAMETER)
    applied = funicular(structure)

    def objective(q):
        state = equilibrium_state(q, structure.nodes[fdm.indices_fixed], fdm, applied)
        member = member_forces(model, state.xyz, diameters, section, applied)
        return jnp.sum(member.axial_force**2)

    gradient = jax.grad(objective)(q)

    assert jnp.all(jnp.isfinite(gradient))
    assert float(jnp.min(jnp.abs(gradient))) > 0.0
