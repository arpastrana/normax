import jax
import jax.numpy as jnp
import numpy as np
import pytest
from smax import LoadCase
from smax import PinnedSupport
from smax import PointLoad
from smax import Structure as Frame
from smax import compile_structure
from smax import diagnose_mechanisms
from smax import element_forces
from smax import solve

from normax.analysis import fixities
from normax.analysis.smax import forces
from normax.analysis.smax import frame
from normax.analysis.smax import prepare
from normax.ec3.sizing import Steel
from normax.ec3.sizing import Tube
from normax.formfinding import equilibrium
from normax.formfinding import graph
from normax.structures import arch

# A 10 m arch of ten members, rising about a third of its span under a force
# density of 75 N/mm and a 20 kN load at every free node. Units are mm and N.
SPAN = 10_000.0
LOAD = 20_000.0
NUM_EDGES = 10
FORCE_DENSITY = -75.0

# The arch lies in the XZ plane, so it has no thickness along Y.
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
    return arch(num_edges=NUM_EDGES, span=SPAN, rise=SPAN / 3.0, load=LOAD)


@pytest.fixture(scope="module")
def steel():
    return Steel()


@pytest.fixture(scope="module")
def tube(steel):
    return Tube.at_class_limit(steel.f_y, 3)


@pytest.fixture(scope="module")
def model(structure, steel, tube):
    return prepare(structure, steel, tube, normal=NORMAL)


@pytest.fixture(scope="module")
def q():
    return jnp.full(NUM_EDGES, FORCE_DENSITY)


@pytest.fixture(scope="module")
def state(q, structure):
    return equilibrium(q, structure, graph(structure))


@pytest.fixture(scope="module")
def member(model, state, steel, tube):
    return forces(
        model,
        state.xyz,
        jnp.full(NUM_EDGES, DIAMETER),
        steel,
        tube,
    )


def deviation(diameter, steel, tube, load=LOAD, force_density=FORCE_DENSITY):
    """
    Largest relative gap between the analysed and the funicular axial force.
    """
    structure = arch(num_edges=NUM_EDGES, span=SPAN, rise=SPAN / 3.0, load=load)
    q = jnp.full(NUM_EDGES, force_density)
    state = equilibrium(q, structure, graph(structure))

    funicular = q * state.lengths[:, 0]
    member = forces(
        prepare(structure, steel, tube, normal=NORMAL),
        state.xyz,
        jnp.full(NUM_EDGES, diameter),
        steel,
        tube,
    )

    return float(jnp.max(jnp.abs(member.n_ed - funicular) / jnp.abs(funicular)))


def span_field(structure, xyz, steel, tube):
    """
    The whole span field, to check the assumptions `forces` collapses it under.
    """
    compiled = compile_structure(
        frame(structure, xyz, jnp.full(NUM_EDGES, DIAMETER), steel, tube, normal=NORMAL)
    )
    applied = [
        PointLoad(node, load=structure.loads[node])
        for node in range(structure.loads.shape[0])
    ]
    response = solve(compiled, LoadCase(applied, compiled))

    return element_forces(compiled, response, num_samples=2)


# --------------------------------------------------------------------------- #
# The model the analysis runs on
# --------------------------------------------------------------------------- #
def test_the_arch_is_not_a_mechanism_once_the_plane_is_restrained(
    structure, state, steel, tube
):
    model = frame(
        structure,
        state.xyz,
        jnp.full(NUM_EDGES, DIAMETER),
        steel,
        tube,
        normal=NORMAL,
    )

    assert diagnose_mechanisms(model).num_mechanisms == 0


def test_a_planar_arch_on_pinned_supports_alone_is_a_mechanism(
    structure, state, steel, tube
):
    model = frame(
        structure,
        state.xyz,
        jnp.full(NUM_EDGES, DIAMETER),
        steel,
        tube,
        normal=None,
    )
    unrestrained = Frame(
        model.nodes,
        model.elements,
        [PinnedSupport(0), PinnedSupport(NUM_EDGES)],
    )

    # Rotating the whole arch about the line joining its supports strains no
    # member and moves no support. Restraining the out-of-plane translation is
    # what removes that mode, and without it the solve returns nan.
    assert diagnose_mechanisms(unrestrained).num_mechanisms == 1


def test_a_support_is_pinned_and_never_fixed(structure):
    flags = fixities(structure, NORMAL)
    supports = np.asarray(structure.supports)

    assert np.all(flags[supports, :3] == True)
    assert np.all(flags[supports, 3:] == False)


def test_a_free_node_is_restrained_only_out_of_the_plane(structure):
    flags = fixities(structure, NORMAL)
    free = [n for n in range(structure.nodes.shape[0]) if n not in (0, NUM_EDGES)]

    for node in free:
        assert flags[node, NORMAL] == True
        assert np.count_nonzero(flags[node]) == 1


def test_a_three_dimensional_structure_restrains_nothing_beyond_its_supports(structure):
    flags = fixities(structure, None)

    assert np.count_nonzero(flags) == 3 * structure.supports.shape[0]


@pytest.mark.parametrize("normal", [-1, 3, 5])
def test_an_axis_that_is_not_a_global_axis_is_refused(structure, normal):
    with pytest.raises(ValueError):
        fixities(structure, normal)


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
    gap = jnp.abs(member.n_ed - funicular) / jnp.abs(funicular)

    assert float(jnp.max(gap)) < TOLERANCE_AXIAL


def test_the_analysis_agrees_with_form_finding_on_the_sign_of_every_member(
    q, state, member
):
    funicular = q * state.lengths[:, 0]

    assert np.all(jnp.sign(member.n_ed) == jnp.sign(funicular))


def test_bending_is_secondary_to_axial_action(q, state, member):
    funicular = q * state.lengths[:, 0]
    peak = jnp.max(jnp.abs(member.m_y_ed), axis=1)
    ratio = peak / jnp.abs(funicular * state.lengths[:, 0])

    assert float(jnp.max(ratio)) < TOLERANCE_BENDING


def test_the_axial_force_does_not_vary_along_a_member(structure, state, steel, tube):
    field = span_field(structure, state.xyz, steel, tube)

    assert np.allclose(field.nx[:, 0], field.nx[:, 1], rtol=1e-12)


def test_the_reported_axial_force_is_the_one_the_solver_recovered(
    structure, state, steel, tube, member
):
    field = span_field(structure, state.xyz, steel, tube)

    assert np.allclose(member.n_ed, field.nx[:, 0], rtol=1e-15)


def test_an_arch_in_a_plane_carries_no_minor_axis_moment(member):
    assert float(jnp.max(jnp.abs(member.m_z_ed))) == 0.0


def test_the_end_moments_of_neighbouring_members_agree(member):
    # Continuity at a shared node, and the sign convention that goes with it.
    assert np.allclose(member.m_y_ed[:-1, 1], member.m_y_ed[1:, 0], rtol=1e-10)


def test_the_arch_carries_no_moment_at_a_pinned_base(member):
    assert float(abs(member.m_y_ed[0, 0])) < 1e-6
    assert float(abs(member.m_y_ed[-1, 1])) < 1e-6


# --------------------------------------------------------------------------- #
# Why the gap is what it is
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("diameter", [50.0, 100.0, 200.0])
def test_the_gap_is_quadratic_in_the_diameter(diameter, steel, tube):
    # A beam chain through a funicular polygon cannot turn a kink on axial force
    # alone, and the bending it needs instead scales as the square of the radius
    # of gyration over the length. So the gap closes as the members thin, and it
    # is a property of their slenderness rather than of the load or the steel.
    reference = deviation(100.0, steel, tube)
    scaled = reference * (diameter / 100.0) ** 2

    assert deviation(diameter, steel, tube) == pytest.approx(scaled, rel=0.01)


@pytest.mark.parametrize("e_mod", [70_000.0, 400_000.0])
def test_the_gap_does_not_depend_on_the_modulus(e_mod, steel, tube):
    softer = deviation(DIAMETER, steel._replace(e_mod=e_mod), tube)

    assert softer == pytest.approx(deviation(DIAMETER, steel, tube), rel=1e-9)


@pytest.mark.parametrize("scale", [0.1, 10.0])
def test_the_gap_does_not_depend_on_the_scale_of_the_loading(scale, steel, tube):
    scaled = deviation(
        DIAMETER,
        steel,
        tube,
        load=LOAD * scale,
        force_density=FORCE_DENSITY * scale,
    )

    assert scaled == pytest.approx(deviation(DIAMETER, steel, tube), rel=1e-9)


# --------------------------------------------------------------------------- #
# The gradient crosses both stages
# --------------------------------------------------------------------------- #
def test_the_gradient_through_both_stages_matches_central_differences(
    q, structure, model, steel, tube
):
    fdm = graph(structure)
    diameters = jnp.full(NUM_EDGES, DIAMETER)

    def objective(q):
        state = equilibrium(q, structure, fdm)
        member = forces(model, state.xyz, diameters, steel, tube)
        return jnp.sum(member.n_ed**2)

    gradient = jax.grad(objective)(q)

    step = 1e-3
    for edge in (0, NUM_EDGES // 2):
        plus = objective(q.at[edge].add(step))
        minus = objective(q.at[edge].add(-step))
        central = (plus - minus) / (2.0 * step)

        assert float(gradient[edge]) == pytest.approx(float(central), rel=1e-7)


def test_the_gradient_through_both_stages_is_finite(q, structure, model, steel, tube):
    fdm = graph(structure)
    diameters = jnp.full(NUM_EDGES, DIAMETER)

    def objective(q):
        state = equilibrium(q, structure, fdm)
        member = forces(model, state.xyz, diameters, steel, tube)
        return jnp.sum(member.n_ed**2)

    gradient = jax.grad(objective)(q)

    assert jnp.all(jnp.isfinite(gradient))
    assert float(jnp.min(jnp.abs(gradient))) > 0.0
