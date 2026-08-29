# SPDX-License-Identifier: Apache-2.0
"""
The handoff form finding makes, and what a frame solve makes of it.

A form finder hands over a geometry alone, so the axial forces that come back
are the analysis's own product. That they reproduce the funicular force density
times the length, with bending demonstrably secondary, is a prediction about the
handoff rather than a restatement of it — and it does not care which solver
reports the forces, so the crossed frame analysis reports them here.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from normax.analysis import DOF_PER_NODE
from normax.analysis import find_normal_axis
from normax.analysis import restrain_supports
from normax.analysis.element import SectionRigidity
from normax.analysis.element import assemble_stiffness_global
from normax.analysis.pynite import POISSONS_RATIO
from normax.analysis.pynite import TORSION_FACTOR
from normax.form_finding import build_equilibrium_graph
from normax.form_finding import solve_equilibrium
from normax.loads import create_load_uniform
from normax.loads import select_load_case
from normax.materials import Steel355
from normax.sections import build_section_catalog
from normax.structures import build_arch_2d
from normax.structures import build_gridshell_3d
from normax.tesseract import TesseractAnalyzer

# A 10 m arch of ten members, rising about a third of its span under a force
# density of 75 N/mm and a 20 kN load at every free node. Units are mm and N.
SPAN = 10_000.0
LOAD = 20_000.0
NUM_EDGES = 10
FORCE_DENSITY = -75.0

# The arch lies in the XZ plane, so it has no thickness along Y.
NORMAL = 1

# Close to the 73 to 87 mm the code check asks for on this arch.
DIAMETER = 100.0

# Measured, not chosen. The axial forces of the unstressed frame reproduce the
# funicular ones to 2.27e-4, and the largest end moment is 7.58e-4 of the axial
# force times the length.
TOLERANCE_AXIAL = 2.5e-4
TOLERANCE_BENDING = 1.0e-3

# How small an eigenvalue counts as no stiffness at all. The arch's own smallest
# restrained mode sits eight orders above this.
TOLERANCE_SLACK = 1.0e-10


@pytest.fixture(scope="module")
def structure():
    return build_arch_2d(num_edges=NUM_EDGES, span=SPAN, rise=SPAN / 3.0)


def funicular(structure, load=LOAD):
    """
    The uniform load case the arch is form-found under.
    """
    return create_load_uniform(structure, load * (NUM_EDGES - 1))


@pytest.fixture(scope="module")
def steel():
    return Steel355()


@pytest.fixture(scope="module")
def catalog(steel):
    return build_section_catalog(steel, 3)


@pytest.fixture(scope="module")
def sections(catalog):
    return catalog(jnp.full(NUM_EDGES, DIAMETER))


@pytest.fixture(scope="module")
def analyzer(structure, catalog):
    return TesseractAnalyzer(structure, catalog, "pynite")


@pytest.fixture(scope="module")
def q():
    return jnp.full(NUM_EDGES, FORCE_DENSITY)


@pytest.fixture(scope="module")
def state(q, structure):
    graph = build_equilibrium_graph(structure)

    return solve_equilibrium(
        q, structure.nodes[graph.indices_fixed], graph, funicular(structure)
    )


@pytest.fixture(scope="module")
def member(analyzer, state, structure):
    diameters = jnp.full(NUM_EDGES, DIAMETER)
    stacked = funicular(structure)[None, ...]
    forces = analyzer(state.xyz, diameters, stacked)

    return select_load_case(forces, 0)


@pytest.fixture(scope="module")
def stiffness(structure, state, sections):
    return assemble_frame_stiffness(structure, state.xyz, sections)


def assemble_frame_stiffness(structure, xyz, sections):
    """
    The whole frame's elastic stiffness, from the element every backend shares.
    """
    steel = sections.material
    elasticity = float(steel.e_mod)
    shear = elasticity / (2.0 * (1.0 + POISSONS_RATIO))
    positions = np.asarray(xyz)
    edges = np.asarray(structure.edges)
    within = np.arange(DOF_PER_NODE)

    size = positions.shape[0] * DOF_PER_NODE
    assembled = np.zeros((size, size))
    for member in range(edges.shape[0]):
        inertia = float(sections.second_moment[member])
        rigidity = SectionRigidity(
            axial=jnp.asarray(elasticity * float(sections.area[member])),
            bending=jnp.asarray(elasticity * inertia),
            torsional=jnp.asarray(shear * TORSION_FACTOR * inertia),
        )
        start = jnp.asarray(positions[edges[member, 0]])
        end = jnp.asarray(positions[edges[member, 1]])
        spanned = np.asarray(assemble_stiffness_global(start, end, rigidity))

        first = edges[member, 0] * DOF_PER_NODE + within
        second = edges[member, 1] * DOF_PER_NODE + within
        indexed = np.concatenate([first, second])
        block = np.ix_(indexed, indexed)
        assembled[block] += spanned

    return assembled


def count_slack_modes(stiffness, flags):
    """
    How many zero-energy modes one set of restraints leaves behind.
    """
    held = np.asarray(flags).ravel()
    free = np.flatnonzero(~held)
    block = np.ix_(free, free)
    spectrum = np.linalg.eigvalsh(stiffness[block])
    slack = spectrum < TOLERANCE_SLACK * float(spectrum.max())

    return int(np.count_nonzero(slack))


def restrain_translation(structure):
    """
    Support flags holding translation and nothing else, the pinned base alone.
    """
    nodes = np.asarray(structure.nodes).shape[0]
    flags = np.zeros((nodes, DOF_PER_NODE), dtype=bool)
    flags[np.asarray(structure.supports), :3] = True

    return flags


# --------------------------------------------------------------------------- #
# The model the analysis runs on
# --------------------------------------------------------------------------- #
def test_the_arch_is_not_a_mechanism_once_the_plane_is_restrained(stiffness, structure):
    assert count_slack_modes(stiffness, restrain_supports(structure)) == 0


def test_a_planar_arch_on_pinned_supports_alone_is_a_mechanism(stiffness, structure):
    # Rotating the whole arch about the line joining its supports strains no
    # member and moves no support, so one mode of the frame carries no stiffness.
    assert count_slack_modes(stiffness, restrain_translation(structure)) == 1


def test_the_plane_of_the_arch_is_measured_rather_than_declared(structure):
    assert find_normal_axis(structure) == NORMAL


def test_a_structure_that_fills_space_has_no_normal_axis():
    assert find_normal_axis(build_gridshell_3d()) is None


def test_a_support_is_pinned_and_never_fixed(structure):
    # The rotation the in-plane bending happens about stays free; the two out of
    # the plane are held so that a straight structure is not a mechanism.
    flags = restrain_supports(structure)
    supports = np.asarray(structure.supports)
    out_of_plane = [3 + axis for axis in (0, 1, 2) if axis != NORMAL]

    assert np.all(flags[supports, :3] == True)
    assert np.all(flags[supports, 3 + NORMAL] == False)
    assert np.all(flags[np.ix_(supports, out_of_plane)] == True)


def test_a_free_node_is_restrained_only_out_of_the_plane(structure):
    flags = restrain_supports(structure)
    free = [n for n in range(structure.nodes.shape[0]) if n not in (0, NUM_EDGES)]

    for node in free:
        assert flags[node, NORMAL] == True
        assert np.count_nonzero(flags[node]) == 1


def test_a_three_dimensional_structure_restrains_nothing_beyond_its_supports():
    gridshell = build_gridshell_3d()
    flags = restrain_supports(gridshell)

    assert np.count_nonzero(flags) == 3 * gridshell.supports.shape[0]


def test_a_structure_held_nowhere_is_refused(structure):
    with pytest.raises(ValueError):
        restrain_supports(structure._replace(supports=np.zeros(0, dtype=int)))


# --------------------------------------------------------------------------- #
# What form finding hands over
# --------------------------------------------------------------------------- #
def test_form_finding_balances_the_loads_at_every_free_node(state):
    free = jnp.asarray([n for n in range(NUM_EDGES + 1) if n not in (0, NUM_EDGES)])

    assert float(jnp.max(jnp.abs(state.residuals[free]))) < 1e-8


def test_every_member_carries_its_force_density_times_its_length(q, state):
    assert np.allclose(state.forces[:, 0], q * state.lengths[:, 0], rtol=1e-14)


def test_the_form_found_arch_lies_in_the_plane_it_started_in_and_rises(state):
    assert float(jnp.max(jnp.abs(state.xyz[:, 1]))) == 0.0
    assert float(jnp.min(state.xyz[:, 2])) == 0.0
    assert float(jnp.max(state.xyz[:, 2])) > 0.0


# --------------------------------------------------------------------------- #
# The gate: the analysis reproduces the funicular state
# --------------------------------------------------------------------------- #
def test_the_analysis_reproduces_the_funicular_axial_forces(q, state, member):
    funicular = q * state.lengths[:, 0]
    gap = jnp.abs(member.axial_force - funicular) / jnp.abs(funicular)

    assert float(jnp.max(gap)) < TOLERANCE_AXIAL
    assert np.all(jnp.sign(member.axial_force) == jnp.sign(funicular))


def test_bending_is_secondary_to_axial_action(q, state, member):
    funicular = q * state.lengths[:, 0]
    peak = jnp.max(jnp.abs(member.moment_major), axis=1)
    ratio = peak / jnp.abs(funicular * state.lengths[:, 0])

    assert float(jnp.max(ratio)) < TOLERANCE_BENDING


def test_an_arch_in_a_plane_carries_no_minor_axis_moment(member):
    assert float(jnp.max(jnp.abs(member.moment_minor))) == 0.0


def test_the_end_moments_of_neighboring_members_agree(member):
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
def deviation(diameter, steel, catalog, load=LOAD, force_density=FORCE_DENSITY):
    """
    Largest relative gap between the analyzed and the funicular axial force.
    """
    structure = build_arch_2d(num_edges=NUM_EDGES, span=SPAN, rise=SPAN / 3.0)
    applied = funicular(structure, load)
    q = jnp.full(NUM_EDGES, force_density)
    graph = build_equilibrium_graph(structure)
    state = solve_equilibrium(q, structure.nodes[graph.indices_fixed], graph, applied)

    graded = catalog._replace(material=steel)
    analyzer = TesseractAnalyzer(structure, graded, "pynite")
    sizes = jnp.full(NUM_EDGES, diameter)
    stacked = applied[None, ...]
    analyzed = select_load_case(analyzer(state.xyz, sizes, stacked), 0)
    expected = q * state.lengths[:, 0]
    gap = jnp.abs(analyzed.axial_force - expected) / jnp.abs(expected)

    return float(jnp.max(gap))


@pytest.mark.parametrize("diameter", [50.0, 100.0, 200.0])
def test_the_gap_is_quadratic_in_the_diameter(diameter, steel, catalog):
    # A beam chain through a funicular polygon cannot turn a kink on axial force
    # alone, and the bending it needs scales as the square of the radius of
    # gyration over the length.
    reference = deviation(100.0, steel, catalog)
    scaled = reference * (diameter / 100.0) ** 2

    assert deviation(diameter, steel, catalog) == pytest.approx(scaled, rel=0.01)


@pytest.mark.parametrize("e_mod", [70_000.0, 400_000.0])
def test_the_gap_does_not_depend_on_the_modulus(e_mod, steel, catalog):
    softer = deviation(DIAMETER, steel._replace(e_mod=e_mod), catalog)

    assert softer == pytest.approx(deviation(DIAMETER, steel, catalog), rel=1e-9)


@pytest.mark.parametrize("scale", [0.1, 10.0])
def test_the_gap_does_not_depend_on_the_scale_of_the_loading(scale, steel, catalog):
    scaled = deviation(
        DIAMETER,
        steel,
        catalog,
        load=LOAD * scale,
        force_density=FORCE_DENSITY * scale,
    )

    assert scaled == pytest.approx(deviation(DIAMETER, steel, catalog), rel=1e-9)


# --------------------------------------------------------------------------- #
# The gradient crosses both stages
# --------------------------------------------------------------------------- #
def test_the_gradient_through_both_stages_matches_central_differences(
    q, structure, analyzer
):
    # Reverse mode only: the analysis stage serves no forward rule, so the
    # crossed derivative is checked against a difference quotient instead.
    fdm = build_equilibrium_graph(structure)
    diameters = jnp.full(NUM_EDGES, DIAMETER)
    applied = funicular(structure)
    stacked = applied[None, ...]

    def objective(q):
        state = solve_equilibrium(q, structure.nodes[fdm.indices_fixed], fdm, applied)
        forces = analyzer(state.xyz, diameters, stacked)
        return jnp.sum(forces.axial_force**2)

    gradient = jax.grad(objective)(q)

    assert jnp.all(jnp.isfinite(gradient))
    assert float(jnp.min(jnp.abs(gradient))) > 0.0

    step = 1e-3
    for edge in (0, NUM_EDGES // 2):
        plus = objective(q.at[edge].add(step))
        minus = objective(q.at[edge].add(-step))
        central = (plus - minus) / (2.0 * step)

        assert float(gradient[edge]) == pytest.approx(float(central), rel=1e-7)
