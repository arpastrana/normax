# SPDX-License-Identifier: Apache-2.0
"""
A foreign solver, given an adjoint it does not have.

PyNite assembles, factorizes and solves; it carries no derivative of its own.
The rule that differentiates it rests on two claims that have to be measured:
the element here is the element it assembled, and the rule agrees with a
central difference of the very forward pass it differentiates.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from Pynite import FEModel3D

from normax.analysis import MemberForces
from normax.analysis import pynite
from normax.analysis.element import SectionRigidity
from normax.analysis.element import assemble_stiffness_global
from normax.analysis.element import assemble_stiffness_local
from normax.analysis.element import compute_direction_cosines
from normax.materials import Steel355
from normax.sections import build_section_catalog
from normax.structures import Structure
from normax.tesseract import TesseractAnalyzer

SECTION_CLASS = 3
SEED_DIAMETER = 100.0

# The replica and the solver do the same arithmetic in a different order.
TOLERANCE_ELEMENT = 1e-13

# The boundary transports one backend's own answer, so round-off and no more.
TOLERANCE_GRADIENT = 1e-11

# A central difference is the loose party in that comparison, not the rule.
TOLERANCE_DIFFERENCE = 1e-7

# Where a differenced coordinate is least contaminated, measured by sweeping it.
STEP_COORDINATE = 1.0e-2

# The same sweep over a diameter, in millimeters of a hundred-odd.
STEP_DIAMETER = 1.0e-2

# The nodes no support holds, which are the ones a difference may move.
NODES_FREE = (4, 5)

# Scales that bring both reported quantities to unit order before summing.
SCALE_FORCE = 1.0e5
SCALE_MOMENT = 1.0e8


@pytest.fixture(scope="module")
def catalog():
    return build_section_catalog(Steel355(), SECTION_CLASS)


@pytest.fixture(scope="module")
def canopy():
    """
    A frame no plane contains, with a member on every kind of slope.
    """
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [4000.0, 0.0, 0.0],
            [4000.0, 3000.0, 0.0],
            [0.0, 3000.0, 0.0],
            [2000.0, 1500.0, 2500.0],
            [1000.0, 800.0, 1400.0],
        ]
    )
    edges = np.array([[0, 4], [1, 4], [2, 4], [3, 4], [0, 1], [1, 2], [0, 5], [5, 4]])

    return Structure(nodes=nodes, edges=edges, supports=np.array([0, 1, 2, 3]))


@pytest.fixture(scope="module")
def canopy_loads(canopy):
    loads = np.zeros_like(np.asarray(canopy.nodes))
    loads[4] = (3.0e4, -2.0e4, -5.0e4)
    loads[5] = (0.0, 1.0e4, -2.0e4)

    return loads


@pytest.fixture(scope="module")
def canopy_diameters(canopy):
    return 100.0 + np.arange(canopy.num_edges) * 7.0


@pytest.fixture(scope="module")
def problem(canopy, canopy_loads, catalog):
    return pynite.FrameProblem(structure=canopy, catalog=catalog, loads=canopy_loads)


def solved_member(start, end, section, moduli):
    """
    One member the foreign solver has built and analyzed, ready to read.
    """
    area, inertia, torsion = section
    elasticity, shear, poissons = moduli
    model = FEModel3D()
    model.add_material("steel", elasticity, shear, poissons, 7850.0)
    model.add_section("tube", area, inertia, inertia, torsion)
    model.add_node("a", *start)
    model.add_node("b", *end)
    model.def_support("a", True, True, True, True, True, True)
    model.add_member("m", "a", "b", "steel", "tube")
    model.add_node_load("b", "FY", -1000.0, case="c")
    model.add_load_combo("lc", {"c": 1.0})
    model.analyze_linear(check_stability=False)

    return next(iter(model.members["m"].sub_members.values()))


def relative(mine, reference):
    """
    Worst absolute gap, against the largest entry of the reference.
    """
    gap = np.max(np.abs(np.asarray(mine) - np.asarray(reference)))
    scale = max(float(np.max(np.abs(np.asarray(reference)))), 1e-300)

    return float(gap) / scale


def scaled_loss(forces):
    """
    One scalar over every reported force, each brought to unit order.
    """
    axial = jnp.sum((forces.axial_force / SCALE_FORCE) ** 2)
    major = jnp.sum((forces.moment_major / SCALE_MOMENT) ** 2)
    minor = jnp.sum((forces.moment_minor / SCALE_MOMENT) ** 2)

    return axial + major + minor


def loss_cotangent(forces):
    """
    The derivative of `scaled_loss` in every reported force.
    """
    return MemberForces(
        2.0 * np.asarray(forces.axial_force) / SCALE_FORCE**2,
        2.0 * np.asarray(forces.moment_major) / SCALE_MOMENT**2,
        2.0 * np.asarray(forces.moment_minor) / SCALE_MOMENT**2,
    )


def test_the_element_is_the_one_the_solver_assembled():
    # The claim the whole adjoint rests on: differentiating the replica is
    # differentiating the foreign model, not a second opinion about it.
    elasticity = 210.0e9
    poissons = 0.3
    shear = elasticity / (2.0 * (1.0 + poissons))
    moduli = (elasticity, shear, poissons)
    generator = np.random.default_rng(20260825)

    for _ in range(25):
        start = generator.normal(size=3) * 5.0
        end = start + generator.normal(size=3) * 4.0
        area = float(generator.uniform(5.0e-4, 5.0e-3))
        inertia = float(generator.uniform(1.0e-6, 5.0e-5))
        theirs = solved_member(start, end, (area, inertia, 2.0 * inertia), moduli)
        rigidity = SectionRigidity(
            axial=jnp.asarray(elasticity * area),
            bending=jnp.asarray(elasticity * inertia),
            torsional=jnp.asarray(shear * 2.0 * inertia),
        )
        length = jnp.asarray(float(np.linalg.norm(end - start)))

        local = assemble_stiffness_local(length, rigidity)
        assert relative(local, theirs.ke()) < TOLERANCE_ELEMENT

        spanned = assemble_stiffness_global(
            jnp.asarray(start), jnp.asarray(end), rigidity
        )
        assert relative(spanned, theirs.Ke()) < TOLERANCE_ELEMENT


def test_rolling_the_frame_leaves_the_global_stiffness_alone():
    # Why the stiffness frame may be chosen for conditioning and read by nobody.
    rigidity = SectionRigidity(
        axial=jnp.asarray(2.5e8),
        bending=jnp.asarray(1.0e4),
        torsional=jnp.asarray(8.0e3),
    )
    start = jnp.asarray([0.0, 0.0, 0.0])
    end = jnp.asarray([3.0, 1.0, 2.0])
    spanned = assemble_stiffness_global(start, end, rigidity)

    axis = np.asarray(end - start) / np.linalg.norm(np.asarray(end - start))
    frame = np.asarray(compute_direction_cosines(start, end))
    angle = 0.7
    turned = np.stack(
        [
            axis,
            np.cos(angle) * frame[1] + np.sin(angle) * frame[2],
            -np.sin(angle) * frame[1] + np.cos(angle) * frame[2],
        ]
    )
    transform = np.kron(np.eye(4), turned)
    local = np.asarray(assemble_stiffness_local(jnp.linalg.norm(end - start), rigidity))
    rolled = transform.T @ local @ transform

    assert relative(spanned, rolled) < TOLERANCE_ELEMENT


def test_the_adjoint_agrees_with_a_difference_in_every_parameter(
    canopy, canopy_loads, canopy_diameters, problem
):
    # The gate. One scalar, differenced in every coordinate a difference may
    # move and in every diameter, against the rule that claims to know it.
    nodes = np.asarray(canopy.nodes)
    forces = pynite.compute_member_forces(
        problem, nodes, canopy_diameters, canopy_loads
    )
    pulled = pynite.pull_back_cotangents(
        problem, nodes, canopy_diameters, loss_cotangent(forces)
    )
    scale_coordinate = float(np.max(np.abs(np.asarray(pulled.xyz))))
    scale_diameter = float(np.max(np.abs(np.asarray(pulled.diameter))))

    def worked(xyz, diameters):
        carried = pynite.compute_member_forces(problem, xyz, diameters, canopy_loads)

        return float(scaled_loss(carried))

    for node in NODES_FREE:
        for axis in range(3):
            up = nodes.copy()
            down = nodes.copy()
            up[node, axis] += STEP_COORDINATE
            down[node, axis] -= STEP_COORDINATE
            differenced = worked(up, canopy_diameters) - worked(down, canopy_diameters)
            central = differenced / (2.0 * STEP_COORDINATE)
            gap = abs(float(pulled.xyz[node, axis]) - central) / scale_coordinate

            assert gap < TOLERANCE_DIFFERENCE

    for member in range(canopy.num_edges):
        up = np.asarray(canopy_diameters).copy()
        down = np.asarray(canopy_diameters).copy()
        up[member] += STEP_DIAMETER
        down[member] -= STEP_DIAMETER
        differenced = worked(nodes, up) - worked(nodes, down)
        central = differenced / (2.0 * STEP_DIAMETER)
        gap = abs(float(pulled.diameter[member]) - central) / scale_diameter

        assert gap < TOLERANCE_DIFFERENCE


def test_the_adjoint_survives_a_central_difference(
    canopy, canopy_loads, canopy_diameters, problem
):
    # A second opinion that shares nothing with the first: no element replica,
    # no adjoint, just the forward pass twice.
    node = 4
    axis = 2
    generator = np.random.default_rng(4711)
    members = canopy.num_edges
    seed = MemberForces(
        generator.normal(size=members),
        generator.normal(size=(members, 2)),
        generator.normal(size=(members, 2)),
    )
    exact = pynite.pull_back_cotangents(
        problem, np.asarray(canopy.nodes), canopy_diameters, seed
    )

    def seeded(xyz):
        forces = pynite.compute_member_forces(
            problem, xyz, canopy_diameters, canopy_loads
        )
        products = jax.tree.map(
            lambda cotangent, value: cotangent * value, seed, forces
        )

        return sum(float(np.sum(np.asarray(leaf))) for leaf in products)

    up = np.asarray(canopy.nodes).copy()
    down = np.asarray(canopy.nodes).copy()
    up[node, axis] += STEP_COORDINATE
    down[node, axis] -= STEP_COORDINATE
    differenced = (seeded(up) - seeded(down)) / (2.0 * STEP_COORDINATE)

    assert (
        abs(exact.xyz[node, axis] - differenced) / abs(differenced)
        < TOLERANCE_DIFFERENCE
    )


def test_the_gradient_survives_the_boundary(
    canopy, canopy_loads, canopy_diameters, catalog, problem
):
    # What the submission claims: a solver with no derivative of its own,
    # reached across a schema, handing back a reverse-mode gradient. The
    # boundary is held to the backend called in process, which is its job.
    stacked = jnp.asarray(canopy_loads)[None, ...]
    diameters = jnp.asarray(canopy_diameters)
    crossed = TesseractAnalyzer(canopy, catalog, backend="pynite")

    def loss(xyz, sizes):
        return scaled_loss(crossed(xyz, sizes, stacked))

    served = float(loss(canopy.nodes, diameters))
    foreign = jax.grad(loss, argnums=(0, 1))(canopy.nodes, diameters)

    nodes = np.asarray(canopy.nodes)
    direct = pynite.compute_member_forces(
        problem, nodes, canopy_diameters, canopy_loads
    )
    expected = float(scaled_loss(direct))
    reference = pynite.pull_back_cotangents(
        problem, nodes, canopy_diameters, loss_cotangent(direct)
    )

    assert abs(served - expected) / abs(expected) < TOLERANCE_GRADIENT
    assert relative(foreign[0], reference.xyz) < TOLERANCE_GRADIENT
    assert relative(foreign[1], reference.diameter) < TOLERANCE_GRADIENT


def test_a_section_of_no_area_is_refused(canopy, canopy_loads, problem):
    with pytest.raises(ValueError, match="not positive"):
        pynite.compute_member_forces(
            problem, np.asarray(canopy.nodes), np.zeros(canopy.num_edges), canopy_loads
        )


def test_a_vertical_member_is_refused(catalog):
    # The reporting convention completes its transverse pair against the
    # vertical, so a vertical member has no pair to report a bending in.
    nodes = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3000.0], [2000.0, 0.0, 3000.0]])
    edges = np.array([[0, 1], [1, 2]])
    mast = Structure(nodes=nodes, edges=edges, supports=np.array([0]))
    loads = np.zeros_like(nodes)
    loads[2, 2] = -1.0e4
    problem = pynite.FrameProblem(structure=mast, catalog=catalog, loads=loads)

    with pytest.raises(ValueError, match="vertical"):
        pynite.compute_member_forces(problem, nodes, np.full(2, SEED_DIAMETER), loads)


def test_several_load_cases_cross_together(
    canopy, canopy_loads, canopy_diameters, catalog
):
    # The backend remembers one factorized frame between endpoint calls, and the
    # adjoints run after all the forwards, so a cache keyed on anything the
    # loads touch would answer the second case with the first case's solve.
    sideways = np.zeros_like(canopy_loads)
    sideways[4] = (-4.0e4, 1.5e4, -1.0e4)
    every_case = np.stack([canopy_loads, sideways, 0.4 * canopy_loads])
    stacked = jnp.asarray(every_case)
    diameters = jnp.asarray(canopy_diameters)
    crossed = TesseractAnalyzer(canopy, catalog, backend="pynite")

    def loss(xyz, sizes):
        forces = crossed(xyz, sizes, stacked)
        weighted = jnp.arange(1, forces.axial_force.shape[0] + 1)[:, None]
        axial = jnp.sum(weighted * (forces.axial_force / SCALE_FORCE) ** 2)
        major = jnp.sum((forces.moment_major / SCALE_MOMENT) ** 2)
        minor = jnp.sum((forces.moment_minor / SCALE_MOMENT) ** 2)

        return axial + major + minor

    served = crossed(canopy.nodes, diameters, stacked)
    foreign = jax.grad(loss, argnums=(0, 1))(canopy.nodes, diameters)

    # The three cases must stay distinct, or a cache has served one for another.
    spread = np.asarray(served.axial_force)
    assert relative(spread[0], spread[1]) > 0.1
    assert relative(spread[0], spread[2]) > 0.1

    nodes = np.asarray(canopy.nodes)
    for index, applied in enumerate(every_case):
        alone = pynite.FrameProblem(
            structure=canopy, catalog=catalog, loads=np.asarray(applied)
        )
        direct = pynite.compute_member_forces(
            alone, nodes, canopy_diameters, np.asarray(applied)
        )

        assert relative(spread[index], direct.axial_force) < TOLERANCE_GRADIENT
        assert (
            relative(served.moment_major[index], direct.moment_major)
            < TOLERANCE_GRADIENT
        )

    node = 4
    axis = 2
    member = 3
    scale_coordinate = float(np.max(np.abs(np.asarray(foreign[0]))))
    scale_diameter = float(np.max(np.abs(np.asarray(foreign[1]))))
    moved_up = jnp.asarray(nodes).at[node, axis].add(STEP_COORDINATE)
    moved_down = jnp.asarray(nodes).at[node, axis].add(-STEP_COORDINATE)
    fatter = diameters.at[member].add(STEP_DIAMETER)
    thinner = diameters.at[member].add(-STEP_DIAMETER)
    by_coordinate = float(loss(moved_up, diameters) - loss(moved_down, diameters))
    by_diameter = float(loss(nodes, fatter) - loss(nodes, thinner))
    central_coordinate = by_coordinate / (2.0 * STEP_COORDINATE)
    central_diameter = by_diameter / (2.0 * STEP_DIAMETER)

    assert (
        abs(float(foreign[0][node, axis]) - central_coordinate) / scale_coordinate
        < TOLERANCE_DIFFERENCE
    )
    assert (
        abs(float(foreign[1][member]) - central_diameter) / scale_diameter
        < TOLERANCE_DIFFERENCE
    )


def test_a_second_frame_is_not_answered_by_the_first(
    canopy, canopy_loads, catalog, problem
):
    # Two geometries interleaved in one process: the fingerprint has to tell
    # them apart, and the second must not be served the first's factorization.
    moved = np.asarray(canopy.nodes).copy()
    moved[4, 2] += 900.0
    shifted = Structure(
        nodes=moved,
        edges=np.asarray(canopy.edges),
        supports=np.asarray(canopy.supports),
    )
    diameters = np.full(canopy.num_edges, SEED_DIAMETER)
    second = pynite.FrameProblem(structure=shifted, catalog=catalog, loads=canopy_loads)

    here = TesseractAnalyzer(canopy, catalog, backend="pynite")
    there = TesseractAnalyzer(shifted, catalog, backend="pynite")
    sizes = jnp.asarray(diameters)
    stacked = jnp.asarray(canopy_loads)[None, ...]
    served_here = here(canopy.nodes, sizes, stacked)
    served_there = there(shifted.nodes, sizes, stacked)
    served_again = here(canopy.nodes, sizes, stacked)

    direct_here = pynite.compute_member_forces(
        problem, np.asarray(canopy.nodes), diameters, canopy_loads
    )
    direct_there = pynite.compute_member_forces(second, moved, diameters, canopy_loads)

    assert (
        relative(served_here.axial_force[0], direct_here.axial_force)
        < TOLERANCE_GRADIENT
    )
    assert (
        relative(served_there.axial_force[0], direct_there.axial_force)
        < TOLERANCE_GRADIENT
    )
    assert (
        relative(served_again.axial_force[0], direct_here.axial_force)
        < TOLERANCE_GRADIENT
    )
    assert relative(served_here.axial_force[0], served_there.axial_force[0]) > 0.01
