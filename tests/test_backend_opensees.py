# SPDX-License-Identifier: Apache-2.0
"""
The OpenSees backend against the other shipped solver, and against differences.

Every agreement here is between a C++ solver differentiated by rules compiled
into it and a Python solver differentiated by hand. Neither is a reimplementation
of the other, so a tolerance is a measurement rather than a round-trip. What the
compiled sensitivities are held to is a central difference of the solve itself.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from normax.analysis import MemberForces
from normax.analysis import opensees
from normax.analysis import pynite
from normax.form_finding import FdmFormFinder
from normax.form_finding import build_equilibrium_graph
from normax.form_finding import solve_equilibrium
from normax.loads import create_load_uniform
from normax.materials import Steel355
from normax.sections import build_section_catalog
from normax.structures import build_arch_2d
from normax.tesseract import TesseractAnalyzer

# The same 10 m arch rising 3 m under 180 kN the rest of the suite uses.
SPAN = 10_000.0
RISE = 3_000.0
TOTAL_LOAD = 180_000.0
NUM_EDGES = 10
NORMAL = 1
SEED = 100.0

# Two solvers agreeing on a value they compute independently. Measured at
# 1.3e-15 on the axial force and 1.6e-12 on the moments.
TOLERANCE_PRIMAL = 1e-11

# Compiled sensitivities against a difference of the solve they differentiate,
# where the difference is the loose party. Measured at 9.1e-10 and 7.9e-8.
TOLERANCE_DIFFERENCE = 1e-6

# Force densities to a loss over the forces, across the boundary and back,
# differenced rather than compared. Measured at 7.3e-9.
TOLERANCE_GRADIENT = 1e-7

# Steps at which each difference is least contaminated, measured by sweeping.
STEP_COORDINATE = 1.0e-2
STEP_DIAMETER = 1.0e-2
STEP_DENSITY = 1.0e-4

# Where a coordinate is differenced: a node near the springing, and the crown.
NODES_DIFFERENCED = (1, 5)
AXES_IN_PLANE = (0, 2)

# Where a diameter is differenced: both end members and one in the middle.
MEMBERS_DIFFERENCED = (0, 4, 9)

# Scales that bring both reported quantities to unit order before summing.
SCALE_FORCE = 1.0e5
SCALE_MOMENT = 1.0e8


@pytest.fixture(scope="module")
def catalog():
    return build_section_catalog(Steel355(), 3)


@pytest.fixture(scope="module")
def structure():
    return build_arch_2d(num_edges=NUM_EDGES, span=SPAN, rise=RISE)


@pytest.fixture(scope="module")
def funicular(structure):
    return create_load_uniform(structure, TOTAL_LOAD)


@pytest.fixture(scope="module")
def densities(structure, funicular):
    """
    The uniform force density that reaches the target rise.
    """
    fdm = build_equilibrium_graph(structure)
    trial = jnp.full(NUM_EDGES, -1.0)
    state = solve_equilibrium(trial, structure.nodes[fdm.indices_fixed], fdm, funicular)

    return trial * float(jnp.max(state.xyz[:, 2])) / RISE


@pytest.fixture(scope="module")
def xyz(structure, densities, funicular):
    return FdmFormFinder(structure)(densities, funicular).xyz


@pytest.fixture(scope="module")
def diameters():
    return jnp.full(NUM_EDGES, SEED)


@pytest.fixture(scope="module")
def prepared(structure, catalog, funicular):
    """
    Both shipped solvers' models, prepared once from the same structure.
    """
    plane = opensees.prepare_model(structure, catalog, NORMAL)
    space = pynite.FrameProblem(structure, catalog, np.asarray(funicular))

    return plane, space


def relative(actual, expected):
    """
    Worst absolute gap over the largest entry of the reference.
    """
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    scale = float(np.max(np.abs(expected)))

    return float(np.max(np.abs(actual - expected))) / (scale if scale > 0.0 else 1.0)


@pytest.fixture(scope="module")
def forces(prepared, xyz, diameters, catalog, funicular):
    ops, space = prepared
    mine = opensees.compute_member_forces(ops, xyz, diameters, catalog, funicular)
    theirs = pynite.compute_member_forces(
        space, np.asarray(xyz), np.asarray(diameters), np.asarray(funicular)
    )

    return mine, theirs


def test_the_two_solvers_agree_on_the_axial_force(forces):
    mine, theirs = forces

    assert relative(mine.axial_force, theirs.axial_force) < TOLERANCE_PRIMAL


def test_the_two_solvers_agree_on_both_end_moments(forces):
    mine, theirs = forces

    assert relative(mine.moment_major, theirs.moment_major) < TOLERANCE_PRIMAL


def test_a_plane_frame_carries_no_minor_axis_moment(forces):
    mine, theirs = forces

    assert np.all(np.asarray(mine.moment_minor) == 0.0)
    assert np.all(np.asarray(theirs.moment_minor) == 0.0)


@pytest.fixture(scope="module")
def blocks(prepared, xyz, diameters, catalog, funicular):
    ops, _ = prepared

    return opensees.compute_force_jacobian(ops, xyz, diameters, catalog, funicular)


@pytest.fixture(scope="module")
def solved(prepared, catalog, funicular):
    """
    The forward pass alone, at any geometry and any set of diameters.
    """
    ops, _ = prepared

    def run(coords, sizes):
        return opensees.compute_member_forces(
            ops, jnp.asarray(coords), jnp.asarray(sizes), catalog, funicular
        )

    return run


def test_the_forces_move_with_a_coordinate_as_a_difference_says(
    blocks, solved, xyz, diameters
):
    # The compiled sensitivities against two solves of the model they were
    # compiled into, which is the only witness that shares nothing with them.
    base = np.asarray(xyz)
    scale_axial = float(np.max(np.abs(np.asarray(blocks.axial_force_xyz))))
    scale_moment = float(np.max(np.abs(np.asarray(blocks.moment_major_xyz))))

    for node in NODES_DIFFERENCED:
        for axis in AXES_IN_PLANE:
            up = base.copy()
            down = base.copy()
            up[node, axis] += STEP_COORDINATE
            down[node, axis] -= STEP_COORDINATE
            plus = solved(up, diameters)
            minus = solved(down, diameters)
            axial = np.asarray(plus.axial_force) - np.asarray(minus.axial_force)
            moment = np.asarray(plus.moment_major) - np.asarray(minus.moment_major)
            central_axial = axial / (2.0 * STEP_COORDINATE)
            central_moment = moment / (2.0 * STEP_COORDINATE)
            reported_axial = np.asarray(blocks.axial_force_xyz)[:, node, axis]
            reported_moment = np.asarray(blocks.moment_major_xyz)[:, :, node, axis]

            assert (
                np.max(np.abs(reported_axial - central_axial)) / scale_axial
                < TOLERANCE_DIFFERENCE
            )
            assert (
                np.max(np.abs(reported_moment - central_moment)) / scale_moment
                < TOLERANCE_DIFFERENCE
            )


def test_the_forces_move_with_a_diameter_as_a_difference_says(
    blocks, solved, xyz, diameters
):
    base = np.asarray(diameters)
    scale_axial = float(np.max(np.abs(np.asarray(blocks.axial_force_diameter))))
    scale_moment = float(np.max(np.abs(np.asarray(blocks.moment_major_diameter))))

    for member in MEMBERS_DIFFERENCED:
        fatter = base.copy()
        thinner = base.copy()
        fatter[member] += STEP_DIAMETER
        thinner[member] -= STEP_DIAMETER
        plus = solved(xyz, fatter)
        minus = solved(xyz, thinner)
        axial = np.asarray(plus.axial_force) - np.asarray(minus.axial_force)
        moment = np.asarray(plus.moment_major) - np.asarray(minus.moment_major)
        central_axial = axial / (2.0 * STEP_DIAMETER)
        central_moment = moment / (2.0 * STEP_DIAMETER)
        reported_axial = np.asarray(blocks.axial_force_diameter)[:, member]
        reported_moment = np.asarray(blocks.moment_major_diameter)[:, :, member]

        assert (
            np.max(np.abs(reported_axial - central_axial)) / scale_axial
            < TOLERANCE_DIFFERENCE
        )
        assert (
            np.max(np.abs(reported_moment - central_moment)) / scale_moment
            < TOLERANCE_DIFFERENCE
        )


def test_nothing_in_the_plane_moves_when_a_node_leaves_it(prepared, xyz, diameters):
    # The separation the two-dimensional model relies on, measured not assumed,
    # through the shipped solver that has a third dimension to lose it in.
    _, space = prepared
    generator = np.random.default_rng(4711)
    seed = MemberForces(
        generator.normal(size=NUM_EDGES),
        generator.normal(size=(NUM_EDGES, 2)),
        np.zeros((NUM_EDGES, 2)),
    )
    pulled = pynite.pull_back_cotangents(
        space, np.asarray(xyz), np.asarray(diameters), seed
    )
    reached = np.asarray(pulled.xyz)

    assert np.all(reached[:, NORMAL] == 0.0)
    assert float(np.max(np.abs(reached))) > 0.0


def test_a_three_dimensional_frame_is_refused(structure, catalog):
    with pytest.raises(ValueError, match="planar"):
        opensees.prepare_model(structure, catalog, None)


def test_a_frame_that_is_not_flat_is_refused(
    prepared, xyz, diameters, catalog, funicular
):
    ops, _ = prepared
    warped = jnp.asarray(xyz).at[1, NORMAL].set(500.0)

    with pytest.raises(ValueError, match="not planar"):
        opensees.compute_member_forces(ops, warped, diameters, catalog, funicular)


def test_a_load_out_of_the_plane_is_refused(
    prepared, xyz, diameters, catalog, funicular
):
    ops, _ = prepared
    pushed = jnp.asarray(funicular).at[1, NORMAL].set(1_000.0)

    with pytest.raises(ValueError, match="normal axis"):
        opensees.compute_member_forces(ops, xyz, diameters, catalog, pushed)


def test_the_environment_selects_the_backend(
    structure, catalog, xyz, diameters, funicular, forces
):
    crossed = TesseractAnalyzer(structure, catalog, backend="opensees")
    served = crossed(xyz, diameters, jnp.asarray(funicular)[None, ...])
    mine, _ = forces

    assert crossed.backend == "opensees"
    assert np.array_equal(
        np.asarray(served.axial_force[0]), np.asarray(mine.axial_force)
    )
    assert np.array_equal(
        np.asarray(served.moment_major[0]), np.asarray(mine.moment_major)
    )


def test_the_gradient_matches_a_difference_of_the_crossed_route(
    structure, catalog, densities, diameters, funicular
):
    # Force densities through the form finder, the analysis and a loss over the
    # forces: the DDM sweep contracted into a VJP, against two runs of itself.
    finder = FdmFormFinder(structure)
    stacked = jnp.asarray(funicular)[None, ...]
    crossed = TesseractAnalyzer(structure, catalog, backend="opensees")

    def loss(q):
        forces = crossed(finder(q, funicular).xyz, diameters, stacked)
        axial = jnp.sum((forces.axial_force / SCALE_FORCE) ** 2)
        major = jnp.sum((forces.moment_major / SCALE_MOMENT) ** 2)

        return axial + major

    gradient = jax.grad(loss)(densities)
    scale = float(jnp.max(jnp.abs(gradient)))

    assert scale > 0.0

    for edge in (0, NUM_EDGES // 2):
        step = abs(float(densities[edge])) * STEP_DENSITY
        plus = loss(densities.at[edge].add(step))
        minus = loss(densities.at[edge].add(-step))
        central = float((plus - minus) / (2.0 * step))

        assert abs(float(gradient[edge]) - central) / scale < TOLERANCE_GRADIENT
