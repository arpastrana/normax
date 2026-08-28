# SPDX-License-Identifier: Apache-2.0
"""
The OpenSees backend against the smax oracle, which is the point of having two.

Every agreement here is between a C++ solver differentiated by rules compiled
into it and a JAX solver differentiated by tracing. Neither is a reimplementation
of the other, so a tolerance is a measurement rather than a round-trip.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from normax.analysis import opensees
from normax.analysis import smax
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
# 1.4e-15 on the axial force and 9.0e-13 on the moments.
TOLERANCE_PRIMAL = 1e-11

# Hand-derived C++ sensitivities against traced autodiff. Measured at 1.1e-11.
TOLERANCE_JACOBIAN = 1e-9

# Force densities to a loss over the forces, across the boundary and back.
TOLERANCE_GRADIENT = 1e-9

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
def prepared(structure, catalog):
    """
    Both backends' models, prepared once from the same structure.
    """
    return (
        opensees.prepare_model(structure, catalog, NORMAL),
        smax.prepare_model(structure, catalog(SEED)),
    )


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
    ops, traced = prepared
    mine = opensees.compute_member_forces(ops, xyz, diameters, catalog, funicular)
    theirs = smax.compute_member_forces(
        traced, xyz, diameters, catalog(SEED), funicular
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
def traced(prepared, xyz, diameters, catalog, funicular):
    """
    The same derivatives, taken by tracing the oracle.
    """
    _, model = prepared

    def run(coords, sizes):
        member = smax.compute_member_forces(
            model, coords, sizes, catalog(SEED), funicular
        )

        return {"axial_force": member.axial_force, "moment_major": member.moment_major}

    return jax.jacfwd(run, argnums=0)(xyz, diameters), jax.jacfwd(run, argnums=1)(
        xyz, diameters
    )


def test_the_forces_move_with_a_coordinate_as_autodiff_says(blocks, traced):
    by_coordinate, _ = traced

    assert (
        relative(blocks.axial_force_xyz, by_coordinate["axial_force"])
        < TOLERANCE_JACOBIAN
    )
    assert (
        relative(blocks.moment_major_xyz, by_coordinate["moment_major"])
        < TOLERANCE_JACOBIAN
    )


def test_the_forces_move_with_a_diameter_as_autodiff_says(blocks, traced):
    _, by_diameter = traced

    assert (
        relative(blocks.axial_force_diameter, by_diameter["axial_force"])
        < TOLERANCE_JACOBIAN
    )
    assert (
        relative(blocks.moment_major_diameter, by_diameter["moment_major"])
        < TOLERANCE_JACOBIAN
    )


def test_nothing_in_the_plane_moves_when_a_node_leaves_it(
    prepared, xyz, diameters, catalog, funicular
):
    # The separation the two-dimensional model relies on, measured not assumed.
    _, model = prepared

    def run(coords):
        member = smax.compute_member_forces(
            model, coords, diameters, catalog(SEED), funicular
        )

        return {"axial_force": member.axial_force, "moment_major": member.moment_major}

    jacobian = jax.jacfwd(run)(xyz)

    for block in jacobian.values():
        assert np.all(np.asarray(block)[..., NORMAL] == 0.0)


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


def test_the_gradient_is_the_same_whichever_solver_produced_it(
    structure, catalog, densities, diameters, funicular
):
    # Force densities through the form finder, the analysis and a loss over the
    # forces: the DDM sweep contracted into a VJP against the traced oracle.
    finder = FdmFormFinder(structure)
    stacked = jnp.asarray(funicular)[None, ...]

    def loss(analyzer, q):
        forces = analyzer(finder(q, funicular).xyz, diameters, stacked)
        axial = jnp.sum((forces.axial_force / SCALE_FORCE) ** 2)
        major = jnp.sum((forces.moment_major / SCALE_MOMENT) ** 2)

        return axial + major

    crossed = TesseractAnalyzer(structure, catalog, backend="opensees")
    mine = jax.grad(lambda q: loss(crossed, q))(densities)

    traced = smax.SmaxAnalyzer(structure, catalog(SEED))
    theirs = jax.grad(lambda q: loss(traced, q))(densities)

    assert np.max(np.abs(np.asarray(mine))) > 0.0
    assert relative(mine, theirs) < TOLERANCE_GRADIENT
