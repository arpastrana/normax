# Copyright 2026 Rafael Pastrana
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The OpenSees backend against the smax one, which is the point of having two.

Every agreement here is between a C++ solver differentiated by rules compiled
into it and a JAX solver differentiated by tracing. Neither is a reimplementation
of the other, so a tolerance is a measurement rather than a round-trip.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from normax.analysis import opensees as backend_opensees
from normax.analysis.smax import member_forces as forces_smax
from normax.analysis.smax import prepare_model as prepare_smax
from normax.design import DesignParameters
from normax.design import StructuralDesignPipeline
from normax.design import compute_mass
from normax.form_finding.fdm import equilibrium_graph
from normax.form_finding.fdm import equilibrium_state
from normax.loads import assemble_load_cases as load_cases_of
from normax.loads import loads_uniform
from normax.materials import Steel355
from normax.sizing.ec3 import thinnest_family
from normax.structures import build_arch_2d
from normax.tesseract import TesseractAnalyzer
from normax.tesseract import TesseractFormFinder
from normax.tesseract import TesseractSizer
from normax.tesseract import analysis_backend
from normax.tesseract import local_chain

# The same 10 m arch rising 3 m under 180 kN the rest of the suite uses.
SPAN = 10_000.0
RISE = 3_000.0
TOTAL_LOAD = 180_000.0
NUM_EDGES = 10
NORMAL = 1
SEED = 100.0

# Two solvers agreeing on a value they compute independently. Measured at
# 1.4e-15 on the axial force and 9.0e-13 on the moments, the latter being
# larger only because a moment is a difference of larger numbers.
TOLERANCE_PRIMAL = 1e-11

# Hand-derived C++ sensitivities against traced autodiff, worst over every
# block. Measured at 1.1e-11, against the 1e-6 the roadmap asked for.
TOLERANCE_JACOBIAN = 1e-9

# End to end, force densities to a mass. Measured at 1.7e-15 on the mass and
# 3.0e-12 on its gradient.
TOLERANCE_MASS = 1e-12
TOLERANCE_GRADIENT = 1e-9


@pytest.fixture(scope="module")
def steel():
    return Steel355()


@pytest.fixture(scope="module")
def catalogue(steel):
    # The class-limit wall proportion, as bare geometry: both backends read the
    # ratio and the grade, and neither has any use for the class.
    return thinnest_family(steel, 3)


@pytest.fixture(scope="module")
def setup():
    """
    The arch, its connectivity, and the `q` that reaches the target rise.
    """
    structure = build_arch_2d(num_edges=NUM_EDGES, span=SPAN, rise=RISE)
    fdm = equilibrium_graph(structure)

    trial = jnp.full(NUM_EDGES, -1.0)
    state = equilibrium_state(
        trial, structure.nodes[fdm.indices_fixed], fdm, funicular(structure)
    )
    reached = jnp.max(state.xyz[:, 2])

    return structure, fdm, trial * reached / RISE


def funicular(structure):
    """
    The uniform load case the arch is form-found under.
    """
    return loads_uniform(structure, TOTAL_LOAD / (NUM_EDGES - 1))


@pytest.fixture(scope="module")
def geometry(setup):
    structure, fdm, q = setup

    return structure, equilibrium_state(
        q, structure.nodes[fdm.indices_fixed], fdm, funicular(structure)
    ).xyz


@pytest.fixture(scope="module")
def diameters():
    return jnp.full(NUM_EDGES, SEED)


def relative(actual, expected):
    """
    Worst absolute gap over the largest entry of the reference.
    """
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    scale = float(np.max(np.abs(expected)))

    return float(np.max(np.abs(actual - expected))) / (scale if scale > 0.0 else 1.0)


@pytest.fixture(scope="module")
def prepared(geometry, steel, catalogue):
    """
    Both backends' models, prepared once from the same structure.
    """
    structure, _ = geometry

    return (
        backend_opensees.prepare_model(structure, catalogue, normal=NORMAL),
        prepare_smax(structure, catalogue(SEED)),
    )


def test_the_two_solvers_agree_on_the_axial_force(
    prepared, geometry, diameters, steel, catalogue
):
    structure, xyz = geometry
    ops, smax = prepared

    mine = backend_opensees.member_forces(
        ops, xyz, diameters, catalogue, funicular(structure)
    )
    theirs = forces_smax(smax, xyz, diameters, catalogue(SEED), funicular(structure))

    assert relative(mine.axial_force, theirs.axial_force) < TOLERANCE_PRIMAL


def test_the_two_solvers_agree_on_both_end_moments(
    prepared, geometry, diameters, steel, catalogue
):
    structure, xyz = geometry
    ops, smax = prepared

    mine = backend_opensees.member_forces(
        ops, xyz, diameters, catalogue, funicular(structure)
    )
    theirs = forces_smax(smax, xyz, diameters, catalogue(SEED), funicular(structure))

    assert relative(mine.moment_major, theirs.moment_major) < TOLERANCE_PRIMAL


def test_a_plane_frame_carries_no_minor_axis_moment(
    prepared, geometry, diameters, steel, catalogue
):
    structure, xyz = geometry
    ops, _ = prepared

    mine = backend_opensees.member_forces(
        ops, xyz, diameters, catalogue, funicular(structure)
    )

    assert np.all(np.asarray(mine.moment_minor) == 0.0)


@pytest.fixture(scope="module")
def blocks(prepared, geometry, diameters, steel, catalogue):
    structure, xyz = geometry
    ops, _ = prepared

    return backend_opensees.force_jacobian(
        ops, xyz, diameters, catalogue, funicular(structure)
    )


@pytest.fixture(scope="module")
def traced(prepared, geometry, diameters, steel, catalogue):
    """
    The same derivatives, taken by tracing the other backend.
    """
    structure, xyz = geometry
    _, smax = prepared

    def run(coords, sizes):
        member = forces_smax(smax, coords, sizes, catalogue(SEED), funicular(structure))

        return {
            "axial_force": member.axial_force,
            "end_moments_major": member.moment_major,
        }

    return (
        jax.jacfwd(run, argnums=0)(xyz, diameters),
        jax.jacfwd(run, argnums=1)(xyz, diameters),
    )


def test_the_axial_force_moves_with_a_coordinate_as_autodiff_says(blocks, traced):
    by_coordinate, _ = traced

    assert (
        relative(blocks.axial_force_xyz, by_coordinate["axial_force"])
        < TOLERANCE_JACOBIAN
    )


def test_the_end_moments_move_with_a_coordinate_as_autodiff_says(blocks, traced):
    by_coordinate, _ = traced

    assert (
        relative(blocks.moment_major_xyz, by_coordinate["end_moments_major"])
        < TOLERANCE_JACOBIAN
    )


def test_the_axial_force_moves_with_a_diameter_as_autodiff_says(blocks, traced):
    _, by_diameter = traced

    assert (
        relative(blocks.axial_force_diameter, by_diameter["axial_force"])
        < TOLERANCE_JACOBIAN
    )


def test_the_end_moments_move_with_a_diameter_as_autodiff_says(blocks, traced):
    _, by_diameter = traced

    assert (
        relative(blocks.moment_major_diameter, by_diameter["end_moments_major"])
        < TOLERANCE_JACOBIAN
    )


def test_nothing_in_the_plane_moves_when_a_node_leaves_it(
    geometry, diameters, steel, catalogue
):
    """
    The separation the two-dimensional model relies on, measured not assumed.
    """
    structure, xyz = geometry
    model = prepare_smax(structure, catalogue(SEED))

    def run(coords):
        member = forces_smax(
            model, coords, diameters, catalogue(SEED), funicular(structure)
        )

        return {
            "axial_force": member.axial_force,
            "end_moments_major": member.moment_major,
        }

    jacobian = jax.jacfwd(run)(xyz)

    for block in jacobian.values():
        assert np.all(np.asarray(block)[..., NORMAL] == 0.0)


def test_the_one_block_the_plane_cannot_reach_is_the_minor_axis_moment(
    geometry, diameters, steel, catalogue
):
    """
    Nonzero in three dimensions, so the blindness is real rather than nominal.
    """
    structure, xyz = geometry
    model = prepare_smax(structure, catalogue(SEED))

    def run(coords):
        return forces_smax(
            model, coords, diameters, catalogue(SEED), funicular(structure)
        ).moment_minor

    jacobian = np.asarray(jax.jacfwd(run)(xyz))

    assert np.max(np.abs(jacobian[..., NORMAL])) > 0.0
    assert np.all(np.delete(jacobian, NORMAL, axis=-1) == 0.0)


def test_a_three_dimensional_frame_is_refused(geometry, diameters, steel, catalogue):
    structure, xyz = geometry

    with pytest.raises(ValueError, match="planar"):
        backend_opensees.prepare_model(structure, catalogue, normal=None)


def test_a_frame_that_is_not_flat_is_refused(geometry, diameters, steel, catalogue):
    structure, xyz = geometry
    warped = jnp.asarray(xyz).at[1, NORMAL].set(500.0)
    model = backend_opensees.prepare_model(structure, catalogue, normal=NORMAL)

    with pytest.raises(ValueError, match="not planar"):
        backend_opensees.member_forces(
            model, warped, diameters, catalogue, funicular(structure)
        )


def test_a_load_out_of_the_plane_is_refused(geometry, diameters, steel, catalogue):
    structure, xyz = geometry
    pushed = jnp.asarray(funicular(structure)).at[1, NORMAL].set(1_000.0)
    model = backend_opensees.prepare_model(structure, catalogue, normal=NORMAL)

    with pytest.raises(ValueError, match="normal axis"):
        backend_opensees.member_forces(model, xyz, diameters, catalogue, pushed)


@pytest.fixture(scope="module")
def chain():
    return local_chain()


@pytest.fixture(scope="module")
def objective(setup, diameters, steel, catalogue, chain):
    structure, _, _ = setup

    pipeline = StructuralDesignPipeline(
        TesseractFormFinder(structure, chain.formfinding),
        TesseractAnalyzer(structure, chain.analysis, catalogue, NORMAL),
        TesseractSizer(structure, chain.ec3, catalogue),
    )

    applied = funicular(structure)
    loads = load_cases_of([applied])

    def total(q):
        return compute_mass(pipeline(DesignParameters(q, diameters), loads))

    return total


@pytest.fixture(scope="module")
def masses(setup, objective):
    _, _, q = setup
    out = {}

    for name in ("smax", "opensees"):
        with analysis_backend(name):
            out[name] = (float(objective(q)), np.asarray(jax.grad(objective)(q)))

    return out


def test_the_mass_is_the_same_whichever_solver_produced_it(masses):
    mine, _ = masses["opensees"]
    theirs, _ = masses["smax"]

    assert abs(mine - theirs) / theirs < TOLERANCE_MASS


def test_the_gradient_is_the_same_whichever_solver_produced_it(masses):
    _, mine = masses["opensees"]
    _, theirs = masses["smax"]

    assert relative(mine, theirs) < TOLERANCE_GRADIENT


def test_the_gradient_is_not_trivially_zero(masses):
    _, mine = masses["opensees"]

    assert np.max(np.abs(mine)) > 0.0


def test_the_backend_is_restored_after_the_block():
    before = os.environ.get("NORMAX_ANALYSIS_BACKEND")

    with analysis_backend("opensees"):
        assert os.environ["NORMAX_ANALYSIS_BACKEND"] == "opensees"

    assert os.environ.get("NORMAX_ANALYSIS_BACKEND") == before


def test_the_backend_is_restored_when_the_block_raises():
    before = os.environ.get("NORMAX_ANALYSIS_BACKEND")

    with pytest.raises(RuntimeError), analysis_backend("opensees"):
        raise RuntimeError("boom")

    assert os.environ.get("NORMAX_ANALYSIS_BACKEND") == before
