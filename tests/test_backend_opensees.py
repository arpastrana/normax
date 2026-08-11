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
from normax.analysis.smax import forces as forces_smax
from normax.analysis.smax import prepare as prepare_smax
from normax.composition import backend
from normax.composition import local
from normax.composition import mass as mass_composed
from normax.ec3.material import SteelGrade
from normax.ec3.sizing import TubeCatalogue
from normax.formfinding import equilibrium
from normax.formfinding import graph
from normax.structures import arch

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
    return SteelGrade()


@pytest.fixture(scope="module")
def catalogue(steel):
    return TubeCatalogue.at_class_limit(steel.f_y, 3)


@pytest.fixture(scope="module")
def setup():
    """
    The arch, its connectivity, and the `q` that reaches the target rise.
    """
    load = TOTAL_LOAD / (NUM_EDGES - 1)
    structure = arch(num_edges=NUM_EDGES, span=SPAN, rise=RISE, load=load)
    fdm = graph(structure)

    trial = jnp.full(NUM_EDGES, -1.0)
    reached = jnp.max(equilibrium(trial, structure, fdm).xyz[:, 2])

    return structure, fdm, trial * reached / RISE


@pytest.fixture(scope="module")
def geometry(setup):
    structure, fdm, q = setup

    return structure, equilibrium(q, structure, fdm).xyz


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
        backend_opensees.prepare(structure, steel, catalogue, normal=NORMAL),
        prepare_smax(structure, steel, catalogue, normal=NORMAL),
    )


def test_the_two_solvers_agree_on_the_axial_force(
    prepared, geometry, diameters, steel, catalogue
):
    _, xyz = geometry
    ops, smax = prepared

    mine = backend_opensees.forces(ops, xyz, diameters, steel, catalogue)
    theirs = forces_smax(smax, xyz, diameters, steel, catalogue)

    assert relative(mine.n_ed, theirs.n_ed) < TOLERANCE_PRIMAL


def test_the_two_solvers_agree_on_both_end_moments(
    prepared, geometry, diameters, steel, catalogue
):
    _, xyz = geometry
    ops, smax = prepared

    mine = backend_opensees.forces(ops, xyz, diameters, steel, catalogue)
    theirs = forces_smax(smax, xyz, diameters, steel, catalogue)

    assert relative(mine.m_y_ed, theirs.m_y_ed) < TOLERANCE_PRIMAL


def test_a_plane_frame_carries_no_minor_axis_moment(
    prepared, geometry, diameters, steel, catalogue
):
    _, xyz = geometry
    ops, _ = prepared

    mine = backend_opensees.forces(ops, xyz, diameters, steel, catalogue)

    assert np.all(np.asarray(mine.m_z_ed) == 0.0)


@pytest.fixture(scope="module")
def blocks(prepared, geometry, diameters, steel, catalogue):
    _, xyz = geometry
    ops, _ = prepared

    return backend_opensees.jacobian(ops, xyz, diameters, steel, catalogue)


@pytest.fixture(scope="module")
def traced(prepared, geometry, diameters, steel, catalogue):
    """
    The same derivatives, taken by tracing the other backend.
    """
    _, xyz = geometry
    _, smax = prepared

    def run(coords, sizes):
        member = forces_smax(smax, coords, sizes, steel, catalogue)

        return {"n_ed": member.n_ed, "m_y_ed": member.m_y_ed}

    return (
        jax.jacfwd(run, argnums=0)(xyz, diameters),
        jax.jacfwd(run, argnums=1)(xyz, diameters),
    )


def test_the_axial_force_moves_with_a_coordinate_as_autodiff_says(blocks, traced):
    by_coordinate, _ = traced

    assert relative(blocks.n_ed_xyz, by_coordinate["n_ed"]) < TOLERANCE_JACOBIAN


def test_the_end_moments_move_with_a_coordinate_as_autodiff_says(blocks, traced):
    by_coordinate, _ = traced

    assert relative(blocks.m_y_ed_xyz, by_coordinate["m_y_ed"]) < TOLERANCE_JACOBIAN


def test_the_axial_force_moves_with_a_diameter_as_autodiff_says(blocks, traced):
    _, by_diameter = traced

    assert relative(blocks.n_ed_diameter, by_diameter["n_ed"]) < TOLERANCE_JACOBIAN


def test_the_end_moments_move_with_a_diameter_as_autodiff_says(blocks, traced):
    _, by_diameter = traced

    assert relative(blocks.m_y_ed_diameter, by_diameter["m_y_ed"]) < TOLERANCE_JACOBIAN


def test_nothing_in_the_plane_moves_when_a_node_leaves_it(
    geometry, diameters, steel, catalogue
):
    """
    The separation the two-dimensional model relies on, measured not assumed.
    """
    structure, xyz = geometry
    model = prepare_smax(structure, steel, catalogue, normal=NORMAL)

    def run(coords):
        member = forces_smax(model, coords, diameters, steel, catalogue)

        return {"n_ed": member.n_ed, "m_y_ed": member.m_y_ed}

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
    model = prepare_smax(structure, steel, catalogue, normal=NORMAL)

    def run(coords):
        return forces_smax(model, coords, diameters, steel, catalogue).m_z_ed

    jacobian = np.asarray(jax.jacfwd(run)(xyz))

    assert np.max(np.abs(jacobian[..., NORMAL])) > 0.0
    assert np.all(np.delete(jacobian, NORMAL, axis=-1) == 0.0)


def test_a_three_dimensional_frame_is_refused(geometry, diameters, steel, catalogue):
    structure, xyz = geometry

    with pytest.raises(ValueError, match="planar"):
        backend_opensees.prepare(structure, steel, catalogue, normal=None)


def test_a_frame_that_is_not_flat_is_refused(geometry, diameters, steel, catalogue):
    structure, xyz = geometry
    warped = jnp.asarray(xyz).at[1, NORMAL].set(500.0)
    model = backend_opensees.prepare(structure, steel, catalogue, normal=NORMAL)

    with pytest.raises(ValueError, match="not planar"):
        backend_opensees.forces(model, warped, diameters, steel, catalogue)


def test_a_load_out_of_the_plane_is_refused(geometry, diameters, steel, catalogue):
    structure, xyz = geometry
    pushed = structure._replace(
        loads=jnp.asarray(structure.loads).at[1, NORMAL].set(1_000.0)
    )
    model = backend_opensees.prepare(pushed, steel, catalogue, normal=NORMAL)

    with pytest.raises(ValueError, match="normal axis"):
        backend_opensees.forces(model, xyz, diameters, steel, catalogue)


@pytest.fixture(scope="module")
def chain():
    return local()


@pytest.fixture(scope="module")
def objective(setup, diameters, steel, catalogue, chain):
    structure, _, _ = setup

    def total(q):
        return mass_composed(
            q,
            diameters,
            structure,
            chain,
            steel,
            catalogue,
            normal=NORMAL,
            plastic=False,
        )

    return total


@pytest.fixture(scope="module")
def masses(setup, objective):
    _, _, q = setup
    out = {}

    for name in ("smax", "opensees"):
        with backend(name):
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

    with backend("opensees"):
        assert os.environ["NORMAX_ANALYSIS_BACKEND"] == "opensees"

    assert os.environ.get("NORMAX_ANALYSIS_BACKEND") == before


def test_the_backend_is_restored_when_the_block_raises():
    before = os.environ.get("NORMAX_ANALYSIS_BACKEND")

    with pytest.raises(RuntimeError), backend("opensees"):
        raise RuntimeError("boom")

    assert os.environ.get("NORMAX_ANALYSIS_BACKEND") == before
