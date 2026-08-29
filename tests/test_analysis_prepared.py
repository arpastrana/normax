# SPDX-License-Identifier: Apache-2.0
"""
Assembling and factorizing a frame once, and what that buys and costs.

The expensive half of a frame solve does not depend on the loading, so the
PyNite path assembles and factorizes into a `PreparedFrame` and reuses it for
every load case and for the adjoint after them. This is the in-process host math
the crossed backend runs, called directly: no boundary and no container in the
way of the claim.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from normax.analysis import MemberForces
from normax.analysis import pynite
from normax.form_finding import FdmFormFinder
from normax.loads import create_load_uniform
from normax.materials import Steel355
from normax.sections import build_section_catalog
from normax.structures import build_arch_2d

# A 10 m arch of ten members under a 20 kN load at every free node, in the XZ
# plane, at about the size the code check asks for. Units are mm and N.
SPAN = 10_000.0
LOAD = 20_000.0
NUM_EDGES = 10
FORCE_DENSITY = -75.0
DIAMETER = 100.0

# Mapping the reading over load cases reassociates the arithmetic. Measured at
# 5.0e-16 on the axial force and 1.2e-14 on the end moments.
TOLERANCE_STACKED = 1e-13

# Scales that bring both reported quantities to unit order before seeding.
SCALE_FORCE = 1.0e5
SCALE_MOMENT = 1.0e8


def relative(actual, expected):
    """
    Worst absolute gap over the largest entry of the reference.
    """
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    scale = float(np.max(np.abs(expected)))

    return float(np.max(np.abs(actual - expected))) / (scale if scale > 0.0 else 1.0)


@pytest.fixture(scope="module")
def steel():
    return Steel355()


@pytest.fixture(scope="module")
def catalog(steel):
    return build_section_catalog(steel, 3)


@pytest.fixture(scope="module")
def structure():
    return build_arch_2d(num_edges=NUM_EDGES, span=SPAN, rise=SPAN / 3.0)


@pytest.fixture(scope="module")
def applied(structure):
    return np.asarray(create_load_uniform(structure, LOAD * (NUM_EDGES - 1)))


@pytest.fixture(scope="module")
def sideways(applied):
    """
    A second load case no part of the first is a multiple of.
    """
    pushed = np.zeros_like(applied)
    pushed[3] = (5.0e4, 0.0, 0.0)

    return pushed


@pytest.fixture(scope="module")
def xyz(structure, applied):
    found = FdmFormFinder(structure)(jnp.full(NUM_EDGES, FORCE_DENSITY), applied).xyz

    return np.asarray(found)


@pytest.fixture(scope="module")
def diameters():
    return np.full(NUM_EDGES, DIAMETER)


@pytest.fixture(scope="module")
def problem(structure, catalog, applied):
    return pynite.FrameProblem(structure=structure, catalog=catalog, loads=applied)


@pytest.fixture(scope="module")
def prepared(problem, xyz, diameters):
    return pynite.prepare_frame(problem, xyz, diameters)


def test_a_prepared_frame_and_a_fresh_one_report_the_same_forces(
    problem, xyz, diameters, applied, prepared
):
    # Reusing the decomposition may not change the answer by one bit, or the
    # cheap route and the honest route are two different analyses.
    fresh = pynite.compute_member_forces(problem, xyz, diameters, applied)
    reused = pynite.compute_member_forces(problem, xyz, diameters, applied, prepared)

    assert np.all(np.asarray(fresh.axial_force) == np.asarray(reused.axial_force))
    assert np.all(np.asarray(fresh.moment_major) == np.asarray(reused.moment_major))


def test_a_prepared_frame_carries_no_load_case_of_its_own(
    problem, xyz, diameters, applied, prepared
):
    once = pynite.compute_member_forces(problem, xyz, diameters, applied, prepared)
    twice = pynite.compute_member_forces(problem, xyz, diameters, applied, prepared)
    halved = pynite.compute_member_forces(
        problem, xyz, diameters, 0.5 * applied, prepared
    )

    assert np.all(np.asarray(once.axial_force) == np.asarray(twice.axial_force))
    assert np.allclose(
        np.asarray(halved.axial_force), 0.5 * np.asarray(once.axial_force)
    )


def test_a_prepared_frame_answers_at_the_geometry_it_was_prepared_at(
    problem, xyz, diameters, applied
):
    # The trap this path has and the traced one did not: a prepared frame is
    # the geometry, and the coordinates beside it are then never read.
    lifted = xyz.copy()
    lifted[:, 2] *= 1.4
    stale = pynite.prepare_frame(problem, lifted, diameters)

    at_lifted = pynite.compute_member_forces(problem, lifted, diameters, applied)
    via_stale = pynite.compute_member_forces(problem, xyz, diameters, applied, stale)
    honest = pynite.compute_member_forces(problem, xyz, diameters, applied)

    assert np.all(
        np.asarray(via_stale.axial_force) == np.asarray(at_lifted.axial_force)
    )
    assert relative(via_stale.axial_force, honest.axial_force) > 1e-3


def test_several_load_cases_cost_one_factorization(
    problem, xyz, diameters, applied, sideways, prepared
):
    stacked = np.stack([applied, sideways])
    together = pynite.solve_displacements(prepared, stacked)
    first = pynite.solve_displacements(prepared, applied[None, ...])
    second = pynite.solve_displacements(prepared, sideways[None, ...])

    assert np.all(together[0] == first[0])
    assert np.all(together[1] == second[0])

    single = pynite.compute_member_forces(problem, xyz, diameters, applied, prepared)
    many = pynite.compute_member_forces(problem, xyz, diameters, stacked, prepared)
    spread = np.asarray(many.axial_force)

    assert np.asarray(single.axial_force).shape == spread.shape[1:]
    assert relative(spread[0], single.axial_force) < TOLERANCE_STACKED
    assert relative(spread[0], spread[1]) > 0.1


def test_the_modulus_reaches_the_assembly_it_was_prepared_with(
    problem, steel, catalog, xyz, diameters, applied, prepared
):
    # Member forces of a uniform-E linear frame are E-independent, so the
    # displacement is what distinguishes an injected modulus from a baked one.
    graded = catalog._replace(material=steel._replace(e_mod=2.0 * steel.e_mod))
    stiffer = problem._replace(catalog=graded)
    prepared_stiffer = pynite.prepare_frame(stiffer, xyz, diameters)

    soft = pynite.solve_displacements(prepared, applied[None, ...])[0]
    hard = pynite.solve_displacements(prepared_stiffer, applied[None, ...])[0]

    assert relative(2.0 * hard, soft) < TOLERANCE_STACKED

    limber = pynite.compute_member_forces(problem, xyz, diameters, applied, prepared)
    rigid = pynite.compute_member_forces(
        stiffer, xyz, diameters, applied, prepared_stiffer
    )

    assert relative(rigid.axial_force, limber.axial_force) < TOLERANCE_STACKED
    assert relative(rigid.moment_major, limber.moment_major) < TOLERANCE_STACKED


def test_the_geometry_and_the_diameters_are_live_leaves(
    problem, xyz, diameters, applied, prepared
):
    # The adjoint has to read every leaf a design variable reaches; a leaf left
    # at a placeholder would pull back a silent zero instead.
    carried = pynite.compute_member_forces(problem, xyz, diameters, applied, prepared)
    seed = MemberForces(
        axial_force=2.0 * np.asarray(carried.axial_force) / SCALE_FORCE**2,
        moment_major=2.0 * np.asarray(carried.moment_major) / SCALE_MOMENT**2,
        moment_minor=np.zeros_like(np.asarray(carried.moment_minor)),
    )

    fresh = pynite.pull_back_cotangents(problem, xyz, diameters, seed)
    reused = pynite.pull_back_cotangents(problem, xyz, diameters, seed, prepared)

    for pulled in (fresh.xyz, fresh.diameter):
        assert np.all(np.isfinite(pulled))
        assert float(np.min(np.abs(pulled[np.nonzero(pulled)]))) > 0.0
        assert float(np.max(np.abs(pulled))) > 0.0

    assert np.all(fresh.xyz == reused.xyz)
    assert np.all(fresh.diameter == reused.diameter)
