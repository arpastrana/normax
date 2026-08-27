# SPDX-License-Identifier: Apache-2.0
"""
A size may not depend on the local frame the analysis happened to pick.

Two solvers orient a member's cross-section differently, and a tube has no weak
axis for that to be a property of. So the check reads the bending without naming
an axis, and these hold it to that.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from normax.analysis import MemberForces
from normax.analysis import pynite
from normax.analysis.smax import SmaxAnalyzer
from normax.loads import select_load_case
from normax.materials import Steel355
from normax.sections import build_section_family
from normax.sizing.ec3 import coerce_member_actions
from normax.structures import Structure

SECTION_CLASS = 3
SEED_DIAMETER = 100.0

# Tight enough that only the invariant reading passes, loose enough for a solve.
TOLERANCE_INVARIANT = 1e-9


@pytest.fixture(scope="module")
def family():
    return build_section_family(Steel355(), SECTION_CLASS)


@pytest.fixture(scope="module")
def canopy():
    """
    A frame no plane contains, so both bending components are live.
    """
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [4000.0, 0.0, 0.0],
            [4000.0, 3000.0, 0.0],
            [0.0, 3000.0, 0.0],
            [2000.0, 1500.0, 2500.0],
        ]
    )
    edges = np.array([[0, 4], [1, 4], [2, 4], [3, 4], [0, 1], [1, 2]])

    return Structure(nodes=nodes, edges=edges, supports=np.array([0, 1, 2, 3]))


def rotated_pair(forces, angle):
    """
    The same bending, reported about local axes turned by one angle.
    """
    cosine = np.cos(angle)
    sine = np.sin(angle)
    major = np.asarray(forces.moment_major)
    minor = np.asarray(forces.moment_minor)

    return MemberForces(
        axial_force=forces.axial_force,
        moment_major=jnp.asarray(cosine * major - sine * minor),
        moment_minor=jnp.asarray(sine * major + cosine * minor),
    )


def assert_same_actions(mine, theirs):
    """
    Every design action agrees, scaled by the largest entry of the reference.
    """
    for field in theirs._fields:
        expected = np.asarray(getattr(theirs, field))
        scale = max(float(np.max(np.abs(expected))), 1.0)
        gap = float(np.max(np.abs(expected - np.asarray(getattr(mine, field)))))
        assert gap / scale < TOLERANCE_INVARIANT, field


@pytest.mark.parametrize("angle", [0.3, 1.0, 2.4, -0.7])
def test_turning_the_local_frame_leaves_the_design_actions_alone(angle):
    # The whole claim, without a solver in the way: one bending, two frames.
    members = 5
    generator = np.random.default_rng(20260825)
    forces = MemberForces(
        axial_force=jnp.asarray(generator.normal(size=members) * 1.0e4),
        moment_major=jnp.asarray(generator.normal(size=(members, 2)) * 1.0e6),
        moment_minor=jnp.asarray(generator.normal(size=(members, 2)) * 1.0e6),
    )

    assert_same_actions(
        coerce_member_actions(rotated_pair(forces, angle)),
        coerce_member_actions(forces),
    )


def test_collinear_ends_keep_the_curvature_they_had():
    # A plane frame reports one component, and there the signed reading is not
    # in doubt: a reversal must stay a reversal, or the factor doubles.
    ends = jnp.asarray([[1.0e6, 1.0e6], [1.0e6, -1.0e6], [1.0e6, 0.0]])
    forces = MemberForces(
        axial_force=jnp.zeros(3),
        moment_major=ends,
        moment_minor=jnp.zeros((3, 2)),
    )

    factor = np.asarray(coerce_member_actions(forces).moment_factor_major)

    # Table B.3, first row: one for a uniform moment, floored under reversal.
    assert factor[0] == pytest.approx(1.0)
    assert factor[1] == pytest.approx(0.4)
    assert factor[2] == pytest.approx(0.6)


def test_two_solvers_demand_the_same_design_actions(canopy, family):
    # The regression this file exists for. One traced, one not; one vertical on
    # the third axis, one on the second; and the actions must not know.
    diameters = jnp.full((canopy.num_edges,), SEED_DIAMETER)
    loads = np.zeros_like(np.asarray(canopy.nodes))
    loads[4, 2] = -6.0e4
    loads[4, 0] = 2.0e4

    stacked = SmaxAnalyzer(canopy, family(SEED_DIAMETER))(
        canopy.nodes, diameters, jnp.asarray(loads)[None, ...]
    )
    traced = select_load_case(stacked, 0)
    problem = pynite.FrameProblem(structure=canopy, catalogue=family, loads=loads)
    foreign = pynite.compute_member_forces(
        problem, np.asarray(canopy.nodes), np.asarray(diameters), loads
    )

    assert_same_actions(coerce_member_actions(foreign), coerce_member_actions(traced))
