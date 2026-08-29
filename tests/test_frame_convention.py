# SPDX-License-Identifier: Apache-2.0
"""
A size may not depend on the local frame the analysis happened to pick.

A tube has no weak axis, so how one bending splits over two transverse axes is a
reporting convention and not a force. This repository keeps that convention in
one place — `compute_direction_cosines`, which completes its pair against the
vertical — and the design actions the check reads follow the structure rather
than the solver: turn the whole frame about the vertical and nothing changes,
and the two routes to the same solver demand the same sizes.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from normax.analysis import pynite
from normax.loads import select_load_case
from normax.materials import Steel355
from normax.sections import build_section_catalog
from normax.sizing.blueprint import coerce_member_actions
from normax.sizing.blueprint import reduce_moments
from normax.structures import Structure
from normax.tesseract import TesseractAnalyzer

SECTION_CLASS = 3
SEED_DIAMETER = 100.0

# Tight enough that only a frame-following reading passes, loose enough for a
# solve. Measured at 2.1e-15 turned and bitwise across the boundary.
TOLERANCE_INVARIANT = 1e-9


@pytest.fixture(scope="module")
def catalog():
    return build_section_catalog(Steel355(), SECTION_CLASS)


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


@pytest.fixture(scope="module")
def canopy_loads(canopy):
    """
    One load case with a component the shell has no symmetry about.
    """
    pushed = np.zeros_like(np.asarray(canopy.nodes))
    pushed[4, 2] = -6.0e4
    pushed[4, 0] = 2.0e4

    return pushed


@pytest.fixture(scope="module")
def canopy_diameters(canopy):
    return jnp.full((canopy.num_edges,), SEED_DIAMETER)


def read_design_actions(forces):
    """
    What the shipped check reads off one load case of an analysis.
    """
    return coerce_member_actions(
        forces.axial_force, forces.moment_major, forces.moment_minor
    )


def turn_about_vertical(vectors, angle):
    """
    The same vectors, turned about the vertical the convention is completed on.
    """
    cosine = np.cos(angle)
    sine = np.sin(angle)
    block = [
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ]
    turning = np.asarray(block)

    return np.asarray(vectors) @ turning.T


def assert_same_actions(mine, theirs):
    """
    Every design action agrees, scaled by the largest entry of the reference.
    """
    for field in theirs._fields:
        expected = np.asarray(getattr(theirs, field))
        scale = max(float(np.max(np.abs(expected))), 1.0)
        gap = float(np.max(np.abs(expected - np.asarray(getattr(mine, field)))))
        assert gap / scale < TOLERANCE_INVARIANT, field

    demanded = reduce_moments(mine).moment
    reference = reduce_moments(theirs).moment
    scale = max(float(np.max(np.abs(reference))), 1.0)

    assert float(np.max(np.abs(demanded - reference))) / scale < TOLERANCE_INVARIANT


def analyze_crossed(structure, catalog, diameters, loads):
    """
    One load case of the crossed analysis, without its load case axis.
    """
    analyzer = TesseractAnalyzer(structure, catalog, "pynite")
    stacked = analyzer(structure.nodes, diameters, jnp.asarray(loads)[None, ...])

    return select_load_case(stacked, 0)


@pytest.fixture(scope="module")
def canopy_actions(canopy, catalog, canopy_diameters, canopy_loads):
    forces = analyze_crossed(canopy, catalog, canopy_diameters, canopy_loads)

    return read_design_actions(forces)


@pytest.mark.parametrize("angle", [0.3, 1.0, 2.4, -0.7])
def test_turning_the_structure_leaves_the_design_actions_alone(
    angle, canopy, catalog, canopy_diameters, canopy_loads, canopy_actions
):
    # The whole claim: the reporting frame is completed against the vertical, so
    # it turns with the structure and the actions never learn that it turned.
    nodes = turn_about_vertical(canopy.nodes, angle)
    turned = Structure(
        nodes=nodes,
        edges=np.asarray(canopy.edges),
        supports=np.asarray(canopy.supports),
    )
    loads = turn_about_vertical(canopy_loads, angle)
    forces = analyze_crossed(turned, catalog, canopy_diameters, loads)

    assert_same_actions(read_design_actions(forces), canopy_actions)


def test_the_worse_end_is_the_moment_the_check_reads():
    # A cross-section check at the worse end, so a reversal is not reduced: the
    # equivalent uniform moment of Table B.3 belongs to a buckling check.
    ends = np.array([[1.0e6, 1.0e6], [1.0e6, -1.0e6], [1.0e6, 0.0]])
    actions = coerce_member_actions(np.zeros(3), ends, np.zeros((3, 2)))
    demand = reduce_moments(actions)

    assert np.allclose(demand.moment, 1.0e6)
    assert np.all(demand.major.winner == 0)
    assert np.all(demand.major.sign == 1.0)

    # And the two axes superpose linearly per eq. (6.2), never as a resultant.
    split = coerce_member_actions(
        np.zeros(1), np.array([[1.0e6, 0.0]]), np.array([[0.0, 6.0e5]])
    )

    assert np.allclose(reduce_moments(split).moment, 1.6e6)


def test_two_routes_demand_the_same_design_actions(
    canopy, catalog, canopy_diameters, canopy_loads, canopy_actions
):
    # The regression this file exists for. One crossed and stacked over load
    # cases, one called in process on a single case; the actions must not know.
    problem = pynite.FrameProblem(structure=canopy, catalog=catalog, loads=canopy_loads)
    forces = pynite.compute_member_forces(
        problem,
        np.asarray(canopy.nodes),
        np.asarray(canopy_diameters),
        canopy_loads,
    )

    assert_same_actions(read_design_actions(forces), canopy_actions)
