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
A foreign solver, given an adjoint it does not have.

PyNite assembles, factorizes and solves; it carries no derivative of its own.
The rule that differentiates it rests on two claims that have to be measured:
the element here is the element it assembled, and the rule is exact.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from Pynite import FEModel3D

from normax.analysis import MemberForces
from normax.analysis import pynite
from normax.analysis.element import SectionRigidity
from normax.analysis.element import member_frame
from normax.analysis.element import stiffness_global
from normax.analysis.element import stiffness_local
from normax.analysis.smax import SmaxAnalyzer
from normax.materials import Steel355
from normax.sections import build_section_family
from normax.structures import Structure
from normax.tesseract import TesseractAnalyzer
from normax.tesseract import analysis_tesseract

SECTION_CLASS = 3
SEED_DIAMETER = 100.0

# The replica and the solver do the same arithmetic in a different order.
TOLERANCE_ELEMENT = 1e-13

# Two exact gradients of one structure, so likewise round-off and no more.
TOLERANCE_GRADIENT = 1e-11

# A central difference is the loose party in that comparison, not the rule.
TOLERANCE_DIFFERENCE = 1e-7

# Where a differenced coordinate is least contaminated, measured by sweeping it.
STEP_COORDINATE = 1.0e-2

# Scales that bring both reported quantities to unit order before summing.
SCALE_FORCE = 1.0e5
SCALE_MOMENT = 1.0e8


@pytest.fixture(scope="module")
def family():
    return build_section_family(Steel355(), SECTION_CLASS)


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
def problem(canopy, canopy_loads, family):
    return pynite.FrameProblem(structure=canopy, catalogue=family, loads=canopy_loads)


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

        local = stiffness_local(length, rigidity)
        assert relative(local, theirs.ke()) < TOLERANCE_ELEMENT

        spanned = stiffness_global(jnp.asarray(start), jnp.asarray(end), rigidity)
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
    spanned = stiffness_global(start, end, rigidity)

    axis = np.asarray(end - start) / np.linalg.norm(np.asarray(end - start))
    frame = np.asarray(member_frame(start, end))
    angle = 0.7
    turned = np.stack(
        [
            axis,
            np.cos(angle) * frame[1] + np.sin(angle) * frame[2],
            -np.sin(angle) * frame[1] + np.cos(angle) * frame[2],
        ]
    )
    transform = np.kron(np.eye(4), turned)
    local = np.asarray(stiffness_local(jnp.linalg.norm(end - start), rigidity))
    rolled = transform.T @ local @ transform

    assert relative(spanned, rolled) < TOLERANCE_ELEMENT


def test_the_adjoint_agrees_with_a_traced_solver(
    canopy, canopy_loads, canopy_diameters, family, problem
):
    # The gate. One structure, one scalar, two exact gradients: a hand-written
    # adjoint of a solver that has none, and autodiff of one that is traced.
    analyzer = SmaxAnalyzer(canopy, family(SEED_DIAMETER))
    stacked = jnp.asarray(canopy_loads)[None, ...]

    def traced_loss(xyz, diameters):
        return scaled_loss(analyzer(xyz, diameters, stacked))

    traced = jax.grad(traced_loss, argnums=(0, 1))(
        canopy.nodes, jnp.asarray(canopy_diameters)
    )

    nodes = np.asarray(canopy.nodes)
    forces = pynite.member_forces(problem, nodes, canopy_diameters, canopy_loads)
    pulled = pynite.force_cotangents(
        problem, nodes, canopy_diameters, loss_cotangent(forces)
    )

    assert relative(pulled.xyz, traced[0]) < TOLERANCE_GRADIENT
    assert relative(pulled.diameter, traced[1]) < TOLERANCE_GRADIENT


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
    exact = pynite.force_cotangents(
        problem, np.asarray(canopy.nodes), canopy_diameters, seed
    )

    def seeded(xyz):
        forces = pynite.member_forces(problem, xyz, canopy_diameters, canopy_loads)
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
    canopy, canopy_loads, canopy_diameters, family
):
    # What the submission claims: a solver with no derivative of its own,
    # reached across a schema, handing back a reverse-mode gradient.
    stacked = jnp.asarray(canopy_loads)[None, ...]
    diameters = jnp.asarray(canopy_diameters)

    def loss(analyzer, xyz, sizes):
        return scaled_loss(analyzer(xyz, sizes, stacked))

    crossed = TesseractAnalyzer(canopy, analysis_tesseract("pynite"), family, None)
    served = float(loss(crossed, canopy.nodes, diameters))
    foreign = jax.grad(lambda x, d: loss(crossed, x, d), argnums=(0, 1))(
        canopy.nodes, diameters
    )

    traced = SmaxAnalyzer(canopy, family(SEED_DIAMETER))
    expected = float(loss(traced, canopy.nodes, diameters))
    reference = jax.grad(lambda x, d: loss(traced, x, d), argnums=(0, 1))(
        canopy.nodes, diameters
    )

    assert abs(served - expected) / abs(expected) < TOLERANCE_GRADIENT
    assert relative(foreign[0], reference[0]) < TOLERANCE_GRADIENT
    assert relative(foreign[1], reference[1]) < TOLERANCE_GRADIENT


def test_a_section_of_no_area_is_refused(canopy, canopy_loads, problem):
    with pytest.raises(ValueError, match="not positive"):
        pynite.member_forces(
            problem, np.asarray(canopy.nodes), np.zeros(canopy.num_edges), canopy_loads
        )


def test_a_vertical_member_is_refused(family):
    # The reporting convention completes its transverse pair against the
    # vertical, so a vertical member has no pair to report a bending in.
    nodes = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3000.0], [2000.0, 0.0, 3000.0]])
    edges = np.array([[0, 1], [1, 2]])
    mast = Structure(nodes=nodes, edges=edges, supports=np.array([0]))
    loads = np.zeros_like(nodes)
    loads[2, 2] = -1.0e4
    problem = pynite.FrameProblem(structure=mast, catalogue=family, loads=loads)

    with pytest.raises(ValueError, match="vertical"):
        pynite.member_forces(problem, nodes, np.full(2, SEED_DIAMETER), loads)


def test_several_load_cases_cross_together(
    canopy, canopy_loads, canopy_diameters, family
):
    # The backend remembers one factorized frame between endpoint calls, and the
    # adjoints run after all the forwards, so a cache keyed on anything the
    # loads touch would answer the second case with the first case's solve.
    sideways = np.zeros_like(canopy_loads)
    sideways[4] = (-4.0e4, 1.5e4, -1.0e4)
    stacked = jnp.asarray(np.stack([canopy_loads, sideways, 0.4 * canopy_loads]))
    diameters = jnp.asarray(canopy_diameters)

    def loss(analyzer, xyz, sizes):
        forces = analyzer(xyz, sizes, stacked)
        weighted = jnp.arange(1, forces.axial_force.shape[0] + 1)[:, None]
        axial = jnp.sum(weighted * (forces.axial_force / SCALE_FORCE) ** 2)
        major = jnp.sum((forces.moment_major / SCALE_MOMENT) ** 2)
        minor = jnp.sum((forces.moment_minor / SCALE_MOMENT) ** 2)

        return axial + major + minor

    crossed = TesseractAnalyzer(canopy, analysis_tesseract("pynite"), family, None)
    served = crossed(canopy.nodes, diameters, stacked)
    foreign = jax.grad(lambda x, d: loss(crossed, x, d), argnums=(0, 1))(
        canopy.nodes, diameters
    )

    traced = SmaxAnalyzer(canopy, family(SEED_DIAMETER))
    expected = traced(canopy.nodes, diameters, stacked)
    reference = jax.grad(lambda x, d: loss(traced, x, d), argnums=(0, 1))(
        canopy.nodes, diameters
    )

    # The three cases must stay distinct, or a cache has served one for another.
    spread = np.asarray(served.axial_force)
    assert relative(spread[0], spread[1]) > 0.1
    assert relative(spread[0], spread[2]) > 0.1

    assert relative(served.axial_force, expected.axial_force) < TOLERANCE_GRADIENT
    assert relative(served.moment_major, expected.moment_major) < TOLERANCE_GRADIENT
    assert relative(foreign[0], reference[0]) < TOLERANCE_GRADIENT
    assert relative(foreign[1], reference[1]) < TOLERANCE_GRADIENT


def test_a_second_frame_is_not_answered_by_the_first(
    canopy, canopy_loads, family, problem
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
    second = pynite.FrameProblem(
        structure=shifted, catalogue=family, loads=canopy_loads
    )

    client = analysis_tesseract("pynite")
    here = TesseractAnalyzer(canopy, client, family, None)
    there = TesseractAnalyzer(shifted, client, family, None)
    sizes = jnp.asarray(diameters)
    stacked = jnp.asarray(canopy_loads)[None, ...]
    served_here = here(canopy.nodes, sizes, stacked)
    served_there = there(shifted.nodes, sizes, stacked)
    served_again = here(canopy.nodes, sizes, stacked)

    direct_here = pynite.member_forces(
        problem, np.asarray(canopy.nodes), diameters, canopy_loads
    )
    direct_there = pynite.member_forces(second, moved, diameters, canopy_loads)

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
