from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from blueprints.codes.eurocode.en_1993_1_1_2005.chapter_6_ultimate_limit_state.formula_6_6 import (
    Form6Dot6DesignPlasticResistanceGrossCrossSection,
)
from blueprints.codes.eurocode.en_1993_1_1_2005.chapter_6_ultimate_limit_state.formula_6_10 import (
    Form6Dot10NcRdClass1And2And3,
)
from conftest import load_tesseract_api
from jax.test_util import check_grads

from normax.analysis import MemberForces
from normax.materials import Steel355
from normax.sections import TubeFamily
from normax.sections import build_section_family
from normax.sizing import MemberSizes
from normax.sizing import blueprint as blueprint_module
from normax.sizing.blueprint import DIAMETER_MINIMUM
from normax.sizing.blueprint import BlueprintSizer
from normax.sizing.blueprint import checked_members
from normax.sizing.blueprint import host_family
from normax.sizing.blueprint import sized_members
from normax.sizing.ec3 import Ec3Sizer
from normax.structures import build_arch_2d
from normax.tesseract import BlueprintClient
from normax.tesseract import blueprint_tesseract

# The proof this file makes: an external, non-differentiable, scalar code
# library fills the sizing contract and carries an exact adjoint — in process
# through a pure_callback, and across a Tesseract boundary through the same
# host functions, so the two agree bit for bit rather than to a tolerance.

NUM_EDGES = 4

# A wall proportion this file picks for itself; no class limit chose it.
RATIO = 50.0
YIELD_SAMPLE = 355.0

# Invariant 6.5 of CLAUDE.md, philosophy-independent.
TOLERANCE_UTILIZATION = 1e-9

# The hand adjoint against central differences.
TARGET = 1e-8

# The bisection against the closed-form cubic root.
TOLERANCE_ROOT = 1e-12

# Two load cases of actions with distinct end magnitudes and one clamped
# member, so both branches of the hand adjoint are exercised in every test. No
# end pair ties: at a tie the traced and the hand rule pick different, equally
# valid subgradients.
AXIAL = jnp.asarray([[-2.0e5, 1.5e5, -8.0e4, -1.0e2], [3.0e5, -2.5e5, 1.2e5, -4.0e5]])
END_MAJOR = jnp.asarray(
    [
        [[3.0e6, -1.0e6], [5.0e5, 2.0e5], [-1.0e7, 4.0e6], [0.0, 1.0e3]],
        [[-2.0e6, 4.0e6], [1.0e6, -3.0e6], [7.0e6, 2.0e6], [5.0e5, -8.0e5]],
    ]
)
END_MINOR = jnp.asarray(
    [
        [[2.0e5, -6.0e5], [1.0e5, -4.0e4], [1.0e6, -3.0e5], [2.0e2, -5.0e1]],
        [[3.0e5, 1.0e5], [-2.0e5, 5.0e5], [4.0e5, -9.0e5], [6.0e4, 7.0e4]],
    ]
)
HELD = jnp.asarray([150.0, 80.0, 200.0, 30.0])
LENGTHS = jnp.full(NUM_EDGES, 1000.0)


@pytest.fixture(scope="module")
def structure():
    return build_arch_2d(num_edges=NUM_EDGES)


@pytest.fixture(scope="module")
def family():
    return TubeFamily(RATIO, Steel355())


@pytest.fixture(scope="module")
def sizer(structure, family):
    return BlueprintSizer(structure, family)


@pytest.fixture(scope="module")
def forces():
    return MemberForces(AXIAL, END_MAJOR, END_MINOR)


@pytest.fixture(scope="module")
def remote(structure, family):
    return BlueprintClient(structure, blueprint_tesseract(), family)


def test_the_backend_names_no_ec3_library():
    backend = Path(blueprint_module.__file__).read_text()
    imported = [line for line in backend.splitlines() if line.startswith("from ")]

    assert not any("ec3x" in line for line in imported)
    assert any("blueprints" in line for line in imported)


def test_the_two_axial_formulas_agree():
    # 6.6 and 6.10 are the same expression for class 1-3; the check calls 6.10.
    yielding = Form6Dot6DesignPlasticResistanceGrossCrossSection(
        a=7367.0, f_y=YIELD_SAMPLE, gamma_m0=1.0
    )
    squashing = Form6Dot10NcRdClass1And2And3(a=7367.0, f_y=YIELD_SAMPLE, gamma_m0=1.0)

    assert float(yielding) == float(squashing)


def test_the_sizer_fills_the_contract_and_is_fully_stressed(sizer, forces):
    sizes = sizer(forces, LENGTHS)
    diameter = np.asarray(sizes.sections.diameter)
    used = np.asarray(sizes.utilization)
    free = diameter > DIAMETER_MINIMUM

    assert isinstance(sizes, MemberSizes)
    assert np.any(free) and not np.all(free)
    assert np.allclose(used[free], 1.0, rtol=0.0, atol=TOLERANCE_UTILIZATION)
    assert np.all(used <= 1.0 + TOLERANCE_UTILIZATION)


def test_the_reread_agrees_with_the_sizes(sizer, forces):
    sizes = sizer(forces, LENGTHS)
    demanded = sizes.sections.diameter
    reread = sizer.compute_utilization(demanded, forces, LENGTHS)
    free = np.asarray(demanded) > DIAMETER_MINIMUM

    assert np.allclose(
        np.asarray(reread)[free], 1.0, rtol=0.0, atol=TOLERANCE_UTILIZATION
    )


def test_the_check_matches_check_grads(sizer):
    # check_grads perturbs by a small absolute step, so the map is rescaled to
    # unit order before it judges the hand rule.
    scale_diameter = 100.0
    scale_axial = 1.0e5
    scale_moment = 1.0e6

    def scaled(size, force, major, minor):
        return checked_members(
            sizer.host,
            size * scale_diameter,
            force * scale_axial,
            major * scale_moment,
            minor * scale_moment,
        )

    size = jnp.asarray([[1.2, 0.8, 1.4, 0.3], [1.5, 0.9, 2.0, 1.1]])
    arguments = (
        size,
        AXIAL / scale_axial,
        END_MAJOR / scale_moment,
        END_MINOR / scale_moment,
    )
    check_grads(scaled, arguments, order=1, modes=("rev",))


def test_the_sizing_map_matches_check_grads(sizer):
    scale_axial = 1.0e5
    scale_moment = 1.0e6

    def scaled(force, major, minor):
        diameter, _ = sized_members(
            sizer.host, force * scale_axial, major * scale_moment, minor * scale_moment
        )

        return diameter / 100.0

    arguments = (
        AXIAL / scale_axial,
        END_MAJOR / scale_moment,
        END_MINOR / scale_moment,
    )
    check_grads(scaled, arguments, order=1, modes=("rev",))


def test_central_differences_are_the_oracle(sizer):
    # The implicit quotients against the numerical truth at a relative step,
    # on the free members, tension and compression alike.
    def total(force):
        diameter, _ = sized_members(sizer.host, force, END_MAJOR, END_MINOR)

        return jnp.sum(diameter)

    gradient = jax.grad(total)(AXIAL)
    for load_case in range(2):
        for member in range(NUM_EDGES):
            if load_case == 0 and member == 3:
                continue
            step = 1e-6 * float(jnp.abs(AXIAL[load_case, member]))
            bumped = total(AXIAL.at[load_case, member].add(step))
            lowered = total(AXIAL.at[load_case, member].add(-step))
            numeric = (bumped - lowered) / (2.0 * step)

            assert (
                abs(float(gradient[load_case, member] - numeric) / float(numeric))
                < TARGET
            )


def test_the_cubic_root_agrees_with_the_bisection(sizer):
    # U(d) = 1 is the depressed cubic d^3 - a d - b = 0 with one positive root.
    host = host_family(RATIO, YIELD_SAMPLE)
    diameter, _ = sized_members(sizer.host, AXIAL, END_MAJOR, END_MINOR)
    moment = jnp.max(jnp.abs(END_MAJOR), axis=-1) + jnp.max(jnp.abs(END_MINOR), axis=-1)
    for load_case in range(2):
        for member in range(NUM_EDGES):
            demand_axial = float(jnp.abs(AXIAL[load_case, member])) / (
                host.area_coefficient * YIELD_SAMPLE
            )
            demand_moment = float(moment[load_case, member]) / (
                host.modulus_coefficient * YIELD_SAMPLE
            )
            cubic = np.roots([1.0, 0.0, -demand_axial, -demand_moment])
            positive = cubic[np.isreal(cubic) & (cubic.real > 0.0)].real
            expected = max(float(positive[0]), DIAMETER_MINIMUM)

            assert positive.shape == (1,)
            assert np.isclose(
                float(diameter[load_case, member]),
                expected,
                rtol=TOLERANCE_ROOT,
                atol=0.0,
            )


def test_tension_and_compression_size_alike(sizer):
    pulled, _ = sized_members(sizer.host, jnp.abs(AXIAL), END_MAJOR, END_MINOR)
    pushed, _ = sized_members(sizer.host, -jnp.abs(AXIAL), END_MAJOR, END_MINOR)

    assert np.array_equal(np.asarray(pulled), np.asarray(pushed))


def test_an_unloaded_member_sits_at_the_clamp(sizer):
    idle = MemberForces(
        jnp.zeros((1, NUM_EDGES)),
        jnp.zeros((1, NUM_EDGES, 2)),
        jnp.zeros((1, NUM_EDGES, 2)),
    )
    sizes = sizer(idle, LENGTHS)

    def total(axial_force):
        forces = MemberForces(axial_force, idle.moment_major, idle.moment_minor)

        return jnp.sum(sizer(forces, LENGTHS).sections.diameter)

    gradient = jax.grad(total)(idle.axial_force)

    assert np.allclose(np.asarray(sizes.sections.diameter), DIAMETER_MINIMUM)
    assert np.allclose(np.asarray(sizes.utilization), 0.0)
    assert np.allclose(np.asarray(gradient), 0.0)


def test_a_wall_less_family_is_refused(structure):
    with pytest.raises(ValueError, match="wall"):
        BlueprintSizer(structure, TubeFamily(2.0, Steel355()))


def test_the_gradient_survives_jit(sizer, forces):
    weights = jnp.arange(1.0, 1.0 + AXIAL.size).reshape(AXIAL.shape)

    def total(diameters):
        return jnp.sum(weights * sizer.compute_utilization(diameters, forces, LENGTHS))

    eager = jax.grad(total)(HELD)
    compiled = jax.jit(jax.grad(total))(HELD)

    assert np.all(np.isfinite(np.asarray(eager)))
    assert np.array_equal(np.asarray(eager), np.asarray(compiled))


def test_the_crossed_sizes_are_the_local_ones_bit_for_bit(sizer, remote, forces):
    local = sizer(forces, LENGTHS)
    crossed = remote(forces, LENGTHS)

    assert np.array_equal(
        np.asarray(crossed.sections.diameter), np.asarray(local.sections.diameter)
    )
    assert np.array_equal(
        np.asarray(crossed.utilization), np.asarray(local.utilization)
    )


def test_the_crossed_check_is_the_local_one_bit_for_bit(sizer, remote, forces):
    local = sizer.compute_utilization(HELD, forces, LENGTHS)
    crossed = remote.compute_utilization(HELD, forces, LENGTHS)

    assert np.array_equal(np.asarray(crossed), np.asarray(local))


def test_the_crossed_gradients_are_the_local_ones_bit_for_bit(sizer, remote, forces):
    # Both sides pull through the same host cotangent functions, so the
    # boundary changes nothing about the derivative, not even its last bit.
    weights = jnp.arange(1.0, 1.0 + AXIAL.size).reshape(AXIAL.shape)

    def held_total(block, diameters):
        return jnp.sum(weights * block.compute_utilization(diameters, forces, LENGTHS))

    def sized_total(block, carried):
        return jnp.sum(weights * block(carried, LENGTHS).sections.diameter)

    local_held = jax.grad(lambda d: held_total(sizer, d))(HELD)
    crossed_held = jax.grad(lambda d: held_total(remote, d))(HELD)
    local_sized = jax.grad(lambda f: sized_total(sizer, f))(forces)
    crossed_sized = jax.grad(lambda f: sized_total(remote, f))(forces)

    assert np.array_equal(np.asarray(crossed_held), np.asarray(local_held))
    for local_leaf, crossed_leaf in zip(local_sized, crossed_sized, strict=True):
        assert np.array_equal(np.asarray(crossed_leaf), np.asarray(local_leaf))


@pytest.fixture(scope="module")
def boundary():
    return load_tesseract_api("blueprint_check")


@pytest.fixture(scope="module")
def crossing(boundary):
    return boundary.InputSchema(
        axial_force=np.asarray(AXIAL[0]),
        end_moments_major=np.asarray(END_MAJOR[0]),
        end_moments_minor=np.asarray(END_MINOR[0]),
        diameter_held=np.asarray(HELD),
        f_y=YIELD_SAMPLE,
        gamma_m0=1.0,
        ratio=RATIO,
        diameter_min=DIAMETER_MINIMUM,
    )


def test_the_boundary_reports_the_clamp_mask(boundary, crossing):
    crossed = boundary.apply(crossing)

    assert np.array_equal(np.asarray(crossed["clamped"]), [0.0, 0.0, 0.0, 1.0])


def test_a_cotangent_on_the_clamp_mask_is_refused(boundary, crossing):
    with pytest.raises(ValueError, match="clamped"):
        boundary.vector_jacobian_product(
            crossing, ["axial_force"], ["clamped"], {"clamped": np.ones(NUM_EDGES)}
        )


def test_the_solve_never_reads_the_held_size(boundary, crossing):
    seed = np.asarray([1.0, -2.0, 0.5, 3.0])
    pulled = boundary.vector_jacobian_product(
        crossing,
        ["diameter_held"],
        ["diameter", "utilization"],
        {"diameter": seed, "utilization": seed},
    )

    assert np.all(pulled["diameter_held"] == 0.0)


def test_the_pinned_utilization_has_no_derivative_where_the_check_decided(
    boundary, crossing
):
    seed = np.asarray([1.0, -2.0, 0.5, 3.0])
    pulled = boundary.vector_jacobian_product(
        crossing, ["axial_force"], ["utilization"], {"utilization": seed}
    )

    assert np.all(pulled["axial_force"][:3] == 0.0)
    assert pulled["axial_force"][3] != 0.0


def test_the_ec3_sizer_agrees_when_buckling_is_silenced(structure, family, sizer):
    # Drive the EC3 sizer's buckling length to zero and set its moment
    # combination linear, and the two libraries size alike — exactly in
    # tension, and to the first-order slenderness residual in compression.
    silenced = Ec3Sizer(structure, family, resultant=False)
    moment = jnp.asarray([2.0e6, 0.0, 5.0e7, 1.0e6])
    ends = jnp.stack([moment, moment], axis=-1)[None, :, :]
    forces = MemberForces(
        jnp.asarray([[-5.0e5, -3.0e5, -1.0e4, 2.0e5]]), ends, jnp.zeros_like(ends)
    )

    naive = sizer(forces, jnp.full(NUM_EDGES, 4000.0)).sections.diameter
    checked = silenced(forces, jnp.full(NUM_EDGES, 1e-3)).sections.diameter

    assert np.allclose(np.asarray(checked), np.asarray(naive), rtol=1e-7, atol=0.0)


def test_the_diameter_floor_matches_the_catalogue(structure):
    checked = Ec3Sizer(structure, build_section_family(Steel355(), 3))

    assert float(checked.catalogue.diameter_min) == DIAMETER_MINIMUM


def test_the_host_coefficients_match_the_sections(family):
    unit = family(1.0)
    host = host_family(RATIO, YIELD_SAMPLE)

    assert float(unit.area) == host.area_coefficient
    assert float(2.0 * unit.second_moment) == pytest.approx(
        host.modulus_coefficient, rel=1e-15
    )


def test_the_private_evaluator_is_the_public_clause():
    # The bisection runs through Blueprints' `_evaluate`; if a release moves
    # it, this fails here rather than silently changing every size.
    host = host_family(RATIO, YIELD_SAMPLE)
    generator = np.random.default_rng(20260825)

    assert blueprint_module.EVALUATOR_REACHED
    for _ in range(200):
        diameter = float(generator.uniform(30.0, 900.0))
        axial = float(generator.uniform(-6.0e5, 6.0e5))
        moment = float(generator.uniform(0.0, 3.0e7))
        through_class = blueprint_module._check_scalar(diameter, axial, moment, host)
        through_evaluate = blueprint_module._probe_scalar(diameter, axial, moment, host)

        assert through_evaluate == pytest.approx(through_class, rel=1e-15, abs=0.0)
