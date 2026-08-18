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
from scipy.optimize import minimize

from normax.analysis import MemberForces
from normax.analysis.smax import SmaxAnalyzer
from normax.design import DesignParameters
from normax.design import StructuralDesignPipeline
from normax.design import compute_mass
from normax.design import design_envelope
from normax.form_finding.fdm import FdmFormFinder
from normax.loads import assemble_load_cases as load_cases_of
from normax.loads import loads_uniform
from normax.materials import Steel355
from normax.sections import TubeFamily
from normax.sizing import MemberSizes
from normax.sizing import blueprint as blueprint_module
from normax.sizing.blueprint import DIAMETER_MINIMUM
from normax.sizing.blueprint import BlueprintSizer
from normax.sizing.blueprint import checked_utilization
from normax.sizing.blueprint import demand_moment
from normax.sizing.blueprint import host_family
from normax.sizing.blueprint import sized_diameter
from normax.sizing.ec3 import Ec3Sizer
from normax.sizing.ec3 import thinnest_family
from normax.structures import build_arch_2d
from normax.tesseract import BlueprintClient
from normax.tesseract import blueprint_tesseract

# The proof this file exists to make: an external, non-differentiable, scalar
# code library — blueprints, LGPL, experiment-only — fills the sizing contract
# and carries an exact adjoint, in process through a pure_callback with a
# hand-derived implicit rule. The one EC3 name here is normax's own adapter,
# imported to be disagreed with.

SPAN = 10_000.0
RISE = 3_000.0
TOTAL_LOAD = 180_000.0
NUM_EDGES = 10

# The diameter the frame is analyzed with before the check has spoken.
SEED = 100.0

# A wall proportion this file picks for itself; no class limit chose it.
RATIO = 50.0

# Invariant 6.5 of CLAUDE.md, philosophy-independent.
TOLERANCE_UTILIZATION = 1e-9

# The hand adjoint against central differences, experiment-01's bar.
TARGET = 1e-8

# The bisection against the closed-form cubic root.
TOLERANCE_ROOT = 1e-12

# The boundary against the in-process block: values, then gradients.
TOLERANCE_PARITY = 1e-14
TOLERANCE_DERIVATIVE = 1e-12

# Sample actions for the stage-alone derivative checks, in N and Nmm.
AXIAL_SAMPLE = jnp.asarray([-2.0e5, 1.5e5, -8.0e4])
MOMENT_SAMPLE = jnp.asarray([3.0e6, 5.0e5, 1.0e7])

AREA_SAMPLE = 7367.034773
YIELD_SAMPLE = 355.0
GAMMA_SAMPLE = 1.0


@pytest.fixture(scope="module")
def structure():
    return build_arch_2d(num_edges=NUM_EDGES, span=SPAN, rise=RISE)


@pytest.fixture(scope="module")
def pipeline(structure):
    grade = Steel355()
    family = TubeFamily(RATIO, grade)

    return StructuralDesignPipeline(
        FdmFormFinder(structure),
        SmaxAnalyzer(structure, family(SEED)),
        BlueprintSizer(structure, family),
    )


@pytest.fixture(scope="module")
def params(structure):
    trial = jnp.full(NUM_EDGES, -1.0)
    shape = FdmFormFinder(structure)(trial, funicular(structure))
    reached = jnp.max(shape.xyz[:, 2])

    return DesignParameters(trial * reached / RISE, jnp.full(NUM_EDGES, SEED))


def funicular(structure):
    """
    The uniform load case the arch is form-found under.
    """
    return loads_uniform(structure, TOTAL_LOAD / (NUM_EDGES - 1))


@pytest.fixture(scope="module")
def one_case(structure):
    return load_cases_of([funicular(structure)])


def test_this_file_names_no_ec3_library():
    # The drift alarm for the whole claim: the blueprints sizer only proves
    # the seam if neither it nor this file leans on the EC3 library.
    here = Path(__file__).read_text()
    backend = Path(blueprint_module.__file__).read_text()
    imported_here = [line for line in here.splitlines() if line.startswith("from ")]
    imported_backend = [
        line for line in backend.splitlines() if line.startswith("from ")
    ]

    assert not any("ec3x" in line for line in imported_here)
    assert not any("ec3x" in line for line in imported_backend)
    assert any("blueprints" in line for line in imported_backend)


def test_the_two_axial_formulas_agree():
    # 6.6 and 6.10 are the same expression for class 1-3; the residual calls
    # 6.10 (its name states the class scope) and this pins the equivalence.
    yielding = Form6Dot6DesignPlasticResistanceGrossCrossSection(
        a=AREA_SAMPLE, f_y=YIELD_SAMPLE, gamma_m0=GAMMA_SAMPLE
    )
    squashing = Form6Dot10NcRdClass1And2And3(
        a=AREA_SAMPLE, f_y=YIELD_SAMPLE, gamma_m0=GAMMA_SAMPLE
    )

    assert float(yielding) == float(squashing)


def test_a_third_philosophy_fills_the_contract(pipeline, params, one_case):
    design = pipeline(params, one_case)

    assert isinstance(design.sizes, MemberSizes)
    assert np.all(np.asarray(design.sizes.sections.diameter) > 0.0)


def test_the_blueprint_sizer_is_fully_stressed_too(pipeline, params, one_case):
    # The invariant is the sizing map's, not any standard's: wherever the size
    # was free to move, it is worked to exactly one.
    design = pipeline(params, one_case)
    diameter = np.asarray(design.sizes.sections.diameter)
    used = np.asarray(design.sizes.utilization)
    free = diameter > DIAMETER_MINIMUM

    assert np.any(free)
    assert np.allclose(used[free], 1.0, rtol=0.0, atol=TOLERANCE_UTILIZATION)
    assert np.all(used <= 1.0 + TOLERANCE_UTILIZATION)


def test_the_reread_agrees_with_the_sizes(pipeline, params, one_case):
    design = pipeline(params, one_case)
    reread = pipeline.sizer.utilization(
        design.sizes.sections.diameter[0], design.forces, design.shape.lengths
    )
    free = np.asarray(design.sizes.sections.diameter[0]) > DIAMETER_MINIMUM

    assert np.allclose(
        np.asarray(reread)[:, free], 1.0, rtol=0.0, atol=TOLERANCE_UTILIZATION
    )


def test_the_mass_still_differentiates_end_to_end(pipeline, params, one_case):
    # The composition's whole point survives the host crossing: one exact
    # gradient across all three blocks, the callback under jit and grad alike.
    def objective(q):
        design = pipeline(DesignParameters(q, params.diameters), one_case)

        return compute_mass(design_envelope(design))

    gradient = jax.grad(objective)(params.force_densities)
    compiled = jax.jit(jax.grad(objective))(params.force_densities)

    assert np.all(np.isfinite(np.asarray(gradient)))
    assert float(jnp.min(jnp.abs(gradient))) > 0.0
    assert np.allclose(np.asarray(gradient), np.asarray(compiled), rtol=1e-12, atol=0.0)


def test_the_philosophies_disagree_on_a_compressed_arch(
    structure, pipeline, params, one_case
):
    # A different philosophy, not a reimplementation: EC3 sees buckling and a
    # cross-section check does not, so a compressed arch is sized differently.
    limit_state = StructuralDesignPipeline(
        pipeline.formfinder,
        pipeline.analyzer,
        Ec3Sizer(structure, thinnest_family(Steel355(), 3)),
    )

    naive = pipeline(params, one_case).sizes.sections.diameter
    checked = limit_state(params, one_case).sizes.sections.diameter

    assert not np.allclose(np.asarray(naive), np.asarray(checked), rtol=1e-2)


def test_the_implicit_tangent_matches_check_grads():
    # check_grads perturbs by a small absolute step, so the map is rescaled to
    # unit order before it judges the hand rule, forward and reverse alike.
    scale_axial = 1.0e5
    scale_moment = 1.0e6

    def scaled(force, bent):
        solved = sized_diameter(
            RATIO, YIELD_SAMPLE, force * scale_axial, bent * scale_moment
        )

        return solved / 100.0

    force = AXIAL_SAMPLE / scale_axial
    bent = MOMENT_SAMPLE / scale_moment
    check_grads(scaled, (force, bent), order=1, modes=("fwd", "rev"))


def test_the_checker_partials_match_check_grads():
    scale_diameter = 100.0
    scale_axial = 1.0e5
    scale_moment = 1.0e6

    def scaled(size, force, bent):
        return checked_utilization(
            RATIO,
            YIELD_SAMPLE,
            size * scale_diameter,
            force * scale_axial,
            bent * scale_moment,
        )

    size = jnp.asarray([1.2, 0.8, 1.4])
    force = AXIAL_SAMPLE / scale_axial
    bent = MOMENT_SAMPLE / scale_moment
    check_grads(scaled, (size, force, bent), order=1, modes=("fwd", "rev"))


def test_central_differences_are_the_oracle():
    # The hand-derived implicit quotients against the numerical truth, at a
    # relative step, tension and compression alike.
    def total(force, bent):
        return jnp.sum(sized_diameter(RATIO, YIELD_SAMPLE, force, bent))

    gradient_axial, gradient_moment = jax.grad(total, argnums=(0, 1))(
        AXIAL_SAMPLE, MOMENT_SAMPLE
    )
    for index in range(AXIAL_SAMPLE.shape[0]):
        step_axial = 1e-6 * float(jnp.abs(AXIAL_SAMPLE[index]))
        step_moment = 1e-6 * float(jnp.abs(MOMENT_SAMPLE[index]))
        bumped = total(AXIAL_SAMPLE.at[index].add(step_axial), MOMENT_SAMPLE)
        lowered = total(AXIAL_SAMPLE.at[index].add(-step_axial), MOMENT_SAMPLE)
        numeric_axial = (bumped - lowered) / (2.0 * step_axial)
        bumped = total(AXIAL_SAMPLE, MOMENT_SAMPLE.at[index].add(step_moment))
        lowered = total(AXIAL_SAMPLE, MOMENT_SAMPLE.at[index].add(-step_moment))
        numeric_moment = (bumped - lowered) / (2.0 * step_moment)

        assert (
            float(
                jnp.abs(gradient_axial[index] - numeric_axial) / jnp.abs(numeric_axial)
            )
            < TARGET
        )
        assert (
            float(
                jnp.abs(gradient_moment[index] - numeric_moment)
                / jnp.abs(numeric_moment)
            )
            < TARGET
        )


def test_the_cubic_root_agrees_with_the_bisection():
    # U(d) = 1 is the depressed cubic d^3 - a d - b = 0 with one positive
    # root; 55 halvings of a bracket at most 2.67 wide must sit on it.
    family = host_family(RATIO, YIELD_SAMPLE)
    solved = sized_diameter(RATIO, YIELD_SAMPLE, AXIAL_SAMPLE, MOMENT_SAMPLE)
    for index in range(AXIAL_SAMPLE.shape[0]):
        demand_axial = float(
            jnp.abs(AXIAL_SAMPLE[index]) / (family.area_coefficient * YIELD_SAMPLE)
        )
        demand_moment = float(
            MOMENT_SAMPLE[index] / (family.modulus_coefficient * YIELD_SAMPLE)
        )
        cubic = np.roots([1.0, 0.0, -demand_axial, -demand_moment])
        positive = cubic[(np.isreal(cubic)) & (cubic.real > 0.0)].real

        assert positive.shape == (1,)
        assert np.isclose(
            float(solved[index]), float(positive[0]), rtol=TOLERANCE_ROOT, atol=0.0
        )


def test_tension_and_compression_size_alike():
    # A cross-section check reads the magnitude alone: no sign branch exists,
    # and the sizes are bit-for-bit equal under a flipped force.
    pulled = sized_diameter(RATIO, YIELD_SAMPLE, jnp.abs(AXIAL_SAMPLE), MOMENT_SAMPLE)
    pushed = sized_diameter(RATIO, YIELD_SAMPLE, -jnp.abs(AXIAL_SAMPLE), MOMENT_SAMPLE)

    assert np.array_equal(np.asarray(pulled), np.asarray(pushed))


def test_an_unloaded_member_sits_at_the_clamp(structure):
    # Zero actions: no root, the catalogue floor binds, the tangent is dead,
    # and blueprints never saw a non-positive trial to raise on.
    grade = Steel355()
    family = TubeFamily(RATIO, grade)
    sizer = BlueprintSizer(structure, family)
    idle = MemberForces(
        jnp.zeros((1, NUM_EDGES)),
        jnp.zeros((1, NUM_EDGES, 2)),
        jnp.zeros((1, NUM_EDGES, 2)),
    )
    sizes = sizer(idle, jnp.full(NUM_EDGES, 1000.0))

    def total(axial_force):
        forces = MemberForces(axial_force, idle.moment_major, idle.moment_minor)
        sized = sizer(forces, jnp.full(NUM_EDGES, 1000.0))

        return jnp.sum(sized.sections.diameter)

    gradient = jax.grad(total)(idle.axial_force)

    assert np.allclose(np.asarray(sizes.sections.diameter), DIAMETER_MINIMUM)
    assert np.allclose(np.asarray(sizes.utilization), 0.0)
    assert np.allclose(np.asarray(gradient), 0.0)


@pytest.fixture(scope="module")
def boundary():
    return load_tesseract_api("blueprint_check")


@pytest.fixture(scope="module")
def crossing(boundary):
    # One case of actions with distinct end magnitudes and one clamped member,
    # so both branches of the hand adjoint are exercised in every test. No
    # end pair ties exactly: at a tie the two rules pick different, equally
    # valid subgradients (jax splits, the hand rule routes to the first).
    axial = np.asarray([-2.0e5, 1.5e5, -8.0e4, -1.0e2])
    end_major = np.asarray(
        [[3.0e6, -1.0e6], [5.0e5, 2.0e5], [-1.0e7, 4.0e6], [0.0, 1.0e3]]
    )
    end_minor = np.asarray(
        [[2.0e5, -6.0e5], [1.0e5, -4.0e4], [1.0e6, -3.0e5], [2.0e2, -5.0e1]]
    )

    return boundary.InputSchema(
        axial_force=axial,
        end_moments_major=end_major,
        end_moments_minor=end_minor,
        f_y=YIELD_SAMPLE,
        gamma_m0=GAMMA_SAMPLE,
        ratio=RATIO,
        diameter_min=DIAMETER_MINIMUM,
    )


def traced_sizes(axial, end_major, end_minor):
    """
    Route A's sizing map over raw actions, for the boundary to be checked against.
    """
    forces = MemberForces(axial, end_major, end_minor)
    moment = demand_moment(forces)
    solved = sized_diameter(RATIO, YIELD_SAMPLE, axial, moment)

    return jnp.maximum(solved, DIAMETER_MINIMUM)


def traced_check(axial, end_major, end_minor):
    """
    Route A's utilization at the size just chosen, mirroring the boundary.
    """
    forces = MemberForces(axial, end_major, end_minor)
    moment = demand_moment(forces)
    clamped = traced_sizes(axial, end_major, end_minor)

    return checked_utilization(RATIO, YIELD_SAMPLE, clamped, axial, moment)


def test_the_boundary_values_match_the_local_block(boundary, crossing):
    # The anti-duplication drift alarm: the tesseract restates the host solver,
    # and the restatement is pinned to route A at bit-identical.
    crossed = boundary.apply(crossing)
    axial = jnp.asarray(crossing.axial_force)
    end_major = jnp.asarray(crossing.end_moments_major)
    end_minor = jnp.asarray(crossing.end_moments_minor)
    local_sizes = traced_sizes(axial, end_major, end_minor)
    local_used = traced_check(axial, end_major, end_minor)

    assert np.array_equal(np.asarray(crossed["diameter"]), np.asarray(local_sizes))
    assert np.array_equal(np.asarray(crossed["utilization"]), np.asarray(local_used))
    assert np.array_equal(np.asarray(crossed["clamped"]), [0.0, 0.0, 0.0, 1.0])


def test_the_hand_adjoint_matches_the_traced_rule(boundary, crossing):
    # The literal NumPy VJP against jax.vjp of route A, per input, for a
    # cotangent on each differentiable output in turn.
    fields = ["axial_force", "end_moments_major", "end_moments_minor"]
    axial = jnp.asarray(crossing.axial_force)
    end_major = jnp.asarray(crossing.end_moments_major)
    end_minor = jnp.asarray(crossing.end_moments_minor)
    seed = np.asarray([1.0, -2.0, 0.5, 3.0])

    _, pull_sizes = jax.vjp(traced_sizes, axial, end_major, end_minor)
    traced = pull_sizes(jnp.asarray(seed))
    handmade = boundary.vector_jacobian_product(
        crossing, fields, ["diameter"], {"diameter": seed}
    )
    for name, expected in zip(fields, traced, strict=True):
        assert np.allclose(handmade[name], np.asarray(expected), rtol=1e-12, atol=1e-18)

    _, pull_check = jax.vjp(traced_check, axial, end_major, end_minor)
    traced = pull_check(jnp.asarray(seed))
    handmade = boundary.vector_jacobian_product(
        crossing, fields, ["utilization"], {"utilization": seed}
    )
    for name, expected in zip(fields, traced, strict=True):
        assert np.allclose(handmade[name], np.asarray(expected), rtol=1e-12, atol=1e-18)
    # Where the check decided the size, the reported one is pinned there, and
    # the hand rule says exactly zero — no cancellation noise.
    assert np.all(handmade["axial_force"][:3] == 0.0)


def test_a_cotangent_on_the_clamp_mask_is_refused(boundary, crossing):
    with pytest.raises(ValueError, match="clamped"):
        boundary.vector_jacobian_product(
            crossing, ["axial_force"], ["clamped"], {"clamped": np.ones(4)}
        )


def test_the_client_composes_into_the_pipeline(structure, pipeline, params, one_case):
    # The boundary is transparent: same design to bit-level, same mass
    # gradient to the derivative tolerance, both sides eager.
    grade = Steel355()
    family = TubeFamily(RATIO, grade)
    remote = StructuralDesignPipeline(
        pipeline.formfinder,
        pipeline.analyzer,
        BlueprintClient(structure, blueprint_tesseract(), family),
    )

    local_design = pipeline(params, one_case)
    crossed_design = remote(params, one_case)

    assert np.allclose(
        np.asarray(local_design.sizes.sections.diameter),
        np.asarray(crossed_design.sizes.sections.diameter),
        rtol=TOLERANCE_PARITY,
        atol=0.0,
    )
    assert np.allclose(
        np.asarray(local_design.sizes.utilization),
        np.asarray(crossed_design.sizes.utilization),
        rtol=0.0,
        atol=TOLERANCE_PARITY,
    )

    def local_mass(q):
        design = pipeline(DesignParameters(q, params.diameters), one_case)

        return compute_mass(design_envelope(design))

    def crossed_mass(q):
        design = remote(DesignParameters(q, params.diameters), one_case)

        return compute_mass(design_envelope(design))

    oracle = jax.grad(local_mass)(params.force_densities)
    carried = jax.grad(crossed_mass)(params.force_densities)
    largest = float(jnp.max(jnp.abs(oracle)))

    assert np.allclose(
        np.asarray(oracle) / largest,
        np.asarray(carried) / largest,
        rtol=0.0,
        atol=TOLERANCE_DERIVATIVE,
    )


def test_the_simultaneous_optimum_is_fully_stressed(pipeline, params, one_case):
    # Mode B: the diameters are the optimizer's variables and the check is a
    # constraint, so the fully-stressed state arrives as active constraints
    # rather than as a bisection's answer — and it is the same state.
    sizer = pipeline.sizer
    shape = pipeline.formfinder(params.force_densities, one_case.formfinding)
    density = sizer.family.material.density

    def weigh(diameters):
        sections = sizer.family(diameters)

        return jnp.sum(sections.area * shape.lengths) * density

    def slack(diameters):
        forces = pipeline.analyzer(shape.xyz, diameters, one_case.analysis)
        used = sizer.utilization(diameters, forces, shape.lengths)

        return 1.0 - used.ravel()

    weigh_and_slope = jax.jit(jax.value_and_grad(weigh))
    slack_compiled = jax.jit(slack)
    slack_jacobian = jax.jit(jax.jacrev(slack))

    def objective(x):
        value, slope = weigh_and_slope(jnp.asarray(x))

        return float(value), np.asarray(slope, dtype=np.float64)

    def feasible(x):
        return np.asarray(slack_compiled(jnp.asarray(x)), dtype=np.float64)

    def feasible_jacobian(x):
        return np.asarray(slack_jacobian(jnp.asarray(x)), dtype=np.float64)

    held = {"type": "ineq", "fun": feasible, "jac": feasible_jacobian}
    bounds = [(DIAMETER_MINIMUM, None)] * NUM_EDGES
    start = np.full(NUM_EDGES, SEED)
    found = minimize(
        objective,
        start,
        jac=True,
        method="SLSQP",
        bounds=bounds,
        constraints=[held],
        options={"maxiter": 200, "ftol": 1e-12},
    )

    assert found.success
    answered = jnp.asarray(found.x)
    forces = pipeline.analyzer(shape.xyz, answered, one_case.analysis)
    used = sizer.utilization(answered, forces, shape.lengths)
    worked = np.max(np.asarray(used), axis=0)
    unbound = np.asarray(answered) > DIAMETER_MINIMUM + 1e-9

    assert np.all(unbound)
    assert np.allclose(worked, 1.0, rtol=0.0, atol=1e-6)

    # The nested route re-derives the same sizes: one sizer pass at the
    # optimizer's own forces is the self-consistent fixed point it found.
    sized = sizer(forces, shape.lengths)
    demanded = np.max(np.asarray(sized.sections.diameter), axis=0)

    assert np.allclose(demanded, np.asarray(answered), rtol=1e-5, atol=0.0)


def test_the_constraint_jacobian_matches_the_checker_partials(
    pipeline, params, one_case
):
    # At fixed forces the constraint Jacobian is diagonal, and its diagonal is
    # the hand-derived partial of the check in the diameter.
    design = pipeline(params, one_case)
    trial = jnp.full(NUM_EDGES, 80.0)

    def used_at(diameters):
        reread = pipeline.sizer.utilization(
            diameters, design.forces, design.shape.lengths
        )

        return reread[0]

    jacobian = np.asarray(jax.jacrev(used_at)(trial))

    family = host_family(RATIO, YIELD_SAMPLE)
    axial = np.asarray(design.forces.axial_force[0])
    moment = np.asarray(demand_moment(design.forces)[0])
    demand_axial = np.abs(axial) / (family.area_coefficient * YIELD_SAMPLE)
    demand_moment_units = moment / (family.modulus_coefficient * YIELD_SAMPLE)
    size = np.asarray(trial)
    slope = -(2.0 * demand_axial / size**3 + 3.0 * demand_moment_units / size**4)

    off_diagonal = jacobian - np.diag(np.diag(jacobian))

    assert np.all(off_diagonal == 0.0)
    assert np.allclose(np.diag(jacobian), slope, rtol=TOLERANCE_DERIVATIVE, atol=0.0)


def test_the_ec3_sizer_agrees_when_buckling_is_silenced(structure):
    # The stress test's claim, pinned: drive the EC3 sizer's buckling length
    # to zero and set its moment combination linear, and the two libraries
    # size alike — exactly in tension, and to the first-order slenderness
    # residual 0.6*lambda*n*m in compression. All disagreement is section 6.3.
    family = TubeFamily(RATIO, Steel355())
    blueprint = BlueprintSizer(structure, family)
    silenced = Ec3Sizer(structure, family, resultant=False)
    moment = jnp.asarray([2.0e6, 0.0, 5.0e7, 1.0e6])
    ends = jnp.stack([moment, moment], axis=-1)[None, :, :]
    forces = MemberForces(
        jnp.asarray([[-5.0e5, -3.0e5, -1.0e4, 2.0e5]]), ends, jnp.zeros_like(ends)
    )
    members = moment.shape[0]

    naive = blueprint(forces, jnp.full(members, 4000.0)).sections.diameter
    checked = silenced(forces, jnp.full(members, 1e-3)).sections.diameter

    assert np.allclose(np.asarray(checked), np.asarray(naive), rtol=1e-7, atol=0.0)


def test_the_diameter_floor_matches_the_catalogue(structure):
    # The cross-repo drift alarm the constant's comment promises: both
    # pipelines clamp to the same floor, read here off the EC3 adapter's own
    # catalogue rather than by importing the clause library into this file.
    checked = Ec3Sizer(structure, thinnest_family(Steel355(), 3))

    assert float(checked.catalogue.diameter_min) == DIAMETER_MINIMUM


def test_the_host_coefficients_match_the_sections():
    # Three statements of the annulus exist (sections.py and the two twins);
    # this pins the host's to the neutral container's at a unit diameter, so
    # a wall-convention change in one cannot drift past a green suite.
    family = TubeFamily(RATIO, Steel355())
    unit = family(1.0)
    host = host_family(RATIO, YIELD_SAMPLE)

    assert float(unit.area) == host.area_coefficient
    assert float(unit.modulus_elastic) == pytest.approx(
        host.modulus_coefficient, rel=1e-15
    )
