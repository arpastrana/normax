import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from ec3x.section import DIAMETER_MINIMUM

from normax.analysis.smax import SmaxAnalyzer
from normax.design import DesignParameters
from normax.design import StructuralDesignPipeline
from normax.design import compute_mass
from normax.design import compute_member_mass
from normax.figures import draw_design_figures
from normax.form_finding import FdmFormFinder
from normax.loads import LoadCases
from normax.loads import assemble_load_cases
from normax.loads import load_half_span
from normax.loads import load_uniform
from normax.loads import select_load_case
from normax.materials import Steel355
from normax.optimization import OptimizationAnswer
from normax.sections import build_section_family
from normax.sizing.ec3 import Ec3Sizer
from normax.sizing.ec3 import coerce_member_actions
from normax.structures import build_arch_2d

matplotlib.use("Agg")

# A 10 m arch rising 3 m under 180 kN spread over its free nodes. Units are
# millimeters and newtons.
SPAN = 10_000.0
RISE = 3_000.0
TOTAL_LOAD = 180_000.0
NUM_EDGES = 10

# The diameter the frame is analyzed with before the check has spoken.
SEED = 100.0

# Invariant 6.5 of CLAUDE.md, at sections demanded from the same forces.
TOLERANCE_UTILIZATION = 1e-9

# Relative step at which the central difference plateaus, and the agreement
# measured there, scaled by the largest component of the gradient.
STEP = 1e-5
TOLERANCE_GRADIENT = 5e-8

# Three load cases make the sum longer to accumulate, so cancellation dominates
# a decade sooner and the plateau sits at a larger step.
STEP_CASES = 1e-4
TOLERANCE_GRADIENT_CASES = 2e-7


@pytest.fixture(scope="module")
def steel():
    return Steel355()


@pytest.fixture(scope="module")
def structure():
    return build_arch_2d(num_edges=NUM_EDGES, span=SPAN, rise=RISE)


@pytest.fixture(scope="module")
def one_case(structure):
    return assemble_load_cases([load_uniform(structure, TOTAL_LOAD)])


@pytest.fixture(scope="module")
def three_cases(structure):
    cases = [
        load_uniform(structure, TOTAL_LOAD),
        load_half_span(structure, TOTAL_LOAD, factor=0.25),
        load_half_span(structure, TOTAL_LOAD, factor=0.25, mirrored=True),
    ]

    return assemble_load_cases(cases)


@pytest.fixture(scope="module")
def force_densities(structure, one_case):
    """Force densities reaching the target rise, so the arch is the same one."""
    trial = jnp.full(NUM_EDGES, -1.0)
    shape = FdmFormFinder(structure)(trial, one_case.formfinding)

    return trial * jnp.max(shape.xyz[:, 2]) / RISE


def pipeline_of(structure, steel, section_class):
    """
    The three blocks a design is solved by, compiled against the arch.
    """
    family = build_section_family(steel, section_class)

    return StructuralDesignPipeline(
        FdmFormFinder(structure),
        SmaxAnalyzer(structure, family(SEED)),
        Ec3Sizer(structure, family),
    )


def demanded_design(pipeline, q, loads):
    """
    The design at the sections the funicular case demands of the seed forces.
    """
    seeded = jnp.full(NUM_EDGES, SEED)
    shape = pipeline.formfinder(q, loads.formfinding)
    forces = pipeline.analyzer(shape.xyz, seeded, loads.analysis)
    sizes = pipeline.sizer(forces, shape.lengths)
    diameters = jnp.max(sizes.sections.diameter, axis=0)

    return diameters, forces, shape


@pytest.fixture(scope="module")
def pipeline(structure, steel):
    return pipeline_of(structure, steel, 3)


@pytest.fixture(scope="module")
def checked(pipeline, force_densities, one_case):
    """The one-case design, checked at the sections that case demanded."""
    diameters, _, _ = demanded_design(pipeline, force_densities, one_case)
    params = DesignParameters(force_densities, diameters)

    return pipeline(params, one_case)


# --------------------------------------------------------------------------- #
# The invariant the sizing map is built to hold, read through the check
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("section_class", [2, 3])
def test_every_member_is_utilized_exactly_once_over(
    structure, steel, force_densities, one_case, section_class
):
    # The check reads the demanded sections at the same forces they were sized
    # from, so the invariant holds exactly rather than to the frozen-seed gap.
    pipeline = pipeline_of(structure, steel, section_class)
    diameters, forces, shape = demanded_design(pipeline, force_densities, one_case)
    worked = pipeline.sizer.compute_utilization(diameters, forces, shape.lengths)

    assert np.allclose(np.asarray(worked), 1.0, rtol=0.0, atol=TOLERANCE_UTILIZATION)


@pytest.mark.parametrize("section_class", [2, 3])
def test_no_member_is_pinned_to_the_catalogue_minimum(
    structure, steel, force_densities, one_case, section_class
):
    pipeline = pipeline_of(structure, steel, section_class)
    diameters, _, _ = demanded_design(pipeline, force_densities, one_case)

    assert float(jnp.min(diameters)) > float(DIAMETER_MINIMUM)


# --------------------------------------------------------------------------- #
# What the composition produces
# --------------------------------------------------------------------------- #
def test_the_mass_is_the_sum_over_members(pipeline, checked, steel):
    tubes = pipeline.sizer.family(checked.sizes.sections.diameter)
    by_hand = steel.density * jnp.sum(tubes.area * checked.shape.lengths)

    assert float(compute_mass(checked)) == pytest.approx(float(by_hand), rel=1e-14)


def test_every_member_is_in_compression(checked):
    assert np.all(np.asarray(checked.forces.axial_force) < 0.0)


def test_the_arch_is_symmetric_about_midspan(checked):
    diameters = np.asarray(checked.sizes.sections.diameter)
    lengths = np.asarray(checked.shape.lengths)

    assert np.allclose(diameters, diameters[::-1], rtol=1e-9)
    assert np.allclose(lengths, lengths[::-1], rtol=1e-12)


def test_a_shorter_buckling_length_never_needs_a_larger_tube(
    pipeline, force_densities, one_case
):
    _, forces, shape = demanded_design(pipeline, force_densities, one_case)
    longer = pipeline.sizer(forces, shape.lengths)
    shorter = pipeline.sizer(forces, shape.lengths * 0.5)

    assert np.all(
        np.asarray(shorter.sections.diameter) < np.asarray(longer.sections.diameter)
    )


def test_the_thinner_walled_class_is_the_lighter_one(
    structure, steel, force_densities, one_case
):
    # Both classes carry the same forces, and the Class 3 limit puts more of
    # the steel far from the axis, so the thinner wall wins on gross area.
    def class_mass(section_class):
        pipeline = pipeline_of(structure, steel, section_class)
        _, forces, shape = demanded_design(pipeline, force_densities, one_case)
        sizes = pipeline.sizer(forces, shape.lengths)
        squeezed = pipeline.sizer.family(sizes.sections.diameter[0])

        return sizes, float(compute_member_mass(squeezed, shape.lengths))

    plastic, mass_plastic = class_mass(2)
    elastic, mass_elastic = class_mass(3)

    assert mass_elastic < mass_plastic
    assert np.all(
        np.asarray(elastic.sections.diameter) > np.asarray(plastic.sections.diameter)
    )


def test_design_actions_reduce_the_two_ends(pipeline, force_densities, three_cases):
    """The design moment is the larger end, which is Table B.3 read by the check."""
    shape = pipeline.formfinder(force_densities, three_cases.formfinding)
    seeded = jnp.full(NUM_EDGES, SEED)
    forces = pipeline.analyzer(shape.xyz, seeded, three_cases.analysis)
    acting = coerce_member_actions(select_load_case(forces, 1))

    largest = jnp.max(jnp.abs(forces.moment_major[1]), axis=1)

    assert jnp.array_equal(acting.axial_force, forces.axial_force[1])
    assert jnp.allclose(jnp.abs(acting.moment_major), largest)


# --------------------------------------------------------------------------- #
# The gradient across all three stages
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("section_class", [2, 3])
def test_the_mass_gradient_matches_central_differences(
    structure, steel, force_densities, one_case, section_class
):
    pipeline = pipeline_of(structure, steel, section_class)
    seeded = jnp.full(NUM_EDGES, SEED)

    def objective(q):
        return compute_mass(pipeline(DesignParameters(q, seeded), one_case))

    q = force_densities
    gradient = jax.grad(objective)(q)
    scale = float(jnp.max(jnp.abs(gradient)))

    for edge in (0, NUM_EDGES // 2, NUM_EDGES - 1):
        step = abs(float(q[edge])) * STEP
        plus = objective(q.at[edge].add(step))
        minus = objective(q.at[edge].add(-step))
        central = float((plus - minus) / (2.0 * step))

        assert abs(float(gradient[edge]) - central) / scale < TOLERANCE_GRADIENT


def test_the_mass_gradient_is_finite_and_nowhere_zero(
    pipeline, force_densities, one_case
):
    seeded = jnp.full(NUM_EDGES, SEED)

    def objective(q):
        return compute_mass(pipeline(DesignParameters(q, seeded), one_case))

    gradient = jax.grad(objective)(force_densities)

    assert np.all(np.isfinite(np.asarray(gradient)))
    assert float(jnp.min(jnp.abs(gradient))) > 0.0


def test_forward_and_reverse_mode_agree_on_the_mass(
    pipeline, force_densities, one_case
):
    seeded = jnp.full(NUM_EDGES, SEED)

    def objective(q):
        return compute_mass(pipeline(DesignParameters(q, seeded), one_case))

    forward = jax.jacfwd(objective)(force_densities)
    reverse = jax.grad(objective)(force_densities)

    assert np.allclose(np.asarray(forward), np.asarray(reverse), rtol=1e-12)


def test_the_gradient_changes_sign_across_the_arch(pipeline, force_densities, one_case):
    # The springing gains more from length than it loses to section, and the
    # crown the other way round, so the fully-stressed mass crosses zero in
    # between. Read through the sizing map, where the section answers the force.
    seeded = jnp.full(NUM_EDGES, SEED)

    def objective(q):
        shape = pipeline.formfinder(q, one_case.formfinding)
        forces = pipeline.analyzer(shape.xyz, seeded, one_case.analysis)
        sizes = pipeline.sizer(forces, shape.lengths)
        sections = pipeline.sizer.family(sizes.sections.diameter[0])

        return compute_member_mass(sections, shape.lengths)

    gradient = jax.grad(objective)(force_densities)

    assert float(gradient[0]) > 0.0
    assert float(gradient[NUM_EDGES // 2]) < 0.0


def test_the_utilization_gradient_survives_the_composition(
    pipeline, force_densities, three_cases
):
    # One reverse pass through all three blocks, against a central difference.
    seeded = jnp.full(NUM_EDGES, SEED)

    def worked(q):
        design = pipeline(DesignParameters(q, seeded), three_cases)

        return jnp.sum(design.sizes.utilization)

    q = force_densities
    gradient = jax.grad(worked)(q)
    scale = float(jnp.max(jnp.abs(gradient)))

    assert np.all(np.isfinite(np.asarray(gradient)))

    for edge in (0, NUM_EDGES // 2, NUM_EDGES - 1):
        step = abs(float(q[edge])) * STEP_CASES
        plus = worked(q.at[edge].add(step))
        minus = worked(q.at[edge].add(-step))
        central = float((plus - minus) / (2.0 * step))

        assert abs(float(gradient[edge]) - central) / scale < TOLERANCE_GRADIENT_CASES


def test_the_utilization_moves_with_the_checked_diameters(
    pipeline, force_densities, one_case
):
    # The diameters reach both the frame's stiffness and the check, so the
    # derivative is live in every member.
    def worked(diameters):
        design = pipeline(DesignParameters(force_densities, diameters), one_case)

        return jnp.sum(design.sizes.utilization)

    gradient = jax.grad(worked)(jnp.full(NUM_EDGES, SEED))

    assert np.all(np.isfinite(np.asarray(gradient)))
    assert float(jnp.min(jnp.abs(gradient))) > 0.0


def test_the_pipeline_compiles_under_jit(pipeline, force_densities, three_cases):
    """The compiled blocks cross a trace as a pytree, with nothing rebuilt."""
    params = DesignParameters(force_densities, jnp.full(NUM_EDGES, SEED))

    def composed(blocks, design_params, loads):
        design = blocks(design_params, loads)

        return compute_mass(design) + jnp.sum(design.sizes.utilization)

    traced = eqx.filter_jit(composed)(pipeline, params, three_cases)
    eager = composed(pipeline, params, three_cases)

    assert float(traced) == pytest.approx(float(eager), rel=1e-12)


# --------------------------------------------------------------------------- #
# One structure against several load cases
# --------------------------------------------------------------------------- #
def test_the_funicular_load_case_is_not_the_one_that_decides_the_design(
    pipeline, force_densities, three_cases
):
    # A shape form-found under the first case carries it axially, so it is the
    # benign one and the members are worked hardest by what it could not see.
    diameters, _, _ = demanded_design(pipeline, force_densities, three_cases)
    design = pipeline(DesignParameters(force_densities, diameters), three_cases)
    deciding = np.argmax(np.asarray(design.sizes.utilization), axis=0)

    assert 0 not in set(deciding.tolist())


def test_a_load_case_reaches_the_analysis(
    structure, pipeline, force_densities, one_case
):
    # The load case is an argument to the analysis alone; form finding keeps
    # the loads the shape answers to.
    seeded = jnp.full(NUM_EDGES, SEED)
    shaped = pipeline(DesignParameters(force_densities, seeded), one_case)

    asymmetric = load_half_span(structure, TOTAL_LOAD)
    patched_loads = LoadCases(one_case.formfinding, jnp.stack([asymmetric]))
    patched = pipeline(DesignParameters(force_densities, seeded), patched_loads)

    assert np.allclose(np.asarray(shaped.shape.xyz), np.asarray(patched.shape.xyz))
    assert float(jnp.max(jnp.abs(patched.forces.moment_major))) > float(
        jnp.max(jnp.abs(shaped.forces.moment_major))
    )


# --------------------------------------------------------------------------- #
# The figures a run saves
# --------------------------------------------------------------------------- #
def test_the_design_figures_build(structure, pipeline, force_densities, three_cases):
    diameters, _, _ = demanded_design(pipeline, force_densities, three_cases)
    design = pipeline(DesignParameters(force_densities, diameters), three_cases)

    variables = np.concatenate([np.asarray(force_densities), np.asarray(diameters)])
    objectives = np.asarray(
        [float(compute_mass(design)), 0.9 * float(compute_mass(design))]
    )
    violations = np.asarray([0.1, 0.0])
    answer = OptimizationAnswer(variables, objectives, violations, 12, True)

    labels = ("LC1", "LC2", "LC3")
    designs = {"start": design, "answer": design}
    drawn, descended = draw_design_figures(structure, designs, labels, answer)

    assert len(drawn.axes) > 0
    assert len(descended.axes) > 0
    plt.close(drawn)
    plt.close(descended)
