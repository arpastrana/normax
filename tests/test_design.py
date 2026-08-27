import jax.numpy as jnp
import numpy as np
import pytest

from normax.analysis.smax import SmaxAnalyzer
from normax.design import DesignConstraints
from normax.design import DesignParameters
from normax.design import DesignProblem
from normax.design import StructuralDesignPipeline
from normax.design import bound_variables
from normax.design import compute_mass
from normax.design import compute_member_mass
from normax.design import count_coordinates
from normax.design import design_maps
from normax.design import envelope_diameters
from normax.design import evaluate_constraints
from normax.design import expand_variables
from normax.design import fold_variables
from normax.design import initialize_optimization_variables
from normax.design import optimize_design
from normax.design import read_coordinates
from normax.design import read_design
from normax.design import read_member_densities
from normax.design import unfold_diameters
from normax.form_finding import FdmFormFinder
from normax.form_finding import build_equilibrium_graph
from normax.form_finding import build_plan_basis
from normax.form_finding import select_free_nodes
from normax.form_finding import solve_equilibrium
from normax.loads import assemble_load_cases
from normax.loads import load_half_span
from normax.loads import load_uniform
from normax.materials import Steel355
from normax.optimization import OptimizationBudget
from normax.sections import TubeFamily
from normax.sections import build_section_family
from normax.sizing.ec3 import Ec3Sizer
from normax.structures import build_arch_2d
from normax.structures import build_warren_2d
from normax.symmetry import SignGuard
from normax.symmetry import build_member_spread
from normax.symmetry import permute_members

# A 10 m arch rising 3 m under 180 kN spread over its free nodes. Units are
# millimeters and newtons.
SPAN = 10_000.0
RISE = 3_000.0
TOTAL_LOAD = 180_000.0
NUM_EDGES = 10

# The diameter the frame is analyzed with before the check has spoken.
SEED = 100.0

# What the arch's descent is held to beside the check.
FLOOR = 25.0
BOUNDS = (-500.0, -1.0)

# Invariant 6.5 of CLAUDE.md, read at the enveloped start.
TOLERANCE_UTILIZATION = 1e-9


@pytest.fixture(scope="module")
def grade():
    return Steel355()


@pytest.fixture(scope="module")
def family(grade):
    return build_section_family(grade, 3)


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


@pytest.fixture(scope="module")
def pipeline(structure, family):
    return StructuralDesignPipeline(
        FdmFormFinder(structure),
        SmaxAnalyzer(structure, family(SEED)),
        Ec3Sizer(structure, family),
    )


@pytest.fixture(scope="module")
def params(force_densities):
    return DesignParameters(force_densities, jnp.full(NUM_EDGES, SEED))


@pytest.fixture(scope="module")
def problem(structure, pipeline, three_cases):
    constraints = DesignConstraints(FLOOR, 0.0, None, None, None, BOUNDS)

    return DesignProblem(structure, pipeline, three_cases, None, None, constraints)


# --------------------------------------------------------------------------- #
# The Warren problem, where the two linear maps are not the identity
# --------------------------------------------------------------------------- #
WARREN_BAYS = 8


def warren_mirror():
    bottom = WARREN_BAYS - np.arange(WARREN_BAYS + 1)
    top = 2 * WARREN_BAYS - np.arange(WARREN_BAYS)

    return np.concatenate([bottom, top])


@pytest.fixture(scope="module")
def warren():
    return build_warren_2d(num_bays=WARREN_BAYS, span=SPAN, depth=1_200.0)


@pytest.fixture(scope="module")
def warren_problem(warren, family):
    blocks = StructuralDesignPipeline(
        FdmFormFinder(warren),
        SmaxAnalyzer(warren, family(SEED)),
        Ec3Sizer(warren, family),
    )
    loads = assemble_load_cases([load_uniform(warren, TOTAL_LOAD)])
    basis = build_plan_basis(warren, warren_mirror(), pivoted=True)
    spread = build_member_spread(warren, (warren_mirror(),))
    constraints = DesignConstraints(FLOOR, 0.0, None, None, None, None)

    return DesignProblem(warren, blocks, loads, basis, spread, constraints)


@pytest.fixture(scope="module")
def warren_q(warren_problem):
    """A density vector inside the held-plan span, negative throughout."""
    basis = warren_problem.basis
    xi = -1.0 - np.linspace(0.0, 1.0, basis.width)

    return np.asarray(basis.densities(jnp.asarray(xi)))


# --------------------------------------------------------------------------- #
# The sizer block is built from its family alone
# --------------------------------------------------------------------------- #
def test_the_sizer_reads_its_class_off_its_family(structure, grade):
    for section_class in (1, 2, 3):
        family = build_section_family(grade, section_class)
        sizer = Ec3Sizer(structure, family)

        assert sizer.section_class == section_class


def test_the_sizer_refuses_a_class_four_family(structure, grade):
    with pytest.raises(ValueError):
        Ec3Sizer(structure, TubeFamily(200.0, grade))


# --------------------------------------------------------------------------- #
# What the composed blocks do
# --------------------------------------------------------------------------- #
def test_the_form_finder_matches_the_free_function(
    structure, force_densities, one_case
):
    graph = build_equilibrium_graph(structure)
    state = solve_equilibrium(
        force_densities,
        structure.nodes[graph.indices_fixed],
        graph,
        one_case.formfinding,
    )
    shape = FdmFormFinder(structure)(force_densities, one_case.formfinding)

    assert jnp.array_equal(shape.xyz, state.xyz)
    assert jnp.array_equal(shape.lengths, state.lengths[:, 0])


def test_the_analyzer_stacks_one_load_case_per_row(pipeline, params, three_cases):
    shape = pipeline.formfinder(params.coordinates, three_cases.formfinding)
    forces = pipeline.analyzer(shape.xyz, params.diameters, three_cases.analysis)

    assert forces.axial_force.shape == (3, NUM_EDGES)
    assert forces.moment_major.shape == (3, NUM_EDGES, 2)

    for load_case in range(3):
        alone = pipeline.analyzer(
            shape.xyz, params.diameters, three_cases.analysis[load_case][None]
        )
        assert jnp.array_equal(alone.axial_force[0], forces.axial_force[load_case])
        assert jnp.array_equal(alone.moment_major[0], forces.moment_major[load_case])


def test_a_geometry_is_form_found_once_for_every_load_case(
    pipeline, params, one_case, three_cases
):
    # The shape answers to one load case by construction, so adding cases to
    # check against must leave it exactly where it was.
    single = pipeline(params, one_case)
    several = pipeline(params, three_cases)

    assert jnp.array_equal(several.shape.xyz, single.shape.xyz)
    assert jnp.array_equal(several.shape.lengths, single.shape.lengths)


def test_repeating_a_load_case_repeats_its_utilization_row(pipeline, params, one_case):
    # Bit-equal within one call; the single-case call batches its solve
    # differently, so against it the rows agree only to round-off.
    once = pipeline(params, one_case)
    twice = pipeline(params, assemble_load_cases([one_case.analysis[0]] * 2))

    assert jnp.array_equal(twice.sizes.utilization[0], twice.sizes.utilization[1])
    assert np.allclose(
        np.asarray(twice.sizes.utilization[0]),
        np.asarray(once.sizes.utilization[0]),
        rtol=1e-12,
    )


def test_the_pipeline_checks_the_diameters_it_was_given(pipeline, params, three_cases):
    # The pipeline no longer sizes: the sections are the family at the given
    # diameters, and the utilization is the check read at them.
    design = pipeline(params, three_cases)
    expected = pipeline.sizer.compute_utilization(
        params.diameters, design.forces, design.shape.lengths
    )

    assert jnp.array_equal(design.sizes.sections.diameter, params.diameters)
    assert jnp.array_equal(
        design.sizes.sections.thickness,
        pipeline.sizer.family(params.diameters).thickness,
    )
    assert jnp.array_equal(design.sizes.utilization, expected)


def test_the_utilization_falls_as_the_diameters_grow(pipeline, params, three_cases):
    lean = pipeline(params, three_cases)
    fattened = DesignParameters(params.coordinates, params.diameters * 1.2)
    stout = pipeline(fattened, three_cases)

    assert jnp.all(stout.sizes.utilization < lean.sizes.utilization)


def test_the_mass_is_the_sum_of_what_the_members_weigh(pipeline, params, one_case):
    design = pipeline(params, one_case)
    tubes = pipeline.sizer.family(params.diameters)
    expected = tubes.material.density * jnp.sum(tubes.area * design.shape.lengths)

    assert float(compute_mass(design)) == pytest.approx(float(expected), rel=1e-14)
    assert float(
        compute_member_mass(design.sizes.sections, design.shape.lengths)
    ) == float(compute_mass(design))


# --------------------------------------------------------------------------- #
# The variable vector and its two linear maps
# --------------------------------------------------------------------------- #
def test_the_variable_vector_is_coordinates_then_diameters(problem, force_densities):
    diameters = np.linspace(80.0, 120.0, NUM_EDGES)
    x = np.concatenate([np.asarray(force_densities), diameters])
    expanded = expand_variables(problem, jnp.asarray(x))

    assert count_coordinates(problem) == NUM_EDGES
    assert np.allclose(np.asarray(expanded.coordinates), np.asarray(force_densities))
    assert np.allclose(np.asarray(expanded.diameters), diameters)


def test_folding_is_the_identity_without_a_subspace(problem, force_densities):
    q = np.asarray(force_densities)
    diameters = np.linspace(80.0, 120.0, NUM_EDGES)
    x = fold_variables(problem, q, diameters)

    assert np.array_equal(x, np.concatenate([q, diameters]))
    assert np.array_equal(read_coordinates(problem, q), q)
    assert np.array_equal(np.asarray(read_member_densities(problem, jnp.asarray(q))), q)


def test_the_bounds_box_the_densities_and_floor_the_diameters(problem):
    boxes = bound_variables(problem)

    assert len(boxes) == 2 * NUM_EDGES
    assert boxes[:NUM_EDGES] == [BOUNDS] * NUM_EDGES
    assert boxes[NUM_EDGES:] == [(FLOOR, None)] * NUM_EDGES


def test_subspace_coordinates_take_no_box(warren_problem):
    boxes = bound_variables(warren_problem)
    width = warren_problem.basis.width
    patterns = warren_problem.spread.shape[1]

    assert len(boxes) == width + patterns
    assert boxes[:width] == [(None, None)] * width
    assert boxes[width:] == [(FLOOR, None)] * patterns


def test_member_densities_expands_the_basis(warren_problem):
    basis = warren_problem.basis
    xi = jnp.asarray(-1.0 - np.linspace(0.0, 1.0, basis.width))

    assert jnp.array_equal(
        read_member_densities(warren_problem, xi), basis.densities(xi)
    )


def test_the_expansion_holds_the_drawn_plan(warren_problem, warren_q):
    # Expanded parameters are member-wide and keep the plan by construction.
    width = warren_problem.basis.width
    patterns = warren_problem.spread.shape[1]
    xi = np.asarray(warren_problem.basis.coordinates(warren_q))
    x = np.concatenate([xi, np.full(patterns, SEED)])

    expanded = expand_variables(warren_problem, jnp.asarray(x))

    assert expanded.coordinates.shape == (warren_problem.structure.num_edges,)
    assert count_coordinates(warren_problem) == width
    assert np.allclose(np.asarray(expanded.coordinates), warren_q)


def test_the_folded_diameters_keep_the_mirror(warren_problem, warren_q):
    rng = np.random.default_rng(3)
    diameters = rng.uniform(60.0, 160.0, warren_problem.structure.num_edges)
    x = fold_variables(warren_problem, warren_q, diameters)

    expanded = expand_variables(warren_problem, jnp.asarray(x))
    targets = permute_members(warren_mirror(), warren_problem.structure)
    unfolded = np.asarray(expanded.diameters)

    assert np.allclose(unfolded[targets], unfolded)
    assert np.all(unfolded >= diameters - 1e-12)
    assert np.array_equal(unfold_diameters(warren_problem, x), unfolded)


def test_reading_back_an_in_span_vector_is_exact(warren_problem, warren_q):
    xi = read_coordinates(warren_problem, warren_q)
    rebuilt = read_member_densities(warren_problem, jnp.asarray(xi))

    assert np.abs(np.asarray(rebuilt) - warren_q).max() < 1e-9


# --------------------------------------------------------------------------- #
# The inequality rows
# --------------------------------------------------------------------------- #
def test_the_rows_start_with_the_utilization(problem, params):
    design = problem.pipeline(params, problem.loads)
    rows = np.asarray(evaluate_constraints(problem, params, design))
    leading = 1.0 - np.asarray(design.sizes.utilization).ravel()

    assert rows.shape == (3 * NUM_EDGES,)
    assert np.allclose(rows, leading)


def test_each_optional_row_family_appears_when_asked(problem, params):
    guarded = np.arange(3)
    guard = SignGuard(-np.ones(3), guarded, 1.0, 10.0)
    held = DesignConstraints(FLOOR, 100.0, 4_000.0, -500.0, guard, BOUNDS)
    asked = problem._replace(constraints=held)

    design = asked.pipeline(params, asked.loads)
    rows = np.asarray(evaluate_constraints(asked, params, design))
    heights = select_free_nodes(asked.structure).size

    assert rows.size == 3 * NUM_EDGES + 2 * heights + NUM_EDGES + guarded.size


def test_the_sign_rows_read_the_guarded_densities(problem, params):
    guarded = np.arange(3)
    guard = SignGuard(-np.ones(3), guarded, 1.0, 10.0)
    held = DesignConstraints(FLOOR, 0.0, None, None, guard, BOUNDS)
    asked = problem._replace(constraints=held)

    design = asked.pipeline(params, asked.loads)
    rows = np.asarray(evaluate_constraints(asked, params, design))
    signed = -np.asarray(params.coordinates)[guarded]

    assert np.allclose(rows[-3:], (signed - guard.margin) / guard.scale)


# --------------------------------------------------------------------------- #
# Where a search starts, and how a vector reads back as a design
# --------------------------------------------------------------------------- #
def test_the_enveloped_start_covers_every_load_case(problem, force_densities):
    # Exact at the seed forces the envelope was sized from; re-analyzed at its
    # own sections the forces shift, which is the frozen-seed gap a search closes.
    pipeline = problem.pipeline
    diameters = envelope_diameters(problem, np.asarray(force_densities), SEED)
    held = jnp.asarray(diameters)

    shape = pipeline.formfinder(force_densities, problem.loads.formfinding)
    seeded = jnp.full(NUM_EDGES, SEED)
    frozen = pipeline.analyzer(shape.xyz, seeded, problem.loads.analysis)
    at_seed = np.asarray(
        pipeline.sizer.compute_utilization(held, frozen, shape.lengths)
    )
    settled = pipeline(DesignParameters(force_densities, held), problem.loads)
    reworked = np.asarray(settled.sizes.utilization)

    assert np.all(at_seed <= 1.0 + TOLERANCE_UTILIZATION)
    assert np.allclose(at_seed.max(axis=0), 1.0, rtol=0.0, atol=TOLERANCE_UTILIZATION)
    assert np.allclose(reworked.max(axis=0), 1.0, rtol=0.0, atol=0.05)


def test_the_enveloped_start_respects_the_floor(problem, force_densities):
    raised = problem._replace(
        constraints=problem.constraints._replace(diameter_floor=500.0)
    )
    diameters = envelope_diameters(raised, np.asarray(force_densities), SEED)

    assert np.all(diameters >= 500.0)


def test_initialize_optimization_variables_compose_the_two_reads(
    problem, force_densities
):
    q = np.asarray(force_densities)
    start = initialize_optimization_variables(problem, q, SEED)
    diameters = envelope_diameters(problem, read_coordinates(problem, q), SEED)

    assert np.array_equal(start, fold_variables(problem, q, diameters))


def test_read_design_evaluates_the_pipeline_at_the_expanded_parameters(
    problem, force_densities
):
    start = initialize_optimization_variables(
        problem, np.asarray(force_densities), SEED
    )
    design = read_design(problem, start)
    expanded = expand_variables(problem, jnp.asarray(start))
    expected = problem.pipeline(expanded, problem.loads)

    assert jnp.array_equal(design.shape.xyz, expected.shape.xyz)
    assert jnp.array_equal(design.sizes.utilization, expected.sizes.utilization)


# --------------------------------------------------------------------------- #
# The compiled maps and the descent over them
# --------------------------------------------------------------------------- #
def test_the_maps_agree_with_their_eager_counterparts(problem, force_densities):
    maps = design_maps(problem)
    start = initialize_optimization_variables(
        problem, np.asarray(force_densities), SEED
    )
    x = jnp.asarray(start)

    design = read_design(problem, start)
    expanded = expand_variables(problem, x)
    weighed, _ = maps.objective(x)
    rows = np.asarray(maps.slack(x))
    expected = np.asarray(evaluate_constraints(problem, expanded, design))

    assert float(weighed) == pytest.approx(float(compute_mass(design)), rel=1e-12)
    assert np.allclose(rows, expected, rtol=0.0, atol=1e-12)


def test_a_satisfied_start_pays_no_penalty(problem, force_densities):
    # With zero multipliers and no violation the augmented objective is the
    # normalized mass alone.
    maps = design_maps(problem)
    q = np.asarray(force_densities)
    fattened = 1.5 * envelope_diameters(problem, q, SEED)
    x = jnp.asarray(fold_variables(problem, q, fattened))

    rows = np.asarray(maps.slack(x))
    assert np.all(rows > 0.0)

    reference = jnp.asarray(float(maps.objective(x)[0]))
    resting = jnp.zeros(rows.size)
    augmented, _ = maps.augmented(x, resting, reference, reference)

    assert float(augmented) == pytest.approx(1.0, rel=1e-12)


def test_the_descent_reports_the_mass_of_the_point_it_ends_on(problem, force_densities):
    start = initialize_optimization_variables(
        problem, np.asarray(force_densities), SEED
    )
    budget = OptimizationBudget(
        rounds=3,
        iterations=10,
        settled=10,
        opening=1,
        penalty=1.0,
        growth=4.0,
        ceiling=1e8,
        tolerance=1e-3,
        quiet=1e-8,
    )
    answer = optimize_design(problem, start, budget)
    landed = compute_mass(read_design(problem, answer.variables))

    assert answer.objectives.shape == answer.violations.shape
    assert answer.variables.shape == start.shape
    assert float(answer.objectives[0]) == pytest.approx(
        float(compute_mass(read_design(problem, start))), rel=1e-12
    )
    assert float(answer.objectives[-1]) == pytest.approx(float(landed), rel=1e-12)
