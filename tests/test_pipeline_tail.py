# SPDX-License-Identifier: Apache-2.0
"""
A pipeline whose tail is cut, and the objectives each length can answer.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest
from jax.test_util import check_grads

from normax.design import DesignConstraints
from normax.design import DesignParameters
from normax.design import DesignProblem
from normax.design import StructuralDesignPipeline
from normax.design import build_compliance_objective
from normax.design import compute_compliance
from normax.design import compute_mass
from normax.design import compute_mass_problem
from normax.design import design_maps
from normax.design import evaluate_constraints
from normax.design import expand_variables
from normax.design import fold_variables
from normax.form_finding import FdmFormFinder
from normax.loads import assemble_load_cases
from normax.loads import create_load_uniform
from normax.materials import Steel355
from normax.optimization import DescentHistory
from normax.optimization import OptimizationSolution
from normax.sections import build_section_catalog
from normax.structures import build_arch_2d
from normax.tesseract import TesseractAnalyzer
from normax.tesseract import TesseractSizer
from normax.visualization import DescentPanel
from normax.visualization import DescentTrace
from normax.visualization import animate_descent
from normax.visualization import draw_design_figures
from normax.visualization.animations import FRAMES_HELD
from normax.visualization.animations import FRAMES_MOST
from normax.visualization.animations import HEIGHT_DRAWING
from normax.visualization.animations import name_frame
from normax.visualization.animations import pick_frames
from normax.visualization.animations import read_drawing_height

SPAN = 4_000.0
RISE = 1_200.0
TOTAL_LOAD = 180_000.0
NUM_EDGES = 4

SEED = 120.0
DIAMETER_FLOOR = 20.0
DENSITY = -80.0


@pytest.fixture(scope="module")
def structure():
    return build_arch_2d(num_edges=NUM_EDGES, span=SPAN, rise=RISE)


@pytest.fixture(scope="module")
def loads(structure):
    return assemble_load_cases([create_load_uniform(structure, TOTAL_LOAD)])


@pytest.fixture(scope="module")
def catalog():
    return build_section_catalog(Steel355(), 3)


@pytest.fixture(scope="module")
def params():
    densities = jnp.full(NUM_EDGES, DENSITY)
    diameters = jnp.full(NUM_EDGES, SEED)

    return DesignParameters(densities, diameters)


def build_pipeline(structure, catalog, blocks):
    """
    A pipeline carrying the named tail: shape, shape and analysis, or all three.
    """
    analyzer = None
    sizer = None
    if blocks >= 2:
        analyzer = TesseractAnalyzer(structure, catalog, "pynite")
    if blocks >= 3:
        sizer = TesseractSizer(structure, catalog, "blueprint")

    return StructuralDesignPipeline(FdmFormFinder(structure), analyzer, sizer)


def test_a_cut_tail_leaves_the_fields_its_blocks_never_filled(
    structure, catalog, loads, params
):
    shaped = build_pipeline(structure, catalog, 1)(params, loads)
    analyzed = build_pipeline(structure, catalog, 2)(params, loads)
    whole = build_pipeline(structure, catalog, 3)(params, loads)

    assert (shaped.forces, shaped.sizes) == (None, None)
    assert analyzed.forces is not None and analyzed.sizes is None
    assert whole.forces is not None and whole.sizes is not None

    # The shape is the same one whichever blocks ran behind it.
    assert jnp.allclose(shaped.shape.xyz, whole.shape.xyz)
    assert jnp.allclose(analyzed.shape.lengths, whole.shape.lengths)


def test_a_check_with_no_analysis_behind_it_is_refused(structure, catalog):
    formfinder = FdmFormFinder(structure)
    sizer = TesseractSizer(structure, catalog, "blueprint")

    with pytest.raises(ValueError, match="needs an analyzer"):
        StructuralDesignPipeline(formfinder, None, sizer)


def test_a_design_reports_what_it_holds_and_refuses_what_it_does_not(
    structure, catalog, loads, params
):
    shaped = build_pipeline(structure, catalog, 1)(params, loads)
    analyzed = build_pipeline(structure, catalog, 2)(params, loads)
    whole = build_pipeline(structure, catalog, 3)(params, loads)

    assert float(compute_mass(whole)) > 0.0
    assert float(compute_compliance(whole)) > 0.0

    with pytest.raises(ValueError, match="no sections"):
        compute_mass(shaped)
    with pytest.raises(ValueError, match="no member forces"):
        compute_compliance(shaped)
    with pytest.raises(ValueError, match="sections the forces were found at"):
        compute_compliance(analyzed)


def test_compliance_matches_the_closed_form_it_is_written_as(
    structure, catalog, loads, params
):
    whole = build_pipeline(structure, catalog, 3)(params, loads)
    sections = whole.sizes.sections
    forces = whole.forces
    lengths = whole.shape.lengths

    e_mod = float(sections.material.e_mod)
    area = np.asarray(sections.area)
    inertia = np.asarray(sections.second_moment)
    spans = np.asarray(lengths)

    axial = np.asarray(forces.axial_force) ** 2 * spans / (2.0 * e_mod * area)
    bending = np.zeros_like(axial)
    for moments in (forces.moment_major, forces.moment_minor):
        near = np.asarray(moments)[..., 0]
        far = np.asarray(moments)[..., 1]
        squared = near**2 + near * far + far**2
        bending = bending + squared * spans / (6.0 * e_mod * inertia)

    by_hand = float(np.sum(axial + bending))
    assert float(compute_compliance(whole)) == pytest.approx(by_hand, rel=1e-12)


def test_compliance_falls_as_a_member_is_fattened(structure, catalog, loads):
    pipeline = build_pipeline(structure, catalog, 3)
    densities = jnp.full(NUM_EDGES, DENSITY)

    strained = []
    for diameter in (SEED, 1.5 * SEED):
        params = DesignParameters(densities, jnp.full(NUM_EDGES, diameter))
        strained.append(float(compute_compliance(pipeline(params, loads))))

    assert strained[1] < strained[0]


def build_problem(structure, catalog, loads, blocks, objective=None):
    """
    A problem over a pipeline of the named tail length.
    """
    pipeline = build_pipeline(structure, catalog, blocks)
    held = DesignConstraints(DIAMETER_FLOOR, 0.0, RISE, 0.0, None, None)
    problem = DesignProblem(structure, pipeline, loads, held)
    if objective is not None:
        problem = problem._replace(objective=objective)

    return problem


def test_the_objective_slot_defaults_to_the_mass(structure, catalog, loads, params):
    problem = build_problem(structure, catalog, loads, 3)

    assert problem.objective is compute_mass_problem

    design = problem.pipeline(params, loads)
    weighed = float(compute_mass_problem(problem, params))
    assert weighed == pytest.approx(float(compute_mass(design)), rel=1e-12)


def test_a_compliance_search_runs_on_a_pipeline_with_no_check(
    structure, catalog, loads, params
):
    strain = build_compliance_objective(catalog)
    problem = build_problem(structure, catalog, loads, 2, strain)

    x = jnp.concatenate([params.shape_parameters, params.diameters])
    maps = design_maps(problem)
    value, slope = maps.objective(x)

    assert float(value) > 0.0
    assert np.all(np.isfinite(np.asarray(slope)))
    assert np.any(np.asarray(slope) != 0.0)

    # And it agrees with reading the compliance off a whole design at the same
    # point, where the third block is present to supply the sections.
    whole = build_pipeline(structure, catalog, 3)(params, loads)
    assert float(value) == pytest.approx(float(compute_compliance(whole)), rel=1e-10)


def test_a_pipeline_with_no_check_states_only_its_geometry_rows(
    structure, catalog, loads, params
):
    problem = build_problem(structure, catalog, loads, 2)
    design = problem.pipeline(params, loads)
    rows = np.asarray(evaluate_constraints(problem, params, design))

    # Two rise-and-sag rows per free node, and no utilization rows at all.
    free = NUM_EDGES - 1
    assert rows.size == 2 * free
    assert np.all(np.isfinite(rows))


def test_a_problem_holding_nothing_at_all_is_refused(structure, catalog, loads, params):
    pipeline = build_pipeline(structure, catalog, 2)
    bare = DesignConstraints(DIAMETER_FLOOR, 0.0, None, None, None, None)
    problem = DesignProblem(structure, pipeline, loads, bare)
    design = pipeline(params, loads)

    with pytest.raises(ValueError, match="states no constraints"):
        evaluate_constraints(problem, params, design)


def test_the_mass_objective_refuses_a_pipeline_with_no_check(
    structure, catalog, loads, params
):
    problem = build_problem(structure, catalog, loads, 2)

    with pytest.raises(ValueError, match="no check"):
        compute_mass_problem(problem, params)


def test_the_compliance_gradient_survives_a_finite_difference_check(
    structure, catalog, loads, params
):
    strain = build_compliance_objective(catalog)
    problem = build_problem(structure, catalog, loads, 2, strain)
    x = jnp.concatenate([params.shape_parameters, params.diameters])

    # Scaled so the two halves of the vector are comparable to the stepper: a
    # density is order 100 and a diameter order 100 here, so no rescale is
    # needed, but the compliance itself is large and the check is relative.
    def strained(variables):
        return strain(problem, expand_variables(problem, variables))

    check_grads(strained, (x,), order=1, modes=("rev",), atol=1e-5, rtol=1e-5)


def test_a_checkless_run_draws_a_descent_but_no_utilization_figure(
    structure, catalog, loads, params
):
    designs = {
        "start": build_pipeline(structure, catalog, 2)(params, loads),
        "answer": build_pipeline(structure, catalog, 2)(params, loads),
    }
    walked = DescentHistory(
        iterates=np.zeros((2, 2 * NUM_EDGES - 1)),
        objectives=np.array([2.0, 1.0]),
        violations=np.array([0.0, 0.0]),
        round_index=np.arange(2),
    )
    answer = OptimizationSolution(
        parameters=np.zeros(2 * NUM_EDGES - 1),
        rounds=walked,
        iterations=None,
        evaluations=2,
        converged=True,
    )

    trace = DescentTrace("auglag", answer.rounds, 1e-6)
    panel = DescentPanel("objective", "round", (trace,))
    drawn, descended = draw_design_figures(structure, designs, ("LC1",), panel)

    # No design carries a utilization to color by, so there is no first figure
    # rather than an empty one — draw_utilization reads a widest diameter and a
    # least-worked member across the designs, and neither exists over none.
    assert drawn is None
    assert descended is not None


def test_a_descent_recorded_per_iteration_draws_its_round_crossings(
    structure, catalog, loads, params
):
    designs = {"answer": build_pipeline(structure, catalog, 2)(params, loads)}
    width = 2 * NUM_EDGES - 1
    walked = DescentHistory(
        iterates=np.zeros((5, width)),
        objectives=np.array([3.0, 2.5, 2.2, 2.1, 2.0]),
        violations=np.array([1.0, 1e-2, 1e-4, 0.0, 0.0]),
        round_index=np.array([0, 1, 1, 2, 2]),
    )
    answer = OptimizationSolution(
        parameters=np.zeros(width),
        rounds=DescentHistory(
            iterates=np.zeros((3, width)),
            objectives=np.array([3.0, 2.2, 2.0]),
            violations=np.array([1.0, 1e-4, 0.0]),
            round_index=np.arange(3),
        ),
        iterations=walked,
        evaluations=9,
        converged=True,
    )

    trace = DescentTrace("auglag", answer.iterations, 1e-6)
    panel = DescentPanel("objective", "iteration", (trace,))
    _, descended = draw_design_figures(structure, designs, ("LC1",), panel)

    # Two rounds after the first, so two rules behind the curve on each panel.
    violated, descent = descended.axes[0], descended.axes[1]
    for panel in (violated, descent):
        crossings = [line for line in panel.lines if line.get_linewidth() == 0.5]
        assert len(crossings) == 2
    plt.close(descended)


# --------------------------------------------------------------------------- #
# The animation of a descent
# --------------------------------------------------------------------------- #
def traced_walk(problem, params, points):
    """
    A walk of the named length, every point the same folded variable vector.
    """
    folded = fold_variables(
        problem,
        np.asarray(params.shape_parameters),
        np.asarray(params.diameters),
    )
    iterates = np.repeat(np.asarray(folded)[None, :], points, axis=0)
    objectives = np.linspace(3.0, 2.0, points)
    violations = np.geomspace(1.0, 1e-8, points)

    return DescentHistory(iterates, objectives, violations, np.arange(points))


def test_an_animation_draws_one_frame_per_recorded_point(
    structure, catalog, loads, params
):
    problem = build_problem(structure, catalog, loads, 3)
    walked = traced_walk(problem, params, 4)
    panel = DescentPanel(
        "mass [t]", "iteration", (DescentTrace("auglag", walked, 1e-6),)
    )

    played = animate_descent(problem, panel)

    # The held frames at the end sit on the answer, so the walk is not extended.
    frames = len(list(played.new_frame_seq()))
    assert frames == walked.objectives.size + FRAMES_HELD
    plt.close("all")


def test_an_animation_refuses_a_pipeline_that_carries_no_check(
    structure, catalog, loads, params
):
    problem = build_problem(structure, catalog, loads, 2)
    walked = traced_walk(problem, params, 3)
    panel = DescentPanel(
        "objective", "iteration", (DescentTrace("auglag", walked, 1e-6),)
    )

    with pytest.raises(ValueError, match="no check"):
        animate_descent(problem, panel)


def test_an_animation_refuses_more_than_one_descent(structure, catalog, loads, params):
    problem = build_problem(structure, catalog, loads, 3)
    walked = traced_walk(problem, params, 3)
    trace = DescentTrace("auglag", walked, 1e-6)
    panel = DescentPanel("mass [t]", "iteration", (trace, trace))

    with pytest.raises(ValueError, match="one descent"):
        animate_descent(problem, panel)


def test_a_flat_shape_still_gets_a_readable_drawing_panel():
    # A span a hundred times its own height would otherwise ask for a panel of
    # a few millimeters and push the curves off the page.
    tall = read_drawing_height((0.0, 10000.0), (0.0, 100.0))

    assert tall == pytest.approx(HEIGHT_DRAWING)


def test_a_frame_is_named_for_its_round_at_either_resolution():
    walked = DescentHistory(
        iterates=np.zeros((3, 1)),
        objectives=np.zeros(3),
        violations=np.zeros(3),
        round_index=np.array([0, 1, 1]),
    )
    fine = DescentPanel("mass [t]", "iteration", (DescentTrace("a", walked, 1e-6),))
    coarse = DescentPanel("mass [t]", "round", (DescentTrace("a", walked, 1e-6),))

    assert name_frame(fine, walked, 2) == "iteration 2, round 1"
    # A round at a time, the two numbers are one, so only one is printed.
    assert name_frame(coarse, walked, 2) == "round 2"


def test_a_long_walk_is_thinned_to_a_watchable_number_of_frames():
    # An even stride, the whole descent seen at lower resolution rather than
    # truncated, and the answer kept whatever the stride would have left.
    for count in (69, FRAMES_MOST, FRAMES_MOST + 1, 617, 1243):
        picked = pick_frames(count)
        assert picked.size <= FRAMES_MOST
        assert int(picked[0]) == 0
        assert int(picked[-1]) == count - 1
        assert np.all(np.diff(picked) > 0)
    # Short walks are drawn point for point.
    assert np.array_equal(pick_frames(69), np.arange(69))
