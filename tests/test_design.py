from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from normax.analysis.smax import SmaxAnalyzer
from normax.design import DesignParameters
from normax.design import StructuralDesignPipeline
from normax.design import compute_mass
from normax.design import design_envelope
from normax.design import optimize_staggered
from normax.ec3.material import Steel
from normax.ec3.section import TubeCatalogue
from normax.form_finding.fdm import FdmFormFinder
from normax.form_finding.fdm import equilibrium_graph
from normax.form_finding.fdm import equilibrium_state
from normax.loads import assemble_load_cases as load_cases_of
from normax.loads import loads_half_span
from normax.loads import loads_uniform
from normax.loads import select_load_case
from normax.optimization import SearchResult
from normax.sizing.ec3 import Ec3Sizer
from normax.sizing.ec3 import design_actions
from normax.structures import build_arch_2d

# A 10 m arch rising 3 m under 180 kN spread over its free nodes. Units are
# millimeters and newtons.
SPAN = 10_000.0
RISE = 3_000.0
TOTAL_LOAD = 180_000.0
NUM_EDGES = 10

# The diameter the frame is analyzed with before the check has spoken.
SEED = 100.0

# Sharpness of the envelope in the several-load-case tests.
SHARPNESS = 50.0

# Relative step at which the central difference plateaus, and the agreement
# measured there, scaled by the largest component of the gradient. Three load
# cases make the mass longer to accumulate than one, so cancellation dominates
# sooner and the plateau sits at a larger step than a single case wants.
STEP = 1e-4
TOLERANCE_GRADIENT = 1e-7

# Bounds wide enough that a descent is not pinned by them, and what it may spend.
BOUNDS = (-500.0, -1.0)
STAGGERED_ITERATIONS = 10

# Largest fractional movement in a diameter a settled coupling may still show.
TOLERANCE_SETTLING = 1e-6

# Below this, two masses of the same design disagree only by round-off.
TOLERANCE_ROUNDOFF = 1e-9


@pytest.fixture(scope="module")
def steel():
    return Steel()


@pytest.fixture(scope="module")
def catalogue(steel):
    return TubeCatalogue.at_class_limit(steel, 3)


@pytest.fixture(scope="module")
def structure():
    return build_arch_2d(num_edges=NUM_EDGES, span=SPAN, rise=RISE)


def funicular(structure):
    """
    The uniform load case the arch is form-found under.
    """
    return loads_uniform(structure, TOTAL_LOAD / (NUM_EDGES - 1))


@pytest.fixture(scope="module")
def force_densities(structure):
    """Force densities reaching the target rise, so the arch is the same one."""
    graph = equilibrium_graph(structure)
    trial = jnp.full(NUM_EDGES, -1.0)
    state = equilibrium_state(
        trial, structure.nodes[graph.indices_fixed], graph, funicular(structure)
    )

    return trial * jnp.max(state.xyz[:, 2]) / RISE


@pytest.fixture(scope="module")
def pipeline(structure, steel, catalogue):
    return StructuralDesignPipeline(
        FdmFormFinder(structure),
        SmaxAnalyzer(structure, catalogue(SEED)),
        Ec3Sizer(structure, catalogue),
    )


@pytest.fixture(scope="module")
def params(force_densities):
    return DesignParameters(force_densities, jnp.full(NUM_EDGES, SEED))


def mass_objective(pipeline, loads):
    """
    The enveloped mass of a design, and the design that was weighed.
    """

    def objective(params):
        design = pipeline(params, loads)
        sized = design_envelope(design, SHARPNESS)

        return compute_mass(sized), sized

    return objective


class StaggeredRun(NamedTuple):
    """
    One staggered search, and how many times it traced its objective.
    """

    found: SearchResult
    traces: int


def round_seams(trajectory):
    """
    Where one round hands over to the next, which is a repeated iterate.

    A search records its starting point before it steps, so a warm-started round
    repeats the row its predecessor ended on. That is the only mark of a round
    boundary the concatenated trajectory carries.
    """
    return [
        index
        for index in range(len(trajectory.q) - 1)
        if jnp.array_equal(trajectory.q[index], trajectory.q[index + 1])
    ]


@pytest.fixture(scope="module")
def one_case(structure):
    applied = funicular(structure)

    return load_cases_of([applied])


@pytest.fixture(scope="module")
def three_cases(structure):
    load = TOTAL_LOAD / (NUM_EDGES - 1)
    cases = [
        loads_uniform(structure, load),
        loads_half_span(structure, load, factor=0.25),
        loads_half_span(structure, load, factor=0.25, mirrored=True),
    ]

    return load_cases_of(cases)


@pytest.fixture(scope="module")
def staggered(pipeline, params, three_cases):
    """One staggered search, counting the traces so every test shares them."""
    traces = []
    weighed = mass_objective(pipeline, three_cases)

    def counted(design_params):
        traces.append(1)

        return weighed(design_params)

    found = optimize_staggered(
        counted,
        params,
        bounds=BOUNDS,
        iterations=STAGGERED_ITERATIONS,
    )

    return StaggeredRun(found, len(traces))


def test_sizer_reads_its_class_off_its_family(structure, steel):
    """The class is derived from the family and never accepted beside it."""
    for section_class in (1, 2, 3):
        catalogue = TubeCatalogue.at_class_limit(steel, section_class)
        sizer = Ec3Sizer(structure, catalogue)

        assert sizer.section_class == section_class


def test_sizer_refuses_a_class_four_family(structure, steel):
    """A family too slender to be checked by these clauses is refused at build."""
    with pytest.raises(ValueError):
        Ec3Sizer(structure, TubeCatalogue(200.0, 3, steel))


def test_form_finder_matches_the_free_function(structure, force_densities, one_case):
    """The block wraps the solve and changes nothing about it."""
    graph = equilibrium_graph(structure)
    state = equilibrium_state(
        force_densities,
        structure.nodes[graph.indices_fixed],
        graph,
        one_case.formfinding,
    )
    shape = FdmFormFinder(structure)(force_densities, one_case.formfinding)

    assert jnp.array_equal(shape.xyz, state.xyz)
    assert jnp.array_equal(shape.lengths, state.lengths[:, 0])


def test_analyzer_stacks_one_load_case_per_row(pipeline, params, three_cases):
    """Each row of the stacked forces is the analysis of the matching case."""
    shape = pipeline.formfinder(params.force_densities, three_cases.formfinding)
    forces = pipeline.analyzer(shape.xyz, params.diameters, three_cases.analysis)

    assert forces.axial_force.shape == (3, NUM_EDGES)
    assert forces.moment_major.shape == (3, NUM_EDGES, 2)

    for load_case in range(3):
        alone = pipeline.analyzer(
            shape.xyz, params.diameters, three_cases.analysis[load_case][None]
        )
        assert jnp.array_equal(alone.axial_force[0], forces.axial_force[load_case])
        assert jnp.array_equal(alone.moment_major[0], forces.moment_major[load_case])


def test_repeating_a_load_case_changes_nothing(pipeline, params, one_case):
    # A case listed twice is the same structure asked the same question twice,
    # so every field has to come back identical rather than merely close. Taken
    # at the true largest, since a smooth envelope over two equal sizes is
    # deliberately above both of them.
    once = pipeline(params, one_case)
    twice = pipeline(params, load_cases_of([one_case.analysis[0]] * 2))

    demanded_once = once.sizes.sections.diameter
    demanded_twice = twice.sizes.sections.diameter

    assert jnp.array_equal(demanded_twice[0], demanded_once[0])
    assert jnp.array_equal(demanded_twice[1], demanded_once[0])
    assert jnp.array_equal(
        design_envelope(twice).sizes.sections.diameter,
        design_envelope(once).sizes.sections.diameter,
    )
    assert compute_mass(design_envelope(twice)) == compute_mass(design_envelope(once))


def test_a_geometry_is_form_found_once_for_every_load_case(
    pipeline, params, one_case, three_cases
):
    # The shape answers to one load case by construction, so adding cases to
    # check against must leave it exactly where it was.
    single = pipeline(params, one_case)
    several = pipeline(params, three_cases)

    assert jnp.array_equal(several.shape.xyz, single.shape.xyz)
    # The lengths are also what the check buckles every member over.
    assert jnp.array_equal(several.shape.lengths, single.shape.lengths)


def test_the_largest_is_the_default_and_the_envelope_bounds_it(
    pipeline, params, three_cases
):
    """The envelope never understates the size any load case demands."""
    design = pipeline(params, three_cases)
    largest = design_envelope(design)
    enveloped = design_envelope(design, SHARPNESS)

    assert jnp.array_equal(
        largest.sizes.sections.diameter,
        jnp.max(design.sizes.sections.diameter, axis=0),
    )
    assert jnp.all(enveloped.sizes.sections.diameter >= largest.sizes.sections.diameter)
    assert compute_mass(enveloped) >= compute_mass(largest)


def test_one_load_case_is_never_enveloped(pipeline, params, one_case):
    """An envelope over one case is the identity, so it is not taken."""
    design = pipeline(params, one_case)

    for sharpness in (None, 1.0, SHARPNESS):
        covered = design_envelope(design, sharpness)

        assert jnp.array_equal(
            covered.sizes.sections.diameter, design.sizes.sections.diameter[0]
        )


def test_every_member_is_exactly_stressed(pipeline, params, one_case):
    """Invariant 6.5 of CLAUDE.md, through the composed blocks."""
    design = pipeline(params, one_case)

    assert jnp.max(jnp.abs(design.sizes.utilization - 1.0)) < 1e-9


def test_the_governing_load_case_is_fully_stressed(pipeline, params, three_cases):
    """A size covers every case, and works exactly for the one that set it."""
    design = pipeline(params, three_cases)
    worst = jnp.max(design.sizes.utilization, axis=0)

    assert jnp.all(design.sizes.utilization <= 1.0 + 1e-9)
    assert jnp.max(jnp.abs(worst - 1.0)) < 1e-9


def test_the_mass_is_the_sum_of_what_the_members_weigh(pipeline, params, one_case):
    """Mass is geometry: a mass per length times a length, added up."""
    design = design_envelope(pipeline(params, one_case))
    tubes = pipeline.sizer.catalogue(design.sizes.sections.diameter)
    expected = pipeline.sizer.steel.density * jnp.sum(tubes.area * design.shape.lengths)

    assert compute_mass(design) == pytest.approx(float(expected), rel=1e-14)


def test_the_gradient_survives_the_composition(pipeline, params, three_cases):
    # One reverse pass through three blocks, against a difference of two masses.
    # Nothing vouches for it but the forward pass, which is the point: the
    # composition is not being compared with another implementation of itself.
    def composed(q, blocks, loads):
        design = blocks(DesignParameters(q, params.diameters), loads)
        return compute_mass(design_envelope(design, SHARPNESS))

    objective = eqx.filter_jit(composed)
    gradient = eqx.filter_jit(jax.grad(composed))(
        params.force_densities, pipeline, three_cases
    )
    scale = float(jnp.max(jnp.abs(gradient)))
    q = params.force_densities

    assert jnp.all(jnp.isfinite(gradient))

    for member in (0, NUM_EDGES // 2, NUM_EDGES - 1):
        step = abs(float(q[member])) * STEP
        plus = objective(q.at[member].add(step), pipeline, three_cases)
        minus = objective(q.at[member].add(-step), pipeline, three_cases)
        difference = float((plus - minus) / (2.0 * step))

        assert abs(float(gradient[member]) - difference) / scale < TOLERANCE_GRADIENT


def test_the_pipeline_compiles_under_jit(pipeline, params, three_cases):
    """The compiled blocks cross a trace as a pytree, with nothing rebuilt."""

    def composed(blocks, design, loads):
        return compute_mass(design_envelope(blocks(design, loads), SHARPNESS))

    traced = eqx.filter_jit(composed)(pipeline, params, three_cases)
    eager = composed(pipeline, params, three_cases)

    assert float(traced) == pytest.approx(float(eager), rel=1e-12)


def test_design_actions_reduce_the_two_ends(pipeline, params, three_cases):
    """The design moment is the larger end, which is Table B.3 read by the check."""
    shape = pipeline.formfinder(params.force_densities, three_cases.formfinding)
    forces = pipeline.analyzer(shape.xyz, params.diameters, three_cases.analysis)
    acting = design_actions(select_load_case(forces, 1))

    largest = jnp.max(jnp.abs(forces.moment_major[1]), axis=1)

    assert jnp.array_equal(acting.axial_force, forces.axial_force[1])
    assert jnp.allclose(jnp.abs(acting.moment_major), largest)


def test_the_staggered_search_closes_its_coupling(staggered, pipeline, three_cases):
    # The claim the routine exists to make: the answer is a design analyzed at
    # its own sections, so the mass reported is one the structure really has.
    settled = staggered.found.aux.sizes.sections.diameter
    answer = DesignParameters(staggered.found.trajectory.q[-1], settled)
    weighed = design_envelope(pipeline(answer, three_cases), SHARPNESS)
    demanded = weighed.sizes.sections.diameter

    assert float(jnp.max(jnp.abs(demanded / settled - 1.0))) < TOLERANCE_SETTLING


def test_the_staggered_search_traces_the_objective_twice(staggered):
    # Once for the descent's value and gradient, once for the value alone the
    # settling passes need. Both take the diameters as an argument rather than
    # capturing them, so neither is retraced however many rounds the coupling
    # takes; capturing them would compile all three blocks per round.
    assert round_seams(staggered.found.trajectory)
    assert staggered.traces == 2


def test_a_round_starts_where_the_last_one_stopped(staggered):
    """
    A seam is one point weighed under two seeds.

    The force densities repeat, so the round was warm-started; the mass moves
    anyway, because only the seed changed between the two rows. Both together are
    what a refresh between descents means, and the first seam is where the seed
    moves furthest.
    """
    walked = staggered.found.trajectory
    seams = round_seams(walked)

    assert seams

    first = seams[0]
    before = float(walked.mass[first])
    after = float(walked.mass[first + 1])

    assert abs(after / before - 1.0) > TOLERANCE_ROUNDOFF


def test_the_staggered_search_reports_the_value_it_ends_on(staggered):
    """The answer is the last row of the concatenated trajectory."""
    walked = staggered.found.trajectory

    assert float(staggered.found.value) == float(walked.mass[-1])


def test_the_staggered_search_refuses_a_coupling_that_has_not_settled(
    pipeline,
    params,
    three_cases,
):
    # A mass computed at diameters the design does not have is not returned, so
    # a successful return is itself the evidence that the coupling closed. One
    # round cannot close it, the seed being the wrong sections by construction.
    weighed = mass_objective(pipeline, three_cases)

    with pytest.raises(ValueError, match="still moving"):
        optimize_staggered(
            weighed,
            params,
            bounds=BOUNDS,
            iterations=STAGGERED_ITERATIONS,
            rounds=1,
        )


def test_the_two_caps_are_refused_separately(pipeline, params, three_cases):
    # The message says which loop stalled, a round of the search or a pass of the
    # settling inside one, so a budget that ran out names the budget to raise.
    weighed = mass_objective(pipeline, three_cases)

    with pytest.raises(ValueError, match="passes at fixed force densities"):
        optimize_staggered(
            weighed,
            params,
            bounds=BOUNDS,
            iterations=1,
            settling_passes=1,
        )


def test_the_frozen_seed_misreports_the_mass(staggered, pipeline, three_cases):
    """
    Weighing the answer at the seed instead reports a mass it does not have.

    The direction is not asserted, because it is not signable: a member analyzed
    fatter than it is attracts force, so a seed that is uniform where the design
    is not can land either side of it however thin it is.
    """
    answer = staggered.found.trajectory.q[-1]
    frozen = DesignParameters(answer, jnp.full(NUM_EDGES, SEED))
    weighed = design_envelope(pipeline(frozen, three_cases), SHARPNESS)
    reported = float(compute_mass(weighed))

    assert abs(reported / float(staggered.found.value) - 1.0) > TOLERANCE_ROUNDOFF
