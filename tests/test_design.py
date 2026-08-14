import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from normax.analysis.smax import SmaxAnalyzer
from normax.design import DesignParameters
from normax.design import DesignPipeline
from normax.design import calculate_mass
from normax.design import load_cases as load_cases_of
from normax.ec3.material import Steel
from normax.ec3.section import TubeCatalogue
from normax.form_finding.fdm import FdmFormFinder
from normax.form_finding.fdm import equilibrium_graph
from normax.form_finding.fdm import equilibrium_state
from normax.sizing import Ec3Sizer
from normax.sizing import design_actions
from normax.structures import arch_2d
from normax.structures import loads_half_span
from normax.structures import loads_uniform

# A 10 m arch rising 3 m under 180 kN spread over its free nodes. Units are
# millimeters and newtons.
SPAN = 10_000.0
RISE = 3_000.0
TOTAL_LOAD = 180_000.0
NUM_EDGES = 10

# The arch lies in the XZ plane, so it has no thickness along Y.
NORMAL = 1

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


@pytest.fixture(scope="module")
def steel():
    return Steel()


@pytest.fixture(scope="module")
def catalogue(steel):
    return TubeCatalogue.at_class_limit(steel.f_y, 3)


@pytest.fixture(scope="module")
def structure():
    return arch_2d(num_edges=NUM_EDGES, span=SPAN, rise=RISE)


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
    blocks = DesignPipeline(
        FdmFormFinder(),
        SmaxAnalyzer(steel, catalogue, NORMAL),
        Ec3Sizer(steel, catalogue),
    )
    return blocks.compile(structure)


@pytest.fixture(scope="module")
def params(force_densities):
    return DesignParameters(force_densities, jnp.full(NUM_EDGES, SEED))


@pytest.fixture(scope="module")
def one_case(structure):
    applied = funicular(structure)

    return load_cases_of(applied, [applied])


@pytest.fixture(scope="module")
def three_cases(structure):
    load = TOTAL_LOAD / (NUM_EDGES - 1)
    cases = [
        loads_uniform(structure, load),
        loads_half_span(structure, load, factor=0.25),
        loads_half_span(structure, load, factor=0.25, mirrored=True),
    ]

    return load_cases_of(funicular(structure), cases)


def test_sizer_reads_its_class_off_its_family(steel):
    """The class is derived from the family and never accepted beside it."""
    for section_class in (1, 2, 3):
        catalogue = TubeCatalogue.at_class_limit(steel.f_y, section_class)
        assert Ec3Sizer(steel, catalogue).section_class == section_class


def test_sizer_refuses_a_class_four_family(steel):
    """A family too slender to be checked by these clauses is refused at build."""
    with pytest.raises(ValueError):
        Ec3Sizer(steel, TubeCatalogue(ratio=200.0))


def test_uncompiled_blocks_refuse_to_run(steel, catalogue, params, one_case):
    """A block that was never given a structure says so rather than failing late."""
    with pytest.raises(ValueError):
        FdmFormFinder()(params.q, one_case.formfinding)

    with pytest.raises(ValueError):
        blocks = DesignPipeline(
            FdmFormFinder(),
            SmaxAnalyzer(steel, catalogue, NORMAL),
            Ec3Sizer(steel, catalogue),
        )
        blocks(params, one_case)


def test_form_finder_matches_the_free_function(structure, force_densities, one_case):
    """The block wraps the solve and changes nothing about it."""
    graph = equilibrium_graph(structure)
    state = equilibrium_state(
        force_densities,
        structure.nodes[graph.indices_fixed],
        graph,
        one_case.formfinding,
    )
    shape = FdmFormFinder().compile(structure)(force_densities, one_case.formfinding)

    assert jnp.array_equal(shape.xyz, state.xyz)
    assert jnp.array_equal(shape.lengths, state.lengths[:, 0])
    assert jnp.array_equal(shape.forces, state.forces[:, 0])


def test_analyzer_stacks_one_load_case_per_row(pipeline, params, three_cases):
    """Each row of the stacked forces is the analysis of the matching case."""
    shape = pipeline.formfinder(params.q, three_cases.formfinding)
    forces = pipeline.analyzer(shape, params.diameters, three_cases.analysis)

    assert forces.axial_force.shape == (3, NUM_EDGES)
    assert forces.moment_major.shape == (3, NUM_EDGES, 2)

    for load_case in range(3):
        alone = pipeline.analyzer(
            shape, params.diameters, three_cases.analysis[load_case][None]
        )
        assert jnp.array_equal(alone.axial_force[0], forces.axial_force[load_case])
        assert jnp.array_equal(alone.moment_major[0], forces.moment_major[load_case])


def test_repeating_a_load_case_changes_nothing(pipeline, params, one_case):
    # A case listed twice is the same structure asked the same question twice,
    # so every field has to come back identical rather than merely close. Taken
    # at the true largest, since a smooth envelope over two equal sizes is
    # deliberately above both of them.
    once = pipeline(params, one_case)
    twice = pipeline(
        params,
        load_cases_of(one_case.formfinding, [one_case.analysis[0]] * 2),
    )

    assert jnp.array_equal(twice.diameters, once.diameters)
    assert jnp.array_equal(twice.required[0], once.required[0])
    assert jnp.array_equal(twice.required[1], once.required[0])
    assert calculate_mass(twice) == calculate_mass(once)


def test_a_geometry_is_form_found_once_for_every_load_case(
    pipeline, params, one_case, three_cases
):
    # The shape answers to one load case by construction, so adding cases to
    # check against must leave it exactly where it was.
    single = pipeline(params, one_case)
    several = pipeline(params, three_cases, SHARPNESS)

    assert jnp.array_equal(several.xyz, single.xyz)
    assert jnp.array_equal(several.lengths, single.lengths)
    assert jnp.array_equal(several.buckling_length, single.buckling_length)


def test_the_largest_is_the_default_and_the_envelope_bounds_it(
    pipeline, params, three_cases
):
    """The envelope never understates the size any load case demands."""
    largest = pipeline(params, three_cases)
    enveloped = pipeline(params, three_cases, SHARPNESS)

    assert jnp.array_equal(largest.diameters, jnp.max(largest.required, axis=0))
    assert jnp.all(enveloped.diameters >= largest.diameters)
    assert calculate_mass(enveloped) >= calculate_mass(largest)


def test_one_load_case_is_never_enveloped(pipeline, params, one_case):
    """An envelope over one case is the identity, so it is not taken."""
    for sharpness in (None, 1.0, SHARPNESS):
        design = pipeline(params, one_case, sharpness)
        assert jnp.array_equal(design.diameters, design.required[0])


def test_every_member_is_exactly_stressed(pipeline, params, one_case):
    """Invariant 6.5 of CLAUDE.md, through the composed blocks."""
    design = pipeline(params, one_case)

    assert jnp.max(jnp.abs(design.utilization - 1.0)) < 1e-9


def test_the_governing_load_case_is_fully_stressed(pipeline, params, three_cases):
    """A size covers every case, and works exactly for the one that set it."""
    design = pipeline(params, three_cases)
    worst = jnp.max(design.utilization, axis=0)

    assert jnp.all(design.utilization <= 1.0 + 1e-9)
    assert jnp.max(jnp.abs(worst - 1.0)) < 1e-9


def test_the_mass_is_the_sum_of_what_the_members_weigh(pipeline, params, one_case):
    """Mass is geometry: a mass per length times a length, added up."""
    design = pipeline(params, one_case)
    tubes = pipeline.sizer.catalogue.tube_at(design.diameters)
    expected = pipeline.sizer.steel.density * jnp.sum(tubes.area * design.lengths)

    assert calculate_mass(design) == pytest.approx(float(expected), rel=1e-14)


def test_the_gradient_survives_the_composition(pipeline, params, three_cases):
    # One reverse pass through three blocks, against a difference of two masses.
    # Nothing vouches for it but the forward pass, which is the point: the
    # composition is not being compared with another implementation of itself.
    def composed(q, blocks, loads):
        design = blocks(DesignParameters(q, params.diameters), loads, SHARPNESS)
        return calculate_mass(design)

    objective = eqx.filter_jit(composed)
    gradient = eqx.filter_jit(jax.grad(composed))(params.q, pipeline, three_cases)
    scale = float(jnp.max(jnp.abs(gradient)))
    q = params.q

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
        return calculate_mass(blocks(design, loads, SHARPNESS))

    traced = eqx.filter_jit(composed)(pipeline, params, three_cases)
    eager = composed(pipeline, params, three_cases)

    assert float(traced) == pytest.approx(float(eager), rel=1e-12)


def test_design_actions_reduce_the_two_ends(pipeline, params, three_cases):
    """The design moment is the larger end, which is Table B.3 read by the check."""
    shape = pipeline.formfinder(params.q, three_cases.formfinding)
    forces = pipeline.analyzer(shape, params.diameters, three_cases.analysis)
    acting = design_actions(forces, 1)

    largest = jnp.max(jnp.abs(forces.moment_major[1]), axis=1)

    assert jnp.array_equal(acting.axial_force, forces.axial_force[1])
    assert jnp.allclose(jnp.abs(acting.moment_major), largest)
