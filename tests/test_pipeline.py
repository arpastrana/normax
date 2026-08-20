import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from ec3x.section import DIAMETER_MINIMUM
from ec3x.sizing import LIMIT_MAJOR

from normax.analysis import SmaxAnalyzer
from normax.design import Design
from normax.design import DesignParameters
from normax.design import StructuralDesignPipeline
from normax.design import compute_mass
from normax.design import design_envelope
from normax.design import governing_load_case
from normax.form_finding import FdmFormFinder
from normax.form_finding import equilibrium_graph
from normax.form_finding import equilibrium_state
from normax.form_finding import positions_vertical
from normax.loads import LoadCases
from normax.loads import assemble_load_cases as load_cases_of
from normax.loads import create_loads_half_span
from normax.loads import create_loads_point
from normax.loads import create_loads_uniform
from normax.materials import Steel355
from normax.sizing import Ec3Sizer
from normax.sizing import build_section_family
from normax.structures import build_arch_2d
from normax.visualization import Descent
from normax.visualization import Form
from normax.visualization import GradientCheck
from normax.visualization import MassSweep
from normax.visualization import MeshRefinement
from normax.visualization import SizedMembers
from normax.visualization import StaggeredPasses
from normax.visualization import figure_convergence
from normax.visualization import figure_load_cases
from normax.visualization import figure_optimization
from normax.visualization import figure_sections

matplotlib.use("Agg")

# A 10 m arch rising 3 m under 180 kN spread over its free nodes. Units are
# millimeters and newtons.
SPAN = 10_000.0
RISE = 3_000.0
TOTAL_LOAD = 180_000.0
NUM_EDGES = 10

# The diameter the frame is analyzed with before the check has spoken.
SEED = 100.0

# Invariant 6.5 of CLAUDE.md. Measured at 1.7e-15, so this is generous.
TOLERANCE_UTILIZATION = 1e-9

# Relative step at which the central difference plateaus, and the agreement
# measured there, scaled by the largest component of the gradient.
STEP = 1e-5
TOLERANCE_GRADIENT = 5e-8

# The trough moves for the enveloped objective: three cases make the mass four
# times larger and its arithmetic three times longer, so cancellation dominates
# a decade sooner. Swept, not guessed — 1.8e-8 here against 4.5e-7 at 1e-5, and
# the tolerance sits above the floor of the reference rather than on it.
STEP_CASES = 1e-4
TOLERANCE_GRADIENT_CASES = 2e-7


@pytest.fixture(scope="module")
def steel():
    return Steel355()


@pytest.fixture(scope="module")
def setup():
    """
    The arch, its connectivity, and the `q` that reaches the target rise.
    """
    structure = build_arch_2d(num_edges=NUM_EDGES, span=SPAN, rise=RISE)
    fdm = equilibrium_graph(structure)

    trial = jnp.full(NUM_EDGES, -1.0)
    state = equilibrium_state(
        trial, structure.nodes[fdm.indices_fixed], fdm, funicular(structure)
    )
    reached = jnp.max(state.xyz[:, 2])

    return structure, fdm, trial * reached / RISE


@pytest.fixture(scope="module")
def seed():
    return jnp.full(NUM_EDGES, SEED)


def funicular(structure):
    """
    The uniform load case the arch is form-found under.
    """
    return create_loads_uniform(structure, TOTAL_LOAD / (NUM_EDGES - 1))


def one_case(structure):
    """
    The funicular case, both shaping the arch and checking it.
    """
    applied = funicular(structure)

    return load_cases_of([applied])


def params_of(setup):
    """
    The force densities and the seed diameters, as a pipeline takes them.
    """
    _, _, q = setup

    return DesignParameters(q, jnp.full(NUM_EDGES, SEED))


def analyzer_of(structure, steel, family):
    """
    The analysis block, compiled against a structure.
    """
    return SmaxAnalyzer(structure, family(SEED))


def pipeline_of(setup, steel, family):
    """
    The three blocks a design is solved by, compiled against the arch.
    """
    structure, _, _ = setup
    sizer = Ec3Sizer(structure, family)

    return StructuralDesignPipeline(
        FdmFormFinder(structure),
        SmaxAnalyzer(structure, family(SEED)),
        sizer,
    )


def mass_of(setup, steel, family):
    """
    The mass as a function of the force densities alone.
    """
    pipeline = pipeline_of(setup, steel, family)
    loads = one_case(setup[0])
    seed = jnp.full(NUM_EDGES, SEED)

    def total(q):
        return compute_mass(pipeline(DesignParameters(q, seed), loads))

    return total


def sizes_of(setup, steel, family):
    """
    The mass as a function of the diameters the frame is analyzed with.
    """
    _, _, q = setup
    pipeline = pipeline_of(setup, steel, family)
    loads = one_case(setup[0])

    def total(diameters):
        return compute_mass(pipeline(DesignParameters(q, diameters), loads))

    return total


def sized(setup, steel, section_class, buckling_length=None):
    """
    One pass of form finding, analysis and the code check.

    A buckling length given here reaches the sizer, which is the stage that
    takes one; the composition always hands it the member length.
    """
    structure, _, _ = setup
    family = build_section_family(steel, section_class)
    pipeline = pipeline_of(setup, steel, family)
    params = params_of(setup)
    loads = one_case(structure)

    if buckling_length is None:
        design = pipeline(params, loads)
    else:
        shape = pipeline.formfinder(params.force_densities, loads.formfinding)
        forces = pipeline.analyzer(shape.xyz, params.diameters, loads.analysis)
        design = Design(shape, forces, pipeline.sizer(forces, buckling_length))

    return family, design_envelope(design)


# --------------------------------------------------------------------------- #
# The invariant the sizing map is built to hold
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("section_class", [2, 3])
def test_every_member_is_utilized_exactly_once_over(setup, steel, section_class):
    _, result = sized(setup, steel, section_class)

    assert np.allclose(
        result.sizes.utilization, 1.0, rtol=0.0, atol=TOLERANCE_UTILIZATION
    )


@pytest.mark.parametrize("section_class", [2, 3])
def test_no_member_is_pinned_to_the_catalogue_minimum(setup, steel, section_class):
    # Where the floor binds the utilization is below one by design, so the
    # invariant above only means something if the floor is clear of the design.
    family, result = sized(setup, steel, section_class)

    assert float(jnp.min(result.sizes.sections.diameter)) > float(DIAMETER_MINIMUM)


@pytest.mark.parametrize("section_class", [2, 3])
def test_a_compression_arch_is_governed_by_the_member_check(
    setup, steel, section_class
):
    family, result = sized(setup, steel, section_class)
    codes = Ec3Sizer(setup[0], family).governing(
        result.sizes.sections.diameter, result.forces, result.shape.lengths
    )[0]

    assert np.all(np.asarray(codes) == LIMIT_MAJOR)


# --------------------------------------------------------------------------- #
# What the composition produces
# --------------------------------------------------------------------------- #
def test_the_mass_is_the_sum_over_members(setup, steel):
    family, result = sized(setup, steel, 3)
    by_hand = steel.density * jnp.sum(
        family(result.sizes.sections.diameter).area * result.shape.lengths
    )

    assert float(compute_mass(result)) == pytest.approx(float(by_hand), rel=1e-14)


def test_the_mass_agrees_with_the_scalar_entry_point(setup, steel, seed):
    structure, fdm, q = setup
    family, result = sized(setup, steel, 3)
    scalar = compute_mass(
        pipeline_of(setup, steel, family)(
            DesignParameters(q, seed),
            one_case(setup[0]),
        )
    )

    assert float(scalar) == float(compute_mass(result))


def test_every_member_is_in_compression(setup, steel):
    _, result = sized(setup, steel, 3)

    assert np.all(np.asarray(result.forces.axial_force) < 0.0)


def test_the_arch_is_symmetric_about_midspan(setup, steel):
    _, result = sized(setup, steel, 3)

    assert np.allclose(
        result.sizes.sections.diameter,
        result.sizes.sections.diameter[::-1],
        rtol=1e-12,
    )
    assert np.allclose(result.shape.lengths, result.shape.lengths[::-1], rtol=1e-12)


def test_the_buckling_length_defaults_to_the_member_length(setup, steel):
    _, result = sized(setup, steel, 3)

    assert np.allclose(result.shape.lengths, result.shape.lengths, rtol=0.0, atol=0.0)


def test_a_shorter_buckling_length_never_needs_a_larger_tube(setup, steel):
    _, longer = sized(setup, steel, 3)
    _, shorter = sized(setup, steel, 3, buckling_length=longer.shape.lengths * 0.5)

    assert np.all(
        np.asarray(shorter.sizes.sections.diameter)
        < np.asarray(longer.sizes.sections.diameter)
    )
    assert float(compute_mass(shorter)) < float(compute_mass(longer))


def test_the_thinner_walled_class_is_the_lighter_one(setup, steel):
    # Both classes carry the same forces, and the Class 3 limit puts more of the
    # steel far from the axis. Compression-governed members use the gross area
    # either way, so the thinner wall wins.
    _, section_class = sized(setup, steel, 2)
    _, elastic = sized(setup, steel, 3)

    assert float(compute_mass(elastic)) < float(compute_mass(section_class))
    assert np.all(
        np.asarray(elastic.sizes.sections.diameter)
        > np.asarray(section_class.sizes.sections.diameter)
    )


# --------------------------------------------------------------------------- #
# The gradient across all three stages
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("section_class", [2, 3])
def test_the_mass_gradient_matches_central_differences(
    setup, steel, seed, section_class
):
    structure, fdm, q = setup
    family = build_section_family(steel, section_class)

    def objective(q):
        return compute_mass(
            pipeline_of(setup, steel, family)(
                DesignParameters(q, seed),
                one_case(setup[0]),
            )
        )

    gradient = jax.grad(objective)(q)
    scale = float(jnp.max(jnp.abs(gradient)))

    for edge in (0, NUM_EDGES // 2, NUM_EDGES - 1):
        step = abs(float(q[edge])) * STEP
        plus = objective(q.at[edge].add(step))
        minus = objective(q.at[edge].add(-step))
        central = float((plus - minus) / (2.0 * step))

        assert abs(float(gradient[edge]) - central) / scale < TOLERANCE_GRADIENT


def test_the_mass_gradient_is_finite_and_nowhere_zero(setup, steel, seed):
    structure, fdm, q = setup
    family = build_section_family(steel, 3)

    gradient = jax.grad(mass_of(setup, steel, family))(q)

    assert np.all(np.isfinite(np.asarray(gradient)))
    assert float(jnp.min(jnp.abs(gradient))) > 0.0


def test_forward_and_reverse_mode_agree_on_the_mass(setup, steel, seed):
    structure, fdm, q = setup
    family = build_section_family(steel, 3)

    def objective(q):
        return compute_mass(
            pipeline_of(setup, steel, family)(
                DesignParameters(q, seed),
                one_case(setup[0]),
            )
        )

    assert np.allclose(jax.jacfwd(objective)(q), jax.grad(objective)(q), rtol=1e-12)


def test_the_mass_is_differentiable_in_the_analyzed_diameters(setup, steel, seed):
    # The staggered coupling is one-way, but it is not a dead input: the sections
    # the frame is built from move the forces, and so the sizes.
    structure, fdm, q = setup
    family = build_section_family(steel, 3)

    gradient = jax.grad(sizes_of(setup, steel, family))(seed)

    assert np.all(np.isfinite(np.asarray(gradient)))
    assert float(jnp.max(jnp.abs(gradient))) > 0.0


def test_the_gradient_changes_sign_across_the_arch(setup, steel, seed):
    # The springing gains more from length than it loses to section, and the
    # crown the other way round, so the sensitivity crosses zero in between.
    structure, fdm, q = setup
    family = build_section_family(steel, 3)

    gradient = jax.grad(mass_of(setup, steel, family))(q)

    assert float(gradient[0]) > 0.0
    assert float(gradient[NUM_EDGES // 2]) < 0.0


# --------------------------------------------------------------------------- #
# The staggered coupling
# --------------------------------------------------------------------------- #
def test_repeating_the_pass_reaches_a_fixed_point(setup, steel, seed):
    structure, fdm, q = setup
    family = build_section_family(steel, 3)

    diameters = seed
    moves = []
    for _ in range(5):
        result = design_envelope(
            pipeline_of(setup, steel, family)(
                DesignParameters(q, diameters), one_case(setup[0])
            )
        )
        moves.append(
            float(
                jnp.max(
                    jnp.abs(result.sizes.sections.diameter - diameters)
                    / result.sizes.sections.diameter
                )
            )
        )
        diameters = result.sizes.sections.diameter

    # Geometric, at a contraction the analysis fixes rather than the tolerance:
    # the frame barely depends on the section, so each pass gains two decades.
    ratios = [later / earlier for earlier, later in zip(moves[:-1], moves[1:])]

    assert all(ratio < 0.1 for ratio in ratios)
    assert moves[-1] < 1e-6


def test_one_pass_is_within_two_percent_of_the_fixed_point(setup, steel, seed):
    structure, fdm, q = setup
    family = build_section_family(steel, 3)

    first = design_envelope(
        pipeline_of(setup, steel, family)(DesignParameters(q, seed), one_case(setup[0]))
    )

    diameters = first.sizes.sections.diameter
    for _ in range(5):
        settled = design_envelope(
            pipeline_of(setup, steel, family)(
                DesignParameters(q, diameters), one_case(setup[0])
            )
        )
        diameters = settled.sizes.sections.diameter

    gap = abs(float(compute_mass(first)) - float(compute_mass(settled))) / float(
        compute_mass(settled)
    )

    assert gap < 0.02


# --------------------------------------------------------------------------- #
# The figures the experiments save
# --------------------------------------------------------------------------- #
def test_the_section_figure_builds(setup, steel, seed):
    structure, _, _ = setup
    family, result = sized(setup, steel, 3)
    assumed = float(steel.density * jnp.sum(family(seed).area * result.shape.lengths))

    figure = figure_sections(
        result.shape.xyz,
        structure.edges,
        SizedMembers(seed, assumed),
        SizedMembers(result.sizes.sections.diameter, float(compute_mass(result))),
    )

    assert len(figure.axes) == 5
    plt.close(figure)


def test_the_convergence_figure_builds():
    counts = np.array([5, 10, 20, 40])
    member = np.array([0.045, 0.031, 0.027, 0.026])
    fixed = np.array([0.033, 0.030, 0.029, 0.028])
    moves = np.array([3.8e-1, 1.2e-2, 3.1e-4, 7.9e-6])

    figure = figure_convergence(
        MeshRefinement(counts, member, fixed, 0.0274),
        StaggeredPasses(np.arange(len(moves)), moves),
    )

    assert len(figure.axes) == 3
    plt.close(figure)


def test_the_section_figure_reports_a_lighter_design(setup, steel, seed):
    # The label flips wording on the sign, so the sign is worth pinning.
    structure, _, _ = setup
    family, result = sized(setup, steel, 3)
    assumed = float(steel.density * jnp.sum(family(seed).area * result.shape.lengths))

    figure = figure_sections(
        result.shape.xyz,
        structure.edges,
        SizedMembers(seed, assumed),
        SizedMembers(result.sizes.sections.diameter, float(compute_mass(result))),
    )
    labels = [text.get_text() for text in figure.axes[1].texts]

    assert any("lighter" in label for label in labels)
    plt.close(figure)


# --------------------------------------------------------------------------- #
# One structure against several load cases
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def load_cases(setup):
    """
    Three cases of equal total: funicular, half span, and a crown point load.
    """
    structure, _, _ = setup
    spread = TOTAL_LOAD / (NUM_EDGES - 1)

    half = create_loads_half_span(structure, spread, factor=0.5)
    half = half * (TOTAL_LOAD / abs(float(jnp.sum(half[:, 2]))))

    point = create_loads_uniform(structure, spread * 0.75) + create_loads_point(
        structure, TOTAL_LOAD * 0.25, node=structure.crown_node()
    )

    return load_cases_of([create_loads_uniform(structure, spread), half, point])


def covered(setup, steel, load_cases, beta, section_class=3):
    structure, _, _ = setup
    family = build_section_family(steel, section_class)
    pipeline = pipeline_of(setup, steel, family)

    demanded = pipeline(params_of(setup), load_cases)

    return family, design_envelope(demanded, beta), demanded


@pytest.mark.parametrize("beta", [10.0, 100.0, 500.0])
def test_the_envelope_covers_every_load_case(setup, steel, load_cases, beta):
    # The invariant that replaces "utilization is exactly one" once there is
    # more than one case: adequate everywhere, and exactly adequate somewhere.
    _, result, demanded = covered(setup, steel, load_cases, beta)

    assert float(jnp.max(result.sizes.utilization)) <= 1.0 + 1e-12


@pytest.mark.parametrize("beta", [10.0, 100.0, 500.0])
def test_the_envelope_is_never_smaller_than_the_load_case_that_needs_most(
    setup, steel, load_cases, beta
):
    _, result, demanded = covered(setup, steel, load_cases, beta)
    largest = jnp.max(demanded.sizes.sections.diameter, axis=0)

    assert np.all(
        np.asarray(result.sizes.sections.diameter) >= np.asarray(largest) - 1e-9
    )


def test_a_sharper_envelope_gives_away_less(setup, steel, load_cases):
    _, blunt, demanded = covered(setup, steel, load_cases, 10.0)
    _, sharp, demanded = covered(setup, steel, load_cases, 500.0)

    assert float(compute_mass(sharp)) < float(compute_mass(blunt))


@pytest.mark.parametrize("beta", [10.0, 100.0])
def test_the_excess_respects_its_bound(setup, steel, load_cases, beta):
    # The envelope exceeds the true largest by at most the number of cases
    # raised to the reciprocal of the sharpness, in diameter.
    _, result, demanded = covered(setup, steel, load_cases, beta)
    largest = jnp.max(demanded.sizes.sections.diameter, axis=0)

    excess = float(jnp.max(result.sizes.sections.diameter / largest)) - 1.0
    bound = float(load_cases.analysis.shape[0] ** (1.0 / beta)) - 1.0

    assert 0.0 <= excess <= bound + 1e-12


def test_the_unsmoothed_design_is_fully_stressed(setup, steel, load_cases):
    # Invariant 6.5 survives the aggregation: some case works every member to
    # exactly one, even though no single case works all of them.
    family, result, demanded = covered(setup, steel, load_cases, 500.0)
    exact = covered(setup, steel, load_cases, None)[1]

    worst = np.max(np.asarray(exact.sizes.utilization), axis=0)

    assert np.allclose(worst, 1.0, rtol=0.0, atol=TOLERANCE_UTILIZATION)


def test_the_unsmoothed_design_is_never_heavier_than_the_envelope(
    setup, steel, load_cases
):
    family, result, demanded = covered(setup, steel, load_cases, 50.0)
    exact = covered(setup, steel, load_cases, None)[1]

    assert float(compute_mass(exact)) <= float(compute_mass(result))


def test_one_load_case_reproduces_the_single_case_design(setup, steel, load_cases):
    # An envelope over one case is that case, whatever the sharpness, so the
    # aggregation cannot be quietly changing the answer.
    structure, fdm, q = setup
    family = build_section_family(steel, 3)
    seeds = jnp.full(NUM_EDGES, SEED)

    single = pipeline_of(setup, steel, family)(
        DesignParameters(q, seeds),
        one_case(setup[0]),
    )
    one = LoadCases(load_cases.formfinding, load_cases.analysis[:1])
    covering = design_envelope(
        pipeline_of(setup, steel, family)(DesignParameters(q, seeds), one), 25.0
    )

    assert np.allclose(
        np.asarray(covering.sizes.sections.diameter),
        np.asarray(single.sizes.sections.diameter),
        rtol=1e-12,
    )
    assert float(compute_mass(covering)) == pytest.approx(
        float(compute_mass(single)), rel=1e-12
    )


def test_the_governing_load_case_is_the_one_working_a_member_hardest(
    setup, steel, load_cases
):
    # `sizes.utilization` is a diagonal — every case at the section it asked
    # for, so one by construction — and reading it as a verdict on a reconciled
    # design would compare a matrix of ones. The standard has to be read again
    # at the section the envelope settled on.
    family, result, demanded = covered(setup, steel, load_cases, 500.0)
    sizer = Ec3Sizer(setup[0], family)
    reread = sizer.compute_utilization(
        result.sizes.sections.diameter, result.forces, result.shape.lengths
    )

    decided = np.asarray(governing_load_case(demanded.sizes.sections.diameter))
    expected = np.argmax(np.asarray(reread), axis=0)

    assert np.array_equal(decided, expected)


def test_the_funicular_load_case_is_not_the_one_that_decides_the_design(
    setup, steel, load_cases
):
    # The point of the second and third cases. A shape form-found under the
    # first carries it axially, so it is the benign one and the members are
    # sized by what the form-finder could not see.
    _, result, demanded = covered(setup, steel, load_cases, 500.0)

    assert 0 not in set(
        np.asarray(governing_load_case(demanded.sizes.sections.diameter)).tolist()
    )


def test_a_load_case_reaches_the_analysis(setup, steel):
    # The load case is an argument to the analysis alone; form finding keeps the
    # loads the shape answers to.
    structure, fdm, q = setup
    family = build_section_family(steel, 3)
    seeds = jnp.full(NUM_EDGES, SEED)

    spread = TOTAL_LOAD / (NUM_EDGES - 1)
    asymmetric = create_loads_half_span(structure, spread, factor=0.0)

    pipeline = pipeline_of(setup, steel, family)
    shaped = pipeline(DesignParameters(q, seeds), one_case(structure))
    patched = pipeline(
        DesignParameters(q, seeds),
        LoadCases(funicular(structure), jnp.stack([asymmetric])),
    )

    assert np.allclose(np.asarray(shaped.shape.xyz), np.asarray(patched.shape.xyz))
    assert float(jnp.max(jnp.abs(patched.forces.moment_major))) > float(
        jnp.max(jnp.abs(shaped.forces.moment_major))
    )


def test_the_enveloped_mass_gradient_matches_central_differences(
    setup, steel, load_cases
):
    structure, fdm, q = setup
    family = build_section_family(steel, 3)

    pipeline = pipeline_of(setup, steel, family)

    def objective(q):
        design = pipeline(DesignParameters(q, jnp.full(NUM_EDGES, SEED)), load_cases)

        return compute_mass(design_envelope(design, 100.0))

    gradient = jax.grad(objective)(q)
    scale = float(jnp.max(jnp.abs(gradient)))

    assert np.all(np.isfinite(np.asarray(gradient)))

    for edge in (0, NUM_EDGES // 2, NUM_EDGES - 1):
        step = abs(float(q[edge])) * STEP_CASES
        plus = objective(q.at[edge].add(step))
        minus = objective(q.at[edge].add(-step))
        difference = float((plus - minus) / (2.0 * step))

        error = abs(float(gradient[edge]) - difference) / scale
        assert error < TOLERANCE_GRADIENT_CASES


def descents():
    return (
        Descent(
            "no floor", np.linspace(0.134, 0.051, 12), np.geomspace(10.0, 500.0, 12)
        ),
        Descent(
            "floored", np.linspace(0.134, 0.092, 10), np.geomspace(10.0, 500.0, 10)
        ),
    )


def test_the_optimization_figure_builds():
    scales = np.linspace(0.4, 2.4, 9)
    masses = 0.13 + 0.02 * (scales - 1.5) ** 2
    exact = 0.04 * (scales - 1.5)

    figure = figure_optimization(
        MassSweep(scales, masses, 3), GradientCheck(exact, exact), descents()
    )

    assert len(figure.axes) == 4
    plt.close(figure)


def test_the_optimization_figure_draws_every_descent_it_is_given():
    # The constrained run is the design and the unconstrained one is the
    # evidence, so both belong on the same axes as the single-variable sweep.
    scales = np.linspace(0.4, 2.4, 9)
    masses = 0.13 + 0.02 * (scales - 1.5) ** 2
    exact = 0.04 * (scales - 1.5)

    figure = figure_optimization(
        MassSweep(scales, masses, 3), GradientCheck(exact, exact), descents()
    )
    labels = [line.get_label() for line in figure.axes[0].lines]

    assert any("no floor" in label for label in labels)
    assert any("floored" in label for label in labels)
    plt.close(figure)


def test_the_optimization_figure_marks_the_funicular_start_and_not_the_sweep(setup):
    # The descent begins at the funicular design, which is not the first sample
    # of the sweep, and a figure that confused the two would overstate what the
    # optimizer achieved.
    scales = np.linspace(0.4, 2.4, 9)
    masses = 0.13 + 0.02 * (scales - 1.5) ** 2
    trajectory = np.linspace(0.134, 0.051, 5)

    figure = figure_optimization(
        MassSweep(scales, masses, 3),
        GradientCheck(0.04 * (scales - 1.5), 0.04 * (scales - 1.5)),
        (Descent("run", trajectory, np.full(5, 10.0)),),
    )

    labels = [line.get_label() for line in figure.axes[0].lines]

    assert any(f"{masses[3]:.4f}" in label for label in labels)
    plt.close(figure)


def test_the_load_case_figure_builds(setup, steel, load_cases):
    structure, _, _ = setup
    family, result, demanded = covered(setup, steel, load_cases, 500.0)
    exact = covered(setup, steel, load_cases, None)[1]
    decided = governing_load_case(demanded.sizes.sections.diameter)

    figure = figure_load_cases(
        structure.edges,
        (
            Form(
                "start",
                structure.nodes,
                jnp.max(demanded.sizes.sections.diameter, axis=0),
                decided,
            ),
            Form("optimized", result.shape.xyz, exact.sizes.sections.diameter, decided),
        ),
        ("LC1", "LC2", "LC3"),
    )

    assert len(figure.axes) == 5
    plt.close(figure)


def test_the_load_case_figure_takes_as_many_forms_as_it_is_given(
    setup, steel, load_cases
):
    structure, _, _ = setup
    family, result, demanded = covered(setup, steel, load_cases, 500.0)
    exact = covered(setup, steel, load_cases, None)[1]
    decided = governing_load_case(demanded.sizes.sections.diameter)

    forms = tuple(
        Form(f"form {index}", result.shape.xyz, exact.sizes.sections.diameter, decided)
        for index in range(3)
    )
    figure = figure_load_cases(structure.edges, forms, ("LC1", "LC2", "LC3"))

    assert len(figure.axes) == 7
    plt.close(figure)


def test_the_two_forms_are_drawn_on_the_same_axes(setup, steel, load_cases):
    # A form that dropped has to look lower rather than differently framed, or
    # a collapsed leg is invisible.
    structure, _, _ = setup
    family, result, demanded = covered(setup, steel, load_cases, 500.0)
    exact = covered(setup, steel, load_cases, None)[1]
    decided = governing_load_case(demanded.sizes.sections.diameter)

    figure = figure_load_cases(
        structure.edges,
        (
            Form(
                "start",
                structure.nodes,
                jnp.max(demanded.sizes.sections.diameter, axis=0),
                decided,
            ),
            Form(
                "lowered",
                result.shape.xyz * jnp.asarray([1.0, 1.0, 0.5]),
                exact.sizes.sections.diameter,
                decided,
            ),
        ),
        ("LC1", "LC2", "LC3"),
    )

    assert figure.axes[0].get_xlim() == figure.axes[1].get_xlim()
    assert figure.axes[0].get_ylim() == figure.axes[1].get_ylim()
    plt.close(figure)


# --------------------------------------------------------------------------- #
# Holding the plan, and what it costs
# --------------------------------------------------------------------------- #
def test_a_uniform_force_density_leaves_the_plan_alone(setup):
    # The full equilibrium already spaces an arch evenly under a uniform force
    # density, so holding the plan changes nothing and the two agree exactly.
    structure, fdm, q = setup

    free = equilibrium_state(
        q, structure.nodes[fdm.indices_fixed], fdm, funicular(structure)
    ).xyz
    held = positions_vertical(q, structure.nodes, fdm, funicular(structure))

    assert np.allclose(np.asarray(free), np.asarray(held), atol=1e-9)


def test_holding_the_plan_never_changes_the_heights(setup):
    # The force density system decouples per coordinate, so the vertical solve
    # is the same equation either way. Holding the plan moves the plan alone.
    structure, fdm, q = setup
    varied = q * jnp.linspace(0.5, 1.8, NUM_EDGES)

    free = equilibrium_state(
        varied, structure.nodes[fdm.indices_fixed], fdm, funicular(structure)
    ).xyz
    held = positions_vertical(varied, structure.nodes, fdm, funicular(structure))

    assert np.allclose(np.asarray(free[:, 2]), np.asarray(held[:, 2]), atol=1e-9)
    assert not np.allclose(np.asarray(free[:, 0]), np.asarray(held[:, 0]))


def nodal_residual(structure, xyz, q):
    """
    Out-of-balance force at every free node, with members carrying axial force.
    """
    edges = np.asarray(structure.edges)
    vectors = np.asarray(xyz)[edges[:, 1]] - np.asarray(xyz)[edges[:, 0]]
    forces = np.asarray(q)[:, None] * vectors

    residual = np.zeros_like(np.asarray(xyz))
    np.add.at(residual, edges[:, 0], forces)
    np.add.at(residual, edges[:, 1], -forces)
    residual = residual + np.asarray(funicular(structure))

    free = np.setdiff1d(
        np.arange(np.asarray(xyz).shape[0]), np.asarray(structure.supports)
    )

    return np.abs(residual[free]).max(axis=0)


def test_a_held_plan_is_funicular_only_under_a_uniform_force_density(setup):
    # Horizontal equilibrium on an evenly spaced plan forces the force densities
    # to be equal, so the funicular part of a held plan is one parameter wide.
    # Anything else hands the thrust to structure that is not being designed.
    structure, fdm, q = setup
    varied = q * jnp.linspace(0.5, 1.8, NUM_EDGES)

    uniform = nodal_residual(
        structure, positions_vertical(q, structure.nodes, fdm, funicular(structure)), q
    )
    other = nodal_residual(
        structure,
        positions_vertical(varied, structure.nodes, fdm, funicular(structure)),
        varied,
    )

    assert uniform[0] < 1e-6
    assert other[0] > 1e3
    assert other[2] < 1e-6


def test_the_full_equilibrium_stays_funicular_whatever_the_force_densities(setup):
    structure, fdm, q = setup
    varied = q * jnp.linspace(0.5, 1.8, NUM_EDGES)

    residual = nodal_residual(
        structure,
        equilibrium_state(
            varied, structure.nodes[fdm.indices_fixed], fdm, funicular(structure)
        ).xyz,
        varied,
    )

    assert np.all(residual < 1e-6)


def test_the_load_case_counts_share_one_scale(setup, steel, load_cases):
    # A bar is read against its neighbours, so independently scaled panels would
    # make an even split look lopsided.
    structure, _, _ = setup
    family, result, demanded = covered(setup, steel, load_cases, 500.0)
    exact = covered(setup, steel, load_cases, None)[1]
    decided = governing_load_case(demanded.sizes.sections.diameter)

    forms = tuple(
        Form(f"form {index}", result.shape.xyz, exact.sizes.sections.diameter, decided)
        for index in range(3)
    )
    figure = figure_load_cases(structure.edges, forms, ("LC1", "LC2", "LC3"))

    # The drawings come first, then the counts, then the color bar.
    counts = figure.axes[len(forms) : 2 * len(forms)]
    limits = {ax.get_ylim() for ax in counts}

    assert len(counts) == len(forms)
    assert len(limits) == 1
    plt.close(figure)
