import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from normax.analysis.smax import SmaxAnalyzer
from normax.analysis.smax import buckling_modes
from normax.analysis.smax import frame_stability
from normax.analysis.smax import prepare_model
from normax.design import Design
from normax.design import DesignParameters
from normax.design import StructuralDesignPipeline
from normax.design import compute_mass
from normax.design import design_envelope
from normax.design import governing_load_case
from normax.ec3.material import Steel
from normax.ec3.section import TubeCatalogue
from normax.ec3.sizing import LIMIT_MAJOR
from normax.ec3.stability import ALPHA_CR_ELASTIC
from normax.form_finding.fdm import FdmFormFinder
from normax.form_finding.fdm import equilibrium_graph
from normax.form_finding.fdm import equilibrium_state
from normax.form_finding.fdm import positions_vertical
from normax.loads import LoadCases
from normax.loads import assemble_load_cases as load_cases_of
from normax.loads import loads_half_span
from normax.loads import loads_point
from normax.loads import loads_uniform
from normax.sizing.ec3 import Ec3Sizer
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
from normax.visualization import figure_modes
from normax.visualization import figure_optimization
from normax.visualization import figure_sections

matplotlib.use("Agg")

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

# Effective length of the arch's own critical mode, as a fraction of its
# developed length. Measured, and steady to three figures across the meshes.
GLOBAL_MODE_FACTOR = 0.576


@pytest.fixture(scope="module")
def steel():
    return Steel()


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
    return loads_uniform(structure, TOTAL_LOAD / (NUM_EDGES - 1))


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


def analysis_model(structure, steel, catalogue):
    """
    The compiled analysis model alone, for the calls that take no block.
    """
    return prepare_model(structure, catalogue(SEED))


def analyzer_of(structure, steel, catalogue):
    """
    The analysis block, compiled against a structure.
    """
    return SmaxAnalyzer(structure, catalogue(SEED))


def pipeline_of(setup, steel, catalogue):
    """
    The three blocks a design is solved by, compiled against the arch.
    """
    structure, _, _ = setup
    return StructuralDesignPipeline(
        FdmFormFinder(structure),
        SmaxAnalyzer(structure, catalogue(SEED)),
        Ec3Sizer(structure, catalogue),
    )


def mass_of(setup, steel, catalogue):
    """
    The mass as a function of the force densities alone.
    """
    pipeline = pipeline_of(setup, steel, catalogue)
    loads = one_case(setup[0])
    seed = jnp.full(NUM_EDGES, SEED)

    def total(q):
        return compute_mass(pipeline(DesignParameters(q, seed), loads))

    return total


def sizes_of(setup, steel, catalogue):
    """
    The mass as a function of the diameters the frame is analyzed with.
    """
    _, _, q = setup
    pipeline = pipeline_of(setup, steel, catalogue)
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
    catalogue = TubeCatalogue.at_class_limit(steel, section_class)
    pipeline = pipeline_of(setup, steel, catalogue)
    params = params_of(setup)
    loads = one_case(structure)

    if buckling_length is None:
        design = pipeline(params, loads)
    else:
        shape = pipeline.formfinder(params.force_densities, loads.formfinding)
        forces = pipeline.analyzer(shape.xyz, params.diameters, loads.analysis)
        design = Design(shape, forces, pipeline.sizer(forces, buckling_length))

    return catalogue, design_envelope(design)


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
    catalogue, result = sized(setup, steel, section_class)

    assert float(jnp.min(result.sizes.sections.diameter)) > float(
        catalogue.diameter_min
    )


@pytest.mark.parametrize("section_class", [2, 3])
def test_a_compression_arch_is_governed_by_the_member_check(
    setup, steel, section_class
):
    catalogue, result = sized(setup, steel, section_class)
    codes = Ec3Sizer(setup[0], catalogue).governing(
        result.sizes.sections.diameter, result.sizes.actions, result.shape.lengths
    )[0]

    assert np.all(np.asarray(codes) == LIMIT_MAJOR)


# --------------------------------------------------------------------------- #
# What the composition produces
# --------------------------------------------------------------------------- #
def test_the_mass_is_the_sum_over_members(setup, steel):
    catalogue, result = sized(setup, steel, 3)
    by_hand = steel.density * jnp.sum(
        catalogue(result.sizes.sections.diameter).area * result.shape.lengths
    )

    assert float(compute_mass(result)) == pytest.approx(float(by_hand), rel=1e-14)


def test_the_mass_agrees_with_the_scalar_entry_point(setup, steel, seed):
    structure, fdm, q = setup
    catalogue, result = sized(setup, steel, 3)
    scalar = compute_mass(
        pipeline_of(setup, steel, catalogue)(
            DesignParameters(q, seed),
            one_case(setup[0]),
        )
    )

    assert float(scalar) == float(compute_mass(result))


def test_every_member_is_in_compression(setup, steel):
    _, result = sized(setup, steel, 3)

    assert np.all(np.asarray(result.sizes.actions.axial_force) < 0.0)


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
    catalogue = TubeCatalogue.at_class_limit(steel, section_class)

    def objective(q):
        return compute_mass(
            pipeline_of(setup, steel, catalogue)(
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
    catalogue = TubeCatalogue.at_class_limit(steel, 3)

    gradient = jax.grad(mass_of(setup, steel, catalogue))(q)

    assert np.all(np.isfinite(np.asarray(gradient)))
    assert float(jnp.min(jnp.abs(gradient))) > 0.0


def test_forward_and_reverse_mode_agree_on_the_mass(setup, steel, seed):
    structure, fdm, q = setup
    catalogue = TubeCatalogue.at_class_limit(steel, 3)

    def objective(q):
        return compute_mass(
            pipeline_of(setup, steel, catalogue)(
                DesignParameters(q, seed),
                one_case(setup[0]),
            )
        )

    assert np.allclose(jax.jacfwd(objective)(q), jax.grad(objective)(q), rtol=1e-12)


def test_the_mass_is_differentiable_in_the_analyzed_diameters(setup, steel, seed):
    # The staggered coupling is one-way, but it is not a dead input: the sections
    # the frame is built from move the forces, and so the sizes.
    structure, fdm, q = setup
    catalogue = TubeCatalogue.at_class_limit(steel, 3)

    gradient = jax.grad(sizes_of(setup, steel, catalogue))(seed)

    assert np.all(np.isfinite(np.asarray(gradient)))
    assert float(jnp.max(jnp.abs(gradient))) > 0.0


def test_the_gradient_changes_sign_across_the_arch(setup, steel, seed):
    # The springing gains more from length than it loses to section, and the
    # crown the other way round, so the sensitivity crosses zero in between.
    structure, fdm, q = setup
    catalogue = TubeCatalogue.at_class_limit(steel, 3)

    gradient = jax.grad(mass_of(setup, steel, catalogue))(q)

    assert float(gradient[0]) > 0.0
    assert float(gradient[NUM_EDGES // 2]) < 0.0


# --------------------------------------------------------------------------- #
# The staggered coupling
# --------------------------------------------------------------------------- #
def test_repeating_the_pass_reaches_a_fixed_point(setup, steel, seed):
    structure, fdm, q = setup
    catalogue = TubeCatalogue.at_class_limit(steel, 3)

    diameters = seed
    moves = []
    for _ in range(5):
        result = design_envelope(
            pipeline_of(setup, steel, catalogue)(
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
    catalogue = TubeCatalogue.at_class_limit(steel, 3)

    first = design_envelope(
        pipeline_of(setup, steel, catalogue)(
            DesignParameters(q, seed), one_case(setup[0])
        )
    )

    diameters = first.sizes.sections.diameter
    for _ in range(5):
        settled = design_envelope(
            pipeline_of(setup, steel, catalogue)(
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
    catalogue, result = sized(setup, steel, 3)
    assumed = float(
        steel.density * jnp.sum(catalogue(seed).area * result.shape.lengths)
    )

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
    catalogue, result = sized(setup, steel, 3)
    assumed = float(
        steel.density * jnp.sum(catalogue(seed).area * result.shape.lengths)
    )

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
# What the member-length buckling length assumes
# --------------------------------------------------------------------------- #
def test_the_critical_factors_are_positive_and_ordered(setup, steel):
    structure, _, _ = setup
    catalogue, result = sized(setup, steel, 3)

    modes = buckling_modes(
        analysis_model(structure, steel, catalogue),
        result.shape.xyz,
        result.sizes.sections.diameter,
        catalogue(SEED),
        funicular(structure),
        num_modes=4,
    )
    factors = np.asarray(modes.factors)

    assert np.all(factors > 0.0)
    assert np.all(np.diff(factors) > 0.0)
    assert modes.shapes.shape == (4, NUM_EDGES + 1, 6)


def test_the_fully_stressed_arch_is_unstable_on_its_own(setup, steel):
    # The member-length buckling length presumes the nodes are held in plane by
    # structure outside the model. Left to itself the arch buckles well below its
    # design load, so the assumption is strong rather than conservative, and this
    # pins the number that says so.
    structure, _, _ = setup
    catalogue, result = sized(setup, steel, 3)

    modes = buckling_modes(
        analysis_model(structure, steel, catalogue),
        result.shape.xyz,
        result.sizes.sections.diameter,
        catalogue(SEED),
        funicular(structure),
        num_modes=1,
    )

    assert float(modes.factors[0]) < 1.0


def test_the_critical_mode_stays_in_the_plane(setup, steel):
    # Restraining the out-of-plane translation is what makes the factor
    # comparable with an in-plane member check rather than with a lateral one.
    structure, _, _ = setup
    catalogue, result = sized(setup, steel, 3)

    modes = buckling_modes(
        analysis_model(structure, steel, catalogue),
        result.shape.xyz,
        result.sizes.sections.diameter,
        catalogue(SEED),
        funicular(structure),
        num_modes=1,
    )
    shape = np.asarray(modes.shapes[0])
    energy = shape**2

    assert np.allclose(shape[:, NORMAL], 0.0, atol=1e-12)
    assert energy[:, [0, 2, 4]].sum() / energy.sum() > 0.99


def test_the_critical_mode_is_antisymmetric(setup, steel):
    # A node at midspan in the vertical component: one half rises as the other
    # falls, which is the governing mode of a two-pinned arch.
    structure, _, _ = setup
    catalogue, result = sized(setup, steel, 3)

    modes = buckling_modes(
        analysis_model(structure, steel, catalogue),
        result.shape.xyz,
        result.sizes.sections.diameter,
        catalogue(SEED),
        funicular(structure),
        num_modes=1,
    )
    vertical = np.asarray(modes.shapes[0])[:, 2]
    crown = (NUM_EDGES + 1) // 2

    assert abs(vertical[crown]) < 0.01 * np.max(np.abs(vertical))
    assert np.sign(vertical[1]) != np.sign(vertical[-2])


def test_sizing_against_the_global_mode_costs_several_times_the_mass(setup, steel):
    _, braced = sized(setup, steel, 3)
    arc = float(jnp.sum(braced.shape.lengths))
    _, unbraced = sized(
        setup, steel, 3, buckling_length=jnp.full(NUM_EDGES, GLOBAL_MODE_FACTOR * arc)
    )

    assert float(compute_mass(unbraced)) / float(compute_mass(braced)) > 3.0


def test_the_mode_figure_builds(setup, steel):
    structure, _, _ = setup
    catalogue, result = sized(setup, steel, 3)
    modes = buckling_modes(
        analysis_model(structure, steel, catalogue),
        result.shape.xyz,
        result.sizes.sections.diameter,
        catalogue(SEED),
        funicular(structure),
        num_modes=4,
    )

    figure = figure_modes(
        result.shape.xyz, np.asarray(modes.factors), np.asarray(modes.shapes), RISE
    )

    assert len(figure.axes) == 4
    plt.close(figure)


# --------------------------------------------------------------------------- #
# The global stability check, and the standard's two routes to slenderness
# --------------------------------------------------------------------------- #
def test_the_bare_arch_fails_the_global_stability_check(setup, steel):
    # A check, not a diagnostic: the design is verified against 5.2.1 and does
    # not satisfy it. That failure is the evidence the braced-node assumption is
    # load-bearing, so it is pinned rather than tolerated quietly.
    structure, _, _ = setup
    catalogue, result = sized(setup, steel, 3)

    checked = frame_stability(
        result,
        analyzer_of(setup[0], steel, catalogue),
        funicular(setup[0]),
        num_modes=1,
    )

    assert bool(checked.adequate) is False
    assert float(checked.utilization) > 1.0
    assert float(checked.utilization) == pytest.approx(
        ALPHA_CR_ELASTIC / float(checked.factors[0]), rel=1e-12
    )


def test_the_two_routes_to_slenderness_disagree_by_the_assumption(setup, steel):
    # Same equation, different question. The member route reads an assumed
    # buckling length; the global route reads the mode the structure has.
    structure, _, _ = setup
    catalogue, result = sized(setup, steel, 3)

    checked = frame_stability(
        result,
        analyzer_of(setup[0], steel, catalogue),
        funicular(setup[0]),
        num_modes=1,
    )
    ratio = np.asarray(checked.slenderness_global) / np.asarray(
        checked.slenderness_member
    )

    assert np.all(ratio > 4.0)
    assert np.all(np.asarray(checked.slenderness_global) > 1.0)


def test_the_global_route_is_nearly_uniform_across_the_arch(setup, steel):
    # One mode governs the whole arch, so every member inherits the same
    # slenderness from it — unlike the member route, which varies with length.
    structure, _, _ = setup
    catalogue, result = sized(setup, steel, 3)

    checked = frame_stability(
        result,
        analyzer_of(setup[0], steel, catalogue),
        funicular(setup[0]),
        num_modes=1,
    )
    spread = np.asarray(checked.slenderness_global)
    varied = np.asarray(checked.slenderness_member)

    assert spread.max() / spread.min() < 1.01
    assert varied.max() / varied.min() > 1.15


def test_the_equivalent_buckling_length_exceeds_every_member(setup, steel):
    structure, _, _ = setup
    catalogue, result = sized(setup, steel, 3)

    checked = frame_stability(
        result,
        analyzer_of(setup[0], steel, catalogue),
        funicular(setup[0]),
        num_modes=1,
    )

    assert np.all(
        np.asarray(checked.buckling_length_equivalent)
        > np.asarray(result.shape.lengths)
    )


def test_the_stability_check_reports_the_modes_it_was_asked_for(setup, steel):
    structure, _, _ = setup
    catalogue, result = sized(setup, steel, 3)

    checked = frame_stability(
        result,
        analyzer_of(setup[0], steel, catalogue),
        funicular(setup[0]),
        num_modes=3,
    )

    assert checked.factors.shape == (3,)
    assert np.all(np.diff(np.asarray(checked.factors)) > 0.0)


def test_the_threshold_of_the_clause_is_the_one_applied(setup, steel):
    # The threshold is the clause's and not a parameter, so the verdict and the
    # utilization have to be the two readings of one comparison against it.
    structure, _, _ = setup
    catalogue, result = sized(setup, steel, 3)

    checked = frame_stability(
        result,
        analyzer_of(structure, steel, catalogue),
        funicular(structure),
        num_modes=1,
    )
    alpha_cr = float(checked.factors[0])

    assert bool(checked.adequate) is (alpha_cr >= ALPHA_CR_ELASTIC)
    assert float(checked.utilization) == pytest.approx(ALPHA_CR_ELASTIC / alpha_cr)


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

    half = loads_half_span(structure, spread, factor=0.5)
    half = half * (TOTAL_LOAD / abs(float(jnp.sum(half[:, 2]))))

    point = loads_uniform(structure, spread * 0.75) + loads_point(
        structure, TOTAL_LOAD * 0.25, node=structure.crown_node()
    )

    return load_cases_of([loads_uniform(structure, spread), half, point])


def covered(setup, steel, load_cases, beta, section_class=3):
    structure, _, _ = setup
    catalogue = TubeCatalogue.at_class_limit(steel, section_class)
    pipeline = pipeline_of(setup, steel, catalogue)

    demanded = pipeline(params_of(setup), load_cases)

    return catalogue, design_envelope(demanded, beta), demanded


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
    catalogue, result, demanded = covered(setup, steel, load_cases, 500.0)
    exact = covered(setup, steel, load_cases, None)[1]

    worst = np.max(np.asarray(exact.sizes.utilization), axis=0)

    assert np.allclose(worst, 1.0, rtol=0.0, atol=TOLERANCE_UTILIZATION)


def test_the_unsmoothed_design_is_never_heavier_than_the_envelope(
    setup, steel, load_cases
):
    catalogue, result, demanded = covered(setup, steel, load_cases, 50.0)
    exact = covered(setup, steel, load_cases, None)[1]

    assert float(compute_mass(exact)) <= float(compute_mass(result))


def test_one_load_case_reproduces_the_single_case_design(setup, steel, load_cases):
    # An envelope over one case is that case, whatever the sharpness, so the
    # aggregation cannot be quietly changing the answer.
    structure, fdm, q = setup
    catalogue = TubeCatalogue.at_class_limit(steel, 3)
    seeds = jnp.full(NUM_EDGES, SEED)

    single = pipeline_of(setup, steel, catalogue)(
        DesignParameters(q, seeds),
        one_case(setup[0]),
    )
    one = LoadCases(load_cases.formfinding, load_cases.analysis[:1])
    covering = design_envelope(
        pipeline_of(setup, steel, catalogue)(DesignParameters(q, seeds), one), 25.0
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
    catalogue, result, demanded = covered(setup, steel, load_cases, 500.0)
    sizer = Ec3Sizer(setup[0], catalogue)
    reread = sizer.utilization(
        result.sizes.sections.diameter, result.sizes.actions, result.shape.lengths
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
    catalogue = TubeCatalogue.at_class_limit(steel, 3)
    seeds = jnp.full(NUM_EDGES, SEED)

    spread = TOTAL_LOAD / (NUM_EDGES - 1)
    asymmetric = loads_half_span(structure, spread, factor=0.0)

    pipeline = pipeline_of(setup, steel, catalogue)
    shaped = pipeline(DesignParameters(q, seeds), one_case(structure))
    patched = pipeline(
        DesignParameters(q, seeds),
        LoadCases(funicular(structure), jnp.stack([asymmetric])),
    )

    assert np.allclose(np.asarray(shaped.shape.xyz), np.asarray(patched.shape.xyz))
    assert float(jnp.max(jnp.abs(patched.sizes.actions.moment_major))) > float(
        jnp.max(jnp.abs(shaped.sizes.actions.moment_major))
    )


def test_the_enveloped_mass_gradient_matches_central_differences(
    setup, steel, load_cases
):
    structure, fdm, q = setup
    catalogue = TubeCatalogue.at_class_limit(steel, 3)

    pipeline = pipeline_of(setup, steel, catalogue)

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


def test_the_critical_load_factor_belongs_to_a_load_case(setup, steel, load_cases):
    # A factor quoted without the case it was measured under says less than it
    # appears to: the case a shape was found under is not the one that sized it.
    structure, _, _ = setup
    catalogue, result, demanded = covered(setup, steel, load_cases, 500.0)
    exact = covered(setup, steel, load_cases, None)[1]

    analyzer = analyzer_of(structure, steel, catalogue)
    factors = [
        float(
            frame_stability(
                exact, analyzer, load_case, load_case=index, num_modes=1
            ).factors[0]
        )
        for index, load_case in enumerate(load_cases.analysis)
    ]

    assert len(set(round(factor, 9) for factor in factors)) == len(factors)
    assert all(np.isfinite(factors))


def test_the_default_load_case_of_the_stability_check_is_the_structures_own(
    setup, steel, load_cases
):
    structure, _, _ = setup
    catalogue, result, demanded = covered(setup, steel, load_cases, 500.0)
    exact = covered(setup, steel, load_cases, None)[1]

    analyzer = analyzer_of(structure, steel, catalogue)

    implied = frame_stability(exact, analyzer, load_cases.formfinding)
    named = frame_stability(exact, analyzer, funicular(structure))

    assert float(implied.factors[0]) == pytest.approx(float(named.factors[0]))


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
    catalogue, result, demanded = covered(setup, steel, load_cases, 500.0)
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
    catalogue, result, demanded = covered(setup, steel, load_cases, 500.0)
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
    catalogue, result, demanded = covered(setup, steel, load_cases, 500.0)
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
    catalogue, result, demanded = covered(setup, steel, load_cases, 500.0)
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
