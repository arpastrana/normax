import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from normax.analysis import buckling
from normax.ec3.section import area
from normax.ec3.sizing import LIMIT_MAJOR
from normax.ec3.sizing import Steel
from normax.ec3.sizing import Tube
from normax.ec3.sizing import is_plastic
from normax.ec3.stability import ALPHA_CR_ELASTIC
from normax.formfinding import equilibrium
from normax.formfinding import graph
from normax.pipeline import design
from normax.pipeline import governing
from normax.pipeline import mass
from normax.pipeline import stability
from normax.structures import arch
from normax.visualization import figure_convergence
from normax.visualization import figure_modes
from normax.visualization import figure_sections

matplotlib.use("Agg")

# A 10 m arch rising 3 m under 180 kN spread over its free nodes. Units are
# millimetres and newtons.
SPAN = 10_000.0
RISE = 3_000.0
TOTAL_LOAD = 180_000.0
NUM_EDGES = 10

# The arch lies in the XZ plane, so it has no thickness along Y.
NORMAL = 1

# The diameter the frame is analysed with before the check has spoken.
SEED = 100.0

# Invariant 6.5 of CLAUDE.md. Measured at 1.7e-15, so this is generous.
TOLERANCE_UTILIZATION = 1e-9

# Relative step at which the central difference plateaus, and the agreement
# measured there, scaled by the largest component of the gradient.
STEP = 1e-5
TOLERANCE_GRADIENT = 5e-8

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
    load = TOTAL_LOAD / (NUM_EDGES - 1)
    structure = arch(num_edges=NUM_EDGES, span=SPAN, rise=RISE, load=load)
    fdm = graph(structure)

    trial = jnp.full(NUM_EDGES, -1.0)
    reached = jnp.max(equilibrium(trial, structure, fdm).xyz[:, 2])

    return structure, fdm, trial * reached / RISE


@pytest.fixture(scope="module")
def seed():
    return jnp.full(NUM_EDGES, SEED)


def sized(setup, steel, cross_section_class, **kwargs):
    """
    One pass of form finding, analysis and the code check.
    """
    structure, fdm, q = setup
    tube = Tube.at_class_limit(steel.f_y, cross_section_class)

    result = design(
        q,
        jnp.full(NUM_EDGES, SEED),
        structure,
        fdm,
        steel,
        tube,
        normal=NORMAL,
        plastic=is_plastic(cross_section_class),
        **kwargs,
    )

    return tube, result


# --------------------------------------------------------------------------- #
# The invariant the sizing map is built to hold
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cross_section_class", [2, 3])
def test_every_member_is_utilized_exactly_once_over(setup, steel, cross_section_class):
    _, result = sized(setup, steel, cross_section_class)

    assert np.allclose(result.utilization, 1.0, rtol=0.0, atol=TOLERANCE_UTILIZATION)


@pytest.mark.parametrize("cross_section_class", [2, 3])
def test_no_member_is_pinned_to_the_catalogue_minimum(
    setup, steel, cross_section_class
):
    # Where the floor binds the utilization is below one by design, so the
    # invariant above only means something if the floor is clear of the design.
    tube, result = sized(setup, steel, cross_section_class)

    assert float(jnp.min(result.diameters)) > float(tube.diameter_min)


@pytest.mark.parametrize("cross_section_class", [2, 3])
def test_a_compression_arch_is_governed_by_the_member_check(
    setup, steel, cross_section_class
):
    tube, result = sized(setup, steel, cross_section_class)
    codes = governing(result, steel, tube, plastic=is_plastic(cross_section_class))

    assert np.all(np.asarray(codes) == LIMIT_MAJOR)


# --------------------------------------------------------------------------- #
# What the composition produces
# --------------------------------------------------------------------------- #
def test_the_mass_is_the_sum_over_members(setup, steel):
    tube, result = sized(setup, steel, 3)
    by_hand = steel.density * jnp.sum(
        area(result.diameters, tube.ratio) * result.lengths
    )

    assert float(result.mass) == pytest.approx(float(by_hand), rel=1e-14)


def test_the_mass_agrees_with_the_scalar_entry_point(setup, steel, seed):
    structure, fdm, q = setup
    tube, result = sized(setup, steel, 3)
    scalar = mass(q, seed, structure, fdm, steel, tube, normal=NORMAL, plastic=False)

    assert float(scalar) == float(result.mass)


def test_every_member_is_in_compression(setup, steel):
    _, result = sized(setup, steel, 3)

    assert np.all(np.asarray(result.n_ed) < 0.0)


def test_the_arch_is_symmetric_about_midspan(setup, steel):
    _, result = sized(setup, steel, 3)

    assert np.allclose(result.diameters, result.diameters[::-1], rtol=1e-12)
    assert np.allclose(result.lengths, result.lengths[::-1], rtol=1e-12)


def test_the_buckling_length_defaults_to_the_member_length(setup, steel):
    _, result = sized(setup, steel, 3)

    assert np.allclose(result.l_cr, result.lengths, rtol=0.0, atol=0.0)


def test_a_shorter_buckling_length_never_needs_a_larger_tube(setup, steel):
    _, longer = sized(setup, steel, 3)
    _, shorter = sized(setup, steel, 3, l_cr=longer.lengths * 0.5)

    assert np.all(np.asarray(shorter.diameters) < np.asarray(longer.diameters))
    assert float(shorter.mass) < float(longer.mass)


def test_the_thinner_walled_class_is_the_lighter_one(setup, steel):
    # Both classes carry the same forces, and the Class 3 limit puts more of the
    # steel far from the axis. Compression-governed members use the gross area
    # either way, so the thinner wall wins.
    _, plastic = sized(setup, steel, 2)
    _, elastic = sized(setup, steel, 3)

    assert float(elastic.mass) < float(plastic.mass)
    assert np.all(np.asarray(elastic.diameters) > np.asarray(plastic.diameters))


# --------------------------------------------------------------------------- #
# The gradient across all three stages
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cross_section_class", [2, 3])
def test_the_mass_gradient_matches_central_differences(
    setup, steel, seed, cross_section_class
):
    structure, fdm, q = setup
    tube = Tube.at_class_limit(steel.f_y, cross_section_class)
    plastic = is_plastic(cross_section_class)

    def objective(q):
        return mass(
            q, seed, structure, fdm, steel, tube, normal=NORMAL, plastic=plastic
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
    tube = Tube.at_class_limit(steel.f_y, 3)

    gradient = jax.grad(mass)(
        q, seed, structure, fdm, steel, tube, normal=NORMAL, plastic=False
    )

    assert np.all(np.isfinite(np.asarray(gradient)))
    assert float(jnp.min(jnp.abs(gradient))) > 0.0


def test_forward_and_reverse_mode_agree_on_the_mass(setup, steel, seed):
    structure, fdm, q = setup
    tube = Tube.at_class_limit(steel.f_y, 3)

    def objective(q):
        return mass(q, seed, structure, fdm, steel, tube, normal=NORMAL, plastic=False)

    assert np.allclose(jax.jacfwd(objective)(q), jax.grad(objective)(q), rtol=1e-12)


def test_the_mass_is_differentiable_in_the_analysed_diameters(setup, steel, seed):
    # The staggered coupling is one-way, but it is not a dead input: the sections
    # the frame is built from move the forces, and so the sizes.
    structure, fdm, q = setup
    tube = Tube.at_class_limit(steel.f_y, 3)

    gradient = jax.grad(mass, argnums=1)(
        q, seed, structure, fdm, steel, tube, normal=NORMAL, plastic=False
    )

    assert np.all(np.isfinite(np.asarray(gradient)))
    assert float(jnp.max(jnp.abs(gradient))) > 0.0


def test_the_gradient_changes_sign_across_the_arch(setup, steel, seed):
    # The springing gains more from length than it loses to section, and the
    # crown the other way round, so the sensitivity crosses zero in between.
    structure, fdm, q = setup
    tube = Tube.at_class_limit(steel.f_y, 3)

    gradient = jax.grad(mass)(
        q, seed, structure, fdm, steel, tube, normal=NORMAL, plastic=False
    )

    assert float(gradient[0]) > 0.0
    assert float(gradient[NUM_EDGES // 2]) < 0.0


# --------------------------------------------------------------------------- #
# The staggered coupling
# --------------------------------------------------------------------------- #
def test_repeating_the_pass_reaches_a_fixed_point(setup, steel, seed):
    structure, fdm, q = setup
    tube = Tube.at_class_limit(steel.f_y, 3)

    diameters = seed
    moves = []
    for _ in range(5):
        result = design(
            q, diameters, structure, fdm, steel, tube, normal=NORMAL, plastic=False
        )
        moves.append(
            float(jnp.max(jnp.abs(result.diameters - diameters) / result.diameters))
        )
        diameters = result.diameters

    # Geometric, at a contraction the analysis fixes rather than the tolerance:
    # the frame barely depends on the section, so each pass gains two decades.
    ratios = [later / earlier for earlier, later in zip(moves[:-1], moves[1:])]

    assert all(ratio < 0.1 for ratio in ratios)
    assert moves[-1] < 1e-6


def test_one_pass_is_within_two_percent_of_the_fixed_point(setup, steel, seed):
    structure, fdm, q = setup
    tube = Tube.at_class_limit(steel.f_y, 3)

    first = design(q, seed, structure, fdm, steel, tube, normal=NORMAL, plastic=False)

    diameters = first.diameters
    for _ in range(5):
        settled = design(
            q, diameters, structure, fdm, steel, tube, normal=NORMAL, plastic=False
        )
        diameters = settled.diameters

    gap = abs(float(first.mass) - float(settled.mass)) / float(settled.mass)

    assert gap < 0.02


# --------------------------------------------------------------------------- #
# The figures the experiments save
# --------------------------------------------------------------------------- #
def test_the_section_figure_builds(setup, steel, seed):
    structure, _, _ = setup
    tube, result = sized(setup, steel, 3)
    assumed = float(steel.density * jnp.sum(area(seed, tube.ratio) * result.lengths))

    figure = figure_sections(
        result.xyz,
        structure.edges,
        seed,
        result.diameters,
        assumed,
        float(result.mass),
    )

    assert len(figure.axes) == 5
    plt.close(figure)


def test_the_convergence_figure_builds():
    counts = np.array([5, 10, 20, 40])
    member = np.array([0.045, 0.031, 0.027, 0.026])
    fixed = np.array([0.033, 0.030, 0.029, 0.028])
    moves = np.array([3.8e-1, 1.2e-2, 3.1e-4, 7.9e-6])

    figure = figure_convergence(
        counts, member, fixed, 0.0274, np.arange(len(moves)), moves
    )

    assert len(figure.axes) == 3
    plt.close(figure)


def test_the_section_figure_reports_a_lighter_design(setup, steel, seed):
    # The label flips wording on the sign, so the sign is worth pinning.
    structure, _, _ = setup
    tube, result = sized(setup, steel, 3)
    assumed = float(steel.density * jnp.sum(area(seed, tube.ratio) * result.lengths))

    figure = figure_sections(
        result.xyz, structure.edges, seed, result.diameters, assumed, float(result.mass)
    )
    labels = [text.get_text() for text in figure.axes[1].texts]

    assert any("lighter" in label for label in labels)
    plt.close(figure)


# --------------------------------------------------------------------------- #
# What the member-length buckling length assumes
# --------------------------------------------------------------------------- #
def test_the_critical_factors_are_positive_and_ordered(setup, steel):
    structure, _, _ = setup
    tube, result = sized(setup, steel, 3)

    modes = buckling(
        structure,
        result.xyz,
        result.diameters,
        steel,
        tube,
        normal=NORMAL,
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
    tube, result = sized(setup, steel, 3)

    modes = buckling(
        structure, result.xyz, result.diameters, steel, tube, normal=NORMAL, num_modes=1
    )

    assert float(modes.factors[0]) < 1.0


def test_the_critical_mode_stays_in_the_plane(setup, steel):
    # Restraining the out-of-plane translation is what makes the factor
    # comparable with an in-plane member check rather than with a lateral one.
    structure, _, _ = setup
    tube, result = sized(setup, steel, 3)

    modes = buckling(
        structure, result.xyz, result.diameters, steel, tube, normal=NORMAL, num_modes=1
    )
    shape = np.asarray(modes.shapes[0])
    energy = shape**2

    assert np.allclose(shape[:, NORMAL], 0.0, atol=1e-12)
    assert energy[:, [0, 2, 4]].sum() / energy.sum() > 0.99


def test_the_critical_mode_is_antisymmetric(setup, steel):
    # A node at midspan in the vertical component: one half rises as the other
    # falls, which is the governing mode of a two-pinned arch.
    structure, _, _ = setup
    tube, result = sized(setup, steel, 3)

    modes = buckling(
        structure, result.xyz, result.diameters, steel, tube, normal=NORMAL, num_modes=1
    )
    vertical = np.asarray(modes.shapes[0])[:, 2]
    crown = (NUM_EDGES + 1) // 2

    assert abs(vertical[crown]) < 0.01 * np.max(np.abs(vertical))
    assert np.sign(vertical[1]) != np.sign(vertical[-2])


def test_sizing_against_the_global_mode_costs_several_times_the_mass(setup, steel):
    _, braced = sized(setup, steel, 3)
    arc = float(jnp.sum(braced.lengths))
    _, unbraced = sized(
        setup, steel, 3, l_cr=jnp.full(NUM_EDGES, GLOBAL_MODE_FACTOR * arc)
    )

    assert float(unbraced.mass) / float(braced.mass) > 3.0


def test_the_mode_figure_builds(setup, steel):
    structure, _, _ = setup
    tube, result = sized(setup, steel, 3)
    modes = buckling(
        structure, result.xyz, result.diameters, steel, tube, normal=NORMAL, num_modes=4
    )

    figure = figure_modes(
        result.xyz, np.asarray(modes.factors), np.asarray(modes.shapes), RISE
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
    tube, result = sized(setup, steel, 3)

    checked = stability(result, structure, steel, tube, normal=NORMAL, num_modes=1)

    assert bool(checked.adequate) is False
    assert float(checked.utilization) > 1.0
    assert float(checked.utilization) == pytest.approx(
        ALPHA_CR_ELASTIC / float(checked.factors[0]), rel=1e-12
    )


def test_the_two_routes_to_slenderness_disagree_by_the_assumption(setup, steel):
    # Same equation, different question. The member route reads an assumed
    # buckling length; the global route reads the mode the structure has.
    structure, _, _ = setup
    tube, result = sized(setup, steel, 3)

    checked = stability(result, structure, steel, tube, normal=NORMAL, num_modes=1)
    ratio = np.asarray(checked.slenderness_global) / np.asarray(
        checked.slenderness_member
    )

    assert np.all(ratio > 4.0)
    assert np.all(np.asarray(checked.slenderness_global) > 1.0)


def test_the_global_route_is_nearly_uniform_across_the_arch(setup, steel):
    # One mode governs the whole arch, so every member inherits the same
    # slenderness from it — unlike the member route, which varies with length.
    structure, _, _ = setup
    tube, result = sized(setup, steel, 3)

    checked = stability(result, structure, steel, tube, normal=NORMAL, num_modes=1)
    spread = np.asarray(checked.slenderness_global)
    varied = np.asarray(checked.slenderness_member)

    assert spread.max() / spread.min() < 1.01
    assert varied.max() / varied.min() > 1.15


def test_the_equivalent_buckling_length_exceeds_every_member(setup, steel):
    structure, _, _ = setup
    tube, result = sized(setup, steel, 3)

    checked = stability(result, structure, steel, tube, normal=NORMAL, num_modes=1)

    assert np.all(np.asarray(checked.l_cr_global) > np.asarray(result.lengths))


def test_the_stability_check_reports_the_modes_it_was_asked_for(setup, steel):
    structure, _, _ = setup
    tube, result = sized(setup, steel, 3)

    checked = stability(result, structure, steel, tube, normal=NORMAL, num_modes=3)

    assert checked.factors.shape == (3,)
    assert np.all(np.diff(np.asarray(checked.factors)) > 0.0)


def test_a_stricter_threshold_never_makes_a_frame_adequate(setup, steel):
    structure, _, _ = setup
    tube, result = sized(setup, steel, 3)

    lenient = stability(
        result, structure, steel, tube, normal=NORMAL, num_modes=1, threshold=0.05
    )
    strict = stability(
        result, structure, steel, tube, normal=NORMAL, num_modes=1, threshold=15.0
    )

    assert bool(lenient.adequate) is True
    assert bool(strict.adequate) is False
    assert float(strict.utilization) > float(lenient.utilization)
