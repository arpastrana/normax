# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from normax.config import ConstraintsConfig
from normax.design import build_design_constraints
from normax.form_finding import DrawnShapeInitializer
from normax.form_finding import FdmFormFinder
from normax.form_finding import HeightsFormFinder
from normax.form_finding import LensShapeInitializer
from normax.form_finding import SignGuardSpec
from normax.form_finding import assemble_balance_rows
from normax.form_finding import build_plan_basis
from normax.form_finding import fit_densities
from normax.form_finding import select_free_nodes
from normax.loads import create_load_uniform
from normax.loads import read_polar_plan
from normax.structures import build_arch_2d
from normax.structures import build_gridshell_3d
from normax.structures import build_structure
from normax.structures import build_vierendeel_2d
from normax.structures import build_warren_2d
from normax.symmetry import build_height_groups
from normax.symmetry import find_mirror_nodes
from normax.symmetry import find_rotated_nodes
from normax.symmetry import fold_heights
from normax.symmetry import fold_values
from normax.symmetry import permute_free_nodes
from normax.symmetry import permute_members
from normax.symmetry import sketch_lens

NUM_BAYS = 8
SPAN = 10.0
DEPTH = 1.0
TOTAL = 15.0

# Margin a guarded start must clear, as a share of the guarded median.
MARGIN_FRACTION = 0.05

# Constraints naming nothing but the guard, which is what these tests read.
WARREN_CONSTRAINTS = ConstraintsConfig(
    diameter_min=0.0,
    length_min=0.0,
    rise_max=None,
    sag_min=None,
    sign_margin_fraction=MARGIN_FRACTION,
    sign_guard=None,
    bounds=None,
)


def balance_gap(structure, xyz, q, loads):
    """
    Largest violation of the full nodal balance at a drawn geometry.
    """
    balance = assemble_balance_rows(structure, np.asarray(xyz), (0, 1, 2))
    nodes_free = select_free_nodes(structure)
    columns = [np.asarray(loads)[nodes_free, axis] for axis in (0, 1, 2)]
    applied = np.concatenate(columns)

    return float(np.abs(balance @ np.asarray(q) - applied).max())


def plan_balance(structure):
    """
    The horizontal balance rows a held plan must annihilate.
    """
    return assemble_balance_rows(structure, structure.nodes, (0, 1))


def mirrored_nodes():
    bottom = NUM_BAYS - np.arange(NUM_BAYS + 1)
    top = 2 * NUM_BAYS - np.arange(NUM_BAYS)

    return np.concatenate([bottom, top])


def vierendeel_mirror():
    bottom = NUM_BAYS - np.arange(NUM_BAYS + 1)
    top = 2 * NUM_BAYS + 1 - np.arange(NUM_BAYS + 1)

    return np.concatenate([bottom, top])


@pytest.fixture(scope="module")
def warren():
    return build_warren_2d(num_bays=NUM_BAYS, span=SPAN, depth=DEPTH)


@pytest.fixture(scope="module")
def lens(warren):
    return sketch_lens(warren, 0.06 * SPAN, 0.08 * SPAN)


@pytest.fixture(scope="module")
def warren_loads(warren):
    return create_load_uniform(warren, TOTAL)


@pytest.fixture(scope="module")
def lens_fit(warren, lens, warren_loads):
    return fit_densities(warren, lens, warren_loads)


@pytest.fixture(scope="module")
def vierendeel():
    return build_vierendeel_2d(num_bays=NUM_BAYS, span=SPAN, depth=DEPTH)


@pytest.fixture(scope="module")
def vierendeel_lens(vierendeel):
    return sketch_lens(vierendeel, 0.06 * SPAN, 0.08 * SPAN)


def deck_loads(structure):
    loads = np.zeros((structure.num_nodes, 3))
    loads[1:NUM_BAYS, 2] = -1.0

    return loads


# --------------------------------------------------------------------------- #
# The width of the held-plan subspace
# --------------------------------------------------------------------------- #
def test_a_chain_has_one_independent_edge():
    chain = build_arch_2d(num_edges=10)

    assert build_plan_basis(chain, None, "svd").width == 1


def test_the_warren_has_sixteen(warren):
    # 16 = 15 free heights + 1 state of self-stress; see experiment 16.
    assert build_plan_basis(warren, None, "svd").width == 16


def test_the_gridshell_has_thirteen():
    # 84 edges minus rank 71; no silent boundary-hoop coefficients remain.
    assert build_plan_basis(build_gridshell_3d(), None, "svd").width == 13


def test_every_gridshell_coefficient_moves_the_shell():
    structure = build_gridshell_3d()
    basis = build_plan_basis(structure, None, "svd")
    balance = plan_balance(structure)

    assert np.abs(balance).max(axis=0).min() > 0.0
    assert np.linalg.matrix_rank(basis.columns) == basis.width


def test_the_symmetric_warren_has_nine(warren):
    # 9 = 8 symmetric height motions + the self-stress, itself symmetric.
    assert build_plan_basis(warren, mirrored_nodes(), "svd").width == 9


def test_a_symmetric_chain_still_has_one():
    chain = build_arch_2d(num_edges=10)

    assert build_plan_basis(chain, 10 - np.arange(11), "svd").width == 1


def test_the_vierendeel_has_nine(vierendeel):
    assert build_plan_basis(vierendeel, None, "svd").width == 9


def test_the_symmetric_vierendeel_has_six(vierendeel):
    assert build_plan_basis(vierendeel, vierendeel_mirror(), "svd").width == 6


# --------------------------------------------------------------------------- #
# What the columns satisfy
# --------------------------------------------------------------------------- #
def test_the_orthonormal_basis_is_orthonormal(warren):
    basis = build_plan_basis(warren, None, "svd")
    gram = basis.columns.T @ basis.columns

    assert np.allclose(gram, np.eye(basis.width))


def test_the_basis_annihilates_the_plan_balance(warren):
    basis = build_plan_basis(warren, None, "svd")

    assert np.abs(plan_balance(warren) @ basis.columns).max() < 1e-12


def test_the_symmetric_basis_stays_in_the_full_subspace(warren):
    basis = build_plan_basis(warren, mirrored_nodes(), "svd")

    assert np.abs(plan_balance(warren) @ basis.columns).max() < 1e-12


def test_the_symmetric_basis_is_mirror_invariant(warren):
    basis = build_plan_basis(warren, mirrored_nodes(), "svd")
    targets = permute_members(mirrored_nodes(), warren)

    assert np.allclose(basis.columns[targets], basis.columns)


def test_a_mirror_that_breaks_edges_is_rejected(warren):
    scrambled = np.arange(warren.num_nodes)[::-1]

    with pytest.raises(ValueError):
        build_plan_basis(warren, scrambled, "svd")


def test_the_verticals_escape_the_plan_balance(vierendeel):
    balance = plan_balance(vierendeel)

    assert np.abs(balance[:, 2 * NUM_BAYS :]).max() == 0.0


def test_a_floating_top_chord_is_forced_to_zero(vierendeel):
    nodes = np.asarray(vierendeel.nodes)
    edges = np.asarray(vierendeel.edges)
    floating = build_structure(nodes, edges, np.array([0, NUM_BAYS]))

    basis = build_plan_basis(floating, None, "svd")

    assert basis.width == 8
    assert np.abs(basis.columns[NUM_BAYS : 2 * NUM_BAYS]).max() < 1e-12


# --------------------------------------------------------------------------- #
# The pivoted convention
# --------------------------------------------------------------------------- #
def test_the_pivoted_basis_spans_the_full_subspace(warren):
    basis = build_plan_basis(warren, None, "pivoted")

    assert basis.width == 16
    assert np.abs(plan_balance(warren) @ basis.columns).max() < 1e-12
    assert np.allclose(basis.columns[basis.independents], np.eye(16))


def test_the_pivoted_coefficients_read_back(warren, lens_fit):
    basis = build_plan_basis(warren, None, "pivoted")
    xi = basis.coefficients(lens_fit.q)
    rebuilt = np.asarray(basis.densities(jnp.asarray(xi)))

    assert np.array_equal(xi, lens_fit.q[basis.independents])
    assert np.abs(rebuilt - lens_fit.q).max() < 1e-9


def test_the_orthonormal_coefficients_read_back(warren, lens_fit):
    # The lens moves heights alone, so the free fit already holds the plan.
    basis = build_plan_basis(warren, None, "svd")
    xi = basis.coefficients(lens_fit.q)
    rebuilt = np.asarray(basis.densities(jnp.asarray(xi)))

    assert np.allclose(xi, basis.columns.T @ lens_fit.q)
    assert np.abs(rebuilt - lens_fit.q).max() < 1e-9


def test_the_pivoted_symmetric_warren_has_nine(warren):
    basis = build_plan_basis(warren, mirrored_nodes(), "pivoted")

    assert basis.width == 9
    assert np.abs(plan_balance(warren) @ basis.columns).max() < 1e-12


def test_the_pivoted_symmetric_basis_is_mirror_invariant(warren):
    basis = build_plan_basis(warren, mirrored_nodes(), "pivoted")
    targets = permute_members(mirrored_nodes(), warren)

    assert np.allclose(basis.columns[targets], basis.columns)


def test_the_pivoted_gridshell_has_thirteen():
    assert build_plan_basis(build_gridshell_3d(), None, "pivoted").width == 13


def test_the_pivoted_vierendeel_elects_the_verticals(vierendeel):
    basis = build_plan_basis(vierendeel, None, "pivoted")
    verticals = set(range(2 * NUM_BAYS, 3 * NUM_BAYS - 1))

    assert basis.width == 9
    assert verticals.issubset(set(basis.independents.tolist()))


# --------------------------------------------------------------------------- #
# Fitting densities to a drawn geometry
# --------------------------------------------------------------------------- #
def test_the_fit_reaches_a_drawn_lens_exactly(lens_fit):
    assert lens_fit.gap < 1e-12
    assert lens_fit.self_stresses.shape == (31, 1)


def test_the_solve_reproduces_the_fitted_lens(warren, lens, lens_fit, warren_loads):
    solved = FdmFormFinder(warren)(jnp.asarray(lens_fit.q), warren_loads)

    assert np.abs(np.asarray(solved.xyz) - lens).max() < 1e-9


def test_the_self_stress_leaves_the_lens_balanced(warren, lens, lens_fit, warren_loads):
    shifted = lens_fit.q + 10.0 * lens_fit.self_stresses[:, 0]

    assert balance_gap(warren, lens, shifted, warren_loads) < 1e-12


def test_the_balance_reports_an_unbalanced_guess(warren, lens, warren_loads):
    q = np.ones(warren.num_edges)

    assert balance_gap(warren, lens, q, warren_loads) > 1e-2


def test_the_subspace_fit_reaches_the_vierendeel_lens(vierendeel, vierendeel_lens):
    loads = deck_loads(vierendeel)
    basis = build_plan_basis(vierendeel, None, "svd")
    fit = fit_densities(vierendeel, vierendeel_lens, loads, basis)

    assert fit.gap < 1e-12
    assert fit.self_stresses.shape == (23, 1)
    assert np.abs(plan_balance(vierendeel) @ fit.q).max() < 1e-12


def test_the_solve_reproduces_the_vierendeel_lens(vierendeel, vierendeel_lens):
    loads = deck_loads(vierendeel)
    basis = build_plan_basis(vierendeel, None, "svd")
    fit = fit_densities(vierendeel, vierendeel_lens, loads, basis)

    solved = FdmFormFinder(vierendeel)(jnp.asarray(fit.q), jnp.asarray(loads))

    assert np.abs(np.asarray(solved.xyz) - vierendeel_lens).max() < 1e-9


def test_the_load_path_split_leaves_the_lens_balanced(vierendeel, vierendeel_lens):
    loads = deck_loads(vierendeel)
    basis = build_plan_basis(vierendeel, None, "svd")
    fit = fit_densities(vierendeel, vierendeel_lens, loads, basis)
    shifted = fit.q + 10.0 * fit.self_stresses[:, 0]

    assert balance_gap(vierendeel, vierendeel_lens, shifted, loads) < 1e-12


def test_the_free_fit_abandons_an_unreachable_chord(vierendeel):
    xyz = np.asarray(vierendeel.nodes).copy()
    along = xyz[:, 0] / SPAN
    parabola = 4.0 * along * (1.0 - along)
    quartic = 16.0 * (along * (1.0 - along)) ** 2
    xyz[: NUM_BAYS + 1, 2] -= 0.06 * SPAN * parabola[: NUM_BAYS + 1]
    xyz[NUM_BAYS + 1 :, 2] += 0.08 * SPAN * quartic[NUM_BAYS + 1 :]

    fit = fit_densities(vierendeel, xyz, deck_loads(vierendeel))

    assert fit.gap < 1e-12
    assert np.abs(fit.q[NUM_BAYS : 2 * NUM_BAYS]).max() < 1e-9


# --------------------------------------------------------------------------- #
# The expansion a search differentiates through
# --------------------------------------------------------------------------- #
def test_the_expansion_stays_in_the_span(warren):
    basis = build_plan_basis(warren, None, "pivoted")
    xi = jnp.asarray(-1.0 - np.linspace(0.0, 1.0, basis.width))
    q = basis.densities(xi)

    assert np.abs(plan_balance(warren) @ np.asarray(q)).max() < 1e-12


def test_the_gradient_chains_through_the_basis(warren, lens_fit, warren_loads):
    finder = FdmFormFinder(warren)
    basis = build_plan_basis(warren, None, "svd")
    xi = jnp.asarray(basis.coefficients(lens_fit.q))

    def spanned_length(coefficient):
        return jnp.sum(finder(basis.densities(coefficient), warren_loads).lengths)

    def member_length(q):
        return jnp.sum(finder(q, warren_loads).lengths)

    slope_xi = np.asarray(jax.grad(spanned_length)(xi))
    slope_q = np.asarray(jax.grad(member_length)(basis.densities(xi)))

    assert np.abs(slope_xi - basis.columns.T @ slope_q).max() < 1e-12


def test_the_geometric_mirror_matches_the_indexed_one(warren, vierendeel):
    assert np.array_equal(find_mirror_nodes(warren, "x"), mirrored_nodes())
    assert np.array_equal(find_mirror_nodes(vierendeel, "x"), vierendeel_mirror())


def test_the_geometric_rotation_permutes_the_rings_of_a_cap():
    cap = build_gridshell_3d(num_rings=3, num_spokes=8)
    plan = read_polar_plan(cap)
    rotated = find_rotated_nodes(cap, plan.num_spokes)
    ringed = plan.ring > 0
    assert np.array_equal(plan.ring[rotated], plan.ring)
    assert rotated[~ringed] == np.flatnonzero(~ringed)
    turned = (plan.spoke[ringed] + 1) % plan.num_spokes
    assert np.array_equal(plan.spoke[rotated][ringed], turned)


def test_an_unsymmetric_structure_has_no_mirror(warren):
    tilted = warren._replace(nodes=warren.nodes.at[3, 2].add(1.0))
    with pytest.raises(ValueError):
        find_mirror_nodes(tilted, "x")


def guarded_warren(margin_fraction):
    """
    A spec signing the Warren's chords, at a given margin.
    """
    bays = NUM_BAYS
    signs = np.concatenate([np.ones(bays), -np.ones(bays - 1)])

    return SignGuardSpec(signs, np.arange(2 * bays - 1), margin_fraction)


def test_a_zero_margin_still_signs_the_start_but_hands_over_no_guard(
    warren, warren_loads
):
    lens = {"sag": 0.06 * SPAN, "rise": 0.08 * SPAN, "held_plan": False}
    initializer = LensShapeInitializer(lens)
    guarded = guarded_warren(0.0)
    density_start = initializer(warren, np.asarray(warren_loads), None, guarded)
    held = build_design_constraints(WARREN_CONSTRAINTS, guarded, density_start)

    assert held.sign_guard is None
    assert np.all(guarded.signs * density_start[: guarded.members.size] >= 0.0)


def test_the_guard_is_scaled_at_the_shifted_densities_not_the_raw_fit(
    warren, warren_loads
):
    lens = {"sag": 0.06 * SPAN, "rise": 0.08 * SPAN, "held_plan": False}
    initializer = LensShapeInitializer(lens)
    guarded = guarded_warren(MARGIN_FRACTION)
    density_start = initializer(warren, np.asarray(warren_loads), None, guarded)
    held = build_design_constraints(WARREN_CONSTRAINTS, guarded, density_start)

    fit = initializer.fit_start(warren, np.asarray(warren_loads), None)
    assert fit.self_stresses.shape[1] > 0
    assert not np.allclose(fit.q, density_start)

    # The margin is a share of the median guarded density, so scaling it at the
    # raw fit rather than at the shifted start would leave a different number.
    shifted = float(np.median(np.abs(density_start[guarded.members])))
    assert held.sign_guard is not None
    assert held.sign_guard.scale == pytest.approx(shifted)
    assert held.sign_guard.margin == pytest.approx(MARGIN_FRACTION * shifted)

    # And the start clears the margin it was scaled against.
    signed = guarded.signs * density_start[guarded.members]
    assert signed.min() >= held.sign_guard.margin


def test_a_held_lens_fit_needs_a_basis(warren, warren_loads):
    lens = {"sag": 0.06 * SPAN, "rise": 0.08 * SPAN, "held_plan": True}
    initializer = LensShapeInitializer(lens)
    with pytest.raises(ValueError):
        initializer(warren, np.asarray(warren_loads), None, None)


def test_a_drawn_fit_is_held_to_the_basis(warren, warren_loads):
    # A balanceable drawn geometry lands in the span either way, to 1e-14, so
    # the guard is that the restricted solve is the one actually run.
    basis = build_plan_basis(warren, None, "svd")
    loads = np.asarray(warren_loads)
    xyz = np.asarray(warren.nodes)
    fit = DrawnShapeInitializer({}).fit_start(warren, loads, basis)
    held = fit_densities(warren, xyz, loads, basis)
    free = fit_densities(warren, xyz, loads)

    assert np.array_equal(fit.q, held.q)
    assert not np.array_equal(fit.q, free.q)


def test_a_lens_missing_a_key_is_refused():
    # The file's mapping is the initializer's own fields, so a key it omits is
    # refused where the example builds it.
    described = {"sag": 0.6, "rise": 0.8}
    with pytest.raises(ValueError, match="held_plan"):
        LensShapeInitializer(described)


def test_a_drawn_start_that_names_a_field_is_refused():
    with pytest.raises(ValueError, match="nothing"):
        DrawnShapeInitializer({"sag": 0.6})


# --------------------------------------------------------------------------- #
# Folding the written heights by the mirror
# --------------------------------------------------------------------------- #
def test_the_mirror_folds_the_free_heights_into_pairs(warren, vierendeel):
    arch = build_arch_2d(num_edges=10, span=SPAN, rise=0.0)

    for structure, folded in ((arch, 5), (warren, 8), (vierendeel, 8)):
        mirror = find_mirror_nodes(structure, "x")
        groups = build_height_groups(structure, (mirror,))
        free = select_free_nodes(structure).size
        assert groups.shape == (free, folded)
        # Every free node belongs to exactly one orbit, and a mirrored pair to
        # the same one, so the columns partition the free nodes.
        assert np.array_equal(groups.sum(axis=1), np.ones(free))
        assert set(np.unique(groups.sum(axis=0))) <= {1.0, 2.0}


def test_no_permutation_at_all_folds_nothing(warren):
    assert build_height_groups(warren, (None,)) is None


def test_a_permutation_carrying_a_free_node_onto_a_support_is_refused(warren):
    # A support and a free node swapped: not a symmetry of the free set, and
    # its orbits would fold a height onto one the finder never writes.
    spoiled = np.arange(warren.num_nodes)
    supported = int(np.asarray(warren.supports)[0])
    free = int(select_free_nodes(warren)[0])
    spoiled[free], spoiled[supported] = supported, free

    with pytest.raises(ValueError, match="maps onto support"):
        permute_free_nodes(spoiled, warren)


def test_a_folded_finder_writes_a_mirror_symmetric_shape(warren):
    # The property the whole fold exists for: whatever the coefficients, the
    # geometry comes out symmetric, so a mirrored load case is a reindexing of
    # its partner rather than a case the design can lean away from.
    mirror = find_mirror_nodes(warren, "x")
    groups = build_height_groups(warren, (mirror,))
    finder = HeightsFormFinder(warren, None, groups)
    seeded = np.random.default_rng(0).standard_normal(finder.count_shape_coefficients())

    heights = np.asarray(finder.expand_shape_coefficients(jnp.asarray(seeded)))
    shape = finder(jnp.asarray(heights), jnp.zeros((warren.num_nodes, 3)))

    lifted = np.asarray(shape.xyz)[:, 2]
    assert np.allclose(lifted, lifted[np.asarray(mirror)], atol=1e-12)


def test_an_unfolded_finder_does_not(warren):
    finder = HeightsFormFinder(warren, None, None)
    mirror = find_mirror_nodes(warren, "x")
    seeded = np.random.default_rng(0).standard_normal(finder.count_shape_coefficients())

    heights = np.asarray(finder.expand_shape_coefficients(jnp.asarray(seeded)))
    shape = finder(jnp.asarray(heights), jnp.zeros((warren.num_nodes, 3)))

    lifted = np.asarray(shape.xyz)[:, 2]
    assert not np.allclose(lifted, lifted[np.asarray(mirror)], atol=1e-12)


def test_folding_a_height_is_the_left_inverse_of_expanding_it(warren):
    mirror = find_mirror_nodes(warren, "x")
    groups = build_height_groups(warren, (mirror,))
    finder = HeightsFormFinder(warren, None, groups)
    seeded = np.random.default_rng(1).standard_normal(finder.count_shape_coefficients())

    heights = np.asarray(finder.expand_shape_coefficients(jnp.asarray(seeded)))

    assert np.allclose(fold_heights(heights, groups), seeded, atol=1e-12)


def test_a_folded_height_is_the_mean_of_its_orbit_not_the_largest(warren):
    # fold_values takes the largest so a diameter covers every member it
    # sizes; a height folded that way would lift a start off its own mirror.
    mirror = find_mirror_nodes(warren, "x")
    groups = build_height_groups(warren, (mirror,))
    lopsided = np.zeros(select_free_nodes(warren).size)
    lopsided[0] = 2.0

    folded = fold_heights(lopsided, groups)

    assert folded.max() == pytest.approx(1.0)
    assert fold_values(lopsided, groups).max() == pytest.approx(2.0)
