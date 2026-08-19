import jax
import jax.numpy as jnp
import numpy as np
import pytest

from normax.form_finding.fdm import FdmFormFinder
from normax.form_finding.fdm import SubspaceFormFinder
from normax.form_finding.fdm import density_basis
from normax.form_finding.fdm import equilibrium_gap
from normax.form_finding.fdm import equilibrium_graph
from normax.form_finding.fdm import fit_densities
from normax.form_finding.fdm import pivoted_basis
from normax.form_finding.fdm import plan_equilibrium
from normax.form_finding.fdm import positions_vertical
from normax.loads import loads_uniform
from normax.structures import build_arch_2d
from normax.structures import build_gridshell_3d
from normax.structures import build_structure
from normax.structures import build_vierendeel_2d
from normax.structures import build_warren_2d

NUM_BAYS = 8
SPAN = 10.0
DEPTH = 1.0
LOAD = 1.0


@pytest.fixture(scope="module")
def warren():
    return build_warren_2d(num_bays=NUM_BAYS, span=SPAN, depth=DEPTH)


@pytest.fixture(scope="module")
def lens(warren):
    xyz = np.asarray(warren.nodes).copy()
    shape = 4.0 * (xyz[:, 0] / SPAN) * (1.0 - xyz[:, 0] / SPAN)
    xyz[: NUM_BAYS + 1, 2] -= 0.06 * SPAN * shape[: NUM_BAYS + 1]
    xyz[NUM_BAYS + 1 :, 2] += 0.08 * SPAN * shape[NUM_BAYS + 1 :]
    return xyz


def test_a_chain_has_one_independent_edge():
    assert density_basis(build_arch_2d(num_edges=10)).shape[1] == 1


def test_the_warren_has_sixteen(warren):
    # 16 = 15 free heights + 1 state of self-stress; see experiment 16.
    assert density_basis(warren).shape[1] == 16


def test_the_gridshell_has_twenty_five():
    # 96 edges minus rank 71: the polar symmetry drops three balance rows.
    assert density_basis(build_gridshell_3d()).shape[1] == 25


def test_the_basis_is_orthonormal(warren):
    basis = density_basis(warren)

    assert np.allclose(basis.T @ basis, np.eye(basis.shape[1]))


def test_the_basis_annihilates_the_plan_balance(warren):
    balance = plan_equilibrium(warren)
    basis = density_basis(warren)

    assert np.abs(balance @ basis).max() < 1e-12


def test_the_fit_reaches_a_drawn_lens_exactly(warren, lens):
    loads = loads_uniform(warren, LOAD)
    fit = fit_densities(warren, lens, loads)

    assert fit.gap < 1e-12
    assert fit.self_stresses.shape == (31, 1)


def test_the_vertical_solve_reproduces_the_fitted_lens(warren, lens):
    loads = loads_uniform(warren, LOAD)
    fit = fit_densities(warren, lens, loads)

    graph = equilibrium_graph(warren)
    solved = positions_vertical(jnp.asarray(fit.q), warren.nodes, graph, loads)

    assert np.abs(np.asarray(solved) - lens).max() < 1e-10


def test_the_self_stress_leaves_the_lens_balanced(warren, lens):
    loads = loads_uniform(warren, LOAD)
    fit = fit_densities(warren, lens, loads)
    shifted = fit.q + 10.0 * fit.self_stresses[:, 0]

    assert equilibrium_gap(warren, lens, shifted, loads) < 1e-12


def test_the_gap_reports_an_unbalanced_guess(warren, lens):
    loads = loads_uniform(warren, LOAD)
    q = np.ones(warren.num_edges)

    assert equilibrium_gap(warren, lens, q, loads) > 1e-2


def _mirrored_nodes():
    bottom = NUM_BAYS - np.arange(NUM_BAYS + 1)
    top = 2 * NUM_BAYS - np.arange(NUM_BAYS)
    return np.concatenate([bottom, top])


def test_the_symmetric_warren_has_nine(warren):
    # 9 = 8 symmetric height motions + the self-stress, itself symmetric.
    assert density_basis(warren, _mirrored_nodes()).shape[1] == 9


def test_a_symmetric_chain_still_has_one():
    chain = build_arch_2d(num_edges=10)

    assert density_basis(chain, 10 - np.arange(11)).shape[1] == 1


def test_the_symmetric_basis_stays_in_the_full_subspace(warren):
    balance = plan_equilibrium(warren)
    basis = density_basis(warren, _mirrored_nodes())

    assert np.abs(balance @ basis).max() < 1e-12


def test_the_symmetric_basis_is_mirror_invariant(warren):
    basis = density_basis(warren, _mirrored_nodes())

    edges = np.sort(np.asarray(warren.edges), axis=1)
    reflected = np.sort(_mirrored_nodes()[np.asarray(warren.edges)], axis=1)
    lookup = {tuple(pair): index for index, pair in enumerate(edges.tolist())}
    targets = [lookup[tuple(pair)] for pair in reflected.tolist()]

    assert np.allclose(basis[targets], basis)


def test_a_mirror_that_breaks_edges_is_rejected(warren):
    scrambled = np.arange(warren.num_nodes)[::-1]

    with pytest.raises(ValueError):
        density_basis(warren, scrambled)


def test_the_pivoted_basis_spans_the_full_subspace(warren):
    pivot = pivoted_basis(warren)
    balance = plan_equilibrium(warren)

    assert pivot.basis.shape[1] == 16
    assert np.abs(balance @ pivot.basis).max() < 1e-12
    assert np.allclose(pivot.basis[pivot.independents], np.eye(16))


def test_the_pivoted_coordinates_read_back(warren, lens):
    loads = loads_uniform(warren, LOAD)
    fit = fit_densities(warren, lens, loads)
    pivot = pivoted_basis(warren)
    rebuilt = pivot.basis @ fit.q[pivot.independents]

    assert np.abs(rebuilt - fit.q).max() < 1e-9


def test_the_pivoted_symmetric_warren_has_nine(warren):
    pivot = pivoted_basis(warren, _mirrored_nodes())

    assert pivot.basis.shape[1] == 9
    assert np.abs(plan_equilibrium(warren) @ pivot.basis).max() < 1e-12


def test_the_pivoted_symmetric_basis_is_mirror_invariant(warren):
    pivot = pivoted_basis(warren, _mirrored_nodes())

    edges = np.sort(np.asarray(warren.edges), axis=1)
    reflected = np.sort(_mirrored_nodes()[np.asarray(warren.edges)], axis=1)
    lookup = {tuple(pair): index for index, pair in enumerate(edges.tolist())}
    targets = [lookup[tuple(pair)] for pair in reflected.tolist()]

    assert np.allclose(pivot.basis[targets], pivot.basis)


def test_the_pivoted_gridshell_has_twenty_five():
    pivot = pivoted_basis(build_gridshell_3d())

    assert pivot.basis.shape[1] == 25


@pytest.fixture(scope="module")
def vierendeel():
    return build_vierendeel_2d(num_bays=NUM_BAYS, span=SPAN, depth=DEPTH)


@pytest.fixture(scope="module")
def vierendeel_lens(vierendeel):
    xyz = np.asarray(vierendeel.nodes).copy()
    shape = 4.0 * (xyz[:, 0] / SPAN) * (1.0 - xyz[:, 0] / SPAN)
    xyz[: NUM_BAYS + 1, 2] -= 0.06 * SPAN * shape[: NUM_BAYS + 1]
    xyz[NUM_BAYS + 1 :, 2] += 0.08 * SPAN * shape[NUM_BAYS + 1 :]
    return xyz


def _deck_loads(structure):
    loads = np.zeros((structure.num_nodes, 3))
    loads[1:NUM_BAYS, 2] = -LOAD
    return loads


def _vierendeel_mirror():
    bottom = NUM_BAYS - np.arange(NUM_BAYS + 1)
    top = 2 * NUM_BAYS + 1 - np.arange(NUM_BAYS + 1)
    return np.concatenate([bottom, top])


def test_the_vierendeel_has_nine(vierendeel):
    assert density_basis(vierendeel).shape[1] == 9


def test_the_symmetric_vierendeel_has_six(vierendeel):
    assert density_basis(vierendeel, _vierendeel_mirror()).shape[1] == 6


def test_the_verticals_escape_the_plan_balance(vierendeel):
    balance = plan_equilibrium(vierendeel)

    assert np.abs(balance[:, 2 * NUM_BAYS :]).max() == 0.0


def test_a_floating_top_chord_is_forced_to_zero(vierendeel):
    nodes = np.asarray(vierendeel.nodes)
    edges = np.asarray(vierendeel.edges)
    floating = build_structure(nodes, edges, np.array([0, NUM_BAYS]))

    basis = density_basis(floating)

    assert basis.shape[1] == 8
    assert np.abs(basis[NUM_BAYS : 2 * NUM_BAYS]).max() < 1e-12


def test_the_subspace_fit_reaches_the_vierendeel_lens(vierendeel, vierendeel_lens):
    loads = _deck_loads(vierendeel)
    basis = density_basis(vierendeel)
    fit = fit_densities(vierendeel, vierendeel_lens, loads, basis)

    assert fit.gap < 1e-12
    assert fit.self_stresses.shape == (23, 1)
    assert np.abs(plan_equilibrium(vierendeel) @ fit.q).max() < 1e-12


def test_the_vertical_solve_reproduces_the_vierendeel_lens(vierendeel, vierendeel_lens):
    loads = _deck_loads(vierendeel)
    basis = density_basis(vierendeel)
    fit = fit_densities(vierendeel, vierendeel_lens, loads, basis)

    graph = equilibrium_graph(vierendeel)
    solved = positions_vertical(
        jnp.asarray(fit.q), vierendeel.nodes, graph, jnp.asarray(loads)
    )

    assert np.abs(np.asarray(solved) - vierendeel_lens).max() < 1e-10


def test_the_load_path_split_leaves_the_lens_balanced(vierendeel, vierendeel_lens):
    loads = _deck_loads(vierendeel)
    basis = density_basis(vierendeel)
    fit = fit_densities(vierendeel, vierendeel_lens, loads, basis)
    shifted = fit.q + 10.0 * fit.self_stresses[:, 0]

    assert equilibrium_gap(vierendeel, vierendeel_lens, shifted, loads) < 1e-12


def test_the_free_fit_abandons_an_unreachable_chord(vierendeel):
    xyz = np.asarray(vierendeel.nodes).copy()
    along = xyz[:, 0] / SPAN
    parabola = 4.0 * along * (1.0 - along)
    quartic = 16.0 * (along * (1.0 - along)) ** 2
    xyz[: NUM_BAYS + 1, 2] -= 0.06 * SPAN * parabola[: NUM_BAYS + 1]
    xyz[NUM_BAYS + 1 :, 2] += 0.08 * SPAN * quartic[NUM_BAYS + 1 :]

    fit = fit_densities(vierendeel, xyz, _deck_loads(vierendeel))

    assert fit.gap < 1e-12
    assert np.abs(fit.q[NUM_BAYS : 2 * NUM_BAYS]).max() < 1e-9


def test_the_pivoted_vierendeel_elects_the_verticals(vierendeel):
    pivot = pivoted_basis(vierendeel)
    verticals = set(range(2 * NUM_BAYS, 3 * NUM_BAYS - 1))

    assert pivot.basis.shape[1] == 9
    assert verticals.issubset(set(pivot.independents.tolist()))


def test_the_subspace_finder_expands_before_solving(warren, lens):
    loads = loads_uniform(warren, LOAD)
    fit = fit_densities(warren, lens, loads)
    inner = FdmFormFinder(warren)
    finder = SubspaceFormFinder(inner, density_basis(warren))
    xi = jnp.asarray(finder.read_coordinates(fit.q))

    routed = finder(xi, loads)
    direct = inner(finder.member_densities(xi), loads)

    assert np.array_equal(np.asarray(routed.xyz), np.asarray(direct.xyz))
    assert np.array_equal(np.asarray(routed.lengths), np.asarray(direct.lengths))


def test_the_orthonormal_coordinates_read_back_through_the_finder(warren, lens):
    loads = loads_uniform(warren, LOAD)
    fit = fit_densities(warren, lens, loads)
    finder = SubspaceFormFinder(FdmFormFinder(warren), density_basis(warren))

    xi = finder.read_coordinates(fit.q)
    rebuilt = np.asarray(finder.member_densities(jnp.asarray(xi)))

    assert np.abs(rebuilt - fit.q).max() < 1e-9


def test_the_pivoted_coordinates_read_back_through_the_finder(warren, lens):
    loads = loads_uniform(warren, LOAD)
    fit = fit_densities(warren, lens, loads)
    pivot = pivoted_basis(warren)
    finder = SubspaceFormFinder(FdmFormFinder(warren), pivot.basis, pivot.independents)

    xi = finder.read_coordinates(fit.q)
    rebuilt = np.asarray(finder.member_densities(jnp.asarray(xi)))

    assert np.array_equal(xi, fit.q[pivot.independents])
    assert np.abs(rebuilt - fit.q).max() < 1e-9


def test_the_subspace_gradient_chains_through_the_basis(warren, lens):
    loads = loads_uniform(warren, LOAD)
    fit = fit_densities(warren, lens, loads)
    inner = FdmFormFinder(warren)
    basis = density_basis(warren)
    finder = SubspaceFormFinder(inner, basis)
    xi = jnp.asarray(finder.read_coordinates(fit.q))

    def spanned_length(coordinate):
        return jnp.sum(finder(coordinate, loads).lengths)

    def member_length(q):
        return jnp.sum(inner(q, loads).lengths)

    slope_xi = np.asarray(jax.grad(spanned_length)(xi))
    slope_q = np.asarray(jax.grad(member_length)(finder.member_densities(xi)))

    assert np.abs(slope_xi - basis.T @ slope_q).max() < 1e-12
