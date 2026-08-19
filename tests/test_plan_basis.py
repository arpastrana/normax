import jax.numpy as jnp
import numpy as np
import pytest

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
