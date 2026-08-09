import jax.numpy as jnp
import numpy as np
import pytest

from normax.structures import arch
from normax.structures import cable
from normax.structures import gridshell


@pytest.fixture
def structures():
    return [cable(), arch(), gridshell()]


def test_arrays_are_jax(structures):
    for structure in structures:
        assert isinstance(structure.nodes, jnp.ndarray)
        assert isinstance(structure.edges, jnp.ndarray)
        assert isinstance(structure.supports, jnp.ndarray)
        assert isinstance(structure.loads, jnp.ndarray)


def test_dtypes(structures):
    for structure in structures:
        assert structure.nodes.dtype == jnp.float64
        assert structure.loads.dtype == jnp.float64
        assert jnp.issubdtype(structure.edges.dtype, jnp.integer)
        assert jnp.issubdtype(structure.supports.dtype, jnp.integer)


def test_shapes_agree(structures):
    for structure in structures:
        assert structure.nodes.shape == structure.loads.shape
        assert structure.nodes.ndim == 2
        assert structure.nodes.shape[1] == 3
        assert structure.edges.shape[1] == 2


def test_edges_are_valid(structures):
    for structure in structures:
        num_nodes = structure.nodes.shape[0]
        assert jnp.all(structure.edges >= 0)
        assert jnp.all(structure.edges < num_nodes)
        assert jnp.all(structure.edges[:, 0] != structure.edges[:, 1])


def test_edges_are_unique(structures):
    for structure in structures:
        edges = np.sort(np.asarray(structure.edges), axis=1)
        assert len(np.unique(edges, axis=0)) == edges.shape[0]


def test_supports_are_valid(structures):
    for structure in structures:
        supports = np.asarray(structure.supports)
        assert supports.ndim == 1
        assert len(np.unique(supports)) == supports.size
        assert np.all(supports >= 0)
        assert np.all(supports < structure.nodes.shape[0])


def test_loads_hang_from_free_nodes_only(structures):
    for structure in structures:
        loads = np.asarray(structure.loads)
        supports = np.asarray(structure.supports)
        free = np.setdiff1d(np.arange(loads.shape[0]), supports)

        assert np.all(loads[supports] == 0.0)
        assert np.all(loads[free, 2] < 0.0)
        assert np.all(loads[:, :2] == 0.0)


def test_cable_counts():
    structure = cable(num_edges=7)

    assert structure.nodes.shape[0] == 8
    assert structure.edges.shape[0] == 7
    assert structure.supports.tolist() == [0, 7]


def test_cable_spans_supports():
    structure = cable(num_edges=7, span=12.0)
    nodes = np.asarray(structure.nodes)

    assert nodes[0].tolist() == [0.0, 0.0, 0.0]
    assert nodes[-1].tolist() == [12.0, 0.0, 0.0]


def test_cable_is_planar():
    assert np.all(np.asarray(cable().nodes)[:, 1] == 0.0)


def test_cable_sags_and_arch_rises():
    sag = np.asarray(cable(num_edges=8, sag=2.0).nodes)[:, 2]
    rise = np.asarray(arch(num_edges=8, rise=2.0).nodes)[:, 2]

    assert sag.min() == pytest.approx(-2.0)
    assert rise.max() == pytest.approx(2.0)
    assert sag == pytest.approx(-rise)


def test_cable_and_arch_share_a_topology():
    assert np.all(np.asarray(cable().edges) == np.asarray(arch().edges))
    assert np.all(np.asarray(cable().supports) == np.asarray(arch().supports))


def test_line_starts_flat():
    assert np.all(np.asarray(cable().nodes)[:, 2] == 0.0)


@pytest.mark.parametrize("num_edges", [0, -1])
def test_line_rejects_empty_discretization(num_edges):
    with pytest.raises(ValueError):
        cable(num_edges=num_edges)


def test_line_rejects_nonpositive_span():
    with pytest.raises(ValueError):
        arch(span=0.0)


def test_gridshell_counts():
    structure = gridshell(num_rings=3, num_spokes=6)

    assert structure.nodes.shape[0] == 1 + 3 * 6
    assert structure.edges.shape[0] == 2 * 3 * 6
    assert structure.supports.shape[0] == 6


def test_gridshell_supports_are_the_outer_ring():
    structure = gridshell(num_rings=3, num_spokes=6, radius=5.0)
    nodes = np.asarray(structure.nodes)
    supports = np.asarray(structure.supports)

    assert np.allclose(np.linalg.norm(nodes[supports, :2], axis=1), 5.0)
    assert np.allclose(nodes[supports, 2], 0.0)


def test_gridshell_apex():
    nodes = np.asarray(gridshell(rise=2.5).nodes)

    assert nodes[0].tolist() == [0.0, 0.0, 2.5]
    assert nodes[:, 2].max() == pytest.approx(2.5)


@pytest.mark.parametrize("rise", [0.5, 2.0, 5.0, 8.0])
def test_gridshell_nodes_lie_on_a_sphere(rise):
    radius = 5.0
    nodes = np.asarray(
        gridshell(num_rings=4, num_spokes=9, radius=radius, rise=rise).nodes
    )

    radius_sphere = (radius**2 + rise**2) / (2.0 * rise)
    center = np.array([0.0, 0.0, rise - radius_sphere])

    assert np.linalg.norm(nodes - center, axis=1) == pytest.approx(radius_sphere)


def test_gridshell_hoops_close_each_ring():
    structure = gridshell(num_rings=2, num_spokes=5)
    nodes = np.asarray(structure.nodes)
    edges = np.asarray(structure.edges)

    hoops = edges[2 * 5 :]
    lengths = np.linalg.norm(nodes[hoops[:, 0]] - nodes[hoops[:, 1]], axis=1)

    assert lengths[:5] == pytest.approx(lengths[0])
    assert lengths[5:] == pytest.approx(lengths[5])


def test_gridshell_radials_reach_the_apex():
    structure = gridshell(num_rings=3, num_spokes=6)
    edges = np.asarray(structure.edges)

    assert np.count_nonzero(edges[:, 0] == 0) == 6


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_rings": 0},
        {"num_spokes": 2},
        {"radius": 0.0},
        {"rise": 0.0},
        {"rise": -1.0},
    ],
)
def test_gridshell_rejects_degenerate_inputs(kwargs):
    with pytest.raises(ValueError):
        gridshell(**kwargs)
