import jax.numpy as jnp
import numpy as np
import pytest

from normax.structures import arch
from normax.structures import cable
from normax.structures import crown
from normax.structures import gridshell
from normax.structures import loads_half_span
from normax.structures import loads_point
from normax.structures import loads_uniform


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


# --------------------------------------------------------------------------- #
# Load cases
# --------------------------------------------------------------------------- #
@pytest.fixture
def loaded():
    return arch(num_edges=10, span=10_000.0, rise=3_000.0, load=20_000.0)


def test_a_uniform_case_is_what_the_structure_was_built_with(loaded):
    assert np.allclose(
        np.asarray(loads_uniform(loaded, 20_000.0)), np.asarray(loaded.loads)
    )


@pytest.mark.parametrize("case", ["uniform", "half", "point"])
def test_no_load_is_ever_applied_to_a_support(loaded, case):
    # A support carries a load straight to ground, so one placed there is not a
    # load case but a bookkeeping error.
    cases = {
        "uniform": loads_uniform(loaded, 20_000.0),
        "half": loads_half_span(loaded, 20_000.0),
        "point": loads_point(loaded, 50_000.0, node=int(loaded.supports[0])),
    }
    assert np.allclose(np.asarray(cases[case])[np.asarray(loaded.supports)], 0.0)


@pytest.mark.parametrize("case", ["uniform", "half", "point"])
def test_every_load_points_down(loaded, case):
    cases = {
        "uniform": loads_uniform(loaded, 20_000.0),
        "half": loads_half_span(loaded, 20_000.0),
        "point": loads_point(loaded, 50_000.0, node=5),
    }
    applied = np.asarray(cases[case])
    assert np.all(applied[:, :2] == 0.0)
    assert np.all(applied[:, 2] <= 0.0)


def test_a_half_span_case_loads_one_half_more_than_the_other(loaded):
    applied = np.asarray(loads_half_span(loaded, 20_000.0, factor=0.5)[:, 2])
    along = np.asarray(loaded.nodes[:, 0])
    middle = 0.5 * (along.min() + along.max())

    free = np.asarray(applied != 0.0)
    assert np.all(np.abs(applied[free & (along <= middle)]) == 20_000.0)
    assert np.all(np.abs(applied[free & (along > middle)]) == 10_000.0)


def test_a_half_span_case_with_no_factor_leaves_one_half_bare(loaded):
    applied = np.asarray(loads_half_span(loaded, 20_000.0, factor=0.0)[:, 2])
    along = np.asarray(loaded.nodes[:, 0])
    middle = 0.5 * (along.min() + along.max())

    assert np.all(applied[along > middle] == 0.0)
    assert np.sum(applied[along <= middle]) < 0.0


@pytest.mark.parametrize("axis", [-1, 3])
def test_a_half_span_case_refuses_an_axis_that_is_not_a_dimension(loaded, axis):
    with pytest.raises(ValueError, match="axis"):
        loads_half_span(loaded, 20_000.0, axis=axis)


def test_a_point_case_loads_exactly_one_node(loaded):
    applied = np.asarray(loads_point(loaded, 50_000.0, node=5)[:, 2])

    assert np.count_nonzero(applied) == 1
    assert applied[5] == -50_000.0


def test_load_cases_add(loaded):
    # Cases are arrays and nothing else, so a distributed load with a point load
    # on top of it needs no generator of its own.
    combined = loads_uniform(loaded, 10_000.0) + loads_point(loaded, 50_000.0, node=5)
    total = float(jnp.sum(combined[:, 2]))

    assert total == pytest.approx(-(9 * 10_000.0 + 50_000.0))


def test_the_crown_is_the_highest_node(loaded):
    index = crown(loaded)

    assert isinstance(index, int)
    assert float(loaded.nodes[index, 2]) == float(jnp.max(loaded.nodes[:, 2]))
