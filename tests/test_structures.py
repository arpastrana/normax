import jax.numpy as jnp
import numpy as np
import pytest

from normax.loads import create_loads_half_span
from normax.loads import create_loads_point
from normax.loads import create_loads_uniform
from normax.structures import build_arch_2d
from normax.structures import build_gridshell_3d
from normax.structures import build_vierendeel_2d
from normax.structures import build_warren_2d


@pytest.fixture
def structures():
    return [
        build_arch_2d(),
        build_gridshell_3d(),
        build_warren_2d(),
        build_vierendeel_2d(),
    ]


def test_arrays_are_jax(structures):
    for structure in structures:
        assert isinstance(structure.nodes, jnp.ndarray)
        assert isinstance(structure.edges, jnp.ndarray)
        assert isinstance(structure.supports, jnp.ndarray)


def test_dtypes(structures):
    for structure in structures:
        assert structure.nodes.dtype == jnp.float64
        assert jnp.issubdtype(structure.edges.dtype, jnp.integer)
        assert jnp.issubdtype(structure.supports.dtype, jnp.integer)


def test_shapes_agree(structures):
    for structure in structures:
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
        loads = np.asarray(create_loads_uniform(structure, 1.0))
        supports = np.asarray(structure.supports)
        free = np.setdiff1d(np.arange(loads.shape[0]), supports)

        assert np.all(loads[supports] == 0.0)
        assert np.all(loads[free, 2] < 0.0)
        assert np.all(loads[:, :2] == 0.0)


def test_arch_counts():
    structure = build_arch_2d(num_edges=7)

    assert structure.nodes.shape[0] == 8
    assert structure.edges.shape[0] == 7
    assert structure.supports.tolist() == [0, 7]


def test_arch_spans_supports():
    structure = build_arch_2d(num_edges=7, span=12.0)
    nodes = np.asarray(structure.nodes)

    assert nodes[0].tolist() == [0.0, 0.0, 0.0]
    assert nodes[-1].tolist() == [12.0, 0.0, 0.0]


def test_arch_is_planar():
    assert np.all(np.asarray(build_arch_2d().nodes)[:, 1] == 0.0)


def test_the_arch_rises_to_its_crown_and_returns_to_its_supports():
    rise = np.asarray(build_arch_2d(num_edges=8, rise=2.0).nodes)[:, 2]

    assert rise.max() == pytest.approx(2.0)
    assert rise[0] == pytest.approx(0.0)
    assert rise[-1] == pytest.approx(0.0)


def test_an_arch_of_no_rise_starts_flat():
    assert np.all(np.asarray(build_arch_2d().nodes)[:, 2] == 0.0)


@pytest.mark.parametrize("num_edges", [0, -1])
def test_the_arch_rejects_empty_discretization(num_edges):
    with pytest.raises(ValueError):
        build_arch_2d(num_edges=num_edges)


def test_the_arch_rejects_a_nonpositive_span():
    with pytest.raises(ValueError):
        build_arch_2d(span=0.0)


def test_warren_counts():
    structure = build_warren_2d(num_bays=8)

    assert structure.nodes.shape[0] == 17
    assert structure.edges.shape[0] == 31
    assert structure.supports.tolist() == [0, 8]


def test_warren_chords_sit_level():
    structure = build_warren_2d(num_bays=6, span=12.0, depth=1.5)
    nodes = np.asarray(structure.nodes)

    assert np.all(nodes[:7, 2] == 0.0)
    assert np.all(nodes[7:, 2] == 1.5)
    assert np.all(nodes[:, 1] == 0.0)


def test_warren_top_chord_is_offset_half_a_bay():
    structure = build_warren_2d(num_bays=6, span=12.0)
    nodes = np.asarray(structure.nodes)

    assert nodes[7:, 0] == pytest.approx(nodes[:6, 0] + 1.0)


def test_warren_edge_families_come_in_order():
    structure = build_warren_2d(num_bays=4)
    edges = np.asarray(structure.edges)

    assert edges[:4].tolist() == [[0, 1], [1, 2], [2, 3], [3, 4]]
    assert edges[4:7].tolist() == [[5, 6], [6, 7], [7, 8]]
    assert edges[7:11].tolist() == [[0, 5], [1, 6], [2, 7], [3, 8]]
    assert edges[11:].tolist() == [[5, 1], [6, 2], [7, 3], [8, 4]]


@pytest.mark.parametrize("num_bays", [1, 0, -2])
def test_the_warren_rejects_too_few_bays(num_bays):
    with pytest.raises(ValueError):
        build_warren_2d(num_bays=num_bays)


def test_the_warren_rejects_a_nonpositive_depth():
    with pytest.raises(ValueError):
        build_warren_2d(depth=0.0)


def test_vierendeel_counts():
    structure = build_vierendeel_2d(num_bays=8)

    assert structure.nodes.shape[0] == 18
    assert structure.edges.shape[0] == 23
    assert structure.supports.tolist() == [0, 8, 9, 17]


def test_vierendeel_chords_sit_level():
    structure = build_vierendeel_2d(num_bays=6, span=12.0, depth=1.5)
    nodes = np.asarray(structure.nodes)

    assert np.all(nodes[:7, 2] == 0.0)
    assert np.all(nodes[7:, 2] == 1.5)
    assert np.all(nodes[:, 1] == 0.0)


def test_vierendeel_verticals_are_plumb():
    structure = build_vierendeel_2d(num_bays=6, span=12.0)
    nodes = np.asarray(structure.nodes)

    assert nodes[7:, 0] == pytest.approx(nodes[:7, 0])


def test_vierendeel_edge_families_come_in_order():
    structure = build_vierendeel_2d(num_bays=4)
    edges = np.asarray(structure.edges)

    assert edges[:4].tolist() == [[0, 1], [1, 2], [2, 3], [3, 4]]
    assert edges[4:8].tolist() == [[5, 6], [6, 7], [7, 8], [8, 9]]
    assert edges[8:].tolist() == [[1, 6], [2, 7], [3, 8]]


@pytest.mark.parametrize("num_bays", [1, 0, -2])
def test_the_vierendeel_rejects_too_few_bays(num_bays):
    with pytest.raises(ValueError):
        build_vierendeel_2d(num_bays=num_bays)


def test_the_vierendeel_rejects_a_nonpositive_depth():
    with pytest.raises(ValueError):
        build_vierendeel_2d(depth=0.0)


def test_gridshell_counts():
    structure = build_gridshell_3d(num_rings=3, num_spokes=6)

    assert structure.nodes.shape[0] == 1 + 3 * 6
    assert structure.edges.shape[0] == 3 * 6 + (3 - 1) * 6
    assert structure.supports.shape[0] == 6


def test_gridshell_leaves_the_boundary_ring_unhooped():
    structure = build_gridshell_3d(num_rings=3, num_spokes=6)
    edges = np.asarray(structure.edges)
    supports = set(np.asarray(structure.supports).tolist())

    both = [pair for pair in edges.tolist() if set(pair) <= supports]

    assert both == []


def test_gridshell_supports_are_the_outer_ring():
    structure = build_gridshell_3d(num_rings=3, num_spokes=6, radius=5.0)
    nodes = np.asarray(structure.nodes)
    supports = np.asarray(structure.supports)

    assert np.allclose(np.linalg.norm(nodes[supports, :2], axis=1), 5.0)
    assert np.allclose(nodes[supports, 2], 0.0)


def test_gridshell_apex():
    nodes = np.asarray(build_gridshell_3d(rise=2.5).nodes)

    assert nodes[0].tolist() == [0.0, 0.0, 2.5]
    assert nodes[:, 2].max() == pytest.approx(2.5)


@pytest.mark.parametrize("rise", [0.5, 2.0, 5.0, 8.0])
def test_gridshell_nodes_lie_on_a_sphere(rise):
    radius = 5.0
    nodes = np.asarray(
        build_gridshell_3d(num_rings=4, num_spokes=9, radius=radius, rise=rise).nodes
    )

    radius_sphere = (radius**2 + rise**2) / (2.0 * rise)
    center = np.array([0.0, 0.0, rise - radius_sphere])

    assert np.linalg.norm(nodes - center, axis=1) == pytest.approx(radius_sphere)


def test_gridshell_hoops_close_each_ring():
    structure = build_gridshell_3d(num_rings=3, num_spokes=5)
    nodes = np.asarray(structure.nodes)
    edges = np.asarray(structure.edges)

    hoops = edges[3 * 5 :]
    lengths = np.linalg.norm(nodes[hoops[:, 0]] - nodes[hoops[:, 1]], axis=1)

    assert lengths.size == 2 * 5
    assert lengths[:5] == pytest.approx(lengths[0])
    assert lengths[5:] == pytest.approx(lengths[5])


def test_gridshell_radials_reach_the_apex():
    structure = build_gridshell_3d(num_rings=3, num_spokes=6)
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
        build_gridshell_3d(**kwargs)


# --------------------------------------------------------------------------- #
# Load cases
# --------------------------------------------------------------------------- #
@pytest.fixture
def loaded():
    return build_arch_2d(num_edges=10, span=10_000.0, rise=3_000.0)


def test_a_uniform_load_case_hangs_the_same_force_from_every_free_node(loaded):
    # A structure carries no load of its own, so a load case is built from it
    # rather than read off it.
    applied = np.asarray(create_loads_uniform(loaded, 20_000.0))
    free = np.setdiff1d(np.arange(applied.shape[0]), np.asarray(loaded.supports))

    assert np.allclose(applied[free, 2], -20_000.0)
    assert np.all(applied[np.asarray(loaded.supports)] == 0.0)


@pytest.mark.parametrize("load_case", ["uniform", "half", "point"])
def test_no_load_is_ever_applied_to_a_support(loaded, load_case):
    # A support carries a load straight to ground, so one placed there is not a
    # load case but a bookkeeping error.
    load_cases = {
        "uniform": create_loads_uniform(loaded, 20_000.0),
        "half": create_loads_half_span(loaded, 20_000.0),
        "point": create_loads_point(loaded, 50_000.0, node=int(loaded.supports[0])),
    }
    assert np.allclose(
        np.asarray(load_cases[load_case])[np.asarray(loaded.supports)], 0.0
    )


@pytest.mark.parametrize("load_case", ["uniform", "half", "point"])
def test_every_load_points_down(loaded, load_case):
    load_cases = {
        "uniform": create_loads_uniform(loaded, 20_000.0),
        "half": create_loads_half_span(loaded, 20_000.0),
        "point": create_loads_point(loaded, 50_000.0, node=5),
    }
    applied = np.asarray(load_cases[load_case])
    assert np.all(applied[:, :2] == 0.0)
    assert np.all(applied[:, 2] <= 0.0)


def test_a_half_span_load_case_loads_one_half_more_than_the_other(loaded):
    applied = np.asarray(create_loads_half_span(loaded, 20_000.0, factor=0.5)[:, 2])
    along = np.asarray(loaded.nodes[:, 0])
    middle = 0.5 * (along.min() + along.max())

    free = np.asarray(applied != 0.0)
    assert np.all(np.abs(applied[free & (along <= middle)]) == 20_000.0)
    assert np.all(np.abs(applied[free & (along > middle)]) == 10_000.0)


def test_a_half_span_load_case_with_no_factor_leaves_one_half_bare(loaded):
    applied = np.asarray(create_loads_half_span(loaded, 20_000.0, factor=0.0)[:, 2])
    along = np.asarray(loaded.nodes[:, 0])
    middle = 0.5 * (along.min() + along.max())

    assert np.all(applied[along > middle] == 0.0)
    assert np.sum(applied[along <= middle]) < 0.0


def test_a_mirrored_half_span_load_case_loads_the_other_half(loaded):
    applied = np.asarray(create_loads_half_span(loaded, 20_000.0, mirrored=True)[:, 2])
    along = np.asarray(loaded.nodes[:, 0])
    middle = 0.5 * (along.min() + along.max())

    assert np.all(applied[along < middle] == 0.0)
    assert np.sum(applied[along >= middle]) < 0.0


def test_a_mirrored_half_span_load_case_is_the_reflection_of_the_unmirrored_one(loaded):
    near = np.asarray(create_loads_half_span(loaded, 20_000.0, factor=0.5)[:, 2])
    far = np.asarray(
        create_loads_half_span(loaded, 20_000.0, factor=0.5, mirrored=True)[:, 2]
    )

    assert np.allclose(near, far[::-1])
    assert np.isclose(near.sum(), far.sum())


@pytest.mark.parametrize("axis", [-1, 3])
def test_a_half_span_load_case_refuses_an_axis_that_is_not_a_dimension(loaded, axis):
    with pytest.raises(ValueError, match="axis"):
        create_loads_half_span(loaded, 20_000.0, axis=axis)


def test_a_point_load_case_loads_exactly_one_node(loaded):
    applied = np.asarray(create_loads_point(loaded, 50_000.0, node=5)[:, 2])

    assert np.count_nonzero(applied) == 1
    assert applied[5] == -50_000.0


def test_load_cases_add(loaded):
    # Cases are arrays and nothing else, so a distributed load with a point load
    # on top of it needs no generator of its own.
    combined = create_loads_uniform(loaded, 10_000.0) + create_loads_point(
        loaded, 50_000.0, node=5
    )
    total = float(jnp.sum(combined[:, 2]))

    assert total == pytest.approx(-(9 * 10_000.0 + 50_000.0))


def test_the_crown_is_the_highest_node(loaded):
    index = loaded.crown_node()

    assert isinstance(index, int)
    assert float(loaded.nodes[index, 2]) == float(jnp.max(loaded.nodes[:, 2]))
