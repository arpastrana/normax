# SPDX-License-Identifier: Apache-2.0
"""The shared bridge problem drawing."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.patches import FancyArrow
from matplotlib.patches import Polygon

from normax.loads import create_load_deck
from normax.loads import create_load_deck_half
from normax.loads import create_load_deck_point
from normax.loads import create_load_sector
from normax.loads import create_load_tributary
from normax.structures import build_arch_2d
from normax.structures import build_gridshell_3d
from normax.visualization import draw_problem_plan
from normax.visualization import draw_problem_setup


@pytest.fixture
def bridge():
    """A flat, evenly divided span with a node at midspan."""
    return build_arch_2d(num_edges=10, span=10_000.0, rise=0.0)


@pytest.fixture
def cases(bridge):
    """The three load patterns shared by the planar examples."""
    return np.stack(
        [
            create_load_deck(bridge, 180_000.0),
            create_load_deck_half(bridge, 90_000.0),
            create_load_deck_point(bridge, 90_000.0),
        ]
    )


def test_problem_setup_stacks_complete_load_case_drawings(bridge, cases):
    names = ("Uniform", "Asymmetric", "Midspan point")
    figure = draw_problem_setup(
        bridge,
        cases,
        names,
        r"$L = 10\,\mathrm{m}$",
        person_height=1750.0,
    )

    assert len(figure.axes) == 3
    assert [ax.get_title(loc="left") for ax in figure.axes] == [
        "(1)  Uniform",
        "(2)  Asymmetric",
        "(3)  Midspan point",
    ]
    assert all(not ax.get_frame_on() for ax in figure.axes)
    assert len({ax.get_xlim() for ax in figure.axes}) == 1
    assert len({ax.get_ylim() for ax in figure.axes}) == 1
    assert all(
        any(line.get_gid() == "problem-midspan" for line in ax.lines)
        for ax in figure.axes
    )
    assert any(artist.get_gid() == "problem-span" for artist in figure.axes[-1].texts)

    expected_arrows = [np.count_nonzero(np.linalg.norm(load, axis=1)) for load in cases]
    for ax, expected in zip(figure.axes, expected_arrows, strict=True):
        arrows = [patch for patch in ax.patches if isinstance(patch, FancyArrow)]
        supports = [patch for patch in ax.patches if type(patch) is Polygon]
        support_nodes = [
            line
            for line in ax.lines
            if line.get_marker() == "o" and line.get_markerfacecolor() == "#1a1a1a"
        ]
        assert len(arrows) == expected
        assert len(supports) == 2
        assert len(support_nodes) == 1
        assert len(support_nodes[0].get_xdata()) == 2
        assert all(np.min(arrow.get_xy()[:, 1]) > 0.0 for arrow in arrows)
        people = [patch for patch in ax.patches if patch.get_gid() == "problem-person"]
        assert len(people) == 2
        heights = np.concatenate([patch.get_path().vertices[:, 1] for patch in people])
        assert np.isclose(heights.max(), 1750.0)
        assert np.isclose(heights.min(), 0.0)

    plt.close(figure)


def test_problem_setup_requires_one_name_per_case(bridge, cases):
    with pytest.raises(ValueError, match="one entry per load case"):
        draw_problem_setup(bridge, cases, ("only one",))


def test_problem_setup_offers_a_horizontal_slide_layout(bridge, cases):
    names = ("Uniform", "Asymmetric", "Midspan point")
    figure = draw_problem_setup(bridge, cases, names, layout="horizontal")

    positions = [ax.get_position() for ax in figure.axes]
    assert len(positions) == 3
    assert positions[0].x0 < positions[1].x0 < positions[2].x0
    assert all(np.isclose(position.y0, positions[0].y0) for position in positions)
    plt.close(figure)


def test_problem_setup_turns_a_solid_shell():
    shell = build_gridshell_3d(4, 8, 5000.0, 2000.0, False, False)
    loaded = np.zeros((shell.num_nodes, 3))
    free = np.setdiff1d(np.arange(shell.num_nodes), np.asarray(shell.supports))
    loaded[free, 2] = -1.0
    figure = draw_problem_setup(shell, loaded[None], ("Uniform",), layout="horizontal")

    ax = figure.axes[0]
    arrows = [patch for patch in ax.patches if isinstance(patch, FancyArrow)]
    pins = [patch for patch in ax.patches if type(patch) is Polygon]
    assert len(arrows) == free.size
    assert not pins
    plt.close(figure)


def test_problem_plan_shades_a_level_case_whole_and_a_drift_as_its_sector():
    shell = build_gridshell_3d(4, 8, 5000.0, 2000.0, False, False)
    tributary = create_load_tributary(shell, 1.0e-3)
    drift = create_load_sector(shell, 1.0e-3, center=2, spokes=3, factor=0.5)
    cases = np.stack([tributary, drift])
    figure = draw_problem_plan(shell, cases, ("Tributary", "Drift"))

    free = np.setdiff1d(np.arange(shell.num_nodes), np.asarray(shell.supports))
    level, drifted = figure.axes
    level_cells = [
        patch for patch in level.patches if patch.get_gid() == "problem-cell"
    ]
    drift_cells = [
        patch for patch in drifted.patches if patch.get_gid() == "problem-cell"
    ]
    assert len(level_cells) == free.size
    # Three spokes wide over the three free rings, plus the crown a sector always holds.
    assert len(drift_cells) == 3 * 3 + 1
    plt.close(figure)
