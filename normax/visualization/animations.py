# SPDX-License-Identifier: Apache-2.0
"""
A descent as it happened: the design moving beside the curve that scored it.

Every frame is one recorded point of the walk, carried back through the
pipeline for the design it stands for and drawn against the curve revealed to
exactly that point. Nothing is smoothed, interpolated or resampled, so a frame
is a design the search really evaluated.
"""

from pathlib import Path
from typing import NamedTuple

import imageio_ffmpeg
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int
from matplotlib.animation import FFMpegWriter
from matplotlib.animation import FuncAnimation

from normax.design import DesignProblem
from normax.design import create_design
from normax.optimization import DescentHistory
from normax.visualization.plots import FAINT
from normax.visualization.plots import GREY
from normax.visualization.plots import INK
from normax.visualization.plots import SHADES
from normax.visualization.plots import UTILIZATION_CAP
from normax.visualization.plots import UTILIZATION_FLOOR
from normax.visualization.plots import UTILIZATION_TICKS
from normax.visualization.plots import ColorRange
from normax.visualization.plots import DescentPanel
from normax.visualization.plots import DiameterRange
from normax.visualization.plots import DrawnStructure
from normax.visualization.plots import draw_members
from normax.visualization.plots import draw_outline
from normax.visualization.plots import draw_round_starts
from normax.visualization.plots import paint_figure
from normax.visualization.plots import read_drawing_height
from normax.visualization.plots import read_member_widths
from normax.visualization.plots import read_round_bounds
from normax.visualization.plots import read_violation_floor

# Frames a second the walk is played back at.
FRAMES_RATE = 8

# Frames the answer is held on at the end, so the last design can be read.
FRAMES_HELD = 12

# Most frames a walk is drawn at, whatever its length. A descent of two
# thousand iterations is a real record and an unwatchable film, so a long walk
# is sampled at an even stride rather than drawn point by point. The curves
# still carry every point; it is the design and the head that jump.
FRAMES_MOST = 240

# Color the curves are drawn in, matching the single trace of a still figure.
SHADE_DRAWN = SHADES[0]

# Inches across every panel, and down the two curves and the page's margins.
WIDTH_FIGURE = 6.4
HEIGHT_VIOLATION = 1.3
HEIGHT_OBJECTIVE = 1.8
HEIGHT_MARGINS = 1.6


class WalkedDesigns(NamedTuple):
    """
    What every design a descent passed through is drawn from, and its scales.

    Attributes
    ----------
    shapes :
        Node positions at every recorded point, in the order reached.
    diameters :
        Outer diameter of every member at every point.
    envelopes :
        Worst utilization over the load cases of every member at every point.
    width_scale :
        Least and largest diameter over the whole walk, which set the widths
        the members are drawn at.

    Notes
    -----
    The three columns are read off the designs once rather than per frame, and
    the width scale over the whole walk rather than per frame: a width that
    rescaled as the animation ran would show the same member changing when only
    the extremes around it had moved. The colors need no such scale, running
    the whole unit range in every drawing.
    """

    shapes: tuple[Float[Array, "nodes 3"], ...]
    diameters: tuple[Float[Array, "members"], ...]
    envelopes: tuple[Float[Array, "members"], ...]
    width_scale: DiameterRange


def rebuild_walk(problem: DesignProblem, history: DescentHistory) -> WalkedDesigns:
    """
    Carry every point of a recorded walk back through the pipeline.

    Parameters
    ----------
    problem :
        The problem the walk belongs to, supplying the blocks and the loads.
    history :
        Where the descent went, read for its iterates.

    Returns
    -------
    walked :
        What every design is drawn from, and the scales they share.

    Raises
    ------
    ValueError
        If the pipeline carries no check, an animation colored by utilization
        having nothing to color by and no honest stand-in for it.

    Notes
    -----
    The pipeline is deterministic in its parameters, so this reconstructs the
    designs the search really saw rather than approximating them. One forward
    pass a frame, and no gradient.
    """
    if problem.pipeline.sizer is None:
        raise ValueError(
            "this pipeline carries no check, so its members have no utilization "
            "to be colored by and no sections to be drawn at"
        )

    shapes = []
    diameters = []
    envelopes = []
    for step in history.iterates:
        design = create_design(problem, step)
        sizes = design.sizes
        if sizes is None:
            raise ValueError("the pipeline returned no sections at a recorded point")
        shapes.append(np.asarray(design.shape.xyz))
        diameters.append(np.asarray(sizes.sections.diameter))
        envelopes.append(np.asarray(sizes.utilization).max(axis=0))

    widest = max(float(column.max()) for column in diameters)
    thinnest = min(float(column.min()) for column in diameters)
    scale = DiameterRange(thinnest, widest)
    walked = WalkedDesigns(tuple(shapes), tuple(diameters), tuple(envelopes), scale)

    return walked


def pick_frames(count: int) -> Int[np.ndarray, "frames"]:
    """
    Which points of a walk are drawn, thinned to a watchable number.

    Parameters
    ----------
    count :
        How many points the walk holds.

    Returns
    -------
    picked :
        Indices into the walk, in order, the last point always among them.

    Notes
    -----
    An even stride rather than a truncation, so the whole descent is seen and
    only its resolution drops. The answer is kept whatever the stride leaves,
    since a film of a search that stops short of what it found is a lie about
    the search.
    """
    if count <= FRAMES_MOST:
        return np.arange(count)

    stride = int(np.ceil(count / FRAMES_MOST))
    picked = np.arange(0, count, stride)
    if int(picked[-1]) != count - 1:
        picked = np.append(picked, count - 1)

    return picked


def read_drawn_bounds(
    walked: WalkedDesigns,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    The horizontal and vertical limits holding every shape a descent reached.

    Parameters
    ----------
    walked :
        Every design the descent passed through.

    Returns
    -------
    limits :
        The horizontal pair and the vertical pair, each with a margin.

    Notes
    -----
    Read over the whole walk, so the drawing never pans or zooms under the
    shape: a node that appears to move is one that moved.
    """
    every = np.concatenate(walked.shapes)
    margin = 0.05 * float(np.ptp(every[:, 0]))
    across = (float(every[:, 0].min()) - margin, float(every[:, 0].max()) + margin)
    upward = (float(every[:, 2].min()) - margin, float(every[:, 2].max()) + margin)

    return across, upward


def read_objective_bounds(values: Float[Array, "steps"]) -> tuple[float, float]:
    """
    The vertical limits the objective curve is drawn between.

    Parameters
    ----------
    values :
        Objective at every point of the walk.

    Returns
    -------
    limits :
        The pair, padded so neither the start nor the answer sits on a spine.
    """
    lowest = float(np.min(values))
    highest = float(np.max(values))
    padding = 0.05 * max(highest - lowest, abs(highest))

    return lowest - padding, highest + padding


def name_frame(panel: DescentPanel, history: DescentHistory, frame: int) -> str:
    """
    What a frame is called: where it sits in the walk, and in which round.

    Parameters
    ----------
    panel :
        The descent being animated, read for what one point is called.
    history :
        Where the descent went, read for the round each point came out of.
    frame :
        Index of the point being drawn.

    Returns
    -------
    title :
        The point and the round it came out of, or the point alone where the
        walk was recorded a round at a time and the two would say one thing.
    """
    if panel.axis == "round":
        return f"round {frame}"

    numbered = np.asarray(history.round_index)

    return f"{panel.axis} {frame}, round {int(numbered[frame])}"


def animate_descent(problem: DesignProblem, panel: DescentPanel) -> FuncAnimation:
    """
    The design at every point of a descent, beside the curve that scored it.

    Parameters
    ----------
    problem :
        The problem the descent ran on, supplying the structure and the blocks.
    panel :
        The descent to animate, and what its two axes are called. Exactly one
        trace, an animation having one design to draw at a time.

    Returns
    -------
    played :
        The animation, for a writer to save or a notebook to show.

    Raises
    ------
    ValueError
        If the panel carries anything other than a single trace.

    Notes
    -----
    Three panels: the design, the violation on a logarithmic axis, and the
    objective, the lower two sharing the axis the walk is indexed by, each
    carrying a head at the point being drawn. Every limit, width and color
    scale is set from the whole walk before the first frame, so what moves
    between frames is the design and the curves alone.
    The colors are capped at full utilization, so an overworked member of an
    infeasible frame saturates rather than rescaling the whole walk around it.

    A walk longer than `FRAMES_MOST` is sampled at an even stride: the curves
    still show every point and the head jumps along them, which keeps a long
    descent watchable and keeps one pipeline call per frame rather than one
    per iterate.
    """
    if len(panel.traces) != 1:
        raise ValueError(f"an animation draws one descent, not {len(panel.traces)}")

    trace = panel.traces[0]
    history = trace.history
    # The curves carry every point; only the designs are rebuilt and drawn at
    # the picked ones, which is where the cost and the running time both sit.
    picked = pick_frames(int(np.size(history.objectives)))
    drawn_walk = DescentHistory(
        history.iterates[picked],
        history.objectives[picked],
        history.violations[picked],
        history.round_index[picked],
    )
    walked = rebuild_walk(problem, drawn_walk)
    across, upward = read_drawn_bounds(walked)

    values = np.asarray(history.objectives)
    gaps = np.asarray(history.violations)
    crossings = read_round_bounds(history)
    floor = read_violation_floor(panel.traces)
    placed = np.maximum(gaps, floor)
    steps = np.arange(values.size)
    edges = problem.structure.edges
    started = walked.shapes[0]

    tall = read_drawing_height(across, upward, WIDTH_FIGURE)
    proportions = (tall, HEIGHT_VIOLATION, HEIGHT_OBJECTIVE)
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(WIDTH_FIGURE, sum(proportions) + HEIGHT_MARGINS),
        height_ratios=proportions,
        layout="constrained",
    )
    drawing, violated, descent = axes
    violated.sharex(descent)
    violated.tick_params(labelbottom=False)

    outline = draw_outline(drawing, started, edges)
    outline.set_label("starting shape")
    supports = problem.structure.supports
    blank = DrawnStructure(
        started, edges, np.zeros(len(edges)), walked.width_scale, supports
    )
    coloring = ColorRange(None, UTILIZATION_FLOOR, UTILIZATION_CAP)
    members = draw_members(drawing, blank, coloring)

    # draw_members plots the nodes and then the supports, and both move with
    # the design.
    dotted, seated = drawing.lines[-2], drawing.lines[-1]
    drawing.set_xlim(*across)
    drawing.set_ylim(*upward)
    drawing.legend(loc="lower center", fontsize=8, frameon=False)

    # Above rather than beside, so the drawing keeps the width of the curves
    # under it and one iteration sits at one place down the whole page.
    bar = figure.colorbar(
        members, ax=drawing, location="top", fraction=0.09, pad=0.03, aspect=44
    )
    bar.set_ticks(list(UTILIZATION_TICKS))
    bar.set_label("utilization", fontsize=9)
    bar.outline.set_edgecolor(FAINT)
    bar.outline.set_linewidth(0.6)
    named = drawing.text(
        0.015, 0.93, "", transform=drawing.transAxes, fontsize=10, va="top", color=INK
    )

    violated.axhspan(floor, trace.tolerance, color=GREY, alpha=0.15, lw=0.0)
    violated.axhline(trace.tolerance, color=GREY, ls="--", lw=1.0, label="tolerance")
    (walking,) = violated.plot([], [], "-", color=SHADE_DRAWN, lw=1.4)
    entered = draw_round_starts(violated, [], [], SHADE_DRAWN, None)
    (standing,) = violated.plot([], [], "o", color=SHADE_DRAWN, ms=4.5)
    violated.set_xlim(0, max(values.size - 1, 1))
    violated.set_yscale("log")
    violated.set_ylim(floor, float(placed.max()) * 2.0)
    violated.set_ylabel("constraints violation")
    violated.legend(frameon=False, fontsize=9)
    violated.grid(alpha=0.3, which="both")

    (reading,) = descent.plot([], [], "-", color=SHADE_DRAWN, lw=1.6, label=trace.title)
    started_round = "round start" if crossings.size > 0 else None
    rounding = draw_round_starts(descent, [], [], SHADE_DRAWN, started_round)
    (sitting,) = descent.plot([], [], "o", color=SHADE_DRAWN, ms=4.5)
    descent.set_ylim(*read_objective_bounds(values))
    descent.set_xlabel(panel.axis)
    descent.set_ylabel(panel.heading)
    descent.legend(frameon=False, fontsize=9, loc="upper right")
    descent.grid(alpha=0.3)
    paint_figure(figure)

    pairs = np.asarray(edges)

    def draw_frame(frame: int) -> None:
        held = min(frame, picked.size - 1)
        reached = int(picked[held])
        nodes = walked.shapes[held]
        starts = nodes[pairs[:, 0]][:, [0, 2]]
        ends = nodes[pairs[:, 1]][:, [0, 2]]
        segments = list(np.stack([starts, ends], axis=1))
        widths = read_member_widths(walked.diameters[held], walked.width_scale)

        members.set_segments(segments)
        members.set_linewidth(list(widths))
        members.set_array(walked.envelopes[held])
        dotted.set_data(nodes[:, 0], nodes[:, 2])
        seated.set_data(nodes[supports, 0], nodes[supports, 2])
        named.set_text(name_frame(panel, history, reached))

        shown = slice(0, reached + 1)
        entries = crossings[crossings <= reached]
        walking.set_data(steps[shown], placed[shown])
        reading.set_data(steps[shown], values[shown])
        entered.set_data(steps[entries], placed[entries])
        rounding.set_data(steps[entries], values[entries])

        # A head on each curve, so the first frame shows a point rather than a
        # line of one and the eye has the current design to follow.
        standing.set_data(steps[reached : reached + 1], placed[reached : reached + 1])
        sitting.set_data(steps[reached : reached + 1], values[reached : reached + 1])

    played = FuncAnimation(
        figure,
        draw_frame,
        frames=picked.size + FRAMES_HELD,
        interval=1000 // FRAMES_RATE,
        blit=False,
    )

    return played


def save_animation(played: FuncAnimation, path: Path) -> None:
    """
    Write an animation to disk as an MP4.

    Parameters
    ----------
    played :
        The animation to write.
    path :
        Where the file goes.

    Notes
    -----
    H.264 rather than a GIF, which carries a 256-color palette and one frame
    per file: the same walk lands an order of magnitude smaller and reads
    without banding, and a video is what a writeup embeds. The encoder is the
    binary `imageio-ffmpeg` ships as a wheel rather than one the machine is
    asked to have, so a run still writes its animation on every install.
    """
    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    # H.264 refuses an odd pixel dimension, which a figure sized in inches reaches.
    evened = "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white"
    settings = ["-vf", evened, "-pix_fmt", "yuv420p"]
    writer = FFMpegWriter(fps=FRAMES_RATE, extra_args=settings)

    played.save(path, writer=writer)
