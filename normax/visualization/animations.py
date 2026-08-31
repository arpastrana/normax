# SPDX-License-Identifier: Apache-2.0
"""
A descent as it happened: the design moving beside the curve that scored it.

Every frame is one recorded point of the walk, carried back through the
pipeline for the design it stands for and drawn against the curve revealed to
exactly that point. Nothing is smoothed, interpolated or resampled, so a frame
is a design the search really evaluated.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import NamedTuple

import imageio_ffmpeg
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int
from matplotlib.animation import FFMpegFileWriter
from matplotlib.animation import FuncAnimation

from normax.design import DesignProblem
from normax.design import compute_mass
from normax.design import create_design
from normax.optimization import DescentHistory
from normax.visualization.plots import FAINT
from normax.visualization.plots import GREY
from normax.visualization.plots import INK
from normax.visualization.plots import ISOMETRIC_AZIMUTH
from normax.visualization.plots import LEGEND_PLACE
from normax.visualization.plots import LEGEND_RIM
from normax.visualization.plots import SHADES
from normax.visualization.plots import SIZE_LABEL
from normax.visualization.plots import UTILIZATION_CAP
from normax.visualization.plots import UTILIZATION_FLOOR
from normax.visualization.plots import UTILIZATION_TICKS
from normax.visualization.plots import ColorRange
from normax.visualization.plots import DescentPanel
from normax.visualization.plots import DiameterRange
from normax.visualization.plots import DrawnLimits
from normax.visualization.plots import DrawnStructure
from normax.visualization.plots import capitalize_label
from normax.visualization.plots import clear_for_caption
from normax.visualization.plots import clear_for_legend
from normax.visualization.plots import draw_members
from normax.visualization.plots import draw_outline
from normax.visualization.plots import draw_round_starts
from normax.visualization.plots import paint_figure
from normax.visualization.plots import project_view
from normax.visualization.plots import read_drawing_height
from normax.visualization.plots import read_member_widths
from normax.visualization.plots import read_round_bounds
from normax.visualization.plots import read_violation_floor

# Frames a second the walk is played back at, and how long it plays for. The
# duration is prescribed and the frame count derived, not the other way round:
# a walk's length then sets the resolution a film is drawn at and never how
# long it lasts, so two runs of different lengths still play for the same time.
FRAMES_RATE = 24
SECONDS_PLAYED = 15.0
SECONDS_HELD = 1.5

# Both derived, so a change of rate keeps the seconds it was chosen for.
FRAMES_PLAYED = int(round(FRAMES_RATE * SECONDS_PLAYED))
FRAMES_HELD = int(round(FRAMES_RATE * SECONDS_HELD))

# Pixels an inch every frame is rendered at, stated rather than inherited. A
# backend reporting a device pixel ratio doubles the figure's own dpi, so the
# same call wrote 1280 across on a Retina screen and 640 on a headless machine:
# the film's resolution depended on where it was made. Matches the dpi the
# still figures are written at, so a frame and a figure carry the same detail.
ANIMATION_DPI = 200

# A GIF carries the film unchanged: every frame, at the video's own rate and
# resolution. Neither is reduced, so nothing can drift out of step with the
# video and nothing has to be traded against reading the axis labels. What that
# costs is bytes, and a GIF costs them by construction: one frame per file, a
# palette per frame, and no prediction between them.

# Color the curves are drawn in, matching the single trace of a still figure.
SHADE_DRAWN = SHADES[0]

# Inches across every panel, and down the two curves and the page's margins.
WIDTH_FIGURE = 6.4
HEIGHT_VIOLATION = 1.3
HEIGHT_OBJECTIVE = 1.8
HEIGHT_MARGINS = 1.6

# Points the two captions over the drawing are set at, and where they sit in
# the drawing's own coordinates. Two-fifths larger than the ten they were drawn
# at, which is what a caption read from across a room asks for: these are the
# only numbers on the page that say which design is on screen.
# Degrees between the views a swept framing is read at. Five is far finer than
# the silhouette changes over.
SWEPT_STEP = 5.0

SIZE_CAPTION = 14.0
CAPTION_LEFT = 0.015
CAPTION_RIGHT = 0.985
CAPTION_LEVEL = 0.93


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
    masses :
        Total mass of the design at every point.
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
    masses: tuple[float, ...]
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
    masses = []
    for step in history.iterates:
        design = create_design(problem, step)
        sizes = design.sizes
        if sizes is None:
            raise ValueError("the pipeline returned no sections at a recorded point")
        shapes.append(np.asarray(design.shape.xyz))
        diameters.append(np.asarray(sizes.sections.diameter))
        envelopes.append(np.asarray(sizes.utilization).max(axis=0))
        # Read off the design rather than off the recorded objective, which is
        # the mass only where mass is what the search minimized.
        masses.append(float(compute_mass(design)))

    widest = max(float(column.max()) for column in diameters)
    thinnest = min(float(column.min()) for column in diameters)
    scale = DiameterRange(thinnest, widest)
    walked = WalkedDesigns(
        tuple(shapes), tuple(diameters), tuple(envelopes), tuple(masses), scale
    )

    return walked


def count_played_frames(reach: int) -> int:
    """
    How many frames a walk of a given length is played over.

    Parameters
    ----------
    reach :
        Points the walk holds, or the longest walk it is drawn beside.

    Returns
    -------
    frames :
        The prescribed count, or the walk's own length where it is shorter.

    Notes
    -----
    Capped at the walk's length because a frame beyond it repeats a design
    already drawn: past that point a longer film carries no more of the search,
    only more bytes. A short walk therefore plays for less than the prescribed
    time rather than standing still through it.
    """
    return max(min(FRAMES_PLAYED, reach), 1)


def pace_frames(count: int, span: int) -> Int[np.ndarray, "frames"]:
    """
    Which points of a walk are drawn, paced against the longest beside it.

    Parameters
    ----------
    count :
        How many points this walk holds.
    span :
        How many points the longest walk drawn beside it holds. Its own count
        where it is drawn alone.

    Returns
    -------
    picked :
        Indices into this walk, one per frame, in order, the last point always
        among them.

    Notes
    -----
    Evenly spaced over the span rather than taken at an integer stride, which
    is what lets an exact frame count be hit and therefore an exact duration be
    prescribed. Every walk drawn against one span gets the same number of
    frames and frame `k` stands at the same iteration in all of them, so two
    films of one structure are read against each other frame for frame. A walk
    shorter than the span holds its answer for the rest of the schedule, which
    is what makes a short descent read as one that finished sooner rather than
    one that was cut off.
    """
    reach = max(span, count)
    frames = count_played_frames(reach)
    if frames <= 1:
        return np.zeros(1, dtype=int)

    reached = np.rint(np.arange(frames) * (reach - 1) / (frames - 1))

    return np.minimum(reached, count - 1).astype(int)


def pick_frames(count: int) -> Int[np.ndarray, "frames"]:
    """
    Which points of a walk drawn on its own are drawn.

    Parameters
    ----------
    count :
        How many points the walk holds.

    Returns
    -------
    picked :
        Indices into the walk, in order, the last point always among them.
    """
    return pace_frames(count, count)


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


def read_swept_bounds(
    walked: WalkedDesigns,
    across: tuple[float, float],
    upward: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Limits holding every design of a walk seen from every view of a turn.

    Parameters
    ----------
    walked :
        Every design the descent passed through, in three dimensions.
    across :
        Horizontal limits the standing view asked for.
    upward :
        Vertical limits the standing view asked for.

    Returns
    -------
    limits :
        The horizontal pair and the vertical pair, wide enough for any frame of
        any turn of this walk.

    Notes
    -----
    Every design against every view rather than the design each frame happens
    to pair its view with. Pairing would give a tighter box, and a different
    one for each walk: two baselines reaching their greatest height at
    different iterations would be framed differently and could not be set side
    by side, which is the whole point of holding a framing. Sampled every
    `SWEPT_STEP` degrees, since the silhouette varies smoothly with the view.
    Unioned with the standing box, so a film is never framed more tightly than
    the drawings beside it.
    """
    azimuths = np.arange(0.0, 360.0, SWEPT_STEP) + ISOMETRIC_AZIMUTH
    lows = []
    highs = []
    floors = []
    ceilings = []
    for shape in walked.shapes:
        for azimuth in azimuths:
            viewed = project_view(shape, float(azimuth))
            lows.append(float(viewed[:, 0].min()))
            highs.append(float(viewed[:, 0].max()))
            floors.append(float(viewed[:, 2].min()))
            ceilings.append(float(viewed[:, 2].max()))

    lowest = min(lows)
    highest = max(highs)
    margin = 0.05 * (highest - lowest)
    turned_across = (lowest - margin, highest + margin)
    turned_upward = (min(floors) - margin, max(ceilings) + margin)
    widened_across = (
        min(across[0], turned_across[0]),
        max(across[1], turned_across[1]),
    )
    widened_upward = (
        min(upward[0], turned_upward[0]),
        max(upward[1], turned_upward[1]),
    )

    return widened_across, widened_upward


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


class GifReduction(NamedTuple):
    """
    How much of a film's detail a GIF of it is allowed to keep.

    Attributes
    ----------
    width :
        Pixels across, the height following the film's proportions.
    rate :
        Frames a second, thinned from the film's own rate.
    colors :
        Entries in the palette generated from the film.

    Notes
    -----
    A GIF stores one palette index per pixel and compresses with LZW, which
    needs long runs of one byte to work on. A drawing of hundreds of
    antialiased members offers few, and a turning one offers no frame worth
    reusing either, so the file lands near a tenth of its raw pixel count
    whatever the palette does. Measured on the shell, the pixels are the whole
    story: dithering is worth about a tenth of the size and halving the width
    is worth three and a half times it. Width is therefore the lever in both
    directions -- it buys the size back, and it is the only thing that makes a
    member thinner than a pixel legible again -- and it is a deliberate trade
    of detail against bytes rather than a cleverer encoding.
    """

    width: int
    rate: float
    colors: int


# What a GIF meant for a page rather than for review is reduced to. Measured on
# the shell, whose film is 1280 wide and 127.8 MB as a faithful GIF: 27.1 MB at
# these numbers. Ten frames a second rather than twelve because a GIF states
# its delay in whole hundredths and only ten divides a hundred, so the reduced
# film runs the length of the one it came from instead of eight percent long.
GIF_FOR_READING = GifReduction(960, 10.0, 128)


def pace_azimuth(
    frames: int,
    turns: float,
) -> Float[np.ndarray, "frames"]:
    """
    The view each frame is taken from, spread over the whole film.

    Parameters
    ----------
    frames :
        How many frames the film runs for, held frames included.
    turns :
        Full revolutions to spin through. Zero returns the one standing view.

    Returns
    -------
    azimuths :
        Degrees around the upward axis, one per frame, opening on the standing
        view every drawing uses.

    Notes
    -----
    Spread over the whole film rather than over the walk, so a whole number of
    turns lands the last frame back on the first frame's view and a looping
    film closes on itself; the held frames at the end become a turntable of
    the answer rather than a freeze. Divided by the frame count rather than by
    one less, since frame zero and frame `frames` are the same view and only
    one of them is drawn.
    """
    standing = np.full(max(frames, 1), float(ISOMETRIC_AZIMUTH))
    if turns == 0.0:
        return standing

    spun = np.arange(max(frames, 1)) * 360.0 * turns / max(frames, 1)

    return standing + spun


def sort_by_depth(
    viewed: Float[np.ndarray, "nodes 3"],
    pairs: Int[np.ndarray, "members 2"],
) -> Int[np.ndarray, "members"]:
    """
    The order members are drawn in so the nearer ones cover the farther.

    Parameters
    ----------
    viewed :
        Projected node positions, whose second column measures toward the
        viewer.
    pairs :
        The two nodes of every member.

    Returns
    -------
    order :
        Members from farthest to nearest, which is the order to draw them.

    Notes
    -----
    A line collection draws in the order it is given and carries no depth of
    its own, so without this the member that happens to come first in the
    edge list stays in front however the view turns. Sorted on each member's
    midpoint, which is a painter's order: it settles which of two members
    reads as nearer, and cannot cut one that genuinely crosses the other.

    Stable, so equal depths keep the edge list's own order: a planar walk has
    one depth throughout and must come out drawn exactly as it was before any
    of this, which an unstable sort would not guarantee.
    """
    depths = 0.5 * (viewed[pairs[:, 0], 1] + viewed[pairs[:, 1], 1])

    return np.argsort(depths, kind="stable")


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

    Notes
    -----
    Returned ready to draw, both words capitalized and split by a slash rather
    than a comma, so the caption reads as two counters and not as a sentence.
    A caller adds nothing to it.
    """
    counted = capitalize_label(panel.axis)
    if panel.axis == "round":
        return f"{counted} {frame}"

    numbered = np.asarray(history.round_index)

    return f"{counted} {frame} / Round {int(numbered[frame])}"


def animate_descent(
    problem: DesignProblem,
    panel: DescentPanel,
    limits: DrawnLimits | None = None,
    turns: float = 0.0,
) -> FuncAnimation:
    """
    The design at every point of a descent, beside the curve that scored it.

    Parameters
    ----------
    problem :
        The problem the descent ran on, supplying the structure and the blocks.
    panel :
        The descent to animate, and what its two axes are called. Exactly one
        trace, an animation having one design to draw at a time.
    limits :
        Limits to hold every panel to, or None to read them off this walk
        alone. Held, so several films of one structure share a framing, a pace
        and a set of ticks.
    turns :
        Full revolutions about the upward axis to spin the design through over
        the film. Zero draws the one standing view. A structure lying in one
        plane never turns whatever this says, having no depth to show.

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
    counted = int(np.size(history.objectives))
    span = None if limits is None else limits.steps
    walk_span = counted if span is None else max(int(span), counted)
    # One schedule for every walk drawn together, so frame k is iteration k in
    # each of them and a shorter descent holds its answer rather than ending.
    picked = pick_frames(counted) if span is None else pace_frames(counted, walk_span)
    drawn_walk = DescentHistory(
        history.iterates[picked],
        history.objectives[picked],
        history.violations[picked],
        history.round_index[picked],
    )
    walked = rebuild_walk(problem, drawn_walk)
    # The walk is kept in its own three dimensions and turned once per frame,
    # so a spin costs a projection rather than any geometry. A planar walk is
    # returned unchanged by the projection, which is what makes `turns`
    # harmless on a structure that cannot show a turn.
    azimuths = pace_azimuth(picked.size + FRAMES_HELD, turns)
    standing = walked._replace(shapes=[project_view(shape) for shape in walked.shapes])
    across, upward = read_drawn_bounds(standing)
    if limits is not None:
        across, upward = limits.across, limits.upward
    # Framed for every view the film passes through, not just the one it opens
    # on: a box read off the standing view alone would crop the design as it
    # came round. Only where the framing is this walk's own -- limits handed in
    # are held exactly, their caller having swept them over every baseline.
    if turns != 0.0 and limits is None:
        across, upward = read_swept_bounds(walked, across, upward)
    upward = clear_for_caption(clear_for_legend(upward))

    values = np.asarray(history.objectives)
    gaps = np.asarray(history.violations)
    crossings = read_round_bounds(history)
    floor = read_violation_floor(panel.traces)
    placed = np.maximum(gaps, floor)
    steps = np.arange(values.size)
    edges = problem.structure.edges
    started = standing.shapes[0]

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
    outline.set_label("Starting shape")
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
    entries = len(drawing.get_legend_handles_labels()[0])
    legend = drawing.legend(
        loc=LEGEND_PLACE,
        fontsize=8,
        frameon=True,
        framealpha=0.9,
        ncol=max(entries, 1),
    )
    legend.get_frame().set_linewidth(LEGEND_RIM)
    legend.get_frame().set_edgecolor(FAINT)

    # Above rather than beside, so the drawing keeps the width of the curves
    # under it and one iteration sits at one place down the whole page.
    bar = figure.colorbar(
        members, ax=drawing, location="top", fraction=0.09, pad=0.03, aspect=44
    )
    bar.set_ticks(
        list(UTILIZATION_TICKS),
        labels=[f"{tick:.1f}" for tick in UTILIZATION_TICKS],
    )
    bar.set_label("Utilization", fontsize=SIZE_LABEL)
    bar.outline.set_edgecolor(FAINT)
    bar.outline.set_linewidth(0.6)
    # What frame this is, and what the design on it weighs. Two captions rather
    # than one long line: the left is where the search stands and the right is
    # what it has bought, and neither grows into the other as the counts climb.
    named = drawing.text(
        CAPTION_LEFT,
        CAPTION_LEVEL,
        "",
        transform=drawing.transAxes,
        fontsize=SIZE_CAPTION,
        va="top",
        color=INK,
    )
    weighed = drawing.text(
        CAPTION_RIGHT,
        CAPTION_LEVEL,
        "",
        transform=drawing.transAxes,
        fontsize=SIZE_CAPTION,
        va="top",
        ha="right",
        color=INK,
    )

    (walking,) = violated.plot([], [], "-", color=SHADE_DRAWN, lw=1.4)
    entered = draw_round_starts(violated, [], [], SHADE_DRAWN, None)
    (standing,) = violated.plot([], [], "o", color=SHADE_DRAWN, ms=4.5)
    violated.set_xlim(0, max(walk_span - 1, 1))
    violated.set_yscale("log")
    if limits is None:
        violated.set_ylim(floor, float(placed.max()) * 2.0)
    else:
        violated.set_ylim(*limits.violation)

    # Shaded off the axis's own floor rather than this walk's, and after the
    # limits are set: where several films share an axis its floor is the least
    # of theirs, and a band drawn to a higher one leaves the satisfied region
    # looking open at the bottom.
    violated.axhspan(
        violated.get_ylim()[0], trace.tolerance, color=GREY, alpha=0.15, lw=0.0
    )
    violated.axhline(trace.tolerance, color=GREY, ls="--", lw=1.0, label="Tolerance")
    violated.set_ylabel("Constraints violation", fontsize=SIZE_LABEL)
    violated.legend(frameon=False, fontsize=9)
    violated.grid(alpha=0.3, which="both")

    (reading,) = descent.plot(
        [], [], "-", color=SHADE_DRAWN, lw=1.6, label=capitalize_label(trace.title)
    )
    started_round = "Round start" if crossings.size > 0 else None
    rounding = draw_round_starts(descent, [], [], SHADE_DRAWN, started_round)
    (sitting,) = descent.plot([], [], "o", color=SHADE_DRAWN, ms=4.5)
    descent.set_ylim(
        *(read_objective_bounds(values) if limits is None else limits.objective)
    )
    descent.set_xlabel(capitalize_label(panel.axis), fontsize=SIZE_LABEL)
    descent.set_ylabel(capitalize_label(panel.heading), fontsize=SIZE_LABEL)
    descent.legend(frameon=False, fontsize=9, loc="upper right")
    descent.grid(alpha=0.3)
    paint_figure(figure)

    pairs = np.asarray(edges)

    opened = walked.shapes[0]

    def draw_frame(frame: int) -> None:
        held = min(frame, picked.size - 1)
        reached = int(picked[held])
        azimuth = float(azimuths[min(frame, azimuths.size - 1)])
        nodes = project_view(walked.shapes[held], azimuth)
        # Farthest first, so a member that comes round to the back stops
        # covering the one in front of it.
        order = sort_by_depth(nodes, pairs)
        starts = nodes[pairs[order, 0]][:, [0, 2]]
        ends = nodes[pairs[order, 1]][:, [0, 2]]
        segments = list(np.stack([starts, ends], axis=1))
        widths = read_member_widths(walked.diameters[held], walked.width_scale)
        sorted_widths = list(np.asarray(widths)[order])
        sorted_colors = np.asarray(walked.envelopes[held])[order]

        members.set_segments(segments)
        members.set_linewidth(sorted_widths)
        members.set_array(sorted_colors)
        dotted.set_data(nodes[:, 0], nodes[:, 2])
        seated.set_data(nodes[supports, 0], nodes[supports, 2])

        # The ghost turns with the design: a start left standing while the
        # answer spins reads as two structures rather than one and its shadow.
        ghosted = project_view(opened, azimuth)
        ghost_starts = ghosted[pairs[:, 0]][:, [0, 2]]
        ghost_ends = ghosted[pairs[:, 1]][:, [0, 2]]
        ghost_segments = list(np.stack([ghost_starts, ghost_ends], axis=1))
        outline.set_segments(ghost_segments)
        named.set_text(name_frame(panel, history, reached))
        worked = float(np.mean(walked.envelopes[held]))
        weighed.set_text(f"Mass {walked.masses[held]:.3f} t / Utilization {worked:.3f}")

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

    # Drawn once here, before any writer grabs a frame. A constrained layout
    # solves at draw time and settles only after its first pass, so whatever is
    # grabbed on that pass carries a page laid out about a pixel off from every
    # frame after it: invisible in a video, a flick at a looping GIF's seam.
    # Done on the figure while it is still in hand rather than in the writer,
    # which would have to reach inside the animation for it.
    figure.draw_without_rendering()

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

    Frames go to disk as images and are encoded from there, rather than piped as
    raw bytes. A pipe carries no dimensions, so the writer states them as
    `int(size * dpi)` while the rasterizer rounds up, and a drawing sized from
    its own aspect ratio makes that product fractional: the two then disagree by
    a row, every frame is read one row out of step with the last, and the
    picture creeps down the screen. H.264 encodes that as a translation and
    hides it; a GIF does not.
    """
    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    # H.264 refuses an odd pixel dimension, which a figure sized in inches reaches.
    evened = "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white"
    settings = ["-vf", evened, "-pix_fmt", "yuv420p"]
    writer = FFMpegFileWriter(fps=FRAMES_RATE, extra_args=settings)

    played.save(path, writer=writer, dpi=ANIMATION_DPI)


def convert_to_gif(
    video: Path,
    path: Path,
    reduced: GifReduction | None = None,
) -> None:
    """
    A GIF of a video already written, narrowed and slowed to be worth its bytes.

    Parameters
    ----------
    video :
        The MP4 to read.
    path :
        Where the GIF goes.
    reduced :
        How far to narrow and thin the film, or None to keep every frame at
        full width -- faithful to the film, and on a drawing of any density
        far too heavy for a page to carry.

    Raises
    ------
    RuntimeError
        If the encoder refuses the conversion, carrying what it reported.

    Notes
    -----
    Every frame at the video's own rate and resolution, so the two play in
    step and the axis labels read as they do in the film. A GIF pays for that
    in bytes -- one frame per file, a palette per frame, no prediction between
    them -- and lands many times the size of the video it came from.

    A GIF states its delays in hundredths of a second, so a rate that does not
    divide a hundred cannot be written exactly. The nearest representable delay
    is used, which shifts the total slightly; the frames themselves are all
    there. The palette is generated from the walk rather than taken from a
    default that never saw it, which is the most that can be done about a
    256-color ramp.

    Converted from the written video rather than rendered again, because the
    frames cost a pipeline pass each and the transcode costs seconds: a GIF of
    a run already filmed asks for no part of the search to be repeated.
    """
    if reduced is None:
        palette = "split[a][b];[a]palettegen[p];[b][p]paletteuse"
    else:
        # Scaled and thinned before the palette is generated, so the palette
        # describes the frames that are actually written. Undithered, which on
        # a drawing is the sharper answer: both dithers lay a texture over the
        # near-white ground, and an ordered one reads as the very pixelation a
        # reduced GIF is accused of. The cost is mild banding along the
        # colorbar's ramp, the one smooth gradient on the page.
        shrunk = f"scale={reduced.width}:-1:flags=lanczos,fps={reduced.rate:g}"
        generated = f"palettegen=max_colors={reduced.colors}"
        used = "paletteuse=dither=none"
        palette = f"{shrunk},split[a][b];[a]{generated}[p];[b][p]{used}"
    asked = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vf",
        palette,
        "-loop",
        "0",
        str(path),
    ]
    finished = subprocess.run(asked, capture_output=True)
    if finished.returncode != 0:
        reported = finished.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"the encoder refused a GIF of {video}: {reported}")


def save_gif(played: FuncAnimation, path: Path) -> None:
    """
    Write an animation to disk as a GIF, by way of the video it comes from.

    Parameters
    ----------
    played :
        The animation to write.
    path :
        Where the GIF goes.

    Notes
    -----
    Not what a run writes: `save_animation` is, and this is for a reader that
    renders an image and not a video -- a README on a forge that strips the
    video tag, most often. Asked for rather than produced, since the bytes are
    real; `convert_to_gif` carries the reasoning and the reduction. Where the
    video is already on disk, call that instead and nothing is rendered twice.
    """
    with tempfile.TemporaryDirectory() as room:
        video = Path(room) / "walk.mp4"
        save_animation(played, video)
        convert_to_gif(video, path)
