# SPDX-License-Identifier: Apache-2.0
"""
Figures for the examples, in matplotlib and nothing else.

Every function takes arrays and returns a figure; nothing here calls `show`.
Member widths are drawn to a shared exaggeration rather than to scale, since a
tube one percent of the span wide would show nothing drawn truthfully.
"""

from collections.abc import Sequence
from typing import NamedTuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import Colormap
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from normax.design import Design
from normax.optimization import DescentHistory
from normax.structures import Structure

# Points of line width given to the thinnest and the thickest member drawn.
WIDTH_MIN = 1.8
WIDTH_MAX = 11.0

# Where the utilization colormap is cut out of plasma. Both ends of a full
# perceptual map run into the page: the dark end disappears on black and the
# light end on white, so the drawing is read through the middle of one, which
# stays a color on either ground.
STRESS_LOW = 0.28
STRESS_HIGH = 0.90

# The utilization the colors run between, pinned rather than read off the
# designs drawn: a member of a given color means the same worked fraction in
# every figure, across parametrizations and across runs.
UTILIZATION_FLOOR = 0.0
UTILIZATION_CAP = 1.0

# What the bar is labeled at. Quarters, against the ten or so a locator picks
# for a unit range, the bar being read for where a member sits rather than for
# a number.
UTILIZATION_TICKS = (0.0, 0.25, 0.5, 0.75, 1.0)

# Color of everything that is a reference rather than a result.
GREY = "0.55"

# The page every figure is drawn on, and the three tones of ink on it: the
# titles, what is read off an axis, and what only rules one.
GROUND = "#ffffff"
INK = "#1a1a1a"
MUTED = "#595959"
FAINT = "#8c8c8c"

# The colors a result is drawn in, in the order several are drawn.
SHADES = ("#31688e", "#35b779", "#c0392b")

# A node is a white disk with a dark rim, the way the plotters of JAX FDM and
# compas draw one: points across it, the rim's width, and its color.
NODE_SIZE = 4.5
NODE_RIM = 0.9
NODE_EDGE = INK

# A free node is a disk of the page's own color, a support the same disk
# filled in, so what is held reads without a legend on either ground.
NODE_FILL = GROUND

# A load case governs every member within this distance of the member's worst.
TIE_MARGIN = 1e-9

# Where a satisfied point is drawn, as a share of the smallest violation seen.
VIOLATION_DECADE = 0.1

# Longest walk whose every point is still marked rather than only drawn.
MARKED_STEPS = 40

# Inches of colorbar thickness, of the caption under it, and of the drawing
# width the vertical labels take.
BAR_THICKNESS = 0.12
BAR_CAPTION = 0.5
WIDTH_LABELS = 0.9

# Inches kept clear at the right edge, where the last tick label overhangs.
MARGIN_EDGE = 0.16

# Inches across one drawn design, and down the labels and bar around it.
WIDTH_DRAWING = 6.4
HEIGHT_DESIGN = 1.3

# Shortest a drawing is allowed to be, whatever the shape's proportions.
HEIGHT_DRAWING = 1.3

# Points across the open marker sitting on the first point of a round.
ROUND_MARK = 3.0

sampled = mpl.colormaps["plasma"](np.linspace(STRESS_LOW, STRESS_HIGH, 256))
UTILIZATION_MAP = ListedColormap(sampled, name="normax_utilization")


class DiameterRange(NamedTuple):
    """
    The diameters the thinnest and the thickest drawn line stand for.

    Attributes
    ----------
    narrowest :
        Diameter drawn at the least width.
    widest :
        Diameter drawn at the greatest width.

    Notes
    -----
    Read across every design a figure holds rather than per drawing, so two
    designs compare, and stretched rather than scaled: the drawn widths say
    where the material went, not how many millimeters across a member is. A
    design whose diameters differ by a tenth would be a row of identical lines
    drawn to scale, and the distribution is the thing being looked at.
    """

    narrowest: float
    widest: float


class DrawnStructure(NamedTuple):
    """
    A structure, and the widths its members are to be drawn at.

    Attributes
    ----------
    xyz :
        Position of every node. The X and Z coordinates are drawn.
    edges :
        The two node indices spanned by every member.
    diameters :
        Outer diameter of every member, setting its drawn width.
    width_scale :
        The diameters the least and the greatest drawn width stand for.
    supports :
        Indices of the nodes held in place, drawn filled in.
    """

    xyz: Float[Array, "nodes 3"]
    edges: Int[Array, "members 2"]
    diameters: Float[Array, "members"]
    width_scale: DiameterRange
    supports: Int[Array, "supports"]


class ColorRange(NamedTuple):
    """
    What members are colored by, and the range the colors span.

    Attributes
    ----------
    values :
        Quantity to color members by. If None, the diameters are used.
    vmin :
        Lower end of the range. If None, taken from the data.
    vmax :
        Upper end of the range. If None, taken from the data.
    cmap :
        The colormap the range is rendered through.
    """

    values: Float[Array, "members"] | None = None
    vmin: float | None = None
    vmax: float | None = None
    cmap: Colormap | str = UTILIZATION_MAP


def read_member_widths(
    diameters: Float[Array, "members"],
    scale: DiameterRange,
) -> Float[np.ndarray, "members"]:
    """
    Points of line width for every member, stretched across the drawn range.

    Parameters
    ----------
    diameters :
        Outer diameter of every member.
    scale :
        The diameters the least and the greatest width stand for.

    Returns
    -------
    widths :
        Width of every drawn member, in points.

    Notes
    -----
    The narrowest member of the figure comes out at the least width and the
    widest at the greatest, whatever the two diameters are, so a difference of
    a few percent is read at a glance. Where every member is one diameter the
    range is empty and they are all drawn at the middle width.
    """
    sizes = np.asarray(diameters)
    spanned = scale.widest - scale.narrowest
    if spanned <= 0.0:
        return np.full(sizes.shape, 0.5 * (WIDTH_MIN + WIDTH_MAX))

    share = (sizes - scale.narrowest) / spanned

    return WIDTH_MIN + (WIDTH_MAX - WIDTH_MIN) * share


def draw_members(
    ax: Axes,
    drawn: DrawnStructure,
    coloring: ColorRange = ColorRange(),
) -> LineCollection:
    """
    Draw a planar structure with every member as wide as its diameter.

    Parameters
    ----------
    ax :
        The axis to draw on.
    drawn :
        The structure and the widths its members are drawn at.
    coloring :
        What to color members by, and over what range.

    Returns
    -------
    members :
        The drawn collection, for a color bar to be attached to.

    Notes
    -----
    A node is drawn as a white disk with a dark rim over the members, which is
    how the plotters of JAX FDM and compas draw one: the rim reads against a
    member of any color, and the fill hides the joint where two members of
    different widths meet. A support is the same disk filled in, so what is
    held reads without a legend. The disk is sized in points rather than in the
    structure's own units, so it is the same size whatever the span drawn.
    """
    nodes = np.asarray(drawn.xyz)
    pairs = np.asarray(drawn.edges)
    sizes = np.asarray(drawn.diameters)

    starts = nodes[pairs[:, 0]][:, [0, 2]]
    ends = nodes[pairs[:, 1]][:, [0, 2]]
    segments = np.stack([starts, ends], axis=1)
    values = sizes if coloring.values is None else np.asarray(coloring.values)
    vmin = values.min() if coloring.vmin is None else coloring.vmin
    vmax = values.max() if coloring.vmax is None else coloring.vmax

    members = LineCollection(
        segments,
        linewidths=read_member_widths(sizes, drawn.width_scale),
        array=values,
        cmap=coloring.cmap,
        capstyle="round",
    )
    members.set_clim(vmin, vmax)
    ax.add_collection(members)

    held = np.asarray(drawn.supports)
    ax.plot(
        nodes[:, 0],
        nodes[:, 2],
        "o",
        ls="none",
        mfc=NODE_FILL,
        mec=NODE_EDGE,
        mew=NODE_RIM,
        ms=NODE_SIZE,
        zorder=3,
    )
    ax.plot(
        nodes[held, 0],
        nodes[held, 2],
        "o",
        ls="none",
        mfc=NODE_EDGE,
        mec=NODE_EDGE,
        mew=NODE_RIM,
        ms=NODE_SIZE,
        zorder=4,
    )
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_xlabel("x [mm]", fontsize=9)
    ax.set_ylabel("z [mm]", fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(labelsize=8)

    return members


def paint_figure(figure: Figure) -> None:
    """
    Paint a finished figure's page and chrome.

    Parameters
    ----------
    figure :
        The figure to paint, with every artist already on it.

    Notes
    -----
    Applied once a figure is built rather than through the global style, which
    would change the colors of every other plot in the importing process. Only
    the page and the chrome are painted here: the members, the curves and the
    colorbar read on a page of either color, the utilization map being the
    middle of a perceptual one rather than the whole of it, so a dark figure is
    the four constants above and nothing else.
    """
    figure.set_facecolor(GROUND)
    for ax in figure.axes:
        ax.set_facecolor(GROUND)
        ax.title.set_color(INK)
        ax.xaxis.label.set_color(INK)
        ax.yaxis.label.set_color(INK)
        ax.tick_params(colors=FAINT, labelcolor=MUTED, width=0.8, length=3)
        for spine in ax.spines.values():
            spine.set_color(FAINT)
        legend = ax.get_legend()
        if legend is None:
            continue
        for written in legend.get_texts():
            written.set_color(INK)


def draw_outline(
    ax: Axes,
    xyz: Float[Array, "nodes 3"],
    edges: Int[Array, "members 2"],
) -> LineCollection:
    """
    Draw a structure as a thin dashed line, for a result to be read against.

    Parameters
    ----------
    ax :
        The axis to draw on.
    xyz :
        Position of every node of the shape to outline.
    edges :
        The two node indices spanned by every member.

    Returns
    -------
    outline :
        The drawn collection, for a legend to be attached to.

    Notes
    -----
    Drawn member by member, so a structure that is not one chain outlines
    correctly, and without width, so it is not read as a second result.
    """
    nodes = np.asarray(xyz)
    pairs = np.asarray(edges)
    starts = nodes[pairs[:, 0]][:, [0, 2]]
    ends = nodes[pairs[:, 1]][:, [0, 2]]
    segments = np.stack([starts, ends], axis=1)

    outline = LineCollection(
        segments,
        linewidths=0.8,
        colors=GREY,
        linestyles="--",
        zorder=0,
    )
    ax.add_collection(outline)

    return outline


class UtilizationForm(NamedTuple):
    """
    One design to draw, and how hard the check works each of its members.

    Attributes
    ----------
    title :
        Name of the design, shown above its drawing.
    xyz :
        Position of every node.
    diameters :
        Outer diameter of every member.
    utilization :
        Demand over resistance of every member under every load case.
    """

    title: str
    xyz: Float[Array, "nodes 3"]
    diameters: Float[Array, "members"]
    utilization: Float[Array, "load_cases members"]


def count_governed_members(
    utilization: Float[Array, "load_cases members"],
) -> Int[np.ndarray, "load_cases"]:
    """
    How many members each load case governs, ties counted toward each case.

    Parameters
    ----------
    utilization :
        Demand over resistance of every member under every load case.

    Returns
    -------
    counts :
        Members per case, a tied member appearing under each of its cases.

    Notes
    -----
    Mirror-paired cases tie to solver precision on a symmetric design, and an
    argmax would hand every such tie to the lower index.
    """
    table = np.asarray(utilization)
    worst = table.max(axis=0)
    tied = table >= worst[None, :] - TIE_MARGIN

    return tied.sum(axis=1)


def read_drawing_height(
    across: tuple[float, float],
    upward: tuple[float, float],
    width: float,
) -> float:
    """
    How tall a drawing panel must be for the shape to sit in it undistorted.

    Parameters
    ----------
    across :
        The horizontal limits the drawing is held to.
    upward :
        The vertical limits the drawing is held to.
    width :
        Inches the panel is given across.

    Returns
    -------
    tall :
        Inches, floored so a very flat shape still gets a readable panel.

    Notes
    -----
    The drawing is at equal aspect, so a height chosen without reading the
    shape's proportions leaves the panel mostly empty and pushes whatever
    shares the page off it. A span twice as wide as it is high asks for half
    the width.
    """
    spanned = (upward[1] - upward[0]) / (across[1] - across[0])

    return max(spanned * width, HEIGHT_DRAWING)


def draw_utilization(
    structure: Structure,
    forms: Sequence[UtilizationForm],
    reference: Float[Array, "nodes 3"] | None = None,
) -> Figure:
    """
    The designs themselves, drawn down the page and colored by utilization.

    Parameters
    ----------
    structure :
        The structure the designs are of, read for its members and supports.
    forms :
        The designs to compare, in the order they are to be drawn.
    reference :
        Shape to outline behind the designs read against it, or None to draw
        none. The form it is the shape of is left without one.

    Returns
    -------
    figure :
        One drawing per design down the page, sharing a width scale, axis
        limits and a utilization colorbar.

    Notes
    -----
    The colorbar runs the whole unit range rather than the range the designs
    drawn happen to cover, so a color means one worked fraction in every figure
    and two runs are compared by eye.

    The reference is outlined the way an animation outlines the shape its walk
    left from: thin, dashed and grey behind the members, so the design in front
    of it is read against where it started without the two being read as two
    results. It is skipped on the drawing of the reference itself, which would
    otherwise outline a shape under itself.

    Stacked rather than set side by side, so a shape wider than it is tall gets
    the whole page to be wide across and the reader compares two designs at the
    same scale down one column. The drawings share an axis and only the lower
    is labeled along it, and the bar runs the width of both: the three read as
    one column rather than as three panels of different widths.
    """
    envelopes = [np.asarray(form.utilization).max(axis=0) for form in forms]
    widest = max(float(np.max(np.asarray(form.diameters))) for form in forms)
    thinnest = min(float(np.min(np.asarray(form.diameters))) for form in forms)
    scale = DiameterRange(thinnest, widest)
    rows = len(forms)

    shapes = [np.asarray(form.xyz) for form in forms]
    if reference is not None:
        shapes.append(np.asarray(reference))
    both = np.concatenate(shapes)
    margin = 0.05 * float(np.ptp(both[:, 0]))
    across = (float(both[:, 0].min()) - margin, float(both[:, 0].max()) + margin)
    upward = (float(both[:, 2].min()) - margin, float(both[:, 2].max()) + margin)

    spread = WIDTH_DRAWING - WIDTH_LABELS
    tall = read_drawing_height(across, upward, spread)
    reserved = BAR_THICKNESS + BAR_CAPTION
    height = rows * tall + HEIGHT_DESIGN + reserved
    figure, axes = plt.subplots(
        rows,
        1,
        figsize=(WIDTH_DRAWING, height),
        sharex=True,
        squeeze=False,
        layout="constrained",
    )
    drawings = axes[:, 0]

    edges = structure.edges
    for ax, form, envelope in zip(drawings, forms, envelopes):
        if reference is not None and form.xyz is not reference:
            outline = draw_outline(ax, reference, edges)
            outline.set_label("starting shape")
        drawn = DrawnStructure(
            form.xyz, edges, form.diameters, scale, structure.supports
        )
        coloring = ColorRange(envelope, UTILIZATION_FLOOR, UTILIZATION_CAP)
        members = draw_members(ax, drawn, coloring)
        ax.set_xlim(*across)
        ax.set_ylim(*upward)
        ax.set_title(form.title, fontsize=11)

    for ax in drawings[:-1]:
        ax.set_xlabel("")

    if reference is not None:
        drawings[0].legend(loc="lower center", fontsize=8, frameon=False)

    # The band the bar sits in is held out of the layout, and the bar is placed
    # in it against a settled drawing, which is what makes the widths agree.
    banded = reserved / height
    edged = MARGIN_EDGE / WIDTH_DRAWING
    figure.get_layout_engine().set(rect=(0.0, banded, 1.0 - edged, 1.0 - banded))
    figure.draw_without_rendering()

    box = drawings[-1].get_position()
    placed = (box.x0, BAR_CAPTION / height, box.width, BAR_THICKNESS / height)
    strip = figure.add_axes(placed)
    strip.set_in_layout(False)
    bar = figure.colorbar(members, cax=strip, orientation="horizontal")
    bar.set_ticks(list(UTILIZATION_TICKS))
    bar.set_label("utilization", fontsize=9)
    bar.outline.set_edgecolor(FAINT)
    bar.outline.set_linewidth(0.6)
    paint_figure(figure)

    return figure


def draw_governing_cases(
    forms: Sequence[UtilizationForm],
    names: tuple[str, ...],
) -> Figure:
    """
    How many members each load case governs, one panel per design.

    Parameters
    ----------
    forms :
        The designs to compare, in the order they are to be drawn.
    names :
        Name of every load case, in index order.

    Returns
    -------
    figure :
        One bar chart per design, all reading against the same count.

    Notes
    -----
    Its own figure rather than a row under the drawings: a bar chart wants a
    square panel and a shape wants a wide one, and the two together left the
    drawings squeezed and the bars stretched.
    """
    load_cases = len(names)
    columns = len(forms)

    figure, axes = plt.subplots(
        1,
        columns,
        figsize=(3.6 * columns, 3.4),
        squeeze=False,
        layout="constrained",
    )
    counted = axes[0]
    for ax in counted[1:]:
        ax.sharey(counted[0])

    for ax, form in zip(counted, forms):
        counts = count_governed_members(form.utilization)
        ax.bar(np.arange(load_cases), counts, 0.6, color=SHADES[0])
        ax.set_xticks(np.arange(load_cases))
        ax.set_xticklabels(names, fontsize=9)
        ax.set_ylabel("members governed")
        ax.set_title(form.title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    paint_figure(figure)

    return figure


class DescentTrace(NamedTuple):
    """
    One constrained descent, read at every point it was recorded at.

    Attributes
    ----------
    title :
        Name of the search, shown in the legend.
    history :
        Where it went, its objective and violation at every point.
    tolerance :
        Violation at or under which a point counts as a design.

    Notes
    -----
    The tolerance travels with the descent rather than with the drawing, being
    the budget the search was actually run under. The rows behind the violation
    are normalized where they are assembled, each family divided by its own
    scale, so the worst of them is comparable across families and no row
    dominates the measure by carrying larger units.
    """

    title: str
    history: DescentHistory
    tolerance: float


class DescentPanel(NamedTuple):
    """
    The descents drawn on one pair of axes, and what their two axes are called.

    Attributes
    ----------
    heading :
        Name and unit of the objective, from `name_objective`.
    axis :
        What one point of a curve is: a round, or an inner iteration.
    traces :
        The descents to compare, in the order they are drawn.
    """

    heading: str
    axis: str
    traces: tuple[DescentTrace, ...]


def read_round_bounds(history: DescentHistory) -> Int[np.ndarray, "rounds"]:
    """
    The points at which a walk crossed from one outer round into the next.

    Parameters
    ----------
    history :
        Where a descent went.

    Returns
    -------
    crossings :
        Index of the first point of every round after the first, empty on a
        walk that was recorded a round at a time.

    Notes
    -----
    Empty on the coarse walk by construction, every point there being its own
    round, so a drawing marks the crossings unconditionally and gets lines only
    where they say something.
    """
    numbered = np.asarray(history.round_index)
    if numbered.size == np.unique(numbered).size:
        return np.empty(0, dtype=int)

    return np.flatnonzero(np.diff(numbered)) + 1


def draw_round_starts(
    axis: Axes,
    steps: Int[np.ndarray, "rounds"],
    values: Float[np.ndarray, "rounds"],
    color: str,
    label: str | None,
) -> Line2D:
    """
    Open circles on the points at which the walk entered a new outer round.

    Parameters
    ----------
    axis :
        Where the curve being marked is drawn.
    steps :
        Position along the curve of every crossing to mark.
    values :
        The curve's value at each of them.
    color :
        Color of the curve the marks belong to.
    label :
        Legend entry, given to the bottom panel's marks alone, the panels
        sharing an axis and the marks standing at the same crossings on both.

    Returns
    -------
    marked :
        The marks, for an animation to reveal a prefix of.

    Notes
    -----
    Unfilled and in the curve's own color, so a mark reads as a point of the
    curve rather than as a second series, and so a curve already carrying a
    marker at every point still shows its crossings through the ring. Marks on
    the curve replaced rules across the panel, which crossed the gridlines at
    unrelated places and read as an axis rather than as an event.
    """
    (marked,) = axis.plot(
        steps,
        values,
        "o",
        ls="none",
        mfc="none",
        mec=color,
        mew=1.1,
        ms=ROUND_MARK,
        label=label,
    )

    return marked


def track_best_feasible(
    objective: Float[np.ndarray, "rounds"],
    violation: Float[np.ndarray, "rounds"],
    tolerance: float,
) -> Float[np.ndarray, "rounds"]:
    """
    The best objective reached at a satisfied round, up to and including each.

    Parameters
    ----------
    objective :
        Objective at the end of every round.
    violation :
        Worst violation over the rows at the end of every round.
    tolerance :
        Violation at or under which a round counts as a design.

    Returns
    -------
    best :
        The running best, and `nan` at every round before the first satisfied
        one, where no design has been found yet.

    Notes
    -----
    A running minimum over the satisfied rounds only, which is monotone by
    construction and answers what the raw column cannot: what the search would
    have handed over had it stopped here. The gap before the first satisfied
    round is the honest reading of a search still crossing the infeasible
    region, and drawing it as `nan` leaves it a gap rather than a floor.
    Prefix-closed, so the value at a round depends on no round after it: one
    call on the whole run and a frame is the curve sliced to it.
    """
    values = np.asarray(objective, dtype=np.float64)
    gaps = np.asarray(violation, dtype=np.float64)
    admitted = np.where(gaps <= tolerance, values, np.inf)
    running = np.minimum.accumulate(admitted)
    best = np.where(np.isfinite(running), running, np.nan)

    return best


def read_violation_floor(traces: Sequence[DescentTrace]) -> float:
    """
    The value an exactly satisfied point is drawn at on a logarithmic axis.

    Parameters
    ----------
    traces :
        The descents about to be drawn.

    Returns
    -------
    floor :
        One decade below the smaller of the least violation any of them
        actually reached and the tightest tolerance any of them was run under.

    Notes
    -----
    A satisfied point sits at zero, which no logarithmic axis can place.
    Clipping to a decade below puts it off the bottom of the data, where it
    reads as satisfied, and distorts no value that was really attained. The
    tolerance is taken into the floor as well, since a run that lands cleanly
    can leave every violation it reached above its own tolerance, and a
    tolerance line drawn off the bottom of the axis says nothing.
    """
    columns = [np.asarray(trace.history.violations) for trace in traces]
    gaps = np.concatenate(columns)
    positive = gaps[gaps > 0.0]
    tightest = min(trace.tolerance for trace in traces)
    if positive.size == 0:
        reached = tightest
    else:
        reached = float(positive.min())

    return min(reached, tightest) * VIOLATION_DECADE


def draw_objective_descent(panel: DescentPanel) -> Figure:
    """
    A constrained descent as the two curves that explain each other.

    Parameters
    ----------
    panel :
        The descents to draw, and what their two axes are called.

    Returns
    -------
    figure :
        Violation against the panel's axis on a logarithmic scale over the
        objective against the same, shared so a rise in one is read against
        the fall in the other.

    Notes
    -----
    The objective is the one the search read, free to rise while feasibility is
    bought, rather than the best satisfied point so far: the running best is
    monotone but undefined until the first satisfied round and flat across every
    round that improves nothing, which broke it into pieces that read as a
    different search on each. Where a walk was recorded finer than a round, the
    first point of every round after the first carries an open marker, so a step
    in the curve is read against the multiplier update that caused it. Every
    limit is set from the whole run rather than from what is drawn, so the same
    axes hold while a frame reveals a prefix of the curve.
    """
    proportions = (1.0, 1.6)
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(6.0, 5.6),
        sharex=True,
        height_ratios=proportions,
        layout="constrained",
    )
    violated, descent = axes
    shades = SHADES
    floor = read_violation_floor(panel.traces)

    named = False
    for index, trace in enumerate(panel.traces):
        color = shades[index % len(shades)]
        history = trace.history
        values = np.asarray(history.objectives)
        gaps = np.asarray(history.violations)
        steps = np.arange(values.size)
        marks = "-o" if values.size <= MARKED_STEPS else "-"
        placed = np.maximum(gaps, floor)

        violated.plot(steps, placed, marks, color=color, lw=1.4, ms=1.8)
        descent.plot(
            steps, values, marks, color=color, lw=1.6, ms=1.8, label=trace.title
        )

        crossings = read_round_bounds(history)
        entry = None if named or crossings.size == 0 else "round start"
        named = named or crossings.size > 0
        draw_round_starts(violated, steps[crossings], placed[crossings], color, None)
        draw_round_starts(descent, steps[crossings], values[crossings], color, entry)

    levels = sorted({trace.tolerance for trace in panel.traces})
    violated.axhspan(floor, levels[0], color=GREY, alpha=0.15, lw=0.0)
    for order, level in enumerate(levels):
        titled = "tolerance" if order == 0 else None
        violated.axhline(level, color=GREY, ls="--", lw=1.0, label=titled)

    spent = max(np.size(trace.history.objectives) for trace in panel.traces) - 1
    violated.set_xlim(0, max(spent, 1))
    violated.set_yscale("log")
    violated.set_ylim(bottom=floor)
    minimized = panel.heading.split(" [")[0]
    headline = f"Constrained {minimized} minimization"
    violated.set_ylabel("constraints violation")
    violated.set_title(headline, fontsize=11)
    violated.legend(frameon=False, fontsize=9)
    violated.grid(alpha=0.3, which="both")

    descent.set_xlabel(panel.axis)
    descent.set_ylabel(panel.heading)
    descent.legend(frameon=False, fontsize=9)
    descent.grid(alpha=0.3)
    paint_figure(figure)

    return figure


class DrawnFigures(NamedTuple):
    """
    The figures a run writes, one field per file.

    Attributes
    ----------
    designs :
        The designs colored by utilization, or None where none carries one.
    load_cases :
        Members governed per load case, or None on the same condition.
    optimization :
        The constrained descent, drawn whatever the pipeline held.
    """

    designs: Figure | None
    load_cases: Figure | None
    optimization: Figure


def draw_design_figures(
    structure: Structure,
    designs: dict[str, Design],
    case_names: tuple[str, ...],
    panel: DescentPanel,
) -> DrawnFigures:
    """
    The three figures a run draws: its designs, who governs them, the descent.

    Parameters
    ----------
    structure :
        The structure supplying the connectivity.
    designs :
        The designs to draw, keyed by the name each appears under.
    case_names :
        Name of every load case, naming who governs each member.
    panel :
        The descent to draw, and what its objective is called.

    Returns
    -------
    figures :
        One field per file the run writes.

    Notes
    -----
    The designs are drawn in the order they are given and the last of them is
    outlined behind the rest, so a run naming its answer first and its start
    last gets the start dashed in behind the answer.

    A design whose pipeline carried no check is left out, and where that leaves
    nothing to draw both design figures are None rather than empty —
    `draw_utilization` reads a widest diameter and a least-worked member across
    the designs, and neither exists over no designs. The descent figure is drawn
    either way, its curve being whatever the search minimized.
    """
    forms = []
    for title, design in designs.items():
        if design.sizes is None:
            continue
        form = UtilizationForm(
            title,
            design.shape.xyz,
            design.sizes.sections.diameter,
            design.sizes.utilization,
        )
        forms.append(form)

    # The last design given is the one the others are read against, and the
    # run names its start last.
    started = forms[-1].xyz if len(forms) > 1 else None
    drawn = draw_utilization(structure, forms, started) if forms else None
    governed = draw_governing_cases(forms, case_names) if forms else None
    descended = draw_objective_descent(panel)

    return DrawnFigures(drawn, governed, descended)
