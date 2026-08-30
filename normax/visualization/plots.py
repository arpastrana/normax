# SPDX-License-Identifier: Apache-2.0
"""
Figures for the examples, in matplotlib and nothing else.

Every function takes arrays and returns a figure; nothing here calls `show`.
Member widths are drawn to a shared exaggeration rather than to scale, since a
tube one percent of the span wide would show nothing drawn truthfully.
"""

from collections.abc import Sequence
from typing import Literal
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
from matplotlib.patches import PathPatch
from matplotlib.patches import Polygon
from matplotlib.path import Path

from normax.design import Design
from normax.optimization import DescentHistory
from normax.structures import Structure

# The typeface every figure is set in: Computer Modern, which matplotlib ships
# rather than asks the machine to have, so a figure reads like the text around
# it on any install and no LaTeX is required. `text.usetex` would render
# through a real LaTeX and look better still, at the cost of a toolchain a
# reader must install before a figure will draw at all.
#
# The ASCII hyphen is not a preference: cmr10 carries no U+2212, so leaving the
# unicode minus on prints a missing-glyph box on every negative tick, and this
# repository's axes are full of them.
SERIF_STYLE = {
    "font.family": "serif",
    "font.serif": ["cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
}

# Points an axis label and a title are set at. Both a third larger than the
# nine and eleven they were drawn at before, which is what a figure printed at
# column width rather than read on a screen asks for.
SIZE_LABEL = 11.7
SIZE_PANEL = 13.0
SIZE_TITLE = 14.3

# Applied on import, so every figure this module draws is set the same way
# whichever entry point asked for it. A caller wanting matplotlib's own look
# back restores it with `matplotlib.rcdefaults()`. The two sizes are set here
# as well as passed, since not every label states one.
plt.rcParams.update(SERIF_STYLE)
plt.rcParams.update({"axes.labelsize": SIZE_LABEL, "axes.titlesize": SIZE_TITLE})


def clear_for_legend(upward: tuple[float, float]) -> tuple[float, float]:
    """
    Vertical limits with room kept under the design for the legend.

    Parameters
    ----------
    upward :
        Least and greatest coordinate the design reaches.

    Returns
    -------
    cleared :
        The same pair, its lower bound dropped by `LEGEND_CLEARANCE` of the
        span.

    Notes
    -----
    A fraction of the span rather than a fixed distance, so it means the same
    thing at any scale, and applied to whatever limits arrive: where several
    runs share a framing they share the clearance too, and the aspect ratio
    stays equal across them.
    """
    low, high = upward
    dropped = low - LEGEND_CLEARANCE * (high - low)

    return dropped, high


def capitalize_label(text: str) -> str:
    """
    A label with its first letter capitalized and the rest left alone.

    Parameters
    ----------
    text :
        The label as it is written in the code.

    Returns
    -------
    capitalized :
        The same label, opening on a capital.

    Notes
    -----
    `str.capitalize` lowercases everything after the first character, which
    would turn a unit or a name into nonsense -- `mass [T]`, `Auglag` from
    `AUGLAG`. Only the first character is touched, so `constraints violation`
    becomes `Constraints violation`. A coordinate's own name is not a label to
    capitalize and is written out as it stands.
    """
    if not text:
        return text

    return text[0].upper() + text[1:]


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
# The ends and the middle, and nothing between: a bar read for whether a member
# is worked, half worked or spent wants three marks, not five, and one decimal
# says all any of them says.
UTILIZATION_TICKS = (0.0, 0.5, 1.0)

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

# The coordinate names, set as mathematics: the name italic and its unit
# upright, which is the convention and is what `\text` inside `$...$` gets.
# matplotlib's own mathtext, not a LaTeX installation, so a figure still draws
# on a machine that has none.
LABEL_ACROSS = r"$x~\text{[mm]}$"
LABEL_UPWARD = r"$z~\text{[mm]}$"

# Where a drawing's legend sits, and the hairline box around it. Centered at
# the foot and laid out in one row rather than stacked: a corner is clear on one
# structure and covered on the next, where a single row is a third the height of
# a stack and clears the foot of any of them. Boxed, since it sits over the
# drawing rather than under it.
LEGEND_PLACE = "lower center"
LEGEND_RIM = 0.5

# Share of a drawing's height kept clear beneath the design, so the legend sits
# under it rather than over it. The legend is a row about a twentieth of the
# height; this is more, because the lowest support is a disk with a rim and it
# is the thing a reader looks for first.
LEGEND_CLEARANCE = 0.12

# Points of line width the shape a search left from is outlined at. Two fifths
# lighter than the 0.8 it was drawn at, so a start reads as the ghost behind a
# design rather than as a second result.
WIDTH_OUTLINE = 0.48

# A node is a white disk with a dark rim, the way the plotters of jax-fdm and
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

# The technical drawing of the shared bridge problem: load arrows and support
# glyphs are sized against the span, so all three rows use one visual scale.
SETUP_LOAD_COLOR = "#00960a"
SETUP_LOAD_LENGTH = 0.10
SETUP_LOAD_GAP = 0.012
SETUP_SUPPORT_SIZE = 0.0385
SETUP_MEMBER_WIDTH = 2.0
SETUP_PERSON_COLOR = "#bfbfbf"

sampled = mpl.colormaps["plasma"](np.linspace(STRESS_LOW, STRESS_HIGH, 256))
UTILIZATION_MAP = ListedColormap(sampled, name="normax_utilization")


class DrawnLimits(NamedTuple):
    """
    Every limit a set of runs is drawn to rather than reading off its own walk.

    Attributes
    ----------
    across :
        Least and greatest coordinate across the drawing.
    upward :
        Least and greatest coordinate up the drawing.
    steps :
        Points in the longest walk, which the curves are drawn across and which
        paces an animation's frames.
    objective :
        Least and greatest value the objective axis spans.
    violation :
        Least and greatest value the violation axis spans, on its log scale.

    Notes
    -----
    A figure left to its own extents frames each run differently, so two runs
    of one structure come out at different aspect ratios and with different
    ticks, and neither the shapes nor the curves can be read side by side. A
    caller drawing several runs computes the union once and hands it to each.
    """

    across: tuple[float, float]
    upward: tuple[float, float]
    steps: int
    objective: tuple[float, float]
    violation: tuple[float, float]


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


ISOMETRIC_AZIMUTH = 45.0
ISOMETRIC_ELEVATION = 35.264389682754654


def project_view(
    xyz: Float[Array, "nodes 3"],
) -> Float[np.ndarray, "nodes 3"]:
    """
    The coordinates a drawing reads its two axes off, planar or solid.

    Parameters
    ----------
    xyz :
        Node positions as the pipeline computed them.

    Returns
    -------
    turned :
        Positions whose first column runs across the page and whose third runs
        up it, which is the pair every drawing here slices. The second carries
        depth into the page, which nothing draws.

    Notes
    -----
    A structure lying in one plane is drawn as it stands, so a planar run is
    unchanged to the last bit. A solid one is turned isometric instead: a side
    view of a cap shows one silhouette and hides the whole of the surface,
    where an isometric view shows the plan and the rise at once. Planar is read
    off the geometry rather than declared, since a held plan keeps a planar
    structure exactly planar and the test is therefore exact.
    """
    points = np.asarray(xyz, dtype=float)
    if float(np.ptp(points[:, 1])) == 0.0:
        return points

    azimuth = np.radians(ISOMETRIC_AZIMUTH)
    elevation = np.radians(ISOMETRIC_ELEVATION)
    across = np.array([-np.sin(azimuth), np.cos(azimuth), 0.0])
    upward = np.array(
        [
            -np.sin(elevation) * np.cos(azimuth),
            -np.sin(elevation) * np.sin(azimuth),
            np.cos(elevation),
        ]
    )
    into = np.cross(across, upward)

    turned = np.empty_like(points)
    turned[:, 0] = points @ across
    turned[:, 1] = points @ into
    turned[:, 2] = points @ upward

    return turned


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
    how the plotters of jax-fdm and compas draw one: the rim reads against a
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
        label="Free node",
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
        label="Fixed node",
    )
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_xlabel(LABEL_ACROSS, fontsize=SIZE_LABEL)
    ax.set_ylabel(LABEL_UPWARD, fontsize=SIZE_LABEL)
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


def _draw_setup_structure(ax: Axes, structure: Structure) -> None:
    """Draw the members and nodes of one problem-setup row."""
    nodes = np.asarray(structure.nodes)
    pairs = np.asarray(structure.edges)
    supports = np.asarray(structure.supports)
    points = nodes[:, [0, 2]]
    segments = points[pairs]

    members = LineCollection(
        segments,
        colors=MUTED,
        linewidths=SETUP_MEMBER_WIDTH,
        capstyle="round",
        zorder=2,
    )
    ax.add_collection(members)
    ax.plot(
        points[:, 0],
        points[:, 1],
        "o",
        ls="none",
        mfc=NODE_FILL,
        mec=NODE_EDGE,
        mew=NODE_RIM,
        ms=NODE_SIZE,
        zorder=4,
    )
    ax.plot(
        points[supports, 0],
        points[supports, 1],
        "o",
        ls="none",
        mfc=NODE_EDGE,
        mec=NODE_EDGE,
        mew=NODE_RIM,
        ms=NODE_SIZE,
        zorder=5,
    )


def _draw_pinned_support(ax: Axes, point: np.ndarray, size: float) -> None:
    """Draw one outlined pin on hatched ground, after the Vix 2D glyph."""
    x, z = point
    base = z - 0.82 * size
    half = 0.48 * size
    triangle = Polygon(
        ((x, z), (x - half, base), (x + half, base)),
        closed=True,
        facecolor="none",
        edgecolor=INK,
        linewidth=1.1,
        joinstyle="round",
        zorder=3,
    )
    ax.add_patch(triangle)

    ground_half = 0.68 * size
    ax.plot(
        (x - ground_half, x + ground_half),
        (base, base),
        color=INK,
        linewidth=1.1,
        solid_capstyle="round",
        zorder=3,
    )
    roots = np.linspace(x - 0.56 * size, x + 0.56 * size, 6)
    hatch = 0.18 * size
    hatches = [((root, base), (root - hatch, base - hatch)) for root in roots]
    ax.add_collection(LineCollection(hatches, colors=INK, linewidths=0.8, zorder=3))


def _draw_setup_loads(
    ax: Axes,
    xyz: Float[Array, "nodes 3"],
    loads: Float[Array, "nodes 3"],
    longest: float,
) -> None:
    """Draw proportional force arrows stopping just clear of their nodes."""
    points = np.asarray(xyz)[:, [0, 2]]
    forces = np.asarray(loads)[:, [0, 2]]
    magnitudes = np.linalg.norm(forces, axis=1)
    peak = float(magnitudes.max())
    if peak <= 0.0:
        return
    gap = SETUP_LOAD_GAP * longest / SETUP_LOAD_LENGTH

    for point, force, magnitude in zip(points, forces, magnitudes):
        if magnitude <= 0.0:
            continue
        length = longest * float(magnitude / peak)
        direction = force / magnitude
        tip = point - direction * gap
        start = tip - direction * length
        delta = tip - start
        ax.arrow(
            start[0],
            start[1],
            delta[0],
            delta[1],
            width=0.024 * length,
            head_width=0.10 * length,
            head_length=0.22 * length,
            length_includes_head=True,
            fc=SETUP_LOAD_COLOR,
            ec=SETUP_LOAD_COLOR,
            lw=0.0,
            zorder=5,
        )


def _draw_span_dimension(
    ax: Axes,
    left: float,
    right: float,
    height: float,
    label: str,
    span: float,
) -> None:
    """Draw one understated dimension beneath the last setup row."""
    dimension = ax.annotate(
        "",
        xy=(right, height),
        xytext=(left, height),
        arrowprops={
            "arrowstyle": "<->",
            "color": FAINT,
            "linewidth": 0.8,
            "shrinkA": 0.0,
            "shrinkB": 0.0,
        },
        zorder=1,
    )
    dimension.set_gid("problem-span")
    ax.text(
        0.5 * (left + right),
        height - 0.008 * span,
        label,
        ha="center",
        va="top",
        color=MUTED,
        fontsize=8,
        bbox={"facecolor": GROUND, "edgecolor": "none", "pad": 0.8},
        zorder=2,
    )


def _draw_person(ax: Axes, center: float, level: float, height: float) -> None:
    """Draw a subtle side-view walking silhouette at its true model height."""
    body_outline = np.array(
        [
            (-0.025, 0.865),
            (-0.070, 0.815),
            (-0.115, 0.700),
            (-0.190, 0.565),
            (-0.185, 0.520),
            (-0.150, 0.545),
            (-0.075, 0.650),
            (-0.055, 0.610),
            (-0.075, 0.525),
            (-0.055, 0.465),
            (-0.115, 0.285),
            (-0.205, 0.055),
            (-0.255, 0.015),
            (-0.245, 0.000),
            (-0.155, 0.000),
            (-0.105, 0.045),
            (-0.015, 0.260),
            (0.020, 0.390),
            (0.070, 0.275),
            (0.160, 0.080),
            (0.205, 0.025),
            (0.285, 0.010),
            (0.300, 0.000),
            (0.205, 0.000),
            (0.145, 0.035),
            (0.030, 0.215),
            (0.060, 0.445),
            (0.080, 0.525),
            (0.055, 0.615),
            (0.070, 0.700),
            (0.145, 0.625),
            (0.220, 0.535),
            (0.255, 0.515),
            (0.250, 0.555),
            (0.175, 0.655),
            (0.095, 0.780),
            (0.045, 0.825),
            (0.025, 0.865),
        ]
    )
    head_outline = np.array(
        [
            (-0.030, 0.855),
            (-0.065, 0.875),
            (-0.105, 0.900),
            (-0.135, 0.885),
            (-0.115, 0.925),
            (-0.075, 0.970),
            (-0.020, 1.000),
            (0.035, 0.990),
            (0.075, 0.955),
            (0.080, 0.930),
            (0.115, 0.915),
            (0.080, 0.900),
            (0.060, 0.870),
            (0.025, 0.850),
        ]
    )

    def patch_outline(outline: np.ndarray) -> PathPatch:
        scaled = outline.copy()
        scaled[:, 0] = center + height * scaled[:, 0]
        scaled[:, 1] = level + height * scaled[:, 1]
        vertices = np.vstack([scaled, scaled[0]])
        codes = [
            Path.MOVETO,
            *([Path.LINETO] * (len(scaled) - 1)),
            Path.CLOSEPOLY,
        ]

        return PathPatch(
            Path(vertices, codes),
            facecolor=SETUP_PERSON_COLOR,
            edgecolor="none",
            alpha=0.48,
            zorder=1,
        )

    body = patch_outline(body_outline)
    head = patch_outline(head_outline)
    body.set_gid("problem-person")
    ax.add_patch(body)
    head.set_gid("problem-person")
    ax.add_patch(head)


def draw_problem_setup(
    structure: Structure,
    load_cases: Float[Array, "load_cases nodes 3"],
    names: Sequence[str],
    span_label: str | None = None,
    layout: Literal["vertical", "horizontal"] = "vertical",
    person_height: float | None = None,
) -> Figure:
    """
    The shared bridge problem, one supported and loaded span per panel.

    Parameters
    ----------
    structure :
        The idealized deck supplying its nodes, span, and end supports.
    load_cases :
        Force applied at every node under each case to draw.
    names :
        Title of every load case, in row order.
    span_label :
        Text written beneath the final portrait panel or center landscape
        panel, or None to draw no dimension.
    layout :
        Whether cases run down one column for a paper or across one row for a
        slide.
    person_height :
        Height of a scale figure standing on the right half of the deck, in
        the structure's units, or None to draw no person.

    Returns
    -------
    figure :
        The load cases in the requested layout, each repeating the boundary
        conditions so every panel reads as a complete problem statement.

    Notes
    -----
    This is deliberately a drawing of the common bridge deck rather than any
    one structural topology. The arch, Warren, and Vierendeel examples span
    the same deck under the same three patterns, so one figure serves all of
    them without implying that their optimized forms are identical.

    Loads follow Vix's 2D convention: arrows scale to the largest nodal force
    within their row and stop just clear of the loaded node. Supports use its
    crisp outlined pin over hatched ground, while the page, ink, nodes, and
    titles use the rest of Normax's figure palette.
    """
    applied = np.asarray(load_cases)
    if applied.ndim != 3 or applied.shape[1:] != (structure.num_nodes, 3):
        expected = ("load_cases", structure.num_nodes, 3)
        raise ValueError(f"load_cases must have shape {expected}, got {applied.shape}")
    if len(names) != applied.shape[0]:
        raise ValueError(
            f"names must have one entry per load case, got {len(names)} for "
            f"{applied.shape[0]} cases"
        )
    if layout not in ("vertical", "horizontal"):
        raise ValueError(f"layout must be 'vertical' or 'horizontal', got {layout!r}")
    if person_height is not None and person_height <= 0.0:
        raise ValueError(f"person_height must be positive, got {person_height}")

    nodes = np.asarray(structure.nodes)
    span = float(np.ptp(nodes[:, 0]))
    if span <= 0.0:
        raise ValueError("a problem setup needs a positive horizontal span")

    cases = applied.shape[0]
    x_margin = 0.105 * span
    lower = float(nodes[:, 2].min()) - 0.11 * span
    upper = float(nodes[:, 2].max()) + 0.13 * span
    if person_height is not None:
        upper = max(upper, float(nodes[:, 2].min()) + person_height + 0.015 * span)
    across = (float(nodes[:, 0].min()) - x_margin, float(nodes[:, 0].max()) + x_margin)
    upward = (lower, upper)
    if layout == "vertical":
        grid = (cases, 1)
        width = WIDTH_DRAWING
        height = cases * read_drawing_height(across, upward, width)
    else:
        grid = (1, cases)
        width = 4.4 * cases
        height = 2.15
    figure, axes = plt.subplots(
        *grid,
        figsize=(width, height),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    drawings = axes.ravel()

    supports = np.asarray(structure.supports)
    support_size = SETUP_SUPPORT_SIZE * span
    load_length = SETUP_LOAD_LENGTH * span
    middle = 0.5 * (across[0] + across[1])
    person_center = float(nodes[:, 0].min()) + 0.75 * span
    deck_level = float(nodes[:, 2].min())
    for order, (ax, loads, name) in enumerate(zip(drawings, applied, names), start=1):
        centerline = ax.axvline(
            middle,
            color=FAINT,
            linewidth=0.7,
            linestyle=(0, (2, 3)),
            alpha=0.32,
            zorder=0,
        )
        centerline.set_gid("problem-midspan")
        if person_height is not None:
            _draw_person(ax, person_center, deck_level, person_height)
        _draw_setup_structure(ax, structure)
        for support in supports:
            _draw_pinned_support(ax, nodes[support, [0, 2]], support_size)
        _draw_setup_loads(ax, nodes, loads, load_length)
        ax.set_xlim(*across)
        ax.set_ylim(*upward)
        ax.set_aspect("equal")
        ax.set_title(
            f"({order})  {capitalize_label(name)}", loc="left", fontsize=SIZE_PANEL
        )
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])

    if span_label is not None:
        left = float(nodes[:, 0].min())
        right = float(nodes[:, 0].max())
        height = float(nodes[:, 2].min()) - 0.078 * span
        dimensioned = drawings[-1] if layout == "vertical" else drawings[cases // 2]
        _draw_span_dimension(dimensioned, left, right, height, span_label, span)

    engine = figure.get_layout_engine()
    if engine is not None:
        engine.set(h_pad=0.02, hspace=0.0, w_pad=0.03, wspace=0.01)

    paint_figure(figure)

    return figure


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
        linewidths=WIDTH_OUTLINE,
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
    limits: DrawnLimits | None = None,
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
    limits :
        Limits to hold the drawing to, or None to read them off the designs.

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
    if limits is not None:
        across, upward = limits.across, limits.upward
    upward = clear_for_legend(upward)

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
            outline.set_label("Starting shape")
        drawn = DrawnStructure(
            form.xyz, edges, form.diameters, scale, structure.supports
        )
        coloring = ColorRange(envelope, UTILIZATION_FLOOR, UTILIZATION_CAP)
        members = draw_members(ax, drawn, coloring)
        ax.set_xlim(*across)
        ax.set_ylim(*upward)
        ax.set_title(capitalize_label(form.title), fontsize=SIZE_TITLE)

    for ax in drawings[:-1]:
        ax.set_xlabel("")

    # On a drawing that carries the outline, which is never the reference's own:
    # the run names its start first and the start is the reference, so the
    # legend belongs on the design drawn against it. Boxed, since a corner
    # legend sits over the drawing rather than under it.
    outlined = drawings[-1] if reference is not None else drawings[0]
    entries = len(outlined.get_legend_handles_labels()[0])
    legend = outlined.legend(
        loc=LEGEND_PLACE,
        fontsize=8,
        frameon=True,
        framealpha=0.9,
        ncol=max(entries, 1),
    )
    legend.get_frame().set_linewidth(LEGEND_RIM)
    legend.get_frame().set_edgecolor(FAINT)

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
    bar.set_ticks(
        list(UTILIZATION_TICKS),
        labels=[f"{tick:.1f}" for tick in UTILIZATION_TICKS],
    )
    bar.set_label("Utilization", fontsize=SIZE_LABEL)
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
        ax.set_xticklabels([capitalize_label(name) for name in names], fontsize=9)
        ax.set_ylabel("Members governed", fontsize=SIZE_LABEL)
        ax.set_title(capitalize_label(form.title), fontsize=SIZE_PANEL)
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


def draw_objective_descent(
    panel: DescentPanel,
    limits: DrawnLimits | None = None,
) -> Figure:
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
            steps,
            values,
            marks,
            color=color,
            lw=1.6,
            ms=1.8,
            label=capitalize_label(trace.title),
        )

        crossings = read_round_bounds(history)
        entry = None if named or crossings.size == 0 else "Round start"
        named = named or crossings.size > 0
        draw_round_starts(violated, steps[crossings], placed[crossings], color, None)
        draw_round_starts(descent, steps[crossings], values[crossings], color, entry)

    levels = sorted({trace.tolerance for trace in panel.traces})
    spent = max(np.size(trace.history.objectives) for trace in panel.traces) - 1
    if limits is not None:
        spent = limits.steps - 1
    violated.set_xlim(0, max(spent, 1))
    violated.set_yscale("log")
    if limits is None:
        violated.set_ylim(bottom=floor)
    else:
        violated.set_ylim(*limits.violation)

    # Shaded off the axis's own floor rather than this run's, and after the
    # limits are set: where several runs share an axis its floor is the least
    # of theirs, and a band drawn to a higher one leaves the satisfied region
    # looking open at the bottom.
    violated.axhspan(violated.get_ylim()[0], levels[0], color=GREY, alpha=0.15, lw=0.0)
    for order, level in enumerate(levels):
        titled = "Tolerance" if order == 0 else None
        violated.axhline(level, color=GREY, ls="--", lw=1.0, label=titled)
    minimized = panel.heading.split(" [")[0]
    headline = f"Constrained {minimized} minimization"
    violated.set_ylabel("Constraints violation", fontsize=SIZE_LABEL)
    violated.set_title(capitalize_label(headline), fontsize=SIZE_TITLE)
    violated.legend(frameon=False, fontsize=9)
    violated.grid(alpha=0.3, which="both")

    descent.set_xlabel(capitalize_label(panel.axis), fontsize=SIZE_LABEL)
    descent.set_ylabel(capitalize_label(panel.heading), fontsize=SIZE_LABEL)
    if limits is not None:
        descent.set_xlim(0, max(spent, 1))
        descent.set_ylim(*limits.objective)
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
    limits: DrawnLimits | None = None,
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
    The designs are drawn in the order they are given and the first of them is
    outlined behind the rest, so a run naming its start first gets the start
    dashed in behind the answer and the figure reads in the order the search
    went.

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
            project_view(design.shape.xyz),
            design.sizes.sections.diameter,
            design.sizes.utilization,
        )
        forms.append(form)

    # The first design given is the one the others are read against, and the
    # run names its start first, so a figure reads left to right and top to
    # bottom in the order the search went.
    started = forms[0].xyz if len(forms) > 1 else None
    drawn = draw_utilization(structure, forms, started, limits) if forms else None
    governed = draw_governing_cases(forms, case_names) if forms else None
    descended = draw_objective_descent(panel, limits)

    return DrawnFigures(drawn, governed, descended)
