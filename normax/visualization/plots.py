# SPDX-License-Identifier: Apache-2.0
"""
Figures for the examples, in matplotlib and nothing else.

Every function takes arrays and returns a figure; nothing here calls `show`.
Member widths are drawn to a shared exaggeration rather than to scale, since a
tube one percent of the span wide would show nothing drawn truthfully.
"""

from collections.abc import Sequence
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from normax.design import Design
from normax.optimization import OptimizationSolution
from normax.structures import Structure

# Points of line width given to the thickest member of a drawing.
WIDTH_MAX = 9.0

# Color of everything that is a reference rather than a result.
GREY = "0.55"

# A load case governs every member within this distance of the member's worst.
TIE_MARGIN = 1e-9


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
    widest :
        Diameter drawn at the full width, shared so two drawings compare.
    """

    xyz: Float[Array, "nodes 3"]
    edges: Int[Array, "members 2"]
    diameters: Float[Array, "members"]
    widest: float


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
        Name of the colormap the range is rendered through.
    """

    values: Float[Array, "members"] | None = None
    vmin: float | None = None
    vmax: float | None = None
    cmap: str = "viridis"


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
        linewidths=WIDTH_MAX * sizes / drawn.widest,
        array=values,
        cmap=coloring.cmap,
        capstyle="round",
    )
    members.set_clim(vmin, vmax)
    ax.add_collection(members)

    ax.plot(nodes[:, 0], nodes[:, 2], ".", color="0.2", markersize=2.5, zorder=3)
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z [mm]")

    return members


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


def draw_utilization(
    edges: Int[Array, "members 2"],
    forms: Sequence[UtilizationForm],
    names: tuple[str, ...],
    reference: Float[Array, "nodes 3"] | None = None,
) -> Figure:
    """
    Designs colored by envelope utilization, above who governs each of them.

    Parameters
    ----------
    edges :
        The two node indices spanned by every member.
    forms :
        The designs to compare, in the order they are to be drawn.
    names :
        Name of every load case, in index order.
    reference :
        Shape to outline behind every form, or None to draw none.

    Returns
    -------
    figure :
        One drawing per design sharing a width scale, axis limits and a
        utilization colorbar, above a count of members governed per case.

    Notes
    -----
    The colorbar is capped at one, the cap fully stressed members sit on,
    and floored just under the least worked member across the designs.
    """
    envelopes = [np.asarray(form.utilization).max(axis=0) for form in forms]
    widest = max(float(np.max(np.asarray(form.diameters))) for form in forms)
    lowest = min(float(np.min(envelope)) for envelope in envelopes)
    floor = np.floor(lowest * 20.0) / 20.0
    load_cases = len(names)
    columns = len(forms)

    figure, axes = plt.subplots(
        2,
        columns,
        figsize=(4.6 * columns, 7.0),
        height_ratios=[2.0, 1.0],
        squeeze=False,
        layout="constrained",
    )
    for ax in axes[1, 1:]:
        ax.sharey(axes[1, 0])

    shapes = [np.asarray(form.xyz) for form in forms]
    if reference is not None:
        shapes.append(np.asarray(reference))
    both = np.concatenate(shapes)
    margin = 0.05 * float(np.ptp(both[:, 0]))

    for ax, form, envelope in zip(axes[0], forms, envelopes):
        if reference is not None:
            outline = draw_outline(ax, reference, edges)
            outline.set_label("starting shape")
        drawn = DrawnStructure(form.xyz, edges, form.diameters, widest)
        coloring = ColorRange(envelope, floor, 1.0)
        members = draw_members(ax, drawn, coloring)
        ax.set_xlim(float(both[:, 0].min()) - margin, float(both[:, 0].max()) + margin)
        ax.set_ylim(float(both[:, 2].min()) - margin, float(both[:, 2].max()) + margin)
        ax.set_title(form.title, fontsize=11)

    if reference is not None:
        axes[0][0].legend(loc="lower center", fontsize=8, frameon=False)

    bar = figure.colorbar(
        members,
        ax=axes[0].tolist(),
        shrink=0.85,
        aspect=14,
        pad=0.02,
    )
    bar.set_label("envelope utilization", fontsize=9)

    for ax, form in zip(axes[1], forms):
        counts = count_governed_members(form.utilization)
        ax.bar(np.arange(load_cases), counts, 0.6, color="#31688e")
        ax.set_xticks(np.arange(load_cases))
        ax.set_xticklabels(names, fontsize=8, rotation=15)
        ax.set_ylabel("members governed")
        ax.set_title(form.title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    return figure


class DescentTrace(NamedTuple):
    """
    One constrained descent, read as the objective at the end of every round.

    Attributes
    ----------
    title :
        Name of the search, shown in the legend.
    mass :
        Objective at the end of every round, the starting value first.
    """

    title: str
    mass: Float[np.ndarray, "rounds"]


def draw_mass_descent(traces: Sequence[DescentTrace]) -> Figure:
    """
    Constrained descents side by side, one line of objective per search.

    Parameters
    ----------
    traces :
        The descents to compare, in the order they are drawn.

    Returns
    -------
    figure :
        One panel of mass against round, shared so the lines compare where
        each flattens.
    """
    figure, descent = plt.subplots(figsize=(6.0, 4.0), layout="constrained")
    shades = ("#31688e", "#35b779", "#c0392b")

    for index, trace in enumerate(traces):
        mass = np.asarray(trace.mass)
        rounds = np.arange(len(mass))
        color = shades[index % len(shades)]
        descent.plot(rounds, mass, "-", color=color, lw=1.4, label=trace.title)

    descent.set_xlabel("round")
    descent.set_ylabel("mass [t]")
    descent.set_title("The constrained descents", fontsize=11)
    descent.legend(frameon=False, fontsize=9)
    descent.grid(alpha=0.3)

    return figure


def draw_design_figures(
    structure: Structure,
    designs: dict[str, Design],
    case_names: tuple[str, ...],
    solution: OptimizationSolution,
) -> tuple[Figure | None, Figure]:
    """
    The two figures a run draws: its designs, and the descent between them.

    Parameters
    ----------
    structure :
        The structure supplying the connectivity.
    designs :
        The designs to draw, keyed by the name each appears under.
    case_names :
        Name of every load case, naming who governs each member.
    solution :
        What the descent arrived at, whose objective column is drawn.

    Returns
    -------
    figures :
        The designs colored by utilization, or None where no design carries a
        utilization to color by, and the objective against the round.

    Notes
    -----
    A design whose pipeline carried no check is left out, and where that leaves
    nothing to draw the first figure is None rather than empty — `draw_utilization`
    reads a widest diameter and a least-worked member across the designs, and
    neither exists over no designs. The descent figure is drawn either way, its
    column being whatever the search minimized.
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
    drawn = draw_utilization(structure.edges, forms, case_names) if forms else None

    traces = (DescentTrace("auglag", solution.objectives),)
    descended = draw_mass_descent(traces)

    return drawn, descended
