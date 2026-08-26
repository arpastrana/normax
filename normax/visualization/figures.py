# Copyright 2026 Rafael Pastrana
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Figures for the pipeline, in matplotlib and nothing else.

The experiments compute and this module draws. Every function takes arrays and
an axis, so a figure can be recomposed without touching what produced it, and
nothing here calls `show` — the experiments save and move on.

Sizes are drawn to a stated exaggeration rather than to scale. A hundred
millimeter tube on a ten meter arch is a line one percent of the span wide, so
drawing it truthfully would show nothing; the factor is written into the axis
label instead of being left for the reader to guess.
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
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from normax.optimization import Trajectory

# Points of line width given to the thickest member of a drawing.
WIDTH_MAX = 9.0

# Color of everything that is a reference rather than a result.
GREY = "0.55"


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
        Diameter drawn at the full width. Shared between drawings, so that two
        of them may be compared rather than each being normalized to itself.
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

    Notes
    -----
    Every field defaults, so a drawing that has no opinion about color passes
    nothing. Fixing the ends is what lets two drawings share one color bar.
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

    segments = np.stack(
        [nodes[pairs[:, 0]][:, [0, 2]], nodes[pairs[:, 1]][:, [0, 2]]],
        axis=1,
    )
    values = sizes if coloring.values is None else np.asarray(coloring.values)

    members = LineCollection(
        segments,
        linewidths=WIDTH_MAX * sizes / drawn.widest,
        array=values,
        cmap=coloring.cmap,
        capstyle="round",
    )
    members.set_clim(
        values.min() if coloring.vmin is None else coloring.vmin,
        values.max() if coloring.vmax is None else coloring.vmax,
    )
    ax.add_collection(members)

    ax.plot(nodes[:, 0], nodes[:, 2], ".", color="0.2", markersize=2.5, zorder=3)
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z [mm]")

    return members


class SizedMembers(NamedTuple):
    """
    One set of member sizes, and the mass they come to.

    Attributes
    ----------
    diameters :
        Outer diameter of every member.
    mass :
        Total mass at those diameters.
    """

    diameters: Float[Array, "members"]
    mass: float


def figure_sections(
    xyz: Float[Array, "nodes 3"],
    edges: Int[Array, "members 2"],
    before: SizedMembers,
    after: SizedMembers,
) -> Figure:
    """
    The same arch before and after EN 1993-1-1 has decided its members.

    Parameters
    ----------
    xyz :
        Position of every node at equilibrium.
    edges :
        The two node indices spanned by every member.
    before :
        Sizes every member was assumed to have, and their mass.
    after :
        Sizes EN 1993-1-1 requires of every member, and their mass.

    Returns
    -------
    figure :
        Two arch drawings above a bar chart of the sizes.

    Notes
    -----
    The two drawings share one width scale and one color range, so a member
    that shrank looks thinner rather than merely differently normalized.
    """
    assumed = np.asarray(before.diameters)
    required = np.asarray(after.diameters)

    widest = float(max(np.max(assumed), np.max(required)))
    narrowest = float(min(np.min(assumed), np.min(required)))
    coloring = ColorRange(vmin=narrowest, vmax=widest)

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12.0, 7.5),
        height_ratios=[2.0, 1.0],
        layout="constrained",
    )

    titles = (
        f"Assumed, uniform — {before.mass:.4f} t",
        f"Required by EN 1993-1-1 — {after.mass:.4f} t",
    )
    for ax, sizes, title in zip(axes[0], (before.diameters, after.diameters), titles):
        members = draw_members(ax, DrawnStructure(xyz, edges, sizes, widest), coloring)
        ax.set_title(title, fontsize=11)

    figure.colorbar(members, ax=axes[0].tolist(), label="diameter [mm]", shrink=0.85)

    shift = after.mass / before.mass - 1.0
    axes[0, 1].text(
        0.02,
        0.95,
        f"{abs(shift):.1%} {'lighter' if shift < 0.0 else 'heavier'}",
        transform=axes[0, 1].transAxes,
        va="top",
        fontsize=10,
    )

    span = axes[1, 0]
    index = np.arange(len(required))
    span.bar(index - 0.2, assumed, 0.4, label="assumed", color=GREY)
    span.bar(index + 0.2, required, 0.4, label="required", color="#31688e")
    span.set_xlabel("member")
    span.set_ylabel("diameter [mm]")
    span.set_xticks(index)
    span.legend(frameon=False, fontsize=9)
    span.grid(axis="y", alpha=0.3)

    ratios = axes[1, 1]
    ratios.bar(index, required / assumed, 0.6, color="#35b779")
    ratios.axhline(1.0, color="0.2", lw=1.0)
    ratios.set_xlabel("member")
    ratios.set_ylabel("required / assumed")
    ratios.set_xticks(index)
    ratios.grid(axis="y", alpha=0.3)

    figure.supxlabel(
        f"Widths drawn to scale against each other, thickest member at "
        f"{WIDTH_MAX:.0f} pt",
        fontsize=9,
    )

    return figure


class MeshRefinement(NamedTuple):
    """
    How the mass settles as the mesh is refined.

    Attributes
    ----------
    counts :
        Number of members in each mesh.
    mass_member :
        Total mass with each member buckling over its own length.
    mass_fixed :
        Total mass with a buckling length held independent of the mesh.
    limit :
        Mass the mesh-independent sequence extrapolates to.
    """

    counts: Int[np.ndarray, "meshes"]
    mass_member: Float[np.ndarray, "meshes"]
    mass_fixed: Float[np.ndarray, "meshes"]
    limit: float


class StaggeredPasses(NamedTuple):
    """
    How far each repetition of the analysis and the check moves the sizes.

    Attributes
    ----------
    passes :
        Index of each pass through the staggered analysis and check.
    moves :
        Largest relative change in diameter produced by each pass.
    """

    passes: Int[np.ndarray, "passes"]
    moves: Float[np.ndarray, "passes"]


def figure_convergence(
    refinement: MeshRefinement,
    staggering: StaggeredPasses,
) -> Figure:
    """
    How the mass settles as the mesh refines, and as the staggering is repeated.

    Parameters
    ----------
    refinement :
        The mass under mesh refinement, and the limit it extrapolates to.
    staggering :
        How far each repetition of the analysis and the check moves the sizes.

    Returns
    -------
    figure :
        Three panels: the mass, its order of convergence, and the staggering.

    Notes
    -----
    The middle panel carries a first-order reference line rather than a fitted
    slope, so the reader compares against a claim instead of against a fit.
    """
    counts = refinement.counts
    mass_member = refinement.mass_member
    mass_fixed = refinement.mass_fixed

    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.4), layout="constrained")

    ax = axes[0]
    ax.plot(counts, mass_member, "o-", color="#440154", label=r"$L_{cr}$ = member")
    ax.plot(counts, mass_fixed, "s-", color="#31688e", label=r"$L_{cr}$ fixed")
    ax.axhline(
        refinement.limit,
        color="#31688e",
        ls="--",
        lw=1.0,
        label=r"limit, $L_{cr}$ fixed",
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(np.asarray(counts))
    ax.set_xticklabels([str(int(count)) for count in counts])
    ax.set_xlabel("members")
    ax.set_ylabel("mass [t]")
    ax.set_title("Mass under mesh refinement", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    steps = np.asarray(counts)[1:]
    for series, color, label in (
        (mass_member, "#440154", r"$L_{cr}$ = member"),
        (mass_fixed, "#31688e", r"$L_{cr}$ fixed"),
    ):
        values = np.asarray(series)
        change = np.abs(np.diff(values)) / np.abs(values[1:])
        ax.loglog(steps, change, "o-", color=color, label=label)
    reference = change[0] * steps[0] / steps
    ax.loglog(steps, reference, ls="--", color=GREY, lw=1.0, label="first order")
    ax.set_xlabel("members")
    ax.set_ylabel("relative change on refinement")
    ax.set_title("Order of convergence", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    ax.semilogy(staggering.passes, staggering.moves, "o-", color="#35b779")
    ax.set_xlabel("pass through analysis and check")
    ax.set_ylabel("largest relative move in diameter")
    ax.set_title("The staggered coupling", fontsize=11)
    ax.grid(alpha=0.3, which="both")

    return figure


class HandoffForces(NamedTuple):
    """
    What the two stages say a member carries, and how much bending it sees.

    Attributes
    ----------
    lengths :
        Length of every member.
    funicular :
        Axial force form finding predicts, being force density times length.
    analyzed :
        Axial force the frame analysis reports.
    moments :
        Largest end moment of every member, in magnitude.
    """

    lengths: Float[Array, "members"]
    funicular: Float[Array, "members"]
    analyzed: Float[Array, "members"]
    moments: Float[Array, "members"]


class GapScaling(NamedTuple):
    """
    How the disagreement between the two stages grows with the section.

    Attributes
    ----------
    diameters :
        Diameters the disagreement was measured at.
    gaps :
        Worst relative disagreement at each of those diameters.
    reference :
        Diameter the quadratic reference line is anchored at.
    """

    diameters: Float[np.ndarray, "sizes"]
    gaps: Float[np.ndarray, "sizes"]
    reference: float


class GradientCheck(NamedTuple):
    """
    A derivative taken exactly, beside the same one taken numerically.

    Attributes
    ----------
    exact :
        Derivative from the composed gradient.
    numeric :
        The same derivative from a central difference.
    """

    exact: Float[Array, "members"]
    numeric: Float[np.ndarray, "members"]


def figure_handoff(
    forces: HandoffForces,
    scaling: GapScaling,
    gradient: GradientCheck,
) -> Figure:
    """
    Whether form finding and the frame analysis agree, and why they cannot quite.

    Parameters
    ----------
    forces :
        What each stage says the members carry, and the bending they see.
    scaling :
        How the disagreement grows with the section it was measured at.
    gradient :
        The gradient across both stages, exactly and by central differences.

    Returns
    -------
    figure :
        Four panels: the forces, the disagreement, its law, and the gradient.
    """
    lengths = forces.lengths
    funicular = forces.funicular
    analyzed = forces.analyzed
    moments = forces.moments

    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), layout="constrained")

    ax = axes[0, 0]
    index = np.arange(len(np.asarray(funicular)))
    ax.plot(index, np.asarray(funicular) / 1e3, "o-", color="#440154", label=r"$q\,L$")
    ax.plot(index, np.asarray(analyzed) / 1e3, "x--", color="#fde725", label="smax")
    ax.set_xlabel("member")
    ax.set_ylabel("axial force [kN]")
    ax.set_title("Form finding against the frame analysis", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    gap = np.abs(np.asarray(analyzed) - np.asarray(funicular)) / np.abs(
        np.asarray(funicular)
    )
    share = np.abs(np.asarray(moments)) / np.abs(
        np.asarray(funicular) * np.asarray(lengths)
    )
    ax.semilogy(index, gap, "o-", color="#31688e", label="axial gap")
    ax.semilogy(index, share, "s-", color="#35b779", label=r"$M / (N L)$")
    ax.set_xlabel("member")
    ax.set_ylabel("relative")
    ax.set_title("The gap, member by member", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1, 0]
    sizes = np.asarray(scaling.diameters)
    measured = np.asarray(scaling.gaps)
    anchor = measured[np.argmin(np.abs(sizes - scaling.reference))]
    ax.loglog(sizes, measured, "o-", color="#440154", label="measured")
    ax.loglog(
        sizes,
        anchor * (sizes / scaling.reference) ** 2,
        ls="--",
        color=GREY,
        lw=1.0,
        label=r"$\propto d^{2}$",
    )
    ax.set_xlabel("diameter [mm]")
    ax.set_ylabel("worst axial gap")
    ax.set_title("The gap is quadratic in the diameter", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1, 1]
    exact = np.asarray(gradient.exact)
    numeric = np.asarray(gradient.numeric)
    ax.plot(index, exact, "o", color="#440154", label="autodiff")
    ax.plot(index, numeric, "x", color="#fde725", markersize=9, label="central")
    ax.set_xlabel("edge")
    ax.set_ylabel(r"$\partial \, \Sigma N^2 / \partial q$")
    worst = np.max(np.abs(exact - numeric) / np.abs(numeric))
    ax.set_title(f"Gradient across both stages, worst {worst:.1e}", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3)

    return figure


class Descent(NamedTuple):
    """
    One run of the optimizer, for a figure to draw against the sweep.

    Attributes
    ----------
    title :
        Name of the run, shown in the legend.
    mass :
        Objective at every iterate.
    beta :
        Envelope sharpness each iterate was taken under.
    """

    title: str
    mass: Float[np.ndarray, "steps"]
    beta: Float[np.ndarray, "steps"]


class MassSweep(NamedTuple):
    """
    The mass along a one-variable sweep, and where the descent started on it.

    Attributes
    ----------
    scales :
        Multiple of the starting force densities at each sample of the sweep.
    masses :
        Total mass at each of those multiples.
    start :
        Index of the sample the descent began from, which is the funicular
        design rather than the first sample of the sweep.
    """

    scales: Float[np.ndarray, "samples"]
    masses: Float[np.ndarray, "samples"]
    start: int


def figure_optimization(
    sweep: MassSweep,
    gradient: GradientCheck,
    descents: Sequence[Descent],
) -> Figure:
    """
    The one-variable mass curve, its gradient, and what twenty variables buy.

    Parameters
    ----------
    sweep :
        The mass along the sweep, and the sample the descent began from.
    gradient :
        The directional derivative along the sweep, exactly and by a central
        difference.
    descents :
        The optimizer runs to draw, the one the design is taken from last.

    Returns
    -------
    figure :
        The sweep, the gradient check and the descents, side by side.

    Notes
    -----
    The sweep is the best a single variable can do, so a run ending below its
    minimum is the figure's point rather than an inconsistency: the two are
    searching spaces of different size. **How far below is only meaningful once
    the members are stopped from collapsing**, which is why an unconstrained run
    is drawn beside a constrained one rather than instead of it.

    The gradient panel scales its error by the largest slope of the whole
    sweep. Dividing by the local slope would report a large error wherever the
    curve is flat, which is exactly where the minimum is.
    """
    scales = sweep.scales
    masses = sweep.masses
    start = sweep.start
    exact = gradient.exact
    numeric = gradient.numeric

    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), layout="constrained")

    best = int(np.argmin(masses))
    shades = ("#c0392b", "#35b779", "#31688e")

    curve = axes[0]
    curve.plot(scales, masses, "-o", color="#31688e", ms=3.5, label="single $q$")
    curve.plot(
        scales[best],
        masses[best],
        "o",
        color="#fde725",
        mec="0.2",
        ms=9,
        label=f"best single $q$, {masses[best]:.4f} t",
        zorder=4,
    )
    curve.plot(
        scales[start],
        masses[start],
        "s",
        color="0.2",
        ms=6,
        label=f"funicular start, {masses[start]:.4f} t",
        zorder=4,
    )
    for index, run in enumerate(descents):
        curve.axhline(
            float(run.mass[-1]),
            color=shades[index % len(shades)],
            ls="--",
            lw=1.4,
            label=f"{run.title}, {float(run.mass[-1]):.4f} t",
        )
    curve.set_xlabel("force density, as a multiple of the funicular value")
    curve.set_ylabel("mass [t]")
    curve.set_title("One variable, and twenty", fontsize=11)
    curve.legend(frameon=False, fontsize=8)
    curve.grid(alpha=0.3)

    slope = axes[1]
    slope.plot(scales, exact, "-", color="#31688e", lw=1.8, label="composed gradient")
    slope.plot(
        scales,
        numeric,
        "o",
        color="#fde725",
        mec="0.2",
        ms=5,
        label="central difference",
    )
    slope.axhline(0.0, color="0.2", lw=1.0)
    slope.set_xlabel("force density, as a multiple of the funicular value")
    slope.set_ylabel(r"$\partial$ mass $/ \partial k$ [t]")
    worst = float(np.max(np.abs(exact - numeric)) / np.max(np.abs(exact)))
    slope.set_title(f"Gradient against the sweep, worst {worst:.1e}", fontsize=11)
    slope.legend(frameon=False, fontsize=8.5)
    slope.grid(alpha=0.3)

    descent = axes[2]
    for index, run in enumerate(descents):
        steps = np.arange(len(run.mass))
        descent.plot(
            steps,
            run.mass,
            "-",
            color=shades[index % len(shades)],
            lw=1.4,
            label=run.title,
        )
        scatter = descent.scatter(
            steps, run.mass, c=run.beta, cmap="viridis", norm="log", s=14, zorder=2
        )
    descent.axhline(
        masses[best], color="#fde725", ls=":", lw=1.4, label="best single $q$"
    )
    descent.set_xlabel("iteration")
    descent.set_ylabel("mass [t]")
    kept = 1.0 - float(descents[-1].mass[-1]) / masses[start]
    descent.set_title(f"Descent, {kept:.1%} lighter than funicular", fontsize=11)
    descent.legend(frameon=False, fontsize=8)
    descent.grid(alpha=0.3)
    figure.colorbar(scatter, ax=descent, label=r"envelope sharpness $\beta$")

    return figure


def figure_trajectory(
    trajectories: Sequence[Trajectory],
    *,
    titles: Sequence[str] | None = None,
    concatenated: bool = False,
) -> Figure:
    """
    The objective at every iterate, for one search or several in a row.

    Parameters
    ----------
    trajectories :
        The runs to draw, each recorded by one search.
    titles :
        Name of each run, shown in the legend. Runs are numbered when no
        names are given.
    concatenated :
        Whether to draw the runs end to end on one iteration axis, as a
        single continued search, rather than overlaid from iteration zero.

    Returns
    -------
    figure :
        The descent of the objective, one curve per run.

    Notes
    -----
    A concatenated seam repeats an iterate rather than hiding it: a search
    records its starting point before it steps, so a warm-started run weighs
    its predecessor's answer again, and the step in the objective across the
    seam is a real change of measure rather than a plotting artifact.

    The iterates are colored by their envelope sharpness only when every run
    carries one. Zero is what a search stamps when its caller had no
    sharpness to give, and a logarithmic color scale has no place for it.
    """
    masses = [np.asarray(walked.mass) for walked in trajectories]
    sharpnesses = [np.asarray(walked.beta) for walked in trajectories]
    if titles is None:
        titles = tuple(f"run {index + 1}" for index in range(len(trajectories)))

    figure, ax = plt.subplots(figsize=(7.0, 4.2), layout="constrained")

    shades = ("#c0392b", "#35b779", "#31688e")
    stamped = all(float(np.min(sharpness)) > 0.0 for sharpness in sharpnesses)

    # One norm across every run, or the colors of two runs cannot be compared.
    coloring = None
    if stamped:
        dimmest = min(float(np.min(sharpness)) for sharpness in sharpnesses)
        sharpest = max(float(np.max(sharpness)) for sharpness in sharpnesses)
        coloring = LogNorm(dimmest, sharpest)

    offset = 0
    scatter = None
    for index, (mass, sharpness) in enumerate(zip(masses, sharpnesses, strict=True)):
        steps = np.arange(len(mass)) + offset
        ax.plot(
            steps,
            mass,
            "-",
            color=shades[index % len(shades)],
            lw=1.4,
            label=titles[index],
        )
        if coloring is not None:
            scatter = ax.scatter(
                steps, mass, c=sharpness, cmap="viridis", norm=coloring, s=14, zorder=2
            )
        if concatenated:
            offset += len(mass)
            if index < len(masses) - 1:
                ax.axvline(offset - 0.5, color=GREY, ls=":", lw=1.0)

    ax.set_xlabel("iteration")
    ax.set_ylabel("objective [t]")
    final = float(masses[-1][-1])
    ax.set_title(f"Descent, {final:.4f} t at the answer", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    if scatter is not None:
        figure.colorbar(scatter, ax=ax, label=r"envelope sharpness $\beta$")

    return figure


class Form(NamedTuple):
    """
    One shape to draw, and what decided each of its members.

    Attributes
    ----------
    title :
        Name of the form, shown above its drawing.
    xyz :
        Position of every node.
    diameters :
        Outer diameter of every member.
    governing :
        Index of the load case working each member hardest.
    """

    title: str
    xyz: Float[Array, "nodes 3"]
    diameters: Float[Array, "members"]
    governing: Int[Array, "members"]


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
    Drawn along the members rather than through the nodes in order, so a
    structure that is not a single chain outlines correctly. It carries no
    width: a reference shape is there to be compared against, and drawing it to
    scale would invite it to be read as a second result.
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


def figure_load_cases(
    edges: Int[Array, "members 2"],
    forms: Sequence[Form],
    names: tuple[str, ...],
    reference: Float[Array, "nodes 3"] | None = None,
) -> Figure:
    """
    Which load case decides each member, as the form moves.

    Parameters
    ----------
    edges :
        The two node indices spanned by every member.
    forms :
        The shapes to compare, in the order they are to be drawn.
    names :
        Name of every load case, in index order.
    reference :
        Shape to outline behind every form, or None to draw none. The shape the
        search started from, so that how far each result moved is visible in the
        drawing rather than only in the numbers.

    Returns
    -------
    figure :
        One drawing per form above a count of which load case governs how many
        members.

    Notes
    -----
    **The picture only a differentiable code check can produce.** No member was
    reassigned to a load case; the form moved, which changed how much bending
    each one raises where, and the pattern followed. A check that returns a verdict
    rather than a derivative can draw the first panel but has no way to search
    for the others.

    Every drawing shares one width scale and one pair of axis limits, so a form
    that dropped looks lower rather than merely differently framed, and a member
    that grew looks thicker. The counts share a scale for the same reason: a bar
    is read against its neighbours, and three independently scaled panels would
    make an even split look like a lopsided one. On this arch that matters: an
    unconstrained search collapses members into near-vertical clusters, which
    independently scaled axes would hide.
    """
    widest = max(float(np.max(np.asarray(form.diameters))) for form in forms)
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

    for ax, form in zip(axes[0], forms):
        if reference is not None:
            outline = draw_outline(ax, reference, edges)
            outline.set_label("starting shape")
        members = draw_members(
            ax,
            DrawnStructure(form.xyz, edges, form.diameters, widest),
            ColorRange(form.governing, 0.0, load_cases - 1.0),
        )
        ax.set_xlim(float(both[:, 0].min()) - margin, float(both[:, 0].max()) + margin)
        ax.set_ylim(float(both[:, 2].min()) - margin, float(both[:, 2].max()) + margin)
        ax.set_title(form.title, fontsize=11)

    if reference is not None:
        axes[0][0].legend(loc="lower center", fontsize=8, frameon=False)

    bar = figure.colorbar(
        members,
        ax=axes[0].tolist(),
        ticks=np.arange(load_cases),
        shrink=0.7,
        aspect=14,
        pad=0.02,
    )
    bar.ax.set_yticklabels(names, fontsize=9)

    for ax, form in zip(axes[1], forms):
        decided = np.asarray(form.governing)
        counts = [int(np.sum(decided == load_case)) for load_case in range(load_cases)]
        ax.bar(np.arange(load_cases), counts, 0.6, color="#31688e")
        ax.set_xticks(np.arange(load_cases))
        ax.set_xticklabels(names, fontsize=8, rotation=15)
        ax.set_ylabel("members governed")
        ax.set_title(form.title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    return figure


class BackendAgreement(NamedTuple):
    """
    How closely two solvers agree on the mass gradient, and at what sizes.

    Attributes
    ----------
    members :
        Number of members in each frame measured.
    gaps :
        Worst relative disagreement in the mass gradient at each size.
    tolerance :
        Agreement the roadmap asked for, drawn as a reference.
    """

    members: Int[np.ndarray, "sizes"]
    gaps: Float[np.ndarray, "sizes"]
    tolerance: float


class BackendTimings(NamedTuple):
    """
    What each solver's derivatives cost, alone and inside the composition.

    Attributes
    ----------
    parameters :
        Number of quantities the direct differentiation sweep registers.
    stage :
        Seconds the analysis stage alone spends on its derivatives, by backend.
    pipeline :
        Seconds the whole composition spends on a value and gradient, by
        backend.
    """

    parameters: Int[np.ndarray, "sizes"]
    stage: dict[str, Float[np.ndarray, "sizes"]]
    pipeline: dict[str, Float[np.ndarray, "sizes"]]


def figure_backends(
    agreement: BackendAgreement,
    timings: BackendTimings,
) -> Figure:
    """
    Two solvers agreeing on a gradient, and disagreeing about what it costs.

    Parameters
    ----------
    agreement :
        How closely the two solvers agree on the mass gradient.
    timings :
        What each solver's derivatives cost, alone and in the composition.

    Returns
    -------
    figure :
        The agreement, the cost of the stage alone, and the cost of the whole
        composition.

    Notes
    -----
    **The middle panel is the scaling claim and the right panel is why it does
    not decide anything here.** Direct differentiation reuses one factorization,
    so a parameter costs a back-substitution and the sweep grows with the
    parameter count; a traced backend answers in one reverse pass whatever the
    count. That difference is real and visible in isolation, and at these sizes
    it is buried under what the composition costs regardless of who solves.

    Every cost axis is logarithmic, the two backends differing by more than an
    order of magnitude, and a linear axis would draw one of them flat.
    """
    members = agreement.members
    tolerance = agreement.tolerance

    figure, axes = plt.subplots(1, 3, figsize=(WIDTH_MAX, 3.4), layout="constrained")

    axes[0].axhline(tolerance, color=GREY, linestyle="--", linewidth=1.0)
    axes[0].annotate(
        f"asked for {tolerance:.0e}",
        (members[0], tolerance),
        textcoords="offset points",
        xytext=(4, -12),
        color=GREY,
        fontsize=8,
    )
    axes[0].plot(members, agreement.gaps, "o-", color="#31688e", markersize=4)
    axes[0].set_yscale("log")
    axes[0].set_ylim(top=tolerance * 30.0)
    axes[0].set_xlabel("members")
    axes[0].set_ylabel("worst relative gap in dmass/dq")
    axes[0].set_title("DDM against traced autodiff", fontsize=10)
    axes[0].grid(alpha=0.3)

    styles = {"smax": ("#440154", "o"), "opensees": ("#35b779", "s")}

    for name, series in timings.stage.items():
        color, marker = styles[name]
        axes[1].plot(
            timings.parameters,
            series,
            marker + "-",
            color=color,
            markersize=4,
            label=name,
        )

    axes[1].set_yscale("log")
    axes[1].set_xlabel("parameters registered")
    axes[1].set_ylabel("seconds")
    axes[1].set_title("the analysis stage alone\ncompiled, prepared once", fontsize=10)
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    for name, series in timings.pipeline.items():
        color, marker = styles[name]
        axes[2].plot(
            members, series, marker + "-", color=color, markersize=4, label=name
        )

    axes[2].set_yscale("log")
    axes[2].set_xlabel("members")
    axes[2].set_ylabel("seconds")
    axes[2].set_title("the whole composition\nassembled per crossing", fontsize=10)
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=8)

    return figure


class BeamStatics(NamedTuple):
    """
    Bending moment along a beam, from the solver and from statics.

    Attributes
    ----------
    positions :
        Position of every node along the span.
    exact :
        Bending moment statics puts at each of them.
    computed :
        Bending moment the frame analysis reports there.
    """

    positions: Float[np.ndarray, "nodes"]
    exact: Float[np.ndarray, "nodes"]
    computed: Float[np.ndarray, "nodes"]


class BeamSizing(NamedTuple):
    """
    Diameter every member needs, from the check and in closed form.

    Attributes
    ----------
    members :
        Index of every member.
    positions :
        Midpoint of every member along the span.
    required :
        Diameter the sizing map returns.
    closed_form :
        Diameter the inverted bending check puts at the same actions.
    """

    members: Int[np.ndarray, "members"]
    positions: Float[np.ndarray, "members"]
    required: Float[np.ndarray, "members"]
    closed_form: Float[np.ndarray, "members"]


def figure_benchmark(statics: BeamStatics, sizing: BeamSizing) -> Figure:
    """
    Both stages of a straight beam against the arithmetic that predicts them.

    Parameters
    ----------
    statics :
        Bending moment along the beam, from the solver and from statics.
    sizing :
        Diameter every member needs, from the check and in closed form.

    Returns
    -------
    figure :
        Three panels: the moment diagram, the sizes, and both disagreements.

    Notes
    -----
    The predicted series is drawn as a line and the computed one as markers on
    top of it, so agreement reads as markers sitting on a curve rather than as
    two curves a reader has to separate.

    The third panel is scaled by the largest value of each series rather than by
    each entry's own, a moment at a support being zero and its relative error
    meaningless.
    """
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.4), layout="constrained")

    moment_exact = np.asarray(statics.exact) / 1e6
    moment_computed = np.asarray(statics.computed) / 1e6

    ax = axes[0]
    ax.plot(statics.positions, moment_exact, "-", color=GREY, lw=2.0, label="statics")
    ax.plot(
        statics.positions,
        moment_computed,
        "o",
        color="#440154",
        markersize=5,
        label="frame analysis",
    )
    ax.set_xlabel("position along the span [mm]")
    ax.set_ylabel("bending moment [kNm]")
    ax.set_title("The moment diagram", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(
        sizing.members,
        sizing.closed_form,
        "-",
        color=GREY,
        lw=2.0,
        label="closed form",
    )
    ax.plot(
        sizing.members,
        sizing.required,
        "s",
        color="#31688e",
        markersize=5,
        label="sizing map",
    )
    ax.set_xticks(np.asarray(sizing.members))
    ax.set_xlabel("member")
    ax.set_ylabel("diameter [mm]")
    ax.set_title("The size EN 1993-1-1 requires", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[2]
    moment_scale = max(float(np.max(np.abs(statics.exact))), np.finfo(float).tiny)
    size_scale = max(float(np.max(np.abs(sizing.closed_form))), np.finfo(float).tiny)
    moment_gap = np.abs(np.asarray(statics.computed) - np.asarray(statics.exact))
    size_gap = np.abs(np.asarray(sizing.required) - np.asarray(sizing.closed_form))
    span = float(np.max(statics.positions))
    floor = np.finfo(float).eps
    ax.semilogy(
        np.asarray(statics.positions) / span,
        np.maximum(moment_gap / moment_scale, floor),
        "o-",
        color="#440154",
        markersize=4,
        label="moment against statics",
    )
    ax.semilogy(
        np.asarray(sizing.positions) / span,
        np.maximum(size_gap / size_scale, floor),
        "s-",
        color="#31688e",
        markersize=4,
        label="diameter against closed form",
    )
    ax.axhline(floor, color=GREY, ls="--", lw=1.0, label="machine epsilon")
    ax.set_xlabel("fraction of the span")
    ax.set_ylabel("scaled disagreement")
    ax.set_title("What the two stages disagree by", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3, which="both")

    return figure


def figure_beam_profile(
    positions: Float[np.ndarray, "nodes"],
    before: SizedMembers,
    after: SizedMembers,
) -> Figure:
    """
    A straight beam at the depth EN 1993-1-1 requires, drawn to scale.

    Parameters
    ----------
    positions :
        Position of every node along the span.
    before :
        Sizes every member was assumed to have, and their mass.
    after :
        Sizes EN 1993-1-1 requires of every member, and their mass.

    Returns
    -------
    figure :
        The beam in elevation, each member as deep as its diameter.

    Notes
    -----
    Drawn to scale rather than exaggerated, unlike the arch drawings: a beam has
    no shape of its own to be crowded out, so the depth is the only thing to
    show and it is legible at a true 1:26. The assumed uniform depth is outlined
    behind it, which is what makes the taper read as a result rather than as a
    shape someone chose.
    """
    nodes = np.asarray(positions)
    required = np.asarray(after.diameters)
    assumed = np.asarray(before.diameters)

    figure, ax = plt.subplots(1, 1, figsize=(11.0, 3.4), layout="constrained")
    spread = max(float(np.ptp(required)), np.finfo(float).tiny)
    colors = plt.get_cmap("viridis")((required - required.min()) / spread)

    for member in range(required.shape[0]):
        left = nodes[member]
        width = nodes[member + 1] - left
        depth = required[member]
        ax.add_patch(
            plt.Rectangle(
                (left, -0.5 * depth),
                width,
                depth,
                facecolor=colors[member],
                edgecolor="white",
                linewidth=0.5,
            )
        )

    uniform = 0.5 * float(np.max(assumed))
    ax.plot(
        [nodes[0], nodes[-1], nodes[-1], nodes[0], nodes[0]],
        [-uniform, -uniform, uniform, uniform, -uniform],
        ls="--",
        lw=1.0,
        color=GREY,
    )

    heavier = after.mass / before.mass - 1.0
    ax.set_xlim(nodes[0], nodes[-1])
    ax.set_ylim(-0.75 * required.max(), 0.75 * required.max())
    ax.set_aspect("equal")
    ax.set_xlabel("position along the span [mm]")
    ax.set_ylabel("depth [mm]")
    ax.set_title(
        f"Required by EN 1993-1-1 — {after.mass:.4f} t, {heavier:.1%} heavier",
        fontsize=11,
    )
    ax.text(
        0.01,
        0.97,
        f"dashed: assumed uniform, {before.mass:.4f} t",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        color="0.35",
    )

    return figure


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
        Worst utilization of every member over the load cases.
    governed :
        How many members each load case governs, in case order. The caller
        owns the counting, tie policy included — mirror-paired cases tie to
        solver precision on symmetric designs, and an argmax here would
        split those ties by index order.
    """

    title: str
    xyz: Float[Array, "nodes 3"]
    diameters: Float[Array, "members"]
    utilization: Float[Array, "members"]
    governed: Int[np.ndarray, "cases"]


def figure_utilization(
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
        One drawing per design, sharing one width scale, one pair of axis
        limits and one utilization colorbar, above a count of which load
        case governs how many members.

    Notes
    -----
    The colorbar is capped at one — the feasible ceiling, which fully
    stressed members sit exactly on — and floored just under the least
    worked member across all the designs, so the color range is spent on
    the diversity that exists rather than on the empty run down to zero.
    A member at the cap is at a binding constraint; a visibly colder one is
    either resting on the diameter floor or buying force relief for its
    fully stressed neighbors, which only an indeterminate structure can do.

    The counts underneath answer the question the coloring no longer can:
    which case put each member at its ceiling. They share one scale, so a
    bar is read against its neighbors across the designs, and they arrive
    counted rather than labeled — a member tied between cases may appear
    under each of them, so a panel's bars can sum past the member count.
    """
    widest = max(float(np.max(np.asarray(form.diameters))) for form in forms)
    lowest = min(float(np.min(np.asarray(form.utilization))) for form in forms)
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

    for ax, form in zip(axes[0], forms):
        if reference is not None:
            outline = draw_outline(ax, reference, edges)
            outline.set_label("starting shape")
        members = draw_members(
            ax,
            DrawnStructure(form.xyz, edges, form.diameters, widest),
            ColorRange(form.utilization, floor, 1.0),
        )
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
        counts = [int(count) for count in np.asarray(form.governed)]
        ax.bar(np.arange(load_cases), counts, 0.6, color="#31688e")
        ax.set_xticks(np.arange(load_cases))
        ax.set_xticklabels(names, fontsize=8, rotation=15)
        ax.set_ylabel("members governed")
        ax.set_title(form.title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    return figure


class DescentTrace(NamedTuple):
    """
    One constrained descent, read as the objective at every iterate.

    Attributes
    ----------
    title :
        Name of the search, shown in the legend.
    mass :
        Objective at every iterate, the start included.
    """

    title: str
    mass: Float[np.ndarray, "steps"]


def figure_mass_descent(traces: Sequence[DescentTrace]) -> Figure:
    """
    Constrained descents side by side, one line of objective per search.

    Parameters
    ----------
    traces :
        The descents to compare, in the order they are drawn.

    Returns
    -------
    figure :
        One panel of mass against iteration.

    Notes
    -----
    A single shared panel rather than one per search: the comparison is where
    each line flattens, and separately scaled axes would hide the gap the
    figure exists to show. The first and last shades match the palette of
    `figure_parametrization` so a search reads the same across experiments.
    """
    figure, descent = plt.subplots(figsize=(6.0, 4.0), layout="constrained")
    shades = ("#31688e", "#35b779", "#c0392b")

    for index, trace in enumerate(traces):
        steps = np.arange(len(trace.mass))
        color = shades[index % len(shades)]
        descent.plot(steps, trace.mass, "-", color=color, lw=1.4, label=trace.title)

    descent.set_xlabel("iteration")
    descent.set_ylabel("mass [t]")
    descent.set_title("The constrained descents", fontsize=11)
    descent.legend(frameon=False, fontsize=9)
    descent.grid(alpha=0.3)

    return figure


class SearchTrace(NamedTuple):
    """
    One search's descent, and how funicular its iterates stayed.

    Attributes
    ----------
    title :
        Name of the search, shown in the legend.
    mass :
        Objective at every iterate.
    bending :
        Largest bending-to-axial ratio of any member at every iterate, under
        the load case the shape answers to.
    """

    title: str
    mass: Float[np.ndarray, "steps"]
    bending: Float[np.ndarray, "steps"]


class StartSpread(NamedTuple):
    """
    The mass each search reaches from every matched start.

    Attributes
    ----------
    labels :
        Name of every start, in the order the masses are given.
    mass_density :
        Mass the single force density reaches from each start.
    mass_heights :
        Mass the free heights reach from each start.

    Notes
    -----
    A start only one search can take carries NaN in the other search's slot,
    and the figure draws no marker there.
    """

    labels: tuple[str, ...]
    mass_density: Float[np.ndarray, "starts"]
    mass_heights: Float[np.ndarray, "starts"]


def figure_parametrization(
    traces: Sequence[SearchTrace],
    spread: StartSpread,
    closed: StartSpread | None = None,
    constrained: float | None = None,
) -> Figure:
    """
    A physics-informed parametrization against free coordinates, side by side.

    Parameters
    ----------
    traces :
        The matched-start descent of every search, in the order they are drawn.
    spread :
        The mass each search reaches from every start.
    closed :
        The same masses with the coupling closed, or None where no staggered
        runs were made. Shares the spread's start order.
    constrained :
        Mass the simultaneous density-and-diameters search reaches, or None
        where none ran. One level rather than one mass per start, because
        that search lands on the same answer from every start — which is
        exactly what a horizontal line says and a row of markers would not.

    Returns
    -------
    figure :
        The descents, the bending ratio along them, and the start dependence.

    Notes
    -----
    Three panels because the comparison makes three claims. The descent panel
    shows where each search ends; the bending panel shows what its iterates
    passed through on the way, on a logarithmic axis because the searches differ
    by orders of magnitude; the spread panel shows what each start bought,
    which is where a larger design space pays for itself or does not.

    One color per search across all three panels, so a search reads as one
    entity wherever it appears, and the spread panel's markers differ in shape
    so it survives being printed without color. The closed-coupling masses
    wear the same marks hollow: filled against open is the frozen seed
    against the settled sections, per search, without a third color.
    """
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), layout="constrained")
    descent, quality, robustness = axes
    shades = ("#31688e", "#c0392b")
    markers = ("o", "s")

    for index, trace in enumerate(traces):
        steps = np.arange(len(trace.mass))
        color = shades[index % len(shades)]
        descent.plot(steps, trace.mass, "-", color=color, lw=1.4, label=trace.title)
        quality.plot(steps, trace.bending, "-", color=color, lw=1.4)

    descent.set_xlabel("iteration")
    descent.set_ylabel("objective [t]")
    descent.set_title("The two descents, matched start", fontsize=11)
    descent.legend(frameon=False, fontsize=8)
    descent.grid(alpha=0.3)

    quality.set_yscale("log")
    quality.set_xlabel("iteration")
    quality.set_ylabel(r"$\max_k \, |M| \, / \, (|N| \, L)$")
    quality.set_title("How funicular the iterates stayed", fontsize=11)
    quality.grid(alpha=0.3, which="both")

    positions = np.arange(len(spread.labels))
    reaches = (spread.mass_density, spread.mass_heights)
    for index, trace in enumerate(traces):
        robustness.plot(
            positions,
            reaches[index],
            markers[index % len(markers)],
            color=shades[index % len(shades)],
            markersize=8,
            label=trace.title,
        )
    if closed is not None:
        settled = (closed.mass_density, closed.mass_heights)
        for index, trace in enumerate(traces):
            # Dodged sideways: a coupling shift of a tenth of a percent would
            # otherwise sit exactly under its own frozen marker.
            robustness.plot(
                positions + 0.18,
                settled[index],
                markers[index % len(markers)],
                color=shades[index % len(shades)],
                markersize=8,
                markerfacecolor="none",
                label=f"{trace.title}, staggered",
            )
    if constrained is not None:
        robustness.axhline(
            constrained,
            color=shades[0],
            ls="--",
            lw=1.4,
            label="density and diameters, constrained",
        )
    robustness.set_xticks(positions)
    robustness.set_xticklabels(spread.labels)
    robustness.set_xlim(-0.5, len(spread.labels) - 0.5)
    robustness.set_xlabel("starting shape")
    robustness.set_ylabel("mass at the answer [t]")
    robustness.set_title("What each start bought", fontsize=11)
    robustness.legend(frameon=False, fontsize=8)
    robustness.grid(alpha=0.3)

    return figure


class TrussForm(NamedTuple):
    """
    One truss shape to draw, and the axial force its members carry.

    Attributes
    ----------
    title :
        Name of the form, shown above its drawing.
    xyz :
        Position of every node.
    forces :
        Axial force of every member, negative in compression.
    """

    title: str
    xyz: Float[Array, "nodes 3"]
    forces: Float[Array, "members"]


def figure_truss_forms(
    edges: Int[Array, "members 2"],
    forms: Sequence[TrussForm],
    reference: Float[Array, "nodes 3"],
    reference_label: str = "drawn truss",
) -> Figure:
    """
    Truss shapes side by side, members as wide as the force they carry.

    Parameters
    ----------
    edges :
        The two node indices spanned by every member.
    forms :
        The shapes to compare, in the order they are to be drawn.
    reference :
        Shape to outline behind every form.
    reference_label :
        What the outline is called in the legend.

    Returns
    -------
    figure :
        One drawing per form, four to a row, sharing one force scale and one
        pair of axis limits.

    Notes
    -----
    Forces sit on one diverging scale, compression on the red side and tension
    on the blue, so a chord that swaps role between panels swaps color rather
    than merely shade. Width carries magnitude, floored so that a member
    carrying nothing stays visible as a thin line instead of vanishing.
    """
    peak = max(float(np.max(np.abs(np.asarray(form.forces)))) for form in forms)
    columns = min(4, len(forms))
    rows = -(-len(forms) // columns)

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.8 * columns, (3.2 if rows == 1 else 2.1) * rows),
        squeeze=False,
        layout="constrained",
    )

    flat = axes.ravel()
    for ax in flat[len(forms) :]:
        ax.set_axis_off()

    shapes = [np.asarray(form.xyz) for form in forms]
    shapes.append(np.asarray(reference))
    every = np.concatenate(shapes)
    margin = 0.05 * float(np.ptp(every[:, 0]))
    across = (float(every[:, 0].min()) - margin, float(every[:, 0].max()) + margin)
    upward = (float(every[:, 2].min()) - margin, float(every[:, 2].max()) + margin)

    for index, (ax, form) in enumerate(zip(flat, forms)):
        outline = draw_outline(ax, reference, edges)
        if index == 0:
            outline.set_label(reference_label)
        magnitudes = np.abs(np.asarray(form.forces))
        widths = np.maximum(magnitudes, 0.05 * peak)
        members = draw_members(
            ax,
            DrawnStructure(form.xyz, edges, widths, peak),
            ColorRange(form.forces, -peak, peak, "RdBu"),
        )
        ax.set_xlim(across)
        ax.set_ylim(upward)
        ax.set_title(form.title, fontsize=10)
        if index % columns:
            ax.set_ylabel("")
        if index < len(forms) - columns:
            ax.set_xlabel("")

    flat[0].legend(loc="upper right", fontsize=8, frameon=False)

    bar = figure.colorbar(
        members,
        ax=flat.tolist(),
        shrink=0.85 if rows == 1 else 0.6,
        aspect=14 if rows == 1 else 25,
        pad=0.02,
    )
    bar.set_label("axial force [N]")

    return figure


class SubspaceMode(NamedTuple):
    """
    One direction of the held-plan density subspace, made visible.

    Attributes
    ----------
    title :
        Name of the mode, shown above its drawing.
    xyz :
        Geometry displaced along the mode, exaggerated for the eye.
    densities :
        The density direction itself, one component per member.
    """

    title: str
    xyz: Float[Array, "nodes 3"]
    densities: Float[Array, "members"]


class ShapeVariation(NamedTuple):
    """
    One form-found shape, named for the panel it is drawn in.

    Attributes
    ----------
    title :
        Name of the variation, shown above its drawing.
    xyz :
        Position of every node.
    """

    title: str
    xyz: Float[Array, "nodes 3"]


def figure_shape_variations(
    edges: Int[Array, "members 2"],
    variations: Sequence[ShapeVariation],
) -> Figure:
    """
    Every shape drawn as a wireframe in three dimensions, three to a row.

    Parameters
    ----------
    edges :
        The two node indices spanned by every member.
    variations :
        The shapes to draw, in the order they are to be drawn.

    Returns
    -------
    figure :
        A grid of wireframes sharing one set of axis limits, one viewing
        angle and one height colorbar.

    Notes
    -----
    One set of limits and one color scale across every panel, so a shape is
    read against its neighbors rather than against its own extent — a panel
    autoscaled to itself would make every variation look alike.

    Members are colored by the height of their midpoint, which is the whole of
    what a held plan lets a variation change: the plan is fixed by
    construction, so two panels differ in nothing but height.
    """
    shapes = [np.asarray(variation.xyz) for variation in variations]
    every = np.concatenate(shapes)
    spans = np.ptp(every, axis=0)
    margin = 0.05 * float(np.max(spans))
    lowest = every.min(axis=0) - margin
    highest = every.max(axis=0) + margin
    scale = Normalize(float(every[:, 2].min()), float(every[:, 2].max()))

    columns = 3
    rows = -(-len(variations) // columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.2 * columns, 3.3 * rows),
        squeeze=False,
        subplot_kw={"projection": "3d"},
    )

    flat = axes.ravel()
    for ax in flat[len(variations) :]:
        ax.set_axis_off()

    spans_named = np.asarray(edges)
    drawn = None
    for ax, variation in zip(flat, variations):
        nodes = np.asarray(variation.xyz)
        starts = nodes[spans_named[:, 0]]
        ends = nodes[spans_named[:, 1]]
        segments = np.stack([starts, ends], axis=1)
        heights = 0.5 * (starts[:, 2] + ends[:, 2])

        drawn = Line3DCollection(segments, cmap="viridis", norm=scale, linewidths=0.7)
        drawn.set_array(heights)
        ax.add_collection3d(drawn)

        ax.set_xlim(lowest[0], highest[0])
        ax.set_ylim(lowest[1], highest[1])
        ax.set_zlim(lowest[2], highest[2])
        ax.set_box_aspect((spans[0], spans[1], max(spans[2], 0.2 * spans[0])))
        ax.view_init(elev=24.0, azim=-58.0)
        ax.set_title(variation.title, fontsize=9, pad=2.0)
        # Height is read off the shared colorbar, so no axis needs to say it.
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    # Constrained layout collapses a 3D grid once a shared colorbar joins it,
    # and 3D axes carry margins a negative spacing has to claw back.
    figure.subplots_adjust(
        left=0.0,
        right=0.9,
        top=0.97,
        bottom=0.0,
        wspace=-0.08,
        hspace=-0.04,
    )
    bar = figure.colorbar(
        drawn,
        ax=flat.tolist(),
        shrink=0.42,
        aspect=28,
        pad=0.0,
    )
    bar.set_label("height [mm]")

    return figure


def figure_density_modes(
    edges: Int[Array, "members 2"],
    reference: Float[Array, "nodes 3"],
    modes: Sequence[SubspaceMode],
) -> Figure:
    """
    Every independent direction of the held-plan subspace, one panel each.

    Parameters
    ----------
    edges :
        The two node indices spanned by every member.
    reference :
        The undisplaced shape, outlined behind every mode.
    modes :
        The directions to draw, in the order they are to be drawn.

    Returns
    -------
    figure :
        A grid of drawings, four to a row, sharing one color scale.

    Notes
    -----
    Each panel shows one direction twice over: the solid shape is the geometry
    displaced along the mode, and its coloring is the density change driving
    it. A panel that changes color without moving the shape is a state of
    self-stress — force redistribution the geometry cannot see.
    """
    peak = max(float(np.max(np.abs(np.asarray(mode.densities)))) for mode in modes)
    columns = 4
    rows = -(-len(modes) // columns)

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.9 * columns, 1.9 * rows),
        squeeze=False,
        layout="constrained",
    )

    flat = axes.ravel()
    for ax in flat[len(modes) :]:
        ax.set_axis_off()

    shapes = [np.asarray(mode.xyz) for mode in modes]
    shapes.append(np.asarray(reference))
    every = np.concatenate(shapes)
    margin = 0.05 * float(np.ptp(every[:, 0]))
    across = (float(every[:, 0].min()) - margin, float(every[:, 0].max()) + margin)
    upward = (float(every[:, 2].min()) - margin, float(every[:, 2].max()) + margin)

    num_members = np.asarray(edges).shape[0]
    widths = np.ones(num_members)

    for index, (ax, mode) in enumerate(zip(flat, modes)):
        draw_outline(ax, reference, edges)
        members = draw_members(
            ax,
            DrawnStructure(mode.xyz, edges, widths, 3.0),
            ColorRange(mode.densities, -peak, peak, "RdBu"),
        )
        ax.set_xlim(across)
        ax.set_ylim(upward)
        ax.set_title(mode.title, fontsize=9)
        if index % columns:
            ax.set_ylabel("")
        if index < len(modes) - columns:
            ax.set_xlabel("")

    bar = figure.colorbar(
        members,
        ax=flat.tolist(),
        shrink=0.6,
        aspect=25,
        pad=0.01,
    )
    bar.set_label("density direction")

    return figure
