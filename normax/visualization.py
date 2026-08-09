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
millimetre tube on a ten metre arch is a line one percent of the span wide, so
drawing it truthfully would show nothing; the factor is written into the axis
label instead of being left for the reader to guess.
"""

import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

# Points of line width given to the thickest member of a drawing.
WIDTH_MAX = 9.0

# Colour of everything that is a reference rather than a result.
GREY = "0.55"


def draw_members(
    ax: Axes,
    xyz: Float[Array, "nodes 3"],
    edges: Int[Array, "members 2"],
    diameters: Float[Array, "members"],
    widest: float,
    colors: Float[Array, "members"] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> LineCollection:
    """
    Draw a planar structure with every member as wide as its diameter.

    Parameters
    ----------
    ax :
        The axis to draw on.
    xyz :
        Position of every node. The X and Z coordinates are used.
    edges :
        The two node indices spanned by every member.
    diameters :
        Outer diameter of every member, setting its drawn width.
    widest :
        Diameter that is drawn at the full width, shared between drawings so
        that two of them may be compared.
    colors :
        Quantity to colour members by. If None, the diameters are used.
    vmin :
        Lower end of the colour range. If None, taken from the data.
    vmax :
        Upper end of the colour range. If None, taken from the data.

    Returns
    -------
    members :
        The drawn collection, for a colour bar to be attached to.
    """
    nodes = np.asarray(xyz)
    pairs = np.asarray(edges)
    sizes = np.asarray(diameters)

    segments = np.stack(
        [nodes[pairs[:, 0]][:, [0, 2]], nodes[pairs[:, 1]][:, [0, 2]]],
        axis=1,
    )
    values = sizes if colors is None else np.asarray(colors)

    members = LineCollection(
        segments,
        linewidths=WIDTH_MAX * sizes / widest,
        array=values,
        cmap="viridis",
        capstyle="round",
    )
    members.set_clim(
        values.min() if vmin is None else vmin,
        values.max() if vmax is None else vmax,
    )
    ax.add_collection(members)

    ax.plot(nodes[:, 0], nodes[:, 2], ".", color="0.2", markersize=2.5, zorder=3)
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z [mm]")

    return members


def figure_sections(
    xyz: Float[Array, "nodes 3"],
    edges: Int[Array, "members 2"],
    before: Float[Array, "members"],
    after: Float[Array, "members"],
    mass_before: float,
    mass_after: float,
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
        Diameter every member was assumed to have.
    after :
        Diameter EN 1993-1-1 requires of every member.
    mass_before :
        Total mass at the assumed diameters.
    mass_after :
        Total mass at the required diameters.

    Returns
    -------
    figure :
        Two arch drawings above a bar chart of the sizes.

    Notes
    -----
    The two drawings share one width scale and one colour range, so a member
    that shrank looks thinner rather than merely differently normalised.
    """
    widest = float(max(np.max(np.asarray(before)), np.max(np.asarray(after))))
    narrowest = float(min(np.min(np.asarray(before)), np.min(np.asarray(after))))

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12.0, 7.5),
        height_ratios=[2.0, 1.0],
        layout="constrained",
    )

    titles = (
        f"Assumed, uniform — {mass_before:.4f} t",
        f"Required by EN 1993-1-1 — {mass_after:.4f} t",
    )
    for ax, sizes, title in zip(axes[0], (before, after), titles):
        members = draw_members(
            ax, xyz, edges, sizes, widest, vmin=narrowest, vmax=widest
        )
        ax.set_title(title, fontsize=11)

    figure.colorbar(members, ax=axes[0].tolist(), label="diameter [mm]", shrink=0.85)

    shift = mass_after / mass_before - 1.0
    axes[0, 1].text(
        0.02,
        0.95,
        f"{abs(shift):.1%} {'lighter' if shift < 0.0 else 'heavier'}",
        transform=axes[0, 1].transAxes,
        va="top",
        fontsize=10,
    )

    span = axes[1, 0]
    index = np.arange(len(np.asarray(after)))
    span.bar(index - 0.2, np.asarray(before), 0.4, label="assumed", color=GREY)
    span.bar(index + 0.2, np.asarray(after), 0.4, label="required", color="#31688e")
    span.set_xlabel("member")
    span.set_ylabel("diameter [mm]")
    span.set_xticks(index)
    span.legend(frameon=False, fontsize=9)
    span.grid(axis="y", alpha=0.3)

    ratios = axes[1, 1]
    ratios.bar(index, np.asarray(after) / np.asarray(before), 0.6, color="#35b779")
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


def figure_convergence(
    counts: Int[np.ndarray, "meshes"],
    mass_member: Float[np.ndarray, "meshes"],
    mass_fixed: Float[np.ndarray, "meshes"],
    limit: float,
    passes: Int[np.ndarray, "passes"],
    moves: Float[np.ndarray, "passes"],
) -> Figure:
    """
    How the mass settles as the mesh refines, and as the staggering is repeated.

    Parameters
    ----------
    counts :
        Number of members in each mesh.
    mass_member :
        Total mass with each member buckling over its own length.
    mass_fixed :
        Total mass with a buckling length held independent of the mesh.
    limit :
        Mass the mesh-independent sequence extrapolates to.
    passes :
        Index of each pass through the staggered analysis and check.
    moves :
        Largest relative change in diameter produced by each pass.

    Returns
    -------
    figure :
        Three panels: the mass, its order of convergence, and the staggering.

    Notes
    -----
    The middle panel carries a first-order reference line rather than a fitted
    slope, so the reader compares against a claim instead of against a fit.
    """
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.4), layout="constrained")

    ax = axes[0]
    ax.plot(counts, mass_member, "o-", color="#440154", label=r"$L_{cr}$ = member")
    ax.plot(counts, mass_fixed, "s-", color="#31688e", label=r"$L_{cr}$ fixed")
    ax.axhline(
        limit,
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
    ax.semilogy(passes, moves, "o-", color="#35b779")
    ax.set_xlabel("pass through analysis and check")
    ax.set_ylabel("largest relative move in diameter")
    ax.set_title("The staggered coupling", fontsize=11)
    ax.grid(alpha=0.3, which="both")

    return figure


def figure_handoff(
    lengths: Float[Array, "members"],
    funicular: Float[Array, "members"],
    analysed: Float[Array, "members"],
    moments: Float[Array, "members"],
    diameters: Float[np.ndarray, "sizes"],
    gaps: Float[np.ndarray, "sizes"],
    reference: float,
    autodiff: Float[Array, "members"],
    central: Float[np.ndarray, "members"],
) -> Figure:
    """
    Whether form finding and the frame analysis agree, and why they cannot quite.

    Parameters
    ----------
    lengths :
        Length of every member.
    funicular :
        Axial force form finding predicts, being force density times length.
    analysed :
        Axial force the frame analysis reports.
    moments :
        Largest end moment of every member, in magnitude.
    diameters :
        Diameters the disagreement was measured at.
    gaps :
        Worst relative disagreement at each of those diameters.
    reference :
        Diameter the quadratic reference line is anchored at.
    autodiff :
        Gradient of a scalar of the analysis with respect to force density.
    central :
        The same gradient by central differences.

    Returns
    -------
    figure :
        Four panels: the forces, the disagreement, its law, and the gradient.
    """
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), layout="constrained")

    ax = axes[0, 0]
    index = np.arange(len(np.asarray(funicular)))
    ax.plot(index, np.asarray(funicular) / 1e3, "o-", color="#440154", label=r"$q\,L$")
    ax.plot(index, np.asarray(analysed) / 1e3, "x--", color="#fde725", label="smax")
    ax.set_xlabel("member")
    ax.set_ylabel("axial force [kN]")
    ax.set_title("Form finding against the frame analysis", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    gap = np.abs(np.asarray(analysed) - np.asarray(funicular)) / np.abs(
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
    sizes = np.asarray(diameters)
    measured = np.asarray(gaps)
    anchor = measured[np.argmin(np.abs(sizes - reference))]
    ax.loglog(sizes, measured, "o-", color="#440154", label="measured")
    ax.loglog(
        sizes,
        anchor * (sizes / reference) ** 2,
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
    exact = np.asarray(autodiff)
    numeric = np.asarray(central)
    ax.plot(index, exact, "o", color="#440154", label="autodiff")
    ax.plot(index, numeric, "x", color="#fde725", markersize=9, label="central")
    ax.set_xlabel("edge")
    ax.set_ylabel(r"$\partial \, \Sigma N^2 / \partial q$")
    worst = np.max(np.abs(exact - numeric) / np.abs(numeric))
    ax.set_title(f"Gradient across both stages, worst {worst:.1e}", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3)

    return figure


def figure_modes(
    xyz: Float[Array, "nodes 3"],
    factors: Float[np.ndarray, "modes"],
    shapes: Float[np.ndarray, "modes nodes 6"],
    height: float,
) -> Figure:
    """
    The shapes a frame buckles into, and the load factor of each.

    Parameters
    ----------
    xyz :
        Position of every node at equilibrium.
    factors :
        Multiple of the applied load at which each mode becomes critical.
    shapes :
        Displacement of every node in each mode.
    height :
        Extent of the structure the modes are scaled against.

    Returns
    -------
    figure :
        One panel per mode, the undeformed shape behind each.

    Notes
    -----
    Each mode is scaled to a fixed fraction of the structure's height, because
    an eigenvector carries no amplitude of its own. The vertical component is
    read for symmetry: a mode with a node at midspan is the antisymmetric one,
    and for a two-pinned arch that is the mode that governs.
    """
    modes = np.asarray(factors).shape[0]
    columns = min(modes, 2)
    rows = int(np.ceil(modes / columns))

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(6.0 * columns, 3.2 * rows),
        squeeze=False,
        layout="constrained",
    )

    nodes = np.asarray(xyz)
    middle = nodes.shape[0] // 2

    for mode, ax in enumerate(axes.ravel()):
        if mode >= modes:
            ax.axis("off")
            continue

        shape = np.asarray(shapes)[mode]
        amplitude = np.max(np.abs(shape[:, [0, 2]]))
        push = 0.22 * height / max(amplitude, 1e-30)

        ax.plot(nodes[:, 0], nodes[:, 2], "-o", color=GREY, ms=3, lw=1.2)
        ax.plot(
            nodes[:, 0] + shape[:, 0] * push,
            nodes[:, 2] + shape[:, 2] * push,
            "-o",
            color="#440154",
            ms=3,
            lw=1.8,
        )

        crown = abs(shape[middle, 2])
        kind = "antisymmetric" if crown < 0.1 * amplitude else "symmetric"
        ax.set_title(
            rf"mode {mode}:  $\alpha_{{cr}}$ = {float(factors[mode]):.4f}  ({kind})",
            fontsize=10,
        )
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("z [mm]")

    return figure
