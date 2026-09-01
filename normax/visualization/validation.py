# SPDX-License-Identifier: Apache-2.0
"""Numerical validation figures in the same visual language as Normax plots.

Every function accepts plain arrays and returns a matplotlib figure.  The
measurements, filesystem exports, and provenance records live elsewhere, so
these functions are deterministic drawing transforms and never call ``show``.
"""

from collections.abc import Sequence
from functools import wraps

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter

from normax.visualization.plots import FAINT
from normax.visualization.plots import GREY
from normax.visualization.plots import GROUND
from normax.visualization.plots import INK
from normax.visualization.plots import MUTED
from normax.visualization.plots import SHADES
from normax.visualization.plots import paint_figure

FIGURE_SIZE = (12.6, 3.55)
MARKER_SIZE = 28.0
LINE_WIDTH = 1.25

# Computer Modern as a font bundled with matplotlib rather than through
# ``text.usetex``: the figures keep the paper's LaTeX character without making
# a TeX installation part of their runtime or vector-export contract. STIX
# supplies the mathematical minus that the historical Roman text face lacks.
VALIDATION_STYLE = {
    "font.family": ("cmr10", "serif"),
    "font.weight": "normal",
    "mathtext.fontset": "stix",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
}


def latex_serif(function):
    """Draw one complete figure under the local validation-paper typography."""

    @wraps(function)
    def styled(*args, **kwargs):
        with mpl.rc_context(VALIDATION_STYLE):
            return function(*args, **kwargs)

    return styled


def finish_axis(ax: Axes, *, grid: str = "y") -> None:
    """Apply the full-frame engineering-paper chrome to one panel."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
    ax.grid(axis=grid, color=FAINT, alpha=0.18, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.title.set_fontsize(10)
    ax.title.set_fontweight("normal")
    ax.title.set_x(0.02)
    ax.title.set_ha("left")
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        labelsize=8,
        length=3.5,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=2.0,
    )
    ax.xaxis.label.set_size(8.5)
    ax.yaxis.label.set_size(8.5)


def annotate_bars(ax: Axes, bars, values: np.ndarray) -> None:
    """Write compact measured values above logarithmic timing bars."""
    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.3g} s",
            (bar.get_x() + 0.5 * bar.get_width(), value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=MUTED,
        )
    # Headroom over the tallest bar, so its label clears the axes frame.
    tallest = float(np.max(values))
    low, high = ax.get_ylim()
    ax.set_ylim(low, max(high, 2.0 * tallest))


def draw_error_budget(
    ax: Axes,
    labels: Sequence[str],
    errors: Sequence[float],
    tolerances: Sequence[float],
) -> None:
    """Draw measured errors against their declared numerical bounds."""
    bounds = np.asarray(tolerances, dtype=float)
    raw = np.asarray(errors, dtype=float)
    positive = np.concatenate([raw[raw > 0.0], bounds[bounds > 0.0]])
    floor = float(np.min(positive)) / 5.0
    found = np.maximum(raw, floor)
    positions = np.arange(len(labels))
    for y, error, bound in zip(positions, found, bounds):
        ax.plot([error, bound], [y, y], color=FAINT, alpha=0.42, linewidth=0.8)
    ax.scatter(found, positions, s=MARKER_SIZE, color=SHADES[0], zorder=3)
    ax.scatter(
        bounds,
        positions,
        s=MARKER_SIZE + 7,
        marker="|",
        color=INK,
        linewidths=1.4,
        zorder=3,
    )
    ax.set_xscale("log")
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel(r"scaled error  $\cdot$  black tick = declared bound")
    finish_axis(ax)


@latex_serif
def draw_pipeline_validation(
    reverse: Sequence[float],
    central: Sequence[float],
    parameter_kinds: Sequence[str],
    error_labels: Sequence[str],
    errors: Sequence[float],
    tolerances: Sequence[float],
    timing_labels: Sequence[str],
    timing_seconds: Sequence[float],
) -> Figure:
    """Signed end-to-end parity, error budget, and measured computational cost."""
    reverse_host = np.asarray(reverse, dtype=float)
    central_host = np.asarray(central, dtype=float)
    kinds = np.asarray(parameter_kinds)
    figure, axes = plt.subplots(1, 3, figsize=FIGURE_SIZE, layout="constrained")

    ax = axes[0]
    limit = 1.08
    ax.plot([-limit, limit], [-limit, limit], color=GREY, linewidth=0.9, zorder=1)
    for index, kind in enumerate(dict.fromkeys(kinds)):
        selected = kinds == kind
        scale = max(float(np.max(np.abs(central_host[selected]))), np.finfo(float).tiny)
        ax.scatter(
            central_host[selected] / scale,
            reverse_host[selected] / scale,
            s=MARKER_SIZE,
            facecolors=GROUND if index else SHADES[index],
            edgecolors=SHADES[index],
            linewidths=1.1,
            label=kind,
            zorder=3,
        )
    ax.axhline(0.0, color=FAINT, linewidth=0.6, alpha=0.35)
    ax.axvline(0.0, color=FAINT, linewidth=0.6, alpha=0.35)
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("central difference, normalized")
    ax.set_ylabel("reverse mode, normalized")
    ax.set_title("1  Signed gradient parity")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    finish_axis(ax)

    axes[1].set_title("2  Error against declared bounds")
    draw_error_budget(axes[1], error_labels, errors, tolerances)

    ax = axes[2]
    timing = np.asarray(timing_seconds, dtype=float)
    bars = ax.bar(
        np.arange(len(timing)),
        timing,
        color=(FAINT, SHADES[0], SHADES[1])[: len(timing)],
        width=0.64,
    )
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(timing)), timing_labels)
    ax.set_ylabel("wall time [s], log scale")
    ax.set_title("3  Actual warmed wall time")
    annotate_bars(ax, bars, timing)
    finish_axis(ax)

    paint_figure(figure)

    return figure


@latex_serif
def draw_code_validation(
    force_errors: Sequence[float],
    moment_errors: Sequence[float],
    case_labels: Sequence[str],
    sharpness: Sequence[float],
    envelope_excess: Sequence[float],
    envelope_bound: Sequence[float],
    check_labels: Sequence[str],
    check_errors: Sequence[float],
    check_tolerances: Sequence[float],
) -> Figure:
    """Four-way code derivatives, smooth envelope convergence, and route checks."""
    figure, axes = plt.subplots(1, 3, figsize=FIGURE_SIZE, layout="constrained")

    ax = axes[0]
    positions = np.arange(len(case_labels))
    ax.semilogy(
        positions,
        np.maximum(force_errors, np.finfo(float).tiny),
        "o-",
        color=SHADES[0],
        linewidth=LINE_WIDTH,
        markersize=4.5,
        label=r"$\partial d/\partial N$",
    )
    ax.semilogy(
        positions,
        np.maximum(moment_errors, np.finfo(float).tiny),
        "o-",
        color=SHADES[1],
        markerfacecolor=GROUND,
        linewidth=LINE_WIDTH,
        markersize=4.5,
        label=r"$\partial d/\partial M$",
    )
    ax.axhline(1e-8, color=INK, linewidth=0.8, linestyle="--", label="target")
    ax.set_xticks(positions, case_labels)
    ax.set_xlabel("member case")
    ax.set_ylabel("worst relative disagreement")
    ax.set_title("1  Code derivative, four ways")
    ax.legend(frameon=False, fontsize=7.2)
    finish_axis(ax)

    ax = axes[1]
    beta = np.asarray(sharpness, dtype=float)
    ax.loglog(
        beta,
        envelope_bound,
        color=FAINT,
        linewidth=0.9,
        linestyle="--",
        label=r"$\log(n)/\beta$ bound",
    )
    ax.loglog(
        beta,
        envelope_excess,
        "o-",
        color=SHADES[0],
        linewidth=LINE_WIDTH,
        markersize=4.5,
        label="measured mass excess",
    )
    ax.set_xlabel(r"smooth-max sharpness $\beta$")
    ax.set_ylabel("relative excess")
    ax.set_title("2  Conservative envelope annealing")
    ax.legend(frameon=False, fontsize=7.2)
    finish_axis(ax)

    axes[2].set_title("3  Boundary and branch checks")
    draw_error_budget(axes[2], check_labels, check_errors, check_tolerances)

    paint_figure(figure)

    return figure


@latex_serif
def draw_pynite_validation(
    steps: Sequence[float],
    node_errors: Sequence[float],
    diameter_errors: Sequence[float],
    route_labels: Sequence[str],
    route_errors: Sequence[float],
    route_tolerances: Sequence[float],
    timing_labels: Sequence[str],
    timing_seconds: Sequence[float],
    finite_difference_measured: bool,
) -> Figure:
    """Step convergence, route parity, and scaling of the PyNite adjoint."""
    figure, axes = plt.subplots(1, 3, figsize=FIGURE_SIZE, layout="constrained")

    ax = axes[0]
    ax.loglog(
        steps,
        node_errors,
        "o-",
        color=SHADES[0],
        linewidth=LINE_WIDTH,
        markersize=4.5,
        label="nodal coordinates",
    )
    ax.loglog(
        steps,
        diameter_errors,
        "o-",
        color=SHADES[1],
        markerfacecolor=GROUND,
        linewidth=LINE_WIDTH,
        markersize=4.5,
        label="diameters",
    )
    ax.set_xlabel("relative difference step")
    ax.set_ylabel("scaled gradient error")
    ax.set_title("1  Central-difference step sweep")
    ax.legend(frameon=False, fontsize=7.2)
    finish_axis(ax)

    axes[1].set_title("2  Adjoint route agreement")
    draw_error_budget(axes[1], route_labels, route_errors, route_tolerances)

    ax = axes[2]
    timing = np.asarray(timing_seconds, dtype=float)
    bars = ax.bar(
        np.arange(len(timing)),
        timing,
        color=(FAINT, SHADES[0], SHADES[1], SHADES[2])[: len(timing)],
        width=0.64,
    )
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(timing)), timing_labels)
    ax.set_ylabel("wall time [s], log scale")
    qualifier = "measured" if finite_difference_measured else "FD projected"
    ax.set_title(f"3  Cost at 1267 parameters ({qualifier})")
    annotate_bars(ax, bars, timing)
    finish_axis(ax)

    paint_figure(figure)

    return figure
