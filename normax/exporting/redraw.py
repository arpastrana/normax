# SPDX-License-Identifier: Apache-2.0
"""
Draw a run's figures again from what it archived, without searching again.

`normax.exporting.records.export_design` writes the figures at the end of a
search, which ties every change of drawing style to a fresh descent. On a
landscape with more than one basin that is not merely slow: a re-run can land
somewhere else and quietly move the numbers a figure is captioning. The archive
holds every iterate the descent passed through, so the designs and the curves
can be rebuilt exactly, and a style is then iterated on at one forward pass a
frame with the answer held fixed.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from normax.config import RunConfig
from normax.design import DesignProblem
from normax.design import create_design
from normax.exporting.records import FIGURE_DPI
from normax.loads import number_load_cases
from normax.optimization.auglag import DescentHistory
from normax.reporting import name_objective
from normax.visualization import DescentPanel
from normax.visualization import DescentTrace
from normax.visualization import DrawnLimits
from normax.visualization import animate_descent
from normax.visualization import draw_design_figures
from normax.visualization import save_animation


def read_descent_archive(archive: Path) -> DescentHistory:
    """
    The walk a run recorded, read back off disk.

    Parameters
    ----------
    archive :
        The `.npz` a run wrote, carrying the four walk columns.

    Returns
    -------
    history :
        The walk, at whatever resolution the run kept it.

    Raises
    ------
    ValueError
        If the archive is missing a column a figure reads.
    """
    stored = np.load(archive)
    wanted = ("iterates", "objectives", "violations", "round_index")
    missing = [name for name in wanted if name not in stored.files]
    if missing:
        raise ValueError(f"{archive} carries no {missing}, so no walk to draw")

    return DescentHistory(
        stored["iterates"],
        stored["objectives"],
        stored["violations"],
        stored["round_index"],
    )


def build_descent_panel(
    problem: DesignProblem,
    history: DescentHistory,
    config: RunConfig[Any],
) -> DescentPanel:
    """
    The panel a figure and an animation both read the descent from.

    Parameters
    ----------
    problem :
        The problem the walk belongs to, read for what it minimized.
    history :
        The walk to draw.
    config :
        The run description, read for the violation tolerance.

    Returns
    -------
    panel :
        One trace, named as the search that produced it.

    Notes
    -----
    The axis is called `iteration` where the archive holds more points than
    there are rounds in it, which is how a run recorded per inner iteration is
    told from one recorded per round without being asked.
    """
    rounds = int(np.max(np.asarray(history.round_index))) + 1
    axis = "iteration" if np.size(history.objectives) > rounds else "round"
    trace = DescentTrace("auglag", history, config.optimization.violation_tol)

    return DescentPanel(name_objective(problem), axis, (trace,))


def redraw_run(
    problem: DesignProblem,
    config: RunConfig[Any],
    archive: Path,
    figures: Path,
    limits: DrawnLimits | None = None,
) -> tuple[Path, ...]:
    """
    Every figure a run writes, drawn again from its archive.

    Parameters
    ----------
    problem :
        The problem the archive belongs to, built as the run built it.
    config :
        The run description, read for its load cases and whether to animate.
    archive :
        The `.npz` to read the walk from; its stem names the files written.
    figures :
        Directory the figures go in.
    limits :
        Limits to hold every figure to, or None to read them off this run
        alone. Held, so several runs of one structure come out at one aspect
        ratio, one pace and one set of ticks.

    Returns
    -------
    written :
        Path of every file written, in the order written.

    Notes
    -----
    Every figure is closed once written, so redrawing many runs in one process
    does not accumulate them.

    The start and the answer are the walk's first and last iterates rather than
    anything recomputed, so a redrawn figure shows the design the search really
    reached. The load cases and the tolerance come from the run description,
    which must therefore be the one the archive was written under.
    """
    history = read_descent_archive(archive)
    started = create_design(problem, history.iterates[0])
    arrived = create_design(problem, history.iterates[-1])
    designs = {"start": started, "solution": arrived}
    panel = build_descent_panel(problem, history, config)

    figures.mkdir(parents=True, exist_ok=True)
    stem = archive.stem
    drawn = draw_design_figures(
        problem.structure,
        designs,
        number_load_cases(config.load_cases),
        panel,
        limits,
    )

    written = []
    if drawn.designs is not None:
        path = figures / f"{stem}_designs.png"
        drawn.designs.savefig(path, dpi=FIGURE_DPI)
        written.append(path)
    if drawn.load_cases is not None:
        path = figures / f"{stem}_load_cases.png"
        drawn.load_cases.savefig(path, dpi=FIGURE_DPI)
        written.append(path)
    path = figures / f"{stem}_optimization.png"
    drawn.optimization.savefig(path, dpi=FIGURE_DPI)
    written.append(path)

    # Closed once saved: pyplot holds every figure it made, and a caller
    # redrawing a dozen runs in one process otherwise carries all of them.
    for figure in (drawn.designs, drawn.load_cases, drawn.optimization):
        if figure is not None:
            plt.close(figure)

    if config.output.animate:
        path = figures / f"{stem}_optimization.mp4"
        save_animation(animate_descent(problem, panel, limits), path)
        written.append(path)

    return tuple(written)
