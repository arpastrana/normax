# SPDX-License-Identifier: Apache-2.0
"""
What a run writes to disk: its record, and its figures.

One call exports a whole run or nothing, so an example's `main` stays the
computation and the descent, and the writing rides on a flag in its file.
"""

from pathlib import Path
from typing import Any
from typing import NamedTuple

import numpy as np

from normax.config import RunConfig
from normax.design import ProblemRecord
from normax.loads import number_load_cases
from normax.reporting import Report
from normax.reporting import name_objective
from normax.visualization import DescentPanel
from normax.visualization import DescentTrace
from normax.visualization import animate_descent
from normax.visualization import draw_design_figures
from normax.visualization import save_animation

# Resolution every figure is written at.
FIGURE_DPI = 200


class ExportTarget(NamedTuple):
    """
    Where a run's files go, and what they are named after.

    Attributes
    ----------
    name :
        Stem every written file starts with.
    data :
        Folder the record is written into.
    figures :
        Folder the figures are written into.
    """

    name: str
    data: Path
    figures: Path


def read_export_stem(target: ExportTarget, config: RunConfig[Any]) -> str:
    """
    The stem this run's files are named with.

    Parameters
    ----------
    target :
        Where the files go, and what they are named after.
    config :
        The run config, read for its shape parametrization.

    Returns
    -------
    stem :
        The target's name, suffixed by the parametrization wherever the shape
        was not form-found.

    Notes
    -----
    One example runs three parametrizations now, so a bare structure name would
    have each overwrite the last and leave a record that does not say which
    search made it. The form-found run keeps the plain name, being the one the
    reported numbers belong to.
    """
    word = config.form_finding.shape_parametrization
    if word == "fdm":
        return target.name

    return f"{target.name}_{word}"


def export_design(
    record: ProblemRecord,
    config: RunConfig[Any],
    target: ExportTarget,
) -> None:
    """
    A run's record and figures on disk, or nothing when the run asks none.

    Parameters
    ----------
    record :
        What the run arrived at.
    config :
        The run config, naming its load cases and whether it exports.
    target :
        Where the files go.

    Notes
    -----
    The record carries the walk at the finest resolution the descent kept it —
    every inner iteration where the budget asked for them, every round
    otherwise — with the objective, the violation and the round each point came
    out of. Enough to replay the descent's figure, to read the answer back into
    the problem that produced it, or to carry every point back through the
    pipeline for the design behind it.
    """
    if not config.output.export:
        return

    solution = record.solution
    stem = read_export_stem(target, config)
    target.data.mkdir(exist_ok=True)
    archive = target.data / f"{stem}.npz"
    # The finer walk where the descent kept one, since the coarse one is its
    # own subsequence and a figure reads either without asking which it got.
    walked = solution.rounds if solution.iterations is None else solution.iterations
    columns = {
        "parameters": solution.parameters,
        "iterates": walked.iterates,
        "objectives": walked.objectives,
        "violations": walked.violations,
        "round_index": walked.round_index,
    }
    np.savez(archive, **columns)

    target.figures.mkdir(exist_ok=True)
    # The start first, so a drawing reads from where the search left from to
    # what it arrived at: left to right across the load cases, top to bottom
    # down the designs.
    designs = {"start": record.initial, "solution": record.optimized}
    labels = number_load_cases(config.load_cases)
    structure = record.problem.structure
    trace = DescentTrace("auglag", walked, config.optimization.violation_tol)
    axis = "round" if solution.iterations is None else "iteration"
    panel = DescentPanel(name_objective(record.problem), axis, (trace,))
    figures = draw_design_figures(structure, designs, labels, panel)
    if figures.designs is not None:
        figures.designs.savefig(target.figures / f"{stem}_designs.png", dpi=FIGURE_DPI)
    if figures.load_cases is not None:
        governed = target.figures / f"{stem}_load_cases.png"
        figures.load_cases.savefig(governed, dpi=FIGURE_DPI)
    optimized = target.figures / f"{stem}_optimization.png"
    figures.optimization.savefig(optimized, dpi=FIGURE_DPI)

    played = None
    if config.output.animate:
        played = target.figures / f"{stem}_optimization.mp4"
        spun = animate_descent(record.problem, panel, None, config.output.turns)
        save_animation(spun, played)

    # Continues the run's report, so the heading separates itself from it.
    report = Report(verbose=config.output.verbose)
    report.started = True
    report.write_heading("The record")
    written = [("figures", str(target.figures)), ("data", str(archive))]
    if played is not None:
        written.append(("animation", str(played)))
    report.write_entries(written)
