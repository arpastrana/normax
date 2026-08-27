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
from normax.design import DesignRecord
from normax.loads import label_load_cases
from normax.reporting import Report
from normax.visualization import draw_design_figures

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


def export_design(
    record: DesignRecord,
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
    The record carries the variable vector the descent stopped on and the
    objective and violation of every round, enough to replay the descent's
    figure or to read the answer back into the problem that produced it.
    """
    if not config.output.export:
        return

    answer = record.answer
    target.data.mkdir(exist_ok=True)
    archive = target.data / f"{target.name}.npz"
    np.savez(
        archive,
        variables=answer.variables,
        objectives=answer.objectives,
        violations=answer.violations,
    )

    target.figures.mkdir(exist_ok=True)
    designs = {"start": record.initial, "answer": record.optimized}
    labels = label_load_cases(config.load_cases)
    structure = record.problem.structure
    drawn, descended = draw_design_figures(structure, designs, labels, answer)
    drawn.savefig(target.figures / f"{target.name}_designs.png", dpi=FIGURE_DPI)
    descended.savefig(target.figures / f"{target.name}_descent.png", dpi=FIGURE_DPI)

    # Continues the run's report, so the heading separates itself from it.
    report = Report(verbose=config.output.verbose)
    report.started = True
    report.write_heading("The record")
    written = [("figures", str(target.figures)), ("data", str(archive))]
    report.write_entries(written)
