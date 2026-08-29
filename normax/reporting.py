# SPDX-License-Identifier: Apache-2.0
"""
What a run prints, in one place and one style.

A table states each column once — a heading and the format its cells take — and
the widths follow from the text actually printed. Every writer carries its own
verbosity, so a caller silences a whole report by constructing a quiet one.
"""

import textwrap
from collections.abc import Sequence
from typing import Any
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from normax.config import RunConfig
from normax.design import Design
from normax.design import DesignProblem
from normax.design import ProblemRecord
from normax.design import compute_compliance
from normax.design import compute_mass
from normax.design import compute_mass_problem
from normax.optimization import OptimizationSolution

# Spaces of indentation given to anything printed under a heading.
INDENT = "  "

# Width of the rules a banner is drawn between.
RULE_WIDTH = 78

CellValue = float | int | str


class ReportColumn(NamedTuple):
    """
    One column of a printed table.

    Attributes
    ----------
    heading :
        Text of the column header.
    format_spec :
        Format specification its cells are rendered with, braces excluded. A
        cell that is already a string is printed as it stands, so a column may
        leave an entry blank without carrying a second format.
    align :
        Alignment of both the heading and the cells, as a format-specification
        character.
    """

    heading: str
    format_spec: str = ""
    align: str = ">"


class ToleranceCheck(NamedTuple):
    """
    One measured quantity, beside the bound it has to stay under.

    Attributes
    ----------
    label :
        What was measured.
    worst :
        Largest value the measurement reached.
    tolerance :
        Bound the measurement is asserted against.
    """

    label: str
    worst: float
    tolerance: float

    @property
    def satisfied(self) -> bool:
        """
        Whether the measurement stayed under its bound.
        """
        return self.worst < self.tolerance


def format_cell(value: CellValue, column: ReportColumn) -> str:
    """
    Render one cell, leaving text that arrives already rendered alone.
    """
    if isinstance(value, str):
        return value

    return format(value, column.format_spec)


def format_table(
    columns: Sequence[ReportColumn],
    rows: Sequence[Sequence[CellValue]],
) -> list[str]:
    """
    A heading and its rows, every column widened to the text it has to hold.
    """
    rendered = [
        [format_cell(value, column) for value, column in zip(row, columns)]
        for row in rows
    ]

    widths = []
    for index, column in enumerate(columns):
        cells = [row[index] for row in rendered]
        widths.append(max(len(column.heading), *(len(cell) for cell in cells), 0))

    def joined(cells: Sequence[str]) -> str:
        parts = [
            f"{cell:{column.align}{width}}"
            for cell, column, width in zip(cells, columns, widths)
        ]

        return (INDENT + " ".join(parts)).rstrip()

    lines = [joined([column.heading for column in columns])]
    lines.extend(joined(row) for row in rendered)

    return lines


def verify_checks(checks: Sequence[ToleranceCheck]) -> bool:
    """
    Whether every measurement stayed under its bound.
    """
    return all(check.satisfied for check in checks)


class Report:
    """
    Prints a report, or stays silent throughout when it is not verbose.

    Attributes
    ----------
    verbose :
        Whether anything is printed at all.
    started :
        Whether a line has been written yet, which is what keeps a heading from
        opening the report with a blank line.
    """

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose
        self.started = False

    def write_line(self, text: str = "") -> None:
        """
        One line, exactly as given.
        """
        if not self.verbose:
            return

        print(text)
        self.started = True

    def write_heading(self, title: str) -> None:
        """
        A section title, separated from whatever came before it.
        """
        if not self.verbose:
            return

        if self.started:
            self.write_line()
        self.write_line(title)

    def write_banner(self, title: str) -> None:
        """
        A title between rules, for a pass that is a report in its own right.
        """
        rule = "=" * RULE_WIDTH
        self.write_heading(rule)
        self.write_line(title)
        self.write_line(rule)

    def write_note(self, text: str) -> None:
        """
        Prose under the current heading, rewrapped so the source may wrap freely.

        Blank lines separate paragraphs and every other line break is the
        author's convenience rather than the reader's, so the text is filled
        again to the width of a banner rather than printed as it was written.
        """
        if not self.verbose:
            return

        if self.started:
            self.write_line()

        width = RULE_WIDTH - len(INDENT)
        paragraphs = textwrap.dedent(text).strip().split("\n\n")
        for index, paragraph in enumerate(paragraphs):
            if index:
                self.write_line()
            for line in textwrap.wrap(paragraph, width):
                self.write_line(INDENT + line)

    def write_entries(self, entries: Sequence[tuple[str, str]]) -> None:
        """
        Labeled values, the labels widened to the longest of them.
        """
        if not self.verbose:
            return

        width = max((len(label) for label, _ in entries), default=0)
        for label, value in entries:
            self.write_line(f"{INDENT}{label:<{width}} {value}".rstrip())

    def write_table(
        self,
        columns: Sequence[ReportColumn],
        rows: Sequence[Sequence[CellValue]],
    ) -> None:
        """
        A heading row and its rows, aligned to each other.
        """
        if not self.verbose:
            return

        for line in format_table(columns, rows):
            self.write_line(line)

    def write_checks(self, checks: Sequence[ToleranceCheck]) -> None:
        """
        Every measurement beside the bound it was asserted against.
        """
        columns = (
            ReportColumn("measurement", align="<"),
            ReportColumn("worst", ".2e"),
            ReportColumn("of", ".1e"),
        )
        rows = [(check.label, check.worst, check.tolerance) for check in checks]
        self.write_table(columns, rows)

    def write_verdict(self, passed: bool) -> None:
        """
        The one word the experiment is read for.
        """
        self.write_heading("PASS" if passed else "FAIL")


def name_objective(problem: DesignProblem) -> str:
    """
    The column heading this problem's objective is printed under.

    Parameters
    ----------
    problem :
        The problem, read for what its search minimizes.

    Returns
    -------
    heading :
        Name and unit of the objective, or a bare `objective` for one this
        module does not know.

    Notes
    -----
    The objective is a slot, so the heading cannot be a constant: a compliance
    search printed under `mass [t]` reports a real number against the wrong
    name, which is worse than printing no name at all.
    """
    if problem.objective is compute_mass_problem:
        return "mass [t]"
    if getattr(problem.objective, "__name__", "") == "compute_compliance_problem":
        return "compliance [N mm]"

    return "objective"


def report_descent(
    report: Report,
    solution: OptimizationSolution,
    heading: str = "mass [t]",
) -> None:
    """
    Every round of a descent, its objective beside its violation.

    Parameters
    ----------
    report :
        Where to print.
    solution :
        What the descent arrived at.
    heading :
        What the objective column is called, from `name_objective`.
    """
    columns = (
        ReportColumn("round"),
        ReportColumn(heading, ".6f"),
        ReportColumn("violation", ".2e"),
    )
    coarse = solution.rounds
    walked = zip(coarse.objectives, coarse.violations, strict=True)
    rows = [(index, value, gap) for index, (value, gap) in enumerate(walked)]
    report.write_table(columns, rows)

    ended = "converged" if solution.converged else "stopped on its round budget"
    entries = [("evaluations", str(solution.evaluations)), ("ended", ended)]
    if solution.iterations is not None:
        traced = np.size(solution.iterations.objectives)
        entries.insert(1, ("iterations traced", str(traced)))
    report.write_entries(entries)


def summarize_design(report: Report, design: Design, title: str) -> None:
    """
    What a design weighs, how hard it is worked, and how it sits.

    Parameters
    ----------
    report :
        Where to print.
    design :
        The design to read.
    title :
        Heading the entries are printed under.

    Notes
    -----
    A design whose pipeline was cut short is reported for what it holds: the
    mass and the utilization go with the check, the compliance with the
    analysis, and the geometry is always there.
    """
    heights = np.asarray(design.shape.xyz)[:, 2]
    entries = []
    if design.sizes is not None:
        diameters = np.asarray(design.sizes.sections.diameter)
        worst = float(jnp.max(design.sizes.utilization))
        entries.append(("mass [t]", f"{float(compute_mass(design)):.6f}"))
        entries.append(("utilization, worst", f"{worst:.6f}"))
        entries.append(
            ("diameters [mm]", f"{diameters.min():.1f} to {diameters.max():.1f}")
        )
    if design.forces is not None and design.sizes is not None:
        strained = float(compute_compliance(design))
        entries.append(("compliance [N mm]", f"{strained:.6e}"))
    entries.append(
        ("shortest member [mm]", f"{float(jnp.min(design.shape.lengths)):.1f}")
    )
    entries.append(("heights [mm]", f"{heights.min():.1f} to {heights.max():.1f}"))
    report.write_heading(title)
    report.write_entries(entries)


def report_families(
    report: Report,
    design: Design,
    families: Sequence[tuple[str, slice]],
) -> None:
    """
    Diameter and utilization ranges per member family.

    Parameters
    ----------
    report :
        Where to print.
    design :
        The design to read.
    families :
        Name and member slice of every family.
    """
    if design.sizes is None:
        return

    diameters = np.asarray(design.sizes.sections.diameter)
    worked = np.asarray(jnp.max(design.sizes.utilization, axis=0))
    columns = (
        ReportColumn("family", align="<"),
        ReportColumn("d min [mm]", ".1f"),
        ReportColumn("d max [mm]", ".1f"),
        ReportColumn("U min", ".4f"),
        ReportColumn("U max", ".4f"),
    )
    rows = []
    for name, members in families:
        sizes = diameters[members]
        used = worked[members]
        rows.append((name, sizes.min(), sizes.max(), used.min(), used.max()))
    report.write_table(columns, rows)


def list_unused_settings(config: RunConfig[Any]) -> tuple[str, ...]:
    """
    The form-finding settings this run's parametrization never read.

    Parameters
    ----------
    config :
        The run config, read for its parametrization and what the file set.

    Returns
    -------
    unused :
        Names of the settings the file gave and the parametrization ignored,
        in the order a file writes them, or nothing where all were read.

    Notes
    -----
    A parametrization may be named on the command line, over a file written for
    another one, so a setting can go unread without anything being wrong. It is
    reported rather than refused: refusing would make the override useless on
    every file that ships, and staying silent would let a density box or a sign
    guard look honored when it was not.
    """
    found = config.form_finding
    written = found.shape_parametrization != "fdm"

    named = {
        "basis": written and found.basis is not None,
        "density_start": written and bool(found.density_start),
        "bounds": written and config.constraints.bounds is not None,
        "sign_guard": written and config.constraints.sign_guard is not None,
        "height_start": (
            found.shape_parametrization != "heights" and bool(found.height_start)
        ),
    }

    return tuple(setting for setting, given in named.items() if given)


def report_design(
    record: ProblemRecord,
    config: RunConfig[Any],
    title: str,
) -> None:
    """
    The whole report of a run, or nothing when the run is not verbose.

    Parameters
    ----------
    record :
        What the run arrived at.
    config :
        The run config, naming its backends and whether it prints.
    title :
        Banner the report opens with.
    """
    report = Report(verbose=config.output.verbose)
    report.write_banner(title)

    entries = [
        ("shape", config.form_finding.shape_parametrization),
        ("analysis", config.analysis.backend),
        ("sizing", config.sizing.backend),
    ]
    basis = record.problem.pipeline.formfinder.basis
    if basis is not None:
        entries.append(("coefficients", str(basis.width)))
    entries.append(("parameters", str(record.solution.parameters.size)))
    report.write_heading("Backends")
    report.write_entries(entries)

    unused = list_unused_settings(config)
    if unused:
        named = ", ".join(unused)
        report.write_note(
            f"This parametrization read none of: {named}. The file describes "
            "a form-found run and the shape was taken another way, so those "
            "settings are carried and unused rather than honored."
        )

    report.write_heading("The descent")
    report_descent(report, record.solution, name_objective(record.problem))
    summarize_design(report, record.initial, "The start")
    summarize_design(report, record.optimized, "The answer")
    if record.families:
        report_families(report, record.optimized, record.families)

    if record.optimized.sizes is None:
        return

    mass_initial = float(compute_mass(record.initial))
    mass_optimized = float(compute_mass(record.optimized))
    saved = 1.0 - mass_optimized / mass_initial
    saving = [("saved", f"{100.0 * saved:.2f} %")]
    report.write_entries(saving)
