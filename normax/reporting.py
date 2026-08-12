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
What the experiments print, in one place and one style.

The experiments measure and this module prints, as `normax.visualization` draws.
A table states each column once — a heading and the format its cells take — and
the widths follow from the text that is actually printed, so nothing is padded by
hand and no heading can drift out of step with the row beneath it.

Every writer carries its own verbosity. A quiet writer returns from each call
without printing, so a caller silences a whole report by constructing one rather
than by threading a flag through the functions that compute.
"""

import textwrap
from collections.abc import Sequence
from typing import NamedTuple

# Spaces of indentation given to anything printed under a heading.
INDENT = "  "

# Width of the rules a banner is drawn between.
RULE_WIDTH = 78

CellValue = float | int | str


class ColumnSpec(NamedTuple):
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


def format_cell(value: CellValue, column: ColumnSpec) -> str:
    """
    Render one cell, leaving text that arrives already rendered alone.
    """
    if isinstance(value, str):
        return value

    return format(value, column.format_spec)


def table_lines(
    columns: Sequence[ColumnSpec],
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


def checks_passed(checks: Sequence[ToleranceCheck]) -> bool:
    """
    Whether every measurement stayed under its bound.
    """
    return all(check.satisfied for check in checks)


class ReportWriter:
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
        columns: Sequence[ColumnSpec],
        rows: Sequence[Sequence[CellValue]],
    ) -> None:
        """
        A heading row and its rows, aligned to each other.
        """
        if not self.verbose:
            return

        for line in table_lines(columns, rows):
            self.write_line(line)

    def write_checks(self, checks: Sequence[ToleranceCheck]) -> None:
        """
        Every measurement beside the bound it was asserted against.
        """
        columns = (
            ColumnSpec("measurement", align="<"),
            ColumnSpec("worst", ".2e"),
            ColumnSpec("of", ".1e"),
        )
        rows = [(check.label, check.worst, check.tolerance) for check in checks]
        self.write_table(columns, rows)

    def write_verdict(self, passed: bool) -> None:
        """
        The one word the experiment is read for.
        """
        self.write_heading("PASS" if passed else "FAIL")
