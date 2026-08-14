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
The sizing map under axial force and biaxial bending, differentiated four ways.

Extends the single-strut check to the full interaction. There is no closed form
to compare against here, so the oracles are the forward tangent, its reverse
transposition, a central difference, and the gradient of the mass objective the
optimizer will actually descend.

Also confirms that removing the moments reproduces the axial answer exactly, and
reports which limit state decides each member.

Run with `uv run python experiments/02_pipeline_gradcheck.py`.
"""

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float

from normax.ec3.actions import MemberActions
from normax.ec3.classification import is_plastic
from normax.ec3.material import Steel
from normax.ec3.section import TubeCatalogue
from normax.ec3.sizing import LIMIT_CROSS_SECTION
from normax.ec3.sizing import LIMIT_MAJOR
from normax.ec3.sizing import LIMIT_MINIMUM_SIZE
from normax.ec3.sizing import LIMIT_MINOR
from normax.ec3.sizing import LIMIT_TENSION
from normax.ec3.sizing import diameter_required
from normax.ec3.sizing import governing_limit_state
from normax.ec3.sizing import mass_of_tubes
from normax.ec3.sizing import utilization_design
from normax.reporting import Report
from normax.reporting import ReportColumn
from normax.reporting import ToleranceCheck
from normax.reporting import checks_passed

STEEL = Steel()
TARGET = 1e-6
TOLERANCE_UNITY = 1e-9

# Moment factors of a member bent in single curvature, as Table B.3 reads them.
MOMENT_FACTOR = 0.9

LIMIT_NAMES = {
    LIMIT_MINIMUM_SIZE: "minimum size",
    LIMIT_TENSION: "tension",
    LIMIT_CROSS_SECTION: "cross-section",
    LIMIT_MAJOR: "Eq. 6.61",
    LIMIT_MINOR: "Eq. 6.62",
}

# A relative step for each argument, since one absolute step cannot serve
# newtons and newton-millimeters at once.
STEP = 1e-6

# The actions a case is probed in, and how each reads in a table.
ARGUMENTS = (
    ("axial_force", "force"),
    ("moment_major", "major moment"),
    ("moment_minor", "minor moment"),
    ("buckling_length", "length"),
)

CLASSES = (2, 3)


class MemberCase(NamedTuple):
    """
    One member's design actions, and the length it may buckle over.

    Attributes
    ----------
    axial_force :
        Design axial force, negative in compression.
    moment_major :
        Design moment about the major axis.
    moment_minor :
        Design moment about the minor axis.
    buckling_length :
        Length the member is checked against in buckling.
    """

    axial_force: float
    moment_major: float
    moment_minor: float
    buckling_length: float

    @property
    def label(self) -> str:
        """
        The case as it appears in the leftmost column of a table.
        """
        moments = f"{self.moment_major / 1e6:.0f}/{self.moment_minor / 1e6:.0f}"

        return f"{self.axial_force / 1e3:.0f} kN {moments} kNm"

    @property
    def actions(self) -> MemberActions:
        """
        The same actions as the clause functions take them.
        """
        moments = (self.moment_major, self.moment_minor)
        actions = MemberActions(
            self.axial_force, *moments, MOMENT_FACTOR, MOMENT_FACTOR
        )

        return actions


class ClassBranch(NamedTuple):
    """
    A section class, and the wall proportion that sits at its limit.

    Attributes
    ----------
    section_class :
        Class the resistances are evaluated on.
    catalogue :
        Tube family whose ratio holds the section at that class limit.
    """

    section_class: int
    catalogue: TubeCatalogue

    @classmethod
    def at_limit(cls, section_class: int) -> "ClassBranch":
        """
        The branch whose wall proportion sits exactly at a class limit.
        """
        catalogue = TubeCatalogue.at_class_limit(STEEL, section_class)
        branch = cls(section_class, catalogue)

        return branch

    @property
    def label(self) -> str:
        """
        The branch as a heading, with the proportion it stands for.
        """
        behavior = "plastic" if is_plastic(self.section_class) else "elastic"

        return (
            f"Class {self.section_class} ({behavior}), "
            f"d/t = {float(self.catalogue.ratio):.2f}"
        )


class ProbeResult(NamedTuple):
    """
    One action's derivative, beside a central difference of the same solve.

    Attributes
    ----------
    label :
        The case the derivative was taken at.
    argument :
        Action the derivative is with respect to.
    reverse :
        Derivative from the transposed implicit rule.
    numeric :
        Central difference of the forward solve.
    """

    label: str
    argument: str
    reverse: float
    numeric: float

    @property
    def relative(self) -> float:
        """
        Relative departure of the derivative from the central difference.
        """
        return abs(self.reverse - self.numeric) / max(abs(self.numeric), 1e-300)


CASES = (
    MemberCase(-5e5, 4e7, 1.5e7, 4000.0),
    MemberCase(-5e5, 4e7, 0.0, 4000.0),
    MemberCase(-9e5, 8e7, 6e7, 12000.0),
    MemberCase(0.0, 4e7, 1.5e7, 4000.0),
    MemberCase(5e5, 4e7, 1.5e7, 4000.0),
    MemberCase(-5e4, 5e6, 5e6, 8000.0),
)

PROBE_COLUMNS = (
    ReportColumn("case", align="<"),
    ReportColumn("argument", align="<"),
    ReportColumn("reverse", "+.12e"),
    ReportColumn("central diff", "+.12e"),
    ReportColumn("rel", ".2e"),
)


def diameter_of(case: MemberCase, branch: ClassBranch) -> Float[Array, ""]:
    """
    Fully-stressed diameter under the full interaction.
    """
    diameter = diameter_required(case.actions, case.buckling_length, branch.catalogue)

    return diameter


def central_difference(function: Callable[[float], float], x: float, step: float):
    """
    Central difference of a scalar function.
    """
    return (function(x + step) - function(x - step)) / (2.0 * step)


def probe_case(case: MemberCase, branch: ClassBranch) -> list[ProbeResult]:
    """
    Every action of one case differentiated, and central-differenced beside it.
    """
    probed = []
    for field, argument in ARGUMENTS:
        value = getattr(case, field)
        if value == 0.0:
            continue

        def sized(x, field=field):
            moved = case._replace(**{field: x})

            return diameter_of(moved, branch)

        reverse = float(jax.grad(sized)(value))
        quotient = central_difference(
            lambda x: float(sized(x)), value, abs(value) * STEP
        )
        result = ProbeResult(case.label, argument, reverse, float(quotient))
        probed.append(result)

    return probed


def report_probes(report: Report, branch: ClassBranch) -> float:
    """
    Every action of every case on one class branch, and the worst disagreement.
    """
    probed = [result for case in CASES for result in probe_case(case, branch)]
    rows = [
        (result.label, result.argument, result.reverse, result.numeric, result.relative)
        for result in probed
    ]

    report.write_heading(branch.label)
    report.write_table(PROBE_COLUMNS, rows)

    return max(result.relative for result in probed)


def report_modes(report: Report, branch: ClassBranch) -> None:
    """
    That forward mode and reverse mode return the same number.
    """
    rows = []
    for case in CASES[:3]:

        def sized(axial_force, case=case):
            moved = case._replace(axial_force=axial_force)

            return diameter_of(moved, branch)

        forward = float(jax.jacfwd(sized)(case.axial_force))
        reverse = float(jax.grad(sized)(case.axial_force))
        gap = abs(forward - reverse) / max(abs(reverse), 1e-300)
        rows.append((f"{case.axial_force / 1e3:.0f} kN", gap))

    columns = (
        ReportColumn("case", align="<"),
        ReportColumn("forward-reverse gap", ".2e"),
    )

    report.write_heading("Forward and reverse are the same derivative")
    report.write_table(columns, rows)


def report_axial_limit(report: Report) -> None:
    """
    That removing the moments reproduces the axial answer on either branch.
    """
    rows = []
    for section_class in CLASSES:
        branch = ClassBranch.at_limit(section_class)
        case = MemberCase(-5e5, 0.0, 0.0, 4000.0)
        actions = MemberActions(case.axial_force, 0.0, 0.0, 1.0, 1.0)
        bare = diameter_required(actions, case.buckling_length, branch.catalogue)
        with_moment = float(diameter_of(case, branch))
        axial_only = float(bare)
        gap = abs(with_moment - axial_only)
        rows.append((f"Class {section_class}", with_moment, axial_only, gap))

    columns = (
        ReportColumn("branch", align="<"),
        ReportColumn("interaction", ".12f"),
        ReportColumn("axial only", ".12f"),
        ReportColumn("gap", ".2e"),
    )

    report.write_heading("Removing the moments reproduces the axial answer")
    report.write_table(columns, rows)


def report_limit_states(report: Report, branch: ClassBranch) -> float:
    """
    Utilization and governing limit state at the solved diameter.
    """
    rows = []
    worst = 0.0
    for case in CASES:
        diameter = diameter_of(case, branch)
        tube = branch.catalogue(diameter)
        utilization = utilization_design(tube, case.actions, case.buckling_length)
        limit_state = governing_limit_state(
            tube, case.actions, case.buckling_length, branch.catalogue
        )
        demand = float(utilization)
        worst = max(worst, abs(demand - 1.0))
        rows.append(
            (case.label, float(diameter), demand, LIMIT_NAMES[float(limit_state)])
        )

    columns = (
        ReportColumn("case", align="<"),
        ReportColumn("d [mm]", ".3f"),
        ReportColumn("utilization", ".15f"),
        ReportColumn("governing", align="<"),
    )

    report.write_heading("Utilization and governing limit state at the solved diameter")
    report.write_table(columns, rows)

    return worst


def report_objective(report: Report, branch: ClassBranch) -> None:
    """
    That the mass of several members is differentiable in their axial forces.
    """
    forces = jnp.asarray([-5e5, -9e5, 5e5, -5e4])
    lengths = jnp.asarray([4000.0, 12000.0, 4000.0, 8000.0])

    def objective(axial_force):
        actions = MemberActions(axial_force, 4e7, 1.5e7, MOMENT_FACTOR, MOMENT_FACTOR)
        sizes = diameter_required(actions, lengths, branch.catalogue)
        tubes = branch.catalogue(sizes)

        return mass_of_tubes(tubes, lengths)

    gradient = jax.grad(objective)(forces)
    total = float(objective(forces)) * 1e3
    finite = bool(jnp.all(jnp.isfinite(gradient)))
    entries = (
        ("mass", f"{total:.2f} kg"),
        ("d(mass)/d(force)", f"{gradient}"),
        ("all finite", f"{finite}"),
    )

    report.write_heading("The mass objective is differentiable end to end")
    report.write_entries(entries)


def main(verbose: bool = True) -> None:
    """
    Gradcheck every action, on both class branches.
    """
    report = Report(verbose)
    report.write_line("The sizing map under axial force and biaxial bending")

    branches = [ClassBranch.at_limit(section_class) for section_class in CLASSES]
    probed = [report_probes(report, branch) for branch in branches]
    worst_derivative = max(probed)

    elastic = ClassBranch.at_limit(3)
    report_modes(report, elastic)
    report_axial_limit(report)
    worst_unity = report_limit_states(report, elastic)
    report_objective(report, elastic)

    checks = (
        ToleranceCheck("derivative disagreement", worst_derivative, TARGET),
        ToleranceCheck("departure from unity", worst_unity, TOLERANCE_UNITY),
    )
    report.write_heading("Summary")
    report.write_checks(checks)
    report.write_verdict(checks_passed(checks))


if __name__ == "__main__":
    main()
