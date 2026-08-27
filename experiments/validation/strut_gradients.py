# SPDX-License-Identifier: Apache-2.0
"""
Four independent derivatives of one strut, tabulated against each other.

The milestone that de-risks everything downstream. A single circular hollow
section is sized to carry an axial force over a buckling length, and the
sensitivity of its diameter to both is computed four ways that share almost no
code:

    forward     the implicit tangent rule of ec3x.sizing
    reverse     that same rule, transposed by JAX into an adjoint
    closed      ec3x.adjoint, derived on paper and written out in full
    numeric     a central difference of the forward solve

Run with `uv run python experiments/01_single_strut_gradcheck.py`.
"""

from collections.abc import Callable
from collections.abc import Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
from ec3x.actions import MemberActions
from ec3x.adjoint import derivative_force
from ec3x.adjoint import derivative_force_tension
from ec3x.adjoint import derivative_length
from ec3x.material import Steel
from ec3x.section import TubeCatalogue
from ec3x.sizing import diameter_required
from ec3x.sizing import utilization_design
from jaxtyping import Array
from jaxtyping import Float

from normax.reporting import Report
from normax.reporting import ReportColumn
from normax.reporting import ToleranceCheck
from normax.reporting import verify_checks

TITLE = "Four independent derivatives of one strut, tabulated against each other."

STEEL = Steel()
CATALOGUE = TubeCatalogue.at_class_limit(STEEL, 3)
SECTION_CLASS = 3

TARGET = 1e-8
TOLERANCE_UNITY = 1e-9

# Relative step the central differences are taken at.
STEP = 1e-6


class StrutCase(NamedTuple):
    """
    One strut: an axial force, and the length it may buckle over.

    Attributes
    ----------
    axial_force :
        Design axial force, negative in compression.
    buckling_length :
        Length the member is checked against in buckling.
    """

    axial_force: float
    buckling_length: float

    @property
    def label(self) -> str:
        """
        The case as it appears in the leftmost column of a table.
        """
        force = self.axial_force / 1e3
        span = self.buckling_length / 1e3

        return f"{force:.0f} kN, {span:.0f} m"


class DerivativeSet(NamedTuple):
    """
    One derivative, obtained four ways that share almost no code.

    Attributes
    ----------
    forward :
        Implicit tangent rule, in forward mode.
    reverse :
        The same rule, transposed by JAX into an adjoint.
    closed :
        Hand-derived closed form of `ec3x.adjoint`.
    numeric :
        Central difference of the forward solve.
    """

    forward: float
    reverse: float
    closed: float
    numeric: float

    @property
    def worst(self) -> float:
        """
        Largest relative departure from the closed form.
        """
        against_forward = relative_gap(self.forward, self.closed)
        against_reverse = relative_gap(self.reverse, self.closed)
        against_numeric = relative_gap(self.numeric, self.closed)

        return max(against_forward, against_reverse, against_numeric)

    @property
    def verdict(self) -> str:
        """
        Whether the four agree to the target.
        """
        return "ok" if self.worst < TARGET else "FAIL"


COMPRESSION = (
    StrutCase(-1e4, 4000.0),
    StrutCase(-1e5, 4000.0),
    StrutCase(-5e5, 4000.0),
    StrutCase(-5e5, 12000.0),
    StrutCase(-2e6, 8000.0),
    StrutCase(-1e7, 6000.0),
)

TENSION = (
    StrutCase(1e4, 4000.0),
    StrutCase(1e5, 4000.0),
    StrutCase(5e5, 4000.0),
    StrutCase(5e6, 4000.0),
)

COLUMNS = (
    ReportColumn("case", align="<"),
    ReportColumn("forward", "+.12e"),
    ReportColumn("reverse", "+.12e"),
    ReportColumn("closed form", "+.12e"),
    ReportColumn("central diff", "+.12e"),
    ReportColumn("worst", ".2e"),
    ReportColumn("verdict", align="<"),
)

UTILIZATION_COLUMNS = (
    ReportColumn("case", align="<"),
    ReportColumn("utilization", ".16f"),
)

FORCE_TITLE = "Compression, sensitivity of the diameter to the axial force"
LENGTH_TITLE = "Compression, sensitivity of the diameter to the buckling length"
TENSION_TITLE = "Tension, where the answer is closed form and buckling never enters"


def relative_gap(actual: float, expected: float) -> float:
    """
    Relative difference between two derivatives.
    """
    return abs(actual - expected) / max(abs(expected), 1e-300)


def diameter_of(case: StrutCase) -> Float[Array, ""]:
    """
    Fully-stressed diameter under axial force alone.
    """
    actions = MemberActions(case.axial_force, 0.0, 0.0, 1.0, 1.0)
    diameter = diameter_required(actions, case.buckling_length, CATALOGUE)

    return diameter


def utilization_at(diameter: Float[Array, ""], case: StrutCase) -> Float[Array, ""]:
    """
    Utilization at a diameter, from the exact clause functions.
    """
    tube = CATALOGUE(diameter)
    actions = MemberActions(case.axial_force, 0.0, 0.0, 1.0, 1.0)
    demand = utilization_design(tube, actions, case.buckling_length)

    return demand


def central_difference(function: Callable[[float], float], x: float, step: float):
    """
    Central difference of a scalar function.
    """
    return (function(x + step) - function(x - step)) / (2.0 * step)


def derivatives_force(case: StrutCase) -> DerivativeSet:
    """
    Sensitivity of the diameter to the axial force, four ways.
    """

    def sized(axial_force):
        probed = case._replace(axial_force=axial_force)

        return diameter_of(probed)

    solved = diameter_of(case)
    step = abs(case.axial_force) * STEP
    exact = derivative_force(solved, case.axial_force, case.buckling_length, CATALOGUE)
    quotient = central_difference(lambda x: float(sized(x)), case.axial_force, step)

    forward = float(jax.jacfwd(sized)(case.axial_force))
    reverse = float(jax.grad(sized)(case.axial_force))
    found = DerivativeSet(forward, reverse, float(exact), float(quotient))

    return found


def derivatives_length(case: StrutCase) -> DerivativeSet:
    """
    Sensitivity of the diameter to the buckling length, four ways.
    """

    def sized(buckling_length):
        probed = case._replace(buckling_length=buckling_length)

        return diameter_of(probed)

    solved = diameter_of(case)
    step = case.buckling_length * STEP
    exact = derivative_length(solved, case.axial_force, case.buckling_length, CATALOGUE)
    quotient = central_difference(lambda x: float(sized(x)), case.buckling_length, step)

    forward = float(jax.jacfwd(sized)(case.buckling_length))
    reverse = float(jax.grad(sized)(case.buckling_length))
    found = DerivativeSet(forward, reverse, float(exact), float(quotient))

    return found


def derivatives_tension(case: StrutCase) -> DerivativeSet:
    """
    The same sensitivity where the answer is closed form and buckling is absent.
    """

    def sized(axial_force):
        probed = case._replace(axial_force=axial_force)

        return diameter_of(probed)

    step = abs(case.axial_force) * STEP
    exact = derivative_force_tension(case.axial_force, CATALOGUE)
    quotient = central_difference(lambda x: float(sized(x)), case.axial_force, step)

    forward = float(jax.jacfwd(sized)(case.axial_force))
    reverse = float(jax.grad(sized)(case.axial_force))
    found = DerivativeSet(forward, reverse, float(exact), float(quotient))

    return found


def derivative_row(label: str, found: DerivativeSet) -> tuple[str | float, ...]:
    """
    One measured derivative, as the table prints it.
    """
    numbers = (found.forward, found.reverse, found.closed, found.numeric)
    row = (label, *numbers, found.worst, found.verdict)

    return row


def report_derivatives(
    report: Report,
    title: str,
    measured: Sequence[tuple[str, DerivativeSet]],
) -> float:
    """
    One block of the comparison table, and the worst disagreement in it.
    """
    rows = [derivative_row(label, found) for label, found in measured]

    report.write_heading(title)
    report.write_table(COLUMNS, rows)

    return max(found.worst for _, found in measured)


def report_utilization(report: Report, cases: Sequence[StrutCase]) -> float:
    """
    The invariant the derivative rests on, and the worst departure from it.
    """
    demands = [float(utilization_at(diameter_of(case), case)) for case in cases]
    rows = [(case.label, demand) for case, demand in zip(cases, demands)]

    report.write_heading("The invariant the derivative rests on")
    report.write_table(UTILIZATION_COLUMNS, rows)

    return max(abs(demand - 1.0) for demand in demands)


def main(verbose: bool = True) -> None:
    """
    Tabulate the four derivatives over a range of struts.
    """
    report = Report(verbose)

    entries = (
        ("d/t", f"{float(CATALOGUE.ratio):.2f}"),
        ("agreement target", f"{TARGET:.0e}"),
    )
    report.write_line(TITLE)
    report.write_heading("S355 hot-finished tube at the Class 3 limit")
    report.write_entries(entries)

    by_force = [(case.label, derivatives_force(case)) for case in COMPRESSION]
    by_length = [(case.label, derivatives_length(case)) for case in COMPRESSION]
    in_tension = [
        (f"{case.axial_force / 1e3:.0f} kN", derivatives_tension(case))
        for case in TENSION
    ]

    worst_force = report_derivatives(report, FORCE_TITLE, by_force)
    worst_length = report_derivatives(report, LENGTH_TITLE, by_length)
    worst_tension = report_derivatives(report, TENSION_TITLE, in_tension)
    worst_derivative = max(worst_force, worst_length, worst_tension)

    worst_unity = report_utilization(report, COMPRESSION)

    checks = (
        ToleranceCheck("derivative disagreement", worst_derivative, TARGET),
        ToleranceCheck("departure from unity", worst_unity, TOLERANCE_UNITY),
    )
    report.write_heading("Summary")
    report.write_checks(checks)
    report.write_verdict(verify_checks(checks))


if __name__ == "__main__":
    jnp.set_printoptions(precision=12)
    main()
