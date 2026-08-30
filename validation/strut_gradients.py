# SPDX-License-Identifier: Apache-2.0
"""
Four independent derivatives of one strut, tabulated against each other.

The milestone that de-risks everything downstream. A single circular hollow
section is sized to carry an axial force beside an end moment, and the
sensitivity of its diameter to both is computed four ways that share almost no
code:

    adjoint     `jax.grad` through the sizing Tesseract, which takes the far
                side's hand-written piecewise adjoint in one crossing
    host        that same rule called in process, plain NumPy, no boundary
    implicit    the implicit function theorem applied in this script, to a
                residual of its own and at a root it finds by Cardano
    numeric     a central difference of the crossed forward pass

Everything is read off the shipped stack: `TesseractSizer` on the `blueprint`
backend, whose check is Eurocode 3 eq. (6.10) and (6.14) summed linearly at
cross-section level. That check reads no buckling length and implements no
§6.3.1, so its residual is `a/d^2 + b/d^3 - 1` rather than the buckling one,
and the `chi <= 1` cap is not a branch this stack has. The two branches it does
have are both exercised here: `sign(N_Ed)`, tension against compression, and
the catalog floor, where the diameter stops moving and the utilization starts.

The sizing Tesseract serves `vector_jacobian_product` and no
`jacobian_vector_product`, so there is no forward-mode leg to take across the
boundary: the adjoint is the only derivative that crosses, and the other three
routes are what hold it honest.

Run with `uv run python validation/strut_gradients.py`.
"""

import math
from collections.abc import Callable
from collections.abc import Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from jaxtyping import Float

from normax.analysis import MemberForces
from normax.materials import Steel355
from normax.reporting import Report
from normax.reporting import ReportColumn
from normax.reporting import ToleranceCheck
from normax.reporting import verify_checks
from normax.sections import build_section_catalog
from normax.sizing.blueprint import DIAMETER_MINIMUM
from normax.sizing.blueprint import GAMMA_M0
from normax.sizing.blueprint import ActionCotangents
from normax.sizing.blueprint import MemberActions
from normax.sizing.blueprint import SizeCotangents
from normax.sizing.blueprint import coerce_member_actions
from normax.sizing.blueprint import coerce_section_coefficients
from normax.sizing.blueprint import size_cotangents
from normax.structures import Structure
from normax.tesseract import TesseractSizer

TITLE = "Four independent derivatives of one strut, tabulated against each other."

SECTION_CLASS = 3
CATALOG = build_section_catalog(Steel355(), SECTION_CLASS)
RATIO = float(CATALOG.ratio)
YIELD_STRENGTH = float(CATALOG.material.f_y)
HOST = coerce_section_coefficients(RATIO, YIELD_STRENGTH)

# The CHS properties of the project notes, written out rather than read off HOST.
AREA_COEFFICIENT = math.pi * (RATIO - 1.0) / RATIO**2
MODULUS_COEFFICIENT = 2.0 * (math.pi / 64.0) * (1.0 - (1.0 - 2.0 / RATIO) ** 4)

TARGET = 1e-8
TOLERANCE_UNITY = 1e-9

# The three analytic routes share no arithmetic, so only rounding separates them.
TOLERANCE_ANALYTIC = 1e-14

# Relative step the central differences are taken at.
STEP = 1e-6

# Accepted, ignored and never serialized: the shipped check reads no length.
BUCKLING_LENGTH = jnp.asarray([4000.0])


class StrutCase(NamedTuple):
    """
    One strut: an axial force, and the moment at one of its ends.

    Attributes
    ----------
    axial_force :
        Design axial force, negative in compression.
    end_moment :
        Major-axis moment at the first end, the other end unbent.
    """

    axial_force: float
    end_moment: float

    @property
    def label(self) -> str:
        """
        The case as it appears in the leftmost column of a table.
        """
        force = self.axial_force / 1e3
        bent = self.end_moment / 1e6

        return f"{force:.0f} kN, {bent:.2f} kNm"


class DerivativeSet(NamedTuple):
    """
    One derivative, obtained four ways that share almost no code.

    Attributes
    ----------
    adjoint :
        The far side's hand adjoint, reached with `jax.grad`.
    host :
        The same rule called in process, with no boundary between.
    implicit :
        The implicit function theorem applied in this script.
    numeric :
        Central difference of the crossed forward pass.
    """

    adjoint: float
    host: float
    implicit: float
    numeric: float

    @property
    def analytic(self) -> float:
        """
        Largest departure of the two adjoints from the implicit rule.
        """
        against_adjoint = relative_gap(self.adjoint, self.implicit)
        against_host = relative_gap(self.host, self.implicit)

        return max(against_adjoint, against_host)

    @property
    def worst(self) -> float:
        """
        Largest relative departure from the implicit rule.
        """
        against_numeric = relative_gap(self.numeric, self.implicit)

        return max(self.analytic, against_numeric)

    @property
    def verdict(self) -> str:
        """
        Whether the four agree to the target.
        """
        return "ok" if self.worst < TARGET else "FAIL"


COMPRESSION = (
    StrutCase(-1e4, 0.0),
    StrutCase(-1e5, 0.0),
    StrutCase(-5e5, 0.0),
    StrutCase(-5e5, 1.0e7),
    StrutCase(-2e6, 5.0e7),
    StrutCase(-1e7, 0.0),
)

TENSION = (
    StrutCase(1e4, 0.0),
    StrutCase(1e5, 0.0),
    StrutCase(5e5, 0.0),
    StrutCase(5e5, 1.0e7),
    StrutCase(5e6, 0.0),
)

BENDING = (
    StrutCase(-1e4, 1.0e6),
    StrutCase(-5e5, 1.0e7),
    StrutCase(-2e6, 5.0e7),
    StrutCase(1e5, 5.0e6),
)

FLOORED = (
    StrutCase(-2e3, 0.0),
    StrutCase(-5e3, 0.0),
    StrutCase(1e3, 1.0e4),
)

COLUMNS = (
    ReportColumn("case", align="<"),
    ReportColumn("crossed adjoint", "+.12e"),
    ReportColumn("host adjoint", "+.12e"),
    ReportColumn("implicit rule", "+.12e"),
    ReportColumn("central diff", "+.12e"),
    ReportColumn("worst", ".2e"),
    ReportColumn("verdict", align="<"),
)

UTILIZATION_COLUMNS = (
    ReportColumn("case", align="<"),
    ReportColumn("diameter [mm]", ".9f"),
    ReportColumn("utilization", ".16f"),
)

FORCE_TITLE = "Compression, sensitivity of the diameter to the axial force"
BENDING_TITLE = "Bending, sensitivity of the diameter to the end moment"
TENSION_TITLE = "Tension, where the pullback flips sign with sign(N_Ed)"
FLOOR_TITLE = "At the catalog floor, sensitivity of the utilization to the force"


def relative_gap(actual: float, expected: float) -> float:
    """
    Relative difference between two derivatives.
    """
    return abs(actual - expected) / max(abs(expected), 1e-300)


def central_difference(function: Callable[[float], float], x: float, step: float):
    """
    Central difference of a scalar function.
    """
    return (function(x + step) - function(x - step)) / (2.0 * step)


def build_strut_sizer() -> TesseractSizer:
    """
    The shipped check, over a structure of one member and two nodes.

    Returns
    -------
    sizer :
        Blueprints' cross-section check, behind its Tesseract boundary.
    """
    nodes = jnp.zeros((2, 3))
    edges = jnp.asarray([[0, 1]], dtype=jnp.int64)
    supports = jnp.asarray([0], dtype=jnp.int64)
    strut = Structure(nodes, edges, supports)

    return TesseractSizer(strut, CATALOG, "blueprint")


def build_strut_forces(
    axial_force: Float[Array, ""],
    end_moment: Float[Array, ""],
) -> MemberForces:
    """
    One strut's actions, shaped as one load case on one member.

    Parameters
    ----------
    axial_force :
        Design axial force, negative in compression.
    end_moment :
        Major-axis moment at the first end.

    Returns
    -------
    forces :
        What the single member carries, carrying the load case axis the
        crossed check reads.
    """
    axial = jnp.reshape(jnp.asarray(axial_force), (1, 1))
    unbent = jnp.zeros_like(jnp.asarray(end_moment))
    pair = jnp.stack([jnp.asarray(end_moment), unbent])
    major = jnp.reshape(pair, (1, 1, 2))
    minor = jnp.zeros((1, 1, 2))

    return MemberForces(axial, major, minor)


def build_member_actions(case: StrutCase) -> MemberActions:
    """
    The same actions as the host check reads them, contiguous float64.
    """
    axial = np.asarray([[case.axial_force]])
    major = np.asarray([[[case.end_moment, 0.0]]])
    minor = np.zeros((1, 1, 2))

    return coerce_member_actions(axial, major, minor)


def size_crossed(
    sizer: TesseractSizer,
    axial_force: Float[Array, ""],
    end_moment: Float[Array, ""],
) -> Float[Array, ""]:
    """
    The diameter the crossed check demands of one strut.
    """
    forces = build_strut_forces(axial_force, end_moment)
    sized = sizer(forces, BUCKLING_LENGTH)

    return sized.sections.diameter[0, 0]


def check_crossed(
    sizer: TesseractSizer,
    axial_force: Float[Array, ""],
    end_moment: Float[Array, ""],
) -> Float[Array, ""]:
    """
    The utilization the crossed check reads at the size it just chose.
    """
    forces = build_strut_forces(axial_force, end_moment)
    sized = sizer(forces, BUCKLING_LENGTH)

    return sized.utilization[0, 0]


def pull_host(case: StrutCase, on_utilization: bool) -> ActionCotangents:
    """
    The host adjoint's pullback of one seeded output onto the actions.

    Parameters
    ----------
    case :
        The strut whose actions are sized.
    on_utilization :
        Whether the seed sits on the utilization rather than the diameter.

    Returns
    -------
    pulled :
        Cotangent on the axial force and on both end-moment arrays.
    """
    actions = build_member_actions(case)
    live = np.ones((1, 1))
    quiet = np.zeros((1, 1))
    if on_utilization:
        seeded = SizeCotangents(quiet, live)
    else:
        seeded = SizeCotangents(live, quiet)

    return size_cotangents(actions, HOST, seeded)


def compute_residual(
    diameter: Float[Array, ""],
    axial_force: Float[Array, ""],
    end_moment: Float[Array, ""],
) -> Float[Array, ""]:
    """
    The check's residual, written out here rather than imported.

    Parameters
    ----------
    diameter :
        Outer diameter of the trial tube.
    axial_force :
        Design axial force, negative in compression.
    end_moment :
        Major-axis moment at the first end.

    Returns
    -------
    residual :
        Utilization less one, which the sizing map drives to zero.

    Notes
    -----
    Smooth and explicit in all three arguments, so `jax.grad` supplies every
    partial the implicit rule needs and only the inversion is by hand.
    """
    squashing = AREA_COEFFICIENT * diameter**2 * YIELD_STRENGTH / GAMMA_M0
    bending = MODULUS_COEFFICIENT * diameter**3 * YIELD_STRENGTH / GAMMA_M0
    demand = jnp.abs(axial_force) / squashing + jnp.abs(end_moment) / bending

    return demand - 1.0


def read_demands(case: StrutCase) -> tuple[float, float]:
    """
    The two coefficients of the cubic the residual becomes.

    Parameters
    ----------
    case :
        The strut whose actions are reduced.

    Returns
    -------
    demand_axial :
        Coefficient `a` of `d^3 = a d + b`.
    demand_moment :
        Coefficient `b` of `d^3 = a d + b`.

    Notes
    -----
    Multiplying `a/d^2 + b/d^3 = 1` through by `d^3` is what turns the check
    into a depressed cubic with one positive root.
    """
    scale_axial = GAMMA_M0 / (AREA_COEFFICIENT * YIELD_STRENGTH)
    scale_moment = GAMMA_M0 / (MODULUS_COEFFICIENT * YIELD_STRENGTH)
    demand_axial = abs(case.axial_force) * scale_axial
    demand_moment = abs(case.end_moment) * scale_moment

    return demand_axial, demand_moment


def solve_cubic(demand_axial: float, demand_moment: float) -> float:
    """
    The one positive root of `d^3 = a d + b`, by Cardano.

    Parameters
    ----------
    demand_axial :
        Coefficient `a`, non-negative.
    demand_moment :
        Coefficient `b`, non-negative.

    Returns
    -------
    root :
        The diameter the check is exactly satisfied at.

    Notes
    -----
    Descartes' rule gives exactly one positive root for non-negative
    coefficients. Where the discriminant is negative all three roots are real
    and the trigonometric form is the clean one, the largest root being the
    positive one. No bracket and no bisection, so the root shares nothing with
    the host's search but the equation it solves.
    """
    discriminant = (0.5 * demand_moment) ** 2 - (demand_axial / 3.0) ** 3
    if discriminant >= 0.0:
        offset = math.sqrt(discriminant)
        upper = math.cbrt(0.5 * demand_moment + offset)
        lower = math.cbrt(0.5 * demand_moment - offset)

        return upper + lower

    reach = 2.0 * math.sqrt(demand_axial / 3.0)
    turn = 1.5 * demand_moment * math.sqrt(3.0 / demand_axial) / demand_axial

    return reach * math.cos(math.acos(turn) / 3.0)


def solve_implicit(case: StrutCase) -> float:
    """
    The diameter this script's own root find demands of one strut.
    """
    demand_axial, demand_moment = read_demands(case)

    return solve_cubic(demand_axial, demand_moment)


def derive_implicit_force(case: StrutCase) -> float:
    """
    Sensitivity of the diameter to the axial force, by the implicit rule.

    Notes
    -----
    `dD/dN = -(dR/dN)/(dR/dd)` at the root, the two partials from `jax.grad`
    of the residual and the inversion the only hand-derived step.
    """
    root = solve_implicit(case)
    by_size = jax.grad(compute_residual, 0)(root, case.axial_force, case.end_moment)
    by_force = jax.grad(compute_residual, 1)(root, case.axial_force, case.end_moment)

    return -float(by_force) / float(by_size)


def derive_implicit_moment(case: StrutCase) -> float:
    """
    Sensitivity of the diameter to the end moment, by the implicit rule.
    """
    root = solve_implicit(case)
    by_size = jax.grad(compute_residual, 0)(root, case.axial_force, case.end_moment)
    by_bent = jax.grad(compute_residual, 2)(root, case.axial_force, case.end_moment)

    return -float(by_bent) / float(by_size)


def derive_implicit_floored(case: StrutCase) -> float:
    """
    Sensitivity of the utilization to the axial force at the floor.

    Notes
    -----
    Below the catalog minimum the size is frozen, so there is no root to
    linearize at and no inversion: the utilization is explicit in the actions
    and its derivative is the residual's bare partial at the floor.
    """
    held = DIAMETER_MINIMUM
    by_force = jax.grad(compute_residual, 1)(held, case.axial_force, case.end_moment)

    return float(by_force)


def derive_force(sizer: TesseractSizer, case: StrutCase) -> DerivativeSet:
    """
    Sensitivity of the diameter to the axial force, four ways.
    """

    def sized(axial_force):
        return size_crossed(sizer, axial_force, case.end_moment)

    step = abs(case.axial_force) * STEP
    adjoint = float(jax.grad(sized)(case.axial_force))
    pulled = pull_host(case, on_utilization=False)
    host = float(pulled.axial[0, 0])
    implicit = derive_implicit_force(case)
    numeric = central_difference(lambda x: float(sized(x)), case.axial_force, step)

    return DerivativeSet(adjoint, host, implicit, numeric)


def derive_moment(sizer: TesseractSizer, case: StrutCase) -> DerivativeSet:
    """
    Sensitivity of the diameter to the end moment, four ways.
    """

    def sized(end_moment):
        return size_crossed(sizer, case.axial_force, end_moment)

    step = abs(case.end_moment) * STEP
    adjoint = float(jax.grad(sized)(case.end_moment))
    pulled = pull_host(case, on_utilization=False)
    host = float(pulled.end_major[0, 0, 0])
    implicit = derive_implicit_moment(case)
    numeric = central_difference(lambda x: float(sized(x)), case.end_moment, step)

    return DerivativeSet(adjoint, host, implicit, numeric)


def derive_floored(sizer: TesseractSizer, case: StrutCase) -> DerivativeSet:
    """
    Sensitivity of the utilization to the axial force at the floor, four ways.
    """

    def used(axial_force):
        return check_crossed(sizer, axial_force, case.end_moment)

    step = abs(case.axial_force) * STEP
    adjoint = float(jax.grad(used)(case.axial_force))
    pulled = pull_host(case, on_utilization=True)
    host = float(pulled.axial[0, 0])
    implicit = derive_implicit_floored(case)
    numeric = central_difference(lambda x: float(used(x)), case.axial_force, step)

    return DerivativeSet(adjoint, host, implicit, numeric)


def derivative_row(label: str, found: DerivativeSet) -> tuple[str | float, ...]:
    """
    One measured derivative, as the table prints it.
    """
    numbers = (found.adjoint, found.host, found.implicit, found.numeric)
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


def read_analytic(measured: Sequence[tuple[str, DerivativeSet]]) -> float:
    """
    Worst departure between the three analytic routes of one block.
    """
    return max(found.analytic for _, found in measured)


def report_utilization(
    report: Report,
    sizer: TesseractSizer,
    cases: Sequence[StrutCase],
) -> float:
    """
    The invariant the derivative rests on, and the worst departure from it.
    """
    sizes = [
        float(size_crossed(sizer, case.axial_force, case.end_moment)) for case in cases
    ]
    demands = [
        float(check_crossed(sizer, case.axial_force, case.end_moment)) for case in cases
    ]
    triples = zip(cases, sizes, demands)
    rows = [(case.label, size, demand) for case, size, demand in triples]

    report.write_heading("The invariant the derivative rests on")
    report.write_table(UTILIZATION_COLUMNS, rows)

    return max(abs(demand - 1.0) for demand in demands)


def measure_agreement(cases: Sequence[StrutCase], sizer: TesseractSizer) -> float:
    """
    Worst departure of this script's own root find from the crossed solve.
    """
    gaps = []
    for case in cases:
        crossed = float(size_crossed(sizer, case.axial_force, case.end_moment))
        gaps.append(relative_gap(solve_implicit(case), crossed))

    return max(gaps)


def measure_frozen(cases: Sequence[StrutCase], sizer: TesseractSizer) -> float:
    """
    The largest size sensitivity the floor left alive, which should be none.
    """
    frozen = []
    for case in cases:

        def sized(axial_force, bent=case.end_moment):
            return size_crossed(sizer, axial_force, bent)

        frozen.append(abs(float(jax.grad(sized)(case.axial_force))))

    return max(frozen)


def main(verbose: bool = True) -> None:
    """
    Tabulate the four derivatives over a range of struts.
    """
    report = Report(verbose)
    sizer = build_strut_sizer()

    entries = (
        ("d/t", f"{RATIO:.2f}"),
        ("catalog floor", f"{DIAMETER_MINIMUM:.1f} mm"),
        ("agreement target", f"{TARGET:.0e}"),
    )
    report.write_line(TITLE)
    report.write_heading("S355 hot-finished tube at the Class 3 limit")
    report.write_entries(entries)

    by_force = [(case.label, derive_force(sizer, case)) for case in COMPRESSION]
    by_moment = [(case.label, derive_moment(sizer, case)) for case in BENDING]
    in_tension = [(case.label, derive_force(sizer, case)) for case in TENSION]
    at_floor = [(case.label, derive_floored(sizer, case)) for case in FLOORED]

    worst_force = report_derivatives(report, FORCE_TITLE, by_force)
    worst_moment = report_derivatives(report, BENDING_TITLE, by_moment)
    worst_tension = report_derivatives(report, TENSION_TITLE, in_tension)
    worst_floor = report_derivatives(report, FLOOR_TITLE, at_floor)
    worst_derivative = max(worst_force, worst_moment, worst_tension, worst_floor)
    blocks = (by_force, by_moment, in_tension, at_floor)
    worst_analytic = max(read_analytic(block) for block in blocks)

    worst_unity = report_utilization(report, sizer, COMPRESSION)
    worst_root = measure_agreement(COMPRESSION + TENSION + BENDING, sizer)
    worst_frozen = measure_frozen(FLOORED, sizer)

    checks = (
        ToleranceCheck("derivative disagreement", worst_derivative, TARGET),
        ToleranceCheck("analytic disagreement", worst_analytic, TOLERANCE_ANALYTIC),
        ToleranceCheck("departure from unity", worst_unity, TOLERANCE_UNITY),
        ToleranceCheck("root find disagreement", worst_root, TARGET),
        ToleranceCheck("size still moving at the floor", worst_frozen, TARGET),
    )
    report.write_heading("Summary")
    report.write_checks(checks)
    report.write_verdict(verify_checks(checks))


if __name__ == "__main__":
    jnp.set_printoptions(precision=12)
    main()
