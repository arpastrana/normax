# SPDX-License-Identifier: Apache-2.0
"""
The crossed sizing map under axial force and biaxial bending, differentiated.

Sizes a set of members through `TesseractSizer`, the shipped check: a scalar
Blueprints cross-section check hosted behind a Tesseract schema, whose adjoint
is literal NumPy written by hand. Nothing here is a second implementation, so
there is no closed form to lean on in the interaction — the oracles are a
central difference of the crossed forward pass, `jax.test_util.check_grads`
over the same map, and closed-form arithmetic where the moments are silenced.

    reverse       the far side's hand-written pullback, taken by jax.grad
    central diff  a central difference of the crossed bisection
    check_grads   JAX's own comparison of the pullback to its numerics
    closed        the root of the axial-only check, on paper

Also confirms three things the crossed check says about itself: that removing
both end moments reproduces the axial answer exactly, that no buckling length
is read at all, and that a cotangent on the clamp mask is refused rather than
answered with zeros.

The check is Eurocode 3 eq. (6.2) with eq. (6.10) and eq. (6.14), and nothing
else. No §6.3.1 flexural buckling, no §6.3.2 lateral-torsional buckling — a
CHS is doubly symmetric — and no shear or torsion. So a member's size is
decided by the cross-section check or by the catalog minimum, and those two
are the whole governing vocabulary.

Run with `uv run python validation/interaction_gradients.py`.
"""

from collections.abc import Callable
from collections.abc import Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.test_util import check_grads
from jaxtyping import Array
from jaxtyping import Float

from normax.analysis import MemberForces
from normax.materials import Steel355
from normax.reporting import Report
from normax.reporting import ReportColumn
from normax.reporting import ToleranceCheck
from normax.reporting import verify_checks
from normax.sections import TubeCatalog
from normax.sections import build_section_catalog
from normax.sizing import MemberSizes
from normax.sizing.blueprint import DIAMETER_MINIMUM
from normax.sizing.blueprint import GAMMA_M0
from normax.structures import build_arch_2d
from normax.tesseract import TesseractSizer

STEEL = Steel355()

TARGET = 1e-6
TOLERANCE_UNITY = 1e-9

# The bisection against the axial-only root, in millimeters.
TOLERANCE_REDUCTION = 1e-9

# A length dependence is structurally absent here, so nothing is allowed.
TOLERANCE_LENGTH = 1e-12

# The far end's moment as a fraction of the near end's, so one end always wins
# and no pair ties: at a tie two equally valid subgradients are available.
END_FRACTION = -0.4

# How far the length is stretched to show that the check never reads it.
LENGTH_FACTOR = 10.0

# What decided a member's size, in the vocabulary of normax.sizing.blueprint.
GOVERNING_NAMES = {False: "cross-section", True: "minimum size"}

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
        Design moment about the major axis, at the near end.
    moment_minor :
        Design moment about the minor axis, at the near end.
    buckling_length :
        Length the member would be checked against in buckling.
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
        moments = f"{self.moment_major / 1e6:g}/{self.moment_minor / 1e6:g}"

        return f"{self.axial_force / 1e3:g} kN {moments} kNm"


class ClassBranch(NamedTuple):
    """
    A section class, and the crossed sizer whose wall sits at its limit.

    Attributes
    ----------
    section_class :
        Class whose Table 5.2 limit fixes the wall proportion.
    catalog :
        Tube catalog holding the section at that class limit.
    sizer :
        The crossed check, sizing members drawn from that catalog.

    Notes
    -----
    The class fixes the wall and nothing else: eq. (6.10) reads the gross area
    for classes 1 to 3 alike and eq. (6.14) is elastic bending, so the two
    branches differ in geometry rather than in which clause is read.
    """

    section_class: int
    catalog: TubeCatalog
    sizer: TesseractSizer

    @property
    def label(self) -> str:
        """
        The branch as a heading, with the proportion it stands for.
        """
        return (
            f"Class {self.section_class} wall limit, "
            f"d/t = {float(self.catalog.ratio):.2f}"
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
        Derivative from the far side's hand-written pullback.
    numeric :
        Central difference of the crossed forward solve.
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


# The last case is small enough that the catalog minimum decides it, so both
# branches of the hand adjoint are exercised by every table below.
CASES = (
    MemberCase(-5e5, 4e7, 1.5e7, 4000.0),
    MemberCase(-5e5, 4e7, 0.0, 4000.0),
    MemberCase(-9e5, 8e7, 6e7, 12000.0),
    MemberCase(0.0, 4e7, 1.5e7, 4000.0),
    MemberCase(5e5, 4e7, 1.5e7, 4000.0),
    MemberCase(-5e4, 5e6, 5e6, 8000.0),
    MemberCase(-1e2, 1e3, 5e2, 2000.0),
)

STRUCTURE = build_arch_2d(num_edges=len(CASES))

PROBE_COLUMNS = (
    ReportColumn("case", align="<"),
    ReportColumn("argument", align="<"),
    ReportColumn("reverse", "+.12e"),
    ReportColumn("central diff", "+.12e"),
    ReportColumn("rel", ".2e"),
)


def build_class_branch(section_class: int) -> ClassBranch:
    """
    The branch whose wall proportion sits exactly at a class limit.

    Parameters
    ----------
    section_class :
        Class whose Table 5.2 limit fixes the wall proportion.

    Returns
    -------
    branch :
        The catalog at that limit, and the crossed sizer over it.
    """
    catalog = build_section_catalog(STEEL, section_class)
    sizer = TesseractSizer(STRUCTURE, catalog, "blueprint")
    branch = ClassBranch(section_class, catalog, sizer)

    return branch


def build_end_pair(moment: float) -> Float[Array, "ends"]:
    """
    One near-end moment and the smaller far-end moment beside it.
    """
    pair = [moment, END_FRACTION * moment]

    return jnp.stack(pair)


def build_member_forces(cases: Sequence[MemberCase]) -> MemberForces:
    """
    Every case read as one load case of members, moments at both ends.
    """
    forces = [jnp.asarray(case.axial_force) for case in cases]
    ends_major = [build_end_pair(case.moment_major) for case in cases]
    ends_minor = [build_end_pair(case.moment_minor) for case in cases]
    axial = jnp.stack(forces)[None]
    major = jnp.stack(ends_major)[None]
    minor = jnp.stack(ends_minor)[None]
    carried = MemberForces(axial, major, minor)

    return carried


def build_buckling_lengths(
    cases: Sequence[MemberCase],
) -> Float[Array, "members"]:
    """
    The length every case would be checked against in buckling.
    """
    lengths = [jnp.asarray(case.buckling_length) for case in cases]

    return jnp.stack(lengths)


def size_cases(cases: Sequence[MemberCase], branch: ClassBranch) -> MemberSizes:
    """
    Size every case across the boundary, one load case per crossing.
    """
    carried = build_member_forces(cases)
    lengths = build_buckling_lengths(cases)

    return branch.sizer(carried, lengths)


def read_diameter(
    cases: Sequence[MemberCase],
    index: int,
    branch: ClassBranch,
) -> Float[Array, ""]:
    """
    Fully-stressed diameter of one member under the full interaction.
    """
    sizes = size_cases(cases, branch)

    return sizes.sections.diameter[0, index]


def read_clamp_mask(
    cases: Sequence[MemberCase],
    branch: ClassBranch,
) -> Float[np.ndarray, "members"]:
    """
    The schema's non-differentiable clamp mask, straight off the crossing.
    """
    carried = build_member_forces(cases)
    held = jnp.full_like(carried.axial_force, DIAMETER_MINIMUM)
    crossed = branch.sizer.cross_check(carried, held, solve=True)

    return np.asarray(crossed[0]["clamped"])


def central_difference(
    function: Callable[[float], Float[Array, ""]],
    x: float,
    step: float,
) -> float:
    """
    Central difference of a scalar function.
    """
    return (float(function(x + step)) - float(function(x - step))) / (2.0 * step)


def probe_case(index: int, branch: ClassBranch) -> list[ProbeResult]:
    """
    Every action of one case differentiated, and central-differenced beside it.
    """
    case = CASES[index]
    probed = []
    for field, argument in ARGUMENTS:
        value = getattr(case, field)
        if value == 0.0:
            continue

        def sized(x, field=field):
            replaced = {field: x}
            moved = case._replace(**replaced)
            cases = CASES[:index] + (moved,) + CASES[index + 1 :]

            return read_diameter(cases, index, branch)

        reverse = float(jax.grad(sized)(value))
        quotient = central_difference(sized, value, abs(value) * STEP)
        result = ProbeResult(case.label, argument, reverse, quotient)
        probed.append(result)

    return probed


def report_probes(report: Report, branch: ClassBranch) -> float:
    """
    Every action of every case on one class branch, and the worst disagreement.
    """
    indices = range(len(CASES))
    probed = [result for index in indices for result in probe_case(index, branch)]
    rows = [
        (result.label, result.argument, result.reverse, result.numeric, result.relative)
        for result in probed
    ]

    report.write_heading(branch.label)
    report.write_table(PROBE_COLUMNS, rows)

    return max(result.relative for result in probed)


def report_check_grads(report: Report, branch: ClassBranch) -> bool:
    """
    That JAX's own gradcheck accepts the crossed pullback on every member.
    """
    scale_axial = 1.0e5
    scale_moment = 1.0e7
    scale_diameter = 100.0
    lengths = build_buckling_lengths(CASES)
    carried = build_member_forces(CASES)

    def scaled(force, major, minor):
        acting = MemberForces(
            force * scale_axial, major * scale_moment, minor * scale_moment
        )
        sizes = branch.sizer(acting, lengths)

        return sizes.sections.diameter / scale_diameter

    arguments = (
        carried.axial_force / scale_axial,
        carried.moment_major / scale_moment,
        carried.moment_minor / scale_moment,
    )
    try:
        check_grads(scaled, arguments, order=1, modes=("rev",))
    except AssertionError:
        accepted = False
    else:
        accepted = True

    entries = (
        ("arguments", "force, major moment, minor moment"),
        ("order and mode", "1, reverse"),
        ("verdict", "accepted" if accepted else "REJECTED"),
    )

    report.write_heading("check_grads over the whole member set")
    report.write_entries(entries)

    return accepted


def report_length_blindness(report: Report, branch: ClassBranch) -> float:
    """
    That the crossed check reads no buckling length at all.
    """
    carried = build_member_forces(CASES)
    lengths = build_buckling_lengths(CASES)
    near = branch.sizer(carried, lengths)
    far = branch.sizer(carried, lengths * LENGTH_FACTOR)

    def total(stretched):
        sizes = branch.sizer(carried, stretched)

        return jnp.sum(sizes.sections.diameter)

    gradient = jax.grad(total)(lengths)
    moved = jnp.abs(far.sections.diameter - near.sections.diameter)
    gap = float(jnp.max(moved))
    pull = float(jnp.max(jnp.abs(gradient)))

    report.write_heading("The crossed check reads no buckling length")
    report.write_note(
        """
        Blueprints implements no §6.3.1 flexural buckling, so the shipped
        check is a cross-section check and the sizing schema carries no
        length. The contract accepts one, ignores it and never serializes it,
        so the derivative below is structurally absent rather than small.
        """
    )
    entries = (
        (f"largest diameter change at {LENGTH_FACTOR:.0f}x length [mm]", f"{gap:.2e}"),
        ("largest d(diameter)/d(length)", f"{pull:.2e}"),
    )
    report.write_entries(entries)

    return max(gap, pull)


def report_axial_limit(report: Report) -> float:
    """
    That silencing both end moments reproduces the axial answer exactly.
    """
    rows = []
    worst = 0.0
    for section_class in CLASSES:
        branch = build_class_branch(section_class)
        silenced = [case._replace(moment_major=0.0, moment_minor=0.0) for case in CASES]
        sizes = size_cases(silenced, branch)
        area_unit = float(branch.catalog(1.0).area)
        f_y = float(branch.catalog.material.f_y)
        reduced = np.asarray(sizes.sections.diameter)[0]
        for case, crossed in zip(CASES, reduced, strict=True):
            demand = abs(case.axial_force) * GAMMA_M0 / (area_unit * f_y)
            closed = max(np.sqrt(demand), DIAMETER_MINIMUM)
            gap = abs(float(crossed) - float(closed))
            worst = max(worst, gap)
            rows.append(
                (f"Class {section_class}", case.label, float(crossed), closed, gap)
            )

    columns = (
        ReportColumn("branch", align="<"),
        ReportColumn("case", align="<"),
        ReportColumn("moments silenced", ".9f"),
        ReportColumn("closed form", ".9f"),
        ReportColumn("gap", ".2e"),
    )

    report.write_heading("Removing the moments reproduces the axial answer")
    report.write_table(columns, rows)

    return worst


def report_limit_states(report: Report, branch: ClassBranch) -> float:
    """
    Utilization and what decided the size, at the solved diameter.
    """
    sizes = size_cases(CASES, branch)
    clamped = read_clamp_mask(CASES, branch)
    diameters = np.asarray(sizes.sections.diameter)[0]
    used = np.asarray(sizes.utilization)[0]

    rows = []
    worst = 0.0
    walked = zip(CASES, diameters, used, clamped, strict=True)
    for case, diameter, demand, flag in walked:
        decided = bool(flag)
        if not decided:
            worst = max(worst, abs(float(demand) - 1.0))
        rows.append(
            (case.label, float(diameter), float(demand), GOVERNING_NAMES[decided])
        )

    columns = (
        ReportColumn("case", align="<"),
        ReportColumn("d [mm]", ".3f"),
        ReportColumn("utilization", ".15f"),
        ReportColumn("governing", align="<"),
    )

    report.write_heading("Utilization and what decided the size")
    report.write_table(columns, rows)

    return worst


def report_clamp_refusal(report: Report, branch: ClassBranch) -> bool:
    """
    That a cotangent on the clamp mask is refused rather than answered.
    """
    carried = build_member_forces(CASES)
    held = jnp.full_like(carried.axial_force, DIAMETER_MINIMUM)

    def masked(axial_force):
        acting = MemberForces(axial_force, carried.moment_major, carried.moment_minor)
        crossed = branch.sizer.cross_check(acting, held, solve=True)

        return jnp.sum(crossed[0]["clamped"])

    refused = ""
    try:
        jax.grad(masked)(carried.axial_force)
    except ValueError as raised:
        refused = str(raised).split(".")[0]

    entries = (
        ("seeded output", "clamped"),
        ("answer", refused if refused else "ANSWERED, which it must not be"),
    )

    report.write_heading("A cotangent on the clamp mask is refused")
    report.write_entries(entries)

    return bool(refused)


def report_objective(report: Report, branch: ClassBranch) -> None:
    """
    That the mass of several members is differentiable in their axial forces.
    """
    carried = build_member_forces(CASES)
    lengths = build_buckling_lengths(CASES)
    density = float(branch.catalog.material.density)

    def objective(axial_force):
        acting = MemberForces(axial_force, carried.moment_major, carried.moment_minor)
        sizes = branch.sizer(acting, lengths)

        return density * jnp.sum(sizes.sections.area * lengths)

    gradient = jax.grad(objective)(carried.axial_force)
    total = float(objective(carried.axial_force)) * 1e3
    finite = bool(jnp.all(jnp.isfinite(gradient)))
    entries = (
        ("mass", f"{total:.2f} kg"),
        ("d(mass)/d(force)", f"{np.asarray(gradient)[0]}"),
        ("all finite", f"{finite}"),
    )

    report.write_heading("The mass objective is differentiable end to end")
    report.write_entries(entries)


def main(verbose: bool = True) -> None:
    """
    Gradcheck every action of the crossed check, on both class branches.
    """
    report = Report(verbose)
    report.write_line("The crossed sizing map under axial force and biaxial bending")

    branches = [build_class_branch(section_class) for section_class in CLASSES]
    probed = [report_probes(report, branch) for branch in branches]
    worst_derivative = max(probed)

    thinnest = branches[-1]
    accepted = report_check_grads(report, thinnest)
    worst_length = report_length_blindness(report, thinnest)
    worst_reduction = report_axial_limit(report)
    worst_unity = report_limit_states(report, thinnest)
    refused = report_clamp_refusal(report, thinnest)
    report_objective(report, thinnest)

    checks = (
        ToleranceCheck("derivative disagreement", worst_derivative, TARGET),
        ToleranceCheck("departure from unity", worst_unity, TOLERANCE_UNITY),
        ToleranceCheck(
            "axial reduction gap [mm]", worst_reduction, TOLERANCE_REDUCTION
        ),
        ToleranceCheck("length dependence", worst_length, TOLERANCE_LENGTH),
    )
    report.write_heading("Summary")
    report.write_checks(checks)
    report.write_verdict(verify_checks(checks) and accepted and refused)


if __name__ == "__main__":
    main()
