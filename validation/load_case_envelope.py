# SPDX-License-Identifier: Apache-2.0
"""
Aggregating several load cases into one differentiable size per member.

A member must satisfy every load case, so its size is the largest any case
demands. That largest is not differentiable, and a gradient taken through it
sees one case per step and stalls. `design_envelope` replaces it with a smooth
maximum, taken in the logarithm of the diameter so the sharpness is
dimensionless.

The envelope is normax's own and no clause's. EN 1993-1-1 checks one member
under one combination and has no opinion about how several combinations are
reconciled into one section, so this is the piece of the pipeline the standard
does not govern and the only place its cost can be priced. The sizes it
reconciles come from the check that ships — Blueprints across a Tesseract
boundary, `TesseractSizer(structure, catalog, "blueprint")` — so what is
measured here is what runs.

The envelope never understates the true largest, so annealing the sharpness
upward drives it onto that largest from above and the design stays adequate
throughout. This reports how much is given away at each sharpness, and that
every case keeps a gradient a hard maximum would have withheld.

Front and centre is the member whose axial force reverses between load cases.
The crossed check reads `|N_Ed|`, so the diameter it asks for is even in the
axial force while its derivative is odd — the `sign(N_Ed)` branch of the
hand-written adjoint, one of the two branches the sizing map genuinely has.
Both halves are reported: the size is unmoved by a flipped force, the adjoint
flips with it, and the adjoint is verified against a central difference of the
crossed forward pass.

Run with `uv run python validation/load_case_envelope.py`.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from jaxtyping import Float

from normax.analysis import MemberForces
from normax.design import Design
from normax.design import compute_mass
from normax.materials import Steel355
from normax.optimization.nested import design_envelope
from normax.reporting import Report
from normax.reporting import ReportColumn
from normax.reporting import ToleranceCheck
from normax.reporting import verify_checks
from normax.sections import build_section_catalog
from normax.sizing import MemberSizes
from normax.structures import DesignShape
from normax.structures import Structure
from normax.structures import build_structure
from normax.structures import compute_member_lengths
from normax.tesseract import TesseractSizer

SECTION_CLASS = 3

LENGTHS = jnp.asarray([4000.0, 6000.0, 5000.0, 4500.0])

# Three load cases over four members: symmetric, half-span asymmetric, and a
# crown point load. The third member reverses from compression to tension.
FORCES_BY_CASE = [
    [-6e5, -4e5, -3e5, -5e5],
    [-8e5, -2e5, 2e5, -6e5],
    [-3e5, -7e5, -1e5, -9e5],
]
MOMENTS_BY_CASE = [
    [2e7, 1e7, 5e6, 1.5e7],
    [5e7, 3e7, 2e7, 4e7],
    [1e7, 6e7, 1e7, 2e7],
]

FORCES = jnp.asarray(FORCES_BY_CASE)
MOMENTS = jnp.asarray(MOMENTS_BY_CASE)

SHARPNESSES = (5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0)

# The sharpness the live-case comparison is made at.
SHARPNESS = 50.0

# The member whose axial force reverses between load cases.
REVERSING = 2

NUM_CASES, NUM_MEMBERS = FORCES.shape

# Kilograms in a tonne, the unit a mass comes back in.
KILOGRAMS = 1e3

# Kilograms per kilonewton in a tonne per newton.
GRADIENT_SCALE = 1e6

# Relative step the central differences are taken at.
STEP = 1e-5

TOLERANCE_UNITY = 1e-9
TOLERANCE_EVEN = 1e-12
TOLERANCE_DERIVATIVE = 1e-6

# A sign mismatch is an integer count, so anything under a half is none.
TOLERANCE_SIGN = 0.5


def build_member_chain(lengths: Float[Array, "members"]) -> Structure:
    """
    A chain of members of the given lengths, strung along the x axis.

    Parameters
    ----------
    lengths :
        Length of every member.

    Returns
    -------
    structure :
        The chain, which the crossed check is built on and reads for nothing.

    Notes
    -----
    A cross-section check needs no connectivity, so the structure is here to
    carry the lengths a mass is read off and nothing else.
    """
    spans = np.asarray(lengths, dtype=np.float64)
    stations = np.concatenate([[0.0], np.cumsum(spans)])
    flat = np.zeros_like(stations)
    nodes = np.stack([stations, flat, flat], axis=1)
    starts = np.arange(spans.shape[0])
    edges = np.stack([starts, starts + 1], axis=1)
    supports = np.array([0, spans.shape[0]])

    return build_structure(nodes, edges, supports)


STRUCTURE = build_member_chain(LENGTHS)
MEMBER_LENGTHS = compute_member_lengths(STRUCTURE.nodes, STRUCTURE.edges)
SHAPE = DesignShape(STRUCTURE.nodes, MEMBER_LENGTHS)

CATALOG = build_section_catalog(Steel355(), SECTION_CLASS)
SIZER = TesseractSizer(STRUCTURE, CATALOG, "blueprint")

# Single curvature, so the check reads the same magnitude at either end.
END_MOMENTS_MAJOR = jnp.stack([MOMENTS, -MOMENTS], axis=-1)
END_MOMENTS_MINOR = jnp.zeros_like(END_MOMENTS_MAJOR)


class AnnealStep(NamedTuple):
    """
    What the smooth envelope gives away at one sharpness.

    Attributes
    ----------
    beta :
        Sharpness of the envelope.
    mass :
        Mass of the smoothly sized structure, in kilograms.
    excess :
        Fraction by which that exceeds the mass of the exact largest.
    bound :
        Fraction the envelope's own bound allows it to exceed by.
    finite :
        Whether the gradient of the mass is finite throughout.
    """

    beta: float
    mass: float
    excess: float
    bound: float
    finite: bool


class LiveCases(NamedTuple):
    """
    How many load cases reach one member's size with a gradient.

    Attributes
    ----------
    member :
        Index of the member.
    smooth :
        Cases with a non-zero gradient under the smooth envelope.
    hard :
        Cases with a non-zero gradient under a hard maximum.
    """

    member: int
    smooth: int
    hard: int


class ReversalRow(NamedTuple):
    """
    One load case as the sign-changing member meets it.

    Attributes
    ----------
    load_case :
        Number of the load case, counted from one.
    axial :
        Axial force the member carries, in kilonewtons.
    diameter :
        Diameter the crossed check asks of it in that case, in millimeters.
    utilization :
        Demand over resistance at that diameter.
    slope :
        Derivative of that diameter in that case's own axial force, in
        millimeters per kilonewton, which is the sign branch unweighted.
    adjoint :
        Derivative of the enveloped mass in that force, from the crossed
        adjoint, in kilograms per kilonewton.
    central :
        The same derivative by a central difference of the forward pass.
    """

    load_case: int
    axial: float
    diameter: float
    utilization: float
    slope: float
    adjoint: float
    central: float


def read_member_forces(
    axial: Float[Array, "load_cases members"],
) -> MemberForces:
    """
    The actions the check reads: these axial forces and the stated moments.

    Parameters
    ----------
    axial :
        Axial force of every member under every load case, tension positive.

    Returns
    -------
    forces :
        The container an analysis would have handed the check.
    """
    forces = MemberForces(axial, END_MOMENTS_MAJOR, END_MOMENTS_MINOR)

    return forces


def size_load_cases(axial: Float[Array, "load_cases members"]) -> MemberSizes:
    """
    What the crossed check asks of every member under every load case.

    Parameters
    ----------
    axial :
        Axial force of every member under every load case.

    Returns
    -------
    sizes :
        One fully-stressed section per load case and member, and how hard each
        is worked there.
    """
    forces = read_member_forces(axial)

    return SIZER(forces, MEMBER_LENGTHS)


def build_case_design(axial: Float[Array, "load_cases members"]) -> Design:
    """
    The design the envelope is asked to reconcile.

    Parameters
    ----------
    axial :
        Axial force of every member under every load case.

    Returns
    -------
    design :
        The chain, its actions, and one section per load case and member.
    """
    forces = read_member_forces(axial)
    sizes = SIZER(forces, MEMBER_LENGTHS)
    design = Design(SHAPE, forces, sizes)

    return design


def weigh_envelope(
    axial: Float[Array, "load_cases members"],
    sharpness: float | None,
) -> Float[Array, ""]:
    """
    Mass of the design once the load cases are reconciled, in tonnes.

    Parameters
    ----------
    axial :
        Axial force of every member under every load case.
    sharpness :
        Sharpness of the envelope, or None for the true largest.

    Returns
    -------
    mass :
        Total mass of the enveloped members.
    """
    design = build_case_design(axial)
    covered = design_envelope(design, sharpness)

    return compute_mass(covered)


def anneal_step(exact_mass: float, beta: float) -> AnnealStep:
    """
    The smoothed mass at one sharpness, against the exact one and its bound.
    """
    smoothed = float(weigh_envelope(FORCES, beta)) * KILOGRAMS
    gradient = jax.grad(weigh_envelope)(FORCES, beta)
    excess = (smoothed - exact_mass) / exact_mass
    bound = float(jnp.log(NUM_CASES) / beta)
    finite = bool(jnp.all(jnp.isfinite(gradient)))
    step = AnnealStep(beta, smoothed, excess, bound, finite)

    return step


def live_cases(
    member: int,
    smooth: Float[Array, "load_cases members"],
    hard: Float[Array, "load_cases members"],
) -> LiveCases:
    """
    Cases that reach one member's size with a gradient, under either aggregation.
    """
    live_smooth = int(jnp.sum(jnp.abs(smooth[:, member]) > 0.0))
    live_hard = int(jnp.sum(jnp.abs(hard[:, member]) > 0.0))
    counted = LiveCases(member, live_smooth, live_hard)

    return counted


def difference_mass(load_case: int, member: int) -> float:
    """
    Central difference of the enveloped mass in one case's axial force.

    Parameters
    ----------
    load_case :
        Which load case the force is perturbed in.
    member :
        Which member's force is perturbed.

    Returns
    -------
    slope :
        Derivative in kilograms per kilonewton.
    """
    force = float(FORCES[load_case, member])
    step = STEP * abs(force)
    raised = FORCES.at[load_case, member].add(step)
    lowered = FORCES.at[load_case, member].add(-step)
    up = float(weigh_envelope(raised, SHARPNESS))
    down = float(weigh_envelope(lowered, SHARPNESS))

    return GRADIENT_SCALE * (up - down) / (2.0 * step)


def read_case_diameter(
    axial: Float[Array, "load_cases members"],
    load_case: int,
) -> Float[Array, ""]:
    """
    The diameter the crossed check asks of the reversing member in one case.

    Parameters
    ----------
    axial :
        Axial force of every member under every load case.
    load_case :
        Which load case the diameter is read in.

    Returns
    -------
    diameter :
        Outer diameter that case demands of the reversing member.
    """
    sizes = size_load_cases(axial)

    return sizes.sections.diameter[load_case, REVERSING]


def slope_diameter(load_case: int) -> float:
    """
    Derivative of that diameter in its own axial force, before any envelope.

    Parameters
    ----------
    load_case :
        Which load case the derivative is taken in.

    Returns
    -------
    slope :
        Derivative in millimeters per kilonewton, carrying `sign(N_Ed)`.
    """
    pulled = jax.grad(read_case_diameter)(FORCES, load_case)

    return float(pulled[load_case, REVERSING]) * 1e3


def flip_reversing() -> Float[Array, "load_cases"]:
    """
    Diameter the check asks of the reversing member with its force negated.

    Returns
    -------
    diameters :
        One diameter per load case, which an even check leaves where they were.
    """
    flipped = FORCES.at[:, REVERSING].multiply(-1.0)
    sizes = size_load_cases(flipped)

    return sizes.sections.diameter[:, REVERSING]


def read_reversal(
    load_case: int,
    sizes: MemberSizes,
    adjoint: Float[Array, "load_cases members"],
) -> ReversalRow:
    """
    The reversing member's row for one load case, adjoint beside difference.
    """
    axial = float(FORCES[load_case, REVERSING]) / 1e3
    diameter = float(sizes.sections.diameter[load_case, REVERSING])
    used = float(sizes.utilization[load_case, REVERSING])
    slope = slope_diameter(load_case)
    pulled = float(adjoint[load_case, REVERSING]) * GRADIENT_SCALE
    numeric = difference_mass(load_case, REVERSING)
    row = ReversalRow(load_case + 1, axial, diameter, used, slope, pulled, numeric)

    return row


def report_sizes(report: Report, sizes: MemberSizes) -> float:
    """
    What each load case asks of each member, and the exact largest of them.
    """
    demanded = sizes.sections.diameter
    exact = jnp.max(demanded, axis=0)
    exact_mass = float(weigh_envelope(FORCES, None)) * KILOGRAMS

    per_case_columns = [
        ReportColumn(f"case {case + 1} [mm]", ".2f") for case in range(NUM_CASES)
    ]
    columns = (
        ReportColumn("member"),
        *per_case_columns,
        ReportColumn("exact max [mm]", ".2f"),
    )
    rows = []
    for member in range(NUM_MEMBERS):
        sized = [float(demanded[case, member]) for case in range(NUM_CASES)]
        rows.append((member, *sized, float(exact[member])))

    entries = (("exact mass", f"{exact_mass:.2f} kg"),)

    report.write_line("Three load cases over four members, S355 at the Class 3 limit")
    report.write_table(columns, rows)
    report.write_entries(entries)

    return exact_mass


def report_annealing(report: Report, exact_mass: float) -> None:
    """
    What the smoothing costs at each sharpness, and what bounds it.
    """
    columns = (
        ReportColumn("beta", ".0f"),
        ReportColumn("mass [kg]", ".2f"),
        ReportColumn("excess", ".3%"),
        ReportColumn("bound log(cases)/beta", ".3%"),
        ReportColumn("gradient finite"),
    )
    annealed = [anneal_step(exact_mass, beta) for beta in SHARPNESSES]
    rows = [
        (step.beta, step.mass, step.excess, step.bound, str(step.finite))
        for step in annealed
    ]

    report.write_heading("Annealing the sharpness")
    report.write_table(columns, rows)
    report.write_note(
        """
        The excess is an overestimate of the size, never an underestimate, so
        every intermediate design satisfies the standard.
        """
    )


def report_live_cases(report: Report) -> None:
    """
    That every case sees a gradient, which a hard maximum would not give.
    """
    smooth = jax.grad(weigh_envelope)(FORCES, SHARPNESS)
    hard = jax.grad(weigh_envelope)(FORCES, None)

    columns = (
        ReportColumn("member"),
        ReportColumn("smooth, cases with a gradient"),
        ReportColumn("hard maximum"),
    )
    counted = [live_cases(member, smooth, hard) for member in range(NUM_MEMBERS)]
    rows = [(found.member, found.smooth, found.hard) for found in counted]

    report.write_heading("Every case sees a gradient, which a hard maximum would not")
    report.write_table(columns, rows)


def report_reversal(report: Report, sizes: MemberSizes) -> list[ToleranceCheck]:
    """
    The sign branch: an even size, an odd adjoint, and both measured.
    """
    adjoint = jax.grad(weigh_envelope)(FORCES, SHARPNESS)
    reversal = [
        read_reversal(load_case, sizes, adjoint) for load_case in range(NUM_CASES)
    ]

    columns = (
        ReportColumn("case"),
        ReportColumn("axial [kN]", ".0f"),
        ReportColumn("diameter [mm]", ".2f"),
        ReportColumn("utilization", ".9f"),
        ReportColumn("dd/dN [mm/kN]", ".5f"),
        ReportColumn("adjoint [kg/kN]", ".4e"),
        ReportColumn("central [kg/kN]", ".4e"),
    )
    rows = [tuple(row) for row in reversal]

    report.write_heading(f"Member {REVERSING}, whose axial force changes sign")
    report.write_table(columns, rows)

    standing = sizes.sections.diameter[:, REVERSING]
    negated = flip_reversing()
    even_gap = float(jnp.max(jnp.abs(negated - standing) / standing))
    branched = jnp.asarray([row.slope for row in reversal])
    signs = jnp.sign(branched) - jnp.sign(FORCES[:, REVERSING])
    mismatched = float(jnp.sum(jnp.abs(signs) > 0.0))
    slopes = [(row.adjoint, row.central) for row in reversal]
    scale = max(abs(numeric) for _, numeric in slopes)
    gaps = [abs(pulled - numeric) / scale for pulled, numeric in slopes]
    off_unity = float(jnp.max(jnp.abs(sizes.utilization - 1.0)))

    report.write_note(
        """
        The check reads the magnitude of the axial force, so the size a
        reversal asks for is unmoved by it and the derivative is what flips.
        That branch is dispatched on the far side of the boundary and arrives
        as a sign in the adjoint, finite throughout.
        """
    )

    unity = ToleranceCheck("utilization off unity", off_unity, TOLERANCE_UNITY)
    even = ToleranceCheck("diameter under a negated force", even_gap, TOLERANCE_EVEN)
    signed = ToleranceCheck("dd/dN sign vs sign(N_Ed)", mismatched, TOLERANCE_SIGN)
    differenced = ToleranceCheck(
        "adjoint vs a central difference, scaled", max(gaps), TOLERANCE_DERIVATIVE
    )
    checks = [unity, even, signed, differenced]

    return checks


def main(verbose: bool = True) -> None:
    """
    Price the envelope, and report the sign branch it has to carry.
    """
    report = Report(verbose)
    sizes = size_load_cases(FORCES)

    exact_mass = report_sizes(report, sizes)
    report_annealing(report, exact_mass)
    report_live_cases(report)
    checks = report_reversal(report, sizes)

    report.write_heading("Measured against its bound")
    report.write_checks(checks)
    report.write_verdict(verify_checks(checks))


if __name__ == "__main__":
    main()
