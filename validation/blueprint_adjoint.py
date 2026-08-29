# SPDX-License-Identifier: Apache-2.0
"""
The shipped sizing adjoint, differentiated four ways and tabulated.

Blueprints is scalar Python: its EN 1993-1-1 formula classes subclass `float`
and cannot be traced. What is validated here is the hand-written adjoint of
`normax.sizing.blueprint`, the host half of the sizing Tesseract and a shipped
component. The same check is differentiated twice over — in process behind
`jax.pure_callback` with a hand-derived implicit rule, and across the boundary
the package ships, whose derivative endpoint is literal NumPy — and no leg of
the comparison is a second code library, so nothing here can pass by
inheriting a mistake.

    forward     the implicit tangent rule of normax.sizing.blueprint
    reverse     that same rule, transposed by JAX into an adjoint
    closed      implicit differentiation of the cubic d^3 = a d + b, on paper
    numeric     a central difference of the host bisection

The five member cases straddle zero, so the adjoint's `sign(N_Ed)` branch is
dispatched both ways; the catalog floor is its other branch, and the members
it rather than the check decides are the ones the unity assertion masks out.
There is no reduction-factor branch to exercise, the cross-section check
having no reduction factor.

The arch section reads that rule end to end: both routes' mass gradients
against each other, and against central differences of the crossed forward
pass. The frame is analyzed across the analysis Tesseract on both routes, so
the sizer's boundary is the only thing that differs between them. Reverse mode
only out there — a Tesseract serves `vector_jacobian_product` and no tangent
endpoint, which is what an augmented Lagrangian aggregating its rows into one
scalar asks for. Forward mode appears in the single-member section alone,
where the rule is a `custom_jvp` that never leaves the process.

The last section prices the philosophy gap: Blueprints implements no member
buckling, so its cross-section check sizes a compressed arch thinner than
EN 1993-1-1 6.3.1 does. The buckling size beside it is written out here from
the standard's own equations. That gap is the point, not an error.

Blueprints is LGPL-2.1, experiment-only, waived 2026-08-15.

Run with `uv run python validation/blueprint_adjoint.py`.
"""

import math
from collections.abc import Callable
from collections.abc import Sequence
from functools import partial
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from jaxtyping import Float

from normax.design import Design
from normax.design import DesignParameters
from normax.design import StructuralDesignPipeline
from normax.design import compute_mass
from normax.form_finding import FdmFormFinder
from normax.loads import assemble_load_cases
from normax.loads import create_load_uniform
from normax.materials import E_MODULUS
from normax.materials import Steel355
from normax.optimization.nested import design_envelope
from normax.reporting import Report
from normax.reporting import ReportColumn
from normax.reporting import ToleranceCheck
from normax.reporting import verify_checks
from normax.sections import TubeCatalog
from normax.sizing import MemberSizes
from normax.sizing.blueprint import DIAMETER_MINIMUM
from normax.sizing.blueprint import GAMMA_M0
from normax.sizing.blueprint import MemberActions
from normax.sizing.blueprint import SectionCoefficients
from normax.sizing.blueprint import SizeCotangents
from normax.sizing.blueprint import _check_partials
from normax.sizing.blueprint import check_cotangents
from normax.sizing.blueprint import check_members
from normax.sizing.blueprint import coerce_section_coefficients
from normax.sizing.blueprint import size_cotangents
from normax.sizing.blueprint import size_members
from normax.sizing.contract import AbstractMemberSizer
from normax.structures import Structure
from normax.structures import build_arch_2d
from normax.tesseract import TesseractAnalyzer
from normax.tesseract import TesseractSizer

TITLE = "One non-differentiable code library, differentiated four ways."

SPAN = 10_000.0
RISE = 3_000.0
TOTAL_LOAD = 180_000.0
NUM_EDGES = 10

# The diameter the frame is analyzed with before the check has spoken.
SEED = 100.0

# Class 3 at S355, so the buckling comparison shares the exact same geometry.
RATIO = 50.0

YIELD_STRENGTH = 355.0

# EN 1993-1-1 Table 6.1, curve a, which Table 6.2 gives hot-finished tubes.
IMPERFECTION = 0.21

# EN 1993-1-1 6.1, the recommended value for member buckling.
GAMMA_M1 = 1.0

# The smallest diameter the buckling bracket starts from, in millimeters.
PROBE_SMALLEST = 1.0

# Halvings of that bracket, enough to reach the root to the last bit.
BUCKLING_HALVINGS = 100

TARGET = 1e-8
TOLERANCE_UNITY = 1e-9
TOLERANCE_PARITY = 1e-14
TOLERANCE_DERIVATIVE = 1e-12
TOLERANCE_NUMERIC = 1e-8

# Relative step the central differences are taken at.
STEP = 1e-6

# The same, for the mass gradient, where the differences are sharpest.
STEP_DENSITY = 1e-5


class MemberCase(NamedTuple):
    """
    One member: an axial force and the demand moment it carries beside it.

    Attributes
    ----------
    axial_force :
        Design axial force, negative in compression.
    moment :
        Demand moment, non-negative.
    """

    axial_force: float
    moment: float

    @property
    def label(self) -> str:
        """
        The case as it appears in the leftmost column of a table.
        """
        force = self.axial_force / 1e3
        bent = self.moment / 1e6

        return f"{force:.0f} kN, {bent:.2f} kNm"


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
        Implicit differentiation of the cubic, derived on paper.
    numeric :
        Central difference of the host bisection.
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


LOADED = (
    MemberCase(-5.0e5, 1.0e6),
    MemberCase(-3.0e5, 0.0),
    MemberCase(-1.0e4, 5.0e7),
    MemberCase(2.0e5, 1.0e6),
    MemberCase(1.0e6, 3.0e6),
)

DERIVATIVE_COLUMNS = (
    ReportColumn("case", align="<"),
    ReportColumn("forward", "+.12e"),
    ReportColumn("reverse", "+.12e"),
    ReportColumn("closed form", "+.12e"),
    ReportColumn("central diff", "+.12e"),
    ReportColumn("worst", ".2e"),
    ReportColumn("verdict", align="<"),
)

PARITY_COLUMNS = (
    ReportColumn("member", align="<"),
    ReportColumn("in process [mm]", ".9f"),
    ReportColumn("boundary [mm]", ".9f"),
    ReportColumn("gap", ".2e"),
)

GRADIENT_COLUMNS = (
    ReportColumn("edge", align="<"),
    ReportColumn("in process", "+.12e"),
    ReportColumn("boundary", "+.12e"),
    ReportColumn("central diff", "+.12e"),
    ReportColumn("route gap", ".2e"),
    ReportColumn("difference gap", ".2e"),
)

GAP_COLUMNS = (
    ReportColumn("member", align="<"),
    ReportColumn("section check [mm]", ".3f"),
    ReportColumn("6.3.1 buckling [mm]", ".3f"),
    ReportColumn("ratio", ".3f"),
)

FORCE_TITLE = "Sensitivity of the diameter to the axial force, four ways"
MOMENT_TITLE = "Sensitivity of the diameter to the demand moment, four ways"


def relative_gap(actual: float, expected: float) -> float:
    """
    Relative difference between two numbers.
    """
    return abs(actual - expected) / max(abs(expected), 1e-300)


def central_difference(function: Callable[[float], float], x: float, step: float):
    """
    Central difference of a scalar function.
    """
    return (function(x + step) - function(x - step)) / (2.0 * step)


# The in-process route: Blueprints behind a callback, with its own adjoint.
#
# Blueprints is scalar Python and cannot be traced, so the host call is wrapped
# in `jax.pure_callback` and the derivative supplied by hand. The rule is the
# package's own `size_cotangents`; what this script owns is the JAX plumbing
# around it, which the package shed when the sizer moved across the boundary.


def call_host_sizer(coefficients, axial, end_major, end_minor):
    """
    Run Blueprints on the host and return the sizes it demands.
    """
    actions = MemberActions(
        np.asarray(axial), np.asarray(end_major), np.asarray(end_minor)
    )
    sized = size_members(actions, coefficients)

    return np.asarray(sized.diameter), np.asarray(sized.utilization)


def build_sized_call(coefficients: SectionCoefficients) -> Callable:
    """
    A traceable sizing map over one host catalog, differentiated by hand.

    Parameters
    ----------
    coefficients :
        The wall proportion, grade and floor the check is read at.

    Returns
    -------
    sized :
        Actions in, demanded diameter and utilization out, differentiable.
    """

    @jax.custom_vjp
    def sized(axial, end_major, end_minor):
        return primal(axial, end_major, end_minor)

    def primal(axial, end_major, end_minor):
        shapes = (
            jax.ShapeDtypeStruct(axial.shape, axial.dtype),
            jax.ShapeDtypeStruct(axial.shape, axial.dtype),
        )
        call = partial(call_host_sizer, coefficients)

        return jax.pure_callback(call, shapes, axial, end_major, end_minor)

    def forward(axial, end_major, end_minor):
        return primal(axial, end_major, end_minor), (axial, end_major, end_minor)

    def backward(residual, cotangent):
        axial, end_major, end_minor = residual
        seeded = SizeCotangents(np.asarray(cotangent[0]), np.asarray(cotangent[1]))
        actions = MemberActions(
            np.asarray(axial), np.asarray(end_major), np.asarray(end_minor)
        )
        pulled = size_cotangents(actions, coefficients, seeded)

        return (
            jnp.asarray(pulled.axial),
            jnp.asarray(pulled.end_major),
            jnp.asarray(pulled.end_minor),
        )

    sized.defvjp(forward, backward)

    return sized


def call_host_check(coefficients, diameters, axial, end_major, end_minor):
    """
    Run Blueprints' held check on the host and return the utilization.
    """
    axial = np.asarray(axial)
    actions = MemberActions(axial, np.asarray(end_major), np.asarray(end_minor))
    # One size per member is checked in every case, so the held sizes carry the
    # load case axis the check reports against.
    held = np.broadcast_to(np.asarray(diameters), axial.shape)

    return np.asarray(check_members(held, actions, coefficients))


def build_held_call(coefficients: SectionCoefficients) -> Callable:
    """
    A traceable held check over one host catalog, differentiated by hand.

    Parameters
    ----------
    coefficients :
        The wall proportion, grade and floor the check is read at.

    Returns
    -------
    held :
        Sizes and actions in, utilization out, differentiable in both.
    """

    @jax.custom_vjp
    def held(diameters, axial, end_major, end_minor):
        return primal(diameters, axial, end_major, end_minor)

    def primal(diameters, axial, end_major, end_minor):
        shape = jax.ShapeDtypeStruct(axial.shape, axial.dtype)
        call = partial(call_host_check, coefficients)

        return jax.pure_callback(call, shape, diameters, axial, end_major, end_minor)

    def forward(diameters, axial, end_major, end_minor):
        carried = (diameters, axial, end_major, end_minor)

        return primal(*carried), carried

    def backward(residual, cotangent):
        diameters, axial, end_major, end_minor = residual
        actions = MemberActions(
            np.asarray(axial), np.asarray(end_major), np.asarray(end_minor)
        )
        axial_host = np.asarray(axial)
        held = np.broadcast_to(np.asarray(diameters), axial_host.shape)
        pulled = check_cotangents(held, actions, coefficients, np.asarray(cotangent))
        by_size = jnp.asarray(pulled.diameter_held)
        if by_size.ndim > jnp.ndim(diameters):
            by_size = jnp.sum(by_size, axis=0)

        return (
            by_size,
            jnp.asarray(pulled.actions.axial),
            jnp.asarray(pulled.actions.end_major),
            jnp.asarray(pulled.actions.end_minor),
        )

    held.defvjp(forward, backward)

    return held


class CallbackSizer(AbstractMemberSizer):
    """
    Blueprints in process, behind `jax.pure_callback` and a hand adjoint.

    Attributes
    ----------
    structure :
        The structure the block is built on.
    catalog :
        The tube catalog the sizes are drawn from.

    Notes
    -----
    The same check the sizing Tesseract serves, reached without leaving the
    process, so the two routes differ in the boundary alone.
    """

    structure: Structure
    catalog: TubeCatalog
    coefficients: SectionCoefficients = eqx.field(static=True)
    sized: Callable = eqx.field(static=True)
    held: Callable = eqx.field(static=True)

    def __init__(self, structure: Structure, catalog: TubeCatalog) -> None:
        """
        Build the in-process sizer on a structure and its tube catalog.
        """
        self.structure = structure
        self.catalog = catalog
        self.coefficients = coerce_section_coefficients(
            float(catalog.ratio), catalog.material.f_y
        )
        self.sized = build_sized_call(self.coefficients)
        self.held = build_held_call(self.coefficients)

    def __call__(self, forces, buckling_length) -> MemberSizes:
        """
        Size every member for every load case, each on its own.
        """
        diameter, utilization = self.sized(
            forces.axial_force, forces.moment_major, forces.moment_minor
        )
        sections = self.catalog(diameter)

        return MemberSizes(sections, utilization)

    def compute_utilization(self, diameters, forces, buckling_length):
        """
        How hard the sizes the caller owns are worked, by the same check.
        """
        return self.held(
            diameters, forces.axial_force, forces.moment_major, forces.moment_minor
        )


def call_host_partials(coefficients, diameter, axial, moment):
    """
    The check's closed-form partials at one solved size, on the host.
    """
    partials = _check_partials(
        np.asarray(diameter), np.asarray(axial), np.asarray(moment), coefficients
    )

    return (
        np.asarray(partials.slope),
        np.asarray(partials.axial),
        np.asarray(partials.moment),
    )


def build_member_call(coefficients: SectionCoefficients) -> Callable:
    """
    One member's fully-stressed diameter, with a forward tangent rule.

    Parameters
    ----------
    coefficients :
        The wall proportion, grade and floor the check is read at.

    Returns
    -------
    size_one :
        Axial force and demand moment in, diameter out, differentiable both
        ways: the rule is the tangent, and JAX transposes it for the adjoint.

    Notes
    -----
    The implicit function theorem on the check's own residual, whose partials
    the package computes in closed form. A tangent rule rather than a cotangent
    one is what lets the same rule serve forward and reverse mode.
    """

    @jax.custom_jvp
    def size_one(axial, moment):
        shape = jax.ShapeDtypeStruct(jnp.shape(axial), jnp.result_type(float))
        ends = jnp.stack([moment, moment], axis=-1)
        zeros = jnp.zeros_like(ends)
        call = partial(call_host_sizer, coefficients)
        diameter, _ = jax.pure_callback(call, (shape, shape), axial, ends, zeros)

        return diameter

    @size_one.defjvp
    def size_one_jvp(primals, tangents):
        axial, moment = primals
        d_axial, d_moment = tangents
        diameter = size_one(axial, moment)

        shape = jax.ShapeDtypeStruct(jnp.shape(axial), jnp.result_type(float))
        call = partial(call_host_partials, coefficients)
        slope, by_axial, by_moment = jax.pure_callback(
            call, (shape, shape, shape), diameter, axial, moment
        )
        moved = by_axial * d_axial + by_moment * d_moment

        return diameter, -moved / slope

    return size_one


# The check the single-member claims are read through, on the host.
HOST_CATALOG = coerce_section_coefficients(RATIO, YIELD_STRENGTH)

# The same check, traceable: the single-member claims differentiate through it.
SIZED_CALL = build_member_call(HOST_CATALOG)


def diameter_of(case: MemberCase) -> Float[Array, ""]:
    """
    Fully-stressed diameter of one member, unclamped.

    Notes
    -----
    The moment is carried at both ends, so the governing end is the moment the
    case names whichever end the check reads.
    """
    axial = jnp.asarray([case.axial_force])
    moment = jnp.asarray([case.moment])

    return SIZED_CALL(axial, moment)[0]


def closed_derivatives(case: MemberCase) -> tuple[float, float]:
    """
    Both sensitivities from the cubic, derived on paper and written out.

    Notes
    -----
    At the root, `d^3 - a d - b = 0` with `a` and `b` the axial and bending
    demands in diameter units, so `dd/da = d / (3 d^2 - a)` and
    `dd/db = 1 / (3 d^2 - a)` — a derivation that never states the check's
    utilization, and so shares no algebra with the implicit rule it judges.
    """
    catalog = coerce_section_coefficients(RATIO, YIELD_STRENGTH)
    demand_axial = (
        abs(case.axial_force) * GAMMA_M0 / (catalog.area_coefficient * catalog.f_y)
    )
    solved = float(diameter_of(case))
    steepness = 3.0 * solved**2 - demand_axial

    scale_axial = GAMMA_M0 / (catalog.area_coefficient * catalog.f_y)
    scale_moment = GAMMA_M0 / (catalog.modulus_coefficient * catalog.f_y)
    by_force = np.sign(case.axial_force) * scale_axial * solved / steepness
    by_moment = scale_moment / steepness

    return float(by_force), float(by_moment)


def derivatives_force(case: MemberCase) -> DerivativeSet:
    """
    Sensitivity of the diameter to the axial force, four ways.
    """

    def sized(axial_force):
        probed = case._replace(axial_force=axial_force)

        return diameter_of(probed)

    step = abs(case.axial_force) * STEP
    closed, _ = closed_derivatives(case)
    quotient = central_difference(lambda x: float(sized(x)), case.axial_force, step)
    forward = float(jax.jacfwd(sized)(case.axial_force))
    reverse = float(jax.grad(sized)(case.axial_force))

    return DerivativeSet(forward, reverse, closed, float(quotient))


def derivatives_moment(case: MemberCase) -> DerivativeSet:
    """
    Sensitivity of the diameter to the demand moment, four ways.
    """

    def sized(moment):
        probed = case._replace(moment=moment)

        return diameter_of(probed)

    step = max(abs(case.moment), 1e6) * STEP
    _, closed = closed_derivatives(case)
    if case.moment == 0.0:
        # A demand moment lives on [0, inf): the derivative at zero is
        # one-sided, so the probe is the second-order one-sided stencil, at a
        # wider step that amortizes the bisection's own ulp-level noise.
        stride = 100.0 * step
        stenciled = -3.0 * float(sized(0.0)) + 4.0 * float(sized(stride))
        quotient = (stenciled - float(sized(2.0 * stride))) / (2.0 * stride)
    else:
        quotient = central_difference(lambda x: float(sized(x)), case.moment, step)
    forward = float(jax.jacfwd(sized)(case.moment))
    reverse = float(jax.grad(sized)(case.moment))

    return DerivativeSet(forward, reverse, closed, float(quotient))


def build_arch_problem() -> tuple[StructuralDesignPipeline, StructuralDesignPipeline]:
    """
    The same arch pipeline twice: the sizer in process, and across a boundary.

    Returns
    -------
    local :
        The pipeline whose check runs behind a callback in this process.
    crossed :
        The pipeline whose check runs behind the sizing Tesseract.

    Notes
    -----
    One analyzer serves both, and it crosses the analysis boundary in either,
    so the sizer's boundary is the only difference the arch tables read.
    """
    structure = build_arch_2d(num_edges=NUM_EDGES, span=SPAN, rise=RISE)
    grade = Steel355()
    catalog = TubeCatalog(RATIO, grade)
    formfinder = FdmFormFinder(structure)
    analyzer = TesseractAnalyzer(structure, catalog, "pynite")

    local = StructuralDesignPipeline(
        formfinder, analyzer, CallbackSizer(structure, catalog)
    )
    crossed = StructuralDesignPipeline(
        formfinder, analyzer, TesseractSizer(structure, catalog, "blueprint")
    )

    return local, crossed


def read_arch_parameters(pipeline: StructuralDesignPipeline) -> DesignParameters:
    """
    Force densities that land the funicular arch on its intended rise.
    """
    structure = pipeline.sizer.structure
    trial = jnp.full(NUM_EDGES, -1.0)
    loads = create_load_uniform(structure, TOTAL_LOAD / (NUM_EDGES - 1))
    shape = pipeline.formfinder(trial, loads)
    reached = jnp.max(shape.xyz[:, 2])

    return DesignParameters(trial * reached / RISE, jnp.full(NUM_EDGES, SEED))


def build_mass_objective(
    pipeline: StructuralDesignPipeline,
    params: DesignParameters,
    loads,
) -> Callable:
    """
    The mass as a function of the force densities alone.

    Parameters
    ----------
    pipeline :
        The three blocks, whichever side of the boundary their check sits on.
    params :
        The design the diameters are read from; its densities are replaced.
    loads :
        The load cases every stage is run under.

    Returns
    -------
    compute_objective :
        Force densities in, total mass out, differentiable in reverse mode.

    Notes
    -----
    Eager on purpose: a compiled trace around a Tesseract call has closed a
    process-global file descriptor here, so the composed side is never jitted.
    """

    def compute_objective(densities):
        moved = DesignParameters(densities, params.diameters)
        design = pipeline(moved, loads)

        return compute_mass(design)

    return compute_objective


def compute_differences(
    objective: Callable,
    densities: Float[Array, "edges"],
) -> Float[np.ndarray, "edges"]:
    """
    Central differences of a scalar objective in every force density.

    Parameters
    ----------
    objective :
        Force densities in, one number out.
    densities :
        The point the differences are taken at.

    Returns
    -------
    quotients :
        One difference quotient per force density.

    Notes
    -----
    The forward pass alone, which is all a Tesseract needs to answer: the
    gradient this judges is the one the boundary's own adjoint returned.
    """
    center = np.asarray(densities)
    quotients = []
    for index in range(center.size):
        step = abs(float(center[index])) * STEP_DENSITY
        moved = center.copy()
        moved[index] = center[index] + step
        raised = float(objective(jnp.asarray(moved)))
        moved[index] = center[index] - step
        lowered = float(objective(jnp.asarray(moved)))
        quotients.append((raised - lowered) / (2.0 * step))

    return np.asarray(quotients)


def sized_design(pipeline: StructuralDesignPipeline, params, loads) -> Design:
    """
    The design the check demands, rather than a verdict on the one held.

    Notes
    -----
    Calling the pipeline runs the held check and echoes the diameters back;
    this experiment is about the sizing map, so it asks the sizer itself and
    reconciles the load cases afterwards.
    """
    shape = pipeline.formfinder(params.shape_parameters, loads.formfinding)
    forces = pipeline.analyzer(shape.xyz, params.diameters, loads.analysis)
    sizes = pipeline.sizer(forces, shape.lengths)

    return design_envelope(Design(shape, forces, sizes))


def report_derivatives(
    report: Report,
    title: str,
    measured: Sequence[tuple[str, DerivativeSet]],
) -> float:
    """
    One block of the comparison table, and the worst disagreement in it.
    """

    def derivative_row(label, found):
        columns = (found.forward, found.reverse, found.closed, found.numeric)

        return (label, *columns, found.worst, found.verdict)

    rows = [derivative_row(label, found) for label, found in measured]

    report.write_heading(title)
    report.write_table(DERIVATIVE_COLUMNS, rows)

    return max(found.worst for _, found in measured)


def report_parity(report: Report, local: Design, crossed: Design) -> float:
    """
    The two routes' designs, member by member, and the worst gap between them.
    """
    ours = np.asarray(local.sizes.sections.diameter)
    theirs = np.asarray(crossed.sizes.sections.diameter)
    gaps = [relative_gap(b, a) for a, b in zip(ours, theirs)]
    rows = [
        (f"{index}", a, b, gap)
        for index, (a, b, gap) in enumerate(zip(ours, theirs, gaps))
    ]

    report.write_heading("Route A against route B, the same design twice")
    report.write_table(PARITY_COLUMNS, rows)

    return max(gaps)


def report_gradients(
    report: Report,
    in_process: Float[Array, "edges"],
    crossed: Float[Array, "edges"],
    numeric: Float[np.ndarray, "edges"],
) -> tuple[float, float]:
    """
    Both routes' mass gradients and the differences, scaled by the largest.

    Parameters
    ----------
    report :
        Where the table is written.
    in_process :
        The gradient through the callback route.
    crossed :
        The gradient through the boundary the package ships.
    numeric :
        Central differences of the crossed forward pass.

    Returns
    -------
    worst_route :
        Largest scaled gap between the two routes.
    worst_numeric :
        Largest scaled gap between the crossed gradient and the differences.
    """
    ours = np.asarray(in_process)
    theirs = np.asarray(crossed)
    quotients = np.asarray(numeric)
    largest = float(np.max(np.abs(ours)))
    route = [abs(b - a) / largest for a, b in zip(ours, theirs)]
    against = [abs(c - b) / largest for b, c in zip(theirs, quotients)]
    measured = zip(ours, theirs, quotients, route, against)
    rows = [(f"{index}", *found) for index, found in enumerate(measured)]

    report.write_heading("The mass gradient, by two routes and by difference")
    report.write_table(GRADIENT_COLUMNS, rows)

    return max(route), max(against)


# EN 1993-1-1 6.3.1, the member check Blueprints does not implement.


def compute_slenderness(diameter: float, length: float) -> float:
    """
    Non-dimensional slenderness of a CHS strut.

    Parameters
    ----------
    diameter :
        Outer diameter of the tube.
    length :
        Buckling length, taken as the member itself.

    Returns
    -------
    slenderness :
        The bar-lambda of EN 1993-1-1 6.3.1.3, Eq. 6.50.

    Notes
    -----
    With the wall a fixed proportion of the diameter the radius of gyration is
    proportional to it, so the slenderness falls as one over the diameter.
    """
    shape = HOST_CATALOG.modulus_coefficient / (2.0 * HOST_CATALOG.area_coefficient)
    gyration = math.sqrt(shape) * diameter
    reference = math.pi * gyration / math.sqrt(YIELD_STRENGTH / E_MODULUS)

    return length / reference


def compute_reduction(slenderness: float) -> float:
    """
    The buckling reduction factor, capped at one.

    Parameters
    ----------
    slenderness :
        The bar-lambda the member reaches.

    Returns
    -------
    reduction :
        The chi of EN 1993-1-1 6.3.1.2, Eq. 6.49, never above one.
    """
    shifted = IMPERFECTION * (slenderness - 0.2)
    factor = 0.5 * (1.0 + shifted + slenderness**2)
    reduction = 1.0 / (factor + math.sqrt(factor**2 - slenderness**2))

    return min(reduction, 1.0)


def solve_residual(residual: Callable[[float], float]) -> float:
    """
    The one root of a residual that rises strictly in the diameter.

    Parameters
    ----------
    residual :
        Capacity less demand, negative below the root.

    Returns
    -------
    root :
        The diameter the residual vanishes at, bracketed then halved.

    Notes
    -----
    Capacity rises strictly because the area grows as the diameter squared
    while the slenderness falls, so the bracket needs only to be widened until
    it contains the root and bisection is unconditionally safe.
    """
    low = PROBE_SMALLEST
    high = PROBE_SMALLEST
    while residual(high) < 0.0:
        high *= 2.0
    for _ in range(BUCKLING_HALVINGS):
        middle = 0.5 * (low + high)
        if residual(middle) < 0.0:
            low = middle
        else:
            high = middle
    root = 0.5 * (low + high)

    return root


def size_for_buckling(axial_force: float, length: float) -> float:
    """
    The diameter EN 1993-1-1 6.3.1 works to exactly one, floored.

    Parameters
    ----------
    axial_force :
        Design axial force, negative in compression.
    length :
        Buckling length, taken as the member itself.

    Returns
    -------
    diameter :
        The size the member check demands, at the catalog's minimum or above.

    Notes
    -----
    A tie does not buckle, so tension is 6.2.3, Eq. 6.6 in closed form; a strut
    is Eq. 6.47 with the reduction factor, and its residual is bisected.
    """
    scale = GAMMA_M0 / (HOST_CATALOG.area_coefficient * YIELD_STRENGTH)
    if axial_force >= 0.0:
        stretched = math.sqrt(axial_force * scale)

        return max(stretched, DIAMETER_MINIMUM)

    def residual(diameter):
        slenderness = compute_slenderness(diameter, length)
        reduction = compute_reduction(slenderness)
        area = HOST_CATALOG.area_coefficient * diameter**2
        capacity = reduction * area * YIELD_STRENGTH / GAMMA_M1

        return capacity - abs(axial_force)

    buckled = solve_residual(residual)

    return max(buckled, DIAMETER_MINIMUM)


def select_worst_axial(
    axial_force: Float[Array, "load_cases members"],
) -> Float[np.ndarray, "members"]:
    """
    Each member's governing axial force, the largest it carries anywhere.
    """
    carried = np.atleast_2d(np.asarray(axial_force))
    governing = np.argmax(np.abs(carried), axis=0)
    members = np.arange(carried.shape[1])

    return carried[governing, members]


def report_philosophy(report: Report, design: Design) -> None:
    """
    The philosophy gap: the same arch sized without and with member buckling.
    """
    axial = select_worst_axial(design.forces.axial_force)
    lengths = np.asarray(design.shape.lengths)
    naive = np.asarray(design.sizes.sections.diameter)
    strict = [size_for_buckling(force, span) for force, span in zip(axial, lengths)]
    measured = zip(naive, strict)
    rows = [(f"{index}", a, b, b / a) for index, (a, b) in enumerate(measured)]

    report.write_heading("The philosophy gap: no buckling against EN 1993-1-1")
    report.write_table(GAP_COLUMNS, rows)
    report.write_note(
        "Same arch, same forces: the ratio prices the member check. On the "
        "left is Blueprints' cross-section check, axial force with bending and "
        "no 6.3.1; on the right is flexural buckling for the same axial force "
        "over a buckling length of one member, Eq. 6.47 with Eq. 6.49 and "
        "Eq. 6.50 written out in this file. Blueprints implements no such "
        "clause, and this gap is that absence. The left column carries an end "
        "moment the right one does not, so the ratio understates the price by "
        "the few percent that bending adds."
    )


def main(verbose: bool = True) -> None:
    """
    Tabulate the two routes over one member and over the arch.
    """
    report = Report(verbose)
    report.write_line(TITLE)

    entries = (
        ("d/t", f"{RATIO:.1f}"),
        ("f_y", f"{YIELD_STRENGTH:.0f} MPa"),
        ("diameter floor", f"{DIAMETER_MINIMUM} mm"),
        ("agreement target", f"{TARGET:.0e}"),
    )
    report.write_heading("S355 tube, Blueprints' cross-section check")
    report.write_entries(entries)

    by_force = [(case.label, derivatives_force(case)) for case in LOADED]
    by_moment = [(case.label, derivatives_moment(case)) for case in LOADED]
    worst_force = report_derivatives(report, FORCE_TITLE, by_force)
    worst_moment = report_derivatives(report, MOMENT_TITLE, by_moment)
    worst_derivative = max(worst_force, worst_moment)

    local_pipeline, crossed_pipeline = build_arch_problem()
    params = read_arch_parameters(local_pipeline)
    structure = local_pipeline.sizer.structure
    loads = assemble_load_cases(
        [create_load_uniform(structure, TOTAL_LOAD / (NUM_EDGES - 1))]
    )

    local_design = sized_design(local_pipeline, params, loads)
    crossed_design = sized_design(crossed_pipeline, params, loads)
    worst_parity = report_parity(report, local_design, crossed_design)

    local_mass = build_mass_objective(local_pipeline, params, loads)
    crossed_mass = build_mass_objective(crossed_pipeline, params, loads)
    in_process = jax.grad(local_mass)(params.shape_parameters)
    carried = jax.grad(crossed_mass)(params.shape_parameters)
    quotients = compute_differences(crossed_mass, params.shape_parameters)
    worst_gradient, worst_numeric = report_gradients(
        report, in_process, carried, quotients
    )

    # Utilization keeps its load case axis; one section per member is checked
    # in every case, so the floor mask is broadcast across them.
    used = np.asarray(local_design.sizes.utilization)
    sized = np.asarray(local_design.sizes.sections.diameter) > DIAMETER_MINIMUM
    free = np.broadcast_to(sized, used.shape)
    worst_unity = float(np.max(np.abs(used[free] - 1.0)))

    report_philosophy(report, crossed_design)

    gradient_check = ToleranceCheck(
        "route parity on the gradient", worst_gradient, TOLERANCE_DERIVATIVE
    )
    numeric_check = ToleranceCheck(
        "the gradient against differences", worst_numeric, TOLERANCE_NUMERIC
    )
    checks = (
        ToleranceCheck("derivative disagreement", worst_derivative, TARGET),
        ToleranceCheck("route parity on the sizes", worst_parity, TOLERANCE_PARITY),
        gradient_check,
        numeric_check,
        ToleranceCheck("departure from unity", worst_unity, TOLERANCE_UNITY),
    )
    report.write_heading("Summary")
    report.write_checks(checks)
    report.write_verdict(verify_checks(checks))


if __name__ == "__main__":
    jnp.set_printoptions(precision=12)
    main()
