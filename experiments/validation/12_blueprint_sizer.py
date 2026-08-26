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
One non-differentiable code library, differentiated two ways and tabulated.

Blueprints is scalar Python: its EN 1993-1-1 formula classes subclass `float`
and cannot be traced. This experiment sizes an arch through it twice — in
process behind `jax.pure_callback` with a hand-derived implicit rule, and
across a Tesseract boundary whose derivative endpoint is literal NumPy — and
measures that the two routes agree with each other, with a closed form, and
with central differences.

    forward     the implicit tangent rule of normax.sizing.blueprint
    reverse     that same rule, transposed by JAX into an adjoint
    closed      implicit differentiation of the cubic d^3 = a d + b, on paper
    numeric     a central difference of the host bisection

The last section prices the philosophy gap: Blueprints implements no member
buckling, so its cross-section check sizes a compressed arch thinner than
EN 1993-1-1's member check does. That gap is the point, not an error.

Blueprints is LGPL-2.1, experiment-only, waived 2026-08-15.

Run with `uv run --group pipeline python
experiments/12_blueprint_sizer.py`.
"""

from collections.abc import Callable
from collections.abc import Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from jaxtyping import Float

from normax.analysis import SmaxAnalyzer
from normax.design import Design
from normax.design import DesignParameters
from normax.design import StructuralDesignPipeline
from normax.design import compute_mass
from normax.design import design_envelope
from normax.form_finding import FdmFormFinder
from normax.loads import assemble_load_cases
from normax.loads import create_loads_uniform
from normax.materials import Steel355
from normax.reporting import Report
from normax.reporting import ReportColumn
from normax.reporting import ToleranceCheck
from normax.reporting import checks_passed
from normax.sections import TubeFamily
from normax.sizing import DIAMETER_MINIMUM
from normax.sizing import GAMMA_M0
from normax.sizing import BlueprintSizer
from normax.sizing import Ec3Sizer
from normax.sizing import host_family
from normax.sizing import sized_diameter
from normax.structures import build_arch_2d
from normax.tesseract import BlueprintClient
from normax.tesseract import blueprint_tesseract

TITLE = "One non-differentiable code library, differentiated two ways."

SPAN = 10_000.0
RISE = 3_000.0
TOTAL_LOAD = 180_000.0
NUM_EDGES = 10

# The diameter the frame is analyzed with before the check has spoken.
SEED = 100.0

# Class 3 at S355, so the EC3 comparison shares the exact same geometry.
RATIO = 50.0

YIELD_STRENGTH = 355.0

TARGET = 1e-8
TOLERANCE_UNITY = 1e-9
TOLERANCE_PARITY = 1e-14
TOLERANCE_DERIVATIVE = 1e-12

# Relative step the central differences are taken at.
STEP = 1e-6


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
    ReportColumn("scaled gap", ".2e"),
)

GAP_COLUMNS = (
    ReportColumn("member", align="<"),
    ReportColumn("blueprints [mm]", ".3f"),
    ReportColumn("EN 1993-1-1 [mm]", ".3f"),
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


def diameter_of(case: MemberCase) -> Float[Array, ""]:
    """
    Fully-stressed diameter of one member, unclamped.
    """
    axial = jnp.asarray([case.axial_force])
    moment = jnp.asarray([case.moment])
    solved = sized_diameter(RATIO, YIELD_STRENGTH, axial, moment)

    return solved[0]


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
    family = host_family(RATIO, YIELD_STRENGTH)
    demand_axial = (
        abs(case.axial_force) * GAMMA_M0 / (family.area_coefficient * family.f_y)
    )
    solved = float(diameter_of(case))
    steepness = 3.0 * solved**2 - demand_axial

    scale_axial = GAMMA_M0 / (family.area_coefficient * family.f_y)
    scale_moment = GAMMA_M0 / (family.modulus_coefficient * family.f_y)
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


def arch_problem() -> tuple[StructuralDesignPipeline, StructuralDesignPipeline]:
    """
    The same arch pipeline twice: the sizer in process, and across a boundary.
    """
    structure = build_arch_2d(num_edges=NUM_EDGES, span=SPAN, rise=RISE)
    grade = Steel355()
    family = TubeFamily(RATIO, grade)
    formfinder = FdmFormFinder(structure)
    analyzer = SmaxAnalyzer(structure, family(SEED))

    local = StructuralDesignPipeline(
        formfinder, analyzer, BlueprintSizer(structure, family)
    )
    crossed = StructuralDesignPipeline(
        formfinder,
        analyzer,
        BlueprintClient(structure, blueprint_tesseract(), family),
    )

    return local, crossed


def arch_parameters(pipeline: StructuralDesignPipeline) -> DesignParameters:
    """
    Force densities that land the funicular arch on its intended rise.
    """
    structure = pipeline.sizer.structure
    trial = jnp.full(NUM_EDGES, -1.0)
    loads = create_loads_uniform(structure, TOTAL_LOAD / (NUM_EDGES - 1))
    shape = pipeline.formfinder(trial, loads)
    reached = jnp.max(shape.xyz[:, 2])

    return DesignParameters(trial * reached / RISE, jnp.full(NUM_EDGES, SEED))


def mass_objective(pipeline: StructuralDesignPipeline, params: DesignParameters, loads):
    """
    The enveloped mass as a compiled function of the force densities alone.
    """

    def objective(q):
        design = pipeline(DesignParameters(q, params.diameters), loads)

        return compute_mass(design_envelope(design))

    return jax.jit(objective)


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
    ours = np.asarray(local.sizes.sections.diameter[0])
    theirs = np.asarray(crossed.sizes.sections.diameter[0])
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
    oracle: Float[Array, "members"],
    carried: Float[Array, "members"],
) -> float:
    """
    Both routes' mass gradients, scaled by the largest component.
    """
    largest = float(jnp.max(jnp.abs(oracle)))
    ours = np.asarray(oracle)
    theirs = np.asarray(carried)
    gaps = [abs(b - a) / largest for a, b in zip(ours, theirs)]
    rows = [
        (f"{index}", a, b, gap)
        for index, (a, b, gap) in enumerate(zip(ours, theirs, gaps))
    ]

    report.write_heading("The mass gradient, in process and across the boundary")
    report.write_table(GRADIENT_COLUMNS, rows)

    return max(gaps)


def report_philosophy(report: Report, local: Design, params, loads) -> None:
    """
    The philosophy gap: the same arch sized without and with member buckling.
    """
    structure = build_arch_2d(num_edges=NUM_EDGES, span=SPAN, rise=RISE)
    grade = Steel355()
    family = TubeFamily(RATIO, grade)
    checked_pipeline = StructuralDesignPipeline(
        FdmFormFinder(structure),
        SmaxAnalyzer(structure, family(SEED)),
        Ec3Sizer(structure, family),
    )
    checked = checked_pipeline(params, loads)

    naive = np.asarray(local.sizes.sections.diameter[0])
    strict = np.asarray(checked.sizes.sections.diameter[0])
    rows = [
        (f"{index}", a, b, b / a) for index, (a, b) in enumerate(zip(naive, strict))
    ]

    report.write_heading("The philosophy gap: no buckling against EN 1993-1-1")
    report.write_table(GAP_COLUMNS, rows)
    report.write_note(
        "Same family, same forces: the ratio prices the member check. "
        "Blueprints implements no 6.3.1 flexural buckling, and this gap is "
        "that absence."
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

    local_pipeline, crossed_pipeline = arch_problem()
    params = arch_parameters(local_pipeline)
    structure = local_pipeline.sizer.structure
    loads = assemble_load_cases(
        [create_loads_uniform(structure, TOTAL_LOAD / (NUM_EDGES - 1))]
    )

    local_design = local_pipeline(params, loads)
    crossed_design = crossed_pipeline(params, loads)
    worst_parity = report_parity(report, local_design, crossed_design)

    local_mass = mass_objective(local_pipeline, params, loads)
    crossed_mass = mass_objective(crossed_pipeline, params, loads)
    oracle = jax.grad(local_mass)(params.force_densities)
    carried = jax.grad(crossed_mass)(params.force_densities)
    worst_gradient = report_gradients(report, oracle, carried)

    used = np.asarray(local_design.sizes.utilization)
    free = np.asarray(local_design.sizes.sections.diameter) > DIAMETER_MINIMUM
    worst_unity = float(np.max(np.abs(used[free] - 1.0)))

    report_philosophy(report, local_design, params, loads)

    gradient_check = ToleranceCheck(
        "route parity on the gradient", worst_gradient, TOLERANCE_DERIVATIVE
    )
    checks = (
        ToleranceCheck("derivative disagreement", worst_derivative, TARGET),
        ToleranceCheck("route parity on the sizes", worst_parity, TOLERANCE_PARITY),
        gradient_check,
        ToleranceCheck("departure from unity", worst_unity, TOLERANCE_UNITY),
    )
    report.write_heading("Summary")
    report.write_checks(checks)
    report.write_verdict(checks_passed(checks))


if __name__ == "__main__":
    jnp.set_printoptions(precision=12)
    main()
