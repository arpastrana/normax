# SPDX-License-Identifier: Apache-2.0
"""
A solver with no derivative, given one, and held to differences of itself.

PyNite is a space frame analysis in plain Python. It has no tape, no tangent and
no sensitivity command, and no amount of configuration will produce one — which
makes it the clearest case the Tesseract boundary has. What crosses the schema
here is a gradient the solver on the far side cannot compute.

The rule is exact and this experiment is what says so. Three claims, measured in
order, each one a precondition for the next:

1. **The element differentiated is the element assembled.** A derivative of a
   lookalike would be a plausible number about a different structure. The
   replica is held against PyNite's own matrices, which the foreign solver
   assembles itself and no other library is asked about.
2. **The stiffness does not depend on the frame.** An axisymmetric section
   cannot tell one roll of its transverse axes from another, which is what
   licenses reading bending invariants rather than components, and what lets two
   solvers disagree about axes while agreeing about a design.
3. **The gradient is right.** Against central differences of the very forward
   solve the rule differentiates, over a sweep of step sizes: the agreement
   traces a V, best in the middle and worse in both directions, which is what an
   exact rule looks like and what a wrong one cannot fake. A second reference
   would only prove two implementations share a habit; its own primal cannot be
   wrong in the same way its adjoint is.

Two rows sit beside that one. The same rule reached across the analysis schema
is held to the differences too, and then to the in-process rule directly, which
is round-off rather than approximation. Only reverse mode crosses: the schema
serves a vector-Jacobian product and no tangent, which is everything a gradient
of an aggregated scalar asks for. The block norms of a traced JAX solver, taken
once and frozen, keep the older claim that an independent exact answer agreed.

Then the cost, which is the reason any of it matters: the same gradient by
central differences of the forward solve, priced.

Run it as `uv run python validation/pynite_adjoint.py`, or `--quiet` for the
verdict alone.
"""

import sys
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from jaxtyping import Float
from Pynite import FEModel3D

from normax.analysis import MemberForces
from normax.analysis import pynite
from normax.analysis.element import SectionRigidity
from normax.analysis.element import assemble_stiffness_global
from normax.analysis.element import assemble_stiffness_local
from normax.analysis.element import compute_direction_cosines
from normax.config import AnalysisConfig
from normax.materials import Steel355
from normax.reporting import Report
from normax.reporting import ReportColumn
from normax.reporting import ToleranceCheck
from normax.reporting import verify_checks
from normax.sections import build_section_catalog
from normax.structures import Structure
from normax.structures import build_gridshell_3d
from normax.tesseract import build_analyzer

jax.config.update("jax_enable_x64", True)

SECTION_CLASS = 3
SEED_DIAMETER = 100.0

# How many randomly oriented members the element claim is measured over.
ELEMENT_TRIALS = 40

# The shell the cost is priced on.
SHELL_RINGS = 16
SHELL_SPOKES = 16
SHELL_RADIUS = 5000.0
SHELL_RISE = 2000.0
SHELL_PRESSURE = 1.5e3

# Two constructions of one matrix, so only the last bits of a double may differ.
TOLERANCE_ELEMENT = 1.0e-13

# One exact rule, fetched twice, so likewise; and a norm frozen from an oracle.
TOLERANCE_GRADIENT = 1.0e-11

# What a central difference can referee: its own truncation, not the rule's error.
TOLERANCE_DIFFERENCE = 1.0e-8

# Steps the reference is swept over, in millimeters, to show the agreement's V.
DIFFERENCE_STEPS = (1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1)

# The sweep's floor, which is the step the reported gaps are read at.
DIFFERENCE_STEP = 1.0e-3

# Recorded from smax at tag local-dev; see docs/oracle_removal.md.
FROZEN_NODE_NORM = 7.90769880190019154587e-03
FROZEN_DIAMETER_NORM = 3.56475439539378179121e-03

# Scales that bring both reported quantities to unit order before summing them.
SCALE_FORCE = 1.0e5
SCALE_MOMENT = 1.0e8


class FrameSample(NamedTuple):
    """
    One structure, its loading and its sections, ready for either solver.

    Attributes
    ----------
    structure :
        The connectivity and the supported nodes.
    diameters :
        Outer diameter of every member.
    loads :
        Force applied at every node.
    """

    structure: Structure
    diameters: Float[np.ndarray, "members"]
    loads: Float[np.ndarray, "nodes 3"]


class GradientGaps(NamedTuple):
    """
    What each route's gradient disagreed with the reference it is held to by.

    Attributes
    ----------
    by_node :
        The in-process rule against central differences, over the coordinates.
    by_member :
        The in-process rule against central differences, over the diameters.
    crossed :
        The same rule reached across the schema, against those differences.
    boundary :
        The crossed rule against the in-process one that serves it.
    frozen :
        The rule's block norms against a traced solver's, recorded once.
    """

    by_node: float
    by_member: float
    crossed: float
    boundary: float
    frozen: float


class GradientMeasurement(NamedTuple):
    """The finite-difference sweep and route gaps behind the gradient report."""

    steps: tuple[float, ...]
    node_errors: tuple[float, ...]
    diameter_errors: tuple[float, ...]
    gaps: GradientGaps


class CostMeasurement(NamedTuple):
    """Measured cost of the PyNite primal, adjoint, and difference reference."""

    nodes: int
    members: int
    parameters: int
    forward_seconds: float
    load_cases_seconds: float
    adjoint_seconds: float
    finite_difference_seconds: float
    finite_difference_measured: bool


def relative(actual: Float[Array, "..."], expected: Float[Array, "..."]) -> float:
    """
    Worst absolute gap, against the largest entry of the reference.

    Parameters
    ----------
    actual :
        The quantity under test.
    expected :
        What it is held to.

    Returns
    -------
    gap :
        Relative worst-case disagreement.
    """
    difference = np.max(np.abs(np.asarray(actual) - np.asarray(expected)))
    scale = max(float(np.max(np.abs(np.asarray(expected)))), 1.0e-300)

    return float(difference) / scale


def canopy_sample() -> FrameSample:
    """
    A frame no plane contains, so both bending components are live.

    Returns
    -------
    sample :
        The structure, its diameters and its loading.

    Notes
    -----
    Deliberately not a shell. A funicular geometry leaves its members almost
    free of bending, which is exactly the case in which a wrong reading of the
    bending would not show. Every member here leans differently and carries
    moment at both ends.
    """
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [4000.0, 0.0, 0.0],
            [4000.0, 3000.0, 0.0],
            [0.0, 3000.0, 0.0],
            [2000.0, 1500.0, 2500.0],
            [1000.0, 800.0, 1400.0],
        ]
    )
    edges = np.array([[0, 4], [1, 4], [2, 4], [3, 4], [0, 1], [1, 2], [0, 5], [5, 4]])
    structure = Structure(nodes=nodes, edges=edges, supports=np.array([0, 1, 2, 3]))

    loads = np.zeros_like(nodes)
    loads[4] = (3.0e4, -2.0e4, -5.0e4)
    loads[5] = (0.0, 1.0e4, -2.0e4)
    diameters = 100.0 + np.arange(edges.shape[0]) * 7.0

    return FrameSample(structure=structure, diameters=diameters, loads=loads)


def shell_sample() -> FrameSample:
    """
    The gridshell cap the cost is priced on, as the acts build it.

    Returns
    -------
    sample :
        The structure, its diameters and its loading.
    """
    structure = build_gridshell_3d(
        SHELL_RINGS,
        SHELL_SPOKES,
        SHELL_RADIUS,
        SHELL_RISE,
        False,
        False,
    )
    nodes = np.asarray(structure.nodes)
    loads = np.zeros_like(nodes)
    supported = np.asarray(structure.supports).ravel()
    free = np.setdiff1d(np.arange(nodes.shape[0]), supported)
    loads[free, 2] = -SHELL_PRESSURE
    diameters = np.full(structure.num_edges, SEED_DIAMETER)

    return FrameSample(structure=structure, diameters=diameters, loads=loads)


def solved_member(start, end, area, inertia, moduli) -> FEModel3D:
    """
    One member the foreign solver has built and analyzed, ready to read.

    Parameters
    ----------
    start :
        Position of the member's first end, in meters.
    end :
        Position of the member's second end, in meters.
    area :
        Cross-sectional area, in square meters.
    inertia :
        Second moment about either transverse axis, in meters to the fourth.
    moduli :
        Elastic modulus, shear modulus and Poisson's ratio.

    Returns
    -------
    member :
        The solved element, whose own stiffness matrices can be read off it.
    """
    elasticity, shear, poissons = moduli
    model = FEModel3D()
    model.add_material("steel", elasticity, shear, poissons, 7850.0)
    model.add_section("tube", area, inertia, inertia, 2.0 * inertia)
    model.add_node("a", *start)
    model.add_node("b", *end)
    model.def_support("a", True, True, True, True, True, True)
    model.add_member("m", "a", "b", "steel", "tube")
    model.add_node_load("b", "FY", -1000.0, case="c")
    model.add_load_combo("lc", {"c": 1.0})
    model.analyze_linear(check_stability=False)

    return next(iter(model.members["m"].sub_members.values()))


def element_claim(report: Report) -> tuple[float, float]:
    """
    Measure the replica against the matrices the solver assembled.

    Parameters
    ----------
    report :
        Where the tables are written.

    Returns
    -------
    worst :
        Worst relative gap on the local matrix, and on the global one.
    """
    report.write_heading("The element differentiated is the element assembled")
    report.write_note(
        "A hand-written adjoint is a claim about a foreign model, so the "
        "replica it differentiates is held against that model's own matrices. "
        "The frames are unrelated: this repository completes its transverse "
        "pair one way and the solver another, which is why the global matrix "
        "agreeing is the stronger of the two rows."
    )

    elasticity = 210.0e9
    poissons = 0.3
    shear = elasticity / (2.0 * (1.0 + poissons))
    moduli = (elasticity, shear, poissons)
    generator = np.random.default_rng(20260825)

    worst_local = 0.0
    worst_global = 0.0
    for _ in range(ELEMENT_TRIALS):
        start = generator.normal(size=3) * 5.0
        end = start + generator.normal(size=3) * 4.0
        area = float(generator.uniform(5.0e-4, 5.0e-3))
        inertia = float(generator.uniform(1.0e-6, 5.0e-5))
        theirs = solved_member(start, end, area, inertia, moduli)
        rigidity = SectionRigidity(
            axial=jnp.asarray(elasticity * area),
            bending=jnp.asarray(elasticity * inertia),
            torsional=jnp.asarray(shear * 2.0 * inertia),
        )
        length = jnp.asarray(float(np.linalg.norm(end - start)))
        local = assemble_stiffness_local(length, rigidity)
        spanned = assemble_stiffness_global(
            jnp.asarray(start), jnp.asarray(end), rigidity
        )
        worst_local = max(worst_local, relative(local, theirs.ke()))
        worst_global = max(worst_global, relative(spanned, theirs.Ke()))

    columns = (
        ReportColumn("matrix", align="<"),
        ReportColumn("worst relative gap", ".3e"),
    )
    rows = [
        ["local, about the member's own axes", worst_local],
        ["global, about the solver's axes", worst_global],
    ]
    report.write_table(columns, rows)
    report.write_line()

    return worst_local, worst_global


def roll_claim(report: Report) -> float:
    """
    Measure how much a roll of the transverse axes moves the global stiffness.

    Parameters
    ----------
    report :
        Where the tables are written.

    Returns
    -------
    worst :
        Worst relative change over the angles tried.

    Notes
    -----
    This is the property the whole reading convention rests on. If it holds,
    two solvers may orient a tube differently and still assemble one stiffness,
    so a design that reads only invariants of the bending pair cannot depend on
    a choice neither of them agreed to make.
    """
    report.write_heading("The stiffness does not know how the frame was rolled")
    report.write_note(
        "Turning a member's transverse axes about its own axis, at fixed "
        "geometry and section. An axisymmetric tube has equal second moments, "
        "so the global matrix cannot change, and this element carries one "
        "bending rigidity because of it — the rows are round-off."
    )

    rigidity = SectionRigidity(
        axial=jnp.asarray(2.5e8),
        bending=jnp.asarray(1.0e4),
        torsional=jnp.asarray(8.0e3),
    )
    start = jnp.asarray([0.0, 0.0, 0.0])
    end = jnp.asarray([3.0, 1.0, 2.0])
    spanned = np.asarray(assemble_stiffness_global(start, end, rigidity))
    length = jnp.linalg.norm(end - start)
    frame = np.asarray(compute_direction_cosines(start, end))
    axis = frame[0]
    local = np.asarray(assemble_stiffness_local(length, rigidity))

    angles = (0.3, 0.9, 1.7, 2.6)
    rows = []
    worst = 0.0
    for angle in angles:
        turned = np.stack(
            [
                axis,
                np.cos(angle) * frame[1] + np.sin(angle) * frame[2],
                -np.sin(angle) * frame[1] + np.cos(angle) * frame[2],
            ]
        )
        transform = np.kron(np.eye(4), turned)
        rolled = transform.T @ local @ transform
        gap = relative(spanned, rolled)
        worst = max(worst, gap)
        rows.append([f"{np.degrees(angle):.0f} degrees", gap])

    columns = (
        ReportColumn("roll", align="<"),
        ReportColumn("relative change in the global stiffness", ".3e"),
    )
    report.write_table(columns, rows)
    report.write_line()

    return worst


def compute_scalar(forces: MemberForces) -> Float[Array, ""]:
    """
    One number off every quantity the schema calls differentiable.

    Parameters
    ----------
    forces :
        What every member carries, under one load case or several.

    Returns
    -------
    total :
        Sum of squares of the axial force and both end moments, each scaled to
        unit order first.

    Notes
    -----
    The scalar is arbitrary and stands in for a check. It reads every reported
    quantity, so no block of the Jacobian goes untested by a gradient of it.
    """
    axial = jnp.sum((forces.axial_force / SCALE_FORCE) ** 2)
    major = jnp.sum((forces.moment_major / SCALE_MOMENT) ** 2)
    minor = jnp.sum((forces.moment_minor / SCALE_MOMENT) ** 2)
    total = axial + major + minor

    return total


def compute_forward(
    problem: pynite.FrameProblem,
    nodes: Float[np.ndarray, "nodes 3"],
    diameters: Float[np.ndarray, "members"],
    loads: Float[np.ndarray, "nodes 3"],
) -> float:
    """
    The scalar, from the foreign solver alone and with no derivative taken.

    Parameters
    ----------
    problem :
        The frame and its section catalog.
    nodes :
        Position of every node.
    diameters :
        Outer diameter of every member.
    loads :
        Force applied at every node.

    Returns
    -------
    value :
        The scalar the gradients are of.
    """
    forces = pynite.compute_member_forces(problem, nodes, diameters, loads)
    value = float(compute_scalar(forces))

    return value


def compute_differences(
    problem: pynite.FrameProblem,
    sample: FrameSample,
    step: float,
) -> pynite.Cotangents:
    """
    Central differences of the forward solve, one parameter at a time.

    Parameters
    ----------
    problem :
        The frame and its section catalog.
    sample :
        The frame the gradient is taken on.
    step :
        Half-width of the difference, in millimeters for either parameter.

    Returns
    -------
    differenced :
        The approximate gradient, by node coordinate and by diameter.

    Notes
    -----
    Two solves per parameter, which is the price the exact rule is measured
    against later. Both parameters carry length units, so one step serves both.
    """
    nodes = np.asarray(sample.structure.nodes, dtype=float)
    diameters = np.asarray(sample.diameters, dtype=float)
    loads = sample.loads
    by_node = np.zeros_like(nodes)
    by_member = np.zeros_like(diameters)

    for node in range(nodes.shape[0]):
        for axis in range(3):
            raised = nodes.copy()
            raised[node, axis] += step
            lowered = nodes.copy()
            lowered[node, axis] -= step
            above = compute_forward(problem, raised, diameters, loads)
            below = compute_forward(problem, lowered, diameters, loads)
            by_node[node, axis] = (above - below) / (2.0 * step)

    for member in range(diameters.shape[0]):
        raised = diameters.copy()
        raised[member] += step
        lowered = diameters.copy()
        lowered[member] -= step
        above = compute_forward(problem, nodes, raised, loads)
        below = compute_forward(problem, nodes, lowered, loads)
        by_member[member] = (above - below) / (2.0 * step)

    differenced = pynite.Cotangents(xyz=by_node, diameter=by_member)

    return differenced


def pull_gradient(
    problem: pynite.FrameProblem,
    sample: FrameSample,
) -> pynite.Cotangents:
    """
    The exact gradient of the scalar, in process, by the hand-written rule.

    Parameters
    ----------
    problem :
        The frame and its section catalog.
    sample :
        The frame the gradient is taken on.

    Returns
    -------
    pulled :
        Cotangent on every node coordinate and every diameter.

    Notes
    -----
    The scalar is a sum of squares, so its cotangent on each reported force is
    twice that force over the square of its scale — the same left factor a
    Jacobian contraction would carry, handed to the rule instead.
    """
    nodes = np.asarray(sample.structure.nodes, dtype=float)
    diameters = sample.diameters
    forces = pynite.compute_member_forces(problem, nodes, diameters, sample.loads)
    seeded = MemberForces(
        2.0 * np.asarray(forces.axial_force) / SCALE_FORCE**2,
        2.0 * np.asarray(forces.moment_major) / SCALE_MOMENT**2,
        2.0 * np.asarray(forces.moment_minor) / SCALE_MOMENT**2,
    )
    pulled = pynite.pull_back_cotangents(problem, nodes, diameters, seeded)

    return pulled


def measure_gradient(sample: FrameSample) -> GradientMeasurement:
    """Measure the adjoint, boundary, and finite-difference references."""
    structure = sample.structure
    catalog = build_section_catalog(Steel355(), SECTION_CLASS)
    diameters = jnp.asarray(sample.diameters)
    stacked = jnp.asarray(sample.loads)[None, ...]
    problem = pynite.FrameProblem(
        structure=structure, catalog=catalog, loads=sample.loads
    )

    pulled = pull_gradient(problem, sample)
    swept = {
        step: compute_differences(problem, sample, step) for step in DIFFERENCE_STEPS
    }
    node_errors = tuple(
        relative(pulled.xyz, swept[step].xyz) for step in DIFFERENCE_STEPS
    )
    diameter_errors = tuple(
        relative(pulled.diameter, swept[step].diameter) for step in DIFFERENCE_STEPS
    )

    analysis = AnalysisConfig({"diameter": SEED_DIAMETER}, "pynite")
    crossed = build_analyzer(structure, catalog, analysis)

    def loss(xyz, sizes):
        forces = crossed(xyz, sizes, stacked)

        return compute_scalar(forces)

    served = jax.grad(loss, argnums=(0, 1))(structure.nodes, diameters)
    referenced = swept[DIFFERENCE_STEP]
    node_norm = float(np.linalg.norm(np.asarray(pulled.xyz)))
    member_norm = float(np.linalg.norm(np.asarray(pulled.diameter)))
    gaps = GradientGaps(
        by_node=relative(pulled.xyz, referenced.xyz),
        by_member=relative(pulled.diameter, referenced.diameter),
        crossed=max(
            relative(served[0], referenced.xyz),
            relative(served[1], referenced.diameter),
        ),
        boundary=max(
            relative(served[0], pulled.xyz),
            relative(served[1], pulled.diameter),
        ),
        frozen=max(
            relative(node_norm, FROZEN_NODE_NORM),
            relative(member_norm, FROZEN_DIAMETER_NORM),
        ),
    )

    return GradientMeasurement(
        tuple(DIFFERENCE_STEPS), node_errors, diameter_errors, gaps
    )


def report_sweep(report: Report, measured: GradientMeasurement) -> None:
    """Write the finite-difference step sweep from structured measurements."""
    columns = (
        ReportColumn("step [mm]", ".0e"),
        ReportColumn("by node", ".3e"),
        ReportColumn("by diameter", ".3e"),
    )
    rows = list(zip(measured.steps, measured.node_errors, measured.diameter_errors))
    report.write_table(columns, rows)
    report.write_note(
        "Truncation falls as the step falls and round-off rises, so an exact "
        "rule is approached from both sides while a wrong one would flatten "
        "onto its own error. The floor is where the gaps below are read."
    )
    report.write_line()


def gradient_claim(report: Report, sample: FrameSample) -> GradientGaps:
    """
    Measure the adjoint against central differences of its own forward solve.

    Parameters
    ----------
    report :
        Where the tables are written.
    sample :
        The frame the gradients are taken on.

    Returns
    -------
    gaps :
        Every route's worst relative disagreement with what it is held to.

    Notes
    -----
    Differencing the rule's own primal is the stronger reference, because an
    error the adjoint makes cannot also be made by the forward solve it is the
    derivative of. Only reverse mode crosses the schema — a vector-Jacobian
    product is served and no tangent is — which is all a gradient of one
    aggregated scalar ever asks for.
    """
    report.write_heading("The gradient is right")
    report.write_note(
        "One scalar over one frame, differentiated three ways: by this "
        "repository's adjoint of a solver that has none, by the same adjoint "
        "reached across the analysis schema, and by central differences of the "
        "forward solve that adjoint differentiates."
    )

    measured = measure_gradient(sample)
    report_sweep(report, measured)
    gaps = measured.gaps

    columns = (
        ReportColumn("route", align="<"),
        ReportColumn("held against", align="<"),
        ReportColumn("worst relative gap", ".3e"),
    )
    rows = [
        ["adjoint, in process", "central differences, by node", gaps.by_node],
        ["adjoint, in process", "central differences, by diameter", gaps.by_member],
        ["adjoint, across the schema", "central differences, both", gaps.crossed],
        ["adjoint, across the schema", "the same rule in process", gaps.boundary],
        ["adjoint, in process", "frozen norms of a traced solver", gaps.frozen],
    ]
    report.write_table(columns, rows)
    report.write_line()

    return gaps


def measure_cost(
    sample: FrameSample, *, run_finite_difference: bool = False
) -> CostMeasurement:
    """Measure the PyNite primal and adjoint, optionally running all differences."""
    structure = sample.structure
    catalog = build_section_catalog(Steel355(), SECTION_CLASS)
    nodes = np.asarray(structure.nodes)
    members = structure.num_edges
    width = nodes.shape[0] * 3 + members
    problem = pynite.FrameProblem(
        structure=structure, catalog=catalog, loads=sample.loads
    )

    def fastest(call, repeats=3):
        best = float("inf")
        for _ in range(repeats):
            start = time.perf_counter()
            call()
            best = min(best, time.perf_counter() - start)

        return best

    # Warmed first: the first call through either compiles, and a cost table
    # that reported a compilation would be measuring the wrong thing.  The
    # best of three warmed calls damps scheduler noise without hiding setup.
    pynite.compute_member_forces(problem, nodes, sample.diameters, sample.loads)
    forward = fastest(
        lambda: pynite.compute_member_forces(
            problem, nodes, sample.diameters, sample.loads
        )
    )

    stacked = np.stack([sample.loads, 0.6 * sample.loads, 0.4 * sample.loads])
    pynite.compute_member_forces(problem, nodes, sample.diameters, stacked)
    together = fastest(
        lambda: pynite.compute_member_forces(problem, nodes, sample.diameters, stacked)
    )

    seed = MemberForces(
        axial_force=np.ones(members),
        moment_major=np.ones((members, 2)),
        moment_minor=np.ones((members, 2)),
    )
    pynite.pull_back_cotangents(problem, nodes, sample.diameters, seed)
    adjoint = fastest(
        lambda: pynite.pull_back_cotangents(problem, nodes, sample.diameters, seed)
    )

    if run_finite_difference:
        start = time.perf_counter()
        compute_differences(problem, sample, DIFFERENCE_STEP)
        differenced = time.perf_counter() - start
    else:
        differenced = 2.0 * width * forward

    return CostMeasurement(
        nodes=nodes.shape[0],
        members=members,
        parameters=width,
        forward_seconds=forward,
        load_cases_seconds=together,
        adjoint_seconds=adjoint,
        finite_difference_seconds=differenced,
        finite_difference_measured=run_finite_difference,
    )


def cost_claim(report: Report, sample: FrameSample) -> CostMeasurement:
    """Price the exact gradient against differencing the forward solve."""
    report.write_heading("What the rule buys")
    measured = measure_cost(sample)

    report.write_entries(
        [
            ("nodes", f"{measured.nodes}"),
            ("members", f"{measured.members}"),
            ("parameters the stage differentiates", f"{measured.parameters}"),
        ]
    )
    report.write_line()

    columns = (
        ReportColumn("route", align="<"),
        ReportColumn("seconds", ".3f"),
        ReportColumn("against the exact rule", align=">"),
    )
    rows = [
        ["one forward solve", measured.forward_seconds, ""],
        [
            "three load cases, one call",
            measured.load_cases_seconds,
            f"{measured.load_cases_seconds / measured.forward_seconds:.2f}x",
        ],
        ["one reverse-mode gradient", measured.adjoint_seconds, "1x"],
        [
            "central differences, every parameter",
            measured.finite_difference_seconds,
            f"{measured.finite_difference_seconds / measured.adjoint_seconds:.0f}x",
        ],
    ]
    report.write_table(columns, rows)
    report.write_line()
    report.write_note(
        "The differenced figure is arithmetic on the measured solve time, not "
        "a run: two solves per parameter is what it would take, and at this "
        "size a descent needing hundreds of gradients could not afford one."
    )
    report.write_line()

    return measured


def main(verbose: bool = True) -> None:
    """
    Run every claim and close with a verdict.

    Parameters
    ----------
    verbose :
        Whether to print the tables.
    """
    report = Report(verbose)
    report.write_banner("PyNite, and the adjoint it does not have")

    worst_local, worst_global = element_claim(report)
    worst_roll = roll_claim(report)
    gaps = gradient_claim(report, canopy_sample())
    cost_claim(report, shell_sample())

    checks = (
        ToleranceCheck(
            "local element matches the solver's", worst_local, TOLERANCE_ELEMENT
        ),
        ToleranceCheck(
            "global element matches the solver's", worst_global, TOLERANCE_ELEMENT
        ),
        ToleranceCheck(
            "global stiffness is roll invariant", worst_roll, TOLERANCE_ELEMENT
        ),
        ToleranceCheck(
            "adjoint by node matches differences", gaps.by_node, TOLERANCE_DIFFERENCE
        ),
        ToleranceCheck(
            "adjoint by member matches differences",
            gaps.by_member,
            TOLERANCE_DIFFERENCE,
        ),
        ToleranceCheck(
            "crossed gradient matches differences", gaps.crossed, TOLERANCE_DIFFERENCE
        ),
        ToleranceCheck(
            "crossed gradient matches the rule in process",
            gaps.boundary,
            TOLERANCE_GRADIENT,
        ),
        ToleranceCheck(
            "gradient norms match the frozen reference", gaps.frozen, TOLERANCE_GRADIENT
        ),
    )
    report.write_checks(checks)
    report.write_verdict(verify_checks(checks))


if __name__ == "__main__":
    main("--quiet" not in sys.argv[1:])
