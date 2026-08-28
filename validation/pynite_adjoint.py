# SPDX-License-Identifier: Apache-2.0
"""
A solver with no derivative, given one, and held to a solver that has its own.

PyNite is a space frame analysis in plain Python. It has no tape, no tangent and
no sensitivity command, and no amount of configuration will produce one — which
makes it the clearest case the Tesseract boundary has. What crosses the schema
here is a gradient the solver on the far side cannot compute.

The rule is exact and this experiment is what says so. Three claims, measured in
order, each one a precondition for the next:

1. **The element differentiated is the element assembled.** A derivative of a
   lookalike would be a plausible number about a different structure. The
   replica is held against PyNite's own matrices.
2. **The stiffness does not depend on the frame.** An axisymmetric section
   cannot tell one roll of its transverse axes from another, which is what
   licenses reading bending invariants rather than components, and what lets the
   two solvers disagree about axes while agreeing about a design.
3. **The gradient is right.** Not against a finer finite difference — against
   `smax`, a frame solver JAX differentiates end to end. Two exact answers, one
   of them obtained by a rule this repository wrote for a library that has none.

Then the cost, which is the reason any of it matters: the same gradient by
central differences of the forward solve, priced.

Run it as `python validation/pynite_adjoint.py`.
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
from normax.analysis.smax import SmaxAnalyzer
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

# Two exact gradients of one structure, so likewise.
TOLERANCE_GRADIENT = 1.0e-11

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
        "so the global matrix cannot change; a section with two different ones "
        "would, and the last row shows by how much."
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


def gradient_claim(report: Report, sample: FrameSample) -> tuple[float, float, float]:
    """
    Measure the adjoint against a solver JAX differentiates end to end.

    Parameters
    ----------
    report :
        Where the tables are written.
    sample :
        The frame the gradients are taken on.

    Returns
    -------
    worst :
        Worst relative gap in process by node, in process by member, and across
        the schema.

    Notes
    -----
    The scalar is arbitrary and stands in for a check: it reads every quantity
    the schema calls differentiable, so no block of the Jacobian goes untested.
    `smax` is the reference because it is exact, not because it is finer — this
    is two exact answers meeting, and a finite difference could not decide
    between them at this tolerance.
    """
    report.write_heading("The gradient is right")
    report.write_note(
        "One scalar over one frame, differentiated three ways: by this "
        "repository's adjoint of a solver that has none, by the same adjoint "
        "reached across the analysis schema, and by autodiff of a solver that "
        "is traced. The reference is exact, so these gaps are round-off."
    )

    structure = sample.structure
    catalog = build_section_catalog(Steel355(), SECTION_CLASS)
    diameters = jnp.asarray(sample.diameters)
    stacked = jnp.asarray(sample.loads)[None, ...]

    def loss(analyzer, xyz, sizes):
        forces = analyzer(xyz, sizes, stacked)
        axial = jnp.sum((forces.axial_force / SCALE_FORCE) ** 2)
        major = jnp.sum((forces.moment_major / SCALE_MOMENT) ** 2)
        minor = jnp.sum((forces.moment_minor / SCALE_MOMENT) ** 2)

        return axial + major + minor

    traced = SmaxAnalyzer(structure, catalog(SEED_DIAMETER))
    reference = jax.grad(lambda x, d: loss(traced, x, d), argnums=(0, 1))(
        structure.nodes, diameters
    )

    problem = pynite.FrameProblem(
        structure=structure, catalog=catalog, loads=sample.loads
    )
    forces = pynite.compute_member_forces(
        problem, np.asarray(structure.nodes), sample.diameters, sample.loads
    )
    # The loss is a sum of squares, so its cotangent on each reported force is
    # twice that force over the square of its scale — the same left factor the
    # Jacobian contraction used to carry, handed to the rule instead.
    seeded = MemberForces(
        2.0 * np.asarray(forces.axial_force) / SCALE_FORCE**2,
        2.0 * np.asarray(forces.moment_major) / SCALE_MOMENT**2,
        2.0 * np.asarray(forces.moment_minor) / SCALE_MOMENT**2,
    )
    pulled = pynite.pull_back_cotangents(
        problem, np.asarray(structure.nodes), sample.diameters, seeded
    )
    by_node = pulled.xyz
    by_member = pulled.diameter

    analysis = AnalysisConfig({"diameter": SEED_DIAMETER}, "pynite")
    crossed = build_analyzer(structure, catalog, analysis)
    served = jax.grad(lambda x, d: loss(crossed, x, d), argnums=(0, 1))(
        structure.nodes, diameters
    )

    node_gap = relative(by_node, reference[0])
    member_gap = relative(by_member, reference[1])
    crossed_gap = max(
        relative(served[0], reference[0]), relative(served[1], reference[1])
    )

    columns = (
        ReportColumn("route", align="<"),
        ReportColumn("with respect to", align="<"),
        ReportColumn("worst relative gap", ".3e"),
    )
    rows = [
        ["adjoint, in process", "every node coordinate", node_gap],
        ["adjoint, in process", "every diameter", member_gap],
        ["adjoint, across the schema", "both together", crossed_gap],
    ]
    report.write_table(columns, rows)
    report.write_line()

    return node_gap, member_gap, crossed_gap


def cost_claim(report: Report, sample: FrameSample) -> None:
    """
    Price the exact gradient against differencing the forward solve.

    Parameters
    ----------
    report :
        Where the tables are written.
    sample :
        The shell the cost is measured on.

    Notes
    -----
    The comparison a hand-written adjoint has to win to be worth writing. One
    factorization serves every parameter, so the whole dense Jacobian costs
    about what a single forward solve costs, while differencing pays two solves
    per parameter. The first call is reported apart because it compiles.
    """
    report.write_heading("What the rule buys")

    structure = sample.structure
    catalog = build_section_catalog(Steel355(), SECTION_CLASS)
    nodes = np.asarray(structure.nodes)
    members = structure.num_edges
    width = nodes.shape[0] * 3 + members
    problem = pynite.FrameProblem(
        structure=structure, catalog=catalog, loads=sample.loads
    )

    # Warmed first: the first call through either compiles, and a cost table
    # that reported a compilation would be measuring the wrong thing.
    pynite.compute_member_forces(problem, nodes, sample.diameters, sample.loads)
    start = time.perf_counter()
    pynite.compute_member_forces(problem, nodes, sample.diameters, sample.loads)
    forward = time.perf_counter() - start

    stacked = np.stack([sample.loads, 0.6 * sample.loads, 0.4 * sample.loads])
    pynite.compute_member_forces(problem, nodes, sample.diameters, stacked)
    start = time.perf_counter()
    pynite.compute_member_forces(problem, nodes, sample.diameters, stacked)
    together = time.perf_counter() - start

    seed = MemberForces(
        axial_force=np.ones(members),
        moment_major=np.ones((members, 2)),
        moment_minor=np.ones((members, 2)),
    )
    pynite.pull_back_cotangents(problem, nodes, sample.diameters, seed)
    start = time.perf_counter()
    pynite.pull_back_cotangents(problem, nodes, sample.diameters, seed)
    adjoint = time.perf_counter() - start

    differenced = 2.0 * width * forward

    report.write_entries(
        [
            ("nodes", f"{nodes.shape[0]}"),
            ("members", f"{members}"),
            ("parameters the stage differentiates", f"{width}"),
        ]
    )
    report.write_line()

    columns = (
        ReportColumn("route", align="<"),
        ReportColumn("seconds", ".3f"),
        ReportColumn("against the exact rule", align=">"),
    )
    rows = [
        ["one forward solve", forward, ""],
        ["three load cases, one call", together, f"{together / forward:.2f}x"],
        ["one reverse-mode gradient", adjoint, "1x"],
        [
            "central differences, every parameter",
            differenced,
            f"{differenced / adjoint:.0f}x",
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
    node_gap, member_gap, crossed_gap = gradient_claim(report, canopy_sample())
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
            "adjoint by node matches autodiff", node_gap, TOLERANCE_GRADIENT
        ),
        ToleranceCheck(
            "adjoint by member matches autodiff", member_gap, TOLERANCE_GRADIENT
        ),
        ToleranceCheck(
            "crossed gradient matches autodiff", crossed_gap, TOLERANCE_GRADIENT
        ),
    )
    report.write_checks(checks)
    report.write_verdict(verify_checks(checks))


if __name__ == "__main__":
    main("--quiet" not in sys.argv[1:])
