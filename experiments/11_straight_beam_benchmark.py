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
A straight beam, where the answer is known in closed form.

Two stages rather than three: `smax` says what the members carry and the sizing
map returns the diameter at which EN 1993-1-1 is exactly satisfied. There is no
form finding and no force density — the geometry is given, so the shape is an
input rather than an unknown, and what is left is the T2 to T3 handoff alone.

The beam is the arch of experiment 03 with its z coordinate projected back to the
ground plane: same span, same discretization, same total load, same supports.
Only the shape differs, which is what makes the two comparable.

One load case, and four things checked against arithmetic rather than against
another solver.

    axial       zero to machine precision, a straight beam under vertical load
                carrying nothing along its own axis
    statics     the end moments against the exact moment diagram of a simply
                supported beam under equal point loads
    sizing      the diameter against the closed-form inverse of the bending
                check, on both class branches
    coupling    the analysis is statically determinate, so the sizes cannot
                change the forces and one staggered pass is exact rather than
                merely close

**This is the benchmark because nothing here needs a solver to be believed.**
The arch has no closed form: its moments are the elastic leftover of an
unstressed reference state and its sizes come from a bisection. A straight beam
has both — the moment diagram is statics and the required diameter is a cube
root — so each stage is checked against arithmetic instead of against the other.

**One load case, deliberately.** The arch answers to three because an asymmetric
case is what raises the bending a funicular shape cannot carry axially. A beam
carries everything in bending already, so a second case would move the numbers
without testing anything the first one leaves untested.

**The staggered coupling is exact here, and that is the point of running it.**
Member forces of a determinate structure do not depend on the sections, so the
diameters the frame is analyzed with cannot reach the diameters the check
returns. On the arch the same loop costs about 1.2% of the mass on its first
pass. The difference is indeterminacy, not the code.

**A member in pure bending is reported as governed by tension.** The limit state
splits the cross-section check on the sign of the axial force, and a beam sits at
exactly zero, which takes the branch that carries no buckling reduction. The
branch is the right one; only its name reads oddly on a member carrying no axial
force at all.

Run with `uv run --group pipeline python experiments/11_straight_beam_benchmark.py`.
"""

from pathlib import Path
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from ec3x.actions import MemberActions
from ec3x.classification import is_plastic
from ec3x.material import Steel
from ec3x.section import TubeCatalogue
from ec3x.sizing import diameter_required
from ec3x.sizing import end_moments
from ec3x.sizing import governing_limit_state
from ec3x.sizing import mass_of_tubes
from ec3x.sizing import utilization_design
from jaxtyping import Array
from jaxtyping import Float
from smax import CompiledStructure

from normax.analysis import member_forces
from normax.analysis import prepare_model
from normax.loads import create_loads_uniform
from normax.reporting import Report
from normax.reporting import ReportColumn
from normax.reporting import ToleranceCheck
from normax.reporting import checks_passed
from normax.sizing import neutral_sections
from normax.structures import Structure
from normax.structures import build_arch_2d
from normax.visualization import BeamSizing
from normax.visualization import BeamStatics
from normax.visualization import SizedMembers
from normax.visualization import figure_beam_profile
from normax.visualization import figure_benchmark

# The arch of experiment 03, flattened. Units are millimeters and newtons.
SPAN = 10_000.0
TOTAL_LOAD = 180_000.0
NUM_EDGES = 20

# Rise of the arch the beam is projected from. It sets the node spacing along the
# span and nothing else, every z coordinate being dropped.
ARCH_RISE = 3_000.0

# The diameter the frame is analyzed with before the check has spoken. It cannot
# reach the answer here, the structure being determinate, and the stagger says so.
SEED = 100.0

# One case, carrying the whole load. A beam is in bending under any of them.
CASE_NAME = "LC1 uniform"

PASSES = 4

TOLERANCE_UTILIZATION = 1e-9
TOLERANCE_CLOSED_FORM = 1e-12

# Newtons, against a total applied load of 180 kN.
TOLERANCE_AXIAL = 1e-6

# The moment diagram is exact statics and the sizes follow it, so what is left in
# both is the conditioning of the linear solve. Measured floor on this beam:
# 1.8e-13 at ten members, 1.5e-12 at twenty, 1.0e-11 at forty and 1.2e-10 at
# eighty. It grows with the mesh, so both are pinned with headroom rather than on
# the floor — a solver that was actually wrong would miss by percent.
TOLERANCE_STATICS = 1e-10
TOLERANCE_STAGGER = 1e-10

FIGURES = Path(__file__).resolve().parent.parent / "figures"

# Compiled programs outlive the process, so a second run pays for arithmetic
# alone. Every compilation here is well under the one second the persistent
# cache keeps by default, which would otherwise leave all of them out of it.
COMPILATION_CACHE = Path(__file__).resolve().parent.parent / ".jax_cache"
COMPILATION_CACHE.mkdir(exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(COMPILATION_CACHE))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

STEEL = Steel()
SECTION_CLASS = 3
CATALOGUE = TubeCatalogue.at_class_limit(STEEL, SECTION_CLASS)
# The analysis takes normax's neutral sections, so the standard's tube is
# restated at the boundary rather than duck-typed through it.
SECTION_SEED = neutral_sections(CATALOGUE(SEED))

CLASSES = (2, 3)

LIMIT_NAMES = {
    0.0: "catalogue minimum",
    1.0: "tension",
    2.0: "cross-section",
    3.0: "6.61 major",
    4.0: "6.62 minor",
}

# The reads the reports make, compiled. Left eager each one costs an XLA
# compilation per primitive, which is most of what reporting a design costs.
member_forces_compiled = eqx.filter_jit(member_forces)
governing_compiled = eqx.filter_jit(governing_limit_state)


class BeamProblem(NamedTuple):
    """
    The flattened beam, its analysis model, and the one case it answers to.

    Attributes
    ----------
    structure :
        The structure supplying the connectivity, the supports and the nodes.
    model :
        The compiled analysis model, from `normax.analysis.smax.prepare_model`.
    loads :
        Force applied at every node in the single load case.
    """

    structure: Structure
    model: CompiledStructure
    loads: Float[Array, "nodes 3"]

    @property
    def num_edges(self) -> int:
        """
        Number of members in this mesh.
        """
        return int(self.structure.edges.shape[0])

    @property
    def seed(self) -> Float[Array, "edges"]:
        """
        Seed diameter used for the first analysis before sizing.
        """
        return jnp.full(self.num_edges, SEED)

    @property
    def positions(self) -> Float[np.ndarray, "nodes"]:
        """
        Position of every node along the span.
        """
        return np.asarray(self.structure.nodes[:, 0])

    @property
    def lengths(self) -> Float[Array, "edges"]:
        """
        Length of every member of the undeformed beam.
        """
        nodes = self.structure.nodes
        spans = nodes[self.structure.edges[:, 1]] - nodes[self.structure.edges[:, 0]]

        return jnp.linalg.norm(spans, axis=1)


class BeamDesign(NamedTuple):
    """
    Everything the two stages produce for one set of sizes.

    Attributes
    ----------
    actions :
        Axial force, both design moments and both moment factors.
    lengths :
        Length of every member.
    diameters :
        Outer diameter EN 1993-1-1 requires of every member.
    utilization :
        Demand over resistance at those diameters.
    mass :
        Total mass of the members.
    """

    actions: MemberActions
    lengths: Float[Array, "edges"]
    diameters: Float[Array, "edges"]
    utilization: Float[Array, "edges"]
    mass: Float[Array, ""]


class StaggerRun(NamedTuple):
    """
    What repeating the analysis and the check does to the sizes.

    Attributes
    ----------
    moves :
        Largest relative change in diameter produced by each pass.
    masses :
        Mass after each pass.
    """

    moves: Float[np.ndarray, "passes"]
    masses: Float[np.ndarray, "passes"]

    @property
    def worst_move_after_first(self) -> float:
        """
        Largest diameter move any pass after the first one makes.
        """
        return float(np.max(self.moves[1:]))


class BeamResults(NamedTuple):
    """
    Everything measured about the beam that a figure or a summary reads.

    Attributes
    ----------
    design :
        The fully-stressed design on the class the comparisons are made at.
    statics :
        Bending moment along the beam, computed and predicted.
    sizing :
        Diameter of every member, required and in closed form.
    stagger :
        What repeating the analysis and the check does to the sizes.
    """

    design: BeamDesign
    statics: BeamStatics
    sizing: BeamSizing
    stagger: StaggerRun

    @property
    def worst_statics(self) -> float:
        """
        Worst departure of a moment from statics, scaled by the largest.
        """
        scale = float(np.max(np.abs(self.statics.exact)))
        gap = np.abs(self.statics.computed - self.statics.exact)

        return float(np.max(gap)) / scale

    @property
    def worst_closed_form(self) -> float:
        """
        Worst relative departure of a diameter from its closed-form inverse.
        """
        gap = np.abs(self.sizing.required - self.sizing.closed_form)

        return float(np.max(gap / self.sizing.closed_form))


def beam_problem() -> BeamProblem:
    """
    The arch of experiment 03, laid flat, with its model prepared on it.

    Neither the model nor the load case depends on a size, so both are built here
    and passed to everything below. The analysis model is the expensive one and
    the reason the design can be compiled at all: preparing it reads support
    flags in Python, which a tracer cannot follow.
    """
    spread = TOTAL_LOAD / (NUM_EDGES - 1)
    arch = build_arch_2d(num_edges=NUM_EDGES, span=SPAN, rise=ARCH_RISE)
    structure = arch._replace(nodes=arch.nodes.at[:, 2].set(0.0))
    model = prepare_model(structure, SECTION_SEED)
    setup = BeamProblem(structure, model, create_loads_uniform(structure, spread))

    return setup


@eqx.filter_jit
def build(
    setup: BeamProblem,
    diameters: Float[Array, "edges"],
    catalogue: TubeCatalogue,
) -> BeamDesign:
    """
    Analyze the beam at one set of sizes, then size it to satisfy the standard.

    Compiled, which is what makes the staggered passes affordable: every caller
    here passes the same prepared model, so one trace serves all of them.
    """
    section = neutral_sections(catalogue(SEED))
    member = member_forces(
        setup.model,
        setup.structure.nodes,
        diameters,
        section,
        setup.loads,
    )
    moment_major, factor_major = end_moments(
        member.moment_major[:, 0], member.moment_major[:, 1]
    )
    moment_minor, factor_minor = end_moments(
        member.moment_minor[:, 0], member.moment_minor[:, 1]
    )
    actions = MemberActions(
        member.axial_force, moment_major, moment_minor, factor_major, factor_minor
    )

    lengths = setup.lengths
    required = diameter_required(actions, lengths, catalogue)
    sized = catalogue(required)
    used = utilization_design(sized, actions, lengths)
    design = BeamDesign(actions, lengths, required, used, mass_of_tubes(sized, lengths))

    return design


def worst_utilization(design: BeamDesign) -> float:
    """
    Largest absolute departure of utilization from unity.
    """
    return float(jnp.max(jnp.abs(design.utilization - 1.0)))


def moment_statics(position: float, setup: BeamProblem) -> float:
    """
    Bending moment at a position, from the statics of a simply supported beam.
    """
    applied = -np.asarray(setup.loads[:, 2])
    reaction = 0.5 * float(np.sum(applied))
    lever = np.where(setup.positions < position, position - setup.positions, 0.0)

    return reaction * position - float(np.sum(applied * lever))


def diameter_closed_form(
    moment: Float[Array, "edges"],
    catalogue: TubeCatalogue,
    section_class: int,
) -> Float[Array, "edges"]:
    """
    Diameter carrying a moment in bending alone, inverted in closed form.

    With no axial force the check is the bending stress alone, and every section
    modulus is a monomial in the diameter, so the size inverts as a cube root.
    The unit-diameter modulus comes from the section module rather than being
    restated here, so the two cannot drift apart.
    """
    unit = catalogue(1.0)
    modulus = (
        unit.modulus_plastic if is_plastic(section_class) else unit.modulus_elastic
    )
    demanded = jnp.asarray(moment) * STEEL.gamma_m0 / (modulus * STEEL.f_y)

    return demanded ** (1.0 / 3.0)


def beam_statics(setup: BeamProblem, catalogue: TubeCatalogue) -> BeamStatics:
    """
    Bending moment at every node, from the solver and from statics.

    Read from the raw end moments rather than from the actions, the check having
    already reduced each member's two ends to the larger of them.
    """
    member = member_forces_compiled(
        setup.model,
        setup.structure.nodes,
        setup.seed,
        catalogue,
        setup.loads,
    )
    ends = np.asarray(member.moment_major)

    computed = []
    exact = []
    for node in range(setup.positions.shape[0]):
        computed.append(abs(float(ends[0, 0] if node == 0 else ends[node - 1, 1])))
        exact.append(abs(moment_statics(float(setup.positions[node]), setup)))

    return BeamStatics(setup.positions, np.asarray(exact), np.asarray(computed))


def beam_sizing(
    design: BeamDesign,
    positions: Float[np.ndarray, "nodes"],
    catalogue: TubeCatalogue,
    section_class: int,
) -> BeamSizing:
    """
    Diameter of every member, required by the check and in closed form.
    """
    closed = diameter_closed_form(design.actions.moment_major, catalogue, section_class)
    members = np.arange(design.diameters.shape[0])
    midpoints = 0.5 * (positions[:-1] + positions[1:])
    sizing = BeamSizing(
        members, midpoints, np.asarray(design.diameters), np.asarray(closed)
    )

    return sizing


def run_stagger(
    setup: BeamProblem,
    catalogue: TubeCatalogue,
    section_class: int,
) -> StaggerRun:
    """
    Run analyse-and-size passes, recording diameter moves and mass.
    """
    diameters = setup.seed
    moves = []
    masses = []

    for _ in range(PASSES):
        result = build(setup, diameters, catalogue)
        shift = jnp.abs(result.diameters - diameters) / result.diameters
        moves.append(float(jnp.max(shift)))
        masses.append(float(result.mass))
        diameters = result.diameters

    run = StaggerRun(np.asarray(moves), np.asarray(masses))

    return run


def report_beam(report: Report, setup: BeamProblem) -> None:
    """
    The span, the discretization, the load and the support scheme.
    """
    heights = np.asarray(setup.structure.nodes[:, 2])
    supports = np.asarray(setup.structure.supports).tolist()
    entries = (
        ("span", f"{SPAN / 1e3:.1f} m over {setup.num_edges} members"),
        ("member length", f"{float(setup.lengths[0]):.1f} mm"),
        ("largest z", f"{float(np.max(np.abs(heights))):.1e} mm"),
        ("load case", f"{CASE_NAME}, {TOTAL_LOAD / 1e3:.1f} kN"),
        ("supports", f"pinned at nodes {supports}"),
    )

    report.write_line("The beam, being the arch of experiment 03 laid flat")
    report.write_entries(entries)


def report_design(
    report: Report,
    design: BeamDesign,
    catalogue: TubeCatalogue,
) -> None:
    """
    Every member's actions, size, utilization and governing limit state.
    """
    codes = governing_compiled(
        catalogue(design.diameters),
        design.actions,
        design.lengths,
        catalogue,
    )
    limits = {LIMIT_NAMES[float(code)] for code in codes}

    columns = (
        ReportColumn("member"),
        ReportColumn("N [kN]", ".1e"),
        ReportColumn("M [kNm]", ".4f"),
        ReportColumn("d [mm]", ".5f"),
        ReportColumn("utilization", ".16f"),
    )
    rows = []
    for member in range(design.diameters.shape[0]):
        force = float(design.actions.axial_force[member]) / 1e3
        moment = float(design.actions.moment_major[member]) / 1e6
        diameter = float(design.diameters[member])
        utilization = float(design.utilization[member])
        rows.append((member, force, moment, diameter, utilization))

    entries = (
        ("mass", f"{float(design.mass):.9f} t"),
        ("worst |u - 1|", f"{worst_utilization(design):.2e}"),
        ("governing", ", ".join(sorted(limits))),
    )
    ratio = float(catalogue.ratio)

    report.write_heading(f"Class {catalogue.section_class}, d/t = {ratio:.3f}")
    report.write_table(columns, rows)
    report.write_entries(entries)


def report_statics(report: Report, statics: BeamStatics) -> None:
    """
    The moment the solver reports beside the moment statics predicts.
    """
    scale = float(np.max(np.abs(statics.exact)))
    columns = (
        ReportColumn("node"),
        ReportColumn("x [mm]", ".0f"),
        ReportColumn("M statics [kNm]", ".9f"),
        ReportColumn("M smax [kNm]", ".9f"),
        ReportColumn("scaled", ".2e"),
    )
    rows = []
    for node in range(statics.positions.shape[0]):
        exact = float(statics.exact[node])
        computed = float(statics.computed[node])
        rows.append(
            (
                node,
                float(statics.positions[node]),
                exact / 1e6,
                computed / 1e6,
                abs(computed - exact) / scale,
            )
        )

    report.write_heading("The moment diagram against statics")
    report.write_table(columns, rows)
    report.write_note(
        """
        Scaled by the largest moment rather than by each node's own, the moment
        at a support being zero and its relative error meaningless.
        """
    )


def report_sizing(report: Report, sizing: BeamSizing, statics: BeamStatics) -> None:
    """
    The required diameter beside the cube root that predicts it.
    """
    columns = (
        ReportColumn("member"),
        ReportColumn("d required [mm]", ".9f"),
        ReportColumn("d closed form [mm]", ".9f"),
        ReportColumn("relative", ".2e"),
    )
    rows = []
    for index in range(sizing.members.shape[0]):
        required = float(sizing.required[index])
        hand = float(sizing.closed_form[index])
        rows.append(
            (int(sizing.members[index]), required, hand, abs(required - hand) / hand)
        )

    report.write_heading("The required diameter against its closed-form inverse")
    report.write_table(columns, rows)
    report.write_note(
        """
        With no axial force the check is the bending stress alone, and every
        section modulus is a monomial in the diameter, so the size inverts as a
        cube root.
        """
    )


def report_stagger(report: Report, run: StaggerRun) -> None:
    """
    That repeating the analysis and the check changes nothing here.
    """
    columns = (
        ReportColumn("pass"),
        ReportColumn("relative move", ".3e"),
        ReportColumn("mass [t]", ".9f"),
    )
    rows = [
        (step, float(move), float(mass))
        for step, (move, mass) in enumerate(zip(run.moves, run.masses))
    ]

    report.write_heading("The staggered coupling closes in one pass")
    report.write_table(columns, rows)
    report.write_note(
        """
        A determinate structure carries the same forces whatever its sections
        are, so the diameters the frame was analyzed with never reach the
        diameters the check returns. On the arch the same loop gives away about
        1.2% of the mass on its first pass.
        """
    )


def write_figures(setup: BeamProblem, results: BeamResults) -> None:
    """
    The beam drawn at its sizes, and both stages against their predictions.
    """
    design = results.design
    FIGURES.mkdir(exist_ok=True)

    seed_tubes = CATALOGUE(setup.seed)
    assumed = float(mass_of_tubes(seed_tubes, design.lengths))
    seeded = SizedMembers(setup.seed, assumed)
    sized = SizedMembers(design.diameters, float(design.mass))
    profile = figure_beam_profile(setup.positions, seeded, sized)
    profile.savefig(FIGURES / "11_profile.png", dpi=160, bbox_inches="tight")

    benchmark = figure_benchmark(results.statics, results.sizing)
    benchmark.savefig(FIGURES / "11_benchmark.png", dpi=160, bbox_inches="tight")


def main(verbose: bool = True) -> None:
    """
    Run the beam benchmark, write the report and the figures.
    """
    report = Report(verbose)
    setup = beam_problem()

    report_beam(report, setup)

    # The same seed diameters sized twice, once per class branch.
    designs = {}
    for section_class in CLASSES:
        catalogue = TubeCatalogue.at_class_limit(STEEL, section_class)
        design = build(setup, setup.seed, catalogue)
        report_design(report, design, catalogue)
        designs[section_class] = design

    # What both branches owe: every member sized to satisfy the standard exactly.
    departure = max(worst_utilization(design) for design in designs.values())

    # Class 3 carries the comparisons, being the ratio the tubes are held at.
    design = designs[SECTION_CLASS]

    # A straight beam under vertical load alone carries nothing along its axis.
    axial = float(jnp.max(jnp.abs(design.actions.axial_force)))

    # The moment diagram is statics and the size is a cube root, so both stages
    # are checked against arithmetic rather than against each other.
    statics = beam_statics(setup, CATALOGUE)
    sizing = beam_sizing(design, setup.positions, CATALOGUE, SECTION_CLASS)
    # The sizes fed back as the stiffness they were found with, which changes nothing.
    stagger = run_stagger(setup, CATALOGUE, SECTION_CLASS)
    # One container, so a report and a figure read the same measurements.
    results = BeamResults(design, statics, sizing, stagger)

    report_statics(report, results.statics)
    report_sizing(report, results.sizing, results.statics)
    report_stagger(report, results.stagger)

    write_figures(setup, results)

    # Five claims, each against a number written down rather than measured.
    check_util = ToleranceCheck("Overstressed!", departure, TOLERANCE_UTILIZATION)
    check_axial = ToleranceCheck("Axial force [N]", axial, TOLERANCE_AXIAL)
    check_statics = ToleranceCheck(
        "Moment against statics", results.worst_statics, TOLERANCE_STATICS
    )
    check_closed = ToleranceCheck(
        "Diameter against closed form", results.worst_closed_form, TOLERANCE_CLOSED_FORM
    )
    check_stagger = ToleranceCheck(
        "Stagger move after pass one",
        results.stagger.worst_move_after_first,
        TOLERANCE_STAGGER,
    )
    checks = (check_util, check_axial, check_statics, check_closed, check_stagger)

    report.write_heading("Summary")
    report.write_checks(checks)
    report.write_heading(f"figures written to {FIGURES}")

    # A nan fails every bound already; this says so rather than relying on it.
    is_finite = jnp.all(jnp.isfinite(design.diameters))
    report.write_verdict(checks_passed(checks) and bool(is_finite))


if __name__ == "__main__":
    jnp.set_printoptions(precision=12)
    main()
