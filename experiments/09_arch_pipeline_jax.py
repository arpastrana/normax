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
Force densities to a mass, differentiably, with EN 1993-1-1 in the middle.

All three stages in one process and one gradient: `jax-fdm` finds the shape,
`smax` says what the members carry, and the sizing map returns the diameter at
which the standard is exactly satisfied. `normax.pipeline.mass` is the scalar,
and `jax.grad` of it crosses all three.

Three things are checked, and two more are measured rather than asserted.

    utilization  one to machine precision for every member, both class branches
    gradient     against central differences, at the step where they plateau
    refinement   the mass converges, first order in the number of members
    coupling     the analysis needs sections and the check returns them, so the
                 pass is repeated until the diameters stop moving
    stability    the critical load factor of the finished design, which is what
                 the member-length buckling length assumes rather than proves

**The buckling length is the member's own length, and that is a strong
assumption.** It presumes every node is held in plane by structure outside the
model, so a member can only buckle between its ends. The arch on its own does
not satisfy it: the critical load factor of the fully-stressed design is far
below one, in a mode that sways over the whole span rather than over one member,
and sizing against that mode instead costs several times the mass. Both numbers
are printed, so the assumption sits next to what it is worth.

**Refinement needs the shape held fixed, not the loads.** Leaving the nodal load
and the force density alone changes the arch as the mesh changes, and the mass
then moves for a reason that has nothing to do with discretization. The total
load is fixed here and the force densities are rescaled so the crown rise is
exactly the target — the force density method is linear in the coordinates, so
one trial solve fixes the scale with no formula and no iteration.

**Every call made in a loop is compiled, and the programs are kept between
runs.** A mesh, an edge or a pass is one call of the same program on different
numbers, so tracing once and reusing it is the whole difference between minutes
and seconds here. What compiling changes about the arithmetic is the order XLA
associates it in, which moves the last two or three bits: the sizes and the
critical load factors are unchanged, while the central differences below, being
differences of nearly equal numbers, move in their third significant figure.

Run with `uv run --group pipeline python experiments/09_arch_pipeline_jax.py`.
"""

from collections.abc import Callable
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax_fdm.equilibrium import EquilibriumStructure
from jaxtyping import Array
from jaxtyping import Float

from normax.analysis.smax import Model
from normax.analysis.smax import buckling_modes
from normax.analysis.smax import prepare_model
from normax.ec3.material import SteelGrade
from normax.ec3.section import TubeCatalogue
from normax.ec3.sizing import mass_of_tubes
from normax.ec3.stability import ALPHA_CR_ELASTIC
from normax.formfinding import equilibrium_graph
from normax.formfinding import equilibrium_state
from normax.pipeline import Design
from normax.pipeline import ProblemSetup
from normax.pipeline import Stability
from normax.pipeline import design_members
from normax.pipeline import frame_stability
from normax.pipeline import governing_states
from normax.pipeline import total_mass
from normax.reporting import ColumnSpec
from normax.reporting import ReportWriter
from normax.reporting import ToleranceCheck
from normax.reporting import checks_passed
from normax.structures import Structure
from normax.structures import arch_2d
from normax.visualization import MeshRefinement
from normax.visualization import SizedMembers
from normax.visualization import StaggeredPasses
from normax.visualization import figure_convergence
from normax.visualization import figure_modes
from normax.visualization import figure_sections

# A 10 m arch rising 3 m, carrying 180 kN spread over its free nodes. Units are
# millimeters and newtons.
SPAN = 10_000.0
RISE = 3_000.0
TOTAL_LOAD = 180_000.0
NUM_EDGES = 10

# The arch lies in the XZ plane, so it has no thickness along Y.
NORMAL = 1

# The diameter the frame is analyzed with before the check has spoken. Only the
# stiffness depends on it, and the check overwrites it.
SEED = 100.0

# Meshes for the refinement study, doubling.
MESHES = (5, 10, 20, 40, 80, 160)

# A buckling length independent of the mesh, so refinement measures the
# discretization rather than the slenderness.
BUCKLING_LENGTH = 1_000.0

PASSES = 6

# Modes to report, and the effective length of the arch's own critical mode as a
# fraction of its developed length. Measured, and steady to three figures across
# a 32-fold range of mesh density.
NUM_MODES = 4
GLOBAL_MODE_FACTOR = 0.576

# Relative step at which the central difference plateaus. Smaller is dominated
# by cancellation, larger by truncation, and the experiment prints the sweep.
STEPS = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
STEP = 1e-5

TOLERANCE_UTILIZATION = 1e-9
TOLERANCE_GRADIENT = 5e-8

# Refinement is first order, so successive halving of the change is the claim.
TOLERANCE_ORDER = 0.15

FIGURES = Path(__file__).resolve().parent.parent / "figures"

# Compiled programs outlive the process, so a second run pays for arithmetic
# alone. Every compilation here is well under the one second the persistent
# cache keeps by default, which would otherwise leave all of them out of it.
COMPILATION_CACHE = Path(__file__).resolve().parent.parent / ".jax_cache"
COMPILATION_CACHE.mkdir(exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(COMPILATION_CACHE))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

STEEL = SteelGrade()

# Preparing the analysis model needs a section family to stand up a frame, and
# every property of it is replaced per call, so one seed serves both classes.
CATALOGUE_SEED = TubeCatalogue.at_class_limit(STEEL.f_y, 3)

CLASSES = (2, 3)

LIMIT_NAMES = {
    0.0: "catalogue minimum",
    1.0: "tension",
    2.0: "cross-section",
    3.0: "6.61 major",
    4.0: "6.62 minor",
}

# Every one of these is called in a loop, over a mesh or over an edge, so each is
# traced once and run from its compiled program afterwards. Left eager they cost
# an XLA compilation per primitive per shape, which is most of what this
# experiment used to spend its time on.
design_compiled = eqx.filter_jit(design_members)
state_compiled = eqx.filter_jit(equilibrium_state)
stability_compiled = eqx.filter_jit(frame_stability)
modes_compiled = eqx.filter_jit(buckling_modes)


class ArchSetup(NamedTuple):
    """
    Everything one mesh of the arch needs before a force density is chosen.

    Attributes
    ----------
    structure :
        The structure supplying the connectivity, the supports and the loads.
    graph :
        The form-finding connectivity, from `normax.formfinding`.
    model :
        The prepared analysis model, from `normax.analysis.smax`.
    q :
        Force densities that reach the target rise on this mesh.
    """

    structure: Structure
    graph: EquilibriumStructure
    model: Model
    q: Float[Array, "edges"]

    @property
    def num_edges(self) -> int:
        """
        Number of members in this mesh.
        """
        return int(self.q.shape[0])

    @property
    def seed(self) -> Float[Array, "edges"]:
        """
        The diameter the frame is analyzed with before the check has spoken.
        """
        return jnp.full(self.num_edges, SEED)

    def problem_for(self, catalogue: TubeCatalogue) -> ProblemSetup:
        """
        The prepared problem, on one section family.
        """
        problem = ProblemSetup(self.structure, self.graph, self.model, STEEL, catalogue)

        return problem


class ClassDesign(NamedTuple):
    """
    The design one class branch arrives at, and what it was solved against.

    Attributes
    ----------
    section_class :
        Class the resistances were evaluated on.
    catalogue :
        Tube family whose ratio holds the section at that class limit.
    problem :
        The prepared problem the design came from.
    design :
        Geometry, actions, sizes and mass of the finished design.
    """

    section_class: int
    catalogue: TubeCatalogue
    problem: ProblemSetup
    design: Design

    @property
    def departure(self) -> float:
        """
        Worst departure of a member's utilization from unity.
        """
        return float(jnp.max(jnp.abs(self.design.utilization - 1.0)))


class GradientRow(NamedTuple):
    """
    One force density's derivative, beside a central difference of the same.

    Attributes
    ----------
    edge :
        Index of the edge the force density belongs to.
    exact :
        Derivative from tracing all three stages together.
    numeric :
        Central difference of the same composed objective.
    difference :
        Their difference, measured against the size of the whole gradient.
    """

    edge: int
    exact: float
    numeric: float
    difference: float


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
    def cost(self) -> float:
        """
        Fraction of the mass the one-shot pass of the objective gives away.
        """
        return abs(self.masses[0] - self.masses[-1]) / self.masses[-1]


class RefinementRow(NamedTuple):
    """
    The mass one mesh arrives at, on either basis for the buckling length.

    Attributes
    ----------
    count :
        Number of members in the mesh.
    arc :
        Developed length of the arch on that mesh.
    mass_member :
        Mass with each member buckling over its own length.
    mass_fixed :
        Mass with a buckling length held independent of the mesh.
    """

    count: int
    arc: float
    mass_member: float
    mass_fixed: float


class RefinementStudy(NamedTuple):
    """
    How the mass settles as the mesh is refined.

    Attributes
    ----------
    rows :
        One entry per mesh, in the order they were refined.
    limit :
        Mass the mesh-independent sequence extrapolates to.
    ratios :
        Ratio of successive changes, which is two for a first-order sequence.
    """

    rows: tuple[RefinementRow, ...]
    limit: float
    ratios: Float[np.ndarray, "meshes"]

    @property
    def worst_order(self) -> float:
        """
        Worst relative departure of those ratios from first order.
        """
        return float(np.max(np.abs(self.ratios - 2.0)) / 2.0)

    @property
    def by_member(self) -> Float[np.ndarray, "meshes"]:
        """
        Mass on each mesh with the member-length buckling length.
        """
        return np.asarray([row.mass_member for row in self.rows])

    @property
    def by_fixed(self) -> Float[np.ndarray, "meshes"]:
        """
        Mass on each mesh with the buckling length held fixed.
        """
        return np.asarray([row.mass_fixed for row in self.rows])


class ArchResults(NamedTuple):
    """
    Everything measured about one design that a figure or a summary reads.

    Attributes
    ----------
    refinement :
        How the mass settles as the mesh is refined.
    stagger :
        What repeating the analysis and the check does to the sizes.
    stability :
        The frame's own critical load factors and slendernesses.
    """

    refinement: RefinementStudy
    stagger: StaggerRun
    stability: Stability


def arch_setup(num_edges: int) -> ArchSetup:
    """
    The arch, its form-finding connectivity, and the `q` that reaches the rise.
    """
    load = TOTAL_LOAD / (num_edges - 1)
    structure = arch_2d(num_edges=num_edges, span=SPAN, rise=RISE, load=load)
    graph = equilibrium_graph(structure)
    model = prepare_model(structure, STEEL, CATALOGUE_SEED, normal=NORMAL)

    trial = jnp.full(num_edges, -1.0)
    state = state_compiled(trial, structure, graph)
    reached = jnp.max(state.xyz[:, 2])
    q = trial * reached / RISE
    setup = ArchSetup(structure, graph, model, q)

    return setup


def central_difference(
    function: Callable[[Float[Array, "edges"]], Float[Array, ""]],
    x: Float[Array, "edges"],
    index: int,
    step: float,
) -> float:
    """
    Central difference of a scalar function in one entry of its argument.
    """
    forward = function(x.at[index].add(step))
    backward = function(x.at[index].add(-step))

    return float((forward - backward) / (2.0 * step))


def disagreement(exact: float, numeric: float, scale: float) -> float:
    """
    Gradient error, measured against the size of the whole gradient.

    Two of this arch's ten sensitivities are twenty times smaller than the rest,
    where they change sign. Dividing by the component would report an absolute
    difference of 1e-13 as an error of 1e-7 and say nothing about the derivative,
    so the largest component sets the scale instead.
    """
    return abs(exact - numeric) / scale


def design_for(setup: ArchSetup, section_class: int) -> ClassDesign:
    """
    The fully-stressed design on one class branch.
    """
    catalogue = TubeCatalogue.at_class_limit(STEEL.f_y, section_class)
    problem = setup.problem_for(catalogue)
    design = design_compiled(setup.q, setup.seed, problem, section_class=section_class)
    branch = ClassDesign(section_class, catalogue, problem, design)

    return branch


def stagger_run(setup: ArchSetup, branch: ClassDesign, passes: int) -> StaggerRun:
    """
    Repeat the staggered analysis and check, reporting how far each pass moves.
    """
    diameters = setup.seed
    moves = []
    masses = []

    for _ in range(passes):
        result = design_compiled(
            setup.q,
            diameters,
            branch.problem,
            section_class=branch.section_class,
        )
        shift = jnp.abs(result.diameters - diameters) / result.diameters
        moves.append(float(jnp.max(shift)))
        masses.append(float(result.mass))
        diameters = result.diameters

    run = StaggerRun(np.asarray(moves), np.asarray(masses))

    return run


def refinement_study(catalogue: TubeCatalogue, section_class: int) -> RefinementStudy:
    """
    The mass on every mesh, and the order the sequence converges at.
    """
    rows = []
    for count in MESHES:
        refined = arch_setup(count)
        problem = refined.problem_for(catalogue)
        free = design_compiled(
            refined.q, refined.seed, problem, section_class=section_class
        )
        fixed_length = jnp.full(count, BUCKLING_LENGTH)
        held = design_compiled(
            refined.q,
            refined.seed,
            problem,
            section_class=section_class,
            buckling_length=fixed_length,
        )
        arc = float(jnp.sum(free.lengths))
        row = RefinementRow(count, arc, float(free.mass), float(held.mass))
        rows.append(row)

    by_fixed = np.asarray([row.mass_fixed for row in rows])

    # Richardson, for a sequence converging first order in the member count.
    limit = 2.0 * by_fixed[-1] - by_fixed[-2]
    changes = np.abs(np.diff(by_fixed)) / np.abs(by_fixed[1:])
    ratios = changes[:-1] / changes[1:]
    study = RefinementStudy(tuple(rows), float(limit), ratios)

    return study


def report_arch(report: ReportWriter, setup: ArchSetup) -> None:
    """
    The shape every number below belongs to.
    """
    state = state_compiled(setup.q, setup.structure, setup.graph)
    rise = float(jnp.max(state.xyz[:, 2]))
    entries = (
        ("span", f"{SPAN / 1e3:.1f} m over {setup.num_edges} members"),
        ("crown rise", f"{rise:.4f} mm"),
        ("force density", f"{float(setup.q[0]):.6f} N/mm"),
        ("total load", f"{TOTAL_LOAD / 1e3:.1f} kN"),
    )

    report.write_line("The arch")
    report.write_entries(entries)


def report_design(report: ReportWriter, branch: ClassDesign) -> None:
    """
    Every member of one class branch, and what the standard decided about it.
    """
    design = branch.design
    codes = governing_states(design, branch.problem, section_class=branch.section_class)
    limits = {LIMIT_NAMES[float(code)] for code in codes}

    columns = (
        ColumnSpec("member"),
        ColumnSpec("N [kN]", ".4f"),
        ColumnSpec("M [kNm]", ".5f"),
        ColumnSpec("d [mm]", ".4f"),
        ColumnSpec("utilization", ".16f"),
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
        ("worst |u - 1|", f"{branch.departure:.2e}"),
        ("governing", ", ".join(sorted(limits))),
    )
    ratio = float(branch.catalogue.ratio)

    report.write_heading(f"Class {branch.section_class}, d/t = {ratio:.3f}")
    report.write_table(columns, rows)
    report.write_entries(entries)


def report_steps(
    report: ReportWriter,
    objective: Callable[[Float[Array, "edges"]], Float[Array, ""]],
    setup: ArchSetup,
    gradient: Float[Array, "edges"],
) -> None:
    """
    That the central difference plateaus before it is trusted.
    """
    scale = float(jnp.max(jnp.abs(gradient)))
    edges = (0, setup.num_edges // 2, setup.num_edges - 1)
    rows = []
    for relative in STEPS:
        worst = 0.0
        for edge in edges:
            step = abs(float(setup.q[edge])) * relative
            numeric = central_difference(objective, setup.q, edge, step)
            worst = max(worst, disagreement(float(gradient[edge]), numeric, scale))
        rows.append((relative, worst))

    columns = (
        ColumnSpec("relative step", ".0e"),
        ColumnSpec("worst scaled error", ".3e"),
    )

    report.write_heading("The central difference plateaus before it is trusted")
    report.write_table(columns, rows)


def report_gradient(report: ReportWriter, rows: Sequence[GradientRow]) -> float:
    """
    The gradient of the mass, and the worst scaled error in it.
    """
    columns = (
        ColumnSpec("edge"),
        ColumnSpec("autodiff", ".14e"),
        ColumnSpec("central", ".14e"),
        ColumnSpec("scaled", ".2e"),
    )
    printed = [(row.edge, row.exact, row.numeric, row.difference) for row in rows]

    report.write_heading(f"The gradient of the mass, at a relative step of {STEP:.0e}")
    report.write_table(columns, printed)

    return max(row.difference for row in rows)


def report_stagger(report: ReportWriter, run: StaggerRun) -> None:
    """
    That repeating the analysis and the check closes geometrically.
    """
    columns = (
        ColumnSpec("pass"),
        ColumnSpec("relative move", ".3e"),
        ColumnSpec("mass [t]", ".9f"),
        ColumnSpec("ratio", ".4f"),
    )
    rows = []
    for step, (move, mass) in enumerate(zip(run.moves, run.masses)):
        ratio = "" if step == 0 else float(run.moves[step] / run.moves[step - 1])
        rows.append((step, float(move), float(mass), ratio))

    entries = (("one pass costs", f"{run.cost:.3%} of the mass"),)

    report.write_heading("The staggered coupling closes geometrically")
    report.write_table(columns, rows)
    report.write_entries(entries)


def report_refinement(report: ReportWriter, study: RefinementStudy) -> None:
    """
    That the mass converges, first order in the number of members.
    """
    columns = (
        ColumnSpec("members"),
        ColumnSpec("arc [mm]", ".4f"),
        ColumnSpec("mass, Lcr=member", ".9f"),
        ColumnSpec("mass, Lcr fixed", ".9f"),
    )
    rows = [(row.count, row.arc, row.mass_member, row.mass_fixed) for row in study.rows]
    ratios = np.array2string(study.ratios, precision=3)
    entries = (
        ("extrapolated limit", f"{study.limit:.9f} t"),
        ("change ratios", ratios),
        ("worst departure from first order", f"{study.worst_order:.3f}"),
    )

    report.write_heading("The mass converges as the mesh refines")
    report.write_table(columns, rows)
    report.write_entries(entries)


def report_stability(report: ReportWriter, branch: ClassDesign, checked: Stability):
    """
    The global stability check, and both routes to the same slenderness.
    """
    verdict = "SATISFIED" if bool(checked.adequate) else "NOT SATISFIED"
    entries = (
        ("alpha_cr", f"{float(checked.factors[0]):.4f}"),
        ("threshold", f"{ALPHA_CR_ELASTIC:.1f}"),
        ("utilization", f"{float(checked.utilization):.3f}"),
        ("verdict", verdict),
    )

    report.write_heading("The global stability check, EN 1993-1-1 5.2.1(3)")
    report.write_entries(entries)

    columns = (
        ColumnSpec("member"),
        ColumnSpec("6.50 from L_cr", ".4f"),
        ColumnSpec("6.3.4 from a_cr", ".4f"),
        ColumnSpec("ratio", ".2f"),
        ColumnSpec("L_cr,global [mm]", ".1f"),
    )
    rows = []
    for member in range(branch.design.diameters.shape[0]):
        from_length = float(checked.slenderness_member[member])
        from_factor = float(checked.slenderness_global[member])
        equivalent = float(checked.buckling_length_equivalent[member])
        rows.append(
            (member, from_length, from_factor, from_factor / from_length, equivalent)
        )

    report.write_heading("Both of the standard's routes to the same slenderness")
    report.write_table(columns, rows)


def report_assumption(
    report: ReportWriter,
    setup: ArchSetup,
    branch: ClassDesign,
    checked: Stability,
) -> None:
    """
    What sizing against the frame's own mode would have cost instead.
    """
    arc = float(jnp.sum(branch.design.lengths))
    global_length = jnp.full(setup.num_edges, GLOBAL_MODE_FACTOR * arc)
    unbraced = design_compiled(
        setup.q,
        setup.seed,
        branch.problem,
        section_class=branch.section_class,
        buckling_length=global_length,
    )
    penalty = float(unbraced.mass) / float(branch.design.mass)

    factors = np.array2string(np.asarray(checked.factors), precision=4)
    against_global = f"sized against L_cr = {GLOBAL_MODE_FACTOR:.3f} arc"
    entries = (
        ("critical load factors", factors),
        ("arc length", f"{arc:.1f} mm"),
        ("sized against L_cr = L", f"{float(branch.design.mass):.6f} t"),
        (against_global, f"{float(unbraced.mass):.6f} t, x{penalty:.2f}"),
    )

    report.write_heading("What the member-length assumption is worth")
    report.write_entries(entries)
    report.write_note(
        """
        The member-length basis presumes every node is held in plane by structure
        outside the model. The bare model does not satisfy 5.2.1, which is what
        makes that assumption load-bearing.
        """
    )


def write_figures(
    report: ReportWriter,
    setup: ArchSetup,
    branch: ClassDesign,
    results: ArchResults,
) -> None:
    """
    Every figure this experiment is the source of.
    """
    design = branch.design
    study = results.refinement
    FIGURES.mkdir(exist_ok=True)

    seed_tubes = branch.catalogue.tube_at(setup.seed)
    assumed = float(mass_of_tubes(seed_tubes, design.lengths, STEEL))
    seeded = SizedMembers(setup.seed, assumed)
    sized = SizedMembers(design.diameters, float(design.mass))
    sections = figure_sections(design.xyz, setup.structure.edges, seeded, sized)
    sections.savefig(FIGURES / "09_sections.png", dpi=160, bbox_inches="tight")

    counts = np.asarray(MESHES)
    refinement = MeshRefinement(counts, study.by_member, study.by_fixed, study.limit)
    passes = np.arange(len(results.stagger.moves))
    staggered = StaggeredPasses(passes, results.stagger.moves)
    convergence = figure_convergence(refinement, staggered)
    convergence.savefig(FIGURES / "09_convergence.png", dpi=160, bbox_inches="tight")

    modes = modes_compiled(
        setup.model,
        design.xyz,
        design.diameters,
        STEEL,
        branch.catalogue,
        num_modes=NUM_MODES,
    )
    factors = np.asarray(results.stability.factors)
    shapes = figure_modes(design.xyz, factors, np.asarray(modes.shapes), RISE)
    shapes.savefig(FIGURES / "09_modes.png", dpi=160, bbox_inches="tight")
    report.write_heading(f"figures written to {FIGURES}")


def main(verbose: bool = True) -> None:
    """
    Run the whole pipeline on one arch, and check what it returns.
    """
    report = ReportWriter(verbose)
    setup = arch_setup(NUM_EDGES)

    report_arch(report, setup)

    branches = [design_for(setup, section_class) for section_class in CLASSES]
    for branch in branches:
        report_design(report, branch)
    worst_utilization = max(branch.departure for branch in branches)

    elastic = branches[-1]

    def objective(q):
        return total_mass(
            q, setup.seed, elastic.problem, section_class=elastic.section_class
        )

    value_of = eqx.filter_jit(objective)
    gradient_of = eqx.filter_jit(jax.grad(objective))

    gradient = gradient_of(setup.q)
    scale = float(jnp.max(jnp.abs(gradient)))

    report_steps(report, value_of, setup, gradient)

    rows = []
    for edge in range(setup.num_edges):
        step = abs(float(setup.q[edge])) * STEP
        numeric = central_difference(value_of, setup.q, edge, step)
        exact = float(gradient[edge])
        difference = disagreement(exact, numeric, scale)
        rows.append(GradientRow(edge, exact, numeric, difference))

    worst_gradient = report_gradient(report, rows)

    refinement = refinement_study(elastic.catalogue, elastic.section_class)
    stagger = stagger_run(setup, elastic, PASSES)
    stability = stability_compiled(elastic.design, elastic.problem, num_modes=NUM_MODES)
    results = ArchResults(refinement, stagger, stability)

    report_stagger(report, results.stagger)
    report_refinement(report, results.refinement)
    report_stability(report, elastic, results.stability)
    report_assumption(report, setup, elastic, results.stability)

    write_figures(report, setup, elastic, results)

    checks = (
        ToleranceCheck(
            "departure from unity", worst_utilization, TOLERANCE_UTILIZATION
        ),
        ToleranceCheck("gradient error", worst_gradient, TOLERANCE_GRADIENT),
        ToleranceCheck(
            "departure from first order",
            results.refinement.worst_order,
            TOLERANCE_ORDER,
        ),
    )
    report.write_heading("Summary")
    report.write_checks(checks)
    report.write_verdict(
        checks_passed(checks) and bool(jnp.all(jnp.isfinite(gradient)))
    )


if __name__ == "__main__":
    jnp.set_printoptions(precision=12)
    main()
