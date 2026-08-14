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
which the standard is exactly satisfied. `normax.design.compute_mass` is the scalar,
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

from normax.analysis.smax import SmaxAnalyzer
from normax.analysis.smax import Stability
from normax.analysis.smax import buckling_modes
from normax.analysis.smax import frame_stability
from normax.design import DesignParameters
from normax.design import DesignPipeline
from normax.design import LoadCases
from normax.design import MemberSections
from normax.design import calculate_mass
from normax.design import load_cases
from normax.ec3.material import Steel
from normax.ec3.section import TubeCatalogue
from normax.ec3.sizing import mass_of_tubes
from normax.ec3.stability import ALPHA_CR_ELASTIC
from normax.form_finding.fdm import FdmFormFinder
from normax.form_finding.fdm import equilibrium_graph
from normax.form_finding.fdm import equilibrium_state
from normax.reporting import Report
from normax.reporting import ReportColumn
from normax.reporting import ToleranceCheck
from normax.reporting import checks_passed
from normax.sizing import Ec3Sizer
from normax.structures import Structure
from normax.structures import arch_2d
from normax.structures import loads_uniform
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

STEEL = Steel()

# Preparing the analysis model needs a section family to stand up a frame, and
# every property of it is replaced per call, so one seed serves both classes.
CATALOGUE_SEED = TubeCatalogue.at_class_limit(STEEL, 3)

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
equilibrium_state_compiled = eqx.filter_jit(equilibrium_state)
frame_stability_compiled = eqx.filter_jit(frame_stability)
buckling_modes_compiled = eqx.filter_jit(buckling_modes)


class ArchSetup(NamedTuple):
    """
    Mesh, form-finding graph, analysis model, and rise-reaching force densities.
    """

    structure: Structure
    graph: EquilibriumStructure
    analyzer: SmaxAnalyzer
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
        Seed diameter used for the first analysis before sizing.
        """
        return jnp.full(self.num_edges, SEED)

    @property
    def funicular(self) -> Float[Array, "nodes 3"]:
        """
        The uniform load case the arch is form-found under.
        """
        return loads_uniform(self.structure, TOTAL_LOAD / (self.num_edges - 1))

    @property
    def loads(self) -> LoadCases:
        """
        The one load case the arch is shaped by and checked against.
        """
        applied = self.funicular

        return load_cases(applied, [applied])

    @property
    def params(self) -> DesignParameters:
        """
        The force densities and the seed diameters, as a pipeline takes them.
        """
        return DesignParameters(self.q, self.seed)


class GradientRow(NamedTuple):
    """
    One edge's autodiff derivative beside its central difference.
    """

    edge: int
    exact: float
    numeric: float
    difference: float


class StaggerRun(NamedTuple):
    """
    Per-pass diameter moves and masses from staggered analyse-size.
    """

    moves: Float[np.ndarray, "passes"]
    masses: Float[np.ndarray, "passes"]

    @property
    def one_pass_cost(self) -> float:
        """
        Relative mass given away by stopping after the first pass.
        """
        return abs(self.masses[0] - self.masses[-1]) / self.masses[-1]


class RefinementRow(NamedTuple):
    """
    Mass on one mesh for member-length and fixed buckling lengths.
    """

    count: int
    arc: float
    mass_member: float
    mass_fixed: float


class RefinementStudy(NamedTuple):
    """
    Masses across meshes, Richardson limit, and change ratios.
    """

    rows: tuple[RefinementRow, ...]
    limit: float
    ratios: Float[np.ndarray, "meshes"]

    @property
    def worst_order(self) -> float:
        """
        Worst relative departure of change ratios from first order.
        """
        return float(np.max(np.abs(self.ratios - 2.0)) / 2.0)

    @property
    def masses_member(self) -> Float[np.ndarray, "meshes"]:
        """
        Mass on each mesh with member-length buckling.
        """
        return np.asarray([row.mass_member for row in self.rows])

    @property
    def masses_fixed(self) -> Float[np.ndarray, "meshes"]:
        """
        Mass on each mesh with a mesh-independent buckling length.
        """
        return np.asarray([row.mass_fixed for row in self.rows])


class ArchResults(NamedTuple):
    """
    Refinement, stagger, and stability results for one design.
    """

    refinement: RefinementStudy
    stagger: StaggerRun
    stability: Stability


def arch_setup(num_edges: int) -> ArchSetup:
    """
    Build the arch mesh, topologies, and rise-reaching force densities.
    """
    structure = arch_2d(num_edges=num_edges, span=SPAN, rise=RISE)
    graph = equilibrium_graph(structure)
    analyzer = SmaxAnalyzer(STEEL, CATALOGUE_SEED, NORMAL).compile(structure)
    applied = loads_uniform(structure, TOTAL_LOAD / (num_edges - 1))

    trial = jnp.full(num_edges, -1.0)
    state = equilibrium_state_compiled(trial, structure, graph, applied)
    reached = jnp.max(state.xyz[:, 2])
    q = trial * reached / RISE
    setup = ArchSetup(structure, graph, analyzer, q)

    return setup


def pipeline_from_setup(setup: ArchSetup, catalogue: TubeCatalogue) -> DesignPipeline:
    """
    Bind steel and a catalogue into the three blocks for this mesh.
    """
    blocks = DesignPipeline(
        FdmFormFinder(),
        SmaxAnalyzer(STEEL, catalogue, NORMAL),
        Ec3Sizer(STEEL, catalogue),
    )

    return blocks.compile(setup.structure)


def worst_utilization(design: MemberSections) -> float:
    """
    Largest absolute departure of utilization from unity.
    """
    return float(jnp.max(jnp.abs(design.utilization - 1.0)))


def central_difference(
    function: Callable[[Float[Array, "edges"]], Float[Array, ""]],
    x: Float[Array, "edges"],
    index: int,
    step: float,
) -> float:
    """
    Central difference of a scalar in one entry of its argument.
    """
    forward = function(x.at[index].add(step))
    backward = function(x.at[index].add(-step))

    return float((forward - backward) / (2.0 * step))


def scaled_gradient_error(exact: float, numeric: float, scale: float) -> float:
    """
    Absolute gradient error scaled by the largest component.
    """
    return abs(exact - numeric) / scale


def gradient_rows(
    objective: Callable[[Float[Array, "edges"]], Float[Array, ""]],
    setup: ArchSetup,
    gradient: Float[Array, "edges"],
    scale: float,
) -> list[GradientRow]:
    """
    Autodiff versus central difference of mass, one row per edge.
    """
    rows = []
    for edge in range(setup.num_edges):
        step = abs(float(setup.q[edge])) * STEP
        numeric = central_difference(objective, setup.q, edge, step)
        exact = float(gradient[edge])
        difference = scaled_gradient_error(exact, numeric, scale)
        rows.append(GradientRow(edge, exact, numeric, difference))

    return rows


def design_at_class(
    setup: ArchSetup, section_class: int
) -> tuple[MemberSections, DesignPipeline]:
    """
    Fully-stressed design of the arch on one section class.
    """
    catalogue = TubeCatalogue.at_class_limit(STEEL, section_class)
    pipeline = pipeline_from_setup(setup, catalogue)
    design = eqx.filter_jit(pipeline)(setup.params, setup.loads)

    return design, pipeline


def run_stagger(
    setup: ArchSetup,
    pipeline: DesignPipeline,
    passes: int,
) -> StaggerRun:
    """
    Run analyse-and-size passes, recording diameter moves and mass.
    """
    diameters = setup.seed
    moves = []
    masses = []

    for _ in range(passes):
        result = eqx.filter_jit(pipeline)(
            DesignParameters(setup.q, diameters), setup.loads
        )
        shift = jnp.abs(result.diameters - diameters) / result.diameters
        moves.append(float(jnp.max(shift)))
        masses.append(float(calculate_mass(result)))
        diameters = result.diameters

    run = StaggerRun(np.asarray(moves), np.asarray(masses))

    return run


def refinement_study(catalogue: TubeCatalogue, section_class: int) -> RefinementStudy:
    """
    Mass on each mesh and the first-order convergence of that sequence.
    """
    rows = []
    for count in MESHES:
        refined = arch_setup(count)
        pipeline = eqx.filter_jit(pipeline_from_setup(refined, catalogue))
        free = pipeline(refined.params, refined.loads)
        fixed_length = jnp.full(count, BUCKLING_LENGTH)
        held = pipeline(refined.params, refined.loads, buckling_length=fixed_length)
        arc = float(jnp.sum(free.lengths))
        row = RefinementRow(
            count, arc, float(calculate_mass(free)), float(calculate_mass(held))
        )
        rows.append(row)

    masses_fixed = np.asarray([row.mass_fixed for row in rows])

    # Richardson, for a sequence converging first order in the member count.
    limit = 2.0 * masses_fixed[-1] - masses_fixed[-2]
    changes = np.abs(np.diff(masses_fixed)) / np.abs(masses_fixed[1:])
    ratios = changes[:-1] / changes[1:]
    study = RefinementStudy(tuple(rows), float(limit), ratios)

    return study


def report_arch(report: Report, setup: ArchSetup) -> None:
    """
    Write the arch span, rise, force density, and total load.
    """
    state = equilibrium_state_compiled(
        setup.q, setup.structure, setup.graph, setup.funicular
    )
    rise = float(jnp.max(state.xyz[:, 2]))
    entries = (
        ("span", f"{SPAN / 1e3:.1f} m over {setup.num_edges} members"),
        ("crown rise", f"{rise:.4f} mm"),
        ("force density", f"{float(setup.q[0]):.6f} N/mm"),
        ("total load", f"{TOTAL_LOAD / 1e3:.1f} kN"),
    )

    report.write_line("The arch")
    report.write_entries(entries)


def report_design(
    report: Report, design: MemberSections, pipeline: DesignPipeline
) -> None:
    """
    Write each member's actions, size, utilization, and governing limit.
    """
    codes = pipeline.sizer.governing(
        design.diameters, design.actions, design.buckling_length
    )[0]
    limits = {LIMIT_NAMES[float(code)] for code in codes}

    columns = (
        ReportColumn("member"),
        ReportColumn("N [kN]", ".4f"),
        ReportColumn("M [kNm]", ".5f"),
        ReportColumn("d [mm]", ".4f"),
        ReportColumn("utilization", ".16f"),
    )
    rows = []
    for member in range(design.diameters.shape[0]):
        force = float(design.actions.axial_force[0, member]) / 1e3
        moment = float(design.actions.moment_major[0, member]) / 1e6
        diameter = float(design.diameters[member])
        utilization = float(design.utilization[0, member])
        rows.append((member, force, moment, diameter, utilization))

    entries = (
        ("mass", f"{float(calculate_mass(design)):.9f} t"),
        ("worst |u - 1|", f"{worst_utilization(design):.2e}"),
        ("governing", ", ".join(sorted(limits))),
    )
    ratio = float(pipeline.sizer.catalogue.ratio)

    section_class = pipeline.sizer.section_class
    report.write_heading(f"Class {section_class}, d/t = {ratio:.3f}")
    report.write_table(columns, rows)
    report.write_entries(entries)


def compare_gradient_numerical(
    report: Report,
    objective: Callable[[Float[Array, "edges"]], Float[Array, ""]],
    setup: ArchSetup,
    gradient: Float[Array, "edges"],
) -> None:
    """
    Write central-difference error versus relative step on three edges.
    """
    scale = float(jnp.max(jnp.abs(gradient)))
    edges = (0, setup.num_edges // 2, setup.num_edges - 1)
    rows = []
    for relative in STEPS:
        worst = 0.0
        for edge in edges:
            step = abs(float(setup.q[edge])) * relative
            numeric = central_difference(objective, setup.q, edge, step)
            worst = max(
                worst, scaled_gradient_error(float(gradient[edge]), numeric, scale)
            )
        rows.append((relative, worst))

    columns = (
        ReportColumn("relative step", ".0e"),
        ReportColumn("worst scaled error", ".3e"),
    )

    report.write_heading("The central difference plateaus before it is trusted")
    report.write_table(columns, rows)


def compare_gradient_autodiff(report: Report, rows: Sequence[GradientRow]) -> float:
    """
    Write autodiff versus central difference per edge; return worst error.
    """
    columns = (
        ReportColumn("edge"),
        ReportColumn("autodiff", ".14e"),
        ReportColumn("central", ".14e"),
        ReportColumn("scaled", ".2e"),
    )
    printed = [(row.edge, row.exact, row.numeric, row.difference) for row in rows]

    report.write_heading(f"The gradient of the mass, at a relative step of {STEP:.0e}")
    report.write_table(columns, printed)

    return max(row.difference for row in rows)


def compare_stagger_closure(report: Report, run: StaggerRun) -> None:
    """
    Write stagger-pass moves, masses, and the one-pass mass cost.
    """
    columns = (
        ReportColumn("pass"),
        ReportColumn("relative move", ".3e"),
        ReportColumn("mass [t]", ".9f"),
        ReportColumn("ratio", ".4f"),
    )
    rows = []
    for step, (move, mass) in enumerate(zip(run.moves, run.masses)):
        ratio = "" if step == 0 else float(run.moves[step] / run.moves[step - 1])
        rows.append((step, float(move), float(mass), ratio))

    entries = (("one pass costs", f"{run.one_pass_cost:.3%} of the mass"),)

    report.write_heading("The staggered coupling closes geometrically")
    report.write_table(columns, rows)
    report.write_entries(entries)


def compare_mesh_refinement(report: Report, study: RefinementStudy) -> None:
    """
    Write mesh-refinement masses, limit, and first-order change ratios.
    """
    columns = (
        ReportColumn("members"),
        ReportColumn("arc [mm]", ".4f"),
        ReportColumn("mass, Lcr=member", ".9f"),
        ReportColumn("mass, Lcr fixed", ".9f"),
    )
    rows = [(row.count, row.arc, row.mass_member, row.mass_fixed) for row in study.rows]
    ratios = np.array2string(study.ratios, precision=3)
    entries = (
        ("extrapolated limit", f"{study.limit:.9f} t"),
        ("change ratios", ratios),
        ("worst oversizing from first order", f"{study.worst_order:.3f}"),
    )

    report.write_heading("The mass converges as the mesh refines")
    report.write_table(columns, rows)
    report.write_entries(entries)


def check_frame_stability(
    report: Report, design: MemberSections, checked: Stability
) -> None:
    """
    Write the frame stability verdict and both slenderness routes.
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
        ReportColumn("member"),
        ReportColumn("6.50 from L_cr", ".4f"),
        ReportColumn("6.3.4 from a_cr", ".4f"),
        ReportColumn("ratio", ".2f"),
        ReportColumn("L_cr,global [mm]", ".1f"),
    )
    rows = []
    for member in range(design.diameters.shape[0]):
        from_length = float(checked.slenderness_member[member])
        from_factor = float(checked.slenderness_global[member])
        equivalent = float(checked.buckling_length_equivalent[member])
        rows.append(
            (member, from_length, from_factor, from_factor / from_length, equivalent)
        )

    report.write_heading("Both of the standard's routes to the same slenderness")
    report.write_table(columns, rows)


def compare_buckling_length_basis(
    report: Report,
    setup: ArchSetup,
    design: MemberSections,
    pipeline: DesignPipeline,
    checked: Stability,
) -> None:
    """
    Compare mass under member-length L_cr versus the frame's own mode.
    """
    arc = float(jnp.sum(design.lengths))
    global_length = jnp.full(setup.num_edges, GLOBAL_MODE_FACTOR * arc)
    unbraced = eqx.filter_jit(pipeline)(
        setup.params, setup.loads, buckling_length=global_length
    )
    penalty = float(calculate_mass(unbraced)) / float(calculate_mass(design))

    factors = np.array2string(np.asarray(checked.factors), precision=4)
    against_global = f"sized against L_cr = {GLOBAL_MODE_FACTOR:.3f} arc"
    entries = (
        ("critical load factors", factors),
        ("arc length", f"{arc:.1f} mm"),
        ("sized against L_cr = L", f"{float(calculate_mass(design)):.6f} t"),
        (against_global, f"{float(calculate_mass(unbraced)):.6f} t, x{penalty:.2f}"),
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


def generate_figures(
    report: Report,
    setup: ArchSetup,
    design: MemberSections,
    pipeline: DesignPipeline,
    results: ArchResults,
) -> None:
    """
    Write the sections, convergence, and buckling-mode figures.
    """
    study = results.refinement
    FIGURES.mkdir(exist_ok=True)

    seed_tubes = pipeline.sizer.catalogue(setup.seed)
    assumed = float(mass_of_tubes(seed_tubes, design.lengths))
    seeded = SizedMembers(setup.seed, assumed)
    sized = SizedMembers(design.diameters, float(calculate_mass(design)))
    sections = figure_sections(design.xyz, setup.structure.edges, seeded, sized)
    sections.savefig(FIGURES / "09_sections.png", dpi=160, bbox_inches="tight")

    counts = np.asarray(MESHES)
    refinement = MeshRefinement(
        counts, study.masses_member, study.masses_fixed, study.limit
    )
    passes = np.arange(len(results.stagger.moves))
    staggered = StaggeredPasses(passes, results.stagger.moves)
    convergence = figure_convergence(refinement, staggered)
    convergence.savefig(FIGURES / "09_convergence.png", dpi=160, bbox_inches="tight")

    modes = buckling_modes_compiled(
        setup.analyzer.model,
        design.xyz,
        design.diameters,
        STEEL,
        pipeline.sizer.catalogue,
        setup.funicular,
        num_modes=NUM_MODES,
    )
    factors = np.asarray(results.stability.factors)
    shapes = figure_modes(design.xyz, factors, np.asarray(modes.shapes), RISE)
    shapes.savefig(FIGURES / "09_modes.png", dpi=160, bbox_inches="tight")
    report.write_heading(f"figures written to {FIGURES}")


def main(verbose: bool = True) -> None:
    """
    Run the arch pipeline checks and write the report and figures.
    """
    report = Report(verbose)
    # The arch, its form-finding graph, its frame model, and the q reaching the rise.
    setup = arch_setup(NUM_EDGES)

    report_arch(report, setup)

    # The same q and seed diameters sized twice, once per class branch.
    designs = {}
    pipelines = {}
    for section_class in CLASSES:
        design, pipeline = design_at_class(setup, section_class)
        report_design(report, design, pipeline)
        designs[section_class] = design
        pipelines[section_class] = pipeline

    # What both branches owe: every member sized to satisfy the standard exactly.
    departure = max(worst_utilization(design) for design in designs.values())

    # Class 3 carries everything below, being the ratio the tubes are held at.
    section_class = 3
    design = designs[section_class]
    pipeline = pipelines[section_class]

    # All three stages as one scalar function: force densities in, tonnes out.
    def objective(q):
        return calculate_mass(pipeline(DesignParameters(q, setup.seed), setup.loads))

    # Traced once each, then called fifty-one times by the two sweeps below.
    objective_value = eqx.filter_jit(objective)
    objective_gradient = eqx.filter_jit(jax.grad(objective))

    # One reverse pass back through the check, the analysis and the form finding.
    gradient = objective_gradient(setup.q)
    # The largest component, since two of these sensitivities change sign.
    scale = float(jnp.max(jnp.abs(gradient)))

    # Three edges over five step sizes, to find where the difference plateaus.
    compare_gradient_numerical(report, objective_value, setup, gradient)

    # Every edge at that step, each derivative beside a difference of two masses.
    rows = gradient_rows(objective_value, setup, gradient, scale)
    worst_gradient = compare_gradient_autodiff(report, rows)

    # The same design on six meshes, doubling, so the mass can be seen to settle.
    refinement = refinement_study(pipeline.sizer.catalogue, section_class)
    # The sizes fed back as the stiffness they were found with, until they hold.
    stagger = run_stagger(setup, pipeline, PASSES)
    # What the finished frame does as a whole, which L_cr = L assumed rather than proved
    stability = frame_stability_compiled(
        design, setup.analyzer, setup.funicular, num_modes=NUM_MODES
    )
    # One container, so a report and a figure read the same measurements.
    results = ArchResults(refinement, stagger, stability)

    compare_stagger_closure(report, results.stagger)
    compare_mesh_refinement(report, results.refinement)
    check_frame_stability(report, design, results.stability)
    compare_buckling_length_basis(report, setup, design, pipeline, results.stability)

    # The figures, which take the design and the measurements and nothing else.
    generate_figures(report, setup, design, pipeline, results)

    # Two claims harvested above and one from the refinement, each against its bound.
    check_util = ToleranceCheck("Overstressed!", departure, TOLERANCE_UTILIZATION)
    check_grad = ToleranceCheck("Gradient error", worst_gradient, TOLERANCE_GRADIENT)
    check_order = ToleranceCheck(
        "Oversizing from first order",
        results.refinement.worst_order,
        TOLERANCE_ORDER,
    )
    checks = (check_util, check_grad, check_order)

    report.write_heading("Summary")
    report.write_checks(checks)

    # A nan fails every bound already; this says so rather than relying on it.
    is_grad_finite = jnp.all(jnp.isfinite(gradient))
    is_checks_passed = checks_passed(checks)
    report.write_verdict(is_checks_passed and is_grad_finite)


if __name__ == "__main__":
    jnp.set_printoptions(precision=12)
    main()
