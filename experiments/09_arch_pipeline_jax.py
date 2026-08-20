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

**The buckling length is the member's own length, and that is a strong
assumption.** It presumes every node is held in plane by structure outside the
model, so a member can only buckle between its ends. Verifying that against the
stability of the whole frame is future work: nothing in this package computes a
critical load factor, and no experiment prices the assumption.

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

from normax.analysis import SmaxAnalyzer
from normax.design import Design
from normax.design import DesignParameters
from normax.design import StructuralDesignPipeline
from normax.design import compute_mass
from normax.design import design_envelope
from normax.form_finding import FdmFormFinder
from normax.form_finding import equilibrium_graph
from normax.form_finding import equilibrium_state
from normax.loads import LoadCases
from normax.loads import assemble_load_cases
from normax.loads import create_loads_uniform
from normax.materials import Steel355
from normax.reporting import Report
from normax.reporting import ReportColumn
from normax.reporting import ToleranceCheck
from normax.reporting import checks_passed
from normax.sizing import Ec3Sizer
from normax.sizing import build_section_family
from normax.sizing import design_actions
from normax.structures import Structure
from normax.structures import build_arch_2d
from normax.visualization import MeshRefinement
from normax.visualization import SizedMembers
from normax.visualization import StaggeredPasses
from normax.visualization import figure_convergence
from normax.visualization import figure_sections

# A 10 m arch rising 3 m, carrying 180 kN spread over its free nodes. Units are
# millimeters and newtons.
SPAN = 10_000.0
RISE = 3_000.0
TOTAL_LOAD = 180_000.0
NUM_EDGES = 10

# The diameter the frame is analyzed with before the check has spoken. Only the
# stiffness depends on it, and the check overwrites it.
SEED = 100.0

# Meshes for the refinement study, doubling.
MESHES = (5, 10, 20, 40, 80, 160)

# A buckling length independent of the mesh, so refinement measures the
# discretization rather than the slenderness.
BUCKLING_LENGTH = 1_000.0

PASSES = 6

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

GRADE = Steel355()

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
        return create_loads_uniform(self.structure, TOTAL_LOAD / (self.num_edges - 1))

    @property
    def xyz_fixed(self) -> Float[Array, "nodes_fixed 3"]:
        """
        Position of every support, which form finding holds where it is.
        """
        return self.structure.nodes[self.graph.indices_fixed]

    @property
    def loads(self) -> LoadCases:
        """
        The one load case the arch is shaped by and checked against.
        """
        applied = self.funicular

        return assemble_load_cases([applied])

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
    Refinement and stagger studies for one design.
    """

    refinement: RefinementStudy
    stagger: StaggerRun


def arch_setup(num_edges: int) -> ArchSetup:
    """
    Build the arch mesh, topologies, and rise-reaching force densities.
    """
    structure = build_arch_2d(num_edges=num_edges, span=SPAN, rise=RISE)
    graph = equilibrium_graph(structure)
    # Standing a frame up needs a section, and every property of it is replaced
    # per call; the seed tube is drawn from the class-3 family as bare geometry.
    seeded = build_section_family(GRADE, 3)
    analyzer = SmaxAnalyzer(structure, seeded(SEED))
    applied = create_loads_uniform(structure, TOTAL_LOAD / (num_edges - 1))

    trial = jnp.full(num_edges, -1.0)
    xyz_fixed = structure.nodes[graph.indices_fixed]
    state = equilibrium_state_compiled(trial, xyz_fixed, graph, applied)
    reached = jnp.max(state.xyz[:, 2])
    q = trial * reached / RISE
    setup = ArchSetup(structure, graph, analyzer, q)

    return setup


def pipeline_from_setup(
    setup: ArchSetup, section_class: int
) -> StructuralDesignPipeline:
    """
    Bind a section class and this mesh into the three blocks.
    """
    structure = setup.structure
    family = build_section_family(GRADE, section_class)
    sizer = Ec3Sizer(structure, family)
    blocks = StructuralDesignPipeline(
        FdmFormFinder(structure),
        SmaxAnalyzer(structure, family(SEED)),
        sizer,
    )

    return blocks


def design_at_length(
    pipeline: StructuralDesignPipeline,
    params: DesignParameters,
    loads: LoadCases,
    buckling_length: Float[Array, "members"],
) -> Design:
    """
    The design a stated buckling length asks for, rather than the member length.

    The pipeline hands the member length to the check, which is the braced-node
    assumption this experiment measures the cost of, so the three blocks are
    composed here to put a length of its own choosing in that place.
    """
    shape = pipeline.formfinder(params.force_densities, loads.formfinding)
    forces = pipeline.analyzer(shape.xyz, params.diameters, loads.analysis)
    sizes = pipeline.sizer(forces, buckling_length)
    design = Design(shape, forces, sizes)

    return design_envelope(design)


# Called on every mesh of the refinement study, so traced rather than eager.
design_at_length_compiled = eqx.filter_jit(design_at_length)


def worst_utilization(design: Design) -> float:
    """
    Largest absolute departure of utilization from unity.
    """
    return float(jnp.max(jnp.abs(design.sizes.utilization - 1.0)))


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
) -> tuple[Design, StructuralDesignPipeline]:
    """
    Fully-stressed design of the arch on one section class.
    """
    pipeline = pipeline_from_setup(setup, section_class)
    by_case = eqx.filter_jit(pipeline)(setup.params, setup.loads)
    design = design_envelope(by_case)

    return design, pipeline


def run_stagger(
    setup: ArchSetup,
    pipeline: StructuralDesignPipeline,
    passes: int,
) -> StaggerRun:
    """
    Run analyse-and-size passes, recording diameter moves and mass.
    """
    diameters = setup.seed
    moves = []
    masses = []

    for _ in range(passes):
        params = DesignParameters(setup.q, diameters)
        by_case = eqx.filter_jit(pipeline)(params, setup.loads)
        result = design_envelope(by_case)
        required = result.sizes.sections.diameter
        shift = jnp.abs(required - diameters) / required
        moves.append(float(jnp.max(shift)))
        masses.append(float(compute_mass(result)))
        diameters = required

    run = StaggerRun(np.asarray(moves), np.asarray(masses))

    return run


def refinement_study(section_class: int) -> RefinementStudy:
    """
    Mass on each mesh and the first-order convergence of that sequence.
    """
    rows = []
    for count in MESHES:
        refined = arch_setup(count)
        blocks = pipeline_from_setup(refined, section_class)
        pipeline = eqx.filter_jit(blocks)
        free = design_envelope(pipeline(refined.params, refined.loads))
        fixed_length = jnp.full(count, BUCKLING_LENGTH)
        held = design_at_length_compiled(
            blocks, refined.params, refined.loads, fixed_length
        )
        arc = float(jnp.sum(free.shape.lengths))
        row = RefinementRow(
            count, arc, float(compute_mass(free)), float(compute_mass(held))
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
        setup.q, setup.xyz_fixed, setup.graph, setup.funicular
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
    report: Report, design: Design, pipeline: StructuralDesignPipeline
) -> None:
    """
    Write each member's actions, size, utilization, and governing limit.
    """
    diameters = design.sizes.sections.diameter
    actions = jax.vmap(design_actions)(design.forces)
    codes = pipeline.sizer.governing(diameters, design.forces, design.shape.lengths)[0]
    limits = {LIMIT_NAMES[float(code)] for code in codes}

    columns = (
        ReportColumn("member"),
        ReportColumn("N [kN]", ".4f"),
        ReportColumn("M [kNm]", ".5f"),
        ReportColumn("d [mm]", ".4f"),
        ReportColumn("utilization", ".16f"),
    )
    rows = []
    for member in range(diameters.shape[0]):
        force = float(actions.axial_force[0, member]) / 1e3
        moment = float(actions.moment_major[0, member]) / 1e6
        diameter = float(diameters[member])
        utilization = float(design.sizes.utilization[0, member])
        rows.append((member, force, moment, diameter, utilization))

    entries = (
        ("mass", f"{float(compute_mass(design)):.9f} t"),
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


def generate_figures(
    report: Report,
    setup: ArchSetup,
    design: Design,
    pipeline: StructuralDesignPipeline,
    results: ArchResults,
) -> None:
    """
    Write the sections and convergence figures.
    """
    study = results.refinement
    FIGURES.mkdir(exist_ok=True)

    diameters = design.sizes.sections.diameter
    xyz = design.shape.xyz
    seed_tubes = pipeline.sizer.family(setup.seed)
    assumed = float(GRADE.density * jnp.sum(seed_tubes.area * design.shape.lengths))
    seeded = SizedMembers(setup.seed, assumed)
    sized = SizedMembers(diameters, float(compute_mass(design)))
    sections = figure_sections(xyz, setup.structure.edges, seeded, sized)
    sections.savefig(FIGURES / "09_sections.png", dpi=160, bbox_inches="tight")

    counts = np.asarray(MESHES)
    refinement = MeshRefinement(
        counts, study.masses_member, study.masses_fixed, study.limit
    )
    passes = np.arange(len(results.stagger.moves))
    staggered = StaggeredPasses(passes, results.stagger.moves)
    convergence = figure_convergence(refinement, staggered)
    convergence.savefig(FIGURES / "09_convergence.png", dpi=160, bbox_inches="tight")
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
        params = DesignParameters(q, setup.seed)
        by_case = pipeline(params, setup.loads)

        return compute_mass(design_envelope(by_case))

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
    refinement = refinement_study(section_class)
    # The sizes fed back as the stiffness they were found with, until they hold.
    stagger = run_stagger(setup, pipeline, PASSES)
    # One container, so a report and a figure read the same measurements.
    results = ArchResults(refinement, stagger)

    compare_stagger_closure(report, results.stagger)
    compare_mesh_refinement(report, results.refinement)

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
