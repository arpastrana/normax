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
One schema, two solvers that disagree about how a derivative is obtained.

The analysis stage is swapped underneath a pipeline that does not know it
happened. `smax` is a JAX frame solver traced end to end; OpenSees is C++ behind
a command interface, differentiated by rules hand-derived element by element and
compiled in years before this pipeline existed. Neither is a reimplementation of
the other, so every agreement below is a measurement.

Four passes:

    agreement  the member forces, then every block of the Jacobian, then the
               mass and its gradient end to end
    blind      the one derivative a two-dimensional model cannot reach, and why
               the composition never asks for it
    scaling    what each backend pays for a value and for a gradient, against
               the size of the frame
    optimize   the P4 descent driven by each backend in turn, compared on the
               answer rather than on the derivative

**The 2D restriction is OpenSees' and not a simplification here.** Its Direct
Differentiation Method reaches a nodal coordinate in two dimensions and returns
zero or wrong values in three. See `CHANGELOG.md` under `## OpenSees DDM spike`.

Requires both the `spike` extra and the `pipeline` group:
    uv run --extra spike --group pipeline python \
        experiments/04_backend_agreement.py [pass]

with `pass` one of agreement, blind, scaling, optimize, or omitted for all.
"""

import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax_fdm.equilibrium import EquilibriumStructure
from jaxtyping import Array
from jaxtyping import Float

from normax.analysis import opensees as backend_opensees
from normax.analysis.smax import member_forces as forces_smax
from normax.analysis.smax import prepare_model as prepare_smax
from normax.design import DesignParameters
from normax.design import DesignPipeline
from normax.design import LoadCases
from normax.design import calculate_mass
from normax.design import load_cases
from normax.ec3.material import Steel
from normax.ec3.section import TubeCatalogue
from normax.form_finding.fdm import equilibrium_graph
from normax.form_finding.fdm import equilibrium_state
from normax.optimization import Trajectory
from normax.optimization import minimize_bounded
from normax.optimization import value_and_gradient
from normax.reporting import Report
from normax.reporting import ReportColumn
from normax.structures import Structure
from normax.structures import arch_2d
from normax.structures import loads_uniform
from normax.tesseract import TesseractAnalyzer
from normax.tesseract import TesseractFormFinder
from normax.tesseract import TesseractSizer
from normax.tesseract import analysis_backend
from normax.tesseract import local_chain
from normax.visualization import BackendAgreement
from normax.visualization import BackendTimings
from normax.visualization import figure_backends

# A 10 m arch rising 3 m, carrying 180 kN. The same one the rest of the
# experiments use, so the numbers here sit beside theirs.
SPAN = 10_000.0
RISE = 3_000.0
TOTAL_LOAD = 180_000.0
NUM_EDGES = 10

# The arch lies in the XZ plane, so it has no thickness along Y.
NORMAL = 1

# The diameter the frame is analyzed with before the check has spoken.
SEED = 100.0

# Frame sizes for the cost sweep. Each adds two coordinate parameters per node
# and two section parameters per member to the sweep OpenSees performs.
MESHES = (5, 10, 20, 40)

# Timed calls after the warm-up, at each size. Odd, so the median is a sample
# rather than an average of two.
REPEATS = 7

# What the roadmap asked the two backends to agree to.
TOLERANCE_ASKED = 1e-6

# The descent, matching `experiments/03`. The force densities may move a decade
# either side of the funicular value, the bound keeping them away from zero
# where the force density system is singular.
DECADES = 10.0
ITERATIONS = 60

FIGURES = Path(__file__).resolve().parent.parent / "figures"

STEEL = Steel()
CATALOGUE = TubeCatalogue.at_class_limit(STEEL.f_y, 3)

BACKENDS = ("smax", "opensees")


class ArchSetup(NamedTuple):
    """
    The arch every pass below is run on.

    Attributes
    ----------
    structure :
        The structure supplying the connectivity, the supports and the loads.
    graph :
        The form-finding connectivity, from `normax.form_finding`.
    q :
        Force densities that reach the target rise.
    """

    structure: Structure
    graph: EquilibriumStructure
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

    @property
    def xyz(self) -> Float[Array, "nodes 3"]:
        """
        The form-found geometry the analysis stage is handed.
        """
        state = equilibrium_state(
            self.q,
            self.structure.nodes[self.graph.indices_fixed],
            self.graph,
            self.funicular,
        )

        return state.xyz

    @property
    def loads(self) -> LoadCases:
        """
        The one load case the arch is shaped by and checked against.
        """
        applied = self.funicular

        return load_cases(applied, [applied])

    @property
    def funicular(self) -> Float[Array, "nodes 3"]:
        """
        The uniform load case the arch is form-found under.
        """
        return loads_uniform(self.structure, TOTAL_LOAD / (self.num_edges - 1))

    @property
    def bounds(self) -> tuple[float, float]:
        """
        The box the force densities may move in, a decade either side.
        """
        box = (float(self.q[0]) * DECADES, float(self.q[0]) / DECADES)

        return box


class BackendSeconds(NamedTuple):
    """
    What one piece of work cost each backend.

    Attributes
    ----------
    smax :
        Seconds the traced JAX solver took.
    opensees :
        Seconds the C++ solver differentiated by DDM took.
    """

    smax: float
    opensees: float

    @property
    def ratio(self) -> float:
        """
        Traced over DDM, below one where tracing wins.
        """
        return self.smax / self.opensees


class ScalingRow(NamedTuple):
    """
    What one frame size cost, and how closely the two backends agreed on it.

    Attributes
    ----------
    num_edges :
        Number of members in the frame.
    parameters :
        Number of quantities the direct differentiation sweep registers.
    gap :
        Worst relative disagreement in the mass gradient at this size.
    stage :
        Seconds the analysis stage alone spends on its derivatives.
    value :
        Seconds the whole composition spends on one mass.
    pipeline :
        Seconds the whole composition spends on a value and gradient.
    """

    num_edges: int
    parameters: int
    gap: float
    stage: BackendSeconds
    value: BackendSeconds
    pipeline: BackendSeconds


class DescentResult(NamedTuple):
    """
    One descent, driven by one backend.

    Attributes
    ----------
    backend :
        Which solver drove it.
    walked :
        Force densities and objective values, step by step.
    elapsed :
        Seconds the search took, the compilation excluded.
    compiling :
        Seconds the objective took to compile, paid once before the clock.
    """

    backend: str
    walked: Trajectory
    elapsed: float
    compiling: float

    @property
    def steps(self) -> int:
        """
        Steps the search took.
        """
        return int(self.walked.mass.shape[0])

    @property
    def per_step(self) -> float:
        """
        Milliseconds spent per step of the search.
        """
        return self.elapsed / self.steps * 1e3


def arch_setup(num_edges: int) -> ArchSetup:
    """
    The arch, its form-finding connectivity, and the `q` that reaches the rise.
    """
    structure = arch_2d(num_edges=num_edges, span=SPAN, rise=RISE)
    graph = equilibrium_graph(structure)
    applied = loads_uniform(structure, TOTAL_LOAD / (num_edges - 1))

    trial = jnp.full(num_edges, -1.0)
    state = equilibrium_state(
        trial, structure.nodes[graph.indices_fixed], graph, applied
    )
    reached = jnp.max(state.xyz[:, 2])
    setup = ArchSetup(structure, graph, trial * reached / RISE)

    return setup


def relative(actual, expected) -> float:
    """
    Worst absolute gap over the largest entry of the reference.
    """
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    scale = float(np.max(np.abs(expected)))

    return float(np.max(np.abs(actual - expected))) / (scale if scale > 0.0 else 1.0)


def mass_objective(setup: ArchSetup, chain) -> Callable[[Float[Array, "edges"]], Any]:
    """
    Force densities to a mass, through whichever backend is selected.
    """
    structure = setup.structure
    pipeline = DesignPipeline(
        TesseractFormFinder(chain.formfinding),
        TesseractAnalyzer(chain.analysis, STEEL, CATALOGUE, NORMAL),
        TesseractSizer(chain.ec3, STEEL, CATALOGUE),
    ).compile(structure)

    loads = setup.loads

    def total(q):
        return calculate_mass(pipeline(DesignParameters(q, setup.seed), loads))

    return total


def steady(call: Callable[[], Any], repeats: int = REPEATS) -> float:
    """
    Seconds per call once nothing is being compiled for the first time.

    Parameters
    ----------
    call :
        The work to time, taking no arguments and returning its result.
    repeats :
        Times to run it after the warm-up.

    Returns
    -------
    seconds :
        Median seconds per call.

    Notes
    -----
    **The median rather than the mean, because the composed timings are noisy.**
    A crossing of the boundary is host-side work of a few hundred milliseconds, and
    one sample landing at three times the rest — a collection, a page fault, the
    scheduler — moves a mean of five enough to reverse which backend looks faster.
    The stage timings are stable either way, so nothing is lost by taking the
    middle sample for both.

    **The warm-up is not optional and it is not noise.** The section slopes come
    from `jax.grad` of the closed forms, so the first call at a new member count
    compiles a kernel and reports two orders of magnitude more than the second.
    Timing it cold would measure XLA and call it direct differentiation. An
    optimizer pays that once and this cost hundreds of times.

    **The result is waited on rather than merely dispatched.** JAX returns before
    a computation has run, so timing without blocking measures the queueing of the
    traced backend against the completion of the C++ one, and flatters the first
    by however much of it is still outstanding. Whatever the call returns is
    blocked on; a call returning nothing would be timed wrongly and silently.
    """
    jax.block_until_ready(call())

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(call())
        samples.append(time.perf_counter() - start)

    return float(np.median(samples))


def stage_cost(setup: ArchSetup) -> BackendSeconds:
    """
    Seconds each backend spends on the analysis stage's derivatives alone.

    Isolated from the composition on purpose. The whole pipeline pays for form
    finding, a sizing bisection and two boundary crossings whoever solves the
    frame, and those dominate at these sizes; the scaling claim is about the
    stage.

    Notes
    -----
    **Both backends are prepared once and timed on the work that remains**, which
    is how the stage's contract says to use them. Preparing inside the timed call
    would charge the traced backend for compiling an assembly it is meant to reuse
    and charge neither for what an optimizer actually pays per iterate.

    **The traced Jacobian is compiled.** Uncompiled it runs two orders of
    magnitude slower, and comparing that against a C++ sweep measures Python
    dispatch rather than either differentiation strategy. The compilation is a
    fixed cost per frame size and the warm-up excludes it, exactly as it excludes
    the kernel the section slopes need on the other side.
    """
    xyz = setup.xyz
    diameters = setup.seed
    prepared_ddm = backend_opensees.prepare_model(
        setup.structure, STEEL, CATALOGUE, normal=NORMAL
    )
    prepared_smax = prepare_smax(setup.structure, STEEL, CATALOGUE, normal=NORMAL)

    def ddm():
        return backend_opensees.force_jacobian(
            prepared_ddm, xyz, diameters, STEEL, CATALOGUE, setup.funicular
        )

    run = traced_forces(prepared_smax, setup.funicular)
    coordinates = eqx.filter_jit(jax.jacfwd(run, argnums=0))
    sections = eqx.filter_jit(jax.jacfwd(run, argnums=1))

    def traced():
        blocks = (coordinates(xyz, diameters), sections(xyz, diameters))

        return blocks

    seconds = BackendSeconds(steady(traced), steady(ddm))

    return seconds


def traced_forces(prepared, applied) -> Callable[..., dict[str, Float[Array, "..."]]]:
    """
    The two member forces the composition consumes, as a differentiable map.
    """

    def run(coords, sizes):
        member = forces_smax(prepared, coords, sizes, STEEL, CATALOGUE, applied)
        forces = {
            "axial_force": member.axial_force,
            "end_moments_major": member.moment_major,
        }

        return forces

    return run


def agreement(report: Report) -> float:
    """
    Member forces, Jacobian blocks, and the mass gradient end to end.
    """
    report.write_banner("Two solvers on one schema -- agreement")

    setup = arch_setup(NUM_EDGES)
    xyz = setup.xyz
    diameters = setup.seed
    prepared_ddm = backend_opensees.prepare_model(
        setup.structure, STEEL, CATALOGUE, normal=NORMAL
    )
    prepared_smax = prepare_smax(setup.structure, STEEL, CATALOGUE, normal=NORMAL)

    mine = backend_opensees.member_forces(
        prepared_ddm, xyz, diameters, STEEL, CATALOGUE, setup.funicular
    )
    theirs = forces_smax(
        prepared_smax, xyz, diameters, STEEL, CATALOGUE, setup.funicular
    )

    force_columns = (
        ReportColumn("force", align="<"),
        ReportColumn("worst relative", ".3e"),
        ReportColumn("", align="<"),
    )
    force_rows = []
    for name in ("axial_force", "moment_major"):
        gap = relative(getattr(mine, name), getattr(theirs, name))
        force_rows.append((name, gap, ""))
    minor = float(np.max(np.abs(np.asarray(mine.moment_minor))))
    force_rows.append(("moment_minor", minor, "exactly zero in a plane frame"))

    report.write_heading("member forces, DDM backend against the traced one")
    report.write_table(force_columns, force_rows)

    blocks = backend_opensees.force_jacobian(
        prepared_ddm, xyz, diameters, STEEL, CATALOGUE, setup.funicular
    )
    run = traced_forces(prepared_smax, setup.funicular)
    by_coordinate = jax.jacfwd(run, argnums=0)(xyz, diameters)
    by_diameter = jax.jacfwd(run, argnums=1)(xyz, diameters)

    axial_xyz = ("axial_force_xyz", blocks.axial_force_xyz)
    moment_xyz = ("moment_major_xyz", blocks.moment_major_xyz)
    axial_diameter = ("axial_force_diameter", blocks.axial_force_diameter)
    moment_diameter = ("moment_major_diameter", blocks.moment_major_diameter)
    pairs = (
        (*axial_xyz, by_coordinate["axial_force"]),
        (*moment_xyz, by_coordinate["end_moments_major"]),
        (*axial_diameter, by_diameter["axial_force"]),
        (*moment_diameter, by_diameter["end_moments_major"]),
    )

    block_columns = (
        ReportColumn("block", align="<"),
        ReportColumn("shape", align="<"),
        ReportColumn("worst relative", ".3e"),
    )
    block_rows = [
        (name, str(np.asarray(ddm).shape), relative(ddm, traced))
        for name, ddm, traced in pairs
    ]
    worst = max(relative(ddm, traced) for _, ddm, traced in pairs)
    block_entries = (("worst over every block", f"{worst:.3e}"),)

    report.write_heading("Jacobian blocks, hand-derived C++ against traced autodiff")
    report.write_table(block_columns, block_rows)
    report.write_entries(block_entries)

    total = mass_objective(setup, local_chain())
    masses = {}
    gradients = {}
    for name in BACKENDS:
        with analysis_backend(name):
            masses[name] = float(total(setup.q))
            gradients[name] = np.asarray(jax.grad(total)(setup.q))

    mass_gap = abs(masses["opensees"] - masses["smax"]) / masses["smax"]
    grad_gap = relative(gradients["opensees"], gradients["smax"])

    by_backend = [(name, f"mass {masses[name]:.9f} t") for name in BACKENDS]
    asked = f"worst relative {grad_gap:.3e}, asked {TOLERANCE_ASKED:.0e}"
    entries = (
        *by_backend,
        ("mass", f"relative gap {mass_gap:.3e}"),
        ("dmass/dq", asked),
    )

    report.write_heading("end to end, force densities to a mass")
    report.write_entries(entries)

    return grad_gap


def blind(report: Report) -> None:
    """
    The one derivative the plane cannot carry, and why nothing asks for it.
    """
    report.write_banner("The block a two-dimensional model cannot reach")

    setup = arch_setup(NUM_EDGES)
    xyz = setup.xyz
    diameters = setup.seed
    prepared = prepare_smax(setup.structure, STEEL, CATALOGUE, normal=NORMAL)

    def run(coords):
        member = forces_smax(
            prepared, coords, diameters, STEEL, CATALOGUE, setup.funicular
        )
        forces = {
            "axial_force": member.axial_force,
            "end_moments_major": member.moment_major,
            "end_moments_minor": member.moment_minor,
        }

        return forces

    jacobian = jax.jacfwd(run)(xyz)

    columns = (
        ReportColumn("output", align="<"),
        ReportColumn("d/dx", ".6e"),
        ReportColumn("d/dy", ".6e"),
        ReportColumn("d/dz", ".6e"),
        ReportColumn("", align="<"),
    )
    rows = []
    for name, block in jacobian.items():
        sizes = [
            float(np.max(np.abs(np.asarray(block)[..., axis]))) for axis in range(3)
        ]
        mark = "<- normal" if name == "end_moments_minor" else ""
        rows.append((name, *sizes, mark))

    report.write_heading("the three-dimensional Jacobian, by global axis")
    report.write_table(columns, rows)
    report.write_note(
        """
        The response separates: nothing in the plane moves when a node leaves it,
        and the minor-axis moment moves only then. A plane model carries every
        block but that one.
        """
    )

    interior = xyz.shape[0] // 2
    tangent = jnp.zeros_like(xyz).at[interior, NORMAL].set(1.0)
    _, pushed = jax.jvp(run, (xyz,), (tangent,))

    entries = [
        (name, f"{float(np.max(np.abs(np.asarray(value)))):.6e}")
        for name, value in pushed.items()
    ]

    report.write_heading(f"the same, pushing node {interior} alone out of the plane")
    report.write_entries(entries)
    report.write_note(
        """
        One node, because translating every node out of the plane together is a
        rigid motion and strains nothing — it would read as blindness where there
        is none.
        """
    )

    def positions(q):
        state = equilibrium_state(
            q,
            setup.structure.nodes[setup.graph.indices_fixed],
            setup.graph,
            setup.funicular,
        )

        return state.xyz

    reachable = jax.jacfwd(positions)(setup.q)
    out_of_plane = float(np.max(np.abs(np.asarray(reachable)[:, NORMAL, :])))
    entries = (("worst |d xyz[normal] / dq|", f"{out_of_plane:.3e}"),)

    report.write_heading("and what form finding can do about it")
    report.write_entries(entries)
    report.write_note(
        """
        The force density method decouples per coordinate, so a planar arch stays
        planar for every q. The blind block is multiplied by zero.
        """
    )


def scaling_row(num_edges: int) -> ScalingRow:
    """
    One frame size, timed through the composition and through the stage alone.
    """
    setup = arch_setup(num_edges)
    total = mass_objective(setup, local_chain())

    gradients = {}
    values = {}
    seconds = {}
    for name in BACKENDS:
        with analysis_backend(name):
            gradient = jax.grad(total)
            values[name] = steady(lambda f=total: f(setup.q))
            seconds[name] = steady(lambda f=gradient: f(setup.q))
            gradients[name] = np.asarray(gradient(setup.q))

    parameters = 2 * (num_edges + 1) + 2 * num_edges
    gap = relative(gradients["opensees"], gradients["smax"])
    stage = stage_cost(setup)
    value = BackendSeconds(values["smax"], values["opensees"])
    pipeline = BackendSeconds(seconds["smax"], seconds["opensees"])
    row = ScalingRow(num_edges, parameters, gap, stage, value, pipeline)

    return row


def scaling(report: Report) -> None:
    """
    What a value and a gradient cost each backend, against frame size.

    Notes
    -----
    **Two different things are timed and they answer different questions.** The
    stage alone compares one backend's derivatives against the other's with both
    prepared once and the traced one compiled, which is what a caller of the stage
    pays per iterate. The whole composition runs through the Tesseracts, so it
    also pays form finding, a sizing bisection and two boundary crossings, and
    those dominate at these sizes whoever solves the frame.

    **The composed path is compiled but not prepared once.** Its solve is compiled
    inside the backend, so what the composed columns still carry is one assembly
    per crossing: a boundary is stateless and keeps nothing between calls. The
    in-process pipeline is the one that also reuses a prepared model, and
    `experiments/03` is where that shows.
    """
    report.write_banner("Cost against frame size")

    rows = [scaling_row(num_edges) for num_edges in MESHES]

    composed_columns = (
        ReportColumn("members"),
        ReportColumn("params"),
        ReportColumn("backend"),
        ReportColumn("value [s]", ".3f"),
        ReportColumn("grad [s]", ".3f"),
    )
    composed_rows = []
    for row in rows:
        for name in BACKENDS:
            value = getattr(row.value, name)
            gradient = getattr(row.pipeline, name)
            composed_rows.append((row.num_edges, row.parameters, name, value, gradient))

    report.write_heading("the whole composition, one value then a value and gradient")
    report.write_note(
        "through the Tesseracts, warmed, the assembly rebuilt at each crossing"
    )
    report.write_table(composed_columns, composed_rows)

    stage_columns = (
        ReportColumn("members"),
        ReportColumn("params"),
        ReportColumn("DDM [ms]", ".1f"),
        ReportColumn("traced [ms]", ".1f"),
    )
    stage_rows = [
        (row.num_edges, row.parameters, row.stage.opensees * 1e3, row.stage.smax * 1e3)
        for row in rows
    ]

    report.write_heading("the analysis stage alone, every derivative it can report")
    report.write_note("both prepared once, the traced one compiled, warm-up excluded")
    report.write_table(stage_columns, stage_rows)

    winner_columns = (
        ReportColumn("members"),
        ReportColumn("worst relative", ".3e"),
        ReportColumn("ms per param", ".3f"),
        ReportColumn("stage traced/DDM", ".2f"),
        ReportColumn("composition traced/DDM", ".2f"),
    )
    winner_rows = []
    for row in rows:
        per_param = row.stage.opensees / row.parameters * 1e3
        winner_rows.append(
            (row.num_edges, row.gap, per_param, row.stage.ratio, row.pipeline.ratio)
        )

    report.write_heading("agreement, cost per parameter, and which backend wins")
    report.write_table(winner_columns, winner_rows)
    report.write_note(
        """
        Below one the traced gradient wins. Both are compiled; the composition
        prepares its assembly per crossing, a boundary keeping nothing between
        calls.
        """
    )

    members = np.asarray([row.num_edges for row in rows])
    gaps = np.asarray([row.gap for row in rows])
    agreed = BackendAgreement(members, gaps, TOLERANCE_ASKED)

    parameters = np.asarray([row.parameters for row in rows])
    stage = {}
    pipeline = {}
    for name in BACKENDS:
        stage[name] = np.asarray([getattr(row.stage, name) for row in rows])
        pipeline[name] = np.asarray([getattr(row.pipeline, name) for row in rows])
    timings = BackendTimings(parameters, stage, pipeline)

    figure = figure_backends(agreed, timings)
    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / "04_backends.png"
    figure.savefig(path, dpi=200)
    report.write_heading(f"wrote {path}")


def descend_with(setup: ArchSetup, backend: str) -> DescentResult:
    """
    The descent driven by one backend, its compilation timed separately.
    """
    total = mass_objective(setup, local_chain())

    with analysis_backend(backend):
        gradient = value_and_gradient(total)

        start = time.perf_counter()
        jax.block_until_ready(gradient(setup.q))
        compiling = time.perf_counter() - start

        start = time.perf_counter()
        walked = minimize_bounded(
            total,
            setup.q,
            bounds=setup.bounds,
            iterations=ITERATIONS,
            gradient=gradient,
        )
        elapsed = time.perf_counter() - start

    found = DescentResult(backend, walked, elapsed, compiling)

    return found


def optimize(report: Report) -> None:
    """
    The same descent, driven by each backend in turn.

    Notes
    -----
    **Compiled before the clock starts, and the compilation reported beside the
    search rather than inside it.** Each objective is traced once here and the
    compiled program handed to the search, so the elapsed time of a descent is the
    work it did. Leaving it inside would charge one backend a fixed cost the other
    never pays and call the difference a solver comparison.

    The compilation is a real cost and is printed, not hidden. It is paid once per
    objective however long the search runs, so it matters on a descent of seven
    steps and vanishes on one of several hundred.
    """
    report.write_banner("The same optimization, one solver swapped for the other")

    setup = arch_setup(NUM_EDGES)
    report.write_heading(f"one variable per member, bounds {setup.bounds}")

    results = [descend_with(setup, name) for name in BACKENDS]
    columns = (
        ReportColumn("backend", align="<"),
        ReportColumn("mass [t]", ".9f"),
        ReportColumn("steps"),
        ReportColumn("seconds", ".1f"),
        ReportColumn("ms/step", ".0f"),
        ReportColumn("compiled in [s]", ".2f"),
    )
    rows = []
    for found in results:
        mass = float(found.walked.mass[-1])
        timings = (found.elapsed, found.per_step, found.compiling)
        rows.append((found.backend, mass, found.steps, *timings))

    report.write_table(columns, rows)

    first, second = results
    first_mass = float(first.walked.mass[-1])
    reached = abs(float(second.walked.mass[-1]) - first_mass)
    q_gap = relative(second.walked.q[-1], first.walked.q[-1])
    speedup = first.elapsed / second.elapsed
    entries = (
        ("mass", f"relative gap {reached / first_mass:.3e}"),
        ("q", f"worst relative {q_gap:.3e}"),
        ("steps", f"{first.steps} against {second.steps}"),
        ("wall clock", f"{speedup:.1f}x faster on the C++ backend, compiled"),
    )

    report.write_heading("the two answers")
    report.write_entries(entries)


PASSES = {
    "agreement": agreement,
    "blind": blind,
    "scaling": scaling,
    "optimize": optimize,
}


def main(verbose: bool = True) -> None:
    """
    Run the requested passes, or every one of them.
    """
    requested = sys.argv[1:] or list(PASSES)

    for name in requested:
        if name not in PASSES:
            raise SystemExit(f"unknown pass {name!r}; choose from {list(PASSES)}")

    report = Report(verbose)
    for name in requested:
        PASSES[name](report)
        report.write_line()


if __name__ == "__main__":
    main()
