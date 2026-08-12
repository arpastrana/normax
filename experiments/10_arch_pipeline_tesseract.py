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
The same arch, the same mass, the same gradient — across three Tesseracts.

Experiment 09 ran the pipeline as one process and one JAX trace. This runs it as
three components with schemas between them, each differentiating in its own way,
and asks whether anything changed. Nothing should: the boundary is a claim about
composition, not about arithmetic.

Four things are reported.

    schemas      what each stage promises, and what it promises a derivative in
    parity       the design and the gradient, against experiment 09's answers
    directions   forward mode against reverse mode, through all three stages
    refusal      what happens to a cotangent on a non-differentiable output

**The in-process pipeline is the oracle, and that is the point rather than an
apology.** Pasteur's own caveat is that a single developer with a single stack
might not need Tesseracts, and the honest answer to it is not that the boundary
is convenient. It is that the boundary is free in the answer — measured here to
the last bits — and that a second analysis backend which JAX cannot trace at all
slots in behind the same schema without anything above it changing. It is not
free in wall clock, and the seconds below say so: both sides are compiled, and
crossing three schemas still costs what serializing and reassembling costs.

**Compiling the composed side does not fold the boundary away.** A Tesseract is a
primitive JAX lowers a callback for, so all three stages are crossed on every call
of the compiled program rather than once while tracing — three crossings per call,
counted at the stages' own apply endpoints. Compiling changes nothing in the
answers here, to the last bit, and the programs are kept between runs.

Run with `uv run --group pipeline python experiments/10_arch_pipeline_tesseract.py`.
"""

import os
import time
from collections.abc import Callable
from collections.abc import Iterator
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
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

from normax.analysis.smax import prepare_model
from normax.composition import Chain
from normax.composition import ProblemSetup as ComposedSetup
from normax.composition import design_members as design_composed
from normax.composition import local_chain
from normax.composition import total_mass as mass_composed
from normax.ec3.material import SteelGrade
from normax.ec3.section import TubeCatalogue
from normax.formfinding import equilibrium_graph
from normax.formfinding import equilibrium_state
from normax.pipeline import ProblemSetup
from normax.pipeline import design_members as design_in_process
from normax.pipeline import governing_states
from normax.pipeline import total_mass as mass_in_process
from normax.reporting import ColumnSpec
from normax.reporting import ReportWriter
from normax.reporting import ToleranceCheck
from normax.reporting import checks_passed
from normax.structures import Structure
from normax.structures import arch_2d

# Compiled programs outlive the process, so a second run pays for arithmetic and
# for crossings alone. Every compilation here is well under the one second the
# persistent cache keeps by default, which would otherwise leave all of them out.
COMPILATION_CACHE = Path(__file__).resolve().parent.parent / ".jax_cache"
COMPILATION_CACHE.mkdir(exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(COMPILATION_CACHE))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

# The oracle is compiled, as experiment 09, the parity test and the README all
# run it. The Tesseract stages compile internally, so an eager oracle would put
# two different fusion schedules either side of the comparison and charge the
# difference to the boundary.
design_compiled = eqx.filter_jit(design_in_process)
mass_compiled = eqx.filter_jit(mass_in_process)

# The arch of experiment 09, unchanged, so the two are comparable.
SPAN = 10_000.0
RISE = 3_000.0
TOTAL_LOAD = 180_000.0
NUM_EDGES = 10

# The arch lies in the XZ plane, so it has no thickness along Y.
NORMAL = 1

# The diameter the frame is analyzed with before the check has spoken.
SEED = 100.0

CLASSES = (2, 3)

# Values cross the boundary exactly. Derivatives do not, and not because of the
# boundary: each stage linearizes on its own here and all three linearize
# together in process, so the same sum accumulates in a different order.
TOLERANCE_PARITY = 1e-14
TOLERANCE_DERIVATIVE = 1e-12

# The end moments are the exception, and the arch is the reason rather than the
# boundary. A funicular shape carries its design case axially, so the moment is
# what is left over: measured here it is 3.9e-4 of the axial action times the
# length. A last-bit difference in the analysis inputs is amplified by the
# reciprocal of that ratio before it reaches the moment, so the floor sits three
# orders above the axial force it came from — 3.6e-13 against 7e-16. The moment
# factors read a ratio of the two end moments and inherit it, and the diameter
# inherits a fiftieth of it, the moment being worth that much of the utilization.
TOLERANCE_MOMENT = 1e-11
MOMENT_FIELDS = (
    "moment_major",
    "moment_minor",
    "moment_factor_major",
    "moment_factor_minor",
)

# Serializing across a socket costs a few more digits than importing the module
# does, so the containers are held to a looser bound than the in-process chain.
TOLERANCE_SERVED = 1e-11

# The two stages that containerize, and the tag `tesseract build` gives them.
IMAGES = ("normax-formfinding", "normax-ec3-check")
VERSION = "0.1.0"

STEEL = SteelGrade()

LIMIT_NAMES = {
    0.0: "catalogue minimum",
    1.0: "tension",
    2.0: "cross-section",
    3.0: "6.61 major",
    4.0: "6.62 minor",
}


class ArchSetup(NamedTuple):
    """
    The arch every comparison below is made on.

    Attributes
    ----------
    structure :
        The structure supplying the connectivity, the supports and the loads.
    graph :
        The form-finding connectivity, from `normax.formfinding`.
    q :
        Force densities that reach the target rise.
    """

    structure: Structure
    graph: EquilibriumStructure
    q: Float[Array, "edges"]

    @property
    def seed(self) -> Float[Array, "edges"]:
        """
        The diameter the frame is analyzed with before the check has spoken.
        """
        return jnp.full(NUM_EDGES, SEED)


class TimedCall(NamedTuple):
    """
    What a call returned, and how long the call took.

    Attributes
    ----------
    result :
        Whatever the call returned.
    seconds :
        Wall-clock seconds it took, the first call excluded as warm-up.
    """

    result: Any
    seconds: float


class ParityWorst(NamedTuple):
    """
    The worst disagreement between the composed chain and the oracle.

    Attributes
    ----------
    value :
        Worst scaled difference on a quantity that is not an end moment.
    moment :
        Worst scaled difference on an end moment or a moment factor.
    gradient :
        Worst scaled difference on the gradient of the mass.
    """

    value: float
    moment: float
    gradient: float

    def worse_than(self, other: "ParityWorst") -> "ParityWorst":
        """
        The worse of two measurements, field by field.
        """
        value = max(self.value, other.value)
        moment = max(self.moment, other.moment)
        gradient = max(self.gradient, other.gradient)
        worst = ParityWorst(value, moment, gradient)

        return worst


def arch_setup() -> ArchSetup:
    """
    The arch, its form-finding connectivity, and the `q` that reaches the rise.
    """
    load = TOTAL_LOAD / (NUM_EDGES - 1)
    structure = arch_2d(num_edges=NUM_EDGES, span=SPAN, rise=RISE, load=load)
    graph = equilibrium_graph(structure)

    trial = jnp.full(NUM_EDGES, -1.0)
    state = equilibrium_state(trial, structure, graph)
    reached = jnp.max(state.xyz[:, 2])
    setup = ArchSetup(structure, graph, trial * reached / RISE)

    return setup


def named_fields(container: NamedTuple) -> Iterator[tuple[str, Any]]:
    """
    Every field of a result, with a nested container expanded one level.

    A container holds quantities of different units, so comparing one as a
    single array scales a moment by an axial force and reports a ratio of no
    physical meaning. Each leaf is measured against itself instead.
    """
    for field in container._fields:
        value = getattr(container, field)
        if hasattr(value, "_fields"):
            for inner in value._fields:
                yield f"{field}.{inner}", getattr(value, inner)
        else:
            yield field, value


def relative(oracle, composed) -> float:
    """
    Largest disagreement between two arrays, scaled by the size of the first.
    """
    left = np.asarray(oracle, dtype=np.float64)
    right = np.asarray(composed, dtype=np.float64)
    scale = max(float(np.max(np.abs(left))), np.finfo(np.float64).tiny)

    return float(np.max(np.abs(left - right))) / scale


def timed(call: Callable[[], Any]) -> TimedCall:
    """
    A call's result and how long it took, the first call excluded as warm-up.
    """
    call()
    start = time.perf_counter()
    result = call()
    seconds = time.perf_counter() - start
    call = TimedCall(result, seconds)

    return call


def report_schemas(report: ReportWriter, chain: Chain) -> None:
    """
    What every stage carries, and what it will differentiate.
    """
    report.write_line("The three schemas")
    for stage, tesseract in zip(chain._fields, chain):
        schemas = tesseract.openapi_schema["components"]["schemas"]
        inputs = schemas["Apply_InputSchema"]["properties"]
        outputs = schemas["Apply_OutputSchema"]["properties"]
        wrt = schemas["ApplyInputSchema"]["differentiable_arrays"]
        of = schemas["ApplyOutputSchema"]["differentiable_arrays"]

        opaque = ", ".join(sorted(set(outputs) - set(of))) or "-"
        entries = (
            ("in", ", ".join(sorted(inputs))),
            ("out", ", ".join(sorted(outputs))),
            ("d/d", ", ".join(sorted(wrt))),
            ("d of", ", ".join(sorted(of))),
            ("not diff", opaque),
        )

        report.write_heading(stage)
        report.write_entries(entries)


def report_parity(
    report: ReportWriter,
    setup: ArchSetup,
    chain: Chain,
    section_class: int,
) -> ParityWorst:
    """
    The design and the gradient, taken in process and taken across the boundary.
    """
    catalogue = TubeCatalogue.at_class_limit(STEEL.f_y, section_class)
    model = prepare_model(setup.structure, STEEL, catalogue, normal=NORMAL)
    problem = ProblemSetup(setup.structure, setup.graph, model, STEEL, catalogue)
    composed_problem = ComposedSetup(setup.structure, chain, STEEL, catalogue)

    oracle = timed(
        lambda: design_compiled(
            setup.q, setup.seed, problem, section_class=section_class
        )
    )

    # The composed side compiles as a closure rather than as an alias like the
    # oracle: its problem holds the structure, every crossing serializes that to
    # numpy for the payload, and an argument of a compiled function arrives as a
    # tracer that `np.asarray` refuses. Only the force densities vary per call.
    def composed_design(q):
        return design_composed(
            q,
            setup.seed,
            composed_problem,
            normal=NORMAL,
            section_class=section_class,
        )

    design_of = eqx.filter_jit(composed_design)

    composed = timed(lambda: design_of(setup.q))

    rows = []
    worst_value = 0.0
    worst_moment = 0.0
    for (label, oracle_leaf), (_, composed_leaf) in zip(
        named_fields(oracle.result), named_fields(composed.result)
    ):
        left = np.asarray(oracle_leaf, dtype=np.float64)
        right = np.asarray(composed_leaf, dtype=np.float64)
        scaled = relative(left, right)

        if label.rpartition(".")[2] in MOMENT_FIELDS:
            limit = TOLERANCE_MOMENT
            worst_moment = max(worst_moment, scaled)
        else:
            limit = TOLERANCE_PARITY
            worst_value = max(worst_value, scaled)

        first = (float(left.ravel()[0]), float(right.ravel()[0]))
        rows.append((label, *first, scaled, limit))

    columns = (
        ColumnSpec("field", align="<"),
        ColumnSpec("in process", ".14e"),
        ColumnSpec("composed", ".14e"),
        ColumnSpec("scaled", ".2e"),
        ColumnSpec("held to", ".0e"),
    )
    ratio = float(catalogue.ratio)

    report.write_heading(f"Class {section_class}, d/t = {ratio:.3f}")
    report.write_table(columns, rows)

    codes = governing_states(oracle.result, problem, section_class=section_class)
    limits = {LIMIT_NAMES[float(code)] for code in codes}
    departure = float(jnp.max(jnp.abs(composed.result.utilization - 1.0)))
    seconds = (
        f"{oracle.seconds:.4f} in process, {composed.seconds:.4f} composed,"
        " both compiled"
    )
    entries = (
        ("governing", ", ".join(sorted(limits))),
        ("mass", f"{float(composed.result.mass):.12f} t"),
        ("worst |u-1|", f"{departure:.2e}"),
        ("seconds", seconds),
    )

    report.write_entries(entries)

    def in_process(q):
        return mass_compiled(q, setup.seed, problem, section_class=section_class)

    def composed_mass(q):
        return mass_composed(
            q,
            setup.seed,
            composed_problem,
            normal=NORMAL,
            section_class=section_class,
        )

    exact_of = eqx.filter_jit(jax.grad(in_process))
    crossed_of = eqx.filter_jit(jax.grad(composed_mass))

    exact = timed(lambda: exact_of(setup.q))
    crossed = timed(lambda: crossed_of(setup.q))
    scale = float(jnp.max(jnp.abs(exact.result)))

    gradients = []
    for edge in range(NUM_EDGES):
        in_process_value = float(exact.result[edge])
        composed_value = float(crossed.result[edge])
        scaled = abs(in_process_value - composed_value) / scale
        gradients.append((edge, in_process_value, composed_value, scaled))

    columns = (
        ColumnSpec("edge"),
        ColumnSpec("in process", ".14e"),
        ColumnSpec("composed", ".14e"),
        ColumnSpec("scaled", ".2e"),
    )
    seconds = (
        f"{exact.seconds:.4f} in process, {crossed.seconds:.4f} composed, both compiled"
    )
    entries = (
        ("sum", f"{float(jnp.sum(crossed.result)):.14e}"),
        ("seconds", seconds),
    )

    report.write_table(columns, gradients)
    report.write_entries(entries)

    worst_gradient = max(row[3] for row in gradients)
    worst = ParityWorst(worst_value, worst_moment, worst_gradient)

    return worst


def report_modes(report: ReportWriter, setup: ArchSetup, chain: Chain) -> float:
    """
    Forward mode against reverse mode, through all three stages.
    """
    catalogue = TubeCatalogue.at_class_limit(STEEL.f_y, 3)

    composed_problem = ComposedSetup(setup.structure, chain, STEEL, catalogue)

    def objective(q):
        return mass_composed(
            q,
            setup.seed,
            composed_problem,
            normal=NORMAL,
            section_class=3,
        )

    def forward_derivative(q, tangent):
        _, derivative = jax.jvp(objective, (q,), (tangent,))

        return derivative

    forward_of = eqx.filter_jit(forward_derivative)
    gradient_of = eqx.filter_jit(jax.grad(objective))

    direction = jnp.ones_like(setup.q)
    forward = forward_of(setup.q, direction)
    gradient = gradient_of(setup.q)
    reverse = float(jnp.sum(gradient * direction))
    modes = relative(reverse, forward)
    entries = (
        ("forward", f"{float(forward):.14e}"),
        ("reverse", f"{reverse:.14e}"),
        ("scaled difference", f"{modes:.2e}"),
    )

    report.write_heading("Forward mode and reverse mode, through all three stages")
    report.write_entries(entries)

    return modes


def refusal_message(setup: ArchSetup, chain: Chain, catalogue: TubeCatalogue) -> str:
    """
    What the check says when asked to differentiate its own diagnostic.
    """
    nodes = jnp.asarray(setup.structure.nodes)
    edges = np.asarray(setup.structure.edges, dtype=np.int64)
    analysis_inputs = {
        "xyz": nodes,
        "diameter": setup.seed,
        "edges": edges,
        "supports": np.asarray(setup.structure.supports, dtype=np.int64),
        "loads": np.asarray(setup.structure.loads, dtype=np.float64),
        "f_y": STEEL.f_y,
        "e_mod": STEEL.e_mod,
        "density": STEEL.density,
        "ratio": catalogue.ratio,
        "normal": NORMAL,
    }
    member = apply_tesseract(chain.analysis, analysis_inputs)

    spans = nodes[edges[:, 1]] - nodes[edges[:, 0]]
    lengths = jnp.linalg.norm(spans, axis=1)

    def limit_states(axial_force):
        check_inputs = {
            "axial_force": axial_force,
            "end_moments_major": member["end_moments_major"],
            "end_moments_minor": member["end_moments_minor"],
            "lengths": lengths,
            "buckling_length": lengths,
            "f_y": STEEL.f_y,
            "e_mod": STEEL.e_mod,
            "density": STEEL.density,
            "gamma_m0": STEEL.gamma_m0,
            "gamma_m1": STEEL.gamma_m1,
            "ratio": catalogue.ratio,
            "alpha": STEEL.alpha,
            "diameter_min": catalogue.diameter_min,
            "section_class": 3,
            "resultant": True,
        }
        sized = apply_tesseract(chain.ec3, check_inputs)

        return jnp.sum(sized["governing"])

    try:
        jax.grad(limit_states)(member["axial_force"])
    except ValueError as error:
        return str(error).splitlines()[0]

    return "nothing was refused, which means the diagnostic is differentiable"


def report_served(
    report: ReportWriter,
    setup: ArchSetup,
    chain: Chain,
    catalogue: TubeCatalogue,
) -> float | None:
    """
    The same mass and gradient with two stages in containers, if asked for.

    Set `NORMAX_SERVED_OUTPUT` to a directory the container runtime can bind,
    which on macOS means one the file sharing settings reach. Building the two
    images first is a prerequisite; the analysis stays in process either way,
    since `smax` is not published and its image cannot be built.
    """
    directory = os.environ.get("NORMAX_SERVED_OUTPUT")
    if directory is None:
        skipped = "Served containers skipped; set NORMAX_SERVED_OUTPUT to run them"
        report.write_heading(skipped)

        return None

    def objective(stages):
        composed_problem = ComposedSetup(setup.structure, stages, STEEL, catalogue)

        def total(q):
            return mass_composed(
                q,
                setup.seed,
                composed_problem,
                normal=NORMAL,
                section_class=3,
            )

        return total

    imported = objective(chain)
    imported_of = eqx.filter_jit(imported)
    imported_gradient_of = eqx.filter_jit(jax.grad(imported))

    reference = timed(lambda: imported_of(setup.q))
    gradient = timed(lambda: imported_gradient_of(setup.q))

    with (
        Tesseract.from_image(f"{IMAGES[0]}:{VERSION}", output_path=directory) as first,
        Tesseract.from_image(f"{IMAGES[1]}:{VERSION}", output_path=directory) as third,
    ):
        served = Chain(formfinding=first, analysis=chain.analysis, ec3=third)
        crossing = objective(served)
        served_of = eqx.filter_jit(crossing)
        served_gradient_of = eqx.filter_jit(jax.grad(crossing))

        total = timed(lambda: served_of(setup.q))
        crossed = timed(lambda: served_gradient_of(setup.q))

    value_gap = relative(reference.result, total.result)
    gradient_gap = relative(gradient.result, crossed.result)
    seconds = f"{total.seconds:.3f} for a mass, {crossed.seconds:.3f} for a gradient"
    entries = (
        ("mass", f"{float(total.result):.14e}"),
        ("scaled difference", f"{value_gap:.2e}"),
        ("gradient difference", f"{gradient_gap:.2e}"),
        ("seconds", seconds),
    )

    report.write_heading("The same chain with form finding and the check in containers")
    report.write_entries(entries)

    return gradient_gap


def main(verbose: bool = True) -> None:
    """
    Run the pipeline across three schemas, and compare it against one process.
    """
    report = ReportWriter(verbose)
    setup = arch_setup()
    chain = local_chain()

    report_schemas(report, chain)

    report.write_heading("The same design, taken twice")
    worst = ParityWorst(0.0, 0.0, 0.0)
    for section_class in CLASSES:
        found = report_parity(report, setup, chain, section_class)
        worst = worst.worse_than(found)

    modes = report_modes(report, setup, chain)

    catalogue = TubeCatalogue.at_class_limit(STEEL.f_y, 3)
    refused = refusal_message(setup, chain, catalogue)
    entries = (("refused", refused),)

    report.write_heading("A cotangent on a non-differentiable output is refused")
    report.write_entries(entries)

    served = report_served(report, setup, chain, catalogue)

    checks = [
        ToleranceCheck("value error", worst.value, TOLERANCE_PARITY),
        ToleranceCheck("end moment error", worst.moment, TOLERANCE_MOMENT),
        ToleranceCheck("gradient error", worst.gradient, TOLERANCE_DERIVATIVE),
        ToleranceCheck("forward against reverse", modes, TOLERANCE_DERIVATIVE),
    ]
    if served is not None:
        crossing = ToleranceCheck("served against imported", served, TOLERANCE_SERVED)
        checks.append(crossing)

    report.write_heading("Summary")
    report.write_checks(checks)
    report.write_verdict(checks_passed(checks))


if __name__ == "__main__":
    main()
