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

Run with `uv run --group pipeline python experiments/10_arch_pipeline_tesseract.py`.
"""

import os
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
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
from normax.structures import arch_2d

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

# The diameter the frame is analysed with before the check has spoken.
SEED = 100.0

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


def setup():
    """
    The arch, its form-finding connectivity, and the `q` that reaches the rise.
    """
    load = TOTAL_LOAD / (NUM_EDGES - 1)
    structure = arch_2d(num_edges=NUM_EDGES, span=SPAN, rise=RISE, load=load)
    graph_fdm = equilibrium_graph(structure)

    trial = jnp.full(NUM_EDGES, -1.0)
    reached = jnp.max(equilibrium_state(trial, structure, graph_fdm).xyz[:, 2])

    return structure, graph_fdm, trial * reached / RISE


def named_fields(container):
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


def relative(oracle, composed):
    """
    Largest disagreement between two arrays, scaled by the size of the first.
    """
    left = np.asarray(oracle, dtype=np.float64)
    right = np.asarray(composed, dtype=np.float64)
    scale = max(float(np.max(np.abs(left))), np.finfo(np.float64).tiny)

    return float(np.max(np.abs(left - right))) / scale


def timed(call):
    """
    A call's result and how long it took, the first call excluded as warm-up.
    """
    call()
    start = time.perf_counter()
    result = call()

    return result, time.perf_counter() - start


def report_schemas(chain):
    """
    What every stage carries, and what it will differentiate.
    """
    print("The three schemas")
    for stage, tesseract in zip(chain._fields, chain):
        schemas = tesseract.openapi_schema["components"]["schemas"]
        inputs = schemas["Apply_InputSchema"]["properties"]
        outputs = schemas["Apply_OutputSchema"]["properties"]
        wrt = schemas["ApplyInputSchema"]["differentiable_arrays"]
        of = schemas["ApplyOutputSchema"]["differentiable_arrays"]

        print(f"\n  {stage}")
        print(f"    in          {', '.join(sorted(inputs))}")
        print(f"    out         {', '.join(sorted(outputs))}")
        print(f"    d/d         {', '.join(sorted(wrt))}")
        print(f"    d of        {', '.join(sorted(of))}")
        print(f"    not diff    {', '.join(sorted(set(outputs) - set(of))) or '-'}")


def refusal(chain, structure, seed, catalogue):
    """
    What the check says when asked to differentiate its own diagnostic.
    """
    member = apply_tesseract(
        chain.analysis,
        {
            "xyz": jnp.asarray(structure.nodes),
            "diameter": seed,
            "edges": np.asarray(structure.edges, dtype=np.int64),
            "supports": np.asarray(structure.supports, dtype=np.int64),
            "loads": np.asarray(structure.loads, dtype=np.float64),
            "f_y": STEEL.f_y,
            "e_mod": STEEL.e_mod,
            "density": STEEL.density,
            "ratio": catalogue.ratio,
            "normal": NORMAL,
        },
    )
    lengths = jnp.linalg.norm(
        jnp.asarray(structure.nodes)[structure.edges[:, 1]]
        - jnp.asarray(structure.nodes)[structure.edges[:, 0]],
        axis=1,
    )

    def limit_states(axial_force):
        sized = apply_tesseract(
            chain.ec3,
            {
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
            },
        )

        return jnp.sum(sized["governing"])

    try:
        jax.grad(limit_states)(member["axial_force"])
    except ValueError as error:
        return str(error).splitlines()[0]

    return "nothing was refused, which means the diagnostic is differentiable"


def report_served(chain, structure, q, seed, catalogue):
    """
    The same mass and gradient with two stages in containers, if asked for.

    Set `NORMAX_SERVED_OUTPUT` to a directory the container runtime can bind,
    which on macOS means one the file sharing settings reach. Building the two
    images first is a prerequisite; the analysis stays in process either way,
    since `smax` is not published and its image cannot be built.
    """
    directory = os.environ.get("NORMAX_SERVED_OUTPUT")
    if directory is None:
        print("\nServed containers skipped; set NORMAX_SERVED_OUTPUT to run them")
        return None

    print("\nThe same chain with form finding and the check in containers")

    def objective(stages):
        return lambda q: mass_composed(
            q,
            seed,
            ComposedSetup(structure, stages, STEEL, catalogue),
            normal=NORMAL,
            section_class=3,
        )

    reference, _ = timed(lambda: objective(chain)(q))
    gradient, _ = timed(lambda: jax.grad(objective(chain))(q))

    with (
        Tesseract.from_image(f"{IMAGES[0]}:{VERSION}", output_path=directory) as first,
        Tesseract.from_image(f"{IMAGES[1]}:{VERSION}", output_path=directory) as third,
    ):
        served = Chain(formfinding=first, analysis=chain.analysis, ec3=third)
        total, seconds_mass = timed(lambda: objective(served)(q))
        crossed, seconds_gradient = timed(lambda: jax.grad(objective(served))(q))

    print(f"  mass                {float(total):.14e}")
    print(f"  scaled difference   {relative(reference, total):.2e}")
    print(f"  gradient difference {relative(gradient, crossed):.2e}")
    print(
        f"  seconds             {seconds_mass:.3f} for a mass,"
        f" {seconds_gradient:.3f} for a gradient"
    )

    return relative(gradient, crossed)


def main():
    structure, graph_fdm, q = setup()
    seed = jnp.full(NUM_EDGES, SEED)
    chain = local_chain()

    report_schemas(chain)

    print("\nThe same design, taken twice")
    worst_value = 0.0
    worst_moment = 0.0
    worst_gradient = 0.0

    for section_class in (2, 3):
        catalogue = TubeCatalogue.at_class_limit(STEEL.f_y, section_class)
        model = prepare_model(structure, STEEL, catalogue, normal=NORMAL)
        problem = ProblemSetup(structure, graph_fdm, model, STEEL, catalogue)
        composed_problem = ComposedSetup(structure, chain, STEEL, catalogue)

        oracle, seconds_oracle = timed(
            lambda problem=problem, section_class=section_class: design_compiled(
                q,
                seed,
                problem,
                section_class=section_class,
            )
        )
        composed, seconds_chain = timed(
            lambda composed_problem=composed_problem, section_class=section_class: (
                design_composed(
                    q,
                    seed,
                    composed_problem,
                    normal=NORMAL,
                    section_class=section_class,
                )
            )
        )

        print(f"\n  Class {section_class}, d/t = {float(catalogue.ratio):.3f}")
        print(
            f"    {'field':>27} {'in process':>22} {'composed':>22}"
            f" {'scaled':>10} {'held to':>9}"
        )
        for (label, oracle_leaf), (_, composed_leaf) in zip(
            named_fields(oracle), named_fields(composed)
        ):
            left = np.asarray(oracle_leaf, dtype=np.float64)
            right = np.asarray(composed_leaf, dtype=np.float64)
            scaled = relative(left, right)

            leaf = label.rpartition(".")[2]
            if leaf in MOMENT_FIELDS:
                limit = TOLERANCE_MOMENT
                worst_moment = max(worst_moment, scaled)
            else:
                limit = TOLERANCE_PARITY
                worst_value = max(worst_value, scaled)

            print(
                f"    {label:>27} {float(left.ravel()[0]):>22.14e}"
                f" {float(right.ravel()[0]):>22.14e} {scaled:>10.2e}"
                f" {limit:>9.0e}"
            )

        codes = governing_states(oracle, problem, section_class=section_class)
        limits = {LIMIT_NAMES[float(code)] for code in codes}
        print(f"    governing    {', '.join(sorted(limits))}")
        print(f"    mass         {float(composed.mass):.12f} t")
        departure = float(jnp.max(jnp.abs(composed.utilization - 1.0)))
        print(f"    worst |u-1|  {departure:.2e}")
        print(
            f"    seconds      {seconds_oracle:.4f} in process,"
            f" {seconds_chain:.4f} composed, both compiled"
        )

        def in_process(q, problem=problem, section_class=section_class):
            return mass_compiled(
                q,
                seed,
                problem,
                section_class=section_class,
            )

        def composed_mass(
            q, composed_problem=composed_problem, section_class=section_class
        ):
            return mass_composed(
                q,
                seed,
                composed_problem,
                normal=NORMAL,
                section_class=section_class,
            )

        exact, seconds_oracle = timed(lambda: jax.grad(in_process)(q))
        crossed, seconds_chain = timed(lambda: jax.grad(composed_mass)(q))

        print(f"\n    {'edge':>5} {'in process':>22} {'composed':>22} {'scaled':>10}")
        for edge in range(NUM_EDGES):
            scaled = abs(float(exact[edge]) - float(crossed[edge])) / float(
                jnp.max(jnp.abs(exact))
            )
            worst_gradient = max(worst_gradient, scaled)
            print(
                f"    {edge:>5} {float(exact[edge]):>22.14e}"
                f" {float(crossed[edge]):>22.14e} {scaled:>10.2e}"
            )
        print(f"    sum          {float(jnp.sum(crossed)):.14e}")
        print(
            f"    seconds      {seconds_oracle:.4f} in process,"
            f" {seconds_chain:.4f} composed, both compiled"
        )

    print("\nForward mode and reverse mode, through all three stages")
    catalogue = TubeCatalogue.at_class_limit(STEEL.f_y, 3)

    def objective(q):
        return mass_composed(
            q,
            seed,
            ComposedSetup(structure, chain, STEEL, catalogue),
            normal=NORMAL,
            section_class=3,
        )

    direction = jnp.ones_like(q)
    _, forward = jax.jvp(objective, (q,), (direction,))
    reverse = float(jnp.sum(jax.grad(objective)(q) * direction))
    modes = relative(reverse, forward)
    print(f"  forward             {float(forward):.14e}")
    print(f"  reverse             {reverse:.14e}")
    print(f"  scaled difference   {modes:.2e}")

    print("\nA cotangent on a non-differentiable output is refused")
    refused = refusal(chain, structure, seed, catalogue)
    print(f"  {refused}")

    served = report_served(chain, structure, q, seed, catalogue)

    print("\nSummary")
    print(
        f"  worst value error       {worst_value:.2e}  (target {TOLERANCE_PARITY:.0e})"
    )
    print(
        f"  worst end moment error  {worst_moment:.2e}  (target {TOLERANCE_MOMENT:.0e})"
    )
    print(
        f"  worst gradient error    {worst_gradient:.2e}"
        f"  (target {TOLERANCE_DERIVATIVE:.0e})"
    )
    print(f"  forward against reverse {modes:.2e}  (target {TOLERANCE_DERIVATIVE:.0e})")
    if served is not None:
        print(
            f"  served against imported {served:.2e}  (target {TOLERANCE_SERVED:.0e})"
        )

    passed = (
        worst_value < TOLERANCE_PARITY
        and worst_moment < TOLERANCE_MOMENT
        and worst_gradient < TOLERANCE_DERIVATIVE
        and modes < TOLERANCE_DERIVATIVE
        and (served is None or served < TOLERANCE_SERVED)
    )
    print(f"\n{'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
