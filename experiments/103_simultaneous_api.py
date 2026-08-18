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
The showcase arch, redesigned with the sizes as the optimizer's own variables.

**The same problem as `101_api.py`, formulated the other way.** There the check
is a solver: it bisects every member to a fully-stressed size inside the
pipeline, the analysis runs at frozen seed sections, and the answer is settled
to self-consistency afterwards. Here the diameters join the force densities as
decision variables, the check becomes the inequality constraint `U <= 1`, and
a constrained optimizer finds the fully-stressed state as active constraints —
self-consistent by construction, with no envelope and nothing to settle,
because there is one size per member and one constraint per load case instead
of one size per load case.

**Everything up to the formulation is borrowed, not copied.** The config file,
the arch, the load cases, the pipeline with its backend switch and the viewer
are `101_api.py`'s own functions, imported the way experiment 102 imports
them. What this file adds is the objective in `(q, d)`, the constraint slack
read off `AbstractMemberSizer.compute_utilization` — which is exactly a constraint
function's signature — and the SLSQP wiring with analytic Jacobians.

**Both backends fit the slot.** The blueprint check differentiates through a
hand-derived rule behind `jax.pure_callback`; the EC3 check traces, buckling
included, so the constrained search prices member stability on every step.
The same file, the same key in `arch.yaml`, two design philosophies.

Run with `uv run --group pipeline --group viz python
experiments/103_simultaneous_api.py [arch.yaml]`. The run ends in a viewer
holding the seed and the optimized designs, and returns when its window closes.
"""

import importlib.util
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from jaxtyping import Array
from jaxtyping import Float
from scipy.optimize import minimize

from normax.design import Design
from normax.design import DesignParameters
from normax.design import StructuralDesignPipeline
from normax.design import compute_mass
from normax.design import design_envelope
from normax.loads import LoadCases
from normax.optimization import Trajectory
from normax.optimization import penalized_mass
from normax.sizing import MemberSizes
from normax.sizing.blueprint import DIAMETER_MINIMUM
from normax.visualization import figure_trajectory

# The arch and the search, unless another file is named on the command line.
CONFIG = Path(__file__).with_name("arch.yaml")

FIGURES = Path(__file__).resolve().parent.parent / "figures"


class SimultaneousConfig(NamedTuple):
    """
    What the constrained search is allowed to spend.

    Attributes
    ----------
    iterations :
        Most iterations to spend.
    tolerance :
        Objective tolerance the solver stops at.
    """

    iterations: int
    tolerance: float


class ConstrainedProblem(NamedTuple):
    """
    The search as the constrained solver sees it: compiled maps and a start.

    Attributes
    ----------
    weigh :
        Compiled value and gradient of the penalized mass in `(q, d)`.
    slack :
        Compiled constraint slack, one minus the utilization, flattened.
    slack_jacobian :
        Compiled Jacobian of that slack.
    start :
        The seed force densities and diameters, concatenated.
    """

    weigh: object
    slack: object
    slack_jacobian: object
    start: Float[Array, "variables"]


class SearchAnswer(NamedTuple):
    """
    What the constrained search found, and what it spent.

    Attributes
    ----------
    force_densities :
        Force density of every member at the answer.
    diameters :
        Outer diameter of every member at the answer.
    trajectory :
        Force densities and objective at every iterate, for the figure.
    evaluations :
        Iterations and function evaluations the solver reported.
    elapsed :
        Wall-clock seconds of the solve, compilation excluded.
    """

    force_densities: Float[Array, "members"]
    diameters: Float[Array, "members"]
    trajectory: Trajectory
    evaluations: str
    elapsed: float


def load_showcase(path: Path) -> ModuleType:
    """
    The 101 experiment as a module, its digit-led name notwithstanding.

    Parameters
    ----------
    path :
        File the showcase experiment lives in.

    Returns
    -------
    module :
        The loaded module, whose builders this experiment reuses.
    """
    spec = importlib.util.spec_from_file_location("api_101", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def parse_simultaneous(text: str) -> SimultaneousConfig:
    """
    The constrained search's own section of the same file 101 reads.

    Parameters
    ----------
    text :
        Text of the file describing the run.

    Returns
    -------
    config :
        The budgets of the constrained search.
    """
    document = yaml.safe_load(text)

    return SimultaneousConfig(**document["simultaneous"])


def assemble_design(
    pipeline: StructuralDesignPipeline,
    loads: LoadCases,
    force_densities: Float[Array, "members"],
    diameters: Float[Array, "members"],
) -> Design:
    """
    A whole design at one point of the search, for the viewer and the reports.

    Parameters
    ----------
    pipeline :
        The three blocks, supplying the form finder, analyzer and check.
    loads :
        The form-finding case and the checked cases.
    force_densities :
        Force density of every member.
    diameters :
        Outer diameter of every member — the search's own, not a sizer's.

    Returns
    -------
    design :
        Shape, forces, and the given sections with their utilization.
    """
    shape = pipeline.formfinder(force_densities, loads.formfinding)
    forces = pipeline.analyzer(shape.xyz, diameters, loads.analysis)
    used = pipeline.sizer.compute_utilization(diameters, forces, shape.lengths)
    sections = pipeline.sizer.family(diameters)
    sizes = MemberSizes(sections, used)

    return Design(shape, forces, sizes)


def constrained_problem(
    pipeline: StructuralDesignPipeline,
    loads: LoadCases,
    params: DesignParameters,
    floor_length: float,
    floor: NamedTuple,
) -> ConstrainedProblem:
    """
    Compile the objective and the constraints the solver will call.

    Parameters
    ----------
    pipeline :
        The three blocks, differentiated through as one function.
    loads :
        The form-finding case and the checked cases.
    params :
        The seed force densities and diameters.
    floor_length :
        Shortest member the design is allowed.
    floor :
        The floor's sharpness and weight, from the shared config.

    Returns
    -------
    problem :
        Compiled value-and-gradient, slack and Jacobian, and the start point.

    Notes
    -----
    The mass is penalized by the same length floor 101 descends under, since
    moving the force densities invites the same member collapse either way.
    The check enters as constraints instead of as a solver, so there is no
    envelope: one diameter per member has to satisfy every case at once.
    """
    members = params.force_densities.shape[0]
    family = pipeline.sizer.family
    density = family.material.density

    def split_variables(x):
        return x[:members], x[members:]

    def weigh(x):
        force_densities, diameters = split_variables(x)
        shape = pipeline.formfinder(force_densities, loads.formfinding)
        sections = family(diameters)
        mass = jnp.sum(sections.area * shape.lengths) * density

        return penalized_mass(
            mass,
            shape.lengths,
            floor_length,
            beta=floor.sharpness,
            weight=floor.weight,
        )

    def slack(x):
        force_densities, diameters = split_variables(x)
        shape = pipeline.formfinder(force_densities, loads.formfinding)
        forces = pipeline.analyzer(shape.xyz, diameters, loads.analysis)
        used = pipeline.sizer.compute_utilization(diameters, forces, shape.lengths)

        return 1.0 - used.ravel()

    weigh_and_slope = jax.jit(jax.value_and_grad(weigh))
    slack_compiled = jax.jit(slack)
    slack_jacobian = jax.jit(jax.jacrev(slack))

    start = jnp.concatenate([params.force_densities, params.diameters])
    weigh_and_slope(start)
    slack_compiled(start)
    slack_jacobian(start)

    return ConstrainedProblem(weigh_and_slope, slack_compiled, slack_jacobian, start)


def solve_constrained(
    problem: ConstrainedProblem,
    bounds: NamedTuple,
    searched: SimultaneousConfig,
    members: int,
) -> SearchAnswer:
    """
    Spend the budget: SLSQP over shape and sizes, analytic Jacobians throughout.

    Parameters
    ----------
    problem :
        The compiled maps and the start point.
    bounds :
        The box the force densities may move in, from the shared config.
    searched :
        The budgets of the constrained search.
    members :
        Number of members, splitting the variable vector.

    Returns
    -------
    answer :
        The optimum, the visited iterates, and what the solve spent.

    Notes
    -----
    The diameters are floored at the same catalogue minimum every sizer
    clamps to, as a bound rather than a constraint: a bound never needs a
    multiplier, and the fully-stressed condition stays readable off the
    constraint activities alone.
    """

    def objective(x):
        value, slope = problem.weigh(jnp.asarray(x))

        return float(value), np.asarray(slope, dtype=np.float64)

    def feasible(x):
        return np.asarray(problem.slack(jnp.asarray(x)), dtype=np.float64)

    def feasible_jacobian(x):
        return np.asarray(problem.slack_jacobian(jnp.asarray(x)), dtype=np.float64)

    visited = [np.asarray(problem.start, dtype=np.float64)]

    def record_step(x):
        visited.append(np.asarray(x, dtype=np.float64))

    held = {"type": "ineq", "fun": feasible, "jac": feasible_jacobian}
    force_box = [(bounds.min, bounds.max)] * members
    size_box = [(DIAMETER_MINIMUM, None)] * members

    started = time.perf_counter()
    found = minimize(
        objective,
        np.asarray(problem.start),
        jac=True,
        method="SLSQP",
        bounds=force_box + size_box,
        constraints=[held],
        callback=record_step,
        options={"maxiter": searched.iterations, "ftol": searched.tolerance},
    )
    elapsed = time.perf_counter() - started

    steps = np.stack(visited)
    weighed = [float(problem.weigh(jnp.asarray(step))[0]) for step in visited]
    trajectory = Trajectory(
        jnp.asarray(steps[:, :members]),
        jnp.asarray(weighed),
        jnp.zeros(len(visited)),
    )

    spent = f"{found.nit} iterations, {found.nfev} evaluations"
    answer = SearchAnswer(
        jnp.asarray(found.x[:members]),
        jnp.asarray(found.x[members:]),
        trajectory,
        spent,
        elapsed,
    )

    return answer


def main(config_path: Path) -> None:
    """
    Redesign the arch with the sizes as variables, and report what it bought.

    Parameters
    ----------
    config_path :
        File naming the arch and the settings, shared verbatim with 101.
    """
    api = load_showcase(Path(__file__).with_name("101_api.py"))

    config_text = config_path.read_text()
    config = api.parse_config(config_text)
    searched = parse_simultaneous(config_text)
    structure = api.build_arch(config.structure)
    loads = api.arch_load_cases(structure, config.load_cases)
    pipeline = api.build_pipeline(structure, config)
    params = api.initialize_parameters(structure, config)
    print(f"Sizing backend: {config.sizing.backend}")

    floor = config.optimization.length_floor
    floor_length = floor.fraction * config.structure.span / config.structure.num_edges

    # The baseline 101 starts from: the seed shape, sized by the check and
    # enveloped at the same sharpness, so the saving below is comparable.
    seeded = pipeline(params, loads)
    sharpness = jnp.asarray(config.optimization.envelope_sharpness)
    baseline = compute_mass(design_envelope(seeded, sharpness))

    problem = constrained_problem(pipeline, loads, params, floor_length, floor)
    answer = solve_constrained(
        problem, config.optimization.bounds, searched, structure.num_edges
    )

    optimized = assemble_design(
        pipeline, loads, answer.force_densities, answer.diameters
    )
    mass_opt = compute_mass(optimized)
    used = np.asarray(optimized.sizes.utilization)
    worked = np.max(used, axis=0)
    slackness = np.asarray(
        problem.slack(jnp.concatenate([answer.force_densities, answer.diameters]))
    )

    lengths = optimized.shape.lengths
    shortest = float(jnp.min(lengths))
    stubs = int(jnp.sum(lengths < floor_length))

    print(f"Mass of the sized seed design: {float(baseline):.9f} t")
    print(f"Mass after the direct search: {float(mass_opt):.9f} t")
    print(f"Saved: {100.0 * (1.0 - mass_opt / baseline):.3f} %")
    print(f"Worst constraint violation: {float(np.max(-slackness)):.3e}")
    print(
        f"Members worked to one: {int(np.sum(worked > 1.0 - 1e-6))}"
        f" of {structure.num_edges}"
    )
    print(
        f"Shortest member: {shortest:.1f} mm against a floor of {floor_length:.1f} mm"
    )
    print(f"Members under the floor: {stubs} of {structure.num_edges}")
    print(f"Solver spent: {answer.evaluations} in {answer.elapsed:.3f} s")
    print("Nothing to settle: the analysis ran at the answer's own sections.")

    trajectories = (answer.trajectory,)
    titles = ("constrained search",)
    descent_figure = figure_trajectory(trajectories, titles=titles)
    FIGURES.mkdir(exist_ok=True)
    descent_figure.savefig(FIGURES / "103_trajectory.png", dpi=200)
    print(f"Descent figure: {FIGURES / '103_trajectory.png'}")

    initial = assemble_design(pipeline, loads, params.force_densities, params.diameters)
    designs = {"seed": initial, "optimized": optimized}
    case_names = tuple(load_case.name for load_case in config.load_cases)
    api.view_designs(structure, pipeline.analyzer, loads, designs, case_names)

    print("\nHasta la vista, baby!")


if __name__ == "__main__":
    jnp.set_printoptions(precision=12)
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else CONFIG)
