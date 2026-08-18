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
function's signature — and the SLSQP wiring with analytic Jacobians. The
`simultaneous` section of the config picks the formulation: `force_densities`
moves one value per member or a single shared one, `length_floor` holds every
member at or above the floor as a hard inequality, and `fixed_rise` pins the
crown at the seed design's form-found rise as an equality. The mass is always
descended bare: geometry is held by the solver, never by a penalty.

**Every backend fits the slot.** The EC3 check traces, buckling included, so
the constrained search prices member stability on every step; the blueprint
check differentiates through a hand-derived rule behind `jax.pure_callback`;
and `blueprint_tesseract` reaches that same check across a Tesseract
boundary, so every constraint evaluation is a crossing and every constraint
Jacobian pulls its rows one by one through a hand-written NumPy adjoint.
The same file, the same key in `arch.yaml`.

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
from normax.form_finding import FormFoundShape
from normax.loads import LoadCases
from normax.optimization import Trajectory
from normax.sizing import MemberSizes
from normax.sizing.blueprint import DIAMETER_MINIMUM
from normax.structures import Structure
from normax.visualization import figure_trajectory

# The arch and the search, unless another file is named on the command line.
CONFIG = Path(__file__).with_name("arch.yaml")

FIGURES = Path(__file__).resolve().parent.parent / "figures"

# The 2D arch rises along Z; see `build_arch_2d`.
VERTICAL_AXIS = 2


class SimultaneousConfig(NamedTuple):
    """
    What the constrained search is allowed to spend, and how it moves the form.

    Attributes
    ----------
    iterations :
        Most iterations to spend.
    tolerance :
        Objective tolerance the solver stops at.
    force_densities :
        Whether every member's force density moves alone or one shared value
        moves them all: ``independent`` or ``shared``.
    length_floor :
        Whether member lengths are held at or above the floor, as hard
        inequality constraints.
    fixed_rise :
        Whether the crown is held at the seed design's form-found rise, as an
        equality constraint.
    """

    iterations: int
    tolerance: float
    force_densities: str = "independent"
    length_floor: bool = False
    fixed_rise: bool = False


class VariableLayout(NamedTuple):
    """
    How the solver's flat vector maps onto force densities and diameters.

    Attributes
    ----------
    members :
        Number of members, each with its own diameter variable.
    densities :
        Number of force-density variables: the member count, or one shared.
    """

    members: int
    densities: int


def variable_layout(parametrization: str, members: int) -> VariableLayout:
    """
    Size the variable blocks for the parametrization named in the config.

    Parameters
    ----------
    parametrization :
        Name of the force-density parametrization: ``independent`` moves one
        variable per member, ``shared`` moves a single value for all.
    members :
        Number of members in the structure.

    Returns
    -------
    layout :
        The sizes of the two variable blocks.
    """
    if parametrization == "independent":
        return VariableLayout(members, members)
    if parametrization == "shared":
        return VariableLayout(members, 1)

    raise ValueError(f"Unknown force-density parametrization: {parametrization!r}")


def spread_variables(
    layout: VariableLayout,
    variables: Float[Array, "variables"],
) -> tuple[Float[Array, "members"], Float[Array, "members"]]:
    """
    Split the flat vector into per-member force densities and diameters.

    Parameters
    ----------
    layout :
        The sizes of the two variable blocks.
    variables :
        The solver's flat vector, force densities first.

    Returns
    -------
    force_densities :
        Force density of every member, a shared value broadcast across all.
    diameters :
        Outer diameter of every member.
    """
    densities = variables[: layout.densities]
    force_densities = jnp.broadcast_to(densities, (layout.members,))
    diameters = variables[layout.densities :]

    return force_densities, diameters


class ShapeConstraints(NamedTuple):
    """
    The geometric conditions fed to the solver beside the code check.

    Attributes
    ----------
    floor_active :
        Whether member lengths are held at or above the floor.
    floor_length :
        Shortest member the design is allowed.
    rise_active :
        Whether the crown is held at the seed design's rise.
    crown_node :
        Node the rise is measured at.
    rise_target :
        Height the crown is held at, read off the form-found seed.
    """

    floor_active: bool
    floor_length: float
    rise_active: bool
    crown_node: int
    rise_target: float


def shape_constraints(
    searched: SimultaneousConfig,
    structure: Structure,
    seed_shape: FormFoundShape,
    floor_length: float,
) -> ShapeConstraints:
    """
    Read the geometry toggles named in the config, anchored at the seed.

    Parameters
    ----------
    searched :
        The switches of the constrained search.
    structure :
        The arch, naming its crown node.
    seed_shape :
        The form-found seed, whose rise the crown may be held at.
    floor_length :
        Shortest member the design is allowed.

    Returns
    -------
    constraints :
        The geometric conditions and their anchors.
    """
    crown = structure.crown_node()
    rise_target = float(seed_shape.xyz[crown, VERTICAL_AXIS])

    return ShapeConstraints(
        searched.length_floor,
        floor_length,
        searched.fixed_rise,
        crown,
        rise_target,
    )


class GeometryConstraint(NamedTuple):
    """
    One compiled geometric condition, in the solver's own vocabulary.

    Attributes
    ----------
    kind :
        SLSQP constraint type: ``ineq`` holds the residual at or above zero,
        ``eq`` holds it at zero.
    residual :
        Compiled residual map of the flat variable vector.
    jacobian :
        Compiled Jacobian of that residual.
    """

    kind: str
    residual: object
    jacobian: object


def scipy_constraint(geometry: GeometryConstraint) -> dict[str, object]:
    """
    Wrap a compiled geometric condition the way SLSQP expects it.

    Parameters
    ----------
    geometry :
        The condition to wrap.

    Returns
    -------
    constraint :
        The dictionary ``scipy.optimize.minimize`` consumes.
    """

    def fun(x):
        return np.asarray(geometry.residual(jnp.asarray(x)), dtype=np.float64)

    def jac(x):
        return np.asarray(geometry.jacobian(jnp.asarray(x)), dtype=np.float64)

    return {"type": geometry.kind, "fun": fun, "jac": jac}


class ConstrainedProblem(NamedTuple):
    """
    The search as the constrained solver sees it: compiled maps and a start.

    Attributes
    ----------
    weigh :
        Compiled value and gradient of the mass in `(q, d)`.
    slack :
        Compiled constraint slack, one minus the utilization, flattened.
    slack_jacobian :
        Compiled Jacobian of that slack.
    geometry :
        The compiled geometric conditions the config activated.
    start :
        The seed force densities and diameters, concatenated.
    """

    weigh: object
    slack: object
    slack_jacobian: object
    geometry: tuple[GeometryConstraint, ...]
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
    variables :
        The answer in the solver's own coordinates, for re-evaluating maps.
    trajectory :
        Force densities and objective at every iterate, for the figure.
    evaluations :
        Iterations and function evaluations the solver reported.
    elapsed :
        Wall-clock seconds of the solve, compilation excluded.
    """

    force_densities: Float[Array, "members"]
    diameters: Float[Array, "members"]
    variables: Float[Array, "variables"]
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
    layout: VariableLayout,
    constraints: ShapeConstraints,
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
    layout :
        The sizes of the two variable blocks.
    constraints :
        The geometric conditions the config activated, and their anchors.

    Returns
    -------
    problem :
        Compiled value-and-gradient, slack, Jacobian, geometry and start.

    Notes
    -----
    The mass is descended bare. The geometry the penalty used to guard is
    held by the solver instead: the length floor as inequalities, the crown
    rise as an equality, each with its own compiled Jacobian. The check
    enters as constraints instead of as a solver, so there is no envelope:
    one diameter per member has to satisfy every case at once. A shared
    force density reaches the pipeline broadcast across the members, so its
    gradient arrives as the sum of theirs.
    """
    family = pipeline.sizer.family
    density = family.material.density

    def weigh(x):
        force_densities, diameters = spread_variables(layout, x)
        shape = pipeline.formfinder(force_densities, loads.formfinding)
        sections = family(diameters)
        mass = jnp.sum(sections.area * shape.lengths) * density

        return mass

    def slack(x):
        force_densities, diameters = spread_variables(layout, x)
        shape = pipeline.formfinder(force_densities, loads.formfinding)
        forces = pipeline.analyzer(shape.xyz, diameters, loads.analysis)
        used = pipeline.sizer.compute_utilization(diameters, forces, shape.lengths)

        return 1.0 - used.ravel()

    def floor_slack(x):
        force_densities, _ = spread_variables(layout, x)
        shape = pipeline.formfinder(force_densities, loads.formfinding)

        return shape.lengths - constraints.floor_length

    def rise_residual(x):
        force_densities, _ = spread_variables(layout, x)
        shape = pipeline.formfinder(force_densities, loads.formfinding)
        rise = shape.xyz[constraints.crown_node, VERTICAL_AXIS]

        return jnp.atleast_1d(rise - constraints.rise_target)

    weigh_and_slope = jax.jit(jax.value_and_grad(weigh))
    slack_compiled = jax.jit(slack)
    slack_jacobian = jax.jit(jax.jacrev(slack))

    geometry = []
    if constraints.floor_active:
        held_floor = GeometryConstraint(
            "ineq", jax.jit(floor_slack), jax.jit(jax.jacrev(floor_slack))
        )
        geometry.append(held_floor)
    if constraints.rise_active:
        held_rise = GeometryConstraint(
            "eq", jax.jit(rise_residual), jax.jit(jax.jacrev(rise_residual))
        )
        geometry.append(held_rise)

    # The seed is uniform, so its first entries seed the shared layout too.
    seed_densities = params.force_densities[: layout.densities]
    start = jnp.concatenate([seed_densities, params.diameters])
    weigh_and_slope(start)
    slack_compiled(start)
    slack_jacobian(start)
    for held in geometry:
        held.residual(start)
        held.jacobian(start)

    return ConstrainedProblem(
        weigh_and_slope, slack_compiled, slack_jacobian, tuple(geometry), start
    )


def solve_constrained(
    problem: ConstrainedProblem,
    bounds: NamedTuple,
    searched: SimultaneousConfig,
    layout: VariableLayout,
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
    layout :
        The sizes of the two variable blocks.

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
    shaped = [scipy_constraint(geometry) for geometry in problem.geometry]
    force_box = [(bounds.min, bounds.max)] * layout.densities
    size_box = [(DIAMETER_MINIMUM, None)] * layout.members

    started = time.perf_counter()
    found = minimize(
        objective,
        np.asarray(problem.start),
        jac=True,
        method="SLSQP",
        bounds=force_box + size_box,
        constraints=[held, *shaped],
        callback=record_step,
        options={"maxiter": searched.iterations, "ftol": searched.tolerance},
    )
    elapsed = time.perf_counter() - started

    steps = np.stack(visited)
    weighed = [float(problem.weigh(jnp.asarray(step))[0]) for step in visited]
    # A shared force density is spread per member, keeping the figure's axes.
    walked = np.broadcast_to(
        steps[:, : layout.densities], (steps.shape[0], layout.members)
    )
    trajectory = Trajectory(
        jnp.asarray(walked),
        jnp.asarray(weighed),
        jnp.zeros(len(visited)),
    )

    variables = jnp.asarray(found.x)
    force_densities, diameters = spread_variables(layout, variables)
    spent = f"{found.nit} iterations, {found.nfev} evaluations"
    answer = SearchAnswer(
        force_densities,
        diameters,
        variables,
        trajectory,
        spent,
        elapsed,
    )

    return answer


def tabulate_members(
    diameters_seed: Float[Array, "members"],
    diameters_found: Float[Array, "members"],
    worked_seed: Float[np.ndarray, "members"],
    worked_found: Float[np.ndarray, "members"],
) -> None:
    """
    Print every member's size and worst-case utilization, seed and answer.

    Parameters
    ----------
    diameters_seed :
        Outer diameter of every member at the seed.
    diameters_found :
        Outer diameter of every member at the answer.
    worked_seed :
        Worst utilization of every member across the cases, at the seed.
    worked_found :
        Worst utilization of every member across the cases, at the answer.

    Notes
    -----
    The seed columns are at the seed diameters, not at a sizer's answer, so
    they read as the feasibility of the guess rather than as a design.
    """
    print("Members, worst case across load cases:")
    header = (
        f"{'member':>6}  {'diameter start':>14}  {'diameter optimized':>18}"
        f"  {'utilization start':>17}  {'utilization optimized':>21}"
    )
    print(header)
    rows = zip(
        np.asarray(diameters_seed),
        np.asarray(diameters_found),
        np.asarray(worked_seed),
        np.asarray(worked_found),
        strict=True,
    )
    for member, (d_seed, d_opt, u_seed, u_opt) in enumerate(rows):
        print(
            f"{member:>6}  {d_seed:>14.2f}  {d_opt:>18.2f}"
            f"  {u_seed:>17.6f}  {u_opt:>21.6f}"
        )


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
    layout = variable_layout(searched.force_densities, structure.num_edges)

    floor = config.optimization.length_floor
    floor_length = floor.fraction * config.structure.span / config.structure.num_edges
    seed_shape = pipeline.formfinder(params.force_densities, loads.formfinding)
    constraints = shape_constraints(searched, structure, seed_shape, floor_length)

    floor_state = "constrained" if constraints.floor_active else "free"
    rise_state = "fixed" if constraints.rise_active else "free"
    print(f"Sizing backend: {config.sizing.backend}")
    print(f"Analysis backend: {config.analysis.backend}")
    print(f"Force densities: {searched.force_densities}")
    print(f"Member lengths: {floor_state}")
    print(f"Crown rise: {rise_state} (seed at {constraints.rise_target:.1f} mm)")

    # The baseline 101 starts from: the seed shape, sized by the check and
    # enveloped at the same sharpness, so the saving below is comparable.
    seeded = pipeline(params, loads)
    sharpness = jnp.asarray(config.optimization.envelope_sharpness)
    baseline = compute_mass(design_envelope(seeded, sharpness))

    problem = constrained_problem(pipeline, loads, params, layout, constraints)
    answer = solve_constrained(problem, config.optimization.bounds, searched, layout)

    optimized = assemble_design(
        pipeline, loads, answer.force_densities, answer.diameters
    )
    initial = assemble_design(pipeline, loads, params.force_densities, params.diameters)
    mass_opt = compute_mass(optimized)
    mass_drawn = compute_mass(initial)
    used = np.asarray(optimized.sizes.utilization)
    worked = np.max(used, axis=0)
    worked_seed = np.max(np.asarray(initial.sizes.utilization), axis=0)
    slackness = np.asarray(problem.slack(answer.variables))

    lengths = optimized.shape.lengths
    shortest = float(jnp.min(lengths))
    stubs = int(jnp.sum(lengths < floor_length))

    # The drawn seed is the structure as guessed, every member at the seed
    # diameter, unsized — bulk the sizer strips before the baseline is set.
    print(f"Mass of the drawn seed design, unsized: {float(mass_drawn):.9f} t")
    print(f"Mass of the sized seed design: {float(baseline):.9f} t")
    print(f"Mass after the direct search: {float(mass_opt):.9f} t")
    print(f"Saved vs the sized seed: {100.0 * (1.0 - mass_opt / baseline):.3f} %")
    print(f"Saved vs the drawn seed: {100.0 * (1.0 - mass_opt / mass_drawn):.3f} %")
    print(f"Worst constraint violation: {float(np.max(-slackness)):.3e}")
    print(
        f"Members worked to one: {int(np.sum(worked > 1.0 - 1e-6))}"
        f" of {structure.num_edges}"
    )
    tabulate_members(params.diameters, answer.diameters, worked_seed, worked)
    print(
        f"Shortest member: {shortest:.1f} mm against a floor of {floor_length:.1f} mm"
        f" ({floor_state})"
    )
    print(f"Members under the floor: {stubs} of {structure.num_edges}")
    rise_opt = float(optimized.shape.xyz[constraints.crown_node, VERTICAL_AXIS])
    print(
        f"Crown rise: seed {constraints.rise_target:.1f} mm"
        f" -> optimized {rise_opt:.1f} mm ({rise_state})"
    )
    print(f"Optimizer spent: {answer.evaluations} in {answer.elapsed:.3f} s")
    print("Nothing to settle: the analysis ran at the answer's own sections.")

    trajectories = (answer.trajectory,)
    titles = ("constrained search",)
    descent_figure = figure_trajectory(trajectories, titles=titles)
    FIGURES.mkdir(exist_ok=True)
    descent_figure.savefig(FIGURES / "103_trajectory.png", dpi=200)
    print(f"Descent figure: {FIGURES / '103_trajectory.png'}")

    designs = {"seed": initial, "optimized": optimized}
    case_names = tuple(load_case.name for load_case in config.load_cases)
    api.view_designs(structure, pipeline.analyzer, loads, designs, case_names)

    print("\nHasta la vista, baby!")


if __name__ == "__main__":
    jnp.set_printoptions(precision=12)
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else CONFIG)
