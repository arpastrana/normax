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
Which constrained solver reaches the end-to-end answer for the fewest seconds.

The 16x16 gridshell spends 98% of its wall clock inside one call: the
constraint Jacobian, one forward-mode pass per variable. No change of solver
makes that call cheaper, so a solver is worth swapping only if it asks for the
call fewer times. SLSQP asked 4060 times on the 16x16 cap and still stopped at
its iteration limit, which is the measurement this spike answers.

Two drivers over the identical maps, from the identical start:

    SLSQP       scipy's dense active-set sequential quadratic program, the
                incumbent every search in experiment 23 is descended by.
    augmented   an augmented Lagrangian whose rows are folded into the
                objective before it is differentiated, so a gradient is one
                reverse pass rather than a forward tangent per variable, then
                a short SLSQP run from its landing for the certificate.

The second is the only one that changes what is asked of the maps rather than
how the answers are used. Everything that reads `jacobian` is bounded below by
the call that dominates the clock; folding the rows into the objective deletes
the call instead. What it gives up is the solver's own stopping certificate,
which the polish restores for the price of a handful of iterations.

Two more are available behind `--separable`, and are off by default because
the question they answered is closed:

    MMA         nlopt's method of moving asymptotes, the separable convex
                approximation structural sizing is usually written against.
    CCSA        nlopt's conservative convex separable approximation, MMA's
                globally convergent sibling.

Both converge strictly inside the feasible region, which a fully-stressed
optimum is not in.

What is reported per driver: how many times each map was called, the wall
clock, the mass reached, and the worst utilization the answer shows when it is
read back against **every** load case. The last of these is what makes the
comparison a measurement rather than a race — a driver that arrives lighter by
leaving a constraint behind has not arrived.

Only the end-to-end search is driven, and only from the nominal start. A
multi-start comparison prices the landscape, and the landscape is not what is
in question here.

Any structural family can be driven, the profile being read out of whichever
experiment exports one. The four drivers are not equally worth measuring on
every family: the augmented one buys its advantage on the constraint Jacobian,
which is the whole clock on a shell of five hundred members and a fraction of
it on a truss of thirty, so a truss race prices the answer rather than the
seconds.

Run with `uv run --group pipeline --extra spike python
experiments/24_mma_spike.py [run.yaml] [profile.py]`.
"""

import importlib.util
import sys
import time
from pathlib import Path
from typing import Callable
from typing import NamedTuple

import jax.numpy as jnp
import nlopt
import numpy as np
import yaml
from jaxtyping import Float
from normax.searches import AUGMENTED_DEFAULT
from normax.searches import POLISH_ADMISSION
from normax.searches import POLISH_ITERATIONS
from normax.searches import SEARCH_FORMFOUND
from normax.searches import StructureProfile
from normax.searches import augmented_budget
from normax.searches import folding_maps
from normax.searches import prepare_problem
from normax.searches import read_answer
from normax.searches import search_boxes
from normax.searches import search_maps
from normax.searches import search_starts
from scipy.optimize import minimize

from normax.optimization import ConstrainedMaps
from normax.optimization import OptimizationBudget
from normax.optimization import descend_augmented
from normax.reporting import Report
from normax.reporting import ReportColumn

# The profile driven when a run names no other.
EXPERIMENT = Path(__file__).resolve().parents[2] / "examples" / "gridshell.py"

# Budgets the three drivers share. Evaluations rather than iterations, which
# is the only currency all three count in the same units.
EVALUATIONS = 3000
TOLERANCE_RELATIVE = 1.0e-6

# How far under one a converged answer must hold every utilization. The shared
# flow's own feasibility bound, so a spike answer is judged as a run's is.
TOLERANCE_FEASIBILITY = 1.0e-6


class DriverCall(NamedTuple):
    """
    One solver's landing, and what it spent reaching it.

    Attributes
    ----------
    name :
        Which solver was driven.
    variables :
        The variable vector it stopped on.
    seconds :
        Wall clock of the whole descent.
    calls :
        How many times each of the three maps was asked for a value.
    note :
        Whatever the solver said about why it stopped.
    """

    name: str
    variables: Float[np.ndarray, "variables"]
    seconds: float
    calls: dict[str, int]
    note: str


class CountedMaps(NamedTuple):
    """
    The three maps, wrapped so every call is counted.

    Attributes
    ----------
    weigh :
        The mass and its gradient.
    slack :
        How far under one every utilization sits, and the other held rows.
    jacobian :
        The slack's derivative in every variable.
    augmented :
        The mass and the rows as one scalar, with its gradient.
    mass :
        The mass and its gradient in the structure's own units, which is what
        an augmented objective is scaled by and what a round of it reports.
    calls :
        The running count, shared by all five and reset between drivers.

    Notes
    -----
    **`weigh` is normalized and `mass` is not, and both are counted the same.**
    The three solvers that read a Jacobian are all held to one relative
    tolerance, which needs a normalized objective; an augmented objective does
    its own scaling inside the traced program and needs the real mass to scale
    by. They are two readings of one call, so both count under `weigh` and no
    driver is charged for a conversion another was spared.
    """

    weigh: Callable[[Float[np.ndarray, "variables"]], tuple[float, np.ndarray]]
    slack: Callable[[Float[np.ndarray, "variables"]], np.ndarray]
    jacobian: Callable[[Float[np.ndarray, "variables"]], np.ndarray]
    augmented: Callable[..., tuple[float, np.ndarray]]
    mass: Callable[[Float[np.ndarray, "variables"]], tuple[float, np.ndarray]]
    calls: dict[str, int]


def counted_maps(maps, start: Float[np.ndarray, "variables"]) -> CountedMaps:
    """
    Wrap a search's compiled maps in host arrays and a call counter.

    Parameters
    ----------
    maps :
        The search's compiled maps.
    start :
        The variable vector the objective is normalized at.

    Returns
    -------
    counted :
        The same three maps, returning NumPy and counting their calls.

    Notes
    -----
    Every driver is handed this one object, so none of them can be charged for
    a conversion another was spared.

    The objective is divided by its value at the start, exactly as the shared
    descent divides it, so a tolerance means the same share of the mass to all
    three and none of them is measured against a stopping rule the others were
    not held to.
    """
    calls = {"weigh": 0, "slack": 0, "jacobian": 0, "augmented": 0}
    reference = abs(float(maps.weigh(jnp.asarray(start))[0])) or 1.0

    def weigh(x):
        calls["weigh"] += 1
        value, slope = maps.weigh(jnp.asarray(x))
        scaled = float(value) / reference

        return scaled, np.asarray(slope, dtype=np.float64) / reference

    def slack(x):
        calls["slack"] += 1

        return np.asarray(maps.slack(jnp.asarray(x)), dtype=np.float64)

    def jacobian(x):
        calls["jacobian"] += 1

        return np.asarray(maps.jacobian(jnp.asarray(x)), dtype=np.float64)

    def mass(x):
        calls["weigh"] += 1
        value, slope = maps.weigh(jnp.asarray(x))

        return float(value), np.asarray(slope, dtype=np.float64)

    def augmented(x, multipliers, penalty, scale):
        calls["augmented"] += 1
        value, slope = maps.augmented(jnp.asarray(x), multipliers, penalty, scale)

        return float(value), np.asarray(slope, dtype=np.float64)

    return CountedMaps(weigh, slack, jacobian, augmented, mass, calls)


def drive_slsqp(
    maps: CountedMaps,
    start: Float[np.ndarray, "variables"],
    boxes: list[tuple[float | None, float | None]],
) -> DriverCall:
    """
    The incumbent, in one round rather than the shared flow's five.

    Parameters
    ----------
    maps :
        The counted maps.
    start :
        The variable vector to leave from.
    boxes :
        One bound pair per variable.

    Returns
    -------
    call :
        Where it landed and what it spent.

    Notes
    -----
    One round, because a round is a restart at the previous answer and the
    other two drivers are given no such help.
    """
    held = {"type": "ineq", "fun": maps.slack, "jac": maps.jacobian}
    options = {"maxiter": EVALUATIONS, "ftol": TOLERANCE_RELATIVE}

    clock = time.perf_counter()
    found = minimize(
        maps.weigh,
        np.asarray(start, dtype=np.float64),
        jac=True,
        method="SLSQP",
        bounds=boxes,
        constraints=[held],
        options=options,
    )
    seconds = time.perf_counter() - clock

    return DriverCall(
        "SLSQP", np.asarray(found.x), seconds, dict(maps.calls), found.message
    )


def drive_nlopt(
    name: str,
    algorithm: int,
    maps: CountedMaps,
    seed: tuple[
        Float[np.ndarray, "variables"], list[tuple[float | None, float | None]]
    ],
) -> DriverCall:
    """
    One of nlopt's separable-approximation drivers over the same maps.

    Parameters
    ----------
    name :
        What to call it in the table.
    algorithm :
        The nlopt algorithm constant.
    maps :
        The counted maps.
    seed :
        The variable vector to leave from, and one bound pair per variable.

    Returns
    -------
    call :
        Where it landed and what it spent.

    Notes
    -----
    nlopt states inequalities as `c(x) <= 0` where the shared flow states them
    as slack at or above zero, so both the rows and their Jacobian change sign
    on the way in. The gradient buffers are written in place, which is the
    whole of nlopt's calling convention and the one place a transcription
    error would look like a solver result.

    A vector constraint rather than one per row: nlopt would otherwise call
    the Jacobian once per row, and the Jacobian is the whole cost.
    """
    start, boxes = seed
    lower = np.array([-np.inf if low is None else low for low, _ in boxes])
    upper = np.array([np.inf if high is None else high for _, high in boxes])

    def objective(x, gradient):
        value, slope = maps.weigh(x)
        if gradient.size:
            gradient[:] = slope

        return value

    def violation(result, x, gradient):
        result[:] = -maps.slack(x)
        if gradient.size:
            gradient[:] = -maps.jacobian(x)

    rows = maps.slack(np.asarray(start, dtype=np.float64)).size
    maps.calls["slack"] = 0

    driver = nlopt.opt(algorithm, int(np.size(start)))
    driver.set_min_objective(objective)
    driver.add_inequality_mconstraint(violation, np.zeros(rows))
    driver.set_lower_bounds(lower)
    driver.set_upper_bounds(upper)
    driver.set_ftol_rel(TOLERANCE_RELATIVE)
    driver.set_maxeval(EVALUATIONS)

    clock = time.perf_counter()
    try:
        landed = driver.optimize(np.asarray(start, dtype=np.float64))
        note = f"nlopt result {driver.last_optimize_result()}"
    except Exception as complaint:
        landed = np.asarray(start, dtype=np.float64)
        note = f"raised: {complaint}"
    seconds = time.perf_counter() - clock

    return DriverCall(name, np.asarray(landed), seconds, dict(maps.calls), note)


def named_profile(path: Path) -> StructureProfile:
    """
    The search profile an experiment exports, whichever family it describes.

    Parameters
    ----------
    path :
        The experiment script owning the profile.

    Returns
    -------
    profile :
        The structural family to drive.

    Raises
    ------
    ValueError
        If the script exports no profile, or more than one.

    Notes
    -----
    Found by type rather than by name, so a spike driving a new family needs no
    edit here and cannot be pointed at the wrong constant by a typo. A script
    exporting two would make the choice silent, which is why it is refused.
    """
    spec = importlib.util.spec_from_file_location("profiled", path)
    experiment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiment)
    exported = [
        found
        for found in vars(experiment).values()
        if isinstance(found, StructureProfile)
    ]
    if len(exported) != 1:
        raise ValueError(
            f"{path.name} must export exactly one search profile, found {len(exported)}"
        )

    return exported[0]


def drive_augmented(
    maps: CountedMaps,
    seed: tuple[
        Float[np.ndarray, "variables"], list[tuple[float | None, float | None]]
    ],
    budget: OptimizationBudget,
) -> DriverCall:
    """
    The augmented Lagrangian, then a short constrained polish.

    Parameters
    ----------
    maps :
        The counted maps.
    seed :
        The variable vector to leave from, and one bound pair per variable.
    budget :
        Rounds, inner iterations, the penalty schedule and the stopping rules.

    Returns
    -------
    call :
        Where it landed and what it spent, the polish included.

    Notes
    -----
    **The polish is part of the driver, not a second driver.** An augmented
    landing is stationary to whatever the outer loop reached and carries no
    statement about it, where the others stop on a criterion of their own;
    running a constrained solver from the landing is what makes them
    comparable, and it is charged to this driver's clock and call counts.

    A polish that finds real work left to do is a signal about the outer loop
    rather than a rescue, so its iteration count is worth reading beside the
    round count in the note.

    **A landing that never approached feasibility is refused, not polished.**
    An opening penalty too weak for the structure leaves the descent inside
    its own infeasible dive, holding a mass that no design supports, and a
    constrained solver started from such a point does not repair it — it
    wanders, and reports whatever it wandered to. Refusing keeps the driver's
    row a statement about the augmented descent rather than about how far a
    rescue happened to get, and the note says what was refused and by how
    much.
    """
    start, boxes = seed
    programs = ConstrainedMaps(maps.augmented, maps.mass, maps.slack)
    held = {"type": "ineq", "fun": maps.slack, "jac": maps.jacobian}
    options = {"maxiter": POLISH_ITERATIONS, "ftol": TOLERANCE_RELATIVE}
    opened = np.asarray(start, dtype=np.float64)

    clock = time.perf_counter()
    try:
        found = descend_augmented(programs, opened, boxes, budget)
        rounds = int(found.violations.size) - 1
        stopped = "converged" if found.converged else "round budget"
        reached = float(found.violations[-1])
        opening = f"{stopped} after {rounds} rounds, worst violation {reached:.1e}"
        if reached > POLISH_ADMISSION:
            landed = np.asarray(found.variables)
            note = (
                f"{opening}; REFUSED, over the {POLISH_ADMISSION:.0e} the polish "
                f"is admitted at — the opening penalty is too weak here"
            )
        else:
            polished = minimize(
                maps.weigh,
                found.variables,
                jac=True,
                method="SLSQP",
                bounds=boxes,
                constraints=[held],
                options=options,
            )
            landed = np.asarray(polished.x)
            note = f"{opening}; polish {polished.nit} iterations, {polished.message}"
    except (ValueError, FloatingPointError, RuntimeError) as complaint:
        landed = opened
        note = f"raised: {complaint}"
    seconds = time.perf_counter() - clock

    return DriverCall("augmented", landed, seconds, dict(maps.calls), note)


def report_drivers(
    report: Report,
    calls: list[DriverCall],
    reads: dict[str, object],
) -> None:
    """
    Every landing side by side, each read against every load case.

    Parameters
    ----------
    report :
        Where the table is written.
    calls :
        What each driver spent.
    reads :
        Each driver's answer read back as a design, keyed by driver name.
    """
    columns = (
        ReportColumn("driver", align="<"),
        ReportColumn("seconds", ".1f"),
        ReportColumn("weigh"),
        ReportColumn("slack"),
        ReportColumn("jacobian"),
        ReportColumn("augmented"),
        ReportColumn("mass [t]", ".6f"),
        ReportColumn("max U", ".9f"),
        ReportColumn("feasible", align="<"),
    )
    rows = []
    for call in calls:
        read = reads[call.name]
        violation = max(0.0, float(read.utilization.max()) - 1.0)
        rows.append(
            (
                call.name,
                call.seconds,
                call.calls["weigh"],
                call.calls["slack"],
                call.calls["jacobian"],
                call.calls["augmented"],
                read.mass,
                float(read.utilization.max()),
                "yes" if violation <= TOLERANCE_FEASIBILITY else "NO",
            )
        )

    report.write_table(columns, rows)


def main(path: Path, experiment: Path, separable: bool) -> None:
    """
    Prepare the end-to-end search once and hand it to each driver in turn.

    Parameters
    ----------
    path :
        The YAML file describing the run.
    experiment :
        The experiment script exporting the profile to drive.
    separable :
        Whether to drive nlopt's two separable-approximation methods as well.
    """
    profile = named_profile(experiment)

    family = profile.banner.split(" — ")[0]
    report = Report()
    report.write_banner(f"{family}: which solver reaches the end-to-end answer soonest")

    described = path.read_text()
    config = profile.parse_task(described)
    relaxed = augmented_budget(yaml.safe_load(described)) or AUGMENTED_DEFAULT
    structure = profile.build_structure(config)
    plan = profile.build_loads(structure, config)
    folding_by = folding_maps(profile, config, structure)
    problem = prepare_problem(structure, config, plan, folding_by)

    start = profile.signed_start(problem, config)
    if profile.sign_guard is None:
        guard = None
    else:
        guard = profile.sign_guard(config, start)
    limits = profile.height_limits(config)
    maps = search_maps(problem, limits, config.descent.length_floor, guard)
    finder = problem.pipeline.formfinder
    shape = finder.formfinder(jnp.asarray(start.q), problem.loads.formfinding)
    starts = search_starts(problem, start, shape.xyz, config.descent.diameter_floor)
    boxes = search_boxes(problem, config.descent.diameter_floor, limits)

    search = maps[SEARCH_FORMFOUND]
    seed = np.asarray(starts[SEARCH_FORMFOUND], dtype=np.float64)
    box = boxes[SEARCH_FORMFOUND]
    counted = counted_maps(search, seed)
    rows = counted.slack(seed).size

    held = [problem.case_names[index] for index in problem.cases_held]
    entries = (
        ("structure", f"{structure.num_nodes} nodes, {structure.num_edges} members"),
        ("variables", f"{seed.size}"),
        ("inequality rows", f"{rows}"),
        ("load cases with rows", f"{len(held)} of {len(problem.case_names)}"),
        ("described by", f"{path.name} through {experiment.name}"),
        ("chord sign guard", "none" if guard is None else f"{guard.chords.size} rows"),
        ("evaluation budget", f"{EVALUATIONS}"),
        (
            "augmented budget",
            f"{relaxed.rounds} rounds, opening penalty {relaxed.penalty:g}, "
            f"{relaxed.iterations} inner iterations for {relaxed.opening}",
        ),
    )
    report.write_heading("The problem every driver is handed")
    report.write_entries(entries)

    resting = {"weigh": 0, "slack": 0, "jacobian": 0, "augmented": 0}
    drivers = []
    counted.calls.update(resting)
    drivers.append(drive_slsqp(counted, seed, box))
    if separable:
        for name, algorithm in (("MMA", nlopt.LD_MMA), ("CCSA", nlopt.LD_CCSAQ)):
            counted.calls.update(resting)
            drivers.append(drive_nlopt(name, algorithm, counted, (seed, box)))
    counted.calls.update(resting)
    drivers.append(drive_augmented(counted, (seed, box), relaxed))

    width = int(finder.basis.shape[1])
    spread = problem.folding.diameters
    reads = {}
    for call in drivers:
        coordinates = jnp.asarray(call.variables)
        folded = coordinates[width:]
        diameters = spread @ folded if spread is not None else folded
        found = finder(coordinates[:width], problem.loads.formfinding)
        reads[call.name] = read_answer(
            problem, found.xyz, np.asarray(diameters), config.descent
        )

    report.write_heading(f"The {len(drivers)} drivers, from the same start")
    report_drivers(report, drivers, reads)

    report.write_heading("What each solver said")
    report.write_entries(tuple((call.name, call.note) for call in drivers))


if __name__ == "__main__":
    jnp.set_printoptions(precision=12)
    given = [word for word in sys.argv[1:] if not word.startswith("-")]
    asked = "--separable" in sys.argv[1:]
    described = (
        Path(given[0])
        if given
        else Path(__file__).resolve().parents[1] / "gridshell_16.yaml"
    )
    profiled = Path(given[1]) if len(given) > 1 else EXPERIMENT
    main(described, profiled, asked)
