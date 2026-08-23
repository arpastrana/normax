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

Three drivers over the identical maps, from the identical start:

    SLSQP       scipy's dense active-set sequential quadratic program, the
                incumbent every route in experiment 23 is descended by.
    MMA         nlopt's method of moving asymptotes, the separable convex
                approximation structural sizing is usually written against.
    CCSA        nlopt's conservative convex separable approximation, MMA's
                globally convergent sibling.

What is reported per driver: how many times each map was called, the wall
clock, the mass reached, and the worst utilization the answer shows when it is
read back against **every** load case. The last of these is what makes the
comparison a measurement rather than a race — a driver that arrives lighter by
leaving a constraint behind has not arrived.

Only the end-to-end route is driven, and only from the nominal start. A
multi-start comparison prices the landscape, and the landscape is not what is
in question here.

Run with `uv run --group pipeline --extra spike python
experiments/24_mma_spike.py [gridshell_16.yaml]`.
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
from design_routes import ROUTE_FORMFOUND
from design_routes import folding_maps
from design_routes import prepare_problem
from design_routes import read_answer
from design_routes import route_boxes
from design_routes import route_maps
from design_routes import route_starts
from jaxtyping import Float
from scipy.optimize import minimize

from normax.reporting import Report
from normax.reporting import ReportColumn

# Where experiment 23 keeps the profile this spike drives.
EXPERIMENT = Path(__file__).with_name("23_gridshell_optimize.py")

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
    calls :
        The running count, shared by all three and reset between drivers.
    """

    weigh: Callable[[Float[np.ndarray, "variables"]], tuple[float, np.ndarray]]
    slack: Callable[[Float[np.ndarray, "variables"]], np.ndarray]
    jacobian: Callable[[Float[np.ndarray, "variables"]], np.ndarray]
    calls: dict[str, int]


def counted_maps(maps, start: Float[np.ndarray, "variables"]) -> CountedMaps:
    """
    Wrap a route's compiled maps in host arrays and a call counter.

    Parameters
    ----------
    maps :
        The route's compiled maps.
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
    calls = {"weigh": 0, "slack": 0, "jacobian": 0}
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

    return CountedMaps(weigh, slack, jacobian, calls)


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


def report_drivers(
    report: Report,
    calls: list[DriverCall],
    reads: dict[str, object],
) -> None:
    """
    The three landings side by side, each read against every load case.

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
                read.mass,
                float(read.utilization.max()),
                "yes" if violation <= TOLERANCE_FEASIBILITY else "NO",
            )
        )

    report.write_table(columns, rows)


def main(path: Path) -> None:
    """
    Prepare the end-to-end route once and hand it to each driver in turn.

    Parameters
    ----------
    path :
        The YAML file describing the run.
    """
    spec = importlib.util.spec_from_file_location("experiment", EXPERIMENT)
    experiment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiment)
    profile = experiment.GRIDSHELL_PROFILE

    report = Report()
    report.write_banner("Which solver reaches the end-to-end answer soonest")

    config = profile.parse_task(path.read_text())
    structure = profile.build_structure(config)
    plan = profile.build_loads(structure, config)
    folding_by = folding_maps(profile, config, structure)
    problem = prepare_problem(structure, config, plan, folding_by)

    start = profile.signed_start(problem, config)
    guard = profile.sign_guard(config, start)
    limits = profile.height_limits(config)
    maps = route_maps(problem, limits, config.descent.length_floor, guard)
    finder = problem.pipeline.formfinder
    shape = finder.formfinder(jnp.asarray(start.q), problem.loads.formfinding)
    starts = route_starts(problem, start, shape.xyz, config.descent.diameter_floor)
    boxes = route_boxes(problem, config.descent.diameter_floor, limits)

    route = maps[ROUTE_FORMFOUND]
    seed = np.asarray(starts[ROUTE_FORMFOUND], dtype=np.float64)
    box = boxes[ROUTE_FORMFOUND]
    counted = counted_maps(route, seed)
    rows = counted.slack(seed).size

    held = [problem.case_names[index] for index in problem.cases_held]
    entries = (
        ("structure", f"{structure.num_nodes} nodes, {structure.num_edges} members"),
        ("variables", f"{seed.size}"),
        ("inequality rows", f"{rows}"),
        ("load cases with rows", f"{len(held)} of {len(problem.case_names)}"),
        ("evaluation budget", f"{EVALUATIONS}"),
    )
    report.write_heading("The problem all three are handed")
    report.write_entries(entries)

    drivers = []
    counted.calls.update({"weigh": 0, "slack": 0, "jacobian": 0})
    drivers.append(drive_slsqp(counted, seed, box))
    for name, algorithm in (("MMA", nlopt.LD_MMA), ("CCSA", nlopt.LD_CCSAQ)):
        counted.calls.update({"weigh": 0, "slack": 0, "jacobian": 0})
        drivers.append(drive_nlopt(name, algorithm, counted, (seed, box)))

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

    report.write_heading("The three drivers, from the same start")
    report_drivers(report, drivers, reads)

    report.write_heading("What each solver said")
    report.write_entries(tuple((call.name, call.note) for call in drivers))


if __name__ == "__main__":
    jnp.set_printoptions(precision=12)
    described = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    main(described or Path(__file__).with_name("gridshell_16.yaml"))
