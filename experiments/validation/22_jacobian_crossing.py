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
Price how a constraint Jacobian crosses the blueprint boundary.

A simultaneous search holds the code check as constraints, and its Jacobian
used to cross the boundary once per constraint row: a Tesseract request is
stateless, so every row re-did the primal before pulling one cotangent
through the hand-written adjoint. The `jacobian` endpoint collapses that to
one crossing per load case — the server writes its per-member pulls into
diagonal blocks and tesseract-jax contracts them client-side — and this
experiment measures what that buys rather than assuming it.

Four cells, timed and counted: {reverse, forward} x {endpoint, sequential}.
Reverse mode is what experiment 103's SLSQP calls; forward mode is the same
Jacobian asked column-wise, priced here because the constraint slack is wide
(rows exceed columns under a shared force density). The routes must agree to
the last bit — the blocks hold the same pulls the products contract — and a
tolerance check holds them to what is measured.

Then the whole of experiment 103's search runs on both routes, so the
headline is end to end: same optimizer, same answer, different wall clock.

The served leg repeats the cells over HTTP and is skipped unless
`NORMAX_SERVED_OUTPUT` names a directory the container runtime can bind;
build the image first with `tesseract build tesseracts/sizing`.

Run with `uv run --group pipeline python experiments/22_jacobian_crossing.py
[jacobian_crossing.yaml]`.
"""

import importlib.util
import os
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from tesseract_core import Tesseract

from normax.design import StructuralDesignPipeline
from normax.reporting import Report
from normax.reporting import ReportColumn
from normax.reporting import ToleranceCheck
from normax.reporting import checks_passed
from normax.tesseract import TesseractSizer
from normax.tesseract import sizing_tesseract

# The two routes carry the same pulls, so their entries may differ only by
# the client-side contraction's round-off; measured at zero, held with room.
TOLERANCE_ROUTE = 1e-15

# Reverse and forward mode reassociate differently; scaled by the largest entry.
TOLERANCE_MODE = 1e-12

# The two end-to-end answers, as masses in tonnes.
TOLERANCE_ANSWER = 1e-9


class CrossingBudget(NamedTuple):
    """
    How the timing samples are spent.

    Attributes
    ----------
    repeats :
        Timed calls per cell; the median is reported.
    """

    repeats: int


class ServedImage(NamedTuple):
    """
    The container the served leg runs against.

    Attributes
    ----------
    image :
        Name of the built blueprint-check image.
    version :
        Tag of that image.
    """

    image: str
    version: str


class StudyConfig(NamedTuple):
    """
    Everything the crossing study is described by.

    Attributes
    ----------
    arch_config :
        File the arch, the loads and the search budgets are read from.
    crossing :
        The timing budget.
    served :
        The container the served leg runs against.
    """

    arch_config: str
    crossing: CrossingBudget
    served: ServedImage


class StudyProblem(NamedTuple):
    """
    One route's compiled maps, with its crossing counters attached.

    Attributes
    ----------
    problem :
        The compiled objective, slack, Jacobian and start from experiment 103.
    forward :
        Forward-mode Jacobian of the same slack, compiled.
    calls :
        The boundary's derivative-endpoint call counters, mutated in place.
    """

    problem: NamedTuple
    forward: object
    calls: dict[str, int]


def parse_config(text: str) -> StudyConfig:
    """
    The crossing study a file describes.

    Parameters
    ----------
    text :
        Text of the file describing the run.

    Returns
    -------
    config :
        The arch file, the timing budget and the served image.

    Raises
    ------
    TypeError
        If the text names a field that does not exist, or omits one that does.
    """
    document = yaml.safe_load(text)

    config = StudyConfig(
        arch_config=document["arch_config"],
        crossing=CrossingBudget(**document["crossing"]),
        served=ServedImage(**document["served"]),
    )

    return config


def load_experiment(path: Path) -> ModuleType:
    """
    An experiment as a module, its digit-led name notwithstanding.

    Parameters
    ----------
    path :
        File the experiment lives in.

    Returns
    -------
    module :
        The loaded module, whose builders this experiment reuses.
    """
    spec = importlib.util.spec_from_file_location(f"api_{path.stem[:3]}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def counted_endpoints(client: Tesseract) -> dict[str, int]:
    """
    Wrap a Tesseract's derivative endpoints with call counters.

    Parameters
    ----------
    client :
        The Tesseract whose crossings are counted.

    Returns
    -------
    calls :
        Counts per endpoint name, mutated by every later crossing.
    """
    calls = {"jacobian": 0, "vector_jacobian_product": 0, "jacobian_vector_product": 0}

    def counting(name):
        true_endpoint = getattr(client, name)

        def counted(*args, **kwargs):
            calls[name] += 1

            return true_endpoint(*args, **kwargs)

        return counted

    for name in calls:
        setattr(client, name, counting(name))

    return calls


def reset_counts(calls: dict[str, int]) -> None:
    """
    Zero every endpoint counter before a counted call.
    """
    for name in calls:
        calls[name] = 0


def steady(call, repeats: int) -> float:
    """
    Median steady-state seconds of a call, compilation excluded.

    Parameters
    ----------
    call :
        The zero-argument call to time.
    repeats :
        Timed calls the median is taken over.

    Returns
    -------
    seconds :
        Median wall-clock seconds of one call.

    Notes
    -----
    The warm-up call is load-bearing — the first call compiles — and every
    sample is blocked on, since a call returning futures would be timed
    wrongly and silently.
    """
    jax.block_until_ready(call())

    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        jax.block_until_ready(call())
        samples.append(time.perf_counter() - started)

    return float(np.median(samples))


class ArchScaffold(NamedTuple):
    """
    Everything experiment 103 builds around a pipeline, built once here.

    Attributes
    ----------
    simultaneous :
        Experiment 103 as a module, supplying the compiled problem and solver.
    arch :
        The parsed arch configuration, for the bounds and the budgets.
    searched :
        The constrained search's budgets.
    loads :
        The form-finding case and the checked cases.
    params :
        The seed force densities and diameters.
    layout :
        The sizes of the two variable blocks.
    constraints :
        The geometric conditions the configuration activated.
    formfinder :
        The form-finding block both routes share.
    analyzer :
        The analysis block both routes share.
    structure :
        The arch itself.
    family :
        The section family every sizer draws from.
    """

    simultaneous: ModuleType
    arch: NamedTuple
    searched: NamedTuple
    loads: NamedTuple
    params: NamedTuple
    layout: NamedTuple
    constraints: NamedTuple
    formfinder: object
    analyzer: object
    structure: object
    family: object


def build_scaffold(config: StudyConfig, config_dir: Path) -> ArchScaffold:
    """
    Build the shared problem every route is measured on.

    Parameters
    ----------
    config :
        The crossing study's own settings.
    config_dir :
        Directory the arch configuration is read beside.

    Returns
    -------
    scaffold :
        The parsed configs, the shared blocks and 103's machinery.

    Notes
    -----
    The sizing backend named in the arch file is deliberately ignored: this
    study is about the blueprint boundary, so it builds its own clients and
    borrows only the form finder and the analyzer from the showcase builder.
    """
    here = Path(__file__).parent
    simultaneous = load_experiment(here / "103_simultaneous_api.py")
    api = load_experiment(here.parents[1] / "examples" / "arch.py")

    config_text = (config_dir / config.arch_config).read_text()
    arch = api.parse_config(config_text)
    searched = simultaneous.parse_simultaneous(config_text)
    structure = api.build_arch(arch.structure)
    loads = api.arch_load_cases(structure, arch.load_cases)
    base = api.build_pipeline(structure, arch)
    params = api.initialize_parameters(structure, arch)
    layout = simultaneous.variable_layout(searched.force_densities, structure.num_edges)

    floor = arch.optimization.length_floor
    floor_length = floor.fraction * arch.structure.span / arch.structure.num_edges
    seed_shape = base.formfinder(params.force_densities, loads.formfinding)
    constraints = simultaneous.shape_constraints(
        searched, structure, seed_shape, floor_length
    )

    scaffold = ArchScaffold(
        simultaneous,
        arch,
        searched,
        loads,
        params,
        layout,
        constraints,
        base.formfinder,
        base.analyzer,
        structure,
        base.sizer.family,
    )

    return scaffold


def route_problem(
    scaffold: ArchScaffold,
    client: Tesseract,
    materialize_jacobian: bool | None,
) -> StudyProblem:
    """
    Compile one route's maps over a counted boundary.

    Parameters
    ----------
    scaffold :
        The shared problem.
    client :
        The boundary the route crosses.
    materialize_jacobian :
        How its batched derivatives cross; `None` takes the endpoint,
        `False` forces one product crossing per row.

    Returns
    -------
    study :
        The compiled maps, the forward-mode Jacobian and the counters.
    """
    calls = counted_endpoints(client)
    sizer = TesseractSizer(
        scaffold.structure, client, scaffold.family, materialize_jacobian
    )
    pipeline = StructuralDesignPipeline(scaffold.formfinder, scaffold.analyzer, sizer)
    problem = scaffold.simultaneous.constrained_problem(
        pipeline,
        scaffold.loads,
        scaffold.params,
        scaffold.layout,
        scaffold.constraints,
    )
    forward = jax.jit(jax.jacfwd(problem.slack))
    forward(problem.start)

    return StudyProblem(problem, forward, calls)


def report_cells(
    report: Report,
    routes: dict[str, StudyProblem],
    repeats: int,
) -> dict[str, np.ndarray]:
    """
    Time and count every {mode, route} cell, and hand back the Jacobians.

    Parameters
    ----------
    report :
        The report the table is written to.
    routes :
        Each route's compiled maps and counters, keyed by route name.
    repeats :
        Timed calls per cell.

    Returns
    -------
    jacobians :
        The reverse-mode Jacobian of each route, for the parity checks.
    """
    columns = (
        ReportColumn("mode", "", "<"),
        ReportColumn("route", "", "<"),
        ReportColumn("jacobian", "d"),
        ReportColumn("vjp", "d"),
        ReportColumn("jvp", "d"),
        ReportColumn("median [ms]", ".2f"),
    )

    rows = []
    jacobians = {}
    for route, study in routes.items():
        modes = {
            "reverse": study.problem.slack_jacobian,
            "forward": study.forward,
        }
        for mode, differentiate in modes.items():
            seconds = steady(lambda: differentiate(study.problem.start), repeats)
            reset_counts(study.calls)
            answer = jax.block_until_ready(differentiate(study.problem.start))
            row = (
                mode,
                route,
                study.calls["jacobian"],
                study.calls["vector_jacobian_product"],
                study.calls["jacobian_vector_product"],
                seconds * 1e3,
            )
            rows.append(row)
            if mode == "reverse":
                jacobians[route] = np.asarray(answer)
    report.write_table(columns, rows)

    return jacobians


def solve_routes(
    report: Report,
    scaffold: ArchScaffold,
    routes: dict[str, StudyProblem],
) -> list[ToleranceCheck]:
    """
    Run 103's whole search on each route and report what each one spent.

    Parameters
    ----------
    report :
        The report the entries are written to.
    scaffold :
        The shared problem, supplying the bounds and the budgets.
    routes :
        Each route's compiled maps, keyed by route name.

    Returns
    -------
    checks :
        The agreement of the two answers, as a tolerance check.
    """
    masses = {}
    entries = []
    for route, study in routes.items():
        answer = scaffold.simultaneous.solve_constrained(
            study.problem,
            scaffold.arch.optimization.bounds,
            scaffold.searched,
            scaffold.layout,
        )
        mass, _ = study.problem.weigh(answer.variables)
        masses[route] = float(mass)
        entries.append((f"{route} mass [t]", f"{float(mass):.9f}"))
        spent = f"{answer.evaluations} in {answer.elapsed:.3f} s"
        entries.append((f"{route} spent", spent))
    report.write_entries(entries)

    gap = abs(masses["endpoint"] - masses["sequential"])
    check = ToleranceCheck("end-to-end mass agreement [t]", gap, TOLERANCE_ANSWER)

    return [check]


def crossing_study(
    report: Report,
    scaffold: ArchScaffold,
    clients: tuple[Tesseract, Tesseract],
    repeats: int,
) -> list[ToleranceCheck]:
    """
    Measure the four cells and the route agreement over one pair of boundaries.

    Parameters
    ----------
    report :
        The report everything is written to.
    scaffold :
        The shared problem.
    clients :
        Two boundaries to the same check, one per route, counted apart.
    repeats :
        Timed calls per cell.

    Returns
    -------
    checks :
        The parity of the routes and the modes.
    """
    endpoint_client, sequential_client = clients
    routes = {
        "endpoint": route_problem(scaffold, endpoint_client, None),
        "sequential": route_problem(scaffold, sequential_client, False),
    }

    start = routes["endpoint"].problem.start
    slack_rows = int(np.asarray(routes["endpoint"].problem.slack(start)).size)
    entries = [
        ("constraint rows", f"{slack_rows}"),
        ("variable columns", f"{int(np.asarray(start).size)}"),
    ]
    report.write_entries(entries)

    jacobians = report_cells(report, routes, repeats)

    routed = jacobians["endpoint"]
    rowed = jacobians["sequential"]
    scale = float(np.max(np.abs(rowed)))
    route_gap = float(np.max(np.abs(routed - rowed))) / scale

    forward = jax.block_until_ready(routes["endpoint"].forward(start))
    mode_gap = float(np.max(np.abs(np.asarray(forward) - routed))) / scale

    route_check = ToleranceCheck(
        "endpoint against sequential, scaled", route_gap, TOLERANCE_ROUTE
    )
    mode_check = ToleranceCheck(
        "forward against reverse, scaled", mode_gap, TOLERANCE_MODE
    )
    checks = [route_check, mode_check]

    return checks


def report_served(
    report: Report,
    scaffold: ArchScaffold,
    config: StudyConfig,
) -> list[ToleranceCheck]:
    """
    Repeat the cells over HTTP, where the environment offers a directory.

    Parameters
    ----------
    report :
        The report everything is written to.
    scaffold :
        The shared problem.
    config :
        The image name and the timing budget.

    Returns
    -------
    checks :
        The served parity checks, or nothing where the leg is skipped.
    """
    directory = os.environ.get("NORMAX_SERVED_OUTPUT")
    if directory is None:
        report.write_heading(
            "Served containers skipped; set NORMAX_SERVED_OUTPUT to run them"
        )
        return []

    report.write_heading("Served over HTTP")
    tagged = f"{config.served.image}:{config.served.version}"
    with (
        Tesseract.from_image(tagged, output_path=directory) as first,
        Tesseract.from_image(tagged, output_path=directory) as second,
    ):
        checks = crossing_study(
            report, scaffold, (first, second), config.crossing.repeats
        )

    return checks


def main(path: Path) -> None:
    """
    Run the study, write the report, and end on a verdict.

    Parameters
    ----------
    path :
        The YAML file describing the run.
    """
    config = parse_config(path.read_text())
    scaffold = build_scaffold(config, path.parent)

    report = Report()
    report.write_banner("The Jacobian crossing — one boundary call, or one per row")

    report.write_heading("In process")
    clients = (sizing_tesseract("blueprint"), sizing_tesseract("blueprint"))
    checks = crossing_study(report, scaffold, clients, config.crossing.repeats)

    report.write_heading("End to end, experiment 103's search on both routes")
    solve_clients = (sizing_tesseract("blueprint"), sizing_tesseract("blueprint"))
    routes = {
        "endpoint": route_problem(scaffold, solve_clients[0], None),
        "sequential": route_problem(scaffold, solve_clients[1], False),
    }
    checks += solve_routes(report, scaffold, routes)

    checks += report_served(report, scaffold, config)

    report.write_heading("Summary")
    report.write_checks(checks)
    report.write_verdict(checks_passed(checks))


if __name__ == "__main__":
    jnp.set_printoptions(precision=12)
    described = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    main(described or Path(__file__).with_name("jacobian_crossing.yaml"))
