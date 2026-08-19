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
The Warren truss designed end to end, against two searches without a form finder.

Three constrained searches over the same members, the same load cases, the
same analysis and the same code check, differing only in how — and whether —
the geometry moves. The end-to-end route moves the held-plan subspace
coordinates of experiment 16 together with every member diameter: the form
finder turns the coordinates into a geometry, the frame analysis into member
forces, the EN 1993-1-1 check into utilizations. The free-heights route drops
the form finder and hands the optimizer the height of every free node
directly, driving the same T2 and T3 alone. The sizing-only route holds the
truss as drawn and moves the diameters alone. All three run the same SLSQP
under hard `U <= 1` per member and load case, analytic Jacobians throughout.
The machinery is shared with the Vierendeel of experiment 19 and lives in
`truss_routes`; this file owns what is Warren about the run.

**On this truss the two shaped routes span the same geometries.** Experiment
16 counted every held-plan geometry funicular-reachable — sixteen independent
edges against fifteen free heights plus one self-stress — so unlike the arch
of experiment 15, where the heights were a strict superset, here the density
route gives nothing away. Whether the two parametrizations also *land* on the
same design is what the run measures: any gap between them is landscape and
conditioning, never reach.

Four load cases, all on the bottom chord: the uniform deck the shape is
form-found under, the two half-span cases that swap the diagonals between
tension and compression — those three of equal total — and a fraction of that
total concentrated at the midspan deck node.
One diameter per member has to satisfy all four at once, so the envelope is
a KKT condition rather than a reconciliation.

**The truss is once statically indeterminate, and it shows twice.** The
thrust-vs-tie split of the funicular fit is not what the elastic frame
carries: internal forces depend on the stiffness distribution, so the
funicular `q L` and the analyzed axial force disagree where the arch is
determinate and they agree to machine precision. And the frozen-seed
envelope that seeds each search is measurably infeasible once the frame is
re-analyzed at its own sections — the `∂N/∂d` coupling the arch priced at a
tenth of a percent is orders larger here. Both are measured and reported;
the simultaneous formulation holds the coupling inside the gradient, so
neither survives to the answers.

The YAML's `limit_rise` switch puts a lid on how tall either shaped route may
grow: no vertex above `rise_factor` times the drawn depth. The free-heights
route carries the lid as a box bound on its own variables; the end-to-end
route, whose heights are outputs of the form finder, carries it as one
normalized inequality row per free node. The sag stays free either way — it
is a ceiling, not a box — and the sizing-only route never notices it.

The descents restart until quiet: SLSQP is rerun from its own answer until a
round no longer moves, each restart refreshing the quadratic model. The two
shaped routes leave from the same matched start — the signed lens, written
once as basis coordinates and once as heights. The report compares the routes
by shape, by count of variables, by mass across all load cases, and by member
utilization, family by family.

Run with `uv run --group pipeline python experiments/18_warren_optimize.py
[warren_optimize.yaml]`.
"""

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jaxtyping import Int
from truss_routes import FIGURES
from truss_routes import ROUTE_FORMFOUND
from truss_routes import ROUTE_HEIGHTS
from truss_routes import TOLERANCE_GRADIENT
from truss_routes import TOLERANCE_PROJECTION
from truss_routes import TOLERANCE_SHAPE
from truss_routes import RouteProblem
from truss_routes import StartMeasures
from truss_routes import StartPoint
from truss_routes import TaskConfig
from truss_routes import descend_all
from truss_routes import force_agreement
from truss_routes import lens_geometry
from truss_routes import mirrored_edges
from truss_routes import parse_config
from truss_routes import prepare_problem
from truss_routes import report_families
from truss_routes import report_governing
from truss_routes import report_gradient
from truss_routes import report_routes
from truss_routes import report_summary
from truss_routes import rise_ceiling
from truss_routes import route_boxes
from truss_routes import route_checks
from truss_routes import route_maps
from truss_routes import route_reads
from truss_routes import route_starts
from truss_routes import route_variables
from truss_routes import seed_openings
from truss_routes import signed_shift
from truss_routes import start_entries
from truss_routes import write_figures

from normax.form_finding.fdm import fit_densities
from normax.reporting import Report
from normax.reporting import ToleranceCheck
from normax.reporting import checks_passed
from normax.structures import build_warren_2d


def mirrored_nodes(num_bays: int) -> Int[np.ndarray, "nodes"]:
    """
    Mirror image of every node index about midspan, chord by chord.
    """
    bottom = num_bays - np.arange(num_bays + 1)
    top = 2 * num_bays - np.arange(num_bays)

    return np.concatenate([bottom, top])


def member_families(num_bays: int) -> tuple[tuple[str, slice], ...]:
    """
    Name and member slice of every family, in the generator's order.

    Parameters
    ----------
    num_bays :
        Number of bottom-chord segments the span is divided into.

    Returns
    -------
    families :
        The two chords and the two diagonal directions.
    """
    families = (
        ("bottom chord", slice(0, num_bays)),
        ("top chord", slice(num_bays, 2 * num_bays - 1)),
        ("rising diagonals", slice(2 * num_bays - 1, 3 * num_bays - 1)),
        ("falling diagonals", slice(3 * num_bays - 1, 4 * num_bays - 1)),
    )

    return families


def signed_start(problem: RouteProblem, config: TaskConfig) -> StartPoint:
    """
    The lens fit of experiment 16, signed and written in the searched basis.

    Parameters
    ----------
    problem :
        The prepared truss.
    config :
        The run description, supplying the sketch and the sign margin.

    Returns
    -------
    start :
        The signed densities, their coordinates, and the projection gap.

    Notes
    -----
    Fitted in the full edge space and then projected, rather than fitted in
    the basis directly: the restricted fit's self-stress direction rides at
    the least-squares rank cutoff and is unreliably detected, while the free
    fit's is orders below it. The projection costs nothing measurable — the
    signed densities hold the plan, so they already live in the basis's span,
    and the gap is reported rather than assumed.
    """
    bays = config.structure.num_bays
    span = config.structure.span

    sketch = config.sketch
    lens = lens_geometry(
        problem.structure, span, bays, sketch.sag_lens, sketch.rise_lens
    )

    fit = fit_densities(problem.structure, lens, problem.loads.formfinding)
    mode = fit.self_stresses[:, 0]

    signs = np.concatenate([np.ones(bays), -np.ones(bays - 1)])
    chords = np.arange(2 * bays - 1)
    margin = config.subspace.margin_fraction * float(np.median(np.abs(fit.q[:bays])))

    shifted = signed_shift(fit.q, mode, signs, chords, margin)

    finder = problem.pipeline.formfinder
    xi = finder.read_coordinates(shifted.q)
    rebuilt = np.asarray(finder.basis) @ xi
    projection = float(np.linalg.norm(rebuilt - shifted.q) / np.linalg.norm(shifted.q))

    return StartPoint(shifted.q, xi, lens, projection, fit.gap)


def main(path: Path) -> None:
    """
    Run the three routes, write the report, and save the figures.

    Parameters
    ----------
    path :
        The YAML file describing the run.
    """
    report = Report()
    report.write_banner("Warren truss — three routes to a design")

    config = parse_config(path.read_text())
    budget = config.descent
    bays = config.structure.num_bays

    structure = build_warren_2d(bays, config.structure.span, config.structure.depth)
    nodes_mirrored = mirrored_nodes(bays)
    problem = prepare_problem(
        structure, config, nodes_mirrored, mirrored_edges(nodes_mirrored, structure)
    )

    start = signed_start(problem, config)
    finder = problem.pipeline.formfinder
    shape = finder.formfinder(jnp.asarray(start.q), problem.loads.formfinding)
    reproduction = float(jnp.max(jnp.abs(shape.xyz - jnp.asarray(start.lens))))
    disagreement = force_agreement(problem, start, shape.xyz)

    ceiling = rise_ceiling(budget, config.structure.depth)
    maps = route_maps(problem, ceiling, budget.length_floor)
    starts = route_starts(problem, start, shape.xyz, budget.diameter_floor)
    opening_found, opening_drawn = seed_openings(maps, starts)
    measures = StartMeasures(reproduction, disagreement, opening_found, opening_drawn)

    report.write_heading("The start, and what the indeterminacy does to it")
    entries = start_entries(config, problem, start, measures, ceiling)
    report.write_entries(tuple(entries))

    best_found = report_gradient(
        report, maps[ROUTE_FORMFOUND], starts[ROUTE_FORMFOUND], ROUTE_FORMFOUND
    )
    best_heights = report_gradient(
        report, maps[ROUTE_HEIGHTS], starts[ROUTE_HEIGHTS], ROUTE_HEIGHTS
    )
    best_error = max(best_found, best_heights)

    report.write_heading("Descending the three routes")
    boxes = route_boxes(problem, budget.diameter_floor, ceiling)
    answers = descend_all(report, maps, starts, boxes, budget)

    reads = route_reads(problem, answers, budget)
    report_routes(report, reads, answers, route_variables(problem))
    report_families(report, reads, member_families(bays))
    report_governing(report, reads)
    report_summary(report, reads, config, ceiling)

    write_figures(problem, reads, answers, "18_warren")
    report.write_heading(f"figures written to {FIGURES}")

    checks = [
        ToleranceCheck("gradient scaled error", best_error, TOLERANCE_GRADIENT),
        ToleranceCheck("projection gap", start.projection, TOLERANCE_PROJECTION),
        ToleranceCheck("lens reproduction [mm]", reproduction, TOLERANCE_SHAPE),
    ]
    routed, sound = route_checks(reads, answers, ceiling)
    checks.extend(routed)
    passed = checks_passed(tuple(checks)) and sound

    report.write_checks(tuple(checks))
    report.write_verdict(passed)


if __name__ == "__main__":
    jnp.set_printoptions(precision=12)
    described = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    main(described or Path(__file__).with_name("warren_optimize.yaml"))
