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
The Vierendeel truss designed end to end, against two searches without a form finder.

The Warren race of experiment 18, rerun on the truss where funicularity is
scarce. Same three constrained searches over the same four load cases, the
same frame analysis and the same EN 1993-1-1 check, the machinery shared in
`truss_routes`: the end-to-end route moves held-plan basis coordinates and
every diameter through the whole pipeline, the free-heights route hands the
optimizer the height of every free node directly, the sizing-only route holds
the truss as drawn.

**On this truss the two shaped routes do not span the same geometries.** The
Warren's counting made every held-plan geometry funicular-reachable, so any
gap between its routes was landscape alone. Remove the diagonals and the
counting flips: experiment 17 measured nine independent edges — two uniform
chord families and seven free verticals — against fourteen free heights, one
of the nine blind. Funicular shapes are a strict submanifold of what the
free-heights route can draw, so here the comparison prices the funicular
parametrization itself: what the form finder's discipline costs — or buys —
against a route that may bend the chords however the analysis likes.

**Rigid joints put end moments everywhere.** A pin-jointed Vierendeel is a
mechanism, so T2 models every member as a rigid-jointed beam and the panels
carry load through joint bending — the interaction rows 6.61 and 6.62 govern
rather than the axial resistance alone. The funicular start still zeroes the
shaping case's moments as well as any geometry can, but no shape makes a
Vierendeel momentless under the asymmetric cases.

**The start must be fitted inside the basis.** Offered a sketch, the free
least squares happily abandons the top chord: it zeroes the hangers, reports
a deceptively small balance gap, and hands back a vertical stiffness that is
singular — the degeneracy experiment 17 ran on purpose. The lens here is
fitted inside the held-plan basis, where plan balance is exact by
construction, and shifted along the one self-stress — the load-path split
between hanging deck and arching top chord — until both chords carry their
signs. The vertical stiffness the form finder solves is watched at the start
and at the answer, since the descent is free to drive toward the
degenerate chord-off states.

Run with `uv run --group pipeline python experiments/19_vierendeel_optimize.py
[vierendeel_optimize.yaml]`.
"""

import sys
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax_fdm.equilibrium import EquilibriumStructure
from jaxtyping import Float
from jaxtyping import Int
from truss_routes import FIGURES
from truss_routes import ROUTE_FORMFOUND
from truss_routes import ROUTE_HEIGHTS
from truss_routes import TOLERANCE_FEASIBILITY
from truss_routes import TOLERANCE_GRADIENT
from truss_routes import TOLERANCE_PROJECTION
from truss_routes import TOLERANCE_SHAPE
from truss_routes import ChordSigns
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

from normax.form_finding.fdm import equilibrium_graph
from normax.form_finding.fdm import fit_densities
from normax.reporting import Report
from normax.reporting import ToleranceCheck
from normax.reporting import checks_passed
from normax.structures import build_vierendeel_2d
from normax.structures import member_lengths

# How exactly the restricted fit balances the lens, scaled by the total load.
TOLERANCE_FIT = 1e-11


class StiffnessSpectrum(NamedTuple):
    """
    The vertical stiffness the form finder solves, read at one density vector.

    Attributes
    ----------
    negatives :
        Count of negative eigenvalues — mixed member signs, not a defect.
    size :
        Count of eigenvalues, one per free node.
    condition :
        Ratio of the largest eigenvalue magnitude to the smallest. Explodes
        when the densities approach the degenerate chord-off states.
    """

    negatives: int
    size: int
    condition: float


def mirrored_nodes(num_bays: int) -> Int[np.ndarray, "nodes"]:
    """
    Mirror image of every node index about midspan, chord by chord.
    """
    bottom = num_bays - np.arange(num_bays + 1)
    top = 2 * num_bays + 1 - np.arange(num_bays + 1)

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
        The two chords and the verticals — no diagonals on a Vierendeel.
    """
    families = (
        ("bottom chord", slice(0, num_bays)),
        ("top chord", slice(num_bays, 2 * num_bays)),
        ("verticals", slice(2 * num_bays, None)),
    )

    return families


def signed_start(problem: RouteProblem, config: TaskConfig) -> StartPoint:
    """
    The lens fitted inside the searched basis, signed along the load-path split.

    Parameters
    ----------
    problem :
        The prepared truss.
    config :
        The run description, supplying the sketch and the sign margin.

    Returns
    -------
    start :
        The signed densities, their coordinates, the projection gap, and the
        balance gap the restricted fit left.

    Notes
    -----
    Fitted inside the held-plan basis — the reverse of the Warren recipe, and
    for the reason experiment 17 measured: offered a sketch off the funicular
    manifold, the free least squares abandons the top chord, reports a
    deceptively small gap, and returns a singular vertical stiffness. The
    restricted fit keeps plan balance exact by construction, and its one
    self-stress is the load-path split between hanging deck and arching top
    chord, which the sign shift then moves along until both chords carry
    their signs. The verticals stay free: a hanger in the lens is a post in
    another shape.
    """
    bays = config.structure.num_bays
    span = config.structure.span

    sketch = config.sketch
    lens = lens_geometry(
        problem.structure, span, bays, sketch.sag_lens, sketch.rise_lens
    )

    finder = problem.pipeline.formfinder
    basis = np.asarray(finder.basis)
    fit = fit_densities(problem.structure, lens, problem.loads.formfinding, basis)
    mode = fit.self_stresses[:, 0]

    signs = np.concatenate([np.ones(bays), -np.ones(bays)])
    chords = np.arange(2 * bays)
    margin_fraction = config.subspace.margin_fraction
    margin = margin_fraction * float(np.median(np.abs(fit.q[chords])))

    shifted = signed_shift(fit.q, mode, signs, chords, margin)

    xi = finder.read_coordinates(shifted.q)
    rebuilt = basis @ xi
    projection = float(np.linalg.norm(rebuilt - shifted.q) / np.linalg.norm(shifted.q))

    return StartPoint(shifted.q, xi, lens, projection, fit.gap)


def stiffness_spectrum(
    graph: EquilibriumStructure,
    q: Float[np.ndarray, "edges"],
) -> StiffnessSpectrum:
    """
    Sign count and conditioning of the vertical stiffness at one density.

    Parameters
    ----------
    graph :
        The form-finding connectivity.
    q :
        Force density of every edge.

    Returns
    -------
    spectrum :
        The negative-eigenvalue count and the condition number.
    """
    connectivity = np.asarray(graph.connectivity_free)
    stiffness = connectivity.T @ (q[:, None] * connectivity)
    eigen = np.linalg.eigvalsh(stiffness)

    negatives = int(np.sum(eigen < 0.0))
    condition = float(np.abs(eigen).max() / np.abs(eigen).min())

    return StiffnessSpectrum(negatives, eigen.size, condition)


def main(path: Path) -> None:
    """
    Run the three routes, write the report, and save the figures.

    Parameters
    ----------
    path :
        The YAML file describing the run.
    """
    report = Report()
    report.write_banner("Vierendeel truss — three routes to a design")

    config = parse_config(path.read_text())
    budget = config.descent
    bays = config.structure.num_bays

    structure = build_vierendeel_2d(bays, config.structure.span, config.structure.depth)
    graph = equilibrium_graph(structure)
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
    signs = np.concatenate([np.ones(bays), -np.ones(bays)])
    chords = np.arange(2 * bays)
    scale = float(np.median(np.abs(start.q[chords])))
    guard = ChordSigns(signs, chords, config.subspace.margin_fraction * scale, scale)
    maps = route_maps(problem, ceiling, budget.length_floor, guard)
    starts = route_starts(problem, start, shape.xyz, budget.diameter_floor)
    opening_found, opening_drawn = seed_openings(maps, starts)
    measures = StartMeasures(reproduction, disagreement, opening_found, opening_drawn)

    fit_scaled = start.gap / config.loads.total
    opening = stiffness_spectrum(graph, start.q)

    report.write_heading("The start, and what the scarcity does to it")
    entries = start_entries(config, problem, start, measures, ceiling)
    entries.append(("lens fit gap / total load", f"{fit_scaled:.2e}"))
    entries.append(
        (
            "chord sign margin [N/mm]",
            f"{guard.margin:.1f} on {chords.size} chords, linear rows",
        )
    )
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

    width = int(finder.basis.shape[1])
    q_final = np.asarray(finder.basis) @ answers[ROUTE_FORMFOUND].variables[:width]
    landing = stiffness_spectrum(graph, q_final)
    signed_final = float(np.min(signs * q_final[chords]))

    xyz_heights = jnp.asarray(reads[ROUTE_HEIGHTS].xyz)
    lengths_heights = member_lengths(xyz_heights, problem.structure.edges)
    shortest = float(jnp.min(lengths_heights))

    report.write_heading("The degeneracies, watched at the answers")
    entries = (
        (
            "vertical stiffness at the start",
            f"{opening.negatives} negative of {opening.size}, "
            f"cond {opening.condition:.1e}",
        ),
        (
            "vertical stiffness at the answer",
            f"{landing.negatives} negative of {landing.size}, "
            f"cond {landing.condition:.1e}",
        ),
        (
            "chord sign slack at the answer [N/mm]",
            f"{signed_final:.1f} against the {guard.margin:.1f} margin",
        ),
        (
            "shortest member, free heights [mm]",
            f"{shortest:.0f} against the {budget.length_floor:.0f} floor",
        ),
    )
    report.write_entries(entries)

    report_summary(report, reads, config, ceiling)

    write_figures(problem, reads, answers, "19_vierendeel")
    report.write_heading(f"figures written to {FIGURES}")

    checks = [
        ToleranceCheck("gradient scaled error", best_error, TOLERANCE_GRADIENT),
        ToleranceCheck("projection gap", start.projection, TOLERANCE_PROJECTION),
        ToleranceCheck("lens reproduction [mm]", reproduction, TOLERANCE_SHAPE),
        ToleranceCheck("lens fit gap / total load", fit_scaled, TOLERANCE_FIT),
    ]
    undersign = max(0.0, (guard.margin - signed_final) / guard.scale)
    checks.append(
        ToleranceCheck("chord sign violation", undersign, TOLERANCE_FEASIBILITY)
    )
    undershort = max(0.0, (budget.length_floor - shortest) / budget.length_floor)
    checks.append(
        ToleranceCheck(
            "free heights length violation", undershort, TOLERANCE_FEASIBILITY
        )
    )
    routed, sound = route_checks(reads, answers, ceiling)
    checks.extend(routed)
    passed = checks_passed(tuple(checks)) and sound

    report.write_checks(tuple(checks))
    report.write_verdict(passed)


if __name__ == "__main__":
    jnp.set_printoptions(precision=12)
    described = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    main(described or Path(__file__).with_name("vierendeel_optimize.yaml"))
