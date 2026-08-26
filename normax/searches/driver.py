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
A structure's profile, and the run that races the three searches over it.
"""

from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jaxtyping import Int

from normax.form_finding import equilibrium_graph
from normax.reporting import Report
from normax.reporting import ToleranceCheck
from normax.reporting import checks_passed
from normax.searches.config import TaskConfig
from normax.searches.descent import descend_all
from normax.searches.descent import descent_plan
from normax.searches.descent import load_answers
from normax.searches.descent import save_answers
from normax.searches.descent import seed_openings
from normax.searches.folding import ChordSigns
from normax.searches.folding import folding_maps
from normax.searches.loads import LoadPlan
from normax.searches.maps import HeightTruss
from normax.searches.maps import search_boxes
from normax.searches.maps import search_maps
from normax.searches.maps import search_starts
from normax.searches.maps import search_variables
from normax.searches.problem import DesignProblem
from normax.searches.problem import SearchRead
from normax.searches.problem import StartMeasures
from normax.searches.problem import StartPoint
from normax.searches.problem import ViewRequest
from normax.searches.problem import prepare_problem
from normax.searches.problem import stiffness_spectrum
from normax.searches.reporting import force_agreement
from normax.searches.reporting import report_families
from normax.searches.reporting import report_governing
from normax.searches.reporting import report_gradient
from normax.searches.reporting import report_searches
from normax.searches.reporting import report_summary
from normax.searches.reporting import search_checks
from normax.searches.reporting import search_reads
from normax.searches.reporting import start_entries
from normax.searches.reporting import write_figures
from normax.searches.settings import FIGURES
from normax.searches.settings import SEARCH_DRAWN
from normax.searches.settings import SEARCH_FORMFOUND
from normax.searches.settings import SEARCH_ORDER
from normax.searches.settings import TOLERANCE_FEASIBILITY
from normax.searches.settings import TOLERANCE_FIT
from normax.searches.settings import TOLERANCE_GRADIENT
from normax.searches.settings import TOLERANCE_PROJECTION
from normax.searches.settings import TOLERANCE_SHAPE
from normax.searches.settings import searches_present
from normax.structures import Structure
from normax.structures import member_lengths


class StructureProfile(NamedTuple):
    """
    What one structural family contributes to the shared three-search flow.

    Attributes
    ----------
    banner :
        Title the run's report opens with.
    prefix :
        Stem the run's figure files are named under.
    start_heading :
        Heading of the report's start block, carrying the family's story.
    parse_task :
        Reader of the run's YAML, the three family-shaped sections being the
        profile's to name and the other four `shared_sections`'.
    build_structure :
        The generator, taking the parsed run description.
    mirrored_nodes :
        The node the mirror carries each node onto.
    sections_rotated :
        The node a rotation carries each node onto, folding the diameters
        further, or None from a family with no rotation to offer and from a
        run that declines it.
    heights_rotated :
        The same, for the free-heights search's heights. Kept apart from
        `sections_rotated` because the two answer different questions —
        fabrication for the sections, and what the search comparison means for
        the heights — so a run may fold either without the other.
    member_families :
        Name and member slice of every family, in the generator's order.
    build_loads :
        The load cases, named, and the total the distributed ones carry.
    height_limits :
        The ceiling and floor the shape is held between, each family reading
        the multiple against the one length it is drawn by.
    signed_start :
        The start recipe — how the densities are fitted and signed differs
        per family, and each recipe is a measured decision.
    sign_guard :
        Builder of the density-sign rows, or None where the subspace has no
        degenerate states worth guarding. A builder may also answer None for
        a particular run, which is how a family makes its guard a switch.
    extent :
        The summary's one entry saying how far the shape travelled.

    Notes
    -----
    Everything else is topology-blind and lives in `run_searches`: an
    experiment is this record, a module docstring telling the family's
    story, and a YAML naming the run.
    """

    banner: str
    prefix: str
    start_heading: str
    parse_task: Callable[[str], TaskConfig]
    build_structure: Callable[[TaskConfig], Structure]
    mirrored_nodes: Callable[[TaskConfig], Int[np.ndarray, "nodes"]]
    sections_rotated: Callable[[TaskConfig], Int[np.ndarray, "nodes"] | None] | None
    heights_rotated: Callable[[TaskConfig], Int[np.ndarray, "nodes"] | None] | None
    member_families: Callable[[TaskConfig], tuple[tuple[str, slice], ...]]
    build_loads: Callable[[Structure, TaskConfig], LoadPlan]
    height_limits: Callable[[TaskConfig], HeightTruss]
    signed_start: Callable[[DesignProblem, TaskConfig], StartPoint]
    sign_guard: Callable[[TaskConfig, StartPoint], ChordSigns | None] | None
    extent: Callable[[TaskConfig, SearchRead], tuple[str, str]]


def run_searches(profile: StructureProfile, path: Path) -> ViewRequest | None:
    """
    Run one structure's three searches, write the report, and save the figures.

    Parameters
    ----------
    profile :
        The structural family to run.
    path :
        The YAML file describing the run.

    Notes
    -----
    A file asking for `solo_search` descends the viewer's search alone. Every
    table then holds one row, every comparison entry is dropped, and the
    verdict rests on convergence and feasibility rather than on beating a
    baseline that was never descended.
    """
    report = Report()
    report.write_banner(profile.banner)

    config = profile.parse_task(path.read_text())
    budget = config.descent

    structure = profile.build_structure(config)
    graph = equilibrium_graph(structure)
    plan = profile.build_loads(structure, config)
    folding_by = folding_maps(profile, config, structure)
    problem = prepare_problem(structure, config, plan, folding_by)

    start = profile.signed_start(problem, config)
    finder = problem.pipeline.formfinder
    shape = finder.formfinder(jnp.asarray(start.q), problem.loads.formfinding)
    reproduction = float(jnp.max(jnp.abs(shape.xyz - jnp.asarray(start.lens))))
    disagreement = force_agreement(problem, start, shape.xyz)

    if profile.sign_guard is None:
        guard = None
    else:
        guard = profile.sign_guard(config, start)

    limits = profile.height_limits(config)
    maps = search_maps(problem, limits, budget.length_floor, guard)
    starts = search_starts(problem, start, shape.xyz, budget.diameter_floor)
    opening_found, opening_drawn = seed_openings(maps, starts)
    measures = StartMeasures(reproduction, disagreement, opening_found, opening_drawn)

    fit_scaled = start.gap / plan.total
    opening = stiffness_spectrum(graph, start.q)

    report.write_heading(profile.start_heading)
    entries = start_entries(config, problem, start, measures, limits)
    entries.append(("lens fit gap / total load", f"{fit_scaled:.2e}"))
    held = problem.cases_held
    built = len(problem.case_names)
    if held.size < built:
        reindexed = [
            name for index, name in enumerate(problem.case_names) if index not in held
        ]
        mirrored = ", ".join(reindexed)
        entries.append(
            (
                "load cases with rows",
                f"{held.size} of {built}, {mirrored} reindexed onto a held case",
            )
        )
    if guard is not None:
        entries.append(
            (
                "chord sign margin [N/mm]",
                f"{guard.margin:.1f} on {guard.chords.size} chords, linear rows",
            )
        )
    report.write_entries(tuple(entries))

    if config.viewer.solo_search:
        searches = (config.viewer.search,)
    else:
        searches = SEARCH_ORDER

    shaped = [search for search in searches if search != SEARCH_DRAWN]
    errors = [
        report_gradient(report, maps[search], starts[search], search)
        for search in shaped
    ]
    best_error = max(errors) if errors else 0.0

    held = load_answers(config) if budget.reuse_answers else None
    recalled = held is not None and all(search in held for search in searches)

    named = "the three searches" if len(searches) > 1 else searches[0]
    boxes = search_boxes(problem, budget.diameter_floor, limits)
    if recalled:
        report.write_heading(f"Reading {named} back")
        answers = {search: held[search] for search in searches}
        for search in searches_present(answers):
            answer = answers[search]
            report.write_line(
                f"{search}: {answer.masses[-1]:.6f} t "
                f"in {answer.iterations} iterations, descended earlier"
            )
    else:
        report.write_heading(f"Descending {named}")
        descended = {search: maps[search] for search in searches}
        seeds = {search: starts[search] for search in searches}
        bounds = {search: boxes[search] for search in searches}
        answers = descend_all(report, descended, seeds, bounds, descent_plan(config))
        save_answers(path, config, answers)

    reads = search_reads(problem, answers, budget)
    report_searches(report, reads, answers, search_variables(problem))
    report_families(report, reads, profile.member_families(config))
    report_governing(report, reads, problem.case_names)

    width = int(finder.basis.shape[1])
    # The stiffness at the landing and the sign slack are the form finder's
    # own readouts, so a run without that search reports them at the start.
    if SEARCH_FORMFOUND in answers:
        q_final = np.asarray(finder.basis) @ answers[SEARCH_FORMFOUND].variables[:width]
    else:
        q_final = start.q
    landing = stiffness_spectrum(graph, q_final)

    shortest = {}
    for search in shaped:
        xyz_search = jnp.asarray(reads[search].xyz)
        lengths_search = member_lengths(xyz_search, problem.structure.edges)
        shortest[search] = float(jnp.min(lengths_search))

    report.write_heading("The degeneracies, watched at the answers")
    entries = [
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
    ]
    for search in shaped:
        entries.append(
            (
                f"shortest member, {search} [mm]",
                f"{shortest[search]:.0f} against the {budget.length_floor:.0f} floor",
            )
        )
    if guard is not None:
        signed_final = float(np.min(guard.signs * q_final[guard.chords]))
        entries.append(
            (
                "chord sign slack at the answer [N/mm]",
                f"{signed_final:.1f} against the {guard.margin:.1f} margin",
            )
        )
    report.write_entries(tuple(entries))

    report_summary(report, reads, config, limits, profile.extent)

    write_figures(problem, reads, answers, profile.prefix)
    report.write_heading(f"figures written to {FIGURES}")

    checks = [
        ToleranceCheck("gradient scaled error", best_error, TOLERANCE_GRADIENT),
        ToleranceCheck("projection gap", start.projection, TOLERANCE_PROJECTION),
        ToleranceCheck("lens reproduction [mm]", reproduction, TOLERANCE_SHAPE),
        ToleranceCheck("lens fit gap / total load", fit_scaled, TOLERANCE_FIT),
    ]
    # A held plan already floors every length at its own plan projection, so a
    # family that needs no floor of its own states zero and is not checked
    # against it.
    floor = budget.length_floor
    if floor > 0.0:
        for search in shaped:
            undershort = max(0.0, (floor - shortest[search]) / floor)
            checks.append(
                ToleranceCheck(
                    f"{search} length violation", undershort, TOLERANCE_FEASIBILITY
                )
            )
    if guard is not None:
        undersign = max(0.0, (guard.margin - signed_final) / guard.scale)
        checks.append(
            ToleranceCheck("chord sign violation", undersign, TOLERANCE_FEASIBILITY)
        )
    searched, sound = search_checks(reads, answers, limits)
    checks.extend(searched)
    passed = checks_passed(tuple(checks)) and sound

    report.write_checks(tuple(checks))
    report.write_verdict(passed)

    # A window holds the process until it closes, so the caller decides.
    if not config.viewer.enabled:
        return None

    return ViewRequest(problem, reads, (config.viewer.search,), config.viewer)
