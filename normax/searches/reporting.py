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
Reading a landing back, and every table written about it.
"""

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
from ec3x.material import Steel
from ec3x.resistance import SHEAR_THRESHOLD
from ec3x.resistance import area_shear
from ec3x.resistance import utilization_shear
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from normax.analysis import MemberForces
from normax.analysis import SmaxAnalyzer
from normax.reporting import Report
from normax.reporting import ReportColumn
from normax.reporting import ToleranceCheck
from normax.searches.config import DescentConfig
from normax.searches.config import TaskConfig
from normax.searches.folding import pattern_count
from normax.searches.folding import unfolded_values
from normax.searches.maps import HeightTruss
from normax.searches.maps import height_scale
from normax.searches.maps import limit_label
from normax.searches.problem import DesignProblem
from normax.searches.problem import SearchAnswer
from normax.searches.problem import SearchMaps
from normax.searches.problem import SearchRead
from normax.searches.problem import StartMeasures
from normax.searches.problem import StartPoint
from normax.searches.settings import ACTIVE_UTILIZATION
from normax.searches.settings import FIGURES
from normax.searches.settings import FLOOR_SLACK
from normax.searches.settings import GAMMA_M0_SHEAR
from normax.searches.settings import GRADIENT_STEPS
from normax.searches.settings import SEARCH_DRAWN
from normax.searches.settings import SEARCH_FORMFOUND
from normax.searches.settings import SEARCH_HEIGHTS
from normax.searches.settings import TIE_MARGIN
from normax.searches.settings import TOLERANCE_FEASIBILITY
from normax.searches.settings import TOLERANCE_GRADIENT
from normax.searches.settings import searches_present
from normax.sections import MemberSections
from normax.sections import TubeFamily
from normax.structures import member_lengths
from normax.visualization import DescentTrace
from normax.visualization import UtilizationForm
from normax.visualization import figure_mass_descent
from normax.visualization import figure_utilization


def shear_fraction(
    family: TubeFamily,
    sections: MemberSections,
    forces: MemberForces,
) -> float:
    """
    The design shear of the worst member, as a fraction of its plastic resistance.

    Parameters
    ----------
    family :
        The tube family the sections were drawn from, supplying the material.
    sections :
        The sections the answer settled on.
    forces :
        The analysis at that answer, carrying the shear the check leaves out.

    Returns
    -------
    fraction :
        Largest fraction over members and load cases.

    Notes
    -----
    Eq. 6.17 through `ec3x`, read once per component and taken at its worst
    rather than on a resultant of the two. The shear area of a tube is the same
    whichever way the force acts, which makes a resultant tempting, and whether
    one is sanctioned is an open question in that package — the worst component
    needs no ruling and is the same number wherever the other is zero, which on
    a planar frame it is.

    Read at the answer rather than bounded ahead of it: a bound over a demand
    mix describes a member that might exist, and what 6.2.10 asks about is the
    member that does.
    """
    steel = Steel(f_y=family.material.f_y, gamma_m0=GAMMA_M0_SHEAR)
    mobilized = area_shear(sections.area)

    major = np.asarray(utilization_shear(forces.shear_major, mobilized, steel))
    minor = np.asarray(utilization_shear(forces.shear_minor, mobilized, steel))

    return float(np.max(np.maximum(major, minor)))


def sheared_forces(
    problem: DesignProblem,
    xyz: Float[Array, "nodes 3"],
    diameters: Float[Array, "members"],
    forces: MemberForces,
) -> MemberForces:
    """
    An analysis carrying shear member by member, re-solved if the one given does not.

    Parameters
    ----------
    problem :
        The prepared structure, read for the frame the re-solve is built on.
    xyz :
        The answer's geometry.
    diameters :
        The answer's sections.
    forces :
        The analysis already read at that answer.

    Returns
    -------
    sheared :
        The same analysis where it already carries shear per member, and an
        in-process re-solve where it does not.

    Notes
    -----
    **A stage is entitled to carry less than the container it fills.** A
    `MemberForces` defaults its shear to a scalar zero, so a stage answering
    only axial force and end moments leaves that default in place, and the
    load-case stacking then makes three scalars into an array shaped like a
    load case axis and nothing else. Dividing that by a per-member resistance
    is what raises rather than what lies, which is the good outcome.

    The re-solve is the viewer's own fallback, for the viewer's own reason: a
    stage that does not carry a quantity cannot be asked for it, so the
    quantity is recomputed in process at the answer's own geometry and
    sections. It is a retelling, and the backend-agreement suite bounds it.

    **The test is what arrived, not which stage sent it.** A stage whose
    schema grows shear stops falling back here without this reading changing,
    and a stage that never carries it is covered whether or not it crosses a
    boundary.
    """
    members = int(problem.structure.num_edges)
    if np.ndim(forces.shear_major) > 1 and np.shape(forces.shear_major)[-1] == members:
        return forces

    analyzer = SmaxAnalyzer(problem.structure, problem.pipeline.sizer.family(100.0))

    return analyzer(xyz, diameters, problem.loads.analysis)


def read_answer(
    problem: DesignProblem,
    xyz: Float[Array, "nodes 3"],
    diameters: Float[np.ndarray, "edges"],
    budget: DescentConfig,
) -> SearchRead:
    """
    One answer read back as a design, at its own geometry and sections.

    Parameters
    ----------
    problem :
        The prepared truss.
    xyz :
        The answer's geometry.
    diameters :
        The answer's sections.
    budget :
        The budgets, read for the diameter floor.

    Returns
    -------
    read :
        The mass, the shape extremes, and the utilization member by member.
    """
    family = problem.pipeline.sizer.family

    lengths = member_lengths(xyz, problem.structure.edges)
    sized = jnp.asarray(diameters)
    forces = problem.pipeline.analyzer(xyz, sized, problem.loads.analysis)
    used = problem.pipeline.sizer.compute_utilization(sized, forces, lengths)

    sections = family(sized)
    mass = float(jnp.sum(sections.area * lengths) * family.material.density)

    utilization = np.asarray(jnp.max(used, axis=0))
    utilization_cases = np.asarray(used)
    active = int(np.sum(utilization > ACTIVE_UTILIZATION))
    floored = int(np.sum(diameters < budget.diameter_floor + FLOOR_SLACK))

    reflected = diameters[problem.edges_mirrored]
    mirror = float(np.max(np.abs(diameters - reflected)) / np.max(diameters))

    sheared = sheared_forces(problem, xyz, sized, forces)
    shear = shear_fraction(family, sections, sheared)

    read = SearchRead(
        mass,
        np.asarray(xyz),
        float(jnp.max(jnp.asarray(xyz)[:, 2])),
        float(jnp.min(jnp.asarray(xyz)[:, 2])),
        diameters,
        utilization,
        utilization_cases,
        active,
        floored,
        mirror,
        shear,
    )

    return read


def search_reads(
    problem: DesignProblem,
    answers: dict[str, SearchAnswer],
    budget: DescentConfig,
) -> dict[str, SearchRead]:
    """
    Every search's answer read back as a design, keyed by search.

    Parameters
    ----------
    problem :
        The prepared truss.
    answers :
        Every search's descent record.
    budget :
        The budgets, read for the diameter floor.

    Returns
    -------
    reads :
        Every answer at its own geometry and sections. A search the run never
        descended is absent rather than seeded, so every reader downstream
        sees the same searches the descent did.
    """
    width = int(problem.pipeline.formfinder.basis.shape[1])
    spread_heights = problem.folding.heights
    spread_diameters = problem.folding.diameters
    count = pattern_count(spread_heights, int(problem.nodes_free.shape[0]))

    reads = {}
    if SEARCH_FORMFOUND in answers:
        found = answers[SEARCH_FORMFOUND].variables
        xi_final = jnp.asarray(found[:width])
        shape_final = problem.pipeline.formfinder(xi_final, problem.loads.formfinding)
        d_found = unfolded_values(found[width:], spread_diameters)
        reads[SEARCH_FORMFOUND] = read_answer(problem, shape_final.xyz, d_found, budget)

    if SEARCH_HEIGHTS in answers:
        heights = answers[SEARCH_HEIGHTS].variables
        z_final = unfolded_values(heights[:count], spread_heights)
        xyz_heights = problem.structure.nodes.at[problem.nodes_free, 2].set(
            jnp.asarray(z_final)
        )
        d_heights = unfolded_values(heights[count:], spread_diameters)
        reads[SEARCH_HEIGHTS] = read_answer(problem, xyz_heights, d_heights, budget)

    if SEARCH_DRAWN in answers:
        d_drawn = unfolded_values(answers[SEARCH_DRAWN].variables, spread_diameters)
        drawn = problem.structure.nodes
        reads[SEARCH_DRAWN] = read_answer(problem, drawn, d_drawn, budget)

    return reads


def force_agreement(
    problem: DesignProblem,
    start: StartPoint,
    xyz: Float[Array, "nodes 3"],
) -> float:
    """
    How far the elastic axial forces sit from the funicular prediction.

    Parameters
    ----------
    problem :
        The prepared truss.
    start :
        The signed lens fit, supplying the funicular densities.
    xyz :
        The lens geometry the frame is analyzed at.

    Returns
    -------
    disagreement :
        Worst `|N - q L|` under the shaping case, scaled by the largest
        funicular force.

    Notes
    -----
    On the determinate arch this number sat at solver precision. On an
    indeterminate truss the elastic frame chooses its own load-path split,
    so the disagreement is structural rather than numerical — the reason T1
    hands T2 geometry only, never member forces.
    """
    lengths = member_lengths(xyz, problem.structure.edges)
    seed = problem.diameters_seed
    forces = problem.pipeline.analyzer(xyz, seed, problem.loads.analysis)

    funicular = start.q * np.asarray(lengths)
    elastic = np.asarray(forces.axial_force)[0]
    disagreement = float(np.abs(elastic - funicular).max() / np.abs(funicular).max())

    return disagreement


def start_entries(
    config: TaskConfig,
    problem: DesignProblem,
    start: StartPoint,
    measures: StartMeasures,
    limits: HeightTruss,
) -> list[tuple[str, str]]:
    """
    The start block's entries: the basis, the limits, and the seed numbers.

    Parameters
    ----------
    config :
        The run description.
    problem :
        The prepared truss, supplying the variable counts.
    start :
        The signed lens fit, read for the projection gap.
    measures :
        The seed numbers, measured before any descent has moved.
    limits :
        The ceiling and the floor no vertex may leave.

    Returns
    -------
    entries :
        Label-and-value pairs, for the experiment to extend and write.
    """
    width = int(problem.pipeline.formfinder.basis.shape[1])
    count = pattern_count(problem.folding.heights, int(problem.nodes_free.shape[0]))
    searched = "symmetric" if config.subspace.symmetric else "full"
    lidded = limit_label(limits.ceiling, config.descent.rise_factor)
    grounded = limit_label(limits.floor, config.descent.sag_factor)
    elastic = f"{measures.disagreement:.1%} of the largest force"

    entries = [
        ("searched basis", f"{searched} {config.subspace.basis}"),
        ("rise ceiling", lidded),
        ("sag floor", grounded),
        ("geometry variables, end to end", f"{width}"),
        ("geometry variables, free heights", f"{count}"),
        ("projection gap", f"{start.projection:.2e} of |q|"),
        ("lens reproduction [mm]", f"{measures.reproduction:.2e}"),
        ("elastic vs funicular, LC1", elastic),
        ("seed envelope infeasibility, lens", f"{-measures.opening_found:.1%}"),
        ("seed envelope infeasibility, drawn", f"{-measures.opening_drawn:.1%}"),
    ]

    return entries


def report_gradient(
    report: Report,
    maps: SearchMaps,
    start: Float[np.ndarray, "variables"],
    label: str,
) -> float:
    """
    One search's gradient against a directional central difference.

    Parameters
    ----------
    report :
        Where the sweep is written.
    maps :
        The search's compiled maps.
    start :
        The point the derivative is taken at.
    label :
        Name of the search, for the heading.

    Returns
    -------
    best :
        The smallest scaled disagreement over the swept steps.

    Notes
    -----
    One seeded random direction rather than the coordinate axes: the probe
    moves the geometry variables and the diameters at once, which is exactly
    the mixture SLSQP steps through — for the end-to-end search the form
    finder and the frame analysis inside the same derivative, for the
    free-heights search the coordinates straight into the analysis.
    """
    generator = np.random.default_rng(2026)
    drawn = generator.normal(size=start.shape[0])
    direction = drawn / np.linalg.norm(drawn)

    point = jnp.asarray(start)
    _, slope = maps.weigh(point)
    exact = float(jnp.sum(slope * jnp.asarray(direction)))
    magnitude = float(np.linalg.norm(start))

    columns = (
        ReportColumn("relative step", ".0e"),
        ReportColumn("central difference", ".9e"),
        ReportColumn("scaled error", ".2e"),
    )
    rows = []
    best = float("inf")
    for relative in GRADIENT_STEPS:
        step = magnitude * relative
        forward, _ = maps.weigh(point + step * jnp.asarray(direction))
        backward, _ = maps.weigh(point - step * jnp.asarray(direction))
        quotient = (float(forward) - float(backward)) / (2.0 * step)
        scaled = abs(exact - quotient) / abs(exact)
        best = min(best, scaled)
        rows.append((relative, quotient, scaled))

    entries = (
        ("exact directional derivative", f"{exact:.9e}"),
        ("best scaled error", f"{best:.2e} ({TOLERANCE_GRADIENT:.0e})"),
    )

    report.write_heading(f"The {label} gradient, checked at the start")
    report.write_table(columns, rows)
    report.write_entries(entries)

    return best


def report_searches(
    report: Report,
    reads: dict[str, SearchRead],
    answers: dict[str, SearchAnswer],
    variables: dict[str, int],
) -> None:
    """
    The searches side by side, by every measure the comparison makes.

    Parameters
    ----------
    report :
        Where the table is written.
    reads :
        Each search's answer read back, keyed by search.
    answers :
        Each search's descent record, keyed by search.
    variables :
        Each search's variable count, keyed by search.
    """
    columns = (
        ReportColumn("search", align="<"),
        ReportColumn("variables"),
        ReportColumn("iterations"),
        ReportColumn("converged", align="<"),
        ReportColumn("mass [t]", ".6f"),
        ReportColumn("rise [mm]", ".0f"),
        ReportColumn("sag [mm]", ".0f"),
        ReportColumn("max U", ".9f"),
        ReportColumn("fully stressed"),
        ReportColumn("at floor"),
    )
    rows = []
    for search in searches_present(reads):
        read = reads[search]
        rows.append(
            (
                search,
                variables[search],
                answers[search].iterations,
                "yes" if answers[search].converged else "NO",
                read.mass,
                read.rise,
                read.sag,
                float(read.utilization.max()),
                read.active,
                read.floored,
            )
        )

    report.write_heading("The searches, side by side")
    report.write_table(columns, rows)


def report_families(
    report: Report,
    reads: dict[str, SearchRead],
    families: tuple[tuple[str, slice], ...],
) -> None:
    """
    Sections and utilizations family by family, for every search.

    Parameters
    ----------
    report :
        Where the table is written.
    reads :
        Each search's answer read back, keyed by search.
    families :
        Name and member slice of every family, in the generator's order.
    """
    columns = (
        ReportColumn("search", align="<"),
        ReportColumn("family", align="<"),
        ReportColumn("d min [mm]", ".1f"),
        ReportColumn("d max [mm]", ".1f"),
        ReportColumn("U min", ".3f"),
        ReportColumn("U max", ".3f"),
    )
    rows = []
    for search in searches_present(reads):
        read = reads[search]
        for name, members in families:
            rows.append(
                (
                    search,
                    name,
                    float(read.diameters[members].min()),
                    float(read.diameters[members].max()),
                    float(read.utilization[members].min()),
                    float(read.utilization[members].max()),
                )
            )

    report.write_heading("Sections and utilization, family by family")
    report.write_table(columns, rows)


def governed_counts(read: SearchRead) -> Int[np.ndarray, "cases"]:
    """
    How many members each load case governs, ties counted toward each case.

    Parameters
    ----------
    read :
        One search's answer read back, supplying the utilization table.

    Returns
    -------
    counts :
        Members per case, a tied member appearing under each of its cases.

    Notes
    -----
    A member counts toward every case within the tie margin of its worst,
    so the counts may sum past the member count. Splitting ties by index
    order instead misreports a symmetric design: the mirror pairs the two
    half-span cases exactly on self-mirrored members and to solver
    precision everywhere else, and the first index would collect every
    such coin flip.
    """
    table = read.utilization_cases
    worst = table.max(axis=0)
    tied = table >= worst[None, :] - TIE_MARGIN

    return tied.sum(axis=1)


def report_governing(
    report: Report,
    reads: dict[str, SearchRead],
    names: tuple[str, ...],
) -> None:
    """
    How many members each load case governs, per search.

    Parameters
    ----------
    report :
        Where the table is written.
    reads :
        Each search's answer read back, keyed by search.
    names :
        Name of every built load case, in build order.
    """
    columns = [ReportColumn("search", align="<")]
    for name in names:
        columns.append(ReportColumn(name))

    rows = []
    for search in searches_present(reads):
        counts = governed_counts(reads[search])
        rows.append((search, *[int(count) for count in counts]))

    report.write_heading("Members governed, case by case")
    report.write_table(tuple(columns), rows)


def truss_extent(config: TaskConfig, read: SearchRead) -> tuple[str, str]:
    """
    The depth a truss reached, against the depth it was drawn at.

    Parameters
    ----------
    config :
        The run description, read for the drawn depth.
    read :
        The end-to-end answer, read for the rise and the sag it spans.

    Returns
    -------
    entry :
        The label and its value, ready for the summary block.
    """
    reached = read.rise - read.sag

    return (
        "depth at the answer [mm]",
        f"{reached:.0f}, drawn at {config.structure.depth:.0f}",
    )


def shell_extent(config: TaskConfig, read: SearchRead) -> tuple[str, str]:
    """
    The rise a shell reached, against the rise it was drawn at.

    Parameters
    ----------
    config :
        The run description, read for the drawn rise.
    read :
        The end-to-end answer, read for the height its crown reached.

    Returns
    -------
    entry :
        The label and its value, ready for the summary block.
    """
    return (
        "rise at the answer [mm]",
        f"{read.rise:.0f}, drawn at {config.structure.rise:.0f}",
    )


def report_summary(
    report: Report,
    reads: dict[str, SearchRead],
    config: TaskConfig,
    limits: HeightTruss,
    extent: Callable[[TaskConfig, SearchRead], tuple[str, str]],
) -> None:
    """
    The masses, the gaps between the searches, and the shape extremes.

    Parameters
    ----------
    report :
        Where the summary is written.
    reads :
        Each search's answer read back, keyed by search.
    config :
        The run description, read for the limit factors.
    limits :
        The ceiling and the floor no vertex may leave.
    extent :
        The profile's one entry saying how far the shape travelled, a truss
        reading it as a depth and a shell as a rise.
    """
    searches = searches_present(reads)
    lidded = limit_label(limits.ceiling, config.descent.rise_factor)
    grounded = limit_label(limits.floor, config.descent.sag_factor)

    entries = [
        (f"mass, {search}", f"{reads[search].mass:.6f} t") for search in searches
    ]

    both = SEARCH_FORMFOUND in reads and SEARCH_DRAWN in reads
    if both:
        saving = 1.0 - reads[SEARCH_FORMFOUND].mass / reads[SEARCH_DRAWN].mass
        entries.append(("the geometry bought", f"{saving:.1%}"))

    shaped = SEARCH_FORMFOUND in reads and SEARCH_HEIGHTS in reads
    if shaped:
        read_found = reads[SEARCH_FORMFOUND]
        read_heights = reads[SEARCH_HEIGHTS]
        searches_gap = read_heights.mass / read_found.mass - 1.0
        heights_z = read_heights.xyz[:, 2]
        shapes_gap = float(np.abs(read_found.xyz[:, 2] - heights_z).max())
        entries.append(("free heights vs end to end", f"{searches_gap:+.2%}"))
        entries.append(("the shaped answers differ by [mm]", f"{shapes_gap:.0f}"))

    leading = SEARCH_FORMFOUND if SEARCH_FORMFOUND in reads else searches[0]
    travelled = reads[leading]
    entries.append(extent(config, travelled))
    entries.append(("rise ceiling", lidded))
    entries.append(("sag floor", grounded))
    for search in searches:
        mirrored = f"{reads[search].mirror:.2e}"
        entries.append((f"diameter mirror gap, {search}", mirrored))

    report.write_heading("Summary")
    report.write_entries(tuple(entries))


def search_checks(
    reads: dict[str, SearchRead],
    answers: dict[str, SearchAnswer],
    limits: HeightTruss,
) -> tuple[list[ToleranceCheck], bool]:
    """
    Every search's feasibility checks, and the converged-and-lighter verdict.

    Parameters
    ----------
    reads :
        Each search's answer read back, keyed by search.
    answers :
        Each search's descent record, keyed by search.
    limits :
        The ceiling and the floor no vertex may leave.

    Returns
    -------
    checks :
        Constraint, rise, sag and excluded-shear checks, one of each per search.
    sound :
        Whether every search converged and both shaped searches beat sizing.
    """
    searches = searches_present(reads)
    checks = []
    for search in searches:
        violation = max(0.0, float(reads[search].utilization.max()) - 1.0)
        checks.append(
            ToleranceCheck(
                f"{search} constraint violation", violation, TOLERANCE_FEASIBILITY
            )
        )
    shaped = [search for search in searches if search != SEARCH_DRAWN]
    if limits.ceiling is not None:
        for search in shaped:
            overrise = max(0.0, (reads[search].rise - limits.ceiling) / limits.ceiling)
            checks.append(
                ToleranceCheck(
                    f"{search} rise violation", overrise, TOLERANCE_FEASIBILITY
                )
            )
    if limits.floor is not None:
        for search in shaped:
            below = limits.floor - reads[search].sag
            oversag = max(0.0, below / height_scale(limits))
            checks.append(
                ToleranceCheck(
                    f"{search} sag violation", oversag, TOLERANCE_FEASIBILITY
                )
            )

    for search in searches:
        checks.append(
            ToleranceCheck(
                f"{search} shear fraction", reads[search].shear, SHEAR_THRESHOLD
            )
        )

    converged = all(answers[search].converged for search in searches)
    # A solo run has nothing to be lighter than, so it is judged on
    # convergence and feasibility alone.
    lighter = all(
        reads[search].mass < reads[SEARCH_DRAWN].mass
        for search in shaped
        if SEARCH_DRAWN in reads
    )
    sound = converged and lighter

    return checks, sound


def write_figures(
    problem: DesignProblem,
    reads: dict[str, SearchRead],
    answers: dict[str, SearchAnswer],
    prefix: str,
) -> None:
    """
    The three final designs, and the descents that reached them.

    Parameters
    ----------
    problem :
        The prepared truss, supplying the drawn geometry to outline.
    reads :
        Each search's answer read back, keyed by search.
    answers :
        Each search's descent record, keyed by search.
    prefix :
        Stem the figure files are named under, e.g. `18_warren`.
    """
    FIGURES.mkdir(exist_ok=True)

    forms = []
    for search in searches_present(reads):
        read = reads[search]
        title = f"{search} — {read.mass:.4f} t"
        counts = governed_counts(read)
        drawn = UtilizationForm(
            title, read.xyz, read.diameters, read.utilization, counts
        )
        forms.append(drawn)
    designs = figure_utilization(
        problem.structure.edges,
        forms,
        problem.case_names,
        reference=problem.structure.nodes,
    )
    designs.savefig(FIGURES / f"{prefix}_designs.png", dpi=200, bbox_inches="tight")

    variables = {
        SEARCH_FORMFOUND: "ξ and d",
        SEARCH_HEIGHTS: "z and d",
        SEARCH_DRAWN: "d",
    }
    traces = tuple(
        DescentTrace(f"{search} ({variables[search]})", answers[search].masses)
        for search in searches_present(answers)
    )
    descents = figure_mass_descent(traces)
    descents.savefig(FIGURES / f"{prefix}_descent.png", dpi=200, bbox_inches="tight")
