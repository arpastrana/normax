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
The member buckling clause, read off every converged design.

EN 1993-1-1 6.3.1 is the clause the pipeline exists to differentiate through:
lengthening a member raises its slenderness and drops the reduction factor, and
a form finder alone cannot see that term. That is a claim to be measured rather
than asserted. A clause that never binds prices nothing, and a reduction factor
pinned at one would make the whole member check a cross-section check wearing a
longer name.

This reads the slenderness and the reduction factor off designs that exist —
the arch at the simultaneous optimum of experiment 103, and both trusses and
the gridshell at each answer experiments 18, 19 and 23 descend to.

Two things are checked. **Whether the clause binds**: 6.3.1.2(3) leaves the
factor at one below a slenderness of 0.2, so a design sitting entirely under
that offset would be paying for machinery it never uses. And **how much it
takes**: the factor is the fraction of the yielding resistance a compressed
member keeps, so one minus it is what buckling removed at the size the design
settled on.

The slenderness is Eq. 6.50 by way of the radius of gyration, at the buckling
length the sizer was given, which is the member length under the standing
policy. The imperfection factor is curve a, hot-finished.

Run with `uv run --group pipeline python experiments/26_buckling_audit.py`.
"""

import importlib.util
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from ec3x.resistance import SLENDERNESS_OFFSET
from ec3x.resistance import reduction_buckling
from normax.searches import StructureProfile

from normax import searches
from normax.reporting import Report
from normax.reporting import ReportColumn
from normax.reporting import ToleranceCheck
from normax.reporting import verify_checks
from normax.sections import TubeFamily
from normax.structures import compute_member_lengths

# The run descriptions the examples take, and the folder this file sits in.
EXPERIMENTS = Path(__file__).resolve().parents[1]
VALIDATION = Path(__file__).resolve().parent

# The four examples, which own the run descriptions an audit reads designs from.
EXAMPLES = Path(__file__).resolve().parents[2] / "examples"

# EN 1993-1-1 Table 6.2, curve a: hot-finished circular hollow sections.
ALPHA_HOT_FINISHED = 0.21

# A factor at or above this is one for reporting, short of floating-point noise.
UNREDUCED = 0.999


class BucklingReading(NamedTuple):
    """
    What 6.3.1 did to one converged design.

    Attributes
    ----------
    label :
        Name of the design the reading came from.
    shaped :
        Whether the search that reached it was free to move the geometry.
    members :
        Count of members in it.
    compressed :
        Count of members some load case puts in compression.
    reduced :
        Count of members whose reduction factor falls short of one.
    slenderness_median :
        Median relative slenderness over every member.
    slenderness_worst :
        Largest relative slenderness over every member.
    reduction_median :
        Median reduction factor over the compressed members.
    reduction_worst :
        Smallest reduction factor over the compressed members.
    """

    label: str
    shaped: bool
    members: int
    compressed: int
    reduced: int
    slenderness_median: float
    slenderness_worst: float
    reduction_median: float
    reduction_worst: float

    @property
    def capacity_removed(self) -> float:
        """
        The fraction of the yielding resistance buckling took, at the median.
        """
        return 1.0 - self.reduction_median


def loaded_module(path: Path):
    """
    One script, loaded by path rather than imported by name.

    Parameters
    ----------
    path :
        The file to load.

    Returns
    -------
    module :
        The loaded module.

    Notes
    -----
    Neither the examples nor the numbered experiments are importable names, so
    a script that reuses another reaches it by path. Taking the file rather
    than a stem is what lets one loader serve both folders.
    """
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def buckling_reading(
    label: str,
    shaped: bool,
    family: TubeFamily,
    diameters: np.ndarray,
    lengths: np.ndarray,
    axial: np.ndarray,
) -> BucklingReading:
    """
    One design's slenderness and reduction factor, member by member.

    Parameters
    ----------
    label :
        Name the reading is reported under.
    shaped :
        Whether the search that reached it was free to move the geometry.
    family :
        The tube family the design was sized from, read for the material.
    diameters :
        Outer diameter of every member.
    lengths :
        Length of every member, taken as the buckling length.
    axial :
        Axial force of every member in every load case, tension positive.

    Returns
    -------
    reading :
        The slenderness and reduction extremes over the design.

    Notes
    -----
    The reference slenderness is `pi sqrt(E / f_y)`, so the relative
    slenderness is the geometric slenderness measured against it. A circular
    hollow section is axisymmetric, so one radius of gyration serves both axes
    and there is no second curve to take the worse of.
    """
    steel = family.material
    sections = family(jnp.asarray(diameters))

    reference = float(np.pi * np.sqrt(steel.e_mod / steel.f_y))
    slender = jnp.asarray(lengths) / sections.radius_of_gyration / reference
    reduction = reduction_buckling(slender, ALPHA_HOT_FINISHED)

    slenderness = np.asarray(slender)
    factor = np.asarray(reduction)
    compressed = np.asarray(jnp.min(jnp.atleast_2d(axial), axis=0)) < 0.0

    if compressed.any():
        among = factor[compressed]
    else:
        among = factor

    return BucklingReading(
        label,
        shaped,
        int(slenderness.size),
        int(compressed.sum()),
        int((factor < UNREDUCED).sum()),
        float(np.median(slenderness)),
        float(np.max(slenderness)),
        float(np.median(among)),
        float(np.min(among)),
    )


def read_the_arch() -> BucklingReading:
    """
    The arch at the simultaneous optimum of experiment 103.

    Returns
    -------
    reading :
        What 6.3.1 took from it.

    Notes
    -----
    Experiment 103's own sequence, called rather than copied, following the
    pattern experiment 20 established for reaching this design.
    """
    showcase = loaded_module(VALIDATION / "103_simultaneous_api.py")
    api = showcase.load_showcase(EXAMPLES / "arch.py")

    text = (EXAMPLES / "arch.yaml").read_text()
    config = api.parse_config(text)
    searched = showcase.parse_simultaneous(text)

    structure = api.build_arch(config.structure)
    loads = api.arch_load_cases(structure, config.load_cases)
    pipeline = api.build_pipeline(structure, config)
    params = api.initialize_parameters(structure, config)
    layout = showcase.variable_layout(searched.force_densities, structure.num_edges)

    floor = config.optimization.length_floor
    bay = config.structure.span / config.structure.num_edges
    seed_shape = pipeline.formfinder(params.force_densities, loads.formfinding)
    constraints = showcase.shape_constraints(
        searched, structure, seed_shape, floor.fraction * bay
    )

    problem = showcase.constrained_problem(pipeline, loads, params, layout, constraints)
    answer = showcase.solve_constrained(
        problem, config.optimization.bounds, searched, layout
    )
    densities, diameters = showcase.spread_variables(layout, answer.variables)
    design = showcase.assemble_design(pipeline, loads, densities, diameters)

    return buckling_reading(
        "arch, 103 optimum",
        True,
        pipeline.sizer.family,
        np.asarray(diameters),
        np.asarray(design.shape.lengths),
        np.asarray(design.forces.axial_force),
    )


def read_a_structure(
    profile: StructureProfile,
    described: Path,
) -> tuple[BucklingReading, ...]:
    """
    One structure's answers, each read at its own converged design.

    Parameters
    ----------
    profile :
        The structure's profile, as its own experiment declares it.
    described :
        Stem of the configuration file that experiment runs on.

    Returns
    -------
    readings :
        One reading per search the run descended, in the order they are raced.

    Notes
    -----
    The shared flow of `normax.searches` up to the reads, without its report, so
    the answers read here are the ones those experiments report. A stored
    answer is reused where the run description allows it.
    """
    config = profile.parse_task(described.read_text())
    budget = config.descent

    structure = profile.build_structure(config)
    plan = profile.build_loads(structure, config)
    folding_by = searches.folding_maps(profile, config, structure)
    problem = searches.prepare_problem(structure, config, plan, folding_by)

    start = profile.signed_start(problem, config)
    finder = problem.pipeline.formfinder
    shape = finder.formfinder(jnp.asarray(start.q), problem.loads.formfinding)
    if profile.sign_guard is None:
        guard = None
    else:
        guard = profile.sign_guard(config, start)

    limits = profile.height_limits(config)
    maps = searches.search_maps(problem, limits, budget.length_floor, guard)
    starts = searches.search_starts(problem, start, shape.xyz, budget.diameter_floor)
    boxes = searches.search_boxes(problem, budget.diameter_floor, limits)
    descending = searches.descent_plan(config)
    quiet = Report(verbose=False)
    answers = searches.descend_all(quiet, maps, starts, boxes, descending)
    reads = searches.search_reads(problem, answers, budget)

    readings = []
    for search in searches.SEARCH_ORDER:
        if search not in reads:
            continue
        read = reads[search]
        xyz = jnp.asarray(read.xyz)
        diameters = jnp.asarray(read.diameters)
        forces = problem.pipeline.analyzer(xyz, diameters, problem.loads.analysis)
        lengths = compute_member_lengths(xyz, problem.structure.edges)
        reading = buckling_reading(
            f"{described.stem.split('_')[0]}, {search}",
            search != searches.SEARCH_DRAWN,
            problem.pipeline.sizer.family,
            np.asarray(diameters),
            np.asarray(lengths),
            np.asarray(forces.axial_force),
        )
        readings.append(reading)

    return tuple(readings)


READING_COLUMNS = (
    ReportColumn("design", "", "<"),
    ReportColumn("members"),
    ReportColumn("compressed"),
    ReportColumn("reduced"),
    ReportColumn("median lambda", ".3f"),
    ReportColumn("worst lambda", ".3f"),
    ReportColumn("median chi", ".4f"),
    ReportColumn("worst chi", ".4f"),
    ReportColumn("capacity taken", ".1%"),
)


def report_readings(report: Report, readings: tuple[BucklingReading, ...]) -> None:
    """
    Every design's slenderness and reduction factor, one row each.

    Parameters
    ----------
    report :
        Where the table is written.
    readings :
        The readings, in the order they are to appear.
    """
    rows = [
        (
            reading.label,
            reading.members,
            reading.compressed,
            reading.reduced,
            reading.slenderness_median,
            reading.slenderness_worst,
            reading.reduction_median,
            reading.reduction_worst,
            reading.capacity_removed,
        )
        for reading in readings
    ]

    report.write_heading("Relative slenderness and the reduction factor it earns")
    report.write_table(READING_COLUMNS, rows)


def main() -> None:
    """
    Read 6.3.1 off every converged design and say whether it binds.
    """
    report = Report()
    report.write_banner("The buckling clause, read off every converged design")

    trusses = loaded_module(EXAMPLES / "warren.py")
    vierendeel = loaded_module(EXAMPLES / "vierendeel.py")
    gridshell = loaded_module(EXAMPLES / "gridshell.py")

    readings = (
        read_the_arch(),
        *read_a_structure(trusses.WARREN_PROFILE, EXAMPLES / "warren.yaml"),
        *read_a_structure(vierendeel.VIERENDEEL_PROFILE, EXAMPLES / "vierendeel.yaml"),
        *read_a_structure(
            gridshell.GRIDSHELL_PROFILE, EXPERIMENTS / "gridshell_16.yaml"
        ),
    )

    report_readings(report, readings)

    report.write_heading("Does the clause bind at all")
    report.write_note(
        "6.3.1.2(3) leaves the factor at one below a relative slenderness of "
        f"{SLENDERNESS_OFFSET}. A design entirely below it would make the "
        "member check a cross-section check under another name. What is "
        "asserted is the narrower claim the designs support: wherever the "
        "search could move the geometry, every member is reduced. A search "
        "that may not move it is reported and not asserted, because a "
        "structure drawn stocky is entitled to sit under the offset."
    )

    unreduced = [
        reading.label
        for reading in readings
        if reading.shaped and reading.reduced < reading.members
    ]
    frozen = [
        f"{reading.label} ({reading.reduced}/{reading.members})"
        for reading in readings
        if not reading.shaped and reading.reduced < reading.members
    ]
    checks = (
        ToleranceCheck("shaped designs holding an unreduced member", len(unreduced), 1),
    )
    report.write_checks(checks)

    if unreduced:
        report.write_note("Not every member reduced in: " + ", ".join(unreduced))
    if frozen:
        report.write_note(
            "At a geometry the search could not move, partly under the "
            "offset: " + ", ".join(frozen)
        )

    report.write_verdict(verify_checks(checks))


if __name__ == "__main__":
    main()
