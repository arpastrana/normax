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
The shear the design check leaves out, read off every converged design.

EN 1993-1-1 6.2.10 lets shear be ignored in the bending and axial checks while
the design shear stays under half the plastic shear resistance. That is an
exemption to be measured, not assumed, and a bound over a demand mix is not the
measurement: it is a statement about a member that might exist. This reads the
analyzed shear off designs that do exist — the arch at the simultaneous optimum
of experiment 103, and both trusses at each of the three answers experiments 18
and 19 descend to.

The demand is the vector resultant of the two shears, which a circular section
may take because it resists shear the same way in every direction. The
resistance is Eq. 6.18 over the shear area of 6.2.6(3), `A_v = 2A/pi`, at the
diameters the design settled on.

Run with `uv run --group pipeline python experiments/20_shear_audit.py`.
"""

import importlib.util
import sys
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from ec3x.material import Steel
from ec3x.resistance import SHEAR_THRESHOLD
from ec3x.resistance import area_shear
from ec3x.resistance import resistance_shear

from normax.analysis import MemberForces
from normax.reporting import Report
from normax.reporting import ReportColumn
from normax.reporting import ToleranceCheck
from normax.reporting import checks_passed
from normax.sections import TubeFamily

sys.path.insert(0, str(Path(__file__).resolve().parent))

from truss_routes import TrussProfile  # noqa: E402

EXPERIMENTS = Path(__file__).resolve().parent

# The partial factor every sizer in the repo states for itself, EN 1993-1-1 §6.1.
GAMMA_M0 = 1.0

# What a design may not reach if the exclusion of 6.2.6 is to stay honest.
TOLERANCE_EXEMPTION = SHEAR_THRESHOLD

# A newton-millimetre of torsion, which is a millionth of the moments in play.
# Declining 6.2.7 rests on the torsion being zero, so the zero gets asserted.
TOLERANCE_TORSION = 1.0


class ShearReading(NamedTuple):
    """
    One converged design's shear, against the resistance it was sized against.

    Attributes
    ----------
    label :
        Name of the design the reading was taken from.
    worst :
        Largest design shear over members and load cases, as a fraction of the
        plastic shear resistance.
    middle :
        Median of that fraction, which says whether the worst is a lone member
        or the whole frame.
    demand :
        Design shear of the worst member, in newtons.
    capacity :
        Plastic shear resistance of the worst member, in newtons.
    family :
        Name of the member family the worst member belongs to, blank where the
        design carries no family map.
    torsion :
        Largest torsional moment over members and load cases, in
        newton-millimetres. Declining 6.2.7 rests on this being zero.
    """

    label: str
    worst: float
    middle: float
    demand: float
    capacity: float
    family: str
    torsion: float


def family_of(
    member: int,
    members: int,
    families: tuple[tuple[str, slice], ...] | None,
) -> str:
    """
    Name of the family one member sits in.

    Parameters
    ----------
    member :
        Index of the member.
    members :
        Count of members, which resolves an open-ended slice.
    families :
        Name and span of every family, or None where the design has no map.

    Returns
    -------
    family :
        The family's name, or the empty string.
    """
    if families is None:
        return ""

    for name, span in families:
        first, last, _ = span.indices(members)
        if first <= member < last:
            return name

    return ""


def shear_reading(
    label: str,
    family: TubeFamily,
    diameters: np.ndarray,
    forces: MemberForces,
    families: tuple[tuple[str, slice], ...] | None = None,
) -> ShearReading:
    """
    Read one design's worst shear as a fraction of its plastic resistance.

    Parameters
    ----------
    label :
        Name the reading is reported under.
    family :
        The tube family the design's sections are drawn from.
    diameters :
        Outer diameter of every member, as the design settled it.
    forces :
        The analysis at that design, carrying the shear the check leaves out.
    families :
        Name and span of every member family, for attributing the worst member.

    Returns
    -------
    reading :
        The worst fraction, the median, and the member that carries it.

    Notes
    -----
    The two shears combine as a vector resultant rather than a sum, which is
    exact for a section that resists shear alike in every direction, and the
    resultant is the demand the one resistance is compared against.
    """
    sections = family(jnp.asarray(diameters))
    steel = Steel(f_y=family.material.f_y, gamma_m0=GAMMA_M0)
    capacity = np.asarray(resistance_shear(area_shear(sections.area), steel))

    major = np.asarray(forces.shear_major)
    minor = np.asarray(forces.shear_minor)
    demand = np.sqrt(major**2 + minor**2)
    fraction = demand / capacity

    members = fraction.shape[-1]
    flat = fraction.reshape(-1)
    worst = int(np.argmax(flat))
    member = worst % members
    named = family_of(member, members, families)

    twisted = float(np.max(np.abs(np.asarray(forces.torsion_moment))))

    return ShearReading(
        label,
        float(flat[worst]),
        float(np.median(flat)),
        float(demand.reshape(-1)[worst]),
        float(np.broadcast_to(capacity, fraction.shape).reshape(-1)[worst]),
        named,
        twisted,
    )


def load_experiment(name: str):
    """
    One experiment module, loaded by path rather than imported by name.

    Parameters
    ----------
    name :
        Stem of the experiment's file.

    Returns
    -------
    module :
        The loaded module.

    Notes
    -----
    The numbered experiments are not importable names, so this is the pattern
    experiment 102 established for reusing one from another.
    """
    spec = importlib.util.spec_from_file_location(name, EXPERIMENTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def read_the_arch() -> ShearReading:
    """
    The arch at the simultaneous optimum of experiment 103.

    Returns
    -------
    reading :
        Its worst shear as a fraction of the plastic resistance.

    Notes
    -----
    Experiment 103's own sequence, called rather than copied, so the design read
    here is the one that experiment reports.
    """
    showcase = load_experiment("103_simultaneous_api")
    api = showcase.load_showcase(EXPERIMENTS / "101_api.py")

    text = (EXPERIMENTS / "arch.yaml").read_text()
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

    return shear_reading(
        "arch, 103 optimum", pipeline.sizer.family, diameters, design.forces
    )


def read_a_truss(profile: TrussProfile, stem: str) -> tuple[ShearReading, ...]:
    """
    One truss's three answers, each read at its own converged design.

    Parameters
    ----------
    profile :
        The truss's profile, as its own experiment declares it.
    stem :
        Stem of the configuration file that experiment runs on.

    Returns
    -------
    readings :
        One reading per route, in the order the routes are raced.

    Notes
    -----
    The shared flow of `truss_routes` up to the reads, without its report: the
    descent is the same one experiments 18 and 19 run, so the answers read here
    are theirs.
    """
    routes = load_experiment("truss_routes")

    config = routes.parse_config((EXPERIMENTS / f"{stem}.yaml").read_text())
    budget = config.descent
    bays = config.structure.num_bays

    structure = profile.build_structure(
        bays, config.structure.span, config.structure.depth
    )
    mirrored = profile.mirrored_nodes(bays)
    problem = routes.prepare_problem(
        structure, config, mirrored, routes.mirrored_edges(mirrored, structure)
    )

    start = profile.signed_start(problem, config)
    finder = problem.pipeline.formfinder
    shape = finder.formfinder(jnp.asarray(start.q), problem.loads.formfinding)
    if profile.chord_guard is None:
        guard = None
    else:
        guard = profile.chord_guard(config, start)

    limits = routes.height_truss(budget, config.structure.depth)
    maps = routes.route_maps(problem, limits, budget.length_floor, guard)
    starts = routes.route_starts(problem, start, shape.xyz, budget.diameter_floor)
    boxes = routes.route_boxes(problem, budget.diameter_floor, limits)
    answers = routes.descend_all(Report(verbose=False), maps, starts, boxes, budget)
    reads = routes.route_reads(problem, answers, budget)

    families = profile.member_families(bays)
    readings = []
    for route in routes.ROUTE_ORDER:
        read = reads[route]
        forces = problem.pipeline.analyzer(
            jnp.asarray(read.xyz),
            jnp.asarray(read.diameters),
            problem.loads.analysis,
        )
        reading = shear_reading(
            f"{stem.split('_')[0]}, {route}",
            problem.pipeline.sizer.family,
            read.diameters,
            forces,
            families,
        )
        readings.append(reading)

    return tuple(readings)


def report_readings(report: Report, readings: tuple[ShearReading, ...]) -> None:
    """
    Every design's shear beside the exemption it has to stay under.

    Parameters
    ----------
    report :
        The report the table is written to.
    readings :
        One reading per design.
    """
    columns = (
        ReportColumn("design", "", "<"),
        ReportColumn("V_Ed [kN]", ".1f"),
        ReportColumn("V_pl,Rd [kN]", ".1f"),
        ReportColumn("worst", ".4f"),
        ReportColumn("median", ".4f"),
        ReportColumn("under 0.5 by", ".1f"),
        ReportColumn("worst member", "", "<"),
    )
    rows = [
        (
            reading.label,
            reading.demand / 1e3,
            reading.capacity / 1e3,
            reading.worst,
            reading.middle,
            SHEAR_THRESHOLD / reading.worst,
            reading.family,
        )
        for reading in readings
    ]
    report.write_table(columns, rows)


def main() -> None:
    """
    Read the excluded clause's demand off every converged design in the repo.
    """
    report = Report()
    report.write_banner("The shear the check leaves out, measured")

    warren = load_experiment("18_warren_optimize")
    vierendeel = load_experiment("19_vierendeel_optimize")

    readings = [read_the_arch()]
    readings.extend(read_a_truss(warren.WARREN_PROFILE, "warren_optimize"))
    readings.extend(read_a_truss(vierendeel.VIERENDEEL_PROFILE, "vierendeel_optimize"))
    ordered = tuple(sorted(readings, key=lambda reading: -reading.worst))

    report.write_heading("Design shear as a fraction of the plastic resistance")
    report_readings(report, ordered)
    report.write_note(
        "A Vierendeel carries its transverse load through frame action, so the "
        "shear lands in the verticals; a Warren hands the same load to its "
        "diagonals axially and barely sees any. Optimizing the geometry lowers "
        "the ratio rather than raising it: funicular members carry less moment, "
        "so the shear that moment differentiates to falls faster than the "
        "resistance does as the members thin."
    )

    report.write_heading("Are the exclusions of 6.2.6 and 6.2.7 still honest")
    shears = tuple(
        ToleranceCheck(f"{reading.label}, shear", reading.worst, TOLERANCE_EXEMPTION)
        for reading in ordered
    )
    torsions = tuple(
        ToleranceCheck(
            f"{reading.label}, torsion [N mm]", reading.torsion, TOLERANCE_TORSION
        )
        for reading in ordered
    )
    checks = shears + torsions
    report.write_checks(checks)
    report.write_verdict(checks_passed(checks))


if __name__ == "__main__":
    jnp.set_printoptions(precision=12)
    main()
