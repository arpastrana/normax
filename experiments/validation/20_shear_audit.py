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
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from ec3x.material import Steel
from ec3x.resistance import SHEAR_THRESHOLD
from ec3x.resistance import area_shear
from ec3x.resistance import resistance_shear
from ec3x.resistance import utilization_shear
from normax.searches import StructureProfile

from normax import searches
from normax.analysis import MemberForces
from normax.reporting import Report
from normax.reporting import ReportColumn
from normax.reporting import ToleranceCheck
from normax.reporting import verify_checks
from normax.sections import TubeFamily

# The run descriptions the examples take, and the folder this file sits in.
EXPERIMENTS = Path(__file__).resolve().parents[1]
VALIDATION = Path(__file__).resolve().parent

# The four examples, which own the run descriptions an audit reads designs from.
EXAMPLES = Path(__file__).resolve().parents[2] / "examples"

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
        Design shear of the worst member, in newtons, on the component that
        governs it.
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
    Eq. 6.17 through `ec3x`, once per component and taken at its worst. The two
    are deliberately not combined: a tube's shear area is the same whichever way
    the force acts, which makes a resultant tempting, but whether one is
    sanctioned is an open question in that package and the worst component needs
    no ruling. On a planar frame the minor component is zero, so the two readings
    coincide, and this asserts that rather than assuming it.
    """
    sections = family(jnp.asarray(diameters))
    steel = Steel(f_y=family.material.f_y, gamma_m0=GAMMA_M0)
    mobilized = area_shear(sections.area)

    major = np.asarray(utilization_shear(forces.shear_major, mobilized, steel))
    minor = np.asarray(utilization_shear(forces.shear_minor, mobilized, steel))
    fraction = np.maximum(major, minor)

    resultant = np.sqrt(major**2 + minor**2)
    if not np.allclose(resultant, fraction, rtol=0.0, atol=0.0):
        raise ValueError("a minor-axis shear is present: the two readings differ")

    members = fraction.shape[-1]
    flat = fraction.reshape(-1)
    worst = int(np.argmax(flat))
    member = worst % members
    named = family_of(member, members, families)

    twisted = float(np.max(np.abs(np.asarray(forces.torsion_moment))))

    shears = np.maximum(
        np.abs(np.asarray(forces.shear_major)),
        np.abs(np.asarray(forces.shear_minor)),
    )
    capacity = np.asarray(resistance_shear(mobilized, steel))

    return ShearReading(
        label,
        float(flat[worst]),
        float(np.median(flat)),
        float(np.broadcast_to(shears, fraction.shape).reshape(-1)[worst]),
        float(np.broadcast_to(capacity, fraction.shape).reshape(-1)[worst]),
        named,
        twisted,
    )


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

    return shear_reading(
        "arch, 103 optimum", pipeline.sizer.family, diameters, design.forces
    )


def read_a_truss(
    profile: StructureProfile,
    described: Path,
) -> tuple[ShearReading, ...]:
    """
    One truss's three answers, each read at its own converged design.

    Parameters
    ----------
    profile :
        The truss's profile, as its own experiment declares it.
    described :
        Stem of the configuration file that experiment runs on.

    Returns
    -------
    readings :
        One reading per search, in the order the searches are raced.

    Notes
    -----
    The shared flow of `normax.searches` up to the reads, without its report: the
    descent is the same one experiments 18 and 19 run, so the answers read here
    are theirs.
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
    plan = searches.descent_plan(config)
    answers = searches.descend_all(Report(verbose=False), maps, starts, boxes, plan)
    reads = searches.search_reads(problem, answers, budget)

    families = profile.member_families(config)
    readings = []
    for search in searches.SEARCH_ORDER:
        read = reads[search]
        forces = problem.pipeline.analyzer(
            jnp.asarray(read.xyz),
            jnp.asarray(read.diameters),
            problem.loads.analysis,
        )
        reading = shear_reading(
            f"{described.stem.split('_')[0]}, {search}",
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

    warren = loaded_module(EXAMPLES / "warren.py")
    vierendeel = loaded_module(EXAMPLES / "vierendeel.py")

    readings = [read_the_arch()]
    readings.extend(read_a_truss(warren.WARREN_PROFILE, EXAMPLES / "warren.yaml"))
    readings.extend(
        read_a_truss(vierendeel.VIERENDEEL_PROFILE, EXAMPLES / "vierendeel.yaml")
    )
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
    report.write_verdict(verify_checks(checks))


if __name__ == "__main__":
    jnp.set_printoptions(precision=12)
    main()
