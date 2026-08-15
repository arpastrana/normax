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
Which wall proportion is lighter: the Class 2 limit or the Class 3 limit.

ec3x's docs/clauses.md records the trade and declines to call it. A thinner wall buys
more area for the same weight of steel, which is what the Class 3 limit offers;
but Class 3 forfeits the shape factor of 1.326 in bending and reads the weaker
column of Table B.1. Which wins depends on how much of the demand is bending,
and that is a number rather than an argument.

The sweep also reports the shear the excluded clause would have seen, which is
what open item 0d of ec3x's docs/clauses.md asks for: the exclusion of 6.2.6 is only
honest while the design shear stays under half the plastic shear resistance.

Run with `uv run python experiments/05_class_ratio_sweep.py`.
"""

from collections.abc import Sequence
from typing import NamedTuple

import jax.numpy as jnp
from ec3x.actions import MemberActions
from ec3x.classification import is_plastic
from ec3x.material import Steel
from ec3x.resistance import SHEAR_THRESHOLD
from ec3x.resistance import area_shear
from ec3x.resistance import resistance_shear
from ec3x.section import TubeCatalogue
from ec3x.sizing import diameter_required
from ec3x.sizing import mass_of_tubes
from jaxtyping import Array
from jaxtyping import Float

from normax.reporting import Report
from normax.reporting import ReportColumn

STEEL = Steel()
LENGTH = 6000.0

# A demand mix swept from pure compression to pure bending, holding the axial
# force and growing the moment.
MOMENTS = (0.0, 1e7, 2e7, 4e7, 8e7, 1.6e8)
FORCE = -6e5

# Moment factors of a member bent in single curvature, as Table B.3 reads them.
MOMENT_FACTOR = 0.9

CLASSES = (2, 3)

# Samples the crossover is looked for over, and the range they span.
CROSSOVER_SAMPLES = 321
CROSSOVER_MOMENT_MAX = 1.6e8

# A simply supported span carrying an end moment has a shear of about four
# moments over its length, which is the worst plausible pairing.
SHEAR_FACTOR = 4.0


def behavior_of(catalogue: TubeCatalogue) -> str:
    """
    Whether a family's class takes plastic or elastic section properties.

    Parameters
    ----------
    catalogue :
        Tube family whose ratio holds the section at a class limit.

    Returns
    -------
    behavior :
        "plastic" for Classes 1 and 2, "elastic" for Class 3.

    Notes
    -----
    A function rather than a container's property, because there is nothing left
    to pair the class with: a family carries the class its ratio sits at, so the
    branch this sweep compares *is* a catalogue.
    """
    return "plastic" if is_plastic(catalogue.section_class) else "elastic"


class MassComparison(NamedTuple):
    """
    What one demand mix costs on each class branch.

    Attributes
    ----------
    moment :
        Major-axis moment the members are sized against.
    diameters :
        Fully-stressed diameter on each branch, in the order the branches came.
    masses :
        Mass of the member on each branch, in kilograms.
    """

    moment: float
    diameters: tuple[float, ...]
    masses: tuple[float, ...]

    @property
    def lighter(self) -> int:
        """
        Index of the branch that is lighter at this demand mix.
        """
        indices = range(len(self.masses))

        return int(min(indices, key=lambda index: self.masses[index]))

    @property
    def saving(self) -> float:
        """
        Fraction of the heavier design the lighter one gives back.
        """
        return abs(self.masses[0] - self.masses[1]) / max(self.masses)


class CrossoverResult(NamedTuple):
    """
    Where the two branches weigh the same, if they do so in the swept range.

    Attributes
    ----------
    moment :
        Moment the two masses cross at, or None where they never do.
    lighter :
        Section class that is lighter below that moment.
    """

    moment: float | None
    lighter: int


class ShearCheck(NamedTuple):
    """
    The shear the excluded clause 6.2.6 would have seen at one demand mix.

    Attributes
    ----------
    moment :
        Major-axis moment the member is sized against.
    diameter :
        Fully-stressed diameter at that moment.
    resistance :
        Plastic shear resistance of the sized section.
    demand :
        Design shear the worst plausible span pairing implies.
    """

    moment: float
    diameter: float
    resistance: float
    demand: float

    @property
    def ratio(self) -> float:
        """
        Design shear as a fraction of the plastic shear resistance.
        """
        return self.demand / self.resistance

    @property
    def flag(self) -> str:
        """
        Whether the exclusion of 6.2.6 stops being honest at this mix.
        """
        return "" if self.ratio < SHEAR_THRESHOLD else "EXCEEDS HALF"


class ReadingPair(NamedTuple):
    """
    Eq. 6.42 read as a resultant stress, and read as a linear sum.

    Attributes
    ----------
    moment :
        Moment applied about both axes at once.
    resultant :
        Diameter the resultant reading asks for.
    linear :
        Diameter the linear-sum reading asks for.
    """

    moment: float
    resultant: float
    linear: float

    @property
    def widening(self) -> float:
        """
        Fraction by which the linear reading widens the tube.
        """
        return self.linear / self.resultant - 1.0

    @property
    def area_growth(self) -> float:
        """
        Fraction by which that widening grows the area, which is the mass.
        """
        return (self.linear / self.resultant) ** 2 - 1.0


def diameter_for(
    catalogue: TubeCatalogue,
    moment_major: float,
    moment_minor: float = 0.0,
) -> Float[Array, ""]:
    """
    Fully-stressed diameter on one class branch.
    """
    moments = (moment_major, moment_minor)
    actions = MemberActions(FORCE, *moments, MOMENT_FACTOR, MOMENT_FACTOR)
    diameter = diameter_required(actions, LENGTH, catalogue)

    return diameter


def mass_for(catalogue: TubeCatalogue, moment_major: float) -> float:
    """
    Mass in kilograms of one member on one class branch.
    """
    diameter = diameter_for(catalogue, moment_major)
    tube = catalogue(diameter)

    return float(mass_of_tubes(tube, LENGTH)) * 1e3


def compare_masses(
    families: Sequence[TubeCatalogue],
    moment: float,
) -> MassComparison:
    """
    Diameter and mass on every branch at one demand mix.
    """
    diameters = tuple(float(diameter_for(catalogue, moment)) for catalogue in families)
    masses = tuple(mass_for(catalogue, moment) for catalogue in families)
    compared = MassComparison(moment, diameters, masses)

    return compared


def crossover_moment(families: Sequence[TubeCatalogue]) -> CrossoverResult:
    """
    The moment at which the two branches weigh the same, if there is one.
    """
    sampled = jnp.linspace(0.0, CROSSOVER_MOMENT_MAX, CROSSOVER_SAMPLES)
    gaps = [
        mass_for(families[0], moment) - mass_for(families[1], moment)
        for moment in sampled
    ]
    difference = jnp.asarray(gaps)
    changes = jnp.where(jnp.diff(jnp.sign(difference)) != 0)[0]
    below = families[0] if float(difference[0]) < 0.0 else families[1]

    if changes.size == 0:
        crossing = CrossoverResult(None, below.section_class)
    else:
        moment = float(sampled[int(changes[0])])
        crossing = CrossoverResult(moment, below.section_class)

    return crossing


def shear_check(catalogue: TubeCatalogue, moment: float) -> ShearCheck:
    """
    What clause 6.2.6 would have seen at one demand mix.
    """
    diameter = diameter_for(catalogue, moment)
    area = area_shear(catalogue(diameter).area)
    steel = Steel(f_y=STEEL.f_y, gamma_m0=STEEL.gamma_m0)
    resistance = resistance_shear(area, steel)
    demand = SHEAR_FACTOR * moment / LENGTH
    checked = ShearCheck(moment, float(diameter), float(resistance), demand)

    return checked


def reading_pair(catalogue: TubeCatalogue, moment: float) -> ReadingPair:
    """
    The two readings of Eq. 6.42, under equal moments about both axes.
    """
    actions = MemberActions(FORCE, moment, moment, MOMENT_FACTOR, MOMENT_FACTOR)
    diameters = []
    for choice in (True, False):
        diameter = diameter_required(actions, LENGTH, catalogue, resultant=choice)
        diameters.append(float(diameter))

    readings = ReadingPair(moment, diameters[0], diameters[1])

    return readings


def report_families(report: Report, families: Sequence[TubeCatalogue]) -> None:
    """
    The wall proportion each class limit stands for.
    """
    entries = []
    for catalogue in families:
        ratio = float(catalogue.ratio)
        label = f"Class {catalogue.section_class}"
        proportion = f"d/t = {ratio:.3f} ({behavior_of(catalogue)})"
        entries.append((label, proportion))

    title = "Class 2 limit against Class 3 limit, S355, 6 m member, 600 kN compression"

    report.write_line(title)
    report.write_entries(entries)


def report_masses(report: Report, families: Sequence[TubeCatalogue]) -> None:
    """
    Which class limit is lighter, over the whole demand mix.
    """
    compared = [compare_masses(families, moment) for moment in MOMENTS]

    columns = (
        ReportColumn("M_y [kNm]", ".0f"),
        ReportColumn("d Class 2 [mm]", ".2f"),
        ReportColumn("d Class 3 [mm]", ".2f"),
        ReportColumn("kg Class 2", ".2f"),
        ReportColumn("kg Class 3", ".2f"),
        ReportColumn("lighter", align="<"),
        ReportColumn("saving", ".2%"),
    )
    rows = []
    for found in compared:
        lighter = f"Class {families[found.lighter].section_class}"
        sizes = (*found.diameters, *found.masses)
        rows.append((found.moment / 1e6, *sizes, lighter, found.saving))

    report.write_heading("What each demand mix costs on either branch")
    report.write_table(columns, rows)

    crossing = crossover_moment(families)
    report.write_heading("Where the crossover sits")
    if crossing.moment is None:
        report.write_note(
            f"""
            No crossover in the swept range; Class {crossing.lighter} is lighter
            throughout.
            """
        )
    else:
        report.write_note(
            f"""
            The two are equal near M_y = {crossing.moment / 1e6:.1f} kNm.
            Below it one class wins, above it the other.
            """
        )


def report_shear(report: Report, catalogue: TubeCatalogue) -> None:
    """
    The shear the excluded clause would have seen, open item 0d.
    """
    checked = [shear_check(catalogue, moment) for moment in MOMENTS[1:]]
    columns = (
        ReportColumn("M_y [kNm]", ".0f"),
        ReportColumn("d [mm]", ".2f"),
        ReportColumn("V_pl,Rd [kN]", ".1f"),
        ReportColumn("V_Ed simple span [kN]", ".1f"),
        ReportColumn("ratio", ".3f"),
        ReportColumn("", align="<"),
    )
    rows = []
    for found in checked:
        resistance = found.resistance / 1e3
        demand = found.demand / 1e3
        row = (
            found.moment / 1e6,
            found.diameter,
            resistance,
            demand,
            found.ratio,
            found.flag,
        )
        rows.append(row)

    heading = "Shear the excluded clause would have seen (clauses.md open item 0d)"

    report.write_heading(heading)
    report.write_table(columns, rows)
    report.write_note(
        f"""
        The exclusion of 6.2.6 stays honest while every ratio is under
        {SHEAR_THRESHOLD}. Recompute this on the converged design before quoting
        it.
        """
    )


def report_readings(report: Report, catalogue: TubeCatalogue) -> None:
    """
    The two readings of Eq. 6.42, and what choosing between them costs.
    """
    report.write_heading("The two readings of Eq. 6.42, on the Class 3 branch")
    report.write_note(
        """
        The guide says 6.2.9.2 permits only a linear interaction of stresses;
        the ECCS says the stress is evaluated by an elastic stress analysis.
        Blueprints implements 6.42 with the stress as an input and so does not
        settle it; Karamba implements no 6.2.9 at all. See ec3x's docs/clauses.md.
        """
    )
    columns = (
        ReportColumn("M_y = M_z [kNm]", ".0f"),
        ReportColumn("resultant [mm]", ".2f"),
        ReportColumn("linear sum [mm]", ".2f"),
        ReportColumn("diameter", ".2%"),
        ReportColumn("area", ".2%"),
    )
    readings = [reading_pair(catalogue, moment) for moment in MOMENTS[1:]]
    rows = [
        (
            found.moment / 1e6,
            found.resultant,
            found.linear,
            found.widening,
            found.area_growth,
        )
        for found in readings
    ]

    report.write_table(columns, rows)
    report.write_note(
        """
        The gap closes wherever Eq. 6.61 governs, since that equation already
        sums the two moments linearly, and vanishes under uniaxial bending.
        """
    )


def main(verbose: bool = True) -> None:
    """
    Sweep the demand mix and report which class limit is lighter.
    """
    report = Report(verbose)
    families = [
        TubeCatalogue.at_class_limit(STEEL, section_class) for section_class in CLASSES
    ]

    report_families(report, families)
    report_masses(report, families)
    report_shear(report, families[1])
    report_readings(report, families[1])


if __name__ == "__main__":
    main()
