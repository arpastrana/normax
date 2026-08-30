# SPDX-License-Identifier: Apache-2.0
"""
What a thinner wall buys, priced by the check the package actually ships.

The wall proportion `d/t` is the one section parameter this project holds fixed,
and it holds it at the Class 3 limit of Eurocode 3 Table 5.2. This sweep prices
that choice: it builds the section catalog at the Class 2 limit and at the Class
3 limit, hands each to the crossed Blueprints check, and sizes the same demand
mix on both. A thinner wall reaches a given elastic modulus with less area, so it
should be lighter wherever bending is live and exactly a tie in pure compression.
Whether it is, and by how much, is a number rather than an argument.

Nothing here reads a plastic section modulus, because the shipped check does not:
Blueprints' eq. (6.14) is elastic at every class, so the shape factor a Class 2
section would have earned is outside what the crossed sizer implements. The sweep
prices the wall proportion alone, which is the parameter the project chooses.

The sweep also bounds the shear the declined clause 6.2.6 would have seen, which
is what licenses declining it: Eurocode 3 6.2.10 permits ignoring shear below
half the plastic shear resistance. `docs/results.md#scope-and-limitations`
states exactly what that check can license. Here the demand is a bound rather
than a reading, this sweep having no frame to read one off — the analyzed shear
of a real design is what settles it.

Run with `uv run python validation/class_ratio_sweep.py`.
"""

from collections.abc import Sequence
from typing import NamedTuple

import jax.numpy as jnp
from blueprints.codes.eurocode.en_1993_1_1_2005.chapter_6_ultimate_limit_state.formula_6_18 import (  # noqa: E501
    Form6Dot18DesignPlasticShearResistance,
)
from blueprints.codes.eurocode.en_1993_1_1_2005.chapter_6_ultimate_limit_state.formula_6_18_sub_av import (  # noqa: E501
    Form6Dot18SubGCircularHollowSection,
)
from jaxtyping import Array
from jaxtyping import Float

from normax.analysis import MemberForces
from normax.materials import Steel355
from normax.reporting import Report
from normax.reporting import ReportColumn
from normax.sections import CLASS_LIMITS
from normax.sections import TubeCatalog
from normax.sections import build_section_catalog
from normax.sizing.blueprint import GAMMA_M0
from normax.structures import build_arch_2d
from normax.tesseract import TesseractSizer

STEEL = Steel355()
LENGTH = 6000.0

# A demand mix swept from pure compression to pure bending, holding the axial
# force and growing the moment.
MOMENTS = (0.0, 1e7, 2e7, 4e7, 8e7, 1.6e8)
FORCE = -6e5

CLASSES = (2, 3)

# Samples the crossover is looked for over, and the range they span.
CROSSOVER_SAMPLES = 321
CROSSOVER_MOMENT_MAX = 1.6e8

# A gap this small a fraction of the heavier mass is a tie, not a crossing.
CROSSOVER_TOLERANCE = 1e-9

# Nodal loading makes the shear exactly the end-moment difference over the
# length, so a moment bounded either way is worst antisymmetric, at twice it.
SHEAR_FACTOR = 2.0

# Eurocode 3 6.2.10(1): shear is ignorable below half the plastic resistance.
SHEAR_THRESHOLD = 0.5


class ClassSweep(NamedTuple):
    """
    One class branch's answer over a whole array of demand mixes.

    Attributes
    ----------
    section_class :
        The Table 5.2 class whose limit fixed the wall proportion.
    catalog :
        The catalog sitting on that class's limit.
    diameters :
        Fully-stressed diameter at every demand mix.
    masses :
        Mass of the member at every demand mix, in kilograms.
    """

    section_class: int
    catalog: TubeCatalog
    diameters: Float[Array, "members"]
    masses: Float[Array, "members"]


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
        Section class that is lighter below that moment, or None where the two
        branches weigh the same at every sample.
    """

    moment: float | None
    lighter: int | None


class ShearCheck(NamedTuple):
    """
    The shear the declined clause 6.2.6 would have seen at one demand mix.

    Attributes
    ----------
    moment :
        Major-axis moment the member is sized against.
    diameter :
        Fully-stressed diameter at that moment.
    resistance :
        Plastic shear resistance of the sized section.
    demand :
        Largest design shear the sized moment admits, from equilibrium.

    Notes
    -----
    The demand is a bound rather than a measurement, this sweep having no frame
    to measure on. It is exact as a bound: an analysis reports the shear every
    member carries, and auditing a real design means reading that.
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
        Whether declining 6.2.6 stops being honest at this mix.
        """
        return "" if self.ratio < SHEAR_THRESHOLD else "EXCEEDS HALF"


class ReadingPair(NamedTuple):
    """
    Biaxial bending read as a resultant, and read as the linear sum it ships as.

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


def build_sizer(catalog: TubeCatalog) -> TesseractSizer:
    """
    The crossed Blueprints check, over one catalog.

    Parameters
    ----------
    catalog :
        The catalog whose ratio sits on a class limit.

    Returns
    -------
    sizer :
        The check, reached across the sizing Tesseract's boundary.

    Notes
    -----
    A sizer is built from a structure it reads for nothing, a cross-section
    check having nothing to settle from a connectivity, so any structure does.
    """
    structure = build_arch_2d()

    return TesseractSizer(structure, catalog, "blueprint")


def build_member_forces(
    moment_major: Float[Array, "members"],
    moment_minor: Float[Array, "members"],
) -> MemberForces:
    """
    One load case of demands: the held axial force beside the swept moments.

    Parameters
    ----------
    moment_major :
        Major-axis moment at the worse end of every member.
    moment_minor :
        Minor-axis moment at the worse end of every member.

    Returns
    -------
    forces :
        What every member carries, with a load case axis of one.

    Notes
    -----
    The far end carries nothing, so the check's reduction to the worse end of
    each axis returns the moment as it was written here.
    """
    axial = jnp.full_like(moment_major, FORCE)
    quiet = jnp.zeros_like(moment_major)
    ends_major = jnp.stack([moment_major, quiet], axis=-1)
    ends_minor = jnp.stack([moment_minor, quiet], axis=-1)

    return MemberForces(axial[None], ends_major[None], ends_minor[None])


def size_demands(
    sizer: TesseractSizer,
    moment_major: Float[Array, "members"],
    moment_minor: Float[Array, "members"],
) -> Float[Array, "members"]:
    """
    The fully-stressed diameter of every demand mix, in one crossing.

    Parameters
    ----------
    sizer :
        The crossed check the sizes are solved against.
    moment_major :
        Major-axis moment at the worse end of every member.
    moment_minor :
        Minor-axis moment at the worse end of every member.

    Returns
    -------
    diameters :
        Outer diameter each demand mix is worked to exactly one at.

    Notes
    -----
    Every demand mix rides the members axis of a single load case, so the whole
    sweep is one crossing rather than one per mix.
    """
    forces = build_member_forces(moment_major, moment_minor)
    buckling_length = jnp.full(moment_major.shape, LENGTH)
    sizes = sizer(forces, buckling_length)

    return sizes.sections.diameter[0]


def weigh_members(
    catalog: TubeCatalog,
    diameters: Float[Array, "members"],
) -> Float[Array, "members"]:
    """
    Mass in kilograms of one member of the swept length at every diameter.

    Parameters
    ----------
    catalog :
        The catalog the diameters are walled by.
    diameters :
        Outer diameter of every member.

    Returns
    -------
    masses :
        Mass of every member, in kilograms.

    Notes
    -----
    The per-member half of `normax.design.compute_member_mass`, which sums; a
    sweep compares mix by mix and so must not sum. Tonnes to kilograms because
    the package works in newtons and millimeters throughout.
    """
    sections = catalog(diameters)
    tonnes = sections.material.density * sections.area * LENGTH

    return tonnes * 1e3


def sweep_class(
    section_class: int,
    moments: Float[Array, "members"],
) -> ClassSweep:
    """
    Every demand mix sized on one class branch.

    Parameters
    ----------
    section_class :
        The Table 5.2 class whose limit fixes the wall proportion.
    moments :
        Major-axis moment of every demand mix.

    Returns
    -------
    sweep :
        The branch's diameters and masses over the whole mix.
    """
    catalog = build_section_catalog(STEEL, section_class)
    sizer = build_sizer(catalog)
    quiet = jnp.zeros_like(moments)
    diameters = size_demands(sizer, moments, quiet)
    masses = weigh_members(catalog, diameters)

    return ClassSweep(section_class, catalog, diameters, masses)


def compare_sweeps(
    sweeps: Sequence[ClassSweep],
    moments: Float[Array, "members"],
) -> list[MassComparison]:
    """
    Pair the branches up, demand mix by demand mix.

    Parameters
    ----------
    sweeps :
        One sweep per class branch, over the same moments.
    moments :
        Major-axis moment of every demand mix.

    Returns
    -------
    compared :
        One comparison per demand mix.
    """
    compared = []
    for index in range(int(moments.shape[0])):
        diameters = tuple(float(sweep.diameters[index]) for sweep in sweeps)
        masses = tuple(float(sweep.masses[index]) for sweep in sweeps)
        compared.append(MassComparison(float(moments[index]), diameters, masses))

    return compared


def find_crossover(classes: Sequence[int]) -> CrossoverResult:
    """
    The moment at which the two branches weigh the same, if there is one.

    Parameters
    ----------
    classes :
        The two Table 5.2 classes whose limits are compared.

    Returns
    -------
    crossing :
        The moment the masses cross at and which class is lighter below it.

    Notes
    -----
    Pure compression is a tie by construction — a squashing resistance is
    proportional to area, so both branches ask for the same area and differ
    only in the diameter they reach it at — so samples closer than a tolerance
    are read as ties rather than as sign changes.
    """
    moments = jnp.linspace(0.0, CROSSOVER_MOMENT_MAX, CROSSOVER_SAMPLES)
    sweeps = [sweep_class(section_class, moments) for section_class in classes]
    gaps = sweeps[0].masses - sweeps[1].masses
    heavier = jnp.maximum(sweeps[0].masses, sweeps[1].masses)
    parted = jnp.abs(gaps) > CROSSOVER_TOLERANCE * heavier
    if not bool(jnp.any(parted)):
        return CrossoverResult(None, None)

    signs = jnp.sign(gaps[parted])
    separated = moments[parted]
    changes = jnp.where(jnp.diff(signs) != 0)[0]
    below = sweeps[0] if float(signs[0]) < 0.0 else sweeps[1]
    if changes.size == 0:
        return CrossoverResult(None, below.section_class)

    crossing = float(separated[int(changes[0])])

    return CrossoverResult(crossing, below.section_class)


def check_shear(catalog: TubeCatalog, diameter: float, moment: float) -> ShearCheck:
    """
    What the declined clause 6.2.6 would have seen at one demand mix.

    Parameters
    ----------
    catalog :
        The catalog the diameter is walled by.
    diameter :
        Outer diameter the member was sized to.
    moment :
        Major-axis moment the member was sized against.

    Returns
    -------
    checked :
        The plastic shear resistance beside the largest shear the mix admits.

    Notes
    -----
    Blueprints' eq. (6.18subg) supplies the shear area of a circular hollow
    section and eq. (6.18) the plastic shear resistance, so the diagnostic
    reads the same library the shipped check does.
    """
    section = catalog(jnp.asarray(diameter))
    area = float(section.area)
    shear_area = Form6Dot18SubGCircularHollowSection(a=area)
    resistance = Form6Dot18DesignPlasticShearResistance(
        a_v=float(shear_area), f_y=float(catalog.material.f_y), gamma_m0=GAMMA_M0
    )
    demand = SHEAR_FACTOR * moment / LENGTH

    return ShearCheck(moment, diameter, float(resistance), demand)


def compare_readings(
    sizer: TesseractSizer,
    moments: Float[Array, "members"],
) -> list[ReadingPair]:
    """
    Biaxial bending read as a resultant against the linear sum that ships.

    Parameters
    ----------
    sizer :
        The crossed check both readings are solved against.
    moments :
        Moment applied about both axes at once, per demand mix.

    Returns
    -------
    readings :
        The diameter each reading asks for, per demand mix.

    Notes
    -----
    Both readings go through the shipped check, the resultant one as the
    equivalent uniaxial moment `sqrt(M_y^2 + M_z^2)`. The check itself sums the
    two axes linearly, which is the conservative reading of Eurocode 3 eq.
    (6.2); a resultant stress is what an elastic stress analysis would report.
    """
    quiet = jnp.zeros_like(moments)
    resultants = jnp.sqrt(2.0) * moments
    sized_resultant = size_demands(sizer, resultants, quiet)
    sized_linear = size_demands(sizer, moments, moments)
    readings = []
    for index in range(int(moments.shape[0])):
        moment = float(moments[index])
        pair = ReadingPair(
            moment, float(sized_resultant[index]), float(sized_linear[index])
        )
        readings.append(pair)

    return readings


def report_catalogs(report: Report, sweeps: Sequence[ClassSweep]) -> None:
    """
    The wall proportion each class limit stands for.
    """
    entries = []
    for sweep in sweeps:
        ratio = float(sweep.catalog.ratio)
        limit = f"Table 5.2 {CLASS_LIMITS[sweep.section_class]:.0f} eps^2"
        label = f"Class {sweep.section_class}"
        proportion = f"d/t = {ratio:.3f} ({limit}, t/d = {1.0 / ratio:.4f})"
        entries.append((label, proportion))

    title = "Class 2 limit against Class 3 limit, S355, 6 m member, 600 kN compression"

    report.write_line(title)
    report.write_entries(entries)
    report.write_note(
        """
        Both branches are checked by the crossed Blueprints sizer, which reads
        the elastic modulus of eq. (6.14) at either class, so what separates
        them here is the wall proportion and nothing else.
        """
    )


def report_masses(
    report: Report,
    sweeps: Sequence[ClassSweep],
    moments: Float[Array, "members"],
) -> None:
    """
    Which class limit is lighter, over the whole demand mix.
    """
    compared = compare_sweeps(sweeps, moments)

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
        lighter = f"Class {sweeps[found.lighter].section_class}"
        sizes = (*found.diameters, *found.masses)
        rows.append((found.moment / 1e6, *sizes, lighter, found.saving))

    report.write_heading("What each demand mix costs on either branch")
    report.write_table(columns, rows)

    crossing = find_crossover(CLASSES)
    report.write_heading("Where the crossover sits")
    if crossing.lighter is None:
        report.write_note(
            """
            The two branches weigh the same at every sample in the swept range.
            """
        )
    elif crossing.moment is None:
        report.write_note(
            f"""
            No crossover in the swept range; Class {crossing.lighter} is lighter
            throughout, the pure-compression tie aside.
            """
        )
    else:
        report.write_note(
            f"""
            The two are equal near M_y = {crossing.moment / 1e6:.1f} kNm.
            Below it one class wins, above it the other.
            """
        )


def report_shear(report: Report, sweep: ClassSweep) -> None:
    """
    The shear the declined clause would have seen, on one branch.

    A bound over the demand mix, not a reading off a structure. What the ratio
    does on a converged design is what licenses declining 6.2.6, and the
    analyzed shear a backend reports is what answers it.
    """
    checked = []
    for index in range(1, int(sweep.diameters.shape[0])):
        diameter = float(sweep.diameters[index])
        checked.append(check_shear(sweep.catalog, diameter, MOMENTS[index]))

    columns = (
        ReportColumn("M_y [kNm]", ".0f"),
        ReportColumn("d [mm]", ".2f"),
        ReportColumn("V_pl,Rd [kN]", ".1f"),
        ReportColumn("V_Ed bound [kN]", ".1f"),
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

    worst = max(found.ratio for found in checked)
    heading = f"Shear the declined clause would have seen, Class {sweep.section_class}"

    report.write_heading(heading)
    report.write_table(columns, rows)
    report.write_note(
        f"""
        Declining 6.2.6 stays honest while every ratio is under
        {SHEAR_THRESHOLD}, and the worst here is {worst:.3f}. Every demand is
        the largest a member carrying this moment over {LENGTH:.0f} mm can see,
        the shear under nodal loading being the end-moment difference over the
        length and the moment bounded both ways. A shorter member at the same
        moment sees more, in inverse proportion. Read the analyzed shear off a
        converged design rather than quoting this.
        """
    )


def report_readings(report: Report, sweep: ClassSweep) -> None:
    """
    The two readings of biaxial bending, and what choosing between them costs.
    """
    report.write_heading("A resultant reading against the linear sum, Class 3 branch")
    report.write_note(
        """
        The shipped check sums the two axes' moments linearly at the worse end,
        which is the conservative reading of eq. (6.2); 6.2.9.2 read as an
        elastic stress analysis would combine them as a resultant instead.
        Blueprints implements eq. (6.42) with the stress as an input and so does
        not settle it, and the crossed check does not offer the choice, so the
        resultant reading is priced here as an equivalent uniaxial moment.
        """
    )
    columns = (
        ReportColumn("M_y = M_z [kNm]", ".0f"),
        ReportColumn("resultant [mm]", ".2f"),
        ReportColumn("linear sum [mm]", ".2f"),
        ReportColumn("diameter", ".2%"),
        ReportColumn("area", ".2%"),
    )
    sizer = build_sizer(sweep.catalog)
    moments = jnp.asarray(MOMENTS[1:])
    readings = compare_readings(sizer, moments)
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
        The gap is widest where bending governs and vanishes under uniaxial
        bending, where the two readings are the same equation.
        """
    )


def main(verbose: bool = True) -> None:
    """
    Sweep the demand mix and report what the thinner wall buys.
    """
    report = Report(verbose)
    moments = jnp.asarray(MOMENTS)
    sweeps = [sweep_class(section_class, moments) for section_class in CLASSES]

    report_catalogs(report, sweeps)
    report_masses(report, sweeps, moments)
    report_shear(report, sweeps[1])
    report_readings(report, sweeps[1])


if __name__ == "__main__":
    main()
