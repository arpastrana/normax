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
Aggregating several load cases into one differentiable size per member.

A member must satisfy every load case, so its size is the largest any case
demands. That largest is not differentiable, and a gradient taken through it
sees one case per step and stalls. The smooth envelope of normax.ec3.sizing
replaces it, in the logarithm of the diameter so the sharpness is dimensionless.

The envelope never understates the true largest, so annealing the sharpness
upward drives it onto that largest from above and the design stays adequate
throughout. This reports how much is given away at each sharpness, which is the
number the annealing schedule of P4 has to be chosen against.

Also exercises the case the discontinuity at zero axial force makes awkward: a
member whose axial force changes sign between load cases.

Run with `uv run python experiments/06_load_case_aggregation.py`.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float

from normax.ec3.actions import MemberActions
from normax.ec3.material import SteelGrade
from normax.ec3.section import TubeCatalogue
from normax.ec3.sizing import diameter_envelope
from normax.ec3.sizing import diameter_required
from normax.ec3.sizing import mass_of_tubes
from normax.reporting import Report
from normax.reporting import ReportColumn

STEEL = SteelGrade()
CATALOGUE = TubeCatalogue.at_class_limit(STEEL.f_y, 3)
SECTION_CLASS = 3

# Moment factors of a member bent in single curvature, as Table B.3 reads them.
MOMENT_FACTOR = 0.9

LENGTHS = jnp.asarray([4000.0, 6000.0, 5000.0, 4500.0])

# Three load cases over four members: symmetric, half-span asymmetric, and a
# crown point load. The third member reverses from compression to tension.
FORCES_BY_CASE = [
    [-6e5, -4e5, -3e5, -5e5],
    [-8e5, -2e5, 2e5, -6e5],
    [-3e5, -7e5, -1e5, -9e5],
]
MOMENTS_BY_CASE = [
    [2e7, 1e7, 5e6, 1.5e7],
    [5e7, 3e7, 2e7, 4e7],
    [1e7, 6e7, 1e7, 2e7],
]

FORCES = jnp.asarray(FORCES_BY_CASE)
MOMENTS = jnp.asarray(MOMENTS_BY_CASE)

SHARPNESSES = (5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0)

# The sharpness the live-case comparison is made at.
SHARPNESS = 50.0

# The member whose axial force reverses between load cases.
REVERSING = 2

NUM_CASES, NUM_MEMBERS = FORCES.shape


class AnnealStep(NamedTuple):
    """
    What the smooth envelope gives away at one sharpness.

    Attributes
    ----------
    beta :
        Sharpness of the envelope.
    mass :
        Mass of the smoothly sized structure, in kilograms.
    excess :
        Fraction by which that exceeds the mass of the exact largest.
    bound :
        Fraction the envelope's own bound allows it to exceed by.
    finite :
        Whether the gradient of the mass is finite throughout.
    """

    beta: float
    mass: float
    excess: float
    bound: float
    finite: bool


class LiveCases(NamedTuple):
    """
    How many load cases reach one member's size with a gradient.

    Attributes
    ----------
    member :
        Index of the member.
    smooth :
        Cases with a non-zero gradient under the smooth envelope.
    hard :
        Cases with a non-zero gradient under a hard maximum.
    """

    member: int
    smooth: int
    hard: int


def diameters_per_case(forces: Float[Array, "cases members"]) -> Float[Array, "..."]:
    """
    Fully-stressed diameter of every member under every load case.
    """
    actions = MemberActions(forces, MOMENTS, 0.0, MOMENT_FACTOR, MOMENT_FACTOR)
    diameters = diameter_required(
        actions,
        LENGTHS,
        STEEL,
        CATALOGUE,
        section_class=SECTION_CLASS,
    )

    return diameters


def mass_smooth(forces: Float[Array, "cases members"], beta: float) -> Float[Array, ""]:
    """
    Mass of the structure sized by the smooth envelope at a given sharpness.
    """
    per_case = diameters_per_case(forces)
    sizes = diameter_envelope(per_case, beta)
    tubes = CATALOGUE.tube_at(sizes)

    return mass_of_tubes(tubes, LENGTHS, STEEL)


def mass_hard(forces: Float[Array, "cases members"]) -> Float[Array, ""]:
    """
    Mass of the same structure sized by the true largest of the load cases.
    """
    per_case = diameters_per_case(forces)
    sizes = jnp.max(per_case, axis=0)
    tubes = CATALOGUE.tube_at(sizes)

    return mass_of_tubes(tubes, LENGTHS, STEEL)


def anneal_step(exact_mass: float, beta: float) -> AnnealStep:
    """
    The smoothed mass at one sharpness, against the exact one and its bound.
    """
    smoothed = float(mass_smooth(FORCES, beta)) * 1e3
    gradient = jax.grad(mass_smooth)(FORCES, beta)
    excess = (smoothed - exact_mass) / exact_mass
    bound = float(jnp.log(NUM_CASES) / beta)
    finite = bool(jnp.all(jnp.isfinite(gradient)))
    step = AnnealStep(beta, smoothed, excess, bound, finite)

    return step


def live_cases(member: int, smooth: Float[Array, "..."], hard: Float[Array, "..."]):
    """
    Cases that reach one member's size with a gradient, under either aggregation.
    """
    live_smooth = int(jnp.sum(jnp.abs(smooth[:, member]) > 0.0))
    live_hard = int(jnp.sum(jnp.abs(hard[:, member]) > 0.0))
    counted = LiveCases(member, live_smooth, live_hard)

    return counted


def report_sizes(report: Report, per_case: Float[Array, "..."]) -> float:
    """
    What each load case asks of each member, and the exact largest of them.
    """
    exact = jnp.max(per_case, axis=0)
    tubes = CATALOGUE.tube_at(exact)
    exact_mass = float(mass_of_tubes(tubes, LENGTHS, STEEL)) * 1e3

    per_case_columns = [
        ReportColumn(f"case {case + 1} [mm]", ".2f") for case in range(NUM_CASES)
    ]
    columns = (
        ReportColumn("member"),
        *per_case_columns,
        ReportColumn("exact max [mm]", ".2f"),
    )
    rows = []
    for member in range(NUM_MEMBERS):
        sizes = [float(per_case[case, member]) for case in range(NUM_CASES)]
        rows.append((member, *sizes, float(exact[member])))

    entries = (("exact mass", f"{exact_mass:.2f} kg"),)

    report.write_line("Three load cases over four members, S355 at the Class 3 limit")
    report.write_table(columns, rows)
    report.write_entries(entries)

    return exact_mass


def report_annealing(report: Report, exact_mass: float) -> None:
    """
    What the smoothing costs at each sharpness, and what bounds it.
    """
    columns = (
        ReportColumn("beta", ".0f"),
        ReportColumn("mass [kg]", ".2f"),
        ReportColumn("excess", ".3%"),
        ReportColumn("bound log(cases)/beta", ".3%"),
        ReportColumn("gradient finite"),
    )
    annealed = [anneal_step(exact_mass, beta) for beta in SHARPNESSES]
    rows = [
        (step.beta, step.mass, step.excess, step.bound, str(step.finite))
        for step in annealed
    ]

    report.write_heading("Annealing the sharpness")
    report.write_table(columns, rows)
    report.write_note(
        """
        The excess is an overestimate of the size, never an underestimate, so
        every intermediate design satisfies the standard.
        """
    )


def report_live_cases(report: Report, per_case: Float[Array, "..."]) -> None:
    """
    That every case sees a gradient, which a hard maximum would not give.
    """
    smooth = jax.grad(mass_smooth)(FORCES, SHARPNESS)
    hard = jax.grad(mass_hard)(FORCES)

    columns = (
        ReportColumn("member"),
        ReportColumn("smooth, cases with a gradient"),
        ReportColumn("hard maximum"),
    )
    counted = [live_cases(member, smooth, hard) for member in range(NUM_MEMBERS)]
    rows = [(found.member, found.smooth, found.hard) for found in counted]

    report.write_heading("Every case sees a gradient, which a hard maximum would not")
    report.write_table(columns, rows)

    entries = (
        (f"member {REVERSING} forces per case", f"{FORCES[:, REVERSING]}"),
        ("sizes per case", f"{per_case[:, REVERSING]}"),
        ("gradient per case", f"{smooth[:, REVERSING]}"),
    )

    report.write_heading("The member that changes sign between cases")
    report.write_entries(entries)
    report.write_note(
        """
        Finite throughout, despite the standard being discontinuous at zero.
        """
    )


def main(verbose: bool = True) -> None:
    """
    Anneal the sharpness and report what the smoothing costs.
    """
    report = Report(verbose)
    per_case = diameters_per_case(FORCES)

    exact_mass = report_sizes(report, per_case)
    report_annealing(report, exact_mass)
    report_live_cases(report, per_case)


if __name__ == "__main__":
    main()
