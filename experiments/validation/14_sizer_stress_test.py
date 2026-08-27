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
Two implementations of one standard, stress-tested against each other.

The blueprints sizer and the EC3 sizer disagree by up to a quarter of a
diameter on a compressed member, and the whole disagreement is one clause:
Blueprints implements no §6.3.1 flexural buckling. This experiment makes that
attribution quantitative by driving the EC3 sizer toward the cross-section
regime — the buckling length toward zero, so chi rides its cap at one, and
the moment combination set linear to match — and measuring what remains.

What remains converges to zero at first order in the slenderness: the class-3
interaction factor is `k_yy = C_m (1 + 0.6 lambda n)`, so silencing the
length leaves a residual `0.6 lambda n m` that the sweep below halves with
every halving of the length. In tension, where the member check never runs,
the two implementations agree to machine precision at any length.

The gradients are cross-checked the same way: two hand-derived implicit
rules, one over a host bisection of Blueprints residuals, one a `custom_jvp`
inside ec3x, agreeing about the same root.

Blueprints is LGPL-2.1, experiment-only, waived 2026-08-15.

Run with `uv run --group pipeline python
experiments/14_sizer_stress_test.py`.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from jaxtyping import Float

from normax.analysis import MemberForces
from normax.materials import Steel355
from normax.reporting import Report
from normax.reporting import ReportColumn
from normax.reporting import ToleranceCheck
from normax.reporting import verify_checks
from normax.sections import TubeFamily
from normax.sizing import BlueprintSizer
from normax.sizing import Ec3Sizer
from normax.structures import build_arch_2d

TITLE = "Two implementations of one standard, stress-tested against each other."

RATIO = 50.0
MEMBER_LENGTH = 4000.0

# Small enough that the interaction residual sits below the parity bound.
SILENCED_LENGTH = 1e-3

SWEEP_LENGTHS = (MEMBER_LENGTH, 100.0, 1.0, 1e-3, 1e-6)

# Measured 5.8e-9 at the silenced length; the residual is 0.6 lambda n m.
TOLERANCE_SILENCED = 1e-7
TOLERANCE_TENSION = 1e-14
TOLERANCE_GRADIENT = 1e-6

CASE_COLUMNS = (
    ReportColumn("case", align="<"),
    ReportColumn("blueprints [mm]", ".9f"),
    ReportColumn("EC3, silenced [mm]", ".9f"),
    ReportColumn("gap", ".2e"),
    ReportColumn("EC3, 4 m [mm]", ".3f"),
    ReportColumn("buckling", ".3f"),
)

SWEEP_COLUMNS = (
    ReportColumn("buckling length [mm]", ".0e"),
    ReportColumn("worst gap", ".3e"),
    ReportColumn("gap over length", ".3e"),
)

GRADIENT_COLUMNS = (
    ReportColumn("case", align="<"),
    ReportColumn("blueprints", "+.9e"),
    ReportColumn("EC3, silenced", "+.9e"),
    ReportColumn("gap", ".2e"),
)


class StressCase(NamedTuple):
    """
    One member of the stress grid: an axial force and equal end moments.

    Attributes
    ----------
    axial_force :
        Design axial force, negative in compression.
    moment :
        Moment at both ends of both members, so Table B.3 reads it unchanged.
    """

    axial_force: float
    moment: float

    @property
    def label(self) -> str:
        """
        The case as it appears in the leftmost column of a table.
        """
        force = self.axial_force / 1e3
        bent = self.moment / 1e6

        return f"{force:.0f} kN, {bent:.2f} kNm"


GRID = (
    StressCase(-5.0e5, 2.0e6),
    StressCase(-3.0e5, 0.0),
    StressCase(-1.0e4, 5.0e7),
    StressCase(-2.0e6, 1.0e7),
    StressCase(-1.0e3, 1.0e5),
    StressCase(2.0e5, 1.0e6),
    StressCase(1.0e6, 3.0e6),
    StressCase(5.0e3, 0.0),
)


class SizerPair(NamedTuple):
    """
    The two implementations under test, configured to be comparable.

    Attributes
    ----------
    blueprint :
        The blueprints-backed cross-section sizer.
    ec3 :
        The EC3 sizer with the linear moment combination, so the only
        remaining difference is what §6.3 adds.
    """

    blueprint: BlueprintSizer
    ec3: Ec3Sizer


def sizer_pair() -> SizerPair:
    """
    Both sizers over one class-3 family, moments combined the same way.
    """
    structure = build_arch_2d(num_edges=10, span=10_000.0, rise=3_000.0)
    family = TubeFamily(RATIO, Steel355())
    blueprint = BlueprintSizer(structure, family)
    ec3 = Ec3Sizer(structure, family, resultant=False)

    return SizerPair(blueprint, ec3)


def grid_forces() -> MemberForces:
    """
    The stress grid as one load case of member forces.
    """
    axial = jnp.asarray([[case.axial_force for case in GRID]])
    moment = jnp.asarray([case.moment for case in GRID])
    ends = jnp.stack([moment, moment], axis=-1)[None, :, :]
    minor = jnp.zeros_like(ends)

    return MemberForces(axial, ends, minor)


def sized_diameters(
    sizer: BlueprintSizer | Ec3Sizer,
    forces: MemberForces,
    buckling_length: float,
) -> Float[Array, "members"]:
    """
    One sizer's answers over the grid, at one buckling length.
    """
    lengths = jnp.full(len(GRID), buckling_length)
    sizes = sizer(forces, lengths)

    return sizes.sections.diameter[0]


def report_cases(report: Report, pair: SizerPair, forces: MemberForces) -> float:
    """
    The grid, sized three ways, and the worst silenced-length gap.
    """
    naive = np.asarray(sized_diameters(pair.blueprint, forces, MEMBER_LENGTH))
    silenced = np.asarray(sized_diameters(pair.ec3, forces, SILENCED_LENGTH))
    designed = np.asarray(sized_diameters(pair.ec3, forces, MEMBER_LENGTH))

    gaps = np.abs(silenced - naive) / naive
    ratios = designed / naive
    rows = [
        (case.label, naive[i], silenced[i], gaps[i], designed[i], ratios[i])
        for i, case in enumerate(GRID)
    ]

    report.write_heading("The grid, sized by both implementations")
    report.write_table(CASE_COLUMNS, rows)
    report.write_note(
        "The buckling column prices §6.3 at a 4 m member; the silenced "
        "column is the same clause library with that length driven to zero."
    )

    return float(np.max(gaps))


def report_sweep(report: Report, pair: SizerPair, forces: MemberForces) -> None:
    """
    The convergence of the two implementations as the length is silenced.
    """
    naive = np.asarray(sized_diameters(pair.blueprint, forces, MEMBER_LENGTH))

    rows = []
    for length in SWEEP_LENGTHS:
        checked = np.asarray(sized_diameters(pair.ec3, forces, length))
        worst = float(np.max(np.abs(checked - naive) / naive))
        rows.append((length, worst, worst / length))

    report.write_heading("What remains is first order in the slenderness")
    report.write_table(SWEEP_COLUMNS, rows)
    report.write_note(
        "The residual is the class-3 interaction factor's 0.6 lambda n m, so "
        "the gap over the length is constant once buckling stops governing."
    )


def report_gradients(report: Report, pair: SizerPair, forces: MemberForces) -> float:
    """
    The two implicit rules' force sensitivities at the silenced length.
    """

    def naive_total(axial):
        probed = MemberForces(axial, forces.moment_major, forces.moment_minor)

        return jnp.sum(sized_diameters(pair.blueprint, probed, MEMBER_LENGTH))

    def checked_total(axial):
        probed = MemberForces(axial, forces.moment_major, forces.moment_minor)

        return jnp.sum(sized_diameters(pair.ec3, probed, SILENCED_LENGTH))

    ours = np.asarray(jax.grad(naive_total)(forces.axial_force))[0]
    theirs = np.asarray(jax.grad(checked_total)(forces.axial_force))[0]
    gaps = np.abs(theirs - ours) / np.maximum(np.abs(ours), 1e-300)
    rows = [(case.label, ours[i], theirs[i], gaps[i]) for i, case in enumerate(GRID)]

    report.write_heading("Two hand-derived implicit rules, one root")
    report.write_table(GRADIENT_COLUMNS, rows)

    return float(np.max(gaps))


def main(verbose: bool = True) -> None:
    """
    Stress the two implementations against each other over the grid.
    """
    report = Report(verbose)
    report.write_line(TITLE)

    entries = (
        ("family", f"S355, d/t = {RATIO:.1f} (class 3)"),
        ("parity settings", "resultant=False, buckling length driven to zero"),
        ("silenced length", f"{SILENCED_LENGTH:.0e} mm"),
    )
    report.write_heading("The comparison, and what makes it fair")
    report.write_entries(entries)

    pair = sizer_pair()
    forces = grid_forces()

    worst_silenced = report_cases(report, pair, forces)
    report_sweep(report, pair, forces)
    worst_gradient = report_gradients(report, pair, forces)

    pulled = MemberForces(
        jnp.abs(forces.axial_force), forces.moment_major, forces.moment_minor
    )
    naive = np.asarray(sized_diameters(pair.blueprint, pulled, MEMBER_LENGTH))
    checked = np.asarray(sized_diameters(pair.ec3, pulled, MEMBER_LENGTH))
    worst_tension = float(np.max(np.abs(checked - naive) / naive))

    tension_check = ToleranceCheck(
        "tension parity at any length", worst_tension, TOLERANCE_TENSION
    )
    checks = (
        ToleranceCheck("silenced-length parity", worst_silenced, TOLERANCE_SILENCED),
        tension_check,
        ToleranceCheck("gradient agreement", worst_gradient, TOLERANCE_GRADIENT),
    )
    report.write_heading("Summary")
    report.write_checks(checks)
    report.write_verdict(verify_checks(checks))


if __name__ == "__main__":
    jnp.set_printoptions(precision=12)
    main()
