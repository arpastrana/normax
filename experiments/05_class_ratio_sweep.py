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

docs/clauses.md records the trade and declines to call it. A thinner wall buys
more area for the same weight of steel, which is what the Class 3 limit offers;
but Class 3 forfeits the shape factor of 1.326 in bending and reads the weaker
column of Table B.1. Which wins depends on how much of the demand is bending,
and that is a number rather than an argument.

The sweep also reports the shear the excluded clause would have seen, which is
what open item 0d of docs/clauses.md asks for: the exclusion of 6.2.6 is only
honest while the design shear stays under half the plastic shear resistance.

Run with `uv run python experiments/05_class_ratio_sweep.py`.
"""

import jax.numpy as jnp

from normax.ec3.resistance import SHEAR_THRESHOLD
from normax.ec3.resistance import area_shear
from normax.ec3.resistance import resistance_shear
from normax.ec3.section import area
from normax.ec3.sizing import Steel
from normax.ec3.sizing import Tube
from normax.ec3.sizing import diameter_required
from normax.ec3.sizing import is_plastic
from normax.ec3.sizing import mass

STEEL = Steel()
LENGTH = 6000.0

# A demand mix swept from pure compression to pure bending, holding the axial
# force and growing the moment.
MOMENTS = jnp.asarray([0.0, 1e7, 2e7, 4e7, 8e7, 1.6e8])
FORCE = -6e5

CLASSES = (2, 3)


def sized(cross_section_class, m_y_ed, m_z_ed=0.0):
    """
    Fully-stressed diameter on one class branch.
    """
    tube = Tube.at_class_limit(STEEL.f_y, cross_section_class)

    return diameter_required(
        FORCE,
        m_y_ed,
        m_z_ed,
        0.9,
        0.9,
        LENGTH,
        STEEL,
        tube,
        plastic=is_plastic(cross_section_class),
    )


def member_mass(cross_section_class, m_y_ed):
    """
    Mass of one member of unit count on one class branch.
    """
    tube = Tube.at_class_limit(STEEL.f_y, cross_section_class)

    return mass(sized(cross_section_class, m_y_ed), LENGTH, STEEL, tube)


def main() -> None:
    """
    Sweep the demand mix and report which class limit is lighter.
    """
    print("Class 2 limit against Class 3 limit, S355, 6 m member, 600 kN compression\n")

    for cross_section_class in CLASSES:
        tube = Tube.at_class_limit(STEEL.f_y, cross_section_class)
        branch = "plastic" if is_plastic(cross_section_class) else "elastic"
        print(
            f"  Class {cross_section_class}: d/t = {float(tube.ratio):.3f}  ({branch})"
        )

    print(
        f"\n  {'M_y [kNm]':<12}{'d Class 2':<14}{'d Class 3':<14}"
        f"{'kg Class 2':<14}{'kg Class 3':<14}{'lighter':<12}{'saving'}"
    )

    for moment in MOMENTS:
        sizes = [float(sized(k, moment)) for k in CLASSES]
        masses = [float(member_mass(k, moment)) * 1e3 for k in CLASSES]
        lighter = CLASSES[int(jnp.argmin(jnp.asarray(masses)))]
        saving = abs(masses[0] - masses[1]) / max(masses) * 100.0
        print(
            f"  {float(moment) / 1e6:<12.0f}{sizes[0]:<14.2f}{sizes[1]:<14.2f}"
            f"{masses[0]:<14.2f}{masses[1]:<14.2f}Class {lighter:<6}{saving:5.2f}%"
        )

    print("\nWhere the crossover sits")
    fine = jnp.linspace(0.0, 1.6e8, 321)
    heavier = jnp.asarray([float(member_mass(2, m)) for m in fine])
    lighter = jnp.asarray([float(member_mass(3, m)) for m in fine])
    difference = heavier - lighter
    sign_change = jnp.where(jnp.diff(jnp.sign(difference)) != 0)[0]

    if sign_change.size == 0:
        winner = 2 if difference[0] < 0 else 3
        print(
            f"  no crossover in the swept range; Class {winner} is lighter throughout"
        )
    else:
        crossing = float(fine[int(sign_change[0])])
        print(f"  the two are equal near M_y = {crossing / 1e6:.1f} kNm")
        print("  below it one class wins, above it the other")

    print("\nShear the excluded clause would have seen (docs/clauses.md open item 0d)")
    print(
        f"  {'M_y [kNm]':<12}{'d [mm]':<12}{'V_pl,Rd [kN]':<16}"
        f"{'V_Ed for a simple span [kN]':<30}{'ratio'}"
    )

    for moment in MOMENTS[1:]:
        d = sized(3, moment)
        tube = Tube.at_class_limit(STEEL.f_y, 3)
        resistance = resistance_shear(
            area_shear(area(d, tube.ratio)), STEEL.f_y, STEEL.gamma_m0
        )
        # A simply supported span carrying that end moment has a shear of about
        # four moments over its length, which is the worst plausible pairing.
        shear = 4.0 * float(moment) / LENGTH
        ratio = shear / float(resistance)
        flag = "" if ratio < SHEAR_THRESHOLD else "   EXCEEDS HALF"
        print(
            f"  {float(moment) / 1e6:<12.0f}{float(d):<12.2f}"
            f"{float(resistance) / 1e3:<16.1f}{shear / 1e3:<30.1f}{ratio:.3f}{flag}"
        )

    print(
        f"\n  The exclusion of 6.2.6 stays honest while every ratio is under "
        f"{SHEAR_THRESHOLD}."
    )
    print("  Recompute this on the converged design before quoting it.")

    print("\nThe two readings of Eq. 6.42, on the Class 3 branch")
    print("  The guide says 6.2.9.2 permits only a linear interaction of stresses;")
    print("  the ECCS says the stress is evaluated by an elastic stress analysis.")
    print("  Blueprints implements 6.42 with the stress as an input and so does not")
    print("  settle it; Karamba implements no 6.2.9 at all. See docs/clauses.md.\n")
    print(
        f"  {'M_y = M_z [kNm]':<18}{'resultant':<14}{'linear sum':<14}"
        f"{'diameter':<12}{'area'}"
    )

    tube = Tube.at_class_limit(STEEL.f_y, 3)
    for moment in MOMENTS[1:]:
        readings = [
            float(
                diameter_required(
                    FORCE,
                    moment,
                    moment,
                    0.9,
                    0.9,
                    LENGTH,
                    STEEL,
                    tube,
                    plastic=False,
                    resultant=choice,
                )
            )
            for choice in (True, False)
        ]
        widening = readings[1] / readings[0]
        print(
            f"  {float(moment) / 1e6:<18.0f}{readings[0]:<14.2f}{readings[1]:<14.2f}"
            f"{widening - 1.0:<12.2%}{widening**2 - 1.0:.2%}"
        )

    print("\n  The gap closes wherever Eq. 6.61 governs, since that equation already")
    print("  sums the two moments linearly, and vanishes under uniaxial bending.")


if __name__ == "__main__":
    main()
