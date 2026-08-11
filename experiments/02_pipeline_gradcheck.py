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
The sizing map under axial force and biaxial bending, differentiated four ways.

Extends the single-strut check to the full interaction. There is no closed form
to compare against here, so the oracles are the forward tangent, its reverse
transposition, a central difference, and the gradient of the mass objective the
optimizer will actually descend.

Also confirms that removing the moments reproduces the axial answer exactly, and
reports which limit state decides each member.

Run with `uv run python experiments/02_pipeline_gradcheck.py`.
"""

import jax
import jax.numpy as jnp

from normax.ec3.actions import MemberActions
from normax.ec3.material import SteelGrade
from normax.ec3.sizing import LIMIT_CROSS_SECTION
from normax.ec3.sizing import LIMIT_MAJOR
from normax.ec3.sizing import LIMIT_MINIMUM_SIZE
from normax.ec3.sizing import LIMIT_MINOR
from normax.ec3.sizing import LIMIT_TENSION
from normax.ec3.sizing import TubeCatalogue
from normax.ec3.sizing import diameter_required
from normax.ec3.sizing import governing_limit_state
from normax.ec3.sizing import is_plastic
from normax.ec3.sizing import mass
from normax.ec3.sizing import utilization_design

STEEL = SteelGrade()
TARGET = 1e-6

NAMES = {
    LIMIT_MINIMUM_SIZE: "minimum size",
    LIMIT_TENSION: "tension",
    LIMIT_CROSS_SECTION: "cross-section",
    LIMIT_MAJOR: "Eq. 6.61",
    LIMIT_MINOR: "Eq. 6.62",
}

# force [N], major moment [N mm], minor moment [N mm], buckling length [mm]
CASES = [
    (-5e5, 4e7, 1.5e7, 4000.0),
    (-5e5, 4e7, 0.0, 4000.0),
    (-9e5, 8e7, 6e7, 12000.0),
    (0.0, 4e7, 1.5e7, 4000.0),
    (5e5, 4e7, 1.5e7, 4000.0),
    (-5e4, 5e6, 5e6, 8000.0),
]

# A relative step for each argument, since one absolute step cannot serve
# newtons and newton-millimetres at once.
STEPS = (1e-6, 1e-6, 1e-6, 1e-6)

LABELS = ("force", "major moment", "minor moment", "length")


def size(n_ed, m_y_ed, m_z_ed, l_cr, catalogue, plastic):
    """
    Fully-stressed diameter under the full interaction.
    """
    return diameter_required(
        MemberActions(n_ed, m_y_ed, m_z_ed, 0.9, 0.9),
        l_cr,
        STEEL,
        catalogue,
        plastic=plastic,
    )


def central(f, x, step):
    """
    Central difference of a scalar function.
    """
    return (f(x + step) - f(x - step)) / (2.0 * step)


def main() -> None:
    """
    Gradcheck every action, on both class branches.
    """
    print("The sizing map under axial force and biaxial bending\n")

    worst = 0.0
    for cross_section_class in (2, 3):
        catalogue = TubeCatalogue.at_class_limit(STEEL.f_y, cross_section_class)
        plastic = is_plastic(cross_section_class)
        branch = "plastic" if plastic else "elastic"
        ratio = float(catalogue.ratio)
        print(f"Class {cross_section_class} ({branch}), d/t = {ratio:.2f}")
        print(
            f"  {'case':<26}{'argument':<15}{'reverse':<22}"
            f"{'central diff':<22}{'rel':<10}"
        )

        for actions in CASES:
            for index, label in enumerate(LABELS):

                def at(x, actions=actions, index=index):
                    probed = list(actions)
                    probed[index] = x

                    return size(*probed, catalogue, plastic)

                value = actions[index]
                if value == 0.0:
                    continue

                reverse = float(jax.grad(at)(value))
                numeric = float(
                    central(lambda x: float(at(x)), value, abs(value) * STEPS[index])
                )
                relative = abs(reverse - numeric) / max(abs(numeric), 1e-300)
                worst = max(worst, relative)

                name = (
                    f"{actions[0] / 1e3:.0f} kN "
                    f"{actions[1] / 1e6:.0f}/{actions[2] / 1e6:.0f} kNm"
                )
                print(
                    f"  {name:<26}{label:<15}{reverse:+.12e}  "
                    f"{numeric:+.12e}  {relative:8.2e}"
                )
        print()

    print("Forward and reverse are the same derivative")
    catalogue = TubeCatalogue.at_class_limit(STEEL.f_y, 3)
    for actions in CASES[:3]:

        def at(x, actions=actions):
            probed = list(actions)
            probed[0] = x

            return size(*probed, catalogue, False)

        forward = float(jax.jacfwd(at)(actions[0]))
        reverse = float(jax.grad(at)(actions[0]))
        gap = abs(forward - reverse) / max(abs(reverse), 1e-300)
        print(f"  {actions[0] / 1e3:>8.0f} kN   forward-reverse gap {gap:.2e}")

    print("\nRemoving the moments reproduces the axial answer")
    for cross_section_class in (2, 3):
        catalogue = TubeCatalogue.at_class_limit(STEEL.f_y, cross_section_class)
        plastic = is_plastic(cross_section_class)
        with_moment = float(size(-5e5, 0.0, 0.0, 4000.0, catalogue, plastic))
        axial_only = float(
            diameter_required(
                MemberActions(-5e5, 0.0, 0.0, 1.0, 1.0),
                4000.0,
                STEEL,
                catalogue,
                plastic=plastic,
            )
        )
        print(
            f"  Class {cross_section_class}   {with_moment:.12f}  vs  "
            f"{axial_only:.12f}   gap {abs(with_moment - axial_only):.2e}"
        )

    print("\nUtilization and governing limit state at the solved diameter")
    catalogue = TubeCatalogue.at_class_limit(STEEL.f_y, 3)
    worst_check = 0.0
    for actions in CASES:
        d = size(*actions, catalogue, False)
        demand = float(
            utilization_design(
                d,
                MemberActions(*actions[:3], 0.9, 0.9),
                actions[3],
                STEEL,
                catalogue,
                plastic=False,
            )
        )
        code = float(
            governing_limit_state(
                d,
                MemberActions(*actions[:3], 0.9, 0.9),
                actions[3],
                STEEL,
                catalogue,
                plastic=False,
            )
        )
        worst_check = max(worst_check, abs(demand - 1.0))
        name = (
            f"{actions[0] / 1e3:.0f} kN "
            f"{actions[1] / 1e6:.0f}/{actions[2] / 1e6:.0f} kNm"
        )
        print(f"  {name:<26}d = {float(d):8.3f} mm   u = {demand:.15f}   {NAMES[code]}")

    print("\nThe mass objective is differentiable end to end")
    forces = jnp.asarray([-5e5, -9e5, 5e5, -5e4])
    lengths = jnp.asarray([4000.0, 12000.0, 4000.0, 8000.0])

    def objective(n_ed):
        sizes = diameter_required(
            MemberActions(n_ed, 4e7, 1.5e7, 0.9, 0.9),
            lengths,
            STEEL,
            catalogue,
            plastic=False,
        )

        return mass(sizes, lengths, STEEL, catalogue)

    total = float(objective(forces))
    gradient = jax.grad(objective)(forces)
    print(f"  mass {total * 1e3:.2f} kg")
    print(f"  d(mass)/d(force) {jnp.asarray(gradient)}")
    print(f"  all finite: {bool(jnp.all(jnp.isfinite(gradient)))}")

    print(f"\nworst derivative disagreement   {worst:.2e}")
    print(f"worst departure from unity      {worst_check:.2e}")
    print("\nPASS" if worst < TARGET and worst_check < 1e-9 else "\nFAIL")


if __name__ == "__main__":
    main()
