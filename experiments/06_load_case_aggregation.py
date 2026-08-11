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

import jax
import jax.numpy as jnp

from normax.ec3.actions import MemberActions
from normax.ec3.material import SteelGrade
from normax.ec3.sizing import TubeCatalogue
from normax.ec3.sizing import diameter_envelope
from normax.ec3.sizing import diameter_required
from normax.ec3.sizing import is_plastic
from normax.ec3.sizing import mass

STEEL = SteelGrade()
CATALOGUE = TubeCatalogue.at_class_limit(STEEL.f_y, 3)
PLASTIC = is_plastic(3)

LENGTHS = jnp.asarray([4000.0, 6000.0, 5000.0, 4500.0])

# Three load cases over four members: symmetric, half-span asymmetric, and a
# crown point load. The third member reverses from compression to tension.
FORCES = jnp.asarray(
    [
        [-6e5, -4e5, -3e5, -5e5],
        [-8e5, -2e5, 2e5, -6e5],
        [-3e5, -7e5, -1e5, -9e5],
    ]
)
MOMENTS = jnp.asarray(
    [
        [2e7, 1e7, 5e6, 1.5e7],
        [5e7, 3e7, 2e7, 4e7],
        [1e7, 6e7, 1e7, 2e7],
    ]
)

SHARPNESS = [5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0]


def sizes_per_case():
    """
    Fully-stressed diameter of every member under every load case.
    """
    return diameter_required(
        MemberActions(FORCES, MOMENTS, 0.0, 0.9, 0.9),
        LENGTHS,
        STEEL,
        CATALOGUE,
        plastic=PLASTIC,
    )


def total_mass(beta):
    """
    Mass of the structure sized by the smooth envelope at a given sharpness.
    """
    return mass(diameter_envelope(sizes_per_case(), beta), LENGTHS, STEEL, CATALOGUE)


def main() -> None:
    """
    Anneal the sharpness and report what the smoothing costs.
    """
    per_case = sizes_per_case()
    exact = jnp.max(per_case, axis=0)
    exact_mass = float(mass(exact, LENGTHS, STEEL, CATALOGUE)) * 1e3

    print("Three load cases over four members, S355 at the Class 3 limit\n")
    print(f"  {'member':<10}{'case 1':<12}{'case 2':<12}{'case 3':<12}{'exact max'}")
    for member in range(per_case.shape[1]):
        row = "".join(f"{float(per_case[case, member]):<12.2f}" for case in range(3))
        print(f"  {member:<10}{row}{float(exact[member]):.2f}")

    print(f"\n  exact mass {exact_mass:.2f} kg")

    print("\nAnnealing the sharpness")
    print(
        f"  {'beta':<10}{'mass [kg]':<14}{'excess':<12}"
        f"{'bound log(cases)/beta':<24}{'gradient finite'}"
    )

    for beta in SHARPNESS:
        smoothed = float(total_mass(beta)) * 1e3
        excess = (smoothed - exact_mass) / exact_mass * 100.0
        bound = float(jnp.log(per_case.shape[0]) / beta) * 100.0
        gradient = jax.grad(
            lambda f: mass(
                diameter_envelope(
                    diameter_required(
                        MemberActions(f, MOMENTS, 0.0, 0.9, 0.9),
                        LENGTHS,
                        STEEL,
                        CATALOGUE,
                        plastic=PLASTIC,
                    ),
                    beta,
                ),
                LENGTHS,
                STEEL,
                CATALOGUE,
            )
        )(FORCES)
        finite = bool(jnp.all(jnp.isfinite(gradient)))
        print(
            f"  {beta:<10.0f}{smoothed:<14.2f}{excess:>7.3f}%    "
            f"{bound:>17.3f}%      {finite}"
        )

    print("\n  The excess is an overestimate of the size, never an underestimate,")
    print("  so every intermediate design satisfies the standard.")

    print("\nEvery case sees a gradient, which a hard maximum would not give")
    beta = 50.0

    def objective(forces):
        sizes = diameter_required(
            MemberActions(forces, MOMENTS, 0.0, 0.9, 0.9),
            LENGTHS,
            STEEL,
            CATALOGUE,
            plastic=PLASTIC,
        )

        return mass(diameter_envelope(sizes, beta), LENGTHS, STEEL, CATALOGUE)

    def hard(forces):
        sizes = diameter_required(
            MemberActions(forces, MOMENTS, 0.0, 0.9, 0.9),
            LENGTHS,
            STEEL,
            CATALOGUE,
            plastic=PLASTIC,
        )

        return mass(jnp.max(sizes, axis=0), LENGTHS, STEEL, CATALOGUE)

    smooth_gradient = jax.grad(objective)(FORCES)
    hard_gradient = jax.grad(hard)(FORCES)

    print(f"  {'':<10}{'smooth, cases with a gradient':<34}{'hard maximum'}")
    for member in range(FORCES.shape[1]):
        live_smooth = int(jnp.sum(jnp.abs(smooth_gradient[:, member]) > 0.0))
        live_hard = int(jnp.sum(jnp.abs(hard_gradient[:, member]) > 0.0))
        print(f"  member {member:<3}{live_smooth:<34}{live_hard}")

    print("\nThe member that changes sign between cases")
    reversing = 2
    print(f"  member {reversing} forces per case: {FORCES[:, reversing]}")
    print(f"  sizes per case: {per_case[:, reversing]}")
    print(f"  gradient per case: {smooth_gradient[:, reversing]}")
    print("  finite throughout, despite the standard being discontinuous at zero")


if __name__ == "__main__":
    main()
