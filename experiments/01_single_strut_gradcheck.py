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
Four independent derivatives of one strut, tabulated against each other.

The milestone that de-risks everything downstream. A single circular hollow
section is sized to carry an axial force over a buckling length, and the
sensitivity of its diameter to both is computed four ways that share almost no
code:

    forward     the implicit tangent rule of normax.ec3.sizing
    reverse     that same rule, transposed by JAX into an adjoint
    closed      normax.ec3.adjoint, derived on paper and written out in full
    numeric     a central difference of the forward solve

Run with `uv run python experiments/01_single_strut_gradcheck.py`.
"""

import jax
import jax.numpy as jnp

from normax.ec3.actions import MemberActions
from normax.ec3.adjoint import derivative_force
from normax.ec3.adjoint import derivative_force_tension
from normax.ec3.adjoint import derivative_length
from normax.ec3.material import SteelGrade
from normax.ec3.section import TubeCatalogue
from normax.ec3.sizing import diameter_required
from normax.ec3.sizing import utilization_design

STEEL = SteelGrade()
CATALOGUE = TubeCatalogue.at_class_limit(STEEL.f_y, 3)
SECTION_CLASS = 3

TARGET = 1e-8

CASES = [
    (-1e4, 4000.0),
    (-1e5, 4000.0),
    (-5e5, 4000.0),
    (-5e5, 12000.0),
    (-2e6, 8000.0),
    (-1e7, 6000.0),
]

TENSION = [1e4, 1e5, 5e5, 5e6]


def size(axial_force, buckling_length):
    """
    Fully-stressed diameter under axial force alone.
    """
    return diameter_required(
        MemberActions(axial_force, 0.0, 0.0, 1.0, 1.0),
        buckling_length,
        STEEL,
        CATALOGUE,
        section_class=SECTION_CLASS,
    )


def used(d, axial_force, buckling_length):
    """
    Utilization at a diameter, from the exact clause functions.
    """
    return utilization_design(
        CATALOGUE.tube_at(d),
        MemberActions(axial_force, 0.0, 0.0, 1.0, 1.0),
        buckling_length,
        STEEL,
        section_class=SECTION_CLASS,
    )


def central(f, x, step):
    """
    Central difference of a scalar function.
    """
    return (f(x + step) - f(x - step)) / (2.0 * step)


def relative(a, b):
    """
    Relative difference between two derivatives.
    """
    return abs(a - b) / max(abs(b), 1e-300)


def report(label, forward, reverse, closed, numeric):
    """
    One row of the comparison table.
    """
    worst = max(
        relative(forward, closed),
        relative(reverse, closed),
        relative(numeric, closed),
    )
    verdict = "ok" if worst < TARGET else "FAIL"
    print(
        f"  {label:<22}{forward:+.12e}  {reverse:+.12e}  "
        f"{closed:+.12e}  {numeric:+.12e}  {worst:8.2e}  {verdict}"
    )

    return worst


def main() -> None:
    """
    Tabulate the four derivatives over a range of struts.
    """
    print(__doc__.strip().splitlines()[0])
    ratio = float(CATALOGUE.ratio)
    print(f"\nS355 hot-finished tube at the Class 3 limit, d/t = {ratio:.2f}")
    print(f"agreement target {TARGET:.0e}\n")

    header = (
        f"  {'case':<22}{'forward':<20}{'reverse':<20}"
        f"{'closed form':<20}{'central diff':<20}{'worst':<10}"
    )

    print("Compression, sensitivity of the diameter to the axial force")
    print(header)
    worst_overall = 0.0
    for axial_force, buckling_length in CASES:
        solved = size(axial_force, buckling_length)
        forward = float(jax.jacfwd(size)(axial_force, buckling_length))
        reverse = float(jax.grad(size)(axial_force, buckling_length))
        closed = float(
            derivative_force(solved, axial_force, buckling_length, STEEL, CATALOGUE)
        )
        numeric = float(
            central(
                lambda x: float(size(x, buckling_length)),
                axial_force,
                abs(axial_force) * 1e-6,
            )
        )
        label = f"{axial_force / 1e3:.0f} kN, {buckling_length / 1e3:.0f} m"
        worst_overall = max(
            worst_overall, report(label, forward, reverse, closed, numeric)
        )

    print("\nCompression, sensitivity of the diameter to the buckling length")
    print(header)
    for axial_force, buckling_length in CASES:
        solved = size(axial_force, buckling_length)
        forward = float(jax.jacfwd(size, argnums=1)(axial_force, buckling_length))
        reverse = float(jax.grad(size, argnums=1)(axial_force, buckling_length))
        closed = float(
            derivative_length(solved, axial_force, buckling_length, STEEL, CATALOGUE)
        )
        numeric = float(
            central(
                lambda x: float(size(axial_force, x)),
                buckling_length,
                buckling_length * 1e-6,
            )
        )
        label = f"{axial_force / 1e3:.0f} kN, {buckling_length / 1e3:.0f} m"
        worst_overall = max(
            worst_overall, report(label, forward, reverse, closed, numeric)
        )

    print("\nTension, where the answer is closed form and buckling never enters")
    print(header)
    for axial_force in TENSION:
        forward = float(jax.jacfwd(size)(axial_force, 4000.0))
        reverse = float(jax.grad(size)(axial_force, 4000.0))
        closed = float(derivative_force_tension(axial_force, STEEL, CATALOGUE))
        numeric = float(
            central(lambda x: float(size(x, 4000.0)), axial_force, axial_force * 1e-6)
        )
        label = f"{axial_force / 1e3:.0f} kN"
        worst_overall = max(
            worst_overall, report(label, forward, reverse, closed, numeric)
        )

    print("\nThe invariant the derivative rests on")
    worst_check = 0.0
    for axial_force, buckling_length in CASES:
        demand = float(
            used(size(axial_force, buckling_length), axial_force, buckling_length)
        )
        worst_check = max(worst_check, abs(demand - 1.0))
        force = axial_force / 1e3
        span = buckling_length / 1e3
        print(f"  {force:>8.0f} kN, {span:>4.0f} m   u = {demand:.16f}")

    print(f"\nworst derivative disagreement   {worst_overall:.2e}")
    print(f"worst departure from unity      {worst_check:.2e}")

    if worst_overall < TARGET and worst_check < 1e-9:
        print("\nPASS")
    else:
        print("\nFAIL")


if __name__ == "__main__":
    jnp.set_printoptions(precision=12)
    main()
