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
Form finding hands a geometry to a frame solver, and the forces are compared.

The first crossing of a real boundary in this pipeline. `jax-fdm` finds the
shape that carries the loads in pure compression; `smax` is handed that shape
and nothing else, and works out for itself what the members carry. The two
never exchange a force, so their agreement is a prediction rather than an
identity.

It cannot be exact, and the reason is worth stating. Form finding returns a
polygon with a kink at every node, and a chain of beams cannot turn a kink on
axial force alone: continuity of rotation demands a moment. That moment scales
as the square of the radius of gyration over the member length, so the gap
closes as the members thin, and it depends on neither the modulus nor the scale
of the loading. The table below shows all three.

Run with `uv run --group pipeline python experiments/08_arch_formfind_analyse.py`.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from smax import diagnose_mechanisms

from normax.analysis.smax import forces
from normax.analysis.smax import frame
from normax.analysis.smax import prepare
from normax.ec3.material import SteelGrade
from normax.ec3.sizing import TubeCatalogue
from normax.formfinding import equilibrium
from normax.formfinding import graph
from normax.structures import arch
from normax.visualization import figure_handoff

# A 10 m arch of ten members under a 20 kN load at every free node. Units are
# millimetres and newtons throughout, as in every other module here.
SPAN = 10_000.0
LOAD = 20_000.0
NUM_EDGES = 10
FORCE_DENSITY = -75.0

# The arch lies in the XZ plane, so it has no thickness along Y. Without this
# the frame is a mechanism and the solve returns nan.
NORMAL = 1

# Near the size EN 1993-1-1 asks for on this arch, and the size the recorded
# tolerances belong to.
DIAMETER = 100.0

DIAMETERS = [50.0, 100.0, 200.0, 400.0]
MODULI = [70_000.0, 210_000.0, 400_000.0]
SCALES = [0.1, 1.0, 10.0]

TOLERANCE_AXIAL = 2.5e-4
TOLERANCE_BENDING = 1.0e-3
TOLERANCE_GRADIENT = 1e-7

FIGURES = Path(__file__).resolve().parent.parent / "figures"

STEEL = SteelGrade()
CATALOGUE = TubeCatalogue.at_class_limit(STEEL.f_y, 3)


def funicular(load, force_density):
    """
    Form-find the arch, and report the state the analysis has to reproduce.
    """
    structure = arch(num_edges=NUM_EDGES, span=SPAN, rise=SPAN / 3.0, load=load)
    fdm = graph(structure)
    q = jnp.full(NUM_EDGES, force_density)
    state = equilibrium(q, structure, fdm)

    return structure, fdm, state, q * state.lengths[:, 0]


def gap(diameter, steel, load=LOAD, force_density=FORCE_DENSITY):
    """
    Largest relative disagreement on axial force, and the bending that explains it.
    """
    structure, _, state, axial = funicular(load, force_density)
    member = forces(
        prepare(structure, steel, CATALOGUE, normal=NORMAL),
        state.xyz,
        jnp.full(NUM_EDGES, diameter),
        steel,
        CATALOGUE,
    )

    deviation = jnp.max(jnp.abs(member.n_ed - axial) / jnp.abs(axial))
    peak = jnp.max(jnp.abs(member.m_y_ed), axis=1)
    bending = jnp.max(peak / jnp.abs(axial * state.lengths[:, 0]))

    return float(deviation), float(bending)


def central(f, x, index, step):
    """
    Central difference of a scalar function in one entry of its argument.
    """
    return (f(x.at[index].add(step)) - f(x.at[index].add(-step))) / (2.0 * step)


def main():
    structure, fdm, state, axial = funicular(LOAD, FORCE_DENSITY)
    diameters = jnp.full(NUM_EDGES, DIAMETER)
    prepared = prepare(structure, STEEL, CATALOGUE, normal=NORMAL)
    member = forces(prepared, state.xyz, diameters, STEEL, CATALOGUE)

    model = frame(structure, state.xyz, diameters, STEEL, CATALOGUE, normal=NORMAL)
    mechanisms = diagnose_mechanisms(model).num_mechanisms

    rise = float(jnp.max(state.xyz[:, 2]))
    spread = float(jnp.max(jnp.abs(state.xyz[:, 1])))
    residual = float(jnp.max(jnp.abs(state.residuals[1:-1])))

    print("The shape form finding found")
    print(f"  crown rise                     {rise:.1f} mm")
    print(f"  out-of-plane spread            {spread:.1e} mm")
    print(f"  worst residual at a free node  {residual:.1e} N")
    print(f"  zero-energy modes in the frame {mechanisms}")

    print(f"\nMember by member, at a diameter of {DIAMETER:.0f} mm")
    print(f"  {'':>4} {'q L [kN]':>12} {'smax N [kN]':>13} {'gap':>10} {'M/(N L)':>10}")
    for edge in range(NUM_EDGES):
        peak = float(jnp.max(jnp.abs(member.m_y_ed[edge])))
        length = float(state.lengths[edge, 0])
        expected = float(axial[edge])
        analysed = float(member.n_ed[edge])
        print(
            f"  {edge:>4} {expected / 1e3:>12.4f} {analysed / 1e3:>13.4f}"
            f" {abs(analysed - expected) / abs(expected):>10.2e}"
            f" {peak / abs(expected * length):>10.2e}"
        )

    print("\nThe gap is quadratic in the diameter")
    print(f"  {'d [mm]':>8} {'gap':>10} {'M/(N L)':>10} {'gap / (d/100)^2':>16}")
    for diameter in DIAMETERS:
        deviation, bending = gap(diameter, STEEL)
        print(
            f"  {diameter:>8.1f} {deviation:>10.2e} {bending:>10.2e}"
            f" {deviation / (diameter / DIAMETER) ** 2:>16.2e}"
        )

    print("\nAnd free of the modulus, which cancels between bending and axial")
    for e_mod in MODULI:
        deviation, _ = gap(DIAMETER, STEEL._replace(e_mod=e_mod))
        print(f"  E = {e_mod:>9.0f} N/mm2   gap = {deviation:.12e}")

    print("\nAnd free of the scale of the loading, which leaves the shape alone")
    for scale in SCALES:
        deviation, _ = gap(DIAMETER, STEEL, LOAD * scale, FORCE_DENSITY * scale)
        print(f"  loads and q x {scale:<5.1f}     gap = {deviation:.12e}")

    def objective(q):
        state = equilibrium(q, structure, fdm)
        member = forces(prepared, state.xyz, diameters, STEEL, CATALOGUE)

        return jnp.sum(member.n_ed**2)

    q = jnp.full(NUM_EDGES, FORCE_DENSITY)
    gradient = jax.grad(objective)(q)

    print("\nThe gradient crosses both stages")
    print(f"  {'edge':>4} {'autodiff':>18} {'central':>18} {'relative':>10}")
    worst_gradient = 0.0
    differences = []
    for edge in range(NUM_EDGES):
        differences.append(float(central(objective, q, edge, 1e-3)))
        exact = float(gradient[edge])
        relative = abs(exact - differences[-1]) / abs(differences[-1])
        worst_gradient = max(worst_gradient, relative)
        print(f"  {edge:>4} {exact:>18.4f} {differences[-1]:>18.4f} {relative:>10.2e}")

    deviation, bending = gap(DIAMETER, STEEL)

    FIGURES.mkdir(exist_ok=True)
    handoff = figure_handoff(
        state.lengths[:, 0],
        axial,
        member.n_ed,
        jnp.max(jnp.abs(member.m_y_ed), axis=1),
        np.asarray(DIAMETERS),
        np.asarray([gap(diameter, STEEL)[0] for diameter in DIAMETERS]),
        DIAMETER,
        gradient,
        np.asarray(differences),
    )
    handoff.savefig(FIGURES / "08_handoff.png", dpi=160, bbox_inches="tight")
    print(f"\nfigure written to {FIGURES / '08_handoff.png'}")

    print()
    for label, worst, tolerance in (
        ("axial disagreement", deviation, TOLERANCE_AXIAL),
        ("bending share", bending, TOLERANCE_BENDING),
        ("gradient error", worst_gradient, TOLERANCE_GRADIENT),
    ):
        print(f"worst {label:<20} {worst:.2e}   of {tolerance:.1e}")

    passed = (
        mechanisms == 0
        and deviation < TOLERANCE_AXIAL
        and bending < TOLERANCE_BENDING
        and worst_gradient < TOLERANCE_GRADIENT
        and bool(jnp.all(jnp.isfinite(gradient)))
    )
    print("\nPASS" if passed else "\nFAIL")


if __name__ == "__main__":
    jnp.set_printoptions(precision=12)
    main()
