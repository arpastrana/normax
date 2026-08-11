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
Force densities to a mass, differentiably, with EN 1993-1-1 in the middle.

All three stages in one process and one gradient: `jax-fdm` finds the shape,
`smax` says what the members carry, and the sizing map returns the diameter at
which the standard is exactly satisfied. `normax.pipeline.mass` is the scalar,
and `jax.grad` of it crosses all three.

Three things are checked, and two more are measured rather than asserted.

    utilization  one to machine precision for every member, both class branches
    gradient     against central differences, at the step where they plateau
    refinement   the mass converges, first order in the number of members
    coupling     the analysis needs sections and the check returns them, so the
                 pass is repeated until the diameters stop moving
    stability    the critical load factor of the finished design, which is what
                 the member-length buckling length assumes rather than proves

**The buckling length is the member's own length, and that is a strong
assumption.** It presumes every node is held in plane by structure outside the
model, so a member can only buckle between its ends. The arch on its own does
not satisfy it: the critical load factor of the fully-stressed design is far
below one, in a mode that sways over the whole span rather than over one member,
and sizing against that mode instead costs several times the mass. Both numbers
are printed, so the assumption sits next to what it is worth.

**Refinement needs the shape held fixed, not the loads.** Leaving the nodal load
and the force density alone changes the arch as the mesh changes, and the mass
then moves for a reason that has nothing to do with discretization. The total
load is fixed here and the force densities are rescaled so the crown rise is
exactly the target — the force density method is linear in the coordinates, so
one trial solve fixes the scale with no formula and no iteration.

Run with `uv run --group pipeline python experiments/09_arch_pipeline_jax.py`.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from normax.analysis.smax import buckling_modes
from normax.analysis.smax import prepare_model
from normax.ec3.material import SteelGrade
from normax.ec3.section import TubeCatalogue
from normax.ec3.sizing import mass_of_tubes
from normax.ec3.stability import ALPHA_CR_ELASTIC
from normax.formfinding import equilibrium_graph
from normax.formfinding import equilibrium_state
from normax.pipeline import design_members
from normax.pipeline import frame_stability
from normax.pipeline import governing_states
from normax.pipeline import total_mass
from normax.structures import arch_2d
from normax.visualization import figure_convergence
from normax.visualization import figure_modes
from normax.visualization import figure_sections

# A 10 m arch rising 3 m, carrying 180 kN spread over its free nodes. Units are
# millimetres and newtons.
SPAN = 10_000.0
RISE = 3_000.0
TOTAL_LOAD = 180_000.0
NUM_EDGES = 10

# The arch lies in the XZ plane, so it has no thickness along Y.
NORMAL = 1

# The diameter the frame is analysed with before the check has spoken. Only the
# stiffness depends on it, and the check overwrites it.
SEED = 100.0

# Meshes for the refinement study, doubling.
MESHES = (5, 10, 20, 40, 80, 160)

# A buckling length independent of the mesh, so refinement measures the
# discretization rather than the slenderness.
BUCKLING_LENGTH = 1_000.0

PASSES = 6

# Modes to report, and the effective length of the arch's own critical mode as a
# fraction of its developed length. Measured, and steady to three figures across
# a 32-fold range of mesh density.
NUM_MODES = 4
GLOBAL_MODE_FACTOR = 0.576

# Relative step at which the central difference plateaus. Smaller is dominated
# by cancellation, larger by truncation, and the experiment prints the sweep.
STEPS = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
STEP = 1e-5

TOLERANCE_UTILIZATION = 1e-9
TOLERANCE_GRADIENT = 5e-8

# Refinement is first order, so successive halving of the change is the claim.
TOLERANCE_ORDER = 0.15

FIGURES = Path(__file__).resolve().parent.parent / "figures"

STEEL = SteelGrade()

# Preparing the analysis model needs a section family to stand up a frame, and
# every property of it is replaced per call, so one seed serves both classes.
CATALOGUE_SEED = TubeCatalogue.at_class_limit(STEEL.f_y, 3)

LIMIT_NAMES = {
    0.0: "catalogue minimum",
    1.0: "tension",
    2.0: "cross-section",
    3.0: "6.61 major",
    4.0: "6.62 minor",
}


def setup(num_edges):
    """
    The arch, its form-finding connectivity, and the `q` that reaches the rise.
    """
    load = TOTAL_LOAD / (num_edges - 1)
    structure = arch_2d(num_edges=num_edges, span=SPAN, rise=RISE, load=load)
    graph_fdm = equilibrium_graph(structure)
    model = prepare_model(structure, STEEL, CATALOGUE_SEED, normal=NORMAL)

    trial = jnp.full(num_edges, -1.0)
    reached = jnp.max(equilibrium_state(trial, structure, graph_fdm).xyz[:, 2])

    return structure, graph_fdm, model, trial * reached / RISE


def central(f, x, index, step):
    """
    Central difference of a scalar function in one entry of its argument.
    """
    return (f(x.at[index].add(step)) - f(x.at[index].add(-step))) / (2.0 * step)


def disagreement(exact, numeric, scale):
    """
    Gradient error, measured against the size of the whole gradient.

    Two of this arch's ten sensitivities are twenty times smaller than the rest,
    where they change sign. Dividing by the component would report an absolute
    difference of 1e-13 as an error of 1e-7 and say nothing about the derivative,
    so the largest component sets the scale instead.
    """
    return abs(exact - numeric) / scale


def relaxed(q, structure, graph_fdm, model, catalogue, *, section_class, passes):
    """
    Repeat the staggered analysis and check, reporting how far each pass moves.
    """
    diameters = jnp.full(q.shape[0], SEED)
    moves = []
    masses = []

    for _ in range(passes):
        result = design_members(
            q,
            diameters,
            structure,
            graph_fdm,
            model,
            STEEL,
            catalogue,
            section_class=section_class,
        )
        moves.append(
            float(jnp.max(jnp.abs(result.diameters - diameters) / result.diameters))
        )
        masses.append(float(result.mass))
        diameters = result.diameters

    return result, np.asarray(moves), np.asarray(masses)


def main():
    structure, graph_fdm, model, q = setup(NUM_EDGES)
    seed = jnp.full(NUM_EDGES, SEED)

    print("The arch")
    state = equilibrium_state(q, structure, graph_fdm)
    print(f"  span                {SPAN / 1e3:.1f} m over {NUM_EDGES} members")
    print(f"  crown rise          {float(jnp.max(state.xyz[:, 2])):.4f} mm")
    print(f"  force density       {float(q[0]):.6f} N/mm")
    print(f"  total load          {TOTAL_LOAD / 1e3:.1f} kN")

    worst_utilization = 0.0
    designs = {}
    for section_class in (2, 3):
        catalogue = TubeCatalogue.at_class_limit(STEEL.f_y, section_class)
        result = design_members(
            q,
            seed,
            structure,
            graph_fdm,
            model,
            STEEL,
            catalogue,
            section_class=section_class,
        )
        codes = governing_states(result, STEEL, catalogue, section_class=section_class)
        departure = float(jnp.max(jnp.abs(result.utilization - 1.0)))
        worst_utilization = max(worst_utilization, departure)
        designs[section_class] = (catalogue, section_class, result)

        print(f"\nClass {section_class}, d/t = {float(catalogue.ratio):.3f}")
        print(f"  {'member':>7} {'N [kN]':>10} {'M [kNm]':>10} {'d [mm]':>9} {'u':>18}")
        for member in range(NUM_EDGES):
            force = float(result.actions.axial_force[member]) / 1e3
            moment = float(result.actions.moment_major[member]) / 1e6
            print(
                f"  {member:>7} {force:>10.4f} {moment:>10.5f}"
                f" {float(result.diameters[member]):>9.4f}"
                f" {float(result.utilization[member]):>18.16f}"
            )
        print(f"  mass                {float(result.mass):.9f} t")
        print(f"  worst |u - 1|       {departure:.2e}")
        limits = {LIMIT_NAMES[float(code)] for code in codes}
        print(f"  governing           {', '.join(sorted(limits))}")

    catalogue, section_class, result = designs[3]

    print("\nThe central difference plateaus before it is trusted")

    def objective(q):
        return total_mass(
            q,
            seed,
            structure,
            graph_fdm,
            model,
            STEEL,
            catalogue,
            section_class=section_class,
        )

    gradient = jax.grad(objective)(q)
    scale = float(jnp.max(jnp.abs(gradient)))
    edges = (0, NUM_EDGES // 2, NUM_EDGES - 1)
    print(f"  {'relative step':>14} {'worst scaled error':>20}")
    for relative in STEPS:
        worst = 0.0
        for edge in edges:
            step = abs(float(q[edge])) * relative
            numeric = float(central(objective, q, edge, step))
            worst = max(worst, disagreement(float(gradient[edge]), numeric, scale))
        print(f"  {relative:>14.0e} {worst:>20.3e}")

    print(f"\nThe gradient of the mass, at a relative step of {STEP:.0e}")
    print(f"  {'edge':>5} {'autodiff':>22} {'central':>22} {'scaled':>10}")
    worst_gradient = 0.0
    numeric = []
    for edge in range(NUM_EDGES):
        numeric.append(float(central(objective, q, edge, abs(float(q[edge])) * STEP)))
        scaled = disagreement(float(gradient[edge]), numeric[-1], scale)
        worst_gradient = max(worst_gradient, scaled)
        print(
            f"  {edge:>5} {float(gradient[edge]):>22.14e} {numeric[-1]:>22.14e}"
            f" {scaled:>10.2e}"
        )

    print("\nThe staggered coupling closes geometrically")
    _, moves, masses = relaxed(
        q,
        structure,
        graph_fdm,
        model,
        catalogue,
        section_class=section_class,
        passes=PASSES,
    )
    print(f"  {'pass':>5} {'relative move':>15} {'mass [t]':>14} {'ratio':>8}")
    for step, (move, total) in enumerate(zip(moves, masses)):
        ratio = "" if step == 0 else f"{moves[step] / moves[step - 1]:.4f}"
        print(f"  {step:>5} {move:>15.3e} {total:>14.9f} {ratio:>8}")
    print(
        f"  one pass costs {abs(masses[0] - masses[-1]) / masses[-1]:.3%} of the mass"
    )

    print("\nThe mass converges as the mesh refines")
    print(
        f"  {'members':>8} {'arc [mm]':>12} {'mass, Lcr=member':>18}"
        f" {'mass, Lcr fixed':>17}"
    )
    by_member = []
    by_fixed = []
    for count in MESHES:
        refined, refined_graph, refined_model, refined_q = setup(count)
        refined_seed = jnp.full(count, SEED)
        free = design_members(
            refined_q,
            refined_seed,
            refined,
            refined_graph,
            refined_model,
            STEEL,
            catalogue,
            section_class=section_class,
        )
        held = design_members(
            refined_q,
            refined_seed,
            refined,
            refined_graph,
            refined_model,
            STEEL,
            catalogue,
            section_class=section_class,
            buckling_length=jnp.full(count, BUCKLING_LENGTH),
        )
        by_member.append(float(free.mass))
        by_fixed.append(float(held.mass))
        print(
            f"  {count:>8} {float(jnp.sum(free.lengths)):>12.4f}"
            f" {by_member[-1]:>18.9f} {by_fixed[-1]:>17.9f}"
        )

    by_member = np.asarray(by_member)
    by_fixed = np.asarray(by_fixed)

    # Richardson, for a sequence converging first order in the member count.
    limit = 2.0 * by_fixed[-1] - by_fixed[-2]

    changes = np.abs(np.diff(by_fixed)) / np.abs(by_fixed[1:])
    ratios = changes[:-1] / changes[1:]
    print(f"  extrapolated limit  {limit:.9f} t")
    print(f"  change ratios       {np.array2string(ratios, precision=3)}")
    worst_order = float(np.max(np.abs(ratios - 2.0)) / 2.0)
    print(f"  worst departure from first order  {worst_order:.3f}")

    print("\nThe global stability check, EN 1993-1-1 5.2.1(3)")
    checked = frame_stability(
        result,
        model,
        STEEL,
        catalogue,
        num_modes=NUM_MODES,
    )
    factors = np.asarray(checked.factors)
    arc = float(jnp.sum(result.lengths))

    verdict = "SATISFIED" if bool(checked.adequate) else "NOT SATISFIED"
    print(f"  alpha_cr                 {float(factors[0]):.4f}")
    print(f"  threshold                {ALPHA_CR_ELASTIC:.1f}")
    print(f"  utilization              {float(checked.utilization):.3f}")
    print(f"  verdict                  {verdict}")

    print("\nBoth of the standard's routes to the same slenderness")
    print(
        f"  {'member':>7} {'6.50 from L_cr':>15} {'6.3.4 from a_cr':>15}"
        f" {'ratio':>7} {'L_cr,global [mm]':>17}"
    )
    for member in range(NUM_EDGES):
        from_length = float(checked.slenderness_member[member])
        from_factor = float(checked.slenderness_global[member])
        print(
            f"  {member:>7} {from_length:>15.4f} {from_factor:>15.4f}"
            f" {from_factor / from_length:>7.2f}"
            f" {float(checked.buckling_length_equivalent[member]):>17.1f}"
        )

    print("\nWhat the member-length assumption is worth")

    unbraced = design_members(
        q,
        seed,
        structure,
        graph_fdm,
        model,
        STEEL,
        catalogue,
        section_class=section_class,
        buckling_length=jnp.full(NUM_EDGES, GLOBAL_MODE_FACTOR * arc),
    )
    penalty = float(unbraced.mass) / float(result.mass)

    print(f"  critical load factors  {np.array2string(factors, precision=4)}")
    print(f"  arc length             {arc:.1f} mm")
    print(f"  sized against L_cr = L                {float(result.mass):.6f} t")
    print(
        f"  sized against L_cr = {GLOBAL_MODE_FACTOR:.3f} arc  "
        f"{float(unbraced.mass):.6f} t   x{penalty:.2f}"
    )
    print(
        "  the member-length basis presumes every node is held in plane by"
        " structure outside the model"
    )
    print(
        "  the bare model does not satisfy 5.2.1, which is what makes that"
        " assumption load-bearing"
    )

    FIGURES.mkdir(exist_ok=True)

    assumed_mass = float(mass_of_tubes(catalogue.tube_at(seed), result.lengths, STEEL))
    sections = figure_sections(
        result.xyz,
        structure.edges,
        seed,
        result.diameters,
        assumed_mass,
        float(result.mass),
    )
    sections.savefig(FIGURES / "09_sections.png", dpi=160, bbox_inches="tight")

    convergence = figure_convergence(
        np.asarray(MESHES),
        by_member,
        by_fixed,
        limit,
        np.arange(len(moves)),
        moves,
    )
    convergence.savefig(FIGURES / "09_convergence.png", dpi=160, bbox_inches="tight")

    modes = buckling_modes(
        model,
        result.xyz,
        result.diameters,
        STEEL,
        catalogue,
        num_modes=NUM_MODES,
    )
    shapes = figure_modes(result.xyz, factors, np.asarray(modes.shapes), RISE)
    shapes.savefig(FIGURES / "09_modes.png", dpi=160, bbox_inches="tight")
    print(f"\nfigures written to {FIGURES}")

    print()
    for label, worst, tolerance in (
        ("departure from unity", worst_utilization, TOLERANCE_UTILIZATION),
        ("gradient error", worst_gradient, TOLERANCE_GRADIENT),
        ("departure from first order", worst_order, TOLERANCE_ORDER),
    ):
        print(f"worst {label:<27} {worst:.2e}   of {tolerance:.1e}")

    passed = (
        worst_utilization < TOLERANCE_UTILIZATION
        and worst_gradient < TOLERANCE_GRADIENT
        and worst_order < TOLERANCE_ORDER
        and bool(jnp.all(jnp.isfinite(gradient)))
    )
    print("\nPASS" if passed else "\nFAIL")


if __name__ == "__main__":
    jnp.set_printoptions(precision=12)
    main()
