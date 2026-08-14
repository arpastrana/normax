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

Run with `uv run --group pipeline python experiments/08_arch_formfind_analyze.py`.
"""

from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.equilibrium import EquilibriumStructure
from jaxtyping import Array
from jaxtyping import Float
from smax import diagnose_mechanisms

from normax.analysis.smax import frame_model
from normax.analysis.smax import member_forces
from normax.analysis.smax import prepare_model
from normax.design import MemberForces
from normax.ec3.material import Steel
from normax.ec3.section import TubeCatalogue
from normax.form_finding.fdm import equilibrium_graph
from normax.form_finding.fdm import equilibrium_state
from normax.reporting import Report
from normax.reporting import ReportColumn
from normax.reporting import ToleranceCheck
from normax.reporting import checks_passed
from normax.structures import Structure
from normax.structures import arch_2d
from normax.structures import loads_uniform
from normax.visualization import GapScaling
from normax.visualization import GradientCheck
from normax.visualization import HandoffForces
from normax.visualization import figure_handoff

# A 10 m arch of ten members under a 20 kN load at every free node. Units are
# millimeters and newtons throughout, as in every other module here.
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

DIAMETERS = (50.0, 100.0, 200.0, 400.0)
MODULI = (70_000.0, 210_000.0, 400_000.0)
SCALES = (0.1, 1.0, 10.0)

# Step the gradient's central differences are taken at, in force density.
STEP = 1e-3

TOLERANCE_AXIAL = 2.5e-4
TOLERANCE_BENDING = 1.0e-3
TOLERANCE_GRADIENT = 1e-7

FIGURES = Path(__file__).resolve().parent.parent / "figures"

STEEL = Steel()
CATALOGUE = TubeCatalogue.at_class_limit(STEEL.f_y, 3)


class FunicularArch(NamedTuple):
    """
    The shape form finding found, and the forces it carries by construction.

    Attributes
    ----------
    structure :
        The structure supplying the connectivity, the supports and the loads.
    graph :
        The form-finding connectivity, from `normax.form_finding`.
    state :
        The equilibrium state at the chosen force densities.
    axial_force :
        Member force the force density method implies, as `q` times a length.
    loads :
        The load case the arch was form-found under and is analyzed in.
    """

    structure: Structure
    graph: EquilibriumStructure
    state: EquilibriumState
    axial_force: Float[Array, "edges"]
    loads: Float[Array, "nodes 3"]

    @property
    def lengths(self) -> Float[Array, "edges"]:
        """
        Length of every member of the found shape.
        """
        return self.state.lengths[:, 0]


class HandoffGap(NamedTuple):
    """
    How far the analyzed forces depart from the ones form finding implied.

    Attributes
    ----------
    axial :
        Largest relative disagreement on the member axial force.
    bending :
        Largest end moment as a fraction of the axial force times the length,
        which is what explains the disagreement.
    """

    axial: float
    bending: float


class GradientRow(NamedTuple):
    """
    One force density's derivative, beside a central difference of the same.

    Attributes
    ----------
    edge :
        Index of the edge the force density belongs to.
    exact :
        Derivative from tracing form finding and the frame solve together.
    numeric :
        Central difference of the same composed objective.
    """

    edge: int
    exact: float
    numeric: float

    @property
    def relative(self) -> float:
        """
        Relative departure of the derivative from the central difference.
        """
        return abs(self.exact - self.numeric) / abs(self.numeric)


def funicular_arch(load: float, force_density: float) -> FunicularArch:
    """
    Form-find the arch, and report the state the analysis has to reproduce.
    """
    structure = arch_2d(num_edges=NUM_EDGES, span=SPAN, rise=SPAN / 3.0)
    applied = loads_uniform(structure, load)
    graph = equilibrium_graph(structure)
    q = jnp.full(NUM_EDGES, force_density)
    state = equilibrium_state(q, structure.nodes[graph.indices_fixed], graph, applied)
    axial_force = q * state.lengths[:, 0]
    arch = FunicularArch(structure, graph, state, axial_force, applied)

    return arch


def handoff_gap(
    diameter: float,
    steel: Steel,
    load: float = LOAD,
    force_density: float = FORCE_DENSITY,
) -> HandoffGap:
    """
    Largest relative disagreement on axial force, and the bending behind it.
    """
    arch = funicular_arch(load, force_density)
    prepared = prepare_model(arch.structure, steel, CATALOGUE, normal=NORMAL)
    diameters = jnp.full(NUM_EDGES, diameter)
    member = member_forces(
        prepared, arch.state.xyz, diameters, steel, CATALOGUE, arch.loads
    )

    departure = jnp.abs(member.axial_force - arch.axial_force)
    axial = jnp.max(departure / jnp.abs(arch.axial_force))
    peak = jnp.max(jnp.abs(member.moment_major), axis=1)
    reference = jnp.abs(arch.axial_force * arch.lengths)
    bending = jnp.max(peak / reference)
    gap = HandoffGap(float(axial), float(bending))

    return gap


def central_difference(
    function: Callable[[Float[Array, "edges"]], Float[Array, ""]],
    x: Float[Array, "edges"],
    index: int,
    step: float,
) -> float:
    """
    Central difference of a scalar function in one entry of its argument.
    """
    forward = function(x.at[index].add(step))
    backward = function(x.at[index].add(-step))

    return float((forward - backward) / (2.0 * step))


def report_shape(report: Report, arch: FunicularArch, mechanisms: int) -> None:
    """
    What form finding returned, and that the frame built on it is not a mechanism.
    """
    rise = float(jnp.max(arch.state.xyz[:, 2]))
    spread = float(jnp.max(jnp.abs(arch.state.xyz[:, 1])))
    residual = float(jnp.max(jnp.abs(arch.state.residuals[1:-1])))
    entries = (
        ("crown rise", f"{rise:.1f} mm"),
        ("out-of-plane spread", f"{spread:.1e} mm"),
        ("worst residual at a free node", f"{residual:.1e} N"),
        ("zero-energy modes in the frame", f"{mechanisms}"),
    )

    report.write_line("The shape form finding found")
    report.write_entries(entries)


def report_members(
    report: Report,
    arch: FunicularArch,
    member: MemberForces,
) -> None:
    """
    What each member was handed, and what the frame solver made of it.
    """
    columns = (
        ReportColumn("edge"),
        ReportColumn("q L [kN]", ".4f"),
        ReportColumn("smax N [kN]", ".4f"),
        ReportColumn("gap", ".2e"),
        ReportColumn("M/(N L)", ".2e"),
    )
    rows = []
    for edge in range(NUM_EDGES):
        expected = float(arch.axial_force[edge])
        analyzed = float(member.axial_force[edge])
        peak = float(jnp.max(jnp.abs(member.moment_major[edge])))
        gap = abs(analyzed - expected) / abs(expected)
        bending = peak / abs(expected * float(arch.lengths[edge]))
        rows.append((edge, expected / 1e3, analyzed / 1e3, gap, bending))

    report.write_heading(f"Member by member, at a diameter of {DIAMETER:.0f} mm")
    report.write_table(columns, rows)


def report_scaling(report: Report) -> list[HandoffGap]:
    """
    That the gap is quadratic in the diameter, and free of modulus and scale.

    Returns the gap at every diameter, which the figure draws as well.
    """
    by_diameter = [handoff_gap(diameter, STEEL) for diameter in DIAMETERS]
    diameter_columns = (
        ReportColumn("d [mm]", ".1f"),
        ReportColumn("gap", ".2e"),
        ReportColumn("M/(N L)", ".2e"),
        ReportColumn("gap / (d/100)^2", ".2e"),
    )
    diameter_rows = []
    for diameter, found in zip(DIAMETERS, by_diameter):
        scaled = found.axial / (diameter / DIAMETER) ** 2
        diameter_rows.append((diameter, found.axial, found.bending, scaled))

    report.write_heading("The gap is quadratic in the diameter")
    report.write_table(diameter_columns, diameter_rows)

    gap_columns = (ReportColumn("E [N/mm2]", ".0f"), ReportColumn("gap", ".12e"))
    modulus_rows = []
    for e_mod in MODULI:
        steel = STEEL._replace(e_mod=e_mod)
        modulus_rows.append((e_mod, handoff_gap(DIAMETER, steel).axial))

    heading = "And free of the modulus, which cancels between bending and axial"
    report.write_heading(heading)
    report.write_table(gap_columns, modulus_rows)

    scale_columns = (
        ReportColumn("loads and q times", ".1f"),
        ReportColumn("gap", ".12e"),
    )
    scale_rows = []
    for scale in SCALES:
        found = handoff_gap(DIAMETER, STEEL, LOAD * scale, FORCE_DENSITY * scale)
        scale_rows.append((scale, found.axial))

    heading = "And free of the scale of the loading, which leaves the shape alone"
    report.write_heading(heading)
    report.write_table(scale_columns, scale_rows)

    return by_diameter


def report_gradient(report: Report, rows: list[GradientRow]) -> float:
    """
    The gradient that crosses both stages, and the worst error in it.
    """
    columns = (
        ReportColumn("edge"),
        ReportColumn("autodiff", ".4f"),
        ReportColumn("central", ".4f"),
        ReportColumn("relative", ".2e"),
    )
    printed = [(row.edge, row.exact, row.numeric, row.relative) for row in rows]

    report.write_heading("The gradient crosses both stages")
    report.write_table(columns, printed)

    return max(row.relative for row in rows)


def main(verbose: bool = True) -> None:
    """
    Hand one shape across the boundary, and measure what came back.
    """
    report = Report(verbose)

    arch = funicular_arch(LOAD, FORCE_DENSITY)
    diameters = jnp.full(NUM_EDGES, DIAMETER)
    prepared = prepare_model(arch.structure, STEEL, CATALOGUE, normal=NORMAL)
    member = member_forces(
        prepared, arch.state.xyz, diameters, STEEL, CATALOGUE, arch.loads
    )

    model = frame_model(
        arch.structure, arch.state.xyz, diameters, STEEL, CATALOGUE, normal=NORMAL
    )
    mechanisms = diagnose_mechanisms(model).num_mechanisms

    report_shape(report, arch, mechanisms)
    report_members(report, arch, member)
    by_diameter = report_scaling(report)

    def objective(q):
        state = equilibrium_state(
            q,
            arch.structure.nodes[arch.graph.indices_fixed],
            arch.graph,
            arch.loads,
        )
        analyzed = member_forces(
            prepared, state.xyz, diameters, STEEL, CATALOGUE, arch.loads
        )

        return jnp.sum(analyzed.axial_force**2)

    q = jnp.full(NUM_EDGES, FORCE_DENSITY)
    gradient = jax.grad(objective)(q)
    rows = []
    for edge in range(NUM_EDGES):
        numeric = central_difference(objective, q, edge, STEP)
        rows.append(GradientRow(edge, float(gradient[edge]), numeric))

    worst_gradient = report_gradient(report, rows)
    found = handoff_gap(DIAMETER, STEEL)

    peaks = jnp.max(jnp.abs(member.moment_major), axis=1)
    forces = HandoffForces(arch.lengths, arch.axial_force, member.axial_force, peaks)
    axial_gaps = np.asarray([found.axial for found in by_diameter])
    scaling = GapScaling(np.asarray(DIAMETERS), axial_gaps, DIAMETER)
    numeric = np.asarray([row.numeric for row in rows])
    checked = GradientCheck(gradient, numeric)

    FIGURES.mkdir(exist_ok=True)
    handoff = figure_handoff(forces, scaling, checked)
    path = FIGURES / "08_handoff.png"
    handoff.savefig(path, dpi=160, bbox_inches="tight")
    report.write_heading(f"figure written to {path}")

    checks = (
        ToleranceCheck("axial disagreement", found.axial, TOLERANCE_AXIAL),
        ToleranceCheck("bending share", found.bending, TOLERANCE_BENDING),
        ToleranceCheck("gradient error", worst_gradient, TOLERANCE_GRADIENT),
    )
    finite = bool(jnp.all(jnp.isfinite(gradient)))
    passed = checks_passed(checks) and mechanisms == 0 and finite

    report.write_heading("Summary")
    report.write_checks(checks)
    report.write_verdict(passed)


if __name__ == "__main__":
    jnp.set_printoptions(precision=12)
    main()
