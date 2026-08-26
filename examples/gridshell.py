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
A gridshell designed end to end, a space frame behind the analysis boundary.

The truss race moved onto a shell, and onto the solver that can hold one: the
frame analysis is PyNite across the analysis schema, a public solver that does
not differentiate itself and whose adjoint is this repository's. The held-plan
subspace of the densities is thirteen wide, and the drawn cap is already
funicular under its own tributary pressure — every density a strut — so the
search leaves from the drawn geometry exactly. The radials are held in
compression through the descent: let them go and the search flattens the cap
and hangs the members, where no buckling reduction applies.

Run with `uv run python examples/gridshell.py [gridshell.yaml]`.
"""

import sys
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Int

from normax.blocks import build_pipeline
from normax.blocks import design_constraints
from normax.config import RunConfig
from normax.config import case_labels
from normax.config import parse_run
from normax.design import DesignProblem
from normax.design import compute_mass
from normax.design import initial_variables
from normax.design import optimize_design
from normax.design import read_design
from normax.figures import design_figures
from normax.form_finding import fit_densities
from normax.form_finding import held_plan_basis
from normax.loads import LoadCases
from normax.loads import build_load_cases
from normax.materials import Steel355
from normax.reporting import Report
from normax.reporting import report_descent
from normax.reporting import report_design
from normax.reporting import report_families
from normax.sections import build_section_family
from normax.structures import Structure
from normax.structures import build_gridshell_3d
from normax.symmetry import SignGuard
from normax.symmetry import guard_signs
from normax.symmetry import member_spread
from normax.viewer import view_designs

# The shell and the search, unless another file is named on the command line.
CONFIG = Path(__file__).with_name("gridshell.yaml")

REPO = Path(__file__).resolve().parent.parent
FIGURES = REPO / "figures"
DATA = REPO / "data"

COMPILATION_CACHE = REPO / ".jax_cache"
COMPILATION_CACHE.mkdir(exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(COMPILATION_CACHE))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)


class ShellConfig(NamedTuple):
    """
    The gridshell to build.

    Attributes
    ----------
    num_rings :
        Number of rings between the apex and the boundary, boundary included.
    num_spokes :
        Number of spokes radiating from the apex.
    radius :
        Radius of the circular plan of the cap.
    rise :
        Height of the apex above the plane of the boundary.
    oculus :
        Whether the crown is open.
    braced :
        Whether the quads are triangulated.
    polar_diameters :
        Whether the diameters are folded by the polar symmetry as well as the
        mirror, one section per ring per family.
    guard_hoops :
        Whether the compression guard covers the hoops as well as the radials.
    """

    num_rings: int
    num_spokes: int
    radius: float
    rise: float
    oculus: bool
    braced: bool
    polar_diameters: bool
    guard_hoops: bool


def build_shell(config: ShellConfig) -> Structure:
    """
    The drawn cap.
    """
    shell = build_gridshell_3d(
        config.num_rings,
        config.num_spokes,
        config.radius,
        config.rise,
        config.oculus,
        config.braced,
    )

    return shell


def ring_nodes(config: ShellConfig, spokes: np.ndarray) -> Int[np.ndarray, "nodes"]:
    """
    Node indices of every ring under a spoke permutation, the apex fixed.
    """
    offset = 0 if config.oculus else 1
    rings = [
        offset + ring * config.num_spokes + spokes for ring in range(config.num_rings)
    ]
    ringed = np.concatenate(rings)
    if config.oculus:
        return ringed

    return np.concatenate([[0], ringed])


def mirrored_nodes(config: ShellConfig) -> Int[np.ndarray, "nodes"]:
    """
    Mirror image of every node index about the plane through spoke zero.
    """
    spokes = np.arange(config.num_spokes)

    return ring_nodes(config, (-spokes) % config.num_spokes)


def rotated_nodes(config: ShellConfig) -> Int[np.ndarray, "nodes"] | None:
    """
    Node image under a rotation of one spoke, where the diameters fold by it.
    """
    if not config.polar_diameters:
        return None
    spokes = np.arange(config.num_spokes)

    return ring_nodes(config, (spokes + 1) % config.num_spokes)


def member_families(config: ShellConfig) -> tuple[tuple[str, slice], ...]:
    """
    Name and member slice of every family, in the generator's order.
    """
    reaching = config.num_rings - 1 if config.oculus else config.num_rings
    radials = reaching * config.num_spokes
    panels = (config.num_rings - 1) * config.num_spokes
    families = [
        ("radial", slice(0, radials)),
        ("hoop", slice(radials, radials + panels)),
    ]
    if config.braced:
        families.append(("diagonal", slice(radials + panels, None)))

    return tuple(families)


def guarded_members(config: ShellConfig) -> Int[np.ndarray, "guarded"]:
    """
    Members the compression guard holds: the radials, or every member.
    """
    families = member_families(config)
    reach = len(families) if config.guard_hoops else 1
    covered = families[reach - 1][1].stop or 0

    return np.arange(covered)


def compressive_start(
    structure: Structure,
    loads: LoadCases,
    config: RunConfig[ShellConfig, None],
) -> tuple[np.ndarray, SignGuard | None]:
    """
    The drawn cap's own funicular densities, and the guard holding their signs.

    Parameters
    ----------
    structure :
        The cap as drawn.
    loads :
        The load cases, the first of which the fit balances.
    config :
        The run description, read for the guard's reach and margin.

    Returns
    -------
    start :
        The fitted densities, and the guard, or None at a margin of zero.

    Raises
    ------
    ValueError
        If a guarded member of the drawn cap is not compressive by the margin,
        which no shift could repair since the fit has no self-stress.
    """
    fit = fit_densities(structure, np.asarray(structure.nodes), loads.formfinding)
    guarded = guarded_members(config.structure)
    if config.subspace.margin_fraction <= 0.0:
        return fit.q, None

    signs = -np.ones(guarded.size)
    guard = guard_signs(fit.q, signs, guarded, config.subspace.margin_fraction)
    worst = float(np.max(fit.q[guarded]))
    if worst > -guard.margin:
        raise ValueError(
            f"a guarded member is not compressive by the margin {guard.margin:.4f}: "
            f"worst density {worst:.4f}"
        )

    return fit.q, guard


def main(config_path: Path) -> None:
    """
    Design the shell a file describes, and report what the descent bought.

    Parameters
    ----------
    config_path :
        File naming the shell and the settings a design of it is searched for
        under.
    """
    config: RunConfig[ShellConfig, None] = parse_run(
        config_path.read_text(), ShellConfig
    )
    report = Report()
    report.write_banner("Gridshell — one search to a design")

    # The structure, its load cases, and the three blocks built on it.
    structure = build_shell(config.structure)
    loads = build_load_cases(structure, config.load_cases)
    family = build_section_family(Steel355(), config.sizing.section_class)
    pipeline = build_pipeline(structure, family, config.analysis, config.sizing)

    # The subspace holding the plan, folded by the mirror; the diameters folded
    # by the mirror and, where asked, the polar rotation as well.
    mirror = mirrored_nodes(config.structure) if config.subspace.symmetric else None
    basis = held_plan_basis(structure, mirror, config.subspace.pivoted)
    spread = member_spread(structure, (mirror, rotated_nodes(config.structure)))

    # The start, and the guard the descent runs under.
    q_start, guard = compressive_start(structure, loads, config)
    constraints = design_constraints(config.constraints, guard)
    problem = DesignProblem(structure, pipeline, loads, basis, spread, constraints)
    start = initial_variables(problem, q_start, config.analysis.diameter)
    initial = read_design(problem, start)

    report.write_heading("Backends")
    entries = [
        ("analysis", config.analysis.backend),
        ("sizing", config.sizing.backend),
        ("coordinates", str(basis.width)),
        ("variables", str(start.size)),
    ]
    report.write_entries(entries)

    # The descent: one reverse pass per gradient, whatever the constraint set.
    found = optimize_design(problem, start, config.augmented)
    optimized = read_design(problem, found.variables)

    report.write_heading("The descent")
    report_descent(report, found)
    report_design(report, initial, "The start")
    report_design(report, optimized, "The answer")
    report_families(report, optimized, member_families(config.structure))
    saved = 1.0 - float(compute_mass(optimized)) / float(compute_mass(initial))
    report.write_entries([("saved", f"{100.0 * saved:.2f} %")])

    # The record, and the figures.
    DATA.mkdir(exist_ok=True)
    np.savez(
        DATA / "gridshell.npz",
        variables=found.variables,
        objectives=found.objectives,
        violations=found.violations,
    )
    FIGURES.mkdir(exist_ok=True)
    designs = {"start": initial, "answer": optimized}
    labels = case_labels(config.load_cases)
    drawn, descended = design_figures(structure, designs, labels, found)
    drawn.savefig(FIGURES / "gridshell_designs.png", dpi=200)
    descended.savefig(FIGURES / "gridshell_descent.png", dpi=200)
    written = [("figures", str(FIGURES)), ("data", str(DATA / "gridshell.npz"))]
    report.write_entries(written)

    if config.viewer:
        view_designs(structure, pipeline.analyzer, loads, designs, labels)


if __name__ == "__main__":
    jnp.set_printoptions(precision=12)
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else CONFIG)
