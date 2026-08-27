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

import jax.numpy as jnp
import numpy as np
from jaxtyping import Int

from normax.config import RunConfig
from normax.config import parse_config
from normax.design import DesignProblem
from normax.design import DesignRecord
from normax.design import build_design_constraints
from normax.design import evaluate_design
from normax.design import initialize_optimization_variables
from normax.design import optimize_design
from normax.exporting import ExportTarget
from normax.exporting import export_design
from normax.form_finding import build_plan_basis
from normax.form_finding import fit_densities
from normax.loads import LoadCases
from normax.loads import build_load_cases
from normax.materials import Steel355
from normax.reporting import report_design
from normax.sections import build_section_family
from normax.structures import ShellDescription
from normax.structures import Structure
from normax.structures import build_gridshell_3d
from normax.symmetry import SignGuard
from normax.symmetry import build_member_spread
from normax.symmetry import guard_signs
from normax.tesseract import build_pipeline
from normax.viewer import view_design

# The shell and the search, unless another file is named on the command line.
CONFIG = Path(__file__).with_name("gridshell.yaml")

REPO = Path(__file__).resolve().parent.parent
TITLE = "Gridshell — one search to a design"
EXPORT = ExportTarget("gridshell", REPO / "data", REPO / "figures")


def build_shell(description: ShellDescription) -> Structure:
    """
    The drawn cap.
    """
    shell = build_gridshell_3d(
        description.num_rings,
        description.num_spokes,
        description.radius,
        description.rise,
        description.oculus,
        description.braced,
    )

    return shell


def permute_rings(
    description: ShellDescription, spokes: np.ndarray
) -> Int[np.ndarray, "nodes"]:
    """
    Node indices of every ring under a spoke permutation, the apex fixed.
    """
    offset = 0 if description.oculus else 1
    rings = [
        offset + ring * description.num_spokes + spokes
        for ring in range(description.num_rings)
    ]
    ringed = np.concatenate(rings)
    if description.oculus:
        return ringed

    return np.concatenate([[0], ringed])


def mirror_nodes(description: ShellDescription) -> Int[np.ndarray, "nodes"]:
    """
    Mirror image of every node index about the plane through spoke zero.
    """
    spokes = np.arange(description.num_spokes)

    return permute_rings(description, (-spokes) % description.num_spokes)


def rotate_nodes(description: ShellDescription) -> Int[np.ndarray, "nodes"] | None:
    """
    Node image under a rotation of one spoke, where the diameters fold by it.
    """
    if not description.polar_diameters:
        return None
    spokes = np.arange(description.num_spokes)

    return permute_rings(description, (spokes + 1) % description.num_spokes)


def list_families(description: ShellDescription) -> tuple[tuple[str, slice], ...]:
    """
    Name and member slice of every family, in the generator's order.
    """
    reaching = (
        description.num_rings - 1 if description.oculus else description.num_rings
    )
    radials = reaching * description.num_spokes
    panels = (description.num_rings - 1) * description.num_spokes
    families = [
        ("radial", slice(0, radials)),
        ("hoop", slice(radials, radials + panels)),
    ]
    if description.braced:
        families.append(("diagonal", slice(radials + panels, None)))

    return tuple(families)


def select_guarded_members(description: ShellDescription) -> Int[np.ndarray, "guarded"]:
    """
    Members the compression guard holds: the radials, or every member.
    """
    families = list_families(description)
    reach = len(families) if description.guard_hoops else 1
    covered = families[reach - 1][1].stop or 0

    return np.arange(covered)


def initialize_densities(
    structure: Structure,
    loads: LoadCases,
    config: RunConfig[ShellDescription, None],
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
        The run config, read for the guard's reach and margin.

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
    guarded = select_guarded_members(config.structure)
    if config.constraints.sign_margin_fraction <= 0.0:
        return fit.q, None

    signs = -np.ones(guarded.size)
    guard = guard_signs(fit.q, signs, guarded, config.constraints.sign_margin_fraction)
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
    config: RunConfig[ShellDescription, None] = parse_config(
        config_path.read_text(), ShellDescription
    )

    # The structure, its load cases, and the three blocks built on it.
    structure = build_shell(config.structure)
    loads = build_load_cases(structure, config.load_cases)
    family = build_section_family(Steel355(), config.sizing.section_class)
    pipeline = build_pipeline(structure, family, config.analysis, config.sizing)

    # The subspace holding the plan, folded by the mirror; the diameters folded
    # by the mirror and, where asked, the polar rotation as well.
    mirror = mirror_nodes(config.structure) if config.subspace.symmetric else None
    basis = build_plan_basis(structure, mirror, config.subspace.pivoted)
    spread = build_member_spread(structure, (mirror, rotate_nodes(config.structure)))

    # The start, and the guard the descent runs under.
    q_start, guard = initialize_densities(structure, loads, config)
    constraints = build_design_constraints(config.constraints, guard)
    problem = DesignProblem(structure, pipeline, loads, basis, spread, constraints)
    d_start = config.analysis.diameter
    start = initialize_optimization_variables(problem, q_start, d_start)
    initial = evaluate_design(problem, start)

    # The descent: one reverse pass per gradient, whatever the constraint set.
    found = optimize_design(problem, start, config.optimization)
    optimized = evaluate_design(problem, found.variables)

    # What the run arrived at; the report, the record and the viewer read it.
    families = list_families(config.structure)
    record = DesignRecord(problem, found, initial, optimized, families)
    report_design(record, config, TITLE)
    export_design(record, config, EXPORT)
    view_design(record, config)


if __name__ == "__main__":
    jnp.set_printoptions(precision=12)
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else CONFIG)
