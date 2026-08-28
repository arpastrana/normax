# SPDX-License-Identifier: Apache-2.0
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

Run with `uv run python examples/gridshell.py [gridshell.yaml]`. Add
`--shape-parametrization heights` or `fixed` to race the same structure,
loads, analysis and check against a geometry written down rather than found.
"""

import sys
from pathlib import Path

import jax.numpy as jnp

from normax.config import RunArguments
from normax.config import RunConfig
from normax.config import read_run_arguments
from normax.config import read_run_config
from normax.design import DesignProblem
from normax.design import DesignRecord
from normax.design import StructuralDesignPipeline
from normax.design import assign_signs
from normax.design import build_design_constraints
from normax.design import create_design
from normax.design import initialize_optimization_variables
from normax.design import optimize_design
from normax.exporting import ExportTarget
from normax.exporting import export_design
from normax.form_finding import DrawnShapeInitializer
from normax.form_finding import build_form_finder
from normax.form_finding import build_plan_basis
from normax.loads import build_load_cases
from normax.loads import read_polar_plan
from normax.materials import Steel355
from normax.reporting import report_design
from normax.sections import UniformDiameterInitializer
from normax.sections import build_section_catalog
from normax.structures import ShellDescription
from normax.structures import Structure
from normax.structures import build_gridshell_3d
from normax.structures import create_groups_shell
from normax.symmetry import build_section_groups
from normax.symmetry import find_mirror_nodes
from normax.symmetry import find_rotated_nodes
from normax.tesseract import TesseractAnalyzer
from normax.tesseract import TesseractSizer
from normax.visualization import view_design

# The shell and the search, unless another file is named on the command line.
CONFIG = Path(__file__).with_name("gridshell.yaml")

REPO = Path(__file__).resolve().parent.parent
MATERIAL = Steel355()
PRECISION = 12
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


def main(arguments: RunArguments) -> None:
    """
    Design the shell a file describes, and report what the descent bought.

    Parameters
    ----------
    arguments :
        File naming the shell and the settings a design of it is searched for
        under.
    """
    config: RunConfig[ShellDescription] = read_run_config(arguments, ShellDescription)

    # The structure, its load cases, and the catalog both backends draw from.
    structure = build_shell(config.structure)
    loads = build_load_cases(structure, config.load_cases)
    section_catalog = build_section_catalog(MATERIAL, config.sizing.section_class)

    # The plan is held in a subspace, and the symmetries fold the sizes.
    mirror = find_mirror_nodes(structure, config.form_finding.mirror)
    basis = build_plan_basis(structure, mirror, config.form_finding.basis)
    folded = mirror if config.sizing.fold_mirror else None
    rotation = None
    if config.sizing.fold_polar:
        spokes = read_polar_plan(structure).num_spokes
        rotation = find_rotated_nodes(structure, spokes)
    section_groups = build_section_groups(structure, (folded, rotation))

    # The three main computation blocks of the structural design pipeline
    form_finder = build_form_finder(structure, basis, config.form_finding)
    analyzer = TesseractAnalyzer(structure, section_catalog, config.analysis.backend)
    sizer = TesseractSizer(structure, section_catalog, config.sizing.backend)

    pipeline = StructuralDesignPipeline(form_finder, analyzer, sizer)

    # The start: the drawn cap's own fit, signed by the guard the file names.
    groups = create_groups_shell(config.structure)
    guarded = assign_signs(config.constraints, groups, structure.num_edges)
    density_initializer = DrawnShapeInitializer(config.form_finding.density_start)
    density_start = density_initializer(structure, loads.formfinding, basis, guarded)
    constraints = build_design_constraints(config.constraints, guarded, density_start)
    problem = DesignProblem(structure, pipeline, loads, constraints, section_groups)
    diameter_initializer = UniformDiameterInitializer(config.analysis.diameter_start)
    diameter_start = diameter_initializer(structure)
    start = initialize_optimization_variables(problem, density_start, diameter_start)
    initial = create_design(problem, start)

    # The descent: one reverse pass per gradient, whatever the constraint set.
    found = optimize_design(problem, start, config.optimization)
    optimized = create_design(problem, found.variables)

    # Is every member within what EN 1993-1-1 allows, to the search's own
    # tolerance? A fully-stressed design sits on the constraint, not below it.
    slack = 1.0 + config.optimization.violation_tol
    is_design_safe = bool(jnp.all(optimized.sizes.utilization <= slack))
    print(f"design safe: {is_design_safe}")

    # What the run arrived at; the report, the record and the viewer read it.
    record = DesignRecord(problem, found, initial, optimized, groups)
    report_design(record, config, TITLE)
    export_design(record, config, EXPORT)
    view_design(record, config)


if __name__ == "__main__":
    jnp.set_printoptions(precision=PRECISION)
    main(read_run_arguments(sys.argv[1:], CONFIG))
