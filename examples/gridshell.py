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
from normax.design import ProblemRecord
from normax.design import StructuralDesignPipeline
from normax.design import assign_signs
from normax.design import build_design_constraints
from normax.design import create_design
from normax.design import initialize_optimization_parameters
from normax.design import solve_problem
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
from normax.structures import build_shell
from normax.structures import create_groups_shell
from normax.symmetry import build_height_groups
from normax.symmetry import build_section_groups
from normax.symmetry import find_mirror_nodes
from normax.symmetry import find_rotated_nodes
from normax.tesseract import TesseractAnalyzer
from normax.tesseract import TesseractSizer

# The shell and the search, unless another file is named on the command line
CONFIG = Path(__file__).with_name("gridshell.yaml")

REPO = Path(__file__).resolve().parent.parent
MATERIAL = Steel355()
PRECISION = 12
TITLE = "Gridshell — one search to a design"
EXPORT = ExportTarget("gridshell", REPO / "data", REPO / "figures")


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

    # The structure, its load cases, and the catalog both backends draw from
    structure = build_shell(config.structure)
    loads = build_load_cases(structure, config.load_cases)
    section_catalog = build_section_catalog(MATERIAL, config.sizing.section_class)

    # The plan is held in a subspace, and the symmetries fold the sizes
    mirror = find_mirror_nodes(structure, config.form_finding.mirror)
    basis = build_plan_basis(structure, mirror, config.form_finding.basis)
    folded = mirror if config.sizing.fold_mirror else None
    rotation = None
    if config.sizing.fold_polar:
        spokes = read_polar_plan(structure).num_spokes
        rotation = find_rotated_nodes(structure, spokes)
    section_groups = build_section_groups(structure, (folded, rotation))
    # Only the mirror folds a height: the rotation carries no load case onto
    # another, so folding by it would hold a shape the loads do not ask for.
    lifted = mirror if config.form_finding.fold_heights else None
    height_groups = build_height_groups(structure, (lifted,))

    # The three main computation blocks of the structural design pipeline
    form_finder = build_form_finder(
        structure, basis, config.form_finding, height_groups
    )
    analyzer = TesseractAnalyzer(structure, section_catalog, config.analysis.backend)
    sizer = TesseractSizer(structure, section_catalog, config.sizing.backend)

    # One pipeline to rule them all
    pipeline = StructuralDesignPipeline(form_finder, analyzer, sizer)

    # The start: the drawn cap's own fit, signed by the guard the file names
    groups = create_groups_shell(config.structure)
    guarded = assign_signs(config.constraints, groups, structure.num_edges)
    density_initializer = DrawnShapeInitializer(config.form_finding.density_start)
    density_start = density_initializer(structure, loads.formfinding, basis, guarded)
    diameter_initializer = UniformDiameterInitializer(config.analysis.diameter_start)
    diameter_start = diameter_initializer(structure)

    # Constraints spur creativity
    constraints = build_design_constraints(config.constraints, guarded, density_start)

    # Construct the design task with all the ingredients created thus far
    problem = DesignProblem(structure, pipeline, loads, constraints, section_groups)

    # Generate structural design from initial guess of design parameters
    params = initialize_optimization_parameters(problem, density_start, diameter_start)
    design = create_design(problem, params)

    # Search, baby, search...
    solution = solve_problem(
        problem, params, config.optimization, config.output.verbose
    )
    design_found = create_design(problem, solution.parameters)

    # Is every member cross-section compliant to the structural engineering standards?
    slack = 1.0 + config.optimization.violation_tol
    is_design_safe = bool(jnp.all(design_found.sizes.utilization <= slack))
    print(f"Is the design safe? {is_design_safe}")

    # Bureaucracy
    record = ProblemRecord(problem, solution, design, design_found, groups)
    report_design(record, config, TITLE)
    export_design(record, config, EXPORT)


if __name__ == "__main__":
    jnp.set_printoptions(precision=PRECISION)
    main(read_run_arguments(sys.argv[1:], CONFIG))
