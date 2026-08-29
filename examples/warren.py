# SPDX-License-Identifier: Apache-2.0
"""
A Warren truss designed end to end, inside the funicular subspace of its plan.

Same three blocks as the arch, same descent, one difference: the force
densities move in the subspace that holds the drawn plan in horizontal
equilibrium, so every geometry the search reaches keeps the bays where they
were drawn and the coefficients need no bounds. The truss is once statically
indeterminate, so the funicular fit has a state of self-stress — the split
between hanging deck and arching top chord — and the start is shifted along it
until both chords carry their signs.

Run with `uv run python examples/warren.py [warren.yaml]`. Add
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
from normax.design import optimize_design
from normax.exporting import ExportTarget
from normax.exporting import export_design
from normax.form_finding import LensShapeInitializer
from normax.form_finding import build_form_finder
from normax.form_finding import build_plan_basis
from normax.loads import build_load_cases
from normax.materials import Steel355
from normax.reporting import report_design
from normax.sections import UniformDiameterInitializer
from normax.sections import build_section_catalog
from normax.structures import Structure
from normax.structures import TrussDescription
from normax.structures import build_warren_2d
from normax.structures import create_groups_warren
from normax.symmetry import build_section_groups
from normax.symmetry import find_mirror_nodes
from normax.tesseract import TesseractAnalyzer
from normax.tesseract import TesseractSizer

# The truss and the search, unless another file is named on the command line
CONFIG = Path(__file__).with_name("warren.yaml")

REPO = Path(__file__).resolve().parent.parent
MATERIAL = Steel355()
PRECISION = 12
TITLE = "Warren truss — one search to a design"
EXPORT = ExportTarget("warren", REPO / "data", REPO / "figures")


def build_truss(description: TrussDescription) -> Structure:
    """
    The truss as drawn.
    """
    return build_warren_2d(description.num_bays, description.span, description.depth)


def main(arguments: RunArguments) -> None:
    """
    Design the truss a file describes, and report what the descent bought.

    Parameters
    ----------
    arguments :
        File naming the truss and the settings a design of it is searched for
        under.
    """
    config: RunConfig[TrussDescription] = read_run_config(arguments, TrussDescription)

    # The structure, its load cases, and the catalog both backends draw from
    structure = build_truss(config.structure)
    loads = build_load_cases(structure, config.load_cases)
    section_catalog = build_section_catalog(MATERIAL, config.sizing.section_class)

    # The plan is held in a subspace, and the symmetries fold the sizes
    mirror = find_mirror_nodes(structure, config.form_finding.mirror)
    basis = build_plan_basis(structure, mirror, config.form_finding.basis)
    folded = mirror if config.sizing.fold_mirror else None
    section_groups = build_section_groups(structure, (folded, None))

    # The three main computation blocks of the structural design pipeline
    form_finder = build_form_finder(structure, basis, config.form_finding)
    analyzer = TesseractAnalyzer(structure, section_catalog, config.analysis.backend)
    sizer = TesseractSizer(structure, section_catalog, config.sizing.backend)

    # One pipeline to rule them all
    pipeline = StructuralDesignPipeline(form_finder, analyzer, sizer)

    # The start: a lens truss with the same diameter per member and guarded signs
    groups = create_groups_warren(config.structure)
    guarded = assign_signs(config.constraints, groups, structure.num_edges)
    density_initializer = LensShapeInitializer(config.form_finding.density_start)
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
    solution = optimize_design(problem, params, config.optimization)
    design_found = create_design(problem, solution.parameters)

    # Is every member cross-section compliant with the structural engineering standard?
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
