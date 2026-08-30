# SPDX-License-Identifier: Apache-2.0
"""
A Vierendeel truss designed end to end, on the truss where funicularity is scarce.

The Warren without its diagonals. A pin-jointed Vierendeel is a mechanism, so
the frame analysis models every member as a rigid-jointed beam and the panels
carry load through joint bending: no shape makes it momentless under the
asymmetric cases, and the check's interaction of axial force and bending governs
rather than the axial resistance alone. The held-plan subspace is nine wide and
is searched in the members' own densities, and the chord signs are guarded
descent: a chord density crossing zero hands the form finder a singular
stiffness, so the guard keeps every trial point on the signed sheet.

Run with `uv run python examples/vierendeel.py [vierendeel.yaml]`. Add
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
from normax.form_finding import LensShapeInitializer
from normax.form_finding import build_form_finder
from normax.form_finding import build_plan_basis
from normax.form_finding import read_lens_shape
from normax.loads import build_load_cases
from normax.materials import Steel355
from normax.reporting import report_design
from normax.sections import UniformDiameterInitializer
from normax.sections import build_section_catalog
from normax.structures import Structure
from normax.structures import TrussDescription
from normax.structures import build_vierendeel_2d
from normax.structures import create_groups_vierendeel
from normax.symmetry import build_height_groups
from normax.symmetry import build_section_groups
from normax.symmetry import find_mirror_nodes
from normax.tesseract import TesseractAnalyzer
from normax.tesseract import TesseractSizer

# The truss and the search, unless another file is named on the command line
CONFIG = Path(__file__).with_name("vierendeel.yaml")

REPO = Path(__file__).resolve().parent.parent
MATERIAL = Steel355()
PRECISION = 12
TITLE = "Vierendeel truss — one search to a design"
EXPORT = ExportTarget("vierendeel", REPO / "data", REPO / "figures")


def build_truss(description: TrussDescription) -> Structure:
    """
    The truss as drawn, four supports at the chord ends.
    """
    return build_vierendeel_2d(
        description.num_bays, description.span, description.depth
    )


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
    lifted = mirror if config.form_finding.fold_heights else None
    height_groups = build_height_groups(structure, (lifted,))

    # The lens all three parametrizations open on, so a baseline is a shaped
    # truss rather than the flat line the truss happens to be drawn along
    start_shape = read_lens_shape(structure, config.form_finding.density_start)

    # The three main computation blocks of the structural design pipeline
    form_finder = build_form_finder(
        structure, basis, config.form_finding, height_groups, start_shape
    )
    analyzer = TesseractAnalyzer(structure, section_catalog, config.analysis.backend)
    sizer = TesseractSizer(structure, section_catalog, config.sizing.backend)

    # One pipeline to rule them all
    pipeline = StructuralDesignPipeline(form_finder, analyzer, sizer)

    # The start: a lens truss with the same diameter per member and guarded signs
    groups = create_groups_vierendeel(config.structure)
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
    solution = solve_problem(
        problem, params, config.optimization, config.output.verbose
    )
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
