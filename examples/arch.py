# SPDX-License-Identifier: Apache-2.0
"""
A steel arch designed end to end, through three blocks and one gradient.

**The whole project in one file.** A file describes an arch, its load cases and
the search; three blocks — a form finder, a frame analysis and a code check —
are built on that structure and composed into one function; the mass that comes
out has an exact gradient in every force density and every diameter, and an
augmented Lagrangian spends it under the check.

**A block is built from a structure and then called.** The constructor is where
each piece of software sees the structure in its own terms — a form finder wants
connectivity matrices, a frame solver an assembly, a code check nothing at all —
and it runs once, on the host. What is left is a function of design parameters
and load cases, which is what the optimizer differentiates and what compiles.

**The three disagree about how they compute, and the pipeline never asks.** Form
finding traces a linear solve in this process; the frame analysis and the code
check each cross a Tesseract boundary to a host that does not differentiate
itself, and come back with a hand-written adjoint. Swapping a block for one that
runs in process is a different word in the file and nothing else.

Run with `uv run python examples/arch.py [arch.yaml]`. Add
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
from normax.design import build_design_constraints
from normax.design import create_design
from normax.design import initialize_optimization_parameters
from normax.design import solve_problem
from normax.exporting import ExportTarget
from normax.exporting import export_design
from normax.form_finding import UniformDensityInitializer
from normax.form_finding import build_form_finder
from normax.form_finding import build_plan_basis
from normax.loads import build_load_cases
from normax.materials import Steel355
from normax.reporting import report_design
from normax.sections import UniformDiameterInitializer
from normax.sections import build_section_catalog
from normax.structures import ArchDescription
from normax.structures import Structure
from normax.structures import build_arch_2d
from normax.symmetry import build_height_groups
from normax.symmetry import build_section_groups
from normax.symmetry import find_mirror_nodes
from normax.tesseract import TesseractAnalyzer
from normax.tesseract import TesseractSizer

# The arch and the search, unless another file is named on the command line.
CONFIG = Path(__file__).with_name("arch.yaml")

REPO = Path(__file__).resolve().parent.parent
TITLE = "Arch — one search to a design"
MATERIAL = Steel355()
PRECISION = 12
EXPORT = ExportTarget("arch", REPO / "data", REPO / "figures")


def build_arch(description: ArchDescription) -> Structure:
    """
    The arch, as topology and a starting geometry.

    Parameters
    ----------
    description :
        The arch to build.

    Returns
    -------
    structure :
        A funicular arch between two pinned supports.
    """
    return build_arch_2d(description.num_edges, description.span, description.rise)


def main(arguments: RunArguments) -> None:
    """
    Design the arch a file describes, and report what the descent bought.

    Parameters
    ----------
    arguments :
        File naming the arch and the settings a design of it is searched for
        under.
    """
    config: RunConfig[ArchDescription] = read_run_config(arguments, ArchDescription)

    # The structure, its load cases, and the catalog both backends draw from
    structure = build_arch(config.structure)
    loads = build_load_cases(structure, config.load_cases)
    section_catalog = build_section_catalog(MATERIAL, config.sizing.section_class)

    # The plan is held, and a chain leaves one independent density to move;
    # the midspan mirror folds the sections and the written heights
    mirror = find_mirror_nodes(structure, config.form_finding.mirror)
    basis = build_plan_basis(structure, mirror, config.form_finding.basis)
    folded = mirror if config.sizing.fold_mirror else None
    section_groups = build_section_groups(structure, (folded, None))
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

    # The start: one force density and one diameter per member
    density_initializer = UniformDensityInitializer(config.form_finding.density_start)
    density_start = density_initializer(structure, loads.formfinding, basis, None)
    diameter_initializer = UniformDiameterInitializer(config.analysis.diameter_start)
    diameter_start = diameter_initializer(structure)

    # Constraints spur creativity
    constraints = build_design_constraints(config.constraints, None, density_start)

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
    record = ProblemRecord(problem, solution, design, design_found, None)
    report_design(record, config, TITLE)
    export_design(record, config, EXPORT)


if __name__ == "__main__":
    jnp.set_printoptions(precision=PRECISION)
    main(read_run_arguments(sys.argv[1:], CONFIG))
