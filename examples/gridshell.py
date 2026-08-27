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

Run with `uv run python examples/gridshell.py [gridshell.yaml]`.
"""

import sys
from pathlib import Path

import jax.numpy as jnp

from normax.config import RunConfig
from normax.config import parse_config
from normax.design import DesignProblem
from normax.design import DesignRecord
from normax.design import assign_signs
from normax.design import build_design_constraints
from normax.design import evaluate_design
from normax.design import initialize_optimization_variables
from normax.design import optimize_design
from normax.exporting import ExportTarget
from normax.exporting import export_design
from normax.loads import build_load_cases
from normax.materials import Steel355
from normax.reporting import report_design
from normax.sections import build_section_family
from normax.structures import ShellDescription
from normax.structures import Structure
from normax.structures import build_gridshell_3d
from normax.structures import list_shell_families
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


def main(config_path: Path) -> None:
    """
    Design the shell a file describes, and report what the descent bought.

    Parameters
    ----------
    config_path :
        File naming the shell and the settings a design of it is searched for
        under.
    """
    config: RunConfig[ShellDescription] = parse_config(
        config_path.read_text(), ShellDescription
    )

    # The structure, its load cases, and the three blocks built on it.
    structure = build_shell(config.structure)
    loads = build_load_cases(structure, config.load_cases)
    family = build_section_family(Steel355(), config.sizing.section_class)
    pipeline = build_pipeline(
        structure, family, config.form_finding, config.analysis, config.sizing
    )

    # The start: the initializer's densities, signed by the guard the file names.
    families = list_shell_families(config.structure)
    guarded = assign_signs(config.constraints, families, structure.num_edges)
    basis = pipeline.formfinder.basis
    initializer = config.form_finding.initializer
    started = initializer(structure, loads.formfinding, basis, guarded)
    constraints = build_design_constraints(config.constraints, started.guard)
    problem = DesignProblem(structure, pipeline, loads, constraints)
    d_start = config.analysis.diameter
    start = initialize_optimization_variables(problem, started.q, d_start)
    initial = evaluate_design(problem, start)

    # The descent: one reverse pass per gradient, whatever the constraint set.
    found = optimize_design(problem, start, config.optimization)
    optimized = evaluate_design(problem, found.variables)

    # What the run arrived at; the report, the record and the viewer read it.
    record = DesignRecord(problem, found, initial, optimized, families)
    report_design(record, config, TITLE)
    export_design(record, config, EXPORT)
    view_design(record, config)


if __name__ == "__main__":
    jnp.set_printoptions(precision=12)
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else CONFIG)
