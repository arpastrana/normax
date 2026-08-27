# SPDX-License-Identifier: Apache-2.0
"""
A Vierendeel truss designed end to end, on the truss where funicularity is scarce.

The Warren without its diagonals. A pin-jointed Vierendeel is a mechanism, so
the frame analysis models every member as a rigid-jointed beam and the panels
carry load through joint bending: no shape makes it momentless under the
asymmetric cases, and the check's interaction of axial force and bending governs
rather than the axial resistance alone. The held-plan subspace is nine wide and
is searched in member coordinates, and the chord signs are guarded through the
descent: a chord density crossing zero hands the form finder a singular
stiffness, so the guard keeps every trial point on the signed sheet.

Run with `uv run python examples/vierendeel.py [vierendeel.yaml]`.
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
from normax.structures import Structure
from normax.structures import TrussDescription
from normax.structures import build_vierendeel_2d
from normax.structures import list_vierendeel_families
from normax.tesseract import build_pipeline
from normax.visualization import view_design

# The truss and the search, unless another file is named on the command line.
CONFIG = Path(__file__).with_name("vierendeel.yaml")

REPO = Path(__file__).resolve().parent.parent
TITLE = "Vierendeel truss — one search to a design"
EXPORT = ExportTarget("vierendeel", REPO / "data", REPO / "figures")


def build_truss(description: TrussDescription) -> Structure:
    """
    The truss as drawn, four supports at the chord ends.
    """
    return build_vierendeel_2d(
        description.num_bays, description.span, description.depth
    )


def main(config_path: Path) -> None:
    """
    Design the truss a file describes, and report what the descent bought.

    Parameters
    ----------
    config_path :
        File naming the truss and the settings a design of it is searched for
        under.
    """
    config: RunConfig[TrussDescription] = parse_config(
        config_path.read_text(), TrussDescription
    )

    # The structure, its load cases, and the three blocks built on it.
    structure = build_truss(config.structure)
    loads = build_load_cases(structure, config.load_cases)
    family = build_section_family(Steel355(), config.sizing.section_class)
    pipeline = build_pipeline(
        structure, family, config.form_finding, config.analysis, config.sizing
    )

    # The start: the initializer's densities, signed by the guard the file names.
    families = list_vierendeel_families(config.structure)
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
