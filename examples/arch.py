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

Run with `uv run python examples/arch.py [arch.yaml]`.
"""

import sys
from pathlib import Path

import jax.numpy as jnp

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
from normax.loads import build_load_cases
from normax.materials import Steel355
from normax.reporting import report_design
from normax.sections import build_section_family
from normax.structures import ArchDescription
from normax.structures import Structure
from normax.structures import build_arch_2d
from normax.tesseract import build_pipeline
from normax.viewer import view_design

# The arch and the search, unless another file is named on the command line.
CONFIG = Path(__file__).with_name("arch.yaml")

REPO = Path(__file__).resolve().parent.parent
TITLE = "Arch — one search to a design"
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


def main(config_path: Path) -> None:
    """
    Design the arch a file describes, and report what the descent bought.

    Parameters
    ----------
    config_path :
        File naming the arch and the settings a design of it is searched for
        under.
    """
    config: RunConfig[ArchDescription] = parse_config(
        config_path.read_text(), ArchDescription
    )

    # The structure, its load cases, and the three blocks built on it. The
    # grade is named once, and both backends draw tubes from the same family.
    structure = build_arch(config.structure)
    loads = build_load_cases(structure, config.load_cases)
    family = build_section_family(Steel355(), config.sizing.section_class)
    pipeline = build_pipeline(
        structure, family, config.form_finding, config.analysis, config.sizing
    )

    # The start: one force density in every member, no guard, no folding.
    started = config.form_finding.initializer(structure, loads.formfinding, None, None)
    constraints = build_design_constraints(config.constraints, None)
    problem = DesignProblem(structure, pipeline, loads, constraints)
    d_start = config.analysis.diameter
    start = initialize_optimization_variables(problem, started.q, d_start)
    initial = evaluate_design(problem, start)

    # The descent: one reverse pass per gradient, whatever the constraint set.
    found = optimize_design(problem, start, config.optimization)
    optimized = evaluate_design(problem, found.variables)

    # What the run arrived at; the report, the record and the viewer read it.
    record = DesignRecord(problem, found, initial, optimized, ())
    report_design(record, config, TITLE)
    export_design(record, config, EXPORT)
    view_design(record, config)


if __name__ == "__main__":
    jnp.set_printoptions(precision=12)
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else CONFIG)
