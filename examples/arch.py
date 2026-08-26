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
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

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
from normax.loads import build_load_cases
from normax.materials import Steel355
from normax.reporting import Report
from normax.reporting import report_descent
from normax.reporting import report_design
from normax.sections import build_section_family
from normax.structures import Structure
from normax.structures import build_arch_2d
from normax.viewer import view_designs

# The arch and the search, unless another file is named on the command line.
CONFIG = Path(__file__).with_name("arch.yaml")

REPO = Path(__file__).resolve().parent.parent
FIGURES = REPO / "figures"
DATA = REPO / "data"

COMPILATION_CACHE = REPO / ".jax_cache"
COMPILATION_CACHE.mkdir(exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(COMPILATION_CACHE))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)


class ArchConfig(NamedTuple):
    """
    The arch to build.

    Attributes
    ----------
    num_edges :
        Number of members the arch is discretized into.
    span :
        Horizontal distance between the two supports.
    rise :
        Height of the parabola the starting geometry rises along.
    """

    num_edges: int
    span: float
    rise: float


class SketchConfig(NamedTuple):
    """
    Where the search starts.

    Attributes
    ----------
    force_density :
        Force density every member starts at. Negative in compression.
    """

    force_density: float


def build_arch(config: ArchConfig) -> Structure:
    """
    The arch, as topology and a starting geometry.

    Parameters
    ----------
    config :
        The arch to build.

    Returns
    -------
    structure :
        A funicular arch between two pinned supports.
    """
    return build_arch_2d(config.num_edges, config.span, config.rise)


def main(config_path: Path) -> None:
    """
    Design the arch a file describes, and report what the descent bought.

    Parameters
    ----------
    config_path :
        File naming the arch and the settings a design of it is searched for
        under.
    """
    config: RunConfig[ArchConfig, SketchConfig] = parse_run(
        config_path.read_text(), ArchConfig, SketchConfig
    )
    report = Report()
    report.write_banner("Arch — one search to a design")

    # The structure, its load cases, and the three blocks built on it. The
    # grade is named once, and both backends draw tubes from the same family.
    structure = build_arch(config.structure)
    loads = build_load_cases(structure, config.load_cases)
    family = build_section_family(Steel355(), config.sizing.section_class)
    pipeline = build_pipeline(structure, family, config.analysis, config.sizing)

    # An arch is a chain: every force density is its own coordinate, boxed.
    constraints = design_constraints(config.constraints, None)
    problem = DesignProblem(structure, pipeline, loads, None, None, constraints)

    # The start: a uniform force density, and the diameters a frozen-seed
    # analysis asks of it.
    q_start = np.full(structure.num_edges, config.sketch.force_density)
    start = initial_variables(problem, q_start, config.analysis.diameter)
    initial = read_design(problem, start)

    report.write_heading("Backends")
    entries = [
        ("analysis", config.analysis.backend),
        ("sizing", config.sizing.backend),
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
    saved = 1.0 - float(compute_mass(optimized)) / float(compute_mass(initial))
    report.write_entries([("saved", f"{100.0 * saved:.2f} %")])

    # The record, and the figures.
    DATA.mkdir(exist_ok=True)
    np.savez(
        DATA / "arch.npz",
        variables=found.variables,
        objectives=found.objectives,
        violations=found.violations,
    )
    FIGURES.mkdir(exist_ok=True)
    designs = {"start": initial, "answer": optimized}
    labels = case_labels(config.load_cases)
    drawn, descended = design_figures(structure, designs, labels, found)
    drawn.savefig(FIGURES / "arch_designs.png", dpi=200)
    descended.savefig(FIGURES / "arch_descent.png", dpi=200)
    written = [("figures", str(FIGURES)), ("data", str(DATA / "arch.npz"))]
    report.write_entries(written)

    if config.viewer:
        view_designs(structure, pipeline.analyzer, loads, designs, labels)


if __name__ == "__main__":
    jnp.set_printoptions(precision=12)
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else CONFIG)
