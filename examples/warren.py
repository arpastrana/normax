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
A Warren truss designed end to end, inside the funicular subspace of its plan.

Same three blocks as the arch, same descent, one difference: the force
densities move in the subspace that holds the drawn plan in horizontal
equilibrium, so every geometry the search reaches keeps the bays where they
were drawn and the coordinates need no bounds. The truss is once statically
indeterminate, so the funicular fit has a state of self-stress — the split
between hanging deck and arching top chord — and the start is shifted along it
until both chords carry their signs.

Run with `uv run python examples/warren.py [warren.yaml]`.
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
from normax.form_finding import LensShapeInitializer
from normax.form_finding import build_plan_basis
from normax.form_finding import fit_densities
from normax.loads import LoadCases
from normax.loads import build_load_cases
from normax.materials import Steel355
from normax.reporting import report_design
from normax.sections import build_section_family
from normax.structures import Structure
from normax.structures import TrussDescription
from normax.structures import build_warren_2d
from normax.symmetry import build_member_spread
from normax.symmetry import guard_signs
from normax.symmetry import shift_densities
from normax.symmetry import sketch_lens
from normax.tesseract import build_pipeline
from normax.viewer import view_design

# The truss and the search, unless another file is named on the command line.
CONFIG = Path(__file__).with_name("warren.yaml")

REPO = Path(__file__).resolve().parent.parent
TITLE = "Warren truss — one search to a design"
EXPORT = ExportTarget("warren", REPO / "data", REPO / "figures")


def build_truss(description: TrussDescription) -> Structure:
    """
    The truss as drawn.
    """
    return build_warren_2d(description.num_bays, description.span, description.depth)


def mirror_nodes(description: TrussDescription) -> Int[np.ndarray, "nodes"]:
    """
    Mirror image of every node index about midspan, chord by chord.
    """
    bays = description.num_bays
    bottom = bays - np.arange(bays + 1)
    top = 2 * bays - np.arange(bays)

    return np.concatenate([bottom, top])


def list_families(description: TrussDescription) -> tuple[tuple[str, slice], ...]:
    """
    Name and member slice of every family, in the generator's order.
    """
    bays = description.num_bays
    families = (
        ("bottom chord", slice(0, bays)),
        ("top chord", slice(bays, 2 * bays - 1)),
        ("rising diagonals", slice(2 * bays - 1, 3 * bays - 1)),
        ("falling diagonals", slice(3 * bays - 1, 4 * bays - 1)),
    )

    return families


def initialize_densities(
    structure: Structure,
    loads: LoadCases,
    config: RunConfig[TrussDescription, LensShapeInitializer],
) -> np.ndarray:
    """
    The lens fit, shifted along its self-stress until both chords are signed.

    Parameters
    ----------
    structure :
        The truss as drawn.
    loads :
        The load cases, the first of which the fit balances.
    config :
        The run config, read for the sketch and the sign margin.

    Returns
    -------
    q :
        Force density of every member at the start, the bottom chord in
        tension and the top chord in compression.

    Notes
    -----
    Fitted in the full member space and then read into the basis: the signed
    densities hold the plan, so they already live in its span.
    """
    bays = config.structure.num_bays
    lens = sketch_lens(structure, config.start.sag_lens, config.start.rise_lens)
    fit = fit_densities(structure, lens, loads.formfinding)

    signs = np.concatenate([np.ones(bays), -np.ones(bays - 1)])
    chords = np.arange(2 * bays - 1)
    guard = guard_signs(fit.q, signs, chords, config.constraints.sign_margin_fraction)

    return shift_densities(fit.q, fit.self_stresses[:, 0], guard)


def main(config_path: Path) -> None:
    """
    Design the truss a file describes, and report what the descent bought.

    Parameters
    ----------
    config_path :
        File naming the truss and the settings a design of it is searched for
        under.
    """
    config: RunConfig[TrussDescription, LensShapeInitializer] = parse_config(
        config_path.read_text(), TrussDescription, LensShapeInitializer
    )

    # The structure, its load cases, and the three blocks built on it.
    structure = build_truss(config.structure)
    loads = build_load_cases(structure, config.load_cases)
    family = build_section_family(Steel355(), config.sizing.section_class)
    pipeline = build_pipeline(structure, family, config.analysis, config.sizing)

    # The subspace holding the plan, and the mirror folding densities and
    # diameters alike.
    mirror = mirror_nodes(config.structure) if config.subspace.symmetric else None
    basis = build_plan_basis(structure, mirror, config.subspace.pivoted)
    spread = build_member_spread(structure, (mirror,))
    constraints = build_design_constraints(config.constraints, None)
    problem = DesignProblem(structure, pipeline, loads, basis, spread, constraints)

    # The start: the signed lens fit, and the diameters a frozen-seed analysis
    # asks of it.
    q_start = initialize_densities(structure, loads, config)
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
