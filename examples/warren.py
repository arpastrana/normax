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
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Int

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
from normax.form_finding import fit_densities
from normax.form_finding import held_plan_basis
from normax.loads import LoadCases
from normax.loads import build_load_cases
from normax.materials import Steel355
from normax.reporting import Report
from normax.reporting import report_descent
from normax.reporting import report_design
from normax.reporting import report_families
from normax.sections import build_section_family
from normax.structures import Structure
from normax.structures import build_warren_2d
from normax.symmetry import guard_signs
from normax.symmetry import lens_geometry
from normax.symmetry import member_spread
from normax.symmetry import signed_shift
from normax.viewer import view_designs

# The truss and the search, unless another file is named on the command line.
CONFIG = Path(__file__).with_name("warren.yaml")

REPO = Path(__file__).resolve().parent.parent
FIGURES = REPO / "figures"
DATA = REPO / "data"

COMPILATION_CACHE = REPO / ".jax_cache"
COMPILATION_CACHE.mkdir(exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(COMPILATION_CACHE))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)


class TrussConfig(NamedTuple):
    """
    The truss to build.

    Attributes
    ----------
    num_bays :
        Number of bottom-chord segments the span is divided into.
    span :
        Horizontal distance between the two supports.
    depth :
        Height of the top chord above the bottom chord, as drawn.
    """

    num_bays: int
    span: float
    depth: float


class SketchConfig(NamedTuple):
    """
    The lens the start is fitted to.

    Attributes
    ----------
    sag_lens :
        Depth the sketch hangs its bottom chord to at midspan.
    rise_lens :
        Height the sketch arches its top chord to at midspan.
    """

    sag_lens: float
    rise_lens: float


def build_truss(config: TrussConfig) -> Structure:
    """
    The truss as drawn.
    """
    return build_warren_2d(config.num_bays, config.span, config.depth)


def mirrored_nodes(config: TrussConfig) -> Int[np.ndarray, "nodes"]:
    """
    Mirror image of every node index about midspan, chord by chord.
    """
    bays = config.num_bays
    bottom = bays - np.arange(bays + 1)
    top = 2 * bays - np.arange(bays)

    return np.concatenate([bottom, top])


def member_families(config: TrussConfig) -> tuple[tuple[str, slice], ...]:
    """
    Name and member slice of every family, in the generator's order.
    """
    bays = config.num_bays
    families = (
        ("bottom chord", slice(0, bays)),
        ("top chord", slice(bays, 2 * bays - 1)),
        ("rising diagonals", slice(2 * bays - 1, 3 * bays - 1)),
        ("falling diagonals", slice(3 * bays - 1, 4 * bays - 1)),
    )

    return families


def signed_start(
    structure: Structure,
    loads: LoadCases,
    config: RunConfig[TrussConfig, SketchConfig],
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
        The run description, read for the sketch and the sign margin.

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
    lens = lens_geometry(structure, config.sketch.sag_lens, config.sketch.rise_lens)
    fit = fit_densities(structure, lens, loads.formfinding)

    signs = np.concatenate([np.ones(bays), -np.ones(bays - 1)])
    chords = np.arange(2 * bays - 1)
    guard = guard_signs(fit.q, signs, chords, config.subspace.margin_fraction)

    return signed_shift(fit.q, fit.self_stresses[:, 0], guard)


def main(config_path: Path) -> None:
    """
    Design the truss a file describes, and report what the descent bought.

    Parameters
    ----------
    config_path :
        File naming the truss and the settings a design of it is searched for
        under.
    """
    config: RunConfig[TrussConfig, SketchConfig] = parse_run(
        config_path.read_text(), TrussConfig, SketchConfig
    )
    report = Report()
    report.write_banner("Warren truss — one search to a design")

    # The structure, its load cases, and the three blocks built on it.
    structure = build_truss(config.structure)
    loads = build_load_cases(structure, config.load_cases)
    family = build_section_family(Steel355(), config.sizing.section_class)
    pipeline = build_pipeline(structure, family, config.analysis, config.sizing)

    # The subspace holding the plan, and the mirror folding densities and
    # diameters alike.
    mirror = mirrored_nodes(config.structure) if config.subspace.symmetric else None
    basis = held_plan_basis(structure, mirror, config.subspace.pivoted)
    spread = member_spread(structure, (mirror,))
    constraints = design_constraints(config.constraints, None)
    problem = DesignProblem(structure, pipeline, loads, basis, spread, constraints)

    # The start: the signed lens fit, and the diameters a frozen-seed analysis
    # asks of it.
    q_start = signed_start(structure, loads, config)
    start = initial_variables(problem, q_start, config.analysis.diameter)
    initial = read_design(problem, start)

    report.write_heading("Backends")
    entries = [
        ("analysis", config.analysis.backend),
        ("sizing", config.sizing.backend),
        ("coordinates", str(basis.width)),
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
    report_families(report, optimized, member_families(config.structure))
    saved = 1.0 - float(compute_mass(optimized)) / float(compute_mass(initial))
    report.write_entries([("saved", f"{100.0 * saved:.2f} %")])

    # The record, and the figures.
    DATA.mkdir(exist_ok=True)
    np.savez(
        DATA / "warren.npz",
        variables=found.variables,
        objectives=found.objectives,
        violations=found.violations,
    )
    FIGURES.mkdir(exist_ok=True)
    designs = {"start": initial, "answer": optimized}
    labels = case_labels(config.load_cases)
    drawn, descended = design_figures(structure, designs, labels, found)
    drawn.savefig(FIGURES / "warren_designs.png", dpi=200)
    descended.savefig(FIGURES / "warren_descent.png", dpi=200)
    written = [("figures", str(FIGURES)), ("data", str(DATA / "warren.npz"))]
    report.write_entries(written)

    if config.viewer:
        view_designs(structure, pipeline.analyzer, loads, designs, labels)


if __name__ == "__main__":
    jnp.set_printoptions(precision=12)
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else CONFIG)
