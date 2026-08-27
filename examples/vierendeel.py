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
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jaxtyping import Int

from normax.config import RunConfig
from normax.config import parse_config
from normax.design import DesignProblem
from normax.design import DesignRecord
from normax.design import build_design_constraints
from normax.design import initialize_optimization_variables
from normax.design import optimize_design
from normax.design import read_design
from normax.exporting import ExportTarget
from normax.exporting import export_design
from normax.form_finding import PlanBasis
from normax.form_finding import build_plan_basis
from normax.form_finding import fit_densities
from normax.loads import LoadCases
from normax.loads import build_load_cases
from normax.materials import Steel355
from normax.reporting import report_design
from normax.sections import build_section_family
from normax.structures import Structure
from normax.structures import build_vierendeel_2d
from normax.symmetry import SignGuard
from normax.symmetry import build_member_spread
from normax.symmetry import guard_signs
from normax.symmetry import shift_densities
from normax.symmetry import sketch_lens
from normax.tesseract import build_pipeline
from normax.viewer import view_design

# The truss and the search, unless another file is named on the command line.
CONFIG = Path(__file__).with_name("vierendeel.yaml")

REPO = Path(__file__).resolve().parent.parent
TITLE = "Vierendeel truss — one search to a design"
EXPORT = ExportTarget("vierendeel", REPO / "data", REPO / "figures")


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
    The truss as drawn, four supports at the chord ends.
    """
    return build_vierendeel_2d(config.num_bays, config.span, config.depth)


def mirror_nodes(config: TrussConfig) -> Int[np.ndarray, "nodes"]:
    """
    Mirror image of every node index about midspan, chord by chord.
    """
    bays = config.num_bays
    bottom = bays - np.arange(bays + 1)
    top = 2 * bays + 1 - np.arange(bays + 1)

    return np.concatenate([bottom, top])


def list_families(config: TrussConfig) -> tuple[tuple[str, slice], ...]:
    """
    Name and member slice of every family, in the generator's order.
    """
    bays = config.num_bays
    families = (
        ("bottom chord", slice(0, bays)),
        ("top chord", slice(bays, 2 * bays)),
        ("verticals", slice(2 * bays, None)),
    )

    return families


def sign_chords(config: TrussConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    The sign each chord member must carry, and which members the chords are.
    """
    bays = config.num_bays
    signs = np.concatenate([np.ones(bays), -np.ones(bays)])
    chords = np.arange(2 * bays)

    return signs, chords


def initialize_densities(
    structure: Structure,
    basis: PlanBasis,
    loads: LoadCases,
    config: RunConfig[TrussConfig, SketchConfig],
) -> tuple[np.ndarray, SignGuard]:
    """
    The lens fitted inside the basis, signed along the load-path split.

    Parameters
    ----------
    structure :
        The truss as drawn.
    basis :
        The held-plan subspace the fit is restricted to.
    loads :
        The load cases, the first of which the fit balances.
    config :
        The run description, read for the sketch and the sign margin.

    Returns
    -------
    start :
        The signed densities, and the guard that keeps them signed.

    Notes
    -----
    Fitted inside the basis rather than freely: offered a sketch off the
    funicular manifold, the free least squares abandons the top chord and
    returns a singular vertical stiffness. The restricted fit keeps plan
    balance exact, and its one self-stress is the split between hanging deck
    and arching top chord.
    """
    lens = sketch_lens(structure, config.sketch.sag_lens, config.sketch.rise_lens)
    fit = fit_densities(structure, lens, loads.formfinding, basis)

    signs, chords = sign_chords(config.structure)
    guard = guard_signs(fit.q, signs, chords, config.subspace.margin_fraction)
    q = shift_densities(fit.q, fit.self_stresses[:, 0], guard)

    return q, guard_signs(q, signs, chords, config.subspace.margin_fraction)


def main(config_path: Path) -> None:
    """
    Design the truss a file describes, and report what the descent bought.

    Parameters
    ----------
    config_path :
        File naming the truss and the settings a design of it is searched for
        under.
    """
    config: RunConfig[TrussConfig, SketchConfig] = parse_config(
        config_path.read_text(), TrussConfig, SketchConfig
    )

    # The structure, its load cases, and the three blocks built on it.
    structure = build_truss(config.structure)
    loads = build_load_cases(structure, config.load_cases)
    family = build_section_family(Steel355(), config.sizing.section_class)
    pipeline = build_pipeline(structure, family, config.analysis, config.sizing)

    # The subspace holding the plan, in member coordinates, and the mirror
    # folding densities and diameters alike.
    mirror = mirror_nodes(config.structure) if config.subspace.symmetric else None
    basis = build_plan_basis(structure, mirror, config.subspace.pivoted)
    spread = build_member_spread(structure, (mirror,))

    # The start, and the guard the descent runs under.
    q_start, guard = initialize_densities(structure, basis, loads, config)
    guarded = guard if config.subspace.margin_fraction > 0.0 else None
    constraints = build_design_constraints(config.constraints, guarded)
    problem = DesignProblem(structure, pipeline, loads, basis, spread, constraints)
    d_start = config.analysis.diameter
    start = initialize_optimization_variables(problem, q_start, d_start)
    initial = read_design(problem, start)

    # The descent: one reverse pass per gradient, whatever the constraint set.
    found = optimize_design(problem, start, config.augmented)
    optimized = read_design(problem, found.variables)

    # What the run arrived at; the report, the record and the viewer read it.
    families = list_families(config.structure)
    record = DesignRecord(problem, found, initial, optimized, families)
    report_design(record, config, TITLE)
    export_design(record, config, EXPORT)
    view_design(record, config)


if __name__ == "__main__":
    jnp.set_printoptions(precision=12)
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else CONFIG)
