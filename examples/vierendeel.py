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
The Vierendeel truss designed end to end, against two searches without a form finder.

The Warren race of experiment 18, rerun on the truss where funicularity is
scarce. Same three constrained searches, same analysis, same EN 1993-1-1
check, same rise ceiling and sag floor, the whole flow shared in
`normax.searches`; this file is the Vierendeel's profile — its generator, its
mirror, its families, its start recipe, and the one guard it alone needs.

**On this truss the two shaped searches do not span the same geometries.** The
Warren's counting made every held-plan geometry funicular-reachable, so any
gap between its searches was landscape alone. Remove the diagonals and the
counting flips: experiment 17 measured nine independent edges — two uniform
chord families and seven free verticals — against fourteen free heights, one
of the nine blind. Funicular shapes are a strict submanifold of what the
free-heights search can draw, so here the comparison prices the funicular
parametrization itself.

**Rigid joints put end moments everywhere.** A pin-jointed Vierendeel is a
mechanism, so T2 models every member as a rigid-jointed beam and the panels
carry load through joint bending — the interaction rows 6.61 and 6.62 govern
rather than the axial resistance alone. The funicular start still zeroes the
shaping case's moments as well as any geometry can, but no shape makes a
Vierendeel momentless under the asymmetric cases.

**Two degeneracies are guarded, one by this profile.** The start is fitted
inside the held-plan basis — offered a sketch, the free least squares
abandons the top chord and hands back a singular vertical stiffness, the
trap experiment 17 ran on purpose — and the chord-sign rows keep the descent
on the signed sheet of the manifold, where both chords carry their signs and
a funicular geometry exists; the optimizer rides that margin at the answer.
The member-length floor, which walls off the collapse of a vertical, is the
shared machinery's business and holds every vertical at its drawn length.

Run with `uv run --group pipeline --group viz python
examples/vierendeel.py [vierendeel.yaml]`.
"""

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jaxtyping import Int

from normax.form_finding import fit_densities
from normax.searches import ChordSigns
from normax.searches import DesignProblem
from normax.searches import StartPoint
from normax.searches import StructureProfile
from normax.searches import TaskConfig
from normax.searches import lens_geometry
from normax.searches import parse_truss
from normax.searches import run_searches
from normax.searches import signed_shift
from normax.searches import truss_extent
from normax.searches import truss_heights
from normax.searches import truss_loads
from normax.structures import Structure
from normax.structures import build_vierendeel_2d
from normax.visualization.viewer import view_answers


def mirrored_nodes(config: TaskConfig) -> Int[np.ndarray, "nodes"]:
    """
    Mirror image of every node index about midspan, chord by chord.
    """
    num_bays = config.structure.num_bays

    bottom = num_bays - np.arange(num_bays + 1)
    top = 2 * num_bays + 1 - np.arange(num_bays + 1)

    return np.concatenate([bottom, top])


def member_families(config: TaskConfig) -> tuple[tuple[str, slice], ...]:
    """
    Name and member slice of every family, in the generator's order.

    Parameters
    ----------
    config :
        The run description, read for the bay count.

    Returns
    -------
    families :
        The two chords and the verticals — no diagonals on a Vierendeel.
    """
    num_bays = config.structure.num_bays

    families = (
        ("bottom chord", slice(0, num_bays)),
        ("top chord", slice(num_bays, 2 * num_bays)),
        ("verticals", slice(2 * num_bays, None)),
    )

    return families


def build_truss(config: TaskConfig) -> Structure:
    """
    The truss the run describes.

    Parameters
    ----------
    config :
        The run description, read for the bays, the span and the depth.

    Returns
    -------
    structure :
        The drawn truss.
    """
    sketch = config.structure

    return build_vierendeel_2d(sketch.num_bays, sketch.span, sketch.depth)


def signed_start(problem: DesignProblem, config: TaskConfig) -> StartPoint:
    """
    The lens fitted inside the searched basis, signed along the load-path split.

    Parameters
    ----------
    problem :
        The prepared truss.
    config :
        The run description, supplying the sketch and the sign margin.

    Returns
    -------
    start :
        The signed densities, their coordinates, the projection gap, and the
        balance gap the restricted fit left.

    Notes
    -----
    Fitted inside the held-plan basis — the reverse of the Warren recipe, and
    for the reason experiment 17 measured: offered a sketch off the funicular
    manifold, the free least squares abandons the top chord, reports a
    deceptively small gap, and returns a singular vertical stiffness. The
    restricted fit keeps plan balance exact by construction, and its one
    self-stress is the load-path split between hanging deck and arching top
    chord, which the sign shift then moves along until both chords carry
    their signs. The verticals stay free: a hanger in the lens is a post in
    another shape.
    """
    bays = config.structure.num_bays
    span = config.structure.span

    sketch = config.sketch
    lens = lens_geometry(
        problem.structure, span, bays, sketch.sag_lens, sketch.rise_lens
    )

    finder = problem.pipeline.formfinder
    basis = np.asarray(finder.basis)
    fit = fit_densities(problem.structure, lens, problem.loads.formfinding, basis)
    mode = fit.self_stresses[:, 0]

    signs = np.concatenate([np.ones(bays), -np.ones(bays)])
    chords = np.arange(2 * bays)
    margin_fraction = config.subspace.margin_fraction
    margin = margin_fraction * float(np.median(np.abs(fit.q[chords])))

    shifted = signed_shift(fit.q, mode, signs, chords, margin)

    xi = finder.read_coordinates(shifted.q)
    rebuilt = basis @ xi
    projection = float(np.linalg.norm(rebuilt - shifted.q) / np.linalg.norm(shifted.q))

    return StartPoint(shifted.q, xi, lens, projection, fit.gap)


def chord_guard(config: TaskConfig, start: StartPoint) -> ChordSigns:
    """
    The chord-sign rows keeping the descent off the chord-off degeneracy.

    Parameters
    ----------
    config :
        The run description, supplying the bays and the sign margin.
    start :
        The signed lens fit, scaling the margin and the rows.

    Returns
    -------
    guard :
        Signs, chords, margin and scale, for the end-to-end slack.
    """
    bays = config.structure.num_bays
    signs = np.concatenate([np.ones(bays), -np.ones(bays)])
    chords = np.arange(2 * bays)
    scale = float(np.median(np.abs(start.q[chords])))

    return ChordSigns(signs, chords, config.subspace.margin_fraction * scale, scale)


VIERENDEEL_PROFILE = StructureProfile(
    banner="Vierendeel truss — three searches to a design",
    prefix="19_vierendeel",
    start_heading="The start, and what the scarcity does to it",
    parse_task=parse_truss,
    build_structure=build_truss,
    mirrored_nodes=mirrored_nodes,
    sections_rotated=None,
    heights_rotated=None,
    member_families=member_families,
    build_loads=truss_loads,
    height_limits=truss_heights,
    signed_start=signed_start,
    sign_guard=chord_guard,
    extent=truss_extent,
)


if __name__ == "__main__":
    jnp.set_printoptions(precision=12)
    described = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    requested = run_searches(
        VIERENDEEL_PROFILE,
        described or Path(__file__).with_name("vierendeel.yaml"),
    )

    # A window blocks until it closes, so the run reports first and draws last.
    if requested is not None:
        view_answers(requested)
