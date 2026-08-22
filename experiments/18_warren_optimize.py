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
The Warren truss designed end to end, against two searches without a form finder.

Three constrained searches over the same members, the same load cases, the
same analysis and the same code check, differing only in how — and whether —
the geometry moves. The end-to-end route moves the held-plan subspace
coordinates of experiment 16 together with the member diameters: the form
finder turns the coordinates into a geometry, the frame analysis into member
forces, the EN 1993-1-1 check into utilizations. The free-heights route drops
the form finder and hands the optimizer the node heights directly, driving
the same T2 and T3 alone. The sizing-only route holds the truss as drawn and
moves the diameters alone. All three run the same SLSQP under hard `U <= 1`
per member and load case, analytic Jacobians throughout, inside the rise
ceiling and above the sag floor. The flow lives in `design_routes` and is
shared with the Vierendeel of experiment 19; this file is the Warren's
profile — its generator, its mirror, its families, and its start recipe.

**On this truss the two shaped routes span the same geometries.** Experiment
16 counted every held-plan geometry funicular-reachable — sixteen independent
edges against fifteen free heights plus one self-stress — so unlike the arch
of experiment 15, where the heights were a strict superset, here the density
route gives nothing away. Whether the two parametrizations also *land* on the
same design is what the run measures: any gap between them is landscape and
conditioning, never reach.

**The truss is once statically indeterminate, and it shows twice.** The
thrust-vs-tie split of the funicular fit is not what the elastic frame
carries: internal forces depend on the stiffness distribution, so the
funicular `q L` and the analyzed axial force disagree where the arch is
determinate and they agree to machine precision. And the frozen-seed envelope
that seeds each search is measurably infeasible once the frame is re-analyzed
at its own sections. Both are measured and reported; the simultaneous
formulation holds the coupling inside the gradient, so neither survives to
the answers.

**The start is fitted free, then projected** — the reverse of the Vierendeel
recipe, and a measured decision: on the Warren the basis-restricted fit's
self-stress direction rides at the least-squares rank cutoff and is
unreliably detected, while the free fit's is orders below it, and the signed
densities hold the plan so the projection costs nothing but a reported gap.

Run with `uv run --group pipeline --group viz python experiments/18_warren_optimize.py
[warren_optimize.yaml]`.
"""

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from design_routes import RouteProblem
from design_routes import RouteProfile
from design_routes import StartPoint
from design_routes import TaskConfig
from design_routes import lens_geometry
from design_routes import parse_truss
from design_routes import run_routes
from design_routes import signed_shift
from design_routes import truss_extent
from design_routes import truss_heights
from design_routes import truss_loads
from jaxtyping import Int

from normax.form_finding import fit_densities
from normax.structures import Structure
from normax.structures import build_warren_2d


def mirrored_nodes(config: TaskConfig) -> Int[np.ndarray, "nodes"]:
    """
    Mirror image of every node index about midspan, chord by chord.
    """
    num_bays = config.structure.num_bays

    bottom = num_bays - np.arange(num_bays + 1)
    top = 2 * num_bays - np.arange(num_bays)

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
        The two chords and the two diagonal directions.
    """
    num_bays = config.structure.num_bays

    families = (
        ("bottom chord", slice(0, num_bays)),
        ("top chord", slice(num_bays, 2 * num_bays - 1)),
        ("rising diagonals", slice(2 * num_bays - 1, 3 * num_bays - 1)),
        ("falling diagonals", slice(3 * num_bays - 1, 4 * num_bays - 1)),
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

    return build_warren_2d(sketch.num_bays, sketch.span, sketch.depth)


def signed_start(problem: RouteProblem, config: TaskConfig) -> StartPoint:
    """
    The lens fit of experiment 16, signed and written in the searched basis.

    Parameters
    ----------
    problem :
        The prepared truss.
    config :
        The run description, supplying the sketch and the sign margin.

    Returns
    -------
    start :
        The signed densities, their coordinates, and the projection gap.

    Notes
    -----
    Fitted in the full edge space and then projected, rather than fitted in
    the basis directly: the restricted fit's self-stress direction rides at
    the least-squares rank cutoff and is unreliably detected, while the free
    fit's is orders below it. The projection costs nothing measurable — the
    signed densities hold the plan, so they already live in the basis's span,
    and the gap is reported rather than assumed.
    """
    bays = config.structure.num_bays
    span = config.structure.span

    sketch = config.sketch
    lens = lens_geometry(
        problem.structure, span, bays, sketch.sag_lens, sketch.rise_lens
    )

    fit = fit_densities(problem.structure, lens, problem.loads.formfinding)
    mode = fit.self_stresses[:, 0]

    signs = np.concatenate([np.ones(bays), -np.ones(bays - 1)])
    chords = np.arange(2 * bays - 1)
    margin = config.subspace.margin_fraction * float(np.median(np.abs(fit.q[:bays])))

    shifted = signed_shift(fit.q, mode, signs, chords, margin)

    finder = problem.pipeline.formfinder
    xi = finder.read_coordinates(shifted.q)
    rebuilt = np.asarray(finder.basis) @ xi
    projection = float(np.linalg.norm(rebuilt - shifted.q) / np.linalg.norm(shifted.q))

    return StartPoint(shifted.q, xi, lens, projection, fit.gap)


WARREN_PROFILE = RouteProfile(
    banner="Warren truss — three routes to a design",
    prefix="18_warren",
    start_heading="The start, and what the indeterminacy does to it",
    parse_task=parse_truss,
    build_structure=build_truss,
    mirrored_nodes=mirrored_nodes,
    sections_rotated=None,
    heights_rotated=None,
    member_families=member_families,
    build_loads=truss_loads,
    height_limits=truss_heights,
    signed_start=signed_start,
    sign_guard=None,
    extent=truss_extent,
)


if __name__ == "__main__":
    jnp.set_printoptions(precision=12)
    described = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    run_routes(
        WARREN_PROFILE,
        described or Path(__file__).with_name("warren_optimize.yaml"),
    )
