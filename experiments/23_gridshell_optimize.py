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
The gridshell designed end to end, against two searches without a form finder.

The truss race of experiments 18 and 19, moved onto a shell. Same three
constrained searches over the same members, the same load cases, the same
frame analysis and the same EN 1993-1-1 check, the whole flow shared in
`design_routes`; this file is the gridshell's profile — its generator, its
mirror, its families, its start recipe, and the sign guard it runs under.

**The funicular subspace is thirteen wide against thirty-seven free heights.**
Holding the plan leaves a null space of the horizontal balance whose dimension
is a rank rather than a formula: 84 members less rank 71, the polar symmetry
costing three rows. Differentiating the form finder confirms every one of the
thirteen moves the shell, the spectrum falling as one dominant rise mode and
then degenerate pairs — the polar harmonics. Free heights is therefore a
strict superset, the Vierendeel situation rather than the Warren's, so a gap
in favour of the form finder cannot be reach and has to be landscape.

**The load is a pressure, and it is spread by tributary area.** Sharing a
total equally over the nodes of a polar grid overloads the crown and starves
the rim, the tributary areas running threefold across the rings, and the
funicular answer to that is a peakier cap than a uniform pressure asks for.
Stating the load as a pressure also states what the supports carry: the
boundary ring's tributary share goes straight to ground, so the structure is
loaded by less than the pressure times the whole plan, and the report prints
both numbers.

**The densities are held in compression, and the drawn cap already is.** Under
its own tributary pressure the generated cap fits an all-compression funicular
with no self-stress at all — a unique set of densities, every one of them a
strut — so the start needs no shift off the drawn geometry, unlike either
truss. The guard exists for the descent rather than the start: `q = B xi` is
linear, so one row per member holds every trial point inside the compression
cone, where the vertical stiffness is negative definite and a form-finding
solve cannot go singular.

**What the design does not inherit from the load.** The shell is folded about
one mirror plane whatever the loading does, a fabrication constraint rather
than a response — nine coordinates and forty-six diameters. The drift case is
centred on that same plane so it stays self-symmetric, which is what lets an
asymmetric case ride on a symmetric design without blinding the readout of
which case governs.

Run with `uv run --group pipeline --group viz python
experiments/23_gridshell_optimize.py [gridshell_optimize.yaml]`.
"""

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from design_routes import ChordSigns
from design_routes import RouteProblem
from design_routes import RouteProfile
from design_routes import StartPoint
from design_routes import TaskConfig
from design_routes import parse_shell
from design_routes import run_routes
from design_routes import shell_extent
from design_routes import shell_heights
from design_routes import shell_loads
from jaxtyping import Int

from normax.form_finding import fit_densities
from normax.structures import Structure
from normax.structures import build_gridshell_3d


def build_shell(config: TaskConfig) -> Structure:
    """
    The gridshell the run describes.

    Parameters
    ----------
    config :
        The run description, read for the rings, spokes, radius and rise.

    Returns
    -------
    structure :
        The drawn cap.
    """
    sketch = config.structure

    return build_gridshell_3d(
        sketch.num_rings, sketch.num_spokes, sketch.radius, sketch.rise
    )


def mirrored_nodes(config: TaskConfig) -> Int[np.ndarray, "nodes"]:
    """
    Mirror image of every node index about the plane through spoke zero.

    Parameters
    ----------
    config :
        The run description, read for the rings and the spokes.

    Returns
    -------
    nodes_mirrored :
        The node each node is carried onto, the apex onto itself.

    Notes
    -----
    The reflection sending spoke `k` to `-k` fixes spoke zero and, on an even
    spoke count, the spoke opposite it. It is an involution, which is what
    `folding_matrix` needs to read orbits off it, and the drift case is
    centred on the same plane so no load case is asymmetric about it.
    """
    sketch = config.structure
    spokes = np.arange(sketch.num_spokes)
    reflected = (-spokes) % sketch.num_spokes

    rings = [
        1 + ring * sketch.num_spokes + reflected for ring in range(sketch.num_rings)
    ]

    return np.concatenate([[0], np.concatenate(rings)])


def member_families(config: TaskConfig) -> tuple[tuple[str, slice], ...]:
    """
    Name and member slice of every family, in the generator's order.

    Parameters
    ----------
    config :
        The run description, read for the rings and the spokes.

    Returns
    -------
    families :
        The radial members, then the hoops that close every ring but the
        pinned one.
    """
    radials = config.structure.num_rings * config.structure.num_spokes

    families = (
        ("radial", slice(0, radials)),
        ("hoop", slice(radials, None)),
    )

    return families


def signed_start(problem: RouteProblem, config: TaskConfig) -> StartPoint:
    """
    The drawn cap's own funicular densities, read into the searched basis.

    Parameters
    ----------
    problem :
        The prepared shell.
    config :
        The run description, read for the sign margin the start must clear.

    Returns
    -------
    start :
        The fitted densities, their coordinates, the drawn geometry they
        reproduce, the projection gap and the balance gap.

    Raises
    ------
    ValueError
        If the drawn cap's funicular is not strictly compressive by the
        margin, which no shift here could repair — the generator, the
        pressure or the margin would have to change instead.

    Notes
    -----
    The simplest of the three start recipes, and the reason is measured
    rather than assumed: under its tributary pressure the cap fits with no
    state of self-stress, so its funicular densities are unique and there is
    nothing to shift along. They are also already strictly compressive, so
    the start sits inside the guard rather than being pushed there, and the
    end-to-end route leaves from the drawn geometry exactly.
    """
    structure = problem.structure
    drawn = np.asarray(structure.nodes)

    fit = fit_densities(structure, drawn, problem.loads.formfinding)
    q = np.asarray(fit.q)

    scale = float(np.median(np.abs(q)))
    margin = config.subspace.margin_fraction * scale
    if q.max() > -margin:
        raise ValueError(
            f"the drawn cap is not compressive by the {margin:.4f} margin: "
            f"worst density {q.max():.4f} on member {int(np.argmax(q))}"
        )

    finder = problem.pipeline.formfinder
    xi = finder.read_coordinates(q)
    rebuilt = np.asarray(finder.basis) @ xi
    projection = float(np.linalg.norm(rebuilt - q) / np.linalg.norm(q))

    return StartPoint(q, xi, drawn, projection, fit.gap)


def sign_guard(config: TaskConfig, start: StartPoint) -> ChordSigns | None:
    """
    The rows holding every density in compression through the descent.

    Parameters
    ----------
    config :
        The run description, supplying the sign margin. At a margin of zero
        or less the guard is off and the search may hang the shell.
    start :
        The fitted start, scaling the margin and the rows.

    Returns
    -------
    guard :
        Signs, members, margin and scale, for the end-to-end slack, or None
        where the run asks for no guard at all.

    Notes
    -----
    Every member is guarded rather than a named family: a shell has no chord
    whose sign carries the structure's chain, and the constraint asked of it
    is the design one — a compression-only funicular. The rows come free as
    constraints go, being exactly linear in the searched coordinates, and
    they buy a guarantee no truss guard could give: with every density
    negative the vertical stiffness is negative definite, so no trial point
    inside the feasible set can hand the analysis a singular form finder.

    **The switch exists because the constraint is expensive and binds.** Let
    go of it and the search does not find a better shell — it stops designing
    a shell at all, flattening the cap and hanging the members in tension,
    where EN 1993-1-1 applies no buckling reduction and the same load is
    carried by far less steel. Which is the clause doing its work, and worth
    pricing rather than hiding: the guarded answer is what a compression
    structure costs.
    """
    if config.subspace.margin_fraction <= 0.0:
        return None

    members = start.q.size
    signs = -np.ones(members)
    guarded = np.arange(members)
    scale = float(np.median(np.abs(start.q)))

    return ChordSigns(signs, guarded, config.subspace.margin_fraction * scale, scale)


GRIDSHELL_PROFILE = RouteProfile(
    banner="Gridshell — three routes to a design",
    prefix="23_gridshell",
    start_heading="The start, and the compression it already carries",
    parse_task=parse_shell,
    build_structure=build_shell,
    mirrored_nodes=mirrored_nodes,
    member_families=member_families,
    build_loads=shell_loads,
    height_limits=shell_heights,
    signed_start=signed_start,
    sign_guard=sign_guard,
    extent=shell_extent,
)


if __name__ == "__main__":
    jnp.set_printoptions(precision=12)
    described = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    run_routes(
        GRIDSHELL_PROFILE,
        described or Path(__file__).with_name("gridshell_optimize.yaml"),
    )
