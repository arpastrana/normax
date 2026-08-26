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
`normax.searches`; this file is the gridshell's profile — its generator, its
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
examples/gridshell.py [gridshell.yaml]`.
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
from normax.searches import parse_shell
from normax.searches import run_searches
from normax.searches import shell_extent
from normax.searches import shell_heights
from normax.searches import shell_loads
from normax.structures import Structure
from normax.structures import build_gridshell_3d
from normax.visualization.viewer import view_answers


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
        sketch.num_rings,
        sketch.num_spokes,
        sketch.radius,
        sketch.rise,
        sketch.oculus,
        sketch.braced,
    )


def radial_count(config: TaskConfig) -> int:
    """
    How many radial members the shell has, the hoops following them.

    Parameters
    ----------
    config :
        The run description, read for the rings, the spokes and the crown.

    Returns
    -------
    radials :
        Members before the first hoop in the generator's order. An open crown
        costs one ring of them, the spoke that reached the apex.
    """
    sketch = config.structure
    reaching = sketch.num_rings - 1 if sketch.oculus else sketch.num_rings

    return reaching * sketch.num_spokes


def panel_count(config: TaskConfig) -> int:
    """
    How many panels lie between consecutive rings.

    Parameters
    ----------
    config :
        The run description, read for the rings and the spokes.

    Returns
    -------
    panels :
        One per spoke per gap between rings. It counts the hoops of every
        hooped ring, and each of the two diagonal families of a braced cap.
    """
    sketch = config.structure

    return (sketch.num_rings - 1) * sketch.num_spokes


def guarded_members(config: TaskConfig) -> Int[np.ndarray, "members"]:
    """
    Members the compression guard holds, in the generator's order.

    Parameters
    ----------
    config :
        The run description, read for the families and the guard's reach.

    Returns
    -------
    guarded :
        The radials alone, or every member. The radials come first in the
        generator's order, so either reach is a prefix.

    Notes
    -----
    `guard_hoops` reaches past the hoops to the diagonals of a braced cap. The
    switch names the decision it started as — meridian compression guarded, the
    rest of the grid free — and a diagonal belongs on the free side of that
    line for the same reason a hoop does.
    """
    sketch = config.structure
    radials = radial_count(config)
    panels = panel_count(config)
    diagonals = 2 * panels if sketch.braced else 0
    covered = radials + panels + diagonals if sketch.guard_hoops else radials

    return np.arange(covered)


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
        The node each node is carried onto, the apex onto itself where the
        crown is closed.

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
    offset = 0 if sketch.oculus else 1

    rings = [
        offset + ring * sketch.num_spokes + reflected
        for ring in range(sketch.num_rings)
    ]
    ringed = np.concatenate(rings)
    if sketch.oculus:
        return ringed

    return np.concatenate([[0], ringed])


def spoke_rotation(config: TaskConfig) -> Int[np.ndarray, "nodes"]:
    """
    Node image under a rotation of one spoke.

    Parameters
    ----------
    config :
        The run description, read for the rings, the spokes and the crown.

    Returns
    -------
    nodes_rotated :
        The node each node turns onto, the apex onto itself where the crown is
        closed.

    Notes
    -----
    One spoke is the whole generator. Composed with the mirror it generates
    the full dihedral group of the grid, so union-find over the two lands the
    complete polar orbits — one per ring per family — rather than the pairs a
    single reflection leaves.
    """
    sketch = config.structure
    spokes = np.arange(sketch.num_spokes)
    turned = (spokes + 1) % sketch.num_spokes
    offset = 0 if sketch.oculus else 1

    rings = [
        offset + ring * sketch.num_spokes + turned for ring in range(sketch.num_rings)
    ]
    ringed = np.concatenate(rings)
    if sketch.oculus:
        return ringed

    return np.concatenate([[0], ringed])


def sections_rotated(config: TaskConfig) -> Int[np.ndarray, "nodes"] | None:
    """
    The rotation the diameters are folded by, where the run asks for one.

    Parameters
    ----------
    config :
        The run description, read for the sections' polar switch.

    Returns
    -------
    nodes_rotated :
        The one-spoke rotation, or None to leave the sections folded by the
        mirror alone.

    Notes
    -----
    A fabrication constraint and nothing else: it leaves one section per ring
    per family, whatever the loading asks for spoke by spoke. The drift cases
    stay one-sided, so what this buys in buildability it pays for in mass.
    """
    if not config.structure.polar_diameters:
        return None

    return spoke_rotation(config)


def heights_rotated(config: TaskConfig) -> Int[np.ndarray, "nodes"] | None:
    """
    The rotation the free heights are folded by, where the run asks for one.

    Parameters
    ----------
    config :
        The run description, read for the heights' polar switch.

    Returns
    -------
    nodes_rotated :
        The one-spoke rotation, or None to leave the heights folded by the
        mirror alone.

    Notes
    -----
    **This one changes what the comparison means, and the switch exists to
    make that explicit.** Folded by the mirror alone the free heights are a
    strict superset of the shapes the form finder reaches, so a gap between
    the searches is a statement about the landscape. Folded polar they are one
    height per ring — a space of the same dimension as the density basis that
    neither contains it nor sits inside it, because a funicular shape need not
    be axisymmetric even when its plan is.
    """
    if not config.structure.polar_heights:
        return None

    return spoke_rotation(config)


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
        The radial members, the hoops that close every ring but the pinned
        one, and on a braced cap the panel diagonals, both directions read as
        one family.
    """
    radials = radial_count(config)
    panels = panel_count(config)
    hoops = radials + panels

    families = [
        ("radial", slice(0, radials)),
        ("hoop", slice(radials, hoops)),
    ]
    if config.structure.braced:
        families.append(("diagonal", slice(hoops, None)))

    return tuple(families)


def signed_start(problem: DesignProblem, config: TaskConfig) -> StartPoint:
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
        If a run that guards the signs starts from a cap whose guarded members
        are not strictly compressive by the margin, which no shift here could
        repair — the generator, the pressure, the guard's reach or the margin
        would have to change instead. A run with the guard off is not checked,
        having nothing for the check to protect.

    Notes
    -----
    The simplest of the three start recipes, and the reason is measured
    rather than assumed: under its tributary pressure the closed quad cap fits
    with no state of self-stress, so its funicular densities are unique and
    there is nothing to shift along. They are also already strictly
    compressive, so the start sits inside the guard rather than being pushed
    there, and the end-to-end search leaves from the drawn geometry exactly.

    **A braced cap is the opposite case and needs the guard off.** Triangulating
    the panels buys states of self-stress by the dozen, so the fit is no longer
    unique and the least-squares representative it returns carries tension in
    the radials even though all-compression members of the same family exist.
    Picking one of those would take a program over the self-stress space rather
    than a shift along a single mode, so a braced run drops the sign guard
    instead and lets the descent settle the signs.
    """
    structure = problem.structure
    drawn = np.asarray(structure.nodes)

    fit = fit_densities(structure, drawn, problem.loads.formfinding)
    q = np.asarray(fit.q)

    guarded = guarded_members(config)
    held = q[guarded]
    scale = float(np.median(np.abs(held)))
    margin = config.subspace.margin_fraction * scale
    if margin > 0.0 and held.max() > -margin:
        worst = int(guarded[np.argmax(held)])
        raise ValueError(
            f"the drawn cap's guarded members are not compressive by the "
            f"{margin:.4f} margin: worst density {held.max():.4f} "
            f"on member {worst}"
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
    **Which members are held is a design decision, not a numerical one.** At
    the default reach the radials are guarded and the hoops are not, which is
    the dome's own division of labour: meridian compression is what makes a
    cap carry as an arch, while hoop tension in the lower rings is the classic
    membrane response to it rather than a defect to design away. Setting
    `guard_hoops` extends the guard over every member and asks instead for a
    wholly compressive net, which a dome can give but has to buy — it is the
    lower hoops that would otherwise have taken the ring tension, and holding
    them in compression pushes that work back onto the shape.

    The rows are exactly linear in the searched coordinates, `q` being `B xi`,
    so the quadratic subproblem holds every trial point on the signed sheet
    rather than merely the answer. What it no longer buys is a definite
    vertical stiffness — free hoops may cross zero and make it indefinite —
    so `RECOIL_SLACK` in the shared descent is what catches an unfactorizable
    trial frame.

    **The switch exists because the sign is a design decision.** Let the
    radials go and the search stops designing a shell at all: it flattens the
    cap and hangs the members, where EN 1993-1-1 applies no buckling
    reduction and the same load is carried by far less steel. That answer
    then rides whatever sag floor the run happens to state rather than any
    optimum, which is why it is a diagnostic here and never a baseline.
    """
    if config.subspace.margin_fraction <= 0.0:
        return None

    guarded = guarded_members(config)
    signs = -np.ones(guarded.size)
    scale = float(np.median(np.abs(start.q[guarded])))

    return ChordSigns(signs, guarded, config.subspace.margin_fraction * scale, scale)


GRIDSHELL_PROFILE = StructureProfile(
    banner="Gridshell — three searches to a design",
    prefix="23_gridshell",
    start_heading="The start, and the compression it already carries",
    parse_task=parse_shell,
    build_structure=build_shell,
    mirrored_nodes=mirrored_nodes,
    sections_rotated=sections_rotated,
    heights_rotated=heights_rotated,
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
    requested = run_searches(
        GRIDSHELL_PROFILE,
        described or Path(__file__).with_name("gridshell.yaml"),
    )

    # A window blocks until it closes, so the run reports first and draws last.
    if requested is not None:
        view_answers(requested)
