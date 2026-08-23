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
The three constrained routes to a truss design, shared across experiments.

Machinery for racing three searches over the same members, load cases, frame
analysis and code check, differing only in how — and whether — the geometry
moves: end to end over held-plan basis coordinates and diameters through the
whole pipeline, free heights over node heights and diameters through the
analysis and the check alone, and sizing only over the diameters at the drawn
geometry. All three run the same SLSQP under hard `U <= 1` per member and
load case, analytic Jacobians throughout, restarted from their own answer
until a round no longer moves.

The experiments own what differs between trusses: the generator, the node
mirror, the member families, and how the starting densities are fitted and
signed. Everything here is topology-blind — it reads the truss through a
`RouteProblem` and the run description through a `TaskConfig`.
"""

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import vix
import yaml
from ec3x.material import Steel
from ec3x.resistance import SHEAR_THRESHOLD
from ec3x.resistance import area_shear
from ec3x.resistance import utilization_shear
from jax_fdm.equilibrium import EquilibriumStructure
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int
from scipy.optimize import minimize
from smax import LoadCase

from normax.analysis import MemberForces
from normax.analysis import SmaxAnalyzer
from normax.analysis import frame_model
from normax.design import Design
from normax.design import StructuralDesignPipeline
from normax.design import design_envelope
from normax.form_finding import FdmFormFinder
from normax.form_finding import FormFoundShape
from normax.form_finding import SubspaceFormFinder
from normax.form_finding import density_basis
from normax.form_finding import equilibrium_graph
from normax.form_finding import pivoted_basis
from normax.loads import LoadCases
from normax.loads import assemble_load_cases
from normax.loads import create_loads_point
from normax.loads import create_loads_tributary
from normax.materials import Steel355
from normax.reporting import Report
from normax.reporting import ReportColumn
from normax.reporting import ToleranceCheck
from normax.reporting import checks_passed
from normax.sections import MemberSections
from normax.sections import TubeFamily
from normax.sizing import Ec3Sizer
from normax.sizing import build_section_family
from normax.structures import Structure
from normax.structures import member_lengths
from normax.visualization import DescentTrace
from normax.visualization import UtilizationForm
from normax.visualization import figure_mass_descent
from normax.visualization import figure_utilization

CASE_NAMES = (
    "LC1 uniform deck",
    "LC2 half span",
    "LC3 half span mirrored",
    "LC4 midspan point",
)

# The shell's cases: a pressure, and a drift with its own mirror image, so the
# pair is jointly symmetric about the plane the design is folded by.
SHELL_NAMES = (
    "LC1 uniform pressure",
    "LC2 sector drift",
    "LC3 mirrored drift",
)

# Relative steps the central difference sweeps, and the worst scaled error the
# directional derivative may show at its plateau.
GRADIENT_STEPS = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
TOLERANCE_GRADIENT = 1e-6

# Worst constraint violation an answer may show — SLSQP holds its constraints
# to its own ftol, measured orders below this headroom.
TOLERANCE_FEASIBILITY = 1e-6

# How exactly the signed lens densities live in the searched basis, and how
# exactly the full form-finding solve reproduces the drawn lens from them.
TOLERANCE_PROJECTION = 1e-9
TOLERANCE_SHAPE = 1e-8

# How exactly the start's density fit balances the lens, scaled by the load.
TOLERANCE_FIT = 1e-11

# How exactly one load case must reflect onto another before its rows are
# dropped as a reindexing, as a share of the largest force in any case.
TOLERANCE_MIRRORED = 1e-12

# A member is counted fully stressed above this envelope utilization, and
# counted at the floor within this distance of the bound.
ACTIVE_UTILIZATION = 0.999
FLOOR_SLACK = 1e-6

# Violation a trial point is charged when its frame cannot be factorized —
# enormous against the order-one slack rows, so the line search recoils.
RECOIL_SLACK = 1e3

# Fixed, so a multi-start run is a measurement rather than a lottery.
SCATTER_SEED = 20260820

# Slack a scattered landing may sit at and still be counted feasible enough to
# compete; the run's own checks hold the winner to the real tolerance.
SCATTER_SLACK = -1e-6

# Growth passes a repair is allowed before it gives up on a landing. Capacity
# is strictly increasing in the diameter, so each pass moves the right way; a
# pass is needed at all only because a fatter member attracts more force.
REPAIR_PASSES = 8

# A member is governed by every case within this distance of its worst:
# mirror-paired cases tie exactly on self-mirrored members, and splitting a
# tie by index order would misreport a symmetric design as lopsided.
TIE_MARGIN = 1e-9

FIGURES = Path(__file__).resolve().parent.parent / "figures"

# Where a descent's answer is kept, so that looking at a design again is a
# read rather than a rerun.
DESIGNS = Path(__file__).resolve().parent.parent / "designs"
DESIGNS.mkdir(exist_ok=True)

# Both routes compile a gradient and a Jacobian program; the persistent cache
# keeps reruns from paying the compilations again.
COMPILATION_CACHE = Path(__file__).resolve().parent.parent / ".jax_cache"
COMPILATION_CACHE.mkdir(exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(COMPILATION_CACHE))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

# The fixture every pinned tolerance was measured at, so code rather than file.
GRADE = Steel355()
SECTION_CLASS = 3

ROUTE_FORMFOUND = "end to end"
ROUTE_HEIGHTS = "free heights"
ROUTE_DRAWN = "sizing only"
ROUTE_ORDER = (ROUTE_FORMFOUND, ROUTE_HEIGHTS, ROUTE_DRAWN)

# The truss is planar in XZ, so the axial force and the moment about y are
# the whole of what a member carries.
FORCE_DIAGRAMS = ("nx", "my")

# EN 1993-1-1 §6.1's recommended value, as every sizer in the repo states it.
GAMMA_M0_SHEAR = 1.0


def routes_present(keyed: dict[str, object]) -> tuple[str, ...]:
    """
    Which routes a keyed collection holds, in the shared order.

    Parameters
    ----------
    keyed :
        Anything keyed by route — maps, starts, answers or reads.

    Returns
    -------
    routes :
        The routes present, ordered as `ROUTE_ORDER` orders them.

    Notes
    -----
    Every table and every check reads its route list off the collection it is
    handed rather than off `ROUTE_ORDER`, which is what lets a solo run write
    the same report with one row in it. The order is still the shared one, so
    a full run's tables are unchanged to the character.
    """
    return tuple(route for route in ROUTE_ORDER if route in keyed)


class TrussConfig(NamedTuple):
    """
    The truss to build.

    Attributes
    ----------
    num_bays :
        Number of bottom-chord segments the span is divided into. Even, so
        that a bottom-chord node sits exactly at midspan for the point case.
    span :
        Horizontal distance between the two supports.
    depth :
        Height of the top chord above the bottom chord, as drawn.
    """

    num_bays: int
    span: float
    depth: float


class ShellConfig(NamedTuple):
    """
    The gridshell to build.

    Attributes
    ----------
    num_rings :
        Number of rings between the apex and the boundary, boundary included.
    num_spokes :
        Number of spokes radiating from the apex.
    radius :
        Radius of the circular plan of the cap.
    rise :
        Height of the apex above the plane of the boundary.
    oculus :
        Whether the crown is open. The apex node and the members reaching it
        are then absent, the first ring bounding a hole that carries no load.
    braced :
        Whether the quads are triangulated, both diagonals of every panel. It
        is what widens the held-plan basis: a quad cap leaves a fraction of its
        free node count, a triangulated one several times it.
    polar_diameters :
        Whether the diameters are folded by the polar symmetry as well as the
        mirror, leaving one section per ring per family instead of one per
        mirror pair. A fabrication constraint rather than a response to the
        loading: the drift cases stay one-sided, and the sections simply
        cannot answer them spoke by spoke.
    polar_heights :
        Whether the free-heights route's heights are folded the same way,
        leaving one height per ring. It changes what the comparison means:
        the two routes then search spaces of the same dimension, neither one
        inside the other, so a gap between them is no longer reach.
    guard_hoops :
        Whether the compression guard covers the hoops as well as the radials.
        Off is the shell's own division of labour, meridian compression with
        a free hoop that may take the membrane tension a dome asks of its
        lower rings; on forbids that and designs a wholly compressive net.
    """

    num_rings: int
    num_spokes: int
    radius: float
    rise: float
    oculus: bool
    braced: bool
    polar_diameters: bool
    polar_heights: bool
    guard_hoops: bool


class ShellLoads(NamedTuple):
    """
    The pressure the shell answers to, and where the cases put it.

    Attributes
    ----------
    pressure :
        Downward force per unit of plan area, which every distributed case
        carries the same total of.
    sector_spokes :
        How many spokes the drift case loads fully. Odd, so it centres on a
        spoke rather than between two.
    sector_center :
        Spoke the drift sector is centred on. Off the mirror plane the case is
        genuinely one-sided and its reflection is a second, different case;
        on the plane the two would coincide and the parse refuses it.
    drift_factor :
        Fraction of the pressure the plan outside the sector keeps, before
        the case is rescaled back to the shared total. The shell's reading of
        the trusses' `half_factor`: a drift is a redistribution over the
        whole roof, not a spotlight on a quarter of it, and a case that
        unloads three quarters entirely is a stress test rather than a
        snow load.
    asymmetric_cases :
        Whether the two drift cases are built at all. Off leaves the uniform
        pressure alone, which is the one loading a polar structure shares
        every symmetry with — so the answer ought to come back rotationally
        symmetric, and a search that returns anything else is reporting on
        itself rather than on the structure.

    Notes
    -----
    **The drifts come as a mirrored pair, which is what keeps a symmetric
    design honest.** One sector alone would ask a folded design to answer an
    unfolded load, and the fold would then average two halves the structure
    never sees separately. Reflecting the case instead gives the pair the
    mirror symmetry the design already has, so folding costs nothing and the
    one-sided demand still reaches every member — the trusses' half-span pair,
    read onto a disc.

    No apex case. A polar grid's crown is one node of many when it exists and
    no node at all once an oculus opens, so a point load there is a property
    of the drawing rather than of the structure.
    """

    pressure: float
    sector_spokes: int
    sector_center: int
    drift_factor: float
    asymmetric_cases: bool


class LoadConfig(NamedTuple):
    """
    The load every case carries, however it sits.

    Attributes
    ----------
    total :
        Total downward force of every distributed case.
    half_factor :
        Fraction of the deck load the unloaded half keeps in the asymmetric
        cases, before the case is rescaled back to the shared total.
    point_factor :
        Fraction of the total the midspan point case concentrates. The one
        case exempt from the shared total: a lone wheel is not the whole
        deck, and at the full total it governs nearly every member.
    mirrored_case :
        Whether the mirrored half-span case is built. On a fully folded
        problem its constraint rows duplicate the other half-span case's
        through the mirror, so deleting it buys a quarter of every analysis
        without moving the optimum; on an unfolded problem it is what keeps
        the unloaded half honest, and the parse refuses to drop it.
    """

    total: float
    half_factor: float
    point_factor: float
    mirrored_case: bool


class SketchConfig(NamedTuple):
    """
    The lens the end-to-end route starts from.

    Attributes
    ----------
    sag_lens :
        Depth the sketch hangs its bottom chord to at midspan.
    rise_lens :
        Height the sketch arches its top chord to at midspan.
    """

    sag_lens: float
    rise_lens: float


class SubspaceConfig(NamedTuple):
    """
    Which held-plan basis the geometry variables span.

    Attributes
    ----------
    symmetric :
        Whether the search runs on the mirror-symmetric basis.
    basis :
        Which coordinates span the subspace: `svd` for the orthonormal
        null-space basis, `pivoted` for the member-named independent-edge
        basis QR pivoting elects. The two span the identical subspace, so
        switching prices the coordinates, never the reachable designs.
    margin_fraction :
        Sign margin the starting chords must clear, as a share of their
        median density.
    """

    symmetric: bool
    basis: str
    margin_fraction: float


class AnalysisConfig(NamedTuple):
    """
    What the frame is analyzed with, before either search has spoken.

    Attributes
    ----------
    diameter :
        Outer diameter every member is seeded with.
    """

    diameter: float


class DescentConfig(NamedTuple):
    """
    The budgets the constrained searches share.

    Attributes
    ----------
    iterations :
        Most iterations to spend in each SLSQP round.
    rounds :
        Most restarts, each rerun from the previous round's answer.
    tolerance :
        Convergence tolerance of the constrained solver.
    diameter_floor :
        Smallest diameter any member may take, as a bound rather than a
        constraint, so the fully-stressed condition stays readable off the
        constraint activities alone.
    length_floor :
        Smallest length any member may keep while the free-heights route
        moves the geometry, as inequality rows — a collapsed member is a
        singular frame, not a light one. At least half the drawn depth, so
        a vertical stays a member rather than a near-hinge.
    limit_rise :
        Whether any vertex is kept under the rise ceiling.
    rise_factor :
        The ceiling, as a multiple of the drawn depth.
    limit_sag :
        Whether any vertex is kept above the sag floor.
    starts :
        How many points each route is descended from, the first being the
        nominal start and the rest scattered around it. One is a single
        descent and the search a local one.
    scatter :
        Relative spread of the scattered starts, as a fraction of each
        variable's own value.
    reuse_answers :
        Whether to read a stored answer back instead of descending to it
        again, where one is held for this same run description. Every run
        writes its answers whatever this says; only reading them is a
        decision.
    sag_factor :
        The floor, as a multiple of the drawn depth below zero: at 1.0 on a
        1000 mm truss no vertex may hang under -1000 mm — the mirror of the
        ceiling on the other side of the supports.
    """

    iterations: int
    rounds: int
    tolerance: float
    diameter_floor: float
    length_floor: float
    limit_rise: bool
    rise_factor: float
    limit_sag: bool
    sag_factor: float
    starts: int
    scatter: float
    reuse_answers: bool = False


class ViewerConfig(NamedTuple):
    """
    Whether the run ends in a viewer, and which answer it draws there.

    Attributes
    ----------
    enabled :
        Whether an answer is opened in a viewer once the report is written.
    route :
        Which route's answer to draw, named as the routes are named.
    solo_route :
        Whether to descend `route` alone and leave the other two undone. The
        report then holds whatever a single route can say — its own landing,
        its families, its checks — and drops every entry that is a comparison.
    load_case :
        Name, or leading part of a name, of the one load case to draw the
        response of. Empty draws every case the run was checked against.
    load_scale :
        Multiple the load glyphs are drawn at. A load true to its own size is
        a load nobody can see on a shell that spans ten metres, so the scene
        exaggerates it and says by how much rather than leaving the reader to
        guess.

    Notes
    -----
    Off by every shipped file, because a viewer blocks until its window closes
    and a run that stalls is one a sweep cannot make.

    One route rather than a set of them: two answers occupy nearly the same
    space, so a scene holding both is read by switching halves of it off, and
    naming the one wanted is the shorter way to the same look.

    **A solo run is for iterating, never for reporting a result.** The gaps
    between the routes are the point of the comparison, and a run that
    descends one route cannot state them; what it buys is the loop between
    changing a file and seeing the shape, which the slowest route otherwise
    sets the pace of.
    """

    enabled: bool
    route: str
    solo_route: bool
    load_case: str = ""
    load_scale: float = 1.0


class TaskConfig(NamedTuple):
    """
    Everything a run is described by.

    Attributes
    ----------
    structure :
        The structure to build, in whichever family's terms it is drawn.
    loads :
        The load every case carries, in whichever family's terms it is stated.
    sketch :
        The lens the end-to-end route starts from, or None where the drawn
        geometry is already the start and no sketch is needed.
    subspace :
        Which held-plan basis the geometry variables span.
    analysis :
        What the frame is seeded with.
    descent :
        The budgets the searches share.
    viewer :
        Whether the run ends in a viewer.

    Notes
    -----
    The first three sections are the profile's to parse and the profile's to
    read; nothing in the shared flow touches a field of them. The last four
    are family-blind, which is what lets one flow run a truss and a shell.
    """

    structure: TrussConfig | ShellConfig
    loads: LoadConfig | ShellLoads
    sketch: SketchConfig | None
    subspace: SubspaceConfig
    analysis: AnalysisConfig
    descent: DescentConfig
    viewer: ViewerConfig


def shared_sections(document: dict[str, object]) -> dict[str, object]:
    """
    The four sections every family describes a run with, parsed and checked.

    Parameters
    ----------
    document :
        The loaded YAML document.

    Returns
    -------
    sections :
        Keyword arguments naming the family-blind half of a `TaskConfig`.

    Raises
    ------
    ValueError
        If the basis or the viewer's route is not one this flow serves.
    TypeError
        If a section names a field that does not exist, or omits one it does.

    Notes
    -----
    No container carries a default, so a file missing a field is refused
    rather than quietly completed.
    """
    subspace = SubspaceConfig(**document["subspace"])
    viewer = ViewerConfig(**document["viewer"])

    if subspace.basis not in ("svd", "pivoted"):
        raise ValueError(f"basis must be svd or pivoted, got {subspace.basis}")
    if viewer.route not in ROUTE_ORDER:
        named = ", ".join(ROUTE_ORDER)
        raise ValueError(f"viewer route must be one of {named}, got {viewer.route}")

    return {
        "subspace": subspace,
        "analysis": AnalysisConfig(**document["analysis"]),
        "descent": DescentConfig(**document["descent"]),
        "viewer": viewer,
    }


def parse_truss(text: str) -> TaskConfig:
    """
    The truss and the budgets a run is described by.

    Parameters
    ----------
    text :
        Text of the file describing the run.

    Returns
    -------
    config :
        The truss, and the settings its routes are compared under.

    Raises
    ------
    ValueError
        If the length floor is shallower than half the drawn depth, or the
        mirrored case is deleted from a problem that is not folded.
    """
    document = yaml.safe_load(text)

    config = TaskConfig(
        structure=TrussConfig(**document["structure"]),
        loads=LoadConfig(**document["loads"]),
        sketch=SketchConfig(**document["sketch"]),
        **shared_sections(document),
    )
    shallowest = 0.5 * config.structure.depth
    if config.descent.length_floor < shallowest:
        raise ValueError(
            f"length_floor must be at least half the depth, {shallowest}, "
            f"got {config.descent.length_floor}"
        )
    if not config.loads.mirrored_case and not config.subspace.symmetric:
        raise ValueError(
            "the mirrored half-span case can only be deleted from a symmetric"
            " problem — set symmetric: true, or keep mirrored_case: true"
        )

    return config


def parse_shell(text: str) -> TaskConfig:
    """
    The gridshell and the budgets a run is described by.

    Parameters
    ----------
    text :
        Text of the file describing the run.

    Returns
    -------
    config :
        The shell, and the settings its routes are compared under.

    Raises
    ------
    ValueError
        If the drift sector cannot be centred on a spoke, is wider than the
        shell has spokes, or is centred on the mirror plane itself, where the
        two drift cases would be the same case twice.

    Notes
    -----
    No sketch section: the generated cap is already funicular under its own
    uniform case, so the end-to-end route starts on the drawn geometry rather
    than on a lens sketched beside it.

    No length-floor rule either. A held plan bounds every member length below
    by its own plan projection, which no search can shorten, so the floor that
    keeps a truss's verticals from collapsing has nothing to do here.
    """
    document = yaml.safe_load(text)

    config = TaskConfig(
        structure=ShellConfig(**document["structure"]),
        loads=ShellLoads(**document["loads"]),
        sketch=None,
        **shared_sections(document),
    )
    spokes = config.loads.sector_spokes
    if spokes % 2 == 0:
        raise ValueError(
            f"sector_spokes must be odd to centre on a spoke, got {spokes}"
        )
    if spokes > config.structure.num_spokes:
        raise ValueError(
            f"sector_spokes must not exceed num_spokes, "
            f"{config.structure.num_spokes}, got {spokes}"
        )
    center = config.loads.sector_center
    turns = config.structure.num_spokes
    if not 0 <= center < turns:
        raise ValueError(f"sector_center must be a spoke in [0, {turns}), got {center}")
    reflected = (-center) % turns
    if config.loads.asymmetric_cases and reflected == center:
        raise ValueError(
            f"sector_center {center} lies on the mirror plane, so the drift "
            f"and its reflection are the same case — centre it off the plane, "
            f"anywhere but 0 and {turns // 2}"
        )

    return config


class MirrorFolding(NamedTuple):
    """
    Pattern matrices folding the mirror into the searched variables.

    Attributes
    ----------
    diameters :
        One column per mirror orbit of members, or None to size every member
        independently.
    heights :
        One column per mirror orbit of free nodes, or None to move every
        height independently.

    Notes
    -----
    The symmetric switch folds the whole problem, not just the density
    basis: a pattern variable is the shared value of its orbit, expanding is
    one matmul, and every route then searches a mirror-symmetric design
    space. With the switch off both matrices are None and the routes run on
    the full variables, untouched.
    """

    diameters: Float[Array, "edges patterns_diameter"] | None
    heights: Float[Array, "nodes_free patterns_height"] | None


class RouteProblem(NamedTuple):
    """
    The prepared truss, its blocks, and the subspace the geometry moves in.

    Attributes
    ----------
    structure :
        The truss the blocks were built against, supplying the drawn geometry
        the sizing-only route holds.
    pipeline :
        The three blocks, each already bound to the truss on the host. The
        first is a `SubspaceFormFinder`, so the end-to-end route's geometry
        variables are the coordinates the block itself declares.
    loads :
        The case the shape answers to, and the cases every route is checked
        against.
    case_names :
        Name of every built load case, in build order.
    cases_held :
        Which load cases the descents carry inequality rows for. A symmetric
        search may need fewer than were built, and every answer is still read
        and checked against all of them.
    folding :
        Pattern matrices folding the mirror into every route's variables,
        None-valued when the search is not symmetric.
    edges_mirrored :
        The member the midspan mirror carries each member onto.
    nodes_free :
        Indices of the nodes whose height the free-heights route moves.
    diameters_seed :
        Outer diameter the frame is analyzed at before any search sizes it.
    """

    structure: Structure
    pipeline: StructuralDesignPipeline
    loads: LoadCases
    case_names: tuple[str, ...]
    cases_held: Int[np.ndarray, "cases_held"]
    folding: MirrorFolding
    edges_mirrored: Int[np.ndarray, "edges"]
    nodes_free: Int[Array, "nodes_free"]
    diameters_seed: Float[Array, "edges"]


class StartPoint(NamedTuple):
    """
    Where the end-to-end route leaves from, and how exactly it was matched.

    Attributes
    ----------
    q :
        The signed lens densities, chords carrying their signs.
    xi :
        The same densities as coordinates of the searched basis.
    lens :
        The sketch the densities were fitted to.
    projection :
        How much of the signed densities the basis fails to express.
    gap :
        Balance violation the density fit left at the sketch.
    """

    q: Float[np.ndarray, "edges"]
    xi: Float[np.ndarray, "independents"]
    lens: Float[np.ndarray, "nodes 3"]
    projection: float
    gap: float


class SignShift(NamedTuple):
    """
    A fit shifted along its self-stress until the chords carry their signs.

    Attributes
    ----------
    q :
        The shifted densities, each chord clearing its sign margin.
    window :
        Interval of shifts that sign the chords, after capping.
    shift :
        The shift taken, the feasible one nearest zero, stepped inside.
    """

    q: Float[np.ndarray, "edges"]
    window: tuple[float, float]
    shift: float


class ChordSigns(NamedTuple):
    """
    The sign each chord density must keep, entering the slack as linear rows.

    Attributes
    ----------
    signs :
        Sign each chord member must carry, positive for tension.
    chords :
        Indices of the chord members the signs speak about.
    margin :
        Density each chord must clear beyond zero, in its own sign.
    scale :
        Density the rows are normalized by, putting them at the utilization
        rows' scale.

    Notes
    -----
    A guard for trusses whose held-plan subspace touches degenerate states:
    a chord density crossing zero switches off that chord's chain, the
    vertical stiffness the form finder solves turns singular, and the frame
    analysis is handed a non-finite geometry. The rows are exactly linear in
    the searched coordinates, so the quadratic subproblem holds every trial
    point on the signed sheet of the manifold rather than merely the answer.
    """

    signs: Float[np.ndarray, "chords"]
    chords: Int[np.ndarray, "chords"]
    margin: float
    scale: float


def folding_matrix(
    mirrors: Int[np.ndarray, "items"],
) -> Float[np.ndarray, "items patterns"]:
    """
    One column per mirror orbit, carrying each of its members at one.

    Parameters
    ----------
    mirrors :
        The item the mirror carries each item onto.

    Returns
    -------
    spread :
        Matrix expanding one value per orbit into a full, symmetric vector.
    """
    columns = []
    seen = set()
    for index, partner in enumerate(mirrors.tolist()):
        if index in seen:
            continue
        column = np.zeros(mirrors.size)
        column[index] = 1.0
        column[partner] = 1.0
        columns.append(column)
        seen.add(index)
        seen.add(partner)

    return np.stack(columns, axis=1)


def orbit_matrix(
    mappings: tuple[Int[np.ndarray, "items"], ...],
) -> Float[np.ndarray, "items patterns"]:
    """
    One column per orbit of the group several permutations generate.

    Parameters
    ----------
    mappings :
        The item each permutation carries each item onto, one array per
        generator. A single generator that is an involution reproduces
        `folding_matrix` exactly, column for column.

    Returns
    -------
    spread :
        Matrix expanding one value per orbit into a full vector that every
        generator leaves unchanged.

    Notes
    -----
    Orbits come from union-find over every generator at once, so the group
    they generate is folded rather than each generator separately — a mirror
    and a one-spoke rotation together give the whole dihedral group, not two
    reflections. Columns are ordered by their smallest member, which keeps the
    pattern order stable as generators are added or dropped.

    Folding is a restriction of the search, not a symmetrisation of the
    answer: a pattern variable *is* the shared value of its orbit, so the
    design cannot break the symmetry however unsymmetric the loading is.
    """
    size = int(mappings[0].size)
    parent = list(range(size))

    def root_of(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for mapping in mappings:
        for index, image in enumerate(mapping.tolist()):
            left = root_of(index)
            right = root_of(int(image))
            if left != right:
                parent[max(left, right)] = min(left, right)

    orbits: dict[int, list[int]] = {}
    for index in range(size):
        orbits.setdefault(root_of(index), []).append(index)

    columns = []
    for root in sorted(orbits):
        column = np.zeros(size)
        column[orbits[root]] = 1.0
        columns.append(column)

    return np.stack(columns, axis=1)


def folded_seed(
    values: Float[np.ndarray, "items"],
    spread: Float[Array, "items patterns"] | None,
) -> Float[np.ndarray, "patterns"]:
    """
    Fold a full seed vector into one value per mirror orbit.

    Parameters
    ----------
    values :
        The full seed, one value per item.
    spread :
        The orbit columns, or None to keep the seed as it is.

    Returns
    -------
    seed :
        The largest value of each orbit — an envelope, so a folded diameter
        seed still covers both members it now sizes at once.
    """
    if spread is None:
        return values

    columns = np.asarray(spread).T
    folded = [float(values[column > 0.0].max()) for column in columns]

    return np.asarray(folded)


def unfolded_values(
    values: Float[np.ndarray, "patterns"],
    spread: Float[Array, "items patterns"] | None,
) -> Float[np.ndarray, "items"]:
    """
    Expand pattern values back into one value per item.

    Parameters
    ----------
    values :
        One value per mirror orbit.
    spread :
        The orbit columns, or None when the values are already full.

    Returns
    -------
    expanded :
        The full vector, orbit members carrying their shared value.
    """
    if spread is None:
        return values

    return np.asarray(spread) @ values


def pattern_count(spread: Float[Array, "items patterns"] | None, full: int) -> int:
    """
    How many variables a folded block searches.

    Parameters
    ----------
    spread :
        The orbit columns, or None when the block is not folded.
    full :
        The unfolded count.

    Returns
    -------
    count :
        One per orbit when folded, the full count otherwise.
    """
    if spread is None:
        return full

    return int(spread.shape[1])


class StartMeasures(NamedTuple):
    """
    What the seed numbers say before any descent has moved.

    Attributes
    ----------
    reproduction :
        How far the full form-finding solve puts the truss from the lens.
    disagreement :
        How far the elastic axial forces sit from the funicular prediction.
    opening_found :
        Smallest constraint slack of the lens seed, negative when infeasible.
    opening_drawn :
        Smallest constraint slack of the drawn seed, negative when infeasible.
    """

    reproduction: float
    disagreement: float
    opening_found: float
    opening_drawn: float


class RouteMaps(NamedTuple):
    """
    The compiled maps a constrained descent calls, over one variable vector.

    Attributes
    ----------
    weigh :
        The mass and its gradient together.
    slack :
        How far under one every member's utilization sits, per load case.
    jacobian :
        The slack's derivative in every variable, by forward mode — the
        variables are the short axis against members times cases.
    repair :
        Grower of the diameters of a landing that missed feasibility, or None
        from a route that offers no repair.

    Notes
    -----
    **A repair is not a relaxation.** A landing that stops short of the
    constraints is cheaper than a feasible one by construction, so accepting
    it would bias every reported mass downward by an amount nothing bounds.
    Growing the diameters instead walks the same design back onto the
    constraint surface and prices the walk, which is the difference between
    reporting an optimum and reporting a point the solver happened to stop at.

    It is sound because the resistance the check computes is strictly
    increasing in the diameter — the same monotonicity that makes the sizing
    map's bisection unconditionally safe. It is iterative rather than closed
    because a fatter member is a stiffer one, and stiffness redistributes the
    force the resistance is measured against.
    """

    weigh: object
    slack: object
    jacobian: object
    repair: object = None


class RouteAnswer(NamedTuple):
    """
    What one constrained descent arrived at, and the road there.

    Attributes
    ----------
    variables :
        The variable vector the solver stopped on.
    masses :
        Objective at every iterate, the start included, across all rounds.
    iterations :
        Iterations spent over every round.
    converged :
        Whether the last round reported clean convergence.
    """

    variables: Float[np.ndarray, "variables"]
    masses: Float[np.ndarray, "steps"]
    iterations: int
    converged: bool


class RouteRead(NamedTuple):
    """
    One answer read back as a design.

    Attributes
    ----------
    mass :
        Mass of the frame analyzed at its own sections.
    xyz :
        Position of every node of the answer.
    rise :
        Height of the highest node.
    sag :
        Height of the lowest node, negative below the supports.
    diameters :
        Outer diameter of every member.
    utilization :
        Worst utilization of every member over the load cases.
    utilization_cases :
        Utilization of every member under every load case, the table every
        governing count is read from.
    active :
        Count of members whose envelope utilization sits at one.
    floored :
        Count of members resting on the diameter floor.
    mirror :
        How far the diameters depart from their own reflection.
    shear :
        Largest design shear over members and load cases, as a fraction of the
        plastic shear resistance. EN 1993-1-1 6.2.10 lets the check leave shear
        out only while this stays under half, so the answer carries the number
        that says whether it may.
    """

    mass: float
    xyz: Float[np.ndarray, "nodes 3"]
    rise: float
    sag: float
    diameters: Float[np.ndarray, "edges"]
    utilization: Float[np.ndarray, "edges"]
    utilization_cases: Float[np.ndarray, "cases edges"]
    active: int
    floored: int
    mirror: float
    shear: float


def build_load_cases(
    structure: Structure,
    weight: LoadConfig,
    num_bays: int,
) -> LoadCases:
    """
    Four cases of equal total, every one on the bottom chord alone.

    Parameters
    ----------
    structure :
        The truss to load.
    weight :
        The total and the asymmetry factor.
    num_bays :
        Number of bottom-chord segments, locating the interior deck nodes.

    Returns
    -------
    loads :
        The uniform deck the shape answers to, the two half-span cases, and
        a fraction of the total concentrated at the midspan deck node.

    Notes
    -----
    The arch experiments' load family, moved onto the deck: the top chord
    carries nothing directly, matching a bridge whose traffic runs on the
    bottom chord. The three distributed cases are rescaled to the shared
    total so none wins by simply carrying less; the point case carries its
    own fraction of it.
    """
    if num_bays % 2:
        raise ValueError(f"num_bays must be even for a midspan node, got {num_bays}")

    interior = np.arange(1, num_bays)
    along = np.asarray(structure.nodes)[interior, 0]
    middle = 0.5 * float(np.asarray(structure.nodes)[num_bays, 0])

    def deck_case(weights: Float[np.ndarray, "interior"]) -> Float[Array, "nodes 3"]:
        scaled = weights * (weight.total / float(weights.sum()))
        cases = [
            create_loads_point(structure, float(load), node=int(node))
            for node, load in zip(interior, scaled)
        ]

        return jnp.sum(jnp.stack(cases), axis=0)

    uniform = deck_case(np.ones(interior.size))
    near = deck_case(np.where(along <= middle, 1.0, weight.half_factor))
    concentrated = weight.total * weight.point_factor
    point = create_loads_point(structure, concentrated, node=num_bays // 2)
    cases = [uniform, near, point]
    if weight.mirrored_case:
        far = deck_case(np.where(along >= middle, 1.0, weight.half_factor))
        cases.insert(2, far)

    return assemble_load_cases(cases)


def load_names(weight: LoadConfig) -> tuple[str, ...]:
    """
    Name of every load case built, in build order.

    Parameters
    ----------
    weight :
        The load description, read for whether the mirrored case is built.

    Returns
    -------
    names :
        The case names, keeping their identities when one is deleted.
    """
    if weight.mirrored_case:
        return CASE_NAMES

    return (CASE_NAMES[0], CASE_NAMES[1], CASE_NAMES[3])


class LoadPlan(NamedTuple):
    """
    Every case a run is checked against, named, and what they each weigh.

    Attributes
    ----------
    cases :
        The case the shape is found under, and the stack every route is
        checked against.
    names :
        Name of every case, in build order.
    total :
        Downward force each distributed case carries, the scale the start's
        equilibrium gap is reported against.

    Notes
    -----
    A profile's one job on the loading side is to return this: how a family
    spreads a load over its own nodes is the one part of a load case no
    shared flow can know, while the three things read afterwards are the same
    for every family.
    """

    cases: LoadCases
    names: tuple[str, ...]
    total: float


def truss_loads(structure: Structure, config: TaskConfig) -> LoadPlan:
    """
    The deck cases of experiments 18 and 19, gathered into a plan.

    Parameters
    ----------
    structure :
        The truss to load.
    config :
        The run description, read for the loads and the bay count.

    Returns
    -------
    plan :
        The four deck cases, or three where the mirrored one is deleted.
    """
    weight = config.loads
    cases = build_load_cases(structure, weight, config.structure.num_bays)

    return LoadPlan(cases, load_names(weight), weight.total)


def tributary_areas(sketch: ShellConfig) -> Float[np.ndarray, "nodes"]:
    """
    Plan area every node of a polar cap carries.

    Parameters
    ----------
    sketch :
        The cap the generator was asked to draw.

    Returns
    -------
    areas :
        Plan area of every node, the apex first where there is one and then
        ring by ring.

    Notes
    -----
    Each ring owns the annulus reaching halfway to its neighbours, split
    evenly between its spokes; the apex owns the disc inside the first such
    boundary. The areas therefore sum to the whole plan exactly, which is what
    makes the supports' share readable as the difference between the stated
    pressure's total and the total actually applied.

    **An oculus is open, so it carries nothing.** The first ring then owns
    only the annulus outside itself, and the areas sum to the plan less the
    hole — the run's stated pressure buys less total load than the same
    pressure on a closed cap, which is part of what the opening costs.
    """
    rings = sketch.num_rings
    spokes = sketch.num_spokes

    rhos = sketch.radius * np.arange(1, rings + 1) / rings
    inner = np.concatenate([[0.0], 0.5 * (rhos[:-1] + rhos[1:])])
    outer = np.concatenate([0.5 * (rhos[:-1] + rhos[1:]), [sketch.radius]])

    inner[0] = rhos[0] if sketch.oculus else 0.5 * rhos[0]

    annuli = np.pi * (outer**2 - inner**2) / spokes
    ring_areas = np.repeat(annuli, spokes)
    if sketch.oculus:
        return ring_areas

    apex = np.pi * inner[0] ** 2

    return np.concatenate([[apex], ring_areas])


def sector_areas(
    sketch: ShellConfig,
    weight: ShellLoads,
    areas: Float[np.ndarray, "nodes"],
    center: int,
) -> Float[np.ndarray, "nodes"]:
    """
    The tributary areas a drift over one sector loads each node through.

    Parameters
    ----------
    sketch :
        The cap the generator was asked to draw.
    weight :
        The loading, read for the sector width and what the rest keeps.
    areas :
        Plan area of every node, as the tributary rule shares it out.
    center :
        Spoke the sector is centred on.

    Returns
    -------
    drifting :
        Each node's area, kept whole inside the sector and scaled by
        `drift_factor` outside it. A crown node, sitting on every sector's
        axis, is always inside.

    Notes
    -----
    **The drift grades rather than spotlights.** The sector keeps the full
    pressure and the rest of the plan keeps its fraction, which is the
    trusses' half-span construction read onto a disc. Emptying the plan
    outside the sector instead would concentrate the whole roof's load on a
    slice of it once rescaled — a stress test rather than a snow load, and one
    whose feasible set is measurably harder to descend.
    """
    spokes = sketch.num_spokes
    reach = weight.sector_spokes // 2

    offset = (np.arange(spokes) - center + reach) % spokes
    within = offset <= 2 * reach
    tiled = np.tile(within, sketch.num_rings)
    inside = tiled if sketch.oculus else np.concatenate([[True], tiled])

    return np.where(inside, areas, weight.drift_factor * areas)


def shell_loads(structure: Structure, config: TaskConfig) -> LoadPlan:
    """
    A uniform pressure, a drift over one sector, and that drift reflected.

    Parameters
    ----------
    structure :
        The shell to load.
    config :
        The run description, read for the pressure and the sector.

    Returns
    -------
    plan :
        The three cases, every one of them carrying the same total.

    Notes
    -----
    **The stated pressure and the applied total are two different numbers.**
    The pressure acts on the whole plan, but the boundary ring's tributary
    share sits on supported nodes and goes straight to ground, so the total
    the structure carries is what is left. That remainder is the plan's total,
    and both drift cases are rescaled onto it so no case wins by carrying
    less.

    **The second drift is the first one's mirror image**, built by reflecting
    the sector's centre rather than by permuting the case, so the two are the
    same construction at two centres and their asymmetries cancel over the
    pair. A design folded about that plane therefore loses nothing: what one
    case asks of a member, the other asks of its mirror twin.
    """
    sketch = config.structure
    weight = config.loads
    areas = tributary_areas(sketch)

    uniform = create_loads_tributary(structure, weight.pressure, jnp.asarray(areas))
    total = float(jnp.sum(jnp.abs(uniform)))

    if not weight.asymmetric_cases:
        return LoadPlan(assemble_load_cases([uniform]), SHELL_NAMES[:1], total)

    center = weight.sector_center
    reflected = (-center) % sketch.num_spokes

    drifts = []
    for spoke in (center, reflected):
        drifting = sector_areas(sketch, weight, areas, spoke)
        drift = create_loads_tributary(
            structure, weight.pressure, jnp.asarray(drifting)
        )
        carried = float(jnp.sum(jnp.abs(drift)))
        drifts.append(drift * (total / carried))

    cases = assemble_load_cases([uniform, *drifts])

    return LoadPlan(cases, SHELL_NAMES, total)


def lens_geometry(
    structure: Structure,
    span: float,
    num_bays: int,
    sag: float,
    rise: float,
) -> Float[np.ndarray, "nodes 3"]:
    """
    The drawn truss with each chord bent into a parabola, the plan held.

    Parameters
    ----------
    structure :
        The truss as drawn.
    span :
        Horizontal distance between the two supports.
    num_bays :
        Number of bottom-chord segments, splitting the nodes into chords.
    sag :
        Depth the bottom chord hangs to at midspan.
    rise :
        Height the top chord arches to at midspan, above its drawn line.

    Returns
    -------
    xyz :
        The sketch, every horizontal coordinate as drawn.
    """
    xyz = np.asarray(structure.nodes).copy()
    shape = 4.0 * (xyz[:, 0] / span) * (1.0 - xyz[:, 0] / span)

    bottom = slice(0, num_bays + 1)
    top = slice(num_bays + 1, None)
    xyz[bottom, 2] -= sag * shape[bottom]
    xyz[top, 2] += rise * shape[top]

    return xyz


def mirrored_edges(
    nodes_mirrored: Int[np.ndarray, "nodes"],
    structure: Structure,
) -> Int[np.ndarray, "edges"]:
    """
    Index of every member's mirror image about midspan.

    Parameters
    ----------
    nodes_mirrored :
        The node the mirror carries each node onto.
    structure :
        The truss supplying the members the mirror permutes.

    Returns
    -------
    edges_mirrored :
        The member the mirror carries each member onto.
    """
    return permuted_members(nodes_mirrored, structure)


def permuted_members(
    nodes_permuted: Int[np.ndarray, "nodes"],
    structure: Structure,
) -> Int[np.ndarray, "edges"]:
    """
    Index of every member's image under a permutation of the nodes.

    Parameters
    ----------
    nodes_permuted :
        The node the permutation carries each node onto.
    structure :
        The structure supplying the members the permutation acts on.

    Returns
    -------
    edges_permuted :
        The member the permutation carries each member onto.

    Raises
    ------
    KeyError
        If some member's image is not itself a member, which means the
        permutation is not a symmetry of the structure.

    Notes
    -----
    Members are matched unordered, so a permutation that reverses a member
    still finds it. Nothing here assumes the permutation is an involution: a
    rotation is looked up the same way a reflection is.
    """
    edges = np.asarray(structure.edges)
    ordered = np.sort(edges, axis=1)
    moved = np.sort(nodes_permuted[edges], axis=1)

    lookup = {tuple(pair): index for index, pair in enumerate(ordered.tolist())}
    targets = [lookup[tuple(pair)] for pair in moved.tolist()]

    return np.asarray(targets)


class FoldingMaps(NamedTuple):
    """
    The permutations a run folds each kind of variable by.

    Attributes
    ----------
    nodes_mirrored :
        The node the mirror carries each node onto. Restricts the density
        basis, which is folded by the mirror alone whatever else is.
    nodes_folded :
        The node permutations the free heights are folded by, the mirror
        first.
    members_folded :
        The member permutations the diameters are folded by, the mirror
        first.

    Notes
    -----
    Three entries rather than one because the three kinds of variable need not
    fold by the same group, and on a polar grid they deliberately do not: a
    section may be folded as far as fabrication wants, while folding the
    geometry changes what a comparison between the routes even means.
    """

    nodes_mirrored: Int[np.ndarray, "nodes"]
    nodes_folded: tuple[Int[np.ndarray, "nodes"], ...]
    members_folded: tuple[Int[np.ndarray, "edges"], ...]


def folding_maps(
    profile: "RouteProfile",
    config: TaskConfig,
    structure: Structure,
) -> FoldingMaps:
    """
    Every permutation a run folds its variables by, gathered once.

    Parameters
    ----------
    profile :
        The structural family, read for the mirror and for whichever
        rotations it offers.
    config :
        The run description, which the profile reads to decide which
        rotations are wanted.
    structure :
        The structure the permutations act on.

    Returns
    -------
    folding :
        The mirror, the height permutations and the member permutations. The
        mirror leads both tuples, so a caller wanting it alone reads the first
        entry.
    """
    nodes_mirrored = profile.mirrored_nodes(config)

    heights = [nodes_mirrored]
    if profile.heights_rotated is not None:
        turned = profile.heights_rotated(config)
        if turned is not None:
            heights.append(turned)

    sections = [nodes_mirrored]
    if profile.sections_rotated is not None:
        turned = profile.sections_rotated(config)
        if turned is not None:
            sections.append(turned)

    members = tuple(permuted_members(nodes, structure) for nodes in sections)

    return FoldingMaps(nodes_mirrored, tuple(heights), members)


def signed_shift(
    q: Float[np.ndarray, "edges"],
    mode: Float[np.ndarray, "edges"],
    signs: Float[np.ndarray, "chords"],
    chords: Int[np.ndarray, "chords"],
    margin: float,
) -> SignShift:
    """
    Shift densities along a self-stress until every chord carries its sign.

    Parameters
    ----------
    q :
        The fitted densities to shift.
    mode :
        The self-stress direction to shift along.
    signs :
        Sign each chord member must carry, positive for tension.
    chords :
        Indices of the chord members the signs speak about.
    margin :
        Density each chord must clear beyond zero, in its own sign.

    Returns
    -------
    shifted :
        The signed densities, the feasible window, and the shift taken.

    Notes
    -----
    Each chord member asks its sign of the shift as one linear inequality, so
    the feasible set is an interval and is intersected exactly. Of the
    feasible shifts the one nearest zero is taken, stepped a twentieth of the
    window inside it. Members off the chords are left free on purpose: a
    hanger in one shape is a post in another, and a sign pinned here would
    fight the physics later.
    """
    values = signs * q[chords]
    slopes = signs * mode[chords]

    cap = 20.0 * float(np.abs(q).max())
    lower, upper = -cap, cap
    for value, slope in zip(values, slopes):
        if slope > 1e-12:
            lower = max(lower, (margin - value) / slope)
        elif slope < -1e-12:
            upper = min(upper, (margin - value) / slope)
        elif value < margin:
            raise ValueError("a chord ignores the self-stress and misses its sign")
    if lower > upper:
        raise ValueError("no self-stress shift signs both chords at once")

    inset = 0.05 * (upper - lower)
    shift = float(np.clip(0.0, lower + inset, upper - inset))

    return SignShift(q + shift * mode, (lower, upper), shift)


def cases_constrained(
    plan: LoadPlan,
    structure: Structure,
    nodes_mirrored: Int[np.ndarray, "nodes"],
    symmetric: bool,
) -> Int[np.ndarray, "cases_held"]:
    """
    Which load cases a descent needs rows for, mirror duplicates dropped.

    Parameters
    ----------
    plan :
        The load cases the profile built.
    structure :
        The structure the cases load, read for the mirror it is drawn with.
    nodes_mirrored :
        The node the mirror carries each node onto.
    symmetric :
        Whether the search is confined to the mirror-symmetric subspace.

    Returns
    -------
    cases_held :
        Indices into the built cases, in build order.

    Notes
    -----
    **A mirrored case is a reindexing, not a second condition.** Where every
    reachable design is mirror-symmetric — a symmetric geometry basis and
    sections folded by the same mirror — a case that is another case
    reflected produces the reflected response, so its utilization rows are
    the first case's rows under a permutation of the members. The feasible
    set is identical without them and the analysis they would have cost is
    saved at every iterate.

    Dropping rows is not the same as dropping the case: every answer is read
    and every feasibility check is made against all of them, so a claim of
    redundancy that is false shows up as a violated constraint rather than as
    a quiet omission.

    The rule declines wherever it cannot see that the reflection is exact —
    an asymmetric search, a mirror that does not preserve height, or a case
    with a horizontal component the reflection would turn.
    """
    cases = np.asarray(plan.cases.analysis)
    every = np.arange(cases.shape[0])
    if not symmetric:
        return every

    heights = np.asarray(structure.nodes)[:, 2]
    upright = bool(np.allclose(heights, heights[nodes_mirrored]))
    vertical = bool(np.allclose(cases[:, :, :2], 0.0))
    if not (upright and vertical):
        return every

    scale = float(np.max(np.abs(cases)))
    held = [0]
    for case in range(1, cases.shape[0]):
        reflected = cases[case][nodes_mirrored]
        gaps = [np.max(np.abs(reflected - cases[kept])) for kept in held]
        if min(gaps) > TOLERANCE_MIRRORED * scale:
            held.append(case)

    return np.asarray(held)


def prepare_problem(
    structure: Structure,
    config: TaskConfig,
    plan: LoadPlan,
    folding_by: FoldingMaps,
) -> RouteProblem:
    """
    The structure, its prepared blocks, and the searched basis.

    Parameters
    ----------
    structure :
        The structure the experiment built.
    config :
        The run description.
    plan :
        The load cases the profile built, already named.
    folding_by :
        Every permutation the run folds its variables by.

    Returns
    -------
    problem :
        Everything the routes read, gathered once on the host.

    Notes
    -----
    **The three kinds of variable need not fold by the same group.** The
    density basis is folded by the mirror alone, always, so the end-to-end
    route's dimension is a property of the structure. Sections may be folded
    as far as fabrication wants, carrying no argument with them. Heights are
    the delicate one: folded by the mirror alone they are a strict superset of
    what the form finder reaches, which is what makes a gap between the routes
    a statement about the landscape; folded polar they are a space of their own
    that neither contains nor is contained by it.

    **A section keeps its folding whether or not the search is symmetric.** A
    run may unfold the geometry to ask what the mirror was costing it and
    still want one diameter per ring, that being a decision about how the
    thing is built rather than about where the search may go. Only a family
    offering no rotation, in a run that asked for no symmetry, leaves the
    sections free member by member.
    """
    loads = plan.cases

    nodes_mirrored = folding_by.nodes_mirrored
    mirror = nodes_mirrored if config.subspace.symmetric else None
    if config.subspace.basis == "pivoted":
        pivot = pivoted_basis(structure, mirror)
        finder = SubspaceFormFinder(
            FdmFormFinder(structure), pivot.basis, pivot.independents
        )
    else:
        held = density_basis(structure, mirror)
        finder = SubspaceFormFinder(FdmFormFinder(structure), held)

    family = build_section_family(GRADE, SECTION_CLASS)
    blocks = StructuralDesignPipeline(
        finder,
        SmaxAnalyzer(structure, family(config.analysis.diameter)),
        Ec3Sizer(structure, family),
    )

    everyone = np.arange(structure.num_nodes)
    frees = np.setdiff1d(everyone, np.asarray(structure.supports))
    nodes_free = jnp.asarray(frees)

    # A rotation is offered only where the family's own switch asked for one,
    # which is fabrication speaking rather than the subspace.
    rotated = len(folding_by.members_folded) > 1
    if config.subspace.symmetric or rotated:
        spread_diameters = orbit_matrix(folding_by.members_folded)
        folded_diameters = jnp.asarray(spread_diameters)
    else:
        folded_diameters = None

    if config.subspace.symmetric:
        positions = {int(node): place for place, node in enumerate(frees.tolist())}
        among_free = [
            np.asarray([positions[int(nodes[node])] for node in frees.tolist()])
            for nodes in folding_by.nodes_folded
        ]
        spread_heights = orbit_matrix(tuple(among_free))
        folding = MirrorFolding(folded_diameters, jnp.asarray(spread_heights))
    else:
        folding = MirrorFolding(folded_diameters, None)

    held_cases = cases_constrained(
        plan, structure, nodes_mirrored, config.subspace.symmetric
    )
    diameters_seed = jnp.full(structure.num_edges, config.analysis.diameter)
    # The mirror comes first out of `folded_members`, and it is the one the
    # reported diameter gap is read against whatever else folds the sections.
    edges_mirrored = folding_by.members_folded[0]

    problem = RouteProblem(
        structure,
        blocks,
        loads,
        plan.names,
        held_cases,
        folding,
        edges_mirrored,
        nodes_free,
        diameters_seed,
    )

    return problem


class HeightTruss(NamedTuple):
    """
    The ceiling and the floor the shaped routes keep their heights inside.

    Attributes
    ----------
    ceiling :
        Height no vertex may rise above, or None to leave the rise free.
    floor :
        Height no vertex may hang under, or None to leave the sag free.

    Notes
    -----
    Each limit travels the way a route can carry it: a box bound where the
    heights are variables, one normalized inequality row per free node where
    they are outputs of the form finder. Neither is a box around the truss —
    either side may be off on its own.
    """

    ceiling: float | None
    floor: float | None


def height_scale(limits: HeightTruss) -> float:
    """
    Length the sag rows are normalized by, putting them at the utilization scale.

    Parameters
    ----------
    limits :
        The ceiling and the floor the shape is held between.

    Returns
    -------
    scale :
        The floor's own depth where it has one, and the ceiling otherwise.

    Notes
    -----
    A floor at zero is a real limit — no vertex below the plane of the
    supports — but it is its own distance from zero, so it can normalize
    nothing. The ceiling stands in, being the one other length the run states
    about heights. Where the floor is nonzero this is exactly the depth the
    rows were always divided by, so no truss's descent path moves.
    """
    if limits.floor:
        return abs(limits.floor)
    if limits.ceiling:
        return abs(limits.ceiling)

    return 1.0


def truss_heights(config: TaskConfig) -> HeightTruss:
    """
    The height limits a truss run keeps its vertices inside.

    Parameters
    ----------
    config :
        The run description, read for the switches, the factors and the depth.

    Returns
    -------
    limits :
        The ceiling above and the floor below, None where a side is off.
    """
    return height_truss(config.descent, config.structure.depth)


def shell_heights(config: TaskConfig) -> HeightTruss:
    """
    The height limits a shell run keeps its vertices inside.

    Parameters
    ----------
    config :
        The run description, read for the switches, the factors and the plan
        radius.

    Returns
    -------
    limits :
        The ceiling above and the floor below, None where a side is off.

    Notes
    -----
    **The plan radius is the reference, not the drawn rise.** A shell's height
    limits are the room it has to shelter, which is stated against what it
    spans rather than against how high it happens to have been drawn: the
    radius is half the span, so a `rise_factor` of one is a ceiling at half
    the span and stays that whatever rise the generator is given. A truss
    scales its limits by its drawn depth instead, having no span-like length
    of its own that a height should be read against.

    A `sag_factor` of zero puts the floor on the plane of the supports, which
    is the useful setting here: a shell that dips below its own supports is
    not sheltering anything, and forbidding it outright is cheaper than
    pricing it.
    """
    return height_truss(config.descent, config.structure.radius)


def height_truss(budget: DescentConfig, reference: float) -> HeightTruss:
    """
    The height limits a run keeps its vertices inside, read from the budgets.

    Parameters
    ----------
    budget :
        The budgets, read for the switches and the factors.
    reference :
        The drawn length both limits are multiples of.

    Returns
    -------
    limits :
        The ceiling above and the floor below, None where a side is off.
    """
    depth = reference
    if budget.limit_rise:
        ceiling = budget.rise_factor * depth
    else:
        ceiling = None
    if budget.limit_sag:
        floor = -budget.sag_factor * depth
    else:
        floor = None

    return HeightTruss(ceiling, floor)


def limit_label(limit: float | None, factor: float) -> str:
    """
    One height limit spelled for a report entry.

    Parameters
    ----------
    limit :
        The limit's height, or None when that side is free.
    factor :
        The limit as a multiple of the drawn depth.

    Returns
    -------
    label :
        The limit in millimeters and as its multiple, or `off`.
    """
    if limit is None:
        return "off"

    return f"{limit:.0f} mm, {factor:g}x the drawn depth"


def envelope_diameters(
    problem: RouteProblem,
    xyz: Float[Array, "nodes 3"],
    floor: float,
) -> Float[np.ndarray, "edges"]:
    """
    The frozen-seed envelope sections at one geometry, floored.

    Parameters
    ----------
    problem :
        The prepared truss.
    xyz :
        The geometry to seed a search at.
    floor :
        Smallest diameter any member may take.

    Returns
    -------
    diameters :
        One diameter per member, satisfying every case at the seed forces.

    Notes
    -----
    Frozen-seed on purpose: this is the classical design office move — analyze
    at a guess, size to the forces — and how infeasible it turns out to be
    once the frame is re-analyzed at these very sections is one of the
    numbers the experiments exist to print.
    """
    lengths = member_lengths(xyz, problem.structure.edges)
    seed = problem.diameters_seed
    forces = problem.pipeline.analyzer(xyz, seed, problem.loads.analysis)
    sizes = problem.pipeline.sizer(forces, lengths)
    design = Design(FormFoundShape(xyz, lengths), forces, sizes)
    sized = design_envelope(design, None)

    diameters = np.asarray(sized.sizes.sections.diameter)

    return np.maximum(diameters, floor)


def formfound_maps(
    problem: RouteProblem,
    limits: HeightTruss,
    length_floor: float,
    chord_signs: ChordSigns | None,
) -> RouteMaps:
    """
    The end-to-end route's compiled maps, over coordinates and diameters.

    Parameters
    ----------
    problem :
        The prepared truss.
    limits :
        The ceiling and the floor no vertex may leave.
    length_floor :
        Smallest length any member may keep, entering as inequality rows for
        the members whose held plan projection is under it.
    chord_signs :
        Signs the chord densities must keep, or None when the subspace has
        no degenerate states worth guarding.

    Returns
    -------
    maps :
        The mass with its gradient, the slack, the slack's Jacobian, and a
        repair that grows the diameters of a landing that missed feasibility.

    Notes
    -----
    **The repair grows diameters and moves no coordinate.** Only the
    utilization rows answer to a section; the height limits and the chord
    signs are functions of the geometry alone, so a landing that missed one of
    those is beyond repair and stays refused. Each pass grows a folded
    diameter by the square root of the worst utilization over the members it
    serves, which under-grows nothing — resistance rises at least as fast as
    the square of the diameter — and the passes repeat because a fatter member
    is stiffer and draws more force.

    The variable vector is the basis coordinates followed by every diameter,
    so the analysis runs at the search's own geometry and sections: the whole
    `∂N/∂ξ` and `∂N/∂d` feedback rides inside the gradient. Every geometry
    the search can reach holds the plan by construction — the coordinates
    span the null space of the horizontal balance, so no bound on them is a
    bound on funicularity.

    Here a height is an output of the form finder rather than a variable, so
    both height limits enter as one inequality row per free node — normalized
    by the limit, so they sit at the utilization rows' scale — where the
    free-heights route carries the same limits as plain box bounds. The
    chord signs enter the same way, one linear row per chord member, and so
    does the length floor: the signed funicular tends to keep members long
    on its own, but that is a tendency, and the floor makes it a constraint
    on the same members the free-heights route guards.
    """
    formfinder = problem.pipeline.formfinder
    analyzer = problem.pipeline.analyzer
    sizer = problem.pipeline.sizer
    family = sizer.family
    width = int(formfinder.basis.shape[1])
    spread = problem.folding.diameters
    held_cases = problem.loads.analysis[problem.cases_held]

    plan = np.asarray(problem.structure.nodes)[:, :2]
    edges = np.asarray(problem.structure.edges)
    spans_plan = np.linalg.norm(plan[edges[:, 1]] - plan[edges[:, 0]], axis=1)
    collapsible = np.flatnonzero(spans_plan < length_floor)

    def sized_members(x: Float[Array, "variables"]) -> Float[Array, "edges"]:
        if spread is None:
            return x[width:]
        return spread @ x[width:]

    def weigh(x: Float[Array, "variables"]) -> Float[Array, ""]:
        shape = formfinder(x[:width], problem.loads.formfinding)
        sections = family(sized_members(x))
        mass = jnp.sum(sections.area * shape.lengths) * family.material.density

        return mass

    def slack(x: Float[Array, "variables"]) -> Float[Array, "constraints"]:
        diameters = sized_members(x)
        shape = formfinder(x[:width], problem.loads.formfinding)
        forces = analyzer(shape.xyz, diameters, held_cases)
        used = sizer.compute_utilization(diameters, forces, shape.lengths)
        rows = [1.0 - used.ravel()]
        if limits.ceiling is not None:
            heights = shape.xyz[problem.nodes_free, 2]
            rows.append((limits.ceiling - heights) / limits.ceiling)
        if limits.floor is not None:
            heights = shape.xyz[problem.nodes_free, 2]
            rows.append((heights - limits.floor) / height_scale(limits))
        if collapsible.size:
            exposed = shape.lengths[collapsible]
            rows.append((exposed - length_floor) / length_floor)
        if chord_signs is not None:
            q = formfinder.member_densities(x[:width])
            signed = chord_signs.signs * q[chord_signs.chords]
            rows.append((signed - chord_signs.margin) / chord_signs.scale)

        return jnp.concatenate(rows)

    def grown(
        shape: FormFoundShape,
        folded: Float[Array, "patterns"],
    ) -> Float[Array, "patterns"]:
        diameters = spread @ folded if spread is not None else folded
        forces = analyzer(shape.xyz, diameters, held_cases)
        used = sizer.compute_utilization(diameters, forces, shape.lengths)
        worst = jnp.max(used, axis=0)
        if spread is None:
            demanded = worst
        else:
            masked = jnp.where(spread.T > 0.0, worst[None, :], 0.0)
            demanded = jnp.max(masked, axis=1)

        return folded * jnp.sqrt(jnp.maximum(demanded, 1.0))

    def repair(x: Float[Array, "variables"]) -> Float[Array, "variables"]:
        held = jnp.asarray(x)
        coordinates = held[:width]
        folded = held[width:]
        # The coordinates never move, so the shape is found once for them all.
        shape = formfinder(coordinates, problem.loads.formfinding)
        for _ in range(REPAIR_PASSES):
            folded = grown(shape, folded)

        return jnp.concatenate([coordinates, folded])

    maps = RouteMaps(
        jax.jit(jax.value_and_grad(weigh)),
        jax.jit(slack),
        jax.jit(jax.jacfwd(slack)),
        repair,
    )

    return maps


def heights_maps(problem: RouteProblem, length_floor: float) -> RouteMaps:
    """
    The free-heights route's compiled maps, over heights and diameters.

    Parameters
    ----------
    problem :
        The prepared truss, supplying the free nodes whose height moves.
    length_floor :
        Smallest length any member may keep, entering as inequality rows.

    Returns
    -------
    maps :
        The mass with its gradient, the slack, and the slack's Jacobian.

    Notes
    -----
    The pipeline minus its first block: the variable vector is the height of
    every free node followed by every diameter, the geometry is written down
    rather than form-found, and the same T2 and T3 run on it. The plan is
    held by never moving it, so no member can shorten past its own horizontal
    projection — but a member joining nodes of equal plan position, a
    Vierendeel vertical, can still be collapsed by a height crossing, which
    hands the analysis a singular frame. The length floor walls that off,
    and its rows exist only for the members whose held projection is under
    the floor: everywhere else the plan already enforces them, so trusses
    without such members run without the rows, untouched. Nothing here keeps
    an iterate funicular, and the heights answer to the analysis alone. The
    rise ceiling, when asked for, is the driver's business: heights are
    variables here, so it arrives as a box bound rather than as constraint
    rows.
    """
    analyzer = problem.pipeline.analyzer
    sizer = problem.pipeline.sizer
    family = sizer.family
    spread_heights = problem.folding.heights
    spread_diameters = problem.folding.diameters
    count = pattern_count(spread_heights, int(problem.nodes_free.shape[0]))
    held_cases = problem.loads.analysis[problem.cases_held]

    plan = np.asarray(problem.structure.nodes)[:, :2]
    edges = np.asarray(problem.structure.edges)
    spans_plan = np.linalg.norm(plan[edges[:, 1]] - plan[edges[:, 0]], axis=1)
    collapsible = np.flatnonzero(spans_plan < length_floor)

    def free_heights(x: Float[Array, "variables"]) -> Float[Array, "nodes_free"]:
        if spread_heights is None:
            return x[:count]
        return spread_heights @ x[:count]

    def sized_members(x: Float[Array, "variables"]) -> Float[Array, "edges"]:
        if spread_diameters is None:
            return x[count:]
        return spread_diameters @ x[count:]

    def written_shape(heights: Float[Array, "nodes_free"]) -> FormFoundShape:
        xyz = problem.structure.nodes.at[problem.nodes_free, 2].set(heights)
        lengths = member_lengths(xyz, problem.structure.edges)

        return FormFoundShape(xyz, lengths)

    def weigh(x: Float[Array, "variables"]) -> Float[Array, ""]:
        shape = written_shape(free_heights(x))
        sections = family(sized_members(x))

        return jnp.sum(sections.area * shape.lengths) * family.material.density

    def slack(x: Float[Array, "variables"]) -> Float[Array, "constraints"]:
        shape = written_shape(free_heights(x))
        diameters = sized_members(x)
        forces = analyzer(shape.xyz, diameters, held_cases)
        used = sizer.compute_utilization(diameters, forces, shape.lengths)
        rows = [1.0 - used.ravel()]
        if collapsible.size:
            exposed = shape.lengths[collapsible]
            rows.append((exposed - length_floor) / length_floor)

        return jnp.concatenate(rows)

    maps = RouteMaps(
        jax.jit(jax.value_and_grad(weigh)),
        jax.jit(slack),
        jax.jit(jax.jacfwd(slack)),
    )

    return maps


def drawn_maps(problem: RouteProblem) -> RouteMaps:
    """
    The sizing-only route's compiled maps, over the diameters alone.

    Parameters
    ----------
    problem :
        The prepared truss, supplying the drawn geometry that never moves.

    Returns
    -------
    maps :
        The mass with its gradient, the slack, and the slack's Jacobian.
    """
    analyzer = problem.pipeline.analyzer
    sizer = problem.pipeline.sizer
    family = sizer.family
    xyz = problem.structure.nodes
    lengths = member_lengths(xyz, problem.structure.edges)
    spread = problem.folding.diameters
    held_cases = problem.loads.analysis[problem.cases_held]

    def sized_members(x: Float[Array, "variables"]) -> Float[Array, "edges"]:
        if spread is None:
            return x
        return spread @ x

    def weigh(x: Float[Array, "variables"]) -> Float[Array, ""]:
        sections = family(sized_members(x))

        return jnp.sum(sections.area * lengths) * family.material.density

    def slack(x: Float[Array, "variables"]) -> Float[Array, "constraints"]:
        diameters = sized_members(x)
        forces = analyzer(xyz, diameters, held_cases)
        used = sizer.compute_utilization(diameters, forces, lengths)

        return 1.0 - used.ravel()

    maps = RouteMaps(
        jax.jit(jax.value_and_grad(weigh)),
        jax.jit(slack),
        jax.jit(jax.jacfwd(slack)),
    )

    return maps


def route_maps(
    problem: RouteProblem,
    limits: HeightTruss,
    length_floor: float,
    chord_signs: ChordSigns | None = None,
) -> dict[str, RouteMaps]:
    """
    Every route's compiled maps, keyed by route.

    Parameters
    ----------
    problem :
        The prepared truss.
    limits :
        The ceiling and the floor no vertex may leave.
    length_floor :
        Smallest length the shaped routes may draw any member at.
    chord_signs :
        Signs the end-to-end chord densities must keep, or None for none.

    Returns
    -------
    maps :
        The three routes' maps, in the shared route names.
    """
    maps = {
        ROUTE_FORMFOUND: formfound_maps(problem, limits, length_floor, chord_signs),
        ROUTE_HEIGHTS: heights_maps(problem, length_floor),
        ROUTE_DRAWN: drawn_maps(problem),
    }

    return maps


def route_starts(
    problem: RouteProblem,
    start: StartPoint,
    shape_xyz: Float[Array, "nodes 3"],
    floor: float,
) -> dict[str, Float[np.ndarray, "variables"]]:
    """
    Every route's starting variable vector, keyed by route.

    Parameters
    ----------
    problem :
        The prepared truss.
    start :
        The signed lens fit both shaped routes leave from.
    shape_xyz :
        The form-found lens geometry, sizing the shaped routes' seed.
    floor :
        Smallest diameter any member may take.

    Returns
    -------
    starts :
        The variable vectors, the two shaped routes matched to one geometry.
    """
    spread_diameters = problem.folding.diameters
    spread_heights = problem.folding.heights

    sized_found = envelope_diameters(problem, shape_xyz, floor)
    sized_drawn = envelope_diameters(problem, problem.structure.nodes, floor)
    d_found = folded_seed(sized_found, spread_diameters)
    d_drawn = folded_seed(sized_drawn, spread_diameters)

    x_found = np.concatenate([start.xi, d_found])
    z_full = np.asarray(start.lens)[np.asarray(problem.nodes_free), 2]
    z_start = folded_seed(z_full, spread_heights)
    x_heights = np.concatenate([z_start, d_found])

    starts = {
        ROUTE_FORMFOUND: x_found,
        ROUTE_HEIGHTS: x_heights,
        ROUTE_DRAWN: d_drawn,
    }

    return starts


def route_boxes(
    problem: RouteProblem,
    floor: float,
    limits: HeightTruss,
) -> dict[str, list[tuple[float | None, float | None]]]:
    """
    Every route's bound pairs, keyed by route.

    Parameters
    ----------
    problem :
        The prepared truss, supplying the variable counts.
    floor :
        Smallest diameter any member may take.
    limits :
        The ceiling and the floor boxing the free-heights variables.

    Returns
    -------
    boxes :
        One bound pair per variable, per route.
    """
    width = int(problem.pipeline.formfinder.basis.shape[1])
    count = pattern_count(problem.folding.heights, int(problem.nodes_free.shape[0]))
    members = pattern_count(problem.folding.diameters, problem.structure.num_edges)

    boxes = {
        ROUTE_FORMFOUND: [(None, None)] * width + [(floor, None)] * members,
        ROUTE_HEIGHTS: [(limits.floor, limits.ceiling)] * count
        + [(floor, None)] * members,
        ROUTE_DRAWN: [(floor, None)] * members,
    }

    return boxes


def route_variables(problem: RouteProblem) -> dict[str, int]:
    """
    Every route's variable count, keyed by route.

    Parameters
    ----------
    problem :
        The prepared truss, supplying the counts.

    Returns
    -------
    variables :
        Geometry variables plus diameters, per route.
    """
    width = int(problem.pipeline.formfinder.basis.shape[1])
    count = pattern_count(problem.folding.heights, int(problem.nodes_free.shape[0]))
    members = pattern_count(problem.folding.diameters, problem.structure.num_edges)

    variables = {
        ROUTE_FORMFOUND: width + members,
        ROUTE_HEIGHTS: count + members,
        ROUTE_DRAWN: members,
    }

    return variables


def seed_openings(
    maps: dict[str, RouteMaps],
    starts: dict[str, Float[np.ndarray, "variables"]],
) -> tuple[float, float]:
    """
    Smallest constraint slack of the lens seed and of the drawn seed.

    Parameters
    ----------
    maps :
        Every route's compiled maps.
    starts :
        Every route's starting variable vector.

    Returns
    -------
    opening_found :
        Smallest slack of the end-to-end seed, negative when infeasible.
    opening_drawn :
        Smallest slack of the sizing-only seed, negative when infeasible.
    """
    slack_found = maps[ROUTE_FORMFOUND].slack(jnp.asarray(starts[ROUTE_FORMFOUND]))
    slack_drawn = maps[ROUTE_DRAWN].slack(jnp.asarray(starts[ROUTE_DRAWN]))

    opening_found = float(np.min(np.asarray(slack_found)))
    opening_drawn = float(np.min(np.asarray(slack_drawn)))

    return opening_found, opening_drawn


class StartScatter(NamedTuple):
    """
    Where a multi-start descent leaves from, and what holds it in.

    Attributes
    ----------
    start :
        The nominal variable vector the scattered points are drawn around.
    boxes :
        One bound pair per variable.
    """

    start: Float[np.ndarray, "variables"]
    boxes: list[tuple[float | None, float | None]]


def descend_route(
    maps: RouteMaps,
    start: Float[np.ndarray, "variables"],
    boxes: list[tuple[float | None, float | None]],
    budget: DescentConfig,
) -> RouteAnswer:
    """
    SLSQP under hard `U <= 1`, restarted from its own answer until quiet.

    Parameters
    ----------
    maps :
        The route's compiled maps.
    start :
        The variable vector to leave from.
    boxes :
        One bound pair per variable.
    budget :
        Iterations per round, rounds, and the solver tolerance.

    Returns
    -------
    answer :
        The variables, the mass at every iterate, and how the solver ended.

    Notes
    -----
    Each restart hands SLSQP a fresh quadratic model at the previous answer,
    which is what moves it off the slow tail of a single long run; the loop
    stops the first time a round barely moves. The mass trajectory is read
    through the compiled objective at every iterate, one cheap extra
    evaluation against the figure it buys.

    **The objective is handed over divided by its value at the start, so the
    budget's tolerance is a relative one.** SLSQP tests `|f - f0| < acc` and
    `|s| < acc` against the same number, and the two quantities have no
    common scale: a mass in tonnes is a fraction of one while a step is taken
    over variables of order a hundred. Dividing the objective through makes
    the first test read as a share of the mass, which is what the number in a
    run description is meant to say, and leaves it meaning the same thing on
    a cap of any size.

    A threshold that is relative is not thereby a loose one. The descent has
    a long shallow tail and most of a stopping rule's cost is paid there, so
    the tolerance a run states is a decision about how much of that tail to
    buy, and the mass it settles at moves with it.

    A line-search trial point can leave the model's domain entirely: a
    geometry whose frame cannot be factorized raises from inside the compiled
    slack. Such a point is answered with a uniform, enormous violation
    instead — infeasible is the truthful reading of a structure that cannot
    stand — and the merit function walks the search back into the domain.
    Accepted iterates never sit there, so the Jacobian stays unguarded.
    """

    def weighed(x):
        value, slope = maps.weigh(jnp.asarray(x))

        return float(value), np.asarray(slope, dtype=np.float64)

    reference = abs(weighed(start)[0]) or 1.0

    def objective(x):
        value, slope = weighed(x)

        return value / reference, slope / reference

    def feasible(x):
        return np.asarray(maps.slack(jnp.asarray(x)), dtype=np.float64)

    def feasible_jacobian(x):
        return np.asarray(maps.jacobian(jnp.asarray(x)), dtype=np.float64)

    rows = feasible(start).size

    def guarded_slack(x):
        try:
            return feasible(x)
        except ValueError:
            return np.full(rows, -RECOIL_SLACK)

    masses = [weighed(start)[0]]

    def track(x):
        masses.append(weighed(x)[0])

    held = {"type": "ineq", "fun": guarded_slack, "jac": feasible_jacobian}
    options = {"maxiter": budget.iterations, "ftol": budget.tolerance}

    x = np.asarray(start, dtype=np.float64)
    spent = 0
    converged = False
    for _ in range(budget.rounds):
        found = minimize(
            objective,
            x,
            jac=True,
            method="SLSQP",
            bounds=boxes,
            constraints=[held],
            callback=track,
            options=options,
        )
        x = np.asarray(found.x)
        spent += int(found.nit)
        converged = found.status == 0
        if found.nit <= 1:
            break

    return RouteAnswer(x, np.asarray(masses), spent, converged)


def scattered_points(
    start: Float[np.ndarray, "variables"],
    boxes: list[tuple[float | None, float | None]],
    budget: DescentConfig,
) -> list[Float[np.ndarray, "variables"]]:
    """
    The nominal start, and however many scattered ones the run asks for.

    Parameters
    ----------
    start :
        The nominal starting variable vector.
    boxes :
        One bound pair per variable, which no scattered point may leave.
    budget :
        The budgets, read for the count and the spread.

    Returns
    -------
    points :
        The nominal point first, so a run asking for one start descends
        exactly what a single-start run would.

    Notes
    -----
    Scattered multiplicatively, each variable by its own value, because the
    coordinates and the diameters differ by orders of magnitude and one
    absolute spread would be a rounding error to one and a catastrophe to the
    other. The seed is fixed: a multi-start answer that cannot be reproduced
    is not a measurement.
    """
    points = [np.asarray(start, dtype=np.float64)]
    if budget.starts <= 1:
        return points

    lower = np.array([-np.inf if low is None else low for low, _ in boxes])
    upper = np.array([np.inf if high is None else high for _, high in boxes])
    stream = np.random.default_rng(SCATTER_SEED)

    for _ in range(budget.starts - 1):
        noise = stream.normal(0.0, budget.scatter, size=points[0].size)
        points.append(np.clip(points[0] * (1.0 + noise), lower, upper))

    return points


def descend_best(
    report: Report,
    maps: RouteMaps,
    seed: StartScatter,
    budget: DescentConfig,
) -> RouteAnswer:
    """
    Descend from every scattered start and keep the lightest feasible landing.

    Parameters
    ----------
    report :
        Where each start's landing is written as it happens.
    maps :
        The route's compiled maps.
    seed :
        The nominal variable vector to scatter around, and its bounds.
    budget :
        The budgets, read for the start count as well as the descent.

    Returns
    -------
    answer :
        The best landing's record. Its mass trajectory is that one descent's,
        so a figure drawn from it shows a real descent rather than a mixture.

    Notes
    -----
    A scattered start can leave the model's domain — a geometry whose frame
    will not factorize raises before the first quadratic model is built — so
    a start that cannot be evaluated is dropped rather than allowed to end
    the run. Landings are compared only among those that came back feasible,
    a search that lands outside the constraints having answered a different
    question.

    **Every start reports as it lands.** A multi-start descent is the longest
    thing a run does and the whole of it used to be one silent stretch, so a
    run that was going badly looked exactly like a run that was going well.
    The line says which start, what it reached and whether it was kept, which
    is also the record of how much the scattering earned.

    **A landing that missed feasibility is repaired before it is refused.** A
    descent that stops a fraction short of the constraints has still found a
    design, and throwing it away discards whatever basin it found along with
    the fraction it missed by; growing its diameters onto the constraint
    surface keeps the first and prices the second. A landing the repair cannot
    rescue — one that missed a height limit or a chord sign, neither of which a
    section answers to — is refused exactly as before, and the line says what
    it missed by either way.
    """
    start, boxes = seed
    best = None
    points = scattered_points(start, boxes, budget)
    for index, point in enumerate(points):
        named = "nominal" if index == 0 else f"scattered {index}"
        try:
            answer = descend_route(maps, point, boxes, budget)
            slack = float(np.min(np.asarray(maps.slack(jnp.asarray(answer.variables)))))
        except (ValueError, FloatingPointError):
            report.write_line(
                f"  start {index + 1}/{len(points)} ({named}): left the domain"
            )
            continue
        mended = ""
        if slack < SCATTER_SLACK and maps.repair is not None:
            missed = -slack
            grown = np.asarray(maps.repair(jnp.asarray(answer.variables)))
            slack_grown = float(np.min(np.asarray(maps.slack(jnp.asarray(grown)))))
            if slack_grown >= SCATTER_SLACK:
                mass, _ = maps.weigh(jnp.asarray(grown))
                trail = np.append(answer.masses, float(mass))
                answer = answer._replace(variables=grown, masses=trail)
                slack = slack_grown
                mended = f", missed by {missed:.2e} and repaired"
        if slack < SCATTER_SLACK:
            report.write_line(
                f"  start {index + 1}/{len(points)} ({named}): "
                f"{answer.masses[-1]:.6f} t, infeasible by {-slack:.2e}"
            )
            continue
        kept = best is None or answer.masses[-1] < best.masses[-1]
        report.write_line(
            f"  start {index + 1}/{len(points)} ({named}): {answer.masses[-1]:.6f} t "
            f"in {answer.iterations} iterations, "
            f"{'converged' if answer.converged else 'no convergence'}"
            f"{mended}{', best so far' if kept else ''}"
        )
        if kept:
            best = answer

    if best is None:
        raise ValueError("no scattered start reached a feasible landing")

    return best


def descent_digest(config: TaskConfig) -> str:
    """
    A fingerprint of everything about a run that decides where it lands.

    Parameters
    ----------
    config :
        The run description.

    Returns
    -------
    digest :
        Hexadecimal digest of the run description with the viewer removed.

    Notes
    -----
    **The viewer is left out on purpose.** Which route to draw, which case to
    draw it under, and whether to open a window at all decide nothing about
    the descent, and a stored answer that a change of camera invalidated would
    be worthless for the one thing it is for. Everything else is in: change a
    ring, a pressure, a bound or a budget and the stored answer stops being an
    answer to the question being asked.
    """
    described = repr(config._replace(viewer=None))

    return hashlib.sha256(described.encode()).hexdigest()


def answers_stored(config: TaskConfig) -> Path:
    """
    Where a run description's answers are kept.

    Parameters
    ----------
    config :
        The run description.

    Returns
    -------
    stored :
        The file the run's answers are written to and read back from.

    Notes
    -----
    **Named by what was descended, not by the file that asked for it.** Two
    descriptions that differ only in which route to draw, or whether to open a
    window, pose the identical question, and a store keyed by filename would
    make the second of them pay for the first one's answer all over again. The
    file that wrote it is kept inside for a reader to recognize it by.
    """
    return DESIGNS / f"{descent_digest(config)[:16]}.npz"


def save_answers(
    path: Path,
    config: TaskConfig,
    answers: dict[str, RouteAnswer],
) -> Path:
    """
    Write every descended answer beside the run description that reached it.

    Parameters
    ----------
    path :
        The YAML file describing the run.
    config :
        The run description, fingerprinted so a stale answer is not reused.
    answers :
        Each route's descent record, keyed by route.

    Returns
    -------
    stored :
        The file written.

    Notes
    -----
    **A solo run adds to the store rather than replacing it.** Descending one
    route says nothing about the other two, so an answer already held under
    the same fingerprint is kept and the descended routes are written over it.
    A fingerprint that does not match is a different question, and the store
    is begun again.

    Only the variables are needed to rebuild a design — every mass,
    utilization and diagram the report carries is recomputed from them — but
    the trajectory and the two things the solver said about how it stopped are
    written too, because a report that could not state them would be a
    different report from the one the descent wrote.
    """
    held = load_answers(config) or {}
    held.update(answers)

    stored = {"described": np.array(path.name)}
    for route, answer in held.items():
        stem = route.replace(" ", "-")
        stored[f"{stem}.variables"] = np.asarray(answer.variables)
        stored[f"{stem}.masses"] = np.asarray(answer.masses)
        stored[f"{stem}.iterations"] = np.array(answer.iterations)
        stored[f"{stem}.converged"] = np.array(answer.converged)

    target = answers_stored(config)
    np.savez(target, **stored)

    return target


def load_answers(config: TaskConfig) -> dict[str, RouteAnswer] | None:
    """
    Read back the answers this run description was already descended to.

    Parameters
    ----------
    config :
        The run description.

    Returns
    -------
    answers :
        Each stored route's descent record, or None where this description
        has not been descended.
    """
    target = answers_stored(config)
    if not target.exists():
        return None

    stored = np.load(target, allow_pickle=False)
    answers = {}
    for route in ROUTE_ORDER:
        stem = route.replace(" ", "-")
        if f"{stem}.variables" not in stored:
            continue
        answers[route] = RouteAnswer(
            stored[f"{stem}.variables"],
            stored[f"{stem}.masses"],
            int(stored[f"{stem}.iterations"]),
            bool(stored[f"{stem}.converged"]),
        )

    return answers or None


def descend_all(
    report: Report,
    maps: dict[str, RouteMaps],
    starts: dict[str, Float[np.ndarray, "variables"]],
    boxes: dict[str, list[tuple[float | None, float | None]]],
    budget: DescentConfig,
) -> dict[str, RouteAnswer]:
    """
    Descend every route in the shared order, reporting each landing.

    Parameters
    ----------
    report :
        Where each route's landing line is written.
    maps :
        Every route's compiled maps.
    starts :
        Every route's starting variable vector.
    boxes :
        Every route's bound pairs.
    budget :
        The budgets the routes share.

    Returns
    -------
    answers :
        Every route's descent record, keyed by route.
    """
    answers = {}
    for route in routes_present(starts):
        report.write_line(f"{route}, from {budget.starts} starts")
        seed = StartScatter(starts[route], boxes[route])
        answer = descend_best(report, maps[route], seed, budget)
        answers[route] = answer
        report.write_line(
            f"{route}: {answer.masses[-1]:.6f} t in {answer.iterations} iterations"
        )

    return answers


def shear_fraction(
    family: TubeFamily,
    sections: MemberSections,
    forces: MemberForces,
) -> float:
    """
    The design shear of the worst member, as a fraction of its plastic resistance.

    Parameters
    ----------
    family :
        The tube family the sections were drawn from, supplying the material.
    sections :
        The sections the answer settled on.
    forces :
        The analysis at that answer, carrying the shear the check leaves out.

    Returns
    -------
    fraction :
        Largest fraction over members and load cases.

    Notes
    -----
    Eq. 6.17 through `ec3x`, read once per component and taken at its worst
    rather than on a resultant of the two. The shear area of a tube is the same
    whichever way the force acts, which makes a resultant tempting, and whether
    one is sanctioned is an open question in that package — the worst component
    needs no ruling and is the same number wherever the other is zero, which on
    a planar frame it is.

    Read at the answer rather than bounded ahead of it: a bound over a demand
    mix describes a member that might exist, and what 6.2.10 asks about is the
    member that does.
    """
    steel = Steel(f_y=family.material.f_y, gamma_m0=GAMMA_M0_SHEAR)
    mobilized = area_shear(sections.area)

    major = np.asarray(utilization_shear(forces.shear_major, mobilized, steel))
    minor = np.asarray(utilization_shear(forces.shear_minor, mobilized, steel))

    return float(np.max(np.maximum(major, minor)))


def read_answer(
    problem: RouteProblem,
    xyz: Float[Array, "nodes 3"],
    diameters: Float[np.ndarray, "edges"],
    budget: DescentConfig,
) -> RouteRead:
    """
    One answer read back as a design, at its own geometry and sections.

    Parameters
    ----------
    problem :
        The prepared truss.
    xyz :
        The answer's geometry.
    diameters :
        The answer's sections.
    budget :
        The budgets, read for the diameter floor.

    Returns
    -------
    read :
        The mass, the shape extremes, and the utilization member by member.
    """
    family = problem.pipeline.sizer.family

    lengths = member_lengths(xyz, problem.structure.edges)
    sized = jnp.asarray(diameters)
    forces = problem.pipeline.analyzer(xyz, sized, problem.loads.analysis)
    used = problem.pipeline.sizer.compute_utilization(sized, forces, lengths)

    sections = family(sized)
    mass = float(jnp.sum(sections.area * lengths) * family.material.density)

    utilization = np.asarray(jnp.max(used, axis=0))
    utilization_cases = np.asarray(used)
    active = int(np.sum(utilization > ACTIVE_UTILIZATION))
    floored = int(np.sum(diameters < budget.diameter_floor + FLOOR_SLACK))

    reflected = diameters[problem.edges_mirrored]
    mirror = float(np.max(np.abs(diameters - reflected)) / np.max(diameters))

    shear = shear_fraction(family, sections, forces)

    read = RouteRead(
        mass,
        np.asarray(xyz),
        float(jnp.max(jnp.asarray(xyz)[:, 2])),
        float(jnp.min(jnp.asarray(xyz)[:, 2])),
        diameters,
        utilization,
        utilization_cases,
        active,
        floored,
        mirror,
        shear,
    )

    return read


def route_reads(
    problem: RouteProblem,
    answers: dict[str, RouteAnswer],
    budget: DescentConfig,
) -> dict[str, RouteRead]:
    """
    Every route's answer read back as a design, keyed by route.

    Parameters
    ----------
    problem :
        The prepared truss.
    answers :
        Every route's descent record.
    budget :
        The budgets, read for the diameter floor.

    Returns
    -------
    reads :
        Every answer at its own geometry and sections. A route the run never
        descended is absent rather than seeded, so every reader downstream
        sees the same routes the descent did.
    """
    width = int(problem.pipeline.formfinder.basis.shape[1])
    spread_heights = problem.folding.heights
    spread_diameters = problem.folding.diameters
    count = pattern_count(spread_heights, int(problem.nodes_free.shape[0]))

    reads = {}
    if ROUTE_FORMFOUND in answers:
        found = answers[ROUTE_FORMFOUND].variables
        xi_final = jnp.asarray(found[:width])
        shape_final = problem.pipeline.formfinder(xi_final, problem.loads.formfinding)
        d_found = unfolded_values(found[width:], spread_diameters)
        reads[ROUTE_FORMFOUND] = read_answer(problem, shape_final.xyz, d_found, budget)

    if ROUTE_HEIGHTS in answers:
        heights = answers[ROUTE_HEIGHTS].variables
        z_final = unfolded_values(heights[:count], spread_heights)
        xyz_heights = problem.structure.nodes.at[problem.nodes_free, 2].set(
            jnp.asarray(z_final)
        )
        d_heights = unfolded_values(heights[count:], spread_diameters)
        reads[ROUTE_HEIGHTS] = read_answer(problem, xyz_heights, d_heights, budget)

    if ROUTE_DRAWN in answers:
        d_drawn = unfolded_values(answers[ROUTE_DRAWN].variables, spread_diameters)
        drawn = problem.structure.nodes
        reads[ROUTE_DRAWN] = read_answer(problem, drawn, d_drawn, budget)

    return reads


def force_agreement(
    problem: RouteProblem,
    start: StartPoint,
    xyz: Float[Array, "nodes 3"],
) -> float:
    """
    How far the elastic axial forces sit from the funicular prediction.

    Parameters
    ----------
    problem :
        The prepared truss.
    start :
        The signed lens fit, supplying the funicular densities.
    xyz :
        The lens geometry the frame is analyzed at.

    Returns
    -------
    disagreement :
        Worst `|N - q L|` under the shaping case, scaled by the largest
        funicular force.

    Notes
    -----
    On the determinate arch this number sat at solver precision. On an
    indeterminate truss the elastic frame chooses its own load-path split,
    so the disagreement is structural rather than numerical — the reason T1
    hands T2 geometry only, never member forces.
    """
    lengths = member_lengths(xyz, problem.structure.edges)
    seed = problem.diameters_seed
    forces = problem.pipeline.analyzer(xyz, seed, problem.loads.analysis)

    funicular = start.q * np.asarray(lengths)
    elastic = np.asarray(forces.axial_force)[0]
    disagreement = float(np.abs(elastic - funicular).max() / np.abs(funicular).max())

    return disagreement


def start_entries(
    config: TaskConfig,
    problem: RouteProblem,
    start: StartPoint,
    measures: StartMeasures,
    limits: HeightTruss,
) -> list[tuple[str, str]]:
    """
    The start block's entries: the basis, the limits, and the seed numbers.

    Parameters
    ----------
    config :
        The run description.
    problem :
        The prepared truss, supplying the variable counts.
    start :
        The signed lens fit, read for the projection gap.
    measures :
        The seed numbers, measured before any descent has moved.
    limits :
        The ceiling and the floor no vertex may leave.

    Returns
    -------
    entries :
        Label-and-value pairs, for the experiment to extend and write.
    """
    width = int(problem.pipeline.formfinder.basis.shape[1])
    count = pattern_count(problem.folding.heights, int(problem.nodes_free.shape[0]))
    searched = "symmetric" if config.subspace.symmetric else "full"
    lidded = limit_label(limits.ceiling, config.descent.rise_factor)
    grounded = limit_label(limits.floor, config.descent.sag_factor)
    elastic = f"{measures.disagreement:.1%} of the largest force"

    entries = [
        ("searched basis", f"{searched} {config.subspace.basis}"),
        ("rise ceiling", lidded),
        ("sag floor", grounded),
        ("geometry variables, end to end", f"{width}"),
        ("geometry variables, free heights", f"{count}"),
        ("projection gap", f"{start.projection:.2e} of |q|"),
        ("lens reproduction [mm]", f"{measures.reproduction:.2e}"),
        ("elastic vs funicular, LC1", elastic),
        ("seed envelope infeasibility, lens", f"{-measures.opening_found:.1%}"),
        ("seed envelope infeasibility, drawn", f"{-measures.opening_drawn:.1%}"),
    ]

    return entries


def report_gradient(
    report: Report,
    maps: RouteMaps,
    start: Float[np.ndarray, "variables"],
    label: str,
) -> float:
    """
    One route's gradient against a directional central difference.

    Parameters
    ----------
    report :
        Where the sweep is written.
    maps :
        The route's compiled maps.
    start :
        The point the derivative is taken at.
    label :
        Name of the route, for the heading.

    Returns
    -------
    best :
        The smallest scaled disagreement over the swept steps.

    Notes
    -----
    One seeded random direction rather than the coordinate axes: the probe
    moves the geometry variables and the diameters at once, which is exactly
    the mixture SLSQP steps through — for the end-to-end route the form
    finder and the frame analysis inside the same derivative, for the
    free-heights route the coordinates straight into the analysis.
    """
    generator = np.random.default_rng(2026)
    drawn = generator.normal(size=start.shape[0])
    direction = drawn / np.linalg.norm(drawn)

    point = jnp.asarray(start)
    _, slope = maps.weigh(point)
    exact = float(jnp.sum(slope * jnp.asarray(direction)))
    magnitude = float(np.linalg.norm(start))

    columns = (
        ReportColumn("relative step", ".0e"),
        ReportColumn("central difference", ".9e"),
        ReportColumn("scaled error", ".2e"),
    )
    rows = []
    best = float("inf")
    for relative in GRADIENT_STEPS:
        step = magnitude * relative
        forward, _ = maps.weigh(point + step * jnp.asarray(direction))
        backward, _ = maps.weigh(point - step * jnp.asarray(direction))
        quotient = (float(forward) - float(backward)) / (2.0 * step)
        scaled = abs(exact - quotient) / abs(exact)
        best = min(best, scaled)
        rows.append((relative, quotient, scaled))

    entries = (
        ("exact directional derivative", f"{exact:.9e}"),
        ("best scaled error", f"{best:.2e} ({TOLERANCE_GRADIENT:.0e})"),
    )

    report.write_heading(f"The {label} gradient, checked at the start")
    report.write_table(columns, rows)
    report.write_entries(entries)

    return best


def report_routes(
    report: Report,
    reads: dict[str, RouteRead],
    answers: dict[str, RouteAnswer],
    variables: dict[str, int],
) -> None:
    """
    The routes side by side, by every measure the comparison makes.

    Parameters
    ----------
    report :
        Where the table is written.
    reads :
        Each route's answer read back, keyed by route.
    answers :
        Each route's descent record, keyed by route.
    variables :
        Each route's variable count, keyed by route.
    """
    columns = (
        ReportColumn("route", align="<"),
        ReportColumn("variables"),
        ReportColumn("iterations"),
        ReportColumn("converged", align="<"),
        ReportColumn("mass [t]", ".6f"),
        ReportColumn("rise [mm]", ".0f"),
        ReportColumn("sag [mm]", ".0f"),
        ReportColumn("max U", ".9f"),
        ReportColumn("fully stressed"),
        ReportColumn("at floor"),
    )
    rows = []
    for route in routes_present(reads):
        read = reads[route]
        rows.append(
            (
                route,
                variables[route],
                answers[route].iterations,
                "yes" if answers[route].converged else "NO",
                read.mass,
                read.rise,
                read.sag,
                float(read.utilization.max()),
                read.active,
                read.floored,
            )
        )

    report.write_heading("The routes, side by side")
    report.write_table(columns, rows)


def report_families(
    report: Report,
    reads: dict[str, RouteRead],
    families: tuple[tuple[str, slice], ...],
) -> None:
    """
    Sections and utilizations family by family, for every route.

    Parameters
    ----------
    report :
        Where the table is written.
    reads :
        Each route's answer read back, keyed by route.
    families :
        Name and member slice of every family, in the generator's order.
    """
    columns = (
        ReportColumn("route", align="<"),
        ReportColumn("family", align="<"),
        ReportColumn("d min [mm]", ".1f"),
        ReportColumn("d max [mm]", ".1f"),
        ReportColumn("U min", ".3f"),
        ReportColumn("U max", ".3f"),
    )
    rows = []
    for route in routes_present(reads):
        read = reads[route]
        for name, members in families:
            rows.append(
                (
                    route,
                    name,
                    float(read.diameters[members].min()),
                    float(read.diameters[members].max()),
                    float(read.utilization[members].min()),
                    float(read.utilization[members].max()),
                )
            )

    report.write_heading("Sections and utilization, family by family")
    report.write_table(columns, rows)


def governed_counts(read: RouteRead) -> Int[np.ndarray, "cases"]:
    """
    How many members each load case governs, ties counted toward each case.

    Parameters
    ----------
    read :
        One route's answer read back, supplying the utilization table.

    Returns
    -------
    counts :
        Members per case, a tied member appearing under each of its cases.

    Notes
    -----
    A member counts toward every case within the tie margin of its worst,
    so the counts may sum past the member count. Splitting ties by index
    order instead misreports a symmetric design: the mirror pairs the two
    half-span cases exactly on self-mirrored members and to solver
    precision everywhere else, and the first index would collect every
    such coin flip.
    """
    table = read.utilization_cases
    worst = table.max(axis=0)
    tied = table >= worst[None, :] - TIE_MARGIN

    return tied.sum(axis=1)


def report_governing(
    report: Report,
    reads: dict[str, RouteRead],
    names: tuple[str, ...],
) -> None:
    """
    How many members each load case governs, per route.

    Parameters
    ----------
    report :
        Where the table is written.
    reads :
        Each route's answer read back, keyed by route.
    names :
        Name of every built load case, in build order.
    """
    columns = [ReportColumn("route", align="<")]
    for name in names:
        columns.append(ReportColumn(name))

    rows = []
    for route in routes_present(reads):
        counts = governed_counts(reads[route])
        rows.append((route, *[int(count) for count in counts]))

    report.write_heading("Members governed, case by case")
    report.write_table(tuple(columns), rows)


def truss_extent(config: TaskConfig, read: RouteRead) -> tuple[str, str]:
    """
    The depth a truss reached, against the depth it was drawn at.

    Parameters
    ----------
    config :
        The run description, read for the drawn depth.
    read :
        The end-to-end answer, read for the rise and the sag it spans.

    Returns
    -------
    entry :
        The label and its value, ready for the summary block.
    """
    reached = read.rise - read.sag

    return (
        "depth at the answer [mm]",
        f"{reached:.0f}, drawn at {config.structure.depth:.0f}",
    )


def shell_extent(config: TaskConfig, read: RouteRead) -> tuple[str, str]:
    """
    The rise a shell reached, against the rise it was drawn at.

    Parameters
    ----------
    config :
        The run description, read for the drawn rise.
    read :
        The end-to-end answer, read for the height its crown reached.

    Returns
    -------
    entry :
        The label and its value, ready for the summary block.
    """
    return (
        "rise at the answer [mm]",
        f"{read.rise:.0f}, drawn at {config.structure.rise:.0f}",
    )


def report_summary(
    report: Report,
    reads: dict[str, RouteRead],
    config: TaskConfig,
    limits: HeightTruss,
    extent: Callable[[TaskConfig, RouteRead], tuple[str, str]],
) -> None:
    """
    The masses, the gaps between the routes, and the shape extremes.

    Parameters
    ----------
    report :
        Where the summary is written.
    reads :
        Each route's answer read back, keyed by route.
    config :
        The run description, read for the limit factors.
    limits :
        The ceiling and the floor no vertex may leave.
    extent :
        The profile's one entry saying how far the shape travelled, a truss
        reading it as a depth and a shell as a rise.
    """
    routes = routes_present(reads)
    lidded = limit_label(limits.ceiling, config.descent.rise_factor)
    grounded = limit_label(limits.floor, config.descent.sag_factor)

    entries = [(f"mass, {route}", f"{reads[route].mass:.6f} t") for route in routes]

    both = ROUTE_FORMFOUND in reads and ROUTE_DRAWN in reads
    if both:
        saving = 1.0 - reads[ROUTE_FORMFOUND].mass / reads[ROUTE_DRAWN].mass
        entries.append(("the geometry bought", f"{saving:.1%}"))

    shaped = ROUTE_FORMFOUND in reads and ROUTE_HEIGHTS in reads
    if shaped:
        read_found = reads[ROUTE_FORMFOUND]
        read_heights = reads[ROUTE_HEIGHTS]
        routes_gap = read_heights.mass / read_found.mass - 1.0
        heights_z = read_heights.xyz[:, 2]
        shapes_gap = float(np.abs(read_found.xyz[:, 2] - heights_z).max())
        entries.append(("free heights vs end to end", f"{routes_gap:+.2%}"))
        entries.append(("the shaped answers differ by [mm]", f"{shapes_gap:.0f}"))

    travelled = reads[ROUTE_FORMFOUND] if ROUTE_FORMFOUND in reads else reads[routes[0]]
    entries.append(extent(config, travelled))
    entries.append(("rise ceiling", lidded))
    entries.append(("sag floor", grounded))
    for route in routes:
        entries.append((f"diameter mirror gap, {route}", f"{reads[route].mirror:.2e}"))

    report.write_heading("Summary")
    report.write_entries(tuple(entries))


def route_checks(
    reads: dict[str, RouteRead],
    answers: dict[str, RouteAnswer],
    limits: HeightTruss,
) -> tuple[list[ToleranceCheck], bool]:
    """
    Every route's feasibility checks, and the converged-and-lighter verdict.

    Parameters
    ----------
    reads :
        Each route's answer read back, keyed by route.
    answers :
        Each route's descent record, keyed by route.
    limits :
        The ceiling and the floor no vertex may leave.

    Returns
    -------
    checks :
        Constraint, rise, sag and excluded-shear checks, one of each per route.
    sound :
        Whether every route converged and both shaped routes beat sizing.
    """
    routes = routes_present(reads)
    checks = []
    for route in routes:
        violation = max(0.0, float(reads[route].utilization.max()) - 1.0)
        checks.append(
            ToleranceCheck(
                f"{route} constraint violation", violation, TOLERANCE_FEASIBILITY
            )
        )
    shaped = [route for route in routes if route != ROUTE_DRAWN]
    if limits.ceiling is not None:
        for route in shaped:
            overrise = max(0.0, (reads[route].rise - limits.ceiling) / limits.ceiling)
            checks.append(
                ToleranceCheck(
                    f"{route} rise violation", overrise, TOLERANCE_FEASIBILITY
                )
            )
    if limits.floor is not None:
        for route in shaped:
            oversag = max(0.0, (limits.floor - reads[route].sag) / height_scale(limits))
            checks.append(
                ToleranceCheck(f"{route} sag violation", oversag, TOLERANCE_FEASIBILITY)
            )

    for route in routes:
        checks.append(
            ToleranceCheck(
                f"{route} shear fraction", reads[route].shear, SHEAR_THRESHOLD
            )
        )

    converged = all(answers[route].converged for route in routes)
    # A solo run has nothing to be lighter than, so it is judged on
    # convergence and feasibility alone.
    lighter = all(
        reads[route].mass < reads[ROUTE_DRAWN].mass
        for route in shaped
        if ROUTE_DRAWN in reads
    )
    sound = converged and lighter

    return checks, sound


def write_figures(
    problem: RouteProblem,
    reads: dict[str, RouteRead],
    answers: dict[str, RouteAnswer],
    prefix: str,
) -> None:
    """
    The three final designs, and the descents that reached them.

    Parameters
    ----------
    problem :
        The prepared truss, supplying the drawn geometry to outline.
    reads :
        Each route's answer read back, keyed by route.
    answers :
        Each route's descent record, keyed by route.
    prefix :
        Stem the figure files are named under, e.g. `18_warren`.
    """
    FIGURES.mkdir(exist_ok=True)

    forms = []
    for route in routes_present(reads):
        read = reads[route]
        title = f"{route} — {read.mass:.4f} t"
        counts = governed_counts(read)
        drawn = UtilizationForm(
            title, read.xyz, read.diameters, read.utilization, counts
        )
        forms.append(drawn)
    designs = figure_utilization(
        problem.structure.edges,
        forms,
        problem.case_names,
        reference=problem.structure.nodes,
    )
    designs.savefig(FIGURES / f"{prefix}_designs.png", dpi=200, bbox_inches="tight")

    variables = {
        ROUTE_FORMFOUND: "ξ and d",
        ROUTE_HEIGHTS: "z and d",
        ROUTE_DRAWN: "d",
    }
    traces = tuple(
        DescentTrace(f"{route} ({variables[route]})", answers[route].masses)
        for route in routes_present(answers)
    )
    descents = figure_mass_descent(traces)
    descents.savefig(FIGURES / f"{prefix}_descent.png", dpi=200, bbox_inches="tight")


def view_answers(
    problem: RouteProblem,
    reads: dict[str, RouteRead],
    routes: tuple[str, ...],
    viewer_config: ViewerConfig,
) -> None:
    """
    Open named answers in a viewer, in the frame solver's own terms.

    Parameters
    ----------
    problem :
        The prepared truss, supplying the connectivity, the blocks and the
        load cases every answer is drawn under.
    reads :
        Each route's answer read back, keyed by route.
    routes :
        Which routes to draw, each appearing under its own name.
    viewer_config :
        The run's viewer section, read for which case to draw and how far to
        exaggerate its load glyphs.

    Notes
    -----
    **A response per case is a response per solve.** Drawing one case is the
    difference between opening a scene and assembling the frame again for
    every condition it was checked against, and a reader comparing shapes
    rarely wants more than the one the shape was found under.

    The sections are the answer's own diameters walled by the family every
    block was built on, not a re-sizing: an envelope would replace them with
    the sizer's demand and draw a structure the report never mentioned.

    Each response comes from `SmaxAnalyzer.solve_response`, the same injected
    assembly and solve the member forces were read from, so the diagrams are
    the analysis rather than a retelling.

    Each response carries its displaced shape at true scale, which on a stiff
    truss is a shape a slider has to open up rather than one that reads off the
    screen unaided.

    Every registration is named apart. A viewer's `add` replaces a same-named
    one, so a loads group sharing its response's name would tear that response
    down instead of joining it. Each response also names its parent outright,
    which is required rather than tidy: the routes share a scene, so the frame
    a response belongs to is ambiguous otherwise.

    The caller names the routes, one or several. Several share a scene and
    nearly a location, so they are told apart by switching frames off from the
    panel rather than by looking.

    Support reactions are asked for by name and refused, so a design is read
    by what its members carry rather than by what its supports push back. The
    viewer registers the glyphs whatever it is told and only draws them when
    asked, so this pins the answer rather than skipping the work.

    Blocks until the window closes.
    """
    viewer = vix.Viewer(show_reactions=False)
    load_case = viewer_config.load_case

    for route in routes:
        read = reads[route]
        xyz = jnp.asarray(read.xyz)
        sections = problem.pipeline.sizer.family(jnp.asarray(read.diameters))
        frame = frame_model(problem.structure, xyz, sections)
        viewer.add(frame, name=route)

        for index, case_name in enumerate(problem.case_names):
            if not case_name.startswith(load_case):
                continue
            case_loads = problem.loads.analysis[index]
            response = problem.pipeline.analyzer.solve_response(
                xyz,
                sections.diameter,
                case_loads,
            )
            viewer.add(
                response,
                name=f"{route} — {case_name}",
                structure=route,
                show_deformation=True,
                show_forces=FORCE_DIAGRAMS,
            )

            loads_drawn = LoadCase.from_array(case_loads, frame)
            viewer.add(
                loads_drawn,
                name=f"{route} — {case_name} — loads at {viewer_config.load_scale:g}x",
                structure=route,
                load_scale=viewer_config.load_scale,
            )

    viewer.show()


class StiffnessSpectrum(NamedTuple):
    """
    The vertical stiffness the form finder solves, read at one density vector.

    Attributes
    ----------
    negatives :
        Count of negative eigenvalues — mixed member signs, not a defect.
    size :
        Count of eigenvalues, one per free node.
    condition :
        Ratio of the largest eigenvalue magnitude to the smallest. Explodes
        when the densities approach a degenerate state.
    """

    negatives: int
    size: int
    condition: float


def stiffness_spectrum(
    graph: EquilibriumStructure,
    q: Float[np.ndarray, "edges"],
) -> StiffnessSpectrum:
    """
    Sign count and conditioning of the vertical stiffness at one density.

    Parameters
    ----------
    graph :
        The form-finding connectivity.
    q :
        Force density of every edge.

    Returns
    -------
    spectrum :
        The negative-eigenvalue count and the condition number.
    """
    connectivity = np.asarray(graph.connectivity_free)
    stiffness = connectivity.T @ (q[:, None] * connectivity)
    eigen = np.linalg.eigvalsh(stiffness)

    negatives = int(np.sum(eigen < 0.0))
    condition = float(np.abs(eigen).max() / np.abs(eigen).min())

    return StiffnessSpectrum(negatives, eigen.size, condition)


class RouteProfile(NamedTuple):
    """
    What one structural family contributes to the shared three-route flow.

    Attributes
    ----------
    banner :
        Title the run's report opens with.
    prefix :
        Stem the run's figure files are named under.
    start_heading :
        Heading of the report's start block, carrying the family's story.
    parse_task :
        Reader of the run's YAML, the three family-shaped sections being the
        profile's to name and the other four `shared_sections`'.
    build_structure :
        The generator, taking the parsed run description.
    mirrored_nodes :
        The node the mirror carries each node onto.
    sections_rotated :
        The node a rotation carries each node onto, folding the diameters
        further, or None from a family with no rotation to offer and from a
        run that declines it.
    heights_rotated :
        The same, for the free-heights route's heights. Kept apart from
        `sections_rotated` because the two answer different questions —
        fabrication for the sections, and what the route comparison means for
        the heights — so a run may fold either without the other.
    member_families :
        Name and member slice of every family, in the generator's order.
    build_loads :
        The load cases, named, and the total the distributed ones carry.
    height_limits :
        The ceiling and floor the shape is held between, each family reading
        the multiple against the one length it is drawn by.
    signed_start :
        The start recipe — how the densities are fitted and signed differs
        per family, and each recipe is a measured decision.
    sign_guard :
        Builder of the density-sign rows, or None where the subspace has no
        degenerate states worth guarding. A builder may also answer None for
        a particular run, which is how a family makes its guard a switch.
    extent :
        The summary's one entry saying how far the shape travelled.

    Notes
    -----
    Everything else is topology-blind and lives in `run_routes`: an
    experiment is this record, a module docstring telling the family's
    story, and a YAML naming the run.
    """

    banner: str
    prefix: str
    start_heading: str
    parse_task: Callable[[str], TaskConfig]
    build_structure: Callable[[TaskConfig], Structure]
    mirrored_nodes: Callable[[TaskConfig], Int[np.ndarray, "nodes"]]
    sections_rotated: Callable[[TaskConfig], Int[np.ndarray, "nodes"] | None] | None
    heights_rotated: Callable[[TaskConfig], Int[np.ndarray, "nodes"] | None] | None
    member_families: Callable[[TaskConfig], tuple[tuple[str, slice], ...]]
    build_loads: Callable[[Structure, TaskConfig], LoadPlan]
    height_limits: Callable[[TaskConfig], HeightTruss]
    signed_start: Callable[[RouteProblem, TaskConfig], StartPoint]
    sign_guard: Callable[[TaskConfig, StartPoint], ChordSigns | None] | None
    extent: Callable[[TaskConfig, RouteRead], tuple[str, str]]


def run_routes(profile: RouteProfile, path: Path) -> None:
    """
    Run one structure's three routes, write the report, and save the figures.

    Parameters
    ----------
    profile :
        The structural family to run.
    path :
        The YAML file describing the run.

    Notes
    -----
    A file asking for `solo_route` descends the viewer's route alone. Every
    table then holds one row, every comparison entry is dropped, and the
    verdict rests on convergence and feasibility rather than on beating a
    baseline that was never descended.
    """
    report = Report()
    report.write_banner(profile.banner)

    config = profile.parse_task(path.read_text())
    budget = config.descent

    structure = profile.build_structure(config)
    graph = equilibrium_graph(structure)
    plan = profile.build_loads(structure, config)
    folding_by = folding_maps(profile, config, structure)
    problem = prepare_problem(structure, config, plan, folding_by)

    start = profile.signed_start(problem, config)
    finder = problem.pipeline.formfinder
    shape = finder.formfinder(jnp.asarray(start.q), problem.loads.formfinding)
    reproduction = float(jnp.max(jnp.abs(shape.xyz - jnp.asarray(start.lens))))
    disagreement = force_agreement(problem, start, shape.xyz)

    if profile.sign_guard is None:
        guard = None
    else:
        guard = profile.sign_guard(config, start)

    limits = profile.height_limits(config)
    maps = route_maps(problem, limits, budget.length_floor, guard)
    starts = route_starts(problem, start, shape.xyz, budget.diameter_floor)
    opening_found, opening_drawn = seed_openings(maps, starts)
    measures = StartMeasures(reproduction, disagreement, opening_found, opening_drawn)

    fit_scaled = start.gap / plan.total
    opening = stiffness_spectrum(graph, start.q)

    report.write_heading(profile.start_heading)
    entries = start_entries(config, problem, start, measures, limits)
    entries.append(("lens fit gap / total load", f"{fit_scaled:.2e}"))
    held = problem.cases_held
    built = len(problem.case_names)
    if held.size < built:
        reindexed = [
            name for index, name in enumerate(problem.case_names) if index not in held
        ]
        mirrored = ", ".join(reindexed)
        entries.append(
            (
                "load cases with rows",
                f"{held.size} of {built}, {mirrored} reindexed onto a held case",
            )
        )
    if guard is not None:
        entries.append(
            (
                "chord sign margin [N/mm]",
                f"{guard.margin:.1f} on {guard.chords.size} chords, linear rows",
            )
        )
    report.write_entries(tuple(entries))

    if config.viewer.solo_route:
        routes = (config.viewer.route,)
    else:
        routes = ROUTE_ORDER

    shaped = [route for route in routes if route != ROUTE_DRAWN]
    errors = [
        report_gradient(report, maps[route], starts[route], route) for route in shaped
    ]
    best_error = max(errors) if errors else 0.0

    held = load_answers(config) if budget.reuse_answers else None
    recalled = held is not None and all(route in held for route in routes)

    named = "the three routes" if len(routes) > 1 else routes[0]
    boxes = route_boxes(problem, budget.diameter_floor, limits)
    if recalled:
        report.write_heading(f"Reading {named} back")
        answers = {route: held[route] for route in routes}
        for route in routes_present(answers):
            answer = answers[route]
            report.write_line(
                f"{route}: {answer.masses[-1]:.6f} t "
                f"in {answer.iterations} iterations, descended earlier"
            )
    else:
        report.write_heading(f"Descending {named}")
        descended = {route: maps[route] for route in routes}
        seeds = {route: starts[route] for route in routes}
        bounds = {route: boxes[route] for route in routes}
        answers = descend_all(report, descended, seeds, bounds, budget)
        save_answers(path, config, answers)

    reads = route_reads(problem, answers, budget)
    report_routes(report, reads, answers, route_variables(problem))
    report_families(report, reads, profile.member_families(config))
    report_governing(report, reads, problem.case_names)

    width = int(finder.basis.shape[1])
    # The stiffness at the landing and the sign slack are the form finder's
    # own readouts, so a run without that route reports them at the start.
    if ROUTE_FORMFOUND in answers:
        q_final = np.asarray(finder.basis) @ answers[ROUTE_FORMFOUND].variables[:width]
    else:
        q_final = start.q
    landing = stiffness_spectrum(graph, q_final)

    shortest = {}
    for route in shaped:
        xyz_route = jnp.asarray(reads[route].xyz)
        lengths_route = member_lengths(xyz_route, problem.structure.edges)
        shortest[route] = float(jnp.min(lengths_route))

    report.write_heading("The degeneracies, watched at the answers")
    entries = [
        (
            "vertical stiffness at the start",
            f"{opening.negatives} negative of {opening.size}, "
            f"cond {opening.condition:.1e}",
        ),
        (
            "vertical stiffness at the answer",
            f"{landing.negatives} negative of {landing.size}, "
            f"cond {landing.condition:.1e}",
        ),
    ]
    for route in shaped:
        entries.append(
            (
                f"shortest member, {route} [mm]",
                f"{shortest[route]:.0f} against the {budget.length_floor:.0f} floor",
            )
        )
    if guard is not None:
        signed_final = float(np.min(guard.signs * q_final[guard.chords]))
        entries.append(
            (
                "chord sign slack at the answer [N/mm]",
                f"{signed_final:.1f} against the {guard.margin:.1f} margin",
            )
        )
    report.write_entries(tuple(entries))

    report_summary(report, reads, config, limits, profile.extent)

    write_figures(problem, reads, answers, profile.prefix)
    report.write_heading(f"figures written to {FIGURES}")

    checks = [
        ToleranceCheck("gradient scaled error", best_error, TOLERANCE_GRADIENT),
        ToleranceCheck("projection gap", start.projection, TOLERANCE_PROJECTION),
        ToleranceCheck("lens reproduction [mm]", reproduction, TOLERANCE_SHAPE),
        ToleranceCheck("lens fit gap / total load", fit_scaled, TOLERANCE_FIT),
    ]
    # A held plan already floors every length at its own plan projection, so a
    # family that needs no floor of its own states zero and is not checked
    # against it.
    floor = budget.length_floor
    if floor > 0.0:
        for route in shaped:
            undershort = max(0.0, (floor - shortest[route]) / floor)
            checks.append(
                ToleranceCheck(
                    f"{route} length violation", undershort, TOLERANCE_FEASIBILITY
                )
            )
    if guard is not None:
        undersign = max(0.0, (guard.margin - signed_final) / guard.scale)
        checks.append(
            ToleranceCheck("chord sign violation", undersign, TOLERANCE_FEASIBILITY)
        )
    routed, sound = route_checks(reads, answers, limits)
    checks.extend(routed)
    passed = checks_passed(tuple(checks)) and sound

    report.write_checks(tuple(checks))
    report.write_verdict(passed)

    # Last, because the window holds the process until it closes.
    if config.viewer.enabled:
        view_answers(problem, reads, (config.viewer.route,), config.viewer)
