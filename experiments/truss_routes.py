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

from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int
from scipy.optimize import minimize

from normax.analysis.smax import SmaxAnalyzer
from normax.design import Design
from normax.design import StructuralDesignPipeline
from normax.design import design_envelope
from normax.form_finding import FormFoundShape
from normax.form_finding.fdm import FdmFormFinder
from normax.form_finding.fdm import SubspaceFormFinder
from normax.form_finding.fdm import density_basis
from normax.form_finding.fdm import pivoted_basis
from normax.loads import LoadCases
from normax.loads import assemble_load_cases
from normax.loads import loads_point
from normax.materials import Steel355
from normax.reporting import Report
from normax.reporting import ReportColumn
from normax.reporting import ToleranceCheck
from normax.sizing.ec3 import Ec3Sizer
from normax.sizing.ec3 import thinnest_family
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

# A member is counted fully stressed above this envelope utilization, and
# counted at the floor within this distance of the bound.
ACTIVE_UTILIZATION = 0.999
FLOOR_SLACK = 1e-6

# Violation a trial point is charged when its frame cannot be factorized —
# enormous against the order-one slack rows, so the line search recoils.
RECOIL_SLACK = 1e3

# A member is governed by every case within this distance of its worst:
# mirror-paired cases tie exactly on self-mirrored members, and splitting a
# tie by index order would misreport a symmetric design as lopsided.
TIE_MARGIN = 1e-9

FIGURES = Path(__file__).resolve().parent.parent / "figures"

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


class TaskConfig(NamedTuple):
    """
    Everything a run is described by.

    Attributes
    ----------
    structure :
        The truss to build.
    loads :
        The load every case carries.
    sketch :
        The lens the end-to-end route starts from.
    subspace :
        Which held-plan basis the geometry variables span.
    analysis :
        What the frame is seeded with.
    descent :
        The budgets the searches share.
    """

    structure: TrussConfig
    loads: LoadConfig
    sketch: SketchConfig
    subspace: SubspaceConfig
    analysis: AnalysisConfig
    descent: DescentConfig


def parse_config(text: str) -> TaskConfig:
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
    TypeError
        If the text names a field that does not exist, or omits one that does.

    Notes
    -----
    No container carries a default, so a file missing a field is refused
    rather than quietly completed.
    """
    document = yaml.safe_load(text)

    config = TaskConfig(
        structure=TrussConfig(**document["structure"]),
        loads=LoadConfig(**document["loads"]),
        sketch=SketchConfig(**document["sketch"]),
        subspace=SubspaceConfig(**document["subspace"]),
        analysis=AnalysisConfig(**document["analysis"]),
        descent=DescentConfig(**document["descent"]),
    )
    if config.subspace.basis not in ("svd", "pivoted"):
        raise ValueError(f"basis must be svd or pivoted, got {config.subspace.basis}")
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
    """

    weigh: object
    slack: object
    jacobian: object


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
            loads_point(structure, float(load), node=int(node))
            for node, load in zip(interior, scaled)
        ]

        return jnp.sum(jnp.stack(cases), axis=0)

    uniform = deck_case(np.ones(interior.size))
    near = deck_case(np.where(along <= middle, 1.0, weight.half_factor))
    concentrated = weight.total * weight.point_factor
    point = loads_point(structure, concentrated, node=num_bays // 2)
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
    edges = np.asarray(structure.edges)
    ordered = np.sort(edges, axis=1)
    reflected = np.sort(nodes_mirrored[edges], axis=1)

    lookup = {tuple(pair): index for index, pair in enumerate(ordered.tolist())}
    targets = [lookup[tuple(pair)] for pair in reflected.tolist()]

    return np.asarray(targets)


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


def prepare_problem(
    structure: Structure,
    config: TaskConfig,
    nodes_mirrored: Int[np.ndarray, "nodes"],
    edges_mirrored: Int[np.ndarray, "edges"],
) -> RouteProblem:
    """
    The truss, its prepared blocks, and the searched basis.

    Parameters
    ----------
    structure :
        The truss the experiment built.
    config :
        The run description.
    nodes_mirrored :
        The node the midspan mirror carries each node onto.
    edges_mirrored :
        The member the midspan mirror carries each member onto.

    Returns
    -------
    problem :
        Everything the routes read, gathered once on the host.
    """
    loads = build_load_cases(structure, config.loads, config.structure.num_bays)

    mirror = nodes_mirrored if config.subspace.symmetric else None
    if config.subspace.basis == "pivoted":
        pivot = pivoted_basis(structure, mirror)
        finder = SubspaceFormFinder(
            FdmFormFinder(structure), pivot.basis, pivot.independents
        )
    else:
        held = density_basis(structure, mirror)
        finder = SubspaceFormFinder(FdmFormFinder(structure), held)

    family = thinnest_family(GRADE, SECTION_CLASS)
    blocks = StructuralDesignPipeline(
        finder,
        SmaxAnalyzer(structure, family(config.analysis.diameter)),
        Ec3Sizer(structure, family),
    )

    everyone = np.arange(structure.num_nodes)
    frees = np.setdiff1d(everyone, np.asarray(structure.supports))
    nodes_free = jnp.asarray(frees)

    if config.subspace.symmetric:
        positions = {int(node): place for place, node in enumerate(frees.tolist())}
        partners = [positions[int(nodes_mirrored[node])] for node in frees.tolist()]
        spread_diameters = folding_matrix(edges_mirrored)
        spread_heights = folding_matrix(np.asarray(partners))
        folding = MirrorFolding(
            jnp.asarray(spread_diameters), jnp.asarray(spread_heights)
        )
    else:
        folding = MirrorFolding(None, None)

    diameters_seed = jnp.full(structure.num_edges, config.analysis.diameter)

    problem = RouteProblem(
        structure,
        blocks,
        loads,
        load_names(config.loads),
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


def height_truss(budget: DescentConfig, depth: float) -> HeightTruss:
    """
    The height limits a run keeps its trusses inside, read from the budgets.

    Parameters
    ----------
    budget :
        The budgets, read for the switches and the factors.
    depth :
        The drawn depth both limits are multiples of.

    Returns
    -------
    limits :
        The ceiling above and the floor below, None where a side is off.
    """
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
        The mass with its gradient, the slack, and the slack's Jacobian.

    Notes
    -----
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
        forces = analyzer(shape.xyz, diameters, problem.loads.analysis)
        used = sizer.compute_utilization(diameters, forces, shape.lengths)
        rows = [1.0 - used.ravel()]
        if limits.ceiling is not None:
            heights = shape.xyz[problem.nodes_free, 2]
            rows.append((limits.ceiling - heights) / limits.ceiling)
        if limits.floor is not None:
            heights = shape.xyz[problem.nodes_free, 2]
            rows.append((heights - limits.floor) / -limits.floor)
        if collapsible.size:
            exposed = shape.lengths[collapsible]
            rows.append((exposed - length_floor) / length_floor)
        if chord_signs is not None:
            q = formfinder.member_densities(x[:width])
            signed = chord_signs.signs * q[chord_signs.chords]
            rows.append((signed - chord_signs.margin) / chord_signs.scale)

        return jnp.concatenate(rows)

    maps = RouteMaps(
        jax.jit(jax.value_and_grad(weigh)),
        jax.jit(slack),
        jax.jit(jax.jacfwd(slack)),
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
        forces = analyzer(shape.xyz, diameters, problem.loads.analysis)
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

    def sized_members(x: Float[Array, "variables"]) -> Float[Array, "edges"]:
        if spread is None:
            return x
        return spread @ x

    def weigh(x: Float[Array, "variables"]) -> Float[Array, ""]:
        sections = family(sized_members(x))

        return jnp.sum(sections.area * lengths) * family.material.density

    def slack(x: Float[Array, "variables"]) -> Float[Array, "constraints"]:
        diameters = sized_members(x)
        forces = analyzer(xyz, diameters, problem.loads.analysis)
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

    A line-search trial point can leave the model's domain entirely: a
    geometry whose frame cannot be factorized raises from inside the compiled
    slack. Such a point is answered with a uniform, enormous violation
    instead — infeasible is the truthful reading of a structure that cannot
    stand — and the merit function walks the search back into the domain.
    Accepted iterates never sit there, so the Jacobian stays unguarded.
    """

    def objective(x):
        value, slope = maps.weigh(jnp.asarray(x))

        return float(value), np.asarray(slope, dtype=np.float64)

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

    masses = [objective(start)[0]]

    def track(x):
        masses.append(objective(x)[0])

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
    for route in ROUTE_ORDER:
        answer = descend_route(maps[route], starts[route], boxes[route], budget)
        answers[route] = answer
        report.write_line(
            f"{route}: {answer.masses[-1]:.6f} t in {answer.iterations} iterations"
        )

    return answers


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
        Every answer at its own geometry and sections.
    """
    width = int(problem.pipeline.formfinder.basis.shape[1])
    spread_heights = problem.folding.heights
    spread_diameters = problem.folding.diameters
    count = pattern_count(spread_heights, int(problem.nodes_free.shape[0]))

    xi_final = jnp.asarray(answers[ROUTE_FORMFOUND].variables[:width])
    shape_final = problem.pipeline.formfinder(xi_final, problem.loads.formfinding)

    z_final = unfolded_values(answers[ROUTE_HEIGHTS].variables[:count], spread_heights)
    xyz_heights = problem.structure.nodes.at[problem.nodes_free, 2].set(
        jnp.asarray(z_final)
    )

    d_found = unfolded_values(
        answers[ROUTE_FORMFOUND].variables[width:], spread_diameters
    )
    d_heights = unfolded_values(
        answers[ROUTE_HEIGHTS].variables[count:], spread_diameters
    )
    d_drawn = unfolded_values(answers[ROUTE_DRAWN].variables, spread_diameters)

    read_found = read_answer(problem, shape_final.xyz, d_found, budget)
    read_heights = read_answer(problem, xyz_heights, d_heights, budget)
    read_drawn = read_answer(problem, problem.structure.nodes, d_drawn, budget)

    reads = {
        ROUTE_FORMFOUND: read_found,
        ROUTE_HEIGHTS: read_heights,
        ROUTE_DRAWN: read_drawn,
    }

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
        ReportColumn("mass [t]", ".6f"),
        ReportColumn("rise [mm]", ".0f"),
        ReportColumn("sag [mm]", ".0f"),
        ReportColumn("max U", ".9f"),
        ReportColumn("fully stressed"),
        ReportColumn("at floor"),
    )
    rows = []
    for route in ROUTE_ORDER:
        read = reads[route]
        rows.append(
            (
                route,
                variables[route],
                answers[route].iterations,
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
    for route in ROUTE_ORDER:
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
    for route in ROUTE_ORDER:
        counts = governed_counts(reads[route])
        rows.append((route, *[int(count) for count in counts]))

    report.write_heading("Members governed, case by case")
    report.write_table(tuple(columns), rows)


def report_summary(
    report: Report,
    reads: dict[str, RouteRead],
    config: TaskConfig,
    limits: HeightTruss,
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
        The run description, read for the drawn depth and the limit factors.
    limits :
        The ceiling and the floor no vertex may leave.
    """
    read_found = reads[ROUTE_FORMFOUND]
    read_heights = reads[ROUTE_HEIGHTS]
    read_drawn = reads[ROUTE_DRAWN]

    saving = 1.0 - read_found.mass / read_drawn.mass
    depth_found = read_found.rise - read_found.sag
    routes_gap = read_heights.mass / read_found.mass - 1.0
    shapes_gap = float(np.abs(read_found.xyz[:, 2] - read_heights.xyz[:, 2]).max())
    lidded = limit_label(limits.ceiling, config.descent.rise_factor)
    grounded = limit_label(limits.floor, config.descent.sag_factor)

    entries = (
        ("mass, end to end", f"{read_found.mass:.6f} t"),
        ("mass, free heights", f"{read_heights.mass:.6f} t"),
        ("mass, sizing only", f"{read_drawn.mass:.6f} t"),
        ("the geometry bought", f"{saving:.1%}"),
        ("free heights vs end to end", f"{routes_gap:+.2%}"),
        ("the shaped answers differ by [mm]", f"{shapes_gap:.0f}"),
        (
            "depth at the answer [mm]",
            f"{depth_found:.0f}, drawn at {config.structure.depth:.0f}",
        ),
        ("rise ceiling", lidded),
        ("sag floor", grounded),
        ("diameter mirror gap, end to end", f"{read_found.mirror:.2e}"),
        ("diameter mirror gap, free heights", f"{read_heights.mirror:.2e}"),
        ("diameter mirror gap, sizing only", f"{read_drawn.mirror:.2e}"),
    )
    report.write_heading("Summary")
    report.write_entries(entries)


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
        Constraint, rise and sag violations, one check per route.
    sound :
        Whether every route converged and both shaped routes beat sizing.
    """
    checks = []
    for route in ROUTE_ORDER:
        violation = max(0.0, float(reads[route].utilization.max()) - 1.0)
        checks.append(
            ToleranceCheck(
                f"{route} constraint violation", violation, TOLERANCE_FEASIBILITY
            )
        )
    if limits.ceiling is not None:
        for route in (ROUTE_FORMFOUND, ROUTE_HEIGHTS):
            overrise = max(0.0, (reads[route].rise - limits.ceiling) / limits.ceiling)
            checks.append(
                ToleranceCheck(
                    f"{route} rise violation", overrise, TOLERANCE_FEASIBILITY
                )
            )
    if limits.floor is not None:
        for route in (ROUTE_FORMFOUND, ROUTE_HEIGHTS):
            oversag = max(0.0, (limits.floor - reads[route].sag) / -limits.floor)
            checks.append(
                ToleranceCheck(f"{route} sag violation", oversag, TOLERANCE_FEASIBILITY)
            )

    converged = all(answers[route].converged for route in ROUTE_ORDER)
    lighter = reads[ROUTE_FORMFOUND].mass < reads[ROUTE_DRAWN].mass
    lighter = lighter and reads[ROUTE_HEIGHTS].mass < reads[ROUTE_DRAWN].mass
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
    for route in ROUTE_ORDER:
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

    traces = (
        DescentTrace(f"{ROUTE_FORMFOUND} (ξ and d)", answers[ROUTE_FORMFOUND].masses),
        DescentTrace(f"{ROUTE_HEIGHTS} (z and d)", answers[ROUTE_HEIGHTS].masses),
        DescentTrace(f"{ROUTE_DRAWN} (d)", answers[ROUTE_DRAWN].masses),
    )
    descents = figure_mass_descent(traces)
    descents.savefig(FIGURES / f"{prefix}_descent.png", dpi=200, bbox_inches="tight")
