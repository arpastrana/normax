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
Held-plan form finding on a Warren truss, and the subspace it happens in.

Experiment 15 proved the held plan degenerate on a chain: horizontal balance
at a two-edge node collapses to `q_after = q_before`, so the funicular part of
a fixed plan is one parameter wide and an optimizer given twenty is given
nineteen ways to leave funicularity. This experiment runs the same counting on
a truss and gets the opposite verdict. Diagonals add members faster than they
add balance rows, so the eight-bay Warren truss holds a sixteen-wide subspace
of force densities that keep its plan — every bay staying even and every
diagonal aligned by construction rather than by penalty, while the chords bend
into a lens. The count is exact bookkeeping: sixteen is the fifteen free
heights plus one state of self-stress, because on this topology every held-plan
geometry is reachable and one density direction moves forces without moving a
node.

Five things are put in front of the eye.

    forms       a drawn lens and a drawn flat-deck truss, each handed exact
                densities by a least-squares fit of the balance — the fit
                shifted along the self-stress until the bottom chord is all
                tension and the top chord all compression
    modes       every direction of the searched subspace, split by the height
                Jacobian into shape motions ordered by how far they move the
                truss, and one blind direction — the self-stress, recolored
                forces on an unmoved geometry
    variations  the same directions stepped for real: each one handed back to
                the vertical solve and drawn with the forces it carries
    pivoted     the variations retold in member coordinates: QR pivoting
                elects the independent edges the TNA way, and each panel
                nudges one named member's density and lets the transfer fill
                in the rest
    free plan   the same density families handed to the unrestrained solver,
                where the bays drift and the shape leaves; the picture of why
                the plan is held

Which subspace is searched is the YAML's `symmetric` switch. On, mirror
symmetry shrinks sixteen to nine — eight symmetric height motions and the
self-stress, which is symmetric itself — imposed as extra rows on the same
null-space solve rather than by orbit bookkeeping, and costing the start
nothing: the lens is symmetric, so its densities already are. Off, the full
basis is studied, antisymmetric wiggles included. Both widths are counted in
the report either way; the switch decides what the modes, the variations and
the checks operate on.

Mixed signs put the vertical stiffness outside the force density method's
comfort zone: the matrix goes indefinite. Indefinite is not singular — the
solve stands, and the spectrum is printed rather than assumed.

Pure form finding: no frame analysis, no code check, no optimizer. The
verdicts here are balance residuals and subspace dimensions, the ground the
truss experiments after this one stand on.

Run with `uv run --group pipeline python experiments/16_warren_formfind.py
[warren.yaml]`.
"""

import sys
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from jax_fdm.equilibrium import EquilibriumStructure
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from normax.form_finding import DensityFit
from normax.form_finding import PivotedBasis
from normax.form_finding import density_basis
from normax.form_finding import equilibrium_gap
from normax.form_finding import equilibrium_graph
from normax.form_finding import equilibrium_state
from normax.form_finding import fit_densities
from normax.form_finding import pivoted_basis
from normax.form_finding import positions_vertical
from normax.loads import create_loads_point
from normax.reporting import Report
from normax.reporting import ToleranceCheck
from normax.reporting import checks_passed
from normax.structures import Structure
from normax.structures import build_arch_2d
from normax.structures import build_gridshell_3d
from normax.structures import build_warren_2d
from normax.structures import member_lengths
from normax.visualization import SubspaceMode
from normax.visualization import TrussForm
from normax.visualization import figure_density_modes
from normax.visualization import figure_truss_forms

# A singular value this far under the largest moves nothing: the blind cut.
BLIND_FRACTION = 1e-9

# Measured at the tolerances' scale and asserted with two orders of headroom.
TOLERANCE_FIT = 1e-11
TOLERANCE_SHAPE = 1e-8
TOLERANCE_BALANCE = 1e-11
TOLERANCE_BLIND = 1e-10
TOLERANCE_ALIGNMENT = 1e-9

FIGURES = Path(__file__).resolve().parent.parent / "figures"


class TrussProblem(NamedTuple):
    """
    The truss, its deck load, the sketches, and the subspace study's settings.

    Attributes
    ----------
    num_bays :
        Number of bottom-chord segments the span is divided into.
    span :
        Horizontal distance between the two supports.
    depth :
        Height of the top chord above the bottom chord, as drawn.
    load :
        Deck load at every interior bottom-chord node.
    sag_lens :
        Depth the lens sketch hangs its bottom chord to at midspan.
    rise_lens :
        Height the lens sketch arches its top chord to at midspan.
    rise_deck :
        Height the flat-deck sketch arches its top chord to at midspan.
    symmetric :
        Whether the subspace study runs on the mirror-symmetric basis.
    margin_fraction :
        Sign margin the chords must clear, as a share of their median density.
    amplitude :
        Height the eye is shown per unit step along a subspace mode.
    """

    num_bays: int
    span: float
    depth: float
    load: float
    sag_lens: float
    rise_lens: float
    rise_deck: float
    symmetric: bool
    margin_fraction: float
    amplitude: float


class SignShift(NamedTuple):
    """
    A fit shifted along its self-stress until the chords carry their signs.

    Attributes
    ----------
    q :
        The shifted densities, bottom chord positive and top chord negative.
    window :
        Interval of shifts that sign the chords, after capping.
    shift :
        The shift taken, the feasible one nearest zero.
    """

    q: Float[np.ndarray, "edges"]
    window: tuple[float, float]
    shift: float


class HeldPlan(NamedTuple):
    """
    Everything the subspace study reads, gathered once.

    Attributes
    ----------
    structure :
        The truss as drawn, supplying the plan that is held.
    graph :
        The form-finding connectivity.
    basis :
        Orthonormal basis of the density subspace being searched.
    loads :
        The deck load case every shape here answers to.
    """

    structure: Structure
    graph: EquilibriumStructure
    basis: Float[np.ndarray, "edges independents"]
    loads: Float[Array, "nodes 3"]


class ModeSplit(NamedTuple):
    """
    The height Jacobian's split of the subspace, shape motion from self-stress.

    Attributes
    ----------
    jacobian :
        Derivative of every free height along every subspace direction.
    motion :
        Orthonormal height patterns, one column per singular value.
    singulars :
        Height motion per unit step along each direction, largest first.
    directions :
        Orthonormal subspace directions, one per row, the least mobile last.
    """

    jacobian: Float[np.ndarray, "nodes_free independents"]
    motion: Float[np.ndarray, "nodes_free nodes_free"]
    singulars: Float[np.ndarray, "sigmas"]
    directions: Float[np.ndarray, "independents independents"]


def load_problem(path: Path) -> TrussProblem:
    """
    Read the experiment's description from its YAML file.

    Parameters
    ----------
    path :
        The YAML file describing the truss, the load and the subspace study.

    Returns
    -------
    problem :
        The description, one flat record.
    """
    described = yaml.safe_load(path.read_text())
    structure = described["structure"]
    loads = described["loads"]
    sketch = described["sketch"]
    subspace = described["subspace"]

    return TrussProblem(
        num_bays=int(structure["num_bays"]),
        span=float(structure["span"]),
        depth=float(structure["depth"]),
        load=float(loads["deck"]),
        sag_lens=float(sketch["sag_lens"]),
        rise_lens=float(sketch["rise_lens"]),
        rise_deck=float(sketch["rise_deck"]),
        symmetric=bool(subspace["symmetric"]),
        margin_fraction=float(subspace["margin_fraction"]),
        amplitude=float(subspace["amplitude"]),
    )


def deck_loads(problem: TrussProblem, structure: Structure) -> Float[Array, "nodes 3"]:
    """
    The deck's weight, one point load at every interior bottom-chord node.

    Parameters
    ----------
    problem :
        The experiment's description, read for the load and the bay count.
    structure :
        The truss to load.

    Returns
    -------
    loads :
        Force applied at every node, the top chord carrying none.
    """
    interior = range(1, problem.num_bays)
    cases = [
        create_loads_point(structure, problem.load, node=node) for node in interior
    ]

    return jnp.sum(jnp.stack(cases), axis=0)


def lens_geometry(
    problem: TrussProblem,
    structure: Structure,
    sag: float,
    rise: float,
) -> Float[np.ndarray, "nodes 3"]:
    """
    The drawn truss with each chord bent into a parabola, the plan held.

    Parameters
    ----------
    problem :
        The experiment's description, read for the span and the bay count.
    structure :
        The truss as drawn.
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
    shape = 4.0 * (xyz[:, 0] / problem.span) * (1.0 - xyz[:, 0] / problem.span)

    bottom = slice(0, problem.num_bays + 1)
    top = slice(problem.num_bays + 1, None)
    xyz[bottom, 2] -= sag * shape[bottom]
    xyz[top, 2] += rise * shape[top]

    return xyz


def signed_densities(problem: TrussProblem, fit: DensityFit) -> SignShift:
    """
    Shift a fit along its self-stress until the chords carry their signs.

    Parameters
    ----------
    problem :
        The experiment's description, read for the families and the margin.
    fit :
        The fit to shift, with exactly one state of self-stress.

    Returns
    -------
    shifted :
        The signed densities, the feasible window, and the shift taken.

    Notes
    -----
    Each chord member asks its sign of the shift as one linear inequality, so
    the feasible set is an interval and is intersected exactly rather than
    scanned. Of the feasible shifts the one nearest zero is taken, stepped a
    twentieth of the window inside it: self-stress raises every force it
    touches, so the least of it that signs the chords is the least material.
    The diagonals are left free on purpose: a Warren diagonal swaps between
    tension and compression as the load moves, and a sign pinned here would
    fight the physics later.
    """
    bays = problem.num_bays
    mode = fit.self_stresses[:, 0]

    signs = np.concatenate([np.ones(bays), -np.ones(bays - 1)])
    chords = np.arange(2 * bays - 1)
    margin = problem.margin_fraction * float(np.median(np.abs(fit.q[:bays])))

    values = signs * fit.q[chords]
    slopes = signs * mode[chords]

    cap = 20.0 * float(np.abs(fit.q).max())
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
    q = fit.q + shift * mode

    return SignShift(q, (lower, upper), shift)


def mirrored_nodes(problem: TrussProblem) -> Int[np.ndarray, "nodes"]:
    """
    Mirror image of every node index about midspan.

    Parameters
    ----------
    problem :
        The experiment's description, read for the bay count.

    Returns
    -------
    nodes_mirrored :
        The node the mirror carries each node onto, chord by chord.
    """
    bays = problem.num_bays
    bottom = bays - np.arange(bays + 1)
    top = 2 * bays - np.arange(bays)

    return np.concatenate([bottom, top])


def member_name(problem: TrussProblem, edge: int) -> str:
    """
    A member's family and its position in it, spelled for a panel title.

    Parameters
    ----------
    problem :
        The experiment's description, read for the bay count.
    edge :
        Index of the member in the generator's family ordering.

    Returns
    -------
    name :
        The family and the one-based position, reading left to right.
    """
    bays = problem.num_bays
    if edge < bays:
        return f"bottom chord {edge + 1}"
    if edge < 2 * bays - 1:
        return f"top chord {edge - bays + 1}"
    if edge < 3 * bays - 1:
        return f"rising diagonal {edge - 2 * bays + 2}"

    return f"falling diagonal {edge - 3 * bays + 2}"


def mirrored_edges(
    problem: TrussProblem,
    structure: Structure,
) -> Int[np.ndarray, "edges"]:
    """
    Index of every member's mirror image about midspan.

    Parameters
    ----------
    problem :
        The experiment's description, read for the mirror.
    structure :
        The truss supplying the members the mirror permutes.

    Returns
    -------
    edges_mirrored :
        The member the mirror carries each member onto.
    """
    edges = np.asarray(structure.edges)
    ordered = np.sort(edges, axis=1)
    reflected = np.sort(mirrored_nodes(problem)[edges], axis=1)

    lookup = {tuple(pair): index for index, pair in enumerate(ordered.tolist())}
    targets = [lookup[tuple(pair)] for pair in reflected.tolist()]

    return np.asarray(targets)


def solved_heights(
    plan: HeldPlan,
    xi: Float[Array, "independents"],
) -> Float[Array, "nodes_free"]:
    """
    Heights of the free nodes at one coordinate of the held-plan subspace.

    Parameters
    ----------
    plan :
        The subspace study's inputs.
    xi :
        Coordinate along the basis columns.

    Returns
    -------
    heights :
        Solved height of every free node, in the graph's free-node order.
    """
    q = jnp.asarray(plan.basis) @ xi
    xyz = positions_vertical(q, plan.structure.nodes, plan.graph, plan.loads)

    return xyz[plan.graph.indices_free, 2]


def split_subspace(plan: HeldPlan, xi: Float[Array, "independents"]) -> ModeSplit:
    """
    Split the subspace at a point into shape motions and the blind direction.

    Parameters
    ----------
    plan :
        The subspace study's inputs.
    xi :
        Coordinate the Jacobian is taken at.

    Returns
    -------
    split :
        The Jacobian and its singular value decomposition.
    """
    jacobian = jax.jacfwd(solved_heights, argnums=1)(plan, xi)
    motion, singulars, directions = np.linalg.svd(np.asarray(jacobian))

    return ModeSplit(np.asarray(jacobian), motion, singulars, directions)


def moving_sigmas(split: ModeSplit) -> Float[np.ndarray, "moving"]:
    """
    The singular values that move the shape, the blind tail cut off.

    Parameters
    ----------
    split :
        The Jacobian's split of the subspace.

    Returns
    -------
    sigmas :
        The singular values above the blind cut, largest first.
    """
    return split.singulars[split.singulars > BLIND_FRACTION * split.singulars[0]]


def visible_modes(
    problem: TrussProblem,
    split: ModeSplit,
    plan: HeldPlan,
    lens: Float[np.ndarray, "nodes 3"],
) -> list[SubspaceMode]:
    """
    Every subspace direction as a displaced, recolored drawing of the truss.

    Parameters
    ----------
    problem :
        The experiment's description, read for the amplitude.
    split :
        The Jacobian's split of the subspace.
    plan :
        The subspace study's inputs.
    lens :
        The geometry the modes displace, the fitted lens.

    Returns
    -------
    modes :
        One drawable mode per direction, the self-stress last and unmoved.
    """
    nodes_free = np.asarray(plan.graph.indices_free)
    largest = float(split.singulars[0])

    modes = []
    for index, direction in enumerate(split.directions):
        densities = plan.basis @ direction
        xyz = np.asarray(lens).copy()
        sigma = float(split.singulars[index]) if index < split.singulars.size else 0.0
        if sigma > BLIND_FRACTION * largest:
            xyz[nodes_free, 2] += problem.amplitude * split.motion[:, index]
            title = f"mode {index + 1} — σ = {sigma:.3g}"
        else:
            title = "self-stress — σ ≈ 0"
        modes.append(SubspaceMode(title, xyz, densities))

    return modes


def variation_forms(
    problem: TrussProblem,
    split: ModeSplit,
    plan: HeldPlan,
    xi: Float[Array, "independents"],
) -> list[TrussForm]:
    """
    The truss re-form-found one step along every direction of a basis.

    Parameters
    ----------
    problem :
        The experiment's description, read for the amplitude.
    split :
        The Jacobian's split of the basis being varied.
    plan :
        The subspace study's inputs, holding that basis.
    xi :
        Coordinate of the start the variations step away from.

    Returns
    -------
    forms :
        One solved, force-colored truss per direction, the blind one last.

    Notes
    -----
    Nonlinear where `visible_modes` is linearized: each step is handed back to
    the vertical solve, so what is drawn is a shape the form finder actually
    returns. Steps are sized to move the truss about one amplitude; the blind
    direction, which moves nothing, is stepped by half the coordinate's norm
    so the force redistribution shows instead.
    """
    largest = float(split.singulars[0])

    forms = []
    for index, direction in enumerate(split.directions):
        sigma = float(split.singulars[index]) if index < split.singulars.size else 0.0
        if sigma > BLIND_FRACTION * largest:
            step = problem.amplitude / sigma
            title = f"mode {index + 1} — σ = {sigma:.3g}"
        else:
            step = 0.5 * float(np.linalg.norm(np.asarray(xi)))
            title = "self-stress — σ ≈ 0"

        stepped = np.asarray(xi) + step * direction
        q = plan.basis @ stepped
        nodes = plan.structure.nodes
        xyz = positions_vertical(jnp.asarray(q), nodes, plan.graph, plan.loads)
        lengths = member_lengths(xyz, plan.structure.edges)
        forces = q * np.asarray(lengths)
        forms.append(TrussForm(title, np.asarray(xyz), forces))

    return forms


def pivoted_variations(
    problem: TrussProblem,
    plan: HeldPlan,
    pivot: PivotedBasis,
    xi: Float[Array, "independents"],
) -> list[TrussForm]:
    """
    The truss re-form-found with one independent density nudged at a time.

    Parameters
    ----------
    problem :
        The experiment's description, read for the amplitude and the names.
    plan :
        The subspace study's inputs, holding the pivoted basis.
    pivot :
        The pivoted basis, read for which member each coordinate is.
    xi :
        The independent densities of the start.

    Returns
    -------
    forms :
        One solved, force-colored truss per independent edge.

    Notes
    -----
    Coordinate-axis variations rather than singular directions: each panel
    answers what one member's density does, which is the question the pivoted
    basis exists to make askable. Steps are sized from the height Jacobian's
    columns to move the truss about one amplitude; a coordinate the shape
    barely answers is stepped by half the start's norm instead, so its force
    redistribution shows.
    """
    jacobian = np.asarray(jax.jacfwd(solved_heights, argnums=1)(plan, xi))
    gains = np.linalg.norm(jacobian, axis=0)
    mirrors = mirrored_edges(problem, plan.structure)

    forms = []
    for index, edge in enumerate(pivot.independents.tolist()):
        title = member_name(problem, edge)
        if problem.symmetric and mirrors[edge] != edge:
            title = f"{title} and mirror"
        if gains[index] > BLIND_FRACTION * gains.max():
            step = problem.amplitude / gains[index]
            title = f"{title} — gain {gains[index]:.3g}"
        else:
            step = 0.5 * float(np.linalg.norm(np.asarray(xi)))
            title = f"{title} — gain ≈ 0"

        stepped = np.asarray(xi).copy()
        stepped[index] += step
        q = plan.basis @ stepped
        nodes = plan.structure.nodes
        xyz = positions_vertical(jnp.asarray(q), nodes, plan.graph, plan.loads)
        lengths = member_lengths(xyz, plan.structure.edges)
        forces = q * np.asarray(lengths)
        forms.append(TrussForm(title, np.asarray(xyz), forces))

    return forms


def freeplan_form(
    problem: TrussProblem,
    plan: HeldPlan,
    q: Float[np.ndarray, "edges"],
) -> TrussForm:
    """
    What grouped densities of the same scale do when the plan is left free.

    Parameters
    ----------
    problem :
        The experiment's description, read for the families.
    plan :
        The truss and the load case, the held plan being ignored here.
    q :
        The signed lens densities, read only for their scale.

    Returns
    -------
    form :
        The unrestrained equilibrium shape and its member forces.

    Notes
    -----
    The cautionary panel. One density per family is the natural first
    parametrization, and the unrestrained solver honors it — with drifting
    bays and a shape that leaves the drawn truss altogether. Holding the plan
    is what the rest of this experiment buys.
    """
    bays = problem.num_bays
    scale = float(np.median(np.abs(q[:bays])))
    families = [
        np.full(bays, scale),
        np.full(bays - 1, -scale),
        np.full(2 * bays, 0.25 * scale),
    ]
    grouped = np.concatenate(families)

    xyz_fixed = plan.structure.nodes[plan.graph.indices_fixed]
    state = equilibrium_state(jnp.asarray(grouped), xyz_fixed, plan.graph, plan.loads)
    forces = grouped * np.asarray(state.lengths[:, 0])

    return TrussForm("free plan, grouped densities", state.xyz, forces)


def report_subspace(report: Report, problem: TrussProblem, plan: HeldPlan) -> None:
    """
    The independent-edge counts, and which basis the study operates on.

    Parameters
    ----------
    report :
        The report to write into.
    problem :
        The experiment's description, read for the mirror and the switch.
    plan :
        The subspace study's inputs, holding the searched basis.
    """
    members = plan.structure.num_edges
    nodes_free = plan.structure.num_nodes - int(plan.structure.supports.shape[0])

    width_full = density_basis(plan.structure).shape[1]
    mirror = mirrored_nodes(problem)
    width_symmetric = density_basis(plan.structure, mirror).shape[1]
    searched = "symmetric" if problem.symmetric else "full"

    chain = density_basis(build_arch_2d(num_edges=10, span=problem.span)).shape[1]
    shell = density_basis(build_gridshell_3d()).shape[1]

    report.write_heading("Independent edges of the held plan")
    entries = [
        ("members", f"{members}"),
        ("free nodes", f"{nodes_free}"),
        ("full width", f"{width_full} = {nodes_free} heights + 1 self-stress"),
        (
            "symmetric width",
            f"{width_symmetric} = {width_symmetric - 1} heights + 1 self-stress",
        ),
        ("searched basis", f"{searched} ({plan.basis.shape[1]})"),
        ("chain of ten, same span", f"{chain}"),
        ("gridshell, four rings", f"{shell}"),
    ]
    report.write_entries(entries)


def report_stiffness(
    report: Report,
    plan: HeldPlan,
    q: Float[np.ndarray, "edges"],
) -> None:
    """
    The vertical stiffness spectrum at mixed-sign densities, measured.

    Parameters
    ----------
    report :
        The report to write into.
    plan :
        The subspace study's inputs.
    q :
        The signed lens densities.
    """
    connectivity = np.asarray(plan.graph.connectivity_free)
    stiffness = connectivity.T @ (q[:, None] * connectivity)
    eigen = np.linalg.eigvalsh(stiffness)

    negatives = int(np.sum(eigen < 0.0))
    condition = float(np.abs(eigen).max() / np.abs(eigen).min())

    report.write_heading("Vertical stiffness at mixed signs")
    entries = [
        ("negative eigenvalues", f"{negatives} of {eigen.size}"),
        ("condition number", f"{condition:.2e}"),
    ]
    report.write_entries(entries)


def main(path: Path) -> None:
    """
    Run the study, write the report, and save the figures.

    Parameters
    ----------
    path :
        The YAML file describing the run.
    """
    problem = load_problem(path)
    bays = problem.num_bays

    report = Report()
    report.write_banner("Warren truss — held-plan form finding")

    structure = build_warren_2d(bays, problem.span, problem.depth)
    graph = equilibrium_graph(structure)
    loads = deck_loads(problem, structure)

    if problem.symmetric:
        basis = density_basis(structure, mirrored_nodes(problem))
    else:
        basis = density_basis(structure)
    plan = HeldPlan(structure, graph, basis, loads)

    report_subspace(report, problem, plan)

    lens = lens_geometry(problem, structure, problem.sag_lens, problem.rise_lens)
    fit = fit_densities(structure, lens, loads)
    shifted = signed_densities(problem, fit)

    deck = lens_geometry(problem, structure, 0.0, problem.rise_deck)
    fit_deck = fit_densities(structure, deck, loads)
    shifted_deck = signed_densities(problem, fit_deck)

    report.write_heading("Fitting densities to the sketches")
    window = shifted.window
    entries = [
        ("lens fit gap [N]", f"{fit.gap:.2e}"),
        ("lens self-stresses", f"{fit.self_stresses.shape[1]}"),
        ("lens sign window", f"[{window[0]:.1f}, {window[1]:.1f}]"),
        (
            "lens bottom chord q [N/mm]",
            f"{shifted.q[:bays].min():+.1f} to {shifted.q[:bays].max():+.1f}",
        ),
        (
            "lens top chord q [N/mm]",
            f"{shifted.q[bays : 2 * bays - 1].min():+.1f}"
            f" to {shifted.q[bays : 2 * bays - 1].max():+.1f}",
        ),
        ("flat-deck fit gap [N]", f"{fit_deck.gap:.2e}"),
    ]
    report.write_entries(entries)

    solved = positions_vertical(jnp.asarray(shifted.q), structure.nodes, graph, loads)
    shape_gap = float(np.abs(np.asarray(solved)[:, 2] - lens[:, 2]).max())
    balance_gap = equilibrium_gap(structure, np.asarray(solved), shifted.q, loads)

    report_stiffness(report, plan, shifted.q)

    xi = jnp.asarray(basis.T @ shifted.q)
    rebuilt = basis @ np.asarray(xi)
    rebuilding = float(np.linalg.norm(rebuilt - shifted.q))
    rebuilding = rebuilding / float(np.linalg.norm(shifted.q))

    split = split_subspace(plan, xi)
    moving = moving_sigmas(split)
    blind = float(np.linalg.norm(split.jacobian @ split.directions[-1]))
    stress = basis @ split.directions[-1]
    alignment = abs(float(stress @ fit.self_stresses[:, 0]))

    report.write_heading("The Jacobian's split of the searched subspace")
    entries = [
        ("moving directions", f"{moving.size} of {basis.shape[1]}"),
        ("largest σ", f"{moving[0]:.3g}"),
        ("smallest moving σ", f"{moving[-1]:.3g}"),
        ("blind direction |J v|", f"{blind:.2e}"),
        ("blind vs self-stress", f"|cos| = {alignment:.12f}"),
        ("start reconstruction gap", f"{rebuilding:.2e} of |q|"),
    ]
    report.write_entries(entries)

    if problem.symmetric:
        pivot = pivoted_basis(structure, mirrored_nodes(problem))
    else:
        pivot = pivoted_basis(structure)
    plan_pivoted = HeldPlan(structure, graph, pivot.basis, loads)
    xi_pivoted = jnp.asarray(shifted.q[pivot.independents])
    rebuilt_pivoted = pivot.basis @ np.asarray(xi_pivoted)
    rebuilding_pivoted = float(np.linalg.norm(rebuilt_pivoted - shifted.q))
    rebuilding_pivoted = rebuilding_pivoted / float(np.linalg.norm(shifted.q))

    report.write_heading("Independent edges by QR pivoting")
    named = [member_name(problem, edge) for edge in pivot.independents.tolist()]
    entries = [("independent edges", ", ".join(named))]
    entries.append(("largest transfer coefficient", f"{np.abs(pivot.basis).max():.2f}"))
    entries.append(("start reconstruction gap", f"{rebuilding_pivoted:.2e} of |q|"))
    report.write_entries(entries)

    free = freeplan_form(problem, plan, shifted.q)
    bays_x = np.diff(np.asarray(free.xyz)[: bays + 1, 0])
    report.write_heading("The free plan, for contrast")
    entries = [
        ("nominal bay [mm]", f"{problem.span / bays:.0f}"),
        ("free-plan bays [mm]", f"{bays_x.min():.0f} to {bays_x.max():.0f}"),
    ]
    report.write_entries(entries)

    lengths_lens = member_lengths(jnp.asarray(lens), structure.edges)
    lengths_deck = member_lengths(jnp.asarray(deck), structure.edges)
    forms = [
        TrussForm("lens, deck suspended", lens, shifted.q * np.asarray(lengths_lens)),
        TrussForm(
            "lens, deck on the bottom chord",
            deck,
            shifted_deck.q * np.asarray(lengths_deck),
        ),
    ]

    FIGURES.mkdir(exist_ok=True)
    figure = figure_truss_forms(structure.edges, forms, structure.nodes)
    figure.savefig(FIGURES / "16_warren_forms.png", dpi=200, bbox_inches="tight")
    figure = figure_truss_forms(structure.edges, [free], structure.nodes)
    figure.savefig(FIGURES / "16_warren_freeplan.png", dpi=200, bbox_inches="tight")
    modes = visible_modes(problem, split, plan, lens)
    figure = figure_density_modes(structure.edges, lens, modes)
    figure.savefig(FIGURES / "16_density_modes.png", dpi=200, bbox_inches="tight")
    variations = variation_forms(problem, split, plan, xi)
    figure = figure_truss_forms(structure.edges, variations, lens, "lens start")
    figure.savefig(FIGURES / "16_variations.png", dpi=160, bbox_inches="tight")
    named_forms = pivoted_variations(problem, plan_pivoted, pivot, xi_pivoted)
    figure = figure_truss_forms(structure.edges, named_forms, lens, "lens start")
    path = FIGURES / "16_variations_pivoted.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    report.write_heading(f"figures written to {FIGURES}")

    checks = (
        ToleranceCheck("lens fit gap / load", fit.gap / problem.load, TOLERANCE_FIT),
        ToleranceCheck(
            "flat-deck fit gap / load", fit_deck.gap / problem.load, TOLERANCE_FIT
        ),
        ToleranceCheck("shape reproduction [mm]", shape_gap, TOLERANCE_SHAPE),
        ToleranceCheck(
            "balance gap / load", balance_gap / problem.load, TOLERANCE_BALANCE
        ),
        ToleranceCheck("blind direction / σ max", blind / moving[0], TOLERANCE_BLIND),
        ToleranceCheck(
            "self-stress misalignment", 1.0 - alignment, TOLERANCE_ALIGNMENT
        ),
        ToleranceCheck("start reconstruction / |q|", rebuilding, TOLERANCE_ALIGNMENT),
        ToleranceCheck(
            "pivoted reconstruction / |q|", rebuilding_pivoted, TOLERANCE_ALIGNMENT
        ),
    )
    counted = basis.shape[1] == moving.size + 1
    counted = counted and pivot.basis.shape[1] == basis.shape[1]
    passed = checks_passed(checks) and counted

    report.write_heading("Summary")
    report.write_checks(checks)
    report.write_verdict(passed)


if __name__ == "__main__":
    jnp.set_printoptions(precision=12)
    described = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    main(described or Path(__file__).with_name("warren.yaml"))
