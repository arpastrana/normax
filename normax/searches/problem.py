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
Everything the three searches share, assembled once.
"""

import os
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax_fdm.equilibrium import EquilibriumStructure
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from normax.analysis import AbstractFrameAnalyzer
from normax.analysis import SmaxAnalyzer
from normax.analysis import normal_axis
from normax.design import StructuralDesignPipeline
from normax.form_finding import FdmFormFinder
from normax.form_finding import SubspaceFormFinder
from normax.form_finding import density_basis
from normax.form_finding import pivoted_basis
from normax.loads import LoadCases
from normax.searches.config import TaskConfig
from normax.searches.config import ViewerConfig
from normax.searches.folding import FoldingMaps
from normax.searches.folding import MirrorFolding
from normax.searches.folding import orbit_matrix
from normax.searches.loads import LoadPlan
from normax.searches.settings import ANALYSIS_PLANAR
from normax.searches.settings import ANALYSIS_SMAX
from normax.searches.settings import GRADE
from normax.searches.settings import SECTION_CLASS
from normax.searches.settings import SIZING_EC3
from normax.searches.settings import TOLERANCE_MIRRORED
from normax.sections import TubeFamily
from normax.sizing import AbstractMemberSizer
from normax.sizing import Ec3Sizer
from normax.sizing import build_section_family
from normax.structures import Structure
from normax.tesseract import BACKEND_VARIABLE
from normax.tesseract import BlueprintClient
from normax.tesseract import TesseractAnalyzer
from normax.tesseract import blueprint_tesseract
from normax.tesseract import local_chain


class DesignProblem(NamedTuple):
    """
    The prepared truss, its blocks, and the subspace the geometry moves in.

    Attributes
    ----------
    structure :
        The truss the blocks were built against, supplying the drawn geometry
        the sizing-only search holds.
    pipeline :
        The three blocks, each already bound to the truss on the host. The
        first is a `SubspaceFormFinder`, so the end-to-end search's geometry
        variables are the coordinates the block itself declares.
    loads :
        The case the shape answers to, and the cases every search is checked
        against.
    case_names :
        Name of every built load case, in build order.
    cases_held :
        Which load cases the descents carry inequality rows for. A symmetric
        search may need fewer than were built, and every answer is still read
        and checked against all of them.
    folding :
        Pattern matrices folding the mirror into every search's variables,
        None-valued when the search is not symmetric.
    edges_mirrored :
        The member the midspan mirror carries each member onto.
    nodes_free :
        Indices of the nodes whose height the free-heights search moves.
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
    Where the end-to-end search leaves from, and how exactly it was matched.

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


class SearchMaps(NamedTuple):
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
        from a search that offers no repair.
    augmented :
        The mass and the rows as one scalar, with its gradient — the same
        search read by an augmented Lagrangian instead of by a constrained
        solver, and the multipliers, the penalty and the reference mass are
        arguments of it rather than constants inside it.

    Notes
    -----
    **The augmented map is the same search at a different derivative price.**
    A constrained solver reads `slack` and `jacobian`, which is one forward
    tangent per variable through the whole pipeline; the augmented map folds
    the rows into the objective before anything is differentiated, so a
    gradient is one reverse pass whatever the constraint set. On a frame with
    a row per member per load case that is most of the cost of a search. What
    it gives up is the solver's own convergence certificate, which a short run
    of `descend_search` from its landing restores.

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
    augmented: object = None


class SearchAnswer(NamedTuple):
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
    violations :
        Worst violation over the rows at the end of every round, from a search
        that reports one. None from a constrained solver, which holds the rows
        itself and has none to report.

    Notes
    -----
    **A mass without the violation beside it is not a design.** An augmented
    descent reads its mass at points that may be far outside the constraints,
    and a landing there can look far cheaper than any feasible answer — half
    the mass, on a truss measured at a weak penalty. Carrying the column is
    what lets a caller refuse such a landing instead of handing it on to
    something that will treat it as a starting point.
    """

    variables: Float[np.ndarray, "variables"]
    masses: Float[np.ndarray, "steps"]
    iterations: int
    converged: bool
    violations: Float[np.ndarray, "steps"] | None = None


class SearchRead(NamedTuple):
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


def built_analyzer(
    structure: Structure,
    family: TubeFamily,
    config: TaskConfig,
) -> AbstractFrameAnalyzer:
    """
    The frame analysis a run description asked for.

    Parameters
    ----------
    structure :
        The structure the stage is built on.
    family :
        The tube family the seed section is drawn from.
    config :
        The run description, read for the backend and the seed diameter.

    Returns
    -------
    analyzer :
        The stage, in process or across a boundary.

    Notes
    -----
    The crossed backend reads its solver from the environment, so an experiment
    picks once for the whole process rather than once per block. Its solver is
    planar, and the spike ruling holds it to two dimensions.
    """
    section = family(config.analysis.diameter)
    if config.analysis.backend == ANALYSIS_SMAX:
        return SmaxAnalyzer(structure, section)

    os.environ[BACKEND_VARIABLE] = config.analysis.backend.removesuffix("_tesseract")
    chain = local_chain()

    # Only a planar solver is told which plane; the traced one measures its own
    # and the space-frame one has no such restriction to state.
    if config.analysis.backend in ANALYSIS_PLANAR:
        normal = normal_axis(structure)
    else:
        normal = None

    return TesseractAnalyzer(structure, chain.analysis, family, normal)


def built_sizer(
    structure: Structure,
    family: TubeFamily,
    config: TaskConfig,
) -> AbstractMemberSizer:
    """
    The code check a run description asked for.

    Parameters
    ----------
    structure :
        The structure the stage is built on.
    family :
        The tube family every size is drawn from.
    config :
        The run description, read for the backend.

    Returns
    -------
    sizer :
        The stage, in process or across a boundary.

    Notes
    -----
    Both backends draw from the same family, so what differs downstream is the
    check rather than the geometry it is asked about.
    """
    if config.sizing.backend == SIZING_EC3:
        return Ec3Sizer(structure, family)

    return BlueprintClient(structure, blueprint_tesseract(), family)


def prepare_problem(
    structure: Structure,
    config: TaskConfig,
    plan: LoadPlan,
    folding_by: FoldingMaps,
) -> DesignProblem:
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
        Everything the searches read, gathered once on the host.

    Notes
    -----
    **The three kinds of variable need not fold by the same group.** The
    density basis is folded by the mirror alone, always, so the end-to-end
    search's dimension is a property of the structure. Sections may be folded
    as far as fabrication wants, carrying no argument with them. Heights are
    the delicate one: folded by the mirror alone they are a strict superset of
    what the form finder reaches, which is what makes a gap between the searches
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
        built_analyzer(structure, family, config),
        built_sizer(structure, family, config),
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

    problem = DesignProblem(
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


class ViewRequest(NamedTuple):
    """
    Everything a viewer needs, handed back for a caller to open or ignore.

    Attributes
    ----------
    problem :
        The prepared structure, supplying the connectivity, the blocks and the
        load cases every answer is drawn under.
    reads :
        Each search's answer read back, keyed by search.
    searches :
        Which searches to draw, each appearing under its own name.
    viewer_config :
        The run's viewer section, read for which case to draw and how far to
        exaggerate its load glyphs.

    Notes
    -----
    A window blocks the process until it closes, so a search returns this
    rather than opening one: the run is complete and reported by the time the
    request is handed back, and whether a window follows is the caller's.
    """

    problem: "DesignProblem"
    reads: dict[str, "SearchRead"]
    searches: tuple[str, ...]
    viewer_config: ViewerConfig
