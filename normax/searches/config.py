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
The run description, and reading one out of a file.
"""

from typing import NamedTuple

import yaml

from normax.optimization import AugmentedBudget
from normax.searches.settings import ANALYSIS_BACKENDS
from normax.searches.settings import ANALYSIS_SMAX
from normax.searches.settings import METHOD_ORDER
from normax.searches.settings import METHOD_SLSQP
from normax.searches.settings import SEARCH_ORDER
from normax.searches.settings import SIZING_BACKENDS
from normax.searches.settings import SIZING_EC3


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
        Whether the free-heights search's heights are folded the same way,
        leaving one height per ring. It changes what the comparison means:
        the two searches then search spaces of the same dimension, neither one
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
    The lens the end-to-end search starts from.

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
    backend :
        Which solver answers the stage. `smax` traces in process,
        `smax_tesseract` is the same solver across a Tesseract boundary, and
        `opensees` and `pynite` are two further solvers across that same
        schema, neither of which differentiates itself. `opensees` is the
        two-dimensional demo alone — it refuses a geometry that leaves its
        plane — so a shell asks for `pynite`, which is a space frame and whose
        adjoint is this repository's.
    """

    diameter: float
    backend: str = ANALYSIS_SMAX


class SizingConfig(NamedTuple):
    """
    Which implementation of the standard the members are sized against.

    Attributes
    ----------
    backend :
        `ec3` traces EN 1993-1-1 in process, `blueprint_tesseract` reaches
        Blueprints' cross-section check across a Tesseract boundary. The two
        answer different questions: Blueprints implements no 6.3.1 flexural
        buckling, so a design sized through it is not a design sized to the
        member check.
    """

    backend: str = SIZING_EC3


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
        Smallest length any member may keep while the free-heights search
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
        How many points each search is descended from, the first being the
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
    method :
        Which search descends every search. `slsqp` holds the rows as explicit
        constraints and pays a Jacobian per iteration; `augmented` folds them
        into the objective and pays one reverse pass, then polishes with the
        first for its certificate.
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
    method: str = METHOD_SLSQP


class ViewerConfig(NamedTuple):
    """
    Whether the run ends in a viewer, and which answer it draws there.

    Attributes
    ----------
    enabled :
        Whether an answer is opened in a viewer once the report is written.
    search :
        Which search's answer to draw, named as the searches are named.
    solo_search :
        Whether to descend `search` alone and leave the other two undone. The
        report then holds whatever a single search can say — its own landing,
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

    One search rather than a set of them: two answers occupy nearly the same
    space, so a scene holding both is read by switching halves of it off, and
    naming the one wanted is the shorter way to the same look.

    **A solo run is for iterating, never for reporting a result.** The gaps
    between the searches are the point of the comparison, and a run that
    descends one search cannot state them; what it buys is the loop between
    changing a file and seeing the shape, which the slowest search otherwise
    sets the pace of.
    """

    enabled: bool
    search: str
    solo_search: bool
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
        The lens the end-to-end search starts from, or None where the drawn
        geometry is already the start and no sketch is needed.
    subspace :
        Which held-plan basis the geometry variables span.
    analysis :
        What the frame is seeded with, and which solver answers it.
    sizing :
        Which implementation of the standard the members are sized against.
    descent :
        The budgets the searches share.
    viewer :
        Whether the run ends in a viewer.
    augmented :
        Budgets for the augmented method, or None to take the defaults. Read
        whatever the method is, so that switching to it and back does not
        change what a file says.

    Notes
    -----
    The first three sections are the profile's to parse and the profile's to
    read; nothing in the shared flow touches a field of them. The rest are
    family-blind, which is what lets one flow run a truss and a shell.
    """

    structure: TrussConfig | ShellConfig
    loads: LoadConfig | ShellLoads
    sketch: SketchConfig | None
    subspace: SubspaceConfig
    analysis: AnalysisConfig
    descent: DescentConfig
    viewer: ViewerConfig
    sizing: SizingConfig = SizingConfig()
    augmented: AugmentedBudget | None = None


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
        If the basis, the viewer's search or the descent method is not one this
        flow serves.
    TypeError
        If a section names a field that does not exist, or omits one it does.

    Notes
    -----
    Only the two sections a file may leave out carry defaults — the descent
    method, and the budgets belonging to it — so a file missing anything else
    is refused rather than quietly completed.

    **The augmented budget is read whatever the method is**, so that switching
    a file's method and switching it back leaves the file saying the same
    thing. Whether it counts towards the answer store's fingerprint is a
    separate question, settled by `descent_digest`: a budget the run does not
    use decides nothing and is not counted.
    """
    subspace = SubspaceConfig(**document["subspace"])
    viewer = ViewerConfig(**document["viewer"])
    descent = DescentConfig(**document["descent"])

    if subspace.basis not in ("svd", "pivoted"):
        raise ValueError(f"basis must be svd or pivoted, got {subspace.basis}")
    if viewer.search not in SEARCH_ORDER:
        named = ", ".join(SEARCH_ORDER)
        raise ValueError(f"viewer search must be one of {named}, got {viewer.search}")
    if descent.method not in METHOD_ORDER:
        named = ", ".join(METHOD_ORDER)
        raise ValueError(f"method must be one of {named}, got {descent.method}")

    analysis = AnalysisConfig(**document["analysis"])
    if analysis.backend not in ANALYSIS_BACKENDS:
        named = ", ".join(ANALYSIS_BACKENDS)
        raise ValueError(f"analysis backend must be one of {named}")

    sizing = SizingConfig(**document.get("sizing", {}))
    if sizing.backend not in SIZING_BACKENDS:
        named = ", ".join(SIZING_BACKENDS)
        raise ValueError(f"sizing backend must be one of {named}")

    return {
        "subspace": subspace,
        "analysis": analysis,
        "sizing": sizing,
        "descent": descent,
        "viewer": viewer,
        "augmented": augmented_budget(document),
    }


def augmented_budget(document: dict[str, object]) -> AugmentedBudget | None:
    """
    The augmented budget a run description names, cast, or None for none.

    Parameters
    ----------
    document :
        The loaded YAML document.

    Returns
    -------
    budget :
        Rounds, inner iterations, the penalty schedule and the stopping rules,
        or None where the file names none and the defaults apply.

    Raises
    ------
    TypeError
        If the section names a field that does not exist, or omits one it does.

    Notes
    -----
    Every field is cast on the way in. YAML reads an exponent without a signed
    power as a string, so a ceiling written `1.0e8` would arrive as text and
    first be noticed several rounds into a descent, where it is compared
    against a penalty.
    """
    named = document.get("augmented")
    if named is None:
        return None

    counts = ("rounds", "iterations", "settled", "opening")
    read = {key: int(value) for key, value in named.items() if key in counts}
    scales = {key: float(value) for key, value in named.items() if key not in counts}
    read.update(scales)

    return AugmentedBudget(**read)


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
        The truss, and the settings its searches are compared under.

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
        The shell, and the settings its searches are compared under.

    Raises
    ------
    ValueError
        If the drift sector cannot be centred on a spoke, is wider than the
        shell has spokes, or is centred on the mirror plane itself, where the
        two drift cases would be the same case twice.

    Notes
    -----
    No sketch section: the generated cap is already funicular under its own
    uniform case, so the end-to-end search starts on the drawn geometry rather
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
