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
A three-dimensional frame analysis in PyNite, and the adjoint written for it.

PyNite is plain Python over NumPy and SciPy with no derivative of its own, so
this module is the forward pass and a reverse rule assembled by hand from the
matrices PyNite exposes: the element is restated in JAX and held equal to the
solver's own by test, and equilibrium is differentiated as an implicit function
so one factorization answers for every parameter.

**Two frames must not be confused.** PyNite hard-codes its vertical as the
second global axis, so coordinates are rotated into it and forces rotated back;
and the bending components are reported in `member_frame`, whose transverse pair
is completed against this repository's vertical. **PyNite's axial sign is
inverted**: its local end-force vector reports tension as negative, so every
reading is negated on the way out. A physical member segments wherever a load
is attached along it, which nodal loading never does.
"""

import warnings
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int
from Pynite import FEModel3D
from Pynite.Analysis import _partition_D
from Pynite.Analysis import _prepare_model
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import MatrixRankWarning
from scipy.sparse.linalg import SuperLU
from scipy.sparse.linalg import splu

from normax.analysis import DOF_PER_NODE
from normax.analysis import MemberForces
from normax.analysis.element import DOF_PER_MEMBER
from normax.analysis.element import REFERENCE_MARGIN
from normax.analysis.element import SectionRigidity
from normax.analysis.element import member_frame
from normax.analysis.element import stiffness_global
from normax.sections import MemberSections
from normax.sections import TubeFamily
from normax.structures import Structure
from normax.units import MEGAPASCAL
from normax.units import MILLIMETER
from normax.units import NEWTON_MILLIMETER

# EN 1993-1-1 3.2.6, as the traced oracle reads it, so both solve one steel.
POISSONS_RATIO = 0.3

# What the one load pattern and the one combination are called.
CASE_NAME = "applied"
COMBO_NAME = "design"

# One end flips, so a uniform moment reads equal and of one sign at both ends.
DIAGRAM_SIGN = np.array([-1.0, 1.0])

# Carries this repository's axes onto PyNite's: a rotation, so a moment turns
# like a force and its transpose is its inverse.
ROTATION = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])

# The polar moment of an annulus is what its two second moments add to.
TORSION_FACTOR = 2.0

# Where each reported quantity sits in one member's reading.
READING_AXIAL = 0
READING_MAJOR = slice(1, 3)
READING_MINOR = slice(3, 5)


class FrameProblem(NamedTuple):
    """
    The frame as described, apart from the two things a derivative is taken in.

    Attributes
    ----------
    structure :
        The connectivity and the supported nodes.
    catalogue :
        The section family, whose ratio fixes the wall thickness.
    loads :
        Force applied at every node, in newtons.
    """

    structure: Structure
    catalogue: TubeFamily
    loads: Float[np.ndarray, "nodes 3"]


class Cotangents(NamedTuple):
    """
    A cotangent pulled back onto the inputs a derivative was taken in.

    Attributes
    ----------
    xyz :
        Cotangent on every node coordinate.
    diameter :
        Cotangent on every member's diameter.
    """

    xyz: Float[np.ndarray, "nodes 3"]
    diameter: Float[np.ndarray, "members"]


class MemberState(NamedTuple):
    """
    What one member's reading is a function of.

    Attributes
    ----------
    start :
        Position of the member's first end, in this repository's axes.
    end :
        Position of the member's second end, in this repository's axes.
    diameter :
        Outer diameter of the member.
    displacement :
        Solved displacement of both its ends, in the solver's axes.
    """

    start: Float[Array, "*members 3"]
    end: Float[Array, "*members 3"]
    diameter: Float[Array, "*members"]
    displacement: Float[Array, "*members dofs_member"]


class PreparedFrame(NamedTuple):
    """
    One frame assembled and factorized, before any load is applied.

    Attributes
    ----------
    free :
        Indices of the unrestrained degrees of freedom.
    factorized :
        The free stiffness, decomposed, ready for any number of right-hand sides.
    indexed :
        Which twelve degrees of freedom each member spans.
    numbered :
        Where each node's degrees of freedom sit, the solver renumbering them.
    members :
        Every member's ends and diameter, in this repository's axes.
    """

    free: Int[np.ndarray, "dofs_free"]
    factorized: SuperLU
    indexed: Int[np.ndarray, "members dofs_member"]
    numbered: Int[np.ndarray, "nodes"]
    members: MemberState


def vertical_upward(
    vectors: Float[np.ndarray, "rows 3"],
) -> Float[np.ndarray, "rows 3"]:
    """
    The same vectors with the repository's vertical where PyNite expects one.

    Parameters
    ----------
    vectors :
        Positions or forces, one row each, in the repository's axes.

    Returns
    -------
    turned :
        The same quantities about a vertical PyNite agrees with.

    Notes
    -----
    A rotation, not a permutation: exchanging two axes would flip handedness
    and every moment sign with it.
    """
    turned = np.asarray(vectors)

    return np.stack([turned[:, 0], turned[:, 2], -turned[:, 1]], axis=1)


def _node_name(node: int) -> str:
    """
    What one node is called, PyNite keying its containers by name.
    """
    return f"node-{node}"


def _member_name(member: int) -> str:
    """
    What one member is called, PyNite keying its containers by name.
    """
    return f"member-{member}"


def frame_model(
    structure: Structure,
    xyz: Float[np.ndarray, "nodes 3"],
    sections: MemberSections,
) -> FEModel3D:
    """
    The frame a solve needs, assembled in SI.

    Parameters
    ----------
    structure :
        The connectivity and the supported nodes.
    xyz :
        Position of every node, in millimeters and the solver's axes.
    sections :
        The tube every member is analyzed as, and the steel it is rolled from.

    Returns
    -------
    model :
        The assembled frame, unsolved.

    Notes
    -----
    Supports restrain translation and leave rotation free, the pinned base the
    form finder saw. The torsion constant is the sum of the two second moments,
    exact for a circular section.
    """
    steel = sections.material
    elasticity = float(steel.e_mod) * MEGAPASCAL
    shear = elasticity / (2.0 * (1.0 + POISSONS_RATIO))

    positions = np.asarray(xyz) * MILLIMETER
    edges = np.asarray(structure.edges)
    restrained = {int(node) for node in np.asarray(structure.supports).ravel()}

    areas = np.asarray(sections.area) * MILLIMETER**2
    inertias = np.asarray(sections.second_moment) * MILLIMETER**4

    model = FEModel3D()
    model.add_material("steel", elasticity, shear, POISSONS_RATIO, 1.0)

    for node in range(positions.shape[0]):
        model.add_node(_node_name(node), *(float(value) for value in positions[node]))

    for node in sorted(restrained):
        model.def_support(_node_name(node), True, True, True, False, False, False)

    for member in range(edges.shape[0]):
        name = _member_name(member)
        inertia = float(inertias[member])
        model.add_section(name, float(areas[member]), inertia, inertia, 2.0 * inertia)
        first = _node_name(int(edges[member, 0]))
        second = _node_name(int(edges[member, 1]))
        model.add_member(name, first, second, "steel", name)

    return model


def member_rigidity(
    diameter: Float[Array, ""],
    catalogue: TubeFamily,
) -> SectionRigidity:
    """
    The three rigidities one diameter implies, in SI.

    Parameters
    ----------
    diameter :
        Outer diameter of the member, in millimeters.
    catalogue :
        The section family, whose ratio fixes the wall thickness.

    Returns
    -------
    rigidity :
        Axial, bending and torsional rigidity.
    """
    section = catalogue(diameter * MILLIMETER)
    elasticity = section.material.e_mod * MEGAPASCAL
    shear = elasticity / (2.0 * (1.0 + POISSONS_RATIO))

    return SectionRigidity(
        axial=elasticity * section.area,
        bending=elasticity * section.second_moment,
        torsional=shear * TORSION_FACTOR * section.second_moment,
    )


def member_stiffness(
    start: Float[Array, "3"],
    end: Float[Array, "3"],
    diameter: Float[Array, ""],
    catalogue: TubeFamily,
) -> Float[Array, "dofs_member dofs_member"]:
    """
    One member's global elastic stiffness, in the solver's own axes.

    Parameters
    ----------
    start :
        Position of the member's first end, in this repository's axes.
    end :
        Position of the member's second end, in this repository's axes.
    diameter :
        Outer diameter of the member.
    catalogue :
        The section family, whose ratio fixes the wall thickness.

    Returns
    -------
    stiffness :
        Twelve by twelve, about the axes the solver assembled in.

    Notes
    -----
    Stated in the solver's axes and differentiated with respect to this
    repository's, so the chain rule crosses the two conventions once, here.
    """
    turning = jnp.asarray(ROTATION)
    rigidity = member_rigidity(diameter, catalogue)
    first = turning @ start * MILLIMETER
    second = turning @ end * MILLIMETER

    return stiffness_global(first, second, rigidity)


def member_actions(
    state: MemberState,
    catalogue: TubeFamily,
) -> Float[Array, "readings"]:
    """
    Everything the stage reports about one member, from what it depends on.

    Parameters
    ----------
    state :
        The member's ends, its diameter and its solved end displacements.
    catalogue :
        The section family, whose ratio fixes the wall thickness.

    Returns
    -------
    reading :
        Axial force, both end moments about the first transverse axis, then
        both about the second.

    Notes
    -----
    The forward pass and the adjoint read this one function, so the reported
    force and its slope cannot disagree. Multiplying the element stiffness by
    the displacement also replaces a Python loop over the solver's own end
    forces with one mapped pass.
    """
    stiffness = member_stiffness(state.start, state.end, state.diameter, catalogue)
    acting = stiffness @ state.displacement
    inverse = jnp.asarray(ROTATION).T

    force = inverse @ acting[0:3]
    ends = jnp.stack([inverse @ acting[3:6], inverse @ acting[9:12]])

    frame = member_frame(state.start, state.end)
    axial = -jnp.dot(force, frame[0])
    flip = jnp.asarray(DIAGRAM_SIGN)
    major = ends @ frame[1] * flip
    minor = ends @ frame[2] * flip
    bending = jnp.concatenate([major, minor]) / NEWTON_MILLIMETER

    return jnp.concatenate([jnp.atleast_1d(axial), bending])


def _pull_one_reading(
    state: MemberState,
    cotangent: Float[Array, "readings"],
    catalogue: TubeFamily,
) -> MemberState:
    """
    One member's reading, pulled back onto everything it was read from.
    """

    def reading(acting: MemberState) -> Float[Array, "readings"]:
        return member_actions(acting, catalogue)

    _, backward = jax.vjp(reading, state)
    (pulled,) = backward(cotangent)

    return pulled


# Compiled once and reused: fixed programs over fixed shapes, and dispatching
# them uncompiled costs far more than running them.
_STIFFNESS_SLOPES = jax.jit(
    jax.vmap(jax.jacfwd(member_stiffness, argnums=(0, 1, 2)), in_axes=(0, 0, 0, None))
)

_READ_MEMBERS = jax.vmap(member_actions, in_axes=(0, None))

_READ_CASES = jax.jit(
    jax.vmap(_READ_MEMBERS, in_axes=(MemberState(None, None, None, 0), None))
)

_PULL_READINGS = jax.jit(jax.vmap(_pull_one_reading, in_axes=(0, 0, None)))


def refuse_unusable(
    xyz: Float[np.ndarray, "nodes 3"],
    sections: MemberSections,
) -> None:
    """
    Refuse a model the solver would accept and answer nonsense about.

    Raises
    ------
    ValueError
        If a coordinate is not finite, or a section property is not positive.
    """
    positions = np.asarray(xyz)
    if not np.all(np.isfinite(positions)):
        raise ValueError("a node coordinate is not finite")

    areas = np.asarray(sections.area)
    inertias = np.asarray(sections.second_moment)
    if not (np.all(np.isfinite(areas)) and np.all(np.isfinite(inertias))):
        raise ValueError("a section property is not finite")
    if not (np.all(areas > 0.0) and np.all(inertias > 0.0)):
        raise ValueError("a section property is not positive")


def refuse_upright(
    xyz: Float[np.ndarray, "nodes 3"],
    edges: Int[np.ndarray, "members 2"],
) -> None:
    """
    Refuse a member the reporting convention cannot resolve a bending about.

    Raises
    ------
    ValueError
        If a member lies within `REFERENCE_MARGIN` of the vertical, or has
        coincident ends.

    Notes
    -----
    The stiffness is invariant to the roll, but `member_frame` completes its
    pair against the vertical and a vertical member has none. Nothing here is
    refused for a shell, whose members all lean.
    """
    positions = np.asarray(xyz)
    spans = np.asarray(edges)
    along = positions[spans[:, 1]] - positions[spans[:, 0]]
    lengths = np.linalg.norm(along, axis=1)
    if not np.all(lengths > 0.0):
        raise ValueError("a member has coincident ends")

    leaning = np.linalg.norm(along[:, :2], axis=1) / lengths
    if np.any(leaning < REFERENCE_MARGIN):
        upright = int(np.argmin(leaning))
        raise ValueError(
            f"member {upright} lies within {REFERENCE_MARGIN} of the vertical, "
            "which the reporting convention cannot resolve a bending about"
        )


def prepared_frame(
    problem: FrameProblem,
    xyz: Float[np.ndarray, "nodes 3"],
    diameters: Float[np.ndarray, "members"],
) -> PreparedFrame:
    """
    Assemble the frame once and factorize it once, before any load is applied.

    Parameters
    ----------
    problem :
        The frame and its section family. Its loads are not read here.
    xyz :
        Position of every node, in this repository's axes and millimeters.
    diameters :
        Outer diameter of every member.

    Returns
    -------
    prepared :
        The decomposed free stiffness and the per-member views a reading needs.

    Raises
    ------
    RuntimeError
        If a support is prescribed a displacement, or the frame is singular.

    Notes
    -----
    The stiffness does not depend on the loading, so nothing here does either.
    PyNite factorizes afresh per load combination; holding the decomposition
    turns every case after the first into a back-substitution.
    """
    structure = problem.structure
    edges = np.asarray(structure.edges)
    positions = np.asarray(xyz)
    sizes = jnp.asarray(np.asarray(diameters))
    sections = problem.catalogue(sizes)

    refuse_unusable(positions, sections)
    refuse_upright(positions, edges)

    upright = Structure(
        nodes=vertical_upward(positions),
        edges=edges,
        supports=np.asarray(structure.supports),
    )
    model = frame_model(upright, upright.nodes, sections)
    model.add_load_combo(COMBO_NAME, {CASE_NAME: 1.0})
    _prepare_model(model)

    free, _, held = _partition_D(model)
    free = np.asarray(free, dtype=np.int64).ravel()
    if not np.all(np.asarray(held) == 0.0):
        raise RuntimeError(
            "a support is prescribed a displacement; this solve assumes none"
        )

    stiffness = model.Ke(COMBO_NAME, log=False, check_stability=False, sparse=True)
    restricted = csc_matrix(stiffness.tocsc()[free][:, free])

    # A singular frame is a warning and a garbage number further down.
    with warnings.catch_warnings():
        warnings.simplefilter("error", MatrixRankWarning)
        try:
            decomposed = splu(restricted)
        except (MatrixRankWarning, RuntimeError) as singular:
            raise RuntimeError(
                "the frame is singular; the solve is unusable"
            ) from singular

    numbered = np.array(
        [model.nodes[_node_name(node)].ID for node in range(len(model.nodes))]
    )
    within = np.arange(DOF_PER_NODE)
    starts = numbered[edges[:, 0]][:, None] * DOF_PER_NODE + within
    ends = numbered[edges[:, 1]][:, None] * DOF_PER_NODE + within
    indexed = np.concatenate([starts, ends], axis=1)

    members = MemberState(
        start=jnp.asarray(positions[edges[:, 0]]),
        end=jnp.asarray(positions[edges[:, 1]]),
        diameter=sizes,
        displacement=jnp.zeros((edges.shape[0], DOF_PER_MEMBER)),
    )

    return PreparedFrame(free, decomposed, indexed, numbered, members)


def case_displacements(
    prepared: PreparedFrame,
    loads: Float[np.ndarray, "load_cases nodes 3"],
) -> Float[np.ndarray, "load_cases dofs"]:
    """
    Solve every load case against the decomposition already held.

    Parameters
    ----------
    prepared :
        The assembled and factorized frame.
    loads :
        Force applied at every node, per load case, in newtons.

    Returns
    -------
    displaced :
        Every degree of freedom's displacement, per case, in the solver's axes.

    Notes
    -----
    The right-hand side is the nodal load scattered onto its degrees of freedom;
    under nodal loading the fixed-end reactions PyNite would subtract are zero.
    """
    turned = np.stack([vertical_upward(load_case) for load_case in np.asarray(loads)])
    cases = turned.shape[0]
    num_dofs = prepared.numbered.size * DOF_PER_NODE

    forcing = np.zeros((num_dofs, cases))
    rows = (prepared.numbered[:, None] * DOF_PER_NODE + np.arange(3)).ravel()
    for load_case in range(cases):
        np.add.at(forcing[:, load_case], rows, turned[load_case].ravel())

    solved = np.asarray(prepared.factorized.solve(forcing[prepared.free]))
    if not np.all(np.isfinite(solved)):
        raise RuntimeError("the solved displacements are not finite")

    displaced = np.zeros((cases, num_dofs))
    displaced[:, prepared.free] = solved.T

    return displaced


def member_forces(
    problem: FrameProblem,
    xyz: Float[np.ndarray, "nodes 3"],
    diameters: Float[np.ndarray, "members"],
    loads: Float[np.ndarray, "*load_cases nodes 3"],
    prepared: PreparedFrame | None = None,
) -> MemberForces:
    """
    Internal forces of a space frame, under one load case or several.

    Parameters
    ----------
    problem :
        The frame and its section family.
    xyz :
        Position of every node, in this repository's axes and millimeters.
    diameters :
        Outer diameter of every member.
    loads :
        Force applied at every node, with or without a leading load case axis.
    prepared :
        The frame already assembled at this geometry and these diameters, or
        None to prepare one.

    Returns
    -------
    forces :
        Axial force and both end moments, carrying a load case axis exactly
        when the loading did.

    Notes
    -----
    Several load cases cost one assembly and one factorization.
    """
    applied = np.asarray(loads)
    stacked = applied.ndim == 3
    cases = applied if stacked else applied[None, ...]

    if prepared is None:
        prepared = prepared_frame(problem, xyz, diameters)
    displaced = case_displacements(prepared, cases)
    moved = prepared.members._replace(
        displacement=jnp.asarray(displaced[:, prepared.indexed])
    )

    acting = np.asarray(_READ_CASES(moved, problem.catalogue))
    if not stacked:
        acting = acting[0]

    return MemberForces(
        axial_force=acting[..., READING_AXIAL],
        moment_major=acting[..., READING_MAJOR],
        moment_minor=acting[..., READING_MINOR],
    )


def force_cotangents(
    problem: FrameProblem,
    xyz: Float[np.ndarray, "nodes 3"],
    diameters: Float[np.ndarray, "members"],
    cotangent: MemberForces,
    prepared: PreparedFrame | None = None,
) -> Cotangents:
    """
    Pull a cotangent on the reported forces back onto the inputs.

    Parameters
    ----------
    problem :
        The frame, its section family and its one load case.
    xyz :
        Position of every node, in this repository's axes and millimeters.
    diameters :
        Outer diameter of every member.
    cotangent :
        Cotangent on each reported quantity, without a load case axis.
    prepared :
        The frame already assembled at this geometry and these diameters, or
        None to prepare one.

    Returns
    -------
    pulled :
        Cotangent on every node coordinate and every diameter.

    Notes
    -----
    One solve, whatever the parameter count: the cotangent is pulled through
    each member's reading, gathered into one adjoint load, solved against the
    decomposed stiffness, and contracted against element-local stiffness
    derivatives. Nodal loads do not move with the geometry, so no load
    derivative enters.
    """
    if prepared is None:
        prepared = prepared_frame(problem, xyz, diameters)
    displaced = case_displacements(prepared, np.asarray(problem.loads)[None, ...])[0]
    state = prepared.members._replace(
        displacement=jnp.asarray(displaced[prepared.indexed])
    )
    edges = np.asarray(problem.structure.edges)
    num_dofs = displaced.size

    seed = np.concatenate(
        [
            np.asarray(cotangent.axial_force)[:, None],
            np.asarray(cotangent.moment_major),
            np.asarray(cotangent.moment_minor),
        ],
        axis=1,
    )
    explicit = _PULL_READINGS(state, jnp.asarray(seed), problem.catalogue)

    # One adjoint load, gathered from every member that reads a displacement.
    acting = np.zeros(num_dofs)
    np.add.at(acting, prepared.indexed, np.asarray(explicit.displacement))

    adjoint = np.zeros(num_dofs)
    adjoint[prepared.free] = prepared.factorized.solve(acting[prepared.free])

    slopes = _STIFFNESS_SLOPES(
        state.start, state.end, state.diameter, problem.catalogue
    )
    slope_start, slope_end, slope_diameter = (np.asarray(block) for block in slopes)
    borne = adjoint[prepared.indexed]
    local = displaced[prepared.indexed]

    implicit_start = -np.einsum("mi,mija,mj->ma", borne, slope_start, local)
    implicit_end = -np.einsum("mi,mija,mj->ma", borne, slope_end, local)
    implicit_diameter = -np.einsum("mi,mij,mj->m", borne, slope_diameter, local)

    by_node = np.zeros((np.asarray(xyz).shape[0], 3))
    np.add.at(by_node, edges[:, 0], np.asarray(explicit.start) + implicit_start)
    np.add.at(by_node, edges[:, 1], np.asarray(explicit.end) + implicit_end)

    return Cotangents(by_node, np.asarray(explicit.diameter) + implicit_diameter)
