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
A three-dimensional frame analysis in PyNite, and what it takes to read it.

**Nothing here is traced.** PyNite is plain Python over NumPy and SciPy with no
derivative of its own, so this module is the forward pass alone and the adjoint
that reaches it is assembled by hand from the matrices PyNite does expose. The
planar backend beside it cannot analyze a shell — its solver is two-dimensional
and refuses a geometry that leaves the plane — and the traced backend is the
one this repository owns. What is wanted for three dimensions is a solver
someone else wrote, which is what this is.

Three conventions are worth stating, because two of them differ from the traced
backend and the third is easy to get backwards.

**Tension is positive here, and negative in PyNite.** Its local end-force
vector reports a bar in tension as a negative axial force, so every reading is
negated on the way out.

**Both bending components cross, and neither is combined.** A section's two
moments reach the check as separate fields and the check decides whether to
resolve them, so nothing here resolves them first. That matters more than it
looks: a solver's local axes are its own convention, so two correct solvers
split one physical bending differently and only the resultant is comparable
between them. Compare invariants, never components.

**A physical member segments.** PyNite splits a member wherever something is
attached along it and reports each piece separately. Under nodal loading alone
the split never happens, which is the only loading this schema carries, so the
first piece is the member — but a load along a span would break that and the
reading asserts it rather than assuming it.
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
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import MatrixRankWarning
from scipy.sparse.linalg import SuperLU
from scipy.sparse.linalg import splu

from normax.analysis import DOF_PER_NODE
from normax.analysis import MemberForces
from normax.analysis.element import REFERENCE_MARGIN
from normax.analysis.element import SectionRigidity
from normax.analysis.element import member_frame
from normax.analysis.element import stiffness_global
from normax.sections import MemberSections
from normax.sections import TubeFamily
from normax.structures import Structure
from normax.units import to_meters
from normax.units import to_newton_millimeters
from normax.units import to_pascals

# EN 1993-1-1 3.2.6, as the traced backend reads it, so both solve one steel.
POISSONS_RATIO = 0.3

# What the one load pattern and the one combination are called.
CASE_NAME = "applied"
COMBO_NAME = "design"

# What turns nodal actions into the bending diagram: one end flips, so that a
# uniform moment reads equal and of one sign, which is what the check states.
DIAGRAM_SIGN = np.array([-1.0, 1.0])

# Carries this repository's axes onto PyNite's. A rotation, so its transpose is
# its inverse and a moment turns like a force.
ROTATION = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])

# The polar moment of an annulus is what its two second moments add to.
TORSION_FACTOR = 2.0

# Where each end's force and moment sit in PyNite's global end-force vector.
FORCE_COLUMNS = ((0, 1, 2), (6, 7, 8))
MOMENT_COLUMNS = ((3, 4, 5), (9, 10, 11))


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
    **PyNite hard-codes its up direction as the second global axis**, and builds
    every local frame around it: the horizontal local axis is the one
    perpendicular to that vertical. This repository's vertical is the third
    axis. Handing the coordinates over unturned therefore asks a solver to
    orient its members about a horizontal, which decides nothing physical —
    a solve is invariant to it, and the forces come back correct — but the
    frames the moments are reported in mean nothing.

    A rotation, not a permutation: exchanging two axes reflects, which would
    flip handedness and every moment sign with it. This carries the third axis
    onto the second and the second onto the negative third, whose determinant
    is one.
    """
    turned = np.asarray(vectors)

    return np.stack([turned[:, 0], turned[:, 2], -turned[:, 1]], axis=1)


def vertical_downward(
    vectors: Float[np.ndarray, "rows 3"],
) -> Float[np.ndarray, "rows 3"]:
    """
    The inverse of `vertical_upward`, for reading a solved answer back.

    Parameters
    ----------
    vectors :
        Positions, forces or moments, one row each, in PyNite's axes.

    Returns
    -------
    turned :
        The same quantities about this repository's vertical.

    Notes
    -----
    A moment is a pseudovector, which under a reflection would not transform
    like a force. `vertical_upward` is a rotation and so is this, which is why
    one inverse serves both and why the handedness argument there was worth
    making.
    """
    turned = np.asarray(vectors)

    return np.stack([turned[:, 0], -turned[:, 2], turned[:, 1]], axis=1)


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
        Position of every node, in millimeters.
    sections :
        The tube every member is analyzed as, and the steel it is rolled from.

    Returns
    -------
    model :
        The assembled frame, unsolved.

    Notes
    -----
    Built in meters and pascals, because that is what the traced backend
    converts to before it solves, and two solvers reading one steel is the whole
    point of the comparison.

    **The torsion constant is the sum of the two second moments.** That is exact
    for a circular section rather than an approximation of one, the polar moment
    of an annulus being what the two bending moments add to.

    Supports restrain translation and leave rotation free, which is the pinned
    base the form finder saw.
    """
    steel = sections.material
    elasticity = float(to_pascals(steel.e_mod))
    shear = elasticity / (2.0 * (1.0 + POISSONS_RATIO))

    positions = np.asarray(to_meters(np.asarray(xyz)))
    edges = np.asarray(structure.edges)
    restrained = {int(node) for node in np.asarray(structure.supports).ravel()}

    # Square and fourth powers of a length, so the conversions compound.
    areas = np.asarray(sections.area) * 1.0e-6
    inertias = np.asarray(sections.second_moment) * 1.0e-12

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
        model.add_member(
            name,
            _node_name(int(edges[member, 0])),
            _node_name(int(edges[member, 1])),
            "steel",
            name,
        )

    return model


def _node_name(node: int) -> str:
    """
    What one node is called, PyNite keying its own containers by name.
    """
    return f"node-{node}"


def _member_name(member: int) -> str:
    """
    What one member is called, PyNite keying its own containers by name.
    """
    return f"member-{member}"


def apply_loads(
    model: FEModel3D,
    loads: Float[np.ndarray, "nodes 3"],
) -> None:
    """
    Register one load case on an assembled frame, and the combination reading it.

    Parameters
    ----------
    model :
        The assembled frame.
    loads :
        Force applied at every node, in newtons.

    Notes
    -----
    A node carrying nothing is skipped rather than registered at zero, which
    keeps the pattern the size of the loading rather than the size of the frame.
    """
    applied = np.asarray(loads)
    directions = ("FX", "FY", "FZ")

    for node in range(applied.shape[0]):
        for axis, direction in enumerate(directions):
            component = float(applied[node, axis])
            if component != 0.0:
                model.add_node_load(
                    _node_name(node), direction, component, case=CASE_NAME
                )

    model.add_load_combo(COMBO_NAME, {CASE_NAME: 1.0})


def _end_forces(
    model: FEModel3D,
    num_members: int,
) -> Float[np.ndarray, "members 12"]:
    """
    The global end-force vector of every member, as PyNite reports it.

    Parameters
    ----------
    model :
        The solved frame.
    num_members :
        How many members the frame was built with.

    Returns
    -------
    forces :
        Twelve components per member, in PyNite's own axes, units and signs.

    Raises
    ------
    RuntimeError
        If a member segmented, which nodal loading does not do.

    Notes
    -----
    **Global, not local.** The local vector would arrive resolved about axes
    PyNite chose for itself, and those are neither this repository's convention
    nor anything a second solver would reproduce. Read globally and projected
    here, the components answer to one stated convention instead, and the
    derivative of the reading needs the global element stiffness alone.
    """
    forces = np.empty((num_members, 12))

    for member in range(num_members):
        pieces = model.members[_member_name(member)].sub_members
        if len(pieces) != 1:
            raise RuntimeError(
                f"member {member} segmented into {len(pieces)} pieces; this "
                "reading holds only under nodal loading"
            )
        whole = next(iter(pieces.values()))
        forces[member] = np.asarray(whole.F(COMBO_NAME)).ravel()

    return forces


def projected_forces(
    xyz: Float[np.ndarray, "nodes 3"],
    edges: Int[np.ndarray, "members 2"],
    global_forces: Float[np.ndarray, "members 12"],
) -> MemberForces:
    """
    Global end forces resolved onto the convention the check is stated in.

    Parameters
    ----------
    xyz :
        Position of every node, in this repository's axes and millimeters.
    edges :
        The two nodes every member spans.
    global_forces :
        Twelve global components per member, in newtons and newton meters.

    Returns
    -------
    forces :
        Axial force, both end moments, both shears and the torsion.

    Notes
    -----
    Axial force and torsion are the components along the member axis and are
    invariant: every convention agrees on them. The two bending components are
    not — they are the transverse moment resolved onto `member_frame`'s pair,
    and a solver that completed its own pair differently reports different
    numbers for the same bending. Only the invariants of the pair are shared,
    which is why the check reads those and not these.

    **A shear pairs with the moment it differentiates**, so the shear reported
    beside the bending about one transverse axis is the force along the other.
    Reading the letters instead puts a real force in the wrong component.
    """
    positions = np.asarray(xyz)
    spans = np.asarray(edges)
    starts = positions[spans[:, 0]]
    ends = positions[spans[:, 1]]
    frames = np.asarray(jax.vmap(member_frame)(jnp.asarray(starts), jnp.asarray(ends)))

    axis = frames[:, 0]
    first = frames[:, 1]
    second = frames[:, 2]

    forces_at = [global_forces[:, columns] for columns in FORCE_COLUMNS]
    moments_at = [global_forces[:, columns] for columns in MOMENT_COLUMNS]

    # Tension is positive here and the first end's action points back along the axis.
    axial = -np.einsum("mi,mi->m", forces_at[0], axis)
    torsion = np.einsum("mi,mi->m", moments_at[0], axis)

    bending_first = np.stack(
        [np.einsum("mi,mi->m", m, first) for m in moments_at], axis=1
    )
    bending_second = np.stack(
        [np.einsum("mi,mi->m", m, second) for m in moments_at], axis=1
    )

    return MemberForces(
        axial_force=axial,
        moment_major=to_newton_millimeters(bending_first * DIAGRAM_SIGN),
        moment_minor=to_newton_millimeters(bending_second * DIAGRAM_SIGN),
        shear_major=np.einsum("mi,mi->m", forces_at[0], second),
        shear_minor=np.einsum("mi,mi->m", forces_at[0], first),
        torsion_moment=to_newton_millimeters(torsion),
    )


def member_forces(
    structure: Structure,
    xyz: Float[np.ndarray, "nodes 3"],
    sections: MemberSections,
    loads: Float[np.ndarray, "nodes 3"],
) -> MemberForces:
    """
    Internal forces of a space frame under one load case.

    Parameters
    ----------
    structure :
        The connectivity and the supported nodes.
    xyz :
        Position of every node, in millimeters.
    sections :
        The tube every member is analyzed as.
    loads :
        Force applied at every node, in newtons.

    Returns
    -------
    forces :
        Axial force, both end moments, both shears and the torsion, in the units
        and signs the rest of the pipeline states.

    Notes
    -----
    The axial force is taken at the first end and negated: nodal loading leaves
    it constant along the span, and PyNite calls a bar in tension negative.
    Moments are reported at both ends because nodal loading leaves them linear
    in between, which is what makes the first row of EN 1993-1-1 Table B.3 exact
    downstream rather than approximate.
    """
    edges = np.asarray(structure.edges)
    model = _solved_frame(structure, xyz, sections, loads)

    num_members = int(edges.shape[0])
    solved = _end_forces(model, num_members)

    turned = np.concatenate(
        [
            vertical_downward(solved[:, columns])
            for columns in (*FORCE_COLUMNS, *MOMENT_COLUMNS)
        ],
        axis=1,
    )
    ordered = turned[:, [0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11]]

    return projected_forces(np.asarray(xyz), edges, ordered)


def bending_resultant(
    moment_major: Float[np.ndarray, "members ends"],
    moment_minor: Float[np.ndarray, "members ends"],
) -> Float[np.ndarray, "members ends"]:
    """
    The bending a circular section actually feels, whatever axes reported it.

    Parameters
    ----------
    moment_major :
        Major-axis moment at each end of every member.
    moment_minor :
        Minor-axis moment at each end of every member.

    Returns
    -------
    resultant :
        Magnitude of the two combined, per end.

    Notes
    -----
    **This is the quantity to compare two solvers on.** Local axes are a
    solver's own convention, so two correct answers split one physical bending
    differently and their components need not agree; the resultant is the same
    number in every frame. An axisymmetric section is what makes it meaningful
    rather than merely invariant.

    It is deliberately not what crosses the schema. Resolving the two is the
    check's decision and the check has a switch for it.
    """
    return np.hypot(np.asarray(moment_major), np.asarray(moment_minor))


def stiffness_free(
    model: FEModel3D,
) -> tuple[Int[np.ndarray, "dofs_free"], csc_matrix]:
    """
    The free degrees of freedom of a solved frame, and the stiffness among them.

    Parameters
    ----------
    model :
        The solved frame.

    Returns
    -------
    partitioned :
        Indices of the free degrees of freedom, and the stiffness restricted to
        them, sparse.

    Notes
    -----
    A solved linear system is an implicit function, so differentiating
    equilibrium gives the derivative of the displacement as another solve
    against this same matrix. One factorization therefore serves every
    parameter, which is what makes an exact rule cheaper than a perturbation of
    each — and it is the solver's own matrix, so the rule differentiates the
    model that was actually solved.

    Sparse and left unfactorized: the caller decides how many right-hand sides
    it has, and a factorization is worth reusing across all of them.
    """
    free, _, _ = _partition_D(model)
    indices = np.asarray(free, dtype=np.int64).ravel()
    stiffness = model.Ke(COMBO_NAME, log=False, check_stability=False, sparse=True)
    restricted = csc_matrix(stiffness.tocsc()[indices][:, indices])

    return indices, restricted


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

    Notes
    -----
    The split is the schema's own: the coordinates and the diameters are what
    the stage calls differentiable, and everything else describes the problem
    they are varied within. Grouping the rest keeps a derivative's signature to
    the point it is taken at.
    """

    structure: Structure
    catalogue: TubeFamily
    loads: Float[np.ndarray, "nodes 3"]


class ReadingCotangent(NamedTuple):
    """
    A cotangent on everything the stage reports as differentiable.

    Attributes
    ----------
    axial_force :
        Cotangent on each member's axial force.
    moment_major :
        Cotangent on each end moment about the first transverse axis.
    moment_minor :
        Cotangent on each end moment about the second transverse axis.
    """

    axial_force: Float[np.ndarray, "members"]
    moment_major: Float[np.ndarray, "members ends"]
    moment_minor: Float[np.ndarray, "members ends"]


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


class AdjointState(NamedTuple):
    """
    One solve, factorized once, with everything either derivative rule reads.

    Attributes
    ----------
    displaced :
        Solved displacement of every degree of freedom, in the solver's axes.
    free :
        Indices of the unrestrained degrees of freedom.
    factorized :
        The free stiffness, decomposed, ready for any number of right-hand sides.
    indexed :
        Which twelve degrees of freedom each member spans.
    starts :
        Position of every member's first end.
    ends :
        Position of every member's second end.
    diameters :
        Outer diameter of every member.
    moved :
        Solved displacement of every member's twelve degrees of freedom.

    Notes
    -----
    The factorization is the expensive step and it is shared. A forward rule
    spends one back-substitution per parameter against it and a reverse rule
    spends exactly one, whatever the parameter count — which is the difference
    between the two endpoints rather than a detail of either.
    """

    displaced: Float[np.ndarray, "dofs"]
    free: Int[np.ndarray, "dofs_free"]
    factorized: SuperLU
    indexed: Int[np.ndarray, "members dofs_member"]
    starts: Float[Array, "members 3"]
    ends: Float[Array, "members 3"]
    diameters: Float[Array, "members"]
    moved: Float[Array, "members dofs_member"]


class Jacobian(NamedTuple):
    """
    How every reported force moves with the geometry and with the sections.

    Attributes
    ----------
    axial_force_xyz :
        Derivative of each member's axial force with respect to every node.
    axial_force_diameter :
        Derivative of each member's axial force with respect to every diameter.
    moment_major_xyz :
        Derivative of each end moment about the first transverse axis with
        respect to every node.
    moment_major_diameter :
        The same, with respect to every diameter.
    moment_minor_xyz :
        Derivative of each end moment about the second transverse axis with
        respect to every node.
    moment_minor_diameter :
        The same, with respect to every diameter.

    Notes
    -----
    Every block is dense and every block is filled. The planar backend beside
    this one carries no minor-axis blocks, a plane frame having no out-of-plane
    bending to differentiate, so nothing here is a structural zero standing in
    for a derivative that went unreached.
    """

    axial_force_xyz: Float[np.ndarray, "members nodes 3"]
    axial_force_diameter: Float[np.ndarray, "members members"]
    moment_major_xyz: Float[np.ndarray, "members ends nodes 3"]
    moment_major_diameter: Float[np.ndarray, "members ends members"]
    moment_minor_xyz: Float[np.ndarray, "members ends nodes 3"]
    moment_minor_diameter: Float[np.ndarray, "members ends members"]


# Where each reported quantity sits in one member's flattened reading.
READING_AXIAL = 0
READING_MAJOR = (1, 2)
READING_MINOR = (3, 4)


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

    Notes
    -----
    Read through the same section family the check itself reads, so the two
    stages cannot drift apart in what a tube is. The element stiffness is
    exactly linear in all three, so the derivative with respect to a diameter
    is the section geometry's derivative and nothing else — no perturbation of
    the element, and no step size.
    """
    section = catalogue(to_meters(diameter))
    elasticity = to_pascals(section.material.e_mod)
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
    repository's, so the chain rule crosses the two conventions once, here,
    rather than at every call site. The matrix itself is held equal to the
    solver's own to near machine precision by test, which is what licenses
    differentiating this instead of it.
    """
    turning = jnp.asarray(ROTATION)
    rigidity = member_rigidity(diameter, catalogue)
    first = to_meters(turning @ start)
    second = to_meters(turning @ end)

    return stiffness_global(first, second, rigidity)


def member_reading(
    start: Float[Array, "3"],
    end: Float[Array, "3"],
    diameter: Float[Array, ""],
    displacement: Float[Array, "dofs_member"],
    catalogue: TubeFamily,
) -> Float[Array, "reading"]:
    """
    What the check reads off one member, as a function of all it depends on.

    Parameters
    ----------
    start :
        Position of the member's first end, in this repository's axes.
    end :
        Position of the member's second end, in this repository's axes.
    diameter :
        Outer diameter of the member.
    displacement :
        Solved displacement of both its ends, in the solver's axes.
    catalogue :
        The section family, whose ratio fixes the wall thickness.

    Returns
    -------
    reading :
        Axial force, then both end moments about the first transverse axis,
        then both about the second.

    Notes
    -----
    **This one function carries the entire explicit derivative.** It rebuilds
    the element stiffness, multiplies by the displacement, turns the answer into
    this repository's axes and resolves it onto the stated convention. Its
    derivative in the first three arguments is every way a reading moves at
    fixed displacement; its derivative in the fourth is what the implicit term
    gets contracted with. Nothing about the element is differentiated by hand.

    The result is flat rather than named because a Jacobian assembly slices it,
    and one five-wide vector states that slicing in a single place.
    """
    stiffness = member_stiffness(start, end, diameter, catalogue)
    acting = stiffness @ displacement
    inverse = jnp.asarray(ROTATION).T

    force = inverse @ acting[0:3]
    moment_start = inverse @ acting[3:6]
    moment_end = inverse @ acting[9:12]

    frame = member_frame(start, end)
    axial = -jnp.dot(force, frame[0])
    ends = jnp.stack([moment_start, moment_end])
    first = ends @ frame[1] * jnp.asarray(DIAGRAM_SIGN)
    second = ends @ frame[2] * jnp.asarray(DIAGRAM_SIGN)
    bending = to_newton_millimeters(jnp.concatenate([first, second]))
    reading = jnp.concatenate([jnp.atleast_1d(axial), bending])

    return reading


def refuse_unusable(
    xyz: Float[np.ndarray, "nodes 3"],
    sections: MemberSections,
) -> None:
    """
    Refuse a model the solver would accept and answer nonsense about.

    Parameters
    ----------
    xyz :
        Position of every node.
    sections :
        The tube every member is analyzed as.

    Raises
    ------
    ValueError
        If a coordinate is not finite, or a section property is not positive.

    Notes
    -----
    Nothing downstream validates these. A section of zero area assembles a
    singular stiffness and a coordinate of nan propagates silently through the
    whole solve, and in both cases what comes back is a number rather than a
    complaint. An optimizer that has wandered somewhere impossible is better
    told here than three stages later.
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

    Parameters
    ----------
    xyz :
        Position of every node.
    edges :
        The two nodes every member spans.

    Raises
    ------
    ValueError
        If a member lies too near the vertical.

    Notes
    -----
    `member_frame` completes its transverse pair against the vertical, so a
    vertical member has no pair and a nearly vertical one has an ill-conditioned
    one. The stiffness does not care — it is invariant to the roll — but the two
    reported bending components would be arbitrary and their derivative large
    and meaningless. **Nothing here is refused for a shell**, whose members all
    lean; a frame with columns wants either the planar backend or a convention
    stated for it.
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


def _solved_frame(
    structure: Structure,
    xyz: Float[np.ndarray, "nodes 3"],
    sections: MemberSections,
    loads: Float[np.ndarray, "nodes 3"],
) -> FEModel3D:
    """
    One assembled, loaded and solved frame.

    Parameters
    ----------
    structure :
        The connectivity and the supported nodes.
    xyz :
        Position of every node, in this repository's axes and millimeters.
    sections :
        The tube every member is analyzed as.
    loads :
        Force applied at every node, in newtons.

    Returns
    -------
    model :
        The solved frame, in the solver's axes.

    Notes
    -----
    Shared so that the forward pass and the derivative solve one identical
    model. A stage that built them apart could report a force and a slope that
    belong to different structures, and nothing downstream would notice.
    """
    upright = Structure(
        nodes=vertical_upward(np.asarray(xyz)),
        edges=np.asarray(structure.edges),
        supports=np.asarray(structure.supports),
    )
    refuse_unusable(xyz, sections)
    refuse_upright(np.asarray(xyz), np.asarray(structure.edges))

    model = frame_model(upright, upright.nodes, sections)
    apply_loads(model, vertical_upward(np.asarray(loads)))

    # A singular factorization is a warning here and a finite garbage number
    # downstream, so it is promoted before anything reads the answer.
    with warnings.catch_warnings():
        warnings.simplefilter("error", MatrixRankWarning)
        try:
            model.analyze_linear(check_stability=False)
        except MatrixRankWarning as singular:
            raise RuntimeError(
                "the frame is singular; the solve is unusable"
            ) from singular

    if not np.all(np.isfinite(np.asarray(model.D(COMBO_NAME)))):
        raise RuntimeError("the solved displacements are not finite")

    return model


def _member_dofs(
    model: FEModel3D,
    edges: Int[np.ndarray, "members 2"],
) -> Int[np.ndarray, "members dofs_member"]:
    """
    Which global degrees of freedom each member's end forces are indexed by.

    Parameters
    ----------
    model :
        The solved frame.
    edges :
        The two nodes every member spans.

    Returns
    -------
    indexed :
        Twelve global degree-of-freedom indices per member.

    Notes
    -----
    The solver renumbers its nodes, so a node's position in the displacement
    vector is read off the model rather than assumed to be the order it was
    added in.
    """
    numbered = np.array(
        [model.nodes[_node_name(node)].ID for node in range(len(model.nodes))]
    )
    within = np.arange(DOF_PER_NODE)
    starts = numbered[edges[:, 0]][:, None] * DOF_PER_NODE + within
    ends = numbered[edges[:, 1]][:, None] * DOF_PER_NODE + within

    return np.concatenate([starts, ends], axis=1)


def adjoint_state(
    problem: FrameProblem,
    xyz: Float[np.ndarray, "nodes 3"],
    diameters: Float[np.ndarray, "members"],
) -> AdjointState:
    """
    Solve once, factorize once, and gather what either derivative rule reads.

    Parameters
    ----------
    problem :
        The frame, its section family and its loading.
    xyz :
        Position of every node, in this repository's axes and millimeters.
    diameters :
        Outer diameter of every member.

    Returns
    -------
    state :
        The solved displacements, the decomposed free stiffness, and the
        per-member views both rules index by.

    Notes
    -----
    The solver's matrix and the solver's displacements, so whatever is
    differentiated afterwards is a statement about the model that was solved.
    Its nodes are renumbered internally, so a member's degrees of freedom are
    read off the model rather than assumed from the order they were added in.
    """
    structure = problem.structure
    edges = np.asarray(structure.edges)
    model = _solved_frame(structure, xyz, problem.catalogue(diameters), problem.loads)

    displaced = np.asarray(model.D(COMBO_NAME)).ravel()
    free, restricted = stiffness_free(model)
    indexed = _member_dofs(model, edges)
    positions = np.asarray(xyz)

    return AdjointState(
        displaced=displaced,
        free=free,
        factorized=splu(restricted),
        indexed=indexed,
        starts=jnp.asarray(positions[edges[:, 0]]),
        ends=jnp.asarray(positions[edges[:, 1]]),
        diameters=jnp.asarray(np.asarray(diameters)),
        moved=jnp.asarray(displaced[indexed]),
    )


def _element_slopes(
    problem: FrameProblem,
    state: AdjointState,
) -> tuple[Float[Array, "..."], Float[Array, "..."], Float[Array, "..."]]:
    """
    How each member's global stiffness moves with its ends and its diameter.

    Parameters
    ----------
    problem :
        The frame, its section family and its loading.
    state :
        The solved frame.

    Returns
    -------
    slopes :
        Derivative of the twelve by twelve block with respect to the first end,
        the second end, and the diameter.

    Notes
    -----
    Taken by autodiff of the element this repository states and holds equal to
    the solver's own, so these are exact and no step size enters. They are
    element-local, so the whole set is one mapped pass over the members rather
    than anything that touches the assembled matrix.
    """

    def stiffness(start, end, diameter):
        return member_stiffness(start, end, diameter, problem.catalogue)

    shaped = (state.starts, state.ends, state.diameters)

    return (
        np.asarray(jax.vmap(jax.jacfwd(stiffness, 0))(*shaped)),
        np.asarray(jax.vmap(jax.jacfwd(stiffness, 1))(*shaped)),
        np.asarray(jax.vmap(jax.jacfwd(stiffness, 2))(*shaped)),
    )


def force_cotangents(
    problem: FrameProblem,
    xyz: Float[np.ndarray, "nodes 3"],
    diameters: Float[np.ndarray, "members"],
    cotangent: ReadingCotangent,
) -> Cotangents:
    """
    Pull a cotangent on the reported forces back onto the inputs.

    Parameters
    ----------
    problem :
        The frame, its section family and its loading.
    xyz :
        Position of every node, in this repository's axes and millimeters.
    diameters :
        Outer diameter of every member.
    cotangent :
        Cotangent on each reported quantity.

    Returns
    -------
    pulled :
        Cotangent on every node coordinate and every diameter.

    Notes
    -----
    **One solve, whatever the parameter count.** This is the reverse rule
    proper, not a slice of a Jacobian: the cotangent is pulled through each
    member's reading, gathered into a single adjoint load, solved once against
    the already-decomposed stiffness, and contracted against element-local
    stiffness derivatives. `force_jacobian` costs a back-substitution per
    parameter and is the right shape when every block is wanted; a descent wants
    this one, and the difference grows with the structure.

    Nodal loads do not move with the geometry, so no load derivative enters and
    the adjoint load is the reading's alone.
    """
    state = adjoint_state(problem, xyz, diameters)
    edges = np.asarray(problem.structure.edges)
    nodes = int(np.asarray(xyz).shape[0])

    seed = jnp.stack(
        [
            jnp.asarray(cotangent.axial_force),
            jnp.asarray(cotangent.moment_major)[:, 0],
            jnp.asarray(cotangent.moment_major)[:, 1],
            jnp.asarray(cotangent.moment_minor)[:, 0],
            jnp.asarray(cotangent.moment_minor)[:, 1],
        ],
        axis=1,
    )

    def pulled(start, end, diameter, displacement, sown):
        def reading(*args):
            return member_reading(*args, problem.catalogue)

        _, backward = jax.vjp(reading, start, end, diameter, displacement)

        return backward(sown)

    over = (state.starts, state.ends, state.diameters, state.moved, seed)
    by_start, by_end, by_diameter, by_displacement = jax.vmap(pulled)(*over)

    # One adjoint load, gathered from every member that reads a displacement.
    acting = np.zeros(state.displaced.size)
    np.add.at(acting, state.indexed, np.asarray(by_displacement))

    adjoint = np.zeros(state.displaced.size)
    adjoint[state.free] = state.factorized.solve(acting[state.free])

    slope_start, slope_end, slope_diameter = _element_slopes(problem, state)
    borne = adjoint[state.indexed]
    local = state.displaced[state.indexed]

    implicit_start = -np.einsum("mi,mija,mj->ma", borne, slope_start, local)
    implicit_end = -np.einsum("mi,mija,mj->ma", borne, slope_end, local)
    implicit_diameter = -np.einsum("mi,mij,mj->m", borne, slope_diameter, local)

    by_node = np.zeros((nodes, 3))
    np.add.at(by_node, edges[:, 0], np.asarray(by_start) + implicit_start)
    np.add.at(by_node, edges[:, 1], np.asarray(by_end) + implicit_end)

    return Cotangents(
        xyz=by_node,
        diameter=np.asarray(by_diameter) + implicit_diameter,
    )


def force_jacobian(
    problem: FrameProblem,
    xyz: Float[np.ndarray, "nodes 3"],
    diameters: Float[np.ndarray, "members"],
) -> Jacobian:
    """
    How every reported force moves with the geometry and with the diameters.

    Parameters
    ----------
    problem :
        The frame, its section family and its loading.
    xyz :
        Position of every node, in this repository's axes and millimeters.
    diameters :
        Outer diameter of every member.

    Returns
    -------
    jacobian :
        Dense blocks, one per reported quantity and input.

    Notes
    -----
    **An exact rule, not a sweep.** Equilibrium is an implicit function of the
    inputs, so differentiating it gives the derivative of the displacement as a
    second solve against the stiffness already factorized — each parameter costs
    a back-substitution rather than a fresh analysis. The solver contributes the
    assembly, the factorization and the solve; the element derivative and this
    rule are this repository's, which is the whole of what it means to give a
    foreign solver an adjoint.

    Every block, whatever was asked for, because they share one factorization.
    That makes this the right shape for a materialized Jacobian and the wrong
    shape for a single gradient: `force_cotangents` answers the latter in one
    solve instead of one per parameter.

    Supports are prescribed at zero displacement whatever the geometry does, so
    their rows contribute nothing and the solve is over the free set alone.
    """
    state = adjoint_state(problem, xyz, diameters)
    edges = np.asarray(problem.structure.edges)
    members = int(edges.shape[0])
    nodes = int(np.asarray(xyz).shape[0])

    def reading(start, end, diameter, displacement):
        return member_reading(start, end, diameter, displacement, problem.catalogue)

    over = (state.starts, state.ends, state.diameters, state.moved)
    by_start = np.asarray(jax.vmap(jax.jacrev(reading, 0))(*over))
    by_end = np.asarray(jax.vmap(jax.jacrev(reading, 1))(*over))
    by_diameter = np.asarray(jax.vmap(jax.jacrev(reading, 2))(*over))
    by_displacement = np.asarray(jax.vmap(jax.jacrev(reading, 3))(*over))

    slope_start, slope_end, slope_diameter = _element_slopes(problem, state)

    width = nodes * 3 + members
    reach = by_start.shape[1]
    forcing = np.zeros((state.displaced.size, width))
    explicit = np.zeros((members, reach, width))

    for member in range(members):
        dofs = state.indexed[member]
        local = state.displaced[dofs]
        first = edges[member, 0] * 3 + np.arange(3)
        second = edges[member, 1] * 3 + np.arange(3)
        section = nodes * 3 + member

        explicit[member][:, first] += by_start[member]
        explicit[member][:, second] += by_end[member]
        explicit[member][:, section] += by_diameter[member]

        forcing[np.ix_(dofs, first)] -= np.einsum(
            "ija,j->ia", slope_start[member], local
        )
        forcing[np.ix_(dofs, second)] -= np.einsum(
            "ija,j->ia", slope_end[member], local
        )
        forcing[np.ix_(dofs, [section])] -= (slope_diameter[member] @ local)[:, None]

    moving = np.zeros((state.displaced.size, width))
    moving[state.free] = state.factorized.solve(forcing[state.free])

    total = explicit.copy()
    for member in range(members):
        total[member] += by_displacement[member] @ moving[state.indexed[member]]

    return Jacobian(
        axial_force_xyz=total[:, READING_AXIAL, : nodes * 3].reshape(members, nodes, 3),
        axial_force_diameter=total[:, READING_AXIAL, nodes * 3 :],
        moment_major_xyz=total[:, READING_MAJOR, : nodes * 3].reshape(
            members, 2, nodes, 3
        ),
        moment_major_diameter=total[:, READING_MAJOR, nodes * 3 :],
        moment_minor_xyz=total[:, READING_MINOR, : nodes * 3].reshape(
            members, 2, nodes, 3
        ),
        moment_minor_diameter=total[:, READING_MINOR, nodes * 3 :],
    )
