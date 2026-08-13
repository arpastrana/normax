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
The OpenSees backend of the analysis stage, differentiated by DDM.

A C++ finite element solver reached through `openseespy`, whose adjoints were
hand-derived element by element and which nothing about JAX can see into. It
computes the same contract `normax.analysis.smax` does and arrives at its
derivatives the opposite way: forward, one parameter at a time, from rules
compiled into the library rather than traced from the source.

**Two dimensions only, and that is a property of OpenSees rather than a
simplification here.** Its Direct Differentiation Method reaches a nodal
coordinate in 2D and agrees with central differences to 7.4e-9; in 3D beams
return identically zero and trusses return values that are wrong rather than
absent, because `LinearCrdTransf3d` implements none of the shape-sensitivity
family its 2D counterpart does. Anything three-dimensional goes through the
other backend. See `CHANGELOG.md` under `## OpenSees DDM spike`.

**Elements are `forceBeamColumn` over `section('Elastic')`, and the choice is
not free.** `elasticBeamColumn` accepts every parameter, returns a tag, reads
the value back, and then yields identically zero sensitivities with no warning.
`dispBeamColumn` gets displacement sensitivities right and section-force
sensitivities wrong by up to 12x, and section forces are what the check
downstream consumes.

**One block of the Jacobian is unreachable, and it is exactly one.** A planar
frame's response separates: the axial force and the in-plane moment do not move
when a node leaves the plane, and the out-of-plane moment does not move when a
node travels within it. A model built in the plane therefore carries every
derivative except `∂M_z,Ed/∂xyz` along the normal axis, which it reports as
zero. Form finding cannot move a planar arch out of its plane either, so the
block is annihilated by the composition rather than merely small.
"""

from typing import Any
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import openseespy.opensees as ops
from jaxtyping import Array
from jaxtyping import Float

from normax.analysis import support_fixities
from normax.ec3.material import Steel
from normax.ec3.section import TubeCatalogue
from normax.stages import MemberForces
from normax.structures import Structure
from normax.units import MILLIMETER
from normax.units import to_meters
from normax.units import to_newton_millimeters
from normax.units import to_pascals

# Integration points along a force-based element. The first and the last sit on
# the end sections, which is where the moments the check consumes are read.
NUM_INTEGRATION_POINTS = 5

# Degrees of freedom of a node of a plane frame: two translations and a
# rotation.
DOF_PER_NODE_PLANAR = 3

# Component of a section's force vector, as `sensSectionForce` orders it.
DOF_AXIAL = 1
DOF_MOMENT = 2


class Model(NamedTuple):
    """
    Everything this backend can settle before a geometry or a size is chosen.

    Attributes
    ----------
    structure :
        The structure supplying the connectivity, the supports and the loads.
    spanned :
        The plane the frame is modeled in.

    Notes
    -----
    **Almost nothing, and that is the honest answer for this backend.** OpenSees
    holds one global model with no handle to it, so a domain cannot be built once
    and updated; every call wipes and reassembles. What can be settled ahead of
    time is which two global axes the frame lives in, and that is what this
    carries.

    The contract is shared with `normax.analysis.smax` all the same. A stage that
    prepares once and solves many times fits a solver that reuses an assembly and
    a solver that cannot, and the difference between them showing up as an empty
    model rather than as a different call is the point.
    """

    structure: Structure
    spanned: "Plane"


class Plane(NamedTuple):
    """
    The two global axes a planar frame is modeled in.

    Attributes
    ----------
    axes :
        Indices of the two global axes the plane spans, in increasing order.
    normal :
        Index of the global axis the frame has no thickness along.

    Notes
    -----
    Increasing order rather than a right-handed pair, so the map into the
    solver's own axes is a slice and needs no sign. A frame in the XZ plane
    therefore reaches OpenSees with its Z along the solver's Y, which is where
    gravity already points.
    """

    axes: tuple[int, int]
    normal: int


class Jacobian(NamedTuple):
    """
    Every derivative the solver reports, as dense blocks.

    Attributes
    ----------
    axial_force_xyz :
        Derivative of each member's axial force in every nodal coordinate.
    axial_force_diameter :
        Derivative of each member's axial force in every member's diameter.
    moment_major_xyz :
        Derivative of each end moment in every nodal coordinate.
    moment_major_diameter :
        Derivative of each end moment in every member's diameter.

    Notes
    -----
    Dense in the members: a section is a property of one element, but changing
    it redistributes force through the whole frame, so no block is diagonal.

    The minor-axis moment has no blocks. It is identically zero in a plane
    frame and so is every derivative of it the solver can reach; the one
    derivative it cannot reach is named in this module's own docstring.
    """

    axial_force_xyz: Float[Array, "members nodes 3"]
    axial_force_diameter: Float[Array, "members members"]
    moment_major_xyz: Float[Array, "members ends nodes 3"]
    moment_major_diameter: Float[Array, "members ends members"]


def frame_plane(
    structure: Structure,
    xyz: Float[Array, "nodes 3"],
    normal: int | None,
) -> Plane:
    """
    The plane a frame lies in, checked rather than assumed.

    Parameters
    ----------
    structure :
        The structure supplying the loads.
    xyz :
        Position of every node.
    normal :
        Index of the global axis the frame has no thickness along.

    Returns
    -------
    plane :
        The two axes spanned and the one that is not.

    Raises
    ------
    ValueError
        If no normal axis is given, if it is not 0, 1 or 2, or if the nodes do
        not share one coordinate along it.

    Notes
    -----
    Every rejection here is a case the solver would otherwise accept and answer
    wrongly: a three-dimensional frame flattened into a projection of itself. A
    backend that cannot represent something should say so rather than represent
    something else.

    Loads are checked where they are applied rather than here, since a structure
    is analyzed under load cases other than its own and only the one reaching the
    solver can be the one vouched for.
    """
    if normal is None:
        raise ValueError("the OpenSees backend is planar; give the normal axis")
    if normal not in (0, 1, 2):
        raise ValueError(f"normal must be 0, 1 or 2, got {normal}")

    offsets = np.asarray(xyz)[:, normal]
    if not np.allclose(offsets, offsets[0]):
        spread = float(np.ptp(offsets))
        raise ValueError(f"nodes are not planar along axis {normal}; spread {spread}")

    axes = tuple(axis for axis in range(3) if axis != normal)

    return Plane(axes=axes, normal=normal)


def prepare_model(
    structure: Structure,
    steel: Steel,
    catalogue: TubeCatalogue,
    *,
    normal: int | None,
) -> Model:
    """
    Settle the plane the frame is modeled in, and nothing else.

    Parameters
    ----------
    structure :
        The structure supplying the connectivity, the supports and the loads.
    steel :
        Material properties. Unused, the domain being rebuilt per call.
    catalogue :
        The section family. Unused, for the same reason.
    normal :
        Index of the global axis the frame has no thickness along.

    Returns
    -------
    model :
        The structure and the plane it spans.

    Raises
    ------
    ValueError
        If no normal axis is given, if it is not 0, 1 or 2, or if the starting
        geometry does not share one coordinate along it.

    Notes
    -----
    The plane is read from the starting geometry rather than from a form-found
    one, which makes planarity a property of the structure and fixes the axis map
    before any force density is chosen. The geometry actually analyzed is checked
    again per call, so a shape that leaves the plane is still refused.

    The material and the section family are accepted and ignored, so that this
    reads the same as the other backend's `prepare`. Nothing about a plane frame
    can be precomputed from either.
    """
    return Model(
        structure=structure, spanned=frame_plane(structure, structure.nodes, normal)
    )


def _build_model(
    model: Model,
    xyz: Float[Array, "nodes 3"],
    diameters: Float[Array, "members"],
    steel: Steel,
    catalogue: TubeCatalogue,
    *,
    loads: Float[Array, "nodes 3"],
    parameters: bool,
) -> int:
    """
    Assemble and solve the frame, optionally registering every DDM parameter.

    Parameters
    ----------
    model :
        The structure and its plane, from `prepare`.
    xyz :
        Position of every node.
    diameters :
        Outer diameter of every member.
    steel :
        Material properties. Only the modulus reaches a plane frame.
    catalogue :
        The section family, whose ratio fixes the wall thickness.
    loads :
        Force applied at every node.
    parameters :
        Whether to register the nodal coordinates and section properties.

    Returns
    -------
    count :
        Number of parameters registered.

    Raises
    ------
    ValueError
        If the geometry does not lie in the model's plane, or if the load case
        applied has a component along the normal axis.

    Notes
    -----
    Every element carries its own section, so a parameter registered on one
    element perturbs that element alone. Sharing a section across members would
    make a correct sensitivity read as wrong.

    The whole model is rebuilt per call. OpenSees holds one global model and no
    handle to it, so there is nothing to update in place, and the solve is a few
    milliseconds at the sizes this backend is used at.

    **The geometry and the load case are vouched for here rather than upstream.**
    Both change from call to call while the plane does not, so this is the only
    place that sees the numbers a solve is actually given.
    """
    structure = model.structure
    spanned = model.spanned

    frame_plane(structure, xyz, spanned.normal)

    out_of_plane = np.asarray(loads)[:, spanned.normal]
    if np.any(out_of_plane != 0.0):
        raise ValueError(
            f"loads have components along the normal axis {spanned.normal}"
        )

    coordinates = np.asarray(to_meters(xyz))[:, list(spanned.axes)]
    edges = np.asarray(structure.edges)
    applied = np.asarray(loads)[:, list(spanned.axes)]
    flags = support_fixities(structure, spanned.normal)

    outer = to_meters(diameters)
    areas = np.asarray(catalogue.tube_at(outer).area)
    inertias = np.asarray(catalogue.tube_at(outer).second_moment)
    e_mod = float(to_pascals(steel.e_mod))

    num_nodes = coordinates.shape[0]
    num_members = edges.shape[0]

    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", DOF_PER_NODE_PLANAR)

    for node in range(num_nodes):
        ops.node(node + 1, float(coordinates[node, 0]), float(coordinates[node, 1]))

    for node in range(num_nodes):
        restrained = [int(flags[node, axis]) for axis in spanned.axes]
        if any(restrained):
            ops.fix(node + 1, restrained[0], restrained[1], 0)

    ops.geomTransf("Linear", 1)
    for member in range(num_members):
        tag = member + 1
        ops.section(
            "Elastic", tag, e_mod, float(areas[member]), float(inertias[member])
        )
        ops.beamIntegration("Lobatto", tag, tag, NUM_INTEGRATION_POINTS)
        ops.element(
            "forceBeamColumn",
            tag,
            int(edges[member, 0]) + 1,
            int(edges[member, 1]) + 1,
            1,
            tag,
        )

    ops.timeSeries("Constant", 1)
    ops.pattern("Plain", 1, 1)
    for node in range(num_nodes):
        if np.any(applied[node] != 0.0):
            ops.load(node + 1, float(applied[node, 0]), float(applied[node, 1]), 0.0)

    count = 0
    if parameters:
        for spec in _parameter_specifications(num_nodes, num_members):
            count += 1
            ops.parameter(count, *spec)

    ops.system("FullGeneral")
    ops.numberer("Plain")
    ops.constraints("Plain")
    ops.integrator("LoadControl", 1.0)
    ops.algorithm("Linear")
    ops.analysis("Static")
    if count:
        ops.sensitivityAlgorithm("-computeAtEachStep")
    ops.analyze(1)

    return count


def _parameter_specifications(
    num_nodes: int, num_members: int
) -> list[tuple[Any, ...]]:
    """
    The DDM parameter of every differentiable quantity, in tag order.

    Parameters
    ----------
    num_nodes :
        Number of nodes in the frame.
    num_members :
        Number of members in the frame.

    Returns
    -------
    specifications :
        Argument tuples for `parameter`, one per registered quantity.

    Notes
    -----
    Coordinates first and sections after, so a tag can be recovered from an
    index without a lookup. The second moment is named `I` in a two-dimensional
    model and `Iz` in a three-dimensional one; the wrong name binds to nothing
    and is indistinguishable from a missing derivative.

    Diameters are not registered. A section is what the solver understands, so
    the area and the second moment are the parameters and the chain rule to a
    diameter is taken here rather than asked of OpenSees.
    """
    coordinates = [
        ("node", node + 1, "coord", direction + 1)
        for node in range(num_nodes)
        for direction in range(2)
    ]
    sections = [
        ("element", member + 1, name)
        for member in range(num_members)
        for name in ("A", "I")
    ]

    return coordinates + sections


def _read_forces(num_members: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Section forces at both ends of every member.

    Parameters
    ----------
    num_members :
        Number of members in the frame.

    Returns
    -------
    forces :
        Axial force of every member, and its moment at each end.

    Notes
    -----
    Read at the first and last integration points, which a Lobatto rule places
    exactly on the end sections. The axial force is taken once because nodal
    loading leaves it constant along the span.
    """
    axial = np.empty(num_members)
    moments = np.empty((num_members, 2))

    for member in range(num_members):
        first = ops.eleResponse(member + 1, "section", 1, "force")
        last = ops.eleResponse(member + 1, "section", NUM_INTEGRATION_POINTS, "force")
        axial[member] = first[0]
        moments[member, 0] = first[1]
        moments[member, 1] = last[1]

    return axial, moments


def member_forces(
    model: Model,
    xyz: Float[Array, "nodes 3"],
    diameters: Float[Array, "members"],
    steel: Steel,
    catalogue: TubeCatalogue,
    loads: Float[Array, "nodes 3"],
) -> MemberForces:
    """
    Internal forces of a plane frame under a load case.

    Parameters
    ----------
    model :
        The structure and its plane, from `prepare`.
    xyz :
        Position of every node, from form finding.
    diameters :
        Outer diameter of every member.
    steel :
        Material properties.
    catalogue :
        The section family, whose ratio fixes the wall thickness.
    loads :
        Force applied at every node.

    Returns
    -------
    forces :
        Axial force and both end moments of every member.

    Notes
    -----
    **Not differentiable by tracing.** The solve happens in C++ behind a command
    interface, so a JAX tracer reaching this function has nothing to record. Ask
    `jacobian` for derivatives instead, which is what the stage's endpoints do.

    The minor-axis moment is returned as exact zeros. A plane frame under
    in-plane load carries none, so this is the value rather than a placeholder.
    """
    _build_model(model, xyz, diameters, steel, catalogue, loads=loads, parameters=False)

    axial, moments = _read_forces(model.structure.num_edges)

    return MemberForces(
        axial_force=jnp.asarray(axial),
        moment_major=jnp.asarray(to_newton_millimeters(moments)),
        moment_minor=jnp.zeros_like(jnp.asarray(moments)),
    )


def _section_slopes(
    diameters: Float[Array, "members"],
    catalogue: TubeCatalogue,
) -> tuple[np.ndarray, np.ndarray]:
    """
    How a member's area and second moment move with its diameter.

    Parameters
    ----------
    diameters :
        Outer diameter of every member.
    catalogue :
        The section family, whose ratio fixes the wall thickness.

    Returns
    -------
    slopes :
        Derivative of the area and of the second moment, in SI per millimeter.

    Notes
    -----
    Taken with `jax.grad` of the closed forms the check itself uses, so the two
    stages cannot drift apart in what a section is. The result is per millimeter
    of diameter because the schema states diameters in millimeters while the
    model is assembled in meters.
    """
    outer = to_meters(diameters)

    d_area = jax.vmap(jax.grad(lambda d: catalogue.tube_at(d).area))(outer)
    d_inertia = jax.vmap(jax.grad(lambda d: catalogue.tube_at(d).second_moment))(outer)

    return (
        np.asarray(d_area) * MILLIMETER,
        np.asarray(d_inertia) * MILLIMETER,
    )


def _force_sensitivity(element: int, section: int, dof: int, tag: int) -> float:
    """
    One entry of a section's force sensitivity to one parameter.

    Parameters
    ----------
    element :
        Tag of the element to read.
    section :
        Index of the integration point along it, counted from one.
    dof :
        Component of the section force vector, counted from one.
    tag :
        Tag of the registered parameter.

    Returns
    -------
    sensitivity :
        Derivative of that component in that parameter.

    Notes
    -----
    **The return starts at the requested component rather than being it**, so
    the value asked for is the first entry and indexing by the component number
    quietly returns a neighbour.

    Passing a tag that was never registered does not raise: it terminates the
    process with exit 139 and no traceback. Every tag reaching here comes from
    the count `_build` returns for that reason.
    """
    reading = ops.sensSectionForce(element, section, dof, tag)

    return float(reading[0]) if isinstance(reading, list) else float(reading)


def force_jacobian(
    model: Model,
    xyz: Float[Array, "nodes 3"],
    diameters: Float[Array, "members"],
    steel: Steel,
    catalogue: TubeCatalogue,
    loads: Float[Array, "nodes 3"],
) -> Jacobian:
    """
    Every derivative of the member forces, from one solve and a sweep.

    Parameters
    ----------
    model :
        The structure and its plane, from `prepare`.
    xyz :
        Position of every node, from form finding.
    diameters :
        Outer diameter of every member.
    steel :
        Material properties.
    catalogue :
        The section family, whose ratio fixes the wall thickness.
    loads :
        Force applied at every node.

    Returns
    -------
    jacobian :
        Dense derivative blocks of the axial force and the end moments.

    Notes
    -----
    Forward mode, which is what the Direct Differentiation Method is. One
    factorization is formed and reused, so each additional parameter costs a
    back-substitution rather than a solve, and the whole Jacobian arrives in a
    single sweep. Both the tangent and the cotangent rules are contractions of
    it, which is why the expensive direction is the parameter count and not the
    number of outputs.

    The column along the normal axis is left at zero. It is the one derivative a
    model built in the plane cannot reach, and it belongs to the minor-axis
    moment alone, which this Jacobian does not carry.
    """
    count = _build_model(
        model, xyz, diameters, steel, catalogue, loads=loads, parameters=True
    )

    num_nodes = np.asarray(xyz).shape[0]
    num_members = model.structure.num_edges

    axial = np.zeros((num_members, count))
    moments = np.zeros((num_members, 2, count))

    for tag in range(1, count + 1):
        for member in range(num_members):
            element = member + 1
            axial[member, tag - 1] = _force_sensitivity(element, 1, DOF_AXIAL, tag)
            moments[member, 0, tag - 1] = _force_sensitivity(
                element, 1, DOF_MOMENT, tag
            )
            moments[member, 1, tag - 1] = _force_sensitivity(
                element, NUM_INTEGRATION_POINTS, DOF_MOMENT, tag
            )

    return _assemble_blocks(
        ParameterSweep(axial, moments, num_nodes), diameters, catalogue, model.spanned
    )


class ParameterSweep(NamedTuple):
    """
    A direct-differentiation sweep, and the layout of its parameter columns.

    Attributes
    ----------
    axial :
        Derivative of every member's axial force in every parameter.
    moments :
        Derivative of every end moment in every parameter.
    num_nodes :
        Number of nodes in the frame.

    Notes
    -----
    The node count belongs here rather than beside it, being what says where the
    coordinate columns end and the section columns begin. The member count is
    not carried: it is the leading axis of `axial`, and a second copy of a number
    already present is a chance for the two to disagree.
    """

    axial: np.ndarray
    moments: np.ndarray
    num_nodes: int


def _assemble_blocks(
    sweep: ParameterSweep,
    diameters: Float[Array, "members"],
    catalogue: TubeCatalogue,
    spanned: Plane,
) -> Jacobian:
    """
    Rearrange a parameter sweep into the blocks the stage differentiates in.

    Parameters
    ----------
    sweep :
        The raw derivatives, and the node count laying out their columns.
    diameters :
        Outer diameter of every member.
    catalogue :
        The section family, whose ratio fixes the wall thickness.
    spanned :
        The plane the frame is modeled in.

    Returns
    -------
    jacobian :
        The same numbers, indexed by coordinate and by diameter.

    Notes
    -----
    Two changes of variable happen here and nowhere else. The solver's two
    in-plane directions are scattered back into three global ones, leaving the
    normal column zero; and its area and second moment are contracted with the
    section slopes into one derivative per diameter, which is the variable the
    schema actually carries.

    Millimeters re-enter on both. Coordinates are differentiated per meter and
    moments returned in newton meters, so each block is scaled once rather than
    at every use.
    """
    axial = sweep.axial
    moments = sweep.moments
    num_nodes = sweep.num_nodes
    num_members = axial.shape[0]

    coordinates = 2 * num_nodes
    d_area, d_inertia = _section_slopes(diameters, catalogue)

    axial_force_xyz = np.zeros((num_members, num_nodes, 3))
    moment_major_xyz = np.zeros((num_members, 2, num_nodes, 3))

    for index, axis in enumerate(spanned.axes):
        columns = slice(index, coordinates, 2)
        axial_force_xyz[:, :, axis] = axial[:, columns] * MILLIMETER
        moment_major_xyz[:, :, :, axis] = (
            to_newton_millimeters(moments[:, :, columns]) * MILLIMETER
        )

    sections = axial[:, coordinates:].reshape(num_members, num_members, 2)
    axial_force_diameter = sections[:, :, 0] * d_area + sections[:, :, 1] * d_inertia

    sections = moments[:, :, coordinates:].reshape(num_members, 2, num_members, 2)
    moment_major_diameter = to_newton_millimeters(
        sections[:, :, :, 0] * d_area + sections[:, :, :, 1] * d_inertia
    )

    return Jacobian(
        axial_force_xyz=jnp.asarray(axial_force_xyz),
        axial_force_diameter=jnp.asarray(axial_force_diameter),
        moment_major_xyz=jnp.asarray(moment_major_xyz),
        moment_major_diameter=jnp.asarray(moment_major_diameter),
    )
