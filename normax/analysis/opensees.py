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
The OpenSees frame analysis, differentiated by the Direct Differentiation Method.

A C++ solver reached through `openseespy`, whose sensitivities are compiled into
the library and which nothing about JAX can see into. It arrives at the same
contract the traced oracle does the opposite way: forward, one parameter at a
time, from one factorization.

**Two dimensions only, and that is a property of OpenSees.** Its DDM reaches a
nodal coordinate in 2D; in 3D beams return zero and trusses return wrong values,
because `LinearCrdTransf3d` implements no shape sensitivity. The evidence is in
`CHANGELOG.md` under `## OpenSees DDM spike`.

**Elements are `forceBeamColumn` over `section('Elastic')`, never
`elasticBeamColumn`**, which accepts every parameter and yields identically zero
sensitivities with no warning; `dispBeamColumn` gets section forces wrong by up
to 12x. One block is unreachable and it is exactly one: the minor-axis moment's
derivative along the normal axis, which form finding cannot excite either.
"""

from typing import Any
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import openseespy.opensees as ops
from jaxtyping import Array
from jaxtyping import Float

from normax.analysis import MemberForces
from normax.analysis import normal_axis
from normax.analysis import support_fixities
from normax.sections import TubeFamily
from normax.structures import Structure
from normax.units import MEGAPASCAL
from normax.units import MILLIMETER
from normax.units import NEWTON_MILLIMETER

# Integration points along a force-based element; the first and the last sit on
# the end sections, where the moments are read.
NUM_INTEGRATION_POINTS = 5

# Degrees of freedom of a node of a plane frame.
DOF_PER_NODE_PLANAR = 3

# Component of a section's force vector, as `sensSectionForce` orders it.
DOF_AXIAL = 1
DOF_MOMENT = 2


class Plane(NamedTuple):
    """
    The two global axes a planar frame is modeled in.

    Attributes
    ----------
    axes :
        Indices of the two global axes the plane spans, in increasing order, so
        the map into the solver's axes is a slice and needs no sign.
    normal :
        Index of the global axis the frame has no thickness along.
    """

    axes: tuple[int, int]
    normal: int


class Model(NamedTuple):
    """
    Everything this backend can settle before a geometry or a size is chosen.

    Attributes
    ----------
    structure :
        The structure supplying the connectivity and the supports.
    spanned :
        The plane the frame is modeled in.

    Notes
    -----
    Almost nothing: OpenSees holds one global model with no handle to it, so
    every call wipes and reassembles. The contract is shared with the other
    backends all the same.
    """

    structure: Structure
    spanned: Plane


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
    The minor-axis moment has no blocks: it is identically zero in a plane frame
    and so is every derivative of it the solver can reach.
    """

    axial_force_xyz: Float[np.ndarray, "members nodes 3"]
    axial_force_diameter: Float[np.ndarray, "members members"]
    moment_major_xyz: Float[np.ndarray, "members ends nodes 3"]
    moment_major_diameter: Float[np.ndarray, "members ends members"]


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
        Number of nodes, which says where the coordinate columns end.
    """

    axial: Float[np.ndarray, "members parameters"]
    moments: Float[np.ndarray, "members ends parameters"]
    num_nodes: int


class PlaneForces(NamedTuple):
    """
    What one plane-frame solve reports about every member.

    Attributes
    ----------
    axial :
        Axial force of every member.
    moments :
        In-plane moment at each end of every member, in the solver's units.
    """

    axial: Float[np.ndarray, "members"]
    moments: Float[np.ndarray, "members ends"]


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
        The structure whose restrained axis the plane must agree with.
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
        If no normal axis is given, if the nodes do not share one coordinate
        along it, or if it is not the axis a three-dimensional solve of the
        same structure would restrain.

    Notes
    -----
    Every rejection is a case the solver would otherwise answer wrongly, as a
    projection of a frame it cannot represent.
    """
    if normal is None:
        raise ValueError("the OpenSees backend is planar; give the normal axis")
    if normal not in (0, 1, 2):
        raise ValueError(f"normal must be 0, 1 or 2, got {normal}")

    offsets = np.asarray(xyz)[:, normal]
    if not np.allclose(offsets, offsets[0]):
        spread = float(np.ptp(offsets))
        raise ValueError(f"nodes are not planar along axis {normal}; spread {spread}")

    measured = normal_axis(structure)
    if measured != normal:
        raise ValueError(f"the structure is restrained along axis {measured}")

    axes = tuple(axis for axis in range(3) if axis != normal)

    return Plane(axes=axes, normal=normal)


def prepare_model(
    structure: Structure,
    family: TubeFamily,
    normal: int | None,
) -> Model:
    """
    Settle the plane the frame is modeled in, and nothing else.

    Parameters
    ----------
    structure :
        The structure supplying the connectivity and the supports.
    family :
        The section family. Unused, the domain being rebuilt per call.
    normal :
        Index of the global axis the frame has no thickness along.

    Returns
    -------
    model :
        The structure and the plane it spans.

    Raises
    ------
    ValueError
        If the starting geometry does not lie in the plane named.
    """
    spanned = frame_plane(structure, structure.nodes, normal)

    return Model(structure=structure, spanned=spanned)


def _refuse_unusable(
    coordinates: Float[np.ndarray, "nodes 2"],
    areas: Float[np.ndarray, "members"],
    inertias: Float[np.ndarray, "members"],
) -> None:
    """
    Refuse what `section("Elastic")` accepts silently and fails on later.

    Raises
    ------
    ValueError
        If any value is not finite, or any section property is not positive.
    """
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("node coordinates are not all finite")

    for label, values in (("areas", areas), ("second moments", inertias)):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"section {label} are not all finite")
        if np.any(values <= 0.0):
            smallest = float(np.min(values))
            raise ValueError(f"section {label} must be positive, smallest {smallest}")


def _refuse_unregistered(count: int) -> None:
    """
    Refuse a sweep over tags the domain does not hold, which would segfault.

    Raises
    ------
    RuntimeError
        If the domain holds anything other than exactly those tags.
    """
    registered = sorted(int(tag) for tag in ops.getParamTags())
    if registered != list(range(1, count + 1)):
        raise RuntimeError(
            f"registered {count} parameters but the domain holds {len(registered)}"
        )


def _parameter_specifications(
    num_nodes: int,
    num_members: int,
) -> list[tuple[Any, ...]]:
    """
    The DDM parameter of every differentiable quantity, in tag order.

    Notes
    -----
    Coordinates first and sections after. The second moment is `I` in 2D and
    `Iz` in 3D; the wrong name binds to nothing and looks like a zero. Diameters
    are not registered: the chain rule to one is taken here.
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


class FrameSolve(NamedTuple):
    """
    What one assembly of the domain is given beyond the model.

    Attributes
    ----------
    xyz :
        Position of every node.
    diameters :
        Outer diameter of every member.
    loads :
        Force applied at every node.
    parameters :
        Whether to register the coordinates and section properties for DDM.
    """

    xyz: Float[Array, "nodes 3"]
    diameters: Float[Array, "members"]
    loads: Float[Array, "nodes 3"]
    parameters: bool


def _build_model(model: Model, family: TubeFamily, given: FrameSolve) -> int:
    """
    Assemble and solve the frame, optionally registering every DDM parameter.

    Parameters
    ----------
    model :
        The structure and its plane, from `prepare_model`.
    family :
        The section family, whose ratio fixes the wall and whose grade supplies
        the modulus.
    given :
        The geometry, the sizes, the load case, and whether to register.

    Returns
    -------
    count :
        Number of parameters registered.

    Raises
    ------
    ValueError
        If the geometry leaves the plane, or the load case has a component
        along the normal axis.
    RuntimeError
        If the static solve fails.

    Notes
    -----
    Every element carries its own section, so a parameter registered on one
    element perturbs that element alone.
    """
    structure = model.structure
    spanned = model.spanned

    frame_plane(structure, given.xyz, spanned.normal)

    out_of_plane = np.asarray(given.loads)[:, spanned.normal]
    if np.any(out_of_plane != 0.0):
        raise ValueError(
            f"loads have components along the normal axis {spanned.normal}"
        )

    coordinates = np.asarray(given.xyz)[:, list(spanned.axes)] * MILLIMETER
    edges = np.asarray(structure.edges)
    applied = np.asarray(given.loads)[:, list(spanned.axes)]
    flags = support_fixities(structure)

    sections = family(jnp.asarray(given.diameters) * MILLIMETER)
    areas = np.asarray(sections.area)
    inertias = np.asarray(sections.second_moment)
    e_mod = float(family.material.e_mod) * MEGAPASCAL

    num_nodes = coordinates.shape[0]
    num_members = edges.shape[0]

    _refuse_unusable(coordinates, areas, inertias)

    # The analysis owns the sensitivity algorithm, so the domain is wiped whole.
    ops.wipeAnalysis()
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
        first = int(edges[member, 0]) + 1
        second = int(edges[member, 1]) + 1
        ops.element("forceBeamColumn", tag, first, second, 1, tag)

    ops.timeSeries("Constant", 1)
    ops.pattern("Plain", 1, 1)
    for node in range(num_nodes):
        if np.any(applied[node] != 0.0):
            ops.load(node + 1, float(applied[node, 0]), float(applied[node, 1]), 0.0)

    count = 0
    if given.parameters:
        for spec in _parameter_specifications(num_nodes, num_members):
            count += 1
            ops.parameter(count, *spec)
        _refuse_unregistered(count)

    ops.system("FullGeneral")
    ops.numberer("Plain")
    ops.constraints("Plain")
    ops.integrator("LoadControl", 1.0)
    ops.algorithm("Linear")
    ops.analysis("Static")
    if count:
        ops.sensitivityAlgorithm("-computeAtEachStep")

    status = ops.analyze(1)
    if status != 0:
        raise RuntimeError(f"the static solve failed, ops.analyze returned {status}")

    return count


def _read_forces(num_members: int) -> PlaneForces:
    """
    Section forces at both ends of every member, from the solved domain.

    Notes
    -----
    Moments are read at the first and last integration points, which a Lobatto
    rule places on the end sections. The axial force is constant under nodal
    loading, so it is read once.
    """
    axial = np.empty(num_members)
    moments = np.empty((num_members, 2))

    for member in range(num_members):
        first = ops.eleResponse(member + 1, "section", 1, "force")
        last = ops.eleResponse(member + 1, "section", NUM_INTEGRATION_POINTS, "force")
        axial[member] = first[0]
        moments[member, 0] = first[1]
        moments[member, 1] = last[1]

    return PlaneForces(axial, moments)


def member_forces(
    model: Model,
    xyz: Float[Array, "nodes 3"],
    diameters: Float[Array, "members"],
    family: TubeFamily,
    loads: Float[Array, "nodes 3"],
) -> MemberForces:
    """
    Internal forces of a plane frame under a load case.

    Parameters
    ----------
    model :
        The structure and its plane, from `prepare_model`.
    xyz :
        Position of every node, from form finding.
    diameters :
        Outer diameter of every member.
    family :
        The section family, whose ratio fixes the wall and whose grade supplies
        the modulus.
    loads :
        Force applied at every node.

    Returns
    -------
    forces :
        Axial force and both end moments of every member, the minor one exact
        zeros.

    Notes
    -----
    Not differentiable by tracing; `force_jacobian` carries the derivatives.
    Plain NumPy arrays, because this runs inside an FFI callback that forbids
    allocating a JAX array.
    """
    given = FrameSolve(xyz, diameters, loads, parameters=False)
    _build_model(model, family, given)

    read = _read_forces(model.structure.num_edges)
    bending = read.moments / NEWTON_MILLIMETER

    return MemberForces(
        axial_force=read.axial,
        moment_major=bending,
        moment_minor=np.zeros_like(bending),
    )


def _section_slopes(
    diameters: Float[Array, "members"],
    family: TubeFamily,
) -> tuple[Float[np.ndarray, "members"], Float[np.ndarray, "members"]]:
    """
    How a member's area and second moment move with its diameter.

    Returns
    -------
    slopes :
        Derivative of the area and of the second moment, in SI per millimeter.
    """
    outer = jnp.asarray(diameters) * MILLIMETER

    d_area = jax.vmap(jax.grad(lambda d: family(d).area))(outer)
    d_inertia = jax.vmap(jax.grad(lambda d: family(d).second_moment))(outer)

    return np.asarray(d_area) * MILLIMETER, np.asarray(d_inertia) * MILLIMETER


def _force_sensitivity(element: int, section: int, dof: int, tag: int) -> float:
    """
    One entry of a section's force sensitivity to one parameter.

    Notes
    -----
    `sensSectionForce` returns the section vector starting at the requested
    component, so element 0 is the value asked for. A tag never registered
    terminates the process with exit 139 and no traceback.
    """
    reading = ops.sensSectionForce(element, section, dof, tag)

    return float(reading[0]) if isinstance(reading, list) else float(reading)


def force_jacobian(
    model: Model,
    xyz: Float[Array, "nodes 3"],
    diameters: Float[Array, "members"],
    family: TubeFamily,
    loads: Float[Array, "nodes 3"],
) -> Jacobian:
    """
    Every derivative of the member forces, from one solve and a sweep.

    Parameters
    ----------
    model :
        The structure and its plane, from `prepare_model`.
    xyz :
        Position of every node, from form finding.
    diameters :
        Outer diameter of every member.
    family :
        The section family, whose ratio fixes the wall and whose grade supplies
        the modulus.
    loads :
        Force applied at every node.

    Returns
    -------
    jacobian :
        Dense derivative blocks of the axial force and the end moments.

    Notes
    -----
    Forward mode, which is what DDM is: one factorization reused, so each
    parameter costs a back-substitution, and the whole Jacobian arrives in one
    sweep. The cotangent rule is a contraction of it.
    """
    given = FrameSolve(xyz, diameters, loads, parameters=True)
    count = _build_model(model, family, given)

    num_nodes = np.asarray(xyz).shape[0]
    num_members = model.structure.num_edges

    axial = np.zeros((num_members, count))
    moments = np.zeros((num_members, 2, count))

    for tag in range(1, count + 1):
        for member in range(num_members):
            element = member + 1
            column = tag - 1
            axial[member, column] = _force_sensitivity(element, 1, DOF_AXIAL, tag)
            moments[member, 0, column] = _force_sensitivity(element, 1, DOF_MOMENT, tag)
            moments[member, 1, column] = _force_sensitivity(
                element, NUM_INTEGRATION_POINTS, DOF_MOMENT, tag
            )

    sweep = ParameterSweep(axial, moments, num_nodes)

    return _assemble_blocks(sweep, diameters, family, model.spanned)


def _assemble_blocks(
    sweep: ParameterSweep,
    diameters: Float[Array, "members"],
    family: TubeFamily,
    spanned: Plane,
) -> Jacobian:
    """
    Rearrange a parameter sweep into the blocks the stage differentiates in.

    Notes
    -----
    Two changes of variable: the solver's two in-plane directions are scattered
    back into three global ones, leaving the normal column zero, and its area
    and second moment are contracted with the section slopes into one derivative
    per diameter. Millimeters re-enter on both.
    """
    axial = sweep.axial
    moments = sweep.moments
    num_nodes = sweep.num_nodes
    num_members = axial.shape[0]

    coordinates = 2 * num_nodes
    d_area, d_inertia = _section_slopes(diameters, family)

    axial_force_xyz = np.zeros((num_members, num_nodes, 3))
    moment_major_xyz = np.zeros((num_members, 2, num_nodes, 3))

    for index, axis in enumerate(spanned.axes):
        columns = slice(index, coordinates, 2)
        axial_force_xyz[:, :, axis] = axial[:, columns] * MILLIMETER
        moment_major_xyz[:, :, :, axis] = moments[:, :, columns] * (
            MILLIMETER / NEWTON_MILLIMETER
        )

    sections = axial[:, coordinates:].reshape(num_members, num_members, 2)
    axial_force_diameter = sections[:, :, 0] * d_area + sections[:, :, 1] * d_inertia

    sections = moments[:, :, coordinates:].reshape(num_members, 2, num_members, 2)
    moment_major_diameter = (
        sections[:, :, :, 0] * d_area + sections[:, :, :, 1] * d_inertia
    ) / NEWTON_MILLIMETER

    return Jacobian(
        axial_force_xyz,
        axial_force_diameter,
        moment_major_xyz,
        moment_major_diameter,
    )
