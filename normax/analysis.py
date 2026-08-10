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
Frame analysis of a form-found geometry, the second stage of the pipeline.

Form finding hands over a geometry and nothing else: no prestress, no initial
member forces. The frame is analysed from an unstressed reference state, so it
must deform elastically before any internal force appears, and the axial forces
that come back are `smax`'s own product rather than a restatement of the force
densities that shaped it. Their agreement is a prediction, and it is what
`tests/test_equilibrium_consistency.py` measures.

Members are beams, not bars, so the analysis also returns the bending the
form-finder could not see. That is the reason this stage exists: the code check
downstream consumes moments, and a pin-jointed form-finder has none to give.

All lengths, forces and stresses cross the boundary through `normax.units`.
"""

from typing import NamedTuple

import numpy as np
from jaxtyping import Array
from jaxtyping import Bool
from jaxtyping import Float
from smax import BeamElement
from smax import LoadCase
from smax import Material
from smax import Node
from smax import PipeSection
from smax import PointLoad
from smax import Structure as Frame
from smax import Support
from smax import compile_structure
from smax import element_forces
from smax import solve
from smax import solve_buckling
from smax.dof import DOF_PER_NODE

from normax.ec3.sizing import Steel
from normax.ec3.sizing import Tube
from normax.structures import Structure
from normax.units import to_kilograms_per_cubic_metre
from normax.units import to_metres
from normax.units import to_newton_millimetres
from normax.units import to_pascals

# EN 1993-1-1 3.2.6. Enters only the torsional and out-of-plane response, both
# of which vanish in a planar frame under in-plane load.
POISSONS_RATIO = 0.3


class MemberForces(NamedTuple):
    """
    The internal forces a frame analysis reports for every member.

    Attributes
    ----------
    n_ed :
        Axial force, tension positive.
    m_y_ed :
        Bending moment about the major axis, at each end of the member.
    m_z_ed :
        Bending moment about the minor axis, at each end of the member.

    Notes
    -----
    Moments are given at the two ends rather than sampled along the span,
    because loads are applied at nodes alone and the moment therefore varies
    linearly in between. That is what makes the first row of EN 1993-1-1 Table
    B.3 exact here, and `normax.ec3.sizing.end_moments` consumes the two ends
    directly.

    The axial force is one number per member for the same reason: with no load
    along the span it does not vary, and the analysis is linear.
    """

    n_ed: Float[Array, "members"]
    m_y_ed: Float[Array, "members ends"]
    m_z_ed: Float[Array, "members ends"]


class Buckling(NamedTuple):
    """
    The elastic instability of a whole frame, rather than of one member.

    Attributes
    ----------
    factors :
        Multiple of the applied load at which the frame buckles, smallest first.
    shapes :
        Displacement of every node in each mode, in the six degrees of freedom.

    Notes
    -----
    The smallest factor is what EN 1993-1-1 writes as `α_cr`. A value below one
    says the frame becomes unstable before it is loaded to its design value, and
    a member check cannot see that: it reads the slenderness of one member over
    an assumed buckling length, while this reads the mode the whole structure
    has.

    Mode shapes are not normalised to any physical amplitude. An eigenvector
    fixes a shape and not a size, so only ratios within a mode mean anything.
    """

    factors: Float[Array, "modes"]
    shapes: Float[Array, "modes nodes 6"]


def fixities(
    structure: Structure,
    normal: int | None,
) -> Bool[np.ndarray, "nodes 6"]:
    """
    Which degrees of freedom are restrained at every node.

    Parameters
    ----------
    structure :
        The structure whose supported nodes are to be restrained.
    normal :
        Index of the global axis a planar structure has no thickness along, or
        None for a structure that occupies all three dimensions.

    Returns
    -------
    fixities :
        A flag per node and degree of freedom, ordered as translations then
        rotations.

    Raises
    ------
    ValueError
        If the normal axis is not 0, 1 or 2.

    Notes
    -----
    Supports restrain translation and leave rotation free, so a base carries no
    moment. Form finding restrains translation and nothing else, and a fixed
    base would inject end moments the form-finder never saw.

    A planar structure analysed by a three-dimensional solver is a mechanism:
    rotating the whole of it about the line joining its supports strains no
    member and moves no support, so the stiffness matrix is singular and the
    solve returns nan rather than a plausible wrong answer. Restraining the one
    translation normal to the plane removes that mode. The two rotations out of
    the plane are left free, being unexcited by an in-plane load, and the
    in-plane results are unchanged by restraining them.
    """
    if normal is not None and normal not in (0, 1, 2):
        raise ValueError(f"normal must be 0, 1 or 2, or None, got {normal}")

    num_nodes = structure.nodes.shape[0]

    flags = np.zeros((num_nodes, DOF_PER_NODE), dtype=bool)
    flags[np.asarray(structure.supports), :3] = True

    if normal is not None:
        flags[:, normal] = True

    return flags


def frame(
    structure: Structure,
    xyz: Float[Array, "nodes 3"],
    diameters: Float[Array, "members"],
    steel: Steel,
    tube: Tube,
    *,
    normal: int | None,
) -> Frame:
    """
    The frame model an analysis runs on, in coherent SI.

    Parameters
    ----------
    structure :
        The structure supplying the connectivity and the supported nodes.
    xyz :
        Position of every node, from form finding.
    diameters :
        Outer diameter of every member.
    steel :
        Material properties. Only the modulus and the density reach the model.
    tube :
        The section family, whose ratio fixes the wall thickness.
    normal :
        Index of the global axis a planar structure has no thickness along, or
        None for a structure that occupies all three dimensions.

    Returns
    -------
    frame :
        The nodes, beam elements and supports of the frame.

    Notes
    -----
    Every member is a beam with the full six degrees of freedom at each node,
    so bending and torsion are present by construction rather than released.

    Poisson's ratio is not carried by the material container, since no clause
    implemented here needs it, and is supplied from EN 1993-1-1 3.2.6.
    """
    material = Material(
        elasticity_modulus=to_pascals(steel.e_mod),
        yield_stress=to_pascals(steel.f_y),
        density=to_kilograms_per_cubic_metre(steel.density),
        poissons_ratio=POISSONS_RATIO,
    )

    positions = to_metres(xyz)
    nodes = [Node(index, xyz=positions[index]) for index in range(xyz.shape[0])]

    outer = to_metres(diameters)
    wall = outer / tube.ratio
    edges = np.asarray(structure.edges)
    elements = [
        BeamElement(
            member,
            nodes=(int(edges[member, 0]), int(edges[member, 1])),
            material=material,
            section=PipeSection(
                outer_diameter=outer[member],
                thickness=wall[member],
            ),
        )
        for member in range(edges.shape[0])
    ]

    flags = fixities(structure, normal)
    supports = [
        Support(node, flags[node]) for node in range(xyz.shape[0]) if flags[node].any()
    ]

    return Frame(nodes, elements, supports)


def forces(
    structure: Structure,
    xyz: Float[Array, "nodes 3"],
    diameters: Float[Array, "members"],
    steel: Steel,
    tube: Tube,
    *,
    normal: int | None,
    loads: Float[Array, "nodes 3"] | None = None,
) -> MemberForces:
    """
    Internal forces of a frame under a load case.

    Parameters
    ----------
    structure :
        The structure supplying the connectivity and the supports.
    xyz :
        Position of every node, from form finding.
    diameters :
        Outer diameter of every member.
    steel :
        Material properties.
    tube :
        The section family, whose ratio fixes the wall thickness.
    normal :
        Index of the global axis a planar structure has no thickness along, or
        None for a structure that occupies all three dimensions.
    loads :
        Force applied at every node. If None, the structure's own loads.

    Returns
    -------
    forces :
        Axial force and both end moments of every member.

    Notes
    -----
    Differentiable in the geometry and in the diameters, since the frame is
    assembled inside this call from those arrays rather than read off a model
    built beforehand.

    The load case is an argument because a structure is form-found under one
    case and has to be checked under several. Only the first of them leaves the
    members free of bending, that being the case the shape was chosen for.

    The reference state is unstressed, so the nodes displace before any force
    appears. Those displacements are the elastic response the form-finder does
    not model, not an error, and they are the whole of the gap between these
    axial forces and the product of force density and length.
    """
    model = compile_structure(
        frame(structure, xyz, diameters, steel, tube, normal=normal)
    )

    applied_loads = structure.loads if loads is None else loads
    applied = [
        PointLoad(node, load=applied_loads[node])
        for node in range(applied_loads.shape[0])
    ]
    response = solve(model, LoadCase(applied, model))

    field = element_forces(model, response, num_samples=2)

    return MemberForces(
        n_ed=field.nx[:, 0],
        m_y_ed=to_newton_millimetres(field.my),
        m_z_ed=to_newton_millimetres(field.mz),
    )


def buckling(
    structure: Structure,
    xyz: Float[Array, "nodes 3"],
    diameters: Float[Array, "members"],
    steel: Steel,
    tube: Tube,
    *,
    normal: int | None,
    num_modes: int = 1,
    loads: Float[Array, "nodes 3"] | None = None,
) -> Buckling:
    """
    Load factors at which the frame becomes elastically unstable.

    Parameters
    ----------
    structure :
        The structure supplying the connectivity and the supports.
    xyz :
        Position of every node, from form finding.
    diameters :
        Outer diameter of every member.
    steel :
        Material properties.
    tube :
        The section family, whose ratio fixes the wall thickness.
    normal :
        Index of the global axis a planar structure has no thickness along, or
        None for a structure that occupies all three dimensions.
    num_modes :
        Number of modes to return, smallest factor first. Static.
    loads :
        Load case the frame buckles under. If None, the structure's own loads.

    Returns
    -------
    buckling :
        The critical load factors and their mode shapes.

    Notes
    -----
    **The factor belongs to a load case and not to a structure.** A frame sized
    for its worst case is not necessarily least stable under that case, so a
    factor quoted without the case it was measured under says less than it
    appears to.

    **A diagnostic, never a differentiated quantity.** The eigenproblem is pure
    JAX and would trace, but an eigenvalue derivative is undefined where two
    modes cross, and a design under optimization moves modes around. Read the
    factors beside a design and size against a buckling length instead.

    The smallest factor is the multiple of the applied load at which the whole
    frame buckles, so it measures the assumption a member-by-member buckling
    length makes rather than restating it: a member check reads one member's
    slenderness, while this reads the mode the structure actually has. The two
    meet through the member slenderness, since the ratio of the cross-section
    resistance factor to the critical factor is that slenderness squared.

    Restraining the one translation normal to a plane leaves the modes in that
    plane, which is what makes the factor comparable with an in-plane member
    check rather than with a lateral one.
    """
    model = compile_structure(
        frame(structure, xyz, diameters, steel, tube, normal=normal)
    )

    applied_loads = structure.loads if loads is None else loads
    applied = [
        PointLoad(node, load=applied_loads[node])
        for node in range(applied_loads.shape[0])
    ]
    response = solve_buckling(model, LoadCase(applied, model), num_modes=num_modes)

    return Buckling(
        factors=response.buckling_factors,
        shapes=response.mode_shapes,
    )
