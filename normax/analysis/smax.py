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
The `smax` backend of the analysis stage, differentiated by tracing autodiff.

A JAX frame solver, so the assembly and the solve are traced end to end and the
derivatives come out of the same machinery that produced the geometry upstream.
The frame is analyzed from an unstressed reference state, so it must deform
elastically before any internal force appears, and the axial forces that come
back are `smax`'s own product rather than a restatement of the force densities
that shaped it. Their agreement is a prediction, and it is what
`tests/test_equilibrium_consistency.py` measures.

Three dimensions throughout, which is what the gridshell needs and what a direct
differentiation backend cannot supply. This is the reference the second backend
is measured against rather than the interesting one: the argument the stage makes
is that a solver which cannot be traced at all fits behind the same contract.

**The assembly is compiled once and the traced values are injected into it.**
Compiling a frame flattens model objects into arrays and works out the degree of
freedom maps, and only the second half depends on anything an optimizer varies —
which is to say none of it does. `prepare` runs both on the host and `forces`
replaces every array leaf that a geometry or a size reaches, so the maps are
computed once for a structure rather than once per call. That is also what makes
the stage jittable: compilation reads support flags with a Python conditional,
so it cannot happen inside a trace, and after this it never does.

The stage's own vocabulary — what a member force is, which degrees of freedom a
support restrains — lives in `normax.analysis` and is shared with every backend.
All lengths, forces and stresses cross the boundary through `normax.units`.
"""

from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from jaxtyping import Float
from smax import BeamElement
from smax import CompiledStructure
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

from normax.analysis import Buckling
from normax.analysis import MemberForces
from normax.analysis import support_fixities
from normax.ec3.material import SteelGrade
from normax.ec3.section import TubeCatalogue
from normax.structures import Structure
from normax.units import to_kilograms_per_cubic_meter
from normax.units import to_meters
from normax.units import to_newton_millimeters
from normax.units import to_pascals

# EN 1993-1-1 3.2.6. Enters only the torsional and out-of-plane response, both
# of which vanish in a planar frame under in-plane load.
POISSONS_RATIO = 0.3

# Torsion constant of a thin ring over its second moment. A tube is doubly
# symmetric, so the polar moment is twice the bending one.
TORSION_FACTOR = 2.0


class Model(NamedTuple):
    """
    Everything this backend can work out before a geometry or a size is chosen.

    Attributes
    ----------
    compiled :
        The compiled assembly: parameter arrays and the degree of freedom maps.
    loads :
        The structure's own load case, compiled into dense nodal channels.

    Notes
    -----
    Built by `prepare` on the host and reused for every call. The arrays it
    carries are placeholders wherever a geometry, a size or a material property
    reaches them, and `forces` replaces all of those; what is kept is the half
    that no optimizer varies, which is the connectivity, the support pattern and
    the degree of freedom maps built from them.

    The load case is compiled from the structure's own loads, so it doubles as
    the default case. A caller passing a load case of its own gets the same
    channels with different values rather than a second compilation.
    """

    compiled: CompiledStructure
    loads: LoadCase


def frame_model(
    structure: Structure,
    xyz: Float[Array, "nodes 3"],
    diameters: Float[Array, "members"],
    steel: SteelGrade,
    catalogue: TubeCatalogue,
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
    catalogue :
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
        density=to_kilograms_per_cubic_meter(steel.density),
        poissons_ratio=POISSONS_RATIO,
    )

    positions = to_meters(xyz)
    nodes = [Node(index, xyz=positions[index]) for index in range(xyz.shape[0])]

    outer = to_meters(diameters)
    wall = outer / catalogue.ratio
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

    flags = support_fixities(structure, normal)
    supports = [
        Support(node, flags[node]) for node in range(xyz.shape[0]) if flags[node].any()
    ]

    return Frame(nodes, elements, supports)


def prepare_model(
    structure: Structure,
    steel: SteelGrade,
    catalogue: TubeCatalogue,
    *,
    normal: int | None,
) -> Model:
    """
    Compile everything a solve needs that no design variable reaches.

    Parameters
    ----------
    structure :
        The structure supplying the connectivity, the supports and the loads.
    steel :
        Material properties. Placeholders only, replaced at every call.
    catalogue :
        The section family, whose ratio fixes the wall thickness.
    normal :
        Index of the global axis a planar structure has no thickness along, or
        None for a structure that occupies all three dimensions.

    Returns
    -------
    model :
        The compiled assembly and the structure's own load case.

    Notes
    -----
    **Host-side, and never called from inside a traced function.** Compiling a
    frame decides the degree of freedom maps by reading support flags with a
    Python conditional, which a tracer cannot follow; running it once here is
    what leaves everything downstream jittable.

    The starting geometry and the smallest tube in the family stand in for the
    values that matter. Both are overwritten by `forces` before anything is
    assembled, so what they are cannot reach a result — only their shapes can,
    and those come from the structure.
    """
    placeholder = jnp.full(structure.edges.shape[0], catalogue.diameter_min)

    compiled = compile_structure(
        frame_model(
            structure, structure.nodes, placeholder, steel, catalogue, normal=normal
        )
    )

    applied = [
        PointLoad(node, load=structure.loads[node])
        for node in range(structure.nodes.shape[0])
    ]

    return Model(compiled=compiled, loads=LoadCase(applied, compiled))


def _injected_assembly(
    model: Model,
    xyz: Float[Array, "nodes 3"],
    diameters: Float[Array, "members"],
    steel: SteelGrade,
    catalogue: TubeCatalogue,
) -> CompiledStructure:
    """
    A compiled assembly with every traced leaf replaced.

    Parameters
    ----------
    model :
        The compiled assembly, from `prepare`.
    xyz :
        Position of every node, from form finding.
    diameters :
        Outer diameter of every member.
    steel :
        Material properties.
    catalogue :
        The section family, whose ratio fixes the wall thickness.

    Returns
    -------
    compiled :
        The same maps, carrying the geometry and the sections given here.

    Notes
    -----
    **Every array a derivative might be taken through is replaced, not reused.**
    A leaf left alone keeps the placeholder `prepare` built it from, and since
    that placeholder is a constant the gradient with respect to it is silently
    zero rather than an error. The section properties come from
    `normax.ec3.section` rather than from the solver's own section class, so the
    two stages cannot disagree about what a tube is; they agree algebraically.

    Poisson's ratio is deliberately not replaced. It is a constant of this
    backend taken from EN 1993-1-1 rather than a field of the material container,
    so nothing upstream can vary it and the shear modulus follows the modulus
    that is replaced.

    Elements are grouped by type in the compiled assembly, so each group is
    indexed by the global member ids it holds. One group is the usual case here,
    every member being a beam.
    """
    outer = to_meters(diameters)
    gross = catalogue.tube_at(outer).area
    inertia = catalogue.tube_at(outer).second_moment

    e_mod = to_pascals(jnp.asarray(steel.e_mod))
    f_y = to_pascals(jnp.asarray(steel.f_y))
    density = to_kilograms_per_cubic_meter(jnp.asarray(steel.density))

    compiled = eqx.tree_at(
        lambda c: c.params.xyz, model.compiled, to_meters(jnp.asarray(xyz))
    )

    groups = enumerate(compiled.topology.element_group_ids)
    for index, member_ids in groups:
        held = jnp.asarray(member_ids)
        compiled = eqx.tree_at(
            lambda c, i=index: (
                c.params.element_groups[i].section.area,
                c.params.element_groups[i].section.Iy,
                c.params.element_groups[i].section.Iz,
                c.params.element_groups[i].section.J,
                c.params.element_groups[i].material.elasticity_modulus,
                c.params.element_groups[i].material.yield_stress,
                c.params.element_groups[i].material.density,
            ),
            compiled,
            (
                gross[held],
                inertia[held],
                inertia[held],
                TORSION_FACTOR * inertia[held],
                jnp.broadcast_to(e_mod, held.shape),
                jnp.broadcast_to(f_y, held.shape),
                jnp.broadcast_to(density, held.shape),
            ),
        )

    return compiled


def _load_case(
    model: Model,
    loads: Float[Array, "nodes 3"],
) -> LoadCase:
    """
    The structure's compiled load channels, carrying a different load case.

    Parameters
    ----------
    model :
        The compiled assembly, from `prepare`.
    loads :
        Force applied at every node.

    Returns
    -------
    loads :
        A load case of the same shape as the model's own.

    Notes
    -----
    A compiled load case is a dense channel per node and degree of freedom, so
    swapping a load case is an array replacement rather than a second compilation. The
    three translations are written and the three moments left at zero, loads
    being forces here.
    """
    channels = jnp.zeros_like(model.loads.node_loads)

    return eqx.tree_at(
        lambda c: c.node_loads, model.loads, channels.at[:, :3].set(loads)
    )


def member_forces(
    model: Model,
    xyz: Float[Array, "nodes 3"],
    diameters: Float[Array, "members"],
    steel: SteelGrade,
    catalogue: TubeCatalogue,
    *,
    loads: Float[Array, "nodes 3"] | None = None,
) -> MemberForces:
    """
    Internal forces of a frame under a load case.

    Parameters
    ----------
    model :
        The compiled assembly, from `prepare`.
    xyz :
        Position of every node, from form finding.
    diameters :
        Outer diameter of every member.
    steel :
        Material properties.
    catalogue :
        The section family, whose ratio fixes the wall thickness.
    loads :
        Force applied at every node. If None, the structure's own loads.

    Returns
    -------
    forces :
        Axial force and both end moments of every member.

    Notes
    -----
    Differentiable in the geometry, in the diameters and in every material
    property, since all of them are injected into the assembly here rather than
    baked into it when the model was built.

    The load case is an argument because a structure is form-found under one
    load case and has to be checked under several. Only the first of them leaves
    the members free of bending, that being the one the shape was chosen for.

    The reference state is unstressed, so the nodes displace before any force
    appears. Those displacements are the elastic response the form-finder does
    not model, not an error, and they are the whole of the gap between these
    axial forces and the product of force density and length.
    """
    compiled = _injected_assembly(model, xyz, diameters, steel, catalogue)
    load_case = model.loads if loads is None else _load_case(model, loads)

    response = solve(compiled, load_case)
    field = element_forces(compiled, response, num_samples=2)

    return MemberForces(
        axial_force=field.nx[:, 0],
        moment_major=to_newton_millimeters(field.my),
        moment_minor=to_newton_millimeters(field.mz),
    )


def buckling_modes(
    model: Model,
    xyz: Float[Array, "nodes 3"],
    diameters: Float[Array, "members"],
    steel: SteelGrade,
    catalogue: TubeCatalogue,
    *,
    num_modes: int = 1,
    loads: Float[Array, "nodes 3"] | None = None,
) -> Buckling:
    """
    Load factors at which the frame becomes elastically unstable.

    Parameters
    ----------
    model :
        The compiled assembly, from `prepare`.
    xyz :
        Position of every node, from form finding.
    diameters :
        Outer diameter of every member.
    steel :
        Material properties.
    catalogue :
        The section family, whose ratio fixes the wall thickness.
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
    for its worst load case is not necessarily least stable under it, so a
    factor quoted without the load case it was measured under says less than it
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
    compiled = _injected_assembly(model, xyz, diameters, steel, catalogue)
    load_case = model.loads if loads is None else _load_case(model, loads)

    response = solve_buckling(compiled, load_case, num_modes=num_modes)

    return Buckling(
        factors=response.buckling_factors,
        shapes=response.mode_shapes,
    )
