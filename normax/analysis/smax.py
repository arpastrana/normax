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
The `smax` frame analysis, differentiated by tracing autodiff.

A JAX frame solver in three dimensions, so the assembly and the solve are traced
end to end. It is the oracle the shipped backends are measured against, not one
of them. The assembly is compiled once on the host, where support flags are
read with a Python conditional, and every leaf a geometry or a size reaches is
injected into it per call, which is what leaves the stage jittable. A load case
is the same dense nodal array every other stage speaks in.
"""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from jaxtyping import Float
from smax import BeamElement
from smax import CompiledStructure
from smax import Material
from smax import Node
from smax import PipeSection
from smax import Response
from smax import Structure as Frame
from smax import Support
from smax import compile_structure
from smax import element_forces
from smax import solve

from normax.analysis import AbstractFrameAnalyzer
from normax.analysis import MemberForces
from normax.analysis import restrain_supports
from normax.loads import stack_load_cases
from normax.sections import MemberSections
from normax.sections import TubeFamily
from normax.structures import Structure
from normax.units import MEGAPASCAL
from normax.units import MILLIMETER
from normax.units import NEWTON_MILLIMETER
from normax.units import TONNE_PER_CUBIC_MILLIMETER

# EN 1993-1-1 3.2.6. Enters only the torsional and out-of-plane response.
POISSONS_RATIO = 0.3

# The polar moment of an annulus is twice its second moment.
TORSION_FACTOR = 2.0


def assemble_frame_model(
    structure: Structure,
    xyz: Float[Array, "nodes 3"],
    section: MemberSections,
) -> Frame:
    """
    The frame model an analysis runs on, in coherent SI.

    Parameters
    ----------
    structure :
        The structure supplying the connectivity and the supported nodes.
    xyz :
        Position of every node, from form finding.
    section :
        The tube every member is built as, one section or one per member.

    Returns
    -------
    frame :
        The nodes, beam elements and supports of the frame.

    Notes
    -----
    Every member is a beam with six degrees of freedom at each node. The
    supports are what `restrain_supports` makes of the structure's own, measured
    from the geometry rather than declared.
    """
    steel = section.material
    material = Material(
        elasticity_modulus=steel.e_mod * MEGAPASCAL,
        yield_stress=steel.f_y * MEGAPASCAL,
        density=steel.density * TONNE_PER_CUBIC_MILLIMETER,
        poissons_ratio=POISSONS_RATIO,
    )

    positions = xyz * MILLIMETER
    nodes = [Node(index, xyz=positions[index]) for index in range(xyz.shape[0])]

    members = (structure.num_edges,)
    outer = jnp.broadcast_to(jnp.asarray(section.diameter), members) * MILLIMETER
    wall = jnp.broadcast_to(jnp.asarray(section.thickness), members) * MILLIMETER
    edges = np.asarray(structure.edges)
    elements = []
    for member in range(edges.shape[0]):
        pipe = PipeSection(outer_diameter=outer[member], thickness=wall[member])
        ends = (int(edges[member, 0]), int(edges[member, 1]))
        elements.append(
            BeamElement(member, nodes=ends, material=material, section=pipe)
        )

    flags = restrain_supports(structure)
    supports = [
        Support(node, flags[node]) for node in range(xyz.shape[0]) if flags[node].any()
    ]

    return Frame(nodes, elements, supports)


def prepare_model(
    structure: Structure,
    section: MemberSections,
) -> CompiledStructure:
    """
    Compile everything a solve needs that no design variable reaches.

    Parameters
    ----------
    structure :
        The structure supplying the connectivity and the supported nodes.
    section :
        A placeholder tube, replaced at every call, carrying the material.

    Returns
    -------
    model :
        The compiled assembly: parameter arrays and the degree of freedom maps.

    Notes
    -----
    Host-side and never traced. Only the shapes of the placeholder geometry and
    section survive into a result; their values are overwritten per call.
    """
    frame = assemble_frame_model(structure, structure.nodes, section)

    return compile_structure(frame)


def _injected_assembly(
    model: CompiledStructure,
    xyz: Float[Array, "nodes 3"],
    diameters: Float[Array, "members"],
    section: MemberSections,
) -> CompiledStructure:
    """
    A compiled assembly with every traced leaf replaced.

    Parameters
    ----------
    model :
        The compiled assembly, from `prepare_model`.
    xyz :
        Position of every node, from form finding.
    diameters :
        Outer diameter of every member.
    section :
        The tube supplying the wall proportion and the material.

    Returns
    -------
    compiled :
        The same maps, carrying the geometry and the sections given here.

    Notes
    -----
    A leaf left alone keeps its placeholder, and its gradient is then silently
    zero, so everything a derivative might be taken through is replaced. The
    section properties come from `normax.sections`, so the two stages cannot
    disagree about what a tube is.
    """
    family = TubeFamily(section.ratio, section.material)
    sections = family(diameters * MILLIMETER)
    steel = sections.material
    gross = sections.area
    inertia = sections.second_moment

    e_mod = jnp.asarray(steel.e_mod) * MEGAPASCAL
    f_y = jnp.asarray(steel.f_y) * MEGAPASCAL
    density = jnp.asarray(steel.density) * TONNE_PER_CUBIC_MILLIMETER

    compiled = eqx.tree_at(lambda c: c.params.xyz, model, jnp.asarray(xyz) * MILLIMETER)

    groups = enumerate(compiled.topology.element_group_ids)
    for index, member_ids in groups:
        held = jnp.asarray(member_ids)
        replaced = (
            gross[held],
            inertia[held],
            inertia[held],
            TORSION_FACTOR * inertia[held],
            jnp.broadcast_to(e_mod, held.shape),
            jnp.broadcast_to(f_y, held.shape),
            jnp.broadcast_to(density, held.shape),
        )
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
            replaced,
        )

    return compiled


def compute_member_forces(
    model: CompiledStructure,
    xyz: Float[Array, "nodes 3"],
    diameters: Float[Array, "members"],
    section: MemberSections,
    loads: Float[Array, "nodes 3"],
) -> MemberForces:
    """
    Internal forces of a frame under a load case.

    Parameters
    ----------
    model :
        The compiled assembly, from `prepare_model`.
    xyz :
        Position of every node, from form finding.
    diameters :
        Outer diameter of every member.
    section :
        The tube supplying the wall proportion and the material.
    loads :
        Force applied at every node.

    Returns
    -------
    forces :
        Axial force and both end moments of every member.

    Notes
    -----
    Differentiable in the geometry, the diameters and every material property.
    The reference state is unstressed, so the nodes displace before any force
    appears, and that elastic response is the whole gap between these axial
    forces and the force density times the length.
    """
    compiled = _injected_assembly(model, xyz, diameters, section)

    response = solve(compiled, loads)
    field = element_forces(compiled, response, num_samples=2)

    return MemberForces(
        axial_force=field.nx[:, 0],
        moment_major=field.my / NEWTON_MILLIMETER,
        moment_minor=field.mz / NEWTON_MILLIMETER,
    )


class SmaxAnalyzer(AbstractFrameAnalyzer):
    """
    A frame analysis traced in JAX, as a block of the design pipeline.

    Attributes
    ----------
    section :
        The tube the members are analyzed as, supplying the wall proportion the
        diameters of a call are walled by and the material they are made of.
    model :
        The assembly, compiled when the block is built.

    Notes
    -----
    Load cases are looped over rather than mapped, this solver not being
    traceable through `vmap`; each costs a solve and nothing else.
    """

    section: MemberSections
    model: CompiledStructure

    def __init__(
        self,
        structure: Structure,
        section: MemberSections,
    ) -> None:
        """
        Build an analyzer by compiling a structure's assembly.

        Parameters
        ----------
        structure :
            The structure supplying the connectivity and the supported nodes.
        section :
            The tube the frame is analyzed as.
        """
        self.section = section
        self.model = prepare_model(structure, section)

    def solve_response(
        self,
        xyz: Float[Array, "nodes 3"],
        diameters: Float[Array, "members"],
        loads: Float[Array, "nodes 3"],
    ) -> Response:
        """
        The solver's whole response under one load case, for a viewer to draw.

        Parameters
        ----------
        xyz :
            Position of every node, from a form finder.
        diameters :
            Outer diameter of every member, setting the stiffness.
        loads :
            Force applied at every node.

        Returns
        -------
        response :
            Displacements and reactions, in the solver's own terms and units.
        """
        compiled = _injected_assembly(self.model, xyz, diameters, self.section)

        return solve(compiled, loads)

    def __call__(
        self,
        xyz: Float[Array, "nodes 3"],
        diameters: Float[Array, "members"],
        loads: Float[Array, "load_cases nodes 3"],
    ) -> MemberForces:
        """
        Analyze one geometry under every load case it is checked against.

        Parameters
        ----------
        xyz :
            Position of every node, from a form finder.
        diameters :
            Outer diameter of every member, setting the stiffness.
        loads :
            Force applied at every node in every load case.

        Returns
        -------
        forces :
            Axial force and both end moments, per load case and member.
        """
        analyzed = []
        for load_case in loads:
            forces = compute_member_forces(
                self.model, xyz, diameters, self.section, load_case
            )
            analyzed.append(forces)

        return stack_load_cases(analyzed)
