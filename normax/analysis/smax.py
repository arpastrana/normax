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

**A load case is a dense nodal array and stays one all the way to the solve.**
`smax` accepts one wherever it accepts a load case of its own, so nothing here
builds load objects, compiles channels or scatters values into them. That is the
same array `normax.loads` produces and the same one the OpenSees backend takes,
which is what makes a load case the one thing the two backends cannot disagree
about.

The stage's own vocabulary — what a member force is, which degrees of freedom a
support restrains — lives in `normax.analysis` and is shared with every backend.
All lengths, forces and stresses cross the boundary through `normax.units`.
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
from smax import Structure as Frame
from smax import Support
from smax import compile_structure
from smax import element_forces
from smax import solve

from normax.analysis import AbstractFrameAnalyzer
from normax.analysis import MemberForces
from normax.analysis import support_fixities
from normax.loads import stack_load_cases
from normax.materials import SteelGrade
from normax.sections import MemberSections
from normax.sections import TubeFamily
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


def frame_model(
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
        The tube every member is built as, and what it is made of. One section
        or one per member.

    Returns
    -------
    frame :
        The nodes, beam elements and supports of the frame.

    Notes
    -----
    Every member is a beam with the full six degrees of freedom at each node,
    so bending and torsion are present by construction rather than released.

    **A section rather than a family.** A frame is built out of the sections it
    has, and choosing one for a member is the check's business downstream; a
    family here would be an offer of sizes nothing in an analysis has any use
    for. The two leaves a tube carries are the two a pipe section wants, so no
    wall is derived on the way in.

    The supports are what `support_fixities` makes of the structure's own, which
    is more than it names wherever a planar structure would be a mechanism in
    three dimensions. Nothing about the plane is passed in, here or anywhere
    above: it is measured from the geometry.

    Poisson's ratio is not carried by the material container, since no clause
    implemented here needs it, and is supplied from EN 1993-1-1 3.2.6.
    """
    steel = section.material
    material = Material(
        elasticity_modulus=to_pascals(steel.e_mod),
        yield_stress=to_pascals(steel.f_y),
        density=to_kilograms_per_cubic_meter(steel.density),
        poissons_ratio=POISSONS_RATIO,
    )

    positions = to_meters(xyz)
    nodes = [Node(index, xyz=positions[index]) for index in range(xyz.shape[0])]

    members = (structure.num_edges,)
    outer = to_meters(jnp.broadcast_to(jnp.asarray(section.diameter), members))
    wall = to_meters(jnp.broadcast_to(jnp.asarray(section.thickness), members))
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

    flags = support_fixities(structure)
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
        The tube every member starts as, and what it is made of. A placeholder
        geometry, replaced at every call, carrying the material that is not.

    Returns
    -------
    model :
        The compiled assembly: parameter arrays and the degree of freedom maps.

    Notes
    -----
    **Host-side, and never called from inside a traced function.** Compiling a
    frame decides the degree of freedom maps by reading support flags with a
    Python conditional, which a tracer cannot follow; running it once here is
    what leaves everything downstream jittable.

    The starting geometry and the section given stand in for the values that
    matter. Both are overwritten by `member_forces` before anything is assembled,
    so what they are cannot reach a result — only their shapes can, and those
    come from the structure.

    **No load case is compiled here**, since the solver takes a dense nodal
    array and builds its own channels. What this settles is the assembly alone,
    and a load case is an argument of every call rather than a field of it.
    """
    frame = frame_model(structure, structure.nodes, section)

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
        The compiled assembly, from `prepare`.
    xyz :
        Position of every node, from form finding.
    diameters :
        Outer diameter of every member.
    section :
        The tube the members are built as, supplying the wall proportion the
        diameters are walled by and the material.

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
    `normax.sections` rather than from the solver's own section class, so the
    two stages cannot disagree about what a tube is; they agree algebraically.

    **The tube the block holds is restated at the diameters given rather than
    consulted.** What survives of it is its wall proportion, its grade and its
    class — everything but the geometry a caller replaces — and the restatement
    goes through the family those three define, that being the one place a wall
    is chosen for a diameter.

    Poisson's ratio is deliberately not replaced. It is a constant of this
    backend taken from EN 1993-1-1 rather than a field of the material container,
    so nothing upstream can vary it and the shear modulus follows the modulus
    that is replaced.

    Elements are grouped by type in the compiled assembly, so each group is
    indexed by the global member ids it holds. One group is the usual case here,
    every member being a beam.
    """
    family = TubeFamily(section.ratio, section.material)
    sections = family(to_meters(diameters))
    steel = sections.material
    gross = sections.area
    inertia = sections.second_moment

    e_mod = to_pascals(jnp.asarray(steel.e_mod))
    f_y = to_pascals(jnp.asarray(steel.f_y))
    density = to_kilograms_per_cubic_meter(jnp.asarray(steel.density))

    compiled = eqx.tree_at(lambda c: c.params.xyz, model, to_meters(jnp.asarray(xyz)))

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


def member_forces(
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
        The compiled assembly, from `prepare`.
    xyz :
        Position of every node, from form finding.
    diameters :
        Outer diameter of every member.
    section :
        The tube the members are built as, supplying the wall proportion the
        diameters are walled by and the material.
    loads :
        Force applied at every node.

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
    It reaches the solver as the same dense nodal array every other stage speaks
    in, so nothing here compiles, scatters or reshapes a load.

    The reference state is unstressed, so the nodes displace before any force
    appears. Those displacements are the elastic response the form-finder does
    not model, not an error, and they are the whole of the gap between these
    axial forces and the product of force density and length.
    """
    compiled = _injected_assembly(model, xyz, diameters, section)

    response = solve(compiled, loads)
    field = element_forces(compiled, response, num_samples=2)

    return MemberForces(
        axial_force=field.nx[:, 0],
        moment_major=to_newton_millimeters(field.my),
        moment_minor=to_newton_millimeters(field.mz),
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
    **One section, not a family.** A solver is configured with the sections its
    elements have, and choosing between sizes is what the check downstream does;
    a family here would offer an analysis a choice it never makes. What the tube
    is consulted for on a call is everything but its diameter, that being what
    the call replaces.

    The material travels on the block rather than reaching it per call, since a
    solver is configured with a material in its own terms. It is injected into
    the assembly on every call all the same, which is what leaves a derivative
    with respect to a material property defined rather than silently zero.

    **A structure is all the block is told.** Whether it is planar, and what a
    three-dimensional solve of it therefore needs restrained beyond the supports
    it names, is measured by `support_fixities` while the assembly is compiled.
    That decision is static either way — it selects degrees of freedom rather
    than scaling any number — so nothing about it becomes a traced leaf.

    Load cases are looped over rather than mapped, this solver not being
    traceable through `vmap`, so the cost is linear in their number. Each is
    handed over as the dense nodal array it already is, so the loop costs a
    solve and nothing else: no case is compiled, and the block carries none.
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
            The tube the frame is analyzed as, whose wall proportion walls the
            diameters of a call and whose grade supplies the material.
        """
        self.section = section
        self.model = prepare_model(structure, section)

    @property
    def steel(self) -> SteelGrade:
        """
        The material the frame is analyzed with, free of any standard.
        """
        return self.section.material

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
            forces = member_forces(
                self.model,
                xyz,
                diameters,
                self.section,
                load_case,
            )
            analyzed.append(forces)

        return stack_load_cases(analyzed)
