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

**The frame stability check lives here too, and only here.** It is the one
question that needs an eigensolve rather than a linear solve, so only a backend
that can trace one can answer it; the OpenSees one cannot. It is soft
validation — nothing it produces feeds a design or a mass, nothing crosses a
Tesseract boundary, and nothing is differentiated, an eigenvalue derivative
being undefined where two modes cross and a symmetric structure having
degenerate pairs. `normax/ec3/stability.py` implements the clauses it reads.

The stage's own vocabulary — what a member force is, which degrees of freedom a
support restrains — lives in `normax.analysis` and is shared with every backend.
All lengths, forces and stresses cross the boundary through `normax.units`.
"""

from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from jaxtyping import Bool
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
from smax import solve_buckling

from normax.analysis import AbstractFrameAnalyzer
from normax.analysis import Buckling
from normax.analysis import MemberForces
from normax.analysis import support_fixities
from normax.design import Design
from normax.ec3.material import Steel
from normax.ec3.resistance import force_critical
from normax.ec3.resistance import slenderness_from_force
from normax.ec3.section import Tube
from normax.ec3.section import TubeCatalogue
from normax.ec3.stability import ALPHA_CR_ELASTIC
from normax.ec3.stability import amplifier_resistance
from normax.ec3.stability import buckling_length_global
from normax.ec3.stability import is_adequate
from normax.ec3.stability import slenderness_global
from normax.ec3.stability import utilization_frame as utilization_stability
from normax.loads import stack_load_cases
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
    section: Tube,
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
    section: Tube,
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
    section: Tube,
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
    `normax.ec3.section` rather than from the solver's own section class, so the
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
    family = TubeCatalogue(section.ratio, section.section_class, section.material)
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
    section: Tube,
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


def buckling_modes(
    model: CompiledStructure,
    xyz: Float[Array, "nodes 3"],
    diameters: Float[Array, "members"],
    section: Tube,
    loads: Float[Array, "nodes 3"],
    *,
    num_modes: int = 1,
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
    section :
        The tube the members are built as, supplying the wall proportion the
        diameters are walled by and the material.
    loads :
        Load case the frame buckles under.
    num_modes :
        Number of modes to return, smallest factor first. Static.

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
    compiled = _injected_assembly(model, xyz, diameters, section)

    response = solve_buckling(compiled, loads, num_modes=num_modes)

    return Buckling(
        factors=response.buckling_factors,
        shapes=response.mode_shapes,
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

    section: Tube
    model: CompiledStructure

    def __init__(
        self,
        structure: Structure,
        section: Tube,
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
    def steel(self) -> Steel:
        """
        The material the frame is analyzed with.
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


class Stability(NamedTuple):
    """
    The global stability check of a finished design, and both routes to it.

    Attributes
    ----------
    factors :
        Critical load factors of the frame, smallest first.
    utilization :
        Demand over resistance of EN 1993-1-1 §5.2.1, at most one where
        first-order analysis is adequate.
    adequate :
        Whether the frame satisfies that clause.
    slenderness_member :
        Slenderness each member has from its assumed buckling length, Eq. 6.50.
    slenderness_global :
        Slenderness each member has from the frame's critical load factor,
        §6.3.4(3).
    buckling_length_equivalent :
        Buckling length the critical load factor is equivalent to.

    Notes
    -----
    The two slendernesses are the same equation given different questions. Their
    ratio is the size of the assumption a member buckling length makes, and it is
    one wherever that assumption is right.

    The clauses behind `utilization` and `adequate` are EN 1993-1-1 §5.2.1(3),
    verified in `docs/clauses.md`.
    """

    factors: Float[Array, "modes"]
    utilization: Float[Array, ""]
    adequate: Bool[Array, ""]
    slenderness_member: Float[Array, "members"]
    slenderness_global: Float[Array, "members"]
    buckling_length_equivalent: Float[Array, "members"]


def frame_stability(
    design: Design,
    analyzer: "SmaxAnalyzer",
    loads: Float[Array, "nodes 3"],
    *,
    load_case: int = 0,
    num_modes: int = 1,
) -> Stability:
    """
    Check a finished design against the stability of the frame it sits in.

    Parameters
    ----------
    design :
        A design, from a pipeline.
    analyzer :
        The analysis block, supplying the assembly and the material.
    loads :
        Load case the frame buckles under.
    load_case :
        Index of the design's load case the axial forces are read from. Static.
    num_modes :
        Number of critical load factors to return. Static.

    Returns
    -------
    stability :
        The critical load factors, the verdict, and both routes to slenderness.

    Notes
    -----
    Global stability is not covered by what this package designs; this only
    says how far the buckling length that produced a design can be trusted.

    It could not enter the sizing map in any case. That roots a member check,
    which is local and monotone in one diameter, while this is a property of the
    whole frame: a design failing here is not made to pass by growing one member,
    and the remedy is bracing or a different buckling length.

    **Never differentiated.** The eigenproblem would trace, but an eigenvalue
    derivative is undefined where two modes cross — and they do, since a
    symmetric structure has degenerate pairs.

    The sections are restated at the designed diameters through the family the
    block's tube belongs to rather than read off the design, so the wall stays
    proportional to the diameter. An envelope over load cases smooths a diameter
    and a wall independently and can leave the two barely out of proportion.

    Members carrying no axial force report nan for both slendernesses and for the
    equivalent buckling length, a factor scaling the whole load having nothing to
    say about a member the load never reaches.
    """
    modes = buckling_modes(
        analyzer.model,
        design.shape.xyz,
        design.sizes.sections.diameter,
        analyzer.section,
        loads,
        num_modes=num_modes,
    )
    alpha_cr = modes.factors[0]

    steel = analyzer.steel
    analyzed = analyzer.section
    family = TubeCatalogue(analyzed.ratio, analyzed.section_class, steel)
    tubes = family(design.sizes.sections.diameter)
    gross = tubes.area
    inertia = tubes.second_moment
    axial_force = design.sizes.actions.axial_force[load_case]

    return Stability(
        factors=modes.factors,
        utilization=utilization_stability(alpha_cr, ALPHA_CR_ELASTIC),
        adequate=is_adequate(alpha_cr, ALPHA_CR_ELASTIC),
        slenderness_member=slenderness_from_force(
            gross, steel, force_critical(inertia, design.shape.lengths, steel)
        ),
        slenderness_global=slenderness_global(
            amplifier_resistance(gross, steel, axial_force), alpha_cr
        ),
        buckling_length_equivalent=buckling_length_global(
            alpha_cr, axial_force, inertia, steel
        ),
    )
