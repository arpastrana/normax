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
The three stages composed into one differentiable function, from `q` to a mass.

Form finding chooses the shape, a frame analysis says what the members carry,
and EN 1993-1-1 says how big they have to be. Each stage differentiates in a
different way — a traced linear solve, a traced assembly and solve, and an
implicit tangent taken at the root of a residual — and the composition hides
that from the caller: `mass` is a scalar with a gradient in `q`.

This module is the in-process version, and it is the oracle the Tesseract
composition is measured against rather than scaffolding for it. Keeping a
baseline that reproduces the same mass and the same gradient is what turns
"the Tesseracts run" into "the boundary is transparent".

**The coupling between the analysis and the check is staggered.** A frame cannot
be analysed without sections, and the sections are what the check returns, so
the diameters the frame is built from are an input here and the diameters the
check requires are an output. One pass is taken, not a fixed point. On a
funicular arch the two are close because the analysis barely depends on the
section — the internal forces move as the square of the diameter over the
length — but the gap is real and `experiments/09_arch_pipeline_jax.py` measures
how fast repeating the pass closes it.
"""

from typing import NamedTuple

import jax.numpy as jnp
from jax_fdm.equilibrium import EquilibriumStructure
from jaxtyping import Array
from jaxtyping import Bool
from jaxtyping import Float
from jaxtyping import Int

from normax.analysis.smax import Model
from normax.analysis.smax import buckling
from normax.analysis.smax import forces
from normax.ec3.actions import MemberActions
from normax.ec3.material import SteelGrade
from normax.ec3.resistance import force_critical
from normax.ec3.resistance import slenderness_from_force
from normax.ec3.section import area
from normax.ec3.section import second_moment
from normax.ec3.sizing import Tube
from normax.ec3.sizing import diameter_envelope as envelope_ec3
from normax.ec3.sizing import diameter_required as diameter_ec3
from normax.ec3.sizing import end_moments
from normax.ec3.sizing import governing_limit_state as governing_ec3
from normax.ec3.sizing import mass as mass_ec3
from normax.ec3.sizing import utilization_design as utilization_ec3
from normax.ec3.stability import ALPHA_CR_ELASTIC
from normax.ec3.stability import amplifier_resistance
from normax.ec3.stability import buckling_length_global
from normax.ec3.stability import is_adequate
from normax.ec3.stability import slenderness_global
from normax.ec3.stability import utilization_frame as utilization_stability
from normax.formfinding import equilibrium
from normax.structures import Structure


def actions(
    model: Model,
    xyz: Float[Array, "nodes 3"],
    diameters: Float[Array, "members"],
    steel: SteelGrade,
    tube: Tube,
    *,
    loads: Float[Array, "nodes 3"] | None = None,
) -> MemberActions:
    """
    Analyse a geometry and read the result as the check states its terms.

    Parameters
    ----------
    model :
        The prepared analysis model, from `normax.analysis.smax.prepare`.
    xyz :
        Position of every node, from form finding.
    diameters :
        Outer diameter of every member, setting the stiffness.
    steel :
        Material properties.
    tube :
        The section family every member is drawn from.
    loads :
        Load case to analyse under. If None, the structure's own loads.

    Returns
    -------
    actions :
        Axial force, both design moments and both moment factors.
    """
    member = forces(model, xyz, diameters, steel, tube, loads=loads)

    m_y_ed, c_my = end_moments(member.m_y_ed[:, 0], member.m_y_ed[:, 1])
    m_z_ed, c_mz = end_moments(member.m_z_ed[:, 0], member.m_z_ed[:, 1])

    return MemberActions(member.n_ed, m_y_ed, m_z_ed, c_my, c_mz)


class Design(NamedTuple):
    """
    Everything the three stages produce for one set of force densities.

    Attributes
    ----------
    xyz :
        Position of every node at equilibrium.
    lengths :
        Length of every member.
    actions :
        Axial force, both design moments and both moment factors of every
        member.
    l_cr :
        Buckling length used for every member.
    diameters :
        Outer diameter EN 1993-1-1 requires of every member.
    utilization :
        Demand over resistance at those diameters. One wherever a clause decided
        the size, and below one wherever the catalogue minimum did.
    mass :
        Total mass of the members.

    Notes
    -----
    Every field is a differentiable leaf. The diagnostic saying which limit
    state decided each size is deliberately absent, since a concrete cotangent
    on it would be an error; ask `governing` for it separately.

    The actions are carried whole rather than flattened to the major axis, so
    that a design can be re-checked against exactly what sized it.
    """

    xyz: Float[Array, "nodes 3"]
    lengths: Float[Array, "members"]
    actions: MemberActions
    l_cr: Float[Array, "members"]
    diameters: Float[Array, "members"]
    utilization: Float[Array, "members"]
    mass: Float[Array, ""]


def design(
    q: Float[Array, "members"],
    diameters: Float[Array, "members"],
    structure: Structure,
    graph: EquilibriumStructure,
    model: Model,
    steel: SteelGrade,
    tube: Tube,
    *,
    plastic: bool,
    resultant: bool = True,
    l_cr: Float[Array, "members"] | None = None,
    loads: Float[Array, "nodes 3"] | None = None,
) -> Design:
    """
    Form-find, analyse and size, in that order.

    Parameters
    ----------
    q :
        Force density of every member. Negative in compression.
    diameters :
        Diameters the frame is analysed with, being the previous outer iterate
        of the check. They set the stiffness, not the resistance.
    structure :
        The structure supplying the connectivity, the supports and the loads.
    graph :
        The form-finding connectivity, from `normax.formfinding.graph`.
    model :
        The prepared analysis model, from `normax.analysis.smax.prepare`.
    steel :
        Material properties and partial factors.
    tube :
        The section family every member is drawn from.
    plastic :
        Whether the section is Class 1 or 2. Static, never a traced value.
    resultant :
        Whether the two moments combine as a resultant in the cross-section
        check, or as a linear sum.
    l_cr :
        Buckling length of every member. If None, each member buckles over its
        own length.
    loads :
        Load case the frame is analysed under. If None, the structure's own
        loads, which are also the ones it is form-found under.

    Returns
    -------
    design :
        The geometry, the member actions, the required sizes and the mass.

    Notes
    -----
    Differentiable in the force densities, in the analysed diameters, and in
    every material property. The cross-section class selects a clause rather
    than scaling a number, so it stays a static Python value throughout.

    **Form finding always uses the structure's own loads, and only the analysis
    takes a case.** The shape answers to one load case by construction, that
    being what makes it funicular, and choosing the shape again for every case
    would mean a different structure per case rather than one structure checked
    against several.

    **The default buckling length is the member's own length, and that is a
    strong assumption rather than a conservative one.** It presumes every node is
    held in position by structure outside the model, so that a member can only
    buckle between its ends. Where that holds — a gridshell, whose hoop members
    brace its radial ones — it is the right reading. Where it does not, the
    structure buckles in a mode spanning many members and the assumption is
    unsafe, not cautious: measured on the arch, the critical load factor of the
    fully-stressed design is far below one, and sizing against the mode the
    structure actually has costs several times the mass. `normax.analysis.buckling`
    reports that factor so the assumption is visible next to the design.

    The buckling length is therefore an argument and never derived from the mesh.
    A member length shortens as the mesh refines, which changes the physics
    rather than the discretization: slenderness falls, the reduction factor
    approaches one, and the mass converges to the squash limit instead of to a
    mesh-independent design.
    """
    state = equilibrium(q, structure, graph)
    lengths = state.lengths[:, 0]

    acting = actions(model, state.xyz, diameters, steel, tube, loads=loads)

    buckling = lengths if l_cr is None else l_cr

    required = diameter_ec3(
        acting, buckling, steel, tube, plastic=plastic, resultant=resultant
    )

    used = utilization_ec3(
        required, acting, buckling, steel, tube, plastic=plastic, resultant=resultant
    )

    return Design(
        xyz=state.xyz,
        lengths=lengths,
        actions=acting,
        l_cr=buckling,
        diameters=required,
        utilization=used,
        mass=mass_ec3(required, lengths, steel, tube),
    )


def mass(
    q: Float[Array, "members"],
    diameters: Float[Array, "members"],
    structure: Structure,
    graph: EquilibriumStructure,
    model: Model,
    steel: SteelGrade,
    tube: Tube,
    *,
    plastic: bool,
    resultant: bool = True,
    l_cr: Float[Array, "members"] | None = None,
    loads: Float[Array, "nodes 3"] | None = None,
) -> Float[Array, ""]:
    """
    Total mass of the members EN 1993-1-1 requires at a set of force densities.

    Parameters
    ----------
    q :
        Force density of every member. Negative in compression.
    diameters :
        Diameters the frame is analysed with, being the previous outer iterate.
    structure :
        The structure supplying the connectivity, the supports and the loads.
    graph :
        The form-finding connectivity, from `normax.formfinding.graph`.
    model :
        The prepared analysis model, from `normax.analysis.smax.prepare`.
    steel :
        Material properties and partial factors.
    tube :
        The section family every member is drawn from.
    plastic :
        Whether the section is Class 1 or 2. Static, never a traced value.
    resultant :
        Whether the two moments combine as a resultant in the cross-section
        check, or as a linear sum.
    l_cr :
        Buckling length of every member. If None, each member buckles over its
        own length.
    loads :
        Load case the frame is analysed under. If None, the structure's own
        loads.

    Returns
    -------
    mass :
        Total mass.

    Notes
    -----
    The objective the whole pipeline exists to serve, and the scalar `jax.grad`
    is taken of. One reverse pass crosses all three stages, so its cost does not
    grow with the number of force densities.

    One load case only. A structure has to carry all of them, so the objective
    an optimizer sees is `envelope` rather than this.
    """
    return design(
        q,
        diameters,
        structure,
        graph,
        model,
        steel,
        tube,
        plastic=plastic,
        resultant=resultant,
        l_cr=l_cr,
        loads=loads,
    ).mass


class Envelope(NamedTuple):
    """
    One structure sized to carry every load case it is checked against.

    Attributes
    ----------
    xyz :
        Position of every node at equilibrium.
    lengths :
        Length of every member.
    l_cr :
        Buckling length used for every member.
    n_ed :
        Axial force of every member under every case, tension positive.
    m_y_ed :
        Larger major-axis end moment of every member under every case.
    m_z_ed :
        Larger minor-axis end moment of every member under every case.
    c_my :
        Major-axis moment factor of every member under every case.
    c_mz :
        Minor-axis moment factor of every member under every case.
    required :
        Diameter every case demands of every member on its own.
    diameters :
        Diameter carrying every case, being the smooth envelope of the above.
    utilization :
        Demand over resistance of every member under every case, at those
        diameters. At most one everywhere, and one for the case that governs.
    mass :
        Total mass of the members.

    Notes
    -----
    The geometry is shared, because a structure is form-found once and then
    asked to survive several things. Only the actions and the sizes carry a
    case axis.

    Every field is a differentiable leaf. Which case governs each member is a
    diagnostic and is absent for the same reason `Design` omits the limit
    state; ask `governing_case` for it.

    The actions are carried in full, rather than only the ones a report would
    print, so the design can be checked again at a different set of sizes
    without analysing anything a second time. `unsmoothed` is what does that.
    """

    xyz: Float[Array, "nodes 3"]
    lengths: Float[Array, "members"]
    l_cr: Float[Array, "members"]
    n_ed: Float[Array, "cases members"]
    m_y_ed: Float[Array, "cases members"]
    m_z_ed: Float[Array, "cases members"]
    c_my: Float[Array, "cases members"]
    c_mz: Float[Array, "cases members"]
    required: Float[Array, "cases members"]
    diameters: Float[Array, "members"]
    utilization: Float[Array, "cases members"]
    mass: Float[Array, ""]


def envelope(
    q: Float[Array, "members"],
    diameters: Float[Array, "members"],
    structure: Structure,
    graph: EquilibriumStructure,
    model: Model,
    steel: SteelGrade,
    tube: Tube,
    loads: Float[Array, "cases nodes 3"],
    beta: float | Float[Array, ""],
    *,
    plastic: bool,
    resultant: bool = True,
    l_cr: Float[Array, "members"] | None = None,
) -> Envelope:
    """
    Form-find once, analyse every load case, and size for the worst of them.

    Parameters
    ----------
    q :
        Force density of every member. Negative in compression.
    diameters :
        Diameters the frame is analysed with, being the previous outer iterate
        of the check. They set the stiffness, not the resistance.
    structure :
        The structure supplying the connectivity, the supports and the loads it
        is form-found under.
    graph :
        The form-finding connectivity, from `normax.formfinding.graph`.
    model :
        The prepared analysis model, from `normax.analysis.smax.prepare`.
    steel :
        Material properties and partial factors.
    tube :
        The section family every member is drawn from.
    loads :
        Force applied at every node in every load case.
    beta :
        Sharpness of the envelope. The design approaches the smallest adequate
        one from above as it grows.
    plastic :
        Whether the section is Class 1 or 2. Static, never a traced value.
    resultant :
        Whether the two moments combine as a resultant in the cross-section
        check, or as a linear sum.
    l_cr :
        Buckling length of every member. If None, each member buckles over its
        own length.

    Returns
    -------
    envelope :
        The geometry, the actions under every case, the sizes and the mass.

    Notes
    -----
    **The objective an optimizer sees.** A member has one size and has to
    satisfy every case, so its size is the largest any case demands. That
    largest is not differentiable, and a gradient taken through it sees one case
    per step and stalls; the smooth envelope of `normax.ec3.sizing` replaces it,
    taken in the logarithm of the diameter so the sharpness is dimensionless.

    The envelope never understates the true largest, so **the design is adequate
    at every sharpness** and annealing drives it onto the smallest adequate one
    from above. What it gives away is bounded by the number of cases raised to
    the reciprocal of the sharpness, in diameter.

    Form finding runs once, under the structure's own loads. The shape answers
    to that case alone, which is what makes it funicular; the others are
    departures from it, and the bending they raise is the reason the analysis
    stage is in the pipeline at all.

    Cases are looped over rather than mapped, the analysis stage not being
    traceable through `vmap`, so the cost is linear in their number.
    """
    state = equilibrium(q, structure, graph)
    lengths = state.lengths[:, 0]
    buckling = lengths if l_cr is None else l_cr

    acting = [
        actions(model, state.xyz, diameters, steel, tube, loads=case) for case in loads
    ]

    required = jnp.stack(
        [
            diameter_ec3(
                case, buckling, steel, tube, plastic=plastic, resultant=resultant
            )
            for case in acting
        ]
    )

    covering = envelope_ec3(required, beta)

    used = jnp.stack(
        [
            utilization_ec3(
                covering,
                case,
                buckling,
                steel,
                tube,
                plastic=plastic,
                resultant=resultant,
            )
            for case in acting
        ]
    )

    # Stacked field by field rather than as a MemberActions, whose fields carry
    # one member axis and not the case axis these gain.
    n_ed, m_y_ed, m_z_ed, c_my, c_mz = (jnp.stack(field) for field in zip(*acting))

    return Envelope(
        xyz=state.xyz,
        lengths=lengths,
        l_cr=buckling,
        n_ed=n_ed,
        m_y_ed=m_y_ed,
        m_z_ed=m_z_ed,
        c_my=c_my,
        c_mz=c_mz,
        required=required,
        diameters=covering,
        utilization=used,
        mass=mass_ec3(covering, lengths, steel, tube),
    )


class Unsmoothed(NamedTuple):
    """
    The design the envelope is an upper bound on, taken at the true largest.

    Attributes
    ----------
    diameters :
        Largest diameter any case demands of each member.
    utilization :
        Demand over resistance of every member under every case, at those
        diameters. Exactly one for the case that governs each member.
    mass :
        Total mass of the members.

    Notes
    -----
    What the design would be with no smoothing at all, and so the number to
    report rather than the annealed one. It is not differentiable in any useful
    way, the largest of a set having a gradient that sees one case at a time,
    which is the whole reason the envelope exists.
    """

    diameters: Float[Array, "members"]
    utilization: Float[Array, "cases members"]
    mass: Float[Array, ""]


def unsmoothed(
    result: Envelope,
    steel: SteelGrade,
    tube: Tube,
    *,
    plastic: bool,
    resultant: bool = True,
) -> Unsmoothed:
    """
    Re-check an enveloped design at the smallest sizes that satisfy every case.

    Parameters
    ----------
    result :
        An enveloped design, from `envelope`.
    steel :
        Material properties and partial factors.
    tube :
        The section family every member is drawn from.
    plastic :
        Whether the section is Class 1 or 2. Static, never a traced value.
    resultant :
        Whether the two moments combine as a resultant in the cross-section
        check, or as a linear sum.

    Returns
    -------
    unsmoothed :
        The sizes, the utilization under every case, and the mass.

    Notes
    -----
    **What the smoothing cost, measured rather than bounded.** The envelope
    never understates what a case demands, so this is always the lighter design
    and the difference is what annealing is driving to zero.

    Nothing is analysed again. The actions belong to the geometry and the
    diameters the frame was analysed with, neither of which changes here, so
    only the check is repeated.
    """
    sizes = jnp.max(result.required, axis=0)

    used = jnp.stack(
        [
            utilization_ec3(
                sizes,
                MemberActions(
                    result.n_ed[case],
                    result.m_y_ed[case],
                    result.m_z_ed[case],
                    result.c_my[case],
                    result.c_mz[case],
                ),
                result.l_cr,
                steel,
                tube,
                plastic=plastic,
                resultant=resultant,
            )
            for case in range(result.required.shape[0])
        ]
    )

    return Unsmoothed(
        diameters=sizes,
        utilization=used,
        mass=mass_ec3(sizes, result.lengths, steel, tube),
    )


def governing_case(result: Envelope) -> Int[Array, "members"]:
    """
    Which load case decided each member's size.

    Parameters
    ----------
    result :
        An enveloped design, from `envelope`.

    Returns
    -------
    governing_case :
        Index of the case working each member hardest.

    Notes
    -----
    **Non-differentiable**, and kept out of `Envelope` for that reason.

    The picture only a differentiable code check can produce: as the form
    changes, the pattern of which case governs where reorganises, and it does so
    because the shape decides how much bending each case raises rather than
    because any member was reassigned.
    """
    return jnp.argmax(result.utilization, axis=0)


def governing(
    result: Design,
    steel: SteelGrade,
    tube: Tube,
    *,
    plastic: bool,
    resultant: bool = True,
) -> Float[Array, "members"]:
    """
    Which limit state decided each member's size.

    Parameters
    ----------
    result :
        A design, from `design`.
    steel :
        Material properties and partial factors.
    tube :
        The section family every member is drawn from.
    plastic :
        Whether the section is Class 1 or 2. Static, never a traced value.
    resultant :
        Whether the two moments combine as a resultant in the cross-section
        check, or as a linear sum.

    Returns
    -------
    governing :
        One of the limit-state codes of `normax.ec3.sizing`.

    Notes
    -----
    **Non-differentiable**, and kept out of `Design` for that reason. Read it
    beside a design, never through one.

    The design is re-checked against the actions it was sized for, which is what
    makes the answer the one that decided the size rather than a second opinion
    on a subset of them.
    """
    return governing_ec3(
        result.diameters,
        result.actions,
        result.l_cr,
        steel,
        tube,
        plastic=plastic,
        resultant=resultant,
    )


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
    l_cr_global :
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
    l_cr_global: Float[Array, "members"]


def stability(
    result: Design,
    model: Model,
    steel: SteelGrade,
    tube: Tube,
    *,
    num_modes: int = 1,
    threshold: float = ALPHA_CR_ELASTIC,
    loads: Float[Array, "nodes 3"] | None = None,
) -> Stability:
    """
    Check a finished design against the stability of the frame it sits in.

    Parameters
    ----------
    result :
        A design, from `design`.
    model :
        The prepared analysis model, from `normax.analysis.smax.prepare`.
    steel :
        Material properties and partial factors.
    tube :
        The section family every member is drawn from.
    num_modes :
        Number of critical load factors to return. Static.
    threshold :
        Factor the frame must reach for first-order analysis to be adequate.
    loads :
        Load case the frame buckles under. If None, the structure's own loads.

    Returns
    -------
    stability :
        The critical load factors, the verdict, and both routes to slenderness.

    Notes
    -----
    **Soft validation, deliberately outside the pipeline.** Nothing here feeds
    `design` or `mass`, and no critical load factor crosses a Tesseract boundary.
    Global stability is not covered by what this package designs; this only says
    how far the buckling length that produced a design can be trusted.

    It could not enter the sizing map in any case. That roots a member check,
    which is local and monotone in one diameter, while this is a property of the
    whole frame: a design failing here is not made to pass by growing one member,
    and the remedy is bracing or a different buckling length.

    **Never differentiated.** The eigenproblem would trace, but an eigenvalue
    derivative is undefined where two modes cross — and they do, since a
    symmetric structure has degenerate pairs.

    Members carrying no axial force report nan for both slendernesses and for the
    equivalent buckling length, a factor scaling the whole load having nothing to
    say about a member the load never reaches.
    """
    modes = buckling(
        model,
        result.xyz,
        result.diameters,
        steel,
        tube,
        num_modes=num_modes,
        loads=loads,
    )
    alpha_cr = modes.factors[0]

    gross = area(result.diameters, tube.ratio)
    inertia = second_moment(result.diameters, tube.ratio)

    return Stability(
        factors=modes.factors,
        utilization=utilization_stability(alpha_cr, threshold),
        adequate=is_adequate(alpha_cr, threshold),
        slenderness_member=slenderness_from_force(
            gross, steel, force_critical(inertia, result.l_cr, steel)
        ),
        slenderness_global=slenderness_global(
            amplifier_resistance(gross, steel, result.actions.n_ed), alpha_cr
        ),
        l_cr_global=buckling_length_global(
            alpha_cr, result.actions.n_ed, inertia, steel
        ),
    )
