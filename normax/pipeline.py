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

from jax_fdm.equilibrium import EquilibriumStructure
from jaxtyping import Array
from jaxtyping import Bool
from jaxtyping import Float

from normax.analysis import buckling
from normax.analysis import forces
from normax.ec3.resistance import n_cr
from normax.ec3.resistance import slenderness
from normax.ec3.section import area
from normax.ec3.section import second_moment
from normax.ec3.sizing import Steel
from normax.ec3.sizing import Tube
from normax.ec3.sizing import diameter as diameter_ec3
from normax.ec3.sizing import end_moments
from normax.ec3.sizing import governing as governing_ec3
from normax.ec3.sizing import mass as mass_ec3
from normax.ec3.sizing import utilization as utilization_ec3
from normax.ec3.stability import ALPHA_CR_ELASTIC
from normax.ec3.stability import buckling_length as buckling_length_global
from normax.ec3.stability import is_adequate
from normax.ec3.stability import resistance_factor
from normax.ec3.stability import slenderness_global
from normax.ec3.stability import utilization as utilization_stability
from normax.formfinding import equilibrium
from normax.structures import Structure


class Design(NamedTuple):
    """
    Everything the three stages produce for one set of force densities.

    Attributes
    ----------
    xyz :
        Position of every node at equilibrium.
    lengths :
        Length of every member.
    n_ed :
        Axial force of every member, tension positive.
    m_ed :
        Larger end moment of every member, in magnitude.
    c_m :
        Equivalent uniform moment factor of every member.
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
    """

    xyz: Float[Array, "nodes 3"]
    lengths: Float[Array, "members"]
    n_ed: Float[Array, "members"]
    m_ed: Float[Array, "members"]
    c_m: Float[Array, "members"]
    l_cr: Float[Array, "members"]
    diameters: Float[Array, "members"]
    utilization: Float[Array, "members"]
    mass: Float[Array, ""]


def design(
    q: Float[Array, "members"],
    diameters: Float[Array, "members"],
    structure: Structure,
    graph: EquilibriumStructure,
    steel: Steel,
    tube: Tube,
    *,
    normal: int | None,
    plastic: bool,
    resultant: bool = True,
    l_cr: Float[Array, "members"] | None = None,
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
    steel :
        Material properties and partial factors.
    tube :
        The section family every member is drawn from.
    normal :
        Index of the global axis a planar structure has no thickness along, or
        None for a structure that occupies all three dimensions.
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
    design :
        The geometry, the member actions, the required sizes and the mass.

    Notes
    -----
    Differentiable in the force densities, in the analysed diameters, and in
    every material property. The cross-section class selects a clause rather
    than scaling a number, so it stays a static Python value throughout.

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

    member = forces(
        structure,
        state.xyz,
        diameters,
        steel,
        tube,
        normal=normal,
    )

    m_ed, c_m = end_moments(member.m_y_ed[:, 0], member.m_y_ed[:, 1])
    m_minor, c_minor = end_moments(member.m_z_ed[:, 0], member.m_z_ed[:, 1])

    buckling = lengths if l_cr is None else l_cr

    required = diameter_ec3(
        member.n_ed,
        m_ed,
        m_minor,
        c_m,
        c_minor,
        buckling,
        steel,
        tube,
        plastic=plastic,
        resultant=resultant,
    )

    used = utilization_ec3(
        required,
        member.n_ed,
        m_ed,
        m_minor,
        c_m,
        c_minor,
        buckling,
        steel,
        tube,
        plastic=plastic,
        resultant=resultant,
    )

    return Design(
        xyz=state.xyz,
        lengths=lengths,
        n_ed=member.n_ed,
        m_ed=m_ed,
        c_m=c_m,
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
    steel: Steel,
    tube: Tube,
    *,
    normal: int | None,
    plastic: bool,
    resultant: bool = True,
    l_cr: Float[Array, "members"] | None = None,
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
    steel :
        Material properties and partial factors.
    tube :
        The section family every member is drawn from.
    normal :
        Index of the global axis a planar structure has no thickness along, or
        None for a structure that occupies all three dimensions.
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
    mass :
        Total mass.

    Notes
    -----
    The objective the whole pipeline exists to serve, and the scalar `jax.grad`
    is taken of. One reverse pass crosses all three stages, so its cost does not
    grow with the number of force densities.
    """
    return design(
        q,
        diameters,
        structure,
        graph,
        steel,
        tube,
        normal=normal,
        plastic=plastic,
        resultant=resultant,
        l_cr=l_cr,
    ).mass


def governing(
    result: Design,
    steel: Steel,
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
    """
    zeros = result.m_ed * 0.0

    return governing_ec3(
        result.diameters,
        result.n_ed,
        result.m_ed,
        zeros,
        result.c_m,
        zeros + 1.0,
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
        Eq. 6.64.
    l_cr_global :
        Buckling length the critical load factor is equivalent to.

    Notes
    -----
    The two slendernesses are the same equation given different questions. Their
    ratio is the size of the assumption a member buckling length makes, and it is
    one wherever that assumption is right.

    ⚠️ The clauses behind `utilization` and `adequate` are unverified — see
    `docs/clauses.md` open item 0f.
    """

    factors: Float[Array, "modes"]
    utilization: Float[Array, ""]
    adequate: Bool[Array, ""]
    slenderness_member: Float[Array, "members"]
    slenderness_global: Float[Array, "members"]
    l_cr_global: Float[Array, "members"]


def stability(
    result: Design,
    structure: Structure,
    steel: Steel,
    tube: Tube,
    *,
    normal: int | None,
    num_modes: int = 1,
    threshold: float = ALPHA_CR_ELASTIC,
) -> Stability:
    """
    Check a finished design against the stability of the frame it sits in.

    Parameters
    ----------
    result :
        A design, from `design`.
    structure :
        The structure supplying the connectivity, the supports and the loads.
    steel :
        Material properties and partial factors.
    tube :
        The section family every member is drawn from.
    normal :
        Index of the global axis a planar structure has no thickness along, or
        None for a structure that occupies all three dimensions.
    num_modes :
        Number of critical load factors to return. Static.
    threshold :
        Factor the frame must reach for first-order analysis to be adequate.

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
        structure,
        result.xyz,
        result.diameters,
        steel,
        tube,
        normal=normal,
        num_modes=num_modes,
    )
    alpha_cr = modes.factors[0]

    gross = area(result.diameters, tube.ratio)
    inertia = second_moment(result.diameters, tube.ratio)

    return Stability(
        factors=modes.factors,
        utilization=utilization_stability(alpha_cr, threshold),
        adequate=is_adequate(alpha_cr, threshold),
        slenderness_member=slenderness(
            gross, steel.f_y, n_cr(inertia, result.l_cr, steel.e_mod)
        ),
        slenderness_global=slenderness_global(
            resistance_factor(gross, steel.f_y, result.n_ed), alpha_cr
        ),
        l_cr_global=buckling_length_global(alpha_cr, result.n_ed, inertia, steel.e_mod),
    )
