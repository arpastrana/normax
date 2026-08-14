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
What a design is, what each block contributes to one, and how the three compose.

A design is found by three blocks in a row: a form finder chooses the shape, a
frame analysis says what the members carry, and a code check says how big they
have to be. Each is written by different people, in a different language, and
differentiates in a different way. This module says what the three agree on and
nothing whatsoever about how any of them computes, then composes them.

**A block is built from a structure and then called, and the split is where the
topology lives.** Its constructor reads a structure and settles whatever that
particular software can settle from the connectivity, the supports and their
positions alone: a form finder wants connectivity matrices, a frame solver wants
an assembly and degree of freedom maps, a code check wants nothing at all. That
runs on the host, outside any traced call, and it may do arbitrary work —
compile an assembly, open a session against a service. What is left is a
function of design parameters and loads, which is what an optimizer calls.

That split is what keeps a topology out of an objective. A built block is a pure
function of things no optimizer varies, so rebuilding it per iterate is waste,
and for a traced backend it is worse than waste: building reads support flags in
Python, so doing it inside the trace is what stops a stage being jitted.

**Load cases arrive stacked, and a block decides what to do with them.** A
structure is form-found under one load case and checked against several, so only
the analysis and the check carry a load case axis. Handing a block all of them at
once lets a solver that can answer them together do so in one crossing, which is
what a Tesseract or a service reached over a network wants; a solver that cannot
loops internally and pays what it would have paid anyway.

Nothing here says a block must be traceable. A block may trace, may carry a
hand-written rule through `custom_vjp`, or may serialize its arguments and cross
a network — the contract is a signature and a derivative, not an implementation.

**What is asked of a design is not part of one.** A mass is `ρ Σ A L`, and which
load case governs each member is an `argmax` — arithmetic over the fields below
rather than anything a stage produced, which is why both are functions here and
neither is a field. Reconciling the load cases lives further out still, in
`design_envelope`, and whether the frame is stable at all is
`normax.stability`.
"""

from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from normax.analysis import AbstractFrameAnalyzer
from normax.analysis import MemberForces
from normax.ec3.section import Tube
from normax.ec3.sizing import diameter_envelope
from normax.form_finding import AbstractFormFinder
from normax.form_finding import FormFoundShape
from normax.loads import LoadCases
from normax.loads import count_load_cases
from normax.sizing import AbstractMemberSizer
from normax.sizing import MemberSizes


class DesignParameters(NamedTuple):
    """
    The quantities that vary between evaluations of the pipeline.

    Attributes
    ----------
    force_densities :
        Force density of every member. Negative in compression.
    diameters :
        Outer diameter every member is analyzed with.

    Notes
    -----
    The diameters set the stiffness the frame is analyzed with and not the
    resistance it is checked against, which the sizing block returns. They are
    therefore the previous outer iterate of a staggered coupling rather than a
    design variable in their own right, and an optimizer that varies the shape
    alone still has to supply them.

    **A sharpness is not here, and that is deliberate.** Reconciling load cases
    is smoothing rather than a stage, so it happens above the pipeline in
    `design_envelope` and its sharpness is an argument of the loss
    rather than a parameter of the design.
    """

    force_densities: Float[Array, "members"]
    diameters: Float[Array, "members"]


class Design(NamedTuple):
    """
    One structure carried through all three stages.

    Attributes
    ----------
    shape :
        The geometry form finding settled on, and its member lengths.
    forces :
        What every member carries under every load case.
    sizes :
        The sections the check demands, and the actions it read.

    Notes
    -----
    One field per stage, in the order they ran, and nothing that no stage
    produced. A mass is not here because it is `ρ Σ A L` — arithmetic over two
    of these fields, which `compute_mass` does — and neither is
    a utilization, which is the standard asked a second question at a size it
    did not choose.

    **The load case axis lives in `forces` and `sizes` and not in `shape`.** A
    structure is form-found once and checked against several cases, so the
    geometry is shared and the sections are not — until `design_envelope`
    reconciles them, which returns this same container with one section per
    member.
    """

    shape: FormFoundShape
    forces: MemberForces
    sizes: MemberSizes


class StructuralDesignPipeline(eqx.Module):
    """
    Form finding, analysis and a code check, composed into one function.

    Attributes
    ----------
    formfinder :
        The block that chooses the shape.
    analyzer :
        The block that says what the members carry.
    sizer :
        The block that says how big they have to be.

    Notes
    -----
    Each block differentiates in its own way — a traced linear solve, a traced
    assembly and solve, and an implicit tangent taken at the root of a residual —
    and the composition hides that from the caller: a design comes back with a
    gradient in the force densities. Nothing here asks how any block computes,
    or in what language, or on which side of a network boundary, which is what
    makes one of them replaceable without touching the other two.

    **Every backend sees the structure in its own terms, and it does so when it
    is built rather than here.** A pipeline is three blocks already bound to one
    structure, so calling one is a pure function of things an optimizer varies
    and nothing that reads a support flag in Python survives into the trace.

    **No topology reaches here, and nothing is decided here.** A pipeline is
    three blocks and nothing else. It holds no connectivity, so a member length
    comes off the shape a form finder measured; it reconciles no load case, that
    being smoothing rather than a stage; and it computes no mass. What comes
    back is one field per stage and nothing that no stage produced.

    **The coupling between the analysis and the check is staggered.** A frame
    cannot be analyzed without sections, and the sections are what the check
    returns, so the diameters the frame is built from are an input and the
    diameters the check requires are an output. One pass is taken, not a fixed
    point.
    """

    formfinder: AbstractFormFinder
    analyzer: AbstractFrameAnalyzer
    sizer: AbstractMemberSizer

    def __call__(
        self,
        params: DesignParameters,
        loads: LoadCases,
    ) -> Design:
        """
        Form-find once, analyze every load case, and size for each of them.

        Parameters
        ----------
        params :
            Force densities, and the diameters the frame is analyzed with.
        loads :
            The load case the shape answers to, and the ones it is checked
            against.

        Returns
        -------
        design :
            The shape, what the members carry, and what the check demands of
            them under every load case on its own.

        Notes
        -----
        **Form finding runs once and the other two run for every load case.**
        The shape answers to one load case by construction, that being what
        makes it funicular; choosing the shape again for each of them would mean
        a different structure per load case rather than one structure checked
        against several.

        **Nothing is reconciled here.** Each load case gets the section it
        demands on its own, and a member has one size in the end;
        `design_envelope` is what collapses the two, and it is
        smoothing rather than a clause.

        **Every member is assumed to buckle over its own length, and nothing
        here can say otherwise.** That is a strong assumption rather than a
        conservative one: it presumes every node is held in position by
        structure outside the model, and where that does not hold the frame
        buckles in a mode spanning many members and the assumption is unsafe.
        `frame_stability` measures the gap, by recovering the buckling length a
        critical load factor is equivalent to.

        The clauses below take a buckling length as an argument and always
        will — EN 1993-1-1 Eq. 6.50 is written in `L_cr`, not in a member
        length. What is fixed is this composition's choice of what to pass, and
        that choice is temporary; see `docs/clauses.md`.
        """
        shape = self.formfinder(params.force_densities, loads.formfinding)
        forces = self.analyzer(shape.xyz, params.diameters, loads.analysis)
        sizes = self.sizer(forces, shape.lengths)

        return Design(shape, forces, sizes)


def governing_load_case(
    diameters: Float[Array, "load_cases members"],
) -> Int[Array, "members"]:
    """
    Which load case decided each member's size.

    Parameters
    ----------
    diameters :
        Outer diameter every load case demands of every member on its own,
        before they are reconciled.

    Returns
    -------
    governing_load_case :
        Index of the load case working each member hardest.

    Notes
    -----
    **Read from the sizes demanded rather than from a utilization**, which costs
    an `argmax` instead of re-reading the standard. The two agree because
    capacity is strictly increasing in the diameter: at the reconciled section
    the case that demanded the largest one is exactly satisfied and every other
    case is at a section larger than it asked for, so its utilization is
    strictly below one. Exact at the true largest, and measured to agree at the
    sharpnesses this package anneals over.

    **Non-differentiable**, and absent from any design for that reason.

    The picture only a differentiable code check can produce: as the form
    changes, the pattern of which load case governs where reorganizes, and it
    does so because the shape decides how much bending each one raises rather
    than because any member was reassigned.
    """
    return jnp.argmax(diameters, axis=0)


def compute_mass(design: Design) -> Float[Array, ""]:
    """
    Total mass of a design.

    Parameters
    ----------
    design :
        A design whose sections have been reconciled to one per member.

    Returns
    -------
    mass :
        Total mass of the members.

    Notes
    -----
    **The objective the whole pipeline exists to serve, and the scalar
    `jax.grad` is taken of.** One reverse pass crosses all three blocks, so its
    cost does not grow with the number of force densities.

    `ρ Σ A L`, and every term of it is already on the design: the length from
    the shape a form finder settled, the area and the density from the section a
    check chose and the material it is cut from. Nothing here is a clause, which
    is why no block computes it and why this needs no block to.

    Called on a design whose load case axis has been collapsed. On one that
    still carries it, this is the mass each load case would need on its own.
    """
    sections = design.sizes.sections
    per_length = sections.material.density * sections.area

    return jnp.sum(per_length * design.shape.lengths)


def design_envelope(
    design: Design,
    sharpness: float | Float[Array, ""] | None = None,
) -> Design:
    """
    Reconcile the load cases into one section per member.

    Parameters
    ----------
    design :
        A design whose sections carry a load case axis.
    sharpness :
        Sharpness of the envelope. If None, the true largest section any load
        case demands.

    Returns
    -------
    design :
        The same design with one section per member and the axis collapsed.

    Notes
    -----
    **Smoothing rather than a clause, which is why it is a function here and not
    a method of the check.** A member has one size and has to satisfy every load
    case, so its size is the largest any of them demands; that largest is not
    differentiable, and a gradient taken through it sees one load case per step
    and stalls. The smooth envelope never understates it, so the design stays
    adequate at every sharpness and annealing drives it onto the smallest
    adequate one from above.

    **The geometry is enveloped and the grade and the class ride through
    untouched.** The envelope is scale-equivariant — taken in the logarithm, a
    constant factor passes straight through it — so enveloping a diameter and a
    thickness that is that diameter over a fixed ratio preserves the ratio, and
    no catalogue is needed to re-derive a wall. What a tube is made of has no
    load case axis to reduce, which is why the two geometric fields are named
    here rather than mapped over.

    **Nothing here reads a standard, and nothing needs to.** `utilization`
    passes through untouched because it belongs to the per-case sections and
    still describes them; what the reconciled section is worth is a different
    question, and `AbstractMemberSizer.utilization` is what answers it for a
    report. Which case governs each member needs neither — see
    `governing_load_case`.

    A single load case is returned as it stands, whatever the sharpness. The
    envelope over one case is the identity in exact arithmetic and a logarithm
    followed by an exponential in floating point, so taking it would cost the
    last bits of a size for nothing.
    """
    demanded = design.sizes.sections
    cases = count_load_cases(demanded.diameter)

    def cover_cases(
        field: Float[Array, "load_cases members"],
    ) -> Float[Array, "members"]:
        if cases == 1:
            return field[0]
        if sharpness is None:
            return jnp.max(field, axis=0)

        return diameter_envelope(field, sharpness)

    diameters = cover_cases(demanded.diameter)
    thicknesses = cover_cases(demanded.thickness)
    covering = Tube(diameters, thicknesses, demanded.material, demanded.section_class)
    sizes = MemberSizes(covering, design.sizes.actions, design.sizes.utilization)

    return Design(design.shape, design.forces, sizes)
