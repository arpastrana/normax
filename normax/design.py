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
`normax.stability`. Searching for a design is asked of one too, which is what
`optimize_staggered` does: closing the coupling between the analysis and the
check needs a diameter read off a design, and `normax.optimization` knows what a
descent is without knowing what a design is.
"""

from collections.abc import Callable
from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
from ec3x.section import Tube
from jax.scipy.special import logsumexp
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from normax.analysis import AbstractFrameAnalyzer
from normax.analysis import MemberForces
from normax.form_finding import AbstractFormFinder
from normax.form_finding import FormFoundShape
from normax.loads import LoadCases
from normax.loads import count_load_cases
from normax.optimization import ObjectiveValue
from normax.optimization import SearchResult
from normax.optimization import Trajectory
from normax.optimization import minimize_bounded
from normax.optimization import value_and_gradient
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
        The sections the check demands, and how hard each one is worked.

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


def diameter_envelope(
    diameters: Float[Array, "load_cases members"],
    beta: float | Float[Array, ""],
) -> Float[Array, "members"]:
    """
    Smooth envelope of a member's size over several load cases.

    Parameters
    ----------
    diameters :
        Diameter required by each load case, one row per case.
    beta :
        Sharpness. The envelope approaches the true largest as it grows.

    Returns
    -------
    diameter :
        Diameter covering every load case.

    Notes
    -----
    Not EN 1993-1-1, or any standard. A member must satisfy every load case, so
    its size is the largest any load case demands; that largest is not
    differentiable, and a gradient taken through it sees one load case at a
    time and stalls. Reconciling the cases is smoothing rather than a clause,
    which is why it lives here, above the blocks that implement standards.

    The envelope is taken in the logarithm of the diameter, which makes the
    sharpness dimensionless and so comparable between structures of different
    size. It never understates the largest, and exceeds it by at most the
    logarithm of the number of load cases over the sharpness, so annealing the
    sharpness upward drives it onto the true largest from above. Being an upper
    bound is the safe direction: the design stays adequate throughout.
    """
    logarithms = jnp.log(diameters)
    smoothed = logsumexp(beta * logarithms, axis=0) / beta

    return jnp.exp(smoothed)


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
    sizes = MemberSizes(covering, design.sizes.utilization)

    return Design(design.shape, design.forces, sizes)


def settle_diameters(
    objective: Callable[[DesignParameters], ObjectiveValue],
    params: DesignParameters,
    *,
    settling_passes: int = 400,
    settling_tolerance: float = 1e-6,
) -> Float[Array, "members"]:
    """
    The diameters an analysis at these force densities asks of itself.

    Parameters
    ----------
    objective :
        The mass, as a function of a whole set of design parameters, returning
        the design it weighed alongside. Its sections must be reconciled to one
        per member, which is what `design_envelope` does.
    params :
        Force densities to hold, and the diameters to start the analysis at.
    settling_passes :
        Most analyses to spend before the coupling is called stalled.
    settling_tolerance :
        Largest fractional movement in any diameter that counts as settled.

    Returns
    -------
    settled :
        Diameters the analysis and the check agree on, at these force densities.

    Raises
    ------
    ValueError
        If the diameters are still moving when the passes run out.

    Notes
    -----
    **The analysis needs sections and the check is what returns them**, so a
    design read off one pass was analyzed at sections it does not have. Repeating
    the pass at the sections just demanded closes that: sizing is a contraction in
    the diameters, and its fixed point is a structure analyzed at its own sections.

    **Forward passes rather than another descent**, because the contraction is slow
    near its fixed point and a pass of it costs one forward evaluation where a
    descent costs many gradients. Nothing here moves the force densities.

    A raise rather than a returned residual, so that what comes back needs no
    checking. Passes exhausted means the sizes and the stiffnesses are chasing each
    other, which no tolerance on the answer would make untrue.

    The starting diameters are restated at their own dtype, which is what makes the
    second pass the same program as the first. A seed written as
    `jnp.full(members, 100.0)` is weakly typed where the sizes a check returns are
    not, and the two are different abstract values however equal their contents:
    left alone they compile the pipeline twice.
    """
    weighed = eqx.filter_jit(objective)
    assumed = jnp.asarray(params.diameters, dtype=params.diameters.dtype)
    moved = float("inf")

    for _ in range(settling_passes):
        _, design = weighed(DesignParameters(params.force_densities, assumed))
        demanded = design.sizes.sections.diameter
        moved = float(jnp.max(jnp.abs(demanded / assumed - 1.0)))
        assumed = demanded

        if moved < settling_tolerance:
            return demanded

    raise ValueError(
        f"diameters still moving by {moved:.3e} after {settling_passes} "
        f"passes at fixed force densities, above {settling_tolerance:.3e}"
    )


def optimize_staggered(
    objective: Callable[[DesignParameters], ObjectiveValue],
    params: DesignParameters,
    *,
    bounds: tuple[float, float],
    iterations: int = 50,
    rounds: int = 12,
    settling_passes: int = 400,
    settling_tolerance: float = 1e-6,
) -> SearchResult:
    """
    Minimize in the force densities, refreshing the analysis diameters per round.

    Parameters
    ----------
    objective :
        The mass, as a function of a whole set of design parameters, returning
        the design it weighed alongside. Its sections must be reconciled to one
        per member, which is what `design_envelope` does.
    params :
        Force densities to start from, and the diameters the first round is
        analyzed with.
    bounds :
        Smallest and largest value any force density may take.
    iterations :
        Most iterations to spend in each round.
    rounds :
        Most descents to spend before the coupling is called stalled.
    settling_passes :
        Most analyses one round may spend closing the coupling at fixed force
        densities.
    settling_tolerance :
        Largest fractional movement in any diameter that counts as settled, both
        of a settling pass and of a whole round.

    Returns
    -------
    found :
        The last round's answer, the design behind it, and every iterate of
        every round.

    Raises
    ------
    ValueError
        If the coupling has not closed, either within a round's settling passes or
        within the round cap. The mass is then one computed at diameters the design
        does not have, so it is not returned.

    Notes
    -----
    **A round exists because the frame cannot be analyzed without sections and
    the sections are what the check returns.** A whole descent runs at one set of
    diameters, the coupling is then closed where that descent stopped, and the
    structure is redesigned from there at the sections it settled on. What
    converges is the coupling rather than the search: on the round that settling
    no longer moves the diameters, the mass reported is that of a design analyzed
    at its own sections.

    **Refreshing per round rather than per evaluation is what keeps a quasi-Newton
    search usable.** A line search compares values of one function, and a seed
    that moves inside it hands the search a different function at every trial; the
    curvature accumulated from differences of those gradients is contaminated the
    same way. A refresh between descents leaves each of them a fixed function.

    **The coupling is settled by `settle_diameters` and not by another descent.**
    Sizing is a contraction in the diameters, but a slow one near its fixed point,
    so buying one pass of it per descent spends gradients on something a forward
    evaluation resolves. Settling inside the round leaves the round count measuring
    what it should: how many times the search had to be redone because the sections
    it assumed had changed.

    **Every round shares the same two compiled programs.** The diameters are an
    argument of both rather than a constant captured in them, so a round is the
    same program as its neighbour at different values and the compilations are paid
    once. Capturing them instead recompiles all three blocks per round, which costs
    more than the search does.

    A raise rather than a returned residual, so that what comes back needs no
    checking. A cap reached means the sizes and the stiffnesses are chasing each
    other, which no tolerance on the answer would make untrue. Both caps belong to
    the caller for that reason: they are the budget a problem is given, not a
    property of the coupling.

    **The sharpness recorded against every iterate is zero**, that being what
    `minimize_bounded` stamps when a caller has none to give. A sharpness belongs
    to the objective's closure here rather than to the schedule, so the rounds of
    a staggered search are not told apart by it the way an annealed one's are.
    """
    iterates = []
    masses = []
    sharpnesses = []
    residual = float("inf")

    seed_diameters = jnp.asarray(params.diameters, dtype=params.diameters.dtype)
    current = DesignParameters(params.force_densities, seed_diameters)

    def seeded_objective(
        force_densities: Float[Array, "members"],
        diameters: Float[Array, "members"],
    ) -> ObjectiveValue:
        seeded = DesignParameters(force_densities, diameters)

        return objective(seeded)

    compiled = value_and_gradient(seeded_objective, has_aux=True)

    for _ in range(rounds):
        held = current.diameters
        found = minimize_bounded(
            lambda x, seed=held: seeded_objective(x, seed),
            current.force_densities,
            bounds=bounds,
            iterations=iterations,
            has_aux=True,
            gradient=lambda x, seed=held: compiled(x, seed),
        )
        walked = found.trajectory
        iterates.append(walked.q)
        masses.append(walked.mass)
        sharpnesses.append(walked.beta)

        answer = walked.q[-1]
        settled = settle_diameters(
            objective,
            DesignParameters(answer, held),
            settling_passes=settling_passes,
            settling_tolerance=settling_tolerance,
        )
        current = DesignParameters(answer, settled)
        residual = float(jnp.max(jnp.abs(settled / held - 1.0)))

        if residual < settling_tolerance:
            break
    else:
        raise ValueError(
            f"diameters still moving by {residual:.3e} after "
            f"{rounds} rounds, above {settling_tolerance:.3e}"
        )

    trajectory = Trajectory(
        q=jnp.concatenate(iterates),
        mass=jnp.concatenate(masses),
        beta=jnp.concatenate(sharpnesses),
    )

    return SearchResult(found.value, found.aux, trajectory)
