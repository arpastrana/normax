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

Nothing here knows how any of the three computes, or in what language, or on
which side of a network boundary. `normax.design` states what a block is,
`normax.form_finding`, `normax.analysis.smax` and `normax.sizing` hold three that
compute in this process, and `normax.tesseract` holds three that do not.
Composing the second set is the same call as composing the first, and
`tests/test_tesseract_parity.py` is where that is measured rather than claimed.

**The coupling between the analysis and the check is staggered.** A frame cannot
be analyzed without sections, and the sections are what the check returns, so
the diameters the frame is built from are an input here and the diameters the
check requires are an output. One pass is taken, not a fixed point. On a
funicular arch the two are close because the analysis barely depends on the
section — the internal forces move as the square of the diameter over the
length — but the gap is real and `experiments/09_arch_pipeline_jax.py` measures
how fast repeating the pass closes it.
"""

from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Bool
from jaxtyping import Float
from jaxtyping import Int

from normax.analysis.smax import SmaxAnalyzer
from normax.analysis.smax import buckling_modes
from normax.design import AbstractFormFinder
from normax.design import AbstractFrameAnalyzer
from normax.design import AbstractMemberSizer
from normax.design import Design
from normax.design import DesignParameters
from normax.ec3.resistance import force_critical
from normax.ec3.resistance import slenderness_from_force
from normax.ec3.stability import ALPHA_CR_ELASTIC
from normax.ec3.stability import amplifier_resistance
from normax.ec3.stability import buckling_length_global
from normax.ec3.stability import is_adequate
from normax.ec3.stability import slenderness_global
from normax.ec3.stability import utilization_frame as utilization_stability
from normax.loads import LoadCases


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
    analyzer: SmaxAnalyzer,
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
    modes = buckling_modes(
        analyzer.model,
        design.shape.xyz,
        design.sizes.sections.diameters,
        analyzer.steel,
        analyzer.catalogue,
        loads,
        num_modes=num_modes,
    )
    alpha_cr = modes.factors[0]

    steel = analyzer.steel
    tubes = analyzer.catalogue.tube_at(design.sizes.sections.diameters)
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


def governing_load_case(
    utilization: Float[Array, "load_cases members"],
) -> Int[Array, "members"]:
    """
    Which load case decided each member's size.

    Parameters
    ----------
    utilization :
        Demand over resistance of every member under every load case.

    Returns
    -------
    governing_load_case :
        Index of the load case working each member hardest.

    Notes
    -----
    **Non-differentiable**, and absent from any design for that reason.

    The picture only a differentiable code check can produce: as the form
    changes, the pattern of which load case governs where reorganizes, and it
    does so because the shape decides how much bending each one raises rather
    than because any member was reassigned.
    """
    return jnp.argmax(utilization, axis=0)


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
        `normax.sizing.size_envelope` is what collapses the two, and it is
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
        shape = self.formfinder(params.q, loads.formfinding)
        forces = self.analyzer(shape, params.diameters, loads.analysis)
        sizes = self.sizer(forces, shape.lengths)

        return Design(shape, forces, sizes)
