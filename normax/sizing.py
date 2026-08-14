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
EN 1993-1-1 as a block of the design pipeline.

`normax.ec3` implements the standard: what a section is, what it resists, and
the diameter at which a member is exactly satisfied. This module is the adapter
that lets a pipeline call it beside a form finder and a frame analysis, and it
is deliberately thin — every clause it reaches lives there, and nothing about a
clause is decided here.

The separation is what keeps `normax.ec3` free of any opinion about pipelines,
so a second standard added beside it inherits none of ours.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float

from normax.design import AbstractMemberSizer
from normax.design import Design
from normax.design import MemberForces
from normax.design import MemberSizes
from normax.ec3.actions import MemberActions
from normax.ec3.material import Steel
from normax.ec3.section import MemberSection
from normax.ec3.section import TubeCatalogue
from normax.ec3.sizing import diameter_envelope
from normax.ec3.sizing import diameter_required
from normax.ec3.sizing import end_moments
from normax.ec3.sizing import governing_limit_state
from normax.ec3.sizing import utilization_design
from normax.loads import count_load_cases
from normax.loads import select_load_case
from normax.structures import Structure


def design_actions(forces: MemberForces) -> MemberActions:
    """
    Read one load case of an analysis in the terms the standard states.

    Parameters
    ----------
    forces :
        What every member carries under one load case, with no load case axis.

    Returns
    -------
    actions :
        Axial force, both design moments and both moment factors.

    Notes
    -----
    **EN 1993-1-1 Table B.3, first row, and nothing else.** Two end moments
    become a design moment and an equivalent uniform moment factor: the checks
    of 6.3.3 are written for a member under a uniform moment, and a real one
    almost never has one, so the standard converts the diagram it does have into
    the uniform moment that would be equally severe. That reduction is lossy —
    the factor cannot be recovered from the design moment — which is why both
    come back. Nodal loading leaves the moment linear along the span, so the row
    is exact here rather than approximate.

    An analysis stops one step short of this and the step belongs to the check.

    **One load case, and vectorized rather than indexed.** Every operation is
    elementwise over members, so several load cases are `jax.vmap` of this over
    the leading axis of a stacked container, and the check runs batched rather
    than looped.
    """
    moment_major, factor_major = end_moments(
        forces.moment_major[:, 0], forces.moment_major[:, 1]
    )
    moment_minor, factor_minor = end_moments(
        forces.moment_minor[:, 0], forces.moment_minor[:, 1]
    )
    acting = MemberActions(
        forces.axial_force,
        moment_major,
        moment_minor,
        factor_major,
        factor_minor,
    )

    return acting


def size_envelope(
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

    **The tube is enveloped leafwise and the wall follows exactly.** The
    envelope is scale-equivariant — taken in the logarithm, a constant factor
    passes straight through it — so enveloping a diameter and a thickness that
    is that diameter over a fixed ratio preserves the ratio, and no catalogue is
    needed to re-derive a wall.

    A single load case is returned as it stands, whatever the sharpness. The
    envelope over one case is the identity in exact arithmetic and a logarithm
    followed by an exponential in floating point, so taking it would cost the
    last bits of a size for nothing.
    """
    demanded = design.sizes.sections.tubes

    if count_load_cases(demanded) == 1:
        covering = select_load_case(demanded, 0)
    elif sharpness is None:
        covering = jax.tree.map(lambda field: jnp.max(field, axis=0), demanded)
    else:
        covering = jax.tree.map(
            lambda field: diameter_envelope(field, sharpness), demanded
        )

    sections = MemberSection(covering, design.sizes.sections.material)
    sizes = MemberSizes(sections, design.sizes.actions)

    return Design(design.shape, design.forces, sizes)


class Ec3Sizer(AbstractMemberSizer):
    """
    EN 1993-1-1, as a block of the design pipeline.

    Attributes
    ----------
    structure :
        The structure whose members are sized. Read for nothing.
    steel :
        Material properties and partial factors.
    catalogue :
        The section family every member is drawn from.
    resultant :
        Whether the two moments combine as a resultant in the cross-section
        check, or as a linear sum.
    section_class :
        Cross-section class, read from the family rather than given.

    Notes
    -----
    **The cross-section class is derived and never accepted.** It selects a
    clause rather than scaling a number, so it has to be static, and a class
    named independently of the family it describes is free to contradict it —
    which would allow the plastic clauses to be applied to a Class 3 wall. It is
    read once when the block is built, on the host, where the yield strength is
    a concrete number and the Class 4 refusal can fire.

    **The structure settles nothing, and that is the honest answer here.** A
    code check reads one member at a time and knows nothing of connectivity, so
    there is no view of one for this block to build. It is taken all the same,
    because a block that needs no topology saying so is the point rather than an
    omission: the three constructors are alike, and what differs between them is
    how much each finds worth keeping.

    Every load case is sized for on its own. Reconciling several of them into
    one size per member is smoothing rather than a clause, and belongs above a
    block that implements a standard.
    """

    structure: Structure
    steel: Steel
    catalogue: TubeCatalogue
    resultant: bool = eqx.field(static=True)
    section_class: int = eqx.field(static=True)

    def __init__(
        self,
        structure: Structure,
        steel: Steel,
        catalogue: TubeCatalogue,
        resultant: bool = True,
    ) -> None:
        """
        Build a sizer, reading its cross-section class off its section family.

        Parameters
        ----------
        structure :
            The structure whose members are sized. Read for nothing.
        steel :
            Material properties and partial factors.
        catalogue :
            The section family every member is drawn from.
        resultant :
            Whether the two moments combine as a resultant in the cross-section
            check, or as a linear sum.

        Raises
        ------
        ValueError
            If the family's ratio classifies as Class 4.
        """
        self.structure = structure
        self.steel = steel
        self.catalogue = catalogue
        self.resultant = resultant
        self.section_class = catalogue.section_class(steel.f_y)

    def __call__(
        self,
        forces: MemberForces,
        buckling_length: Float[Array, "members"],
    ) -> MemberSizes:
        """
        Size every member for every load case, each on its own.

        Parameters
        ----------
        forces :
            What every member carries under every load case.
        buckling_length :
            Length every member is assumed to buckle over.

        Returns
        -------
        sizes :
            The section each load case demands, and the actions read to get it.

        Notes
        -----
        The size is the root of a residual that is monotone in the diameter, so
        it is unique, and the block carries an implicit tangent at that root
        rather than differentiating the solve that found it.

        Every load case runs through one `vmap` rather than a Python loop, the
        check being elementwise over members and the root find batching cleanly.
        """

        def size_case(carried: MemberForces):
            acting = design_actions(carried)
            demanded = diameter_required(
                acting,
                buckling_length,
                self.steel,
                self.catalogue,
                section_class=self.section_class,
                resultant=self.resultant,
            )

            return acting, demanded

        actions, demanded = jax.vmap(size_case)(forces)
        sections = MemberSection(self.catalogue.tube_at(demanded), self.steel)

        return MemberSizes(sections, actions)

    def governing(
        self,
        diameters: Float[Array, "members"],
        actions: MemberActions,
        buckling_length: Float[Array, "members"],
    ) -> Float[Array, "load_cases members"]:
        """
        Which limit state decided each member's size, under each load case.

        Parameters
        ----------
        diameters :
            Outer diameter every member was given.
        actions :
            Design actions to check against, every field carrying a leading load
            case axis.
        buckling_length :
            Length every member is assumed to buckle over.

        Returns
        -------
        governing :
            One of the limit-state codes of `normax.ec3.sizing`.

        Notes
        -----
        **Non-differentiable**, which is why no design carries it and why the
        abstract block does not require it: a concrete cotangent on it raises
        rather than passing quietly. Read it beside a design, never through one.

        The picture only a differentiable code check can produce. As the form
        changes, the pattern of which clause governs where reorganizes, and it
        does so because the shape decides how much bending each load case raises
        rather than because any member was reassigned.
        """
        tubes = self.catalogue.tube_at(diameters)

        def governing_case(acting: MemberActions):
            return governing_limit_state(
                tubes,
                acting,
                buckling_length,
                self.steel,
                self.catalogue,
                section_class=self.section_class,
                resultant=self.resultant,
            )

        return jax.vmap(governing_case)(actions)

    def utilization(
        self,
        diameters: Float[Array, "members"],
        actions: MemberActions,
        buckling_length: Float[Array, "members"],
    ) -> Float[Array, "load_cases members"]:
        """
        Re-read a finished design against the standard that sized it.

        Parameters
        ----------
        diameters :
            Outer diameter every member was given.
        actions :
            Design actions to check against, every field carrying a leading load
            case axis.
        buckling_length :
            Length every member is assumed to buckle over.

        Returns
        -------
        utilization :
            Demand over resistance of every member under every load case.

        Notes
        -----
        Asked of a design after several load cases have been reconciled, so the
        sizes are not the ones any single case demanded and the answer is at
        most one rather than exactly one. It is exactly one for whichever case
        governs each member.
        """
        tubes = self.catalogue.tube_at(diameters)

        def utilization_case(acting: MemberActions):
            return utilization_design(
                tubes,
                acting,
                buckling_length,
                self.steel,
                section_class=self.section_class,
                resultant=self.resultant,
            )

        return jax.vmap(utilization_case)(actions)
