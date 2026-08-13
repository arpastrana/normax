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
import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float

from normax.ec3.actions import MemberActions
from normax.ec3.material import Steel
from normax.ec3.section import TubeCatalogue
from normax.ec3.sizing import diameter_required
from normax.ec3.sizing import end_moments
from normax.ec3.sizing import governing_limit_state
from normax.ec3.sizing import utilization_design
from normax.loads import count_load_cases
from normax.loads import select_load_case
from normax.loads import stack_load_cases
from normax.stages import AbstractMemberSizer
from normax.stages import MemberForces
from normax.stages import MemberSizes
from normax.structures import Structure


def design_actions(
    forces: MemberForces,
    load_case: int,
) -> MemberActions:
    """
    Read one load case of an analysis in the terms the standard states.

    Parameters
    ----------
    forces :
        What every member carries under every load case.
    load_case :
        Index of the load case to read.

    Returns
    -------
    actions :
        Axial force, both design moments and both moment factors.

    Notes
    -----
    The reduction from two end moments to a design moment and a factor is
    EN 1993-1-1 Table B.3, so an analysis stops one step short of this and the
    step belongs to the check. That the two ends are enough is a property of
    nodal loading, which leaves the moment linear along the span.

    Which case to read is `normax.loads`'s business and the reduction is this
    module's, so the two are one call each rather than an index repeated over
    every field.
    """
    carried = select_load_case(forces, load_case)

    moment_major, factor_major = end_moments(
        carried.moment_major[:, 0], carried.moment_major[:, 1]
    )
    moment_minor, factor_minor = end_moments(
        carried.moment_minor[:, 0], carried.moment_minor[:, 1]
    )
    acting = MemberActions(
        carried.axial_force,
        moment_major,
        moment_minor,
        factor_major,
        factor_minor,
    )

    return acting


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
            The design actions read, and the diameter each load case demands.

        Notes
        -----
        The size is the root of a residual that is monotone in the diameter, so
        it is unique, and the block carries an implicit tangent at that root
        rather than differentiating the solve that found it.
        """
        num_load_cases = count_load_cases(forces)
        per_case = [design_actions(forces, index) for index in range(num_load_cases)]

        demanded = [
            diameter_required(
                acting,
                buckling_length,
                self.steel,
                self.catalogue,
                section_class=self.section_class,
                resultant=self.resultant,
            )
            for acting in per_case
        ]
        sizes = MemberSizes(stack_load_cases(per_case), jnp.stack(demanded))

        return sizes

    def mass_per_length(
        self,
        diameters: Float[Array, "members"],
    ) -> Float[Array, "members"]:
        """
        Mass a member of a given size carries per unit of its length.

        Parameters
        ----------
        diameters :
            Outer diameter of every member.

        Returns
        -------
        mass_per_length :
            Mass per unit length of every member.

        Notes
        -----
        The wall follows the diameter through the family's ratio, so a size is
        one number and the area it implies is exact rather than interpolated.
        """
        tubes = self.catalogue.tube_at(diameters)

        return self.steel.density * tubes.area

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
        num_load_cases = count_load_cases(actions)

        governing = [
            governing_limit_state(
                tubes,
                select_load_case(actions, index),
                buckling_length,
                self.steel,
                self.catalogue,
                section_class=self.section_class,
                resultant=self.resultant,
            )
            for index in range(num_load_cases)
        ]

        return jnp.stack(governing)

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
        num_load_cases = count_load_cases(actions)

        used = [
            utilization_design(
                tubes,
                select_load_case(actions, index),
                buckling_length,
                self.steel,
                section_class=self.section_class,
                resultant=self.resultant,
            )
            for index in range(num_load_cases)
        ]

        return jnp.stack(used)
