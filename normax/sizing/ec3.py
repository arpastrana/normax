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

`ec3x` implements the standard: what a section is, what it resists, and
the diameter at which a member is exactly satisfied. This module is the adapter
that lets a pipeline call it beside a form finder and a frame analysis, and it
is deliberately thin — every clause it reaches lives there, and nothing about a
clause is decided here.

The separation is what keeps `ec3x` free of any opinion about pipelines,
so a second standard added beside it inherits none of ours.
"""

import equinox as eqx
import jax
from ec3x.actions import MemberActions
from ec3x.classification import ratio_at_class_limit
from ec3x.classification import section_class_at_ratio
from ec3x.material import Steel
from ec3x.section import Tube
from ec3x.section import TubeCatalogue
from ec3x.sizing import diameter_required
from ec3x.sizing import end_moments
from ec3x.sizing import governing_limit_state
from ec3x.sizing import utilization_design
from jaxtyping import Array
from jaxtyping import Float

from normax.analysis import DESIGN_AXES
from normax.analysis import MemberForces
from normax.materials import SteelGrade
from normax.sections import MemberSections
from normax.sections import TubeFamily
from normax.sizing import AbstractMemberSizer
from normax.sizing import MemberSizes
from normax.structures import Structure


def design_steel(grade: SteelGrade) -> Steel:
    """
    Read a steel grade in the terms the standard states.

    Parameters
    ----------
    grade :
        The steel as a certificate states it, free of any standard.

    Returns
    -------
    steel :
        The same steel with EN 1993-1-1's partial factors and imperfection
        factor beside it, at their defaults.

    Notes
    -----
    The certificate half crosses unchanged; what is added is what only this
    standard can add — the partial factors of §6.1 and the buckling curve of
    Table 6.2. The defaults are the UK National Annex factors and curve a,
    the hot-finished hollow section, and the sizer freezes them: other
    factors, or the cold-formed curve c, are future work rather than an
    argument.
    """
    return Steel(
        f_y=grade.f_y,
        e_mod=grade.e_mod,
        density=grade.density,
        f_u=grade.f_u,
    )


def build_section_family(grade: SteelGrade, section_class: int) -> TubeFamily:
    """
    The section family as thin as a given class allows, from a bare grade.

    Parameters
    ----------
    grade :
        The steel as a certificate states it, free of any standard.
    section_class :
        Class 1, 2 or 3, whose Table 5.2 limit fixes the wall proportion.

    Returns
    -------
    family :
        The family whose ratio sits exactly on that class's limit.

    Raises
    ------
    ValueError
        If the class is not 1, 2 or 3.

    Notes
    -----
    EN 1993-1-1 Table 5.2 sheet 3. Sitting on the limit maximises the wall
    slenderness, and so minimises material, while staying inside the class —
    the way a family is ordinarily chosen here. The derivation is clause work,
    so it lives with the sizer rather than on the neutral container it returns:
    a driver names the standard by importing this, and the family it gets back
    names nothing.
    """
    ratio = ratio_at_class_limit(grade.f_y, section_class)

    return TubeFamily(ratio, grade)


def neutral_sections(tubes: Tube) -> MemberSections:
    """
    Restate the standard's tubes as the sections a design carries.

    Parameters
    ----------
    tubes :
        Tubes as this standard's catalogue generated them.

    Returns
    -------
    sections :
        The same geometry and the same steel, with everything a clause decided
        left behind.

    Notes
    -----
    The inverse crossing of `design_steel` and `design_actions`: those read
    neutral records in the standard's terms on the way in, and this strips the
    standard's terms on the way out. What is dropped is the class — a label
    that selects clauses, meaningless to any other standard — and the partial
    factors riding on the material. The geometry and the certificate half of
    the steel cross unchanged, so nothing a mass or a re-analysis reads moves.
    """
    steel = tubes.material
    grade = SteelGrade(
        f_y=steel.f_y,
        f_u=steel.f_u,
        e_mod=steel.e_mod,
        density=steel.density,
    )

    return MemberSections(tubes.diameter, tubes.thickness, grade)


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


class Ec3Sizer(AbstractMemberSizer):
    """
    EN 1993-1-1, as a block of the design pipeline.

    Attributes
    ----------
    structure :
        The structure whose members are sized. Read for nothing.
    catalogue :
        The section family in the standard's terms, derived from the neutral
        family the block was configured by.
    resultant :
        Whether the two moments combine as a resultant in the cross-section
        check, or as a linear sum.
    section_class :
        Cross-section class, derived from the family's ratio rather than
        accepted beside it.

    Notes
    -----
    **The block is configured by neutral containers and converts internally.**
    A driver hands in a `TubeFamily` — a ratio and a certificate-level grade,
    naming no EC3 vocabulary — and everything the standard adds is derived in
    here at its defaults: the partial factors, the buckling curve, the class
    the ratio falls in. That is the model a second standard's sizer replicates
    beside this one; other factors or a cold-formed curve are future work.

    **A grade is not an argument here, because a section family already has
    one.** The ratio only means something read at a yield strength, so a grade
    handed in beside the family could be a different one; taking the family
    alone makes that unrepresentable.

    **The cross-section class is derived and never taken on trust.** It selects
    a clause rather than scaling a number, so it has to be static, and it is
    classified once when the block is built, on the host, where the ratio and
    the yield strength are concrete numbers and the Class 4 refusal can fire.

    **The structure settles nothing, and that is the honest answer here.** A
    code check reads one member at a time and knows nothing of connectivity, so
    there is no view of one for this block to build. It is taken all the same,
    because a block that needs no topology saying so is the point rather than
    an omission.

    Every load case is sized for on its own. Reconciling several of them into
    one size per member is smoothing rather than a clause, and belongs above a
    block that implements a standard.
    """

    structure: Structure
    catalogue: TubeCatalogue
    resultant: bool = eqx.field(static=True)
    section_class: int = eqx.field(static=True)

    def __init__(
        self,
        structure: Structure,
        family: TubeFamily,
        resultant: bool = True,
    ) -> None:
        """
        Build a sizer over a section family stated as bare geometry.

        Parameters
        ----------
        structure :
            The structure whose members are sized. Read for nothing.
        family :
            The section family every member is drawn from, whose ratio fixes
            the wall proportion and whose grade supplies the material.
        resultant :
            Whether the two moments combine as a resultant in the cross-section
            check, or as a linear sum.

        Raises
        ------
        ValueError
            If the family's ratio classifies as Class 4.
        """
        steel = design_steel(family.material)
        section_class = section_class_at_ratio(family.ratio, steel.f_y)
        catalogue = TubeCatalogue(family.ratio, section_class, steel)

        self.structure = structure
        self.catalogue = catalogue
        self.resultant = resultant
        self.section_class = section_class

    @property
    def steel(self) -> Steel:
        """
        The steel every member is cut from, in this standard's terms.
        """
        return self.catalogue.material

    @property
    def family(self) -> TubeFamily:
        """
        The section family this block sizes over, as bare geometry.

        Notes
        -----
        What an analysis wants of the sizer's family — the wall proportion and
        the grade, nothing a clause decided — so a driver that configured this
        block can build the frame's sections off it without importing the
        standard's library. Reading the ratio here rather than restating it is
        what keeps one number from being derived twice.
        """
        steel = self.catalogue.material
        grade = SteelGrade(
            f_y=steel.f_y,
            f_u=steel.f_u,
            e_mod=steel.e_mod,
            density=steel.density,
        )

        return TubeFamily(self.catalogue.ratio, grade)

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
            The section each load case demands, and how hard it is worked.

        Notes
        -----
        The size is the root of a residual that is monotone in the diameter, so
        it is unique, and the block carries an implicit tangent at that root
        rather than differentiating the solve that found it.

        **The utilization comes back with the size rather than being asked for
        afterwards.** It is the same clause read at the size just chosen, so it
        costs a fraction of the root find that produced it, and a caller holding
        sizes never has to know how to re-derive it. That it is one is the
        invariant the map exists to hold.

        Every load case runs through one `vmap` rather than a Python loop, the
        check being elementwise over members and the root find batching cleanly.
        """

        def size_case(carried: MemberForces):
            acting = design_actions(carried)
            demanded = diameter_required(
                acting,
                buckling_length,
                self.catalogue,
                resultant=self.resultant,
            )
            used = utilization_design(
                self.catalogue(demanded),
                acting,
                buckling_length,
                resultant=self.resultant,
            )

            return demanded, used

        demanded, used = jax.vmap(size_case, in_axes=(DESIGN_AXES,))(forces)
        sections = neutral_sections(self.catalogue(demanded))

        return MemberSizes(sections, used)

    def governing(
        self,
        diameters: Float[Array, "members"],
        forces: MemberForces,
        buckling_length: Float[Array, "members"],
    ) -> Float[Array, "load_cases members"]:
        """
        Which limit state decided each member's size, under each load case.

        Parameters
        ----------
        diameters :
            Outer diameter every member was given.
        forces :
            What every member carries under every load case, reduced to design
            actions here because that reduction is a clause.
        buckling_length :
            Length every member is assumed to buckle over.

        Returns
        -------
        governing :
            One of the limit-state codes of `ec3x.sizing`.

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
        tubes = self.catalogue(diameters)

        def governing_case(carried: MemberForces):
            acting = design_actions(carried)

            return governing_limit_state(
                tubes,
                acting,
                buckling_length,
                self.catalogue,
                resultant=self.resultant,
            )

        return jax.vmap(governing_case, in_axes=(DESIGN_AXES,))(forces)

    def compute_utilization(
        self,
        diameters: Float[Array, "members"],
        forces: MemberForces,
        buckling_length: Float[Array, "members"],
    ) -> Float[Array, "load_cases members"]:
        """
        Check sizes the caller owns against EN 1993-1-1.

        Parameters
        ----------
        diameters :
            Outer diameter every member was given.
        forces :
            What every member carries under every load case, reduced to design
            actions here because that reduction is a clause.
        buckling_length :
            Length every member is assumed to buckle over.

        Returns
        -------
        utilization :
            Demand over resistance of every member under every load case.

        Notes
        -----
        Asked of a design after several load cases have been reconciled — at
        most one, and exactly one for whichever case governs each member — or
        of an optimizer's own diameters, where it is the differentiable
        constraint held at or under one, member buckling included since the
        whole check traces.
        """
        tubes = self.catalogue(diameters)

        def utilization_case(carried: MemberForces):
            acting = design_actions(carried)

            return utilization_design(
                tubes,
                acting,
                buckling_length,
                resultant=self.resultant,
            )

        return jax.vmap(utilization_case, in_axes=(DESIGN_AXES,))(forces)
