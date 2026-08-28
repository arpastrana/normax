# SPDX-License-Identifier: Apache-2.0
"""
EN 1993-1-1 as a block of the design pipeline.

`ec3x` implements the standard: what a section is, what it resists, and the
diameter at which a member is exactly satisfied. This module is the thin
adapter that lets a pipeline call it beside a form finder and a frame
analysis; every clause it reaches lives there, and nothing about a clause is
decided here.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from ec3x.actions import MemberActions
from ec3x.classification import section_class_at_ratio
from ec3x.material import Steel
from ec3x.section import Tube
from ec3x.section import TubeCatalogue as Ec3Catalogue
from ec3x.sizing import diameter_required
from ec3x.sizing import moment_factor_linear
from ec3x.sizing import utilization_design
from jaxtyping import Array
from jaxtyping import Float

from normax.analysis import MemberForces
from normax.materials import SteelGrade
from normax.sections import MemberSections
from normax.sections import TubeCatalog
from normax.sizing import AbstractMemberSizer
from normax.sizing import MemberSizes
from normax.structures import Structure


def coerce_material(grade: SteelGrade) -> Steel:
    """
    Read a steel grade in the terms the standard states.

    Parameters
    ----------
    grade :
        The steel as a certificate states it, free of any standard.

    Returns
    -------
    steel :
        The same steel with the standard's partial factors and buckling curve
        beside it, at their defaults.
    """
    return Steel(
        f_y=grade.f_y,
        e_mod=grade.e_mod,
        density=grade.density,
        f_u=grade.f_u,
    )


def coerce_member_sections(tubes: Tube) -> MemberSections:
    """
    Restate the standard's tubes as the sections a design carries.

    Parameters
    ----------
    tubes :
        Tubes as this standard's catalog generated them.

    Returns
    -------
    sections :
        The same geometry and the same steel, with the class and the partial
        factors left behind.
    """
    steel = tubes.material
    grade = SteelGrade(
        f_y=steel.f_y,
        f_u=steel.f_u,
        e_mod=steel.e_mod,
        density=steel.density,
    )

    return MemberSections(tubes.diameter, tubes.thickness, grade)


def read_axisymmetric_moment(
    forces: MemberForces,
) -> tuple[Float[Array, "members"], Float[Array, "members"]]:
    """
    The design moment and its factor, read without naming a local axis.

    Parameters
    ----------
    forces :
        What every member carries under one load case, with no load case axis.

    Returns
    -------
    design_and_factor :
        The larger end moment in magnitude, and the moment ratio of Table B.3.

    Notes
    -----
    A circular hollow section has no major axis, so which of the two moments
    carries which part of one bending is a solver's frame convention. What no
    frame changes is each end's moment vector length and the angle between the
    two: the design moment is the longer vector, and the ratio is the dot
    product over the larger squared length, which collinear ends reduce to the
    signed reading exactly.
    """
    first = jnp.stack([forces.moment_major[:, 0], forces.moment_minor[:, 0]], axis=-1)
    second = jnp.stack([forces.moment_major[:, 1], forces.moment_minor[:, 1]], axis=-1)

    together = jnp.sum(first * second, axis=-1)
    larger = jnp.maximum(
        jnp.linalg.norm(first, axis=-1), jnp.linalg.norm(second, axis=-1)
    )
    bent = larger > 0.0
    ratio = jnp.where(bent, together / jnp.where(bent, larger, 1.0) ** 2, 1.0)

    return larger, moment_factor_linear(ratio)


def coerce_member_actions(forces: MemberForces) -> MemberActions:
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
    EN 1993-1-1 Table B.3, first row: two end moments become a design moment
    and an equivalent uniform moment factor, exact under nodal loading since
    the moment is linear along the span. The two axes are read as one because
    the section is a tube, so the minor moment comes back zero and its factor
    one.
    """
    moment, factor = read_axisymmetric_moment(forces)
    absent = jnp.zeros_like(moment)
    acting = MemberActions(
        forces.axial_force,
        moment,
        absent,
        factor,
        jnp.ones_like(factor),
    )

    return acting


class Ec3Sizer(AbstractMemberSizer):
    """
    EN 1993-1-1, as a block of the design pipeline.

    Attributes
    ----------
    structure :
        The structure whose members are sized. Read for nothing.
    ec3_catalog :
        The section catalog in the standard's terms.
    resultant :
        Whether the two moments combine as a resultant in the cross-section
        check, or as a linear sum.
    section_class :
        Cross-section class, derived from the catalog's ratio.

    Notes
    -----
    Configured by a neutral catalog and converting internally: the partial
    factors, the buckling curve and the class are derived here at their
    defaults. The class selects a clause, so it is classified once, on the
    host, where the Class 4 refusal can fire. Every load case is sized on its
    own; reconciling them is smoothing rather than a clause.
    """

    structure: Structure
    ec3_catalog: Ec3Catalogue
    resultant: bool = eqx.field(static=True)
    section_class: int = eqx.field(static=True)

    def __init__(
        self,
        structure: Structure,
        catalog: TubeCatalog,
        resultant: bool = True,
    ) -> None:
        """
        Build a sizer over a section catalog stated as bare geometry.

        Parameters
        ----------
        structure :
            The structure whose members are sized. Read for nothing.
        catalog :
            The section catalog every member is drawn from.
        resultant :
            Whether the two moments combine as a resultant in the cross-section
            check, or as a linear sum.

        Raises
        ------
        ValueError
            If the catalog's ratio classifies as Class 4.
        """
        steel = coerce_material(catalog.material)
        section_class = section_class_at_ratio(catalog.ratio, steel.f_y)
        ec3_catalog = Ec3Catalogue(catalog.ratio, section_class, steel)

        self.structure = structure
        self.ec3_catalog = ec3_catalog
        self.resultant = resultant
        self.section_class = section_class

    @property
    def catalog(self) -> TubeCatalog:
        """
        The section catalog this block sizes over, as bare geometry.
        """
        steel = self.ec3_catalog.material
        grade = SteelGrade(
            f_y=steel.f_y,
            f_u=steel.f_u,
            e_mod=steel.e_mod,
            density=steel.density,
        )

        return TubeCatalog(self.ec3_catalog.ratio, grade)

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
        The size is the root of a residual monotone in the diameter, so it is
        unique, and the block carries an implicit tangent at that root.
        """

        def size_case(carried: MemberForces):
            acting = coerce_member_actions(carried)
            demanded = diameter_required(
                acting,
                buckling_length,
                self.ec3_catalog,
                resultant=self.resultant,
            )
            used = utilization_design(
                self.ec3_catalog(demanded),
                acting,
                buckling_length,
                resultant=self.resultant,
            )

            return demanded, used

        demanded, used = jax.vmap(size_case)(forces)
        sections = coerce_member_sections(self.ec3_catalog(demanded))

        return MemberSizes(sections, used)

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
            What every member carries under every load case.
        buckling_length :
            Length every member is assumed to buckle over.

        Returns
        -------
        utilization :
            Demand over resistance of every member under every load case,
            member buckling included since the whole check traces.
        """
        tubes = self.ec3_catalog(diameters)

        def utilization_case(carried: MemberForces):
            acting = coerce_member_actions(carried)

            return utilization_design(
                tubes,
                acting,
                buckling_length,
                resultant=self.resultant,
            )

        return jax.vmap(utilization_case)(forces)
