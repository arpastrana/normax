# SPDX-License-Identifier: Apache-2.0
"""
A circular hollow section as geometry, and the catalog that generates one.

Nothing here names a standard. No clause defines the annulus formulas below —
a standard is what reads them — and what a clause decides about a section, its
class above all, is deliberately absent: a class selects clauses, so it lives
with the sizer that owns those clauses and never travels on a design.

**A catalog is a parametrized cross-section generator, and a section is what it
generates.** Calling a catalog at a diameter returns the section walled by the
catalog's fixed diameter-to-thickness ratio, carrying the grade it is rolled
from, so nothing downstream is handed a geometry and a material that could
disagree.

A section's geometry is stored as an outer diameter and a wall thickness, and
every property is derived on access. Two geometric leaves rather than eight is
what keeps the derivative honest: a diameter and a wall cannot drift apart,
and there is one place where a wall is chosen for a diameter.

The arithmetic below is the annulus algebra and nothing else, so
`tests/test_sections.py` holds it to that algebra rather than to a second
implementation of it. Two implementations can inherit one shared mistake; a
closed form cannot.
"""

import abc
from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from jaxtyping import Float

from normax.config import check_start_fields
from normax.materials import SteelGrade
from normax.structures import Structure

# Eurocode 3 Table 5.2 sheet 3: d/t limit per class, in multiples of epsilon squared.
CLASS_LIMITS = {1: 50.0, 2: 70.0, 3: 90.0}


class MemberSections(NamedTuple):
    """
    Every member's circular hollow section, and the steel it is cut from.

    Attributes
    ----------
    diameter :
        Outer diameter.
    thickness :
        Wall thickness.
    material :
        The steel the sections are cut from, free of any standard.

    Notes
    -----
    **A section travels with its grade, because a mass cannot be read off
    geometry alone.** An area is geometry and a density is a material, and
    mass per unit length needs both. What a section does *not* travel with is
    anything a clause decided — a cross-section class is the standard's label,
    so a second sizer reading the same members hands back this same container
    with nothing missing and nothing foreign.

    Every geometric property is computed on access rather than stored, so the
    container has two geometric leaves and no derived quantity can disagree
    with the two it came from. Under `jit` the repeated subexpressions are
    eliminated; eagerly they cost a few flops.

    The load case axis is variadic, like an analysis's member-force container.
    A check answers one size per member per load case, so the sections it
    returns carry that axis on their geometry and not on their grade;
    reconciling them into one size per member collapses it, and both ranks are
    the same container.
    """

    diameter: Float[Array, "*load_cases members"]
    thickness: Float[Array, "*load_cases members"]
    material: SteelGrade

    @property
    def ratio(self) -> Float[Array, "*load_cases members"]:
        """
        Diameter-to-thickness ratio.

        Returns
        -------
        ratio :
            Outer diameter over wall thickness.
        """
        return jnp.asarray(self.diameter) / self.thickness

    @property
    def diameter_inner(self) -> Float[Array, "*load_cases members"]:
        """
        Inner diameter of the bore.

        Returns
        -------
        diameter_inner :
            Outer diameter less two wall thicknesses.
        """
        return jnp.asarray(self.diameter) - 2.0 * jnp.asarray(self.thickness)

    @property
    def area(self) -> Float[Array, "*load_cases members"]:
        """
        Gross cross-sectional area.

        Returns
        -------
        area :
            Gross area of the annulus.

        Notes
        -----
        Written as the mean circumference times the wall rather than as a
        difference of two squares, which for a thin wall would cancel most of
        its significant digits away.
        """
        wall = jnp.asarray(self.thickness)

        return jnp.pi * wall * (jnp.asarray(self.diameter) - wall)

    @property
    def second_moment(self) -> Float[Array, "*load_cases members"]:
        """
        Second moment of area about any centroidal axis.

        Returns
        -------
        second_moment :
            Second moment of area.

        Notes
        -----
        A circular hollow section is doubly symmetric, so the second moment is
        the same about every centroidal axis.
        """
        outer = jnp.asarray(self.diameter)
        bore = self.diameter_inner

        return (jnp.pi / 64.0) * (outer**4 - bore**4)


class TubeCatalog(NamedTuple):
    """
    The catalog of circular hollow sections a member's size moves along.

    Attributes
    ----------
    ratio :
        Diameter-to-thickness ratio, fixed so that a member carries a single
        size variable.
    material :
        The steel the catalog is rolled from, free of any standard.

    Notes
    -----
    **Where the ratio comes from is not this container's business.** The one
    this project runs at is a class limit of Eurocode 3 Table 5.2, but that
    derivation is clause work and lives with the sizer that performs it; the
    catalog itself is geometry, and a swept ratio or one read off a published
    section builds the same container.

    With the wall proportional to the diameter, every property of the
    generated section is a monomial in the diameter times a function of the
    ratio alone. That is what a sizer's monotonicity argument stands on, and
    it belongs to the catalog rather than to any one section — a tube whose
    wall is held fixed as its diameter grows does not satisfy it.
    """

    ratio: float | Float[Array, ""]
    material: SteelGrade

    def __call__(
        self,
        diameter: Float[Array, "*load_cases members"],
    ) -> MemberSections:
        """
        Generate the sections of a given diameter.

        Parameters
        ----------
        diameter :
            Outer diameter.

        Returns
        -------
        sections :
            Sections of that diameter, walled by this catalog's ratio and
            carrying its grade.

        Notes
        -----
        The one place a wall is chosen for a diameter, so a diameter can never
        be paired with the wrong one, nor a section with the wrong grade.
        """
        thickness = jnp.asarray(diameter) / self.ratio

        return MemberSections(diameter, thickness, self.material)


def build_section_catalog(grade: SteelGrade, section_class: int) -> TubeCatalog:
    """
    The section catalog as thin as a given class allows.

    Parameters
    ----------
    grade :
        The steel as a certificate states it.
    section_class :
        Class 1, 2 or 3, whose Table 5.2 limit fixes the wall proportion.

    Returns
    -------
    catalog :
        The catalog whose ratio sits exactly on that class's limit.

    Raises
    ------
    ValueError
        If the class is not 1, 2 or 3.

    Notes
    -----
    Eurocode 3 Table 5.2 sheet 3, `d/t <= k epsilon^2` with `epsilon^2 =
    235 / f_y`. Sitting on the limit maximizes the wall slenderness, and so
    minimizes material, while staying inside the class, so classification is
    exact by construction and needs no smoothing.
    """
    if section_class not in CLASS_LIMITS:
        raise ValueError(f"section_class must be 1, 2 or 3, got {section_class}")

    ratio = CLASS_LIMITS[section_class] * 235.0 / grade.f_y

    return TubeCatalog(ratio, grade)


class AbstractDiameterInitializer(eqx.Module):
    """
    What generates the diameters a search starts from.

    Notes
    -----
    The counterpart of `normax.form_finding.AbstractDensityInitializer`, and
    deliberately the thinner of the two. A density start is a fit — a drawn
    geometry or a sketched lens has to be balanced before it is funicular —
    whereas a diameter start is a guess the analysis runs at until the check
    replaces it, so a concrete initializer states only what those diameters
    are. Anything an initializer needs beyond the structure it takes when it
    is built, as the form finders do.
    """

    @abc.abstractmethod
    def __call__(self, structure: Structure) -> Float[np.ndarray, "members"]:
        """
        The diameter every member starts at.

        Parameters
        ----------
        structure :
            The structure whose members are to be seeded.

        Returns
        -------
        diameters :
            Outer diameter of every member at the start.
        """


class UniformDiameterInitializer(AbstractDiameterInitializer):
    """
    One diameter in every member.

    Attributes
    ----------
    diameter :
        Outer diameter every member starts at, in millimeters.
    """

    diameter: float

    def __init__(self, described: dict[str, float]):
        """
        Read the one diameter a file described.

        Parameters
        ----------
        described :
            What the file gave the start, naming `diameter` alone.
        """
        check_start_fields(described, ("diameter",))
        self.diameter = float(described["diameter"])

    def __call__(self, structure: Structure) -> Float[np.ndarray, "members"]:
        """
        Every member at the one diameter.
        """
        return np.full(structure.num_edges, self.diameter)
