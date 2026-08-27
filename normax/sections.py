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
A circular hollow section as geometry, and the family that generates one.

Nothing here names a standard. No clause defines the annulus formulas below —
a standard is what reads them — and what a clause decides about a section, its
class above all, is deliberately absent: a class selects clauses, so it lives
with the sizer that owns those clauses and never travels on a design.

**A family is a parametrized cross-section generator, and a section is what it
generates.** Calling a family at a diameter returns the section walled by the
family's fixed diameter-to-thickness ratio, carrying the grade it is rolled
from, so nothing downstream is handed a geometry and a material that could
disagree.

A section's geometry is stored as an outer diameter and a wall thickness, and
every property is derived on access. Two geometric leaves rather than eight is
what keeps the derivative honest: a diameter and a wall cannot drift apart,
and there is one place where a wall is chosen for a diameter.

The arithmetic below is stated identically in `ec3x.section`, deliberately:
the two libraries must agree about what a tube is bit for bit, and
`tests/test_sections.py` is the drift alarm.
"""

from typing import NamedTuple

import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float

from normax.materials import SteelGrade

# EN 1993-1-1 Table 5.2 sheet 3: d/t limit per class, in multiples of epsilon squared.
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


class TubeFamily(NamedTuple):
    """
    The family of circular hollow sections a member's size moves along.

    Attributes
    ----------
    ratio :
        Diameter-to-thickness ratio, fixed so that a member carries a single
        size variable.
    material :
        The steel the family is rolled from, free of any standard.

    Notes
    -----
    **Where the ratio comes from is not this container's business.** The one
    this project runs at is a class limit of EN 1993-1-1 Table 5.2, but that
    derivation is clause work and lives with the sizer that performs it; the
    family itself is geometry, and a swept ratio or one read off a published
    section builds the same container.

    With the wall proportional to the diameter, every property of the
    generated section is a monomial in the diameter times a function of the
    ratio alone. That is what a sizer's monotonicity argument stands on, and
    it belongs to the family rather than to any one section — a tube whose
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
            Sections of that diameter, walled by this family's ratio and
            carrying its grade.

        Notes
        -----
        The one place a wall is chosen for a diameter, so a diameter can never
        be paired with the wrong one, nor a section with the wrong grade.
        """
        thickness = jnp.asarray(diameter) / self.ratio

        return MemberSections(diameter, thickness, self.material)


def build_section_family(grade: SteelGrade, section_class: int) -> TubeFamily:
    """
    The section family as thin as a given class allows.

    Parameters
    ----------
    grade :
        The steel as a certificate states it.
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
    EN 1993-1-1 Table 5.2 sheet 3, `d/t <= k epsilon^2` with `epsilon^2 =
    235 / f_y`. Sitting on the limit maximizes the wall slenderness, and so
    minimizes material, while staying inside the class, so classification is
    exact by construction and needs no smoothing.
    """
    if section_class not in CLASS_LIMITS:
        raise ValueError(f"section_class must be 1, 2 or 3, got {section_class}")

    ratio = CLASS_LIMITS[section_class] * 235.0 / grade.f_y

    return TubeFamily(ratio, grade)
