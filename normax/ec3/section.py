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
A circular hollow section, and the family a member's size is drawn from.

Geometry, not EN 1993-1-1. No clause defines the annulus formulas below; the
standard is what reads them, and the two places a class number appears here are
labels on a family rather than clauses applied to one.

**A catalogue is a parametrized cross-section generator, and a tube is what it
generates.** Calling a catalogue at a diameter returns a tube carrying
everything it was generated with — the wall the family gives that diameter, the
grade it is rolled from and the class its wall proportion falls in — so nothing
downstream has to be handed a section and a grade that could disagree.

A tube's geometry is stored as an outer diameter and a wall thickness, and every
property is derived on access. Two geometric leaves rather than eight is what
keeps the derivative honest: a diameter and a wall cannot drift apart, and there
is one place where a wall is chosen for a diameter.

Sizing a member means moving along a catalogue, whose fixed
diameter-to-thickness ratio makes the wall proportional to the diameter. Every
property is then a monomial in the diameter times a function of the ratio
alone, which is what makes the fully-stressed map well posed: the radius of
gyration is proportional to the diameter, so slenderness falls as the diameter
grows while area grows quadratically, and buckling capacity is strictly
increasing in the diameter with a unique root. **That argument belongs to the
catalogue and not to the tube** — a tube whose wall is held fixed as its
diameter grows does not satisfy it.
"""

import collections

import jax
import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float

from normax.ec3.classification import SectionClass
from normax.ec3.classification import ratio_at_class_limit
from normax.ec3.classification import section_class_at_ratio
from normax.ec3.material import Steel

# EN 10210 lists no smaller hot-finished tube, so a member is never sized below
# this however light its actions.
DIAMETER_MINIMUM = 21.3


def _is_traced(value: float | Float[Array, ""]) -> bool:
    """
    Whether a value is a tracer rather than a number that can be compared.

    Returns
    -------
    traced :
        True inside a jitted or differentiated function, for a value that
        function is a function of.
    """
    return isinstance(value, jax.core.Tracer)


_TubeFields = collections.namedtuple(
    "_TubeFields", ("diameter", "thickness", "material", "section_class")
)

_CatalogueFields = collections.namedtuple(
    "_CatalogueFields", ("ratio", "section_class", "material", "diameter_min")
)


class Tube(_TubeFields):
    """
    One member's circular hollow section, and what it is made of.

    Attributes
    ----------
    diameter :
        Outer diameter.
    thickness :
        Wall thickness.
    material :
        The steel it is rolled from, and the partial factors applied to it.
    section_class :
        Cross-section class of its wall proportion, 1, 2 or 3.

    Notes
    -----
    **A section travels with its grade and its class, because neither a mass nor
    a resistance can be read off geometry alone.** An area is geometry and a
    density is a material, and mass per unit length needs both; a section modulus
    is geometry and the class is what says which of the two to take. Carrying all
    of it is what lets a design be weighed and re-read without reaching back into
    the block that sized it.

    Every geometric property is computed on access rather than stored, so a tube
    has two geometric leaves and no derived quantity can disagree with the two it
    came from. Under `jit` the repeated subexpressions are eliminated; eagerly
    they cost a few flops.

    **The class is coerced to a `SectionClass` when a tube is built, and every
    other field is a leaf.** A class selects a clause, so it has to be a Python
    integer wherever one reads it, and a bare integer here would be a leaf and so
    a tracer the moment the container is a primal of the sizing map under `jit`.
    `SectionClass` is an integer that travels in the tree structure rather than in
    the leaves, so it survives `jit`, `grad` and `vmap` as itself while the
    geometry and the grade stay differentiable. Coercing in the constructor is what
    makes that unconditional rather than a convention a caller has to remember.

    The load case axis is variadic, like `normax.analysis.MemberForces`. A check
    answers one size per member per load case, so the sections it returns carry
    that axis on their geometry and not on their grade; reconciling them into one
    size per member collapses it, and both ranks are the same container.
    """

    diameter: Float[Array, "*load_cases members"]
    thickness: Float[Array, "*load_cases members"]
    material: Steel
    section_class: SectionClass

    def __new__(
        cls,
        diameter: Float[Array, "*load_cases members"],
        thickness: Float[Array, "*load_cases members"],
        material: Steel,
        section_class: int,
    ) -> "Tube":
        """
        Build a tube, holding its class out of the leaves.

        Returns
        -------
        tube :
            The same four values, the class as a `SectionClass`.
        """
        return super().__new__(
            cls, diameter, thickness, material, SectionClass(section_class)
        )

    @classmethod
    def _make(cls, values: "tuple[object, ...]") -> "Tube":
        """
        Rebuild from an iterable, through the constructor rather than around it.

        Returns
        -------
        tube :
            A tube with its class coerced, which `_replace` then inherits.
        """
        return cls(*values)

    @property
    def ratio(self) -> Float[Array, "*load_cases members"]:
        """
        Diameter-to-thickness ratio.

        Returns
        -------
        ratio :
            Outer diameter over wall thickness, the slenderness EN 1993-1-1
            Table 5.2 sheet 3 classifies on.
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
        the same about every centroidal axis and flexural buckling governs.
        There is no lateral-torsional buckling to check.
        """
        outer = jnp.asarray(self.diameter)
        bore = self.diameter_inner

        return (jnp.pi / 64.0) * (outer**4 - bore**4)

    @property
    def radius_of_gyration(self) -> Float[Array, "*load_cases members"]:
        """
        Radius of gyration about any centroidal axis.

        Returns
        -------
        radius_of_gyration :
            Radius of gyration.

        Notes
        -----
        Proportional to the diameter whenever the wall is, which is what a
        catalogue guarantees. Slenderness then varies as the reciprocal of the
        diameter.
        """
        return jnp.sqrt(self.second_moment / self.area)

    @property
    def modulus_elastic(self) -> Float[Array, "*load_cases members"]:
        """
        Elastic section modulus.

        Returns
        -------
        modulus_elastic :
            Second moment over the outer radius.
        """
        return 2.0 * self.second_moment / jnp.asarray(self.diameter)

    @property
    def modulus_plastic(self) -> Float[Array, "*load_cases members"]:
        """
        Plastic section modulus.

        Returns
        -------
        modulus_plastic :
            Plastic section modulus.
        """
        outer = jnp.asarray(self.diameter)

        return (outer**3 - self.diameter_inner**3) / 6.0


class TubeCatalogue(_CatalogueFields):
    """
    The family of circular hollow sections a member is drawn from.

    Attributes
    ----------
    ratio :
        Diameter-to-thickness ratio, fixed so that a member carries a single
        size variable.
    section_class :
        Cross-section class the ratio falls in, 1, 2 or 3.
    material :
        The steel the family is rolled from, and the partial factors applied
        to it.
    diameter_min :
        Smallest diameter the family offers.

    Notes
    -----
    **A section family is not geometry.** The number that defines the one used
    here is the class limit of EN 1993-1-1 Table 5.2 sheet 3, which is a function
    of the grade and of the class, so the grade and the class are the family's
    identity rather than companions it is handed beside. Holding all three
    together is what lets a call site take a catalogue alone, and what stops a
    grade travelling beside a ratio that was derived from a different one.

    Calling the catalogue at a diameter generates the tube, which carries the
    grade and the class onward. That is the only place a wall is chosen for a
    diameter.

    **The class travels in the tree structure and the ratio in the leaves, and the
    split is the whole reason `SectionClass` exists.** A class selects between
    clauses, so a clause needs it as a Python integer; a bare integer here would be
    a leaf and so a tracer the moment the catalogue is a primal of the sizing map
    under `jit`, and the branch it selects would raise. The ratio has to stay a
    leaf for the opposite reason: §3 wants it freeable, the gradient tests
    differentiate the map in it, and the class sweep moves it across two class
    boundaries. The constructor coerces the class so neither can be got wrong by a
    caller.

    `verified_class` is what confirms the label against the ratio before a block
    trusts it, since a label named beside a number is free to contradict it.

    The buckling curve is not a field. It follows the fabrication route rather
    than the shape, so it belongs to the grade.
    """

    ratio: float | Float[Array, ""]
    section_class: SectionClass
    material: Steel
    diameter_min: float | Float[Array, ""]

    def __new__(
        cls,
        ratio: float | Float[Array, ""],
        section_class: int,
        material: Steel,
        diameter_min: float | Float[Array, ""] = DIAMETER_MINIMUM,
    ) -> "TubeCatalogue":
        """
        Build a family, holding its class out of the leaves.

        Returns
        -------
        catalogue :
            The same four values, the class as a `SectionClass`.

        Notes
        -----
        The default sits here rather than in the class body. A namedtuple reads a
        field through a descriptor on the class, and assigning a default beside the
        annotation shadows it — every instance would then report the default
        whatever it was built with.
        """
        return super().__new__(
            cls, ratio, SectionClass(section_class), material, diameter_min
        )

    @classmethod
    def _make(cls, values: "tuple[object, ...]") -> "TubeCatalogue":
        """
        Rebuild from an iterable, through the constructor rather than around it.

        Returns
        -------
        catalogue :
            A family with its class coerced, which `_replace` then inherits.
        """
        return cls(*values)

    def __call__(self, diameter: Float[Array, "*load_cases members"]) -> Tube:
        """
        Generate the tube of a given diameter.

        Parameters
        ----------
        diameter :
            Outer diameter.

        Returns
        -------
        tube :
            A tube of that diameter, walled by this family's ratio and carrying
            its grade and its class.

        Notes
        -----
        The one place a wall is chosen for a diameter, so a diameter can never be
        paired with the wrong one, nor a section with the wrong grade. A
        catalogue is called rather than asked for a tube because generating one
        is the whole of what it does.

        The catalogue minimum is deliberately not applied here: it is a limit on
        what may be ordered rather than on what the formulas mean, and the sizing
        map applies it outside the root it solves for.
        """
        thickness = jnp.asarray(diameter) / self.ratio

        return Tube(diameter, thickness, self.material, self.section_class)

    def verified_class(self) -> SectionClass:
        """
        This family's class, confirmed against the ratio that fixes it.

        Returns
        -------
        section_class :
            Class 1, 2 or 3, as the integer a clause can select on.

        Raises
        ------
        ValueError
            If the ratio classifies as Class 4, or as a class other than the one
            named.

        Notes
        -----
        EN 1993-1-1 Table 5.2 sheet 3. A class named beside a ratio is free to
        contradict it, and allowing the plastic clauses onto a Class 3 wall is
        unsafe, so a block reads its class through this rather than off the field.

        **Verified where the numbers are concrete and trusted where they are
        not.** A traced ratio or grade has no value to classify, and every
        production call site builds a catalogue on the host, where the comparison
        runs. Under a tracer there is nothing to compare against and the label
        stands.
        """
        if _is_traced(self.ratio) or _is_traced(self.material.f_y):
            return self.section_class

        classified = section_class_at_ratio(self.ratio, self.material.f_y)

        if classified != self.section_class:
            raise ValueError(
                f"a ratio of {float(self.ratio)} at f_y {float(self.material.f_y)} "
                f"is Class {classified}, not Class {self.section_class}; "
                "the class selects a clause and cannot contradict the wall"
            )

        return classified

    @classmethod
    def at_class_limit(
        cls,
        material: Steel,
        section_class: int,
        diameter_min: float | Float[Array, ""] = DIAMETER_MINIMUM,
    ) -> "TubeCatalogue":
        """
        The family whose wall is as thin as a given class allows.

        Parameters
        ----------
        material :
            The steel the family is rolled from.
        section_class :
            Class 1, 2 or 3.
        diameter_min :
            Smallest diameter the family offers.

        Returns
        -------
        catalogue :
            A catalogue whose ratio sits exactly on that class's limit.

        Raises
        ------
        ValueError
            If the class is not 1, 2 or 3.

        Notes
        -----
        EN 1993-1-1 Table 5.2 sheet 3. Sitting on the limit maximises the wall
        slenderness, and so minimises material, while staying inside the class.
        Class 4 is refused: beyond the third limit the tube is a shell and
        EN 1993-1-6 applies instead.

        The way a family is ordinarily built, the ratio being the answer rather
        than the input. `TubeCatalogue` itself is for the other case, where a
        ratio is named — a swept one, or one read off a published section — and
        the class is what is claimed about it.
        """
        ratio = ratio_at_class_limit(material.f_y, section_class)

        return cls(ratio, section_class, material, diameter_min)
