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

Geometry, not EN 1993-1-1. No clause defines these; they are the standard
annulus formulas, carried by the two numbers that fix a tube rather than
recomputed from a diameter at every call site.

A tube is stored as an outer diameter and a wall thickness, and every property
is derived on access. Two leaves rather than eight is what keeps the derivative
honest: a diameter and a wall cannot drift apart, and there is one place where
a wall is chosen for a diameter.

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

from typing import NamedTuple

import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float

from normax.ec3.classification import ratio_at_class_limit

# EN 10210 lists no smaller hot-finished tube, so a member is never sized below
# this however light its actions.
DIAMETER_MINIMUM = 21.3


class Tube(NamedTuple):
    """
    One member's circular hollow section.

    Attributes
    ----------
    diameter :
        Outer diameter.
    thickness :
        Wall thickness.

    Notes
    -----
    Every other property is computed on access rather than stored, so a tube is
    two leaves and no derived quantity can disagree with the two it came from.
    Under `jit` the repeated subexpressions are eliminated; eagerly they cost a
    few flops.
    """

    diameter: Float[Array, "members"]
    thickness: Float[Array, "members"]

    @property
    def ratio(self) -> Float[Array, "members"]:
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
    def diameter_inner(self) -> Float[Array, "members"]:
        """
        Inner diameter of the bore.

        Returns
        -------
        diameter_inner :
            Outer diameter less two wall thicknesses.
        """
        return jnp.asarray(self.diameter) - 2.0 * jnp.asarray(self.thickness)

    @property
    def area(self) -> Float[Array, "members"]:
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
    def second_moment(self) -> Float[Array, "members"]:
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
    def radius_of_gyration(self) -> Float[Array, "members"]:
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
    def modulus_elastic(self) -> Float[Array, "members"]:
        """
        Elastic section modulus.

        Returns
        -------
        modulus_elastic :
            Second moment over the outer radius.
        """
        return 2.0 * self.second_moment / jnp.asarray(self.diameter)

    @property
    def modulus_plastic(self) -> Float[Array, "members"]:
        """
        Plastic section modulus.

        Returns
        -------
        modulus_plastic :
            Plastic section modulus.
        """
        outer = jnp.asarray(self.diameter)

        return (outer**3 - self.diameter_inner**3) / 6.0


class TubeCatalogue(NamedTuple):
    """
    The family of circular hollow sections a member is drawn from.

    Attributes
    ----------
    ratio :
        Diameter-to-thickness ratio, fixed so that a member carries a single
        size variable.
    diameter_min :
        Smallest diameter the family offers.

    Notes
    -----
    The ratio fixes the cross-section class, but the class itself is not held
    here: it selects between two clauses and so must stay a static Python value,
    while every field of this container is a traceable leaf.

    The buckling curve is not held here either. It follows the fabrication
    route rather than the shape, so it belongs to the grade.
    """

    ratio: float | Float[Array, ""]
    diameter_min: float | Float[Array, ""] = DIAMETER_MINIMUM

    def tube(self, diameter: Float[Array, "members"]) -> Tube:
        """
        The tube of a given diameter, walled by this catalogue's ratio.

        Parameters
        ----------
        diameter :
            Outer diameter.

        Returns
        -------
        tube :
            A tube of that diameter, with the wall the family gives it.

        Notes
        -----
        The one place a wall is chosen for a diameter, so a diameter can never
        be paired with the wrong one. The catalogue minimum is deliberately not
        applied here: it is a limit on what may be ordered rather than on what
        the formulas mean, and the sizing map applies it outside the root it
        solves for.
        """
        return Tube(diameter, jnp.asarray(diameter) / self.ratio)

    @classmethod
    def at_class_limit(
        cls,
        f_y: float | Float[Array, ""],
        cross_section_class: int,
        diameter_min: float | Float[Array, ""] = DIAMETER_MINIMUM,
    ) -> "TubeCatalogue":
        """
        The family whose wall is as thin as a given class allows.

        Parameters
        ----------
        f_y :
            Yield strength.
        cross_section_class :
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
        """
        return cls(ratio_at_class_limit(f_y, cross_section_class), diameter_min)
