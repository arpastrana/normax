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
Closed-form cross-section properties of a circular hollow section.

Geometry, not EN 1993-1-1. No clause defines these; they are the standard
annulus formulas, written in terms of the outer diameter and a fixed
diameter-to-thickness ratio so that a member carries a single size variable.

Every property is a monomial in the diameter times a function of the ratio
alone. That is what makes the fully-stressed sizing map well posed: the radius
of gyration is proportional to the diameter, so slenderness falls as the
diameter grows while area grows quadratically, and buckling capacity is
strictly increasing in diameter with a unique root.
"""

import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float


def thickness(
    diameter: Float[Array, "members"],
    ratio: float | Float[Array, ""],
) -> Float[Array, "members"]:
    """
    Wall thickness.

    Parameters
    ----------
    diameter :
        Outer diameter.
    ratio :
        Diameter-to-thickness ratio.

    Returns
    -------
    thickness :
        Wall thickness.
    """
    outer = jnp.asarray(diameter)

    return outer / ratio


def diameter_inner(
    diameter: Float[Array, "members"],
    ratio: float | Float[Array, ""],
) -> Float[Array, "members"]:
    """
    Inner diameter of the bore.

    Parameters
    ----------
    diameter :
        Outer diameter.
    ratio :
        Diameter-to-thickness ratio.

    Returns
    -------
    diameter_inner :
        Inner diameter, the outer diameter less two wall thicknesses.
    """
    outer = jnp.asarray(diameter)

    return outer * (1.0 - 2.0 / ratio)


def area(
    diameter: Float[Array, "members"],
    ratio: float | Float[Array, ""],
) -> Float[Array, "members"]:
    """
    Gross cross-sectional area.

    Parameters
    ----------
    diameter :
        Outer diameter.
    ratio :
        Diameter-to-thickness ratio.

    Returns
    -------
    area :
        Gross area of the annulus.

    Notes
    -----
    Equivalent to the annulus area pi t (d - t), rewritten so the diameter
    appears once. Quadratic in the diameter.
    """
    outer = jnp.asarray(diameter)

    return jnp.pi * outer**2 * (ratio - 1.0) / ratio**2


def second_moment(
    diameter: Float[Array, "members"],
    ratio: float | Float[Array, ""],
) -> Float[Array, "members"]:
    """
    Second moment of area about any centroidal axis.

    Parameters
    ----------
    diameter :
        Outer diameter.
    ratio :
        Diameter-to-thickness ratio.

    Returns
    -------
    second_moment :
        Second moment of area.

    Notes
    -----
    A circular hollow section is doubly symmetric, so the second moment is the
    same about every centroidal axis and flexural buckling governs. Quartic in
    the diameter.
    """
    outer = jnp.asarray(diameter)
    bore = 1.0 - 2.0 / ratio

    return (jnp.pi / 64.0) * outer**4 * (1.0 - bore**4)


def radius_of_gyration(
    diameter: Float[Array, "members"],
    ratio: float | Float[Array, ""],
) -> Float[Array, "members"]:
    """
    Radius of gyration about any centroidal axis.

    Parameters
    ----------
    diameter :
        Outer diameter.
    ratio :
        Diameter-to-thickness ratio.

    Returns
    -------
    radius_of_gyration :
        Radius of gyration.

    Notes
    -----
    Proportional to the diameter, with a constant fixed by the ratio alone.
    Slenderness therefore varies as the reciprocal of the diameter.
    """
    return jnp.sqrt(second_moment(diameter, ratio) / area(diameter, ratio))


def modulus_elastic(
    diameter: Float[Array, "members"],
    ratio: float | Float[Array, ""],
) -> Float[Array, "members"]:
    """
    Elastic section modulus.

    Parameters
    ----------
    diameter :
        Outer diameter.
    ratio :
        Diameter-to-thickness ratio.

    Returns
    -------
    modulus_elastic :
        Elastic section modulus, the second moment over the outer radius.
    """
    return 2.0 * second_moment(diameter, ratio) / diameter


def modulus_plastic(
    diameter: Float[Array, "members"],
    ratio: float | Float[Array, ""],
) -> Float[Array, "members"]:
    """
    Plastic section modulus.

    Parameters
    ----------
    diameter :
        Outer diameter.
    ratio :
        Diameter-to-thickness ratio.

    Returns
    -------
    modulus_plastic :
        Plastic section modulus.
    """
    outer = jnp.asarray(diameter)
    bore = diameter_inner(diameter, ratio)

    return (outer**3 - bore**3) / 6.0
