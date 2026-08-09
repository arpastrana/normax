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
Cross-section classification of a circular hollow section, EN 1993-1-1 Table 5.2.

Tubular sections only, Table 5.2 sheet 3. The three limits are on the
diameter-to-thickness ratio and apply to sections in bending, in compression
or in both, so a single classification covers every axial case here.

The class is assembled by counting exceeded limits rather than by branching,
which keeps it usable under jit and vmap.
"""

import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

# EN 1993-1-1 Table 5.2 sheet 3. Multiply by epsilon SQUARED, not by epsilon.
CLASS_LIMIT_FACTORS = (50.0, 70.0, 90.0)

# Relative width of the inclusive bound of Table 5.2. A section pinned to a limit
# carries a wall thickness of the diameter over that limit, and recovering the
# ratio from the two returns it only to within rounding — measured at 1e-16
# relative, which a strict comparison turns into the class above. The tolerance is
# some ten thousand times that and some ten orders below any difference in `d/t`
# a real section has, so it separates rounding from geometry and nothing else.
CLASS_LIMIT_TOLERANCE = 1e-12


def epsilon(f_y: float | Float[Array, ""]) -> Float[Array, ""]:
    """
    Material factor for the slenderness limits.

    Parameters
    ----------
    f_y :
        Yield strength.

    Returns
    -------
    epsilon :
        Material factor, one at the 235 reference grade and falling as the
        grade rises.

    Notes
    -----
    EN 1993-1-1 5.5.2, Table 5.2. Defined as the square root of 235 over the
    yield strength.
    """
    return jnp.sqrt(235.0 / jnp.asarray(f_y))


def class_limits(f_y: float | Float[Array, ""]) -> Float[Array, "3"]:
    """
    Slenderness limits separating the four cross-section classes.

    Parameters
    ----------
    f_y :
        Yield strength.

    Returns
    -------
    class_limits :
        Upper limits on the diameter-to-thickness ratio for Classes 1, 2 and 3,
        in that order. A ratio above the third limit is Class 4.

    Notes
    -----
    EN 1993-1-1 5.5.2, Table 5.2 sheet 3, tubular sections: 50, 70 and 90 times
    epsilon squared. Beyond the Class 3 limit the section is a shell and
    EN 1993-1-6 applies instead, which is outside the scope of this package.
    """
    factors = jnp.asarray(CLASS_LIMIT_FACTORS)

    return factors * epsilon(f_y) ** 2


def classify(
    ratio: Float[Array, "members"],
    f_y: float | Float[Array, ""],
) -> Int[Array, "members"]:
    """
    Cross-section class of a circular hollow section.

    Parameters
    ----------
    ratio :
        Diameter-to-thickness ratio.
    f_y :
        Yield strength.

    Returns
    -------
    cross_section_class :
        Class 1, 2, 3 or 4.

    Notes
    -----
    EN 1993-1-1 5.5.2, Table 5.2 sheet 3. The limits are stated as inclusive
    upper bounds, so a ratio sitting exactly on a limit takes the class below
    it. Counting the limits a ratio exceeds avoids branching on a traced value.

    That inclusive bound is applied to within `CLASS_LIMIT_TOLERANCE`. A section
    designed to sit on a limit reaches it only to within rounding once its ratio
    has been through a wall thickness and back, and a strict comparison would
    hand back the class above for half of a set of members that were all built to
    the same ratio. The widening is far below any difference in `d/t` between
    real sections, so it changes no classification that geometry decides.
    """
    limits = class_limits(f_y) * (1.0 + CLASS_LIMIT_TOLERANCE)
    exceeded = jnp.asarray(ratio)[..., None] > limits

    return 1 + jnp.sum(exceeded, axis=-1)
