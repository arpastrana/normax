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

# Classes this package implements. Beyond the third limit the tube is a shell and
# EN 1993-1-6 applies instead.
CLASSES_IMPLEMENTED = (1, 2, 3)

# Classes taking plastic section properties, EN 1993-1-1 Table 6.7.
PLASTIC_CLASSES = (1, 2)

# Relative width of the inclusive bound of Table 5.2. A section pinned to a limit
# carries a wall thickness of the diameter over that limit, and recovering the
# ratio from the two returns it only to within rounding — measured at 1e-16
# relative, which a strict comparison turns into the class above. The tolerance is
# some ten thousand times that and some ten orders below any difference in `d/t`
# a real section has, so it separates rounding from geometry and nothing else.
CLASS_LIMIT_TOLERANCE = 1e-12


def material_factor(f_y: float | Float[Array, ""]) -> Float[Array, ""]:
    """
    Material factor for the slenderness limits.

    Parameters
    ----------
    f_y :
        Yield strength.

    Returns
    -------
    material_factor :
        Material factor epsilon, one at the 235 reference grade and falling as
        the grade rises.

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

    return factors * material_factor(f_y) ** 2


def is_plastic(section_class: int) -> bool:
    """
    Whether a cross-section class takes plastic section properties.

    Parameters
    ----------
    section_class :
        Class 1, 2 or 3.

    Returns
    -------
    plastic :
        True for Classes 1 and 2.

    Raises
    ------
    ValueError
        If the class is not 1, 2 or 3.

    Notes
    -----
    EN 1993-1-1 Table 6.7, the characteristic resistances by class. This
    selects the section modulus, the cross-section clause and the column of
    Table B.1, so it is a build-time choice and never a traced value.

    Class 4 is refused rather than reported as elastic. It takes effective
    section properties under 6.2.2.5, which this package does not implement, and
    answering False would run Class 3's clauses on a shell.
    """
    _validate_class(section_class)

    return section_class in PLASTIC_CLASSES


def _validate_class(section_class: int) -> None:
    """
    Refuse a class this package does not implement.

    Raises
    ------
    ValueError
        If the class is not 1, 2 or 3.
    """
    if section_class not in CLASSES_IMPLEMENTED:
        raise ValueError(
            f"class must be 1, 2 or 3, not {section_class}; "
            "beyond the Class 3 limit EN 1993-1-6 applies"
        )


def ratio_at_class_limit(
    f_y: float | Float[Array, ""],
    section_class: int,
) -> Float[Array, ""]:
    """
    Slenderness limit of one cross-section class.

    Parameters
    ----------
    f_y :
        Yield strength.
    section_class :
        Class 1, 2 or 3.

    Returns
    -------
    ratio :
        Largest diameter-to-thickness ratio still inside that class.

    Raises
    ------
    ValueError
        If the class is not 1, 2 or 3.

    Notes
    -----
    EN 1993-1-1 Table 5.2 sheet 3, one row of `class_limits`. The class is a
    Python integer selecting a row rather than a traced value, so refusing
    Class 4 here is an ordinary exception.

    Inverse of `section_class_at_ratio`, and the round trip through either order
    is exact to within the inclusive bound the classification applies.
    """
    _validate_class(section_class)

    return class_limits(f_y)[section_class - 1]


def classify_section(
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
    section_class :
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


def section_class_at_ratio(
    ratio: float | Float[Array, ""],
    f_y: float | Float[Array, ""],
) -> int:
    """
    Cross-section class of one ratio, as a value that can select a clause.

    Parameters
    ----------
    ratio :
        Diameter-to-thickness ratio, a single value rather than one per member.
    f_y :
        Yield strength.

    Returns
    -------
    section_class :
        Class 1, 2 or 3, as a Python integer.

    Raises
    ------
    ValueError
        If the ratio is not a single value, or if it classifies as Class 4.

    Notes
    -----
    EN 1993-1-1 Table 5.2 sheet 3, and the inverse of `ratio_at_class_limit`.
    `classify_section` is the same clause for a whole set of members and returns
    an array; this returns an integer, which is what a clause selector has to be.

    A traced ratio has no integer value, so this refuses to run inside a jitted
    or differentiated function. That is the honest failure: the class chooses
    between clauses, and a derivative with respect to the ratio cannot move it.
    """
    if jnp.shape(ratio) != ():
        raise ValueError(
            f"the class selects a clause, so it takes one ratio, not "
            f"{jnp.shape(ratio)}; a section family has a single ratio"
        )

    section_class = int(classify_section(ratio, f_y))
    _validate_class(section_class)

    return section_class
