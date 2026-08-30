# SPDX-License-Identifier: Apache-2.0
"""
The steel a member is cut from, as a mill certificate states it.

Nothing here names a standard. A partial factor and a buckling curve select
clauses, so they belong to the block that applies them — a sizer reads this
grade in its own standard's terms, and two sizers reading the same grade
disagree only about what they add to it, never about what the steel is.

That is what lets a pipeline, its experiments and its tests describe a material
without importing any standard's library.
"""

from typing import NamedTuple

from jaxtyping import Array
from jaxtyping import Float

# Eurocode 3 3.2.6, and every other steel standard's value too.
E_MODULUS = 210000.0

# Density of structural steel, in tonnes per cubic millimeter, so that a mass
# in tonnes follows from millimeters and newtons.
DENSITY = 7.85e-9


class SteelGrade(NamedTuple):
    """
    The steel as supplied, before any standard has read it.

    Attributes
    ----------
    f_y :
        Yield strength.
    f_u :
        Ultimate tensile strength.
    e_mod :
        Modulus of elasticity.
    density :
        Density.

    Notes
    -----
    **The strengths have no default, because a default is a grade chosen
    silently.** A caller names its grade — `Steel355()` — or
    states the strengths itself; the modulus and the density default because
    they are the same for every structural steel. Every field is a leaf, so a
    gradient may be taken with respect to any of them.

    **What is deliberately absent is everything a clause decides.** The partial
    factors are a safety format and the imperfection factor selects a buckling
    curve, so both belong to the standard that applies them; a sizer builds its
    own material record from this grade plus its standard's factors. The four
    fields here are the ones a certificate states, and they mean the same thing
    under every standard.
    """

    f_y: float | Float[Array, ""]
    f_u: float | Float[Array, ""]
    e_mod: float | Float[Array, ""] = E_MODULUS
    density: float | Float[Array, ""] = DENSITY


class Steel355(SteelGrade):
    """
    Grade S355 structural steel, as its certificate states it.

    Notes
    -----
    The nominal strengths of EN 10025 for the thicknesses this project runs
    at. The fields stay constructor arguments rather than being pinned,
    because a pytree round trip rebuilds the instance positionally.
    """

    def __new__(
        cls,
        f_y: float | Float[Array, ""] = 355.0,
        f_u: float | Float[Array, ""] = 490.0,
        e_mod: float | Float[Array, ""] = E_MODULUS,
        density: float | Float[Array, ""] = DENSITY,
    ) -> "Steel355":
        """
        Build the grade at its certificate values.
        """
        return super().__new__(cls, f_y, f_u, e_mod, density)
