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

# EN 1993-1-1 3.2.6, and every other steel standard's value too.
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
    The defaults are S355. Every field is a leaf, so a gradient may be taken
    with respect to any of them.

    **What is deliberately absent is everything a clause decides.** The partial
    factors are a safety format and the imperfection factor selects a buckling
    curve, so both belong to the standard that applies them; a sizer builds its
    own material record from this grade plus its standard's factors. The four
    fields here are the ones a certificate states, and they mean the same thing
    under every standard.
    """

    f_y: float | Float[Array, ""] = 355.0
    f_u: float | Float[Array, ""] = 490.0
    e_mod: float | Float[Array, ""] = E_MODULUS
    density: float | Float[Array, ""] = DENSITY
