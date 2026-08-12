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
The steel a member is cut from, and the partial factors applied to it.

A leaf. Nothing here computes a resistance or reads a geometry, which is what
lets every other module in the package take a grade without importing the one
that sizes members.

The constants live beside the container that defaults from them. Evaluating a
default runs when the class body does, so a grade defined apart from its
constants would close an import cycle that no deferred import could open.
"""

from typing import NamedTuple

from jaxtyping import Array
from jaxtyping import Float

# EN 1993-1-1 6.1. Nationally determined; these are the values the UK National
# Annex sets in clause NA.2.15.
GAMMA_M0 = 1.0
GAMMA_M1 = 1.0
GAMMA_M2 = 1.25

# EN 1993-1-1 3.2.6.
E_MODULUS = 210000.0

# Density of structural steel, in tonnes per cubic millimeter, so that a mass
# in tonnes follows from millimeters and newtons.
DENSITY = 7.85e-9

# EN 1993-1-1 Table 6.1. Table 6.2 selects the curve: hollow sections are
# curve a hot finished and curve c cold formed, and a0 or c respectively at
# the 460 grade.
IMPERFECTION_FACTORS = {
    "a0": 0.13,
    "a": 0.21,
    "b": 0.34,
    "c": 0.49,
    "d": 0.76,
}


class SteelGrade(NamedTuple):
    """
    The steel as supplied, and the partial factors applied to it.

    Attributes
    ----------
    f_y :
        Yield strength.
    e_mod :
        Modulus of elasticity.
    density :
        Density.
    gamma_m0 :
        Partial factor for cross-section resistance.
    gamma_m1 :
        Partial factor for member instability.
    f_u :
        Ultimate tensile strength.
    gamma_m2 :
        Partial factor for resistance to fracture in tension.
    alpha :
        Imperfection factor of the buckling curve, EN 1993-1-1 Table 6.1.

    Notes
    -----
    The defaults are S355 with the partial factors of the UK National Annex,
    clause NA.2.15. Every field is a leaf, so a gradient may be taken with
    respect to any of them.

    **The imperfection factor records a fabrication route, not a chemistry.**
    EN 1993-1-1 Table 6.2 gives a hot-finished hollow section curve a and a
    cold-formed one curve c, so the same grade drawn two ways is two values of
    this container differing in that field alone. It sits here because the
    buckling curve follows the steel rather than the shape, and because a
    circular hollow section fixes its shape by one ratio and nothing else.
    """

    f_y: float | Float[Array, ""] = 355.0
    e_mod: float | Float[Array, ""] = E_MODULUS
    density: float | Float[Array, ""] = DENSITY
    gamma_m0: float | Float[Array, ""] = GAMMA_M0
    gamma_m1: float | Float[Array, ""] = GAMMA_M1
    f_u: float | Float[Array, ""] = 490.0
    gamma_m2: float | Float[Array, ""] = GAMMA_M2
    alpha: float | Float[Array, ""] = IMPERFECTION_FACTORS["a"]
