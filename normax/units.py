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
Conversions between the units EN 1993-1-1 is written in and coherent SI.

The standard tabulates lengths in millimetres and strengths in newtons per
square millimetre, and every module under `normax.ec3` follows it, carrying
masses in tonnes so that the density is a plain number. A general-purpose
mechanics solver works in coherent SI instead. The two systems meet here, in
one place, rather than at every call across the boundary.

Force is the newton in both systems and needs no conversion. That is the one
quantity crossing the boundary untouched, and its absence from this module is
deliberate.

Each constant is the value of one `normax` unit expressed in its SI
counterpart, so a conversion into SI multiplies and a conversion out divides.
"""

from jaxtyping import Array
from jaxtyping import Float

# Metres in a millimetre.
MILLIMETRE = 1e-3

# Pascals in a newton per square millimetre. The two are the megapascal.
MEGAPASCAL = 1e6

# Newton metres in a newton millimetre.
NEWTON_MILLIMETRE = 1e-3

# Kilograms per cubic metre in a tonne per cubic millimetre.
TONNE_PER_CUBIC_MILLIMETRE = 1e12

# A converted quantity is a scalar material property as readily as an array of
# member lengths, so these functions fix no shape.
Quantity = float | Float[Array, "..."]


def to_metres(length: Quantity) -> Quantity:
    """
    Convert a length from millimetres to metres.

    Parameters
    ----------
    length :
        Length in millimetres.

    Returns
    -------
    length :
        The same length in metres.
    """
    return length * MILLIMETRE


def to_millimetres(length: Quantity) -> Quantity:
    """
    Convert a length from metres to millimetres.

    Parameters
    ----------
    length :
        Length in metres.

    Returns
    -------
    length :
        The same length in millimetres.
    """
    return length / MILLIMETRE


def to_pascals(stress: Quantity) -> Quantity:
    """
    Convert a stress from newtons per square millimetre to pascals.

    Parameters
    ----------
    stress :
        Stress in newtons per square millimetre.

    Returns
    -------
    stress :
        The same stress in pascals.
    """
    return stress * MEGAPASCAL


def to_newtons_per_square_millimetre(stress: Quantity) -> Quantity:
    """
    Convert a stress from pascals to newtons per square millimetre.

    Parameters
    ----------
    stress :
        Stress in pascals.

    Returns
    -------
    stress :
        The same stress in newtons per square millimetre.
    """
    return stress / MEGAPASCAL


def to_newton_metres(moment: Quantity) -> Quantity:
    """
    Convert a moment from newton millimetres to newton metres.

    Parameters
    ----------
    moment :
        Moment in newton millimetres.

    Returns
    -------
    moment :
        The same moment in newton metres.
    """
    return moment * NEWTON_MILLIMETRE


def to_newton_millimetres(moment: Quantity) -> Quantity:
    """
    Convert a moment from newton metres to newton millimetres.

    Parameters
    ----------
    moment :
        Moment in newton metres.

    Returns
    -------
    moment :
        The same moment in newton millimetres.
    """
    return moment / NEWTON_MILLIMETRE


def to_kilograms_per_cubic_metre(density: Quantity) -> Quantity:
    """
    Convert a density from tonnes per cubic millimetre to kilograms per cubic metre.

    Parameters
    ----------
    density :
        Density in tonnes per cubic millimetre.

    Returns
    -------
    density :
        The same density in kilograms per cubic metre.
    """
    return density * TONNE_PER_CUBIC_MILLIMETRE


def to_tonnes_per_cubic_millimetre(density: Quantity) -> Quantity:
    """
    Convert a density from kilograms per cubic metre to tonnes per cubic millimetre.

    Parameters
    ----------
    density :
        Density in kilograms per cubic metre.

    Returns
    -------
    density :
        The same density in tonnes per cubic millimetre.
    """
    return density / TONNE_PER_CUBIC_MILLIMETRE
