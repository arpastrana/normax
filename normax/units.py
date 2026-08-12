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

The standard tabulates lengths in millimeters and strengths in newtons per
square millimeter, and every module under `normax.ec3` follows it, carrying
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

# Meters in a millimeter.
MILLIMETER = 1e-3

# Pascals in a newton per square millimeter. The two are the megapascal.
MEGAPASCAL = 1e6

# Newton meters in a newton millimeter.
NEWTON_MILLIMETER = 1e-3

# Kilograms per cubic meter in a tonne per cubic millimeter.
TONNE_PER_CUBIC_MILLIMETER = 1e12

# A converted quantity is a scalar material property as readily as an array of
# member lengths, so these functions fix no shape.
Quantity = float | Float[Array, "..."]


def to_meters(length: Quantity) -> Quantity:
    """
    Convert a length from millimeters to meters.

    Parameters
    ----------
    length :
        Length in millimeters.

    Returns
    -------
    length :
        The same length in meters.
    """
    return length * MILLIMETER


def to_millimeters(length: Quantity) -> Quantity:
    """
    Convert a length from meters to millimeters.

    Parameters
    ----------
    length :
        Length in meters.

    Returns
    -------
    length :
        The same length in millimeters.
    """
    return length / MILLIMETER


def to_pascals(stress: Quantity) -> Quantity:
    """
    Convert a stress from newtons per square millimeter to pascals.

    Parameters
    ----------
    stress :
        Stress in newtons per square millimeter.

    Returns
    -------
    stress :
        The same stress in pascals.
    """
    return stress * MEGAPASCAL


def to_newtons_per_square_millimeter(stress: Quantity) -> Quantity:
    """
    Convert a stress from pascals to newtons per square millimeter.

    Parameters
    ----------
    stress :
        Stress in pascals.

    Returns
    -------
    stress :
        The same stress in newtons per square millimeter.
    """
    return stress / MEGAPASCAL


def to_newton_meters(moment: Quantity) -> Quantity:
    """
    Convert a moment from newton millimeters to newton meters.

    Parameters
    ----------
    moment :
        Moment in newton millimeters.

    Returns
    -------
    moment :
        The same moment in newton meters.
    """
    return moment * NEWTON_MILLIMETER


def to_newton_millimeters(moment: Quantity) -> Quantity:
    """
    Convert a moment from newton meters to newton millimeters.

    Parameters
    ----------
    moment :
        Moment in newton meters.

    Returns
    -------
    moment :
        The same moment in newton millimeters.
    """
    return moment / NEWTON_MILLIMETER


def to_kilograms_per_cubic_meter(density: Quantity) -> Quantity:
    """
    Convert a density from tonnes per cubic millimeter to kilograms per cubic meter.

    Parameters
    ----------
    density :
        Density in tonnes per cubic millimeter.

    Returns
    -------
    density :
        The same density in kilograms per cubic meter.
    """
    return density * TONNE_PER_CUBIC_MILLIMETER


def to_tonnes_per_cubic_millimeter(density: Quantity) -> Quantity:
    """
    Convert a density from kilograms per cubic meter to tonnes per cubic millimeter.

    Parameters
    ----------
    density :
        Density in kilograms per cubic meter.

    Returns
    -------
    density :
        The same density in tonnes per cubic millimeter.
    """
    return density / TONNE_PER_CUBIC_MILLIMETER
