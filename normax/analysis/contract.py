# SPDX-License-Identifier: Apache-2.0
"""
What a frame analysis reports, and the shape of the block that reports it.
"""

import abc
from typing import NamedTuple

import equinox as eqx
from jaxtyping import Array
from jaxtyping import Float


class MemberForces(NamedTuple):
    """
    The internal forces a frame analysis reports.

    Attributes
    ----------
    axial_force :
        Axial force of every member, tension positive.
    moment_major :
        Bending moment about the major axis, at each end of every member.
    moment_minor :
        Bending moment about the minor axis, at each end of every member.

    Notes
    -----
    The load case axis is variadic: a solver answers one case and returns
    these fields without it, a block answers every case and returns them
    stacked. Moments are given at the two ends because loads are applied at
    nodes alone, so the moment varies linearly in between.
    """

    axial_force: Float[Array, "*load_cases members"]
    moment_major: Float[Array, "*load_cases members ends"]
    moment_minor: Float[Array, "*load_cases members ends"]


class AbstractFrameAnalyzer(eqx.Module):
    """
    An elastic analysis of a frame whose members bend as well as stretch.

    Notes
    -----
    Built from the structure it is to analyze and the section it is configured
    with; everything a solver can assemble before a geometry is chosen is
    assembled there. Takes coordinates rather than a form-found shape, since a
    frame analysis needs a geometry and nothing else a form finder settled.
    """

    @abc.abstractmethod
    def __call__(
        self,
        xyz: Float[Array, "nodes 3"],
        diameters: Float[Array, "members"],
        loads: Float[Array, "load_cases nodes 3"],
    ) -> MemberForces:
        """
        Analyze one geometry under every load case it is checked against.

        Parameters
        ----------
        xyz :
            Position of every node, from a form finder.
        diameters :
            Outer diameter of every member, setting the stiffness.
        loads :
            Force applied at every node in every load case.

        Returns
        -------
        forces :
            Axial force and both end moments, per load case and member.
        """
