# SPDX-License-Identifier: Apache-2.0
"""
What a size costs and how hard it works, and the shape of the block that
says so.
"""

import abc
from typing import NamedTuple

import equinox as eqx
from jaxtyping import Array
from jaxtyping import Float

from normax.analysis import MemberForces
from normax.sections import MemberSections
from normax.sections import TubeFamily


class MemberSizes(NamedTuple):
    """
    The sections of a design, and how hard each one is worked.

    Attributes
    ----------
    sections :
        The section of every member.
    utilization :
        Demand over resistance of every member under every load case, at
        these sections.

    Notes
    -----
    Nothing here names a clause's vocabulary. Reducing an analysis to what a
    standard reads is clause work, so each sizer applies its own reduction and
    the container carries only the answer.
    """

    sections: MemberSections
    utilization: Float[Array, "load_cases members"]


class AbstractMemberSizer(eqx.Module):
    """
    A design standard, read as a differentiable check and a sizing map.

    Attributes
    ----------
    family :
        The section family every member is drawn from, as bare geometry.

    Notes
    -----
    Built from a structure like the other two blocks, and the one place the
    shared contract costs something: a code check reads one member at a time
    and has nothing to settle from a connectivity.
    """

    family: eqx.AbstractVar[TubeFamily]

    @abc.abstractmethod
    def compute_utilization(
        self,
        diameters: Float[Array, "members"],
        forces: MemberForces,
        buckling_length: Float[Array, "members"],
    ) -> Float[Array, "load_cases members"]:
        """
        Check sizes the caller owns against this standard.

        Parameters
        ----------
        diameters :
            Outer diameter every member was given.
        forces :
            What every member carries under every load case.
        buckling_length :
            Length every member is assumed to buckle over.

        Returns
        -------
        utilization :
            Demand over resistance of every member under every load case — the
            differentiable constraint a search holds at or under one.
        """

    @abc.abstractmethod
    def __call__(
        self,
        forces: MemberForces,
        buckling_length: Float[Array, "members"],
    ) -> MemberSizes:
        """
        Size every member for every load case, each on its own.

        Parameters
        ----------
        forces :
            What every member carries under every load case.
        buckling_length :
            Length every member is assumed to buckle over.

        Returns
        -------
        sizes :
            The section each load case demands, and how hard it is worked —
            one wherever the size was free to move.
        """
