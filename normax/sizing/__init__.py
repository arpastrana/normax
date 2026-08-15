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
What a member sizer is, independent of the standard behind it.

This module holds the contract alone: the container a sizer returns and the
abstract block a pipeline calls. Every clause lives in a backend beside it —
`normax.sizing.ec3` today — and nothing about any clause is decided here.

No backend is re-exported. A call site names the standard it sizes against,
`from normax.sizing.ec3 import Ec3Sizer`, the way an analysis names its solver.
"""

import abc
from typing import NamedTuple

import equinox as eqx
from ec3x.section import Tube
from jaxtyping import Array
from jaxtyping import Float

from normax.analysis import MemberForces


class MemberSizes(NamedTuple):
    """
    What a code check decided, and what it is worth.

    Attributes
    ----------
    sections :
        The section every load case demands of every member on its own.
    utilization :
        Demand over resistance of every member under its own load case, **at the
        section that case demanded** rather than at any reconciled one.

    Notes
    -----
    **What the check read is not recorded here, and that is what keeps the
    contract standard-agnostic.** Reducing an analysis to the quantities a
    standard states — a design moment and an equivalent uniform moment factor,
    for EN 1993-1-1 — is clause work, and its product is different under a
    different standard. Each sizer applies its own reduction to the forces a
    design already carries, so no field of this container names any clause's
    vocabulary.

    **Reading the utilization as a verdict on a finished design is the mistake
    to avoid.** It is a diagonal rather than a matrix:
    entry *(i, m)* is member *m* under case *i* at the section **case i**
    demanded, so it is exactly one — that is what a fully-stressed size means —
    except where the catalogue minimum bound and a member is oversized for want
    of a smaller tube. It is the invariant of the sizing map, stored rather than
    re-derived.

    **It survives `normax.design.design_envelope` unchanged, and still refers to
    the per-case sizes.** Reconciling the cases collapses the sections and needs
    no standard; re-reading the reconciled section would, and is a separate
    question for a report to ask. Which case governs each member does *not*
    need it — that is an `argmax` over the sections demanded, since capacity is
    strictly increasing in the diameter.

    **Every load case is sized for separately, and nothing is combined here.** A
    member has one size and has to satisfy all of them, but reconciling that is
    smoothing rather than a clause and belongs above a block that implements a
    standard. `normax.design.design_envelope` is where it happens.
    """

    sections: Tube
    utilization: Float[Array, "load_cases members"]


class AbstractMemberSizer(eqx.Module):
    """
    A design standard, read as a map from what a member carries to how big it is.

    Notes
    -----
    A standard is a normative text rather than a solver. It has no derivatives,
    and it is ordinarily implemented as scalar branchy code returning verdicts;
    what makes it a block here is that it answers the inverted question — not
    "does this section pass" but "what section passes exactly" — which has a
    derivative wherever the resistance is monotone in the size.

    The two methods are the two questions a standard can be asked, and both are
    clause work. What a set of actions demands is the first; how hard a size
    that the block did not choose is working is the second, and a design is
    re-read through it after several load cases have been reconciled.

    Built from a structure like the other two, and that is the one place where
    the shared contract costs something rather than buying something: a code
    check reads one member at a time and knows nothing of connectivity, so it
    has nothing to settle. Saying so is the point rather than an omission.
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
            The section each load case demands, and how hard that section is
            worked — one wherever the size was free to move, below one where
            the catalogue minimum bound.
        """

    @abc.abstractmethod
    def utilization(
        self,
        diameters: Float[Array, "members"],
        forces: MemberForces,
        buckling_length: Float[Array, "members"],
    ) -> Float[Array, "load_cases members"]:
        """
        Re-read a finished design against the standard that sized it.

        Parameters
        ----------
        diameters :
            Outer diameter every member was given.
        forces :
            What every member carries under every load case, which the sizer
            reduces to its own standard's terms before checking.
        buckling_length :
            Length every member is assumed to buckle over.

        Returns
        -------
        utilization :
            Demand over resistance of every member under every load case.
        """
