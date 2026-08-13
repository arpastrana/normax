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
The vocabulary every block of the design pipeline shares.

A design is found by three blocks in a row: a form finder chooses the shape, a
frame analysis says what the members carry, and a code check says how big they
have to be. Each is written by different people, in a different language, and
differentiates in a different way. This module says what the three agree on, and
nothing whatsoever about how any of them computes.

**A block is built from a structure and then called, and the split is where the
topology lives.** Its constructor reads a structure and settles whatever that
particular software can settle from the connectivity, the supports and their
positions alone: a form finder wants connectivity matrices, a frame solver wants
an assembly and degree of freedom maps, a code check wants nothing at all. That
runs on the host, outside any traced call, and it may do arbitrary work —
compile an assembly, open a session against a service. What is left is a
function of design parameters and loads, which is what an optimizer calls.

That split is what keeps a topology out of an objective. A built block is a pure
function of things no optimizer varies, so rebuilding it per iterate is waste,
and for a traced backend it is worse than waste: building reads support flags in
Python, so doing it inside the trace is what stops a stage being jitted.

**Load cases arrive stacked, and a block decides what to do with them.** A
structure is form-found under one load case and checked against several, so only
the analysis and the check carry a load case axis. Handing a block all of them at
once lets a solver that can answer them together do so in one crossing, which is
what a Tesseract or a service reached over a network wants; a solver that cannot
loops internally and pays what it would have paid anyway.

Nothing here says a block must be traceable. A block may trace, may carry a
hand-written rule through `custom_vjp`, or may serialize its arguments and cross
a network — the contract is a signature and a derivative, not an implementation.
"""

import abc
from typing import NamedTuple

import equinox as eqx
from jaxtyping import Array
from jaxtyping import Float

from normax.ec3.actions import MemberActions


class DesignParameters(NamedTuple):
    """
    The quantities that vary between evaluations of the pipeline.

    Attributes
    ----------
    q :
        Force density of every member. Negative in compression.
    diameters :
        Outer diameter every member is analyzed with.
    sharpness :
        Sharpness of the envelope over load cases. None asks for the true
        largest size any of them demands.

    Notes
    -----
    The diameters set the stiffness the frame is analyzed with and not the
    resistance it is checked against, which the sizing block returns. They are
    therefore the previous outer iterate of a staggered coupling rather than a
    design variable in their own right, and an optimizer that varies the shape
    alone still has to supply them.

    **The sharpness travels with the parameters because it is annealed**, and a
    schedule that raised it by rebuilding the pipeline would compile a program
    per round. Reaching the objective as an array leaf instead makes every round
    the same program at a different value. It is a smoothing rather than a
    design variable all the same, and no gradient is taken with respect to it.
    """

    q: Float[Array, "members"]
    diameters: Float[Array, "members"]
    sharpness: Float[Array, ""] | None = None


class FormFoundShape(NamedTuple):
    """
    The geometry a form finder settles on.

    Attributes
    ----------
    xyz :
        Position of every node at equilibrium.

    Notes
    -----
    **The handoff downstream is a geometry and nothing else** — no prestress, no
    initial member forces, and no quantity derived from the coordinates. A frame
    analysis is given this, starts from an unstressed reference state, and finds
    its own axial forces; that those agree with the form finder's is a
    prediction that gets tested rather than an input that gets imposed.

    Everything a form finder could also report is recoverable from what it does:
    a member force is its force density times its length, and a length is
    `normax.structures.member_lengths` of these coordinates. Reporting either
    would be handing a caller arithmetic it can do, and inviting two stages to
    do it differently.

    One field, deliberately. What a form finder is for is the shape, and the
    container says so; it is a container rather than a bare array so that a form
    finder which one day settles more than coordinates has somewhere to put it.
    """

    xyz: Float[Array, "nodes 3"]


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
    **The load case axis is variadic, and that is what lets one container serve
    a solver and a block.** A solver answers one load case at a time and returns
    these fields without it; a block answers every case and returns the same
    fields with it, stacked by `stack_load_cases`. The two differ in rank and in
    nothing else, so a second container for the unstacked form would be this one
    with a line removed.

    Moments are given at the two ends rather than sampled along the span,
    because loads are applied at nodes alone and the moment therefore varies
    linearly in between. That is what makes the first row of EN 1993-1-1 Table
    B.3 exact here, and the reduction to a design moment belongs to the check
    rather than to the analysis.

    The axial force is one number per member and load case for the same reason:
    with no load along the span it does not vary, and the analysis is linear.
    """

    axial_force: Float[Array, "*load_cases members"]
    moment_major: Float[Array, "*load_cases members ends"]
    moment_minor: Float[Array, "*load_cases members ends"]


class MemberSizes(NamedTuple):
    """
    What a code check demands of every member, under every load case.

    Attributes
    ----------
    actions :
        The design actions the check read, every field carrying a leading load
        case axis.
    required :
        Diameter every load case demands of every member on its own.

    Notes
    -----
    **Every load case is sized for separately, and nothing is combined here.** A
    member has one size and has to satisfy all of them, but reconciling that is
    smoothing rather than a clause and belongs above a block that implements a
    standard.

    The actions are returned as well as consumed, because reducing two end
    moments to a design moment and a factor is itself a clause. A design that
    carries them can be checked again at a different set of sizes without
    analyzing anything a second time.
    """

    actions: MemberActions
    required: Float[Array, "load_cases members"]


class AbstractFormFinder(eqx.Module):
    """
    A parametrization of the shapes a structure may take in equilibrium.

    Notes
    -----
    Maps force densities and a load case to a geometry that carries that case
    without bending. Concrete form finders differ in which quantities they treat
    as independent, not in the mechanics they encode, which is why they share
    one shape and one call signature.

    Built from the structure it is to shape, and from nothing else that varies.
    """

    @abc.abstractmethod
    def __call__(
        self,
        q: Float[Array, "members"],
        loads: Float[Array, "nodes 3"],
    ) -> FormFoundShape:
        """
        Find the shape that carries a load case at given force densities.

        Parameters
        ----------
        q :
            Force density of every member. Negative in compression.
        loads :
            Force applied at every node.

        Returns
        -------
        shape :
            The geometry at equilibrium.
        """

    @abc.abstractmethod
    def member_lengths(
        self,
        xyz: Float[Array, "nodes 3"],
    ) -> Float[Array, "members"]:
        """
        Measure every member of a geometry this block could have produced.

        Parameters
        ----------
        xyz :
            Position of every node.

        Returns
        -------
        lengths :
            Distance between the two nodes of every member.

        Notes
        -----
        **The counterpart of the sizer's `mass_per_length`, and here for the
        same reason.** A total mass is `ρ Σ A L`: the section family supplies
        the `A` and only the block owning one can, and the connectivity supplies
        the `L` and this is the block that owns one. Neither is a clause or a
        solve, and neither can be answered by a caller holding a geometry alone.

        Asked of a geometry rather than reported beside one, because a shape is
        what a form finder is for and a length is arithmetic about a shape. A
        caller that wants both makes two calls and can see that it did.
        """


class AbstractFrameAnalyzer(eqx.Module):
    """
    An elastic analysis of a frame whose members bend as well as stretch.

    Notes
    -----
    A form finder hands over a geometry and nothing else: no prestress and no
    initial member forces. The frame is analyzed from an unstressed reference
    state, so it must deform before any internal force appears, and the axial
    forces that come back are the analysis's own product rather than a
    restatement of the force densities that shaped it.

    Members are beams and not bars, so the analysis also reports the bending a
    form finder could not see. That is the reason this block exists: the check
    downstream consumes moments, and a pin-jointed form finder has none to give.

    Built from the structure it is to analyze, and from the material and section
    family it is configured with. Everything a solver can assemble before a
    geometry is chosen is assembled there.
    """

    @abc.abstractmethod
    def __call__(
        self,
        shape: FormFoundShape,
        diameters: Float[Array, "members"],
        loads: Float[Array, "load_cases nodes 3"],
    ) -> MemberForces:
        """
        Analyze one geometry under every load case it is checked against.

        Parameters
        ----------
        shape :
            The geometry to analyze, from a form finder.
        diameters :
            Outer diameter of every member, setting the stiffness.
        loads :
            Force applied at every node in every load case.

        Returns
        -------
        forces :
            Axial force and both end moments, per load case and member.
        """


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
            The design actions read, and the diameter each load case demands.
        """

    @abc.abstractmethod
    def mass_per_length(
        self,
        diameters: Float[Array, "members"],
    ) -> Float[Array, "members"]:
        """
        Mass a member of a given size carries per unit of its length.

        Parameters
        ----------
        diameters :
            Outer diameter of every member.

        Returns
        -------
        mass_per_length :
            Mass per unit length of every member.

        Notes
        -----
        The one quantity a block is asked for that no clause decides. A section
        family fixes both the area and the material, so only the block that owns
        one can answer, and published section tables state exactly this. What is
        done with it — a total mass, an objective — is not the block's business,
        which is why it stops here rather than returning a mass.
        """

    @abc.abstractmethod
    def utilization(
        self,
        diameters: Float[Array, "members"],
        actions: MemberActions,
        buckling_length: Float[Array, "members"],
    ) -> Float[Array, "load_cases members"]:
        """
        Re-read a finished design against the standard that sized it.

        Parameters
        ----------
        diameters :
            Outer diameter every member was given.
        actions :
            The design actions to check against, every field carrying a leading
            load case axis.
        buckling_length :
            Length every member is assumed to buckle over.

        Returns
        -------
        utilization :
            Demand over resistance of every member under every load case.
        """
