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
Frame analysis of a form-found geometry, the second stage of the pipeline.

Form finding hands over a geometry and nothing else: no prestress, no initial
member forces. The frame is analyzed from an unstressed reference state, so it
must deform elastically before any internal force appears, and the axial forces
that come back are the analysis's own product rather than a restatement of the
force densities that shaped it. Their agreement is a prediction, and it is what
`tests/test_equilibrium_consistency.py` measures.

Members are beams, not bars, so the analysis also returns the bending the
form-finder could not see. That is the reason this stage exists: the code check
downstream consumes moments, and a pin-jointed form-finder has none to give.

**This module says what the stage means; a backend beside it says how a solver
computes it.** `normax.analysis.smax` traces a JAX frame solver in three
dimensions and `normax.analysis.opensees` drives a C++ one in two. Nothing here
imports either, so the contracts below are readable without a solver installed
and neither backend inherits the other's dependencies.

**Every backend is reached in two calls, and the split is where the topology
lives.** `prepare_model` reads a structure and returns a model: whatever that
solver can work out from the connectivity and the supports alone, built on the
host and outside any traced call. `member_forces` then takes that model with a
geometry and a set of sizes and returns what the members carry. The model is
opaque and belongs to the backend, so a solver that can precompute an assembly
holds one and a solver that must rebuild its domain per call holds only what it
needs to do that.

The split is what keeps a topology out of the objective. A model is a pure
function of things no optimizer varies, so recomputing it per iterate is waste,
and for a traced backend it is worse than waste: compilation reads support flags
in Python, so a rebuild inside the trace is what stops the stage being jitted.

**What a backend returns is `normax.design.MemberForces`, one load case of it.**
A solver answers one case at a time and a block answers all of them, and the two
are the same container at two ranks rather than two containers; `stack_load_cases`
is what puts them together. A load case reaches a backend as the dense nodal
array of `normax.loads`, which is the one thing two backends cannot disagree
about.

All lengths, forces and stresses cross the boundary through `normax.units`.
"""

import abc
from typing import NamedTuple

import equinox as eqx
import numpy as np
from jaxtyping import Array
from jaxtyping import Bool
from jaxtyping import Float

from normax.structures import Structure

# Three translations and three rotations: what a node of a frame in space has,
# and the width of a fixity row whichever solver reads it.
DOF_PER_NODE = 6


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

    **It takes coordinates rather than a form-found shape.** A frame analysis
    needs a geometry and has no use for anything else a form finder settled, so
    asking for the container would make this contract unreadable without the
    form finder's own dependencies for no gain.
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


def normal_axis(structure: Structure) -> int | None:
    """
    The global axis a structure has no thickness along, if it has one.

    Parameters
    ----------
    structure :
        The structure whose starting geometry is measured.

    Returns
    -------
    normal :
        Index of the axis every node shares a coordinate along, or None for a
        structure that occupies all three dimensions.

    Notes
    -----
    Read from the geometry rather than declared beside it. Whether a structure
    is planar is a fact about the structure, and a caller asked to restate it
    can only restate it wrongly — a plane named for a structure that does not
    lie in one is a projection the solver would answer anyway.

    The starting geometry is what is measured, so the answer is a static Python
    value and may pick out degrees of freedom. Form finding moves nodes within
    the plane the generators lay out and never out of it, so a form-found
    geometry is planar wherever the guess was.

    **A straight structure is planar in two ways, and only one restraint may be
    taken.** Restraining both axes it is thin along would leave it unable to
    bend at all. The axes are tried as Y, then X, then Z, so the one restrained
    is horizontal wherever there is a choice and the vertical plane a structure
    is loaded in stays free.
    """
    nodes = np.asarray(structure.nodes)

    for axis in (1, 0, 2):
        offsets = nodes[:, axis]
        if np.allclose(offsets, offsets[0]):
            return axis

    return None


def support_fixities(structure: Structure) -> Bool[np.ndarray, "nodes 6"]:
    """
    Which degrees of freedom are restrained at every node.

    Parameters
    ----------
    structure :
        The structure whose supported nodes are to be restrained.

    Returns
    -------
    fixities :
        A flag per node and degree of freedom, ordered as translations then
        rotations.

    Raises
    ------
    ValueError
        If the structure names no supports, which no fixity can stand in for.

    Notes
    -----
    **What a structure asks for is pinned supports, and what a solver needs is
    sometimes more than that.** The difference is worked out here rather than
    handed in: the plane is measured with `normal_axis` and the restraints that
    a three-dimensional solve of a planar structure additionally needs are
    added. Only the supports themselves cannot be supplied, a structure held
    nowhere being under-determined in a way no fixity describes.

    **Pinned and never fixed is a rule about structures that occupy all three
    dimensions.** There a support restrains translation and leaves every rotation
    free, so a base carries no moment: form finding restrains translation and
    nothing else, and a fixed base would inject end moments the form-finder never
    saw.

    **A planar structure deviates from that rule, and has to.** Analyzed by a
    three-dimensional solver it is a mechanism — rotating the whole of it about
    the line joining its supports strains no member and moves no support, so the
    stiffness matrix is singular and the solve returns nan rather than a
    plausible wrong answer. Restraining the one translation normal to the plane,
    at every node, removes that mode.

    **A straight planar structure needs more than that.** Rotating a beam about
    its own axis moves no node at all, its members lying on the line joining the
    supports, so the normal translation never engages and the mode survives as a
    uniform twist. The two rotations out of the plane are restrained at the
    supports to remove it, which is what a bearing does. What is never restrained
    anywhere is the rotation the in-plane bending happens about, so a base still
    carries no moment and the in-plane results are unchanged.

    Those two rotations are restrained wherever the structure is planar, straight
    or not. In-plane and out-of-plane response decouple exactly in a planar
    frame, so a restraint the curved case does not need cannot reach its results
    either.

    A solver working in the plane itself has no such mode to remove, so it reads
    the support columns and ignores the normal one.
    """
    supports = np.asarray(structure.supports)
    if supports.size == 0:
        raise ValueError("a structure with no supports cannot be analyzed")

    flags = np.zeros((structure.num_nodes, DOF_PER_NODE), dtype=bool)
    flags[supports, :3] = True

    normal = normal_axis(structure)
    if normal is not None:
        flags[:, normal] = True
        rotations = [DOF_PER_NODE // 2 + axis for axis in (0, 1, 2) if axis != normal]
        flags[np.ix_(supports, rotations)] = True

    return flags
