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

**What a backend returns is `normax.stages.MemberForces`, one load case of it.**
A solver answers one case at a time and a block answers all of them, and the two
are the same container at two ranks rather than two containers; `stack_load_cases`
is what puts them together. A load case reaches a backend as the dense nodal
array of `normax.loads`, which is the one thing two backends cannot disagree
about.

All lengths, forces and stresses cross the boundary through `normax.units`.
"""

from typing import NamedTuple

import numpy as np
from jaxtyping import Array
from jaxtyping import Bool
from jaxtyping import Float

from normax.structures import Structure

# Three translations and three rotations: what a node of a frame in space has,
# and the width of a fixity row whichever solver reads it.
DOF_PER_NODE = 6


class Buckling(NamedTuple):
    """
    The elastic instability of a whole frame, rather than of one member.

    Attributes
    ----------
    factors :
        Multiple of the applied load at which the frame buckles, smallest first.
    shapes :
        Displacement of every node in each mode, in the six degrees of freedom.

    Notes
    -----
    The smallest factor is what EN 1993-1-1 writes as `α_cr`. A value below one
    says the frame becomes unstable before it is loaded to its design value, and
    a member check cannot see that: it reads the slenderness of one member over
    an assumed buckling length, while this reads the mode the whole structure
    has.

    Mode shapes are not normalized to any physical amplitude. An eigenvector
    fixes a shape and not a size, so only ratios within a mode mean anything.
    """

    factors: Float[Array, "modes"]
    shapes: Float[Array, "modes nodes 6"]


def support_fixities(
    structure: Structure,
    normal: int | None,
) -> Bool[np.ndarray, "nodes 6"]:
    """
    Which degrees of freedom are restrained at every node.

    Parameters
    ----------
    structure :
        The structure whose supported nodes are to be restrained.
    normal :
        Index of the global axis a planar structure has no thickness along, or
        None for a structure that occupies all three dimensions.

    Returns
    -------
    fixities :
        A flag per node and degree of freedom, ordered as translations then
        rotations.

    Raises
    ------
    ValueError
        If the normal axis is not 0, 1 or 2.

    Notes
    -----
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

    A solver working in the plane itself has no such mode to remove, so it reads
    the support columns and ignores the normal one.
    """
    if normal is not None and normal not in (0, 1, 2):
        raise ValueError(f"normal must be 0, 1 or 2, or None, got {normal}")

    num_nodes = structure.num_nodes

    flags = np.zeros((num_nodes, DOF_PER_NODE), dtype=bool)
    flags[np.asarray(structure.supports), :3] = True

    if normal is not None:
        flags[:, normal] = True
        rotations = [DOF_PER_NODE // 2 + axis for axis in (0, 1, 2) if axis != normal]
        flags[np.ix_(np.asarray(structure.supports), rotations)] = True

    return flags
