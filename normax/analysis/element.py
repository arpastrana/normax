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
The frame element, so a solver that carries no derivative can borrow one.

A foreign solver assembles, factorizes and solves; what it cannot do is say how
its answer moves. An adjoint needs one thing from the element level — the
derivative of the global element stiffness with respect to the geometry and the
section — and the way to obtain it exactly is to state the element here, in JAX,
and prove it is the same element the solver assembled.

**The proof is a test, not a claim.** `stiffness_local` and `stiffness_global`
are held against the foreign solver's own matrices to near machine precision, so
differentiating this module differentiates that solver's model rather than a
lookalike. Nothing here is approximate and no step size enters.

Two properties of a circular hollow section are what make this short, and both
were measured rather than assumed.

**One bending rigidity, not two.** A doubly symmetric tube has equal second
moments about both transverse axes, so the element takes a single bending term
and the two directions cannot disagree.

**The global element stiffness does not know how the frame was rolled.** Turning
a member's transverse axes about its own axis leaves the global matrix
unchanged — exactly, to a part in ten thousand trillion, when the two second
moments are equal. That is why the transverse basis below may be chosen for
conditioning alone and needs to match no other convention: two solvers that
orient a tube differently still assemble the same stiffness.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

# Translations and rotations at both ends of one member.
DOF_PER_MEMBER = 12

# How many end blocks the direction cosines repeat over.
BLOCKS_PER_MEMBER = 4

# What the transverse axes are completed against: the repository's vertical.
REFERENCE_AXIS = 2

# How square to the vertical a member must stay for that reference to hold.
REFERENCE_MARGIN = 1.0e-6


class SectionRigidity(NamedTuple):
    """
    What the element stiffness depends on, once the section has been read.

    Attributes
    ----------
    axial :
        Product of the elastic modulus and the area.
    bending :
        Product of the elastic modulus and the second moment, one value for
        both transverse directions because the section is axisymmetric.
    torsional :
        Product of the shear modulus and the torsion constant.

    Notes
    -----
    The element stiffness is exactly linear in each of these three, which is why
    a derivative with respect to a diameter needs no perturbation: the chain
    rule closes on the section geometry alone.
    """

    axial: Float[Array, "*members"]
    bending: Float[Array, "*members"]
    torsional: Float[Array, "*members"]


def stiffness_local(
    length: Float[Array, ""],
    rigidity: SectionRigidity,
) -> Float[Array, "dofs_member dofs_member"]:
    """
    Elastic stiffness of one member, about its own axes.

    Parameters
    ----------
    length :
        Distance between the member's ends.
    rigidity :
        The three products of a modulus and a section property.

    Returns
    -------
    stiffness :
        Twelve by twelve, ordered translations then rotations at the first end,
        then the same at the second.

    Notes
    -----
    Bernoulli–Euler: no shear area enters, so a deep member is stiffer here than
    it is in reality. That matches the foreign solver this stands in for, which
    is the property that matters — an adjoint must differentiate the model that
    was solved, not a better one.
    """
    axial = rigidity.axial / length
    torsion = rigidity.torsional / length
    shear = 12.0 * rigidity.bending / length**3
    coupling = 6.0 * rigidity.bending / length**2
    bending_near = 4.0 * rigidity.bending / length
    bending_far = 2.0 * rigidity.bending / length
    empty = jnp.zeros_like(axial)

    block = [
        [
            axial,
            empty,
            empty,
            empty,
            empty,
            empty,
            -axial,
            empty,
            empty,
            empty,
            empty,
            empty,
        ],
        [
            empty,
            shear,
            empty,
            empty,
            empty,
            coupling,
            empty,
            -shear,
            empty,
            empty,
            empty,
            coupling,
        ],
        [
            empty,
            empty,
            shear,
            empty,
            -coupling,
            empty,
            empty,
            empty,
            -shear,
            empty,
            -coupling,
            empty,
        ],
        [
            empty,
            empty,
            empty,
            torsion,
            empty,
            empty,
            empty,
            empty,
            empty,
            -torsion,
            empty,
            empty,
        ],
        [
            empty,
            empty,
            -coupling,
            empty,
            bending_near,
            empty,
            empty,
            empty,
            coupling,
            empty,
            bending_far,
            empty,
        ],
        [
            empty,
            coupling,
            empty,
            empty,
            empty,
            bending_near,
            empty,
            -coupling,
            empty,
            empty,
            empty,
            bending_far,
        ],
        [
            -axial,
            empty,
            empty,
            empty,
            empty,
            empty,
            axial,
            empty,
            empty,
            empty,
            empty,
            empty,
        ],
        [
            empty,
            -shear,
            empty,
            empty,
            empty,
            -coupling,
            empty,
            shear,
            empty,
            empty,
            empty,
            -coupling,
        ],
        [
            empty,
            empty,
            -shear,
            empty,
            coupling,
            empty,
            empty,
            empty,
            shear,
            empty,
            coupling,
            empty,
        ],
        [
            empty,
            empty,
            empty,
            -torsion,
            empty,
            empty,
            empty,
            empty,
            empty,
            torsion,
            empty,
            empty,
        ],
        [
            empty,
            empty,
            -coupling,
            empty,
            bending_far,
            empty,
            empty,
            empty,
            coupling,
            empty,
            bending_near,
            empty,
        ],
        [
            empty,
            coupling,
            empty,
            empty,
            empty,
            bending_far,
            empty,
            -coupling,
            empty,
            empty,
            empty,
            bending_near,
        ],
    ]
    stiffness = jnp.asarray(block)

    return stiffness


def stiffness_frame(
    start: Float[Array, "3"],
    end: Float[Array, "3"],
) -> Float[Array, "3 3"]:
    """
    Any orthonormal frame on a member, chosen for conditioning alone.

    Parameters
    ----------
    start :
        Position of the member's first end.
    end :
        Position of the member's second end.

    Returns
    -------
    frame :
        Rows are the member axis and two transverse axes, in that order.

    Notes
    -----
    **The choice carries no meaning and nothing may read it.** It exists only to
    build the global element stiffness, which is invariant to a roll about the
    member axis when the section is axisymmetric — so this returns whichever
    frame is best conditioned rather than whichever one a convention would
    prefer. The transverse pair is completed against the global axis the member
    leans on least, which is what keeps the cross product away from zero for
    every orientation, vertical members included.

    `member_frame` is the one to read: it answers to a stated convention and
    turns smoothly, at the price of degenerating for a vertical member.
    """
    span = end - start
    axis = span / jnp.linalg.norm(span)
    leaning = jnp.argmin(jnp.abs(axis))
    reference = jnp.eye(3)[leaning]
    crossed = jnp.cross(axis, reference)
    first = crossed / jnp.linalg.norm(crossed)
    second = jnp.cross(axis, first)
    frame = jnp.stack([axis, first, second])

    return frame


def member_frame(
    start: Float[Array, "3"],
    end: Float[Array, "3"],
) -> Float[Array, "3 3"]:
    """
    Direction cosines carrying global components onto one member's own axes.

    Parameters
    ----------
    start :
        Position of the member's first end.
    end :
        Position of the member's second end.

    Returns
    -------
    frame :
        Rows are the member axis and the two transverse axes, in that order.

    Notes
    -----
    The transverse pair is completed against the vertical, which is the ordinary
    structural convention and, unlike a per-member choice, turns smoothly as the
    geometry moves — a member that leans further does not suddenly report its
    bending about a different axis. **It degenerates for a vertical member**,
    where the cross product vanishes and the pair is undefined; a caller that
    can present one is expected to refuse it, `REFERENCE_MARGIN` being how
    square to the vertical a member has to stay.

    The global element stiffness is invariant to a roll about the member axis
    when the section is axisymmetric, so this convention has to agree with no
    other one, and no other backend's components can be compared against it.
    What survives the choice is the pair's invariants — the bending magnitude
    and the angle between the two ends — and those are what a check of an
    axisymmetric section reads.
    """
    span = end - start
    axis = span / jnp.linalg.norm(span)
    reference = jnp.eye(3)[REFERENCE_AXIS]
    crossed = jnp.cross(axis, reference)
    first = crossed / jnp.linalg.norm(crossed)
    second = jnp.cross(axis, first)
    frame = jnp.stack([axis, first, second])

    return frame


def member_transform(
    frame: Float[Array, "3 3"],
) -> Float[Array, "dofs_member dofs_member"]:
    """
    The direction cosines repeated over a member's four end blocks.

    Parameters
    ----------
    frame :
        Direction cosines of one member.

    Returns
    -------
    transform :
        Block diagonal, carrying twelve global components onto twelve local ones.
    """
    repeats = jnp.eye(BLOCKS_PER_MEMBER)
    transform = jnp.kron(repeats, frame)

    return transform


def stiffness_global(
    start: Float[Array, "3"],
    end: Float[Array, "3"],
    rigidity: SectionRigidity,
) -> Float[Array, "dofs_member dofs_member"]:
    """
    Elastic stiffness of one member, about the global axes.

    Parameters
    ----------
    start :
        Position of the member's first end.
    end :
        Position of the member's second end.
    rigidity :
        The three products of a modulus and a section property.

    Returns
    -------
    stiffness :
        Twelve by twelve, in global components.

    Notes
    -----
    This is the one object an adjoint needs from the element level. It serves
    both terms at once — the implicit one, where equilibrium is differentiated,
    and the explicit one, where the member's own end forces are read back — so
    nothing else about the element has to be differentiated at all.
    """
    frame = stiffness_frame(start, end)
    transform = member_transform(frame)
    length = jnp.linalg.norm(end - start)
    local = stiffness_local(length, rigidity)
    stiffness = transform.T @ local @ transform

    return stiffness


def stiffness_members(
    xyz: Float[Array, "nodes 3"],
    edges: Int[np.ndarray, "members 2"],
    rigidity: SectionRigidity,
) -> Float[Array, "members dofs_member dofs_member"]:
    """
    Elastic stiffness of every member, about the global axes.

    Parameters
    ----------
    xyz :
        Position of every node.
    edges :
        The two nodes every member spans, as static index data.
    rigidity :
        The three products of a modulus and a section property, per member.

    Returns
    -------
    stiffness :
        One twelve by twelve block per member.
    """
    starts = xyz[edges[:, 0]]
    ends = xyz[edges[:, 1]]
    over_members = jax.vmap(stiffness_global)
    stiffness = over_members(starts, ends, rigidity)

    return stiffness
