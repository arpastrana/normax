# SPDX-License-Identifier: Apache-2.0
"""
The frame element, stated in JAX so a solver without derivatives can borrow one.

An adjoint needs the derivative of the global element stiffness in the geometry
and the section, and the exact way to get it is to state the element here and
prove it is the element the foreign solver assembled. The proof is a test that
holds `assemble_stiffness_local` and `assemble_stiffness_global` against the
solver's own matrices to near machine precision. Two properties of a tube keep
this short: one bending rigidity serves both transverse axes, and the global
stiffness is invariant to a roll about the member axis, so the transverse basis
may be chosen for conditioning alone.
"""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from jaxtyping import Float

# Translations and rotations at both ends of one member.
DOF_PER_MEMBER = 12

# How many end blocks the direction cosines repeat over.
BLOCKS_PER_MEMBER = 4

# What the transverse axes are completed against: the repository's vertical.
REFERENCE_AXIS = 2

# How square to the vertical a member must stay for that reference to hold.
REFERENCE_MARGIN = 1.0e-6

# Which local degrees of freedom each rigidity couples, at both ends.
DOFS_AXIAL = (0, 6)
DOFS_TORSION = (3, 9)
DOFS_BENDING_Z = (1, 5, 7, 11)
DOFS_BENDING_Y = (2, 4, 8, 10)

# How a rigidity over a length couples two ends.
PAIR_COUPLING = np.array([[1.0, -1.0], [-1.0, 1.0]])


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
    """

    axial: Float[Array, "*members"]
    bending: Float[Array, "*members"]
    torsional: Float[Array, "*members"]


def assemble_bending_block(
    length: Float[Array, ""],
    bending: Float[Array, ""],
    sign: float,
) -> Float[Array, "4 4"]:
    """
    Bernoulli-Euler bending stiffness of one member in one transverse plane.

    Parameters
    ----------
    length :
        Distance between the member's ends.
    bending :
        Product of the elastic modulus and the second moment.
    sign :
        Sign of the shear-rotation coupling, which the two planes take opposite.

    Returns
    -------
    block :
        Four by four, ordered translation and rotation at each end.
    """
    shear = 12.0 * bending / length**3
    coupling = sign * 6.0 * bending / length**2
    near = 4.0 * bending / length
    far = 2.0 * bending / length
    block = [
        [shear, coupling, -shear, coupling],
        [coupling, near, -coupling, far],
        [-shear, -coupling, shear, -coupling],
        [coupling, far, -coupling, near],
    ]

    return jnp.asarray(block)


def assemble_stiffness_local(
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
    Bernoulli-Euler with no shear area, matching the foreign solver this stands
    in for: an adjoint must differentiate the model that was solved.
    """
    pair = jnp.asarray(PAIR_COUPLING)
    axial = np.ix_(DOFS_AXIAL, DOFS_AXIAL)
    torsion = np.ix_(DOFS_TORSION, DOFS_TORSION)
    bending_z = np.ix_(DOFS_BENDING_Z, DOFS_BENDING_Z)
    bending_y = np.ix_(DOFS_BENDING_Y, DOFS_BENDING_Y)

    stiffness = jnp.zeros((DOF_PER_MEMBER, DOF_PER_MEMBER))
    stiffness = stiffness.at[axial].set(rigidity.axial / length * pair)
    stiffness = stiffness.at[torsion].set(rigidity.torsional / length * pair)
    stiffness = stiffness.at[bending_z].set(
        assemble_bending_block(length, rigidity.bending, 1.0)
    )
    stiffness = stiffness.at[bending_y].set(
        assemble_bending_block(length, rigidity.bending, -1.0)
    )

    return stiffness


def choose_stiffness_frame(
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
    Nothing may read this choice: the global stiffness of an axisymmetric
    section is invariant to it. The transverse pair is completed against the
    global axis the member leans on least, so every orientation stays well
    conditioned. `compute_direction_cosines` is the one that answers to a convention.
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


def compute_direction_cosines(
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
    The transverse pair is completed against the vertical, the ordinary
    structural convention, which turns smoothly with the geometry. **It
    degenerates for a vertical member**, where the cross product vanishes, and
    `REFERENCE_MARGIN` is how square to the vertical a member has to stay. Only
    the pair's invariants survive the convention, and those are what a check of
    an axisymmetric section reads.
    """
    span = end - start
    axis = span / jnp.linalg.norm(span)
    reference = jnp.eye(3)[REFERENCE_AXIS]
    crossed = jnp.cross(axis, reference)
    first = crossed / jnp.linalg.norm(crossed)
    second = jnp.cross(axis, first)
    frame = jnp.stack([axis, first, second])

    return frame


def tile_member_transform(
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


def assemble_stiffness_global(
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
    The one object an adjoint needs from the element level: it serves the
    implicit term, where equilibrium is differentiated, and the explicit one,
    where the member's own end forces are read back.
    """
    frame = choose_stiffness_frame(start, end)
    transform = tile_member_transform(frame)
    length = jnp.linalg.norm(end - start)
    local = assemble_stiffness_local(length, rigidity)
    stiffness = transform.T @ local @ transform

    return stiffness
