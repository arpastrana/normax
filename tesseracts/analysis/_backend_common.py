# SPDX-License-Identifier: Apache-2.0
"""
What the analysis schema hands its backends, read once off the wire.

Every backend solves the same frame and pulls back the same cotangent, so the
schema's dictionaries are turned into typed containers here and nowhere else.
"""

from typing import Any
from typing import NamedTuple

import numpy as np
from jaxtyping import Float

from normax.analysis import MemberForces
from normax.materials import SteelGrade
from normax.sections import TubeFamily
from normax.structures import Structure


class AnalyzedFrame(NamedTuple):
    """
    One frame as the schema describes it, ready for any backend.

    Attributes
    ----------
    structure :
        The connectivity and the supports, its nodes at the geometry analyzed.
    family :
        The section family, whose ratio fixes the wall and whose grade supplies
        the material.
    diameters :
        Outer diameter of every member.
    loads :
        Force applied at every node, one load case.
    normal :
        Index of the global axis a planar frame has no thickness along, or None.
    """

    structure: Structure
    family: TubeFamily
    diameters: Float[np.ndarray, "members"]
    loads: Float[np.ndarray, "nodes 3"]
    normal: int | None


def read_frame(inputs: dict[str, Any]) -> AnalyzedFrame:
    """
    The frame the schema's inputs describe.

    Parameters
    ----------
    inputs :
        The validated input fields of the analysis schema.

    Returns
    -------
    frame :
        The same frame, in the containers the backends take.

    Notes
    -----
    The schema carries no ultimate strength, so the grade gets a zero there,
    loud if anything reads it.
    """
    structure = Structure(
        nodes=np.asarray(inputs["xyz"]),
        edges=np.asarray(inputs["edges"]),
        supports=np.asarray(inputs["supports"]),
    )
    grade = SteelGrade(inputs["f_y"], 0.0, inputs["e_mod"], inputs["density"])
    family = TubeFamily(inputs["ratio"], grade)

    return AnalyzedFrame(
        structure,
        family,
        np.asarray(inputs["diameter"]),
        np.asarray(inputs["loads"]),
        inputs["normal"],
    )


def read_cotangent(
    cotangent_vector: dict[str, Any],
    members: int,
) -> MemberForces:
    """
    A cotangent on every reported force, zero wherever none was asked.

    Parameters
    ----------
    cotangent_vector :
        Cotangent on each output field a derivative is taken of.
    members :
        Number of members, sizing the fields left out.

    Returns
    -------
    cotangent :
        Cotangent on the axial force and both end moments.
    """
    seeded = {
        "axial_force": np.zeros(members),
        "end_moments_major": np.zeros((members, 2)),
        "end_moments_minor": np.zeros((members, 2)),
    }
    for name, value in cotangent_vector.items():
        seeded[name] = np.asarray(value)

    return MemberForces(
        seeded["axial_force"],
        seeded["end_moments_major"],
        seeded["end_moments_minor"],
    )


def force_outputs(forces: MemberForces) -> dict[str, np.ndarray]:
    """
    The schema's output fields, as plain arrays.

    Parameters
    ----------
    forces :
        What a backend reported about every member.

    Returns
    -------
    outputs :
        The axial force and both end moments, keyed as the schema names them.

    Notes
    -----
    NumPy rather than JAX, because an endpoint runs inside a foreign-function
    callback that forbids allocating a JAX array.
    """
    return {
        "axial_force": np.asarray(forces.axial_force),
        "end_moments_major": np.asarray(forces.moment_major),
        "end_moments_minor": np.asarray(forces.moment_minor),
    }
