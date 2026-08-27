# SPDX-License-Identifier: Apache-2.0
"""
The OpenSees backend of the analysis stage, differentiated by DDM.

A C++ solver behind a command interface, so nothing here is traced: the reverse
rule contracts a Jacobian the solver sweeps out one parameter at a time. Two
dimensions, because that is where OpenSees reaches a nodal coordinate at all;
`normax.analysis.opensees` carries the model and the reasons.
"""

import numpy as np
from _backend_common import AnalyzedFrame

from normax.analysis import MemberForces
from normax.analysis import opensees


def solve_forces(frame: AnalyzedFrame) -> MemberForces:
    """
    Internal forces of the frame, from one solve.

    Parameters
    ----------
    frame :
        The frame as the schema describes it.

    Returns
    -------
    forces :
        Axial force and both end moments of every member.
    """
    model = opensees.prepare_model(frame.structure, frame.family, frame.normal)

    return opensees.compute_member_forces(
        model, frame.structure.nodes, frame.diameters, frame.family, frame.loads
    )


def forces_vjp(frame: AnalyzedFrame, cotangent: MemberForces) -> dict[str, np.ndarray]:
    """
    Pull a cotangent on the reported forces back onto the coordinates and sizes.

    Parameters
    ----------
    frame :
        The frame as the schema describes it.
    cotangent :
        Cotangent on every reported force.

    Returns
    -------
    cotangents :
        Cotangent on `xyz` and on `diameter`.

    Notes
    -----
    Contracts the dense blocks of one DDM sweep, so what a traced solver gets in
    one reverse pass costs a sweep whose length is the parameter count. The
    minor-axis cotangent contributes nothing: a plane frame's minor moment is
    identically zero, and its one nonzero derivative is the block the plane
    cannot reach.
    """
    model = opensees.prepare_model(frame.structure, frame.family, frame.normal)
    blocks = opensees.compute_force_jacobian(
        model, frame.structure.nodes, frame.diameters, frame.family, frame.loads
    )
    axial = np.asarray(cotangent.axial_force)
    major = np.asarray(cotangent.moment_major)

    by_xyz = np.tensordot(axial, blocks.axial_force_xyz, axes=1) + np.tensordot(
        major, blocks.moment_major_xyz, axes=2
    )
    by_diameter = np.tensordot(
        axial, blocks.axial_force_diameter, axes=1
    ) + np.tensordot(major, blocks.moment_major_diameter, axes=2)

    return {"xyz": by_xyz, "diameter": by_diameter}
