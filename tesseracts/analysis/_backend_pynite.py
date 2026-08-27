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
The PyNite backend of the analysis stage, differentiated by an adjoint we wrote.

A space frame solver in plain Python that carries no derivative of any kind.
PyNite assembles, factorizes and solves; this repository states the element,
holds it against PyNite's matrices by test, and differentiates equilibrium as an
implicit function. `normax.analysis.pynite` carries the rule and the reasons.
"""

import hashlib

import numpy as np
from _backend_common import AnalyzedFrame

from normax.analysis import MemberForces
from normax.analysis import pynite

# One assembled and factorized frame, remembered between endpoint calls. The
# expensive half of a solve does not depend on the loading, and every load case
# in one evaluation and the adjoint after each asks for it at one geometry, so
# the key excludes the loads and one entry suffices. Safe only under serialized
# dispatch, which `normax.tesseract.pin_dispatch_thread` provides.
_PREPARED: dict[bytes, pynite.PreparedFrame] = {}


def _fingerprint(frame: AnalyzedFrame) -> bytes:
    """
    What identifies the frame an assembly and a factorization belong to.

    Notes
    -----
    Content rather than identity, the schema rebuilding its payload per call;
    shape and dtype enter along with the bytes. The loads are deliberately
    absent, the preparation not reading them.
    """
    digest = hashlib.blake2b(digest_size=32)
    arrays = (
        frame.structure.nodes,
        frame.diameters,
        frame.structure.edges,
        frame.structure.supports,
    )
    for value in arrays:
        held = np.ascontiguousarray(value)
        digest.update(str(held.dtype).encode())
        digest.update(str(held.shape).encode())
        digest.update(held.tobytes())
    steel = frame.family.material
    for scalar in (steel.f_y, steel.e_mod, steel.density, frame.family.ratio):
        digest.update(repr(float(scalar)).encode())

    return digest.digest()


def _prepared_frame(
    problem: pynite.FrameProblem,
    frame: AnalyzedFrame,
) -> pynite.PreparedFrame:
    """
    The assembled and factorized frame, fresh only if the last one differed.
    """
    fingerprint = _fingerprint(frame)
    held = _PREPARED.get(fingerprint)
    if held is not None:
        return held

    prepared = pynite.prepare_frame(problem, frame.structure.nodes, frame.diameters)
    _PREPARED.clear()
    _PREPARED[fingerprint] = prepared

    return prepared


def _frame_problem(frame: AnalyzedFrame) -> pynite.FrameProblem:
    """
    The frame in the container the rule takes; the normal axis is not read.
    """
    return pynite.FrameProblem(frame.structure, frame.family, frame.loads)


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
    problem = _frame_problem(frame)
    prepared = _prepared_frame(problem, frame)

    return pynite.compute_member_forces(
        problem, frame.structure.nodes, frame.diameters, frame.loads, prepared
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
    One solve whatever the parameter count, against the factorization the
    forward pass already paid for, so a gradient costs about what the forward
    pass costs.
    """
    problem = _frame_problem(frame)
    prepared = _prepared_frame(problem, frame)
    pulled = pynite.pull_back_cotangents(
        problem, frame.structure.nodes, frame.diameters, cotangent, prepared
    )

    return {"xyz": pulled.xyz, "diameter": pulled.diameter}
