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
The `smax` backend of the analysis stage, differentiated by tracing autodiff.

A JAX frame solver, so the whole assembly and solve is traceable and the
derivatives come out of the same machinery that produced them upstream. It is
the reference the second backend is measured against rather than the interesting
one: the argument the analysis stage makes is that a solver which cannot be
traced at all can sit behind this same schema.

Three dimensions throughout, which is what the gridshell needs and what a direct
differentiation backend cannot supply.
"""

from typing import Any

import jax.numpy as jnp

from normax.analysis import forces
from normax.ec3.sizing import Steel
from normax.ec3.sizing import Tube
from normax.structures import Structure


def solve(inputs: dict[str, Any]) -> dict[str, jnp.ndarray]:
    """
    Internal forces of the frame the inputs describe.

    Parameters
    ----------
    inputs :
        The validated input fields of the analysis schema.

    Returns
    -------
    outputs :
        Axial force and both end moments of every member.

    Notes
    -----
    The frame is assembled inside this call from the coordinates and the
    diameters, so both are differentiable leaves rather than properties of a
    model built beforehand. The reference state is unstressed: the nodes displace
    before any internal force appears, and that elastic response is the whole of
    the gap between these axial forces and the ones form finding predicted.

    Yield strength and density reach the material but not the answer, a linear
    elastic analysis under nodal loads having no use for either. They are carried
    so that the schema still describes the frame when self-weight or a nonlinear
    backend arrives.
    """
    xyz = jnp.asarray(inputs["xyz"])

    structure = Structure(
        nodes=xyz,
        edges=jnp.asarray(inputs["edges"]),
        supports=jnp.asarray(inputs["supports"]),
        loads=jnp.asarray(inputs["loads"]),
    )

    steel = Steel(
        f_y=inputs["f_y"],
        e_mod=inputs["e_mod"],
        density=inputs["density"],
    )
    tube = Tube(ratio=inputs["ratio"])

    member = forces(
        structure,
        xyz,
        jnp.asarray(inputs["diameter"]),
        steel,
        tube,
        normal=inputs["normal"],
    )

    return {
        "n_ed": member.n_ed,
        "m_y_ed": member.m_y_ed,
        "m_z_ed": member.m_z_ed,
    }
