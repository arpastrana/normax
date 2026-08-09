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
The same three stages, composed across Tesseract boundaries instead of in one
process.

`normax.pipeline` runs the pipeline as ordinary Python: three modules calling
each other, one JAX trace over all of it. This module runs it as three
Tesseracts, each with its own schema, each differentiating in its own way, with
serialized arrays between them. Both expose `q` and a mass, and both are
differentiable in `q`.

**The in-process version is the oracle, not the scaffolding.** Reproducing its
mass and its gradient through the boundary is what turns "the Tesseracts run"
into "the boundary is transparent", and that claim cannot be made afterwards
without a baseline to make it against. `tests/test_tesseract_parity.py` is where
it is made.

What the boundary costs, and what it buys, are both visible here. It costs the
loss of everything a schema cannot carry: the connectivity is rebuilt from flat
arrays on every call, and objects give way to arrays. It buys the only property
the pipeline actually needs, which is that no stage has to be written in the
same language, or differentiate in the same way, as the one before it.

The check reports which limit state decided each size, and this composition
drops it. That is deliberate: it is non-differentiable, and a concrete cotangent
on it raises rather than passing quietly. Read it beside a finished design with
`normax.pipeline.governing`.
"""

from pathlib import Path
from typing import NamedTuple

import numpy as np
from jaxtyping import Array
from jaxtyping import Float
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

from normax.ec3.sizing import Steel
from normax.ec3.sizing import Tube
from normax.pipeline import Design
from normax.structures import Structure

# Where the three Tesseract API modules live, relative to the package.
TESSERACTS = Path(__file__).resolve().parent.parent / "tesseracts"

STAGES = ("formfinding", "analysis", "ec3_check")


class Chain(NamedTuple):
    """
    The three Tesseracts of the pipeline, in the order they run.

    Attributes
    ----------
    formfinding :
        Force densities to a geometry.
    analysis :
        A geometry to the internal forces its members carry.
    ec3 :
        Member actions to the sizes EN 1993-1-1 requires, and a mass.

    Notes
    -----
    Any client will do, whether it imports a module in this process or talks to
    a container over HTTP. Nothing below asks which, and that is the point of
    the boundary being a schema rather than a call.
    """

    formfinding: Tesseract
    analysis: Tesseract
    ec3: Tesseract


def local(root: Path = TESSERACTS) -> Chain:
    """
    A chain that imports the three API modules into this process.

    Parameters
    ----------
    root :
        Directory holding one subdirectory per stage.

    Returns
    -------
    chain :
        The three stages.

    Raises
    ------
    FileNotFoundError
        If a stage has no API module under that directory.

    Notes
    -----
    No containers and no network, so the composition can be tested wherever the
    dependencies are installed. It exercises every endpoint a served Tesseract
    exposes, since the client is the same one either way, but it proves nothing
    about the image: `tesseract build` is what does that.
    """
    modules = {stage: root / stage / "tesseract_api.py" for stage in STAGES}

    for stage, module in modules.items():
        if not module.is_file():
            raise FileNotFoundError(f"no API module for stage {stage!r} at {module}")

    return Chain(
        formfinding=Tesseract.from_tesseract_api(modules["formfinding"]),
        analysis=Tesseract.from_tesseract_api(modules["analysis"]),
        ec3=Tesseract.from_tesseract_api(modules["ec3_check"]),
    )


def design(
    q: Float[Array, "members"],
    diameters: Float[Array, "members"],
    structure: Structure,
    chain: Chain,
    steel: Steel,
    tube: Tube,
    *,
    normal: int | None,
    plastic: bool,
    resultant: bool = True,
    l_cr: Float[Array, "members"] | None = None,
) -> Design:
    """
    Form-find, analyse and size, each in its own Tesseract.

    Parameters
    ----------
    q :
        Force density of every member. Negative in compression.
    diameters :
        Diameters the frame is analysed with, being the previous outer iterate
        of the check. They set the stiffness, not the resistance.
    structure :
        The structure supplying the connectivity, the supports and the loads.
    chain :
        The three Tesseracts, from `local` or built from images.
    steel :
        Material properties and partial factors.
    tube :
        The section family every member is drawn from.
    normal :
        Index of the global axis a planar structure has no thickness along, or
        None for a structure that occupies all three dimensions.
    plastic :
        Whether the section is Class 1 or 2. Static, never a traced value.
    resultant :
        Whether the two moments combine as a resultant in the cross-section
        check, or as a linear sum.
    l_cr :
        Buckling length of every member. If None, each member buckles over its
        own length.

    Returns
    -------
    design :
        The geometry, the member actions, the required sizes and the mass.

    Notes
    -----
    The same design `normax.pipeline.design` returns, field for field, and the
    same gradient. Everything that differs is on the inside: the connectivity is
    a Tesseract's own business here rather than an argument, so there is no
    form-finding graph to pass, and the cross-section class and the plane of a
    planar structure cross as static fields of a schema rather than as Python
    keywords.

    The three stages carry different opinions about what is differentiable.
    Form finding and the check will differentiate with respect to any material
    property they are given; the analysis will not, and its differentiable
    inputs stop at the coordinates and the diameters. That is not an oversight
    but the constraint of a schema meant to be satisfiable by a solver whose
    adjoints were written by hand.
    """
    edges = np.asarray(structure.edges, dtype=np.int64)
    supports = np.asarray(structure.supports, dtype=np.int64)
    loads = np.asarray(structure.loads, dtype=np.float64)

    shape = apply_tesseract(
        chain.formfinding,
        {
            "q": q,
            "nodes": np.asarray(structure.nodes, dtype=np.float64),
            "edges": edges,
            "supports": supports,
            "loads": loads,
        },
    )

    member = apply_tesseract(
        chain.analysis,
        {
            "xyz": shape["xyz"],
            "diameter": diameters,
            "edges": edges,
            "supports": supports,
            "loads": loads,
            "f_y": steel.f_y,
            "e_mod": steel.e_mod,
            "density": steel.density,
            "ratio": tube.ratio,
            "normal": normal,
        },
    )

    lengths = shape["lengths"]
    buckling = lengths if l_cr is None else l_cr

    sized = apply_tesseract(
        chain.ec3,
        {
            "n_ed": member["n_ed"],
            "m_y_ed": member["m_y_ed"],
            "m_z_ed": member["m_z_ed"],
            "lengths": lengths,
            "l_cr": buckling,
            "f_y": steel.f_y,
            "e_mod": steel.e_mod,
            "density": steel.density,
            "gamma_m0": steel.gamma_m0,
            "gamma_m1": steel.gamma_m1,
            "ratio": tube.ratio,
            "alpha": tube.alpha,
            "diameter_min": tube.diameter_min,
            "plastic": plastic,
            "resultant": resultant,
        },
    )

    return Design(
        xyz=shape["xyz"],
        lengths=lengths,
        n_ed=member["n_ed"],
        m_ed=sized["m_ed"],
        c_m=sized["c_m"],
        l_cr=buckling,
        diameters=sized["diameter"],
        utilization=sized["utilization"],
        mass=sized["mass"],
    )


def mass(
    q: Float[Array, "members"],
    diameters: Float[Array, "members"],
    structure: Structure,
    chain: Chain,
    steel: Steel,
    tube: Tube,
    *,
    normal: int | None,
    plastic: bool,
    resultant: bool = True,
    l_cr: Float[Array, "members"] | None = None,
) -> Float[Array, ""]:
    """
    Total mass EN 1993-1-1 requires at a set of force densities, across three
    Tesseracts.

    Parameters
    ----------
    q :
        Force density of every member. Negative in compression.
    diameters :
        Diameters the frame is analysed with, being the previous outer iterate.
    structure :
        The structure supplying the connectivity, the supports and the loads.
    chain :
        The three Tesseracts, from `local` or built from images.
    steel :
        Material properties and partial factors.
    tube :
        The section family every member is drawn from.
    normal :
        Index of the global axis a planar structure has no thickness along, or
        None for a structure that occupies all three dimensions.
    plastic :
        Whether the section is Class 1 or 2. Static, never a traced value.
    resultant :
        Whether the two moments combine as a resultant in the cross-section
        check, or as a linear sum.
    l_cr :
        Buckling length of every member. If None, each member buckles over its
        own length.

    Returns
    -------
    mass :
        Total mass.

    Notes
    -----
    The scalar `jax.grad` is taken of, and a scalar for a reason: one loss means
    one cotangent, so each stage is asked for a single reverse pass and the
    chain costs three round trips rather than three per output component.
    """
    return design(
        q,
        diameters,
        structure,
        chain,
        steel,
        tube,
        normal=normal,
        plastic=plastic,
        resultant=resultant,
        l_cr=l_cr,
    ).mass
