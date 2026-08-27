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
The two blocks that cross a Tesseract boundary, and the clients that reach them.

A frame analysis hosted by a solver that does not differentiate itself, and a
code check hosted by a library that never heard of a gradient, each behind a
schema with a hand-written adjoint. On the JAX side they are ordinary blocks:
`apply_tesseract` is a primitive, so `jax.grad` through either takes the
server's `vector_jacobian_product` in one crossing.
"""

import functools
import os
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract
from tesseract_jax.tesseract_compat import Jaxeract

from normax.analysis import AbstractFrameAnalyzer
from normax.analysis import MemberForces
from normax.analysis import find_normal_axis
from normax.config import AnalysisConfig
from normax.config import SizingConfig
from normax.design import StructuralDesignPipeline
from normax.form_finding import FdmFormFinder
from normax.loads import count_load_cases
from normax.loads import select_load_case
from normax.loads import stack_load_cases
from normax.sections import TubeFamily
from normax.sizing import AbstractMemberSizer
from normax.sizing import MemberSizes
from normax.sizing.blueprint import DIAMETER_MINIMUM
from normax.sizing.blueprint import GAMMA_M0
from normax.sizing.blueprint import snapshot_family
from normax.structures import Structure

# The crossed solvers, and which of them is planar and must be told its plane.
ANALYSIS_CROSSED = ("opensees", "pynite")
ANALYSIS_PLANAR = ("opensees",)

# The crossed checks.
SIZING_CROSSED = ("blueprint",)

# Where the Tesseract API modules live, relative to the package.
TESSERACTS = Path(__file__).resolve().parent.parent / "tesseracts"

# What each stage reads to choose who answers it.
ANALYSIS_VARIABLE = "NORMAX_ANALYSIS_BACKEND"
SIZING_VARIABLE = "NORMAX_SIZING_BACKEND"

# Endpoints whose work must not move between threads.
PINNED_ENDPOINTS = ("apply", "jacobian_vector_product", "vector_jacobian_product")

# One worker owning every dispatch across the process, and a re-entrancy flag.
_DISPATCH_OWNER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tesseract")
_DISPATCHING = threading.local()


def _dispatch_owned(work: Callable[..., Any], args: tuple, kwargs: dict) -> Any:
    """
    Run one dispatch with the owner thread marked as claimed.
    """
    _DISPATCHING.held = True
    try:
        return work(*args, **kwargs)
    finally:
        _DISPATCHING.held = False


def pin_dispatch_thread() -> None:
    """
    Make every Tesseract endpoint run on one thread, for the whole process.

    Notes
    -----
    Tesseract-JAX lowers a call to an XLA host callback with no ordering and no
    thread affinity, so under `jit` the runtime runs several dispatches at once.
    A local Tesseract runs the API in this process, where a solver owning one
    mutable domain is corrupted by that, and the local client redirects file
    descriptors around every call, which races process-wide. Pinning rather
    than locking, because a library with thread-affine state needs one owner.
    This belongs upstream; `NORMAX_PIN_DISPATCH=0` leaves the endpoints alone.
    """
    if os.environ.get("NORMAX_PIN_DISPATCH") == "0":
        return

    def pin(work: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(work)
        def pinned(*args: Any, **kwargs: Any) -> Any:
            if getattr(_DISPATCHING, "held", False):
                return work(*args, **kwargs)
            submitted = _DISPATCH_OWNER.submit(_dispatch_owned, work, args, kwargs)

            return submitted.result()

        return pinned

    for name in PINNED_ENDPOINTS:
        work = getattr(Jaxeract, name, None)
        if work is None or getattr(work, "__wrapped__", None) is not None:
            continue
        setattr(Jaxeract, name, pin(work))


# Armed on import: a crossed call is unsafe before it, so there is no window.
pin_dispatch_thread()


def open_tesseract(stage: str, root: Path = TESSERACTS) -> Tesseract:
    """
    A client that imports one stage's API module into this process.

    Parameters
    ----------
    stage :
        Directory of the stage under the root.
    root :
        Directory holding one subdirectory per stage.

    Returns
    -------
    client :
        The stage, behind the same client a served container answers.

    Raises
    ------
    FileNotFoundError
        If the stage has no API module under that directory.

    Notes
    -----
    No containers and no network, so the composition is tested wherever the
    dependencies are installed. It proves nothing about the image.
    """
    module = root / stage / "tesseract_api.py"
    if not module.is_file():
        raise FileNotFoundError(f"no API module for stage {stage!r} at {module}")

    return Tesseract.from_tesseract_api(module)


def open_tesseract_analysis(backend: str, root: Path = TESSERACTS) -> Tesseract:
    """
    The analysis stage, its solver picked for the whole process.

    Parameters
    ----------
    backend :
        Which solver answers the stage, `opensees` or `pynite`.
    root :
        Directory holding one subdirectory per stage.

    Returns
    -------
    client :
        The analysis stage.

    Notes
    -----
    The stage reads its solver from the environment, since a schema cannot
    carry a choice about who implements it and a container is configured once
    at startup.
    """
    os.environ[ANALYSIS_VARIABLE] = backend

    return open_tesseract("analysis", root)


def open_tesseract_sizing(backend: str, root: Path = TESSERACTS) -> Tesseract:
    """
    The sizing stage, its check picked for the whole process.

    Parameters
    ----------
    backend :
        Which check answers the stage, `blueprint` being the one that ships.
    root :
        Directory holding one subdirectory per stage.

    Returns
    -------
    client :
        The sizing stage.

    Notes
    -----
    The stage reads its check from the environment, since a schema cannot
    carry a choice about who implements it and a container is configured once
    at startup.
    """
    os.environ[SIZING_VARIABLE] = backend

    return open_tesseract("sizing", root)


class TesseractAnalyzer(AbstractFrameAnalyzer):
    """
    A frame analysis, reached across a Tesseract boundary.

    Attributes
    ----------
    client :
        The analysis Tesseract.
    family :
        The section family the frame is analyzed with, whose ratio fixes the
        wall and whose grade supplies the material.
    normal :
        Index of the global axis a planar structure has no thickness along, or
        None for a structure occupying all three dimensions.
    edges :
        The two node indices spanned by every member.
    supports :
        Indices of the nodes whose translation is restrained.

    Notes
    -----
    The differentiable inputs stop at the coordinates and the diameters; a
    material property crosses as a plain number, because the schema is meant to
    be satisfiable by a solver whose adjoints were written by hand. One load
    case crosses per call, the schema carrying one.
    """

    client: Tesseract
    family: TubeFamily
    normal: int | None = eqx.field(static=True)
    edges: Int[Array, "members 2"]
    supports: Int[Array, "supports"]

    def __init__(
        self,
        structure: Structure,
        client: Tesseract,
        family: TubeFamily,
        normal: int | None = None,
    ) -> None:
        """
        Build an analyzer that crosses a boundary to solve.

        Parameters
        ----------
        structure :
            The structure supplying the connectivity and the supported nodes.
        client :
            The analysis Tesseract.
        family :
            The section family the frame is analyzed with.
        normal :
            Index of the global axis a planar structure has no thickness along,
            or None for a structure occupying all three dimensions.
        """
        self.client = client
        self.family = family
        self.normal = normal
        self.edges = jnp.asarray(structure.edges, dtype=jnp.int64)
        self.supports = jnp.asarray(structure.supports, dtype=jnp.int64)

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
        steel = self.family.material
        per_case = []
        for load_case in loads:
            inputs = {
                "xyz": xyz,
                "diameter": diameters,
                "edges": self.edges,
                "supports": self.supports,
                "loads": load_case,
                "f_y": steel.f_y,
                "e_mod": steel.e_mod,
                "density": steel.density,
                "ratio": self.family.ratio,
                "normal": self.normal,
            }
            crossed = apply_tesseract(self.client, inputs, vmap_method="sequential")
            forces = MemberForces(
                crossed["axial_force"],
                crossed["end_moments_major"],
                crossed["end_moments_minor"],
            )
            per_case.append(forces)

        return stack_load_cases(per_case)


class TesseractSizer(AbstractMemberSizer):
    """
    A cross-section check, reached across a Tesseract boundary.

    Attributes
    ----------
    client :
        The check's Tesseract.
    family :
        The section family every member is drawn from.
    ratio :
        The family's wall proportion, snapshotted for the host.
    f_y :
        The family's yield strength, snapshotted for the host.

    Notes
    -----
    Every question crosses the boundary: the sizes come off the solve's
    outputs, and a size the caller owns goes over as `diameter_held` and comes
    back as `utilization_held`. A descent constrained on this block therefore
    crosses on every evaluation, and its gradient takes the far side's
    hand-written adjoint in one crossing.
    """

    client: Tesseract
    family: TubeFamily
    ratio: float = eqx.field(static=True)
    f_y: float = eqx.field(static=True)

    def __init__(
        self,
        structure: Structure,
        client: Tesseract,
        family: TubeFamily,
    ) -> None:
        """
        Build a sizer that crosses a boundary for every question.

        Parameters
        ----------
        structure :
            The structure whose members are sized. Read for nothing.
        client :
            The check's Tesseract.
        family :
            The section family every member is drawn from.

        Raises
        ------
        ValueError
            If the family's ratio leaves no wall at all.
        """
        ratio, f_y = snapshot_family(family)

        self.client = client
        self.family = family
        self.ratio = ratio
        self.f_y = f_y

    def cross_check(
        self,
        forces: MemberForces,
        diameter_held: Float[Array, "*load_cases members"],
        solve: bool,
    ) -> list[dict[str, Array]]:
        """
        Cross the boundary once per load case, at a held size.

        Parameters
        ----------
        forces :
            What every member carries under every load case.
        diameter_held :
            Outer diameter the held-size check is read at, per member, or per
            load case and member.
        solve :
            Whether the far side runs the sizing solve, or only the held
            check — the solve is the expensive half, so a caller who reads
            none of its outputs declines it.

        Returns
        -------
        crossed :
            The schema's outputs, one dictionary per load case.
        """
        held = jnp.broadcast_to(diameter_held, jnp.shape(forces.axial_force))
        crossed = []
        for load_case in range(count_load_cases(forces)):
            acting = select_load_case(forces, load_case)
            inputs = {
                "axial_force": acting.axial_force,
                "end_moments_major": acting.moment_major,
                "end_moments_minor": acting.moment_minor,
                "diameter_held": held[load_case],
                "f_y": jnp.asarray(self.f_y),
                "gamma_m0": jnp.asarray(GAMMA_M0),
                "ratio": jnp.asarray(self.ratio),
                "diameter_min": jnp.asarray(DIAMETER_MINIMUM),
                "solve": solve,
            }
            answer = apply_tesseract(self.client, inputs, vmap_method="sequential")
            crossed.append(answer)

        return crossed

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
            Accepted, ignored, and never serialized: the check's schema carries
            no length, which is the cross-section philosophy stated on the wire.

        Returns
        -------
        sizes :
            The diameter each load case demands, and how hard it is worked.
        """
        placeholder = jnp.full_like(forces.axial_force, DIAMETER_MINIMUM)
        crossed = self.cross_check(forces, placeholder, solve=True)
        demanded = jnp.stack([answer["diameter"] for answer in crossed])
        used = jnp.stack([answer["utilization"] for answer in crossed])

        return MemberSizes(self.family(demanded), used)

    def compute_utilization(
        self,
        diameters: Float[Array, "members"],
        forces: MemberForces,
        buckling_length: Float[Array, "members"],
    ) -> Float[Array, "load_cases members"]:
        """
        Check sizes the caller owns against Blueprints' cross-section check.

        Parameters
        ----------
        diameters :
            Outer diameter every member was given.
        forces :
            What every member carries under every load case.
        buckling_length :
            Accepted, ignored, and never serialized.

        Returns
        -------
        utilization :
            Demand over resistance of every member under every load case.

        Notes
        -----
        Crosses without the far side's sizing solve: this question reads only
        the held check, and the solve is the expensive half of a crossing.
        """
        crossed = self.cross_check(forces, diameters, solve=False)

        return jnp.stack([answer["utilization_held"] for answer in crossed])


def build_analyzer(
    structure: Structure,
    family: TubeFamily,
    config: AnalysisConfig,
) -> AbstractFrameAnalyzer:
    """
    The frame analysis a run description asks for.

    Parameters
    ----------
    structure :
        The structure the block is built on.
    family :
        The section family the frame is analyzed with.
    config :
        The backend.

    Returns
    -------
    analyzer :
        The block, behind its boundary.

    Raises
    ------
    ValueError
        If the backend is not one this module knows.
    """
    if config.backend not in ANALYSIS_CROSSED:
        raise ValueError(f"unknown analysis backend {config.backend!r}")

    normal = find_normal_axis(structure) if config.backend in ANALYSIS_PLANAR else None
    client = open_tesseract_analysis(config.backend)

    return TesseractAnalyzer(structure, client, family, normal)


def build_sizer(
    structure: Structure,
    family: TubeFamily,
    config: SizingConfig,
) -> AbstractMemberSizer:
    """
    The code check a run description asks for.

    Parameters
    ----------
    structure :
        The structure the block is built on.
    family :
        The section family every size is drawn from.
    config :
        The backend.

    Returns
    -------
    sizer :
        The block, behind its boundary.

    Raises
    ------
    ValueError
        If the backend is not one this module knows.
    """
    if config.backend not in SIZING_CROSSED:
        raise ValueError(f"unknown sizing backend {config.backend!r}")

    client = open_tesseract_sizing(config.backend)

    return TesseractSizer(structure, client, family)


def build_pipeline(
    structure: Structure,
    family: TubeFamily,
    analysis: AnalysisConfig,
    sizing: SizingConfig,
) -> StructuralDesignPipeline:
    """
    The three blocks a run composes, built on one structure.

    Parameters
    ----------
    structure :
        The structure every block is built from.
    family :
        The section family both the analysis and the check draw tubes from, so
        whatever differs downstream is the check itself.
    analysis :
        Which solver fills the analysis slot.
    sizing :
        Which check fills the sizing slot.

    Returns
    -------
    pipeline :
        A form finder, a frame analysis and a code check, composed.
    """
    pipeline = StructuralDesignPipeline(
        FdmFormFinder(structure),
        build_analyzer(structure, family, analysis),
        build_sizer(structure, family, sizing),
    )

    return pipeline
