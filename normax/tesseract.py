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
The same three blocks, reached across a Tesseract boundary.

`normax.form_finding`, `normax.analysis.smax` and `normax.sizing` each hold a
block that computes in this process. This module holds one of each that does not:
the arguments are serialized against a schema, a container answers, and what
comes back is the same container the in-process block returns. Nothing in
`normax.design` can tell which it was handed, and that is the whole claim.

**The in-process blocks are the oracle, not the scaffolding.** Reproducing their
design and their gradient through the boundary is what turns "the Tesseracts run"
into "the boundary is transparent", and that claim cannot be made afterwards
without a baseline to make it against. `tests/test_tesseract_parity.py` is where
it is made, and it now runs one pipeline over two sets of blocks rather than two
implementations of one pipeline.

What the boundary costs, and what it buys, are both visible here. It costs the
loss of everything a schema cannot carry: the connectivity is rebuilt from flat
arrays on every call, and objects give way to arrays. It buys the only property
the pipeline actually needs, which is that no block has to be written in the same
language, or differentiate in the same way, as the one before it.

**One question of the standard crosses and the other does not.** The check is
asked what size a set of actions demands, which is what its schema carries; asked
how hard a size it did not choose is working, this block answers in process,
holding an `Ec3Sizer` for exactly that. An asymmetry worth naming rather than
hiding, and the reason it is a field rather than a private detail.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from ec3x.actions import MemberActions
from ec3x.material import Steel
from ec3x.section import TubeCatalogue
from ec3x.sizing import utilization_design
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

from normax.analysis import AbstractFrameAnalyzer
from normax.analysis import MemberForces
from normax.form_finding import AbstractFormFinder
from normax.form_finding import FormFoundShape
from normax.loads import count_load_cases
from normax.loads import select_load_case
from normax.loads import stack_load_cases
from normax.sizing import AbstractMemberSizer
from normax.sizing import MemberSizes
from normax.sizing.ec3 import Ec3Sizer
from normax.sizing.ec3 import neutral_sections
from normax.structures import Structure
from normax.structures import member_lengths

# Where the three Tesseract API modules live, relative to the package.
TESSERACTS = Path(__file__).resolve().parent.parent / "tesseracts"

STAGES = ("formfinding", "analysis", "ec3_check")

# What the analysis stage reads to choose its solver. Named here so a caller
# switching backends does not have to know the stage's own spelling.
BACKEND_VARIABLE = "NORMAX_ANALYSIS_BACKEND"


@contextmanager
def analysis_backend(name: str) -> Iterator[None]:
    """
    Run the analysis stage on a named solver for the duration of a block.

    Parameters
    ----------
    name :
        Backend to select, `smax` or `opensees`.

    Yields
    ------
    None
        The block runs with that backend selected.

    Notes
    -----
    The stage takes its backend from the environment, since a schema cannot
    carry a choice about who implements it and a container is configured once at
    startup. Comparing two backends in one process is the case that needs more
    than that, and this makes the switch a block rather than a global edit: the
    previous value is restored on the way out, exceptions included.

    Nothing is rebuilt. The same chain serves either solver, which is the claim
    the boundary makes rather than an optimization.
    """
    previous = os.environ.get(BACKEND_VARIABLE)
    os.environ[BACKEND_VARIABLE] = name

    try:
        yield
    finally:
        if previous is None:
            del os.environ[BACKEND_VARIABLE]
        else:
            os.environ[BACKEND_VARIABLE] = previous


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
        Member actions to the sizes EN 1993-1-1 requires.

    Notes
    -----
    Any client will do, whether it imports a module in this process or talks to
    a container over HTTP. Nothing below asks which, and that is the point of
    the boundary being a schema rather than a call.
    """

    formfinding: Tesseract
    analysis: Tesseract
    ec3: Tesseract


def local_chain(root: Path = TESSERACTS) -> Chain:
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


class TesseractFormFinder(AbstractFormFinder):
    """
    Form finding, reached across a Tesseract boundary.

    Attributes
    ----------
    client :
        The form-finding Tesseract.
    nodes :
        Starting position of every node.
    edges :
        The two node indices spanned by every member.
    supports :
        Indices of the nodes whose position is fixed.

    Notes
    -----
    The topology crosses as flat arrays on every call, a schema carrying arrays
    and not objects, so this block settles almost nothing: what it keeps is the
    three arrays in the dtypes the schema names, converted once rather than per
    call.
    """

    client: Tesseract
    nodes: Float[Array, "nodes 3"]
    edges: Int[Array, "members 2"]
    supports: Int[Array, "supports"]

    def __init__(self, structure: Structure, client: Tesseract) -> None:
        """
        Build a form finder that crosses a boundary to shape a structure.

        Parameters
        ----------
        structure :
            The structure supplying the topology and the supported nodes.
        client :
            The form-finding Tesseract.
        """
        self.client = client
        self.nodes = jnp.asarray(structure.nodes, dtype=jnp.float64)
        self.edges = jnp.asarray(structure.edges, dtype=jnp.int64)
        self.supports = jnp.asarray(structure.supports, dtype=jnp.int64)

    def __call__(
        self,
        q: Float[Array, "members"],
        loads: Float[Array, "nodes 3"],
    ) -> FormFoundShape:
        """
        Find the shape that carries a load case at given force densities.

        Parameters
        ----------
        q :
            Force density of every member. Negative in compression.
        loads :
            Force applied at every node.

        Returns
        -------
        shape :
            The geometry at equilibrium, and its member lengths.

        Notes
        -----
        The lengths are measured here rather than carried by the schema. A
        length is a distance between two nodes, geometry that no standard has an
        opinion on, so computing it locally cannot disagree with the far side
        and asking for it would pay a round trip for a subtraction.
        """
        crossed = apply_tesseract(
            self.client,
            {
                "q": q,
                "nodes": self.nodes,
                "edges": self.edges,
                "supports": self.supports,
                "loads": loads,
            },
        )
        lengths = member_lengths(crossed["xyz"], self.edges)
        shape = FormFoundShape(crossed["xyz"], lengths)

        return shape


class TesseractAnalyzer(AbstractFrameAnalyzer):
    """
    A frame analysis, reached across a Tesseract boundary.

    Attributes
    ----------
    client :
        The analysis Tesseract.
    catalogue :
        The section family the frame is analyzed with, whose ratio fixes the wall
        thickness and whose grade supplies the material.
    normal :
        Index of the global axis a planar structure has no thickness along, or
        None for a structure that occupies all three dimensions.
    edges :
        The two node indices spanned by every member.
    supports :
        Indices of the nodes whose translation is restrained.

    Notes
    -----
    **This block carries fewer opinions about what is differentiable than the
    others, and that is the schema's doing rather than an oversight.** Its
    differentiable inputs stop at the coordinates and the diameters; a material
    property crosses as a plain number, because the schema is meant to be
    satisfiable by a solver whose adjoints were written by hand.

    One load case crosses per call, the schema carrying one, so the round trips
    grow with their number while the boundary itself does not.
    """

    client: Tesseract
    catalogue: TubeCatalogue
    normal: int | None = eqx.field(static=True)
    edges: Int[Array, "members 2"]
    supports: Int[Array, "supports"]

    def __init__(
        self,
        structure: Structure,
        client: Tesseract,
        catalogue: TubeCatalogue,
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
        catalogue :
            The section family the frame is analyzed with, whose ratio fixes the
            wall thickness and whose grade supplies the material.
        normal :
            Index of the global axis a planar structure has no thickness along,
            or None for a structure that occupies all three dimensions.
        """
        self.client = client
        self.catalogue = catalogue
        self.normal = normal
        self.edges = jnp.asarray(structure.edges, dtype=jnp.int64)
        self.supports = jnp.asarray(structure.supports, dtype=jnp.int64)

    @property
    def steel(self) -> Steel:
        """
        The material the frame is analyzed with.
        """
        return self.catalogue.material

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
        analyzed = [
            apply_tesseract(
                self.client,
                {
                    "xyz": xyz,
                    "diameter": diameters,
                    "edges": self.edges,
                    "supports": self.supports,
                    "loads": load_case,
                    "f_y": self.steel.f_y,
                    "e_mod": self.steel.e_mod,
                    "density": self.steel.density,
                    "ratio": self.catalogue.ratio,
                    "normal": self.normal,
                },
            )
            for load_case in loads
        ]

        per_case = [
            MemberForces(
                forces["axial_force"],
                forces["end_moments_major"],
                forces["end_moments_minor"],
            )
            for forces in analyzed
        ]

        return stack_load_cases(per_case)


class TesseractSizer(AbstractMemberSizer):
    """
    EN 1993-1-1, reached across a Tesseract boundary.

    Attributes
    ----------
    client :
        The check's Tesseract.
    local :
        The same standard in this process, answering the questions the schema
        does not carry.

    Notes
    -----
    **The sizes cross and the re-check does not.** The schema asks what size a
    set of actions demands, which is what the standard decides; how hard a size
    the block did not choose is working is a second question it has no endpoint
    for, and it is answered in process. So is the mass a member of a given size
    carries per unit length, which is a property of a section family rather than
    a clause.

    Naming that as a field rather than reaching for an import is the honest form
    of it: this block is remote for one of its three answers and local for two,
    and a reader can see which.

    The cross-section class is confirmed by the block it delegates to, and
    crosses as a static field of the schema.
    """

    client: Tesseract
    local: Ec3Sizer

    def __init__(
        self,
        structure: Structure,
        client: Tesseract,
        catalogue: TubeCatalogue,
        resultant: bool = True,
    ) -> None:
        """
        Build a sizer that crosses a boundary to size and stays home to check.

        Parameters
        ----------
        structure :
            The structure whose members are sized. Read for nothing.
        client :
            The check's Tesseract.
        catalogue :
            The section family every member is drawn from, with the grade it is
            rolled from and the class its wall falls in.
        resultant :
            Whether the two moments combine as a resultant in the cross-section
            check, or as a linear sum.

        Raises
        ------
        ValueError
            If the family's ratio classifies as Class 4, or as a class other
            than the one the family names.
        """
        self.client = client
        self.local = Ec3Sizer(structure, catalogue, resultant)

    @property
    def steel(self) -> Steel:
        """
        Material properties and partial factors.
        """
        return self.local.steel

    @property
    def catalogue(self) -> TubeCatalogue:
        """
        The section family every member is drawn from.
        """
        return self.local.catalogue

    @property
    def section_class(self) -> int:
        """
        Cross-section class, confirmed against the family rather than given.
        """
        return self.local.section_class

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
            Length every member is assumed to buckle over.

        Returns
        -------
        sizes :
            The diameter each load case demands, and how hard it is worked.

        Notes
        -----
        EN 1993-1-1 Table B.3 is applied on the far side rather than here, so
        what comes back is a design moment and a factor rather than two end
        moments. Reducing one to the other is a clause, and this block does not
        second-guess the one that owns it: the utilization below is re-read at
        the factors the boundary reported rather than at a local reduction.
        """
        local = self.local
        carried = [
            select_load_case(forces, load_case)
            for load_case in range(count_load_cases(forces))
        ]

        crossed = [
            apply_tesseract(
                self.client,
                {
                    "axial_force": acting.axial_force,
                    "end_moments_major": acting.moment_major,
                    "end_moments_minor": acting.moment_minor,
                    "buckling_length": buckling_length,
                    "f_y": local.steel.f_y,
                    "e_mod": local.steel.e_mod,
                    "density": local.steel.density,
                    "gamma_m0": local.steel.gamma_m0,
                    "gamma_m1": local.steel.gamma_m1,
                    "ratio": local.catalogue.ratio,
                    "alpha": local.steel.alpha,
                    "diameter_min": local.catalogue.diameter_min,
                    "section_class": local.section_class,
                    "resultant": local.resultant,
                },
            )
            for acting in carried
        ]

        per_case = [
            MemberActions(
                acting.axial_force,
                sized["moment_major"],
                sized["moment_minor"],
                sized["moment_factor_major"],
                sized["moment_factor_minor"],
            )
            for acting, sized in zip(carried, crossed)
        ]
        demanded = jnp.stack([sized["diameter"] for sized in crossed])
        sections = local.catalogue(demanded)
        actions = stack_load_cases(per_case)

        def used_case(diameter: Float[Array, "members"], acting: MemberActions):
            return utilization_design(
                local.catalogue(diameter),
                acting,
                buckling_length,
                resultant=local.resultant,
            )

        used = jax.vmap(used_case)(demanded, actions)

        return MemberSizes(neutral_sections(sections), used)

    def governing(
        self,
        diameters: Float[Array, "members"],
        forces: MemberForces,
        buckling_length: Float[Array, "members"],
    ) -> Float[Array, "load_cases members"]:
        """
        Which limit state decided each member's size, under each load case.

        Parameters
        ----------
        diameters :
            Outer diameter every member was given.
        forces :
            What every member carries under every load case, which the check
            reduces to design actions itself.
        buckling_length :
            Length every member is assumed to buckle over.

        Returns
        -------
        governing :
            One of the limit-state codes of `ec3x.sizing`.

        Notes
        -----
        Answered in process, like the re-check it is read beside. The schema
        does carry this diagnostic, but only for a size it chose itself, and a
        design that has been reconciled across load cases is not at one.
        """
        return self.local.governing(diameters, forces, buckling_length)

    def utilization(
        self,
        diameters: Float[Array, "members"],
        forces: MemberForces,
        buckling_length: Float[Array, "members"],
    ) -> Float[Array, "load_cases members"]:
        """
        Re-read a finished design against the standard that sized it.

        Parameters
        ----------
        diameters :
            Outer diameter every member was given.
        forces :
            What every member carries under every load case, which the check
            reduces to design actions itself.
        buckling_length :
            Length every member is assumed to buckle over.

        Returns
        -------
        utilization :
            Demand over resistance of every member under every load case.

        Notes
        -----
        Answered in process, the schema having no endpoint that checks at a size
        rather than solving for one. It is the same clauses either way, which is
        why the answer is the one the boundary would have given.
        """
        return self.local.utilization(diameters, forces, buckling_length)
