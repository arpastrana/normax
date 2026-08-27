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
What a run is configured by, read from a file.

Every section a design shares — the load cases, the two backends, the subspace,
the constraints, the descent's budget — is a container here. What the structure
is (`normax.structures`) and where its search starts (`normax.form_finding`)
vary by structure, so an example hands those two container types to
`parse_config`.
"""

from typing import Generic
from typing import NamedTuple
from typing import TypeVar

import yaml

from normax.optimization import OptimizationBudget

StructureT = TypeVar("StructureT")
StartT = TypeVar("StartT")


class LoadCaseConfig(NamedTuple):
    """
    One load case.

    Attributes
    ----------
    name :
        Pattern to apply, a key of `normax.loads.LOAD_PATTERNS`.
    magnitude :
        Total downward force the case carries, or the pressure for a pattern
        stated per unit of plan area.
    options :
        Whatever else the pattern reads, by keyword.
    """

    name: str
    magnitude: float
    options: dict[str, float | int | bool] = {}


class AnalysisConfig(NamedTuple):
    """
    What the frame is analyzed with.

    Attributes
    ----------
    diameter :
        Outer diameter every member is seeded with before the search sizes it.
    backend :
        Which solver fills the analysis slot: `smax` traces the frame in this
        process, `opensees` and `pynite` cross a Tesseract boundary to a host
        solver — the first planar, the second a space frame.
    """

    diameter: float
    backend: str


class SizingConfig(NamedTuple):
    """
    What the standard is read at, and which implementation reads it.

    Attributes
    ----------
    section_class :
        Cross-section class the wall thickness sits at the limit of.
    backend :
        Which check fills the sizing slot, every one across the Tesseract
        boundary: `blueprint` for Blueprints' cross-section check.
    """

    section_class: int
    backend: str


class SubspaceConfig(NamedTuple):
    """
    Which held-plan subspace the force densities move in.

    Attributes
    ----------
    symmetric :
        Whether the densities and the diameters are folded by the mirror.
    pivoted :
        Whether the coordinates are the densities of members elected
        independent, rather than projections on an orthonormal basis.
    """

    symmetric: bool
    pivoted: bool


class BoundsConfig(NamedTuple):
    """
    The box the force densities may move in, where they are the coordinates.

    Attributes
    ----------
    min :
        Smallest value any force density may take.
    max :
        Largest value any force density may take.
    """

    min: float
    max: float


class ConstraintsConfig(NamedTuple):
    """
    What the design is held to beside the check.

    Attributes
    ----------
    diameter_min :
        Smallest diameter any member may take, as a bound.
    length_min :
        Smallest length any member may keep, as rows. Zero turns it off.
    rise_max :
        Height no free node may rise above, or None for no cap.
    sag_min :
        Height no free node may hang below, or None for no floor.
    sign_margin_fraction :
        Sign margin the guarded force densities must clear, as a share of
        their median at the start. Zero or less turns the sign guard off.
    bounds :
        The box on the force densities, or None where the densities are not
        the coordinates.
    """

    diameter_min: float
    length_min: float
    rise_max: float | None
    sag_min: float | None
    sign_margin_fraction: float
    bounds: BoundsConfig | None


class OutputConfig(NamedTuple):
    """
    What a run does with its answer once the descent has ended.

    Attributes
    ----------
    verbose :
        Whether the run prints its report.
    export :
        Whether the run writes its record and its figures.
    viewer :
        Whether the run ends in a viewer.
    """

    verbose: bool
    export: bool
    viewer: bool


class RunConfig(NamedTuple, Generic[StructureT, StartT]):
    """
    Everything a run is configured by.

    Attributes
    ----------
    structure :
        Parameters the structure is generated from — an `ArchDescription`,
        `TrussDescription` or `ShellDescription` from `normax.structures`, never
        the built `Structure`; the example's builder turns one into the other.
    start :
        Parameters the starting force densities are generated from — a
        `UniformDensityInitializer` or `LensShapeInitializer` from
        `normax.form_finding` — or None where the drawn geometry is the start.
    load_cases :
        The cases the structure carries, the first of which shapes it.
    analysis :
        What the frame is analyzed with.
    sizing :
        What the standard is read at.
    subspace :
        The held-plan subspace, or None to move every force density freely.
    constraints :
        What the design is held to beside the check.
    optimization :
        What the descent may spend, and when it stops.
    output :
        What the run prints, writes and opens once the descent has ended.
    """

    structure: StructureT
    start: StartT | None
    load_cases: tuple[LoadCaseConfig, ...]
    analysis: AnalysisConfig
    sizing: SizingConfig
    subspace: SubspaceConfig | None
    constraints: ConstraintsConfig
    optimization: OptimizationBudget
    output: OutputConfig


def parse_config(
    text: str,
    structure_type: type[StructureT],
    start_type: type[StartT] | None = None,
) -> RunConfig[StructureT, StartT]:
    """
    The run config a file holds.

    Parameters
    ----------
    text :
        Text of the file describing the run.
    structure_type :
        Description the `structure` section is read into: `ArchDescription`,
        `TrussDescription` or `ShellDescription`, the parameters of one generator
        in `normax.structures`.
    start_type :
        Initializer the `start` section is read into, from `normax.form_finding`,
        or None for a run whose drawn geometry is the start.

    Returns
    -------
    config :
        The run config: its structure description, its start initializer, and
        every shared section.

    Raises
    ------
    TypeError
        If a section names a field that does not exist, or omits one that does.

    Notes
    -----
    No container carries a default, so a file missing a field is refused rather
    than quietly completed. Every budget is cast on the way in: YAML reads an
    exponent without a signed power as a string.
    """
    document = yaml.safe_load(text)

    start = None
    if start_type is not None:
        start = start_type(**document["start"])

    subspace = None
    if document.get("subspace") is not None:
        subspace = SubspaceConfig(**document["subspace"])

    held = dict(document["constraints"])
    bounds = held.pop("bounds", None)
    if bounds is not None:
        bounds = BoundsConfig(**bounds)
    constraints = ConstraintsConfig(bounds=bounds, **held)

    counts = (
        "rounds_max",
        "rounds_warmup",
        "iterations_warmup",
        "iterations_after_warmup",
    )
    named = document["optimization"]
    budget = {key: int(value) for key, value in named.items() if key in counts}
    scales = {key: float(value) for key, value in named.items() if key not in counts}
    budget.update(scales)
    load_cases = tuple(LoadCaseConfig(**entry) for entry in document["load_cases"])

    config = RunConfig(
        structure=structure_type(**document["structure"]),
        start=start,
        load_cases=load_cases,
        analysis=AnalysisConfig(**document["analysis"]),
        sizing=SizingConfig(**document["sizing"]),
        subspace=subspace,
        constraints=constraints,
        optimization=OptimizationBudget(**budget),
        output=OutputConfig(**document["output"]),
    )

    return config
