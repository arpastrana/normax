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
What a run is described by, read from a file.

Every section a design shares — the load cases, the two backends, the subspace,
the constraints, the descent's budget — is a container here. What the structure
is, and how its start is sketched, belongs to the example describing it, which
hands its own container types to `parse_run`.
"""

from typing import Generic
from typing import NamedTuple
from typing import TypeVar

import yaml

from normax.optimization import AugmentedBudget

StructureT = TypeVar("StructureT")
SketchT = TypeVar("SketchT")


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
    margin_fraction :
        Sign margin guarded members must clear, as a share of their median
        density. Zero or less turns the guard off.
    """

    symmetric: bool
    pivoted: bool
    margin_fraction: float


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
    diameter_floor :
        Smallest diameter any member may take, as a bound.
    length_floor :
        Smallest length any member may keep, as rows. Zero turns it off.
    rise_ceiling :
        Height no free node may rise above, or None for no ceiling.
    sag_floor :
        Height no free node may hang below, or None for no floor.
    bounds :
        The box on the force densities, or None where the densities are not
        the coordinates.
    """

    diameter_floor: float
    length_floor: float
    rise_ceiling: float | None
    sag_floor: float | None
    bounds: BoundsConfig | None


class RunConfig(NamedTuple, Generic[StructureT, SketchT]):
    """
    Everything a run is described by.

    Attributes
    ----------
    structure :
        The structure to build, in the example's own terms.
    sketch :
        How the start is sketched, in the example's own terms, or None.
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
    augmented :
        What the descent may spend, and when it stops.
    viewer :
        Whether the run ends in a viewer.
    """

    structure: StructureT
    sketch: SketchT | None
    load_cases: tuple[LoadCaseConfig, ...]
    analysis: AnalysisConfig
    sizing: SizingConfig
    subspace: SubspaceConfig | None
    constraints: ConstraintsConfig
    augmented: AugmentedBudget
    viewer: bool


def parse_run(
    text: str,
    structure_type: type[StructureT],
    sketch_type: type[SketchT] | None = None,
) -> RunConfig[StructureT, SketchT]:
    """
    The run a file describes.

    Parameters
    ----------
    text :
        Text of the file describing the run.
    structure_type :
        Container the `structure` section is read into.
    sketch_type :
        Container the `sketch` section is read into, or None for a run whose
        start needs no sketch.

    Returns
    -------
    config :
        The run.

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

    sketch = None
    if sketch_type is not None:
        sketch = sketch_type(**document["sketch"])

    subspace = None
    if document.get("subspace") is not None:
        subspace = SubspaceConfig(**document["subspace"])

    held = dict(document["constraints"])
    bounds = held.pop("bounds", None)
    if bounds is not None:
        bounds = BoundsConfig(**bounds)
    constraints = ConstraintsConfig(bounds=bounds, **held)

    counts = ("rounds", "iterations", "settled", "opening")
    named = document["augmented"]
    budget = {key: int(value) for key, value in named.items() if key in counts}
    scales = {key: float(value) for key, value in named.items() if key not in counts}
    budget.update(scales)
    load_cases = tuple(LoadCaseConfig(**entry) for entry in document["load_cases"])

    config = RunConfig(
        structure=structure_type(**document["structure"]),
        sketch=sketch,
        load_cases=load_cases,
        analysis=AnalysisConfig(**document["analysis"]),
        sizing=SizingConfig(**document["sizing"]),
        subspace=subspace,
        constraints=constraints,
        augmented=AugmentedBudget(**budget),
        viewer=bool(document["viewer"]),
    )

    return config


def case_labels(load_cases: tuple[LoadCaseConfig, ...]) -> tuple[str, ...]:
    """
    A label per load case, the pattern's name and whatever options it took.

    Parameters
    ----------
    load_cases :
        The cases as described.

    Returns
    -------
    labels :
        One label per case, in order.
    """
    labels = []
    for load_case in load_cases:
        options = " ".join(f"{key}={value}" for key, value in load_case.options.items())
        labels.append(f"{load_case.name} {options}".strip())

    return tuple(labels)
