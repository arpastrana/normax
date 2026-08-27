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

Every section a design shares — the form finding, the load cases, the two
backends, the constraints, the descent's budget — is a container here. What the
structure is varies by structure, so an example hands `parse_config` the
description type from `normax.structures` its `structure` section is read into.
"""

from typing import Generic
from typing import NamedTuple
from typing import TypeVar

import yaml

from normax.form_finding import AbstractDensityInitializer
from normax.form_finding import build_density_initializer
from normax.optimization import OptimizationBudget

StructureT = TypeVar("StructureT")


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
    fold_mirror :
        Whether the diameters are folded by the form finder's mirror, one
        section per mirrored pair.
    fold_polar :
        Whether the diameters are folded by a one-spoke rotation as well, one
        section per ring per family.
    """

    section_class: int
    backend: str
    fold_mirror: bool
    fold_polar: bool


class FormFindingConfig(NamedTuple):
    """
    How the form finder is parametrized, and where its densities start.

    Attributes
    ----------
    basis :
        Convention of the held-plan basis the densities move in — `pivoted`
        for the independent members' own densities, `svd` for projections on
        an orthonormal basis — or None to move every density freely.
    mirror :
        Axis the mirror plane stands normal to, folding the densities by that
        symmetry, or None for no symmetry.
    initializer :
        What generates the force densities the search starts from.
    """

    basis: str | None
    mirror: str | None
    initializer: AbstractDensityInitializer


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
    sign_guard :
        The sign each guarded member family must keep, `tension` or
        `compression` by family name, or None for no guard.
    bounds :
        The box on the force densities, or None where the densities are not
        the coordinates.
    """

    diameter_min: float
    length_min: float
    rise_max: float | None
    sag_min: float | None
    sign_margin_fraction: float
    sign_guard: dict[str, str] | None
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


class RunConfig(NamedTuple, Generic[StructureT]):
    """
    Everything a run is configured by.

    Attributes
    ----------
    structure :
        Parameters the structure is generated from — an `ArchDescription`,
        `TrussDescription` or `ShellDescription` from `normax.structures`, never
        the built `Structure`; the example's builder turns one into the other.
    form_finding :
        How the form finder is parametrized, and where its densities start.
    load_cases :
        The cases the structure carries, the first of which shapes it.
    analysis :
        What the frame is analyzed with.
    sizing :
        What the standard is read at.
    constraints :
        What the design is held to beside the check.
    optimization :
        What the descent may spend, and when it stops.
    output :
        What the run prints, writes and opens once the descent has ended.
    """

    structure: StructureT
    form_finding: FormFindingConfig
    load_cases: tuple[LoadCaseConfig, ...]
    analysis: AnalysisConfig
    sizing: SizingConfig
    constraints: ConstraintsConfig
    optimization: OptimizationBudget
    output: OutputConfig


def parse_config(
    text: str,
    structure_type: type[StructureT],
) -> RunConfig[StructureT]:
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

    Returns
    -------
    config :
        The run config: its structure description and every shared section.

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

    named = dict(document["form_finding"])
    initializer = build_density_initializer(named.pop("force_density"))
    form_finding = FormFindingConfig(initializer=initializer, **named)

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
        form_finding=form_finding,
        load_cases=load_cases,
        analysis=AnalysisConfig(**document["analysis"]),
        sizing=SizingConfig(**document["sizing"]),
        constraints=constraints,
        optimization=OptimizationBudget(**budget),
        output=OutputConfig(**document["output"]),
    )

    return config
