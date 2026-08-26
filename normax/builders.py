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
Every builder in one file: the blocks a run description names, built.

The one place every shipping backend is named: the two Tesseract crossings,
picked here so nothing downstream asks which was chosen. The oracles are not
named here — a validation run constructs them directly, so the shipping path
imports neither oracle package.
"""

from normax.analysis import AbstractFrameAnalyzer
from normax.analysis import normal_axis
from normax.config import AnalysisConfig
from normax.config import ConstraintsConfig
from normax.config import SizingConfig
from normax.design import DesignConstraints
from normax.design import StructuralDesignPipeline
from normax.form_finding import FdmFormFinder
from normax.materials import SteelGrade
from normax.sections import TubeFamily
from normax.sizing import AbstractMemberSizer
from normax.structures import Structure
from normax.symmetry import SignGuard
from normax.tesseract import TesseractAnalyzer
from normax.tesseract import TesseractSizer
from normax.tesseract import analysis_tesseract
from normax.tesseract import sizing_tesseract

# The crossed solvers, and which of them is planar and must be told its plane.
ANALYSIS_CROSSED = ("opensees", "pynite")
ANALYSIS_PLANAR = ("opensees",)

# The crossed checks.
SIZING_CROSSED = ("blueprint",)

# EN 1993-1-1 Table 5.2 sheet 3: d/t limits of a tube in compression, per class,
# in multiples of epsilon squared.
CLASS_LIMITS = {1: 50.0, 2: 70.0, 3: 90.0}


def build_section_family(grade: SteelGrade, section_class: int) -> TubeFamily:
    """
    The section family as thin as a given class allows.

    Parameters
    ----------
    grade :
        The steel as a certificate states it.
    section_class :
        Class 1, 2 or 3, whose Table 5.2 limit fixes the wall proportion.

    Returns
    -------
    family :
        The family whose ratio sits exactly on that class's limit.

    Raises
    ------
    ValueError
        If the class is not 1, 2 or 3.

    Notes
    -----
    EN 1993-1-1 Table 5.2 sheet 3, `d/t <= k epsilon^2` with `epsilon^2 =
    235 / f_y`. Sitting on the limit maximizes the wall slenderness, and so
    minimizes material, while staying inside the class, so classification is
    exact by construction and needs no smoothing.
    """
    if section_class not in CLASS_LIMITS:
        raise ValueError(f"section_class must be 1, 2 or 3, got {section_class}")

    ratio = CLASS_LIMITS[section_class] * 235.0 / grade.f_y

    return TubeFamily(ratio, grade)


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

    normal = normal_axis(structure) if config.backend in ANALYSIS_PLANAR else None
    client = analysis_tesseract(config.backend)

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

    client = sizing_tesseract(config.backend)

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


def build_design_constraints(
    config: ConstraintsConfig,
    guard: SignGuard | None,
) -> DesignConstraints:
    """
    What the design is held to, read off a run description.

    Parameters
    ----------
    config :
        The floors, the height limits and the density box the file names.
    guard :
        The sign guard the start scaled, or None for none.

    Returns
    -------
    constraints :
        Everything the descent is held to beside the check.
    """
    bounds = None if config.bounds is None else (config.bounds.min, config.bounds.max)
    constraints = DesignConstraints(
        config.diameter_floor,
        config.length_floor,
        config.rise_ceiling,
        config.sag_floor,
        guard,
        bounds,
    )

    return constraints
