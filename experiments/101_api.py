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
A steel arch designed end to end, through three blocks and one gradient.

**The whole project in one file.** A file describes an arch, its load cases and
the search; three blocks — a form finder, a frame analysis and EN 1993-1-1 —
are built on that structure and composed into one function; the mass that comes
out has an exact gradient in the force densities, and a bounded descent spends
it. Nothing else happens here, so what the composition costs to write is exactly
what is written.

**A block is built from a structure and then called.** The constructor is where
each piece of software gets to see the structure in its own terms — a form
finder wants connectivity matrices, a frame solver wants an assembly and degree
of freedom maps, a code check wants nothing at all — and it runs once, on the
host. What is left is a function of design parameters and load cases, which is
what the optimizer differentiates and what compiles.

**The three disagree about how they compute, and the pipeline never asks.** Form
finding traces a linear solve, the frame analysis traces an assembly, and the
code check carries a hand-derived tangent at the root of a residual. Replacing
one with a block that crosses a Tesseract boundary, or wraps a solver written in
another language, is a different argument to `StructuralDesignPipeline` and
nothing else:
`normax.tesseract` holds three such blocks and
`tests/test_tesseract_parity.py` measures that the swap changes no number.

Run with `uv run --group pipeline python experiments/101_api.py [arch.yaml]`.
"""

import sys
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import yaml
from jaxtyping import Array
from jaxtyping import Float

from normax.analysis.smax import SmaxAnalyzer
from normax.design import Design
from normax.design import DesignParameters
from normax.design import StructuralDesignPipeline
from normax.design import compute_mass
from normax.design import design_envelope
from normax.ec3.material import Steel
from normax.ec3.section import TubeCatalogue
from normax.form_finding.fdm import FdmFormFinder
from normax.loads import LoadCases
from normax.loads import assemble_load_cases
from normax.loads import create_loads_by_name
from normax.optimization import minimize_bounded
from normax.optimization import value_and_gradient
from normax.sizing import Ec3Sizer
from normax.structures import Structure
from normax.structures import build_arch_2d

# The arch and the search, unless another file is named on the command line.
CONFIG = Path(__file__).with_name("arch.yaml")

# The arch lies in the XZ plane, so it has no thickness along Y. A fact about
# the structure this script builds rather than a choice made about it.
NORMAL = 1

COMPILATION_CACHE = Path(__file__).resolve().parent.parent / ".jax_cache"
COMPILATION_CACHE.mkdir(exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(COMPILATION_CACHE))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)


class ArchConfig(NamedTuple):
    """
    The arch to build.

    Attributes
    ----------
    num_edges :
        Number of members the arch is discretized into.
    span :
        Horizontal distance between the two supports.
    rise :
        Height of the parabola the starting geometry rises along.
    """

    num_edges: int
    span: float
    rise: float


class FormFindingConfig(NamedTuple):
    """
    What the shape is searched from.

    Attributes
    ----------
    force_density :
        Force density every member starts at. Negative in compression.
    """

    force_density: float


class AnalysisConfig(NamedTuple):
    """
    What the frame is analyzed with, before the check has spoken.

    Attributes
    ----------
    diameter :
        Outer diameter every member is analyzed at.
    envelope_sharpness :
        Sharpness of the envelope reconciling the load cases into one size.
    """

    diameter: float
    envelope_sharpness: float


class SizingConfig(NamedTuple):
    """
    What the standard is read at.

    Attributes
    ----------
    section_class :
        Cross-section class the wall thickness is set to sit at the limit of.
    """

    section_class: int


class LoadCaseConfig(NamedTuple):
    """
    One load case.

    Attributes
    ----------
    name :
        Pattern to apply, a key of `normax.loads.LOAD_CASE_REGISTRY`.
    magnitude :
        Total downward force the case carries.
    """

    name: str
    magnitude: float


class BoundsConfig(NamedTuple):
    """
    The box the force densities may move in.

    Attributes
    ----------
    min :
        Smallest value any force density may take.
    max :
        Largest value any force density may take.

    Notes
    -----
    Not a design constraint. The bounds keep the force densities away from zero,
    where the force density system is singular and a funicular shape stops
    existing, so a bound that binds means the search has left the region where
    the model means anything.
    """

    min: float
    max: float


class OptimizationConfig(NamedTuple):
    """
    What the descent is allowed to spend and where it may go.

    Attributes
    ----------
    iterations :
        Most iterations to spend.
    bounds :
        The box the force densities may move in.
    """

    iterations: int
    bounds: BoundsConfig


class TaskConfig(NamedTuple):
    """
    Everything a run is described by.

    Attributes
    ----------
    structure :
        The arch to build.
    form_finding :
        What the shape is searched from.
    analysis :
        What the frame is analyzed with.
    sizing :
        What the standard is read at.
    optimization :
        What the descent is allowed to spend.
    load_cases :
        The cases the arch carries, the first of which shapes it.
    """

    structure: ArchConfig
    form_finding: FormFindingConfig
    analysis: AnalysisConfig
    sizing: SizingConfig
    optimization: OptimizationConfig
    load_cases: tuple[LoadCaseConfig, ...]


def read_config(path: Path) -> TaskConfig:
    """
    The arch, the load cases and the search a run is described by.

    Parameters
    ----------
    path :
        File to read.

    Returns
    -------
    config :
        The arch, and the settings a design of it is searched for under.

    Raises
    ------
    TypeError
        If the file names a field that does not exist, or omits one that does.

    Notes
    -----
    No container carries a default, so a file missing a field is refused rather
    than quietly completed. The file is the description of the run, and half a
    description is not one.
    """
    document = yaml.safe_load(path.read_text())
    searched = document["optimization"]

    bounds = BoundsConfig(**searched["bounds"])
    load_cases = tuple(LoadCaseConfig(**entry) for entry in document["load_cases"])

    config = TaskConfig(
        structure=ArchConfig(**document["structure"]),
        form_finding=FormFindingConfig(**document["form_finding"]),
        analysis=AnalysisConfig(**document["analysis"]),
        sizing=SizingConfig(**document["sizing"]),
        optimization=OptimizationConfig(searched["iterations"], bounds),
        load_cases=load_cases,
    )

    return config


def build_arch(config: ArchConfig) -> Structure:
    """
    The arch, as topology and a starting geometry.

    Parameters
    ----------
    config :
        The arch to build.

    Returns
    -------
    structure :
        A funicular arch between two pinned supports.
    """
    return build_arch_2d(config.num_edges, config.span, config.rise)


def arch_load_cases(
    structure: Structure,
    config: tuple[LoadCaseConfig, ...],
) -> LoadCases:
    """
    The load case the arch is shaped by, and every case it is checked against.

    Parameters
    ----------
    structure :
        The structure to load.
    config :
        The cases to build, the first of which shapes the arch.

    Returns
    -------
    loads :
        The form-finding case and the checked cases.

    Notes
    -----
    The first case shapes the arch and is checked as well, the others being
    departures from it. A funicular shape carries its own case axially and
    nothing else, so any redistribution has to be carried in bending, which is
    the reason a frame analysis sits between the form finder and the check.
    """
    applied = []
    for load_case in config:
        case = create_loads_by_name(load_case.name, structure, load_case.magnitude)
        applied.append(case)

    return assemble_load_cases(applied)


def initialize_parameters(
    structure: Structure,
    config: TaskConfig,
) -> DesignParameters:
    """
    The design parameters the search starts from.

    Parameters
    ----------
    structure :
        The structure being designed, supplying the member count.
    config :
        The settings the starting values are read from.

    Returns
    -------
    params :
        A uniform force density and a uniform seed diameter, one of each per
        member.
    """
    force_densities = jnp.full(structure.num_edges, config.form_finding.force_density)
    diameters = jnp.full(structure.num_edges, config.analysis.diameter)

    return DesignParameters(force_densities, diameters)


def update_parameters(
    force_densities: Float[Array, "members"],
    params: DesignParameters,
) -> DesignParameters:
    """
    The same design parameters at different force densities.

    Parameters
    ----------
    force_densities :
        Force density of every member. Negative in compression.
    params :
        The parameters supplying everything the optimizer does not vary.

    Returns
    -------
    params :
        The new force densities and the seed diameters.

    Notes
    -----
    The force densities are the only variables of the search, so the objective
    takes an array rather than a container and this is where the container is
    put back together. What it restores is the diameters the frame is analyzed
    with, which the check overwrites.
    """
    return DesignParameters(force_densities, params.diameters)


def main(config_path: Path) -> None:
    """
    Design the arch a file describes, and report what the descent bought.

    Parameters
    ----------
    config_path :
        File naming the arch and the settings a design of it is searched for
        under.
    """
    config = read_config(config_path)
    structure = build_arch(config.structure)
    loads = arch_load_cases(structure, config.load_cases)

    material = Steel()
    catalogue = TubeCatalogue.at_class_limit(material, config.sizing.section_class)

    # Three swappable blocks
    pipeline = StructuralDesignPipeline(
        FdmFormFinder(structure),
        SmaxAnalyzer(structure, catalogue, NORMAL),
        Ec3Sizer(structure, catalogue),
    )

    params = initialize_parameters(structure, config)
    sharpness = jnp.asarray(config.analysis.envelope_sharpness)

    # The objective hands back the design it weighed, so nothing is designed twice.
    def objective(
        force_densities: Float[Array, "members"],
    ) -> tuple[Float[Array, ""], Design]:
        moved = update_parameters(force_densities, params)
        design = pipeline(moved, loads)
        sized = design_envelope(design, sharpness)
        mass = compute_mass(sized)

        return mass, sized

    # Gradient voodoo
    objective_and_gradient = value_and_gradient(objective, has_aux=True)
    (mass, _), gradient = objective_and_gradient(params.force_densities)

    # The same objective handed to a bounded descent, which never sees a block.
    bounds = config.optimization.bounds
    found = minimize_bounded(
        objective,
        params.force_densities,
        bounds=(bounds.min, bounds.max),
        iterations=config.optimization.iterations,
        has_aux=True,
        gradient=objective_and_gradient,
    )

    # The answer, and the design behind it, out of the search that already ran it.
    mass_opt = found.value
    design_opt = found.aux

    print(f"Mass at the start: {float(mass):.9f} t")
    print(f"Gradient in q: {gradient}")
    print(f"Mass after the descent: {float(mass_opt):.9f} t")
    print(f"Saved: {100.0 * (1.0 - mass_opt / mass):.3f} %")
    # Every load case exactly satisfied at the size it demanded, which is the
    # invariant the sizing map exists to hold.
    fully_stressed = float(jnp.min(design_opt.sizes.utilization))
    print(f"Utilization as sized: {fully_stressed:.12f}")

    print("\nHasta la vista, baby!")


if __name__ == "__main__":
    jnp.set_printoptions(precision=12)
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else CONFIG)
