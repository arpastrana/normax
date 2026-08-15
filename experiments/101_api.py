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

**The objective is a mass with a floor under the shortest member.** Nothing in a
member check objects to a vanishing member, and two things reward one: its mass
is an area times a length, and its buckling length is its own length, so as it
shortens it becomes both free and unbucklable. Left alone the search collapses
members rather than improving the form, and the mass it reports is the collapse.
The floor is a multiplicative penalty reading a ratio of lengths, so it needs no
mass scale, and a weight of zero in the file turns it off.

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
from normax.design import settle_diameters
from normax.form_finding.fdm import FdmFormFinder
from normax.loads import LoadCases
from normax.loads import assemble_load_cases
from normax.loads import create_loads_by_name
from normax.materials import Steel355
from normax.optimization import minimize_bounded
from normax.optimization import penalized_mass
from normax.optimization import value_and_gradient
from normax.sizing.ec3 import Ec3Sizer
from normax.sizing.ec3 import thinnest_family
from normax.structures import Structure
from normax.structures import build_arch_2d
from normax.visualization import figure_trajectory

# The arch and the search, unless another file is named on the command line.
CONFIG = Path(__file__).with_name("arch.yaml")

FIGURES = Path(__file__).resolve().parent.parent / "figures"

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
    """

    diameter: float


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


class FloorConfig(NamedTuple):
    """
    The shortest member the design is allowed, and how hard it is held there.

    Attributes
    ----------
    fraction :
        Shortest member allowed, as a fraction of the nominal bay — the span
        divided by the member count.
    sharpness :
        Sharpness of the smooth minimum the penalty reads the shortest member
        with.
    weight :
        Size of the inflation at a member of zero length. Zero turns the floor
        off.

    Notes
    -----
    A fraction rather than a length, so the same file describes the same intent
    at any span or any discretization. A design constraint, unlike the bounds
    beside it, and the only one this run carries.
    """

    fraction: float
    sharpness: float
    weight: float


class OptimizationConfig(NamedTuple):
    """
    What the descent is allowed to spend and where it may go.

    Attributes
    ----------
    iterations :
        Most iterations to spend.
    settling_passes :
        Most analyses to spend closing the analysis and the check at the answer.
    settling_tolerance :
        Largest fractional movement in any diameter that counts as settled.
    envelope_sharpness :
        Sharpness of the envelope reconciling the load cases into one size.
    bounds :
        The box the force densities may move in.
    length_floor :
        The shortest member the design is allowed.

    Notes
    -----
    **The sharpness belongs to the search rather than to any stage.** Reconciling
    the load cases is smoothing, so no block sees it: the analysis reconciles
    nothing and the check reads one load case at a time. What it is is a
    continuation parameter — the envelope approaches the true largest size as it
    grows, so raising it across rounds drives the design onto the smallest adequate
    one from above, which is a property of the descent and sits beside its budget.

    **The settling budget is not spent by the descent.** The frame is analyzed at
    the seed diameters for the whole search, and the passes are what the answer is
    re-analyzed with afterwards to measure what that shortcut cost. They buy a
    number for the writeup rather than a better design.
    """

    iterations: int
    settling_passes: int
    settling_tolerance: float
    envelope_sharpness: float
    bounds: BoundsConfig
    length_floor: FloorConfig


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
    searched = dict(document["optimization"])

    bounds = BoundsConfig(**searched.pop("bounds"))
    length_floor = FloorConfig(**searched.pop("length_floor"))
    optimization = OptimizationConfig(
        bounds=bounds, length_floor=length_floor, **searched
    )
    load_cases = tuple(LoadCaseConfig(**entry) for entry in document["load_cases"])

    config = TaskConfig(
        structure=ArchConfig(**document["structure"]),
        form_finding=FormFindingConfig(**document["form_finding"]),
        analysis=AnalysisConfig(**document["analysis"]),
        sizing=SizingConfig(**document["sizing"]),
        optimization=optimization,
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

    # The one place the standard is named. Everything EC3-flavored — the
    # partial factors, the class-limit wall — is derived inside the block.
    grade = Steel355()
    family = thinnest_family(grade, config.sizing.section_class)
    sizer = Ec3Sizer(structure, family)

    # The analysis is configured with one tube; the check is what chooses between
    # them. The tube is drawn from the same family, read as bare geometry.
    section = family(config.analysis.diameter)

    # Three swappable blocks
    pipeline = StructuralDesignPipeline(
        FdmFormFinder(structure),
        SmaxAnalyzer(structure, section),
        sizer,
    )

    params = initialize_parameters(structure, config)
    searched = config.optimization
    sharpness = jnp.asarray(searched.envelope_sharpness)

    # The floor is stated as a fraction of the nominal bay, so it is a length here.
    floor = searched.length_floor
    floor_length = floor.fraction * config.structure.span / config.structure.num_edges

    # The objective hands back the design it weighed, so nothing is designed twice.
    def objective(params: DesignParameters) -> tuple[Float[Array, ""], Design]:
        design = pipeline(params, loads)
        sized = design_envelope(design, sharpness)
        mass = compute_mass(sized)
        penalized = penalized_mass(
            mass,
            sized.shape.lengths,
            floor_length,
            beta=floor.sharpness,
            weight=floor.weight,
        )

        return penalized, sized

    # Gradient voodoo. Differentiating the container reaches both of its leaves,
    # and the force densities are the leaf a descent is allowed to move.
    objective_and_gradient = jax.jit(jax.value_and_grad(objective, has_aux=True))
    (value, design), gradient = objective_and_gradient(params)
    mass = compute_mass(design)

    # The force densities are the only thing the descent moves, so the diameters
    # the frame is analyzed with stay at the seed for the whole search.
    def weigh_shape(force_densities: Float[Array, "members"]):
        return objective(DesignParameters(force_densities, params.diameters))

    bounds = searched.bounds
    found = minimize_bounded(
        weigh_shape,
        params.force_densities,
        bounds=(bounds.min, bounds.max),
        iterations=searched.iterations,
        has_aux=True,
        gradient=value_and_gradient(weigh_shape, has_aux=True),
    )

    # The answer, and the design behind it, out of the search that already ran it.
    # The search reads the penalized value; the mass is what the design weighs.
    value_opt = found.value
    design_opt = found.aux
    answer = found.trajectory.q[-1]
    mass_opt = compute_mass(design_opt)

    # What the shortcut costs, measured rather than assumed: the frame is analyzed
    # at the seed throughout, so the answer is re-analyzed once at the sections the
    # check demanded of it. Forward passes only, and no gradient is spent here.
    settled = settle_diameters(
        objective,
        DesignParameters(answer, params.diameters),
        settling_passes=searched.settling_passes,
        settling_tolerance=searched.settling_tolerance,
    )
    _, settled_design = objective(DesignParameters(answer, settled))
    honest = compute_mass(settled_design)

    slope = gradient.force_densities
    lengths = design_opt.shape.lengths
    shortest = float(jnp.min(lengths))
    stubs = int(jnp.sum(lengths < floor_length))

    print(f"Mass at the start: {float(mass):.9f} t")
    print(f"Objective at the start: {float(value):.9f} t")
    print(f"L1 norm of the initial gradient: {jnp.linalg.norm(slope, ord=1)}")
    print(f"Mass after the descent: {float(mass_opt):.9f} t")
    print(f"Objective after the descent: {float(value_opt):.9f} t")
    print(f"Saved: {100.0 * (1.0 - mass_opt / mass):.3f} %")
    print(
        f"Shortest member: {shortest:.1f} mm against a floor of {floor_length:.1f} mm"
    )
    print(f"Members under the floor: {stubs} of {structure.num_edges}")
    print(f"The answer re-analyzed at its own sections: {float(honest):.9f} t")
    coupling_error = 100.0 * (honest / mass_opt - 1.0)
    print(f"Cost of analyzing at the seed throughout: {coupling_error:+.5f} %")
    # Every load case exactly satisfied at the size it demanded, which is the
    # invariant the sizing map exists to hold.
    fully_stressed = float(jnp.min(design_opt.sizes.utilization))
    print(f"Utilization as sized: {fully_stressed:.12f}")

    trajectories = (found.trajectory,)
    titles = ("penalized descent",)
    descent_figure = figure_trajectory(trajectories, titles=titles)
    FIGURES.mkdir(exist_ok=True)
    descent_figure.savefig(FIGURES / "101_trajectory.png", dpi=200)
    print(f"Descent figure: {FIGURES / '101_trajectory.png'}")

    print("\nHasta la vista, baby!")


if __name__ == "__main__":
    jnp.set_printoptions(precision=12)
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else CONFIG)
