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

Run with `uv run --group pipeline --group viz python experiments/101_api.py
[arch.yaml]`. The run ends in a viewer holding the initial and the optimized
designs, and returns when its window closes.
"""

import os
import sys
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import vix
import yaml
from jaxtyping import Array
from jaxtyping import Float
from smax import LoadCase

from normax.analysis import SmaxAnalyzer
from normax.analysis import frame_model
from normax.analysis import normal_axis
from normax.design import Design
from normax.design import DesignParameters
from normax.design import StructuralDesignPipeline
from normax.design import compute_mass
from normax.design import design_envelope
from normax.design import settle_diameters
from normax.form_finding import FdmFormFinder
from normax.loads import LoadCases
from normax.loads import assemble_load_cases
from normax.loads import create_loads_by_name
from normax.materials import Steel355
from normax.optimization import minimize_bounded
from normax.optimization import penalized_mass
from normax.optimization import value_and_gradient
from normax.replay import save_trajectory
from normax.sizing import BlueprintSizer
from normax.sizing import Ec3Sizer
from normax.sizing import build_section_family
from normax.structures import Structure
from normax.structures import build_arch_2d
from normax.tesseract import BACKEND_VARIABLE
from normax.tesseract import BlueprintClient
from normax.tesseract import TesseractAnalyzer
from normax.tesseract import blueprint_tesseract
from normax.tesseract import local_chain
from normax.visualization import figure_trajectory

# The arch and the search, unless another file is named on the command line.
CONFIG = Path(__file__).with_name("arch.yaml")

FIGURES = Path(__file__).resolve().parent.parent / "figures"

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"

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
    backend :
        Which solver fills the analysis slot: `smax` traces the frame in this
        process, `opensees` crosses a Tesseract boundary to a host solver
        carrying DDM sensitivities — the planar demo, so the arch alone.
    """

    diameter: float
    backend: str


class SizingConfig(NamedTuple):
    """
    What the standard is read at, and which implementation reads it.

    Attributes
    ----------
    section_class :
        Cross-section class the wall thickness is set to sit at the limit of.
    backend :
        Which sizer fills the check's slot: `ec3` for the full member check,
        `blueprint` for the cross-section-only check hosted from Blueprints,
        `blueprint_tesseract` for that same check reached across a Tesseract
        boundary.

    Notes
    -----
    The two backends implement different physics — Blueprints has no member
    buckling — so they are two design philosophies behind one workflow, not
    two routes to one answer. The tubes are drawn from the same class-limit
    family either way, so whatever differs in the designs is the check.
    """

    section_class: int
    backend: str


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


def parse_config(text: str) -> TaskConfig:
    """
    The arch, the load cases and the search a run is described by.

    Parameters
    ----------
    text :
        Text of the file describing the run, taken verbatim so the same bytes
        can ride inside a trajectory artifact.

    Returns
    -------
    config :
        The arch, and the settings a design of it is searched for under.

    Raises
    ------
    TypeError
        If the text names a field that does not exist, or omits one that does.

    Notes
    -----
    No container carries a default, so a file missing a field is refused rather
    than quietly completed. The file is the description of the run, and half a
    description is not one.
    """
    document = yaml.safe_load(text)
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


def build_pipeline(
    structure: Structure,
    config: TaskConfig,
) -> StructuralDesignPipeline:
    """
    The three blocks the search composes, built on one structure.

    Parameters
    ----------
    structure :
        The structure every block is built from.
    config :
        The settings the standard is read at and the frame is analyzed with.

    Returns
    -------
    pipeline :
        A form finder, a frame analysis and a code check, composed.
    """
    # The one place the standard is named, and now the one place it is picked.
    # Both backends draw tubes from the same class-limit family, so whatever
    # differs downstream is the check itself, not the geometry.
    grade = Steel355()
    family = build_section_family(grade, config.sizing.section_class)
    backend = config.sizing.backend
    if backend == "ec3":
        sizer = Ec3Sizer(structure, family)
    elif backend == "blueprint":
        sizer = BlueprintSizer(structure, family)
    elif backend == "blueprint_tesseract":
        sizer = BlueprintClient(structure, blueprint_tesseract(), family)
    else:
        raise ValueError(
            f"unknown sizing backend {backend!r}: ec3, blueprint or blueprint_tesseract"
        )

    # The analysis is configured with one tube; the check is what chooses between
    # them. The tube is drawn from the same family, read as bare geometry.
    section = family(config.analysis.diameter)

    solver = config.analysis.backend
    if solver == "smax":
        analyzer = SmaxAnalyzer(structure, section)
    elif solver == "opensees":
        # The stage reads its solver from the environment; an experiment picks
        # once for the whole process rather than per block.
        os.environ[BACKEND_VARIABLE] = "opensees"
        chain = local_chain()
        analyzer = TesseractAnalyzer(
            structure, chain.analysis, family, normal_axis(structure)
        )
    else:
        raise ValueError(f"unknown analysis backend {solver!r}: smax or opensees")

    # Three swappable blocks
    pipeline = StructuralDesignPipeline(
        FdmFormFinder(structure),
        analyzer,
        sizer,
    )

    return pipeline


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


def view_designs(
    structure: Structure,
    analyzer: SmaxAnalyzer | TesseractAnalyzer,
    loads: LoadCases,
    designs: dict[str, Design],
    case_names: tuple[str, ...],
) -> None:
    """
    Inspect designs interactively, in the frame solver's own terms.

    Parameters
    ----------
    structure :
        The structure supplying the connectivity and the supported nodes.
    analyzer :
        The analysis block, whose model builder and solve the viewer reads.
    loads :
        The checked load cases, each solved and drawn for every design.
    designs :
        The designs to draw, keyed by the name each appears under.
    case_names :
        Name of every checked case, naming its response in the viewer.

    Notes
    -----
    Every design is gathered through `frame_model` — the same builder the
    analysis block compiled its assembly from — at its own form-found
    geometry and reconciled sections, so what the viewer draws is the frame
    the analysis saw. Each response comes from `SmaxAnalyzer.solve_response`,
    the same injected assembly and solve the member forces were read from,
    so the deformations and the diagrams are the analysis, not a retelling.

    Each case's loads are registered under their own name: the viewer's
    `add` replaces a same-named registration, so a loads group sharing the
    response's name would tear the response down instead of joining it.

    A crossed analyzer carries no response solver, so its designs are
    re-solved in process at their own geometry and sections — a retelling,
    and one the backend-agreement suite bounds tightly.

    Blocks until the window closes.
    """
    if isinstance(analyzer, TesseractAnalyzer):
        analyzer = SmaxAnalyzer(structure, analyzer.family(100.0))

    viewer = vix.Viewer()

    for name, design in designs.items():
        sections = design.sizes.sections
        frame = frame_model(structure, design.shape.xyz, sections)
        viewer.add(frame, name=name)

        for index, case_name in enumerate(case_names):
            load_case = loads.analysis[index]
            response = analyzer.solve_response(
                design.shape.xyz,
                sections.diameter,
                load_case,
            )
            viewer.add(
                response,
                name=f"{name}-{case_name}",
                structure=name,
                show_deformation=False,
                show_forces=("nx", "my"),
            )

            viewer.add(
                LoadCase.from_array(load_case, frame),
                name=f"{name}-{case_name}-loads",
                structure=name,
            )

    viewer.show()


def main(config_path: Path) -> None:
    """
    Design the arch a file describes, and report what the descent bought.

    Parameters
    ----------
    config_path :
        File naming the arch and the settings a design of it is searched for
        under.
    """
    config_text = config_path.read_text()
    config = parse_config(config_text)
    structure = build_arch(config.structure)
    loads = arch_load_cases(structure, config.load_cases)
    pipeline = build_pipeline(structure, config)
    print(f"Sizing backend: {config.sizing.backend}")
    print(f"Analysis backend: {config.analysis.backend}")

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
        sharpness=searched.envelope_sharpness,
        has_aux=True,
        gradient=value_and_gradient(weigh_shape, has_aux=True),
    )

    # The record a replay reconstructs every intermediate design from, with the
    # file that described the run embedded so the artifact is self-contained.
    ARTIFACTS.mkdir(exist_ok=True)
    artifact_path = ARTIFACTS / "101_trajectory.npz"
    save_trajectory(artifact_path, found.trajectory, config_text)
    print(f"Trajectory artifact: {artifact_path}")

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

    # The two designs, inspected where the analysis lives. The initial one is
    # the seed the descent left, the optimized one the answer analyzed at the
    # sections it demanded of itself.
    designs = {"initial": design, "optimized": settled_design}
    case_names = tuple(load_case.name for load_case in config.load_cases)
    view_designs(structure, pipeline.analyzer, loads, designs, case_names)

    print("\nHasta la vista, baby!")


if __name__ == "__main__":
    jnp.set_printoptions(precision=12)
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else CONFIG)
