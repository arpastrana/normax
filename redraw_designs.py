# SPDX-License-Identifier: Apache-2.0
"""
Draw every example's figures again from its archives, without searching again.

Each structure's baselines are rebuilt together, so the drawings of one
structure share a framing, a pace and a set of ticks, and a change of style is
one command over `data/` rather than a descent per route. Nothing is optimized:
the answers cannot move, which on a landscape with more than one basin is the
point rather than a convenience.

Run from the repository root:

    uv run python redraw_designs.py                     # every structure
    uv run python redraw_designs.py arch vierendeel     # some of them
    uv run python redraw_designs.py arch --gif          # and write GIFs too
"""

import sys
from pathlib import Path
from typing import Any
from typing import NamedTuple

import numpy as np

from normax.config import RunArguments
from normax.config import RunConfig
from normax.config import read_run_config
from normax.design import DesignProblem
from normax.design import StructuralDesignPipeline
from normax.design import assign_signs
from normax.design import build_design_constraints
from normax.exporting.redraw import build_descent_panel
from normax.exporting.redraw import read_descent_archive
from normax.exporting.redraw import redraw_run
from normax.form_finding import DrawnShapeInitializer
from normax.form_finding import LensShapeInitializer
from normax.form_finding import UniformDensityInitializer
from normax.form_finding import build_form_finder
from normax.form_finding import build_plan_basis
from normax.form_finding import read_lens_shape
from normax.form_finding import read_parabolic_shape
from normax.loads import build_load_cases
from normax.loads import read_polar_plan
from normax.materials import Steel355
from normax.optimization.auglag import DescentHistory
from normax.sections import build_section_catalog
from normax.structures import ArchDescription
from normax.structures import ShellDescription
from normax.structures import TrussDescription
from normax.structures import build_arch_2d
from normax.structures import build_gridshell_3d
from normax.structures import build_vierendeel_2d
from normax.structures import build_warren_2d
from normax.structures import create_groups_shell
from normax.structures import create_groups_vierendeel
from normax.structures import create_groups_warren
from normax.symmetry import build_height_groups
from normax.symmetry import build_section_groups
from normax.symmetry import find_mirror_nodes
from normax.symmetry import find_rotated_nodes
from normax.tesseract import TesseractAnalyzer
from normax.tesseract import TesseractSizer
from normax.visualization import DrawnLimits
from normax.visualization import convert_to_gif
from normax.visualization.animations import pick_frames
from normax.visualization.animations import read_drawn_bounds
from normax.visualization.animations import read_objective_bounds
from normax.visualization.animations import rebuild_walk
from normax.visualization.plots import project_view
from normax.visualization.plots import read_violation_floor

DATA = Path("data")
FIGURES = Path("figures")
ROUTES = ("fdm", "heights", "fixed")
MATERIAL = Steel355()


class ExampleKind(NamedTuple):
    """
    What one example is, in the five ways the four of them differ.

    Attributes
    ----------
    described :
        The description type its `structure` block is read into.
    raise_structure :
        Builds the structure from that description.
    name_families :
        Names the member families a sign guard is stated over.
    open_shape :
        Reads the geometry every route opens on, or None where the drawing is
        the start and no route needs telling.
    fit_start :
        Fits the densities the search leaves from.
    """

    described: type
    raise_structure: Any
    name_families: Any
    open_shape: Any
    fit_start: Any


KINDS = {
    "arch": ExampleKind(
        ArchDescription,
        lambda d: build_arch_2d(d.num_edges, d.span, d.rise),
        None,
        lambda s, c: read_parabolic_shape(s, c.form_finding.height_start),
        lambda c: UniformDensityInitializer(c.form_finding.density_start),
    ),
    "warren": ExampleKind(
        TrussDescription,
        lambda d: build_warren_2d(d.num_bays, d.span, d.depth),
        create_groups_warren,
        lambda s, c: read_lens_shape(s, c.form_finding.density_start),
        lambda c: LensShapeInitializer(c.form_finding.density_start),
    ),
    "vierendeel": ExampleKind(
        TrussDescription,
        lambda d: build_vierendeel_2d(d.num_bays, d.span, d.depth),
        create_groups_vierendeel,
        lambda s, c: read_lens_shape(s, c.form_finding.density_start),
        lambda c: LensShapeInitializer(c.form_finding.density_start),
    ),
    "gridshell": ExampleKind(
        ShellDescription,
        lambda d: build_gridshell_3d(
            d.num_rings, d.num_spokes, d.radius, d.rise, d.oculus, d.braced
        ),
        create_groups_shell,
        None,
        lambda c: DrawnShapeInitializer(c.form_finding.density_start),
    ),
}


def read_archive_name(example: str, route: str) -> Path:
    """
    The archive a route wrote, named as the example's export target names it.
    """
    stem = example if route == "fdm" else f"{example}_{route}"

    return DATA / f"{stem}.npz"


def build_problem(example: str, route: str) -> tuple[DesignProblem, RunConfig[Any]]:
    """
    One example's design task, built exactly as `examples/<example>.py` builds it.

    Parameters
    ----------
    example :
        Which example to build, a key of `KINDS`.
    route :
        Shape parametrization to build the form finder for.

    Returns
    -------
    built :
        The problem, and the run description it was read from.
    """
    kind = KINDS[example]
    arguments = RunArguments(Path("examples") / f"{example}.yaml", route)
    config = read_run_config(arguments, kind.described)
    structure = kind.raise_structure(config.structure)
    loads = build_load_cases(structure, config.load_cases)
    catalog = build_section_catalog(MATERIAL, config.sizing.section_class)

    mirror = find_mirror_nodes(structure, config.form_finding.mirror)
    basis = build_plan_basis(structure, mirror, config.form_finding.basis)
    folded = mirror if config.sizing.fold_mirror else None
    rotation = None
    if config.sizing.fold_polar:
        rotation = find_rotated_nodes(structure, read_polar_plan(structure).num_spokes)
    section_groups = build_section_groups(structure, (folded, rotation))
    lifted = mirror if config.form_finding.fold_heights else None
    height_groups = build_height_groups(structure, (lifted,))

    opened = None if kind.open_shape is None else kind.open_shape(structure, config)
    form_finder = build_form_finder(
        structure, basis, config.form_finding, height_groups, opened
    )
    pipeline = StructuralDesignPipeline(
        form_finder,
        TesseractAnalyzer(structure, catalog, config.analysis.backend),
        TesseractSizer(structure, catalog, config.sizing.backend),
    )

    guarded = None
    if kind.name_families is not None:
        families = kind.name_families(config.structure)
        guarded = assign_signs(config.constraints, families, structure.num_edges)
    density_start = kind.fit_start(config)(structure, loads.formfinding, basis, guarded)
    constraints = build_design_constraints(config.constraints, guarded, density_start)
    problem = DesignProblem(structure, pipeline, loads, constraints, section_groups)

    return problem, config


def read_shared_limits(example: str) -> DrawnLimits:
    """
    Limits wide enough for every baseline of one example.

    Parameters
    ----------
    example :
        Which example to take the union over.

    Returns
    -------
    limits :
        The union of what each baseline would have set on its own: the drawing
        extents, the longest walk, and the two curve axes.
    """
    extents = []
    counts = []
    objectives = []
    violations = []
    for route in ROUTES:
        archive = read_archive_name(example, route)
        if not archive.exists():
            continue
        problem, config = build_problem(example, route)
        history = read_descent_archive(archive)
        counted = int(np.size(history.objectives))
        counts.append(counted)
        picked = pick_frames(counted)
        drawn = DescentHistory(
            history.iterates[picked],
            history.objectives[picked],
            history.violations[picked],
            history.round_index[picked],
        )
        walked = rebuild_walk(problem, drawn)
        walked = walked._replace(
            shapes=[project_view(shape) for shape in walked.shapes]
        )
        extents.append(read_drawn_bounds(walked))
        objectives.append(read_objective_bounds(np.asarray(history.objectives)))
        panel = build_descent_panel(problem, history, config)
        floor = read_violation_floor(panel.traces)
        highest = float(np.maximum(np.asarray(history.violations), floor).max())
        violations.append((floor, highest * 2.0))

    across = (min(e[0][0] for e in extents), max(e[0][1] for e in extents))
    upward = (min(e[1][0] for e in extents), max(e[1][1] for e in extents))
    objective = (min(o[0] for o in objectives), max(o[1] for o in objectives))
    violation = (min(v[0] for v in violations), max(v[1] for v in violations))

    return DrawnLimits(across, upward, max(counts), objective, violation)


def redraw_example(example: str, gifs: bool) -> None:
    """
    Redraw every baseline of one example, reporting what each archive wrote.
    """
    limits = read_shared_limits(example)
    print(f"\n{example}")
    print(
        f"  framing across {limits.across[0]:.1f}..{limits.across[1]:.1f}"
        f"   upward {limits.upward[0]:.1f}..{limits.upward[1]:.1f}"
    )
    print(
        f"  curves  steps 0..{limits.steps - 1}"
        f"   objective {limits.objective[0]:.6f}..{limits.objective[1]:.6f}"
        f"   violation {limits.violation[0]:.2e}..{limits.violation[1]:.2e}"
    )
    for route in ROUTES:
        archive = read_archive_name(example, route)
        if not archive.exists():
            print(f"  {route}: no archive at {archive}, skipped")
            continue
        problem, config = build_problem(example, route)
        written = redraw_run(problem, config, archive, FIGURES, limits)
        names = ", ".join(path.name for path in written)
        print(f"  {route}: {archive.name} -> {names}")
        if gifs:
            video = FIGURES / f"{archive.stem}_optimization.mp4"
            if video.exists():
                gif = video.with_suffix(".gif")
                convert_to_gif(video, gif)
                print(f"          + {gif.name}")


def main(asked: tuple[str, ...], gifs: bool) -> None:
    """
    Redraw every named example.
    """
    for example in asked:
        redraw_example(example, gifs)


if __name__ == "__main__":
    given = [word for word in sys.argv[1:] if not word.startswith("--")]
    wanted = tuple(given) or tuple(KINDS)
    unknown = [word for word in wanted if word not in KINDS]
    if unknown:
        raise SystemExit(f"unknown example {unknown}, known: {list(KINDS)}")
    main(wanted, "--gif" in sys.argv[1:])
