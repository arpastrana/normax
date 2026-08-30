# SPDX-License-Identifier: Apache-2.0
"""
Draw the gridshell's figures again from its archives, without searching again.

The three baselines are rebuilt in one process, so a change of drawing style is
one command over `data/` rather than three descents -- and the answers cannot
move, which on a landscape with more than one basin is the point rather than a
convenience.

Run from the repository root:

    uv run python redraw_gridshell.py                 # all three baselines
    uv run python redraw_gridshell.py fdm             # one of them
"""

import sys
from pathlib import Path

from normax.config import RunArguments
from normax.config import read_run_config
from normax.design import DesignProblem
from normax.design import StructuralDesignPipeline
from normax.design import assign_signs
from normax.design import build_design_constraints
from normax.exporting.redraw import read_descent_archive
from normax.exporting.redraw import redraw_run
from normax.form_finding import DrawnShapeInitializer
from normax.form_finding import build_form_finder
from normax.form_finding import build_plan_basis
from normax.loads import build_load_cases
from normax.loads import read_polar_plan
from normax.materials import Steel355
from normax.sections import build_section_catalog
from normax.structures import ShellDescription
from normax.structures import build_gridshell_3d
from normax.structures import create_groups_shell
from normax.symmetry import build_height_groups
from normax.symmetry import build_section_groups
from normax.symmetry import find_mirror_nodes
from normax.symmetry import find_rotated_nodes
from normax.tesseract import TesseractAnalyzer
from normax.tesseract import TesseractSizer
from normax.visualization import DrawnLimits
from normax.visualization.animations import pick_frames
from normax.visualization.animations import read_drawn_bounds
from normax.visualization.animations import read_objective_bounds
from normax.visualization.animations import rebuild_walk
from normax.visualization.plots import project_view
from normax.visualization.plots import read_violation_floor

CONFIG = Path("examples/gridshell.yaml")
DATA = Path("data")
FIGURES = Path("figures")
ROUTES = ("fdm", "heights", "fixed")
MATERIAL = Steel355()


def read_archive_name(route: str) -> Path:
    """
    The archive a route wrote, named as the example's export target names it.
    """
    stem = "gridshell" if route == "fdm" else f"gridshell_{route}"

    return DATA / f"{stem}.npz"


def build_shell_problem(route: str) -> tuple[DesignProblem, object]:
    """
    The gridshell design task, built exactly as `examples/gridshell.py` builds it.

    Parameters
    ----------
    route :
        Shape parametrization to build the form finder for.

    Returns
    -------
    built :
        The problem, and the run description it was read from.
    """
    config = read_run_config(RunArguments(CONFIG, route), ShellDescription)
    described = config.structure
    structure = build_gridshell_3d(
        described.num_rings,
        described.num_spokes,
        described.radius,
        described.rise,
        described.oculus,
        described.braced,
    )
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

    form_finder = build_form_finder(
        structure, basis, config.form_finding, height_groups
    )
    pipeline = StructuralDesignPipeline(
        form_finder,
        TesseractAnalyzer(structure, catalog, config.analysis.backend),
        TesseractSizer(structure, catalog, config.sizing.backend),
    )

    groups = create_groups_shell(described)
    guarded = assign_signs(config.constraints, groups, structure.num_edges)
    initializer = DrawnShapeInitializer(config.form_finding.density_start)
    density_start = initializer(structure, loads.formfinding, basis, guarded)
    constraints = build_design_constraints(config.constraints, guarded, density_start)
    problem = DesignProblem(structure, pipeline, loads, constraints, section_groups)

    return problem, config


def read_shared_limits(routes: tuple[str, ...]) -> DrawnLimits:
    """
    Limits wide enough for every baseline, so all of them share one framing.

    Parameters
    ----------
    routes :
        Baselines to take the union over.

    Returns
    -------
    limits :
        The union of what each baseline would have set on its own: the drawing
        extents, the longest walk, and the two curve axes.

    Notes
    -----
    Read over the walks rather than the answers, since a film frames the whole
    descent. One forward pass per drawn frame per baseline, which is the work
    the drawing itself does and is why this is computed once and handed down.
    """
    import numpy as np

    from normax.exporting.redraw import build_descent_panel
    from normax.optimization.auglag import DescentHistory

    extents = []
    counts = []
    objectives = []
    violations = []
    for route in routes:
        archive = read_archive_name(route)
        if not archive.exists():
            continue
        problem, config = build_shell_problem(route)
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


def main(wanted: tuple[str, ...]) -> None:
    """
    Redraw every named baseline, reporting what each archive held and wrote.
    """
    limits = read_shared_limits(ROUTES)
    print(
        f"shared framing : across {limits.across[0]:.1f}..{limits.across[1]:.1f}"
        f"   upward {limits.upward[0]:.1f}..{limits.upward[1]:.1f}"
    )
    print(
        f"shared curves  : steps 0..{limits.steps - 1}"
        f"   objective {limits.objective[0]:.6f}..{limits.objective[1]:.6f}"
        f"   violation {limits.violation[0]:.2e}..{limits.violation[1]:.2e}"
    )
    for route in wanted:
        archive = read_archive_name(route)
        if not archive.exists():
            print(f"{route}: no archive at {archive}, skipped")
            continue
        problem, config = build_shell_problem(route)
        written = redraw_run(problem, config, archive, FIGURES, limits)
        names = ", ".join(path.name for path in written)
        print(f"{route}: {archive.name} -> {names}")


if __name__ == "__main__":
    asked = tuple(sys.argv[1:]) or ROUTES
    unknown = [route for route in asked if route not in ROUTES]
    if unknown:
        raise SystemExit(f"unknown route {unknown}, known: {list(ROUTES)}")
    main(asked)
