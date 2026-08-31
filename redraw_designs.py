# SPDX-License-Identifier: Apache-2.0
"""
Draw every example's figures again from its archives, without searching again.

Each structure's baselines are rebuilt together, so the drawings of one
structure share a framing, a pace and a set of ticks, and a change of style is
one command over `data/` rather than a descent per route. Nothing is optimized:
the answers cannot move, which on a landscape with more than one basin is the
point rather than a convenience.

The planar structures share one drawing box on top of that, so an arch and a
truss can be set beside each other on a slide and their ground lines agree.
The box is the union over the whole group whichever of its members is named,
which costs a limits pass per member and is what keeps a figure from depending
on what was asked for.

Stills only unless a film is asked for: redrawing the drawings takes seconds
and redrawing the animations takes minutes, so the cheap pass is the default
whatever `output.animate` says in the run description.

Run from the repository root:

    uv run python redraw_designs.py                     # every structure
    uv run python redraw_designs.py arch vierendeel     # some of them
    uv run python redraw_designs.py arch --film         # animations as well
    uv run python redraw_designs.py arch --film --gif   # and a GIF of each
    uv run python redraw_designs.py arch --web         # a GIF light enough to embed
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
from normax.visualization import GIF_FOR_READING
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

# Examples drawn to one box. The three planar ones span the same 10000 mm and
# stand on the same ground, so a shared box makes them comparable rather than
# merely equal in size; the shell shares its own with nothing.
FRAMINGS = (("arch", "warren", "vierendeel"), ("gridshell",))
MATERIAL = Steel355()


class GifRequest(NamedTuple):
    """
    Which GIFs of a film are wanted.

    Attributes
    ----------
    faithful :
        Every frame at the film's own width, which is heavy by construction.
    reduced :
        Narrowed and thinned until a page can carry it.
    """

    faithful: bool
    reduced: bool


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


def count_archives(example: str) -> int:
    """
    How many of one example's baselines left an archive to redraw.
    """
    written = [read_archive_name(example, route).exists() for route in ROUTES]

    return sum(written)


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

    Raises
    ------
    ValueError
        If no baseline of the example wrote an archive, so there is nothing to
        read a limit off.

    Notes
    -----
    The drawing extents this returns are the example's own. A caller drawing
    several examples to one box widens them afterwards; the curve axes are
    never widened, since an objective is in the example's own tonnes and a
    walk is as long as that example's descent ran.
    """
    if count_archives(example) == 0:
        raise ValueError(f"no baseline of {example} wrote an archive under {DATA}")

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
        # The standing view alone, spin or no spin. A film that turns is drawn
        # inside this box rather than widening it: the drawings beside it read
        # the same limits, and measured, every view of the turn fits with the
        # legend and caption room the film adds to spare.
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


def read_framing_limits(group: tuple[str, ...]) -> dict[str, DrawnLimits]:
    """
    Every example of one group, framed alike and curved on its own axes.

    Parameters
    ----------
    group :
        Examples sharing a drawing box, an entry of `FRAMINGS`.

    Returns
    -------
    framed :
        The limits to draw each example of the group under, keyed by example.
        Examples that wrote no archive are absent.

    Notes
    -----
    Only the drawing box is shared. Holding the curve axes across examples
    would flatten the lighter structure's descent into a line, since the masses
    differ severalfold, and would draw every walk out to the longest of them.
    What a slide needs is the shapes at one scale, which is the box alone.
    """
    limits = {}
    for example in group:
        if count_archives(example) == 0:
            continue
        limits[example] = read_shared_limits(example)
    if not limits:
        return limits

    across = (
        min(limit.across[0] for limit in limits.values()),
        max(limit.across[1] for limit in limits.values()),
    )
    upward = (
        min(limit.upward[0] for limit in limits.values()),
        max(limit.upward[1] for limit in limits.values()),
    )

    return {
        example: limit._replace(across=across, upward=upward)
        for example, limit in limits.items()
    }


def convert_films(archive: Path, wanted: GifRequest) -> tuple[str, ...]:
    """
    Every GIF asked for of one run's film, reported by name.

    Parameters
    ----------
    archive :
        The `.npz` whose stem names the film and the GIFs beside it.
    wanted :
        Which GIFs to write.

    Returns
    -------
    written :
        Name of every GIF written, in the order written.
    """
    video = FIGURES / f"{archive.stem}_optimization.mp4"
    if not video.exists():
        return ()

    written = []
    if wanted.faithful:
        faithful = video.with_suffix(".gif")
        convert_to_gif(video, faithful)
        written.append(faithful.name)
    if wanted.reduced:
        # Named apart rather than replacing the faithful one: the two answer
        # different questions, and a page cannot carry the heavy one.
        light = video.with_name(f"{video.stem}_web.gif")
        convert_to_gif(video, light, GIF_FOR_READING)
        written.append(light.name)

    return tuple(written)


def redraw_example(
    example: str,
    limits: DrawnLimits,
    films: bool,
    wanted: GifRequest,
) -> None:
    """
    Redraw every baseline of one example, reporting what each archive wrote.

    Parameters
    ----------
    example :
        Which example to redraw, a key of `KINDS`.
    limits :
        The limits every baseline is held to, from `read_framing_limits`.
    films :
        Whether to write the animation as well as the drawings.
    wanted :
        Which GIFs to write of each film already beside the archive.
    """
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
        # Overridden rather than passed on: `redraw_run` reads the run
        # description for what to write, and a sixth argument to say otherwise
        # is a worse answer than describing the run that is actually being made.
        if not films:
            config = config._replace(output=config.output._replace(animate=False))
        written = redraw_run(problem, config, archive, FIGURES, limits)
        names = ", ".join(path.name for path in written)
        print(f"  {route}: {archive.name} -> {names}")
        converted = convert_films(archive, wanted)
        if converted:
            print(f"          + {', '.join(converted)}")


def main(asked: tuple[str, ...], films: bool, wanted: GifRequest) -> None:
    """
    Redraw every named example, a framing group at a time.

    Parameters
    ----------
    asked :
        Examples to redraw, keys of `KINDS`.
    films :
        Whether to write each descent's animation as well as its drawings.
    wanted :
        Which GIFs to convert each film to.
    """
    for group in FRAMINGS:
        chosen = [example for example in group if example in asked]
        if not chosen:
            continue
        framed = read_framing_limits(group)
        for example in chosen:
            if example not in framed:
                print(f"\n{example}: no archive under {DATA}, skipped")
                continue
            redraw_example(example, framed[example], films, wanted)


FLAGS = ("--film", "--gif", "--web")

if __name__ == "__main__":
    asked_flags = [word for word in sys.argv[1:] if word.startswith("-")]
    # Refused rather than ignored: a flag read as no flag leaves every example
    # named, and this redraws all of them by way of answering `--help`.
    strange = [word for word in asked_flags if word not in FLAGS]
    if strange:
        raise SystemExit(
            f"unknown option {strange}, known: {list(FLAGS)}\n"
            f"usage: redraw_designs.py [{'|'.join(KINDS)}] ..."
            f" [--film] [--gif] [--web]"
        )
    given = [word for word in sys.argv[1:] if not word.startswith("-")]
    wanted = tuple(given) or tuple(KINDS)
    unknown = [word for word in wanted if word not in KINDS]
    if unknown:
        raise SystemExit(f"unknown example {unknown}, known: {list(KINDS)}")
    gifs = GifRequest("--gif" in asked_flags, "--web" in asked_flags)
    main(wanted, "--film" in asked_flags, gifs)
