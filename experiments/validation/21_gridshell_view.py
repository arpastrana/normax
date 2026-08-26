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
Eyeball the gridshell generator before any experiment stands on it.

The generator draws a polar grid on a spherical cap: an apex, a node per spoke
on every ring, radial members running outward and hoop members closing every
ring but the outermost, which is pinned and needs none: a member spanning two
supports appears in no equilibrium equation and moves no node. This script puts
that output in front of the eye before the gridshell optimization is built, so
the drawn shape, the rise and the candidate seed diameters are reviewed as
geometry rather than discovered as a misshapen answer three stages downstream.

Three things are reported and one is drawn.

    counts      nodes, members split radial from hoop, and supports, so the
                sizes of every downstream container are known in advance
    rings       one row per ring — plan radius, height above the boundary
                plane, and the radial and hoop member lengths meeting it —
                the cap's shape read as numbers
    seeds       one row per candidate seed diameter — the wall the family
                pairs with it and the shell's mass at that uniform size —
                what a starting guess costs before any check trims it
    viewer      the same structure drawn once per seed diameter as tubes at
                true size, each frame named apart and switched from the panel

The viewer draws the generated starting geometry: no form finding, no
analysis, no code check. It opens last because it blocks, and the YAML's
`viewer.enabled` switch turns it off for a headless run.

Run with `uv run --group pipeline --group viz python
experiments/21_gridshell_view.py [gridshell.yaml]`.
"""

import sys
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import vix
import yaml

from normax.analysis import frame_model
from normax.materials import Steel355
from normax.reporting import Report
from normax.reporting import ReportColumn
from normax.sections import TubeFamily
from normax.sizing import build_section_family
from normax.structures import Structure
from normax.structures import build_gridshell_3d
from normax.structures import member_lengths


class GridshellSketch(NamedTuple):
    """
    The cap the generator is asked to draw.

    Attributes
    ----------
    num_rings :
        Number of rings between the apex and the boundary, boundary included.
    num_spokes :
        Number of spokes radiating from the apex.
    radius :
        Radius of the circular plan of the cap.
    rise :
        Height of the apex above the plane of the boundary.
    """

    num_rings: int
    num_spokes: int
    radius: float
    rise: float


class SectionSeeds(NamedTuple):
    """
    The candidate starting sections a review compares.

    Attributes
    ----------
    section_class :
        Cross-section class the tube family sits at the limit of.
    seed_diameters :
        Outer diameters the structure is drawn and priced at, one frame each.
    """

    section_class: int
    seed_diameters: tuple[float, ...]


class ViewerConfig(NamedTuple):
    """
    Whether the blocking window opens at all.

    Attributes
    ----------
    enabled :
        Whether the viewer opens after the report; off is a headless run.
    """

    enabled: bool


class SceneConfig(NamedTuple):
    """
    Everything the review run is described by.

    Attributes
    ----------
    structure :
        The cap the generator draws.
    sections :
        The tube family and the seed diameters the drawing is priced at.
    viewer :
        Whether the blocking window opens.
    """

    structure: GridshellSketch
    sections: SectionSeeds
    viewer: ViewerConfig


def parse_config(text: str) -> SceneConfig:
    """
    The gridshell and the seeds a review is described by.

    Parameters
    ----------
    text :
        Text of the file describing the run.

    Returns
    -------
    config :
        The cap, the seed diameters and the viewer switch.

    Raises
    ------
    TypeError
        If the text names a field that does not exist, or omits one that does.
    ValueError
        If no seed diameter is given, so an empty review is refused.

    Notes
    -----
    No container carries a default, so a file missing a field is refused
    rather than quietly completed.
    """
    document = yaml.safe_load(text)

    sections = SectionSeeds(**document["sections"])
    seeds = tuple(float(diameter) for diameter in sections.seed_diameters)
    sections = sections._replace(seed_diameters=seeds)

    config = SceneConfig(
        structure=GridshellSketch(**document["structure"]),
        sections=sections,
        viewer=ViewerConfig(**document["viewer"]),
    )

    if not config.sections.seed_diameters:
        raise ValueError("seed_diameters must name at least one diameter")

    return config


def report_counts(
    report: Report,
    sketch: GridshellSketch,
    structure: Structure,
) -> None:
    """
    The container sizes every downstream stage inherits from the generator.

    Parameters
    ----------
    report :
        The report the entries are written to.
    sketch :
        The cap the generator was asked to draw.
    structure :
        The generator's output.
    """
    num_polar = sketch.num_rings * sketch.num_spokes
    radius_sphere = (sketch.radius**2 + sketch.rise**2) / (2.0 * sketch.rise)

    entries = [
        ("nodes", f"{structure.num_nodes}"),
        ("members", f"{structure.num_edges}"),
        ("members, radial", f"{num_polar}"),
        ("members, hoop", f"{num_polar - sketch.num_spokes}"),
        ("supports", f"{sketch.num_spokes}"),
        ("plan radius [mm]", f"{sketch.radius:.1f}"),
        ("rise [mm]", f"{sketch.rise:.1f}"),
        ("sphere radius [mm]", f"{radius_sphere:.1f}"),
    ]
    report.write_entries(entries)


def report_rings(
    report: Report,
    sketch: GridshellSketch,
    structure: Structure,
) -> None:
    """
    The cap's shape read ring by ring, off the generated nodes themselves.

    Parameters
    ----------
    report :
        The report the table is written to.
    sketch :
        The cap the generator was asked to draw.
    structure :
        The generator's output, whose geometry the rows are measured from.

    Notes
    -----
    Every number is read off the built structure rather than recomputed from
    the cap formula, so a generator bug shows up here instead of hiding behind
    arithmetic repeated from its source.
    """
    grid = (sketch.num_rings, sketch.num_spokes)
    hooped = (sketch.num_rings - 1, sketch.num_spokes)
    ring_nodes = np.asarray(structure.nodes)[1:].reshape(*grid, 3)
    lengths = np.asarray(member_lengths(structure.nodes, structure.edges))
    radial = lengths[: grid[0] * grid[1]].reshape(grid)
    hoop = lengths[grid[0] * grid[1] :].reshape(hooped)

    plan_radii = np.linalg.norm(ring_nodes[:, 0, :2], axis=1)
    heights = ring_nodes[:, 0, 2]

    columns = (
        ReportColumn("ring", "d"),
        ReportColumn("plan radius [mm]", ".1f"),
        ReportColumn("height [mm]", ".1f"),
        ReportColumn("radial L [mm]", ".1f"),
        ReportColumn("hoop L [mm]", ".1f"),
    )
    rows = [
        (
            ring + 1,
            plan_radii[ring],
            heights[ring],
            radial[ring, 0],
            hoop[ring, 0] if ring < hooped[0] else "pinned",
        )
        for ring in range(sketch.num_rings)
    ]
    report.write_table(columns, rows)


def report_seeds(
    report: Report,
    config: SceneConfig,
    structure: Structure,
    family: TubeFamily,
) -> None:
    """
    What each candidate seed diameter costs before any check trims it.

    Parameters
    ----------
    report :
        The report the table is written to.
    config :
        The run description naming the seed diameters.
    structure :
        The generator's output, supplying the lengths the mass integrates.
    family :
        The tube family pairing every diameter with its wall.

    Notes
    -----
    The mass is the shell at one uniform diameter, `ρ Σ A L` over the drawn
    geometry — a starting point's price, not a design's.
    """
    lengths = member_lengths(structure.nodes, structure.edges)

    columns = (
        ReportColumn("seed d [mm]", ".1f"),
        ReportColumn("wall t [mm]", ".2f"),
        ReportColumn("d/t", ".1f"),
        ReportColumn("mass [t]", ".6f"),
    )
    rows = []
    for diameter in config.sections.seed_diameters:
        sections = family(jnp.asarray(diameter))
        per_length = sections.material.density * sections.area
        mass = float(jnp.sum(per_length * lengths))
        row = (diameter, float(sections.thickness), float(family.ratio), mass)
        rows.append(row)
    report.write_table(columns, rows)


def view_seeds(
    config: SceneConfig,
    structure: Structure,
    family: TubeFamily,
) -> None:
    """
    Open the drawn structure once per seed diameter, tubes at true size.

    Parameters
    ----------
    config :
        The run description naming the seed diameters.
    structure :
        The generator's output, drawn at its own starting geometry.
    family :
        The tube family pairing every diameter with its wall.

    Notes
    -----
    Every registration is named apart, because a viewer's `add` replaces a
    same-named one. The frames sit in the same place on purpose: a seed is
    compared by switching frames off from the panel, not by looking sideways.

    Blocks until the window closes.
    """
    viewer = vix.Viewer(show_reactions=False)

    for diameter in config.sections.seed_diameters:
        sections = family(jnp.asarray(diameter))
        frame = frame_model(structure, structure.nodes, sections)
        viewer.add(frame, name=f"seed {diameter:g} mm")

    viewer.show()


def main(path: Path) -> None:
    """
    Build the gridshell, write the report, and open the viewer last.

    Parameters
    ----------
    path :
        The YAML file describing the run.
    """
    config = parse_config(path.read_text())
    sketch = config.structure

    structure = build_gridshell_3d(
        sketch.num_rings,
        sketch.num_spokes,
        sketch.radius,
        sketch.rise,
    )
    family = build_section_family(Steel355(), config.sections.section_class)

    report = Report()
    report.write_banner("Gridshell generator — shape, height, seed diameters")

    report.write_heading("Counts and cap")
    report_counts(report, sketch, structure)

    report.write_heading("Rings, apex down to the boundary")
    report_rings(report, sketch, structure)

    report.write_heading("Seed diameters at one uniform size")
    report_seeds(report, config, structure, family)

    if config.viewer.enabled:
        view_seeds(config, structure, family)


if __name__ == "__main__":
    jnp.set_printoptions(precision=12)
    described = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    main(described or Path(__file__).with_name("gridshell_view.yaml"))
