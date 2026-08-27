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
What the 16x16 gridshell's held-plan design space looks like, before optimizing.

Experiment 23 searches this space and reports one point in it. This draws the
space itself: the drawn cap, and five shapes reached by perturbing the
independent force densities around it and form-finding what comes back. No
analysis, no sizing, no descent — an equilibrium solve is milliseconds, so the
picture is cheap and it says what the optimizer had to choose between.

**The coordinates are named members.** The basis is pivoted rather than
orthonormal, so each of the 23 coordinates is the density of one actual edge
that QR pivoting elected independent, and a perturbation is a statement about
that edge. The two bases span the identical subspace; only the axes differ.

**The plan is held, so height is the whole of what varies.** Every coordinate
lies in the null space of the horizontal balance, so no reachable shape moves a
node in plan. Two panels of the figure differ in nothing but z, which is why
the members are colored by height and why the plan needs no drawing.

**The compression guard is off, and that is a decision.** Experiment 23 holds
the radials in compression along its whole search, because letting them go
stops it designing a shell at all — it flattens the cap and hangs the members,
which the code rewards with no buckling reduction and far less steel. That is
a statement about where a search may travel, not about which shapes exist, and
this draws the shapes. Guarding here refused 99.98% of draws, so the survivors
were whatever squeaked through rather than a fair spread; no draw ever failed
to factorize without it; and the sag floor keeps every shape a dome regardless.
Turn `guarded` on to see the sliver the search moves through instead.

**A draw is still refused three ways**, and each is counted apart: for leaving
the compression cone where that is asked for, for an equilibrium that will not
factorize, and for a shape that leaves the rise ceiling or the sag floor the
descent holds. A funicular hanging below its own supports is a legitimate
equilibrium and an unreachable design.

What it writes:

    report      the cap and every sample, by rise, sag, member length and how
                far its heights travelled from the cap
    figure      six wireframes sharing one set of limits and one height scale

Run with `uv run --group pipeline --group viz python
experiments/25_gridshell_space.py [gridshell_space.yaml]`.
"""

import importlib.util
import sys
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import yaml
from jaxtyping import Float
from normax.searches import FIGURES
from normax.searches import ChordSigns
from normax.searches import DesignProblem
from normax.searches import HeightTruss
from normax.searches import folding_maps
from normax.searches import parse_shell
from normax.searches import prepare_problem

from normax.reporting import Report
from normax.reporting import ReportColumn
from normax.structures import compute_member_lengths
from normax.visualization import ShapeVariation
from normax.visualization import figure_shape_variations

# Where experiment 23 keeps the profile this reads its structure through.
EXPERIMENT = Path(__file__).resolve().parents[2] / "examples" / "gridshell.py"

# Stem the figure is named under.
PREFIX = "25_gridshell_space"


class SpaceConfig(NamedTuple):
    """
    How much of the design space to sample, and how widely.

    Attributes
    ----------
    samples :
        How many perturbed shapes to keep, beside the drawn cap.
    spread :
        Spread of the perturbation, as a fraction of each independent density.
    seed :
        Seed of the draw, so that a rerun redraws the same space.
    attempts :
        Most draws to make before giving up on filling the sample count.
    guarded :
        Whether to refuse a draw that takes a guarded member out of
        compression. On, the sampling shows the space the search moves in;
        off, it shows every held-plan equilibrium the plan admits, hanging
        shapes and mixed ones included.
    """

    samples: int
    spread: float
    seed: int
    attempts: int
    guarded: bool


class RefusalCount(NamedTuple):
    """
    Why the draws that were thrown away were thrown away.

    Attributes
    ----------
    cone :
        Draws that put a guarded member on the wrong side of zero.
    singular :
        Draws whose equilibrium would not factorize, or came back non-finite.
    bounds :
        Draws that form-found to a shape leaving the height limits.
    """

    cone: int
    singular: int
    bounds: int

    @property
    def total(self) -> int:
        """
        How many draws were refused for any reason at all.
        """
        return self.cone + self.singular + self.bounds


class SampledShape(NamedTuple):
    """
    One accepted draw, and the shape it form-found to.

    Attributes
    ----------
    name :
        What to call it in the report and above its panel.
    xi :
        The independent densities the draw landed on.
    xyz :
        Position of every node at equilibrium.
    lengths :
        Length of every member at that geometry.
    """

    name: str
    xi: Float[np.ndarray, "independents"]
    xyz: Float[np.ndarray, "nodes 3"]
    lengths: Float[np.ndarray, "members"]


def limit_named(limit: float | None) -> str:
    """
    A height limit as the report states it, or its absence.

    Parameters
    ----------
    limit :
        The limit, or None where the run holds none.

    Returns
    -------
    named :
        The limit in millimeters, or the word for having none.
    """
    if limit is None:
        return "none"

    return f"{limit:.0f} mm"


def parse_space(text: str) -> SpaceConfig:
    """
    Read the sampling section of a run description.

    Parameters
    ----------
    text :
        The YAML file's contents.

    Returns
    -------
    space :
        How much of the space to sample, and how widely.

    Notes
    -----
    The other sections are the shared parser's, which reads the keys it knows
    and ignores this one.
    """
    document = yaml.safe_load(text)

    return SpaceConfig(**document["space"])


def form_found(
    problem: DesignProblem,
    xi: Float[np.ndarray, "independents"],
) -> tuple[Float[np.ndarray, "nodes 3"], Float[np.ndarray, "members"]] | None:
    """
    The shape a set of independent densities holds, or None if it holds none.

    Parameters
    ----------
    problem :
        The prepared shell, supplying the form finder and the load case.
    xi :
        The independent densities to solve at.

    Returns
    -------
    found :
        The geometry and the member lengths, or None where the equilibrium
        will not factorize or came back non-finite.

    Notes
    -----
    A draw that cannot be solved is not an error to raise but an answer: the
    point is outside the space, which is the thing being measured.
    """
    finder = problem.pipeline.formfinder
    try:
        shape = finder(jnp.asarray(xi), problem.loads.formfinding)
    except (ValueError, FloatingPointError):
        return None

    xyz = np.asarray(shape.xyz)
    lengths = np.asarray(shape.lengths)
    if not (np.all(np.isfinite(xyz)) and np.all(np.isfinite(lengths))):
        return None

    return xyz, lengths


def compressive(
    problem: DesignProblem,
    guard: ChordSigns | None,
    xi: Float[np.ndarray, "independents"],
) -> bool:
    """
    Whether a draw keeps the guarded chords on their own side of zero.

    Parameters
    ----------
    problem :
        The prepared shell, supplying the basis the densities are read from.
    guard :
        The signs the chords must keep, or None to accept every draw.
    xi :
        The independent densities to test.

    Returns
    -------
    signed :
        Whether every guarded chord clears its margin.
    """
    if guard is None:
        return True

    finder = problem.pipeline.formfinder
    q = np.asarray(finder.read_member_densities(jnp.asarray(xi)))
    signed = guard.signs * q[guard.chords]

    return bool(np.all(signed >= guard.margin))


def within_bounds(
    xyz: Float[np.ndarray, "nodes 3"],
    limits: HeightTruss,
) -> bool:
    """
    Whether a shape stays between the ceiling and the floor the descent holds.

    Parameters
    ----------
    xyz :
        Position of every node.
    limits :
        The ceiling and the floor, either of which may be absent.

    Returns
    -------
    held :
        Whether every node lies within both limits.

    Notes
    -----
    The same two bounds the descent carries as inequality rows, applied here
    as an acceptance test. Without them the sampling draws from the held-plan
    compression cone, which is a larger space than the one the search moves
    in: a funicular that hangs below its own supports is a legitimate
    equilibrium and an unreachable design.
    """
    heights = xyz[:, 2]
    if limits.ceiling is not None and float(heights.max()) > limits.ceiling:
        return False
    if limits.floor is not None and float(heights.min()) < limits.floor:
        return False

    return True


def sampled_shapes(
    problem: DesignProblem,
    guard: ChordSigns | None,
    xi_drawn: Float[np.ndarray, "independents"],
    space: SpaceConfig,
    limits: HeightTruss,
) -> tuple[list[SampledShape], RefusalCount]:
    """
    Perturb the independent densities until enough draws land in the space.

    Parameters
    ----------
    problem :
        The prepared shell.
    guard :
        The signs the chords must keep, or None to accept every draw.
    xi_drawn :
        The independent densities of the drawn cap, perturbed around.
    space :
        How many draws to keep, how widely to spread them and from what seed.
    limits :
        The ceiling and the floor a shape must stay between.

    Returns
    -------
    kept :
        The accepted draws, in the order they were drawn.
    refused :
        How many draws were discarded, and for which of the three reasons.

    Notes
    -----
    Multiplicative rather than additive, so a coordinate is perturbed in
    proportion to the force it already carries: an edge barely stressed in the
    funicular is not handed the same absolute swing as the crown's meridian.
    """
    draw = np.random.default_rng(space.seed)
    kept = []
    cone = 0
    singular = 0
    outside = 0

    for _ in range(space.attempts):
        if len(kept) == space.samples:
            break
        noise = draw.standard_normal(xi_drawn.size)
        xi = xi_drawn * (1.0 + space.spread * noise)
        if not compressive(problem, guard, xi):
            cone += 1
            continue
        found = form_found(problem, xi)
        if found is None:
            singular += 1
            continue
        xyz, lengths = found
        if not within_bounds(xyz, limits):
            outside += 1
            continue
        kept.append(SampledShape(f"sample {len(kept) + 1}", xi, xyz, lengths))

    return kept, RefusalCount(cone, singular, outside)


def report_space(
    report: Report,
    problem: DesignProblem,
    cap: SampledShape,
    shapes: list[SampledShape],
) -> None:
    """
    The cap and every sample, by the few numbers a shape is read by.

    Parameters
    ----------
    report :
        Where the table is written.
    problem :
        The prepared shell, supplying the basis the densities are read from.
    cap :
        The drawn cap, the first row and the reference the rest are measured
        against.
    shapes :
        The accepted draws.
    """
    finder = problem.pipeline.formfinder
    columns = (
        ReportColumn("shape", align="<"),
        ReportColumn("rise [mm]", ".0f"),
        ReportColumn("sag [mm]", ".0f"),
        ReportColumn("L min [mm]", ".0f"),
        ReportColumn("L max [mm]", ".0f"),
        ReportColumn("q min", ".4f"),
        ReportColumn("q max", ".4f"),
        ReportColumn("dz rms [mm]", ".0f"),
        ReportColumn("dz max [mm]", ".0f"),
    )

    heights_cap = cap.xyz[:, 2]
    rows = []
    for shape in [cap, *shapes]:
        q = np.asarray(finder.read_member_densities(jnp.asarray(shape.xi)))
        drift = shape.xyz[:, 2] - heights_cap
        rows.append(
            (
                shape.name,
                float(shape.xyz[:, 2].max()),
                float(shape.xyz[:, 2].min()),
                float(shape.lengths.min()),
                float(shape.lengths.max()),
                float(q.min()),
                float(q.max()),
                float(np.sqrt(np.mean(drift**2))),
                float(np.abs(drift).max()),
            )
        )

    report.write_table(columns, rows)


def write_figure(
    problem: DesignProblem,
    cap: SampledShape,
    shapes: list[SampledShape],
) -> Path:
    """
    Draw the cap and every sample as wireframes sharing one scale.

    Parameters
    ----------
    problem :
        The prepared shell, supplying the connectivity.
    cap :
        The drawn cap, drawn first and named as itself.
    shapes :
        The accepted draws.

    Returns
    -------
    written :
        The file written.
    """
    drawn = [ShapeVariation("drawn cap — the funicular", cap.xyz)]
    for shape in shapes:
        rise = float(shape.xyz[:, 2].max())
        drawn.append(ShapeVariation(f"{shape.name} — rise {rise:.0f} mm", shape.xyz))

    figure = figure_shape_variations(problem.structure.edges, drawn)
    written = FIGURES / f"{PREFIX}.png"
    figure.savefig(written, dpi=200, bbox_inches="tight")

    return written


def main(path: Path) -> None:
    """
    Prepare the shell, sample around its funicular, report and draw.

    Parameters
    ----------
    path :
        The YAML file describing the run.
    """
    spec = importlib.util.spec_from_file_location("experiment", EXPERIMENT)
    experiment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiment)
    profile = experiment.GRIDSHELL_PROFILE

    report = Report()
    report.write_banner("The 16x16 gridshell's design space, sampled")

    described = path.read_text()
    config = parse_shell(described)
    space = parse_space(described)
    if config.subspace.basis != "pivoted":
        raise ValueError(
            f"this experiment reads its coordinates as member densities, "
            f"which asks for the pivoted basis, got {config.subspace.basis}"
        )

    structure = profile.build_structure(config)
    plan = profile.build_loads(structure, config)
    folding_by = folding_maps(profile, config, structure)
    problem = prepare_problem(structure, config, plan, folding_by)

    start = profile.signed_start(problem, config)
    guard = profile.sign_guard(config, start) if space.guarded else None
    limits = profile.height_limits(config)
    finder = problem.pipeline.formfinder
    xi_drawn = np.asarray(start.xi)

    lengths_cap = compute_member_lengths(jnp.asarray(start.lens), structure.edges)
    cap = SampledShape(
        "drawn cap",
        xi_drawn,
        np.asarray(start.lens),
        np.asarray(lengths_cap),
    )

    folded = "mirror imposed" if config.subspace.symmetric else "no symmetry imposed"
    entries = (
        ("structure", f"{structure.num_nodes} nodes, {structure.num_edges} members"),
        (
            "independent densities",
            f"{int(finder.basis.shape[1])}, pivoted, {folded}",
        ),
        ("plan", "held — every reachable shape moves in height alone"),
        ("lens fit gap / total load", f"{start.gap / plan.total:.2e}"),
        ("perturbation", f"{space.spread:.0%} of each density, seed {space.seed}"),
        (
            "compression guard",
            "none"
            if guard is None
            else f"{guard.chords.size} chords, margin {guard.margin:.4f} N/mm",
        ),
        ("rise ceiling", limit_named(limits.ceiling)),
        ("sag floor", limit_named(limits.floor)),
    )
    report.write_heading("The space, and where it is sampled from")
    report.write_entries(entries)

    shapes, refused = sampled_shapes(problem, guard, xi_drawn, space, limits)
    drawn_total = len(shapes) + refused.total
    report.write_heading(
        f"{len(shapes)} shapes kept of {drawn_total} drawn, "
        f"{len(shapes) / drawn_total:.3%} accepted"
    )
    causes = (
        ("refused, left the compression cone", f"{refused.cone}"),
        ("refused, would not factorize", f"{refused.singular}"),
        ("refused, outside the height limits", f"{refused.bounds}"),
    )
    report.write_entries(causes)
    if len(shapes) < space.samples:
        report.write_note(
            f"Only {len(shapes)} of {space.samples} draws landed in the space "
            f"within {space.attempts} attempts. Narrow the spread, or read the "
            f"refusal counts as the answer."
        )
    report_space(report, problem, cap, shapes)

    written = write_figure(problem, cap, shapes)
    report.write_line()
    report.write_line(f"figure written to {written}")


if __name__ == "__main__":
    jnp.set_printoptions(precision=12)
    described = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    main(described or Path(__file__).with_name("gridshell_space.yaml"))
