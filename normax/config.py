# SPDX-License-Identifier: Apache-2.0
"""
What a run is configured by, read from a file.

Every section a design shares — the form finding, the load cases, the two
backends, the constraints, the descent's budget — is a container here. What the
structure is varies by structure, so an example hands `read_run_config` the
description type from `normax.structures` its `structure` section is read into.

Nothing here is built, only read. A start arrives as the fields of the
initializer it belongs to, and the example names that class and constructs it.

A file describes a whole run, so an example takes one on its command line and
little else. The exception is the shape parametrization: racing the three
against each other is the point of the comparison, and asking for the next one
should not mean keeping three near-identical files, so a flag overrides the
word the file names.
"""

import argparse
from pathlib import Path
from typing import Generic
from typing import NamedTuple
from typing import TypeVar

import yaml

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
    diameter_start :
        Fields of the diameter initializer the example builds, by keyword:
        `diameter` for one outer diameter in every member, before the check
        sizes them.
    backend :
        Which solver fills the analysis slot, `opensees` or `pynite`. Both
        cross a Tesseract boundary to a host solver — the first planar, the
        second a space frame.
    """

    diameter_start: dict[str, float]
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
    shape_parametrization :
        Which block fills the pipeline's first slot, and so what the design
        variables mean — `fdm` for force densities through an equilibrium
        solve, `heights` for the free nodes' height written down, `fixed` for
        the drawn geometry, which moves the diameters alone. The two written
        parametrizations read none of the fields below, and a run reports which
        of them it left unread.
    basis :
        Convention of the held-plan basis the densities move in — `pivoted`
        for the independent members' own densities, `svd` for projections on
        an orthonormal basis — or None to move every density freely.
    mirror :
        Axis the mirror plane stands normal to, folding the densities by that
        symmetry, or None for no symmetry.
    fold_heights :
        Whether the free nodes' heights are folded by that mirror, one height
        per mirrored pair. Read by `heights` alone, and refused where the
        section names no mirror to fold by. Only the mirror folds a height,
        never a rotation: a mirror that carries the load cases onto each other
        leaves the answer symmetric rather than constrained, while a rotation
        no load case respects would hold a shape the loads do not.
    density_start :
        Fields of the density initializer the example builds, by keyword:
        `force_density` for a uniform start, `sag`, `rise` and `held_plan` for
        a lens sketch, or None where the fit reads none — a drawn one, or a
        parametrization that never fits a density at all. The one field here
        carrying a default, and it defaults to None rather than to an empty
        mapping: a NamedTuple evaluates its defaults once, so a mutable one
        would be a single dict shared by every config in the process.
    height_start :
        Fields of the written-heights start, by keyword: `rise` for the crown a
        generated parabolic lift over the drawn plan reaches. None leaves that
        route starting from the drawn heights themselves, which is right when
        the drawing is the shape somebody meant and wrong when it is flat — a
        flat geometry is a stationary point of both the mass and the
        utilization, and no descent leaves one. Read by `heights` alone.
    """

    shape_parametrization: str
    basis: str | None
    mirror: str | None
    fold_heights: bool
    density_start: dict[str, float | bool] | None = None
    height_start: dict[str, float] | None = None


class BoundsConfig(NamedTuple):
    """
    The box the force densities may move in, where they are the coefficients.

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
        the coefficients.
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
    animate :
        Whether the run writes an animation of its descent, which needs the
        finer record `optimization.trace_iterations` keeps and a pipeline that
        carries a check.
    """

    verbose: bool
    export: bool
    animate: bool


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


def check_start_fields(
    described: dict[str, float | bool] | None,
    wanted: tuple[str, ...],
) -> None:
    """
    Refuse a start whose fields are not the ones its initializer reads.

    Parameters
    ----------
    described :
        What a file gave the start, which is the initializer's own fields, or
        None where it named no start.
    wanted :
        Names the initializer reads, which the file must name exactly.

    Raises
    ------
    ValueError
        If the file names anything else, or omits one of them.

    Notes
    -----
    An initializer is its own schema, so a start is checked where it is read
    into one rather than at the parse, and the message names both sides.

    A file naming no start reads the same as one naming an empty start, which
    is what an initializer reading no fields wants. Normalizing here rather
    than at each call site is what keeps a `null` in a file from surfacing as
    `set(None)` inside some initializer's constructor, a long way from the line
    that caused it.
    """
    named_fields = described or {}
    if set(named_fields) == set(wanted):
        return

    named = ", ".join(sorted(wanted)) or "nothing"
    raise ValueError(f"a start must name {named}, got {sorted(named_fields)}")


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
    ValueError
        If the heights are to be folded by a mirror the file does not name.

    Notes
    -----
    No container carries a default but one — `form_finding.density_start`, which
    a fit reading no fields may leave out — so a file missing any other field is
    refused rather than quietly completed. Every budget is cast on the way in:
    YAML reads an exponent without a signed power as a string. A start's fields
    are carried across as they were written, and are checked by the initializer
    the example builds from them, not here.
    """
    document = yaml.safe_load(text)

    described = document["form_finding"]
    if described.get("fold_heights") and described.get("mirror") is None:
        raise ValueError(
            "form_finding.fold_heights asks to fold the heights by a mirror, "
            "but form_finding.mirror names none"
        )

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
    flags = ("trace_iterations",)
    named = document["optimization"]
    budget = {key: int(value) for key, value in named.items() if key in counts}
    switches = {key: bool(value) for key, value in named.items() if key in flags}
    scales = {
        key: float(value)
        for key, value in named.items()
        if key not in counts and key not in flags
    }
    budget.update(switches)
    budget.update(scales)
    load_cases = tuple(LoadCaseConfig(**entry) for entry in document["load_cases"])

    config = RunConfig(
        structure=structure_type(**document["structure"]),
        form_finding=FormFindingConfig(**document["form_finding"]),
        load_cases=load_cases,
        analysis=AnalysisConfig(**document["analysis"]),
        sizing=SizingConfig(**document["sizing"]),
        constraints=constraints,
        optimization=OptimizationBudget(**budget),
        output=OutputConfig(**document["output"]),
    )

    return config


# What an example prints for `--help`. The module docstring would be this
# module's, not the example's, and argparse has no way to reach the caller.
RUN_USAGE = "Design a structure a file describes, and report what it bought."


class RunArguments(NamedTuple):
    """
    What an example was asked for on its command line.

    Attributes
    ----------
    config_path :
        File describing the run.
    shape_parametrization :
        Parametrization overriding the one the file names, or None to keep it.
    """

    config_path: Path
    shape_parametrization: str | None


def read_run_arguments(
    argv: list[str],
    default_path: Path,
) -> RunArguments:
    """
    The command line an example was invoked with.

    Parameters
    ----------
    argv :
        Arguments after the script name, which is `sys.argv[1:]`.
    default_path :
        File to read where the command line names none.

    Returns
    -------
    arguments :
        The file to read, and any parametrization overriding it.

    Notes
    -----
    The parametrization is carried rather than checked: `build_form_finder` is
    the one place a word is refused, so a typo is reported once and in the same
    terms wherever it came from.
    """
    parser = argparse.ArgumentParser(description=RUN_USAGE)
    parser.add_argument(
        "config_path",
        nargs="?",
        type=Path,
        default=default_path,
        help="file describing the run; defaults to the one beside the example",
    )
    parser.add_argument(
        "--shape-parametrization",
        default=None,
        help="fdm, heights or fixed, overriding the word the file names",
    )
    parsed = parser.parse_args(argv)

    return RunArguments(parsed.config_path, parsed.shape_parametrization)


def read_run_config(
    arguments: RunArguments,
    structure_type: type[StructureT],
) -> RunConfig[StructureT]:
    """
    The run config a command line asks for, overrides applied.

    Parameters
    ----------
    arguments :
        The file to read, and any parametrization overriding it.
    structure_type :
        Description the `structure` section is read into.

    Returns
    -------
    config :
        The run config, with the command line's parametrization where it named
        one.
    """
    config = parse_config(arguments.config_path.read_text(), structure_type)
    named = arguments.shape_parametrization
    if named is None:
        return config

    form_finding = config.form_finding._replace(shape_parametrization=named)

    return config._replace(form_finding=form_finding)
