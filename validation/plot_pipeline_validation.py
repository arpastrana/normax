# SPDX-License-Identifier: Apache-2.0
"""Generate the three numerical validation figures and their provenance.

Nothing here edits documentation.  Each figure is written as PNG, SVG, and
PDF, while JSON supplies a human-readable provenance record and NPZ preserves
the plotted arrays at full precision.

Run with ``uv run python validation/plot_pipeline_validation.py``.  Pass
``--project-pynite-fd`` for a fast development run whose large-frame finite
difference cost is an explicit projection rather than a measured sweep, or
``--render-only`` to redraw an existing numerical record without measuring it.
"""

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import UTC
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from blueprint_adjoint import TARGET
from blueprint_adjoint import TOLERANCE_DERIVATIVE
from blueprint_adjoint import TOLERANCE_NUMERIC
from blueprint_adjoint import TOLERANCE_PARITY
from blueprint_adjoint import TOLERANCE_UNITY
from blueprint_adjoint import measure_blueprint
from load_case_envelope import TOLERANCE_DERIVATIVE as ENVELOPE_TOLERANCE
from load_case_envelope import measure_envelope
from pipeline_gradients import TOLERANCE
from pipeline_gradients import measure_pipeline
from pynite_adjoint import TOLERANCE_DIFFERENCE
from pynite_adjoint import TOLERANCE_GRADIENT
from pynite_adjoint import canopy_sample
from pynite_adjoint import measure_cost
from pynite_adjoint import measure_gradient
from pynite_adjoint import shell_sample

from normax.visualization import draw_code_validation
from normax.visualization import draw_pipeline_validation
from normax.visualization import draw_pynite_validation

REPO = Path(__file__).resolve().parent.parent
FIGURES = REPO / "figures"
RESULTS = Path(__file__).resolve().parent / "results"
FORMATS = ("png", "svg", "pdf")
DPI = 240


def read_arguments() -> argparse.Namespace:
    """Read whether to measure fully, project one cost, or only redraw."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--project-pynite-fd",
        action="store_true",
        help="project the 1267-parameter FD cost from its measured primal",
    )
    mode.add_argument(
        "--render-only",
        action="store_true",
        help="redraw the existing JSON record without rerunning measurements",
    )

    return parser.parse_args()


def package_versions() -> dict[str, str]:
    """Versions of the numerical packages that materially affect the record."""
    distributions = (
        "normax",
        "jax",
        "matplotlib",
        "numpy",
        "openseespy",
        "pynitefea",
        "tesseract-core",
        "tesseract-jax",
        "blue-prints",
    )
    versions = {}
    for distribution in distributions:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not installed"

    return versions


def git_provenance() -> dict[str, str | bool]:
    """Read the immutable commit identifier and whether this worktree differs."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    # The run's own outputs cannot dirty their own provenance; code still can.
    status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            ".",
            ":(exclude)figures",
            ":(exclude)validation/results",
        ],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    return {"commit": commit, "worktree_dirty": bool(status.strip())}


def export_figure(figure, stem: str) -> list[str]:
    """Write one figure in the raster and vector formats used by the project."""
    written = []
    for suffix in FORMATS:
        target = FIGURES / f"{stem}.{suffix}"
        figure.savefig(target, dpi=DPI, bbox_inches="tight", facecolor="white")
        written.append(str(target.relative_to(REPO)))
    plt.close(figure)

    return written


def measured_checks(measured) -> tuple[list[str], list[float], list[float]]:
    """Select the code checks that make route, difference, and branch immediate."""
    envelope = measured[1]
    blueprint = measured[0]
    reversal = envelope.checks[-1]
    labels = [
        "size route parity",
        "gradient route parity",
        "gradient vs differences",
        "utilization at root",
        "force-reversal branch",
    ]
    errors = [
        blueprint.worst_parity,
        blueprint.worst_gradient,
        blueprint.worst_numeric,
        blueprint.worst_unity,
        reversal.worst,
    ]
    tolerances = [
        TOLERANCE_PARITY,
        TOLERANCE_DERIVATIVE,
        TOLERANCE_NUMERIC,
        TOLERANCE_UNITY,
        reversal.tolerance,
    ]

    return labels, errors, tolerances


def json_record(pipeline, blueprint, envelope, gradient, cost, figures) -> dict:
    """Build a JSON-safe record from the same raw values the plots consume."""
    annealed = envelope.annealed
    gaps = gradient.gaps

    return {
        "schema": "normax.validation.v1",
        "generated_utc": datetime.now(UTC).isoformat(),
        "git": git_provenance(),
        "runtime": {
            "python": platform.python_version(),
            "packages": package_versions(),
            "hardware": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "processor": platform.processor() or "unreported",
                "logical_cpus": os.cpu_count(),
            },
        },
        "pipeline": {
            "description": (
                "mean squared utilization; FDM -> OpenSees -> Blueprints; "
                "uniform, asymmetric, and midspan point load cases"
            ),
            "parameter_names": list(pipeline.parameter_names),
            "parameter_kinds": list(pipeline.parameter_kinds),
            "reverse": list(pipeline.reverse),
            "central_difference": list(pipeline.central),
            "scaled_errors": {
                "force_density": pipeline.density_error,
                "diameter": pipeline.diameter_error,
                "combined": pipeline.combined_error,
            },
            "tolerance": TOLERANCE,
            "timings_seconds": {
                "forward": pipeline.forward_seconds,
                "reverse": pipeline.reverse_seconds,
                "central_difference": pipeline.central_seconds,
            },
        },
        "code": {
            "force_derivative_errors": [found.worst for _, found in blueprint.by_force],
            "moment_derivative_errors": [
                found.worst for _, found in blueprint.by_moment
            ],
            "targets": {
                "four_way_derivative": TARGET,
                "size_route": TOLERANCE_PARITY,
                "gradient_route": TOLERANCE_DERIVATIVE,
                "gradient_difference": TOLERANCE_NUMERIC,
                "utilization_root": TOLERANCE_UNITY,
            },
            "route_errors": {
                "size": blueprint.worst_parity,
                "gradient": blueprint.worst_gradient,
                "gradient_difference": blueprint.worst_numeric,
                "utilization_root": blueprint.worst_unity,
            },
            "envelope": [
                {
                    "beta": step.beta,
                    "mass_kg": step.mass,
                    "relative_excess": step.excess,
                    "relative_bound": step.bound,
                    "gradient_finite": step.finite,
                }
                for step in annealed
            ],
            "force_reversal": [row._asdict() for row in envelope.reversal],
        },
        "pynite": {
            "difference_steps": list(gradient.steps),
            "node_errors": list(gradient.node_errors),
            "diameter_errors": list(gradient.diameter_errors),
            "route_errors": gaps._asdict(),
            "route_tolerances": {
                "by_node": TOLERANCE_DIFFERENCE,
                "by_member": TOLERANCE_DIFFERENCE,
                "crossed": TOLERANCE_DIFFERENCE,
                "boundary": TOLERANCE_GRADIENT,
                "frozen": TOLERANCE_GRADIENT,
            },
            "scale": {
                "nodes": cost.nodes,
                "members": cost.members,
                "parameters": cost.parameters,
            },
            "timings_seconds": {
                "forward": cost.forward_seconds,
                "three_load_cases": cost.load_cases_seconds,
                "adjoint": cost.adjoint_seconds,
                "finite_difference": cost.finite_difference_seconds,
                "finite_difference_measured": cost.finite_difference_measured,
            },
        },
        "figures": figures,
    }


def save_arrays(pipeline, blueprint, envelope, gradient, cost) -> None:
    """Preserve every plotted numerical array without JSON conversion."""
    np.savez(
        RESULTS / "validation_measurements.npz",
        pipeline_reverse=np.asarray(pipeline.reverse),
        pipeline_central=np.asarray(pipeline.central),
        pipeline_errors=np.asarray(
            [pipeline.density_error, pipeline.diameter_error, pipeline.combined_error]
        ),
        pipeline_timings=np.asarray(
            [
                pipeline.forward_seconds,
                pipeline.reverse_seconds,
                pipeline.central_seconds,
            ]
        ),
        code_force_errors=np.asarray([found.worst for _, found in blueprint.by_force]),
        code_moment_errors=np.asarray(
            [found.worst for _, found in blueprint.by_moment]
        ),
        envelope_beta=np.asarray([step.beta for step in envelope.annealed]),
        envelope_excess=np.asarray([step.excess for step in envelope.annealed]),
        envelope_bound=np.asarray([step.bound for step in envelope.annealed]),
        pynite_steps=np.asarray(gradient.steps),
        pynite_node_errors=np.asarray(gradient.node_errors),
        pynite_diameter_errors=np.asarray(gradient.diameter_errors),
        pynite_route_errors=np.asarray(gradient.gaps),
        pynite_timings=np.asarray(
            [
                cost.forward_seconds,
                cost.load_cases_seconds,
                cost.adjoint_seconds,
                cost.finite_difference_seconds,
            ]
        ),
    )


def redraw_existing() -> None:
    """Redraw the recorded arrays without changing their numerical provenance."""
    source = RESULTS / "validation_provenance.json"
    record = json.loads(source.read_text(encoding="utf-8"))
    pipeline = record["pipeline"]
    timing = pipeline["timings_seconds"]
    errors = pipeline["scaled_errors"]
    figures = {}

    figure = draw_pipeline_validation(
        pipeline["reverse"],
        pipeline["central_difference"],
        pipeline["parameter_kinds"],
        ("force-density path", "diameter path", "complete vector"),
        (errors["force_density"], errors["diameter"], errors["combined"]),
        (pipeline["tolerance"],) * 3,
        ("forward", "reverse", "central FD"),
        (timing["forward"], timing["reverse"], timing["central_difference"]),
    )
    figures["pipeline"] = export_figure(figure, "validation_pipeline")

    code = record["code"]
    annealed = code["envelope"]
    reversal = code["force_reversal"]
    reversal_scale = max(abs(row["central"]) for row in reversal)
    reversal_error = max(
        abs(row["adjoint"] - row["central"]) / reversal_scale for row in reversal
    )
    route = code["route_errors"]
    targets = code["targets"]
    check_labels = (
        "size route parity",
        "gradient route parity",
        "gradient vs differences",
        "utilization at root",
        "force-reversal branch",
    )
    check_errors = (
        route["size"],
        route["gradient"],
        route["gradient_difference"],
        route["utilization_root"],
        reversal_error,
    )
    check_tolerances = (
        targets["size_route"],
        targets["gradient_route"],
        targets["gradient_difference"],
        targets["utilization_root"],
        ENVELOPE_TOLERANCE,
    )
    figure = draw_code_validation(
        code["force_derivative_errors"],
        code["moment_derivative_errors"],
        tuple(str(index + 1) for index in range(len(code["force_derivative_errors"]))),
        [step["beta"] for step in annealed],
        [step["relative_excess"] for step in annealed],
        [step["relative_bound"] for step in annealed],
        check_labels,
        check_errors,
        check_tolerances,
    )
    figures["code"] = export_figure(figure, "validation_code")

    pynite = record["pynite"]
    route = pynite["route_errors"]
    bounds = pynite["route_tolerances"]
    timing = pynite["timings_seconds"]
    figure = draw_pynite_validation(
        pynite["difference_steps"],
        pynite["node_errors"],
        pynite["diameter_errors"],
        (
            "coordinates vs FD",
            "diameters vs FD",
            "crossed vs FD",
            "boundary parity",
            "frozen reference norms",
        ),
        (
            route["by_node"],
            route["by_member"],
            route["crossed"],
            route["boundary"],
            route["frozen"],
        ),
        (
            bounds["by_node"],
            bounds["by_member"],
            bounds["crossed"],
            bounds["boundary"],
            bounds["frozen"],
        ),
        ("forward", "3 cases", "adjoint", "central FD"),
        (
            timing["forward"],
            timing["three_load_cases"],
            timing["adjoint"],
            timing["finite_difference"],
        ),
        timing["finite_difference_measured"],
    )
    figures["pynite"] = export_figure(figure, "validation_pynite")

    print(f"redrew {sum(map(len, figures.values()))} figure files from {source}")


def main() -> None:
    """Measure, draw, export, and preserve one internally consistent record."""
    arguments = read_arguments()
    FIGURES.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if arguments.render_only:
        redraw_existing()
        return

    pipeline = measure_pipeline()
    blueprint = measure_blueprint()
    envelope = measure_envelope()
    gradient = measure_gradient(canopy_sample())
    cost = measure_cost(
        shell_sample(), run_finite_difference=not arguments.project_pynite_fd
    )

    figures = {}
    pipeline_figure = draw_pipeline_validation(
        pipeline.reverse,
        pipeline.central,
        pipeline.parameter_kinds,
        ("force-density path", "diameter path", "complete vector"),
        (pipeline.density_error, pipeline.diameter_error, pipeline.combined_error),
        (TOLERANCE, TOLERANCE, TOLERANCE),
        ("forward", "reverse", "central FD"),
        (
            pipeline.forward_seconds,
            pipeline.reverse_seconds,
            pipeline.central_seconds,
        ),
    )
    figures["pipeline"] = export_figure(pipeline_figure, "validation_pipeline")

    check_labels, check_errors, check_tolerances = measured_checks(
        (blueprint, envelope)
    )
    code_figure = draw_code_validation(
        [found.worst for _, found in blueprint.by_force],
        [found.worst for _, found in blueprint.by_moment],
        tuple(str(index + 1) for index in range(len(blueprint.by_force))),
        [step.beta for step in envelope.annealed],
        [step.excess for step in envelope.annealed],
        [step.bound for step in envelope.annealed],
        check_labels,
        check_errors,
        check_tolerances,
    )
    figures["code"] = export_figure(code_figure, "validation_code")

    gaps = gradient.gaps
    route_labels = (
        "coordinates vs FD",
        "diameters vs FD",
        "crossed vs FD",
        "boundary parity",
        "frozen reference norms",
    )
    route_errors = tuple(gaps)
    route_tolerances = (
        TOLERANCE_DIFFERENCE,
        TOLERANCE_DIFFERENCE,
        TOLERANCE_DIFFERENCE,
        TOLERANCE_GRADIENT,
        TOLERANCE_GRADIENT,
    )
    pynite_figure = draw_pynite_validation(
        gradient.steps,
        gradient.node_errors,
        gradient.diameter_errors,
        route_labels,
        route_errors,
        route_tolerances,
        ("forward", "3 cases", "adjoint", "central FD"),
        (
            cost.forward_seconds,
            cost.load_cases_seconds,
            cost.adjoint_seconds,
            cost.finite_difference_seconds,
        ),
        cost.finite_difference_measured,
    )
    figures["pynite"] = export_figure(pynite_figure, "validation_pynite")

    record = json_record(pipeline, blueprint, envelope, gradient, cost, figures)
    target = RESULTS / "validation_provenance.json"
    target.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    save_arrays(pipeline, blueprint, envelope, gradient, cost)

    print(f"wrote {sum(map(len, figures.values()))} figure files")
    print(f"wrote {target.relative_to(REPO)}")
    print(f"wrote {(RESULTS / 'validation_measurements.npz').relative_to(REPO)}")


if __name__ == "__main__":
    sys.exit(main())
