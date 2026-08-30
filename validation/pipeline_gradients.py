# SPDX-License-Identifier: Apache-2.0
"""Measure the crossed pipeline against central differences, end to end.

The scalar below is a validation contraction, not an optimization objective:
it sums squared utilization over three load cases so every stage contributes
to one reverse sweep.  Its gradient therefore crosses form finding, OpenSees
analysis, and the Blueprints code check before it is compared with independent
central differences of the complete forward pass.

Run with ``uv run python validation/pipeline_gradients.py``.
"""

import time
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from jaxtyping import Float

from normax.design import DesignParameters
from normax.design import StructuralDesignPipeline
from normax.form_finding import FdmFormFinder
from normax.loads import assemble_load_cases
from normax.loads import create_load_deck_point
from normax.loads import create_load_half_span
from normax.loads import create_load_uniform
from normax.materials import Steel355
from normax.reporting import Report
from normax.reporting import ReportColumn
from normax.reporting import ToleranceCheck
from normax.reporting import verify_checks
from normax.sections import build_section_catalog
from normax.structures import build_arch_2d
from normax.tesseract import TesseractAnalyzer
from normax.tesseract import TesseractSizer

SPAN = 10_000.0
RISE = 3_000.0
NUM_EDGES = 10
TOTAL_LOAD = 180_000.0
HALF_LOAD = 90_000.0
POINT_LOAD = 90_000.0
DIAMETER = 100.0
STEP = 1e-4
TOLERANCE = 5e-6


class PipelineMeasurement(NamedTuple):
    """Raw derivatives, errors, and wall times of the crossed pipeline."""

    parameter_names: tuple[str, ...]
    parameter_kinds: tuple[str, ...]
    reverse: tuple[float, ...]
    central: tuple[float, ...]
    density_error: float
    diameter_error: float
    combined_error: float
    forward_seconds: float
    reverse_seconds: float
    central_seconds: float


def relative(reference: np.ndarray, compared: np.ndarray) -> float:
    """Largest absolute gap scaled by the largest reference component."""
    scale = max(float(np.max(np.abs(reference))), np.finfo(float).tiny)

    return float(np.max(np.abs(reference - compared))) / scale


def build_problem() -> tuple[StructuralDesignPipeline, DesignParameters, object]:
    """Build the shared arch and its three checked load cases."""
    structure = build_arch_2d(num_edges=NUM_EDGES, span=SPAN, rise=RISE)
    catalog = build_section_catalog(Steel355(), 3)
    formfinder = FdmFormFinder(structure)
    analyzer = TesseractAnalyzer(structure, catalog, backend="opensees")
    sizer = TesseractSizer(structure, catalog, backend="blueprint")
    pipeline = StructuralDesignPipeline(formfinder, analyzer, sizer)

    cases = (
        create_load_uniform(structure, TOTAL_LOAD),
        create_load_half_span(structure, HALF_LOAD),
        create_load_deck_point(structure, POINT_LOAD),
    )
    loads = assemble_load_cases(cases)
    trial = jnp.full(NUM_EDGES, -1.0)
    shape = formfinder(trial, loads.formfinding)
    densities = trial * jnp.max(shape.xyz[:, 2]) / RISE
    parameters = DesignParameters(densities, jnp.full(NUM_EDGES, DIAMETER))

    return pipeline, parameters, loads


def build_objective(
    pipeline: StructuralDesignPipeline,
    loads,
) -> Callable[[Float[Array, "parameters"]], Float[Array, ""]]:
    """Concatenate both parameter blocks into one utilization contraction."""

    def objective(vector):
        split = vector.shape[0] // 2
        parameters = DesignParameters(vector[:split], vector[split:])
        utilization = pipeline(parameters, loads).sizes.utilization

        return jnp.mean(jnp.square(utilization))

    return objective


def central_gradient(
    objective: Callable,
    parameters: Float[Array, "parameters"],
) -> Float[np.ndarray, "parameters"]:
    """Central differences of the complete crossed forward pass."""
    center = np.asarray(parameters, dtype=float)
    quotients = []
    for index, value in enumerate(center):
        stride = STEP * max(abs(float(value)), 1.0)
        moved = center.copy()
        moved[index] += stride
        raised = float(objective(jnp.asarray(moved)))
        moved[index] -= 2.0 * stride
        lowered = float(objective(jnp.asarray(moved)))
        quotients.append((raised - lowered) / (2.0 * stride))

    return np.asarray(quotients)


def measure_pipeline() -> PipelineMeasurement:
    """Measure signed derivative parity and actual warmed wall times."""
    pipeline, parameters, loads = build_problem()
    vector = jnp.concatenate([parameters.shape_parameters, parameters.diameters])
    objective = build_objective(pipeline, loads)
    differentiated = jax.value_and_grad(objective)

    # Warm both paths once.  Tesseract startup and JAX tracing are real setup
    # costs, but neither belongs in a marginal derivative comparison.
    objective(vector)
    differentiated(vector)

    started = time.perf_counter()
    objective(vector)
    forward_seconds = time.perf_counter() - started

    started = time.perf_counter()
    _, reverse = differentiated(vector)
    reverse_seconds = time.perf_counter() - started

    started = time.perf_counter()
    central = central_gradient(objective, vector)
    central_seconds = time.perf_counter() - started

    reverse_host = np.asarray(reverse)
    split = NUM_EDGES
    density_error = relative(central[:split], reverse_host[:split])
    diameter_error = relative(central[split:], reverse_host[split:])
    combined_error = relative(central, reverse_host)
    names = tuple(f"q{index}" for index in range(split)) + tuple(
        f"d{index}" for index in range(split)
    )
    kinds = ("force density",) * split + ("diameter",) * split

    return PipelineMeasurement(
        parameter_names=names,
        parameter_kinds=kinds,
        reverse=tuple(float(value) for value in reverse_host),
        central=tuple(float(value) for value in central),
        density_error=density_error,
        diameter_error=diameter_error,
        combined_error=combined_error,
        forward_seconds=forward_seconds,
        reverse_seconds=reverse_seconds,
        central_seconds=central_seconds,
    )


def report_measurement(report: Report, measured: PipelineMeasurement) -> None:
    """Write the terminal report from an already completed measurement."""
    columns = (
        ReportColumn("parameter", align="<"),
        ReportColumn("kind", align="<"),
        ReportColumn("reverse", "+.8e"),
        ReportColumn("central", "+.8e"),
    )
    rows = zip(
        measured.parameter_names,
        measured.parameter_kinds,
        measured.reverse,
        measured.central,
    )
    report.write_heading("Signed derivatives of the three-stage pipeline")
    report.write_table(columns, rows)
    report.write_heading("Measured cost, after warmup")
    report.write_entries(
        (
            ("one forward pass", f"{measured.forward_seconds:.4f} s"),
            ("one reverse sweep", f"{measured.reverse_seconds:.4f} s"),
            ("central differences", f"{measured.central_seconds:.4f} s"),
        )
    )
    checks = (
        ToleranceCheck("force-density gradient", measured.density_error, TOLERANCE),
        ToleranceCheck("diameter gradient", measured.diameter_error, TOLERANCE),
        ToleranceCheck("complete gradient", measured.combined_error, TOLERANCE),
    )
    report.write_heading("Measured against its bound")
    report.write_checks(checks)
    report.write_verdict(verify_checks(checks))


def main(verbose: bool = True) -> None:
    """Run and report the complete pipeline measurement."""
    report = Report(verbose)
    report.write_line(
        "Form finding -> OpenSees -> Blueprints, three load cases, one adjoint"
    )
    report_measurement(report, measure_pipeline())


if __name__ == "__main__":
    main()
