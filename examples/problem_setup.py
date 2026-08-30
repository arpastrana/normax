# SPDX-License-Identifier: Apache-2.0
"""
The bridge problem shared by the arch, Warren, and Vierendeel examples.

All three planar systems span the same flat deck and carry the same three load
patterns, so their common boundary conditions and loading belong in one figure
rather than beside each topology. Run with
`uv run python examples/problem_setup.py`; the figure is written to
`figures/problem_setup.{png,svg,pdf}` and a three-column `problem_setup_landscape`
variant without running a design search.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from normax.config import LoadCaseConfig
from normax.config import parse_config
from normax.loads import build_load_cases
from normax.structures import ArchDescription
from normax.structures import TrussDescription
from normax.structures import build_arch_2d
from normax.visualization import draw_problem_setup

REPO = Path(__file__).resolve().parent.parent
OUTPUT = REPO / "figures" / "problem_setup"
FIGURE_DPI = 200
FIGURE_FORMATS = ("png", "svg", "pdf")
FIGURE_LAYOUTS = {"problem_setup": "vertical", "problem_setup_landscape": "horizontal"}

# An even number puts a node exactly at midspan. This is an idealized deck,
# deliberately independent of the discretization of any one structural system.
NUM_SEGMENTS = 10
PERSON_HEIGHT = 1750.0

CONFIGS = (
    (Path(__file__).with_name("arch.yaml"), ArchDescription),
    (Path(__file__).with_name("warren.yaml"), TrussDescription),
    (Path(__file__).with_name("vierendeel.yaml"), TrussDescription),
)

CASE_TITLES = (
    "Uniform load",
    "Asymmetric half-span load",
    "Point load at midspan",
)


def read_common_problem() -> tuple[float, tuple[LoadCaseConfig, ...]]:
    """
    Read the span and load cases the three planar examples have in common.

    Raises
    ------
    ValueError
        If one example drifts away from the shared bridge problem.
    """
    configs = [parse_config(path.read_text(), kind) for path, kind in CONFIGS]
    span = float(configs[0].structure.span)
    load_cases = configs[0].load_cases
    if any(float(config.structure.span) != span for config in configs[1:]):
        raise ValueError("the three planar examples no longer share one span")
    if any(config.load_cases != load_cases for config in configs[1:]):
        raise ValueError("the three planar examples no longer share their load cases")

    return span, load_cases


def main() -> None:
    """Draw and save the problem statement without running an optimization."""
    span, described = read_common_problem()
    deck = build_arch_2d(NUM_SEGMENTS, span, rise=0.0)
    loads = build_load_cases(deck, described)
    span_label = rf"$L = {span / 1000.0:g}\,\mathrm{{m}}$"
    OUTPUT.parent.mkdir(exist_ok=True)
    outputs = []
    for stem, layout in FIGURE_LAYOUTS.items():
        figure = draw_problem_setup(
            deck,
            loads.analysis,
            CASE_TITLES,
            span_label,
            layout=layout,
            person_height=PERSON_HEIGHT,
        )
        target = OUTPUT.with_name(stem)
        for extension in FIGURE_FORMATS:
            output = target.with_suffix(f".{extension}")
            figure.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight")
            outputs.append(output)
        plt.close(figure)
    print(f"Saved {', '.join(str(output) for output in outputs)}")


if __name__ == "__main__":
    main()
