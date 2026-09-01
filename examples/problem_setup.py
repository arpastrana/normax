# SPDX-License-Identifier: Apache-2.0
"""
The load cases of the examples, drawn without running any design search.

The three planar systems span the same flat deck and carry the same three load
patterns, so their common boundary conditions and loading belong in one figure
rather than beside each topology; the gridshell carries its own pressure cases
and gets its own row. Run with `uv run python examples/problem_setup.py`; the
figures are written to `figures/problem_setup.{png,svg,pdf}`, a three-column
`problem_setup_landscape` variant, a `problem_setup_gridshell` landscape, and
a `problem_setup_gridshell_plan` top view shading each pressure over the roof.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from normax.config import LoadCaseConfig
from normax.config import parse_config
from normax.loads import build_load_cases
from normax.structures import ArchDescription
from normax.structures import ShellDescription
from normax.structures import TrussDescription
from normax.structures import build_arch_2d
from normax.structures import build_shell
from normax.visualization import draw_problem_plan
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

GRIDSHELL = Path(__file__).with_name("gridshell.yaml")

SHELL_TITLES = (
    "Tributary pressure",
    "Drift over spoke 4",
    "Drift over spoke 12",
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


def save_figure(figure: Figure, target: Path) -> list[Path]:
    """
    Save one figure in every shipped format, then close it.
    """
    outputs = []
    for extension in FIGURE_FORMATS:
        output = target.with_suffix(f".{extension}")
        figure.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight")
        outputs.append(output)
    plt.close(figure)

    return outputs


def main() -> None:
    """
    Draw and save the problem statements without running an optimization.
    """
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
        outputs.extend(save_figure(figure, OUTPUT.with_name(stem)))

    config = parse_config(GRIDSHELL.read_text(), ShellDescription)
    shell = build_shell(config.structure)
    shell_loads = build_load_cases(shell, config.load_cases)
    diameter = 2.0 * config.structure.radius
    shell_label = rf"$D = {diameter / 1000.0:g}\,\mathrm{{m}}$"
    shell_figure = draw_problem_setup(
        shell,
        shell_loads.analysis,
        SHELL_TITLES,
        shell_label,
        layout="horizontal",
    )
    shell_target = OUTPUT.with_name("problem_setup_gridshell")
    outputs.extend(save_figure(shell_figure, shell_target))
    plan_figure = draw_problem_plan(
        shell,
        shell_loads.analysis,
        SHELL_TITLES,
        shell_label,
    )
    plan_target = OUTPUT.with_name("problem_setup_gridshell_plan")
    outputs.extend(save_figure(plan_figure, plan_target))
    print(f"Saved {', '.join(str(output) for output in outputs)}")


if __name__ == "__main__":
    main()
