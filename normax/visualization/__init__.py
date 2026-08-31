# SPDX-License-Identifier: Apache-2.0
"""
What a run draws: figures written to disk.

The figures are matplotlib and nothing else, so every install has them, and so
is the animation, which Pillow writes rather than ffmpeg. An interactive viewer
lived here too until 2026-08-28, drawing whole solver responses in three
dimensions through packages that were never dependencies of this one; it went
with the oracles it drew with.
"""

from normax.visualization.animations import GIF_FOR_READING
from normax.visualization.animations import GifReduction
from normax.visualization.animations import animate_descent
from normax.visualization.animations import convert_to_gif
from normax.visualization.animations import save_animation
from normax.visualization.animations import save_gif
from normax.visualization.plots import DescentPanel
from normax.visualization.plots import DescentTrace
from normax.visualization.plots import DrawnFigures
from normax.visualization.plots import DrawnLimits
from normax.visualization.plots import draw_design_figures
from normax.visualization.plots import draw_problem_setup
from normax.visualization.plots import read_round_bounds
from normax.visualization.plots import track_best_feasible
from normax.visualization.validation import draw_code_validation
from normax.visualization.validation import draw_pipeline_validation
from normax.visualization.validation import draw_pynite_validation

__all__ = [
    "DescentPanel",
    "DescentTrace",
    "DrawnLimits",
    "DrawnFigures",
    "animate_descent",
    "GIF_FOR_READING",
    "GifReduction",
    "convert_to_gif",
    "draw_design_figures",
    "draw_code_validation",
    "draw_pipeline_validation",
    "draw_problem_setup",
    "draw_pynite_validation",
    "read_round_bounds",
    "save_animation",
    "save_gif",
    "track_best_feasible",
]
