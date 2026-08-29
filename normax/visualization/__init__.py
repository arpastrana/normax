# SPDX-License-Identifier: Apache-2.0
"""
What a run draws: figures written to disk.

The figures are matplotlib and nothing else, so every install has them, and so
is the animation, which Pillow writes rather than ffmpeg. An interactive viewer
lived here too until 2026-08-28, drawing whole solver responses in three
dimensions through packages that were never dependencies of this one; it went
with the oracles it drew with.
"""

from normax.visualization.animations import animate_descent
from normax.visualization.animations import save_animation
from normax.visualization.plots import DescentPanel
from normax.visualization.plots import DescentTrace
from normax.visualization.plots import draw_design_figures
from normax.visualization.plots import read_round_bounds
from normax.visualization.plots import track_best_feasible

__all__ = [
    "DescentPanel",
    "DescentTrace",
    "animate_descent",
    "draw_design_figures",
    "read_round_bounds",
    "save_animation",
    "track_best_feasible",
]
