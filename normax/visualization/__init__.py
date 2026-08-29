# SPDX-License-Identifier: Apache-2.0
"""
What a run draws: figures written to disk.

The figures are matplotlib and nothing else, so every install has them. An
interactive viewer lived here too until 2026-08-28, drawing whole solver
responses in three dimensions through packages that were never dependencies of
this one; it went with the oracles it drew with.
"""

from normax.visualization.plots import draw_design_figures

__all__ = [
    "draw_design_figures",
]
