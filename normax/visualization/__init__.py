# SPDX-License-Identifier: Apache-2.0
"""
What a run draws: figures written to disk, and designs looked at.

Two ways of seeing the same design, told apart by what they need. The figures
are matplotlib and nothing else, so every install has them. The viewer reads a
whole solver response and draws it in three dimensions, which takes packages
that are not dependencies of this one, so it is loaded only where they are
found and stood in for everywhere else.
"""

from normax.visualization.guard import VIEWER_PACKAGES
from normax.visualization.guard import find_viewer
from normax.visualization.plots import draw_design_figures

if find_viewer():
    from normax.visualization.viewer import view_design
    from normax.visualization.viewer import view_designs
else:
    from normax.visualization.unavailable import view_design
    from normax.visualization.unavailable import view_designs

__all__ = [
    "VIEWER_PACKAGES",
    "draw_design_figures",
    "find_viewer",
    "view_design",
    "view_designs",
]
