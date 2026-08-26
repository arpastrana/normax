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
Every drawing the pipeline makes, one submodule per backend.

`figures` returns matplotlib figures and never shows them, and is what this
package re-exports so a caller imports two levels deep.

`frames` and `viewer` are deliberately absent. `frames` drives polyscope
offscreen and reaches the frame solver through the replay it draws; `viewer`
opens a window that blocks until it closes and reaches for the local viewer.
Both are imported by path, by the scripts that want them, so drawing a figure
never pays for either.
"""

from normax.visualization.figures import GREY
from normax.visualization.figures import WIDTH_MAX
from normax.visualization.figures import BackendAgreement
from normax.visualization.figures import BackendTimings
from normax.visualization.figures import BeamSizing
from normax.visualization.figures import BeamStatics
from normax.visualization.figures import ColorRange
from normax.visualization.figures import Descent
from normax.visualization.figures import DescentTrace
from normax.visualization.figures import DrawnStructure
from normax.visualization.figures import Form
from normax.visualization.figures import GapScaling
from normax.visualization.figures import GradientCheck
from normax.visualization.figures import HandoffForces
from normax.visualization.figures import MassSweep
from normax.visualization.figures import MeshRefinement
from normax.visualization.figures import SearchTrace
from normax.visualization.figures import ShapeVariation
from normax.visualization.figures import SizedMembers
from normax.visualization.figures import StaggeredPasses
from normax.visualization.figures import StartSpread
from normax.visualization.figures import SubspaceMode
from normax.visualization.figures import TrussForm
from normax.visualization.figures import UtilizationForm
from normax.visualization.figures import draw_members
from normax.visualization.figures import draw_outline
from normax.visualization.figures import figure_backends
from normax.visualization.figures import figure_beam_profile
from normax.visualization.figures import figure_benchmark
from normax.visualization.figures import figure_convergence
from normax.visualization.figures import figure_density_modes
from normax.visualization.figures import figure_handoff
from normax.visualization.figures import figure_load_cases
from normax.visualization.figures import figure_mass_descent
from normax.visualization.figures import figure_optimization
from normax.visualization.figures import figure_parametrization
from normax.visualization.figures import figure_sections
from normax.visualization.figures import figure_shape_variations
from normax.visualization.figures import figure_trajectory
from normax.visualization.figures import figure_truss_forms
from normax.visualization.figures import figure_utilization

__all__ = [
    "BackendAgreement",
    "BackendTimings",
    "BeamSizing",
    "BeamStatics",
    "ColorRange",
    "Descent",
    "DescentTrace",
    "DrawnStructure",
    "Form",
    "GREY",
    "GapScaling",
    "GradientCheck",
    "HandoffForces",
    "MassSweep",
    "MeshRefinement",
    "SearchTrace",
    "ShapeVariation",
    "SizedMembers",
    "StaggeredPasses",
    "StartSpread",
    "SubspaceMode",
    "TrussForm",
    "UtilizationForm",
    "WIDTH_MAX",
    "draw_members",
    "draw_outline",
    "figure_backends",
    "figure_beam_profile",
    "figure_benchmark",
    "figure_convergence",
    "figure_density_modes",
    "figure_handoff",
    "figure_load_cases",
    "figure_mass_descent",
    "figure_optimization",
    "figure_parametrization",
    "figure_sections",
    "figure_shape_variations",
    "figure_trajectory",
    "figure_truss_forms",
    "figure_utilization",
]
