# SPDX-License-Identifier: Apache-2.0
"""
The contract a frame analysis fills as a block of the pipeline.

A form finder hands over a geometry and nothing else, so the axial forces that
come back are the analysis's own product rather than a restatement of the force
densities that shaped it, and the bending beside them is what the check reads.
Backends live beside this module and import nothing from each other.
"""

from normax.analysis.contract import AbstractFrameAnalyzer
from normax.analysis.contract import MemberForces
from normax.analysis.supports import DOF_PER_NODE
from normax.analysis.supports import find_normal_axis
from normax.analysis.supports import restrain_supports

__all__ = [
    "DOF_PER_NODE",
    "AbstractFrameAnalyzer",
    "MemberForces",
    "find_normal_axis",
    "restrain_supports",
]
