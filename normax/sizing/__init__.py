# SPDX-License-Identifier: Apache-2.0
"""
The contract a design standard fills as a block of the pipeline.

A standard is a normative text rather than a solver: it has no derivatives and
is ordinarily implemented as scalar branchy code returning verdicts. What makes
it a block is that it answers two questions with a derivative — how hard a size
is working, and what size would work exactly. Implementations live beside this
module and import nothing from each other.
"""

from normax.sizing.contract import AbstractMemberSizer
from normax.sizing.contract import MemberSizes

__all__ = [
    "AbstractMemberSizer",
    "MemberSizes",
]
