# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import numpy as np
import pytest

from normax.config import RunArguments
from normax.config import read_run_config
from normax.form_finding import build_form_finder
from normax.structures import ArchDescription
from normax.structures import TrussDescription
from normax.structures import build_arch_2d
from normax.structures import build_vierendeel_2d
from normax.structures import build_warren_2d
from normax.symmetry import build_height_groups
from normax.symmetry import build_section_groups
from normax.symmetry import find_mirror_nodes
from normax.symmetry import permute_free_nodes
from normax.symmetry import permute_members

# The structures whose shipped runs fold by a midspan mirror, and the builder
# each file is read into.
FOLDED_RUNS = {
    "arch": (build_arch_2d, ArchDescription),
    "warren": (build_warren_2d, TrussDescription),
    "vierendeel": (build_vierendeel_2d, TrussDescription),
}


@pytest.mark.parametrize("name", sorted(FOLDED_RUNS))
def test_the_shipped_run_folds_its_heights_and_its_diameters(name):
    # The orbit maps and the folded basis are tested where they are built. What
    # is tested here is the wiring, which is where this last broke: the arch
    # carried `fold_mirror: true` past an example hardcoding no groups at all,
    # and `build_form_finder` dropped the orbits before they reached the
    # written-heights finder. Both folds are linear maps on the host, so no
    # solver, basis or start is built and the whole file runs in milliseconds.
    builder, description = FOLDED_RUNS[name]
    path = Path("examples") / f"{name}.yaml"
    config = read_run_config(RunArguments(path, "heights"), description)
    structure = builder(*tuple(config.structure))
    rng = np.random.default_rng(7)

    assert config.form_finding.fold_heights is True
    assert config.sizing.fold_mirror is True

    mirror = find_mirror_nodes(structure, config.form_finding.mirror)

    # The diameters, which every baseline sizes through: the groups sit on the
    # problem rather than on the form finder, so `fixed` folds them too.
    section_groups = build_section_groups(structure, (mirror, None))
    diameters = section_groups @ rng.uniform(50.0, 400.0, section_groups.shape[1])
    members = permute_members(mirror, structure)

    assert section_groups.shape[1] < structure.num_edges
    assert np.array_equal(diameters, diameters[members])

    # The heights, which only the written-geometry baseline moves. One 1.0 per
    # row in either orbit map, so both are exact rather than close.
    height_groups = build_height_groups(structure, (mirror,))
    formfinder = build_form_finder(structure, None, config.form_finding, height_groups)
    coefficients = rng.uniform(200.0, 2600.0, formfinder.count_shape_coefficients())
    heights = np.asarray(formfinder.expand_shape_coefficients(coefficients))
    nodes_free = permute_free_nodes(mirror, structure)

    assert heights.size == nodes_free.size
    assert formfinder.count_shape_coefficients() < heights.size
    assert np.array_equal(heights, heights[nodes_free])
