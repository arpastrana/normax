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
The 101 search replayed and rendered, one polyscope frame per iterate.

**The artifact is the whole input.** `101_api.py` records where its search
went and embeds the file that described the run, so this experiment rebuilds
the same structure, loads and pipeline from the artifact alone and carries
every iterate back through them. The replay is exact because the search
analyzed every iterate at the same frozen seed diameters handed back here —
and it is checked rather than trusted: the penalized objective recomputed
from the replay is compared against the trajectory's own record.

**Members are drawn at their actual size.** The tube radius is the reconciled
outer diameter in world units, times a stated exaggeration, so the animation
shows the sections reorganizing as the form moves. Four fields color the
members — the worst axial force, the utilization at the reconciled section,
the governing load case and the diameter itself — each on one color scale for
the whole animation. The governing index rides a continuous colormap over
half-open integer bands, polyscope having no categorical edge colors.

Frames only: numbered PNGs per field, under `figures/102_frames/`, ready for
whatever assembles them.

Run with `uv run --group pipeline --group viz python
experiments/102_animation.py [artifact.npz]`.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import jax.numpy as jnp
import numpy as np

from normax.optimization import penalized_mass
from normax.rendering import RenderSettings
from normax.rendering import initialize_scene
from normax.rendering import render_frames
from normax.replay import DesignHistory
from normax.replay import load_trajectory
from normax.replay import replay_trajectory

# The artifact 101 writes, unless another file is named on the command line.
ARTIFACT = Path(__file__).resolve().parent.parent / "artifacts" / "101_trajectory.npz"

FRAMES = Path(__file__).resolve().parent.parent / "figures" / "102_frames"

# Factor the drawn tube radius exceeds the true one by, stated when rendering.
EXAGGERATION = 2.0


def load_showcase(path: Path) -> ModuleType:
    """
    The 101 experiment as a module, its digit-led name notwithstanding.

    Parameters
    ----------
    path :
        File the showcase experiment lives in.

    Returns
    -------
    module :
        The loaded module, whose builders this experiment reuses.
    """
    spec = importlib.util.spec_from_file_location("api_101", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def print_members(history: DesignHistory, governing_names: tuple[str, ...]) -> None:
    """
    Every member at the answer, one line each.

    Parameters
    ----------
    history :
        The replayed search, whose last step is the answer.
    governing_names :
        Name of every load case, indexed by the governing column.
    """
    diameters = np.asarray(history.diameter[-1])
    thicknesses = np.asarray(history.thickness[-1])
    forces = np.asarray(history.axial_force[-1])
    utilizations = np.asarray(history.utilization[-1])
    governing = np.asarray(history.governing[-1])

    print("\nmember  d [mm]   t [mm]   worst N [kN]  utilization  governs")
    for member in range(diameters.shape[0]):
        worst_case = int(np.argmax(np.abs(forces[:, member])))
        axial = forces[worst_case, member] / 1e3
        worked = float(np.max(utilizations[:, member]))
        name = governing_names[int(governing[member])]
        print(
            f"{member:6d}  {diameters[member]:7.1f}  {thicknesses[member]:7.2f}"
            f"  {axial:12.1f}  {worked:11.6f}  {name}"
        )


def main(artifact_path: Path) -> None:
    """
    Replay the recorded search, check it against its record, render it.

    Parameters
    ----------
    artifact_path :
        The npz artifact a search was recorded into.
    """
    api = load_showcase(Path(__file__).with_name("101_api.py"))

    artifact = load_trajectory(artifact_path)
    config = api.parse_config(artifact.config_text)
    structure = api.build_arch(config.structure)
    loads = api.arch_load_cases(structure, config.load_cases)
    pipeline = api.build_pipeline(structure, config)
    params = api.initialize_parameters(structure, config)

    history = replay_trajectory(pipeline, loads, artifact.trajectory, params.diameters)

    # The replay checked against the record: the trajectory's column is the
    # penalized objective, so the comparison goes through the same penalty.
    floor = config.optimization.length_floor
    floor_length = floor.fraction * config.structure.span / config.structure.num_edges
    weighed = [
        penalized_mass(
            history.mass[step],
            history.lengths[step],
            floor_length,
            beta=floor.sharpness,
            weight=floor.weight,
        )
        for step in range(history.mass.shape[0])
    ]
    recomputed = jnp.stack(weighed)
    gap = float(jnp.max(jnp.abs(recomputed / artifact.trajectory.mass - 1.0)))

    steps = history.mass.shape[0]
    print(f"Steps replayed: {steps}")
    print(f"Largest relative gap to the recorded objective: {gap:.3e}")
    print(f"Mass at the answer: {float(history.mass[-1]):.9f} t")

    load_case_names = tuple(load_case.name for load_case in config.load_cases)
    print_members(history, load_case_names)

    print(f"\nTube radii drawn at {EXAGGERATION:.0f}x their true size.")
    edges = np.asarray(structure.edges)
    initialize_scene()

    field_settings = (
        RenderSettings("utilization", None, EXAGGERATION, "viridis"),
        RenderSettings("diameter", None, EXAGGERATION, "viridis"),
    )
    for settings in field_settings:
        directory = FRAMES / settings.field
        written = render_frames(history, edges, settings, directory)
        print(f"{settings.field}: {len(written)} frames in {directory}")

    print("\nHasta la vista, baby!")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ARTIFACT)
