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
The path each search's best descent took, for the animations.

A multi-start run keeps its landings and throws the paths away, which is the
right trade while the question is where a search ends up. Once the question is
what it did on the way, the path is wanted — and it can be had without having
kept it, because the starts are drawn from a fixed seed.

**That the answer comes back identical is the point.** A reconstructed start
that landed somewhere else would be a different descent wearing the same label,
so each re-descent is checked against the mass the multi-start run recorded.

What is written is one frame per objective evaluation. Two things about those
frames are worth knowing before drawing them.

**The path dives below where it lands, and that is the method rather than a
better answer.** A small opening penalty leaves the mass in charge, so the
search crosses the infeasible region before the multipliers pull it back onto
the constraint surface. Drawing the mass alone reads as a regression; draw the
violation beside it.

**A frame is an evaluation, not an accepted step.** A line search tries points
it then rejects, so consecutive frames can move backwards. Fine for a morph, and
for a curve that only descends there is the per-round trail instead.

Run it from the repository root:

    uv run --group pipeline --extra spike python experiments/28_descent_paths.py
"""

import importlib.util
import json
import time
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from normax import searches  # noqa: E402

# The run descriptions the examples take, and the folder this file sits in.
EXPERIMENTS = Path(__file__).resolve().parents[1]
VALIDATION = Path(__file__).resolve().parent

# The four examples, which own the run descriptions an audit reads designs from.
EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
PATHS = EXPERIMENTS.parent / "trajectories"

# The run whose landings are being redrawn, and where they were kept.
DESCRIBED = EXAMPLES / "gridshell.yaml"
KEPT = {
    searches.SEARCH_FORMFOUND: "end_to_end",
    searches.SEARCH_HEIGHTS: "free_heights",
    searches.SEARCH_DRAWN: "sizing_only",
}

# How many starts the multi-start run spent, and how close to feasible a landing
# had to be to count. Both must match that run or a different winner is chosen.
STARTS = 24
FEASIBLE = 1.0e-4


def loaded_module(path: Path):
    """
    One script, loaded by path rather than imported by name.

    Parameters
    ----------
    path :
        The file to load.

    Returns
    -------
    module :
        The loaded module.

    Notes
    -----
    Neither the examples nor the numbered experiments are importable names, so
    a script that reuses another reaches it by path. Taking the file rather
    than a stem is what lets one loader serve both folders.
    """
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def winning_start(stem: str) -> dict:
    """
    The lightest feasible landing the multi-start run recorded for one search.

    Parameters
    ----------
    stem :
        What the search's kept landings are filed under.

    Returns
    -------
    landing :
        That start's record, the start index among it.
    """
    recorded = json.loads((PATHS / f"{stem}_starts.json").read_text())
    feasible = [
        entry
        for entry in recorded
        if entry.get("mass") is not None and entry["violation"] <= FEASIBLE
    ]

    return min(feasible, key=lambda entry: entry["mass"])


def prepared_searches(profile):
    """
    Every search's maps, its starts and its bounds, as the run description asks.

    Parameters
    ----------
    profile :
        The gridshell's search profile.

    Returns
    -------
    prepared :
        The maps by search, the scattered starts by search, and the bounds by
        search.
    """
    with open(DESCRIBED) as handle:
        config = searches.parse_shell(handle)

    structure = profile.build_structure(config)
    plan = profile.build_loads(structure, config)
    folding = searches.folding_maps(profile, config, structure)
    problem = searches.prepare_problem(structure, config, plan, folding)
    opening = profile.signed_start(problem, config)
    guard = None if profile.sign_guard is None else profile.sign_guard(config, opening)
    limits = profile.height_limits(config)
    descent = searches.descent_plan(config)

    maps = searches.search_maps(problem, limits, descent.budget.length_floor, guard)
    finder = problem.pipeline.formfinder
    shape = finder.formfinder(jnp.asarray(opening.q), problem.loads.formfinding)
    seeded = searches.search_starts(
        problem, opening, shape.xyz, descent.budget.diameter_floor
    )
    boxes = searches.search_boxes(problem, descent.budget.diameter_floor, limits)
    widened = descent.budget._replace(starts=STARTS)

    scattered = {
        search: searches.scattered_points(
            np.asarray(seeded[search], dtype=np.float64), boxes[search], widened
        )
        for search in KEPT
    }
    budget = descent.augmented or searches.AUGMENTED_DEFAULT

    return maps, scattered, boxes, budget


def recorded_descent(maps, opening, boxes, budget):
    """
    One descent, with every objective evaluation kept.

    Parameters
    ----------
    maps :
        The search's compiled maps.
    opening :
        The variable vector to descend from.
    boxes :
        One bound pair per variable.
    budget :
        What the augmented descent may spend.

    Returns
    -------
    walked :
        The answer, every visited point, and the mass at each of them.
    """
    visited = []
    inner = maps.augmented

    def watched(variables, multipliers, penalty, scale):
        visited.append(np.asarray(variables, dtype=np.float64).copy())

        return inner(variables, multipliers, penalty, scale)

    answer = searches.descend_augmented_search(
        maps._replace(augmented=watched), opening, boxes, budget
    )
    steps = np.stack(visited)
    masses = np.array([abs(float(maps.weigh(jnp.asarray(z))[0])) for z in steps])

    return answer, steps, masses


def main() -> None:
    """
    Redraw every search's winning path and write it beside the landings.
    """
    PATHS.mkdir(exist_ok=True)
    profile = loaded_module(EXAMPLES / "gridshell.py").GRIDSHELL_PROFILE
    maps, scattered, boxes, budget = prepared_searches(profile)

    for search, stem in KEPT.items():
        expected = winning_start(stem)
        started = int(expected["start"])
        began = time.perf_counter()
        answer, steps, masses = recorded_descent(
            maps[search], scattered[search][started], boxes[search], budget
        )
        spent = (time.perf_counter() - began) / 60.0

        landed = abs(float(maps[search].weigh(jnp.asarray(answer.variables))[0]))
        agrees = landed == expected["mass"]
        np.savez(
            PATHS / f"{stem}.npz",
            search=np.array(search),
            start=np.array(started),
            opening=scattered[search][started],
            steps=steps,
            masses=masses,
            landing=np.asarray(answer.variables),
            rounds=np.asarray(answer.masses),
        )

        print(
            f"  {search:14s} start {started:2d}: {steps.shape[0]:5d} frames of "
            f"{steps.shape[1]:2d} variables, {spent:.1f} min",
            flush=True,
        )
        print(
            f"  {'':14s} landed {landed:.9f}, recorded {expected['mass']:.9f}"
            f" -> {'REPRODUCED' if agrees else 'DIFFERS'}",
            flush=True,
        )
        print(
            f"  {'':14s} mass along the path {masses.max():.6f} down to "
            f"{masses.min():.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
