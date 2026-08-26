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
A recorded nested search, turned back into the designs it walked through.

A trajectory remembers force densities and nothing else, and the pipeline is
deterministic in its parameters, so carrying every iterate back through it at
the same frozen seed diameters reconstructs every intermediate design exactly.
An artifact embeds the run's own description beside the record, so a replay
needs nothing from the process that ran the search. The one figure here draws
the objective the search read.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure

from normax.design import DesignParameters
from normax.design import StructuralDesignPipeline
from normax.design import compute_mass
from normax.extras.nested import Trajectory
from normax.extras.nested import design_envelope
from normax.extras.nested import governing_load_case
from normax.extras.nested import size_design
from normax.loads import LoadCases

# Color of everything that is a reference rather than a result.
GREY = "0.55"


class TrajectoryArtifact(NamedTuple):
    """
    A recorded search and the file that described its run, self-contained.

    Attributes
    ----------
    trajectory :
        Where the optimizer went, in the order it went there.
    config_text :
        Verbatim text of the file the run was described by.
    """

    trajectory: Trajectory
    config_text: str


class DesignHistory(NamedTuple):
    """
    Every iterate of a search, carried through the pipeline that found it.

    Attributes
    ----------
    xyz :
        Position of every node at every step.
    lengths :
        Length of every member at every step.
    diameter :
        Reconciled outer diameter of every member at every step.
    thickness :
        Reconciled wall thickness of every member at every step.
    axial_force :
        Axial force in every member under every load case at every step,
        tension positive.
    utilization :
        Demand over resistance at the reconciled section, per load case.
    governing :
        Index of the load case that decided each member's size, per step.
    mass :
        Total mass of the design at every step.

    Notes
    -----
    The utilization is the reconciled section re-read against its standard —
    at most one, exactly one for the governing case — and the mass is the
    design's, where a trajectory's own column is the penalized objective.
    """

    xyz: Float[Array, "steps nodes 3"]
    lengths: Float[Array, "steps members"]
    diameter: Float[Array, "steps members"]
    thickness: Float[Array, "steps members"]
    axial_force: Float[Array, "steps load_cases members"]
    utilization: Float[Array, "steps load_cases members"]
    governing: Int[Array, "steps members"]
    mass: Float[Array, "steps"]


def save_trajectory(
    path: Path,
    trajectory: Trajectory,
    config_text: str,
) -> None:
    """
    Write a search and the file that described it into one npz artifact.

    Parameters
    ----------
    path :
        File to write, ending in `.npz`.
    trajectory :
        Where the optimizer went, in the order it went there.
    config_text :
        Verbatim text of the file the run was described by.

    Notes
    -----
    The text rides as a plain unicode array, so reading back never pickles.
    """
    embedded = np.array(config_text)
    np.savez(
        path,
        q=np.asarray(trajectory.q),
        mass=np.asarray(trajectory.mass),
        beta=np.asarray(trajectory.beta),
        config_text=embedded,
    )


def load_trajectory(path: Path) -> TrajectoryArtifact:
    """
    Read a search and its run description back out of an npz artifact.

    Parameters
    ----------
    path :
        File to read, as written by `save_trajectory`.

    Returns
    -------
    artifact :
        The recorded search and the text that described its run.
    """
    with np.load(path, allow_pickle=False) as archive:
        trajectory = Trajectory(
            q=jnp.asarray(archive["q"]),
            mass=jnp.asarray(archive["mass"]),
            beta=jnp.asarray(archive["beta"]),
        )
        config_text = archive["config_text"].item()

    return TrajectoryArtifact(trajectory, config_text)


def replay_step(
    pipeline: StructuralDesignPipeline,
    loads: LoadCases,
    params: DesignParameters,
    sharpness: Float[Array, ""] | None,
) -> DesignHistory:
    """
    One iterate carried through the pipeline, as a history of one step.

    Parameters
    ----------
    pipeline :
        The three blocks the search composed.
    loads :
        The form-finding case and the checked cases.
    params :
        The iterate's force densities, and the frozen diameters the frame
        was analyzed with.
    sharpness :
        Sharpness of the envelope, or None for the true largest.

    Returns
    -------
    frame :
        A one-step history, ready to be stacked with its neighbors.

    Notes
    -----
    Which case governs is read before the envelope, the utilization after it.
    """
    design = size_design(pipeline, params, loads)
    governing = governing_load_case(design.sizes.sections.diameter)

    sized = design_envelope(design, sharpness)
    covering = sized.sizes.sections
    worked = pipeline.sizer.compute_utilization(
        covering.diameter,
        design.forces,
        design.shape.lengths,
    )
    mass = compute_mass(sized)

    frame = DesignHistory(
        xyz=sized.shape.xyz[None],
        lengths=sized.shape.lengths[None],
        diameter=covering.diameter[None],
        thickness=covering.thickness[None],
        axial_force=design.forces.axial_force[None],
        utilization=worked[None],
        governing=governing[None],
        mass=mass[None],
    )

    return frame


def replay_trajectory(
    pipeline: StructuralDesignPipeline,
    loads: LoadCases,
    trajectory: Trajectory,
    diameters: Float[Array, "members"],
) -> DesignHistory:
    """
    Every iterate of a recorded search, carried back through its pipeline.

    Parameters
    ----------
    pipeline :
        The three blocks the search composed, rebuilt from the same structure.
    loads :
        The load cases the search designed against, rebuilt the same way.
    trajectory :
        Where the optimizer went, in the order it went there.
    diameters :
        The frozen seed diameters every iterate was analyzed at.

    Returns
    -------
    history :
        Every step's design, stacked along a leading step axis.

    Raises
    ------
    ValueError
        If the trajectory mixes iterates taken under an envelope with
        iterates taken under none.

    Notes
    -----
    The sharpness is read off the trajectory's own column and passed traced,
    so one compilation covers the replay; a zero column replays under the
    true largest.
    """
    recorded = np.asarray(trajectory.beta)

    def step_frame(
        q: Float[Array, "members"],
        sharpness: Float[Array, ""] | None,
    ) -> DesignHistory:
        params = DesignParameters(q, diameters)
        return replay_step(pipeline, loads, params, sharpness)

    compiled = eqx.filter_jit(step_frame)

    if np.all(recorded == recorded[0]):
        sharpness = None if recorded[0] == 0.0 else trajectory.beta[0]
        frames = [compiled(q, sharpness) for q in trajectory.q]
    elif np.all(recorded > 0.0):
        walked = zip(trajectory.q, trajectory.beta, strict=True)
        frames = [compiled(q, sharpness) for q, sharpness in walked]
    else:
        raise ValueError(
            "trajectory mixes zero and positive sharpnesses, so the envelope "
            "its iterates were taken under cannot be read back"
        )

    history = jax.tree.map(lambda *steps: jnp.concatenate(steps), *frames)

    return history


def figure_trajectory(
    trajectories: Sequence[Trajectory],
    *,
    titles: Sequence[str] | None = None,
    concatenated: bool = False,
) -> Figure:
    """
    The objective at every iterate, for one search or several in a row.

    Parameters
    ----------
    trajectories :
        The runs to draw, each recorded by one search.
    titles :
        Name of each run, shown in the legend, or None to number them.
    concatenated :
        Whether to draw the runs end to end on one iteration axis rather
        than overlaid from iteration zero.

    Returns
    -------
    figure :
        The descent of the objective, one curve per run.

    Notes
    -----
    Iterates are colored by sharpness only when every run carries a positive
    one; a logarithmic color scale has no place for the zero stamp.
    """
    masses = [np.asarray(walked.mass) for walked in trajectories]
    sharpnesses = [np.asarray(walked.beta) for walked in trajectories]
    if titles is None:
        titles = tuple(f"run {index + 1}" for index in range(len(trajectories)))

    figure, ax = plt.subplots(figsize=(7.0, 4.2), layout="constrained")

    shades = ("#c0392b", "#35b779", "#31688e")
    stamped = all(float(np.min(sharpness)) > 0.0 for sharpness in sharpnesses)

    # One norm across every run, or the colors of two runs cannot be compared.
    coloring = None
    if stamped:
        dimmest = min(float(np.min(sharpness)) for sharpness in sharpnesses)
        sharpest = max(float(np.max(sharpness)) for sharpness in sharpnesses)
        coloring = LogNorm(dimmest, sharpest)

    offset = 0
    scatter = None
    for index, (mass, sharpness) in enumerate(zip(masses, sharpnesses, strict=True)):
        steps = np.arange(len(mass)) + offset
        ax.plot(
            steps,
            mass,
            "-",
            color=shades[index % len(shades)],
            lw=1.4,
            label=titles[index],
        )
        if coloring is not None:
            scatter = ax.scatter(
                steps, mass, c=sharpness, cmap="viridis", norm=coloring, s=14, zorder=2
            )
        if concatenated:
            offset += len(mass)
            if index < len(masses) - 1:
                ax.axvline(offset - 0.5, color=GREY, ls=":", lw=1.0)

    ax.set_xlabel("iteration")
    ax.set_ylabel("objective [t]")
    final = float(masses[-1][-1])
    ax.set_title(f"Descent, {final:.4f} t at the answer", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    if scatter is not None:
        figure.colorbar(scatter, ax=ax, label=r"envelope sharpness $\beta$")

    return figure
