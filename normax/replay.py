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
A recorded search, turned back into the designs it walked through.

A trajectory remembers force densities and nothing else, so the designs the
search moved through are gone the moment it returns. They are not lost: the
pipeline that produced them is deterministic in its parameters, so carrying
every iterate back through it reconstructs every intermediate design exactly —
exactly, because the search analyzed every iterate at the same frozen seed
diameters a replay hands back in.

The artifact half of this module makes the record durable. A search and the
file that described its run are written into one archive, so a replay needs
nothing from the process that ran the search — a different process, on a
different day, rebuilds the pipeline from the embedded text and walks the
same designs.

Everything here is host-side bookkeeping around one jitted step. Nothing is
differentiated, and nothing imports a renderer: this module ends at arrays.
"""

from pathlib import Path
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from normax.design import DesignParameters
from normax.design import StructuralDesignPipeline
from normax.design import compute_mass
from normax.design import design_envelope
from normax.design import governing_load_case
from normax.loads import LoadCases
from normax.optimization import Trajectory


class TrajectoryArtifact(NamedTuple):
    """
    A recorded search and the file that described its run, self-contained.

    Attributes
    ----------
    trajectory :
        Where the optimizer went, in the order it went there.
    config_text :
        Verbatim text of the file the run was described by.

    Notes
    -----
    The text rather than a path, so the artifact stays true when the file it
    was read from moves on. What configured the run is embedded in the record
    of the run, and nothing else has to still exist for a replay to agree
    with it.
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
    **The utilization here is the reconciled section re-read against its
    standard**, `AbstractMemberSizer.compute_utilization` — at most one, and exactly
    one for the case that governs. It is not the fully-stressed diagonal a
    design carries, which is one by construction and says nothing a color
    could show.

    **The mass here is the design's mass**, where a trajectory's own column is
    the objective the search read — penalized wherever a floor was on.
    Reconciling the two is the caller's arithmetic, through the same penalty.
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
    The text rides as a plain unicode array, so reading the artifact back
    never needs pickling. The directory is the caller's to make.
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
        Sharpness of the envelope reconciling the load cases. None takes the
        true largest section any case demands.

    Returns
    -------
    frame :
        A one-step history, ready to be stacked with its neighbors.

    Notes
    -----
    Which case governs is read before the envelope, on the sizes the cases
    demanded on their own; the utilization is read after it, at the one
    section every case must live with.
    """
    design = pipeline(params, loads)
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
        The frozen seed diameters every iterate was analyzed at, which is
        what makes the replay exact rather than approximate.

    Returns
    -------
    history :
        Every step's design, stacked along a leading step axis.

    Raises
    ------
    ValueError
        If the trajectory mixes iterates taken under an envelope with
        iterates taken under none, which no single replay can honor.

    Notes
    -----
    The sharpness each step ran under is read back off the trajectory's own
    column. A constant column replays as one compiled program; an annealed
    one passes the sharpness as a traced argument, so every round still
    shares a single compilation; a zero column is a search that enveloped
    with the true largest, and replays under it.
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
