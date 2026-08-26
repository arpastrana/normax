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
Frames for a replayed search, in polyscope and nothing else.

The experiments replay and this module renders. Members are drawn as tubes at
their actual outer diameter — in world units, times a stated exaggeration —
and colored by one field of the history at a time, on one color scale for the
whole animation so the frames compare. Nothing here opens a viewer: every
function writes screenshots and returns.

The camera and the scene extents are fixed once, over the union of every
step's geometry, before the first frame is taken. Left to itself polyscope
recenters on whatever it currently shows, and a shape that moves between
frames would drag the camera with it.

This module computes with numpy alone. The history it receives is the last
JAX product in the chain, and the conversion at entry is the boundary.
"""

from pathlib import Path
from typing import NamedTuple

import numpy as np
import polyscope as ps
from jaxtyping import Float
from jaxtyping import Int

from normax.replay import DesignHistory

# Pixels of the window every screenshot is taken at.
WINDOW_SIZE = (1600, 900)

# Name the structure is registered under with polyscope.
NETWORK_NAME = "structure"


class RenderSettings(NamedTuple):
    """
    How a history is turned into frames.

    Attributes
    ----------
    field :
        Field of the history to color members by: "axial_force",
        "utilization", "governing" or "diameter".
    load_case :
        Index of the load case to read the field at. None takes the worst
        case per member, and fields without a load case axis ignore it.
    exaggeration :
        Factor the drawn tube radius exceeds the true one by. One draws to
        scale, where a hundred millimeter tube on a ten meter span is a hair.
    colormap :
        Name of the polyscope colormap the field is drawn with.
    """

    field: str
    load_case: int | None
    exaggeration: float
    colormap: str


def initialize_scene() -> None:
    """
    Start polyscope, bare of everything a structure would bring along.

    Notes
    -----
    Orthographic and undecorated: no ground plane, z up, a fixed window. The
    camera is not placed here — polyscope renders nothing framed on an empty
    scene, so the view is fixed by `frame_camera` once a structure exists.
    """
    ps.init()
    ps.set_window_size(*WINDOW_SIZE)
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("none")
    ps.set_view_projection_mode("orthographic")


def frame_camera(positions: Float[np.ndarray, "steps nodes 3"]) -> None:
    """
    Fix the camera and the scene extents over a whole animation.

    Parameters
    ----------
    positions :
        Every node position of every step, whose union the view is framed on.

    Notes
    -----
    Called after a structure is registered, or the view frames nothing.
    Looking along the y axis, which faces the plane the arches live in; a
    genuinely spatial structure still reads, in elevation. Left to itself
    polyscope recenters on whatever it currently shows, and a shape that
    moves between frames would drag the camera with it.
    """
    flattened = positions.reshape(-1, 3)
    low = flattened.min(axis=0)
    high = flattened.max(axis=0)
    center = 0.5 * (low + high)
    extent = float(np.linalg.norm(high - low))

    # In polyscope's orthographic view the eye distance is the zoom.
    offset = np.asarray([0.0, 0.85 * extent, 0.0])
    eye = center - offset

    ps.set_automatically_compute_scene_extents(False)
    ps.set_bounding_box(low, high)
    ps.look_at(eye, center)


def frame_colors(
    history: DesignHistory,
    field: str,
    load_case: int | None,
) -> Float[np.ndarray, "steps members"]:
    """
    What every member is colored by at every step, one number per member.

    Parameters
    ----------
    history :
        The replayed search.
    field :
        Field of the history to read: "axial_force", "utilization",
        "governing" or "diameter".
    load_case :
        Index of the load case to read at. None takes the worst case per
        member — the largest utilization, or the axial force largest in
        magnitude with its sign kept.

    Returns
    -------
    colors :
        The field, reduced to one value per member per step.

    Raises
    ------
    ValueError
        If the field is not one this module knows how to color by.
    """
    if field == "diameter":
        return np.asarray(history.diameter)
    if field == "governing":
        return np.asarray(history.governing, dtype=float)
    if field == "utilization":
        cased = np.asarray(history.utilization)
        if load_case is not None:
            return cased[:, load_case]
        return cased.max(axis=1)
    if field == "axial_force":
        cased = np.asarray(history.axial_force)
        if load_case is not None:
            return cased[:, load_case]
        worst_case = np.argmax(np.abs(cased), axis=1)
        signed_worst = np.take_along_axis(cased, worst_case[:, None, :], axis=1)
        return signed_worst[:, 0]

    raise ValueError(f"no color rule for field {field!r}")


def color_limits(
    field: str,
    colors: Float[np.ndarray, "steps members"],
    load_cases: int,
) -> tuple[float, float]:
    """
    One color scale for a whole animation.

    Parameters
    ----------
    field :
        Field the colors were read from.
    colors :
        Every frame's colors at once, whose range the scale covers.
    load_cases :
        Number of load cases, which the governing field is an index into.

    Returns
    -------
    limits :
        Smallest and largest value the colormap is stretched over.

    Notes
    -----
    An axial force that changes sign gets a scale symmetric about zero, so
    tension and compression read as hue on a diverging map; one that never
    does spans what it spans, keeping the contrast a wasted half would cost.
    The governing index gets half-open bands, one per case.
    """
    smallest = float(colors.min())
    largest = float(colors.max())

    if field == "axial_force" and smallest < 0.0 < largest:
        widest = max(-smallest, largest)
        return (-widest, widest)
    if field == "governing":
        return (-0.5, load_cases - 0.5)

    return (smallest, largest)


def render_frames(
    history: DesignHistory,
    edges: Int[np.ndarray, "members 2"],
    settings: RenderSettings,
    directory: Path,
) -> list[Path]:
    """
    One numbered PNG per step of a history, colored by one field.

    Parameters
    ----------
    history :
        The replayed search.
    edges :
        Which two nodes each member connects.
    settings :
        The field to color by, and how to draw it.
    directory :
        Directory the frames are written into, made if it is missing.

    Returns
    -------
    written :
        The frame files, in step order.

    Notes
    -----
    Registering the network under the same name replaces it, so rendering
    several fields in a row reuses one scene, and the camera is re-fixed on
    the same union of steps every time. The tube radius is the true member
    radius times the stated exaggeration, in world units.
    """
    positions = np.asarray(history.xyz)
    connectivity = np.asarray(edges)
    radii = 0.5 * np.asarray(history.diameter) * settings.exaggeration
    colors = frame_colors(history, settings.field, settings.load_case)

    load_cases = int(np.asarray(history.axial_force).shape[1])
    limits = color_limits(settings.field, colors, load_cases)

    directory.mkdir(parents=True, exist_ok=True)
    network = ps.register_curve_network(NETWORK_NAME, positions[0], connectivity)
    frame_camera(positions)

    written = []
    for index in range(positions.shape[0]):
        network.update_node_positions(positions[index])
        network.add_scalar_quantity("radius", radii[index], defined_on="edges")
        network.set_edge_radius_quantity("radius", autoscale=False)
        network.add_scalar_quantity(
            settings.field,
            colors[index],
            defined_on="edges",
            enabled=True,
            vminmax=limits,
            cmap=settings.colormap,
        )

        frame_path = directory / f"frame_{index:04d}.png"
        ps.screenshot(str(frame_path), transparent_bg=False)
        written.append(frame_path)

    return written
