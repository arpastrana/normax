# SPDX-License-Identifier: Apache-2.0
"""
The Blueprints backend of the sizing schema.

The check, the bisection and the hand adjoint are the host functions of
`normax.sizing.blueprint`, plain NumPy over a scalar library with no
derivatives of any kind; this module maps the schema's fields onto them and
nothing else. The two solve outputs and the held check pull back through
separate host rules.
"""

from typing import Any

import numpy as np

from normax.sizing.blueprint import ActionCotangents
from normax.sizing.blueprint import HostActions
from normax.sizing.blueprint import HostCatalog
from normax.sizing.blueprint import SizeCotangents
from normax.sizing.blueprint import check_cotangents
from normax.sizing.blueprint import check_members
from normax.sizing.blueprint import coerce_member_actions
from normax.sizing.blueprint import coerce_section_catalog
from normax.sizing.blueprint import size_cotangents
from normax.sizing.blueprint import size_members


def _read_catalog(inputs: dict[str, Any]) -> HostCatalog:
    """
    The section catalog the flat schema scalars describe.
    """
    return coerce_section_catalog(
        float(inputs["ratio"]),
        float(inputs["f_y"]),
        float(inputs["gamma_m0"]),
        float(inputs["diameter_min"]),
    )


def _read_actions(inputs: dict[str, Any]) -> HostActions:
    """
    The member actions the schema arrays describe.
    """
    return coerce_member_actions(
        inputs["axial_force"], inputs["end_moments_major"], inputs["end_moments_minor"]
    )


def solve_sizes(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Run the check on one load case's actions.

    Parameters
    ----------
    raw :
        The schema's input fields, dumped to plain arrays and scalars.

    Returns
    -------
    outputs :
        The required sizes, both utilizations and the clamp mask. Without the
        solve, the held size and its utilization echoed, and no clamp claimed.
    """
    catalog = _read_catalog(raw)
    actions = _read_actions(raw)
    held = np.asarray(raw["diameter_held"], dtype=np.float64)
    utilization_held = check_members(held, actions, catalog)
    if raw["solve"]:
        sized = size_members(actions, catalog)
        diameter = sized.diameter
        utilization = sized.utilization
        clamped = sized.clamped.astype(np.float64)
    else:
        diameter = held
        utilization = utilization_held
        clamped = np.zeros_like(held)
    outputs = {
        "diameter": diameter,
        "utilization": utilization,
        "utilization_held": utilization_held,
        "clamped": clamped,
    }

    return outputs


def sizes_vjp(raw: dict[str, Any], seeds: dict[str, Any]) -> dict[str, Any]:
    """
    Pull the seeded cotangents back to the actions and the held size.

    Parameters
    ----------
    raw :
        The schema's input fields, dumped to plain arrays and scalars.
    seeds :
        Cotangent on every differentiable output, zeros where unseeded.

    Returns
    -------
    gathered :
        Cotangent on every differentiable input.

    Notes
    -----
    Without the solve, `diameter` is the held size and `utilization` is the
    held check, so their seeds pull through the identity and the held rule.
    """
    catalog = _read_catalog(raw)
    actions = _read_actions(raw)
    held = np.asarray(raw["diameter_held"], dtype=np.float64)
    if raw["solve"]:
        sized_seed = SizeCotangents(seeds["diameter"], seeds["utilization"])
        from_sizes = size_cotangents(actions, catalog, sized_seed)
        held_seed = seeds["utilization_held"]
        echoed = np.zeros_like(held)
    else:
        quiet_axial = np.zeros_like(actions.axial)
        quiet_major = np.zeros_like(actions.end_major)
        quiet_minor = np.zeros_like(actions.end_minor)
        from_sizes = ActionCotangents(quiet_axial, quiet_major, quiet_minor)
        held_seed = seeds["utilization_held"] + seeds["utilization"]
        echoed = seeds["diameter"]
    from_held = check_cotangents(held, actions, catalog, held_seed)
    gathered = {
        "axial_force": from_sizes.axial + from_held.actions.axial,
        "end_moments_major": from_sizes.end_major + from_held.actions.end_major,
        "end_moments_minor": from_sizes.end_minor + from_held.actions.end_minor,
        "diameter_held": from_held.diameter_held + echoed,
    }

    return gathered
