# SPDX-License-Identifier: Apache-2.0
"""
Whether this install can open the interactive viewer.

Asked of the import system rather than of a failed import, so a package that
is present but raises on import is a fault to see and not a viewer silently
declared absent.
"""

import importlib.util

# The viewer's packages, neither of them a dependency of normax.
VIEWER_PACKAGES = ("smax", "vix")


def find_viewer() -> bool:
    """
    Whether the packages the interactive viewer draws with are installed.

    Returns
    -------
    found :
        True where every one of the viewer's packages can be imported.
    """
    specs = [importlib.util.find_spec(name) for name in VIEWER_PACKAGES]

    return all(spec is not None for spec in specs)
