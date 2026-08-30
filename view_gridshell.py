# SPDX-License-Identifier: Apache-2.0
"""
Orbit the form-found gridshell, colored by whatever governs it.

Rebuilds the answer stored in `data/gridshell.npz` through the shipped pipeline
and opens it in polyscope. Members are drawn at their designed radius, so a fat
tube is a working member; every scalar the design carries is registered as a
quantity the GUI can switch between.

Run from the repository root:

    uv run python view_gridshell.py            # the form-found answer
    uv run python view_gridshell.py drawn      # the drawn cap it started from
"""

import sys
from pathlib import Path

import numpy as np
import polyscope as ps
from jaxtyping import Float

from normax.config import LoadCaseConfig
from normax.config import RunArguments
from normax.config import read_run_config
from normax.design import Design
from normax.design import DesignProblem
from normax.design import StructuralDesignPipeline
from normax.design import assign_signs
from normax.design import build_design_constraints
from normax.design import compute_mass
from normax.design import create_design
from normax.design import initialize_optimization_parameters
from normax.form_finding import DrawnShapeInitializer
from normax.form_finding import build_form_finder
from normax.form_finding import build_plan_basis
from normax.loads import build_load_cases
from normax.loads import read_polar_plan
from normax.materials import Steel355
from normax.sections import UniformDiameterInitializer
from normax.sections import build_section_catalog
from normax.structures import ShellDescription
from normax.structures import Structure
from normax.structures import build_gridshell_3d
from normax.structures import create_groups_shell
from normax.symmetry import build_height_groups
from normax.symmetry import build_section_groups
from normax.symmetry import find_mirror_nodes
from normax.symmetry import find_rotated_nodes
from normax.tesseract import TesseractAnalyzer
from normax.tesseract import TesseractSizer

CONFIG = Path("examples/gridshell.yaml")
RECORD = Path("data/gridshell.npz")
BASELINE = Path("data/gridshell_spoke3_13.npz")
EDGE_RADIUS = 30.0  # millimeters, the drawn tube radius of every member
SUPPORT_RADIUS = 90.0  # millimeters, one size for every support marker
LOAD_LENGTH = 1500.0  # millimeters the largest force in a case stands
LOAD_RADIUS = 45.0  # millimeters, the largest force's drawn thickness
LOAD_COLORS = ((0.95, 0.35, 0.15), (0.20, 0.55, 0.95), (0.35, 0.80, 0.35))
GHOST_COLOR = (0.65, 0.65, 0.65)
AXES_LENGTH = 6500.0  # millimeters, past the plan radius so the triad clears the shell
AXES_RADIUS = 40.0  # millimeters, the triad's drawn thickness
AXES_COLORS = ((0.90, 0.15, 0.15), (0.15, 0.75, 0.25), (0.20, 0.40, 0.95))
PLANE_COLOR = (0.95, 0.85, 0.30)
PLANE_ALPHA = 0.22


def build_shell_problem(
    route: str = "fdm",
    fold_basis: bool = True,
) -> tuple[DesignProblem, np.ndarray]:
    """
    The gridshell design task the example states, and its drawn start.
    """
    config = read_run_config(RunArguments(CONFIG, route), ShellDescription)
    described = config.structure
    structure = build_gridshell_3d(
        described.num_rings,
        described.num_spokes,
        described.radius,
        described.rise,
        described.oculus,
        described.braced,
    )
    loads = build_load_cases(structure, config.load_cases)
    catalog = build_section_catalog(Steel355(), config.sizing.section_class)
    mirror = find_mirror_nodes(structure, config.form_finding.mirror)
    basis = build_plan_basis(
        structure, mirror if fold_basis else None, config.form_finding.basis
    )
    rotation = find_rotated_nodes(structure, read_polar_plan(structure).num_spokes)
    section_groups = build_section_groups(structure, (mirror, rotation))
    height_groups = build_height_groups(structure, (mirror,))
    form_finder = build_form_finder(
        structure, basis, config.form_finding, height_groups
    )
    pipeline = StructuralDesignPipeline(
        form_finder,
        TesseractAnalyzer(structure, catalog, config.analysis.backend),
        TesseractSizer(structure, catalog, config.sizing.backend),
    )
    groups = create_groups_shell(described)
    guarded = assign_signs(config.constraints, groups, structure.num_edges)
    density_initializer = DrawnShapeInitializer(config.form_finding.density_start)
    density_start = density_initializer(structure, loads.formfinding, basis, guarded)
    constraints = build_design_constraints(config.constraints, guarded, density_start)
    problem = DesignProblem(structure, pipeline, loads, constraints, section_groups)
    initializer = UniformDiameterInitializer(config.analysis.diameter_start)
    diameter_start = initializer(structure)
    start = initialize_optimization_parameters(problem, density_start, diameter_start)

    return problem, np.asarray(start)


def register_design(
    name: str,
    problem: DesignProblem,
    parameters: np.ndarray,
    primary: bool = True,
) -> Design:
    """
    Draw one design, colored by what the code check has to say about it.

    Only the primary design enables a quantity and its colorbar, so the scene
    carries one scale rather than one per registered design.
    """
    design = create_design(problem, parameters)
    xyz = np.asarray(design.shape.xyz, dtype=float)
    edges = np.asarray(problem.structure.edges, dtype=int)
    diameter = np.asarray(design.sizes.sections.diameter, dtype=float)
    utilization = np.asarray(design.sizes.utilization, dtype=float)
    axial = np.asarray(design.forces.axial_force, dtype=float)

    network = ps.register_curve_network(name, xyz, edges, enabled=True)
    # Absolute millimeters, not relative: a relative radius resolves against the
    # scene length scale as it stands when applied, so two designs registered in
    # turn would come out different sizes.
    network.set_radius(EDGE_RADIUS, relative=False)
    if not primary:
        # A flat gray ghost, so the design being read stays the colored one.
        network.set_color(GHOST_COLOR)

    # Every quantity carries its own colorbar, so a color reads as a number
    # without leaving the window.
    bar = {"defined_on": "edges", "onscreen_colorbar_enabled": primary}
    # Fixed to the code limit, so the color says how close to failing a member
    # is rather than how it compares to its neighbors.
    network.add_scalar_quantity(
        "utilization",
        utilization.max(axis=0),
        cmap="viridis",
        vminmax=(0.0, 1.0),
        enabled=primary,
        **bar,
    )
    worst_case = utilization.argmax(axis=0)
    network.add_scalar_quantity(
        "governing case", worst_case, datatype="categorical", cmap="turbo", **bar
    )
    # Symmetric about zero and shared across the cases, so the colormap's
    # midpoint is the sign change and the cases compare directly.
    reach = float(np.abs(axial).max())
    for index in range(axial.shape[0]):
        network.add_scalar_quantity(
            f"axial case {index} [N]",
            axial[index],
            cmap="coolwarm",
            vminmax=(-reach, reach),
            **bar,
        )

    mass = float(compute_mass(design))
    lower = xyz.min(axis=0)
    upper = xyz.max(axis=0)
    print(
        f"{name}: mass {mass:.6f} t, worst utilization {utilization.max():.6f}, "
        f"apex {xyz[:, 2].max():.1f} mm, "
        f"diameter {diameter.min():.1f}-{diameter.max():.1f} mm"
    )
    print(
        f"  {len(xyz)} nodes, {len(edges)} edges; bounds "
        f"x {lower[0]:.0f}..{upper[0]:.0f}  y {lower[1]:.0f}..{upper[1]:.0f}  "
        f"z {lower[2]:.0f}..{upper[2]:.0f}"
    )

    return design


def register_load_case(
    label: str,
    applied: Float[np.ndarray, "nodes 3"],
    xyz: Float[np.ndarray, "nodes 3"],
    color: tuple[float, float, float],
    enabled: bool,
) -> None:
    """
    One load case as headless vertical lines, sized by the force each carries.

    Parameters
    ----------
    label :
        What the case is called in the panel.
    applied :
        Force applied at every node in this case.
    xyz :
        The geometry the case is carried on, read for where each force acts.
    color :
        Flat color the case is drawn in.
    enabled :
        Whether the case is shown on opening, or waits in the panel.

    Notes
    -----
    A line stands above the node the force acts at rather than hanging below it,
    so no line disappears inside the shell. Length and thickness both scale with
    magnitude against the largest force in the case, so a case reads as a
    pattern and two cases of the same total look alike.
    """
    magnitude = np.linalg.norm(np.asarray(applied, dtype=float), axis=1)
    loaded = np.flatnonzero(magnitude > 0.0)
    if loaded.size == 0:
        print(f"{label}: no node carries a force, nothing drawn")
        return

    share = magnitude[loaded] / magnitude[loaded].max()
    feet = np.asarray(xyz, dtype=float)[loaded]
    heads = feet + np.column_stack(
        [np.zeros_like(share), np.zeros_like(share), share * LOAD_LENGTH]
    )
    nodes = np.vstack([feet, heads])
    count = loaded.size
    edges = np.column_stack([np.arange(count), np.arange(count) + count])

    network = ps.register_curve_network(label, nodes, edges, enabled=enabled)
    network.set_radius(LOAD_RADIUS, relative=False)
    network.set_color(color)
    # Registered to drive the thickness, not to be read: no colorbar, and the
    # scene's one scale stays with the design.
    network.add_scalar_quantity(
        "force [N]",
        share * float(magnitude[loaded].max()),
        defined_on="edges",
        enabled=False,
        onscreen_colorbar_enabled=False,
    )
    network.set_edge_radius_quantity("force [N]", autoscale=True)
    print(
        f"{label}: {count} forces, "
        f"{magnitude[loaded].min():.2f}..{magnitude[loaded].max():.2f} N, "
        f"total {magnitude[loaded].sum():.1f} N"
    )


def register_axes() -> None:
    """
    A triad at the plan's center, red along x, green along y, blue along z.

    Notes
    -----
    Polyscope draws no world frame of its own, so the triad is three segments
    carrying a per-edge color. It stands past the plan radius, where nothing of
    the structure can hide it.
    """
    nodes = np.zeros((4, 3))
    nodes[1, 0] = AXES_LENGTH
    nodes[2, 1] = AXES_LENGTH
    nodes[3, 2] = AXES_LENGTH
    edges = np.array([[0, 1], [0, 2], [0, 3]])

    network = ps.register_curve_network("axes", nodes, edges, enabled=True)
    network.set_radius(AXES_RADIUS, relative=False)
    network.add_color_quantity(
        "axis", np.asarray(AXES_COLORS), defined_on="edges", enabled=True
    )
    print(f"axes: x red, y green, z blue, {AXES_LENGTH:.0f} mm from the plan center")


def register_mirror_plane(axis: str, height: float) -> None:
    """
    The plane the shape fold reflects about, drawn so it can be sighted along.

    Parameters
    ----------
    axis :
        Axis the plane stands normal to, as the run description names it.
    height :
        How far up the plane is drawn, so it covers the structure it folds.

    Notes
    -----
    Translucent and registered last, since it is a reading aid rather than part
    of the structure. `mirror: y` reflects the y coordinate, so its plane is
    y = 0 -- the xz plane -- and not the plane the axis is named after.
    """
    reach = AXES_LENGTH
    if axis == "y":
        corners = [
            [-reach, 0.0, 0.0],
            [reach, 0.0, 0.0],
            [reach, 0.0, height],
            [-reach, 0.0, height],
        ]
    else:
        corners = [
            [0.0, -reach, 0.0],
            [0.0, reach, 0.0],
            [0.0, reach, height],
            [0.0, -reach, height],
        ]
    vertices = np.asarray(corners)
    faces = np.array([[0, 1, 2, 3]])

    mesh = ps.register_surface_mesh(
        f"mirror plane ({axis})",
        vertices,
        faces,
        enabled=True,
        color=PLANE_COLOR,
        transparency=PLANE_ALPHA,
        back_face_policy="identical",
    )
    mesh.set_smooth_shade(False)
    plane = "xz" if axis == "y" else "yz"
    print(f"mirror plane: reflects {axis}, so the plane drawn is {plane} at {axis} = 0")


def register_supports(problem: DesignProblem) -> None:
    """
    The support markers, once for every design: no design moves them.
    """
    register_supports_at(problem.structure)


def register_supports_at(structure: Structure) -> None:
    """
    One marker per support, at one radius.
    """
    xyz = np.asarray(structure.nodes, dtype=float)
    supports = np.asarray(structure.supports, dtype=int)
    cloud = ps.register_point_cloud("supports", xyz[supports], enabled=True)
    cloud.set_radius(SUPPORT_RADIUS, relative=False)
    print(f"supports: {len(supports)} markers at one radius of {SUPPORT_RADIUS:.0f} mm")


def read_sector_centers(word: str) -> tuple[int, int]:
    """
    The two spokes a pair of drift cases is centered on, as `first,second`.
    """
    first, second = (int(part) for part in word.split(","))

    return first, second


def prototype_loads(proposed: tuple[int, int]) -> None:
    """
    Draw a proposed drift pair on the cap as drawn, beside the shipped pair.

    Parameters
    ----------
    proposed :
        Spokes the two drift cases would be centered on.

    Notes
    -----
    Nothing is optimized and nothing is sized: this reads the run description
    for its structure and its load patterns alone, so a distribution can be
    approved before a search is spent on it.
    """
    config = read_run_config(RunArguments(CONFIG, route), ShellDescription)
    described = config.structure
    structure = build_gridshell_3d(
        described.num_rings,
        described.num_spokes,
        described.radius,
        described.rise,
        described.oculus,
        described.braced,
    )
    xyz = np.asarray(structure.nodes, dtype=float)
    drawn = ps.register_curve_network(
        "cap as drawn", xyz, np.asarray(structure.edges, dtype=int), enabled=True
    )
    drawn.set_radius(EDGE_RADIUS, relative=False)
    drawn.set_color(GHOST_COLOR)

    shipped = tuple(
        int(case.options["center"])
        for case in config.load_cases
        if case.name == "sector"
    )
    uniform = [case for case in config.load_cases if case.name != "sector"]
    for case in uniform:
        applied = build_load_cases(structure, [case]).analysis[0]
        register_load_case(
            f"{case.name} (both)", np.asarray(applied), xyz, LOAD_COLORS[2], False
        )

    for tag, centers, color, enabled in (
        ("proposed", proposed, LOAD_COLORS[0], True),
        ("shipped", shipped, LOAD_COLORS[1], False),
    ):
        for order, center in enumerate(centers):
            described_case = LoadCaseConfig(
                "sector", 1.0e-3, {"center": center, "spokes": 3, "factor": 0.5}
            )
            applied = build_load_cases(structure, [described_case]).analysis[0]
            register_load_case(
                f"{tag} drift {order + 1}: spoke {center}",
                np.asarray(applied),
                xyz,
                color,
                enabled and order == 0,
            )

    register_supports_at(structure)
    register_axes()
    register_mirror_plane(config.form_finding.mirror, float(xyz[:, 2].max()))


def main() -> None:
    """
    Open the shell in an orbitable window.
    """
    wanted = sys.argv[1] if len(sys.argv) > 1 else "found"

    if wanted == "loads":
        given = sys.argv[2] if len(sys.argv) > 2 else "4,12"
        ps.init()
        ps.set_up_dir("z_up")
        ps.set_front_dir("neg_y_front")
        ps.set_ground_plane_mode("shadow_only")
        prototype_loads(read_sector_centers(given))
        ps.reset_camera_to_home_view()
        ps.show()
        return

    if wanted == "compare":
        ghost = Path(sys.argv[2]) if len(sys.argv) > 2 else BASELINE
        ps.init()
        ps.set_up_dir("z_up")
        ps.set_front_dir("neg_y_front")
        ps.set_ground_plane_mode("shadow_only")
        problem, _ = build_shell_problem()
        design = register_design("found", problem, np.load(RECORD)["parameters"])
        # The form-finding case is the uniform one and did not change, so the
        # ghost's geometry is exactly what its own run reached. Its utilization
        # is not: that is recomputed against the load cases in force now.
        register_design(
            ghost.stem, problem, np.load(ghost)["parameters"], primary=False
        )
        register_supports(problem)
        register_axes()
        config = read_run_config(RunArguments(CONFIG, "fdm"), ShellDescription)
        height = float(np.asarray(design.shape.xyz)[:, 2].max())
        register_mirror_plane(config.form_finding.mirror, height)
        ps.reset_camera_to_home_view()
        ps.show()
        return

    problem, start = build_shell_problem()

    ps.init()
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")

    design = register_design("found", problem, np.load(RECORD)["parameters"])
    if wanted == "drawn":
        register_design("drawn", problem, start, primary=False)
    register_supports(problem)

    xyz = np.asarray(design.shape.xyz, dtype=float)
    applied = np.asarray(problem.loads.analysis, dtype=float)
    for index in range(applied.shape[0]):
        register_load_case(index, applied[index], xyz, enabled=index == 0)

    register_axes()
    config = read_run_config(RunArguments(CONFIG, "fdm"), ShellDescription)
    register_mirror_plane(config.form_finding.mirror, float(xyz[:, 2].max()))

    # Frame the scene on what was just registered, or a stale extent can leave
    # the structure outside the view.
    ps.reset_camera_to_home_view()
    ps.show()


if __name__ == "__main__":
    main()
