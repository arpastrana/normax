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
Does OpenSees' Direct Differentiation Method reach a nodal coordinate?

T1 hands T2 a geometry, so the analysis backend has to return `∂N/∂xyz`. DDM
parametrises material and section properties by design, and whether a nodal
coordinate can be registered at all decides which of three architectures T2
gets. This script answers that empirically rather than from the documentation,
in five passes:

    matrix    every (element, dimension, parameter) pair, DDM vs central
              differences, classified OK / WRONG / MISSING
    coord3d   the 3D coordinate result under step-size refinement, to
              establish which of the two disagreeing numbers to believe
    forces    `∂N/∂xyz` rather than `∂u/∂xyz` — the quantity T3 actually
              consumes, and the one that decides the element
    printb    whether the pseudo-load vector is reachable without a solve,
              which would buy reverse mode over OpenSees at O(1)
    timing    what the DDM sweep costs against the finite-difference fallback

Each pass runs in its own subprocess: OpenSees exits hard, without a traceback,
after a few hundred model rebuilds in one interpreter, and a pass that dies
must not take the others with it.

Requires the `spike` extra:
    uv sync --extra spike
    uv run python experiments/07_opensees_ddm_spike.py [pass]

with `pass` one of matrix, coord3d, forces, printb, timing, or omitted for all.
"""

import subprocess
import sys
import time
from collections.abc import Callable
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import openseespy.opensees as ops

from normax.reporting import ColumnSpec
from normax.reporting import ReportWriter

E0 = 210e9
A0 = 7.37e-3
I0 = 4.96e-5
G0 = 80.7692e9
J0 = 9.92e-5
LOAD6 = [-100e3, -8e3, -10e3, 0.0, 0.0, 0.0]
TOL = 1e-5

# Relative step every central difference in this file is taken at.
STEP = 1e-6

ELEMENTS = ("truss", "elasticBeamColumn", "dispBeamColumn", "forceBeamColumn")


class FrameGeometry(NamedTuple):
    """
    One test frame: where its nodes are, what holds it, and what loads it.

    Attributes
    ----------
    coords :
        Coordinates of every node, by tag.
    fixed :
        Tags of the fully restrained nodes.
    loaded :
        Tag of the node the load is applied at.
    bars :
        Node pairs every element spans, in tag order.
    """

    coords: dict[int, list[float]]
    fixed: tuple[int, ...]
    loaded: int
    bars: tuple[tuple[int, int], ...]


class ModelSpec(NamedTuple):
    """
    Everything one assembly needs, so that a perturbed copy is one `_replace`.

    Attributes
    ----------
    element :
        Element formulation under test.
    ndm :
        Dimensions the model is built in.
    coords :
        Coordinates of every node, by tag.
    properties :
        Section properties of every element, by tag. Each element carries its
        own, so a parameter registered on element 1 can be central-differenced
        by perturbing element 1 alone.
    """

    element: str
    ndm: int
    coords: dict[int, list[float]]
    properties: dict[int, dict[str, float]]

    @property
    def ndf(self) -> int:
        """
        Degrees of freedom per node, which the formulation decides.
        """
        if self.element == "truss":
            return self.ndm

        return 3 if self.ndm == 2 else 6

    @property
    def inertia(self) -> str:
        """
        Name the second moment is registered under, `I` in 2D and `Iz` in 3D.

        The wrong name binds to nothing and reads as a missing derivative.
        """
        return "I" if self.ndm == 2 else "Iz"


class ParameterSpec(NamedTuple):
    """
    One quantity registered with the direct differentiation method.

    Attributes
    ----------
    kind :
        Either `element`, for a section property, or `node`, for a coordinate.
    tag :
        Tag of the element or node the quantity belongs to.
    name :
        Property name, or `coord` for a coordinate.
    direction :
        Global axis of a coordinate, one-based, and unused otherwise.
    """

    kind: str
    tag: int
    name: str
    direction: int = 0

    @property
    def command(self) -> tuple[str | int, ...]:
        """
        The arguments `ops.parameter` takes after its tag.
        """
        if self.kind == "node":
            arguments = (self.kind, self.tag, self.name, self.direction)
        else:
            arguments = (self.kind, self.tag, self.name)

        return arguments

    @property
    def label(self) -> str:
        """
        How the quantity reads in a table.
        """
        return f"coord{self.direction}" if self.kind == "node" else self.name


class ParameterVerdict(NamedTuple):
    """
    How one registered parameter fared against its difference quotient.

    Attributes
    ----------
    status :
        One of OK, WRONG, MISSING, or NO-SIGNAL.
    informative :
        Degrees of freedom whose difference quotient rose above the noise.
    detail :
        The worst disagreeing degree of freedom, spelled out.
    """

    status: str
    informative: int
    detail: str


def geometry(element: str, ndm: int) -> FrameGeometry:
    """
    Coordinates, restrained nodes, loaded node and connectivity per case.

    Trusses need a stable pin-jointed assembly, so a triangle in 2D and a
    tripod in 3D. Beams use a kinked cantilever whose interior node is offset
    transversely, so that perturbing a coordinate reorients the members
    instead of only changing a length.
    """
    if element == "truss" and ndm == 2:
        coords = {1: [0.0, 0.0], 2: [4.0, 0.0], 3: [2.0, -1.5]}
        bars = ((1, 3), (2, 3))
        frame = FrameGeometry(coords, (1, 2), 3, bars)

        return frame

    if element == "truss":
        coords = {
            1: [0.0, 0.0, 0.0],
            2: [4.0, 0.0, 0.0],
            3: [2.0, 3.0, 0.0],
            4: [2.0, 1.0, -2.0],
        }
        bars = ((1, 4), (2, 4), (3, 4))
        frame = FrameGeometry(coords, (1, 2, 3), 4, bars)

        return frame

    coords = {
        1: [0.0, 0.0, 0.0][:ndm],
        2: [2.0, 0.5, 0.0][:ndm],
        3: [4.0, 0.0, 0.0][:ndm],
    }
    bars = ((1, 2), (2, 3))
    frame = FrameGeometry(coords, (1,), 3, bars)

    return frame


def model_spec(element: str, ndm: int) -> ModelSpec:
    """
    The unperturbed model of one case, every element carrying its own section.

    `G` and `J` are held independent of `E` and `I` on purpose: a shared
    expression would make the difference quotient perturb quantities the DDM
    parameter does not.
    """
    frame = geometry(element, ndm)
    section = {"E": E0, "A": A0, "Iz": I0, "Iy": I0}
    properties = {tag: dict(section) for tag in range(1, len(frame.bars) + 1)}
    spec = ModelSpec(element, ndm, frame.coords, properties)

    return spec


def build_model(spec: ModelSpec, parameters: Sequence[ParameterSpec] = ()) -> None:
    """
    Assemble and solve one model, optionally with DDM parameters registered.
    """
    frame = geometry(spec.element, spec.ndm)
    ndf = spec.ndf

    ops.wipe()
    ops.model("basic", "-ndm", spec.ndm, "-ndf", ndf)
    for tag, xyz in spec.coords.items():
        ops.node(tag, *xyz)
    for tag in frame.fixed:
        ops.fix(tag, *([1] * ndf))

    if spec.element == "truss":
        for tag, (start, end) in enumerate(frame.bars, 1):
            ops.uniaxialMaterial("Elastic", tag, spec.properties[tag]["E"])
            ops.element("truss", tag, start, end, spec.properties[tag]["A"], tag)
    else:
        if spec.ndm == 2:
            ops.geomTransf("Linear", 1)
        else:
            ops.geomTransf("Linear", 1, 0.0, 0.0, 1.0)
        for tag, (start, end) in enumerate(frame.bars, 1):
            section = spec.properties[tag]
            if spec.element == "elasticBeamColumn":
                if spec.ndm == 2:
                    ops.element(
                        "elasticBeamColumn",
                        tag,
                        start,
                        end,
                        section["A"],
                        section["E"],
                        section["Iz"],
                        1,
                    )
                else:
                    ops.element(
                        "elasticBeamColumn",
                        tag,
                        start,
                        end,
                        section["A"],
                        section["E"],
                        G0,
                        J0,
                        section["Iy"],
                        section["Iz"],
                        1,
                    )
                continue
            if spec.ndm == 2:
                ops.section("Elastic", tag, section["E"], section["A"], section["Iz"])
            else:
                ops.section(
                    "Elastic",
                    tag,
                    section["E"],
                    section["A"],
                    section["Iz"],
                    section["Iy"],
                    G0,
                    J0,
                )
            forced = spec.element == "forceBeamColumn"
            rule = "Lobatto" if forced else "Legendre"
            ops.beamIntegration(rule, tag, tag, 5 if forced else 3)
            ops.element(spec.element, tag, start, end, 1, tag)

    ops.timeSeries("Constant", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(frame.loaded, *LOAD6[:ndf])

    for tag, parameter in enumerate(parameters, 1):
        ops.parameter(tag, *parameter.command)

    ops.system("FullGeneral")
    ops.numberer("Plain")
    ops.constraints("Plain")
    ops.integrator("LoadControl", 1.0)
    ops.algorithm("Linear")
    ops.analysis("Static")
    if parameters:
        ops.sensitivityAlgorithm("-computeAtEachStep")
    ops.analyze(1)


def property_shifted(spec: ModelSpec, name: str, step: float) -> ModelSpec:
    """
    The same model with one property of element 1 moved by a step.
    """
    properties = {tag: dict(values) for tag, values in spec.properties.items()}
    properties[1][name] += step
    moved = spec._replace(properties=properties)

    return moved


def coordinate_shifted(spec: ModelSpec, parameter: ParameterSpec, step) -> ModelSpec:
    """
    The same model with one coordinate of one node moved by a step.
    """
    coords = {tag: list(xyz) for tag, xyz in spec.coords.items()}
    coords[parameter.tag][parameter.direction - 1] += step
    moved = spec._replace(coords=coords)

    return moved


def displacements_of(node: int, ndf: int) -> list[float]:
    """
    Every displacement of one node of the model standing in the interpreter.
    """
    displacements = [ops.nodeDisp(node, dof) for dof in range(1, ndf + 1)]

    return displacements


def difference_quotient(
    spec: ModelSpec,
    parameter: ParameterSpec,
    step: float,
    read: Callable[[], list[float]],
) -> list[float]:
    """
    Central difference of whatever `read` returns, in one parameter.
    """
    shifted = []
    for sign in (+1, -1):
        if parameter.kind == "node":
            moved = coordinate_shifted(spec, parameter, sign * step)
        else:
            name = "Iz" if parameter.name == "I" else parameter.name
            moved = property_shifted(spec, name, sign * step)
        build_model(moved)
        shifted.append(read())

    quotient = [(high - low) / (2 * step) for high, low in zip(*shifted)]

    return quotient


def base_step(spec: ModelSpec, parameter: ParameterSpec) -> float:
    """
    The quantity a relative step is taken against, and the step itself.
    """
    if parameter.kind == "node":
        return max(abs(spec.coords[parameter.tag][parameter.direction - 1]), 1.0)

    name = "Iz" if parameter.name == "I" else parameter.name

    return spec.properties[1][name]


def classify(ddm: Sequence[float], quotient: Sequence[float], floor: float):
    """
    Compare one parameter's sensitivity vector against its difference quotient.

    A zero is only evidence of a missing derivative where the difference
    quotient is above the noise floor, because several sensitivities are
    legitimately zero — a transverse response does not depend on the area.
    """
    status, worst, detail, seen = "NO-SIGNAL", 0.0, "", 0
    for dof, (found, reference) in enumerate(zip(ddm, quotient), 1):
        if abs(reference) <= floor:
            continue
        seen += 1
        if status == "NO-SIGNAL":
            status = "OK"
        if found == 0.0:
            missing = f"dof{dof}: cd={reference:+.5e} ddm=0"
            verdict = ParameterVerdict("MISSING", seen, missing)

            return verdict
        error = abs(found - reference) / abs(reference)
        if error > worst:
            worst = error
            detail = f"dof{dof}: ddm={found:+.6e} cd={reference:+.6e} rel={error:.2e}"
        if error > TOL:
            status = "WRONG"

    verdict = ParameterVerdict(status, seen, detail)

    return verdict


def capability_matrix(report: ReportWriter) -> None:
    """
    Every (element, dimension, parameter) pair against central differences.
    """
    report.write_banner("DDM capability matrix -- sensNodeDisp vs central differences")

    matrix_columns = (
        ColumnSpec("parameter"),
        ColumnSpec("status"),
        ColumnSpec("signal", align="<"),
        ColumnSpec("worst", align="<"),
    )
    summary_columns = (
        ColumnSpec("case", align="<"),
        ColumnSpec("parameter"),
        ColumnSpec("status"),
    )
    summary = []
    for element in ELEMENTS:
        for ndm in (2, 3):
            spec = model_spec(element, ndm)
            frame = geometry(element, ndm)
            names = ["E", "A"] if element == "truss" else ["E", "A", spec.inertia]
            parameters = [ParameterSpec("element", 1, name) for name in names]
            parameters += [
                ParameterSpec("node", frame.loaded, "coord", axis)
                for axis in range(1, ndm + 1)
            ]

            build_model(spec, parameters)
            reference = displacements_of(frame.loaded, spec.ndf)
            ddm = {}
            for index in range(1, len(parameters) + 1):
                ddm[index] = [
                    ops.sensNodeDisp(frame.loaded, dof, index)
                    for dof in range(1, spec.ndf + 1)
                ]
            largest = max(abs(value) for value in reference)

            rows = []
            for index, parameter in enumerate(parameters, 1):
                base = base_step(spec, parameter)
                step = base * STEP

                def read_displacements(spec=spec, frame=frame):
                    return displacements_of(frame.loaded, spec.ndf)

                quotient = difference_quotient(
                    spec, parameter, step, read_displacements
                )
                floor = STEP * largest / base
                verdict = classify(ddm[index], quotient, floor)
                signal = f"{verdict.informative} informative dof"
                row = (parameter.label, verdict.status, signal, verdict.detail)
                rows.append(row)
                case = f"{element} ndm={ndm}"
                summary.append((case, parameter.label, verdict.status))

            report.write_heading(f"{element}, ndm={ndm}, ndf={spec.ndf}")
            report.write_table(matrix_columns, rows)

    report.write_heading("SUMMARY")
    report.write_table(summary_columns, summary)


def coord_3d_evidence(report: ReportWriter) -> None:
    """
    Which number to believe where 3D DDM and central differences disagree.

    A difference quotient that is stable across four decades of step size is
    not a truncation artefact, so a DDM value that disagrees with it is wrong.
    The axis-aligned single bar is included because it is the one 3D case with
    a closed form.
    """
    report.write_banner("3D coordinate sensitivity -- step-size refinement")

    length, axial = 4.0, -100e3

    def bar(x2, registered=False):
        ops.wipe()
        ops.model("basic", "-ndm", 3, "-ndf", 3)
        ops.node(1, 0.0, 0.0, 0.0)
        ops.node(2, x2, 0.0, 0.0)
        ops.fix(1, 1, 1, 1)
        ops.fix(2, 0, 1, 1)
        ops.uniaxialMaterial("Elastic", 1, E0)
        ops.element("truss", 1, 1, 2, A0, 1)
        ops.timeSeries("Constant", 1)
        ops.pattern("Plain", 1, 1)
        ops.load(2, axial, 0.0, 0.0)
        if registered:
            ops.parameter(1, "node", 2, "coord", 1)
        ops.system("FullGeneral")
        ops.numberer("Plain")
        ops.constraints("Plain")
        ops.integrator("LoadControl", 1.0)
        ops.algorithm("Linear")
        ops.analysis("Static")
        if registered:
            ops.sensitivityAlgorithm("-computeAtEachStep")
        ops.analyze(1)

    bar(length, registered=True)
    displacement = ops.nodeDisp(2, 1)
    found = ops.sensNodeDisp(2, 1, 1)
    exact = displacement / length

    entries = (
        ("ddm", f"{found:+.10e}"),
        ("exact", f"{exact:+.10e}, ratio {found / exact:.8f}"),
    )

    report.write_heading("single bar along global x, u = P L / (E A) so du/dx2 = u / L")
    report.write_entries(entries)

    spec = model_spec("truss", 3)
    frame = geometry("truss", 3)

    report.write_heading("tripod, bars not aligned with any global axis")
    rows = []
    for direction in (1, 2, 3):
        parameter = ParameterSpec("node", frame.loaded, "coord", direction)
        build_model(spec, [parameter])
        found = [ops.sensNodeDisp(frame.loaded, dof, 1) for dof in (1, 2, 3)]
        rows.append((f"coord{direction}", "ddm", *found))

        def read_displacements(frame=frame):
            displacements = [ops.nodeDisp(frame.loaded, dof) for dof in (1, 2, 3)]

            return displacements

        for relative in (1e-4, 1e-6, 1e-8):
            step = relative * 4.0
            quotient = difference_quotient(spec, parameter, step, read_displacements)
            rows.append((f"coord{direction}", f"cd h={relative:.0e}", *quotient))

    columns = (
        ColumnSpec("parameter"),
        ColumnSpec("source", align="<"),
        ColumnSpec("d/dx", "+.8e"),
        ColumnSpec("d/dy", "+.8e"),
        ColumnSpec("d/dz", "+.8e"),
    )

    report.write_table(columns, rows)


def force_sensitivity(report: ReportWriter) -> None:
    """
    `∂N/∂xyz`, which is what T3 consumes, rather than `∂u/∂xyz`.

    Section forces are one step further down the chain than displacements, and
    the two beam formulations do not agree there even though both get
    displacements right. `sensSectionForce` returns the section vector starting
    at the requested dof, so element 0 is the value asked for; indexing `dof-1`
    quietly returns a neighbour. Passing an unregistered parameter tag segfaults
    the process outright.
    """
    report.write_banner("section-force sensitivity to a nodal coordinate -- 2D")

    for element in ("forceBeamColumn", "dispBeamColumn"):
        spec = model_spec(element, 2)
        points = 5 if element == "forceBeamColumn" else 3
        parameters = (
            ParameterSpec("node", 3, "coord", 1),
            ParameterSpec("node", 3, "coord", 2),
            ParameterSpec("node", 2, "coord", 2),
        )
        build_model(spec, parameters)

        def section_forces():
            forces = {}
            for tag in (1, 2):
                for station in (1, points):
                    response = ops.eleResponse(tag, "section", station, "force")
                    forces[(tag, station)] = response

            return forces

        worst = 0.0
        rows = []
        for index, parameter in enumerate(parameters, 1):
            place = spec.coords[parameter.tag][parameter.direction - 1]
            step = STEP * max(abs(place), 1.0)
            shifted = []
            for sign in (+1, -1):
                moved = coordinate_shifted(spec, parameter, sign * step)
                build_model(moved)
                shifted.append(section_forces())
            build_model(spec, parameters)

            for tag in (1, 2):
                for station in (1, points):
                    for dof in (1, 2):
                        found = ops.sensSectionForce(tag, station, dof, index)
                        if isinstance(found, list):
                            found = found[0]
                        low = shifted[1][(tag, station)]
                        high = shifted[0][(tag, station)]
                        quotient = (high[dof - 1] - low[dof - 1]) / (2 * step)
                        if abs(quotient) <= 1e3:
                            continue
                        error = abs(found - quotient) / abs(quotient)
                        worst = max(worst, error)
                        name = "N" if dof == 1 else "M"
                        where = (
                            f"n{parameter.tag} c{parameter.direction}"
                            f" e{tag} s{station} d{name}"
                        )
                        rows.append((where, found, quotient, error))

        columns = (
            ColumnSpec("where", align="<"),
            ColumnSpec("ddm", "+.6e"),
            ColumnSpec("central diff", "+.6e"),
            ColumnSpec("relative", ".2e"),
        )
        verdict = "AVAILABLE" if worst < 1e-5 else "NOT RELIABLE"
        entries = (("worst", f"{worst:.3e}, dN/dxyz {verdict}"),)

        report.write_heading(element)
        report.write_table(columns, rows)
        report.write_entries(entries)


def printb_semantics(report: ReportWriter) -> None:
    """
    Is the pseudo-load vector reachable, in any sensitivity mode?

    Reverse mode over OpenSees would need P_i = ∂f/∂θ_i − (∂K/∂θ_i) u without
    paying for a solve per parameter. Reconstructing it as K (du/dθ_i) proves
    what the vector should look like, but presupposes the very solve it would
    replace.
    """
    report.write_banner("pseudo-load reachability -- what printB returns")

    num_elements = 6
    coords = {
        tag + 1: [4.0 * tag / num_elements, 0.0] for tag in range(num_elements + 1)
    }

    def cantilever(parameters=(), mode=None):
        ops.wipe()
        ops.model("basic", "-ndm", 2, "-ndf", 3)
        for tag, xy in coords.items():
            ops.node(tag, *xy)
        ops.fix(1, 1, 1, 1)
        ops.geomTransf("Linear", 1)
        for tag in range(1, num_elements + 1):
            ops.section("Elastic", tag, E0, A0, I0)
            ops.beamIntegration("Legendre", tag, tag, 3)
            ops.element("dispBeamColumn", tag, tag, tag + 1, 1, tag)
        ops.timeSeries("Constant", 1)
        ops.pattern("Plain", 1, 1)
        for tag in range(2, num_elements + 2):
            ops.load(tag, 0.0, -20e3, 0.0)
        for tag, parameter in enumerate(parameters, 1):
            ops.parameter(tag, *parameter.command)
        ops.system("FullGeneral")
        ops.numberer("Plain")
        ops.constraints("Plain")
        ops.integrator("LoadControl", 1.0)
        ops.algorithm("Linear")
        ops.analysis("Static")
        if parameters and mode:
            ops.sensitivityAlgorithm(mode)
        ops.analyze(1)
        if parameters and mode == "-computeByCommand":
            ops.computeGradients()

    ndof = 3 * (num_elements + 1) - 3
    for mode in (None, "-computeAtEachStep", "-computeByCommand"):
        parameters = () if mode is None else (ParameterSpec("element", 1, "E"),)
        cantilever(parameters, mode)
        stiffness = np.array(ops.printA("-ret")).reshape(ndof, ndof)
        rhs = np.array(ops.printB("-ret"))
        moved = [
            ops.nodeDisp(tag, dof)
            for tag in range(2, num_elements + 2)
            for dof in (1, 2, 3)
        ]
        displacement = np.array(moved)
        applied = stiffness @ displacement

        entries = [
            ("|B|", f"{np.linalg.norm(rhs):.5e}"),
            ("|K u|", f"{np.linalg.norm(applied):.5e}, the applied load"),
        ]
        if parameters:
            slopes = [
                ops.sensNodeDisp(tag, dof, 1)
                for tag in range(2, num_elements + 2)
                for dof in (1, 2, 3)
            ]
            sensitivity = np.array(slopes)
            pseudo = stiffness @ sensitivity
            norm = np.linalg.norm(pseudo)
            gap = np.linalg.norm(rhs - pseudo) / norm
            entries.append(("|P_1|", f"{norm:.5e}, |B-P_1|/|P_1| {gap:.3e}"))

        report.write_heading(f"mode {mode}")
        report.write_entries(entries)


def cost_scaling(report: ReportWriter) -> None:
    """
    What the DDM sweep costs, against the finite-difference fallback.

    The fallback is one solve per parameter plus one; DDM reuses a single
    factorization and pays a back-substitution per parameter. The ratio is the
    number the P5 scaling plot needs.
    """
    report.write_banner("cost scaling -- DDM sweep vs finite differences")

    columns = (
        ColumnSpec("n_params"),
        ColumnSpec("DDM total [ms]", ".2f"),
        ColumnSpec("sens only [ms]", ".2f"),
        ColumnSpec("per param [ms]", ".4f"),
        ColumnSpec("FD equiv [ms]", ".2f"),
        ColumnSpec("speedup", ".1f"),
    )

    def arch(num_elements, num_parameters=0, solve=True):
        ops.wipe()
        ops.model("basic", "-ndm", 2, "-ndf", 3)
        for index in range(num_elements + 1):
            x = 40.0 * index / num_elements
            ops.node(index + 1, x, 8.0 * 4 * x * (40.0 - x) / 1600.0)
        ops.fix(1, 1, 1, 1)
        ops.fix(num_elements + 1, 1, 1, 1)
        ops.geomTransf("Linear", 1)
        for tag in range(1, num_elements + 1):
            ops.section("Elastic", tag, E0, A0, I0)
            ops.beamIntegration("Legendre", tag, tag, 3)
            ops.element("dispBeamColumn", tag, tag, tag + 1, 1, tag)
        ops.timeSeries("Constant", 1)
        ops.pattern("Plain", 1, 1)
        for tag in range(2, num_elements + 1):
            ops.load(tag, 0.0, -20e3, 0.0)
        parameters = [
            ParameterSpec("element", tag, ["E", "A", "Iz"][tag % 3])
            for tag in range(1, num_elements + 1)
        ]
        parameters += [
            ParameterSpec("node", tag, "coord", 2) for tag in range(2, num_elements + 1)
        ]
        for tag, parameter in enumerate(parameters[:num_parameters], 1):
            ops.parameter(tag, *parameter.command)
        ops.system("FullGeneral")
        ops.numberer("Plain")
        ops.constraints("Plain")
        ops.integrator("LoadControl", 1.0)
        ops.algorithm("Linear")
        ops.analysis("Static")
        if num_parameters:
            ops.sensitivityAlgorithm("-computeAtEachStep")
        if solve:
            ops.analyze(1)

    def fastest(call, repeats=3):
        best = float("inf")
        for _ in range(repeats):
            start = time.perf_counter()
            call()
            best = min(best, time.perf_counter() - start)

        return best

    for num_elements in (20, 100, 400):
        assembly = fastest(lambda n=num_elements: arch(n, solve=False))
        plain = fastest(lambda n=num_elements: arch(n))
        ndof = 3 * (num_elements + 1) - 6

        entries = (
            ("assemble", f"{assembly * 1e3:.2f} ms"),
            ("solve", f"{(plain - assembly) * 1e3:.3f} ms"),
        )

        report.write_heading(f"nel={num_elements}, ndof={ndof}")
        report.write_entries(entries)

        rows = []
        for count in (10, 50, 2 * num_elements - 2):
            total = fastest(lambda n=num_elements, k=count: arch(n, k))
            sensitivity = total - plain
            fallback = assembly + (count + 1) * plain
            per_param = sensitivity / count * 1e3
            speedup = fallback / total
            row = (
                count,
                total * 1e3,
                sensitivity * 1e3,
                per_param,
                fallback * 1e3,
                speedup,
            )
            rows.append(row)

        report.write_table(columns, rows)


PASSES = {
    "matrix": capability_matrix,
    "coord3d": coord_3d_evidence,
    "forces": force_sensitivity,
    "printb": printb_semantics,
    "timing": cost_scaling,
}


def main(verbose: bool = True) -> None:
    """
    Run one pass, or fan every pass out into its own subprocess.
    """
    report = ReportWriter(verbose)

    if len(sys.argv) > 1:
        PASSES[sys.argv[1]](report)
        return

    for name in PASSES:
        report.write_heading("#" * 78)
        report.write_line(f"# {name}")
        report.write_line("#" * 78)
        sys.stdout.flush()
        done = subprocess.run([sys.executable, __file__, name], check=False)
        if done.returncode != 0:
            report.write_entries(
                (("pass", f"{name!r} exited {done.returncode} -- OpenSees died"),)
            )


if __name__ == "__main__":
    main()
