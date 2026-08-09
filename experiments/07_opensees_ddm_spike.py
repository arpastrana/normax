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

import numpy as np
import openseespy.opensees as ops

E0 = 210e9
A0 = 7.37e-3
I0 = 4.96e-5
G0 = 80.7692e9
J0 = 9.92e-5
LOAD6 = [-100e3, -8e3, -10e3, 0.0, 0.0, 0.0]
TOL = 1e-5


def geometry(element, ndm):
    """
    Coordinates, restrained nodes, loaded node and connectivity per case.

    Trusses need a stable pin-jointed assembly, so a triangle in 2D and a
    tripod in 3D. Beams use a kinked cantilever whose interior node is offset
    transversely, so that perturbing a coordinate reorients the members
    instead of only changing a length.
    """
    if element == "truss":
        if ndm == 2:
            coords = {1: [0.0, 0.0], 2: [4.0, 0.0], 3: [2.0, -1.5]}
            return coords, [1, 2], 3, [(1, 3), (2, 3)]
        coords = {
            1: [0.0, 0.0, 0.0],
            2: [4.0, 0.0, 0.0],
            3: [2.0, 3.0, 0.0],
            4: [2.0, 1.0, -2.0],
        }
        return coords, [1, 2, 3], 4, [(1, 4), (2, 4), (3, 4)]
    coords = {
        1: [0.0, 0.0, 0.0][:ndm],
        2: [2.0, 0.5, 0.0][:ndm],
        3: [4.0, 0.0, 0.0][:ndm],
    }
    return coords, [1], 3, [(1, 2), (2, 3)]


def build(element, ndm, coords, props, specs=()):
    """
    Assemble and solve one model, optionally with DDM parameters registered.

    Every element carries its own section and its own property values, so a
    parameter registered on element 1 can be central-differenced by perturbing
    element 1 alone. `G` and `J` are held independent of `E` and `I` for the
    same reason: a shared expression would make the difference quotient
    perturb quantities the DDM parameter does not.
    """
    ndf = ndm if element == "truss" else (3 if ndm == 2 else 6)
    _, fixed, loaded, bars = geometry(element, ndm)

    ops.wipe()
    ops.model("basic", "-ndm", ndm, "-ndf", ndf)
    for tag, xyz in coords.items():
        ops.node(tag, *xyz)
    for tag in fixed:
        ops.fix(tag, *([1] * ndf))

    if element == "truss":
        for e, (i, j) in enumerate(bars, 1):
            ops.uniaxialMaterial("Elastic", e, props[e]["E"])
            ops.element("truss", e, i, j, props[e]["A"], e)
    else:
        if ndm == 2:
            ops.geomTransf("Linear", 1)
        else:
            ops.geomTransf("Linear", 1, 0.0, 0.0, 1.0)
        for e, (i, j) in enumerate(bars, 1):
            p = props[e]
            if element == "elasticBeamColumn":
                if ndm == 2:
                    ops.element(
                        "elasticBeamColumn", e, i, j, p["A"], p["E"], p["Iz"], 1
                    )
                else:
                    args = (p["A"], p["E"], G0, J0, p["Iy"], p["Iz"])
                    ops.element("elasticBeamColumn", e, i, j, *args, 1)
                continue
            if ndm == 2:
                ops.section("Elastic", e, p["E"], p["A"], p["Iz"])
            else:
                ops.section("Elastic", e, p["E"], p["A"], p["Iz"], p["Iy"], G0, J0)
            rule = "Lobatto" if element == "forceBeamColumn" else "Legendre"
            ops.beamIntegration(rule, e, e, 5 if element == "forceBeamColumn" else 3)
            ops.element(element, e, i, j, 1, e)

    ops.timeSeries("Constant", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(loaded, *LOAD6[:ndf])

    for k, spec in enumerate(specs, 1):
        ops.parameter(k, *spec)

    ops.system("FullGeneral")
    ops.numberer("Plain")
    ops.constraints("Plain")
    ops.integrator("LoadControl", 1.0)
    ops.algorithm("Linear")
    ops.analysis("Static")
    if specs:
        ops.sensitivityAlgorithm("-computeAtEachStep")
    ops.analyze(1)
    return ndf, loaded


def classify(ddm, cd, floor):
    """
    Compare one parameter's sensitivity vector against its difference quotient.

    A zero is only evidence of a missing derivative where the difference
    quotient is above the noise floor, because several sensitivities are
    legitimately zero — a transverse response does not depend on the area.
    """
    status, worst, detail, seen = "NO-SIGNAL", 0.0, "", 0
    for k, (dv, ref) in enumerate(zip(ddm, cd), 1):
        if abs(ref) <= floor:
            continue
        seen += 1
        if status == "NO-SIGNAL":
            status = "OK"
        if dv == 0.0:
            return "MISSING", seen, f"dof{k}: cd={ref:+.5e} ddm=0"
        rel = abs(dv - ref) / abs(ref)
        if rel > worst:
            worst, detail = rel, f"dof{k}: ddm={dv:+.6e} cd={ref:+.6e} rel={rel:.2e}"
        if rel > TOL:
            status = "WRONG"
    return status, seen, detail


def capability_matrix():
    """Every (element, dimension, parameter) pair against central differences."""
    print("=" * 78)
    print("DDM capability matrix -- sensNodeDisp vs central differences")
    print("=" * 78)
    verdicts = {}
    for element in ("truss", "elasticBeamColumn", "dispBeamColumn", "forceBeamColumn"):
        for ndm in (2, 3):
            coords, _, _, bars = geometry(element, ndm)
            props = {
                e: {"E": E0, "A": A0, "Iz": I0, "Iy": I0}
                for e in range(1, len(bars) + 1)
            }
            # The second moment is named `I` in 2D and `Iz` in 3D; the wrong
            # name binds to nothing and reads as a missing derivative.
            inertia = "I" if ndm == 2 else "Iz"
            keys = ["E", "A"] if element == "truss" else ["E", "A", inertia]
            specs = [("element", 1, k) for k in keys]
            _, loaded = build(element, ndm, coords, props, specs)
            specs += [("node", loaded, "coord", d) for d in range(1, ndm + 1)]
            ndf, loaded = build(element, ndm, coords, props, specs)

            u0 = [ops.nodeDisp(loaded, k) for k in range(1, ndf + 1)]
            ddm = {
                i: [ops.sensNodeDisp(loaded, k, i) for k in range(1, ndf + 1)]
                for i in range(1, len(specs) + 1)
            }
            umax = max(abs(v) for v in u0)
            print(f"\n  {element}  ndm={ndm}  ndf={ndf}")
            row = {}
            for i, spec in enumerate(specs, 1):
                if spec[0] == "element":
                    label = spec[2]
                    key = "Iz" if label == "I" else label
                    base = props[1][key]
                    step = base * 1e-6
                    shifted = []
                    for sign in (+1, -1):
                        pert = {e: dict(p) for e, p in props.items()}
                        pert[1][key] += sign * step
                        build(element, ndm, coords, pert)
                        shifted.append(
                            [ops.nodeDisp(loaded, k) for k in range(1, ndf + 1)]
                        )
                else:
                    node, direction = spec[1], spec[3]
                    label = f"coord{direction}"
                    base = max(abs(coords[node][direction - 1]), 1.0)
                    step = base * 1e-6
                    shifted = []
                    for sign in (+1, -1):
                        pert = {t: list(c) for t, c in coords.items()}
                        pert[node][direction - 1] += sign * step
                        build(element, ndm, pert, props)
                        shifted.append(
                            [ops.nodeDisp(loaded, k) for k in range(1, ndf + 1)]
                        )

                cd = [(a - b) / (2 * step) for a, b in zip(*shifted)]
                status, seen, detail = classify(ddm[i], cd, 1e-6 * umax / base)
                row[label] = status
                print(
                    f"    {label:>7}  {status:>9}  ({seen} informative dof)  {detail}"
                )
            verdicts[f"{element} ndm={ndm}"] = row

    print("\n" + "-" * 78)
    print("SUMMARY")
    for case, row in verdicts.items():
        print(f"  {case:<32} " + "  ".join(f"{k}={v}" for k, v in row.items()))


def coord_3d_evidence():
    """
    Which number to believe where 3D DDM and central differences disagree.

    A difference quotient that is stable across four decades of step size is
    not a truncation artefact, so a DDM value that disagrees with it is wrong.
    The axis-aligned single bar is included because it is the one 3D case with
    a closed form.
    """
    print("=" * 78)
    print("3D coordinate sensitivity -- step-size refinement")
    print("=" * 78)

    length, axial = 4.0, -100e3

    def bar(x2, param=False):
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
        if param:
            ops.parameter(1, "node", 2, "coord", 1)
        ops.system("FullGeneral")
        ops.numberer("Plain")
        ops.constraints("Plain")
        ops.integrator("LoadControl", 1.0)
        ops.algorithm("Linear")
        ops.analysis("Static")
        if param:
            ops.sensitivityAlgorithm("-computeAtEachStep")
        ops.analyze(1)

    bar(length, param=True)
    disp = ops.nodeDisp(2, 1)
    ddm = ops.sensNodeDisp(2, 1, 1)
    exact = disp / length
    print("\n  single bar along global x, u = P L / (E A) so du/dx2 = u / L")
    print(f"    ddm    {ddm:+.10e}")
    print(f"    exact  {exact:+.10e}    ratio {ddm / exact:.8f}")

    coords, _, loaded, bars = geometry("truss", 3)
    props = {e: {"E": E0, "A": A0} for e in range(1, len(bars) + 1)}
    print("\n  tripod, bars not aligned with any global axis")
    for direction in (1, 2, 3):
        build("truss", 3, coords, props, [("node", loaded, "coord", direction)])
        ddm = [ops.sensNodeDisp(loaded, k, 1) for k in (1, 2, 3)]
        print(f"    coord{direction}  ddm = " + "  ".join(f"{v:+.8e}" for v in ddm))
        for rel_step in (1e-4, 1e-6, 1e-8):
            step = rel_step * 4.0
            shifted = []
            for sign in (+1, -1):
                pert = {t: list(c) for t, c in coords.items()}
                pert[loaded][direction - 1] += sign * step
                build("truss", 3, pert, props)
                shifted.append([ops.nodeDisp(loaded, k) for k in (1, 2, 3)])
            cd = [(a - b) / (2 * step) for a, b in zip(*shifted)]
            print(
                f"       h={rel_step:.0e}  cd  = " + "  ".join(f"{v:+.8e}" for v in cd)
            )


def force_sensitivity():
    """
    `∂N/∂xyz`, which is what T3 consumes — not `∂u/∂xyz`, which is what
    `sensNodeDisp` returns.

    Section forces are one step further down the chain than displacements, and
    the two beam formulations do not agree there even though both get
    displacements right. `sensSectionForce` returns the section vector starting
    at the requested dof, so element 0 is the value asked for; indexing `dof-1`
    quietly returns a neighbour. Passing an unregistered parameter tag segfaults
    the process outright.
    """
    print("=" * 78)
    print("section-force sensitivity to a nodal coordinate -- 2D")
    print("=" * 78)

    for element in ("forceBeamColumn", "dispBeamColumn"):
        coords, _, _, _ = geometry(element, 2)
        props = {e: {"E": E0, "A": A0, "Iz": I0, "Iy": I0} for e in (1, 2)}
        npt = 5 if element == "forceBeamColumn" else 3
        specs = [
            ("node", 3, "coord", 1),
            ("node", 3, "coord", 2),
            ("node", 2, "coord", 2),
        ]
        build(element, 2, coords, props, specs)

        worst = 0.0
        print(f"\n  {element}")
        for i, spec in enumerate(specs, 1):
            node, direction = spec[1], spec[3]
            step = 1e-6 * max(abs(coords[node][direction - 1]), 1.0)
            shifted = []
            for sign in (+1, -1):
                pert = {t: list(c) for t, c in coords.items()}
                pert[node][direction - 1] += sign * step
                build(element, 2, pert, props)
                shifted.append(
                    {
                        (e, s): ops.eleResponse(e, "section", s, "force")
                        for e in (1, 2)
                        for s in (1, npt)
                    }
                )
            build(element, 2, coords, props, specs)
            for ele in (1, 2):
                for sec in (1, npt):
                    for dof in (1, 2):
                        ddm = ops.sensSectionForce(ele, sec, dof, i)
                        if isinstance(ddm, list):
                            ddm = ddm[0]
                        lo, hi = shifted[1][(ele, sec)], shifted[0][(ele, sec)]
                        cd = (hi[dof - 1] - lo[dof - 1]) / (2 * step)
                        if abs(cd) <= 1e3:
                            continue
                        rel = abs(ddm - cd) / abs(cd)
                        worst = max(worst, rel)
                        name = "N" if dof == 1 else "M"
                        where = f"n{node} c{direction} e{ele} s{sec} d{name}"
                        print(
                            f"    {where:>20}  ddm {ddm:+13.6e}  "
                            f"cd {cd:+13.6e}  rel {rel:8.2e}"
                        )
        verdict = "AVAILABLE" if worst < 1e-5 else "NOT RELIABLE"
        print(f"    worst {worst:.3e}  ->  dN/dxyz {verdict}")


def printb_semantics():
    """
    Is the pseudo-load vector reachable, in any sensitivity mode?

    Reverse mode over OpenSees would need P_i = ∂f/∂θ_i − (∂K/∂θ_i) u without
    paying for a solve per parameter. Reconstructing it as K (du/dθ_i) proves
    what the vector should look like, but presupposes the very solve it would
    replace.
    """
    print("=" * 78)
    print("pseudo-load reachability -- what printB returns")
    print("=" * 78)

    nel = 6
    coords = {k + 1: [4.0 * k / nel, 0.0] for k in range(nel + 1)}

    def arch(specs=(), mode=None):
        ops.wipe()
        ops.model("basic", "-ndm", 2, "-ndf", 3)
        for tag, xy in coords.items():
            ops.node(tag, *xy)
        ops.fix(1, 1, 1, 1)
        ops.geomTransf("Linear", 1)
        for e in range(1, nel + 1):
            ops.section("Elastic", e, E0, A0, I0)
            ops.beamIntegration("Legendre", e, e, 3)
            ops.element("dispBeamColumn", e, e, e + 1, 1, e)
        ops.timeSeries("Constant", 1)
        ops.pattern("Plain", 1, 1)
        for t in range(2, nel + 2):
            ops.load(t, 0.0, -20e3, 0.0)
        for k, spec in enumerate(specs, 1):
            ops.parameter(k, *spec)
        ops.system("FullGeneral")
        ops.numberer("Plain")
        ops.constraints("Plain")
        ops.integrator("LoadControl", 1.0)
        ops.algorithm("Linear")
        ops.analysis("Static")
        if specs and mode:
            ops.sensitivityAlgorithm(mode)
        ops.analyze(1)
        if specs and mode == "-computeByCommand":
            ops.computeGradients()

    ndof = 3 * (nel + 1) - 3
    for mode in (None, "-computeAtEachStep", "-computeByCommand"):
        specs = () if mode is None else (("element", 1, "E"),)
        arch(specs, mode)
        stiffness = np.array(ops.printA("-ret")).reshape(ndof, ndof)
        rhs = np.array(ops.printB("-ret"))
        disp = np.array(
            [ops.nodeDisp(t, d) for t in range(2, nel + 2) for d in (1, 2, 3)]
        )
        applied = stiffness @ disp
        print(f"\n  mode {str(mode):>20}   |B| {np.linalg.norm(rhs):12.5e}")
        print(
            f"  {'':>25}   |K u| {np.linalg.norm(applied):12.5e}   (the applied load)"
        )
        if specs:
            sens = np.array(
                [
                    ops.sensNodeDisp(t, d, 1)
                    for t in range(2, nel + 2)
                    for d in (1, 2, 3)
                ]
            )
            pseudo = stiffness @ sens
            gap = np.linalg.norm(rhs - pseudo) / np.linalg.norm(pseudo)
            norm = np.linalg.norm(pseudo)
            print(f"  {'':>25}   |P_1| {norm:12.5e}   |B-P_1|/|P_1| {gap:.3e}")


def cost_scaling():
    """
    What the DDM sweep costs, against the finite-difference fallback.

    The fallback is one solve per parameter plus one; DDM reuses a single
    factorization and pays a back-substitution per parameter. The ratio is the
    number the P5 scaling plot needs.
    """
    print("=" * 78)
    print("cost scaling -- DDM sweep vs finite differences")
    print("=" * 78)

    def arch(nel, n_params=0, solve=True):
        ops.wipe()
        ops.model("basic", "-ndm", 2, "-ndf", 3)
        for k in range(nel + 1):
            x = 40.0 * k / nel
            ops.node(k + 1, x, 8.0 * 4 * x * (40.0 - x) / 1600.0)
        ops.fix(1, 1, 1, 1)
        ops.fix(nel + 1, 1, 1, 1)
        ops.geomTransf("Linear", 1)
        for e in range(1, nel + 1):
            ops.section("Elastic", e, E0, A0, I0)
            ops.beamIntegration("Legendre", e, e, 3)
            ops.element("dispBeamColumn", e, e, e + 1, 1, e)
        ops.timeSeries("Constant", 1)
        ops.pattern("Plain", 1, 1)
        for t in range(2, nel + 1):
            ops.load(t, 0.0, -20e3, 0.0)
        specs = [("element", e, ["E", "A", "Iz"][e % 3]) for e in range(1, nel + 1)]
        specs += [("node", t, "coord", 2) for t in range(2, nel + 1)]
        for k, spec in enumerate(specs[:n_params], 1):
            ops.parameter(k, *spec)
        ops.system("FullGeneral")
        ops.numberer("Plain")
        ops.constraints("Plain")
        ops.integrator("LoadControl", 1.0)
        ops.algorithm("Linear")
        ops.analysis("Static")
        if n_params:
            ops.sensitivityAlgorithm("-computeAtEachStep")
        if solve:
            ops.analyze(1)

    def fastest(fn, reps=3):
        best = float("inf")
        for _ in range(reps):
            start = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - start)
        return best

    for nel in (20, 100, 400):
        assembly = fastest(lambda nel=nel: arch(nel, solve=False))
        plain = fastest(lambda nel=nel: arch(nel))
        ndof = 3 * (nel + 1) - 6
        solve = (plain - assembly) * 1e3
        print(
            f"\n  nel={nel}  ndof={ndof}   "
            f"assemble {assembly * 1e3:.2f} ms   solve {solve:.3f} ms"
        )
        print(
            f"  {'n_params':>9} {'DDM total':>11} {'sens only':>11} "
            f"{'per param':>11} {'FD equiv':>11} {'speedup':>8}"
        )
        for n in (10, 50, 2 * nel - 2):
            total = fastest(lambda nel=nel, n=n: arch(nel, n))
            sens = total - plain
            fd = assembly + (n + 1) * plain
            print(
                f"  {n:>9} {total * 1e3:8.2f} ms {sens * 1e3:8.2f} ms "
                f"{sens / n * 1e3:8.4f} ms {fd * 1e3:8.2f} ms {fd / total:7.1f}x"
            )


PASSES = {
    "matrix": capability_matrix,
    "coord3d": coord_3d_evidence,
    "forces": force_sensitivity,
    "printb": printb_semantics,
    "timing": cost_scaling,
}


def main():
    """Run one pass, or fan every pass out into its own subprocess."""
    if len(sys.argv) > 1:
        PASSES[sys.argv[1]]()
        return
    for name in PASSES:
        print(f"\n{'#' * 78}\n# {name}\n{'#' * 78}", flush=True)
        done = subprocess.run([sys.executable, __file__, name], check=False)
        if done.returncode != 0:
            print(f"\n  pass {name!r} exited {done.returncode} -- OpenSees died")


if __name__ == "__main__":
    main()
