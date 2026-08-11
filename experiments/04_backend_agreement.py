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
One schema, two solvers that disagree about how a derivative is obtained.

The analysis stage is swapped underneath a pipeline that does not know it
happened. `smax` is a JAX frame solver traced end to end; OpenSees is C++ behind
a command interface, differentiated by rules hand-derived element by element and
compiled in years before this pipeline existed. Neither is a reimplementation of
the other, so every agreement below is a measurement.

Four passes:

    agreement  the member forces, then every block of the Jacobian, then the
               mass and its gradient end to end
    blind      the one derivative a two-dimensional model cannot reach, and why
               the composition never asks for it
    scaling    what each backend pays for a value and for a gradient, against
               the size of the frame
    optimize   the P4 descent driven by each backend in turn, compared on the
               answer rather than on the derivative

**The 2D restriction is OpenSees' and not a simplification here.** Its Direct
Differentiation Method reaches a nodal coordinate in two dimensions and returns
zero or wrong values in three. See `CHANGELOG.md` under `## OpenSees DDM spike`.

Requires both the `spike` extra and the `pipeline` group:
    uv run --extra spike --group pipeline python \
        experiments/04_backend_agreement.py [pass]

with `pass` one of agreement, blind, scaling, optimize, or omitted for all.
"""

import sys
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from normax.analysis import opensees as backend_opensees
from normax.analysis.smax import forces as forces_smax
from normax.analysis.smax import prepare as prepare_smax
from normax.composition import backend
from normax.composition import local
from normax.composition import mass as mass_composed
from normax.ec3.material import SteelGrade
from normax.ec3.sizing import TubeCatalogue
from normax.formfinding import equilibrium
from normax.formfinding import graph
from normax.optimization import descend
from normax.optimization import value_and_gradient
from normax.structures import arch
from normax.visualization import figure_backends

# A 10 m arch rising 3 m, carrying 180 kN. The same one the rest of the
# experiments use, so the numbers here sit beside theirs.
SPAN = 10_000.0
RISE = 3_000.0
TOTAL_LOAD = 180_000.0
NUM_EDGES = 10

# The arch lies in the XZ plane, so it has no thickness along Y.
NORMAL = 1

# The diameter the frame is analysed with before the check has spoken.
SEED = 100.0

# Frame sizes for the cost sweep. Each adds two coordinate parameters per node
# and two section parameters per member to the sweep OpenSees performs.
MESHES = (5, 10, 20, 40)

# Timed calls after the warm-up, at each size. Odd, so the median is a sample
# rather than an average of two.
REPEATS = 7

# What the roadmap asked the two backends to agree to.
TOLERANCE_ASKED = 1e-6

# The descent, matching `experiments/03`. The force densities may move a decade
# either side of the funicular value, the bound keeping them away from zero
# where the force density system is singular.
DECADES = 10.0
ITERATIONS = 60

FIGURES = Path(__file__).resolve().parent.parent / "figures"

STEEL = SteelGrade()
CATALOGUE = TubeCatalogue.at_class_limit(STEEL.f_y, 3)

BACKENDS = ("smax", "opensees")


def setup(num_edges):
    """
    The arch, its form-finding connectivity, and the `q` that reaches the rise.
    """
    load = TOTAL_LOAD / (num_edges - 1)
    structure = arch(num_edges=num_edges, span=SPAN, rise=RISE, load=load)
    graph_fdm = graph(structure)

    trial = jnp.full(num_edges, -1.0)
    reached = jnp.max(equilibrium(trial, structure, graph_fdm).xyz[:, 2])

    return structure, graph_fdm, trial * reached / RISE


def relative(actual, expected):
    """
    Worst absolute gap over the largest entry of the reference.
    """
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    scale = float(np.max(np.abs(expected)))

    return float(np.max(np.abs(actual - expected))) / (scale if scale > 0.0 else 1.0)


def objective(chain, structure, num_edges):
    """
    Force densities to a mass, through whichever backend is selected.
    """
    seed = jnp.full(num_edges, SEED)

    def total(q):
        return mass_composed(
            q,
            seed,
            structure,
            chain,
            STEEL,
            CATALOGUE,
            normal=NORMAL,
            plastic=False,
        )

    return total


def agreement():
    """Member forces, Jacobian blocks, and the mass gradient end to end."""
    print("=" * 78)
    print("Two solvers on one schema -- agreement")
    print("=" * 78)

    structure, graph_fdm, q = setup(NUM_EDGES)
    xyz = equilibrium(q, structure, graph_fdm).xyz
    diameters = jnp.full(NUM_EDGES, SEED)

    mine = backend_opensees.forces(
        backend_opensees.prepare(structure, STEEL, CATALOGUE, normal=NORMAL),
        xyz,
        diameters,
        STEEL,
        CATALOGUE,
    )
    theirs = forces_smax(
        prepare_smax(structure, STEEL, CATALOGUE, normal=NORMAL),
        xyz,
        diameters,
        STEEL,
        CATALOGUE,
    )

    print("\nmember forces, DDM backend against the traced one")
    for name in ("n_ed", "m_y_ed"):
        gap = relative(getattr(mine, name), getattr(theirs, name))
        print(f"  {name:<10} worst relative {gap:.3e}")
    minor = float(np.max(np.abs(np.asarray(mine.m_z_ed))))
    print(f"  {'m_z_ed':<10} exactly zero in a plane frame, max {minor:.1e}")

    blocks = backend_opensees.jacobian(
        backend_opensees.prepare(structure, STEEL, CATALOGUE, normal=NORMAL),
        xyz,
        diameters,
        STEEL,
        CATALOGUE,
    )

    def run(coords, sizes):
        member = forces_smax(
            prepare_smax(structure, STEEL, CATALOGUE, normal=NORMAL),
            coords,
            sizes,
            STEEL,
            CATALOGUE,
        )

        return {"n_ed": member.n_ed, "m_y_ed": member.m_y_ed}

    by_coordinate = jax.jacfwd(run, argnums=0)(xyz, diameters)
    by_diameter = jax.jacfwd(run, argnums=1)(xyz, diameters)

    print("\nJacobian blocks, hand-derived C++ against traced autodiff")
    pairs = (
        ("n_ed_xyz", blocks.n_ed_xyz, by_coordinate["n_ed"]),
        ("m_y_ed_xyz", blocks.m_y_ed_xyz, by_coordinate["m_y_ed"]),
        ("n_ed_diameter", blocks.n_ed_diameter, by_diameter["n_ed"]),
        ("m_y_ed_diameter", blocks.m_y_ed_diameter, by_diameter["m_y_ed"]),
    )
    worst = 0.0
    for name, ddm, traced in pairs:
        gap = relative(ddm, traced)
        worst = max(worst, gap)
        shape = str(np.asarray(ddm).shape)
        print(f"  {name:<18} {shape:<22} worst relative {gap:.3e}")
    print(f"  worst over every block  {worst:.3e}")

    chain = local()
    total = objective(chain, structure, NUM_EDGES)

    print("\nend to end, force densities to a mass")
    results = {}
    for name in BACKENDS:
        with backend(name):
            results[name] = (float(total(q)), np.asarray(jax.grad(total)(q)))
        print(f"  {name:<10} mass {results[name][0]:.9f} t")

    mass_gap = abs(results["opensees"][0] - results["smax"][0]) / results["smax"][0]
    grad_gap = relative(results["opensees"][1], results["smax"][1])
    print(f"  mass       relative gap {mass_gap:.3e}")
    print(f"  dmass/dq   worst relative {grad_gap:.3e}, asked {TOLERANCE_ASKED:.0e}")

    return grad_gap


def blind():
    """The one derivative the plane cannot carry, and why nothing asks for it."""
    print("=" * 78)
    print("The block a two-dimensional model cannot reach")
    print("=" * 78)

    structure, graph_fdm, q = setup(NUM_EDGES)
    xyz = equilibrium(q, structure, graph_fdm).xyz
    diameters = jnp.full(NUM_EDGES, SEED)

    def run(coords):
        member = forces_smax(
            prepare_smax(structure, STEEL, CATALOGUE, normal=NORMAL),
            coords,
            diameters,
            STEEL,
            CATALOGUE,
        )

        return {"n_ed": member.n_ed, "m_y_ed": member.m_y_ed, "m_z_ed": member.m_z_ed}

    jacobian = jax.jacfwd(run)(xyz)

    print("\nthe three-dimensional Jacobian, by global axis")
    print(f"  {'output':<10} {'d/dx':>14} {'d/dy':>14} {'d/dz':>14}")
    for name, block in jacobian.items():
        sizes = [
            float(np.max(np.abs(np.asarray(block)[..., axis]))) for axis in range(3)
        ]
        mark = " <- normal" if name == "m_z_ed" else ""
        print(f"  {name:<10} " + " ".join(f"{s:>14.6e}" for s in sizes) + mark)

    print("\n  The response separates: nothing in the plane moves when a node")
    print("  leaves it, and the minor-axis moment moves only then. A plane model")
    print("  carries every block but that one.")

    interior = xyz.shape[0] // 2
    tangent = jnp.zeros_like(xyz).at[interior, NORMAL].set(1.0)
    _, pushed = jax.jvp(run, (xyz,), (tangent,))

    print(f"\nthe same, pushing node {interior} alone out of the plane")
    for name, value in pushed.items():
        print(f"  {name:<10} {float(np.max(np.abs(np.asarray(value)))):.6e}")
    print("  One node, because translating every node out of the plane together")
    print("  is a rigid motion and strains nothing — it would read as blindness")
    print("  where there is none.")

    reachable = jax.jacfwd(lambda qq: equilibrium(qq, structure, graph_fdm).xyz)(q)
    out_of_plane = float(np.max(np.abs(np.asarray(reachable)[:, NORMAL, :])))

    print("\nand what form finding can do about it")
    print(f"  worst |d xyz[normal] / dq|  {out_of_plane:.3e}")
    print("  The force density method decouples per coordinate, so a planar arch")
    print("  stays planar for every q. The blind block is multiplied by zero.")


def stage_cost(structure, xyz, diameters):
    """
    Seconds each backend spends on the analysis stage's derivatives alone.

    Isolated from the composition on purpose. The whole pipeline pays for form
    finding, a sizing bisection and two boundary crossings whoever solves the
    frame, and those dominate at these sizes; the scaling claim is about the
    stage.

    Notes
    -----
    **Both backends are prepared once and timed on the work that remains**, which
    is how the stage's contract says to use them. Preparing inside the timed call
    would charge the traced backend for compiling an assembly it is meant to reuse
    and charge neither for what an optimizer actually pays per iterate.

    **The traced Jacobian is compiled.** Uncompiled it runs two orders of
    magnitude slower, and comparing that against a C++ sweep measures Python
    dispatch rather than either differentiation strategy. The compilation is a
    fixed cost per frame size and the warm-up excludes it, exactly as it excludes
    the kernel the section slopes need on the other side.
    """
    prepared_ddm = backend_opensees.prepare(structure, STEEL, CATALOGUE, normal=NORMAL)
    prepared_smax = prepare_smax(structure, STEEL, CATALOGUE, normal=NORMAL)

    def ddm():
        return backend_opensees.jacobian(prepared_ddm, xyz, diameters, STEEL, CATALOGUE)

    def run(coords, sizes):
        member = forces_smax(prepared_smax, coords, sizes, STEEL, CATALOGUE)

        return {"n_ed": member.n_ed, "m_y_ed": member.m_y_ed}

    coordinates = eqx.filter_jit(jax.jacfwd(run, argnums=0))
    sections = eqx.filter_jit(jax.jacfwd(run, argnums=1))

    def traced():
        return (
            coordinates(xyz, diameters),
            sections(xyz, diameters),
        )

    return {"opensees": steady(ddm), "smax": steady(traced)}


def steady(call, repeats=REPEATS):
    """
    Seconds per call once nothing is being compiled for the first time.

    Parameters
    ----------
    call :
        The work to time, taking no arguments and returning its result.
    repeats :
        Times to run it after the warm-up.

    Returns
    -------
    seconds :
        Median seconds per call.

    Notes
    -----
    **The median rather than the mean, because the composed timings are noisy.**
    A crossing of the boundary is host-side work of a few hundred milliseconds, and
    one sample landing at three times the rest — a collection, a page fault, the
    scheduler — moves a mean of five enough to reverse which backend looks faster.
    The stage timings are stable either way, so nothing is lost by taking the
    middle sample for both.

    **The warm-up is not optional and it is not noise.** The section slopes come
    from `jax.grad` of the closed forms, so the first call at a new member count
    compiles a kernel and reports two orders of magnitude more than the second.
    Timing it cold would measure XLA and call it direct differentiation. An
    optimizer pays that once and this cost hundreds of times.

    **The result is waited on rather than merely dispatched.** JAX returns before
    a computation has run, so timing without blocking measures the queueing of the
    traced backend against the completion of the C++ one, and flatters the first
    by however much of it is still outstanding. Whatever the call returns is
    blocked on; a call returning nothing would be timed wrongly and silently.
    """
    jax.block_until_ready(call())

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(call())
        samples.append(time.perf_counter() - start)

    return float(np.median(samples))


def scaling():
    """
    What a value and a gradient cost each backend, against frame size.

    Notes
    -----
    **Two different things are timed and they answer different questions.** The
    stage alone compares one backend's derivatives against the other's with both
    prepared once and the traced one compiled, which is what a caller of the stage
    pays per iterate. The whole composition runs through the Tesseracts, so it
    also pays form finding, a sizing bisection and two boundary crossings, and
    those dominate at these sizes whoever solves the frame.

    **The composed path is compiled but not prepared once.** Its solve is compiled
    inside the backend, so what the composed columns still carry is one assembly
    per crossing: a boundary is stateless and keeps nothing between calls. The
    in-process pipeline is the one that also reuses a prepared model, and
    `experiments/03` is where that shows.
    """
    print("=" * 78)
    print("Cost against frame size")
    print("=" * 78)

    members = []
    parameters = []
    gaps = []
    stage = {name: [] for name in BACKENDS}
    pipeline = {name: [] for name in BACKENDS}

    print("\n  the whole composition, one value then a value and gradient")
    print("  through the Tesseracts, warmed, the assembly rebuilt at each crossing")
    print(f"  {'members':>8} {'params':>7} {'backend':>9} {'value':>9} {'grad':>9}")
    for num_edges in MESHES:
        structure, graph_fdm, q = setup(num_edges)
        xyz = equilibrium(q, structure, graph_fdm).xyz
        diameters = jnp.full(num_edges, SEED)
        chain = local()
        total = objective(chain, structure, num_edges)

        grads = {}
        count = 2 * (num_edges + 1) + 2 * num_edges
        for name in BACKENDS:
            with backend(name):
                gradient = jax.grad(total)

                seconds_value = steady(lambda f=total: f(q))
                seconds_grad = steady(lambda f=gradient: f(q))
                grads[name] = np.asarray(gradient(q))

            pipeline[name].append(seconds_grad)
            print(
                f"  {num_edges:>8} {count:>7} {name:>9}"
                f" {seconds_value:>9.3f} {seconds_grad:>9.3f}"
            )

        alone = stage_cost(structure, xyz, diameters)
        for name in BACKENDS:
            stage[name].append(alone[name])

        members.append(num_edges)
        parameters.append(count)
        gaps.append(relative(grads["opensees"], grads["smax"]))

    print("\n  the analysis stage alone, every derivative it can report")
    print("  both prepared once, the traced one compiled, warm-up excluded")
    print(f"  {'members':>8} {'params':>7} {'DDM [ms]':>10} {'traced [ms]':>12}")
    for index, num_edges in enumerate(members):
        print(
            f"  {num_edges:>8} {parameters[index]:>7}"
            f" {stage['opensees'][index] * 1e3:>10.1f}"
            f" {stage['smax'][index] * 1e3:>12.1f}"
        )

    print("\n  agreement at every size")
    for num_edges, gap in zip(members, gaps):
        print(f"    {num_edges:>4} members   worst relative {gap:.3e}")

    print("\n  cost per registered parameter, which is what DDM pays by")
    for index, num_edges in enumerate(members):
        per = stage["opensees"][index] / parameters[index] * 1e3
        print(f"    {num_edges:>4} members   {per:.3f} ms/param")

    print("\n  the traced gradient over the DDM one, below one where tracing wins")
    for index, num_edges in enumerate(members):
        alone = stage["smax"][index] / stage["opensees"][index]
        whole = pipeline["smax"][index] / pipeline["opensees"][index]
        print(
            f"    {num_edges:>4} members   stage {alone:.2f}x"
            f"   composition {whole:.2f}x"
        )
    print("    both are compiled; the composition prepares its assembly per")
    print("    crossing, a boundary keeping nothing between calls")

    figure = figure_backends(
        np.asarray(members),
        np.asarray(parameters),
        np.asarray(gaps),
        {name: np.asarray(series) for name, series in stage.items()},
        {name: np.asarray(series) for name, series in pipeline.items()},
        TOLERANCE_ASKED,
    )
    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / "04_backends.png"
    figure.savefig(path, dpi=200)
    print(f"\n  wrote {path}")


def optimize():
    """
    The same descent, driven by each backend in turn.

    Notes
    -----
    **Compiled before the clock starts, and the compilation reported beside the
    search rather than inside it.** Each objective is traced once here and the
    compiled program handed to `descend`, so the elapsed time of a descent is the
    work it did. Leaving it inside would charge one backend a fixed cost the other
    never pays and call the difference a solver comparison.

    The compilation is a real cost and is printed, not hidden. It is paid once per
    objective however long the search runs, so it matters on a descent of seven
    steps and vanishes on one of several hundred.
    """
    print("=" * 78)
    print("The same optimization, one solver swapped for the other")
    print("=" * 78)

    structure, _, q = setup(NUM_EDGES)
    chain = local()
    total = objective(chain, structure, NUM_EDGES)

    bounds = (float(q[0]) * DECADES, float(q[0]) / DECADES)
    print(f"\n  one variable per member, bounds {bounds}")

    results = {}
    for name in BACKENDS:
        with backend(name):
            gradient = value_and_gradient(total)

            start = time.perf_counter()
            jax.block_until_ready(gradient(q))
            compiling = time.perf_counter() - start

            start = time.perf_counter()
            result = descend(
                total, q, bounds=bounds, iterations=ITERATIONS, gradient=gradient
            )
            elapsed = time.perf_counter() - start

        results[name] = (result, elapsed)
        steps = result.mass.shape[0]
        print(
            f"\n  {name:<9} mass {float(result.mass[-1]):.9f} t"
            f"  in {steps} steps, {elapsed:.1f} s"
            f"  ({elapsed / steps * 1e3:.0f} ms/step)"
        )
        print(f"  {'':<9} compiled in {compiling:.2f} s, before the clock started")

    first, _ = results["smax"]
    second, _ = results["opensees"]

    reached = abs(float(second.mass[-1]) - float(first.mass[-1]))
    print("\n  the two answers")
    print(f"    mass         relative gap {reached / float(first.mass[-1]):.3e}")
    print(f"    q            worst relative {relative(second.q[-1], first.q[-1]):.3e}")
    print(f"    steps        {first.mass.shape[0]} against {second.mass.shape[0]}")

    speedup = results["smax"][1] / results["opensees"][1]
    print(f"    wall clock   {speedup:.1f}x faster on the C++ backend, compiled")


PASSES = {
    "agreement": agreement,
    "blind": blind,
    "scaling": scaling,
    "optimize": optimize,
}


def main():
    requested = sys.argv[1:] or list(PASSES)

    for name in requested:
        if name not in PASSES:
            raise SystemExit(f"unknown pass {name!r}; choose from {list(PASSES)}")

    for name in requested:
        PASSES[name]()
        print()


if __name__ == "__main__":
    main()
