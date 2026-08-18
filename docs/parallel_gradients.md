# Parallel gradients and the third backend — decided 2026-08-15

Two decisions, taken together. **Blueprints may be a runtime dependency**:
hard constraint 1 in `CLAUDE.md` now prohibits ingesting its source, not
importing the unmodified package — ec3x keeps the stricter oracle-only posture
under its own rules. And **parallel finite differences apply to Blueprints
alone**: OpenSees has a cleaner derivative than any perturbation sweep, so the
process pool is not for it.

## The Blueprints backend — finite differences, in parallel

A Blueprints-backed checker is a third backend differentiated a third way —
traced (`smax`), implicit adjoint (OpenSees, below), numerical (Blueprints) —
which is the swappability thesis at full strength. Blueprints is plain
NumPy with no derivative story of its own, so its gradient endpoints come from
Tesseract Core's experimental helpers (`tesseract_core.runtime.experimental`:
`finite_difference_jacobian`, `_jvp`, `_vjp`).

Those helpers are serial. Their source (checked 2026-08-15,
`finite_differences.py` on `main`) is a plain `for idx in np.ndindex(...)`
loop calling `apply_fn` once or twice per input element — no multiprocessing,
no MPI, no threads anywhere in the module. A central-difference VJP is `2n`
full applies, one after another. The JVP is exempt: a directional derivative
is O(1) applies whatever the input dimension, so there is nothing to
parallelize there.

The perturbations share nothing, so the sweep is embarrassingly parallel:
partition them over a `concurrent.futures.ProcessPoolExecutor` (processes,
not threads — the checks are pure Python under the GIL), gather, and contract
with the cotangent. Speedup is the worker count, minus process startup and
payload pickling. Because `mpi4py.futures.MPIPoolExecutor` implements the same
`Executor` protocol, the identical code runs on a cluster unchanged. The
Tesseract contract never sees any of this; only how the endpoint pays.

## OpenSees — the implicit function theorem, not finite differences

The elastic analysis is the solved system `K(x) u = f(x)`, and a solved system
is an implicit function: differentiate the equilibrium and the derivative of
`u` is another solve against the already-factorized `K` — a JVP from
`K du = df − dK u`, a VJP from the adjoint `Kᵀ λ = cotangent`. OpenSees can be
asked for the assembled global stiffness matrix, so the rule needs the solver
only for `K` and `u`; the sweep over parameters collapses into one linear
solve per direction, and there is nothing left worth parallelizing. This is
the same rule the sizing map already ships in ec3x (`custom_jvp` via the
implicit function theorem over a monotone residual) — the pipeline would then
derive both of its opaque stages the same way, which is a better sentence in
the writeup than any speedup.

What the IFT route still owes: the partials `dK/dx · u` and `df/dx` with
respect to geometry and section fields have to come from somewhere — assembled
element by element analytically, or by finite differences on element matrices,
which are small. The DDM sweep in
`tesseracts/analysis/_backend_opensees.py` stays as the reference to verify
against, the same role central differences played when DDM itself was
validated (agreement to 7.4e-9, `CHANGELOG.md`).

## What it buys the hackathon

Three backends, three derivative mechanisms, one contract — and the
backend-agreement plot gains curves that mean something: reverse-traced in one
pass, implicit adjoint in one solve, finite differences at `O(n/P)`. The
depth claim is no longer "we parallelized a loop" but "each backend pays for
its gradient in its own currency, measured."

**Order of work against the Aug 27 freeze:** the Blueprints backend with
serial FD endpoints first — it must exist before its parallelization means
anything; the executor over its perturbations second (small, isolated); the
OpenSees IFT rule third, DDM staying as its oracle. Decide by Aug 22 how many
of the three fit.
