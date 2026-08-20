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

## Addendum 2026-08-20 — the study feeding the Aug 22 decision

The batched-Jacobian client and the parallel-FD client are not competitors.
They answer different questions at different layers: the endpoint makes the
derivative crossings the pipeline already does cheap, and creates no new
derivative; finite differences give a gradient to code that has none, and a
coarse one. They also compose — finite differences produce a full Jacobian by
construction, so an FD backend is the natural tenant behind a `jacobian`
endpoint, while serving FD through the sequential VJP path would multiply the
two costs: crossings times perturbation sweeps.

### The batched-Jacobian endpoint

None of the four servers defines `jacobian`; `tesseract-jax` 0.4.1 already
materializes through it whenever it exists, and the stacked-cotangent
alternative is blocked upstream twice, so the endpoint is the only supported
route. For it: it targets the measured pain directly (experiment 103 crossed
was 37 s against 1.0 s in process, ~13.7 ms a crossing, ~30 sequential
crossings per SLSQP iteration, each redoing the primal because a request is
stateless — the endpoint wins whenever `(R-1)(t+P) > (O-R)a`, and the
measured `O/R` of 1, 2 and ~1.7 across the stages makes that essentially
always true at R ≈ 30); two stages get it almost free (`blueprint_check`'s
Jacobian is diagonal in members, and the DDM sweep already assembles the
dense Jacobian it then discards rows of); it serves `jacfwd` too, composing
with the cheaper row count of forward mode on a wide constraint slack; and
gradient exactness is untouched. Against it: it is pure acceleration with a
one-crossing floor; the traced servers implement it as a server-side
`jacrev` that recompiles per shape; materialized payloads grow with the
Jacobian's area, irrelevant at this scale; and the served half of the
measurement is blocked for `ec3_check` by the sibling-repo wheel outside the
build context, which is why the standing order — `blueprint_check` first,
measure, then roll out — is the right sequencing.

### The parallel-FD client

For it: the strongest why-Tesseract sentence available — a stock library
nobody instrumented composes into an end-to-end gradient, paying in its own
currency, and the claim generalizes to every legacy code the building-code
argument gestures at; the sweep is embarrassingly parallel with the cluster
story free; the FD helpers already exist so only the executor is new; and it
slots behind the `jacobian` endpoint for one crossing. Against it, and these
are load-bearing: central differences top out around 1e-8 to 1e-10 relative
where the exact adjoints hold 1e-12 to 1e-14, and FD is order-one wrong at
kinks — exactly where a fully-stressed optimizer parks every member
(`U = 1` active, the χ cap, the tension–compression sign), branches the hand
adjoints dispatch analytically; fed to SLSQP at `ftol` 1e-12 that noise
stalls or false-converges a descent, so FD must never drive one; a
central-difference Jacobian of the check is hundreds of primal applies per
iteration, which a local pool divides but never closes against one primal
plus a cheap adjoint; the repo already carries an FD scar (the zero-moment
gradcheck needed a widened one-sided stencil); and it is a third route for a
stage that already has two exact ones — narrative depth, not capability.

### Recommendation

Build the endpoint as decided: `blueprint_check` first, measure in process
and served, roll out on the realized number — it is needed regardless of the
FD question. Treat parallel-FD as a demonstration and never as the
optimizer's gradient: if the endpoint lands fast and the gridshell is on
track by ~Aug 24, build the serial backend plus the executor as an
experiment-level exhibit — a third curve on the backend-agreement plot, its
accuracy quoted honestly against the exact adjoints, with one honest
sentence about finite differences at active constraints. If time is tight,
cut it and keep the future-work paragraph; the three-mechanisms claim
already stands on traced, DDM and hand adjoint.
