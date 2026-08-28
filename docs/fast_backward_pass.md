# Building a fast backward pass for a solver that has none

*An appendix. Every number here was measured on the 16×16 gridshell — 257 nodes,
496 members, 1267 differentiable parameters, three load cases — unless it says
otherwise. Reproduce them with `validation/pynite_adjoint.py`.*

PyNite is a space-frame analysis in plain Python. It has no tape, no tangent and
no sensitivity command, and no configuration produces one. Giving it a gradient
took six stages. The first made it *correct*; the other five made it fast, and
each was aimed at something the previous stage's measurement had proved was
actually expensive — which, four times out of five, was not where we had
guessed.

The arc is worth reporting because the intuitions it overturned are the ones
most people bring to a differentiable-programming boundary.

| stage | what changed | a crossed evaluation |
|---|---|---|
| 1 | exact element, implicit-function adjoint, dense Jacobian | 0.92 s |
| 2 | compile the element derivatives | 0.71 s |
| 3 | a true reverse rule instead of a Jacobian slice | — |
| 4 | build the right-hand side, keep the factorization | — |
| 5 | read the end forces from our own element | — |
| 6 | remember the assembled frame between endpoint calls | **0.077 s** |

A descent of 2408 evaluations fell from **37 minutes to about 3**, and the
gradient's agreement with a traced JAX solver never moved: **2.9e-12** over the
coordinates throughout.

---

## Stage 1 — Make it exact before making it fast

Two properties, both measured rather than assumed, decided the whole design.

**The element we differentiate is the element the solver assembled.** We state
the frame element in JAX and hold it against PyNite's own matrices over randomly
oriented members: the local stiffness agrees to **7.7e-18** and the global to
**5.7e-16**. That is what licenses differentiating a replica — it is not a
lookalike, it is the same element, and the test says so on every run.

**The global element stiffness does not know how the frame was rolled.** Turning
a member's transverse axes about its own axis moves the global matrix by
**1.7e-16** when the two second moments are equal, against 8.7e-3 when they are
not. A circular hollow section is what buys this, and three things follow: the
frame used to *build* stiffness may be chosen for conditioning alone; the frame
used to *report* bending is a convention rather than a fact; and the adjoint
needs only `∂Ke/∂p`, never the transformation's derivative.

The rejected alternative is worth stating. A **semi-analytic** rule — finite
differences of the element matrices, no global re-solve — was prototyped and
reached 1.0e-7. It was dropped because semi-analytic beam sensitivities have a
known pathology under rigid-body rotation whose error grows with slenderness,
which is exactly the regime of a slender gridshell. A number measured at one
configuration is not a bound along a descent.

**The trap here is the oracle.** A central difference cannot referee a
derivative at 1e-14; its own best agreement is 2.1e-10. What it *can* do is
confirm the shape of the error: sweeping the step gives a clean V, minimising at
h≈1e-2 and worsening in both directions. That V is the proof of exactness — a
wrong rule plateaus instead. For the value itself we check against a solver JAX
differentiates end to end, which is an independent *exact* answer rather than a
finer approximation.

## Stage 2 — Most of the JAX cost was dispatch, not arithmetic

Profiling the adjoint put 0.0502 s in the element derivatives. Compiled, the
same computation takes **0.0006 s** — 84× — because it was being dispatched
uncompiled every call, not because 496 members × 12×12×7 is a lot of arithmetic.

The lesson generalises past this project: when a JAX step inside an otherwise
non-JAX pipeline looks expensive, measure it compiled before believing it.

## Stage 3 — A reverse rule is not a slice of a Jacobian

The first adjoint served every endpoint from one dense Jacobian, mirroring the
sibling backend that wraps a solver whose sensitivities are forward-mode by
construction. That is the right shape when every block is wanted and the wrong
one for a gradient: it costs a back-substitution **per parameter** where a
reverse rule costs **one, whatever the parameter count**.

Written properly — pull the cotangent through each member's reading, gather one
adjoint load, solve once, contract element-locally — the reverse rule is
**0.046 s against 0.419 s** for the dense route, and the gap grows with the
structure because one is O(1) solves and the other O(parameters).

This is also the honest reason the planar backend beside it *cannot* be given
the same treatment. Its solver's Direct Differentiation Method returns one
Jacobian column per registered parameter; there is no call that returns a row,
and the adjoint's right-hand side is not exposed. The asymmetry is a result, not
a limitation to apologise for: **three backends, three differentiation
strategies, and their costs scale differently.**

## Stage 4 — A vector of zeros cost more than the factorization

Here the profile stopped agreeing with intuition entirely. Breaking one solve
apart:

| step | seconds |
|---|---|
| build the model from scratch | 0.0015 |
| `Ke()` assembly | 0.0257 |
| **fixed-end reactions, per load case** | **0.0176** |
| `splu` factorization | 0.0064 |
| nodal load vector | 0.0006 |
| one back-substitution | 0.0001 |
| reaction recovery | 0.0051 |

Three findings, in ascending order of surprise.

**Rebuilding the model was 1% of the cost.** It had been the first hypothesis.

**The fixed-end reactions are identically zero and cost more than the
factorization.** The solver builds its right-hand side as a nodal load vector
minus every member's fixed-end reactions. Under nodal loading there are none —
the vector is exactly zero — and producing it takes 0.0176 s per load case
against 0.0064 s to factorize. Building the right-hand side from the loads
already in hand takes **0.0000 s** and agrees to the last bit.

**An identical matrix was decomposed once per load case.** The stiffness does
not depend on the loading, and the solver knows this — its linear analysis
assembles once and loops the combinations. But it solves each with a routine
that factorizes afresh on every call, and the package contains no factorization
object anywhere. Three cases: **0.0200 s one at a time, 0.0070 s batched**, with
identical results.

That last one is a contribution rather than a workaround, and it is worth
sending upstream: a one-block change gives ~3× on any multi-combination linear
model.

## Stage 5 — Ask the solver for displacements, not for forces

Reading the end forces back through the solver's own per-member loop cost
**0.108 s**. The global end forces are the element stiffness times the
displacement, and we already hold that stiffness compiled and proven equal — the
same arithmetic in one mapped pass costs **0.0031 s**.

The side effect matters more than the speed. The forward pass and the derivative
now read **one function**: what the check reads is the head of what the adjoint
differentiates. A stage whose reported force came from one expression and whose
slope came from another would be free to disagree with itself. This one cannot.

## Stage 6 — The schema change we did not need

The remaining cost was that each of three load cases crossed the boundary
separately, each paying its own assembly and factorization. The obvious fix was
to widen the schema so one crossing carries the whole load-case stack — a change
touching every backend and the parity tests.

Profiling made it unnecessary. **The expensive half of a solve does not depend
on the loading at all**: assembling and factorizing is 0.0413 s of a 0.0414 s
forward pass. So one remembered frame, keyed on the geometry and the diameters,
serves every load case in an evaluation *and* the adjoint that follows each —
hit five times in six. Three load cases now cost **1.00×** what one costs.

The key omitting the loads is also why a **single** entry suffices. Under
reverse-mode automatic differentiation all the forward calls complete before any
backward call begins, so a cache keyed on anything the loads touch would be
evicted mid-evaluation. Keyed on what the preparation actually reads, only one
live entry ever exists.

It buys speed with a constraint, and the constraint is stated in the code: the
backend now depends on dispatch being serialized, which holds because every
endpoint runs on one owner thread. A served runtime would need a lock of its
own. Two tests cover what would otherwise be silent — a gradient over three
genuinely different load cases, and two geometries interleaved in one process.

---

## What the boundary actually cost

The premise of the whole exercise was that a Tesseract boundary is expensive.
Isolated properly — the *same* solver on both sides, which is the only way to
measure it — the boundary is **12%**. It was never the problem. The 74× gap
between a crossed descent and an in-process one was a pure-Python finite-element
solver against a compiled XLA program, and closing it took no architectural
change at all.

At the end the crossed gradient costs **0.16×** what the traced solver costs in
process. A hand-written adjoint over a library with no derivatives, reached
across a schema, is now six times faster than autodiff through a JAX-native
solver — because the adjoint is O(1) in the parameter count and does only the
work the answer requires.

## Five ways we mismeasured, and how each showed itself

Reported because they cost real time, and because most of them look like
results.

**Repeating the same point.** The first benchmark of the cache showed 30×. It
timed the same geometry repeatedly, so every repeat hit the cache — while a
descent moves the geometry each iteration and always misses once. Measured the
way a descent behaves, the true figure is 12×.

**Timing a cold call.** The cost table in the agreement experiment was reporting
a JAX compilation as if it were the forward solve.

**Measuring a different structure.** The same table priced a 248-member diagrid
while every act runs a 496-member gridshell — a different generator with a
similar name.

**Reading elapsed time as work.** A run showing 87 minutes elapsed against 53 of
CPU looked like it had stalled. The machine had been asleep.

**Blaming the wrong process.** A launcher process sitting at 0% CPU looked
wedged; the actual worker underneath it was at 117%. Silence in the log was
block-buffered output, not a hang — redirecting to a file buffers, and the fix
is to run unbuffered.

---

# Appendix to the appendix: the same method, applied to the check

Making the analysis cheap moved the bottleneck rather than removing it. Profiled
again, one crossed evaluation was **94% the check and 6% the solver**, and
everything in process — form-finding, 1730 constraint rows, the augmented
assembly — was 1.4 ms, half a percent. So the method repeated itself on a stage
written by someone else.

**The cost was allocation, not arithmetic.** Sizing 496 members builds **54,560
clause objects**: fifty bisection halvings per member, two clauses each.
Constructing the pair takes 1.958 µs; a whole utilization read takes 1.92 µs.
The formulas are free.

Three changes, in the same spirit as the six before them.

**Call the clause's own evaluator.** Blueprints' formula is a `float` subclass
whose constructor calls `_evaluate` and wraps the result — so calling that
directly is the library's arithmetic without the object, five times cheaper and
bit-identical. The search runs on it; **the answer does not**. Every reported
utilization is still read through the public class, so a section the library
would refuse is refused where it is reported.

That is a dependency on a private method, which is a real cost and is treated as
one: it is checked once at import for existing and for agreeing with its own
class, falls through to constructing the clause where it does not, and a test
pins the two at 1e-15.

**Stop the reverse rule re-solving the forward problem.** Both endpoints now
share one solved state — and it needs room for more than one point. That is the
trap from stage 6 inverted: there, the key was independent of the loading, so
one entry could never be evicted. Here the key *is* the actions, one evaluation
asks about more than one point, and reverse-mode runs every forward call before
any backward call. Measured with a single entry: **four misses, no hits.** The
same reasoning gave opposite answers on two stages, and only measuring told them
apart.

**Spend fifty halvings, not fifty-five.** Over four thousand random members the
diameter lands within one ulp of the larger budget and the utilization read back
is one to 2.776e-15 — against the 1e-9 the pipeline requires.

Together: the check from 0.245 s to about 0.040 s, an evaluation from 0.262 s to
0.11 s, and the crossed shell descent from 37 minutes to **4.8**.

## Parallelism was measured and declined

| 496 members | speedup |
|---|---|
| threads, 2 / 4 / 8 | **1.00×** |
| processes, warm pool | 4.0× on the bisection, 2.4× on the endpoint |
| processes, cold pool | **0.02×** — spawning costs 42× the work |
| free-threaded interpreter | 1.8×, on a runtime the stack cannot use |

Threads are flat by construction: the clause is a `float` subclass whose
attribute writes go through an overridden `__setattr__`, all pure Python, so
nothing releases the interpreter lock. Processes are the only real gain, and
they would require a pool outliving a host-callback endpoint that already
carries a known file-descriptor hazard — 2.4× does not buy that risk. Free
threading is the sharpest lesson: on the *allocating* code it was a **2×
regression**, because 27,000 shared class objects make reference counts
contend. The optimization that helped and the parallelism that would have hurt
were the same measurement away.

## What the design costs, stated rather than apologised for

A fully vectorized bisection over all members at once is **0.32 ms against
13.8 ms**. So calling a normative library scalar by scalar costs about an order
of magnitude — on a stage that, after this, is no longer the bottleneck. That is
the price of the boundary being the point, and it is worth naming precisely
instead of leaving implied.

## A closing caution: fast enough to see the noise

Cheap descents made something visible that expensive ones had hidden. Three runs
of an **identical** configuration landed at **0.105635, 0.091569 and 0.114863 t**
— a spread of about a tenth — while the forward pass stayed bit-identical
throughout, verified at a fixed point after every change. Round-off amplified
through four hundred inner iterations a round is enough to choose a different
basin.

The optimization did not cause this; it exposed it. At 37 minutes a run nobody
was going to discover it.

**And then the range was measured, which settled it.** Twenty-four starts on each
of the three routes — 4h20 of unattended compute, only affordable because of the
work above:

| route | best [t] | median | spread across starts | coefficient of variation | feasible |
|---|---|---|---|---|---|
| end to end | 0.074724 | 0.075659 | 53.7% | 14.5% | 23/24 |
| free heights | 0.136011 | 0.144273 | 24.0% | 6.1% | 24/24 |
| sizing only | 0.145735 | 0.148004 | 2.3% | 0.5% | 24/24 |

Three things the single-start numbers had hidden.

**A run is bit-reproducible inside a build.** Each route's first start, descended
twice, returns an identical mass. The three figures that differed came from three
different *builds* — a much milder claim than nondeterminism, and the right one.

**The scatter is the parametrization's, not the boundary's.** A frozen geometry
is nearly convex and all twenty-four starts agree to 2.3%. Searching over
geometry is what makes landings differ.

**The nominal start was the worst of twenty-four.** Every number quoted from a
single start came from it, and it is 54% heavier than the best. That is what
turned a 2.3% difference between two design checks into a reported 41.7% — a
wrong attribution that survived a day because both sides of the comparison
happened to be drawn from the same bad basin.

So the closing lesson is not "report a range" but something narrower and more
uncomfortable: **a comparison of two pipelines is only as good as the search
behind each side of it**, and matching the method on both sides is not a
refinement, it is the difference between a result and an artifact.

# The record those numbers came from, and how to plot it

The measurement is kept rather than regenerated, in `data/trajectories/`, and it
splits by what it cost.

**The expensive part.** `*_starts.json` and `*.log` — 92 KB holding all 24
fixed-seed starts per route, and behind them the 4h20 above. Every landing:
mass, worst violation, violated rows, evaluations, wall clock, and the design
variables. This is where the table comes from.

**The cheap part.** `*.npz` — about 2 MB and nine minutes, the path of each
route's winning start at one frame per objective evaluation, kept for the
descent animations. Each file holds `opening` (the start), `steps` (every
evaluation's variables), `masses` (mass per frame), `rounds` (the per-round
trail the method reports), `landing`, `route` and `start`.

**Regeneration was by fixed seed, not by luck.** The starts were drawn from a
fixed `SCATTER_SEED`, so a winner reconstructed exactly, and all three
re-descents reproduced their recorded mass to every digit — which is what says
the reconstruction was the same descent rather than a near neighbour.

**The script that made them is gone, deleted 2026-08-27.** It drove the
dissolved `normax.searches` through thirteen call sites, seven of them the
scattered multi-start apparatus and unique to it, so reviving it was a rebuild
rather than a repair. These files are therefore a record rather than a
regenerable output: what is on disk is what there is.

## Two things to know before plotting

**The paths dive below where they land, and that is the method working.** End to
end reaches 0.073284 mid-descent and settles at 0.074724; sizing only touches
0.116707 and settles at 0.145735. A small opening penalty lets the mass lead
until the multipliers pull it back onto the constraint surface, so those minima
are infeasible points and not better answers. Plot the violation beside the mass
or the picture says the opposite of what happened. Sizing only starts at 7.89 t
and ends at 0.146, a sixty-fold sweep — that axis wants to be logarithmic.

**A frame is an evaluation, not an accepted step.** The line search evaluates
points it then rejects, so consecutive frames can move backwards. Fine for a
morph; for a monotone curve use `rounds`, or take a running minimum.
