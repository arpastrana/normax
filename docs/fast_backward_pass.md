# Building a fast backward pass for a solver that has none

Unless noted, measurements use the 16×16 gridshell with 257 nodes, 496 members,
1,267 differentiable parameters, and three load cases. Reproduce them with
`validation/pynite_adjoint.py`.

PyNite is a structural analysis solver in plain Python with no tape, tangent,
or sensitivity command. Its reverse rule took six stages. The first established
correctness.
Measurement chose the next five.

| stage | change | crossed evaluation |
|---|---|---:|
| 1 | exact element, implicit adjoint, dense Jacobian | 0.92 s |
| 2 | compile element derivatives | 0.71 s |
| 3 | use a true reverse rule | not isolated |
| 4 | build the right-hand side and retain the factorization | not isolated |
| 5 | recover forces from the verified element | not isolated |
| 6 | cache the assembled frame across endpoint calls | **0.077 s** |

A 2,408-evaluation descent fell from **37 minutes to about 3**. The gradient
stayed fixed while the implementation changed.

## Accuracy first

The JAX element must match the element PyNite assembles. Randomly oriented
members agree to **7.7e-18** in local stiffness and **5.7e-16** globally.

Circular hollow sections have equal second moments. Rolling their transverse
axes changes the global matrix by only **1.7e-16**. The assembly frame can
therefore favor conditioning, while the reporting frame remains a convention.
The reverse rule needs `∂Ke/∂p`, not the frame derivative. This result does not
extend to unequal second moments.

A rejected semi-analytic rule used finite differences of element matrices but
no global re-solve. It reached 1.0e-7 at one configuration. That was not enough.
Rigid-body rotation can amplify semi-analytic beam errors with slenderness, the
regime of a gridshell.

Central differences cannot certify a 1e-14 derivative. They can expose its
error shape. On a 6-node, 8-member canopy with 26 parameters, step size produces
the expected interior minimum:

| step [mm] | node gradient | diameter gradient |
|---|---:|---:|
| 1e-05 | 1.364e-07 | 1.564e-07 |
| 1e-04 | 1.072e-08 | 1.823e-08 |
| **1e-03** | **1.555e-09** | **2.218e-09** |
| 1e-02 | 4.949e-09 | 1.240e-08 |
| 1e-01 | 5.065e-07 | 1.242e-06 |

The crossed adjoint agrees with the same rule in process at **2.587e-15** and
with frozen gradient-block norms at **1.168e-14**. See
[results.md](results.md#validation-evidence) for the complete evidence ladder.

## 1. Compile the element derivatives

Profiling assigned 0.0502 s to element derivatives. Compilation reduced the
same work to **0.0006 s**, an 84× change. Dispatch, not arithmetic, was the cost.

## 2. Write a reverse rule

The first implementation built a dense Jacobian. That matches a forward-mode
solver but wastes work for one scalar objective. It pays one back-substitution
per parameter. A reverse rule pays one solve.

The reverse rule pulls the cotangent through member-force recovery, gathers one
adjoint load, solves once, then contracts by element. It costs **0.046 s** versus
**0.419 s** for the dense route. The gap grows with parameter count.

OpenSees cannot use the same shortcut through its public API. Its Direct
Differentiation Method returns a column for each registered parameter. It
exposes neither an adjoint row nor the required right-hand side. Different
backends therefore use different derivative strategies and scaling laws.

## 3. Build only the needed right-hand side

One solve breaks down as follows:

| step | seconds |
|---|---:|
| build model | 0.0015 |
| assemble `Ke()` | 0.0257 |
| **fixed-end reactions per load case** | **0.0176** |
| factorize with `splu` | 0.0064 |
| build nodal load vector | 0.0006 |
| back-substitute | 0.0001 |
| recover reactions | 0.0051 |

Rebuilding was only 1% of cost. The surprise was a vector of zeros. Under nodal
loading, fixed-end reactions vanish, yet PyNite spent 0.0176 s per case building
them. Constructing the right-hand side from known loads took effectively zero
time and agreed bit for bit.

The stiffness matrix also stays fixed across load cases. PyNite assembled once
but refactorized inside each solve. Three cases cost **0.0200 s** separately and
**0.0070 s** with one factorization. The same change could benefit any linear
multi-combination model upstream.

## 4. Recover forces from the verified element

PyNite's per-member force loop cost **0.108 s**. A mapped multiplication of the
verified element stiffness by displacements cost **0.0031 s**.

This also joined the primal and derivative. The code check now reads the same
element function that the adjoint differentiates.

## 5. Cache preparation, not results

Three load cases originally crossed separately and repeated assembly and
factorization. A wider schema looked necessary. Profiling showed otherwise.
Preparation accounts for 0.0413 s of a 0.0414 s forward pass and does not depend
on loads.

One cached frame, keyed by geometry and diameters, serves every load case and
its adjoint. Five of six calls hit. On the 16×16 shell, three cases cost
**2.24×** one solve instead of 3×. An earlier draft claimed 1×, which the script
never printed.

One cache entry suffices because reverse mode completes forward calls before
backward calls. The key excludes loads because preparation excludes them. A
served runtime would need its own lock. Tests cover three distinct load cases
and two interleaved geometries.

## Boundary cost

With the same solver on each side, the Tesseract boundary adds **12%**. The
original 74× gap compared Python finite elements with compiled XLA. It was not a
boundary result.

The final crossed gradient costs **0.16×** the traced in-process solver. The
hand adjoint is faster because it uses one solve and computes only the requested
contraction.

## Benchmark traps

Five mistakes looked like results:

- **Repeated point.** A 30× cache result measured only hits. A moving descent
  showed 12×.
- **Cold call.** One table counted JAX compilation as a solve.
- **Wrong structure.** One timing used a 248-member diagrid instead of the
  496-member gridshell.
- **Elapsed time.** A sleeping machine looked stalled.
- **Wrong process.** An idle launcher hid a busy worker. Buffered logs hid its
  progress.

## Applying the method to the code check

Once analysis was cheap, one evaluation spent **94%** in the check and 6% in
the solver. In-process work took 1.4 ms.

Sizing 496 members allocated **54,560 clause objects**. Fifty bisection steps
constructed two objects per step and member. One clause pair took 1.958 µs,
about as long as an entire utilization read. Allocation was the cost.

Three changes followed:

1. Call Blueprints' internal `_evaluate` directly during search. It is five
   times cheaper and bit-identical. Final reported utilization still uses the
   public class. Import checks verify the internal evaluator and fall back when
   absent.
2. Share solved sizing states between forward and reverse endpoints. Unlike the
   frame cache, this cache needs multiple entries because actions belong to the
   key. One entry produced four misses and no hits.
3. Use fifty bisection steps instead of fifty-five. Across 4,000 random members,
   diameters stay within one ulp and utilization within 2.776e-15. The pipeline
   tolerance is 1e-9.

The check fell from 0.245 s to about 0.040 s. One evaluation fell from 0.262 s
to 0.11 s. The crossed shell descent fell from 37 minutes to **4.8**.

## Parallelism declined

| 496 members | speedup |
|---|---:|
| threads, 2 / 4 / 8 | **1.00×** |
| processes, warm pool | 4.0× for bisection, 2.4× per endpoint |
| processes, cold pool | **0.02×** |
| free-threaded interpreter | 1.8× on an unusable runtime |

Blueprints clause construction is pure Python, so threads cannot help. A warm
process pool helps but conflicts with a host-callback endpoint that already has
a file-descriptor hazard. A cold pool costs 42 times the work. Free threading
made the allocating version twice as slow because shared reference counts
contended. A 2.4× endpoint gain did not justify the added failure modes.

A fully vectorized bisection takes **0.32 ms**, compared with **13.8 ms** for
scalar library calls. That order-of-magnitude cost buys evaluation through the
normative implementation. After the changes, it is no longer the bottleneck.

## Fast enough to see optimization noise

Three nominally identical configurations built at different times landed at
**0.105635, 0.091569, and 0.114863 t**. Fixed-point forward evaluations remained
bit-identical. Round-off over hundreds of iterations selected different basins.

Twenty-four starts per route made the effect measurable:

| route | best [t] | median | spread | coefficient of variation | feasible |
|---|---:|---:|---:|---:|---:|
| end to end | 0.074724 | 0.075659 | 53.7% | 14.5% | 23/24 |
| free heights | 0.136011 | 0.144273 | 24.0% | 6.1% | 24/24 |
| sizing only | 0.145735 | 0.148004 | 2.3% | 0.5% | 24/24 |

Repeated starts within one build returned identical masses. Different builds
produced the earlier scatter. Geometry search, not the boundary, created most of
the basin sensitivity. The nominal start was the worst of 24 and 54% heavier
than the best.

The lesson is narrow. A pipeline comparison is only as sound as the search on
both sides. Match start budgets and report distributions when claiming
robustness.

The 24-start table is diagnostic, not a headline result. The final comparison
protocol lives in [results.md](results.md). Plot violation beside mass, report
the accepted feasible landing, and treat a recorded frame as an objective
evaluation rather than necessarily an accepted line-search step.
