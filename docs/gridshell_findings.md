# The 16x16 gridshell — what the runs measured

The record of what the gridshell experiments establish, with the numbers and the
caveats attached to each. Every mass here is the product of
`experiments/23_gridshell_optimize.py` on a run description named below, and
every answer is stored under `designs/` so a claim can be reopened rather than
re-descended.

Reasoning and rationale for the machinery live in `CHANGELOG.md`; this file is
the results and their standing.

> **⚠ Every mass below predates 2026-08-28 and is not comparable with a current
> run.** On that date the moment demand stopped being `|M_y| + |M_z|` read at
> each axis's own end and became the magnitude of the worse end's moment vector,
> which is what EN 1993-1-1 gives for a circular hollow section (6.2.9.2
> eq. (6.42), and 6.2.9.1(6) eq. (6.41) with alpha = beta = 2). The old
> reduction inflated the demand on this shell by 1.220 on average and 1.885 at
> worst, so **every mass here is too heavy** — the shipped example fell from
> 0.105268 t to 0.082971 t, 21.2%. The *relative* findings below — which
> parametrization wins, how the basis behaves, where the gradient is small —
> were not re-measured and are the part worth trusting; the absolute masses are
> history. `CHANGELOG.md` carries the reasoning under the entry naming the
> reduction.

---

## The structure and the loading

16 rings by 16 spokes on a spherical cap, radius 5000 mm, drawn rise 2000 mm,
closed crown, unbraced quads, boundary ring unhooped. **257 nodes, 496 members**
(256 radial, 240 hoop), 16 pinned supports.

Pressure over the plan by tributary area, three cases carrying the same total:
LC1 uniform, LC2 a drift over one sector, LC3 that drift reflected. The drift
keeps full pressure inside the sector and `drift_factor` of it outside, then
rescales — a redistribution over the whole roof rather than a spotlight, so
every node carries load under every case.

Sections are folded by the full dihedral group: **one diameter per ring per
family**, 31 in all, every member of a ring identical to 0.000e+00 mm.

---

## 1. The headline

At 1 kN/m² of plan, on the pivoted basis with 29 held-plan coordinates:

| route | variables | mass | against the baseline |
|---|---|---|---|
| sizing only | 31 | 0.126869 t | — |
| free heights | 272 | 0.124310 t | −2.0% |
| **end to end** | 60 | **0.077209 t** | **−39.1%** |

Five starts each, all converged. The `sizing only` baseline is verified
configuration-independent — 0.126869 against 0.126868 on a differently
parametrized run, identical to six digits — so all three routes divide by the
same number.

**Quote 39.1% with its load level attached.** See §4.

---

## 2. The parametrization decides what is findable

The three routes search the same structure, the same loads and the same code
check. They differ only in the design variables:

| route | variables | constraints beyond `U <= 1` |
|---|---|---|
| sizing only | 31 diameters | — |
| free heights | 241 nodal heights + 31 diameters | rise and sag as box bounds |
| end to end | 29 force densities + 31 diameters | 482 rise/sag rows, 256 chord-sign rows |

**The free-heights route is strictly advantaged and still loses by an order of
magnitude.** With the plan held, every shape the form finder reaches is a height
field over that plan, so free heights searches a superset of the end-to-end
space — eight times the variables, no compression guard, identical starts,
rounds and iteration caps. All three routes begin from the same drawn cap, to
1.1e-11 mm of geometry and 1.4e-12 mm of seeded diameter.

**And it barely moves the structure.** Nodal heights travel 4.8 mm rms at
1 kN/m² and 14.0 mm rms at 2, against the form finder's 363.7 and 1295.2 mm on
the same cap. Doubling the load triples the displacement and leaves the shape
static, so *the load was too light for shape to matter* is not the explanation.

### Why: reach, not smoothness

Three explanations were measured and two discarded.

- **Not a vanishing gradient.** `dmass/dz` is 3.06e-08 t/mm against 1.66e-04
  for a diameter, which reads as 5,437x — but a millimetre is 0.05% of a height
  and 4% of a tube. Scale-free, the ratio is **122x** at the answer, falling to
  **90x** when the load doubles. Two orders, not four. **Do not quote 5,437x.**
- **Not curvature, and not a kink at the funicular manifold.** Perturb and
  re-size every diameter to `U = 1`: heights cost +9.2 / +40.5 / +120.7% for
  1 / 5 / 15% relative moves — linear, not quadratic — and densities cost about
  the same *absolute* mass per step, 0.011 t at 1%. Heights are only 1.07x,
  1.30x and 1.48x worse as the step grows.
- **Reach.** The lighter design lies **363 mm rms** away in height coordinates;
  a 15% height scatter reaches **238 mm rms**. It is outside the search cloud.
  In density coordinates the same design is found *by* a 15% scatter.

The straight line between the two answers, re-sizing at every step:

| t | mass [t] | max U | rms z move |
|---|---|---|---|
| 0.00 | 0.124379 | 1.002880 | 0 |
| 0.25 | 0.125276 | 1.044784 | 91 mm |
| 0.50 | 0.120570 | 1.023007 | 181 mm |
| 0.75 | 0.087326 | 1.001470 | 272 mm |
| 1.00 | 0.077209 | 1.000000 | 363 mm |

Minus 3% over the first 181 mm with a barrier a quarter of the way, then minus
28% in the next 91. **Flat, then a cliff.** The barrier is understated: `max U`
peaks mid-path because the re-sizing cannot close on the least funicular shapes,
so those masses read too light.

### The same result within one route

The end-to-end problem on an orthonormal (`svd`) and a pivoted, member-named
basis of the **identical subspace** — same reachable designs, same constraints,
same budget, `cond(B)` 1.0 against 16.4:

| basis | coordinates | mass |
|---|---|---|
| symmetric svd | 23 | 0.073013 t |
| full svd | 29 | 0.102264 t |
| full pivoted | 29 | 0.077209 t |

A warm start confirms the lighter design is a local optimum of the
svd-parametrized problem too: SLSQP moves **zero coordinates in one iteration**
when placed there. Ten converged descents across three coordinate systems found
it once from a cold start.

**Every answer came back mirror-symmetric** — 5.8e-10, 6.4e-09, 1.6e-09 mm —
whether or not the mirror was imposed. The symmetry restriction excludes
nothing; it is the *coordinates* that decide the outcome, not the subspace.

---

## 3. Two structural families

The pipeline returns qualitatively distinct designs, and which one it finds is
the parametrization's doing.

| | corrugated | dome |
|---|---|---|
| rise | 2043–2251 mm | 3622–3743 mm |
| worst ring spread | 1105–1783 mm | 1032 mm |
| spread / rise | 0.54–0.79 | 0.29 |
| hoops at the catalog minimum | 240/240 | 96–144/240 |
| mass at 1 kN/m² | 0.073–0.077 t | 0.102 t |

The corrugated family carries as meridian arches deepened over the drifted
sectors and asks nothing of its rings; the dome family recruits hoop action.
Both are verified feasible under all three load cases by the unmodified forward
check.

On the symmetric-basis answer the ridges sit at spokes 3 and 13 — **exactly the
two drift sector centres** — with creases at 5 and 11: mean height 1968 mm over
the drifted sectors against 1246 mm elsewhere. The shape deepens the arch where
the load is heaviest, which is the funicular response, and the creases are what
holding the plan makes it pay in exchange.

---

## 4. The advantage is load-dependent

Same structure, same basis, same budget, pressure doubled. Form finding is
exactly scale invariant — the equilibrium is linear in `q` and in the load, and
the held-plan basis never sees a load — so the reachable shapes and the
multiplicative scatter cloud are unchanged. Only the code check responds.

| pressure | sizing only | free heights | end to end | geometry bought |
|---|---|---|---|---|
| 1 kN/m² | 0.126869 t | 0.124310 t | 0.077209 t | **39.1%** |
| 2 kN/m² | 0.200449 t | 0.190526 t | 0.163294 t | **18.5%** |

**The mechanism is identifiable.** The staged baseline scales **1.58x** —
sublinearly, because capacity grows faster than `d^2`, so members become
stockier and the buckling reduction moves toward unity — while the end-to-end
answer scales **2.11x**. It forgoes that discount because at the lower load a
third of its members sit at the catalog minimum, contributing mass but no
work; doubling the load forces 96 of them into service and the corrugation's
relative amplitude halves, from 0.54 of the rise to 0.29.

**Both numbers belong in any report.** The advantage is real at both load
levels, but part of it at the lower one is a property of the catalog rather
than of EN 1993-1-1. A third point at 0.5 kN/m² would turn this into a trend.

---

## 5. Loadings that do not produce corrugation

Widening the drift was tried, on the hypothesis that a broader asymmetry would
produce corrugation from the loading rather than from a lucky start. It does the
opposite.

| sector | share of plan | peak nodal ratio | worst ring spread | ring-11 band |
|---|---|---|---|---|
| 3 spokes | 18.8% | 1.68x | 1105 mm | 1093 mm |
| 5 spokes, antipodal | 31.3% | 1.52x | 857 mm | 800 mm |
| 9 spokes | 56.3% | 1.28x | 746 mm | 291 mm |

**Corrugation tracks the peak load ratio, not the breadth of the asymmetry.**
With `drift_factor` at 0.5 the inside/outside ratio is pinned at 2:1 whatever
the width, so widening the sector only shrinks the rescale factor and flattens
the shape. The lever that raises local contrast is `drift_factor`, not
`sector_spokes`: at 3 spokes it gives 1.68x at 0.5, 2.56x at 0.25.

The two wide loadings did produce the only **asymmetric** answers on record —
mirror gaps of 630 and 862 mm, where every narrow-drift answer is symmetric to
~1e-9 mm. Breadth buys asymmetry; sharpness buys corrugation.

---

## 6. Standing caveats

- **Every mass is a local optimum from multi-start SLSQP, not a global one.**
  Five starts span 40% of mass on this cap and the lightest design was found
  once in ten converged descents. Report the parametrization and the start
  budget alongside any number.
- **Scattered starts are not comparable across routes.** A 15% scatter moves
  the shape 68.9 mm rms in density coordinates and 238.1 mm rms in height
  coordinates. The nominal starts are identical; compare best against best.
- **`free heights` carries no compression guard**, so its heavier landings may
  include tension nets a guarded search would refuse. Its best landing came
  from the shared nominal start, which is the one the comparison rests on.
- **Sections are folded polar.** On a design whose geometry answers a one-sided
  load this is a real restriction — the shape can respond to the asymmetry and
  the steel cannot — and it is likely part of why the half-drift case bought
  only 20.6%.
- **Self-weight coupling is ignored** and the stated pressure is unfactored.
- **`polar_heights` is inert whenever `subspace.symmetric` is false**, because
  heights fold only under that flag. A run that reads `polar_heights: true`
  with an unfolded basis gives the free-heights route all 241 heights.
- The re-sizing operator used in §2 is an approximate fixed point — 14 passes
  of `sqrt(U)`, exact on the funicular manifold and 0.29% over off it — so the
  barrier on the height-space slice is understated.

---

## 7. Where each answer lives

| run description | store | routes held |
|---|---|---|
| `gridshell_16.yaml` | `492e4e30a7606d37` | end to end |
| `gridshell_16_pivoted.yaml` | `a9e85332bd06767d` | all three |
| `gridshell_16_pivoted_2x.yaml` | `27bd0ca3da795705` | all three |
| `gridshell_16_fullbasis.yaml` | `00c63eb01136be04` | end to end |
| `gridshell_16_free.yaml` | `e69799442b9d7d95` | end to end, sizing only |
| `gridshell_16_halfdrift.yaml` | `64bd1149c9de02e9` | end to end, sizing only |
| `gridshell_16_quarter.yaml` | `1639181d047cafea` | end to end |

Each has a `_view` twin that opens its answer in the viewer without descending
it again. `gridshell_16_pivoted_all.yaml` reads all three routes back and
rebuilds the full comparison report and both figures in seconds.

`designs/3a5846c1444d3ed9.npz` is orphaned — a rise-capped run at 0.110630 t
whose description was edited in place and no longer exists.
