# Changelog

## Unreleased

### P1 — EC3 core

- **`normax/ec3/section.py`**: closed-form circular-hollow-section geometry —
  `thickness`, `diameter_inner`, `area`, `second_moment`, `radius_of_gyration`,
  `modulus_elastic`, `modulus_plastic`. Every property takes the outer diameter
  and the diameter-to-thickness ratio, and every one is a monomial in the
  diameter times a function of the ratio alone. The ratio stays a free
  argument rather than being pinned at `90ε²` inside the module: fixing it is a
  sizing decision, and the validation fixture needs `d/t = 24.45`. The two
  section moduli are unused by the axial MVP but are tabulated in the guide's
  Figure 6.21, so they ship with real targets rather than untested.
- **`normax/ec3/classification.py`**: EN 1993-1-1 Table 5.2 sheet 3, tubular
  sections. `classify` counts how many of the three limits a ratio exceeds and
  adds one, so it is branch-free and traces cleanly under `jit` and `vmap`
  without a `lax.cond`. It takes the ratio rather than a diameter, because the
  class depends on `d/t` and the grade and on nothing else. A ratio sitting
  exactly on a limit takes the class below it, matching the standard's
  inclusive `≤`.
- **`normax/ec3/resistance.py`**: §6.2.3 (Eqs. 6.6, 6.7 and their minimum),
  §6.2.4 (Eq. 6.10), §6.3.1.3 (Eq. 6.50, both forms), §6.3.1.2 (Eq. 6.49 and
  the unnumbered `Φ` beneath it), §6.3.1 (Eq. 6.47). **The clause layer takes
  section properties, not a diameter.** Eq. 6.10 is a statement about area, and
  keeping the transcription free of CHS geometry is what lets the guide's own
  §6.2 worked examples — a flat bar and a UKC — run through it unmodified as
  fixtures. Composition with `section.py` happens one layer up, in P2's sizing
  map. Eqs. 6.11/6.48 (Class 4) and 6.51–6.53 (Class 4 slenderness, torsional
  modes) are deliberately absent: the fixed ratio pins the section at the
  Class 3 boundary, and a CHS is closed and doubly symmetric.
- **`χ` needs no guard inside its square root.** `Φ² − λ̄²` bottoms out around
  0.093 across all five buckling curves for `λ̄ ∈ [0, 5]`, so it never
  approaches zero in the reachable range. A `clip` there would be a silent
  change to the gradient, which matters because this term is on the
  differentiated path.
- **Tests: 613, written before the implementations.** Six files. The primary
  fixture chains classification, cross-section resistance and buckling through
  one worked example; a second covers the guide's two §6.2 examples; three
  cover each module's properties; one holds the Blueprints comparison.

### P1b — N+M scope expansion

Axial-only was not defensible once asymmetric load cases enter, and it left T2
barely motivated: if T3 only consumes axial force, form-finding already supplies
it and a frame solver earns nothing. The check now consumes moments.

- **`normax/ec3/interaction.py`**: EN 1993-1-1 Annex B method 2 — the `k_ij`
  factors, the equivalent uniform moment factor, and the 6.3.3 member check.
- **`normax/ec3/resistance.py`**: §6.2.5 Eqs. 6.13/6.14, the reduced plastic
  moment of §6.2.9.1(5), and the biaxial resultant of §6.2.9.1(6).
- Both the plastic and the elastic branches of Table B.1 are implemented,
  selected by the configured `d/t`. That selection is **static**, so it is an
  ordinary Python choice and no branch on a traced value.

**Two errors caught before they shipped.**

- **`d/t = 90ε²` makes every member Class 3, not Class 1 or 2.** That was chosen
  when the scope was axial-only, where Classes 1–3 all use the gross area so it
  cost nothing. With bending it costs 24.6% of bending capacity (`W_el` instead
  of `W_pl`; the CHS shape factor is 1.326), makes `M_N,Rd = M_pl(1 − n^1.7)`
  inapplicable in favor of the Eq. 6.42 stress check, and switches which column
  of Table B.1 applies — a different formula *and* different couplings
  (`k_yz = k_zz` not `0.6k_zz`; `k_zy = 0.8k_yy` not `0.6k_yy`). The ratio now
  stays a config parameter with both branches built, and 70ε² against 90ε² is an
  experiment rather than an assumption.
- **Eqs. 6.61 and 6.62 do not collapse into one equation for a CHS**, contrary
  to an earlier draft of `docs/clauses.md` and of the roadmap. They share the
  axial term but weight the moments oppositely, so they agree only when the two
  moments are equal — at `M_y` = 100, `M_z` = 20 kNm they give 0.91 and 0.72.
  It is still one check, taking the larger of the two.

**The bisection precondition is now established, not assumed.**
`tests/test_sizing_monotonicity.py` composes section → resistance → interaction
and asserts that the utilization is **strictly decreasing in the diameter**,
across six force combinations and both class branches, with exactly one
crossing of unity. CLAUDE.md §4 argued this for the axial-only case from
`A ∝ d²` and `i ∝ d`; bending adds terms, so the claim had to be re-derived
rather than inherited. It holds: the moduli rise, the reduction factor rises,
and the axial ratios fall so the interaction factors fall with them. P2's
`lax.while_loop` bisection is safe.

**Governing diagnostics.** `governing_equation` reports which of 6.61/6.62 was
taken and `cap_is_active` whether the bound on an interaction factor binds.
Both are non-differentiable, reported beside the utilization rather than
through it, and both are shared with the utilization via one private helper so
they cannot drift out of step. The cap turns out to bind at exactly `λ̄ = 1` on
both branches — the plastic slope passes its 0.8 bound and the elastic `0.6λ̄`
its 0.6 bound at the same place.

**Blueprints, after the expansion.** Its relevance grew from three formulas to
sixteen: Eqs. 6.13 and 6.14 now oracle the two moment resistances, and Eq. 6.41
oracles the biaxial collapse — it takes `α` and `β` as arguments, so feeding it
`α = β = 2` with equal resistances checks the CHS reduction to a resultant
against an independent implementation, by verdict and by value. It remains
blind to §6.3 entirely, so `interaction.py` has no oracle but the ECCS worked
examples, and it has no circular-hollow reduced moment either.

**A frame checked against its published member forces.**
`tests/test_worked_example_frame.py` takes the ECCS manual's 47 m portal frame
(Design Example 2) and Example 5.2's segment-by-segment rafter check, feeds the
book's own member forces and resistances in, and confirms our assembly of
Eqs. 6.61 and 6.62 reproduces its numbers. Nothing here analyses a frame —
T1 and T2 do not exist — but it does establish parity on the member check
against a real structure rather than an isolated member.

That required splitting §6.3.3 from Annex B: `checks` now takes the four
interaction factors as arguments and `interaction_factors` derives them from
Table B.1, with `utilization` composing the two. They are different clauses,
and a source that publishes its own factors can now be reproduced without also
adopting the table they came from — which matters here, since the frame is an
I-section using Table B.2 while we read B.1's hollow row.

Two scope limits, both stated in the file: the frame is planar, so only the
uniaxial path is exercised, and one segment asserts **0.64 where the book
prints 0.62** — erratum E5, recomputed from the book's own inputs.

**Tolerances on printed utilizations are absolute, not relative.** These are
printed to two decimal places, so the honest band is the rounding half-width.
Our 0.4653 against a printed 0.47 is correct rounding yet sits 1.01% away and
fails a 1% relative band. Same lesson as `Φ` earlier: relative tolerance is the
wrong instrument for a small fixed-decimal number.

**Two interpretations, recorded as such.** EN sends a CHS to Table B.1, but that
table lists only I-sections and RHS-sections and neither textbook says which row
a circular tube takes; we read the RHS row, and Karamba3d's implementation
resolves it identically. And `C_m` comes from the linear row of Table B.3, which
is exact rather than approximate here because loads are applied at nodes only,
so no member carries load along its span.

The resultant `√(M_y² + M_z²)` is exact at cross-section level, where both
exponents are two, and **inadmissible** in 6.61/6.62, whose terms are linear —
confirmed absent from both books and every NCCI they cite.

### Corrections to `docs/clauses.md`

`references/9780727741721.pdf` (Gardner & Nethercot, 2nd edn) was added to the
repo, which made it possible to check the spec against the source rather than
against a transcription. Three things came out of that.

- **The validation fixture is Worked Example 6.7, not 6.2.** Example 6.7,
  "buckling resistance of a compression member" (pp. 61–63), is the CHS
  244.5 × 10. The guide's Example 6.2 is "cross-section resistance in
  compression" for a 254 × 254 × 73 UKC — a different section, a different
  clause, no buckling. Both `docs/clauses.md` and `ROADMAP.md` carried the
  wrong number. The test filenames are now neutral of example numbers so a
  future edition renumbering cannot invalidate them again.
- **The guide misevaluates its own Class-1 limit, and the transcription was
  faithful.** Page 62 prints `Limit for Class 1 section = 50ε² = 40.7`. But
  40.7 is `50ε` (`50 × 0.8136 = 40.68`); `50ε²` is 33.10. Table 5.2 sheet 3 on
  p. 41 prints `d/t ≤ 50ε²`, and that table's own `ε²` row gives 0.66 for S355,
  so the formula is right and only the arithmetic is wrong. The initial reading
  here — that the spec had mis-transcribed a correct book — was backwards, and
  the errata note now says so. Nothing structural changes: `d/t = 24.45` is
  Class 1 at either limit. Implemented and asserted as 33.10.
- **Two open items closed.** `γ_M0 = γ_M1 = 1.00` come from UK National Annex
  clause NA.2.15, cited across five of the guide's worked examples. Table 6.5
  gives hollow sections curve `a` hot finished and curve `c` cold formed, with
  `a0`/`c` respectively at S460 — both previous guesses were right. `E =
  210 000` is clause 3.2.6. Equation numbers 6.5, 6.9 and 6.46 stay open: the
  guide reproduces none of them, so only the standard itself would settle them.

### Full audit of the guide's worked examples

Every worked example in the book was transcribed and recomputed, to find more
validation points for the forward pass before the pipeline phases begin.

- **There is no multi-member benchmark anywhere in the guide.** Three
  independent sweeps agree: every labeled example (5.1, 6.1–6.10, 7.1,
  13.1–13.3) is a single cross-section or a single isolated member. Chapter 12
  has no worked example, Chapter 14 is tables with no load take-down, and
  Chapter 11 contains no numerals at all. Examples 6.7 and 6.10 describe their
  member as sitting in a multi-storey frame, but the frame action is an input.
  This is structural, not an oversight: EN 1993-1-1 checks resistances *given*
  member forces and never computes them. The guide can validate T3 exhaustively
  and T1/T2 not at all. Frame-level validation must come from elsewhere —
  `smax/docs/benchmarks.md` already catalogs the candidates, of which
  Ziemian & Ziemian (2021), *Data in Brief* 39:107564 is the strongest: 22
  planar frames, 1 to 40 storeys, machine-readable geometry, per-member
  reference moments and `α_cr`, CC BY.
- **`tests/test_worked_examples_buckling.py`**: Examples 6.9, 6.10 and 13.3
  added as fixtures, plus 6.8's lateral-torsional values as an algebra check.
  This takes the published `χ` points from **one to eight**, spanning curves a,
  b and c over a slenderness of 0.23 to 1.42, and adds seven `N_cr` points
  across four orders of magnitude. Buckling was by a wide margin the thinnest
  part of the forward pass: one published value, with everything else resting
  on property tests that fix the curve's shape but no point on it.
- No new modules were needed. That is the payoff from `resistance.py` taking
  section properties rather than a diameter — none of these members is a CHS.
- The guide's remaining examples (shear, bending, bending-plus-axial
  interaction, lateral-torsional buckling, the Annex A/B interaction factors,
  and I-section classification) all require modules outside the scope set in
  CLAUDE.md §3, and were left alone.

### The guide's own arithmetic is unreliable

The `50ε² = 40.7` error found earlier is not isolated. The audit turned up
**eleven printed numbers that are wrong**, plus about fifteen substitutions
that do not evaluate to their own printed answers. All are catalogd in
`docs/clauses.md`. The two worst:

- **Example 6.5, p. 50** computes `ρ` from `V_pl,Rd = 689.2 kN` when the
  example established `664.3 kN` three lines above. `ρ` should be 0.34, not
  0.27, and `M_y,V,Rd` 380.9 kN m, not 386.8.
- **Example 6.3, Figure 6.9** prints an elastic section modulus of
  2 536 249 mm³ that exceeds the section's own *plastic* modulus of
  2 352 736 mm³ — impossible. The text uses the correct 2 124 800.

The systematic habit is that the guide computes with unrounded intermediates
while printing rounded ones, so a printed substitution is usually false even
where its answer is right. Two consequences for this repo: no guide value is
asserted tighter than 1%, and no printed intermediate is ever fed back in.

This matters beyond housekeeping. The project's claim is that a design code is
a normative text rather than a solver — no derivatives, no execution, nothing
that checks it. An authoritative commentary carrying eleven arithmetic errors,
one of them physically impossible, is the sharpest available evidence for that
claim, and belongs in the writeup.

### Test tolerances, and two properties that are weaker than they look

- **Every fixture row is asserted twice**, against the guide's printed column at
  1% and against the closed form at 0.5%. A single 0.5% assertion against the
  printed column would have failed: the guide rounds intermediates to 2 s.f.,
  which puts `Φ = 0.744221` a full 0.57% from its printed 0.74. Asserting only
  the closed form would have dropped the independent check entirely, so both
  are kept at their respective honest tolerances.
- **`χ` is strictly decreasing only above `λ̄ = 0.2`, and the curve ordering
  `a0 > a > b > c > d` is strict only there too.** Below the offset the `χ ≤ 1`
  cap binds and all five curves are flat at exactly 1.0, which is §6.3.1.2(3)
  working as intended. Asserting strictness across the whole range would have
  asserted something false, so the tests state the flat region as its own
  property instead.
- **`χ` approaches the Euler bound `1/λ̄²` from below, and slowly** — the ratio
  is 0.960 at `λ̄ = 5` and still only 0.9996 at `λ̄ = 500`. The asymptote test
  therefore evaluates at `λ̄ = 1000`.

### Blueprints as an oracle, not a source

Blueprints is LGPL-2.1 and this repo is Apache-2.0, so it appears in exactly
one test module and is called, never read. Its formula classes subclass
`float` and work in mm/N/MPa, so they compare directly. Coverage was verified
by listing its EN 1993-1-1 chapter 6: **§6.3 member buckling and cross-section
classification are both absent** — the chapter jumps from `formula_6_45` to
`formula_6_54`.

**Corrected 2026-08-09.** That gap is real but it is *only* §6.3. The original
audit read it as sparse coverage of §6.2 as well, and reported only Eqs. 6.6,
6.7 and 6.10 as present. In fact §6.2 runs almost unbroken from 6.1 to 6.45, so
**Eqs. 6.18, 6.42, 6.43 and 6.44 were all available and went unused** — including
6.18, against which the shear work of P2 should have been checked under invariant
2 and initially was not. Oracle tests for 6.18 and 6.44 have since been added. `χ`, `λ̄` and the class
limits therefore have no second implementation to check against and rest on the
guide alone. That gap is worth stating in the writeup, since it is the gap this
package fills.

Its 259-entry CHS profile table also carries the fixture section itself, giving
`section.py` an independent geometric check across five profiles. The table's
properties are polygon-meshed rather than closed form, and the discretization
error tracks diameter rather than wall ratio: ~1e-5 at CHS 244.5, ~2.5e-3 at
CHS 21.3. The 5e-3 tolerance there is set by Blueprints' mesh, not by us.

## P2 — The fully-stressed sizing map

### The map, and why it is differentiated forward rather than backward

`normax/ec3/sizing.py` solves for the diameter at which the utilization is
exactly one, and `normax/ec3/adjoint.py` derives the same sensitivities on paper
as an independent oracle. Together they close the gap between checking a member
and sizing one.

**`CLAUDE.md` §4 and the ROADMAP both specify `custom_vjp`. We used
`custom_jvp`.** The reason is the one `smax` records in its own roadmap for the
same switch: *"`custom_vjp` blocks forward-mode AD, which is needed for
`jax.jacfwd`… the JVP rule is simpler (one solve vs. adjoint solve + outer
product) and JAX's auto-transposition provides the VJP for free."* For a scalar
root the tangent rule is a single line — the drift of the check at a frozen
diameter over its slope in the diameter, negated — and JAX transposes it into
the adjoint automatically. T3 therefore supports `jacfwd` as well as `grad`,
which matters when P5 lands a forward-mode DDM backend in T2.

Verified before a line of the module was written, on a toy of the same shape:
forward, reverse, `jacfwd`, `jacrev`, `jit`, `vmap` and `check_grads` in all
three mode combinations, exact to 1e-16, with a `lax.fori_loop` inside the rule.
The loop is primal-only and partial-evaluates to a constant, so transposition is
untroubled by it.

### The residual is the larger of two checks the standard requires separately

`max(§6.3.3 member, §6.2.9 cross-section)`, with the member check switched off in
tension because 6.3.3 covers compression only. Neither bounds the other: with an
equivalent uniform moment factor below one, 6.3.3 permits a moment 6.2.9
refuses. One bisection covers tension and compression alike, so the closed-form
tension branch the ROADMAP anticipated was not needed — though it survives in
`adjoint.py` as an oracle.

Monotonicity was measured rather than assumed, over eight action cases by two
class branches by two buckling lengths: strictly decreasing in all 32, exactly
one root wherever the root is interior.

### Two defects caught before they could hide

**`M_res / M_N,Rd` loses the squash check.** Writing 6.2.9.1 as a quotient
returns zero at zero moment however large the axial force: a member at 95.5% of
squash reads as completely unutilized. It also has a pole exactly where axial
force alone exhausts the section, goes negative beyond it — reporting a section
overloaded three times over as safe — and loses precision as it is approached,
drifting off unity by 18× the invariant's tolerance.

The clause rearranges. `M_res ≤ M_pl,Rd(1 − n^1.7)` divided through by the
*unreduced* plastic moment is `n^1.7 + M_res/M_pl,Rd ≤ 1`. Same inequality, same
root, and the identity `u_add − 1 = (1 − n^1.7)(u_ratio − 1)` holds to one unit
in the last place, so the positive prefactor cancels in the implicit derivative
and the adjoint is unchanged. The sum is finite and strictly increasing
everywhere, and at zero moment it reduces to `n^1.7`, whose root is an axial
ratio of one — Eq. 6.10 recovered for free.

**An unguarded square root poisons every axial-only gradient.**
`moment_resultant` at the origin gave `(nan, nan)`, and — the part that matters
— the NaN survived losing a `jnp.maximum`, because a cotangent of zero times an
undefined value is still undefined. That is exactly the path a member with no
moment takes, so the single-strut gradcheck, the milestone meant to de-risk the
whole phase, would have failed on it. Guarded with a double `where`; the
gradient at the origin is now the symmetric subgradient of zero, which is the
only rotation-invariant choice at a cone point.

### The catalogue minimum had to come out of the search

Folding the smallest available tube into the bisection bracket let the search
stop at a diameter where the check is *not* satisfied — and the implicit
function theorem is only valid at a root. Every sensitivity there was fabricated:
plausible magnitudes, correct signs, no relationship to the truth, and a
utilization silently at 5e-5 instead of one. The floor is now applied outside the
solved map as an ordinary `jnp.maximum`, whose own tangent rule routes the
sensitivity to the floor and zeroes the actions — measured as exactly 0 for the
actions and exactly 1 for the minimum, against a truth of the same.

### Results

`experiments/01_single_strut_gradcheck.py` — four oracles over ten struts:
forward tangent, reverse transposition, closed form, central difference. Worst
disagreement **3.38e-9** against a 1e-8 target, and forward, reverse and closed
form agree to every printed digit; the whole residual is finite-difference
truncation. Utilization at the root, **1.0 ± 2.8e-15**.

`experiments/02_pipeline_gradcheck.py` — the full interaction. Worst **5.57e-8**,
forward-to-reverse gap **0.00e+00**, and removing the moments reproduces the
axial answer to *exactly* zero on both class branches.

`experiments/05_class_ratio_sweep.py` — the question `docs/clauses.md` declined
to call, answered. **The crossover sits near `M_y = 72.5 kNm`** for a 6 m member
at 600 kN: below it the Class 3 limit is lighter by up to 9%, above it Class 2's
shape factor of 1.326 wins by up to 3.4%. The wall proportion is a genuine design
variable, not a fixed choice. The same script sweeps `V_Ed/V_pl,Rd` for open item
0d and finds it peaks at 0.12, so excluding §6.2.6 remains honest.

`experiments/06_load_case_aggregation.py` — the smooth envelope over load cases,
in log-diameter. Excess falls 23.9% → 0.000% as sharpness anneals 5 → 500, always
from above, so every intermediate design satisfies the standard. All three cases
keep a live gradient where a hard maximum gives exactly one.

### The standard is discontinuous, and no relaxation fixes that

Three `max`es appear and need three different treatments. The caps on `χ` and on
the interaction factors, and the two comparisons, are C⁰ kinks: the implicit
function theorem handles them, since it needs the check differentiable only *at
the root*, and whichever branch is active there is the one `jax.jvp`
differentiates. The envelope over load cases is a discrete maximum and genuinely
earns its Kreisselmeier–Steinhauser relaxation.

The tension gate is neither. Measured, at `M_y = M_z` and `C_m = 1`: utilization
jumps by **1.131** on the plastic branch and **1.414** on the elastic one, worth
about 26% in area. Uniaxial bending is continuous. A sigmoid across a jump gives
a large bounded derivative that is wrong throughout the transition band — worse
than the subgradient — so the gate stays exact and `governing` flags it.

**Nothing inside the check is smoothed, deliberately.** Softening `χ ≤ 1` would
give `χ > 1`, a capacity above squash, and a diameter that does not satisfy
EN 1993-1-1. Root-finding on the exact clause buys exact compliance; the implicit
function theorem buys the gradient. The two do not have to be traded.

### Also delivered

- §6.2.9 cross-section utilizations for both class branches, and §6.2.6 shear
  (`A_v = 2A/π`, Eq. 6.18), verified against the guide text rather than memory.
- **Both readings of Eq. 6.42 are implemented**, selected by `resultant=` and
  threaded through the sizing map, so the disagreement is a number rather than an
  argument. The guide says 6.2.9.2 permits *"only a linear interaction of
  stresses"*; the ECCS says the stress is *"evaluated by an elastic stress
  analysis"*. Checked against both implementations: **Karamba is silent** (no
  §6.2.9 anywhere in its EC3 source) and **Blueprints is neutral on 6.42** (it
  takes the stress as an input) but writes the Class 4 analogue Eq. 6.44 as an
  explicit **three-term linear sum**, which is now an oracle for our sum variant.

  The default stays the resultant: on a circular perimeter that is the actual
  peak stress, and Eq. 6.44's *minimum* effective moduli are the drafting of a
  general-section envelope for shapes whose extreme fibre sees both moments,
  which a circle's does not.

  **The choice matters far less than first feared.** It moves nothing wherever
  Eq. 6.61 governs, because 6.61 already sums the two moments linearly, and
  nothing under uniaxial bending. Measured: 0.00% across a compression sweep from
  10 to 80 kNm, reaching 0.93% in diameter only at 160 kNm. It bites at 12.25% in
  diameter and 26.0% in area **only in pure bending**, and 9.0%/18.8% for a
  tension member — that is, only where the cross-section check governs.
- `governing` reports five limit states; `end_moments` reduces a pair of end
  moments to a design moment and Table B.3's factor, exact under nodal loading.
- `check_grads` runs on arguments rescaled to order one. Its step is *absolute*,
  so a moment of 1.5e7 N·mm is perturbed by a relative 1e-12 and its central
  difference is pure cancellation.
- At exactly zero moment the two corners resolve differently — the member check
  reports a one-sided slope, the resultant a symmetric zero — so the total
  understates the right-hand slope. Confined to the origin, pinned by a test.

### The T2 to T3 contract, for P3

`smax` is the JAX analysis backend, not `sax`. `element_forces(...)` returns
`nx` **tension-positive** and `my`/`mz` in the classic beam-diagram convention,
sampled at `num_samples=2` — exactly the two end moments `end_moments` wants, so
Table B.3's linear row stays exact. **A unit adapter is required**: `smax` works
in coherent SI (N, m, Pa) and `normax` in mm/N/N·mm⁻².

## OpenSees DDM spike

Run 2026-08-09, ahead of the Aug 12 milestone. `experiments/07_opensees_ddm_spike.py`
reproduces every number below; `openseespy` 3.8.0.0 is a `spike` optional extra, so
CI never installs it.

### The answer splits by dimension

CLAUDE.md §9 offered three outcomes. Both (a) and (b) landed, in different halves
of the problem:

- **In 2D, DDM reaches a nodal coordinate.** `parameter(tag, 'node', n, 'coord', d)`
  binds and returns the right number. Against central differences on a kinked
  cantilever, worst relative error **7.4e-9** across every (parameter, DOF) pair.
  This is outcome (a) — the best case — and it covers P4's 2D arch entirely.
- **In 3D, it does not.** Beams return identically zero for all three coordinate
  directions. The truss is worse: `coord3` is always zero, while `coord1` and
  `coord2` return values that are **wrong rather than absent** — off by 2.5x and
  1.9x on a tripod whose bars are not axis-aligned. This is outcome (b).

Per the agreed stopping rule the spike stopped at the finding rather than starting
a workaround inside the timebox.

**The decision it fed, taken 2026-08-09: OpenSees is the 2D swappability
demonstration, and `smax` carries anything 3D.** The alternative — wrapping the
OpenSees solve in a `custom_jvp` and supplying the geometric tangent DDM lacks —
is not being built. It would mean re-deriving `∂K/∂x` per element type to recover
a capability the primary backend already has, and the 2D case needs none of it:
`∂N/∂xyz` from JAX autodiff against the same quantity from C++ DDM is exactly the
backend-agreement plot P5 wants, available with no extra machinery. The
restriction costs the demo nothing, because the demo was always going to be the
2D arch.

### `elasticBeamColumn` carries no DDM at all

The most actionable fact, and it is invisible from the API: `parameter()` accepts
`E`, `A`, `Iz`, `Iy`, `G`, `J` on `elasticBeamColumn`, returns a tag, and
`getParamValue` reads the bound value back correctly. Every resulting sensitivity
is then **identically zero**, in 2D and 3D alike, with no warning. Registration
succeeding says nothing about `getResistingForceSensitivity` existing — exactly the
trap §9 anticipated.

`SRC/element/elasticBeamColumn/ElasticBeam3d.cpp` on current `master` is the whole
story in one file: `setParameter` accepts `"E"`, `"A"`, `"Iz"`, `"Iy"`, `"G"`,
`"J"`, `"Avy"`, `"Avz"`, `"releasez"`, `"releasey"`, and the class implements
**no** `getResistingForceSensitivity`, no `commitSensitivity`, no
`activateParameter`. The registration path is fully built and the differentiation
path does not exist, so nothing can warn.

`dispBeamColumn` and `forceBeamColumn` with `section('Elastic', ...)` do carry it:
`E`, `A`, `I`/`Iz`, `Iy` all correct to ~1e-9, in both dimensions, with forward
displacements matching the analytic cantilever to 4e-16. **T2's OpenSees backend
must be built on those, never on `elasticBeamColumn`.** The section route is not a
fibre approximation — it is the same linear-elastic beam.

This matches the documented support. OpenSees' own guidance is to check for
`getResistingForceSensitivity` in the element rather than trust the command
surface, and the maintainers describe DDM as enabled for "the basic element
formulations like `dispBeamColumn`, `forceBeamColumn`, and a handful of solid
elements, as well as fibre sections" — a list `elasticBeamColumn` is absent from.
The OpenSeesPy `parameter` page documents no parameter surface at all: "the
specific set of parameters ... will be added in the future."

### The 3D numbers are wrong, not noisy

Worth stating separately because a silently wrong gradient is the worst failure
mode available. The central differences are stable to eight digits across step
sizes from 1e-4 to 1e-8, so they are the trustworthy side of the disagreement.
The one 3D case DDM gets exactly right is a single bar aligned with the axis being
perturbed (ratio to the closed form 1.00000000).

**The source says why, and it is unfinished work rather than a design boundary.**
`SRC/element/truss/Truss.cpp` computes the direction-cosine derivatives for a
nodal-coordinate parameter from a planar formula, with the third term commented
out — verbatim, in both `getResistingForceSensitivity` and `commitSensitivity`:

```c
double dx = L*cosX[0];
double dy = L*cosX[1];
//double dz = L*cosX[2];
```

That single comment produces both observed failures: `coord3` is identically zero
because `dz` is never used, and `coord1`/`coord2` are wrong for any bar not lying
in the xy-plane because the remaining formula assumes one.

**No element swap fixes this**, which is the fact worth carrying into P5. For beams
the gap sits one layer below the element, in the coordinate transformation they all
delegate geometry to. `LinearCrdTransf2d.cpp` implements the full family —
`getGlobalResistingForceShapeSensitivity`, `getdLdh`, `getd1overLdh`,
`isShapeSensitivity`, `getBasicTrialDispShapeSensitivity`. Its 3D counterpart
implements **none** of them, carrying only `getBasicDisplSensitivity`, and neither
`PDeltaCrdTransf3d` nor `CorotCrdTransf3d` fills the gap. Every 3D beam element in
OpenSees therefore has no geometric sensitivity to inherit, regardless of
formulation. Checked against current `master`, so there is nothing to upgrade to.

Berkeley's DDM documentation lists geometry among the supported parameter
categories and documents `parameter $tag node $nodeTag coord $dir`, so the
2D-only reach is a gap in the implementation rather than a stated limitation.

### `∂N/∂xyz` is the quantity that matters, and it narrows the element choice

`sensNodeDisp` gives `∂u/∂xyz`. T3 consumes member forces, not displacements, so
the load-bearing quantity is `∂N/∂xyz` and `∂M/∂xyz` — one step further down the
chain, reached through `sensSectionForce`, the only element-level sensitivity
command in the surface. Checked against central differences of
`eleResponse(ele, 'section', k, 'force')` on the same kinked cantilever:

- **`forceBeamColumn`: correct.** Worst relative error **7.8e-8** over every
  (parameter, element, section, dof) entry — axial and moment alike.
- **`dispBeamColumn`: wrong.** Worst relative error **11.9**, with entries an
  order of magnitude out (`-4.85e4` against a difference quotient of `-3.77e3`).
  The two element types produce *identical* difference quotients, so the
  disagreement is in `dispBeamColumn`'s DDM, not in the reference.

Plausibly this is the force-based formulation carrying section forces as its
primary interpolated quantity while the displacement-based one recovers them
through the section stiffness. Whatever the cause, **T2's OpenSees backend should
use `forceBeamColumn`**. `dispBeamColumn` remains fine for `∂u/∂θ` and for section
properties, which is all the earlier passes exercised — it is specifically the
coordinate-to-section-force path that fails.

Two hazards in `sensSectionForce` worth pinning, both of which produced confident
nonsense before being understood:

- It returns **the section vector starting at the requested dof**, so element `0`
  is the value asked for. Indexing `[dof-1]` silently returns a neighbouring
  component — which is what made the first run of this check read as broken.
- Passing a **parameter tag that was never registered segfaults the process**:
  exit 139, no traceback, no output. Not an exception, not a warning.

### Pseudo-loads are not reachable, and matter less than assumed

Step 4 asked whether `P_i = ∂f/∂θ_i − (∂K/∂θ_i)u` can be read without a solve,
which would buy reverse mode over OpenSees at O(1). **No.** `printB('-ret')`
returns the converged residual — 1.07e-8 against an applied load of 4.9e4 — under
`-computeAtEachStep`, `-computeByCommand` and no sensitivity mode at all.
Reconstructing `P_i` as `K (du/dθ_i)` works but presupposes the solve it would
replace, and nothing in the 237-command surface exposes `formSensitivityRHS`.

The motivation is also weaker than it looked. DDM reuses one factorization, so each
extra parameter costs a back-substitution, not a solve: **~6–12% of a full solve
per parameter**, flat in parameter count at every size measured (0.0036 ms/param at
57 DOF, 0.036 at 297, 0.41 at 1197). Against finite differences that is 7x to 17x,
widening with both model size and parameter count.

Two numbers for the P5 scaling plot, which wants "T2's VJP scales with parameter
count, T1's and T3's don't": at 400 elements and 798 parameters the sensitivity
sweep is **330 ms against a 6.5 ms solve** — 50x the forward cost, still 17x
cheaper than the 5.6 s finite-difference equivalent. At P4's scale (20 elements,
57 DOF, 38 parameters) the whole DDM sweep is 0.21 ms and the FD fallback 2.5 ms,
so **the fallback is affordable outright** if the backend ever needs it.

### Four ways the test itself lied first

Recorded because each produced a confident, wrong reading of OpenSees:

- A parameter registered on element 1 must be central-differenced by perturbing
  **element 1 alone**. Sharing one property across the model made correct
  sensitivities read as `WRONG`.
- `G` computed as `E/(2(1+ν))` makes the difference quotient for `E` drag `G` with
  it, so the torsional DOF disagrees. Likewise writing one `I` into `Iz`, `Iy` and
  `J` at once. Both made working 3D section derivatives read as broken.
- The second moment is named **`I` in 2D and `Iz` in 3D**. The wrong name binds to
  nothing and is indistinguishable from a missing derivative.
- Judging "informative" by a threshold scaled with the step size skips genuine
  `∂u/∂E` values of order 1e-13 and scores a **vacuous pass**. The natural scale is
  `u/|θ|`.

Every conclusion above survived a rerun after all four were fixed.

## P3 step 1 — the T1 to T2 handoff

`jax-fdm` finds the shape and `smax` analyses it. They exchange a geometry and
nothing else — no prestress, no initial member forces — so the axial forces that
come back are `smax`'s own product and their agreement with `q · L` is a
prediction rather than an identity. `experiments/08_arch_formfind_analyse.py`
runs it and `tests/test_equilibrium_consistency.py` gates it.

### The dependency question, and the interim guard

`jax-fdm` is on PyPI; **`smax` is not yet — it is being published before the
deadline** (decided 2026-08-09). Until then both live in a **`pipeline`
dependency group**, `smax` as an editable path source, and CI installs
`--group dev` alone and never sees them.

`tests/conftest.py` drops the pipeline tests from collection when either import
is missing. **1441 cases in CI, 1470 with the group installed** — the difference
is `tests/test_equilibrium_consistency.py`.

The guard is a hand-maintained filename list, which is safe because **omitting a
file turns CI red at collection rather than passing quietly**: importing
`normax.analysis` without `smax` raises `ModuleNotFoundError` before any test
runs. Verified in a clean `pip install . --group dev` environment. `ruff check`
and `ruff format --check` are unaffected either way, since neither imports.

Two things to do when `smax` goes public: delete the marked block in
`tests/conftest.py` and move both packages into the project dependencies. Note
`jax-fdm` caps at Python `<3.14` while `normax` allows `<3.15`, so that move
needs a marker or a narrower `requires-python`. Independently of all this, each
Tesseract carries its own `tesseract_requirements.txt`; the group only exists
for running the pipeline in process.

### A planar arch on pinned supports alone is a mechanism

Two pinned supports restrain translation and nothing else, so **rotating the
whole arch about the line joining them strains no member and moves no support.**
That mode has zero energy, the free-free stiffness block is singular, and
`smax` returns **nan rather than a plausible wrong answer** — which is the good
outcome, since a plausible wrong answer would have been believed.

Restraining the one translation normal to the plane removes it. The two
out-of-plane rotations are left free: they are unexcited by an in-plane load and
come back **exactly zero**, and restraining them changes the in-plane result in
no digit. `normax.analysis.fixities` takes the normal axis, or `None` for a
structure that occupies all three dimensions.

This does not touch `CLAUDE.md` §3. The **supports** are still pinned, with the
in-plane rotation free at the base, so no end moment is injected. The
out-of-plane restraint is a plane-of-symmetry condition on a two-dimensional
idealisation, and it is what OpenSees builds natively with `-ndm 2 -ndf 3` — so
the P5 backend comparison is against the same structure, not a stiffer one.

`smax`'s own `examples/arch_2d.py` has the same singularity and reports its
error against a baseline without noticing. Worth reporting upstream.

### The gap is quadratic in the diameter, and that is the whole of it

Measured on a 10 m, ten-member arch at `q = -75 N/mm` under 20 kN per node,
sized near the 73–87 mm the code check asks for:

| `d` [mm] | worst axial gap | worst `M / (N L)` | gap / `(d/100)²` |
|---|---|---|---|
| 50 | 5.68e-5 | 1.89e-4 | 2.27e-4 |
| 100 | 2.27e-4 | 7.58e-4 | 2.27e-4 |
| 200 | 9.09e-4 | 3.03e-3 | 2.27e-4 |
| 400 | 3.63e-3 | 1.21e-2 | 2.27e-4 |

**Constant in the last column to three figures over an eightfold range.** The
mechanism is not elastic shortening, which was the first guess and is wrong:
form finding returns a polygon with a kink at every node, and a chain of beams
cannot turn a kink on axial force alone, because continuity of rotation demands
a moment. That moment scales as `(i / L)²`, and `i ∝ d`.

Two consequences, both asserted rather than argued: the gap is **free of the
modulus** to 1e-12, since `EI` and `EA` both carry `E` and it cancels; and free
of the **scale of the loading** to 1e-12, when loads and `q` scale together so
the shape is unchanged. A tolerance pinned on a single number would have hidden
both. **Tolerances recorded at `d = 100 mm`: 2.5e-4 axial, 1.0e-3 bending.**

### What else the gate found

- Form finding balances the loads at every free node to **1.5e-10 N** against
  20 kN, and `state.forces` equals `q · L` to 1e-14.
- The arch stays exactly planar: the FDM system decouples per coordinate, so
  with no load along `Y` the free `Y` coordinates are **identically zero**, and
  `m_z` comes back **exactly zero** from the analysis.
- Axial force does not vary along a member, so **one number per member is not an
  approximation** — loads are nodal and the analysis is linear.
- End moments of neighbouring members agree at the shared node, and both bases
  carry **zero moment**, which is the pinned support showing up in the output.
- `jax.grad` of a scalar of `smax`'s output with respect to `q` crosses both
  stages and matches central differences to **2.9e-10**.

### Two modules, and the unit adapter written first

`normax/units.py` carries the conversions and nothing else, and was tested on
its own before either stage was wired: `normax` is millimetres, newtons and
`N·mm⁻²` with masses in tonnes, `smax` is coherent SI. **Force is the newton in
both and needs no conversion** — the one quantity crossing untouched, and its
absence from the module is deliberate rather than an oversight. 40 cases,
including that a stress times an area is the same force in either system.

`normax/formfinding.py` and `normax/analysis.py` split along the line between
topology and number. Connectivity is built once on the host; only `q`, the
geometry and the diameters enter the traced call. `analysis.frame` assembles the
`smax` model **inside** the differentiated function, which is what lets the
gradient reach the geometry and, in step 2, the diameters.

`MemberForces` carries the axial force and **both end moments** per member, per
the frozen T2 to T3 contract: nodal loads make the moment diagram linear between
nodes, so row 1 of Table B.3 is exact and `sizing.end_moments` consumes the two
ends directly. **The T2 schema stub still advertises a single peak `m_ed` and is
stale** — it predates P1b. It needs updating in step 3.

## P3 step 2 — the three stages as one differentiable function

`normax/pipeline.py` composes form finding, analysis and the code check into
`mass(q, ...)`, a scalar with a gradient in the force densities that crosses all
three. `experiments/09_arch_pipeline_jax.py` runs it, `tests/test_pipeline.py`
gates it, and this is the oracle step 3's Tesseract composition is measured
against rather than scaffolding for it.

Measured on a 10 m arch rising 3 m, ten members, 180 kN spread over its free
nodes, S355:

| | Class 2 | Class 3 |
|---|---|---|
| `d/t` | 46.338 | 59.577 |
| diameters [mm] | 61.7 – 78.1 | 72.5 – 87.5 |
| mass [t] | 0.032106 | 0.031199 |
| worst `\|u − 1\|` | 1.3e-15 | 1.7e-15 |

**The Class 3 design is the lighter one**, by 2.8%, and that is the documented
rationale in `CLAUDE.md` §3 showing up as a number: every member is governed by
the 6.61 member check, where Classes 1 to 3 all use the gross area, so the
thinner wall at the Class 3 limit wins on material even though the tube is fatter.
`tests/test_pipeline.py` asserts the ordering rather than leaving it as a claim.

### The utilization invariant holds to machine precision

**1.7e-15 against the 1e-9 of invariant 6.5**, on both class branches. The
invariant only means something because no member is sitting on the 21.3 mm
catalogue floor, where the utilization is below one by design — so that is
asserted alongside it.

### The gradient, and a metric that had to be fixed first

`dmass/dq` agrees with central differences to **1.2e-8**. Two things were needed
to say that honestly.

**The step had to be swept, not guessed.** Textbook V: 3.8e-6 at a relative step
of 1e-3, down to 1.2e-8 at 1e-5, back up to 1.0e-6 at 1e-7. Truncation on one
side, cancellation on the other. The experiment prints the sweep.

**A per-component relative error was the wrong instrument.** Two of the ten
sensitivities are twenty times smaller than the rest, and a 1.6e-13 absolute
difference there reads as an error of 1.3e-7 while saying nothing about the
derivative. The measure is now the absolute difference over the largest
component of the gradient. Same fix in spirit as the fixed-decimal tolerance
lesson from P1b: the instrument has to match the quantity.

### Two of the ten sensitivities nearly vanish, and the reason is physical

The mass gradient is not small there by accident — **it changes sign**, and
members 1 and 8 are the ones nearest the crossing. Decomposing it at frozen
diameters against the full derivative:

| | springing (0, 9) | crown (4, 5) |
|---|---|---|
| length term, `Σ A ∂L/∂q` | +2.96e-5 | +4.4e-7 |
| sizing term, `Σ L A' ∂d/∂q` | −1.45e-5 | −2.41e-5 |
| total | **+1.52e-5** | **−2.37e-5** |

The sizing term is nearly uniform across members, because `∂Σ\|N\|/∂q` is −846 for
every edge to three figures — the total axial demand of a funicular arch is set
by its thrust, and any single force density moves that thrust almost identically.
The length term instead falls by a factor of 67 from springing to crown, and that
spread has a specific cause.

**A uniform `q` fixes the horizontal projection.** The FDM system decouples per
coordinate, so a constant force density leaves the `x` solution evenly spaced:
every member spans exactly `span / n`, measured to a spread of 3.1e-12 mm over
1000 mm. Length is then `dx / cos θ`, so the steep springing members are the long
ones — 1471.87 mm against 1007.17 mm at the crown, a factor of 1.46. And because
`N = q · L` with `q` uniform, **the axial force ratio is exactly the length
ratio**, 1.4614 both. The springing is doubly penalised, longer and more heavily
loaded, so it is also the fattest: area ratio 1.4575.

**That is what amplifies the spread, and it is not the obvious mechanism.**
`∂arc/∂q_k` alone spreads only 3.91x, and the diagonal `A_k ∂L_k/∂q_k` only 2.13x.
The `∂L_j/∂q_k` Jacobian is a large positive diagonal, +10.9 to +15.9, against
many small negatives — perturbing one edge lengthens it and shortens every other.
Weighting by area **aligns** that positive diagonal with the fattest member at the
springing and **misaligns** it at the crown, where the induced shortening lands on
those same fat springing members and nearly cancels it: 3773.9 against 56.3.

**P4 should expect this to move.** Optimizing `q` per edge breaks the
constant-projection property outright, so both the length spread and the location
of the sign change are properties of the uniform starting point rather than of the
arch.

**At the springing, length dominates; at the crown, section does.** They cancel
one member in from each support. The sum over all edges is −8.5e-5, so weakening
the force densities uniformly makes the arch lighter — it rises, the thrust
drops, and the section saved beats the length bought. **That is the tension
`CLAUDE.md` §2 says the optimizer exists to resolve, confirmed to have an
interior optimum above 3 m rise before P4 goes looking for it.**

### Mesh refinement: convergent, first order, and not what the gate assumed

The ROADMAP asked for the mass to be "stable under mesh refinement". It is not
stable, it is **convergent, first order in the member count**, and reporting it
as stable would have been wrong.

Getting to that statement needed the refinement protocol fixed first. **Holding
the nodal load and the force density fixed changes the arch as the mesh changes**,
so the mass moves for a reason that has nothing to do with discretization: the
crown rise came out as exactly `3000 · n/(n−1)` mm, drifting 20% between the
coarsest and finest meshes. The total load is now fixed and the force densities
rescaled so the rise is exactly the target — the FDM `z` solution scales as
`1/|q|` at fixed loads, so one trial solve fixes the scale with no formula.

With the shape pinned, over meshes of 5 to 160 members:

- the arc length converges **second order** — 12128.99, 12028.09, 12039.63,
  12042.51, 12043.23, 12043.41 mm
- the largest axial force converges **first order**, because the end member's
  chord angle approaches the parabola's steeper tangent linearly in the segment
  length — 127.26 down to 117.44 kN
- so the mass converges first order. Successive relative changes fall by
  **1.856, 2.067, 2.030, 2.015** — halving, as claimed. Richardson gives
  **0.027351 t**.

**`L_cr = L` superposes a second effect that is physics, not discretization.**
Refining shortens the members, `L/i` at the crown falls from 73.5 to 5.5, `χ`
approaches one and buckling stops governing, so the sequence converges to the
squash limit instead. Both curves are reported. This is exactly the `L_cr = L`
caveat P7 has to state, and it is the reason the buckling length is an argument
to `design` rather than a convention inside it.

### The staggered coupling is weak, and now quantified

A frame cannot be analysed without sections and the sections are what the check
returns, so `design` takes the analysed diameters as an input and returns the
required ones. Repeating the pass contracts by a **constant factor of about
1/39**: relative moves of 3.80e-1, 1.17e-2, 3.05e-4, 7.91e-6, 2.05e-7, 5.31e-9.

**One pass costs 1.22% of the mass** against the fixed point. That is the price
of the one-way gradient the T2 schema documents, and it is now a number rather
than an acknowledged unknown. The contraction is fast because the analysis barely
depends on the section — the same `(i/L)²` smallness that step 1 measured.

### Figures

`normax/visualization.py`, matplotlib only, no styles and no `show`. The
experiments compute and it draws, so a figure can be recomposed without touching
what produced it. Three, in `figures/`:

- **`09_sections.png`** — the arch before and after the check has spoken, members
  drawn at a width proportional to diameter and coloured on one shared scale, over
  a per-member bar chart. 36.3% lighter than the uniform 100 mm assumption.
- **`09_convergence.png`** — the mass against mesh density for both buckling-length
  conventions, the order of convergence against a first-order reference line, and
  the staggered coupling contracting.
- **`08_handoff.png`** — step 1's outputs, for assessing them by eye rather than
  by tolerance: `q · L` against `smax` per member, the gap and the bending share,
  the quadratic law against a `d²` reference, and autodiff against central
  differences.

Widths are drawn to a stated exaggeration rather than to scale — a 100 mm tube on
a 10 m arch is 1% of the span — and the factor is written into the figure instead
of left for the reader to infer. `matplotlib` joins the `pipeline` group;
`tests/test_pipeline.py` smoke-tests both figure builders under `Agg`.

**Test counts: 1441 in CI, 1491 locally.** The 50-case difference is
`test_equilibrium_consistency.py` and `test_pipeline.py`, both dropped at
collection when the pipeline packages are absent.

### The buckling length is an input, and the member-length choice is a strong assumption

**Decided 2026-08-09.** `L_cr` is an argument to `normax.pipeline.design`, never
derived from the mesh. The default is the member's own length, and that is
**presumed, not conservative**: it asserts that every node is held in position by
structure outside the model, so a member can only buckle between its ends. For a
gridshell, whose hoop members brace its radial ones, that is the right reading.
For the arch it is an idealisation of a rib in a braced system, and the writeup
has to say so rather than let it pass as caution.

**How far it is from true, measured rather than argued.** `normax.analysis.buckling`
wraps `smax.solve_buckling` and returns the critical load factors of the whole
frame. On the fully-stressed ten-member arch, all four lowest modes are below the
design load:

| mode | `α_cr` | shape |
|---|---|---|
| 0 | **0.1291** | antisymmetric sway — one half rises as the other falls |
| 1 | 0.3119 | symmetric — crown lifts, quarter points drop |
| 2 | 0.5642 | antisymmetric, two waves |
| 3 | 0.8885 | symmetric, three waves |

**Mode 0 is the classic two-pinned arch mode**, with `u_z` antisymmetric about
midspan and zero at the crown while `u_x` is symmetric and non-zero there — the
whole arch sways sideways. The buckle wavelength is the arch, not the segment,
which is exactly what a member-length buckling length cannot see. Backing out
`L_cr = π√(EI/(α_cr N_Ed))` gives **0.576 of the developed length, steady to three
figures across a 32-fold range of mesh density** — mesh-independent, as a global
mode must be.

**The out-of-plane restraint is already counted, and it is already exhausted.**
`u_y` is identically zero in every mode, and adding the out-of-plane rotations to
the restraint set leaves `α_cr` at 0.1291 to four decimals with the same mode
shapes. So 0.129 is the in-plane, already-stiffened figure rather than the
pessimistic one; a bare three-dimensional arch would be weaker still. It also
re-confirms P3 step 1's finding that restraining the one translation is enough.

**What sizing against the real mode would cost:** `L_cr = 0.576 × arc` puts the
diameters at 137–151 mm instead of 72–87 mm and the mass at **0.101669 t against
0.031199 t, a factor of 3.26** — and even then `α_cr` reaches only 1.41, far short
of the order of ten at which second-order effects are normally set aside. **So the
bare arch is not rescuable by sizing at this span, rise and load**, which is why
the braced-rib idealisation is the honest reading rather than a convenience. Both
masses are printed by `experiments/09_arch_pipeline_jax.py` and the ratio is
asserted in `tests/test_pipeline.py`, so the sensitivity is a result and not a
remark.

**`α_cr` is a diagnostic and is never differentiated.** The eigenproblem is pure
JAX and would trace, but an eigenvalue derivative is undefined where two modes
cross and an optimizer moves modes around. It is read beside a design, like
`governing`.

**Not implemented, and deliberately:** EN 1993-1-1's own use of `α_cr` — the
threshold above which second-order effects may be neglected, and the sway
amplifier — is **not in `docs/clauses.md`** and so is not verified. The factor is
reported as a number with no clause verdict attached. Verifying §5.2.1 against the
standard is a prerequisite for turning it into a check.

`figures/09_modes.png` draws the four modes over the undeformed arch.

### `α_cr` became a check, and the standard's two doors to slenderness

**Changed 2026-08-09 on instruction**, from reporting the critical load factor to
checking against it. `normax/ec3/stability.py` and `normax.pipeline.stability`.

⚠️ **This is a deliberate exception to the rule that nothing marked UNVERIFIED
gets implemented.** §5.2 and §6.3.4 were not in `docs/clauses.md`; they are now,
transcribed from memory and flagged, with **open item 0f** recording that every
threshold and equation number in them is unchecked — `α_cr ≥ 10` and `≥ 15`, the
`1/(1 − 1/α_cr)` amplifier and its `α_cr ≥ 3` floor, and the numbers 5.1, 5.2 and
6.64. The threshold is a parameter with a flagged default, so correcting it is one
line. **None of it may reach the writeup unverified.**

**A check, not a diagnostic, and the arch fails it.** `α_cr = 0.1291` against a
threshold of 10 gives a utilization of **77.4**, and `is_adequate` returns False.
That failure is now a pinned test rather than a caveat, which is the point: it is
the evidence that the braced-node reading of `L_cr = L` is load-bearing rather
than cautious.

**The check cannot enter the sizing map, and that is structural.** The bisection
roots a *member* check, which is local and monotone in one diameter. Global
stability is a property of the whole frame: a design that fails it is not made to
pass by growing one member, and the remedy is bracing or a different buckling
length. So `stability` reads a finished `Design` and never appears inside it.

#### Two doors into the same quantity

The more interesting half. EN 1993-1-1 reaches the slenderness driving `χ` twice
over — §6.3.1.3 Eq. 6.50 from a **member** buckling length, §6.3.4 Eq. 6.64 from a
**system** critical load factor — and for pure compression they are the same
equation. With `α_ult,k = A f_y / N_Ed` and `α_cr = N_cr / N_Ed`:

```
α_ult,k / α_cr = (A f_y / N_Ed) · (N_Ed / N_cr) = A f_y / N_cr = λ̄²
```

**That identity needs no source and is asserted as algebra**, over four diameters,
three axial forces and three buckling lengths, agreeing to 1e-14. It is also what
makes `L_cr = π√(EI/(α_cr N_Ed))` a legitimate inversion rather than a fudge, and
that round trip is tested too.

**Fed the arch, the two doors disagree by the assumption and not by an error:**

| member | Eq. 6.50 from `L_cr` | Eq. 6.64 from `α_cr` | ratio | `L_cr` implied [mm] |
|---|---|---|---|---|
| 0 | 0.6334 | 2.9821 | 4.71 | 6930.1 |
| 4 | 0.5232 | 2.9863 | 5.71 | 5748.5 |

**The global route is almost uniform across the arch** — 2.978 to 2.986, a spread
of 0.3% — because one mode governs the whole structure, while the member route
varies by 21% with member length. That contrast is the physical content: a
buckling length is a statement about a member, a critical load factor is a
statement about a structure, and the standard will accept either. A pipeline that
differentiates the code can walk through either door and report both, which is
something no scalar branchy implementation of the same clauses offers.

**`normax/ec3/stability.py` needs no pipeline guard** — it is pure EC3 with no
solver dependency, so its **61 tests run in CI** alongside the rest of the clause
layer. Only the composed `pipeline.stability`, which calls the eigensolver, sits
behind the guard.

### Stability is soft validation and stays out of the chain

**Decided 2026-08-09.** `normax/ec3/stability.py` and `normax.pipeline.stability`
size nothing, enter no gradient, and cross no Tesseract boundary. **Global
stability is therefore not covered by what this package designs**, and the writeup
says so as a limitation rather than implying otherwise.

The boundary is a scope decision, and a cheap one. Putting a critical load factor
into the T2 schema would change a schema the roadmap freezes on day one, add a
non-differentiable output that must be popped before every gradient, and oblige
**every** analysis backend to supply it — which the OpenSees backend cannot
without real work, since linear buckling there needs the geometric stiffness
assembled by hand. That is a direct cost against the swappability thesis, paid for
a second structural feature. The thesis is that backpropagation through a design
code works; stability is out of scope, stated rather than hidden.

**Both structures fail the check, and both numbers go in the writeup.** The arch
at `α_cr = 0.129`, the gridshell at **0.372** — measured on a 48-member cap, 25
nodes, rise over span 0.135. Shallow caps are snap-through-prone, so the rise is
worth sweeping before P4 fixes the geometry.

**Two corrections to what was recorded earlier.**

- **The self-consistent fixed point does not oscillate.** The two-cycling reported
  before came from an ad hoc update rule, not the physics. With the correct
  per-member inversion `L_cr = π√(EI/(α_cr N_Ed))` it converges in **one pass**:
  `α_cr` 0.129 → 1.1613 → 1.1611 → 1.1610, mass 0.0312 → 0.0935 → 0.0932,
  monotone. Under-relaxation only slows it. Not adopted, but no longer mis-recorded.
- **Repeated eigenvalues are real here, not a hypothetical.** The eight-fold
  symmetric gridshell returns `[0.3722 0.3722 0.6246 0.7193]` — a degenerate pair
  at the critical mode. That is exactly where an eigenvalue derivative is
  undefined, so "never differentiate `α_cr`" now has a counterexample behind it
  rather than a caution.

**One defect fixed.** A gridshell's boundary hoops span support to support and
carry exactly zero axial force, and `resistance_factor` and `buckling_length`
divided by it, reporting `inf`. They now return **nan**: infinity would read as a
statement about the member, while nan says the question does not apply, and a
reduction over the members says so too instead of quietly absorbing it. The
two-doors bridge is only meaningful for a member carrying the load the global mode
is scaled against, which is now documented and tested.

### Classification is now exact by construction in floating point too

`CLASS_LIMIT_TOLERANCE = 1e-12`, a relative widening of the inclusive bound of
Table 5.2. CLAUDE.md §3 argues that pinning `d/t` makes classification exact by
construction, and in exact arithmetic it is — but a member is *built* by pinning
the ratio and taking its wall as the diameter over it, and recovering `d/t` from
the diameter and the wall returns it only to within rounding. Measured spread
**7.1e-15 absolute on 59.577**, and a strict comparison then hands back Class 4:

```
before:  arch class 3 -> classes {3, 4}   class 2 -> classes {2, 3}
after:   arch class 3 -> classes {3}      class 2 -> classes {2}
```

Nothing in the design path was wrong — the class is a static Python integer and
the ratio a constant, so `classify` is never called on a recovered ratio there.
But **anyone auditing a design by recomputing `d/t` would have concluded we had
violated our own scope boundary**, which is exactly the check a reader of this
repo should be able to make.

**Which side of the limit the rounding lands on is grade-dependent**, and the
first attempt at a test asserted it always straddles, which is false for S235,
S275 and S460. It straddles at **S355**, the grade every design here uses, and
that is what the test now pins — alongside the invariant that matters, that a
family built to one ratio classifies to one class over 64 diameters and all five
grades. The tolerance is ten thousand times the observed rounding and ten orders
below any difference in `d/t` between real sections, so it separates rounding
from geometry and nothing else: CHS 508×8 at `d/t` 63.5 is still Class 4.

### What is actually a variable, and what follows from it

Worth stating plainly, since the section has two dimensions and only one degree
of freedom.

| quantity | role | how it moves |
|---|---|---|
| `q`, per edge | **the design variable** | what P4's optimizer will drive |
| `d`, per member | **solved, not chosen** | the root of utilization = 1, with an exact tangent by IFT |
| `t`, per member | **not a variable at all** | `t = d / r`, exactly |
| `r = d/t` | config, per design | fixed by `Tube.at_class_limit`; differentiable but pinned |

So the diameter and the thickness are never optimized separately, and the
thickness is never optimized at all: **one section variable per member, and it is
solved for rather than searched over.** That is what makes the sizing map a map
and not an inner optimization, and it is why the utilization comes back at
1.0 ± 1.7e-15 instead of at whatever an optimizer left it at.

`r` is a genuine leaf and carries a gradient — `∂mass/∂r = -6.279e-05`, matching
central differences to six figures — so freeing it is a one-line change.

**Correction.** An earlier draft of this entry said the class "cannot be a traced
value". That is wrong: `lax.switch` branches on a traced value perfectly well, and
CLAUDE.md invariant 4 names it as the tool for exactly this. The claim also
contradicted an invariant this repo already follows.

**The real reason `r` is pinned is that the class boundary is a discontinuity in
the standard, and tracing it does not remove one.** `M_Rk` steps from `W_pl f_y`
to `W_el f_y` across it, a drop of 24.6% for a CHS. `lax.switch` would make that
step *expressible*, not smooth. With `r` fixed the question never arises, because
`t = d/r` leaves `d/t` invariant in `d`: no diameter the optimizer chooses can
change the class. Freeing `r` within one class is available today; freeing it
across a boundary is a piecewise problem best answered by optimizing inside each
class and comparing, which is what `experiments/05_class_ratio_sweep.py` is for.

For the record, the other leaves that carry gradients and could become variables:
`f_y` at -6.814e-05, `e_mod` at -1.580e-08, the buckling curve's imperfection
factor `alpha` at +1.266e-02, and `gamma_m1` at +2.751e-02. **The partial factor
having a derivative is the thesis in one number** — a quantity that exists only
because a committee wrote it down, carrying a sensitivity through a form-finder.

---

## P3 step 3 — the three stages as three Tesseracts

The pipeline now exists twice. `normax/pipeline.py` runs it in one process and
one JAX trace; `normax/composition.py` runs it as three Tesseracts with schemas
and serialized arrays between them. Both expose `q` and a mass, both are
differentiable in `q`, and `tests/test_tesseract_parity.py` asserts they agree.
`experiments/10_arch_pipeline_tesseract.py` prints the comparison.

### The boundary is free, and that is a measurement rather than a hope

On the same 10 m arch, ten members, both class branches:

| | worst scaled difference | target |
|---|---|---|
| every field of the design | **6.7e-16** | 1e-14 |
| `dmass/dq` | **3.6e-14** | 1e-12 |
| forward mode against reverse | **2.7e-14** | 1e-12 |
| two stages in containers, over HTTP | **1.5e-13** | 1e-11 |

The roadmap asked for 1e-10 on the mass and the gradient. Values cross exactly:
the geometry, the member forces, the diameters and the mass are bit-identical on
the Class 3 branch and differ by one unit in the last place on Class 2.

**Derivatives are looser than values, and not because of the boundary.** Each
stage linearizes on its own here and all three linearize together in process, so
the same sum accumulates in a different order and the implicit tangent divides by
a slope differing in its last bits. Forward mode against reverse mode, both
entirely inside the composition, disagree by 2.7e-14 — the same size. The
boundary is not what costs the digits; splitting the linearization is.

### What each schema carries, and why

| | differentiable in | differentiable of | static |
|---|---|---|---|
| T1 formfinding | `q` | `xyz`, `lengths`, `forces` | topology, loads |
| T2 analysis | `xyz`, `diameter` | `n_ed`, `m_y_ed`, `m_z_ed` | material, `normal` |
| T3 ec3_check | every action and material property | every output but `governing` | `plastic`, `resultant`, `diameter_min` |

**T2's differentiable inputs are exactly the two that direct differentiation can
supply**, and that is the constraint the frozen schema is built around rather
than an accident. A nodal coordinate and a section property are what the OpenSees
spike established are reachable; adding a third would be a promise the second
backend cannot keep. A test pins the set at `{xyz, diameter}` so it cannot drift.

**T3 differentiates in everything it is given except the catalogue floor.** It is
pure JAX with no second implementation to satisfy, so `f_y`, `e_mod`, both
partial factors, the `d/t` ratio and the imperfection factor all carry gradients
across the boundary. The two stages disagreeing about what is differentiable is
the honest state of affairs, and it is visible in the schema rather than buried.

**The critical load factor is not in T2's output schema, and a test says so.**
Putting it there would oblige every backend to produce one.

### The corrections the stale stubs needed

Three, all recorded in the roadmap before the work started and all real:
`m_ed` was a single peak moment and is now both end moments at `(members, 2)`,
which is what makes the first row of Table B.3 exact; the backend was named
`sax` and is `smax`; and `_solve_fdm` raised `NotImplementedError` where
`normax.formfinding` already existed. A fourth was found in the writing:
`chi_buckling` and the section formulas were reimplemented inside the T3 stub,
which would have been a second copy of the clauses to keep in step with
`docs/clauses.md`. Every stage now imports `normax` and transcribes nothing.

### Table B.3 lives in the check, not in the analysis

`end_moments` reduces two end moments to a design moment and an equivalent
uniform moment factor, and that reduction moved from the composition layer into
T3. It is a clause of EN 1993-1-1 and not a product of an analysis, so a solver
should have no opinion on it. The consequence is that T2's schema stays free of
anything a C++ frame solver would have to be taught, and T3 reports `m_ed` and
`c_m` as outputs — which is also what lets the parity test compare every field of
the design rather than only the mass.

### The §5 failure modes, exercised rather than cited

A concrete cotangent on `governing` raises `ValueError` naming the field, and a
test asserts it. The composition drops the diagnostic for that reason, and
`normax.pipeline.governing` reads it beside a finished design instead; the
Tesseract's own `governing` output is checked against that reader and matches
exactly. A Python list is refused at the boundary with a `TypeError`, also
pinned. Every schema declares `Float64` and a test walks every output and the
gradient checking the dtype, because the upstream examples are float32 and a
single float32 stage would downcast silently and cost eight digits.

### Two of the three stages containerize today

`tesseract build` produces `normax-formfinding:0.1.0` and
`normax-ec3-check:0.1.0`, and swapping either into the chain in place of the
imported module changes the mass by 2.6e-15 and the gradient by 1.5e-13. Warm,
a mass costs 0.21 s served against 0.18 s imported and a gradient 0.67 s against
0.38 s, so the round trips are visible but not dominant at this scale.

**T2's image does not build, and the reason is `smax` rather than the schema.**
It is not on PyPI, so `tesseract_requirements.txt` names it and the build fails
at the dependency install; the day it is published the build works with no edit.
Nothing else waits on that — the composition imports the module in process.

Three things about the build that cost time:

- **A Tesseract version must be three dot-separated numbers.** `0.1`, which is
  what `pyproject.toml` carries, is rejected by the config validator.
- **The base image's Python is too old**, so `python_version: "3.12"` is required
  in every `build_config`.
- **Only `tesseract_api.py` is copied into the image.** The analysis backend is a
  separate module so a second one can arrive without touching the schema, which
  means it needs a `package_data` entry.

`tesseract_requirements.txt` installs `normax` from `../../dist/*.whl`, so
`uv build` has to run first. The alternative — naming the repo root as a local
dependency — copies the whole tree into the build context, and this one carries
a 780 MB virtual environment.

### Serving a container on macOS needs an explicit output path

`Tesseract.from_image` defaults its `output_path` to a `tempfile.mkdtemp` under
`/var/folders`, which the container runtime's file sharing does not reach. The
mount then silently becomes an empty root-owned directory inside the container
and every endpoint that opens a run directory fails with `PermissionError` on
`/tesseract/output_data`. Passing a path under `$HOME` fixes it. The failure
names a path inside the container, so it reads as an image permissions bug and
is not one.

### Tests

**27 cases in `tests/test_tesseract_parity.py`**, added to `PIPELINE_TESTS` in
`tests/conftest.py` because the analysis stage still needs `smax`. They import
the API modules through `Tesseract.from_tesseract_api`, so **the suite needs no
Docker**, per invariant 6. The suite is **1630 locally and still 1538 in CI**,
every new case sitting in an excluded file; the gap is now 92, measured by
collecting the three excluded files rather than inferred. The served-container comparison is deliberately not a
test for the same reason; it runs from experiment 10 when
`NORMAX_SERVED_OUTPUT` names a bindable directory.

---

## P4 — the 2D arch, optimized

`experiments/03_optimize_arch.py` searches twenty force densities for the
lightest arch EN 1993-1-1 will accept under three load cases. The sizes are never
searched over: they are solved for inside the objective at every iterate, so the
optimizer sees an unconstrained scalar and gets back a gradient that has crossed
form finding, a frame analysis and the standard.

**Four prerequisites were folded in**, none of which existed before this phase:
load case generators in `normax/structures.py`; a `loads` argument on the
analysis and on both pipelines; `pipeline.envelope` with `unsmoothed` beside it;
and `normax/optimization.py`. `scipy` became a project dependency.

**The load case is free at the boundary.** Checking a structure against several
cases changes the Python signature of `design` and no part of the frozen T2
contract, because the analysis schema already carried the nodal loads. That is
the schema freeze of P3 step 3 paying for itself one phase later.

### The gate

Interior minimum in the uniform family, at 1.50 times the funicular force
density, and the composed gradient agrees with the sweep to **4.4e-8**.

| | mass [t] | against funicular |
|---|---|---|
| funicular, uniform `q` | 0.134063 | — |
| best uniform `q`, scale 1.50 | 0.129030 | 3.8% lighter |
| twenty variables, 300 mm floor | **0.091552** | **31.7% lighter** |
| twenty variables, unconstrained | 0.047151 | 64.8% lighter, and degenerate |

The one-variable family is what a form-finder can reach by scaling; twenty
variables under a length floor end **29.0% below its best**. That is the whole
argument for a gradient in one number. The unconstrained row is the same search
with nothing stopping it from collapsing members, and it is reported as evidence
rather than as a design — see below.

### The finite-difference step moved, and the reason is not the smoothing

**1e-4 here, not P3's 1e-5.** Swept rather than guessed:

| relative step | 1e-3 | 1e-4 | 1e-5 | 1e-6 | 1e-7 |
|---|---|---|---|---|---|
| worst scaled error | 1.2e-6 | **4.4e-8** | 2.7e-7 | 2.1e-7 | 5.5e-5 |

Three load cases make the mass four times larger and the arithmetic behind it
three times longer, so cancellation dominates a decade sooner and the trough
moves. **The obvious explanation was tested and rejected**: a sharp envelope is
nearly a maximum, so it should be nearly a kink, and a kink would ruin a central
difference. It does not. The error is 4.5e-7 at a step of 1e-5 for every
sharpness from β = 10 to β = 500, identical to three figures, and the two largest
per-member demands differ by 7–27% — nowhere near a tie. The step is the whole
story.

The tolerance is pinned at 2e-7 rather than on the measured floor. 4.4e-8 against
a 5e-8 gate is a 12% margin and would fail on another machine; a gradient that
was actually wrong misses by a thousand times more than the gap between those.

### What the envelope costs, and that the invariant survives

| β | 10 | 25 | 50 | 100 | 250 | 500 |
|---|---|---|---|---|---|---|
| excess mass | 4.35% | 0.449% | 0.037% | 0.0008% | 0.0000% | 0.0000% |
| bound | 24.6% | 9.19% | 4.49% | 2.22% | 0.883% | 0.440% |

The bound is the case count raised to the reciprocal of the sharpness, squared
for a mass rather than a diameter. It is honest but loose — the real excess runs
five to twenty times under it, because it assumes every case ties everywhere.

**Utilization of the unsmoothed design is exactly 1.0 at every sharpness.**
Invariant 6.5 survives the aggregation in the only form it can take with more
than one case: some case works every member to one, though no single case works
all of them. The envelope never understates, so the design is adequate at every
sharpness and annealing approaches the answer from the safe side.

### The funicular case never governs a single member

Not one, before or after. LC2 decides 16 members and LC3 decides 4 at the
starting shape; after the descent that reverses to 4 and 16. **The case the shape
was found under is the benign one by construction**, and everything the design is
actually sized by is invisible to a form-finder. That is the project's premise
arriving as a count rather than an argument, and the reversal is the figure only
a differentiable code check can produce: no member was reassigned, the form
moved and the pattern followed.

### The search collapses members, and a length floor is what stops it

Measured on the unconstrained design: member lengths run **26.7 to 2335 mm, a
ratio of 87**, with **fifteen of twenty members under 100 mm** and one of them
0.20 diameters long. Five members carry 46 kg of the 51 kg total. The descent
turned a twenty-member arch into a five-member arch with fifteen vestigial stubs.

**Two things reward that and nothing objects to it.** A member's mass is an area
times a length, so a vanishing member is free; and its buckling length is that
same length, so `λ̄ → 0` and `χ → 1` and it is also unbucklable — measured,
`λ̄ ≈ 0.007` on the shortest member against 0.94 on the longest. Collapsing an
edge makes it both weightless and strong.

`normax.optimization.penalized` adds a floor: a smooth minimum of the lengths in
log space, the envelope operator with its sign reversed, and a multiplicative
penalty on the fractional violation. Multiplicative because it then needs no mass
scale, squared because the objective stays flat at the floor rather than kinked.

Both descents are run to convergence, 32 and 110 iterations against a budget of
300, so neither stops on its limit and both numbers are optima rather than
bounds.

| | unconstrained | 300 mm floor |
|---|---|---|
| mass [t] | 0.0472 | 0.0916 |
| against the funicular arch | 64.8% lighter | **31.7% lighter** |
| against the best single `q` | 63.5% lighter | **29.0% lighter** |
| shortest member [mm] | 27.3 | 313.4 |
| length ratio | 85.3 | 11.9 |
| members under the floor | 15 of 20 | **0 of 20** |
| force densities on a bound | 14 of 20 | **1 of 20** |
| diameters [mm] | 60 – 136 | 104 – 170 |
| `α_cr`, weakest case | 0.713 | **1.734** |

**Half the headline reduction was collapse rather than design.** 31.7% is the
number to quote.

**More budget buys the unconstrained run a deeper collapse, not a better arch.**
Raising the iteration cap moved it from 0.0510 to 0.0472 t and `α_cr` from 0.812
down to 0.713, with fifteen members still under the floor and fourteen force
densities still on a bound. That is the diagnosis confirming itself: the
unconstrained problem has no interior optimum to find, so it goes on trading
members for stubs until something stops it. The floored problem does have one,
and finds it.

**The floor also makes the problem well posed.** Unconstrained, fourteen of
twenty force densities end on a bound and the answer is set by a box chosen to
keep the model meaningful. With the floor, one does — the design sits in the
interior and the physics decides it.

**The figures compare the constrained design against the single force
density**, which is the comparison that means something: the best a form-finder
can reach by scaling against the best twenty variables can reach without
collapsing anything. The unconstrained run is drawn beside them rather than
instead of them, since it is the evidence for why the floor is there.
`figure_optimization` takes any number of descents and `figure_load_cases` any
number of forms.

**And it recovers the stability margin.** `α_cr` rises from 0.713 to 1.734, so
the floored frame no longer buckles below its design load — measured under every
case, 1.913 for LC1, 1.734 for LC2 and 1.764 for LC3. It is still far from
§5.2.1's threshold, and first-order analysis is still not adequate for it, but
the design is no longer self-evidently unbuildable. The governing pattern also
evens out, LC2 taking 8 members against 4.

### Holding the plan instead is degenerate, and the reason is algebraic

`normax.formfinding.positions_vertical` solves for heights with the plan held, so
no member can shorten past its own projection — a hard bound rather than a
penalty. It does not work, and the arithmetic says so before any experiment does.

The force density system decouples per coordinate, so **holding the plan does not
change the heights at all**: the full solve and the held solve agree to 1e-9 in
`z` for any `q`. It moves the plan and nothing else. What it drops is horizontal
equilibrium, which at a node reads
`q_before (x_before − x) + q_after (x_after − x) = 0` and on an evenly spaced plan
collapses to `q_after = q_before`. **Only a uniform force density leaves a held
plan funicular.**

Measured with non-uniform `q`: the held plan carries **93.4 kN of unbalanced
horizontal force** against 5.5e-10 N for the full equilibrium, and LC1 bending
rises from `|M|/(N·L)` of 2.0e-4 to **0.72**, a factor of 3660. The case the shape
was found under stops being the benign one, which is the premise P4 rests on.

So the funicular subspace of a held plan on this arch is one parameter wide, and
that parameter is the uniform sweep. Four tests pin it.

### Thrust network analysis is the general form, and the count has a trap in it

Fix the plan, write horizontal equilibrium as linear in the force densities —
`B_c = C_free^T diag(C x_c)`, stacked over the horizontal coordinates — and its
nullspace is the funicular design space with the plan fixed. A basis for it is
what TNA calls the independent edges; optimize those, propagate the rest, and the
plan is held **with equilibrium intact**. Noted in `ROADMAP.md` for after the
deadline as a `jax_tna` prototype, with `compas_tno` as the reference.

| | edges | rows | rank | independent edges |
|---|---|---|---|---|
| arch, 20 members | 20 | 19 (x alone) | 19 | **1** |
| gridshell, 4 × 12 | 96 | 74 | 71 | **25** |
| the same, plan jittered | 96 | 74 | 74 | 22 |

A chain has one, which is why holding its plan gives nothing. The shell has
twenty-five.

**The naive count is a lower bound and it is wrong here.** `edges − 2 × free
nodes` gives 22; the symmetric cap is rank-deficient by three and the true
nullity is 25. Breaking the rotational symmetry by jittering the plan restores
full rank and drops it to 22 exactly. The deficiency is the number of free rings
— two at three rings, three at four, four at five — and is independent of the
spoke count. **The geometries here are the symmetric ones**, so a TNA prototype
that trusts the formula under-counts the design space on all of them; the
classification has to come from a rank-revealing factorization. The mechanism
behind the deficiency is not verified, the measurement is.

### Three things the answer is not

**The per-edge optimum is set by its box.** Fourteen of twenty force densities
end on a bound, nine low and five high. Nothing in a member check penalises a
shape for being a bad arch — every member is fully stressed and adequate whatever
the form does between them — so the search keeps going until the box stops it.
Tightening the box to make the answer look converged would be dishonest; the
bound activity is reported instead.

**The form is degenerate, and `figures/03_load_cases.png` shows it.** The right
leg collapses into a near-vertical cluster of nodes and the rise falls from 3000
to 2299 mm. Both drawings share axis limits precisely so that this is visible
rather than framed away.

**The unconstrained descent spends the stability margin.** Like for like under
LC1, `α_cr` falls from **2.72** at the starting arch to **0.873** at the
unconstrained optimum — below one, so the frame buckles before reaching its
design load. The floored design does not: it recovers to 1.913 under the same
case. Member checks were
never going to catch that, and global stability is outside the pipeline by
design. The optimized arch is not buildable and the number is the evidence.

**And quoting one case flatters it.** `analysis.buckling` and
`pipeline.stability` now take a load case, because a frame sized for its worst
case is not necessarily least stable under that case. On the optimized design:

| case | LC1 uniform | LC2 half span | LC3 crown point |
|---|---|---|---|
| unconstrained | 0.873 | **0.713** | 0.988 |
| 300 mm floor | 1.913 | **1.734** | 1.764 |

The weakest is LC2 in both, 18% below the LC1 figure a case-blind check would
have reported, and it is not the case that sized the most members. **The numbers
to carry into P7 are 0.713 and 1.734.**

### Which clause decides, and one deferred question closed

**The cross-section check governs 19 of 20 members**, and Eq. 6.61 just one.
P3 step 2's single-case design was governed by 6.61 everywhere, so admitting load
cases that raise real bending moves the decision from the member check to the
cross-section check. Worth knowing before reading either clause's sensitivity as
representative.

That also settles the Eq. 6.42 question deferred in P1b — how the two moments
combine in the cross-section check — but not in the way the deferral anticipated.
The test was "is the cross-section population large, and does it carry biaxial
moment". The population is large. **The biaxial moment is identically zero**: on
a planar arch under in-plane load `m_z` is 0.0 exactly, so there is nothing to
combine and the resultant and the linear sum are the same number. **The choice
cannot bite until the 3D gridshell**, and it should be decided there.

### The staggered coupling costs 8.7% here, not P3's 1.22%

Measured at the unconstrained optimum, one pass costs **8.7%** of the mass, and
the relaxed sequence settles by the fifth pass. The gap grows as the design
leaves the seed diameter, and the optimizer walks a long way from it: the seed is
100 mm and the answer runs 60 to 136 mm. **The
reported optimum is a one-pass optimum**, and relaxing to the fixed point makes it
lighter still. This is the strongest argument yet for formulation B, which
dissolves the stagger at every iterate rather than at the end.

### The multi-case objective crosses the boundary too

`composition.envelope` mirrors `pipeline.envelope`, and the two agree **exactly**
— 0.0 on every field of the design and on the gradient, with one array differing
by a single unit in the last place. So the objective the optimizer actually
minimizes, three analyses and three checks per call, is as transparent to the
boundary as the single-case design was.

**The cost gap is in the reverse pass, not the forward one.** At ten members and
three cases: 0.505 s composed against 0.493 s in process for a value, 2.4% apart;
1.79 s against 1.10 s for a value and gradient, 63% apart. Each stage linearizes
on its own and there are three times as many round trips, which is the number
P5's scaling plot needs.

Two things had to change to make it fit. **The check now reports both moment
axes** — `m_y_ed`, `m_z_ed`, `c_my`, `c_mz` where it previously reported only the
major ones — so a finished design can be re-read at a size the standard did not
choose without analysing anything again. And **form finding runs once for all the
cases**, the shape answering to one load case by construction, so only the
analysis and the check are walked per case.

**One asymmetry, named rather than hidden.** The envelope over cases is the
optimizer's smoothing and the mass is `ρ Σ A L`, so neither is a clause and both
sit above the chain. But the utilization at the enveloped size *is* a clause, and
it is computed above the chain as well, because the check answers "what size do
these actions need" rather than "how hard would this size work". The sizes cross
the boundary; the re-check does not.

### Cost

A value is 0.5 s and a value-plus-gradient 2 s at twenty members and three cases,
the analysis stage dominating both. The descent is five annealing rounds of
twenty-five iterations, warm-started, and the whole experiment runs in about nine
minutes.

### Tests

**1707 locally and 1582 in CI**, up from 1630 and 1538. 44 in
`tests/test_structures.py` including the load cases, 19 in the new
`tests/test_optimization.py`, the envelope, the aggregation invariants, the
per-case stability check and both new figures in `tests/test_pipeline.py`, and
five more in `tests/test_tesseract_parity.py` for the enveloped design, and
the length floor, the held plan and the figures on top of those. The CI gap is
125, being the three files that need `smax`. The optimizer
tests drive analytic bowls rather than the pipeline, so they run in CI: the
driver is tested against arithmetic, not against the thing it usually drives.

Two behaviours of the driver are pinned because they surprised: **L-BFGS-B steps
once before honouring a limit of zero**, returning a clipped trial point, so
`descend` refuses to report it; and the point it reports last is not always the
best it found, so the trajectory's last row is scipy's answer rather than the
last callback.

---

## §5.2 and §6.3.4 verified — open item 0f closed

**Verified 2026-08-09** against both textbooks, closing the one deliberate
exception to the rule that nothing marked ⚠️ gets implemented. It was taken
because the arch's global stability had to be checked rather than reported, and
P4's headline limitation now quotes `α_cr = 0.713` and `1.734` against these
thresholds. `docs/clauses.md` §5.2/§6.3.4 carries ✅ and open item 0f is struck.

**Every threshold held, so nothing P4 reports moves.** `α_cr ≥ 10` for elastic
analysis and `≥ 15` for plastic are EN 1993-1-1 §5.2.1(3) — guide p. 18, ECCS
pp. 79–80 and again p. 369. The sway amplifier `1/(1 − 1/α_cr)` is §5.2.2(5) and
its `α_cr ≥ 3.0` floor is real — ECCS p. 276. Memory got the numbers right.

**Two equation numbers were wrong.** Eq. 5.1 is the threshold *pair*, not the
definition `α_cr = F_cr/F_Ed`, which EN never numbers — the guide gives it one of
its own making, `(D5.1)`, which is what exposed it. And **6.64 could not be
confirmed at all**: neither book prints EN's numbering for §6.3.4, so the general
method is cited as **§6.3.4(3)** under the same policy as open item 1, with its
check at §6.3.4(2). Eq. 5.2 survives and gained its clause, §5.2.1(4)B.

**The finding that matters is a scope error, not a number.** §6.3.4's `α_cr,op`
is the amplifier reaching instability *in a lateral or lateral-torsional mode*
and takes no account of in-plane flexural buckling (ECCS p. 300); UK NA NA.2.22
narrows it further and the guide advises caution with the whole clause. The mode
`normax.analysis.buckling` measures is in-plane by construction — it restrains
the one translation normal to the plane precisely to keep it there. So §6.3.4 is
where the standard writes this algebra, **not authority for the case we apply it
to**. The two-doors identity is unaffected: it is algebra, needs no source, and
is tested as one. The citation now says which of the two it is leaning on.

Also new from the check and not previously known: **UK NA clause NA.2.9** lowers
the *plastic* limit to `α_cr ≥ 10` for clad structures and to `≥ 5` for portal
frames under gravity loads only, leaving the elastic 10 we use untouched; and the
two books disagree on where `H_Ed` is measured in Eq. 5.2, ECCS recording that
EN's original *bottom of the storey* was corrected to *top* by corrigendum.

`normax/ec3/stability.py` loses its ⚠️ banner and every from-memory marker,
`ALPHA_CR_ELASTIC`, `ALPHA_CR_PLASTIC` and `ALPHA_CR_AMPLIFIABLE` keep their
values, and `normax.pipeline.Stability` cites §5.2.1(3) directly. No test
changed — `test_the_thresholds_are_the_values_the_spec_records` was already
pinning 10 and 15, and the spec now agrees with the standard rather than with
memory.

---

## P5 — swappability, one schema over two solvers

The analysis stage now has a second backend, and the pipeline above it cannot
tell which one ran. `smax` is a JAX frame solver traced end to end; OpenSees is
C++ behind a command interface whose adjoints were hand-derived element by
element and compiled years before this pipeline existed. Neither reimplements the
other, so every agreement below is a measurement rather than a round trip.
`experiments/04_backend_agreement.py` reproduces all of it.

### The gradients agree six orders below what was asked

| | worst relative |
|---|---|
| axial force | 1.4e-15 |
| end moments | 9.0e-13 |
| every Jacobian block, DDM vs traced autodiff | **1.1e-11** |
| the whole VJP through the endpoints | 8.4e-14 |
| mass, end to end | 1.7e-15 |
| **`dmass/dq`, end to end** | **3.0e-12** |

The roadmap asked for 1e-6. The agreement holds as the frame refines — 3.0e-13
at five members, 4.7e-11 at forty — growing only as accumulation does.

**The primal matched on the first attempt**, sign conventions included: the 2D
projection, the units, the supports, the loads and the Lobatto end sections all
lined up with no fitting. That is worth stating because it is the part that
usually costs a day.

### The blind block is exactly one, and the composition annihilates it

A planar frame's response separates, and this is measured rather than argued.
`∂n_ed/∂y` and `∂m_y_ed/∂y` are **exactly zero**; `∂m_z_ed/∂x` and `∂m_z_ed/∂z`
are **exactly zero**; `m_z_ed` itself is identically zero. So a model built in
the plane carries every derivative except `∂m_z_ed/∂y`, which it reports as zero.

Isolated, the gap is total: a tangent moving one interior node out of plane gives
`m_z_ed = 3.83e2` from the traced backend and `0` from the two-dimensional one.
Through the endpoints it appears as a 4.4e-3 discrepancy in the `xyz` cotangent
**confined to the normal column** — the in-plane columns agree to 8.3e-14, and
dropping the `m_z_ed` cotangent brings the whole VJP to 8.4e-14.

**It never reaches the answer.** The force density method decouples per
coordinate, so `∂xyz[normal]/∂q` is **exactly zero** for a planar arch and the
blind block is multiplied by nothing. That is why `dmass/dq` agrees at 3.0e-12
with no caveat attached.

**A uniform out-of-plane tangent is a rigid motion and strains nothing**, so it
reports zero from both backends and demonstrates nothing. The experiment pushes
one interior node for that reason.

### The cost result inverts the prediction, then confirms it

The roadmap expected DDM to be the expensive side — "T2's VJP scales with
parameter count, T1's and T3's don't". Both halves turn out to be true and the
conclusion does not follow, because the constant favours DDM enormously.

Analysis stage alone, steady state:

| members | parameters | DDM | traced | ratio |
|---|---|---|---|---|
| 5 | 22 | 2.1 ms | 315 ms | 152x |
| 10 | 42 | 3.1 ms | 393 ms | 125x |
| 20 | 82 | 7.6 ms | 452 ms | 59x |
| 40 | 162 | 36.4 ms | 585 ms | **16x** |

The predicted shape is there — DDM grows 17x over a 7.4x rise in parameters
while the traced backend grows 1.9x — but the advantage only narrows from 152x
to 16x and does not cross within the measured range. Extrapolating the two
trends puts the crossover in the low hundreds of members, past the fifty-member
gridshell this project targets.

**Warming up is not optional here.** The section slopes come from `jax.grad` of
the closed forms, so the first call at a new member count compiles a kernel and
reports 226 ms where the second reports 3.1 ms. Timed cold, the measurement is
of XLA and is reported as direct differentiation.

**And at pipeline level none of it decides anything.** The whole composition
costs 0.40 s against 10.2 s at ten members — 26x — and the analysis stage is a
small fraction of either. Form finding, the sizing bisection and two boundary
crossings dominate whoever solves the frame.

### The same descent, one solver swapped for the other

Both backends drive the descent from 0.0312 t to **0.024808755 t**, 20% lighter,
in **seven steps each**. They agree to **1.8e-8** on the mass and **4.8e-7** on
the force densities, and the C++ backend takes 2.3 s against 16.9 s.

**That is four orders looser than the 3.0e-12 the gradient agrees to at a
point, and it should be.** A line search amplifies a difference in the last bits
into a slightly different step, and seven steps compound it. Landing on the same
optimum to 1.8e-8 in the objective is the claim; identical trajectories would
have been a claim about arithmetic rather than about the optimizer.

**The starting point has to be inside the box, and it is not obviously so.** The
funicular `q` is −83.3 here, so the decade-either-side bounds of
`experiments/03` are (−833, −8.3). A tighter box clips the start, L-BFGS-B takes
its one clipped step and stops, and the run reports a converged answer *heavier*
than where it began — 0.0389 t against 0.0312 t. Both backends report it
identically, so agreement between them says nothing about whether either is
descending.

### What changed in the code

- **`normax/analysis.py` became `normax/analysis/`**, a stage package with a
  backend module each. `__init__.py` holds what the stage means — `MemberForces`,
  `Buckling`, `fixities` — and imports no solver, so the contracts are readable
  without one installed and neither backend inherits the other's dependencies.
  `smax.py` is the move of the old module; `opensees.py` is new. Behaviour is
  unchanged, pinned by the suite passing at the same count across the split.
- **A backend owns its derivative rules.** `tesseract_api.py` is now a
  dispatcher: `apply`, `jacobian_vector_product` and `vector_jacobian_product`
  ask the backend. Differentiating the forward pass there would have been a claim
  about how a solver works, and only one of the two can be traced at all.
  **The schema is untouched**, which is the whole argument.
- **The backend is read from the environment per call rather than at import**, so
  a process comparing the two can switch between them. `normax.composition.
  backend` makes the switch a block and restores the previous value, exceptions
  included.
- `normax/analysis/opensees.py` builds `forceBeamColumn` over `section('Elastic')`
  and registers `('node', n, 'coord', d)` for the in-plane axes plus
  `('element', e, 'A')` and `('element', e, 'I')` per member. Diameters are not
  registered: a section is what the solver understands, so the chain rule from
  area and second moment to a diameter is taken here with `jax.grad` of the same
  closed forms the check uses, which is what stops the two stages drifting apart
  about what a section is.
- **It refuses what it cannot represent** rather than answering anyway: no normal
  axis, a frame that is not flat along it, or a load with a component along it.
  Each would otherwise be silently projected away.

### The spike's rebuild ceiling does not reproduce

The spike recorded OpenSees exiting hard after a few hundred model rebuilds in
one interpreter, which would have made a full descent impossible in process.
**2000 parameterized sweeps ran with no crash and no drift**, flat at 3.3 ms
each. Whatever the spike hit, it is not reached by rebuilding this model with its
parameters registered, so the backend rebuilds per call and holds nothing.

### Tests

**1724, up from 1707.** Seventeen in the new `tests/test_backend_opensees.py`,
guarded in `conftest.py` on `openseespy` — the `spike` extra, which CI never
installs — and on `smax`, since every assertion there is against it.

## P5b — the topology hoisted out of the objective

`experiments/03_optimize_arch.py` runs in **29 s against over ten minutes**, and
the reason is not a faster solver. The analysis stage was rebuilding a
`smax.CompiledStructure` on every evaluation, three times per objective — once
per load case — and roughly 2,500 times across a full run of the experiment,
producing a bit-identical `Topology` each time.

### The measurement that redirected the work

Rebuilding the topology is **16.1 ms per case, 48 ms per objective, 9.4% of the
forward pass**. That alone would not have been worth a refactor. What made it
worth one is where the Python lives: `smax.topology.build_free_dof_mask` decides
the degree-of-freedom maps with `if support.fixity[i]`, a Python conditional on
what was a traced pytree leaf. **That conditional is inside compilation, not
inside the solve.** Hoisting compilation to the host therefore does not merely
save the 9% — it removes the only thing that stopped the stage being jitted.

| | eager | prepared and jitted | |
|---|---|---|---|
| analysis stage, one case | 120.4 ms | 0.17 ms | 719x |
| `q → member forces` | 115.8 ms | 0.10 ms | 1151x |
| its gradient | 304.0 ms | 0.19 ms | 1564x |
| objective, three cases | 640.4 ms | 0.17 ms | 3686x |
| value and gradient | 1381.4 ms | 0.44 ms | 3127x |

One-off compilation is 0.8 s for the value and 2.2 s for the gradient.

**The upstream change is no longer needed.** Making `Support.fixity` static in
`smax` was recorded as the precondition for jitting anything; it is not. The
blocker was compilation happening inside the trace, and preparing once removes it
without touching the dependency. The upstream field is now optional tidying.

**The sizing bisection is not 0.016% of a design.** That figure does not survive
measurement: eager, `diameter()` over twenty members is **66.5 ms, about 39% of
the forward pass**, since fifty-five halvings of a full utilization check is some
2,750 eager dispatches. The earlier conclusion still holds — do not rewrite it as
Newton — but because `lax.fori_loop` collapses under compilation, not because the
loop was ever cheap eagerly.

### The contract, and why it is symmetric

Every backend is now reached in two calls: `prepare(structure, steel, tube, *,
normal) -> Model` on the host, then `forces(model, xyz, diameters, steel, tube,
*, loads)`. This mirrors `normax.formfinding`, where `graph` is built once and
`equilibrium` consumes it — and the asymmetry it removes is that T1's derived
topology was already hoisted while T2's was not.

**Both backends take it, including the one that cannot use it.** OpenSees holds
one global model with no handle to it, so its `Model` carries only the plane the
frame lies in and every call still wipes and reassembles. A contract that fits a
solver reusing an assembly and a solver that cannot is the stage's claim; the
difference showing up as an empty model rather than as a different call shape is
the point of stating it.

`normax.structures.Structure` became a `NamedTuple`, so it is a pytree and
crosses a jit boundary as four array leaves instead of an unhashable object. It
now matches every other container in the package.

### Nothing is baked, and that is tested rather than asserted

`prepare` builds its template from a placeholder geometry and a placeholder
section, and `forces` replaces **every array leaf a result can depend on** —
`params.xyz`, the four `Section` properties, and all three material arrays, not
just the modulus. A leaf left alone would keep its placeholder, and since that is
a constant the gradient with respect to it would be a silent zero rather than an
error.

`tests/test_analysis_prepared.py` pins it. Forces come back **bitwise identical**
from templates built on the starting geometry, on the form-found geometry, and on
a deliberately absurd seed — unit modulus, unit density, a tube whose smallest
size exceeds anything the arch uses. Liveness is checked per leaf, and for the
modulus against a difference quotient rather than against zero: member forces of
a uniform-E linear frame are E-independent, so the axial force cannot tell an
injected modulus from a baked one, and only a displacement can.

Poisson's ratio is deliberately not injected. It is a constant of the backend
taken from EN 1993-1-1 rather than a field of `Steel`, so nothing upstream varies
it, and the shear modulus follows the modulus that is injected.

### What reproduces, and the one thing that does not

The floored descent — the number P4 says to quote — reproduces: **31.6% against
31.7%** lighter than the funicular arch, shortest member 311.3 against 313.4 mm,
`α_cr` 1.723 against 1.734. Utilization is 1.0 to 1e-12, the gradient agrees with
central differences to 3.7e-08 against a 2e-07 target, and `experiments/04`
returns the same 0.024808755 t in the same seven steps.

**The unconstrained descent does not reproduce: 0.0696 t against 0.0472 t.** It
is not an error, and it is worth recording why.

The function is the same function. Mass agrees to **7.3e-13** and its gradient to
**1.0e-10**, and at the collapsed design the refactored gradient matches central
differences to **7.1e-08** with utilization exactly 1.0. What differs is the path.
Traced step by step, the two descents agree to 1e-6 through step 13, 7e-3 by step
14, and O(1) by step 19 — a 1e-10 gradient difference passing through the
line search's Wolfe threshold tests, then amplifying about tenfold per iteration.

**The unconstrained run was never a determinate quantity.** P4 already recorded
that it has no interior optimum and "goes on trading members for stubs until
something stops it". Measured directly: a **1e-12 relative nudge on the starting
`q` moves its endpoint by 4%**, and the spread across nudges of 0, 1e-12, 1e-9 and
1e-6 is 5.8%, with step counts of 27 to 35. The floored descent under the same
nudges spreads **0.2%**. One problem has an optimum and the other has a plateau,
and only the first has an endpoint worth pinning. Quote the floored number.

### Tolerances that moved, and why

`tests/test_tesseract_parity.py` gains `TOLERANCE_MOMENT = 1e-11` for the end
moments and the factors read off them, against `1e-14` for everything else. The
cause is the arch rather than the boundary: a funicular shape carries its design
case axially, so an end moment is a near-cancellation worth 4e-4 of `N·L`, and its
relative precision is set by that larger scale. Measured, the composed and
in-process designs are **exact** at the Class 3 ratio and differ by 8.2e-13 at
Class 2, where the axial force they came from differs by 7.1e-16 — one ulp,
amplified about a thousandfold. The old `1e-14` held on luck about which values
round-tripped bit-exactly, not on anything about the boundary.

### What was left alone

**The Tesseract path still prepares per call.** `normax/composition.py` crosses a
stateless boundary, so nothing can be reused between crossings without a cache
inside `_backend_smax` keyed on topology. The wrappers were adapted to the new
signatures and are no slower than before; the caching is separate work.

`normax.pipeline.envelope` is not jitted. It is the oracle the Tesseract
composition is measured against, and burying a compilation inside it would muddy
that role. The jit sits at `normax.optimization.descend`, where the value and
gradient are already built together, and at `build` in `experiments/03`.

**An annealing schedule pays one compilation per round.** The sharpness reaches
the objective as a captured constant rather than an argument, and a captured array
is baked into the program instead of traced, so each round traces afresh. Five
rounds is 11 s of compilation against 830 s of eager evaluation; making it one
would mean giving up `descend` being generic over any scalar objective.

### Tests

**1732, up from 1724.** Eight in the new `tests/test_analysis_prepared.py`.

### The backend comparison was measuring Python, and it reversed

`experiments/04` timed the traced backend uncompiled, re-prepared inside the timed
call, and without waiting for JAX to finish. Fixing all three turns the headline
around.

| members | DDM [ms] | traced [ms] | traced / DDM |
|---|---|---|---|
| 5 | 2.2 | 0.3 | 0.15x |
| 10 | 3.0 | 0.7 | 0.23x |
| 20 | 7.7 | 2.0 | 0.26x |
| 40 | 35.8 | 8.9 | 0.20x |

**At the stage, tracing is four to seven times cheaper than the DDM sweep, not
fifteen to a hundred and eighty times dearer.** The earlier figures were an
artifact: `stage_cost` called `prepare` inside the function it timed, so the
traced side rebuilt an assembly per call and compiled nothing, and 388–545 ms of
Python dispatch was being compared against a C++ sweep.

Three defects, all of which mattered:

- **Preparation inside the timed call.** Both backends are now prepared once, which
  is what the stage's contract says to do. It costs OpenSees nothing — its domain
  is rebuilt per call regardless — and it is the whole of the traced backend's
  advantage.
- **No compilation.** The traced Jacobian is now `eqx.filter_jit`-wrapped, with the
  warm-up excluding the compile exactly as it already excluded the kernel the
  section slopes need on the other side.
- **No blocking.** `steady` timed dispatch rather than execution, so it charged the
  C++ backend for completing work and the traced one for queueing it. It now waits
  on whatever the call returns, which is why the timed callables return their
  results.

The composition still favours DDM, **2.6x to 3.2x**, and that is honest: the
composed path crosses a stateless Tesseract boundary, so every crossing rebuilds
the solver behind it and the traced side pays its eager assembly inside the
Tesseract. That is the deferred caching work, and the two panels of
`figures/04_backends.png` now say which is which in their titles.

The composed descent reaches the same **0.024808755 t in seven steps** either way,
agreeing to 1.9e-8 on the mass and 4.8e-7 on the force densities.

### Compilation is paid before the clock starts, and reported

`normax.optimization.value_and_gradient` is new: the compiled value-and-gradient of
an objective, exposed so a caller decides when compilation is paid. `descend` takes
one through a new `gradient` argument and builds its own only if not given one, and
`optimize` builds one per round — a round is a different program from its
neighbour, its sharpness being a captured constant, so nothing carries over.

`experiments/04` now compiles each objective, times that call on its own, and hands
the compiled program to `descend`:

| backend | descent | per step | compile |
|---|---|---|---|
| smax | 7.6 s | 1082 ms | 6.60 s |
| opensees | 2.3 s | 335 ms | 0.41 s |

**A descent of seven steps was more than half compilation on the traced side**,
which is why the earlier 16.9 s against 2.3 s said as much about the tracer as
about either solver. Reported separately it is a fixed cost per objective, so it
matters here and vanishes over a few hundred steps.

`steady` grew a warm-up for the composed timings too, which removes the artifact
where one mesh size read 3.1x against 27x and 26x either side because a single
measurement had caught a warm cache.

### The Tesseract's solve is compiled once, its assembly still per crossing

`tesseracts/analysis/_backend_smax.py` gains `_member_forces`, an
`eqx.filter_jit`-wrapped call taking a prepared model and returning the member
forces. **The only thing that matters is that the wrapper lives at module scope**:
the compilation cache belongs to the wrapper, so one built inside `solve` is a new
cache every crossing and compiles afresh every time.

Nothing else was needed. The cache keys itself on the shapes and dtypes of the
array leaves, so a second load case reuses the program and a second frame size
gets its own, and every array a derivative might be taken through arrives as an
argument — the loads included, since they reach the call inside the model as
ordinary leaves rather than as folded constants.

**It survives the derivative endpoints tracing it, which was the open question.**
`jvp` and `vjp` call `solve` inside `jax.jvp` / `jax.vjp`, and a compiled call
nested in a trace stays compiled rather than being unrolled into it. Measured on a
VJP through the stage: **331.9 ms eager against 4.7 ms, 71x**, cotangents agreeing
to 4.6e-13.

| composed, ten members | before | after | |
|---|---|---|---|
| value | 180.1 ms | 75.3 ms | 2.4x |
| value and gradient | 843.5 ms | 248.3 ms | 3.4x |
| of which `prepare` | 45.7 ms | 35.7 ms | — |
| of which the solve | 415.2 ms | 0 ms | compiled away |

The descent comparison closes with it: **1.2x for the C++ backend against 3.2x**,
at 422 ms per step against 363, both compiled before the clock started. Across the
sweep the composed gradient is **1.12x to 1.31x**, monotone in the member count:

| members | smax | opensees | ratio |
|---|---|---|---|
| 5 | 0.230 s | 0.206 s | 1.12x |
| 10 | 0.265 s | 0.226 s | 1.17x |
| 20 | 0.284 s | 0.248 s | 1.15x |
| 40 | 0.423 s | 0.323 s | 1.31x |

**`steady` reports a median rather than a mean, and that mattered.** At a few
hundred milliseconds a crossing, one sample landing at three times the rest moved
a mean of five enough to reverse which backend looked faster: the first run after
compiling read 0.25x at ten members and 4.33x at forty, non-monotone in both
directions. The median over seven samples is monotone and the spread is 1.1x–1.3x.

**The prepared model is deliberately not cached.** That needs a key over the
topology arriving in the inputs, and the ~36 ms it would save is now the largest
remaining item rather than a decisive one. It is left out on purpose; the reason
the loads must then be passed explicitly rather than baked is recorded in
`ROADMAP.md` under P5b, since it is the trap anyone adding that cache will meet.

### Parity now compares two compiled paths, which is what it meant to compare

Compiling one side of the boundary broke the parity tests, and loosening the
tolerance would have been the wrong fix. Measured against the composed path:

| | eager oracle | compiled oracle |
|---|---|---|
| `n_ed` | 1.8e-15 | **0.0** at Class 2, 4.7e-16 at Class 3 |
| `diameters` | 3.9e-14 | 4.9e-15 |
| `mass` | 3.0e-15 | 6.7e-16 |

So the boundary is still transparent — bitwise, in one case — and what the failing
tests had started measuring was the difference between compiled and eager
arithmetic. `tests/test_tesseract_parity.py` now compiles its in-process oracle,
which restores the premise its tolerance was written under: both sides run the same
program over the same inputs.

**One genuine looseness remains and it is structural.** An enveloped design agrees
to 1.0e-13 on the axial force and 6e-14 to 1.1e-13 on everything downstream, the
mass included. In process the three load cases compile into one program; across
the boundary one solve is compiled and called three times, so the same sums are
accumulated in different units. That cannot be equalised without giving up one
side's structure, so `TOLERANCE_PARITY_ENVELOPE` is 1e-12 and covers the whole
container — an early draft exempted the mass and the utilization, which measurement
showed was false.

`normax.pipeline.envelope` itself is still not compiled. The test compiles it;
the module stays the readable reference.
