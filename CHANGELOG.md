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
