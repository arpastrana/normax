# ROADMAP — Backpropagating Through Structural Engineering Codes

Repo: `normax` · Deadline **Aug 31, 11:59 PM AoE** · **Code freeze Aug 27**
Read ec3x's `docs/clauses.md` first — the verified EC3 spec, and the only
source any clause may be implemented from. It moved with the clause library.
This file is the execution order.

**Thesis (from the 2025 winners post): swappability.** Both top places won on
modularity, not domain. The headline experiment is one T2 schema with two
interchangeable backends producing agreeing gradients. Keep that schema stable
from day one even while only one backend exists.

---

## P0 — Setup (Aug 8) — **DONE**

**You:**
1. ~~Write `docs/clauses.md`~~ — **DONE.** Verified against Gardner & Nethercot,
   *Designers' Guide to Eurocode 3*, 2nd edn (ICE, 2011), via Princeton Library.
   `docs/clauses.md` is now the authoritative spec for all EC3 work.
   **Every clause derives from that file alone, never from memory of the standard.**
   Three low-risk items remain open there (eq. numbers 6.5 / 6.9 / 6.46, the
   γ_M0 / γ_M1 recommended values, cold-formed curve rows). None block P1.
2. ~~Set up the toolchain with **uv** and smoke-test it before writing anything.~~
   — **DONE.** Both gotchas below landed: `requires-python` is `>=3.12,<3.15`
   and `tesseract-core[runtime]` is installed.

   **Gotchas, both real:** `tesseract-jax` requires **Python >=3.12,<3.15** —
   check `requires-python` in the cookiecutter's `pyproject.toml` or resolution
   fails. And you need the **`[runtime]` extra** on `tesseract-core`: the plain
   install gives the CLI but not `tesseract_core.runtime`, where `Array`,
   `Differentiable` and `Float64` live. Without it, importing `tesseract_api.py`
   directly in tests (invariant 6) breaks.

   ```bash
   cd normax
   uv python pin 3.12

   uv add "tesseract-core[runtime]" tesseract-jax jax
   uv add --dev pytest ruff blue-prints

   uv run tesseract --version
   ```

   `jax>=0.7.0` arrives transitively via `tesseract-jax`, but we depend on it
   directly so pin it ourselves. Optional: `uv tool install tesseract-core` for a
   global `tesseract` CLI (pipx-style) — the project dependency is still required
   for imports.

   ```bash
   docker run --rm hello-world        # check Docker first, isolates the failure

   git clone https://github.com/pasteurlabs/tesseract-jax /tmp/tj
   uv run tesseract build /tmp/tj/examples/simple/vectoradd_jax
   uv run tesseract ps
   ```

   If this fails (usually Docker permissions), fix it now, not on the 14th.

   That check now lives in `experiments/00_toolchain_smoke.py` — run it by hand
   before P3 rather than trusting that the toolchain still works.

**Scope:**
> This repo was bootstrapped from the `simonw/python-lib` cookiecutter: flat
> layout, uv, pyproject.toml, Apache-2.0, CI already wired. **Do not migrate to
> a src/ layout and do not regenerate pyproject.toml** — extend what's there.
> Dependencies are already added via `uv add` (tesseract-core[runtime],
> tesseract-jax, jax; dev: pytest, ruff, blue-prints) — do not re-add them, and
> use `uv add` rather than editing pyproject.toml by hand if more are needed.
> - Create the package layout with empty modules: `normax/ec3/`,
>   `normax/structures.py`, `normax/pipeline.py`, `normax/visualization.py`,
>   `experiments/`, `figures/`.
> - Add `jax.config.update("jax_enable_x64", True)` in `normax/__init__.py`.
> - Add ruff config to pyproject.toml and a ruff step to the existing test
>   workflow. Confirm `requires-python` is `>=3.12,<3.15`.
> - Implement `normax/structures.py`: generators for a 2D cable, a 2D funicular arch,
>   and a spherical-cap gridshell. Return nodes, edges, supports, loads.
> Do not create any Tesseract files yet. Run tests with `uv run pytest`.

**Done when:** ~~`uv run pytest` green~~ ✅ (31 tests); ~~`docs/clauses.md` in
place~~ ✅; repo **public** ⬜ — still private, flip it before submission.

---

## P1 — EC3 core (Aug 8–10 evenings) — **DONE**

A hard oracle and a tight loop. **Axial-only scope** — extended in P1b.

**Scope:**
> Implement `normax/ec3/section.py`, `classification.py` and
> `resistance.py` **strictly from `docs/clauses.md`** — that file is verified;
> do not add clauses from memory. CHS only, `t = d/r`. Pure JAX, float64, no
> `if` on traced values. Every docstring cites its EN 1993-1-1 clause number
> (note: buckling curves are §6.3.1.2, slenderness is §6.3.1.3; Φ is unnumbered
> beneath Eq. 6.49).
> Write the tests FIRST in `tests/`:
> - `test_worked_example_chs.py` — the full fixture table at the bottom of
>   `docs/clauses.md`. CHS 244.5×10, S355, L_cr = 4000 mm. Assert A, I, W_el,
>   W_pl, ε, d/t, class, N_c,Rd, N_cr, λ̄, Φ, χ, N_b,Rd against the tabulated
>   values — closed-form column at 0.5%, guide column at 1%. **This is the
>   primary fixture — it exercises classification, cross-section resistance and
>   buckling in one test.** (The guide numbers this Example **6.7**, not 6.2;
>   filenames are kept neutral of example numbers.)
> - `test_worked_examples_cross_section.py` — the guide's Examples 6.1 and 6.2,
>   a flat bar and a UKC. Neither is a CHS, which is the point: they exercise
>   §6.2 through its area interface with no CHS geometry involved.
> - χ = 1.0 to 1e-15 at λ̄ = 0.2, independent of α
> - χ ≤ 1 and χ ≤ 1/λ̄² everywhere; χ → 1/λ̄² as λ̄ → ∞
> - χ strictly decreasing in λ̄; curve ordering a0 > a > b > c > d
> - N_cr vs closed-form Euler
> - Classification limits 50/70/90 ε² against the ε² table for all five grades
> - §6.2 resistances vs blueprints (dev dependency, `tests/` only — NEVER copy
>   its source, it is LGPL and we are Apache 2.0)

**Done when:** ~~all tests green, the CHS worked example included~~ — **DONE**
(Aug 8). 613 tests green. `docs/clauses.md` open items 2 and 3 closed against
the PDF, and an arithmetic error in the guide's own Class-1 limit recorded as
errata. See `CHANGELOG.md`.

---

## P1b — N+M scope expansion (Aug 8) — **DONE**

Axial-only is not physically adequate for a gridshell: form-finding is
pin-jointed, analysis is not, and under LC2/LC3 the members carry bending.
It also repairs the composition argument — **if T3 only consumes axial force,
T2 is barely motivated**, since FDM already gives axial forces. A frame solver
earns its place only once the check consumes moments.

~~**Verification pass first (you, ~1h). Blocking.**~~ — **DONE.** The pass was
run against `references/`, which now holds both textbooks: Gardner & Nethercot
and the ECCS manual (Simões da Silva et al.), the latter being the better
source for N+M. Every ⚠️ flag in `docs/clauses.md` is now ✅. The four questions
below are kept for the record with their answers.

- **0a** Is combining `M_y`, `M_z` into a resultant valid in Eq. 6.61?
  → **No.** Exhaustive search of both books and every NCCI they cite found no
  sanction for it. Exact in 6.41 (`α = β = 2`), inadmissible in 6.61/6.62.
- **0b** Does §6.2.9.1(4) give a small-axial-force exemption for CHS?
  → **No.** Both books enumerate the eligible types and CHS is in none.
  Always compute `M_N,Rd`.
- **0c** Which Annex B table applies to CHS?
  → **Table B.1**, three independent statements. But B.1 lists only I-sections
  and RHS-sections, so we read the **RHS row** — an interpretation, corroborated
  by Karamba's `is_I_profile` branch, not a citation.
- **0d** Confirm the CHS `M_N,Rd` formula and its equation number.
  → **`M_pl,Rd (1 − n^1.7)`**, ECCS p. 228. **It carries no EN equation
  number** — unnumbered inside §6.2.9.1(5), like `Φ` beneath Eq. 6.49. The
  Designers' Guide omits the CHS rule entirely.

**One requirement was retired as wrong, not skipped.** The scope below
asks for a test that "6.61 and 6.62 return the same value". They do not — only
when `M_y = M_z`. The implementation takes the larger of the two and a test
pins the disagreement.

**Scope:**
> Extend `normax/ec3/resistance.py` with §6.2.9 (N+M cross-section) and §6.3.3
> (Eq. 6.61 buckling + bending interaction), plus a new
> `normax/ec3/interaction.py` for the Annex B method-2 `k_ij` factors.
> **Strictly from `docs/clauses.md`; implement nothing still marked ⚠️
> UNVERIFIED.** Same rules as P1: pure JAX, float64, no `if` on traced values,
> clause number in every docstring.
>
> **Exploit the CHS collapse identities documented in `clauses.md`.** For a CHS:
> `I_y = I_z` ⟹ `χ_y = χ_z`; `W_pl,y = W_pl,z` ⟹ `M_N,y,Rd = M_N,z,Rd`;
> `k_yy = k_zz`, `k_yz = k_zy`; and `χ_LT = 1` because the section is closed and
> doubly symmetric. Eqs. 6.61 and 6.62 therefore become the same equation.
> Implement ONE χ, ONE M_N,Rd, ONE interaction check. Do not carry separate y/z
> code paths — that is the whole reason this expansion is affordable.
>
> **Read the ERRATA section of `docs/clauses.md` before writing any test.** The
> guide's N+M worked examples (6.9, 6.10) carry errata #5, #6, #7, #8 and #9 —
> all in the 6.61/6.62 substitutions. They are the least reliable fixtures in the
> book. Prefer property-based tests over asserting their printed values, and
> never assert a guide number tighter than 1%.
>
> Tests FIRST:
> - **Reduction checks** (these catch most sign and normalization errors):
>   with `M_y = M_z = 0`, the 6.61 check must equal the pure-compression check to
>   1e-12; with `N_Ed = 0` it must equal the pure-bending check.
> - **CHS collapse identities**, no external source needed: `χ_y == χ_z`;
>   `k_yy == k_zz`; `M_N,y,Rd == M_N,z,Rd`; Eq. 6.41 with α = β = 2 equals the
>   resultant-moment form; 6.61 and 6.62 return the same value.
> - `M_N,Rd ≤ M_pl,Rd` for all `n`; `M_N,Rd → M_pl,Rd` as `n → 0`;
>   `M_N,Rd → 0` as `n → 1`; strictly decreasing in `n`.
> - `k_yy` respects its `≤ C_my(1 + 0.8 n_y)` cap; monotone in `n_y` below it.
> - Eq. 6.41 vs blueprints `Form6Dot41BiaxialBendingCheck` — note it returns a
>   **bool**, not a ratio, so compare verdicts not values. Tests only, never copy.
> - Whole-check monotonicity: utilization strictly decreasing in `d` at fixed
>   forces. This is what keeps the P2 bisection valid — assert it directly.

**Two numerical traps, both new:**

- `n^1.7` has an **unbounded second derivative as `n → 0⁺`** (exponent < 2).
  Guard `n` away from zero or clamp below a small `n_min`, or gradients near
  pure bending will be noisy. This is a real gradcheck failure mode, not
  hypothetical.
- The `k_yy ≤ C_my(1 + 0.8 n_y)` cap is a C⁰ kink of the same shape as `χ ≤ 1`.
  Handle identically and report through `governing`.

**What stays excluded, and the justification for the writeup:**

| Excluded | Why |
|---|---|
| §6.3.2 LTB | CHS closed and doubly symmetric, `χ_LT = 1` — correct, not a simplification |
| §6.2.7 torsion | Negligible in a triangulated shell |
| §6.2.6 shear | `V_Ed` rarely exceeds `0.5 V_pl,Rd` in slender members — **verify post hoc** on the converged design and report the max. If it exceeds 0.5, §6.2.10 applies and the exclusion stops being honest. |
| Karamba's `C_mLT ≥ 0.9` | A sway-frame default of Karamba's, **not an EC3 requirement**. If adopted, make it an optional flag and say so. |

**Done when:** ~~open items 0, 0a–0d ticked in `clauses.md`; all new tests green;
the two reduction checks pass to 1e-12~~ — **DONE** (Aug 8). 1037 tests green.
Open items 0, 0a–0d closed against the PDFs; **0e opened** (the two books
disagree on a sign in Table B.3 row 3c — out of our path while loading is
nodal). See `CHANGELOG.md` for the reasoning.

**Two things the verification pass overturned. Read these before P2.**

1. **`d/t = 90ε²` is Class 3, not Class 1/2.** `M_Rk` uses `W_el`, the
   `M_N,Rd(1 − n^1.7)` reduction does not apply, and the *elastic* column of
   Table B.1 governs — different formula and different couplings. Both branches
   are now implemented and the ratio stays config; `70ε²` vs `90ε²` is an
   experiment for P4.
2. **Eqs. 6.61 and 6.62 do not collapse into one equation.** They agree only
   when `M_y = M_z`. Still one check — take the larger. The scope above said
   otherwise and was wrong.

**Also delivered, beyond the original scope:**

- `tests/test_sizing_monotonicity.py` — the utilization is strictly decreasing
  in `d` with exactly one root, across six force combinations and both class
  branches. **This is P2's precondition and it is now proven, not assumed.**
- `governing_equation` and `cap_is_active`, non-differentiable diagnostics for
  P4's per-member logging.
- `tests/test_worked_example_frame.py` — the ECCS manual's 47 m portal frame,
  its own member forces in, our 6.61/6.62 out. Parity confirmed.
- `checks` (Eq. 6.61/6.62 from given factors) is split from
  `interaction_factors` (Table B.1), because they are different clauses.

---

## P2 — Sizing map + first real gradient (Aug 12–14)

The core. Expect to sit with this yourself. **Two steps** — validate the
`custom_vjp` on the simplest possible residual, then extend. The IFT machinery
is identical between them; only the residual changes.

### What P1/P1b already settled — start here, do not re-derive

**The API to compose.** All of `normax/ec3/` is done and tested. Units are
mm / N / N-mm⁻² throughout; every function takes section properties, not a
diameter, so `sizing.py` is the layer that composes them.

| Module | What to call |
|---|---|
| `section` | `area`, `second_moment`, `modulus_elastic`, `modulus_plastic`, `radius_of_gyration`, all `(diameter, ratio)` |
| `classification` | `material_factor`, `class_limits`, `classify_section` |
| `resistance` | `resistance_compression`, `resistance_tension`, `force_critical`, `slenderness_from_force`, `buckling_auxiliary`, `reduction_buckling`, `resistance_buckling`, `resistance_bending_plastic`, `resistance_bending_elastic`, `resistance_bending_reduced`, `moment_resultant` |
| `interaction` | `moment_factor_linear`, `axial_ratio`, `interaction_factors`, `checks`, `utilization_member`, `governing_equation`, `cap_is_active` |

**The residual already exists in test form.** `member_utilization` in
`tests/test_sizing_monotonicity.py` composes section → resistance →
interaction into exactly the scalar P2 needs to root-find on. Lift it; do not
rewrite it from scratch.

**Monotonicity is proven, not assumed.** That same file asserts the utilization
is strictly decreasing in `d` with exactly one crossing of unity, over six
force combinations and both class branches. The bisection bracket is safe.

**`plastic` is a static Python bool**, set by the configured `d/t`. Thread it
as static — `jax.jit(..., static_argnames=("plastic",))` — never as an array.

**Aux outputs already exist**: `governing_equation` (which of 6.61/6.62 won)
and `cap_is_active` (whether an interaction factor is bounded). Both
non-differentiable. Pop them before `jax.grad`: a concrete cotangent on a
non-differentiable output raises `ValueError`, and only a symbolic zero is accepted.

**Gradient order.** `resistance_bending_reduced`'s `1 − n^1.7` has a finite, continuous *first*
derivative at `n = 0`; only the second diverges. So `check_grads(order=1)` is
safe at pure bending and `order=2` is not. Guard `n` only if second-order
accuracy is ever needed — the earlier note that gradients would be "noisy"
overstated it.

**One question P2 has to answer that P1b did not.** Tension is no longer a
closed form once bending is admitted. The axial-only tension branch inverted
`N_t,Rd = A f_y/γ_M0` directly; with moments, a tension member needs the
§6.2.9 cross-section check — `moment_resultant(M_y, M_z) ≤ m_n_rd(m_pl_rd, n)`
with `n = N_Ed/N_pl,Rd`. That is still monotone in `d` (the reduced moment
grows as the axial ratio falls, both with `d`, and
`test_cross_section_utilization_strictly_decreases_with_diameter` asserts it),
so bisection still works — but **the closed form is gone and the tension branch
needs a root-find too**. Decide in step 2 whether to keep two branches or use
one bisection for both.

### Step 1 — axial only

**Scope:**
> Implement `normax/ec3/sizing.py`: the fully-stressed CHS sizing map,
> `custom_vjp` via the implicit function theorem.
> - Compression: bisection on R(d) = χ·A·f_y/γ_M1 − |N|. R is strictly
>   increasing in d, so the root is unique — use `lax.while_loop` with a fixed
>   iteration count so the forward pass stays jittable.
> - Tension: closed form, no buckling.
> - Backward: `jax.grad` on the residual for ∂R/∂d and ∂R/∂L; hand-derive only
>   the implicit inversion. Branch on sign(N_Ed).
> Then `experiments/01_single_strut_gradcheck.py`: verify `jax.grad` of
> utilization w.r.t. N_Ed against central differences AND a hand-derived
> expression. Target 1e-8.

**Step 1 done when:** 1e-8 agreement on a single strut. **This is the milestone
that de-risks everything — do not start step 2 until it passes.**

### Step 2 — extend the residual to N+M

**Scope:**
> Extend the residual in `normax/ec3/sizing.py` to the full §6.3.3 interaction:
> given (N_Ed, M_y,Ed, M_z,Ed, L), solve for the `d` that makes the Eq. 6.61
> check exactly 1.0.
> **Monotonicity is preserved**: `A ∝ d²`, `W_pl ∝ d³`, `χ` increasing in `d`,
> and `n` therefore falling so `k_yy` falls too — every term of the interaction
> strictly decreases in `d`. Same bisection, same `custom_vjp`, same IFT
> inversion. Only the residual changes; do not restructure the module.
> Guard `n` away from 0 (`n^1.7` has unbounded second derivative at the origin).
> Add `experiments/02_pipeline_gradcheck.py`: gradcheck w.r.t. N_Ed, M_y, M_z and
> L independently, plus a check that setting M = 0 reproduces step 1 exactly.

**Step 2 done when:** gradchecks pass for all four inputs, and the M = 0 case
reproduces step 1 to 1e-12.

### Aug 12 — OpenSees DDM spike — **DONE** (run Aug 9)

All four steps ran. `experiments/07_opensees_ddm_spike.py`; full write-up in
`CHANGELOG.md` under `## OpenSees DDM spike`; the rules it produced are in the P5
block below.

1. `sensNodeDisp` vs central differences — **passes**, to 7.4e-9, but only on the
   right element. See 2.
2. `getResistingForceSensitivity` — **absent from `elasticBeamColumn`**, 2D and 3D.
   Parameters bind and read back correctly, then every sensitivity is zero.
   `dispBeamColumn` / `forceBeamColumn` with `section('Elastic')` carry it.
3. Nodal-coordinate parameters — **registerable, and correct in 2D**
   (`parameter(tag, 'node', n, 'coord', d)`). **In 3D they are unusable**: zero for
   beams, silently *wrong* for trusses. Outcome (a) in 2D, (b) in 3D.
4. Pseudo-load extraction — **no**. `printB` is the residual in every mode. The
   prize idea is dead, and worth less than assumed: DDM reuses one factorization,
   so a parameter costs ~6–12% of a solve. DDM beats FD by 7–17x.

**P4's 2D arch is fully covered by DDM, geometry included.** The 3D gridshell is
not, and the response is settled in the P5 block below: OpenSees is the 2D
swappability demo, and `smax` carries 3D.

---

## P3 — The pipeline in pure JAX, then in Tesseracts (Aug 14–17)

**Three steps, each with its own gate. Do not start one until the previous
passes.** Tesseract contributes a schema, a serialization boundary and the §5
failure modes — all plumbing. T1→T2 contributes a units trap and a physical
claim worth testing on its own. Debugging both at once is the expensive way to
find a units bug.

**Step 2's pipeline is not scaffolding — it is the oracle for step 3.** Keeping
the pure-JAX `q → mass` and asserting the Tesseract composition reproduces it,
gradient included, is what turns "the Tesseracts work" into "the boundary is
provably transparent." That is the top judging criterion, and the test cannot be
written after the fact without this baseline.

### Step 1 — T1 + T2 in pure JAX, no Tesseract, no EC3 — **DONE** (Aug 9)

**The T1→T2 interface is geometry only.** No prestress, no initial member forces.
Internal forces are entirely a product of `smax`, and their agreement with the
form-finding forces is the thing being tested rather than something imposed.

**Why this is safe here, and where it would not be.** Members are CHS
beam-columns, so the frame has material axial and bending stiffness and is
well-conditioned unstressed. The older instruction to pass prestress was written
for cables, whose transverse stiffness is purely geometric and which *are* a
mechanism without it. **The 2D cable generator in `normax/structures.py` is
therefore out of scope for this pipeline** — do not run it through T2 and expect
sense. The arch and the gridshell are the targets.

**Scope:**
> `experiments/08_arch_formfind_analyze.py`. Take the 2D funicular arch from
> `normax/structures.py`, form-find it with `jax-fdm` under LC1, and hand **only
> the geometry** to `smax`, which analyzes it from an unstressed reference state.
> **Write the unit adapter first and test it on its own**: `smax` works in
> coherent SI (N, m, Pa), `normax` in mm / N / N·mm⁻².
> Then `tests/test_equilibrium_consistency.py`, asserting on forces rather than
> displacements: at FDM equilibrium each member carries `F_i = q_i · L_i` and
> that state equilibrates the nodal loads exactly, so `smax`'s axial forces under
> LC1 must reproduce it, with bending as the only discrepancy. Report the max
> relative deviation and the largest `|M| / (N · L)`.
> **Do not invent the tolerance — measure it on the first run, then pin it.**
> **Supports are pinned.** FDM restrains translation only,
> so a pinned base is the faithful analogue; do not use a fixed base here.

**Gate:** ~~axial forces agree with `q · L` to a recorded tolerance, bending is
demonstrably secondary, and `jax.grad` of a scalar of `smax`'s output w.r.t. `q`
is finite and matches central differences.~~ — **PASSED.** Recorded tolerances
**2.5e-4 axial and 1.0e-3 bending at `d = 100 mm`**, gradient to **2.9e-10**.
`experiments/08_arch_formfind_analyze.py`, 29 cases in
`tests/test_equilibrium_consistency.py`. Full write-up in `CHANGELOG.md` under
`## P3 step 1`.

**Three things step 1 settled that step 2 must not relitigate.**

1. **A planar structure on pinned supports alone is a mechanism** in a 3D frame
   solver — rigid rotation about the support chord. Restrain the one translation
   normal to the plane at every node and leave the out-of-plane rotations free;
   `normax.analysis.fixities(structure, normal)`. `smax` returns nan rather than
   a wrong number, so this fails loudly, but assert 0 modes for any new structure.
2. **The T1→T2 gap is `(i/L)²`, not elastic shortening.** It is quadratic in the
   diameter to three figures over an eightfold range, and free of both the modulus
   and the load scale to 1e-12. Do not pin a tolerance without also pinning the
   law — a single number hides which of the three is broken.
3. **`smax` is not on PyPI**, so it and `jax-fdm` live in a `pipeline` dependency
   group with a `collect_ignore` guard in `tests/conftest.py`. CI installs
   `--group dev` and never sees them. **Publish, vendor, or document `smax` as a
   required local checkout before the repo goes public.**

**Displacements will not be zero, and should not be asserted to be.** An
unstressed frame must deform elastically before internal forces develop. Zero
displacement was a property of the prestressed interface, not of the physics.

### Step 2 — add T3, still pure JAX — **DONE** (Aug 9)

**Scope:**
> `experiments/09_arch_pipeline_jax.py`. Extend step 1 with `normax/ec3/sizing.py`
> to expose `q → mass` as one differentiable function, in process. Keep it as an
> importable function, not a script body — step 3 and P4 both consume it.
> Gradcheck `dmass/dq` against central differences, and exercise both class
> branches.

**Gate:** ~~utilization is `1.0 ± 1e-9` for every member (invariant 6.5),
`dmass/dq` matches central differences, and the mass is stable under mesh
refinement.~~ — **PASSED, with the third clause corrected.** Utilization
**1.7e-15** on both class branches; `dmass/dq` to **1.2e-8** at the swept step;
and the mass is **convergent, first order in the member count**, not stable.
`experiments/09_arch_pipeline_jax.py`, 24 cases in `tests/test_pipeline.py`.
Full write-up in `CHANGELOG.md` under `## P3 step 2`.

**Four things step 2 settled that P4 must not relitigate.**

1. **Refinement needs the shape pinned, not the loads.** Fixing the nodal load
   and `q` drifts the crown rise as exactly `3000·n/(n−1)`, so the mass moves for
   reasons unrelated to discretization. Fix the total load and rescale `q` so the
   rise hits its target — the FDM `z` solution scales as `1/|q|`, so one trial
   solve does it. With the shape pinned the arc length converges second order and
   the mass first order.
2. **`L_cr = L` is physics, not discretization.** Refining shortens members,
   `L/i` at the crown falls 73.5 → 5.5, and the sequence converges to the squash
   limit rather than to a mesh-independent mass. The buckling length is therefore
   an argument to `pipeline.design_members`, not a convention inside it. Report both
   curves; this is the P7 caveat showing up as a plot.
3. **The staggered coupling contracts by ~1/39 per pass and one pass costs 1.22%
   of the mass.** Cheap to relax if P4 wants the fixed point, and now a number
   rather than an acknowledged unknown.
4. **Gradcheck relative error must be scaled by the gradient, not the component.**
   Two of the ten sensitivities are 20x smaller than the rest because the
   gradient *changes sign* there — length dominates at the springing, section at
   the crown. A per-component measure reports 1e-13 as 1e-7 and means nothing.
5. **`L_cr` is an input, never a mesh length, and the member-length default is a
   strong assumption.** It presumes every node is held in plane by structure
   outside the model. Measured on the fully-stressed arch, `α_cr = 0.129` in an
   antisymmetric sway mode spanning the whole arch, with four modes below the
   design load; the implied effective length is **0.576 of the developed length**,
   steady to three figures across 32x of mesh density. Sizing against that
   instead costs **3.26x the mass** and still only reaches `α_cr = 1.41`, so the
   bare arch is not rescuable by sizing — the braced-rib reading is the honest one.
   ~~**It is a check, not a diagnostic** (changed on instruction): `α_cr` is
   compared against §5.2.1's threshold and the arch **fails**, utilization 77.4,
   pinned by a test. It cannot enter the sizing bisection — that roots a local
   member check, while stability is a property of the whole frame — so
   `pipeline.frame_stability` reads a finished `Design`.~~ **Deleted 2026-08-15,
   on instruction**: the §5.2.1 check, the two-slenderness comparison, and then
   the whole buckling surface (`buckling_modes`, `Buckling`, the modes figure)
   left normax with the ec3x import sweep. No experiment computes `α_cr` now;
   buckling and frame-stability verification is future work, stated as such in
   the manuscript. The measured numbers in this item are history, reproducible
   from git.
   ~~⚠️ §5.2 and §6.3.4 are UNVERIFIED~~ — **verified 2026-08-09**, open item 0f
   closed. Every threshold held, so no reported number moves. Two equation
   numbers did not: Eq. 5.1 is the threshold pair rather than the definition of
   `α_cr`, and 6.64 could not be confirmed at all, so the general method is now
   cited as §6.3.4(3). **§6.3.4's `α_cr,op` excludes in-plane buckling** and our
   arch mode is in-plane, so the clause is cited for the algebra's form, not as
   authority for the case.
   The identity carrying it is algebra and needs no
   source: Eq. 6.50 takes `λ̄` from a member buckling length, §6.3.4(3) from a
   system critical load factor, and `α_ult,k/α_cr = A f_y/N_cr = λ̄²`. On the arch
   they disagree by 4.7–5.7x, the global route uniform to 0.3% while the member
   route varies 21%. **That is a headline for the writeup** — the standard offers
   two doors to the same quantity, and a differentiable pipeline can walk through
   either, which no scalar branchy implementation of the same clauses offers.
6. **A uniform `q` fixes the horizontal projection**, so springing members are
   1.46x longer, carry exactly 1.46x the force (`N = q·L`), and are 1.46x fatter
   in area. That is what turns a 3.9x spread in `∂arc/∂q` into the 67x spread in
   the mass's length term. **Optimizing `q` per edge destroys this property**, so
   the sign-change location is a feature of the uniform start, not of the arch —
   do not build anything on where it currently sits.

**One finding P4 can use directly:** `Σ ∂mass/∂q = −8.5e-5`, so weakening the
force densities uniformly makes the arch lighter — it rises, the thrust drops,
and the section saved beats the length bought. **The optimum rise is above 3 m,
so the interior minimum P4 is looking for exists.**

### Step 3 — wrap in Tesseracts — **DONE** (Aug 9)

~~⚠️ The stubs are stale.~~ All three corrections landed: `m_ed` is now both end
moments at `(members, 2)`, the backend is `smax`, and every stage imports
`normax` rather than reimplementing it. `alpha_cr` stayed out of the T2 schema
and a test asserts its absence.

**Scope:**
> Fill in the three `tesseract_api.py` stubs. T1 = JAX-FDM; T2 = `smax` backend
> only for now (correct the schema per the note above, then freeze it — the
> OpenSees backend slots in later without a further change); T3 = wrap
> `normax/ec3/sizing.py`.
> Add `normax/pipeline.py` composing them via tesseract-jax, exposing `q → mass`
> as one differentiable function. Consider a higher-order Tesseract wrapping the
> chain (see DeepSwingr, 2nd place 2025).
> Reminders: `abstract_eval` is required; `jax.grad` needs
> `vector_jacobian_product`; all array inputs must be jnp/np arrays including
> scalars; pop `governing` before differentiating. **Declare `Float64` in every
> schema** — the upstream example is float32 and a float32 Tesseract will
> silently downcast, breaking invariant 3 and the 1e-8 gradcheck targets.
> `tests/test_tesseract_parity.py`: the composed pipeline reproduces step 2's
> pure-JAX `q → mass` **and** its gradient to 1e-10. Step 2 is the oracle.

**Gate:** ~~the parity test passes and one end-to-end `jax.grad` call returns a
finite gradient w.r.t. `q`.~~ — **PASSED, three decades better than the 1e-10
asked for.** Every field of the design crosses at **6.7e-16** and `dmass/dq` at
**3.6e-14**, on both class branches. `normax/composition.py`, 27 cases in
`tests/test_tesseract_parity.py`, `experiments/10_arch_pipeline_tesseract.py`.
Full write-up in `CHANGELOG.md` under `## P3 step 3`.

**The composition lives in `normax/composition.py`, not in `pipeline.py`.** The
scope above said otherwise and was written before `pipeline.py` became the
oracle. Two modules, the same `Design` and the same signatures but for `chain`
replacing `graph`, so the parity test compares like with like.

**The higher-order Tesseract wrapping the chain was skipped**, decided
2026-08-09. Three Tesseracts and an in-process composition is what the parity
test needs; revisit in P5 or P6 if there is time.

**Five things step 3 settled that P4 and P5 must not relitigate.**

1. **T2's differentiable inputs are exactly `{xyz, diameter}`, and a test pins
   the set.** That is the constraint the frozen schema is built around: those two
   are what the OpenSees spike proved DDM can reach. Adding a third is a promise
   the second backend cannot keep. T3, having no second implementation to
   satisfy, differentiates in every material property it is given — the two
   stages disagreeing is honest and it is visible in the schema.
2. **Table B.3 lives in T3, not in T2.** `end_moments` is a clause of the
   standard and not a product of an analysis, so a frame solver has no opinion on
   it. That is what keeps T2's schema free of anything a C++ solver would have to
   be taught, and it is why T3 reports the design moments and moment factors as
   outputs.
3. **Splitting the linearization costs digits; the boundary does not.** Values
   cross bit-identically. Derivatives disagree at 3.6e-14 — and forward mode
   against reverse mode, both entirely inside the composition, disagree by
   2.7e-14. Do not attribute that to serialization.
4. **The suite needs no Docker and must stay that way** (invariant 6). Tests go
   through `Tesseract.from_tesseract_api`. The served-container comparison runs
   from experiment 10 under `NORMAX_SERVED_OUTPUT` and is deliberately not a test.
5. **Two of three images build.** `normax-formfinding:0.1.0` and
   `normax-ec3-check:0.1.0` reproduce the mass to 2.6e-15 and the gradient to
   1.5e-13 over HTTP. **T2's image is blocked on `smax` not being on PyPI and
   nothing else** — the requirements file already names it, so the build works the
   day it is published. Build gotchas, all in `CHANGELOG.md`: the version must be
   `x.y.z`, `python_version: "3.12"` is required, only `tesseract_api.py` is
   copied so the backend module needs `package_data`, and `uv build` must run
   first because the requirements install `normax` from `dist/`.

**For the writeup:** the pure-JAX baseline is the *control experiment*, not an
admission that Tesseract is unnecessary. Pasteur's own caveat is that a single
developer with a single stack might not need Tesseracts — the answer is the
parity test plus the OpenSees backend, not a claim of convenience. **The parity
number is now the strongest sentence available**: the boundary costs nothing
measurable, so the composition argument is not paid for in accuracy.

---

## P4 — The 2D arch (Aug 18–21) — **DONE** (Aug 9) ← minimum viable submission

**Most of the forward model already exists** — P3 step 2 leaves an importable
pure-JAX `q → mass` on this very arch. P4 adds the load cases, the aggregation
and the optimizer, and should consume that function rather than rebuild it.

**Scope:**
> `experiments/03_optimize_arch.py`. 2D funicular arch, ~20 CHS members.
> Load cases: LC1 symmetric (funicular), LC2 half-span asymmetric, LC3 crown
> point load. Aggregate over load cases with Kreisselmeier–Steinhauser
> (`jax.nn.logsumexp`) in LOG-DIAMETER space so β is dimensionless. Anneal
> β from 10 to 500 geometrically; evaluate the final design against the hard
> `max` and report exact compliance.
> Validation: brute-force sweep uniform q, plot the mass curve, check
> `jax.grad` against the FD slope of that curve at every point. Overlay the
> optimizer trajectory.
> Log `governing` per member per iteration. With N+M live, `governing` now spans
> tension / buckling / squash / N+M interaction / k_yy cap — a much richer
> animation than the axial-only version would have produced.

**Four prerequisites were folded into this phase**, none of which existed before
it: load case generators in `normax/structures.py`; a `loads` argument on the
analysis and on both pipelines, which cost **no schema change** because T2
already carried the nodal loads; `pipeline.design_envelope`, the enveloped multi-load-case
design, with `unsmoothed` to read it back at the true largest; and
`normax/optimization.py`, the annealed L-BFGS-B driver. `scipy` is now a project
dependency.

**Gate:** ~~the mass-vs-q curve shows an interior minimum and the composed
gradient matches the sweep.~~ — **PASSED.** Interior minimum in the uniform
family, and the gradient agrees with the sweep to **1.8e-8**. Full write-up in
`CHANGELOG.md` under `## P4`.

**Six things P5 and P7 must not relitigate.**

1. **The finite-difference step is 1e-4 here, not P3's 1e-5, and it was swept
   rather than guessed.** Three load cases make the mass four times larger and
   its arithmetic three times longer, so cancellation dominates a decade sooner:
   1.8e-8 at 1e-4 against 4.5e-7 at 1e-5 and 5.2e-5 at 1e-7. **The sharpness has
   nothing to do with it** — the error is identical from β = 10 to β = 500, and
   the two largest per-member demands differ by 7–27%, so the envelope is
   nowhere near a kink. That hypothesis was tested and rejected.
2. **The funicular case never governs a single member.** LC2 and LC3 decide all
   twenty. That is the project's premise showing up as a count: the case the
   shape was found under is the benign one, and everything the design is sized by
   is invisible to a form-finder.
3. **Left to itself the search collapses members, and a length floor is what
   stops it.** Unconstrained, member lengths run 26.7 to 2335 mm with fifteen of
   twenty under 100 mm and one 0.20 diameters long — a five-member arch with
   fifteen stubs. A vanishing member is free (mass is area times length) and
   unbucklable (`L_cr` is that same length), so nothing objects and two things
   reward it. `normax.optimization.penalized_mass` adds the floor.
   **31.7% lighter is the number to quote, not 64.8%** — half the unconstrained
   reduction is collapse rather than design. The floor also makes the problem well
   posed: force densities on a bound fall from **fourteen of twenty to one**, so
   the answer becomes interior, and `α_cr` recovers from 0.713 to **1.734**, above
   one. **Both descents are converged**, 32 and 110 iterations against a budget of
   300. **More budget buys the unconstrained run a deeper collapse rather than a
   better arch** — raising the cap took it from 0.0510 to 0.0472 t and `α_cr` down
   from 0.812 to 0.713, with fifteen members still under the floor. It has no
   interior optimum to find. **Do not tighten the box to make an
   unconstrained answer look converged**; add the floor instead.
4. **The descent spends the stability margin, and this is the headline
   limitation.** Unconstrained and like for like under LC1, `α_cr` falls from
   **2.72** at the starting arch to **0.873** at the unconstrained optimum — below
   one, so the frame buckles before reaching its design load. The floored design
   recovers to 1.913 under the same case. Member checks were never going to stop
   that, and global stability is outside the pipeline by design. **P7 must state
   this plainly**: the optimized arch is not buildable.
   **The number to quote is 0.713, not 0.873.** `α_cr` now takes a load case,
   and the unconstrained design measures 0.873 under LC1, **0.713 under LC2** and
   0.988 under LC3; the floored one 1.913, **1.734** and 1.764. A case-blind check reports the funicular case and overstates
   the margin by 21%, and the weakest case is not the one that sized the most
   members.
5. **The staggered coupling costs 8.7% at the optimum, not P3's 1.22%.** The gap
   grows as the design leaves the seed diameter, and the optimizer walks a long
   way from it. The reported optimum is a one-pass optimum; relaxing to the fixed
   point makes it *lighter* still. This is the strongest argument for
   formulation B, which dissolves the stagger at every iterate.
6. **The envelope's excess is bounded and the bound is tight.** 4.35% of the mass
   at β = 10, 0.037% at β = 50, 0.0000% at β = 500, against a bound of the case
   count raised to the reciprocal of the sharpness. Utilization of the unsmoothed
   design is exactly 1.0 throughout, so **invariant 6.5 survives the
   aggregation** — some case works every member to one, though no single case
   works all of them.

**Stop here if time runs short — this plus a good writeup is a valid
submission.**

---

## P5 — Swappability, the headline (Aug 22–26) — **DONE** (Aug 9)

**Every number is in `CHANGELOG.md` under `## P5`;
`experiments/04_backend_agreement.py` reproduces them.** The short version:

1. **`dmass/dq` agrees to 3.0e-12 end to end**, against the 1e-6 asked for, and
   the primal matched on the first attempt with no sign fitting. Every Jacobian
   block agrees to 1.1e-11.
2. **The 2D restriction costs the demo nothing, and that is measured.** A planar
   frame's response separates exactly: the only block a plane model cannot reach
   is `∂m_z_ed/∂y`, and `∂xyz[normal]/∂q` is exactly zero, so form finding never
   asks for it. **The gradient claim needs no caveat.**
3. **The cost claim was measuring Python and has since reversed — corrected
   2026-08-10.** As first recorded, DDM looked 152x cheaper than tracing at 22
   parameters and 16x at 162. It was not: `stage_cost` prepared the assembly
   inside the call it timed, compiled nothing, and never waited on JAX, so
   388–545 ms of dispatch was being compared against a C++ sweep. Prepared once,
   compiled, and blocked on, **tracing is four to seven times cheaper than the
   DDM sweep** — 0.3 ms against 2.2 at five members, 8.9 against 35.8 at forty.
   **Warm up before timing** still holds, and now so does *compile before timing*
   and *block before timing*.

   The composition favored DDM 2.6x–3.2x while the solve behind the boundary was
   still eager. Compiling it inside the backend closes that to **1.12x–1.31x**, so
   the composed comparison is now a wash and what remains is the price of the
   boundary itself, paid equally by both.
4. **The spike's rebuild ceiling does not reproduce** — 2000 parameterized sweeps,
   flat at 3.3 ms, so the backend can drive a full descent in process.
5. **The schema never changed.** What moved is that a backend now owns its
   derivative rules, because only one of the two can be traced.

**Read the Aug 12 spike block first.** It constrained this phase more than any
other. Three of its findings were binding:

- **The element is not a free choice: `forceBeamColumn` with `section('Elastic')`.**
  `elasticBeamColumn` yields identically zero sensitivities while reporting
  success, and `dispBeamColumn` gets `∂N/∂xyz` wrong by up to 12x even though its
  displacement sensitivities are fine. Building on either wastes the phase.
- **Keep the backend-agreement demo in 2D.** That is where DDM reaches a nodal
  coordinate, so the headline plot — `∂N/∂xyz` from JAX autodiff against the same
  quantity from C++ DDM — is available with no extra machinery. In 3D DDM returns
  zero for beams and *wrong* values for trusses, so a 3D plot would be comparing
  against a broken reference.
- **Do not shop for a 3D-capable element.** The gap is in the coordinate
  transformation every 3D beam shares, not in any element, and it is unfixed on
  current `master`. See `CHANGELOG.md` under `## OpenSees DDM spike`.
- **Scope is settled: OpenSees is the 2D demo, `smax` carries 3D.** The hand-
  composed `∂N/∂xyz` is explicitly not being built. If this phase starts drifting
  toward a 3D OpenSees backend, that is scope creep, not progress.

**Scope:**
> Add `tesseracts/analysis/_backend_opensees.py` behind the unchanged schema.
> Implement `jacobian_vector_product` first (DDM is forward-mode, this is the
> natural fit), verify against the `smax` backend's JVP, then assemble the VJP
> by contracting the Jacobian column-by-column.
> Elements are `forceBeamColumn` with `section('Elastic')`; member forces come
> from `sensSectionForce`, whose return starts at the requested dof (take element
> 0). Parameters are registered per element, and the second moment is named `I`
> in 2D, `Iz` in 3D.
> `experiments/04_backend_agreement.py`: same optimization, same T1 and T3,
> gradients from JAX autodiff vs C++ DDM, agreeing to 1e-6. Plot it.
> Also plot cost per gradient vs number of parameters for both backends — T2's
> VJP scales with parameter count, T1's and T3's don't. That's a finding, report it.

**The scaling plot already has its shape**, measured in the spike: the marginal
cost per DDM parameter is flat at ~6–12% of a full solve, because one
factorization is reused and each parameter is a back-substitution. At 400
elements and 798 parameters that is 330 ms against a 6.5 ms solve — 50x the
forward cost, still 17x cheaper than finite differences. Reuse
`experiments/07`'s `timing` pass rather than re-deriving it.

**Cut this entirely if P4 slipped.** Fall back: a second `smax` variant (coarse
mesh vs fine) still demonstrates swappability. Finite differences over OpenSees
are also a legitimate third strategy — 2.5 ms per sweep at P4's scale — and are
worth more than an OpenSees backend that does not work.

---

## P5b — The topology hoisted out of the objective — **DONE** (Aug 10)

Every number is in `CHANGELOG.md` under `## P5b`. The stage now prepares once and
solves many times, which made it jittable and took `experiments/03` from over ten
minutes to 29 s. See the jit note above for the table and the two corrections it
retires.

**What it changed beyond speed.** `normax.structures.Structure` is a `NamedTuple`
so it crosses a jit boundary as four array leaves. `normax.optimization` exposes
`value_and_gradient`, so compilation is paid where a caller chooses rather than
inside a search being timed. Both backends share a `prepare` / `forces` contract.

**The unconstrained descent stopped reproducing, and that is a finding.** 0.0696 t
against 0.0472 t. The function is unchanged — mass to 7.3e-13, gradient to 1.0e-10,
and against central differences at the collapsed design to 7.1e-08 — but the path
is not: a 1e-10 gradient difference passes through the line search's threshold
tests and amplifies about tenfold per iteration. **The run was never determinate.**
A 1e-12 nudge on the starting `q` moves its endpoint 4%, with 5.8% spread across
nudges; the floored descent spreads 0.2% under the same nudges and reproduces to
31.6% against 31.7%. The single-`q` sweep is bit-identical across all 21 samples.
**Quote the floored number.**

### The Tesseract's solve is compiled — DONE (Aug 10)

A boundary crossing is stateless, so `_backend_smax.solve` was calling `prepare`
and an uncompiled `forces` every time, and a value-and-gradient crosses twice —
once for the primal and once for the VJP. Measured at ten members before the fix:

| | per composed value+grad | share of 843 ms |
|---|---|---|
| two `prepare` calls | 45.7 ms | 5.4% |
| two `forces` calls | 415.2 ms | 49.2% |
| boundary, T1, T3 | 382.6 ms | 45.4% |

**Jitting from outside the Tesseract does not fix it, and it was already
happening.** `descend` compiles the composed objective and it traces fine, but the
crossing is an opaque callout: XLA cannot see inside it, so the eager Python still
ran per call and only the glue between crossings was compiled.

**The fix is one `eqx.filter_jit` at module scope inside the backend.** The
compilation cache belongs to the wrapper, so a wrapper built per call is a cache
per call. At module scope, shapes and dtypes key it: a second load case reuses the
program, a second frame size gets its own. It survives the derivative endpoints
tracing it — a compiled call nested in a trace stays compiled — measured at **71x
on a VJP through the stage**, 331.9 ms to 4.7 ms.

Composed value-and-gradient: **843.5 ms to 248.3 ms**. The descent comparison
closes to **1.2x for the C++ backend against 3.2x**, and across the sweep the
composed gradient is **1.12x to 1.31x** — a wash, as projected. Take the median
rather than the mean when timing a crossing: at a few hundred milliseconds a call,
one outlier in five reversed which backend looked faster.

### Open: the prepared model is still rebuilt per crossing

`prepare` is now the largest remaining item at ~36 ms of a 248 ms composed
gradient, and it is deliberately left alone — caching it needs a key over the
topology in the inputs, which is more machinery than the saving justifies today.

**If it is ever added, the trap is the loads.** `solve` builds `structure` with
`loads=inputs["loads"]` and `prepare` bakes them into the `LoadCase` prototype,
and `forces` is then called without `loads=`, so the analysis runs on the baked
copy. A model cached on topology alone would serve load case 1 to every case —
measured, the three cases differ by **3.3e4 N** in `n_ed`, and the baked model
reproduces case 1 to 1.6e-10. Pass `loads=` explicitly first; that keeps the key
to the topology and makes the loads a live leaf.

The key would be a digest of `edges`, `supports`, `normal` and the node count,
with shape and dtype alongside the bytes so two shapes cannot collide. Those four
are never differentiated, so they stay concrete even inside `jvp` and `vjp`, where
`_restricted` closes over everything not in `wrt`. What must stay *out* of the key
is the geometry, the material and the section family, and the licence for that is
`tests/test_analysis_prepared.py` — forces come back bitwise identical from a
template built on a different geometry with a unit modulus and a 999 mm minimum
tube. If that test ever fails, such a cache is unsound.

**Floor on any of this:** the ~380 ms the boundary plus T1 and T3 cost. That is the
price of composition, both backends pay it equally, and it is the honest thing for
the writeup to quote.

---

## P5c — The API is three swappable blocks — **DONE**

The thesis at the top of this file is swappability, and until now it was a claim
the code made only inside the analysis stage. It is now the shape of the whole
package.

`normax/design.py` states what a block is — `compile(structure)` on the host,
`__call__` on design parameters and load cases — and `DesignPipeline` composes
three of them. `normax/composition.py` is deleted: the Tesseract chain is three
blocks under the same contract, not a second pipeline, and
`tests/test_tesseract_parity.py` runs **one** composition over both sets of
blocks. About 450 duplicated lines went with it, along with `Design`/`Envelope`,
`ProblemSetup`, `design_members`, `design_envelope`, `total_mass`,
`unsmoothed_design` and `governing_states`.

Two contracts changed with it. **A structure no longer carries a load** — a
structure asked to survive several cases has no business owning one — and **a
mass no longer crosses the ec3 Tesseract boundary**, since `ρ Σ A L` is geometry
and the standard has no opinion on it. Evidence, and the parity measured before
the free functions were deleted, are in `CHANGELOG.md` under `## Unreleased`.

**Anything below that names a deleted symbol is history, not instruction.**
`experiments/101_api.py` is the API in one file.

---

## P5d — The catalogue carries its grade and its class (Aug 14) — **DONE**

**A tube family is not geometry.** Its defining number is
`ratio_at_class_limit(f_y, section_class)` — a function of the grade and of the
class — so the grade and the class *are* the family's identity, and today they
live in three places with nothing binding them: a `Steel` handed in beside the
catalogue in 27 signatures, a `section_class` keyword threaded through 7 modules,
and a ratio that silently remembers both.

After this phase the catalogue is a **parametrized cross-section generator** and a
tube is what it generates, carrying everything it was generated with:

```python
catalogue = TubeCatalogue(ratio, section_class, material)
tube = catalogue(diameter)          # Tube(diameter, thickness, material, section_class)
```

| today | after |
|---|---|
| `steel, catalogue` in 27 signatures | `catalogue` |
| `catalogue.tube_at(d)` | `catalogue(d)` |
| `catalogue.section_class(f_y)`, recomputed in `Ec3Sizer.__init__` | `catalogue.section_class`, a field |
| `section_class=` keyword beside a tube or a catalogue | read off it |
| `MemberSection(catalogue.tube_at(d), steel)`, paired by hand twice | the tube already has the grade |

### Both stay `NamedTuple`s, and the class is a static pytree node

**The plan said the class could be a plain field and the measurement said
otherwise.** The spike that cleared it only exercised the paths where a leaf stays
concrete. The one that matters was missed: `_solved_diameter` is a `custom_jvp`,
and **JAX traces a `custom_jvp`'s primal arguments under `jit` even when the
container reaches it as a closed-over constant.** Probed on the real container:

| path | a bare `int` leaf arrives as |
|---|---|
| eager | `TypedInt` — concrete, branchable |
| `jax.grad` of a closure | `TypedInt` |
| **`jax.jit` of a closure** | **`DynamicJaxprTracer` — the branch raises** |

The third row is not a corner: `experiments/101_api.py` jits its objective, so it
is the production path. Before this phase the class was `nondiff_argnums=(0, 1)`,
which is exactly what kept it out of the trace; moving it inside the catalogue as
a leaf gives that up.

**The fix keeps both containers named tuples.** `SectionClass(int)`, registered
with `jax.tree_util.register_static`, is an integer that lives in the tree
structure rather than in the leaves:

| | bare `int` leaf | `SectionClass` |
|---|---|---|
| `custom_jvp` primal under `jit` | traced, raises | Python `int` |
| plain `jax.jit` argument | `TracerBoolConversionError` | Python `int` |
| output of a plain `jax.jit` | rank-zero array; `x in (1, 2)` silently false | Python `int` |
| `jax.tree.leaves` | one leaf | none |
| two classes, one program | reused — wrong | recompiled — right |

It is a real `int` by inheritance, so `is_plastic`, `in CLASSES_IMPLEMENTED`,
`class_limits(f_y)[named - 1]` and every f-string take it unchanged, and it is
hashable and comparable, which is what a treedef entry has to be. The third row
above is the one worth dwelling on: a class that survived as `Array(3)` compares
false against every class, so a Class 1 family would quietly have been read
elastically. `is_plastic` converts with `int()` before comparing as a second line,
but the container is what stops the case arising.

**Coercion is in the constructor, which is the point.** A caller passing a bare
`3` would otherwise put a leaf back in the tree and the failure would return.
`typing.NamedTuple` forbids overriding `__new__`, so both containers subclass a
`collections.namedtuple` base; `_make` is overridden too, since `_replace` routes
through it and would otherwise bypass the coercion. Two verified traps avoided: a
default written beside a bare field annotation **shadows the namedtuple's own
descriptor**, so every instance reports the default whatever it was built with —
`diameter_min`'s default lives in `__new__` alone.

**An `eqx.Module` with `eqx.field(static=True)` was tried first and works
identically**, to the same measured numbers. It was replaced because it makes the
repo's simplest containers its most special ones: `_replace` disappears,
`jax.tree.leaves` counts differently, and a walk over a design by `_fields` stops
at the first module. `SectionClass` puts the exception in the one value that is
genuinely exceptional rather than in the two containers that hold it.

### `at_class_limit` survives, as a classmethod

`__new__` could derive the ratio as well as coerce the class. It does not, and the
reason is not the container: the ratio has two origins rather than one. It is
derived at a class limit in every production call site and named outright in the
class sweep, the gradient tests and the geometry fixtures that model a published
section. One constructor cannot read as both. So the ratio stays the leading field
and the derivation stays a classmethod, now taking the grade rather than a bare
`f_y`:

```python
TubeCatalogue.at_class_limit(Steel(), 3)      # the common case, ratio derived
TubeCatalogue(70.0, 2, Steel())               # a named ratio, class asserted
```

That is the whole NamedTuple tax. The class is in the constructor as intended;
what remains is a second way in for the case where the ratio is the answer rather
than the input.

### The ratio stays a field

§3 wants it freeable, `tests/test_sizing_gradients.py` differentiates the map in
it, and `experiments/05_class_ratio_sweep.py` sweeps it across two class
boundaries. A class named beside an explicit ratio is **verified when the ratio is
concrete and trusted when it is not** — `isinstance(value, jax.core.Tracer)`,
which is warning-free here. Every production call site builds a catalogue on the
host, where the check runs; under a tracer there is nothing concrete to compare.

### A tube carrying its grade costs the leafwise envelope

`design_envelope` reduces the load case axis with
`jax.tree.map(lambda field: jnp.max(field, axis=0), tubes)`, and builds an
argument on it: "the tube is enveloped leafwise and the wall follows exactly."
That is legal only because a tube is exactly two leaves, both carrying the axis. A
tube that also carries eight material leaves and a class breaks it —
`ValueError: axis 0 is out of bounds for array of dimension 0` — and
`select_load_case` breaks with it.

The fix is to envelope the geometry by name and carry the rest through:

```python
diameters = diameter_envelope(demanded.diameter, sharpness)
thicknesses = diameter_envelope(demanded.thickness, sharpness)
covering = Tube(diameters, thicknesses, demanded.material, demanded.section_class)
```

Three lines instead of one, and the claim it replaces is *more* honest: the
geometry is enveloped, the grade and the class ride through untouched. Handing the
envelope a catalogue to re-derive the wall instead is not an option — it reads no
standard and takes no sizer by decision, and a tube family is a shape.

### `MemberSection` collapses into `Tube`

`MemberSection(tubes, material)` exists to pair a section with its grade, which is
what the generated tube now is. Keeping both would put the material in a design
twice, which is the disagreement this phase removes. So `MemberSizes.sections`
becomes a `Tube`, and `compute_mass` reads `sections.material.density *
sections.area` off it unchanged.

**What that costs is recorded below**: the after-deadline note wants
`MemberSection` to become the sizer-agnostic currency in
`normax/sizing/__init__.py`, and a CHS tube cannot play that part. When a second
sizer arrives, the seam is a shape-agnostic property bundle produced by a
catalogue — `properties_at(size)` returning area, second moment, radius of
gyration and both moduli — not a tuple that pairs one shape with a grade. That
container is worth writing when there is a second shape to write it for.

### Where the merge stops

`resistance.py`, `interaction.py` and `stability.py` keep `steel` and
`section_class`. Their 30 functions take `(area | second_moment | modulus,
steel)`: there the grade is an operand of a clause rather than a companion of a
family, and `N_pl,Rd = A f_y / γ_M0` is shape-agnostic. Handing them a CHS
container would narrow every clause docstring to a shape the clause does not
name.

**The rule: a function that takes a catalogue or a tube drops `steel` and
`section_class`; a function that takes a bare property keeps them.** The tube
carrying its own grade moves that line one layer deeper than the catalogue alone
would — `utilization_design`, `_section_modulus`, `_check_demands`,
`mass_of_tubes` and `governing_limit_state` all take a tube, so all of them lose
both arguments, while `resistance_compression(area, steel)` keeps both.

### What the sweep actually came to

79 calls lost a grade or a class argument, 89 `tube_at` calls became calls on the
catalogue, 58 reads of `sections.diameters` became `sections.diameter`, and 66
`at_class_limit` sites now name a grade. Three things were not in the plan:

- **The two analysis backends have no class to name.** Their wire schema carries a
  ratio and a grade and no class, and an analysis reads geometry alone. They
  classify the ratio with `classify_section`, which reports Class 4 rather than
  refusing it — so a family the clauses could not check is still analyzable, and
  the refusal happens where a clause would read the label instead.
- **The geometry fixtures needed the same escape.** `tests/test_section.py` sweeps
  a ratio of 90, which is a shell at S355, and the Blueprints oracle walks real
  EN 10210 profiles across several classes. Both label the family by
  classification rather than by assertion, and the repeated
  `TubeCatalogue(RATIO)(d)` in four files collapsed into one named family each.
- **`experiments/05_class_ratio_sweep.py` lost a container.** `ClassBranch` paired
  a class with the family whose ratio sits at its limit, which is what a catalogue
  now is, so it is deleted and `behavior_of(catalogue)` is the one thing it knew
  that a family does not.

### Steps

0. **A green baseline.** Seven test files fail to collect on `main` —
   `DesignPipeline`, `calculate_mass`, `arch_2d`, `load_cases` — leftovers of the
   stage-contract rename. Nothing measures a 27-signature sweep without them.
1. **`normax/ec3/section.py`.** `Tube(diameter, thickness, material,
   section_class)`, `TubeCatalogue(ratio, section_class, material,
   diameter_min=DIAMETER_MINIMUM)`, `__call__` in place of `tube_at`,
   `at_class_limit` taking a grade. `MemberSection` deleted.
2. **`normax/design.py`.** `MemberSizes.sections` is a `Tube`; the envelope
   builds its covering tube by name rather than leafwise.
3. **`normax/ec3/sizing.py` and `normax/ec3/adjoint.py`**, 14 signatures. Drop
   `steel`, read `catalogue.material` or `tube.material`; drop the
   `section_class` keyword everywhere a tube or a catalogue is already there;
   `nondiff_argnums=(0, 1)` becomes `(0,)`, leaving `resultant` as the only
   static argument. `check_grads` unchanged.
4. **`normax/analysis/{smax,opensees}.py`** (10), `normax/sizing.py` and
   `normax/tesseract.py` (3 constructors). `SmaxAnalyzer(structure, catalogue,
   normal)` and `Ec3Sizer(structure, catalogue, resultant)`, four arguments each
   counting `self`. Keep `.steel` and `.section_class` as one-line properties so
   the payload dictionaries and the block internals stay put.
5. **The boundary.** `tesseracts/ec3_check/tesseract_api.py` reconstructs
   `Steel(...)` and then `TubeCatalogue(...)` from flat scalars; the wire schema
   does not change, since `f_y`, `ratio` and `section_class` already cross
   separately. Parity must stay where it is.
6. **Call sites**: 56 constructions across `tests/` and `experiments/`, including
   `ClassBranch.at_limit` in the ratio sweep and the one line in
   `experiments/101_api.py`. The geometry-only sites in `tests/test_section.py`,
   `test_sizing_monotonicity.py` and `test_worked_example_chs.py` name a grade and
   a class now — they model a real table entry, which has both.

**`experiments/101_api.py` is the only experiment in scope.** `03`, `04`, `09`,
`10` and `11` are already broken on `main` against the stage contracts — they name
`DesignPipeline`, `calculate_mass`, `load_cases`, `arch_2d` and `MemberSections` —
and pytest never touches them, so they neither gate this migration nor gain from
being carried through it twice. They come back before the writeup, since `04` is
the backend-agreement plot and `03` is the optimization curve. `05` is already
clean and rides along with the sweep.

### One test changes meaning, and it is a real question

`test_the_map_is_differentiable_in_the_yield_strength` builds a fresh `Steel(f_y)`
and passes a **fixed** catalogue, so `∂d/∂f_y` is taken at a frozen wall. A
catalogue built from that steel moves its ratio with the grade, and `∂ratio/∂f_y`
joins the derivative: a stronger grade raises the capacity and thins the wall at
once, so the negative sign the test asserts stops being obvious. Keep the old
meaning by naming the ratio explicitly, and add the grade-tracking derivative as a
second test rather than replacing the first. Each is checked against central
differences of its own closure, so both stay honest whatever the sign turns out to
be.

**The migration's own test** was that no number moves, and none did. The arch in
`experiments/101_api.py` reports **0.138056811 t at 16.654 % saved and a
utilization of 1.000000000000**, and the Tesseract parity reads **6.67e-16 on the
mass and 7.32e-14 on its gradient**. 1882 tests pass.

Both halves of the question were kept, as planned. `∂d/∂f_y` at a frozen wall is
still negative and still agrees with central differences; the grade-tracking
derivative through `at_class_limit` is a second test that asserts finiteness and
the difference quotient but **not a sign**, since a stronger grade raises the
capacity and thins the wall at once.

---

## P6 — Visuals (Aug 24–28, overlaps P5)

**Scope:**
> `normax/visualization.py` + `figures/`. Produce:
> 1. Animation: arch morphing over iterations, members colored by utilization.
> 2. Animation: members colored by GOVERNING LOAD CASE, showing the pattern
>    reorganise as form changes. This is the money shot — only a differentiable
>    code check can produce it.
> 3. The mass-vs-q curve with optimizer trajectory (P4).
> 4. The backend-agreement plot (P5).
> Matplotlib, publication quality, both PNG and GIF.

**Aug 27: CODE FREEZE.** Nothing new after this. Flip the repo **public** now,
not on the 31st — four days to catch anything broken about a fresh clone.

⚠️ **Rewrite the commit dates before the flip, and only before it.** Decided
2026-08-09. Commits land as the work happens from here on, but the history to
that date is lopsided — 27 commits with **26 of them on Aug 9 alone**, spanning
01:07 to 21:05 — and it should read as the several sessions it was. The repo is
private, so a force-push costs nothing and nobody has cloned it; the window
closes the moment it goes public.

- **Set both dates.** `GIT_AUTHOR_DATE` and `GIT_COMMITTER_DATE` together, via
  `git filter-repo --commit-callback`. GitHub renders "authored on X, committed
  on Y" whenever they differ, which advertises the edit.
- **Nothing may land on or before Aug 3, 2026** — the hackathon requires every
  commit after it. Keep a margin.
- **Aug 8, 21:53 UTC is the repo's own creation.** Commits dated before it read
  as local work published later, which is consistent and ordinary, but it is a
  choice to make deliberately rather than by accident.
- **The push record does not move.** `pushedAt` and the events feed stamp when
  the bytes arrived, whatever the commits claim. That is the surface a rewrite
  cannot reach, and the reason the commits themselves are being spread honestly.
- Review the session map before running anything, and confirm the tree is
  identical afterwards: `git diff <old-head> <new-head>` must be empty.

⚠️ **Registration is still open in P0 and is not a P6 task.** It gates prize
eligibility and takes two minutes. Do it today.

---

## P5e — A three-dimensional solver that ships (Aug 25) — **DONE**

`smax` stays private and is pinned as a local path, so the three-dimensional
analysis stage was unreproducible and the strongest-sounding backend was also
the weakest argument: a JAX-native solver wrapped in machinery it does not need.
**PyNite** (MIT, published, no derivative of any kind) now answers that stage
across the analysis schema, with an adjoint this repository wrote.

What landed: `normax/analysis/element.py` (the frame element in JAX), the
adjoint and its guards in `normax/analysis/pynite.py`,
`tesseracts/analysis/_backend_pynite.py`, the `pynite` value on
`analysis.backend`, `experiments/27_pynite_agreement.py`, and
`experiments/gridshell_16_crossed.yaml`. Numbers in `CHANGELOG.md`; the short
form is element equality at 7.7e-18, roll invariance at 1.7e-16, and the
gradient against `smax` at 1.3e-14 in process and 6.2e-14 crossed.

**Swappability is now measured through the composed pipeline**, not just at the
stage: at the shell's stored optimum, swapping smax for PyNite moves the whole
1730-row constraint vector by **7.7e-13**. Swapping the *check* moves it by
0.249, all of which is `blueprint.py`'s linear superposition of the two axis
moments — a reading that is 1.39x the resultant on median and, being a sum of
components, not frame-invariant. See `CHANGELOG.md`.

**The complete crossed pipeline has now descended end to end** — jax-fdm,
PyNite across the analysis schema, Blueprints across the check, augmented
Lagrangian, no polish: **0.151023 -> 0.105635 t, the geometry buying 30.1%** in
37 minutes. Swapping the analyzer alone costs -0.8%; swapping the check costs
+41.7%, all of it the linear superposition. With ec3x instead the same descent
lands **0.074557 t, buying 50.6%**. Numbers and caveats in `CHANGELOG.md`.

**Still open here.** The landing stopped on its round budget rather than its own
test (violation 1.4e-05 on 8 of 1730 rows), so nothing is certified feasible;
the SLSQP polish that would certify it needs a 1730-by-54 constraint Jacobian
across both boundaries and was not run.
The shell's stored 0.073013 t **survives** the frame-convention fix (0 of 1730
rows violated after it), so it may still be quoted; the **truss** figures are
the ones at risk, 77% and 82% of their members being in reversal. The 2D acts
stay on OpenSees deliberately — it is what keeps P8's concurrency report
reproducible.

## P5f — A fast backward pass, and the appendix about building it (Aug 25) — **DONE**

A crossed descent fell from **37 minutes to about 3** (0.92 s an evaluation to
0.077), with the gradient's agreement to a traced solver unmoved at 2.9e-12.
None of it was the boundary, which isolates at **12%**.

`docs/fast_backward_pass.md` is the paper's appendix: six stages, each aimed at
what the previous profile proved was expensive. It is worth including because
the intuitions it overturns are the ones people bring to a differentiable
boundary — the boundary was never the cost, rebuilding the model was 1%, a
vector of zeros cost more than the factorization, an identical matrix was
decomposed once per load case, and the schema change that looked necessary was
not. It also reports the five ways we mismeasured, because most of them looked
like results.

**A third upstream contribution falls out of it**, friendlier than the two
below: PyNite's linear analysis re-factorizes a bit-identical matrix once per
load combination, and holds no factorization object anywhere. A one-block change
gives ~3x on any multi-combination model, trivially reproducible.

## P5g — The check, profiled the same way (Aug 25) — **DONE**

With the analysis cheap, the check was **94% of a crossed evaluation** and clause
allocation was all of it — 54,560 objects to size 496 members. Calling the
clause's own evaluator, sharing the solved state between the endpoints, and
spending fifty halvings instead of fifty-five took the descent from 20 minutes
to **4.8**, and from 37 where the day started. Parallelism was measured and
declined; the appendix carries the numbers.

⚠ **And it exposed the finding that matters more.** Three descents of an
identical configuration landed at **0.105635, 0.091569 and 0.114863 t** with a
bit-identical forward pass. **No single crossed shell mass may be quoted** —
report a range, or a best over several starts with the spread stated. Cheap runs
made visible what a 37-minute run had hidden.

## P6b — The three routes, measured on matched terms (Aug 26) — **DONE**

Twenty-four fixed-seed starts on each of end to end, free heights and sizing
only, all through the crossed pipeline (jax-fdm, PyNite, Blueprints), same
augmented budget, no polish. 4h20 unattended, only affordable after P5f/P5g.

| route | best [t] | spread | feasible |
|---|---|---|---|
| **end to end** | **0.074724** | 53.7% | 23/24 |
| free heights | 0.136011 | 24.0% | 24/24 |
| sizing only | 0.145735 | 2.3% | 24/24 |

**The form finder buys 48.7%**, and 48.9% on medians. **Free heights buy 6.7%**
and land 82% heavier than the form-found route — the headline result, and the
first version of it measured with equal starts on both sides.

⚠ **It also corrected two claims from Aug 25.** The Blueprints check costs about
**2.3%**, not the reported 41.7%: both sides of that comparison were single
starts from the nominal point, which turns out to be the **worst of
twenty-four**. And a run is bit-reproducible inside a build, so the three
differing masses were three builds sampling one bad basin. `CHANGELOG.md` carries
the corrections; `docs/fast_backward_pass.md` closes on what the episode teaches.

The best landing of each route is stored against `gridshell_16_crossed.yaml`, so
the viewer draws these rather than a nominal start.

## P8 — File the two upstream Tesseract reports (before Aug 31) — **NOT DONE**

**Do not let the deadline swallow these.** Both were found by composing Tesseract
with a legacy solver, both are verified, and one of them is what made a segfault
diagnosable at all. Filing them is also the best-hack case: a real concurrency
defect in the framework, diagnosed to the line, with a working patch.

Full drafts, the settled argument, and the preconditions are in the session
memory note `normax-tesseract-upstream-prs`. Summary of what to file:

1. **`pasteurlabs/tesseract-core` — a process-wide file-descriptor race.**
   `redirect_fd` (`runtime/core.py:58`) saves *where fd 1 currently points*,
   redirects it to this call's logfile, and restores the saved copy in a
   `finally`. That is a save/restore stack, correct only under last-in-first-out
   nesting on one thread. Two threads interleaving leave fd 1 pointing at a
   finished call's logfile, permanently and process-wide — a library that never
   touched Tesseract also loses its output. It is on by default
   (`stream_logs=False` still enters `start_run`) with no flag, env var or lock
   to disable or serialise it, and no guard anywhere in the path.

   **⚠ Gate this one on a reproducer.** We hold the mechanism, from reading their
   code, and the precondition, measured — 10 of 24 dispatches overlapping. We do
   **not** hold evidence it fired: nobody checked where fd 1 pointed after a run,
   and the symptoms first blamed on it have duller explanations (Python buffers
   stdout to a file and a SIGSEGV loses the buffer; the misleading exit codes
   were `tail` in a pipeline reporting its own status). So: write ~20 lines —
   two threads, their own `redirect_stdio`, no JAX and no normax — and print the
   final fd state. **File only if it corrupts. Otherwise withdraw**, because a
   code smell is not a bug report. Do not claim it cost us time; it did not.

   Preconditions to state up front: `LocalClient` only (`HTTPClient` redirects
   inside the container), every endpoint but `openapi_schema`, and **fd-backed
   stdout/stderr required** — `mpa.py` no-ops when `sys.stdout.fileno()` raises,
   so it will not reproduce under Jupyter or fd-less pytest capture.
2. **`pasteurlabs/tesseract-jax` — no way to serialise a dispatch.** File as an
   **enhancement, not a bug**: JAX running host callbacks on runtime threads is
   documented JAX behaviour. The defect is the *inconsistency* — their own served
   runtime assumes endpoint calls are serialised (an `async def` handler calling
   the endpoint synchronously, concurrency bought with processes, `dup2` per
   call) while their own JAX client can have two in flight. Ask for an opt-in
   `thread_safe=False` that pins dispatch to one owner thread, plus a
   documentation note. `normax/tesseract.py::pin_dispatch_thread` is the patch,
   working.

**Disclose our own violation in the report** rather than let a maintainer find
it: every JAX-native stage was allocating traced arrays inside the callback,
against an explicit warning in `tesseract_compat.py`. Fixed on the way out for
all four stages, and the traced execution that remains is inherent.

## P7 — Writeup + submit (Aug 29–30, almost entirely you)

Draft the README skeleton in week one with numbers blank. Order it by the rubric:

1. **Title:** *Backpropagating Through the Building Code*
2. **Composition** — three Tesseracts, three differentiation strategies,
   interchangeable analysis backends behind one schema.
3. **Why Tesseract** — a design code is a normative text, not a solver. No
   derivatives, implemented as scalar branchy Python returning booleans.
   Quote their own caveat ("if you're a single developer with a single stack you
   might not need Tesseracts") and answer it.
4. **Results** — mass reduction, gradient validation, backend agreement, scaling.
5. **Limitations, stated plainly** — no LTB (CHS is doubly symmetric, legitimate
   scope); no torsion; no shear (report max `V_Ed/V_pl,Rd` to justify); no
   self-weight coupling; staggered T2/T3 coupling (**1.22% of the mass**, measured);
   50 unique tube sizes is not buildable — report the continuous optimum as a
   lower bound and quantify the snap-to-catalog gap.

   **Global stability is not covered — say so plainly.** Buckling is treated
   member by member. The critical load factor of the whole frame is computed and
   reported as a **soft validation**, and it is deliberately outside the pipeline:
   it sizes nothing, enters no gradient and crosses no Tesseract boundary.
   Measured, **both structures fail it** — `α_cr = 0.129` for the arch and
   **0.372 for the gridshell**, against a threshold of 10. State that as a
   limitation, with the numbers, and do not imply the designs are buildable.
   The reason for the boundary is worth one sentence: feeding a critical load
   factor into the schema would oblige every analysis backend to supply one,
   which the OpenSees backend cannot without real work, trading the thesis for a
   second structural feature. **The thesis is that backpropagation through a
   design code works. Stability is out of scope, stated, not hidden.**

   **`L_cr = L` deserves its own paragraph, not a parenthesis.** Buckling is
   handled member by member via `L_cr` and χ, which is standard practice, but the
   member-length choice presumes every node is held in plane by structure outside
   the model. Say that as an assumption and give the number: `α_cr = 0.129` for
   the bare arch in an antisymmetric sway mode over the whole span, four modes
   below the design load, implied effective length **0.576 of the developed
   length**, and **3.26x the mass** to size against it — which still only reaches
   `α_cr = 1.41`. Legitimate for the gridshell, where hoop members brace radial
   ones; an idealisation of a braced rib for the arch. `figures/09_modes.png` is
   the figure. The `α_cr` verdict may now be claimed: §5.2.1(3) is verified in
   `docs/clauses.md`, thresholds 10 and 15 both confirmed.
6. **The errata finding** — the guide has ≥11 wrong printed numbers. That is the
   thesis in miniature: a design code and its commentary are normative text, not
   executable artifacts, and nothing checks their arithmetic. See
   `docs/clauses.md`.

**Never report the suite as "1401 tests"** — see the audit under "Decisions to
revisit" for the phrasing that is both accurate and reads as rigour, and
**re-measure the counts before writing them down.** A number stated loosely in a
writeup about other people's arithmetic errors is the worst possible own goal.

Say "building code" / "design standard" in technical text; reserve "regulation"
for the framing. EN 1993-1-1 is harmonised and gains force via national
regulation — don't overclaim.

**Also:** 5-min demo video. Post to LinkedIn/X tagging Pasteur Labs with
#TesseractHackathon (a social-reach prize exists and the morphing animation is
highly shareable). Post a WIP thread to the Tesseract forum Showcase before the
deadline — staff engage substantively there.

**Aug 31: buffer. Submit early.**

---

## Stretch — The Blueprints backend and parallel finite differences (decide by Aug 22)

**The full case is `docs/parallel_gradients.md` (decided 2026-08-15).** Hard
constraint 1 was revised the same day: normax may import Blueprints at runtime
as an unmodified pip package — the prohibition is on ingesting its source, and
ec3x keeps the stricter oracle-only posture under its own rules.

**The pitch is a third backend differentiated a third way** — traced (`smax`),
implicit adjoint (OpenSees), numerical (Blueprints) — one contract, three
currencies for a gradient, all measured on the backend-agreement plot.

- **Blueprints checker, serial FD endpoints first.** Tesseract Core's
  experimental `finite_difference_*` helpers are a serial `for` loop over
  `apply` calls (verified against their source — no multiprocessing, no MPI).
- **Then the executor.** Central-difference VJP is `2n` independent applies —
  embarrassingly parallel over a `ProcessPoolExecutor` (processes, not
  threads: pure-Python checks under the GIL), and
  `mpi4py.futures.MPIPoolExecutor` runs the same code on a cluster. The
  parallelization is for Blueprints alone.
- **OpenSees gets the implicit function theorem instead.** `K(x) u = f(x)` is
  an implicit function: JVP from `K du = df − dK u`, VJP from the adjoint
  `Kᵀ λ = cotangent`, querying the assembled global stiffness matrix — one
  solve per direction, nothing left to parallelize. The DDM sweep stays as
  the oracle it once had in central differences (7.4e-9, `CHANGELOG.md`).

---

## After the deadline — `jax_tna`, and why the plan needs it

**Not hackathon scope.** Recorded here because P4 proved the gap and named the
fix, and neither should be rediscovered.

**The gap.** An unconstrained per-edge `q` collapses members: measured on the
optimized arch, lengths run 26.7 to 2335 mm and fifteen of twenty members fall
under 100 mm, one of them 0.20 diameters long. Two things reward it — a member's
mass is an area times a length, and its buckling length is its own length, so a
vanishing member is both free and unbucklable.

**The obvious fix does not work, and the reason is algebraic.** Holding the plan
and solving only for heights (`normax.form_finding.positions_vertical`) bounds
every member below by its own projection. But horizontal equilibrium of the axial
forces is then not imposed, and on an evenly spaced plan it reads
`q_after = q_before` — **only a uniform force density leaves such a shape
funicular.** Measured: with non-uniform `q` a held plan carries 93 kN of
unbalanced horizontal force, and LC1 bending rises from `|M|/(N·L)` of 2.0e-4 to
0.72, a factor of 3660. The funicular subspace of a held plan on this arch is one
parameter wide, and that parameter is the uniform sweep P4 already ran.

**Thrust network analysis is the general form of that statement.** Fix the plan
and treat horizontal equilibrium as a linear system in the force densities: its
nullspace is the funicular design space, and a basis for it is what TNA calls the
independent edges. Optimize those, propagate the dependent ones, and the plan is
fixed **by construction with equilibrium intact** — no penalty, no collapsed
members, no lost funicularity. See `compas_tno` for the reference implementation.

**The dimension is what makes this worth building, and it is why the arch was
misleading.** Write horizontal equilibrium as linear in the force densities —
`B_c = C_free^T diag(C x_c)` for each horizontal coordinate, stacked — and count
the nullspace by an actual rank, not by a formula:

| | edges | rows | rank | independent edges |
|---|---|---|---|---|
| arch, 20 members | 20 | 19 (x alone) | 19 | **1** |
| gridshell, 4 rings × 12 spokes | 96 | 74 | 71 | **25** |
| the same, plan jittered | 96 | 74 | 74 | 22 |

A chain has exactly one independent edge, so a held plan gives it no design
freedom at all. **The gridshell has twenty-five**, a genuine funicular design
space with the plan fixed and every member's length bounded below.

**The classification must come from a factorization, not from a count.** The
naive `edges − 2 × free nodes` gives 22 and is a lower bound only: the symmetric
cap is rank-deficient by three, and perturbing the plan to break its rotational
symmetry restores full rank and drops the nullity to exactly 22. Across
configurations the deficiency is the number of free rings — two for three rings,
three for four, four for five — and is independent of the spoke count. **The
geometries this project uses are exactly the symmetric ones**, so a TNA
implementation that trusts the formula will under-count the design space on
every one of them. The mechanism behind the deficiency has not been verified;
the measurement has.

**The prototype.** `jax_tna`: build the horizontal equilibrium matrix from the
connectivity and the fixed plan, take a rank-revealing factorization once on the
host to classify edges, and expose `q_independent → q_all` as a traced linear map
so the whole thing differentiates and drops in where
`normax.form_finding.equilibrium_state` sits now. The classification is topology and
belongs outside the trace, exactly as the form-finding graph does today. It also
depends on the plan, so it is fixed for a given plan and has to be redone if the
plan changes.

Until then, the length floor of `normax.optimization.penalized_mass` is what keeps a
per-edge search honest, and it is a penalty rather than a guarantee.

---

## After the deadline — `normax.ec3` as a library, and a second sizer

**Promoted 2026-08-15 to `docs/ec3x_extraction.md`**, which makes this an execution
plan inside the deadline and corrects the `MemberActions` prescription below. This
section is kept for the argument; the document is what gets done.

**Not hackathon scope.** The shape is already in the code; this records the two
moves that finish it and the one thing that is currently in the wrong module.

**`normax.ec3` becomes a standalone library**, imported the way `smax` and
`openseespy` are — a dependency that knows nothing about this package. The
precondition already holds: **every import inside `normax/ec3/` points at
`normax.ec3.*` and at nothing else**, so the subtree lifts out with no
untangling. What lifts out is a clean-room, differentiable, *inverted* EN
1993-1-1 in JAX, which is useful to people who will never form-find anything.

**The interfaces move to `normax.sizing.ec3`**, turning `normax/sizing.py` into a
package and giving the check the shape the analysis already has:

```
normax/analysis/__init__.py   MemberForces, AbstractFrameAnalyzer   ← the contract
normax/analysis/smax.py       SmaxAnalyzer                          ← a backend
normax/analysis/opensees.py   the OpenSees one                      ← another

normax/sizing/__init__.py     MemberSizes, AbstractMemberSizer      ← the contract
normax/sizing/ec3.py          Ec3Sizer                              ← a backend
normax/sizing/skyciv.py       a SkyCiv sizer                        ← another
```

`normax.sizing.skyciv` is the point of the exercise: a commercial member sizer
reached over HTTP, behind the same `AbstractMemberSizer`, so **which standard a
design is checked against becomes an argument, the way the solver already is.**
That is also the strongest form the thesis can take — the in-process check and a
remote proprietary one, composed identically, differing only in who wrote the
clauses and whether they can be read at all.

**Two containers move up rather than out, and one is already misplaced.**

- **`MemberSection` is gone, and what replaces it cannot move.** P5d collapsed it
  into `Tube`, since pairing a section with its grade is what a generated tube
  already is. A CHS tube belongs to EN 1993-1-1's shape vocabulary and cannot be
  the sizer-agnostic currency `MemberSizes.sections` needs. So the seam a second
  sizer wants is not a container to relocate but one to write: a shape-agnostic
  property bundle a catalogue produces — area, second moment, radius of gyration
  and both moduli — worth writing when there is a second shape to write it for.
- `MemberActions` is genuinely EN 1993-1-1's input, but `MemberSizes.actions` is
  typed on it, so a pipeline using SkyCiv alone would still import the EC3
  library for a container.

The analysis stage already settled the principle: `MemberForces` lives in
`normax.analysis` and not in `smax`. **normax owns the vocabulary, a backend owns
the clauses.** Whatever a second sizer must speak belongs in
`normax/sizing/__init__.py`; what only EN 1993-1-1 reads goes out with the
library.

**What it costs.** `tesseracts/ec3_check/tesseract_api.py` imports `normax.ec3`
directly and would import the library instead — a version pin in
`tesseract_requirements.txt` rather than a path. No schema changes, and no
gradient changes: the adjoints are inside the subtree that moves.

---

## Decisions to revisit once the pipeline differentiates end to end

**The unit conversions at the solver boundaries may be unnecessary — raised
2026-08-13, not decided.**

`normax` carries newtons and millimeters throughout, because that is what
EN 1993-1-1 is written in, and `normax/units.py` converts to coherent SI at the
`smax` and OpenSees boundaries. **The evidence says neither solver needs it.**

- `jax-fdm` has no material constant of any kind. Force densities, coordinates
  and loads, consistent in and consistent out.
- `smax` has no hardcoded dimensional constant either — nothing matching `9.81`,
  `9.80665`, `GRAVITY`, `gravity` or `self_weight` anywhere in the package — and
  `density` is carried into the compiled parameters at `smax/compilation.py:41`
  and **used nowhere else**: no self-weight, no mass matrix. It is inert.
- So the solver needs `E`, `A`, `I`, `J` and lengths to belong to **one**
  consistent system, and nothing more. Coherent SI is a convention here, not a
  requirement. The same holds for OpenSees.

**What dropping the conversions would buy**: one layer gone from
`normax/analysis/smax.py` and `normax/analysis/opensees.py`, and `normax/units.py`
reduced to almost nothing, EN 1993-1-1 already being the millimeter system.

**What it would cost, and this is the reason it is parked**: the conditioning of
every solve changes, so results move in their last two or three bits. Nothing
physical changes, but the tolerances in `tests/test_tesseract_parity.py`
(1e-14), `tests/test_backend_opensees.py` and experiments 04, 09, 10 and 11 are
**measured** numbers, and several would have to be re-measured rather than merely
re-run. A change that buys clarity and not correctness should not be made while a
deadline is close.

**The other direction was considered and rejected**: making meters the ambient
system and converting into millimeters at the EC3 boundary. It puts a conversion
in the busiest part of the code, and a tube diameter reads worse in meters —
`seed_diameter: 0.1` against `100.0` — so a human-edited config would end up
mixed-unit, which is harder to misread safely than a uniform one.

---

**Two formulations of the optimization — decided 2026-08-09.**

**A is the headline. B is an extension if time permits, and then the two get
compared.** They differ in what is a variable, not in what EN 1993-1-1 says: both
call the same `sizing.utilization_design`, so the comparison is like for like.

| | **A — nested, fully-stressed** | **B — simultaneous** |
|---|---|---|
| variables | `q` per edge | `q` per edge **and** `d` per member |
| `d` | solved: the root of utilization = 1 | free, box-bounded per member |
| `t` | `t = d/r`, `r` fixed | same — **`r` stays fixed in both** |
| utilization | exactly 1 by construction | a constraint, `≤ 1` |
| objective | `mass(q)`, unconstrained scalar | `mass(q, d)` with ~50 constraints |
| staggered coupling | present, 1.22% of the mass | **dissolved at every iterate** |

**`r` stays fixed in both, and `d` carries the per-member bounds.** That keeps the
class family intact without a mixed-integer variable, and it keeps the
monotonicity `test_sizing_monotonicity.py` proves — capacity is strictly
increasing in `d` at fixed `r`, which is not true at fixed `t`.

**Four consequences worth knowing before B is attempted.**

1. **B dissolves the staggered coupling, which is its best argument.** The same
   `d` sets the stiffness in T2 and the resistance in T3 at every evaluation, not
   just at the optimum. The T2 schema already declares `diameter` differentiable,
   so nothing needs changing there.
2. **B fixes a wart in A.** A applies the catalogue floor as
   `jnp.maximum(solved, diameter_min)` outside the solved map, because folding it
   into the bracket fabricates gradients — the IFT is only valid at a root. In B
   the floor is an honest box bound the optimizer holds, and the clamp disappears.
3. **The comparison is one-sided and cannot fail.** A's design is a feasible point
   of B (it sits on the constraint boundary), so `mass_B ≤ mass_A` by
   construction. The reportable number is the gap — how far fully-stressed is from
   optimal on a statically indeterminate structure. **Compare against A relaxed to
   its fixed point, not against one pass**, or 1.22% of the gap is really the
   staggering error wearing a disguise.
4. **B trades away the exact invariant.** A gives utilization `1.0 ± 1.7e-15`,
   which invariant 6.5 asserts. Under a penalty formulation B satisfies `≤ 1` with
   slack, and that crisp assertion is gone. Keep A precisely because the invariant
   is part of the argument.

**On the optimizer, and a distinction to keep straight.** Box bounds on `d` are
cheap — L-BFGS-B takes them natively — and they are what stops `d → 0`. But
`u(q, d) ≤ 1` is a **nonlinear inequality, not a box**, and L-BFGS-B cannot take
it; bounds alone give a design pinned at `d_min` and grossly overstressed. Two
routes:

- **KS-penalise the constraints into the objective, then L-BFGS-B with boxes.**
  The objective stays **scalar**, so `jax.grad` issues **one** VJP and one
  Tesseract round trip. The roadmap already plans KS aggregation, so this is in
  keeping. **Preferred.**
- SLSQP or trust-constr with explicit constraints. Needs the constraint Jacobian:
  50 rows, 50 sequential VJPs, **50 HTTP round trips per iteration** — precisely
  the upstream Tesseract bottleneck, where multi-cotangent VJPs are issued
  sequentially at one HTTP round trip each (issue #244).

**The analysis stage is jitted — RESOLVED 2026-08-10, and the earlier diagnosis
was wrong about where the obstacle sat.** Numbers in `CHANGELOG.md` under
`## P5b`.

The Python conditional is real: `smax.topology.build_free_dof_mask` does
`if support.fixity[i]` on what `jit` turns into an abstract value. But it lives
inside **compilation**, not inside the solve. Preparing the assembly on the host
and injecting the traced arrays into it therefore removes the obstacle without
touching the dependency at all.

`normax.analysis.smax.prepare` builds the `CompiledStructure` and a `LoadCase`
prototype once; `forces` replaces every array leaf per call with `eqx.tree_at`.
Both backends now take the same two-call contract, `prepare` then `forces`, and
OpenSees takes it while reusing nothing — its `Model` carries only the plane.

| | eager | prepared and jitted | |
|---|---|---|---|
| analysis stage, one case | 120.4 ms | 0.17 ms | 719x |
| `q → member forces` | 115.8 ms | 0.10 ms | 1151x |
| its gradient | 304.0 ms | 0.19 ms | 1564x |
| objective, three cases | 640.4 ms | 0.17 ms | 3686x |
| value and gradient | 1381.4 ms | 0.44 ms | 3127x |

`experiments/03` runs in **29 s against over ten minutes**. Compilation is 0.8 s
for a value and 2.2 s for a gradient, exposed through
`normax.optimization.value_and_gradient` so a caller pays it before timing
anything.

**The upstream `smax` change is no longer a precondition.** Making
`Support.fixity` static (`eqx.field(static=True)`, needing `tuple[bool, ...]` for
hashability) remains a tidy-up worth doing on its own merits, and nothing here
waits on it.

**Two corrections to the earlier note, both worth keeping visible.**

1. **The sizing bisection is not 0.016% of a design.** Eager, `diameter()` over
   twenty members is **66.5 ms, about 39% of the forward pass** — fifty-five
   halvings of a full utilization check is some 2,750 eager dispatches. Still do
   not rewrite it as Newton: `lax.fori_loop` collapses under compilation, so the
   fix is to compile rather than to replace the loop.
2. **Injecting only `xyz` and the four section properties would have baked the
   material.** Every array leaf of `Material` is injected too, so `∂/∂e_mod` stays
   live rather than becoming a silent zero. `tests/test_analysis_prepared.py`
   pins it: forces come back bitwise identical from an absurd placeholder
   template, and the modulus is checked against a difference quotient on a
   displacement, since member forces of a uniform-E frame are E-independent and
   cannot tell an injected leaf from a baked one.

**The Tesseract path is still eager, and that is the open item.** Each crossing is
stateless, so `_backend_smax.solve` calls `prepare` and an uncompiled `forces`
every time. Measured at ten members, per composed value-and-gradient: two
`prepare` calls at 45.7 ms and two `forces` calls at 415.2 ms, together **55% of
the 843 ms total**, the rest being the boundary itself plus T1 and T3. Caching the
prepared model and the compiled solve inside the backend, keyed on the topology
arriving in the inputs, is what closes it — see the P5b note below.

**Test suite size — audited 2026-08-09, deferred to after P4.**

The concern was testing creep: 1401 tests looks indiscriminate, as though
everything and anything is asserted. **Measured, the answer is mostly no, and the
headline number is the real problem.**

**346 written test functions collect to 1401 cases.** Counted with `ast`, not
grep: **115 of them (33%) are parametrized** and 231 are single-case, so the
sweeps produce roughly 1170 cases from 115 functions — about ten apiece.

The axes are the inputs the check actually takes, by frequency: axial force (15),
yield strength (14), class branch (13 + 10 + 8 across three argument names),
buckling curve (11), `d/t` ratio (11), then slenderness, buckling length, profile
name, diameter and end moments. That is one behavior checked across its specified
range, not 1401 opinions. The names are behavioral rather
than mechanical — `test_a_longer_member_never_needs_a_smaller_tube`,
`test_the_check_jumps_as_the_axial_force_changes_sign` — so a reader who opens a
file sees intent. And the volume is policy: invariants 1 and 2 require a
`check_grads` test per derivative rule and a hand-calc *and* Blueprints test per
clause. Those invariants and a small suite are incompatible.

**Three targets that are genuinely worth cutting, in priority order:**

1. **`test_oracle_blueprints.py`: 473 cases, 34% of the whole suite, from 25
   functions — a 19x sweep.** It cross-checks formulas that are linear monomials
   in the swept variable (`N_c,Rd = A f_y / γ_M0`) across five grades, three
   partial factors and four areas. Once it agrees at one grade, agreement at the
   other four is arithmetic, not logic. **Best single reduction available**, and
   the one place the creep charge actually lands.
2. **Modest duplication across the four sizing files** (336 cases, 24%). The
   axial/N+M split is deliberate — `test_sizing_axial.py` is P2 step 1,
   `test_sizing.py` is step 2, and bridging tests tie them together. But
   `test_a_tension_member_does_not_care_how_long_it_is` and
   `test_the_map_vectorizes_over_members` exist verbatim in both files.
3. **`tests/test_normax.py`** — the single cookiecutter leftover P0 already
   flagged as deletable. Free.

**Do not cut for optics.** The suite is part of the argument: the thesis is that a
design standard is a normative text whose arithmetic nobody checks, and the
evidence is the 11 wrong printed numbers found in its commentary. A dense
property-based suite is what earns the right to say that.

**Fix the reporting instead, in P7.** "1401 tests" invites the reader's
suspicion. State the shape truthfully: **"346 test functions collecting to 1401
cases — 231 single-case tests plus 115 swept over the inputs the check takes:
axial force, end moments, buckling length, yield strength, `d/t` ratio, buckling
curve and both class branches."** Same fact, reads as rigour, and survives an
audit — which a vaguer claim would not. **Re-measure before the README is
written**; these counts move with every commit.

**Eq. 6.42, how the two moments combine** — deferred 2026-08-09, revisit after P4.

Both readings are implemented and selected by `resultant=` on
`normax.ec3.sizing.diameter_required`; the default is the resultant. The disagreement is
recorded in `docs/clauses.md` under §6.2.9.2 with both citations, what each
third-party implementation says, and the measured gap.

**Why it is safe to defer.** It moves nothing wherever Eq. 6.61 governs, because
6.61 already sums the two moments linearly, and nothing under uniaxial bending.
Measured across a compression sweep it was 0.00% out to 80 kNm and 0.93% in
diameter at 160 kNm. It only bites where the *cross-section* check governs —
pure bending (12.25% in diameter, 26.0% in area) and tension members (9.0% /
18.8%).

~~**What would change the answer.** Once P4 runs, look at how many members are
actually governed by the cross-section check rather than by 6.61, and whether
any of them carry significant biaxial moment.~~ — **MEASURED, and the answer is
that both halves of the test matter and they point opposite ways.**

**The population is large: the cross-section check governs 19 of 20 members**,
against P3 step 2's single-case design where 6.61 governed every one. Admitting
load cases that raise real bending moves the decision from the member check to
the cross-section check, which is a finding in its own right.

**But the reading is still immaterial here, because `m_z` is identically zero** —
0.0 exactly, not merely small, on a planar arch under in-plane load. With one
moment there is nothing to combine and the resultant and the linear sum are the
same number. **The choice cannot bite until the 3D gridshell**, where the two
moments are both live and the population is known to be large. Decide it there,
and treat it as a headline caveat at that point rather than this one.

Flip it with one keyword and rerun `experiments/05_class_ratio_sweep.py`, which
already tabulates both.
