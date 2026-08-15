# EN 1993-1-1 — clause reference for `normax`

**STATUS: VERIFIED** against Gardner, L. & Nethercot, D. (2011), *Designers' Guide
to Eurocode 3: Design of Steel Buildings*, 2nd edn, ICE Publishing
(ISBN 978-0-7277-4172-1), accessed via Princeton University Library.

The book is now in the repo at **`references/9780727741721.pdf`**. Check it
directly rather than trusting this file when a number looks wrong — doing so
on 2026-08-08 corrected the fixture's example number and caught an arithmetic
error in the guide itself. Page numbers below are the book's own.

Version targeted: **EN 1993-1-1:2005+A1:2014**. Do **not** use EN 1993-1-1:2022 —
it restructures the standard and moves buckling to chapter 8.

Equations, symbols and units only. No clause prose in this file.

Scope: CHS members under **axial force plus biaxial bending**. No LTB (CHS is
closed and doubly symmetric, so χ_LT = 1 — a correct exclusion, not a
simplification). No torsion, no shear (see Open items).

> **Guide vs standard table numbering.** The Designers' Guide renumbers EC3's
> tables. Guide Table 6.4 = EN 1993-1-1 **Table 6.1**; Guide Table 6.5 =
> EN 1993-1-1 **Table 6.2**. Cite the EN numbers in docstrings.

---

## Notation

| Symbol | Meaning | Unit |
|---|---|---|
| `d` | outer diameter | mm |
| `t` | wall thickness | mm |
| `d_i` | inner diameter = `d − 2t` | mm |
| `r` | fixed ratio `d/t` | – |
| `A` | gross cross-sectional area | mm² |
| `A_net` | net area at fastener holes | mm² |
| `I` | second moment of area | mm⁴ |
| `i` | radius of gyration `√(I/A)` | mm |
| `f_y` | yield strength | N/mm² |
| `f_u` | ultimate tensile strength | N/mm² |
| `E` | modulus of elasticity = 210 000 (§3.2.6) | N/mm² |
| `ε` | `√(235/f_y)` | – |
| `L_cr` | buckling length | mm |
| `N_Ed` | design axial force (tension +ve) | N |
| `N_cr` | elastic critical force | N |
| `λ̄` | non-dimensional slenderness | – |
| `λ₁` | `π√(E/f_y)` = `93.9ε` | – |
| `Φ` | auxiliary term in the buckling curve | – |
| `χ` | reduction factor, flexural buckling | – |
| `α` | imperfection factor | – |
| `γ_M0, γ_M1, γ_M2` | partial factors | – |

---

## §5.2 and §6.3.4 — Global stability  ✅ VERIFIED 2026-08-09

Verified against both books, page by page. Every threshold survived; two
equation numbers did not. What changed is recorded in open item 0f.

- [x] **§5.2.1(3)**: first-order analysis is adequate when `α_cr ≥ 10` for
      elastic analysis and `α_cr ≥ 15` for plastic analysis. Both numbers
      confirmed — guide p. 18, ECCS pp. 79–80, and again at ECCS p. 369.
- [x] Eq. **5.1 is that threshold pair, not the definition of `α_cr`.**
      `α_cr = F_cr / F_Ed` — the factor by which the design load must be
      multiplied to reach elastic instability **in a global mode** — sits in the
      clause's `where` list and carries no equation number of its own. The guide
      gives it one of its own making, `(D5.1)`, which is what exposed this.
- [x] **UK NA clause NA.2.9** lowers the *plastic* limit to `α_cr ≥ 10` for clad
      structures whose masonry infill or profiled sheeting is not counted as
      stiffening, and to `α_cr ≥ 5` for portal frames under gravity loads only.
      The elastic limit of 10 is untouched. Relevant because we already adopt
      this National Annex for the partial factors (NA.2.15).
- [x] **§5.2.2(5)**: sway effects amplified by `1 / (1 − 1/α_cr)`, applied to
      `H_Ed` and to the equivalent horizontal loads `V_Ed φ` from imperfections,
      valid down to `α_cr ≥ 3.0` only. Below that a genuine second-order analysis
      is required. §5.2.2(6) extends the same factor to multi-storey frames.
      Confirmed ECCS p. 276. **No equation number** — neither book prints EN's,
      so cite the clause alone; neither confirms a `B` suffix on it either.
- [x] Eq. **5.2** (**§5.2.1(4)B**): an approximate `α_cr` for sway modes from
      storey drift, `α_cr = (H_Ed/V_Ed)(h/δ_H,Ed)`, for portal frames with roof
      slopes under 26° and for beam-and-column plane frames, subject to the same
      clause's restriction on axial compression in the rafters. Confirmed guide
      p. 19. **Not used** — we compute `α_cr` from an eigenvalue analysis
      instead, so the approximation is never needed. The books disagree on where
      `H_Ed` is measured; ECCS records that EN's original *bottom of the storey*
      was corrected to *top* by corrigendum.
- [x] **§6.3.4(3)**, the general method: `λ̄_op = √(α_ult,k / α_cr,op)`, with
      `α_ult,k` the amplifier reaching the characteristic cross-section
      resistance and `α_cr,op` the amplifier reaching elastic instability.
      §6.3.4(2) is the check it feeds, `χ_op α_ult,k / γ_M1 ≥ 1`. Confirmed
      ECCS pp. 299–302. **The number 6.64 is not confirmed** — neither book
      prints EN's numbering for this clause, so cite `§6.3.4(3)`.

### §6.3.4 is an out-of-plane method and our use of the algebra is not

`α_cr,op` is defined as the amplifier reaching elastic instability **with respect
to lateral or lateral-torsional buckling**, and ECCS p. 300 states that no
account is taken of in-plane flexural buckling. The UK NA (NA.2.22) narrows it
further, to straight members under in-plane mono-axial bending or compression
with `χ_op = min(χ, χ_LT)`, and the guide recommends the whole clause be used
with caution as it is new and thinly validated.

**The mode we measure on the arch is in-plane by construction** —
`normax.analysis.smax.buckling` restrains the one translation normal to the plane
precisely so the modes stay in it. So §6.3.4 is where the standard writes this
algebra, not authority for the number we report from it. The identity below is
what carries our use; the clause is a citation for the form, not for the case.

### What `normax` passes as `L_cr` — a modelling choice, not a clause

**Every member is assumed to buckle over its own length.**
`StructuralDesignPipeline` measures the form-found geometry and hands that length
to the check, and there is no way to pass anything else: the pipeline takes no
buckling length and a `Design` reports no separate one. Composing the three
blocks by hand is what a stated `L_cr` needs, as
`experiments/09_arch_pipeline_jax.py` does to price this assumption.

That is a **strong assumption rather than a conservative one**. It presumes
every node of the model is held in position by structure outside it. Where that
does not hold the frame buckles in a mode spanning several members, the true
`L_cr` exceeds the member length, and the design is unsafe rather than cautious.

The clause layer is unaffected and stays general: `force_critical`,
`slenderness_from_force` and every function in `normax.ec3.sizing` take `L_cr`
as an argument, because Eq. 6.50 is written in `L_cr` and not in a member
length. What is fixed is one composition's choice of what to pass.

**How to see the size of the assumption.** `frame_stability` reports
`buckling_length_equivalent`, the `L_cr` the frame's own critical load factor
corresponds to, beside `slenderness_member` from the assumed length. Their ratio
is one wherever the assumption holds. Measured on the arch, a global buckling
length costs several times the mass of a member-length one, so this is not a
small choice — it is deferred, not settled.

### The two routes to `λ̄` are the same equation — exact, no source needed

Eq. 6.50 takes the slenderness from a **member** buckling length; §6.3.4(3) takes
it from a **system** critical load factor. For pure compression they are
algebraically identical, since `α_ult,k = A f_y / N_Ed` and `α_cr = N_cr / N_Ed`:

```
α_ult,k / α_cr = (A f_y / N_Ed) · (N_Ed / N_cr) = A f_y / N_cr = λ̄²
```

**This identity needs no reference** and is asserted directly in
`tests/test_stability.py`. It is what lets a buckling length be recovered from a
critical load factor, `L_cr = π √(E I / (α_cr · N_Ed))`, and it is the reason the
two routes may be fed to the same `χ` and compared.

What differs is not the equation but what each route is asked about: Eq. 6.50
answers for one member over an assumed length, §6.3.4(3) for the mode the
structure actually has. On the arch they disagree by a factor of 4.7 in `λ̄`,
which is the size of the braced-node assumption rather than a discrepancy.

---

## §5.5.2 / Table 5.2 (sheet 3) — Classification, tubular sections

Verified. Limits apply to **sections in bending and/or compression**:

- [x] Class 1: `d/t ≤ 50ε²`
- [x] Class 2: `d/t ≤ 70ε²`
- [x] Class 3: `d/t ≤ 90ε²`
- [x] `ε = √(235/f_y)` — **squared in the limits, confirmed**
- [x] Beyond `90ε²`, EN 1993-1-6 (shell buckling) applies — outside our scope,
      and the reason `d/t ≤ 90ε²` is an honesty constraint, not just numerical

Tabulated values from Table 5.2:

| `f_y` | 235 | 275 | 355 | 420 | 460 |
|---|---|---|---|---|---|
| `ε` | 1.00 | 0.92 | 0.81 | 0.75 | 0.71 |
| `ε²` | 1.00 | 0.85 | **0.66** | 0.56 | 0.51 |

**Design decision:** `d/t = 90ε²`, the Class 3/4 boundary. For S355 this is
`90 × 0.66 = 59.4` from the tabulated value, or `59.58` from exact `235/355`.
Use the exact value in code; the table is rounded to 2 d.p.

**Caution.** The guide's own Worked Example 6.7 (p. 62) evaluates the Class-1
limit as 40.7, which is `50ε`, not `50ε²` (= 33.10). The table above is the
authority; the worked example's arithmetic is wrong. See the errata note under
the test fixture.

---

## §6.2.3 — Tension

- [x] Eq. **6.6**: `N_pl,Rd = A · f_y / γ_M0`   — yielding of gross section
- [x] Eq. **6.7**: `N_u,Rd = 0.9 · A_net · f_u / γ_M2`   — net section at holes
- [x] `N_t,Rd = min(N_pl,Rd, N_u,Rd)`
- [ ] Eq. 6.5 (`N_Ed / N_t,Rd ≤ 1.0`) — not reproduced in the guide; number
      inferred. Cite §6.2.3 without the equation number, or confirm separately.

**Used:** 6.6 only. No holes in the MVP.

*Note: the 0.9 factor in 6.7 comes from a statistical evaluation of net-section
plate tests (ECCS, 1990), included so γ_M2 could be harmonized with connection
resistance. Irrelevant to us, but explains the odd constant.*

---

## §6.2.4 — Compression

- [x] Eq. **6.10**: `N_c,Rd = A · f_y / γ_M0`   (Class 1, 2 or 3)
- [x] Eq. **6.11**: `N_c,Rd = A_eff · f_y / γ_M0`   (Class 4)
- [ ] Eq. 6.9 (`N_Ed / N_c,Rd ≤ 1.0`) — not reproduced in the guide; inferred.

**Used:** 6.10. Class 4 unreachable by construction.

---

## §6.2.9 — Bending and axial force  ✅ VERIFIED 2026-08-08

Verified against **both** `references/simoes-da-silva-eccs-design-of-steel-structures.txt`
(ECCS manual — the better source here) and
`references/gardner-nethercot-designers-guide.txt`. Page numbers are the books' own.

> The guide's N+M worked examples (6.9, 6.10) carry errata #5–#9, all in the
> 6.61/6.62 substitutions. Prefer property tests and the ECCS examples.

### §6.2.9.1(4) — when the reduction may be neglected

- [x] Eqs. **6.33 / 6.34** (major axis) and **6.35** (minor axis).
- [x] **CHS is NOT eligible.** Both books enumerate the section types exhaustively
      and CHS appears in none of them. **Always compute `M_N,Rd`.** Self-consistent:
      the criteria are written in terms of `h_w` and `t_w`, which a CHS has not.

*The two books disagree on the scope of the exemption — the guide (p. 51) adds
"other flanged sections" (major) and "rectangular rolled hollow sections and
welded box sections" (minor); the ECCS (p. 228) says I/H only, both axes. The
guide matches EN's wording. Moot for CHS, material if we ever add RHS.*

### §6.2.9.1(5) — reduced plastic moment resistance, CHS

- [x] **`M_N,Rd = M_pl,Rd (1 − n^1.7)`** with `n = N_Ed / N_pl,Rd`.
      **ECCS p. 228, its eq. (3.131).**
- [x] **It carries no EN equation number.** The ECCS gives it only its own number
      and no clause tag; the numbered EN equations are 6.36–6.38 (I/H) and
      6.39–6.40 (rectangular hollow / box). It is an *unnumbered inline formula
      inside §6.2.9.1(5)* — the same situation as `Φ` beneath Eq. 6.49. Cite
      **"§6.2.9.1(5), unnumbered"** and nothing more.
- [x] **No validity limits are printed** — no `d/t` bound, no uniform-thickness
      clause, no fastener-hole caveat, no range on `n`. Pointedly unlike the
      heavily qualified I/H and RHS expressions on the same page.
- [x] **The Designers' Guide omits the CHS rule entirely** (its list on pp. 51–52
      is a closed two-item list). Do not look for it there.
- [x] `M_pl,Rd = W_pl f_y / γ_M0` — Eq. **6.13**.

**Exact parent, for cross-checking only.** ECCS p. 226 eq. (3.119) gives the
closed-form plastic interaction for a circular tube (Lescouarc'h, 1977):

```
M_N,Rd = M_pl,Rd · sin(π(1 − n)/2)
```

The books never state that `1 − n^1.7` approximates it, but it plainly does.
Useful as an independent test oracle. **Note it is not a bound**: the codified
form is below the exact one for `n < 0.692` and *above* it beyond that, peaking
about 4.8% high near `n = 0.9`.

*Smoothness, stated precisely: `d/dn (1 − n^1.7) = −1.7 n^0.7` → 0 as `n → 0⁺`,
so the **first** derivative is finite and continuous at the origin. It is the
**second** derivative, `−1.19 n^−0.3`, that diverges. First-order gradients and
`check_grads(order=1)` are safe; `order=2` will fail near pure bending. Guard
`n` away from zero only if second-order accuracy is needed.*

### §6.2.9.1(6) — biaxial bending

- [x] Eq. **6.41**: `[M_y,Ed/M_N,y,Rd]^α + [M_z,Ed/M_N,z,Rd]^β ≤ 1`
- [x] **For circular hollow sections: `α = β = 2`.** Confirmed independently by
      both books (ECCS p. 229; guide p. 54). I/H take `α = 2`, `β = 5n ≥ 1`;
      rectangular hollow take `α = β = 1.66/(1 − 1.13n²) ≤ 6`.
- [x] Guide Figure 6.16 plots CHS on the same curve as I/H at `n = 0.4`
      (`β = 5 × 0.4 = 2`) — independent confirmation of `α = β = 2`.
- [x] `α = β = 1` is a permitted conservative fallback (both books).
- [x] Blueprints has `Form6Dot41BiaxialBendingCheck` — returns a **bool** via
      `__bool__`, not a ratio. Test oracle only.

**The CHS collapse — exact at cross-section level.** A CHS is axisymmetric, so
`W_pl,y = W_pl,z` and `M_N,y,Rd = M_N,z,Rd = M_N,Rd`. With `α = β = 2`:

```
(M_y² + M_z²) / M_N,Rd²  ≤ 1     ⟺     M_res = √(M_y² + M_z²) ≤ M_N,Rd
```

**One resultant moment, exactly — no approximation, no exponent to smooth.**
This holds *only* at cross-section level. See §6.3.3 for why it does **not**
carry over to the member check.

### §6.2.9.2 / §6.2.9.3 — Class 3 and Class 4

- [x] **Class 3, Eq. 6.42**: `σ_x,Ed ≤ f_y / γ_M0`, elastic stresses on the gross
      section. *(The guide p. 55 misprints this with `=` for `≤`; the ECCS and
      the guide's own prose both have `≤`.)*

#### 🚩 How the two moments combine in Eq. 6.42 — decision recorded 2026-08-09

The two books read this differently, and it is material.

- The **guide p. 55** says clause 6.2.9.2 *"permits only a **linear interaction
  of stresses** arising from combined bending moments and axial force"*, and its
  Class 4 analogue Eq. **6.44** is a three-term linear sum, `N/(A f_y/γ_M0) +
  M_y/(W_y f_y/γ_M0) + M_z/(W_z f_y/γ_M0) ≤ 1`.
- The **ECCS p. 229** says `σ_x,Ed` *"is evaluated by an **elastic stress
  analysis**, based on the gross cross section for class 3"*.

**Decision: the resultant, `n + √(M_y² + M_z²)/M_el,Rd`.** Eq. 6.42 limits a
*stress*, and on a circular perimeter the bending stress at angle `θ` is
`(M_y sinθ − M_z cosθ)/W_el`, whose greatest magnitude around the perimeter is
the resultant over `W_el`. The linear sum evaluates the stress at a point where
neither component actually peaks. The guide's word *linear* contrasts 6.2.9.2
with 6.2.9.1's **plastic** `1 − n^1.7` interaction — it governs how axial and
bending combine, which here is linear — and says nothing about how two bending
components about perpendicular axes combine.

**Interpretation, not citation** — the same standing as the Table B.1 row.
**Both readings are implemented**, selected by `resultant=` on
`resistance.utilization_elastic` and threaded through the sizing map, so the gap
is reported rather than assumed. Measured on a 6 m Class 3 member: **12.25% in
diameter and 26.0% in area** in pure bending, falling to zero under uniaxial
bending and to zero wherever Eq. 6.61 governs, since 6.61 already sums the two
moments linearly.

**What the implementations say — checked 2026-08-09.**

- **Karamba: silent.** No `elast`, `W_el`, `sigma`, `class 3` or any of 6.41–6.44
  anywhere in its 547-line EC3 source. It implements no §6.2.9 at all and cannot
  arbitrate, which matches `references/README.md`.
- **Blueprints: neutral on 6.42, linear on 6.44.**
  `Form6Dot42LongitudinalStressClass3CrossSections` takes `sigma_x_ed` as an
  **input** and only asserts the limit, so it declines to say how the stress is
  assembled — exactly as EN does. But
  `Form6Dot44CombinedCompressionBendingClass4CrossSections` writes the Class 4
  analogue out as an explicit **three-term linear sum**.

So the linear reading carries the guide's prose *and* EN's own written-out form,
independently implemented. The counter that keeps the resultant as the default:
Eq. 6.44 uses `W_eff,y,min` and `W_eff,z,min` — *minimum* moduli, the drafting of
a general-section envelope for shapes whose extreme fibre sees both moments at
once, such as the corner of an I or a box. On a circle the two bending stress
fields are 90° out of phase, so the sum is an upper bound on the stress rather
than the stress, and 6.42 asks for the stress. **Default `resultant=True`, with
the sum one keyword away.**

Note this does **not** carry over to the member check: §6.3.3 keeps the two
moments separate, and that is settled by open item 0a, not by this.
- [x] **Class 4, Eq. 6.44**: the same limit on effective properties, plus the
      neutral-axis shift terms `N_Ed e_N`.
- [x] **Eq. 6.43 is printed in neither book.** If it is ever needed, neither of
      these is a source.

**This clause governs our members whenever `d/t` is set to 90ε²** — see the
section-class policy below.

---

## §6.3.1 — Buckling resistance of compression members

- [x] Eq. **6.47**: `N_b,Rd = χ · A · f_y / γ_M1`   (Class 1, 2 and 3)
- [x] Eq. **6.48**: `N_b,Rd = χ · A_eff · f_y / γ_M1`   (symmetric Class 4)
- [ ] Eq. 6.46 (`N_Ed / N_b,Rd ≤ 1.0`) — not reproduced in the guide; inferred.

**Used:** 6.47.

### §6.3.1.2 — Buckling curves

- [x] Eq. **6.49**: `χ = 1 / (Φ + √(Φ² − λ̄²))`, **but `χ ≤ 1.0`**
- [x] `Φ = 0.5 · [1 + α(λ̄ − 0.2) + λ̄²]` — **carries no equation number**; it sits
      unnumbered beneath Eq. 6.49. Cite as "§6.3.1.2, unnumbered, below Eq. 6.49".
- [x] **§6.3.1.2(3)**: for `λ̄ ≤ 0.2`, or equivalently `N_Ed/N_cr ≤ 0.04`, there is
      no reduction — buckling may be ignored and only §6.2 checks apply.
      This is what `min(χ, 1)` implements.

### §6.3.1.3 — Flexural buckling slenderness

*(Note: this is 6.3.1.3, NOT 6.3.1.2 — the two are separate clauses.)*

- [x] Eq. **6.50**: `λ̄ = √(A f_y / N_cr) = (L_cr / i) · (1 / λ₁)`   (Class 1, 2, 3)
- [x] Eq. **6.51**: `λ̄ = √(A_eff f_y / N_cr) = (L_cr / i) · √(A_eff/A) / λ₁`  (Class 4)
- [x] Eqs. **6.52 / 6.53**: torsional / flexural-torsional `λ̄_T` — **not applicable**.
      These modes are generally limited to cold-formed open sections; CHS is
      closed and doubly symmetric, so flexural buckling governs.
- [x] `N_cr = π² E I / L_cr²` — general theory, used explicitly in the guide's
      worked examples. Not itself a numbered EC3 equation.

---

## §6.3.3 — Uniform members in bending and axial compression  ✅ VERIFIED

- [x] **Eq. 6.61 and Eq. 6.62**, transcribed from guide p. 77 and ECCS p. 234
      (which numbers them 3.144a/b), in agreement:

```
6.61   N_Ed/(χ_y N_Rk/γ_M1) + k_yy·M_y,Ed/(χ_LT M_y,Rk/γ_M1) + k_yz·M_z,Ed/(M_z,Rk/γ_M1) ≤ 1
6.62   N_Ed/(χ_z N_Rk/γ_M1) + k_zy·M_y,Ed/(χ_LT M_y,Rk/γ_M1) + k_zz·M_z,Ed/(M_z,Rk/γ_M1) ≤ 1
```

- [x] **`χ_LT` sits on the `M_y` term only, in both equations.** The `M_z` term
      carries no reduction factor at all. The axial term is the only one that
      differs between the two.
- [x] **`χ_LT = 1.0` for members not susceptible to torsional deformation** —
      stated in both books, and a CHS qualifies (see Annex B below).
- [x] The `ΔM` terms in the full ECCS form are the Class 4 neutral-axis shift and
      are zero for Classes 1–3.

### `N_Rk` and `M_Rk` by class — EN Table 6.7

ECCS Table 3.12, p. 235. **Neither book cites the EN table number**, so cite the
clause, not "Table 6.7".

| Class | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| `A_i` | `A` | `A` | `A` | `A_eff` |
| `W_y` | `W_pl,y` | `W_pl,y` | **`W_el,y`** | `W_eff,y` |
| `W_z` | `W_pl,z` | `W_pl,z` | **`W_el,z`** | `W_eff,z` |
| `ΔM` | 0 | 0 | 0 | `e_N N_Ed` |

`N_Rk = f_y A_i`, `M_i,Rk = f_y W_i`.

### ⚠️ CORRECTION — 6.61 and 6.62 do NOT become the same equation

An earlier draft of this file claimed they collapse. **They do not.** For a CHS,
`χ_y = χ_z = χ` and `M_y,Rk = M_z,Rk = M_Rk`, and `k_yz = 0.6 k_zz`,
`k_zy = 0.6 k_yy` (Class 1/2), so:

```
6.61 = N-term + [ k_yy·M_y + 0.6·k_zz·M_z ] / (M_Rk/γ_M1)
6.62 = N-term + [ 0.6·k_yy·M_y + k_zz·M_z ] / (M_Rk/γ_M1)
```

These coincide **only when `M_y = M_z`**. Verified numerically: at `M_y = 100`,
`M_z = 20` kNm they give 0.91 and 0.72. Nor is `k_yy = k_zz` automatic — the
*forms* match for a CHS, but `C_my` and `C_mz` are read off the moment diagrams
about *different* axes and need not be equal.

**It is still one check**, just not one equation — take the worse:

```
utilization = N-term + [ max(k_yy·M_y, k_zz·M_z) + 0.6·min(k_yy·M_y, k_zz·M_z) ] / (M_Rk/γ_M1)
```

### The resultant moment is NOT admissible here

- [x] **Confirmed absent.** Exhaustive search of both books and every NCCI they
      cite found no sanction anywhere for combining `M_y` and `M_z` into
      `√(M_y² + M_z²)` in 6.61/6.62, for any section type.
- [x] The guide (pp. 76–77) is explicit that the axis-by-axis linearity is the
      deliberate design simplification: *"Although there is a coupling between the
      member response in the two principal planes, this is generally safely
      disregarded in design."*

**So: the resultant is exact at cross-section level (Eq. 6.41, `α = β = 2`) and
inadmissible at member level (6.61/6.62).** Keep the two moments separate here.

---

## Annex B (method 2) — interaction factors  ✅ VERIFIED

Guide Chapter 9 (its Tables 9.1/9.2/9.3 = EN B.1/B.2/B.3); ECCS Tables
3.16/3.17/3.18.

### Which table — CHS goes to Table B.1

- [x] **Table B.1** = members **not** susceptible to torsional deformations;
      **Table B.2** = susceptible.
- [x] **A CHS is not susceptible.** ECCS p. 239 lists, verbatim, *"members with
      circular hollow sections"* first among the not-susceptible cases. Guide
      p. 77 says the same. Cross-check: Annex A's criterion is `I_T ≥ I_y`, and
      for a CHS `I_T = I_p = 2I_y` always — it passes unconditionally.

### 🚩 Table B.1 has no CHS row — decision recorded

Both books reproduce Table B.1 with a *"Type of sections"* column containing only
**I-sections** and **RHS-sections**. **Neither book says which row a circular
tube takes.** EN sends CHS to this table and then does not list it.

**Decision: use the RHS-sections row.** It is the only closed-section entry, and
Karamba3d's EC3 implementation independently resolves it the same way — it
branches on `is_I_profile` and gives every non-I section the RHS expressions.
Record this as an interpretation, not a citation, in the writeup.

### `k_ij`, Class 1 and 2 (plastic) — RHS row

```
k_yy = C_my (1 + (λ̄_y − 0.2)·nʸ)   ≤  C_my (1 + 0.8·nʸ)
k_zz = C_mz (1 + (λ̄_z − 0.2)·nᶻ)   ≤  C_mz (1 + 0.8·nᶻ)     [RHS row; I-sections use (2λ̄_z − 0.6) and a 1.4 cap]
k_yz = 0.6 k_zz
k_zy = 0.6 k_yy
```

### `k_ij`, Class 3 and 4 (elastic) — shared by I and RHS

```
k_yy = C_my (1 + 0.6·λ̄_y·nʸ)  ≤  C_my (1 + 0.6·nʸ)
k_zz = C_mz (1 + 0.6·λ̄_z·nᶻ)  ≤  C_mz (1 + 0.6·nᶻ)
k_yz = k_zz          ← not 0.6 k_zz
k_zy = 0.8 k_yy      ← not 0.6 k_yy
```

with `nʸ = N_Ed/(χ_y N_Rk/γ_M1)` and `nᶻ = N_Ed/(χ_z N_Rk/γ_M1)`.

*Notation warning: EN writes these ratios out in full. The symbols `n_y`/`n_z`
appear nowhere in either book — they are our shorthand. Do not confuse them with
Annex A's `n_pl = N_Ed/(N_Rk/γ_M1)`, which has **no** `χ`.*

- [x] A footnote to Table B.1 permits `k_zy = 0` for I/H and rectangular hollow
      sections under compression plus uniaxial `M_y`. Permissive, not required —
      the ECCS example takes it, the guide's does not.

### `C_m` — Table B.3, and which row we use

- [x] **Row 1, linear moment diagram: `C_m = 0.6 + 0.4ψ ≥ 0.4`.**
      **This is the row we use, and it is exact rather than approximate for us**:
      loads are applied at nodes only, so there is no span loading and the moment
      diagram between nodes is linear by construction.
- [x] `ψ` is the ratio of the smaller to the larger end moment, signed.
- [x] `C_my`, `C_mz`, `C_mLT` are each read off the moment diagram about a
      different axis, between the relevant braced points. `C_mLT` is irrelevant
      to us since `χ_LT = 1`.
- [x] **Sway rule:** for members with a sway buckling mode, `C_my = C_mz = 0.9`.
      Karamba applies this as an unconditional floor (`res = res < 0.9 ? 0.9 :
      res`). That is **not** an EC3 requirement for non-sway members. If adopted,
      make it an optional flag and say so.
- [ ] **Books disagree on one cell** (row 3c, concentrated load): ECCS p. 241
      prints `0.90 + 0.10 α_h(1 + 2ψ)`, guide p. 111 prints `0.90 − 0.10 α_h(1 +
      2ψ)`. Both read from rendered pages, so it is a real printed disagreement.
      **Out of our path** — that cell needs span loading, which we do not have.
      Resolve against EN itself only if span loading is ever added.

### Method 1 vs Method 2

- [x] The UK NA allows either; Annex A is restricted to doubly symmetric sections
      (a CHS qualifies), Annex B applies generally, and hollow sections are inside
      Annex B's *unrestricted* scope, keeping full plastic benefit.
- [x] Neither book prefers one for hollow sections. Method 2 is simpler; the ECCS
      finds Method 2 less conservative in its RHS example and *more* conservative
      in its IPE example, so the ordering is not fixed. **We use Method 2.**

### The non-smoothness that actually matters

1. **The cap on `k_yy`/`k_zz`** — a C⁰ kink, same shape as the `χ ≤ 1` cap.
   Handle identically; report via `governing`.
2. **`C_m`'s `≥ 0.4` floor** — another C⁰ kink, in `ψ`.
3. The rest of Table B.3's piecewise structure never activates for us, because
   nodal loading pins us to row 1. That is a real simplification bought by the
   loading model, and it should be stated in the writeup rather than hidden.

**Sizing-map monotonicity survives.** With `A ∝ d²`, `W ∝ d³`, `χ` increasing in
`d`, and `nʸ`/`nᶻ` therefore falling so the `k_ij` fall too, every term of the
interaction strictly decreases in `d`. The residual stays monotone, the root
stays unique, bisection stays unconditionally safe, and the implicit
differentiation is unchanged. **The expansion adds clauses, not a new method.**

### The discontinuity at zero axial force — measured 2026-08-09

§6.3.3 is titled *"uniform members in bending and axial compression"* and does
not apply in tension, so a member a whisker either side of zero axial force is
held to two different requirements. As `N → 0⁻` the member check tends to the
**linear sum** `C_my·M_y + k_yz·M_z`, while the cross-section check compares the
**resultant**. They do not meet.

| Branch | `k_yz` | jump in utilization at `M_y = M_z`, `C_m = 1` | jump in area |
|---|---|---|---|
| Class 1/2 (plastic) | `0.6 k_zz` | `1.6/√2` = **1.131** | ~21% |
| Class 3 (elastic) | `k_zz` | `2.0/√2` = **1.414** | ~26% |
| either, uniaxial | — | **1.000**, continuous | 0% |

The elastic branch is worse because Table B.1's Class 3/4 column drops the 0.6
factor on `k_yz`. **This is inherited from the standard, not introduced here.**
It cannot be smoothed honestly — a sigmoid across a jump gives a large, bounded
derivative that is wrong throughout the transition band. Report it, flag sign
flips through `governing`, and quote these numbers in the writeup.

---

## SECTION-CLASS POLICY — decided 2026-08-08

`d/t` is a fixed configuration ratio, so **the cross-section class is static, not
traced.** Selecting the plastic or elastic branch is therefore an ordinary Python
choice at build time and introduces no branch on a traced value.

**That is a consequence of the decision, not a constraint that forced it.** A
traced class is perfectly expressible — `lax.switch` branches on a traced value,
and CLAUDE.md invariant 4 names it. The reason not to is that **the class boundary
is a discontinuity in the standard**: `M_Rk` steps from `W_pl f_y` to `W_el f_y`
across it, 24.6% for a CHS, and tracing the step does not smooth it. Fixing `d/t`
means the question never arises, since `t = d/r` leaves the ratio invariant in the
diameter and no size the design takes can change its class.

| `d/t` | Class | `M_Rk` uses | Cross-section N+M | `k_ij` column |
|---|---|---|---|---|
| `70ε²` | 2 | `W_pl` | `M_N,Rd = M_pl(1 − n^1.7)`, Eq. 6.41 | Class 1/2 (plastic) |
| `90ε²` | 3 | `W_el` | Eq. 6.42 elastic stress | Class 3/4 (elastic) |

**Both branches are implemented and the ratio stays a config parameter.** The
original `90ε²` decision (CLAUDE.md §3) was made when the scope was axial-only,
where Classes 1–3 all use the gross area `A` so the choice was free. Bending
breaks that: the shape factor `W_pl/W_el` for a CHS is **1.326**, so Class 3
forfeits 24.6% of bending capacity — against the thin-wall efficiency of a
larger `d/t`. Which wins at the optimum is not decidable a priori; it is an
experiment, and the answer is a result worth reporting.

---

## Table 6.1 — Imperfection factors  *(Guide Table 6.4)*

- [x] curve `a0` → α = 0.13
- [x] curve `a`  → α = 0.21
- [x] curve `b`  → α = 0.34
- [x] curve `c`  → α = 0.49
- [x] curve `d`  → α = 0.76

## Table 6.2 — Buckling curve selection  *(Guide Table 6.5)*

- [x] **Hot-finished CHS → curve `a`** (confirmed in Worked Example 6.7)
- [x] Cold-formed hollow sections → curve `c`; S460 hot-finished → curve `a0`,
      S460 cold-formed → curve `c`. **Confirmed 2026-08-08** against the guide's
      Table 6.5, the "Hollow sections" rows.

**Used:** hot-finished S355 CHS → curve `a`, α = 0.21. Keep as a parameter.

*Material standards: EN 10210 = hot-finished hollow sections;
EN 10219 = cold-formed. `f_y` for S355 CHS is taken from EN 10210-1.*

---

## Partial factors (§6.1)

- [x] `γ_M2 = 1.25` — the EN recommended value, retained in EN 1993-1-8 for
      connection parts. **The UK NA sets `γ_M2 = 1.10`** for net-section
      fracture (used in Worked Example 6.1), so this one is context-dependent.
      Not used in the MVP.
- [x] `γ_M0 = 1.00`, `γ_M1 = 1.00` — **confirmed 2026-08-08**. The guide cites
      **clause NA.2.15 of the UK National Annex** for both, in Worked Examples
      6.1, 6.2, 6.4, 6.8 and 6.9.

Nationally Determined Parameters; National Annexes may differ (the UK NA sets
them under NA.2.14 / NA.2.15). State the values used in the writeup.

---

## CHS section properties (geometry — not EC3)

With `t = d/r` and `d_i = d(1 − 2/r)`:

```
A     = π t (d − t)       = π d² (r − 1) / r²
I     = (π/64)(d⁴ − d_i⁴) = (π/64) d⁴ [1 − (1 − 2/r)⁴]
W_el  = 2I / d
W_pl  = (d³ − d_i³) / 6
i     = √(I/A) = c·d      (c depends only on r)
```

Since `i ∝ d`, we have `λ̄ ∝ 1/d` while `A ∝ d²`, so buckling capacity is
**strictly increasing in `d`**: the sizing root is unique and bisection is
unconditionally safe.

---

## TEST FIXTURE — Guide Worked Example 6.7

*(Corrected 2026-08-08. This was labeled Example 6.2 until it was checked
against `references/9780727741721.pdf`. Example **6.7**, "buckling resistance
of a compression member", pp. 61–63, is the CHS. The guide's Example 6.2 is
"cross-section resistance in compression" for a 254 × 254 × 73 UKC — an
I-section, `A = 9310 mm²`, `N_c,Rd = 3305 kN` — and is a separate fixture.)*

Full validation of the whole chain. Use as `tests/test_worked_example_chs.py`.
Assert against the closed-form column at 0.5% and the guide column at 1%: the
guide rounds intermediates to 2 s.f., which puts `Φ` 0.57% from its own closed
form.

**Input:** hot-finished CHS 244.5 × 10, S355, pinned–pinned, `L_cr = 4000 mm`,
`γ_M0 = γ_M1 = 1.0`, `N_Ed = 2110 kN` compression. (The guide's prose says
"hot-rolled"; Table 6.5's own row reads "hot finished", which is the EN 10210
term.)

| Quantity | Guide | Closed form (verified) |
|---|---|---|
| `f_y` | 355 N/mm² | — |
| `A` | 7370 mm² | 7367.03 |
| `I` | 50 730 000 mm⁴ | 50 731 473 |
| `W_el` | 415 000 mm³ | 414 981 |
| `W_pl` | 550 000 mm³ | 550 236 |
| `ε` | 0.81 | 0.8136 |
| `d/t` | 24.5 | 24.45 |
| Class-1 limit `50ε²` | 40.7 † | **33.10** |
| Class | 1 | 1 |
| `N_c,Rd` (6.10) | 2616 kN | 2615.3 |
| `N_cr` | 6571 kN | 6571.7 |
| `λ̄` (6.50) | 0.63 | 0.6308 |
| curve / `α` | a / 0.21 | — |
| `Φ` | 0.74 | 0.7442 |
| `χ` (6.49) | 0.88 | 0.8779 |
| `N_b,Rd` (6.47) | 2297 kN | 2296.0 |

`A`, `I`, `W_el` and `W_pl` are tabulated in Figure 6.21; the rest are worked
through in the text.

> † **Errata, 2026-08-08 — the error is the guide's, not the transcription's.**
> Page 62 prints, verbatim, `Limit for Class 1 section = 50ε² = 40.7`. But
> 40.7 is `50ε` (`50 × 0.8136 = 40.68`); `50ε²` is `50 × 0.662 = 33.10`. The
> **formula is right and the arithmetic is wrong**: Table 5.2 sheet 3 on p. 41
> prints `d/t ≤ 50ε²` and the `ε²` row of that same table gives 0.66 for S355.
> The guide wrote the squared limit and evaluated the unsquared one.
>
> The verdict is unaffected — `d/t = 24.45` is Class 1 under either reading —
> so this changes nothing structural, only what a test may assert. Implement
> and assert `50ε² = 33.10`.

Verdict in the guide: both `N_c,Rd` and `N_b,Rd` exceed `N_Ed = 2110 kN`;
section acceptable.

---

## SECOND FIXTURE — Guide Worked Examples 6.1 and 6.2

Neither section is a CHS, which is why they are useful: they exercise §6.2
through its area interface with no CHS geometry involved. Use as
`tests/test_worked_examples_cross_section.py`, tolerance 0.1%.

**Example 6.1** (p. 38), tension. A 200 × 25 mm flat bar tie in S275, lap
spliced with six staggered M20 bolts. At 25 mm thickness EN 10025-2 gives
`f_y = 265` and `f_u = 430`. UK NA: `γ_M0 = 1.00`, `γ_M2 = 1.10`.

| Quantity | Guide |
|---|---|
| `A` | 5000 mm² |
| `A_net` | 4406 mm² |
| `N_pl,Rd` (6.6) | 1325 kN |
| `N_u,Rd` (6.7) | 1550 kN |
| `N_t,Rd` | 1325 kN |

**Example 6.2** (p. 40), compression. A 254 × 254 × 73 UKC in S355, short
enough that `λ̄ ≤ 0.2` and §6.2.4 governs alone. `A = 9310 mm²`, `γ_M0 = 1.00`.

| Quantity | Guide |
|---|---|
| `N_c,Rd` (6.10) | 3305 kN |

Its classification uses Table 5.2 sheets 1 and 2 (outstand flange, internal
web) and is out of scope here — this package classifies tubular sections only.

---

## Additional exact fixtures (no source needed)

- **`λ̄ = 0.2` ⟹ `χ = 1.000000` exactly, independent of `α`.**
  `Φ = 0.5[1 + 0 + 0.04] = 0.52`; `√(0.52² − 0.2²) = 0.48`; `χ = 1/1.00`.
  Assert to 1e-15. This pins the `−0.2` offset, the likeliest transcription error.
- `χ ≤ 1` and `χ ≤ 1/λ̄²` for all `λ̄` (never exceeds Euler).
- `χ → 1/λ̄²` as `λ̄ → ∞`; `χ` strictly decreasing in `λ̄`.
- Curve ordering: `χ(a0) > χ(a) > χ(b) > χ(c) > χ(d)` at every `λ̄`.
- `N_cr` vs closed-form Euler.

---

## PUBLISHED BUCKLING-CURVE POINTS — the full set the guide offers

Eq. 6.49 is the one place where property tests are not enough: they fix the
shape of the curve but no absolute value on it. These are every published
`(λ̄, α) → χ` point in the book. All eight are asserted in
`tests/test_worked_examples_buckling.py`; none of the members is a CHS, which
costs nothing because the clause layer takes `A` and `I`, not a diameter.

| Example | Section | Curve | `α` | `λ̄` | `Φ` | `χ` |
|---|---|---|---|---|---|---|
| 6.7 | 244.5 × 10 CHS | a | 0.21 | 0.6308 | 0.7442 | 0.8779 |
| 6.9 | 200 × 100 × 16 RHS, major | a | 0.21 | 1.4155 | 1.6295 | 0.4104 |
| 6.9 | 200 × 100 × 16 RHS, minor | a | 0.21 | 0.8449 | 0.9247 | 0.7690 |
| 6.10 | 305 × 305 × 240 UKC, major | b | 0.34 | 0.2295 | 0.5314 | 0.9895 |
| 6.10 | 305 × 305 × 240 UKC, minor | c | 0.49 | 0.5829 | 0.7637 | 0.7955 |
| 13.3 | 100 × 50 × 3 channel | c | 0.49 | 1.1612 | 1.4097 | 0.4527 |
| 6.8 | 762 × 267 × 173 UKB, seg. BC † | b | 0.34 | 0.54 | 0.7036 | 0.8661 |
| 6.8 | 762 × 267 × 173 UKB, seg. CD † | b | 0.34 | 0.62 | 0.7636 | 0.8269 |

Curves a, b and c; `λ̄` from 0.23 to 1.42. Curve `a0` and curve `d` are never
exercised by the guide, and neither is any `λ̄` above 1.42.

† These two are **lateral-torsional**, Eq. 6.56. This package does not
implement 6.3.2 and does not need to — a CHS is closed and doubly symmetric, so
`χ_LT = 1`. But the general case of Eq. 6.56 is the *same function* of `λ̄` and
`α` as Eq. 6.49, so its printed values are two more points on curve b at a
mid-range slenderness the flexural examples miss. Slenderness is fed in as
printed, since deriving it needs an elastic critical moment we do not compute.

Also usable, and asserted: seven `N_cr` points spanning 127 kN to 153 943 kN —
four orders of magnitude on `π²EI/L²_cr` — plus four more `λ̄` points from
Eq. 6.50 and three more `N_c,Rd` points from Eq. 6.10.

**The Designers' Guide has no multi-member benchmark.** Every one of its worked
examples is a single cross-section or a single isolated member.

**The ECCS manual does — and they are good.** Corrected 2026-08-08 after the
second book was audited. It carries 27 worked examples, including two complete
building designs, and publishes member forces rather than only resistances:

| Example | pp. | Structure | What it publishes |
|---|---|---|---|
| **Design Example 2** | 407–430 | 47 m single-span portal frame, HEA 550 / IPE 600, S355 | `M` and `N` at **11 named sections × 4 load combinations × 4 analysis types** (1st/2nd order, elastic and elastic-plastic), plastic-hinge history, `α_cr` tables, full LTB segment table. **The best frame fixture in either book.** |
| Design Example 1 | 317–342 | 8-storey braced 3D building, Cardington | wind derivation, imperfections, 14 combinations, `α_cr` per combination, 2nd-order comparison, utilization tables. Member forces only for column E1 and beam E1–E4. |
| 2.4 / 4.1 | 93–107, 278–288 | two-storey plane frame | `M`/`N` at 14 sections, no-sway + sway decomposition, storey `α_cr`. Tabulated to 0.1 kNm and **verified exact** to the printed digits. |
| 2.1, 2.2, 2.3 | 52–74 | plane and 3D frames, joint models | `M`, `V`, `N`, `δ` at 6–14 sections per model |
| 3.3 / 3.10 | 130–134, 191–197 | 15 m Warren truss | every bar force, then the buckling check of four member types |
| 3.5, 3.12 | 145–150, 219–223 | continuous beams | shear and moment diagrams, redistribution |

These validate T1 and T2, which the Designers' Guide cannot. See
`CHANGELOG.md` for how they rank against the Ziemian frame set.

---

## ERRATA IN THE GUIDE — read this before trusting a printed number

Audited 2026-08-08 against `references/9780727741721.pdf`. Every worked example
in the book was recomputed. **The guide contains at least eleven printed
numbers that are wrong**, plus a systematic display habit that makes another
fifteen substitutions fail to evaluate to their own answers.

This is not pedantry. It is the project's thesis in miniature: a design code
and its authoritative commentary are *normative text*, not executable
artifacts. Nothing checks their arithmetic. Cite this section in the writeup.

### The pattern

The guide computes with unrounded intermediates and prints rounded ones. So a
line like `72 × 0.92/1.0 = 66.6` is literally false — it evaluates to 66.24 —
while 66.6 is the correct answer from `ε = 0.9244`. **The printed *answers* are
almost always right; the printed *substitutions* are usually not.** Consequence
for fixtures: never assert a guide value tighter than about 1%, and never feed
a printed intermediate back in.

Separately, a handful of values are stale carry-overs from the 1st edition, or
simply wrong.

### Hard errors — a fixture built on these will encode a mistake

| # | Page | Example | Printed | Correct | Effect |
|---|---|---|---|---|---|
| 1 | 62 | 6.7 | `Limit for Class 1 = 50ε² = 40.7` | **33.10**. 40.7 is `50ε` | verdict unchanged, Class 1 either way |
| 2 | 42 | 6.3 | Fig. 6.9 `W_el,y = 2 536 249 mm³` | **2 124 800 mm³** | the printed value exceeds the *plastic* modulus 2 352 736, so it is physically impossible |
| 3 | 50 | 6.5 | `ρ = (2 × 525/689.2 − 1)² = 0.27`, `M_y,V,Rd = 386.8 kN m` | the example's own `V_pl,Rd = 664.3` gives **ρ = 0.34**, `M_y,V,Rd = 380.9 kN m` | 24% error in ρ; verdict unchanged |
| 4 | 89 | 6.10 | `0.25 N_pl,Rd = 0.25 × 8415 = 2104 kN` | `N_pl,Rd = 8109 kN` ⟹ **2027 kN**. 8415 uses `f_y = 275` where the example established **265** | verdict unchanged |
| 5 | 93 | 6.10 | Annex A, Eq. 6.62: `0.53 + 0.16 + 0.15 = 0.85` | third term with `k_zz = 1.33` is **0.28**; true total **0.98** | the stated addends also sum to 0.84, not 0.85 |
| 6 | 85 | 6.9 | Eq. 6.61: `0.07 + 0.84 = 0.92` | second term is **0.85** (total 0.92 is right) | addends do not sum |
| 7 | 85 | 6.9 | Annex B, Eq. 6.62: `0.04 + 0.80 = 0.83` | **0.84** | addends do not sum |
| 8 | 92 | 6.10 | `k_yz = 0.6 k_zz = 0.6 × 0.72 = 0.47` | `k_zz = 0.78`; `0.6 × 0.78 = 0.47` | answer right, multiplicand stale |
| 9 | 92 | 6.10 | `k_zz` limit uses `0.79 × 8415` | should be `χ_z N_Rk = 0.80 × 8109` | printed 1.04 matches neither (1.03 / 1.05) |
| 10 | 75 | 6.8 | `ψ = 0/1362` | the moment at C is **1327 kN m**; 1362 appears nowhere in the example | ψ = 0 regardless |
| 11 | 86, 87 | 6.10 | Fig. 6.30 labels `W_el,z` **twice** (1 276 000 and 1 951 000); p. 87 heads the minor-axis moment `M_y,Ed` | second is **W_pl,z**; moment is **M_z,Ed** | label errors, values used correctly |

Items 1, 2, 3, 4, 5, 6, 7, 8 and 9 were recomputed independently and confirmed.

### Substitutions that do not evaluate to their printed answer

Roughly fifteen instances; the answers are the accurate ones. The largest gaps:

| Page | Example | Printed line | Evaluates to | Book's answer |
|---|---|---|---|---|
| 54 | 6.6 | `591.0 × (1 − 0.42)/(1 − 0.5 × 0.40)` | 428.5 kN m | 425.3 kN m |
| 89 | 6.10 | `1125 × (1 − 0.42)/(1 − 0.5 × 0.22)` | 733.1 kN m | 726.2 kN m |
| 91 | 6.10 | `0.80 × 30 600 × 265/1.0` | 6487 kN | 6450 kN |
| 74 | 6.8 | `0.87 × 6198 × 10³ × 265/1.0` | 1428.9 kN m | 1424 kN m |
| 29 | 5.1 | `456ε/(13α − 1)` with `α = 0.70` | 52.04 | 52.33 (needs α = 0.6966) |
| 42 | 6.3 | `600.0 − 16.0 − 6.0 − 2 × 20 × 0.92 × 6.0` | 357.2 mm | 356.1 mm |
| 148 | 13.3 | `0.45 × 549 × 280/1.0` | 69.2 kN | 69.2 kN, but unrounded χ gives 69.6 |

Also: Example 6.9 writes `N_Rk` as 2946.5, 2946 and 2947 in three consecutive
steps; Example 6.5 truncates `412.775 × 10⁶` to `412`; Example 6.8 truncates
`h/b = 2.8579` to 2.85.

### Figure versus text contradiction

Example 6.7, p. 61: the text states `N_Ed = 2110 kN` and every check uses it,
but **Figure 6.20 is captioned `N_Ed = 1630 kN`**. Use 2110.

---

## Secondary sources

- SCI "Blue Book" — tabulated `N_b,Rd` for standard CHS vs buckling length
- steelconstruction.info — Member design page
- Blueprints (LGPL — **dev/tests only, never copy source**). Confirmed
  2026-08-08 by listing its EN 1993-1-1 chapter 6: it has Eqs. 6.6, 6.7 and
  6.10, in mm/N/MPa, with the formula classes subclassing `float`. It has
  **no §6.3 member buckling** (the chapter jumps `formula_6_45` →
  `formula_6_54`) and **no cross-section classification**. It also ships a
  259-entry CHS profile table, `blueprints.structural_sections.steel.`
  `standard_profiles.chs`, whose geometry is polygon-meshed rather than closed
  form — agreement is ~1e-5 at CHS 244.5 and ~2.5e-3 at CHS 21.3.

## Open items

0. ~~**THE WHOLE N+M BLOCK IS UNVERIFIED**~~ — **CLOSED 2026-08-08.** §6.2.9,
   §6.3.3 and Annex B were verified against both textbooks; every subsection above
   now carries ✅. The guide's N+M worked examples (6.9, 6.10) still carry errata
   #5–#9, so prefer the ECCS examples and property tests over their printed values.
0a. ~~**Is combining `M_y`, `M_z` into a resultant valid in 6.61?**~~ — **CLOSED.
   NO.** Exhaustive search of both books and every NCCI they cite found no
   sanction for it in 6.61/6.62, for any section. It *is* exact in Eq. 6.41
   (`α = β = 2`), at cross-section level only. Keep the moments separate in the
   member check.
0b. ~~**Does §6.2.9.1(4) offer a small-axial-force exemption for CHS?**~~ —
   **CLOSED. NO.** Both books enumerate the eligible section types and CHS is in
   none of them. Always compute `M_N,Rd`.
0c. ~~**Which Annex B table applies to CHS**~~ — **CLOSED. Table B.1**, three
   independent statements. But **Table B.1 has no CHS row**; we take the
   RHS-sections row, corroborated by Karamba. Interpretation, not citation.
0d. **Shear (§6.2.6) and torsion (§6.2.7) are still excluded**, but the tooling
   to audit the exclusion now exists. **Verified 2026-08-09** against the guide:
   `V_pl,Rd = A_v (f_y/√3)/γ_M0` is Eq. **6.18** under clause **6.2.6(2)**, and
   clause **6.2.6(3)** lists *"circular hollow section and tubes of uniform
   thickness: `A_v = 2A/π`"* (the π is lost in the text extraction; 2/π is the
   thin-ring shear form factor). The half-resistance threshold is **§6.2.8(2)**
   for bending and shear and **§6.2.10** once axial force is present too — the
   latter governs here. `area_shear` and `resistance_shear` are implemented; the ratio is
   swept in `experiments/05_class_ratio_sweep.py`, where it peaks at 0.12 on a
   plausible worst pairing. **Still verify post hoc on the converged design.**
0e. **Table B.3 row 3c** — the two books disagree on a sign
   (`0.90 ± 0.10 α_h(1 + 2ψ)`). Out of our path while loading stays nodal, since
   that cell requires span loading. Resolve against EN itself if that changes.
0f. ~~⚠️ **§5.2 and §6.3.4 are UNVERIFIED and are implemented anyway**~~ —
   **CLOSED 2026-08-09.** All of it checked against both books; the section above
   now carries ✅. **Every threshold held**: `α_cr ≥ 10` elastic and `≥ 15`
   plastic (§5.2.1(3)), and the `1/(1 − 1/α_cr)` amplifier with its `α_cr ≥ 3.0`
   floor (§5.2.2(5)). Nothing that P4 quotes moves — the arch's 0.713 and 1.734
   are read against 10 exactly as before. **Two equation numbers were wrong.**
   Eq. 5.1 is the threshold *pair*, not the definition `α_cr = F_cr/F_Ed`, which
   EN never numbers. And **6.64 could not be confirmed at all** — neither book
   prints EN's numbering for §6.3.4 — so it is now cited as §6.3.4(3), per the
   same policy as open item 1. Eq. 5.2 verified, and gained its clause,
   §5.2.1(4)B. Three things came out of the check that memory had not supplied:
   UK NA clause NA.2.9 moves the *plastic* limit (not the elastic one we use),
   §5.2.2(5)'s amplifier has no EN equation number to cite, and — the one that
   matters — **§6.3.4's `α_cr,op` explicitly excludes in-plane flexural
   buckling**, while the arch mode we measure is in-plane. The clause is
   therefore a citation for the algebra's form and not for our case; the identity
   carrying it needs no source and is tested as such.
1. Equation numbers 6.5, 6.9, 6.46 (the `≤ 1.0` utilization checks) — inferred,
   not confirmed. Low risk: cite the clause without the equation number. The
   guide reproduces neither; only the standard itself would settle them.
2. ~~`γ_M0` / `γ_M1` recommended values~~ — **CLOSED 2026-08-08.** Both 1.00,
   from UK NA clause NA.2.15. See Partial factors above.
3. ~~Cold-formed (curve `c`) and S460 (curve `a0`) selection~~ — **CLOSED
   2026-08-08** against Table 6.5. See Table 6.2 above.
