# Designing for shear

**Decided 2026-08-19: not landing for the hackathon.** The measurement in "What
this buys" below is why — every converged design passes 6.2.10 with room, so
designing for shear would move no diameter, and the cost falls almost entirely on
the hand-written NumPy duplicate of the Blueprints wrapper. What shipped instead
is the audit (`experiments/20_shear_audit.py`), the tolerance check that now rides
in every truss and arch run, and the exclusion stated in the README, in
`normax/sizing/blueprint.py` and in ec3x's `docs/clauses.md`.

**This file is the record of what building it would take**, kept so the decision
reads as a decision rather than an omission. It is accurate as of the audit below;
the two repos change together, in the order given, if a structure ever crosses the
threshold.

## Why

`normax/sizing/blueprint.py` imports two Blueprints formulas — `Form6Dot10NcRdClass1And2And3`
and `Form6Dot14MCRdClass3` (`:42-47`) — and justifies that scope like this (`:24-26`):

> The check is cross-section resistance alone, because that is all Blueprints
> implements — it has no §6.3 member buckling and no classification.

The claim about §6.3 is true. The sentence around it is not. Shear (§6.2.6),
torsion (§6.2.7) and bending-with-shear (§6.2.8) *are* cross-section resistance,
and Blueprints ships all three, including `6.18subg`'s `A_v` for a circular hollow
section and `6.28`'s `V_pl,T,Rd` — the two entries that would matter here. So the
docstring attributes an authored scope decision to a limitation that does not
exist. That is the defect to fix whether or not shear is ever checked.

What is *not* wrong, and was verified before writing this: the audit
infrastructure is in place and honest — **as uncommitted work in the tree, not as
committed state.** `normax/analysis/__init__.py`, both backends, three test files
and exp 05 all carry it, and this plan assumes it lands as written. `SecondaryForces` carries `shear_major`,
`shear_minor` and `torsion_moment` (`normax/analysis/__init__.py:74-102`); both
backends fill it and agree to `1e-11`; and three test files read it — the shear
equals the solver's own `vz` at `rtol=1e-15` and equals `ΔM/L` at `rtol=1e-10`
(`tests/test_equilibrium_consistency.py:290-304`), the backends agree
(`tests/test_backend_opensees.py:206-235`), and the crossed boundary's blindness
is pinned deliberately rather than left silent
(`tests/test_tesseract_parity.py:355`). ec3x's `docs/clauses.md` open item 0d
records the whole position, and corrects the old 0.12 bound to **0.059** by the
right mechanism (`V = 2M/L` antisymmetric, not a span-loaded `4M/L`).

The one place a reader of the submission would look has nothing: README's
`## Limitations` lists `∂d/∂q` and global stability and says nothing about shear
or torsion, and CLAUDE.md §3's scope decisions do not mention them either.

## Audit of what landed in ec3x, 2026-08-19

Two commits, `9520626` and `fa78d9c`. **Neither touches the API** —
`git diff 085450f..fa78d9c -- ec3x/` is empty, and the shear surface is still
`SHEAR_THRESHOLD`, `area_shear` and `resistance_shear`, with no caller inside the
package. Suite 1599 pass.

- `9520626` [Docs] corrects the Blueprints inventory and replaces the 0.12 with
  0.059 by the right mechanism. Sound, and it is the correction that prompted this
  plan.
- `fa78d9c` [Tests] oracles `area_shear` against Blueprints'
  `Form6Dot18SubGCircularHollowSection`, closing the invariant-2 gap clauses.md
  had recorded as open ("sits there unused"). Two tests, and the second is the one
  worth noting: it pins the *oracle* to `2/π`, the thin-ring form factor, so the
  pair cannot agree on a shared misreading. An oracle test that only compares two
  implementations proves nothing if both are wrong; this one cannot.

So the verification half of §1 below has landed and the design half has not. Every
insertion point named in §1 is still ahead.

**One container change to fold in, from normax's tree:** `SecondaryForces` is gone.
`MemberForces` now states all six components as direct fields — `shear_major`,
`shear_minor`, `torsion_moment` beside the axial force and the two moments — with
scalar `0.0` defaults, and `DESIGN_AXES = MemberForces(0, 0, 0, None, None, None)`.
Flatter, and it removes a container that existed only to hold three fields apart.

## What the standard actually asks for

Three separate things, and they must not be collapsed:

- **§6.2.6, Eq. 6.17** — a standalone check, `V_Ed / V_c,Rd ≤ 1.0`, with
  `V_pl,Rd = A_v (f_y/√3)/γ_M0` (Eq. 6.18) and `A_v = 2A/π` for a tube of uniform
  thickness (6.2.6(3)).
- **§6.2.8** — below `0.5 V_pl,Rd` the moment resistance is unreduced; above it,
  a reduced yield strength `(1−ρ) f_y` over the shear area, `ρ = (2V_Ed/V_pl,Rd − 1)²`.
- **§6.2.10** — the same threshold once axial force is present too, which is the
  clause that governs here.

Every number above must be read off the standard text in `../ec3x/references/`
during implementation and recorded in `clauses.md`, not taken from this file.

So the reading is `U = max(U_axial+bending, V_Ed/V_pl,Rd)`, with the bending term's
`f_y` reduced only above the threshold.

## The one real fork: where the shear demand comes from

The sizers receive `MemberForces` and a buckling length. Neither carries a shear
the check may read.

**(A) Add shear to T2's output schema.** Honest — the analysis computes it — but it
unfreezes the analysis Tesseract's schema, needs both backends extended (including
the OpenSees sensitivity path, `opensees.py:578-604`, which is not traced), and
inverts `test_the_boundary_does_not_carry_the_secondary_forces`.

**(B) Derive `V = ΔM/L` inside the check.** Exact under nodal loading, works
identically in process and crossed, no schema change. But `ΔM/L` needs the
**member** length, and the sizers are handed a **buckling** length. Those coincide
today only because `design.py:241` passes `shape.lengths` for both; per
[[buckling-length-policy]] `L_cr` is an input and its equality with the member
length is a braced-node assumption. Using it for shear would smuggle in a second,
unrelated assumption.

**(C) Derive it, and give the sizer the member length as its own argument.**
Recommended. `AbstractMemberSizer.__call__(forces, buckling_length, member_length)`
makes both lengths explicit, T2 stays frozen, the crossed path stops being blind
to shear, and the derivation follows the precedent already written down in
`ec3x/actions.py:44-49`: the reduction from two end moments to a design quantity
"belongs to the check", which is why `end_moments` lives in `ec3x.sizing`.

Under (C), `MemberForces.secondary` becomes the **oracle for the derived demand**
rather than dead weight: a test asserts derived-equals-analyzed, which is the
identity already pinned at `rtol=1e-10`.

## What this buys — measured, not bounded

`experiments/20_shear_audit.py` reads the analyzed shear off every converged
design in the repo: the arch at 103's simultaneous optimum, and both trusses at
each of the three answers 18 and 19 descend to. Worst `V_Ed/V_pl,Rd` over members
and load cases, Eq. 6.17 read once per shear component and taken at its worst:

| design | worst | median | under 0.5 by | worst member |
|---|---|---|---|---|
| vierendeel, sizing only | **0.3558** | 0.1264 | 1.4x | verticals |
| vierendeel, free heights | 0.1606 | 0.0037 | 3.1x | verticals |
| arch, 103 optimum | 0.1147 | 0.0257 | 4.4x | — |
| vierendeel, end to end | 0.0962 | 0.0012 | 5.2x | verticals |
| warren, sizing only | 0.0246 | 0.0072 | 20.3x | rising diagonals |
| warren, free heights | 0.0183 | 0.0015 | 27.3x | top chord |
| warren, end to end | 0.0108 | 0.0011 | 46.4x | top chord |

Three things follow, and the first invalidates a number this plan was written on.

**The 0.059 does not bound a frame.** The Vierendeel's sizing-only design reaches
six times it. That bound was taken over a demand mix on one 6 m member and bounds
that member; a frame sets its own shear from its topology.

**Topology decides it.** A Vierendeel has no diagonals, so transverse load crosses
each bay by frame action and lands in the verticals — every Vierendeel worst case
is a vertical. A Warren gives the same load to its diagonals axially and stays 20x
to 46x under. A factor of 15 between two trusses of the same span, depth and load.

**Optimizing lowers the ratio**, against the intuition that thinner members sit
closer to every limit. A funicular member carries less moment and the shear is that
moment's slope, so the demand falls faster than the resistance does: the Vierendeel
goes 0.3558 drawn to 0.0962 end to end while its mass falls 71.6%. The heaviest
design in the study is the one nearest the exemption's edge.

**So the mass answer is still nothing.** Eq. 6.17 needs 1.0 and the worst design
reads 0.3558; §6.2.8's reduction needs 0.5 and nothing reaches it. **Every diameter
would be bit-identical**, which remains the strongest assertion available and is now
measured rather than argued. A branch that cannot fire on the real structure still
needs a test that fires it — a synthetic short-member, large-antisymmetric-moment
fixture, in the manner of exp 14's silencing of buckling by `L → 0`.

**What the measurement does not license** is the sentence "shear is negligible for
CHS frames". It is negligible for *these* frames. 1.4x is not 20x, and a
Vierendeel that was shallower, longer-spanned or more heavily loaded would cross
0.5 and be governed in its verticals first. That belongs in the writeup as a
property of the study.

## Order of work

### 1. ec3x — the clauses

`docs/clauses.md` has **no `## §6.2.6` section and no checkbox for Eq. 6.18**;
shear appears only in the scope paragraph, the Blueprints inventory and open item
0d. Writing those sections is part of the work, not a by-product.

- `ec3x/actions.py:60-64` — `MemberActions` gains the shear demand. Every check
  receives the whole tuple, and `sizing.py:363-366` folds new fields into shape
  inference for free, so this propagates without touching call sites.
- `ec3x/resistance.py` — `area_shear:235` and `resistance_shear:259` already exist
  and Eq. 6.18 is already oracled; they have no caller in the package yet. Add the
  Eq. 6.17 ratio, and the §6.2.8 reduction at the two bending denominators
  (`:467` plastic, `:537` elastic). Guard the ratio at zero shear the way
  `moment_resultant:358-360` guards its sqrt, or a member carrying no shear
  poisons every gradient reaching it.
- `ec3x/sizing.py:322` — `diameter_bracket` inverts each necessary condition in
  closed form; a shear condition inverts as a `sqrt` (`A_v ∝ d²`) and becomes a
  third `jnp.maximum` term. Without it the bracket stops bracketing and the
  bisection returns a wrong root **silently** — value parity would not see it.
- `ec3x/sizing.py:24-26` — re-argue monotonicity in writing. A ratio `∝ 1/d²` is
  decreasing, and the `max` of decreasing functions is decreasing, so uniqueness
  survives; the §6.2.8 reduction also moves the right way, because `ρ` falls as
  `d` grows. The threshold is a C⁰ kink of the kind the repo already carries on
  `χ` and the interaction caps.
- `ec3x/sizing.py:75-79` and `:609-630` — a `LIMIT_SHEAR` code, if shear can
  govern.
- **`adjoint.py` and `_diameter_jvp` need no change.** `sizing.py:446-449` obtains
  both partials by differentiating the check itself and inverts only the implicit
  part, so "whichever branch of the standard governs is the branch that gets
  differentiated". `adjoint.py` is an independent axial-only oracle by design
  (`:15-31`); extending it to shear is optional and separable.
- Tests, per that repo's invariants: hand calc with the clause quoted, plus the
  Blueprints oracle — `6.17`, `6.18subg` (which currently "sits there unused"),
  `6.29rho`, `6.29`/`6.30` — plus `check_grads`, a monotonicity property test, and
  invariant 5's `U = 1 ± 1e-9`.

### 2. normax — the demand and the interface

- `AbstractMemberSizer.__call__` gains the member length (decision C). `design.py:241`
  passes `shape.lengths` for it; `buckling_length` keeps its own meaning.
- `normax/sizing/ec3.py:186-200` — `design_actions` grows the length argument and
  fills the new `MemberActions` field.
- **`DESIGN_AXES = MemberForces(0, 0, 0, None, None, None)`** broadcasts the three
  new components rather than slicing them, and the `None`s are not an oversight —
  they are *required* while the fields carry scalar `0.0` defaults, because `vmap`
  cannot map an axis over a scalar. Harmless today, since no check reads them under
  `vmap`. The moment one does, load case 0's shear reaches every case. Dropping the
  scalar defaults so the components are always stacked is the fix, and it is a
  change to the container rather than to the sizer.
- `tesseracts/ec3_check/` — a `member_length` input beside `buckling_length`, and
  the derived shear reported for audit.

### 3. normax — the Blueprints wrapper, both copies

The expensive half, and the reason this is one job rather than two.

- `normax/sizing/blueprint.py` — a fourth array threads through **eleven** sites:
  `_check_scalar:151`, `_solve_scalar:189`, `_solve_batch:240`, `_check_batch:268`,
  `_callback_solve:299`, `_callback_check:329`, `sized_diameter:362`,
  `_traced_partials:423`, `_sized_jvp:466`, `checked_utilization:491`,
  `_checked_jvp:531`. `HostFamily:68` gains a shear coefficient; `CheckPartials:401`
  gains a partial; both JVP rules gain a term.
- **The bracket is the trap.** `_solve_scalar:225-226` brackets by putting one term
  at exactly 1 and each term at ≤ ½. Because `A_v = (2/π)A ∝ d²`, a shear term read
  as a *linear sum* folds into the axial power and the closed-form cubic root
  survives. A faithful `max(·, V/V_pl)` with a threshold does **not** — the bracket
  and `test_the_cubic_root_agrees_with_the_bisection` both need re-deriving. Take
  fidelity and redo the bracket; do not buy the test back by summing terms the
  standard does not sum.
- `tesseracts/blueprint_check/tesseract_api.py` — the no-JAX duplicate. New
  differentiable schema fields, three partials per channel on `HandPartials:157`
  across all three clamp regimes (`free`, `bound`, `positive`), a fourth pull table
  on `AdjointState:611`, and a branch in both endpoints — the `else: raise` at
  `:752` is what makes a forgotten field fail loudly instead of silently.
- `normax/tesseract.py:739-756` and `:805-821` — `BlueprintClient` builds its
  payload literally, twice.
- Tests that break and must be extended, not deleted: `:284` (cubic root), `:642`
  (hardcodes the slope literal), `:673` (ec3-vs-blueprint agreement at `rtol=1e-7`
  — this is the one that forces both sizers to move together), `:703` (host
  coefficients pinned to the family).

### 4. Torsion — recommend declining, in writing

Do not implement 6.23–6.28. Torsion is identically zero for a planar frame under
in-plane nodal load, asserted rather than assumed in two test files, so the clauses
would add a branch no structure in the repo can reach. Record the exclusion with
its measured zero instead. If it is wanted anyway, it is its own item.

### 5. The documentation defects

- `normax/sizing/blueprint.py:24-26` — say that the sizer declines shear by
  choice, name the clauses declined and the Blueprints formulas that exist for
  them. This is worth doing on its own, before any of the above.
- README `## Limitations` — a shear and torsion entry, with the 0.059 and the
  measured torsional zero. §10 asks for exactly this.
- CLAUDE.md §3 — add shear to the scope decisions.
- ec3x `docs/clauses.md` — the new clause sections, and open item 0d closed.
- Both CHANGELOGs. The stale 0.12 in normax's stays as history; new entries carry
  0.059.

## Verification

1. **Inertness.** Every diameter and every mass bit-identical across the whole
   experiment suite, because the shear term is inactive at 0.059 and the reduction
   never triggers. Any movement is a bug, not a refinement.
2. **The branch fires.** The synthetic high-shear fixture drives `V_Ed/V_pl,Rd`
   past 0.5 and both sizers reduce; `check_grads` passes through the reduced branch
   and across the threshold.
3. **Derived equals analyzed.** The check's `ΔM/L` against
   `MemberForces.secondary.shear_major` at the existing `rtol=1e-10`, in process,
   and the crossed path now agreeing where it used to read zero.
4. **The two philosophies still meet.** `:673`'s ec3-vs-blueprint agreement holds
   at `rtol=1e-7` with shear in both.
5. **Invariant 5.** `U = 1 ± 1e-9` at every sized member, shear included.
6. Full suite in both repos; ruff via the pinned hook.

## Cost, stated plainly

Section 3 is the bulk: a fourth demand through eleven sites, mirrored by hand in a
NumPy duplicate that exists precisely so the two cannot drift, with four tests to
re-derive. It is the largest single item left before the Aug 31 deadline, and §8
still has the gridshell and the writeup. Section 5 alone — an hour — removes the
false claim, which is the part that would actually embarrass the submission.
