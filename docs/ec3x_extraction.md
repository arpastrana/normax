# Extracting `ec3x` — `normax/ec3/` as a library of its own

**Promoted from `ROADMAP.md:1276`, which filed it as after the deadline, and
reconciled against the code as it stands after the P5d and P8 restructurings.**
The sketch there records the shape correctly and gets one prescription wrong; both
are carried over below, the correction with its reason.

**Decided: this happens now, before Aug 31, not after.** What it buys before the
deadline is the claim itself. A submission that says a building code is a
normative text rather than a solver is stronger when the code check is a library
somebody could install without ever form-finding anything, and weaker when it is a
subdirectory of the demo that consumes it.

---

## 1. What lifts out, and why it lifts cleanly

**The seam is one-way and already holds.** Every one of the imports inside
`normax/ec3/` resolves to `jax`, `jaxtyping`, the standard library, or
`normax.ec3.*` — 30 internal import lines, and no line reaching any other part of
this package. There is nothing to untangle: the subtree is already a library that
happens to live in a subdirectory.

**Its dependency set is two packages against normax's six.** `ec3x` needs `jax`
and `jaxtyping`. normax needs those plus `pyyaml`, `scipy`, `tesseract-core` and
`tesseract-jax`, and reaches `jax-fdm`, `smax` and `matplotlib` through its
`pipeline` group. Someone who wants a differentiable EN 1993-1-1 and nothing else
currently pays for a form-finder, a frame solver, a YAML parser and a Tesseract
runtime.

**Most of the test suite goes with it.** Counted, not estimated:

    moving to ec3x    1592 tests across 15 files
    staying            298 tests across 10 files

That is 84% of the 1890 the suite runs. `test_oracle_blueprints.py` alone
is 473 of them, and it is the only file in the repo that imports `blue-prints`, so
the LGPL dev-only oracle leaves normax's dev group entirely and becomes `ec3x`'s
problem — which is where it always belonged, being a numerical oracle for clauses.

**What the library claims on its own.** A clean-room, differentiable, *inverted*
EN 1993-1-1 in JAX: not "does this section pass" but "what section passes
exactly", with a hand-derived tangent at the root of the residual, `float64`
throughout, and every clause carrying its number. That is useful to anyone sizing
steel under a gradient, and none of it is about gridshells.

---

## 2. What stays, and the three things that make the seam honest

### The contract stops speaking one standard's clause vocabulary

**The ROADMAP diagnoses this correctly and prescribes something that cannot
work.** Its words: "`MemberActions` is genuinely EN 1993-1-1's input, but
`MemberSizes.actions` is typed on it, so a pipeline using SkyCiv alone would still
import the EC3 library for a container" — filed under the heading "Two containers
move up rather than out".

`MemberActions` cannot move up. `normax/ec3/sizing.py` takes it in eight annotated
parameters and in both the primal and the tangent tuple of the `custom_jvp` at
`sizing.py:428`, with three more parameters in `resistance.py` and one in
`interaction.py`. A normax-owned `MemberActions` makes `ec3x` import `normax`,
which is the dependency backwards. **It goes out with the library.**

**Nor may it be duplicated above the seam.** A five-field normax container mirroring
it would relocate the problem rather than solve it: the two factor fields are
Table B.3's equivalent uniform moment factors, and no fixed field list is neutral
over standards, because another standard reduces an analysis to different
quantities. Writing EN 1993-1-1's field names into the sizer-agnostic contract is
the thing being fixed, and doing it one module higher is still doing it.

**So the record leaves the contract.**

    MemberSizes(sections, utilization)

and `AbstractMemberSizer.utilization` and `Ec3Sizer.governing` take the
`MemberForces` a design already carries, each sizer applying its own reduction
internally. `coerce_member_actions` goes down into `normax/sizing/ec3.py`, since applying
Table B.3 is clause work. The reduction is elementwise and stateless, so
re-deriving it inside the `vmap` those methods already run is exact and nearly
free.

**Three things in the tree already say this is the right way round.**

- `AbstractFrameAnalyzer.__call__` takes plain `diameters` rather than a section
  container, for exactly this reason — hand a block the raw thing and let it build
  what it needs — and `MemberForces` lives in `normax.analysis` rather than in
  `smax`, which has its own `ElementForces`.
- The only production reader of the record outside a sizer is `frame_stability` at
  `normax/analysis/smax.py:654`, taking `design.sizes.actions.axial_force[load_case]`
  when `design.forces.axial_force[load_case]` is the same array from the stage that
  produced it. The analysis is reading its own output through the check's copy of
  it.
- `coerce_member_actions` sits in the contract module today while applying a clause, one
  import away from the backend that owns it.

**What stays is `Tube`, and the distinction is shape against clause.**
`MemberSizes.sections` still names an `ec3x` type after this change, and that is
deferred rather than fixed here. A tube is a *shape*: this project designs tubes,
and a second sizer reading the same members hands back the same tubes. Its trigger
is a second cross-section, and the ROADMAP already says what to write then — a
shape-agnostic property bundle a catalogue produces, area and second moment and
radius of gyration and both moduli. A Table B.3 factor has no trigger to wait for.
It is already wrong for a second standard today, which is why one is fixed now and
the other waits.

**What the fix costs.** Ten call sites: `normax/design.py:383`,
`normax/analysis/smax.py:654`, `normax/tesseract.py`, experiments 09, 10 and 11,
and `tests/test_pipeline.py`, `tests/test_design.py`,
`tests/test_tesseract_parity.py`. The `field_by_field` parity walk loses
`moment_factor_major` and `moment_factor_minor`, which reach it only through
`sizes.actions`; `TOLERANCE_MOMENT` keeps its purpose through `forces.moment_*`,
and the two factors get compared explicitly against the boundary outputs that
still publish them. **The ec3 Tesseract's `OutputSchema` does not change** — that
boundary is the EC3 block's own, so EC3 vocabulary belongs on it — and
`Ec3TesseractSizer` keeps rebuilding a `MemberActions` internally and recomputing
utilization locally rather than reading the returned one, so nothing it hands back
moves.

### `diameter_envelope` moves up, and the ROADMAP does not mention it

It sits at `normax/ec3/sizing.py:680`, and it is a soft maximum over the load-case
axis. No clause, no standard, nothing that EN 1993-1-1 has an opinion about. Its
only production caller is `normax/design.py:378`, beside `design_envelope`, which
is where it goes; `validation/load_case_envelope.py:144` is the other call
site.

The repo's own docstrings already argue the case. `normax/sizing.py:88`:
reconciling the load cases "is smoothing rather than a clause and belongs above a
block that implements a standard". A function inside the standard's subtree doing
the smoothing the standard is explicitly not responsible for is misfiled, and the
extraction is what makes it visible.

### `stability.py` goes out; `frame_stability` stays

`normax/ec3/stability.py` implements §5.2.1 and §6.3.4 — global elastic
amplification and the frame-level slenderness check — and that is clause work, so
it goes out with the library. `frame_stability` in `normax/analysis/smax.py` stays
and imports it, because reading `α_cr` off a solved eigenproblem is analysis work
that happens to consult a clause. The stage that computes the modes keeps the
function that interprets them.

`normax/analysis/smax.py` therefore keeps importing `ec3x`, eleven import lines of
it, and that is not a wart: a backend is allowed to know which standard it is
reporting against. What the *contract* modules may not do is name a clause
product.

**Reversed 2026-08-15, on instruction.** `frame_stability` and its `Stability`
record were deleted from normax outright rather than moved, and the deletion
was then widened to the whole buckling surface: `buckling_modes`, the
`Buckling` container and `figure_modes` are gone too, and no experiment
computes a critical load factor. The clauses live on in `ec3x/stability.py`,
consumed by nothing here; buckling and frame-stability checks are future work,
stated as such in the manuscript. With that deletion and the neutralization of
`TesseractSizer`'s surface, `normax/sizing/ec3.py` is the only module in the
package importing `ec3x`.

---

## 3. Phases

### 0. Stand on a green baseline, and know which smax it is

**Recorded 2026-08-15: 1890 pass in 31 s, with `../smax` on `main` at `41601c4`.**
That is the number every later phase is judged against.

**The extraction is run against smax `main`, and not against a feature branch.**
`../smax` was sitting on `phase9a-flat-columns`, whose interim tree has already
deleted `smax/compilation.py`, so `normax/analysis/smax.py:68` cannot import
`CompiledStructure` and six files error at collection: `test_pipeline`,
`test_design`, `test_analysis_prepared`, `test_equilibrium_consistency`,
`test_tesseract_parity`, `test_backend_opensees` — 298 tests, every one of them on
the staying side of the split. Checking `main` out fixes it, and that is the whole
remedy; nothing in this document waits on the flat-column work.

**Why it matters beyond convenience.** The gate below is "no number moves", and it
can only be read if the baseline is the same on both sides of a phase. Sharing the
sibling repo's in-flight branch with the extraction would put two moving parts in
one measurement, and a red import from the wrong branch reads exactly like a broken
rename. When `phase9a-flat-columns` lands, re-record the baseline before continuing
rather than assuming 1890 survived it.

### 1. `normax/sizing.py` becomes `normax/sizing/`

`__init__.py` takes `MemberSizes` and `AbstractMemberSizer`; `ec3.py` takes
`Ec3Sizer` and `coerce_member_actions`. **No re-export from `__init__.py`** —
`normax/analysis/__init__.py` imports neither of its backends and every call site
reads `from normax.analysis.smax import SmaxAnalyzer`, so the check mirrors the
analysis rather than inventing a second convention. Eight call sites:
`normax/design.py`, `normax/tesseract.py`, experiments 03, 09, 10 and 101,
`tests/test_pipeline.py`, `tests/test_tesseract_parity.py`, `tests/test_design.py`.

Structural only. No number moves in this phase.

### 2. Take the actions record out of the contract

`MemberSizes` becomes `(sections, utilization)`. `AbstractMemberSizer.utilization`
and `Ec3Sizer.governing` take `MemberForces` and call `coerce_member_actions` inside the
`vmap` they already run, one line each. `frame_stability` reads
`design.forces.axial_force[load_case]`. `design_envelope` rebuilds a two-field
container. The parity walk gains an explicit comparison of the two moment factors
against the boundary dict. `Ec3TesseractSizer` keeps its internal `MemberActions`
and its local utilization recomputation untouched.

**This is the one phase that could legitimately move a number, so it is gated on
its own and done before anything relocates.** A tolerance failure here must not be
confusable with a failure of the move.

### 3. Move `diameter_envelope` up

Into `normax/design.py`, beside `design_envelope`. Its assertions split out of
`tests/test_sizing.py` and stay in normax, before that file moves in phase 5.
`validation/load_case_envelope.py` changes its import.

### 4. Create `~/code/libraries/ec3x`

Flat layout, mirroring the `simonw/python-lib` template normax was bootstrapped
from: `pyproject.toml` naming `ec3x`, dependencies `jax` and `jaxtyping` only,
`requires-python` matching normax's `>=3.12,<3.15`; Apache-2.0 `LICENSE`; the same
ruff configuration and the same pinned pre-commit hooks, so the two repos cannot
drift into disagreeing about line length or import style;
`.github/workflows/test.yml`; and `tests/conftest.py` reduced to the JAX
compilation-cache block, since none of the pipeline or OpenSees collect-ignore
guards have anything to guard.

**`ec3x/__init__.py` must set `jax_enable_x64`, and this is the one silent failure
mode in the whole extraction.** `normax/__init__.py` is the only place in this
package that sets it, and `normax/ec3/` inherits it purely by being a subpackage.
Standalone and unset, every clause runs in float32, nothing raises, and every
measured tolerance in either repo becomes meaningless. The three
`tesseracts/*/tesseract_api.py` already set it themselves for exactly this reason —
they are imported outside the package — and they are the precedent to copy.

### 5. Move the subtree

`normax/ec3/*.py` → `ec3x/*.py`, rewriting the 30 internal `normax.ec3.X` imports
to `ec3x.X`; the 15 test files with theirs. `blue-prints` joins `ec3x`'s dev group
and leaves normax's, along with the note that it is LGPL, dev-only, a test oracle,
and never to be read while writing anything but an assertion.

### 6. Rewire normax

`ec3x` into `dependencies` plus
`[tool.uv.sources] ec3x = { path = "../ec3x", editable = true }`.

**Only the path source copies the `smax` precedent, and the placement deliberately
does not.** `smax` sits in the `pipeline` group because the analysis backend is
optional — `normax/analysis/__init__.py` imports no backend, so the contract reads
without one. `ec3x` cannot be optional: `normax/design.py` imports `Tube`, and
`normax/sizing/__init__.py` will still import it after phase 2. It is a hard
dependency of the core package.

Import lines to rewrite: `normax/analysis/smax.py` (11),
`normax/sizing/ec3.py` (8), `normax/tesseract.py` (4), `normax/design.py` (2),
`normax/analysis/opensees.py` (1), the three `tesseracts/*/tesseract_api.py`, 11
experiments and the 7 remaining test files. Prose mentions too, which a mechanical
sweep will miss: `normax/units.py`, `normax/optimization.py`,
`normax/analysis/smax.py`, `normax/tesseract.py`, `normax/sizing/*` docstrings and
`docs/clauses.md`.

**The requirements files.** `tesseracts/ec3_check/tesseract_requirements.txt`
currently carries the normax wheel alone, and after this it needs the `ec3x` wheel
alone — the Tesseract stops depending on normax entirely, which is the cleanest
single piece of evidence that the seam is real. The two analysis backends need both
and gain a line. **Open, and to be settled when the phase is executed:** the file
reaches `../../dist/normax-0.1-py3-none-any.whl`, two levels up but still inside
this repo, and a sibling repo's `dist/` is outside whatever context `tesseract
build` copies. Either `ec3x`'s wheel is built into normax's `dist/`, or the
requirement names a published version. Do not assume the relative path works
because the existing one does.

---

## 4. The gate

Phase 2 changes a contract and phases 3 to 6 are a rename, so **nothing may
move**, and that is the whole test.

**Phase 2, gated on its own before anything relocates:**

- Every recorded tolerance holds at its recorded value — `TOLERANCE_PARITY` 1e-14,
  `TOLERANCE_SIZE` 1e-13, `TOLERANCE_MOMENT` 1e-11, `TOLERANCE_DERIVATIVE` 5e-12,
  `TOLERANCE_UTILIZATION` 1e-9. **A loosened tolerance is a failed phase, not a new
  measurement.**
- The parity walk's field list is one carrier shorter, and the two moment factors
  are compared explicitly against the boundary outputs instead.

**The move:**

- `import ec3x` in an environment without normax, then assert that a `jnp` array is
  `float64`. The one failure mode that would silently change every number in both
  repos.
- `ec3x`: 1592 tests green.
- normax: 298 green, which is 1890 less what left, and against smax `main`.
- `experiments/101_api.py` reprints 0.138951969 t, 16.114 % saved, +0.24867 %, and
  a utilization of 1.000000000000, bit for bit.
- `experiments/09` reprints the x3.26 buckling-length penalty and `α_cr` 0.1291;
  `10` holds parity at `TOLERANCE_PARITY` and `TOLERANCE_SIZE`; `11` the straight-beam
  benchmark; `04` the backend agreement.
- `tesseract build` for `ec3_check` against the `ec3x` wheel, then
  `tests/test_tesseract_parity.py`.

---

## 5. What this does not buy

**`normax/sizing/skyciv.py` is the point of the exercise, and this document does
not reach it.** A commercial member sizer reached over HTTP, behind the same
`AbstractMemberSizer`, so which standard a design is checked against becomes an
argument the way the solver already is — that is the strongest form the thesis can
take, an in-process check and a remote proprietary one composed identically,
differing only in who wrote the clauses and whether they can be read at all. The
extraction makes it writable and writes none of it.

No schema changes and no gradient changes: the adjoints are inside the subtree that
moves, and they move with it.

**The one thing here that a second sizer needs rather than merely benefits from is
phase 2.** A `MemberSizes` naming Table B.3's design moment and equivalent uniform
moment factor is a container a non-EC3 sizer cannot fill. Everything else in this
document is tidying that a second sizer would survive without.

---

## 6. Known frictions

**Eight of the 15 moving test files cite `docs/clauses.md`, eleven mentions, and
that document stays in normax.** It is the verified specification — the record of
which clause was read against which printed text, and what disagreed — and it is
also the writeup's evidence, so it does not follow the code out. The citations
become repo-qualified, and `ec3x/README.md` says where the verification record
lives. Worth revisiting once `ec3x` has a life of its own: a library whose
correctness argument lives in another repository is an awkward thing to hand
someone.

> **Revisited and reversed, 2026-08-15, same day.** Rafael's call: the record
> transfers to `ec3x/docs/clauses.md`, with `references/` moved wholesale
> beside it (gitignored there as it was here) so its check-the-printed-page
> instruction keeps working where it lives; normax keeps no copy. The eleven
> citations localized again; normax's code and ROADMAP point across. The
> writeup quotes the record's findings, and can quote them from the sibling
> repo.

**`ec3x` on PyPI is unchecked.** It does not matter while `[tool.uv.sources]`
carries a path, and nothing gets published before Aug 31.

**`equinox` is undeclared in normax's dependencies and arrives transitively.**
`AbstractMemberSizer` and `AbstractFrameAnalyzer` are `eqx.Module`s. Noted in
passing; not this document's job, and not `ec3x`'s problem either way, since
nothing in the moving subtree imports it.
