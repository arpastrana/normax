# Removing the oracles — `smax` and `ec3x` leave the package

**Snapshot taken first.** The tag `local-dev` marks `325d15f`, the last commit
at which both oracles are installed, wired and green: 395 tests, four smokes,
every agreement number in `CHANGELOG.md` reproducible. Nothing below has to
preserve a number that tag already holds.

---

## 1. Why this happens before Aug 31, not after

**It compacts the dependency set to what a stranger can install.** `smax` is
pinned as `git+file:///Users/arpj/code/libraries/smax` at a revision, and `vix`
the same way; `ec3x` is a relative path. Those three sources describe one
machine. They sit in the non-default `local-dev` group, so `uv sync` works
without them — but the suite does not, and that is the second reason.

**It ends a contradiction the submission cannot afford to be asked about.**
Both oracles are JAX-native: they differentiate end to end in process, with no
boundary to cross. Their presence in the tree invites exactly the question the
writeup least wants — *if a JAX frame solver and a JAX code check already exist
here, what is Tesseract for?* The honest answer is that they are oracles and
not the shipped stack, and that answer takes a paragraph. Deleting them makes
the question unaskable: after this, the only analysis is a crossed one and the
only check is a crossed one.

**It quadruples what CI actually runs.** Measured, not estimated. `conftest.py`
skips a test file whose imports reach an oracle, so today:

    CI runs      126 tests across  4 files
    CI skips     269 tests across 16 files

Every one of the three packages the skipped files really need — `blue-prints`,
`openseespy`, `pynitefea` — is already a main dependency. The skips exist for
`smax` and `ec3x` alone. Remove them and the guard apparatus in `conftest.py`
goes with them, and CI runs the suite instead of a third of it. For a judging
criterion that names reproducibility, that is the single largest change here.

---

## 2. What couples, counted

| Surface | Extent |
|---|---|
| package modules to delete | `analysis/smax.py` 353 lines, `sizing/ec3.py` 319, `visualization/{viewer,guard,unavailable}.py` 222 — **894 lines** |
| tests to re-point | **16 files, 269 tests** — 68% of the suite |
| `validation/` scripts | 7 of 8 touch an oracle; 4 study `ec3x` itself (**1469 lines**), 3 use one as a reference (**1931 lines**), `opensees_ddm.py` (860) is clean |
| examples | **clean of oracles already** — the four files touch only `view_design` |
| `pyproject.toml` | the whole `local-dev` group and its three `tool.uv.sources` entries |
| `README.md` | **no mention of either oracle** — nothing to retract |

Two facts worth stating because they were checked rather than assumed.
`normax/sections.py` names `ec3x` only in a docstring, so the section
arithmetic has no import to sever. And nothing under `tesseracts/` reaches
either oracle: the servers were always clean.

One correction to the record while we are here. The claim that importing the
package no longer imports an oracle holds only on an install *without* them —
on this machine `find_viewer()` returns true, so `import normax.visualization`
pulls `smax` and `vix` today. The guard hides the leak rather than closing it,
which is a further argument for deleting the viewer rather than re-pointing it.

---

## 3. The three roles an oracle plays, and what replaces each

The 269 tests are not one problem. They are three, and only the second is
interesting.

### Role 1 — fixture (143 tests, 6 files)

`test_design.py` (30), `test_nested.py` (57), `test_pipeline.py` (21),
`test_pipeline_tail.py` (12), `test_comparison.py` (17), `test_replay.py` (6)
build a pipeline out of `SmaxAnalyzer` and `Ec3Sizer` because a pipeline needs
*some* analyzer and *some* sizer. The oracle is scaffolding, never the
assertion.

**Replacement: the crossed clients, in process.**
`Tesseract.from_tesseract_api` already imports a stage's API module into the
test process — no containers, no network, no Docker. So
`TesseractAnalyzer(structure, catalog, backend="pynite")` and
`TesseractSizer(structure, catalog, check="blueprint")` substitute directly.
This is a straight swap of two constructor calls per file, and it makes every
one of those 143 tests exercise the boundary that the submission is about.

### Role 2 — reference (70 tests, 5 files)

`test_tesseract_parity.py` (17), `test_tesseract_sizer.py` (25),
`test_backend_opensees.py` (11), `test_backend_pynite.py` (9),
`test_second_sizer.py` (8) assert that a crossed answer equals an in-process
one. Delete the oracle and the assertion has no right-hand side. This is the
only real loss in the removal, and it needs a decision rather than a swap.

**Replacement, in this order of preference:**

1. **Finite differences, where the claim is about a derivative.** Invariant 1
   already requires `jax.test_util.check_grads` on every derivative rule, and a
   central difference of the crossed forward pass needs no second
   implementation. A hand-written adjoint verified against differences of its
   own primal is a *stronger* claim than one verified against a second
   library, because it cannot inherit a shared mistake.
2. **Frozen goldens, where the claim is about a value.** Record the oracle's
   numbers at tag `local-dev` into a module of literals with the tag named in
   its docstring, and assert against those. The test then says "this backend
   still returns what it returned when an independent JAX implementation
   agreed with it", which is the real content of the current assertion.
3. **Delete, where the test only proved the oracle worked.** Some of
   `test_backend_*.py` compares two references to each other rather than
   testing normax. Those go.

**Do not** stand a missing oracle in with zeros or skips — invariant 7, and the
same reasoning as `__check_init__` refusing a neutralized analyzer under a live
check.

### Role 3 — study of the oracle itself (56 tests, 5 files; 4 `validation/` scripts)

`test_equilibrium_consistency.py` (25) imports six symbols from `smax` directly
and is a study of the T1→T2 handoff *through* it. `test_analysis_prepared.py`
(6) tests `smax.py`'s own prepare-once optimization.
`test_frame_convention.py` (6) reads the roll convention off `SmaxAnalyzer`.
`test_sections.py` (11) and `test_materials.py` (8) assert bitwise agreement
with `ec3x.section` and `ec3x.material`.

In `validation/`, four scripts are studies of `ec3x`'s clauses and adjoint:
`class_ratio_sweep.py`, `interaction_gradients.py`, `load_case_envelope.py`,
`strut_gradients.py` — 1469 lines importing `ec3x.actions`, `ec3x.adjoint`,
`ec3x.classification`, `ec3x.resistance`, `ec3x.sizing`.

**Replacement is different for each half.**

- **The clause studies move to `../ec3x`.** That is where clause work lives by
  §1 of `CLAUDE.md`, where `docs/clauses.md` and `references/` already are, and
  where their imports resolve without a path pin. This is a move, not a
  rewrite.
- **The handoff and convention tests re-point to the crossed PyNite backend.**
  The funicular claim — that `q · L` agrees with a frame solve on the
  form-found geometry — does not care which solver reports the forces. The
  files are rewritten against `TesseractAnalyzer(backend="pynite")`, which
  costs ~320 and ~130 lines of editing respectively.
- **The section and material tests go to closed form, not to another library.**
  Verified while writing this: `normax`'s `area` and `second_moment` reproduce
  `CLAUDE.md` §4's formulas to the last printed digit at `d = 200 mm`,
  `d/t = 90ε²` (2073.84549869 mm² and 10026977.9065 mm⁴). Blueprints was
  checked as a substitute oracle and **rejected**: its `CHSProfile` builds a
  polygon (`accuracy = 6`), returning 2073.81101 mm² — 1.7e-5 relative, so it
  can cross-check loosely but cannot carry a bitwise assertion. Asserting
  against the standard's own algebra is the right answer anyway; a second
  implementation was only ever a convenience.

### The viewer is deleted, not re-pointed

`visualization/viewer.py` draws with `vix` and reads whole responses out of
`SmaxAnalyzer` — it already converts a `TesseractAnalyzer` *into* an
`SmaxAnalyzer` because the schema returns `N` and `M` rather than a response
object. There is nothing to re-point it at. All four examples ship
`viewer: false`, so removal costs a run nothing: three modules, one
`OutputConfig` field, four YAML lines, and eight lines across the four
examples. `plots.py` keeps every figure, and it is matplotlib only.

---

## 4. Phases

Each phase leaves the suite green. Nothing is deleted before its replacement
lands, so the tree is never in a state where the removal has to be finished to
be tested.

### 0. Record the goldens while the oracles still run

Before any deletion. Write the reference numbers the Role-2 tests will assert
against, at tag `local-dev`, into `tests/goldens.py` — a module of literals
whose docstring names the tag and the commit. This is the only phase that
cannot be redone later, and it is why the tag exists.

*Done when* `tests/goldens.py` holds every value a Role-2 test currently reads
from an oracle, and a temporary test asserts each literal against the live
oracle and passes.

### 1. Fixtures move to the crossed clients

Six files, two constructor calls each. Default to `backend="pynite"` rather
than `opensees` (see §7). Keep the composed side eager — the `jit` rule that
`tests/test_tesseract_parity.py` already follows.

*Done when* those six files import no oracle, 143 tests pass, and the suite's
wall clock is measured and recorded before and after.

### 2. Role-2 tests re-point to differences and goldens

Five files. Every derivative claim becomes `check_grads` or an explicit central
difference; every value claim reads `tests/goldens.py`; the comparisons that
only tested the oracle are deleted with a line in `CHANGELOG.md` saying which
and why.

*Done when* no file in `tests/` imports `smax`, `ec3x`,
`normax.analysis.smax` or `normax.sizing.ec3`.

### 3. The clause studies move to `../ec3x`

Four scripts, 1469 lines, moved with `git mv` into ec3x's own `validation/` (or
wherever that repo puts them — ec3x owns its layout). `docs/shear_design.md`
and `docs/ROADMAP.md` get the new pointer. Their run lines change the same way
the last move's did.

*Done when* `validation/` in this repo holds four scripts, none importing
`ec3x`, and each one's run line names its own path.

### 4. The three remaining `validation/` scripts re-point

`blueprint_adjoint.py` (786) and `sizing_formulations.py` (556) use the oracles
as fixtures — same swap as phase 1. `pynite_adjoint.py` (589) names `smax` as
its reference "because it is exact"; it already prices central differences, so
the reference becomes the difference and the prose says so.
`docs/fast_backward_pass.md` is that script's appendix and its claims are
restated at whatever the new reference gives.

*Done when* all four scripts run to completion and `docs/fast_backward_pass.md`
carries numbers a reader can reproduce.

### 5. The viewer goes

Delete `visualization/{viewer,guard,unavailable}.py`; drop `view_design` from
`visualization/__init__.py`; drop `OutputConfig.viewer`, the four
`viewer: false` lines and the four example call sites. Keep
`draw_design_figures`.

*Done when* `import normax.visualization` loads matplotlib and nothing else,
and the four examples run start to finish.

### 6. The package modules go

`git rm normax/analysis/smax.py normax/sizing/ec3.py`. Then remove the
`local-dev` group and its three `tool.uv.sources` entries, and strip
`conftest.py` to `load_tesseract_api` alone.

*Done when* `grep -rn "smax\|ec3x\|vix" normax tests examples validation
tesseracts` returns nothing outside a docstring that is deliberately historical.

### 7. The record catches up

`CLAUDE.md` §2 (the oracle sentence and the stack table), §6 invariant 2, §7
(the layout, the `visualization/` note, the shipping-stack paragraph);
`README.md`'s limitations section gains the sentence that the crossed stack is
now verified against finite differences and frozen references rather than
against a second implementation. `CHANGELOG.md` gets the reasoning.
`docs/{ec3x_extraction,sections_extraction,parallel_gradients,shear_design,ROADMAP}.md`
keep their oracle mentions — they are records of decisions taken while the
oracles existed, and rewriting them would falsify the history. Say that once,
here, rather than per file.

*Done when* nothing in `CLAUDE.md` or `README.md` describes a package that is
no longer there.

---

## 5. The gate

Not one of these is optional, and the first two are the ones that would
actually catch a mistake.

1. `uv run pytest` green, with the count stated. It will not be 395 — phase 2
   deletes some tests and phase 5 removes a config field. State the number and
   what accounts for the difference.
2. **A clean-clone check**, which is the whole point: `git clone` into a fresh
   directory, `uv sync`, `uv run pytest`. No sibling repo on the path. This is
   what a judge does, and until now it could not have worked.
3. The four examples run end to end on their committed defaults, and the arch's
   `fdm` route still lands at 0.150150 t. A moved number here means a fixture
   swap changed the physics, not the plumbing.
4. `--shape-parametrization` still gives three routes on all four structures.
5. CI green with the guard gone, and the collected count in CI equal to the
   local count. That equality is the deliverable of the whole exercise.

---

## 6. What this does not buy, and what it costs

**It does not make the crossed stack more correct.** The oracles agreed with it
to 1.3e-14 on gradients and 6.7e-16 on parity, and that agreement is the reason
to be confident. Deleting them does not withdraw the evidence — the tag holds
it and `CHANGELOG.md` records it — but it does mean the agreement is no longer
*re-measured* on every run. Say that plainly in the writeup rather than letting
a reader discover it: the crossed stack was validated against two independent
JAX implementations during development, and ships verified against finite
differences of its own forward pass.

**It costs test wall clock.** 143 fixture tests move from an in-process JAX
solve to a crossed call. The boundary was measured at 26% on the PyNite
adjoint, so the suite gets slower; how much is a measurement phase 1 owes.

**It loses the interactive viewer**, which is a real capability, not just code.
Nothing in the examples or the report uses it.

**It does not touch `ec3x` itself.** That repo keeps its clauses, its
`docs/clauses.md`, its Blueprints oracle and its own `CLAUDE.md`. It gains four
scripts. Whether it is ever published is a separate decision, after Aug 31.

---

## 7. Known frictions

**The stability risk is concentrated in phase 1, and it is not small.** Two
recorded faults live exactly where 143 tests are about to move. `jit` plus
Tesseract plus `openseespy` closes a file descriptor and once produced a
311-error pytest cascade, and a crossed-OpenSees descent has died with exit 139
mid-run. Both are why most of the suite currently sits on in-process oracles —
a fact that was incidental and is about to become load-bearing. Mitigations,
all already known to work: default fixtures to `pynite` rather than `opensees`;
keep the composed side eager; and land phase 1 file by file, running the whole
suite between each, so a cascade is attributable to one file.

**Two tests become vacuous and should be edited, not left.**
`test_second_sizer.py::test_this_file_names_no_standard_library` asserts that
the file imports no `ec3x`, and `::test_the_contract_imports_no_standard`
asserts `ec3x` is not in `sys.modules`. After removal both pass for the wrong
reason. Keep the `blueprints` half of the second, which is the half that still
says something, and delete the `ec3x` clauses.

**`test_analysis_prepared.py` tests a module that is disappearing.** Its six
tests are about `smax.py`'s prepare-once path. `pynite.py` has a `PreparedFrame`
of its own, so the concept survives the module; whether the file is rewritten
against it or deleted is a judgment for phase 2, and the honest default is
rewrite, because the prepare-once behavior is what made the objective 3686x
faster and nothing else asserts it.

**The `ec3x` adapter's `coerce_member_actions` has a test-only consumer.**
`test_pipeline.py` imports it directly, and also imports `DIAMETER_MINIMUM`
from `ec3x.section` while `normax/sizing/blueprint.py` defines one of its own.
Those two are **equal (21.3 mm), checked**, so that import is a rename and not
a finding. `coerce_member_actions` has no such twin: whatever it does for the
test has to be read and either inlined there or reproduced against the crossed
sizer's schema.

**Order matters in exactly one place.** Phase 0 must precede everything, and
phase 6 must follow everything. The middle five can be reordered freely, and
phases 3 and 5 are independent of the rest — either could land today.

---

## 8. What actually happened — 2026-08-28

**Landed the same day it was planned.** 390 tests pass with `smax`, `ec3x` and
`vix` uninstalled from the environment, not merely unimported. Six of the seven
phases went as written; the deviations below are the interesting part.

### Four things the plan got wrong

**The swap is faster, not slower.** The plan budgeted for a slowdown, reasoning
from the 26% the boundary costs on the PyNite adjoint. Measured on the 10-edge
arch: crossed PyNite analyzes in 1.30 ms against `smax`'s 25.53 ms, and the
crossed Blueprints check sizes in 1.21 ms against `Ec3Sizer`'s 18.69 ms — 20x
and 15x the other way, because the oracles paid JAX dispatch per call where the
crossed blocks pay one host round trip. The suite fell from 17.1 s to 10.9 s.

**Forward mode does not cross.** The plan assumed `check_grads` would drop in.
Neither server implements `jacobian_vector_product` — both offer `apply`,
`abstract_eval` and `vector_jacobian_product` and nothing else — so `jax.jvp`
raises and every gradcheck needs `modes=("rev",)`. `PINNED_ENDPOINTS` names the
jvp, but that list pins Jaxeract methods against thread migration and is not a
claim that a server offers the endpoint. This is the right design rather than a
gap: the augmented Lagrangian aggregates its rows into one scalar, so a gradient
is a single cotangent.

**The two sizers differ by ~6% once bending is live**, not the ~1e-15 measured
on an axial-dominated case. `Ec3Sizer` includes §6.3.1 member buckling and the
shipped Blueprints check does not, so freezing an ec3x diameter as a reference
for a Blueprints-checked design would have baked in that error. Caught before
any golden was recorded that way.

**A shared `tests/goldens.py` was the wrong shape.** Phase 0 was folded into
phase 2, each frozen value living as a module-level constant in the file that
asserts it, with a one-line provenance comment naming the tag. Four agents
worked in parallel; one shared file would have been a collision. Only three
values were frozen in the end — two gradient-block norms in
`validation/pynite_adjoint.py`, four shell reference norms in
`tests/test_tesseract_parity.py`, and four silenced-buckling diameters in
`tests/test_tesseract_sizer.py` — because differences and closed forms covered
the rest.

### `validation/` was repurposed, not exported

The plan filed four clause studies for a move into `../ec3x`. All four were
instead re-pointed at the shipped stack, which validates something that exists.
`strut_gradients.py` now agrees three analytic routes — the crossed adjoint, the
host adjoint, and the implicit-function rule written out in the script from its
own residual — to **1.05e-15**, the 1.39e-09 against a central difference being
the difference's own truncation. `interaction_gradients.py` gradchecks the
crossed sizer in four inputs at 4.08e-09 worst over 47 probes.
`class_ratio_sweep.py` reads its shear diagnostic from Blueprints' `Form6Dot18`
pair, so the diagnostic licensing the declined §6.2.6–6.2.8 now uses the same
library the shipped check does. All eight scripts run to completion.

### Two claims turned out to belong to the oracle

Both surfaced by re-pointing a test and watching it stop being true, which is
the best argument for doing this work rather than deferring it.

`test_frame_convention.py` asserted roll invariance of the *design actions*.
That is a property of `ec3x`'s resultant reduction; the shipped `reduce_moments`
superposes linearly per Eq. 6.2 and is not roll-invariant. The file now asserts
equivariance instead — turn the structure and its loads together and the actions
are unchanged to ~1e-15 — which is true of what ships. Element-level roll
invariance (1.7e-16) is untouched; that is a fact about a CHS stiffness matrix.

`test_analysis_prepared.py` asserted that a model prepared from *any* geometry
gives the same forces, true of `smax`'s placeholder prepare and false of
PyNite's, which factorizes at the real geometry. The file now asserts the
opposite and correct thing.

### The buckling story was rewritten from the standard

Deleting `ec3x` removed the only §6.3.1 code in the tree. No reported number
came from it — the examples always ran the crossed cross-section check, and
`examples/arch.py` still lands at **0.150150 t** — but the mechanism the project
argues about would have become unsubstantiable. So
`validation/blueprint_adjoint.py` implements 6.3.1 from the standard (Eq. 6.50,
Eq. 6.49 capped at one, Eq. 6.47 bisected, Eq. 6.6 on the tension branch) and
prices the shipped check against it: the gap runs **1.541 at the springing to
1.339 at the crown**, where `Ec3Sizer` gave 1.544 to 1.371. The claim is now
made from the standard's own equations, inside this repo, which is a better home
for it than a private package.

### Test count: 395 → 390

Accounted for exactly, not rounded off:

| file | change | why |
|---|---|---|
| `test_materials.py` | −2 | both tested `coerce_material`, the ec3x adapter itself |
| `test_design.py` | −1 | a Class 4 catalog refusal only `ec3x` performs |
| `test_equilibrium_consistency.py` | −1 | a sampled span field the crossed schema does not carry |
| `test_second_sizer.py` | −1 | "this file imports no `ec3x`" passes because the package is gone |
| `test_tesseract_sizer.py` | −1 | a floor comparison that reduced to `21.3 == 21.3` |
| `test_sections.py` | +1 | `i = c·d`, the invariant the monotonicity argument stands on |

### One standing exception, not introduced here

`normax/optimization/nested.py` jits the composed side itself, so
`test_nested.py` and `validation/sizing_formulations.py` compile a crossed
pipeline whatever the rule says. It predates this work and is stable — the
file-descriptor hazard is `openseespy`-specific and neither touches it — but the
rule is a rule, and unwinding it would move every reported wall clock. Left as
found, recorded here.
