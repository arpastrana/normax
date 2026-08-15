# Owning the materials and the sections — `normax/materials.py` and `normax/sections.py`

**The deferred half of `docs/ec3x_extraction.md` §2, executed.** That document
took the actions record out of the contract and parked `Tube` with the words
"deferred rather than fixed here", giving its trigger as a second
cross-section. The trigger turned out to be a second *sizer*: a
Blueprints-backed or SkyCiv-backed block behind `AbstractMemberSizer` cannot be
handed an `ec3x.TubeCatalogue` to fill or an `ec3x.Tube` to return, so the
contract as it stands is only fillable by the one backend it was abstracted
from. Decided 2026-08-15: normax owns its material and its section vocabulary,
and the standard's types stop at the standard's block.

---

## 1. The smell, counted

Three different things hide under one symptom, and they get different
treatment.

**The contract names `ec3x` types.** `normax/sizing/__init__.py` annotates
`MemberSizes.sections: Tube`, and `normax/design.py` *constructs* one inside
`design_envelope` — carrying `section_class`, a clause label, through the two
modules that are supposed to know no standard. One `ec3x` import line in each.

**The configuration layer speaks EC3.** `from ec3x.material import Steel` and
`from ec3x.section import TubeCatalogue` appear 17 times each across
`experiments/` and `tests/` — every pipeline driver names the standard's
material record and the standard's class-limit constructor just to build
blocks. `experiments/101_api.py`, the whole API in one file, opens with both.

**Not a smell: the experiments that study the standard.** 01, 05, 06 and 11
import `diameter_required`, the adjoint derivatives, the `LIMIT_*` codes,
`resistance_shear` — they are experiments *about* EN 1993-1-1 and should import
it. This document leaves them alone, and the same goes for the analysis
backend's eleven clause imports (`ec3x.stability`, `force_critical`,
`slenderness_from_force`): a backend is allowed to know which standard it
reports against. What it is not allowed to need the standard for is geometry.

---

## 2. The design

### `normax/materials.py` — what a mill certificate states

`ec3x.Steel` already splits itself: three fields are physics (`f_y`, `e_mod`,
`density`) and three are the standard's (`gamma_m0`, `gamma_m1`, `alpha` — a
safety format and a buckling-curve selection). The normax-owned record takes
exactly the first half:

    SteelGrade(f_y, e_mod, density)

The yield strength is a material fact, stated on the certificate, standard-free
— it belongs here. The partial factors and the imperfection factor select
clauses and stay on the EC3 side, supplied where the standard is named: the
adapter builds `ec3x.Steel` from a grade plus EC3's factors, and the analysis
backend does the same at the call sites where it consults `ec3x.stability`.
Defaults mirror `ec3x.material` (S355, 210000 MPa, 7.85e-9 t/mm³) so a bare
`SteelGrade()` is the same steel a bare `Steel()` was.

### `normax/sections.py` — tube geometry, and nothing a clause decided

Two containers, mirroring the `ec3x` pair minus everything clause-shaped:

    TubeFamily(ratio, material)                      # diameter → a section
    MemberSections(diameter, thickness, material)    # what a design carries

`MemberSections` carries the derived geometry `ec3x.Tube` already computes —
`ratio`, `diameter_inner`, `area`, `second_moment`, `radius_of_gyration`,
`modulus_elastic`, `modulus_plastic` — as properties, with the arithmetic
copied verbatim from `ec3x.section` (ours, Apache), so the two libraries agree
about what a tube is bit for bit. The geometry fields keep the variadic
`*load_cases members` axis and the material rides through, exactly as today.

Two boundaries are deliberate:

- **No `section_class` field.** The class is EC3's — it selects clauses, and it
  is the one field that forced `register_static` machinery into a pipeline
  container — `custom_jvp` traces its primals under `jit`, so a bare integer
  leaf becomes a tracer the moment the container enters the sizing map, which
  is why `SectionClass` had to travel in the treedef. It lives inside
  `Ec3Sizer`'s own catalogue, where it is already static and already verified.
  The contract pytree becomes plain arrays and a `NamedTuple` material, and the
  parity test's dtype walk loses its "crosses as a Python integer" special
  case, because no design carries an integer anymore.
- **Tube-shaped, not property-shaped.** The ROADMAP's shape-agnostic property
  *bundle* stays deferred with its original trigger, a second cross-section.
  The reason is the envelope: `design_envelope` smooths `diameter` and
  `thickness` and relies on their ratio surviving the smoothing, which holds
  because the envelope is scale-equivariant in the log. Enveloping derived
  properties independently would let `area` and `second_moment` drift out of
  consistency with any tube. The differentiable payload is the geometry
  parameters, honestly.

### The contract, and who converts

`MemberSizes` becomes `(sections: MemberSections, utilization)`.
`design_envelope` rebuilds a `MemberSections`; `compute_mass` reads
`.area` and `.material.density` from normax-owned types. After this,
`normax/design.py` and `normax/sizing/__init__.py` import nothing from `ec3x`
— the same test the actions record passed in the previous extraction.

Each backend converts at its own edge, the `design_actions` pattern:

- **`Ec3Sizer(structure, grade, section_class, resultant)`** builds its
  `TubeCatalogue` internally — choosing `d/t` from a class limit is Table 5.2,
  exactly this block's business, and `verified_class()` keeps firing at
  construction where the numbers are concrete. It sizes in `Tube` as today and
  restates the result as `MemberSections` on return, elementwise and free. Its
  `steel` property keeps returning the `ec3x.Steel` its catalogue holds: EC3
  vocabulary on the EC3 block's own surface is allowed, and
  `Ec3TesseractSizer` reads the gammas off it for the boundary dict.
- **`SmaxAnalyzer(structure, section)`** takes a normax `MemberSections` (one
  tube, as today). The family rebuilds inside `_injected_assembly` and
  `frame_stability` become `TubeFamily(section.ratio, section.material)` — the
  class drops out of both, and it was only ever there because
  `TubeCatalogue`'s constructor demands one; the geometry restatement needs
  the wall proportion and the material alone. Where the backend consults
  `ec3x.stability` it builds an `ec3x.Steel` from the grade at the call site.
  The `steel` property returns the `SteelGrade`, which carries the three
  fields `TesseractAnalyzer` publishes to the analysis schema.
- **The three `tesseracts/` backends** convert the same way at their edges.

### What the drivers look like afterwards

`experiments/101_api.py` imports nothing from `ec3x`:

    from normax.materials import SteelGrade
    from normax.sizing.ec3 import Ec3Sizer

    grade = SteelGrade()
    pipeline = StructuralDesignPipeline(
        FdmFormFinder(structure),
        SmaxAnalyzer(structure, TubeFamily(ratio, grade)(seed_diameter)),
        Ec3Sizer(structure, grade, section_class),
    )

The one place the standard is named is the block that implements it, and the
standard's knobs — the class, the moment-combination flag, in principle the
gammas — are that block's constructor arguments. Where does a driver's `ratio`
come from when the intent is "the Class 3 limit"? From the sizer, which owns
Table 5.2: `Ec3Sizer` exposes the catalogue it built, and the driver reads the
ratio off the block it configured rather than computing a clause itself.

---

## 3. What deliberately does not change

- **`ec3x`: zero churn.** It shipped on 2026-08-15 and its `Steel`, `Tube` and
  `TubeCatalogue` stay its internal working vocabulary, threaded through the
  `custom_jvp` tuples exactly as they are. This whole document is normax-side.
- **Both Tesseract schemas.** `f_y`, the gammas, `ratio` and `section_class`
  on `ec3_check`'s boundary are EC3 vocabulary on the EC3 block's own
  boundary; the analysis schema's `f_y`/`e_mod`/`density`/`ratio` are all
  fields a `SteelGrade` and a `TubeFamily` supply. No schema field moves, so
  no boundary number can.
- **The clause imports in the analysis backend** (`ec3x.stability`,
  `force_critical`, `slenderness_from_force`, `ALPHA_CR_ELASTIC`), per the
  extraction doc's ruling.
- **Experiments 01, 05, 06, 11** keep their deep `ec3x` imports by nature.

---

## 4. Phases

### 0. Baseline

normax 304 pass at `d0b44cb`, `ec3x` 1587 pass at `020298f`, smax on `main` at
`41601c4`. Names settled with Rafael 2026-08-15: `SteelGrade` (carrying `f_y`),
`TubeFamily`, `MemberSections` — the latter two deliberately not reusing
`ec3x`'s names, because the roles differ: no class, no verification.

### 1. `normax/materials.py`

`SteelGrade` and its tests. The adapter function that builds an `ec3x.Steel`
from a grade lives in `normax/sizing/ec3.py` beside the block that needs it.
No call site outside the new module changes yet — the constructors flip in
phase 3, so this phase cannot move a number.

### 2. `normax/sections.py`, and the contract retype — gated alone

`TubeFamily` and `MemberSections` with the geometry properties, tested against
`ec3x.Tube` for bitwise agreement on the same inputs. Then the retype:
`MemberSizes.sections: MemberSections`, `Ec3Sizer` and `Ec3TesseractSizer`
restating their tubes on return, `design_envelope` rebuilding the two-field
geometry, `compute_mass` and `frame_stability` reading normax properties.
`normax/design.py` and `normax/sizing/__init__.py` lose their `ec3x` imports.

**This is the phase that could move a number, so it gates alone.** The
arithmetic is copied verbatim, so the gate is the strict one: every recorded
tolerance at its recorded value (`TOLERANCE_PARITY` 1e-14, `TOLERANCE_SIZE`
1e-13, `TOLERANCE_MOMENT` 1e-11, `TOLERANCE_DERIVATIVE` 5e-12,
`TOLERANCE_UTILIZATION` 1e-9), and `experiments/101_api.py` reprinting
0.138951969 t, 16.114 %, +0.24867 % and a utilization of 1.000000000000 bit
for bit. A loosened tolerance is a failed phase.

### 3. The blocks take normax types

`Ec3Sizer(structure, grade, section_class, resultant)`;
`SmaxAnalyzer(structure, section)` with a normax tube; the family rebuilds in
`normax/analysis/smax.py` and the one `TubeCatalogue` import in
`normax/analysis/opensees.py` switch to `TubeFamily`; the `tesseracts/`
backends convert at their edges. Then the call-site sweep: the 17 + 17
`Steel`/`TubeCatalogue` imports across experiments and tests, of which the
pipeline drivers (02, 03, 04, 08, 09, 10, 101) end the sweep with no `ec3x`
import at all. Same gate as phase 2.

### 4. Prove the seam

A sizer that never imports `ec3x` fills `AbstractMemberSizer` end to end, in
`tests/` — allowable-stress design, `sigma_allow = f_y / 1.67`, closed-form
tube sizing under the axial force. Deliberately a different design philosophy,
not a reimplementation: the point is that the contract is fillable without the
standard's library, made a measured fact — the pipeline composes, the design
envelopes, the mass differentiates, and `grep ec3x` on the test file is empty.

**Stretch, cuttable: a Blueprints-backed sizer.** Blueprints implements the
cross-section formulas (6.6, 6.7, 6.10, 6.13, 6.14, 6.41, 6.42) and 6.44, but
nothing of Annex B, so it can never fill the full member check and full parity
is not the goal. What it proves is the licensing seam Rafael named: a
commercial-or-LGPL implementation consumed behind the contract, dev-only,
imported and never copied, sized by bisection over its own verdicts on the
arch's axially-dominated funicular case. If written, it lives in `tests/`
beside the existing oracle discipline and ships nowhere.

---

## 5. What this buys, and what it does not

The sentence it buys the writeup: **the pipeline contract names no standard.**
A design is geometry, forces, sections and a utilization; which normative text
decided the sections is an argument, the way the solver already is. That is
the strongest available form of the composition claim, and
`normax/sizing/skyciv.py` — still the point of the exercise, still unwritten —
becomes a backend whose inputs already exist.

It does not buy a second cross-section: `MemberSections` is a tube on purpose,
and the shape-agnostic bundle keeps its original trigger. It does not touch
`ec3x`, whose vocabulary is correct *for it*. And it does not resolve the
Tesseract wheel TODO, which is orthogonal and stays deferred.

## 6. Known frictions

- **`SteelGrade` revives a retired name.** The pre-P5d `SteelGrade` was renamed
  to `Steel` and later became `ec3x.Steel`; the name returns for a different,
  normax-owned record. Approved 2026-08-15.
- **`MemberSections` revives a pre-P8 name** for the same reason: the container
  returns to normax ownership. The P8-era type it replaced is gone, so there is
  no collision, only history.
- **Duplicated geometry arithmetic.** `ec3x.Tube` and `MemberSections` state
  the same seven formulas. Accepted: the alternative is a shared geometry
  dependency between the two repos, which re-couples what the extraction just
  cut, for seven lines of algebra that have not changed since Euler. The
  bitwise-agreement test in phase 2 is the drift alarm.
- **`test_design.py` reads `pipeline.sizer.steel.density`** and the parity
  fixtures read `arch.steel` — surfaces where a driver touches the EC3 block's
  own `Steel`. They keep working (the property survives) but the sweep should
  prefer the grade where the reader only wants physics.
