# Changelog

## Unreleased

### A diagrid connectivity

`build_diagrid_3d` builds the same spherical cap out of rhomboids: the polar
grid's meridians and hoops are deleted and the panel diagonals kept in their
place.

**A diagrid is half a grid.** Both diagonals of a quad preserve the parity of
ring plus spoke, so the quad grid carries two diagonal nets that share no node
and cross where no joint is. Keeping one and deleting the nodes the other
needed leaves a staggered mesh of half the nodes, every interior one spanned
by four members and none of them crossing. Matching a quad cap's node count
therefore asks for twice the rings or twice the spokes: 8x12 diagrid against
4x12 quad is 49 nodes either way, at 90 members against 84.

**The drawn cap is not funicular for a diagrid.** The quad cap is, exactly, and
that is what let experiment 23 start from the sketch: its hoops lie in
horizontal planes and carry no vertical load, so the meridians answer the load
alone and the hoops absorb what is left horizontally. Every diagrid member is
inclined, so vertical and horizontal balance couple at every node and the
system is genuinely overdetermined — `rank([A|b])` exceeds `rank(A)` by one at
every rise from a shallow cap to a hemisphere. `fit_densities` returns a
least-squares compromise rather than a fit, and a diagrid start has to come
from a form-finding solve rather than from the drawing.

**An open crown needs its ring.** A diagrid oculus leaves every node of the
first ring spanned by its two descending diagonals alone, and a node with two
members turns freely about the line joining their far ends. `ring_oculus`
restores one hoop there, the only place the family keeps one.

### Three numbers from the last section do not hold

Withdrawn, with the measurements that replace them.

**"The compression constraint costs 34%"** and the pair **"0.057875 t against
0.067404 t, free heights +16.5%"** were both read off an unguarded descent
that had landed exactly on its sag floor. Sweeping the floor moves the answer
and does not converge on one: at -2000 it reads 0.057875 t, at -4000 0.038592,
at -8000 0.054726, in a different and heavier basin. A floor-determined
landing is not an optimum and the difference between it and a guarded one
prices nothing.

**"The -22.9% is the price of a compression structure"** does not survive
either. Free heights lands compression-dominated of its own accord — 38 of its
members are in compression under every load case against the end-to-end
answer's 48 — so whatever separates the routes, it is not that one of them
builds a tension net and the other does not.

What does hold from that section: the basis of 13, the tributary loading, and
the drawn cap fitting an all-compression funicular with no self-stress at all.

### An open crown, and a triangulated one

`build_gridshell_3d` takes `oculus` and `braced`.

**An oculus is a narrowing, not a simplification.** Deleting the apex removes
one spoke's worth of members but only the two balance equations that node
stood for, so the null space a held plan leaves collapses from 13 to **3** on
the 4x12 cap. Doubling the spokes to 4x24 leaves it at 3: the basis is set by
the crown's topology, not by the mesh. The tributary rule treats the hole as a
hole — the first ring owns only the annulus outside itself and the opening
carries nothing — so the stated pressure buys less total load than on a closed
cap, which is part of what the opening costs.

**Bracing is what widens a held-plan basis.** The nullity is the member count
less twice the free nodes, up to rank, so it grows with members per node and
not with the mesh: refining a quad grid adds equations as fast as it adds
members, while triangulating one adds only members. Measured across the 4x12
cap, `basis / free nodes` runs 0.35 for the quad, 0.08 with an oculus, and
**1.24** braced; the oculus recovers from 3 to 36 under bracing, so the open
crown was never the problem and the quad topology was.

**Both diagonals of a panel, never one.** A single diagonal is chiral: the
mirror carries it onto the other diagonal of the same panel, the edge set is
no longer symmetric, and the folding refuses it. Alternating the direction ring
by ring does not repair this, measured rather than assumed. Full triangulation
is the only pattern here that survives the mirror.

### The drifts come in pairs, the foldings come in three, and a route may run alone

**The apex point case is gone, replaced by a mirrored pair of drifts.**
`ShellLoads` carries `sector_center`; the drift at `+center` and its reflection
at `-center` are separate cases, so each is genuinely one-sided while the pair
is symmetric about the plane the design is folded by. Measured on the 4x24
cap: each case's self-mirror gap is 1900 N, LC3 matches LC2 reflected to
0.000e+00, the pair is symmetric to 0.000e+00, and all three carry the
identical total. A crown is one node of many when it exists and no node at all
once an oculus opens, so a point load there was a property of the drawing
rather than of the structure; the parse refuses a sector centred on the mirror
plane, where the two drifts would be the same case twice.

**Three kinds of variable, three foldings.** `FoldingMaps` carries the mirror,
the height permutations and the member permutations, and `prepare_problem`
takes it as one argument. `orbit_matrix` reads orbits by union-find over every
generator at once, so a mirror and a one-spoke rotation fold the whole
dihedral group rather than two reflections separately; on a single involution
it reproduces `folding_matrix` column for column, checked by equality and not
by shape, so the trusses and every mirror-only run are untouched. The density
basis is folded by the mirror alone, always. `polar_diameters` folds the
sections as far as fabrication wants — 312 members onto 10 orbits, one per
ring per family. `polar_heights` folds the free heights the same way and is
kept apart from it, because it changes what the route comparison means rather
than what the design costs.

**A run may descend one route.** `viewer.solo_route` descends the viewer's
route and leaves the other two undone: 12.9 s against 4:37 on the oculus,
bit-identical answer. Every table reads its route list off the collection it
is handed rather than off `ROUTE_ORDER`, so a solo report writes the same
tables with one row and drops the entries that are comparisons; the verdict
then rests on convergence and feasibility rather than on beating a baseline
that was never descended. The routes table also prints whether each descent
converged, added because the verdict was failing with every printed tolerance
passing.

### The gridshell joins the race, and the code check picks the shape

The three-route flow now runs a shell. `experiments/truss_routes.py` is
`experiments/design_routes.py`, `TrussProfile` is `RouteProfile`, and every
callable a profile carries takes the parsed run description rather than a bay
count — so a family that is not drawn from bays, spans and depths can supply
its own. `TaskConfig`'s first three sections (`structure`, `loads`, `sketch`)
became the profile's to name and to read, `shared_sections` parsing the four
that are family-blind; `parse_truss` and `parse_shell` keep their own rules,
and the shell has no sketch at all. Experiments 18, 19 and 20 reproduce every
committed number bit for bit through the move.

**The boundary ring is gone from the generator.** A hoop member spanning two
supports appears in no equilibrium equation: its column of the horizontal
balance is identically zero, its density moves no node. Emitting it widened
the held-plan basis by one silent coordinate per spoke — the 4x12 cap's
nullity of 25 was 13 live directions and 12 that did nothing, which
differentiating the form finder shows directly as 13 nonzero singular values
of `d(xyz)/d(xi)` and 12 zeros. Dropped, the shell has 84 members and a basis
of 13, every coordinate of which moves it. The spectrum is one dominant rise
mode and then degenerate pairs, the polar harmonics.

**The load is a pressure spread by tributary area**, `create_loads_tributary`
converting it once. Sharing a total equally over the nodes of a polar grid
overloads the crown by 73% and starves the rim, the tributary areas running
threefold across the rings, and the funicular answer to that is a peakier cap
than a uniform pressure asks for. The areas sum to the plan exactly, which
makes the supports' share readable: at 1 kN/m2 the cap states **78.54 kN and
applies 60.13 kN**, the boundary ring's annulus going straight to ground.

**Under that load the drawn cap is already a compression-only funicular** —
zero members in tension, zero states of self-stress, a balance gap of 1.6e-11
— so its densities are unique and the end-to-end route starts on the drawn
geometry with nothing to shift. Under the equal-share load it was not: twelve
ring-2 hoops came out in tension, and the tension was the load's artifact
rather than the shape's.

**The compression rows are the design constraint, and they bind.** Every
member is guarded, `q = B xi` being linear, so the quadratic subproblem holds
each trial point inside the cone — where the vertical stiffness is negative
definite and a form-finding solve cannot go singular, a guarantee no truss
guard could give. Priced at the default 4x12 cap, 1 kN/m2, three cases:
**0.087452 t guarded against 0.057875 t unguarded, so compression costs
34%**. What the unguarded search buys is not a better shell but a different
structure — rise 0, **64 of 84 members in tension** — because EN 1993-1-1
applies no buckling reduction to a tie. The clause chooses the geometry.

That makes the shipped comparison asymmetric, and it is reported as such: the
end-to-end route carries a constraint the free-heights route does not, so its
**-22.9%** against free heights is the price of a compression structure. Held
to the same terms, with neither route signed, the form finder wins as the
counting predicts — **0.057875 t against 0.067404 t, free heights +16.5%** —
the Vierendeel situation, thirteen coordinates inside thirty-seven heights.
Buckling is live throughout: 48 of 84 members below `chi` 0.9 at the drawn
cap, the worst at 0.562, and nothing near the catalogue floor.

Two things the shell needs no help with. **No length floor**: a held plan
bounds every member length below by its own plan projection, 647 mm on the
ring-1 hoops, so the rows that cost the Vierendeel 13% of its mass are never
emitted. **No mirrored partner case**: the drift sector is centred on the
mirror plane and holds an odd number of spokes, so all three cases are
self-symmetric, the design folds to nine coordinates and forty-six diameters,
and the governing readout stays fully sighted. Shear stays excluded on the
measurement, worst `V_Ed/V_pl,Rd` **0.052** across the three answers.

### One crossing per Jacobian, and the boundary stops being the bottleneck

The blueprint boundary now offers a `jacobian` endpoint, and tesseract-jax
0.4.1 takes it by itself: whenever a batched `jacrev` or `jacfwd` meets a
server whose OpenAPI paths list `jacobian`, the sequential
one-crossing-per-cotangent-row fallback (upstream issue #244) is bypassed for
one crossing per call site plus a client-side `tensordot`. The endpoint is
pure NumPy and adds no derivative arithmetic — `_adjoint_state`'s per-member
pulls, written into blocks that are diagonal over members, shaped
`(*out_shape, *in_shape)` per the runtime contract. The client's only change
is a `materialize_jacobian` knob on `BlueprintClient` (upstream's own kwarg,
default auto) so the sequential route stays reachable for comparison; a
finding worth keeping is that `vmap_method="sequential"` does NOT disable the
route — the gate runs before the vmap dispatch — and that `jax.grad`'s
single-cotangent VJP never touches it, so mass gradients keep the hand
adjoint.

Measured by `experiments/22_jacobian_crossing.py` on the arch (30 constraint
rows, 11 variables, 3 load cases), median of 7. One slack Jacobian, reverse
mode: **19.2 ms through the endpoint (3 crossings) against 384 ms sequential
(90 crossings) in process — 20x**; served over HTTP, **27.6 ms against 618 ms
— 22x**. Forward mode: 18.2 ms against 136 ms in process (33 crossings —
the jacfwd flip buys 2.8x on the sequential route but is a wash once the
endpoint serves both modes). End to end, experiment 103's whole search on
`blueprint_tesseract` reproduces **0.157469285 t in identical 51 iterations
either way, at 2.35 s against 20.0 s** — the crossed backend now costs ~2.3x
the in-process 1.0 s rather than the recorded 37 s. The routes agree to the
bit (scaled gap exactly 0.0, in process and served); forward against reverse
sits at 1.72e-13. Five new boundary tests pin the blocks to `jax.jacobian`
of the traced rule at the hand-adjoint tolerances, the diagonal structure
exactly, key exactness, the clamp-mask refusal, and the crossing counts (one
`jacobian`, zero VJPs).

Rolled out to the **analysis server** on that number, both backends behind
the one endpoint: the traced backend answers with one `jax.jacrev` over the
same restricted solve its product rules trace, and the DDM backend hands
over the dense blocks its sweep already assembles — the mode that solver was
built for, since both its product rules were keeping one slice of exactly
this Jacobian per crossing. A pair with no block (the minor-axis moment a
plane frame does not carry) is an explicit zero, matching the products'
skip-means-zero reading. Route parity: the DDM routes agree **exactly**; the
traced routes to 1.2e-11 of the largest entry, which on a funicular arch is
nearly zero (~0.1 N/mm against 1e5 N forces), so that is reassociation seen
against a degenerate scale. `formfinding` and `ec3_check` are deliberately
skipped: both are JAX-native delegations that nothing sends batched
cotangents to — `TesseractSizer.compute_utilization` answers in process and
scalar mass gradients are single-cotangent VJPs the endpoint never serves.

The composition that motivated the rollout now runs end to end: **jax-fdm
traced in process, OpenSees behind a Tesseract, Blueprints behind a
Tesseract**, one gradient across all three — experiment 103 on
`analysis: opensees` + `sizing: blueprint_tesseract` lands the same
**0.157469285 t in the same 51 iterations, in 2.783 s** with both boundaries
on their jacobian endpoints, against 1.0 s all-in-process. Surfaced and
fixed on the way: the post-solve shear audit read `forces.shear_major` off a
crossed analysis, whose schema serves the three design fields alone, so the
stacked scalar defaults arrived shaped by load case and broke the division —
the audit now says the shear was not served and skips, and the in-process
default still reads its 0.1147.

### One load case may skip its container

`StructuralDesignPipeline.__call__` now also takes a bare nodal load array and
promotes it to exactly what `assemble_load_cases` builds from a one-entry
list, so the minimum working example loads, shapes and checks in three lines.
The narrowing is on the container type — static under `jit`, never a traced
value — and a stacked array is refused with a `ValueError`, because a raw
stack cannot say which of its cases the shape answers to. Bare-array and
assembled designs are asserted leaf-identical in `tests/test_design.py`.

### The API reads two levels deep, and builders say they build

Three logistics changes surfaced by writing the README's minimum working
example, applied across the package, the tests, the experiments and the
tesseracts. `thinnest_family` is now `build_section_family` — the old name
described the implementation's choice of wall, the new one what the caller
gets. The load-case generators carry the verb their registry sibling already
had: `create_loads_uniform`, `create_loads_half_span`,
`create_loads_half_span_mirrored`, `create_loads_point` beside
`create_loads_by_name`; the registry keys and the YAML case names are
unchanged. And no import reaches three modules deep anymore: the package
`__init__`s re-export their backends at the bottom, so every call site writes
`from normax.sizing import Ec3Sizer` and its kin. Two deliberate exceptions
keep optional dependencies optional: the OpenSees backend is reached as
`from normax.analysis import opensees` because `openseespy` is a spike extra
even in the full environment, and nothing about a re-export could stay lazy.
A consequence worth naming: importing `normax.sizing` now pulls the frame
solver and blue-prints along, so `test_materials` moved from the ec3x-alone
conftest gate to the pipeline and blueprint gates; `test_sections` still runs
with ec3x alone. Full suite 395 pass, CI environment 140 pass, README example
and the gridshell review both rerun bit-identically.

### CI goes green again, two masks deep

The suite had been red on every push since Aug 11 without anyone noticing,
because nothing before the first pull request put the check in front of a
reader. Three causes stacked up, each hiding the next. First, `equinox`
entered `normax/optimization.py` before it entered the project dependencies,
so CI's dev-only environment failed collection from Aug 11 until the
dependency landed on Aug 15. Second, the ruff 0.16 series began formatting
Python code fences inside Markdown, and `ruff format --check .` started
rejecting `ROADMAP.md`'s aligned trailing comments — invisible locally
because the pre-commit `ruff-format` hook only runs on Python files. The
formatter now excludes Markdown; the fences keep their alignment and `ruff
check` still covers everything it did. Third, behind the failing lint step,
`tests/test_plan_basis.py` imports `jax_fdm` at module level but was missing
from `PIPELINE_TESTS`, exactly the staleness its conftest comment warns
about; it is now guarded. A rebuilt CI environment (project plus dev group
alone) collects cleanly and passes 140 tests; the full local suite still
runs all 32 plan-basis tests.

### The excluded shear is measured, and stays excluded

EN 1993-1-1 6.2.10 lets shear out of the bending and axial checks while the
design shear stays under half the plastic shear resistance. That had been
argued from a bound over a demand mix, which describes a member that might
exist; `experiments/20_shear_audit.py` reads it off the members that do — the
arch at experiment 103's simultaneous optimum, and both trusses at each of the
three answers experiments 18 and 19 descend to. Eq. 6.17 is read once per
shear component at the settled diameters and taken at its worst, not on a
resultant of the two: a tube's shear area is the same whichever way the force
acts, which makes a resultant tempting, but whether one is sanctioned is an
open question in `ec3x` and the worst component needs no ruling. On a planar
frame the minor component is zero, so the two readings coincide, and the audit
asserts that rather than assuming it.

Worst `V_Ed/V_pl,Rd`: **0.3558** on the drawn Vierendeel, 0.1606 free heights,
0.1147 on the arch, 0.0962 Vierendeel end to end, and 0.0246 down to 0.0108
across the Warren. Torsion is **exactly zero** on all seven, a planar frame
under in-plane nodal load carrying none, so declining §6.2.7 is now measured
rather than assumed too.

Three things the measurement says that the bound could not. The **0.059 bound
does not bound a frame** — the drawn Vierendeel reaches six times it, because
that figure was taken over one 6 m member and a frame sets its own shear from
its topology. **Topology decides the number, not slenderness**: a Vierendeel
has no diagonals, so transverse load crosses each bay by frame action and
lands in the verticals, where every one of its worst cases sits, while a
Warren hands the same load to its diagonals axially — a factor of 15 between
two trusses of identical span, depth and load. And **optimizing lowers the
ratio**, against the intuition that thinner members sit closer to every limit:
a funicular member carries less moment and the shear is that moment's slope,
so the demand falls faster than the resistance does as the members thin. The
Vierendeel goes 0.3558 drawn to 0.0962 end to end while its mass falls 71.6%,
which makes the heaviest design in the study the one nearest the edge.

**So shear design is not landing.** Eq. 6.17 needs 1.0 against a worst of
0.3558 and §6.2.8's reduction needs 0.5, so a shear term would move no
diameter, while the cost falls almost entirely on the hand-written NumPy
duplicate of the Blueprints wrapper. `docs/shear_design.md` records what
building it would take, so the decision reads as a decision. What ships
instead is the measurement: every truss and arch run now carries the fraction
as a tolerance check against 0.5, so a structure that crosses it fails rather
than waits to be noticed, and the exclusion is stated where a reader will find
it — the README's limitations, the Blueprints sizer's own docstring, which had
attributed the scope to a library that implements all three clauses, and
ec3x's `docs/clauses.md`.

None of it licenses "shear is negligible for tubular frames". It is negligible
for these frames: 0.3558 is a factor of 1.4 under the threshold, not 20, and a
shallower or longer-spanned Vierendeel would cross it and be governed in its
verticals first.

### The analysis reports the shear and the torsion it was already computing

`element_forces` recovers all six components of the internal-force field, and
`member_forces` kept three of them. The other three — two shears and a torsion —
were computed and dropped at the return statement, so nothing downstream could
audit the clause the design check leaves out. EN 1993-1-1 6.2.10 lets shear be
ignored in the bending and axial checks while the design shear stays under half
the plastic shear resistance, and that is an exemption to be measured on a
converged design, not asserted.

`MemberForces` now states all six components as direct fields: `shear_major`,
`shear_minor` and `torsion_moment` beside the axial force and the two moments.
Six fields is a deliberate exception to the five-argument rule, taken because
the six components of a member's internal force are one thing an analysis
reports and not a group with a subgroup inside it. The three the check does not
read default to zero, so a caller stating a demand by hand states only what it
means to and a backend states all six. One number each rather than two — nodal
loading leaves the moment linear, so its slope is constant, and a frame loaded
at its nodes is given no distributed torque.

**A shear pairs with the moment it differentiates, not with the axis it shares a
letter with.** The first version of this read `vy` into `shear_major` beside
`my` in `moment_major`, which is wrong: bending about the major axis varies with
the shear across the minor one, so the major-axis shear is the solver's `vz`.
The magnitudes were right and in the wrong components, which no single-backend
test would have shown. The two backends disagreeing is what caught it, and
`test_the_reported_shear_is_the_one_the_solver_recovered` is what holds it.

The OpenSees backend reads its shear from the element's local end forces rather
than from a section, an elastic section in two dimensions resolving to an axial
force and a moment and carrying no shear of its own. Its minor-axis shear and
torsion are exact zeros on the same grounds the minor-axis moment already was.
The two backends agree on the shear to 1e-11, on a load case skewed off the
funicular so the agreement is not between two empty arrays.

The sizers now `vmap` over the design fields alone, through a `DESIGN_AXES`
prefix, rather than over the whole container. A check has no business mapping a
payload it never reads, and mapping it was what made an idle default unusable.

**What did not change: the Tesseract boundary.** The analysis tesseract's output
schema serves the three design fields, so a design taken across it reports no
secondary forces. Widening a served differentiable API means three new outputs
and the VJP plumbing to match, which is its own change. Until then the
limitation is asserted by `test_the_boundary_does_not_carry_the_secondary_forces`
rather than left to be discovered.

### The class-ratio sweep's shear bound was wrong by a factor of two

`experiments/05_class_ratio_sweep.py` inferred a shear from a moment as
`V = 4M/L`, and ec3x's `docs/clauses.md` quoted the resulting 0.12 as the
evidence that excluding 6.2.6 is honest. The factor came from a span-loaded
beam — a load case this pipeline never has. Under nodal loading the shear is
exactly the end-moment difference over the length, so with each end moment
bounded by `M` the worst pairing is antisymmetric at `V = 2M/L`. The bound now
peaks at 0.059, half the old figure and reached by the right mechanism, and the
report says it is a bound rather than a reading. `V = ΔM/L` is asserted to
`rtol=1e-10` against the solver.

### The end-to-end answer opens in a viewer, when a file asks for one

Both truss experiments end at PNGs, so the design the three-route
comparison exists to produce had never been seen in the frame solver's own
terms. `view_answers` draws named answers into one `vix` scene — a frame at
the answer's geometry and its own diameters, then per load case the
response `solve_response` returns with its displaced shape, the axial force
and the in-plane moment as diagrams, and the loads where they act. The
sections are the optimizer's diameters walled by the family every block was
built on, never a re-sizing: an envelope would substitute the sizer's demand
and draw a structure no table reports.

Which answer is drawn is named in the file too — `viewer.route`, refused
against the route vocabulary the way `basis` already is — so the end-to-end
design and the free-heights one, the route that skips the form finder and
moves heights through the analysis and the check alone, are looked at one at
a time. They are worth swapping between rather than overlaying: the two
answers sit about 1100 mm apart at their furthest node on the Vierendeel, so
a scene holding both is read by switching half of it off. The displaced
shapes come at true scale, which on a truss this stiff is a shape the panel's
slider has to open up before it reads.

A viewer holds the process until its window closes, so it rides behind a
`viewer.enabled` switch that ships **false** in both YAMLs and fires as
the last statement of a run, after the verdict, the figures and the
checks. Off, both experiments reproduce bit for bit — Vierendeel 0.122263
/ 0.145248 / 0.430474 t, Warren 0.062123 / 0.067785 / 0.111808 t, PASS in
each case. On, the Vierendeel scene holds seven registrations: the named
route's frame, and a response and a loads group for each of LC1, LC2 and
LC4. Every one is named apart, because a viewer's `add` replaces a
same-named registration and a loads group sharing its response's name would
tear that response down instead of joining it; and every response names its
parent frame outright, which is what lets a caller ask for more than one
route at once.

The shared module imports `vix` at the top of the file whether the switch
is on or not, so both experiments now want `--group viz` alongside
`--group pipeline`.

### A sag floor mirrors the rise ceiling, and the Vierendeel defaults fold

The `limit_sag`/`sag_factor` switches put a floor under every vertex at
`sag_factor` times the drawn depth below zero — at 1.0 on a 1000 mm truss,
nothing hangs under -1000 mm — mirroring the ceiling's mechanics exactly:
a box bound where heights are variables, one normalized inequality row per
free node where they are the form finder's outputs. Both limits travel as
one `HeightTruss` container, and each shaped route carries rise and sag
violation checks. `vierendeel_optimize.yaml` now defaults to the folded
configuration: `symmetric: true`, `basis: pivoted`, `mirrored_case: false`.

The Vierendeel's length floor also rose to the drawn vertical length —
`length_floor: 1000.0`, so no vertical shrinks under what was drawn.
Measured at those defaults, the end-to-end answer presses four constraint
families at once — rise exactly 2000, sag exactly -1000, verticals exactly
1000, chord sign margin 14.4 N/mm — landing **0.122263 t in 339
iterations**, free heights 0.145248 t, sizing only 0.430474 t, the
geometry buying 71.6%. The floor's price falls almost entirely on the
funicular route: at 500 mm it landed 0.108187 t while free heights was
already at 0.145310 t with verticals of its own choosing near 1000, so
the drawn-length floor costs end to end a further 13.0%, costs free
heights nothing measurable, and narrows the routes gap from +34.3% to
+18.8%. The sag floor alone had cost 1.7% (0.106355 t unfloored). The
Warren at its defaults — 3 cases, floors and limits identical, but no
member with a plan projection under its 500 mm floor — re-measures to
0.062123 / 0.067785 / 0.111808 t, the geometry buying 44.4%.

### The mirrored half-span case can be deleted, behind a parse gate

On a fully folded problem the mirrored half-span case's constraint rows
duplicate the other half-span case's through the mirror —
`U_LC3(m) = U_LC2(mirror m)` to 1.6e-12 — so `loads.mirrored_case: false`
now deletes it and buys a quarter of every analysis and Jacobian.
`parse_config` refuses the deletion on an unfolded problem, where the case
is what keeps the unloaded half honest. Measured on the symmetric pivoted
Vierendeel: the end-to-end mass is identical to the last printed digit
(0.106355 t, in 449 iterations against 530) and sizing-only is identical
outright; free heights lands within 0.2%, a path effect of the smaller row
count. The per-member readout is knowingly half-blind without the case —
far-half members report their worst among the remaining cases, so
attribution and fully-stressed counts shift while the design does not; the
gate is the only guard, by decision.

### The length floor reaches the end-to-end route, and it was not a formality

The member-length floor used to guard the free-heights route alone, on the
assumption that a signed funicular keeps its members long by itself. It
does not: the Vierendeel's end-to-end answers were resting on verticals
shorter than half the depth, buying mass from members the floor exists to
forbid. The same inequality rows — emitted only for members whose held plan
projection is under the floor — now enter the end-to-end slack, and they
BIND: the default answer rises from 0.098845 t to **0.106362 t** with its
shortest vertical at exactly 500 mm, the geometry now buying 75.3% instead
of 77.0%. The Warren emits no rows on either route and reproduces
unchanged. Both shaped routes report their shortest member beside the
floor, and both carry a length-violation check.

### The symmetric flag folds every variable now, and ties are counted fairly

The `symmetric` switch used to symmetrize the density basis alone, so the
free-heights route and every route's diameters searched unsymmetrized
variables and their answers broke mirror symmetry at will. It now folds the
whole problem: `MirrorFolding` in `truss_routes` carries one pattern matrix
for the diameters (one column per mirror orbit of members) and one for the
free heights, a pattern variable is the shared value of its orbit, and all
three routes expand through one matmul. With the switch off both matrices
are None and the routes reproduce their committed numbers bit for bit.

- **Folding the diameters rescued the symmetric Vierendeel search.** With
  the basis alone symmetrized, the symmetric end-to-end descent settled at
  0.145473 t (measured before the end-to-end length floor), far above a
  full-basis answer that is itself mirror-symmetric. Fully folded
  (6 coordinates + 12 diameter patterns, 18 variables) and under every
  constraint, it lands at **0.106355 t — within 7e-6 t of the full-basis
  0.106362 t — in 530 iterations against 1478**: the symmetric and full
  searches now agree on the design. Free heights folds to 8 + 12 = 20
  variables and lands 0.144851 t; every mirror gap is exactly zero by
  construction.
- **The Warren improves too**: at its symmetric default the end-to-end
  route lands 0.061965 t (25 variables, was 0.062026 at 40), free heights
  0.067392 t (24, was 0.068324 at 46), and the sizing-only optimum is the
  same 0.111808 t at 16 patterns as at 31 free diameters — the unfolded
  answer was already symmetric to 4e-9. The earlier numbers remain
  reproducible at their commit.
- **The governing table counts ties toward every tied case.** The mirror
  pairs LC2 and LC3 exactly: `U_LC2(m) = U_LC3(mirror m)` holds to 1.6e-12,
  and a self-mirrored member — the Vierendeel's midspan vertical — ties the
  two cases to 1.4e-12, so an argmax split by index order guaranteed odd,
  lopsided counts (10/9 where the physics says even). A member now counts
  toward every case within `TIE_MARGIN` of its worst, a row may sum past
  the member count, and symmetric designs report paired columns: 12/12,
  10/10, 11/11 on the Warren, 13/13 and 8/8 on the Vierendeel. The figure
  follows: `UtilizationForm` now carries the counted bars (`governed`)
  instead of per-member argmax labels, `governed_counts` feeds the table
  and the drawing from one rule, and no argmax attribution survives
  anywhere in the truss experiments.

### The subspace is a form finder now

`SubspaceFormFinder` in `normax/form_finding/fdm.py` wraps any form finder
in a held-plan density basis, so the design space itself is the block's
first argument: a call takes basis coordinates, expands them through
`member_densities`, and hands the densities to the wrapped finder. The two
coordinate read-back conventions — projections `Bᵀ q` for the orthonormal
basis, `q[independents]` for the pivoted one, never `Bᵀ q` there — now live
in one place, `read_coordinates`, instead of being re-derived per
experiment. The truss experiments build their pipelines with it as the
first block; the searched width is read off `basis.shape[1]`, the
chord-sign guard reads densities through the block, and both experiments
reproduce their numbers (experiment 18 bit for bit). Four tests in
`test_plan_basis.py` pin the expansion, both read-backs, and the chain rule
`∂/∂ξ = Bᵀ ∂/∂q` through the wrapped solve.

### The Vierendeel truss optimized end to end, on machinery both trusses share

Experiment 19 reruns experiment 18's three-route race on the truss where
funicularity is scarce, after the route machinery — the configuration and
its YAML parsing, the four load cases, the compiled maps of all three
routes, the restarted SLSQP, the report tables, the checks and the
figures — moved into `experiments/truss_routes.py`, shared by both
experiments. Experiment 18, slimmed to what is Warren about it, reproduces
its committed numbers bit for bit. The Vierendeel defaults race the full
(never symmetric) svd basis — nine independent edges, two uniform chord
families plus seven free verticals — against fourteen free heights, each
with all 23 diameters, under the 2000 mm rise ceiling, hard `U <= 1` per
member and load case. **The geometry bought 77.0%**: 0.098845 t end to end
against 0.430474 t sizing only, every state PASS, constraint violations
orders under the 1e-6 headroom.

- **Funicular discipline pays double where the joints are rigid.** The
  drawn Vierendeel carries the deck through joint bending alone, and
  sizing it without moving it costs 0.430474 t — diameters to 274 mm,
  verticals idling at U 0.41 while the envelope holds their neighbors at
  one. Bending the chords into simultaneous funiculars starves those
  moments: the end-to-end answer is 4.4x lighter, against 1.8x on the
  axially-competent Warren.
- **The reach argument reverses, and under the ceiling the form finder
  still wins.** On the Warren both shaped routes spanned the same
  geometries, so the +10-15% free-heights gap was pure landscape. Here the
  counting flips — every held-plan geometry is drawable by heights,
  funicular ones are a strict 9-of-14 submanifold — the heights route is
  the superset for the first time, and it *still* lands +40.5% heavier
  (0.138896 t) from the same matched lens start, mirror symmetry visibly
  broken (1.5e-2 against 5.2e-8). Basin caveat: on the symmetric basis the
  two shaped routes nearly tie (+2.11%) in a far heavier basin, and with
  the ceiling lifted the verdict inverts outright (next bullets) — the gap
  is landscape-dependent; treat it as configuration-specific, never as a
  constant of the truss.
- **The optimizer leans on the funicular-existence boundary.** The
  end-to-end route needs the chord-sign rows (`ChordSigns`, linear in the
  coordinates since `q = B ξ`) to keep the search off the degenerate
  chord-off states of experiment 17, where the vertical stiffness turns
  singular and the frame solve returns non-finite — without them the
  descent crashes mid-line-search. At the answer the rows BIND at their
  14.4 N/mm margin; halving the margin buys only 0.85% (0.098010 t) and
  binds again. The watched vertical stiffness stays regular: cond 33 at
  the start, 1.7e2 at the answer, 7 negative eigenvalues of 14 throughout.
- **The heights route has its own degeneracy, and its own wall.** A
  Vierendeel vertical joins two nodes of equal plan position, so a height
  crossing collapses it and hands T2 a singular frame — the Warren never
  could, its plan keeps every member at least 625 mm long. The length
  floor (`length_floor`) enters the heights slack as inequality rows
  emitted only for members whose held plan projection is under the floor,
  so the Warren emits none and reproduces unchanged. The floor is half the
  drawn depth — 500 mm, and `parse_config` refuses less — so a vertical
  stays a member rather than a near-hinge; every full-basis heights answer
  rests exactly on it. A third guard sits in the driver: a trial frame
  that cannot be factorized is answered with a uniform 1e3 violation
  (`RECOIL_SLACK`) instead of a raise — infeasible is the truthful reading
  of a structure that cannot stand — and the line search recoils.
- **The rise ceiling decides which parametrization wins.** Under the lid
  the funicular route wins by 40.5%. Lifting it, the end-to-end descent
  lands *heavier* (0.120785 t against 0.098845 t capped) on a wildly
  asymmetric design (diameter mirror gap 0.498, depth 3554 mm), while the
  free-heights route drops to **0.076276 t — the lightest design of the
  whole study**, 23% under the capped end-to-end best. On the Warren the
  free rise saved the end-to-end route a further 6.6%; on the Vierendeel
  the lid regularizes the funicular landscape and starves the heights
  route, and without it their verdict inverts. The lid is active in every
  capped shaped answer (rise exactly 2000).
- **Coordinates pick the basin, louder than on the Warren.** The pivoted
  basis lands within 0.35% of svd end to end (0.099191 t; its heights
  route 0.144950 t), but the mirror-symmetric basis lands 47% heavier
  (0.145473 t) even though the full-basis answer is itself
  mirror-symmetric to 5.2e-8 — the symmetric subspace contains the good
  answer's neighborhood and the descent still cannot reach it. The Warren
  priced the same switchings at ~1.3%.
- **LC1 never governs, again.** Zero members on every route in every
  state; the default end-to-end envelope splits 13/10/0 over LC2/LC3/LC4.
  The elastic axial forces sit 5.1% from the funicular `q L` at the lens
  (the Warren's split was 22.6%), and the frozen-seed envelopes are
  infeasible re-analyzed at their own sections: 14.8% at the lens, 15.5%
  as drawn.
- **The start is fitted inside the basis, never free** — experiment 17's
  degeneracy is the reason — with the fit gap at 1.4e-15 of the total
  load, the projection gap at 2.9e-16, and the sign shift along the one
  self-stress, the load-path split. End-to-end and free-heights gradients
  agree with central differences to 2.7e-10 and 3.5e-12.

### The Warren truss optimized end to end, against two routes without T1

Experiment 18 puts the held-plan subspace to work: simultaneous SLSQP over
the nine symmetric basis coordinates and all 31 diameters, under hard
`U <= 1` per member and load case, against two searches without the form
finder — one moving the fifteen free heights and the diameters through T2
and T3 alone, one holding the truss as drawn and moving the diameters only.
Four load cases, all on the bottom chord — a shared 180 kN total for the
uniform deck and the two half-span cases, and `point_factor` of it (0.5)
concentrated at the midspan deck node. The point case is the one knob
exempt from the shared total: at the full 180 kN it governed 22–23 of 31
members and every design was essentially its funicular, so it was halved
to make the envelope an actual contest — governing now splits 11/12/8
across LC2/LC3/LC4, and LC1, the case the shape is form-found under,
governs zero members on every route. With the rise left free, **the
geometry bought 48.0%**: 0.058185 t end to end against 0.111808 t sizing
only, every answer feasible to 1.4e-11.

- **The answer is the funicular of the governing cases, not of the shaping
  case.** At the full-total point case the truss left the lens for the
  pitched triangle a midspan point load wants; at the halved default it
  settles into a rounder profile answering the half-span pair — rise
  3042 mm, sag −264 mm, on a 1000 mm drawn truss, free rise. Which case
  writes the shape is a load-model decision, never the form finder's.
- **The truss is once statically indeterminate, and it shows twice.** The
  elastic axial forces disagree with the funicular `q L` by 22.6% of the
  largest force under the shaping case — the thrust-vs-tie split is
  stiffness-dependent, so T1 handing T2 geometry only is not a formality
  here. And the frozen-seed envelope that seeds each search is measurably
  infeasible re-analyzed at its own sections: 5.8% on the lens, 11.5% on
  the drawn truss (14.1% and 23.7% at the full-total point case), where the
  arch priced the same coupling at a tenth of a percent — the sizing-only
  descent spends its opening iterations *gaining* mass to purge it. At the
  full-total point case the indeterminacy also kept six sizing-only members
  strictly under-utilized (U 0.80–0.88 at diameters far above the floor):
  the classical fully-stressed-design-is-not-minimum-weight result, their
  stiffness buying force relief for governing neighbors. At the halved
  default every sizing-only member ends exactly fully stressed.
- **Free heights lose 14.9% with the reach excuse removed.** On the arch of
  experiment 15 the heights were a strict superset of the funicular family;
  on the Warren, experiment 16 counted every held-plan geometry
  funicular-reachable, so the two shaped routes span the same designs and
  any gap is landscape, never reach. From the same matched lens start, the
  same solver and the same constraints, the free-heights route (46
  variables) lands on 0.066848 t against the density route's 0.058185 t —
  +14.9% with the rise free, +10.2% under the ceiling — thousands of
  millimeters away in shape and with mirror symmetry visibly broken
  (diameter mirror gap 2.8e-2 against 3.4e-13). The physics prior wins on
  conditioning alone, and by more the harder the problem gets.
- **A rise ceiling is the YAML's `limit_rise` switch** (on by default,
  `rise_factor` 2.0 — no vertex above twice the drawn depth, 2000 mm; the
  sag stays free). Each route carries the same wall as it naturally can:
  a box bound on the free-heights variables, one normalized inequality row
  per free node for the end-to-end route, whose heights are outputs of the
  form finder. The lid is active at the optimum, not merely satisfied:
  both shaped routes press two nodes onto it at max z = 2000.000000000
  (the constraint rows bind to 9e-11 mm, the box exactly), where the free
  optimum rises to 3042. It costs end to end 6.6% — 0.062026 t capped,
  still 44.5% under sizing only — and free heights land at 0.068324 t,
  +10.2% behind.
- **The answers are symmetric whichever basis runs, but the coordinates
  pick the basin.** The subspace can be spanned two ways — the YAML's
  `basis` switch chooses the orthonormal `svd` columns or the member-named
  `pivoted` ones (QR's independent edges, coordinates read back as
  `q[independents]` rather than projected) — and both, plus the full
  16-wide basis, reach the identical design space (projection gaps
  4e-14). Under the ceiling the svd run lands on 0.062026 t while the
  pivoted run lands on 0.061244 t and the full basis on 0.061253 t —
  the same 1.3%-lighter basin twice, all three answers symmetric to
  6e-4 or better. The reachable designs never changed; the coordinates
  scaled SLSQP's steps differently and rolled it into a different local
  optimum. Not a case for asymmetry, and not for either basis outright:
  a case for treating the answer as basin-dependent on this landscape.
- The end-to-end gradient — form finder, frame analysis and code check in
  one derivative — and the free-heights gradient — coordinates straight
  into the analysis — agree with central differences to 1.4e-09 and
  2.6e-11 along random mixed directions of geometry and diameters. The
  signed lens start projects into the basis to 8.0e-14 of `|q|`; the
  restricted fit is not used for the start because its self-stress
  direction rides at the least-squares rank cutoff (measured 1.4x above
  it), while the free fit's sits orders below.
- Two diagonals rest on the 21.3 mm floor in the end-to-end answer (eight
  at the full-total point case); SLSQP is restarted from its own answer until a round stops
  moving (three to five rounds), which is what gets it off the slow tail
  of a single long run. `figure_mass_descent` and `figure_utilization`
  joined the figures — the designs figure colors every member by its
  envelope utilization, capped at one and floored just under the least
  worked member, so the sizing-only panel shows its under-utilized bottom
  chords and middle diagonals as visibly cold, with the governed-member
  counts per load case kept as bar panels underneath;
  `positions_vertical`'s "not a design space" warning is now scoped to the
  chain it was measured on, deferring to `density_basis` elsewhere.

### The Vierendeel truss, where funicularity gets scarce

Experiment 17 removes the Warren's diagonals and the counting flips back
against the geometry. Verticals project to nothing in the plan balance, so
the horizontal equations split into two full-rank chord chains: the eight-bay
Vierendeel's 23 members hold a **nine-wide held-plan subspace** — one uniform
density family per chord plus seven free verticals — against fourteen free
heights. Funicular shapes are the exception rather than the rule, which is
the strongest form-finding case in the repo: the truss as drawn misses axial
balance by **0.500 of its load** (exactly half — least squares splits the
untransferable hanger load evenly between the chords' residuals), where the
Warren lens missed by 1e-14 of it. `build_vierendeel_2d` joined the
generators, and `fit_densities` grew an optional `basis` to fit inside a
held-plan subspace.

- **Four supports, decided and demonstrated.** Both chords spring at pinned
  corner nodes. The two-support variant with a floating top chord is built in
  the experiment and its basis measured: eight independent edges, and the top
  chord's density is identically zero across all of them — the chain balance
  forces it, so the topology was corrected rather than argued. There are no
  end posts: they would join two supports and carry nothing in any model
  here. T2 will model every member as a rigid-jointed beam, since a
  pin-jointed Vierendeel is a mechanism.
- **The reachable sketches are the simultaneous funiculars.** A parabolic
  lens fits exactly (1.9e-9 N under 140 kN) because both chords are
  funiculars of one hanger-load split; so does the flat deck (3.2e-10 N),
  whose level bottom chord carries nothing vertical. A quartic-top bulge is
  reachable by no split — and the free least squares hides that: it abandons
  the chord, zeroing hangers and top densities (3.3e-13 N/mm left), reports a
  deceptively near-zero gap, and hands back a vertical stiffness with
  condition 1.8e16 whose solve drops the top chord 1603 mm from the sketch.
  The fit gap alone does not certify a sketch reachable; the re-solve does.
  Hence the `basis` parameter: fitting inside the subspace keeps the plan
  balance exact by construction.
- **The self-stress is the load-path split.** The curved Vierendeel carries
  exactly one state of self-stress (Maxwell: m + r − 2n = −1, so s = k − 1),
  and it is the trade between hanging deck and arching top chord — the
  Vierendeel's analogue of the Warren's thrust-vs-tie split. On the lens the
  minimum-norm fit takes both paths (bottom +96.1, top −127.9 N/mm, hangers
  +5.3 to +7.9); on the flat deck it puts everything in the arch and leaves
  the tie at zero, and the sign shift of −226.9 along the self-stress is
  literally the tie tension. The height Jacobian confirms the split: 5 moving
  directions of 6 symmetric (8 of 9 full), blind direction aligned with the
  fitted self-stress to |cos| = 1.0.
- **Symmetry shrinks nine to six** — the two chord families, symmetric
  already, plus four symmetric vertical patterns — under the same
  `symmetric` YAML switch as experiment 16. QR pivoting elects one member
  per chord and every vertical as the independents (vertical columns are
  zero in the balance, so they can never be dependents), largest transfer
  coefficient 1.00.
- The mixed-sign vertical stiffness at the signed lens densities is
  indefinite but comfortable: 7 of 14 negative eigenvalues, condition 33.
- Both flag states of experiment 17 PASS; `tests/test_plan_basis.py` pins
  the widths, the floating-top zero, the exact lens fit, the load-path
  split, the degenerate free fit, and the pivoted election, and
  `tests/test_structures.py` pins the generator's family-ordering contract.

### The Warren truss, and the held plan's independent edges

Experiment 15 proved the held plan degenerate on a chain — one funicular
parameter, `q_after = q_before`. Experiment 16 runs the same counting on an
eight-bay Warren truss and gets the opposite verdict: 31 members against 15
horizontal balance rows leave a **sixteen-wide held-plan subspace**, and the
count is exact bookkeeping — fifteen free heights plus one state of
self-stress, the pin-pin truss's thrust-vs-tie split. Every bay stays even
and every diagonal aligned by construction, because the plan never enters
the solve. The machinery lives in `normax.form_finding.fdm`:
`plan_equilibrium` (the balance, a constant of the topology),
`density_basis` (its null space by SVD, mirror-symmetric on request),
`fit_densities` (exact densities for any drawn geometry, self-stresses
included), `equilibrium_gap`, and `pivoted_basis` (the same subspace in
member coordinates). `build_warren_2d` joined the generators, and
`figure_truss_forms` and `figure_density_modes` joined the figures.

- **Every held-plan sketch is exactly reachable.** The balance is linear in
  the densities at a fixed geometry — 30 equations, 31 unknowns, full rank —
  so a drawn lens and a drawn flat-deck truss fit to gaps of 1.1e-10 and
  9.5e-11 N under 140 kN of deck load, and the vertical solve reproduces the
  lens from the flat truss to 7.3e-12 mm. The raw fit signs its chords
  wrong; the shift along the one self-stress that fixes them is an interval
  ([-679, -21] for the lens), intersected exactly and taken nearest zero.
- **Mixed signs make the vertical stiffness indefinite, not singular.** At
  the signed lens densities: 8 of 15 eigenvalues negative, condition 52.
  The solve stands; the spectrum is printed rather than assumed.
- **The free plan is the cautionary picture.** One density per family, at
  the scale of the signed fit, handed to the unrestrained solver: bay widths
  drift from 156 to 2052 mm on a 1250 mm nominal, nodes land outside the
  span, and the truss hangs inverted below its supports. Right signs and
  right scale do not protect the shape; membership in the subspace does.
- **Mirror symmetry shrinks sixteen to nine** — eight symmetric height
  motions plus the self-stress, itself symmetric — imposed as extra rows on
  the same null-space solve. It costs the start nothing: the lens densities
  rebuild in the symmetric basis to 7.9e-14 of their norm. The
  `symmetric` switch in `warren.yaml` selects which basis the mode split,
  the variations and the checks operate on.
- **The height Jacobian splits the subspace for the eye.** Its SVD orders
  the directions by gain — 425 mm of motion per unit density step down to
  11 — and its null direction matches the algebraic self-stress to
  |cos| = 1.0 at machine precision. Three figures step every direction: the
  linearized modes, the re-form-found variations, and the pivoted
  variations, which nudge one named member at a time.
- **QR with column pivoting elects the independent edges the TNA way.** On
  the Warren it claims the large-coefficient chord columns for the dependent
  block and leaves eight diagonals plus one bottom chord as the named
  variables; the largest transfer coefficient is 1.00, so no independent
  density is amplified into a dependent one. The lens start reads back by
  pure slicing to 3.8e-14 of |q|.
- **The symmetric gridshell has 25 independent edges, not the formula's
  22.** `edges - 2 x free nodes` assumes full rank, and the polar cap's
  balance is rank-deficient by exactly its three free rings; the SVD count
  is the measurement and the formula is a lower bound.

### A form finder as a shape generator, priced against free coordinates

Experiment 15 descends the same arch through two parametrizations of a held
plan: the height of every free node — nineteen variables straight into the
analysis, no form finder — against one uniform force density, which is the
entire funicular subspace of that plan on a chain. The starts are matched
exactly (the same geometry entering both routes, gap 2.7e-12 mm, by the
linearity of the force density system) at 0.25x, 1x and 2x the funicular
rise; two more starts — a flat line and random heights — lie off the
funicular manifold where no force density can begin, so the height route
descends them alone. The direct route's new differentiation path checks
against a directional central difference at 4.0e-8.
`figure_parametrization` joined `normax.visualization` to draw the three
claims side by side.

- **The one-variable route is start-proof and the nineteen-variable route is
  not.** One density lands on 0.12227 t within 1.8e-4 relative from every
  start; the free heights spread 3x over their five. From the shallow
  start they find 0.121085 t — the only run to beat the density optimum,
  by 0.95%, which realizes the superset argument. From the matched start
  they end +2.33% heavier and leave the rise at 3000 mm: the descent stalls
  on the plateau instead of reshaping. From the tall start they stop at
  0.1626 t, barely below the 0.1648 t they started at.
- **The no-prior starts are the argument for the prior.** From the flat
  line the height route never leaves the ground: five iterates, rise 0.0,
  0.478 t — 3.9x the optimum — every member carrying the shaping case in
  pure bending on the box's own floor. From random heights the gradient
  does real work, smoothing the jagged profile into an arch of rise
  4532 mm, but still lands at 0.1416 t, 17% above the best answer, with a
  2.9e-3 mirror gap the symmetric problem never justifies.
- **The prior is visible in the iterates, not only the answers.** The
  density route's worst `|M|/(|N| L)` under the shaping case stays at
  6.1e-3 across all three descents (elastic bending at the seed stiffness,
  largest on the shallow start); the height route's reaches 6.8e-2 — 11x
  more — the moment it leaves the funicular curve.
- Neither route needs the length floor of experiment 03: the plan never
  moves, so no member can shorten past its own projection and the collapse
  mode does not exist. The height box floors at zero instead, which keeps
  the search out of the hanging tension half-space — a cable, not an arch.
- Utilization at every answer holds the fully-stressed invariant to 8.1e-15
  — including the stuck flat line and the random start. Fully stressed is
  every design's condition here, so no answer is "more utilized" than
  another; what a worse shape pays is a fatter section reaching the same
  utilization through bending, and the mass is where that shows.
- The bending measure walls its denominator at one newton-millimeter: a
  straight beam's axial force is identically zero, so the flat start's
  pure-bending ratio would otherwise divide by zero. The wall binds nowhere
  else, and the flat run reports 2.4e8 rather than infinity.
- **The staggered driver closes the `∂N/∂d` feedback and prices it at a
  tenth of a percent.** `optimize_staggered` — in `normax.design` since P8,
  called by nothing until now — reruns every descent with the analysis
  re-sectioned between rounds, each answer re-read at the sections its last
  settling pass demanded. The density route's answers move +0.07% to
  +0.10%, most height runs +0.10% to +0.29%, and the flat line does not
  move at all: closing the coupling rescues nothing the landscape traps.
  The one exception runs the other way — the shallow-start height run rides
  its refreshed sections off the plateau to 0.120275 t, −0.67% below its
  frozen answer and the best design in the experiment. All eight couplings
  closed within the driver's default rounds, the utilization invariant
  holding at the settled sections too. The spread panel draws the settled
  masses as hollow markers dodged beside their frozen pairs — same color,
  same shape, open against filled — so the near-coincidence is the visible
  finding rather than a hidden one.
- **The third formulation makes the diameters variables outright, and it is
  start-proof for free.** SLSQP over the one density and all twenty
  diameters under hard `U <= 1` per member and case — no sizing solve, no
  envelope, the `∂N/∂d` feedback inside the gradient — lands on
  0.122337979 t at rise 1946 mm from all three matched starts, identical to
  nine digits, in 68–97 iterations. Utilization at the answer is exactly
  1.000000000 (violation 5.5e-12): fully stressed as a KKT condition rather
  than by construction. It beats the staggered answers by 0.001–0.045%,
  which prices carrying the coupling inside the gradient against closing it
  between rounds — hundredths of a percent, on top of the tenth the frozen
  seed cost. The spread panel draws it as one dashed level rather than a
  row of markers, a start-independent answer being exactly what a
  horizontal line says.
- The run description moved to `experiments/parametrization.yaml` — the
  arch, the load, the annealing schedule, the boxes, the starts and the
  constrained budgets — parsed strictly with no defaults, the 101 pattern.
  The verdict's tolerances stay in the script, being the experiment's
  assertions rather than its settings. Reading `arch.yaml` instead was
  rejected: that file is 101/102/103's contract (seed 250, blueprint
  sizing), and sharing it would move every number this experiment pins.

### The blueprint boundary carries the held-size question

The `blueprint_check` schema answered one question — what size these actions
demand — so `BlueprintClient.compute_utilization` stayed home, and a
simultaneous search on the client never actually crossed. The schema now
carries both: a `diameter_held` input comes back as `utilization_held`, the
check at sizes the caller owns. The new adjoint is the cheap one — explicit
partials, no root find, no implicit function theorem — added to the same
hand-written NumPy `vector_jacobian_product` and `jacobian_vector_product`,
still with no `import jax` anywhere in the module. The client's re-read now
crosses per load case, `__call__` sends a placeholder the solve never reads,
and both Tesseract clients pass `vmap_method="sequential"` so a `jacrev`
through the boundary becomes one crossing per Jacobian row instead of a
refusal. `sizing.backend` grew a third value, `blueprint_tesseract`, so the
constrained search of experiment 103 runs with every constraint evaluation a
boundary crossing and every constraint Jacobian pulled row by row through
the hand-written adjoint.

- Parity, crossed against in-process: the held check is pinned bit-identical
  to `checked_utilization`, its adjoint matches `jax.vjp` of the traced rule
  per input, and the client's re-read reproduces `BlueprintSizer`'s values
  at `array_equal` with the diameter gradient to 1e-12
  (`tests/test_blueprint_sizer.py`, suite at 25).
- Experiment 103 on `blueprint_tesseract` lands on the in-process blueprint
  answer to the last printed digit (0.157469285 t) in 37 s against 1.0 s —
  the cost is ~30 sequential VJP crossings per SLSQP iteration, one HTTP
  round trip each when served (tesseract upstream issue #244), plain
  function calls in process.

### The analysis solver picks from the file

`arch.yaml` grew `analysis.backend: smax | opensees` beside the sizing
switch, dispatched in `101_api.py`'s `build_pipeline`: `smax` is the
in-process block as before, `opensees` sets the stage's environment variable
and builds a `TesseractAnalyzer` over the local chain with the measured
normal axis — the frame solve becomes a boundary crossing into the host
solver carrying DDM sensitivities. The planar demo only, per the spike
ruling. The viewer falls back to an in-process re-solve when handed a
crossed analyzer, since the schema carries no response — a retelling the
backend-agreement suite bounds.

- Experiment 103 with `opensees` + `ec3` reproduces the smax answer to all
  printed digits (0.157469285 t, violations 1.3e-13) in 28 s.
- The jit-plus-Tesseract-plus-openseespy file-descriptor failure did not
  fire: it is a same-process pytest interaction, and an experiment owns its
  process.

### The direct search's geometry went hard

Experiment 103 no longer penalizes: the length floor is now SLSQP inequality
constraints on the member lengths, and a new `fixed_rise` equality pins the
crown at the seed design's form-found rise (3125.0 mm at the seed force
density — form finding overshoots the drawn 3000), each with its own
compiled Jacobian and each switchable from the `simultaneous` section. The
force densities are switchable too: `independent` moves one per member,
`shared` broadcasts a single value — a rise-only search. The report grew the
drawn seed's unsized mass with savings against both baselines, and a
per-member table of diameters and worst-case utilizations, seed against
answer. The viewer path was repaired on the way: each case's loads register
under their own name (a same-named add replaces, so the loads were silently
tearing down the response diagrams), and a case-name typo that tupled the
characters of "uniform" is gone.

- All-free reproduces the penalty-free answer exactly (0.157469285 t): the
  floor never bound, so hardening it changed nothing. Shared with the rise
  fixed pins the single force density and the search reduces to sizes
  (0.165493 t). Independent with both constraints: 0.146853 t, 11.26 % under
  the sized seed, the crown held at 3125.0 mm exactly, violations ~1e-13.
- The ec3 and blueprint backends coincide on this arch's optimum, and that
  is physics rather than a bug: members of roughly a meter at 170–190 mm
  have negligible slenderness, the reduction factor caps at one, and the
  member check collapses onto the cross-section check. The philosophy gap
  reappears where members are slender.

### The re-read is a verb: `compute_utilization`

`AbstractMemberSizer.utilization` was a misnomer beside the field of the same
spelling: `MemberSizes.utilization` is the stored diagonal — the sizing map's
invariant, one by construction — while the method computes demand over
resistance at whatever sizes the caller owns. The method is now
`compute_utilization` on the abstract block and every implementation
(`Ec3Sizer`, `BlueprintSizer`, `TesseractSizer`, `BlueprintClient`, the ASD
seam proof), matching `compute_mass`'s verb. Docstrings were tightened with
it: the shared summary is mode-neutral ("check sizes the caller owns"), the
container's docstring names the method as the computed counterpart of its
stored field, and both backends state the constraint reading a simultaneous
optimization gives it. The field keeps its name — a record is a noun.

### The showcase problem, with the sizes as the optimizer's own variables

`experiments/103_simultaneous_api.py` redesigns 101's arch the other way
around: the diameters join the force densities as decision variables, the
check enters as the inequality constraints `U <= 1` read off
`AbstractMemberSizer.compute_utilization`, and SLSQP with analytic JAX Jacobians
finds the fully-stressed state as active constraints. No envelope (one size
per member, one constraint per case) and nothing to settle (the analysis
runs at the answer's own sections on every iterate). The scaffolding —
config, arch, loads, pipeline with its backend switch, viewer — is imported
from `101_api.py` the way experiment 102 imports it, not copied; `arch.yaml`
grew a `simultaneous: {iterations, tolerance}` section 101 never reads.

- Measured against the same sized-seed baseline (0.165643794 t), the direct
  search beats the nested descent's settled answer under both backends:
  ec3 0.138496697 t (16.389 % vs the nested 0.139297503 t settled) with the
  full traced buckling check in the constraints; blueprint 0.135972798 t
  (17.913 % vs the nested 0.137475215 t settled) through the pure_callback
  rule. The margin is the `∂d/∂q` coupling the frozen-seed gradient omits,
  now priced on every iteration rather than settled at the end.
- Constraints held to 1.9e-12 (ec3) and 1.6e-13 (blueprint); all ten members
  worked to one under their governing case; solves of 83–91 SLSQP iterations
  in 1–2.4 s, compilation excluded.

### The showcase picks its sizer from the file

`arch.yaml` grew `sizing.backend: ec3 | blueprint` and `101_api.py`'s
`build_pipeline` dispatches on it — the one place the standard was named is
now the one place it is picked, both backends drawing tubes from the same
class-limit family. Everything downstream is sizer-blind, including the
trajectory artifact: experiment 102 replays through 101's own `parse_config`
and `build_pipeline`, so a blueprint-recorded search replays through a
blueprint pipeline with no second schema. `Ec3Sizer` keeps `resultant=True`:
on the 2D arch the minor axis is zero so the flag changes nothing, and the
default is the standard as designed rather than the parity distortion.

- Measured, same file, same seed, one key flipped: ec3 reproduces
  0.138951969 t / 16.114 % / +0.24867 % bit for bit; blueprint lands at
  0.137445496 t / 17.023 % / +0.02162 %, lighter because nothing prices
  buckling and cheaper to settle because no buckling length feeds back.
- The two backends START from the same mass (0.165643794 t, identical to
  print precision), and that is a finding rather than a bug: at the seed the
  envelope is governed by the half-span cases, which EC3 sizes on its
  cross-section check — the same eq. (6.2) arithmetic as Blueprints in 2D
  (governing codes [3,3,...] under the funicular case, [2,2,...] under
  both half-span cases). The descent then drives the design into
  compression, buckling takes over, and the standards part ways.

### A non-differentiable code library carries an exact adjoint, two ways

`normax/sizing/blueprint.py` is a new sizer backed by the Blueprints library
(LGPL-2.1, experiment-only, waived 2026-08-15 — it rides in the `pipeline`
dependency group while the prototype lasts). Blueprints' formula classes
subclass `float` and cannot be traced, so the block hosts them behind
`jax.pure_callback` and carries the derivative itself: a hand-derived
`custom_jvp` whose implicit-function-theorem quotient is taken at the root a
host bisection finds. The check is cross-section resistance alone — eq. (6.2)
with the (6.10) and (6.14) resistances, `Form6Dot10`/`Form6Dot14` constructed
per evaluation — because Blueprints implements no §6.3 member buckling and no
classification. A second route crosses a boundary instead:
`tesseracts/blueprint_check/` has no `import jax` anywhere; its
`vector_jacobian_product` endpoint is literal NumPy arithmetic, unlike
`ec3_check`, which delegates to `jax.vjp`. `BlueprintClient` reads both the
diameter and the diagonal utilization off the wire.

- Measured (experiment 12): the four derivative routes (forward, reverse,
  closed-form cubic, central differences) agree to 6.7e-9; routes A and B are
  bit-identical on the sizes and 2.3e-14 apart on the arch mass gradient;
  the EC3 sizer prices the missing buckling at 5.6–7.9% of diameter on the
  funicular arch.
- Measured (experiment 13, the simultaneous mode): with the diameters as
  decision variables and `U <= 1` as SLSQP constraints, the optimizer lands
  on the nested route's settled sizes to 2.9e-9 and holds the constraints to
  9.4e-14; the active constraints are the fully-stressed condition as a KKT
  condition, and the multipliers match the implicit rule's diagonal quotient
  to the 2.1e-2 the analyzer's force redistribution accounts for. Jointly
  over shape and sizes, SLSQP finds 0.020517 t against the frozen-seed
  descent's settled 0.021716 t — the `∂d/∂q` coupling the nested gradient
  omits, priced.
- Measured (experiment 14, the stress test): drive the EC3 sizer's buckling
  length to zero with `resultant=False`, and the two libraries agree exactly
  in tension and to a first-order-in-slenderness residual in compression —
  the class-3 `k_yy = C_m(1 + 0.6 λ n)` leaves `0.6 λ n m`, and the sweep
  measures gap/length constant at 1.63e-5 per mm over nine decades. All
  disagreement between the implementations is §6.3.
- The zero-moment boundary needs a one-sided stencil: a demand moment lives
  on `[0, ∞)`, the host solver now refuses a signed one loudly, and the
  gradcheck probes the boundary with the second-order one-sided difference
  at a widened step (the bisection returns the root to ~2 ulp, which a unit
  step would amplify past the 1e-8 target).
- At an exact end-moment tie the two routes pick different, equally valid
  subgradients: JAX splits across `jnp.max` ties with `abs'(0) = 1`, the hand
  rule routes to `argmax` with `sign(0) = 0`. Measure-zero, documented, and
  kept out of the parity fixtures.
- `tests/test_blueprint_sizer.py` (22 tests) transposes the second-sizer
  template and adds the derivative oracles, the route-B parity, the mode-B
  optimum and the silenced-buckling parity. `BlueprintClient` and
  `blueprint_tesseract` live in `normax/tesseract.py` beside the chain
  (decided 2026-08-15, lax while prototyping), which makes blueprints an
  import of the composition module — it rides in the `pipeline` dependency
  group, and conftest gates only the three test files that import it
  (`BLUEPRINT_TESTS`), so a blueprints-less environment still runs the rest
  of the pipeline suite rather than silently skipping it.
- A review pass hardened the seams: the hand-derived partials are stated
  once per module (`_traced_partials` serving both custom rules;
  `_solved_state`/`_adjoint_state` serving the tesseract's value and both
  derivative endpoints, so they cannot linearize at a different root than
  they solve), the boundary twin enforces the same ratio refusal as the
  local block plus a positive-floor one, the annulus coefficients mirror
  `MemberSections`' own expressions and two new drift alarms pin the floor
  to the EC3 catalogue and the coefficients to the neutral container.

### The designs are inspected in vix, where the analysis lives

`101_api.py` now ends in a `vix.Viewer` holding the initial and the optimized
designs, each with a response per checked load case (deformed shape, `nx` and
`my` diagrams), and returns when the window closes. The gathering goes through
the analysis block's own vocabulary: `frame_model` builds the `smax.Structure`
vix's recipes dispatch on — the same builder the block compiled its assembly
from, at the design's form-found geometry and reconciled sections — and the
new `SmaxAnalyzer.solve_response(xyz, diameters, loads)` reruns the same
injected assembly and solve `member_forces` reads, returning the whole
`smax.Response` instead of the reduced forces. What the viewer draws is the
analysis, not a retelling.

- vix rides in the `viz` dependency group as an editable path dep, **not** in
  `dev` as first asked: CI installs the dev group with bare `pip install .
  --group dev`, which ignores `[tool.uv.sources]` and would look for `vix` on
  PyPI. Same pattern as smax/ec3x in `pipeline`, and as vix's own pyproject
  keeping smax out of its CI group. 101's run line gains `--group viz`.
- The optimized design shown is the settled one — the answer analyzed at the
  sections it demanded of itself — since an inspector wants a self-consistent
  artifact; the reported optimum differs by the measured 0.25% coupling cost.

### The search is recorded, replayed and rendered

The optimization history is now something you can watch. `101_api.py` writes
its trajectory into `artifacts/101_trajectory.npz` with the yaml text of the
run embedded, and the new `experiments/102_animation.py` rebuilds the
pipeline from the artifact alone, replays every iterate through it, and
renders one polyscope frame per iterate under `figures/102_frames/<field>/` —
members drawn as tubes at their reconciled outer diameter (8x exaggeration,
world units, per-edge radii) and colored by worst axial force, utilization at
the reconciled section, governing load case, or diameter, each on one color
scale for the whole animation. It also prints a per-member table at the
answer: diameter, thickness, worst axial force, utilization, governing case.

- **vix was evaluated and rejected for this.** Its dispatch registry accepts
  only smax types, it exposes no generic color-these-edges-by-this-array
  path, and both of its animation modules are docstring-only stubs. Polyscope
  (2.6.1) is used directly, in a new `viz` dependency group — zero coupling
  to the private package.
- `normax/replay.py` (pure JAX, no renderer) holds the artifact roundtrip and
  the replay: a host loop over one `eqx.filter_jit`ted step, exact because
  the search analyzed every iterate at the frozen seed diameters the replay
  hands back in. Measured: the recomputed penalized objective agrees with the
  recorded trajectory to 2.2e-16 relative over all 16 steps of the 101 run.
  The utilization in the history is `AbstractMemberSizer.compute_utilization` re-read
  at the reconciled section, not the fully-stressed diagonal.
- `normax/rendering.py` is "polyscope and nothing else", numpy-only, the
  mirror of `visualization.py`'s matplotlib charter. Camera and scene extents
  are fixed once over the union of every step, so frames do not jitter — and
  the camera must be placed *after* the first structure registers: polyscope
  renders blank if `look_at`/`set_bounding_box` run against an empty scene,
  which is why `frame_camera` is separate from `initialize_scene`. The
  axial-force scale is symmetric about zero only when the sign actually
  changes; an all-compression arch would otherwise waste half the colormap.
- Bug fixed on the way: 101 never passed `sharpness=` to `minimize_bounded`,
  so `Trajectory.beta` recorded 0.0 while the envelope ran at 50 — an
  artifact would have misdescribed its own run and replayed under the hard
  max. The replay refuses mixed zero/positive beta columns loudly.
- 101 refactored minimally for reuse: `read_config` became
  `parse_config(text)` so the exact bytes get embedded, and the block
  construction moved into `build_pipeline(structure, config)` so 102 rebuilds
  what 101 ran instead of copying it. The printed numbers are unchanged.
- Tests: `tests/test_replay.py` (5 tests, in `PIPELINE_TESTS`) covers the
  artifact roundtrip, replay-matches-record under both the smooth and the
  hard envelope, the mixed-beta refusal, and history invariants. Rendering is
  deliberately untested in CI — it needs a GL context runners do not have.

### A grade is named, never defaulted

`SteelGrade` no longer defaults its strengths to S355 — a default strength is
a grade chosen silently. `Steel355` and `Steel235` are subclasses in
`normax.materials` carrying the EN 10025 nominal certificate values (355/490
and 235/360); the modulus and density stay defaults on the base because they
are the same for every structural steel, so `SteelGrade(f_y=460.0, f_u=540.0)`
still works while a bare `SteelGrade()` is refused with a `TypeError`.

- The subclasses keep the full four-field constructor rather than pinning
  their values, because a pytree round trip under `jit` rebuilds a namedtuple
  positionally as `type(x)(*leaves)` — pinned constructors would break every
  compiled pipeline. `tests/test_materials.py` guards the round trip.
- Every former `SteelGrade()` site now reads `Steel355()` (experiments 03, 04,
  08, 09, 10, 101; eight test files; the README quickstart), so no measured
  number moved: 101 reproduces bit-for-bit and the suite is 310 pass (307 plus
  three new grade tests).
- The two Tesseract analysis backends built their grade without `f_u`,
  silently inheriting 490 they never read. Their schema carries no ultimate
  strength, so they now state `f_u=0.0` — loud if anything ever reads it —
  and `tests/test_analysis_prepared.py`'s absurd placeholder states unit
  strengths for the same reason.

### The sizer's conventions freeze at `__init__`

Rafael's call: this is a hackathon and specificity wins, so the alternate
constructors go and their generality is deferred rather than carried.
`Ec3Sizer.at_class_limit`, `Ec3Sizer.from_family` and
`TesseractSizer.at_class_limit` are deleted; the one way to build a sizer is
`Ec3Sizer(structure, family, resultant)`, taking the neutral `TubeFamily` and
converting internally to what ec3x needs (steel via `design_steel`, class via
`section_class_at_ratio`, then the `TubeCatalogue`). The Class 4 refusal
survives inside the classification call. This is the model a SkyCiv or a
Blueprints sizer replicates later: neutral containers in at `__init__`, the
standard's vocabulary derived inside, nothing EC3-flavored accepted as an
argument.

- What `at_class_limit` served — "a bare grade and a class number" — becomes
  the free function `thinnest_family(grade, section_class)` in the same
  module, wrapping Table 5.2's `ratio_at_class_limit` and returning a neutral
  `TubeFamily`. Drivers still import nothing from ec3x; they now name the
  standard twice (family function + block) instead of once.
- The escape hatch the plain catalogue-taking constructor provided —
  non-default partial factors, the cold-formed curve c — is gone with it.
  Supporting general factors and families is explicitly future work.
- Callers swept: `tesseract.py`, experiments 03/04/08/09/101/10, seven test
  files, and the README quickstart, which loses its two ec3x imports outright.
  `tests/test_pipeline.py` now threads a `TubeFamily` as its oracle — its
  `area` arithmetic is stated identically to `ec3x`'s by design, with
  `tests/test_sections.py` as the drift alarm. Full suite: 307 passed, the
  pre-sweep count.

### ec3x is the adapter's import alone

Rafael asked for the ec3x imports to leave `normax/analysis/smax.py` and
`normax/tesseract.py`; with this sweep `normax/sizing/ec3.py` is the only
module in the package that names the standard's library (the Tesseract API
modules under `tesseracts/` are the far side of the boundary and keep theirs).

- **Buckling and stability leave normax entirely** — Rafael's call, in two
  steps within one session: first `Stability` and `frame_stability` (the §5.2.1
  verdict, its utilization, and the member-vs-global slenderness comparison)
  were deleted rather than moved, then the deletion was widened to the whole
  buckling surface — `buckling_modes`, the `Buckling` container, `figure_modes`
  and every eigensolve call site. The clauses live on in `ec3x/stability.py`,
  consumed by nothing here; **buckling and frame-stability checks are future
  work, stated as such in the manuscript** (README `## Limitations` now says
  so). This reverses the extraction doc's "`frame_stability` stays" ruling —
  §2 of `docs/ec3x_extraction.md` carries the dated reversal.
- **What the reports lose**: 09's §5.2.1 verdict block, two-slenderness table,
  `α_cr` measurements, the member-vs-global buckling-length pricing (the x3.26
  entry and the `09_modes.png` figure); 03's per-case `α_cr` table and the
  weakest-factor rows. The member buckling check inside the sizing map (χ,
  §6.3.1) is the submission's core and is untouched, `L_cr` staying an input.
  Thirteen tests went in total (seven `frame_stability`, six buckling-mode and
  mode-figure ones, including the global-mode sizing-cost test whose constant's
  provenance was the deleted eigensolve); 09 keeps its refinement, stagger and
  gradient studies, and the mesh-independent-`L_cr` refinement column stays,
  being about discretization rather than stability.
- **`TesseractSizer` speaks the normax standards** — Rafael: the annotations
  must be superseded by the neutral containers. The plain constructor takes a
  `TubeFamily` (paralleling `TesseractAnalyzer`), backed by the new
  `Ec3Sizer.from_family`, which derives the class from the family's ratio via
  `section_class_at_ratio` — the inverse of the `family` property, possible
  because the class is a function of ratio and grade on the host. The dead
  EC3-typed properties (`steel`, `catalogue`, `section_class`) are deleted;
  nothing outside the module ever read them.
- **The crossed-factor re-read is gone with the acrobatics.** `__call__` now
  reads only the diameters off the boundary and re-checks them with
  `Ec3Sizer.utilization` — the same clauses over the same forces, paired as
  the diagonal (each case's size against that case alone; the first attempt
  passed the whole stack and grew an axis, caught by the parity suite). The
  boundary's own reduction is still compared directly in
  `test_the_moment_factors_survive_the_boundary`. All 34 parity tests pass
  unchanged, tolerances untouched.

`figure_trajectory` draws the objective at every iterate straight from a
sequence of `Trajectory` runs — the first figure to take the optimizer's own
record rather than a numpy spec, so a driver plots what its search returned
with no adaptation. Rafael's calls: trajectories directly instead of
`Descent`, concatenation as a flag on the figure, wired into
`experiments/101_api.py`, and no tests — it is just viz.

- **`concatenated=True` draws successive runs end to end** on one iteration
  axis with a seam line between them, per-run colors and legend surviving.
  Seam positions come from the known run lengths, not from the repeated-`q` or
  `beta`-diff heuristics, neither of which is universal. The repeated
  warm-start row is kept: the step in the objective across it is a real change
  of measure (a sharper envelope or a refreshed seed weighing the same
  design), which `figure_optimization`'s single concatenated curve hides.
- **The sharpness colorbar appears only when every run carries a positive
  `beta`.** Zero is what `minimize_bounded` stamps when the caller had no
  sharpness to give, and it would break the logarithmic color scale —
  `figure_optimization` has this latent hazard; the new figure guards it.
  When drawn, one `LogNorm` spans all runs; per-scatter autoscaling colored
  each run against itself, which a three-round render showed as identical
  colors for betas 10, 70 and 500 under a colorbar describing only the last.
- **`101_api.py` saves `figures/101_trajectory.png`** after its prints; the
  printed contract reproduces bit for bit (`0.138951969 t | 16.114 % |
  +0.24867 % | u 1.000000000000`). Its single unstamped run exercises the
  zero-beta path on real data.

### The extraction's leftovers, swept

A double-check of the two extractions found three real leftovers, all now
closed.

- **`ec3x` moves from the hard dependencies to the `pipeline` group.** The
  extraction doc ruled it "cannot be optional" because `normax/design.py`
  imported `Tube` and `normax/sizing/__init__.py` would keep importing it —
  both premises the sections extraction removed. Only the backends touch ec3x
  now (`sizing/ec3.py`, `analysis/smax.py`, `tesseract.py`), which is exactly
  smax's situation, so it takes smax's placement: a path dependency in the
  `pipeline` group behind the conftest guards (`ec3x` joins
  `PIPELINE_PACKAGES`; `test_materials.py` and `test_sections.py` get their own
  `EC3X_TESTS` guard, needing ec3x but no frame solver). This also unbreaks CI,
  which would have failed its next push resolving `ec3x` from PyPI, where it
  does not exist. Measured: a plain `uv sync` environment — the CI environment —
  collects and passes 124; the full environment passes 321; experiment 11
  passes.
- **`equinox` is declared.** The extraction doc's §6 flagged it as arriving
  transitively; with the pipeline group gone from CI the transit closed, and
  `normax.design` — an `eqx.Module` — failed to import in a dev-only
  environment while every test that would have noticed sat behind a guard.
  It is a core dependency and now says so.
- **Experiment 11 stopped duck-typing the analysis boundary.** It handed
  `ec3x.Tube` seeds to `prepare_model`/`member_forces`, which annotate
  `MemberSections` — same field names, so no test could catch it. The seeds go
  through `neutral_sections` now. This is the standing hazard the sections
  extraction left: the old and new types duck-type, so the annotations are the
  only guard, and only honesty enforces them.
- The three `tesseract_requirements.txt` TODOs now name the
  `ec3x-0.1.0` wheel, tracking ec3x's version bump.
- Flagged, not changed: the Tesseract image names keep their `normax-` prefix
  (they are the pipeline's stage images, and renaming is tangled with the
  deferred wheel TODO); ec3x has no `publish.yml`, per the extraction plan,
  and needs one only if it ever publishes.

### The verification record follows the clauses out

`docs/clauses.md` transfers to `ec3x/docs/clauses.md`, reversing the extraction
doc's keep-it-here call the same day it flagged the awkwardness: a library
whose correctness argument lives in another repository is a strange thing to
hand someone, and ec3x has a life of its own now. `references/` — chiefly the
Eurocode 3 books — moves with it wholesale (gitignored there as here, per the
usage protocol), so the record's own instruction — check the printed page
rather than trusting the transcription — keeps working where the record lives.
normax keeps no copy: the material exists to verify clauses, and the clauses
live in ec3x.

- The clause content is unchanged. What moved with the vantage point is
  attribution: the buckling-length choice, the writeup pointers, and the
  project-brief references are now explicitly normax's, named as the consumer
  the record grew up beside — and the pointers to `tests/test_stability.py`
  and the worked-example files became local again, because those tests live
  beside the record now.
- normax's seven code citations, `ROADMAP.md`'s read-this-first line and the
  local `CLAUDE.md` layout tree point across to ec3x's copy; ec3x's eight test
  files cite it locally again, and its README links it.

### The materials and the sections come home — phase 4, the seam proved

`tests/test_second_sizer.py` fills `AbstractMemberSizer` with a design
philosophy EN 1993-1-1 would not recognize: allowable-stress design, one
safety factor on yield, no buckling, closed-form fully-stressed sizes. Six
tests: the file's imports name no standard library (a self-scanning drift
alarm), the pipeline composes and returns the same containers, the sizer is
fully stressed to 1e-9, the re-read agrees, the mass gradient survives end to
end, and the two philosophies disagree about a compressed arch's sizes —
different standard, not a reimplementation. 321 pass.

- **The Blueprints-backed stretch sizer is cut**, and the reason is recorded in
  `docs/sections_extraction.md` §4: it would need `blue-prints` importable in
  normax, and the ec3x extraction deliberately moved that LGPL dependency out
  of this repository entirely. A redundant proof is not worth reopening a
  settled licensing boundary.

### The materials and the sections come home — phase 3 executed

The blocks take normax types, and the configuration layer stops speaking EC3.
`Ec3Sizer.at_class_limit(structure, grade, section_class)` derives everything
EC3-flavored — the partial factors, the class-limit wall — inside the block,
and its `family` property hands back the wall proportion and the grade as bare
geometry, so a driver builds the analyzer off the sizer it configured
(`SmaxAnalyzer(structure, sizer.family(SEED))`) and one number is never derived
twice. `TesseractSizer` mirrors both; `TesseractAnalyzer` takes a `TubeFamily`;
the analysis backends and the Tesseract backends rebuild their families as
`TubeFamily`, dropping the `classify_section` calls that existed only because
`TubeCatalogue`'s constructor demands a class an analysis never reads.

- **Measured after the gate: nothing moved.** 315 pass; `101_api.py` bit for
  bit; experiment 10's summary identical to the digit; 04 at 2.605e-07; 09
  reprints x3.26 and alpha_cr 0.1291; 02, 03, 08 and 11 PASS.
- **Drivers 03, 04, 08, 10 and 101 import nothing from `ec3x`.** 09 keeps one
  clause constant (`ALPHA_CR_ELASTIC`, the verdict threshold it reports); 01,
  02, 05, 06 and 11 are studies of the standard and keep their deep imports by
  nature. In 101 the whole EC3 surface is two lines: `SteelGrade()` and
  `Ec3Sizer.at_class_limit(...)`.
- **Experiment 08's modulus study was vacuous, and the sweep exposed it.** Its
  `handoff_gap(diameter, steel)` never passed the steel to the solver — the
  varied modulus died in the signature, so "free of the modulus" compared three
  identical runs. The grade now threads through a `graded_section` built per
  call; the three moduli produce gaps equal to eleven digits and different in
  the last ones, which is what a real invariance looks like. The experiment's
  conclusion was right; its probe was not.
- `frame_stability` builds the `ec3x.Steel` it hands the clause functions at
  the call site, from the analyzer's grade — the backend naming the standard it
  reports against, without the analysis layer importing the sizing package.

### The materials and the sections come home — phases 1–2 executed

`normax/materials.py` carries `SteelGrade(f_y, f_u, e_mod, density)` — the
certificate half of `ec3x.Steel`, verified field-for-field against it — and
`normax/sections.py` carries `TubeFamily` and `MemberSections`, the tube
geometry with the class removed. `MemberSizes.sections` is a `MemberSections`
now: `Ec3Sizer` and the Tesseract sizer restate their tubes through
`neutral_sections` on return (the inverse crossing of `design_steel` and
`design_actions` — the standard's terms stripped on the way out), and
`design_envelope` rebuilds the neutral container. `normax/design.py` and
`normax/sizing/__init__.py` import nothing from `ec3x` anymore.

- **Measured after the gate: nothing moved.** 315 pass (309 plus the six
  bitwise-agreement tests of `tests/test_sections.py`), `101_api.py` bit for
  bit, and experiment 10's summary identical to the digit — value 2.22e-15,
  size 1.35e-14, gradient 7.24e-14 — because the seven annulus formulas are
  stated identically in both libraries, which that test file now alarms on.
- **No design carries an integer anymore.** The `section_class` leaf left the
  contract with the retype, so the parity dtype walk lost its "crosses as a
  Python integer" special case and the contract pytree is plain arrays plus a
  four-float grade.

### The materials and the sections come home — planned

`docs/sections_extraction.md` plans the deferred half of the ec3x extraction:
`normax/materials.py` and `normax/sections.py`, so the pipeline contract and
the configuration layer stop speaking EC3. Decided with Rafael 2026-08-15,
same day as the extraction it completes.

- **The trigger was misfiled.** The extraction doc parked `Tube` in the
  contract with "a second cross-section" as its trigger; the real trigger is a
  second *sizer*. A Blueprints- or SkyCiv-backed block behind
  `AbstractMemberSizer` cannot be handed an `ec3x.TubeCatalogue` to fill, so
  the contract as shipped is only fillable by the backend it was abstracted
  from. Counted: `from ec3x.material import Steel` and
  `from ec3x.section import TubeCatalogue` 17 times each across experiments
  and tests, one `ec3x.Tube` each in `normax/design.py` and
  `normax/sizing/__init__.py`.
- **The split follows `ec3x.Steel`'s own seam**: `SteelGrade(f_y, e_mod,
  density)` is the mill certificate, and the gammas and the imperfection
  factor stay where clauses are selected. `TubeFamily(ratio, material)` and
  `MemberSections(diameter, thickness, material)` are tube geometry with the
  class removed — it was only ever in the analyzer's family rebuild because
  `TubeCatalogue`'s constructor demands one.
- **Deliberately not the shape-agnostic property bundle.** The envelope smooths
  `diameter` and `thickness` and relies on their ratio surviving; enveloping
  derived properties independently would let `area` and `second_moment` drift
  off any tube. The bundle keeps its original trigger, a second cross-section.
- `ec3x` itself does not change, and neither Tesseract schema moves a field.

### `normax/ec3/` is extracted into `ec3x` (extraction phases 3–6)

The remaining phases of `docs/ec3x_extraction.md` are executed. The clause
implementation lives at `../ec3x` — flat layout mirroring this repo's template,
dependencies `jax` and `jaxtyping` alone, the same pinned ruff and pre-commit —
and normax consumes it as a hard dependency through
`[tool.uv.sources] ec3x = { path = "../ec3x", editable = true }`. Phase 3 first
moved `diameter_envelope` up into `normax/design.py` beside `design_envelope`:
a soft maximum over load cases is smoothing, not a clause, and the subtree only
lifts cleanly once nothing above it reaches in for arithmetic no standard owns.

- **The gate, measured.** `import ec3x` in its own environment yields `float64`
  (`ec3x/__init__.py` sets `jax_enable_x64` itself — standalone it would inherit
  nothing, and every clause would run silently in float32). `ec3x`: 1587 pass in
  25 s. normax: 304 pass in 54 s against smax `main`, with `--extra spike`.
  1587 + 304 = 1891 = the pre-extraction suite: the plan's 1592/298 split moved
  by the five envelope tests phase 3 kept in normax, plus phase 2's new
  moment-factor test on the staying side.
- **The experiments reprint their recorded numbers.** `101_api.py`:
  0.138951969 t, 16.114 % saved, +0.24867 %, utilization 1.000000000000, bit for
  bit. `09`: the x3.26 penalty and `alpha_cr` 0.1291. `11`: PASS. `04`: mass gap
  2.6e-07 against OpenSees.
- **Experiment 10 was failing before the extraction, and the fix is the one the
  test suite already carried.** Its parity walk held every non-moment field to
  `TOLERANCE_PARITY` 1e-14, and the class-3 diameter sits at 1.35e-14 since the
  measured-plane restructuring — the recorded size-holds-to-the-moments figure,
  reproduced identically on a stashed pre-extraction tree. The walk now holds
  `diameter` and `thickness` to `TOLERANCE_SIZE` 1e-13 like `field_by_field` in
  `tests/test_tesseract_parity.py`, with its own summary line. PASS, worst size
  1.35e-14, worst value 2.22e-15.
- **`blue-prints` left normax's dev group with the oracle tests.**
  `test_oracle_blueprints.py` was the only importer, so the LGPL dev dependency
  is `ec3x`'s alone now, with its never-read-while-writing note beside it.
- **The moved tests cite `docs/clauses.md` repo-qualified**; the verification
  record stays here, being the writeup's evidence, and `ec3x`'s README says so.
- **The ec3_check Tesseract stops depending on normax entirely** — its
  requirements name the `ec3x` wheel alone. How the images obtain that wheel is
  deferred and marked TODO in all three requirements files: a sibling repo's
  `dist/` is outside the build context, so either `ec3x`'s wheel gets built into
  this repo's `dist/`, or the requirement names a published version. The
  `tesseract build` leg of the gate is deferred with it.
- The `uv add`/`uv remove` exact syncs silently dropped the `pipeline` group and
  the `spike` extra from the environment; the suite shrank to 124 collected
  before `uv sync --group pipeline --extra spike` restored it. Worth knowing the
  next time a dependency edit precedes a test count.

### The actions record leaves the contract (extraction phases 1–2)

Phases 1 and 2 of `docs/ec3x_extraction.md` are executed. Phase 1 splits
`normax/sizing.py` into a package — `normax/sizing/__init__.py` carries
`MemberSizes` and `AbstractMemberSizer`, `normax/sizing/ec3.py` carries
`Ec3Sizer` and `design_actions`, with no re-export so a call site names the
standard it sizes against the way an analysis names its solver. Phase 2 drops
`MemberActions` from the contract: `MemberSizes` is `(sections, utilization)`,
`utilization` and `governing` take the `MemberForces` a design already carries
and apply Table B.3 inside the `vmap` they already run, and `frame_stability`
reads `design.forces.axial_force` — its own stage's output — instead of the
check's copy of it.

- **Measured after the gate: nothing moved.** The suite is 1891 pass (1890 plus
  one new test), every tolerance at its recorded value, and
  `experiments/101_api.py` reprints 0.138951969 t, 16.114 % saved, +0.24867 %
  and a utilization of 1.000000000000 bit for bit. The re-derivation is exact
  because it is the same `design_actions` arithmetic that produced the record,
  now run where it is consumed; the reduction is elementwise and stateless.
- **The moment factors left the parity walk with the record**, reaching it only
  through `sizes.actions`. They are now compared explicitly against the boundary
  outputs that still publish them
  (`test_the_moment_factors_survive_the_boundary`), at `TOLERANCE_MOMENT`, the
  tolerance the end moments they are read from are held to. `TOLERANCE_MOMENT`
  keeps its purpose in the walk through `forces.moment_*`.
- **The ec3 Tesseract boundary did not change.** Its `OutputSchema` still
  publishes the factors — that boundary is the EC3 block's own, so EC3
  vocabulary belongs on it — and `Ec3TesseractSizer` keeps rebuilding a
  `MemberActions` from the boundary's factors and recomputing utilization
  locally.

### The `ec3x` extraction is planned, and dated inside the deadline

`docs/ec3x_extraction.md` promotes the `ROADMAP.md` sketch filed as after the
deadline into an execution plan, reconciled against the code as it stands after
P5d and P8. Nothing has moved yet; the document is what gets done.

- **It happens before Aug 31, and what that buys is the claim.** A submission
  arguing that a building code is a normative text rather than a solver is
  stronger when the check is a library somebody could install without ever
  form-finding anything, and weaker when it is a subdirectory of the demo that
  consumes it.
- **The seam was measured rather than assumed.** Every import inside
  `normax/ec3/` resolves to `jax`, `jaxtyping`, the standard library or
  `normax.ec3.*` — 30 internal lines and none reaching the rest of the package.
  `ec3x` needs two dependencies against normax's six, and 1592 of the 1890 tests
  go with it across 15 files, 473 of them the `blue-prints` oracle, which is the
  only file importing it and so leaves the dev group entirely.
- **The ROADMAP's `MemberActions` prescription is wrong, and the correction is
  the one part of the plan a second sizer needs rather than merely benefits
  from.** It cannot "move up": eight annotated parameters and both `custom_jvp`
  tuples in `ec3/sizing.py` consume it, three more in `resistance.py` and one in
  `interaction.py`, so a normax-owned copy makes the library import the
  application. Duplicating a five-field container above the seam only relocates
  the problem, because two of those fields are Table B.3 artifacts and no fixed
  field list is neutral over standards. So the record leaves the contract:
  `MemberSizes(sections, utilization)`, both round-trip methods reading the
  `MemberForces` a design already carries, and `design_actions` moving down into
  the adapter that owns the clause. The tree already said as much —
  `frame_stability` reads the check's copy of an array the analysis produced.
- **`Tube` staying in `sections` is a different case, and stays deferred.** A
  tube is a shape, this project designs tubes, and a second sizer reading the
  same members hands back the same tubes, so its trigger is a second
  cross-section. A Table B.3 factor has no trigger to wait for, being already
  wrong for a second standard today.
- **The contract change is gated alone, before anything relocates**, so a failure
  in it cannot be read as a broken rename. Every recorded tolerance holds at its
  recorded value, and a loosened one is a failed phase rather than a new
  measurement. The move's own gate leads with importing `ec3x` without normax
  present and asserting `float64`: the package inherits x64 today from
  `normax/__init__.py`, and standalone it would inherit nothing, which is the one
  failure mode that changes every number silently.

### The plane is measured, and the analysis holds one tube

Two arguments left the analysis stage. `SmaxAnalyzer(structure, section)` is the
whole constructor now, where it used to be `(structure, catalogue, normal)`.

- **`normal_axis(structure)` reads the plane off the geometry**, and
  `support_fixities(structure)` takes no axis at all. Whether a structure is
  planar is a fact about the structure, so a caller asked to restate it could
  only restate it wrongly. It reproduces every axis that used to be hardcoded:
  the XZ arch measures 1, the straight X-beam measures 1, the gridshell measures
  None. The axes are tried as Y, X, Z, so a straight structure — thin along two
  axes, of which only one may be restrained or it could not bend — keeps its
  vertical plane free.
- **The fixities a planar structure needs are worked out rather than handed in.**
  The normal translation is restrained at every node, without which a pinned
  planar frame is a mechanism in three dimensions and the solve returns nan; the
  two out-of-plane rotations are restrained at the supports, without which a
  straight one is a mechanism even then. The in-plane bending rotation is never
  restrained anywhere, so a base still carries no moment. The rotations are held
  wherever the structure is planar, curved or straight: in-plane and out-of-plane
  response decouple exactly in a planar frame, so the restraint the curved case
  does not need cannot reach its results either, and making it unconditional
  keeps one rule instead of two.
- **OpenSees keeps a declared normal, and now refuses one that disagrees.** Its
  two in-plane axes become the solver's own axes, so the declaration is load
  bearing there rather than a restatement; `frame_plane` measures the geometry and
  raises when the two disagree.
- **The analysis is configured with one tube; the check is handed the family.** A
  frame is built out of the sections it has, and choosing among sizes is the
  check's business, so a catalogue in an analyzer was an offer of sizes nothing in
  a solve had any use for. What the tube carries onward is its wall proportion,
  its grade and its class, and the family those three define is rebuilt from it
  wherever a traced diameter has to be walled — in `_injected_assembly` and in
  `frame_stability`. `TubeCatalogue.__call__` stays the one place a wall is chosen
  for a diameter.
- **The `smax` Tesseract backend prepares its assembly from the primal
  coordinates.** Measuring a plane needs concrete coordinates, and a derivative
  endpoint's are traced, so `prepare_assembly` runs once on the inputs a caller
  sent and `solve_assembled` is what the trace enters. Preparation was already
  meant to sit outside the trace — compiling reads support flags with a Python
  conditional — and now it provably does.
- **A size is held to what it reads, which is the moments.** The field-by-field
  Tesseract parity moved `sizes.sections.diameter` to 1.17e-14 in class 2 and
  1.35e-14 in class 3, a hair above a `TOLERANCE_PARITY` of 1e-14 that the sizes
  had been sharing with the geometry. The cause was measured rather than guessed:
  substituting one route's moments into the other collapses the disagreement to
  9e-16, and substituting the axial forces alone leaves it exactly where it was, so
  a size inherits the funicular moment's near-cancellation, damped by the small
  share of utilization that moment claims. The check itself crosses exactly — one
  sizer fed both force sets reproduces the same figure, and the stage-alone tests
  still hold a diameter to 1e-14 on identical forces. `TOLERANCE_SIZE` is
  therefore 1e-13, two orders tighter than the moments it descends from. The
  geometry crosses bit-identically either way.

### The driver's mass has a floor under the shortest member

`experiments/101_api.py` penalizes its objective with
`normax.optimization.penalized_mass`, and `experiments/arch.yaml` describes the
floor: `length_floor: {fraction, sharpness, weight}` under `optimization`.

- **The saving it reports is now a design rather than a collapse.** Floored:
  **0.138951969 t from a start of 0.165643794 t, a saving of 16.114%**, shortest
  member 607.8 mm against a 600 mm floor and **none of the ten members under it**.
  Unfloored, the same run with `weight: 0.0`: 0.138056811 t and 16.654%, with
  **2 of 10 members below the floor** and the shortest at 543.2 mm. The extra
  0.54 percentage points were bought by shortening members, not by improving the
  form, which is why the floored number is the one to quote.
- **The floor is stated as a fraction of the nominal bay**, `span / num_edges`, so
  one file means the same thing at any span or discretization. 0.6 of the bay, as
  in `experiments/03_optimize_arch.py`, whose annealed 60-iteration descent shows
  the unconstrained case collapsing far harder — 16 of 20 members below the floor
  at a length ratio of 105.
- **The penalty is why the objective and the mass are now printed apart.** The
  search reads the penalized value and the design weighs what it weighs; at the
  answer they differ by 6e-6 relative, the smooth minimum understating the true
  shortest member by a little and so biting slightly early. `settle_diameters`
  reads only the design the objective hands back, so penalizing the value leaves
  the coupling measurement alone.
- **The six experiments broken by the P8 and P5d restructurings run again**, and
  each reproduces the numbers recorded for it rather than merely importing: `03`,
  `04`, `08`, `09`, `10`, `11`. Two API gaps had to be closed by composing blocks
  by hand rather than by renaming: a pipeline no longer takes a
  `buckling_length=`, which `09` needs to price the member-length assumption, and
  per-case sections live only on an unenveloped design, which is what `03` reads
  to say which load case governs where.

### What a frozen analysis seed costs is measured rather than assumed

The frame analysis wants sections before the check has produced any, so it is run
at a seed diameter and the two are coupled: `d = C(d)`. Two functions in
`normax.design` address that coupling, and the driver uses the cheaper one.

- **`settle_diameters(objective, params, settling_passes=..., settling_tolerance=...)`
  closes the coupling at fixed force densities**, by forward analyses alone, and
  returns the diameters an analysis at those force densities asks of itself.
- **`experiments/101_api.py` runs a plain bounded descent and calls it once at the
  answer.** The frame is analyzed at the 100 mm seed for the whole search, so the
  descent is a single self-consistent minimization in the force densities, and what
  the shortcut cost is then a number rather than a caveat: **0.138056811 t at the
  seed against 0.138367347 t re-analyzed at its own sections, so the seed
  understates the design by +0.22493%**, from a start of 0.165643794 t and a saving
  of 16.654% with both ends weighed the same way. Utilization is 1.000000000000
  either way. Those are the unfloored numbers, which the driver now reaches only at
  `weight: 0.0`; floored it is +0.24867% on 0.138951969 t.
- **`optimize_staggered` is the alternative, and it is not what the driver runs.**
  It alternates descents with settling passes until the force densities stop moving,
  and on the arch it closes in **2 rounds, 16 gradient calls and 31 forward
  passes** — `|Δq|∞` of 1.1e-5 after the first round. It buys a mass reported at
  the design's own sections; it does not buy a better design, because its gradient
  is taken at frozen sections too and so omits `∂d/∂q` either way. For a first
  milestone that trade is not worth the machinery, so the routine stays in the
  library under test and the driver stays a descent.
- **Both live in `normax.design`, not `normax.optimization`.** Closing the coupling
  needs a diameter read off a design, and `normax.optimization` knows what a
  descent is without knowing what a design is — it imports nothing from the
  package and stays that way. `minimize_bounded` needed no change: its evaluation
  cache, its visited set and its post-hoc re-evaluations are all locals of one
  call, so each round is an independent, self-consistent search.
- **A staggered round is a whole descent, because refreshing per evaluation breaks
  the search.** Measured: with the seed moving inside the objective, L-BFGS-B
  stopped after 4 evaluations at the start point reporting `CONVERGENCE: RELATIVE
  REDUCTION OF F <= FACTR*EPSMCH`. A line search compares values of one function,
  and a seed that moves inside it hands the search a different function at every
  trial; the curvature accumulated from differences of those gradients is
  contaminated the same way.
- **Settling is forward passes rather than one coupling pass per descent.** Sizing
  is a contraction in the diameters but a slow one, measured at a ratio of 0.85 per
  pass near its fixed point on `experiments/arch.yaml` and the same whether the
  force densities move or not. One pass per descent left the residual at 1.8e-5
  after 12 descents and still falling by 15% a round, which is gradients spent on
  what a forward evaluation resolves.
- **A raise rather than a returned residual, so what comes back needs no
  checking.** Every cap is the caller's budget: `settling_passes` and
  `settling_tolerance` sit in the `optimization` block of `experiments/arch.yaml`
  beside `iterations`, and `rounds` is a keyword of the staggered search the driver
  no longer passes. The arch is given 400 passes at 1e-6, against the 31 it spends.
- **The sign of that error is not predictable, which is why no test asserts it.**
  Mass rises monotonically with a *uniform* seed — 0.082435 t to 0.083738 t as the
  seed goes 25 mm to 400 mm on the test arch — but a settled seed is not uniform,
  so it is redistribution rather than level that decides: a member analyzed fatter
  than it is attracts force. On the one-variable scan the gap changes sign, +0.32%
  at `q = −100` against −0.04% at `q = −3.16`.
- **It misreports the mass rather than moving the design, as far as one variable
  can show.** Scanned with a single force density shared by every member, the
  frozen and the settled curves are minimized at the same `q` — but that `q` is
  the bound, so what is shown is that a frozen seed does not move a boundary
  optimum, not that it never moves one.
- **Weak typing had been compiling the pipeline twice.** `jnp.full(members,
  100.0)` is weakly typed where the diameters a check returns are not, and the two
  are different abstract values however equal their contents, so the first round
  and the second were different programs. The seed is restated at its own dtype on
  the way in. Two traces remain and are asserted: value-and-gradient for the
  descent, value alone for the settling passes, neither once per round.
- **`update_parameters` is deleted from `experiments/101_api.py`.** Nothing has to
  rebuild the container now, and the start-of-run gradient differentiates the
  container itself and reads its force-density leaf.
- **The omission is stated in the README, under `## Limitations`.** It had been
  measured and argued here without being said anywhere a reader of the submission
  would see it: the gradient is exact for the mass at a frozen seed rather than for
  the mass of a self-consistent design, and closing it means a fixed point
  `d = D(N(d), L)` inside the pipeline, backpropagated end to end.

### A section family carries its grade and its class

`TubeCatalogue(ratio, section_class, material)` is a parametrized cross-section
generator, and `catalogue(diameter)` generates the tube:
`Tube(diameter, thickness, material, section_class)`. A grade is no longer handed
in beside a family, and a class is no longer threaded past one.

- **The family's defining number already knew both.** A catalogue's ratio is
  `ratio_at_class_limit(f_y, section_class)` — a function of the grade and of the
  class — so carrying them separately let a grade travel beside a ratio derived
  from a different one. 79 calls dropped an argument; `Ec3Sizer(structure,
  catalogue)` and `SmaxAnalyzer(structure, catalogue, normal)` take four
  arguments each counting `self`.
- **Where the merge stops is a rule, not a judgment.** A function that takes a
  catalogue or a tube drops `steel` and `section_class`; a function that takes a
  bare section property keeps them. So `utilization_design(tube, actions,
  buckling_length)` loses both while `resistance_compression(area, steel)` keeps
  both — `N_pl,Rd = A f_y / γ_M0` is shape-agnostic, and narrowing it to a CHS
  container would narrow 30 clause docstrings to a shape no clause names.
- **`MemberSection(tubes, material)` is deleted.** It existed to pair a section
  with its grade, which is what a generated tube now is; keeping both would put
  the material in a design twice. `MemberSizes.sections` is a `Tube`, and
  `compute_mass` reads `sections.material.density * sections.area` off it
  unchanged.
- **Both containers stay `NamedTuple`s, and the class is a `SectionClass` — an
  `int` subclass registered as a static pytree node.** It compares, branches,
  formats and does arithmetic as an integer does, so no clause knows it exists;
  what it changes is that it flattens to no leaves and travels in the tree
  structure, so `jit`, `grad` and `vmap` hand it back as the same Python integer.
  The constructors coerce, and `_make` is overridden so `_replace` inherits the
  coercion instead of routing around it — a class that reached a container as a
  bare integer would be a leaf again.
- **What forced that, measured rather than assumed: JAX traces a `custom_jvp`'s
  primal arguments under `jit` even when the container reaches it as a closed-over
  constant.** `_solved_diameter` is that `custom_jvp` and `experiments/101_api.py`
  jits its objective, so it is the production path rather than a corner. Probed on
  the real container, a bare `int` leaf arrives concrete when eager and under
  `jax.grad`, and as a `DynamicJaxprTracer` under `jax.jit` — where the branch it
  selects raises. Before this change the class was `nondiff_argnums=(0, 1)`, which
  is precisely what had been keeping it out of the trace.
- **A class held as a leaf and returned through `jax.jit` was worse than that: it
  was silently wrong.** It comes back a rank-zero array, and `Array(3) in (1, 2,
  3)` compares elementwise and is false for every class, so a Class 1 family read
  through a jitted output would have taken the elastic modulus with no complaint.
  The container is what stops the case arising; `_validate_class` converting first
  is the second line, correct where the value is concrete and a loud
  concretization error where it is a tracer.
- **Two namedtuple traps, both verified and both avoided.** `typing.NamedTuple`
  forbids overriding `__new__`, so the two containers subclass a
  `collections.namedtuple` base and annotate their fields bare. A default written
  beside such an annotation would shadow the field's own descriptor and make every
  instance report the default whatever it was built with, so `diameter_min`'s
  default lives in `__new__` alone.
- **`at_class_limit` stays a classmethod and takes a grade.** A module could
  derive the ratio in `__init__`, but the ratio has two origins: derived at a
  class limit in every production call site, named outright in the class sweep,
  the gradient tests and the geometry fixtures that model a published section. One
  constructor cannot read as both.
- **`verified_class()` is where a label is checked, and a block is what calls
  it.** A class named beside an explicit ratio can contradict it, and letting the
  plastic clauses onto a Class 3 wall is unsafe, so `Ec3Sizer.__init__` reads its
  class through this rather than off the field. Verified where the ratio and the
  grade are concrete and trusted where they are not — under a tracer there is
  nothing to compare, and every family a block is built from is built on the host.
- **The envelope names the two geometric fields instead of mapping over the
  tube.** `jax.tree.map(jnp.max, tubes)` was legal only while a tube was exactly
  two leaves that both carried the load case axis. `design_envelope` now
  envelopes the diameter and the wall and carries the grade and the class
  through — three lines for a claim that is more honest than the one it replaces.
- **The wire format does not change.** `f_y`, `ratio` and `section_class` already
  crossed the ec3 schema as separate scalars and are reassembled on the far side.
  The two analysis backends have no class on the wire and need none, an analysis
  reading geometry alone; they classify the ratio with `classify_section`, which
  reports Class 4 rather than refusing it, so a family the clauses cannot check is
  still analyzable and the refusal waits for a clause to read the label.
- **No number moved.** The arch of `experiments/101_api.py` reports 0.138056811 t
  at 16.654 % saved and a utilization of 1.000000000000; Tesseract parity reads
  6.67e-16 on the mass and 7.32e-14 on its gradient. 1882 tests pass.
- **One test's meaning was a real question, and both halves are kept.**
  `∂d/∂f_y` used to be taken at a frozen wall, since the catalogue was fixed while
  the grade moved. That test now names its ratio explicitly and still asserts a
  negative sign; a second test takes the derivative through a family built at a
  class limit, where `∂ratio/∂f_y` joins and a stronger grade raises the capacity
  and thins the wall at once — so it asserts finiteness and the difference
  quotient and **not a sign**.
- `experiments/05_class_ratio_sweep.py` loses `ClassBranch`, which paired a class
  with the family whose ratio sits at its limit. That pairing is what a catalogue
  is; `behavior_of(catalogue)` is the one thing the container knew that a family
  does not.

### The envelope sharpness moves to the search that anneals it

`experiments/arch.yaml` carries `envelope_sharpness` under `optimization:` rather
than `analysis:`, and `OptimizationConfig` holds it beside `iterations` and
`bounds`.

- **It belongs to no stage, which is why it was awkward under either.** The
  analysis reconciles nothing and the check reads one load case at a time, so no
  block sees a sharpness; `design_envelope` sits above all three. `sizing:` would
  have been the more misleading of the two homes, putting smoothing back inside the
  standard that P8 hoisted it out of.
- **What it is, is a continuation parameter.** The envelope approaches the true
  largest size as the sharpness grows, so raising it across rounds drives the
  design onto the smallest adequate one from above — a property of the descent,
  sitting beside its budget.
- `read_config` now builds `OptimizationConfig` by keyword, so a missing or
  misplaced field there is refused with the `TypeError` its docstring promises
  rather than a `KeyError`. It was the one section still built positionally.

### The suite runs across every core

`pytest-xdist` is a dev dependency and `pyproject.toml` carries the same
`[tool.pytest.ini_options]` block `jax_fdm` uses: `-n auto --dist loadfile`, with
`-v` because bare progress dots are unreadable once workers interleave. Plain
`uv run pytest` is now the fast path and `-n0` serializes for debugging.

- **Measured, on ten cores: 159 s → 30 s.** Two changes compound. `tests/conftest.py`
  enables JAX's persistent compilation cache, which took a run from 159 s to 107 s
  cold and 62 s warm — the suite triggers some 2800 distinct compilations and was
  repeating them every run. Fanning out across cores takes the warm run to 30 s.
- **`--dist loadfile` is not only about legible output, and splitting files is
  slower.** Measured: `loadgroup`, which lets a file spread over several workers,
  runs in 36 s against 30 s and burns 727 % CPU against 434 % — every worker that
  receives part of a file rebuilds that file's module fixtures and recompiles its
  programs, and that costs more than the extra parallelism buys. Pinning a file is
  also what two files need on their own terms: the Tesseract tests serve containers
  through one chain, and OpenSees holds a single global model with no handle to it.
- **One file now bounds the wall clock.** `test_pipeline.py` is a third of the
  suite by itself at some 20 s, so nothing can finish sooner than it does. What is
  left to win there is inside that file: `sized()` and `covered()` are plain
  functions called 51 times, each re-running form finding, three analyses and three
  bisections eagerly.

### The search hands back the design it answered with

`minimize_bounded` and `optimize_annealed` return `SearchResult(value, aux,
trajectory)` instead of a bare `Trajectory`. An objective built with
`has_aux=True` returns `(value, pytree)`, and the search carries out the pytree
belonging to the point it answered with.

- **What it removes is a second compilation, not a forward pass.** Rebuilding the
  answer's design outside the search — `pipeline(params_opt, loads)` in
  `experiments/101_api.py` — traces and compiles a program the search had already
  run hundreds of times, and eagerly at that. The objective already computed the
  design; returning it costs the traffic and nothing else.
- **The optimizer still knows nothing about designs.** To this module the aux is
  a pytree that rode along, so the bowl objectives in `tests/test_optimization.py`
  are unchanged and `aux` is None whenever the objective returns a value alone.
- **Only the newest aux is held.** A pytree per evaluation is a design per
  evaluation, and the answer is not known until the search is over. The answer is
  usually the point evaluated last; when it is not, one call recovers it — tested
  as at most one evaluation more than the same search carrying nothing.
- Call sites take `.trajectory`: `experiments/03_optimize_arch.py`,
  `experiments/04_backend_agreement.py`, and the driver tests.

### Every stage owns its own contract

`normax/design.py` is down to six things — `DesignParameters`, `Design`,
`StructuralDesignPipeline`, and the three questions asked *of* a design. Each
stage's container and abstract block moved to the stage itself:

| module | container | contract |
|---|---|---|
| `normax/form_finding/__init__.py` | `FormFoundShape` | `AbstractFormFinder` |
| `normax/analysis/__init__.py` | `MemberForces` | `AbstractFrameAnalyzer` |
| `normax/sizing.py` | `MemberSizes` | `AbstractMemberSizer` |

- **`normax/form_finding.py` is a package**: the contract in `__init__.py`, the
  force density method in `fdm.py`. That is what keeps `jax-fdm` out of
  `normax.design` — verified, `import normax.design` leaves `jax_fdm` unloaded —
  and out of the T2 container, whose requirements are still the wheel and `smax`.
- **`AbstractFrameAnalyzer.__call__` takes `xyz`, not a `FormFoundShape`.** Both
  analyzers only ever read `shape.xyz`; taking the container would have made the
  analysis contract unreadable without the form finder's dependencies for no
  gain.
- **`normax/pipeline.py` became `normax/stability.py` and then dissolved.**
  `Stability` and `frame_stability` live at the bottom of
  `normax/analysis/smax.py`, the only backend that can answer them: the check
  needs an eigensolve, and the OpenSees one cannot trace one.
- **`node_positions` is deleted.** It was `equilibrium_state(...).xyz` with a
  name on it. `positions_vertical` stays — it is the only executable form of the
  argument that a held plan is funicular under uniform force densities alone.

### The envelope reconciles and nothing else

`design_envelope(design, sharpness)` lives in `normax.design`, takes no sizer,
and reads no standard.

- **`MemberSizes` gained `utilization`**, filled by the check inside the same
  `vmap` that sizes. It is a **diagonal, not a matrix**: entry *(i, m)* is member
  *m* under case *i* at the section **case i** demanded, so it is one by
  construction — measured at 0.9999999999999939 to 1.0000000000000024, which is
  the bisection's convergence and the §6.5 invariant stored rather than asserted.
  Below one only where the catalogue minimum bound.
- **It is never re-derived**, so the envelope carries it through and utilization
  is computed exactly once per design. Reading it as a verdict on a reconciled
  design is the mistake the docstring warns about.
- **`governing_load_case` reads the demanded diameters**, not a utilization.
  `argmax` over the sections each case asked for gives the same answer, since
  capacity is strictly increasing in the diameter: at the reconciled section the
  case that demanded the largest is exactly satisfied and every other is at a
  section larger than it asked for. Exact at the true largest, and measured to
  agree at β = 50. That is what makes the second re-check unnecessary rather than
  merely cheap.

### A design is one field per stage

`normax/stages.py` is `normax/design.py`, and it holds what the name says:
`Design(shape, forces, sizes)`, one field per block in the order they ran and
nothing that no block produced. `MemberSections` is gone with the eight fields
it flattened into one place, and `DesignPipeline` is `StructuralDesignPipeline`.

- **`MemberSizes(sections, actions)` — a check answers two questions.** What it
  decided is a section; what it read is not what the analysis reported, because
  reducing two end moments to a design moment and an equivalent uniform moment
  factor is EN 1993-1-1 Table B.3. That reduction is lossy — under double
  curvature the factor floors at 0.4, a factor of 2.5 on the interaction term,
  and it cannot be recovered from the design moment alone. For `TesseractSizer`
  the clause is applied on the far side, so the actions are the only record of
  what that block read, and recomputing them locally would substitute our
  Table B.3 for theirs.
- **`MemberSection(tubes, material)`** lives beside `Tube` in
  `normax/ec3/section.py`. A section and its material travel together because a
  mass needs an area and a density and neither alone answers what a member
  weighs per metre — the one quantity a published section table states that no
  clause decides. `compute_mass(design)` replaces `calculate_mass` and needs no
  block: every term of `ρ Σ A L` is already on the design.
- **`FormFoundShape` carries its lengths again**, so the pipeline holds no
  connectivity and needs none. `AbstractFormFinder.member_lengths` and
  `AbstractMemberSizer.mass_per_length` both leave the protocol with it.

### The load case envelope is hoisted out of the pipeline

`size_envelope(design, sharpness)` in `normax/sizing.py` reconciles the load
cases and returns the same `Design` with one section per member. The pipeline
decides nothing; `sharpness` leaves `DesignParameters`, being a schedule knob
that no optimizer varies.

- **What it buys is measured, not aesthetic.** The expensive half of a pipeline
  evaluation — a form-find, three frame solves, three root-finds — does not
  depend on the sharpness. `experiments/06_load_case_aggregation.py` sweeps
  seven of them, which cost seven full evaluations and now cost one and seven
  `logsumexp` calls.
- **The tube is enveloped leafwise and the wall follows exactly.** The envelope
  is scale-equivariant in the logarithm, so `envelope(d)/r` and `envelope(d/r)`
  agree to 2.2e-15 and the `d/t` ratio survives a `jax.tree.map` over the whole
  tube — measured at 3.55e-14 on the arch. No catalogue is needed to re-derive a
  wall.
- It is also the step toward the simultaneous formulation: replacing "envelope
  the per-case requirement" with "take `d` as a variable and add `u ≤ 1`" is now
  a change in the loss alone.

### The check runs batched

`design_actions(forces)` takes one load case, is elementwise over members, and
is `jax.vmap`ped rather than indexed. `Ec3Sizer.__call__`, `governing` and
`utilization` all lost their Python loops over load cases, the root find
batching cleanly through its `custom_vjp`. **Bitwise identical to the loop**,
max gap 0.0, checked before the loops were removed.

### A block is built from a structure, and holds only what it reads

The two-call `compile(structure)` dance is gone. A block's constructor takes the
structure and settles whatever that software can settle from it, so building one
is the host-side step and `__call__` is the traced one. `DesignPipeline` is three
blocks and nothing else — it takes no structure, and it asks the form finder for
a member length rather than measuring one.

- **No block keeps a structure it does not read.** `FdmFormFinder` keeps
  `xyz_fixed`, a `(supports, 3)` array, because the graph already holds the
  topology and the supported coordinates are the only thing it cannot; that
  forced `equilibrium_state` to declare what it actually depends on, and its
  second parameter is now `xyz_fixed` rather than a whole `Structure`.
  `SmaxAnalyzer` dropped a `structure` field nothing read. `Ec3Sizer` keeps one
  deliberately, a code check having no topology to settle and saying so.
- **`member_lengths` is on the form finder protocol**, beside the sizer's
  `mass_per_length` and for the same reason. A mass is `ρ Σ A L`: the section
  family supplies the `A` and only the block owning one can, the connectivity
  supplies the `L` and only a block owning one can. Two blocks answering the two
  halves of one product, and a pipeline holding neither.
- **`FormFoundShape` is one field, `xyz`.** The forces went first — they are
  bitwise `q * lengths`, a product of two things the caller already holds — and
  the lengths followed. The T1 Tesseract schema lost both outputs with them; a
  schema that carried either would be paying a round trip for arithmetic.
- **Every member buckles over its own length, and nothing can say otherwise.**
  `DesignPipeline.__call__` takes no buckling length and `MemberSections` reports
  no separate one, since with the assumption fixed it was a second copy of
  `lengths`. The clause layer is untouched and still takes `L_cr`, Eq. 6.50 being
  written in it. Recorded in `docs/clauses.md` as deferred rather than settled:
  it presumes every node is braced by structure outside the model, and where that
  fails the design is unsafe rather than cautious.
- **`DesignParameters` carries the envelope sharpness**, so an annealing schedule
  parameterizes one compiled program instead of compiling one per round.

### One load case, one container, one axis

`normax/loads.py` owns loads and everything that knows what a load case axis
means. `smax` now accepts a dense nodal array wherever it accepts a `LoadCase`,
so a load case is a `(nodes, 3)` array from the generator to the solve.

- **The load acrobatics in the smax backend are deleted**: the `PointLoad` list
  built from zeros, the `_load_case` that scattered three translations into a
  compiled channel, and the `Model` container whose only other field was the
  assembly. `prepare_model` returns a `CompiledStructure` directly. No
  `PointLoad` or `LoadCase` is constructed anywhere in the repo.
- **`LoadCaseForces` and `MemberForces` were one container at two ranks**, born
  of a name collision. `MemberForces` prevails with a variadic axis
  (`"*load_cases members"`): a solver answers one case without it, a block
  answers all of them with it.
- **`stack_load_cases`, `select_load_case` and `count_load_cases`** replace four
  hand-rolled copies of the same indexing — `stacked_actions`,
  `load_case_actions`, two inline stacking blocks in the analyzers, and four
  `X.axial_force.shape[0]` counts. All three are `jax.tree` one-liners, generic
  over the container because none of them reads a field name.
- **A load case magnitude is a total, not a nodal force.** `create_loads_by_name`
  calls its generator at unit load for the shape alone and rescales, which is
  what makes a case on half the span comparable with one on all of it.

### Smaller things that were two things

- **`SteelGrade` is `Steel`.** The `Material`/`Steel` pair could not work:
  a `NamedTuple` subclass inherits no fields, so `Steel()` raised and
  `Steel("S355", 100.0).f_y` silently returned the default.
- **`value_and_gradient` compiles with `jax.jit`**, not `eqx.filter_jit`. Every
  caller passes arrays only. The docstring's warning about Python floats
  compiling a program each was a `filter_jit` fact and is now false — verified,
  three Python floats through one jitted function trace once.
- **`prepare_model` was handing `smax` a `normax.Structure`** where it wanted the
  frame it had just built on the line above, and discarding the frame. Fixed to
  `compile_structure(frame)`.

### What the arch driver actually shows

`experiments/101_api.py` composes the three blocks from `experiments/arch.yaml`
and takes one gradient across all of them. The gradient is real — it agrees with
central differences to 2e-9 — and the Tesseract swap changes the mass by 6.7e-13.

**The descent is not a design.** It converges in ten iterates with eight of ten
force densities pinned to the bound and two members stretched to 3.9 m: the
member collapse `penalized_mass` exists to prevent, which this driver runs
without. A 1e-16 gradient difference moves the endpoint between degenerate
basins, so the mass it reports is the collapse and not an optimum.

### One pipeline, three swappable blocks

The API is three objects and a composition, and the duplication that came of not
having them is gone. `normax/design.py` states what a block is, and
`DesignPipeline` holds one of each and composes them.

- **`normax/composition.py` is deleted.** It ran the same three stages across a
  Tesseract boundary and duplicated `ProblemSetup`, `design_members`,
  `total_mass` and `design_envelope` to do it — about 450 lines that existed only
  because the chain could not be expressed as blocks. `normax/tesseract.py`
  replaces it with three blocks and nothing else, and
  `tests/test_tesseract_parity.py` now runs **one** `DesignPipeline` over two
  sets of blocks rather than comparing two implementations of one pipeline.
- **`Design` and `Envelope` collapse into `MemberSections`.** Load cases reach a
  block stacked, so a single case is one entry and the second container was never
  a different thing. `unsmoothed_design` goes with them: the design at the true
  largest is the same call with no sharpness.
- **The block API reproduced the free functions bitwise** before they were
  deleted — geometry, actions, required sizes, enveloped diameters, utilization
  and mass, on one load case and on three. Only the gradient differed, by 4e-12
  relative, and the two *old* spellings differ from each other by the same amount
  under `jit`, so that is XLA association and not the composition.
- **A structure no longer carries a load.** `Structure` is topology, geometry and
  supports; `normax.design.LoadCases` carries the case a shape answers to and the
  cases it is checked against, and `loads` is a required argument everywhere it
  was a `None` default. A structure asked to survive several cases has no
  business owning one of them.
- **The cross-section class is derived, never given.** `Ec3Sizer` reads it off
  its own catalogue when it is built, on the host, where the yield strength is
  concrete and the Class 4 refusal can fire. It can no longer contradict the wall
  it is checked with.
- **A mass no longer crosses the ec3 Tesseract boundary**, and neither does the
  length it would be multiplied by. Mass is `ρ Σ A L`, geometry rather than a
  resistance, and EN 1993-1-1 has no opinion on it. What the standard decides is
  the size; `calculate_mass` multiplies it by a length outside. The cotangent
  reaches the stage through the diameter instead, so the gradient is unchanged.
- **A single load case is passed through rather than enveloped.** An envelope
  over one case is the identity in exact arithmetic and a logarithm followed by
  an exponential in floating point, so taking it cost the last bits of a size for
  nothing. That is what makes the single-case parity bitwise.
- **`filter_jit` on a Tesseract-backed pipeline closes a file descriptor** once
  the OpenSees backend has run in the same pytest session, and every later test
  then errors out of capture rather than failing an assertion. The parity test
  leaves the composed side eager for that reason; `experiments/10` compiles it
  and is unaffected, running in its own process.

### A straight beam, and the mechanism it found in the supports

`experiments/11_straight_beam_benchmark.py` is the arch of experiment 03 with its
z coordinate set to zero — same span, same twenty members, same 180 kN, same
supports. **Two stages rather than three**: no form finding, no force density, so
what is exercised is the T2 to T3 handoff alone. One load case, because a beam is
already in pure bending and a second would move numbers without testing anything.

- **It is the benchmark because both stages have a written-down answer.** The
  moment diagram is the statics of a simply supported beam under equal point
  loads, matched to 1.5e-12 scaled. The size is a cube root — with no axial force
  the check is bending stress alone and every section modulus is a monomial in
  the diameter — matched to 1.6e-15, with the unit-diameter modulus taken from
  `normax.ec3.section` rather than restated. Axial force is **exactly zero**.
- **The staggered coupling is exact here**, mass bit-identical across four passes
  against about 1.2% on the arch. A determinate structure carries the same forces
  whatever its sections are. The difference is indeterminacy, not the code.
- **Its tolerances are mesh-dependent and pinned with headroom.** The residue is
  the conditioning of the linear solve: 1.8e-13 at ten members, 1.5e-12 at
  twenty, 1.0e-11 at forty, 1.2e-10 at eighty. A first version pinned 1e-12 on
  the ten-member floor and failed as soon as the mesh doubled.
- **`support_fixities` was wrong for a straight planar structure, and the beam is
  what found it.** Restraining the normal translation removes the rotation about
  the line joining the supports only when the members lie off that line. A beam
  lies on it, so the mode survives as a uniform twist and the first run died on a
  singular stiffness matrix; `diagnose_mechanisms` named it exactly, one zero
  eigenvalue with `rx` at 1.0 on every node. The two out-of-plane rotations are
  now restrained **at the supports**, which is what a bearing does. Pinned and
  never fixed is a rule about structures that occupy all three dimensions; a
  planar one deviates from it because otherwise it is a mechanism, and the
  rotation the in-plane bending happens about is still free everywhere. Verified
  inert for the arch: mass, `alpha_cr`, all four buckling factors, the
  slenderness table and the refinement study are byte-identical.
- **Figures**: `11_benchmark.png` and `11_profile.png`, the latter drawing the
  beam in elevation at its required depth **to scale** rather than exaggerated,
  216 mm at the supports to 376 mm at midspan. `figure_sections` degenerates for
  a flat structure, both elevation panels collapsing to a line, so
  `figure_beam_profile` was written beside it.

### Experiment 09 was eager, and it cost 47 of its 52 seconds

Experiments 03 and 10 compile the calls they make in a loop; 09 never did, so
every call ran eager and each primitive compiled a kernel of its own per shape.
Profiling by hooking `jax._src.compiler.backend_compile_and_load` and attributing
each compilation to the running phase: **66% of the run was compilation, spread
over 2726 separate compiles**, and the arithmetic was never the cost.

- **Four calls are compiled at module scope**, in the idiom 03 already uses:
  `design_compiled`, `state_compiled`, `stability_compiled`, `modes_compiled`,
  plus `value_of` and `gradient_of` for the objective and its gradient. Measured
  per call: the gradient goes 4.659 s eager to 0.21 ms compiled, one design 0.140
  to 0.24 ms, the stability check 0.561 s to 0.50 ms. **52.5 s to 14.1 s, and
  2726 compiles to 540.**
- **The compiled programs are kept between runs.** `jax_compilation_cache_dir` was
  unset, so every run repaid compilation; a 5 MB `.jax_cache/` holds it. The
  default `jax_persistent_cache_min_compile_time_secs` of 1.0 would have cached
  none of these, every one being smaller than that, so it is set to zero.
  **14.1 s to 5.9 s warm** — and 52.5 s to 13.9 s on the *uncompiled* script, the
  two being independent. Cold and warm runs print identically.
- **Compiling moves the last two or three bits, and one printed column by more.**
  The refinement, stagger, stability and assumption blocks are byte-identical;
  forces, moments, diameters and both masses are unchanged. The autodiff gradient
  agrees to 1e-14 relative. **The central differences move in their third
  significant figure** — they are differences of nearly equal numbers, so XLA's
  re-association shows up there rather than in the quantities themselves. Worst
  scaled gradient error 6.90e-09 against 6.70e-09, both far inside 5e-8, and
  `worst |u - 1|` 2.89e-15 against 2.44e-15. **That column is not bit-stable and
  should not be quoted to three figures.**
- **What is left is not arithmetic.** `refinement_study` is 8.6 s of the 13.7 s
  uncached run: six meshes, each paying host-side setup plus its own compilations.
  The setup is upstream — `jax_fdm`'s `_indices_free` and `_connectivity_free`
  build index data with JAX ops, one kernel compile each, 54 of them for a
  40-member graph.

### Experiment 10's composed side was eager, so its seconds compared unlike things

The report claimed "both compiled" on every timing line and only half of it was
true: the oracle went through `design_compiled` and `mass_compiled`, while the
composed side was never jitted and only `_backend_smax.solve` was compiled inside
it. So the in-process column measured a compiled program and the composed column
measured eager dispatch around three crossings. **14.15 s to 11.4 s cold and 5.1 s
warm**, on the same persistent cache as experiment 09.

- **The composed side compiles as a closure, not as an alias.** `eqx.filter_jit`
  on `composition.design_members` traces the problem's array leaves, and
  `_form_find` serializes `structure.nodes` with `np.asarray` to build the payload
  — which raises `TracerArrayConversionError` on a traced structure. Capturing the
  problem keeps it concrete and leaves only the force densities varying, which is
  what actually varies per call. The oracle takes its problem as an argument and is
  unaffected, `normax.pipeline` never serializing anything.
- **Compiling does not fold the boundary away.** A Tesseract is a primitive JAX
  lowers a callback for, so all three stages are crossed on every call of the
  compiled program rather than once while tracing. Counted at the stages' own
  `apply` endpoints: **three crossings per call, eager or compiled.** What
  compiling removes is the per-primitive compilation around the crossings.
- **Every composed number is bit-identical**, value and gradient alike. What moved
  is the *oracle* gradient column, which was an eager `jax.grad` of a compiled mass
  and is now a compiled gradient — and it moved **toward** the composed answer:
  worst gradient error **5.49e-14 to 3.66e-14**, back to the 3.6e-14 on record. The
  parity fields, the masses and `worst |u-1|` are untouched.
- **The seconds now measure one thing.** In process reads 0.0003 s on all four rows
  where it used to range 0.0003 to 0.0713, the spread having been eager dispatch
  rather than anything structural; composed reads 0.026 s for a design against
  0.062, and 0.118 s for a gradient against 0.221. The boundary costs about **90x
  on a design and 390x on a gradient**, and that is now a claim about serializing
  and reassembling rather than about who got compiled.

### The crown point gives way to a mirrored half-span pair

The arch answered to one asymmetric case, which loaded the left half and left the
right half at `HALF_FACTOR`. That biases the search towards the half it leaves
light, and an asymmetric optimum then says nothing about whether the asymmetry is
structural or an artefact of the loading. The crown point case is replaced rather
than joined: the set is still three cases, now `LC1 uniform`, `LC2 half span` and
`LC3 half span mirrored`, symmetric about midspan.

- **`loads_half_span` gained `mirrored`.** It loads the far half instead of the
  near one. A node exactly at midspan is loaded either way, which is what makes
  the two cases exact reflections on a symmetric node layout rather than merely
  similar; `tests/test_structures.py` asserts `near == far[::-1]`.
- **`mirror_gap` measures the bias rather than assuming it away.** It reports how
  far a per-member quantity departs from its own reflection. **The floored design
  comes out symmetric — 5.97e-05 on the diameters and 2.62e-05 on the force
  densities** — and its governing split is 10 / 10 to LC2 and LC3, with LC1
  governing nothing.
- **The unconstrained design does not, and that is the plateau rather than the
  loading.** It measures **1.43e-01 on the diameters and 8.84e-02 on the force
  densities**, with a 16 / 4 split. A symmetric problem started from a symmetric
  shape should stay symmetric at every iterate; this one does not, because its
  descent ends wherever floating point stops it. One more reason the
  unconstrained endpoint is not a quotable design.
- **The masses.** Funicular 0.127126 t; best uniform 0.122287 t at 1.50x;
  unconstrained 0.062669 t; floored 0.097971 t. **The floored design is 22.9%
  lighter than the funicular arch**, and the unconstrained one 50.7% — of which
  most is collapse: shortest member 41.1 mm against 310.6 mm, length ratio 101.5
  against 7.0, and 16 of 20 members under the floor against none.
- **Both remain far from stable.** Weakest critical load factor 1.1030 for the
  unconstrained design and 1.9740 for the floored one, against the 10 of
  §5.2.1 — inadequate under every case. `LC1` governs no member of either.
- The finite-difference trough is unmoved at 1e-4, and the worst scaled gradient
  error is 1.60e-08 against a 2e-07 tolerance.

### American English, including in the units API

The repo was written in two dialects at once — `utilization` in the clause code
beside `analysed` in the docstring above it — and `normax/units.py` had the
British spelling in its exported names.

- **`normax.units` is renamed.** `MILLIMETRE`, `NEWTON_MILLIMETRE` and
  `TONNE_PER_CUBIC_MILLIMETRE` become `MILLIMETER`, `NEWTON_MILLIMETER` and
  `TONNE_PER_CUBIC_MILLIMETER`; `to_metres`, `to_millimetres`,
  `to_newtons_per_square_millimetre`, `to_newton_metres`, `to_newton_millimetres`,
  `to_kilograms_per_cubic_metre` and `to_tonnes_per_cubic_millimetre` follow.
  `tests/test_units.py` is the only caller and moves with it.
- **Everything else is prose**: docstrings, comments, the Pydantic field
  descriptions of all three Tesseracts, and the markdown. A field *description*
  is schema documentation and not a field *name*, so nothing that crosses a
  boundary changed and the parity numbers are untouched.
- **`analysis` and `analyses` are American already** and were left alone; only the
  three verb uses of `analyses` became `analyzes`. Replacement was case-sensitive
  throughout, which is what kept `DescentResult` — which contains `centRes` — from
  being mangled by the `centre` rule.

### Experiments — one module prints, and no result travels as a bare tuple

Every script in `experiments/` was hand-formatting its own stdout. A table was
written twice — once as a header string of padded literals and once as a row
f-string of the same widths guessed again — so the two drifted apart as soon as a
column changed, and eleven scripts had eleven house styles. The columns were held
apart by runs of two and three spaces inside the format strings, which is
alignment expressed as whitespace the author has to count.

- **`normax/reporting.py` is new, and stands to stdout as `normax.visualization`
  stands to matplotlib.** A `ColumnSpec` states a column's heading and the format
  its cells take, exactly once; the width follows from the text that is actually
  printed, so a heading cannot fall out of step with the row beneath it and no
  string in an experiment pads anything by hand. `ReportWriter` carries the
  verbosity, so a caller silences a whole report by constructing a quiet writer
  rather than by threading a flag down through the functions that compute.
  `ToleranceCheck` and `checks_passed` replace the summary-and-verdict block that
  six scripts had each written out longhand.
- **Prose is written as prose and rewrapped when printed.** `write_note` dedents
  a triple-quoted paragraph and fills it to the width of a banner, which retires
  the runs of consecutive `print` calls holding one hand-wrapped line each.
- **Results travel in named containers.** `geometry` returned a four-tuple the
  callers destructured positionally, `setup` returned three things in a fixed
  order, `gap` returned two floats distinguishable only by position, and
  `classify` returned a status, a count and a string. Each is now a `NamedTuple`
  with an `Attributes` docstring, and the quantities derived from those fields —
  a worst-case ratio, a label, a pass rate — are properties on the container
  rather than expressions repeated at each call site.
- **No signature takes more than five arguments.** The prepared problem, the load
  cases and the funicular force density of `experiments/03` travel as one
  `ArchProblem`; `experiments/07`'s assembly arguments travel as one `ModelSpec`,
  which also makes a perturbed model a single `_replace` instead of two
  hand-written dictionary copies. `experiments/09` no longer solves the stability
  problem twice, which the split into report functions exposed.
- **Nothing is built inside a call's argument list, or inside a `return`.** Every
  column tuple, row list, entry tuple and container is assembled as its own
  statement, bound to a name that says what it is, and only then passed or
  returned. This is the general form of the rule `## JAX array construction`
  already stated for arrays, and it is what the table-building code above was
  violating worst: a `write_table` whose two arguments were both multi-line
  comprehensions made the reader hold the callee, the argument and the argument's
  contents at once. An AST check over `experiments/` and `normax/reporting.py`
  reports zero of either pattern.
- **`08_arch_formfind_analyse` is now `08_arch_formfind_analyze`.** The old name listed
  the two stages and left out what the script measures, and read as though
  analyzing were the point rather than the thing under test. What it measures is
  whether the two stages agree on a force neither told the other, which is the
  same shape of claim as `04_backend_agreement` makes about two solvers. The word
  was already in the file: the figure is `08_handoff.png` and the containers are
  `HandoffForces` and `HandoffGap`.
- **`experiments/02` claimed Class 3 was plastic.** The branch label read
  `"plastic" if section_class else "elastic"`, which is true for every class that
  is not zero. It now asks `normax.ec3.classification.is_plastic`. This is the
  only number or word in any experiment's output that the refactor changes;
  everything else was diffed against a captured baseline and is identical,
  wall-clock timings excepted.

### P9d — the annealed descent compiled once instead of once per round

A profile of `experiments/03_optimize_arch.py` found the run compile-bound
rather than arithmetic-bound: **81% of 25.0 s went to XLA compilation**, and the
whole pipeline — form finding, frame analysis and the standard — evaluated its
value and gradient in 13 ms. Half the run, 12.6 s over 11 compilations, was
attributable to one line.

- **`optimize_annealed` captured the sharpness instead of passing it.** Each
  round built a fresh closure and handed it to a fresh `eqx.filter_jit`, so five
  rounds were five programs and no compilation carried over. Both halves of that
  mattered independently: a new `filter_jit` per round starts an empty cache, and
  a captured array is baked in as a constant rather than traced. One compiled
  value-and-gradient over `(q, beta)` is now built before the loop and every
  round calls it. `jax.value_and_grad` already differentiates argument zero
  alone, so `value_and_gradient`'s body did not change — only its type and its
  docstring.
- **The schedule is converted to an array first.** The signature accepts a
  `Sequence[float]`, and a Python float leaf is *static* under `eqx.filter_jit`:
  left alone it would compile a program per round and silently give the cost
  back. Converting once also settles the weak/strong dtype, so a list and an
  array trace the same program rather than two.
- **Two docstrings asserted the retrace was unavoidable** — `value_and_gradient`
  claimed rounds "cannot" share, `minimize_bounded` that a schedule "pays for as
  many as it has rounds". Both were false and both would have talked the next
  reader out of the fix, so both now state why one traced sharpness parameterizes
  one program.
- **The experiment's reporting path was eager**, costing 286 XLA compilations for
  3.5 s because `unsmoothed_design`, `governing_load_case`, `frame_stability` and
  `shortest_member` compiled a program per primitive. All four take their clause
  selectors as static keywords, so `eqx.filter_jit` applies unchanged.

**25.0 s to 10.3 s measured, 530 compilations to 226**; the dominant site falls
from 11 compilations and 12.6 s to 2 and 2.5 s, which is one per descent, the
unconstrained and floored objectives being genuinely different functions.

Two tests guard it, because a refactor away from either would be silent. One
counts traces of the objective across a five-round schedule and asserts exactly
one; the other does the same for a schedule of plain floats.
`test_the_sharpness_reaches_the_objective` had read `float(beta)` from inside the
objective, which worked only because the sharpness was concrete; it now asserts
the recorded mass equals the objective recomputed at each iterate's own
sharpness, which tests the same claim from outside.

The value is unchanged to 1.3e-15 and the gradient to 1.4e-11 — the latter being
bisection's root tolerance carried into the implicit adjoint, three orders inside
the experiment's own 2e-7 target. **The floored result is unmoved at 31.7%
lighter than the funicular arch**, `alpha_cr` 1.7227 to 1.7158. The unconstrained
descent lands elsewhere, 0.0663 t against 0.0700 t, which is the plateau already
recorded under P4 rather than a consequence of this change.

`experiments/03_optimize_arch.py` also picked up `ruff format` and three E501
fixes it had been carrying unformatted, and carries the tail of P9's container
work that had been sitting uncommitted in the tree: `SweepReport` and
`FinalReport` in place of the bare tuples `report_sweep` and `report_final`
returned, and a `design_under` helper. Same thread as P9 and P9b, recorded here
because it rode along on this commit rather than on its own.

### P9c — experiment 10, and why it had been failing

Three causes, only one of which was a tolerance.

- **It compared a container as one array.** The loop ran over `Design._fields`
  and called `np.asarray` on `actions`, stacking an axial force in newtons with
  two moments in newton-millimeters and two dimensionless factors, then divided
  the worst difference by the largest element of the lot. That ratio is not a
  relative error of anything: the moment's disagreement was being scaled by the
  axial force, which happens to be 1.8x larger, so the reported figure was also
  understated. The table's own columns gave it away, printing `axial_force[0]`
  under the label `actions`. It now walks leaves with the same `named_fields`
  helper the parity test uses, so every number is measured against the quantity
  it belongs to.
- **The oracle was not compiled, and that was the real defect.** Every other
  consumer — the parity test, experiment 09, the README — runs the in-process
  pipeline under `filter_jit`, and the Tesseract stages compile internally, so
  the comparison had two different fusion schedules either side of it and
  charged the difference to the boundary. Compiling it moves the required
  diameter from 3.23e-14 to **5.05e-15** and the end moment from 2.29e-12 to
  3.58e-13. **The 1e-14 target is unchanged and now passes on its own merits.**
  It also corrects a claim the experiment was making backwards: the seconds read
  0.140 in process against 0.062 composed, implying three schema crossings were
  faster than calling Python. Compiled, it is 0.0003 against 0.0613.
- **The end moments needed a tolerance of their own, and the tests already said
  so.** A funicular arch carries its design case axially, so the moment is the
  residual — measured at 3.9e-4 of the axial action times the length — and a
  last-bit difference in the analysis inputs is amplified by the reciprocal of
  that ratio before it reaches the moment. `tests/test_tesseract_parity.py` has
  carried `TOLERANCE_MOMENT = 1e-11` and that argument since P8; experiment 10
  predates `Design.actions` and was never given the same treatment. It has it
  now, at the same value, against a measured 3.58e-13.
- The diameter inherits the moment's error attenuated by an elasticity of
  **0.02**, measured with `jax.jacfwd` rather than assumed: the moment is worth
  about a fiftieth of the utilization at the root. That predicts 4.7e-14 for the
  eager oracle against 3.23e-14 observed, which is what identified compilation
  rather than conditioning as the cause.

### P9 — the pipeline argument list, and load cases named as such

- **`ProblemSetup` collapses the five arguments both pipelines retyped four
  times each.** `pipeline.py` threaded `structure, graph, model, steel,
  catalogue` and `composition.py` threaded `structure, chain, steel, catalogue`
  through `design_members`, `total_mass`, `design_envelope` and the helpers
  beneath them — the same list written out seven times, which is the drift P8
  existed to remove and which had simply migrated outward from `ec3`. One
  `NamedTuple` per module, deliberately sharing a name the way `Model` does
  across the two analysis backends, since the two are interchangeable ways to
  run one pipeline. Positional counts: `pipeline.design_envelope` 9 → 4,
  `composition._check_load_case` 8 → 5, `design_members` and `total_mass` 7 → 3
  in process and 6 → 3 composed, `frame_stability` 4 → 2, `unsmoothed_design`
  and `governing_states` 3 → 2.
- **The clause selectors stayed out of it.** `section_class`, `resultant` and
  `normal` remain keyword-only rather than joining the container. A
  `NamedTuple` of an `int` and a `bool` is hashable, so `nondiff_argnums` and
  `static_argnames` both accept one and it works — but the same container
  reaching a `jnp.where` traces its bool as a leaf and returns a number with no
  error at all, where today's `if is_plastic(section_class)` raises
  `TracerBoolConversionError`. One argument saved is not worth removing that.
- **The material and the section family travel in the container as well as
  inside the compiled model.** Folding them into `Model` to shorten
  `member_forces` would be the obvious next step and is wrong: `prepare_model`
  holds them as placeholders and `_injected_assembly` overwrites every one, so a
  leaf left alone keeps a constant and its gradient is silently zero rather than
  an error.
- **`beta` is keyword-only.** It sat eighth and ninth in a positional list
  beside `loads`, which is the position a wrong argument goes unnoticed in.
- **Load cases are spelled `load_case`, never `case`.** A cross-section class is
  also numbered 1 to 4, and `cases` beside `section_class` in the same signature
  is a genuine ambiguity. Renamed as identifiers and in the shape annotations —
  `Float[Array, "cases members"]` is now `"load_cases members"` — with the
  English idiom left alone: "the usual case here" and "in any case" are not load
  cases and were not touched. `governing_case` → `governing_load_case`,
  `_check_case` → `_check_load_case`, `sizes_per_case` → `sizes_per_load_case`.
- Verified bitwise: grouping arguments changes no pytree leaf, so the parity
  harness must reproduce every number exactly, and it does — **identical across
  all 48 arrays**, mass and gradients included. 1843 tests pass, unchanged in
  count. Experiments 01, 02, 03, 06 and 09 reproduce their published numbers
  (the arch still 31.7% lighter with a length floor at `alpha_cr` 1.72, the
  member-length assumption still worth 3.26x the mass); 10 still fails its own
  1e-14 value target at 1.30e-12, exactly as it did before this work.
- Fixed in passing: the served-container branch of `experiments/10` called
  `total_mass` with the old signature. It is skipped without
  `NORMAX_SERVED_OUTPUT`, so nothing ran it — pyright caught it, not the tests.

### P9b — the figures and the OpenSees internals

- **`visualization.py` is off the wide-signature list entirely.** Six functions
  took between six and nine plot series each, which is the shape that invites a
  transposed pair no assertion can catch. Eleven small containers replace them,
  in the idiom the module already had with `Descent` and `Form`:
  `DrawnStructure` and `ColorRange` for `draw_members` (8 → 2 and a default),
  `SizedMembers` for `figure_sections` (6 → 4), `MeshRefinement` and
  `StaggeredPasses` for `figure_convergence` (6 → 2), `HandoffForces`,
  `GapScaling` and `GradientCheck` for `figure_handoff` (9 → 3), `MassSweep`
  with `GradientCheck` again for `figure_optimization` (6 → 3), and
  `BackendAgreement` with `BackendTimings` for `figure_backends` (6 → 2).
- **`_assemble_blocks` lost an argument rather than gaining a container.**
  `num_members` was the leading axis of `axial` two lines above the only call
  site, and a second copy of a number already present is a chance for the two to
  disagree. What is left travels as a `ParameterSweep`, which carries the node
  count because that is what says where the coordinate columns end and the
  section columns begin: 7 → 4.
- `_build_model`'s `loads` and `parameters` are keyword-only, leaving 5
  positional. Taking it lower means a container for `steel` and `catalogue`
  across both backends and `ProblemSetup`, which is a separate decision about
  the gradient-injection path rather than a tidy-up.
- Fixed in passing: **`experiments/04`'s agreement section has been dead since
  P8** — it read `end_moments_major` off a `MemberForces`, whose fields were
  renamed to `moment_major`, and crashed before every number it prints. The
  stale spelling was a string inside a loop over field names, so nothing static
  could see it and the traceback came only from running the file. It now
  reproduces the backend agreement end to end: member forces to 2.7e-15, the
  Jacobian blocks to 2.4e-11, and the mass gradient to 7.2e-12 against a 1e-6
  target.
- Verified: 1843 tests pass, the parity harness is still identical across all 48
  arrays, and every figure was regenerated and read back rather than merely
  built — 03, 04, 08 and 09 all reproduce their published panels.

### P8 — ec3 restructuring

- **Fixed: the analytic bracket read `resultant` on a branch the check ignores
  it on, and silently oversized every plastic member.**
  `utilization_cross_section` routes Classes 1 and 2 to `utilization_plastic`,
  which always combines the two moments as a resultant — Eq. 6.41 takes both
  exponents as two for an axisymmetric section, so the collapse is exact algebra
  and there is nothing to choose. `bracket` applied the flag regardless. Its
  "lower bound" then landed *above* the root, every midpoint of the bisection
  satisfied the check, and the search returned the top of an interval it had
  never needed to narrow. At `plastic=True, resultant=False`, `n_ed = 0`,
  `M_y = M_z = 40 kNm`: diameter 199.400 mm against the correct 177.646 mm, a
  utilization of 0.7071 rather than 1, and a member 12.2% oversized — the factor
  is `2^(1/6)`, the cube root of the ratio between the summed and resultant
  moments. Worse for the gradient: the implicit function theorem was being
  applied where the utilization was 0.71, so `∂d/∂M_y` came back 20.5% off
  central differences (1.2858e-6 against 1.0666e-6). It broke CLAUDE.md
  invariant 5 and was reachable from `pipeline.design` and from the Tesseract,
  both of which read `plastic` and `resultant` independently. It hid because
  `test_the_plastic_branch_ignores_the_reading` sizes at `n_ed = -5e5`, where
  6.3.3 governs and sums the moments linearly anyway, and
  `test_both_readings_are_exactly_fully_stressed` only ran the elastic branch.
  The three-way combination — cloned in `utilization_elastic`,
  `utilization_cross_section` and `bracket` — is now one `moment_combined`
  helper the check and the bound both call, so a fourth copy cannot appear.
- **`resultant` belongs in `static_argnames`.** It selects a clause exactly as
  `plastic` does. Both jit tests omitted it and passed only because they relied
  on its default; passing it explicitly raised `TracerBoolConversionError`.
- **`interaction_factors` returns an `InteractionFactors` rather than a bare
  4-tuple.** The tuple was splatted positionally into `checks`'s
  `factor_yy, factor_yz, factor_zy, factor_zz`, where a `yz`/`zy` transposition
  produces a plausible wrong number that no check can refuse.
- **The clause layer is renamed from EN 1993-1-1's symbols to English.**
  `n_pl_rd` → `resistance_yielding`, `m_n_rd` → `resistance_bending_reduced`,
  `chi` → `reduction_buckling`, `phi` → `buckling_auxiliary`, `epsilon` →
  `material_factor`, `classify` → `classify_section`, `c_m_linear` →
  `moment_factor_linear`, and the rest. The symbol
  each function returns now appears in its `Returns` description, so the tie to
  the standard survives; the clause number in `Notes` was already there. Three
  modules defined a function called `utilization` and `sizing` had to import one
  of them aliased: they are now `utilization_frame` (§5.2.1, the whole frame),
  `utilization_member` (6.3.3, one member) and `utilization_design` (the larger
  of the two checks). Renaming also killed two shadowings, `v_pl_rd`'s
  `area_shear` parameter against the function of that name and `n_cr`'s
  `second_moment` against `section.second_moment`.
- **`normax/ec3/material.py`, a pure leaf holding `SteelGrade` and the material
  constants.** `Steel` moves out of the sizing module, so the analysis backends
  and the composition no longer import a bisection and a `custom_jvp` to name a
  material. The constants travel with it because a NamedTuple's defaults are
  evaluated when the class body runs, so a grade defined apart from the
  constants it defaults from closes an import cycle no deferred import can open.
  `resistance.py` and `stability.py` now take the grade rather than loose
  `f_y`/`e_mod`/`gamma_m*`: 17 signatures, `resistance_tension` falling from six
  parameters to three.
- **`SteelGrade` gains `f_u`, `gamma_m2` and `alpha`.** The first two make
  `resistance.py` uniform — `resistance_fracture` and `resistance_tension` could
  not otherwise take a grade — at the cost of two leaves the sizing map never
  reads and whose tangents are zero. `alpha` moves off `Tube`, which held a
  buckling curve, a sizing rule and a catalogue floor in one box: EN 1993-1-1
  Table 6.2 assigns the curve by fabrication route, so the same grade drawn hot
  and cold is two grades, not two shapes. `Tube` is down to `ratio` and
  `diameter_min`.
- **Fixed: `pipeline.governing` invented the minor axis it reported on.**
  `design` sizes with all five actions, but `Design` kept only `n_ed`,
  `m_ed = m_y_ed` and `c_m = c_my`, so the diagnostic rebuilt the missing half
  as `m_z = 0` and `c_mz = 1` and then named a limit state from it. At
  `n = -300 kN`, `M_y = 20 kNm`, `M_z = 60 kNm`, `c_my = 0.9`, `c_mz = 0.4`,
  `L_cr = 4 m` the true code is `LIMIT_CROSS_SECTION` and the fabricated one is
  `LIMIT_MAJOR`: a member sized by 6.2.9 reported as sized by Eq. 6.61.
  Diagnostic only — nothing differentiable read it and the mass was never
  affected — and invisible on the arch fixture, whose minor-axis moment is zero
  anyway, which is why it survived. `Design` now carries a `MemberActions`, so
  there is nothing left to fabricate; the same fix lands on the composed path,
  where `composition.design` was dropping two fields the ec3 Tesseract already
  returns.
- **`MemberActions` groups the five actions the checks read.** A new
  `normax/ec3/actions.py` leaf, replacing `pipeline.Actions`, whose docstring
  had already noted it was "ordered to be splatted straight into the sizing
  map" — the splat is now the container. Seven sizing signatures and the three
  cross-section checks lose four parameters each: `diameter_required` goes from
  ten to four positional arguments, `utilization_cross_section` from nine to
  four plus two flags. The moment factors default to one, the largest Table B.3
  permits, so a member given no factor is checked conservatively. `Envelope`
  keeps its flat fields, its per-case arrays carrying a case axis a container
  annotated `"members"` cannot honestly hold.
- **The section family is `TubeCatalogue`, and it is renamed on a commit of its
  own so the name `Tube` is free before it is reused.** The section object
  arriving next is a `Tube(diameter, thickness)`, which is what the old name
  ought to have meant — the type holding it carried a ratio and a floor and
  never a diameter. Renaming both in one pass would have left `Tube` bound to
  the *family* at every site a sweep missed, and that code compiles, runs, and
  is silently wrong; renaming first makes any survivor a `NameError`. 707
  identifiers moved by token, plus 35 numpydoc labels, and the parameter is
  `catalogue` rather than `tube` everywhere, so nothing reads as a section that
  is not one. Verified afterwards that no binding named `Tube` or `tube` remains
  anywhere in the tree.
- **`Tube(diameter, thickness)` replaces the seven free functions in
  `section.py`.** A tube is two leaves and every property — `ratio`,
  `diameter_inner`, `area`, `second_moment`, `radius_of_gyration`,
  `modulus_elastic`, `modulus_plastic` — is computed on access, so a diameter
  and a wall cannot drift apart and no call site restates the annulus formulas.
  `TubeCatalogue.tube(d)` is the only place a wall is chosen for a diameter.
  `TubeCatalogue` and `DIAMETER_MINIMUM` move into `section.py` beside it, and
  `at_class_limit` delegates to a new `classification.ratio_at_class_limit`,
  which **closes the last `adjoint → sizing` edge**: `adjoint.py` called itself
  "an independent oracle, not the rule the map is differentiated with" and then
  imported the bisection it audits. The ec3 DAG is now acyclic with three
  leaves and `sizing` alone at the top.
- **This step moved the numbers, and the bitwise baseline was re-captured
  once.** Storing a wall instead of a ratio is a reparametrization rather than
  a regrouping: the wall is a division and reading the ratio back is another, so
  `d / (d / r) == r` fails for 20–40% of diameters at one ulp — measured across
  ten thousand of them at six ratios. Nothing downstream can then be bitwise.
  Measured against the step-5 capture: mass 1.7e-15, diameter 2.7e-14, gradient
  7.3e-14, utilization 1.8e-15, and end moment 1.5e-12, that last being the
  funicular near-cancellation `test_tesseract_parity.py` already holds at 1e-11
  and not a new effect. Class 2 sizing was bitwise unchanged throughout; Class 3
  moved one diameter in six. **Bitwise is the right bar again from here on** —
  the remaining steps are regroupings.
- **`area` is written as `pi t (d - t)` rather than as a difference of two
  circles.** For a wall at the Class 3 limit the squares agree to within 7% of
  their own size, so differencing them throws away a digit for nothing.
- **`area_shear` stays in `resistance.py` and did not move onto `Tube`.** The
  plan had it on the section object, but `A_v = 2A/pi` is EN 1993-1-1 6.2.6(3)
  — a clause, not geometry — and `section.py` opens by saying it holds no
  clauses. Putting it there would also have written the constant twice.
- **`_modulus` lost the `resultant` parameter it never read**, since its
  signature was changing anyway.
- **Renaming by spelling was safe here, and only here.** The usual rule is to
  rename by binding, but a scan first established that every `Tube`/`tube`
  binding in the tree referred to the family: no shadowing, no `.tube`
  attribute, no string key. `tokenize` then separates identifiers from strings
  and comments for free, which an AST-span rewrite does not.
- **The README's usage snippet had been broken since the grade moved**, importing
  `Steel` from `normax.ec3.sizing`. It is now correct, one import per line, and
  it was run: 0.0336 t on the twenty-member arch with a finite gradient in every
  force density.
- **6.3.3 reads a `CompressionBendingState` and a `MemberResistance`.** The
  three byte-identical thirteen-parameter signatures — `utilization_member`,
  `_checks`, `governing_equation` — become five arguments and a flag;
  `interaction_factors` goes from ten to five, `checks` from twelve to four.
  `gamma_M1` is read off the grade rather than travelling as a loose defaulted
  argument, so `interaction.py` no longer imports a partial factor of its own
  and every clause reaches it the same way.
- **`CompressionBendingState` is deliberately not `MemberActions`, though the
  fields line up.** The sizing map fed the member check a sign-transformed copy
  — `maximum(-n, 0)`, `abs(m_y)`, `abs(m_z)` — and the raw signed values to the
  cross-section check in the same breath, then wrote the transform out a third
  time in the diagnostic. `from_actions` does it once. One type for both
  conventions would make a tension-positive force reaching 6.3.3 yield a
  *negative* axial ratio, which subtracts from Eqs. 6.61 and 6.62 and reports a
  member as safer than it is; two types make that a checker error instead.
- **The slendernesses stay outside `MemberResistance`.** `checks` does not read
  them — only Annex B does — and `tests/test_worked_example_frame.py` is the test
  for that separability: it supplies Simões da Silva's published reduction
  factors and interaction factors and has no slendernesses at all.
- **`MemberSlenderness` carries the pair, and `about_both_axes` states the
  axisymmetry once.** `lam_y` and `lam_z` traveled as two positional arguments
  through four signatures, so every call site for a circular section passed the
  same value twice — `properties.slenderness, properties.slenderness` in both
  `_demands` and `governing_limit_state`. Two adjacent positional floats of the
  same type also admit a silent transposition wherever the axes do differ, which
  is the same failure `InteractionFactors` was introduced to close. The section
  properties now hold the pair rather than one number, so the duplication is gone
  from the caller and the reason for it is stated on the constructor. `k_yy` and
  `k_zz` stay flat: each is a function of one axis, like `reduction_buckling`.
- **`l_cr` is spelled `buckling_length` in code, and stays `L_cr` in prose.** 181
  identifiers across 20 files, swept from the token stream rather than by regex:
  `l_cr` is a substring of `flexural_critical`, which a word-boundary-free
  substitution would have corrupted, and BSD `sed` has no `\b`. Three string
  literals had to move with their bindings — `parametrize("l_cr", ...)` names a
  test parameter — while the three `"l_cr"` keys building the Tesseract input
  dict had to stay, being the wire format. The schema field keeps the standard's
  symbol; only the local reading it was renamed.
- **The cross-section class travels as itself, in place of `plastic: bool`.**
  A bool cannot say which class, so the two had to be carried side by side and
  were free to disagree: `diameter_required(..., catalogue_at_class_3,
  plastic=True)` ran the plastic clauses on a Class 3 wall and returned 199.600
  mm where 207.181 mm is required — **1.1220 utilized against the class the wall
  actually has**, 12.2% overstressed, with nothing to object. `section_class:
  int` now reaches `sizing`, `pipeline`, `composition`, the Tesseract wire schema
  and the clause entry points, and `TubeCatalogue.section_class(f_y)` reads it
  off the family's own ratio, so the value a caller passes comes from the thing
  it describes. `classification.section_class_at_ratio` is the host-level inverse
  of `ratio_at_class_limit`; it returns a Python `int` and refuses a traced ratio,
  which is honest — the class chooses between clauses and no derivative with
  respect to the ratio can move it.
- **`is_plastic(4)` raised nothing and answered False**, so a Class 4 shell was
  silently checked by Class 3's clauses while `at_class_limit(f_y, 4)` refused
  the same section outright. `is_plastic` moved to `classification.py`, the leaf
  that owns the concept, and now validates: every point of use rejects a class
  this package does not implement.
- **No signature carries a boolean standing in for a class.** The class reaches
  the Table B.1 rows themselves — `k_yy`, `k_zz`, `k_yz`, `k_zy`,
  `cap_is_active` — and `moment_combined`, each converting with `is_plastic` at
  the point it reads its column. Stopping one level higher and handing the rows a
  bool would have left a second encoding of one fact in the package, which is
  what the change exists to remove. Classes 1 and 2 select identical clauses
  here, so 2 is the plastic representative wherever a fixture named `True`.
- **`utilization_plastic` calls `moment_resultant` directly again.** Bug A's fix
  routed it through `moment_combined` with a literal `plastic=True`, but that
  branch was never one of the three clones the helper was introduced to merge —
  it always took the resultant, which 6.2.9.1 makes exact. Restoring the direct
  call leaves `moment_combined` with the two callers that genuinely choose a
  reading, and neither has to invent a class: the elastic branch names
  `CLASS_ELASTIC`, and the analytic bound already has the real one.
- **The sweep was defeated three more times, all caught by tooling rather than by
  reading.** Two locals in `test_sizing_monotonicity.py` held utilization values
  and section moduli and merely happened to be spelled `plastic`; renaming them
  by spelling left assertions referring to a name that no longer existed, which
  `F841` caught. A third site passed the flag *positionally* —
  `size(*probed, catalogue, False)` — invisible to a keyword-based rewrite and
  found only by running `experiments/02`. Parity stayed bitwise across all 48
  arrays, which is the right bar: the class selects exactly the clauses the bool
  did.
- **`sizing._properties` replaces the capacity block that was cloned at two
  sites.** `_demands` promised the utilization and the diagnostic "cannot
  disagree about what governed" while the diagnostic recomputed the block
  independently, so they could. `_modulus` also lost the `resultant` parameter
  none of its callers passed.
- **`cap_is_active`'s docstring labeled its own parameter wrong**, calling
  `c_m` `moment_factor_linear` — collateral from the rename sweep, which
  rewrote a numpydoc label as if it were an identifier.
- **Two AST traps recurred and both were caught by tests, not review.** A call
  through a variable (`branch = utilization_plastic if plastic else ...`) and a
  call through `jax.jit(diameter_required, ...)` are invisible to a sweep that
  matches on the callee's name. Seventy call sites moved mechanically; three
  needed hands. A fourth escaped the suite entirely — `experiments/03` builds a
  `Design` positionally and nothing imports it, so only running the experiment
  found it. **Experiments are not covered by `pytest` and have to be run.**
- **373 identifiers were moved from the AST, not by regex.** `diameter` has 318
  textual occurrences and `governing` 84, nearly all parameters and prose; only
  names actually bound to the renamed imports were touched. The one case the
  first pass got wrong is worth recording: in `sizing.py` five functions take a
  parameter named `diameter` while the module also defines the function, so
  body references to the *parameter* were rewritten to the new function name.
  Renaming by binding, not by spelling, is the only safe way to do this.

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
Eqs. 6.61 and 6.62 reproduces its numbers. Nothing here analyzes a frame —
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

`jax-fdm` finds the shape and `smax` analyzes it. They exchange a geometry and
nothing else — no prestress, no initial member forces — so the axial forces that
come back are `smax`'s own product and their agreement with `q · L` is a
prediction rather than an identity. `experiments/08_arch_formfind_analyze.py`
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
its own before either stage was wired: `normax` is millimeters, newtons and
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

A frame cannot be analyzed without sections and the sections are what the check
returns, so `design` takes the analyzed diameters as an input and returns the
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
  drawn at a width proportional to diameter and colored on one shared scale, over
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
`normax.form_finding` already existed. A fourth was found in the writing:
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

`normax.form_finding.positions_vertical` solves for heights with the plan held, so
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
choose without analyzing anything again. And **form finding runs once for all the
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

Two behaviors of the driver are pinned because they surprised: **L-BFGS-B steps
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
conclusion does not follow, because the constant favors DDM enormously.

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
  `smax.py` is the move of the old module; `opensees.py` is new. Behavior is
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
*, loads)`. This mirrors `normax.form_finding`, where `graph` is built once and
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

The composition still favors DDM, **2.6x to 3.2x**, and that is honest: the
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

### Naming: actions in full, and no single-word function

- **The five design actions carry their names in full.** `n_ed` is
  `axial_force`, `m_y_ed` and `m_z_ed` are `moment_major` and `moment_minor`,
  and `c_my`/`c_mz` are `moment_factor_major`/`moment_factor_minor`. 441
  identifiers. This reverses P8's explicit decision to keep EN 1993-1-1's
  printed symbols as field names, and the reason it held then — that the symbols
  appear verbatim in Eqs. 6.61/6.62 — is now served by the docstrings, which
  still name the clause and the symbol.
- **The wire moved with the code.** All three Tesseract schemas now carry
  `axial_force`, `moment_major`/`moment_minor`, `moment_factor_major`/
  `moment_factor_minor` and `buckling_length`, so a payload and the code reading
  it read alike. The two-column per-end arrays are `end_moments_major` and
  `end_moments_minor`, which resolves a real ambiguity the symbols hid: `m_y_ed`
  named a pair of end moments on the analysis output and a single design moment
  on the check's output, and they are different quantities. `test_tesseract_
  parity.py` asserts the schema field names, so it changed with them. **The
  material fields keep their symbols** — `f_y`, `e_mod`, `gamma_m0`, `gamma_m1`,
  `alpha` — matching `SteelGrade`, which was not part of this rename.
- **The Jacobian block names went too.** `ForceJacobian.n_ed_xyz` and its three
  siblings are compounds that a `\bn_ed\b` sweep cannot see; they are now
  `axial_force_xyz`, `axial_force_diameter`, `moment_major_xyz` and
  `moment_major_diameter`.
- **Every function is at least two words.** 44 in `normax/` and 13 more inside
  the Tesseract modules: `pipeline.mass` is `total_mass`, `pipeline.design` is
  `design_members`, `formfinding.graph` is `equilibrium_graph`,
  `structures.arch` is `arch_2d`, `sizing.mass` is `mass_of_tubes`,
  `interaction.checks` is `interaction_checks`, and so on. `TubeCatalogue.tube`
  is `tube_at`, which is the one method called with parentheses; `Tube.area` and
  `Tube.ratio` stay as they are, being attributes read off a receiver that
  already supplies the context. Tesseract Core's `apply` endpoint keeps its
  mandated name.
- **`sizing.mass` and `pipeline.mass` could not both become `total_mass`** — the
  three-way `utilization` collision P8 untangled is the precedent. `sizing` weighs
  a set of tubes and takes the name `mass_of_tubes`, which the ec3 Tesseract had
  already chosen for it as an import alias. `pipeline` and `composition` do keep
  the same names as each other, deliberately: they are the same three functions
  in process and across the boundary, and the tests import them under aliases to
  compare them.
- **A rename by binding, and five ways it still went wrong.** The sweep resolved
  which local names were actually bound to the functions being renamed, since
  `mass`, `graph`, `actions`, `governing` and `backend` are also container fields
  and parameters. Even so: NamedTuple **field declarations** are `ast.Name` in a
  store context and were renamed with the functions, so `Design.actions` became
  `Design.member_actions` while `Design(actions=...)` kept the keyword; a module
  imported under an alias (`from normax.analysis import opensees as
  backend_opensees`) put the functions behind an attribute the sweep never
  looked at; a token-level pass over the Tesseract backends renamed **`jax.jvp`
  to a function JAX does not have**, because a NAME token after a dot looks like
  any other — caught by the endpoint raising `AttributeError` and reverted;
  the conservative shadow rule skipped uses in files where any function bound the
  name locally, and `ruff --fix` then **deleted the freshly renamed import as
  unused** before the skipped call site was repaired; and renaming keyword
  arguments into **Blueprints**, a third-party oracle, broke six calls, since a
  keyword belongs to the callee's namespace. Only the last two were caught by
  reading; the rest by tests and by `ruff`.
- **Nothing outside this package was renamed.** Verified by extracting every
  foreign import and every attribute read off a foreign module — 230 symbols
  across `jax`, `smax`, `jax-fdm`, `numpy`, `blueprints`, `tesseract-core` and
  the rest — and differing the set against `HEAD`: identical. A rename touches
  our own names only; a keyword argument belongs to the callee's namespace, and
  a module attribute to the module's.
- Parity stayed bitwise across all 48 arrays throughout both renames.

### Experiments, fixed in passing

- **`experiments/10` had been raising since P5b and nothing noticed.** It called
  `pipeline.design` with the pre-P5b signature — six positionals where seven are
  needed, and a `normal=` the function stopped accepting once the topology was
  hoisted out of the objective — and further down referenced `steel.alpha` where
  the module constant is `STEEL`. Both are `TypeError`/`NameError` on the first
  call, so the experiment never reached a number. **`pytest` does not collect
  `experiments/`**, and the project notes recorded it as passing throughout.
- It now runs and reports FAIL on its own value target: `Design.actions`, added
  in P8, entered the field-by-field comparison, and a member force crossing the
  analysis solver agrees at 1.30e-12 against a threshold of 1e-14 set before that
  field existed. The mass is at 6.48e-16 and the gradients at 3.66e-14, both
  inside target. **The threshold is left as it stands** — widening it to make the
  banner read PASS would be tuning the check to the answer. Whether force fields
  deserve their own tolerance is a decision, not a fix.
