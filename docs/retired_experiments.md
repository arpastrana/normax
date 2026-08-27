# The experiments retired before submission, and how to get them back

Ten scripts were deleted on **2026-08-27**. None of them was wrong; every one
measured something the package had since stopped doing in the way the script
asked for it, and repairing them competed with the writeup for the last four
days. The findings they produced are recorded — in `CHANGELOG.md`, in the
documents named below, and in the figures already on disk — so what was lost is
the ability to re-derive a number, not the number.

They are recoverable in full. The tree at **`3d10e5a`** holds every one, under
the numbered names they carried before the 2026-08-27 rename:

    git show 3d10e5a:experiments/validation/20_shear_audit.py > shear_audit.py

**`CHANGELOG.md` and `ROADMAP.md` still name these files, and should.** Both are
dated records of what was planned and what happened, so a path in them describes
the tree as it stood, not as it stands. The rename of 2026-08-27 was applied to
them where it named a surviving script; a path to a retired one is left where it
is. Living documents — `README.md`, `CLAUDE.md` and the decision records — name
none of them.

## What each measured, and what stops it running

| retired name | at `3d10e5a` | measured | blocked by |
|---|---|---|---|
| `shear_audit` | `20_shear_audit.py` | the §6.2.10 exemption, read off converged designs | `normax.searches`, and shear is no longer in `MemberForces` |
| `buckling_audit` | `26_buckling_audit.py` | that §6.3.1 binds on every member of every act | `normax.searches` |
| `gridshell_space` | `25_gridshell_space.py` | the held-plan design space before optimizing | two figure containers; imports `normax.searches` but calls none of it |
| `handoff_forces` | `08_arch_formfind_analyze.py` | the form finder's geometry against the frame solver's forces | three figure containers, and moved analysis imports |
| `beam_benchmark` | `11_straight_beam_benchmark.py` | both stages against a closed-form straight beam | four figure containers, and a `Design` container that has since changed shape |
| `sizer_disagreement` | `14_sizer_stress_test.py` | the two sizers' gap, attributed to §6.3.1 alone | `BlueprintSizer` |
| `simultaneous_search` | `103_simultaneous_api.py` | the showcase arch with sizes as optimizer variables | a `simultaneous:` config section and three `optimization` fields the containers no longer carry |
| `jacobian_crossing` | `22_jacobian_crossing.py` | the price of a constraint Jacobian crossing the boundary | loads `simultaneous_search` by path |
| `descent_animation` | `102_animation.py` | the search replayed, one polyscope frame per iterate | `normax.visualization.frames`, deleted wholesale |
| `toolchain_smoke` | `00_toolchain_smoke.py` | that Docker, tesseract-core and tesseract-jax are alive | nothing — its code is sound; it needs the upstream `vectoradd_jax` image built |

## What a port costs, roughly

**Cheapest.** `gridshell_space`, `handoff_forces` and `beam_benchmark` need their
plotting containers back. Those lived in a 1967-line `normax/visualization.py`,
recoverable at `d754124^`, which still holds nine of the eleven names between
them. The pattern to follow is `blueprint_adjoint`: the script owns what only it
needs, and the package keeps none of it.

**Moderate.** `sizer_disagreement` needs an in-process Blueprints sizer, and
`blueprint_adjoint` already contains one — its `CallbackSizer`, wrapping the
host primitives in `jax.pure_callback` with a `custom_jvp` tangent rule built
from the check's own partials. Lifting it is most of the job.
`simultaneous_search` needs a run description of its own carrying the settings
the shipped config dropped; `jacobian_crossing` follows once it runs.

**Free.** `toolchain_smoke` was never broken. It wants the upstream example
image built first, and reports `ImageNotFound` until it is. Restoring it is a
`git show` and nothing more.

**Hardest, and deliberately deferred.** `shear_audit` and `buckling_audit` share
a harness: six `normax.searches` functions between them, dissolved into the four
example scripts. Rebuilding it once revives both, and `shear_audit` additionally
waits on shear returning to `MemberForces` — see `shear_design.md`, which records
that dependency and the exact route back.

## Two traps a port will meet

**The pipeline checks; it does not size.** Calling it runs the held check and
hands the diameters straight back, so an experiment that wants a fully-stressed
design must call `pipeline.sizer(forces, lengths)` itself and reconcile the load
cases afterwards. Both ports that landed on 2026-08-27 hit this, and it presents
as a silent wrong answer — sizes equal to the seed — rather than as an error.

**The sizer's output carries a load case axis and the checker's does not.**
Taking an envelope over a design that has already been reconciled collapses the
member axis instead, which surfaces as a zero-dimensional array a long way from
its cause.
