# Proposal — serialize local-run stdio redirection in `tesseract-core`

*Prepared 2026-08-29 from `docs/tesseract_stdio_race.md`, the current `tesseract-core` bug-report template and contribution guide, and a smoke test against normax's crossed structural-design pipeline.*

This document is the filing-ready proposal.
The older `docs/tesseract_stdio_race.md` remains the investigation record: it preserves the measurements, the claims that were withdrawn, and the first issue and PR drafts.
This version separates the upstream defect from the conditions under which normax exposed it and includes a runnable minimum reproduction in the issue itself.

## Decision summary

1. **The upstream bug is independent of normax, JAX, OpenSees and the optimizer.**
   `LocalClient.run_tesseract` enters `mpa.start_run` for every endpoint call.
   `redirect_stdio` then swaps process-global file descriptors 1 and 2 through `os.dup2`, with a save-and-restore protocol that is correct only while calls nest in strict last-in-first-out order.
   Concurrent calls do not provide that ordering.
2. **SLSQP plausibly amplified exposure; it did not cause the defect.**
   The SLSQP route present when normax measured overlap asked for the full constraint Jacobian with `jax.jit(jax.jacrev(slack))`.
   That materialized one reverse pullback per constraint row through both crossed stages.
   The current augmented Lagrangian aggregates all rows into one scalar before reverse mode, so it sends one cotangent through each stage.
3. **The causal claim stops there.**
   A current smoke test confirms more crossings under the dense Jacobian, but observes no overlap.
   The old run measured overlap; the new run does not.
   It is therefore supported that SLSQP created more opportunities for exposure, not that SLSQP scheduled the calls concurrently or caused the file-descriptor race.
4. **The issue should be filed before the PR.**
   This follows `tesseract-core/CONTRIBUTING.md`.
   The issue uses the repository's `BUG-REPORT.yml` fields verbatim: Description, Steps to reproduce, Logs, OS, and Tesseract version.
5. **The minimum reproduction exercises the public API.**
   It creates a tiny in-memory Tesseract, opens it with `Tesseract.from_tesseract_api`, and calls one shared local client from four threads.
   This is the same local-client route normax uses without asking maintainers to install normax, JAX, Blueprints or OpenSees.
6. **The minimal correct fix is a module-level `threading.RLock` held for the whole redirection.**
   It must be reentrant because the existing nested-run test re-enters `start_run` on one thread.
   Holding it for the whole run serializes local endpoint bodies; that trade-off is disclosed rather than hidden.

## What SLSQP did, and what the smoke test proves

### Historical route

The route that existed when concurrent dispatch was measured is recoverable in Git history:

- `c295375` added the process-wide dispatch pin after the crossed OpenSees path had been observed with four XLA workers and ten overlapping pairs among twenty-four dispatches.
- `experiments/103_simultaneous_api.py`, retained immediately before `2ed6e15`, constructed `slack_jacobian = jax.jit(jax.jacrev(slack))` and passed that dense Jacobian to `scipy.optimize.minimize(method="SLSQP")`.
- A later reusable helper at `c4c311f:normax/extras/slsqp.py` used `jax.jacfwd(slack)` instead.
  It was deleted in `1c51ea1` on 2026-08-27.
- The old experiment was deleted in `2ed6e15` on 2026-08-27.

Consequently, "the retired SLSQP route sent many independent cotangents" is accurate for the route present at the overlap measurement, but it is not a statement about every SLSQP implementation that existed in the repository.
SLSQP itself is a consumer of the dense Jacobian; JAX/XLA decides where its host callbacks run.

### Why OpenSees exposed the concurrency

Normax's planar analysis stage hosted OpenSees' native C++ engine through OpenSeesPy inside a local Tesseract.
This was precisely the kind of boundary Tesseract is meant to cross: a modern JAX optimization differentiated through a mature solver whose sensitivity machinery predates the pipeline.
OpenSees owns one mutable process-global domain rather than an independent model per Python call.
Each analysis endpoint cleared and rebuilt that domain, solved the frame, and read Direct Differentiation Method sensitivities from the same native state.
Two endpoint calls entering concurrently could therefore erase or mutate the model while the other call was still solving or reading sensitivities.

The observed failure was a native-process termination, usually `exit 139`, at different points between runs and without a Python traceback.
A dispatch ledger flushed before each entry and exit showed four XLA worker threads and ten overlapping pairs among twenty-four calls.
Three controls narrowed the condition: a toy Tesseract did not become concurrent merely because it was jitted, OpenSees survived sequential calls migrating among worker threads, and callbacks within one compiled program remained ordered.
The failing condition was concurrent entry from independent compiled programs into the one OpenSees domain.

The SLSQP formulation magnified that opportunity because it required the entire constraint Jacobian.
For the route measured then, `jax.jacrev(slack)` seeded each constraint row separately, and each seed pulled through the design check and then through the OpenSees analysis Tesseract.
Those local reverse calls reached `tesseract-jax` host callbacks that were emitted without `ordered=True`, so XLA was permitted to have independent callbacks in flight together.
SLSQP did not create the unsafe global state or decide the callback schedule; it repeatedly exercised the path on which both already mattered.

The same overlap crossed `LocalClient.run_tesseract`, which entered `mpa.start_run` and redirected process-global stdout and stderr around every endpoint body.
That is how one scheduling condition exposed two different process-global hazards: OpenSees' mutable native domain and Tesseract Core's descriptor save-and-restore protocol.
Normax's owner-thread dispatch pin protects the solver by serializing calls on one thread.
The proposed upstream `RLock` protects Tesseract Core's stdio invariant, but does not claim to provide thread affinity for OpenSees or other native libraries.

### Smoke test, 2026-08-29

The retired driver cannot be run from the current tree.
Instead, the smoke test reconstructed its derivative operation on the current crossed pipeline:

- four-member arch;
- fixed geometry and four diameter variables;
- one load case and four utilization constraint rows;
- PyNite analysis across one local Tesseract;
- Blueprints check across another local Tesseract;
- `NORMAX_PIN_DISPATCH=0`;
- `Jaxeract.vector_jacobian_product` instrumented for call count, thread name, and temporal overlap;
- historical SLSQP operation: `jax.jit(jax.jacrev(maps.slack))(x)`;
- current augmented operation: `maps.augmented_lagrangian(x, multipliers, penalty, reference)`.

Steady-state results:

| derivative request | VJP crossings | peak concurrent | threads | elapsed |
|---|---:|---:|---|---:|
| dense constraint Jacobian | 8 | 1 | `MainThread` | 0.019–0.021 s |
| scalar augmented pullback | 2 | 1 | `MainThread` | 0.006 s |

The four constraint rows caused four reverse calls through each of the two Tesseract stages: eight VJP crossings instead of two.
This supports the amplification mechanism directly.
It does **not** reproduce concurrent dispatch: all calls ran serially on `MainThread`, matching the 2026-08-28 measurement in `docs/tesseract_stdio_race.md`.

The defensible statement for both filings is therefore:

> Normax observed concurrent local dispatch while its then-current SLSQP route requested a dense constraint Jacobian.
> The affected analysis Tesseract hosted OpenSees' native C++ engine through OpenSeesPy, and overlapping callbacks entered its single mutable process-global domain concurrently.
> The native solver then terminated with `exit 139` rather than a Python exception.
> That route made one reverse crossing per constraint row and therefore created more opportunities for overlap than the current scalarized augmented Lagrangian.
> A current smoke test confirms the larger crossing count but does not reproduce overlap, so SLSQP is treated as an exposure amplifier, not as the cause of this bug.

The upstream report does not depend on that explanation.
Its standalone reproduction uses ordinary Python threads and the public local-client API.

---

# GitHub issue draft

## Suggested title

`LocalClient: concurrent endpoint calls can corrupt process-wide stdout and stderr`

## Description

> [!NOTE]
> This issue is related to [normax](https://github.com/arpastrana/normax), our submission to the [Tesseract Hackathon 2026](https://pasteurlabs.ai/tesseract-hackathon-2026/).
> The defect was discovered while composing normax's local structural-analysis and design-check Tesseracts, and is reproduced here independently of normax.

`LocalClient.run_tesseract` wraps every endpoint invocation in `mpa.start_run`.
That path redirects process-global file descriptors 1 and 2 through `os.dup2` in `runtime/core.py::redirect_fd`, once per call, without synchronization.

Each redirection saves the descriptor's current target, replaces it with the call's logfile, and restores the saved target in `finally`.
That protocol is correct only when overlapping redirections nest in strict last-in-first-out order.
Two threads are not required to exit in that order:

1. Call A redirects stdout to A's logfile.
2. Call B saves stdout, which now points to A's logfile, and redirects it to B's logfile.
3. A and B restore in a non-LIFO order.
4. The last restore installs a saved temporary logfile rather than the host's original stdout.

Observed consequences:

- output from one endpoint can appear in another run's `tesseract.log`;
- after both calls return, the host process can remain attached to a temporary logfile;
- a descriptor owned elsewhere can be closed;
- under pytest, capture teardown can raise `OSError: [Errno 9] Bad file descriptor`, after which later tests fail as a cascade.

The state is process-global, so client count is not material.
One shared local client called from two or more threads is sufficient.
`stream_logs=False` does not avoid the path: `start_run` and the redirection still occur.

The defect was reproduced on `tesseract-core` 1.12.0 and on current `main` at `206d3d7122231f04f04afadce56582adf90a36b9`, the 1.12.0 release commit.

Normax first encountered the concurrency precondition while its SLSQP formulation requested a dense constraint Jacobian through `jax.jacrev`.
The affected analysis Tesseract hosted OpenSees' native C++ engine through OpenSeesPy and rebuilt its single mutable process-global domain for each call.
Independent reverse-mode crossings were permitted to overlap, and concurrent entry into that native state produced nondeterministic `exit 139` terminations.
The same overlapping calls also passed through `LocalClient.run_tesseract` and its process-global stdio redirection, which led to the independent descriptor race reported here.
SLSQP amplified the number of opportunities for overlap; it did not cause either global-state hazard, and this report does not require OpenSees or JAX to reproduce the Tesseract Core defect.

> I searched the open and closed issue tracker, including the existing `start_run`, log-streaming, subprocess-logging, and output-redirection reports, and found no report covering crossed `dup2` restoration during concurrent local runs.

## Steps to reproduce

Create an isolated environment and install the released runtime:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install "tesseract-core[runtime]==1.12.0"
```

Save the following as `reproduce_stdio_race.py`:

```python
import os
import sys
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType

from pydantic import BaseModel
from tesseract_core import Tesseract


class InputSchema(BaseModel):
    value: int


class OutputSchema(BaseModel):
    value: int


def apply(inputs: InputSchema) -> OutputSchema:
    # Widen the overlap between local endpoint calls. The barrier below remains
    # outside the endpoint, so the proposed serialization cannot deadlock it.
    time.sleep(0.005)
    print(f"worker {inputs.value}", flush=True)
    return OutputSchema(value=inputs.value)


api = ModuleType("tesseract_api")
api.InputSchema = InputSchema
api.OutputSchema = OutputSchema
api.apply = apply

RUNNERS = 4
ROUNDS = 25


def fd_identity():
    """Device and inode currently attached to stdout and stderr."""
    return {
        fd: (os.fstat(fd).st_dev, os.fstat(fd).st_ino)
        for fd in (sys.stdout.fileno(), sys.stderr.fileno())
    }


with TemporaryDirectory() as directory:
    client = Tesseract.from_tesseract_api(
        api,
        output_path=Path(directory),
    )
    barrier = threading.Barrier(RUNNERS)
    before = fd_identity()
    saved = {fd: os.dup(fd) for fd in (1, 2)}
    failures = []

    def worker(worker_id):
        try:
            barrier.wait()
            for round_index in range(ROUNDS):
                client.apply(
                    {"value": worker_id},
                    run_id=f"{round_index}-{worker_id}",
                )
        except BaseException as exc:  # preserve failures from worker threads
            failures.append(exc)

    try:
        threads = [
            threading.Thread(target=worker, args=(worker_id,))
            for worker_id in range(RUNNERS)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
            assert not thread.is_alive(), "a local run did not finish"

        after = fd_identity()
    finally:
        # A successful reproduction leaves the process broken. Restore it so
        # the invoking shell or test runner is not poisoned too.
        for fd, backup in saved.items():
            os.dup2(backup, fd)
            os.close(backup)

assert not failures and after == before, (
    f"worker failures: {failures!r}\n"
    f"stdio was left pointing elsewhere: {before} -> {after}"
)
```

Run it:

```bash
python reproduce_stdio_race.py
```

The script must be run where `sys.stdout` and `sys.stderr` expose real file descriptors.
`redirect_stdio` deliberately becomes a no-op for fd-less streams, as can occur in notebooks and under some capture modes.

Each call receives a unique `run_id`, so the threads do not intentionally share a logfile.
The barrier synchronizes only the first call: placing one in every round could strand the remaining workers if an unpatched call raises.
The only shared resource under test is the process-global stdio state.

## Logs

Representative failure modes on an unpatched runtime:

```text
AssertionError: worker failures:
  [RuntimeError("... OSError: [Errno 9] Bad file descriptor ...")]
stdio was left pointing elsewhere:
  {1: (0, 12606148107960175502), 2: (0, ...)} ->
  {1: (16777232, 57614548), 2: (...)}
```

Both descriptors began on the host's pipe and ended on regular files belonging to completed local runs.
The public-API script above was smoke-tested against normax's locally installed 1.11.0 runtime: it completed in 0.9 s, two worker calls raised `Bad file descriptor`, and stdout and stderr both ended on the same regular-file inode.
The original lower-level stress reproduction was separately verified against 1.12.0 and current `main`, where it failed five runs out of five.
The regression test proposed below passes in approximately 0.50 s with the lock.

## OS

Mac

The bug is not expected to be macOS-specific: it follows from process-global POSIX file-descriptor mutation and unsynchronized save/restore.
Only macOS is claimed here because that is where it was reproduced.

## Tesseract version

```text
tesseract-core 1.12.0
main 206d3d7122231f04f04afadce56582adf90a36b9
```

## Suggested resolution

Serialize the whole stdio-redirection lifetime with a module-level `threading.RLock`.
It must be reentrant because the existing nested-run test re-enters `start_run` on one thread.

This deliberately serializes overlapping local endpoint bodies.
That is the minimal behavior consistent with swapping process-global descriptors for the whole call.
If local endpoint concurrency must be preserved, it requires a more involved descriptor ownership protocol rather than independently locking the `dup` and `dup2` instructions.

I have a patch and regression test ready and will open a PR after discussion.

---

# GitHub pull request draft

## Suggested title

`fix(runtime): serialize stdio redirection across overlapping runs`

## Body

> [!NOTE]
> This pull request is related to [normax](https://github.com/arpastrana/normax), our submission to the [Tesseract Hackathon 2026](https://pasteurlabs.ai/tesseract-hackathon-2026/).
> It upstreams a runtime concurrency defect found while exercising normax's local structural-analysis and design-check Tesseracts.

#### Relevant issue or PR

Fixes #709.

#### Description of changes

`redirect_stdio` swaps process-global file descriptors 1 and 2 and restores them by hand.
That protocol is correct only while overlapping redirections nest in strict last-in-first-out order.
Nothing previously enforced that invariant, so local runs overlapping in different threads could cross their restores, contaminate one another's logs, leave stdout or stderr attached to a completed run's temporary file, or close a descriptor still owned by another context.

This change adds a module-level `threading.RLock` and holds it for the entire `redirect_stdio` context, including descriptor restoration and cleanup.
The lock is entered first in the existing `ExitStack`, which makes it the last context released and keeps cleanup inside the critical section.
A shorter critical section cannot protect the saved descriptor between its replacement and eventual restoration.
The lock must be reentrant because `test_nested_runs` enters `start_run` recursively on one thread, which would deadlock with a plain `threading.Lock`.

Serializing the complete redirection context also serializes overlapping local endpoint bodies that use it.
This is intentional because one process's stdout and stderr cannot simultaneously point at two per-run logfiles.
The lock is process-local, so served Tesseracts running in separate worker processes retain process-level concurrency.

Normax encountered the overlap while differentiating a structural-analysis pipeline backed by OpenSees' native C++ engine through OpenSeesPy.
Its then-current SLSQP formulation requested a dense constraint Jacobian, causing one reverse crossing per constraint row through its design-check and analysis Tesseracts.
Independent host callbacks could overlap, concurrently entering OpenSees' single mutable process-global domain and producing native `exit 139` failures.
Those same calls crossed `LocalClient.run_tesseract`, exposing the independent process-global stdio race addressed here.
Normax now pins native solver dispatch to one owner thread, while this change provides the narrower protection required for correct descriptor restoration in any concurrent local workload.

This change does not add callback ordering to `tesseract-jax`, guarantee thread affinity for stateful native solvers, or otherwise expand `LocalClient`'s public thread-safety contract.

#### Testing done

A behavioral regression test starts four threads, gives every run its own base directory, and asserts that file descriptors 1 and 2 retain their original device and inode after all runs finish.
The workers synchronize once before their repeated runs, but no barrier is placed inside `start_run`, so the fixed implementation can serialize without deadlocking.
The test restores the original descriptors in `finally`, allowing an unpatched failure to be reported without poisoning the rest of the test process.

The regression failed five runs out of five against unpatched `main` and passed in approximately 0.50 seconds with the `RLock`.
Existing nested-run coverage passes, establishing that reentrancy is preserved.
The public-client reproduction from #709 passed with four threads making twenty-five calls each through one shared local client.
The focused runtime suite passed with 78 tests, and the complete non-end-to-end suite passed with 669 tests and 132 expected skips.
Every configured pre-commit hook passed over the complete repository.

Verification commands:

```bash
pytest tests/runtime_tests/test_mpa.py tests/runtime_tests/test_core.py
pytest --skip-endtoend
pre-commit run --all-files
```

---

# Filing checklist

1. Make the normax repository public before using its link in the highlighted callouts.
2. Re-run the public-API reproducer on released 1.12.0 and on the exact upstream `main` commit to be filed against.
3. Issue filed through `BUG-REPORT.yml` as [#709](https://github.com/pasteurlabs/tesseract-core/issues/709).
4. Review any maintainer response on #709 before merging or materially expanding the patch.
5. Fork `pasteurlabs/tesseract-core`, branch from current `main`, and apply only the lock and regression test.
6. Sign the Pasteur Labs contributor license agreement if required by the PR.
7. Run the focused tests, then the upstream non-end-to-end suite and pre-commit.
8. Open the PR with `Fixes #709` and the highlighted normax/hackathon callout at the absolute top of the body.
9. Keep the normax/SLSQP history in the issue context, not in the runtime code or regression test.
   The upstream fix stands on the public-API reproduction.
