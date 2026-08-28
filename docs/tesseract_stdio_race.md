# A concurrency defect in tesseract-core's stdio redirection

**Established 2026-08-28: the defect is real, it is in the current release, and
no route this package ships currently reaches it.** The bug reproduces in twelve
lines against `tesseract-core` 1.12.0 and against `main`, and upgrading does not
fix it. Whether *we* can provoke it is a separate and less settled question —
one measurement says yes, a later one says no, and the two are recorded below
rather than reconciled.

**This file is the record and the two texts to file with it.** It exists so the
report is filed from a standalone reproduction that stands on its own, rather
than from a claim about our own exposure that we cannot presently support.

## The defect

`LocalClient.run_tesseract` wraps **every** endpoint call in `mpa.start_run`,
which redirects process-global file descriptors 1 and 2 through `os.dup2` with
no synchronization:

```python
# tesseract_core/runtime/core.py:58-80
def redirect_fd(from_: TextIO, to_: TextIO | int) -> Generator[TextIO, None, None]:
    orig_fd = os.dup(from_.fileno())      # :68
    ...
    os.dup2(to_, from_.fileno())          # :73
    try:
        yield orig_fd_file
    finally:
        os.dup2(orig_fd, from_.fileno())  # :79
        orig_fd_file.close()
```

The redirect is per call (`sdk/tesseract.py:1007`), it is process-global rather
than a `sys.stdout` swap, and it assumes strict LIFO nesting that nothing
enforces. Where two redirections overlap, the second to enter captures the
**first's log file** as its "original" stdout, and whichever exits last installs
that stale descriptor onto fd 1. Two silent consequences: output from one run
lands in another run's `tesseract.log`, and the host process's stdout and stderr
are left pointing at a temporary file permanently.

It also **closes** a descriptor the rest of the process may own. Under `pytest`
that surfaces as `OSError: [Errno 9] Bad file descriptor` from capture teardown,
after which every later test errors.

## What we verified

| claim | verdict |
|---|---|
| Present in the latest PyPI release | **yes** — PyPI latest is 1.12.0, and `main` at `206d3d7` *is* the v1.12.0 release commit. `redirect_fd` is byte-identical to 1.11.0. |
| Already reported upstream | **no** — no matching issues. |
| More clients widen it | **no.** The state is process-global fd 1/2 and the redirect lives inside one call. A single shared client hit by two threads is exactly as exposed as two clients. Concurrent *calls* are the only term. |
| It breaks normax today | **not currently reachable** — but the measurements disagree, see below. |

## Our own exposure — two measurements that disagree

We first concluded that `pin_dispatch_thread` was load-bearing, then that it was
not. **Both were too confident.** The record is two measurements, taken four
days apart on the same JAX 0.11.0, that do not agree:

**2026-08-24**, crossed OpenSees descent, per
[[normax-tesseract-upstream-prs]]: genuine overlap — *"four distinct workers
plus main"*, *"10 of 24 dispatches genuinely overlap, precondition met"*.

**2026-08-28**, the shipped configuration, `NORMAX_PIN_DISPATCH=0`: nothing
overlapped at all.

- the full arch descent returns an identical 0.150150 t, exit 0, byte-identical
  output;
- six independent crossings under one `jit`, on both backends, run on
  `MainThread` with zero temporally overlapping pairs;
- a `jacrev` over a crossed vector-valued function — the multi-cotangent shape —
  likewise: five dispatches, one thread, no overlap;
- the 53 crossing tests pass serially, in the file order that once produced the
  311-error cascade.

**The difference has not been found.** What is known: the augmented Lagrangian
aggregates its rows into one scalar, so it sends a single cotangent where the
retired SLSQP route sent many independent ones, and the two stages are chained
by a data dependency that forbids reordering regardless. Both changes remove
opportunities for concurrency that the August configuration had. That is a
plausible account, not a demonstrated one.

**So: keep the pin.** The precondition is real and has been seen once; it is
simply not reachable by any route this package currently ships. What holds where
this has actually bitten is the separate rule that
`tests/test_tesseract_parity.py` leaves the composed side eager rather than
compiling it — that file still contains no `jit`.

The pin's docstring should not claim that "under `jit` the runtime runs several
dispatches at once", since half the measurements say otherwise. The honest
statement is that XLA is *permitted* to, `emit_python_callback` being emitted
without `ordered=True`.

## The fix, and the test that proves it

Twelve lines in `tesseract_core/runtime/mpa.py`: a module-level reentrant lock
entered first into the existing `ExitStack`, so it releases last.

**Reentrant, not plain** — upstream's own `test_nested_runs` re-enters
`start_run` on one thread, and a `threading.Lock` deadlocks it.

Measured against `main`: the test fails **5 runs out of 5** in ~0.4 s each, and
passes in 0.50 s with the lock. Upstream's `test_mpa.py` and `test_core.py` give
76 passed with the lock in place, the single error being a missing `pytest-mock`
in our environment rather than a regression.

A first version of the test forced a deterministic non-LIFO interleaving through
events. It failed correctly but *passed in 10.39 s* — the lock stops the overlap,
so the handoff deadlocked and event timeouts rescued it. A test that passes by
timing out is a bad test. The version below asserts the **invariant** instead and
is fast under both regimes.

---

## Issue text

> ### `LocalClient`: concurrent endpoint calls corrupt the host process's stdout and stderr
>
> `LocalClient.run_tesseract` wraps every endpoint invocation in `mpa.start_run`,
> which redirects **process-global file descriptors 1 and 2** through `os.dup2`
> with no synchronization (`tesseract_core/runtime/core.py:68-79`). The redirect
> is per call (`sdk/tesseract.py:1007`) and assumes strict LIFO nesting, but
> nothing enforces it.
>
> When two endpoint calls overlap, the second call's `os.dup(1)` captures the
> **first call's log file** rather than the real stdout, and whichever thread
> exits last installs that stale descriptor onto fd 1. Two consequences, both
> silent:
>
> 1. Output from one run is written into another run's `tesseract.log`.
> 2. After both calls return, the host process's stdout and stderr are left
>    pointing at a temporary log file **permanently**.
>
> It also closes a descriptor the rest of the process may own — under `pytest`
> the symptom is `OSError: [Errno 9] Bad file descriptor` raised from capture
> teardown, after which every subsequent test errors.
>
> **Reproduction.** `tesseract-core` 1.12.0 and `main` (`206d3d7`), Python 3.12,
> macOS. Four threads, twenty-five `start_run` blocks each:
>
> ```
> AssertionError: stdio was left pointing elsewhere:
>   {1: (0, 12606148107960175502), 2: (0, ...)} -> {1: (16777232, 57614548), 2: (...)}
> ```
>
> Both descriptors move from a pipe onto a regular file on disk and stay there.
> Fails 5 runs out of 5. A test is attached in the PR below.
>
> Also reproduced through the public API — one shared
> `Tesseract.from_tesseract_api` client, two threads calling `.apply()` — where
> text printed by thread A appeared in thread B's `tesseract.log` and never in
> A's own. Client count is not a factor; concurrent calls are the only term.
>
> **Why this is reachable rather than theoretical.** `tesseract-jax` provokes it:
> `mlir.emit_python_callback(..., has_side_effect=True)`
> (`tesseract_jax/primitive.py:429`) is emitted without `ordered=True`, so XLA is
> free to run independent host callbacks concurrently, and each one reaches
> `run_tesseract`. A JAX program containing two Tesseracts — the pattern the
> pipelines how-to documents — is enough to permit it. We have measured genuine
> overlap once, in a JAX program driving a stateful legacy solver across the
> boundary, and measured none at all in a later one; the report below therefore
> rests on a standalone reproduction that does not depend on provoking XLA.
>
> **Scope.** The same `start_run` path is used by `runtime/serve.py:74`, so a
> served Tesseract running endpoints on more than one thread has the same defect.
> With `--num-workers 1` the documented "requests are processed sequentially"
> guarantee avoids it; nothing equivalent is documented or enforced for
> `LocalClient`, which carries no thread-safety statement either way.
>
> **Suggested fix.** A module-level `threading.RLock` held across the
> `redirect_fd` enter/exit pair in `redirect_stdio`, which is the invariant the
> OS-level redirect already implicitly requires. It must be reentrant: the
> existing `test_nested_runs` re-enters `start_run` on one thread and a plain
> `Lock` deadlocks it. Happy to open a PR — see below.

---

## PR description

> ### fix(runtime): serialize stdio redirection so overlapping runs cannot cross
>
> Fixes #NNN.
>
> `redirect_stdio` swaps process-global file descriptors 1 and 2 and restores
> them by hand. That is correct only while redirections nest, and nothing
> enforced it: two runs overlapping in different threads left the second to enter
> holding the first run's log file as its "original" stdout, and installing it
> process-wide when it exited last. The host process was then writing to a
> temporary file for the rest of its life, and a descriptor another part of the
> process owned had been closed.
>
> **The fix** is a module-level reentrant lock, entered first into the existing
> `ExitStack` so that it releases last:
>
> ```python
> # Serialises stdio redirection. ``redirect_fd`` swaps process-global file
> # descriptors 1 and 2 and restores them by hand, which is only correct while
> # redirections nest. Reentrant so that nested runs on one thread still work.
> _STDIO_REDIRECT_LOCK = threading.RLock()
> ```
>
> Reentrant rather than plain: `test_nested_runs` re-enters `start_run` on one
> thread, and a `threading.Lock` deadlocks it.
>
> **The test** asserts the invariant — file descriptors 1 and 2 must be where
> they started — rather than trying to force a particular interleaving, so it is
> fast whether or not the guard is present. It fails 5 runs out of 5 on `main`
> and passes in 0.50 s with the fix. It restores the descriptors by hand in a
> `finally`, so a regression fails loudly instead of poisoning the rest of the
> session.
>
> **The trade-off, for you to rule on.** Holding the lock for the whole run
> serializes endpoint bodies, so two threads each running an endpoint now block
> each other. That is the minimal correct fix and matches what the OS-level
> redirect already requires. If you would rather keep concurrency, the
> alternative is a lock around the `dup`/`dup2` pairs alone plus an explicit
> save-stack so restores cannot cross — more machinery, and I am happy to rework
> it that way.
>
> Worth considering separately: either pass `ordered=True` in `tesseract-jax`, or
> document that `LocalClient` requires serialized calls, so the two repositories
> agree on the contract.

---

## The patch and the test, verbatim

```python
# tesseract_core/runtime/mpa.py
import threading

# Serialises stdio redirection. ``redirect_fd`` swaps process-global file
# descriptors 1 and 2 and restores them by hand, which is only correct while
# redirections nest. Reentrant so that nested runs on one thread still work.
_STDIO_REDIRECT_LOCK = threading.RLock()


@contextmanager
def redirect_stdio(logfile, log_sink=None):
    ...
    with ExitStack() as stack:
        # Held for the whole redirection, and so released last: two runs
        # overlapping in different threads would otherwise leave the second to
        # enter saving the first run's log file as its "original" stdout, and
        # installing it process-wide when it exits last.
        stack.enter_context(_STDIO_REDIRECT_LOCK)
        f = stack.enter_context(open(logfile, "w"))
        ...
```

```python
# tests/runtime_tests/test_mpa.py
RUNNERS = 4
RUNS_EACH = 25


def _stdio_identity():
    """What file descriptors 1 and 2 currently point at."""
    return {fd: os.stat(fd).st_ino for fd in (1, 2)}


def test_concurrent_runs_do_not_clobber_stdio(tmp_path):
    """Runs in several threads must leave stdout and stderr where they were."""
    before = _stdio_identity()

    # Restored by hand at the end: where the invariant does not hold the
    # descriptors stay broken, and every later test writes to a stale file.
    saved = {fd: os.dup(fd) for fd in (1, 2)}
    failures = []

    def run_repeatedly():
        try:
            for _ in range(RUNS_EACH):
                with start_run(base_dir=tmp_path):
                    pass
        except Exception as exc:  # noqa: BLE001
            failures.append(exc)

    try:
        threads = [threading.Thread(target=run_repeatedly) for _ in range(RUNNERS)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
            assert not thread.is_alive(), "a run never finished"

        after = _stdio_identity()
    finally:
        for fd, backup in saved.items():
            os.dup2(backup, fd)
            os.close(backup)

    assert not failures, f"a run raised: {failures[0]!r}"
    assert after == before, f"stdio was left pointing elsewhere: {before} -> {after}"
```

## Filing order

`CONTRIBUTING.md` asks for it: *"we recommend you open an issue before
contributing code in a pull request. This allows all parties to talk things over
before jumping into action, and increase the likelihood of pull requests getting
merged."* So the issue first, through the `BUG-REPORT.yml` template, then a PR
from a fork carrying `Fixes #N`. Outside contributors do land here — three
merged in the recent window — so a tested PR is worth offering rather than
waiting to be asked.

## One thing upgrading does fix

PR #685, *"fix(sdk): purge auto-created output tempdirs on garbage collection"*,
shipped in 1.12.0 and closes a separate leak we measured: `LocalClient` has no
`__del__` or `close`, `Tesseract.__exit__` short-circuits for
`from_tesseract_api` clients, and this machine had accumulated **1396** orphaned
`tesseract_output_*` directories. We are pinned at 1.11.0. The directories are
inert, so this is not urgent, but it is the one half of the client-lifecycle
question that a version bump settles rather than a code change.

Related: [[normax-tesseract-upstream-prs]] records this and the tesseract-jax
ordering report as the two to file; `normax-jit-tesseract-fd` records the
311-error cascade this defect produced here in August, and the workaround still
in force.
