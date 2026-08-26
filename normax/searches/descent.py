# Copyright 2026 Rafael Pastrana
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Descending a search, from one start or from many, and keeping the answer.
"""

import hashlib
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jaxtyping import Float
from scipy.optimize import minimize

from normax.optimization import AugmentedBudget
from normax.optimization import ConstrainedMaps
from normax.optimization import descend_augmented
from normax.reporting import Report
from normax.searches.config import DescentConfig
from normax.searches.config import TaskConfig
from normax.searches.problem import SearchAnswer
from normax.searches.problem import SearchMaps
from normax.searches.settings import AUGMENTED_DEFAULT
from normax.searches.settings import DESIGNS
from normax.searches.settings import METHOD_AUGMENTED
from normax.searches.settings import METHOD_SLSQP
from normax.searches.settings import POLISH_ADMISSION
from normax.searches.settings import POLISH_ITERATIONS
from normax.searches.settings import POLISH_ROUNDS
from normax.searches.settings import RECOIL_SLACK
from normax.searches.settings import SCATTER_SEED
from normax.searches.settings import SCATTER_SLACK
from normax.searches.settings import SEARCH_DRAWN
from normax.searches.settings import SEARCH_FORMFOUND
from normax.searches.settings import SEARCH_ORDER
from normax.searches.settings import searches_present


def seed_openings(
    maps: dict[str, SearchMaps],
    starts: dict[str, Float[np.ndarray, "variables"]],
) -> tuple[float, float]:
    """
    Smallest constraint slack of the lens seed and of the drawn seed.

    Parameters
    ----------
    maps :
        Every search's compiled maps.
    starts :
        Every search's starting variable vector.

    Returns
    -------
    opening_found :
        Smallest slack of the end-to-end seed, negative when infeasible.
    opening_drawn :
        Smallest slack of the sizing-only seed, negative when infeasible.
    """
    slack_found = maps[SEARCH_FORMFOUND].slack(jnp.asarray(starts[SEARCH_FORMFOUND]))
    slack_drawn = maps[SEARCH_DRAWN].slack(jnp.asarray(starts[SEARCH_DRAWN]))

    opening_found = float(np.min(np.asarray(slack_found)))
    opening_drawn = float(np.min(np.asarray(slack_drawn)))

    return opening_found, opening_drawn


class StartScatter(NamedTuple):
    """
    Where a multi-start descent leaves from, and what holds it in.

    Attributes
    ----------
    start :
        The nominal variable vector the scattered points are drawn around.
    boxes :
        One bound pair per variable.
    """

    start: Float[np.ndarray, "variables"]
    boxes: list[tuple[float | None, float | None]]


def descend_search(
    maps: SearchMaps,
    start: Float[np.ndarray, "variables"],
    boxes: list[tuple[float | None, float | None]],
    budget: DescentConfig,
) -> SearchAnswer:
    """
    SLSQP under hard `U <= 1`, restarted from its own answer until quiet.

    Parameters
    ----------
    maps :
        The search's compiled maps.
    start :
        The variable vector to leave from.
    boxes :
        One bound pair per variable.
    budget :
        Iterations per round, rounds, and the solver tolerance.

    Returns
    -------
    answer :
        The variables, the mass at every iterate, and how the solver ended.

    Notes
    -----
    Each restart hands SLSQP a fresh quadratic model at the previous answer,
    which is what moves it off the slow tail of a single long run; the loop
    stops the first time a round barely moves. The mass trajectory is read
    through the compiled objective at every iterate, one cheap extra
    evaluation against the figure it buys.

    **The objective is handed over divided by its value at the start, so the
    budget's tolerance is a relative one.** SLSQP tests `|f - f0| < acc` and
    `|s| < acc` against the same number, and the two quantities have no
    common scale: a mass in tonnes is a fraction of one while a step is taken
    over variables of order a hundred. Dividing the objective through makes
    the first test read as a share of the mass, which is what the number in a
    run description is meant to say, and leaves it meaning the same thing on
    a cap of any size.

    A threshold that is relative is not thereby a loose one. The descent has
    a long shallow tail and most of a stopping rule's cost is paid there, so
    the tolerance a run states is a decision about how much of that tail to
    buy, and the mass it settles at moves with it.

    A line-search trial point can leave the model's domain entirely: a
    geometry whose frame cannot be factorized raises from inside the compiled
    slack. Such a point is answered with a uniform, enormous violation
    instead — infeasible is the truthful reading of a structure that cannot
    stand — and the merit function walks the search back into the domain.
    Accepted iterates never sit there, so the Jacobian stays unguarded.

    A failure detected inside a compiled program arrives through a host
    callback and surfaces as a runtime error rather than as anything about a
    value, so both are caught. Catching the value error alone leaves a frame
    that will not factorize to return a finite and meaningless number.
    """

    def weighed(x):
        value, slope = maps.weigh(jnp.asarray(x))

        return float(value), np.asarray(slope, dtype=np.float64)

    reference = abs(weighed(start)[0]) or 1.0

    def objective(x):
        value, slope = weighed(x)

        return value / reference, slope / reference

    def feasible(x):
        return np.asarray(maps.slack(jnp.asarray(x)), dtype=np.float64)

    def feasible_jacobian(x):
        return np.asarray(maps.jacobian(jnp.asarray(x)), dtype=np.float64)

    rows = feasible(start).size

    def guarded_slack(x):
        try:
            return feasible(x)
        except (ValueError, RuntimeError):
            return np.full(rows, -RECOIL_SLACK)

    masses = [weighed(start)[0]]

    def track(x):
        masses.append(weighed(x)[0])

    held = {"type": "ineq", "fun": guarded_slack, "jac": feasible_jacobian}
    options = {"maxiter": budget.iterations, "ftol": budget.tolerance}

    x = np.asarray(start, dtype=np.float64)
    spent = 0
    converged = False
    for _ in range(budget.rounds):
        found = minimize(
            objective,
            x,
            jac=True,
            method="SLSQP",
            bounds=boxes,
            constraints=[held],
            callback=track,
            options=options,
        )
        x = np.asarray(found.x)
        spent += int(found.nit)
        converged = found.status == 0
        if found.nit <= 1:
            break

    return SearchAnswer(x, np.asarray(masses), spent, converged)


def descend_augmented_search(
    maps: SearchMaps,
    start: Float[np.ndarray, "variables"],
    boxes: list[tuple[float | None, float | None]],
    budget: AugmentedBudget,
) -> SearchAnswer:
    """
    An augmented Lagrangian descent of one search, in the shared record.

    Parameters
    ----------
    maps :
        The search's compiled maps, whose augmented program is the one called.
    start :
        The variable vector to leave from.
    boxes :
        One bound pair per variable.
    budget :
        Rounds, inner iterations, the penalty schedule and the stopping rules.

    Returns
    -------
    answer :
        The variables, the mass at the end of every round, and how it ended.

    Raises
    ------
    ValueError
        If the search was built without an augmented map.

    Notes
    -----
    **The iteration count is objective evaluations, not solver iterations.**
    An outer round is not comparable with an SLSQP iteration — it costs one
    reverse pass per evaluation where an SLSQP iteration costs a forward
    tangent per variable — so the record carries the number that prices the
    descent rather than the one that would flatter it.

    The mass column is one entry per round, not per iterate, and its early
    entries are read at infeasible points: a small opening penalty leaves the
    mass in charge, and the search dives below every feasible design before
    the multipliers pull it back onto the surface. Only the last entry, with
    the violation beside it, is a design.

    **The violation column travels with it, and a caller must read it.** An
    opening penalty too weak for the structure leaves the descent inside that
    dive rather than bringing it back, and the mass it reports there is a
    design that does not stand. Nothing in the mass column says so.
    """
    if maps.augmented is None:
        raise ValueError("this search was built without an augmented map")

    programs = ConstrainedMaps(maps.augmented, maps.weigh, maps.slack)
    found = descend_augmented(programs, start, boxes, budget)

    return SearchAnswer(
        found.variables,
        found.masses,
        found.evaluations,
        found.converged,
        found.violations,
    )


class DescentPlan(NamedTuple):
    """
    Which search descends a search, and on what budgets.

    Attributes
    ----------
    budget :
        The budgets every method shares — bounds, floors, starts, and the
        iteration and tolerance settings the constrained solver reads.
    augmented :
        Budgets belonging to the augmented method, or None for the defaults.
    """

    budget: DescentConfig
    augmented: AugmentedBudget | None = None


def descent_plan(config: TaskConfig) -> DescentPlan:
    """
    How the run description says its searches are to be descended.

    Parameters
    ----------
    config :
        The run description.

    Returns
    -------
    plan :
        The method's budgets alongside the shared ones.
    """
    return DescentPlan(config.descent, config.augmented)


def descend_started(
    maps: SearchMaps,
    start: Float[np.ndarray, "variables"],
    boxes: list[tuple[float | None, float | None]],
    plan: DescentPlan,
) -> SearchAnswer | None:
    """
    One start descended by whichever method the run description names.

    Parameters
    ----------
    maps :
        The search's compiled maps.
    start :
        The variable vector to leave from.
    boxes :
        One bound pair per variable.
    plan :
        The method and its budgets.

    Returns
    -------
    answer :
        The landing, or None where an augmented descent never approached
        feasibility and the polish was refused.

    Notes
    -----
    **An augmented descent is always polished, and the polish is not optional.**
    The outer loop stops on its own round budget and says nothing about
    stationarity; a short constrained run from the landing is the cheapest
    certificate that the answer is a stationary point rather than the place the
    rounds ran out. It also supplies the exact feasibility the reported
    utilization is asserted against.

    **A landing the polish is refused for is returned as nothing, not as a
    mass.** An opening penalty too weak for the structure leaves the descent
    inside its own infeasible dive, holding a mass no design supports, and
    handing that to a constrained solver produces a number with no meaning.
    Refusing keeps a start that failed distinguishable from one that landed
    heavy.

    The reported iteration count sums the augmented descent's evaluations and
    the polish's iterations, which are not the same currency. It prices the
    landing rather than comparing it with an SLSQP row.
    """
    if plan.budget.method == METHOD_SLSQP:
        return descend_search(maps, start, boxes, plan.budget)

    relaxed = plan.augmented or AUGMENTED_DEFAULT
    answer = descend_augmented_search(maps, start, boxes, relaxed)
    if float(answer.violations[-1]) > POLISH_ADMISSION:
        return None

    polish = plan.budget._replace(iterations=POLISH_ITERATIONS, rounds=POLISH_ROUNDS)
    found = descend_search(maps, answer.variables, boxes, polish)
    trail = np.concatenate([answer.masses, found.masses])

    return found._replace(
        masses=trail,
        iterations=answer.iterations + found.iterations,
        violations=answer.violations,
    )


def scattered_points(
    start: Float[np.ndarray, "variables"],
    boxes: list[tuple[float | None, float | None]],
    budget: DescentConfig,
) -> list[Float[np.ndarray, "variables"]]:
    """
    The nominal start, and however many scattered ones the run asks for.

    Parameters
    ----------
    start :
        The nominal starting variable vector.
    boxes :
        One bound pair per variable, which no scattered point may leave.
    budget :
        The budgets, read for the count and the spread.

    Returns
    -------
    points :
        The nominal point first, so a run asking for one start descends
        exactly what a single-start run would.

    Notes
    -----
    Scattered multiplicatively, each variable by its own value, because the
    coordinates and the diameters differ by orders of magnitude and one
    absolute spread would be a rounding error to one and a catastrophe to the
    other. The seed is fixed: a multi-start answer that cannot be reproduced
    is not a measurement.
    """
    points = [np.asarray(start, dtype=np.float64)]
    if budget.starts <= 1:
        return points

    lower = np.array([-np.inf if low is None else low for low, _ in boxes])
    upper = np.array([np.inf if high is None else high for _, high in boxes])
    stream = np.random.default_rng(SCATTER_SEED)

    for _ in range(budget.starts - 1):
        noise = stream.normal(0.0, budget.scatter, size=points[0].size)
        points.append(np.clip(points[0] * (1.0 + noise), lower, upper))

    return points


def descend_best(
    report: Report,
    maps: SearchMaps,
    seed: StartScatter,
    plan: DescentPlan,
) -> SearchAnswer:
    """
    Descend from every scattered start and keep the lightest feasible landing.

    Parameters
    ----------
    report :
        Where each start's landing is written as it happens.
    maps :
        The search's compiled maps.
    seed :
        The nominal variable vector to scatter around, and its bounds.
    plan :
        The method and its budgets, read for the start count as well as the
        descent.

    Returns
    -------
    answer :
        The best landing's record. Its mass trajectory is that one descent's,
        so a figure drawn from it shows a real descent rather than a mixture.

    Notes
    -----
    A scattered start can leave the model's domain — a geometry whose frame
    will not factorize raises before the first quadratic model is built — so
    a start that cannot be evaluated is dropped rather than allowed to end
    the run. Landings are compared only among those that came back feasible,
    a search that lands outside the constraints having answered a different
    question.

    **Every start reports as it lands.** A multi-start descent is the longest
    thing a run does and the whole of it used to be one silent stretch, so a
    run that was going badly looked exactly like a run that was going well.
    The line says which start, what it reached and whether it was kept, which
    is also the record of how much the scattering earned.

    **A landing that missed feasibility is repaired before it is refused.** A
    descent that stops a fraction short of the constraints has still found a
    design, and throwing it away discards whatever basin it found along with
    the fraction it missed by; growing its diameters onto the constraint
    surface keeps the first and prices the second. A landing the repair cannot
    rescue — one that missed a height limit or a chord sign, neither of which a
    section answers to — is refused exactly as before, and the line says what
    it missed by either way.
    """
    budget = plan.budget
    start, boxes = seed
    best = None
    points = scattered_points(start, boxes, budget)
    for index, point in enumerate(points):
        named = "nominal" if index == 0 else f"scattered {index}"
        try:
            answer = descend_started(maps, point, boxes, plan)
            if answer is None:
                report.write_line(
                    f"  start {index + 1}/{len(points)} ({named}): refused, the "
                    f"descent never approached feasibility"
                )
                continue
            slack = float(np.min(np.asarray(maps.slack(jnp.asarray(answer.variables)))))
        except (ValueError, FloatingPointError, RuntimeError):
            report.write_line(
                f"  start {index + 1}/{len(points)} ({named}): left the domain"
            )
            continue
        mended = ""
        if slack < SCATTER_SLACK and maps.repair is not None:
            missed = -slack
            grown = np.asarray(maps.repair(jnp.asarray(answer.variables)))
            slack_grown = float(np.min(np.asarray(maps.slack(jnp.asarray(grown)))))
            if slack_grown >= SCATTER_SLACK:
                mass, _ = maps.weigh(jnp.asarray(grown))
                trail = np.append(answer.masses, float(mass))
                answer = answer._replace(variables=grown, masses=trail)
                slack = slack_grown
                mended = f", missed by {missed:.2e} and repaired"
        if slack < SCATTER_SLACK:
            report.write_line(
                f"  start {index + 1}/{len(points)} ({named}): "
                f"{answer.masses[-1]:.6f} t, infeasible by {-slack:.2e}"
            )
            continue
        kept = best is None or answer.masses[-1] < best.masses[-1]
        report.write_line(
            f"  start {index + 1}/{len(points)} ({named}): {answer.masses[-1]:.6f} t "
            f"in {answer.iterations} iterations, "
            f"{'converged' if answer.converged else 'no convergence'}"
            f"{mended}{', best so far' if kept else ''}"
        )
        if kept:
            best = answer

    if best is None:
        raise ValueError("no scattered start reached a feasible landing")

    return best


def descent_digest(config: TaskConfig) -> str:
    """
    A fingerprint of everything about a run that decides where it lands.

    Parameters
    ----------
    config :
        The run description.

    Returns
    -------
    digest :
        Hexadecimal digest of the run description with the viewer removed.

    Notes
    -----
    **The viewer is left out on purpose.** Which search to draw, which case to
    draw it under, and whether to open a window at all decide nothing about
    the descent, and a stored answer that a change of camera invalidated would
    be worthless for the one thing it is for. Everything else that decides the
    landing is in: change a ring, a pressure, a bound, a budget or the method
    and the stored answer stops being an answer to the question being asked.

    **A budget belonging to a method the run does not use is left out for the
    same reason.** It decides nothing here, and counting it would make a file
    that carries one describe a different question from a file that does not —
    so a viewer beside a run would have to repeat the whole block verbatim to
    find the answer that run stored, and would silently find nothing when it
    drifted. The budget is still read and still travels on the description; it
    is only the fingerprint that ignores it.
    """
    described = config._replace(viewer=None)
    if described.descent.method != METHOD_AUGMENTED:
        described = described._replace(augmented=None)

    return hashlib.sha256(repr(described).encode()).hexdigest()


def answers_stored(config: TaskConfig) -> Path:
    """
    Where a run description's answers are kept.

    Parameters
    ----------
    config :
        The run description.

    Returns
    -------
    stored :
        The file the run's answers are written to and read back from.

    Notes
    -----
    **Named by what was descended, not by the file that asked for it.** Two
    descriptions that differ only in which search to draw, or whether to open a
    window, pose the identical question, and a store keyed by filename would
    make the second of them pay for the first one's answer all over again. The
    file that wrote it is kept inside for a reader to recognize it by.
    """
    return DESIGNS / f"{descent_digest(config)[:16]}.npz"


def save_answers(
    path: Path,
    config: TaskConfig,
    answers: dict[str, SearchAnswer],
) -> Path:
    """
    Write every descended answer beside the run description that reached it.

    Parameters
    ----------
    path :
        The YAML file describing the run.
    config :
        The run description, fingerprinted so a stale answer is not reused.
    answers :
        Each search's descent record, keyed by search.

    Returns
    -------
    stored :
        The file written.

    Notes
    -----
    **A solo run adds to the store rather than replacing it.** Descending one
    search says nothing about the other two, so an answer already held under
    the same fingerprint is kept and the descended searches are written over it.
    A fingerprint that does not match is a different question, and the store
    is begun again.

    Only the variables are needed to rebuild a design — every mass,
    utilization and diagram the report carries is recomputed from them — but
    the trajectory and the two things the solver said about how it stopped are
    written too, because a report that could not state them would be a
    different report from the one the descent wrote.
    """
    DESIGNS.mkdir(exist_ok=True)
    held = load_answers(config) or {}
    held.update(answers)

    stored = {"described": np.array(path.name)}
    for search, answer in held.items():
        stem = search.replace(" ", "-")
        stored[f"{stem}.variables"] = np.asarray(answer.variables)
        stored[f"{stem}.masses"] = np.asarray(answer.masses)
        stored[f"{stem}.iterations"] = np.array(answer.iterations)
        stored[f"{stem}.converged"] = np.array(answer.converged)

    target = answers_stored(config)
    np.savez(target, **stored)

    return target


def load_answers(config: TaskConfig) -> dict[str, SearchAnswer] | None:
    """
    Read back the answers this run description was already descended to.

    Parameters
    ----------
    config :
        The run description.

    Returns
    -------
    answers :
        Each stored search's descent record, or None where this description
        has not been descended.
    """
    target = answers_stored(config)
    if not target.exists():
        return None

    stored = np.load(target, allow_pickle=False)
    answers = {}
    for search in SEARCH_ORDER:
        stem = search.replace(" ", "-")
        if f"{stem}.variables" not in stored:
            continue
        answers[search] = SearchAnswer(
            stored[f"{stem}.variables"],
            stored[f"{stem}.masses"],
            int(stored[f"{stem}.iterations"]),
            bool(stored[f"{stem}.converged"]),
        )

    return answers or None


def descend_all(
    report: Report,
    maps: dict[str, SearchMaps],
    starts: dict[str, Float[np.ndarray, "variables"]],
    boxes: dict[str, list[tuple[float | None, float | None]]],
    plan: DescentPlan,
) -> dict[str, SearchAnswer]:
    """
    Descend every search in the shared order, reporting each landing.

    Parameters
    ----------
    report :
        Where each search's landing line is written.
    maps :
        Every search's compiled maps.
    starts :
        Every search's starting variable vector.
    boxes :
        Every search's bound pairs.
    plan :
        The method the searches are descended by, and its budgets.

    Returns
    -------
    answers :
        Every search's descent record, keyed by search.
    """
    answers = {}
    for search in searches_present(starts):
        report.write_line(
            f"{search}, from {plan.budget.starts} starts by {plan.budget.method}"
        )
        seed = StartScatter(starts[search], boxes[search])
        answer = descend_best(report, maps[search], seed, plan)
        answers[search] = answer
        report.write_line(
            f"{search}: {answer.masses[-1]:.6f} t in {answer.iterations} iterations"
        )

    return answers
