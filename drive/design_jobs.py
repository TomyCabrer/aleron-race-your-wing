"""drive/design_jobs.py -- the garage's AeroBO runs as LIVE jobs: the engine
works on a thread of its own, the frame watches it.

AeroBO (vendored at `aerobo/`, reached only through `drive/aerobo_bridge.py`)
runs a screen, a section search or a wing search as ONE blocking call that
reports through a progress callback. Its own GUI runs that call on a worker
thread and lets the page watch it; carsim does the same (PLAN2 D5):

* A RUNNER is `runner(emit, stop) -> result`, a closure the bridge freezes
  at launch, so the job flies a snapshot and an edit made while it runs
  cannot reach it. It runs on the worker thread, never touches pygame, the
  models, the library or `Notices`, and reports ONLY through
  `emit(kind, **payload)` (`EVENT_KINDS`). It stops cooperatively: every
  engine call it makes gets a stop rule / cancel reading `stop.is_set()`,
  so a Stop lands after the evaluation in flight.
* An `EngineJob` owns one runner, its ONE daemon thread and the queue the
  thread fills. Its unit is a DRAIN: `RunManager.pump`, once per
  `Garage.frame`, empties the queue without ever blocking into the job's
  counters, records and status. The page draws between drains, so a
  twelve-minute section run is watched, and the queue is the only thing the
  two threads share.
* FINISHING IS ITS OWN UNIT, on the main thread, in a frame of its own: once
  the thread has ended and its queue is drained, `finish()` settles the
  terminal state (done / converged / stopped / error), logs the closing
  lines, toasts and calls the model's `on_finish`, which applies the result.
  Views draw a finished run from the MODEL's stored `outcome()`, never from
  the job, so a re-opened page cannot show a run that no longer applies.
* `ReplayRunner` plays a captured engine run (a fixture in
  `drive/data/aerobo_fixtures/`) through the SAME job path: the UI's
  self-checks and screenshots see real engine output with no engine and no
  timing, and `pause_at=k` freezes a run at k by construction.
* A warning the engine raises on a worker thread becomes that job's Output
  line (the warnings router); every other thread's warning is untouched.
  In the Output log a Python library's deprecation (torch, botorch, gpytorch,
  pymoo) is dropped, and every other engine warning shows ONCE per session
  -- its first sentence and "(once per session)" -- with its full text kept
  in the job's `lines`.
* `Notices` queues the Output log's lines and the toasts for the shell.
* The template functions at the bottom are every line a job logs, every
  status text, tag and toast it shows, plus the few model lines that must
  read the same as a job's; one place, so the two never drift.

Numbers and strings only: no pygame, no `aerobo`, no bridge -- the whole
module is tested headless in a second or two.
"""
from __future__ import annotations

import copy
import json
import math
import os
import queue
import re
import threading
import time
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path


# --------------------------------------------------------------------------- #
#  what a run is                                                               #
# --------------------------------------------------------------------------- #
#: what a runner may emit (PLAN2 6.1):
#:   "eval"  n (1-based, inherited evaluations counted), best (float | None),
#:           f, feasible, g, x
#:   "sweep" i, n, name, status, eligible, ldcr, tc   (one screened section)
#:   "phase" text                                      ("library pass", ...)
#:   "total" n                                         (the N, when known late)
#:   "log"   text, level                               (an engine warning: "warn")
EVENT_KINDS = ("eval", "sweep", "phase", "total", "log")

#: the job kinds that are RUNS: a chip, start / closing lines, a toast.
#: Anything else ("polar", "score", "report", "law", "warm") is a short task
#: on the same machinery, silent unless it fails.
RUN_KINDS = ("screen", "section", "wing", "lap")

#: what each run evaluates, as the status bar names it (PLAN2 6.2 `<what>`)
WHAT = {"screen": "screening", "section": "section (XFOIL)", "wing": "wing (lattice)",
        "lap": "lap (lattice + lap)"}

#: the noun of a run in its terminal status and its toast
NOUN = {"screen": "screening", "section": "section optimisation", "wing": "wing run",
        "lap": "wing run"}

#: the stop_reason of a run the player stopped -- the bridge's STOP_PLAYER,
#: restated so this module never imports the bridge (same text, kept equal)
STOP_PLAYER = "stopped by the player"

#: the phase words of the status bar, and the longer ones of the "now
#: evaluating" block. "Sobol" / "BO" are derived from the run's Sobol block
#: (`JobInfo.n_init`); the others arrive as "phase" events.
PHASE_LONG = {"starting": "starting (the engine is loading)",
              "Sobol": "Sobol (the space-filling start)",
              "BO": "BO (the Gaussian-process search)",
              "SLSQP": "SLSQP (the local polish from the best BO found)",
              "library pass": "library pass (the cached library polars)",
              "shortlist sweep": "shortlist sweep (live XFOIL at this surface's Re)"}

#: how many screened sections a live screen keeps for its "last swept" list
SWEEP_KEEP = 8


def wing_kind(objective) -> str:
    """The job kind of a wing run: AeroBO's lap objective flies a lap per
    evaluation ("lap"), every other car objective only the lattice."""
    return "lap" if objective == "laptime" else "wing"


# --------------------------------------------------------------------------- #
#  the Output log and the toasts                                               #
# --------------------------------------------------------------------------- #
LOG_LEVELS = ("info", "ok", "warn", "error")
TOAST_KINDS = ("positive", "negative", "warning", "info")


class Notices:
    """The two queues the shell drains every frame: Output log lines
    `(stamp, text, level)` and toasts `(text, kind)`. A line is stamped when
    it is logged, not when it is drawn. An unknown level or kind raises: a
    typo there would otherwise draw an unstyled line nobody notices. Only
    the main thread writes here -- a worker's lines arrive through its job's
    queue."""

    CAP = 1000                    # nothing drains before the shell exists

    def __init__(self):
        self._log: list = []
        self._toasts: list = []
        self._seen: set = set()         # the engine warnings this log has shown (`first_time`)

    def log(self, text, level="info"):
        if level not in LOG_LEVELS:
            raise ValueError(f"log level must be one of {LOG_LEVELS}, got {level!r}")
        self._log.append((time.strftime("%H:%M:%S"), str(text), level))
        del self._log[:-self.CAP]

    def toast(self, text, kind="info"):
        if kind not in TOAST_KINDS:
            raise ValueError(f"toast kind must be one of {TOAST_KINDS}, got {kind!r}")
        self._toasts.append((str(text), kind))
        del self._toasts[:-self.CAP]

    def drain_log(self) -> list:
        out, self._log = self._log, []
        return out

    def first_time(self, key) -> bool:
        """True the first time `key` is asked about on this log, False ever
        after: what shows an engine warning once per session (one Notices is
        one garage visit's Output log)."""
        if key in self._seen:
            return False
        self._seen.add(key)
        return True

    def drain_toasts(self) -> list:
        out, self._toasts = self._toasts, []
        return out


# --------------------------------------------------------------------------- #
#  the jobs                                                                    #
# --------------------------------------------------------------------------- #
@dataclass
class JobInfo:
    """The texts a job needs to talk about itself, built by the model.

    `kind` one of `RUN_KINDS` or a short task's name; `surface` "wing" /
    "endplate" for a screen or a section, the slot's wing ("top wing") for a
    wing run; `stage` "af" | "ep" | "w" | "r" | "m"; `evaluator` the status
    bar's `<what>` (`WHAT[kind]` when left empty); `objective` AeroBO's key
    ("composite_goal", "efficiency", ...) and `phrase` the model's words for
    it, from AeroBO's labels; `seed_label` what a section run starts from;
    `problem` e.g. "top wing · car rear wing + endplates (10-D)"; `fmt`
    score -> text; `rseed` the optimiser's seed; `airfoil` / `plate` the
    sections a wing run flies; `optimiser` AeroBO's name ("bo", "bo_slsqp");
    `n_init` the run's Sobol block (the status bar's Sobol | BO phase);
    `handoff` the evaluation after which a fresh `bo_slsqp` run hands its
    best to SLSQP (the BO | SLSQP phase), None for any other run;
    `point` the design point of a screen or section run:
    {"re", "cl", "source": "own" | "library"}."""
    kind: str
    surface: str = ""
    stage: str = ""
    evaluator: str = ""
    objective: str = ""
    phrase: str = ""
    seed_label: str = ""
    problem: str = ""
    fmt: object = None
    recommended: bool = False
    rseed: int = 0
    continued: bool = False
    track: str = ""
    airfoil: str = ""
    plate: str = ""
    optimiser: str = ""
    n_init: int | None = None
    handoff: int | None = None
    point: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.evaluator:
            self.evaluator = WHAT.get(self.kind, self.kind)


def _fmt_default(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    v = float(v)
    return "refused" if v == -math.inf else f"{v:.6g}"


def _fmt(info, v) -> str:
    f = getattr(info, "fmt", None)
    return (f if f is not None else _fmt_default)(v)


def _first_line(text) -> str:
    lines = str(text).strip().splitlines()
    return lines[0] if lines else ""


def run_record(result):
    """The AeroBO run record (`RunResult.to_dict()`) inside a runner's
    result: a wing run's `result["record"]`, a section run's
    `result["report"]["result"]`, or the result itself when it is one; None
    for a screen or a short task."""
    if not isinstance(result, dict):
        return None
    if isinstance(result.get("record"), dict):
        return result["record"]
    rep = result.get("report")
    if isinstance(rep, dict) and isinstance(rep.get("result"), dict):
        return rep["result"]
    if "eval_y" in result or "partial" in result:
        return result
    return None


def _partial(result):
    """True / False when the result says whether the run ended early, None
    when it does not say (a short task)."""
    rec = run_record(result)
    if rec is not None and "partial" in rec:
        return bool(rec["partial"])
    if isinstance(result, dict) and "stopped" in result:
        return bool(result["stopped"])
    return None


class Job:
    """One run, cut into units. The RunManager asks `next_unit()` what the
    next unit is ("drain", "finish", or None when there is nothing left) so
    it can budget the frame, then calls `step()` -- or `finish()` for the
    "finish" unit. `run_to_end()` does all of it at once (blocking callers
    and tests). `EngineJob` is the one kind the design page runs; this base
    keeps the unit protocol the manager steps."""

    def __init__(self, info, owner, on_finish=None, notices=None, clock=None):
        self.info, self.owner = info, owner
        self.on_finish = on_finish
        self.notices = notices
        self.clock = clock if clock is not None else time.perf_counter
        self.state = "running"
        self.k = self.n = 0
        self.phase = ""
        self.records: list = []
        self.current: dict | None = None
        self.t0 = self.clock()
        self.wall = 0.0
        self.error = ""
        self.tb = ""              # the traceback of an error, for whoever debugs it
        self.abandoned = False    # RunManager.cancel: finished without on_finish
        self._begun = self._finished = self._stop_logged = False
        self._status = ("", "busy", 0.0)

    # -- lifecycle ------------------------------------------------------------
    def _log(self, text, level="info") -> None:
        if self.notices is not None and text:
            self.notices.log(text, level)

    def _toast(self, pair) -> None:
        if self.notices is not None and pair is not None:
            self.notices.toast(*pair)

    def begin(self) -> None:
        """Once: the clock starts, the start line is logged, the first status
        is set. `RunManager.start` and `run_to_end` both call it."""
        if self._begun:
            return
        self._begun = True
        self.t0 = self.clock()
        self._open()

    @property
    def done(self) -> bool:
        """No more work units (`finish()` may still be due)."""
        return self.state != "running" or self._exhausted()

    def next_unit(self):
        if self._finished:
            return None
        if self.done:
            return "finish"
        return self._unit()

    def step(self):
        """ONE unit of work. Returns the record it completed, else None. Any
        exception ends the job in state "error"."""
        if self._finished or self.done:
            return None
        if not self._begun:
            self.begin()
        try:
            return self._step()
        except Exception as exc:                          # noqa: BLE001 -- the error rule
            self.state = "error"
            self.error = str(exc) or type(exc).__name__
            self.tb = traceback.format_exc()
            return None

    def stop(self) -> None:
        """Stop after the current unit of work: nothing more is started, and
        what is on record is finished and applied like any other run."""
        if self._finished or self.state != "running" or self.done:
            return
        if not self._begun:
            self.begin()                       # the start line comes before the stop line
        self.state = "stopping"
        if not self._stop_logged:
            self._stop_logged = True
            self._log(self._stop_line(), "warn")
        self._on_stop()

    def run_to_end(self) -> "Job":
        self.begin()
        while not self.done:
            self.step()
        self.finish()
        return self

    def finish(self) -> None:
        """Once: the terminal state, the closing log lines and status, then
        `on_finish(self)` -- the model applies on the main thread."""
        if self._finished:
            return
        self._finished = True
        if not self._begun:
            self.begin()
        self.wall = self.clock() - self.t0
        if self.state == "stopping":
            self.state = "stopped"
        elif self.state == "running":
            self.state = "done"
        self._close()
        if self.on_finish is not None:
            self.on_finish(self)

    @property
    def finished(self) -> bool:
        return self._finished

    def elapsed(self) -> float:
        return self.wall if self._finished else self.clock() - self.t0

    def chip(self):
        return None

    def status(self) -> tuple:
        return self._status

    # -- what a subclass fills in -------------------------------------------
    def _open(self) -> None:
        """Log the start line(s) and set the first status."""

    def _exhausted(self) -> bool:
        """True once no work unit is left."""
        return True

    def _unit(self):
        return "eval"

    def _step(self):
        return None

    def _stop_line(self) -> str:
        return ""

    def _on_stop(self) -> None:
        """What a stop request changes at once."""

    def _close(self) -> None:
        """Log the closing line(s) and set the terminal status."""


# --------------------------------------------------------------------------- #
#  the warnings router                                                         #
# --------------------------------------------------------------------------- #
#: the deprecation categories. The Python libraries under AeroBO raise them
#: (torch's `torch.jit.script is deprecated`, botorch / gpytorch / pymoo API
#: moves): nothing a player can act on, so they never reach the Output log.
#: AeroBO's own code raises none -- its warnings are RuntimeWarnings, and J21
#: re-reads the vendored source to keep that true.
DEPRECATIONS = ("DeprecationWarning", "PendingDeprecationWarning", "FutureWarning")
#: the tail of an engine warning's one Output line
ONCE = "(once per session)"
#: a sentence end in a warning's message: the stop stays on the first
#: sentence, and what follows opens a sentence -- after a space, or straight
#: after ")." (botorch: "...float64)).Please consider scaling...")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(`“‘])|(?<=\)\.)(?=[A-Z])")
#: a captured engine line ("RuntimeWarning: <message>", the fixtures' form)
_WARNING_TEXT = re.compile(r"^([A-Za-z_]\w*Warning): (.*)\Z", re.S)


def is_deprecation(category) -> bool:
    """A warning category (class or name) that is a deprecation."""
    if isinstance(category, type) and issubclass(category, (DeprecationWarning, PendingDeprecationWarning,
                                                            FutureWarning)):
        return True
    name = getattr(category, "__name__", str(category))
    return name in DEPRECATIONS or name.endswith("DeprecationWarning")


def first_sentence(message) -> str:
    """The message up to and including its first sentence's stop."""
    m = _SENTENCE_END.search(str(message))
    return str(message)[:m.start()].rstrip() if m else str(message).strip()


def warning_parts(payload: dict):
    """(category name, message) of an engine warning's "log" event -- the
    router's carries both, a captured one is parsed from its text -- or
    None for a line that is not a warning."""
    cat, msg = payload.get("category"), payload.get("message")
    if cat is None:
        m = _WARNING_TEXT.match(str(payload.get("text", "")))
        if m is None:
            return None
        cat, msg = m.group(1), m.group(2)
    return str(cat), str(msg if msg is not None else "")


class WarningsRouter:
    """`warnings.showwarning`, wrapped once at import. A warning raised on a
    REGISTERED worker thread becomes that job's `("log", text, "warn")`
    event, so the engine's own caveats reach the Output log instead of a
    terminal nobody reads; a warning on any other thread falls through to the
    original, untouched. (`warnings.catch_warnings` swaps module-wide state
    and is not thread-safe, so it is not used.) The warnings filters still
    decide first whether a warning is shown at all."""

    def __init__(self):
        self._sinks: dict = {}
        self._lock = threading.Lock()
        self.original = None

    def install(self) -> None:
        if self.original is None:
            self.original = warnings.showwarning
            warnings.showwarning = self.show

    def register(self, emit) -> None:
        """Route the CALLING thread's warnings to `emit`."""
        with self._lock:
            self._sinks[threading.get_ident()] = emit

    def unregister(self) -> None:
        with self._lock:
            self._sinks.pop(threading.get_ident(), None)

    def routed(self) -> int:
        return len(self._sinks)

    def show(self, message, category, filename, lineno, file=None, line=None):
        sink = self._sinks.get(threading.get_ident())
        if sink is None:
            return self.original(message, category, filename, lineno, file, line)
        if is_deprecation(category):            # a library's (DEPRECATIONS): never an Output line
            return None
        sink("log", text=line_engine_warning(message, category, filename, lineno), level="warn",
             category=getattr(category, "__name__", str(category)), message=str(message))
        return None


ROUTER = WarningsRouter()
ROUTER.install()


# --------------------------------------------------------------------------- #
#  the engine job                                                              #
# --------------------------------------------------------------------------- #
class EngineJob(Job):
    """One AeroBO call on a worker thread, watched from the frame.

    `runner(emit, stop) -> result` (the bridge's, or a `ReplayRunner`) runs
    on ONE daemon thread named "carsim-aerobo-<kind>", started by `begin()`.
    `n` is the N the chip shows (the budget, or spent + extra for a Keep
    going, or the sections a screen sweeps); a "total" or a "sweep" event
    overrides it. `n_prior` is the evaluations a resumed run inherits: the
    counter opens there and the first new evaluation is n_prior + 1. `unit`
    "evaluation" | "section" (a screen). `replay` is the ReplayRunner when
    the runner is one (a harness's `release()` / `wait_paused()`); passed
    alone it IS the runner.

    Live state (read by the views while the job runs): `k` evaluations done
    (inherited counted), `n`, `records` (this run's eval payloads, in
    order), `current` (the last one), `incumbents` [(n, best, x)] each time
    the best improved, `sweep` (the last `SWEEP_KEEP` screened sections),
    `phase`, `lines` (the engine's own log lines). After the end: `result`
    (the runner's return value), `error` / `tb`, `state`, `outcome()`."""

    def __init__(self, info, owner, *, runner=None, n, n_prior=0, unit="evaluation",
                 on_finish=None, notices=None, replay=None, clock=None):
        super().__init__(info, owner, on_finish, notices, clock)
        if runner is None:
            runner = replay
        if runner is None:
            raise ValueError("an EngineJob needs a runner")
        if replay is None and isinstance(runner, ReplayRunner):
            replay = runner
        if unit not in ("evaluation", "section"):
            raise ValueError(f"unit must be 'evaluation' or 'section', got {unit!r}")
        self.runner, self.replay, self.unit = runner, replay, unit
        self.n_prior = max(0, int(n_prior))
        self.k = self.n_prior
        self.n = max(int(n), self.k)
        self.sweep: list = []
        self.incumbents: list = []
        self.lines: list = []
        self.result = None
        self.stop_requested = False
        self.worker_ident = None          # the worker thread's ident, once it runs
        self.finish_ident = None          # the thread finish() ran on (the main one)
        self._q: queue.Queue = queue.Queue()
        self._stop_ev = threading.Event()
        self._thread: threading.Thread | None = None
        self._ended = False
        self._best = None
        self.drained: list = []           # the eval / sweep records of the last drain

    # -- the thread -------------------------------------------------------------
    def begin(self) -> None:
        if self._begun:
            return
        super().begin()                        # the clock, the start line
        self._thread = threading.Thread(target=self._work, daemon=True,
                                        name=f"carsim-aerobo-{self.info.kind}")
        self._thread.start()

    def _work(self) -> None:
        """The worker: the runner, its result or its error, then the end
        marker -- always last, so a drained end means nothing else is due."""
        self.worker_ident = threading.get_ident()
        ROUTER.register(self._emit)
        try:
            res = self.runner(self._emit, self._stop_ev)
        except BaseException as exc:                     # noqa: BLE001 -- reported, not raised
            self._q.put(("__error__", (str(exc) or type(exc).__name__,
                                       traceback.format_exc())))
        else:
            self._q.put(("__result__", res))
        finally:
            ROUTER.unregister()
            self._q.put(("__end__", None))

    def _emit(self, kind, **payload) -> None:
        """The runner's only channel. A kind or level this module does not
        know raises ON THE WORKER, so the runner's typo ends the job in
        "error" instead of drawing nothing."""
        if kind not in EVENT_KINDS:
            raise ValueError(f"unknown event kind {kind!r} (expected one of {EVENT_KINDS})")
        if kind == "log" and payload.get("level", "info") not in LOG_LEVELS:
            raise ValueError(f"log level must be one of {LOG_LEVELS}, got {payload.get('level')!r}")
        self._q.put((kind, payload))

    @property
    def thread(self):
        return self._thread

    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- units ------------------------------------------------------------------
    @property
    def done(self) -> bool:
        """The worker has ended and its queue is drained: only `finish()` is
        due. A stop request does NOT make a job done -- the evaluation in
        flight still reports."""
        return self._ended

    def _exhausted(self) -> bool:
        return self._ended

    def _unit(self):
        return "drain"

    def step(self):
        if self._finished or self._ended:
            return None
        if not self._begun:
            self.begin()
        return self.drain()

    def _step(self):
        return self.drain()

    def drain(self, block_s: float = 0.0):
        """Empty the queue into the job, on the calling (main) thread. Never
        blocks unless `block_s` > 0 (then it waits that long for the FIRST
        event -- `run_all`, so a blocking caller does not spin). Returns the
        last eval / sweep record drained, or None."""
        last, first = None, True
        self.drained = []
        while True:
            try:
                if first and block_s > 0.0:
                    kind, p = self._q.get(timeout=block_s)
                else:
                    kind, p = self._q.get_nowait()
            except queue.Empty:
                break
            first = False
            if kind == "eval":
                last = self._take_eval(p)
                self.drained.append(last)
            elif kind == "sweep":
                last = self._take_sweep(p)
                self.drained.append(last)
            elif kind == "phase":
                self.phase = str(p.get("text", ""))
            elif kind == "total":
                self.n = max(int(p["n"]), self.k)
            elif kind == "log":
                text, level = str(p.get("text", "")), p.get("level", "info")
                shown = self._engine_line(p, text, level)
                if shown is False:              # a library's deprecation: dropped entirely
                    continue
                self.lines.append((text, level))
                if shown:
                    self._log(shown, level)
            elif kind == "__result__":
                self.result = p
            elif kind == "__error__":
                self.error, self.tb = p
            elif kind == "__end__":
                self._ended = True
        return last

    def _engine_line(self, p, text, level):
        """What a "log" event puts in the Output log: False for a library's
        deprecation (dropped, not even kept in `lines`), None for an engine
        warning this log has already shown, else the line -- an engine
        warning shortened to its first sentence + ONCE, any other line as it
        came."""
        parts = warning_parts(p) if level == "warn" else None
        if parts is None:
            return text
        cat, msg = parts
        if is_deprecation(cat):
            return False
        if self.notices is not None and not self.notices.first_time(("engine warning", cat, msg)):
            return None
        return line_engine_warning_once(cat, msg)

    def _take_eval(self, p) -> dict:
        rec = dict(p)
        n_ = int(rec.get("n", self.k + 1))
        rec["n"] = n_
        self.records.append(rec)
        self.current = rec
        self.k = max(self.k, n_)
        self.n = max(self.n, self.k)
        best = rec.get("best")
        if best is not None and (self._best is None or best != self._best):
            self._best = best
            #  the candidate that SET the best reports it as its own score
            #  (AeroBO's progress_cb: f == best, exactly). A best it did not
            #  set was CARRIED in from the evaluations a resumed run inherits:
            #  no evaluation of this run found it, and its design is not this
            #  candidate's -- so neither is claimed (n and x None)
            own = rec.get("f") is not None and rec.get("f") == best
            self.incumbents.append((n_ if own else None, best, rec.get("x") if own else None))
        ni, ho = self.info.n_init, self.info.handoff
        if self.n_prior > 0:
            self.phase = "BO"                  # a resumed run's new points are all BO
        elif ho and n_ > int(ho):
            self.phase = "SLSQP"               # bo_slsqp: past AeroBO's handoff
        elif ni:
            self.phase = "Sobol" if n_ <= int(ni) else "BO"
        return rec

    def _take_sweep(self, p) -> dict:
        row = dict(p)
        self.sweep.append(row)
        del self.sweep[:-SWEEP_KEEP]
        self.current = row
        self.k = max(self.k, int(row.get("i", self.k + 1)))
        if row.get("n") is not None:
            self.n = int(row["n"])
        self.n = max(self.n, self.k)
        self.phase = "shortlist sweep"
        return row

    # -- stop, finish -----------------------------------------------------------
    def _on_stop(self) -> None:
        self.stop_requested = True
        self._stop_ev.set()

    def _stop_line(self) -> str:
        kind = self.info.kind
        if kind == "screen":
            return line_screen_stop_req()
        if kind == "section":
            return line_section_stop_req()
        if kind in ("wing", "lap"):
            return line_wing_stop_req()
        return ""

    def abandon(self) -> None:
        """`RunManager.cancel`: finished WITHOUT on_finish. The thread is
        asked to stop and ends on its own after the evaluation in flight."""
        self.abandoned = True
        self.stop_requested = True
        self._stop_ev.set()
        self._finished = True
        if self.state in ("running", "stopping"):
            self.state = "stopped"
        self.wall = self.clock() - self.t0

    def run_to_end(self, timeout_s=None) -> "EngineJob":
        """Begin, join the thread, drain, finish -- blocking callers and
        tests. Raises if the thread outlives `timeout_s` (a paused replay
        never ends by itself: release it first)."""
        self.begin()
        self._thread.join(timeout_s)
        if self._thread.is_alive():
            raise RuntimeError(f"run_to_end(): the engine thread is still running after "
                               f"{timeout_s} s")
        self.drain()
        self.finish()
        return self

    def settle(self, timeout_s: float = 10.0) -> bool:
        """Wait until the worker is idle -- a replay parked at `pause_at`, or
        the thread ended -- then drain. What a harness calls to freeze a
        "running" state by construction. True when it settled in time."""
        t_end = time.perf_counter() + float(timeout_s)
        while True:
            parked = self.replay is not None and self.replay.paused.is_set()
            if parked or not self.alive():
                break
            if time.perf_counter() > t_end:
                return False
            self._thread.join(0.01)
        self.drain()
        return True

    def finish(self) -> None:
        """Once, on the main thread: the terminal state, the closing lines,
        the terminal status and the toast, then `on_finish(self)`. Blocks
        until the worker has ended if it has not (a caller other than the
        pump)."""
        if self._finished:
            return
        if self._thread is not None and threading.current_thread() is self._thread:
            raise RuntimeError("EngineJob.finish() on its own worker thread: it runs on the "
                               "main thread only")
        if not self._begun:
            self.begin()
        if not self._ended:
            self._thread.join()
            self.drain()
        self.finish_ident = threading.get_ident()
        self._finished = True
        self.wall = self.clock() - self.t0
        self.state = self._terminal()
        self._close()
        if self.on_finish is not None:
            self.on_finish(self)

    def _terminal(self) -> str:
        """error; a player's Stop -> stopped (done if the run had in fact
        completed its budget before the stop landed); a run AeroBO ended
        early by its own rule -> converged; else done."""
        if self.error:
            return "error"
        partial = _partial(self.result)
        if self.stop_requested:
            return "done" if (partial is False and self.k >= self.n) else "stopped"
        if partial:
            return "stopped" if self.stop_reason() == STOP_PLAYER else "converged"
        return "done"

    def stop_reason(self):
        rec = run_record(self.result)
        reason = rec.get("stop_reason") if rec is not None else None
        if reason:
            return reason
        return STOP_PLAYER if self.state == "stopped" else None

    def best(self):
        """The run's best score: the record's, else the last eval's best."""
        rec = run_record(self.result)
        if rec is not None and "best_score" in rec:
            return rec.get("best_score")
        return self.current.get("best") if self.records and self.current else None

    def outcome(self) -> dict:
        """What the model stores and the views draw a finished run from."""
        return {"kind": self.info.kind, "state": self.state, "k": self.k, "n": self.n,
                "n_prior": self.n_prior, "wall": self.elapsed(),
                "stop_reason": self.stop_reason() if self._finished else None,
                "error": _first_line(self.error)}

    # -- what the page shows ------------------------------------------------------
    def chip(self):
        if self._finished or not self._begun:
            return None
        return live_tag(self.k, self.n, self.state == "stopping")

    def status(self) -> tuple:
        if self._finished or not self._begun:
            return self._status
        frac = self.k / self.n if self.n else None
        if self.state == "stopping":
            return (status_stopping(self.unit), "warn", frac)
        el = self.elapsed()
        if self.info.kind not in RUN_KINDS:
            return (status_task(self.info, el), "busy", None)
        if self.unit == "section":
            name = self.current.get("name", "") if self.sweep else ""
            return (status_screen(self.k, self.n, name, self.phase, el), "busy", frac)
        phase = self.phase or ("starting" if self.k == self.n_prior else "")
        return (status_run(self.info.evaluator, self.k, self.n, phase, el, self.n_prior),
                "busy", frac)

    # -- texts ----------------------------------------------------------------------
    def _open(self) -> None:
        i, kind = self.info, self.info.kind
        if kind == "screen":
            self._log(line_screen_start(i, self.n), "info")
        elif kind == "section":
            if self.n_prior:
                self._log(line_continue(i, self.n_prior, self.n, self.n - self.n_prior), "ok")
            else:
                self._log(line_section_start(i, self.n), "info")
        elif kind in ("wing", "lap"):
            if self.n_prior:
                self._log(line_continue(i, self.n_prior, self.n, self.n - self.n_prior), "ok")
            else:
                self._log(line_wing_launch(i, self.n), "ok")

    def _close(self) -> None:
        i, kind, st = self.info, self.info.kind, self.state
        rec = run_record(self.result)
        units = rec.get("score_units") if rec is not None else None
        best = self.best()
        if st == "error":
            self._log(line_screen_failed(self.error) if kind == "screen"
                      else line_run_failed(self.error), "error")
        elif kind == "screen":
            rep = self.result.get("report") if isinstance(self.result, dict) else None
            if st == "stopped":
                self._log(line_screen_stopped(self.k, self.n), "warn")
            elif isinstance(rep, dict) and rep.get("n_screened") is not None:
                self._log(line_screen_done(rep["n_screened"], rep.get("n_eligible", 0),
                                           self.wall), "ok")
                ranked = rep.get("ranked") or []
                if ranked and ranked[0].get("composite") is not None:
                    self._log(line_screen_best(ranked[0].get("name", "?"),
                                               ranked[0]["composite"]), "ok")
            else:
                self._log(line_screen_done_bare(self.wall), "ok")
        elif kind == "section":
            if st == "stopped" and self.k <= self.n_prior:
                self._log(line_section_none(i), "warn")
            elif st == "stopped":
                self._log(line_section_stopped(i, self.k, self.n, best, units), "warn")
            else:
                reason = self.stop_reason() if st == "converged" else None
                self._log(line_section_done(i, best, self.k, self.wall, units, reason), "ok")
        elif kind in ("wing", "lap"):
            feasible = rec.get("feasible") if rec is not None else None
            ok = st in ("done", "converged") and feasible is not False and best is not None
            self._log(line_wing_done(i, best, self.k, self.n, st, self.stop_reason(), units,
                                     feasible), "ok" if ok else "warn")
        out = self.outcome()
        self._status = terminal_status(i, out)
        self._toast(toast_for(i, out))


# --------------------------------------------------------------------------- #
#  fixtures and the replay runner                                              #
# --------------------------------------------------------------------------- #
FIXTURE_FORMAT = 1
FIXTURE_KINDS = ("wing", "section", "screen", "lap")
FIXTURE_DIR = Path(__file__).resolve().parent / "data" / "aerobo_fixtures"


def fixture_path(name) -> Path:
    return FIXTURE_DIR / f"{name}.json"


def fixture_names() -> list:
    """The captured fixtures on disk (stems), sorted; [] before any capture."""
    if not FIXTURE_DIR.is_dir():
        return []
    return sorted(p.stem for p in FIXTURE_DIR.glob("*.json"))


def _evals(events) -> list:
    return [p for k, p in events if k == "eval"]


def _counted(events) -> int:
    return sum(1 for k, _p in events if k in ("eval", "sweep"))


def fixture_problems(fx) -> list:
    """Everything wrong with a fixture's shape (PLAN2 6.4), [] when none:
    {"format": 1, "kind", "meta", "events": [[kind, payload], ...], "result"}
    plus an optional "stopped" variant {"events", "result"} -- the same run
    stopped by the player, captured for real."""
    if not isinstance(fx, dict):
        return ["not a JSON object"]
    out = []
    if fx.get("format") != FIXTURE_FORMAT:
        out.append(f"format {fx.get('format')!r}, expected {FIXTURE_FORMAT}")
    if fx.get("kind") not in FIXTURE_KINDS:
        out.append(f"kind {fx.get('kind')!r} not in {FIXTURE_KINDS}")
    if not isinstance(fx.get("meta"), dict):
        out.append("meta missing")
    if "result" not in fx:
        out.append("result missing")

    def events_ok(evs, where):
        if not isinstance(evs, list):
            out.append(f"{where}events is not a list")
            return
        for j, ev in enumerate(evs):
            if (not isinstance(ev, (list, tuple)) or len(ev) != 2 or ev[0] not in EVENT_KINDS
                    or not isinstance(ev[1], dict)):
                out.append(f"{where}event {j} is not [kind, payload]")
                return
        ns = [p.get("n") for p in _evals(evs)]
        if any(not isinstance(v, int) for v in ns) or any(b <= a for a, b in zip(ns, ns[1:])):
            out.append(f"{where}eval n is not an increasing integer")

    events_ok(fx.get("events"), "")
    var = fx.get("stopped")
    if var is not None:
        if not isinstance(var, dict) or "result" not in var:
            out.append("stopped variant has no result")
        else:
            events_ok(var.get("events"), "stopped ")
    return out


def load_fixture(ref) -> dict:
    """A fixture by path, or by name from `FIXTURE_DIR`. Raises ValueError
    naming every problem with its shape."""
    p = Path(ref)
    if p.suffix != ".json" and not p.exists():
        p = fixture_path(str(ref))
    with open(p, encoding="utf-8") as fh:
        fx = json.load(fh)
    probs = fixture_problems(fx)
    if probs:
        raise ValueError(f"fixture {p.name}: " + "; ".join(probs))
    return fx


def _truncate_record(rec, n_last, dropped_feasible) -> None:
    """A run record cut to its first `n_last` evaluations, as the player's
    Stop would have left it: eval lists and the best-so-far history cut,
    partial, stop_reason STOP_PLAYER, the best re-read from AeroBO's own
    history. A best that lay beyond the cut moves to the first evaluation
    holding the new best, and its breakdown (which belonged to the old one)
    is dropped; `rec["replay"]` says so. Checked against a real capture: the
    cut equals the engine's own stopped record on n_evals, partial,
    stop_reason, best_*, feasible, n_feasible, history and the eval lists;
    only run bookkeeping (bo_iters, failures, n_screened, n_rescue*) stays as
    recorded. For screenshots and tests only -- a real stopped run is the
    fixture's "stopped" variant."""
    total = rec.get("n_evals") or len(rec.get("eval_y") or [])
    for key in ("eval_x", "eval_y", "eval_g", "history"):
        if isinstance(rec.get(key), list):
            rec[key] = rec[key][:n_last]
    rec["n_evals"] = n_last
    rec["partial"] = True
    rec["stop_reason"] = STOP_PLAYER
    hist = rec.get("history") or []
    best = hist[n_last - 1] if len(hist) >= n_last else None
    moved = best != rec.get("best_score")
    if moved:
        ys = rec.get("eval_y") or []
        i = next((j for j, y in enumerate(ys) if best is not None and y == best), None)
        if i is not None:
            rec["best_x"] = (rec.get("eval_x") or [None] * len(ys))[i]
            if isinstance(rec.get("eval_g"), list):
                rec["best_g"] = rec["eval_g"][i]
        rec["best_score"] = best
        rec["feasible"] = best is not None
        if "breakdown" in rec:
            rec["breakdown"] = None
    if isinstance(rec.get("n_feasible"), int):
        rec["n_feasible"] = max(0, rec["n_feasible"] - dropped_feasible)
    rec["replay"] = {"truncated_at": n_last, "of": total, "best_moved": bool(moved)}


class ReplayRunner:
    """A captured engine run as a runner (PLAN2 6.4, D12).

    Emits the fixture's events in order, with no sleeping, and returns its
    recorded result. With `pause_at=k` it parks after the k-th eval / sweep
    event (`paused` is set) until `release()` or the job's stop: released,
    it plays on to the end; stopped, it returns the fixture's "stopped"
    variant when that variant stopped at the same k, else the recorded
    result cut to k evaluations (`_truncate_record`). A stop that lands
    between events (no pause) is honoured before the next evaluation, as the
    engine honours it. Every call replays the same events and a fresh copy
    of the result, so two replays of one fixture are identical."""

    POLL_S = 0.01

    def __init__(self, fixture, pause_at=None):
        if isinstance(fixture, dict):
            probs = fixture_problems(fixture)
            if probs:
                raise ValueError("fixture: " + "; ".join(probs))
            self.fixture = fixture
        else:
            self.fixture = load_fixture(fixture)
        self.pause_at = None if pause_at is None else int(pause_at)
        self.paused = threading.Event()
        self._release = threading.Event()
        self.emitted = 0

    @property
    def kind(self) -> str:
        return self.fixture["kind"]

    def release(self) -> None:
        self._release.set()

    def wait_paused(self, timeout_s: float = 5.0) -> bool:
        return self.paused.wait(timeout_s)

    def __call__(self, emit, stop):
        counted, last_n = 0, None
        for kind, payload in self.fixture["events"]:
            if kind in ("eval", "sweep") and stop.is_set():
                return self._stopped(counted, last_n)
            emit(kind, **copy.deepcopy(payload))
            if kind not in ("eval", "sweep"):
                continue
            counted += 1
            self.emitted = counted
            if kind == "eval":
                last_n = int(payload["n"])
            if self.pause_at is not None and counted == self.pause_at:
                self.paused.set()
                while not (self._release.is_set() or stop.is_set()):
                    self._release.wait(self.POLL_S)
                self.paused.clear()
                if stop.is_set():
                    return self._stopped(counted, last_n)
        return copy.deepcopy(self.fixture["result"])

    def _stopped(self, counted, last_n):
        fx = self.fixture
        var = fx.get("stopped")
        if var is not None and _counted(var.get("events") or []) == counted:
            return copy.deepcopy(var["result"])
        res = copy.deepcopy(fx["result"])
        rec = run_record(res)
        if rec is not None and last_n is not None:
            dropped = sum(1 for p in _evals(fx["events"]) if p["n"] > last_n and p.get("feasible"))
            _truncate_record(rec, last_n, dropped)
        elif isinstance(res, dict):
            res["stopped"] = True
            res["partial"] = True
        return res


# --------------------------------------------------------------------------- #
#  the manager                                                                 #
# --------------------------------------------------------------------------- #
class RunManager:
    """One live run at a time, stepped from `Garage.frame`.

    `pump()` runs the frame's units: an EngineJob's drain costs microseconds,
    so each frame drains ONCE (the queue holds everything the worker said
    since the last frame); a "finish" unit -- the model applying the result
    -- never shares a frame with another unit. The EMA budgeting of a
    stepped Job is kept for API compatibility. `paused` freezes the page's
    view of a run (the screenshot harness; the worker itself is frozen only
    by a ReplayRunner's `pause_at`). `cancel()` abandons a run without
    applying it: its thread is asked to stop and DRAINS in the background,
    and the manager stays `busy` -- `start()` refuses -- until that thread
    has ended, so two engine threads never run at once. `status` is what the
    status bar shows: the live job's, else the last run's terminal status."""

    BUDGET_S = 0.010
    ALPHA = 0.3
    #: the expected cost of each unit kind before this session has timed one
    SEED_S = {"drain": 0.0002, "eval": 0.004, "finish": 0.035}

    def __init__(self, notices, clock=None):
        self.notices = notices
        self.clock = clock if clock is not None else time.perf_counter
        self.live: Job | None = None
        self.draining: list = []
        self.paused = False
        self.ema = dict(self.SEED_S)
        self.measured: dict = {}         # evaluator -> seconds an evaluation, this session
        self._paid: dict = {}            # evaluator -> [wall seconds, evaluations]
        self.status = ("Ready", "idle", None)

    def _prune(self) -> list:
        self.draining = [j for j in self.draining if j.alive()]
        if not self.draining and self.live is None and self.status == status_draining():
            self.status = ("Ready", "idle", None)
        return self.draining

    @property
    def busy(self) -> bool:
        return self.live is not None or bool(self._prune())

    def start(self, job) -> bool:
        """Make `job` the live run; False (and nothing done) while another
        is live or an abandoned one is still finishing -- the caller toasts
        (`line_busy()` for the latter)."""
        if self.busy:
            return False
        if job.notices is None:
            job.notices = self.notices
        self.live = job
        job.begin()
        self.status = job.status()
        return True

    def stop(self) -> bool:
        if self.live is None:
            return False
        self.live.stop()
        self.status = self.live.status()
        return True

    def cancel(self, why: str = "") -> None:
        """Abandon the live run WITHOUT applying it (the page it belongs to
        was re-opened, or the garage is being left): `on_finish` is never
        called, and its thread ends in the background."""
        job, self.live = self.live, None
        if job is None:
            return
        if isinstance(job, EngineJob):
            job.abandon()
            if job.alive():
                self.draining.append(job)
        else:
            job._finished = job.abandoned = True
            if job.state in ("running", "stopping"):
                job.state = "stopped"
        if why and self.notices is not None:
            self.notices.log(line_abandoned(why), "warn")
        self.status = status_draining() if self._prune() else ("Ready", "idle", None)

    def pump(self, budget_s=None) -> list:
        """Run this frame's units; returns the eval / sweep records drained."""
        if self.live is None:
            self._prune()
            return []
        if self.paused:
            return []
        budget = self.BUDGET_S if budget_s is None else float(budget_s)
        clock = self.clock
        t_frame = clock()
        out, ran = [], 0
        while self.live is not None:
            job = self.live
            unit = job.next_unit()
            if unit is None:                     # finished by someone else
                self.live = None
                break
            if ran and (unit == "finish"
                        or clock() - t_frame + self.ema.get(unit, 0.0) > budget):
                break
            t = clock()
            if unit == "finish":
                self._finish(job)
                break
            if unit == "drain":
                job.step()
                self._learn(unit, clock() - t)
                out.extend(job.drained)
                self.status = job.status()
                break                            # one drain a frame: the queue holds it all
            rec = job.step()
            self._learn(unit, clock() - t)
            ran += 1
            if rec is not None:
                out.append(rec)
            self.status = job.status()
        return out

    def _finish(self, job) -> None:
        t = self.clock()
        self.live = None                         # first: on_finish may start the next job
        job.finish()
        self._learn("finish", self.clock() - t)
        if (isinstance(job, EngineJob) and job.replay is None and job.k > job.n_prior
                and job.state in ("done", "converged", "stopped")
                and job.info.kind in ("section", "wing", "lap")):
            paid = self._paid.setdefault(job.info.evaluator, [0.0, 0])
            paid[0] += job.wall
            paid[1] += job.k - job.n_prior
            self.measured[job.info.evaluator] = paid[0] / paid[1]
        self.status = self.live.status() if self.live is not None else job.status()

    def _learn(self, unit, dt) -> None:
        a = self.ALPHA
        self.ema[unit] = (1.0 - a) * self.ema.get(unit, dt) + a * dt

    def run_all(self, max_s: float = 120.0) -> None:
        """Tests and the harness: run the live job to its finish (blocking on
        its queue, not spinning), then join any abandoned thread still
        finishing. Raises if that takes longer than `max_s`, or if the
        manager is paused."""
        if self.live is not None and self.paused:
            raise RuntimeError("run_all() on a paused run manager: unpause it first")
        t0 = time.perf_counter()

        def left():
            r = max_s - (time.perf_counter() - t0)
            if r <= 0.0:
                raise RuntimeError(f"run_all(): still running after {max_s:.0f} s")
            return r

        while self.live is not None:
            job = self.live
            if isinstance(job, EngineJob) and not job.done:
                job.drain(block_s=min(0.05, left()))
                self.status = job.status()
            else:
                self.pump(math.inf)
            left()
        for job in list(self.draining):
            job.thread.join(left())
        if self._prune():
            raise RuntimeError(f"run_all(): an abandoned run is still finishing after "
                               f"{max_s:.0f} s")

    def job_for(self, owner):
        """The LIVE job of `owner`, else None. Post-run content is read from
        the model's own run record, never from a finished job, so a re-opened
        page cannot show a run that no longer applies."""
        job = self.live
        return job if job is not None and job.owner is owner else None


# --------------------------------------------------------------------------- #
#  the texts: status bar, tags, toasts                                         #
# --------------------------------------------------------------------------- #
def clock(seconds) -> str:
    """m:ss, or h:mm:ss past the hour (whole seconds, rounded down)."""
    s = max(0, int(float(seconds)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def status_run(what, k, n, phase="", elapsed=None, n_prior=0) -> str:
    """"<what> evaluation k/N · <phase> · 2:41 · ≈ 6:10 left". The estimate
    is this run's own pace -- elapsed over the evaluations IT flew (a resumed
    run's inherited ones cost it nothing) -- once three have flown."""
    parts = [f"{what} evaluation {k}/{n}"]
    if phase:
        parts.append(phase)
    if elapsed is not None:
        parts.append(clock(elapsed))
        new = k - n_prior
        if new >= 3 and n > k:
            parts.append(f"≈ {clock(math.ceil(elapsed / new * (n - k)))} left")
    return " · ".join(parts)


def status_screen(i, n, name="", phase="", elapsed=None) -> str:
    """"screening · section i/n · <name> · 0:12"; before the first swept
    section, "screening · <phase> · 0:00"."""
    if i > 0:
        parts = ["screening", f"section {i}/{n}"] + ([name] if name else [])
    else:
        parts = ["screening"] + ([phase] if phase else [])
    if elapsed is not None:
        parts.append(clock(elapsed))
    return " · ".join(parts)


def status_task(info, elapsed=None) -> str:
    label = info.phrase or info.problem or info.kind
    return f"{label} · {clock(elapsed)}" if elapsed is not None else label


def status_stopping(unit="evaluation") -> str:
    return f"stopping after this {'section' if unit == 'section' else 'evaluation'}…"


def status_draining() -> tuple:
    return (line_busy(), "warn", None)


def live_tag(k, n, stopping=False) -> tuple:
    """The chip beside a live run's action button: (text, theme colour)."""
    return (f"STOPPING · {k}/{n}", "WARN") if stopping else (f"RUNNING · {k}/{n}", "ACCENT")


def terminal_tag(outcome):
    """(text, theme colour, note) of a FINISHED run, from the model's stored
    outcome; None for no outcome. DONE shows N/N (a run with no counted
    event, such as a screen served from the cached library pass, did all of
    its N); the others show where the run ended."""
    if not outcome:
        return None
    st, k, n = outcome.get("state"), int(outcome.get("k") or 0), int(outcome.get("n") or 0)
    if st == "done":
        kk = k if k > 0 else n
        return (f"DONE · {kk}/{n}", "GOOD", "")
    if st == "converged":
        return (f"CONVERGED · {k}/{n}", "GOOD", "stopped improving")
    if st == "stopped":
        return (f"STOPPED · {k}/{n}", "WARN",
                "stopped by you" if outcome.get("kind") == "screen" else "best kept")
    if st == "error":
        return ("FAILED", "BAD", _first_line(outcome.get("error") or ""))
    return None


def _noun(info) -> str:
    return NOUN.get(info.kind) or info.phrase or info.problem or info.kind


def terminal_status(info, outcome) -> tuple:
    """The status bar once a run has ended: (text, kind, None)."""
    st, k, n = outcome["state"], outcome["k"], outcome["n"]
    wall, noun = clock(outcome.get("wall") or 0.0), _noun(info)
    if info.kind not in RUN_KINDS:
        if st == "error":
            return (f"{noun} failed — {outcome.get('error') or ''}", "error", None)
        return (f"{noun} {'stopped' if st == 'stopped' else 'done'}",
                "warn" if st == "stopped" else "ok", None)
    if st == "error":
        return (f"{noun} failed — {outcome.get('error') or ''}", "error", None)
    if st == "stopped":
        if info.kind == "screen":
            return (f"screening stopped by you · {k}/{n} sections", "warn", None)
        return (f"{noun} stopped by you — best kept · {k}/{n} · {wall}", "warn", None)
    if st == "converged":
        return (f"{noun} converged (stopped improving) · {k}/{n} · {wall}", "ok", None)
    return (f"{noun} done · {k if k > 0 else n}/{n} · {wall}", "ok", None)


def toast_for(info, outcome):
    """The toast a run's end raises: (text, kind), or None (a short task
    that did not fail)."""
    st, k, n = outcome["state"], outcome["k"], outcome["n"]
    noun = _noun(info)
    head = noun[:1].upper() + noun[1:]
    if st == "error":
        return (f"{head} failed: {outcome.get('error') or ''}", "negative")
    if info.kind not in RUN_KINDS:
        return None
    what = "sections" if info.kind == "screen" else "evaluations"
    if st == "stopped":
        if info.kind == "screen":
            return (f"Screening stopped at {k}/{n} sections", "warning")
        return (f"{head} stopped at {k}/{n} evaluations — best kept", "warning")
    if st == "converged":
        return (f"{head} converged — it stopped improving at {k}/{n} evaluations", "positive")
    return (f"{head} done — {k if k > 0 else n}/{n} {what}", "positive")


# --------------------------------------------------------------------------- #
#  the texts: the Output log                                                   #
# --------------------------------------------------------------------------- #
def _re(re) -> str:
    return f"Re {float(re):.3g}" if re is not None else "its own Re"


def _cl(cl) -> str:
    return f"cl {float(cl):.2f}" if cl is not None else "its design cl"


def _best(info, best, units) -> str:
    text = _fmt(info, best)
    return f"{text} {units}" if units and info.fmt is None and best is not None else text


def line_screen_start(info, n) -> str:
    p = info.point or {}
    if p.get("source") == "own":
        text = (f"screening WingLab's library for the {info.surface} at {_cl(p.get('cl'))}: a "
                f"library pass at the cached point, then a live XFOIL sweep of the shortlist at "
                f"{_re(p.get('re'))}")
    else:
        text = (f"screening WingLab's library for the {info.surface} at {_cl(p.get('cl'))}, "
                f"{_re(p.get('re'))} (the cached library point) — {n} sections")
    return text + (f" — {info.phrase}" if info.phrase else "")


def line_screen_stop_req() -> str:
    return "screening will stop after the current section"


def line_screen_stopped(i, n) -> str:
    return f"screening stopped by you after {i} of {n} sections"


def line_screen_done(n, m, wall) -> str:
    return f"screened {n} sections, {m} clear the gates ({wall:.1f} s)"


def line_screen_done_bare(wall) -> str:
    return f"screening finished ({wall:.1f} s)"


def line_screen_best(name, J) -> str:
    return f"best by your weights: {name} (composite {J:.4g})"


def line_screen_dead(label, w, reason) -> str:
    """A weighted criterion this surface cannot rank (model, after a screen)."""
    return f"“{label}” carries {w:.2f} here, and {reason}"


def line_screen_failed(err) -> str:
    return f"screening failed: {err}"


def line_every_refused(n) -> str:
    """Model: a screen whose every section a gate refused."""
    return (f"every one of the {n} library sections was refused before it could be ranked — "
            f"loosen the gates")


def line_section_start(info, n) -> str:
    p = info.point or {}
    at = f" at {_re(p.get('re'))}, {_cl(p.get('cl'))}" if p else ""
    how = f"WingLab's {info.optimiser}" if info.optimiser else "WingLab"
    if info.n_init:
        how += f", the first {info.n_init} a Sobol start"
    what = info.phrase or info.objective
    return (f"shape optimisation for the {info.surface} — {what}, seeded from "
            f"{info.seed_label or 'the family’s anchor'}, {n} evaluations of a live XFOIL "
            f"sweep{at} ({how})")


def line_section_stop_req() -> str:
    return ("optimisation will stop after the current evaluation — the best section found so "
            "far is kept")


def line_section_stopped(info, k, n, best, units=None) -> str:
    return (f"shape optimisation stopped by you after {k} of {n} evaluations — the best section "
            f"is kept (objective {_best(info, best, units)})")


def line_section_none(info) -> str:
    return (f"shape optimisation stopped before any section flew, so there is nothing to keep "
            f"— {info.seed_label or 'the section you had'} still stands")


def line_section_done(info, best, k, wall, units=None, reason=None) -> str:
    tail = f" — it stopped improving ({reason})" if reason else ""
    return (f"shape optimisation finished — best objective {_best(info, best, units)} after {k} "
            f"evaluations ({wall:.1f} s){tail}")


def line_continue(info, was, total, added) -> str:
    """A Keep going: AeroBO resumes, it never re-flies."""
    if info.kind in ("wing", "lap"):
        return (f"continuing: {was} -> {total} evaluations. It RESUMES: the {was} evaluations "
                f"already paid for are this run's training set, the counter continues at "
                f"{was + 1}, and the only designs flown are the {added} new ones.")
    return (f"continuing the section search: {was} -> {total} evaluations. It RESUMES: the {was} "
            f"evaluations already paid for are its training set, the counter continues at "
            f"{was + 1}, and the only sections flown are the {added} new ones.")


def line_score(seed_J, opt_J, objective) -> str:
    """Model, after a section run: the seed and the winner on the screen's
    own weights. Logged "ok", or "warn" when the change is negative."""
    tail = ("which is what this search maximised" if str(objective).startswith("composite")
            else "not the number the search maximised")
    return (f"score under your weights: seed {seed_J:.2f} → optimised {opt_J:.2f} "
            f"({opt_J - seed_J:+.2f}) — the criteria the ranking uses, {tail}")


def line_wing_launch(info, n) -> str:
    flies = f"section {info.airfoil}" if info.airfoil else "the family’s own section"
    if info.plate:
        flies += f", plates {info.plate}"
    how = info.optimiser or "WingLab"
    if info.n_init:
        how += f", the first {info.n_init} a Sobol start"
    what = info.phrase or info.objective
    return (f"launched 1 run of {info.problem} — {what}, flying {flies}, "
            f"{n} evaluations of the {info.evaluator} ({how}), seed {info.rseed}"
            + (" (WingLab's recommended budget)" if info.recommended else ""))


def line_wing_stop_req() -> str:
    return ("stop requested — the current evaluation finishes first, and the best design found "
            "so far is kept")


def line_wing_done(info, best, k, n, state="done", reason=None, units=None,
                   feasible=None) -> str:
    head = f"{info.problem} · seed {info.rseed}"
    if best is None or feasible is False:
        word = {"stopped": "stopped by you", "converged": "converged"}.get(state, "done")
        return f"{head} {word} — no feasible design after {k} evaluations"
    b = _best(info, best, units)
    if state == "stopped":
        return (f"{head} stopped by you — best {b} after {k} of {n} evaluations (partial: Keep "
                f"going resumes it)")
    if state == "converged":
        return f"{head} converged — best {b} after {k} of {n} evaluations; {reason}"
    return f"{head} done — best {b} after {k} evaluations"


def line_engine_warning(message, category, filename, lineno) -> str:
    return (f"engine warning — {getattr(category, '__name__', category)}: {message} "
            f"({os.path.basename(str(filename))}:{lineno})")


def line_engine_warning_once(category, message) -> str:
    """An engine warning's one Output line per session (the full text stays
    in the job's `lines`)."""
    return f"engine warning — {getattr(category, '__name__', category)}: {first_sentence(message)} {ONCE}"


def line_run_failed(err) -> str:
    return f"run failed: {err}"


def line_abandoned(why) -> str:
    return f"run abandoned — nothing was applied ({why})"


def line_busy() -> str:
    """start() refused because an abandoned run's thread is still ending."""
    return "the previous run is still finishing its last evaluation"


# --------------------------------------------------------------------------- #
def _synthetic_fixture(kind="wing", n=12, n_init=4, stopped_at=None) -> dict:
    """A SELF-CHECK fixture in the captured format: NOT an engine run, and
    never shown in the game. Evaluation 3 is refused (AeroBO's -100
    penalty); the best improves at every other evaluation, so any cut moves
    it (15 at 5, 21.5 at 12)."""
    if kind == "screen":
        events = [["phase", {"text": "library pass"}]]
        for i in range(1, n + 1):
            events.append(["sweep", {"i": i, "n": n, "name": f"syn-{i:02d}", "status": "ok",
                                     "eligible": i % 5 != 0, "ldcr": 60.0 + i, "tc": 0.12}])
        rep = {"n_screened": n, "n_eligible": n - n // 5,
               "ranked": [{"name": "syn-07", "composite": 67.8568}]}
        return {"format": FIXTURE_FORMAT, "kind": "screen",
                "meta": {"captured": "synthetic", "aerobo": None, "synthetic": True, "args": {}},
                "events": events, "result": {"report": rep, "candidates": [], "stopped": False}}
    xs, ys, gs, hist, events, best, nf = [], [], [], [], [], None, 0
    for i in range(1, n + 1):
        x = [round(0.05 * i, 6), round(1.0 - 0.03 * i, 6)]
        refused = i == 3
        f = -100.0 if refused else round(10.0 + i - 0.5 * (i % 2 == 0 and i > 5), 6)
        g = [-0.2] if refused else [0.1]
        if not refused:
            nf += 1
            best = f if best is None else max(best, f)
        xs.append(x), ys.append(f), gs.append(g), hist.append(best)
        events.append(["eval", {"n": i, "best": best, "f": f, "feasible": not refused, "g": g,
                                "x": x}])
    ib = ys.index(best)
    rec = {"problem_name": "synthetic", "budget": n, "n_evals": n, "eval_x": xs, "eval_y": ys,
           "eval_g": gs, "history": hist, "best_x": xs[ib], "best_score": best, "best_g": gs[ib],
           "feasible": True, "n_feasible": nf, "partial": False, "stop_reason": None,
           "bo_split": [n_init, n - n_init], "score_units": "synthetic units",
           "breakdown": {"of": "the best design"}}
    result = {"record": rec} if kind in ("wing", "lap") else {"report": {"result": rec}}
    fx = {"format": FIXTURE_FORMAT, "kind": kind,
          "meta": {"captured": "synthetic", "aerobo": None, "synthetic": True,
                   "args": {"budget": n, "n_init": n_init}},
          "events": events, "result": result}
    if stopped_at is not None:
        var_res = copy.deepcopy(result)
        _truncate_record(run_record(var_res), stopped_at,
                         sum(1 for p in _evals(events) if p["n"] > stopped_at and p["feasible"]))
        run_record(var_res)["variant"] = "captured stop"
        fx["stopped"] = {"events": copy.deepcopy(events[:stopped_at]), "result": var_res}
    return fx


def self_check(verbose: bool = True) -> bool:
    ok = True
    t_start = time.perf_counter()

    def rep(tag, passed, msg=""):
        nonlocal ok
        ok = ok and bool(passed)
        if verbose:
            print(f"  [{'ok' if passed else 'FAIL'}] {tag}: {msg}")

    def mk(which="wing", **kw):
        base = {
            "screen": dict(kind="screen", surface="wing", stage="af", objective="composite_goal",
                           point={"re": 682486.0, "cl": 1.0, "source": "own"}),
            "section": dict(kind="section", surface="wing", stage="af", objective="composite_goal",
                            phrase="maximising the composite score of your criteria",
                            seed_label="ah80136 (library)", optimiser="bo", n_init=4,
                            point={"re": 682486.0, "cl": 1.0, "source": "own"}),
            "wing": dict(kind="wing", surface="top wing", stage="w", objective="efficiency",
                         phrase="maximising efficiency CZ/CD",
                         problem="top wing · car rear wing + endplates + free chord law (14-D)",
                         optimiser="bo_slsqp", n_init=2, airfoil="ah80136", plate="NACA 0010",
                         recommended=True),
        }[which]
        base.update(kw)
        return JobInfo(**base)

    def ev(i, best=None, f=None, feasible=True):
        f = float(i) if f is None else f
        return {"n": i, "best": float(i) if best is None else best, "f": f, "feasible": feasible,
                "g": [0.1], "x": [0.1 * i, 0.2]}

    def rec_(partial=False, reason=None, best=10.0, n=10, feasible=True):
        return {"record": {"partial": partial, "stop_reason": reason, "best_score": best,
                           "n_evals": n, "feasible": feasible, "score_units": "CZ/CD"}}

    def lines(nt):
        return [(t, lv) for _s, t, lv in nt.drain_log()]

    main = threading.get_ident()

    # J1 -- live: RUNNING · k/N and the status bar's k/N ---------------------------
    now = [100.0]
    fake = lambda: now[0]                                          # noqa: E731
    gate1, reached1, seen1 = threading.Event(), threading.Event(), []

    def r1(emit, stop):
        for i in (1, 2, 3):
            emit("eval", **ev(i))
        reached1.set()
        gate1.wait(5)
        for i in range(4, 11):
            emit("eval", **ev(i))
        return rec_()

    nt1 = Notices()
    m1 = RunManager(nt1)
    j1 = EngineJob(mk("wing"), "w", runner=r1, n=10, on_finish=seen1.append, clock=fake)
    started = m1.start(j1)
    s_first = m1.status
    reached1.wait(5)
    now[0] = 161.0
    got1 = m1.pump()
    chip1, s1 = j1.chip(), m1.status
    rep("J1 live: RUNNING · k/N chip, the status bar's k/N · phase · elapsed · ≈ left",
        started and chip1 == ("RUNNING · 3/10", "ACCENT") and j1.k == 3 and len(j1.records) == 3
        and s_first == ("wing (lattice) evaluation 0/10 · starting · 0:00", "busy", 0.0)
        and s1 == ("wing (lattice) evaluation 3/10 · BO · 1:01 · ≈ 2:23 left", "busy", 0.3)
        and [r["n"] for r in got1] == [1, 2, 3] and j1.alive() and j1.thread.daemon
        and j1.thread.name == "carsim-aerobo-wing" and len(j1.incumbents) == 3,
        f"{chip1[0]}; {s1[0]}")

    # J2 -- done: the outcome, and the DONE tag drawn FROM the outcome -----------------
    gate1.set()
    m1.run_all()
    out2 = j1.outcome()
    lg1 = lines(nt1)
    rep("J2 done: outcome state done, DONE · N/N from the outcome, terminal status, toast",
        j1.state == "done" and out2 == {"kind": "wing", "state": "done", "k": 10, "n": 10,
                                        "n_prior": 0, "wall": 61.0, "stop_reason": None,
                                        "error": ""}
        and terminal_tag(out2) == ("DONE · 10/10", "GOOD", "") and j1.chip() is None
        and m1.status == ("wing run done · 10/10 · 1:01", "ok", None) and seen1 == [j1]
        and nt1.drain_toasts() == [("Wing run done — 10/10 evaluations", "positive")]
        and lg1[-1] == ("top wing · car rear wing + endplates + free chord law (14-D) · seed 0 "
                        "done — best 10 CZ/CD after 10 evaluations", "ok")
        and not m1.busy and not j1.alive(),
        f"{terminal_tag(out2)[0]}; {m1.status[0]}")

    # J3 -- Stop: after the evaluation in flight; STOPPED · k/N; the lines -------------
    reached3 = threading.Event()

    def r3(emit, stop):
        for i in (1, 2, 3, 4):
            emit("eval", **ev(i))
        reached3.set()
        stop.wait(5)                           # the engine's stop rule, read after an evaluation
        return rec_(partial=True, reason=STOP_PLAYER, best=4.0, n=4)

    nt3 = Notices()
    m3 = RunManager(nt3)
    j3 = EngineJob(mk("wing"), "w", runner=r3, n=10)
    m3.start(j3)
    reached3.wait(5)
    m3.pump()
    m3.stop()
    m3.stop()                                  # a second press says nothing new
    chip3, st3 = j3.chip(), m3.status
    m3.run_all()
    lg3, out3 = lines(nt3), j3.outcome()
    rep("J3 Stop: STOPPING · k/N, then STOPPED · k/N (k < N), best kept, the stop lines",
        chip3 == ("STOPPING · 4/10", "WARN") and st3 == ("stopping after this evaluation…", "warn", 0.4)
        and j3.state == "stopped" and j3.k == 4 < j3.n and out3["stop_reason"] == STOP_PLAYER
        and terminal_tag(out3) == ("STOPPED · 4/10", "WARN", "best kept")
        and sum(1 for t, lv in lg3 if lv == "warn" and t == line_wing_stop_req()) == 1
        and lg3[-1] == ("top wing · car rear wing + endplates + free chord law (14-D) · seed 0 "
                        "stopped by you — best 4 CZ/CD after 4 of 10 evaluations (partial: Keep "
                        "going resumes it)", "warn")
        and nt3.drain_toasts() == [("Wing run stopped at 4/10 evaluations — best kept", "warning")],
        f"{terminal_tag(out3)[0]}; {m3.status[0]}")

    # J4 -- a rule's partial -> converged; a stop that lands after the budget -> done ---
    def r4(emit, stop):
        for i in range(1, 7):
            emit("eval", **ev(i))
        return rec_(partial=True, reason="the search stopped improving", best=6.0, n=6)

    j4 = EngineJob(mk("wing"), "w", runner=r4, n=10, notices=Notices()).run_to_end(5)
    reached4b = threading.Event()

    def r4b(emit, stop):
        for i in range(1, 11):
            emit("eval", **ev(i))
        reached4b.set()
        stop.wait(5)
        return rec_(partial=False, n=10)

    m4 = RunManager(Notices())
    j4b = EngineJob(mk("wing"), "w", runner=r4b, n=10)
    m4.start(j4b)
    reached4b.wait(5)
    m4.pump()
    m4.stop()
    m4.run_all()
    out4 = j4.outcome()
    rep("J4 a rule's partial -> converged (its stop_reason kept); a stop after the full budget -> done",
        j4.state == "converged" and out4["stop_reason"] == "the search stopped improving"
        and terminal_tag(out4) == ("CONVERGED · 6/10", "GOOD", "stopped improving")
        and j4.status() == ("wing run converged (stopped improving) · 6/10 · 0:00", "ok", None)
        and j4b.state == "done" and terminal_tag(j4b.outcome())[0] == "DONE · 10/10",
        f"{terminal_tag(out4)[0]}; stop at 10/10 -> {j4b.state}")

    # J5 -- the runner raises: error, traceback kept, FAILED --------------------------------
    def r5(emit, stop):
        emit("eval", **ev(1))
        emit("eval", **ev(2))
        raise KeyError("the polar bank lost its key")

    class Model:
        applied, state = False, None

        def on_finish(self, job):
            self.state = job.state
            if job.state != "error":           # the error rule: an error has no result
                self.applied = True

    nt5, fm = Notices(), Model()
    m5 = RunManager(nt5)
    j5 = EngineJob(mk("section"), "af", runner=r5, n=164, on_finish=fm.on_finish)
    m5.start(j5)
    m5.run_all()
    out5, lg5 = j5.outcome(), lines(nt5)
    rep("J5 a runner exception: state error, traceback kept, FAILED with the first line",
        j5.state == "error" and fm.state == "error" and not fm.applied and j5.result is None
        and j5.error == "'the polar bank lost its key'" and "KeyError" in j5.tb
        and len(j5.records) == 2
        and terminal_tag(out5) == ("FAILED", "BAD", "'the polar bank lost its key'")
        and lg5[-1] == ("run failed: 'the polar bank lost its key'", "error")
        and j5.status() == ("section optimisation failed — 'the polar bank lost its key'",
                            "error", None)
        and nt5.drain_toasts() == [("Section optimisation failed: 'the polar bank lost its key'",
                                    "negative")],
        lg5[-1][0])

    # J6 -- finish (and on_finish) on the main thread only -----------------------------------
    box6, idents6, refused6 = [], [], []

    def r6(emit, stop):
        idents6.append(threading.get_ident())
        try:
            box6[0].finish()
        except RuntimeError as exc:
            refused6.append(str(exc))
        emit("eval", **ev(1))
        return rec_(n=1)

    fin6 = []
    m6 = RunManager(Notices())
    j6 = EngineJob(mk("wing"), "w", runner=r6, n=1,
                   on_finish=lambda job: fin6.append(threading.get_ident()))
    box6.append(j6)
    m6.start(j6)
    m6.run_all()
    rep("J6 finish runs on the main thread only (idents recorded); the worker cannot finish",
        j6.finish_ident == main and fin6 == [main] and idents6 == [j6.worker_ident]
        and j6.worker_ident != main and len(refused6) == 1 and "main thread only" in refused6[0]
        and j6.state == "done",
        f"worker {j6.worker_ident}, finish {j6.finish_ident}")

    # J7 -- on_finish exactly once, however it is reached -------------------------------------
    fin7 = []

    def r7(emit, stop):
        emit("eval", **ev(1))
        return rec_(n=1)

    m7 = RunManager(Notices())
    j7 = EngineJob(mk("wing"), "w", runner=r7, n=1, on_finish=fin7.append)
    m7.start(j7)
    m7.run_all()
    j7.finish()
    j7.run_to_end(5)
    j7b = EngineJob(mk("wing"), "w", runner=r7, n=1, on_finish=fin7.append).run_to_end(5)
    j7b.finish()
    rep("J7 on_finish is called once, however the finish is reached",
        fin7 == [j7, j7b] and j7.next_unit() is None and not m7.busy)

    # J8 -- cancel: never applied, busy until the thread ends, start() refused ----------------
    reached8, gate8, applied8 = threading.Event(), threading.Event(), []

    def r8(emit, stop):
        emit("eval", **ev(1))
        reached8.set()
        stop.wait(5)
        gate8.wait(5)                          # the evaluation in flight, still finishing
        emit("eval", **ev(2))
        return rec_(partial=True, reason=STOP_PLAYER, n=2)

    nt8 = Notices()
    m8 = RunManager(nt8)
    j8 = EngineJob(mk("wing"), "w", runner=r8, n=10, on_finish=applied8.append)
    m8.start(j8)
    reached8.wait(5)
    m8.pump()
    m8.cancel("the design page was re-opened")
    other = EngineJob(mk("wing"), "w", runner=r7, n=1)
    busy_mid, refused_mid, st_mid = m8.busy, not m8.start(other), m8.status
    begun_mid = other._begun
    gate8.set()
    j8.thread.join(5)
    busy_after = m8.busy
    accepted = m8.start(other)
    m8.run_all()
    lg8 = lines(nt8)
    rep("J8 cancel: on_finish never called, busy until the thread ends, start() refused meanwhile",
        applied8 == [] and j8.abandoned and j8.finished and busy_mid and refused_mid
        and st_mid == (line_busy(), "warn", None) and not begun_mid
        and not busy_after and accepted and other.state == "done"
        and ("run abandoned — nothing was applied (the design page was re-opened)", "warn") in lg8
        and m8.job_for("w") is None,
        f"refused while draining: {refused_mid}; accepted after: {accepted}")

    # J9 -- warnings: the worker's become its log lines, the main thread's are untouched -------
    token = f"{os.getpid()}-{time.time_ns()}"
    shown = []

    def r9(emit, stop):
        warnings.warn(f"carsim self-check worker warning {token}", RuntimeWarning)
        return rec_(n=0)

    nt9 = Notices()
    j9 = EngineJob(mk("wing"), "w", runner=r9, n=1, notices=nt9)
    keep = ROUTER.original
    ROUTER.original = lambda *a, **k: shown.append(str(a[0]))
    try:
        j9.run_to_end(5)
        warnings.warn(f"carsim self-check main warning {token}", UserWarning)
    finally:
        ROUTER.original = keep
    lg9 = lines(nt9)
    wl = [t for t, lv in lg9 if lv == "warn" and "worker warning" in t]
    rep("J9 a worker's warning becomes its job's Output line (first sentence + '(once per session)'; the "
        "full text with its file:line in the job's lines); a main-thread warning falls through",
        warnings.showwarning == ROUTER.show and len(wl) == 1
        and wl[0] == f"engine warning — RuntimeWarning: carsim self-check worker warning {token} {ONCE}"
        and len(j9.lines) == 1 and j9.lines[0][1] == "warn"
        and j9.lines[0][0].startswith(f"engine warning — RuntimeWarning: carsim self-check worker warning {token} ")
        and "(design_jobs.py:" in j9.lines[0][0]
        and shown == [f"carsim self-check main warning {token}"] and ROUTER.routed() == 0,
        wl[0] if wl else "no line")

    # J10 -- replays are deterministic ---------------------------------------------------------------
    syn = _synthetic_fixture("wing", n=12, n_init=4)
    real = fixture_names()

    def replay(fx, **kw):
        j = EngineJob(mk("wing" if fx["kind"] in ("wing", "lap") else fx["kind"]), "r",
                      runner=ReplayRunner(fx, **kw),
                      n=max(1, _counted(fx["events"])),
                      unit="section" if fx["kind"] == "screen" else "evaluation")
        return j

    same10, bad10 = [], []
    for name, fx in [("synthetic", syn)] + [(nm, None) for nm in real]:
        try:
            fx = fx if fx is not None else load_fixture(name)
            a, b = replay(fx).run_to_end(10), replay(fx).run_to_end(10)
            same10.append(a.records == b.records and a.result == b.result
                          and a.records == [p for k, p in fx["events"] if k == "eval"]
                          and a.result == fx["result"] and a.result is not b.result
                          and a.state == b.state and a.k == b.k)
        except Exception as exc:                                       # noqa: BLE001
            bad10.append(f"{name}: {exc}")
    rep("J10 ReplayRunner is deterministic: the same events and result, twice (synthetic + captured)",
        all(same10) and not bad10 and len(same10) == 1 + len(real),
        f"synthetic + {len(real)} captured fixture(s)" + (f"; {bad10}" if bad10 else ""))

    # J11 -- pause_at freezes the run at k, release completes it --------------------------------------
    nt11 = Notices()
    m11 = RunManager(nt11)
    rr11 = ReplayRunner(syn, pause_at=5)
    j11 = EngineJob(mk("wing"), "w", replay=rr11, n=12)
    m11.start(j11)
    parked = rr11.wait_paused(5)
    m11.pump()
    frozen = []
    for _ in range(3):
        m11.pump()
        frozen.append((j11.k, len(j11.records), j11.chip()))
    j11.settle(5)
    rr11.release()
    m11.run_all()
    rep("J11 pause_at=5 freezes the run at 5 (chip 5/12) by construction; release() completes it",
        parked and frozen == [(5, 5, ("RUNNING · 5/12", "ACCENT"))] * 3
        and j11.state == "done" and len(j11.records) == 12 and j11.result == syn["result"]
        and j11.runner is rr11 and j11.replay is rr11,
        f"held at {frozen[0][2][0]}, then {terminal_tag(j11.outcome())[0]}")

    # J12 -- Stop during a paused replay: STOPPED with k records ------------------------------------
    def stop_at(fx, k):
        m = RunManager(Notices())
        rr = ReplayRunner(fx, pause_at=k)
        j = EngineJob(mk("wing"), "w", replay=rr, n=_counted(fx["events"]))
        m.start(j)
        rr.wait_paused(5)
        m.pump()
        m.stop()
        m.run_all()
        return j

    j12 = stop_at(syn, 5)
    r12 = j12.result["record"]
    syn_v = _synthetic_fixture("wing", n=12, n_init=4, stopped_at=5)
    j12v = stop_at(syn_v, 5)
    j12w = stop_at(syn_v, 6)                   # the variant stopped at 5, not 6: cut instead
    var_ok = []
    for nm in real:
        fx = load_fixture(nm)
        if fx.get("stopped") and fx["kind"] in ("wing", "lap", "section"):
            kv = _counted(fx["stopped"]["events"])
            jv = stop_at(fx, kv)
            var_ok.append(jv.state == "stopped" and jv.result == fx["stopped"]["result"]
                          and len(jv.records) == kv)
    rep("J12 Stop at a paused replay: STOPPED · 5/12 with 5 records; the captured variant when it "
        "stopped at the same k",
        j12.state == "stopped" and len(j12.records) == 5 and j12.k == 5
        and terminal_tag(j12.outcome()) == ("STOPPED · 5/12", "WARN", "best kept")
        and r12["n_evals"] == 5 and len(r12["eval_y"]) == len(r12["history"]) == 5
        and r12["partial"] and r12["stop_reason"] == STOP_PLAYER
        and r12["best_score"] == r12["history"][4] == 15.0 and r12["best_x"] == [0.25, 0.85]
        and r12["n_feasible"] == 4 and r12["breakdown"] is None
        and r12["replay"] == {"truncated_at": 5, "of": 12, "best_moved": True}
        and j12v.result == syn_v["stopped"]["result"] and j12w.result != syn_v["stopped"]["result"]
        and j12w.result["record"]["n_evals"] == 6 and all(var_ok),
        f"best re-read at 5: {r12['best_score']}; {len(var_ok)} captured stopped variant(s)")

    # J13 -- "total" and "sweep" set N; a screen's last swept sections --------------------------------
    sfx = _synthetic_fixture("screen", n=31)
    nt13 = Notices()
    m13 = RunManager(nt13)
    rr13 = ReplayRunner(sfx, pause_at=10)
    j13 = EngineJob(mk("screen"), "af", replay=rr13, n=24, unit="section", clock=fake)
    now[0] = 0.0
    m13.start(j13)
    st13a = j13.status()
    rr13.wait_paused(5)
    now[0] = 12.0
    m13.pump()
    st13 = m13.status
    sw = [r["i"] for r in j13.sweep]
    rr13.release()
    m13.run_all()

    def r13(emit, stop):
        emit("total", n=40)
        emit("eval", **ev(1))
        return rec_(n=1)

    j13t = EngineJob(mk("wing"), "w", runner=r13, n=53).run_to_end(5)
    rep("J13 a sweep's n and a total event set N; a screen keeps its last 8 swept sections",
        st13a == ("screening · 0:00", "busy", 0.0) and j13.n == 31 and sw == list(range(3, 11))
        and st13 == ("screening · section 10/31 · syn-10 · 0:12", "busy", 10 / 31)
        and j13.phase == "shortlist sweep" and j13.state == "done"
        and terminal_tag(j13.outcome()) == ("DONE · 31/31", "GOOD", "")
        and j13t.n == 40 and j13t.outcome()["n"] == 40,
        f"24 -> {j13.n} by the sweep, 53 -> {j13t.n} by a total")

    # J14 -- a resumed run: the counter continues, n_prior kept, the pace is its own --------------
    reached14, gate14 = threading.Event(), threading.Event()

    def r14(emit, stop):
        emit("eval", **ev(10))
        reached14.set()
        gate14.wait(5)
        for i in range(11, 17):
            emit("eval", **ev(i))
        return rec_(n=16)

    nt14 = Notices()
    m14 = RunManager(nt14)
    now[0] = 0.0
    j14 = EngineJob(mk("wing"), "w", runner=r14, n=16, n_prior=9, clock=fake)
    m14.start(j14)
    chip14a = j14.chip()
    reached14.wait(5)
    m14.pump()
    chip14b = j14.chip()
    now[0] = 30.0
    st14a = j14.status()
    gate14.set()
    m14.run_all()
    lg14 = lines(nt14)
    rep("J14 resumed: the chip opens at n_prior (9/16) and the first new evaluation is n_prior+1",
        chip14a == ("RUNNING · 9/16", "ACCENT") and chip14b == ("RUNNING · 10/16", "ACCENT")
        and j14.records[0]["n"] == 10 and j14.outcome()["n_prior"] == 9 and j14.k == 16
        and st14a[0] == "wing (lattice) evaluation 10/16 · BO · 0:30"
        and status_run("wing (lattice)", 12, 16, "BO", 30.0, n_prior=9)
        == "wing (lattice) evaluation 12/16 · BO · 0:30 · ≈ 0:40 left"
        and lg14[0] == (line_continue(j14.info, 9, 16, 7), "ok"),
        f"{chip14a[0]} -> {chip14b[0]}; the pace from 3 new evaluations, not 12")

    # J15 -- run_to_end equals the pumped run; the finish takes a frame of its own ----------------
    ja = replay(syn).run_to_end(10)
    m15 = RunManager(Notices())
    jb = replay(syn)
    m15.start(jb)
    frames, finish_frame = [], None
    jb.on_finish = lambda job: frames.append("finish")
    while m15.busy:
        before = len(frames)
        got = m15.pump()
        if frames[before:] == ["finish"] and got:
            finish_frame = "shared"
        frames.append(len(got))
        time.sleep(0.001)
    rep("J15 run_to_end equals the pumped run (records, result, outcome); finish in its own frame",
        ja.records == jb.records and ja.result == jb.result
        and {k: v for k, v in ja.outcome().items() if k != "wall"}
        == {k: v for k, v in jb.outcome().items() if k != "wall"}
        and "finish" in frames and finish_frame is None and jb.state == "done",
        f"{len(frames) - 1} pumps, finish alone")

    # J16 -- Notices levels and kinds; a runner's bad kind or level is its error --------------------
    nt16 = Notices()
    bad_level = bad_kind = False
    try:
        nt16.log("x", "loud")
    except ValueError:
        bad_level = True
    try:
        nt16.toast("x", "shout")
    except ValueError:
        bad_kind = True
    for i in range(Notices.CAP + 5):
        nt16.log(f"line {i}")
    capped = nt16.drain_log()
    j16a = EngineJob(mk("wing"), "w", runner=lambda emit, stop: emit("bogus"), n=1).run_to_end(5)
    j16b = EngineJob(mk("wing"), "w", n=1,
                     runner=lambda emit, stop: emit("log", text="x", level="loud")).run_to_end(5)
    rep("J16 Notices refuse an unknown level / kind and keep the last CAP lines; a runner's typo errs",
        bad_level and bad_kind and len(capped) == Notices.CAP and capped[-1][1] == f"line {Notices.CAP + 4}"
        and nt16.drain_log() == [] and j16a.state == "error" and "unknown event kind 'bogus'" in j16a.error
        and j16b.state == "error" and "log level" in j16b.error,
        f"{j16a.error}")

    # J17 -- screen templates ---------------------------------------------------------------------------
    nt17 = Notices()
    EngineJob(mk("screen"), "af", runner=ReplayRunner(sfx), n=24, unit="section",
              notices=nt17).run_to_end(5)
    s17 = lines(nt17)
    lib = mk("screen", surface="endplate", point={"re": 1.0e6, "cl": 0.0, "source": "library"},
             phrase="symmetric sections only, 229 of 2174")
    m17 = RunManager(nt17)
    j17 = EngineJob(lib, "ep", runner=ReplayRunner(sfx, pause_at=3), n=229, unit="section")
    m17.start(j17)
    j17.replay.wait_paused(5)
    m17.pump()
    m17.stop()
    st17 = m17.status
    m17.run_all()
    s17b = lines(nt17)
    checks17 = {
        "start (own Re)": s17[0] == (
            "screening WingLab's library for the wing at cl 1.00: a library pass at the cached "
            "point, then a live XFOIL sweep of the shortlist at Re 6.82e+05", "info"),
        "done": s17[1][1] == "ok" and s17[1][0].startswith("screened 31 sections, 25 clear the gates ("),
        "best": s17[2] == ("best by your weights: syn-07 (composite 67.86)", "ok"),
        "start (library point)": s17b[0] == (
            "screening WingLab's library for the endplate at cl 0.00, Re 1e+06 (the cached "
            "library point) — 229 sections — symmetric sections only, 229 of 2174", "info"),
        "stop requested": s17b[1] == ("screening will stop after the current section", "warn")
            and st17 == ("stopping after this section…", "warn", 3 / 31),
        "stopped": s17b[2] == ("screening stopped by you after 3 of 31 sections", "warn")
            and j17.status() == ("screening stopped by you · 3/31 sections", "warn", None)
            and terminal_tag(j17.outcome()) == ("STOPPED · 3/31", "WARN", "stopped by you"),
    }
    bad17 = [k for k, v in checks17.items() if not v]
    rep("J17 screen templates: start (own Re / library point), stop requested, stopped, done, best",
        not bad17, f"{len(checks17)} verbatim" if not bad17 else "wrong: " + ", ".join(bad17))

    # J18 -- section templates ----------------------------------------------------------------------------
    nt18 = Notices()
    sec_fx = _synthetic_fixture("section", n=12, n_init=4)
    js18 = EngineJob(mk("section"), "af", runner=ReplayRunner(sec_fx), n=12,
                     notices=nt18).run_to_end(5)
    a18 = lines(nt18)
    m18 = RunManager(nt18)
    jst = EngineJob(mk("section"), "af", runner=ReplayRunner(sec_fx, pause_at=5), n=12)
    m18.start(jst)
    jst.replay.wait_paused(5)
    m18.pump()
    ph18 = jst.phase
    m18.stop()
    m18.run_all()
    b18 = lines(nt18)
    jnone = EngineJob(mk("section"), "af", n=12, notices=nt18,
                      runner=lambda emit, stop: {"report": {"result": {"partial": True,
                                                                        "stop_reason": STOP_PLAYER}}})
    jnone.stop()
    jnone.run_to_end(5)
    c18 = lines(nt18)
    checks18 = {
        "start": a18[0] == (
            "shape optimisation for the wing — maximising the composite score of your criteria, "
            "seeded from ah80136 (library), 12 evaluations of a live XFOIL sweep at Re 6.82e+05, "
            "cl 1.00 (WingLab's bo, the first 4 a Sobol start)", "info"),
        "done": a18[1][1] == "ok" and a18[1][0].startswith(
            "shape optimisation finished — best objective 21.5 synthetic units after 12 "
            "evaluations (") and a18[1][0].endswith(" s)") and js18.state == "done",
        "phase": ph18 == "BO" and js18.phase == "BO",
        "stop requested": b18[1] == (line_section_stop_req(), "warn")
            and b18[1][0] == ("optimisation will stop after the current evaluation — the best "
                              "section found so far is kept"),
        "stopped": b18[2] == ("shape optimisation stopped by you after 5 of 12 evaluations — the "
                              "best section is kept (objective 15 synthetic units)", "warn")
            and jst.status()[0].startswith("section optimisation stopped by you — best kept · 5/12 · "),
        "nothing flew": c18[-1] == ("shape optimisation stopped before any section flew, so there is "
                                    "nothing to keep — ah80136 (library) still stands", "warn"),
    }
    bad18 = [k for k, v in checks18.items() if not v]
    rep("J18 section templates: start, stop requested, stopped, nothing flew, done",
        not bad18, f"{len(checks18)} verbatim" if not bad18 else "wrong: " + ", ".join(bad18))

    # J19 -- wing templates -----------------------------------------------------------------------------
    nt19 = Notices()
    EngineJob(mk("wing", rseed=2), "w", runner=ReplayRunner(syn), n=12, notices=nt19).run_to_end(5)
    a19 = lines(nt19)
    EngineJob(mk("wing"), "w", runner=r4, n=10, notices=nt19).run_to_end(5)
    b19 = lines(nt19)
    EngineJob(mk("wing"), "w", n=10, notices=nt19,
              runner=lambda emit, stop: rec_(best=None, n=10, feasible=False)).run_to_end(5)
    c19 = lines(nt19)
    lap = mk("wing", kind=wing_kind("laptime"), objective="laptime", recommended=False,
             phrase="minimising the lap time round arena", n_init=None, airfoil="", plate="")
    d19 = line_wing_launch(lap, 10)
    checks19 = {
        "launch": a19[0] == (
            "launched 1 run of top wing · car rear wing + endplates + free chord law (14-D) — "
            "maximising efficiency CZ/CD, flying section ah80136, plates NACA 0010, 12 evaluations "
            "of the wing (lattice) (bo_slsqp, the first 2 a Sobol start), seed 2 (WingLab's "
            "recommended budget)", "ok"),
        "done": a19[1] == ("top wing · car rear wing + endplates + free chord law (14-D) · seed 2 "
                           "done — best 21.5 synthetic units after 12 evaluations", "ok"),
        "converged": b19[-1] == ("top wing · car rear wing + endplates + free chord law (14-D) · "
                                 "seed 0 converged — best 6 CZ/CD after 6 of 10 evaluations; the "
                                 "search stopped improving", "ok"),
        "no feasible": c19[-1] == ("top wing · car rear wing + endplates + free chord law (14-D) · "
                                   "seed 0 done — no feasible design after 0 evaluations", "warn"),
        "stop requested": (line_wing_stop_req(), "warn") in lg3 and line_wing_stop_req() == (
            "stop requested — the current evaluation finishes first, and the best design found so "
            "far is kept"),
        "lap run": lap.kind == "lap" and lap.evaluator == "lap (lattice + lap)" and d19 == (
            "launched 1 run of top wing · car rear wing + endplates + free chord law (14-D) — "
            "minimising the lap time round arena, flying the family’s own section, 10 evaluations "
            "of the lap (lattice + lap) (bo_slsqp), seed 0"),
    }
    bad19 = [k for k, v in checks19.items() if not v]
    rep("J19 wing templates: launch, stop requested, stopped (J3), converged, done, no feasible, lap",
        not bad19, f"{len(checks19)} verbatim" if not bad19 else "wrong: " + ", ".join(bad19))

    # J20 -- continue lines, the status bar's clock and estimate, tags and toasts --------------------
    checks20 = {
        "continue (wing)": lg14[0][0] == (
            "continuing: 9 -> 16 evaluations. It RESUMES: the 9 evaluations already paid for are "
            "this run's training set, the counter continues at 10, and the only designs flown are "
            "the 7 new ones."),
        "continue (section)": line_continue(mk("section"), 12, 20, 8) == (
            "continuing the section search: 12 -> 20 evaluations. It RESUMES: the 12 evaluations "
            "already paid for are its training set, the counter continues at 13, and the only "
            "sections flown are the 8 new ones."),
        "clock": [clock(s) for s in (0, 7.9, 161, 3761)] == ["0:00", "0:07", "2:41", "1:02:41"],
        "estimate": status_run("section (XFOIL)", 2, 164, "Sobol", 9.0) ==
            "section (XFOIL) evaluation 2/164 · Sobol · 0:09"
            and status_run("section (XFOIL)", 23, 164, "BO", 104.2) ==
            "section (XFOIL) evaluation 23/164 · BO · 1:44 · ≈ 10:39 left",
        "tags": [terminal_tag({"state": s, "k": 7, "n": 53, "error": "boom\nat line 2"})
                 for s in ("done", "converged", "stopped", "error")] == [
            ("DONE · 7/53", "GOOD", ""), ("CONVERGED · 7/53", "GOOD", "stopped improving"),
            ("STOPPED · 7/53", "WARN", "best kept"), ("FAILED", "BAD", "boom")]
            and terminal_tag(None) is None and terminal_tag({"state": "done", "k": 0, "n": 1})[0]
            == "DONE · 1/1" and live_tag(3, 53) == ("RUNNING · 3/53", "ACCENT"),
        "task": toast_for(JobInfo("polar", phrase="the section's polar"),
                          {"state": "done", "k": 0, "n": 1}) is None
            and toast_for(JobInfo("polar", phrase="the section's polar"),
                          {"state": "error", "k": 0, "n": 1, "error": "XFOIL did not converge"})
            == ("The section's polar failed: XFOIL did not converge", "negative")
            and status_task(JobInfo("law", phrase="deriving the car's law"), 1.2)
            == "deriving the car's law · 0:01",
    }
    bad20 = [k for k, v in checks20.items() if not v]
    rep("J20 continue lines, status clock and estimate, terminal tags, a task's toast",
        not bad20, f"{len(checks20)} verbatim" if not bad20 else "wrong: " + ", ".join(bad20))

    # J21 -- the Output log's engine warnings: a library's deprecation dropped, the rest once a session --
    guide = (f"bo_feasibility='guide' was asked for, but 'x {token}' states nothing this grader can "
             f"measure (rows []). Refusals will be the flat sentinel, as with feasibility='off'.")

    def r21(emit, stop):
        warnings.warn("`torch.jit.script` is deprecated. Please switch to `torch.compile`.", DeprecationWarning)
        warnings.warn(f"an API move {token}", FutureWarning)
        warnings.warn(guide, RuntimeWarning)
        warnings.warn(guide, RuntimeWarning)
        emit("log", text="DeprecationWarning: `torch.jit.script` is deprecated. Please switch.", level="warn")
        emit("log", text=f"RuntimeWarning: {guide}", level="warn")                  # a replay's form
        emit("log", text=f"InputDataWarning: Data {token} is not standardized (std = tensor([0.], "
                         f"dtype=torch.float64)).Please consider scaling the input.", level="warn")
        return rec_(n=0)

    nt21 = Notices()
    with warnings.catch_warnings():
        warnings.simplefilter("always")                  # every call reaches the router, as the capture's did
        ja = EngineJob(mk("section"), "af", runner=r21, n=1, notices=nt21).run_to_end(5)
        jb = EngineJob(mk("section"), "af", runner=r21, n=1, notices=nt21).run_to_end(5)
        jc = EngineJob(mk("section"), "af", runner=r21, n=1, notices=Notices()).run_to_end(5)
    lg21 = [t for t, lv in lines(nt21) if lv == "warn" and "warning —" in t]
    lgc = [t for t, lv in lines(jc.notices) if lv == "warn" and "warning —" in t]
    want21 = [f"engine warning — RuntimeWarning: bo_feasibility='guide' was asked for, but 'x {token}' "
              f"states nothing this grader can measure (rows []). {ONCE}",
              f"engine warning — InputDataWarning: Data {token} is not standardized (std = tensor([0.], "
              f"dtype=torch.float64)). {ONCE}"]
    src = FIXTURE_DIR.parent.parent.parent / "aerobo" / "src" / "aerobo"
    own = [p.name for p in sorted(src.rglob("*.py")) if re.search(r"\b\w*DeprecationWarning\b|\bFutureWarning\b",
                                                                  p.read_text(errors="replace"))]
    rep("J21 engine warnings in the Output log: a library's deprecation (live or replayed) dropped; each "
        "other warning ONCE per session, first sentence + '(once per session)', amber; the full text kept "
        "in the job's lines; a new session shows it again; WingLab's own source raises no deprecation",
        lg21 == want21 and lgc == want21 and ja.state == jb.state == "done"
        and not any("eprecat" in t or "API move" in t for t, _lv in ja.lines)
        and sum(1 for t, _lv in ja.lines if guide in t) == 3 and sum(1 for t, _lv in jb.lines if guide in t) == 3
        and any(t.startswith("InputDataWarning: ") and t.endswith("Please consider scaling the input.")
                for t, _lv in ja.lines)
        and not own,
        f"{len(lg21)} line(s) for 2 jobs; WingLab source {'read' if src.is_dir() else 'absent'}"
        + (f"; deprecations in {own}" if own else "") + ("" if lg21 == want21 else f"; got {lg21}"))

    if verbose:
        print(f"  {'ALL PASS' if ok else 'FAILURES ABOVE'} ({time.perf_counter() - t_start:.1f} s)")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_check() else 1)
