"""Screenshots of every design view, before / during / after (PLAN D7, PLAN2 §9.4).

    python3 -m drive.design_shots --out DIR [--size 1280x800 --size 1600x1000]
                                  [--only af.opt] [--lib runs/library] [--no-scroll]
    python3 -m drive.design_shots            # self-check: both sizes into a temp dir
    python3 -m drive.design_shots --only af.opt   # the self-check of some views only

Every view of the two AeroBO shell pages -- the mission's three tabs and
the sixteen design steps -- is drawn in the states the walks below reach,
at 1280x800 and 1600x1000, and written out as a PNG per scroll position with
a `manifest.json` that says which AeroBO reference shot each file is to be
held against. It is how the look is checked: a view is only comparable with
AeroBO's shot of the SAME moment, scrolled the same way.

EVERY RUN IS A REPLAY. Since the pivot the design page runs AeroBO's own
engine on a worker thread (`design_jobs.EngineJob`). The harness never
starts it: `aerobo_models.replaying()` hands every job a
`design_jobs.ReplayRunner` over a run captured once from the real engine
(`drive/data/aerobo_fixtures/`, `python3 -m drive.aerobo_bridge --capture`),
through the SAME EngineJob path the game uses. No XFOIL, no torch search,
the same pictures on every machine that has the fixtures -- and every
number drawn is one AeroBO's engine produced.

THE CHAIN (the left flank; AeroBO's order). A view's "pre" is the chain
walked up to, but not including, the view's own action:

    state the mission (arena, dry, left) -> own budgets 12 / 12 / 20 (the
    captured runs') -> screen 2 Airfoil at its own Re
    -> take rank 1 -> optimise its shape (the captured 12-evaluation run)
    -> Keep going +6 -> on to 2.8 -> screen the SYMMETRIC plate sections at
    the library point -> take rank 1 -> the wing search (the captured
    20-evaluation bo_slsqp run) -> AeroBO's law and design report

"running" is the view's own action LIVE, frozen BY CONSTRUCTION: the replay
is told to park after k evaluations (`pause_at`), the harness waits until
it has (`EngineJob.settle`), drains it once and sets `runs.paused`, so no
frame moves it while it is drawn -- machine speed cannot move the picture.
The k per capture: a screen 12 sections swept, a section search 3 (inside
AeroBO's Sobol start, `running_a`) and 9 (its BO phase, `running_b`), a
wing search 6 (after the BO -> SLSQP handoff), a Keep going 3 new ones
(`continued`, the inherited evaluations shaded). "post" is the run played
to its end, "stopped" a run stopped by hand at the point the capture was
stopped at (a section at 5, the wing at 8) -- so the replay hands back the
engine's OWN stopped record, never a truncation.

A state with a different PAST -- a re-screen caught live, a second row
taken, a run stopped by hand, the plate's own optimiser, which the chain
above never runs -- is a BRANCH: the chain replayed on a fresh Garage and a
fresh temp library up to the branch point, then walked its own way. Every
step is deterministic, so the replay is the same moment.

WHAT IS NOT DRAWN, and why. The plate's library screen is ONE read of
AeroBO's warm checkpoint (a fraction of a second, no per-section event):
there is no "running" moment of it to freeze, so 2.8's screen and ranking
have no running capture. A screen stopped by hand is not drawn either: no
captured run of it exists, and a replay would show the full ranking under
a STOPPED tag.

SCROLL SERIES. Each view x state is captured at scroll 0, then every
`work_h - 40` px until the content end (`WorkUI.end()`, clamped by the
shell), so nothing below the fold goes unseen; `--no-scroll` keeps scroll 0.

Files: `DIR/{W}x{H}/{NN}_{key}_{state}_s{j}.png` (NN = the view's place in
the tree, 00-18) and `DIR/manifest.json`, one row per file:
`{"file", "key", "state", "scroll", "scroll_px", "size", "aerobo", "spec"}`
-- `aerobo` the reference shot (a 1600x1000 file; the 1280x800 captures of
the same moment point at the same one) or null, `spec` the build-sheet
section that describes the view in that state. With `--only`, the files of
the views rendered are replaced and the rest of an existing manifest kept.

Isolation: every Garage flies on its own temp library (fresh, seeded with
the bundled sections -- or a temp COPY of `--lib`), exports into a temp dir,
writes no AeroBO record (`aerobo_dir` None), and the self-check proves
`runs/` untouched. Headless always; no multiprocessing.
"""

from __future__ import annotations

import os

if __name__ == "__main__":
    #  headless before pygame is imported (drive/__init__.py's rule): this
    #  is a harness, it never opens a window
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import argparse
import fnmatch
import json
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field

import pygame

from . import design_jobs as dj
from .aero.library import Library

# --------------------------------------------------------------------------- #
#  WHAT IS DRAWN                                                               #
# --------------------------------------------------------------------------- #
#: every view of the two shell pages, in the tree's order (NN in the names)
VIEWS = ("m.operating", "m.design", "m.search",
         "af.screen", "af.rank", "af.section", "af.opt",
         "ep.screen", "ep.rank", "ep.section", "ep.opt",
         "w.type", "w.box", "w.solver", "w.conv",
         "r.summary", "r.geometry", "r.loading", "r.evals")
STAGES = ("m", "af", "ep", "w", "r")
SIZES = ((1280, 800), (1600, 1000))
STATE_ORDER = ("pre", "running", "running_a", "running_b", "continued", "post", "stopped")

#: the k a replay is parked at for each running capture (see the docstring).
#: The fixtures fix N: a section search 12 (Sobol start 4), the flank wing
#: 20 (bo_slsqp, Sobol 4, SLSQP after 5), the own-Re main screen 29 sweeps.
SCREEN_UNITS = 12       # sections swept when a live screen is captured
OPT_UNITS = (3, 9)      # section evaluations for the `_a` / `_b` captures
WING_UNITS = 6          # wing evaluations for a running capture
MORE = 6                # Keep going's extra: the captured continuations' own
MORE_UNITS = 3          # new evaluations on record when a Keep going is captured
#: the budgets the captured runs flew (A's --capture): the chain sets them as
#: Mission > Search & budget's OWN budgets right after stating the mission,
#: so every plan, log line and chip after it names the N the replay flies.
#: AeroBO's recommended plan (164 / 164 / 53) is what m.search "pre" shows.
FIXTURE_BUDGETS = {"own_af": 12, "own_ep": 12, "own_w": 20}
#: where a run is stopped by hand: the captured stop points, so the replay
#: returns the engine's own stopped record (design_jobs.ReplayRunner)
STOP_AT = {"section": 5, "wing": 8}
SCROLL_KEEP = 40        # px of the last capture repeated at the top of the next
WALL_TARGET_S = 60.0    # PLAN2 §9.4: both sizes with the scroll series
SETTLE_S = 10.0         # how long a replay may take to reach its parking point

#: (key, state, scroll index) -> AeroBO's reference shot (PLAN §5.3). The
#: shots are 1600x1000; the 1280x800 capture of the same moment points at
#: the same file. 19_wing_running_16s is the same image as 18 (both "done
#: 53/53"), so the wing's running state has no shot: its spec is §5.8.
AEROBO = {
    ("m.operating", "pre", 0): "01_mission_car.png",
    ("m.search", "pre", 0): "02_search_budget.png",
    ("af.screen", "pre", 0): "03_after_accept.png",
    ("af.screen", "running", 0): "04_screening_running.png",
    ("af.screen", "post", 0): "05_after_screen.png",
    ("af.rank", "pre", 0): "06_ranking.png",
    ("af.section", "pre", 0): "07_section.png",
    ("af.opt", "pre", 0): "08_shapeopt.png",
    ("af.opt", "running_a", 0): "09_opt_running_12s.png",
    ("af.opt", "running_a", 1): "11_opt_running_scrolled.png",
    ("af.opt", "running_b", 0): "10_opt_running_52s.png",
    ("af.opt", "stopped", 0): "12b_opt_stopped_top.png",
    ("af.opt", "stopped", 1): "12_opt_stopped.png",
    ("ep.screen", "pre", 0): "13_endplate_screen.png",
    ("w.type", "pre", 0): "14_wing_type.png",
    ("w.box", "pre", 0): "15_wing_box.png",
    ("w.solver", "pre", 0): "16_wing_solver.png",
    ("w.conv", "pre", 0): "17_wing_conv_before.png",
    ("w.conv", "post", 0): "18_wing_launch_4s.png",
    ("w.conv", "post", 1): "20_wing_conv_lower.png",
    ("w.conv", "post", 2): "21_wing_conv_bottom.png",
    ("r.summary", "post", 0): "22a_results_summary_top.png",
    ("r.summary", "post", 1): "22_results_summary.png",
    ("r.geometry", "post", 0): "23_results_geometry.png",
    ("r.loading", "post", 0): "24_results_loading.png",
    ("r.evals", "post", 0): "25_results_evals.png",
}

#: the build-sheet section that describes each view (SPECS/aerobo_*.md, and
#: PLAN §3.2 for the mission, which AeroBO's sheets leave to carsim) ...
SPEC_VIEW = {
    "m.operating": "PLAN §3.2", "m.design": "PLAN §3.2", "m.search": "PLAN §3.2",
    "af.screen": "aerobo_airfoil.md §3", "af.rank": "aerobo_airfoil.md §4",
    "af.section": "aerobo_airfoil.md §5", "af.opt": "aerobo_airfoil.md §6",
    "ep.screen": "aerobo_airfoil.md §8", "ep.rank": "aerobo_airfoil.md §8",
    "ep.section": "aerobo_airfoil.md §8", "ep.opt": "aerobo_airfoil.md §8",
    "w.type": "aerobo_wing.md §2", "w.box": "aerobo_wing.md §3",
    "w.solver": "aerobo_wing.md §4", "w.conv": "aerobo_wing.md §5",
    "r.summary": "aerobo_results.md §3", "r.geometry": "aerobo_results.md §4",
    "r.loading": "aerobo_results.md §5", "r.evals": "aerobo_results.md §6",
}
#: ... and the states that have a section of their own. A view that is
#: merely ON SCREEN while another view's job runs is the lock (shell §9.4);
#: a Keep going is PLAN2 §8.5's inherited band on the live graph.
SPEC_STATE = {
    ("af.screen", "running"): "aerobo_airfoil.md §3.1",
    ("af.screen", "post"): "aerobo_airfoil.md §3.1",
    ("af.rank", "running"): "aerobo_shell.md §9.4",
    ("af.section", "running"): "aerobo_shell.md §9.4",
    ("af.opt", "running_a"): "aerobo_airfoil.md §6.7",
    ("af.opt", "running_b"): "aerobo_airfoil.md §6.7",
    ("af.opt", "continued"): "PLAN2 §8.5",
    ("af.opt", "post"): "aerobo_airfoil.md §6.6",
    ("af.opt", "stopped"): "aerobo_airfoil.md §6.7",
    ("ep.section", "running"): "aerobo_shell.md §9.4",
    ("ep.opt", "running_a"): "aerobo_airfoil.md §6.7",
    ("w.type", "running"): "aerobo_shell.md §9.4",
    ("w.box", "running"): "aerobo_shell.md §9.4",
    ("w.solver", "running"): "aerobo_wing.md §4.6",
    ("w.conv", "pre"): "aerobo_wing.md §5.0",
    ("w.conv", "running"): "aerobo_wing.md §5.8",
    ("w.conv", "continued"): "PLAN2 §8.5",
    ("w.conv", "stopped"): "aerobo_wing.md §4.6",
    **{(k, "pre"): "aerobo_results.md §7" for k in ("r.summary", "r.geometry", "r.loading", "r.evals")},
    **{(k, "running"): "aerobo_results.md §2" for k in ("r.summary", "r.geometry", "r.loading", "r.evals")},
}


def spec_for(key: str, state: str) -> str:
    return SPEC_STATE.get((key, state), SPEC_VIEW[key])


def stage_of(key: str) -> str:
    return key.split(".", 1)[0]


# --------------------------------------------------------------------------- #
#  THE WALKS                                                                   #
# --------------------------------------------------------------------------- #
class HarnessError(RuntimeError):
    """A chain step or a capture that did not do what the harness needs --
    the chain would no longer be at the moment its captures claim."""


@dataclass
class Step:
    name: str               # the moment reached once `act` has run
    act: object             # act(session) -> None
    caps: tuple = ()        # ((key, state), ...) drawn at that moment


def _each(keys, state):
    return tuple((k, state) for k in keys)


W_VIEWS = ("w.type", "w.box", "w.solver", "w.conv")
R_VIEWS = ("r.summary", "r.geometry", "r.loading", "r.evals")

#: the main line: the first screen and run of each stage, and the results.
#: A stage the harness selects is EXPANDED in the tree from then on (nothing
#: collapses it again, AeroBO), so the views are visited in the order the
#: AeroBO session was: 4 Results is first opened after the wing has run, as
#: in shots 17-21, and its "pre" is drawn on the wing branch below.
#: Taking rank 1 is the stage's DECISION (AeroBO: the wing flies it from
#: then on); the optimiser that follows is looked at and not taken, so the
#: wing the chain flies is the captured wing run's own pair of sections
#: (hg40, mi-vawt1) and the results pages agree with the record.
MAIN = (
    Step("fresh", lambda s: s.open_mission(), (("m.operating", "pre"),)),
    Step("stated", lambda s: s.state_mission(),
         (("m.operating", "post"), ("m.design", "pre"), ("m.search", "pre"), ("af.screen", "pre"))),
    Step("own", lambda s: s.own_budgets(), (("m.search", "post"),)),
    Step("af.screening", lambda s: s.start_screen("af", SCREEN_UNITS), (("af.screen", "running"),)),
    Step("af.screened", lambda s: s.run_out(),
         (("af.screen", "post"), ("af.rank", "pre"), ("af.section", "pre"))),
    Step("af.taken", lambda s: s.take("af", 0), (("af.opt", "pre"),)),
    Step("af.opt.a", lambda s: s.start_optimise("af", OPT_UNITS[0]),
         (("af.opt", "running_a"), ("af.section", "running"))),
    Step("af.optimised", lambda s: s.run_out(), (("af.opt", "post"),)),
    Step("af.more", lambda s: s.keep_going("af", MORE_UNITS), (("af.opt", "continued"),)),
    Step("af.done", lambda s: s.run_out(then="af"), (("af.section", "post"), ("ep.screen", "pre"))),
    Step("ep.screened", lambda s: s.screen_out("ep"),
         (("ep.screen", "post"), ("ep.rank", "pre"), ("ep.section", "pre"))),
    Step("ep.taken", lambda s: s.take("ep", 0, then=True),
         (("ep.section", "post"), ("ep.opt", "pre")) + _each(W_VIEWS, "pre")),
    Step("w.running", lambda s: s.start_wing(WING_UNITS), _each(W_VIEWS, "running")),
    Step("w.done", lambda s: s.run_out(), _each(W_VIEWS, "post") + _each(R_VIEWS, "post")),
)


def _replay(upto: str) -> tuple:
    """MAIN's steps up to and including `upto`, without their captures: the
    branch point, reached again on a fresh Garage."""
    names = [st.name for st in MAIN]
    return tuple(Step(st.name, st.act) for st in MAIN[:names.index(upto) + 1])


def _stop_section(s, grp) -> None:
    """Back on rank 1 (the captured run's seed), its optimiser stopped by
    hand at the captured stop point."""
    s.take(grp, 0)
    s.start_optimise(grp, STOP_AT["section"])
    s.stop_out()


def _wing_stopped(s) -> None:
    s.start_wing(STOP_AT["wing"])
    s.stop_out()


#: the branches: a re-screen caught live (the rank view's "running"), a
#: second row taken (the rank's "post"), a section run stopped by hand, the
#: section search in its BO phase, the plate's own optimiser, and on the
#: wing the results before any run, a run stopped by hand and its Keep going
BRANCHES = (
    ("airfoil", _replay("af.screened") + (
        Step("af.rescreening", lambda s: s.start_screen("af", SCREEN_UNITS), (("af.rank", "running"),)),
        Step("af.rescreened", lambda s: s.run_out(), ()),
        Step("af.rank.2", lambda s: s.take("af", 1), (("af.rank", "post"),)),
        Step("af.opt.stopped", lambda s: _stop_section(s, "af"), (("af.opt", "stopped"),)),
    )),
    ("airfoil_b", _replay("af.taken") + (
        Step("af.opt.b", lambda s: s.start_optimise("af", OPT_UNITS[1]), (("af.opt", "running_b"),)),
    )),
    ("endplate", _replay("ep.screened") + (
        Step("ep.rank.2", lambda s: s.take("ep", 1), (("ep.rank", "post"),)),
        Step("ep.opt.a", lambda s: (s.take("ep", 0), s.start_optimise("ep", OPT_UNITS[0])),
             (("ep.opt", "running_a"), ("ep.section", "running"))),
        Step("ep.optimised", lambda s: s.run_out(), (("ep.opt", "post"),)),
    )),
    ("wing", _replay("ep.taken") + (
        Step("r.before", lambda s: None, _each(R_VIEWS, "pre")),
        Step("w.stopped", _wing_stopped, (("w.conv", "stopped"),)),
        Step("w.more", lambda s: s.keep_going("w", MORE_UNITS),
             (("w.conv", "continued"),) + _each(R_VIEWS, "running")),
    )),
)
WALKS = (("main", MAIN),) + BRANCHES


def all_captures() -> list:
    """Every (key, state) the walks draw, in view then state order."""
    caps = {c for _, steps in WALKS for st in steps for c in st.caps}
    return sorted(caps, key=lambda c: (VIEWS.index(c[0]), STATE_ORDER.index(c[1])))


def select_keys(only) -> list:
    """`--only` tokens -> view keys. A token is a view key (`af.opt`), a
    stage (`af`), or a pattern (`r.*`); commas and spaces separate them."""
    if not only:
        return list(VIEWS)
    toks = [t for item in only for t in str(item).replace(",", " ").split() if t]
    keys = set()
    for t in toks:
        hit = ([k for k in VIEWS if stage_of(k) == t.rstrip(".")] if t.rstrip(".") in STAGES
               else [k for k in VIEWS if fnmatch.fnmatchcase(k, t)])
        if not hit:
            raise ValueError(f"--only {t!r}: not a view, a stage or a pattern of them "
                             f"({', '.join(VIEWS)})")
        keys.update(hit)
    return [k for k in VIEWS if k in keys]


# --------------------------------------------------------------------------- #
#  ONE WALK                                                                    #
# --------------------------------------------------------------------------- #
@dataclass
class Shot:
    file: str               # relative to the output dir
    key: str
    state: str
    scroll: int             # index in the view's scroll series
    scroll_px: int
    size: str

    def row(self) -> dict:
        return {"file": self.file, "key": self.key, "state": self.state,
                "scroll": self.scroll, "scroll_px": self.scroll_px, "size": self.size,
                "aerobo": AEROBO.get((self.key, self.state, self.scroll)),
                "spec": spec_for(self.key, self.state)}


@dataclass
class Result:
    out: str
    shots: list = field(default_factory=list)
    failures: list = field(default_factory=list)     # what did not render, and why
    errors: list = field(default_factory=list)       # error lines the session logged
    unfrozen: list = field(default_factory=list)     # a capture whose run moved or ran
    walks: list = field(default_factory=list)        # (label, steps, captures, seconds)
    runs: list = field(default_factory=list)         # (walk, kind, fixture) of every run started
    worst_ms: float = 0.0                            # the slowest shell frame drawn
    wall: float = 0.0


class Session:
    """One walk: a headless Garage on a fresh temp library, the stated
    mission, every engine run replayed from the captured fixtures, exports
    into the temp dir, no AeroBO record written, the pointer parked off the
    window (no hover), and the helpers the steps are written with.
    Everything goes through the Garage's, the DesignPage's, the models' and
    the RunManager's public entry points (the shell's own per-view `scroll` /
    `content_h` / `view_errors` for the series); the one private call is
    `g._draw_page()`, which PLAN §5.3 names."""

    def __init__(self, label, size, tmp, lib_src, out, res, replay, *, scroll=True):
        from .garage import Garage, CarBuild           # lazily: the garage imports the shell
        self.label, self.size, self.out, self.res = label, tuple(size), out, res
        self.rp = replay                               # aerobo_models.Replay
        self.scroll = scroll
        self.tag = f"{size[0]}x{size[1]}"
        root = tempfile.mkdtemp(prefix=f"walk_{label}_{self.tag}_", dir=tmp)
        lib_dir = os.path.join(root, "library")
        if lib_src:
            shutil.copytree(lib_src, lib_dir)
        lib = Library(lib_dir, use_xfoil=False)
        self.g = g = Garage(self.size, CarBuild(), headless=True, lib=lib)
        g.aerobo_dir = None                            # no run record under runs/
        g.export_dir = os.path.join(root, "export")
        g.shell.mouse = (-1, -1)
        #  every error line the session logs is a failure of the harness
        #  run: a view that raised (the shell logs it once), a run that
        #  failed, a chrome read that threw
        real_log = g.notices.log

        def log(text, level="info"):
            if level == "error":
                res.errors.append(f"{self.label} {self.tag}: {text}")
            return real_log(text, level)
        g.notices.log = log

    # -- the models, as the steps name them -----------------------------------------
    def model(self, grp):
        s = self.g.design_page.session
        if s is None:
            raise HarnessError("no design session: the mission was not stated")
        return s.af if grp == "af" else s.ep

    def _job(self):
        job = self.g.runs.live
        if job is None:
            raise HarnessError("nothing is running: the run finished (or never started) "
                               "before its capture")
        return job

    # -- steps ----------------------------------------------------------------------
    def open_mission(self) -> None:
        """The garage's way in (D on the car page), on the published mission."""
        g = self.g
        g.open_mission("left")
        mp = g.mission_page
        mp.params.param("track").set("arena")
        mp.params.param("surf").set("dry")

    def state_mission(self) -> None:
        g = self.g
        if not g.state_mission() or not g.mission.stated or g.page != "section":
            raise HarnessError(f"the mission did not state (page {g.page!r})")
        #  a side wing's job opens 3 Wing on side force; the captured wing
        #  fixtures were flown at AeroBO's default, efficiency, so the harness
        #  picks it (the player may, on any job)
        g.design_page.wing._set_objective("efficiency")

    def own_budgets(self) -> None:
        """Mission > Search & budget: "own", at the captured runs' budgets
        (the form's own rows, as the player types them)."""
        mp = self.g.mission_page
        mp.params.param("smode").set("own")
        for k, v in FIXTURE_BUDGETS.items():
            mp.params.param(k).set(v)
        pol = self.g.policy
        got = {"own_af": pol.own["airfoil"], "own_ep": pol.own["plate"], "own_w": pol.own["wing"]}
        if pol.recommended() or got != FIXTURE_BUDGETS:
            raise HarnessError(f"the own budgets did not take: {pol.mode} {got}")

    def _launch(self, start, kind, n, what) -> None:
        """Start a run with its replay parked after `n` evaluations (sections
        swept, for a screen), wait until it has parked, drain it once and
        freeze the page (`runs.paused`). `n` None: run it out instead."""
        runs = self.g.runs
        self.rp.pause_at = n
        try:
            ok = start()
        finally:
            self.rp.pause_at = None
        job = runs.live
        if not ok or not isinstance(job, dj.EngineJob) or job.info.kind != kind:
            got = f"{type(job).__name__} {getattr(getattr(job, 'info', None), 'kind', '')}"
            raise HarnessError(f"{what} did not start live (got {got.strip()}; "
                               f"{self.g.notices.drain_toasts()[-1:]})")
        rr = job.replay
        if rr is None:
            raise HarnessError(f"{what} is not a replay: the harness never starts the engine")
        args = (rr.fixture.get("meta") or {}).get("args") or {}
        self.res.runs.append((self.label, kind, args.get("variant") or args.get("re_source") or ""))
        if n is None:
            return
        if not job.settle(SETTLE_S):
            raise HarnessError(f"{what}: the replay did not reach {n} in {SETTLE_S:.0f} s")
        runs.pump(0.0)
        if not rr.paused.is_set():
            raise HarnessError(f"{what} ended after {rr.emitted} of the {n} its capture needs "
                               f"({job.state}{': ' + job.error if job.error else ''})")
        runs.paused = True

    def start_screen(self, grp, n) -> None:
        m = self.model(grp)
        if m.effective_re_source() != "mission":
            raise HarnessError(f"{grp}: a live screen is the own-Re one (a sweep per section); "
                               f"without XFOIL the model screens the library point, one read")
        self._launch(m.start_screen, "screen", n, f"the {grp} screen")

    def screen_out(self, grp) -> None:
        """Screen at the LIBRARY point and run it out: the plate's (its only
        captured screen) -- one read of AeroBO's warm checkpoint, no
        per-section event, so nothing to catch running."""
        m = self.model(grp)
        m.re_source = "library"
        self._launch(m.start_screen, "screen", None, f"the {grp} screen")
        self.run_out()
        if not m.ranked:
            raise HarnessError(f"{grp}: the library screen ranked nothing ({m.msg})")

    def start_optimise(self, grp, n) -> None:
        m = self.model(grp)
        #  the plate's shape search is at its OWN Re (the captured run's);
        #  its screen was read at the library point
        m.re_source = "mission"
        self._launch(lambda: m.start_optimise(), "section", n,
                     f"the {grp} optimiser ({m.msg or 'no message'})")

    def start_wing(self, n) -> None:
        dp = self.g.design_page
        self._launch(lambda: dp.run_wing(), "wing", n, f"the wing run ({dp.wing.msg or 'no message'})")

    def keep_going(self, grp, n) -> None:
        """Keep going (K) with the captured continuation's own extra."""
        dp = self.g.design_page
        if grp == "w":
            dp.wing.more = MORE
            self._launch(lambda: dp.run_wing(extend=True), "wing", n, "the wing's Keep going")
        else:
            m = self.model(grp)
            m.opt["more"] = MORE
            self._launch(lambda: m.start_optimise(extend=True), "section", n,
                         f"the {grp} Keep going ({m.msg or 'no message'})")
        job = self.g.runs.live
        if not job.info.continued or job.n_prior <= 0:
            raise HarnessError(f"{grp}: Keep going did not resume (n_prior {job.n_prior})")

    def run_out(self, then=None) -> None:
        """Let the live run play on to its end (and anything its finish
        starts: the wing's law and design report). `then` moves on to the
        stage after that group, as taking a section does."""
        runs = self.g.runs
        job = self._job()
        runs.paused = False
        if job.replay is not None:
            job.replay.release()
        runs.run_all()
        if job.state not in ("done", "converged"):
            raise HarnessError(f"the run ended {job.state!r}, not done ({job.error})")
        if then:
            self.g.design_page.advance(then)

    def stop_out(self) -> None:
        """Stop the live run the way ESC / Stop does, and let it finish."""
        runs = self.g.runs
        job = self._job()
        runs.paused = False
        runs.stop()
        runs.run_all()
        if job.state != "stopped":
            raise HarnessError(f"the run ended {job.state!r}, not stopped ({job.error})")

    def take(self, grp, i, then=False) -> None:
        """Take ranked row `i` (ENTER on the ranking): the stage's decision."""
        m = self.model(grp)
        if len(m.ranked) <= i:
            raise HarnessError(f"{grp}: no rank {i + 1} to take ({len(m.ranked)} ranked)")
        m.use_ranked(i)
        if m.decision != "library" or m.highlight != i:
            raise HarnessError(f"{grp}: rank {i + 1} was not taken ({m.msg})")
        if then:
            self.g.design_page.advance(grp)

    # -- the shell --------------------------------------------------------------------
    def select(self, key) -> None:
        """Show `key` through the shell's own select (the tree / tab path),
        and make sure it is what is showing. A view of a LOCKED stage (4
        Results before a run) is shown by the navigator's forced select, as
        the shell's own check does: its empty state is part of the contract."""
        g = self.g
        stage = stage_of(key)
        if not g.shell.select(stage, key) and stage != "m":
            g.notices.drain_toasts()
            g.design_page.nav.select(key, force=True)
            g.page = "section"
        if stage == "m":
            ok = g.page == "mission" and g.mission_tab == key
        else:
            ok = g.page == "section" and g.design_page.nav.current() == key
        if not ok:
            raise HarnessError(f"{key} could not be selected (page {g.page!r}, showing "
                               f"{g.mission_tab if g.page == 'mission' else g.design_page.nav.current()})")

    def _marker(self):
        """What a pump would move: the pause flag, the live job, its counter,
        its record and its state."""
        runs = self.g.runs
        job = runs.live
        return (runs.paused, job, None if job is None else (job.k, len(job.records), job.state))

    def draw(self, timed=False) -> None:
        g = self.g
        g.shell.mouse = (-1, -1)
        _hold_toasts(g)
        t = time.perf_counter()
        g._draw_page()
        if timed:
            self.res.worst_ms = max(self.res.worst_ms, (time.perf_counter() - t) * 1e3)

    def shoot(self, key, state) -> None:
        """The scroll series of one view in one state. Each position is
        drawn twice -- the first frame settles what a frame learns from the
        one before it (the content height the scroll is clamped with, the
        reveal, the Output's wrap) -- and the second is the picture."""
        g = self.g
        self.select(key)
        running = state.startswith("running") or state == "continued"
        nn = VIEWS.index(key)
        size_dir = os.path.join(self.out, self.tag)
        os.makedirs(size_dir, exist_ok=True)
        want, j = 0, 0
        while True:
            _set_scroll(g, key, want)
            before = self._marker()
            self.draw()
            self.draw(timed=True)
            after = self._marker()
            if running:
                if not (before[0] and before[1] is not None and before == after):
                    self.res.unfrozen.append(f"{self.tag} {key} {state}: paused {before[0]}, "
                                             f"live {before[1] is not None}, "
                                             f"{before[2]} -> {after[2]}")
            elif g.runs.busy:
                self.res.unfrozen.append(f"{self.tag} {key} {state}: a run is live "
                                         f"({type(g.runs.live).__name__})")
            err = g.shell.view_errors.get(key)
            if err:
                self.res.errors.append(f"{self.tag} {key} {state} s{j}: render error: "
                                       f"{err.strip().splitlines()[-1]}")
            px = _get_scroll(g, key)
            name = f"{nn:02d}_{key}_{state}_s{j}.png"
            pygame.image.save(g.screen, os.path.join(size_dir, name))
            self.res.shots.append(Shot(f"{self.tag}/{name}", key, state, j, px, self.tag))
            if not self.scroll:
                break
            work_h = _work_rect(g).h
            end = max(0, _content_h(g, key) - work_h)
            if px >= end:
                break
            j += 1
            want = min(j * (work_h - SCROLL_KEEP), end)
        _set_scroll(g, key, 0)

    def close(self) -> None:
        """A run still parked when the walk ends is let go and joined, so no
        replay thread outlives its walk."""
        runs = self.g.runs
        job = runs.live
        if job is None:
            return
        runs.paused = False
        if getattr(job, "replay", None) is not None:
            job.replay.release()
        try:
            runs.run_all(max_s=SETTLE_S)
        except RuntimeError:
            runs.cancel("the screenshot walk ended")


# --------------------------------------------------------------------------- #
#  THE SHELL, READ AND SET FROM OUTSIDE                                        #
# --------------------------------------------------------------------------- #
def _work_rect(g) -> pygame.Rect:
    """The work area the view is drawn in (PLAN §4.1, the shell's layout)."""
    W, H = g.screen.get_size()
    return pygame.Rect(g.shell.geom(W, H).work)


def _set_scroll(g, key, px) -> None:
    """The work-area scroll of view `key`; the shell clamps it on the next
    draw with the content height of the last one."""
    g.shell.scroll[key] = int(px)


def _get_scroll(g, key) -> int:
    return int(g.shell.scroll.get(key, 0))


def _content_h(g, key) -> int:
    """The view's content height as its last draw laid it out (`WorkUI.end()`)."""
    return int(g.shell.content_h.get(key, 0))


def _hold_toasts(g) -> None:
    """A toast lives 5 s from its first draw: on a slow machine it would be
    gone by the third capture of a moment, on a fast one still there at the
    tenth. Each capture shows it as if just raised, so the pictures do not
    depend on how fast they were taken."""
    for it in g.shell.toasts.items:
        it["t0"] = None


def _clear_toasts(g) -> None:
    """What the last moment raised is not part of the next one -- whether
    or not it was drawn (with `--only`, a moment may have drawn nothing and
    left its toasts queued)."""
    g.notices.drain_toasts()
    g.shell.toasts.items.clear()


# --------------------------------------------------------------------------- #
#  RENDER                                                                      #
# --------------------------------------------------------------------------- #
def _runs_snapshot() -> list:
    """runs/ under the working directory and under the repo (usually the
    same dir): every path, size and mtime, so any write shows. Except the
    garage self-check's own screenshots (`runs/garage_*.png`): that check
    writes them there by design, and one running beside this harness (the
    other agents' acceptance runs) would fail it for a write it never made."""
    roots = {os.path.realpath(os.path.join(p, "runs"))
             for p in (os.getcwd(), os.path.dirname(os.path.dirname(os.path.abspath(__file__))))}
    out = []
    for root in sorted(roots):
        if not os.path.isdir(root):
            continue
        for d, dirs, files in os.walk(root):
            dirs.sort()
            for n in sorted(dirs + files):
                if d == root and fnmatch.fnmatchcase(n, "garage_*.png"):
                    continue
                try:
                    st = os.stat(os.path.join(d, n))
                except OSError:
                    continue
                out.append((os.path.join(d, n), st.st_size, st.st_mtime_ns))
            if len(out) > 50000:
                break
    return out


def _plan(keys, walks=WALKS) -> list:
    """(label, steps to run) per walk that has any of `keys` to draw; a
    walk stops at its last wanted moment."""
    out = []
    for label, steps in walks:
        last = max((i for i, st in enumerate(steps) if any(k in keys for k, _ in st.caps)),
                   default=-1)
        if last >= 0:
            out.append((label, steps[:last + 1]))
    return out


def render(out, *, sizes=SIZES, only=None, lib=None, scroll=True, verbose=True) -> Result:
    """Draw the views in `only` (all by default) at every size in `sizes`
    into `out` and write `out/manifest.json`. Returns what was drawn and
    everything that went wrong; raises nothing a view or a step throws."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    from . import aerobo_models as am                  # lazily: it imports AeroBO
    keys = select_keys(only)
    out = os.path.abspath(out)
    os.makedirs(out, exist_ok=True)
    res = Result(out)
    t0 = time.perf_counter()
    tmp = tempfile.mkdtemp(prefix="carsim_shots_")
    try:
        lib_src = None
        if lib:
            lib_src = os.path.join(tmp, "lib_src")
            shutil.copytree(lib, lib_src)
        with am.replaying() as rp:
            for W, H in sizes:
                tag = f"{W}x{H}"
                size_dir = os.path.join(out, tag)
                os.makedirs(size_dir, exist_ok=True)
                for k in keys:                        # a shorter series must not leave old files
                    for f in os.listdir(size_dir):
                        if f.startswith(f"{VIEWS.index(k):02d}_{k}_") and f.endswith(".png"):
                            os.remove(os.path.join(size_dir, f))
                for label, steps in _plan(keys):
                    _walk(label, steps, (W, H), tmp, lib_src, out, res, keys, scroll, verbose, rp)
        _write_manifest(out, res, keys, sizes)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    res.wall = time.perf_counter() - t0
    return res


def _walk(label, steps, size, tmp, lib_src, out, res, keys, scroll, verbose, rp) -> None:
    t0 = time.perf_counter()
    n0 = len(res.shots)
    tag = f"{size[0]}x{size[1]}"
    rp.pause_at = None
    try:
        s = Session(label, size, tmp, lib_src, out, res, rp, scroll=scroll)
    except Exception as exc:                                   # noqa: BLE001
        res.failures.append(f"{label} {tag}: the Garage did not build: {exc!r}")
        return
    done = 0
    try:
        for st in steps:
            _clear_toasts(s.g)
            try:
                st.act(s)
            except Exception as exc:                           # noqa: BLE001
                #  the chain is no longer at the moment its captures claim:
                #  every capture from here on in this walk is lost, and says so
                lost = [f"{k} {state}" for later in steps[done:] for k, state in later.caps
                        if k in keys]
                res.failures.append(f"{label} {tag}: step {st.name!r} failed: {exc!r}"
                                    + (f" -- not drawn: {', '.join(lost)}" if lost else ""))
                break
            done += 1
            for key, state in st.caps:
                if key not in keys:
                    continue
                try:
                    s.shoot(key, state)
                except Exception as exc:                       # noqa: BLE001
                    res.failures.append(f"{label} {tag}: {key} {state}: {exc!r}")
    finally:
        s.close()
    dt = time.perf_counter() - t0
    res.walks.append((f"{label} {tag}", done, len(res.shots) - n0, dt))
    if verbose:
        print(f"  [walk] {label:9s} {tag:9s} {done:2d} steps, {len(res.shots) - n0:3d} files, "
              f"{dt:5.1f} s", flush=True)


def _write_manifest(out, res, keys, sizes) -> None:
    """`out/manifest.json`: this run's files, plus the rows of an earlier
    manifest for the views and sizes this run did not draw."""
    path = os.path.join(out, "manifest.json")
    drawn = {(f"{W}x{H}", k) for W, H in sizes for k in keys}
    rows = []
    try:
        with open(path) as fh:
            rows = [r for r in json.load(fh) if (r.get("size"), r.get("key")) not in drawn
                    and os.path.exists(os.path.join(out, r.get("file", "")))]
    except (OSError, ValueError, TypeError, AttributeError):
        rows = []
    rows += [sh.row() for sh in res.shots]
    order = {f"{W}x{H}": i for i, (W, H) in enumerate(SIZES)}

    def place(seq, v):                    # a row an older harness wrote sorts last
        return seq.index(v) if v in seq else len(seq)
    rows.sort(key=lambda r: (order.get(r["size"], len(order)), str(r["size"]),
                             place(VIEWS, r["key"]), place(STATE_ORDER, r["state"]),
                             int(r.get("scroll") or 0)))
    with open(path, "w") as fh:
        json.dump(rows, fh, indent=1)


# --------------------------------------------------------------------------- #
#  SELF-CHECK                                                                  #
# --------------------------------------------------------------------------- #
def _single_colour(path) -> bool:
    arr = pygame.surfarray.array3d(pygame.image.load(path))
    return bool((arr == arr[0, 0]).all())


def self_check(verbose: bool = True, *, sizes=SIZES, only=None, lib=None, scroll=True) -> bool:
    """Every view, state and scroll position at both sizes, into a temp dir
    (`--only` narrows it to some views: a W3 agent's gate, PLAN §1.4)."""
    ok = True
    rows = 0

    def rep(tag, passed, msg=""):
        nonlocal ok, rows
        ok = ok and bool(passed)
        rows += 1
        if verbose:
            print(f"  [{'ok' if passed else 'FAIL'}] {tag}: {msg}", flush=True)

    keys = select_keys(only)
    tags = [f"{W}x{H}" for W, H in sizes]
    before = _runs_snapshot()
    tmp = tempfile.mkdtemp(prefix="carsim_shots_check_")
    try:
        res = render(tmp, sizes=sizes, only=only, lib=lib, scroll=scroll, verbose=verbose)
        want = [c for c in all_captures() if c[0] in keys]
        got = {(sh.size, sh.key, sh.state) for sh in res.shots}
        missing = [f"{t} {k} {s}" for t in tags for k, s in want if (t, k, s) not in got]
        per = {t: sum(sh.size == t for sh in res.shots) for t in tags}
        rep(f"every view x state x scroll position drew at {' and '.join(tags)}",
            not res.failures and not missing,
            f"{len(want)} views x states, {len(res.shots)} files "
            f"({', '.join(f'{t}: {n}' for t, n in per.items())}); worst frame "
            f"{res.worst_ms:.0f} ms"
            + (f"; FAILED: {' | '.join(res.failures[:6])}" if res.failures else "")
            + (f"; MISSING: {', '.join(missing[:8])}" if missing else ""))

        paths = [os.path.join(tmp, sh.file) for sh in res.shots]
        absent = [p for p in paths if not os.path.isfile(p) or os.path.getsize(p) == 0]
        flat = [os.path.relpath(p, tmp) for p in paths if p not in absent and _single_colour(p)]
        rep("every file exists and is not a single colour", bool(paths) and not absent and not flat,
            f"{len(paths)} PNGs" + (f"; absent {absent[:4]}" if absent else "")
            + (f"; one colour: {flat[:4]}" if flat else ""))

        with open(os.path.join(tmp, "manifest.json")) as fh:
            man = json.load(fh)
        on_disk = sorted(os.path.relpath(os.path.join(d, f), tmp).replace(os.sep, "/")
                         for d, _, fs in os.walk(tmp) for f in fs if f.endswith(".png"))
        unexplained = [r["file"] for r in man if not r.get("aerobo") and not r.get("spec")]
        #  a shot of the TOP of a view is met by every capture of it; a
        #  scrolled one only once the view's content reaches that far down,
        #  so those are reported, not required
        unmet, deep = [], set()
        for t in tags:
            met = {(r["key"], r["state"], r["scroll"]) for r in man
                   if r["size"] == t and r.get("aerobo")}
            for k, shot in AEROBO.items():
                if k[0] not in keys or k in met:
                    continue
                if k[2] == 0:
                    unmet.append(f"{t} {shot}")
                else:
                    deep.add(shot)
        rep("manifest.json lists every file with its WingLab shot or its spec section",
            sorted(r["file"] for r in man) == on_disk and not unexplained and not unmet,
            f"{len(man)} rows, {sum(bool(r.get('aerobo')) for r in man)} with a shot, the rest "
            f"a spec section"
            + (f"; not scrolled that far: {', '.join(sorted(deep))}" if deep else "")
            + (f"; no shot or spec: {unexplained[:4]}" if unexplained else "")
            + (f"; top-of-view shots with no capture: {unmet}" if unmet else ""))

        n_run = sum(sh.state.startswith("running") or sh.state == "continued" for sh in res.shots)
        wants_run = any(s.startswith("running") or s == "continued" for _, s in want)
        kinds = sorted({k for _w, k, _f in res.runs})
        rep("running captures are frozen (a parked replay, runs.paused); every run was a replay",
            (n_run > 0 or not wants_run) and not res.unfrozen,
            f"{n_run} running captures, {len(res.runs)} replayed runs ({', '.join(kinds)})"
            + (f"; MOVED: {res.unfrozen[:3]}" if res.unfrozen else ""))

        rep("no exception was logged", not res.errors,
            "no error line in any session's log, no view drew its render-error card"
            if not res.errors else " | ".join(res.errors[:4]))

        after = _runs_snapshot()
        rep("runs/ is untouched", after == before,
            (f"nothing written ({len(before)} entries checked; the garage self-check's own "
             f"runs/garage_*.png not counted)") if after == before
            else f"changed: {sorted(set(after) ^ set(before))[:4]}")

        rep("wall time", res.wall <= 2.0 * WALL_TARGET_S,
            f"{res.wall:.1f} s for {len(res.shots)} files in {len(res.walks)} walks "
            f"(PLAN2 target <= {WALL_TARGET_S:.0f} s for everything)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if verbose:
        print(f"design_shots self-check: {rows} rows, {'ALL PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------- #
def _size(text: str) -> tuple:
    try:
        W, H = (int(v) for v in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r}: a size is WxH, e.g. 1600x1000")
    if W < 1280 or H < 720:
        raise argparse.ArgumentTypeError(f"{text}: the shell's minimum is 1280x720")
    return W, H


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m drive.design_shots",
                                 description="Screenshots of every design view (PLAN D7), every "
                                             "run replayed from WingLab's captured fixtures. "
                                             "Without --out: the self-check (of --only's "
                                             "views, when given).")
    ap.add_argument("--out", help="render into DIR (created; the views drawn are replaced)")
    ap.add_argument("--size", type=_size, action="append",
                    help="WxH, repeatable (default 1280x800 and 1600x1000)")
    ap.add_argument("--only", nargs="+", metavar="KEYS",
                    help="views to draw: keys (af.opt), stages (af) or patterns (r.*)")
    ap.add_argument("--lib", help="fly on a temp COPY of this library (default: a fresh one)")
    ap.add_argument("--no-scroll", action="store_true", help="scroll 0 only")
    a = ap.parse_args(argv)
    try:
        select_keys(a.only)
    except ValueError as exc:
        ap.error(str(exc))
    if a.lib and not os.path.isdir(a.lib):
        ap.error(f"--lib {a.lib}: not a directory")
    sizes = tuple(a.size or SIZES)
    if not a.out:
        return 0 if self_check(sizes=sizes, only=a.only, lib=a.lib, scroll=not a.no_scroll) else 1
    res = render(a.out, sizes=sizes, only=a.only, lib=a.lib, scroll=not a.no_scroll)
    for line in res.failures + res.errors + res.unfrozen:
        print(f"  [FAIL] {line}")
    bad = bool(res.failures or res.errors or res.unfrozen)
    print(f"design_shots: {len(res.shots)} files in {res.out} (manifest.json), "
          f"{res.wall:.1f} s, worst frame {res.worst_ms:.0f} ms -- "
          f"{'FAIL' if bad else 'ALL PASS'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
