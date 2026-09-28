"""drive/prerace.py -- the pre-race screen, and a build per track (task 20).

A timed session on a lap map starts HERE, not on the tarmac: the class you
are about to be timed in, the build you are about to drive it in, the class's
top 5 and the medal targets, and one press to go.

    RACE   ENTER / CROSS (or a click): every car to the line, the timer from
           zero. An unchanged build starts in ONE press -- the cursor opens
           on RACE.
    EDIT   the garage, on this build; ENTER there drives it and comes back
           to this screen.
    WINGS  (a build with no wings; round 3 of task 45) 'Try ready-made
           wings': the garage with the garage's W already pressed on the
           first empty slot -- a ready-made wing on the car in one key --
           and, like EDIT, its ENTER drives it and comes back here.
    PICK   any build saved in the garage library, whichever track it was
           designed on, each row with that build's best time in THIS class.

The build a map opens with is the one last RACED on that map
(`runs/records/last_builds.json`, `RecordBook.last_build`): the arena keeps
its low-drag car and the skidpad its big wing without the garage in between.
Coming back from the garage keeps the car just built; `--build` / `--wing`
on the command line win at launch.

Task 41: builds know the car they were made for (`CarBuild.car`, "" for a
build saved before -- an any-car build). The per-map memory is kept per map
AND car, and never hands a car another car's build; the PICK page lists the
driven car's builds first, then the any-car ones, then other cars' (tagged,
still pickable: a van's wing on a Corsa is the player's own experiment), with
the car's own DEFAULT build (Settings.car_build) marked. The same page is
the Settings page's Build row, on every map (`key` None: no times there).

The same page is the pause menu's *Time trial* item mid-session.

When a session OPENS on it (task 33, `session_start`): when its class or
its build (name and content) is not the previous session's -- the first
session of a launch, a map / car / engine / surface change (through a map
with no page too), a PICK -- and back from the garage (EDIT's round trip
ends here). A restart that keeps both -- a ballast change, a tutorial or a
challenge ending on the same class, a race -- drives straight on: the page
would show what was just on it, one more press for nothing.

Pure UI logic: this module builds `drive.menu.Menu` rows and help sections
from a `records.RecordBook`, the medal table (`drive.medals`, lazily -- a
class with no reference lap shows "--") and the library's builds; the `Sim`
owns the menu and dispatches the actions (`drive.drive.Sim._prerace_event`).
It imports no pygame. Scripted and headless runs never see the screen
(`wanted`).

An UNLIMITED session (task 41: a build with a wing past its car's physical
span limit, drive/bodies.py) says so on every part of the page: the build
row and the subtitle carry the tag, an UNLIMITED section names what is past
its limit and the official PB it is NOT competing with, and the top 5 and
the medals are headed as the Unlimited book's -- which `book` then is
(records.unlimited_book). A build's best on the pick page is read from the
book its OWN build files into (`judge`), so an official build shows its
official best in an Unlimited session and the other way round.

Actions the rows return: 'pr_race', 'pr_edit', 'pr_wings', 'pr_pick', 'pr_back',
'pr_build:<name>' (the PICK page), 'set:pr_ghost' (task 22's ghost slot) and
'set:pr_ghosts' (both ghosts shown / hidden: J's toggle, for a pad; task 33).

    python3 -m drive.prerace      the self-check
"""

from __future__ import annotations

import math

from . import records as rec

#: the pick page's row for the build being driven when it is not a saved one
CURRENT = "__current__"
#: a top-5 row's build name is cut to this many letters: with the assists in
#: words ('ABS · TC · steer aid · manual + clutch') and the date, the longest
#: row still fits one line of the TIME TRIAL page's help column
TOP_NAME = 18

PR_HELP = [("PRE-RACE", [
    ("RACE", "every car to the line; the clock starts at the next crossing"),
    ("Build", "pick another saved build (each with its best in this class)"),
    ("Edit", "the garage on this build; its ENTER comes back here"),
    ("Ghosts", "shown / hidden: J on the keyboard, this row on a pad"),
    ("Ghost 2", "LEFT / RIGHT: the reference bot, none, or your P2..P5"),
    ("Wings", "LEFT / RIGHT: the wing mode -- each one its own leaderboard; FREE "
              "(your build as designed) never counts"),
    ("ESC", "the pause menu: Resume drives on from here (laps still count)"),
])]
PICK_HELP = [("PICK A BUILD", [
    ("ENTER / CROSS", "drive that build (a new session on it)"),
    ("time", "its best lap in this class, if any"),
    # whole sentences: the menu wraps a row to its column (a line broken by
    # hand left 'then' and 'try)' alone on a line of their own)
    ("order", "this car's builds, then any-car ones, then other cars' "
              "(tagged [car]: yours to try)"),
    ("(default)", "the build this car opens with; Settings > Default, or the "
                  "garage's F, sets it"),
    ("saving", "the garage's S (pad: OPTIONS)"),
    ("ESC", "back where you came from"),
])]

#: round 3 of task 45 (the owner: wings sooner, no default wings): the row a
#: build with NO wings gets, right under Edit, and its line in the help --
#: the garage opened with its W pressed on the first empty slot
TRY_WINGS = "Try ready-made wings"
TRY_WINGS_HELP = ("Try wings", "the garage's W: a ready-made wing on your car in one key")


def no_wings(build_json) -> bool:
    """Is `build_json` (a CarBuild json) a car with no wing in any slot? The
    published one-panel car (None) and anything unreadable are not: the row
    that offers wings is only for a car that plainly has none."""
    if not isinstance(build_json, dict):
        return False
    slots = build_json.get("slots")
    if not isinstance(slots, dict):
        return False
    return not any(isinstance(v, dict) and v.get("wing") for v in slots.values())


#: short names for a car's tag on a pick row / a library row. The titles
#: live in `cars.CAR_TITLES`; these are the few letters a row has room for.
#: A key missing here (a car added later) falls back to its title's first
#: two words, then to the key itself. (Task 46: the MX-5 and the Citaro are
#: retired; a build tagged with one reads as a Corsa build, so neither tag
#: reaches a row.)
CAR_SHORT = {"corsa": "Civetta", "rally": "Halcón", "540i": "N540",
             "express": "Courier"}


def car_label(car: str) -> str:
    """A car key as a row's tag: 'Civetta', 'Halcón', ... ("" for "")."""
    if not car:
        return ""
    if car in CAR_SHORT:
        return CAR_SHORT[car]
    try:
        import cars as _cars              # lazily: this module stays light
        t = _cars.CAR_TITLES.get(car)
    except Exception:                      # noqa: BLE001
        t = None
    return " ".join(str(t).split()[:2]) if t else str(car)


def pick_order(builds: dict, car: str) -> list:
    """The library's build names in the order a car lists them (task 41):
    `car`'s own builds, then the any-car ones saved before task 41, then
    every other car's -- each group by name. The garage's library page and
    its B key use the same order (`drive.garage`)."""
    def group(n):
        c = rec.build_car(builds[n])
        return 0 if c == car else (1 if c == "" else 2)
    return sorted(builds, key=lambda n: (group(n), n.lower(), n))


def wanted(opts, settings) -> bool:
    """Does a drive session start on the pre-race screen? Only a windowed,
    human-driven session on a map with a lap: never a script, a headless
    run, `--ml-drive`, the dragstrip or a non-standard skidpad."""
    if getattr(opts, "headless", False) or getattr(opts, "script", None):
        return False
    if getattr(opts, "ml_drive", None):
        return False
    if getattr(opts, "render", None) in ("off", "offscreen"):
        return False
    return rec.records_reason(settings.track, getattr(opts, "radius", 50.0),
                              getattr(opts, "cw", False)) is None


def seen_key(key, build_name, build_json) -> tuple:
    """What a session's page shows: the class and the build -- its name and
    its content (`records.build_id`)."""
    return (str(key or ""), str(build_name or ""), rec.build_id(build_name, build_json))


def due(prev, now, forced: bool = False) -> bool:
    """Does this session open on the page? When `now` (this session's
    `seen_key`) is not `prev` (the previous session's; None: none, or a map
    with no page), and when `forced` (back from the garage)."""
    return bool(forced) or prev != now


def session_start(opts, key, build_name, build_json, wanted_now: bool) -> bool:
    """At a drive session's start: does it open on the page? Remembers this
    session's `seen_key` for the next (`opts.prerace_seen`, None without a
    class) and spends the one-shot flags (`opts.prerace_force`, set by the
    garage's return; `opts.prerace_skip`, by the swarm's)."""
    now = seen_key(key, build_name, build_json) if key else None
    opens = bool(wanted_now and now is not None and not getattr(opts, "prerace_skip", False)
                 and due(getattr(opts, "prerace_seen", None), now,
                         forced=getattr(opts, "prerace_force", False)))
    opts.prerace_seen, opts.prerace_force, opts.prerace_skip = now, False, False
    return opens


def default_build(book, track: str, car: str | None = None):
    """(name, CarBuild json) of the build last raced on `track` -- with `car`,
    the one that car last raced there (`RecordBook.last_build`) -- or None."""
    e = book.last_build(track, car)
    if not e:
        return None
    return str(e.get("name", "")), e["build"]


def _medal_table(key: str):
    """(targets or None, best medal label) from drive.medals, lazily; a
    missing or stale table is simply 'no targets'."""
    try:
        from . import medals
        return medals.targets(key)
    except Exception:                      # noqa: BLE001 -- medals are optional
        return None


def _medal_for(key: str, t):
    try:
        from . import medals
        return medals.medal_for(key, t)
    except Exception:                      # noqa: BLE001
        return None


def next_medal(key: str, t, tg=None):
    """The medal a PB of `t` s goes for next: (name, target s, gap s) -- the
    easiest one not won yet (`t <= target` wins it, drive.medals' rule). No
    PB yet (`t` not finite): bronze, the gap None. None when the author time
    is beaten or the class has no targets. `tg`: the targets (default: the
    medal table's)."""
    tg = _medal_table(key) if tg is None else tg
    if not tg or not all(isinstance(tg.get(m), (int, float)) for m in rec.MEDAL_ORDER):
        return None
    if not isinstance(t, (int, float)) or not math.isfinite(t):
        return "bronze", float(tg["bronze"]), None
    for m in reversed(rec.MEDAL_ORDER):    # bronze first: the nearest one up
        if float(tg[m]) < t:
            return m, float(tg[m]), float(t) - float(tg[m])
    return None


#: the gearbox of a top-5 lap in words (the HUD keeps its short 'MAN+CL')
GEARBOX_WORDS = {"auto": "auto", "manual": "manual", "clutch": "manual + clutch"}


def assists_text(a: dict) -> str:
    """The lap's assists in words, as a top-5 row says them:
    'ABS · TC · steer aid · manual + clutch'. "" for no record of them."""
    if not isinstance(a, dict):
        return ""
    tags = [t for t, on in (("ABS", a.get("abs")), ("TC", a.get("tc")),
                            ("steer aid", a.get("steer_aid"))) if on]
    gb = GEARBOX_WORDS.get(a.get("gearbox"), "")
    return " · ".join(tags + ([gb] if gb else []))


class PreRace:
    """The screen's content for one class and one build.

    `builds` is the garage library's `{name: CarBuild json}`; `build_json`
    None means the published one-panel car (a legacy `WingDesign`).

    Task 41: `over` is this session's wings past the car's span limit
    (`bodies.over_limits`; non-empty = an UNLIMITED session, and `book` is
    then the Unlimited book); `books` = {'official': book, 'unlimited': book}
    and `judge(build_json) -> bool` (is that build Unlimited on this car)
    route each PICK row's best to its own book. Without them every best is
    read from `book`, as before.

    The review of task 41 (root design): a session drives a COPY of the
    player's build fitted to its car. `build_json` is that copy (what the
    laps are filed under); `design_json`, when given, is the build as the
    player holds it -- what the library and the per-map memory know, so it
    is what `saved()` looks up -- and `fit(build_json) -> json` fits a
    library build to this car the same way, so a PICK row's best is read
    under the id its laps were filed with. Without them: `build_json` for
    both, and no fitting (the stock cars' bands fit every pre-41 build as it
    is).

    `key` and `book` may be None -- the Settings page's Build row opens the
    PICK page on EVERY map, the dragstrip and a map with no records
    included, where there is no class and no time to show. `car` is the car
    being driven (its builds are listed first) and `default` the build the
    player made that car's default (marked); the session sets both before
    the pick page is drawn."""

    def __init__(self, key: str, book, build_name: str = "", build_json=None,
                 builds=None, titles=None, can_edit: bool = True,
                 over=None, books=None, judge=None,
                 car: str = "", default: str = "", design_json=None, fit=None):
        self.key = key
        self.can_edit = bool(can_edit)     # False: no garage this session (no EDIT row)
        self.book = book
        self.over = [dict(o) for o in (over or [])]
        self.books = dict(books or {})
        self.judge = judge
        self.car = str(car or "")
        self.default = str(default or "")
        self.build_name = str(build_name or "")
        self.build_json = build_json
        self.design_json = design_json if isinstance(design_json, dict) else build_json
        self.fit = fit
        self.builds = dict(builds or {})
        self.titles = dict(titles or {})
        self.ghost_label = None            # task 22: the ghost-2 row, when set
        self.ghosts_on = None              # task 33: the ghosts row (J), when set
        #: task 47 (drive/leaderboard.py): the wing mode's words -- the Wings
        #: row, a player session's -- and this session's LEADERBOARD section
        #: (`leaderboard.prerace_rows`), when set
        self.wings = None
        self.board_rows = None

    # -- what the build is ------------------------------------------------
    def saved(self) -> bool:
        """Is the build being driven one of the library's (same content)? The
        build as the player holds it (`design_json`), not its fitted copy."""
        b = self.builds.get(self.build_name)
        return b is not None and _same_build(b, self.design_json)

    def unsaved(self) -> bool:
        """Does the build being driven say '(not saved)'? Only when there is a
        library for it to be missing from: with no saved builds at all, every
        build would say it. The one rule for the TIME TRIAL page's Build row,
        the pause pages' subtitle and the Settings page's Build row (task
        45: the pause page said '(not saved)' where this page did not)."""
        return bool(self.builds) and not self.saved()

    def _fitted(self, build_json):
        """`build_json` fitted to this car (`fit`), itself without one."""
        if self.fit is None or not isinstance(build_json, dict):
            return build_json
        try:
            return self.fit(build_json)
        except Exception:                  # noqa: BLE001 -- a bad build is not a crash
            return build_json

    @property
    def unlimited(self) -> bool:
        """This session's build is past its car's span limit (task 41)."""
        return bool(self.over)

    def is_unlimited(self, build_json) -> bool:
        """Would `build_json` be an Unlimited build on this car? The session's
        own build is what `over` says; another is asked of `judge` (False
        without one: nothing to judge it with)."""
        if build_json is self.build_json or _same_build(build_json, self.build_json):
            return self.unlimited
        try:
            return bool(self.judge(build_json)) if self.judge is not None else False
        except Exception:                  # noqa: BLE001 -- a bad build is not a crash
            return False

    def build_best(self, name: str, build_json=None) -> float:
        """A build's best lap in THIS class (nan when it has none): by the
        car's content (`records.build_id`), so a saved build is not credited
        with a lap an edited car drove under its name. Read from the book
        that build files into (task 41: official or Unlimited); nan on a map
        with no records (task 41: the class-less picker)."""
        if self.book is None or not self.key:
            return float("nan")
        book = self.book
        if self.books:
            want = "unlimited" if self.is_unlimited(build_json) else "official"
            book = self.books.get(want, self.book)
        return book.build_best(self.key, name, self._fitted(build_json))

    # -- the main page ------------------------------------------------------
    def items(self) -> list:
        name = self.build_name or "(unnamed)"
        tag = "  (not saved)" if self.unsaved() else ""
        if self.unlimited:
            tag += "  UNLIMITED"
        rows = [("RACE", "pr_race"),
                (f"{'Build':<9s}{name}{tag}", "pr_pick")]
        if self.can_edit:
            rows.append(("Edit this build in the garage", "pr_edit"))
        if self.offers_wings():
            rows.append((TRY_WINGS, "pr_wings"))
        if self.wings is not None:         # task 47: LEFT / RIGHT, a new session
            rows.append((f"{'Wings':<9s}{self.wings}", "set:pr_mode"))
        if self.ghosts_on is not None:
            rows.append((f"{'Ghosts':<9s}{'shown' if self.ghosts_on else 'hidden'}  (J)",
                         "set:pr_ghosts"))
        if self.ghost_label is not None:
            rows.append((f"{'Ghost 2':<9s}{self.ghost_label}", "set:pr_ghost"))
        return rows

    def offers_wings(self) -> bool:
        """The 'Try ready-made wings' row (round 3): a garage this session,
        and a build being driven with no wings at all."""
        return self.can_edit and no_wings(self.build_json)

    def help(self) -> list:
        """The page's help column: PR_HELP, with the ready-made wings' line
        under Edit when the row is there."""
        if not self.offers_wings():
            return PR_HELP
        (title, rows), = PR_HELP
        i = [k for k, _ in rows].index("Edit") + 1
        return [(title, rows[:i] + [TRY_WINGS_HELP] + rows[i:])]

    def class_text(self) -> str:
        """The class in proper names -- map, car, engine, surface, as the
        session gave them (`titles`) -- or the key's own words when one is
        missing (`records.class_label`). Both page subtitles say it."""
        ttl = self.titles
        parts = [ttl.get(k) for k in ("track", "car", "engine", "surface")]
        return "  ·  ".join(map(str, parts)) if all(parts) else rec.class_label(self.key)

    def subtitle(self) -> str:
        unl = "   UNLIMITED: not official" if self.unlimited else ""
        return f"{self.class_text()}   build: {self.build_name or '(unnamed)'}{unl}"

    def sections(self) -> list:
        t, c, e, s = rec.split_key(self.key)
        ttl = self.titles
        secs = [("CLASS", [("map", ttl.get("track", t)), ("car", ttl.get("car", car_label(c))),
                           ("engine", ttl.get("engine", e)), ("surface", ttl.get("surface", s))])]
        unl = "UNLIMITED " if self.unlimited else ""
        if self.unlimited:
            #  task 41: what is past its limit, and what the run is NOT
            rows_u = [(str(o.get("slot", "")),
                       f"{o.get('wing', '')}  span {float(o.get('span', 0.0)):.2f} m "
                       f"> max {float(o.get('limit', 0.0)):.2f} m") for o in self.over]
            off = self.books.get("official")
            if off is not None:
                rows_u.append(("official PB", rec.fmt_time(off.pb_time(self.key))
                               + "  (not raced: this run is filed apart)"))
            rows_u.append(("", "laps and medals go to this class's Unlimited"))
            rows_u.append(("", "book: never official, never on a public board"))
            secs.append(("UNLIMITED", rows_u))
        if self.board_rows:                # task 47: this session's leaderboard
            secs.append(("LEADERBOARD", list(self.board_rows)))
        laps = self.book.laps(self.key)
        rows = []
        names = []
        for lp in laps:
            # a lap from another version, or edited by hand, must not take the
            # page down: every field is read defensively
            b = lp.get("build")
            names.append(str((b.get("name") if isinstance(b, dict) else "") or "-")[:TOP_NAME])
        # the names padded to the longest, so the assists and the dates line up
        nw = max((len(n) for n in names), default=0)
        for i, (lp, name) in enumerate(zip(laps, names)):
            rows.append((f"{i + 1}  {rec.fmt_time(lp.get('time'))}",
                         "  ".join(p for p in (f"{name:<{nw}s}", assists_text(lp.get("assists")),
                                               str(lp.get("date", "") or "")[:10]) if p)))
        if not rows:
            rows = [("--", "none yet: your first valid lap is the PB")]
        secs.append((f"{unl}TOP {rec.TOP_N}", rows))
        tg = _medal_table(self.key)
        if tg and tg.get("author"):
            best = self.book.load(self.key).get("best_medal")
            pb = self.book.pb_time(self.key)
            m_pb = _medal_for(self.key, pb)
            order = rec.MEDAL_ORDER
            if m_pb in order and (best not in order or order.index(m_pb) < order.index(best)):
                best = m_pb                # a PB set before the table: it still counts
            mrows = [(m, rec.fmt_time(tg[m]) + ("  *" if math.isfinite(pb) and pb <= tg[m] else ""))
                     for m in ("author", "gold", "silver", "bronze")]      # * = the PB has it
            mrows.append(("yours", (best or "none yet") + (f"  (PB {rec.fmt_time(pb)})"
                                                            if math.isfinite(pb) else "")))
            nx = next_medal(self.key, pb, tg)
            if nx is None:
                mrows.append(("next", "all medals won"))
            else:
                mrows.append(("next", f"{nx[0].upper()} {rec.fmt_time(nx[1])}  "
                                      + (f"({nx[2]:.3f} s to go)" if nx[2] is not None
                                         else "(your first valid lap)")))
            secs.append((f"{unl}MEDALS", mrows))
        else:
            secs.append((f"{unl}MEDALS", [("--", "no reference lap for this class")]))
        return secs + self.help()

    # -- the pick page ----------------------------------------------------------
    def pick_items(self) -> list:
        """The PICK page's rows: the car being driven when it is no saved
        build, then the library's builds in `pick_order` -- this car's, the
        any-car ones, then other cars' tagged `[car]` (task 41) -- each with
        its best time in this class; the car's default marked."""
        rows = []
        if not self.saved():
            rows.append((f"{self.build_name or '(unnamed)':<24s}"
                         f"{rec.fmt_time(self.build_best(self.build_name, self.build_json)):>10s}"
                         f"  (driving, not saved)",
                         "pr_back"))
        for name in pick_order(self.builds, self.car):
            mark = "  <- driving" if (name == self.build_name and self.saved()) else ""
            if name == self.default:
                mark += "  (default)"
            c = rec.build_car(self.builds[name])
            if self.car and c and c != self.car:
                mark += f"  [{car_label(c)}]"
            if self.is_unlimited(self.builds[name]):
                mark += "  UNLIMITED"        # past this car's span limit (task 41)
            rows.append((f"{name[:24]:<24s}"
                         f"{rec.fmt_time(self.build_best(name, self.builds[name])):>10s}{mark}",
                         f"pr_build:{name}"))
        if not self.builds:
            rows.append(("no saved builds yet (the garage's S saves one)", "pr_back"))
        rows.append(("Back", "pr_back"))
        return rows

    def pick_subtitle(self) -> str:
        if not self.key:
            return (f"{car_label(self.car) or 'this car'}'s builds first   "
                    f"no lap times on this map")
        return f"{self.class_text()}   the time is each build's best in this class"


def _same_build(a, b) -> bool:
    """Two CarBuild jsons describe the same car: its labels
    (`records.BUILD_META` -- the name, the library's `builtin` flag, the car
    it was made for) do not change the car."""
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    strip = lambda d: {k: v for k, v in d.items() if k not in rec.BUILD_META}   # noqa: E731
    return strip(a) == strip(b)


# ==================================================================== #
#  SELF-CHECK                                                          #
# ==================================================================== #
def self_check(verbose: bool = True) -> bool:
    import tempfile
    from types import SimpleNamespace
    ok = True
    n_ok = n_all = 0

    def rep(tag, passed, msg=""):
        nonlocal ok, n_ok, n_all
        n_all += 1
        n_ok += bool(passed)
        ok = ok and bool(passed)
        if verbose:
            print(f"  [{'ok' if passed else 'FAIL'}] {tag}: {msg}")

    if verbose:
        print("drive/prerace.py self-check")
    s = SimpleNamespace(track="arena", car="corsa", engine="sport", wet="patch")
    o = SimpleNamespace(headless=False, script=None, ml_drive=None, render=None,
                        radius=50.0, cw=False)
    rep("a windowed session on a lap map opens on it", wanted(o, s))
    cases = dict(headless=SimpleNamespace(**dict(vars(o), headless=True)),
                 script=SimpleNamespace(**dict(vars(o), script="lap")),
                 ml=SimpleNamespace(**dict(vars(o), ml_drive="x.json")),
                 offscreen=SimpleNamespace(**dict(vars(o), render="offscreen")))
    rep("scripted, headless, --ml-drive and offscreen runs skip it",
        not any(wanted(v, s) for v in cases.values()), ", ".join(cases))
    rep("the dragstrip and a 30 m skidpad skip it",
        not wanted(o, SimpleNamespace(**dict(vars(s), track="dragstrip")))
        and not wanted(SimpleNamespace(**dict(vars(o), radius=30.0)),
                       SimpleNamespace(**dict(vars(s), track="skidpad"))))

    root = tempfile.mkdtemp(prefix="carsim_prerace_")
    book = rec.RecordBook(root)
    key = rec.class_key("arena", "corsa", "sport", "patch")
    b_fast = dict(version=2, name="fast", mirror=True, builtin=False, slots={"left": {"wing": "fin"}})
    b_wet = dict(version=2, name="wet", mirror=True, builtin=False, slots={"left": {"wing": "plate"}})
    js = {"fast": b_fast, "wet": b_wet, "gone": None}
    for t, name in ((61.2, "fast"), (62.5, "wet"), (60.9, "fast"), (64.0, "gone")):
        r = rec._fake_rec(t, [20.0, 20.5, 20.4])
        r["build"] = dict(name=name, json=js[name])
        r["assists"] = dict(abs=True, tc=False, steer_aid=True, gearbox="manual")
        book.insert(key, r)
    pr = PreRace(key, book, "fast", dict(b_fast), builds={"fast": b_fast, "wet": b_wet},
                 titles=dict(track="Arena circuit"))
    it = pr.items()
    rep("RACE is the first row (one press)", it[0] == ("RACE", "pr_race") and
        [a for _, a in it] == ["pr_race", "pr_pick", "pr_edit"], str([a for _, a in it]))
    secs = dict((t, r) for t, r in pr.sections())
    top = secs.get(f"TOP {rec.TOP_N}", [])
    rep("top 5 rows, fastest first, with build and assists",
        len(top) == 4 and top[0][0].startswith("1  1:00.900") and "fast" in top[0][1]
        and "ABS · steer aid · manual" in top[0][1], str(top[:2]))
    rep("the assists in words, not the HUD's tags",
        assists_text({"abs": True, "steer_aid": True, "gearbox": "manual"})
        == "ABS · steer aid · manual"
        and assists_text({"abs": True, "tc": True, "steer_aid": True, "gearbox": "clutch"})
        == "ABS · TC · steer aid · manual + clutch"
        and assists_text({"gearbox": "auto"}) == "auto" and assists_text("x") == "",
        assists_text({"abs": True, "steer_aid": True, "gearbox": "manual"}))
    rep("... the names padded, so the assists line up down the top 5",
        len({r[1].index("ABS") for r in top}) == 1, str([r[1] for r in top]))
    lb = rec.RecordBook(tempfile.mkdtemp(prefix="carsim_prerace_"))
    r = rec._fake_rec(59.0, [])
    r.update(build=dict(name="a very long build name indeed", json=b_fast),
             assists=dict(abs=True, tc=True, steer_aid=True, gearbox="clutch"),
             date="2026-09-26T10:00:00")
    lb.insert(key, r)
    long_row = dict(PreRace(key, lb, "x", None).sections())[f"TOP {rec.TOP_N}"][0][1]
    #  the TIME TRIAL page's help column holds about 80 letters at any size
    rep("the longest top-5 row (a cut name, every assist, a date) fits one line",
        long_row.startswith("a very long build ") and len(long_row) <= 72
        and long_row.endswith("manual + clutch  2026-09-26"), f"{len(long_row)}: {long_row}")
    rep("medal block present (targets or '--')", "MEDALS" in secs, str(secs.get("MEDALS"))[:80])
    med = secs.get("MEDALS") or [("--", "")]
    rep("... with the next medal and its gap (or all won)", med[0][0] == "--"
        or (med[-1][0] == "next" and ("s to go)" in med[-1][1] or "all medals" in med[-1][1])),
        str(med[-2:]))
    tg = dict(author=58.0, gold=59.0, silver=61.0, bronze=65.0)
    rep("next medal: the nearest one up with the gap; bronze with no PB; none past the author",
        next_medal(key, 60.5, tg) == ("gold", 59.0, 1.5) and next_medal(key, 70.0, tg)[0] == "bronze"
        and next_medal(key, 59.0, tg)[0] == "author" and next_medal(key, 58.0, tg) is None
        and next_medal(key, float("inf"), tg) == ("bronze", 65.0, None)
        and next_medal(key, 60.0, {}) is None, str(next_medal(key, 60.5, tg)))
    named = PreRace(key, book, "fast", b_fast, titles=dict(
        track="Arena circuit", car="Linden Corsa", engine="Sport", surface="Dry, wet patches"))
    rep("the subtitle: the proper names; the key's words without them",
        named.subtitle().startswith("Arena circuit  ·  Linden Corsa  ·  Sport")
        and pr.subtitle().startswith(rec.class_label(key)), named.subtitle())
    rep("... and the PICK page's the same way",
        named.pick_subtitle().startswith("Arena circuit  ·  Linden Corsa  ·  Sport  ·  "
                                         "Dry, wet patches   ")
        and pr.pick_subtitle().startswith(rec.class_label(key) + "   "),
        named.pick_subtitle())
    pick = pr.pick_items()
    acts = [a for _, a in pick]
    rep("pick lists every saved build with its best in this class",
        acts == ["pr_build:fast", "pr_build:wet", "pr_back"] and "1:00.900" in pick[0][0]
        and "1:02.500" in pick[1][0] and "<- driving" in pick[0][0], str([p[0] for p in pick]))
    pr2 = PreRace(key, book, "my corsa", dict(b_fast, name="my corsa", slots={"left": {"wing": "x"}}),
                  builds={"fast": b_fast})
    rep("an unsaved build is listed first and marked", pr2.pick_items()[0][1] == "pr_back"
        and "not saved" in pr2.pick_items()[0][0] and "(not saved)" in pr2.items()[1][0])
    pr3 = PreRace(key, book, "my corsa", dict(b_fast, name="my corsa"), builds={})
    rep("'(not saved)' only with a library to miss from: one rule (unsaved) for "
        "this page and the pause pages (task 45)",
        pr2.unsaved() and not pr3.unsaved() and not pr.unsaved() and not pr3.saved()
        and "(not saved)" not in pr3.items()[1][0],
        f"{pr2.unsaved()} / {pr3.unsaved()} / {pr.unsaved()}; {pr3.items()[1][0]!r}")
    rep("a build that equals a saved one under another name is not 'saved'",
        not PreRace(key, book, "fast", dict(b_wet, name="fast"), builds={"fast": b_fast}).saved())
    book.set_last_build("arena", "wet", b_wet)
    rep("the default build of a map is the last one raced there",
        default_build(rec.RecordBook(root), "arena") == ("wet", b_wet)
        and default_build(rec.RecordBook(root), "open") is None)
    edited = dict(b_fast, slots={"left": {"wing": "plate"}, "top": {"wing": "rear"}})
    r = rec._fake_rec(58.4, [19.0, 19.7, 19.7])
    r["build"] = dict(name="fast", json=edited)          # an edit that kept the name
    book.insert(key, r)
    pk = PreRace(key, book, "fast", edited, builds={"fast": b_fast}).pick_items()
    rep("a saved build is not credited with a lap an edited car drove under its name",
        "58.400" in pk[0][0] and "not saved" in pk[0][0] and "1:00.900" in pk[1][0],
        str([p[0] for p in pk[:2]]))
    bad = rec.RecordBook(tempfile.mkdtemp(prefix="carsim_prerace_"))
    for b_ in ("fast", 7, None, {"name": 7}):
        r = rec._fake_rec(61.0 + 0.1 * len(str(b_)), [])
        r["build"], r["assists"], r["date"] = b_, "x", None
        bad.insert(key, r)
    try:
        ok_bad = len(dict(PreRace(key, bad, "x", None).sections())[f"TOP {rec.TOP_N}"]) == 4
    except Exception as exc:               # noqa: BLE001
        ok_bad = False
    rep("laps with odd field types do not take the page down", ok_bad)
    rep("no garage this session: no EDIT row",
        [a for _, a in PreRace(key, book, "fast", b_fast, can_edit=False).items()]
        == ["pr_race", "pr_pick"])
    #  round 3 (the owner: wings sooner): a car with no wings gets 'Try
    #  ready-made wings' under Edit, and its line in the help under Edit's
    b_bare = dict(version=2, name="my corsa", mirror=True, builtin=False, car="corsa",
                  slots={"left": {"wing": ""}, "right": {"wing": ""}, "top": {"wing": ""}})
    p_bare = PreRace(key, book, "my corsa", b_bare, builds={"fast": b_fast})
    help_b = [k for k, _ in dict(p_bare.sections())["PRE-RACE"]]
    rep("a build with no wings: 'Try ready-made wings' under Edit, its help under Edit's",
        p_bare.items()[:4] == [("RACE", "pr_race"), p_bare.items()[1],
                               ("Edit this build in the garage", "pr_edit"),
                               ("Try ready-made wings", "pr_wings")]
        and help_b[help_b.index("Edit") + 1] == "Try wings"
        and dict(p_bare.sections())["PRE-RACE"][help_b.index("Try wings")][1]
        == "the garage's W: a ready-made wing on your car in one key", str(help_b))
    rep("... never for a car with a wing, the published panel (None), or with no garage",
        "pr_wings" not in [a for _, a in pr.items()]
        and "Try wings" not in [k for k, _ in dict(pr.sections())["PRE-RACE"]]
        and "pr_wings" not in [a for _, a in PreRace(key, book, "x", None).items()]
        and "pr_wings" not in [a for _, a in PreRace(key, book, "my corsa", b_bare,
                                                     can_edit=False).items()]
        and not no_wings(dict(b_bare, slots=dict(b_bare["slots"], top={"wing": "rear-s1223"})))
        and no_wings(dict(b_bare, slots={})) and not no_wings({"slots": 3})
        and PR_HELP[0][1][2][0] == "Edit", str([a for _, a in pr.items()]))
    pr.ghost_label = "reference bot"
    rep("the ghost row appears when set", pr.items()[-1] == ("Ghost 2  reference bot", "set:pr_ghost"))
    #  task 47: the Wings row (before the ghosts) and the LEADERBOARD section
    pw = PreRace(key, book, "fast", b_fast)
    pw.wings, pw.board_rows = "ONLY TOP", [("board", "Arena  ·  Corsa  ·  ONLY TOP")]
    pw.ghost_label = "reference bot"
    acts_w = [a for _, a in pw.items()]
    secs_w = [t for t, _ in pw.sections()]
    rep("the Wings row (task 47) sits before the ghosts; the LEADERBOARD section after "
        "the CLASS",
        acts_w[-2:] == ["set:pr_mode", "set:pr_ghost"] and ("Wings    ONLY TOP", "set:pr_mode")
        in pw.items() and secs_w[:2] == ["CLASS", "LEADERBOARD"]
        and "set:pr_mode" not in [a for _, a in pr.items()]
        and "LEADERBOARD" not in [t for t, _ in pr.sections()]
        and any(k == "Wings" for k, _ in PR_HELP[0][1]), f"{acts_w} {secs_w}")
    pr.ghosts_on = False
    rep("the ghosts row (J, for a pad) before it, when set",
        pr.items()[-2:] == [("Ghosts   hidden  (J)", "set:pr_ghosts"),
                            ("Ghost 2  reference bot", "set:pr_ghost")], str(pr.items()[-2:]))
    #  task 41: an UNLIMITED session says so on every part of the page, reads
    #  its top 5 and medals from the Unlimited book, and routes each PICK
    #  row's best to the book that build files into
    ubook = rec.unlimited_book(root)
    b_huge = dict(version=2, name="huge", mirror=True, builtin=False,
                  slots={"top": {"wing": "huge-top"}})
    ru = rec._fake_rec(57.0, [19.0, 19.0, 19.0])
    ru.update(build=dict(name="huge", json=b_huge), unlimited=True)
    ubook.insert(key, ru)
    over = [dict(slot="top", wing="huge-top", span=2.6, limit=1.9752)]
    books = dict(official=book, unlimited=ubook)
    judge = lambda js_: bool(js_) and js_.get("name") == "huge"      # noqa: E731
    pu = PreRace(key, ubook, "huge", b_huge, builds={"fast": b_fast, "huge": b_huge},
                 over=over, books=books, judge=judge)
    su = dict(pu.sections())
    rep("an Unlimited session: the build row, the subtitle and the sections say UNLIMITED",
        pu.unlimited and "UNLIMITED" in pu.items()[1][0] and "UNLIMITED" in pu.subtitle()
        and "UNLIMITED" in su and f"UNLIMITED TOP {rec.TOP_N}" in su
        and "UNLIMITED MEDALS" in su and f"TOP {rec.TOP_N}" not in su
        and su[f"UNLIMITED TOP {rec.TOP_N}"][0][0].startswith("1  57.000")
        and any("58.400" in v for _k, v in su["UNLIMITED"]),       # the official PB
        str(su["UNLIMITED"][:2]))
    pk_u = {a: t for t, a in pu.pick_items()}
    rep("...and a PICK row's best is its OWN book's: official for an official build",
        "57.000" in pk_u["pr_build:huge"] and "1:00.900" in pk_u["pr_build:fast"]
        and not pr.unlimited and "UNLIMITED" not in pr.subtitle()
        and "UNLIMITED" not in dict(pr.sections()),
        f"{pk_u['pr_build:huge'].strip()} / {pk_u['pr_build:fast'].strip()}")
    from types import SimpleNamespace
    o = SimpleNamespace()
    stock = key.replace("sport", "stock")
    seq = [session_start(o, *a) for a in (
        (key, "fast", b_fast, True),            # the launch's first session: opens
        (key, "fast", dict(b_fast), True),      # a ballast restart, same class + build
        (None, "", None, False),                # TAB to the dragstrip: no page there
        (key, "fast", b_fast, True),            # ... and back: a new map again, opens
        (stock, "fast", b_fast, True),          # an engine change: opens
        (stock, "copy", dict(b_fast), True),    # PICK the same car under another name
        (stock, "copy", edited, True))]         # the same name, edited in the garage
    o.prerace_force = True                      # back from the garage, nothing changed
    seq.append(session_start(o, stock, "copy", edited, True))
    o.prerace_skip = True                       # back from the swarm, the build changed
    seq.append(session_start(o, key, "fast", b_fast, True))
    seq.append(session_start(o, key, "fast", b_fast, False))   # a headless run: never
    rep("the page opens when the class or build is not the last session's, or from "
        "the garage; flags spent", seq == [True, False, False, True, True, True, True, True,
                                          False, False]
        and not o.prerace_force and not o.prerace_skip, str(seq))
    # -- task 41: builds know their car ------------------------------------
    b_c = dict(b_fast, name="c fast", car="corsa")
    b_bus = dict(b_wet, name="big rally", car="rally")     # task 46: the rally car
    b_any = dict(b_fast, name="any", slots={"left": {"wing": "x"}})       # before task 41
    lib41 = {"c fast": b_c, "big rally": b_bus, "any": b_any, "b slow": dict(b_c, name="b slow")}
    p41 = PreRace(key, book, "c fast", dict(b_c), builds=lib41, car="corsa", default="b slow")
    rows41 = [lbl for lbl, _ in p41.pick_items()]
    acts41 = [a for _, a in p41.pick_items()]
    rep("the pick lists this car's builds, then any-car ones, then other cars' "
        "(tagged); the default marked",
        acts41 == ["pr_build:b slow", "pr_build:c fast", "pr_build:any", "pr_build:big rally",
                   "pr_back"]
        and "(default)" in rows41[0] and "<- driving" in rows41[1]
        and "[Halcón]" in rows41[3] and "[" not in rows41[2] and "[" not in rows41[0],
        str(rows41))
    rep("pick_order: the garage's order too",
        pick_order(lib41, "rally") == ["big rally", "any", "b slow", "c fast"]
        and pick_order(lib41, "") == ["any", "b slow", "big rally", "c fast"],
        str(pick_order(lib41, "rally")))
    #  task 46: a build made for a RETIRED car ('bus', 'mx5') is a Corsa
    #  build now -- listed with the Corsa's own, untagged, never another
    #  car's that no one can drive
    lib46 = dict(lib41, **{"old bus": dict(b_wet, name="old bus", car="bus"),
                           "old mx5": dict(b_fast, name="old mx5", car="mx5")})
    rows46 = {a: t for t, a in PreRace(key, book, "c fast", dict(b_c), builds=lib46,
                                         car="corsa").pick_items()}
    rep("a retired car's build is listed as a Corsa build (untagged)",
        pick_order(lib46, "corsa")[:4] == ["b slow", "c fast", "old bus", "old mx5"]
        and "[" not in rows46["pr_build:old bus"] and "[" not in rows46["pr_build:old mx5"]
        and pick_order(lib46, "rally")[0] == "big rally",
        str(pick_order(lib46, "corsa")))
    rep("a build tagged with its car is the SAME car as its untagged self "
        "(saved, and one build_id)",
        _same_build(b_fast, dict(b_fast, car="rally"))
        and PreRace(key, book, "fast", dict(b_fast, car="corsa"), builds={"fast": b_fast}).saved()
        and rec.build_id("fast", b_fast) == rec.build_id("fast", dict(b_fast, car="rally"))
        and not _same_build(b_fast, b_wet))
    nokey = PreRace(None, None, "c fast", dict(b_c), builds=lib41, car="corsa")
    rows_nk = nokey.pick_items()
    rep("with no class (the dragstrip, from Settings): the same list, no times, no crash",
        [a for _, a in rows_nk] == [a for _, a in PreRace(key, book, "c fast", dict(b_c),
                                                          builds=lib41, car="corsa").pick_items()]
        and "--" in rows_nk[0][0] and "no lap times" in nokey.pick_subtitle(),
        nokey.pick_subtitle())
    #  review of task 41 (root design): a session drives a copy FITTED to
    #  its car. The page reads the build the player holds as saved, and a
    #  PICK row's best under the fitted id its laps were filed with
    fit41 = lambda js_: dict(js_, slots=dict(js_.get("slots", {}),     # noqa: E731
                                             left={"wing": "x", "h": 1.2}))
    f_fast = fit41(b_fast)
    rf = rec._fake_rec(61.5, [20.0, 20.5, 21.0])
    rf.update(build=dict(name="fast", json=f_fast))
    book.insert(stock, rf)
    pf = PreRace(stock, book, "fast", f_fast, builds={"fast": b_fast}, design_json=b_fast,
                 fit=fit41)
    rows_f = {a: t for t, a in pf.pick_items()}
    rep("on a fitted copy: the held build reads as saved, a row's best is its fitted id's",
        pf.saved() and "1:01.500" in rows_f["pr_build:fast"]
        and "<- driving" in rows_f["pr_build:fast"] and len(rows_f) == 2
        and not PreRace(stock, book, "fast", f_fast, builds={"fast": b_fast}).saved(),
        rows_f["pr_build:fast"].strip())
    book.set_last_build("linden", "big rally", b_bus, car="rally")
    rep("the default build of a map is per car",
        default_build(rec.RecordBook(root), "linden", "rally") == ("big rally", b_bus)
        and default_build(rec.RecordBook(root), "linden", "corsa") is None)
    rep("car tags: short names, a later car by its key",
        car_label("rally") == "Halcón" and car_label("") == "" and car_label("zz9") == "zz9")
    if verbose:
        print(f"  {'ALL PASS' if ok else 'FAILURES ABOVE'}: {n_ok}/{n_all} checks")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_check() else 1)
