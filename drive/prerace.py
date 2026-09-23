"""drive/prerace.py -- the pre-race screen, and a build per track (task 20).

A timed session on a lap map starts HERE, not on the tarmac: the class you
are about to be timed in, the build you are about to drive it in, the class's
top 5 and the medal targets, and one press to go.

    RACE   ENTER / CROSS (or a click): every car to the line, the timer from
           zero. An unchanged build starts in ONE press -- the cursor opens
           on RACE.
    EDIT   the garage, on this build; ENTER there drives it and comes back
           to this screen (every drive session on a lap map opens here).
    PICK   any build saved in the garage library, whichever track it was
           designed on, each row with that build's best time in THIS class.

The build a map opens with is the one last RACED on that map
(`runs/records/last_builds.json`, `RecordBook.last_build`): the arena keeps
its low-drag car and the skidpad its big wing without the garage in between.
Coming back from the garage keeps the car just built; `--build` / `--wing`
on the command line win at launch.

The same page is the pause menu's *Time trial* item mid-session.

Pure UI logic: this module builds `drive.menu.Menu` rows and help sections
from a `records.RecordBook`, the medal table (`drive.medals`, lazily -- a
class with no reference lap shows "--") and the library's builds; the `Sim`
owns the menu and dispatches the actions (`drive.drive.Sim._prerace_event`).
It imports no pygame. Scripted and headless runs never see the screen
(`wanted`).

Actions the rows return: 'pr_race', 'pr_edit', 'pr_pick', 'pr_back',
'pr_build:<name>' (the PICK page) and 'set:pr_ghost' (task 22's ghost slot).

    python3 -m drive.prerace      the self-check
"""

from __future__ import annotations

import math

from . import records as rec

#: the pick page's row for the build being driven when it is not a saved one
CURRENT = "__current__"

PR_HELP = [("PRE-RACE", [
    ("RACE", "every car to the line; the clock starts at the next crossing"),
    ("Build", "pick another saved build (each with its best in this class)"),
    ("Edit", "the garage on this build; its ENTER comes back here"),
    ("ESC", "the pause menu: Resume drives on from here (laps still count)"),
])]
PICK_HELP = [("PICK A BUILD", [
    ("ENTER / CROSS", "drive that build (a new session on it)"),
    ("time", "its best lap in this class, if any"),
    ("", "saved builds: the garage's LIBRARY page, S"),
    ("ESC", "back to the pre-race screen"),
])]


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


def default_build(book, track: str):
    """(name, CarBuild json) of the build last raced on `track`, or None."""
    e = book.last_build(track)
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


def assists_text(a: dict) -> str:
    """The lap's assists as short tags: 'ABS TC AID AUTO'."""
    if not isinstance(a, dict):
        return ""
    tags = [t for t, on in (("ABS", a.get("abs")), ("TC", a.get("tc")),
                            ("AID", a.get("steer_aid"))) if on]
    gb = {"auto": "AUTO", "manual": "MAN", "clutch": "MAN+CL"}.get(a.get("gearbox"), "")
    return " ".join(tags + ([gb] if gb else []))


class PreRace:
    """The screen's content for one class and one build.

    `builds` is the garage library's `{name: CarBuild json}`; `build_json`
    None means the published one-panel car (a legacy `WingDesign`)."""

    def __init__(self, key: str, book, build_name: str = "", build_json=None,
                 builds=None, titles=None, can_edit: bool = True):
        self.key = key
        self.can_edit = bool(can_edit)     # False: no garage this session (no EDIT row)
        self.book = book
        self.build_name = str(build_name or "")
        self.build_json = build_json
        self.builds = dict(builds or {})
        self.titles = dict(titles or {})
        self.ghost_label = None            # task 22: the ghost-2 row, when set

    # -- what the build is ------------------------------------------------
    def saved(self) -> bool:
        """Is the build being driven one of the library's (same content)?"""
        b = self.builds.get(self.build_name)
        return b is not None and _same_build(b, self.build_json)

    def build_best(self, name: str, build_json=None) -> float:
        """A build's best lap in THIS class (nan when it has none): by the
        car's content (`records.build_id`), so a saved build is not credited
        with a lap an edited car drove under its name."""
        return self.book.build_best(self.key, name, build_json)

    # -- the main page ------------------------------------------------------
    def items(self) -> list:
        name = self.build_name or "(unnamed)"
        tag = "" if self.saved() or not self.builds else "  (not saved)"
        rows = [("RACE", "pr_race"),
                (f"{'Build':<9s}{name}{tag}", "pr_pick")]
        if self.can_edit:
            rows.append(("Edit this build in the garage", "pr_edit"))
        if self.ghost_label is not None:
            rows.append((f"{'Ghost 2':<9s}{self.ghost_label}", "set:pr_ghost"))
        return rows

    def subtitle(self) -> str:
        return f"{rec.class_label(self.key)}   build: {self.build_name or '(unnamed)'}"

    def sections(self) -> list:
        t, c, e, s = rec.split_key(self.key)
        ttl = self.titles
        secs = [("CLASS", [("map", ttl.get("track", t)), ("car", ttl.get("car", c)),
                           ("engine", ttl.get("engine", e)), ("surface", ttl.get("surface", s))])]
        laps = self.book.laps(self.key)
        rows = []
        for i, lp in enumerate(laps):
            # a lap from another version, or edited by hand, must not take the
            # page down: every field is read defensively
            b = lp.get("build")
            name = str((b.get("name") if isinstance(b, dict) else "") or "-")
            rows.append((f"{i + 1}  {rec.fmt_time(lp.get('time'))}",
                         f"{name[:18]}  {assists_text(lp.get('assists'))}  "
                         f"{str(lp.get('date', '') or '')[:10]}"))
        if not rows:
            rows = [("--", "none yet: your first valid lap is the PB")]
        secs.append((f"TOP {rec.TOP_N}", rows))
        tg = _medal_table(self.key)
        if tg and tg.get("author"):
            best = self.book.load(self.key).get("best_medal")
            pb = self.book.pb_time(self.key)
            m_pb = _medal_for(self.key, pb)
            order = rec.MEDAL_ORDER
            if m_pb in order and (best not in order or order.index(m_pb) < order.index(best)):
                best = m_pb                # a PB set before the table: it still counts
            mrows = [(m, rec.fmt_time(tg[m])) for m in ("author", "gold", "silver", "bronze")]
            mrows.append(("yours", (best or "none yet") + (f"  (PB {rec.fmt_time(pb)})"
                                                            if math.isfinite(pb) else "")))
            secs.append(("MEDALS", mrows))
        else:
            secs.append(("MEDALS", [("--", "no reference lap for this class")]))
        return secs + PR_HELP

    # -- the pick page ----------------------------------------------------------
    def pick_items(self) -> list:
        rows = []
        if not self.saved():
            rows.append((f"{self.build_name or '(unnamed)':<24s}"
                         f"{rec.fmt_time(self.build_best(self.build_name, self.build_json)):>10s}"
                         f"  (driving, not saved)",
                         "pr_back"))
        for name in sorted(self.builds):
            mark = "  <- driving" if (name == self.build_name and self.saved()) else ""
            rows.append((f"{name[:24]:<24s}"
                         f"{rec.fmt_time(self.build_best(name, self.builds[name])):>10s}{mark}",
                         f"pr_build:{name}"))
        if not self.builds:
            rows.append(("no saved builds yet (garage > LIBRARY > S saves one)", "pr_back"))
        rows.append(("Back", "pr_back"))
        return rows

    def pick_subtitle(self) -> str:
        return f"{rec.class_label(self.key)}   the time is each build's best in this class"


def _same_build(a, b) -> bool:
    """Two CarBuild jsons describe the same car (the name and the library's
    `builtin` flag do not change the car)."""
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    strip = lambda d: {k: v for k, v in d.items() if k not in ("name", "builtin")}
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
        and "ABS AID MAN" in top[0][1], str(top[:2]))
    rep("medal block present (targets or '--')", "MEDALS" in secs, str(secs.get("MEDALS"))[:80])
    pick = pr.pick_items()
    acts = [a for _, a in pick]
    rep("pick lists every saved build with its best in this class",
        acts == ["pr_build:fast", "pr_build:wet", "pr_back"] and "1:00.900" in pick[0][0]
        and "1:02.500" in pick[1][0] and "<- driving" in pick[0][0], str([p[0] for p in pick]))
    pr2 = PreRace(key, book, "my corsa", dict(b_fast, name="my corsa", slots={"left": {"wing": "x"}}),
                  builds={"fast": b_fast})
    rep("an unsaved build is listed first and marked", pr2.pick_items()[0][1] == "pr_back"
        and "not saved" in pr2.pick_items()[0][0] and "(not saved)" in pr2.items()[1][0])
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
    pr.ghost_label = "reference bot"
    rep("the ghost row appears when set", pr.items()[-1] == ("Ghost 2  reference bot", "set:pr_ghost"))
    if verbose:
        print(f"  {'ALL PASS' if ok else 'FAILURES ABOVE'}: {n_ok}/{n_all} checks")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_check() else 1)
