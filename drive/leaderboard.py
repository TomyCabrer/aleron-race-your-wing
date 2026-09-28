"""drive/leaderboard.py -- the leaderboards: one per map, car and wing mode (task 47).

The owner (2026-09-28): "Make a clear way for the leaderboard to be seen ...
for the player and for the bot. 1 leaderboard for every track and car type.
Best time and name of car should appear. Also compared to the bot. Should be
for default car. Also for race there should be the same modes as challenge
for the wing types so separate leaderboards. Non stock engines are only for
messing around and don't go to leaderboards (this should be mentioned)."
Their answers to the three questions it raised: the bot side is the player's
own TRAINED bots (a checkpoint -- never the built-in driver); only the
default surface counts; a timed session picks one of the challenges' four
wing modes, and FREE (the build exactly as designed, every G mode) is still
driven but is never on a board.

A BOARD is `track|car|wings`:

    track  records.LAP_TRACKS -- the four circuits, the open map and the
           standard 50 m skidpad (the dragstrip has no lap)
    car    cars.CAR_ORDER
    wings  challenges.CONFIGS -- FULL WING (top + side), ONLY TOP, ONLY TOP
           FIXED, TOP FIXED + SIDE

6 x 5 x 4 = 120 boards on today's maps and cars (the key is any map with
a lap, any car). The engine (Stock: "the default car") and the
surface ('patch', Dry with wet patches, the default) are not in the key: no
other engine or surface is ever on a board.

Each board has two sides, fastest first, TOP_N kept on each:

    you    the player's laps: the best lap of each BUILD, by its name (the
           "name of car"), with the assists and the date
    bots   the player's trained bots' laps: the best of each bot, with the
           build it drove and how the lap was set (a race, or the RACE
           page's Test)

and the page compares the two: your best against your best bot's, and the
gap.

What goes on a board (`why_not` says why a session does not):

    * the Stock engine. Tuned and Sport are for messing around (the owner's
      words) and never go on a board -- the pages, the Settings page's
      Engine row and the session's start say so (`ENGINE_NOTE`);
    * the default surface, 'Dry, wet patches' -- and no T (wet toggle) in
      the lap: the recorder voids such a lap anyway;
    * a wing mode: the session drives its build in one of the four configs
      (`challenges.config_build`: the build's own wings where it has them,
      the stock ones lent where it has none, the side wings gone on a top-
      only mode, the top wing fixed on a FIXED one) with G limited to that
      mode (`challenges.g_modes`). FREE is not a mode on a board;
    * wings within the car's physical limits (task 41: an Unlimited build is
      never official);
    * a valid lap round the circuit: the lap-records' own rule
      (`records.LapRecorder` files it first; only a lap it filed counts).
      A bot's lap in a race counts when its timer calls it valid, it went
      round (95 % of the length) and the bot was not put back on the track
      in it; a Test's lap is the test's own best flying lap.

A bot's lap goes on the board of the CAR it drove, in the session's wing
mode: a race puts every bot in the same mode as the player (its own bred
build with the mode applied -- `race_grid.own_car(config=)`), so a race in
ONLY TOP is an ONLY TOP race for everyone.

Storage: `runs/leaderboard_local/<track>__<car>__<wings>.json`, kind
`carsim-leaderboard-1`, written atomically (records._atomic_json: the old
file kept as `.bak`) and MERGED with the disk before every write, so a
second writer (another instance of the game) loses nothing. A corrupt or
foreign file is moved aside with a note, never a crash. Scripted and
headless runs never read or write it: only `drive.drive`'s interactive
session opens the folder (the self-check uses a temporary one). The local
folder is plan T28's `LocalBackend`; an online backend would read the same
entries (records.publishable is the online rule, and Unlimited laps are
already kept off).

    python3 -m drive.leaderboard            the self-check
    python3 -m drive.leaderboard --show     every board with a time, printed
"""

from __future__ import annotations

import json
import math
import os
import threading
import time

from . import records as rec

KIND = "carsim-leaderboard-1"
BOARDS_DIR = os.path.join("runs", "leaderboard_local")
#: entries kept on each side of a board
TOP_N = 10
#: the two sides of a board
YOU, BOTS = "you", "bots"
SIDES = (YOU, BOTS)

#: the only engine and surface on a board (the owner: "for default car";
#: non-stock engines "are only for messing around")
ENGINE = "stock"
SURFACE = "patch"
#: the challenges' four wing configs, in the page's order; a literal, not an
#: import, so this module loads without the garage (the self-check asserts
#: it is challenges.CONFIGS)
CONFIGS = ("full", "top", "top_fixed", "top_fixed_side")
#: the build as designed, every G mode: driven, never on a board
FREE = "free"
#: the Wings setting's values (Settings.race_wings), in LEFT / RIGHT order
WINGS_MODES = CONFIGS + (FREE,)
WINGS_DEFAULT = FREE
#: a wing mode in a row's few letters
WINGS_SHORT = {"full": "FULL WING", "top": "ONLY TOP", "top_fixed": "ONLY TOP FIXED",
               "top_fixed_side": "TOP FIXED + SIDE", FREE: "FREE"}
#: ... and in the Wings row's words (the challenge page's, plus FREE)
WINGS_LABELS = {"full": "FULL WING: top + side", "top": "ONLY TOP",
                "top_fixed": "ONLY TOP, FIXED", "top_fixed_side": "TOP FIXED + SIDE",
                FREE: "FREE: your build as designed (not on the leaderboards)"}
#: the maps a board exists on (records.LAP_TRACKS: a lap time exists there)
TRACKS = rec.LAP_TRACKS
#: the owner's "this should be mentioned": every page with a board on it
ENGINE_NOTE = ("Tuned and Sport engines are just for messing around: "
               "they never go on the leaderboards (Stock engine only).")
#: the Engine setting's non-stock values, as a sentence starts them
ENGINE_WORDS = {"tuned": "Tuned", "sport": "Sport"}

#: the page's help: what a board is and what counts
RULES = [("WHAT COUNTS", [
    ("boards", "one per map, car and wing mode"),
    ("engine", "Stock only: Tuned and Sport are just for messing around, never on a board"),
    ("surface", "Dry, wet patches (the default) only"),
    ("wings", "a wing mode, picked on TIME TRIAL > Wings (or Settings); FREE never counts"),
    ("limits", "an Unlimited build (a wing past its car's limit) never counts"),
    ("you", "your best valid lap with each build, by its name"),
    ("bots", "your trained bots: a race, or the RACE page's Test, in the same "
             "wing mode on a Stock engine"),
])]


# ==================================================================== #
#  THE BOARD                                                           #
# ==================================================================== #
def board_key(track: str, car: str, wings: str) -> str:
    """`track|car|wings`."""
    return "|".join((str(track), str(car), str(wings)))


def split_board(key: str) -> tuple:
    parts = str(key).split("|")
    if len(parts) != 3:
        raise ValueError(f"not a board key: {key!r}")
    return tuple(parts)


def board_file(key: str) -> str:
    """The file name of a board: `|` -> `__` (Windows forbids `|`)."""
    for p in split_board(key):
        if not p or not all(c.isalnum() or c in "-_" for c in p):
            raise ValueError(f"board key component {p!r} is not a plain name")
    return key.replace("|", "__") + ".json"


def all_boards(cars_order=None) -> list:
    """Every board key: TRACKS x cars x CONFIGS (cars.CAR_ORDER by default)."""
    if cars_order is None:
        import cars as _cars
        cars_order = _cars.CAR_ORDER
    return [board_key(t, c, w) for t in TRACKS for c in cars_order for w in CONFIGS]


def board_title(key: str) -> str:
    """'Arena circuit  ·  Opel Corsa C 1.2  ·  FULL WING'; the key's own
    words when a name cannot be looked up."""
    try:
        t, c, w = split_board(key)
    except ValueError:
        return str(key)
    try:
        from . import track as trk
        tt = trk.TRACK_TITLES.get(t, t)
    except Exception:                      # noqa: BLE001 -- only the words need it
        tt = t
    try:
        import cars as _cars
        ct = _cars.car_name(c) if c in _cars.CARS else c
    except Exception:                      # noqa: BLE001
        ct = c
    return f"{tt}  ·  {ct}  ·  {WINGS_SHORT.get(w, w)}"


def why_not(track: str, car: str, engine: str, surface: str, wings,
            unlimited: bool = False, radius: float = 50.0, cw: bool = False) -> str:
    """"" when a session with these settings races for a board, else why it
    does not -- one line, in the player's words (the TIME TRIAL page and the
    HUD show it). The engine comes first: the owner asked for it to be said."""
    r = rec.records_reason(track, radius, cw)
    if r:
        return r
    if engine != ENGINE:
        return (f"{ENGINE_WORDS.get(engine, str(engine).capitalize())} engine: just for "
                f"messing around, not on the leaderboards (Stock engine only)")
    if surface != SURFACE:
        return "the leaderboards are on the default surface only (Dry, wet patches)"
    if wings not in CONFIGS:
        return "wings FREE: pick a wing mode (TIME TRIAL > Wings) to race for a leaderboard"
    if unlimited:
        return "an Unlimited build (a wing past its car's limit) is never on a leaderboard"
    try:
        import cars as _cars
        if car not in _cars.CARS:
            return f"no leaderboard for the car '{car}'"
    except Exception:                      # noqa: BLE001
        pass
    return ""


def session_board(track, car, engine, surface, wings, unlimited=False,
                  radius: float = 50.0, cw: bool = False) -> tuple:
    """(board key or None, why not) for a session."""
    why = why_not(track, car, engine, surface, wings, unlimited, radius, cw)
    return (None, why) if why else (board_key(track, car, wings), "")


def _stash_bad(path: str, why: str) -> None:
    """A corrupt or foreign board is moved aside, never overwritten blind
    (records._stash_bad's rule, in this module's words)."""
    bad = f"{path}.bad-{time.strftime('%Y%m%d_%H%M%S')}"
    try:
        os.replace(path, bad)
        print(f"leaderboard: {os.path.basename(path)} ignored ({why}); kept as {bad}")
    except OSError:
        print(f"leaderboard: {os.path.basename(path)} ignored ({why})")


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M")


def _entry_ok(e) -> bool:
    return (isinstance(e, dict) and isinstance(e.get("time"), (int, float))
            and not isinstance(e.get("time"), bool)
            and math.isfinite(e["time"]) and e["time"] > 0.0
            and isinstance(e.get("name"), str) and bool(e["name"].strip()))


def _best_each(entries) -> list:
    """The fastest entry of each name, fastest first, TOP_N of them (a tie
    keeps the one set first)."""
    best: dict = {}
    for e in entries:
        if not _entry_ok(e):
            continue
        n = e["name"]
        if n not in best or e["time"] < best[n]["time"]:
            best[n] = e
    return sorted(best.values(), key=lambda e: (e["time"], str(e.get("date", ""))))[:TOP_N]


class Boards:
    """`runs/leaderboard_local/`: every board, loaded on demand, cached.
    Safe from more than one thread (one lock); `save` merges with the disk
    first. A file that cannot be READ is left alone for the session (never
    written over); one that does not PARSE is moved aside (its `.bak` is
    used when there is one)."""

    def __init__(self, root: str = BOARDS_DIR):
        self.root = root
        self._cache: dict = {}
        self._lock = threading.RLock()
        self.notes: list = []
        self.save_error = ""

    def path(self, key: str) -> str:
        return os.path.join(self.root, board_file(key))

    @staticmethod
    def _empty(key: str) -> dict:
        return {"kind": KIND, "key": key, YOU: [], BOTS: []}

    def _parse(self, p: str, key: str) -> dict:
        with open(p) as fh:
            text = fh.read()
        try:
            raw = json.loads(text)
            if not isinstance(raw, dict) or raw.get("kind") != KIND:
                raise ValueError(f"kind {raw.get('kind') if isinstance(raw, dict) else None!r}"
                                 f", want {KIND!r}")
            if raw.get("key") != key:
                raise ValueError(f"key {raw.get('key')!r}")
            d = self._empty(key)
            for side in SIDES:
                v = raw.get(side, [])
                d[side] = _best_each(v if isinstance(v, list) else [])
            return d
        except Exception as exc:           # noqa: BLE001 -- any parse / schema failure
            raise ValueError(str(exc)) from exc

    def load(self, key: str) -> dict:
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            d = self._empty(key)
            p = self.path(key)
            src = p if os.path.exists(p) else (p + ".bak" if os.path.exists(p + ".bak") else None)
            if src is not None:
                try:
                    d = self._parse(src, key)
                except OSError as exc:
                    self.notes.append(f"{os.path.basename(p)}: unreadable ({exc}); "
                                      f"not written this session")
                    print(f"leaderboard: {self.notes[-1]}")
                    d = self._empty(key)
                    d["_unreadable"] = str(exc)
                except ValueError as exc:
                    self.notes.append(f"{os.path.basename(src)}: {exc}")
                    _stash_bad(src, str(exc))
                    d = self._empty(key)
                    if src == p and os.path.exists(p + ".bak"):
                        try:
                            d = self._parse(p + ".bak", key)
                            self.notes.append(f"{os.path.basename(p)}: restored from its .bak")
                        except Exception:  # noqa: BLE001
                            d = self._empty(key)
            self._cache[key] = d
            return d

    def save(self, key: str):
        """Write the board (merged with what is on disk now). None on a
        failed write, with the reason in `save_error`: a board that cannot
        be saved never stops the car."""
        p = self.path(key)
        with self._lock:
            d = self.load(key)
            try:
                if d.get("_unreadable"):
                    raise OSError(f"the file was unreadable at load ({d['_unreadable']}); "
                                  f"not written over")
                if os.path.exists(p):
                    try:
                        disk = self._parse(p, key)
                    except ValueError:
                        disk = None        # the next load stashes it; ours wins
                    if disk is not None:
                        for side in SIDES:
                            d[side] = _best_each(d[side] + disk[side])
                out = {k: v for k, v in d.items() if not k.startswith("_")}
                rec._atomic_json(p, out)
            except OSError as exc:
                self.save_error = f"{type(exc).__name__}: {exc}"
                self.notes.append(f"{os.path.basename(p)}: not saved ({self.save_error})")
                print(f"leaderboard: {os.path.basename(p)} not saved ({self.save_error})")
                return None
            self.save_error = ""
            return p

    # -- reading ----------------------------------------------------------
    def top(self, key: str, side: str = YOU, n: int = TOP_N) -> list:
        with self._lock:
            return list(self.load(key)[side][:n])

    def best(self, key: str, side: str = YOU):
        t = self.top(key, side, 1)
        return t[0] if t else None

    def compare(self, key: str) -> dict:
        """Your best against your best bot's: {you, bot, gap} with gap =
        yours - the bot's (negative: you are faster), None without both."""
        y, b = self.best(key, YOU), self.best(key, BOTS)
        gap = (float(y["time"]) - float(b["time"])) if (y and b) else None
        return dict(you=y, bot=b, gap=gap)

    # -- writing ----------------------------------------------------------
    def submit(self, key: str, side: str, entry: dict, save: bool = True) -> dict:
        """File one lap. `entry`: time, name (+ anything else: build, assists,
        how). Returns {rank: its place on its side (1-based) or None when
        it is not the best of its name or not in the top TOP_N, name_best:
        the name's best so far, board_best: the side's best, saved: the path
        or None}."""
        if side not in SIDES:
            raise ValueError(f"no side {side!r}")
        split_board(key)
        e = dict(entry)
        e.setdefault("date", _now())
        if not _entry_ok(e):
            return dict(rank=None, name_best=False, board_best=False, saved=None)
        with self._lock:
            d = self.load(key)
            before = next((x for x in d[side] if x["name"] == e["name"]), None)
            prev_top = d[side][0]["time"] if d[side] else float("inf")
            d[side] = _best_each(d[side] + [e])
            rank = next((i + 1 for i, x in enumerate(d[side]) if x is e), None)
            out = dict(rank=rank, name_best=rank is not None and (before is None
                                                                  or e["time"] < before["time"]),
                       board_best=rank == 1 and e["time"] < prev_top, saved=None)
            if rank is not None and save:
                out["saved"] = self.save(key)
            return out


# ==================================================================== #
#  WHAT THE PAGES SAY  (pure: rows and sections for drive.menu)         #
# ==================================================================== #
def gap_text(gap) -> str:
    """'you ahead by 4.477 s' / 'bot ahead by 1.203 s' / 'dead heat'."""
    if gap is None or not math.isfinite(gap):
        return ""
    if abs(gap) < 0.0005:
        return "dead heat"
    return (f"you ahead by {-gap:.3f} s" if gap < 0.0 else f"bot ahead by {gap:.3f} s")


def _cut(s, n: int) -> str:
    s = str(s or "")
    return s if len(s) <= n else s[:n - 1] + "~"


def car_row(boards: Boards, key: str, car_label: str) -> str:
    """One car on the LEADERBOARDS page: your best (and its build), your best
    bot's (and its name), and the gap."""
    c = boards.compare(key)
    y, b = c["you"], c["bot"]
    you = (f"you {rec.fmt_time(y['time']):>8s} {_cut(y['name'], 13):<13s}" if y
           else f"you {'--':>8s} {'no lap yet':<13s}")
    bot = (f"bot {rec.fmt_time(b['time']):>8s} {_cut(b['name'], 13):<13s}" if b
           else f"bot {'--':>8s} {'no bot lap':<13s}")
    return f"{_cut(car_label, 7):<8s}{you}  {bot}  {gap_text(c['gap'])}".rstrip()


def _you_rows(entries) -> list:
    rows = []
    for i, e in enumerate(entries):
        a = e.get("assists") if isinstance(e.get("assists"), dict) else {}
        tags = [t for t, on in (("ABS", a.get("abs")), ("TC", a.get("tc")),
                                ("aid", a.get("steer_aid"))) if on]
        rows.append((f"{i + 1:>2d}  {rec.fmt_time(e['time'])}",
                     "  ".join(p for p in (_cut(e["name"], 20), " ".join(tags),
                                           str(e.get("date", ""))[:10]) if p)))
    return rows or [("--", "no lap yet: race this map, car and wing mode")]


def _bot_rows(entries) -> list:
    rows = []
    for i, e in enumerate(entries):
        how = {"race": "race", "test": "test"}.get(e.get("how"), "")
        build = e.get("build") or ""
        rows.append((f"{i + 1:>2d}  {rec.fmt_time(e['time'])}",
                     "  ".join(p for p in (_cut(e["name"], 20),
                                           (f"({_cut(build, 16)})" if build else ""), how,
                                           str(e.get("date", ""))[:10]) if p)))
    return rows or [("--", "no bot lap yet: race a trained bot here, or Test it "
                           "(ESC > Race vs bot)")]


def detail_sections(boards: Boards, key: str, n: int = TOP_N) -> list:
    """One board in full: the comparison, then both sides' top `n`."""
    c = boards.compare(key)
    y, b = c["you"], c["bot"]
    vs = [("you", (f"{rec.fmt_time(y['time'])}  {_cut(y['name'], 22)}" if y else "--")),
          ("bot", (f"{rec.fmt_time(b['time'])}  {_cut(b['name'], 22)}" if b else "--"))]
    if c["gap"] is not None:
        vs.append(("gap", gap_text(c["gap"])))
    return [(f"{board_title(key)}", vs),
            ("YOU: best of each build", _you_rows(boards.top(key, YOU, n))),
            ("YOUR BOTS: best of each bot", _bot_rows(boards.top(key, BOTS, n)))]


def prerace_rows(boards, key, why: str) -> list:
    """The TIME TRIAL page's LEADERBOARD section: this session's board --
    your best, your best bot's, the gap -- or why the session is not on
    one (a Tuned engine, FREE wings ...)."""
    if key is None or boards is None:
        rows = [("not on", why or "no leaderboard for this session")]
        if "engine" in (why or ""):
            rows.append(("", ENGINE_NOTE))
        return rows
    c = boards.compare(key)
    y, b = c["you"], c["bot"]
    rows = [("board", board_title(key)),
            ("you", f"{rec.fmt_time(y['time'])}  {_cut(y['name'], 22)}" if y
             else "no lap yet: your first valid lap is on it"),
            ("bot", f"{rec.fmt_time(b['time'])}  {_cut(b['name'], 22)}" if b
             else "no bot lap yet (ESC > Race vs bot: race or Test one)")]
    if c["gap"] is not None:
        rows.append(("gap", gap_text(c["gap"])))
    rows.append(("all", "ESC > Leaderboards: every map, car and wing mode"))
    return rows


def show(root: str = BOARDS_DIR) -> int:
    """`--show`: every board with a time on it, printed."""
    import cars as _cars
    boards = Boards(root)
    n = 0
    for key in all_boards(_cars.CAR_ORDER):
        c = boards.compare(key)
        if c["you"] is None and c["bot"] is None:
            continue
        n += 1
        print(board_title(key))
        for side, rows in ((YOU, _you_rows(boards.top(key, YOU))),
                           (BOTS, _bot_rows(boards.top(key, BOTS)))):
            print(f"  {side}")
            for k, v in rows:
                print(f"    {k:10s} {v}")
        if c["gap"] is not None:
            print(f"  {gap_text(c['gap'])}")
    print(f"{n} of {len(all_boards(_cars.CAR_ORDER))} boards have a time")
    return 0


# ==================================================================== #
#  SELF-CHECK                                                          #
# ==================================================================== #
def self_check(verbose: bool = True) -> bool:
    import tempfile
    ok = True

    def rep(tag, passed, msg=""):
        nonlocal ok
        ok = ok and bool(passed)
        if verbose:
            print(f"  [{'ok' if passed else 'FAIL'}] {tag}" + (f": {msg}" if msg else ""))

    import cars as _cars
    from . import track as trk
    from .challenges import CONFIGS as CH_CONFIGS, CONFIG_LABELS
    rep("the wing modes are the challenges' four, FREE last; the maps are the records'",
        CONFIGS == tuple(CH_CONFIGS) and WINGS_MODES[-1] == FREE and len(WINGS_MODES) == 5
        and all(WINGS_LABELS[c] == CONFIG_LABELS[c] for c in CONFIGS)
        and TRACKS == rec.LAP_TRACKS and all(t in trk.TRACKS for t in TRACKS)
        and WINGS_DEFAULT == FREE, str(WINGS_MODES))
    keys = all_boards(_cars.CAR_ORDER)
    n_b = len(TRACKS) * len(_cars.CAR_ORDER) * len(CONFIGS)
    rep(f"one board per map x car x wing mode: {len(TRACKS)} x {len(_cars.CAR_ORDER)} x "
        f"{len(CONFIGS)} = {n_b}, each its own file",
        len(keys) == n_b and len({board_file(k) for k in keys}) == n_b
        and board_file("arena|corsa|top_fixed") == "arena__corsa__top_fixed.json"
        and split_board("arena|corsa|full") == ("arena", "corsa", "full"), f"{len(keys)}")
    bad_key = False
    try:
        board_file("arena|../x|full")
    except ValueError:
        bad_key = True
    rep("a key with a path in it is refused", bad_key)

    # -- what counts ---------------------------------------------------------
    k0, w0 = session_board("arena", "corsa", "stock", "patch", "full")
    kt, wt = session_board("arena", "corsa", "tuned", "patch", "full")
    ks, ws = session_board("arena", "corsa", "sport", "patch", "full")
    kw, ww = session_board("arena", "corsa", "stock", "all", "full")
    kf, wf = session_board("arena", "corsa", "stock", "patch", FREE)
    ku, wu = session_board("arena", "corsa", "stock", "patch", "top", unlimited=True)
    kd, wd = session_board("dragstrip", "corsa", "stock", "patch", "full")
    k30, w30 = session_board("skidpad", "corsa", "stock", "patch", "full", radius=30.0)
    rep("Stock + default surface + a wing mode + real limits: on its board",
        k0 == "arena|corsa|full" and w0 == "", f"{k0} {w0!r}")
    rep("a Tuned / Sport engine is NOT, and says it is for messing around (the owner)",
        kt is None and ks is None and "Tuned engine" in wt and "Sport engine" in ws
        and "messing around" in wt and "Stock" in wt and "messing around" in ENGINE_NOTE,
        f"{wt!r}")
    rep("another surface, FREE wings, an Unlimited build, the dragstrip, a 30 m pad: not on one",
        kw is None and "default surface" in ww and kf is None and "FREE" in wf
        and ku is None and "Unlimited" in wu and kd is None and "dragstrip" in wd
        and k30 is None and "50 m" in w30, f"{ww!r} {wf!r} {wu!r} {wd!r} {w30!r}")
    rep("the engine is named first: a Tuned engine on FREE wings still says the engine",
        "Tuned engine" in why_not("arena", "corsa", "tuned", "all", FREE))

    # -- the file --------------------------------------------------------------
    root = tempfile.mkdtemp(prefix="carsim_boards_")
    b = Boards(root)
    key = "arena|corsa|full"
    r1 = b.submit(key, YOU, dict(time=61.5, name="low drag", assists=dict(abs=True, tc=True)))
    r2 = b.submit(key, YOU, dict(time=60.9, name="big wing"))
    r3 = b.submit(key, YOU, dict(time=61.2, name="low drag"))     # the build's new best
    r4 = b.submit(key, YOU, dict(time=62.0, name="big wing"))     # slower: not its best
    rep("you: the best lap of each BUILD, fastest first; a slower lap of a build is no entry",
        [(e["name"], e["time"]) for e in b.top(key, YOU)] == [("big wing", 60.9), ("low drag", 61.2)]
        and r1["rank"] == 1 and r2["rank"] == 1 and r2["board_best"] and r3["rank"] == 2
        and r3["name_best"] and not r3["board_best"] and r4["rank"] is None
        and r2["saved"] and os.path.exists(b.path(key)),
        str([r1, r2, r3, r4]))
    rb1 = b.submit(key, BOTS, dict(time=62.4, name="swarm_arena", build="my corsa", how="race"))
    rb2 = b.submit(key, BOTS, dict(time=61.8, name="swarm_arena", build="my corsa", how="test"))
    c = b.compare(key)
    rep("bots: the best of each bot; compared: your best against your best bot's",
        rb1["rank"] == 1 and rb2["rank"] == 1 and len(b.top(key, BOTS)) == 1
        and c["you"]["name"] == "big wing" and c["bot"]["time"] == 61.8
        and abs(c["gap"] - (60.9 - 61.8)) < 1e-12 and gap_text(c["gap"]) == "you ahead by 0.900 s"
        and gap_text(0.25) == "bot ahead by 0.250 s" and gap_text(None) == "", str(c))
    for i in range(TOP_N + 3):
        b.submit(key, YOU, dict(time=70.0 + i, name=f"b{i}"), save=False)
    b.save(key)
    rep(f"a side keeps its top {TOP_N}",
        len(b.top(key, YOU)) == TOP_N and b.top(key, YOU)[0]["name"] == "big wing")
    b2 = Boards(root)
    rep("a second reader sees the file as written",
        [e["name"] for e in b2.top(key, YOU)] == [e["name"] for e in b.top(key, YOU)]
        and b2.best(key, BOTS)["time"] == 61.8)
    # two writers: each keeps the other's laps (merge on save)
    b3 = Boards(root)
    b3.load(key)
    b.submit(key, YOU, dict(time=59.0, name="writer one"))
    b3.submit(key, BOTS, dict(time=60.0, name="bot two"))
    b4 = Boards(root)
    rep("two writers at once: neither loses the other's lap (merged on save)",
        b4.best(key, YOU)["name"] == "writer one" and b4.best(key, BOTS)["name"] == "bot two",
        f"{b4.best(key, YOU)} {b4.best(key, BOTS)}")
    bad = [dict(time=float("nan"), name="x"), dict(time=-1.0, name="x"), dict(time=5.0, name=""),
           dict(time=True, name="x"), dict(name="x"), "junk"]
    rep("a lap with no time, a bad time or no name is refused",
        all(b.submit(key, YOU, e if isinstance(e, dict) else {}, save=False)["rank"] is None
            for e in bad))
    # a corrupt file: moved aside, the .bak read
    p = b.path(key)
    with open(p, "w") as fh:
        fh.write("{not json")
    b5 = Boards(root)
    got = b5.top(key, YOU)
    stashed = [f for f in os.listdir(root) if ".bad-" in f]
    rep("a corrupt board is moved aside (never a crash) and its .bak read",
        bool(stashed) and bool(got) and bool(b5.notes), f"{stashed} {b5.notes}")
    other = os.path.join(root, board_file("linden|540i|top"))
    with open(other, "w") as fh:
        json.dump(dict(kind="something-else", key="linden|540i|top"), fh)
    b6 = Boards(root)
    rep("a foreign file is ignored with a note", b6.top("linden|540i|top", YOU) == []
        and any("kind" in n for n in b6.notes), str(b6.notes))

    # -- the pages ---------------------------------------------------------------
    row = car_row(b4, key, "Civetta")
    row0 = car_row(b4, "kestrel|express|top", "Courier")
    rep("a car's row: your best and its build, your bot's and its name, the gap",
        row.startswith("Civetta you ") and "writer one" in row and "bot two" in row
        and "you ahead by 1.000 s" in row and "no lap yet" in row0 and "no bot lap" in row0
        and len(row) <= 84, row)
    secs = detail_sections(b4, key)
    rep("a board in full: the comparison, then both sides",
        len(secs) == 3 and secs[1][0].startswith("YOU") and secs[2][0].startswith("YOUR BOTS")
        and secs[1][1][0][1].startswith("writer one") and "Arena" in secs[0][0]
        and any(k == "gap" for k, _ in secs[0][1]), str(secs[0]))
    pr = prerace_rows(b4, key, "")
    pr_t = prerace_rows(None, None, wt)
    rep("the TIME TRIAL page's section: this board, or why not (the engine note said)",
        pr[0][0] == "board" and pr[1][0] == "you" and pr[2][0] == "bot" and pr[3][0] == "gap"
        and pr_t[0] == ("not on", wt) and pr_t[1][1] == ENGINE_NOTE, str(pr_t))
    rep("the rules the page shows name the engine, the surface, the wings, the bots",
        all(any(k == w for k, _ in RULES[0][1]) for w in ("engine", "surface", "wings", "bots"))
        and "messing around" in dict(RULES[0][1])["engine"])
    if verbose:
        print(f"leaderboard self-check: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import sys
    if "--show" in sys.argv[1:]:
        sys.exit(show())
    sys.exit(0 if self_check() else 1)
