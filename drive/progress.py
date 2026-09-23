"""drive/progress.py -- `runs/progress.json`: what the player has done (tasks 23, 25).

One small player file with a SECTION per feature: `tutorial` (the driving
tutorial, drive/tutorial.py: offered, the step reached, done, the wing laps
it measured) and, from task 25, `challenges` (stars per challenge). Each
feature reads and writes only its own section; `save(section)` re-reads the
file and replaces that one section, so two writers (the tutorial, the
challenges, a second instance of the game) never drop each other's work.

The file carries a versioned `kind` (`carsim-progress-1`, in the style of
`SEED_LAP_KIND`). A file that does not parse, or is of another kind, is
IGNORED with a note and moved aside as `progress.json.bad-<stamp>` on the
next save -- never a crash, never written over blind. A file that cannot be
READ (a permission) is left alone for the session: nothing is saved over it.
The write is atomic (`records._atomic_json`: a synced temp file, the old file
kept as `.bak`, which a torn write falls back to).

Player files are never read by a scripted or headless run: only the
interactive session (`drive.drive._interactive_session`, a player session)
opens `runs/progress.json`; the self-checks use a temporary directory.
"""
from __future__ import annotations

import json
import os

PROGRESS_KIND = "carsim-progress-1"
PROGRESS_PATH = os.path.join("runs", "progress.json")


class Progress:
    """`runs/progress.json`, in memory. `section(name)` is a live dict;
    `save(name)` writes that section back (merged with the file)."""

    def __init__(self, path: str = PROGRESS_PATH):
        self.path = path
        self.data: dict = {}
        self.notes: list = []
        self._bad = None              # why the file on disk was ignored
        self._unreadable = False      # could not be READ: never write over it
        self.load()

    # ---------------------------------------------------------------- #
    def _read(self, p: str):
        """(dict or None, why-not). Raises OSError when the file cannot be read."""
        with open(p) as fh:
            txt = fh.read()
        try:
            d = json.loads(txt)
        except ValueError as exc:
            return None, f"not JSON ({exc.__class__.__name__})"
        if not isinstance(d, dict):
            return None, "not an object"
        if d.get("kind") != PROGRESS_KIND:
            return None, f"kind {d.get('kind')!r}, not {PROGRESS_KIND!r}"
        return d, ""

    def _disk(self) -> dict:
        """What is on disk now (the file, else its .bak), {} when neither parses."""
        for src in (self.path, self.path + ".bak"):
            if not os.path.exists(src):
                continue
            d, why = self._read(src)
            if d is not None:
                if src != self.path:
                    self.notes.append(f"{os.path.basename(self.path)}: read its .bak")
                return d
            if src == self.path:
                self._bad = why
        return {}

    def load(self) -> "Progress":
        self._bad = None
        try:
            d = self._disk()
        except OSError as exc:
            self._unreadable = True
            self.notes.append(f"{os.path.basename(self.path)} unreadable "
                              f"({exc.__class__.__name__}); progress is not saved this session")
            d = {}
        if self._bad:
            self.notes.append(f"{os.path.basename(self.path)} ignored ({self._bad})")
        self.data = {k: v for k, v in d.items() if k != "kind" and isinstance(v, dict)}
        return self

    def section(self, name: str) -> dict:
        s = self.data.get(name)
        if not isinstance(s, dict):
            s = self.data[name] = {}
        return s

    def _stash(self, why: str) -> None:
        """The ignored file is moved aside, never overwritten blind."""
        import time
        bad = f"{self.path}.bad-{time.strftime('%Y%m%d_%H%M%S')}"
        k = 2
        while os.path.exists(bad):
            bad, k = f"{self.path}.bad-{time.strftime('%Y%m%d_%H%M%S')}-{k}", k + 1
        os.replace(self.path, bad)
        self.notes.append(f"{os.path.basename(self.path)} ignored ({why}); kept as {bad}")
        print(f"progress: {self.notes[-1]}")

    def save(self, name: str) -> bool:
        """Write section `name` back, merged with whatever else is on disk.
        False (with a note) when it could not be saved; never raises."""
        if self._unreadable:
            return False
        from .records import _atomic_json
        try:
            self._bad = None
            disk = self._disk()
            if self._bad and os.path.exists(self.path):
                self._stash(self._bad)
            out = {k: v for k, v in disk.items() if k != "kind" and isinstance(v, dict)}
            out[name] = self.section(name)
            _atomic_json(self.path, dict(kind=PROGRESS_KIND, **out))
            for k, v in out.items():           # what another writer added, kept
                if k != name:
                    self.data[k] = v
            return True
        except OSError as exc:
            self.notes.append(f"progress NOT saved ({exc.__class__.__name__}: {exc})")
            return False


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

    tmp = tempfile.mkdtemp(prefix="carsim_progress_")
    p = os.path.join(tmp, "progress.json")
    a = Progress(p)
    rep("no file: empty, no note", a.data == {} and not a.notes)
    a.section("tutorial").update(step="steer", offered=True)
    rep("save writes the kind and the section", a.save("tutorial")
        and json.load(open(p)) == {"kind": PROGRESS_KIND,
                                   "tutorial": {"step": "steer", "offered": True}})
    b = Progress(p)
    rep("round trip", b.section("tutorial") == {"step": "steer", "offered": True})
    # two writers, two sections: neither loses the other's
    b.section("challenges")["brake_1"] = {"stars": 2}
    b.save("challenges")
    a.section("tutorial")["step"] = "turn1"
    a.save("tutorial")
    c = Progress(p)
    rep("merge on save keeps the other section",
        c.section("tutorial")["step"] == "turn1"
        and c.section("challenges") == {"brake_1": {"stars": 2}},
        json.dumps(c.data))
    rep("the writer learns the other section", a.section("challenges") == {"brake_1": {"stars": 2}})
    # a corrupt file: ignored with a note, moved aside on the next save
    with open(p, "w") as fh:
        fh.write("{not json")
    os.remove(p + ".bak")
    d = Progress(p)
    rep("a corrupt file is ignored with a note, not a crash",
        d.data == {} and any("ignored" in n for n in d.notes), "; ".join(d.notes))
    d.section("tutorial")["offered"] = True
    d.save("tutorial")
    bad = [f for f in os.listdir(tmp) if f.startswith("progress.json.bad-")]
    rep("... and kept aside (.bad-<stamp>) when the next save lands",
        len(bad) == 1 and Progress(p).section("tutorial") == {"offered": True}, str(bad))
    # another kind: ignored the same way
    with open(p, "w") as fh:
        json.dump({"kind": "carsim-progress-0", "tutorial": {"done": True}}, fh)
    if os.path.exists(p + ".bak"):
        os.remove(p + ".bak")
    e = Progress(p)
    rep("an old kind is ignored with a note", e.data == {} and e.notes, "; ".join(e.notes))
    # a torn write: the .bak is read
    Progress(p).save("tutorial")                 # kind current again; .bak = the old kind
    f = Progress(p)
    f.section("tutorial")["step"] = "lap"
    f.save("tutorial")
    with open(p, "w") as fh:
        fh.write('{"kind": "carsim-progress-1", "tut')
    g = Progress(p)
    rep("a torn file falls back to its .bak", g.section("tutorial") == {},
        json.dumps(g.data))
    # an unwritable place: False and a note, never an exception
    blocker = os.path.join(tmp, "blocker")
    open(blocker, "w").close()
    h = Progress(os.path.join(blocker, "progress.json"))
    h.section("tutorial")["x"] = 1
    rep("a failed save returns False with a note", h.save("tutorial") is False and h.notes,
        "; ".join(h.notes))
    rep("the default path is runs/progress.json",
        PROGRESS_PATH == os.path.join("runs", "progress.json"))
    if verbose:
        print(f"progress self-check: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_check() else 1)
