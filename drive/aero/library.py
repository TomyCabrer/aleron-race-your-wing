"""The in-game library: airfoils, wings and builds as JSON under
runs/library/, plus the polar cache and a background XFOIL worker.

    runs/library/airfoils/<name>.json   AirfoilSpec
    runs/library/wings/<name>.json      WingSpec (with its cached aero)
    runs/library/builds/<name>.json     CarBuild (drive/garage.py owns it)
    runs/library/polars/<sha>_re..json  XFOIL polars (xfoil.py)

A wing refers to its section by NAME, so a section improved later flows
into every wing that uses it once those are re-analysed; a build refers to
its wings by name the same way. Built-in entries (the published fin/plate,
the bundled sections) are re-seeded if deleted; user entries are files the
user owns.

Builds are one list for every car; since task 41 each carries the car it was
made for (`"car"`, "" for a build saved before -- any car), which decides
only the ORDER a car is offered them in (`drive.prerace.pick_order`). A car's
own default build is a NAME in the drive's Settings (`car_build`), so
`rename` leaves it to the caller to move that reference (the garage does).

XFOIL is never run on the caller's thread: `polar()` returns immediately
with the cached XFOIL polar when there is one, otherwise the ESTIMATE, and
queues the XFOIL run; `poll()` hands back the names whose polar just
improved so the caller can re-analyse.
"""

from __future__ import annotations

import json
import os
import queue
import re as _re
import threading
import time

import numpy as np

from .airfoil import AirfoilSpec, bundled_dat_files, load_dat, DATA_DIR
from .polar import Polar, estimate_polar
from .wing import WingSpec, re_bank_snap, V_REF, TOP_PYLON_L, analyse
from . import xfoil

ROOT_DEFAULT = os.path.join("runs", "library")

BUILTIN_AIRFOILS = [
    ("naca0012", "naca", "0012", "symmetric 12 %: endplates, neutral fins"),
    ("naca2412", "naca", "2412", "the AeroBO default section"),
    ("naca4412", "naca", "4412", "4 % camber, forgiving stall"),
    ("naca6412", "naca", "6412", "6 % camber, high lift"),
    ("naca0009", "naca", "0009", "thin symmetric plate-like"),
]
DAT_NOTES = {
    "s1223": "Selig high-lift, cl_max ~2.2 at low Re: the race-wing classic",
    "e423": "Eppler high-lift, thick nose, gentle stall",
    "ch10sm": "CH10 smoothed: very high lift, draggy",
    "fx74cl5140": "Wortmann FX 74-CL5-140: high-lift sailplane flap section",
    "clarky": "Clark Y: flat bottom, easy to build",
    "e387": "Eppler 387: low-Re efficiency",
    "s7055": "Selig S7055: low drag bucket",
    "mh32": "MH 32: low camber, sharp",
    "naca23012": "NACA 23012: low pitching moment",
    "goe795": "Gottingen 795: thick, high lift",
}


def _safe(name: str) -> str:
    s = _re.sub(r"[^a-z0-9_\-]+", "-", name.strip().lower()).strip("-")
    return s or "item"


class Library:
    def __init__(self, root: str = ROOT_DEFAULT, use_xfoil: bool = True):
        self.root = root
        self.dirs = {k: os.path.join(root, k) for k in ("airfoils", "wings", "builds", "polars")}
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)
        self.airfoils: dict[str, AirfoilSpec] = {}
        self.wings: dict[str, WingSpec] = {}
        self.builds: dict[str, dict] = {}
        self._polars: dict[tuple, Polar] = {}
        self._tc: dict = {}
        self.use_xfoil = bool(use_xfoil) and xfoil.available()
        self._q: queue.Queue = queue.Queue()
        self._done: queue.Queue = queue.Queue()
        self._pending: set = set()
        self._worker: threading.Thread | None = None
        self.xfoil_busy = ""
        self.log: list[str] = []
        self.load()
        self.seed_defaults()

    # ---------------------------------------------------------------- io ----
    def load(self) -> None:
        for d in ("airfoils", "wings", "builds"):
            store = getattr(self, d)
            store.clear()
            for f in sorted(os.listdir(self.dirs[d])):
                if not f.endswith(".json"):
                    continue
                try:
                    with open(os.path.join(self.dirs[d], f)) as fh:
                        rec = json.load(fh)
                    if not isinstance(rec, dict):
                        self.log.append(f"{d}/{f}: not a record ({type(rec).__name__}), skipped")
                        continue
                    if d == "airfoils":
                        a = AirfoilSpec.from_json(rec)
                        store[a.name] = a
                    elif d == "wings":
                        w = WingSpec.from_json(rec)
                        store[w.name] = w
                    else:
                        store[str(rec.get("name", f[:-5]))] = rec
                except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                    self.log.append(f"{d}/{f}: {exc}")

    def _write(self, kind: str, name: str, obj: dict, replaces: str | None = None) -> str:
        path = os.path.join(self.dirs[kind], _safe(name) + ".json")
        #  the file is named by the FOLDED name, so 'Kestrel Fast' and
        #  'kestrel-fast' are one file. Callers take `unique_name` first; this
        #  is the guard behind it: another record's file is never overwritten.
        #  A built-in (re-seeded on every start) stays in memory only rather
        #  than take a user's file. `replaces`: the record being RENAMED, whose
        #  own file this may be ('fast' -> 'Fast' is one file, task 41).
        try:
            with open(path) as fh:
                held = json.load(fh)
            held = held.get("name") if isinstance(held, dict) else None
        except (OSError, ValueError):
            held = None
        if held is not None and held != name and held != replaces:
            if obj.get("builtin"):
                self.log.append(f"{kind}/{os.path.basename(path)} holds '{held}': built-in '{name}' not written")
                return path
            raise ValueError(f"{os.path.basename(path)} already holds '{held}'")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=1)
        os.replace(tmp, path)
        return path

    #  written first, then listed: a save that raises (a full disk, a
    #  read-only runs/, the guard in `_write`) leaves no entry that is not on disk
    def save_airfoil(self, a: AirfoilSpec) -> str:
        path = self._write("airfoils", a.name, a.to_json())
        self.airfoils[a.name] = a
        return path

    def save_wing(self, w: WingSpec) -> str:
        path = self._write("wings", w.name, w.to_json())
        self.wings[w.name] = w
        return path

    def save_build(self, b: dict) -> str:
        path = self._write("builds", b["name"], b)
        self.builds[b["name"]] = b
        return path

    def delete(self, kind: str, name: str) -> bool:
        store = getattr(self, kind)
        if name not in store:
            return False
        obj = store.pop(name)
        try:
            os.remove(os.path.join(self.dirs[kind], _safe(name) + ".json"))
        except OSError:
            pass
        if getattr(obj, "builtin", False) or (isinstance(obj, dict) and obj.get("builtin")):
            self.seed_defaults()
        return True

    def rename(self, kind: str, old: str, new: str) -> str:
        """Rename the user build `old` to `new` (task 41); returns the name it
        has now. The new name must be free as a FILE, as every save's is: one
        that folds onto another record's file is taken as `unique_name` gives
        it ('Fast' beside a 'fast' becomes 'Fast-2'), while one that folds onto
        the build's OWN file ('fast' -> 'Fast') is simply that file rewritten.
        The new file is written before the old one goes, so a write that fails
        (a full disk, the guard) leaves the build as it was, under its old
        name. Only builds: a wing is named by the builds that carry it, so a
        wing rename would have to rewrite them (not offered). Raises KeyError
        for a name not in the library, ValueError for a built-in, an empty
        name, another kind, or a failed write's guard.

        Whatever else points at the old name -- a car's default build in the
        drive's Settings (`car_build`), the per-map memory -- is the caller's
        to update: the garage moves the Settings reference with it."""
        if kind != "builds":
            raise ValueError(f"only builds are renamed, not {kind}")
        store = self.builds
        if old not in store:
            raise KeyError(old)
        rec = store[old]
        if isinstance(rec, dict) and rec.get("builtin"):
            raise ValueError(f"'{old}' is built in")
        new = str(new or "").strip()[:32]
        if not new:
            raise ValueError("an empty name")
        if new == old:
            return old
        same_file = _safe(new) == _safe(old)
        if not same_file:
            taken = {_safe(n) for n in store if n != old}
            if _safe(new) in taken:
                new = self.unique_name(kind, new)
        obj = dict(rec, name=new)
        self._write(kind, new, obj, replaces=old)
        if not same_file:
            try:
                os.remove(os.path.join(self.dirs[kind], _safe(old) + ".json"))
            except OSError:
                pass
        store.pop(old, None)
        store[new] = obj
        return new

    def unique_name(self, kind: str, base: str) -> str:
        """`base`, else `base-2`, `base-3`...: the first whose FILE is free.
        Compared folded (`_safe`), as the files are named."""
        taken = {_safe(n) for n in getattr(self, kind)}
        base = base.strip() or "item"
        if _safe(base) not in taken:
            return base
        for i in range(2, 1000):
            n = f"{base}-{i}"
            if _safe(n) not in taken:
                return n
        return f"{base}-{int(time.time())}"

    # ---------------------------------------------------------- defaults ----
    def seed_defaults(self) -> None:
        for name, src, code, note in BUILTIN_AIRFOILS:
            if name not in self.airfoils:
                self.save_airfoil(AirfoilSpec(name, src, code=code, notes=note, builtin=True))
        for f in bundled_dat_files():
            name = f[:-4].lower()
            if name in self.airfoils:
                continue
            try:
                load_dat(os.path.join(DATA_DIR, f))
            except ValueError:
                continue
            self.save_airfoil(AirfoilSpec(name, "dat", file=f, notes=DAT_NOTES.get(name, "UIUC database"),
                                          builtin=True))
        if "fin" not in self.wings:
            self.save_wing(WingSpec("fin", "flank", "naca4412", span=0.78, chord=0.45, taper=1.0,
                                    plate_h=0.0, builtin=True,
                                    notes="the study's clean fin: CL0 0.70, L/D 3.2, S 0.35 (closed form)",
                                    legacy=dict(CL0=0.70, LD=3.2, S=0.35)))
        if "plate" not in self.wings:
            self.save_wing(WingSpec("plate", "flank", "naca6412", span=0.78, chord=0.45, taper=1.0,
                                    plate_h=0.06, builtin=True,
                                    notes="the study's sealed plate: CL0 1.25, L/D 3.2, S 0.35 (closed form)",
                                    legacy=dict(CL0=1.25, LD=3.2, S=0.35)))
        if "flank-e423" not in self.wings:
            w = WingSpec("flank-e423", "flank", "e423", span=0.80, chord=0.45, taper=0.85, plate_h=0.06,
                         builtin=True, notes="a designed flank panel: E423 section, small end plates")
            self.analyse_wing(w)
            self.save_wing(w)
        if "rear-s1223" not in self.wings:
            w = WingSpec("rear-s1223", "top", "s1223", span=1.40, chord=0.30, taper=0.85, plate_h=0.14,
                         builtin=True, notes="a designed rear wing: S1223 inverted, end plates")
            self.analyse_wing(w, ride_h=1.30)
            self.save_wing(w)

    # ------------------------------------------------------------ polars ----
    def polar(self, airfoil: str, re: float, want_xfoil: bool = True) -> Polar:
        """The best polar available NOW for (airfoil, Re-bank); queues XFOIL."""
        a = self.airfoils.get(airfoil)
        if a is None:
            a = self.airfoils.get("naca2412") or AirfoilSpec("naca2412", "naca", "2412")
        reb = re_bank_snap(re)
        key = (a.name, reb)
        pol = self._polars.get(key)
        if pol is not None and (pol.source == "xfoil" or not self.use_xfoil or not want_xfoil):
            return pol
        if self.use_xfoil:
            cp = xfoil.cache_path(self.dirs["polars"], a.fingerprint(), reb, 9.0)
            if os.path.isfile(cp):
                try:
                    with open(cp) as f:
                        pol = Polar.from_json(json.load(f))
                    pol.name = a.name
                    self._polars[key] = pol
                    return pol
                except (OSError, ValueError, KeyError):
                    pass
            if want_xfoil and key not in self._pending:
                self._pending.add(key)
                self._q.put((a, reb))
                self._ensure_worker()
        if pol is None:
            pol = estimate_polar(a.coords(), reb, a.name)
            self._polars[key] = pol
        return self._with_tc(pol, a)

    def _with_tc(self, pol: Polar, a: AirfoilSpec) -> Polar:
        """State the section's thickness ratio on the polar. `estimate_polar`
        measures it on the way past; an XFOIL polar read back from its cache
        does not carry one, and a caller pricing a JUNCTION needs it (see
        `blend`). Measured once per section and remembered."""
        if not pol.tc:
            tc = self._tc.get(a.name)
            if tc is None:
                from . import airfoil as af
                tc = float(af.geometry(a.coords())["tc"])
                self._tc[a.name] = tc
            pol.tc = tc
        return pol

    def has_xfoil_polar(self, airfoil: str, re: float) -> bool:
        p = self._polars.get((airfoil, re_bank_snap(re)))
        return p is not None and p.source == "xfoil"

    def _ensure_worker(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._work, name="carsim-xfoil", daemon=True)
            self._worker.start()

    def _work(self) -> None:
        while True:
            try:
                a, reb = self._q.get(timeout=2.0)
            except queue.Empty:
                self.xfoil_busy = ""
                return
            self.xfoil_busy = f"{a.name} Re {reb:.2g}"
            t0 = time.perf_counter()
            try:
                pol = xfoil.run_polar(a.coords(), reb, a.name, cache_dir=self.dirs["polars"],
                                      fingerprint=a.fingerprint())
            except Exception as exc:                        # never kill the worker
                pol = None
                self.log.append(f"xfoil {a.name}: {exc}")
            self._done.put((a.name, reb, pol, time.perf_counter() - t0))
            self.xfoil_busy = ""

    def poll(self) -> list[tuple[str, float, bool]]:
        """(airfoil, Re, converged) for every XFOIL job that finished since
        the last call; converged ones are now the polar `polar()` returns."""
        out = []
        while True:
            try:
                name, reb, pol, dt = self._done.get_nowait()
            except queue.Empty:
                break
            self._pending.discard((name, reb))
            if pol is not None:
                self._polars[(name, reb)] = pol
                self.log.append(f"xfoil {name} Re {reb:.2g}: {pol.n_rows} rows in {dt:.1f} s")
                out.append((name, reb, True))
            else:
                self.log.append(f"xfoil {name} Re {reb:.2g}: no convergence, estimate kept")
                out.append((name, reb, False))
        return out

    @property
    def xfoil_pending(self) -> int:
        return len(self._pending)

    # ---------------------------------------------------------- analysis ----
    def analyse_wing(self, w: WingSpec, ride_h: float | None = None, V: float | None = None,
                     standoff: float | None = None, wall_side: float = +1.0) -> dict:
        """Re-run the lattice on `w` with the best polar available and store
        the result on the spec. Legacy wings keep their closed form and get
        a display-only aero."""
        V = V_REF[w.role] if V is None else V
        pol = self.polar(w.airfoil, w.reynolds(V))
        #  the END PLATE's own section, if it has been given one. Read at the
        #  plate's own Reynolds number -- its chord is the wing's TIP chord,
        #  not the root -- because a plate a third of the chord long sits a
        #  bank lower and `re_bank_snap` would otherwise quote it at the
        #  wing's number.
        plate_pol = None
        if getattr(w, "plate_airfoil", "") and w.plate_airfoil in self.airfoils:
            plate_pol = self.polar(w.plate_airfoil,
                                   w.reynolds(V) * max(w.taper, 0.05), want_xfoil=False)
        #  the wall's distance is the spec's own design row (`ride_h`); this
        #  used to carry a THIRD default for it (a 0.45 standoff and a 1.30
        #  ride height), pass it explicitly, and so silently overrule whatever
        #  the wing was designed at.
        if w.role == "top" and ride_h is None:
            ride_h = w.ride_h_flown
        if standoff is None:
            standoff = w.ride_h_flown if w.role == "flank" else TOP_PYLON_L
        try:
            aero = analyse(w, pol, V=V, ride_h=ride_h if w.role == "top" else None,
                           standoff=standoff, wall_side=wall_side, plate_polar=plate_pol)
        except ValueError as exc:
            aero = dict(w.aero)
            aero["error"] = str(exc)
            w.aero = aero
            return aero
        aero["airfoil"] = w.airfoil
        aero["plate_airfoil"] = getattr(w, "plate_airfoil", "")
        aero["polar_is_estimate"] = pol.source != "xfoil"
        w.aero = aero
        return aero

    def wing_polar(self, w: WingSpec, V: float | None = None) -> Polar:
        return self.polar(w.airfoil, w.reynolds(V), want_xfoil=False)


# --------------------------------------------------------------------------- #
def self_check(verbose: bool = True) -> bool:
    import shutil
    import tempfile
    ok = True

    def rep(tag, passed, msg=""):
        nonlocal ok
        ok = ok and passed
        if verbose:
            print(f"  [{'ok' if passed else 'FAIL'}] {tag}: {msg}")

    tmp = tempfile.mkdtemp(prefix="carsim_lib_")
    try:
        lib = Library(tmp, use_xfoil=False)
        rep("defaults seeded", len(lib.airfoils) >= 30 and {"fin", "plate", "flank-e423", "rear-s1223"} <= set(lib.wings),
            f"{len(lib.airfoils)} airfoils, {len(lib.wings)} wings")
        w = lib.wings["flank-e423"]
        rep("designed default has aero", "CLa" in w.aero and w.aero["polar_is_estimate"], f"CLa {w.aero.get('CLa', 0):.3f}")
        pol = lib.polar("naca2412", 8.7e5)
        rep("polar snaps to the Re bank", pol.re == 1e6 and pol.source == "estimate", f"Re {pol.re:.2g} {pol.source}")
        w2 = w.copy(name="mine", builtin=False, chord=0.5)
        lib.analyse_wing(w2)
        lib.save_wing(w2)
        lib2 = Library(tmp, use_xfoil=False)
        rep("user wing persists", "mine" in lib2.wings and abs(lib2.wings["mine"].chord - 0.5) < 1e-9, "")
        rep("unique names", lib2.unique_name("wings", "mine") == "mine-2", lib2.unique_name("wings", "mine"))
        try:
            lib2.save_wing(w2.copy(name="Mine"))           # mine.json holds 'mine'
            refused = False
        except ValueError:
            refused = True
        with open(os.path.join(tmp, "builds", "junk.json"), "w") as fh:
            fh.write("[]")
        lib5 = Library(tmp, use_xfoil=False)
        rep("names unique FOLDED, as the files are named; another record's file refused; a non-record skipped",
            lib2.unique_name("wings", "MINE") == "MINE-2" and refused and "Mine" not in lib2.wings
            and lib5.wings["mine"].name == "mine" and not lib5.builds and any("junk" in s for s in lib5.log),
            lib2.unique_name("wings", "MINE"))
        #  task 41: RENAME a build. A plain rename moves the file; one onto a
        #  name another build's file holds is saved beside it (folded, as a
        #  save is); a case-only rename rewrites its own file; a built-in and a
        #  missing name are refused; a reload sees exactly the new names.
        b1 = dict(version=2, name="fast", mirror=True, builtin=False, car="corsa", slots={})
        lib2.save_build(b1)
        lib2.save_build(dict(b1, name="wet"))
        r1 = lib2.rename("builds", "fast", "quick")
        r2 = lib2.rename("builds", "quick", "WET")            # wet.json holds 'wet'
        r3 = lib2.rename("builds", "wet", "Wet")              # its own file
        bad = []
        for args in (("builds", "nope", "x"), ("builds", "Wet", "  "), ("wings", "mine", "x")):
            try:
                lib2.rename(*args)
                bad.append(args)
            except (KeyError, ValueError):
                pass
        lib2.builds["builtin-b"] = dict(b1, name="builtin-b", builtin=True)
        try:
            lib2.rename("builds", "builtin-b", "mine now")
            bad.append("builtin")
        except ValueError:
            pass
        lib2.builds.pop("builtin-b")
        lib6 = Library(tmp, use_xfoil=False)
        files = sorted(f for f in os.listdir(os.path.join(tmp, "builds")) if f != "junk.json")
        rep("Library.rename: moves the file, keeps the folding guard, refuses the rest",
            (r1, r2, r3) == ("quick", "WET-2", "Wet") and not bad
            and sorted(lib6.builds) == ["WET-2", "Wet"] and lib6.builds["WET-2"]["car"] == "corsa"
            and files == ["wet-2.json", "wet.json"], f"{(r1, r2, r3)} files {files} {bad}")
        for n in list(lib2.builds):
            lib2.delete("builds", n)
        lib2.delete("wings", "fin")
        rep("deleting a built-in re-seeds it", "fin" in lib2.wings, "")
        lib2.delete("wings", "mine")
        rep("deleting a user wing removes it", "mine" not in lib2.wings and not os.path.exists(os.path.join(tmp, "wings", "mine.json")), "")
        if xfoil.available():
            lib3 = Library(tmp, use_xfoil=True)
            p0 = lib3.polar("naca2412", 1e6)
            rep("xfoil queued, estimate returned meanwhile", p0.source == "estimate" and lib3.xfoil_pending == 1, "")
            t0 = time.time()
            got = []
            while time.time() - t0 < 120.0 and not got:
                time.sleep(0.2)
                got = lib3.poll()
            rep("xfoil job completes and replaces the estimate",
                bool(got) and got[0][2] and lib3.polar("naca2412", 1e6).source == "xfoil",
                f"{got} in {time.time() - t0:.1f} s")
            lib4 = Library(tmp, use_xfoil=True)
            rep("xfoil polar served from the disk cache", lib4.polar("naca2412", 1e6).source == "xfoil", "")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if verbose:
        print("  ALL PASS" if ok else "  FAILURES ABOVE")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_check() else 1)
