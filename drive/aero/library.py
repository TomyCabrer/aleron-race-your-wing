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
                    if d == "airfoils":
                        a = AirfoilSpec.from_json(rec)
                        store[a.name] = a
                    elif d == "wings":
                        w = WingSpec.from_json(rec)
                        store[w.name] = w
                    else:
                        store[str(rec.get("name", f[:-5]))] = rec
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    self.log.append(f"{d}/{f}: {exc}")

    def _write(self, kind: str, name: str, obj: dict) -> str:
        path = os.path.join(self.dirs[kind], _safe(name) + ".json")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=1)
        os.replace(tmp, path)
        return path

    def save_airfoil(self, a: AirfoilSpec) -> str:
        self.airfoils[a.name] = a
        return self._write("airfoils", a.name, a.to_json())

    def save_wing(self, w: WingSpec) -> str:
        self.wings[w.name] = w
        return self._write("wings", w.name, w.to_json())

    def save_build(self, b: dict) -> str:
        self.builds[b["name"]] = b
        return self._write("builds", b["name"], b)

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

    def unique_name(self, kind: str, base: str) -> str:
        store = getattr(self, kind)
        base = base.strip() or "item"
        if base not in store:
            return base
        for i in range(2, 1000):
            n = f"{base}-{i}"
            if n not in store:
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
                           standoff=standoff, wall_side=wall_side)
        except ValueError as exc:
            aero = dict(w.aero)
            aero["error"] = str(exc)
            w.aero = aero
            return aero
        aero["airfoil"] = w.airfoil
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
