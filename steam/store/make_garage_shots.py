#!/usr/bin/env python3
"""steam/store/make_garage_shots.py -- the README's garage and WingLab
pictures, drawn by the game itself from the OWNER'S saved data.

    python3 steam/store/make_garage_shots.py            # the README's three -> docs/images/
    python3 steam/store/make_garage_shots.py --only all # every shot below
    python3 steam/store/make_garage_shots.py --only garage_car,winglab_geometry
    python3 steam/store/make_garage_shots.py --data DIR # the scratch copy in DIR (kept)

What is drawn (the garage's pages at the game's 1280 x 800; WingLab at
1920 x 1200, its 1280 x 800 layout at 1.5x -- a scale its shell supports):

    garage_car.png          the 3-D garage (CAR page) on the car the garage opens
                            on -- runs/garage_design.json, the last car built --
                            its wings deployed, the camera turned to show the
                            side and top wings
    garage_saved_cars.png   SAVED CARS: every saved build, each drawn carrying
                            its wings
    garage_saved_wings.png  SAVED WINGS: every saved wing with its diagram
    winglab_*.png           WingLab (the AeroBO designer) on the side wing, on the
                            owner's own saved run: the record behind the newest
                            WingLab wing (runs/aerobo/left/records/), re-opened
                            with the section it flew and the law it carries

Isolation: nothing under the repo's runs/ is written. The saved data is COPIED
into a scratch folder, the process runs there (the game's paths are relative
to runs/), and the copy is deleted afterwards. In the copy only, the builds
still called by an old car KEY ('my corsa', 'my express') take the game's own
new-build names ('my civetta', 'my courier'), so no real model name is drawn.

Offscreen and single-process: SDL's dummy drivers, no pools, no XFOIL, no
engine search (the WingLab run is the owner's stored record; the law and the
design report are AeroBO's evaluator on it, as the game derives them).
"""

from __future__ import annotations

import os

if __name__ == "__main__":
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("CARSIM_HEADLESS", "1")

import argparse
import json
import math
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "docs", "images")
SIZE = (1920, 1200)          # WingLab: its 1280 x 800 layout at 1.5x
GARAGE_SIZE = (1280, 800)    # the garage's own pages, as laid out
#: the build names that are an old car KEY -> the game's own new-build name
#: for that car (garage.default_build_name)
RENAME = (("my corsa", "my civetta"), ("my express", "my courier"),
          ("my 540i", "my n540"), ("my rally", "my halcón"))


# --------------------------------------------------------------------------- #
#  the scratch copy of the owner's saved data                                  #
# --------------------------------------------------------------------------- #
def _renamed(name: str) -> str:
    for old, new in RENAME:
        if name == old or name.startswith(old + " "):
            return new + name[len(old):]
    return name


def _slug(name: str) -> str:
    keep = "".join(c if c.isalnum() else "-" for c in name.lower())
    while "--" in keep:
        keep = keep.replace("--", "-")
    return keep.strip("-") or "build"


def make_data(root: str) -> dict:
    """Copy the saved data the garage reads into `root`/runs (read-only on
    the repo's runs/), with the builds renamed. Returns what was copied."""
    src = os.path.join(REPO, "runs")
    dst = os.path.join(root, "runs")
    os.makedirs(dst, exist_ok=True)
    shutil.copytree(os.path.join(src, "library"), os.path.join(dst, "library"))
    for f in ("garage_design.json", "settings.json", "progress.json"):
        if os.path.exists(os.path.join(src, f)):
            shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
    bdir = os.path.join(dst, "library", "builds")
    names = {}
    for f in sorted(os.listdir(bdir)):
        if not f.endswith(".json"):
            continue
        p = os.path.join(bdir, f)
        with open(p) as fh:
            js = json.load(fh)
        new = _renamed(str(js.get("name", "")))
        names[js.get("name")] = new
        js["name"] = new
        os.remove(p)
        with open(os.path.join(bdir, _slug(new) + ".json"), "w") as fh:
            json.dump(js, fh, indent=1)
    for f in ("garage_design.json",):
        p = os.path.join(dst, f)
        if os.path.exists(p):
            with open(p) as fh:
                js = json.load(fh)
            js["name"] = _renamed(str(js.get("name", "")))
            with open(p, "w") as fh:
                json.dump(js, fh, indent=2)
    p = os.path.join(dst, "settings.json")
    if os.path.exists(p):
        with open(p) as fh:
            st = json.load(fh)
        if isinstance(st.get("car_build"), dict):
            st["car_build"] = {k: _renamed(v) for k, v in st["car_build"].items()}
        with open(p, "w") as fh:
            json.dump(st, fh, indent=2)
    #  the WingLab run behind the newest WingLab wing: its record, as the
    #  garage keeps it (runs/aerobo/<slot>/records/wing_<timestamp>.json)
    rec = winglab_record_path(src)
    if rec:
        rel = os.path.relpath(rec, src)
        os.makedirs(os.path.join(dst, os.path.dirname(rel)), exist_ok=True)
        shutil.copy2(rec, os.path.join(dst, rel))
    return {"builds": names, "record": rec}


def newest_winglab_wing(runs: str) -> str | None:
    """The newest saved wing WingLab designed (by file time)."""
    wdir = os.path.join(runs, "library", "wings")
    best = None
    for f in os.listdir(wdir):
        if not f.endswith(".json"):
            continue
        with open(os.path.join(wdir, f)) as fh:
            js = json.load(fh)
        if (js.get("design") or {}).get("engine") != "aerobo":
            continue
        t = os.path.getmtime(os.path.join(wdir, f))
        if best is None or t > best[0]:
            best = (t, js["name"])
    return best[1] if best else None


def winglab_record_path(runs: str, wing: str | None = None) -> str | None:
    """The stored record of WingLab wing `wing`'s run (default the newest)."""
    wing = wing or newest_winglab_wing(runs)
    if not wing:
        return None
    with open(os.path.join(runs, "library", "wings", wing + ".json")) as fh:
        js = json.load(fh)
    d = js.get("design") or {}
    rp = str(d.get("record_path") or "")
    stamp = os.path.splitext(os.path.basename(rp))[0]
    slot = rp.split("/")[2] if rp.count("/") >= 3 else "left"
    p = os.path.join(runs, "aerobo", slot, "records", f"wing_{stamp}.json")
    return p if os.path.exists(p) else None


# --------------------------------------------------------------------------- #
#  the garage, as the game opens it                                            #
# --------------------------------------------------------------------------- #
def system_fonts(tries: int = 6) -> bool:
    """The garage's pages set their text in SysFont's Menlo. On macOS pygame
    reads the font list from XQuartz's fc-list under a 1 s timeout; this
    machine's fc-list takes ~0.8 s, so a busy moment makes pygame fall back to
    its own finder, which fakes a bold face -- a different look from one run
    to the next. The list is read again until fc-list answers, so the
    pictures have the fonts a normal launch finds. True when it did."""
    import warnings
    import pygame.sysfont as sf
    for _ in range(tries):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sf.initsysfonts()
        if not any("fc-list" in str(w.message) for w in caught):
            return True
        sf.Sysfonts.clear()
        sf.Sysalias.clear()
        sf.is_init = False
    sf.initsysfonts()
    return False


def open_garage(size=SIZE, shell_scale: float = 1.5):
    """The garage the drive opens (`drive._painted_garage`), headless, at
    `size`, on the scratch copy: the Settings car, the last car built
    (`CarBuild.load()`, as the drive's launch does), the Paint setting.
    WingLab's shell is drawn at `shell_scale` in a 1920-wide window (its own
    rule gives 1.0 below 2000 px; 1.5 is its 1280 x 800 layout, 1.5x)."""
    import pygame
    from drive import design_shell
    own = getattr(design_shell, "_own_scale_for", design_shell.scale_for)
    design_shell._own_scale_for = own
    design_shell.scale_for = lambda W, H: shell_scale if W >= 1920 else own(W, H)
    from drive import drive as D
    from drive import garage as grg
    from drive.aero.library import Library
    pygame.init()
    system_fonts()
    settings = D.Settings.load()
    lib = Library(os.path.join("runs", "library"), use_xfoil=False)
    design = grg.CarBuild.load()
    g = grg.Garage(tuple(size), design, pad=None, headless=True, lib=lib, car=settings.car,
                   settings=settings)
    g.set_paint(D.paint_rgb(settings, concrete=True))
    g.paint_for = lambda car: D.paint_rgb(settings, car, concrete=True)
    g.aerobo_dir = None                        # no run record written
    g.export_dir = os.path.join("runs", "export")
    g.shell.mouse = (-1, -1)
    return g, settings, lib


def settle(g, frames: int = 30, dt: float = 1.0 / 60.0) -> None:
    for _ in range(frames):
        g.frame(dt)


def save(surf, name: str, out: str) -> str:
    import pygame
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    pygame.image.save(surf, p)
    return p


# --------------------------------------------------------------------------- #
#  the garage shots                                                            #
# --------------------------------------------------------------------------- #
#: the CAR page's camera: round to the front-right quarter, a little high, so
#: the rear (top) wing stands clear of the slot panel on the right and both
#: side wings show (the far one over the roof) -- an orbit and a zoom, what
#: the mouse gives the player
CAR_CAM = dict(sel="right", yaw=-55.0, pitch=30.0, zoom=1.0)


def shot_garage_car(g, out, sel="right", yaw=-55.0, pitch=30.0, zoom=1.0):
    """The CAR page, the wings deployed (SPACE), the side slot selected."""
    g.page = "car"
    g.sel = sel
    g.deploy_cmd = g.deploy = 1.0
    g.hint = ""
    g.cam.reset()
    g.cam.yaw = math.radians(yaw)
    g.cam.pitch = math.radians(pitch)
    g.cam.dist *= zoom
    settle(g, 3)
    return save(g.screen, "garage_car.png", out)


def shot_saved(g, out, which: str):
    """SAVED CARS (G) / SAVED WINGS (L), as the car page opens them."""
    g.page = "car"
    g.sel = "right" if which == "wings" else g.sel
    g.open_saved(which)
    g.hint = ""
    for _ in range(12):                     # the cards' pictures, two a frame
        g.frame(1.0 / 60.0)
    name = "garage_saved_cars.png" if which == "cars" else "garage_saved_wings.png"
    p = save(g.screen, name, out)
    g.close_page()
    return p


# --------------------------------------------------------------------------- #
#  WingLab, on the owner's own run                                             #
# --------------------------------------------------------------------------- #
class Restored:
    """What `restore_winglab` put back, for the report."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def restore_winglab(g, settings, wing_name: str | None = None) -> Restored:
    """Open WingLab on the side slot the way the owner left it after the run
    that made `wing_name` (default: the newest WingLab wing): the mission
    stated on the Settings circuit, dry; stage 2's section the one the run
    flew (the section the wing was saved with, `origin` as the ranking wrote
    it); the Wing type card re-opened on the record's answers
    (`WingModel.reopen`) and the row the owner fixed; the stored record as
    the landed run -- its evaluations from the record's own arrays -- and the
    car's law and AeroBO's design report derived from it by AeroBO's
    evaluator, as `_run_done` / `start_law` / `start_report` do after a run.
    The fitted wing IS the saved one, on the car."""
    import numpy as np
    from drive import aerobo_models as am
    from drive import views_results as vr
    runs = os.path.join(os.getcwd(), "runs")
    wing_name = wing_name or newest_winglab_wing(runs)
    rec_path = winglab_record_path(runs, wing_name)
    if not rec_path:
        raise RuntimeError(f"no stored WingLab record for {wing_name!r}")
    with open(rec_path) as fh:
        rec = json.load(fh)
    rec["record_path"] = os.path.relpath(rec_path)          # as save_record writes it
    spec = g.lib.wings[wing_name]
    key = "left"                                            # the side pair's design slot
    slot = g.build.slot(key)
    if slot.wing != wing_name:
        raise RuntimeError(f"the garage's side slot holds {slot.wing!r}, not {wing_name!r}")
    #  1  the mission, stated (the side wing's own circuit: the Settings one)
    g.open_mission(key)
    mp = g.mission_page
    mp.params.param("track").set(settings.track)
    mp.params.param("surf").set("dry")
    if not g.state_mission():
        raise RuntimeError(f"the mission did not state (page {g.page!r})")
    dp = g.design_page
    s = dp.session
    w = s.wing
    V_rec = float(((rec.get("config") or {}).get("flags") or {}).get("V") or 0.0)
    #  2  the section the run flew (stage 2's decision)
    flags = (rec.get("config") or {}).get("flags") or {}
    sec = flags.get("section_name") or {}
    af = g.lib.airfoils[spec.airfoil]
    coords = np.asarray(af.coords(), float)
    s.af.chosen = dict(source="library", name=str(sec.get("name") or spec.airfoil),
                       tc=am._loop_tc(coords), coords=coords.tolist(),
                       w_upper=list(af.w_upper) if af.w_upper else None,
                       w_lower=list(af.w_lower) if af.w_lower else None,
                       conditions={"re": float(sec.get("re") or 0.0),
                                   "mach": float(sec.get("mach") or 0.0)},
                       library_point=False, origin=str(af.notes or af.origin or ""),
                       n_evals=None)
    s.af.decision = "library"
    s.on_section(s.af)
    #  3  the card and the box the run flew
    w.reopen(rec)
    cfgd = rec.get("config") or {}
    w.fixed = {k: float(v) for k, v in (cfgd.get("pinned") or {}).items()
               if k not in am.bridge.tip_pins(w.choices)}
    w._box_form = None
    #  4  the run that landed
    w.record = rec
    w.records = vr._eval_rows(w, rec)
    w.prior = 0
    w._record_sections = {"main": dict(s.af.chosen), "plate": None}
    n = int(rec.get("n_evals") or 0)
    w.outcome = {"kind": "wing", "state": "done", "k": n, "n": n, "n_prior": 0,
                 "wall": float(rec.get("wall_time_s") or 0.0), "stop_reason": None, "error": None}
    w.more = None
    #  5  the car's law and the design report, from the record (AeroBO's evaluator)
    #  at the speed the record flew (its flags' V): the law the saved wing
    #  carries and the car flies; the mission's lap as the car stands now
    #  may read a hair differently
    law = w.law = am.bridge.derive_law(rec, dict(w.slot_geom(), V=V_rec or float(s.op.V)))
    w.law_outcome = {"kind": "law", "state": "done", "k": 1, "n": 1, "n_prior": 0, "wall": 0.0,
                     "stop_reason": None, "error": None}
    s.results.report = am.bridge.design_report(rec)
    s.results.outcome = {"kind": "report", "state": "done", "k": 1, "n": 1, "n_prior": 0,
                         "wall": 0.0, "stop_reason": None, "error": None}
    #  6  the fitted wing: the saved one, in the slot (put on the car)
    w.spec = spec
    w.slot_updates = {"inc_deg": float(slot.inc_deg)}
    w._section_specs = {}
    w.dirty = False
    w.msg = ""
    dp._slot_wing = slot.wing
    g.notices.drain_toasts()
    g.shell.toasts.items.clear()
    saved = spec.aero or {}
    same_law = all(abs(float(law.get(k, 0.0)) - float(saved.get(k, 0.0))) <= 1e-6 * max(1.0, abs(float(saved.get(k, 0.0))))
                   for k in ("CL0", "CLa", "CL_min", "CL_max", "S", "V_ref", "cd0", "cd2"))
    return Restored(wing=wing_name, record=os.path.relpath(rec_path), V_mission=float(s.op.V),
                    V_record=V_rec, same_law=same_law, n_evals=n,
                    best=rec.get("best_score"), units=rec.get("score_units"),
                    box=(w.bounds_overrides(), cfgd.get("bounds_overrides")),
                    pins=(w.pinned(), cfgd.get("pinned")),
                    objective=w.choices.get("objective"), section=s.af.chosen["name"])


def shot_winglab(g, out, view: str, name: str, scroll: int = 0, mesh=None) -> str:
    """One WingLab view through the shell's own select, drawn three times
    (the first frames settle the layout and clamp the scroll to the
    content's end), the pointer off the window, no toast. `mesh` (azimuth,
    elevation) turns the Geometry view's drawing, as dragging it does."""
    stage = view.split(".", 1)[0]
    if not g.shell.select(stage, view):
        g.design_page.nav.select(view, force=True)
        g.page = "section"
    if mesh is not None:
        g.shell.vstate.setdefault(view, {})["_mesh_cam"] = {"az": float(mesh[0]),
                                                           "el": float(mesh[1])}
    for _ in range(3):
        g.shell.scroll[view] = int(scroll)
        g.shell.mouse = (-1, -1)
        g.notices.drain_toasts()
        g.shell.toasts.items.clear()
        g._draw_page()
    err = g.shell.view_errors.get(view)
    if err:
        raise RuntimeError(f"{view}: {err.strip().splitlines()[-1]}")
    return save(g.screen, name, out)


#: name -> (view, scroll px at the 1.5x scale -- past the end is the end --,
#: the 3-D drawing's (azimuth, elevation) or None)
WINGLAB = {
    "winglab_summary": ("r.summary", 0, None),
    "winglab_geometry": ("r.geometry", 0, None),
    "winglab_wing_3d": ("r.geometry", 100000, (120.0, 25.0)),
    "winglab_loading": ("r.loading", 0, None),
}
GARAGE = ("garage_car", "garage_saved_cars", "garage_saved_wings")
#: the shots the README shows (a bare run makes these; --only all makes every one)
README_SHOTS = "garage_car,garage_saved_cars,winglab_summary"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--only", default=README_SHOTS,
                    help="comma list of shot names, or 'all' (default: the README's)")
    ap.add_argument("--keep", action="store_true", help="keep the scratch data folder")
    ap.add_argument("--data", default="", help="use this scratch folder (made if missing)")
    a = ap.parse_args(argv)
    out = os.path.abspath(a.out)
    root = os.path.abspath(a.data) if a.data else tempfile.mkdtemp(prefix="garage_shots_")
    if not os.path.isdir(os.path.join(root, "runs")):
        info = make_data(root)
        print(f"scratch data: {root}\n  builds renamed {info['builds']}\n  record {info['record']}")
    sys.path.insert(0, REPO)
    os.chdir(root)
    only = {t.strip() for t in a.only.split(",") if t.strip()}
    if "all" in only:
        only = set()
    want = lambda n: not only or n in only  # noqa: E731
    made = []
    try:
        if any(want(n) for n in GARAGE):
            #  the garage's own pages at the game's 1280 x 800: at 1920 x 1200
            #  their text scales 1.5x but not every gap does (the list pages'
            #  key bar is laid out in fixed px, the car page's help lines lose
            #  their last words), so they are drawn at the size they were laid
            #  out for
            g, _st, _lib = open_garage(GARAGE_SIZE)
            settle(g, 5)
            if want("garage_car"):
                made.append(shot_garage_car(g, out, **CAR_CAM))
            if want("garage_saved_cars"):
                made.append(shot_saved(g, out, "cars"))
            if want("garage_saved_wings"):
                made.append(shot_saved(g, out, "wings"))
        if any(want(n) for n in WINGLAB):
            g, st, _lib = open_garage(SIZE)
            settle(g, 3)
            r = restore_winglab(g, st)
            print(f"WingLab: {r.wing} from {r.record}: {r.n_evals} evaluations, best {r.best} "
                  f"{r.units}; objective {r.objective}; section {r.section}\n"
                  f"  V mission {r.V_mission:.4f} / record {r.V_record:.4f}; law = saved wing's: "
                  f"{r.same_law}\n  box now {r.box[0]}\n  box run {r.box[1]}\n"
                  f"  pins now {r.pins[0]} / run {r.pins[1]}")
            for n, (view, px, mesh) in WINGLAB.items():
                if want(n):
                    made.append(shot_winglab(g, out, view, n + ".png", px, mesh))
        for p in made:
            print("wrote", p)
    finally:
        os.chdir(REPO)
        if not a.keep and not a.data:
            shutil.rmtree(root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
