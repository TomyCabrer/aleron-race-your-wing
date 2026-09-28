#!/usr/bin/env python3
"""docs/make_readme_images.py -- the README's gameplay frames, of the OWNER'S
saved builds (2026-09-28: "For the github. use models I have saved").

    python3 docs/make_readme_images.py                  # every frame -> docs/images/
    python3 docs/make_readme_images.py --only halcon    # the frames whose name has 'halcon'
    python3 docs/make_readme_images.py --data ~/x/runs  # another save folder's builds

Each frame is the store screenshots' scene (steam/store/make_store_assets.py,
copied here, not imported: that module chdirs into the repo) with one change:
every car carries a build the player SAVED, not the stock FULL WING set.
A build is applied the way a session applies it (drive.drive._apply_design /
_fitted): `CarBuild.from_json(saved).clamp(lib, car)`, then its
`hud_kwargs(lib)` and `cfg_kwargs(lib)` are the renderer's wing fields
(title.wing_hud's recipe with the saved build in place of the stock one).
The cars replay their circuit's REFERENCE LAP (drive/data/reference_laps.json)
and the wings move as the game's AUTO mode moves them (title.auto_deploy): the
outer side panel and the top wing out in a corner, stowed on a straight.

The saves are never touched: `--data` (default: the repo's runs/) is COPIED
into a temporary folder, which becomes the working directory (the wing
library is the relative runs/library, and loading it may write). In that copy
the builds' display names are the game's own default names (my civetta, my
courier, ...), so no real model name can reach a frame.

Offscreen and single-process: SDL's dummy drivers, no pools. Output:
docs/images/drive_*.jpg (1920 x 1080, JPEG q88) and docs/images/drive_frames.json
(which saved build and wing each frame shows).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(HERE, "images")

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("CARSIM_HEADLESS", "1")
sys.path.insert(0, REPO)

SIZE = (1920, 1080)
FPS = 30
SETTLE_S = 2.5            # s of chase camera before the frame (the eye settles)
JPEG_Q = 88

#: the saved builds, by where they live in the save folder: (file, car key).
#: my_rally is the garage's current design (runs/garage_design.json, the one
#: carrying the newest WingLab flank, flank-winglab-4)
BUILDS = {
    "my_corsa":   ("library/builds/my-corsa.json", "corsa"),
    "my_express": ("library/builds/my-express-autosave.json", "express"),
    "my_rally":   ("garage_design.json", "rally"),
}

#: the frames: circuit, the camera's build, the builds in the train ahead,
#: the moment ('corner' / 'corner2' / 'corner3': the n-th best mid-corner,
#: the outer side panel and the top wing out; 'corner_left[n]' /
#: 'corner_right[n]': the same with that side's panel the one out (a right- /
#: left-hander; the left one stays clear of the minimal HUD's map, bottom
#: right); 'exit': onto a straight, the top wing still out (its hold), the
#: side panels stowed; 'straight': flat out, every
#: wing stowed; or a lap time in s), the HUD ('off' / 'minimal'), and
#: optionally the train's `seed` (the gaps to the cars ahead; default 7)
FRAMES = {
    "drive_civetta_kestrel": dict(track="kestrel", follow="my_corsa",
                                  others=("my_rally",), moment="corner", hud="off"),
    "drive_courier_ashdown": dict(track="ashdown", follow="my_express",
                                  others=("my_corsa",), moment="corner_left", hud="minimal"),
    "drive_halcon_arena":    dict(track="arena", follow="my_rally",
                                  others=("my_express",), moment="corner", hud="off"),
}


# --------------------------------------------------------------------------- #
#  THE SAVE FOLDER, COPIED                                                    #
# --------------------------------------------------------------------------- #
def scratch_saves(data: str) -> str:
    """A temporary working directory holding a COPY of `data`'s library,
    garage design and settings as runs/..., the builds renamed to the game's
    default names. Returns its path (the caller chdirs into it)."""
    root = tempfile.mkdtemp(prefix="readme_images_")
    runs = os.path.join(root, "runs")
    os.makedirs(runs)
    shutil.copytree(os.path.join(data, "library"), os.path.join(runs, "library"))
    for f in ("garage_design.json", "settings.json"):
        if os.path.isfile(os.path.join(data, f)):
            shutil.copy2(os.path.join(data, f), os.path.join(runs, f))
    return root


def shown_name(name: str, car: str) -> str:
    """A build's name as a frame may show it: the model word the player typed
    ('my corsa (autosave)') swapped for the game's default name for the car
    ('my civetta (autosave)'), so no real model name is on screen."""
    from drive.garage import default_build_name
    base = default_build_name(car)                      # 'my civetta'
    low = name.lower()
    for word in ("corsa", "express", "540i", "mx5", "citaro"):
        if low.startswith(f"my {word}"):
            return base + name[len(f"my {word}"):]
    return name


def load_build(key: str):
    """(CarBuild fitted to its car, the saved JSON as renamed, the car key)."""
    from drive.garage import CarBuild, library
    rel, car = BUILDS[key]
    path = os.path.join("runs", rel)
    with open(path) as fh:
        js = json.load(fh)
    js["name"] = shown_name(str(js.get("name", "")), car)
    with open(path, "w") as fh:                         # the COPY only
        json.dump(js, fh, indent=1)
    b = CarBuild.from_json(js).clamp(library(), car)
    return b, js, car


def wing_fields(b) -> dict:
    """title.wing_hud's HudData fields for build `b` (every panel out;
    ReplayCar.at moves them with the corners)."""
    from drive import render as rnd
    from drive.garage import library
    lib = library()
    kw = b.cfg_kwargs(lib)
    out = dict(b.hud_kwargs(lib))
    out.update(h_w=float(kw.get("h_w", rnd.H_W)),
               inc_deg=math.degrees(float(kw.get("delta_dev_geom", 0.0))),
               wing_on=True, wing_deploy=1.0,
               wing_deploy_l=1.0 if out["dev_left"] else 0.0,
               wing_deploy_r=1.0 if out["dev_right"] else 0.0,
               top_deploy=1.0 if out["top_on"] else 0.0)
    return out


def paint_of(car: str):
    """The player's paint for `car` (runs/settings.json), None = factory."""
    from drive.paint import PAINTS
    try:
        with open(os.path.join("runs", "settings.json")) as fh:
            name = (json.load(fh).get("paint") or {}).get(car)
    except (OSError, ValueError):
        name = None
    return PAINTS.get(name) if name else None


# --------------------------------------------------------------------------- #
#  THE FRAME (make_store_assets.render_scene, the saved builds' wings)        #
# --------------------------------------------------------------------------- #
def moment(car, kind) -> float:
    """The lap time (s) of the frame. 'corner': the side panel and the top
    wing furthest out at speed (the n-th best, well apart, for corner2 /
    corner3); 'straight': fast, both wings stowed, not braking; a number: that
    lap time."""
    import numpy as np
    if isinstance(kind, (int, float)):
        return float(kind) % car.T
    if kind == "exit":
        # a corner's exit onto a straight: the top wing still out (its hold),
        # the side panel back in, the car wound up
        ay = np.abs(car.u * car.r)
        ok = (car.top > 0.9) & (car.dep < 0.05) & (ay < 2.5)
        score = np.where(ok, car.u, -1.0)
        return float(car.t[int(np.argmax(score))])
    if kind == "straight":
        ay = np.abs(car.u * car.r)
        calm = (car.dep < 0.02) & (car.top < 0.02) & (ay < 1.0) & (car.ax > -1.0)
        score = np.where(calm, car.u, -1.0)
        return float(car.t[int(np.argmax(score))])
    import re
    m = re.fullmatch(r"corner(_left|_right)?([1-9]?)", str(kind))
    if m is None:
        raise ValueError(f"moment {kind!r}")
    ay = np.abs(car.u * car.r)
    score = ay * (0.3 + car.dep) * (0.6 + 0.4 * car.top) * np.clip(car.u / 25.0, 0.3, 1.0)
    if m.group(1):
        # ReplayCar.side: +1 a left-hander, its RIGHT (outer) panel out
        want = -1 if m.group(1) == "_left" else 1
        score = np.where((car.side == want) & (car.dep > 0.95), score, -1.0)
    n = int(m.group(2) or 1) - 1
    picked = []
    for i in np.argsort(-score).tolist():
        if all(abs(car.t[i] - car.t[j]) > 8.0 for j in picked):
            picked.append(i)
        if len(picked) > n:
            break
    return float(car.t[picked[min(n, len(picked) - 1)]])


def hud_fields(car, t: float) -> None:
    """make_store_assets._hud_fields: speed, g, lap clock and the gear an
    automatic would hold, from the replayed lap."""
    a, sp = car.aux, car.spec
    u = float(car.st.u)
    a.V, a.V_kmh = u, u * 3.6
    a.ay_g, a.ax_g = float(car.st.ay) / 9.81, float(car.st.ax) / 9.81
    a.lap, a.lap_time, a.last_lap, a.best_lap = 2, car.tau(t), car.T, car.T
    try:
        w = u / float(sp.r_roll) * float(sp.finaldrive) * 60.0 / (2 * math.pi)
        revs = [(g + 1, w * float(r)) for g, r in enumerate(sp.gear)]
        ok = [(g, n) for g, n in revs if n <= 0.92 * float(sp.n_cut)]
        g, n = (ok[0] if ok else revs[-1])
        a.gear, a.rpm = int(g), max(float(sp.n_idle), float(n))
    except Exception:             # noqa: BLE001
        pass
    try:
        import cars
        a.car_name = cars.car_name(car.key)
    except Exception:             # noqa: BLE001
        pass


def render_frame(spec: dict, builds: dict):
    """One SIZE frame of `spec`, every car in its saved build. Returns
    (surface, what the camera's car showed: travel of its panels)."""
    from drive import render as rnd
    from drive import title as T
    from drive import track as trk
    follow_key = spec["follow"]
    keys = [follow_key, *spec["others"]]
    by_car = {builds[k][2]: k for k in keys}          # car key -> build key
    av = T.available(spec["track"])
    missing = [builds[k][2] for k in keys if builds[k][2] not in av]
    if missing:
        raise RuntimeError(f"no reference lap on {spec['track']} for {missing}")
    pick = dict(track=spec["track"], cars=[(builds[k][2], *av[builds[k][2]]) for k in keys])
    sc = T.Scene(pick, paint=lambda c: paint_of(c), seed=spec.get("seed", 7),
                 wings=lambda c: wing_fields(builds[by_car[c]][0]))
    f = next(c for c in sc.cars if c.key == builds[follow_key][2])
    sc.order = [sc.cars.index(f)] + [i for i in range(len(sc.cars)) if sc.cars[i] is not f]
    sc.k, sc.follow = 0, f
    f.t0 = (moment(f, spec["moment"]) - SETTLE_S) % f.T
    sc.respace()
    sc.update()
    tr = trk.make_track(sc.track, surfaces=(T.SURFACE != "none"))
    cfg = rnd.ViewConfig(size=SIZE, fps=FPS, mode="chase", hud=spec["hud"],
                         **rnd.look_config("full"))
    cfg.show_vectors = cfg.show_gg = cfg.show_skid = False
    for c in sc.cars:
        rnd.set_car(c.spec)
        rnd.set_paint(c.paint)
        rnd.car_mesh_cached(rnd.car_geom())
    rnd.set_car(f.spec)
    rnd.set_paint(f.paint)
    r = T._SceneRenderer(cfg, tr, headless=True)
    for c in sc.cars:
        rnd.set_car(c.spec)
        r._wing3_key = r._wing3 = None
        r._mesh3(c.aux)
        c.mesh = (r._wing3_key, r._wing3)
    rnd.set_car(f.spec)
    r._wing3_key = r._wing3 = None
    r.scene = sc
    dt = 1.0 / FPS
    for k in range(int(round(SETTLE_S * FPS)) + 1):
        if k:
            sc.advance(dt)
        rnd.set_car(f.spec)
        rnd.set_paint(f.paint)
        r.update_camera(f.st, dt if k else 0.0)
    if spec["hud"] != "off":
        hud_fields(f, sc.t)
    r.draw_frame(f.st, ctl=f.ctl, aux=f.aux)
    shown = dict(lap_time_s=round(f.tau(sc.t), 2), speed_kmh=round(f.st.u * 3.6),
                 side_travel=round(float(f.aux.wing_deploy or 0.0), 2),
                 side=("-" if not f.aux.wing_deploy else "right" if f.aux.wing_side > 0
                       else "left" if f.aux.wing_side < 0 else "-"),
                 top_travel=round(float(f.aux.top_deploy or 0.0), 2),
                 drawn=[d[0] for d in r.drawn])
    return r.screen.copy(), shown


def save_jpg(surf, path: str) -> int:
    import pygame
    from PIL import Image
    raw = pygame.image.tobytes(surf, "RGB")
    Image.frombytes("RGB", surf.get_size(), raw).save(path, quality=JPEG_Q, optimize=True,
                                                     progressive=True)
    return os.path.getsize(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=os.path.join(REPO, "runs"),
                    help="the save folder whose builds are shown (copied, never written)")
    ap.add_argument("--only", default=None, help="only the frames whose name has this")
    a = ap.parse_args(argv)
    data = os.path.abspath(os.path.expanduser(a.data))
    os.makedirs(OUT, exist_ok=True)
    root = scratch_saves(data)
    os.chdir(root)
    try:
        import pygame
        pygame.init()
        pygame.font.init()
        from drive.garage import library
        lib = library()
        builds = {k: load_build(k) for k in BUILDS}
        man_path = os.path.join(OUT, "drive_frames.json")
        try:
            with open(man_path) as fh:
                manifest = json.load(fh)
        except (OSError, ValueError):
            manifest = {}
        manifest = {k: v for k, v in manifest.items() if k in FRAMES}
        for name, spec in FRAMES.items():
            if a.only and a.only not in name:
                continue
            t0 = time.perf_counter()
            surf, shown = render_frame(spec, builds)
            path = os.path.join(OUT, name + ".jpg")
            kb = save_jpg(surf, path) / 1024
            b, js, car = builds[spec["follow"]]
            import cars
            manifest[name] = dict(
                file=f"docs/images/{name}.jpg", track=spec["track"],
                car=cars.car_name(car), build=js["name"],
                build_file=BUILDS[spec["follow"]][0],
                side_wing=b.left.wing, top_wing=b.top.wing,
                ahead=[f"{cars.car_name(builds[k][2])}: {builds[k][1]['name']} "
                       f"(side {builds[k][0].left.wing}, top {builds[k][0].top.wing})"
                       for k in spec["others"]],
                moment=spec["moment"], hud=spec["hud"], **shown)
            print(f"  {name}.jpg  {kb:.0f} KB  {time.perf_counter() - t0:.1f} s  "
                  f"{manifest[name]['car']} / {js['name']}: side {b.left.wing} "
                  f"{shown['side_travel']:.2f} ({shown['side']}), top {b.top.wing} "
                  f"{shown['top_travel']:.2f}, {shown['speed_kmh']} km/h, drawn {shown['drawn']}")
        with open(man_path, "w") as fh:
            json.dump(manifest, fh, indent=1, ensure_ascii=False)
        del lib
    finally:
        os.chdir(REPO)
        shutil.rmtree(root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
