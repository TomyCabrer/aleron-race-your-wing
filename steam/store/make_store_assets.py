#!/usr/bin/env python3
"""steam/store/make_store_assets.py -- every Steam store and library graphic,
rendered by the game itself (Steam prep, 2026-09-28).

    python3 steam/store/make_store_assets.py              # everything -> steam/store/art/
    python3 steam/store/make_store_assets.py --only shots # just the screenshots
    python3 steam/store/make_store_assets.py --only art   # just capsules / library / icons

Rename the game in drive/branding.py, run this again, upload the new files:
the logo and the line under it are drive/title.py's `logo_surfaces` and
`subtitle_surfaces` (the owner's own art, 2026-09-28; a renamed game is set
plainly until it has new art), the backgrounds and the screenshots are the
title screen's live scene -- the reference laps replayed on the dressed
circuits in the chase camera, every car with its full wing set moving in and
out with the corners.

Sizes and formats (partner.steamgames.com/doc/store/assets, checked 2026-09-28):

    store    header capsule   920 x 430   JPG  logo legible, no text but the title
             small capsule    462 x 174   PNG  the logo nearly fills it
             main capsule    1232 x 706   JPG
             vertical capsule 748 x 896   JPG
             page background 1438 x 810   JPG  ambient, quiet (optional)
             screenshots     1920 x 1080  JPG  gameplay only, 5 or more
    library  capsule          600 x 900   PNG
             header           920 x 430   PNG
             hero            3840 x 1240  PNG  NO text; keep the subject in the
                                               centre 1720 x 760 (safe area at full size)
             logo            1280 x 720   PNG  transparent, the logo type only
    icons    community icon   184 x 184   JPG
             client icon       32 x 32    ICO  (+ 16, 48, 256 inside)
             app icon        1024 x 1024  PNG  (packaging/ makes .icns / .ico of its own)

Offscreen and single-process: SDL's dummy drivers, no pools. Needs the repo's
runs/library (the wing library the scene cars' wings come from); without it
the cars are drawn bare. Output: steam/store/art/ plus contact_sheet.png and
manifest.json (what was made, from which scene).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "art")

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("CARSIM_HEADLESS", "1")
sys.path.insert(0, REPO)
os.chdir(REPO)                 # the wing library is the relative runs/library

import numpy as np             # noqa: E402
import pygame                  # noqa: E402

from drive import branding     # noqa: E402
from drive import render as rnd  # noqa: E402
from drive import title as T   # noqa: E402
from drive import track as trk  # noqa: E402

#: the store's navy (the logo's outline), the backdrop of the icons
C_NAVY = (8, 20, 40)

# --------------------------------------------------------------------------- #
#  THE SCENES                                                                 #
# --------------------------------------------------------------------------- #
#: name -> the scene: map, the camera's car, the others in the train ahead,
#: the moment ("corner": the wings furthest out, "fast": the top speed),
#: its paint (None: the factory colour), the camera mode
SCENES = {
    "linden_corsa":   dict(track="linden", follow="corsa", others=("540i", "rally"),
                           moment="corner", paint="cobalt"),
    "kestrel_540i":   dict(track="kestrel", follow="540i", others=("corsa",),
                           moment="corner", paint=None),
    "ashdown_rally":  dict(track="ashdown", follow="rally", others=("express", "corsa"),
                           moment="corner2", paint=None),
    "fairfield_express": dict(track="fairfield", follow="express", others=("540i",),
                              moment="fast", paint=None),
    "arena_corsa":    dict(track="arena", follow="corsa", others=("rally", "540i", "express"),
                           moment="corner3", paint="red"),
    "linden_plan":    dict(track="linden", follow="540i", others=(),
                           moment="corner", paint="yellow", mode="car_up"),
}
#: seconds of chase camera before the frame (the eye settles into the corner)
SETTLE_S = 2.5
FPS = 30


def _lib():
    try:
        from drive.garage import library
        return library()
    except Exception as exc:      # noqa: BLE001 -- then the cars go bare
        print(f"  no wing library ({type(exc).__name__}: {exc}): the cars bare")
        return None


_LIB = None


def _moment(car, kind: str) -> float:
    """The lap time (s) of the frame: 'corner' the wings furthest out at
    speed (the n-th best, well apart, for 'corner2' / 'corner3'), 'fast' the
    top speed."""
    if kind == "fast":
        return float(car.t[int(np.argmax(car.u))])
    ay = np.abs(car.u * car.r)                 # mid-corner, the panel well out
    score = ay * (0.3 + car.dep) * (0.6 + 0.4 * car.top) * np.clip(car.u / 25.0, 0.3, 1.0)
    n = {"corner": 0, "corner2": 1, "corner3": 2}.get(kind, 0)
    order = np.argsort(-score)
    picked = []
    for i in order.tolist():
        if all(abs(car.t[i] - car.t[j]) > 8.0 for j in picked):
            picked.append(i)
        if len(picked) > n:
            break
    return float(car.t[picked[min(n, len(picked) - 1)]])


def _hud_fields(car, t: float) -> None:
    """The HUD's numbers for a replay frame, from the lap and the car's own
    gearing: speed, g, and the gear an automatic would hold (the lowest
    whose revs sit under 92 % of the cut) with its revs."""
    a, sp = car.aux, car.spec
    u = float(car.st.u)
    a.V, a.V_kmh = u, u * 3.6
    a.ay_g, a.ax_g = float(car.st.ay) / 9.81, float(car.st.ax) / 9.81
    # the lap clock: this lap so far, the last and the best = the replayed lap
    a.lap, a.lap_time, a.last_lap, a.best_lap = 2, car.tau(t), car.T, car.T
    try:
        w = u / float(sp.r_roll) * float(sp.finaldrive) * 60.0 / (2 * math.pi)
        revs = [(g + 1, w * float(r)) for g, r in enumerate(sp.gear)]
        ok = [(g, n) for g, n in revs if n <= 0.92 * float(sp.n_cut)]
        g, n = (ok[0] if ok else revs[-1])
        a.gear, a.rpm = int(g), max(float(sp.n_idle), float(n))
    except Exception:             # noqa: BLE001 -- then the replay's gear 3
        pass


def render_scene(name: str, size, graphics: str = "full", hud: str = "off") -> pygame.Surface:
    """One frame of scene `name` at `size`: the game's renderer; `hud` the
    Graphics HUD setting ('off' for the art, 'minimal' as a player sees it)."""
    global _LIB
    spec = SCENES[name]
    if _LIB is None:
        _LIB = _lib() or False
    lib = _LIB or None
    av = T.available(spec["track"])
    keys = [k for k in (spec["follow"], *spec["others"]) if k in av]
    if not keys:
        raise RuntimeError(f"no reference lap on {spec['track']}")
    pick = dict(track=spec["track"], cars=[(k, av[k][0], av[k][1]) for k in keys])
    from drive.paint import PAINTS
    paint_rgb = PAINTS.get(spec.get("paint")) if spec.get("paint") else None
    sc = T.Scene(pick, paint=lambda k: paint_rgb if k == spec["follow"] else None,
                 seed=7, wings=(lambda k: T.wing_hud(k, lib)) if lib is not None else None)
    # the camera's car first, at SETTLE_S before its moment; the train re-laid ahead
    f = next(c for c in sc.cars if c.key == spec["follow"])
    sc.order = [sc.cars.index(f)] + [i for i in range(len(sc.cars)) if sc.cars[i] is not f]
    sc.k, sc.follow = 0, f
    f.t0 = (_moment(f, spec["moment"]) - SETTLE_S) % f.T
    sc.respace()
    sc.update()
    mode = spec.get("mode", "chase")
    tr = trk.make_track(sc.track, surfaces=(T.SURFACE != "none"))
    cfg = rnd.ViewConfig(size=(int(size[0]), int(size[1])), fps=FPS, mode=mode, hud=hud,
                         **rnd.look_config(graphics))
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
    if mode != "chase":
        r.zoom_manual = 2.4      # the plan view closer in: the car reads
    dt = 1.0 / FPS
    n = int(round(SETTLE_S * FPS))
    for k in range(n + 1):
        if k:
            sc.advance(dt)        # < CAM_SWITCH_S: never a cut
        rnd.set_car(f.spec)
        rnd.set_paint(f.paint)
        r.update_camera(f.st, dt if k else 0.0)
    if hud != "off":
        _hud_fields(f, sc.t)
    r.draw_frame(f.st, ctl=f.ctl, aux=f.aux)
    return r.screen.copy()


# --------------------------------------------------------------------------- #
#  THE LOGO                                                                   #
# --------------------------------------------------------------------------- #
def logo(width: int | None = None, height: int | None = None):
    """(logo, shadow) scaled to fit `width` x `height` (either may be None)."""
    lg, _sh = T.logo_surfaces(1.0)
    w0, h0 = lg.get_size()
    k = min((width or 1e9) / w0, (height or 1e9) / h0)
    lg, sh = T.logo_surfaces(max(0.1, k * 0.98))
    if width and lg.get_width() > width or height and lg.get_height() > height:
        k2 = min((width or 1e9) / lg.get_width(), (height or 1e9) / lg.get_height())
        sz = (max(1, int(lg.get_width() * k2)), max(1, int(lg.get_height() * k2)))
        lg, sh = pygame.transform.smoothscale(lg, sz), pygame.transform.smoothscale(sh, sz)
        sh.set_alpha(140)
    return lg, sh


def subtitle(width: int):
    """(subtitle, shadow) `width` px wide: the title's subtitle art, else
    title.SUBTITLE tracked out in the default font (no shadow)."""
    try:
        return T.subtitle_surfaces(width)
    except Exception:             # noqa: BLE001
        pass
    height = max(6, width // 12)
    f = pygame.font.Font(None, max(6, int(height * 1.6)))
    gap = max(1, int(round(height * 0.22)))
    gl = [f.render(c, True, (238, 242, 248)) for c in str(T.SUBTITLE).upper()]
    s = pygame.Surface((sum(g.get_width() for g in gl) + gap * max(0, len(gl) - 1),
                        max(g.get_height() for g in gl)), pygame.SRCALPHA)
    x = 0
    for g in gl:
        s.blit(g, (x, 0), special_flags=pygame.BLEND_RGBA_MAX)
        x += g.get_width() + gap
    return s.subsurface(s.get_bounding_rect()).copy(), None


def lockup(width: int, height: int | None = None, sub: bool = True,
           shadow: bool = True) -> pygame.Surface:
    """The logo (its rule under the letters) and the subtitle lined up with
    that rule, stacked as the title screen draws them, on a transparent
    surface at most `width` x `height`."""
    lg, sh = logo(width, None)
    u = lg.get_width() / 700.0
    x0, x1 = (int(lg.get_width() * f) for f in T.LOGO_RULE)
    st, st_sh = subtitle(x1 - x0) if sub else (None, None)
    if st is not None and st.get_width() > lg.get_width() - x0:
        k = (lg.get_width() - x0) / st.get_width()
        st = pygame.transform.smoothscale(st, (lg.get_width() - x0, max(1, int(st.get_height() * k))))
        st_sh = None
    off = max(2, int(round(6 * u))) if shadow else 0
    gap = int(18 * u)
    H = lg.get_height() + off + ((gap + st.get_height()) if st is not None else 0)
    W = lg.get_width() + off
    s = pygame.Surface((W, H), pygame.SRCALPHA)
    if shadow:
        s.blit(sh, (off, off))
    s.blit(lg, (0, 0))
    if st is not None:
        y = lg.get_height() + gap
        if shadow and st_sh is not None:
            s.blit(st_sh, (x0 + off, y + off))
        s.blit(st, (x0, y))
    if height and s.get_height() > height:
        k = height / s.get_height()
        s = pygame.transform.smoothscale(s, (max(1, int(s.get_width() * k)), height))
    return s


# --------------------------------------------------------------------------- #
#  COMPOSITION                                                                #
# --------------------------------------------------------------------------- #
def cover(src: pygame.Surface, size, focus=(0.5, 0.55), zoom: float = 1.0) -> pygame.Surface:
    """`src` scaled to cover `size` (x `zoom` more) and cropped round `focus`
    (fractions of the scaled source; clamped to its edges)."""
    W, H = size
    k = max(W / src.get_width(), H / src.get_height()) * max(1.0, zoom)
    sw, sh = max(W, int(math.ceil(src.get_width() * k))), max(H, int(math.ceil(src.get_height() * k)))
    s = pygame.transform.smoothscale(src, (sw, sh))
    x = int(min(max(focus[0] * sw - W / 2, 0), sw - W))
    y = int(min(max(focus[1] * sh - H / 2, 0), sh - H))
    return s.subsurface((x, y, W, H)).copy()


def shade(surf: pygame.Surface, side: str, strength: float = 0.75, reach: float = 0.6) -> None:
    """Darken toward one edge ('left', 'top', 'bottom') so the logo reads."""
    W, H = surf.get_size()
    g = pygame.Surface((W, H), pygame.SRCALPHA)
    n = W if side == "left" else H
    lim = max(1, int(n * reach))
    for i in range(lim):
        a = int(255 * strength * (1.0 - i / lim) ** 1.6)
        if side == "left":
            g.fill((*C_NAVY, a), (i, 0, 1, H))
        elif side == "top":
            g.fill((*C_NAVY, a), (0, i, W, 1))
        else:
            g.fill((*C_NAVY, a), (0, H - 1 - i, W, 1))
    surf.blit(g, (0, 0))


def darken(surf: pygame.Surface, amount: float) -> None:
    ov = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
    ov.fill((*C_NAVY, int(255 * amount)))
    surf.blit(ov, (0, 0))


def blur(surf: pygame.Surface, k: int = 8) -> pygame.Surface:
    W, H = surf.get_size()
    s = pygame.transform.smoothscale(surf, (max(1, W // k), max(1, H // k)))
    return pygame.transform.smoothscale(s, (W, H))


def save(surf: pygame.Surface, name: str, made: list, note: str = "") -> None:
    """PNG as is (alpha kept); JPG through Pillow at quality 95 when it is
    there, pygame's own writer otherwise."""
    path = os.path.join(OUT, name)
    if name.lower().endswith(".jpg"):
        try:
            from PIL import Image
            raw = pygame.image.tobytes(surf.convert(24) if surf.get_bitsize() != 24 else surf, "RGB")
            Image.frombytes("RGB", surf.get_size(), raw).save(path, quality=95, subsampling=0)
        except Exception:         # noqa: BLE001
            pygame.image.save(surf, path)
    else:
        pygame.image.save(surf, path)
    made.append(dict(file=name, size=list(surf.get_size()), note=note))
    print(f"  {name:34s} {surf.get_width()}x{surf.get_height()}  {note}")


def icon_square(n: int) -> pygame.Surface:
    """The square icon: the logo on the navy, a thin ice-blue road line."""
    s = pygame.Surface((n, n), pygame.SRCALPHA)
    s.fill((*C_NAVY, 255))
    # a soft ice-blue glow under the name, fading down (opaque rows)
    y0 = int(n * 0.50)
    for i in range(n - y0):
        x = i / (n - y0)                    # up from nothing, then fading down
        k = 0.30 * min(1.0, x / 0.25) ** 2 * (1 - max(0.0, x - 0.25) / 0.75) ** 2
        s.fill(tuple(int(a + (b - a) * k) for a, b in zip(C_NAVY, T.C_LOGO_BOT)) + (255,),
               (0, y0 + i, n, 1))
    lg, sh = logo(int(n * 0.90), int(n * 0.6))
    x, y = (n - lg.get_width()) // 2, int(n * 0.5 - lg.get_height() * 0.55)
    off = max(1, n // 120)
    s.blit(sh, (x + off, y + off))
    s.blit(lg, (x, y))
    return s


# --------------------------------------------------------------------------- #
#  THE ASSETS                                                                 #
# --------------------------------------------------------------------------- #
#: which scene each capsule is cut from, and where its logo goes
CAPSULES = [
    # file, size, scene, focus (x, y), logo box (x, y, w, h fractions), subtitle
    ("header_capsule.jpg", (920, 430), "linden_corsa", (0.35, 0.58), (0.05, 0.10, 0.56, 0.55), True),
    ("small_capsule.png", (462, 174), "linden_corsa", (0.62, 0.55), (0.05, 0.12, 0.90, 0.76), False),
    ("main_capsule.jpg", (1232, 706), "kestrel_540i", (0.38, 0.56), (0.05, 0.08, 0.52, 0.42), True),
    ("vertical_capsule.jpg", (748, 896), "ashdown_rally", (0.50, 0.62), (0.08, 0.06, 0.84, 0.30), True),
    ("library_capsule.png", (600, 900), "ashdown_rally", (0.50, 0.62), (0.08, 0.06, 0.84, 0.28), True),
    ("library_header.png", (920, 430), "linden_corsa", (0.35, 0.58), (0.05, 0.10, 0.56, 0.55), True),
]
#: the screenshots: (scene, HUD) -- two with the HUD a player drives with
SHOTS = [("linden_corsa", "off"), ("kestrel_540i", "minimal"), ("ashdown_rally", "off"),
         ("fairfield_express", "off"), ("arena_corsa", "minimal"), ("linden_plan", "minimal")]


def make_art(made: list, frames: dict) -> None:
    def frame(name, size=(1920, 1080)):
        key = (name, tuple(size), "off")
        if key not in frames:
            t0 = time.perf_counter()
            frames[key] = render_scene(name, size)
            print(f"  (scene {name} {size[0]}x{size[1]}: {time.perf_counter() - t0:.1f} s)")
        return frames[key]

    for fname, size, scene, focus, box, sub in CAPSULES:
        W, H = size
        tall = H > W
        src = frame(scene, (1080, 1920) if tall else (1920, 1080))
        # the landscape ones zoomed a little so the car can sit right of the logo
        bg = cover(src, size, focus, 1.0 if (tall or fname.startswith("small")) else 1.25)
        if fname.startswith("small"):
            darken(bg, 0.45)
        else:
            shade(bg, "top" if tall else "left", 0.80, 0.55 if tall else 0.62)
        bx, by, bw, bh = int(box[0] * W), int(box[1] * H), int(box[2] * W), int(box[3] * H)
        lk = lockup(bw, bh, sub=sub)
        if fname.startswith("small"):     # centred: the logo nearly fills it
            bx, by = (W - lk.get_width()) // 2, (H - lk.get_height()) // 2
        elif tall:
            bx = (W - lk.get_width()) // 2
        bg.blit(lk, (bx, by))
        save(bg, fname, made, f"scene {scene}")

    # the page background: the widest scene, soft and dark (Steam tints it)
    bg = blur(cover(frame("fairfield_express"), (1438, 810)), 6)
    darken(bg, 0.55)
    save(bg, "page_background.jpg", made, "scene fairfield_express, blurred + dark")

    # the library hero: NO text; the subject in the centre (the safe area)
    hero = cover(frame("arena_corsa", (3840, 2160)), (3840, 1240), (0.5, 0.60))
    save(hero, "library_hero.png", made, "scene arena_corsa, no text")

    # the library logo: transparent, the logo type (and the title's second line)
    lk = lockup(1280, 720, sub=True, shadow=False)
    k = min(1280 / lk.get_width(), 720 / lk.get_height())     # 1280 wide or 720 tall
    ll = pygame.transform.smoothscale(lk, (min(1280, round(lk.get_width() * k)),
                                           min(720, round(lk.get_height() * k))))
    save(ll, "library_logo.png", made, "transparent")

    # the icons
    save(icon_square(184), "community_icon.jpg", made)
    big = icon_square(1024)
    save(big, "app_icon_1024.png", made, "for packaging")
    try:
        from PIL import Image
        raw = pygame.image.tobytes(big, "RGBA")
        im = Image.frombytes("RGBA", big.get_size(), raw)
        im.save(os.path.join(OUT, "client_icon.ico"), sizes=[(16, 16), (32, 32), (48, 48), (256, 256)])
        made.append(dict(file="client_icon.ico", size=[32, 32], note="16/32/48/256 inside"))
        print(f"  {'client_icon.ico':34s} 16/32/48/256")
    except Exception as exc:      # noqa: BLE001
        print(f"  client_icon.ico skipped (Pillow: {type(exc).__name__}: {exc})")


def shown_stem(name: str) -> str:
    """A scene name as its file names it: each car KEY in it as the car's
    shown word (drive.ml.swarm.car_word: 'kestrel_540i' -> 'kestrel_n540'),
    so no real model reaches a file name; the scene keys stay."""
    import cars
    from drive.ml.swarm import car_word
    return "_".join(car_word(p) if p in cars.CARS else p for p in name.split("_"))


def make_shots(made: list, frames: dict) -> None:
    for i, (name, hud) in enumerate(SHOTS, 1):
        key = (name, (1920, 1080), hud)
        if key not in frames:
            t0 = time.perf_counter()
            frames[key] = render_scene(name, (1920, 1080), hud=hud)
            print(f"  (scene {name}: {time.perf_counter() - t0:.1f} s)")
        save(frames[key], f"screenshot_{i:02d}_{shown_stem(name)}.jpg", made,
             f"{SCENES[name]['track']}, {SCENES[name]['follow']}, "
             f"{SCENES[name].get('mode', 'chase')}, HUD {hud}")


def contact_sheet(made: list) -> None:
    """Every file as a labelled thumbnail on one sheet (the icons at size)."""
    thumbs = []
    for m in made:
        p = os.path.join(OUT, m["file"])
        if m["file"].endswith(".ico"):
            continue
        s = pygame.image.load(p)
        k = min(360 / s.get_width(), 240 / s.get_height(), 1.0)
        thumbs.append((m["file"], pygame.transform.smoothscale(
            s, (max(1, int(s.get_width() * k)), max(1, int(s.get_height() * k))))))
    cols, cw, ch = 4, 380, 280
    rows = (len(thumbs) + cols - 1) // cols
    sheet = pygame.Surface((cols * cw, rows * ch))
    sheet.fill((40, 44, 52))
    checker = pygame.Surface((cw, ch))
    for yy in range(0, ch, 12):
        for xx in range(0, cw, 12):
            checker.fill((70, 70, 78) if (xx + yy) // 12 % 2 else (90, 90, 98), (xx, yy, 12, 12))
    font = pygame.font.Font(None, 22)
    for i, (name, t) in enumerate(thumbs):
        x, y = (i % cols) * cw + 10, (i // cols) * ch + 10
        if t.get_alpha() is not None or t.get_flags() & pygame.SRCALPHA:
            sheet.blit(checker, (x, y), (0, 0, t.get_width(), t.get_height()))
        sheet.blit(t, (x, y))
        sheet.blit(font.render(name, True, (230, 230, 230)), (x, y + t.get_height() + 4))
    pygame.image.save(sheet, os.path.join(OUT, "contact_sheet.png"))
    print(f"  contact_sheet.png  {sheet.get_width()}x{sheet.get_height()}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", choices=("art", "shots"), default=None)
    a = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    pygame.init()
    pygame.font.init()
    print(f"{branding.GAME_NAME} -- logo {T.LOGO!r}, subtitle {T.SUBTITLE!r} -> {OUT}")
    made, frames = [], {}
    t0 = time.perf_counter()
    if a.only in (None, "shots"):
        make_shots(made, frames)
    if a.only in (None, "art"):
        make_art(made, frames)
    contact_sheet(made)
    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(dict(game=branding.GAME_NAME, logo=T.LOGO, subtitle=T.SUBTITLE,
                       version=branding.VERSION, files=made), fh, indent=1, ensure_ascii=False)
    print(f"done: {len(made)} files in {time.perf_counter() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
