"""drive/race_grid.py -- the race grid's slots, and a bot's own car (task 26).

Plan decision D3: each bot slot is a NAME + a COLOUR + a CHECKPOINT + a
SAVED BUILD. The name is the checkpoint's (`race_bot_label`), the colour is
the slot's (`COLOURS`), the checkpoint is the page's choice, and the build is
the one the bot was BRED in -- a swarm now writes it into the checkpoint's
metadata (`meta['bred']`: the car, whether it was the stock car or yours with
ballast and wing masses, the garage build's JSON, the engine's power scale),
so racing a bot -- or testing it, the page's ranking -- puts it in its own
car by default (the `own` car choice; `own_car`). An older checkpoint that
only names a car and a published wing (`meta['car']`, `meta['wing']`) gets
that stock car with that wing; the built-in driver has no car of its own and
drives yours.

`GRID_MAX` is the largest grid measured to keep the interactive loop real
time (.handoff/26-bots.md): each bot is a second Vehicle stepped at 1 kHz,
~9.5 % of real time on this machine, next to the render's ~5 ms of the
16.7 ms frame. `grid_slot(i)` puts the cars in rows of two, 7 m apart,
behind the line: bot 1 on your left, bot 2 on your right, then a row back.

Pure data and a car builder: no pygame, no drive.ml.
"""
from __future__ import annotations

#: the largest grid (bots) that keeps the interactive loop real time; measured
GRID_MAX = 5
START_OFFSET_M = 2.2               # a bot lines up this far to the side of the user
ROW_M = 7.0                        # and each row this far behind the one before
#: the slots' colours: bot 1 the HUD's accent orange, then blue, violet, rose, lime
#: (none of them the PB ghost's green or the reference ghost's grey)
COLOURS = ((255, 140, 43), (110, 200, 255), (215, 120, 255), (250, 95, 130), (170, 230, 90))
COLOUR_NAMES = ("orange", "blue", "violet", "rose", "lime")
#: the car choices: the bot's own (bred) car, yours, or a stock library car
OWN = "own"


def grid_slot(i: int) -> tuple:
    """(n, s) of grid slot `i` (1-based): left / right of the user, rows of
    two, `ROW_M` apart behind the line (s <= 0, so every gap is read from
    the LINE)."""
    i = max(1, int(i))
    row, side = divmod(i - 1, 2)
    return (START_OFFSET_M if side == 0 else -START_OFFSET_M, -ROW_M * row if row else 0.0)


# --------------------------------------------------------------------------- #
#  task 41: a car too big for a grid box                                       #
# --------------------------------------------------------------------------- #
#: The painted boxes (drive/scenery.py, its GRID_* constants: rows ROW_M
#: apart, +-START_OFFSET_M, each box +-GRID_BOX_HALF wide, a bar across the
#: road just ahead of a Corsa's nose and ticks 1.40 m back from it) hold a
#: car up to 2.0 m wide. Every stock car and the Express fit; the Citaro
#: (2.55 m wide, 11.95 m long) does not: in its slot it would stand 0.35 m
#: into its neighbour and across two rows of paint. So a grid with a car
#: that does not fit is laid out by `grid_layout`:
#:   * a bot that fits takes its slot's box, as before -- unless the USER's
#:     body (plus CLEAR_M) reaches into that box or the space a fitting car
#:     fills in it (its tail may reach ROW_M - bar - CLEAR_M = 4.85 m behind
#:     the slot line), when the bots shift to the next clear boxes in order
#:     (the user in the bus leaves the front row's side boxes and the second
#:     row, s = -7, empty: bot 1 lines up at (2.2, -14));
#:   * a bot that does not fit lines up BEHIND every box the scenery may
#:     paint (GRID_ROWS rows) and every car already placed, on the
#:     centreline, nose BIG_GAP_M behind; a second one behind the first;
#:   * the user always starts at the line (s = 0, n = 0): the lap timer and
#:     the race gap are built on that. A user's bus there covers its own box
#:     and grazes the inner ticks of the front row's side boxes (7.5 cm),
#:     which is why those stay empty.
#: A grid of cars that all fit is `grid_slot(i)` exactly, so every stock
#: race lines up where it always did.
CLEAR_M = 0.10          # m  est  body-to-paint / body-to-body clearance
BIG_GAP_M = 1.0         # m  est  a big car's nose behind the last box / tail


def _box_geom():
    """(half width, bar front past the slot line, tick start past it, rows)
    of a painted box, read from drive/scenery.py so the two cannot differ.
    scenery is numpy + track, no pygame."""
    from . import scenery as _sc
    bar0, bar1 = _sc.GRID_NOSE + 0.10, _sc.GRID_NOSE + 0.30
    return _sc.GRID_BOX_HALF, bar1, bar0 - 1.40, _sc.GRID_ROWS


def footprint(car) -> tuple:
    """(x_front, x_rear, half_w) of a car's body in its CG frame (bodies.body:
    the shell render draws and the wing limits read); a key or a CarSpec."""
    from .bodies import body
    b = body(car)
    return float(b.x_front), float(b.x_rear), float(b.half_w)


def fits_box(car) -> bool:
    """True when `car` stands in a painted box: no wider than the box, and
    its tail clears the next row's bar by CLEAR_M."""
    half, bar1, _t0, _rows = _box_geom()
    xf, xr, w = footprint(car)
    return w <= half and -xr <= ROW_M - bar1 - CLEAR_M


def _overlap(a, b) -> bool:
    """Two (s0, s1, n0, n1) rectangles overlap (open intervals)."""
    return a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]


def grid_layout(user_car, bot_cars) -> dict:
    """{page slot i: (n, s)} for the bots `bot_cars` = [(i, car)] (a car is a
    key or a CarSpec) lining up with the user's `user_car` at the line. See
    the block above; identical to `grid_slot(i)` when every car fits."""
    half, bar1, tick0, rows = _box_geom()
    fit_tail = ROW_M - bar1 - CLEAR_M
    uf, ur, uw = footprint(user_car)
    user = (ur - CLEAR_M, uf + CLEAR_M, -uw - CLEAR_M, uw + CLEAR_M)
    n_boxes = 2 * rows                        # the user's centre box is not a bot's
    free = []
    for j in range(1, n_boxes + 1):
        n_, s_ = grid_slot(j)
        region = (s_ - fit_tail, s_ + bar1, n_ - half, n_ + half)
        if not _overlap(user, region):
            free.append(j)
    out, big = {}, []
    tails = [ur]
    for i, car in bot_cars:
        if fits_box(car):
            j = free[i - 1] if i - 1 < len(free) else n_boxes + i
            out[i] = grid_slot(j)
            tails.append(out[i][1] + footprint(car)[1])
        else:
            big.append((i, car))
    s_nose = min([-ROW_M * (rows - 1) + tick0] + tails) - BIG_GAP_M
    for i, car in big:
        xf, xr, _w = footprint(car)
        out[i] = (0.0, s_nose - xf)
        s_nose = out[i][1] + xr - BIG_GAP_M
    return out


def colour(i: int) -> tuple:
    return COLOURS[(max(1, int(i)) - 1) % len(COLOURS)]


def colour_name(i: int) -> str:
    return COLOUR_NAMES[(max(1, int(i)) - 1) % len(COLOUR_NAMES)]


def bred_meta(car_name: str, stock: bool, build_json, settings, power_scale: float) -> dict:
    """What a swarm writes into its checkpoint about the car it bred in."""
    return dict(car=str(car_name), stock=bool(stock),
                build=(build_json if isinstance(build_json, dict) else None),
                engine=str(getattr(settings, "engine", "")),
                power_scale=float(power_scale),
                ballast=(0.0 if stock else float(getattr(settings, "ballast", 0.0) or 0.0)),
                ballast_at=str(getattr(settings, "ballast_at", "")))


def own_car(meta: dict, base_cfg, lib=None):
    """(CarSpec, VehicleConfig, car name) of the car a bot was BRED in, from
    its checkpoint's meta, on the session's config (assists; the grip scale
    is the car's own). None when the meta names no known car (the built-in
    driver): it then drives the session's car."""
    import cars
    from dataclasses import fields
    from .vehicle import VehicleConfig
    meta = meta if isinstance(meta, dict) else {}
    bred = meta.get("bred") if isinstance(meta.get("bred"), dict) else None
    if bred is None and meta.get("swarm"):
        return None      # a swarm bot saved before task 26 bred in YOUR car ('same'),
        #                  whose build its meta does not record: it drives yours
    name = (bred or {}).get("car") or meta.get("car")
    if name not in cars.CARS:
        return None
    base = cars.get(name)
    pts, aero = [], None
    if bred and isinstance(bred.get("build"), dict) and lib is not None:
        try:
            from .garage import CarBuild
            b = CarBuild.from_json(bred["build"]).clamp(lib)
            #  a pure-legacy build's kwargs carry no devices: the session's
            #  designed wings must not ride along
            aero = {"dev_left": None, "dev_right": None, "top": None, **b.cfg_kwargs(lib)}
            if not bred.get("stock"):
                pts += list(b.mass_points(lib))
        except Exception as exc:           # noqa: BLE001 -- a build this library cannot read
            print(f"race: the bot's build cannot be rebuilt ({type(exc).__name__}: {exc}); "
                  f"its car with its published wing instead")
            aero, pts = None, []
    if aero is None:
        w = str(meta.get("wing") or "off")
        aero = dict(wing=w if w in ("off", "fin", "plate") else "off", x_w=0.97, h_w=0.90,
                    delta_dev_geom=0.0, dev_left=None, dev_right=None, top=None)
    if bred and not bred.get("stock") and float(bred.get("ballast", 0.0) or 0.0) > 0.0:
        at = bred.get("ballast_at") or cars.BALLAST_DEFAULT
        pts.append(cars.ballast_point(base, float(bred["ballast"]), at))
    car = cars.with_masses(base, pts)
    kw = {f.name: getattr(base_cfg, f.name) for f in fields(base_cfg) if f.init}
    kw.update(aero)
    kw["mu_scale"] = float(getattr(car, "mu_scale", 1.0))
    if bred and isinstance(bred.get("power_scale"), (int, float)):
        kw["power_scale"] = float(bred["power_scale"])
    return car, VehicleConfig(**kw), name


# ==================================================================== #
#  SELF-CHECK                                                          #
# ==================================================================== #
def self_check(verbose: bool = True) -> bool:
    import os
    import tempfile
    from types import SimpleNamespace
    ok = True

    def rep(tag, passed, msg=""):
        nonlocal ok
        ok = ok and bool(passed)
        if verbose:
            print(f"  [{'ok' if passed else 'FAIL'}] {tag}" + (f": {msg}" if msg else ""))

    slots = [grid_slot(i) for i in range(1, GRID_MAX + 1)]
    rep("the grid: rows of two, left / right, 7 m apart, every slot its own",
        slots[:3] == [(2.2, 0.0), (-2.2, 0.0), (2.2, -7.0)] and len(set(slots)) == GRID_MAX
        and all(abs(n) <= 6.0 and s <= 0.0 for n, s in slots), str(slots))
    ghost_cols = {(120, 220, 160), (205, 205, 215)}
    rep("a colour per slot, none a time-trial ghost's",
        len({colour(i) for i in range(1, GRID_MAX + 1)}) == GRID_MAX
        and not ghost_cols & set(COLOURS) and colour_name(4) == "rose")
    # task 41: the grid with a car too big for a box. Every grid of cars that
    # fit (the stock three and the Express, as user and as bots) is
    # grid_slot(i) exactly; with the bus anywhere on it, no two bodies come
    # within CLEAR_M and no bus bot touches a box the scenery may paint
    import itertools
    import math
    import cars as _c
    half, bar1, tick0, rows = _box_geom()
    fitting = [k for k in _c.CAR_ORDER if fits_box(k)]
    same = all(grid_layout(u, [(i, k) for i in range(1, GRID_MAX + 1)])
               == {i: grid_slot(i) for i in range(1, GRID_MAX + 1)}
               for u in fitting for k in fitting)
    boxes = [(0.0, 0.0)] + [grid_slot(j) for j in range(1, 2 * rows + 1)]
    boxes = [(s_ + tick0, s_ + bar1, n_ - half, n_ + half) for n_, s_ in boxes]

    def rect(car, n_, s_, pad=0.0):
        xf, xr, w = footprint(car)
        return (s_ + xr - pad, s_ + xf + pad, n_ - w - pad, n_ + w + pad)
    worst, n_grids, bad = -math.inf, 0, []
    for user in ("corsa", "540i", "express", "bus"):
        for n_bots in range(1, GRID_MAX + 1):
            for combo in itertools.product(("corsa", "bus"), repeat=n_bots):
                lay = grid_layout(user, list(enumerate(combo, 1)))
                cars_ = [(user, 0.0, 0.0)] + [(k, *lay[i]) for i, k in enumerate(combo, 1)]
                n_grids += 1
                for (a, na, sa), (b, nb, sb) in itertools.combinations(cars_, 2):
                    if _overlap(rect(a, na, sa, CLEAR_M), rect(b, nb, sb)):
                        bad.append((user, combo, a, b))
                for k, n_, s_ in cars_[1:]:
                    if k == "bus" and any(_overlap(rect(k, n_, s_), bx) for bx in boxes):
                        bad.append((user, combo, "bus on paint"))
                    if k == "bus":
                        worst = max(worst, s_)
    rep("a car too big for a box: clear of every car and of the paint; a grid "
        "that fits is grid_slot exactly", same and not bad,
        f"{len(fitting)} x {len(fitting)} fitting grids identical; {n_grids} grids with "
        f"the bus: {len(bad)} clashes, the nearest bus bot's CG at s {worst:.2f} m, "
        f"user bus -> bot 1 at {grid_layout('bus', [(1, 'corsa')])[1]}" + (
            f"; {bad[:2]}" if bad else ""))
    import cars
    from .vehicle import VehicleConfig
    base = VehicleConfig(abs_on=True, tc_on=False, power_scale=2.0)
    old = own_car(dict(car="540i", wing="plate"), base)
    rep("an old checkpoint: its car, stock, with its published wing, your assists",
        old is not None and old[2] == "540i" and old[0] is cars.get("540i")
        and old[1].wing == "plate" and old[1].abs_on and not old[1].tc_on
        and old[1].mu_scale == float(cars.get("540i").mu_scale) and old[1].power_scale == 2.0)
    rep("the built-in driver (no meta) has no car of its own", own_car({}, base) is None
        and own_car(dict(car="warp"), base) is None)
    from .aero.library import Library
    lib = Library(os.path.join(tempfile.mkdtemp(prefix="carsim_grid_"), "library"),
                  use_xfoil=False)
    build = dict(version=2, name="bred", mirror=True, builtin=False,
                 slots={"left": {"wing": "flank-e423", "x": 0.97, "h": 0.9, "inc_deg": 0.0}})
    st = SimpleNamespace(engine="tuned", ballast=40.0, ballast_at=cars.BALLAST_DEFAULT)
    m_yours = dict(bred=bred_meta("corsa", False, build, st, 1.5))
    m_stock = dict(bred=bred_meta("mx5", True, build, st, 1.5))
    y = own_car(m_yours, base, lib)
    s = own_car(m_stock, base, lib)
    rep("a swarm's bot, bred in YOUR car: its build's wings, their mass and your ballast, "
        "its engine", y is not None and y[1].dev_left is not None and y[1].power_scale == 1.5
        and y[0].m > cars.get("corsa").m + 40.0, f"{y[0].m:.1f} kg" if y else "")
    rep("... bred in a STOCK car: the build's wings, no ballast, no wing mass",
        s is not None and s[2] == "mx5" and s[1].dev_left is not None
        and abs(s[0].m - cars.get("mx5").m) < 1e-9, f"{s[0].m:.1f} kg" if s else "")
    from .garage import CarBuild
    designed = VehicleConfig(**CarBuild.from_json(build).clamp(lib).cfg_kwargs(lib))
    bare = dict(bred=bred_meta("corsa", False, dict(build, slots={"left": {"wing": ""}}), st, 2.0))
    b2 = own_car(bare, designed, lib)
    rep("a bot bred with no wings races with none, whatever your car carries",
        b2 is not None and not b2[1].has_designed() and b2[1].wing == "off"
        and designed.has_designed())
    rep("a swarm bot saved before its car was recorded drives yours (no stock fallback)",
        own_car(dict(swarm="x", car="corsa", wing="off"), base) is None)
    rep("a build this library cannot read falls back to the published wing",
        own_car(dict(bred=dict(car="corsa", build={"slots": 7}), wing="fin"), base, lib)[1].wing
        in ("fin", "off"))
    if verbose:
        print(f"race_grid self-check: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_check() else 1)
