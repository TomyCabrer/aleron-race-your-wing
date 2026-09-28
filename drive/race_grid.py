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
#: car up to 2.0 m wide. Every car in the game fits (task 46: the Corsa,
#: the rally Escort, the 540i and the Express); the Citaro (2.55 m wide,
#: 11.95 m long), retired from the game in task 46 and kept in
#: `cars.RETIRED`, did not: in its slot it would stand 0.35 m into its
#: neighbour and across two rows of paint. The rule stays, for any car that
#: does not fit (the self-check drives it with the retired bus's spec). A
#: grid with a car that does not fit is laid out by `grid_layout`:
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


# --------------------------------------------------------------------------- #
#  task 46: a car picked BY ITS BUILD'S NAME                                   #
# --------------------------------------------------------------------------- #
#: The owner (2026-09-28): "I did a wing for the Renault and when changing car
#: they appear on the other car ... you can only race with cars that have been
#: saved (default cars are also an option). And for bot and racing bot should
#: be the one selected by name same as in race". So every page that picks a
#: car for something other than the session's own car -- the challenge page's
#: Car row, each RACE VS BOT slot's car row, the Deploy-swarm page's Car row
#: -- offers, beside its own choices ('its own', 'same as mine'):
#:   * each STOCK car: the car as `cars.py` has it -- on the race and swarm
#:     pages with NO wings (`stock_car`, `NO_WINGS`); a challenge lends it
#:     its config's stock wings (`challenges.config_build` of no build);
#:   * each SAVED library build, by its name: EXACTLY that build on ITS OWN
#:     car (`build_home`), fitted to that car's slot bands, with its wings
#:     and their mass (`saved_car`) -- never on another car.
#: A saved build's race / swarm choice is 'build:<its library name>'
#: (`saved_choice`); a stock car's is its bare `cars.py` key, as before.
BUILD_PREFIX = "build:"
#: the aero fields a car with no wings drives with (own_car's fallback of a
#: bot with no build and no published wing, made one name)
NO_WINGS = dict(wing="off", dev_left=None, dev_right=None, top=None)


def saved_choice(name: str) -> str:
    """The race / swarm car choice of the saved build `name`."""
    return BUILD_PREFIX + str(name)


def saved_name(choice):
    """The library build name a race / swarm car choice names, else None
    (own, same, a stock car key)."""
    if isinstance(choice, str) and choice.startswith(BUILD_PREFIX):
        return choice[len(BUILD_PREFIX):]
    return None


def build_home(build_json) -> str:
    """The car a SAVED build is driven on when it is picked by name (task
    46): the car it was made for (`records.build_car`, the garage's tag). A
    build saved before builds knew their car (task 41's 'any car') is the
    default car's -- the Corsa, where every build before task 41 was drawn
    and fitted -- so it too is offered on exactly one car. A build made for
    a car taken out of the game (the MX-5, the bus) or any other unknown
    tag is read as a Corsa build (`cars.build_car_key`, task 46: it loads
    as one everywhere) and offered there; '' only when that is not a car."""
    import cars
    from .records import build_car
    c = build_car(build_json) or cars.CAR_DEFAULT
    return c if c in cars.CARS else ""


def saved_builds(builds: dict, car: str, default: str = "") -> list:
    """The library build names picked by name ON `car` (task 46; `builds` is
    the library's {name: json}): the car's `default` build first, then the
    rest by name, as the garage's own lists sort them."""
    names = [n for n, js in (builds or {}).items() if build_home(js) == car]
    return sorted(names, key=lambda n: (n != default, n.lower(), n))


def car_choices(builds: dict, default_of=None) -> list:
    """[(car key, build name or '')] every car in `cars.CAR_ORDER`: its
    stock car ('' = no library build), then its saved builds
    (`saved_builds`, `default_of(car)` first). The order the challenge
    page's Car row and the race / swarm car rows cycle the choices in."""
    import cars
    out = []
    for car in cars.CAR_ORDER:
        out.append((car, ""))
        d = default_of(car) if callable(default_of) else ""
        out += [(car, n) for n in saved_builds(builds, car, d or "")]
    return out


def stock_car(key: str, base_cfg):
    """(CarSpec, VehicleConfig, key) of the STOCK car `key` a race or a
    swarm is given (task 46): the car as `cars.py` has it, NO wings -- the
    session's own wings were made for the session's car and never ride on
    another -- on the session's config (assists, engine) with the car's own
    grip scale."""
    import cars
    from dataclasses import replace
    car = cars.get(key)
    return car, replace(base_cfg, mu_scale=float(car.mu_scale), **NO_WINGS), key


def saved_car(build_json, base_cfg, lib):
    """(CarSpec, VehicleConfig, car key) of a SAVED build picked by name (task
    46): that build on ITS OWN car (`build_home`), fitted to that car's slot
    bands, with its wings and their mass, on the session's config (assists,
    engine) with the car's own grip scale -- no ballast (the session's
    ballast is the session car's). `own_car`'s rebuild of a bot bred in it,
    so a bot bred in a saved build and raced 'its own' is that very car.
    None when the build's car is not in the game."""
    home = build_home(build_json)
    if not home or not isinstance(build_json, dict):
        return None
    return own_car(dict(bred=dict(car=home, stock=False, build=build_json, ballast=0.0)),
                   base_cfg, lib)


def own_label(meta) -> str:
    """What a bot raced in its own car appends to its label: the saved
    build it was bred in by name (task 46), else the car key ('' for none)."""
    meta = meta if isinstance(meta, dict) else {}
    bred = meta.get("bred") if isinstance(meta.get("bred"), dict) else {}
    return str(bred.get("saved") or bred.get("car") or meta.get("car") or "")


def bred_meta(car_name: str, stock: bool, build_json, settings, power_scale: float,
              saved: str = "") -> dict:
    """What a swarm writes into its checkpoint about the car it bred in.
    Task 46: `saved` names the library build it bred in when it was picked
    by name (that build on its own car: its wings and their mass, no
    ballast -- `saved_car`); a stock car's `build_json` is its empty build
    (no wings), so 'its own' rebuilds exactly that."""
    out = dict(car=str(car_name), stock=bool(stock),
               build=(build_json if isinstance(build_json, dict) else None),
               engine=str(getattr(settings, "engine", "")),
               power_scale=float(power_scale),
               ballast=(0.0 if (stock or saved)
                        else float(getattr(settings, "ballast", 0.0) or 0.0)),
               ballast_at=str(getattr(settings, "ballast_at", "")))
    if saved:
        out["saved"] = str(saved)
    return out


def own_car(meta: dict, base_cfg, lib=None, config=None):
    """(CarSpec, VehicleConfig, car name) of the car a bot was BRED in, from
    its checkpoint's meta, on the session's config (assists; the grip scale
    is the car's own). None when the meta names no known car (the built-in
    driver): it then drives the session's car.

    `config` (task 47): a wing mode -- one of the challenges' four configs --
    the race is run in: the bred build carries it (`challenges.config_build`:
    its own wings where it has them, the stock ones lent, the side wings off
    on a top-only mode), so every car on the grid races the same mode. A bot
    with no bred build (a published wing) keeps its wing."""
    import cars
    from dataclasses import fields
    from .records import build_car
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
    if (bred and bred.get("stock") and isinstance(bred.get("build"), dict)
            and build_car(bred["build"]) not in ("", name)):
        #  task 46: a bot bred in a STOCK car before task 46 bred with the
        #  session's wings on it -- another car's build. Those wings never
        #  ride on another car: its own stock car is raced as a stock car
        #  is now, with no wings (an any-car build, or the car's own, stays)
        aero = dict(NO_WINGS, x_w=0.97, h_w=0.90, delta_dev_geom=0.0)
    elif bred and isinstance(bred.get("build"), dict) and lib is not None:
        try:
            from .garage import CarBuild
            if config:
                from .challenges import config_build
                b = config_build(bred["build"], lib, name, config)   # task 47: the race's mode
            else:
                b = CarBuild.from_json(bred["build"]).clamp(lib, name)   # the car it was bred in
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
    # fit (every car in the game, as user and as bots) is grid_slot(i)
    # exactly; with the bus anywhere on it, no two bodies come within CLEAR_M
    # and no bus bot touches a box the scenery may paint. Task 46: no car in
    # the game is too big now; the rule is driven with the retired Citaro's
    # SPEC (`cars.RETIRED`, never offered to a player), which bodies draws
    # and sizes by its name
    import itertools
    import math
    import cars as _c
    half, bar1, tick0, rows = _box_geom()
    fitting = [k for k in _c.CAR_ORDER if fits_box(k)]
    bus_ = _c.RETIRED["bus"]
    rep("every car in the game fits a painted box; the retired bus does not",
        fitting == list(_c.CAR_ORDER) and not fits_box(bus_),
        f"{fitting}; bus {footprint(bus_)}")
    same = all(grid_layout(u, [(i, k) for i in range(1, GRID_MAX + 1)])
               == {i: grid_slot(i) for i in range(1, GRID_MAX + 1)}
               for u in fitting for k in fitting)
    boxes = [(0.0, 0.0)] + [grid_slot(j) for j in range(1, 2 * rows + 1)]
    boxes = [(s_ + tick0, s_ + bar1, n_ - half, n_ + half) for n_, s_ in boxes]

    def rect(car, n_, s_, pad=0.0):
        xf, xr, w = footprint(car)
        return (s_ + xr - pad, s_ + xf + pad, n_ - w - pad, n_ + w + pad)
    worst, n_grids, bad = -math.inf, 0, []
    for user in ("corsa", "540i", "express", bus_):
        for n_bots in range(1, GRID_MAX + 1):
            for combo in itertools.product(("corsa", bus_), repeat=n_bots):
                lay = grid_layout(user, list(enumerate(combo, 1)))
                cars_ = [(user, 0.0, 0.0)] + [(k, *lay[i]) for i, k in enumerate(combo, 1)]
                n_grids += 1
                for (a, na, sa), (b, nb, sb) in itertools.combinations(cars_, 2):
                    if _overlap(rect(a, na, sa, CLEAR_M), rect(b, nb, sb)):
                        bad.append((user, combo, a, b))
                for k, n_, s_ in cars_[1:]:
                    if k is bus_ and any(_overlap(rect(k, n_, s_), bx) for bx in boxes):
                        bad.append((user, combo, "bus on paint"))
                    if k is bus_:
                        worst = max(worst, s_)
    rep("a car too big for a box: clear of every car and of the paint; a grid "
        "that fits is grid_slot exactly", same and not bad,
        f"{len(fitting)} x {len(fitting)} fitting grids identical; {n_grids} grids with "
        f"the bus: {len(bad)} clashes, the nearest bus bot's CG at s {worst:.2f} m, "
        f"user bus -> bot 1 at {grid_layout(bus_, [(1, 'corsa')])[1]}" + (
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
    m_stock = dict(bred=bred_meta("rally", True, build, st, 1.5))
    y = own_car(m_yours, base, lib)
    s = own_car(m_stock, base, lib)
    rep("a swarm's bot, bred in YOUR car: its build's wings, their mass and your ballast, "
        "its engine", y is not None and y[1].dev_left is not None and y[1].power_scale == 1.5
        and y[0].m > cars.get("corsa").m + 40.0, f"{y[0].m:.1f} kg" if y else "")
    rep("... bred in a STOCK car: the build's wings, no ballast, no wing mass",
        s is not None and s[2] == "rally" and s[1].dev_left is not None
        and abs(s[0].m - cars.get("rally").m) < 1e-9, f"{s[0].m:.1f} kg" if s else "")
    #  task 46: a bot bred in a RETIRED car (drive/ml/checkpoints/
    #  mx5_arena_plate.json, an MX-5 bred before the MX-5 left the game)
    #  has no car of its own any more: it drives yours, as the built-in does
    rep("a bot bred in a retired car (the MX-5, the bus) drives yours",
        own_car(dict(bred=bred_meta("mx5", True, build, st, 1.5)), base, lib) is None
        and own_car(dict(car="bus", wing="plate"), base) is None
        and own_car(dict(car="mx5", wing="plate"), base) is None)
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
    #  task 47: a race in a wing mode puts the bred build in that mode
    m_e39 = dict(bred=bred_meta("540i", True, build, st, 1.5))
    top = own_car(m_e39, base, lib, config="top")
    full = own_car(m_e39, base, lib, config="full")
    rep("a race in ONLY TOP: the bot's side wings off, a (stock) top wing on; in FULL "
        "WING its own flanks stay and the top wing is lent",
        top is not None and top[1].dev_left is None and top[1].dev_right is None
        and top[1].top is not None and full is not None and full[1].dev_left is not None
        and full[1].top is not None and top[2] == "540i",
        f"top {top[1].dev_left if top else None} / {top[1].top if top else None}")
    # task 46: a car picked by its build's NAME -- that build on its own car
    # only, a stock car with no wings, and a stock-bred bot never on another
    # car's wings. (The owner's repro: an Express build with a 1.88 m side
    # wing, picked for a Corsa, rode on the Corsa.)
    big = lib.wings["flank-e423"].copy(name="t46-big", builtin=False)
    big.span = 1.88
    lib.save_wing(big)
    exp = dict(version=2, name="my express", mirror=True, builtin=False, car="express",
               slots={"left": {"wing": "t46-big", "x": -1.40, "h": 1.15, "inc_deg": 6.0},
                      "top": {"wing": "rear-s1223", "x": -1.6, "h": 1.9, "inc_deg": 9.0}})
    cor = dict(exp, name="my corsa", car="corsa",
               slots={"left": {"wing": "plate", "x": 0.97, "h": 0.9}})
    anyb = dict(exp, name="old one", car="", slots={"left": {"wing": "fin"}})
    gone = dict(exp, name="gone car", car="warp")
    builds = {b_["name"]: b_ for b_ in (exp, cor, anyb, gone)}
    homes = {n: build_home(b_) for n, b_ in builds.items()}
    ch_ = car_choices(builds, lambda c: "my corsa" if c == "corsa" else "")
    rep("task 46: a saved build is offered on its own car only (an any-car one and one "
        "of a car gone from the game on the Corsa, as they load); the choices go car by "
        "car, stock first, then the car's default, then its builds by name",
        homes == {"my express": "express", "my corsa": "corsa", "old one": cars.CAR_DEFAULT,
                  "gone car": cars.CAR_DEFAULT}
        and ch_[:4] == [("corsa", ""), ("corsa", "my corsa"), ("corsa", "gone car"),
                        ("corsa", "old one")]
        and ("express", "my express") in ch_
        and [c for c, n in ch_ if n == ""] == list(cars.CAR_ORDER)
        and saved_name(saved_choice("my express")) == "my express"
        and saved_name("540i") is None and saved_name(OWN) is None, f"{ch_[:4]} ...")
    #  a stock car on a session whose config carries a designed wing (the
    #  Express's): the stock car has none
    wingy = VehicleConfig(**CarBuild.from_json(exp).clamp(lib, "express").cfg_kwargs(lib))
    sc = stock_car("corsa", wingy)
    se = saved_car(exp, base, lib)
    sx = saved_car(dict(exp, car="warp"), base, lib)
    w_mass = sum(pm.m for pm in CarBuild.from_json(exp).clamp(lib, "express").mass_points(lib))
    rep("task 46: a STOCK car has no wings, whatever the session's car carries; a SAVED "
        "build is that build on its own car, its wings and their mass, your assists",
        wingy.has_designed() and not sc[1].has_designed() and sc[1].wing == "off"
        and sc[0] is cars.get("corsa") and sc[1].mu_scale == float(cars.get("corsa").mu_scale)
        and se is not None and se[2] == "express" and se[1].dev_left is not None
        and se[1].top is not None and se[1].abs_on and se[1].power_scale == 2.0
        and abs(se[0].m - (cars.get("express").m + w_mass)) < 1e-6
        and sx is not None and sx[2] == cars.CAR_DEFAULT,
        f"stock corsa designed {sc[1].has_designed()}; my express {se[0].m:.1f} kg "
        f"(+{w_mass:.1f} kg of wing)" if se else "no saved car")
    m_x = dict(bred=bred_meta("corsa", True, exp, st, 1.0))            # the old way: the
    m_a = dict(bred=bred_meta("corsa", True, anyb, st, 1.0))           # session's wings
    m_s = dict(bred=bred_meta("express", False, exp, st, 1.0, saved="my express"))
    ox, oa, os_ = own_car(m_x, base, lib), own_car(m_a, base, lib), own_car(m_s, base, lib)
    rep("task 46: a bot bred on a stock Corsa with the Express's wings (before task 46) "
        "races its Corsa with none; with an any-car build it keeps it; one bred in a saved "
        "build is that build again, no ballast, and its label names the build",
        not ox[1].has_designed() and ox[1].wing == "off" and ox[2] == "corsa"
        and oa[1].wing == "fin" and m_s["bred"]["ballast"] == 0.0
        and m_s["bred"]["saved"] == "my express" and os_[2] == "express"
        and abs(os_[0].m - se[0].m) < 1e-9 and os_[1].dev_left == se[1].dev_left
        and own_label(m_s) == "my express" and own_label(m_x) == "corsa"
        and own_label({}) == "", f"old stock-bred Corsa designed {ox[1].has_designed()}")
    if verbose:
        print(f"race_grid self-check: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_check() else 1)
