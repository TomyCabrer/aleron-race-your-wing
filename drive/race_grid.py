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
