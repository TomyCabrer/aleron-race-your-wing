"""drive/airbrake.py -- the wing MODE (G / TRIANGLE) and the air brake (task 35).

The owner (2026-09-24): "the user should have the ability to deploy 3 wings
at the same time". The physics always could (`Controls.wing_cmd = (left,
right, top)`, task 18's free wings): the G key reached both flanks, with the
top wing left on its own law, behind a mode the HUD never showed. The mode is
now a named cycle, shown on the ACTIVE AERO panel:

  AUTO       the published law: the OUTER flank in a corner; the top wing by
             its own mode (fixed: out; active: out under braking / steering)
  AIR BRAKE  AUTO's outer flank, plus ALL THREE out while you brake (pedal
             >= BRAKE_ON, in until it is under BRAKE_OFF; above V_MIN): both
             flanks' drags add, their side forces cancel, the top wing adds
             downforce
  ALL 3      all three out, all the time
  LEFT       the left panel only (as before)
  RIGHT      the right panel only (as before)

All of it is gated by the arm toggle (F / CIRCLE), as every wing always was.

AIR BRAKE stays on the free path (`wing_cmd`) the whole time its mode is on:
off the brake it commands the outer flank exactly as the published law picks
it (`_law`: the same deadband, the same 0.3 s hold, no side change while a
panel is out), so a brake on / off never switches the physics between its
two deploy states (the published law's and the free path's are separate,
and a switch would make a panel jump). Everything it does is in `wing_cmd`,
which the lap recorder logs: a lap on the air brake replays bit for bit.

What it is worth, measured (task 35's note has the table): the Corsa with
the plate on both flanks and the rear-s1223 top wing, ABS, full brake. Against
NO wing: braking at 100 km/h 0.4 % shorter, at 150 2.3 %; with the wings
already out, from 200 3.1 %. Against the same car in AUTO (whose active top
wing already comes out under braking) the air brake adds the flanks' drag:
0.7 % from 150 (V38). The flank panels make side force and drag, no
downforce. The tyres do nearly all of the stopping.

A car whose two flanks are not a matching pair (one flank, or two different
panels / stations without the mirror) would pull sideways with both out:
its flanks stay on the published law (the top wing still deploys);
LEFT / RIGHT drive them as before.

No pygame, no physics code: `drive.Sim.step_physics` merges `command()`
into the Controls only when nothing else commanded the wings (`wing_cmd is
None`) and only for a mode other than AUTO, so a script, a replay and every
acceptance run are the same bits (V20). It READS the vehicle's steering
deadband and flank deploy fractions; it writes nothing into the car.
"""
from __future__ import annotations

import dataclasses

AUTO, LEFT, RIGHT, ALL, AIR = 0, 1, -1, 2, 3
#: what G / TRIANGLE steps through
CYCLE = (AUTO, AIR, ALL, LEFT, RIGHT)
LABELS = {AUTO: "AUTO", AIR: "AIR BRAKE", ALL: "ALL 3", LEFT: "LEFT", RIGHT: "RIGHT"}
WHAT = {AUTO: "the outer flank in corners, the top wing by its mode",
        AIR: "the outer flank in corners, all three wings out while you brake",
        ALL: "all three wings out, all the time",
        LEFT: "the left panel only", RIGHT: "the right panel only"}
BRAKE_ON = 0.30          # pedal: the air brake goes out at this or more ...
BRAKE_OFF = 0.15         # ... and stays out until the pedal is under this
V_MIN = 5.0              # m/s: under this there is nothing to brake with air
DEV_HOLD = 0.30          # s: the published law's side hold (vehicle.DEV_HOLD) ...
DEV_DEP_LOCKOUT = 0.05   # ... and its no-switch-while-out lockout (vehicle.DEV_DEP_LOCKOUT)


def next_mode(m: int) -> int:
    """G / TRIANGLE: the next mode in CYCLE (an unknown one -> AUTO's next)."""
    i = CYCLE.index(m) if m in CYCLE else 0
    return CYCLE[(i + 1) % len(CYCLE)]


def _same(a, b) -> bool:
    """Two flank panels that mirror each other: every number the physics
    reads (the force and drag laws, the station, the incidence) equal."""
    try:
        da, db = dataclasses.asdict(a), dataclasses.asdict(b)
    except TypeError:
        return a == b
    da.pop("name", None)
    db.pop("name", None)
    return da.keys() == db.keys() and all(
        abs(float(da[k]) - float(db[k])) <= 1e-9 * max(1.0, abs(float(da[k]))) for k in da)


def pair(cfg) -> bool:
    """Both flanks fitted AND a matching pair, so both out cancel sideways
    (forces and yaw moments): a designed build with the same panel at the
    same station on each side (the mirror), or the published law's pair."""
    dl, dr = getattr(cfg, "dev_left", None), getattr(cfg, "dev_right", None)
    if dl is not None or dr is not None:
        return dl is not None and dr is not None and _same(dl, dr)
    return str(getattr(cfg, "wing", "off") or "off") != "off"


def top_fitted(cfg) -> bool:
    return getattr(cfg, "top", None) is not None


class AirBrake:
    """The mode's command: the air brake's pedal hysteresis (`on`) and, for
    AIR BRAKE off the brake, the published law's side latch (`side`, `hold`)
    run on the free path."""

    def __init__(self):
        self.on = False
        self.side = 0
        self.hold = 0.0
        self._mode = AUTO

    def _law(self, delta: float, dt: float, veh):
        """The published law's flank on the free path: sign(steer) past the
        vehicle's deadband, held DEV_HOLD before a side change, and no change
        while the current panel is out (its free deploy fraction)."""
        st, flank = veh.state, veh.cfg.flank_sgn()

        def flank_of(s):                   # +1 left, -1 right: the law's panel
            return -s * flank              # in a turn of sign s (vehicle._aero)

        def dep_of(s):
            if not s:
                return 0.0
            return st.dep_raw_l if flank_of(s) > 0 else st.dep_raw_r
        db = float(veh.dev_deadband)
        want = 1 if delta > db else (-1 if delta < -db else 0)
        if want != 0 and want != self.side:
            self.hold += dt
            if self.hold >= DEV_HOLD and dep_of(self.side) <= DEV_DEP_LOCKOUT:
                self.side, self.hold = want, 0.0
        else:
            self.hold = 0.0
        if want == 0 or want != self.side:
            return (False, False, None)
        return (True, False, None) if flank_of(self.side) > 0 else (False, True, None)

    def command(self, mode: int, ctl, V: float, veh, dt: float):
        """`Controls.wing_cmd` for this step, or None (the published law)."""
        cfg = veh.cfg
        if mode != self._mode:                 # a new mode: the law's latch as it is
            self._mode = mode
            self.side, self.hold = int(getattr(veh.state, "dev_side", 0)), 0.0
        if mode == LEFT:
            self.on = False
            return (True, False, None)
        if mode == RIGHT:
            self.on = False
            return (False, True, None)
        fl = pair(cfg)
        if mode == ALL:
            self.on = False
            return (True, True, True) if fl else (None, None, True)
        if mode != AIR:
            self.on = False
            return None
        b = float(getattr(ctl, "brake", 0.0) or 0.0)
        self.on = (b >= BRAKE_OFF and V > 0.5 * V_MIN) if self.on else (b >= BRAKE_ON and V >= V_MIN)
        if not fl:                             # no matching pair: the flanks keep
            return (None, None, True) if self.on else None   # the published law
        law = self._law(float(getattr(ctl, "delta", 0.0) or 0.0), dt, veh)
        return (True, True, True) if self.on else law


def showing(mode: int, ab: "AirBrake", ctl, cfg) -> bool:
    """The HUD's 'air brake out' light: AIR BRAKE on, the wings armed, and
    something that deploys (a matching pair or a top wing)."""
    return bool(mode == AIR and ab.on and getattr(ctl, "wing_on", False)
                and (pair(cfg) or top_fitted(cfg)))


# ==================================================================== #
#  SELF-CHECK                                                          #
# ==================================================================== #
def self_check(verbose: bool = True) -> bool:
    from types import SimpleNamespace
    ok = True

    def rep(tag, passed, msg=""):
        nonlocal ok
        ok = ok and bool(passed)
        if verbose:
            print(f"  [{'ok' if passed else 'FAIL'}] {tag}" + (f": {msg}" if msg else ""))

    seen, m = [], AUTO
    for _ in range(len(CYCLE)):
        m = next_mode(m)
        seen.append(LABELS[m])
    rep("G cycles AUTO -> AIR BRAKE -> ALL 3 -> LEFT -> RIGHT -> AUTO",
        seen == ["AIR BRAKE", "ALL 3", "LEFT", "RIGHT", "AUTO"] and next_mode(99) == AIR,
        " ".join(seen))
    @dataclasses.dataclass
    class _P:                              # a flank panel's numbers (vehicle.DevAero's shape)
        name: str = "p"
        S: float = 0.35
        CL0: float = 0.7
        x_w: float = 0.97
        h_w: float = 0.9
        inc: float = 0.0

    def car(dl=None, dr=None, wing="off", top=None, deadband=0.03, flank=1.0):
        cfg = SimpleNamespace(dev_left=dl, dev_right=dr, wing=wing, top=top,
                              flank_sgn=lambda: flank)
        return SimpleNamespace(cfg=cfg, dev_deadband=deadband,
                               state=SimpleNamespace(dep_raw_l=0.0, dep_raw_r=0.0, dev_side=0))
    both = car(_P(name="L"), _P(name="R"))
    skew = car(_P(), _P(x_w=0.2))
    other = car(_P(), _P(S=0.5))
    one = car(_P(), None)
    legacy = car(wing="plate")
    bare = car()
    rep("a pair: a matching designed pair (the mirror) or the published law's; one "
        "flank, a different panel or station is not",
        [pair(v.cfg) for v in (both, skew, other, one, legacy, bare)]
        == [True, False, False, False, True, False])

    def ctl(brake=0.0, delta=0.0):
        return SimpleNamespace(brake=brake, delta=delta)
    ab = AirBrake()
    seq = [ab.command(AIR, ctl(b), v, both, 0.001) for b, v in
           ((0.0, 30.0), (0.29, 30.0), (0.35, 30.0), (0.20, 25.0), (0.10, 20.0),
            (0.50, 3.0), (0.50, 10.0), (0.50, 2.0))]
    out = [c == (True, True, True) for c in seq]
    rep("AIR BRAKE: all three at pedal 0.30, held to 0.15, not under V_MIN; always on "
        "the free path (never None) on a pair", out == [False, False, True, True, False,
                                                        False, True, False]
        and all(c is not None for c in seq), str(out))
    # off the brake it is the published law on the free path: the outer flank
    # after DEV_HOLD of one steer sign (flank_sgn +1: a left turn's outer = right)
    ab = AirBrake()
    law = [ab.command(AIR, ctl(0.0, 0.1), 20.0, both, 0.01) for _ in range(40)]
    rep("AIR BRAKE off the brake: the outer flank, after the law's 0.3 s hold",
        law[0] == (False, False, None) and law[-1] == (False, True, None)
        and law.index((False, True, None)) in (29, 30, 31), str(law.index((False, True, None))))
    rep("ALL 3 on a pair; a car without one keeps its flanks on the law, the top out",
        AirBrake().command(ALL, ctl(), 0.0, both, 0.001) == (True, True, True)
        and AirBrake().command(ALL, ctl(), 30.0, one, 0.001) == (None, None, True)
        and AirBrake().command(AIR, ctl(1.0), 30.0, skew, 0.001) == (None, None, True)
        and AirBrake().command(AIR, ctl(0.0), 30.0, one, 0.001) is None)
    rep("AUTO is the published law (None); LEFT / RIGHT as before",
        AirBrake().command(AUTO, ctl(1.0), 30.0, both, 0.001) is None
        and AirBrake().command(LEFT, ctl(), 0.0, both, 0.001) == (True, False, None)
        and AirBrake().command(RIGHT, ctl(), 0.0, both, 0.001) == (False, True, None))
    ab = AirBrake()
    ab.command(AIR, ctl(1.0), 30.0, both, 0.001)
    lit = [showing(AIR, ab, SimpleNamespace(wing_on=w), c.cfg)
           for w, c in ((True, both), (False, both), (True, bare), (True, car(top=object())))]
    rep("the HUD light: armed, and a pair or a top wing to deploy", lit == [True, False, False, True],
        str(lit))
    if verbose:
        print(f"airbrake self-check: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_check() else 1)
