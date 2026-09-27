"""drive/airbrake.py -- the wing MODE (G / TRIANGLE) and the air brake (tasks 35, 44).

The owner (2026-09-24): "the user should have the ability to deploy 3 wings
at the same time". The physics always could (`Controls.wing_cmd = (left,
right, top)`, task 18's free wings): the G key reached both flanks, with the
top wing left on its own law, behind a mode the HUD never showed. The mode is
now a named cycle, shown on the ACTIVE AERO panel:

  AUTO          the published law: the OUTER flank in a corner; the top wing
                by its own mode (fixed: out; active: out under braking /
                steering)
  AIR BRAKE     AUTO's outer flank, plus ALL THREE out while you brake (pedal
                >= BRAKE_ON, in until it is under BRAKE_OFF; above V_MIN):
                both flanks' drags add, their side forces cancel, the top wing
                adds downforce
  TOP           the top wing alone, on the ACTIVE law whatever its slot's mode
                (out while braking or steering, held TOP_HOLD after); both
                flanks stowed
  TOP FIXED     the top wing alone, always out; both flanks stowed
  TOP FIX+SIDE  the top wing always out; the flanks on the published law
  LEFT          the left panel only (as before)
  RIGHT         the right panel only (as before)

All of it is gated by the arm toggle (F / CIRCLE), as every wing always was.
SHIFT+G steps the cycle backwards (`prev_mode`, task 45): AUTO is one press
back from AIR BRAKE.

The three TOP modes are the owner's (2026-09-25, task 44): "Top wing would
represent the normal wing a car has. Just hide and no use for side wing." A
normal car's wing is its top wing, so the modes that drive it alone stow the
flanks (and the HUD hides them once they are in: `flanks_hidden`), and TOP
FIX+SIDE is the full car with the top wing fixed. G skips the three on a car
with no top wing (`usable`), where they would do nothing but stow the flanks.
ALL 3 (all three out all the time, the old int 2) is gone at the owner's
word -- all three out is AIR BRAKE, only while braking -- and 2 is not
reused, so an old in-process `wing_mode == 2` is simply not restored.

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
None`) and only for a mode other than AUTO (or, after G left a TOP mode,
while the top wing is handed back to the physics: `AirBrake.g_changed`),
so a script, a replay and every acceptance run are the same bits (V20). It
READS the vehicle's steering deadband, flank deploy fractions and top-wing
hold, and TOP's two numbers off drive/vehicle.py (`TOP_HOLD`,
`TOP_BRAKE_ON`); it writes nothing into the car.
"""
from __future__ import annotations

import dataclasses

#: every mode but AUTO is NONZERO: 0 is "no merge" in `drive.Sim.step_physics`
#: (2 was ALL 3, removed in task 44; not reused)
AUTO, LEFT, RIGHT, AIR, TOP, TOP_FIXED, TOP_FIX_SIDE = 0, 1, -1, 3, 4, 5, 6
#: what G / TRIANGLE steps through (`next_mode` skips what the car cannot use)
CYCLE = (AUTO, AIR, TOP, TOP_FIXED, TOP_FIX_SIDE, LEFT, RIGHT)
LABELS = {AUTO: "AUTO", AIR: "AIR BRAKE", TOP: "TOP", TOP_FIXED: "TOP FIXED",
          TOP_FIX_SIDE: "TOP FIX+SIDE", LEFT: "LEFT", RIGHT: "RIGHT"}
#: what the G note says about each, in a player's words: one short line
#: (WHAT_MAX), read at a glance mid-lap (task 45: "the top wing by its own
#: mode" was the garage slot's jargon, and LEFT / RIGHT never said "always")
WHAT = {AUTO: "outer side wing in corners, top as in the garage",
        AIR: "all wings out while you brake",
        TOP: "the top wing only, out when you brake or turn",
        TOP_FIXED: "the top wing only, always out",
        TOP_FIX_SIDE: "top wing always out, outer side wing in corners",
        LEFT: "the left side wing, always out", RIGHT: "the right side wing, always out"}
WHAT_MAX = 48            # characters: the note's line after the mode's label
#: the modes that move the top wing (G offers them only with one fitted)
TOP_MODES = (TOP, TOP_FIXED, TOP_FIX_SIDE)
#: the modes that stow the flanks: hidden and unused (the owner's "just hide")
STOWS_FLANKS = (TOP, TOP_FIXED)
#: the modes that leave the flanks on the published (corner) law
FLANK_LAW = (AUTO, AIR, TOP_FIX_SIDE)
BRAKE_ON = 0.30          # pedal: the air brake goes out at this or more ...
BRAKE_OFF = 0.15         # ... and stays out until the pedal is under this
V_MIN = 5.0              # m/s: under this there is nothing to brake with air
DEV_HOLD = 0.30          # s: the published law's side hold (vehicle.DEV_HOLD) ...
DEV_DEP_LOCKOUT = 0.05   # ... and its no-switch-while-out lockout (vehicle.DEV_DEP_LOCKOUT)


def usable(m: int, cfg) -> bool:
    """A mode G may land on for this car: one in CYCLE, and a TOP mode only
    with a top wing fitted (on a car without one it would only stow the
    flanks). AIR BRAKE / LEFT / RIGHT as before: the refusal for a car with
    no wing at all is the Sim's (`Sim._wings_missing`)."""
    return m in CYCLE and (m not in TOP_MODES or top_fitted(cfg))


def next_mode(m: int, cfg=None) -> int:
    """G / TRIANGLE: the next mode in CYCLE (an unknown one -> AUTO's next);
    with the car's `cfg`, the next one it can use (AUTO always can)."""
    i = CYCLE.index(m) if m in CYCLE else 0
    for k in range(1, len(CYCLE) + 1):
        n = CYCLE[(i + k) % len(CYCLE)]
        if cfg is None or usable(n, cfg):
            return n
    return AUTO


def prev_mode(m: int, cfg=None) -> int:
    """SHIFT+G (task 45): `next_mode` backwards -- the previous mode in CYCLE
    this car can use (an unknown one -> AUTO's previous), so AUTO is one
    press back from AIR BRAKE, not six forward."""
    i = CYCLE.index(m) if m in CYCLE else 0
    for k in range(1, len(CYCLE) + 1):
        n = CYCLE[(i - k) % len(CYCLE)]
        if cfg is None or usable(n, cfg):
            return n
    return AUTO


def flanks_hidden(m: int) -> bool:
    """The HUD and the car's drawing leave the flanks out in this mode (TOP,
    TOP FIXED: stowed and unused, the owner's "just hide")."""
    return m in STOWS_FLANKS


def cruise_top(m: int):
    """What mode `m` commands of the top wing on a straight at speed with no
    brake and no steer -- a stop challenge held at its v0 (`drive.Sim.
    _hold_top`, task 44): True out (TOP FIXED, TOP FIX+SIDE), False in (TOP:
    its law has no trigger there), None the slot's own mode (AUTO, LEFT,
    RIGHT, and AIR BRAKE, whose top wing comes out only on the brake).
    `AirBrake.command`'s top element for that step, with no latch to keep."""
    if m in (TOP_FIXED, TOP_FIX_SIDE):
        return True
    return False if m == TOP else None


_TOP_LAW = None


def _top_law_consts():
    """(TOP_HOLD, TOP_BRAKE_ON), read once off drive/vehicle.py: the TOP
    mode's law IS the physics' active top wing, so its numbers are the
    vehicle's own, never a copy."""
    global _TOP_LAW
    if _TOP_LAW is None:
        from .vehicle import TOP_HOLD, TOP_BRAKE_ON
        _TOP_LAW = (float(TOP_HOLD), float(TOP_BRAKE_ON))
    return _TOP_LAW


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
    """The mode's command: the air brake's pedal hysteresis (`on`), for
    AIR BRAKE off the brake the published law's side latch (`side`, `hold`)
    run on the free path, and for TOP the active top law's hold (`top_hold`).
    `handing`: the top wing is being handed back to the physics after a TOP
    mode (`g_changed`)."""

    def __init__(self):
        self.on = False
        self.side = 0
        self.hold = 0.0
        self.top_hold = 0.0
        self.handing = False
        self._mode = AUTO
        self._keep_top = False

    def g_changed(self, old: int, new: int, veh) -> None:
        """G moved the mode from `old` to `new` (`drive.Sim.handle_event`).
        Two latches would otherwise outlive the change (the task 44 review):

        * the next `command()` takes the car's latches afresh. AUTO never
          calls it (the Sim merges nothing in AUTO), so a challenge run's
          AIR BRAKE -> AUTO -> AIR BRAKE kept the first stint's side latch
          and pedal state: in the opposite corner both flanks were in for
          the 0.3 s side hold before the outer one came out;
        * leaving a mode that commands the top wing (`TOP_MODES`) for one
          where the physics' ACTIVE top law runs again: `vehicle._aero`
          counts its hold down only while nothing commands the top wing, so
          its hold was the one from BEFORE the TOP mode, and the wing came
          out on a straight for up to TOP_HOLD. The hold cannot be written
          into the car on a key press -- a recorded lap replays its
          controls from its start state, and a G press is not a control --
          so the top wing stays on the free path, on the same law with THIS
          object's hold (TOP's own; 0 after the two FIXED modes, whose "out"
          was no trigger), until a step where the law triggers afresh: from
          there the physics' hold is fresh and the two laws are the same
          bits (`handing`, ended in `command()`)."""
        self._mode = None                      # command(): reseed from the car
        self.on = False
        if old in TOP_MODES:
            if old != TOP:
                self.top_hold = 0.0
            stale = True                       # the car's hold froze under old
        else:
            stale = self.handing               # ... and is still stale while handing
        top = getattr(getattr(veh, "cfg", None), "top", None)
        active = top is not None and getattr(top, "mode", "fixed") == "active"
        self.handing = stale and active and new not in TOP_MODES
        self._keep_top = stale                 # TOP takes this hold, not the stale one

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

    @staticmethod
    def _top_want(ctl, veh) -> bool:
        """The active top law's trigger: armed, and the pedal past
        TOP_BRAKE_ON or the steer past the car's deadband."""
        _hold_s, brake_on = _top_law_consts()
        return bool(getattr(ctl, "wing_on", False)) and (
            float(getattr(ctl, "brake", 0.0) or 0.0) > brake_on
            or abs(float(getattr(ctl, "delta", 0.0) or 0.0)) > float(veh.dev_deadband))

    def _top_law(self, ctl, dt: float, veh) -> bool:
        """The physics' ACTIVE top wing (vehicle._aero, `top.mode == 'active'`)
        on the free path, whatever the build's slot says: out while armed and
        the pedal is past TOP_BRAKE_ON or the steer past the deadband, held
        TOP_HOLD after the trigger drops. The same arithmetic, so on an
        active top wing TOP deploys it exactly as AUTO does."""
        hold_s, _brake_on = _top_law_consts()
        want = self._top_want(ctl, veh)
        if want:
            self.top_hold = hold_s
        elif self.top_hold > 0.0:
            self.top_hold = self.top_hold - dt if self.top_hold > dt else 0.0
        return want or self.top_hold > 0.0

    def command(self, mode: int, ctl, V: float, veh, dt: float):
        """`Controls.wing_cmd` for this step, or None (the published law).
        While `handing` (after a TOP mode, `g_changed`) the top wing stays on
        the free path until its law triggers afresh; the Sim calls this in
        AUTO too while it does."""
        cmd = self._command(mode, ctl, V, veh, dt)
        if not self.handing:
            return cmd
        out = self._top_law(ctl, dt, veh)      # the hold keeps time either way
        if cmd is not None and cmd[2] is not None:
            return cmd                         # the mode has the top wing (AIR BRAKE braking)
        if self._top_want(ctl, veh):
            #  the law triggers: the car's hold starts afresh at TOP_HOLD, as
            #  this one just did, and from here the two laws are the same bits
            self.handing = False
            return cmd
        return (None, None, out) if cmd is None else (cmd[0], cmd[1], out)

    def _command(self, mode: int, ctl, V: float, veh, dt: float):
        cfg = veh.cfg
        if mode != self._mode:                 # a new mode: the laws' latches as they are
            self._mode = mode
            self.side, self.hold = int(getattr(veh.state, "dev_side", 0)), 0.0
            if not self._keep_top:             # (the car's own hold, unless it is stale)
                self.top_hold = float(getattr(veh.state, "top_hold", 0.0) or 0.0)
            self._keep_top = False
        if mode == LEFT:
            self.on = False
            return (True, False, None)
        if mode == RIGHT:
            self.on = False
            return (False, True, None)
        if mode == TOP:                        # the flanks stowed; the top wing active
            self.on = False
            return (False, False, self._top_law(ctl, dt, veh))
        if mode == TOP_FIXED:                  # the flanks stowed; the top wing out
            self.on = False
            return (False, False, True)
        if mode == TOP_FIX_SIDE:               # the flanks on the law; the top wing out
            self.on = False
            return (None, None, True)
        fl = pair(cfg)
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
    rep("G cycles AUTO -> AIR BRAKE -> TOP -> TOP FIXED -> TOP FIX+SIDE -> LEFT -> RIGHT "
        "-> AUTO", seen == ["AIR BRAKE", "TOP", "TOP FIXED", "TOP FIX+SIDE", "LEFT", "RIGHT",
                            "AUTO"] and next_mode(99) == AIR, " ".join(seen))
    rep("ALL 3 is gone and 2 is not reused; every mode but AUTO is nonzero (0 = no merge); "
        "every mode has a label and a text", 2 not in CYCLE and "ALL" not in globals()
        and all(k != 0 for k in CYCLE if k != AUTO) and len(set(CYCLE)) == len(CYCLE)
        and set(LABELS) == set(WHAT) == set(CYCLE)
        and set(STOWS_FLANKS) | set(FLANK_LAW) | {LEFT, RIGHT} == set(CYCLE)
        and not set(STOWS_FLANKS) & set(FLANK_LAW), str(sorted(CYCLE)))
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
    tall = car(_P(name="L"), _P(name="R"), top=object())
    top_only = car(top=object())

    def cyc(cfg):
        out, m = [], AUTO
        for _ in range(len(CYCLE)):
            m = next_mode(m, cfg)
            out.append(LABELS[m])
            if m == AUTO:
                break
        return out
    full = ["AIR BRAKE", "TOP", "TOP FIXED", "TOP FIX+SIDE", "LEFT", "RIGHT", "AUTO"]
    no_top = ["AIR BRAKE", "LEFT", "RIGHT", "AUTO"]
    rep("G on a car skips the TOP modes without a top wing (they would only stow the "
        "flanks); AIR BRAKE / LEFT / RIGHT as before",
        cyc(tall.cfg) == full and cyc(top_only.cfg) == full and cyc(both.cfg) == no_top
        and cyc(legacy.cfg) == no_top and next_mode(TOP_FIX_SIDE, both.cfg) == LEFT
        and not usable(TOP, both.cfg) and usable(TOP_FIXED, tall.cfg) and not usable(2, tall.cfg),
        f"with a top {cyc(tall.cfg)}; without {cyc(both.cfg)}")
    cfgs = (None, tall.cfg, both.cfg, legacy.cfg, top_only.cfg)
    mirror = all(prev_mode(next_mode(m_, c_), c_) == m_
                 and next_mode(prev_mode(m_, c_), c_) == m_
                 for c_ in cfgs for m_ in CYCLE if c_ is None or usable(m_, c_))
    rep("SHIFT+G steps back: prev_mode(next_mode(m)) == m for every mode a car can use; "
        "AIR BRAKE -> AUTO in one press; the TOP modes skipped without a top wing",
        mirror and prev_mode(AIR, both.cfg) == AUTO and prev_mode(AUTO, both.cfg) == RIGHT
        and prev_mode(LEFT, both.cfg) == AIR and prev_mode(LEFT, tall.cfg) == TOP_FIX_SIDE
        and prev_mode(99) == RIGHT,
        f"back from AUTO with a top: {LABELS[prev_mode(AUTO, tall.cfg)]}, "
        f"from LEFT without: {LABELS[prev_mode(LEFT, both.cfg)]}")
    long_ = {LABELS[k]: len(v) for k, v in WHAT.items() if len(v) > WHAT_MAX}
    rep(f"every G note's text is one short line (at most {WHAT_MAX} characters)",
        not long_, str(long_) if long_ else
        f"longest {max(len(v) for v in WHAT.values())}")
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
    rep("AIR BRAKE on a car without a matching pair: its flanks keep the law, the top out",
        AirBrake().command(AIR, ctl(1.0), 30.0, skew, 0.001) == (None, None, True)
        and AirBrake().command(AIR, ctl(0.0), 30.0, one, 0.001) is None)
    rep("TOP FIXED: both flanks stowed, the top out; TOP FIX+SIDE: the flanks on the law "
        "(None), the top out -- on any flanks",
        all(AirBrake().command(TOP_FIXED, ctl(b, d), V, c, 0.001) == (False, False, True)
            and AirBrake().command(TOP_FIX_SIDE, ctl(b, d), V, c, 0.001) == (None, None, True)
            for c in (tall, one, legacy, top_only) for b, d, V in ((0.0, 0.0, 30.0),
                                                                   (1.0, 0.2, 0.0))))
    # TOP: the physics' active top law on the free path (vehicle._aero): out
    # while armed with the pedal past 0.05 or the steer past the deadband,
    # held TOP_HOLD (0.8 s) after; the flanks always stowed
    hold_s, brake_on = _top_law_consts()
    ab = AirBrake()

    def armed(brake=0.0, delta=0.0, on=True):
        return SimpleNamespace(brake=brake, delta=delta, wing_on=on)
    dt_ = 0.01
    tops = []
    for k in range(200):                   # 2 s: brake 0.2 s, off, steer 0.2 s, off
        t = k * dt_
        c = armed(brake=0.5 if t < 0.2 else 0.0, delta=0.1 if 1.0 <= t < 1.2 else 0.0)
        cmd = ab.command(TOP, c, 20.0, tall, dt_)
        tops.append(cmd[2])
        if cmd[:2] != (False, False):
            tops.append("flank")
    off_at = [k for k in range(1, len(tops)) if tops[k - 1] is True and tops[k] is False]
    low = [AirBrake().command(TOP, armed(b, d), 20.0, tall, dt_)[2]
           for b, d in ((0.05, 0.0), (0.06, 0.0), (0.0, 0.03), (0.0, 0.031),
                        (0.0, -0.031))]
    rep("TOP: the top wing on the active law (pedal > 0.05 or steer past the deadband, "
        "held 0.8 s), the flanks stowed; the numbers are drive/vehicle.py's",
        "flank" not in tops and tops[0] is True and off_at == [99, 199]
        and low == [False, True, False, True, True] and hold_s == 0.80 and brake_on == 0.05,
        f"out until steps {off_at}; thresholds {low}")
    held = car(top=object())
    held.state.top_hold = 0.5             # AUTO's active top wing, 0.5 s into its hold
    ab = AirBrake()
    kept = ab.command(TOP, armed(), 20.0, held, dt_)[2]
    rep("TOP: nothing out while disarmed (F); entering TOP takes the car's own top hold",
        AirBrake().command(TOP, armed(1.0, 0.5, on=False), 20.0, tall, dt_)[2] is False
        and kept is True and abs(ab.top_hold - (0.5 - dt_)) < 1e-12, f"hold {ab.top_hold}")
    # G changed the mode (the Sim calls g_changed): the next command takes
    # the car's latches afresh. AUTO never calls command(), so a challenge
    # run's AIR BRAKE -> AUTO -> AIR BRAKE kept the first stint's side latch
    # and pedal state (task 44 review)
    ab = AirBrake()
    for _ in range(40):
        ab.command(AIR, ctl(0.0, 0.1), 20.0, both, 0.01)    # a left bend: latched
    ab.command(AIR, ctl(0.5), 20.0, both, 0.01)            # braking: on
    ab.g_changed(AIR, AUTO, both)
    ab.g_changed(AUTO, AIR, both)
    both.state.dev_side = -1                               # AUTO's law went right meanwhile
    again = ab.command(AIR, ctl(0.2, -0.1), 20.0, both, 0.01)
    both.state.dev_side = 0
    rep("G: AIR BRAKE -> AUTO -> AIR BRAKE takes the car's side latch and a fresh pedal "
        "state: the outer flank of the new bend at once, not both in (or all three out)",
        again == (True, False, None), str(again))
    # leaving a TOP mode: an ACTIVE top wing stays on the free path, on the
    # law with this hold (0 after the FIXED modes), until the law triggers
    # afresh -- the car's own hold froze under the TOP mode
    act = car(_P(name="L"), _P(name="R"), top=SimpleNamespace(mode="active"))
    fixd = car(_P(name="L"), _P(name="R"), top=SimpleNamespace(mode="fixed"))
    ab = AirBrake()
    ab.g_changed(TOP_FIX_SIDE, LEFT, act)
    hb = [ab.handing, ab.command(LEFT, armed(), 20.0, act, dt_)]
    ab.g_changed(LEFT, AUTO, act)
    hb += [ab.handing, ab.command(AUTO, armed(), 20.0, act, dt_),
           ab.command(AUTO, armed(brake=0.5), 20.0, act, dt_), ab.handing]
    ab = AirBrake()
    ab.g_changed(TOP_FIXED, AUTO, fixd)
    fixed_slot = (ab.handing, ab.command(AUTO, armed(), 20.0, fixd, dt_))
    ab = AirBrake()
    ab.g_changed(TOP_FIX_SIDE, AIR, act)
    air_hb = [ab.command(AIR, armed(brake=0.5), 20.0, act, dt_), ab.handing,
              ab.command(AIR, armed(), 20.0, act, dt_)]
    ab = AirBrake()
    ab.g_changed(TOP_FIXED, AUTO, act)
    act.state.top_hold = 0.6                               # the car's frozen hold
    ab.g_changed(AUTO, AIR, act)
    ab.g_changed(AIR, TOP, act)
    top_again = ab.command(TOP, armed(), 20.0, act, dt_)
    del act.state.top_hold
    rep("G out of a TOP mode hands an active top wing back: in on its law (hold 0 after "
        "a FIXED mode) through LEFT and AUTO until a trigger, then the published law "
        "(None); AIR BRAKE braking keeps the hold's time; a 'fixed' slot needs none; TOP "
        "again takes this hold, not the car's frozen one",
        hb == [True, (True, False, False), True, (None, None, False), None, False]
        and fixed_slot == (False, None)
        and air_hb == [(True, True, True), True, (False, False, True)]
        and top_again == (False, False, False),
        f"{hb}; fixed slot {fixed_slot}; AIR {air_hb}; TOP {top_again}")
    rep("the flanks are hidden in TOP and TOP FIXED only",
        [flanks_hidden(m_) for m_ in CYCLE] == [m_ in (TOP, TOP_FIXED) for m_ in CYCLE])
    # a stop held at v0 (drive.Sim._hold_top): the top wing as each mode has
    # it cruising, which is command()'s top element with no brake, no steer
    cru = [cruise_top(m_) for m_ in CYCLE]
    cmd0 = [AirBrake().command(m_, armed(), 30.0, tall, 0.001) for m_ in CYCLE]
    rep("cruising (no brake, no steer): the two FIXED modes' top wing out, TOP's in, the "
        "rest the slot's own mode (None) -- command()'s top element, mode by mode",
        cru == [None, None, False, True, True, None, None]
        and cru == [None if c_ is None else c_[2] for c_ in cmd0],
        " ".join(f"{LABELS[m_]}={v_}" for m_, v_ in zip(CYCLE, cru)))
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
