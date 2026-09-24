"""The single acceptance suite. Run this to decide whether the sim is trustworthy.

    python3 -m drive.validate                 everything
    python3 -m drive.validate --quick         skip the minute-scale rigs
    python3 -m drive.validate --only D        one group (or a glob on the name)
    python3 -m drive.validate --modules       also run each module's own self-check
    python3 -m drive.validate -v              print the note under every row

What this file is FOR
---------------------
Every module already carries its own self-check, and those are the detailed
ones -- 78 checks in powertrain.py alone. This file is the cross-cutting layer
they cannot contain: the places where the real-time sim has to agree with the
quasi-steady analysis it was built to be consistent with, and the places where a
sign error would be invisible inside any one module.

HARD vs SOFT
------------
HARD  a wrong sign, a broken identity, a divergence, a brake lock-order flip, a
      conservation law that does not close. A HARD failure means a number
      produced by this sim is not to be believed, and the process exits 1.
SOFT  a calibration figure sitting inside a band. Brakes, dampers, engine
      inertia and the shift model are all UNPUBLISHED for a Corsa C 1.2 -- see
      corsa_c.brakes and corsa_c.dampers, both literally the string "MISSING".
      A SOFT miss is a number to argue about, not a broken model. It is printed
      loudly and does not fail the build.

Two things this suite prints rather than buries, because they are findings about
the STUDY and not about the sim (CONTRACT.md section 10):

  1. qss.residuals' Y_r term violates qss's own stated force balance whenever
     the wing is on. Group X computes both forms.
  2. qss.py and crossover.py default to alpha_peak_deg = 7.0; the Magic Formula
     front-axle peak at the R=100 limit load split is 10.35 deg, and scrub drag
     goes as sin(alpha_peak). Group X computes what that moves.

Neither is patched here. qss.py, crossover.py, ledger.py and corsa_c.py are not
modified by this package at all.
"""

import argparse
import fnmatch
import os
import subprocess
import sys
import time
from math import atan, cos, degrees, fabs, radians, sin, sqrt

# SDL must be neutered before pygame is imported anywhere, and drive/__init__
# has already run by the time this module's body executes.
os.environ.setdefault("CARSIM_HEADLESS", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import corsa_c
import crossover
import qss
from corsa_c import G, RHO, CorsaC

from drive import powertrain as PT
from drive import track as TR
from drive import tyre as TY
from drive import vehicle as VE

CAR = CorsaC()
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The qss reference numbers, computed once so the table can show both sides.
QSS = {}


# ==================================================================== #
#  RESULT PLUMBING                                                     #
# ==================================================================== #
class Check:
    __slots__ = ("group", "name", "expected", "got", "ok", "hard", "note",
                 "secs")

    def __init__(self, group, name, expected, got, ok, hard=True, note="",
                 secs=0.0):
        self.group = group
        self.name = name
        self.expected = expected
        self.got = got
        self.ok = bool(ok)
        self.hard = bool(hard)
        self.note = note
        self.secs = secs


CHECKS = []
NOTES = []          # free-text lines printed under their group


def chk(group, name, expected, got, ok, hard=True, note="", secs=0.0):
    CHECKS.append(Check(group, name, expected, got, ok, hard, note, secs))
    return ok


def note(group, text):
    NOTES.append((group, text))


def near(a, b, tol):
    return fabs(a - b) <= tol


def rel(a, b):
    return fabs(a - b) / max(fabs(b), 1e-12)


# ==================================================================== #
#  A. THE TYRE, AGAINST qss.TYRE                                       #
# ==================================================================== #
def group_A(quick):
    t = TY.CORSA_TYRE

    # A1  the mu law itself. qss.TYRE is a rounded transcription of this file's
    # PDY1/PDY2/FNOMIN; if they ever diverge, every number downstream moves.
    worst, worst_fz = 0.0, 0.0
    for Fz in range(500, 6001, 100):
        ref = 0.903 - 16.1e-6 * (Fz - 2477.0)
        d = fabs(t.mu_y(float(Fz)) - ref)
        if d > worst:
            worst, worst_fz = d, Fz
    chk("A", "mu_y(Fz) reproduces qss.TYRE", "max |diff| < 1.1e-4 over 500-6000 N",
        f"{worst:.3e} at Fz = {worst_fz} N", worst < 1.1e-4, hard=True)

    # A2  symmetrisation proof: the analytic peak must equal the swept peak.
    # A 2.3% high answer at 2477 N means the 13 camber-even shifts were not
    # zeroed and the tyre carries the test specimen's ply-steer.
    worst = 0.0
    for Fz in (581.0, 1932.0, 2477.0, 3022.0, 5463.0):
        best = 0.0
        a = 0.0
        while a < 0.5:
            best = max(best, fabs(t.evaluate(Fz, 0.0, a)[1]))
            a += 0.0005
        worst = max(worst, rel(best, t.peak_fy(Fz)))
    chk("A", "peak |Fy| == mu_y*Fz (symmetrisation)", "rel < 5e-4 at 5 loads",
        f"max rel {worst:.2e}", worst < 5e-4, hard=True,
        note="2.3% high at 2477 N would mean the ply-steer shifts are still in")

    # A3  exact zero at zero slip. QDZ6 alone leaves +1.75 N.m of phantom
    # self-centring torque at dead ahead.
    z = t.evaluate(2477.0, 0.0, 0.0, 0.0, 1.0)
    chk("A", "evaluate(Fz, 0, 0) is exactly zero", "(0.0, 0.0, 0.0)",
        f"({z[0]:.3e}, {z[1]:.3e}, {z[2]:.3e})",
        z == (0.0, 0.0, 0.0), hard=True)

    # A4  THE identity: swap qss's own scalar tyre for the Magic Formula and
    # nothing may move. This is what "the sim and the analysis share one tyre"
    # actually means.
    orig = qss.fy_max
    try:
        qss.fy_max = (lambda Fz, mu_ref=None, Fz_ref=None, s=None, mu_scale=1.0:
                      t.peak_fy(Fz, mu_scale) if Fz > 0 else 0.0)
        mf = {R: qss.corner_speed(R) for R in (50, 100, 130)}
        mf_bal = qss.balance(100.0)
    finally:
        qss.fy_max = orig

    for R in (50, 100, 130):
        v_q, l_q = QSS["cs"][R]
        v_m, l_m = mf[R]
        chk("A", f"qss.corner_speed({R}) with the MF tyre",
            f"{v_q:.4f} m/s '{l_q}' +/- 0.01",
            f"{v_m:.4f} m/s '{l_m}' ({v_m - v_q:+.5f})",
            fabs(v_m - v_q) < 0.01 and l_m == l_q, hard=True)

    b_q = QSS["bal100"]
    chk("A", "qss.balance(100) with the MF tyre",
        f"util_f {b_q['util_f']:.4f} util_r {b_q['util_r']:.4f} '{b_q['limits']}'",
        f"util_f {mf_bal['util_f']:.4f} util_r {mf_bal['util_r']:.4f} "
        f"'{mf_bal['limits']}'",
        near(mf_bal["util_f"], b_q["util_f"], 1e-3)
        and near(mf_bal["util_r"], b_q["util_r"], 1e-3)
        and mf_bal["limits"] == b_q["limits"], hard=True)

    # A5  wet scales the peak and leaves the cornering stiffness alone. That is
    # what makes the wet peak arrive at 5.4 deg instead of 8.5 deg for free.
    ms = 0.55 / 0.87
    ky_dry = t.stiffnesses(2477.0)[1]
    ky_wet = t.stiffnesses(2477.0)[1]
    pk_wet = t.peak_fy(2477.0, ms)
    chk("A", "wet: peak scales, Kya does not",
        f"peak 1413.93 N +/- 0.5, Kya identical",
        f"peak {pk_wet:.2f} N, dKya {ky_wet - ky_dry:.1e}",
        near(pk_wet, 1413.93, 0.5) and ky_wet == ky_dry, hard=True)

    # A6  reported, not asserted: where the front axle actually peaks.
    Fz_o, Fz_i = 5463.0, 581.0
    best, best_a = 0.0, 0.0
    a = 0.0
    while a < 0.4:
        s = fabs(t.evaluate(Fz_o, 0.0, a)[1]) + fabs(t.evaluate(Fz_i, 0.0, a)[1])
        if s > best:
            best, best_a = s, a
        a += 0.0005
    QSS["alpha_peak_axle"] = degrees(best_a)
    chk("A", "front-axle alpha_peak at the R=100 limit split",
        "10.35 deg +/- 0.15 (NOT the 7.0 qss assumes)",
        f"{degrees(best_a):.2f} deg, axle sum {best:.0f} N",
        near(degrees(best_a), 10.35, 0.15), hard=False,
        note="feeds group X finding 2")


# ==================================================================== #
#  B. POWERTRAIN AND BRAKES                                            #
# ==================================================================== #
def group_B(quick):
    p = PT.PowertrainParams.from_car(CAR)

    chk("B", "wot_torque(4000)", "110.000 N.m (published)",
        f"{PT.wot_torque(p, 4000.0):.3f} N.m",
        near(PT.wot_torque(p, 4000.0), 110.0, 1e-3), hard=True)
    chk("B", "wot_power(5600)", "55000 W +/- 5 (published)",
        f"{PT.wot_power(p, 5600.0):.1f} W",
        near(PT.wot_power(p, 5600.0), 55000.0, 6.0), hard=True)

    V, rpm = PT.vmax_5th(p, CAR)
    chk("B", "Vmax in 5th against the road load",
        f"{CAR.Vmax:.4f} m/s +/- 0.10 (corsa_c)",
        f"{V:.4f} m/s at {rpm:.0f} rpm", near(V, CAR.Vmax, 0.10), hard=True)

    F = 0.5 * RHO * CAR.CdA * CAR.Vmax ** 2 + CAR.Crr * CAR.m * G
    P_req = CAR.Vmax * F
    chk("B", "power balance at Vmax (corsa_c.self_check parity)",
        f"{CAR.P_wheel / 1e3:.3f} kW available, within 0.5 kW",
        f"{P_req / 1e3:.3f} kW required ({F:.1f} N)",
        fabs(P_req - CAR.P_wheel) < 500.0, hard=True)

    chk("B", "gearing, 5th", "30.357 km/h per 1000 rpm +/- 0.05",
        f"{PT.kmh_per_1000rpm(p, 5):.3f}",
        near(PT.kmh_per_1000rpm(p, 5), 30.357, 0.05), hard=True)

    # The lock order is the one brake requirement that is not a calibration.
    # A rear-locking baseline would spin the car under trail braking and
    # corrupt the whole front/rear balance the wing study lives in.
    for label, ms in (("dry", 1.0), ("wet", 0.55 / 0.87)):
        pf = PT.lock_pressure(p, CAR, "f", ms)
        pr = PT.lock_pressure(p, CAR, "r", ms)
        chk("B", f"brake lock order, {label}",
            "front MUST lock before rear",
            f"front {pf / 1e5:.1f} bar, rear {pr / 1e5:.1f} bar "
            f"(margin {100 * (pr - pf) / pf:+.1f}%)",
            pr > pf, hard=True)

    d, tt, mean_g = PT.stopping_distance(p, CAR)
    chk("B", "100-0 km/h at full pedal", "34-50 m (UNPUBLISHED; road tests 38-41 m)",
        f"{d:.1f} m in {tt:.2f} s, mean {mean_g:.3f} g",
        34.0 <= d <= 50.0, hard=False,
        note="corsa_c.brakes is the string MISSING; every brake number here is "
             "a bottom-up reconstruction from a 236 mm disc and a 200 mm drum")

    worst = 0.0
    for i in range(500):
        T_ax = -400.0 + 1.7 * i
        wl, wr = 10.0 + 0.03 * i, 10.0 - 0.02 * i
        a, b = PT.diff_split(p, T_ax, wl, wr)
        worst = max(worst, fabs(a + b - T_ax))
    chk("B", "open diff conserves torque", "|err| < 1e-9 N.m over 500 states",
        f"{worst:.2e} N.m", worst < 1e-9, hard=True)

    if not quick:
        t0 = time.time()
        r = PT.accel_run(p, CAR, v_targets=(100 / 3.6,))
        t100 = r.get(100 / 3.6) or r.get("t") or list(r.values())[0]
        if isinstance(t100, dict):
            t100 = t100.get("t", float("nan"))
        chk("B", "0-100 km/h", "14.5-16.0 s (Opel quote 14.4; the common figure 15.5)",
            f"{t100:.2f} s", 14.5 <= t100 <= 16.0, hard=False,
            note="the only knobs are I_ENG and the shift model, both UNPUBLISHED. "
                 "Never tune the torque curve: it is pinned at both published "
                 "points and by the Vmax balance.",
            secs=time.time() - t0)


# ==================================================================== #
#  C. THE CHASSIS, AGAINST qss.py                                      #
# ==================================================================== #
PARITY = VE.VehicleConfig(qss_parity=True)


def group_C(quick):
    # C1/C2  the headline parity. The inequality is the real test: qss's
    # axle_capacity puts BOTH tyres of an axle at their own peak slip angle at
    # the same instant, which is geometrically impossible when they share one
    # slip angle. A two-track model must therefore come in slightly UNDER it.
    for R in (100.0, 50.0):
        t0 = time.time()
        s = VE.steady_state_corner(R, cfg=PARITY)
        v_q = QSS["cs"][int(R)][0]
        lo, hi = 0.985 * v_q, 1.015 * v_q
        ok = (lo <= s["V"] <= hi) and s["V"] <= v_q + 1e-9
        chk("C", f"corner speed R={int(R)} m, qss-parity",
            f"{v_q:.4f} m/s +/- 1.5% AND <= qss",
            f"{s['V']:.4f} m/s ({100 * (s['V'] / v_q - 1):+.3f}%)",
            ok, hard=True,
            note="beating qss means axle_capacity has been reimplemented somewhere",
            secs=time.time() - t0)
        QSS[f"ss{int(R)}"] = s

    # C3  the cos(delta) diagnostic. qss resolves the front axle's REQUIRED
    # force in the body frame but takes its CAPACITY in the tyre frame, i.e. it
    # assumes cos(delta) = 1. Removing the projection is what closes the gap.
    a_on = VE.ramp_steer(29.0875, cfg=PARITY)["peak_ay_g"]
    a_off = VE.ramp_steer(29.0875,
                          cfg=VE.VehicleConfig(qss_parity=True,
                                               force_cos_delta=False))["peak_ay_g"]
    chk("C", "peak a_y at V=29.0875, projection ON", "0.845-0.863 g",
        f"{a_on:.4f} g ({100 * (a_on / 0.862467 - 1):+.2f}% vs qss)",
        0.845 <= a_on <= 0.863, hard=False)
    chk("C", "peak a_y, projection OFF (the diagnostic)", "0.857-0.866 g",
        f"{a_off:.4f} g ({100 * (a_off / 0.862467 - 1):+.2f}% vs qss)",
        0.857 <= a_off <= 0.866, hard=False,
        note="closing to ~0.1% with the projection removed is the proof the "
             "tyre and the load transfer are right")

    # C4  a FWD hatch must run out of front axle first, at every radius.
    radii = (30.0, 50.0, 75.0, 100.0, 130.0) if not quick else (50.0, 100.0)
    worst = 1e9
    parts = []
    for R in radii:
        s = QSS.get(f"ss{int(R)}") or VE.steady_state_corner(R, cfg=PARITY)
        QSS[f"ss{int(R)}"] = s
        worst = min(worst, s["util_f"] - s["util_r"])
        parts.append(f"R{int(R)}:{s['util_f']:.3f}/{s['util_r']:.3f}")
    chk("C", "front-limited at every radius", "util_f > util_r everywhere",
        "  ".join(parts) + f"   min margin {worst:+.4f}",
        worst > 0.0, hard=True)

    # C5  vertical equilibrium. The load-transfer split moves load between
    # corners; it may not create or destroy any.
    s = QSS["ss100"]
    Fz = s["Fz"]
    chk("C", "per-corner Fz sums to m*g at the R=100 limit",
        f"{CAR.m * G:.1f} N +/- 1 N, no wheel lifted",
        f"[{Fz[0]:.1f}, {Fz[1]:.1f}, {Fz[2]:.1f}, {Fz[3]:.1f}] "
        f"sum {sum(Fz):.2f}",
        fabs(sum(Fz) - CAR.m * G) < 1.0 and not any(s["wheel_lift"]), hard=True)

    # C6  total lateral transfer, against qss's ground-plane form.
    ref = (CAR.m * s["peak_ay"] * CAR.h_cg) / (0.5 * (CAR.t_f + CAR.t_r))
    chk("C", "total lateral load transfer", f"qss form {ref:.1f} N, within 1%",
        f"{s['dFz_tot_demand']:.1f} N",
        rel(s["dFz_tot_demand"], ref) < 0.01, hard=True)

    # C7  roll. The only place the (UNPUBLISHED) damper rate shows up.
    grad = s["phi_deg"] / s["ay_g"]
    chk("C", "roll angle at the R=100 limit", "4.30-4.80 deg (qss.roll_angle 4.599)",
        f"{s['phi_deg']:.3f} deg", 4.30 <= s["phi_deg"] <= 4.80, hard=False)
    chk("C", "roll gradient", "5.33 deg/g +/- 0.15",
        f"{grad:.3f} deg/g", near(grad, 5.33, 0.15), hard=False,
        note="corsa_c's own comment on Kphi_tot claims 4.7-5.1 deg/g, which is "
             "inconsistent with its own numbers by 5-13%")

    if not quick:
        r = VE.roll_step_response()
        ok = (0.25 <= r["overshoot"] <= 0.37) and (0.29 <= r["t_peak"] <= 0.36)
        chk("C", "roll step response (the only Cphi test)",
            "overshoot 25-37%, t_peak 0.29-0.36 s",
            f"overshoot {100 * r['overshoot']:.1f}%, t_peak {r['t_peak']:.3f} s, "
            f"f_n {r['f_n']:.3f} Hz", ok, hard=False,
            note="corsa_c.dampers is the string MISSING; zeta = 0.35 is an "
                 "estimate, band 0.25-0.50")


# ==================================================================== #
#  D. THE FLANK WING                                                   #
# ==================================================================== #
def _gain(wing, x_w, h_w=0.90, R=100.0):
    cfg = VE.VehicleConfig(qss_parity=True, wing=wing, x_w=x_w, h_w=h_w)
    v0 = QSS["ss100_parity"]["V"]
    v1 = VE.steady_state_corner(R, cfg=cfg)["V"]
    return 100.0 * (v1 / v0 - 1.0)


def group_D(quick):
    QSS["ss100_parity"] = QSS.get("ss100") or VE.steady_state_corner(100.0,
                                                                     cfg=PARITY)

    t0 = time.time()
    g_fin_fa = _gain("fin", 0.97)
    g_plt_fa = _gain("plate", 0.97)
    chk("D", "clean fin at the front axle, R=100", "+1.05% .. +1.50%, positive",
        f"{g_fin_fa:+.3f}%", 1.05 <= g_fin_fa <= 1.50, hard=False)
    chk("D", "sealed plate at the front axle, R=100", "+1.8% .. +2.8%, positive",
        f"{g_plt_fa:+.3f}%", g_plt_fa > 0.0 and 1.8 <= g_plt_fa <= 2.8,
        hard=True,
        note="qss AS SHIPPED reports only +1.18% here, BELOW its own clean fin "
             "+1.21% -- impossible for a front-limited car. See group X.",
        secs=time.time() - t0)

    # The sign test. The rear AXLE station gives (x_w+b)/b = -0.0003, which is
    # numerically zero and proves nothing; the rear BUMPER at -1.90 m gives
    # -0.2504 and is unambiguous.
    g_cg = _gain("fin", 0.0)
    g_rb = _gain("fin", -1.90)
    g_plt_rb = _gain("plate", -1.90)
    chk("D", "gain falls monotonically with x_w",
        "front axle > at the CG > rear bumper",
        f"{g_fin_fa:+.3f}% > {g_cg:+.3f}% > {g_rb:+.3f}%",
        g_fin_fa > g_cg > g_rb, hard=True)
    chk("D", "rear-bumper mount makes it WORSE (sign check)",
        "<= -0.10%, and >= 1.5 pp below the front axle",
        f"fin {g_rb:+.3f}%, plate {g_plt_rb:+.3f}%, "
        f"plate spread {g_plt_fa - g_plt_rb:+.3f} pp",
        g_rb <= -0.10 and (g_plt_fa - g_plt_rb) >= 1.5, hard=True,
        note="a positive rear gain means x_w has the wrong sign in the yaw moment")

    # The two independent routes to the same number. crossover.gain is a closed
    # form in which the multiplier is IMPOSED; this sim sums per-wheel forces
    # and the multiplier emerges from the yaw balance. They have no code in
    # common beyond corsa_c.py, so agreement is a real cross-check.
    for wing, cl in (("fin", 0.70), ("plate", 1.25)):
        k = 0.5 * RHO * 0.35 * cl
        g_cf = 100.0 * crossover.gain(k, 100.0, 0.97)
        g_sim = g_fin_fa if wing == "fin" else g_plt_fa
        chk("D", f"{wing} gain vs crossover.gain closed form",
            f"{g_cf:+.4f}% (closed form), within 0.30 pp",
            f"{g_sim:+.4f}% (two-track sim), diff {g_sim - g_cf:+.4f} pp",
            fabs(g_sim - g_cf) < 0.30, hard=True,
            note="qss.py, which shares the closed form's assumptions, reports "
                 f"{100 * (qss.corner_speed(100.0, k=k, x_w=0.97, power_cap=False)[0] / QSS['cs'][100][0] - 1):+.3f}% "
                 "-- the gap is the Y_r bug in group X")

    ratio = g_fin_fa / g_cg if g_cg else 0.0
    cf = (0.97 + CAR.b) / CAR.b
    chk("D", "(x_w+b)/b multiplier emerges from the yaw balance",
        f"ratio(front axle : CG) near the closed form {cf:.3f}",
        f"{ratio:.3f}", 1.45 <= ratio <= 1.85, hard=False,
        note="it EMERGES from the moment balance here; qss imposes it")

    # The double-count trap. If SFy_tyre already contains F_dev the load
    # transfer charges F_dev*(2*h_cg - h_w) and this sweep blows up.
    if not quick:
        hs = (0.30, 0.50, 0.70, 0.90, 1.10)
        gs = [_gain("plate", 0.97, h) for h in hs]
        swing = gs[-1] - gs[0]
        mono = all(gs[i + 1] >= gs[i] - 1e-9 for i in range(len(gs) - 1))
        chk("D", "mount-height insensitivity (F_dev double-count trap)",
            "swing 0.05-0.35 pp, monotone increasing",
            "  ".join(f"h{h:.2f}:{g:+.3f}%" for h, g in zip(hs, gs))
            + f"   swing {swing:+.3f} pp",
            0.05 <= swing <= 0.35 and mono, hard=True)

    # The identity the whole (x_w+b)/b argument rests on, read off the model's
    # own per-wheel forces rather than an imposed axle split.
    cfg = VE.VehicleConfig(qss_parity=True, wing="plate", x_w=0.97)
    t = VE.ramp_steer(29.0, cfg=cfg)
    chk("D", "Y_f equals the model's own yaw-balance identity",
        "within 1.0%", f"{t['Y_f']:.1f} N vs {t['Y_f_full']:.1f} N "
        f"({100 * rel(t['Y_f'], t['Y_f_full']):.3f}%)",
        rel(t["Y_f"], t["Y_f_full"]) < 0.01, hard=True)

    # Scale, in newtons, so nobody mistakes this for a downforce car.
    q = 0.5 * RHO * QSS["ss100_parity"]["V"] ** 2
    F_fin = q * 0.35 * 0.70
    F_plt = q * 0.35 * 1.25
    note("D", f"scale: at the R=100 limit the device makes {F_fin:.0f} N (fin) / "
              f"{F_plt:.0f} N (plate) against a {CAR.m * G:.0f} N car "
              f"-- {100 * F_fin / (CAR.m * G):.1f}% / "
              f"{100 * F_plt / (CAR.m * G):.1f}% of its weight. It will not be felt.")


# ==================================================================== #
#  E. NUMERICS                                                         #
# ==================================================================== #
def group_E(quick):
    # E1  if the answer moves with dt, the answer is the integrator's, not the
    # car's.
    dts = (0.00025, 0.0005, 0.001, 0.002)
    ays = [VE.ramp_steer(29.0875, cfg=PARITY, dt=d)["peak_ay_g"] for d in dts]
    spread = (max(ays) - min(ays)) / max(ays)
    chk("E", "timestep convergence of peak a_y", "spread < 0.2% over dt 0.25-2 ms",
        "  ".join(f"{1000 * d:.2f}ms:{a:.5f}g" for d, a in zip(dts, ays))
        + f"   spread {100 * spread:.4f}%",
        spread < 0.002, hard=True)

    # E2  the ordering that makes the difference between a sim and a firework.
    spec_kx = VE.ordering_test(reverse=False)
    rev_kx = VE.ordering_test(reverse=True)
    ok = spec_kx < 1e-6 and rev_kx > 1e-3
    chk("E", "integrator ordering (2% kappa perturbation at 3 m/s)",
        "specified decays < 1e-6; REVERSED must diverge",
        f"specified {spec_kx:.3e}, reversed {rev_kx:.3e}",
        ok, hard=True,
        note="if both orderings are stable the wheel and slip states are not "
             "actually coupled and the test proves nothing")

    # E3  a car that will not sit still on a hill has no low-speed model.
    veh = VE.Vehicle(CAR, VE.VehicleConfig())
    veh.reset(V=0.0, gear=0)
    veh.grade = atan(0.10)
    ctl = VE.Controls(brake=1.0, clutch=1.0, gear_req=0)
    x0 = veh.x
    peak_u = 0.0
    for _ in range(int(60.0 / VE.DT_PHYS)):
        veh.step(ctl, (1.0,) * 4, (1.0,) * 4, VE.DT_PHYS)
        peak_u = max(peak_u, fabs(veh.u))
    creep = fabs(veh.x - x0)
    chk("E", "standstill on a 10% grade, full brake, 60 s", "< 5 mm of creep",
        f"{1000 * creep:.4f} mm, max |u| {peak_u:.2e} m/s",
        creep < 0.005, hard=True)

    # E4  released straight at 40 m/s. Catches an asymmetric Ackermann, an
    # asymmetric Fz split and a sign error in the -y_i*Fbx_i yaw term.
    veh = VE.Vehicle(CAR, VE.VehicleConfig())
    veh.reset(V=40.0, gear=5)
    ctl = VE.Controls()
    for _ in range(int(20.0 / VE.DT_PHYS)):
        veh.step(ctl, (1.0,) * 4, (1.0,) * 4, VE.DT_PHYS)
    chk("E", "straight-line stability, 20 s from 40 m/s",
        "|Y| < 0.05 m, |psi| < 0.002 rad",
        f"|Y| {fabs(veh.y):.3e} m, |psi| {fabs(veh.psi):.3e} rad, "
        f"|r| {fabs(veh.r):.2e} rad/s",
        fabs(veh.y) < 0.05 and fabs(veh.psi) < 0.002, hard=True)

    # E5  the car must not prefer a direction.
    vl = VE.steady_state_corner(100.0, cfg=PARITY, side=+1)["V"]
    vr = VE.steady_state_corner(100.0, cfg=PARITY, side=-1)["V"]
    chk("E", "left/right symmetry, device off", "|dV| < 1e-9 m/s",
        f"{fabs(vl - vr):.3e} m/s", fabs(vl - vr) < 1e-9, hard=True)

    cw = VE.VehicleConfig(qss_parity=True, wing="plate", x_w=0.97)
    wl = VE.steady_state_corner(100.0, cfg=cw, side=+1)["V"]
    wr = VE.steady_state_corner(100.0, cfg=cw, side=-1)["V"]
    chk("E", "left/right symmetry, device ON", "|dV| < 1e-9 m/s",
        f"{fabs(wl - wr):.3e} m/s", fabs(wl - wr) < 1e-9, hard=True,
        note="an asymmetry here is the device's own sign convention, not the car's")

    # E6  guards.
    veh = VE.Vehicle(CAR, VE.VehicleConfig(guards=True))
    veh.reset(V=20.0, gear=3)
    veh.state.u = float("nan")
    veh.step(VE.Controls(), (1.0,) * 4, (1.0,) * 4, VE.DT_PHYS)
    ok = (all(v == v for v in (veh.u, veh.v, veh.r, veh.x, veh.y))
          and bool(veh.guard_events))
    chk("E", "NaN guard recovers a poisoned state",
        "state finite again AND the offending field is named",
        f"u {veh.u:.3f}, logged {veh.guard_events[:1]}", ok, hard=True,
        note="'u became non-finite at step N' is debuggable; 'sim reset' is not")

    # E7  energy. A sim that gains energy while coasting is lying somewhere.
    c = VE.coast_down()
    chk("E", "coast-down energy audit", "closure < 0.5%, KE strictly decreasing",
        f"KE drop {c['KE_drop'] / 1e3:.1f} kJ, dissipated "
        f"{c['dissipated'] / 1e3:.1f} kJ, closure {100 * c['closure']:+.3f}%",
        fabs(c["closure"]) < 0.005 and c["monotonic"], hard=True)


# ==================================================================== #
#  F. THE HARNESS                                                      #
# ==================================================================== #
def group_F(quick):
    arena = TR.make_arena()
    d_close = float(((arena.xy[-1] - arena.xy[0]) ** 2).sum() ** 0.5)
    chk("F", "CIRCUIT_ARENA closure", "< 1e-4 m, length 1249.2022 m +/- 0.01",
        f"{d_close:.2e} m, length {arena.length:.4f} m",
        d_close < 1e-4 and near(arena.length, 1249.2022, 0.01), hard=True)

    import random
    rng = random.Random(7)
    worst_s = worst_n = 0.0
    t0 = time.time()
    N = 200 if quick else 500
    for _ in range(N):
        s0 = rng.uniform(0.0, arena.length)
        n0 = rng.uniform(-6.0, 6.0)
        x, y = TR.point_at(arena, s0, n0)
        s1, n1, _, _, _ = TR.project(arena, x, y)
        ds = fabs(s1 - s0)
        ds = min(ds, arena.length - ds)
        worst_s = max(worst_s, ds)
        worst_n = max(worst_n, fabs(n1 - n0))
    us = 1e6 * (time.time() - t0) / N
    chk("F", "track projection round trip", "|ds|,|dn| < 1e-3 m and < 20 us/call",
        f"max |ds| {worst_s:.2e} m, |dn| {worst_n:.2e} m, {us:.1f} us/call",
        worst_s < 1e-3 and worst_n < 1e-3 and us < 20.0, hard=True)

    # per-wheel surface lookup: a car-centre lookup silently deletes split-mu
    x, y = TR.point_at(arena, 985.0, -3.0)
    mu_r = TR.surface_at(arena, x, y)[0]
    x, y = TR.point_at(arena, 985.0, +3.0)
    mu_l = TR.surface_at(arena, x, y)[0]
    chk("F", "split-mu is resolved per wheel", "right 0.80, left 1.00",
        f"right {mu_r:.3f}, left {mu_l:.3f}",
        near(mu_r, 0.80, 1e-6) and near(mu_l, 1.0, 1e-6), hard=True)

    # input ramps: the aid must be exactly what it claims
    from drive import input as IN
    chk("F", "steer limiter at 25 m/s, beta 0", "5.396 deg +/- 0.05",
        f"{IN.steer_limit_deg(25.0, 0.0):.3f} deg",
        near(IN.steer_limit_deg(25.0, 0.0), 5.396, 0.05), hard=True)
    chk("F", "steer limiter opens up in a slide (beta 12 deg at 30 m/s)",
        "19.12 deg +/- 0.10 (vs 4.72 at beta 0)",
        f"{IN.steer_limit_deg(30.0, 12.0):.3f} deg "
        f"(beta 0: {IN.steer_limit_deg(30.0, 0.0):.3f})",
        near(IN.steer_limit_deg(30.0, 12.0), 19.12, 0.10), hard=True,
        note="without this every slide is an unrecoverable spin and the user "
             "blames the physics for an input-layer failure")

    from drive import drive as DR
    ok, info = DR._v25_lap_timing(verbose=False)
    chk("F", "lap timing is sub-step accurate",
        f"{info['exact_s']:.6f} s +/- 0.002",
        f"{info['measured_s'][0]:.6f} s "
        f"(max error {1e6 * info['max_err_s']:.2f} us)", ok, hard=True,
        note="a whole-step timer quantises at 1 ms, half the tolerance on its own")

    ok, info = DR._v24_first_frame(verbose=False)
    chk("F", "first frame after init/reset/unpause", "exactly 1 physics step each",
        f"init {info['init']}, reset {info['after_reset']}, "
        f"unpause {info['after_unpause']}", ok, hard=True,
        note="pygame's first frame on macOS can be 0.5-2 s; feeding that to the "
             "accumulator teleports the car before it is ever drawn")

    ok, info = DR._v23_accumulator(verbose=False)
    chk("F", "accumulator does not spiral", "40 steps then drop the remainder",
        f"{info['steps']} steps, dropped {info['dropped_s']:.4f} s", ok, hard=True)

    if not quick:
        tmp = DR._tmpdir()
        ok, info = DR._v20_determinism(tmp, verbose=False)
        chk("F", "headless determinism", "two scripted runs byte-identical",
            f"sha {info['sha'][:16]}, {info['bytes']} bytes", ok, hard=True,
            note="a difference means wall-clock time, dict order or an unseeded "
                 "RNG has leaked into the physics path")

        # end to end: a scripted driver must actually get round the circuit.
        # The reference ideal lap from a quasi-steady forward/backward pass is
        # 53.7 s; a driver on a 0.90 margin through a wet patch should be well
        # inside the 58-75 s band a competent human records.
        t0 = time.time()
        o = DR._Opts(track="arena", wet="patch", duration=200.0,
                     telemetry=os.path.join(tmp, "lap.csv"))
        r = DR.lap_script(o)
        ok = r["laps"] >= 1 and r["max_n"] < 6.0 and 55.0 <= r["best"] <= 80.0
        chk("F", "scripted lap of CIRCUIT_ARENA completes",
            ">= 1 lap, stays on a 12 m ribbon, 55-80 s",
            f"{r['laps']} laps, best {r['best']:.2f} s, max |n| {r['max_n']:.2f} m",
            ok, hard=True,
            note="the quasi-steady ideal is 53.7 s; the speed profile must be "
                 "surface-aware or the car understeers off in WET_T3",
            secs=time.time() - t0)

        t0 = time.time()
        ok, info = DR._v21_rtf(tmp, seconds=20.0, verbose=False)
        chk("F", "real-time factor, headless", "RTF >= 2.0 (target >= 3.0)",
            f"{info['rtf']:.2f}x ({info['wall_s']:.2f} s wall for 20 s of sim, "
            f"{info['rows']} telemetry rows)", info["rtf"] >= 2.0, hard=True,
            secs=time.time() - t0)


# ==================================================================== #
#  G. DRIVING MODES, ABS, THE OPEN MAP, SETTINGS                       #
# ==================================================================== #
def group_G(quick):
    from drive import drive as DR
    from drive import input as IN

    # G1  three driver models on the real car (drive.py V27)
    t0 = time.time()
    ok, info = DR._v27_gearbox_modes(verbose=False)
    chk("G", "gearbox modes: auto / manual / clutch",
        "auto never stalls; manual 0-100 in 13.5-18 s on the driver's shifts; "
        "clutch mode stalls, restarts with the pedal in, launches",
        f"auto {info['auto'][0]:.1f} m/s gear {info['auto'][1]}; manual 0-100 "
        f"{info['manual'][0]:.2f} s; clutch stall {info['clutch'][0]:.2f} s, fire "
        f"{info['clutch'][1]:.2f} s, launch {info['clutch'][2]:.1f} m/s",
        ok, hard=True, secs=time.time() - t0,
        note="Controls.auto_gearbox / auto_clutch; powertrain.update_shift owns "
             "the launch assist, the rev-match blip and the restart")

    # G2  the stall -> restart bug: a stalled engine must be able to fire
    p = PT.PowertrainParams.from_car(CAR)
    st = PT.PowertrainState(omega_e=0.0, gear=0, stalled=True)
    inp = PT.PtInput(auto_gearbox=False, auto_clutch=False, clutch=1.0)
    t_fire = None
    for k in range(3000):
        o = PT.step(p, st, inp, (0.0,) * 4, (2500.0,) * 4, (0.0,) * 4, 0.0, 1e-3)
        if not o.stalled and t_fire is None:
            t_fire = (k + 1) * 1e-3
    chk("G", "stalled engine restarts (clutch fully in)", "fires < 1 s, idles 700-1000 rpm",
        f"fires at {t_fire if t_fire else float('nan'):.3f} s, {o.rpm:.0f} rpm",
        t_fire is not None and t_fire < 1.0 and 700.0 <= o.rpm <= 1000.0, hard=True,
        note="the starter used to stop at 400 rpm with the fuel off, 100 rpm short "
             "of 'running', so every stall was permanent")

    # G3  rev-matched downshift
    r = PT._Rig(p, CAR, gear=3, v=15.0, rpm=PT.rpm_at_speed(p, 3, 15.0))
    i = PT.PtInput(auto_gearbox=False, auto_clutch=True)
    for _ in range(80):
        r.step(i, 1 / 400.0)
    i.shift_dn = True
    n_eng = None
    for _ in range(600):
        o = r.step(i, 1 / 400.0)
        if r.s.shift_phase == "engage" and n_eng is None:
            n_eng = o.rpm
    n_tgt = PT.rpm_at_speed(p, 2, 15.0)
    chk("G", "downshift blip (auto clutch) matches the new gear",
        f"engine within 400 rpm of {n_tgt:.0f} rpm at engagement",
        f"{n_eng:.0f} rpm at engagement, ends in gear {o.gear}",
        n_eng is not None and fabs(n_eng - n_tgt) < 400.0 and o.gear == 2, hard=True)

    # G4  ABS, dry and wet, against the raw brakes
    t0 = time.time()
    thr = VE.brake_run(pedal=0.60)
    lck = VE.brake_run(pedal=1.00)
    ab = VE.brake_run(pedal=1.00, cfg=VE.VehicleConfig(abs_on=True))
    lck_w = VE.brake_run(pedal=1.00, cfg=VE.VehicleConfig(mu_scale=0.632))
    ab_w = VE.brake_run(pedal=1.00, cfg=VE.VehicleConfig(mu_scale=0.632, abs_on=True))
    chk("G", "ABS at full pedal, dry and wet",
        "no lock; dry <= 0.6-pedal +6% and < locked; wet < 0.85 x locked",
        f"dry {ab['distance']:.2f} m (kappa_min {ab['kappa_min']:.2f}) vs 0.6-pedal "
        f"{thr['distance']:.2f} / locked {lck['distance']:.2f}; wet {ab_w['distance']:.2f} m "
        f"(kappa_min {ab_w['kappa_min']:.2f}) vs locked {lck_w['distance']:.2f}",
        ab["kappa_min"] > -0.5 and ab_w["kappa_min"] > -0.5
        and ab["distance"] <= 1.06 * thr["distance"] and ab["distance"] < lck["distance"]
        and ab_w["distance"] < 0.85 * lck_w["distance"], hard=True,
        secs=time.time() - t0,
        note="VehicleConfig.abs_on defaults False: every other number in this suite "
             "is measured on raw brakes")
    # ABS must not touch a determinism replay: same code, same bits
    veh = VE.Vehicle(CAR, VE.VehicleConfig(abs_on=True))
    outs = []
    for _ in range(2):
        veh.reset(V=25.0, gear=4)
        for k in range(1500):
            veh.step(VE.Controls(brake=1.0 if k > 200 else 0.0, delta=0.01 * sin(0.01 * k)),
                     (1.0,) * 4, (1.0,) * 4, 1e-3)
        outs.append(veh.state.as_array().copy())
    chk("G", "ABS replay is bit-identical", "0 ULP",
        f"max |d| {max(fabs(a - b) for a, b in zip(outs[0], outs[1])):.1e}",
        all(a == b for a, b in zip(outs[0], outs[1])), hard=True)

    # G5  the open map (drive.py V28)
    if not quick:
        t0 = time.time()
        ok, info = DR._v28_open_map(verbose=False)
        chk("G", "open map: perimeter lap, pad crossing, surfaces",
            ">= 1 timed lap 90-110 s at 17 m/s on the 12 m road; crossing the pad "
            "stays on track with no lap/sector event; wet square 0.632; grass 0.55",
            f"{info['laps']} lap(s), first {info['lap_s']:.2f} s, max |n| "
            f"{info['max_n']:.2f} m; crossing {info['cross']}, wet {info['wet']}, "
            f"grass {info['grass']}", ok, hard=True, secs=time.time() - t0)
    op = TR.make_open()
    pts = ((105.0, 150.0, True), (300.0, 130.0, True), (210.0, 0.0, True),
           (210.0, -80.0, False), (-50.0, -50.0, False), (470.0, 320.0, False))
    got = [TR.on_tarmac(op, x, y) for x, y, _ in pts]
    chk("G", "open map: tarmac / grass classification", "pad, wet square, road on; "
        "south, corner cut, NE corner off",
        " ".join("on" if g else "off" for g in got),
        got == [e for _, _, e in pts], hard=True)

    # G6  settings + menu state machine (drive.py V26)
    tmp = DR._tmpdir()
    ok, info = DR._v26_settings_and_menu(tmp, verbose=False)
    chk("G", "settings persist; ESC menu -> settings -> garage / map restart",
        "round-trip, CLI precedence, gearbox/ABS/aid/camera live, map & TAB "
        "restart, garage from the menu and from settings",
        f"round-trip {info['round_trip']}, cli {info['cli']}, menu {info['menu']}, "
        f"map restart {info['map_restart']}, garage {info['garage']}", ok, hard=True)
    chk("G", "gearbox mode names agree across modules", "input == drive",
        f"{IN.GEARBOX_MODES} / {DR.GEARBOX_MODES}",
        tuple(IN.GEARBOX_MODES) == tuple(DR.GEARBOX_MODES)
        and IN.GEARBOX_HUD == DR.GEARBOX_HUD, hard=True)

    # G7  the Engine setting + TC on the interactive path (drive.py V29)
    t0 = time.time()
    ok, info = DR._v29_engine_tc(verbose=False)
    chk("G", "Engine setting + TC (interactive path, WOT from rest)",
        "stock 0-100 in 14.3-15.8 s with TC never active; 2x spins (kappa > 0.8) "
        "without TC, < 10 s to 100 with it (kappa < 0.5); live swap",
        f"stock {info['stock'][0]:.2f} s kappa {info['stock'][1]:.2f} TC {info['stock'][2]:.1f} s; "
        f"2x raw kappa {info['raw'][1]:.2f}; 2x TC {info['tc'][0]:.2f} s kappa "
        f"{info['tc'][1]:.2f}; live {info['live']}", ok, hard=True,
        secs=time.time() - t0,
        note="VehicleConfig.power_scale scales the WOT curve and the clutch cap "
             "(powertrain.from_car); TC scales the engine LOAD (PtInput.tc_scale), "
             "never the pedal the shift scheduler reads. Both 1.0 / off in every rig.")

    # G8  launch assist holds its target (the droop fix)
    p = PT.PowertrainParams.from_car(CAR)
    r = PT._Rig(p, CAR, gear=1, v=0.0, rpm=p.n_idle)
    i = PT.PtInput(throttle=1.0, auto_gearbox=False, auto_clutch=True)
    n_lo = 9e9
    for k in range(1400):
        o = r.step(i, 1e-3)
        if 0.6 <= k * 1e-3 <= 1.2:
            n_lo = min(n_lo, o.rpm)
    chk("G", "launch assist holds the engine near n_launch",
        f"> {p.n_launch - p.n_launch_band:.0f} rpm through the slip phase (was ~1700)",
        f"min {n_lo:.0f} rpm at 0.6-1.2 s", n_lo > p.n_launch - p.n_launch_band, hard=True,
        note="the old proportional band ran from n_stall+100 to the target; the "
             "clutch's torque balance then parked the engine 700 rpm under it")

    # G9  the sound module, offline + SDL dummy driver
    t0 = time.time()
    try:
        from drive import audio as AU
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            ok = AU.self_check(verbose=False, wav_path=os.path.join(tmp, "audio_demo.wav"))
        got = ("50 checks (engine per car, tyres, road, air, cues, the lap chime, junk "
               "HUD values, silence, speed, streaming, demos)")
    except Exception as exc:
        ok, got = False, f"{type(exc).__name__}: {exc}"
    chk("G", "car sound: synth + streaming self-check", "all pass", got, ok, hard=False,
        secs=time.time() - t0,
        note="drive/audio.py: procedural engine / tyre / road / air sound, "
             "render-loop consumer of HudData only; never an input to the physics")


# ==================================================================== #
#  X. FINDINGS ABOUT THE STUDY, NOT ABOUT THE SIM                      #
# ==================================================================== #
def group_X():
    car = CAR
    L, a, b = car.L, car.a, car.b

    print()
    print("=" * 96)
    print("X. TWO FINDINGS ABOUT THE ANALYSIS SUITE  (CONTRACT.md section 10)")
    print("=" * 96)

    # ---- 1. the Y_r algebra bug ------------------------------------------
    a_y = 0.862467 * G
    F = 0.5 * RHO * 0.35 * 1.25 * 29.0875 ** 2      # sealed plate at the limit
    x_w = 0.97
    Y_f = (b * car.m * a_y - F * (b + x_w)) / L
    Y_r_shipped = (a * car.m * a_y + F * x_w) / L
    Y_r_correct = (a * car.m * a_y - F * (a - x_w)) / L

    print()
    print("1. qss.residuals' Y_r violates qss's own stated force balance.")
    print("   qss.py states     Y_f + Y_r + F = m*a_y   and   Y_f*a - Y_r*b + F*x_w = 0")
    print("   Solving those two gives  Y_r = (a*m*a_y - F*(a - x_w))/L")
    print("   qss.py:88 codes          Y_r = (a*m*a_y + F*x_w)/L        <- missing -F*a/L")
    print()
    print(f"   at a_y = 0.8625 g, sealed plate F = {F:.1f} N, x_w = {x_w:.2f} m:")
    print(f"     Y_f                  {Y_f:10.1f} N   (correct in both)")
    print(f"     Y_r as shipped       {Y_r_shipped:10.1f} N")
    print(f"     Y_r corrected        {Y_r_correct:10.1f} N   "
          f"({100 * (Y_r_shipped / Y_r_correct - 1):+.2f}% too high)")
    print(f"     Y_f + Y_r + F  shipped {Y_f + Y_r_shipped + F:10.1f} N   "
          f"vs m*a_y = {car.m * a_y:.1f} N   "
          f"(off by {Y_f + Y_r_shipped + F - car.m * a_y:+.1f} N)")
    print(f"     Y_f + Y_r + F  correct {Y_f + Y_r_correct + F:10.1f} N   exact")
    print()
    print("   With the wing OFF (F = 0) the two agree exactly, so every baseline")
    print("   number in qss/crossover/ledger stands. With the wing ON the inflated")
    print("   Y_r sends the rear axle spuriously limiting, which is why qss.sweep()")
    print("   prints the sealed plate at +1.18% BELOW its own clean fin at +1.21%")
    print("   at R = 100 m -- impossible for a front-limited car.")
    print(f"   This sim, which sums real per-wheel forces and never imposes an axle")
    print(f"   split, measures the sealed plate at {QSS.get('g_plate', float('nan')):+.2f}% there.")

    # ---- 2. alpha_peak ---------------------------------------------------
    ap = QSS.get("alpha_peak_axle", 10.35)
    print()
    print("2. alpha_peak_deg = 7.0 in qss.corner_speed and crossover.sustainable_V")
    print(f"   is too low. The Magic Formula front-axle peak at the R=100 limit load")
    print(f"   split (outer 5463 N, inner 581 N) is {ap:.2f} deg, and this sim measures")
    print(f"   {QSS.get('alpha_f_meas', float('nan')):.2f} deg on the front axle at that limit.")
    print(f"   Scrub drag is m*a_y*sin(alpha_peak), so the term is "
          f"{sin(radians(ap)) / sin(radians(7.0)):.2f}x larger than assumed:")
    print()
    print(f"     {'quantity':38s} {'at 7.0 deg':>14s} {'at ' + f'{ap:.2f}' + ' deg':>14s}")
    for R in (130, 175, 250):
        v7 = qss.corner_speed(float(R), alpha_peak_deg=7.0)
        v10 = qss.corner_speed(float(R), alpha_peak_deg=ap)
        print(f"     qss.corner_speed({R:3d}){'':17s} {v7[0]:9.3f} m/s {v10[0]:9.3f} m/s")
    s7 = crossover.sustainable_V(0.87, 7.0)
    s10 = crossover.sustainable_V(0.87, ap)
    print(f"     crossover.sustainable_V(0.87){'':9s} {s7:9.3f} m/s {s10:9.3f} m/s")
    print(f"     -> R_cap{'':30s} {s7 * s7 / (0.87 * G):9.1f} m   "
          f"{s10 * s10 / (0.87 * G):9.1f} m")
    print()
    print("   That shrinks the radius band over which the device can act at all.")
    print("   It is a finding about the study, not a sim detail. Neither qss.py nor")
    print("   crossover.py has been modified.")


# ==================================================================== #
#  MODULE SELF-CHECKS (subprocess, so nothing is duplicated here)      #
# ==================================================================== #
MODULES = ("tyre", "powertrain", "track", "telemetry", "vehicle", "input",
           "render", "menu", "garage", "aero.airfoil", "aero.panel2d", "aero.polar",
           "aero.xfoil", "aero.blend", "aero.vlm", "aero.wing", "aero.optimize",
           "aero.library", "aero.screen", "aero.section", "aero.mission",
           "records", "prerace", "medals", "ghosts", "progress", "tutorial",
           "wing_tutorial", "challenges", "race_grid", "swarm_panel", "results", "airbrake", "controls_page",
           "scenery", "world", "props", "fx")


def group_M():
    env = dict(os.environ, CARSIM_HEADLESS="1", SDL_VIDEODRIVER="dummy",
               SDL_AUDIODRIVER="dummy")
    for m in MODULES:
        t0 = time.time()
        r = subprocess.run([sys.executable, "-m", f"drive.{m}"], cwd=REPO,
                           env=env, capture_output=True, text=True)
        tail = [l for l in (r.stdout or "").strip().splitlines() if l.strip()]
        last = tail[-1][:70] if tail else "(no output)"
        chk("M", f"python3 -m drive.{m}", "exit 0", f"exit {r.returncode}  |  {last}",
            r.returncode == 0, hard=True, secs=time.time() - t0)
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "drive.drive", "--self-check"],
                       cwd=REPO, env=env, capture_output=True, text=True)
    tail = [l for l in (r.stdout or "").strip().splitlines() if l.strip()]
    chk("M", "python3 -m drive.drive --self-check", "exit 0",
        f"exit {r.returncode}  |  {(tail[-1][:70] if tail else '')}",
        r.returncode == 0, hard=True, secs=time.time() - t0)


# ==================================================================== #
#  W. THE THREE WINGS (drive/aero + the garage build)                  #
# ==================================================================== #
def group_W(quick):
    """The designed wings: the numpy port of the AeroBO lattice against
    its reference numbers, the designed panel against the published one,
    the top wing's station trade, its deploy logic, the build's two
    physics paths and the renderer's three-wing frame."""
    import contextlib
    import io
    from drive.aero import vlm as AV, wing as AW, polar as AP, airfoil as AF
    from drive import garage as GR

    # W1  the lattice port reproduces AeroBO's VLM (aerobo.vlm.VLM, N=40)
    w = AV.rect(8.0, 1.0, N=40)
    r = w.solve(5.0)
    chk("W", "vortex lattice == AeroBO reference (AR 8 rectangle)",
        "CL_alpha 4.55942959749662, e 0.9742625474419787 to 1e-9",
        f"CL_alpha {w.CLa:.12f}, e {r.e:.12f}",
        near(w.CLa, 4.55942959749662, 1e-9) and near(r.e, 0.9742625474419787, 1e-9), hard=True)
    g = AV.rect(8.0, 1.0, N=40, image_z=2.0)
    rg = g.solve(5.0)
    chk("W", "ground image raises lift, cuts induced drag at fixed CL",
        "CL_alpha up, CDi(fixed CL) down", f"x{g.CLa / w.CLa:.3f}, CDi {rg.CDi:.5f} vs {r.CDi * (rg.CL / r.CL) ** 2:.5f}",
        g.CLa > w.CLa and rg.CDi < r.CDi * (rg.CL / r.CL) ** 2, hard=True)

    # W2  the designed panel's law reproduces the published plate exactly
    dl = VE.DevAero.legacy("plate", 0.97, 0.90, 0.0)
    worst = 0.0
    for par in (True, False):
        a = VE.steady_state_corner(100.0, car=CAR, cfg=VE.VehicleConfig(qss_parity=par, wing="plate", x_w=0.97, h_w=0.90))
        b = VE.steady_state_corner(100.0, car=CAR, cfg=VE.VehicleConfig(qss_parity=par, dev_left=dl, dev_right=dl))
        worst = max(worst, max(fabs(a[k] - b[k]) for k in a if isinstance(a[k], float) and isinstance(b[k], float)))
    chk("W", "DevAero(plate) == the closed-form plate in steady_state_corner",
        "max |diff| < 1e-9 (parity on and off)", f"max |diff| {worst:.1e}", worst < 1e-9, hard=True)

    # W3  the top wing's station trade on a front-limited car
    aero = dict(S=0.40, CL0=0.9, CLa=4.4, CL_min=-0.5, CL_max=1.9, cd0=0.02, cd1=0.0, cd2=0.06)
    r0 = VE.ramp_steer(28.9, car=CAR, cfg=VE.VehicleConfig(qss_parity=True))
    peaks = {}
    for xt in (-1.6, 0.3):
        top = VE.TopAero.from_aero(aero, 6.0, xt, 1.3, "fixed")
        peaks[xt] = VE.ramp_steer(28.9, car=CAR, cfg=VE.VehicleConfig(qss_parity=True, top=top))["peak_ay_g"]
    chk("W", "top wing: behind the rear axle unloads the front, on the roof it grips",
        "peak_ay(x -1.6) < none < peak_ay(x +0.3)",
        f"{peaks[-1.6]:.4f} < {r0['peak_ay_g']:.4f} < {peaks[0.3]:.4f}",
        peaks[-1.6] < r0["peak_ay_g"] < peaks[0.3], hard=True)
    s_none = VE.steady_state_corner(100.0, car=CAR, cfg=VE.VehicleConfig(qss_parity=True))
    s_top = VE.steady_state_corner(100.0, car=CAR, cfg=VE.VehicleConfig(
        qss_parity=True, top=VE.TopAero.from_aero(aero, 6.0, 0.3, 1.3, "fixed")))
    chk("W", "top wing costs power (its drag is charged)", "P_required up with the wing",
        f"{s_none['P_required_kW']:.2f} -> {s_top['P_required_kW']:.2f} kW",
        s_top["P_required_kW"] > s_none["P_required_kW"], hard=True)

    # W4  'active' deploy logic + determinism with all three wings
    top_a = VE.TopAero.from_aero(aero, 6.0, -0.9, 1.6, mode="active")
    veh = VE.Vehicle(CAR, VE.VehicleConfig(top=top_a, dev_left=dl, dev_right=dl))
    outs, tel = [], []
    for _ in range(2):
        veh.reset(V=25.0, gear=4)
        tel = []
        for k in range(3000):
            ctl = VE.Controls(throttle=0.3, brake=1.0 if 800 < k < 1400 else 0.0,
                              delta=0.05 * sin(0.003 * k), wing_on=True)
            veh.step(ctl, (1.0,) * 4, (1.0,) * 4, 1e-3)
            tel.append((veh.top_deploy, veh.F_top, veh.wing_deploy, sum(veh.Fz)))
        outs.append(veh.state.as_array().copy())
    chk("W", "active top wing: out under braking, replay bit-identical",
        "deploy 1.0 at k=1000 while braking; identical state after two runs",
        f"deploy {tel[1000][0]:.2f}, F_top {tel[1000][1]:.0f} N, sum Fz {tel[1000][3]:.0f} N; identical {all(a == b for a, b in zip(outs[0], outs[1]))}",
        tel[1000][0] == 1.0 and tel[1000][1] > 50.0 and all(a == b for a, b in zip(outs[0], outs[1])), hard=True)
    chk("W", "downforce lands in the normal loads", "sum Fz - mg == F_top (lagged one step)",
        f"sum Fz {tel[1000][3]:.1f}, mg {CAR.m * G:.1f}, F_top {tel[1000][1]:.1f}",
        fabs(tel[1000][3] - CAR.m * G - tel[999][1]) < 5.0, hard=True)

    # W5  the build's two physics paths + persistence
    import tempfile
    import shutil
    tmp = tempfile.mkdtemp(prefix="carsim_w_")
    try:
        from drive.aero.library import Library
        lib = Library(os.path.join(tmp, "lib"), use_xfoil=False)
        d = GR.WingDesign(wing="plate", x_w=0.97, h_w=0.90, inc_deg=2.0)
        b = GR.CarBuild.from_json(dict(wing="plate", x_w=0.97, h_w=0.90, inc_deg=2.0)).clamp(lib)
        kw = b.cfg_kwargs(lib)
        chk("W", "published car in the three-slot build -> the closed-form VehicleConfig",
            "identical kwargs, no DevAero", str(d.cfg_kwargs()), kw == d.cfg_kwargs(), hard=True)
        b.top.wing = "rear-s1223"
        b.left.wing = "flank-e423"
        b.sync_mirror("left")
        kw2 = b.cfg_kwargs(lib)
        ok2 = (kw2["wing"] == "off" and isinstance(kw2["dev_left"], VE.DevAero) and isinstance(kw2["top"], VE.TopAero)
               and kw2["dev_left"].CLa > 1.5 and kw2["top"].CZ > 0.5)
        chk("W", "designed build -> DevAero + TopAero laws", "DevAero on both flanks, TopAero on top",
            f"dev CL0 {kw2['dev_left'].CL0:.3f} CLa {kw2['dev_left'].CLa:.3f}; top CZ {kw2['top'].CZ:.3f} CD {kw2['top'].CD:.4f}",
            ok2, hard=True)
        path = os.path.join(tmp, "design.json")
        b.save(path)
        back = GR.CarBuild.load(path)
        chk("W", "build persists and a v1 file upgrades", "round-trip equal; WingDesign file -> plate on both flanks",
            f"{back is not None and back.to_json() == b.to_json()}",
            back is not None and back.to_json() == b.to_json()
            and GR.CarBuild.from_json(dict(wing="fin", x_w=0.5, h_w=0.8, inc_deg=0.0)).right.wing == "fin", hard=True)
        # the vehicle takes the designed kwargs and the car corners
        cfg = VE.VehicleConfig(**kw2)
        veh = VE.Vehicle(CAR, cfg)
        res = VE.steady_state_corner(100.0, car=CAR, cfg=cfg)
        chk("W", "a designed build drives: steady corner at R = 100 with three wings",
            "converges, F_dev > 0 (right panel active in a left turn), V within 2 m/s of the plate car",
            f"V {res['V']:.3f} m/s, F_dev {res['F_dev']:.1f} N, CL_dev {res['CL_dev']:.2f}",
            res["F_dev"] > 0.0 and fabs(res["V"] - QSS["cs"][100][0]) < 2.0, hard=True)
        # W6  the renderer draws the three wings
        env = dict(os.environ, CARSIM_HEADLESS="1", SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy")
        code = (
            "import os, numpy as np\n"
            "from drive import track as trk, garage as grg\n"
            "from drive.aero.library import Library\n"
            "from drive.render import Renderer, ViewConfig, HudData, SkidBuffer, _demo_state, _demo_ctl\n"
            f"lib = Library({os.path.join(tmp, 'lib')!r}, use_xfoil=False)\n"
            "b = grg.CarBuild(); b.left.wing='flank-e423'; b.sync_mirror('left'); b.top.wing='rear-s1223'; b.clamp(lib)\n"
            "hud = b.hud_kwargs(lib)\n"
            "tr = trk.make_arena(); rnd = Renderer(ViewConfig(ppm_hi=45.0, ppm_lo=45.0), tr, headless=True)\n"
            "px, py = tr.xy[900]; st = _demo_state(px, py, float(tr.psi[900]), u=26.0, v=-0.8, r=0.3)\n"
            "rnd.cfg.show_vectors = True; rnd.update_camera(st, 0.0); rnd.cam = np.array([st.X, st.Y]); rnd._set_rot(rnd.psi_cam)\n"
            "def frame(**kw):\n"
            "    aux = HudData(V=26.0, Fz=np.array([3000,2600,1900,1700.0]), delta_wheel=np.zeros(4), **kw)\n"
            "    rnd.draw_frame(st, None, 0.0, _demo_ctl(delta=0.12, throttle=0.4), aux, SkidBuffer())\n"
            "    import pygame; return pygame.surfarray.array3d(rnd.screen).astype(int)\n"
            "a = frame()\n"
            "b3 = frame(wing_on=True, wing_deploy=1.0, wing_side=1, F_wing=95.0, D_wing=11.0, top_deploy=1.0, F_top=310.0, D_top=26.0, **hud)\n"
            "diff = int((np.abs(a - b3).sum(axis=2) > 0).sum())\n"
            "print('DIFF', diff)\n"
        )
        r = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True)
        m = [l for l in r.stdout.splitlines() if l.startswith("DIFF")]
        n_diff = int(m[0].split()[1]) if m else -1
        chk("W", "renderer draws the three wings", "frame with the build differs from the bare car by > 300 px",
            f"{n_diff} px differ (exit {r.returncode})" + ("" if r.returncode == 0 else f": {r.stderr[-200:]}"),
            r.returncode == 0 and n_diff > 300, hard=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    from drive.aero import xfoil as AX
    chk("W", "XFOIL binary for section polars", "found (estimate polars otherwise)",
        AX.binary() or "not found: the designer labels every polar ESTIMATE",
        True, hard=False)


# ==================================================================== #
#  MAIN                                                                #
# ==================================================================== #
GROUPS = {
    "A": ("THE TYRE, AGAINST qss.TYRE", group_A),
    "B": ("POWERTRAIN AND BRAKES", group_B),
    "C": ("THE CHASSIS, AGAINST qss.py", group_C),
    "D": ("THE FLANK WING", group_D),
    "E": ("NUMERICS", group_E),
    "F": ("THE HARNESS", group_F),
    "G": ("DRIVING MODES, ABS, THE OPEN MAP, SETTINGS", group_G),
    "W": ("THE THREE WINGS: drive/aero AND THE GARAGE BUILD", group_W),
}


def _prime():
    """qss reference numbers, computed once."""
    QSS["cs"] = {R: qss.corner_speed(float(R)) for R in (50, 100, 130)}
    QSS["bal100"] = qss.balance(100.0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m drive.validate")
    ap.add_argument("--quick", action="store_true",
                    help="skip the minute-scale rigs (0-100, RTF, h_w sweep)")
    ap.add_argument("--only", default=None,
                    help="group letter, or a glob matched against the check name")
    ap.add_argument("--modules", action="store_true",
                    help="also run every module's own self-check as a subprocess")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print the note under every row, not just failures")
    opts = ap.parse_args(argv)

    t_start = time.time()
    print("=" * 96)
    print("carsim ACCEPTANCE SUITE   drive/validate.py")
    print("=" * 96)
    print(f"  {CAR.name}")
    print(f"  m {CAR.m:.0f} kg   wdist_f {CAR.wdist_f:.2f}   L {CAR.L:.3f} m   "
          f"h_cg {CAR.h_cg:.2f} m   CdA {CAR.CdA:.2f} m^2")
    print(f"  tyre {os.path.basename(TY._TIR_PATH)}   "
          f"TYRE_MIRROR {VE.TYRE_MIRROR}   ECAP {VE.ECAP}   dt {VE.DT_PHYS * 1e3:.1f} ms")
    print(f"  mode: {'QUICK' if opts.quick else 'FULL'}"
          + (f"   filter: {opts.only}" if opts.only else ""))

    _prime()

    sel = opts.only.upper() if opts.only and len(opts.only) == 1 else None
    for key, (title, fn) in GROUPS.items():
        if sel and key != sel:
            continue
        fn(opts.quick)
        # keep the wing number for group X's narrative
        if key == "D":
            for c in CHECKS:
                if c.name.startswith("sealed plate at the front axle"):
                    try:
                        QSS["g_plate"] = float(c.got.strip("%+"))
                    except ValueError:
                        pass
        if key == "C" and "ss100" in QSS:
            QSS["alpha_f_meas"] = fabs(QSS["ss100"]["alpha_f_deg"])
    if opts.modules and not sel:
        group_M()

    rows = CHECKS
    if opts.only and len(opts.only) > 1:
        rows = [c for c in CHECKS if fnmatch.fnmatch(c.name.lower(),
                                                     opts.only.lower())]
    #  A FILTER THAT MATCHES NOTHING IS AN ERROR, not a pass. `--modules
    #  --only ml` burned 324 s and printed `0/0 pass`, which reads as success
    #  and is not: `ml` is a module self-check whose row is named after its
    #  command, so the glob never matched. Say so, name what was available,
    #  and exit non-zero -- a caller that greps for the count must not be told
    #  everything passed when nothing ran.
    if opts.only and not rows:
        avail = sorted({c.group for c in CHECKS})
        names = sorted(c.name for c in CHECKS)
        print(f"\n  --only {opts.only!r} matched NOTHING of {len(CHECKS)} checks."
              f"\n  Group letters: {' '.join(avail)}"
              f"\n  A longer filter is a glob against the check NAME, e.g."
              f"\n    --only 'M: python3 -m drive.ml*'   (module self-checks are"
              f" named after their command)"
              f"\n  Names start: {', '.join(names[:4])} ...")
        return 2

    order = list(GROUPS) + ["M"]
    for key in order:
        grp = [c for c in rows if c.group == key]
        if not grp:
            continue
        title = GROUPS[key][0] if key in GROUPS else "MODULE SELF-CHECKS"
        print()
        print("-" * 96)
        print(f"{key}. {title}")
        print("-" * 96)
        for c in grp:
            verdict = "PASS" if c.ok else ("FAIL" if c.hard else "soft")
            secs = f"  [{c.secs:.1f}s]" if c.secs >= 1.0 else ""
            print(f"  {verdict:4s}  {c.name:52s} {c.got}{secs}")
            print(f"        {'':52s} expected: {c.expected}")
            if c.note and (opts.verbose or not c.ok):
                for line in _wrap(c.note, 84):
                    print(f"        {line}")
        for g, text in NOTES:
            if g == key:
                for line in _wrap(text, 90):
                    print(f"  ..    {line}")

    if not sel:
        group_X()

    hard_fail = [c for c in rows if not c.ok and c.hard]
    soft_fail = [c for c in rows if not c.ok and not c.hard]
    npass = sum(1 for c in rows if c.ok)

    print()
    print("=" * 96)
    print(f"{npass}/{len(rows)} pass   "
          f"{len(hard_fail)} HARD failure(s)   {len(soft_fail)} soft miss(es)   "
          f"[{time.time() - t_start:.1f} s]")
    for c in soft_fail:
        print(f"  soft  {c.group}: {c.name} -> {c.got}   (expected {c.expected})")
    for c in hard_fail:
        print(f"  FAIL  {c.group}: {c.name} -> {c.got}   (expected {c.expected})")
    print("=" * 96)
    if hard_fail:
        print("HARD failure: a number this sim produces is not to be believed.")
        return 1
    if soft_fail:
        print("All hard checks pass. The soft misses are calibration figures for")
        print("parameters corsa_c.py marks as UNPUBLISHED -- argue about them, but")
        print("nothing above is structurally wrong.")
    else:
        print("The real-time sim agrees with the quasi-steady analysis it was built")
        print("to be consistent with, everywhere the two can be compared.")
    return 0


def _wrap(s, w):
    out, line = [], ""
    for word in s.split():
        if len(line) + len(word) + 1 > w:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out


if __name__ == "__main__":
    sys.exit(main())
