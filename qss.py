"""Quasi-steady-state two-track cornering solver for the Corsa C flank-wing study.

Why not Chrono::Vehicle: every projectchrono conda build requires macOS >= 26
(this machine is 15.5), and the osx-64 fallback stops at pychrono 8.0.0. More
to the point, no brake or damper data exists for a Corsa C in any source, so a
full multibody vehicle model could not be honestly parameterised anyway. This
model needs only parameters we actually have.

What it does that a point-mass sim cannot
-----------------------------------------
  * per-CORNER normal loads, so tyre load sensitivity acts where it really acts
  * yaw equilibrium, so the aero force's station x_w matters
  * front/rear axle limits separately, so a FWD car is front-limited
  * the (x_w + b)/b multiplier EMERGES from the moment balance -- it is not
    imposed. Agreement with the closed form is therefore a real cross-check.

Equilibrium solved at each speed
--------------------------------
    Y_f + Y_r + F      = m a_y                       (lateral force)
    Y_f a - Y_r b + F x_w = 0                        (yaw moment about CG)
  ->  Y_f = [ b m a_y - F (b + x_w) ] / L
      Y_r = [ a m a_y + F x_w ] / L

    dFz_tot = ( m a_y h_cg - F h_w ) / t             (moments about the outer
                                                      contact line; frame
                                                      independent -- the roll
                                                      axis sets only the split)
"""

from math import sqrt, sin, radians
from corsa_c import CorsaC, RHO, G

car = CorsaC()

# tyre: mu(Fz) = mu_ref + s (Fz - Fz_ref)   TNO 205/60R15, measured
TYRE = dict(mu_ref=0.903, Fz_ref=2477.0, s=-16.1e-6)


def fy_max(Fz, mu_ref, Fz_ref, s, mu_scale=1.0):
    """Peak lateral force of one tyre at vertical load Fz. Zero if unloaded."""
    if Fz <= 0.0:
        return 0.0
    mu = mu_scale * (mu_ref + s * (Fz - Fz_ref))
    return max(mu, 0.0) * Fz


def tyre_ref(lfzo=1.0):
    """The `TYRE`-shaped reference for a tyre whose load scale is `lfzo`.

    Task 41. `TYRE` is the Corsa's tyre, and it stays THE reference: 1.0
    returns `TYRE` itself -- the same dict object, so every caller that
    passes `**tyre_ref(car's scale)` on a car tyre computes exactly what it
    computed with `**TYRE`. A LOAD-SCALED tyre (drive/tyre.py, "Load
    scaling": the same coefficient set on a tyre rated `lfzo` times higher,
    a truck's) has the same law with the load axis stretched:

        mu(Fz) = mu_ref + (s / lfzo) * (Fz - lfzo * Fz_ref)

    which is exact, not fitted -- the MF D-term is linear in Fz/(LFZO*FNOMIN)
    and `TYRE` is its transcription -- so a bus's utilisation is its own
    tyre's, and at the equivalent load it reads what the Corsa's would.
    """
    lfzo = float(lfzo)
    if lfzo == 1.0:
        return TYRE
    return dict(mu_ref=TYRE["mu_ref"], Fz_ref=TYRE["Fz_ref"] * lfzo,
                s=TYRE["s"] / lfzo)


def car_tyre_refs(car):
    """(front, rear) `tyre_ref`s for a car: its declared per-axle load
    scales (`cars.CarSpec.tyre_lfzo_f/_r`), 1.0 on any car that has none --
    and then both are `TYRE` itself."""
    return (tyre_ref(getattr(car, "tyre_lfzo_f", 1.0)),
            tyre_ref(getattr(car, "tyre_lfzo_r", 1.0)))


def axle_capacity(Fz_static_axle, dFz_axle, mu_scale=1.0, tyre=None):
    """Sum of both tyres' peak lateral force on one axle, after load transfer.

    `tyre` is a `tyre_ref` dict; None is `TYRE`, the study's own tyre."""
    t = TYRE if tyre is None else tyre
    out = Fz_static_axle / 2.0 + dFz_axle
    inn = Fz_static_axle / 2.0 - dFz_axle
    return (fy_max(out, mu_scale=mu_scale, **t)
            + fy_max(inn, mu_scale=mu_scale, **t))


def roll_angle(a_y, F, h_w):
    """Body roll, deg. The wing's term is where its roll benefit actually lives.

    Total lateral load transfer is a ground-plane balance and is frame
    independent -- the roll AXIS does not enter it. What the roll axis sets is
    the roll ANGLE, and on a twist-beam rear axle roll angle maps ~1:1 to
    adverse outer-wheel camber. That is the only reason the roll term belongs
    in this model at all.
    """
    h_ra = 0.160                      # roll axis height at the CG station, m
    M = car.m_s * a_y * (car.h_cg - h_ra) - F * (h_w - h_ra)
    return M / car.Kphi_tot


def residuals(a_y, V, k, x_w, h_w, roll_dist_f, mu_scale, c_cam=0.0):
    """Utilisation of each axle. 1.0 = at the limit.

    c_cam: fractional rear grip lost per degree of body roll (adverse camber
    and roll toe-out on the twist beam). DEFAULT 0.0 -- disabled. Derating the
    rear pushes the car toward OVERSTEER, which is the wrong direction for a
    FWD hatch, so this term as written does not model what it claims. Left in
    place, off, until it can be signed correctly against measured data.

    The understeer that matters here is set instead by roll_dist_f, the front
    share of roll stiffness. 0.74 against a 0.61 front weight split gives a
    4.8% front margin and a_y = 0.862 g -- a properly understeering FWD hatch,
    and the cap sqrt(mu_r/mu_f) = +2.4% that bounds the whole study.
    """
    F = k * V * V
    L = car.L
    W = car.m * G

    Y_f = (car.b * car.m * a_y - F * (car.b + x_w)) / L
    Y_r = (car.a * car.m * a_y + F * x_w) / L

    dFz_tot = (car.m * a_y * car.h_cg - F * h_w) / car.t
    phi = roll_angle(a_y, F, h_w)
    derate = max(1.0 - c_cam * phi, 0.1)

    cap_f = axle_capacity(W * car.wdist_f, dFz_tot * roll_dist_f, mu_scale)
    cap_r = axle_capacity(W * (1 - car.wdist_f), dFz_tot * (1 - roll_dist_f),
                          mu_scale * derate)
    return Y_f / cap_f, Y_r / cap_r


def max_ay(V, k=0.0, x_w=0.0, h_w=0.90, roll_dist_f=0.74, mu_scale=1.0,
           c_cam=0.0):
    """Largest sustainable lateral acceleration at speed V. Bisection on a_y."""
    lo, hi = 0.1, 30.0
    for _ in range(200):
        a = 0.5 * (lo + hi)
        uf, ur = residuals(a, V, k, x_w, h_w, roll_dist_f, mu_scale, c_cam)
        lo, hi = (a, hi) if max(uf, ur) < 1.0 else (lo, a)
    return 0.5 * (lo + hi)


def corner_speed(R, k=0.0, x_w=0.0, h_w=0.90, mu_scale=1.0, power_cap=True,
                 alpha_peak_deg=7.0, c_cam=0.0):
    """Fastest speed through a constant-radius corner.

    Grip limit: V^2/R = max_ay(V), damped fixed point.
    Power limit: scrub drag m a_y sin(alpha_peak) dominates aero drag at these
    speeds and is the term naive analyses omit entirely.
    """
    V = 5.0
    for _ in range(300):
        a = max_ay(V, k, x_w, h_w, mu_scale=mu_scale, c_cam=c_cam)
        V += 0.3 * (sqrt(a * R) - V)
    limited = "grip"

    if power_cap:
        # At the power limit the car is NOT at the grip limit, so scrub drag
        # must be charged at the ACTUAL a_y = Vp^2/R, not at max_ay. Using
        # max_ay here makes the wing look harmful in the power-limited regime
        # purely because it raises the grip it never gets to use.
        roll = car.Crr * car.m * G
        ld_dev = 3.2                      # device L/D, AR 1.2-1.4 fin
        lo, hi = 1.0, 120.0
        for _ in range(200):
            Vp = 0.5 * (lo + hi)
            a_act = min(Vp * Vp / R,
                        max_ay(Vp, k, x_w, h_w, mu_scale=mu_scale, c_cam=c_cam))
            scrub = car.m * a_act * sin(radians(alpha_peak_deg))
            drag = 0.5 * RHO * car.CdA * Vp * Vp + (k / ld_dev) * Vp * Vp
            P = Vp * (scrub + drag + roll)
            lo, hi = (Vp, hi) if P < car.P_wheel else (lo, Vp)
        Vp = 0.5 * (lo + hi)
        if Vp < V:
            V, limited = Vp, "power"
    return V, limited


def balance(R=100.0, k=0.0, x_w=0.0, h_w=0.90, c_cam=0.0):
    """Diagnostic: which axle limits, and by how much margin."""
    V, _ = corner_speed(R, k, x_w, h_w, power_cap=False, c_cam=c_cam)
    a = max_ay(V, k, x_w, h_w, c_cam=c_cam)
    uf, ur = residuals(a, V, k, x_w, h_w, 0.74, 1.0, c_cam)
    return dict(V=V, ay_g=a / G, util_f=uf, util_r=ur,
                phi=roll_angle(a, k * V * V, h_w),
                limits="front" if uf >= ur else "rear",
                margin=1.0 - min(uf, ur))


# ------------------------------------------------------------------ checks --
def cross_check():
    """The multiplier emerges from the moment balance. Compare with the closed
    form V/V0 = 1/sqrt(1 - (kR/m)(x_w+b)/b)."""
    from crossover import gain, MOUNTS
    k = 0.5 * RHO * 0.35 * 0.70

    print("=" * 74)
    print("CROSS-CHECK  two-track solver vs closed form   (R = 100 m, grip only)")
    print("=" * 74)
    print(f"{'mount':16s} {'x_w':>6s} {'QSS gain':>10s} {'closed form':>12s} {'diff':>8s}")
    R = 100.0
    V0, _ = corner_speed(R, k=0.0, power_cap=False)
    for name, x_w in MOUNTS.items():
        V, _ = corner_speed(R, k=k, x_w=x_w, power_cap=False)
        g_qss = V / V0 - 1.0
        g_cf = gain(k, R, x_w)
        print(f"{name:16s} {x_w:+6.2f} {g_qss*100:9.2f}% {g_cf*100:11.2f}% "
              f"{(g_qss-g_cf)*100:+7.2f}%")
    print("\nThe two agree where the closed form is valid. Where they diverge,")
    print("the QSS number is the right one: it carries per-corner load")
    print("sensitivity and separate axle limits, which the closed form lumps.")


def sweep():
    from crossover import MOUNTS
    k_fin = 0.5 * RHO * 0.35 * 0.70
    k_plate = 0.5 * RHO * 0.35 * 1.25
    x_w = MOUNTS["front axle"]

    print("\n" + "=" * 74)
    print("CORNER SPEED vs RADIUS   (power cap on, front-axle mount)")
    print("=" * 74)
    print(f"{'R (m)':>6s} {'baseline':>18s} {'clean fin':>18s} {'sealed plate':>20s}")
    for R in (30, 50, 75, 100, 130, 175, 250):
        V0, l0 = corner_speed(R)
        V1, l1 = corner_speed(R, k=k_fin, x_w=x_w)
        V2, l2 = corner_speed(R, k=k_plate, x_w=x_w)
        print(f"{R:6d} {V0:7.2f} m/s ({l0:5s}) "
              f"{V1:7.2f} ({(V1/V0-1)*100:+5.2f}%) "
              f"{V2:7.2f} ({(V2/V0-1)*100:+5.2f}%)")

    print("\n" + "=" * 74)
    print("WET   mu scaled to 0.55/0.87 = 0.632")
    print("=" * 74)
    ms = 0.55 / 0.87
    print(f"{'R (m)':>6s} {'baseline':>18s} {'clean fin':>18s} {'sealed plate':>20s}")
    for R in (50, 100, 175, 250, 350):
        V0, l0 = corner_speed(R, mu_scale=ms)
        V1, l1 = corner_speed(R, k=k_fin, x_w=x_w, mu_scale=ms)
        V2, l2 = corner_speed(R, k=k_plate, x_w=x_w, mu_scale=ms)
        print(f"{R:6d} {V0:7.2f} m/s ({l0:5s}) "
              f"{V1:7.2f} ({(V1/V0-1)*100:+5.2f}%) "
              f"{V2:7.2f} ({(V2/V0-1)*100:+5.2f}%)")

    print("\n" + "=" * 74)
    print("MOUNT HEIGHT   (R = 100 m, sealed plate) -- sign flips at h_w = h_cg")
    print("=" * 74)
    V0, _ = corner_speed(100.0, power_cap=False)
    for h_w in (0.30, 0.45, 0.55, 0.70, 0.90, 1.10):
        V, _ = corner_speed(100.0, k=k_plate, x_w=x_w, h_w=h_w, power_cap=False)
        tag = "  <- h_cg, load transfer unchanged" if abs(h_w - car.h_cg) < 1e-9 else ""
        print(f"  h_w = {h_w:4.2f} m   {V:6.2f} m/s   {(V/V0-1)*100:+5.2f}%{tag}")


if __name__ == "__main__":
    cross_check()
    sweep()
