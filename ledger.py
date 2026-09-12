"""Lap-time ledger for the deployable flank wing, as a function of the
parameters that are actually uncertain.

No CFD. The dominant unknowns -- the device's installed CL and the car's own
side-force derivative Cs_psi -- are SWEPT, not estimated. If the verdict is
constant across the plausible band, the uncertainty does not matter and the
sweep is itself the result. If it flips inside the band, the sweep tells you
exactly which number is worth measuring.

Ledger structure (60 s circuit):
    corner-speed gain      over 45% of the lap
    exit-speed carry       scaled off the corner gain
    deployed device drag
    installed-mass penalty over 35% of the lap  <- this is what kills it
"""

from math import sqrt, sin, radians
from corsa_c import CorsaC, RHO, G
from crossover import gain, sustainable_V, MOUNTS

car = CorsaC()

LAP_S = 60.0
FRAC_CORNER = 0.45
FRAC_ACCEL = 0.35
S_DEV = 0.35            # m^2 per side, vertical span


def R_cap(mu, alpha_peak_deg=7.0):
    """Largest radius the car can sustain: grip AND power, incl. scrub drag.

    Lower mu cuts a_y and therefore scrub drag proportionally, so V_grip falls
    but R_cap RISES. That is why the wet is this device's best case.
    """
    V = sustainable_V(a_y_g=mu, alpha_peak_deg=alpha_peak_deg)
    return V * V / (mu * G), V


def ledger(CL_dev, dm, mu=0.87, x_w=None, R=None,
           Cs_psi=2.2, beta_deg=-3.7, accel_factor=1.0, verbose=False):
    """Net seconds per lap. Positive = faster.

    Cs_psi        car's own side-force derivative, /rad. The number you would
                  have got from RANS. Sweep it, do not trust one value.
    beta_deg      body slip at the limit; sets both the car's free side force
                  and the wing's incidence bonus.
    accel_factor  1.0 applies constant-power scaling linearly to the whole
                  accel-limited fraction. The adversarial verifier argued this
                  overstates the mass penalty ~2x, so 0.5 is the optimistic end.
    """
    x_w = MOUNTS["front axle"] if x_w is None else x_w
    Rc, _ = R_cap(mu)
    R = min(R, Rc) if R else Rc

    k = 0.5 * RHO * S_DEV * CL_dev
    g_corner = gain(k, R, x_w)
    if g_corner is None:
        return None

    t_corner = LAP_S * FRAC_CORNER
    t_accel = LAP_S * FRAC_ACCEL

    s_corner = t_corner * (1.0 - 1.0 / (1.0 + g_corner))
    s_exit = 0.15 * (g_corner / 0.016)              # scaled off the reference case
    V = sqrt(mu * G * R)
    F_drag = 0.5 * RHO * S_DEV * (CL_dev / 3.2) * V * V   # L/D ~ 3.2 for AR 1.2-1.4
    s_drag = -(F_drag * V / 1e3) * 0.023            # kW -> s, calibrated on 1.3 kW = 0.03 s
    s_mass = -t_accel * (dm / car.m) * accel_factor

    net = s_corner + s_exit + s_drag + s_mass

    if verbose:
        # the car's own apex-directed side force, for scale
        Cy_body = Cs_psi * radians(abs(beta_deg))
        F_body = 0.5 * RHO * car.A * Cy_body * V * V
        F_dev = k * V * V
        print(f"  R      {R:6.1f} m (cap {Rc:.0f})   V {V:5.1f} m/s")
        print(f"  device {F_dev:6.0f} N     body {F_body:6.0f} N "
              f"(ratio {F_body/F_dev:4.2f})")
        print(f"  corner {s_corner:+6.2f}  exit {s_exit:+5.2f}  "
              f"drag {s_drag:+5.2f}  mass {s_mass:+6.2f}   NET {net:+6.2f} s")
    return net


def sweep():
    print("=" * 72)
    print("BASELINE  1.2 16V dry, front-axle mount, +39 kg, clean fin CL=0.70")
    print("=" * 72)
    ledger(0.70, 39.0, verbose=True)

    print("\n" + "=" * 72)
    print("SWEEP 1  installed CL x installed mass   [net s/lap, dry]")
    print("=" * 72)
    masses = (28, 34, 39, 45)
    print(f"{'CL':>6s}" + "".join(f"{m:>10d} kg" for m in masses))
    for CL in (0.55, 0.70, 0.85, 1.00, 1.25):
        row = f"{CL:6.2f}"
        for m in masses:
            n = ledger(CL, m)
            row += f"{n:+10.2f}  " if n is not None else "       n/a  "
        print(row)

    print("\n" + "=" * 72)
    print("SWEEP 2  Cs_psi -- the number CFD would have given you")
    print("=" * 72)
    print("The car's own side force does NOT enter the device's own gain: it")
    print("shifts the BASELINE both configurations share, so it cancels out of")
    print("the delta to first order. It matters for absolute corner speed, not")
    print("for the go/no-go. Shown here so you can say that with a number.\n")
    print(f"{'Cs_psi':>8s} {'body F':>9s} {'device F':>9s} {'ratio':>7s} {'net s/lap':>11s}")
    for Cs in (1.0, 1.5, 2.2, 3.0):
        Rc, _ = R_cap(0.87)
        V = sqrt(0.87 * G * Rc)
        Cy = Cs * radians(3.7)
        F_body = 0.5 * RHO * car.A * Cy * V * V
        F_dev = 0.5 * RHO * S_DEV * 0.70 * V * V
        n = ledger(0.70, 39.0, Cs_psi=Cs)
        print(f"{Cs:8.1f} {F_body:8.0f} N {F_dev:8.0f} N {F_body/F_dev:7.2f} "
              f"{n:+11.2f}")

    print("\n" + "=" * 72)
    print("SWEEP 3  where it actually pays")
    print("=" * 72)
    cases = [
        ("Corsa 1.2, dry,  clean fin", dict(CL_dev=0.70, dm=39, mu=0.87)),
        ("Corsa 1.2, dry,  sealed plate", dict(CL_dev=1.25, dm=39, mu=0.87)),
        ("Corsa 1.2, WET,  clean fin", dict(CL_dev=0.70, dm=39, mu=0.55)),
        ("Corsa 1.2, WET,  sealed plate", dict(CL_dev=1.25, dm=39, mu=0.55)),
        ("... at the CG instead", dict(CL_dev=1.25, dm=39, mu=0.55,
                                       x_w=MOUNTS["at the CG"])),
        ("... at the rear axle", dict(CL_dev=1.25, dm=39, mu=0.55,
                                      x_w=MOUNTS["rear axle"])),
        ("optimistic mass model", dict(CL_dev=1.25, dm=28, mu=0.87,
                                       accel_factor=0.5)),
    ]
    for label, kw in cases:
        n = ledger(**kw)
        verdict = "GO " if n > 0 else "NO-GO"
        print(f"  {label:32s} {n:+6.2f} s/lap   {verdict}")


if __name__ == "__main__":
    sweep()
