"""Kill-or-continue check for the deployable lateral flank wing.

Run this BEFORE writing any simulator code. It answers three questions in
closed form, and if question 1 fails the rest of the project is a pivot to a
conventional (or canted) downforce wing.

  1. Does a lateral wing beat an equal-force downforce wing on this tyre?
  2. What radius can the car actually SUSTAIN (grip AND power, incl. scrub drag)?
  3. What is the corner-speed gain there, given the mounting station?

Governing group, derived
------------------------
    m V^2 / R = mu m g + k V^2,   k = 0.5 rho S CL
    V / V0    = 1 / sqrt(1 - kR/m)                 mu and g cancel

    at the grip limit R = V^2/(mu g), so   kR/m == F_aero / (mu m g)
    -> the group is the FRACTION OF THE TYRES' WORK the wing takes over.
       Radius is a proxy for speed. To first order  dV/V ~ 0.5 F/(mu W).

    On a front-limited car the group is multiplied by (x_w + b)/b, and the
    whole effect is capped at sqrt(mu_r/mu_f), the understeer margin.
"""

from math import sqrt, sin, radians
from corsa_c import CorsaC, RHO, G

car = CorsaC()

# ---------------------------------------------------------------- tyres ----
# mu(Fz) = mu_ref + s*(Fz - Fz_ref).   s = dmu/dFz, per newton.
# The three free road-tyre datasets in circulation collapse onto these fits;
# averaging them is double-counting, they share provenance.
TYRES = {
    "TNO 205/60R15 (measured)": dict(mu_ref=0.903, Fz_ref=2477.0, s=-16.1e-6),
    "mid-band":                 dict(mu_ref=0.950, Fz_ref=2477.0, s=-30.0e-6),
    "MSC Pac2002 235/60R16":    dict(mu_ref=1.137, Fz_ref=2477.0, s=-37.2e-6),
}


def marginal_mu(mu_ref, Fz_ref, s, Fz):
    """d(mu*Fz)/dFz -- what a DOWNFORCE wing actually buys per newton.

    Evaluate at m*g/4, NOT at the front-outer wheel. Because mu is linear in
    Fz, Fy is exactly quadratic, so mean(dY/dFz) == dY/dFz at the mean load,
    independent of how load transfer is split. Using the 5000 N front-outer
    value overstates the lateral wing's advantage by ~1.7x.
    """
    mu = mu_ref + s * (Fz - Fz_ref)
    return mu + Fz * s


def q1_crossover():
    print("=" * 68)
    print("Q1  LATERAL vs DOWNFORCE  (a lateral wing buys grip at 1.000 N/N)")
    print("=" * 68)
    Fz = car.m * G / 4.0
    print(f"evaluated at Fz = m*g/4 = {Fz:.0f} N\n")
    print(f"{'tyre':28s} {'mu':>6s} {'dY/dFz':>8s} {'advantage':>10s} "
          f"{'crossover':>10s}  verdict")
    for name, p in TYRES.items():
        mu = p["mu_ref"] + p["s"] * (Fz - p["Fz_ref"])
        marg = marginal_mu(Fz=Fz, **p)
        adv = 1.0 / marg
        xover = 1.0 + abs(p["s"]) * Fz          # mu_peak below which lateral wins
        ok = "LATERAL WINS" if marg < 1.0 else "downforce wins -- PIVOT"
        print(f"{name:28s} {mu:6.3f} {marg:8.3f} {adv:9.3f}x {xover:10.3f}  {ok}")
    print("\nCrossover condition:  mu_peak(mg/4) + (mg/4)*dmu/dFz  <  1")
    print("i.e. mu_peak < 1 + |dmu/dFz|*(mg/4).  NOT the naive mu < 1.")
    print("The lateral wing adds no vertical load, so its k is untouched by")
    print("load sensitivity. That immunity is the cleanest argument for it.")


# ------------------------------------------------------- accessible radius --
def sustainable_V(a_y_g=0.87, alpha_peak_deg=7.0):
    """Fastest corner the car can HOLD. Scrub drag dominates aero here and is
    the term every naive analysis omits:

        P_wheel = V * [ m*a_y*sin(alpha_peak) + 0.5*rho*CdA*V^2 + Crr*m*g ]
    """
    a_y = a_y_g * G
    scrub = car.m * a_y * sin(radians(alpha_peak_deg))
    roll = car.Crr * car.m * G
    lo, hi = 1.0, 100.0
    for _ in range(200):                       # bisection on the power balance
        V = 0.5 * (lo + hi)
        P = V * (scrub + 0.5 * RHO * car.CdA * V * V + roll)
        lo, hi = (V, hi) if P < car.P_wheel else (lo, V)
    return 0.5 * (lo + hi)


def q2_radius(a_y_g=0.87):
    print("\n" + "=" * 68)
    print("Q2  ACCESSIBLE RADIUS  (grip AND power, incl. scrub drag)")
    print("=" * 68)
    for alpha in (6.0, 7.0, 8.6):
        V = sustainable_V(a_y_g, alpha)
        R = V * V / (a_y_g * G)
        print(f"alpha_peak = {alpha:4.1f} deg -> V = {V:5.1f} m/s "
              f"({V*3.6:5.1f} km/h),  R_cap = {R:6.1f} m")
    print("\nBeyond R_cap the car is POWER limited, not grip limited, and the")
    print("wing does nothing. Truncate every R sweep here or you will report")
    print("gains at radii this car cannot reach.")


# --------------------------------------------------------------- the gain --
MOUNTS = {                      # x_w, m, positive forward of the CG
    "front bumper":  1.74,
    "front axle":    0.97,
    "mid front door": 0.40,
    "at the CG":     0.00,
    "rear axle":    -1.52,
}


def gain(k, R, x_w, m=None, b=None):
    """Fractional corner-speed gain. Returns None if the group runs away."""
    m = m or car.m
    b = b or car.b
    group = (k * R / m) * (x_w + b) / b
    if group >= 1.0:
        return None
    return 1.0 / sqrt(1.0 - group) - 1.0


def q3_gain(understeer_margin=0.05):
    print("\n" + "=" * 68)
    print("Q3  CORNER-SPEED GAIN")
    print("=" * 68)
    # k = 0.5 * rho * S * CL ;  S = 0.35 m^2
    S = 0.35
    ks = {"clean fin  CL=0.70": 0.5 * RHO * S * 0.70,
          "sealed plate C=1.25": 0.5 * RHO * S * 1.25}
    cap = sqrt(1.0 + understeer_margin) - 1.0

    print(f"S = {S} m^2, vertical span (lift is perpendicular to freestream")
    print("AND span -- a horizontal-span panel makes DOWNFORCE, not side force)\n")
    print("mounting-station multiplier (x_w + b)/b:")
    for name, x_w in MOUNTS.items():
        print(f"  {name:16s} x_w = {x_w:+5.2f} m   x{(x_w + car.b)/car.b:5.2f}")

    for kname, k in ks.items():
        print(f"\n{kname}   k = {k:.3f} kg/m,  front-axle mount")
        print(f"  {'R (m)':>7s} {'gain':>8s}")
        for R in (50, 100, 130, 200, 400):
            g = gain(k, R, MOUNTS["front axle"])
            flag = ""
            if g is not None and g > cap:
                flag = f"  <- ABOVE the sqrt(mu_r/mu_f) cap (+{cap*100:.2f}%)"
            if R > 130:
                flag += "  [1.2 16V cannot reach]"
            print(f"  {R:7d} {g*100:7.2f}%{flag}")

    print(f"\nHard cap = sqrt(mu_r/mu_f) - 1 = +{cap*100:.2f}% at a "
          f"{understeer_margin*100:.0f}% understeer margin.")
    print("As the front is relieved the rear becomes limiting. Put this as an")
    print("assertion in the lap sim.")


if __name__ == "__main__":
    q1_crossover()
    q2_radius()
    q3_gain()
