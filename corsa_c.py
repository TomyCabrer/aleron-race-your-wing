"""Opel Corsa C 1.2 16V (Z12XE), 2003 facelift, 5-door, one occupant.

Convention lock
---------------
Mass is EU kerb  = DIN kerb + 75 kg driver + 90% fuel.
Air density      rho = 1.2 kg/m^3.
Angles in degrees at the interface, radians internally.

"est" in the confidence column means NOT PUBLISHED. Opel never released
axle weights, CG height or inertias for this car. Those values are
bottom-up estimates with the stated band; treat them as parameters to be
calibrated, not as data.
"""

from dataclasses import dataclass, field


@dataclass
class CorsaC:
    name: str = "Opel Corsa C 1.2 16V (2003)"

    # --- mass and geometry ---------------------------------------------
    m: float = 1010.0          # kg    running mass (ADAC EU Leergewicht)
    wdist_f: float = 0.61      # est +/-0.02   front weight fraction
    L: float = 2.491           # m     wheelbase          (published)
    t_f: float = 1.429         # m     front track        (1.415 on 1.8 SRi/GSi)
    t_r: float = 1.420         # m     rear track         (1.410 on 1.8 SRi/GSi)
    h_cg: float = 0.55         # est +/-0.04   CG height, h/H = 0.38

    # --- inertias (all estimates; band in the comment) ------------------
    Izz: float = 1200.0        # est   kg m^2   band 1000-1500, DI = 0.80
    Ixx: float = 267.0         # est   kg m^2   band 210-330
    Iyy: float = 1200.0        # est   kg m^2
    m_s: float = 892.0         # est   kg       sprung
    m_us_f: float = 56.0       # est   kg
    m_us_r: float = 62.0       # est   kg

    # --- tyres ----------------------------------------------------------
    tyre: str = "175/65R14 82T on 5.5Jx14 ET49"   # 185/55R15 on 1.8 SRi/GSi
    r_roll: float = 0.283      # m     rolling radius (OD 583.1 mm x 0.97)

    # --- aerodynamics ----------------------------------------------------
    Cd: float = 0.32           # published, 4 sources. 0.30 is a Corsa D figure
    A: float = 2.01            # m^2   published
    CdA: float = 0.66          # m^2   band 0.64-0.70, validated vs Vmax
    Crr: float = 0.012         # est

    # --- powertrain -------------------------------------------------------
    gear: tuple = (3.545, 2.143, 1.429, 1.121, 0.892)   # F13 CR
    gear_rev: float = 3.308
    finaldrive: float = 3.94   # Z12XE (Z14XE 4.29, Z18XE 3.737, Z13DT 3.550)
    eta_drive: float = 0.86    # derived from the top-speed power balance
    P_max: float = 55e3        # W  @ 5600 rpm
    T_max: float = 110.0       # Nm @ 4000 rpm
    Vmax: float = 47.2         # m/s (170 km/h -- NOT 165, that is the 1.3 CDTi)
    steer_ratio: float = 16.0  # est +/-1.5 (derived 15.86:1 from 2.9 turns)

    # --- suspension: EVERY VALUE HERE IS AN ESTIMATE -----------------------
    k_wheel_f: float = 17.5e3  # N/m
    k_wheel_r: float = 11.06e3 # N/m
    k_tyre: float = 200e3      # N/m
    h_rc_f: float = 0.075      # m   MacPherson
    h_rc_r: float = 0.300      # m   twist beam
    Kphi_f: float = 311.0      # Nm/deg  springs only
    Kphi_r: float = 195.0      # Nm/deg  springs only
    Kphi_tot: float = 640.0    # Nm/deg  with ARBs -> 4.7-5.1 deg/g
    rollsteer_r: float = 1.0   # est  deg toe-out per deg roll (twist beam)
    rollcamber_r: float = 1.0  # est  deg adverse outer camber per deg roll

    # --- MISSING FROM EVERY SOURCE. Do not ship a lap sim without them. ----
    brakes: str = "MISSING - no disc/drum dia, pad mu or F/R torque split"
    dampers: str = "MISSING - no rates, no damping ratios"

    # --- derived ------------------------------------------------------------
    @property
    def a(self) -> float:
        """CG to front axle, m."""
        return (1.0 - self.wdist_f) * self.L

    @property
    def b(self) -> float:
        """CG to rear axle, m."""
        return self.wdist_f * self.L

    @property
    def t(self) -> float:
        """Mean track, m."""
        return 0.5 * (self.t_f + self.t_r)

    @property
    def P_wheel(self) -> float:
        """Power at the wheels, W."""
        return self.eta_drive * self.P_max


RHO = 1.2
G = 9.81


def self_check(car: CorsaC = CorsaC()) -> None:
    """Gearing and aero cross-checks. Any parameter set that fails these is wrong."""
    # 5th x FD -> km/h per 1000 rpm
    ratio = car.gear[-1] * car.finaldrive
    kmh_per_1000 = (1000.0 / ratio) * 2 * 3.141592653589793 * car.r_roll * 60 / 1000
    rpm_at_vmax = car.Vmax * 3.6 / kmh_per_1000 * 1000
    print(f"gearing : {kmh_per_1000:5.1f} km/h per 1000 rpm in 5th")
    print(f"          {rpm_at_vmax:6.0f} rpm at Vmax vs 5600 rpm power peak")

    F = 0.5 * RHO * car.CdA * car.Vmax**2 + car.Crr * car.m * G
    print(f"aero    : {car.Vmax * F / 1e3:5.1f} kW at wheels to hold Vmax "
          f"vs {car.P_wheel/1e3:.1f} kW available")


if __name__ == "__main__":
    self_check()
