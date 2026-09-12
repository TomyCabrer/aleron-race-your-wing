# drive/ — interface contract (AUTHORITATIVE)

Real-time drivable simulator for the Opel Corsa C 1.2 16V flank-wing study.
Every module below is written against THIS file. Where a subsystem spec in
`../specs/*.txt` disagrees with this file, THIS FILE WINS. The reconciliations
are recorded at the bottom.

Package root: `/Users/tolomeelsabio/Desktop/carsim`, package `drive/`.
Run as `python3 -m drive.<module>`. `corsa_c.py`, `qss.py`, `crossover.py`,
`ledger.py` live one level up and are importable because the package is run
from the repo root.

## 0. Global conventions — non-negotiable

* **Frame**: ISO body frame. `x` forward, `y` **LEFT**, `z` up. Yaw `psi` and
  yaw rate `r` positive counter-clockwise seen from above = **left turn**.
  `a_y` positive = leftward. Roll `phi` positive about `+x` (right-hand rule)
  = body leans to the RIGHT = outside of a left turn.
* **Wheel order is always `FL, FR, RL, RR` = index 0,1,2,3.** Every 4-tuple
  and every `(4,)` array in every module uses this order. No exceptions.
* **Steer**: `delta > 0` steers LEFT. Road-wheel angle in **radians**.
  Only the front wheels steer.
* **Slip**: `kappa = Vsx / |Vx|` with `Vsx = omega*R_e - Vx`. Positive kappa
  drives the car forward (`Fx > 0`). `alpha = atan2(Vy, |Vx|)`, positive alpha
  gives `Fy < 0` (restoring). Both are computed as **transient (relaxed)**
  states, never formed directly as a quotient — see §4.
* **Angles**: radians internally, degrees only at display/CLI boundaries.
* **Units**: SI. N, m, s, kg, rad, N·m. rpm only at the HUD/CLI boundary.
* **No `Date.now()`-style nondeterminism in the physics path.** `t = n_steps*dt`,
  never accumulated. No wall clock, no unseeded RNG, no dict-iteration-order
  dependence inside `step()`.

## 1. Module map and ownership

| file | owns | may import |
|---|---|---|
| `drive/tyre.py` | Magic Formula 6.2 combined slip from `.tir` | `corsa_c` |
| `drive/powertrain.py` | engine, clutch, gearbox, diff, brakes | `corsa_c`, scipy |
| `drive/vehicle.py` | EOM, load transfer, roll, aero+wing, integrator | `tyre`, `powertrain`, `corsa_c` |
| `drive/track.py` | track geometry, projection, surfaces | numpy |
| `drive/input.py` | keyboard/gamepad → `Controls` | pygame |
| `drive/render.py` | pygame drawing + HUD | `track`, `qss`, pygame |
| `drive/telemetry.py` | CSV logging | csv |
| `drive/plots.py` | matplotlib post-run plots (Agg) | matplotlib, numpy |
| `drive/garage.py` | 3D garage: `CarBuild` (three wing slots) -> `VehicleConfig` kwargs; designer / airfoil / library pages | `corsa_c`, `crossover`, `input`, `menu`, `garage_ui`, `aero`, `vehicle` (the two aero dataclasses only), pygame |
| `drive/garage_ui.py` | widget kit for the garage pages (params, lists, plots, prompt) | pygame, numpy |
| `drive/aero/` | wing-design physics: sections, panel method, polars (XFOIL / estimate), vortex lattice, GP-BO, the library | numpy, scipy, the `xfoil` binary if present |
| `drive/menu.py` | pause / help menu overlay (ESC, OPTIONS); pure UI | pygame only |
| `drive/audio.py` | procedural car sound: `Synth` (numpy) + `CarSound` (one pygame.mixer channel); a render-loop consumer of `HudData`, never an input | numpy, pygame |
| `drive/drive.py` | main loop, CLI, scripted runs | everything |
| `drive/validate.py` | the whole acceptance suite | everything |

`vehicle.py` NEVER imports `track`, `render`, `input`, `aero` or pygame: the
designed wings reach it as two frozen dataclasses of numbers (`DevAero`,
`TopAero`, section 4) that the garage builds. `drive/aero` NEVER imports
pygame; the 1 kHz step never calls a solver.
`render.py` NEVER touches physics state except read-only.

## 2. `drive/tyre.py`

Source data: `tyre_data/TNO_car205_60R15.tir` (FITTYP=62, MF-Tyre/MF-Swift 6.2).
This file **is** the provenance of `qss.TYRE`: its `PDY1=0.8784`,
`PDY2=-0.06445`, `FNOMIN=4000` give `mu_y(2477) = 0.90294`, slope `-16.11e-6/N`,
which is `qss.TYRE = dict(mu_ref=0.903, Fz_ref=2477.0, s=-16.1e-6)` rounded.

**Do NOT rescale FNOMIN or LFZO.** `LFZO = LMUX = LMUY = LKX = LKY = 1.0`.
Rescale geometry only: `R0 = 0.2915`, width `0.175`, aspect `0.65`,
rim radius `0.1778` (175/65R14).

**Symmetrisation is mandatory.** Zero these 13 camber-EVEN shifts:
`PHY1 PHY2 PVY1 PVY2 PHX1 PHX2 PVX1 PVX2 QHZ1 QHZ2 QDZ6 QDZ7 QSX1`.
Keep every camber-ODD shift (`PVY3 PVY4 QHZ3 QHZ4 QDZ8..QDZ11`) and keep
`RHX1 RHY1 RHY2`. After this, `peak|Fy|(Fz) == mu_y(Fz)*Fz` to 5 s.f. and the
model is exactly odd in `(kappa, alpha)`.

```python
def load_tir(path: str) -> dict[str, float]
class TyreModel:
    def __init__(self, tir_path, *, symmetrise=True, R0=0.2915, width=0.175,
                 LFZO=1.0, LMUX=1.0, LMUY=1.0, LKY=1.0, LKX=1.0,
                 CFX=381913.4, CFY=157632.5): ...
    def evaluate(self, Fz, kappa, alpha, gamma=0.0, mu_scale=1.0) -> tuple[float,float,float]
        # -> (Fx, Fy, Mz) in the WHEEL-CARRIER frame, N, N, N.m
    def stiffnesses(self, Fz, gamma=0.0) -> tuple[float, float]      # (Kxk, Kya); Kya < 0
    def relax_lengths(self, Fz, gamma=0.0) -> tuple[float, float]    # (sigma_kappa, sigma_alpha)
    def mu_y(self, Fz, mu_scale=1.0) -> float
    def mu_x(self, Fz, mu_scale=1.0) -> float
    def peak_fy(self, Fz, mu_scale=1.0) -> float                     # analytic mu_y*Fz
CORSA_TYRE: TyreModel      # module-level singleton, parsed ONCE at import
def self_check() -> None   # python3 -m drive.tyre
```

* `evaluate` must be pure and allocation-light. Unpack coefficients into
  `__slots__` attributes; no dict lookups in the hot path. Target < 3 µs/call.
* `Fz <= 50 N` → return `(0.0, 0.0, 0.0)`.
* `Fz` clamped to `[0, 10000]`.
* `evaluate(Fz, 0, 0, 0, 1)` must be **exactly** `(0.0, 0.0, 0.0)`.
* Odd symmetry: `evaluate(Fz, -k, -a) == -evaluate(Fz, k, a)` to 1e-9.
* **Relaxation is NOT done here.** `vehicle.py` owns the transient slip states
  because the integrator ordering is load-bearing (§4). `tyre.py` exposes
  `relax_lengths()` and `stiffnesses()` so `vehicle.py` can do it. Do not
  export a `step_contact` that integrates state.
* `Mz` sign: positive (restoring) for positive alpha; `vehicle.py` flips it in
  reverse, not `tyre.py`.
* The `sigma` floors are mandatory: `sigma_kappa >= 0.05`, `sigma_alpha >= 0.10`.
* `mu_scale` must be fused into `LMUX`/`LMUY` at the top of `evaluate` and used
  in **every** one of `Dx, Dy, SVyg, DVyk, Bt, Br, Dr` — `Bt` and `Br` DIVIDE
  by it.

Full equation list (E1–E39, R1) and every golden number: `../specs/tyre.txt`.

## 3. `drive/powertrain.py`

```python
@dataclass
class PowertrainParams: ...        # classmethod from_car(car: CorsaC, power_scale=1.0)
@dataclass
class PowertrainState: ...         # omega_e, theta_slip, gear, stalled, shift_*
@dataclass
class PtInput:                     # named PtInput here, NOT DriverInput
    throttle: float = 0.0; brake: float = 0.0; clutch: float = 0.0
    handbrake: float = 0.0
    shift_up: bool = False; shift_dn: bool = False    # edges, consumed
    starter: bool = False; auto_gearbox: bool = True
    auto_clutch: bool = True       # the box works the clutch (launch / anti-stall
                                   # assist, downshift blip, self-restart); False
                                   # = H-pattern, the pedal is the only clutch
    tc_scale: float = 1.0          # vehicle._tc's gain on the ENGINE LOAD only;
                                   # `throttle` stays the driver's pedal
@dataclass
class PowertrainOutput:
    T_drive: tuple    # (4,) N.m at the wheels, FL FR RL RR; RL=RR=0 (FWD)
    T_brake: tuple    # (4,) N.m MAGNITUDE, always >= 0
    I_w_eff: tuple    # (4,) kg m^2, gear-dependent on the front
    rpm: float; gear: int; T_eng: float; T_clutch: float
    clutch_slip: float; P_wheel: float; stalled: bool; on_limiter: bool
    load: float = 0.0 # the engine load fraction applied this step (HUD, sound)

def step(p, s, inp: PtInput, omega_w, Fz, Fx, v_x, dt) -> PowertrainOutput
def brake_torques(p, brake, handbrake) -> tuple[float, float]   # (front/wheel, rear/wheel)
def diff_split(p, T_axle, omega_l, omega_r) -> tuple[float, float]
def wot_torque(p, n_e) -> float
def wot_power(p, n_e) -> float
def kmh_per_1000rpm(p, g) -> float
def self_check() -> None
```

* `step()` **returns** wheel torques and effective inertias. It does **not**
  integrate wheel speeds — `vehicle.py` does, because `Fx` couples the wheels
  to the chassis.
* Clutch is a **saturated PD** (`K_c=800`, `C_c=15`, cap `200 N.m`,
  anti-windup on `theta_slip`), never a discrete lock/unlock state machine.
* `eta_eff = 0.86` when `T_c >= 0`, `1/0.86` when `T_c < 0` (losses always
  oppose power flow).
* Engine torque: 19-breakpoint PCHIP (`scipy.interpolate.PchipInterpolator`),
  built **once at import**. Passes exactly through 110 N·m @ 4000 rpm and
  55.0 kW @ 5600 rpm.
* Reflected inertia architecture (a): integrate `omega_e` separately and put
  **only** `0.5*I_TRANS*N_TOT^2*eta` into each front wheel. Never also add
  `I_ENG` to the wheels.
* `C_RR = 0.012` charged as `C_RR*Fz_i*R_ROLL` per wheel. No separate chassis
  rolling-drag term anywhere in the codebase.
* Brakes: `KBF=1.6187e-4`, `KBR=5.4154e-5` N·m/Pa, `P_MAX_LINE=110e5`,
  proportioning knee `30e5` Pa, slope `0.30`. Front must lock before rear in
  both dry and wet — this is a hard test.
* **Three driver models share one shift machine** (`update_shift`, which now
  returns `(clutch_pedal, throttle_scale, blip_load)`):

  | mode | `auto_gearbox` | `auto_clutch` | the driver |
  |---|---|---|---|
  | auto | True | True | steers and pedals |
  | manual | False | True | + shifts (edges); cannot stall |
  | clutch | False | False | + launches on the pedal; can stall |

  The launch / anti-stall assist follows `auto_clutch`, **not** `auto_gearbox`.
  The gear change itself always runs declutch → gate → engage (0.70 s): that
  is the driver's own foot during a shift, in every mode.
* **Launch assist band.** The assist is proportional on engine speed over
  `[max(n_stall + 100, n_tgt − n_launch_band), n_tgt]`, `e = 1` at the target
  (continuous with the locked branch `n_input > n_tgt`), `n_launch_band =
  400 rpm`. The clutch's torque balance fixes the slip equilibrium at
  `e = (T_eng / T_cap)^(2/3) ≈ 0.63`, so the band sets the droop: the old band
  (550 rpm .. target, 1850 wide at WOT) parked the engine at ~1700 rpm
  transmitting ~85 N·m, 0–50 km/h 5.64 s against the rig's 5.12; 400 holds it
  ~150 under 2400 (5.46 s). Narrower acts as a stiff damper on the engine DOF
  (`1.5·T_cap·√e / band`, 5.7 N·m·s/rad here) and couples into the 11.3 Hz
  driveline mode. The zero-throttle anti-stall band (550..850) is unchanged.
* **`power_scale`** (`from_car(car, power_scale)`, the drive's Engine setting):
  `nm_bp` scaled as a whole and `T_clutch_cap = T_CLUTCH_CAP_STOCK ×
  power_scale` (a 2× engine on the stock 200 N·m clutch would slip at its own
  peak); overrun curve, idle governor, limiter and ratios untouched. 1.0 in
  every rig and every script. `PtInput.tc_scale` multiplies the mapped load
  (`load = throttle_map(pedal × thr_scale) × tc_scale`) so the scheduler's
  `N_UP = n_up_a + k·throttle` and the assist's `n_tgt` still see the pedal —
  a TC that cut the pedal made the auto box upshift at 3600 rpm and hunt.
* **Rev-match blip** (`auto_clutch` only): through the gate and engage phases of
  a DOWNSHIFT the engine is fuelled to `min(n_input(target gear), n_cut − 150)`,
  proportional over `n_blip_band = 800 rpm`, dead inside `n_blip_min = 150`.
  An upshift never blips. Measured 3rd→2nd at 15 m/s: engine 3960 rpm at
  engagement against a 4274 target (2595 without), peak clutch slip 269 rpm
  against 1644.
* **Stall → restart.** A stalled engine is cranked by `starter`, by
  `auto_clutch`, or by the clutch pedal fully in (`clutch_engagement <= 0`).
  While cranking it is **fuelled** (load ≥ 0.30, `fuel = 1`) plus `t_start`
  below `n_crank = 400`; it is running again above `n_fire = 500`. The
  previous rule (starter torque only, fuel off while stalled) left the engine
  at 400 rpm for ever — every stall was permanent.

Full equations, all 26 validation numbers: `../specs/powertrain.txt`
(the module self-check now carries 82).

## 4. `drive/vehicle.py` — integrator ordering is load-bearing

State (integrated): `X, Y, psi, u, v, r, phi, p, omega[4], kx[4], ky[4],
dFz_f, dFz_r, dFz_x`, plus `PowertrainState` held by composition.
`kx` is the transient slip ratio, `ky` the transient **tan** of slip angle.

**The order inside one physics step is fixed and must not be "tidied up":**

1. contact-point velocities from the **current** state
2. per-corner `Fz` from the **lagged `dFz_*` states** (no algebraic loop)
3. tyre coefficients (`sigma_x`, `sigma_y`, stiffnesses) from `Fz`
4. **advance `kx`, `ky`** from the **current** `omega` — exact exponential:
   `a = -|Vx|/sigma`, `c = Vs/sigma`,
   `k <- k*exp(a*dt) + (c/a)*(exp(a*dt) - 1)`, falling back to `k + c*dt`
   when `|a*dt| < 1e-6`. **Never form `kappa = Vsx/Vx`.** Clamp
   `kx∈[-1.5,1.5]`, `ky∈[-3,3]`.
5. tyre forces from the **new** `kx`, `ky` via `CORSA_TYRE.evaluate(...)`
   with `kappa = kx`, `alpha = atan(ky)`
6. Besselink low-speed damper, then the friction-ellipse cap at `ECAP = 1.05`
7. **implicit** wheel-spin update (damper term on the LHS):
   `omega <- (omega + (dt/I)*(T_drive - T_brake*tanh(omega/1.0) - R_e*Fx_t
   - C_RR*Fz*R_e*tanh(omega/1.0) + kv*R_e*Vx)) / (1 + dt*kv*R_e^2/I)`
8. body forces/moments → `u, v, r` (semi-implicit Euler, **keep `+v*r` and
   `-u*r`**), then `phi, p`, then pose `X, Y, psi` from the NEW velocities
9. load-transfer states, deploy state

Reversing steps 4 and 5/7 diverges at every dt below 8 ms at 3 m/s — measured.

**Load transfer** (must be algebraically identical to `qss.py`):

```
dFz_tot_demand = (SFy_tyre*h_cg + F_dev*(h_cg - h_w)) / t_bar     t_bar = 1.42450
dFz_x_demand   = (sum(Fxb_i) * h_cg) / L
two-rate split (LT = LPF(dFz_tot_demand, tau_roll=0.09)):
  ddFz_f/dt = (0.101*demand + 0.639*LT - dFz_f) / tau_LT           tau_LT = 0.02
  ddFz_r/dt = (0.221*demand + 0.039*LT - dFz_r) / tau_LT
  steady state -> 0.740 front / 0.260 rear == roll_dist_f
ddFz_x/dt = (dFz_x_demand - dFz_x) / tau_pitch                     tau_pitch = 0.12
Fz_FL = max(m*g*wdist_f/2 - dFz_x/2 - dFz_f, 0)      # + to the RIGHT wheels
Fz_FR = max(m*g*wdist_f/2 - dFz_x/2 + dFz_f, 0)
Fz_RL = max(m*g*(1-wdist_f)/2 + dFz_x/2 - dFz_r, 0)
Fz_RR = max(m*g*(1-wdist_f)/2 + dFz_x/2 + dFz_r, 0)
```

`SFy_tyre` is the sum of **tyre** body-frame lateral forces only — it must NOT
already contain `F_dev`, or the `h_w` sensitivity is doubled.
`roll_dist_f = 0.74` is a **calibration constant**, not a derived one. Do not
"fix" it from the roll centres (that gives 0.51 and breaks everything
downstream in `qss`/`crossover`/`ledger`).

**Roll** (visual + roll-camber only; the TOTAL transfer above is already a
ground-plane statement and must not be rebuilt from spring forces):
`I_roll = 342.8`, `Kphi = 36669 N.m/rad`, `Cphi = 2482 N.m/(rad/s)` (ζ=0.35),
`h_ra = 0.160`, `h_r = h_cg - h_ra = 0.390`. Solve the coupled 2×2 with `a_y`:

```
Mx_ext = m_s*g*h_r*sin(phi) - Kphi*phi - Cphi*p - F_dev*(h_w - h_ra)
[ m        -m_s*h_r ] [a_y ]   [SFy   ]
[ -m_s*h_r  I_roll  ] [pdot] = [Mx_ext]
```

**Steering** (`vehicle.py` receives an already-ramped, already-limited road-wheel
`delta` from `input.py` and applies only the physics):
* Ackermann fraction `A_ack = 0.60` about the bicycle angle.
* Front compliance steer: `delta_F* -= eps_f * Fy_f_prev`, `eps_f = 5.40e-6 rad/N`.
  This is what makes the linear-range understeer real (`V_char ≈ 25 m/s`).
* Rear roll steer `c_rs = 0.0` by default. `corsa_c.rollsteer_r = 1.0` is wrong
  in sign and magnitude — never implement it literally.

**Aero and the flank wing**:
```
q = 0.5*rho*V^2 ;  beta = atan2(v, max(u, 1.0))
D_aero  = q * CdA                                    # CdA = 0.66, at h_aero = h_cg
Fy_body = -q * A * Cs_psi * beta ; Mz_body = Fy_body * x_cp   # A=2.01, Cs_psi=2.2, x_cp=+0.30
sgn_dev = sign of the STEERING COMMAND (deadband 5% of lock, 0.3 s hold)
          — NEVER sign(beta) and NEVER sign(v)
alpha_dev = delta_dev_geom + sgn_dev*(-beta)
CL_dev  = clamp(CL0 + dCLda*alpha_dev, 0, 1.6)       # dCLda = 2.47 /rad; 0 in parity mode
dep     = smoothstepped deploy fraction over 0.45 s
F_dev   = sgn_dev * dep * q * S_DEV * CL_dev         # +y = inward for a LEFT turn
D_dev   = dep * q * S_DEV * CL_dev / 3.2
Mz_dev  = F_dev*x_w - D_dev*y_dev*sgn_dev            # y_dev = -sgn_dev*0.72
```
`S_DEV = 0.35 m²` is **one panel**. Exactly one panel is active at a time.

**Public API**:
```python
@dataclass
class Controls:
    delta: float = 0.0        # rad at the ROAD WHEEL, already ramped/limited
    throttle: float = 0.0; brake: float = 0.0; clutch: float = 0.0
    handbrake: float = 0.0
    gear_req: int = 0         # -1 down, 0 none, +1 up  (edge, consumed)
    auto_gearbox: bool = True
    auto_clutch: bool = True  # see section 3: launch assist, blip, restart
    wing_on: bool = False     # driver's toggle; the actuator lag lives in vehicle
    starter: bool = False

@dataclass
class VehicleConfig:
    qss_parity: bool = False        # Cs_psi=0, dCLda=0 -> reproduces qss exactly
    force_cos_delta: bool = True    # False = diagnostic that removes the cos(delta)
                                    # projection qss omits (0.855 g -> 0.861 g)
    wing: str = 'off'               # 'off' | 'fin' (CL0 0.70) | 'plate' (CL0 1.25)
    x_w: float = 0.97; h_w: float = 0.90
    mu_scale: float = 1.0
    guards: bool = True
    power_scale: float = 1.0        # the Engine setting -> powertrain.from_car; 1.0 in every rig
    tc_on: bool = False             # driver aid; DEFAULT OFF in every rig (below)
    abs_on: bool = False            # driver aid; DEFAULT OFF in every rig (below)
    dev_left: DevAero | None = None   # the DESIGNED panel on the left flank (y > 0)
    dev_right: DevAero | None = None  # ... and on the right; None = the closed form above
    top: TopAero | None = None        # the top (rear / roof) wing; None = none

@dataclass(frozen=True)
class DevAero:      # what a designed flank panel is to the physics
    name; S; CL0; CLa; CL_min; CL_max; cd0; cd1; cd2; x_w; h_w; inc   # inc in rad
    # CL = clamp(CL0 + CLa*alpha_dev, CL_min, CL_max); CD = cd0 + cd1*CL + cd2*CL^2
    # DevAero.legacy('fin'|'plate', x_w, h_w, inc_deg) states the published panel
    # in the same terms (CLa 2.47, CL in [0, 1.6], cd1 = 1/3.2): it reproduces
    # the closed form to round-off (validate W2, 1.4e-14).
@dataclass(frozen=True)
class TopAero:      # the top wing at its mounted incidence
    name; S; CZ; CD; CD_stowed; x_t; h_t; mode ('fixed'|'active'); t_ext 0.45; t_ret 0.30

class Vehicle:
    state: VehicleState
    tel: dict            # the diagnostics dict, rebuilt every step
    def __init__(self, car=CorsaC(), cfg=VehicleConfig()): ...
    def reset(self, x=0.0, y=0.0, psi=0.0, V=0.0, gear=1) -> None: ...
    def step(self, ctl: Controls, mu: Sequence[float], crr: Sequence[float],
             dt: float) -> None: ...
```

**Three wings.** The car carries up to three: a panel on each flank and a
wing on top. The FLANK physics is the device above, unchanged: exactly one
panel is active, the one on the OUTER flank of the turn (`dev_right` for a
left turn, `sgn_dev = +1`), armed by the driver's toggle and the steering
sign. A `DevAero` replaces `CL0 / dCLda / S_DEV / LD_DEV` for its side with
the affine lift law and quadratic drag law `drive/aero/wing.analyse` fitted
from the lattice; `qss_parity` freezes its slip term exactly as it zeroes
`dCLda`. With both sides `None` the code path is the published one, and the
suite holds it bit-for-bit.

The TOP wing is new physics, all of it exactly 0.0 when `top is None`:

```
top_cmd = wing_on                                          (mode 'fixed')
        = wing_on and (brake > 0.05 or |delta| > DEV_DEADBAND), held TOP_HOLD = 0.8 s   ('active')
dep_top = smoothstep(top_raw), actuator t_ext / t_ret as the flank
F_top   = dep_top * q * S * CZ                             # DOWN, positive
D_top   = q * S * (dep_top * CD + (1 - dep_top) * CD_stowed)
SFx    -= D_top
normal_loads(): Fz_front += F_top_prev * (x_t + b)/L / 2 per wheel, Fz_rear += F_top_prev * (a - x_t)/L / 2
demand_x = (SFx_t * h_cg + D_top * (h_t - h_cg)) / L      # the drag's pitch arm
P_req   += V * D_top
```

`F_top_prev` is LAGGED one step into `normal_loads()` exactly as the `dFz_*`
states are (no algebraic loop; `VehicleState.F_top_prev`, non-integrated,
not in `as_array`). The station matters: behind the rear axle
`(x_t + b)/L < 0` and the wing UNLOADS the front of this front-limited car
(measured: peak a_y 0.8550 g -> 0.8488 at x_t = -1.6, 0.8804 at +0.3, group
W). After every `step()` `F_top D_top top_deploy` are current alongside the
flank names above.

`mu` and `crr` are per-wheel `(4,)` surface scales supplied by the caller from
`track.surface_at()`. `Vehicle.step` advances **exactly one physics step of
`dt`** — the accumulator lives in `drive.py`.

After every `step()` these read-only attributes must be current (the renderer,
HUD and telemetry depend on the exact names):

```
x y psi u v r ax ay phi p beta          floats  (ax, ay body-frame at the CG)
Fz Fx Fy alpha kappa delta_wheel omega  (4,) numpy float arrays, FL FR RL RR
rpm gear engaged stalled on_limiter
F_wing D_wing wing_deploy wing_side     (wing_side: -1 right, 0 none, +1 left)
util_f util_r limited_by                ('FRONT' | 'REAR' | 'POWER')
wheel_lift                              (4,) bool
abs_active                              (4,) bool  (ABS gain < 1 on that wheel)
tc_active tc_gain                       bool, float (TC load gain < 1)
eng_load                                float  PowertrainOutput.load
```

`util_f/util_r` MUST be computed by calling `qss.fy_max` with `qss.TYRE`, not
by a local reimplementation of `mu(Fz)`.

**ABS** (`cfg.abs_on`, `Vehicle._abs`): a 4-channel slip-threshold controller
on the HYDRAULIC brake torque between the powertrain call and the wheel ODE;
the handbrake passes through unmodulated. It reads the transient `kx` (never a
`Vsx/Vx` quotient), dumps the gain at `|kx| > 0.16` (10 ms), rebuilds at
`|kx| < 0.13` — fast (20 ms) up to `0.85 ×` the gain that last provoked a
release, slowly (0.30 s) beyond it, which is what lets one controller stop dry
(learned level ≈ 0.65) and wet (≈ 0.35) without locking either. Floor 0.10,
off below 2 m/s. Pure in (state, dt): the replay is bit-identical. Measured
100→0 km/h at full pedal: dry 43.24 m against 44.64 (best fixed pedal) and
49.74 (locked); wet 62.03 m against 80.45 locked. **It is OFF in every rig
and every acceptance number** (the lock-order tests need the wheels to lock);
the interactive drive switches it on from its settings.

**TC** (`cfg.tc_on`, `Vehicle._tc`): engine-only traction control on the
driven axle, applied BEFORE the powertrain call as `PtInput.tc_scale` (a gain
on the engine load; the pedal itself is untouched — see section 3). Reads
`max(kx[FL], kx[FR])`: the target gain is 1 below `TC_SLIP_RESTORE = 0.12`,
falling linearly to `TC_GAIN_MIN = 0.15` at `TC_SLIP_CUT = 0.20`; the gain
slews to it at `1/0.03` per s down and `1/0.12` per s up; gain 1 and inactive
below `TC_V_MIN = 1 m/s` or with the pedal up. Pure in (state, dt). Measured
on the 2× Engine setting, WOT from rest on the open map: front slip ratio
1.50 (limiter-bouncing through all of 1st) without it, 0.31 with it, 0–100
km/h 8.42 s; it never acts on the stock car at WOT on dry tarmac (a
bang-bang cut/restore at 0.14/0.12 was tried first: 0.80 mean load in 1st,
8.78 s). **OFF in every rig**, like the ABS.

**Also in `vehicle.py`** (validation entry points, sharing the same force code):
```python
def steady_state_corner(R, *, car, cfg, power_cap=False, tol=1e-6) -> dict
def ramp_steer(V, rate_deg_s=2.2, T=16.0, *, car, cfg) -> dict   # THE grip-limit driver
def validate(verbose=True) -> bool                               # python3 -m drive.vehicle
```
Quantitative limits come from **open-loop ramp steer**, never from a
closed-loop skidpad controller (it saturates ~12% below the car).

## 5. `drive/track.py`

```python
@dataclass(frozen=True)
class Seg: kind: str; length: float; radius: float = 0.0; turn_deg: float = 0.0
@dataclass(frozen=True)
class SurfacePatch: s0,s1,n0,n1: float; mu_scale=1.0; crr_scale=1.0; label=''; colour=(43,58,74)
@dataclass
class Track: name,width,segs,origin,heading0,closed,surfaces,sector_s,guide_radii,gates
            # build() fills s, xy, psi, kappa, left, right, length, bbox

@dataclass(frozen=True)
class Area: kind: str; params: tuple; mu_scale=1.0; crr_scale=1.0; drivable=True; label=''; colour
            # 'rrect' (cx,cy,hx,hy,r) | 'rect' (x0,y0,x1,y1) | 'circle' (cx,cy,r)
            # contains(x, y) -> bool  (O(1), analytic);  polygon(n_arc) -> (M,2);  bbox()
Track also carries: areas: list[Area]   # open map: world-space tarmac / surface regions
                    features: list      # painted decorations, drawn only:
                                        # ('circle',cx,cy,r,colour,w) ('cone',x,y)
                                        # ('line',x0,y0,x1,y1,colour,w) ('box',x0,y0,x1,y1,colour)
                    title: str          # human name (HUD minimap, menus)

def build(tr) -> Track
def project(tr, x, y) -> tuple[float,float,float,float,int]   # (s, n, kappa, psi_c, i); n>0 = LEFT
def surface_at(tr, x, y, global_wet=1.0) -> tuple[float,float,bool]   # (mu_scale, crr_scale, on_track)
def on_tarmac(tr, x, y, n=None) -> bool          # ribbon OR a drivable area
def start_pose(tr, offset_n=0.0) -> tuple[float,float,float]
def point_at(tr, s, n=0.0) -> tuple[float,float]
def make_arena(); make_open(); make_skidpad(radius=50.0, cw=False); make_dragstrip()
def make_track(name, radius=50.0, cw=False, surfaces=True) -> Track   # the one builder
TRACKS = {'arena':..., 'open':..., 'skidpad':..., 'dragstrip':...}
TRACK_ORDER = ('arena', 'open', 'skidpad', 'dragstrip')      # TAB / the Map setting cycle this
TRACK_TITLES = {...}                                          # menu labels
```
`DS = 0.5 m` sampling, 8.0 m grid hash for projection with 3×3 → 5×5 → full
argmin fallback. Literal geometry (segments, node coordinates, arc centres,
closure to 3.8e-8 m, length 1249.2022 m) is in `../specs/harness.txt` and must
be used verbatim.

**The open map** (`make_open`, name `'open'`): a closed perimeter loop
(straights 420 / 260 m, corners R = 45 m, 12 m wide, length 1642.743 m, sector
lines at 0 / L/3 / 2L/3) whose rounded-rectangle outer edge + 6 m is one
drivable `rrect` Area (522 × 362 m), plus a `rect` Area (250..370, 95..175)
with `mu_scale = MU_WET_SCALE`. `surface_at` on a track with areas: `on` =
ribbon OR any drivable area containing the point; every containing area
multiplies its scales in; off everything = grass (`MU_OFF_TRACK`,
`CRR_OFF_SCALE`). Features (skidpad circles R 50 / 30.48 at (105, 150), a
300 m drag lane at y = 40..46 with 100 m boards, nine slalom cones at 18 m,
a cone gate) are never read by the physics. Geometry is not a spec number.

## 6. `drive/input.py`

```python
class InputSource(Protocol):
    def poll_events(self) -> list[str]                       # once per RENDER frame
    def update(self, dt, V, beta_deg, rpm, gear) -> Controls  # once per PHYSICS step
class KeyboardInput(InputSource): def __init__(self, steer_limit=True, ...)
class GamepadInput(InputSource):  @staticmethod def available() -> bool
class BlendedInput(InputSource)
class ScriptedInput(InputSource): def __init__(self, fn(t, veh, track) -> Controls)
def steer_limit_deg(V, beta_deg, ay_max=8.4608, L=2.491, k_us_deg=3.2,
                    lock_deg=32.625, beta_gain=1.2) -> float
```
* Ramp the **road-wheel angle** at a hand rate (`900 deg/s` at the wheel ÷ 16.0
  = `56.25 deg/s` at the road wheel), never a normalised axis.
* Return-to-centre `45 → 90 deg/s` road wheel with speed; counter-steer rate =
  drive + return.
* Speed limiter `delta_lim(V)` opened up by `+1.2*|beta_deg|` so slides can be
  caught. `steer_limit=False` bypasses the whole aid — every validation script
  uses that path.
* Pedal ramps: throttle 3.5/6.0 s⁻¹, brake 5.0/8.0, clutch 8.0/3.0,
  handbrake 8.0/10.0. Integrated at `DT_PHYS`, key state sampled at 60 Hz.
* Shift and wing keys are **edge-triggered** from the event queue only.
* `ScriptedInput` never reads pygame or the wall clock.
* **Gamepad layouts.** `GamepadInput` picks `'ps'` (DualSense / DualShock,
  by name: SDL HIDAPI order, R2 = axis 5 throttle, L2 = axis 4 brake, cross
  0 handbrake, square 2 clutch, circle 1 wing, triangle 3 wing side, L1/R1
  9/10 shift, options 6 menu, create 4 reset, touchpad 15 garage, d-pad
  11-14 hud/vectors/slowmo/normal, L3/R3 zoom_auto/camera) or `'generic'`
  (the SDL Xbox order that was here before). `~/.carsim_pad.json` overrides
  either. The pad does NOT fold `wing` / `wing_side` into its own state —
  `Sim` owns that toggle (a second copy left the panel armed with both
  readouts saying OFF).
* `BlendedInput` **hot-plugs**: it recounts joysticks every 30 render frames
  and attaches / drops the pad. `pad=None` is the steady state without one.
* `feedback(HudData)` drives rumble at 20 Hz. It reads `time.monotonic()` and
  is therefore called from the **render loop only**, never from `update()`.
* `BACKSPACE` -> `'garage'`; `Sim` ends the session with
  `stop_reason = 'garage'` and `run_garage_cli` re-enters the editor.
* **Pause menu.** `ESC` / OPTIONS / START -> `'menu'`. `set_menu(True)` puts
  every input in menu mode: the keyboard emits only `MENU_KEYS`
  (`nav_up`/`nav_down`/`select`/`menu`/`reset`/`full_reset`/`garage`) and
  reads all held keys as released; the pad emits `MENU_PAD_NAMES` edges plus
  left-stick up/down through `menu.StickNav`. Entering menu mode SEEDS the
  pad's edge state from the live buttons (the OPTIONS press that opened the
  menu must not close it), and menu mode keeps refreshing the driving edge
  state silently (a circle held across the close must not toggle the wing).
  `menu_help(layout)` is the single source of the on-screen key tables.
* **Gearbox modes.** `GEARBOX_MODES = ('auto', 'manual', 'clutch')`,
  `gearbox_flags(mode) -> (auto_gearbox, auto_clutch)`, `GEARBOX_LABELS`,
  `GEARBOX_HUD` (`AUTO` / `MAN` / `MAN+CL`). `KeyboardInput`, `GamepadInput`
  and `BlendedInput` carry `set_gearbox(mode)` and put both flags on every
  `Controls`; `BlendedInput.gearbox` reads the mode back, `set_steer_limit`
  reaches both halves.
* **Session boundaries.** `GamepadInput.seed_edges()` takes the live button
  state as already-pressed; the interactive session calls it on a pad it was
  handed (garage → drive), and the garage seeds its own edge table on the
  first poll — the ✕ that selected *Garage* in the drive's menu must not be
  the ✕ that drives straight back, and vice versa.
* **The drive's limiter runs on `K_US_DEG_MEASURED = 7.82`**, the module's own
  calibration finding; `K_US_DEG = 3.2` stays the function default because
  V16/V17 are quoted from it. Every validation path bypasses the aid anyway.

## 7. `drive/render.py`, `drive/telemetry.py`, `drive/plots.py`

* `Renderer(cfg: ViewConfig, track, headless=False)` with `update_camera`,
  `draw_frame`, `present`, `screenshot`, `frame_ms`.
* The tarmac ribbon is **ONE concave polygon** (0.58 ms) — never per-quad
  tessellation (20.4 ms, misses 60 fps).
* Skid marks capped at 600 visible segments by strided subsampling.
* Tyre-force and wing-force arrows drawn at the **same** px/N. The wing is
  63–227 N against a 9908 N car and must look that small.
* HUD utilisation must call `qss.fy_max` with `qss.TYRE`.
* g-g envelope calls `qss.max_ay` — cache, recompute only when `|ΔV| > 1 m/s`.
* `HudData` carries `x_w, h_w, wing_type, inc_deg` (defaults 0.97 / 0.90 /
  '' / 0): the plan-view panel is drawn at `aux.x_w`, not at a constant, so a
  garage-built car shows its panel where the physics has it.
* `HudData.menu` (default None) is drawn LAST by `draw_frame` when
  `menu.open`, duck-typed (`render.py` does not import `menu.py`).
  `HudData.gearbox` (`'AUTO' | 'MAN' | 'MAN+CL'`), `abs_active` and
  `track_name` are drawn under the gear glyph, in the flag row and on the
  minimap respectively; a stalled engine puts the restart hint in the warn bar.
  `HudData.engine` (`'75 HP' | '110 HP' | '150 HP'`) sits under the gearbox
  label, `tc_active` joins `abs_active` in the flag row (`TC ABS`), and
  `eng_load` (`PowertrainOutput.load`) is carried for the sound.
* **Sound** (`drive/audio.py`). `CarSound(volume)` owns one `pygame.mixer`
  channel (mono 16-bit, `CHUNK = 2048` samples, `BUFFER = 512`) and a `Synth`
  that renders phase-continuous chunks in numpy: engine (firing fundamental
  at rpm/30 Hz, harmonics 2–4 opened by load, the half-order, an exhaust
  ring per firing at `F_RES = 130 Hz`, load-shaped intake noise, limiter
  stutter, starter crank), tyre squeal from `max(util_f, util_r) > 0.90` or
  `max|kappa| > 0.10` above 2 m/s, grass rumble when `not on_track`, wind
  `(V/45)^2`, a clunk on the 0 → gear edge; `paused` fades it out. `Sim.run_
  interactive` calls `audio.update(hud)` right after the pad `feedback(hud)`
  — the render loop, never `step_physics` — and keeps the channel one chunk
  ahead (`play` + `queue`), so the sound trails the physics by 50–90 ms and
  `underruns` counts the frames that ran the queue dry. `CarSound.ok` False
  = no device; the sim drives silently. `python3 -m drive.audio` is its
  self-check (continuity across chunks, firing order in the spectrum, squeal
  band, silence, speed, the streaming path on SDL's dummy driver, a demo
  WAV).
* **Open-map drawing.** `Renderer._prep_track` precomputes each Area's polygon
  and bbox and the feature list. `draw_frame` draws the areas (filled, bbox-
  culled) after the grid and before the ribbon, and the features after the
  marks. `_visible_indices` returns `(runs, s_car, windows)`: on a circuit one
  arclength run + window as before; on a track with areas the centreline
  samples are culled by DISTANCE from the camera and split into contiguous
  runs (wrapping at the seam), each with its own window, because the
  perimeter road is in view on both sides of the pad and an arclength window
  draws only one. Ribbon / edges take runs, kerbs / dashes / marks / patches
  take windows. Measured 2.1–2.4 ms a frame.
* `drive/menu.py`: `Menu(title, items, sections)` with `show / hide /
  handle(cmd) -> action / draw(screen)`. Items are `(label, action)`;
  `select` returns the action and closes, `back`/`menu` return `'resume'`.
  `show(..., title=, idx=, columns=)`: `idx` keeps the cursor when a page
  re-shows itself, `columns=1` stacks every help section (and the note) in
  one column to the right of the items, and the highlight / column origin
  follow the widest item label. Overlay + panel surfaces are cached per size
  (a fresh full-screen SRCALPHA fill was 4 ms); warm draw ≈ 1.3 ms. Fonts are
  SysFont'd lazily: the first draw in a cold process pays the ~100 ms system
  font scan once.
* `HudData` also carries the three wings the renderer draws: `dev_left /
  dev_right` (present), `x_w_left / x_w_right`, `dev_chord / dev_span /
  dev_plate`, `wing_left_name / wing_right_name`, and `top_on top_deploy
  F_top D_top top_x top_span top_chord top_plate top_mode wing_top_name`
  (`CarBuild.hud_kwargs` + the vehicle's `F_top D_top top_deploy`). Both
  flank panels are always drawn -- stowed on the flank, the OUTER one slid
  out and lit while deployed, with its force arrow at the tyre scale -- and
  the top wing as a span-wide bar with plates, outline when stowed, filled
  when out, its drag as the backward arrow. The HUD's `ACTIVE AERO` panel
  has a flank row (L/R, side, %, F, D) and a top row (%, mode, Fz, D).
* `drive/garage.py`: `WingDesign(wing, x_w, h_w, inc_deg)` is still the
  published panel with `cfg_kwargs()` -> exactly the `VehicleConfig` fields
  `_aero` reads (`delta_dev_geom = radians(inc_deg)`); its ranges (x_w keeps
  the 0.45 m chord on the body, h_w 0.40-1.20 m, incidence -10..+15 deg) and
  closed-form readouts (`crossover.gain`) are unchanged. Around it:
  `CarBuild(name, left, right, top: Slot(wing, x, h, inc_deg, mode), mirror)`
  -- three slots holding LIBRARY wings by name. `cfg_kwargs(lib)` returns the
  closed-form kwargs (bit-identical to `WingDesign.cfg_kwargs`) whenever both
  flanks carry the same published panel and there is no top wing, else
  `dev_left / dev_right / top` built by `DevAero.from_aero` / `TopAero.from_aero`
  (a top wing is re-analysed at its slot height first: ground effect).
  `hud_kwargs(lib)` -> the HudData fields above. Persisted to
  `runs/garage_design.json` as `{"version": 2, "slots": ...}`; a v1
  `WingDesign` file upgrades to the same panel on both flanks. Pages: CAR
  (the 3-D view, three slots, `1 2 3` / TAB select, arrows place, `W` cycles
  the slot's library wing, `M` mirror, `T` top mode, `SPACE` deploy preview,
  `ENTER` drive), DESIGNER (`D`: one wing for one slot -- section, span,
  chord, taper, twist, end plates, the slot's mount, a GP-BO optimiser with a
  random-search twin at the same budget; live lattice, spanwise cl, polar,
  the affine/quadratic laws; `S` saves to the library under a NEW name when
  the origin is built-in), AIRFOIL (`A`: the section library ranked by the
  AeroBO screen weights at the design cl / Re, section + polar plots, `X`
  queues XFOIL in a worker thread, `N` a NACA-4 code), LIBRARY (`L`: wings and
  builds; ENTER puts a wing in the selected slot if its role matches, or
  loads a build; `S` saves the car as a build). Wing meshes are lofts of the
  chosen section (flank: vertical, suction side to the car; top: inverted,
  stowed on the deck, raised to its slot on deploy), painter's-sorted,
  backface-culled; the CAR page draws three wings in ~7 ms.
* `drive/aero/` (no pygame): `airfoil` (NACA-4, UIUC .dat, CST), `panel2d`
  (Hess-Smith, validated: NACA 0012 a = 6.91/rad, 4412 alpha_L0 = -4.26 deg),
  `polar` (`Polar` table; `estimate_polar` = panel slope + friction/form-
  factor drag + camber/thickness stall correlation, LABELLED estimate),
  `xfoil` (subprocess + JSON cache, split sweep from 0), `vlm` (horseshoe
  lattice, cosine edges / interlaced stations, tip plates, rigid-wall image,
  Trefftz CDi -- reproduces `aerobo.vlm.VLM` to 1e-12: CL_alpha
  4.55942959749662, e 0.9742625474419787 on the AR 8 rectangle), `wing`
  (`WingSpec` -> `analyse` -> the laws; critical-section stall; `design_point`
  with `crossover.gain`), `optimize` (numpy GP, Matern 5/2, EI, Sobol init),
  `library` (`runs/library/{airfoils,wings,builds,polars}`; 39 seeded
  sections, the published `fin` / `plate` as legacy wings, `flank-e423`,
  `rear-s1223`; XFOIL worker + `poll()`). The Re bank snaps to
  {1e5 ... 3e6}. Estimate vs XFOIL is printed on every read-out.
* Telemetry: fixed 63-column schema (`../specs/harness.txt`), 100 Hz, buffered
  200 rows, `newline=''`, `f'{v:.6g}'` (`repr` under `--telem-precision full`),
  sidecar `.json` with every constant.
* `plots.py` calls `matplotlib.use('Agg')` **before** importing pyplot. No pandas.

## 8. `drive/drive.py`

Fixed-timestep accumulator, `DT_PHYS = 0.001`, `FPS = 60`,
`MAX_SUBSTEPS = 40`, `MAX_FRAME_DT = 0.10`. Force `dt_wall = DT_PHYS` and
`acc = 0` on the first frame, after any reset and after unpause. Slow-mo
multiplies `dt_wall`, never `DT_PHYS`.

`--headless` sets `SDL_VIDEODRIVER=dummy` and `SDL_AUDIODRIVER=dummy` **before
pygame is imported anywhere in the package** (do it in `drive/__init__.py`
guarded by `CARSIM_HEADLESS`).

CLI:
```
python3 -m drive.drive [--track arena|open|skidpad|dragstrip] [--radius 50] [--cw]
  [--wet none|patch|all] [--wing off|fin|plate] [--wing-x 0.97] [--wing-h 0.90]
  [--dt 0.001] [--fps 60] [--size 1280x800] [--camera car_up|world_up|chase]
  [--headless] [--render off|offscreen|window] [--script NAME] [--duration 60]
  [--telemetry PATH] [--telem-hz 100] [--telem-precision 6g|full]
  [--no-steer-limit] [--gearbox auto|manual|clutch] [--manual] [--auto-gearbox]
  [--abs|--no-abs] [--tc|--no-tc] [--engine stock|tuned|sport]
  [--sound off|low|mid|high] [--wing-inc 0.0] [--garage] [--build NAME]
```
`--track`, `--wet`, `--camera`, `--gearbox`, `--engine`, `--abs`, `--tc`,
`--sound`, `--no-steer-limit` default to **None**: an interactive launch fills
them from `runs/settings.json` (an explicit flag wins and is saved back);
scripts and `--headless` runs fill them with the fixed defaults (arena, patch,
car_up, auto, **stock**, ABS off, TC off, sound off, aid on) and never read
the file.

**Settings** (`drive.Settings`: `track`, `engine`, `gearbox`, `abs`, `tc`,
`steer_aid`, `wet`, `camera`, `sound` — `Settings.KEYS`; properties
`power_scale`, `volume`; `load / save / clamp / apply_cli / to_opts / cycle`).
Defaults `ENGINE_DEFAULT = 'sport'` (`ENGINE_SCALE` stock 1.0 / tuned 1.5 /
sport 2.0) and `SOUND_DEFAULT = 'mid'` (`SOUND_VOLUME` 0 / 0.3 / 0.6 / 1.0).
The ESC menu has two pages: PAUSED (Resume / Settings / Reset to sector /
Full reset / Garage / Quit) and SETTINGS (one row per setting with its value —
Map, Engine, Gearbox, ABS, TC, Steer aid, Surface, Camera, Sound — then
Garage, Back). `Sim.apply_setting(key)` cycles, applies live and saves;
`track` and `wet` return True = the session must be rebuilt, which
`Sim.restart()` does by ending it with `stop_reason == 'restart'`. `TAB`
(`'track_next'`) is the same path. `Sim.set_gearbox(mode)` puts the mode on
the input layer and hands a stationary car in gear to the clutch-pedal mode
in NEUTRAL (it would stall on the spot); `Sim.reset` and the session start do
the same. `Sim.set_engine(mode)` swaps `veh.pt_p` for
`from_car(car, power_scale)` live (state untouched). `Sim._audio_apply()`
builds, re-levels or drops `Sim.audio` from `settings.volume`; it does nothing
unless `Sim.sound_enabled`, which only `_interactive_session` sets (a headless
Sim, or V26's duck-typed renderer, never touches the mixer).

**The session loop** is `run_interactive_cli` (`run_garage_cli` = the same
with `opts.garage = True`): one pygame session, `mode` in {garage, drive}.
garage → ENTER / ✕ saves the design, applies it to opts, mode = drive; Quit
ends. drive → `stop_reason` `'garage'` (BACKSPACE, touchpad, the menu or the
settings page) → mode = garage; `'restart'` → a new session; else exit. The
pad object survives every transition and is re-seeded at each boundary
(`GamepadInput.seed_edges`, `Garage._pad_seeded`); the garage also ignores
`'drive'` for its first 0.35 s. `_interactive_session(opts, pad, garage,
settings)` calls `settings.to_opts(opts)` first — the settings are the truth,
opts is the carrier — builds the track with `trk.make_track`, the car with
`abs_on = settings.abs`, the inputs with `steer_limit = settings.steer_aid`
and `k_us_deg = K_US_DEG_MEASURED`, `tc_on = settings.tc`, `power_scale =
settings.power_scale`, and `Sim(..., settings=settings)` with `has_garage`
True whenever `drive.garage` imports; with a real window it sets
`sound_enabled` and calls `_audio_apply()`. The garage design saved in
`runs/garage_design.json` is the car every plain launch drives unless a
`--wing…` flag is explicit. `Sim.handle_event` also owns the view toggles
(`camera`, `zoom_*`, `hud`, `vectors`, `gg`, `skid`) — renderer config only,
never physics.

**The build in the loop.** garage → ENTER / ✕ saves the `CarBuild`
(`runs/garage_design.json`, v2) and applies it to opts (`_apply_design`: the
four legacy fields plus `opts.wing_cfg` = `CarBuild.cfg_kwargs(lib)` and
`opts.hud_cfg` = `CarBuild.hud_kwargs(lib)`). The session builds
`VehicleConfig(**legacy, **wing_cfg)`, hands `hud_cfg` to `Sim.hud_cfg`
(merged into `hud_data()`), starts a designed build ARMED (`wing_on = True`)
exactly as `--wing` does, and `--build NAME` loads a car saved in the garage
library (`runs/library/builds/NAME.json`) instead of the last one built.

`Sim` pause menu: `'menu'` -> `_menu_open()` sets `paused = True`, remembers
whether `P` had paused already, and calls `inp.set_menu(True)`; while
`menu.open` every command goes to `_menu_event` (hotkeys `reset` /
`full_reset` / `garage` act directly, the rest through `Menu.handle`). On the
settings page `set:<key>` actions re-show the page with the cursor kept
(`Menu.show(idx=)`), `resume` / `settings_back` return to the PAUSED page.
`_menu_close()` restores the `P` pause if it was there, else `unpause()` so
the first-frame guard swallows the wall-clock gap. Without a renderer
`'menu'` is a plain pause toggle. `HudData.menu` carries it to the renderer.

`LapTimer.update(..., active=True)`: `active=False` (the car is more than 2 m
outside the ribbon — the middle of the open map's pad, where the projection
flips between the two sides of the loop and `s` jumps by hundreds of metres)
suspends crossing detection; a jump of more than 10 m in one step is ignored
regardless (a real 1 kHz step is < 0.06 m). `Sim.on_track` is the ribbon on a
circuit and `trk.on_tarmac` on a track with areas.

Keys: `↑` throttle, `↓` brake, `←/→` steer, `LSHIFT` fine, `Z` clutch,
`SPACE` handbrake, `S` starter, `E`/`Q` shift up/down, `F` flank wing, `G`
wing side, `R` reset, `SHIFT+R` full reset, `P` pause, `O` single step,
`[`/`]` slow-mo, `C` camera, `-`/`=`/`0` zoom, `H` HUD, `V` vectors, `B` g-g,
`N`/`X` skid, `T` wet, `M` marker, `L` record, `TAB` next map, `BACKSPACE`
garage, `ESC` pause menu / settings. PS5 pad map: section 6.

## 9. Reconciliations (where the subsystem specs disagreed)

1. **Tyre model.** `numerics.txt` proposed a simplified single-shape-function
   tyre; `tyre.txt` proposed full MF6.2 from the `.tir`. **Full MF6.2 wins** —
   it was verified to reproduce `qss.corner_speed` to 0.005% by monkey-patching
   `qss.fy_max`, which is stronger evidence than any refit. Combined slip uses
   the MF `Gxa`/`Gyk` weighting functions, not an elliptical projection.
2. **Relaxation integration.** `tyre.txt` wanted backward Euler inside
   `step_contact`; `numerics.txt` wanted an exact exponential owned by the
   integrator. **Exact exponential in `vehicle.py`** — the staggered ordering is
   the measured difference between stable and divergent at low speed, and it
   must be visible in one place.
3. **Low-speed damper.** `tyre.txt`: raised cosine, `VXLOW = 1.0`, ellipse cap
   at 1.05. `numerics.txt`: Besselink, `V_low = 2.5`, implicit on `omega`.
   **Both**: Besselink form at `V_low = 2.5` with the **implicit** `omega`
   update (`kv*R_e²/I` reaches 2300 s⁻¹ at standstill — explicit blows up), and
   the ellipse cap at `ECAP = 1.05` on the reported force so the damper cannot
   manufacture grip.
4. **Wheel inertia.** `chassis.txt` 0.65, `numerics.txt` 0.85/0.75,
   `powertrain.txt` 0.76/0.73. **`powertrain.txt` wins** (0.76 front / 0.73
   rear) — it is the only bottom-up build from the actual 5.5J×14 + 175/65R14 +
   236 mm disc / 200 mm drum. `powertrain.py` owns these numbers and hands
   `I_w_eff` to `vehicle.py`; `vehicle.py` never hard-codes a wheel inertia.
5. **Engine inertia.** `powertrain.txt` `I_ENG = 0.16`, `I_TRANS = 0.020`, which
   is the pair that produced a verified 15.50 s 0–100 km/h. **Use those.**
   `numerics.txt`'s `Ie = 0.12` came with a different (rigid-clutch) driveline.
6. **Steering filter location.** `chassis.txt` put the ramp+limiter inside
   `vehicle.py`; `harness.txt` put it in the input layer. **Input layer wins** —
   validation scripts must be able to command `delta` directly. `vehicle.py`
   keeps only Ackermann and compliance steer, which are physics.
7. **Wing side selection.** `numerics.txt` `tanh(ay_filt)`, `chassis.txt`
   `sign(steer)`. **`sign(steer)` with a 5% deadband and a 0.3 s hold** — at the
   limit `beta` is negative in a left turn, so anything derived from `beta` or
   `v` deploys the panel on the wrong side exactly when it matters (measured:
   −2.14% instead of +2.35%).
8. **0–100 km/h target.** `powertrain.txt` reproduces the commonly quoted
   ~15.5 s; `numerics.txt` notes Opel's own figure is 14.4 s. Accept
   **14.5–16.0 s** as passing and report the actual number. Never tune the
   torque curve or `eta_drive` to move it — those are pinned by `Vmax`.
9. **Driver aids never enter a measurement.** ABS (`VehicleConfig.abs_on`),
   TC (`tc_on`), the Engine setting (`power_scale`), the steering limiter,
   the sound and the settings file are all OFF / 1.0 / bypassed / unread on
   every scripted, headless and rig path; only the interactive session
   switches them on. The blip, the restart and the launch assist's band are
   part of the driver model (`auto_clutch`) and are on in the scripts that
   drive with the automatic box, which is what the accel and lap acceptance
   numbers already assumed (a driver who blips and holds the launch rpm);
   they move no number outside its band (0–100 km/h 14.97 s → 14.80 s with
   the launch band, scripted arena lap 61.1 s).

## 10. Two findings to report upward (do not silently absorb)

* **`qss.residuals` has an algebra bug in `Y_r`.** Line ~88 has
  `Y_r = (a*m*a_y + F*x_w)/L`; solving `qss`'s own two stated equations gives
  `Y_r = (a*m*a_y - F*(a - x_w))/L`. With the wing OFF (`F = 0`) they agree, so
  every baseline number stands; with the wing ON the rear goes spuriously
  limiting, which is why `qss.sweep()` reports the sealed plate (+1.18%) as
  *worse* than the clean fin (+1.21%) at R=100 — impossible for a front-limited
  car. Do not patch `qss.py` as part of this build; surface it.
* **`alpha_peak_deg = 7.0` in `qss.py`/`crossover.py` is too low.** The MF
  front-axle peak at the R=100 limit load split is 10.35°. Scrub drag goes as
  `sin(alpha_peak)`, so this moves every power-limited answer
  (`corner_speed(130)` 31.88 → 29.03 m/s). Report it; do not change `qss.py`.
