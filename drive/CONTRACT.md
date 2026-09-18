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
| `drive/vehicle.py` | EOM, load transfer, roll, aero+wing, integrator | `tyre`, `powertrain`, `corsa_c`, `cars` |
| `drive/track.py` | track geometry, projection, surfaces | numpy |
| `drive/input.py` | keyboard/gamepad → `Controls` | pygame |
| `drive/render.py` | pygame drawing + HUD | `track`, `qss`, pygame |
| `drive/telemetry.py` | CSV logging | csv |
| `drive/plots.py` | matplotlib post-run plots (Agg) | matplotlib, numpy |
| `drive/garage.py` | 3D garage: `CarBuild` (three wing slots) -> `VehicleConfig` kwargs and the fitted wings' mass; designer / airfoil / library pages | `corsa_c`, `cars`, `crossover`, `input`, `menu`, `garage_ui`, `aero`, `vehicle` (the two aero dataclasses only), pygame |
| `drive/garage_ui.py` | widget kit for the garage pages (params, lists, plots, prompt) | pygame, numpy |
| `drive/aero/` | wing-design physics: sections, panel method, polars (XFOIL / estimate), vortex lattice, GP-BO, the library | numpy, scipy, the `xfoil` binary if present |
| `drive/menu.py` | pause / help menu overlay (ESC, OPTIONS); pure UI | pygame only |
| `drive/audio.py` | procedural car sound: `Synth` (numpy) + `CarSound` (one pygame.mixer channel); a render-loop consumer of `HudData`, never an input | numpy, pygame |
| `drive/drive.py` | main loop, CLI, scripted runs | everything |
| `drive/validate.py` | the whole acceptance suite | everything |

`garage.py`'s SELF-CHECK (and only its self-check) imports
`render.frame_budget_verdict`, to normalise its three page-draw budgets
against the machine's current speed rather than reimplementing the
normaliser. No cycle (`render` never imports `garage`) and nothing on the
interactive or acceptance path reaches it.

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
def tyre_for(tir_path=None, R0=0.2915, width=0.175) -> TyreModel   # cached per size
def mu_curve_matches(tyre, ref=None, loads=(...)) -> bool
def self_check() -> None   # python3 -m drive.tyre
```

* **`tyre_for` is how a different car gets a different tyre size.** It is a
  cache keyed on `(abspath, R0, width)` pre-seeded with `CORSA_TYRE` under the
  Corsa's own key, so `tyre_for()` and `tyre_for(the Corsa's three numbers)`
  return the SAME OBJECT and the default stays bit-for-bit. Call it at
  construction time; never from `step()`. `R0` is the only geometric term any
  equation reads — `width`, `aspect` and `rim_radius` are provenance.
* `mu_curve_matches(tyre)` is the assertion that licenses §4's rule that
  `util_f/util_r` come from `qss.fy_max` with `qss.TYRE`: those are the
  *Corsa's* `mu_ref/Fz_ref/s`, and referencing another car to them is only
  legitimate because there is **one coefficient set** in `tyre_data/` (all
  five loadable `.tir` files are identical except for geometry) and `mu(Fz)`
  contains no geometry. `Vehicle.__init__` asserts it per car. The day
  somebody adds a genuinely different `.tir`, that assertion fires and a
  `TYRE`-shaped dict must be derived from the new tyre instead.

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
    T_drive: tuple    # (4,) N.m at the wheels, FL FR RL RR. The DRIVEN pair
                      # carries it and the other pair is 0.0 -- front-driven
                      # (RL=RR=0) on the Corsa and every acceptance number,
                      # rear-driven (FL=FR=0) on a drive_layout='rwd' car.
    T_brake: tuple    # (4,) N.m MAGNITUDE, always >= 0
    I_w_eff: tuple    # (4,) kg m^2, gear-dependent on the front
    rpm: float; gear: int; T_eng: float; T_clutch: float
    clutch_slip: float; P_wheel: float; stalled: bool; on_limiter: bool
    load: float = 0.0 # the engine load fraction applied this step (HUD, sound)

def step(p, s, inp: PtInput, omega_w, Fz, Fx, v_x, dt) -> PowertrainOutput
def brake_torques(p, brake, handbrake) -> tuple[float, float]   # (front/wheel, rear/wheel)
def driven_pair(p) -> tuple[int, int]        # (0,1) front | (2,3) rear
def I_w_bare(p, driven=True) -> float        # one wheel's own inertia
def I_w_driven(p, g) -> float                # ... with the driveline reflected
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
* **Per-car brakes.** `brake_coeffs(car) -> (kbf, kbr, p_max_line, t_hb_max)`.
  `corsa_c.brakes` still says MISSING and still is — nobody publishes pad mu
  or a torque split — but the HARDWARE is documented, so `CarSpec` carries
  disc/drum diameters, whether the rear is a disc or a drum, and the piston
  diameters, and this turns them into N·m/Pa with **the same formulas the
  BRAKE BLOCK uses** (`BRK_MU_PAD`, `BRK_CSTAR`, `BRK_PAD_H` stay estimated in
  one place). A car matching the Corsa's geometry gets the BRAKE BLOCK's own
  constants back, by identity. `p_max_line` is solved to hold
  `BRK_AUTHORITY = 1.479/1.041` — the same 42 % lock-up over-authority on
  every car, which is what the lock-order tests and the locked-wheel sled
  depend on — and `t_hb_max` scales with the rear axle's own locking torque.
  **Front still locks before rear on every car, dry and wet**, with a larger
  margin than the Corsa's +7.7 % (MX-5 +102 % dry / +155 % wet, 540i +94 % /
  +116 %).
* **Per-car steering lock.** `vehicle.car_lock_rad(car)` =
  `steer_turns * 180 / steer_ratio` in rad, and `Vehicle.lock_rad` /
  `Vehicle.dev_deadband` (5 % of it) come from the car. The Corsa's documented
  2.9 turns at 16.0:1 gives exactly `LOCK_RAD`, asserted at import in
  `_check_reference` — so `LOCK_RAD` is a special case of the function, not a
  rival truth. `input.py`'s `DELTA_LOCK_DEG` is unchanged: the input layer is
  handed a lock by the caller and every validation path bypasses the aid.
* **Per-car engine curve.** `engine_curve(car) -> (rpm_bp, nm_bp, orpm_bp,
  onm_bp)` builds THIS engine's WOT and overrun curves from five anchors on
  the car (`n_peak_torque`, `n_peak_power`, `n_idle`, `n_cut`,
  `displacement`) plus its published `T_max` / `P_max`. **A car whose anchors
  match the Corsa's gets the module constants themselves** — the same tuple
  objects — so the Z12XE curve is identical to the last digit.
  The built curve passes through **both** published points exactly: the rpm
  axis is warped through the knots (0, peak torque, peak power, cut),
  piecewise linear and monotone, and the torque is scaled to `T_max` then
  corrected by a factor ramping 1 → c between the peaks so that
  `T(n_peak_power) == P_max / omega_peak_power`. Overrun scales with
  displacement (`FMEP*Vd/(4*pi)`). `from_car` also takes the rev cut, idle,
  `n_overrev`, the whole shift schedule (as a FRACTION of the cut — leaving
  the Corsa's 6050 upshift on an engine that revs to 7000 short-shifts it 950
  rpm below its own power peak) and `T_clutch_cap` (held at the Corsa's
  1.82× ratio to its own peak) from the car.
  **`CarSpec.engine_scale` is RETIRED and is 1.0 on every car** — it was a
  bodily multiplier on the Corsa's curve, and with a real per-car curve it
  would double-count.
* **The driven axle.** `PowertrainParams.driven` is `'front'` or `'rear'` and
  `from_car` reads it off `CarSpec.drive_layout`. **`'awd'` is REFUSED with a
  `ValueError`**, not silently treated as one of the two: this driveline has
  one clutch, one gearbox and one open diff, and a centre differential with a
  torque split is physics it does not have. Everything that pairs with the
  torque follows the driven pair — `T_drive`, the reflected transmission
  inertia (`I_w_driven`, which reflects into the 0.73 rear wheels instead of
  the 0.76 fronts on a RWD car), the input speed `omega_drv`/`omega_in` the
  clutch and the shift scheduler read, `P_wheel`, `I_w_eff`, and
  `accel_run`'s traction cap. `diff_split` was already axle-agnostic.
  `PowertrainState.I_w_front_eff` keeps its historical name (the spec's golden
  table quotes it) and means the DRIVEN axle.
  **The load transfer needs no new code**: `dFz_x_demand = (sum(Fxb_i)*h_cg)/L`
  is already general, so acceleration transferring load rearward UNLOADS a
  front-driven car and LOADS a rear-driven one on its own. Measured, WOT from
  10 m/s: driven-axle load −504 N (Corsa, FWD), +765 N (MX-5), +2058 N (540i).
  `Vehicle._tc` reads the DRIVEN pair's transient slip, with the same
  open-diff `1 - 2*share` cut floor.
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

**Per-car scaling of the calibrated blocks** (`CarDerived`, `car_derived(car,
cfg)`). Six numbers used to be Corsa C module/config constants and now follow
the fitted car: `h_ra`, `I_roll`, `Cphi`, the four `lltd_*` shares and
`Y_DEV`. The rule is

```
value(car) = value(Corsa) * ( hat(car) / hat(Corsa) )
```

`hat` being the cheap bottom-up estimate of that quantity (roll centres
interpolated at the CG station; `Ixx - m_us*(t/2)^2 + m_s*h_r^2`;
`sqrt(Kphi*I_roll)` at a fixed `zeta_roll`; the instantaneous geometric +
unsprung shares; half the mean track). The **absolute level stays the Corsa's
calibration and only the change between cars is derived** — which is the
honest statement, because the two other cars' whole suspension block is `est`.
It is also bit-for-bit by construction, not by luck: when `car`'s fields equal
`cars.CORSA_C`'s, `hat(car)` and `hat(Corsa)` are the same float, the ratio is
exactly `1.0`, and `x*1.0 == x`. `vehicle._check_reference()` raises at import
if that ever stops being true (a bare `assert` would vanish under `-O`), and
`validate()`'s **T21** prints it.

`roll_dist_f` is **NOT** scaled — it stays `0.74` on every car, for the reason
above plus two more: `Kphi_f/Kphi_r` is `est` on all three cars, so a per-car
LLTD would be a guess dressed as a measurement; and the bottom-up route
provably gives 0.51 for the one car that has data. What does follow the car is
the split of that same 0.74/0.26 between the instantaneous and the elastic
path (the roll centres are per-car), and `lltd_geo_* + lltd_roll_*` is held at
`roll_dist_f` **to the last bit** by writing the roll share as a difference
from the reference pair.

`Vehicle.__init__` also builds `self.tyre = tyre.tyre_for(car.tyre_file,
car.tyre_R0, car.tyre_width)` (§2) — `CORSA_TYRE` itself for the Corsa and for
a plain `CorsaC`, which has no `tyre_*` fields — and sets
`self.tyre_ref_ok = tyre.mu_curve_matches(self.tyre)`, the assertion that
licenses the `qss.TYRE` rule below for a non-Corsa car. `_tyre_eval` takes the
tyre as a trailing argument defaulting to the singleton.

**Added mass** (`cars.PointMass`, `cars.with_masses(car, pts)`,
`cars.ballast_point(car, kg, where)`). Ballast and the garage's fitted wings
reach the physics as a **new `CarSpec`**, never as a patched `m`: the first
moments move `wdist_f` (hence `a`, `b` and every static wheel load) and
`h_cg`, and the parallel-axis theorem moves `Izz / Ixx / Iyy`; `m_s` carries
all of it (nothing a driver adds bolts to an unsprung hub). Springs, roll
centres, wheelbase, track, tyres, gearing and `CdA` are properties of the car
and are untouched, so a ballasted car rolls more and accelerates less on its
own. `with_masses` returns **the same object** at zero added mass, which is
what makes the ratio above exactly 1.0. Stations are quoted from the axles
(`nose` = `a + 0.30`, `seat` = the CG at `h_cg`, `floor` = `-b` at 0.30 m,
`boot` = `-(b + 0.25)` at 0.65 m) so they mean the same thing on every
wheelbase; `seat` is the control case and moves the mass and provably nothing
else.

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
alpha_dev = delta_dev_geom + sgn_dev*(-beta_dev)
beta_dev  = atan2(v + r*x_w, max(u, 1.0))            # cfg.dev_curved_flow True (DEFAULT)
          = beta                                     # False: degrade to qss's model
CL_dev  = clamp(CL0 + dCLda*alpha_dev, 0, 1.6)       # dCLda = 2.47 /rad; 0 in parity mode
dep     = smoothstepped deploy fraction over 0.45 s
F_dev   = sgn_dev * dep * q * S_DEV * CL_dev         # +y = inward for a LEFT turn
D_dev   = dep * q * S_DEV * CL_dev / 3.2
y_dev   = -flank*sgn_dev*Y_DEV                       # Y_DEV = 0.72; flank = +1 OUTER (default), -1 INNER
Mz_dev  = F_dev*x_w + y_dev*D_dev                    # = F_dev*x_w - sgn_dev*Y_DEV*D_dev on the outer flank
```
The old form `Mz_dev = F_dev*x_w - D_dev*y_dev*sgn_dev` squared the sign and
gave `+0.72*D_dev` in BOTH directions — a term that does not mirror, so the
car was not left/right symmetric on paper even though the code is. `Mz` from
the panel is `x*Fy - y*Fx` with `Fx = -D_dev` at `y = -sgn_dev*Y_DEV`, i.e.
`F_dev*x_w - sgn_dev*Y_DEV*D_dev`, which is what `vehicle.py` implements and
has always logged as its own DEVIATION. Checked: left-turn fin
`177.389*0.97 - 0.72*55.434 = 132.155` == the reported `Mz_dev`, and the drag
term yaws the car OUT of the turn, which a force on the outer flank must.

**The curved-flow term.** A body-fixed point at `(x_w, y_dev)` moves at
`V + omega x r`, and with `omega = (0,0,r)` that is `(u - r*y_dev, v + r*x_w)`,
so the panel's OWN flow angle is `atan2(v + r*x_w, u)` and not the body's
`beta`. The `y_dev` term only perturbs the axial component and is second order
in the angle. The sign is the opposite of the intuition: in a left turn
`r > 0` and `v < 0`, so `v + r*x_w` is LESS negative for a panel AHEAD of the
reference point, `|beta|` falls, and **a forward-mounted panel sees LESS
incidence in a corner, not more**.

`cfg.dev_curved_flow` defaults **True** — the physically correct model — and
`False` is the diagnostic that degrades to `qss`'s. That is the direction
`vehicle.py`'s own precedent runs in (`force_cos_delta` defaults True and
includes a projection qss omits; the module docstring says "if this model ever
MATCHES qss, something has been reimplemented that should not have been"), and
physics that is only correct behind a flag is a trap. The acceptance numbers
were **re-baselined deliberately**: the bands absorb it, W 13/13, D 9/9, suite
82/82 and `--modules` 100/100 unchanged. Measured at the R = 100 limit,
`x_w = 0.97`:

| | beta -> beta_dev | CL | ramp-steer gain |
|---|---|---|---|
| fin | -5.807 -> -5.200 deg | 0.9503 -> 0.9215 (-2.76 %) | +3.8587 -> +3.7412 % |
| plate | -7.631 -> -6.944 deg | 1.5789 -> 1.5448 (-1.88 %) | +6.0473 -> +5.9449 % |

and on the scripted arena lap the device's worth falls +0.0582 -> **+0.0279 s**
(fin, more than halved) and +0.1594 -> **+0.1472 s** (plate, -7.7 %). The sign
of the benefit survives in both. `qss_parity` freezes it exactly as it zeroes
`dCLda`, so the parity path is unaffected either way.

**The flank the panel deploys on** (`cfg.dev_flank`, `'outer'` the DEFAULT |
`'inner'`). The flank reaches the physics through **one term only**, and the
shortness of that list is the point: a pure LATERAL force has no lateral moment
arm, so `F_dev`'s yaw moment is `x_w*F_dev` and its roll moment is
`-F_dev*(h_w - h_ra)` wherever across the width of the car it acts, and the
lateral load transfer reads `F_dev*(h_cg - h_w)` with no `y` in it either. What
the flank changes is the **drag's** yaw moment, `y_dev*D_dev`: out of the corner
from the outer flank (−40 N·m against +172 from the lift at the R = 100 limit,
i.e. 23 % of the device's yaw authority), into it from the inner one. Everything
else is bookkeeping — `dev_left` / `dev_right` swap roles, since the slot that is
outer in a left turn is the inner one on the other setting.

The side force does **not** flip with the flank. Keeping it pointed at the turn
centre from the inner flank means turning the section over so the suction
surface faces OUTBOARD, away from the body — a different aerodynamic problem,
which is why `wing.build_lattice` takes `wall_side` (§7). `vehicle.py` never
learns what a wall is: the orientation reaches it inside a `DevAero`'s CL/CD
laws, exactly as a mount does. Measured both ways in
`.handoff/15-inner-flank.md`; **the default stays `'outer'`**.

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
    dev_flank: str = 'outer'        # 'outer' (shipped) | 'inner'; see above
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
        # `car` is a corsa_c.CorsaC or a cars.CarSpec (a field-for-field
        # superset of it, optionally already carrying added mass)
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
F_wing D_wing wing_deploy wing_side     (wing_side is the TURN sign = sgn_dev:
                                         +1 left turn, so the RIGHT panel is
                                         deployed; -1 right turn, so the LEFT
                                         panel is deployed; 0 never armed.
                                         It is NOT the flank index.)
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
below `TC_V_MIN = 1 m/s` or with the pedal up.
**The cut depth is limited to `1 - 2*share`**, `share` being that wheel's
fraction of the front-axle vertical load, floored at `TC_GAIN_MIN`. The 2 is
the open diff: both driven wheels carry equal torque, so the axle's tractive
force is twice the force of the wheel with less grip and a wheel of load
share `s` can still deliver `2s` of the axle's capacity. Straight ahead
`share = 0.5`, the floor is 0 and the full `TC_GAIN_MIN` authority is back,
so every launch number below is unchanged. This is not cosmetic: with full
authority everywhere the unloaded INSIDE front of this FWD car (991 N against
4846 N, `kx` 0.508 against 0.012 at 6 deg of steer) held the gain at 0.256
mid-corner and the car DECELERATED at full throttle, `ax = -0.206 m/s²`
(`.handoff/08-steering.md`, symptom b). Pure in (state, dt). Measured
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
def steer_limit_pair_deg(V, beta_deg, ...same kwargs...) -> tuple[float, float]
```
* Ramp the **road-wheel angle** at a hand rate (`900 deg/s` at the wheel ÷ 16.0
  = `56.25 deg/s` at the road wheel), never a normalised axis.
* Return-to-centre `45 → 90 deg/s` road wheel with speed; counter-steer rate =
  drive + return.
* Speed limiter `delta_lim(V)` opened up by `+1.2*|beta_deg|` so slides can be
  caught. `steer_limit=False` bypasses the whole aid — every validation script
  uses that path.
* **The `beta` bonus is DIRECTIONAL and the clamp is a PAIR.**
  `steer_limit_deg` keeps the symmetric form (the signature is pinned here and
  V16/V17 are quoted from it, and it is still what `validate.py` checks), but
  what `KeyboardInput` and `GamepadInput` clamp against is
  `steer_limit_pair_deg(V, beta_deg) -> (limit on LEFT lock, limit on RIGHT
  lock)`, which spends the whole bonus on the **counter-steer** side and
  leaves lock *into* the slide at the floor. Counter-steer has the SAME sign
  as `beta` (`beta = atan2(v, |u|)` with `v` leftward, so a left-turn slide is
  `beta < 0` and the catch is right lock — the same statement as section 9
  item 7). Symmetric, the term was positive feedback: more lock → more slide →
  bigger `|beta|` → a higher limit → more lock, and at `|beta| = 20 deg` the
  aid handed over full mechanical lock at any speed. Measured, keyboard,
  DOWN+LEFT held from 30 m/s: symmetric `delta_max 22.32 deg` for `dpsi 37.98
  deg` of heading change; directional `delta_max 14.08 deg` for `dpsi 41.10
  deg` — 8.2 deg less lock, 3.1 deg more turn (`.handoff/08-steering.md`).
  The magnitude of the catch is untouched: a 12 deg slide still opens 19.12 deg
  of opposite lock at 30 m/s.
* **The clamp eases down, it does not snap.** Each of the two bounds is a
  state: it rises to the commanded value instantly (a wider limit is never a
  surprise) and falls at `_return_rate_deg(V)`, which is the rate the wheel's
  own self-aligning torque would unwind it at. Without this, a directional
  bound collapsing 20+ deg the instant `beta` changes sign would teleport the
  road wheel. `KeyboardInput.delta_lim_deg` reports whichever bound is binding
  the direction the wheel is actually turned.
* **Trigger rest.** A pad trigger is normalised from the rest value its FIRST
  HID report showed, `-1.0` or `0.0` (`GamepadInput._trig_rest`), so both
  driver conventions give 0 at rest and 1 at the stop; with `-1.0` that is the
  historical `(a+1)/2` exactly. Before that first report every axis reads
  `0.0`, which `(a+1)/2` turns into HALF TRAVEL — the pedals are held at 0
  until `_rest_checked` is true, or a hot-plug hands the car 47% throttle and
  47% brake at once.
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
* `render.set_car(car)` points the module's `_CAR` at the fitted car and drops
  the g-g cache (whose curve is built from `_CAR.CdA / Crr / m / P_wheel`).
  It defaults to `CorsaC()`, so every module self-check and every headless
  renderer is unchanged; `drive.drive` calls it once per session.
  `HudData.car_name` / `mass_kg` are drawn under the minimap — the Ballast
  setting is otherwise invisible, and 200 kg is 20 % of a Corsa.
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
  with `crossover.gain`; `DESIGN_VARS` is the ONE ordered table of design
  variables and it is the garage DESIGNER page's row order -- `BOUNDS`,
  `design_bounds`, `design_x0`, `design_labels` and `apply_design` all read
  it, so the optimiser's vector cannot drift out of step with the page;
  `span_fit(role, h)` is the sill/roof packaging fit that BOTH the page's span
  row and the optimiser's span band must use), `optimize` (numpy GP,
  Matern 5/2, EI, Sobol init; `labels` ride through into the result and a
  label list out of step with `bounds` raises),
  `library` (`runs/library/{airfoils,wings,builds,polars}`; 39 seeded
  sections, the published `fin` / `plate` as legacy wings, `flank-e423`,
  `rear-s1223`; XFOIL worker + `poll()`). The Re bank snaps to
  {1e5 ... 3e6}. Estimate vs XFOIL is printed on every read-out.
* **`WingSpec.mount`** (`wing.MOUNTS = ('pylon', 'endplate', 'none')`) is a
  real aerodynamic choice, not a label, and it applies to either role:
  * `'pylon'` -- **the default, and what every library wing was analysed
    with**. Charges `strut_cd`: two mounts of `standoff` length, wetted-area
    friction on `S_ref`, times a **1.3 form factor** which is the pylon/wing
    junction interference allowance (Hoerner, *Fluid-Dynamic Drag* ch.8).
    Tip plates stay an independent continuous knob.
  * `'endplate'` -- carried by structural tip plates instead. `strut_cd` is
    NOT charged, but `WingSpec.plate_h_flown` raises `plate_h` to
    `MOUNT_PLATE_H` (0.06 m flank / 0.12 m top, `est`: enough flange for two
    fasteners plus edge distance) and **the lattice then produces the reduced
    tip loss itself** -- it is not a correlation. The plates' wetted area is
    charged by the existing `cd_pl` term.
  * `'none'` -- nothing charged. The idealisation to compare against, and the
    parity setting for a legacy panel whose published L/D already includes
    its mounts.
  **`build_lattice(..., wall_side=+1)`** is which side of the panel the body
  wall is on: `+1` (the default) the SUCTION side, which is where the body is
  when the suction surface faces inboard and the panel is on the OUTER flank;
  `-1` the PRESSURE side, which is the same panel turned over — what it takes
  to deploy on the INNER flank and still point the side force at the turn
  centre (§4, `cfg.dev_flank`). Rotating the panel 180° about its span carries
  the tip plates with it and this lattice builds them on the lift (+z) side, so
  they still point +z and now face away from the body: no clip. At the standoff
  the car deploys to the clip does not bind on any library flank wing (0.566 m
  at `RIDE_H0['flank'] = 0.60`, against plates of 0.06–0.16 m), so the two
  orientations differ **only** in which side of the panel the wall is on.
  Measured on `flank-e423` at 5°:
  CL 0.7783 → 0.7694, CDi 0.08111 → 0.07975, e 1.2366 → 1.2291 — slightly
  **less** lift for slightly **less** induced drag with the body on the
  pressure side, both above free air, and the free-air limit comes back
  monotonically as the wall is moved away. `analyse` and `spanwise` pass it
  through and `analyse` returns it, so a stored aero says which orientation it
  was flown in.

  `analyse()` returns `mount` and `mass`; `wing_mass(spec, standoff)` is a
  bottom-up floor (two skins + two plates at `SKIN_KG_M2 = 4.17`, 1.5 mm
  2024-T3, plus `PYLON_KG_M` when pylon-mounted) and is what the garage
  charges to the car through `CarBuild.mass_points`. An unknown mount name
  clamps to `'pylon'`. `plate_h_flown` is what `build_lattice` and
  `hud_kwargs` both read, so the renderer draws the plates the aero flew.
  The mount's entire effect reaches the physics inside `CZ` / `CD` via
  `TopAero.from_aero` -- **`vehicle.py` never learns what a mount is.**
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
  [--car corsa|mx5|540i] [--ballast 0..300] [--ballast-at nose|seat|floor|boot]
  [--wet none|patch|all] [--wing off|fin|plate] [--wing-x 0.97] [--wing-h 0.90]
  [--dt 0.001] [--fps 60] [--size 1280x800] [--camera car_up|world_up|chase]
  [--headless] [--render off|offscreen|window] [--script NAME] [--duration 60]
  [--telemetry PATH] [--telem-hz 100] [--telem-precision 6g|full]
  [--no-steer-limit] [--gearbox auto|manual|clutch] [--manual] [--auto-gearbox]
  [--abs|--no-abs] [--tc|--no-tc] [--engine stock|tuned|sport]
  [--sound off|low|mid|high] [--wing-inc 0.0] [--dev-flank outer|inner]
  [--garage] [--build NAME]
  [--ml-drive CHECKPOINT]
```
`--script drive_probe` is the one scripted entry that deliberately switches
the driver aids ON, because it exists to measure them (section 9 item 9 is
about MEASUREMENTS, and this is not one: `validate.py` never calls it, every
figure it prints is labelled driveability, and it takes the aid state as
arguments rather than reading `runs/settings.json`). It runs both input paths
— `KeyboardInput` fed a synthetic key state through `set_keys()`, and `delta`
commanded directly at the same 56.25 deg/s hand rate — so an input-layer
effect can be told from a physics one. See `.handoff/08-steering.md`.
`--track`, `--wet`, `--camera`, `--gearbox`, `--engine`, `--abs`, `--tc`,
`--sound`, `--no-steer-limit`, `--car`, `--ballast`, `--ballast-at` default to
**None**: an interactive launch fills them from `runs/settings.json` (an
explicit flag wins and is saved back); scripts and `--headless` runs fill them
with the fixed defaults (arena, patch, car_up, auto, **stock**, **corsa**,
**0 kg**, ABS off, TC off, sound off, aid on) and never read the file. An
*explicit* `--car` / `--ballast` does reach a script, the way `--wing` does —
the car is the subject of a measurement, not a driver aid, and measuring
another one with the repo's own rigs is the point of having them. Nothing in
`validate.py` passes either flag, so every acceptance number is the stock
Corsa C with no ballast.

**Settings** (`drive.Settings`: `track`, `car`, `ballast`, `ballast_at`,
`engine`, `gearbox`, `abs`, `tc`, `steer_aid`, `wet`, `camera`, `sound` —
`Settings.KEYS`; properties `power_scale`, `volume`, `car_base`; methods
`car_spec(extra=())`, `ballast_text()`, `load / save / clamp / apply_cli /
to_opts / cycle`).

`Settings.power_scale` is `ENGINE_SCALE[engine]` and nothing else. It briefly
also carried `cars.get(car).engine_scale = T_max/110`, which was how another
car's torque peak rode the existing `power_scale` path with no change in
`powertrain.py`; `engine_curve(car)` (§3) now builds each engine's own curve,
so that multiplier is retired and would double-count. The
Engine row and the HUD label read the car's own PS through `engine_ps /
engine_hud / engine_label`, which reproduce the hard-coded `75 / 110 / 150 HP`
on the Corsa (asserted in `self_check`).

`Settings.car_spec(extra)` is the `CarSpec` a session is built on: the named
car, plus the ballast, plus `extra` (the garage's fitted wings, from
`CarBuild.mass_points(lib)` via `opts.mass_points`). With the stock car, no
ballast and no wings it returns `cars.CORSA_C` **itself**.

`car`, `ballast` and `ballast_at` return **True** from `Sim.apply_setting`, the
same as `track` and `wet`: a different `CarSpec` is a different tyre model,
wheel station set, static load set, derived roll block and gearbox, all of
which are built in `Vehicle.__init__`, so the session is rebuilt rather than
live-patched. `_opts_car(opts)` resolves the scripted/headless car and returns
**None** for the stock Corsa with no ballast, so `_build` constructs `CorsaC()`
exactly as it always did; `_build` gained `car=None` and `mu_scale=None` (the
car's own `mu_scale`, an explicit value still winning, which is what the wet
rigs pass).
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
`--ml-drive CHECKPOINT` puts a trained `drive.ml` policy in the driver's seat
instead of the keyboard/pad, through the SAME local `ScriptedInput` closure the
acceptance scripts use — so no new code reaches the physics path. `drive/ml` is
an OPTIONAL sub-package: **nothing imports it unless this flag is given**, and
a missing checkpoint, a missing numpy or a shape mismatch prints why and hands
the session back to the keyboard rather than stopping it. It is the only hook
`drive/ml` has, and `drive/ml` never imports `pygame`.

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
   TC (`tc_on`), the **Engine setting** (`ENGINE_SCALE`), the steering limiter,
   the sound and the settings file are all OFF / 1.0 / bypassed / unread on
   every scripted, headless and rig path; only the interactive session
   switches them on. `cfg.power_scale` is the driver's Engine setting **and
   nothing else**, and `_build` sets it to exactly `1.0` on every scripted
   path. It briefly also carried the car's own `engine_scale = T_max/110`,
   because otherwise `--car 540i --script accel` measured a 1780 kg car with
   a 110 N·m Corsa engine and reported 0–100 km/h in 24.3 s; `engine_curve`
   (§3) makes that unnecessary, and this reconciliation is back to meaning
   what it originally said. The blip, the restart and the launch assist's band are
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
