# Task 4 — "Check that wings work correctly" : END-TO-END WING AUDIT

Read-only audit of the three wings (left flank panel, right flank panel,
top/rear wing) against `drive/CONTRACT.md` §0/§4/§7/§9.
**No `.py` file in the repo was modified.** All experiments ran from scratch
scripts that import the repo unchanged (a couple monkey-patch `Vehicle._aero`
in-process via `inspect.getsource` + one textual substitution, so the
"wrong-side" runs are provably the shipped code modulo one line).

NOTE ON LINE NUMBERS: `drive/vehicle.py` is being edited concurrently by
another task (the TC work), so line numbers drift. They were re-checked
against the working tree at the end of this audit, and **every patch below is
given as an exact old/new STRING** so it applies regardless of drift.

---

## 1. Sign conventions  — PASS (so far)

`steady_state_corner(R=100)`, `side=+1` (LEFT) and `side=-1` (RIGHT),
default cfg (`qss_parity=False`), `drive/vehicle.py:1487 ramp_steer`:

| wing | side | V (m/s) | a_y (g) | F_dev (N) | D_dev (N) | Mz_dev (N.m) | sgn_dev | beta (deg) | phi (deg) |
|---|---|---|---|---|---|---|---|---|---|
| off   | +1 L | 29.296466 | 0.874906 | 0 | 0 | 0 | +1 | -4.336 | +4.665 |
| off   | -1 R | 29.296466 | 0.874906 | 0 | 0 | 0 | -1 | +4.336 | -4.665 |
| fin   | +1 L | 29.927499 | 0.913002 | +177.389 | 55.434 | +132.155 | +1 | -5.640 | +4.664 |
| fin   | -1 R | 29.927499 | 0.913002 | -177.389 | 55.434 | -132.155 | -1 | +5.640 | -4.664 |
| plate | +1 L | 30.415546 | 0.943023 | +310.835 | 97.136 | +231.572 | +1 | -9.534 | +4.670 |
| plate | -1 R | 30.415546 | 0.943023 | -310.835 | 97.136 | -231.572 | -1 | +9.534 | -4.670 |

* **`sgn_dev = +1` for a LEFT turn** and the code picks `cfg.dev_right`
  (`drive/vehicle.py:849`) = the OUTER flank. PASS.
* **`F_dev > 0` (= +y = LEFT = inward) for a left turn.** PASS, +177.4 N.
* **beta IS negative at the limit in a left turn** (-4.34 g off / -5.64 fin /
  -9.53 plate): the contract's reason for never deriving the side from beta
  is reproduced.
* **phi > 0 in a left turn = leaning RIGHT = outside.** PASS, +4.665 deg.
* **Left/right symmetry is BIT-EXACT**: `V_L - V_R = 0.000e+00` and
  `F_dev(L) + F_dev(R) = 0.000e+00`, `Mz_dev(L) + Mz_dev(R) = 0.000e+00`
  for off / fin / plate. (`TYRE_MIRROR=True` is what buys this.)

### Mz_dev — the code is RIGHT and the CONTRACT is WRONG (already a logged DEVIATION)
`drive/vehicle.py:891` / `:873`: `Mz_dev = F_dev*x_w - sgn*Y_DEV*D_dev`.
The contract writes `Mz_dev = F_dev*x_w - D_dev*y_dev*sgn_dev` with
`y_dev = -sgn_dev*0.72`, which squares the sign and gives `+0.72*D_dev` in
BOTH directions — a non-mirroring term. Derivation: panel at
`y_dev = -sgn*0.72`, drag `Fx = -D_dev`, `Mz = x*Fy - y*Fx`
`= F_dev*x_w - (-sgn*0.72)*(-D_dev) = F_dev*x_w - sgn*0.72*D_dev`. The code
matches the geometry. Check: left turn fin `177.389*0.97 - 0.72*55.434
= 132.155` == reported `Mz_dev`. Sense: `-0.72*D_dev < 0` = yaw to the
RIGHT = out of the left turn, which is what a drag force on the outer flank
must do. Already documented as `vehicle.py:180 DEVIATIONS[1]`.
**No patch wanted in `vehicle.py`; `CONTRACT.md` §4 is the thing that is wrong.**

### Roll arm `-F_dev*(h_w - h_ra)` — PASS
`drive/vehicle.py:1062-1063`. Derivation: force `+y` at `dz = h_w - h_ra`
above the roll axis, `M = r x F = (0,0,dz) x (0,F,0) = (-dz*F, 0, 0)`, so
`Mx = -F_dev*(h_w-h_ra)`. Code sign is the geometry's sign. Physically the
high inward force rolls the body INTO a left turn, i.e. it reduces the
`phi > 0` outward lean — and that is exactly what the table shows:
phi 4.6654 (off) -> 4.6637 (fin) -> 4.6696 (plate) deg *while a_y rises
0.8749 -> 0.9130 -> 0.9430 g*, i.e. the roll gradient falls
5.331 -> 5.108 -> 4.952 deg/g. PASS.

## 3. Deployment side selection — PASS, with the failure mode re-measured

`drive/vehicle.py:809-820`: `want` from `ctl.delta` against
`DEV_DEADBAND = 0.05*LOCK_RAD = 1.631 deg` (`vehicle.py:101`), latched only
after `DEV_HOLD = 0.30 s` of persistence and only while
`dep_raw <= DEV_DEP_LOCKOUT`. **Never `beta`, never `v`.** PASS by inspection
and by the following counter-experiment (side rule replaced in-process by
`sign(v)`, and separately by hard-forcing the panel to the INNER flank):

| wing | side rule | gain vs wing-off at R=100 | F_dev (left turn) | sgn_dev |
|---|---|---|---|---|
| fin   | `sign(steer)` (shipped) | **+2.1540 %** | +177.389 | +1 (outer) |
| fin   | `sign(v)` (rejected)    | **-0.9644 %** | -93.162  | -1 (inner) |
| fin   | inner flank forced      | -0.9644 %     | -93.162  | -1 |
| plate | `sign(steer)` (shipped) | **+3.8198 %** | +310.835 | +1 |
| plate | `sign(v)` (rejected)    | **-1.9374 %** | -188.684 | -1 |
| plate | inner flank forced      | -1.9374 %     | -188.684 | -1 |

Two independent ways of putting the panel on the wrong flank agree to
3e-6 m/s. The contract's recorded pair is "-2.14 % instead of +2.35 %"; the
config behind those two numbers is not recorded, but in `qss_parity=True`
the shipped plate reads **+2.2029 %** (baseline peak a_y 0.8539 g, contract
0.8550 g) so the sign flip and the ~2 % magnitude are both reproduced.

### `wing_side` — it is the TURN sign (`sgn_dev`), not a side index
`drive/vehicle.py:1358`: `self.wing_side = aer["sgn_dev"] if aer["dep"] > 0.0
else st.dev_side`, and `aer["sgn_dev"] = int(sgn)` with `sgn = float(st.dev_side)`
(`:843`, `:866`), `st.dev_side = want` = sign of the steering command (`:817`).
So `wing_side = +1` means **left turn**, and the panel that deploys is
`cfg.dev_right` (`:846`). Measured: `side=+1` (left) -> `sgn_dev=+1`,
`F_dev=+177.4 N`. The renderer's reading is CORRECT; `CONTRACT.md` §4's
parenthetical "(wing_side: -1 right, 0 none, +1 left)" is misleading
documentation. (Cosmetic: the ternary at `:1296` is dead — both branches are
the same value, since `sgn_dev` is just `int(st.dev_side)`.)

---

## 2. The load path into the tyre normal loads — PASS, to round-off

`drive/vehicle.py:759 normal_loads()`. `Fz_f_static = m g wdist_f/2 = 3021.9705 N`,
`Fz_r_static = 1932.0795 N`, `hx = dFz_x/2`; `f0 = static - hx - dFz_f`,
`f1 = static - hx + dFz_f`, `r0/r1 = static + hx -/+ dFz_r`
— **term for term the contract's four expressions**, `max(...,0)` included.
Top-wing share: `df = 0.5*F_top_prev*(x_t+b)/L`, `dr = 0.5*F_top_prev - df`,
and `1 - (x_t+b)/L == (a-x_t)/L` identically, so the code IS the contract's
`(a - x_t)/L` rear share.

1499-step closed-loop run (throttle + brake + sinusoidal steer, plate on both
flanks, top wing at three stations), rebuilding all four `Fz` from the
**states only**:

| x_t | share_f = (x_t+b)/L | max abs(Fz_code - Fz_contract) | max abs(sum Fz - (mg + F_top_prev)) |
|---|---|---|---|
| -1.60 | -0.032312 | 4.547e-13 N | 3.638e-12 N |
| -0.90 | +0.248699 | 4.547e-13 N | 3.638e-12 N |
| +0.30 | +0.730434 | 4.547e-13 N | 3.638e-12 N |

so the front/rear split sums to `F_top` to 4e-12 N out of 193 N (2e-14 rel).

**The lag is exactly one step, bit-exact**: over the first 12 steps of a
deploy, `sum(Fz) - m g` at step k equals `F_top` at step k-1 to <1e-9 N for
every k, and is exactly 0.000e+00 at step 0.

**Downforce pushes DOWN**: straight at 35 m/s, top wing out,
`F_top = +366.343 N` and `sum(Fz) - m g = +366.36 N` (the 0.02 N is the
one-step lag). PASS.

**`SFy_tyre` carries tyre forces only** (`:1038` is the four `Fby` alone;
`F_dev` is added at `:1039` into `SFy`, not `SFy_tyre`). `dFz_tot_demand`
matches `(SFy_tyre*h_cg + F_dev*(h_cg-h_w))/t_bar` to **0.00e+00** at
h_w = 0.40 / 0.90 / 1.20 m; the leaked form (`SFy` instead of `SFy_tyre`)
would read 8.1 / 8.5 / 8.8 % high. PASS.

---

## 1(d). Top wing station — PASS, the contract's numbers reproduce EXACTLY

`ramp_steer(28.9, qss_parity=True)`, `TopAero.from_aero(S=0.40, CL0=0.9,
CLa=4.4, CL_min=-0.5, CL_max=1.9, cd0=0.02, cd1=0, cd2=0.06, inc 6 deg,
h_t=1.3, 'fixed')` -> `CZ=1.360767, CD=0.131101`. Baseline peak a_y
**0.8550 g** (contract: 0.8550).

| x_t (m) | share_f | peak a_y (g) | vs baseline | Fz_front (N) | Fz_rear (N) | util_f / util_r |
|---|---|---|---|---|---|---|
| -1.60 | -0.0323 | **0.8488** | -0.726 % | 6171.4 | 4009.4 | 0.9904 / 0.8641 |
| -1.20 | +0.1283 | 0.8548 | -0.024 % | 6213.9 | 3966.9 | 0.9907 / 0.8802 |
| -0.90 | +0.2487 | 0.8593 | +0.502 % | 6245.0 | 3935.9 | 0.9909 / 0.8924 |
| -0.50 | +0.4093 | 0.8654 | +1.210 % | 6282.1 | 3898.7 | 0.9913 / 0.9091 |
|  0.00 | +0.6100 | 0.8742 | +2.240 % | 6325.2 | 3855.6 | 0.9917 / 0.9344 |
| +0.30 | +0.7304 | **0.8804** | +2.973 % | 6355.1 | 3825.7 | 0.9926 / 0.9520 |
| +0.60 | +0.8509 | 0.8870 | +3.744 % | 6387.3 | 3793.6 | 0.9935 / 0.9705 |
| +0.97 | +0.9994 | 0.8937 | +4.531 % | 6426.1 | 3754.8 | 0.9937 / 0.9909 |

0.8488 at x_t = -1.6 and 0.8804 at +0.3 are the contract's recorded pair, to
4 decimals, and the car is `front`-limited at every station. Monotone in
`share_f`, sign change of the benefit at share_f ~ +0.13. Evidence file:
`runs/wingaudit_top_station.json`.

---

## 3(b). **BUG 1 (MAJOR) — the flank panel can NEVER change flanks once deployed**

**file:line**: `drive/vehicle.py:856`
```python
        armed = has and ctl.wing_on and st.dev_side != 0
```

**What the code does.** `st.dev_side` is only ever *written* inside
`if want != 0 and want != st.dev_side:` (`:814-818`), and the write is gated on
`st.dep_raw <= DEV_DEP_LOCKOUT` (0.05) — "no side change while the panel is
out". But `armed` (and therefore `cmd`, `:857`) asks only whether
`st.dev_side != 0`, so nothing ever *commands* a retraction: once a side is
latched, `dep_raw` is pinned at 1.0, `dep_raw <= 0.05` is never true again, and
the side latch **deadlocks for the rest of the session**. `st.dev_side` also
never returns to 0, so the panel never stows when the steering centres.

**What the contract/physics says.** §4: "exactly one panel is active, the one
on the **OUTER** flank of the turn", `sgn_dev` = sign of the steering command.
After the first left-hand corner the panel must move to the left flank for the
next right-hand corner. It does not.

**Measured magnitude.**
* Chicane rig (30 m/s, +0.10 rad for 2 s, straight 1 s, -0.10 rad for 3 s):
  `dev_side` stays `+1` for all 6 s, `dev_hold` accumulates to 2.901 s (ten
  times `DEV_HOLD`) and is never honoured; `F_dev = +302.40 N` **outward** for
  the whole right-hand phase. `sides ever latched: [0, +1]`.
* Toggling `wing_on` off on the straight does NOT rescue it: `dep_raw` reaches
  0 but `want == 0` there so no hold accumulates, and on re-arming
  `armed` is true again from `dev_side != 0`, so `dep_raw` is back past 0.05
  (0.667) before the 0.3 s hold elapses. Still `[0, +1]`.
* **90 s scripted arena lap, `--wing plate`, `runs/wingaudit_lap_plate.csv`:
  the panel is on the INNER (wrong) flank for 2114 of 5778 cornering samples
  = 36.6 % of cornering time (21.1 s of 57.8 s), and `wing_side` only ever
  takes the values `{0, +1}` across the whole run.** In that 21.1 s the panel
  is worth **-1.94 %** instead of **+3.82 %** of corner speed (measured above),
  a 5.76 pp swing, while still charging its drag.
* Straight-line consequence (30 m/s, `delta = 0`, `wing_on = True`, after one
  left corner): `dep = 1.000`, `F_dev = +302.40 N` sideways, `D_dev = 94.50 N`,
  `Mz_dev = +225.29 N.m`, uncorrected — the free-rolling rig spins the car up
  to `r = 4.21 rad/s` in 5 s.

**Minimal patch — pick ONE.** Both were regression-tested by monkey-patching
`_aero` in-process: `validate --only D` **9/9 pass with every printed number
bit-identical**, `validate --only W` **13/13 pass, identical**, and
`steady_state_corner(100)` with the plate is `V = 30.415546 m/s` (gain
+3.8198 %) in all three variants, so neither patch moves a single acceptance
number.

*Patch A — conservative: retract in order to swap, keep the panel out between
same-direction corners.*
```
OLD:        armed = has and ctl.wing_on and st.dev_side != 0
NEW:        armed = (has and ctl.wing_on and st.dev_side != 0
                     and not (want != 0 and want != st.dev_side))
```
Result: chicane swaps to `dev_side = -1` at t = 3.299 s (0.3 s hold + the
retract to 0.05); lap wrong-flank time **36.6 % -> 2.9 %** (the residue is the
honest swap transient); mean `D_dev` over the lap 34.28 -> 34.23 N.

*Patch B — also stow when the steering centres, which is what a DEPLOYABLE
device is for.*
```
OLD:        armed = has and ctl.wing_on and st.dev_side != 0
NEW:        armed = has and ctl.wing_on and want != 0 and want == st.dev_side
```
Result: chicane swaps at t = 3.299 s AND `dep = 0.000, F_dev = 0.0 N` on the
straight; lap wrong-flank time **36.6 % -> 0.1 %**; mean `D_dev` over the lap
**34.28 -> 20.99 N (-39 %)**, i.e. the drag the device is supposed to save on
the straights is now actually saved.

**Recommendation: patch B.** It is the behaviour §4's "`dep` = smoothstepped
deploy fraction" + `t_ret` + the top wing's explicit `'active'` mode all
imply, it removes a 302 N uncommanded side force on every straight, and it
costs nothing in the acceptance suite. Patch A is the option if someone wants
the smaller behavioural delta.

### 3(c). The rest of the deployment logic — PASS
| item | contract | measured |
|---|---|---|
| `DEV_DEADBAND` | 5 % of lock | `0.028471 rad = 1.6313 deg` at the road wheel (`vehicle.py:101`, `LOCK_RAD = radians(32.625)`) |
| `DEV_HOLD` | 0.3 s | side latched at **t = 0.299 s** of steady steer |
| flank `t_ext` | 0.45 s | `dep_raw` 0 -> 1 in **0.449 s** (latch at 0.299, full at 0.748) |
| flank `t_ret` | 0.30 s | `dep_raw` 1 -> 0 in **0.300 s** after `wing_on = False` |
| `dep` law | smoothstep | `_smoothstep(x) = x*x*(3-2x)` (`vehicle.py:589`); `dep = 0.5033` at half of `t_ext` |
| top `t_ext` | 0.45 | `top_raw` reaches 1.0 at **t = 0.449 s** after `wing_on` ('fixed') |
| top `t_ret` | 0.30 | reaches 0 at t = 2.599 s = 1.500 (brake off) + 0.800 + **0.299** |
| `TOP_HOLD` | 0.80 s | **0.799 s** between the trigger dropping and the retract starting |
| `'fixed'` | out whenever armed | out at 0.449 s, never retracts while `wing_on` |
| `'active'` | `brake > 0.05 or abs(delta) > DEADBAND`, held | out only during the brake window + 0.8 s |

---

## 4. Drag book-keeping — PASS, by an energy balance

* `SFx = SFx_t - D_aero - D_dev - D_top - m g sin(grade)` (`vehicle.py:1039`).
  Over a 2500-step closed-loop run with all three wings,
  `max abs(SFx_code - (SFx_tyre - D_aero - D_dev - D_top)) = 2.842e-13 N`
  with all three drags recomputed independently from `q` and the coefficients
  (`max abs(tel drag - recomputed) = 1.492e-13 N`). **Each drag is charged
  exactly once.**
* `P_req = V*(coriolis + induced + D_aero + D_dev + D_top + Crr*m*g)`:
  `max abs(P_required - that) = 1.455e-11 W`. Validate W3 independently:
  48.50 -> 52.22 kW with the top wing.
* **Coast-down energy closure** (free-rolling so `kappa == 0` and the only
  longitudinal forces ARE the drags), 40 -> 20 m/s,
  `runs/wingaudit_coastdown.json`:

| case | time (s) | distance (m) | W_aero (J) | W_top (J) | (sum W)/dKE |
|---|---|---|---|---|---|
| bare car | 63.762 | 1767.85 | 605997 | 0 | **0.999993901** |
| top wing 'fixed' (CD 0.1311) | 59.086 | 1638.40 | 561810 | 44190 | 0.999993422 |
| top wing stowed, CD_stowed 0.05 | 61.887 | 1715.86 | 588174 | 17823 | 0.999993716 |

  Closure to 6e-6 of dKE in every case: **no drag double-counted, none
  missing**. The top wing costs 4.676 s and 129.45 m (-7.32 %) of coast-down,
  and its own work is 7.29 % of the kinetic energy shed.
* `CD_stowed` is honoured exactly: stowed (`dep_top = 0.000`) with
  `CD_stowed = 0.05`, `D_top = 10.7808 N == q*S*CD_stowed = 10.7808 N`.
* Pitch arm. `demand_x = (SFx_t*h_cg + D_top*(h_t - h_cg))/L`
  (`vehicle.py:1095`) is the contract term for term, and the sign is the
  geometry's: a rearward force `dz` above the CG is a couple `M_y = -D*dz`,
  i.e. nose-UP, i.e. load to the REAR, and `dFz_x > 0` IS rear load in this
  code's convention. Isolated measurement (CZ = 0, CD = 0.20, free-rolling,
  steady at 30 m/s):

| h_t | D_top | dFz_x measured | `D_top*(h_t-h_cg)/L` |
|---|---|---|---|
| 0.55 (= h_cg) | 43.200 N | **+0.0000 N** | +0.0000 N |
| 1.30 | 43.200 N | **+13.0068 N** | +13.0068 N |

  Exact, and zero arm at `h_t = h_cg` as it must be.
* Flank-panel drag for scale: plate deployed at 40 m/s, `D_dev = 126.01 N`
  against `D_aero = 475.23 N` = **+26.5 % of the aero drag budget**. That is
  the standing cost of BUG 1 on every straight.

---

## 5. Frozen numbers at 1 kHz — PASS

* `DevAero` and `TopAero` are both `@dataclass(frozen=True)`
  (`__dataclass_params__.frozen == True`); **every field is a plain `float` or
  `str`** (12 and 10 fields checked, `non-scalar fields: none`), and
  `setattr` raises `FrozenInstanceError`.
* After `import drive.vehicle` in a clean interpreter:
  `drive.aero in sys.modules = False`, `pygame in sys.modules = False`.
  (`scipy` is present, via `drive.powertrain`, which the contract's module map
  allows.) `drive/aero` contains **no** `import pygame` anywhere.
* `_aero`'s entire global surface, from the bytecode:
  `CL_STALL, DEV_DEADBAND, DEV_DEP_LOCKOUT, DEV_HOLD, LD_DEV, RHO, S_DEV,
  TOP_HOLD, Y_DEV, _smoothstep, atan2, bool, dict, fabs, float, int, max,
  min, sqrt` and the methods `DevAero.cd, DevAero.cl, VehicleConfig.cl0,
  cs_psi, dclda`. **No numpy, no interpolation, no table lookup, no solver.**
* `step()` timing, 40 000 steps, gc off, after a 2000-step warm-up:

| configuration | us/step | real-time factor |
|---|---|---|
| no wings | 45.9 | 21.8 x |
| closed-form fin | 46.8 | 21.3 x |
| two `DevAero` | 46.6 | 21.4 x |
| two `DevAero` + `TopAero` | **48.0** | **20.8 x** |

  All three wings cost **2.1 us/step (4.6 %)**, and `_aero` on its own is
  **1.18-1.34 us/call** whatever is fitted. The only allocation is the
  diagnostics dict (`dict(base, ...)` = two dicts a step); it is inside that
  1.3 us and matches the contract's "`tel` is rebuilt every step".

---

## Other findings (all smaller than BUG 1; each with its number)

### BUG 2 (LOW, diagnostic mode only) — `qss_parity` is not path-independent at nonzero incidence
`drive/vehicle.py:873` (designed path) vs `:884-885` (closed form):
```python
            CL = panel.cl(panel.inc if cfg.qss_parity else alpha_dev)   # designed
...
        CL = CL0 + cfg.dclda() * alpha_dev                              # closed form, dclda()=0 in parity
```
Zeroing `dCLda` kills the incidence term as well as the slip term, so in
parity the closed form gives `CL = CL0`, while the designed path gives
`CL = CL0 + CLa*inc`. Measured at R=100, plate, `inc_deg = 2.0`:

| | closed form | `DevAero.legacy` | diff |
|---|---|---|---|
| `qss_parity=True`, inc 0 | CL 1.250000, V 29.580172 | CL 1.250000, V 29.580172 | **0.000e+00** |
| `qss_parity=True`, inc 2 deg | CL 1.250000, V 29.580172 | CL **1.336219**, V 29.634061 | **+6.90 % CL, +0.054 m/s (+0.18 %)** |
| `qss_parity=False`, inc 0 | V 30.415545740 | V 30.415545740 | 0.000e+00 |
| `qss_parity=False`, inc 2 deg | V 30.426366748 | V 30.426366748 | **0.000e+00** |

No acceptance number is affected (W2 uses `inc_deg = 0.0`), and `qss_parity`
is a diagnostic. The contract's own wording ("`qss_parity` freezes its slip
term exactly as it zeroes `dCLda`") is self-contradictory, because zeroing
`dCLda` removes more than the slip term. **Either** change the contract to say
the designed panel keeps its built-in incidence in parity (my preference — the
incidence is geometry, not a slip term), **or** patch:
```
OLD:            CL = panel.cl(panel.inc if cfg.qss_parity else alpha_dev)
NEW:            CL = panel.CL0 if cfg.qss_parity else panel.cl(alpha_dev)
```
(the second form also needs the `CL_min/CL_max` clamp if `CL0` can leave the
band, which it cannot for anything `wing.analyse` produces).

### BUG 3 (COSMETIC) — `VehicleConfig.h_aero` is dead
`drive/vehicle.py:353` declares `h_aero: float = 0.55` with the comment
"A 0.10 m error is 35 N of `dFz_x` at Vmax". It is **never read**:
`demand_x` (`:1095`) puts `SFx_t` at `h_cg` and gives only `D_top` an explicit
arm. Measured: `steady_state_corner(100, plate)` with `h_aero = 0.55` and
`h_aero = 1.50` returns `V = 30.415545740036` both times,
**dV = 0.000e+00**. Either wire it in or delete the field — as it stands it
invites someone to "tune the drag height" and see nothing happen.

### FINDING 4 (MODERATE, contract-level, not a wing bug) — `Fy_body` never enters the lateral load transfer
`drive/vehicle.py:1091`:
```python
        demand = (SFy_tyre * c.h_cg + aer["F_dev"] * (c.h_cg - aer["h_w"])) / self.t_bar
```
and the comment above it claims this is "ALGEBRAICALLY IDENTICAL to qss's
`(m*a_y*h_cg - F*h_w)/t`, because `m*a_y = SFy_tyre + F_dev + Fy_body`".
Substituting that same identity shows the two differ by exactly
`Fy_body*h_cg/t_bar`, which is not in the expression — i.e. the body side
force is treated as acting at ground level, while `cfg.h_aero`'s comment says
the body aero acts at `h_cg`. Measured at R=100:

| wing | `Fy_body` | omitted transfer `Fy_body*h_cg/t` | as % of `dFz_tot_demand` |
|---|---|---|---|
| off | +172.33 N | +66.54 N | 2.03 % |
| fin | +233.90 N | +90.31 N | 2.74 % |
| plate | +408.41 N | +157.69 N | **4.85 %** |

This is exactly what CONTRACT §4's load-transfer block prescribes, and
`qss_parity` zeroes `Cs_psi` so the parity path is untouched, so **the code
matches the contract** — but the comment's claim of identity is false and the
modelling gap grows with `beta` (4.85 % at the plate limit, where
beta = -9.53 deg). Owner's call; nothing to patch blindly.

### FINDING 5 (MINOR, matches the contract) — `D_dev`'s pitch arm is dropped while `D_top`'s is carried
`demand_x` carries `D_top*(h_t - h_cg)` but not `D_dev*(h_w - h_cg)`, and
`h_w = 0.90` is 0.35 m above `h_cg`. At the R=100 plate limit
`D_dev = 97.14 N` -> 34.0 N.m -> **13.65 N of `dFz_x`, 0.22 % of the front
axle load** (fin: 7.79 N, 0.13 %). Consistent with the contract (which names
only `D_top`'s arm) and with `qss`/`ledger`, so this is a contract-level
omission, recorded for completeness.

### FINDING 6 (MINOR) — no garage-built top wing ever has stowed drag
`drive/vehicle.py:478`, `TopAero.from_aero(...)` hardcodes `CD_stowed=0.0`,
and `garage._top_aero` is the only producer. The physics honours the field
exactly (measured: `CD_stowed = 0.05` gives `D_top = 10.7808 N == q*S*0.05`
with `dep_top = 0`), so `'active'` mode's drag saving is idealised — a stowed
wing is aerodynamically free. Defensible if the stowed wing is considered part
of `CdA = 0.66`; say so in a comment or plumb a value through.

### FINDING 7 (COSMETIC) — dead ternary
`drive/vehicle.py:1358`: `self.wing_side = aer["sgn_dev"] if aer["dep"] > 0.0
else st.dev_side`. Both branches are the same value (`aer["sgn_dev"]` is
`int(st.dev_side)` computed in the same call, `:846`/`:866`), so the condition
never changes the answer.

### FINDING 8 (COSMETIC) — an `'active'` top wing stays out after the driver disarms
`drive/vehicle.py:823-830`: when `ctl.wing_on` goes False, `want_t` is False
but `st.top_hold` is not cleared, so `cmd_t` stays 1.0 for up to
`TOP_HOLD = 0.8 s`. Harmless; mention only because a driver-facing toggle
that lags 0.8 s looks like a bug from the cockpit. One-line fix if wanted:
clear `st.top_hold = 0.0` when `not ctl.wing_on`.

### FINDING 9 (COSMETIC, documentation) — `wing_side` is documented backwards in two places
CONTRACT §4's read-only list says "`wing_side`: -1 right, 0 none, +1 left" and
`drive/telemetry.py:107` says "`+1 = panel on the LEFT, -1 = RIGHT`". The value
is the **turn sign**, so `+1` means a LEFT TURN and the panel that is out is
the RIGHT one. `render.py` reads it correctly. Suggested wording for both:
> `wing_side`: the TURN sign (`sgn_dev`) — `+1` = left turn, so the **RIGHT**
> panel is deployed; `-1` = right turn, so the **LEFT** panel is deployed;
> `0` = never armed. It is NOT the flank index.

### FINDING 10 (COSMETIC) — the `qss` §10 `Y_r` bug is replicated in the telemetry fixture
`drive/telemetry.py:481`: `Y_r = (a_cg*m*ay + F_dev*x_w)/L`, the form CONTRACT
§10 flags as wrong (`-F*(a - x_w)` is correct). It is a synthetic fixture, not
physics, and the code labels it "the same moment balance qss uses", so it is
deliberate. Flagging it only so it is not mistaken for a second, independent
derivation. `vehicle.py`'s own `Y_f_expected` (`:1380`) uses the correct
`-F_dev*(b + x_w)` form and closes to 0.000 % (group D).

### FINDING 11 (LATENT) — `_top_share_f` is cached in `__init__`
`drive/vehicle.py:626`. Replacing `cfg.top` on a live `Vehicle` leaves the
axle split stale. Nothing in the repo does that today (every call site builds
a fresh `Vehicle(car, cfg)`), so it is a trap rather than a bug.

---

## HEADLINE NUMBERS

Open-loop `steady_state_corner(R=100)` (the contract's mandated rig), aids
OFF, `qss_parity=False` unless stated.

| quantity | value |
|---|---|
| wing off, V / peak a_y | 29.296466 m/s / 0.874906 g |
| **fin gain** | **+2.1540 %** (V 29.927499, a_y 0.913002 g) |
| **plate gain** | **+3.8198 %** (V 30.415546, a_y 0.943023 g) |
| plate gain, `qss_parity=True` | +2.2029 % (baseline 0.8539 g) |
| designed build (flank-e423 x2, inc 4 deg) | +2.2380 % (F_dev 168.391 N) |
| ... same build + rear-s1223 top wing at x_t = -1.30 | +1.1369 % (F_top 305.89 N, D_top 26.08 N) |
| **left-vs-right symmetry** | `V_L - V_R = 0.000e+00` exactly, for off / fin / plate / designed; `F_dev(L)+F_dev(R) = 0`, `Mz_dev(L)+Mz_dev(R) = 0` |
| wrong-flank counterfactual (fin / plate) | **-0.9644 % / -1.9374 %** |
| panel on the wrong flank on a 90 s arena lap **as shipped** | **36.6 % of cornering time** |
| ... after patch A / patch B | 2.9 % / 0.1 % |
| flank drag at 40 m/s deployed | D_dev 126.01 N = +26.5 % of D_aero (475.23 N) |
| mean D_dev over the arena lap, shipped -> patch B | 34.28 -> 20.99 N (-39 %) |
| top wing coast-down cost (40->20 m/s, CD 0.1311) | 1767.85 -> 1638.40 m (-7.32 %), 4.676 s sooner |
| top wing P_required cost at R=100 (validate W3) | 48.50 -> 52.22 kW |
| **top-wing station**, peak a_y at 28.9 m/s (parity) | 0.8488 g @ x_t -1.6 < **0.8550 g none** < 0.8804 g @ +0.3 — the contract's recorded triple, exactly |
| top station monotone range | -0.726 % @ x_t -1.6 ... +4.531 % @ x_t +0.97; break-even at share_f ~ +0.13 |
| yaw sense, plate: Mz_dev / gain | x_w +0.97: +231.57 N.m, +3.8198 % (rear-limited) ; x_w 0: -59.48, +1.3476 % ; x_w -1.5195: -430.04, **-0.5058 %** |
| roll sense, plate: phi / roll gradient | h_w 0.40: 4.899 deg, 5.2091 deg/g ; 0.90: 4.670, 4.9518 ; 1.20: 4.530, **4.7972** |
| roll identity closure | `phi == (m_s h_r SFy/m - F_dev(h_w-h_ra))/Kphi` to 3e-5 .. 1.8e-4 deg |
| Fz algebra vs the contract, 1499 steps | 4.547e-13 N |
| top-wing split sums to F_top | 3.638e-12 N (of 193 N) |
| F_top lag | exactly one step, bit-exact, 0.000e+00 at step 0 |
| `SFy_tyre` purity | `dFz_tot_demand` == contract form to 0.00e+00; leaked form 8.1-8.8 % high |
| drag charged once | `SFx` residual 2.842e-13 N; coast-down energy closure 0.999994 |
| step() with all three wings | 48.0 us vs 45.9 us bare = +2.1 us (4.6 %), RTF 20.8 x; `_aero` 1.18-1.34 us |

## Evidence files (under `runs/`, which is gitignored — they are on disk only)
* `runs/wingaudit_summary.json` — gains, symmetry, wrong-flank counterfactual, `x_w` and `h_w` sweeps
* `runs/wingaudit_top_station.json` — the eight-station top-wing sweep
* `runs/wingaudit_coastdown.json` — coast-down drag isolation
* `runs/wingaudit_lap_plate.csv`, `runs/wingaudit_lap_off.csv` — the 90 s arena laps behind the 36.6 % number
* `runs/wingaudit_lap_plate_shipped.csv` — the same lap re-run through the monkey-patch harness, shipped code (byte-identical to `wingaudit_lap_plate.csv`)
* `runs/wingaudit_lap_plate_patch.csv` — the same lap under **patch B** (patch A's run overwrote to the same name and was not kept; its number, 2.9 %, is in the table above)

## Worth keeping as a repo script?
**Yes, one thing**: a `validate.py` W-group check that the flank panel actually
changes flanks. Two lines would have caught BUG 1:
```python
    veh = VE.Vehicle(CAR, VE.VehicleConfig(wing="plate")); veh.reset(V=30.0, gear=5)
    veh._hold_V = 30.0; veh._free_roll = True
    sides = set()
    for k in range(6000):
        t = k * 1e-3
        d = 0.10 if t < 2.0 else (0.0 if t < 3.0 else -0.10)
        veh.step(VE.Controls(delta=d, wing_on=True), (1.0,) * 4, (1.0,) * 4, 1e-3)
        sides.add(veh.state.dev_side)
    chk("W", "the flank panel follows the steering into the SECOND corner",
        "dev_side reaches both -1 and +1 in a left-right chicane",
        f"sides seen {sorted(sides)}", {-1, +1} <= sides, hard=True)
```
I did not add it (read-only audit). Everything else I wrote is throwaway and
lives in the scratchpad.

## What I could NOT verify, and why
1. **The contract's exact "-2.14 % instead of +2.35 %" pair** (reconciliation
   7). The sign flip and the ~2 % magnitude both reproduce (`+2.2029 %`
   shipped in parity; `-0.96 %` fin / `-1.94 %` plate on the wrong flank), but
   no combination of `wing` x `qss_parity` x R that I tried lands on that exact
   pair, and the config behind those two numbers is not recorded anywhere in
   the repo. The conclusion they support is confirmed; the two digits are not
   reproducible from what is written down.
2. **Whether "the panel stays out on the straight" was a deliberate choice.**
   Nothing in `CONTRACT.md`, the code comments or `README.md` says either way,
   and the owner is away. I have recommended patch B (stow on centre) on
   physics grounds and shown patch A as the conservative alternative; the
   *deadlock* half of BUG 1 is unambiguously a bug either way.
3. **The renderer's three-wing drawing.** `render.py` is owned by another
   task; I only ruled on `wing_side`'s meaning (FINDING 9) and did not audit
   the wing geometry it draws.
4. **A real-lap lap-time delta from BUG 1.** The scripted `lap` driver on the
   arena completed 0 flying laps inside the 90 s window in both the wing-on and
   wing-off runs, so I have the 36.6 % wrong-flank exposure but no lap-time
   number. The per-corner cost is the measured -1.94 % vs +3.82 % swing.
5. **`h_w`-sensitivity "doubling"**. The contract says leaking `F_dev` into
   `SFy_tyre` "doubles the `h_w` sensitivity". Measured, the leak does NOT
   change the `h_w` *slope* (shipped swing +0.249 pp over h_w 0.30-1.10 vs
   leaked +0.245 pp); what it changes is the `F_dev` coefficient
   (`2 h_cg - h_w` instead of `h_cg - h_w`), which shifts `dFz_tot` up 8.1-8.8 %
   and the plate gain down a uniform ~0.167 pp. `vehicle.py:18-22`'s own
   comment states it correctly; CONTRACT §4's one-line version is loose.
