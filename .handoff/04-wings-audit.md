# Task 4 — "Check that wings work correctly" : END-TO-END WING AUDIT

Read-only audit of the three wings (left flank panel, right flank panel,
top/rear wing) against `drive/CONTRACT.md` §0/§4/§7/§9.
**No `.py` file in the repo was modified.** All experiments ran from scratch
scripts that import the repo unchanged (a couple monkey-patch `Vehicle._aero`
in-process via `inspect.getsource` + one textual substitution, so the
"wrong-side" runs are provably the shipped code modulo one line).

STATUS: in progress — numbers below are final unless marked TODO.

---

## 1. Sign conventions  — PASS (so far)

`steady_state_corner(R=100)`, `side=+1` (LEFT) and `side=-1` (RIGHT),
default cfg (`qss_parity=False`), `drive/vehicle.py:1484 ramp_steer`:

| wing | side | V (m/s) | a_y (g) | F_dev (N) | D_dev (N) | Mz_dev (N.m) | sgn_dev | beta (deg) | phi (deg) |
|---|---|---|---|---|---|---|---|---|---|
| off   | +1 L | 29.296466 | 0.874906 | 0 | 0 | 0 | +1 | -4.336 | +4.665 |
| off   | -1 R | 29.296466 | 0.874906 | 0 | 0 | 0 | -1 | +4.336 | -4.665 |
| fin   | +1 L | 29.927499 | 0.913002 | +177.389 | 55.434 | +132.155 | +1 | -5.640 | +4.664 |
| fin   | -1 R | 29.927499 | 0.913002 | -177.389 | 55.434 | -132.155 | -1 | +5.640 | -4.664 |
| plate | +1 L | 30.415546 | 0.943023 | +310.835 | 97.136 | +231.572 | +1 | -9.534 | +4.670 |
| plate | -1 R | 30.415546 | 0.943023 | -310.835 | 97.136 | -231.572 | -1 | +9.534 | -4.670 |

* **`sgn_dev = +1` for a LEFT turn** and the code picks `cfg.dev_right`
  (`drive/vehicle.py:846`) = the OUTER flank. PASS.
* **`F_dev > 0` (= +y = LEFT = inward) for a left turn.** PASS, +177.4 N.
* **beta IS negative at the limit in a left turn** (-4.34 g off / -5.64 fin /
  -9.53 plate): the contract's reason for never deriving the side from beta
  is reproduced.
* **phi > 0 in a left turn = leaning RIGHT = outside.** PASS, +4.665 deg.
* **Left/right symmetry is BIT-EXACT**: `V_L - V_R = 0.000e+00` and
  `F_dev(L) + F_dev(R) = 0.000e+00`, `Mz_dev(L) + Mz_dev(R) = 0.000e+00`
  for off / fin / plate. (`TYRE_MIRROR=True` is what buys this.)

### Mz_dev — the code is RIGHT and the CONTRACT is WRONG (already a logged DEVIATION)
`drive/vehicle.py:888` / `:873`: `Mz_dev = F_dev*x_w - sgn*Y_DEV*D_dev`.
The contract writes `Mz_dev = F_dev*x_w - D_dev*y_dev*sgn_dev` with
`y_dev = -sgn_dev*0.72`, which squares the sign and gives `+0.72*D_dev` in
BOTH directions — a non-mirroring term. Derivation: panel at
`y_dev = -sgn*0.72`, drag `Fx = -D_dev`, `Mz = x*Fy - y*Fx`
`= F_dev*x_w - (-sgn*0.72)*(-D_dev) = F_dev*x_w - sgn*0.72*D_dev`. The code
matches the geometry. Check: left turn fin `177.389*0.97 - 0.72*55.434
= 132.155` == reported `Mz_dev`. Sense: `-0.72*D_dev < 0` = yaw to the
RIGHT = out of the left turn, which is what a drag force on the outer flank
must do. Already documented as `vehicle.py:177 DEVIATIONS[1]`.
**No patch wanted in `vehicle.py`; `CONTRACT.md` §4 is the thing that is wrong.**

### Roll arm `-F_dev*(h_w - h_ra)` — PASS
`drive/vehicle.py:1059-1060`. Derivation: force `+y` at `dz = h_w - h_ra`
above the roll axis, `M = r x F = (0,0,dz) x (0,F,0) = (-dz*F, 0, 0)`, so
`Mx = -F_dev*(h_w-h_ra)`. Code sign is the geometry's sign. Physically the
high inward force rolls the body INTO a left turn, i.e. it reduces the
`phi > 0` outward lean — and that is exactly what the table shows:
phi 4.6654 (off) -> 4.6637 (fin) -> 4.6696 (plate) deg *while a_y rises
0.8749 -> 0.9130 -> 0.9430 g*, i.e. the roll gradient falls
5.331 -> 5.108 -> 4.952 deg/g. PASS.

## 3. Deployment side selection — PASS, with the failure mode re-measured

`drive/vehicle.py:806-817`: `want` from `ctl.delta` against
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
`drive/vehicle.py:1296`: `self.wing_side = aer["sgn_dev"] if aer["dep"] > 0.0
else st.dev_side`, and `aer["sgn_dev"] = int(sgn)` with `sgn = float(st.dev_side)`
(`:843`, `:863`), `st.dev_side = want` = sign of the steering command (`:814`).
So `wing_side = +1` means **left turn**, and the panel that deploys is
`cfg.dev_right` (`:846`). Measured: `side=+1` (left) -> `sgn_dev=+1`,
`F_dev=+177.4 N`. The renderer's reading is CORRECT; `CONTRACT.md` §4's
parenthetical "(wing_side: -1 right, 0 none, +1 left)" is misleading
documentation. (Cosmetic: the ternary at `:1296` is dead — both branches are
the same value, since `sgn_dev` is just `int(st.dev_side)`.)

---

## 2. The load path into the tyre normal loads — PASS, to round-off

`drive/vehicle.py:756 normal_loads()`. `Fz_f_static = m g wdist_f/2 = 3021.9705 N`,
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

**`SFy_tyre` carries tyre forces only** (`:1035` is the four `Fby` alone;
`F_dev` is added at `:1036` into `SFy`, not `SFy_tyre`). `dFz_tot_demand`
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

**file:line**: `drive/vehicle.py:853`
```python
        armed = has and ctl.wing_on and st.dev_side != 0
```

**What the code does.** `st.dev_side` is only ever *written* inside
`if want != 0 and want != st.dev_side:` (`:811-815`), and the write is gated on
`st.dep_raw <= DEV_DEP_LOCKOUT` (0.05) — "no side change while the panel is
out". But `armed` (and therefore `cmd`, `:854`) asks only whether
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

