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

