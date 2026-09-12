# Task 5 — "When automatic"

## Interpretation (correct me in one line if this is wrong)

The owner's sentence is unfinished: the whole item is literally **"When
automatic"**. It sits between the wing items (3, 4) and the car/weight items
(6, 7), and the sim has two unrelated "automatic" things: the **Gearbox
setting** (`GEARBOX_MODES = ('auto','manual','clutch')`) and the **automatic
wing-side selection** (`G`, `Sim.wing_side_mode`).

**I read it as: something is wrong with the car when the gearbox is in
Automatic.** The automatic wing-side logic is audited in
`.handoff/04-wings-audit.md` §3(b) (a real bug: the flank panel could never
change flanks once deployed) and is being fixed by the `vehicle.py` owner, so
this note is **only about the automatic gearbox** — `_auto_target` /
`update_shift` in `drive/powertrain.py`.

If the owner meant the wing, item 4 already covers it and nothing here is
wasted: the defects below are real and reproducible either way.

---

## Headline

| # | area | verdict |
|---|---|---|
| 1 | shift points vs the torque curve | **PASS** — WOT points are near-optimal (3->4 and 4->5 sit within 1 rpm of the force crossover) |
| 2 | hunting / hysteresis | **DEFECT A** — the hysteresis is NEGATIVE in 1st and 2nd; up to **0.65 shifts/s for ever** in a held-speed cruise |
| 3 | kickdown | **DEFECT C** — engagement itself is on spec (0.400 s), but the box walks down ONE gear per 1.5 s and then overshoots and upshifts back |
| 4 | creep and launch | 0-50 / 0-100 PASS; **DEFECT D** — no creep at all at zero pedal (neutral for ever), and the neutral->1st branch has no overrev guard |
| 5 | cornering / aids | **PASS** on the question asked — the shift count is identical at every steer angle and with TC on or off, at both power scales, while `tc_gain` falls to 0.457 and `kappa_max` hits 1.077. The `tc_scale`/pedal separation is intact. But the sweep turned up **DEFECT E** |
| 6 | other | **DEFECT B** — the upshift schedule is not brake-aware: `4>5@3034` rpm on corner entry. **DEFECT E** — wheelspin makes it upshift at a walking pace: FOURTH GEAR at 6.3 m/s on mu 0.30 at `power_scale` 2.0. **DEFECT F** — a driver-commanded downshift has no overrev guard and is live in auto: 8553 rpm from four taps of `Q` |

All six are reproducible headless. **A, B, C, D, E are fixed** (one commit
each for the scheduler and the regressions); **F is measured, its patch is
written out verbatim, and it is deliberately NOT applied** — it widens the
driver model in all three modes and CONTRACT.md pins that table.

Baseline before any edit (this machine, commit `6b6ff06`):
`python3 -m drive.validate --modules` -> **100/100 pass, 0 HARD, 0 soft [231.0 s]**;
`python3 -m drive.powertrain` -> 82/82 checks.

---

## 2. DEFECT — the up/down schedules OVERLAP in 1st and 2nd (negative hysteresis)

`_auto_target` (`drive/powertrain.py`) schedules

```
N_UP = n_up_a + k*throttle      k = n_up_k12 (3750) for gears 1-2,
                                    n_up_k34 (3650) for gears 3-4
N_DN = n_dn_a + n_dn_k*throttle = 1500 + 3100*throttle   (2200 if brake > 0.3)
```

For the box not to ask for the gear back the instant it has changed up, the
rpm the engine LANDS on after the upshift must be above the downshift line:

    N_UP(g) * gear[g+1]/gear[g]  >  N_DN

Ratio steps of the F13 CR box: 1->2 0.6045, 2->3 0.6668, 3->4 0.7844,
4->5 0.7920. Measured (algebra, `auto_probe.py hyst`):

```
 g  thr    N_UP  n_after    N_DN   margin  overrev-guard  hunts
 1 0.00    2400     1451    1500      -49       no         YES
 1 0.20    3150     1904    2120     -216       no         YES
 1 0.50    4275     2584    3050     -466       no         YES
 1 0.90    5775     3491    4290     -799       no         YES
 1 1.00    6150     3718    4600     -882      YES          no
 2 0.00    2400     1600    1500     +100       no          no
 2 0.10    2775     1850    1810      +40       no          no
 2 0.20    3150     2100    2120      -20       no         YES
 2 0.50    4275     2851    3050     -199       no         YES
 2 1.00    6150     4101    4600     -499      YES          no
 3 1.00    6050     4746    4600     +146      YES          no
 4 1.00    6050     4814    4600     +214      YES          no
```

So:

* **The 1->2 upshift lands below the downshift line at EVERY throttle.**
  At WOT it is saved only by an accident: the overrev guard
  (`n_after = n_e*gear[g-1]/gear[g] < n_overrev`) refuses to drop back
  because 1st would be at 6150 rpm. That guard only bites above 93% pedal
  (`n_up >= 5900/0.6045`), so from 0 to 93% pedal nothing structural stops
  the box asking for 1st again.
* **The 2->3 upshift lands below the line for throttle >= 0.20**, same story.
* 3->4 and 4->5 are fine at every throttle (+146 rpm at the worst point,
  WOT) because the ratio steps are close.

The only thing that stops a visible oscillation is `t_shift_lockout = 0.8 s`
and whether the car can gain the missing rpm inside it. Sweep numbers next
section. This is the same failure mode the contract records as already having
happened once ("a TC that cut the pedal made the auto box upshift at 3600 rpm
and hunt") — that fix removed one *cause*; the schedules themselves were
never separated.


---

## 1. PASS — the WOT shift points are right

`N_UP = n_up_a + k*throttle`, `k = n_up_k12 = 3750` (gears 1-2),
`n_up_k34 = 3650` (gears 3-4). At WOT that is 6150 / 6150 / 6050 / 6050.
The correct best-acceleration upshift point is where the NEXT gear's wheel
force crosses the current gear's. Measured on the 19-breakpoint PCHIP
(110 N.m @ 4000, 55.0 kW @ 5600), `eta_drive = 0.86`, `r_roll = 0.283`:

```
gear   F_next/F_cur at 6000   at 6050   at 6100   crossover     N_UP(WOT)
1->2          0.778            0.793     0.811    > 6200 rpm      6150
2->3          0.863            0.879     0.898    > 6200 rpm      6150
3->4          0.985            1.000     1.017      6050 rpm      6050
4->5          0.992            1.006     1.023    ~ 6035 rpm      6050
```

1st and 2nd have ratio steps so large (0.6045, 0.6668) that the next gear
never catches up before the 6200 rpm cut — the right strategy is to hold to
the limiter, which 6150 does. 3->4 and 4->5 cross at 6050 / ~6035, which is
where the schedule puts them. The code comment ("crossover falls at
6051/6031 rpm") is confirmed to 1 rpm. **Nothing to change.**

At part throttle `N_UP` falls to 2400 at a closed pedal, which is a proper
short-shift. Also PASS.

### `tc_scale` separation is intact
`step()` does `load = throttle_map(clamp(inp.throttle)*thr_scale)` and only
THEN `load *= tc_scale`. `_auto_target` reads `inp.throttle` and `inp.brake`;
the launch assist reads `inp.throttle`. Nothing feeds the scheduler a
modified pedal — grep for `tc_scale` in `powertrain.py` returns the dataclass
field, the docstring and the one multiply on `load`. The recorded past defect
("a TC that cut the pedal made the auto box upshift at 3600 rpm and hunt")
cannot recur through that route. `vehicle.py:1001` confirms the vehicle side:
`pi_.throttle = ctl.throttle` unconditionally, `pi_.tc_scale = self._tc(...)`.

---

## 2. DEFECT A — measured hunting

### A held-speed cruise hunts for ever
A speed-holding driver (PI on speed error, the way a person actually cruises),
20 s, through the real `step()` in `_Rig`. Evidence:
`runs/auto_hold_speed_ps1.csv`, `runs/auto_hold_speed_ps2.csv`.

| v_set | grade | mean pedal | shifts / 20 s | shifts/s | sequence |
|---|---|---|---|---|---|
| 6.0 | 0.06 | 0.15 | 13 | **0.65** | 1>2;2>1;1>2;2>1;1>2;2>1;... |
| 6.0 | 0.03 | 0.09 | 11 | 0.55 | 1>2;2>1;1>2;2>1;... |
| 10.0 | 0.06 | 0.20 | 10 | 0.50 | 2>3;3>2;2>3;3>2;... |
| 24.0 | 0.06 | 0.59 | 9 | 0.45 | 5>4;4>3;3>4;4>5;5>4;4>3;... |
| 6.0 | 0.00 | 0.05 | 8 | 0.40 | 1>2;2>1;1>2;2>1;... |
| 14.0 | 0.03 | 0.18 | 7 | 0.35 | 3>4;4>3;3>4;4>3;... |
| 28.0 | 0.06 | 0.74 | 7 | 0.35 | 5>4;4>3;3>4;4>5;5>4;... |

12 of 33 cells at `power_scale` 1.0 and 6 of 33 at 2.0 oscillate, and they do
not settle: a gear change every 1.5 s, indefinitely, at 22 km/h on a light
pedal. **0.65 shifts/s is the worst measured.** A clean cell is 0.00/s.

`1.5 s` is exactly `t_declutch + t_gate + t_engage + t_shift_lockout =
0.70 + 0.80`, i.e. the reversal fires on the very step the lockout expires.
The lockout is not hysteresis, it is a delay; it sets the hunt's PERIOD, not
whether it hunts.

### At a fixed pedal it hunts once, mid-acceleration
`runs/auto_cruise_sweep_ps1.csv`, 12 s at a constant pedal from a constant
speed, 81 cells. 9 cells shift more than twice; the pattern is a spurious
downshift in the middle of a brisk acceleration:

```
thr=0.70 v0=11.0: 0.400s 1>2   4.038s 2>3   5.537s 3>2   7.036s 2>3
thr=0.70 v0=17.0: 0.747s 2>3   2.246s 3>2   3.745s 2>3   8.742s 3>4
```

So at 70% pedal the box goes 1-2-3-**2**-3: it throws away 1.4 s of drive (two
0.70 s open-clutch shifts) doing nothing. That is a direct acceleration loss
and it is exactly what "when automatic ..." would feel like.

### Two mechanisms, one cause
1. **Structural (fixed pedal).** The rpm the engine lands on after an upshift
   is below the downshift line — the table at the top of this note. Only the
   overrev guard saves WOT, and only above 93% pedal.
2. **Closed-loop (a driver trimming the pedal).** `n_dn_k = 3100` is 31 rpm of
   downshift threshold per 1% of pedal. After an upshift the car is in a taller
   gear and slows, the driver adds pedal, the downshift line rises FASTER than
   the engine speed does, and the box drops back. In the lower gear the car
   accelerates, the driver lifts, `N_UP` falls, it changes up. This one hits
   gears 3-5 too, where the structural margin is positive (+146 rpm at worst) —
   which is why the held-speed sweep hunts at 14, 24, 28 and 32 m/s as well.

Both are cured by the same thing: making the two schedules **mutually
consistent** instead of independent.

---

## 3. DEFECT B — the upshift schedule is not brake-aware

`_auto_target` picks `n_dn = n_dn_brake (2200) if inp.brake > 0.3`, but the
UPSHIFT branch never looks at the brake at all, and at a closed pedal
`N_UP = n_up_a = 2400`. 2400 > 2200, so **on a corner entry the box changes UP
before it changes down.** Measured (brake 0.6, throttle 0, from a 0.3-pedal
cruise):

```
v0=22.0 m/s gear 4:  0.40s 4>5@3034;  1.90s 5>4@981;  3.40s 4>3@708;  4.90s 3>2@708
v0=18.0 m/s gear 4:  0.40s 4>5@2473;  1.90s 5>4@730;  3.40s 4>3@708;  4.90s 3>2@708
v0=14.0 m/s gear 3:  0.40s 3>4@2470;  1.90s 4>3@708;  3.40s 3>2@708;  4.90s 2>1@708
```

Four shifts in 4.9 s of braking, **the first one in the wrong direction**, and
the box is still walking down when the car reaches the apex. `power_scale 2.0`
is identical (the schedule does not see the engine). The contract records that
the corner-entry downshift is load-bearing: "Without it the box sits in 4th at
2000 rpm at every apex and the flank-wing A/B difference is buried in gearing
noise." An upshift ON the brakes is worse than no downshift at all.

Secondary: after a full stop from 30 m/s the box is in **2nd**, not 1st — it
is still cascading down when the car stops, 1.5 s per gear.

---

## 4. DEFECT C — kickdown walks down one gear per 1.5 s, then overshoots

Pedal 0.05 -> 1.00, `power_scale 1.0`:

```
5th @ 2004 rpm (60.8 km/h): 0.400s 5>4@2345  1.90s 4>3@3203  3.40s 3>2@4973  5.74s 2>3@5685
4th @ 1937 rpm (46.8 km/h): 0.400s 4>3@2299  1.90s 3>2@3782  6.49s 2>3@5685
3rd @ 1710 rpm (32.4 km/h): 0.400s 3>2@2364  1.90s 2>1@4612  3.40s 1>2@5694
```

* **Pedal-to-engagement is 0.400 s** = `t_declutch + t_gate`, with drive fully
  restored at 0.700 s. That half is exactly on the contract's 0.70 s
  declutch -> gate -> engage. **PASS.**
* Everything after it is the defect. `_auto_target` can only ever return
  `s.gear - 1`, so a three-gear kickdown takes **3 x (0.70 + 0.80) = 4.5 s** of
  walking, with the clutch open for 2.1 s of it.
* And it **overshoots**: from 5th it reaches 2nd at 4973 rpm and then upshifts
  straight back to 3rd (5.74 s). From 3rd it drops to 1st at 4612 rpm and
  upshifts back at 3.40 s. `n_overrev = 5900` is the only brake on the
  downshift, and 5900 is BELOW `N_UP` at WOT (6150 / 6050), so "legal" includes
  a landing rpm the up-schedule will undo within the lockout.
* At `power_scale 2.0` the same walk happens and the reversal is worse
  (`3>4@5593` twice).

---

## 5. DEFECT D — no creep, and an unguarded neutral -> 1st

From rest, `power_scale 1.0`, 12 s:

```
pedal 0.00: v_end 0.00 km/h   x 0.00 m   gear 0     <-- never engages
pedal 0.03: v_end 5.39 km/h   x 14.40 m  gear 1
pedal 0.05: v_end 6.10 km/h   x 14.75 m  gear 1
```

`_auto_target` has `if s.gear == 0 and (inp.throttle > 0.02 or v_x > 0.5):
return 1`. At a closed pedal from rest **neither term fires, so the box sits
in neutral for ever** and the car will not creep. An automatic creeps; more
importantly the driver then pays the whole 0.70 s declutch->gate->engage
before anything happens when they do touch the pedal. Above 0.02 pedal the
creep itself is sane (5.4-6.1 km/h, engine 705-798 rpm, never stalls).

The same branch is the one real hazard I found: `return 1` is **not checked
against `n_overrev` or `_gear_legal`**. `v_x > 0.5` alone will engage 1st at
any speed — at 30 m/s that is 14 800 rpm of input speed dragged through a
200 N.m clutch. It is hard to reach from the auto box on its own, but the
MANUAL edge path (`want = s.gear + (1 if shift_up else -1)`) has no overrev
guard either and is live in auto mode, so five taps of `Q` at 30 m/s walks
5>4>3>2>1>neutral and this branch then puts it straight back into 1st.

---

## 0-50 / 0-100, before any change

Rigid longitudinal rig (`_Rig`) around the real `step()`, auto box, WOT from
rest, `t_shift_lockout`/schedules as shipped:

| | power_scale 1.0 | power_scale 2.0 |
|---|---|---|
| 0-50 km/h | 5.213 s | 2.853 s |
| 0-100 km/h | 14.342 s | 7.231 s |
| shifts to 100 km/h | 1>2 @ 4.83 s, 2>3 @ 9.64 s | 1>2 @ 2.58 s, 2>3 @ 5.09 s |

(The contract's own 5.46 s / 14.80 s are the validate rigs' numbers, measured
differently; these are this note's own baseline, to be compared only against
the after-column of the same probe.)

---

# THE FIX

All four defects have one root cause: **the up-schedule and the down-schedule
were written independently**, and `_auto_target` could only ever return
`s.gear ± 1`. The patch ties them together with three rules and adds one
parameter. `drive/powertrain.py` only.

### New: `n_up_schedule(p, g, thr) -> float`
The upshift threshold out of gear `g`, in rpm; `inf` in top gear and in
neutral/reverse. Factored out of `_auto_target` because **the downshift guard
has to evaluate it for the gear it is dropping INTO** — that is the whole
hysteresis. (This adds one name to the module's public surface; CONTRACT.md
section 3 needs the line, see "patch to route" below.)

### New: `PowertrainParams.v_shift_hyst = 1.8` (m/s)
The hysteresis between the two schedules, as a **road-speed gap** — the
currency a shift map is actually drawn in, and the only one that is uniform
across the box (300 rpm is 0.64 m/s in 1st and 2.01 m/s in 4th).

### Rule 1 — a downshift must clear the target gear's own upshift line
```python
if not braking and abs(v_x) > speed_at_rpm(
        p, g, n_up_schedule(p, g, thr)) - p.v_shift_hyst:
    break
```
With `v_shift_hyst = 0` this blocks **precisely** the reversal of an upshift
the box itself just made — the upshift's own road speed *is* where the car is
the instant after it, identically, at every pedal. The 1.8 m/s gap on top
covers the pedal trim a speed-holding driver makes, which moves `N_DN` at
31 rpm per 1% of pedal and was the second, closed-loop hunt mechanism.

The **upshift is deliberately not guarded symmetrically.** At WOT the 1-2
landing rpm is 882 rpm below the downshift line, so a symmetric guard would
hold 1st to `4900/0.6045 = 8106` rpm — past the 6200 cut, i.e. it would never
let go of 1st at all. The asymmetry is correct and is now written down in the
parameter's comment.

### Rule 2 — a box on the brakes never changes up
```python
if s.gear < len(p.gear) and not (braking and n_e < p.n_overrev):
```
`braking = inp.brake > 0.3`, the same test that already selects `n_dn_brake`.
The `n_overrev` escape keeps a trail-braking driver off the limiter.

### Rule 3 — a throttle-demand downshift picks its gear in ONE decision
`_auto_target` walks down from `s.gear - 1` (the lowest landing rpm) and keeps
the lowest gear that clears both `n_overrev` and rule 1. **On the brake branch
it stops at one gear**, because there the downshift is triggered by the road
speed decaying past `n_dn_brake` and the box should step down as the car slows,
not jump. On the throttle branch the target gear is the driver's torque demand
and must arrive in one shift.

### Rule 4 — the neutral branch
```python
if s.gear == 0 and (inp.throttle > 0.02 or v_x > 0.5 or inp.brake <= 0.3):
    if abs(rpm_at_speed(p, 1, v_x)) < p.n_overrev:
        return 1
```
`inp.brake <= 0.3` is the creep (an automatic sits in gear, not in neutral);
the `n_overrev` test is the guard that was missing — `v_x > 0.5` alone engaged
1st at ANY road speed.

---

## Why skip-shift, with the number
Time from the kickdown state to a target speed, WOT, with the skip capped at
1 gear (the old sequential behaviour) up to 4:

| from | to | cap 1 (old) | cap 2 | cap 3 | cap 4 |
|---|---|---|---|---|---|
| 5th @ 16.9 m/s | 25 m/s | 8.176 s `5>4 4>3 3>2` | 7.144 s | **6.378 s** `5>2` | 6.378 s |
| 5th @ 16.9 m/s | 30 m/s | 12.527 s | 11.495 s | **10.729 s** | 10.729 s |
| 5th @ 22.0 m/s | 30 m/s | 8.133 s | **7.273 s** `5>3` | 7.273 s | 7.273 s |
| 4th @ 13.0 m/s | 22 m/s | 6.861 s | **5.965 s** `4>2` | 5.965 s | 5.965 s |
| 3rd @ 9.0 m/s | 18 m/s | 5.551 s | **4.716 s** `3>1` | 4.716 s | 4.716 s |
| 5th @ 28.0 m/s | 36 m/s | 11.996 s | **11.180 s** `5>3` | 11.180 s | 11.180 s |

Uncapped wins everywhere, by up to **1.80 s**, and cap 3 == cap 4 because the
`n_overrev` guard never lets a four-gear skip through anyway. So the skip is
left uncapped: `n_overrev` and rule 1 are the only limits, which is one fewer
tunable.

## Why 1.8 m/s, with the sweep
33-cell speed-holding sweep (11 speeds x 3 grades, 20 s each) and the 81-cell
fixed-pedal sweep, at both power scales:

| `v_shift_hyst` | flat cells hunting (ps 1.0 / 2.0) | hill cells (1.0 / 2.0) | total shifts / 33 cells (ps 1.0) | fixed-pedal reversals |
|---|---|---|---|---|
| shipped (no guard) | 3 / 1 | 9 / 5 | 118 | 3 |
| 0.0 | 1 / 1 | 10 / 3 | 90 | 0 |
| 0.6 | 1 / 1 | 8 / 3 | 73 | 0 |
| 1.2 | 1 / 0 | 6 / 3 | 58 | 0 |
| **1.8** | **0 / 0** | 6 / 2 | 56 | 0 |
| 2.4 | 0 / 0 | 5 / 2 | 50 | 0 |
| 3.6 | 0 / 0 | 3 / 1 | 36 | 0 |

1.8 is the smallest value that clears **every flat-road cell at both power
scales**. The cost is the effective downshift threshold, which is
`min(N_DN(thr), the cap)`:

```
effective N_DN, rpm        pedal:  0.00   0.20   0.50   0.80   1.00
  raw schedule                     1500   2120   3050   3980   4600
  hyst 1.8   5->4                  1500   2120   3050   3980   4600   (untouched)
             4->3                  1500   2120   3046   3905   4478
             3->2                  1258   1759   2509   3259   3759
             2->1                   938   1391   2071   2752   3205
  hyst 3.6   4->3                  1346   1919   2778   3637   4209   <- erodes
             5->4                  1483   2064   2935   3806   4387      lug
```
At 1.8 the two TALL-gear lines — the ones that protect against lugging — are
untouched below half pedal and at most 122 rpm short at WOT. 3.6 buys three
more hill cells but starts pulling the 4->3 line in at a closed pedal
(1500 -> 1346), so 1.8 it is. The residual hill cells are a genuinely
between-gears condition (the car cannot hold the set speed in the tall gear
and can in the short one) and curing those needs grade logic, which is out of
scope — see "not verified / not fixed".

---

## 6. DEFECT E — wheelspin makes the auto box upshift at a walking pace

Found while doing item 5 (behaviour in a corner / with the aids), on the FULL
vehicle (`drive/vehicle.py`), headless, WOT from 0.5 m/s in 1st, auto box.
Evidence: `runs/auto_spin_upshift.csv`.

The normal 1->2 upshift at WOT is at **13.05 m/s** (`N_UP(1) = 6150` rpm).
At `power_scale 2.0` (the interactive default, `ENGINE_DEFAULT='sport'`) with
TC off on a low-grip surface:

```
mu_scale  TC   shift events (t, gear, engine rpm, ROAD speed u, kappa)
  1.00   off   3.500s 1>2  5685 rpm  u=12.75 m/s          <- correct
  0.45   off   1.695s 1>2  5685 rpm  u= 3.61 m/s  k=-0.05 <- WRONG
  0.30   off   1.965s 1>2  5685 rpm  u= 3.26 m/s  k=+0.10
                3.464s 2>3  5688 rpm  u= 4.70 m/s  k=-0.15
                4.963s 3>4  5674 rpm  u= 6.30 m/s  k=+0.68
  0.45   on    6.025s 1>2  5685 rpm  u=12.68 m/s          <- correct
```

**On mu 0.30 the box walks to 4th gear at 6.30 m/s — 23 km/h, one sixth of
the road speed 4th normally sees.** The car is then undriveable. On mu 0.45 it
is in 2nd at 13 km/h.

**Cause.** The upshift is scheduled on `n_e`, the ENGINE speed, which under
wheelspin is decoupled from the road. The scheduler has no idea the car is not
moving: the wheels reach 6150 rpm of engine speed against nothing, the line
fires, and it changes up. The downshift branch then cannot recover it, because
`n_e` after the upshift is still high (the wheels are still spinning).

The sim's own surfaces put the car here routinely: `T` wet, the open map's wet
square 0.632 and grass 0.55, and the arena's WET_T3. TC ON hides it (the spin
never reaches the line), but `VehicleConfig.tc_on` defaults **False** and it is
a user-toggled setting in the interactive session. `power_scale 1.0` does not
reach it at these grips — it is specifically the owner's 2x engine.

Independently PASSED in the same sweep (`runs/auto_corner_tc.csv`), 15 m/s in
3rd, WOT, 4 s, steer 0 / 3 / 6 / 9 / 14 deg both ways, `power_scale` 1.0 and
2.0, TC off and on:

* the number of shifts is **identical at every steer angle and with TC on or
  off** (one WOT kickdown 3>2 at 0.399 s in every cell) while `tc_gain` drops
  to 0.457 and `kappa_max` reaches 1.077. A TC-reduced load, a cornering
  engine-speed change and a wheel-slip event do **not** make the box shift.
  The only steer dependence is that the straight-line cases reach the 6150 rpm
  upshift inside the 4 s window and the cornering ones do not, which is right.
* So the gearbox's side of the reported "no acceleration while steering" is
  clean; the mechanism was `_tc`'s authority, which `vehicle.py` has fixed.

---

# BEFORE / AFTER

All measured through the real `powertrain.step()`; the wheelspin rows are the
full `drive/vehicle.py`. Evidence on disk under `runs/` (gitignored, so the
numbers are here).

| what | before | after |
|---|---|---|
| **A. hunting, flat road, held speed** (shifts in the settled 2nd half of a 20 s window, 6 m/s, 0.05 pedal) | 4, for ever | **0** |
| **A. hunting, 6% grade, 6 m/s** | 7 | **0** |
| **A. hunting, flat road, 10 speeds x 20 s** | 1 cell hunting | **0 cells** |
| **A. up-then-down reversals** over 404 (gear, pedal) points | 171 (1->2 at 94 of 101 pedals, 2->3 at 77 of 101 from pedal 0.17) | **0** |
| **A. total shifts**, 33-cell speed-hold sweep, ps 1.0 | 118 | **56** |
| **A. fixed-pedal 12 s cruise**, 81 cells, reversal cells | 3 (`1>2 2>3 3>2 2>3` at 0.70 pedal) | **0** (all 5 multi-shift cells are monotone `1>2>3>4`) |
| **B. corner entry**, 22 m/s in 4th, brake 0.6 | `4>5@3034` then `5>4 4>3 3>2`, still in **2nd** at 6 s | **`4>3 3>2 2>1`**, in **1st** at 4.27 s, 0 upshifts |
| **B. corner entry**, 18 m/s in 4th | `4>5@2473` then 3 downshifts | `4>3 3>2 2>1` |
| **B. corner entry**, 14 m/s in 3rd | `3>4@2470` then 3 downshifts | `3>2 2>1` |
| **C. kickdown** 5th @ 16.9 m/s, pedal 0.05->1.00 | 4th at 0.400 s, then 3rd at 1.90, then 2nd at 3.40, then **back to 3rd at 5.74** | **2nd at 0.400 s**, one shift |
| **C. kickdown** 5th @ 22.0 m/s | 4th at 0.400 s, 3rd at 1.90 s | **3rd at 0.400 s**, one shift |
| **C. kickdown**, time 5th @ 16.9 -> 30 m/s | 12.527 s | **10.729 s** (-1.80 s) |
| **C. kickdown**, time 3rd @ 9.0 -> 18 m/s | 5.551 s | **4.716 s** (-0.84 s) |
| **D. creep** from rest, 0.00 pedal, 12 s | 0.00 km/h, 0.0 m, **neutral for ever** | **5.4 km/h, 14.5 m, 1st, 705 rpm**, never stalls |
| **D. neutral -> 1st at 30 m/s** (1st = 14 139 rpm of input speed) | **gear 1** | **refused** |
| **E. wheelspin upshift**, ps 2.0, TC off, mu 0.45 | `1>2` at 1.695 s, road speed **3.61 m/s** | `1>2` at 6.986 s, **12.68 m/s** |
| **E. wheelspin upshift**, ps 2.0, TC off, mu 0.30 | `1>2`@3.26 m/s, `2>3`@4.70, **`3>4`@6.30 m/s** | one `1>2` at **11.29 m/s** |
| | | |
| **unchanged — WOT accel** 0-50 / 0-100 (rig) | 5.213 / 14.342 s | **5.213 / 14.342 s**, same shift trace |
| **unchanged — WOT accel** ps 2.0 | 2.853 / 7.231 s | **2.853 / 7.231 s** |
| **unchanged — WOT 1->2 / 2->3 road speed**, mu 1.0 | 12.75 / 21.25 m/s | **12.75 / 21.25 m/s**, bit for bit |
| **unchanged — shift count vs steer**, 15 m/s in 3rd, WOT, steer 0..14 deg both ways, ps 1.0 and 2.0, TC on and off | 1 shift in every cell | **1 shift in every cell** |
| **unchanged — coast-down**, lift at 30 / 20 / 14 m/s | `4>5`, `3>4 4>5`, `3>4` | same |

The only *behaviour* change outside the six defects is the closed-pedal
`2>1` on a coast-down: it now waits for 3.29 m/s (12 km/h) instead of firing
at 18 km/h, because that is the hysteresis band doing its job. That is more
natural, and braking to a stop still cascades to 1st (the brake branch skips
the guard).

## Validation

| | before (commit `2e3ef24`) | after |
|---|---|---|
| `python3 -m drive.validate --quick` | 75/75, 0 HARD, 0 soft | 75/75, 0 HARD, 0 soft [61.7 s] |
| `python3 -m drive.validate --modules` | 100/100, 0 HARD, 0 soft [231.0 s] | **99/100, 1 HARD, 0 soft [314.7 s]** — see below |
| `python3 -m drive.powertrain` self-check | 82/82 | **90/90** (8 new, every one of which fails on the shipped scheduler) |

**The one HARD failure is `drive/render.py`'s `V22 frame budget`, and it is not
reachable from this work.**

* The check is a wall clock: `mean <= 12.0 ms and p99 <= 16.0 ms` over 600
  rendered frames. The failing runs report mean 5.38-6.02 ms (never close to
  the 12.0 bar) and **p99 16.50-20.52 ms against the 16.0 bar** — over by
  3-28%.
* Standalone, it is **flaky right now**: 6 consecutive identical runs gave
  `ok / FAIL / FAIL / ok / FAIL / FAIL`, p99 ranging 7.56 to 20.52 ms.
* The machine is loaded: `ps` shows **six `python3` processes at 100% CPU**
  (`multiprocessing.spawn` — the concurrent agent's work), and the whole suite
  has gone from 231 s at baseline to 295-315 s, i.e. 27-36% slower, at 87%
  CPU. The baseline 231 s run was on a quiet machine.
* `drive/render.py` imports `math, os, time, collections, dataclasses, numpy,
  pygame, qss, corsa_c, drive.track` — and **nothing else**. It does not import
  `powertrain`, `vehicle` or `tyre`. A change to the shift scheduler cannot
  influence a pygame blit time.

Every other module subprocess is green, including
`python3 -m drive.powertrain -> 90/90` and `python3 -m drive.drive
--self-check -> ALL PASS`. `--quick` (which does not run the module
subprocesses) is **75/75, 0 HARD, 0 soft [61.7 s]**.

### The one acceptance number that moved

| | before | after | band |
|---|---|---|---|
| scripted lap of CIRCUIT_ARENA | 61.10 s, max \|n\| 3.06 m | **60.84 s**, max \|n\| 3.21 m | 55-80 s, \|n\| < 6.0 m |
| `accel_run` 0-100 km/h | 14.69 s | 14.69 s | 14.5-16.0 s |
| open map perimeter lap | 96.85 s | 96.85 s | 90-110 s |
| `drive.py` V27 auto / manual / clutch | `auto 20.0 m/s gear 2; manual 14.41 s; clutch 0.03 / 0.26 / 11.5` | identical | - |
| `drive.py` V29 engine / TC | `stock 14.93 s kappa 0.08 TC 0.0 s; 2x raw kappa 1.50; 2x TC 8.42 s kappa 0.31` | identical | - |
| `drive.py` accel run 0-100 | 14.802 s | 14.802 s | 14.5-16.0 s |
| headless determinism sha | `04f4fa96253ac84d` | `04f4fa96253ac84d` | - |

The lap is **0.26 s FASTER** (0.4%) and stays well inside its band. That is the
expected direction: the box no longer throws a 0.70 s upshift away on every
corner entry and no longer walks down three gears one at a time out of a
kickdown. `accel_run`'s 0-100 cannot move by construction — it carries its own
`n_up` dict and never calls `_auto_target`.

No acceptance number moved: `gearbox modes` still reads `auto 20.0 m/s gear 2;
manual 0-100 14.41 s; clutch stall 0.03 s, fire 0.26 s, launch 11.5 m/s`, and
`Engine setting + TC` still reads `stock 14.93 s kappa 0.08 TC 0.0 s; 2x raw
kappa 1.50; 2x TC 8.42 s kappa 0.31`, both character-for-character identical
to baseline. The contract's `accel_run` 0-100 (14.5-16.0 s band) is untouched
by construction: `accel_run` carries its own `n_up` dict and never calls
`_auto_target`.

---

# DEFECT F — a driver-commanded downshift has no overrev guard (reachable in AUTO)

**Found, measured, NOT fixed.** The scheduler's own `n_overrev` guard is not
applied to the driver's `Q` / `E` edges, and those edges are live in **auto**
mode too (`update_shift` reads `inp.shift_up / shift_dn` before it consults
`inp.auto_gearbox`). `_gear_legal` only refuses an out-of-range gear and
reverse above 1 m/s forward. Four taps of `Q` at 30 m/s in 5th, auto box,
through the real `step()`:

```
  t=0.00s tap -> gear 5,  3557 rpm   (target gear's input speed  4471 rpm)
  t=0.75s tap -> gear 4,  4403 rpm   (                           5614 rpm)
  t=1.50s tap -> gear 3,  5520 rpm   (                           8280 rpm)
  t=2.25s tap -> gear 2,  7545 rpm   (                          13022 rpm)
  t=3.00s tap -> gear 1,  8551 rpm
  PEAK ENGINE SPEED 8553 rpm   (n_cut 6200, n_overrev 5900)
```

**8553 rpm — 2353 past the cut.** The 6200/6050 latch cuts the fuel but the
driveline keeps spinning the engine, which is exactly the case the latch's own
comment names ("the DRIVELINE spinning the engine past the cut on a missed
downshift"). The auto scheduler would refuse every one of these.

I have **not applied this**, on purpose: it changes what a *driver-commanded*
shift does in all three driver models, and CONTRACT.md section 3 pins the
three driver models as a table. It is a one-line change in a file I own, so it
is the owner's call, not mine. The exact patch, in `drive/powertrain.py`:

```python
# OLD
def _gear_legal(p: PowertrainParams, g: int, v_x: float) -> bool:
    if g < -1 or g > len(p.gear):
        return False
    if g == -1 and v_x > 1.0:
        return False        # refuse reverse above 1 m/s forward
    return True

# NEW
def _gear_legal(p: PowertrainParams, g: int, v_x: float) -> bool:
    if g < -1 or g > len(p.gear):
        return False
    if g == -1 and v_x > 1.0:
        return False        # refuse reverse above 1 m/s forward
    if g >= 1 and abs(rpm_at_speed(p, g, v_x)) >= p.n_overrev:
        return False        # the road speed would overrev it: four taps of Q
        #                     at 30 m/s in 5th reached 8553 rpm, 2353 past the
        #                     cut, which is the case the 6200/6050 latch's own
        #                     comment names (the driveline spinning the engine
        #                     past the cut on a missed downshift). The auto
        #                     scheduler already refuses this; the driver's own
        #                     edges did not, in every mode including auto.
    return True
```

Checked against the suite before deciding not to apply it: the only
driver-commanded downshift in any rig is 3rd -> 2nd at 15 m/s (validate G3 and
`self_check._downshift`), whose target input speed is 4274 rpm, so the guard
would not bite. It should be safe; I am simply not the one to widen the driver
model.

---

# NOT FIXED / NOT VERIFIED

1. **Hunting on a grade at mid speed is not cured** and I chose not to chase
   it. 6 of 33 cells at `power_scale` 1.0 and 2 of 33 at 2.0 still oscillate,
   all of them on a 3-6% grade: 10 m/s (0.21 pedal, 3 settled shifts, was 5),
   14 m/s (0.18 pedal, 4, unchanged), 18 m/s, 24 m/s (0.59 pedal, 5,
   unchanged), 28 m/s, 32 m/s. Instrumented, these are **genuinely
   between-gears**: at 14 m/s on 3% the box changes up at 0.091 pedal and the
   driver then needs 0.180 to hold the speed in the taller gear, so both
   decisions are individually right and the car cannot hold the set speed in
   4th but can in 3rd. Real automatics hunt there too; curing it needs grade
   detection or a shift-point memory, which is a new mechanism and a new piece
   of state, not a hysteresis constant. `v_shift_hyst = 3.6` clears three more
   of them but starts pulling the 4->3 lug-protection line in at a closed
   pedal (1500 -> 1346 rpm), which I judged the worse trade. Flagged for the
   owner as a product decision.
2. **Nothing interactive was exercised.** No pygame session, no pad, no
   Settings round-trip beyond what `validate` already covers. The
   `ENGINE_DEFAULT='sport'` path was tested by passing `power_scale=2.0`
   directly, not through the settings page.
3. **The automatic wing-side selection was not audited** — deliberately, it is
   `.handoff/04-wings-audit.md` section 3(b) and the `vehicle.py` owner's fix.
   If "When automatic" turns out to have meant the wing, this note answers the
   wrong question and the gearbox findings stand on their own.
4. **`n_up_schedule` is a new name on `powertrain.py`'s public surface** and
   CONTRACT.md section 3's API block does not list it. I cannot edit
   CONTRACT.md. The line to add, after `def wot_power(p, n_e) -> float`:
   ```
   def n_up_schedule(p, g, thr) -> float    # the auto upshift line out of gear g
   ```
   and section 3's shift-machine bullet wants a sentence about
   `v_shift_hyst` — suggested wording:
   > **Shift-map hysteresis.** The two schedules are not independent:
   > `_auto_target` refuses a downshift into a gear the box would already
   > have changed UP out of at this road speed and pedal, within
   > `v_shift_hyst = 1.8 m/s`, and requires the same road-speed agreement
   > before an upshift (wheelspin decouples `n_e` from the car: at
   > `power_scale` 2.0 on mu 0.30 the box used to reach 4th at 6.3 m/s).
   > A box on the brakes never changes up. A throttle-demand downshift picks
   > its gear in one decision; the brake branch still steps down one at a
   > time. The upshift is deliberately NOT guarded against the downshift line
   > the way the downshift is guarded against the upshift line — at WOT that
   > would hold 1st past the cut.
5. **`drive/render.py`'s self-check went 26/26 -> 25/26 during this session**
   and it is **not mine** — I never touched `render.py` and the powertrain
   change cannot reach it. Between my baseline and now, the concurrent owner
   of `vehicle.py` landed `cd92b54 vehicle+contract: the roll/transfer/panel
   blocks follow the fitted car` and `ceb5e04 vehicle: the flank panel's side
   latch deadlocked after the first corner`, and has `drive/drive.py`
   uncommitted. Reported upward rather than touched; see the final report for
   which of render's 26 checks it is.

---

# EVIDENCE ON DISK (`runs/` is gitignored — these numbers are the record)

| file | what |
|---|---|
| `runs/auto_hold_speed.csv` | 66 cells (2 power scales x 11 set speeds x 3 grades), 20 s each, PI speed-holding driver. Both schedulers side by side. Aggregate: **settled shifts 63 -> 25**, and all 25 survivors are on a 3-6% grade |
| `runs/auto_cruise_sweep.csv` | 162 cells (2 power scales x 9 pedals x 9 speeds), 12 s at a fixed pedal, both schedulers. Aggregate: **downshift reversals 5 -> 0** |
| `runs/auto_corner_tc.csv` | 28 cells, FULL vehicle: 15 m/s in 3rd, WOT, 4 s, steer 0/3/6/9/14 deg both ways, power_scale 1.0 and 2.0, TC off and on. Shift count, mean `ax`, `tc_gain` min, `kappa_max` |
| `runs/auto_spin_upshift.csv` | 12 cells, FULL vehicle: WOT from rest on mu 1.0 / 0.45 / 0.30, TC off and on, both power scales. The road speed at every upshift |

The harness is **kept in the module**: `self_check._auto_hold(v_set, grade)`
is the speed-holding driver, and the seven other automatic checks are unit
tests on `_auto_target`. The sweep drivers themselves are throwaway and live
in the scratchpad.
