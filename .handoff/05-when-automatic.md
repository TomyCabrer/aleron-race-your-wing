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
| 5 | cornering / aids | `tc_scale` separation is **intact**; but see **DEFECT B** — lifting and braking makes the box change UP |
| 6 | other | **DEFECT B** — the upshift schedule is not brake-aware: 4->5 at 3034 rpm on corner entry |

Baseline before any edit (this machine, commit `2e3ef24`):
`python3 -m drive.validate --modules` -> **100/100 pass, 0 HARD, 0 soft [231.0 s]**.

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
