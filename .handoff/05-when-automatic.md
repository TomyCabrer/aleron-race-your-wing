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
| 1 | shift points vs the torque curve | (pending) |
| 2 | hunting / hysteresis | **DEFECT — the hysteresis is NEGATIVE in 1st and 2nd** |
| 3 | kickdown | (pending) |
| 4 | creep and launch | (pending) |
| 5 | cornering / aids | (pending) |
| 6 | other | (pending) |

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

