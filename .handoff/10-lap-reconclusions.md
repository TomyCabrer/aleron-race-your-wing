# Item 3 — the lap-based wing conclusions, re-run after the flank-latch fix

`ceb5e04` fixed a major bug: the flank panel could **never change flanks after
the first corner and never stowed** (`.handoff/04-wings-audit.md` §3(b)). Every
**lap-based** conclusion about the device predates that fix. The per-corner and
open-loop numbers do not — `wing_ab_script` is a skidpad / ramp-steer rig and
the audit established it is unaffected — so this note is only about laps.

The audit itself could not produce a lap-time delta: "the scripted `lap` driver
on the arena completed **0 flying laps** inside the 90 s window in both the
wing-on and wing-off cases". Item 2's driver work removed that obstacle, so the
measurement is now possible for the first time.

## Method

`--script lap`, arena, `margin 0.90`, `wet patch`, stock Corsa C, 200 s, two
flying laps. The **stale** column is the same code with the one-line latch fix
reverted in a scratch copy of this commit (`armed = has and ctl.wing_on and
st.dev_side != 0`), so the two columns differ by exactly that line and nothing
else. "Wrong flank" is the fraction of *cornering* samples (`|r| > 0.05`) with
the panel deployed on the inner flank.

## The numbers

### Fixed (current `main`)

| wing | laps | best lap | `max_n` | deployed | wrong flank | mean `D_dev` | sides seen |
|---|---|---|---|---|---|---|---|
| off | 2 | 60.8355 s | 3.215 m | 0.0 % | 0.0 % | 0.00 N | −1, 0, +1 |
| fin | 2 | **60.7772 s** | 2.575 m | 82.0 % | **0.1 %** | 11.29 N | −1, 0, +1 |
| plate | 2 | **60.6761 s** | 2.395 m | 82.1 % | **0.1 %** | 19.61 N | −1, 0, +1 |

### Stale (the latch fix reverted)

| wing | laps | best lap | `max_n` | deployed | wrong flank | mean `D_dev` | sides seen |
|---|---|---|---|---|---|---|---|
| off | 2 | 60.8355 s | 3.215 m | 0.0 % | 0.0 % | 0.00 N | −1, 0, +1 |
| fin | 2 | 61.0757 s | 3.354 m | 99.8 % | **38.2 %** | 19.93 N | **0, +1 only** |
| plate | 2 | 60.9921 s | 3.760 m | 99.8 % | **38.2 %** | 35.27 N | **0, +1 only** |

## Which prior conclusions survive, and which flip

**FLIPS — the sign of the device's lap-time effect.** This is the headline.

| | stale | fixed |
|---|---|---|
| fin vs wing off | **+0.240 s SLOWER** | **−0.058 s faster** |
| plate vs wing off | **+0.157 s SLOWER** | **−0.159 s faster** |

With the bug present the device was a net **loss** on an arena lap — it carried
its drag everywhere (the panel never stowed: 99.8 % deployed) and spent 38.2 %
of cornering time pushing the car the wrong way. Any statement of the form "the
device does not pay for itself over a lap", measured before `ceb5e04`, is an
artefact of the bug and should be discarded.

**FLIPS — the device's effect on how well the car holds its line.** Stale,
fitting the device made `max_n` *worse* than wing-off (3.215 → 3.354 fin,
→ 3.760 plate), because of the uncommanded side force on the straights. Fixed,
it makes it *better* (→ 2.575, → 2.395), which is what extra front grip should
do.

**FLIPS — the ordering of fin against plate over a lap.** Stale, both were
losses and the plate's larger drag made it the worse of the two to fit
relative to its gain. Fixed, the lap times are monotonic in device strength:
plate 60.6761 < fin 60.7772 < off 60.8355, which agrees with the per-corner
ranking (+3.82 % plate against +2.15 % fin) for the first time.

**SURVIVES — every per-corner and open-loop number.** `wing_ab_script`'s
headline is the open-loop ramp-steer gain in `qss_parity` mode, and a
steady-state rig holds one steer sign throughout, so `want == dev_side`
always and `armed` was identical before and after. The audit verified this
independently: `validate --only D` 9/9 and `--only W` 13/13 bit-identical
across the fix, and `steady_state_corner(100)` with the plate is
`V = 30.415546 m/s` (gain +3.8198 %) in both. **Do not re-derive these.**

**SURVIVES — the top wing's station conclusion.** A rear wing behind the rear
axle unloads the front of this front-limited car (peak `a_y` 0.8550 → 0.8488 g
at `x_t = −1.6`, 0.8804 at `+0.3`). Open-loop, unaffected.

**SURVIVES, and is now larger — the drag saving.** The audit predicted mean
`D_dev` over the lap would fall 34.28 → 20.99 N (−39 %) under the fix. Measured
here on the plate: **35.27 → 19.61 N, −44 %**; on the fin 19.93 → 11.29 N,
−43 %. The mechanism is the panel stowing on the straights, which it never did.

## The caveat that matters most

**The lap delta badly understates the device, and that is by design, not a
defect.** 0.06–0.16 s on a 60.8 s lap is 0.1–0.26 %, against a per-corner gain
of +2.15 % (fin) and +3.82 % (plate) of corner speed at R = 100 m. Three
reasons, all of them deliberate:

1. `speed_profile`'s own docstring says it targets the **no-device** envelope:
   "the wing's job is to be measurable, not to be driven around". So the
   driver never asks for the extra grip — it only helps where the car happens
   to be at the limit anyway.
2. An arena lap is not all limit cornering; the device can only act where the
   radius is inside its useful band, and `crossover.py`'s `R_cap` is the
   statement of how narrow that band is.
3. The device still carries its drag on the 82 % of cornering time it is
   deployed for.

So: use the **open-loop ramp-steer gain** as the device's figure of merit, as
CONTRACT §4 and `wing_ab_script` both insist. The lap numbers here are the
answer to a narrower question — *does fitting it make this particular scripted
lap faster or slower* — and the answer changed sign.

## Not verified

* One track (arena) and one driver margin (0.90). `--script lap` is
  arena-only by construction.
* The Corsa only. The device is a Corsa study and the other two library cars
  have no `DevAero` designed for them.
* The **mount** variants (pylon / endplate / none, task 3) are *not* in this
  table. They change `CZ`/`CD` at design time through `analyse()`, so they are
  an open-loop aero-design comparison, already measured in
  `.handoff/03-wing-mount.md` (endplate: CLα 4.131 → 4.595, cd0 363 → 340
  counts, 4.96 → 3.52 kg). Putting them on a lap would measure the scripted
  driver, not the mount.
* Two flying laps per configuration, not a distribution. The runs are
  deterministic, so the numbers are exact rather than sampled — but a
  different `margin` would move all of them together.
