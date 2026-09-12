# carsim — deployable lateral flank wing on an Opel Corsa C 1.2 16V

Two halves that answer two different questions about the same car and the same
device, and are held to each other numerically.

**The analysis** (repo root) asks *is the device worth it*, in closed form and at
quasi-steady equilibrium.

**The simulator** (`drive/`) asks *what does it do in the time domain*, with the
same parameters, the same tyre and the same load-transfer algebra — and you can
drive it.

---

## Drive it

```
python3 -m drive.drive                          # last map + settings, keyboard or pad
python3 -m drive.drive --track open             # the open proving ground
python3 -m drive.drive --track skidpad --radius 100
python3 -m drive.drive --track dragstrip
python3 -m drive.drive --gearbox manual         # or: auto | clutch
python3 -m drive.drive --engine stock           # the real 75 hp car (default: sport, 2x)
python3 -m drive.drive --sound off              # or: low | mid | high
python3 -m drive.drive --wing plate --wet all   # sealed plate, wet
python3 -m drive.drive --car mx5                # or: corsa | 540i
python3 -m drive.drive --ballast 200 --ballast-at boot
python3 -m drive.drive --camera chase           # the 3D view from behind
python3 -m drive.drive --garage                 # start in the 3D garage
python3 -m drive.drive --ml-drive drive/ml/checkpoints/arena_plate.json
```

| key | | key | |
|---|---|---|---|
| `↑` `↓` | throttle / brake | `F` | flank wing toggle |
| `←` `→` | steer | `G` | wing side (auto / L / R) |
| `LSHIFT` | fine (half rates, 50% pedal) | `T` | wet toggle |
| `Z` | clutch | `R` / `SHIFT+R` | reset / full reset |
| `SPACE` | handbrake | `P` / `O` | pause / single step |
| `E` `Q` | shift up / down | `[` `]` | slow-mo 0.25x / 1x |
| `S` | starter | `C` | camera |
| `H` `V` `B` `N` `X` | HUD / vectors / g-g / skid / clear | `-` `=` `0` | zoom |
| `M` `L` | telemetry marker / record | `TAB` | next map |
| `BACKSPACE` | garage (3D panel editor) | `ESC` | pause menu / settings |

`ESC` (or `OPTIONS` on the pad) pauses and opens the menu: every keyboard and
pad control on screen, plus *Resume*, *Settings*, *Reset to last sector line*,
*Full reset*, *Garage* and *Quit*. `↑` `↓` / d-pad move, `ENTER` / `✕`
select, `ESC` / `○` / `OPTIONS` resume; `R`, `SHIFT+R` and `BACKSPACE` work as
hotkeys inside it. The car does not move and the pad does not rumble while it
is up. `P` is still the plain pause for `O` single-stepping.

## Settings

*Settings* on the pause menu is a second page; `ENTER` / `✕` cycles a value,
`ESC` / `○` goes back. Everything on it is saved to `runs/settings.json` the
moment it changes and reloaded at the next launch, so the sim starts the way it
was left; a command-line flag overrides the file for that launch (and is then
saved). Scripted and headless runs never read the file.

| setting | values | applies |
|---|---|---|
| Map | Arena circuit / Open proving ground / Skidpad / Dragstrip (`--track`, `TAB`) | restarts the session on the new map, same car |
| Car | Opel Corsa C 1.2 / Mazda MX-5 1.8 / BMW 540i (`--car`) | restarts the session: a different car is a different tyre, load set, roll block and gearbox |
| Ballast | None / 25 / 50 / 75 / 100 / 150 / 200 kg (`--ballast`, 0-300) | restarts the session |
| Ballast at | Nose (front subframe, low) / Passenger seat (at the CG) / Floorpan over the rear axle (low) / Boot floor, behind the rear axle (high) (`--ballast-at`) | restarts the session |
| Engine | Stock 1.2 16V (75 hp) / Tuned (~110 hp) / Sport (~150 hp) (`--engine`) | at once |
| Gearbox | Automatic / Manual (auto clutch) / Manual + clutch pedal (`--gearbox`) | at once |
| ABS | On / Off (`--abs` / `--no-abs`) | at once |
| TC | On / Off (`--tc` / `--no-tc`) | at once |
| Steer aid | On / Off (`--no-steer-limit`) | at once |
| Surface | Dry everywhere / Dry, wet patches / Wet everywhere (`--wet`) | restarts the session |
| Camera | Car up / **Chase (3D)** / World up (`--camera`, `C`) | at once |
| Sound | Off / Low / Medium / High (`--sound`) | at once |
| Garage | opens the 3D panel editor | |

**Engine.** The real car is a 75 hp 1.2 that takes 15 s to 100 km/h, and from
the seat that is slow, so the drive starts on *Sport*: the whole torque curve
times two (the clutch uprated to suit; rev limit, overrun, idle and every gear
ratio unchanged), ~150 hp, 0-100 km/h in 8.4 s with the traction control
holding the fronts. *Tuned* is 1.5x (~110 hp, 10.2 s). *Stock* is the car every
scripted and validation number is measured on, and it is one press away; the
HUD shows `75 HP` / `110 HP` / `150 HP` under the gearbox label. No script
ever reads the setting.

**Camera.** *Car up* and *World up* are the plan views the telemetry and the
g-g envelope are read against. *Chase* is a real 3-D view from behind and
above the car: a perspective camera, a horizon, the tarmac ribbon, kerbs and
centre dashes receding to a vanishing point, and the car ahead of you with all
three wings drawn at their actual deployment state - the outer flank panel
sliding out and lighting up mid-corner, the top wing rising off the deck. The
mesh is the garage's, so a wing looks the same on the road as it did on the
ramp. It costs 4.0 ms a frame against the plan views' 2.1-2.4, well inside the
16.7 ms the 60 fps loop has. The HUD is screen-space and overlays it unchanged.

**TC** is an engine-only traction control on the driven axle, the kind the
OPC had: it scales the engine load (never a brake, and never the pedal the
shift scheduler reads) on the front tyres' transient slip, full load below a
slip ratio of 0.12 falling to 15% at 0.20, fast down and slower up. Without
it the 2x car spins its fronts to the limiter through the whole of 1st (slip
ratio 1.5); with it the launch runs at 0.31 and 0-100 takes 8.4 s. It never
acts on the stock car at full throttle on dry tarmac. `TC` lights in the HUD
flag row while it works. Off in every rig, like the ABS.

**Sound** is synthesised, no files: a four-cylinder engine (firing order
harmonics, the half-order idle lump, one exhaust pop per firing, load-shaped
intake noise, the rev-limiter stutter, the starter turning a stalled engine
over) following the physics' rpm and engine load, tyre squeal from the worst
wheel's slip, a grass rumble off the tarmac, wind with speed squared and a
clunk as a gear engages. It streams through one `pygame.mixer` channel about
50-90 ms behind the physics; the settings page re-levels or drops it live, and
a machine without an audio device just drives silently. `python3 -m
drive.audio` checks the synthesis and writes `runs/selfcheck/audio_demo.wav`.

**Gearbox.** *Automatic* shifts itself and works the clutch. *Manual* is a
paddle box: `E` / `Q` or `R1` / `L1` shift, the box does the launch, the
rev-matched downshift blip and the restart, and it cannot stall. *Manual +
clutch pedal* is the H-pattern: `Z` / `□` is the only clutch, so you launch on
it and it can stall; pushing the clutch fully in (or `S`) restarts it, and a
stationary car is handed to this mode in neutral. Reverse is `Q` below 1st
(refused above 1 m/s). The HUD shows `AUTO` / `MAN` / `MAN+CL` under the gear.

**ABS** is a four-channel slip-threshold controller on the hydraulic torque
(never the handbrake), off below 2 m/s. At full pedal it stops the car from
100 km/h in 43.2 m dry against 44.6 m for the best fixed pedal and 49.7 m with
the wheels locked, and 62.0 m wet against 80.5 m locked, with the fronts still
steering. It is on by default in the drive and off in every validation rig, so
no acceptance number is measured through it. `ABS` lights in the HUD while it
cycles.

## The maps

* **Arena circuit** — the 1249.2 m, seven-corner (R = 30 … 130 m) circuit the
  study is validated on, with its wet patches; lap and sector timing.
* **Open proving ground** — a 522 × 362 m rounded-rectangle of tarmac with a
  12 m road round the edge (lap and sector timing on it) and, inside, the
  things a chassis engineer walks a car through: painted skidpad circles
  (R = 50 m, the ISO 200 ft pad), a nine-cone slalom at 18 m, a marked 300 m
  drag lane with 100 m boards and a wet square (μ 0.632). Everything on the pad
  is tarmac; off it is grass. Drive wherever you like; the lap timer only
  counts crossings made on the road.
* **Skidpad** — one constant-radius guide circle (`--radius`, `--cw`).
* **Dragstrip** — 1500 m straight with 1/8 mile, 1/4 mile and km gates.

## Build it: the 3D garage

```
python3 -m drive.drive --garage                  # editor first, ENTER drives
python3 -m drive.drive --build my-car            # a car saved in the library
python3 -m drive.drive --garage --track skidpad --radius 100
```

The garage is always one keypress away: `BACKSPACE` (touchpad on the pad),
the *Garage* entry on the pause menu or on its settings page. `--garage` merely
starts there. A software-rendered Corsa C you can orbit, carrying up to
**three wings**: a panel on each flank and a wing on top. Each is a slot
holding a wing from the library at a station, a height and an incidence;
left and right mirror each other until `M` unlocks them. The side panel shows
what the physics will see for the selected slot: the lift law, the force and
drag at the R = 100 m limit speed, `crossover.gain` at R = 50 / 100 / 130 m
for a flank panel, the front / rear downforce split for the top wing, and the
stall margin. `SPACE` previews the deploy: the flank panels slide out, the top
wing rises off the deck onto its pylons and takes its incidence. `ENTER`
drives that car on the current map; `BACKSPACE` in the drive comes back with
the car still yours. The build is saved to `runs/garage_design.json` and is
the car every later launch drives, until `--wing …` or `--build` says
otherwise.

| garage key | | garage key | |
|---|---|---|---|
| mouse drag / wheel | orbit / zoom | `1` `2` `3` / `TAB` | select the left / right / top slot |
| `←` `→` | station `x` (SHIFT: 1 cm) | `↑` `↓` | height `h` |
| `[` `]` | incidence ±1° | `W` / `SHIFT+W` | next / previous library wing in the slot |
| `M` | mirror left ↔ right | `T` | top wing: fixed / active (brake + steer) |
| `SPACE` | deploy preview (0.45 s actuator) | `D` `A` `L` | designer / airfoils / library |
| `R` / `C` | car defaults / reset camera | `ENTER` | drive it |
| `ESC` | menu | | |

The published car is still here, bit-for-bit: the built-in wings `fin`
(CL 0.70) and `plate` (CL 1.25) are the study's 0.35 m² panel with its fixed
L/D of 3.2, and a build carrying one of them on both flanks and nothing on top
runs the closed-form device exactly as before (the suite asserts the
`VehicleConfig` is identical).

## Design the wings

The designer (`D`) is a port of the car-wing physics of the AeroBO design
tool (`~/dev/urop-bo-aero`) into the game, numpy only, running live:

* **section** — the library holds 39 sections (NACA 4-digit, and UIUC .dat
  files bundled in `drive/aero/data/airfoils`: S1223, E423, CH10, FX 74,
  Clark Y, MH 32 …) plus any NACA code you type (`N`) and any CST shape the
  optimiser produces. A section's polar comes from **XFOIL** when the binary
  is on the machine (`/opt/homebrew/bin/xfoil` here: a sweep takes ~1.6 s in
  a worker thread and is cached under `runs/library/polars`), otherwise from
  a labelled **estimate** (Hess-Smith panel method for the lift slope and
  zero-lift angle, a friction + form-factor + camber/thickness correlation for
  drag and stall). Every read-out says which one it is looking at.
* **planform** — span, chord, taper, tip twist, end plates, and how the wing
  is **mounted**. A horseshoe
  vortex lattice (cosine edges, interlaced stations, tip plates, and for the
  top wing the track's rigid-wall image: ground effect) flies the section at
  each strip's effective angle, reads the polar for profile drag, and finds
  the stall by the critical-section rule. It reproduces the AeroBO lattice
  to 1e-12. What the 1 kHz physics gets is small: `CL = CL0 + CLα·α` clamped
  at the two stalls, `CD = cd0 + cd1·CL + cd2·CL²`, `S`.
* **mount** — `pylon`, `endplate` or `none`, and it is a real aerodynamic
  choice rather than a label. A *pylon* mount stands the wing off on two
  struts and pays for their wetted area plus a 1.3 form factor for the
  junction interference (Hoerner ch. 8). An *endplate* mount carries the wing
  on its tip plates instead: no strut in the flow, but the plates are forced
  to a structural minimum (0.12 m on the top wing) and the reduced tip loss
  then falls straight out of the lattice, not out of a correlation. *none* is
  the mountless idealisation to compare against. Measured on the 1.40 × 0.30 m
  S1223 top wing at h = 0.45 m: the endplate mount is **lower drag and higher
  lift and lighter** than two pylons (CLα 4.131 → 4.595, e 1.199 → 1.505,
  cd0 363 → 340 counts, 4.96 → 3.52 kg) — the pylons' 23 counts buy nothing,
  the plates' 13 counts buy 11 % of lift slope.
* **optimiser** — `O` runs a Gaussian-process Bayesian optimiser (Matérn 5/2,
  expected improvement, Sobol start; ~1 s for 32 evaluations) over span,
  chord, taper, twist, plates and incidence, against *corner-speed gain at a
  drag cap*, *force / drag at a force floor* or *max force at a drag cap*
  (downforce equivalents for the top wing), and shows the best-so-far trace
  against a random search of the same budget so you can see what BO bought.
* **library** — `S` saves a wing; the LIBRARY page (`L`) puts any saved wing
  in a slot of the matching role and saves or loads whole builds, so a wing
  designed for one car goes on the next.

Physics of the top wing, measured on this front-limited car: a rear wing
mounted behind the rear axle *unloads* the front and costs corner speed
(peak a_y 0.855 → 0.849 g); on the roof it helps (→ 0.880 g). The designer
prints the front / rear split so you can see why. In *active* mode it stays
stowed on the straights and comes out under braking or steering.

## Drive it with a PS5 controller

Pair the DualSense over Bluetooth (hold CREATE + PS until it blinks, pick
*DualSense Wireless Controller* in macOS Bluetooth settings). It hot-plugs:
switch it on before or after launch, in the garage or mid-drive. A DualShock 4
uses the same map.

| pad | | pad | |
|---|---|---|---|
| `R2` / `L2` | throttle / brake | left stick | steer (expo 1.5, speed-limited) |
| `R1` / `L1` | shift up / down | `✕` / `□` | handbrake / clutch (hold) |
| `○` / `△` | wing toggle / wing side | `OPTIONS` / `CREATE` | pause menu / reset |
| d-pad `↑` `↓` | HUD / vectors | d-pad `←` `→` | slow-mo / normal |
| `R3` / `L3` | camera / auto zoom | touchpad | garage |

In the garage: left stick moves the panel, right stick orbits, `L1`/`R1`
incidence, `△` type, `□` defaults, `○` deploy preview, `✕` drive, `OPTIONS`
the menu. A button still held across a garage / drive boundary is not a press
in the next screen (the `✕` that chose *Garage* cannot drive straight back).
The pad rumbles (low motor) as tyre utilisation passes 0.85 and on the grass,
and (high motor) when a wheel locks, spins or lifts; `CARSIM_NO_RUMBLE=1`
silences it. The pad has no starter button: in the two assisted gearbox modes
the engine restarts itself, in the clutch-pedal mode `□` fully in cranks it.

The Sony button/axis order is SDL's HIDAPI layout (R2 = axis 5, L2 = axis 4,
both resting at −1; cross = button 0 … touchpad = 15, mic = 16). The axes are
confirmed on a DualSense over Bluetooth on this Mac (6 axes, 17 buttons, left
stick = axes 0/1, both triggers rest at −1); the button identities are from
the SDL table. If a trigger does not rest at −1 once the pad has reported the
sim prints a one-line warning; run
`python3 -m drive.drive --pad-calib`, read the live indices off the console and
drop the corrections in `~/.carsim_pad.json`:

```json
{"steer": 0, "throttle": 5, "brake": 4, "buttons": {"10": "shift_up", "9": "shift_down"}}
```

The gearbox is automatic by default (`--gearbox manual` / `clutch`, or the
settings page). Steering has a speed-dependent soft lock that opens up with body
slip so a slide can be caught; the drive runs it on the understeer gradient
measured from the model (7.82 deg/g — with the spec's 3.2 the aid capped the
driver at 67% of the angle the car needs at R = 50 m and the grip limit could
not be reached from the seat). `--no-steer-limit` or the *Steer aid* setting
removes it entirely, and every validation script runs that way so the aid never
contaminates a measurement.

Four things about the driveline were changed with the driver in mind, none of
which moves a validated number outside its band: a stalled engine can now be
restarted (the starter used to stop at 400 rpm with the fuel off, 100 rpm short
of "running", so every stall was permanent); the assisted gearbox modes blip the
throttle on a downshift so the engine arrives at the new gear's speed instead of
being dragged there through the loaded front axle (clutch slip at engagement
269 rpm instead of 1644 on a 3rd→2nd at 15 m/s); the launch / anti-stall assist
follows the *clutch* mode, not the *shift* mode, so a paddle-shifted manual
cannot stall; and that assist now actually holds its 2400 rpm launch target —
its proportional band used to run all the way down from 550 rpm, and the
clutch's own torque balance then parked the engine at ~1700 rpm transmitting
85 N·m instead of 100, so every launch bogged (0-50 km/h 5.64 s against the
rig's 5.12; now 5.46, 0-100 on the interactive path 14.93 s).

Physics runs at a fixed 1 kHz behind a 60 fps renderer. Real-time factor headless
is about 12x, so it will not miss frames.

## Run it without a window

```
python3 -m drive.drive --headless --script accel     --track dragstrip --duration 40
python3 -m drive.drive --headless --script brake     --track dragstrip
python3 -m drive.drive --headless --script skidpad_limit --track skidpad --radius 100
python3 -m drive.drive --headless --script wing_ab   --track skidpad --radius 100 --wing plate
python3 -m drive.drive --headless --script lap       --track arena --telemetry runs/lap.csv
```

Scripted runs are bitwise deterministic — no wall clock, no accumulator, no
pygame call anywhere in the physics path. Telemetry is a fixed 65-column CSV at
100 Hz with a sidecar `.json` carrying every constant, so a plot is reproducible
without the session.

```
python3 -m drive.plots all      runs/lap.csv --track arena     # overview, g-g, map, laps
python3 -m drive.plots overview runs/lap.csv
python3 -m drive.plots compare  runs/a.csv runs/b.csv --by distance
```

## Pick a different car, and change its weight

The sim was hardwired to the Corsa. `cars.py` is now a small library of
parameter sets, selectable from the CLI (`--car`) and from *Settings*:

| | mass | wheelbase | % front | power | torque | CdA | layout |
|---|---|---|---|---|---|---|---|
| Opel Corsa C 1.2 16V (2003) | 1010 kg | 2.491 m | 61 | 55 kW | 110 N·m | 0.66 | FWD |
| Mazda MX-5 1.8 (NB2, 2001) | 1140 kg | 2.265 m | 52 | 109 kW | 168 N·m | 0.61 | RWD |
| BMW 540i (E39, 1998) | 1780 kg | 2.830 m | 51 | 210 kW | 440 N·m | 0.66 | RWD |

Published figures carry their source on the line; everything no manufacturer
releases — axle weights, CG height, inertias, the whole suspension block — is
marked `est` with a band, exactly as `corsa_c.py` does it. The inertias use the
Corsa's own dynamic index `Izz/(m·a·b) = 0.80` so the three sets are consistent
with each other rather than three unrelated guesses, and each car passes
`corsa_c.self_check`'s two cross-checks (gearing against rpm at Vmax, and the
top-speed power balance).

The Corsa is the default and is **bit-for-bit** the car every acceptance number
was measured on: its `CarSpec` is built by reading every field off
`corsa_c.CorsaC()` rather than retyping it, and the identity is asserted with
`==` on all 39 fields. Every per-car quantity is scaled as
`value(car) = value(Corsa) × ratio`, so for the Corsa the ratio is exactly
`1.0` and the number is unchanged rather than merely close.

**One tyre's worth of data.** `tyre_data/` holds twelve `.tir` files, but only
five load into `drive/tyre.py` (the rest are FITTYP 6.1 or divide by zero), and
**all five carry the identical Magic Formula coefficient set** — they differ
only in geometry. So a car cannot be given genuinely different tyre
coefficients from what is here, and inventing Pacejka data is not something
this project does. Following the contract's own rule ("rescale geometry only"),
every car reads the validated `TNO_car205_60R15.tir` and overrides the geometry
to its own size; the grip difference between a 2003 touring tyre and a modern
performance tyre is carried by `mu_scale` (+5 % / +8 %), which is a **labelled
calibration, never presented as measurement**. Strip it out and the MX-5 is
worth +0.7 % of peak `a_y` and the 540i **−6.4 %** — pure load sensitivity on
1780 kg.

**Weight.** *Ballast* adds mass and it moves everything mass really moves: the
first moments shift `wdist_f` (so `a`, `b` and every static wheel load), the
CG height, and the parallel-axis theorem shifts `Izz`, `Ixx` and `Iyy`. The
**station matters and is modelled**, with honest heights — a hatchback's boot
floor is ~0.65 m, *above* the 0.55 m CG, so a sandbag in the boot **raises** it;
only floorpan ballast lowers it. 200 kg on the Corsa:

| ballast | % front | h_cg | Izz | peak a_y | roll | 0–100 |
|---|---|---|---|---|---|---|
| none | 61.0 | 0.550 | 1200 | **0.8550 g** | 4.55° | 14.80 s |
| nose | 69.4 | 0.512 | 1470 | 0.8389 g | 5.20° | 17.18 s |
| floor | 50.9 | 0.509 | 1585 | 0.8468 g | 4.61° | 17.38 s |
| boot | 49.3 | 0.567 | 1723 | 0.8353 g | 5.31° | 17.42 s |

Every ballast loses grip (load sensitivity). *Floor* is least bad because 41 mm
of CG drop buys most of it back; *boot* is worst outright but moves the balance
from 61 % to 49 % front, which is the knob worth playing with on a wing study.
The three fitted wings are charged too — a designed build is 14.18 kg of wing
and mount, which lifts the CG 7.7 mm.

## Let it drive itself

`drive/ml/` is an optional sub-package holding an evolution-strategy driving
agent. Nothing in `drive/` imports it, the 1 kHz physics path gains nothing
from it, and the sim behaves identically with it absent:

```
python3 -m drive.ml                             # self-check
python3 -m drive.ml.train --iters 90 --pop 32   # train, writes a checkpoint
python3 -m drive.ml.evaluate drive/ml/checkpoints/arena_plate.json --ablation
python3 -m drive.drive --ml-drive drive/ml/checkpoints/arena_plate.json
```

The policy is a 14 → 16 → 4 tanh net of 308 parameters, and it is a **residual
on a hand-written driver** (pure pursuit, a curvature-limited speed target and
a wing rule), so zero parameters *is* that driver and the search spends its
budget on the line and the aero rather than on rediscovering that grass is
slow. It observes only what a driver at the HUD can see: speed, where it is on
the ribbon, heading error, the curvature 15 / 35 / 70 m ahead, sideslip, yaw
rate, lateral g, the two axle utilisations, the panel's deploy fraction and
whether it is on tarmac. Reward is metres of centreline advanced, with going
off the track **ending the episode** rather than costing points — a soft
penalty is exactly what a policy learns to pay. There is deliberately no reward
term for the wing at all: its drag is already in the physics, and whether to
deploy it has to fall out of lap time or it means nothing.

Training runs at `dt = 2 ms` (33× real time a core; 0.128 m of drift against
1 ms over 20 s) and **every reported number is re-measured at the contract's
1 ms**. 2880 rollouts took 23 minutes on twelve cores. On the arena with the
sealed plate:

| | best lap | v_mean | panel deployed | on the outer flank |
|---|---|---|---|---|
| hand-written baseline | 64.565 s | 19.370 m/s | 62 % | 97 % |
| learned | **58.609 s** | 20.999 m/s | 56 % | 97 % |

**+9.22 % a lap.** The ablation is the more interesting half: the learned
policy is 0.637 s a lap faster *with* the panel than without it, and deploys it
**less** than the baseline while going faster — so part of what it learned is
when not to carry the drag. The baseline cannot lap at all without the device;
it had come to depend on the front grip its own wing rule was buying. It is
trained on one track with one aero configuration, so it is a fast lap on a
memorised circuit, not a general driver.

## Check it

```
python3 -m drive.validate              # 82 cross-cutting checks + 2 findings
python3 -m drive.validate --modules    # + each module's own self-check
python3 -m drive.validate --quick --only D
```

100/100 pass with `--modules`, 0 hard failures. HARD means a sign, an identity, a conservation law
or a brake lock order — a HARD failure means a number this sim produces is not to
be believed, and the process exits 1. SOFT means a calibration figure inside a
band, because `corsa_c.brakes` and `corsa_c.dampers` are both literally the
string `MISSING` and no published source has them. Group G covers the driver
side: the three gearbox modes on the real car, the stall → restart, the
downshift blip, ABS dry and wet (and its bit-identical replay), a scripted lap
of the open map plus a crossing of its pad, the settings / menu state machine
(map restart, garage from the menu and from settings), the Engine setting with
the TC on the interactive path (the stock car untouched, the 2x car held at
the drive peak), the launch assist holding its target, and the sound module's
own check. Group W covers the
three wings: the lattice against the AeroBO reference numbers, the designed
panel against the published one (1.4e-14), the top wing's station trade and
deploy logic, the build's two physics paths and the renderer's three-wing
frame; `--modules` also runs the eight `drive/aero` self-checks.

## What it is held to

| | qss.py | drive/ | |
|---|---|---|---|
| corner speed R=50 m | 20.5679 m/s | 20.4265 | −0.69% |
| corner speed R=100 m | 29.0875 m/s | 28.9426 | −0.50% |
| peak lateral acceleration | 0.8625 g | 0.8550 | −0.86% |
| limiting axle | front | front | util_f > util_r at every radius |
| roll at the R=100 limit | 4.599 deg | 4.553 | 5.33 deg/g |
| total lateral load transfer | 3266.6 N | 3266.7 | |
| Vmax in 5th | 47.2 m/s | 47.2174 | 47.253 kW needed vs 47.300 available |

The sim comes in **under** `qss.py` by design and the suite asserts it strictly:
`qss.axle_capacity` sums each tyre's own peak lateral force, i.e. it puts both
tyres of an axle at their individual peak slip angles at the same instant, which
is impossible when they share one slip angle. Matching `qss` exactly would mean
`axle_capacity` had been reimplemented somewhere.

The tyre is the same tyre. `tyre_data/TNO_car205_60R15.tir` (MF6.2, FITTYP=62) is
the *provenance* of `qss.TYRE`: its `PDY1 = 0.8784`, `PDY2 = -0.06445`,
`FNOMIN = 4000` give μ(2477) = 0.90294 and a slope of −16.11e-6/N, which is
`mu_ref=0.903, s=-16.1e-6` rounded. Monkey-patching `qss.fy_max` with the full
Magic Formula peak moves `qss.corner_speed` by less than 0.0015 m/s.

## The device

At R = 100 m, sealed plate (CL 1.25, S 0.35 m² one panel) at the front-axle
station, dry, in qss-parity mode:

| route | gain |
|---|---|
| `crossover.gain` closed form | **+2.1996%** |
| this sim, two-track, multiplier emergent | **+2.2029%** |
| `qss.py` as shipped | +1.1815% |

The first two share no code beyond `corsa_c.py` — one imposes the
`(x_w + b)/b` multiplier, the other sums per-wheel forces and lets it fall out of
the yaw balance. They agree to **0.003 percentage points**. The third disagrees
because of a bug, below.

Force scale, so nobody mistakes this for a downforce car: 123 N (clean fin) to
220 N (sealed plate) against a 9908 N car. The renderer draws that arrow at the
same px/N as the tyre arrows, so it is about 3 px. You will not feel it. The
evidence is the numbers, not the seat.

Two things the sim will not let you get away with:

* **Above the cap only by going rear-limited.** With the car's own side force and
  the device's body-slip incidence bonus switched on, the gain reads +3.16% and
  +3.82% — but `util_r` hits 1.000 and the limiting axle flips to the rear. On a
  FWD hatch that is a spin, not a lap time. `wing_ab` prints `cap_pct` and
  `rear_limited` next to every gain.
* **The closed-loop skidpad is not a measurement.** Its path-following
  controller saturates about 10% below the car (26.27 m/s against 29.30 at
  R = 100 m) and does not saturate in the same place with the wing on, so its A/B
  difference comes out at +6.9%. It is reported as
  `gain_pct_closedloop_DRIVABILITY_ONLY` and nothing else uses it. Every
  quantitative number comes from the open-loop ramp steer.

## Two findings about the analysis, not the sim

Neither is patched. `corsa_c.py`, `qss.py`, `crossover.py` and `ledger.py` are
unmodified by this package. `python3 -m drive.validate` prints both with numbers.

**1. `qss.residuals` — `Y_r` violates qss's own force balance.** `qss.py:88` has

```python
Y_r = (car.a * car.m * a_y + F * x_w) / L
```

Solving the two equations `qss.py` itself states (`Y_f + Y_r + F = m*a_y` and
`Y_f*a - Y_r*b + F*x_w = 0`) gives `Y_r = (a*m*a_y - F*(a - x_w))/L`. The `-F*a/L`
is missing. With the wing off (`F = 0`) they agree exactly, so every baseline
number stands. With the sealed plate at the R=100 limit `Y_r` comes out 2.60%
high and `Y_f + Y_r + F` misses `m*a_y` by 86.6 N — which sends the rear axle
spuriously limiting, and is why `qss.sweep()` prints the sealed plate (+1.18%)
as *worse* than its own clean fin (+1.21%), impossible for a front-limited car.

**2. `alpha_peak_deg = 7.0` is about 3.4 deg low.** The Magic Formula front-axle
peak at the R=100 limit load split (outer 5463 N, inner 581 N) is 10.34 deg, and
the sim measures 10.05 deg on the front axle there. Scrub drag goes as
`sin(alpha_peak)`, so the term is 1.47x larger than assumed:

| | at 7.0 deg | at 10.34 deg |
|---|---|---|
| `qss.corner_speed(130)` | 31.880 m/s | 29.038 m/s |
| `qss.corner_speed(175)` | 34.012 m/s | 31.222 m/s |
| `crossover.sustainable_V(0.87)` | 30.674 m/s | 24.772 m/s |
| → `R_cap` | 110.2 m | 71.9 m |

That shrinks the radius band over which the device can act at all, which matters
directly to `ledger.py`'s verdict.

## Layout

```
corsa_c.py      parameter dataclass. The only source for anything it publishes.
cars.py         the car library: CarSpec (a superset of CorsaC), three cars,
                added mass as point masses. The Corsa entry is COPIED off
                corsa_c.CorsaC() field by field, so the default cannot drift.
crossover.py    closed-form kill-or-continue: lateral vs downforce, R_cap, gain
ledger.py       lap-time ledger swept over installed CL and installed mass
qss.py          quasi-steady two-track cornering solver — the reference truth
tyre_data/      real .tir Magic Formula files + mf_eval.py
specs/          the five subsystem specs the simulator was built from
drive/
  CONTRACT.md   authoritative interfaces, conventions and reconciliations
  tyre.py       MF6.2 combined slip (Fx, Fy, Mz), symmetrised, from the .tir
  powertrain.py Z12XE curve, saturated-PD clutch, F13 box, open diff, brakes,
                shift machine with three driver models, blip, restart
  vehicle.py    two-track EOM, load transfer, roll, aero + device, integrator, ABS, TC
  track.py      circuit / open proving ground / skidpad / dragstrip, projection,
                surfaces, world-space areas
  input.py      keyboard and gamepad (PS5 layout, hot-plug, rumble), ramps, aid,
                gearbox modes
  render.py     pygame plan views, the 3-D chase camera, and the HUD
  audio.py      procedural engine / tyre / wind / grass sound, streamed via pygame.mixer
  garage.py     software-3D editor: place the flank panel, closed-form readout
  menu.py       pause / help / settings menu (ESC, OPTIONS) shared by the drive and the garage
  telemetry.py  fixed-schema CSV + sidecar json
  plots.py      overview, g-g, track map, laps, A/B compare
  drive.py      main loop, settings, garage <-> drive session loop, CLI,
                scripted virtual drivers
  ml/           optional: an evolution-strategy driving agent (numpy only).
                Nothing in drive/ imports it; --ml-drive is its only hook.
  validate.py   the acceptance suite
runs/           telemetry, plots, settings.json, garage_design.json
```

Read `drive/CONTRACT.md` before changing anything in `drive/`. It records the
conventions (ISO body frame, `y` left, wheel order FL FR RL RR), the eight places
the subsystem specs disagreed and which one won, and the two findings above.

Three things in there are load-bearing and are commented as such in the code:
the integrator ordering (advance the slip states from the *current* wheel speed,
take the force from the *new* slip states, update the wheel speed implicitly —
the reversed order diverges at every timestep below 8 ms at 3 m/s), the device
side taken from the *steering command* and never from body slip (`beta` is
negative in a left turn, so a beta-derived side deploys the panel on the wrong
flank exactly when it matters: −2.14% instead of +2.35%), and `roll_dist_f = 0.74`
being a calibration constant rather than a suspension property (a proper buildup
from the published roll centres gives 0.51 and caps at 0.726).
