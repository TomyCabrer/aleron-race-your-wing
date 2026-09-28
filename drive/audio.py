"""Procedural car sound for the interactive drive.  No sound files.

A physically informed synthesiser: every layer is driven by a quantity the
simulator already publishes in `HudData`, nothing is a loop or a sample, and
the whole mix is rendered in numpy one CHUNK at a time.  The author cannot
listen, so every layer is pinned by the self-check on its spectrum, envelope
or level, and what it SHOULD sound like is written beside it.

What you hear
-------------
* **Engine**, one of four PROFILES read off the HUD every frame
  (`hud.car_key`, else a guess from `hud.car_name`, else the Corsa).  Every
  cylinder FIRING is an event placed at sub-sample precision from the crank
  phase: a four-stroke cycle is two revolutions, so a four fires rpm/30 times
  a second and a V8 rpm/15.  Each event is an exhaust blowdown pulse: a
  sharp front (the valve cracking open - a one-pole at 0.65-4 kHz, sharper
  with load) and an exponential decay lasting a fraction of the firing
  period, briefer at WOT, with its own amplitude (load = combustion
  pressure, the cylinder's fixed imbalance, cycle-to-cycle jitter: 10 % at
  idle, 6 % loaded, rougher again as the throttle shuts) and a timing
  jitter, so the idle LOPES.  (A symmetric pulse was tried first: the idle
  came out as a 2 ms click train with dead air between, centroid 150 Hz,
  inaudible on a laptop speaker.)  Each pulse carries EXHAUST-FLOW NOISE -
  turbulence at the pulse train's own level, swelling at each firing,
  growing with load x rpm, a floor of it left on a shut throttle at high
  rpm - and the train runs through the exhaust: a feedback comb (the
  pipe's round trip, 2L/c), four resonant band-passes (header / muffler /
  tailpipe modes and a low muffler body that keeps the sim's slow real idle
  from being dead air), a gentle waveshaper (rasp) and a low-pass that
  opens with load and rpm - overrun is a soft dull hum, WOT is bright.
  Intake roar is band-passed noise gated by the firing phase, growing as
  rpm^1.8 x load; a mechanical bed (valvetrain, chain, injectors) and a
  quiet valvetrain tick sit under it all.
  Measured at WOT (1500 / 3000 / 4500 / 6000 rpm): 4-17 % of the engine's
  energy off the cycle harmonics (the review found 0.2-3.6 %: an organ), no
  single line over 41 % at 6000 rpm (the Corsa's 200 Hz was 72 %), and the
  three cars in order bright to deep at every rpm (centroid Corsa 545 / 664
  / 760 / 827 Hz, V8 230 / 362 / 468 / 588; the V8 used to sit above the
  Corsa from 3000 rpm; the retired MX-5's sat between them).  Levels (Corsa): 3000 rpm
  WOT -18.3 dBFS RMS (-17.0 before; see A_ENGINE), overrun 13 dB under it;
  the WOT level rises to the cut on every car (no resonance dip: within
  0.5 dB of the loudest point).  Overrun (load 0, 4500 / 6000 rpm) is 4-15 %
  off the harmonics; it was a pure tone, 97-99 % on them (JIT_OVR).  At the
  sim's REAL in-gear idle (551-586 rpm, load 0.24) no 1 ms frame is 40 dB
  under the firing peaks (the Corsa's were 54 %).
  A CONFIRMED lift from high rpm arms a pop window: a per-lift budget of
  exhaust POPS (Corsa 0-2, rally 4-8, V8 2-4; each a WOT-sized blowdown burning
  in the pipe - a dull thump plus a crack, never louder than the WOT peaks)
  spread over ~1.5 s, thick at first, and crackles.  Every shift the physics
  makes starts with a lift (the declutch), so a lift counts only when the
  gear stays in for T_LIFT_CONFIRM, or - the auto box's lift-off upshift -
  when the new gear is in and the load stays off (review: 4-31 pops per
  WOT upshift before).  The rev limiter cuts ignition in a stutter (firings
  dropped in bursts at LIMIT_HZ, the odd pop as it re-lights); an upshift
  cuts it for 70 ms.  A stalled engine on the starter is the starter's gear
  whine, slowed and loaded once per compression.
  - `corsa` 1.2 16V I4: small and buzzy - short pipe, higher modes, the
    brightest pulse.
  - `rally` 2.0 BDA I4 (the Escort RS1800, task 46): a crisp race four
    revving to 9000 - a short 4-2-1 manifold (alternate pulses differ, so
    the CRANK order sits under the firing tone), the brightest note from
    3000 rpm up, the loudest induction (open trumpets), a lumpy race-cam
    idle and the most pops and crackle on a lift.
  - `540i` 4.4 V8, cross-plane: deep - each bank's four pulses are UNEVEN
    (270 / 180 / 90 / 180 deg apart), which is the burble.  The two banks'
    non-firing lines are in exact anti-phase (together they are an even
    90-deg train), so the burble lives in the DIFFERENCE of the banks: mono
    weights them 0.66 / 0.34 (a listener nearer bank 0's tailpipe), and in
    stereo each channel has its own asymmetry - left bank 0 alone, right
    bank 1 over bank 0 - so the burble is in L, in R and in the fold L+R
    (non-firing / firing lines at 3000 rpm WOT: 1.98 / 0.17 / 0.17, mono
    0.20; the last round's pans left R at 0.017, symmetric ones L+R 0.008).
  - `express` 1.4 I4 (the Renault Express van, task 41): a plainer, lower
    four than the Corsa's - a longer pipe, more mechanical bed and tappet
    tick, hardly a pop (WOT centroid 467 / 566 / 646 / 722 Hz, under the
    Corsa's and over the V8's at every rpm).
  (Task 46 retired the MX-5's rorty 1.8 and the Citaro's OM 906 turbo-
    diesel six with their cars; the profile fields the diesel brought -
    `idle_hi`, `rpms`, the cut-scaled demo margins - stay for any engine.)
* **Drivetrain**: a faint final-drive whine at a frequency proportional to
  road speed (straight-cut reverse: higher and ~8x louder), and ONE clunk per
  gear change as the physics makes it: g -> 0 (the 0.25 s neutral gate) ->
  g' is one shift, clunking when g' engages, with the ignition-cut blip on
  an upshift; a neutral that outlasts T_GATE_MAX clunks lightly on its own,
  and neutral-in / reverse are the heaviest.  The clunk's body is ~210 Hz
  (it was 118 Hz: 90 % of it under a laptop speaker's band).
* **Tyres**: the squeal is resonant band-passed noise (Q 9 around 1 kHz)
  plus a wandering tonal chirp, driven by the same slip thresholds as before
  (lateral utilisation past 0.90, |slip ratio| past 0.10) and panned toward
  the wheels doing the sliding (slip power |Fy tan(alpha)| + |Fx kappa| per
  side).  Below it, from utilisation 0.80 to 0.95, a low SCRUB (band noise
  at 380 Hz): the tyres audibly coming up to the limit.  A locked wheel
  (kappa < -0.10) morphs the squeal lower and rougher (a 43 Hz AM); ABS
  chops it at 15 Hz and adds a 15 Hz chatter to the chassis thump.  The
  lock-up share, the band centre and the ABS gate ramp across each chunk.
* **Road** (`hud.surf4` per wheel, else `on_track`): a tarmac rolling hum
  growing with V; spray hiss on 'wet'; on 'kerb' the rumble strip - a thump
  at V / 0.35 m through the chassis resonances, on that side; on 'grass' a
  soft swish and random thumps; on 'gravel' dense crunchy impulses (no
  bump or stone over EXP_CAP x the mean: a lone one hit 0.90-0.91).
* **Air**: wind as band-passed noise, (V/45)^2, its band rising with V and
  its level breathing with slow gusts (decorrelated per channel: width).
  The ACTIVE WING's actuator: a small servo whine only while a flank panel
  or the top wing is MOVING (the time derivative of `wing_deploy_l/_r`,
  `top_deploy`), on the panel's side - audible, but the smallest layer.
* **Cues**: short procedural chimes on the rising edge of `sector_flash`
  (purple: a bright rising fourth, green: one mid note, red: a soft falling
  pair), queued one after another, below the engine. The lap's own chime -
  a new PB, a new best medal - is task 27's `CarSound.chime` (`CHIMES`,
  `chime_wave`), which `drive.Sim._rec_lap` calls once per lap: it knows a
  NEW best medal from a medal merely earned again, which a HUD edge cannot,
  so the lap end is chimed there and only there (never twice).
* **Mix**: stereo when the mixer grants two channels (the engine centred,
  the V8's banks spread, tyres / kerbs / surfaces / servo panned), mono
  otherwise; a chunk-rate bus compressor keyed on the chunk's peak (at most
  BUS_CREST x its RMS), its gain ramping across the chunk from the last
  chunk's value - so a peak early in a chunk meets the old gain and the tanh
  soft clip under CEIL catches it - then a DC blocker (the soft clip
  squashes a pulse train's tall fronts more than its lobes).  Volume 0 and
  a dead engine at rest are exact digital zeros.

How it streams
--------------
Everything is synthesised per CHUNK samples, phase- and state-continuous
across chunks: every oscillator keeps its phase, every filter its state
(scipy.signal.lfilter's C core with carried zi when scipy imports; else
`_IIR`'s own vectorised form, a partial-fraction split into complex one-pole
sections solved with a segmented cumsum, equal to lfilter to 1e-13, and
handed over to lfilter mid-stream without a step), the comb its last D
outputs, every impulse train its sub-sample carry, and every level ramps
linearly inside the chunk from its previous value.  The chunk goes to ONE
pygame.mixer Channel with play() + queue(), kept one chunk ahead, so the
sound trails the physics by one to two chunks (50-90 ms) and a dropped frame
has a chunk of slack before the stream underruns (counted in `underruns`).
One-shots raised by a frame (a gear clunk, a chime) are LATCHED until the
next chunk is synthesised: a chunk is only rendered every ~3 frames.

Rules
-----
* Physics never reads this module.  `update(hud)` is called from the RENDER
  loop once per frame with the frame's HudData, exactly like the pad rumble
  (CONTRACT section 8): the sound is a consumer of the sim, never an input.
  It imports numpy and pygame (and scipy.signal when present), nothing from
  the simulator.
* Synthesis time IS frame time (it runs inside `update`): <= 1.5 ms mean per
  2048-sample chunk, 3 ms hard cap, measured by the self-check.  Measured on
  this Mac under a load average of 7-9: the worst case (V8, stereo, every
  layer at once, one-shots firing) 1.29-1.34 ms mean / 1.45-1.65 ms p99
  (1.17-1.21 / 1.31-1.42 before the flow noise, the bed and the fourth mode,
  interleaved in one run; the Corsa and MX-5 1.15 / 1.3), a Corsa driving in
  mono ~0.7 ms; the numpy filters 1.5-2.0 ms.  A chunk is rendered every
  ~2.8 frames.  The
  HUD mapping runs EVERY frame: 0.011 ms.  Building a CarSound costs ~1 ms:
  scipy.signal (0.27-0.49 s to import) and the chimes (~30 ms) come in on
  `warm_up`'s background thread while the synth runs its numpy filters.
  The game calls `drive.audio.warm_up()` once at session start, Sound on or
  off (0.1 ms, no mixer), so neither ever lands in a frame.
* Every HUD read goes through `_num` / `_int` (finite, a default, a range)
  and every Synth target too; a chunk that still comes out non-finite resets
  every state and plays silence, and the next chunk sounds again.
* The mixer is a wall clock; nothing here is deterministic in the sim's
  sense and nothing here needs to be.  Its own RNG is seeded, so the offline
  self-check and the demo WAVs are repeatable.
* No audio device (or no numpy) means `CarSound.ok` is False and the drive
  is silent; nothing raises out of here.

`python3 -m drive.audio` checks it (continuity at chunk boundaries, the
firing orders in the spectrum, each profile's character and order, the
overrun's roughness, the V8's burble per channel, the real idle, the pops'
budget and arming through the real shift timing, one clunk per shift, the
kerb / gravel / squeal / scrub / lock / ABS / pan / servo / whine layers,
the chimes' edges, silence, DC, junk HUD values, the numpy filter fallback
against lfilter, the synthesis time, the streaming path on SDL's dummy
driver, `warm_up` in a fresh process) and writes ONE lap-like demo per car
to runs/selfcheck/audio_demo_<car>.wav, overwritten each run.
"""
from __future__ import annotations

import collections
import math
import os
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np

# ---- stream --------------------------------------------------------------
RATE = 44100            # Hz  asked of the mixer; whatever it grants is used
CHUNK = 2048            # samples per synthesised block, 46.4 ms at 44.1 kHz:
#                         the stream is kept one chunk ahead, so this is also
#                         how late a render frame may be before an underrun
BUFFER = 512            # mixer callback buffer (11.6 ms)
CHANNELS = 2            # asked when this module opens the mixer itself; a mixer
#                         already open mono is used as is (the synth folds)

# ---- engine (shared) -------------------------------------------------------
N_CYL = 4               # the Corsa's; each PROFILE carries its own slot count
FIRINGS_PER_REV = N_CYL / 2.0       # four-stroke: every cylinder once per 2 rev
A_ENGINE = 0.45         # master level of the engine mix.  0.62 before the flow
#                         noise: the brighter, noisier exhaust is louder into
#                         the bus, which then ducked tyres / wind -9.3 dB at
#                         6000 rpm WOT (Corsa; -5.7 before).  0.45 costs the
#                         engine 0.8 dB at 3000 WOT and ducks 1.8 dB less
RPM_CRANK = 60.0        # rpm below which the engine is silent
RPM_RUN = 450.0         # rpm at which the crank sound has become the engine
ENG_A0 = 0.30           # overrun pulse / WOT pulse (no combustion: pumping only,
#                         est; recordings put overrun 12-16 dB under WOT)
LIMIT_HZ = 16.0         # rev-limiter stutter rate (est: a hard-cut road ECU
#                         bounces 10-20 Hz; the physics' latch is 1 kHz-fast
#                         and the HUD samples it at 60 Hz, so it is re-made here)
LIMIT_DUTY = 0.55       # fraction of each stutter period that is cut
CUT_FLOOR = 0.06        # a cut firing still pumps: its pulse at 6 %
T_SHIFT_CUT = 0.070     # s  ignition cut on an upshift (est, 50-100 ms road ECUs)
POP_GAIN = 1.0          # a pop's pulse vs a WOT firing AT THE SAME RPM.  1.8 x
#                         a fixed level made the pops the loudest events in the
#                         mix (0.83-0.91 peaks against 0.69-0.75 for WOT, review);
#                         a pop is now a WOT-sized blowdown through the overrun's
#                         closed-down filters: a dull thump plus its crack
POP_TAU = 0.70          # s  the pop window after a lift decays with this, and the
#                         lift's pops are spread over it (exponential times)
POP_T_MAX = 2.0         # s  no pop of a window lands later than this
A_CRACK = 0.22          # a pop's crack (ring amplitude at 3.2 kHz)
A_CRACKLE = 0.08        # a crackle's
A_TICK = 0.025          # the valvetrain tick at idle, x the profile's tick
A_INTAKE = 0.50         # intake roar at WOT at the cut (x profile intake): it
#                         grows as rpm^1.8, a roar at the top, nothing at idle.
#                         0.20 measured 15-19.5 dB under the engine at WOT
#                         (review); this is +8 dB
A_FLOW = 1.20           # exhaust-flow noise: turbulence at the pulse train's own
#                         level, swelling at each firing, x (0.65 + 0.35 rpm /
#                         cut) x load.  Without it 96-99.8 % of the engine's
#                         energy sat on the cycle harmonics - an organ (review);
#                         with it 4-17 % is off them at WOT
FLOW_OVR = 0.30         # the flow noise does not stop with the throttle shut: its
FLOW_OVR_X0 = 0.30      # load factor has this floor at the cut, rising linearly
#                         from FLOW_OVR_X0 x cut (est: manifold pressure on a
#                         closed-throttle overrun is ~0.2-0.3 bar, so ~0.3 of
#                         WOT's gas still goes through, and past the nearly shut
#                         throttle plate it is sonic - a hiss)
JIT_OVR = 1.5           # overrun roughness: above the idle band the pulse
#                         amplitude SD grows by this x the profile's own jitter
#                         x (1 - load)^2 (est: pumping-only pulses, no
#                         combustion to set them, reversion at every valve
#                         opening).  With FLOW_OVR: overrun - which plays
#                         through the ~0.4 s zero-load declutch + neutral of
#                         every real upshift - was 97-99 % on the cycle
#                         harmonics, a pure tone (Corsa 6000: 98.9 %, top line
#                         0.79; verifier); off them now at 1500 / 3000 / 4500 /
#                         6000 rpm, load 0: Corsa 7 / 7 / 5 / 4 %, MX-5 6 / 9 /
#                         4 / 8 %, V8 2 / 5 / 13 / 15 % (was 1-6 %), top line
#                         0.76 at 6000; WOT untouched (4-17 %)
A_MECH = 0.0030         # the mechanical bed (valvetrain, chain, injectors):
F_MECH = 1500.0         # Hz, a broad band (Q_MECH) so the sim's real in-gear
Q_MECH = 0.7            # idle (551-586 rpm, a firing every 27-51 ms) is not
#                         dead air between firings (54 % of 1 ms frames 40 dB
#                         under the peaks on the Corsa, review)
JIT_LOAD = 0.06         # pulse amplitude SD under load (was 0.04: COV of IMEP
#                         is 2-5 % loaded; the exhaust adds its own)
DC_HZ = 30.0            # DC blocker on the pulse train (V8 idle firing = 37 Hz)
DC2_HZ = 18.0           # ... and after the waveshaper, whose y|y| is even-order:
#                         +0.009 DC (3.6 % of RMS) on the V8 at 6000 rpm (review)
PULSE_H = 2.30          # pulse height at the reference decay corner FC_REF;
#                         with A_ENGINE, a Corsa at 3000 rpm WOT is 0.12 RMS
FC_REF = 120.0          # Hz
PULSE_GAMMA = 0.65      # height ~ fc^gamma: 0 = every pulse the same HEIGHT
#                         (blowdown pressure is set by load), 1 = the same AREA
#                         (the same mass per firing leaving in less time, so a
#                         taller volume-velocity pulse).  Constant height made
#                         3000 -> 6000 rpm at WOT only +0.6..2.7 dB; recordings
#                         rise 6-10 dB.  0.65 with the rpm law in _engine gives
#                         +3.1 / +3.7 / +5.3 dB (200 Hz - 4 kHz band, Corsa /
#                         MX-5 / V8) through today's bus compressor, which
#                         takes 1-3 dB of it at 6000 rpm
RISE_LO = 650.0         # Hz  the pulse front's corner closed-throttle (est: a
RISE_HI = 4000.0        # Hz  throttled blowdown is blunter) ... and at WOT
PULSE_LOAD = 1.0        # the decay corner is pulse_k x f_fire x (0.75 + this x
#                         load): a WOT blowdown is a briefer, harder slug, so
#                         harmonics 3-10 carry more (was 0.5)
RASP_REF = 0.30         # the waveshaper's y|y| is taken relative to this level

# ---- shifts and lifts, as the physics does them (powertrain.update_shift) -------
T_LIFT_CONFIRM = 0.22   # s  a lift arms the pops only if the gear has not gone
#                         to neutral by then: EVERY shift starts with the
#                         declutch - load 0 at the OLD gear for t_declutch
#                         (0.15 s) - then gear 0 for t_gate (0.25 s).  Arming
#                         on the lift gave 4-31 pops per upshift (review)
T_ENGAGE_CHECK = 0.15   # s  ... or, when a shift followed the lift, the new gear
#                         in and the load still off this long after (at WOT
#                         the engage ramp is past 0.3 load in ~2 frames)
T_LIFT_GIVE_UP = 1.5    # s  a lift neither confirmed nor cancelled by then is dropped
T_GATE_MAX = 0.45       # s  g -> 0 -> g' with the neutral shorter than this is
#                         ONE shift (t_gate 0.25 s + frame jitter + a hitch);
#                         a neutral that lasts is its own, lighter, clunk

# ---- drivetrain --------------------------------------------------------------
K_WHINE = 30.0          # Hz per m/s: final-drive mesh (~60 ring teeth at the
#                         wheel's 0.55 rev/s per m/s, R 0.29 m) -> 900 Hz at 30 m/s
K_WHINE_REV = 190.0     # Hz per m/s: the straight-cut reverse idler
A_WHINE = 0.010         # faint; reverse is REV_WHINE_X louder
REV_WHINE_X = 8.0
A_CLUNK = 0.30
A_STARTER = 0.11
STARTER_DRPM = -300.0   # rpm/s  a stalled engine falling faster is running
#                         down, not being cranked
STARTER_RATIO = 2.0     # Hz per crank rpm: pinion mesh (ring/pinion ~12, 10
#                         teeth): 250 rpm cranking -> 500 Hz

# ---- tyres / road ------------------------------------------------------------
UTIL_SQUEAL = 0.90      # lateral utilisation where the squeal starts
UTIL_FULL = 1.00        # ... and is fully on
UTIL_SCRUB = 0.80       # the scrub starts here and is full at UTIL_SCRUB_FULL
UTIL_SCRUB_FULL = 0.95
KAPPA_SQUEAL = 0.10     # slip ratio where it starts (drive peak ~0.12-0.15)
KAPPA_FULL = 0.25
KAPPA_LOCK = 0.10       # braking slip past which it is a LOCK-UP squeal
V_SQUEAL = 2.0          # m/s  below this tyres do not squeal
F_SQUEAL = 1000.0       # Hz  squeal band centre (rises with the level)
Q_SQUEAL = 9.0
A_SQUEAL = 0.16         # full squeal ~0.11 RMS: loud, under the WOT engine (0.14)
A_SQ_TONE = 0.55        # the chirp within the squeal
F_SCRUB = 380.0         # Hz  a low roar under the squeal band
Q_SCRUB = 2.0
A_SCRUB = 0.042
F_LOCK_X = 0.62         # a lock-up squeal sits this much lower (est: a sliding
#                         locked tyre is a lower, harsher note than a cornering one)
ABS_HZ = 15.0           # ABS modulator cycle (est, published 4-20 Hz)
PAN_SQ = 0.65           # the squeal's pan never goes past this
A_ROLL = 0.028          # tarmac rolling hum at V_ROLL_REF (~0.035 RMS)
V_ROLL_REF = 40.0       # m/s
A_WET = 0.040           # spray hiss at 25 m/s on a fully wet road (~0.04 RMS)
F_WET = 3200.0          # Hz  a broad band, not an open high-pass: spray, not fizz
KERB_PITCH = 0.35       # m  rumble-strip stripe spacing -> a thump at V / 0.35
A_KERB = 0.25           # ring amplitude of one stripe on the 88 Hz chassis mode
KERB_MID = 0.7          # ... on the 310 Hz mode (what a laptop speaker plays)
KERB_RATTLE = 0.25      # ... and a rattle in the crunch band
A_GRASS = 0.12          # the swish
A_THUMP = 0.22          # grass / gravel bumps: mean ring amplitude
A_RUMBLE = 0.035        # the verge's dense small bumps
A_ABS = 0.10            # the ABS modulator's 15 Hz chatter
A_GRAVEL = 0.075        # the crunch: mean stone ring amplitude
EXP_CAP = 2.5           # every exponential amplitude draw (grass / gravel bumps,
#                         stones, crackles) is capped at this x its mean (8 %
#                         of draws; the mean bump 0.7 dB softer).  Uncapped, a
#                         lone big bump drove the soft clip to 0.904-0.909 in
#                         the Corsa's off-track real-Sim runs (verifier) - the
#                         compressor's crest limit (BUS_CREST) rightly lets a
#                         single spike through; capped 0.80-0.83.  504 s of
#                         verge at the mapper's levels (8-35 m/s, one side /
#                         all four / two on grass + two on gravel) peak 0.889
#                         (uncapped 0.920, capped at 3: 0.902)
PAN_SURF = 0.70         # a one-sided kerb / verge excursion pans this far
A_WIND = 0.070          # (V/45)^2 = 1 -> ~0.06 RMS
V_WIND_REF = 45.0       # m/s  the wind level is (V / V_WIND_REF)^2
A_SERVO = 0.020         # the wing actuator: audible, the smallest layer
SERVO_RATE_FULL = 2.2   # 1/s: the flank panel's mean deploy rate (t_ext 0.45 s)
PAN_SERVO = 0.6

# ---- cues ----------------------------------------------------------------------
A_CUE = 0.11            # a chime's peak; the engine at WOT peaks ~0.5
CUE_GAP = 0.16          # s  after a cue's last note before the next cue starts

# ---- bus -----------------------------------------------------------------------
BUS_THR = 0.62          # compressor threshold (chunk peak)
BUS_RATIO = 3.0
BUS_REL = 0.35          # s  release
BUS_CREST = 5.0         # the detector's peak is min(chunk peak, this x chunk RMS)
CEIL = 0.92             # the soft clip's ceiling: headroom under full scale
OUT_DC_HZ = 8.0         # the bus DC blocker after the soft clip

# ---- level dynamics (per frame, first order) -----------------------------------
TAU_SQUEAL_UP = 0.03    # s
TAU_SQUEAL_DOWN = 0.15  # s
TAU_LEVEL = 0.05        # s  wind / grass / road
TAU_PAN = 0.10          # s
TAU_SERVO_UP = 0.02     # s
TAU_SERVO_DOWN = 0.05   # s  so the whine is gone ~0.15 s after the panel stops

NOISE_TAB = 1 << 17     # white noise table (3.0 s); each read is a fresh
#                         seeded offset, which for WHITE noise is as good as new
SURF_SQUEAL = {'tarmac': 1.0, 'kerb': 0.8, 'wet': 0.35, 'grass': 0.0, 'gravel': 0.0}
SURF_HARD = {'tarmac': 1.0, 'kerb': 1.0, 'wet': 1.0, 'grass': 0.0, 'gravel': 0.0}


# ====================================================================== #
#  ENGINE PROFILES                                                       #
# ====================================================================== #
@dataclass(frozen=True)
class EngineProfile:
    """One engine's sound.  Geometry is `est` (no exhaust drawings exist for
    any of the three); idle and cut are the cars.py numbers, idle_gear what
    drive.Sim actually idles at in gear (measured on the dragstrip: the auto
    box's creep load of ~0.24 pulls n_idle down)."""

    key: str
    slots: int                  # firings per four-stroke cycle
    bank: tuple                 # bank of each firing slot (0 / 1); I4 all 0
    cyl_gain: tuple             # fixed pulse imbalance per slot
    idle: float                 # rpm, cars.py n_idle (free, in neutral)
    idle_gear: float            # rpm, measured in gear at rest (Sim, load 0.24)
    cut: float                  # rpm, cars.py n_cut
    pulse_k: float              # pulse DECAY corner / firing rate: 0.64 would
    #                             be a blowdown lasting a quarter period (est)
    rise: float                 # pulse RISE corner x (RISE_LO .. RISE_HI with load):
    #                             a sharp front is what makes a note buzzy
    pipe_ms: float              # comb delay = 2 L / c (c ~ 500 m/s hot)
    pipe_fb: float              # comb feedback: < 0 = open-end reflection
    res: tuple                  # ((Hz, Q, gain), ...) exhaust modes
    direct: float               # the comb's own output in the mix
    lp_lo: float                # Hz  brightness low-pass: overrun ...
    lp_hi: float                # ... to WOT at the cut
    intake_f: float             # Hz  intake roar band (rises 0.8-1.3x with rpm)
    intake: float               # its level
    flow: float                 # exhaust-flow noise level (x A_FLOW)
    flow_lp: float              # Hz  ... its colour: a one-pole on the noise (a big
    #                             pipe's turbulence is darker)
    mech: float                 # the mechanical bed (valvetrain, chain, injectors)
    tick: float                 # valvetrain tick level
    jitter: float               # cycle-to-cycle pulse amplitude SD at idle
    rasp: float                 # waveshaper amount
    pops: tuple                 # (lo, hi) pops per full-strength lift (est: a
    #                             stock road exhaust pops now and then, a free-
    #                             flowing one crackles)
    pop_lim: float              # pops per second while the limiter stutters
    crackle: float              # small crackles per second in a full pop window
    pop_rpm: float              # a lift above this arms the pops
    level: float                # overall trim so the three sit together
    bank_mix: tuple = (0.5, 0.5)   # bank weights in MONO.  Equal weights sum a
    #                                cross-plane V8 back to an even 90-deg train
    #                                and kill the burble; a listener nearer one
    #                                tailpipe hears that bank louder
    bank_lr: tuple = ((1.0, 1.0), (1.0, 1.0))   # STEREO: (bank 0, bank 1) weights
    #                                in the left channel, then in the right.  A
    #                                channel's burble is the bank's own x
    #                                ((w0 - w1) / (w0 + w1))^2, and so is the
    #                                fold's with the column sums; an L/R balance
    #                                that holds at every rpm needs mirrored rows,
    #                                which cancel in L+R.  So a V8 trades a little
    #                                balance for a burble in L, R and L+R
    idle_hi: float = 1800.0     # rpm  the idle's character (lumpy pulses, timing and
    #                                rpm jitter) fades out by here, over a band
    #                                scaled with it (1000 / 900 rpm at 1800).  Task
    #                                41: a 2500-rpm bus diesel CRUISES at 1650, so
    #                                an absolute 1800 would make it lope at speed;
    #                                every petrol keeps 1800 exactly (scale 1.0)
    rpms: tuple = (1500.0, 3000.0, 4500.0, 6000.0)   # the self-check's WOT sweep:
    #                                a diesel is never exercised far above its cut


PROFILES = {
    # Opel Corsa C 1.2 16V (Z12XE), 1199 cc, stock single-box exhaust: small
    # bore, short manifold, a thin buzzy note that never gets deep.  The
    # brightest of the three at every rpm (WOT centroid 545 / 664 / 760 /
    # 827 Hz at 1500 / 3000 / 4500 / 6000 rpm).
    'corsa': EngineProfile(
        key='corsa', slots=4, bank=(0, 0, 0, 0),
        cyl_gain=(1.00, 0.95, 1.03, 0.97), idle=850.0, idle_gear=586.0, cut=6200.0,
        pulse_k=1.25, rise=1.40, pipe_ms=3.2, pipe_fb=-0.38,
        res=((240.0, 2.4, 0.70), (660.0, 3.5, 0.85), (1750.0, 2.6, 0.60),
             (74.0, 5.5, 0.55)),
        direct=0.30, lp_lo=900.0, lp_hi=6000.0, intake_f=420.0, intake=0.55,
        flow=1.00, flow_lp=7000.0, mech=1.00,
        tick=0.40, jitter=0.10, rasp=0.05, pops=(0, 2), pop_lim=0.8, crackle=3.0,
        pop_rpm=4200.0, level=1.00),
    # Ford Escort RS1800's Cosworth BDA (task 46): 1975 cc, 16 valves, twin
    # cams on a belt, a Group 4 tune to 9000 rpm -- the crisp, hard wail of a
    # race four. A short 4-2-1 tubular manifold into a straight-through box
    # (the shortest pipe, the highest modes, the brightest low-pass: the one
    # four brighter than the Corsa's from 3000 rpm up), the 1-4 / 2-3 pairing
    # a milder crank order than a road 4-2-1's, and the loudest INDUCTION of
    # all -- the open trumpets of its twin chokes (intake band 520 Hz). Race
    # cams: a lumpy, uneven idle at 1200 rpm (jitter 0.16), still loping over
    # the Corsa's 1800 (idle_hi 2600: its band scaled with it). A free-flowing
    # exhaust on a rich overrun crackles and pops the most (pops 4-8 per lift,
    # crackle 11/s). Idle and cut are cars.ESCORT_RS1800's (1200 / 9000, est);
    # idle_gear is the Vehicle's in 1st at rest (1078 rpm, creep load 0.11,
    # measured like the others' on the full model). The front (rise 1.30)
    # and the rasp (0.12) are held under a road four's: sharper, the soft
    # clip's DC blocker overshot the ceiling near the 9000 rpm cut (0.935
    # against CEIL 0.92 in the demo lap). All the rest est.
    'rally': EngineProfile(
        key='rally', slots=4, bank=(0, 0, 0, 0),
        cyl_gain=(1.00, 0.74, 1.02, 0.76), idle=1200.0, idle_gear=1078.0, cut=9000.0,
        pulse_k=1.30, rise=1.30, pipe_ms=2.8, pipe_fb=-0.42,
        res=((280.0, 2.4, 0.55), (760.0, 3.2, 0.95), (2000.0, 2.6, 0.70),
             (86.0, 5.0, 0.40)),
        direct=0.36, lp_lo=1000.0, lp_hi=7600.0, intake_f=520.0, intake=1.00,
        flow=0.90, flow_lp=7800.0, mech=0.95,
        tick=0.30, jitter=0.16, rasp=0.12, pops=(4, 8), pop_lim=3.5, crackle=11.0,
        pop_rpm=5200.0, level=0.80, idle_hi=2600.0),
    # BMW 540i (E39, M62 4.4 V8), cross-plane crank, firing every 90 deg; the
    # banks' own pulse trains are uneven, so each bank burbles at the cycle
    # rate.  Twin exhaust: bank 0 left and louder, bank 1 right (bank_lr:
    # L/R +0.7 / +1.8 / +3.5 dB at idle / 3000 WOT / 5000 overrun, the
    # burble in each channel at 3000 WOT 1.98 / 0.17 and L+R 0.17; the
    # pans before had R 0.017, L+R 0.36, L/R +0.7 / +1.8 / +4.3 dB).  The
    # deepest of the three at every rpm: long, blunt pulses (pulse_k 0.6,
    # rise 0.6), low modes, the darkest low-pass and darker flow noise - a V8
    # fires twice as often as a four at the same rpm, so without that its
    # centroid sat ABOVE the Corsa's from 3000 rpm (review).
    '540i': EngineProfile(
        key='540i', slots=8, bank=(0, 1, 1, 0, 1, 0, 0, 1),
        cyl_gain=(1.00, 0.96, 1.03, 0.97, 0.95, 1.02, 0.98, 1.01),
        idle=650.0, idle_gear=551.0, cut=6400.0,
        pulse_k=0.60, rise=0.60, pipe_ms=6.8, pipe_fb=-0.25,
        res=((82.0, 2.2, 0.65), (270.0, 1.8, 0.45), (600.0, 2.2, 0.35),
             (46.0, 4.5, 0.80)),
        direct=0.40, lp_lo=380.0, lp_hi=1550.0, intake_f=230.0, intake=0.50,
        flow=0.80, flow_lp=1300.0, mech=0.80,
        tick=0.20, jitter=0.07, rasp=0.10, pops=(2, 4), pop_lim=1.5, crackle=3.8,
        pop_rpm=3300.0, level=0.95, bank_mix=(0.66, 0.34),
        bank_lr=((1.52, 0.0), (0.66, 1.21))),
    # Renault Express 1.4 (E7J "Energy", 1390 cc SOHC 8V four, task 41): the
    # R5 family's plain engine in a van -- a longer, cheaper single-box
    # exhaust than the Corsa's, so a little lower and coarser (more
    # mechanical bed, a louder tappet tick), no sporty rasp, a stock
    # road exhaust's rare pop.  Idle and cut are cars.EXPRESS_14's
    # (800 / 6000, est); idle_gear is drive.Sim's in 1st at rest, measured
    # like the others (573 rpm, load 0.20).  All the rest est.
    'express': EngineProfile(
        key='express', slots=4, bank=(0, 0, 0, 0),
        cyl_gain=(1.00, 0.93, 1.04, 0.96), idle=800.0, idle_gear=573.0, cut=6000.0,
        pulse_k=1.10, rise=1.20, pipe_ms=3.9, pipe_fb=-0.36,
        res=((205.0, 2.4, 0.65), (560.0, 3.2, 0.80), (1450.0, 2.6, 0.55),
             (68.0, 5.5, 0.60)),
        direct=0.32, lp_lo=820.0, lp_hi=4800.0, intake_f=380.0, intake=0.50,
        flow=0.95, flow_lp=6000.0, mech=1.15,
        tick=0.55, jitter=0.11, rasp=0.06, pops=(0, 1), pop_lim=0.6, crackle=2.0,
        pop_rpm=4000.0, level=1.00),
}
PROFILE_DEFAULT = 'corsa'


def profile_key(hud) -> str:
    """The engine to voice: `hud.car_key` ('corsa' | 'rally' | '540i' |
    'express'), else a guess from `hud.car_name` (cars.CAR_TITLES), else the
    Corsa (task 46: a retired car's key, 'mx5' or 'bus', is the Corsa's, as
    `cars.get` makes it)."""
    k = str(getattr(hud, 'car_key', '') or '').lower()
    if k in PROFILES:
        return k
    name = str(getattr(hud, 'car_name', '') or '').lower()
    #  the display titles ('Halcón RS18 rally', 'Nordwerk N540', 'Rivière
    #  Courier 1.4'); the names from before the rename still read
    if 'rally' in name or 'halc' in name or 'rs18' in name or 'escort' in name:
        return 'rally'
    if '540' in name or 'v8' in name:
        return '540i'
    if 'courier' in name or 'rivi' in name or 'express' in name or 'renault' in name:
        return 'express'
    return PROFILE_DEFAULT


# ====================================================================== #
#  DSP PRIMITIVES                                                        #
# ====================================================================== #
_LFILTER = None         # scipy.signal.lfilter once looked up; False = absent
_WARM = None            # the warm-up thread that imports scipy.signal (`warm_up`)
_WARM_RATES: set = set()    # mixer rates whose chimes are rendered or on their way
WARM_SWITCH = 0.0005    # s  the interpreter's thread switch interval while the
#                         warm-up imports: at the default 5 ms a 0.5 ms frame
#                         task waited up to 15.6 ms for the GIL during the
#                         import, at 0.5 ms 2.6 ms (measured, this Mac)


def _scipy_lfilter():
    """scipy.signal.lfilter, imported on first use.  BLOCKING: 0.27-0.49 s
    on this Mac even with scipy.interpolate / linalg already loaded by the
    powertrain.  The game never pays it inside a frame: `CarSound` builds
    its Synth on the numpy path and `warm_up` imports this in a thread."""
    global _LFILTER
    if _LFILTER is None:
        try:
            from scipy.signal import lfilter
            _LFILTER = _direct_lfilter(lfilter)
        except Exception:
            _LFILTER = False
    return _LFILTER or None


def _direct_lfilter(lfilter):
    """lfilter's own C core when it gives lfilter's answer, else lfilter.
    For an IIR (len(a) > 1, every filter here) scipy 1.17's lfilter is
    atleast_1d / asarray / array-API bookkeeping around
    `_sigtools._linear_filter(b, a, x, axis, zi)`, and 24 calls a chunk paid
    0.05-0.07 ms of that bookkeeping (V8 stereo worst case, timed; cProfile
    said 0.2 ms).
    Checked once against lfilter on a stereo block; any difference, any
    error or a scipy without it -> the public function."""
    try:
        from scipy.signal import _sigtools
        core = _sigtools._linear_filter
        x = np.random.default_rng(1).standard_normal((2, 257))
        for ba in (((0.2,), (1.0, -0.8)), ((0.1, 0.0, -0.1), (1.0, -1.6, 0.8))):
            b, a = (np.asarray(v, dtype=float) for v in ba)
            zi = np.full((2, max(len(a), len(b)) - 1), 0.3)
            y0, z0 = lfilter(b, a, x, axis=-1, zi=zi)
            y1, z1 = core(b, a, x, -1, zi)
            if not (np.array_equal(y0, y1) and np.array_equal(z0, z1)):
                return lfilter

        def fast(b, a, x, axis=-1, zi=None):
            return core(b, a, x, axis, zi)
        return fast
    except Exception:
        return lfilter


def _signal_imported() -> bool:
    """scipy.signal is in sys.modules AND done initialising: a module sits in
    sys.modules from the START of its import, and importing from it then
    waits on the import lock (the first frame took 435 ms that way)."""
    mod = sys.modules.get('scipy.signal')
    return (mod is not None and hasattr(mod, 'lfilter')
            and not getattr(getattr(mod, '__spec__', None), '_initializing', False))


def _lfilter_ready():
    """lfilter if it is already imported, else None - never blocks."""
    if _LFILTER is None and _signal_imported():
        return _scipy_lfilter()                 # imported by someone else: free
    return _LFILTER or None


def warm_up(rate: int = RATE) -> None:
    """The sound's start-up work, OFF the caller's thread: import
    scipy.signal (0.27-0.49 s) and render the chimes at `rate` (~30 ms), once
    per process (the chimes once per rate).  Never blocks, needs no mixer and
    no CarSound: call `drive.audio.warm_up()` once at session start whether
    Sound is on or off, so turning it on later builds a CarSound in ~1 ms
    with lfilter already there.  Measured in a fresh process after
    drive.powertrain, sound off: the call returns in 0.10-0.14 ms, the thread
    is done 0.34-0.43 s later, a 60 fps loop's 3.1 ms of frame work stretched
    to at most 4.1 ms meanwhile, no mixer opened; a second call costs 1 us.

    Without it CarSound calls it itself: its Synth runs on the numpy filters
    until the import lands (1.5-2 ms a chunk instead of 1.1-1.3 in the worst
    case, for the ~20 frames it takes), then hands every filter's state over
    to lfilter without a step (`_IIR.adopt`).  Importing inside the frame
    instead stalled it 0.25-0.44 s (review; 316-435 ms measured).  While the
    import runs the interpreter's switch interval is WARM_SWITCH (restored
    after) so a frame is not held off the GIL for long; a CarSound turned on
    mid-session still saw a few 20-38 ms frames then.  A mixer that opens at
    another rate gets its chimes from CarSound's own call, in the background
    too (scipy imported by someone else used to skip the chimes)."""
    global _WARM
    if _LFILTER is None and _signal_imported():
        _scipy_lfilter()                        # someone else paid for it: free
    r = int(rate)
    chimes = r not in _WARM_RATES
    importing = _LFILTER is None and _WARM is None
    if not (importing or chimes):
        return
    _WARM_RATES.add(r)
    old = sys.getswitchinterval()
    mine = min(old, WARM_SWITCH)

    def run():
        if importing:
            try:
                _scipy_lfilter()
            finally:
                if sys.getswitchinterval() == mine:
                    sys.setswitchinterval(old)
        if chimes:
            for kind in CUES:                   # numpy releases the GIL in these
                _render_cue(kind, r)

    th = threading.Thread(target=run, name='drive.audio warm-up', daemon=True)
    if importing:
        sys.setswitchinterval(mine)
        _WARM = th
    th.start()


def _exp_capped(rng, m: int) -> np.ndarray:
    """m exponential draws of mean 1, none over EXP_CAP (a thump's size)."""
    return np.minimum(rng.exponential(1.0, m), EXP_CAP)


#: the time trial's chimes (task 27), procedural like everything here: a
#: rising arpeggio of decaying sines (a touch of the octave), NOTE_S apart --
#: a NEW PB is four notes to the octave, a new best MEDAL three
CHIMES = {"pb": (659.25, 830.61, 987.77, 1318.51),     # E5 G#5 B5 E6
          "medal": (523.25, 659.25, 783.99)}          # C5 E5 G5
NOTE_S = 0.09           # s between the notes
CHIME_TAIL_S = 0.50     # s the last note rings
CHIME_DECAY_S = 0.16    # s  each note's decay
A_CHIME = 0.40          # peak level, under the engine's A_ENGINE
CHIME_CHANNEL = 1       # the engine streams on channel 0


def chime_wave(kind: str, rate: int = RATE) -> np.ndarray:
    """The chime as float samples in [-A_CHIME, A_CHIME]. Pure."""
    notes = CHIMES.get(kind) or CHIMES["medal"]
    n = int(round((NOTE_S * (len(notes) - 1) + CHIME_TAIL_S) * rate))
    t = np.arange(n) / float(rate)
    y = np.zeros(n)
    for k, f in enumerate(notes):
        t0 = k * NOTE_S
        m = t >= t0
        tt = t[m] - t0
        env = (1.0 - np.exp(-tt / 0.004)) * np.exp(-tt / CHIME_DECAY_S)
        y[m] += (np.sin(2.0 * np.pi * f * tt) + 0.3 * np.sin(4.0 * np.pi * f * tt)) * env
    peak = float(np.max(np.abs(y))) or 1.0
    return y * (A_CHIME / peak)


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _num(v, default: float = 0.0, lo: float = -1e9, hi: float = 1e9) -> float:
    """Every float the mapper and the synth read goes through here: a
    non-number or a non-finite value becomes `default`, the rest is clamped
    to [lo, hi].  One NaN in a smoother or a filter state used to silence
    the sound for the rest of the session (review: util NaN, kappa NaN, V
    inf), or disable it (V NaN reached rng.poisson, rpm NaN math.floor)."""
    try:
        x = float(v)
    except (TypeError, ValueError, OverflowError):
        return default
    if x != x or x in (math.inf, -math.inf):
        return default
    return lo if x < lo else (hi if x > hi else x)


def _int(v, default=None):
    """An integer HUD field (gear, wing_side): None / NaN / 'N' -> default."""
    if isinstance(v, bool):
        return int(v)
    try:
        x = float(v)
    except (TypeError, ValueError, OverflowError):
        return default
    if x != x or x in (math.inf, -math.inf):
        return default
    return int(max(-99.0, min(99.0, x)))


def _bp(f: float, q: float, rate: int):
    """RBJ band-pass, constant 0 dB peak gain."""
    w = 2.0 * math.pi * min(max(f, 5.0), 0.45 * rate) / rate
    al = math.sin(w) / (2.0 * q)
    a0 = 1.0 + al
    return (al / a0, 0.0, -al / a0), (1.0, -2.0 * math.cos(w) / a0, (1.0 - al) / a0)


def _lp2(f: float, q: float, rate: int):
    """RBJ low-pass (q > 0.5: complex poles, which the numpy path needs)."""
    w = 2.0 * math.pi * min(max(f, 5.0), 0.45 * rate) / rate
    al = math.sin(w) / (2.0 * q)
    a0 = 1.0 + al
    c = math.cos(w)
    return (((1.0 - c) / 2.0 / a0, (1.0 - c) / a0, (1.0 - c) / 2.0 / a0),
            (1.0, -2.0 * c / a0, (1.0 - al) / a0))


def _lp1(f: float, rate: int):
    """One-pole low-pass, unity DC gain."""
    p = math.exp(-2.0 * math.pi * min(max(f, 1.0), 0.45 * rate) / rate)
    return (1.0 - p,), (1.0, -p)


def _hp1(f: float, rate: int):
    """One-pole high-pass (DC blocker), unity gain at Nyquist."""
    p = math.exp(-2.0 * math.pi * f / rate)
    return ((1.0 + p) / 2.0, -(1.0 + p) / 2.0), (1.0, -p)


def _bp_ring_gain(f: float, q: float, rate: int) -> float:
    """Gain that makes a unit IMPULSE through `_bp(f, q)` ring at unit
    amplitude: the band-pass's impulse response starts at alpha = sin(w)/2q.
    Every impulse-driven path here is scaled by it, so an amplitude constant
    IS the ring's amplitude whatever the mode's frequency and Q."""
    w = 2.0 * math.pi * min(max(f, 5.0), 0.45 * rate) / rate
    return 2.0 * q / math.sin(w)


def _bp_noise_gain(f: float, q: float, rate: int) -> float:
    """Gain that brings white noise through `_bp(f, q)` back to unit RMS: the
    band-pass's noise bandwidth is pi f / (q rate) of the spectrum."""
    return math.sqrt(q * rate / (math.pi * max(f, 5.0)))


def _powers(p, n: int):
    """(L, p^-k, p^k, p^(k+1), p^L) for `_onepole`: L keeps |p|^-L < 1e120."""
    ap = abs(p)
    L = 256 if ap >= 0.34 else max(4, int(120.0 / -math.log10(max(ap, 1e-30))))
    L = min(L, n)
    k = np.arange(L)
    pk = p ** k
    return L, p ** -k, pk, pk * p, p ** L


def _onepole(x: np.ndarray, p, w0, pw=None):
    """w[n] = x[n] + p w[n-1] along the last axis, from state w0, without a
    Python loop per sample: inside a segment of L samples
    w[k] = p^k (cumsum(x p^-k))[k] + p^(k+1) W, and the segments' start
    states W chain by one scalar recurrence each.  `pw` = `_powers(p, n)`,
    cached by the caller while p is fixed."""
    n = x.shape[-1]
    L, pinv, pk, pk1, pL = pw if pw is not None and pw[0] <= n else _powers(p, n)
    S = -(-n // L)
    dt = np.complex128 if isinstance(p, complex) else np.float64
    if S * L != n:
        xp = np.zeros(x.shape[:-1] + (S * L,), dtype=dt)
        xp[..., :n] = x
    else:
        xp = x.astype(dt, copy=False)
    xs = xp.reshape(x.shape[:-1] + (S, L))
    loc = np.cumsum(xs * pinv, axis=-1) * pk
    W = np.empty(x.shape[:-1] + (S,), dtype=dt)
    wc = np.asarray(w0, dtype=dt)
    for s in range(S):
        W[..., s] = wc
        wc = loc[..., s, -1] + pL * wc
    w = (loc + W[..., None] * pk1).reshape(x.shape[:-1] + (S * L,))[..., :n]
    return w, w[..., -1].copy()


class _IIR:
    """One first- or second-order IIR (b, a) with its state carried across
    chunks along the last axis of a (rows, n) block.  scipy's lfilter when
    importable; else the numpy path: H = d + r/(1 - p z^-1) (+ conj), each
    pole a `_onepole` run on r x, so the carried state is that section's own
    OUTPUT v (y = d x + 2 Re v): when a coefficient moves between chunks the
    output continues from where it was (carrying the unscaled pole state
    instead made y jump by r_new / r_old - x8 when the pulse corner moves
    20 -> 170 Hz).  It supports what this module builds: one real pole, or
    a complex pair (every biquad here has Q > 0.5)."""

    __slots__ = ("b", "a", "z", "lf", "_pf", "_pw")

    def __init__(self, ba, lfilter=None):
        self.lf = lfilter
        self.z = None
        self._pf = self._pw = None
        self.set(ba)

    def set(self, ba) -> None:
        b, a = ba
        b = np.asarray(b, dtype=float)
        a = np.asarray(a, dtype=float)
        if self._pf is not None and np.array_equal(a, self.a) and np.array_equal(b, self.b):
            return                              # unchanged: keep the cached split
        self.b, self.a = b, a
        self._pf = None

    def reset(self) -> None:
        self.z = None

    def copy_from(self, other: "_IIR") -> None:
        """Take `other`'s coefficients and state: this filter then continues
        exactly as `other` would (the squeal's band-centre crossfade)."""
        self.set((other.b, other.a))
        self.z = None if other.z is None else np.array(other.z, copy=True)

    def adopt(self, lfilter) -> None:
        """Switch a numpy-path filter to lfilter mid-stream without a step.
        The numpy path carries each pole section's output v (y = d x + v, or
        d x + 2 Re v for a pair); lfilter's DF2T state is the zero-input
        response, so z0 = p v (one real pole), or z0 = 2 Re(p v) and
        z1 = 2 Re(p^2 v) + a1 z0 (a pair) - taken with the coefficients the
        next chunk would use, which is what the numpy path would have done."""
        if self.lf is not None or lfilter is None:
            return
        self.lf = lfilter
        if self.z is None:
            return
        d, r, p = self._split()
        v = np.asarray(self.z)
        if isinstance(p, complex):
            z0 = 2.0 * (p * v).real
            z1 = 2.0 * (p * p * v).real + float(self.a[1]) * z0
            self.z = np.stack([z0, z1], axis=-1)
        else:
            self.z = np.asarray(p * v, dtype=float)[..., None]
        order = max(len(self.a), len(self.b)) - 1
        if self.z.shape[-1] != order:           # never for what this module builds
            self.z = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.lf is not None:
            order = max(len(self.a), len(self.b)) - 1
            shp = x.shape[:-1] + (order,)
            if self.z is None or self.z.shape != shp:
                self.z = np.zeros(shp)
            y, self.z = self.lf(self.b, self.a, x, axis=-1, zi=self.z)
            return y
        return self._np(x)

    # -- the numpy path --------------------------------------------------
    def _split(self):
        b, a = self.b, self.a
        if len(a) == 2:                           # one real pole
            p = -a[1]
            b0 = b[0]
            b1 = b[1] if len(b) > 1 else 0.0
            d = -b1 / p
            return d, b0 - d, p
        a1, a2 = a[1], a[2]
        disc = a1 * a1 - 4.0 * a2
        if disc >= 0.0:
            raise ValueError("numpy _IIR needs complex poles (Q > 0.5)")
        p = complex(-a1 / 2.0, math.sqrt(-disc) / 2.0)
        b0, b1 = b[0], (b[1] if len(b) > 1 else 0.0)
        b2 = b[2] if len(b) > 2 else 0.0
        d = b2 / a2
        n0, n1 = b0 - d, b1 - d * a1
        r = (n0 + n1 / p) / (1.0 - p.conjugate() / p)
        return d, r, p

    def _np(self, x: np.ndarray) -> np.ndarray:
        if self._pf is None:
            self._pf = self._split()
            self._pw = _powers(self._pf[2], x.shape[-1])
        d, r, p = self._pf
        shp = x.shape[:-1]
        if self.z is None or np.shape(self.z) != shp or \
                (isinstance(p, complex) != np.iscomplexobj(self.z)):
            self.z = np.zeros(shp, dtype=np.complex128 if isinstance(p, complex) else float)
        v, self.z = _onepole(r * x, p, self.z, self._pw)
        if isinstance(p, complex):
            return d * x + 2.0 * v.real
        return d * x + v


class _Comb:
    """Feedback comb y[n] = x[n] + g y[n-D] (the pipe's round trip), its last
    D outputs carried, vectorised in D-sample slabs (a 3-7 ms pipe is 7-15
    slabs per chunk)."""

    __slots__ = ("D", "g", "tail")

    def __init__(self, D: int, g: float):
        self.D, self.g, self.tail = max(int(D), 1), float(g), None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        D, g = self.D, self.g
        rows, n = x.shape
        if self.tail is None or self.tail.shape[0] != rows:
            self.tail = np.zeros((rows, D))
        y = np.empty((rows, D + n))
        y[:, :D] = self.tail
        for s in range(0, n, D):
            e = min(s + D, n)
            y[:, D + s:D + e] = x[:, s:e] + g * y[:, s:e]
        self.tail = y[:, n:].copy()
        return y[:, D:]


def _crossings(ph0: float, inc: np.ndarray, slots: int):
    """Events of a phase accumulator.  `ph` (cycles) advances by `inc` per
    sample from ph0; an event is each crossing of a multiple of 1/slots.
    Returns (pos, k, ph): pos in (0, n] the event's sample position with a
    one-sample latency (so the crossing between the previous chunk's last
    sample and this one's first still lands in this chunk), k the integer
    slot crossed into, ph the per-sample phase."""
    ph = ph0 + np.cumsum(inc)
    m = np.floor(ph * slots)
    m_prev = np.empty_like(m)
    m_prev[0] = math.floor(ph0 * slots)
    m_prev[1:] = m[:-1]
    j = np.flatnonzero(m != m_prev)
    if j.size == 0:
        return j.astype(float), j, ph
    k = m[j]
    p_before = np.where(j > 0, ph[j - 1], ph0)
    frac = (k / slots - p_before) / np.maximum(ph[j] - p_before, 1e-12)
    return j + np.clip(frac, 0.0, 1.0), k.astype(np.int64), ph


def _place(n: int, pos, amp, rows, nrows: int, carry: np.ndarray):
    """Impulses of `amp` at fractional positions `pos` in [0, n], split
    linearly between the two neighbouring samples (sub-sample timing), on
    row `rows`.  Sample n spills into the next chunk through `carry`."""
    e = np.zeros(nrows * (n + 1))
    if len(pos):
        i0 = pos.astype(np.intp)
        w1 = pos - i0
        base = np.asarray(rows, dtype=np.intp) * (n + 1) + i0
        e += np.bincount(base, amp * (1.0 - w1), minlength=nrows * (n + 1))
        e += np.bincount(base + 1, amp * w1, minlength=nrows * (n + 1))
    e = e.reshape(nrows, n + 1)
    e[:, 0] += carry
    return e[:, :n], e[:, n].copy()


def _pan_gains(pan: float):
    """Constant-power pan, -1 left .. +1 right, centre = (1, 1) so a
    centred layer is the same in mono and in each stereo channel."""
    a = (max(-1.0, min(1.0, pan)) + 1.0) * math.pi / 4.0
    return math.sqrt(2.0) * math.cos(a), math.sqrt(2.0) * math.sin(a)


# ====================================================================== #
#  ONE-SHOTS: clunk and chimes                                           #
# ====================================================================== #
_TIMBRES = {
    # (partial ratios, amplitudes, decay s): additive, no filters
    'marimba': ((1.0, 3.93, 9.2), (1.0, 0.22, 0.05), (0.30, 0.07, 0.025)),
    'bell': ((1.0, 2.0, 2.76, 5.40), (1.0, 0.42, 0.30, 0.10), (0.80, 0.50, 0.32, 0.12)),
    'soft': ((1.0, 2.0), (1.0, 0.18), (0.20, 0.06)),
}
_NOTE = {'A4': 440.0, 'D5': 587.33, 'G5': 783.99, 'A5': 880.0, 'C6': 1046.50,
         'D6': 1174.66, 'E6': 1318.51, 'G6': 1567.98, 'A6': 1760.0, 'B6': 1975.53,
         'C7': 2093.00}
CUES = {
    # kind: (timbre, ((onset s, note, amp), ...))
    'sector_purple': ('marimba', ((0.0, 'E6', 1.0), (0.085, 'A6', 1.0))),
    'sector_green': ('marimba', ((0.0, 'C6', 0.9),)),
    'sector_red': ('soft', ((0.0, 'D5', 0.7), (0.10, 'A4', 0.6))),
}
FLASH_CUES = ('purple', 'green', 'red')


_CUE_CACHE: dict = {}


def _render_cue(kind: str, rate: int) -> tuple:
    """(samples, onset of the last note s) of one chime, cached per process:
    a bell is ~2.6 s of additive partials, ~5-10 ms to build, which must not
    land inside a render frame (CarSound builds them all up front)."""
    key = (kind, int(rate))
    if key in _CUE_CACHE:
        return _CUE_CACHE[key]
    timbre, notes = CUES[kind]
    ratios, amps, decays = _TIMBRES[timbre]
    t_last = max(n[0] for n in notes)
    dur = t_last + 3.0 * max(decays)
    t = np.arange(int(dur * rate)) / rate
    y = np.zeros_like(t)
    att = int(0.004 * rate)
    for onset, note, amp in notes:
        i0 = int(onset * rate)
        tt = t[:len(t) - i0]
        f0 = _NOTE[note]
        s = np.zeros_like(tt)
        for rt, a, d in zip(ratios, amps, decays):
            if f0 * rt < 0.45 * rate:
                s += a * np.exp(-tt / d) * np.sin(2.0 * math.pi * f0 * rt * tt)
        s[:att] *= np.linspace(0.0, 1.0, att, endpoint=False)   # no click
        y[i0:] += amp * s
    fade = int(0.02 * rate)
    y[-fade:] *= np.linspace(1.0, 0.0, fade)
    y *= A_CUE / max(float(np.abs(y).max()), 1e-9)
    _CUE_CACHE[key] = (y, t_last)
    return y, t_last


# ====================================================================== #
#  THE SYNTH                                                             #
# ====================================================================== #
class Synth:
    """The signal, without any device.  `chunk(**targets)` returns CHUNK
    float samples in [-1, 1] - shape (n,) mono, (n, 2) stereo - and advances
    every state to the targets.  `event(kind, arg)` queues a one-shot:
    'clunk' (level), 'shift_cut' (seconds), 'cue' (a CUES kind).

    `backend`: None = lfilter when scipy imports (blocking), else numpy, and
    CARSIM_AUDIO_NUMPY=1 forces numpy; 'scipy' = lfilter whatever the
    environment says (the self-check's reference); 'numpy'; 'lazy' = lfilter
    if scipy.signal is already imported, else numpy now and lfilter from the
    first chunk after `warm_up`'s import lands (CarSound: never blocks)."""

    def __init__(self, rate: int = RATE, chunk: int = CHUNK, seed: int = 7,
                 channels: int = 1, car: str = PROFILE_DEFAULT,
                 backend: str | None = None):
        self.rate = int(rate)
        self.n = int(chunk)
        self.nch = 2 if channels >= 2 else 1
        self.rng = np.random.default_rng(seed)
        forced_np = bool(os.environ.get('CARSIM_AUDIO_NUMPY'))
        self._lazy = False
        if backend == 'numpy' or (forced_np and backend != 'scipy'):
            lf = None
        elif backend == 'lazy':
            lf = _lfilter_ready()
            self._lazy = lf is None and _LFILTER is not False
        else:
            lf = _scipy_lfilter()
        self.backend = 'scipy' if lf is not None else 'numpy'
        self._lf = lf
        self._filters: list = []        # every _IIR, for adopt / reset
        self.faults = 0                 # chunks that came out non-finite
        self.pops_fired = 0             # pops placed so far (the self-check)
        self.pop_log = None             # a list -> each pop's sample index is appended
        self.ramp = np.linspace(0.0, 1.0, self.n, endpoint=False)
        self.tsec = np.arange(self.n) / self.rate
        self.t0 = 0                     # samples rendered so far
        self._ntab = self.rng.standard_normal(NOISE_TAB)
        r = self.rate
        # previous targets (ramped from, inside each chunk)
        self.rpm = 0.0
        self.load = 0.0
        self.eng_on = 0.0               # 0 silent .. 1 running
        self.starter = 0.0
        self.squeal = self.lock = self.abs_on = self.scrub = 0.0
        self.sq_pan = 0.0
        self.grass = self.grass_pan = self.gravel = self.gravel_pan = 0.0
        self.kerb = self.kerb_pan = 0.0
        self.roll = self.wet = self.wind = 0.0
        self.limit = self.pops = 0.0
        self.pop_sched: list = []       # sample times of this window's pops
        self.sq_fc = 0.0                # the squeal band centre last chunk
        self.sq_lk = 0.0                # ... and its lock-up share
        self.servo = self.servo_pan = self.servo_rate = 0.0
        self.whine = 0.0
        self.k_whine = K_WHINE
        self.V = 0.0
        self.master = 1.0
        self.volume = 1.0
        self.jit = 0.0                  # idle rpm wander, slow random walk
        self.gust = 0.0                 # wind gust, an OU process
        self.wander = 0.0               # squeal chirp pitch wander, OU
        self.bus_g = 1.0
        self.rel_k = 1.0 - math.exp(-self.n / (r * BUS_REL))
        self.f_out = None               # the bus DC blocker (built below, per channel)
        # phases, in cycles
        self.ph_eng = 0.0
        self.ph_kerb = 0.0
        self.ph_abs = 0.0
        self.ph_sq = 0.0
        self.ph_whine = 0.0
        self.ph_srv = 0.0               # in [0, 9): the 1/9-rate AM rides on it
        self.ph_st = 0.0
        # impulse-train carries
        rows = self.nch
        self.c_crack = np.zeros(1)
        self.c_thump = np.zeros(rows)
        self.c_crunch = np.zeros(rows)
        # chunks an impulse-driven path keeps running after its last impulse,
        # so its resonators ring out instead of being cut off (a click)
        self.live_thump = self.live_crunch = self.live_crack = 0
        self._intake_on = False
        self._ph_valid = False
        self.ph_eng_samples = None
        # shared (profile-free) filters
        F = self._mk
        self.f_intake = F(_bp(400.0, 1.3, r))
        self.f_crack = F(_bp(3200.0, 1.6, r))
        self.f_sq = F(_bp(F_SQUEAL, Q_SQUEAL, r))
        self.f_sq0 = F(_bp(F_SQUEAL, Q_SQUEAL, r))     # the old centre, crossfaded out
        self.f_mech = F(_bp(F_MECH, Q_MECH, r))
        self.f_scrub = F(_bp(F_SCRUB, Q_SCRUB, r))
        self.f_roll = F(_bp(820.0, 0.8, r))
        self.f_rumble = F(_bp(115.0, 1.0, r))
        self.f_wet = F(_bp(F_WET, 0.6, r))
        self.f_swish = F(_bp(480.0, 0.75, r))
        self.f_crunch = F(_bp(2300.0, 1.3, r))
        self.f_thump1 = F(_bp(88.0, 1.6, r))    # chassis / suspension modes (est)
        self.f_thump2 = F(_bp(310.0, 1.3, r))
        self.g_thump1 = _bp_ring_gain(88.0, 1.6, r)
        self.g_thump2 = KERB_MID * _bp_ring_gain(310.0, 1.3, r)
        self.g_crunch = _bp_ring_gain(2300.0, 1.3, r)
        self.g_crack = _bp_ring_gain(3200.0, 1.6, r)
        self.f_wind = F(_bp(400.0, 0.6, r))
        self.f_out = F(_hp1(OUT_DC_HZ, r))
        # one-shots: pending samples consumed chunk by chunk, and the cue queue
        self.fx = np.zeros(0)
        self.cue_free = 0               # samples from the next chunk's start
        self.cut_until = -1             # sample index: ignition cut until
        self.prof = None
        self.set_profile(car)

    def _mk(self, ba) -> _IIR:
        f = _IIR(ba, self._lf)
        self._filters.append(f)
        return f

    def _adopt_lfilter(self) -> None:
        """The lazy backend's hand-over: lfilter has landed (warm_up)."""
        lf = _lfilter_ready()
        if lf is None:
            if _LFILTER is False:
                self._lazy = False               # no scipy after all: numpy for good
            return
        for f in self._filters:
            f.adopt(lf)
        self._lf = lf
        self.backend = 'scipy'
        self._lazy = False

    def _fault_reset(self) -> None:
        """A chunk came out non-finite: start every state afresh (filters,
        comb, carries, phases, bus gain, the random walks) and fade the engine
        back in, so the NEXT chunk with good targets sounds again.  Before,
        one NaN stuck in a filter state or in bus_g for the whole session."""
        self.faults += 1
        for f in self._filters:
            f.reset()
        self.comb.tail = None
        self.c_eng = np.zeros(self.nb)
        self.flow_lvl = np.zeros(self.nb)
        self.c_crack = np.zeros(1)
        self.c_thump = np.zeros(self.nch)
        self.c_crunch = np.zeros(self.nch)
        for k in ('ph_eng', 'ph_kerb', 'ph_abs', 'ph_sq', 'ph_whine', 'ph_srv', 'ph_st',
                  'jit', 'gust', 'wander', 'sq_fc', 'sq_lk'):
            setattr(self, k, 0.0)
        self.bus_g = 1.0
        self.eng_on = self.starter = 0.0
        self.fx = np.zeros(0)
        self.pop_sched = []

    def _noise(self, rows: int = 1) -> np.ndarray:
        n = self.n
        off = int(self.rng.integers(0, NOISE_TAB - rows * n))
        return self._ntab[off:off + rows * n].reshape(rows, n)

    # -- profile ---------------------------------------------------------------
    def set_profile(self, key: str) -> None:
        """Build the engine's own filters.  A change mid-stream fades the new
        engine in from silence (a car change restarts the session anyway)."""
        P = PROFILES.get(key, PROFILES[PROFILE_DEFAULT])
        if self.prof is not None and P.key == self.prof.key:
            return
        r = self.rate
        self.prof = P
        self.nb = 2 if (max(P.bank) > 0) else 1
        old = getattr(self, '_prof_filters', ())
        self._filters = [f for f in self._filters if all(f is not o for o in old)]
        n0 = len(self._filters)
        self.f_rise = self._mk(_lp1(3000.0, r))
        self.f_pulse = self._mk(_lp1(200.0, r))
        self.f_dc = self._mk(_hp1(DC_HZ, r))
        self.comb = _Comb(round(P.pipe_ms * 1e-3 * r), P.pipe_fb)
        self.f_res = [(self._mk(_bp(f, q, r)), g) for f, q, g in P.res]
        self.f_dc2 = self._mk(_hp1(DC2_HZ, r))
        self.f_flow = self._mk(_lp1(P.flow_lp, r))
        pf = math.exp(-2.0 * math.pi * min(P.flow_lp, 0.45 * r) / r)
        self.g_flow = math.sqrt((1.0 - pf) / (1.0 + pf))    # white noise's RMS after it
        self.f_bright = self._mk(_lp2(P.lp_hi, 0.70, r))
        self._prof_filters = tuple(self._filters[n0:])
        self.bank_w = np.asarray(P.bank_mix, dtype=float) * 2.0 if self.nb == 2 else None
        self.c_eng = np.zeros(self.nb)
        self.flow_lvl = np.zeros(self.nb)
        self.cyl_gain = np.asarray(P.cyl_gain, dtype=float)
        self.bank = np.asarray(P.bank, dtype=np.intp)
        if self.t0 > 0:
            self.eng_on = 0.0

    # -- one-shots ---------------------------------------------------------------
    def _add_fx(self, y: np.ndarray, at: int = 0) -> None:
        end = at + len(y)
        if len(self.fx) < end:
            fx = np.zeros(end)
            fx[:len(self.fx)] = self.fx
            self.fx = fx
        self.fx[at:end] += y

    def clunk(self, level: float = 1.0) -> None:
        """A gear engaging: a body thunk (two damped modes ~210 / 440 Hz, a
        little sub-thud at 105 Hz for a bigger speaker) and a short metallic
        tick, each a little different.  The body was at 118 Hz, where 90 % of
        its energy sat under 126 Hz and a laptop speaker played only the tick
        (review); a gearbox case rings in the low hundreds."""
        r, rng = self.rate, self.rng
        n = int(0.09 * r)
        t = np.arange(n) / r
        f1 = 210.0 * (1.0 + 0.06 * rng.standard_normal())
        y = (np.exp(-t / 0.016) * np.sin(2 * math.pi * f1 * t)
             + 0.45 * np.exp(-t / 0.009) * np.sin(2 * math.pi * 2.1 * f1 * t)
             + 0.35 * np.exp(-t / 0.020) * np.sin(2 * math.pi * 0.5 * f1 * t)
             + 0.30 * np.exp(-t / 0.0025) * np.sin(2 * math.pi * 2350.0 * t))
        y[:int(0.0015 * r)] *= np.linspace(0.0, 1.0, int(0.0015 * r), endpoint=False)
        self._add_fx(y * A_CLUNK * float(level))

    def shift_cut(self, seconds: float = T_SHIFT_CUT) -> None:
        self.cut_until = max(self.cut_until, self.t0 + int(seconds * self.rate))

    def cue(self, kind: str) -> None:
        """Queue a chime after whatever chime is already queued."""
        if kind not in CUES:
            return
        y, t_last = _render_cue(kind, self.rate)
        at = self.cue_free
        self._add_fx(y, at)
        self.cue_free = at + int((t_last + CUE_GAP) * self.rate)

    def event(self, kind: str, arg=None) -> None:
        if kind == 'clunk':
            self.clunk(1.0 if arg is None else float(arg))
        elif kind == 'shift_cut':
            self.shift_cut(T_SHIFT_CUT if arg is None else float(arg))
        elif kind == 'cue':
            self.cue(str(arg))

    # -- the chunk ---------------------------------------------------------------
    def chunk(self, rpm: float, load: float, squeal: float = 0.0,
              grass: float = 0.0, wind: float = 0.0, limit: float = 0.0,
              stalled: bool = False, master: float = 1.0,
              volume: float | None = None, *, car: str | None = None,
              V: float = 0.0, gear: int = 1, lock: float = 0.0,
              abs_on: float = 0.0, scrub: float = 0.0, sq_pan: float = 0.0,
              grass_pan: float = 0.0, gravel: float = 0.0, gravel_pan: float = 0.0,
              kerb: float = 0.0, kerb_pan: float = 0.0, roll: float = 0.0,
              wet: float = 0.0, pops: float = 0.0, servo: float = 0.0,
              servo_pan: float = 0.0, servo_rate: float = 0.5,
              starter: float | None = None, **_ignored) -> np.ndarray:
        n, r, ramp = self.n, self.rate, self.ramp
        rng = self.rng
        if self._lazy:
            self._adopt_lfilter()
        if car and car != self.prof.key:
            self.set_profile(str(car))
        P = self.prof
        C = self.nch
        # ---- every target finite and in range (a bad one is its safe value) --
        rpm = _num(rpm, 0.0, 0.0, 20000.0)
        V = _num(V, 0.0, -150.0, 150.0)
        gear = _int(gear, 1)
        u = lambda x: _num(x, 0.0, 0.0, 1.0)                              # noqa: E731
        pm = lambda x: _num(x, 0.0, -1.0, 1.0)                            # noqa: E731
        squeal, grass, wind, limit, lock, abs_on, scrub, gravel, kerb, wet, pops, servo = \
            map(u, (squeal, grass, wind, limit, lock, abs_on, scrub, gravel, kerb, wet, pops, servo))
        roll = _num(roll, 0.0, 0.0, 2.0)
        sq_pan, grass_pan, gravel_pan, kerb_pan, servo_pan = \
            map(pm, (sq_pan, grass_pan, gravel_pan, kerb_pan, servo_pan))
        servo_rate = _num(servo_rate, 0.5, 0.0, 1.0)
        master = _num(master, 1.0, 0.0, 1.0)
        volume = _num(self.volume if volume is None else volume, self.volume, 0.0, 2.0)
        if starter is not None:
            starter = _num(starter, 0.0, 0.0, 1.0)

        def rmp(prev, tgt):
            return prev + (tgt - prev) * ramp

        # ---- targets ---------------------------------------------------------
        load = _num(load, 0.0, 0.0, 1.0)
        # idle wander: a slow random walk of the rpm target, gone by 1800 rpm
        # (by the profile's idle_hi: 1800 on every petrol, scale exactly 1)
        self.jit = 0.85 * self.jit + 0.15 * rng.standard_normal() * 14.0 * (1.0 - load)
        ih = self.prof.idle_hi
        rpm_t = max(rpm, 0.0) + self.jit * _clip01((ih - rpm) / (900.0 * (ih / 1800.0)))
        if rpm <= 0.0:
            rpm_t = 0.0
        eng_on_t = _clip01((rpm - RPM_CRANK) / (RPM_RUN - RPM_CRANK))
        if starter is None:             # the old callers: stalled and turning
            starter = 1.0 if (stalled and rpm > 20.0) else 0.0
        st_t = _clip01(starter) if rpm > 20.0 else 0.0
        if stalled:
            eng_on_t *= 0.35            # compression puffs while it turns over
        rpm_a = rmp(self.rpm, rpm_t)
        load_a = rmp(self.load, load)
        on_prev, on_t = self.eng_on, eng_on_t
        lim_prev, pops_prev = self.limit, self.pops
        whine_t = _clip01(abs(V) / 1.0) * (REV_WHINE_X if gear < 0 else 1.0) \
            * (1.0 if V > 0.3 else 0.0)
        self.rpm, self.load, self.eng_on = rpm_t, load, eng_on_t

        out = np.zeros((C, n))
        t = self.tsec + self.t0 / r
        self._ph_valid = False

        # ---- engine ------------------------------------------------------------
        if pops <= 0.0:
            self.pop_sched = []                 # the window is over (or killed)
        elif pops > pops_prev + 0.05:
            self._arm_pops(pops)                # a new window: its budget
        if on_prev > 0.0 or on_t > 0.0:
            self._engine(out, rpm_a, load_a, rmp(on_prev, on_t),
                         rmp(lim_prev, limit), rmp(pops_prev, pops), t)
        self.limit, self.pops = limit, pops

        # ---- starter -------------------------------------------------------------
        if self.starter > 0.0 or st_t > 0.0:
            st_a = rmp(self.starter, st_t)
            f = STARTER_RATIO * np.maximum(rpm_a, 30.0)
            # the compressions load the motor: once per firing slot it slows
            if self._ph_valid:
                ph_c = self.ph_eng_samples
            else:                           # engine silent this chunk: own crank phase
                ph_c = self.ph_eng + np.cumsum(rpm_a) / (120.0 * r)
                self.ph_eng = float(ph_c[-1] % 4096.0)
            comp = np.cos(2.0 * math.pi * ((ph_c * P.slots) % 1.0))
            f = f * (1.0 + 0.07 * comp)
            ph = self.ph_st + np.cumsum(f) / r
            self.ph_st = float(ph[-1] % 1.0)
            w = 2.0 * math.pi * ph
            sig = (np.sin(w) + 0.5 * np.sin(2.0 * w) + 0.25 * np.sin(3.0 * w)) \
                * (0.62 - 0.38 * comp) * st_a * A_STARTER
            out += sig
        self.starter = st_t

        # ---- drivetrain whine -----------------------------------------------------
        k_w = K_WHINE_REV if gear < 0 else K_WHINE
        if self.whine > 0.0 or whine_t > 0.0:
            f = np.maximum(rmp(self.k_whine * self.V, k_w * abs(V)), 0.0)   # a gear flip glides
            ph = self.ph_whine + np.cumsum(f) / r
            self.ph_whine = float(ph[-1] % 1.0)
            w = 2.0 * math.pi * ph
            amp = rmp(self.whine, whine_t) * A_WHINE \
                * (0.6 + 0.4 * _clip01(abs(V) / 30.0)) * (1.0 + 0.5 * load)
            out += (np.sin(w) + 0.35 * np.sin(2.0 * w)) * amp
        self.whine, self.k_whine = whine_t, k_w

        # ---- the chassis thump path (kerb, verge bumps, ABS) and the crunch --------
        th_pos, th_amp, th_row = [], [], []
        cr_pos, cr_amp, cr_row = [], [], []
        Vs = max(abs(V), abs(self.V))

        def sided(level_prev, level, pan_prev, pan):
            """Per-row levels (start, end) of a panned source."""
            if C == 1:
                return [(level_prev, level)]
            gl0, gr0 = _pan_gains(pan_prev)
            gl1, gr1 = _pan_gains(pan)
            return [(level_prev * gl0, level * gl1), (level_prev * gr0, level * gr1)]

        if self.kerb > 0.0 or kerb > 0.0:
            inc = np.full(n, max(Vs, 0.5) / KERB_PITCH / r)
            pos, _, ph = _crossings(self.ph_kerb, inc, 1)
            self.ph_kerb = float(ph[-1] % 1.0)
            if len(pos):
                fr = np.minimum(pos, n - 1) / n
                for row, (l0, l1) in enumerate(sided(self.kerb, kerb, self.kerb_pan, kerb_pan)):
                    a = (l0 + (l1 - l0) * fr) * A_KERB * (1.0 + 0.12 * rng.standard_normal(len(pos)))
                    th_pos.append(pos)
                    th_amp.append(a)
                    th_row.append(np.full(len(pos), row))
                    cr_pos.append(pos)
                    cr_amp.append(a * KERB_RATTLE)
                    cr_row.append(np.full(len(pos), row))
        if self.abs_on > 0.0 or abs_on > 0.0:
            inc = np.full(n, ABS_HZ / r)
            pos, _, ph = _crossings(self.ph_abs, inc, 1)
            self.ph_abs = float(ph[-1] % 1.0)
            lv = max(self.abs_on, abs_on) * _clip01(Vs / 4.0) * A_ABS
            if len(pos) and lv > 0.0:
                for row in range(C):
                    th_pos.append(pos)
                    th_amp.append(np.full(len(pos), lv))
                    th_row.append(np.full(len(pos), row))
        for lvl_p, lvl, pan_p, pan, kind in ((self.grass, grass, self.grass_pan, grass_pan, 'g'),
                                             (self.gravel, gravel, self.gravel_pan, gravel_pan, 'r')):
            if lvl_p <= 0.0 and lvl <= 0.0:
                continue
            lv = max(lvl_p, lvl)
            rows_lv = sided(lvl_p, lvl, pan_p, pan)
            # bumps: a few big ones a second, more with speed
            rate_b = (2.5 + 0.6 * Vs) if kind == 'g' else (4.0 + 0.8 * Vs)
            for row, (l0, l1) in enumerate(rows_lv):
                m = rng.poisson(rate_b * n / r * lv)
                if m:
                    th_pos.append(rng.random(m) * n)
                    th_amp.append(A_THUMP * 0.5 * (l0 + l1) * _exp_capped(rng, m)
                                  * (0.4 + 0.6 * _clip01(Vs / 15.0)))
                    th_row.append(np.full(m, row))
                # the rumble: dense small bumps
                m = rng.poisson(90.0 * n / r * lv * _clip01(Vs / 1.0))
                if m:
                    th_pos.append(rng.random(m) * n)
                    th_amp.append(A_RUMBLE * 0.5 * (l0 + l1) * rng.standard_normal(m))
                    th_row.append(np.full(m, row))
                if kind == 'r':                    # the crunch: stones
                    m = rng.poisson((40.0 + 30.0 * Vs) * n / r * _clip01(Vs / 0.5))
                    if m:
                        cr_pos.append(rng.random(m) * n)
                        cr_amp.append(A_GRAVEL * 0.5 * (l0 + l1) * _exp_capped(rng, m)
                                      * np.where(rng.random(m) < 0.5, -1.0, 1.0))
                        cr_row.append(np.full(m, row))
        self.live_thump = 2 if th_pos else self.live_thump - 1
        if self.live_thump > 0:
            pos = np.concatenate(th_pos) if th_pos else np.zeros(0)
            exc, self.c_thump = _place(n, np.minimum(pos, n), np.concatenate(th_amp) if th_amp else pos,
                                       np.concatenate(th_row) if th_row else pos.astype(np.intp),
                                       C, self.c_thump)
            out += self.f_thump1(exc) * self.g_thump1 + self.f_thump2(exc) * self.g_thump2
        self.live_crunch = 2 if cr_pos else self.live_crunch - 1
        if self.live_crunch > 0:
            pos = np.concatenate(cr_pos) if cr_pos else np.zeros(0)
            exc, self.c_crunch = _place(n, np.minimum(pos, n), np.concatenate(cr_amp) if cr_amp else pos,
                                        np.concatenate(cr_row) if cr_row else pos.astype(np.intp),
                                        C, self.c_crunch)
            out += self.f_crunch(exc) * self.g_crunch
        self.kerb, self.kerb_pan = kerb, kerb_pan
        abs_prev, self.abs_on = self.abs_on, abs_on

        # ---- surface beds: grass swish, tarmac hum, wet spray -----------------------
        sw_p, sw = self.grass + 0.5 * self.gravel, grass + 0.5 * gravel
        if sw_p > 0.0 or sw > 0.0:
            g = _bp_noise_gain(480.0, 0.75, r)
            sig = self.f_swish(self._noise(1))[0] * g * A_GRASS \
                * (0.35 + 0.65 * _clip01(Vs / 20.0))
            pan_m = (grass_pan * grass + gravel_pan * gravel) / max(grass + gravel, 1e-9)
            pan_p = (self.grass_pan * self.grass + self.gravel_pan * self.gravel) \
                / max(self.grass + self.gravel, 1e-9)
            self._add_panned(out, sig, sw_p, sw, pan_p, pan_m)
        self.grass, self.grass_pan, self.gravel, self.gravel_pan = grass, grass_pan, gravel, gravel_pan
        if self.roll > 0.0 or roll > 0.0:
            nz = self._noise(1)
            sig = (self.f_roll(nz)[0] * _bp_noise_gain(820.0, 0.8, r)
                   + 0.8 * self.f_rumble(nz)[0] * _bp_noise_gain(115.0, 1.0, r))
            out += sig * rmp(self.roll, roll) * A_ROLL
        self.roll = roll
        if self.wet > 0.0 or wet > 0.0:
            out += self.f_wet(self._noise(C)) * (rmp(self.wet, wet) * A_WET * _bp_noise_gain(F_WET, 0.6, r))
        self.wet = wet

        # ---- wind, with gusts ------------------------------------------------------
        if self.wind > 0.0 or wind > 0.0:
            dtc = n / r
            g0 = self.gust
            self.gust = g0 * math.exp(-dtc / 1.6) + 0.55 * math.sqrt(dtc / 1.6) * rng.standard_normal()
            fw = 260.0 + 7.5 * Vs * (1.0 + 0.08 * self.gust)
            self.f_wind.set(_bp(fw, 0.6, r))
            gain = _bp_noise_gain(fw, 0.6, r) * A_WIND
            gust = np.clip(1.0 + 0.22 * rmp(g0, self.gust), 0.4, 1.8)
            out += self.f_wind(self._noise(C)) * (rmp(self.wind, wind) * gust * gain)
        self.wind = wind

        # ---- tyres: scrub, squeal (+ lock, ABS) ---------------------------------------
        if self.scrub > 0.0 or scrub > 0.0:
            sig = self.f_scrub(self._noise(1))[0] * _bp_noise_gain(F_SCRUB, Q_SCRUB, r) * A_SCRUB
            self._add_panned(out, sig, self.scrub, scrub, self.sq_pan, sq_pan)
        self.scrub = scrub
        if self.squeal > 0.0 or squeal > 0.0:
            # the lock-up share, the band centre and the ABS gate all RAMP
            # across the chunk from last chunk's values: stepped, they put a
            # 2.8x burst of > 9 kHz energy on the boundary (review)
            lk = _clip01(lock / max(squeal, lock, 1e-6))          # how much of it is lock-up
            fc = F_SQUEAL * (1.0 + 0.18 * squeal) * (1.0 - (1.0 - F_LOCK_X) * lk)
            fc0 = self.sq_fc if self.sq_fc > 0.0 else fc
            lk_a = rmp(self.sq_lk if self.sq_fc > 0.0 else lk, lk)
            nz = self._noise(1)
            if abs(fc - fc0) > 0.5:
                # crossfade the band from the old centre (a filter that carries
                # on exactly as last chunk's) to the new one (which starts
                # from that state with new coefficients: weight 0 there)
                self.f_sq0.copy_from(self.f_sq)
                self.f_sq.set(_bp(fc, Q_SQUEAL, r))
                b0 = self.f_sq0(nz)[0] * _bp_noise_gain(fc0, Q_SQUEAL, r)
                b1 = self.f_sq(nz)[0] * _bp_noise_gain(fc, Q_SQUEAL, r)
                band = b0 + (b1 - b0) * ramp
            else:
                self.f_sq.set(_bp(fc, Q_SQUEAL, r))
                band = self.f_sq(nz)[0] * _bp_noise_gain(fc, Q_SQUEAL, r)
            dtc = n / r
            w0 = self.wander
            self.wander = w0 * math.exp(-dtc / 0.25) + 0.9 * math.sqrt(dtc / 0.25) * rng.standard_normal()
            f_ch = rmp(fc0, fc) * (1.0 + 0.035 * rmp(w0, self.wander)
                                   + 0.012 * np.sin(2.0 * math.pi * 6.5 * t))
            ph = self.ph_sq + np.cumsum(f_ch) / r
            self.ph_sq = float(ph[-1] % 1.0)
            w = 2.0 * math.pi * ph
            tone = np.sin(w) + 0.25 * np.sin(2.0 * w) + 0.08 * np.sin(3.0 * w)
            sig = 0.60 * band + A_SQ_TONE * tone
            if lk > 0.0 or lk_a[0] > 0.0:                          # harsher: a 43 Hz AM
                sig = sig * (1.0 + 0.7 * lk_a * np.sin(2.0 * math.pi * 43.0 * t))
            if abs_on > 0.0 or abs_prev > 0.0:                     # ABS chops it at 15 Hz
                gate = 0.5 + 0.5 * np.tanh(5.0 * np.sin(2.0 * math.pi * ABS_HZ * t))
                sig = sig * (1.0 - 0.85 * rmp(abs_prev, abs_on) * (1.0 - gate))
            self._add_panned(out, sig * A_SQUEAL, self.squeal, squeal, self.sq_pan, sq_pan)
            self.sq_fc, self.sq_lk = fc, lk
        else:
            self.sq_fc = self.sq_lk = 0.0
        self.squeal, self.lock = squeal, lock
        self.sq_pan = sq_pan

        # ---- the wing actuator -------------------------------------------------------
        if self.servo > 0.0 or servo > 0.0:
            f = 420.0 + 520.0 * rmp(self.servo_rate, _clip01(servo_rate))
            ph = self.ph_srv + np.cumsum(f) / r
            # wrapped modulo 9, not 1: the sin(w / 9) AM rides on this phase,
            # and a mod-1 wrap stepped it at every chunk (a 21.5 Hz zipper)
            self.ph_srv = float(ph[-1] % 9.0)
            w = 2.0 * math.pi * ph
            sig = (np.sin(w) + 0.45 * np.sin(2.0 * w) + 0.25 * np.sin(3.0 * w)) \
                * (1.0 + 0.3 * np.sin(w / 9.0)) * A_SERVO
            self._add_panned(out, sig, self.servo, servo, self.servo_pan, servo_pan)
        self.servo, self.servo_pan, self.servo_rate = servo, servo_pan, _clip01(servo_rate)
        self.V = abs(V)

        # ---- one-shots ---------------------------------------------------------------
        if len(self.fx):
            m = min(n, len(self.fx))
            out[:, :m] += self.fx[:m]
            self.fx = self.fx[m:]
        self.cue_free = max(0, self.cue_free - n)
        self.t0 += n

        # ---- bus: compressor (chunk peak) -> soft clip -> master x volume -------------
        ms_a = rmp(self.master, master)
        vol_a = rmp(self.volume, volume)
        self.master, self.volume = master, volume
        pk = float(np.max(np.abs(out)))
        if not math.isfinite(pk) or not math.isfinite(self.bus_g):
            self._fault_reset()                 # silence now, sound next chunk
            return np.zeros(n) if C == 1 else np.zeros((n, 2))
        # the detector: the chunk's peak, but never more than BUS_CREST x its
        # RMS.  The flow noise and the briefer WOT pulse made the exhaust
        # peakier (crest ~13 dB, was ~11), and keyed on the raw peak the
        # compressor ducked the whole mix for a few spikes (~1 dB more at
        # 3000-6000 rpm WOT, measured); the spikes above are the soft clip's
        pk = min(pk, BUS_CREST * math.sqrt(float(np.vdot(out, out)) / out.size))
        g_t = 1.0 if pk <= BUS_THR else (BUS_THR / pk) ** (1.0 - 1.0 / BUS_RATIO)
        g1 = g_t if g_t < self.bus_g else self.bus_g + (g_t - self.bus_g) * self.rel_k
        gain = rmp(self.bus_g, g1) * (1.0 / CEIL)
        self.bus_g = g1
        # the soft clip squashes a pulse train's tall positive fronts more
        # than its lobes: -0.004 DC (2.5 % of RMS) at WOT - blocked here,
        # before master x volume so volume 0 is still exact zeros
        y = self.f_out(np.tanh(out * gain)) * (CEIL * ms_a * vol_a)
        if C == 1:
            return y[0]
        return np.ascontiguousarray(y.T)

    def _add_panned(self, out, sig, lv_prev, lv, pan_prev, pan) -> None:
        ramp = self.ramp
        if self.nch == 1:
            out[0] += sig * (lv_prev + (lv - lv_prev) * ramp)
            return
        gl0, gr0 = _pan_gains(pan_prev)
        gl1, gr1 = _pan_gains(pan)
        out[0] += sig * (lv_prev * gl0 + (lv * gl1 - lv_prev * gl0) * ramp)
        out[1] += sig * (lv_prev * gr0 + (lv * gr1 - lv_prev * gr0) * ramp)

    # -- the engine --------------------------------------------------------------------
    def _arm_pops(self, strength: float) -> None:
        """A new pop window (the mapper confirmed a lift): its whole budget is
        drawn now - P.pops per full-strength lift, fewer from lower rpm - and
        each pop given a time, exponential with POP_TAU (truncated at
        POP_T_MAX), so they come thick at first and thin out.  A per-firing
        probability made 13-31 pops per lift on the MX-5 (review)."""
        P, rng = self.prof, self.rng
        lo, hi = P.pops
        k = int(round(int(rng.integers(lo, hi + 1)) * _clip01(strength)))
        if k <= 0:
            self.pop_sched = []
            return
        u = rng.random(k)
        tt = -POP_TAU * np.log(1.0 - u * (1.0 - math.exp(-POP_T_MAX / POP_TAU)))
        self.pop_sched = sorted(int(self.t0 + x * self.rate) for x in tt)

    def _engine(self, out, rpm_a, load_a, on_a, lim_a, pops_a, t) -> None:
        P, n, r, rng = self.prof, self.n, self.rate, self.rng
        N, nb = P.slots, self.nb
        ramp = self.ramp
        pos, k, ph = _crossings(self.ph_eng, rpm_a / (120.0 * r), N)
        self.ph_eng = float(ph[-1] % 4096.0)
        self.ph_eng_samples = ph
        self._ph_valid = True
        rpm_m = float(rpm_a.mean())
        load_m = float(load_a.mean())
        f_fire = max(rpm_m, 60.0) / 120.0 * N
        # the pulse: a sharp front (the valve cracking open; sharper at load)
        # and an exponential blowdown whose length follows the firing period,
        # briefer at load
        fc_p = P.pulse_k * f_fire * (0.75 + PULSE_LOAD * load_m) + 20.0
        self.f_pulse.set(_lp1(fc_p, r))
        self.f_rise.set(_lp1(P.rise * (RISE_LO + (RISE_HI - RISE_LO) * load_m ** 0.8), r))
        norm = r / (2.0 * math.pi * fc_p) * (fc_p / FC_REF) ** PULSE_GAMMA
        rpm_x = _clip01(rpm_m / P.cut)
        cr_pos = cr_amp = None
        m = len(pos)
        if m:
            li = np.minimum(pos.astype(np.intp), n - 1)
            ld = load_a[li]
            slot = (k % N).astype(np.intp)
            ih = P.idle_hi                      # 1800 on every petrol (scale 1.0)
            idle = np.clip((ih - rpm_a[li]) / (1000.0 * (ih / 1800.0)), 0.0, 1.0)
            z = rng.standard_normal(m)
            # flow rises with rpm: the pulse's height with it, steeper at the top
            rf = 0.52 + 0.48 * np.clip(rpm_a[li] / P.cut, 0.0, 1.2) ** 1.4
            # cycle-to-cycle variation: large and lumpy at idle, ~6 % at load
            # (est; published COV of IMEP is 2-5 % loaded, 10-20 % at idle),
            # rough again toward a shut throttle (JIT_OVR)
            sd = P.jitter * idle + (JIT_LOAD + JIT_OVR * P.jitter * (1.0 - ld) ** 2) * (1.0 - idle)
            A = (ENG_A0 + (1.0 - ENG_A0) * ld ** 0.8) * self.cyl_gain[slot] * (1.0 + sd * z) * rf
            per = r / np.maximum(rpm_a[li] / 120.0 * N, 1.0)
            pos = np.clip(pos + (0.012 * idle + 0.005) * per * rng.standard_normal(m),
                          0.0, n - 1e-6)
            # ignition cuts: the limiter's stutter, an upshift
            cut = np.zeros(m, dtype=bool)
            if lim_a[-1] > 0.0 or lim_a[0] > 0.0:
                tt = (self.t0 + pos) / r
                cut |= (lim_a[li] > 0.5) & (((tt * LIMIT_HZ) % 1.0) < LIMIT_DUTY)
            if self.cut_until > self.t0:
                cut |= (self.t0 + pos) < self.cut_until
            A = np.where(cut, A * CUT_FLOOR, A)
            # pops: this window's scheduled ones, each on the first overrun
            # firing at or after its time (a firing burning late, in the pipe)
            pop = np.zeros(m, dtype=bool)
            if self.pop_sched:
                over = (ld < 0.15) & ~cut
                keep = []
                end = self.t0 + n
                for ts in self.pop_sched:
                    if ts >= end:
                        keep.append(ts)
                        continue
                    c = np.flatnonzero(over & ~pop & (pos >= ts - self.t0))
                    if c.size:
                        pop[c[0]] = True
                    elif ts - self.t0 > -0.15 * r:      # none yet: wait a little
                        keep.append(ts)
                self.pop_sched = keep
            # ... and the limiter's: a cut firing re-lighting, P.pop_lim a second
            if cut.any() and lim_a.max() > 0.5:
                kl = rng.poisson(P.pop_lim * float((lim_a > 0.5).mean()) * n / r)
                if kl:
                    c = np.flatnonzero(cut & ~pop)
                    if c.size:
                        pop[rng.choice(c, min(kl, c.size), replace=False)] = True
            if pop.any():
                A = np.where(pop, POP_GAIN * rf * (0.75 + 0.5 * rng.random(m)), A)
                self.pops_fired += int(pop.sum())
                if self.pop_log is not None:
                    self.pop_log.extend(int(self.t0 + q) for q in pos[pop])
            # the crack path: valvetrain ticks under the idle, the pops' crack
            tick = P.tick * A_TICK * (0.4 + 0.6 * idle) * (1.0 - 0.5 * ld) * (1.0 + 0.4 * z)
            cr_amp = np.where(pop, A_CRACK * (0.5 + rng.random(m)), tick)
            cr_pos = pos
            rows = self.bank[slot] if nb == 2 else np.zeros(m, dtype=np.intp)
            exc, self.c_eng = _place(n, pos, A * (norm * PULSE_H), rows, nb, self.c_eng)
        else:
            exc, self.c_eng = _place(n, pos, pos, pos.astype(np.intp), nb, self.c_eng)
        # crackles in the pop window
        if pops_a[-1] > 0.0 or pops_a[0] > 0.0:
            mc = rng.poisson(float(pops_a.mean()) * P.crackle * n / r)
            if mc:
                cp = rng.random(mc) * n
                ca = A_CRACKLE * _exp_capped(rng, mc) * np.where(rng.random(mc) < 0.5, -1.0, 1.0)
                cr_pos = cp if cr_pos is None else np.concatenate((cr_pos, cp))
                cr_amp = ca if cr_amp is None else np.concatenate((cr_amp, ca))
        # the firing-phase envelope: 1 at each firing, decaying to 0.007
        env = np.exp(-((ph * N) % 1.0) * 5.0)
        # the blowdown pulses and their flow noise: turbulence at the pulse
        # train's own level (its RMS this chunk, ramped), swelling at each
        # firing.  Riding each pulse multiplicatively instead put the noise's
        # peaks on the pulse's peaks: 5x the pulse height on the V8, and the
        # bus compressor at -12 dB.  It grows with the mass flow, load x rpm,
        # and a shut throttle keeps FLOW_OVR's floor of it
        x = self.f_pulse(self.f_rise(exc))
        lvl = np.sqrt(np.mean(x * x, axis=1))
        flow_a = load_a
        if FLOW_OVR > 0.0 and max(rpm_a[0], rpm_a[-1]) > FLOW_OVR_X0 * P.cut:
            flow_a = np.maximum(load_a, FLOW_OVR * np.clip(
                (rpm_a / P.cut - FLOW_OVR_X0) / (1.0 - FLOW_OVR_X0), 0.0, 1.0))
        if flow_a[0] > 0.0 or flow_a[-1] > 0.0:
            kf = (P.flow * A_FLOW / self.g_flow) * flow_a * (0.65 + 0.35 / P.cut * rpm_a) \
                * (0.4 + 0.6 * env)
            lv = self.flow_lvl[:, None] + (lvl - self.flow_lvl)[:, None] * ramp
            x = x + kf * lv * self.f_flow(self._noise(nb))
        self.flow_lvl = lvl
        # the exhaust
        x = self.f_dc(x)
        x = self.comb(x)
        y = P.direct * x
        for f, g in self.f_res:
            y = y + g * f(x)
        if P.rasp > 0.0:
            y = y + (P.rasp / RASP_REF) * y * np.abs(y)
            y = self.f_dc2(y)                   # the waveshaper's DC
        # the closed-throttle corner rises with rpm too: fixed, the V8's
        # overrun fundamental (400 Hz at 6000 rpm) sat above it and the
        # overrun fell 4.6 dB toward the cut
        fb = P.lp_lo * (0.75 + 0.6 * rpm_x) \
            + (P.lp_hi - P.lp_lo) * (load_m ** 0.7) * (0.55 + 0.45 * rpm_x)
        self.f_bright.set(_lp2(fb, 0.70, r))
        y = self.f_bright(y)
        gain = A_ENGINE * P.level * on_a
        if nb == 2:
            w0, w1 = self.bank_w
            if self.nch == 1:
                out[0] += (w0 * y[0] + w1 * y[1]) * gain
            else:
                (l0, l1), (r0, r1) = P.bank_lr
                out[0] += (l0 * y[0] + l1 * y[1]) * gain
                out[1] += (r0 * y[0] + r1 * y[1]) * gain
        else:
            out += y[0] * gain
        rpm_xa = np.clip(rpm_a / P.cut, 0.0, 1.0)
        # the mechanical bed: valvetrain / chain / injector noise, ticking with
        # the firings, rising with rpm - what fills a slow idle between pulses
        if P.mech > 0.0:
            sig = self.f_mech(self._noise(1))[0] * (_bp_noise_gain(F_MECH, Q_MECH, r) * P.mech * A_MECH)
            out += sig * (0.7 + 0.3 * env) * on_a * (1.0 + 2.0 * rpm_xa)
        # intake roar: band noise gated by the firing phase, rpm x load
        lv_in = P.intake * A_INTAKE * load_m ** 1.3 * rpm_x ** 1.8
        if lv_in > 1e-4 or self._intake_on:
            self._intake_on = lv_in > 1e-4
            fi = P.intake_f * (0.8 + 0.5 * rpm_x)
            self.f_intake.set(_bp(fi, 1.3, r))
            sig = self.f_intake(self._noise(1))[0] * env * _bp_noise_gain(fi, 1.3, r)
            out += sig * (on_a * (P.intake * A_INTAKE) * load_a ** 1.3 * rpm_xa ** 1.8)
        # the crack path (centre)
        self.live_crack = 2 if cr_pos is not None else self.live_crack - 1
        if self.live_crack > 0:
            if cr_pos is None:
                cr_pos = cr_amp = np.zeros(0)
            exc, self.c_crack = _place(n, np.minimum(cr_pos, n), cr_amp,
                                       np.zeros(len(cr_pos), dtype=np.intp), 1, self.c_crack)
            out += self.f_crack(exc)[0] * (on_a * self.g_crack)


# ====================================================================== #
#  HUD -> TARGETS                                                        #
# ====================================================================== #
def _arr4(v, default=0.0, lo=-1e9, hi=1e9):
    """Four per-wheel floats, each through `_num` (a NaN wheel is `default`);
    anything that is not four numbers is four defaults."""
    try:
        a = [_num(x, default, lo, hi) for x in v]
    except TypeError:
        return [default] * 4
    return a if len(a) == 4 else [default] * 4


def flank_deps(hud) -> tuple:
    """(dep_left, dep_right): render.flank_deps' law, re-stated so this
    module never imports the renderer."""
    dl = getattr(hud, 'wing_deploy_l', None)
    dr = getattr(hud, 'wing_deploy_r', None)
    if dl is not None and dr is not None:
        return _num(dl, 0.0, 0.0, 1.0), _num(dr, 0.0, 0.0, 1.0)
    dep = _num(getattr(hud, 'wing_deploy', 0.0), 0.0, 0.0, 1.0)
    side = _int(getattr(hud, 'wing_side', 0), 0)
    if dep <= 0.0 or side == 0:
        return 0.0, 0.0
    return (dep, 0.0) if side < 0 else (0.0, dep)


class _Mapper:
    """HudData -> the synth's targets and one-shot events, once per render
    frame.  Stateful: first-order smoothing, the edges (gear, chimes, the
    lift that arms the pops) and the deploy derivatives live here.  Every
    float read goes through `_num` (finite, a default, a range), so no HUD
    value can put a NaN into a smoother."""

    def __init__(self):
        self.t_last = None
        self.sq = self.lock = self.scrub = self.pan = self.abs = 0.0
        self.grass = self.gravel = self.kerb = 0.0
        self.grass_pan = self.gravel_pan = self.kerb_pan = 0.0
        self.roll = self.wet = self.wind = 0.0
        self.srv = self.srv_pan = self.srv_rate = 0.0
        self.dep_prev = None
        self.rpm_prev = None
        self.drpm = 0.0
        self.load_env = 0.0
        self.pops = 0.0
        self.lift = None                # a lift awaiting confirmation (see __call__)
        self.gear = None                # the last gear read (0 = neutral)
        self.gear_eng = None            # the gear that was engaged before a neutral
        self.t_neutral = 0.0            # how long the current neutral has lasted
        self.neutral_held = True        # this neutral outlasted T_GATE_MAX (clunked)
        self.flash_prev = None
        self.first = True

    def __call__(self, hud, dt: float | None = None) -> dict:
        g = getattr
        if dt is None:
            now = time.monotonic()
            dt = 1.0 / 60.0 if self.t_last is None else min(max(now - self.t_last, 1e-3), 0.1)
            self.t_last = now
        dt = _num(dt, 1.0 / 60.0, 1e-4, 0.25)

        def smooth(cur, tgt, tau):
            return cur + (tgt - cur) * min(dt / tau, 1.0)

        key = profile_key(hud)
        P = PROFILES[key]
        rpm = _num(g(hud, 'rpm', 0.0), 0.0, 0.0, 12000.0)
        load = _num(g(hud, 'eng_load', 0.0), 0.0, 0.0, 1.0)
        V = abs(_num(g(hud, 'V', 0.0), 0.0, -100.0, 100.0))
        stalled = bool(g(hud, 'stalled', False))
        limit = 1.0 if g(hud, 'on_limiter', False) else 0.0
        on_track = bool(g(hud, 'on_track', True))
        paused = bool(g(hud, 'paused', False))
        gear = _int(g(hud, 'gear', None), None)

        # ---- what each wheel is on --------------------------------------------
        s4 = g(hud, 'surf4', None)
        try:
            s4 = [str(s) if s else 'tarmac' for s in s4] if s4 is not None else None
        except TypeError:
            s4 = None
        if s4 is None or len(s4) != 4:
            s4 = ['tarmac'] * 4 if on_track else ['grass'] * 4
        sq_w = [SURF_SQUEAL.get(s, 1.0) for s in s4]
        hard = sum(SURF_HARD.get(s, 1.0) for s in s4) / 4.0

        def sides(kind):
            nl = (s4[0] == kind) + (s4[2] == kind)
            nr = (s4[1] == kind) + (s4[3] == kind)
            pan = PAN_SURF * (nr - nl) / (nl + nr) if nl + nr else 0.0
            return nl + nr, pan

        # ---- tyres ---------------------------------------------------------------
        util = max(_num(g(hud, 'util_f', 0.0), 0.0, 0.0, 3.0),
                   _num(g(hud, 'util_r', 0.0), 0.0, 0.0, 3.0))
        kap = _arr4(g(hud, 'kappa', None), 0.0, -2.0, 2.0)
        alpha = _arr4(g(hud, 'alpha', None), 0.0, -1.4, 1.4)
        Fx = _arr4(g(hud, 'Fx', None), 0.0, -1e5, 1e5)
        Fy = _arr4(g(hud, 'Fy', None), 0.0, -1e5, 1e5)
        vg = _clip01((V - V_SQUEAL) / 3.0)
        surf_sq = sum(sq_w) / 4.0
        lat = _clip01((util - UTIL_SQUEAL) / (UTIL_FULL - UTIL_SQUEAL))
        spin = _clip01((max(kap) - KAPPA_SQUEAL) / (KAPPA_FULL - KAPPA_SQUEAL))
        lock = _clip01((-min(kap) - KAPPA_LOCK) / (KAPPA_FULL - KAPPA_LOCK))
        sq_t = max(lat, spin, lock) * vg * surf_sq
        lock_t = lock * vg * surf_sq
        tau = TAU_SQUEAL_UP if sq_t > self.sq else TAU_SQUEAL_DOWN
        self.sq = smooth(self.sq, sq_t, tau)
        self.lock = smooth(self.lock, lock_t, TAU_SQUEAL_UP if lock_t > self.lock else TAU_SQUEAL_DOWN)
        scrub_t = _clip01((util - UTIL_SCRUB) / (UTIL_SCRUB_FULL - UTIL_SCRUB)) \
            * (1.0 - 0.6 * lat) * vg * surf_sq
        self.scrub = smooth(self.scrub, scrub_t, TAU_LEVEL)
        w = [(abs(Fy[i] * math.tan(alpha[i])) + abs(Fx[i] * kap[i])) * sq_w[i]
             for i in range(4)]
        wl, wr = w[0] + w[2], w[1] + w[3]
        pan_t = PAN_SQ * (wr - wl) / (wr + wl) if wr + wl > 1e-6 else 0.0
        self.pan = smooth(self.pan, pan_t, TAU_PAN)
        abs_t = 1.0 if g(hud, 'abs_active', False) else 0.0
        self.abs = smooth(self.abs, abs_t, 0.03 if abs_t > self.abs else 0.12)

        # ---- road ----------------------------------------------------------------
        nk, kpan = sides('kerb')
        ng, gpan = sides('grass')
        nr, rpan = sides('gravel')
        nw = sum(1 for s in s4 if s == 'wet')
        verge = (0.30 + 0.45 * _clip01(V / 20.0)) * _clip01(V / 0.8)
        self.kerb = smooth(self.kerb, min(nk / 2.0, 1.0) * _clip01(V / 2.0), 0.02)
        self.grass = smooth(self.grass, verge * (ng / 4.0) ** 0.6, TAU_LEVEL)
        self.gravel = smooth(self.gravel, verge * (nr / 4.0) ** 0.6, TAU_LEVEL)
        if nk:
            self.kerb_pan = kpan
        if ng:
            self.grass_pan = gpan
        if nr:
            self.gravel_pan = rpan
        self.roll = smooth(self.roll, hard * min(V / V_ROLL_REF, 1.6) ** 1.3, TAU_LEVEL)
        self.wet = smooth(self.wet, (nw / 4.0) * _clip01(V / 25.0) ** 1.2, TAU_LEVEL)
        self.wind = smooth(self.wind, _clip01((V / V_WIND_REF) ** 2), TAU_LEVEL)

        # ---- the active wing's actuator -------------------------------------------
        dl, dr = flank_deps(hud)
        dtop = _num(g(hud, 'top_deploy', 0.0), 0.0, 0.0, 1.0)
        if self.dep_prev is None:
            rl = rr = rt = 0.0
        else:
            rl = abs(dl - self.dep_prev[0]) / dt
            rr = abs(dr - self.dep_prev[1]) / dt
            rt = abs(dtop - self.dep_prev[2]) / dt
        self.dep_prev = (dl, dr, dtop)
        lvs = (_clip01(rl / SERVO_RATE_FULL), _clip01(rr / SERVO_RATE_FULL),
               _clip01(rt / SERVO_RATE_FULL))
        srv_t = max(lvs)
        self.srv = smooth(self.srv, srv_t, TAU_SERVO_UP if srv_t > self.srv else TAU_SERVO_DOWN)
        if srv_t > 0.0:
            self.srv_pan = PAN_SERVO * (lvs[1] - lvs[0]) / (sum(lvs) + 1e-9)
            self.srv_rate = _clip01(max(rl, rr, rt) / (1.5 * SERVO_RATE_FULL))

        # ---- the starter: stalled AND turning without spinning down.  `stalled`
        # also reads True while a dead engine runs down from n_stall to rest,
        # and a starter whine sweeping down with it is wrong (it is not engaged)
        if self.rpm_prev is not None:
            self.drpm = smooth(self.drpm, (rpm - self.rpm_prev) / dt, 0.05)
        self.rpm_prev = rpm
        starter = 1.0 if (stalled and rpm > 20.0 and self.drpm > STARTER_DRPM) else 0.0

        # ---- gears: the physics shifts g -> 0 -> g' (declutch 0.15 s at the old
        # gear with the load cut, the neutral gate 0.25 s, the engage).  That is
        # ONE shift: one clunk as the new gear engages, the cut on an upshift.
        # Read edge by edge it was two clunks and never a cut (review).  A
        # neutral that outlasts T_GATE_MAX is the driver's: its clunk then.
        events = []
        if gear is not None:
            if self.gear is None:
                self.gear = gear
                self.gear_eng = gear if gear != 0 else None
                self.neutral_held = gear == 0
            elif gear != self.gear:
                if gear == 0:
                    self.t_neutral = 0.0
                    self.neutral_held = False
                    self.gear_eng = self.gear
                else:
                    was = self.gear_eng if self.gear == 0 else self.gear
                    if self.gear == 0 and self.neutral_held:
                        events.append(('clunk', 1.15 if gear < 0 else 1.0))     # neutral-in
                    elif gear < 0 or (was is not None and was < 0):
                        events.append(('clunk', 1.15))                         # reverse
                    elif was is not None and gear > was > 0:
                        events.append(('clunk', 0.85))                         # upshift
                        events.append(('shift_cut', T_SHIFT_CUT))
                    else:
                        events.append(('clunk', 0.75))                         # downshift
                    self.gear_eng = gear
                    self.neutral_held = False
                self.gear = gear
            elif gear == 0 and not self.neutral_held:
                self.t_neutral += dt
                if self.t_neutral >= T_GATE_MAX:
                    events.append(('clunk', 0.45))                             # into neutral
                    self.neutral_held = True

        # ---- the lift that arms the pops, CONFIRMED.  Every shift the physics
        # makes starts with a lift at the old gear (the declutch), so a lift
        # counts only once the gear has stayed in for T_LIFT_CONFIRM - or, if
        # a shift did follow, once the new gear is in and the load STAYS off
        # for T_ENGAGE_CHECK (the auto box changes up on every lift from high
        # rpm; a shift at WOT is back over 0.3 load within ~2 frames)
        self.load_env = max(load, self.load_env * math.exp(-dt / 0.30))
        if self.lift is None:
            if (load < 0.12 and self.load_env > 0.55 and rpm > P.pop_rpm and not stalled
                    and gear != 0):
                self.lift = [0.0, False, 0.0, rpm]      # since the lift, shifted, since engage, rpm
                self.load_env = load                    # one lift, one window
        else:
            lf = self.lift
            lf[0] += dt
            if gear == 0:
                lf[1], lf[2] = True, 0.0
            elif lf[1]:
                lf[2] += dt
            if load > 0.3 or stalled or lf[0] > T_LIFT_GIVE_UP:
                self.lift = None                        # a shift at load, or the throttle back
            elif (not lf[1] and lf[0] >= T_LIFT_CONFIRM) or (lf[1] and lf[2] >= T_ENGAGE_CHECK):
                # strength from the rpm the throttle closed at (the auto box's
                # lift-off upshift has dropped it since); the window then dies
                # as the rpm falls under pop_rpm - 1200 like any other
                if load < 0.12 and rpm > P.pop_rpm - 1200.0:
                    self.pops = max(self.pops, _clip01(0.5 + (lf[3] - P.pop_rpm) / 2000.0))
                self.lift = None
        self.pops *= math.exp(-dt / POP_TAU)
        if load > 0.3 or rpm < P.pop_rpm - 1200.0:
            self.pops = 0.0
        if self.pops < 0.02:
            self.pops = 0.0

        # ---- edges -> one-shots ---------------------------------------------------------
        fl = str(g(hud, 'sector_flash', '') or '')
        fc = str(g(hud, 'flash_col', '') or '')
        cur = (fl, fc) if fl else None
        if not self.first and cur is not None and cur != self.flash_prev and fc in FLASH_CUES:
            events.append(('cue', 'sector_' + fc))
        self.flash_prev = cur
        self.first = False

        return dict(rpm=rpm, load=load, squeal=self.sq, grass=self.grass,
                    wind=self.wind, limit=limit, stalled=stalled,
                    master=0.0 if paused else 1.0, car=key, V=V, starter=starter,
                    gear=int(gear or 0), lock=self.lock, abs_on=self.abs,
                    scrub=self.scrub, sq_pan=self.pan, grass_pan=self.grass_pan,
                    gravel=self.gravel, gravel_pan=self.gravel_pan, kerb=self.kerb,
                    kerb_pan=self.kerb_pan, roll=self.roll, wet=self.wet,
                    pops=self.pops, servo=self.srv, servo_pan=self.srv_pan,
                    servo_rate=self.srv_rate,
                    clunk=any(e[0] == 'clunk' for e in events), events=events)


# ====================================================================== #
#  THE DEVICE                                                            #
# ====================================================================== #
class CarSound:
    """The device side: a Synth streamed through one pygame.mixer Channel.

    `update(hud)` once per render frame.  `set_volume`, `stop`.  `ok` is
    False (and `error` says why) when there is no usable audio device; the
    caller then simply drops the object.  Stereo when the mixer is (or can be
    opened) two-channel; a mixer someone already opened mono is used mono.
    """

    def __init__(self, volume: float = 0.6, rate: int = RATE,
                 chunk: int = CHUNK, seed: int | None = None,
                 channels: int = CHANNELS):
        self.ok = False
        self.error = ""
        self.volume = float(volume)
        self.underruns = 0
        self.chunks = 0
        self._keep: collections.deque = collections.deque(maxlen=6)
        self._map = _Mapper()
        self._events: list = []
        self.nch = 1
        try:
            import pygame
            self._pg = pygame
            init = pygame.mixer.get_init()
            if init is not None and (abs(int(init[1])) != 16 or int(init[2]) < 1):
                pygame.mixer.quit()             # not 16-bit: reopen it ours
                init = None
            if init is not None and channels == 1 and int(init[2]) != 1:
                pygame.mixer.quit()             # mono asked for explicitly
                init = None
            if init is None:
                pygame.mixer.init(frequency=rate, size=-16, channels=max(1, min(int(channels), 2)),
                                  buffer=BUFFER)
            init = pygame.mixer.get_init()
            if init is None:
                self.error = "mixer did not initialise"
                return
            self.rate, fmt, self.nch = int(init[0]), int(init[1]), int(init[2])
            if abs(fmt) != 16:
                self.error = f"mixer format {fmt} is not 16-bit"
                pygame.mixer.quit()
                return
            self._signed = fmt < 0
            if pygame.mixer.get_num_channels() < 1:
                pygame.mixer.set_num_channels(1)
            self.ch = pygame.mixer.Channel(0)
            self.ch.set_volume(1.0)
        except Exception as exc:          # no device, no SDL audio, ...
            self.error = f"{type(exc).__name__}: {exc}"
            return
        try:
            # never block the frame that builds us: scipy.signal (0.27-0.49 s)
            # and the chimes (~30 ms) come in on warm_up's thread, the synth
            # runs its numpy filters until then and hands over seamlessly
            warm_up(self.rate)
            self.synth = Synth(self.rate, chunk, seed=(seed if seed is not None else 7),
                               channels=2 if self.nch >= 2 else 1, backend='lazy')
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return
        self.synth.volume = self.volume
        self.ok = True

    # -- controls --------------------------------------------------------
    def set_volume(self, volume: float) -> None:
        self.volume = float(volume)

    def chime(self, kind: str) -> bool:
        """Play a chime (`CHIMES`) on its own channel, over the engine, at
        the session's volume. False when there is no device."""
        if not self.ok:
            return False
        try:
            pg = self._pg
            if pg.mixer.get_num_channels() <= CHIME_CHANNEL:
                pg.mixer.set_num_channels(CHIME_CHANNEL + 1)
            y = chime_wave(kind, self.rate) * float(self.volume)
            pcm = (np.clip(y, -1.0, 1.0) * 32767.0).astype(np.int16)
            if self.nch > 1:
                pcm = np.repeat(pcm[:, None], self.nch, axis=1)
            if not self._signed:
                pcm = (pcm.astype(np.int32) + 32768).astype(np.uint16)
            snd = pg.mixer.Sound(buffer=pcm.tobytes())
            self._keep.append(snd)
            pg.mixer.Channel(CHIME_CHANNEL).play(snd)
            return True
        except Exception as exc:          # a chime never takes the sound down
            self.error = f"chime: {type(exc).__name__}: {exc}"
            return False

    def stop(self) -> None:
        if not self.ok:
            return
        try:
            self.ch.stop()
        except Exception:
            pass

    # -- per render frame --------------------------------------------------
    def levels(self, hud, dt: float | None = None) -> dict:
        """HudData -> the synth's targets (+ 'events', the one-shots this
        frame raised, and 'clunk' for the old callers).  Stateful (smoothing
        and edges); `dt` defaults to the wall clock between calls."""
        return self._map(hud, dt)

    def _sound(self, lv: dict):
        s = self.synth
        for ev in self._events:
            s.event(*ev)
        self._events.clear()
        y = s.chunk(volume=self.volume, **lv)
        pcm = (np.clip(y, -1.0, 1.0) * 32767.0).astype(np.int16)
        if self.nch > 2:
            full = np.zeros((len(pcm), self.nch), dtype=np.int16)
            full[:, :2] = pcm
            pcm = full
        if not self._signed:
            pcm = (pcm.astype(np.int32) + 32768).astype(np.uint16)
        snd = self._pg.mixer.Sound(buffer=pcm.tobytes())
        self._keep.append(snd)
        self.chunks += 1
        return snd

    def update(self, hud) -> None:
        if not self.ok:
            return
        try:
            lv = self.levels(hud)
            self._events.extend(lv.pop('events', ()))   # latched to the next chunk
            lv.pop('clunk', None)
            ch = self.ch
            if not ch.get_busy():
                if self.chunks:
                    self.underruns += 1
                ch.play(self._sound(lv))
            if ch.get_queue() is None:
                ch.queue(self._sound(lv))
        except Exception as exc:          # a device that went away mid-run
            self.error = f"{type(exc).__name__}: {exc}"
            self.ok = False


# ====================================================================== #
#  SELF-CHECK                                                            #
# ====================================================================== #
IDLE_LOAD = 0.24        # what drive.Sim publishes idling in gear (measured 0.238
#                         at 586 rpm on the dragstrip, auto box, Corsa)
IDLE_LOAD_N = 0.08      # ... and free in neutral, at n_idle
# the physics' shift, as the HUD shows it (powertrain.PowertrainParams:
# t_declutch 0.15, t_gate 0.25, t_engage 0.30 s): the self-check and the demo
# drive the mapper through exactly this, not a gear number that jumps
T_DECLUTCH, T_GATE, T_ENGAGE = 0.15, 0.25, 0.30


def _hud(**kw):
    """A HudData stand-in with the fields this module reads."""
    from types import SimpleNamespace
    d = dict(rpm=850.0, eng_load=IDLE_LOAD, V=0.0, stalled=False, on_limiter=False,
             on_track=True, paused=False, util_f=0.2, util_r=0.2,
             kappa=[0.0] * 4, alpha=[0.0] * 4, Fx=[0.0] * 4, Fy=[0.0] * 4,
             gear=0, abs_active=False, surf4=None, wing_deploy=0.0, wing_side=0,
             wing_deploy_l=0.0, wing_deploy_r=0.0, top_deploy=0.0,
             sector_flash='', flash_col='', lap_rank='', lap_medal='',
             car_key='corsa', car_name='')
    d.update(kw)
    return SimpleNamespace(**d)


def _smoothstep(x: float) -> float:
    x = _clip01(x)
    return x * x * (3.0 - 2.0 * x)


def _shift_frames(g0: int, g1: int, rpm0: float, rpm1: float, fps: int = 60,
                  load_after: float = 1.0, blip: bool = False) -> list:
    """(gear, rpm, load) per frame of one shift as drive.Sim makes it:
    the declutch (load 0 at the OLD gear, the free engine sagging ~900
    rpm/s), the gate (gear 0; a downshift's rev-match blip lifts the rpm
    toward the new gear's), the engage (the new gear, the load ramping back
    to `load_after`, the rpm pulled to the new gear's over ~0.25 s).
    Measured on the dragstrip, Corsa 1 -> 2: 6047 / 5966 / 5678 / 4472 rpm
    at the lift / neutral / engage / +0.43 s."""
    out = []
    rpm = rpm0
    for _ in range(int(round(T_DECLUTCH * fps))):
        rpm -= 900.0 / fps
        out.append((g0, rpm, 0.0))
    for i in range(int(round(T_GATE * fps))):
        if blip:
            rpm += (rpm1 - rpm) * min(1.0, 6.0 / fps)
            out.append((0, rpm, 0.5 if i < 0.6 * T_GATE * fps else 0.0))
        else:
            rpm -= 900.0 / fps
            out.append((0, rpm, 0.0))
    ne = int(round(T_ENGAGE * fps))
    for i in range(ne):
        rpm += (rpm1 - rpm) * min(1.0, 4.0 / fps * 4.0)
        out.append((g1, rpm, load_after * (i + 1) / ne))
    return out


def _lap_script(key: str, fps: int = 60) -> list:
    """A lap-like drive as a list of per-frame HUDs: idle in neutral, 1st in
    (the real in-gear idle and its creep load), a launch with wheelspin, WOT
    through three gears (each shift the physics' declutch -> neutral gate ->
    engage), a hard stop from speed (a lift with pops, a lock-up, ABS, a
    downshift with its blip), a corner (scrub, a slide to the right, the left
    kerb, the right flank panel out and back), the purple sector chime, 3rd to
    the limiter, a lift, an excursion over grass and gravel, a wet stretch,
    the lap's end (green sector, P1, gold), a stop, neutral, reverse, a
    stall, the starter and a catch."""
    P = PROFILES[key]
    sc = P.cut / 6200.0
    # the script's rpm MARGINS (shift 250 under the cut, the limiter's 30 / 60,
    # the 600 / 400 downshift and cruise room) are a petrol's; task 41: a
    # diesel cut at 2500 takes them scaled with its cut. Exactly 1.0 on every
    # engine cut at 6000 or above (the four petrols), so their demos are as were
    ro = min(1.0, P.cut / 6000.0)
    kg = {1: 484.0 * sc, 2: 277.0 * sc, 3: 183.0 * sc}   # rpm per m/s (Corsa ratios)
    acc = {1: 6.5, 2: 4.3, 3: 3.0}
    dt = 1.0 / fps
    driven = (0, 1) if key in ('corsa', 'express') else (2, 3)    # FWD / RWD
    frames = []
    st = dict(V=0.0, gear=0, rpm=P.idle, t=0.0)

    def emit(**kw):
        frames.append(_hud(car_key=key, rpm=st['rpm'], V=st['V'], gear=st['gear'], **kw))
        st['t'] += dt

    def hold(sec, **kw):
        for _ in range(int(round(sec * fps))):
            emit(**kw)

    def shift(g1, load_after=1.0, blip=False, dv=0.0, **kw):
        """One shift at the current speed (changing by dv per second)."""
        for g, rpm, ld in _shift_frames(st['gear'], g1, st['rpm'], kg[g1] * st['V'],
                                        fps, load_after, blip):
            st['gear'], st['rpm'] = g, rpm
            st['V'] = max(st['V'] + dv * dt, 0.0)
            emit(eng_load=ld, **kw)

    # idle in neutral, then 1st: the real in-gear idle (idle_gear, creep load)
    hold(1.0, eng_load=IDLE_LOAD_N)
    st['gear'] = 1
    for i in range(int(0.8 * fps)):
        st['rpm'] = P.idle_gear + (P.idle - P.idle_gear) * math.exp(-i / (0.15 * fps))
        emit(eng_load=IDLE_LOAD)
    # launch and WOT through the gears to ~30 m/s
    t0 = st['t']
    while st['V'] < 30.0 and st['t'] < t0 + 10.0:
        g = st['gear']
        el = st['t'] - t0
        st['V'] += acc[g] * dt
        rpm_w = kg[g] * st['V']
        st['rpm'] = max(rpm_w, 3000.0 * sc) if (g == 1 and rpm_w < 3000.0 * sc) else \
            (rpm_w if st['rpm'] < rpm_w + 50.0 else st['rpm'] + (rpm_w - st['rpm']) * 0.3)
        kap = [0.0] * 4
        if el < 0.5:
            for i in driven:
                kap[i] = 0.18
        emit(eng_load=1.0, kappa=kap, util_f=0.5, util_r=0.4, surf4=('tarmac',) * 4)
        if st['rpm'] >= P.cut - 250.0 * ro and g < 3:
            shift(g + 1, util_f=0.3, util_r=0.3, surf4=('tarmac',) * 4)
    # a hard stop from speed: lift (pops), lock-up, ABS, 3 -> 2 with a blip
    for i in range(int(1.0 * fps)):
        el = i * dt
        st['V'] = max(st['V'] - 9.0 * dt, 14.0)
        kap = [0.0] * 4
        abs_on = False
        if 0.30 <= el < 0.65:
            kap = [-0.20, -0.20, -0.06, -0.06]
        elif 0.65 <= el:
            k = -0.11 + 0.03 * math.sin(2 * math.pi * 15.0 * el)
            kap = [k, k, -0.05, -0.05]
            abs_on = True
        st['rpm'] = kg[st['gear']] * st['V']
        emit(eng_load=0.0, kappa=kap, abs_active=abs_on, util_f=0.8, util_r=0.6,
             surf4=('tarmac',) * 4)
    shift(2, load_after=0.0, blip=True, dv=-9.0, abs_active=True, util_f=0.8, util_r=0.6,
          kappa=[-0.10, -0.10, -0.05, -0.05], surf4=('tarmac',) * 4)
    for i in range(int(0.3 * fps)):
        st['V'] = max(st['V'] - 9.0 * dt, 14.0)
        st['rpm'] = kg[2] * st['V']
        emit(eng_load=0.0, util_f=0.7, util_r=0.6, surf4=('tarmac',) * 4)
    # the corner at ~16 m/s: scrub, a slide (right-hand wheels), the left kerb,
    # the right flank panel out (0.45 s) and back (0.30 s), the purple chime
    st['V'] = 16.0
    for i in range(int(3.4 * fps)):
        el = i * dt
        st['rpm'] = kg[2] * st['V']
        util = 0.70 + 0.22 * _clip01(el / 1.0) + 0.14 * _clip01((el - 1.0) / 0.4) \
            - 0.30 * _clip01((el - 2.4) / 0.5)
        sl = _clip01((util - 0.9) / 0.1)
        alpha = [0.05 + 0.04 * sl, 0.07 + 0.08 * sl, 0.05 + 0.05 * sl, 0.08 + 0.10 * sl]
        Fz = [2600.0, 4300.0, 1900.0, 3400.0]
        Fy = [0.9 * f for f in Fz]
        surf = ('kerb', 'tarmac', 'kerb', 'tarmac') if 1.6 <= el < 2.3 else ('tarmac',) * 4
        dr = _smoothstep((el - 0.3) / 0.45) if el < 2.6 else _smoothstep(1.0 - (el - 2.6) / 0.30)
        flash = ('S1  21.090  -0.123', 'purple') if el >= 2.8 else ('', '')
        emit(eng_load=0.35, util_f=util, util_r=util - 0.02, alpha=alpha, Fy=Fy,
             surf4=surf, wing_deploy_l=0.0, wing_deploy_r=dr,
             sector_flash=flash[0], flash_col=flash[1])
    # exit: WOT 2nd -> 3rd -> the limiter in 3rd for 1.2 s
    fl = dict(sector_flash='S1  21.090  -0.123', flash_col='purple')
    t_lim = None
    while True:
        g = st['gear']
        rpm_w = kg[g] * st['V']
        on_lim = g == 3 and rpm_w >= P.cut - 30.0 * ro
        if on_lim:
            t_lim = st['t'] if t_lim is None else t_lim
            rpm_w = P.cut - 60.0 * ro + 60.0 * ro * math.sin(2 * math.pi * 12.0 * st['t'])
        else:
            st['V'] += acc[g] * dt
        st['rpm'] = rpm_w if st['rpm'] < rpm_w + 50.0 else st['rpm'] + (rpm_w - st['rpm']) * 0.3
        # the purple flash stays up until the limiter (the sim shows it ~2 s),
        # so the lap's green one later is a fresh edge
        emit(eng_load=1.0, on_limiter=on_lim, surf4=('tarmac',) * 4, util_f=0.3,
             util_r=0.3, **(fl if t_lim is None else {}))
        if rpm_w >= P.cut - 250.0 * ro and g < 3:
            shift(g + 1, surf4=('tarmac',) * 4, **fl)
        if t_lim is not None and st['t'] > t_lim + 1.2:
            break
    # a genuine lift from the limiter (pops), brake, 3 -> 2 once 2nd can take it
    for i in range(int(2.0 * fps)):
        st['V'] -= (1.5 if i < 0.8 * fps else 12.0) * dt
        if st['gear'] == 3 and kg[2] * st['V'] < P.cut - 600.0 * ro:
            shift(2, load_after=0.0, blip=True, dv=-8.0, surf4=('tarmac',) * 4,
                  util_f=0.6, util_r=0.5)
        st['rpm'] = kg[st['gear']] * st['V']
        emit(eng_load=0.0, surf4=('tarmac',) * 4, util_f=0.6, util_r=0.5)
    # off: two wheels then four on the grass, then gravel, back on
    for i in range(int(1.2 * fps)):
        st['V'] = max(st['V'] - 5.0 * dt, 8.0)
        st['rpm'] = kg[st['gear']] * st['V']
        surf = ('tarmac', 'grass', 'tarmac', 'grass') if i < 0.4 * fps else ('grass',) * 4
        emit(eng_load=0.1, surf4=surf, on_track=i < 0.4 * fps)
    for i in range(int(1.0 * fps)):
        st['V'] = max(st['V'] - 6.0 * dt, 7.0)
        st['rpm'] = kg[st['gear']] * st['V']
        emit(eng_load=0.1, surf4=('gravel',) * 4, on_track=False)
    for i in range(int(0.8 * fps)):
        st['V'] += 4.0 * dt
        st['rpm'] = kg[2] * st['V']
        emit(eng_load=1.0, surf4=('tarmac',) * 4)
    # a wet stretch
    for i in range(int(1.5 * fps)):
        st['V'] = min(st['V'] + 3.0 * dt, (P.cut - 400.0 * ro) / kg[2])
        st['rpm'] = kg[2] * st['V']
        emit(eng_load=0.9, surf4=('wet',) * 4)
    # the lap's end: green sector, P1, gold -- all on one frame, as the sim
    # does -- then a stop: 2 -> 1 on the way down, the in-gear idle at rest
    end = dict(sector_flash='S3  20.440  -0.080', flash_col='green', lap_rank='P1',
               lap_medal='gold')
    for i in range(int(2.6 * fps)):
        st['V'] = max(st['V'] - 7.0 * dt, 0.0)
        if st['V'] < 12.0 and st['gear'] == 2:
            shift(1, load_after=0.0, blip=True, dv=-7.0, surf4=('tarmac',) * 4, **end)
        rpm_w = kg[st['gear']] * st['V']
        st['rpm'] = max(rpm_w, P.idle_gear)
        emit(eng_load=0.0 if rpm_w > P.idle_gear else IDLE_LOAD, surf4=('tarmac',) * 4, **end)
    st['V'], st['rpm'] = 0.0, P.idle_gear
    hold(0.5, eng_load=IDLE_LOAD)
    # neutral (it holds: its own clunk), reverse
    for g, rpm, _ in _shift_frames(st['gear'], 0, st['rpm'], P.idle, fps, 0.0):
        st['gear'], st['rpm'] = g, max(rpm, P.idle_gear)
        emit(eng_load=IDLE_LOAD_N)
    st['rpm'] = P.idle
    hold(0.6, eng_load=IDLE_LOAD_N)
    st['gear'] = -1
    hold(0.3, eng_load=IDLE_LOAD)
    for i in range(int(2.0 * fps)):
        el = i * dt
        st['V'] = 2.5 * _clip01(el / 0.8) * _clip01((2.0 - el) / 0.4)
        st['rpm'] = P.idle_gear + 380.0 * st['V']
        emit(eng_load=0.3 if el < 1.6 else 0.05, surf4=('tarmac',) * 4)
    st['V'], st['gear'], st['rpm'] = 0.0, 0, P.idle
    hold(0.6, eng_load=IDLE_LOAD_N)
    # a stall, silence, the starter, a catch
    for i in range(int(0.3 * fps)):
        st['rpm'] = P.idle * (1.0 - i / (0.3 * fps))
        emit(eng_load=0.0, stalled=st['rpm'] < 450.0)
    st['rpm'] = 0.0
    hold(1.0, eng_load=0.0, stalled=True)
    st['rpm'] = 250.0
    hold(1.2, eng_load=0.0, stalled=True)
    for i in range(int(0.3 * fps)):
        st['rpm'] = 250.0 + (1400.0 - 250.0) * i / (0.3 * fps)
        emit(eng_load=0.4, stalled=st['rpm'] < 450.0)
    for i in range(int(0.8 * fps)):
        st['rpm'] = P.idle + (1400.0 - P.idle) * math.exp(-i / (0.25 * fps))
        emit(eng_load=0.12)
    st['rpm'] = P.idle
    hold(1.5, eng_load=IDLE_LOAD_N)
    return frames


def _session(frames, key: str, channels: int = 2, seed: int = 11, fps: int = 60,
             synth: Synth | None = None):
    """Frames -> audio exactly as CarSound streams it: the mapper every
    frame, a chunk whenever the stream is less than one chunk ahead, the
    frame's one-shots latched to the next chunk.  Returns (y, events)."""
    s = synth if synth is not None else Synth(RATE, CHUNK, seed=seed, channels=channels, car=key)
    mp = _Mapper()
    dt = 1.0 / fps
    out, pend, log = [], [], []
    t_audio = 0.0
    for i, hud in enumerate(frames):
        lv = mp(hud, dt)
        pend.extend(lv.pop('events'))
        lv.pop('clunk', None)
        while t_audio < i * dt + CHUNK / RATE:
            for ev in pend:
                s.event(*ev)
                log.append((i * dt, ev))
            pend.clear()
            out.append(s.chunk(**lv))
            t_audio += CHUNK / RATE
    return np.concatenate(out), log


def _render(synth: Synth, seconds: float, fn) -> np.ndarray:
    out = []
    n_chunks = int(round(seconds * synth.rate / synth.n))
    for i in range(n_chunks):
        out.append(synth.chunk(**fn(i * synth.n / synth.rate)))
    return np.concatenate(out)


def _spec(y: np.ndarray, rate: int):
    y = y if y.ndim == 1 else y.mean(axis=1)
    sp = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    return np.fft.rfftfreq(len(y), 1.0 / rate), sp


def _peak_hz(y: np.ndarray, rate: int, f_lo: float, f_hi: float) -> float:
    f, sp = _spec(y, rate)
    m = (f >= f_lo) & (f <= f_hi)
    return float(f[m][np.argmax(sp[m])])


def _band_energy(y: np.ndarray, rate: int, lo: float, hi: float) -> float:
    y = y if y.ndim == 1 else y.mean(axis=1)
    spec = np.abs(np.fft.rfft(y)) ** 2
    freqs = np.fft.rfftfreq(len(y), 1.0 / rate)
    return float(spec[(freqs >= lo) & (freqs <= hi)].sum())


def _centroid(y: np.ndarray, rate: int, lo: float = 20.0, hi: float = 8000.0) -> float:
    f, sp = _spec(y, rate)
    m = (f >= lo) & (f <= hi)
    return float((sp[m] * f[m]).sum() / max(sp[m].sum(), 1e-30))


def _lines(y: np.ndarray, rate: int, f0: float, k_max: int, width: float = 3.0):
    """Energy at each multiple k f0 (k = 1..k_max), +-width Hz."""
    f, sp = _spec(y, rate)
    return np.array([sp[(f >= k * f0 - width) & (f <= k * f0 + width)].sum()
                     for k in range(1, k_max + 1)])


def _rms(y) -> float:
    return float(np.sqrt(np.mean(np.square(y))))


def _write_wav(path: str, y: np.ndarray, rate: int) -> None:
    import wave
    y = np.asarray(y)
    nch = 1 if y.ndim == 1 else y.shape[1]
    pcm = (np.clip(y, -1.0, 1.0) * 32767.0).astype(np.int16)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(nch)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def _hf_boundary(y: np.ndarray, rate: int = RATE, n: int = CHUNK, fc: float = 9000.0) -> float:
    """Energy above fc AT the chunk boundaries against inside them (p99.99):
    a step in a level, a band or an AM phase there is a click (review)."""
    yy = y if y.ndim == 2 else y[:, None]
    Y = np.fft.rfft(yy, axis=0)
    Y[np.fft.rfftfreq(len(yy), 1.0 / rate) < fc] = 0.0
    h = np.abs(np.fft.irfft(Y, len(yy), axis=0)).max(axis=1)
    near = np.zeros(len(h), dtype=bool)
    for i in range(n, len(h), n):
        near[max(0, i - 3):i + 3] = True
    return float(h[near].max() / max(float(np.percentile(h[~near], 99.99)), 1e-12))


def _nonharm(y: np.ndarray, rate: int, rpm: float) -> tuple:
    """(share of the energy off the cycle harmonics +-3 Hz, share of the one
    strongest line): 96-99.8 % and up to 90 % before (review: an organ)."""
    f, sp = _spec(y, rate)
    m = f >= 20.0
    tot = sp[m].sum()
    fc = rpm / 120.0
    k = np.round(f / fc)
    harm = (np.abs(f - k * fc) <= 3.0) & (k >= 1) & m
    i = int(np.argmax(sp * m))
    return 1.0 - sp[harm].sum() / tot, sp[np.abs(f - f[i]) <= 3.0].sum() / tot


def self_check(verbose: bool = True, wav_path: str | None = None) -> bool:
    """Every check below, printed when `verbose`; True when all pass.  The
    demos go next to `wav_path` as audio_demo_<car>.wav, ONE per car,
    overwritten (the file `wav_path` itself is no longer written)."""
    results = []

    def chk(name, ok, got):
        results.append((name, bool(ok), got))
        if verbose:
            print(f"  {'PASS' if ok else 'FAIL'}  {name:<50s} {got}")
        return ok

    rate = RATE
    T = lambda **kw: (lambda t: kw)                                  # noqa: E731
    quiet = dict(rpm=0.0, load=0.0)            # tyre / road layers alone

    def frames_rms(y, ms=10.0):
        w = int(rate * ms / 1000.0)
        m = len(y) // w
        return np.sqrt(np.mean(y[:m * w].reshape(m, w) ** 2, axis=1))

    def lowpass(y, fc):
        Y = np.fft.rfft(y)
        Y[np.fft.rfftfreq(len(y), 1.0 / rate) > fc] = 0.0
        return np.fft.irfft(Y, len(y))

    # 0. the engine by the car's shown name alone (no car_key): each title
    #    (cars.CAR_TITLES) and full name (CarSpec.name) voices its own car
    import cars as _cars
    from types import SimpleNamespace as _NS
    nm_miss = [f"{k}: {nm!r}" for k in _cars.CAR_ORDER
               for nm in (_cars.CAR_TITLES[k], _cars.get(k).name)
               if profile_key(_NS(car_key='', car_name=nm)) != k]
    chk("engine by the car's name alone", not nm_miss,
        "; ".join(nm_miss) or f"all {len(_cars.CAR_ORDER)} titles and names")

    # 1. continuity across chunk boundaries: a fast sweep with every layer
    for ch_, car in ((1, 'corsa'), (2, '540i')):
        s = Synth(rate, CHUNK, seed=3, channels=ch_, car=car)
        y = _render(s, 3.0, lambda t: dict(rpm=900.0 + 1700.0 * t, load=0.5 + 0.5 * math.sin(3 * t),
                                           squeal=0.5, wind=0.3, grass=0.2, V=15.0 + 3 * t,
                                           kerb=0.5, kerb_pan=-0.5, roll=0.5, scrub=0.3,
                                           servo=0.5, sq_pan=0.4 * math.sin(2 * t)))
        yy = y if y.ndim == 2 else y[:, None]
        d = np.abs(np.diff(yy, axis=0)).max(axis=1)
        idx = np.arange(CHUNK, len(yy), CHUNK) - 1
        step_b = float(d[idx].max())
        mask = np.ones(len(d), dtype=bool)
        mask[idx] = False
        step_in = float(d[mask].max())
        chk(f"no click at chunk boundaries ({'stereo' if ch_ == 2 else 'mono'})",
            step_b <= 1.3 * step_in and np.all(np.isfinite(y)),
            f"boundary step {step_b:.3f} vs {step_in:.3f} inside")
    chk("levels inside [-1, 1]", float(np.abs(y).max()) <= 1.0, f"peak {np.abs(y).max():.3f}")
    # ... and the layers whose parameters used to STEP there: the squeal's
    # lock-up depth / band centre / ABS gate, the servo's AM (review: x2.8, x8.4)
    tog = []
    for name, fn in (('squeal', lambda i: dict(squeal=float(i % 2), lock=float((i // 2) % 2),
                                               abs_on=float((i // 3) % 2),
                                               sq_pan=(-0.65 if i % 3 else 0.65), V=20.0, **quiet)),
                     ('servo', lambda i: dict(servo=float(i % 2), servo_rate=float((i // 2) % 2),
                                              servo_pan=0.6 * (-1) ** i, **quiet))):
        s = Synth(rate, CHUNK, seed=3, channels=2)
        tog.append((name, _hf_boundary(np.concatenate([s.chunk(**fn(i)) for i in range(60)]), rate)))
    chk("squeal / servo toggled every chunk: no boundary click",
        all(v < 1.5 for _, v in tog),
        " / ".join(f"{k} x{v:.1f}" for k, v in tog) + " > 9 kHz at the boundaries vs inside")

    # 2. the firing orders: a four at 3000 rpm fires at 100 Hz; the strongest
    # line under 1 kHz is a firing harmonic, and the firing harmonics carry
    # more than the half-orders (cycle lines at 25 Hz) between them
    s = Synth(rate, CHUNK, seed=5)
    _render(s, 0.5, T(rpm=3000.0, load=1.0))
    y = _render(s, 2.0, T(rpm=3000.0, load=1.0))
    f0 = 3000.0 / 60.0 * FIRINGS_PER_REV
    pk = _peak_hz(y, rate, 40.0, 1000.0)
    k = pk / f0
    L = _lines(y, rate, f0 / 4.0, 40, 1.0)
    fire, half = L[3::4].sum(), L.sum() - L[3::4].sum()
    chk("engine: strongest line is a firing harmonic",
        abs(k - round(k)) * f0 <= 3.0 and 0.9 <= k <= 10.5, f"peak {pk:.1f} Hz = {k:.2f} x {f0:.0f} Hz")
    chk("engine: firing orders dominate the half-orders", fire > 3.0 * half,
        f"firing / half-order energy {fire / max(half, 1e-30):.1f}")
    s = Synth(rate, CHUNK, seed=5)
    _render(s, 0.5, T(rpm=3000.0, load=0.0))
    y0 = _render(s, 2.0, T(rpm=3000.0, load=0.0))
    chk("load makes it louder", _rms(y) > 2.0 * _rms(y0),
        f"rms WOT {_rms(y):.3f} vs overrun {_rms(y0):.3f}")
    c1, c0 = _centroid(y, rate), _centroid(y0, rate)
    chk("... and brighter", c1 > 1.3 * c0, f"centroid WOT {c1:.0f} Hz vs overrun {c0:.0f} Hz")

    # 3. the engines at WOT: bright to deep in order at every rpm, a car and
    # not an organ (5-15 % of the energy off the harmonics, no single line
    # carrying the note), the V8's burble, the rally BDA's crank order
    RPMS = (1500.0, 3000.0, 4500.0, 6000.0)      # the petrols' (EngineProfile.rpms)
    cents, nh, top, halfs, crank = {}, {}, {}, {}, {}
    for car, P in PROFILES.items():
        for rpm in P.rpms:                   # task 41: a diesel sweeps its own range
            s = Synth(rate, CHUNK, seed=6, car=car)
            _render(s, 0.4, T(rpm=rpm, load=1.0))
            y = _render(s, 1.2 if rpm != P.rpms[1] else 2.0, T(rpm=rpm, load=1.0))
            cents[car, rpm] = _centroid(y, rate)
            nh[car, rpm], top[car, rpm] = _nonharm(y, rate, rpm)
            if rpm == P.rpms[1]:
                L = _lines(y, rate, rpm / 120.0, 48, 1.0)       # cycle-rate lines, +-1 Hz
                firing = L[P.slots - 1::P.slots].sum()
                halfs[car] = (L.sum() - firing) / max(firing, 1e-30)
                crank[car] = L[1::4].sum() / max(firing, 1e-30)   # odd x the crank rate
    #  task 46: the rally BDA the brightest from 3000 rpm up (a race four's
    #  short pipe and open trumpets), the Corsa's small four next, the V8
    #  deepest -- each step at least 5 % at every rpm from 3000
    order = all(cents['rally', r] > 1.05 * cents['corsa', r] > 1.05 ** 2 * cents['540i', r]
                for r in RPMS[1:]) and cents['corsa', RPMS[0]] > 1.1 * cents['540i', RPMS[0]]
    chk("the engines, bright to deep at every rpm (WOT)", order,
        "centroid rally / corsa / 540i " + ", ".join(
            f"{r / 1000:.1f}k {cents['rally', r]:.0f}/{cents['corsa', r]:.0f}/{cents['540i', r]:.0f}"
            for r in RPMS) + " Hz")
    # task 41: the Express's E7J a plainer, lower four than the Corsa's (under
    # it and over the V8 at every rpm)
    chk("the van a lower four than the Corsa",
        all(cents['540i', r] < cents['express', r] < cents['corsa', r] for r in RPMS),
        "centroid express " + "/".join(f"{cents['express', r]:.0f}" for r in RPMS)
        + " Hz (corsa " + "/".join(f"{cents['corsa', r]:.0f}" for r in RPMS) + ")")
    nh_ok = all(0.04 <= nh[c, r] <= 0.20 for c, P in PROFILES.items() for r in P.rpms[1:])
    top_ok = all(top[c, P.rpms[-1]] < 0.5 for c, P in PROFILES.items())
    chk("a car, not an organ: 4-20 % off the harmonics", nh_ok and top_ok,
        "non-harmonic " + ", ".join(f"{c} " + "/".join(f"{100 * nh[c, r]:.0f}" for r in P.rpms)
                                    for c, P in PROFILES.items())
        + " %; top line at the sweep's top (6k) "
        + "/".join(f"{top[c, P.rpms[-1]]:.2f}" for c, P in PROFILES.items()))
    # ... nor the overrun, which plays through the ~0.4 s zero-load declutch +
    # neutral of every real upshift (97-99 % on the harmonics before JIT_OVR)
    ovr = {}
    for car, P in PROFILES.items():
        for rpm in P.rpms[2:]:
            s = Synth(rate, CHUNK, seed=6, car=car)
            _render(s, 0.4, T(rpm=rpm, load=0.0))
            ovr[car, rpm] = _nonharm(_render(s, 1.2, T(rpm=rpm, load=0.0)), rate, rpm)[0]
    chk("overrun: not a pure tone, >= 3 % off the harmonics", all(v >= 0.03 for v in ovr.values()),
        "load 0, 4.5k / 6k: " + ", ".join(
            f"{c} " + "/".join(f"{100 * ovr[c, r]:.0f}" for r in P.rpms[2:])
            for c, P in PROFILES.items()) + " % (was 1-3 %)")
    chk("V8 cross-plane: half-order content (the burble)",
        halfs['540i'] > 3.0 * halfs['corsa'] and halfs['540i'] > 0.15,
        "non-firing / firing lines " + " / ".join(f"{k} {v:.2f}" for k, v in halfs.items()))
    chk("rally BDA 4-2-1: crank order under the firing tone",
        crank['rally'] > 0.02 and crank['rally'] > 5.0 * crank['corsa'],
        f"crank-order / firing lines rally {crank['rally']:.3f} (floor 0.02) vs corsa "
        f"{crank['corsa']:.3f}")
    P8 = PROFILES['540i']
    s = Synth(rate, CHUNK, seed=6, channels=2, car='540i')
    _render(s, 0.3, T(rpm=P8.idle_gear, load=IDLE_LOAD))
    y = _render(s, 1.5, T(rpm=P8.idle_gear, load=IDLE_LOAD))
    corr = float(np.corrcoef(y[:, 0], y[:, 1])[0, 1])
    s = Synth(rate, CHUNK, seed=6, channels=2, car='540i')
    _render(s, 0.4, T(rpm=3000.0, load=1.0))
    y = _render(s, 2.0, T(rpm=3000.0, load=1.0))
    brb = []
    for sig in (y[:, 0], y[:, 1], y.sum(axis=1)):     # L, R, the mono fold L+R
        L = _lines(sig, rate, 25.0, 48, 1.0)
        brb.append((L.sum() - L[7::8].sum()) / max(L[7::8].sum(), 1e-30))
    bal = 20.0 * math.log10(_rms(y[:, 0]) / max(_rms(y[:, 1]), 1e-30))
    chk("V8 stereo: a bank per side, burble in L, R and L+R",
        corr < 0.9 and min(brb) > 0.15 and abs(bal) < 3.0,
        f"burble L {brb[0]:.2f} / R {brb[1]:.2f} (was 0.02) / L+R {brb[2]:.2f} at 3000 WOT; "
        f"L/R correlation {corr:.2f} at idle; L/R {bal:+.1f} dB")
    lv_top = {}
    for car, P in PROFILES.items():
        db_ = []
        for fr in (0.5, 0.65, 0.8, 0.9, 0.98):
            s = Synth(rate, CHUNK, seed=6, car=car)
            _render(s, 0.3, T(rpm=fr * P.cut, load=1.0))
            db_.append(20.0 * math.log10(_rms(_render(s, 0.5, T(rpm=fr * P.cut, load=1.0)))))
        lv_top[car] = db_[-1] - max(db_)
    chk("WOT level holds up to the cut (no resonance dip)", all(v > -2.0 for v in lv_top.values()),
        "at 0.98 x cut vs the loudest of 0.5-0.98: " + " / ".join(f"{k} {v:+.1f}" for k, v in lv_top.items())
        + " dB")

    # 4. the sim's REAL in-gear idle (551-586 rpm, load 0.24): not dead air
    # between firings (54 % of 1 ms frames 40 dB under the peaks, review), and
    # lumpy: the pulse heights (under 4 x the firing rate) vary >= 10 %
    dead, lope = {}, {}
    for car, P in PROFILES.items():
        s = Synth(rate, CHUNK, seed=8, car=car)
        _render(s, 0.5, T(rpm=P.idle_gear, load=IDLE_LOAD))
        y = _render(s, 3.0, T(rpm=P.idle_gear, load=IDLE_LOAD))
        fr = frames_rms(y, 1.0)
        dead[car] = float((fr < 0.01 * np.percentile(fr, 99)).mean())
        ff = P.idle_gear / 120.0 * P.slots
        per = int(rate / ff)
        yl = lowpass(y, 4.0 * ff)
        m = len(yl) // per
        pk = np.abs(yl[:m * per]).reshape(m, per).max(axis=1)
        lope[car] = float(pk.std() / pk.mean())
    chk("the in-gear idle: sound between firings, lumpy",
        all(v < 0.02 for v in dead.values()) and all(v >= 0.10 for v in lope.values()),
        "1 ms frames 40 dB under the peaks " + " / ".join(f"{100 * v:.0f}" for v in dead.values())
        + " %; pulse-height CV " + " / ".join(f"{v:.2f}" for v in lope.values()))

    # 5. the limiter stutters; an upshift cuts the ignition
    s = Synth(rate, CHUNK, seed=4)
    _render(s, 0.3, T(rpm=6150.0, load=1.0))
    fr_off = frames_rms(_render(s, 1.0, T(rpm=6150.0, load=1.0)))
    s = Synth(rate, CHUNK, seed=4)
    _render(s, 0.3, T(rpm=6150.0, load=1.0, limit=1.0))
    fr_on = frames_rms(_render(s, 1.0, T(rpm=6150.0, load=1.0, limit=1.0)))
    q_on = np.percentile(fr_on, 10) / np.percentile(fr_on, 90)
    q_off = np.percentile(fr_off, 10) / np.percentile(fr_off, 90)
    chk("rev limiter: an ignition-cut stutter", q_on < 0.5 * q_off,
        f"10 ms RMS p10/p90 {q_on:.2f} on the limiter vs {q_off:.2f} off it")
    s = Synth(rate, CHUNK, seed=4)
    _render(s, 0.5, T(rpm=4000.0, load=1.0))
    s.shift_cut(T_SHIFT_CUT)
    y = _render(s, 0.2, T(rpm=4000.0, load=1.0))
    a_cut = _rms(y[int(0.012 * rate):int(0.060 * rate)])
    a_run = _rms(y[int(0.110 * rate):int(0.180 * rate)])
    chk("upshift: ignition-cut blip", a_cut < 0.45 * a_run,
        f"rms {a_cut:.3f} in the cut vs {a_run:.3f} after")

    # 6. pops: a per-lift BUDGET (Corsa 0-2, rally 4-8, V8 2-4), spread over the
    # window, each no louder than the engine at WOT at that rpm; the crack
    counts, loud = {}, {}
    for car, P in PROFILES.items():
        rpm = 0.8 * P.cut
        s = Synth(rate, CHUNK, seed=12, car=car)
        _render(s, 0.3, T(rpm=rpm, load=1.0))
        pk_wot = float(np.abs(_render(s, 1.0, T(rpm=rpm, load=1.0))).max())
        cs, pk_pop = [], 0.0
        for seed in (12, 13, 14):
            s = Synth(rate, CHUNK, seed=seed, car=car)
            _render(s, 0.3, T(rpm=rpm, load=0.0))
            n0 = s.pops_fired
            y = _render(s, 2.5, lambda t: dict(rpm=rpm, load=0.0, pops=math.exp(-t / POP_TAU)))
            cs.append(s.pops_fired - n0)
            pk_pop = max(pk_pop, float(np.abs(y).max()))
        counts[car] = cs
        loud[car] = pk_pop / pk_wot
    budget = all(PROFILES[c].pops[0] <= n <= PROFILES[c].pops[1] for c in PROFILES for n in counts[c])
    chk("pops: a budget per lift, never over the WOT peak",
        budget and all(v <= 1.0 for v in loud.values()) and sum(counts['rally']) >= 12,
        "pops per lift " + ", ".join(f"{c} {'/'.join(map(str, v))}" for c, v in counts.items())
        + "; loudest vs WOT peak " + " / ".join(f"{v:.2f}" for v in loud.values()))

    def overrun(pops):
        s = Synth(rate, CHUNK, seed=12, car='rally')
        _render(s, 0.3, T(rpm=6700.0, load=0.0))
        return _render(s, 1.5, lambda t: dict(rpm=6700.0, load=0.0,
                                               pops=pops * math.exp(-t / POP_TAU)))
    yp, y0 = overrun(1.0), overrun(0.0)
    e_r = _band_energy(yp, rate, 2000.0, 6000.0) / max(_band_energy(y0, rate, 2000.0, 6000.0), 1e-30)
    chk("overrun pops and crackles", e_r > 2.0 and np.abs(yp).max() > 1.5 * np.abs(y0).max(),
        f"2-6 kHz energy x{e_r:.1f}, peak {np.abs(yp).max():.2f} vs {np.abs(y0).max():.2f}")

    # 7. tyres: squeal band, lock-up lower, ABS at 15 Hz, scrub, the pan
    s = Synth(rate, CHUNK, seed=9)
    _render(s, 0.3, T(rpm=2500.0, load=0.3, squeal=1.0))
    ys = _render(s, 1.0, T(rpm=2500.0, load=0.3, squeal=1.0))
    s = Synth(rate, CHUNK, seed=9)
    _render(s, 0.3, T(rpm=2500.0, load=0.3, squeal=0.0))
    yn = _render(s, 1.0, T(rpm=2500.0, load=0.3, squeal=0.0))
    e1 = _band_energy(ys, rate, 800.0, 1800.0)
    e0 = _band_energy(yn, rate, 800.0, 1800.0)
    chk("tyre squeal fills 0.8-1.8 kHz", e1 > 8.0 * e0, f"band energy x{e1 / max(e0, 1e-9):.0f}")
    s = Synth(rate, CHUNK, seed=9)
    _render(s, 0.3, T(squeal=1.0, V=20.0, **quiet))
    c_sq = _centroid(_render(s, 1.0, T(squeal=1.0, V=20.0, **quiet)), rate, 300.0, 4000.0)
    s = Synth(rate, CHUNK, seed=9)
    _render(s, 0.3, T(squeal=1.0, lock=1.0, V=20.0, **quiet))
    y_lk = _render(s, 1.0, T(squeal=1.0, lock=1.0, V=20.0, **quiet))
    c_lk = _centroid(y_lk, rate, 300.0, 4000.0)
    chk("lock-up: a lower, harsher squeal", c_lk < 0.8 * c_sq,
        f"centroid {c_lk:.0f} Hz locked vs {c_sq:.0f} Hz cornering")
    s = Synth(rate, CHUNK, seed=9)
    _render(s, 0.3, T(squeal=1.0, lock=1.0, abs_on=1.0, V=20.0, **quiet))
    y_abs = _render(s, 2.0, T(squeal=1.0, lock=1.0, abs_on=1.0, V=20.0, **quiet))
    env = frames_rms(y_abs, 5.0)
    f_env = _peak_hz(env - env.mean(), 200, 4.0, 60.0)
    chk("ABS chatter at ~15 Hz", abs(f_env - ABS_HZ) <= 1.0, f"envelope peak {f_env:.1f} Hz")
    mp = _Mapper()
    for _ in range(30):
        lv85 = mp(_hud(util_f=0.87, util_r=0.85, V=20.0), 1 / 60)
    mp = _Mapper()
    for _ in range(30):
        lv97 = mp(_hud(util_f=0.97, util_r=0.95, V=20.0), 1 / 60)
    s = Synth(rate, CHUNK, seed=9)
    _render(s, 0.3, T(scrub=1.0, V=20.0, **quiet))
    c_sc = _centroid(_render(s, 1.0, T(scrub=1.0, V=20.0, **quiet)), rate, 100.0, 4000.0)
    chk("scrub before the limit (util 0.80-0.95), under the squeal",
        lv85['scrub'] > 0.3 and lv85['squeal'] < 1e-3 and lv97['squeal'] > 0.5
        and c_sc < 0.6 * c_sq,
        f"util 0.87: scrub {lv85['scrub']:.2f} squeal {lv85['squeal']:.3f}; util 0.97: squeal "
        f"{lv97['squeal']:.2f}; centroid {c_sc:.0f} Hz vs squeal {c_sq:.0f} Hz")
    mp = _Mapper()
    hud = _hud(util_f=1.02, util_r=1.0, V=18.0, alpha=[0.05, 0.16, 0.05, 0.18],
               Fy=[2400.0, 3900.0, 1700.0, 3100.0])
    for _ in range(40):
        lv = mp(hud, 1 / 60)
    s = Synth(rate, CHUNK, seed=9, channels=2)
    _render(s, 0.3, T(squeal=lv['squeal'], sq_pan=lv['sq_pan'], V=18.0, **quiet))
    y = _render(s, 1.0, T(squeal=lv['squeal'], sq_pan=lv['sq_pan'], V=18.0, **quiet))
    eL = _band_energy(y[:, 0], rate, 800.0, 1800.0)
    eR = _band_energy(y[:, 1], rate, 800.0, 1800.0)
    chk("stereo: the squeal pans to the sliding side", lv['sq_pan'] > 0.2 and eR > 2.0 * eL,
        f"right wheels sliding: pan {lv['sq_pan']:+.2f}, R/L energy x{eR / max(eL, 1e-30):.1f}")

    # 8. road: the kerb rumble at V / 0.35 m on its side, gravel impulses
    for V in (21.0, 28.0):
        s = Synth(rate, CHUNK, seed=10, channels=2)
        _render(s, 0.3, T(kerb=1.0, kerb_pan=-PAN_SURF, V=V, **quiet))
        y = _render(s, 1.5, T(kerb=1.0, kerb_pan=-PAN_SURF, V=V, **quiet))
        fk = V / KERB_PITCH
        pk = _peak_hz(y, rate, 25.0, 400.0)
        kk = pk / fk
        eL, eR = _rms(y[:, 0]), _rms(y[:, 1])
        chk(f"kerb rumble at V/{KERB_PITCH} m on its side (V {V:.0f} m/s)",
            abs(kk - round(kk)) * fk <= 2.0 and 1 <= round(kk) <= 5 and eL > 2.0 * eR,
            f"peak {pk:.1f} Hz = {kk:.2f} x {fk:.1f} Hz; L/R rms x{eL / max(eR, 1e-30):.1f}")

    def crunch(V):
        s = Synth(rate, CHUNK, seed=13)
        _render(s, 0.3, T(gravel=1.0, V=V, **quiet))
        y = _render(s, 2.0, T(gravel=1.0, V=V, **quiet))
        yf = np.fft.irfft(np.fft.rfft(y) * ((np.fft.rfftfreq(len(y), 1 / rate) > 1500.0)
                                            & (np.fft.rfftfreq(len(y), 1 / rate) < 4500.0)), len(y))
        kurt = float(np.mean(yf ** 4) / max(np.mean(yf ** 2) ** 2, 1e-30))
        return kurt, _rms(yf)
    k15, r15 = crunch(15.0)
    k4, r4 = crunch(4.0)
    chk("gravel: crunchy impulses, denser with speed", k15 > 4.0 and r15 > 1.4 * r4,
        f"1.5-4.5 kHz kurtosis {k15:.1f} (noise = 3), rms x{r15 / max(r4, 1e-30):.1f} from 4 to 15 m/s")

    # 9. the chimes: once per rising edge, distinct, below the engine
    mp = _Mapper()
    seq = ([_hud(lap_rank='P3')] * 5                                   # up at creation: no cue
           + [_hud()] * 10
           + [_hud(sector_flash='S1 20.1', flash_col='purple')] * 60
           + [_hud()] * 20
           + [_hud(sector_flash='S2 20.3', flash_col='')] * 20          # a lap that cannot count
           + [_hud(sector_flash='S2 20.3', flash_col='green')] * 40
           + [_hud(sector_flash='S3 20.5', flash_col='red', lap_rank='P1', lap_medal='gold')] * 90
           + [_hud()] * 10)
    cues = [e[1] for h in seq for e in mp(h, 1 / 60)['events'] if e[0] == 'cue']
    #  P1 / gold on the lap's last frame chime nothing HERE: the lap's chime
    #  is task 27's CarSound.chime, called once by drive.Sim._rec_lap
    want = ['sector_purple', 'sector_green', 'sector_red']
    chk("chimes fire once per rising edge (sectors; the lap's is CarSound.chime)",
        cues == want, " ".join(cues))
    pks = {}
    for kind in ('sector_purple', 'sector_green', 'sector_red'):
        yc, _ = _render_cue(kind, rate)
        pks[kind] = _peak_hz(yc, rate, 200.0, 4000.0)
    distinct = len({round(v / 20.0) for v in pks.values()}) >= 3
    s = Synth(rate, CHUNK, seed=1)
    s.cue('sector_purple')
    s.cue('sector_green')
    y = _render(s, 1.6, T(**quiet))
    on = np.flatnonzero(np.abs(y) > 0.2 * A_CUE)
    gap_ok = s.cue_free == 0 and len(on) > 0
    yr, tl = _render_cue('sector_purple', rate)
    second = np.abs(y[int((tl + CUE_GAP - 0.01) * rate):]).max()
    chk("chimes distinct, queued, below the engine",
        distinct and gap_ok and second > 0.3 * A_CUE and np.abs(y).max() <= 1.6 * A_CUE,
        "peaks " + " / ".join(f"{v:.0f}" for v in pks.values()) + f" Hz; peak {np.abs(y).max():.3f}")

    # 10. the wing's servo: only while a panel moves
    mp = _Mapper()
    lv_s = []
    for i in range(90):
        tt = i / 60.0
        dl = _smoothstep((tt - 0.25) / 0.45)
        lv_s.append(mp(_hud(wing_deploy_l=dl, wing_deploy_r=0.0), 1 / 60)['servo'])
    mid, before, after = max(lv_s[18:40]), max(lv_s[:14]), max(lv_s[-25:])
    s = Synth(rate, CHUNK, seed=2)
    ysv = _render(s, 1.0, T(servo=1.0, servo_rate=0.7, **quiet))
    e_sv = _band_energy(ysv, rate, 350.0, 3500.0)
    s = Synth(rate, CHUNK, seed=2)
    e_0 = _band_energy(_render(s, 1.0, T(servo=0.0, **quiet)), rate, 350.0, 3500.0)
    chk("wing servo whine only while moving", mid > 0.5 and before == 0.0 and after < 0.02
        and e_sv > 1e3 * max(e_0, 1e-30) and _rms(ysv) < 0.05,
        f"level {before:.2f} before / {mid:.2f} moving / {after:.3f} after; rms {_rms(ysv):.3f}")

    # 11. drivetrain: whine frequency ~ V (reverse louder); the clunk's body
    pw = []
    for V in (20.0, 30.0):
        s = Synth(rate, CHUNK, seed=2)
        _render(s, 0.3, T(V=V, **quiet))
        pw.append(_peak_hz(_render(s, 1.0, T(V=V, **quiet)), rate, 300.0, 1500.0))
    s = Synth(rate, CHUNK, seed=2)
    _render(s, 0.3, T(V=2.5, gear=-1, **quiet))
    r_rev = _rms(_render(s, 1.0, T(V=2.5, gear=-1, **quiet)))
    s = Synth(rate, CHUNK, seed=2)
    _render(s, 0.3, T(V=2.5, gear=1, **quiet))
    r_fwd = _rms(_render(s, 1.0, T(V=2.5, gear=1, **quiet)))
    chk("gear whine follows road speed, louder in reverse",
        abs(pw[0] - K_WHINE * 20.0) <= 3.0 and abs(pw[1] - K_WHINE * 30.0) <= 3.0 and r_rev > 4.0 * r_fwd,
        f"{pw[0]:.0f} / {pw[1]:.0f} Hz at 20 / 30 m/s; reverse rms x{r_rev / max(r_fwd, 1e-30):.1f}")
    s = Synth(rate, CHUNK, seed=1)
    s.clunk(1.0)
    f, sp = _spec(s.fx, rate)
    c_cl = float((sp * f).sum() / sp.sum())
    lo_cl = float(sp[f < 126.0].sum() / sp.sum())
    chk("gear clunk: its body at ~200 Hz, not under a laptop's band",
        150.0 <= c_cl <= 400.0 and lo_cl < 0.3,
        f"centroid {c_cl:.0f} Hz, {100 * lo_cl:.0f} % under 126 Hz (was 90 %)")

    # the gear edges as the physics makes them: g -> 0 (0.25 s) -> g' is ONE
    # shift (one clunk as the new gear engages, the cut on an upshift); a
    # neutral that lasts clunks on its own, late by T_GATE_MAX
    fps = 60
    seq, want = [], []

    def put(g, rpm, ld, n=1):
        for _ in range(n):
            seq.append(_hud(gear=g, rpm=rpm, eng_load=ld))

    def shift_seq(g0, g1, rpm0, rpm1, ld, kind):
        fr = _shift_frames(g0, g1, rpm0, rpm1, fps, ld, blip=g1 < g0)
        for g, rpm, l_ in fr:
            if g == 0 and seq[-1].gear != 0 and g1 == 0:     # a neutral that will hold
                want.append((len(seq) + int(math.ceil(T_GATE_MAX * fps)), 'clunk'))
            if g == g1 and seq[-1].gear != g1 and g1 != 0:
                want.append((len(seq), kind))
            put(g, rpm, l_)
    put(0, 850.0, IDLE_LOAD_N, 36)
    want.append((len(seq), 'clunk'))
    put(1, 700.0, IDLE_LOAD, 30)
    shift_seq(1, 2, 5900.0, 3400.0, 1.0, 'up')
    put(2, 3500.0, 1.0, 30)
    shift_seq(2, 3, 5900.0, 3900.0, 1.0, 'up')
    put(3, 4000.0, 1.0, 30)
    shift_seq(3, 2, 3000.0, 4500.0, 0.0, 'clunk')
    put(2, 4400.0, 0.0, 30)
    shift_seq(2, 0, 2000.0, 850.0, 0.0, 'none')
    put(0, 850.0, IDLE_LOAD_N, 48)
    want.append((len(seq), 'clunk'))
    put(-1, 700.0, IDLE_LOAD, 30)
    shift_seq(-1, 0, 700.0, 850.0, 0.0, 'none')
    put(0, 850.0, IDLE_LOAD_N, 48)
    mp = _Mapper()
    got = []
    for i, h in enumerate(seq):
        for e in mp(h, 1.0 / fps)['events']:
            got.append((i, e[0]))
    cl = [i for i, k in got if k == 'clunk']
    cuts = [i for i, k in got if k == 'shift_cut']
    w_cl = [i for i, k in want if k in ('clunk', 'up')]
    w_cut = [i for i, k in want if k == 'up']
    near = len(cl) == len(w_cl) and all(abs(a - b) <= 1 for a, b in zip(cl, w_cl))
    chk("gears: one clunk per shift (real declutch/gate/engage)", near and cuts == w_cut,
        f"{len(cl)} clunks for {len(w_cl)} changes (3 shifts, neutral-in, 2 neutrals held, "
        f"reverse), {len(cuts)} cuts for {len(w_cut)} upshifts, each on its engage frame")

    # the lift that arms the pops, through the mapper with the real shift:
    # a WOT upshift (the declutch IS a lift at the old gear) arms nothing; a
    # genuine lift arms after T_LIFT_CONFIRM; a lift followed by the auto
    # box's lift-off upshift arms once the new gear is in and the load stays off
    def arm_time(frames):
        mp = _Mapper()
        for i, h in enumerate(frames):
            if mp(h, 1.0 / fps)['pops'] > 0.0:
                return i / fps
        return None
    P = PROFILES['rally']
    hi = 0.9 * P.cut
    wot = [_hud(car_key='rally', gear=2, rpm=hi, eng_load=1.0)] * 30
    fr_up = wot + [_hud(car_key='rally', gear=g, rpm=r, eng_load=l_)
                   for g, r, l_ in _shift_frames(2, 3, hi, 0.7 * hi, fps, 1.0)] \
        + [_hud(car_key='rally', gear=3, rpm=0.72 * hi, eng_load=1.0)] * 60
    fr_lift = wot + [_hud(car_key='rally', gear=2, rpm=hi - 900.0 * i / fps, eng_load=0.0)
                     for i in range(90)]
    fr_lu = wot + [_hud(car_key='rally', gear=g, rpm=r, eng_load=0.0)
                   for g, r, _ in _shift_frames(2, 3, hi, 0.7 * hi, fps, 0.0)] \
        + [_hud(car_key='rally', gear=3, rpm=0.7 * hi, eng_load=0.0)] * 60
    t_up, t_lift, t_lu = arm_time(fr_up), arm_time(fr_lift), arm_time(fr_lu)
    t_eng = 0.5 + T_DECLUTCH + T_GATE
    chk("pops: not for a shift, only a real lift",
        t_up is None and t_lift is not None and abs(t_lift - 0.5 - T_LIFT_CONFIRM) < 0.03
        and t_lu is not None and abs(t_lu - t_eng - T_ENGAGE_CHECK) < 0.03,
        f"WOT upshift {'never' if t_up is None else f'{t_up - 0.5:.2f} s'}; lift "
        + (f"+{t_lift - 0.5:.2f} s" if t_lift else "never") + "; lift + lift-off upshift "
        + (f"engage +{t_lu - t_eng:.2f} s" if t_lu else "never"))

    # 12. the starter; silence
    s = Synth(rate, CHUNK, seed=1)
    _render(s, 0.3, T(rpm=250.0, load=0.0, stalled=True))
    y = _render(s, 1.0, T(rpm=250.0, load=0.0, stalled=True))
    pst = _peak_hz(y, rate, 200.0, 2000.0)
    chk("starter crank while stalled", _rms(y) > 0.02 and abs(pst - STARTER_RATIO * 250.0) < 40.0,
        f"rms {_rms(y):.3f}, pinion whine {pst:.0f} Hz")
    for ch_ in (1, 2):
        s = Synth(rate, CHUNK, seed=1, channels=ch_)
        y = _render(s, 0.3, T(rpm=3000.0, load=1.0, volume=0.0, V=20.0, squeal=0.5, wind=0.3, roll=0.5))
        fade = float(np.abs(y[:CHUNK]).max())        # the first chunk ramps 1 -> 0
        chk(f"volume 0 is digital silence ({'stereo' if ch_ == 2 else 'mono'})",
            float(np.abs(y[CHUNK:]).max()) == 0.0 and 0.0 < fade < 1.0,
            f"peak {np.abs(y[CHUNK:]).max()} after a {fade:.2f} fade chunk")
        s = Synth(rate, CHUNK, seed=1, channels=ch_)
        y = _render(s, 0.3, T(rpm=0.0, load=0.0, stalled=True))
        chk(f"dead engine at rest is silent ({'stereo' if ch_ == 2 else 'mono'})",
            float(np.abs(y).max()) == 0.0, f"peak {np.abs(y).max():.2e}")
    dc = {}
    for car, P in PROFILES.items():            # at the sweep's top: 6000
        s = Synth(rate, CHUNK, seed=3, car=car)
        _render(s, 0.5, T(rpm=P.rpms[-1], load=1.0))
        y = _render(s, 2.0, T(rpm=P.rpms[-1], load=1.0))
        dc[car] = abs(float(y.mean())) / _rms(y)
    chk("no DC (the waveshaper, the soft clip)", all(v < 0.005 for v in dc.values()),
        "|mean| / rms at 6000 WOT " + " / ".join(f"{100 * v:.2f}" for v in dc.values())
        + " % (V8 was 3.6 %)")

    # 13. the numpy filters against lfilter (when scipy is here), and the lazy
    # backend's hand-over from one to the other mid-stream
    lf = _scipy_lfilter()
    if lf is not None:
        rng = np.random.default_rng(0)
        x = rng.standard_normal((2, 3 * CHUNK))
        err = 0.0
        for ba in (_lp1(700.0, rate), _hp1(30.0, rate), _bp(240.0, 4.5, rate),
                   _bp(3200.0, 1.6, rate), _lp2(900.0, 0.7, rate), _lp1(20000.0, rate)):
            a, b = _IIR(ba, lf), _IIR(ba, None)
            for c in range(3):
                seg = x[:, c * CHUNK:(c + 1) * CHUNK]
                ya, yb = a(seg), b(seg)
                err = max(err, float(np.abs(ya - yb).max() / max(np.abs(ya).max(), 1e-12)))
        # the reference is lfilter EXPLICITLY: CARSIM_AUDIO_NUMPY=1 once made
        # both of these numpy and the check compared numpy with itself
        sa = Synth(rate, CHUNK, seed=21, channels=2, car='540i', backend='scipy')
        sb = Synth(rate, CHUNK, seed=21, channels=2, car='540i', backend='numpy')
        for s_ in (sa, sb):                # start AT the targets: nothing ramps
            s_.rpm, s_.load, s_.eng_on = 4200.0, 0.8, 1.0
        # fixed targets = fixed coefficients (the wind's band moves with its
        # gusts every chunk, and a moved coefficient meets a carried state in
        # DF2T form in one path and as pole states in the other: both are
        # continuous, they are not the same continuation)
        kw = dict(rpm=4200.0, load=0.8, squeal=0.6, V=25.0, roll=0.6, kerb=0.5,
                  gravel=0.3, scrub=0.3, wet=0.4, servo=0.4)
        d = max(float(np.abs(sa.chunk(**kw) - sb.chunk(**kw)).max()) for _ in range(4))
        for f_ in sb._filters:             # the lazy hand-over: numpy -> lfilter
            f_.adopt(lf)
        d2 = max(float(np.abs(sa.chunk(**kw) - sb.chunk(**kw)).max()) for _ in range(4))
        chk("numpy filter path equals lfilter; hand-over seamless",
            sa.backend == 'scipy' and err < 1e-9 and d < 1e-6 and d2 < 1e-6,
            f"filters {err:.1e} relative, a whole V8 stereo chunk {d:.1e}, after the "
            f"hand-over {d2:.1e}")
    elif verbose:
        print(f"  SKIP  {'numpy filter path equals lfilter':<50s} scipy not importable")

    # 14. robustness: junk / non-finite HUD values never raise, never leave a
    # NaN behind, and the sound comes back on the next good frame (review:
    # util or kappa NaN silenced it for the session, V NaN disabled it)
    nan, inf = float('nan'), float('inf')
    good = dict(rpm=3000.0, eng_load=0.6, V=20.0, gear=3, util_f=0.95, util_r=0.9,
                kappa=[0.0, 0.0, 0.15, 0.15], alpha=[0.08] * 4, Fy=[3000.0] * 4,
                Fx=[500.0] * 4, surf4=('tarmac',) * 4)
    bads = [dict(rpm=nan), dict(eng_load=nan), dict(V=nan), dict(V=inf), dict(V=-30.0),
            dict(util_f=nan), dict(kappa=[nan] * 4), dict(kappa=[inf, -inf, 0, 0]),
            dict(alpha=[nan] * 4, Fy=[inf] * 4), dict(gear=nan), dict(gear='N'),
            dict(wing_deploy_l=None, wing_deploy_r=None, wing_deploy=0.5, wing_side='L'),
            dict(wing_deploy_l=nan, top_deploy=inf), dict(rpm='3000'), dict(kappa=0.3)]

    def run_seq(bad):
        mp, s = _Mapper(), Synth(rate, CHUNK, seed=3, channels=2)
        ys = []
        seq = [_hud(**good)] * 30 + ([bad] if bad is None else [_hud(**{**good, **bad})]) * 3 \
            + [_hud(**good)] * 60
        for i, h in enumerate(seq):
            lv = mp(h, 1 / 60)
            for ev in lv.pop('events'):
                s.event(*ev)
            lv.pop('clunk', None)
            if i % 3 == 0:
                ys.append((i, s.chunk(**lv)))
        tail = np.concatenate([y for i, y in ys if i >= 60])
        return all(np.all(np.isfinite(y)) for _, y in ys), _rms(tail)
    try:
        fin_ref, r_ref = run_seq({})
        worst, n_ok = 1e9, 0
        for bad in bads + [None]:
            fin, r_ = run_seq(bad)
            n_ok += fin and r_ > 0.5 * r_ref
            worst = min(worst, r_ / r_ref)
        err_s = ''
    except Exception as exc:                     # noqa: BLE001 - the check reports it
        n_ok, worst, err_s = -1, 0.0, f"; raised {type(exc).__name__}: {exc}"
    # ... and a NaN inside the synth itself: one silent chunk, then sound again
    s = Synth(rate, CHUNK, seed=3)
    _render(s, 0.3, T(rpm=3000.0, load=0.8))
    s.f_bright.z = np.full_like(np.asarray(s.f_bright.z, dtype=float), nan) \
        if s.f_bright.lf is not None else np.full_like(s.f_bright.z, nan)
    y_bad = s.chunk(rpm=3000.0, load=0.8)
    y_next = _render(s, 0.5, T(rpm=3000.0, load=0.8))
    rec = (float(np.abs(y_bad).max()) == 0.0 and s.faults == 1
           and np.all(np.isfinite(y_next)) and _rms(y_next[CHUNK:]) > 0.02)
    chk("non-finite / junk HUD values: no raise, sound recovers",
        n_ok == len(bads) + 1 and rec,
        f"{max(n_ok, 0)}/{len(bads) + 1} junk frames recovered (tail rms >= {worst:.2f} x a clean "
        f"run); a NaN in a filter: {'one silent chunk, then sound' if rec else 'NOT recovered'}"
        + err_s)

    # 15. speed: synthesis time IS frame time -- the worst case (V8, stereo,
    # every layer on at once, one-shots firing) and a typical one (Corsa, mono)
    def timed(car, ch_, kw, nn=300, events=False):
        s = Synth(rate, CHUNK, seed=2, channels=ch_, car=car)
        for i in range(10):
            s.chunk(**kw(i))
        ts = np.empty(nn)
        for i in range(nn):
            if events:
                if i % 7 == 0:
                    s.event('clunk', 1.0)
                if i % 11 == 0:
                    s.event('shift_cut', T_SHIFT_CUT)
                if i % 13 == 0:
                    s.event('cue', list(CUES)[i % len(CUES)])
            t0 = time.perf_counter()
            s.chunk(**kw(i))
            ts[i] = (time.perf_counter() - t0) * 1e3
        return float(ts.mean()), float(np.percentile(ts, 99))
    worst = lambda i: dict(rpm=1500.0 + (i * 37.0) % 5000.0, load=0.5 + 0.5 * math.sin(i / 7.0),  # noqa: E731
                           squeal=0.8, lock=0.4, abs_on=1.0, scrub=0.5, sq_pan=0.3, grass=0.4,
                           grass_pan=-0.3, gravel=0.5, gravel_pan=0.4, kerb=0.7, kerb_pan=-0.6,
                           roll=0.8, wet=0.6, wind=0.7, V=35.0, gear=-1 if i % 50 < 5 else 3,
                           pops=0.9 * math.exp(-(i % 40) / 15.0),
                           limit=1.0 if i % 40 < 10 else 0.0, servo=0.7,
                           servo_pan=0.4, servo_rate=0.8, stalled=(i % 60 < 3),
                           starter=1.0 if i % 60 < 3 else 0.0)
    typical = lambda i: dict(rpm=2500.0 + 10.0 * i, load=0.7, squeal=0.2, roll=0.5,     # noqa: E731
                             wind=0.3, V=25.0)
    m_w, p_w = timed('540i', 2, worst, events=True)
    m_t, p_t = timed('corsa', 1, typical)
    backend = Synth(rate, CHUNK).backend
    # the numpy fallback only runs without scipy, which the sim itself cannot
    # (powertrain imports scipy.interpolate) - and for the first ~10 chunks
    # of a CarSound while warm_up imports scipy.signal: it is held to the
    # 3 ms hard cap alone, and says so
    b_mean = 1.5 if backend == 'scipy' else 3.0
    try:
        from .render import frame_budget_verdict
        ok_w, det = frame_budget_verdict(m_w, p_w, budget_mean=b_mean, budget_p99=3.0)
    except Exception as exc:              # no renderer: the raw budget
        ok_w = m_w <= b_mean and p_w <= 3.0
        det = f"mean {m_w:.2f} ms, p99 {p_w:.2f} ms (raw; {type(exc).__name__})"
    chk(f"synthesis time per 2048-sample chunk (worst case, {backend})", ok_w,
        det + ('' if backend == 'scipy' else '; numpy fallback: 3 ms hard cap only'))
    chk("... typical (Corsa, mono, driving)", m_t <= b_mean,
        f"mean {m_t:.2f} ms, p99 {p_t:.2f} ms of a {CHUNK / rate * 1e3:.1f} ms chunk")

    # 16. the streaming path on SDL's dummy driver (no speakers involved)
    for ch_ in (2, 1):
        stream_ok = None
        try:
            import pygame
            env_prev = os.environ.get("SDL_AUDIODRIVER")
            os.environ["SDL_AUDIODRIVER"] = "dummy"
            try:
                pygame.mixer.quit()
                t_b = time.perf_counter()
                snd = CarSound(volume=0.5, seed=1, channels=ch_)
                t_b = (time.perf_counter() - t_b) * 1e3
                if snd.ok:
                    hud = _hud(rpm=850.0, eng_load=0.05, gear=1, V=0.0, util_f=0.2, util_r=0.2)
                    t_end = time.perf_counter() + 0.6
                    frames = 0
                    while time.perf_counter() < t_end:
                        hud.rpm = 850.0 + 4000.0 * frames / 36.0
                        hud.gear = 0 if 10 <= frames < 25 else (1 if frames < 10 else 2)
                        hud.sector_flash, hud.flash_col = ('S1', 'green') if frames > 20 else ('', '')
                        snd.update(hud)
                        frames += 1
                        time.sleep(1 / 60.0)
                    stream_ok = snd.ok and snd.chunks >= 8 and snd.nch == ch_ and t_b < 60.0
                    got = (f"{snd.chunks} chunks in {frames} frames, "
                           f"{snd.underruns} underrun(s), rate {snd.rate} x{snd.nch}; "
                           f"built in {t_b:.1f} ms ({snd.synth.backend})")
                    snd.stop()
                else:
                    got = f"mixer unavailable ({snd.error})"
                pygame.mixer.quit()
            finally:
                if env_prev is None:
                    os.environ.pop("SDL_AUDIODRIVER", None)
                else:
                    os.environ["SDL_AUDIODRIVER"] = env_prev
        except Exception as exc:
            got = f"{type(exc).__name__}: {exc}"
        name = f"streams through pygame.mixer ({'stereo' if ch_ == 2 else 'mono'}, dummy)"
        if stream_ok is None:
            if verbose:
                print(f"  SKIP  {name:<50s} {got}")
        else:
            chk(name, stream_ok, got)

    # 16b. the time trial's lap chime (task 27): CarSound.chime
    pb, md = chime_wave("pb", rate), chime_wave("medal", rate)
    seg = int(NOTE_S * rate)
    f1 = _peak_hz(pb[:seg], rate, 200.0, 3000.0)
    tail = float(np.max(np.abs(pb[-int(0.02 * rate):])))
    chk("chime: rising notes, no clipping, it rings out",
        abs(f1 - CHIMES["pb"][0]) < 15.0 and 0.0 < float(np.max(np.abs(pb))) <= A_CHIME + 1e-9
        and tail < 0.1 * A_CHIME and len(md) < len(pb)
        and _peak_hz(pb[3 * seg:4 * seg], rate, 200.0, 3000.0) > f1,
        f"first note {f1:.0f} Hz, peak {np.max(np.abs(pb)):.2f}, tail {tail:.3f}, "
        f"{len(pb) / rate:.2f} s / {len(md) / rate:.2f} s")
    # warm_up() as the game calls it: once at session start, Sound OFF, in a
    # fresh process (this one has scipy.signal by now) - returns at once,
    # opens no mixer, and the import and the chimes land in the background
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = ("import sys, time, threading; sys.path.insert(0, %r); from drive import audio as A; "
            "si = sys.getswitchinterval(); t = time.perf_counter(); A.warm_up(); "
            "c = time.perf_counter() - t; A.warm_up(); "
            "[th.join(30.0) for th in threading.enumerate() if th is not threading.main_thread()]; "
            "import pygame; "
            "print(c * 1e3, A._LFILTER not in (None, False), "
            "sum(1 for k in A._CUE_CACHE if k[1] == A.RATE), pygame.mixer.get_init() is None, "
            "sys.getswitchinterval() == si)") % root
    try:
        env = dict(os.environ, PYGAME_HIDE_SUPPORT_PROMPT='1', SDL_AUDIODRIVER='dummy')
        res = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                             timeout=60, env=env)
        c_ms, lf_ok, n_cue, no_mix, si_ok = res.stdout.split()[-5:]
        want_lf = _scipy_lfilter() is not None
        ok_w = (float(c_ms) < 20.0 and lf_ok == str(want_lf) and int(n_cue) == len(CUES)
                and no_mix == 'True' and si_ok == 'True')
        got = (f"returned in {float(c_ms):.2f} ms; lfilter {'ready' if lf_ok == 'True' else 'absent'}, "
               f"{n_cue}/{len(CUES)} chimes, no mixer opened: {no_mix}, switch interval restored: {si_ok}")
    except Exception as exc:                     # noqa: BLE001 - the check reports it
        ok_w, got = False, f"{type(exc).__name__}: {exc}"
    chk("warm_up() at session start, sound off (fresh process)", ok_w, got)

    # 17. the demos: a lap-like drive per car, stereo, through the real mapper
    # and the real shift timing - ONE file per car, overwritten each run
    if wav_path is None:
        wav_path = os.path.join("runs", "selfcheck", "audio_demo.wav")
    d0 = os.path.dirname(wav_path) or "."
    try:
        written = []
        demo_ok = True
        n_pops = {}
        for car in PROFILES:
            frames = _lap_script(car)
            s = Synth(RATE, CHUNK, seed=11, channels=2, car=car)
            y, log = _session(frames, car, channels=2, synth=s)
            p = os.path.join(d0, f"audio_demo_{car}.wav")
            _write_wav(p, y * 0.95, rate)
            written.append(p)
            cues = [e[1] for _, e in log if e[0] == 'cue']
            n_cl = sum(1 for _, e in log if e[0] == 'clunk')
            n_pops[car] = s.pops_fired
            demo_ok &= (cues == ['sector_purple', 'sector_green']
                        and n_cl == 10 and np.all(np.isfinite(y)) and float(np.abs(y).max()) <= CEIL)
        sizes = [os.path.getsize(p) for p in written]
        chk("demo laps written (one per car, stereo)", min(sizes) > 1000000 and demo_ok,
            f"{len(written)} x ~{len(y) / rate:.0f} s in {d0}/audio_demo_<car>.wav; pops "
            + " / ".join(f"{k} {v}" for k, v in n_pops.items()))
    except OSError as exc:
        chk("demo laps written (one per car, stereo)", False, f"{exc}")

    ok = all(r[1] for r in results)
    if verbose:
        n = sum(1 for r in results if r[1])
        print(f"  {'ALL PASS' if ok else 'FAILURES ABOVE'}   {n}/{len(results)} checks")
    return ok


if __name__ == "__main__":
    sys.exit(0 if self_check() else 1)
