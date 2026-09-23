"""Procedural car sound for the interactive drive.  No sound files.

What you hear
-------------
* **Engine**: a four-cylinder firing at rpm/30 Hz, built from the firing
  fundamental, its harmonics, the half-order (crankshaft) lump that makes an
  idle sound uneven, one exhaust pop per firing (a decaying ring at F_RES)
  and a load-shaped noise burst per firing.  Load (PowertrainOutput.load:
  the pedal map after the shift lift, the blip and the idle governor) opens
  the harmonics and the noise up; overrun is a quiet hum.  The rev limiter
  stutters it; a stalled engine on the starter turns over slowly.
* **Tyres**: a squeal (a vibrato tone around 1 kHz with two harmonics and a
  hiss) driven by the worst wheel's slip - lateral utilisation past 0.90 or a
  slip ratio past 0.10 - on tarmac above walking pace.
* **Grass**: a low rumble off the tarmac.  **Wind**: a hiss with V^2.
* **Gearbox**: a short clunk as a gear engages.

How it streams
--------------
Everything is synthesised per CHUNK samples in numpy, phase-continuous
across chunks (every oscillator keeps its phase, every smoothing filter its
tail, every level ramps linearly inside the chunk from its previous value),
soft-clipped, and handed to ONE pygame.mixer Channel with play() + queue():
the channel is kept one chunk ahead, so the pitch follows the engine with
one to two chunks (50-90 ms) of latency and a dropped frame has a chunk of
slack before the stream underruns (counted in `underruns`).

Rules
-----
* Physics never reads this module.  `update(hud)` is called from the RENDER
  loop once per frame with the frame's HudData, exactly like the pad rumble
  (CONTRACT section 8): the sound is a consumer of the sim, never an input.
* The mixer is a wall clock; nothing here is deterministic in the sim's
  sense and nothing here needs to be.  Its own RNG is seeded so the offline
  self-check is repeatable.
* No audio device (or no numpy) means `CarSound.ok` is False and the drive
  is silent; nothing raises out of here.

`python3 -m drive.audio` synthesises a demo sweep offline (idle, a full-
throttle run to the limiter, a lift, a squeal), checks it (continuity at
chunk boundaries, the firing frequency in the spectrum, the squeal band,
levels, speed, and the streaming path on SDL's dummy driver) and writes
runs/selfcheck/audio_demo.wav to listen to.
"""
from __future__ import annotations

import collections
import math
import os
import time

import numpy as np

# ---- stream --------------------------------------------------------------
RATE = 44100            # Hz  asked of the mixer; whatever it grants is used
CHUNK = 2048            # samples per synthesised block, 46.4 ms at 44.1 kHz:
#                         the stream is kept one chunk ahead, so this is also
#                         how late a render frame may be before an underrun
BUFFER = 512            # mixer callback buffer (11.6 ms)

# ---- engine --------------------------------------------------------------
N_CYL = 4
FIRINGS_PER_REV = N_CYL / 2.0       # four-stroke
F_RES = 130.0           # Hz  exhaust ring per firing
A_ENGINE = 0.62         # master level of the engine mix
RPM_CRANK = 60.0        # rpm below which the engine is silent
RPM_RUN = 450.0         # rpm at which the crank sound has become the engine
LIMIT_HZ = 22.0         # rev-limiter stutter rate

# ---- tyres / road / wind -------------------------------------------------
UTIL_SQUEAL = 0.90      # lateral utilisation where the squeal starts
UTIL_FULL = 1.00        # ... and is fully on
KAPPA_SQUEAL = 0.10     # slip ratio where it starts (drive peak ~0.12-0.15)
KAPPA_FULL = 0.25
V_SQUEAL = 2.0          # m/s  below this tyres do not squeal
F_SQUEAL = 1000.0       # Hz  base tone (rises with the level)
A_SQUEAL = 0.55
A_GRASS = 0.42
A_WIND = 0.30
V_WIND_REF = 45.0       # m/s  the wind level is (V / V_WIND_REF)^2
A_CLUNK = 0.45
T_CLUNK = 0.035         # s

# ---- level dynamics (per frame, first order) -----------------------------
TAU_SQUEAL_UP = 0.03    # s
TAU_SQUEAL_DOWN = 0.15  # s
TAU_LEVEL = 0.05        # s  wind / grass


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


class _Smoother:
    """Moving average of length k with the tail carried across chunks, so the
    filter sees one continuous signal and every chunk boundary is silent."""

    __slots__ = ("k", "kernel", "tail")

    def __init__(self, k: int):
        self.k = int(k)
        # unit-variance for white noise in: sum of k samples / sqrt(k)
        self.kernel = np.full(self.k, 1.0 / math.sqrt(self.k))
        self.tail = np.zeros(self.k - 1)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.k <= 1:
            return x
        buf = np.concatenate((self.tail, x))
        self.tail = buf[-(self.k - 1):]
        return np.convolve(buf, self.kernel, mode="valid")


class Synth:
    """The signal, without any device.  `chunk(**targets)` returns CHUNK
    float samples in [-1, 1] and advances every state to the targets."""

    def __init__(self, rate: int = RATE, chunk: int = CHUNK, seed: int = 7):
        self.rate = int(rate)
        self.n = int(chunk)
        self.rng = np.random.default_rng(seed)
        self.ramp = np.linspace(0.0, 1.0, self.n, endpoint=False)
        self.t0 = 0                     # samples rendered so far
        # phases, in cycles
        self.ph_eng = 0.0
        self.ph_sq = 0.0
        # previous targets (ramped from, inside each chunk)
        self.rpm = 0.0
        self.load = 0.0
        self.eng_on = 0.0               # 0 silent .. 1 running
        self.squeal = 0.0
        self.grass = 0.0
        self.wind = 0.0
        self.limit = 0.0
        self.master = 1.0
        self.volume = 1.0
        self.jit = 0.0                  # idle rpm jitter, slow random walk
        # filters
        self.sm_intake = _Smoother(6)
        self.sm_hiss = _Smoother(2)
        self.sm_grass = _Smoother(40)
        self.sm_wind = _Smoother(10)
        self.sm_clunk = _Smoother(12)
        # one-shot effects: pending sample arrays, consumed chunk by chunk
        self.fx = np.zeros(0)

    # -- one-shots ---------------------------------------------------------
    def clunk(self, level: float = 1.0) -> None:
        n = int(T_CLUNK * self.rate)
        env = np.exp(-np.arange(n) / (0.30 * n))
        burst = self.sm_clunk(self.rng.standard_normal(n)) * env * A_CLUNK * level
        m = max(len(self.fx), n)
        fx = np.zeros(m)
        fx[:len(self.fx)] += self.fx
        fx[:n] += burst
        self.fx = fx

    # -- the chunk ---------------------------------------------------------
    def chunk(self, rpm: float, load: float, squeal: float = 0.0,
              grass: float = 0.0, wind: float = 0.0, limit: float = 0.0,
              stalled: bool = False, master: float = 1.0,
              volume: float | None = None) -> np.ndarray:
        n, r, ramp = self.n, self.rate, self.ramp
        rng = self.rng
        if volume is None:
            volume = self.volume

        # ---- targets, ramped inside the chunk ----------------------------
        load = _clip01(load)
        # idle unevenness: a slow random walk of the target, shrinking with
        # load and with speed (a revving engine is smooth)
        self.jit = 0.85 * self.jit + 0.15 * rng.standard_normal() * 28.0 * (1.0 - load)
        rpm_t = max(rpm, 0.0) + self.jit * _clip01((1800.0 - rpm) / 900.0)
        eng_on_t = _clip01((rpm - RPM_CRANK) / (RPM_RUN - RPM_CRANK))
        if stalled:
            eng_on_t *= 0.35            # turning over on the starter
        rpm_a = self.rpm + (rpm_t - self.rpm) * ramp
        load_a = self.load + (load - self.load) * ramp
        on_a = self.eng_on + (eng_on_t - self.eng_on) * ramp
        sq_a = self.squeal + (squeal - self.squeal) * ramp
        gr_a = self.grass + (grass - self.grass) * ramp
        wd_a = self.wind + (wind - self.wind) * ramp
        lim_a = self.limit + (limit - self.limit) * ramp
        ms_a = self.master + (master - self.master) * ramp
        vol_a = self.volume + (volume - self.volume) * ramp
        self.rpm, self.load, self.eng_on = rpm_t, load, eng_on_t
        self.squeal, self.grass, self.wind, self.limit = squeal, grass, wind, limit
        self.master, self.volume = master, volume

        t = (self.t0 + np.arange(n)) / r
        self.t0 += n

        # ---- engine ------------------------------------------------------
        f = np.maximum(rpm_a, 1.0) / 60.0 * FIRINGS_PER_REV       # firing Hz
        ph = self.ph_eng + np.cumsum(f) / r
        self.ph_eng = math.fmod(ph[-1] + f[-1] / r, 1.0)
        frac = ph % 1.0
        two_pi_ph = 2.0 * math.pi * ph
        fund = np.sin(two_pi_ph)
        h2 = np.sin(2.0 * two_pi_ph)
        h3 = np.sin(3.0 * two_pi_ph)
        h4 = np.sin(4.0 * two_pi_ph)
        half = np.sin(0.5 * two_pi_ph)
        pulse = np.exp(-frac * (4.0 + 3.0 * load_a))
        ring = pulse * np.sin(2.0 * math.pi * F_RES * frac / f)
        intake = self.sm_intake(rng.standard_normal(n)) * pulse
        eng = (0.55 * fund
               + 0.30 * h2 * (0.5 + 0.5 * load_a)
               + 0.18 * h3 * load_a
               + 0.10 * h4 * load_a
               + 0.22 * half * (1.0 - 0.6 * load_a)
               + 0.60 * ring * (0.3 + 0.7 * load_a)
               + 0.35 * intake * (0.2 + 0.8 * load_a))
        amp = (0.16 + 0.55 * load_a) * (0.55 + 0.45 * np.clip(rpm_a / 6500.0, 0.0, 1.0))
        eng *= amp * on_a * A_ENGINE
        if limit > 0.0 or self.limit > 0.0 or np.any(lim_a > 0.0):
            gate = 0.5 + 0.5 * np.tanh(8.0 * np.sin(2.0 * math.pi * LIMIT_HZ * t))
            eng *= 1.0 - lim_a * 0.6 * (1.0 - gate)

        # ---- tyres -------------------------------------------------------
        out = eng
        if squeal > 0.0 or self.squeal > 0.0 or np.any(sq_a > 0.0):
            f_sq = (F_SQUEAL * (1.0 + 0.25 * sq_a)
                    * (1.0 + 0.03 * np.sin(2.0 * math.pi * 7.0 * t)))
            ph_sq = self.ph_sq + np.cumsum(f_sq) / r
            self.ph_sq = math.fmod(ph_sq[-1] + f_sq[-1] / r, 1.0)
            w = 2.0 * math.pi * ph_sq
            tone = np.sin(w) + 0.45 * np.sin(2.0 * w) + 0.25 * np.sin(3.0 * w)
            hiss = self.sm_hiss(rng.standard_normal(n))
            out = out + (0.55 * tone + 0.35 * hiss) * sq_a * A_SQUEAL

        # ---- grass / wind ------------------------------------------------
        if grass > 0.0 or self.grass > 0.0 or np.any(gr_a > 0.0):
            out = out + self.sm_grass(rng.standard_normal(n)) * gr_a * A_GRASS
        if wind > 0.0 or self.wind > 0.0 or np.any(wd_a > 0.0):
            out = out + self.sm_wind(rng.standard_normal(n)) * wd_a * A_WIND

        # ---- one-shots ---------------------------------------------------
        if len(self.fx):
            m = min(n, len(self.fx))
            out = out.copy()
            out[:m] += self.fx[:m]
            self.fx = self.fx[m:]

        # ---- mix ---------------------------------------------------------
        return np.tanh(1.3 * out) * ms_a * vol_a


class CarSound:
    """The device side: a Synth streamed through one pygame.mixer Channel.

    `update(hud)` once per render frame.  `set_volume`, `stop`.  `ok` is
    False (and `error` says why) when there is no usable audio device; the
    caller then simply drops the object.
    """

    def __init__(self, volume: float = 0.6, rate: int = RATE,
                 chunk: int = CHUNK, seed: int | None = None):
        self.ok = False
        self.error = ""
        self.volume = float(volume)
        self.underruns = 0
        self.chunks = 0
        self._keep: collections.deque = collections.deque(maxlen=6)
        self._gear = None
        self._sq = 0.0
        self._grass = 0.0
        self._wind = 0.0
        self._t_last = None
        try:
            import pygame
            self._pg = pygame
            init = pygame.mixer.get_init()
            if init is None or init[0] != rate or init[2] != 1:
                if init is not None:
                    pygame.mixer.quit()
                pygame.mixer.init(frequency=rate, size=-16, channels=1,
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
        self.synth = Synth(self.rate, chunk,
                           seed=(seed if seed is not None else 7))
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
    def levels(self, hud) -> dict:
        """HudData -> the synth's targets.  Pure; also used by the check."""
        now = time.monotonic()
        dt = 1.0 / 60.0 if self._t_last is None else min(max(now - self._t_last, 1e-3), 0.1)
        self._t_last = now

        rpm = float(getattr(hud, "rpm", 0.0))
        load = float(getattr(hud, "eng_load", 0.0))
        V = float(getattr(hud, "V", 0.0))
        stalled = bool(getattr(hud, "stalled", False))
        limit = 1.0 if getattr(hud, "on_limiter", False) else 0.0
        on_track = bool(getattr(hud, "on_track", True))
        paused = bool(getattr(hud, "paused", False))

        util = max(float(getattr(hud, "util_f", 0.0)),
                   float(getattr(hud, "util_r", 0.0)))
        kap = getattr(hud, "kappa", None)
        kmax = 0.0
        if kap is not None:
            try:
                kmax = float(np.max(np.abs(np.asarray(kap, dtype=float))))
            except (TypeError, ValueError):
                kmax = 0.0
        lat = _clip01((util - UTIL_SQUEAL) / (UTIL_FULL - UTIL_SQUEAL))
        lon = _clip01((kmax - KAPPA_SQUEAL) / (KAPPA_FULL - KAPPA_SQUEAL))
        sq_t = max(lat, lon) * _clip01((V - V_SQUEAL) / 3.0) * (1.0 if on_track else 0.25)
        tau = TAU_SQUEAL_UP if sq_t > self._sq else TAU_SQUEAL_DOWN
        self._sq += (sq_t - self._sq) * min(dt / tau, 1.0)

        grass_t = 0.0 if on_track else (0.30 + 0.45 * _clip01(V / 20.0)) * _clip01(V / 0.8)
        wind_t = _clip01((V / V_WIND_REF) ** 2)
        a = min(dt / TAU_LEVEL, 1.0)
        self._grass += (grass_t - self._grass) * a
        self._wind += (wind_t - self._wind) * a

        gear = getattr(hud, "gear", None)
        clunk = (self._gear is not None and gear not in (None, 0)
                 and gear != self._gear and self._gear == 0)
        self._gear = gear
        return dict(rpm=rpm, load=load, squeal=self._sq, grass=self._grass,
                    wind=self._wind, limit=limit, stalled=stalled,
                    master=0.0 if paused else 1.0, clunk=clunk)

    def _sound(self, lv: dict):
        s = self.synth
        if lv.pop("clunk", False):
            s.clunk()
        y = s.chunk(volume=self.volume, **lv)
        pcm = (np.clip(y, -1.0, 1.0) * 32767.0).astype(np.int16)
        if self.nch > 1:
            pcm = np.repeat(pcm[:, None], self.nch, axis=1)
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
            ch = self.ch
            if not ch.get_busy():
                if self.chunks:
                    self.underruns += 1
                ch.play(self._sound(dict(lv)))
                lv["clunk"] = False
            if ch.get_queue() is None:
                ch.queue(self._sound(lv))
        except Exception as exc:          # a device that went away mid-run
            self.error = f"{type(exc).__name__}: {exc}"
            self.ok = False


# ====================================================================== #
#  SELF-CHECK                                                            #
# ====================================================================== #
def _demo_levels(t: float) -> dict:
    """The offline demo: idle 0-2 s, WOT 2-6 s to the limiter, lift at 6 s
    with a squeal 6.5-7.5 s, back to idle by 8 s."""
    if t < 2.0:
        return dict(rpm=850.0, load=0.05, squeal=0.0, wind=0.0, limit=0.0)
    if t < 6.0:
        rpm = min(850.0 + (t - 2.0) * 1500.0, 6200.0)
        return dict(rpm=rpm, load=1.0, squeal=0.0, wind=((t - 2.0) / 6.0) ** 2,
                    limit=1.0 if rpm >= 6150.0 else 0.0)
    if t < 7.5:
        sq = 1.0 if 6.5 <= t < 7.5 else 0.0
        return dict(rpm=max(6100.0 - (t - 6.0) * 2500.0, 2400.0), load=0.0,
                    squeal=sq, wind=0.4, limit=0.0)
    return dict(rpm=max(2400.0 - (t - 7.5) * 3000.0, 850.0), load=0.03,
                squeal=0.0, wind=0.1, limit=0.0)


def _render(synth: Synth, seconds: float, fn) -> np.ndarray:
    out = []
    n_chunks = int(round(seconds * synth.rate / synth.n))
    for i in range(n_chunks):
        out.append(synth.chunk(**fn(i * synth.n / synth.rate)))
    return np.concatenate(out)


def _peak_hz(y: np.ndarray, rate: int, f_hi: float) -> float:
    w = np.hanning(len(y))
    spec = np.abs(np.fft.rfft(y * w))
    freqs = np.fft.rfftfreq(len(y), 1.0 / rate)
    m = freqs <= f_hi
    return float(freqs[m][np.argmax(spec[m])])


def _band_energy(y: np.ndarray, rate: int, lo: float, hi: float) -> float:
    spec = np.abs(np.fft.rfft(y)) ** 2
    freqs = np.fft.rfftfreq(len(y), 1.0 / rate)
    return float(spec[(freqs >= lo) & (freqs <= hi)].sum())


def self_check(verbose: bool = True, wav_path: str | None = None) -> bool:
    results = []

    def chk(name, ok, got):
        results.append((name, bool(ok), got))
        if verbose:
            print(f"  {'PASS' if ok else 'FAIL'}  {name:<46s} {got}")
        return ok

    rate = RATE
    # 1. continuity across chunk boundaries under a fast rpm sweep + squeal
    s = Synth(rate, CHUNK, seed=3)
    y = _render(s, 3.0, lambda t: dict(rpm=900.0 + 1700.0 * t, load=0.5 + 0.5 * math.sin(3 * t),
                                       squeal=0.5, wind=0.3, grass=0.2))
    d = np.abs(np.diff(y))
    idx = np.arange(CHUNK, len(y), CHUNK) - 1
    step_b = float(d[idx].max())
    mask = np.ones(len(d), dtype=bool)
    mask[idx] = False
    step_in = float(d[mask].max())
    chk("no click at chunk boundaries", step_b <= 1.3 * step_in and np.all(np.isfinite(y)),
        f"boundary step {step_b:.3f} vs {step_in:.3f} inside")
    chk("levels inside [-1, 1]", float(np.abs(y).max()) <= 1.0, f"peak {np.abs(y).max():.3f}")

    # 2. the firing frequency: 3000 rpm -> 100 Hz (or a harmonic / half-order)
    s = Synth(rate, CHUNK, seed=5)
    _render(s, 0.5, lambda t: dict(rpm=3000.0, load=1.0))
    y = _render(s, 1.0, lambda t: dict(rpm=3000.0, load=1.0))
    f0 = 3000.0 / 60.0 * FIRINGS_PER_REV
    pk = _peak_hz(y, rate, 600.0)
    k = pk / f0
    chk("engine spectrum peaks on a firing order", abs(k - round(k * 2) / 2) * f0 <= 3.0
        and 0.4 <= k <= 6.5, f"peak {pk:.1f} Hz = {k:.2f} x {f0:.0f} Hz")
    # louder and brighter under load than on overrun
    s = Synth(rate, CHUNK, seed=5)
    _render(s, 0.5, lambda t: dict(rpm=3000.0, load=0.0))
    y0 = _render(s, 1.0, lambda t: dict(rpm=3000.0, load=0.0))
    chk("load makes it louder", float(np.sqrt(np.mean(y ** 2))) > 2.0 * float(np.sqrt(np.mean(y0 ** 2))),
        f"rms WOT {np.sqrt(np.mean(y**2)):.3f} vs overrun {np.sqrt(np.mean(y0**2)):.3f}")

    # 3. squeal band
    s = Synth(rate, CHUNK, seed=9)
    _render(s, 0.3, lambda t: dict(rpm=2500.0, load=0.3, squeal=1.0))
    ys = _render(s, 1.0, lambda t: dict(rpm=2500.0, load=0.3, squeal=1.0))
    s = Synth(rate, CHUNK, seed=9)
    _render(s, 0.3, lambda t: dict(rpm=2500.0, load=0.3, squeal=0.0))
    yn = _render(s, 1.0, lambda t: dict(rpm=2500.0, load=0.3, squeal=0.0))
    e1 = _band_energy(ys, rate, 800.0, 1800.0)
    e0 = _band_energy(yn, rate, 800.0, 1800.0)
    chk("tyre squeal fills 0.8-1.8 kHz", e1 > 8.0 * e0, f"band energy x{e1 / max(e0, 1e-9):.0f}")

    # 4. silence: volume 0, and a dead engine at rest
    s = Synth(rate, CHUNK, seed=1)
    y = _render(s, 0.3, lambda t: dict(rpm=3000.0, load=1.0, volume=0.0))
    fade = float(np.abs(y[:CHUNK]).max())        # the first chunk ramps 1 -> 0
    chk("volume 0 is digital silence (after the fade)",
        float(np.abs(y[CHUNK:]).max()) == 0.0 and 0.0 < fade < 1.0,
        f"peak {np.abs(y[CHUNK:]).max()} after a {fade:.2f} fade chunk")
    s = Synth(rate, CHUNK, seed=1)
    y = _render(s, 0.3, lambda t: dict(rpm=0.0, load=0.0, stalled=True))
    chk("dead engine at rest is silent", float(np.abs(y).max()) < 1e-6, f"peak {np.abs(y).max():.2e}")

    # 5. speed
    s = Synth(rate, CHUNK, seed=2)
    t0 = time.perf_counter()
    nn = 200
    for i in range(nn):
        s.chunk(rpm=900.0 + 25.0 * i, load=0.7, squeal=0.3, wind=0.2, grass=0.1, limit=0.0)
    per = (time.perf_counter() - t0) / nn
    chk("synthesis faster than real time", per < 0.25 * CHUNK / rate,
        f"{per * 1e3:.2f} ms per {CHUNK / rate * 1e3:.1f} ms chunk")

    # 6. the streaming path on SDL's dummy driver (no speakers involved)
    stream_ok = None
    try:
        import pygame
        env_prev = os.environ.get("SDL_AUDIODRIVER")
        os.environ["SDL_AUDIODRIVER"] = "dummy"
        try:
            pygame.mixer.quit()
            snd = CarSound(volume=0.5, seed=1)
            if snd.ok:
                from types import SimpleNamespace
                hud = SimpleNamespace(rpm=850.0, eng_load=0.05, V=0.0, stalled=False,
                                      on_limiter=False, on_track=True, paused=False,
                                      util_f=0.2, util_r=0.2, kappa=np.zeros(4), gear=1)
                t_end = time.perf_counter() + 0.6
                frames = 0
                while time.perf_counter() < t_end:
                    hud.rpm = 850.0 + 4000.0 * frames / 36.0
                    hud.gear = 0 if 10 <= frames < 14 else 2
                    snd.update(hud)
                    frames += 1
                    time.sleep(1 / 60.0)
                stream_ok = snd.ok and snd.chunks >= 8
                got = (f"{snd.chunks} chunks in {frames} frames, "
                       f"{snd.underruns} underrun(s), rate {snd.rate} x{snd.nch}")
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
    if stream_ok is None:
        if verbose:
            print(f"  SKIP  {'streaming (dummy driver)':<46s} {got}")
    else:
        chk("streams through pygame.mixer (dummy driver)", stream_ok, got)

    # 8. the time trial's chimes (task 27)
    pb, md = chime_wave("pb", rate), chime_wave("medal", rate)
    seg = int(NOTE_S * rate)
    f1 = _peak_hz(pb[:seg], rate, 3000.0)
    tail = float(np.max(np.abs(pb[-int(0.02 * rate):])))
    chk("chime: rising notes, no clipping, it rings out",
        abs(f1 - CHIMES["pb"][0]) < 15.0 and 0.0 < float(np.max(np.abs(pb))) <= A_CHIME + 1e-9
        and tail < 0.1 * A_CHIME and len(md) < len(pb)
        and _peak_hz(pb[3 * seg:4 * seg], rate, 3000.0) > f1,
        f"first note {f1:.0f} Hz, peak {np.max(np.abs(pb)):.2f}, tail {tail:.3f}, "
        f"{len(pb) / rate:.2f} s / {len(md) / rate:.2f} s")

    # 7. the demo file
    if wav_path is None:
        wav_path = os.path.join("runs", "selfcheck", "audio_demo.wav")
    try:
        import wave
        s = Synth(rate, CHUNK, seed=11)
        y = _render(s, 8.5, _demo_levels)
        pcm = (np.clip(y * 0.9, -1.0, 1.0) * 32767.0).astype(np.int16)
        os.makedirs(os.path.dirname(wav_path) or ".", exist_ok=True)
        with wave.open(wav_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(pcm.tobytes())
        chk("demo written", os.path.getsize(wav_path) > 100000, wav_path)
    except OSError as exc:
        chk("demo written", False, f"{exc}")

    ok = all(r[1] for r in results)
    if verbose:
        n = sum(1 for r in results if r[1])
        print(f"  {'ALL PASS' if ok else 'FAILURES ABOVE'}   {n}/{len(results)} checks")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_check() else 1)
