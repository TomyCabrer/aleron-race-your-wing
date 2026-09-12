"""Fixed-schema CSV telemetry for the flank-wing driving sim.

Why this file is shaped the way it is
-------------------------------------
The device under study is worth 0.6-1.5% of corner speed (62.9 N at R=50 on a
9908 N car). Nothing about it can be felt through a keyboard, so the ONLY way
the sim answers the project question is by logging enough state, precisely
enough, that `plots.compare(by='distance')` can resolve a few tenths of a
percent between two runs. That forces three decisions:

  * **The column list is a contract, not a convenience.** `TELEMETRY_COLUMNS`
    is copied verbatim from specs/harness.txt and its ORDER is part of the
    interface: plots.py and any external diff tool index by position. A field
    the vehicle model cannot supply is logged as `nan`, never omitted, so the
    schema is the same width for every run ever made.
  * **Two precisions, for two different jobs.** `'6g'` is the working format;
    `repr()` (`precision='full'`) is what the determinism test V20 compares.
    At 6g two runs that differ in the 8th digit compare byte-identical and the
    test passes vacuously, which is worse than no test at all.
  * **Buffered writes.** Measured elsewhere in this project and stated in the
    spec's pitfalls: 100 Hz x 65 columns through `csv.writer` one row at a
    time, flushed, costs more than the physics step it is logging. Buffer 200
    rows, one `writerows` call, and open with `newline=''` so csv owns the
    line terminator (otherwise every row gets a stray \\r under some tooling).

The sidecar `.json` exists because a PNG six months from now is worthless if
nobody can say which car, which tyre file, which wing and which timestep made
it. It carries the full `CorsaC` field dump, the tyre file, the wing config,
every harness constant and DT_PHYS.

Import surface: `csv`, `json`, `os`, `time` for the hot path. `corsa_c` and
`qss` are imported lazily inside the sidecar builder and the test fixture --
the contract's module map lists only `csv` for this file, but it also demands
a CorsaC field dump in the sidecar, so the import is pushed out of the logging
path rather than out of the module.
"""

import csv
import json
import os
import sys
import time

# ------------------------------------------------------------------ schema --
# VERBATIM from specs/harness.txt. Column ORDER is part of the contract.
# NOTE (reported upward, not silently fixed): the spec calls this "63 columns"
# in two places and "60-column header" in a third; the literal list it gives is
# 65 names long. The LIST is authoritative -- it is what plots.py indexes and
# what a diff tool relies on -- so the list is used as written and the count is
# taken from it (`N_COLUMNS`), never hard-coded to 63.
TELEMETRY_COLUMNS = ['t', 'x', 'y', 'psi', 'u', 'v', 'r', 'V', 'beta_deg', 'ax', 'ay', 'ax_g', 'ay_g',
 'gear', 'rpm', 'clutch_eng', 'throttle', 'brake', 'clutch', 'handbrake', 'delta_road_deg', 'steer_in',
 'Fz_fl', 'Fz_fr', 'Fz_rl', 'Fz_rr', 'Fx_fl', 'Fx_fr', 'Fx_rl', 'Fx_rr', 'Fy_fl', 'Fy_fr', 'Fy_rl', 'Fy_rr',
 'alpha_fl_deg', 'alpha_fr_deg', 'alpha_rl_deg', 'alpha_rr_deg', 'kappa_fl', 'kappa_fr', 'kappa_rl', 'kappa_rr',
 'mu_fl', 'mu_fr', 'mu_rl', 'mu_rr', 'omega_fl', 'omega_fr', 'omega_rl', 'omega_rr',
 'util_f', 'util_r', 'limited_by', 'wing_deploy', 'wing_side', 'F_wing', 'D_wing',
 's', 'n', 'kappa_track', 'on_track', 'lap', 'sector', 'lap_time', 'event']

N_COLUMNS = len(TELEMETRY_COLUMNS)

# Columns that are text, not numbers. Everything else is written as a number
# (or as an empty field when the model has nothing to say).
STRING_COLUMNS = ('limited_by', 'event')

# Columns that are integers. Written with str(int) so a gear never appears as
# "3.0" and a lap counter never as "1.00000".
INT_COLUMNS = ('gear', 'wing_side', 'lap', 'sector')

# Columns that are booleans. Written 0/1 -- numpy/np.loadtxt-friendly, and
# unambiguous in a way that "True"/"False" is not across locales and tools.
BOOL_COLUMNS = ('clutch_eng', 'on_track')

COLUMN_UNITS = {
    't': 's  simulated time, n_steps*dt (never a wall clock)',
    'x': 'm  world', 'y': 'm  world (+y = LEFT in the ISO body frame at psi=0)',
    'psi': 'rad  yaw, +ve counter-clockwise = left turn',
    'u': 'm/s  body forward', 'v': 'm/s  body left', 'r': 'rad/s  yaw rate, +ve left',
    'V': 'm/s  speed, sqrt(u^2+v^2)',
    'beta_deg': 'deg  body slip, atan2(v, |u|)',
    'ax': 'm/s^2  body, at the CG', 'ay': 'm/s^2  body, at the CG, +ve left',
    'ax_g': 'g  ax/9.81', 'ay_g': 'g  ay/9.81',
    'gear': '-  0 = neutral, -1 = reverse',
    'rpm': 'rpm  engine',
    'clutch_eng': 'bool  1 = clutch fully engaged',
    'throttle': '0..1', 'brake': '0..1', 'clutch': '0..1, 1 = fully disengaged',
    'handbrake': '0..1',
    'delta_road_deg': 'deg  commanded ROAD-WHEEL angle, +ve steers LEFT',
    'steer_in': '-  raw driver steer demand, -1..+1 (keyboard/pad axis)',
    'Fz_fl': 'N', 'Fz_fr': 'N', 'Fz_rl': 'N', 'Fz_rr': 'N',
    'Fx_fl': 'N  body frame, +ve forward', 'Fx_fr': 'N', 'Fx_rl': 'N', 'Fx_rr': 'N',
    'Fy_fl': 'N  body frame, +ve LEFT', 'Fy_fr': 'N', 'Fy_rl': 'N', 'Fy_rr': 'N',
    'alpha_fl_deg': 'deg  slip angle, atan(ky); +ve alpha gives Fy<0',
    'alpha_fr_deg': 'deg', 'alpha_rl_deg': 'deg', 'alpha_rr_deg': 'deg',
    'kappa_fl': '-  transient slip ratio, Vsx/|Vx|', 'kappa_fr': '-',
    'kappa_rl': '-', 'kappa_rr': '-',
    'mu_fl': '-  surface mu_scale under that wheel (1.0 dry tarmac)',
    'mu_fr': '-', 'mu_rl': '-', 'mu_rr': '-',
    'omega_fl': 'rad/s  wheel spin', 'omega_fr': 'rad/s',
    'omega_rl': 'rad/s', 'omega_rr': 'rad/s',
    'util_f': '-  |Fy_f| / axle capacity from qss.fy_max; 1.0 = at the limit',
    'util_r': '-  ditto rear',
    'limited_by': 'str  FRONT | REAR | POWER',
    'wing_deploy': '0..1  smoothstepped deploy fraction, 0.45 s actuator',
    # SIGN CONVENTION: wing_side is the TURN sign, not a flank index. Measured
    # from vehicle.py (wing_side = sgn_dev = sign of the steering command):
    # steady_state_corner(R=100, side=+1) is a LEFT turn and gives sgn_dev = +1
    # with cfg.dev_RIGHT selected. CONTRACT section 4's old parenthetical
    # ("-1 right, 0 none, +1 left") read as a flank index and was wrong; it and
    # this comment were corrected together.
    'wing_side': '-  the TURN sign (sgn_dev): +1 = left turn (RIGHT panel out), '
                 '-1 = right turn (LEFT panel out), 0 = never armed',
    'F_wing': 'N  device side force, signed +ve LEFT (inward in a left turn)',
    'D_wing': 'N  device drag, F_wing/3.2, always >= 0',
    's': 'm  arclength along the track centreline',
    'n': 'm  lateral offset, +ve LEFT of the centreline',
    'kappa_track': '1/m  centreline curvature at s, +ve = left',
    'on_track': 'bool  |n| <= width/2',
    'lap': '-  completed-lap counter, 0 before the first line crossing',
    'sector': '-  1-based sector index',
    'lap_time': 's  elapsed time in the current lap',
    'event': 'str  mark() labels; empty otherwise',
}

# ---------------------------------------------------- harness constants ------
# Recorded in the sidecar so a plot is reproducible without the session.
# Values and provenance from specs/harness.txt PARAMETERS. These are the SPEC
# values; _harness_constants() overlays whatever the modules that actually ran
# hold, when they are already imported, so the sidecar records reality rather
# than intent.
HARNESS_CONSTANTS = {
    'DT_PHYS': 0.001,            # derived: resolves sigma/V = 13 ms relaxation
    'DT_PHYS_FALLBACK': 0.002,   # derived: 500 Hz fallback if the RTF test fails
    'FPS': 60,                   # published: requirement
    'MAX_SUBSTEPS': 40,          # derived: 2.4x catch-up on a 16.7 ms frame
    'MAX_FRAME_DT': 0.10,        # derived: caps a window drag / GC pause
    'TELEM_HZ': 100,             # derived: >30 samples across a 0.36 s shift
    'SURFACE_LOOKUP_HZ': 200,    # derived: 0.17 m of travel at 33 m/s
    'STEER_RATIO': 16.0,         # est +/-1.5 (corsa_c; 15.86:1 from 2.9 turns)
    'DELTA_LOCK_DEG': 32.625,    # derived: 522 deg at the wheel / 16.0
    'W_HAND_DRIVE_DEG_S': 900.0, # est, band 600-1200 (keyboard concession)
    'K_US_DEG': 3.2,             # est, band 2.0-5.0. STEER LIMITER ONLY, never physics
    'AY_MAX_DRY': 8.4608,        # derived: qss.max_ay(V) with k=0 -> 0.8624671 g
    'BETA_LOCK_GAIN': 1.2,       # est, band 0.8-1.5
    'T_DEPLOY': 0.45,            # est, band 0.30-1.00 s. UNPUBLISHED - no such device
    'S_DEV': 0.35,               # published: ledger.S_DEV, ONE panel
    'LD_DEV': 3.2,               # published: ledger device L/D
    'CL_FIN': 0.70,              # published: crossover.q3_gain clean fin
    'CL_PLATE': 1.25,            # published: crossover.q3_gain sealed plate
    'X_W': 0.97,                 # published: crossover.MOUNTS['front axle']
    'H_W': 0.90,                 # published: qss default h_w
    'MU_WET_SCALE': 0.632183908, # published: qss.sweep 0.55/0.87
    'MU_DAMP_SCALE': 0.80,       # est, band 0.75-0.88
    'MU_OFF_TRACK': 0.55,        # est, not calibrated - a deterrent
    'CRR_OFF_SCALE': 25.0,       # est: Crr 0.012 -> 0.30 ploughing off circuit
    'DS': 0.5,                   # derived: 2499 samples on CIRCUIT_ARENA
    'PROJECTION_GRID_CELL': 8.0, # derived: 3x3 spans 24 m >= half-width
    'RHO': 1.2,                  # published (corsa_c convention lock)
    'G': 9.81,
}

# Where the harness modules keep their own copy of the same number, if they are
# loaded. Only keys already present in HARNESS_CONSTANTS are overlaid, and only
# scalars -- this must never be able to inject junk into the sidecar.
_CONSTANT_SOURCES = ('drive.drive', 'drive.input', 'drive.render', 'drive.track',
                     'drive.vehicle')

DEFAULT_TYRE_FILE = 'tyre_data/TNO_car205_60R15.tir'   # contract section 2

BUFFER_ROWS = 200          # spec pitfall: one writerows per 200 rows, no flush


def _harness_constants(overrides=None):
    """Spec constants, overlaid with what the loaded harness modules actually hold."""
    out = dict(HARNESS_CONSTANTS)
    for mod_name in _CONSTANT_SOURCES:
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        for key in out:
            val = getattr(mod, key, None)
            if isinstance(val, (int, float, str)) and not isinstance(val, bool):
                out[key] = val
    if overrides:
        out.update({k: v for k, v in overrides.items()
                    if isinstance(v, (int, float, str, bool))})
    return out


class TelemetryWriter:
    """Fixed-width CSV logger, decimated from the physics rate to `hz`.

    path       target .csv. Parent directories are created. The sidecar is the
               same path with '.json' in place of the extension.
    hz         logging rate. The decimation stride is derived from `hz` and the
               physics timestep (`meta['dt']`, default DT_PHYS = 0.001): every
               stride-th call to maybe_log() writes a row.
    precision  '6g'   -> f'{v:.6g}'  (working format, ~half the bytes)
               'full' -> repr(v)     (shortest round-trip; V20 determinism)
    meta       provenance for the sidecar: {'car': CorsaC, 'track': str,
               'tyre_file': str, 'wing': dict, 'dt': float, 'harness': dict,
               'cmdline': list, ...}. Anything else is copied through verbatim.
    """

    def __init__(self, path, hz=100, precision='6g', meta=None):
        if precision not in ('6g', 'full'):
            raise ValueError(f"precision must be '6g' or 'full', got {precision!r}")
        self.path = os.fspath(path)
        self.hz = int(hz)
        self.precision = precision
        self.meta = dict(meta or {})

        self.dt = float(self.meta.get('dt', HARNESS_CONSTANTS['DT_PHYS']))
        # stride = physics steps per logged row. round(), not int(): at
        # hz=100, dt=0.001 that is exactly 10; at dt=0.002 it is 5.
        self.stride = max(1, int(round(1.0 / (self.hz * self.dt))))

        d = os.path.dirname(os.path.abspath(self.path))
        if d:
            os.makedirs(d, exist_ok=True)
        self._fh = open(self.path, 'w', newline='')      # newline='' -> csv owns \r\n
        self._w = csv.writer(self._fh)
        self._w.writerow(TELEMETRY_COLUMNS)

        self._buf = []
        self._pending = []        # mark() labels waiting for a row
        self._force_next = False  # a mark must never be dropped between samples
        self.rows_written = 0
        self.marks = []           # (row_index, t, label) -- copied into the sidecar
        self.closed = False

        # Bound as attributes so the hot path does no attribute lookup on self
        # for the formatter choice.
        self._is_full = (precision == 'full')

    # ------------------------------------------------------------ logging --
    def _fmt(self, name, v):
        """One field. Missing -> '' for text, 'nan' for numbers (never omitted)."""
        if v is None:
            return '' if name in STRING_COLUMNS else 'nan'
        # numpy scalars (np.float64 is a float subclass, np.int64 is NOT).
        item = getattr(v, 'item', None)
        if item is not None and not isinstance(v, (str, bytes)):
            v = item()
        if name in STRING_COLUMNS:
            return str(v)
        if name in BOOL_COLUMNS:
            return '1' if v else '0'
        if name in INT_COLUMNS:
            return str(int(v))
        if isinstance(v, bool):
            return '1' if v else '0'
        return repr(float(v)) if self._is_full else f'{float(v):.6g}'

    def maybe_log(self, step_index, row):
        """Log `row` if this physics step falls on the telemetry stride.

        `row` is a plain dict keyed by column name; unknown keys are ignored and
        absent keys become nan/'' so the schema width is invariant.
        """
        if self.closed:
            raise ValueError('maybe_log() after close()')
        if not self._force_next and (step_index % self.stride):
            return
        self._force_next = False

        if self._pending:
            existing = row.get('event') or ''
            merged = ';'.join(self._pending + ([existing] if existing else []))
            row = dict(row)
            row['event'] = merged
            for label in self._pending:
                self.marks.append((self.rows_written, row.get('t'), label))
            self._pending = []

        fmt = self._fmt
        self._buf.append([fmt(c, row.get(c)) for c in TELEMETRY_COLUMNS])
        self.rows_written += 1
        if len(self._buf) >= BUFFER_ROWS:
            self._w.writerows(self._buf)
            self._buf.clear()

    def mark(self, label):
        """Attach `label` to the next logged row.

        The next maybe_log() is forced to emit, so a marker landing between two
        decimation samples is never silently lost. The cost is that one sample
        interval can be short by up to one telemetry period at that instant --
        plots interpolate on t, so nothing downstream cares.
        """
        self._pending.append(str(label).replace(';', ',').replace('\n', ' '))
        self._force_next = True

    # ------------------------------------------------------------ closing --
    def close(self):
        """Flush the buffer, close the CSV, write the sidecar. Idempotent."""
        if self.closed:
            return
        if self._buf:
            self._w.writerows(self._buf)
            self._buf.clear()
        self._fh.close()
        self.closed = True
        if self._pending:
            # Marks after the final row have nowhere to go in the CSV; keep them
            # in the sidecar rather than dropping them without trace.
            for label in self._pending:
                self.marks.append((self.rows_written, None, label))
            self._pending = []
        self._write_sidecar()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    @property
    def sidecar_path(self):
        base, _ = os.path.splitext(self.path)
        return base + '.json'

    def _write_sidecar(self):
        from corsa_c import CorsaC          # lazy: keeps the hot path's imports at csv
        from dataclasses import asdict, is_dataclass

        car = self.meta.get('car') or CorsaC()
        car_dump = asdict(car) if is_dataclass(car) else dict(car)
        # The properties are derived, but a plot that needs `a`, `b` or P_wheel
        # should not have to re-derive them from wdist_f six months from now.
        for prop in ('a', 'b', 't', 'P_wheel'):
            val = getattr(car, prop, None)
            if isinstance(val, (int, float)):
                car_dump[prop + '_derived'] = val

        wing = dict(self.meta.get('wing') or {})
        wing.setdefault('wing', 'off')
        wing.setdefault('x_w', HARNESS_CONSTANTS['X_W'])
        wing.setdefault('h_w', HARNESS_CONSTANTS['H_W'])
        wing.setdefault('S_DEV', HARNESS_CONSTANTS['S_DEV'])
        wing.setdefault('LD_DEV', HARNESS_CONSTANTS['LD_DEV'])
        cl = {'off': 0.0, 'fin': HARNESS_CONSTANTS['CL_FIN'],
              'plate': HARNESS_CONSTANTS['CL_PLATE']}.get(wing['wing'], 0.0)
        wing.setdefault('CL0', cl)
        # k = 0.5*rho*S*CL is what qss/crossover/ledger all take as their handle
        # on the device; store it so plots.gg() can build the envelope directly.
        wing.setdefault('k_eff', 0.5 * HARNESS_CONSTANTS['RHO'] * wing['S_DEV'] * wing['CL0'])

        extra = {k: v for k, v in self.meta.items()
                 if k not in ('car', 'wing', 'harness')}

        doc = {
            'schema': 'carsim-telemetry-1',
            'csv': os.path.basename(self.path),
            'columns': TELEMETRY_COLUMNS,
            'n_columns': N_COLUMNS,
            'units': COLUMN_UNITS,
            'string_columns': list(STRING_COLUMNS),
            'int_columns': list(INT_COLUMNS),
            'bool_columns': list(BOOL_COLUMNS),
            'precision': self.precision,
            'telem_hz': self.hz,
            'stride_steps': self.stride,
            'dt_phys': self.dt,
            'rows': self.rows_written,
            'marks': [{'row': r, 't': t, 'label': lab} for r, t, lab in self.marks],
            'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'car': car_dump,
            'tyre_file': self.meta.get('tyre_file', DEFAULT_TYRE_FILE),
            'wing': wing,
            'harness': _harness_constants(self.meta.get('harness')),
        }
        doc.update(extra)
        with open(self.sidecar_path, 'w') as fh:
            json.dump(doc, fh, indent=2, sort_keys=False, default=str)


def run_path(track, root='runs', tag='', ext='.csv'):
    """runs/<track>[_<tag>]_<YYYYmmdd_HHMMSS>.csv, per the spec's naming rule.

    The wall clock is read HERE and nowhere near the physics path -- it names a
    file, it never enters a state derivative.
    """
    stamp = time.strftime('%Y%m%d_%H%M%S')
    name = f'{track}_{tag}_{stamp}{ext}' if tag else f'{track}_{stamp}{ext}'
    return os.path.join(root, name)


def read_sidecar(csv_path):
    """Load the sidecar for a CSV, or {} if it is missing (older runs)."""
    base, _ = os.path.splitext(os.fspath(csv_path))
    try:
        with open(base + '.json') as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


# ----------------------------------------------------------- test fixture --
def synthetic_run(n_steps=8000, dt=0.001, R=50.0, V0=20.57, dV=1.2,
                  wing='fin', t_wing=2.0, ccw=True):
    """A plausible skidpad lap, as (step_index, row_dict, mark_or_None).

    EXISTS ONLY BECAUSE vehicle.py IS NOT WRITTEN YET. telemetry.py and
    plots.py have to be provable before there is a car to log, so this
    generates a car-shaped signal: a constant-radius lap at V0 with a slow
    speed oscillation, real load transfer, real per-corner loads, and axle
    utilisations computed with qss.fy_max itself (never a local mu(Fz), for
    exactly the reason the spec gives for the HUD). Delete nothing here when
    vehicle.py lands -- the self-checks still use it as a fixture that cannot
    regress with the physics.
    """
    from math import atan2, cos, degrees, pi, sin, sqrt
    from corsa_c import CorsaC, G, RHO
    from qss import TYRE, fy_max

    car = CorsaC()
    m, h_cg, L = car.m, car.h_cg, car.L
    t_bar = car.t
    a_cg, b_cg = car.a, car.b
    W = m * G
    fz_stat_f = W * car.wdist_f / 2.0
    fz_stat_r = W * (1.0 - car.wdist_f) / 2.0
    roll_dist_f = 0.74            # CALIBRATION constant (contract s4), not derived
    x_w = HARNESS_CONSTANTS['X_W']
    S_dev = HARNESS_CONSTANTS['S_DEV']
    cl = {'off': 0.0, 'fin': HARNESS_CONSTANTS['CL_FIN'],
          'plate': HARNESS_CONSTANTS['CL_PLATE']}.get(wing, 0.0)
    k_dev = 0.5 * RHO * S_dev * cl
    T_dep = HARNESS_CONSTANTS['T_DEPLOY']
    sgn = 1.0 if ccw else -1.0    # CCW = left turn = +ve ay, +ve r
    cx, cy = 200.0, 200.0
    track_len = 2.0 * pi * R
    gear_ratio = car.gear[2] * car.finaldrive      # 3rd; 20.6 m/s sits mid-range

    # Cornering stiffness per wheel used only to make alpha look like alpha:
    # ~900 N/deg gives 2-4 deg at the limit, the right order for a road tyre.
    c_alpha = 900.0

    theta = 0.0
    s = 0.0
    lap = 0
    lap_t0 = 0.0
    V_prev = V0
    for i in range(n_steps):
        t = i * dt                                  # never accumulated
        V = V0 + dV * sin(2.0 * pi * t / 8.0)
        ax = (V - V_prev) / dt if i else 0.0
        V_prev = V
        ay = sgn * V * V / R

        theta += sgn * V / R * dt
        s = (s + V * dt) % track_len
        if i and s < V * dt * 1.5 and t - lap_t0 > 3.0:
            lap += 1
            lap_t0 = t
        x = cx + R * cos(theta)
        y = cy + R * sin(theta)
        psi = theta + sgn * pi / 2.0
        r_yaw = sgn * V / R
        beta = -sgn * 0.045                          # limit beta is NEGATIVE in a left turn
        u = V * cos(beta)
        v = V * sin(beta)

        dep = 0.0
        if wing != 'off' and t > t_wing:
            d = min(1.0, (t - t_wing) / T_dep)
            dep = d * d * (3.0 - 2.0 * d)            # smoothstep -> C1 force (V19)
        F_dev = sgn * dep * k_dev * V * V
        D_dev = dep * k_dev * V * V / HARNESS_CONSTANTS['LD_DEV']

        # Load transfer, algebraically as the contract states it.
        Fy_tyre_tot = m * ay - F_dev
        dFz_tot = (Fy_tyre_tot * h_cg + F_dev * (h_cg - HARNESS_CONSTANTS['H_W'])) / t_bar
        dFz_f = roll_dist_f * dFz_tot
        dFz_r = (1.0 - roll_dist_f) * dFz_tot
        dFz_x = m * ax * h_cg / L
        Fz = [max(fz_stat_f - dFz_x / 2.0 - dFz_f, 0.0),
              max(fz_stat_f - dFz_x / 2.0 + dFz_f, 0.0),
              max(fz_stat_r + dFz_x / 2.0 - dFz_r, 0.0),
              max(fz_stat_r + dFz_x / 2.0 + dFz_r, 0.0)]

        # Axle demands from the same moment balance qss uses.
        Y_f = (b_cg * m * ay - F_dev * (b_cg + x_w)) / L
        Y_r = (a_cg * m * ay + F_dev * x_w) / L
        cap = [fy_max(fz, mu_scale=1.0, **TYRE) for fz in Fz]
        cap_f = max(cap[0] + cap[1], 1.0)
        cap_r = max(cap[2] + cap[3], 1.0)
        Fy = [Y_f * cap[0] / cap_f, Y_f * cap[1] / cap_f,
              Y_r * cap[2] / cap_r, Y_r * cap[3] / cap_r]

        drag = 0.5 * RHO * car.CdA * V * V + D_dev
        Fx_tot = m * ax + drag + car.Crr * W
        Fx = [Fx_tot / 2.0, Fx_tot / 2.0, 0.0, 0.0]           # FWD

        # +ve alpha gives Fy < 0 (contract s0), so alpha and Fy have opposite signs.
        alpha_deg = [-fy / c_alpha for fy in Fy]              # deg, ~ -2.5 at the limit
        kappa = [0.012, 0.012, 0.0, 0.0]
        omega = [V / car.r_roll * (1.0 + kappa[j]) for j in range(4)]
        rpm = omega[0] * gear_ratio * 60.0 / (2.0 * pi)

        util_f = abs(Fy[0] + Fy[1]) / cap_f
        util_r = abs(Fy[2] + Fy[3]) / cap_r
        limited = 'FRONT' if util_f >= util_r else 'REAR'

        delta_deg = degrees(atan2(L, R)) + 3.2 * (abs(ay) / G) * (1.0 if ccw else -1.0)
        delta_deg *= sgn

        n_off = 0.30 * sin(2.0 * pi * t / 3.0)
        sector = 1 + int(3.0 * s / track_len)

        mark = None
        if i == 0:
            mark = 'start'
        elif wing != 'off' and abs(t - t_wing) < dt / 2.0:
            mark = f'wing_{wing}_deploy'

        row = {
            't': t, 'x': x, 'y': y, 'psi': psi, 'u': u, 'v': v, 'r': r_yaw, 'V': V,
            'beta_deg': degrees(beta), 'ax': ax, 'ay': ay, 'ax_g': ax / G, 'ay_g': ay / G,
            'gear': 3, 'rpm': rpm, 'clutch_eng': True,
            'throttle': 0.42, 'brake': 0.0, 'clutch': 0.0, 'handbrake': 0.0,
            'delta_road_deg': delta_deg, 'steer_in': delta_deg / 32.625,
            'Fz_fl': Fz[0], 'Fz_fr': Fz[1], 'Fz_rl': Fz[2], 'Fz_rr': Fz[3],
            'Fx_fl': Fx[0], 'Fx_fr': Fx[1], 'Fx_rl': Fx[2], 'Fx_rr': Fx[3],
            'Fy_fl': Fy[0], 'Fy_fr': Fy[1], 'Fy_rl': Fy[2], 'Fy_rr': Fy[3],
            'alpha_fl_deg': alpha_deg[0], 'alpha_fr_deg': alpha_deg[1],
            'alpha_rl_deg': alpha_deg[2], 'alpha_rr_deg': alpha_deg[3],
            'kappa_fl': kappa[0], 'kappa_fr': kappa[1],
            'kappa_rl': kappa[2], 'kappa_rr': kappa[3],
            'mu_fl': 1.0, 'mu_fr': 1.0, 'mu_rl': 1.0, 'mu_rr': 1.0,
            'omega_fl': omega[0], 'omega_fr': omega[1],
            'omega_rl': omega[2], 'omega_rr': omega[3],
            'util_f': util_f, 'util_r': util_r, 'limited_by': limited,
            'wing_deploy': dep, 'wing_side': int(sgn) if dep > 0.0 else 0,
            'F_wing': F_dev, 'D_wing': D_dev,
            's': s, 'n': n_off, 'kappa_track': sgn / R, 'on_track': True,
            'lap': lap, 'sector': sector, 'lap_time': t - lap_t0,
            'event': '',
        }
        yield i, row, mark


# --------------------------------------------------------------- self test --
def self_check(tmpdir=None, verbose=True):
    """python3 -m drive.telemetry -- writes, re-reads and verifies a run."""
    import math

    root = tmpdir or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  os.pardir, 'runs', 'selfcheck')
    root = os.path.abspath(root)
    ok = True
    results = []

    def check(name, cond, detail=''):
        nonlocal ok
        ok = ok and bool(cond)
        results.append((name, bool(cond), detail))
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name:38s} {detail}")

    for precision in ('6g', 'full'):
        path = os.path.join(root, f'skidpad_{precision}.csv')
        meta = dict(track='skidpad', dt=0.001, tyre_file=DEFAULT_TYRE_FILE,
                    wing={'wing': 'fin', 'x_w': 0.97, 'h_w': 0.90},
                    script='synthetic_run', radius=50.0)
        tel = TelemetryWriter(path, hz=100, precision=precision, meta=meta)
        for i, row, mark in synthetic_run(n_steps=8000, dt=0.001):
            if mark:
                tel.mark(mark)
            tel.maybe_log(i, row)
        tel.close()

        size = os.path.getsize(path)
        with open(path, newline='') as fh:
            rows = list(csv.reader(fh))
        header, body = rows[0], rows[1:]

        if verbose:
            print(f"\n{precision}: {path}")
            print(f"  {size} bytes csv, {os.path.getsize(tel.sidecar_path)} bytes json, "
                  f"{len(body)} rows x {len(header)} cols")

        check(f'{precision}: header round-trips', header == TELEMETRY_COLUMNS,
              f'{len(header)} columns')
        check(f'{precision}: every row is {N_COLUMNS} wide',
              all(len(r) == N_COLUMNS for r in body))
        # 8000 steps at stride 10 gives 800 rows, +1 because the mark at the
        # wing deploy (t = 2.0 s, step 2000 -> on-stride) does not add one, but
        # marks placed off-stride would.
        check(f'{precision}: row count == steps/stride',
              len(body) in (800, 801), f'{len(body)} rows, stride {tel.stride}')

        ev = header.index('event')
        events = {r[ev] for r in body if r[ev]}
        check(f'{precision}: event column carries marks',
              events == {'start', 'wing_fin_deploy'}, str(sorted(events)))
        check(f'{precision}: sidecar marks recorded', len(tel.marks) == 2,
              f'{[m[2] for m in tel.marks]}')

        # numeric columns must all parse, and NaN must survive as NaN
        bad = []
        for name in TELEMETRY_COLUMNS:
            if name in STRING_COLUMNS:
                continue
            j = header.index(name)
            try:
                [float(r[j]) for r in body]
            except ValueError:
                bad.append(name)
        check(f'{precision}: all numeric columns parse', not bad, str(bad))

        if precision == 'full':
            # THE point of precision='full': re-reading and re-formatting must
            # reproduce the file byte-for-byte, or the determinism test V20 is
            # comparing rounded values and passes vacuously.
            identical = True
            for r in body:
                for name, cell in zip(TELEMETRY_COLUMNS, r):
                    if name in STRING_COLUMNS or name in INT_COLUMNS or name in BOOL_COLUMNS:
                        continue
                    if repr(float(cell)) != cell:
                        identical = False
                        break
                if not identical:
                    break
            check('full: repr re-reads bit-identically', identical)

            # and the whole file, written twice from the same fixture, is equal
            p2 = os.path.join(root, 'skidpad_full_repeat.csv')
            t2 = TelemetryWriter(p2, hz=100, precision='full', meta=meta)
            for i, row, mark in synthetic_run(n_steps=8000, dt=0.001):
                if mark:
                    t2.mark(mark)
                t2.maybe_log(i, row)
            t2.close()
            with open(path, 'rb') as f1, open(p2, 'rb') as f2:
                same = f1.read() == f2.read()
            check('full: two identical runs are byte-equal', same)

        side = read_sidecar(path)
        need = ('columns', 'units', 'car', 'tyre_file', 'wing', 'harness',
                'dt_phys', 'precision', 'marks')
        missing = [k for k in need if k not in side]
        check(f'{precision}: sidecar has every required key', not missing, str(missing))
        check(f'{precision}: sidecar units cover every column',
              set(side.get('units', {})) == set(TELEMETRY_COLUMNS))
        check(f'{precision}: sidecar car dump is CorsaC',
              side.get('car', {}).get('m') == 1010.0 and 'h_cg' in side.get('car', {}))
        check(f'{precision}: sidecar carries DT_PHYS',
              side.get('harness', {}).get('DT_PHYS') == 0.001)

    # missing keys and NaN handling
    p3 = os.path.join(root, 'sparse.csv')
    t3 = TelemetryWriter(p3, hz=100, precision='6g', meta={'dt': 0.001})
    t3.maybe_log(0, {'t': 0.0, 'V': 12.0})               # everything else absent
    t3.maybe_log(10, {'t': 0.01, 'V': float('nan'), 'limited_by': 'POWER'})
    t3.close()
    with open(p3, newline='') as fh:
        srows = list(csv.reader(fh))
    check('absent fields become nan / empty',
          len(srows) == 3 and len(srows[1]) == N_COLUMNS
          and srows[1][TELEMETRY_COLUMNS.index('gear')] == 'nan'
          and srows[1][TELEMETRY_COLUMNS.index('event')] == '')
    check('nan survives the round trip',
          math.isnan(float(srows[2][TELEMETRY_COLUMNS.index('V')])))

    print(f"\nTELEMETRY_COLUMNS: {N_COLUMNS} columns "
          f"(specs/harness.txt calls the same literal list '63' -- see module docstring)")
    print('telemetry self_check: ' + ('PASS' if ok else 'FAIL'))
    return ok


if __name__ == '__main__':
    raise SystemExit(0 if self_check() else 1)
