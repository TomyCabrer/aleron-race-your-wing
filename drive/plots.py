"""Post-run plots for the flank-wing driving sim. Agg only, csv + numpy only.

Why this file exists at all
---------------------------
The device is worth 0.6-1.5% of corner speed. That is 62.9 N against a 9908 N
car at R=50 -- below the resolution of a speedometer, a seat, or any single
lap time a human will produce on a keyboard. `compare(by='distance')` is the
one view that resolves it: interpolate t(s) for two runs onto a common s grid
and plot t_a(s) - t_b(s). A 1% corner-speed gain held through 45% of a 1249 m
lap shows up as a monotone downward staircase of a few hundred milliseconds,
and the staircase's STEPS land in the corners, which is the qualitative claim
the whole study makes. Everything else in this module is instrumentation for
finding out why a run went wrong.

Two hard constraints from the spec, both learned the expensive way:

  * `matplotlib.use('Agg')` BEFORE pyplot is imported. Not after, not inside a
    function -- importing pyplot first binds the macOS backend and a headless
    run hangs or dies in the window server.
  * No pandas. It is not in the verified dependency list on this machine and
    importing it emits NumPy 1.x/2.x ABI warnings. `csv` plus numpy is enough
    for a fixed 65-column schema whose column order is contractual.
"""

import csv
import math
import os

import matplotlib
matplotlib.use('Agg')          # MUST precede the pyplot import. See docstring.
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from drive.telemetry import STRING_COLUMNS, TELEMETRY_COLUMNS, read_sidecar

DPI = 150
G = 9.81

# One colour per corner, used identically in every figure so a reader learns it
# once. FL/FR/RL/RR is the contract's wheel order everywhere.
WHEEL_COLOURS = ('#4fa3ff', '#e2523f', '#4ec26a', '#d9ce55')
WHEEL_NAMES = ('FL', 'FR', 'RL', 'RR')


# ------------------------------------------------------------------- io ----
def read_run(csv_path):
    """(data, meta): column name -> numpy array, plus the sidecar dict.

    Numeric columns come back as float64 with '' -> nan; the two text columns
    come back as an object array of str. Reading is by NAME but the file's
    header is verified against TELEMETRY_COLUMNS, because the column order is
    part of the contract and a silent reordering is exactly the kind of bug
    that produces a plausible-looking wrong plot.
    """
    path = os.fspath(csv_path)
    with open(path, newline='') as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise ValueError(f'{path}: empty telemetry file')
    header, body = rows[0], rows[1:]
    if header != TELEMETRY_COLUMNS:
        extra = set(header) - set(TELEMETRY_COLUMNS)
        missing = set(TELEMETRY_COLUMNS) - set(header)
        raise ValueError(f'{path}: header is not the contract schema '
                         f'(missing {sorted(missing)}, extra {sorted(extra)})')
    if not body:
        raise ValueError(f'{path}: header only, no rows')

    cols = list(zip(*body))
    data = {}
    for j, name in enumerate(header):
        if name in STRING_COLUMNS:
            data[name] = np.array(cols[j], dtype=object)
        else:
            data[name] = np.array([float(c) if c else math.nan for c in cols[j]])
    return data, read_sidecar(path)


def _out_path(csv_path, out, suffix):
    """PNG next to the CSV unless the caller names one."""
    if out:
        return os.fspath(out)
    base, _ = os.path.splitext(os.fspath(csv_path))
    return f'{base}_{suffix}.png'


def _save(fig, path):
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    return path


def _run_label(csv_path, meta):
    wing = (meta.get('wing') or {}).get('wing', '?')
    track = meta.get('track', '?')
    return f'{os.path.basename(csv_path)}  [track={track} wing={wing}]'


def _mark_lines(ax, data, y=None):
    """Vertical dashed lines wherever mark() wrote an event label."""
    ev = data.get('event')
    if ev is None:
        return
    for t, label in zip(data['t'], ev):
        if isinstance(label, str) and label:
            ax.axvline(t, color='#8b9099', lw=0.8, ls='--', zorder=0)
            ax.annotate(label, (t, ax.get_ylim()[1]), fontsize=6, rotation=90,
                        va='top', ha='right', color='#8b9099')


# -------------------------------------------------------------- overview ---
def overview(csv_path, out=None):
    """Six panels: speed, accelerations, driver inputs, loads, utilisation, attitude."""
    data, meta = read_run(csv_path)
    t = data['t']
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 10.0), sharex=True)
    fig.suptitle('overview   ' + _run_label(csv_path, meta), fontsize=10)

    ax = axes[0][0]
    ax.plot(t, data['V'] * 3.6, color='#d9ce55', lw=1.0)
    ax.set_ylabel('V  [km/h]')
    ax.grid(alpha=0.25)
    _mark_lines(ax, data)

    ax = axes[0][1]
    ax.plot(t, data['ay_g'], color='#4fa3ff', lw=0.9, label='a_y')
    ax.plot(t, data['ax_g'], color='#6fd08c', lw=0.9, label='a_x')
    # 0.8625 g is qss.max_ay with k=0 -- the car's own quasi-steady lateral
    # envelope, and the number every transient here should be judged against.
    for s in (+0.8624671, -0.8624671):
        ax.axhline(s, color='#8b9099', lw=0.7, ls=':')
    ax.set_ylabel('acceleration  [g]')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(alpha=0.25)

    ax = axes[1][0]
    ax.plot(t, data['throttle'], color='#4ec26a', lw=0.9, label='throttle')
    ax.plot(t, data['brake'], color='#e2523f', lw=0.9, label='brake')
    ax.plot(t, data['clutch'], color='#8b9099', lw=0.7, label='clutch')
    ax.set_ylabel('pedals  [0..1]')
    ax.set_ylim(-0.05, 1.05)
    ax2 = ax.twinx()
    ax2.plot(t, data['delta_road_deg'], color='#d9ce55', lw=0.9)
    ax2.set_ylabel('road-wheel angle  [deg]', color='#a08f2a')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(alpha=0.25)

    ax = axes[1][1]
    for name, c, lab in zip(('Fz_fl', 'Fz_fr', 'Fz_rl', 'Fz_rr'),
                            WHEEL_COLOURS, WHEEL_NAMES):
        ax.plot(t, data[name], color=c, lw=0.8, label=lab)
    ax.set_ylabel('F_z  [N]')
    ax.legend(fontsize=7, ncol=4, loc='upper right')
    ax.grid(alpha=0.25)

    ax = axes[2][0]
    ax.plot(t, data['util_f'], color='#4fa3ff', lw=0.9, label='util_f')
    ax.plot(t, data['util_r'], color='#e2523f', lw=0.9, label='util_r')
    ax.axhline(1.0, color='#8b9099', lw=0.9, ls='--')      # the limit, by definition
    ax.set_ylabel('axle utilisation  [-]')
    ax.set_xlabel('t  [s]')
    # 1.25 unless the run actually went past it: a transient overshoot of the
    # quasi-steady limit is the interesting event on this panel and must never
    # be silently clipped off the top.
    u_hi = np.nanmax(np.concatenate([data['util_f'], data['util_r'], [1.2]]))
    ax.set_ylim(0.0, max(1.25, u_hi * 1.05))
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(alpha=0.25)

    ax = axes[2][1]
    ax.plot(t, data['beta_deg'], color='#b04ee0', lw=0.9, label='beta')
    ax.plot(t, np.degrees(data['r']), color='#ffb545', lw=0.9, label='yaw rate')
    ax.set_ylabel('beta [deg] / yaw rate [deg/s]')
    ax.set_xlabel('t  [s]')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, _out_path(csv_path, out, 'overview'))


# -------------------------------------------------------------------- gg ---
def gg_envelope(V, mu_scale=1.0, k_eff=0.0, x_w=0.97, h_w=0.90, n=181):
    """Equation 18, as an (M,2) closed curve in (ay_g, ax_g).

    qss.max_ay bisects 200 times per call, so this is called ONCE per figure at
    a representative speed, never per point. The lateral half-width is the
    car's own quasi-steady limit -- the same function the HUD and the ledger
    use -- so a scatter point outside it is a genuine transient overshoot and
    not a difference of model.
    """
    import qss
    from corsa_c import CorsaC, RHO

    car = CorsaC()
    ay_env = qss.max_ay(V, k=k_eff, x_w=x_w, h_w=h_w, mu_scale=mu_scale)
    drag = 0.5 * RHO * car.CdA * V * V + car.Crr * car.m * G
    ax_pow = min(car.P_wheel / (car.m * max(V, 3.0)), 0.90 * ay_env)

    ay = np.linspace(-ay_env, ay_env, n)
    frac = np.sqrt(np.maximum(0.0, 1.0 - (ay / ay_env) ** 2))
    ax_acc = ax_pow * frac - drag / car.m
    ax_brk = -(ay_env * frac + drag / car.m)
    curve = np.concatenate([np.column_stack([ay, ax_acc]),
                            np.column_stack([ay[::-1], ax_brk[::-1]]),
                            np.column_stack([ay[:1], ax_acc[:1]])])
    return curve / G


def gg(csv_path, out=None):
    """a_y vs a_x scatter coloured by speed, with the qss envelope overlaid."""
    data, meta = read_run(csv_path)
    wing = meta.get('wing') or {}
    k_eff = float(wing.get('k_eff', 0.0))
    x_w = float(wing.get('x_w', 0.97))
    h_w = float(wing.get('h_w', 0.90))
    mu = np.nanmean(np.column_stack([data['mu_fl'], data['mu_fr'],
                                     data['mu_rl'], data['mu_rr']]))
    if not np.isfinite(mu):
        mu = 1.0

    V = data['V']
    fig, ax = plt.subplots(figsize=(7.4, 7.4))
    sc = ax.scatter(data['ay_g'], data['ax_g'], c=V * 3.6, s=4, cmap='viridis',
                    alpha=0.75, linewidths=0)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label('V  [km/h]')

    # Envelope at the median speed of the run, plus the two extremes faintly:
    # the envelope's lateral half-width is speed independent with no device
    # fitted, so a visible spread between the three is the wing doing work.
    v_med = float(np.nanmedian(V))
    for v, alpha, lw, lab in ((v_med, 1.0, 1.2, f'qss envelope @ {v_med:.1f} m/s'),
                              (float(np.nanpercentile(V, 5)), 0.35, 0.8, None),
                              (float(np.nanpercentile(V, 95)), 0.35, 0.8, None)):
        if not np.isfinite(v) or v <= 0.5:
            continue
        c = gg_envelope(v, mu_scale=mu, k_eff=k_eff, x_w=x_w, h_w=h_w)
        ax.plot(c[:, 0], c[:, 1], color='#5a6068', lw=lw, alpha=alpha, label=lab)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.axhline(0, color='#8b9099', lw=0.6)
    ax.axvline(0, color='#8b9099', lw=0.6)
    ax.set_xlabel('a_y  [g]   (+ve = LEFT)')
    ax.set_ylabel('a_x  [g]   (+ve = forward)')
    ax.set_title('g-g   ' + _run_label(csv_path, meta), fontsize=9)
    ax.legend(fontsize=7, loc='lower left')
    ax.grid(alpha=0.25)
    return _save(fig, _out_path(csv_path, out, 'gg'))


# ------------------------------------------------------------- track map ---
def track_map(csv_path, track_name=None, out=None):
    """The driven path, coloured by speed, over the track edges when available.

    drive/track.py is a separate module in the contract's map; if it is not
    importable yet (or the track name is unknown) the path is still drawn on
    its own, because a run's trajectory is diagnostic without any scenery.
    """
    data, meta = read_run(csv_path)
    if track_name is None:
        track_name = meta.get('track')

    fig, ax = plt.subplots(figsize=(9.5, 8.0))
    drew_track = False
    try:
        import inspect

        from drive import track as trackmod
        factory = trackmod.TRACKS[track_name]
        # A skidpad run at R=30.48 must not be drawn over the R=50 default
        # circle. Pass only the kwargs the factory actually declares, so this
        # never breaks when track.py's signatures change.
        params = inspect.signature(factory).parameters
        kwargs = {k: meta[k] for k in ('radius', 'cw') if k in params and k in meta}
        tr = factory(**kwargs)
        ax.plot(tr.left[:, 0], tr.left[:, 1], color='#8b9099', lw=0.8)
        ax.plot(tr.right[:, 0], tr.right[:, 1], color='#8b9099', lw=0.8)
        ax.plot(tr.xy[:, 0], tr.xy[:, 1], color='#6a6f76', lw=0.5, ls='--')
        drew_track = True
    except Exception as exc:                 # module absent, or name unknown
        ax.set_title(f'(track outline unavailable: {type(exc).__name__}) ',
                     loc='right', fontsize=7, color='#8b9099')

    xy = np.column_stack([data['x'], data['y']])
    segs = np.stack([xy[:-1], xy[1:]], axis=1)
    V = data['V']
    lc = LineCollection(segs, cmap='viridis', linewidths=1.8)
    lc.set_array(0.5 * (V[:-1] + V[1:]) * 3.6)
    ax.add_collection(lc)
    cb = fig.colorbar(lc, ax=ax, fraction=0.040, pad=0.03)
    cb.set_label('V  [km/h]')

    ax.plot(xy[0, 0], xy[0, 1], 'o', color='#4ec26a', ms=6, label='start')
    ev = data.get('event')
    if ev is not None:
        for x, y, label in zip(data['x'], data['y'], ev):
            if isinstance(label, str) and label:
                ax.plot(x, y, 'x', color='#ff8c2b', ms=6)
                ax.annotate(label, (x, y), fontsize=6, color='#ff8c2b',
                            xytext=(4, 4), textcoords='offset points')

    ax.set_aspect('equal')
    ax.autoscale_view()
    ax.set_xlabel('x  [m]')
    ax.set_ylabel('y  [m]')
    ax.set_title(f'track map ({track_name})   ' + _run_label(csv_path, meta),
                 fontsize=9)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(alpha=0.2)
    tag = 'trackmap' if drew_track else 'path'
    return _save(fig, _out_path(csv_path, out, tag))


# ------------------------------------------------------------------ laps ---
def _lap_times(data):
    """[(lap_number, lap_time_s)] from the lap counter and t, not from lap_time.

    lap_time in the schema is the RUNNING time in the current lap, so its last
    sample before a lap increments is short by up to one telemetry period.
    Differencing t at the increment is accurate to the same period and does not
    accumulate.
    """
    lap = data['lap']
    t = data['t']
    idx = np.flatnonzero(np.diff(lap) > 0) + 1
    out = []
    for a, b in zip(idx[:-1], idx[1:]):
        out.append((int(lap[a]), float(t[b] - t[a])))
    return out


def laps(csv_path, out=None):
    """Per-lap speed traces against distance, plus the lap-time bars."""
    data, meta = read_run(csv_path)
    lap = data['lap']
    s = data['s']
    V = data['V']

    fig, axes = plt.subplots(2, 1, figsize=(12.0, 8.0),
                             gridspec_kw={'height_ratios': [2, 1]})
    ax = axes[0]
    laps_present = [int(l) for l in np.unique(lap[np.isfinite(lap)])]
    cmap = plt.get_cmap('viridis')
    for i, ln in enumerate(laps_present):
        m = lap == ln
        if m.sum() < 5:
            continue
        c = cmap(i / max(1, len(laps_present) - 1))
        ax.plot(s[m], V[m] * 3.6, lw=1.0, color=c, label=f'lap {ln}')
    ax.set_xlabel('s  [m]')
    ax.set_ylabel('V  [km/h]')
    ax.legend(fontsize=7, ncol=6, loc='lower right')
    ax.grid(alpha=0.25)
    ax.set_title('laps   ' + _run_label(csv_path, meta), fontsize=9)

    ax = axes[1]
    lt = _lap_times(data)
    if lt:
        nums = [n for n, _ in lt]
        times = [v for _, v in lt]
        best = min(times)
        colours = ['#b04ee0' if abs(v - best) < 1e-9 else '#4fa3ff' for v in times]
        xs = np.arange(len(times))
        # Explicit positions and a bounded width: a categorical bar() with one
        # lap otherwise stretches that single bar across the whole axis and
        # reads as a filled panel rather than as a lap time.
        ax.bar(xs, times, width=min(0.6, 0.9), color=colours)
        for x, v in zip(xs, times):
            ax.annotate(f'{v:.3f}', (x, v), ha='center', va='bottom', fontsize=7)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(n) for n in nums])
        # Keep at least six lap-slots of axis so a single completed lap draws as
        # a bar and not as a filled panel.
        span = max(len(times), 6)
        pad = 0.5 * (span - len(times))
        ax.set_xlim(-0.5 - pad, len(times) - 0.5 + pad)
        ax.set_ylabel('lap time  [s]')
        ax.set_xlabel('lap')
        ax.set_ylim(0, max(times) * 1.15)
    else:
        ax.text(0.5, 0.5, 'no completed laps in this run', ha='center',
                va='center', transform=ax.transAxes, fontsize=9, color='#8b9099')
        ax.set_axis_off()
    ax.grid(alpha=0.25, axis='y')

    fig.tight_layout()
    return _save(fig, _out_path(csv_path, out, 'laps'))


# --------------------------------------------------------------- compare ---
def _monotone_distance(data):
    """(distance, t, idx): s unwrapped past the lap discontinuity, made monotone.

    On a closed track s wraps to 0 every lap, so t(s) is multi-valued and
    np.interp would silently produce nonsense on it.

    The wrap is closed with the travel the car actually did over that sample
    (V*dt), NOT with an estimated lap length. Estimating L as max(s) + a
    typical step is wrong by up to one sample of travel (0.2 m at 100 Hz and
    20 m/s), which injects ~10 ms per lap into the delta-time trace -- 4% of
    the ~250 ms the whole device is worth. That artifact is the same order as
    the signal, so it has to go. `idx` indexes back into the ORIGINAL rows so
    other columns can be interpolated on the same grid.
    """
    s = np.array(data['s'], dtype=float)
    t = np.array(data['t'], dtype=float)
    V = np.array(data['V'], dtype=float)
    good = np.isfinite(s) & np.isfinite(t)
    idx = np.flatnonzero(good)
    s, t, V = s[idx], t[idx], V[idx]
    if s.size < 2:
        raise ValueError('not enough samples with a finite s to compare by distance')

    ds = np.diff(s)
    span = float(np.nanmax(s) - np.nanmin(s))
    wrap = ds < -0.5 * max(span, 1e-9)
    if wrap.any():
        v_mid = 0.5 * (V[:-1] + V[1:])
        travel = v_mid * np.diff(t)
        fallback = span + (float(np.median(ds[ds > 0])) if (ds > 0).any() else 0.0)
        ds = np.where(wrap, np.where(np.isfinite(travel) & (travel > 0),
                                     travel, ds + fallback), ds)
    dist = np.concatenate([[0.0], np.cumsum(ds)])

    keep = np.concatenate([[True], np.diff(dist) > 1e-9])
    return dist[keep], t[keep], idx[keep]


def compare(csv_a, csv_b, by='distance', out=None):
    """t_a(s) - t_b(s): THE view that resolves a 0.6-1.5% corner-speed gain.

    Both runs are interpolated onto a common distance grid and zeroed at the
    grid start, so the plot reads as "how far ahead is A, in seconds, at this
    point on the road". A device that buys corner speed shows a staircase whose
    steps sit in the corners and whose treads are flat on the straights; a
    device that only adds drag shows the mirror image. Nothing else in this
    module can tell those two apart.

    by='time' falls back to a common-time grid and plots the speed and distance
    difference instead -- useful for an A/B of transient response, useless for
    a lap-time claim.
    """
    da, ma = read_run(csv_a)
    db, mb = read_run(csv_b)
    lab_a = _run_label(csv_a, ma)
    lab_b = _run_label(csv_b, mb)

    fig, axes = plt.subplots(3, 1, figsize=(12.5, 9.0), sharex=True)

    if by == 'distance':
        sa, ta, ia = _monotone_distance(da)
        sb, tb, ib = _monotone_distance(db)
        s_end = min(sa[-1], sb[-1])
        if s_end <= 0:
            raise ValueError('the two runs share no distance overlap')
        grid = np.linspace(0.0, s_end, 2000)
        t_a = np.interp(grid, sa, ta)
        t_b = np.interp(grid, sb, tb)
        dt = (t_a - t_a[0]) - (t_b - t_b[0])

        axes[0].plot(grid, dt * 1000.0, color='#ff8c2b', lw=1.2)
        axes[0].axhline(0.0, color='#8b9099', lw=0.8, ls='--')
        axes[0].set_ylabel('t_A - t_B  [ms]')
        axes[0].set_title(f'compare by distance\n  A = {lab_a}\n  B = {lab_b}',
                          fontsize=8, loc='left')
        axes[0].annotate(f'end: {dt[-1]*1000:+.1f} ms over {s_end:.1f} m',
                         (grid[-1], dt[-1] * 1000.0), fontsize=8,
                         ha='right', va='bottom', color='#ff8c2b')
        axes[0].grid(alpha=0.25)

        va = np.interp(grid, sa, np.asarray(da['V'])[ia])
        vb = np.interp(grid, sb, np.asarray(db['V'])[ib])
        axes[1].plot(grid, va * 3.6, color='#4fa3ff', lw=0.9, label='A')
        axes[1].plot(grid, vb * 3.6, color='#e2523f', lw=0.9, label='B')
        axes[1].set_ylabel('V  [km/h]')
        axes[1].legend(fontsize=7, loc='lower right')
        axes[1].grid(alpha=0.25)

        with np.errstate(divide='ignore', invalid='ignore'):
            dv_pct = 100.0 * (va - vb) / np.where(vb > 0.1, vb, np.nan)
        axes[2].plot(grid, dv_pct, color='#4ec26a', lw=0.9)
        axes[2].axhline(0.0, color='#8b9099', lw=0.8, ls='--')
        axes[2].set_ylabel('dV/V  [%]')
        axes[2].set_xlabel('s  [m]')
        axes[2].grid(alpha=0.25)
        # The band the study lives in. If the trace does not sit inside it,
        # something other than the device moved.
        for y in (0.6, 1.5):
            axes[2].axhline(y, color='#b04ee0', lw=0.6, ls=':')

        summary = dict(mode='distance', s_overlap=float(s_end),
                       dt_end_s=float(dt[-1]),
                       dv_pct_mean=float(np.nanmean(dv_pct)),
                       dv_pct_max=float(np.nanmax(dv_pct)))
    elif by == 'time':
        t_end = min(float(da['t'][-1]), float(db['t'][-1]))
        grid = np.linspace(0.0, t_end, 2000)
        va = np.interp(grid, da['t'], da['V'])
        vb = np.interp(grid, db['t'], db['V'])
        axes[0].plot(grid, (va - vb) * 3.6, color='#ff8c2b', lw=1.2)
        axes[0].axhline(0.0, color='#8b9099', lw=0.8, ls='--')
        axes[0].set_ylabel('V_A - V_B  [km/h]')
        axes[0].set_title(f'compare by time\n  A = {lab_a}\n  B = {lab_b}',
                          fontsize=8, loc='left')
        axes[0].grid(alpha=0.25)
        axes[1].plot(grid, va * 3.6, color='#4fa3ff', lw=0.9, label='A')
        axes[1].plot(grid, vb * 3.6, color='#e2523f', lw=0.9, label='B')
        axes[1].set_ylabel('V  [km/h]')
        axes[1].legend(fontsize=7, loc='lower right')
        axes[1].grid(alpha=0.25)
        axes[2].plot(grid, np.interp(grid, da['t'], da['ay_g']), color='#4fa3ff', lw=0.8)
        axes[2].plot(grid, np.interp(grid, db['t'], db['ay_g']), color='#e2523f', lw=0.8)
        axes[2].set_ylabel('a_y  [g]')
        axes[2].set_xlabel('t  [s]')
        axes[2].grid(alpha=0.25)
        summary = dict(mode='time', t_overlap=float(t_end))
    else:
        raise ValueError("by must be 'distance' or 'time'")

    fig.tight_layout()
    path = _save(fig, _out_path(csv_a, out, f'compare_{by}'))
    compare.last_summary = summary          # so a caller can assert on the number
    return path


compare.last_summary = {}


# ------------------------------------------------------------------- all ---
def all_plots(csv_path, track_name=None):
    """overview + gg + track map + laps for one run. Returns the paths."""
    return [overview(csv_path), gg(csv_path),
            track_map(csv_path, track_name), laps(csv_path)]


# ------------------------------------------------------------ self check ---
def self_check(tmpdir=None, verbose=True):
    """python3 -m drive.plots --self-check

    vehicle.py does not exist yet, so this writes two synthetic runs through
    the real TelemetryWriter (wing off / clean fin) and renders every figure
    from the resulting CSVs. It proves the whole path -- schema, formatting,
    re-read, interpolation, rendering -- without a car.
    """
    from drive.telemetry import TelemetryWriter, synthetic_run

    root = tmpdir or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  os.pardir, 'runs', 'selfcheck')
    root = os.path.abspath(root)
    ok = True

    def check(name, cond, detail=''):
        nonlocal ok
        ok = ok and bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name:40s} {detail}")

    # Two runs at the same radius: B has no device, A has the clean fin. The
    # fin is worth ~0.6% at R=50, so A is made 0.6% faster -- the fixture is
    # asserting that compare() can RESOLVE that, not that the physics produces it.
    runs = {}
    for tag, wing, v0 in (('A_fin', 'fin', 20.57 * 1.006), ('B_off', 'off', 20.57)):
        path = os.path.join(root, f'skidpad_{tag}.csv')
        tel = TelemetryWriter(path, hz=100, precision='6g',
                              meta=dict(track='skidpad', dt=0.001, radius=50.0,
                                        wing={'wing': wing, 'x_w': 0.97, 'h_w': 0.90},
                                        script='synthetic_run'))
        for i, row, mark in synthetic_run(n_steps=40000, dt=0.001, V0=v0,
                                          wing=wing, t_wing=2.0):
            if mark:
                tel.mark(mark)
            tel.maybe_log(i, row)
        tel.close()
        runs[tag] = path

    data, meta = read_run(runs['A_fin'])
    check('read_run recovers the full schema',
          set(data) == set(TELEMETRY_COLUMNS), f'{len(data)} columns')
    check('read_run finds the sidecar', meta.get('wing', {}).get('wing') == 'fin')
    check('event column survives the round trip',
          sorted({e for e in data['event'] if e}) == ['start', 'wing_fin_deploy'])

    paths = []
    paths += all_plots(runs['A_fin'], 'skidpad')
    paths.append(compare(runs['A_fin'], runs['B_off'], by='distance'))
    summary = dict(compare.last_summary)
    paths.append(compare(runs['A_fin'], runs['B_off'], by='time'))

    for p in paths:
        size = os.path.getsize(p) if os.path.exists(p) else 0
        check(os.path.basename(p), size > 20000, f'{size} bytes')

    # compare() must actually resolve the 0.6% the fixture built in: 0.6%
    # faster over ~800 m of lap is ~ -0.6% of the elapsed time, ~ -240 ms.
    dt_end = summary.get('dt_end_s', 0.0)
    dv = summary.get('dv_pct_mean', 0.0)
    check('compare resolves the 0.6% speed difference',
          0.4 < dv < 0.8, f'mean dV/V = {dv:+.3f}%')
    # And it must get the MAGNITUDE right, not just the sign. A run 0.6% faster
    # everywhere covers s in s/V * (0.006/1.006) less time; at s = 822.6 m and
    # V = 20.57 m/s that is -238.5 ms. The band below is +/-8 ms, which is
    # tighter than the ~10 ms/lap error that closing the lap wrap with an
    # ESTIMATED lap length would introduce -- that is the point of the test.
    s_over = summary.get('s_overlap', 0.0)
    want = -(s_over / 20.57) * (0.006 / 1.006)
    check('compare gets the delta-time magnitude right',
          abs(dt_end - want) < 0.008,
          f'dt at the end = {dt_end*1000:+.1f} ms vs {want*1000:+.1f} ms expected '
          f'over {s_over:.1f} m')

    if verbose:
        print('\nfigures:')
        for p in paths:
            print(f'  {os.path.getsize(p):8d} B  {p}')
        print(f'\ncsv: ' + ', '.join(f'{os.path.getsize(v)} B {os.path.basename(v)}'
                                     for v in runs.values()))
    print('plots self_check: ' + ('PASS' if ok else 'FAIL'))
    return ok


# ------------------------------------------------------------------ main ---
def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(
        prog='python3 -m drive.plots',
        description='Post-run plots for carsim telemetry (Agg, headless).')
    p.add_argument('cmd', nargs='?', default='self-check',
                   choices=['overview', 'gg', 'track', 'laps', 'compare', 'all',
                            'self-check'])
    p.add_argument('csv', nargs='*', help='telemetry .csv (two of them for compare)')
    p.add_argument('--track', default=None, help='track name for the map')
    p.add_argument('--by', default='distance', choices=['distance', 'time'])
    p.add_argument('--out', default=None, help='explicit output PNG path')
    p.add_argument('--self-check', action='store_true', dest='selfcheck')
    a = p.parse_args(argv)

    if a.selfcheck or a.cmd == 'self-check':
        return 0 if self_check() else 1

    if a.cmd == 'compare':
        if len(a.csv) != 2:
            p.error('compare needs exactly two csv paths')
        print(compare(a.csv[0], a.csv[1], by=a.by, out=a.out))
        return 0

    if len(a.csv) != 1:
        p.error(f'{a.cmd} needs exactly one csv path')
    csv_path = a.csv[0]
    if a.cmd == 'overview':
        print(overview(csv_path, a.out))
    elif a.cmd == 'gg':
        print(gg(csv_path, a.out))
    elif a.cmd == 'track':
        print(track_map(csv_path, a.track, a.out))
    elif a.cmd == 'laps':
        print(laps(csv_path, a.out))
    elif a.cmd == 'all':
        for path in all_plots(csv_path, a.track):
            print(path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
