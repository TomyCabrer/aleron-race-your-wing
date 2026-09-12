"""Measure a checkpoint, and draw its learning curve.

    python3 -m drive.ml.evaluate drive/ml/checkpoints/arena_plate.json
    python3 -m drive.ml.evaluate <ckpt> --plot runs/ml_learning.png
    python3 -m drive.ml.evaluate <ckpt> --wing off --track open
    python3 -m drive.ml.evaluate <ckpt> --transfer
    python3 -m drive.ml.evaluate <ckpt> --car mx5
    python3 -m drive.ml.evaluate --car-matrix          (no checkpoint needed)

Every number printed here is measured at `env.DT_EVAL = 1 ms`, the contract's
`DT_PHYS`, not at the 2 ms the trainer uses -- so a quoted lap time is a lap
time and not a training artefact. The baseline row is `Policy()` with
`theta = 0`, which by construction IS `baseline.baseline_action`, so the
before/after comparison is the same code path with and without the learned
residual.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os

import numpy as np

from .env import lap_time, rollout, DT_EVAL, DT_TRAIN, _kappa_at
from .policy import Policy

#: The transfer grid: every track a lap time means something on, and every
#: flank device the car can be fitted with. `dragstrip` is DELIBERATELY absent.
#: It is a 1500 m straight, it is not closed, so `lap_time` cannot report a
#: lap on it at all, and a policy's cornering -- which is the whole question
#: -- leaves no trace on it. Putting it in the table would pad the table, not
#: test the policy.
TRANSFER_TRACKS = ("arena", "open", "skidpad")
TRANSFER_WINGS = ("plate", "off", "fin")

#: The cross-car matrix's cars, in `cars.CAR_ORDER`. Wave 4's question is the
#: owner's original one -- task 9 said "teach the CARS how to drive the aero",
#: plural -- and the three in the library are genuinely different machines:
#: 1010 kg FWD 55 kW, 1140 kg RWD 109 kW, 1780 kg RWD 210 kW.
TRANSFER_CARS = ("corsa", "mx5", "540i")

#: Where the per-car checkpoints live, keyed by the car they trained on. The
#: `corsa` entry is the arena specialist, because the matrix is measured on the
#: arena and the row has to be that car's OWN best policy or the matrix is
#: comparing a specialist against two generalists.
CAR_CKPTS = {
    "corsa": "drive/ml/checkpoints/arena_plate.json",
    "mx5": "drive/ml/checkpoints/mx5_arena_plate.json",
    "540i": "drive/ml/checkpoints/540i_arena_plate.json",
}


def compare(path: str, track: str = "arena", wing: str = "plate",
            T: float = 280.0, verbose: bool = True, car=None) -> dict:
    """Baseline vs learned, at DT_EVAL, same track, same aero, same car."""
    learned = Policy.load(path)
    base = Policy()                     # theta = 0 -> the hand-written driver
    rows = {}
    for tag, pol in (("baseline", base), ("learned", learned)):
        rows[tag] = lap_time(pol, track, wing=wing, dt=DT_EVAL, T=T,
                             car=_car_of(car))
    if verbose:
        print(f"  {track} / wing {wing} / car {car or 'corsa'} / "
              f"dt {DT_EVAL * 1e3:.0f} ms / {T:.0f} s")
        print(f"  {'':9s} {'best lap':>9s} {'laps':>5s} {'v_mean':>7s} {'v_max':>7s} "
              f"{'dist':>8s} {'wing%':>6s} {'outer%':>7s}  ended")
        for tag in ("baseline", "learned"):
            r = rows[tag]
            bl = f"{r['best']:.3f}" if r["best"] else "   --   "
            print(f"  {tag:9s} {bl:>9s} {r['laps']:5d} {r['v_mean']:7.3f} "
                  f"{r['v_max']:7.3f} {r['s']:8.1f} {100 * r['wing_frac']:6.1f} "
                  f"{100 * r['wing_outer_frac']:7.1f}  {r['ended']}")
        b, l = rows["baseline"]["best"], rows["learned"]["best"]
        if b and l:
            print(f"  -> {b - l:+.3f} s a lap ({100 * (b - l) / b:+.2f} %)")
        else:
            print("  -> one of the two did not complete a flying lap; compare "
                  "distance covered instead")
    return rows


def aero_ablation(path: str, track: str = "arena", T: float = 280.0,
                  verbose: bool = True) -> dict:
    """What the learned policy is worth WITH and WITHOUT the device fitted.

    This is the question the whole study asks, put to the agent: a policy
    trained with the plate should be faster with it than without it, and by
    more than the baseline gains, if it has genuinely learned to use it.
    """
    out = {}
    for tag, pol in (("baseline", Policy()), ("learned", Policy.load(path))):
        for wing in ("off", "plate"):
            out[(tag, wing)] = lap_time(pol, track, wing=wing, dt=DT_EVAL, T=T)
    if verbose:
        print(f"\n  aero ablation, {track}, dt {DT_EVAL * 1e3:.0f} ms")
        print(f"  {'':9s} {'wing off':>10s} {'wing plate':>11s} {'delta':>9s}")
        for tag in ("baseline", "learned"):
            a, b = out[(tag, "off")]["best"], out[(tag, "plate")]["best"]
            if a and b:
                print(f"  {tag:9s} {a:10.3f} {b:11.3f} {a - b:+9.3f} s")
            else:
                sa, sb = out[(tag, 'off')]['s'], out[(tag, 'plate')]['s']
                print(f"  {tag:9s} {sa:10.1f} {sb:11.1f} {sb - sa:+9.1f} m "
                      f"(distance: no flying lap)")
    return out


def _cell(arg) -> tuple:
    """One (track, wing) cell of the transfer grid, baseline AND learned.

    Runs in a pool worker and returns plain dicts, so the two rollouts of a
    cell always share the same freshly built track and nothing large crosses
    the pickle boundary. 280 s at 1 ms is ~17 s of wall clock per rollout, so
    a 9-cell grid is 18 rollouts and fits in two passes of a 12-core pool.
    """
    path, track, wing, T, car = arg
    cr = _car_of(car)
    base = lap_time(Policy(), track, wing=wing, dt=DT_EVAL, T=T, car=cr)
    learn = lap_time(Policy.load(path), track, wing=wing, dt=DT_EVAL, T=T,
                     car=cr)
    return (track, wing, base, learn)


def transfer(path: str, tracks=TRANSFER_TRACKS, wings=TRANSFER_WINGS,
             T: float = 280.0, workers: int | None = None,
             verbose: bool = True, car=None) -> dict:
    """Does the gain transfer? Baseline vs learned on every (track, wing) cell.

    The policy in `drive/ml/checkpoints/arena_plate.json` was trained on ONE
    cell of this grid. Every other cell is a configuration it has never seen,
    evaluated UNCHANGED -- no re-tuning, no warm start, the same 308 numbers.
    The comparison that matters in each cell is not the learned lap time on
    its own but the learned lap time against the baseline in the SAME cell,
    because the baseline is what the residual degrades toward if the network's
    output is useless there.

    CALLERS MUST BE UNDER AN `if __name__ == "__main__"` GUARD, or pass
    `workers=1`. macOS defaults `multiprocessing` to *spawn*, which re-imports
    the calling module in every child; an unguarded script that calls this at
    module level therefore calls it again in each child and wedges instead of
    failing loudly. Cost me 14 minutes of a starved process doing 8 s of work.
    `python3 -m drive.ml.evaluate` is guarded, so the CLI is always safe.
    """
    jobs = [(path, tr, wg, T, car) for tr in tracks for wg in wings]
    workers = min(os.cpu_count() or 1, len(jobs)) if workers is None else workers
    if workers > 1:
        with mp.Pool(workers) as pool:
            res = pool.map(_cell, jobs, chunksize=1)
    else:
        res = [_cell(j) for j in jobs]
    out = {(tr, wg): (b, l) for tr, wg, b, l in res}

    if verbose:
        meta = Policy.load(path).meta
        trained = (str(meta.get("track", "?")), str(meta.get("wing", "?")))
        print(f"\n  transfer grid, dt {DT_EVAL * 1e3:.0f} ms, {T:.0f} s, "
              f"car {car or 'corsa'}, trained on {trained[0]} / {trained[1]}"
              f"{' / ' + str(meta['car']) if meta.get('car') else ''}")
        print(f"  {'track':9s} {'wing':6s} {'base lap':>9s} {'learn lap':>9s} "
              f"{'delta':>9s} {'%':>7s} {'base m':>8s} {'learn m':>8s} "
              f"{'b.laps':>6s} {'l.laps':>6s}  base/learn ended")
        for tr in tracks:
            for wg in wings:
                b, l = out[(tr, wg)]
                mark = "  <- trained" if (tr, wg) == trained else ""
                bl = f"{b['best']:.3f}" if b["best"] else "   --   "
                ll = f"{l['best']:.3f}" if l["best"] else "   --   "
                if b["best"] and l["best"]:
                    d = b["best"] - l["best"]
                    ds, ps = f"{d:+.3f}", f"{100 * d / b['best']:+.2f}"
                else:
                    ds, ps = "   --   ", "   --  "
                print(f"  {tr:9s} {wg:6s} {bl:>9s} {ll:>9s} {ds:>9s} {ps:>7s} "
                      f"{b['s']:8.1f} {l['s']:8.1f} {b['laps']:6d} {l['laps']:6d}"
                      f"  {b['ended']}/{l['ended']}{mark}")
    return out


def _car_of(name):
    """'corsa' -> None (so `rollout` builds `CorsaC()` and the path is
    bit-for-bit the pre-wave-4 one); anything else -> that `cars.CarSpec`."""
    if name in (None, "", "corsa"):
        return None
    import cars
    return cars.get(name)


def _car_cell(arg) -> tuple:
    """One (policy-or-baseline, car) cell of the cross-car matrix.

    `path is None` means the hand-written anchor on that car, which is the
    row every learned row has to be read against: the anchor is ONE driver
    (same gains, same 0.75 g plan) given each car's own lock and wheelbase,
    so a column's baseline is a property of the CAR and not of the policy.
    """
    path, car_name, track, wing, T = arg
    pol = Policy() if path is None else Policy.load(path)
    r = lap_time(pol, track, wing=wing, dt=DT_EVAL, T=T,
                 car=_car_of(car_name))
    return (path, car_name, r)


def car_transfer(paths=None, cars_=TRANSFER_CARS, track: str = "arena",
                 wing: str = "plate", T: float = 280.0,
                 workers: int | None = None, verbose: bool = True) -> dict:
    """The cross-car matrix: every (trained-on car x evaluated-on car) cell.

    Built exactly the way wave 3 built the cross-TRACK grid, and asking the
    same question one axis over: **does one policy generalise across cars, or
    does each car need its own?** A policy is 308 numbers tuned against one
    car's grip, power, brakes and driven axle, so there is no a-priori reason
    for it to travel -- and a negative result is a finding about the device
    and the driving, not a failure of the method.

    Evaluated UNCHANGED in every off-diagonal cell: no re-tuning, no warm
    start, the same 308 numbers. What travels with the car and not with the
    policy is only the ANCHOR's two per-car arguments (`lock_rad`,
    `wheelbase`), which is what makes the comparison meaningful rather than
    a units mismatch -- see `policy.Policy.action`.

    Same `__main__`-guard requirement as `transfer`: macOS spawns.
    """
    paths = dict(CAR_CKPTS if paths is None else paths)
    #  a missing checkpoint drops its ROW rather than killing the matrix: the
    #  per-car runs land one at a time and a half-filled matrix is still worth
    #  reading while the next one trains
    rows = [None] + [c for c in cars_
                     if paths.get(c) and os.path.exists(paths[c])]
    jobs = [(paths.get(r) if r else None, c, track, wing, T)
            for r in rows for c in cars_]
    workers = min(os.cpu_count() or 1, len(jobs)) if workers is None else workers
    if workers > 1:
        with mp.Pool(workers) as pool:
            res = pool.map(_car_cell, jobs, chunksize=1)
    else:
        res = [_car_cell(j) for j in jobs]
    out = {(("baseline" if p is None else _row_of(p, paths)), c): r
           for p, c, r in res}

    if verbose:
        print(f"\n  cross-car matrix, {track} / wing {wing} / "
              f"dt {DT_EVAL * 1e3:.0f} ms / {T:.0f} s")
        print(f"  best flying lap; '--' is no flying lap, and then the "
              f"distance covered in {T:.0f} s is in brackets")
        print(f"  {'trained on':12s}" + "".join(f"{c:>26s}" for c in cars_))
        for r in ["baseline"] + [c for c in rows if c]:
            line = f"  {r:12s}"
            for c in cars_:
                cell = out[(r, c)]
                b = out[("baseline", c)]
                if cell["best"]:
                    pc = (f" {100 * (b['best'] - cell['best']) / b['best']:+6.2f} %"
                          if b["best"] and r != "baseline" else "        ")
                    line += f"{cell['best']:12.3f}{pc:>14s}"
                else:
                    line += f"{'--':>9s} ({cell['s']:6.1f} m) {cell['ended'][:6]:>6s}"
            print(line)
        #  the wing statistics are the whole point of the item: different cars
        #  may want different aero timing, and this is where that shows
        print(f"\n  flank panel: % of steps deployed / % of those on the OUTER flank")
        print(f"  {'trained on':12s}" + "".join(f"{c:>20s}" for c in cars_))
        for r in ["baseline"] + [c for c in rows if c]:
            line = f"  {r:12s}"
            for c in cars_:
                cell = out[(r, c)]
                line += (f"{100 * cell['wing_frac']:12.1f} / "
                         f"{100 * cell['wing_outer_frac']:5.1f}")
            print(line)
    return out


def _row_of(path: str, paths: dict) -> str:
    for k, v in paths.items():
        if v == path:
            return k
    return str(path)


#: Curvature above which a station counts as "in a corner" for the timing
#: histogram. The SAME 1/220 m the hand-written wing rule arms on
#: (`baseline.KAPPA_ARM`), so "armed before the corner" is measured against
#: the same definition of corner the anchor uses.
K_CORNER = 1.0 / 220.0


def residual_safety(grids: dict, verbose: bool = True) -> dict:
    """The property that makes a residual safe to ship, counted over grids.

    `grids` is `{label: {(track, wing): (base, learn)}}`, i.e. whatever
    `transfer` returned, one entry per (policy, car) pair. Wave 3 established
    the guarantee on one car over 27 pairs: **worst regression -1.39 %, and it
    never puts a car off the road that the baseline was keeping on.** Wave 4
    has to re-establish it per car, because a guarantee measured on a 1010 kg
    FWD hatch is not a guarantee about a 1780 kg RWD saloon.

    Three classes, and only the third is a safety question:

      BOTH LAP      a percentage means something. The worst one is the claim.
      BASELINE ONLY the residual took a car the anchor was keeping on the road
                    and lost it. This is the number that must be ZERO.
      NEITHER LAPS  no lap time either way, so compare ground covered before
                    the same failure. Less ground is a regression, not a
                    danger -- the anchor was already failing there.
    """
    both, lost, won, neither = [], [], [], []
    for lab, grid in grids.items():
        for (tr, wg), (b, l) in grid.items():
            if b["best"] and l["best"]:
                both.append((lab, tr, wg, b, l,
                             100.0 * (b["best"] - l["best"]) / b["best"]))
            elif b["best"] and not l["best"]:
                lost.append((lab, tr, wg, b, l))
            elif l["best"] and not b["best"]:
                won.append((lab, tr, wg, b, l))
            else:
                neither.append((lab, tr, wg, b, l, l["s"] - b["s"]))
    both.sort(key=lambda r: r[5])
    neither.sort(key=lambda r: r[5])
    out = dict(n_pairs=len(both) + len(lost) + len(won) + len(neither),
               both=both, lost=lost, won=won, neither=neither,
               worst_pct=both[0][5] if both else None,
               n_faster=sum(1 for r in both if r[5] > 0),
               n_slower=sum(1 for r in both if r[5] <= 0),
               n_shorter=sum(1 for r in neither if r[5] < 0))
    if verbose:
        print(f"\n  residual safety over {out['n_pairs']} (policy, cell) pairs")
        print(f"  both lap:        {len(both):3d}  -- faster in {out['n_faster']}, "
              f"slower in {out['n_slower']}")
        if both:
            lab, tr, wg, b, l, pc = both[0]
            print(f"     worst regression {pc:+.2f} % : {lab} on {tr}/{wg}, "
                  f"{l['best']:.3f} s against {b['best']:.3f} s, "
                  f"{l['laps']} laps, ended '{l['ended']}'")
        print(f"  learned rescues a cell the anchor cannot lap: {len(won):3d}")
        print(f"  ** learned LOSES a cell the anchor CAN lap:   {len(lost):3d}"
              f"   <- must be 0")
        for lab, tr, wg, b, l in lost:
            print(f"     {lab} on {tr}/{wg}: anchor {b['best']:.3f} s, "
                  f"learned {l['s']:.1f} m then '{l['ended']}'")
        print(f"  neither laps:    {len(neither):3d}  -- learned covered LESS "
              f"ground in {out['n_shorter']}")
        for lab, tr, wg, b, l, d in neither[:4]:
            print(f"     {lab} on {tr}/{wg}: {l['s']:7.1f} m vs {b['s']:7.1f} m "
                  f"({d:+.1f} m), {b['ended']}/{l['ended']}")
    return out


def wing_timing(path, track: str = "arena", wing: str = "plate",
                car: str = "corsa", T: float = 140.0, verbose: bool = True) -> dict:
    """WHEN in the corner is the panel armed, not just how often.

    `wing_frac` says the device is out 56 % of the time and says nothing about
    whether that is the right 56 %. Two measurements here, and the SECOND is
    the one that answers "do different cars want different aero timing":

    1. the deployed fraction in each of four phases of the circuit, taken from
       the track's own `kappa` array so they are a property of the CIRCUIT and
       not of the driver:

           straight    not in a corner and none within 35 m
           approach    not in a corner but one is within 35 m
           corner A    in a corner, first half of it by arc length
           corner B    in a corner, second half

    2. **how many metres INTO each corner the panel came out.** Negative would
       mean it was already out when the corner arrived.

    A structural fact makes (2) the interesting one and it was found the hard
    way here: the flank device cannot deploy until the driver is *steering*.
    `sgn_dev` is derived from the sign of the steering command past a 5 %
    deadband (CONTRACT section 4, reconciliation 7), so with the wheel
    straight there is no side for the panel to deploy on and `armed` is False
    however long `wing_on` has been true. Measured: the anchor's wing rule
    commands the device on 35 m before the corner (its 35 m lookahead crosses
    `KAPPA_ARM`) and the panel is nevertheless out at only 2 of 9 corner-entry
    stations on an arena lap. **So "when does the device come out" is not a
    lead time the driver chooses; it is a turn-in event**, and what a car can
    vary is how far into the corner it happens and how long it lasts.
    """
    from .. import track as trk
    pol = Policy() if path is None else Policy.load(path)
    tr = trk.make_track(track)
    coll: list = []
    r = lap_time(pol, track, wing=wing, dt=DT_EVAL, T=T, car=_car_of(car),
                 collect=coll)

    #  corner intervals [entry, exit] in s, from |kappa| crossing K_CORNER
    kap = np.abs(np.asarray(tr.kappa, float))
    inc = kap >= K_CORNER
    L = float(tr.length)
    corners = []
    i = 1
    while i < len(inc):
        if inc[i] and not inc[i - 1]:
            j = i
            while j + 1 < len(inc) and inc[j + 1]:
                j += 1
            corners.append((float(tr.s[i]), float(tr.s[j])))
        i += 1

    def phase(s: float) -> str:
        for e, x in corners:
            if e <= s <= x:
                return "cornerA" if s - e <= 0.5 * (x - e) else "cornerB"
        ahead = min(((e - s) % L) for e, _x in corners) if corners else 1e9
        return "approach" if ahead <= 35.0 else "straight"

    cnt = {p: 0 for p in ("straight", "approach", "cornerA", "cornerB")}
    dep = dict(cnt)
    ss, arm = [], []
    for (_t, x, y, _psi, _u, wdep, _side) in coll:
        s, _n, _k, _p, _i = trk.project(tr, x, y)
        ph = phase(s)
        cnt[ph] += 1
        if wdep > 0.05:
            dep[ph] += 1
        ss.append(s)
        arm.append(wdep > 0.05)

    #  metres INTO each corner at which the panel came out. For every corner
    #  the car actually reached, walk forward from 35 m before its entry to
    #  its exit and take the first armed sample. An earlier version measured
    #  every rising edge of `wing_deploy > 0.05` instead and its median was
    #  garbage: the signal chatters (14 edges in a 90 s arena run, twice the
    #  number of corners), so most edges were mid-straight flickers hundreds
    #  of metres from anything.
    intos = []
    inwin = [False] * len(corners)
    done = [False] * len(corners)
    for k in range(len(ss)):
        for c, (e, x) in enumerate(corners):
            lo = (e - 35.0) % L
            here = (lo <= ss[k] <= x) if lo <= x else (ss[k] >= lo or ss[k] <= x)
            if here and not inwin[c]:
                done[c] = False                 # a fresh traversal of it
            inwin[c] = here
            if here and arm[k] and not done[c]:
                #  signed and wrapped, so a panel already out when the corner
                #  arrives reads negative rather than as most of a lap
                intos.append(((ss[k] - e) + 0.5 * L) % L - 0.5 * L)
                done[c] = True
    out = dict(car=car, track=track, wing=wing, n=len(coll),
               frac={p: (dep[p] / cnt[p] if cnt[p] else float("nan"))
                     for p in cnt},
               count=dict(cnt), corners=len(intos), intos=intos,
               into_med=float(np.median(intos)) if intos else float("nan"),
               wing_frac=r["wing_frac"], wing_outer_frac=r["wing_outer_frac"],
               best=r["best"], ended=r["ended"], s=r["s"])
    if verbose:
        f = out["frac"]
        print(f"  {car:6s} {('anchor' if path is None else 'learned'):8s} "
              f"out {100 * out['wing_frac']:5.1f} % "
              f"(outer {100 * out['wing_outer_frac']:5.1f} %)  "
              f"straight {100 * f['straight']:5.1f}  "
              f"approach {100 * f['approach']:5.1f}  "
              f"cnrA {100 * f['cornerA']:5.1f}  cnrB {100 * f['cornerB']:5.1f}  "
              f"out {out['into_med']:+6.1f} m into the corner "
              f"({out['corners']} corners)  {out['ended']}")
    return out


def plot_curve(path: str, out: str) -> str | None:
    """The learning curve out of the checkpoint's own metadata."""
    import matplotlib
    matplotlib.use("Agg")               # before pyplot, as plots.py does
    import matplotlib.pyplot as plt

    pol = Policy.load(path)
    curve = pol.meta.get("curve") or []
    if not curve:
        return None
    it = [c["it"] for c in curve]
    #  The TRUE baseline, re-measured: `curve[0]` is the mean policy AFTER the
    #  first ES update, which on this run had already wandered off the track
    #  (527 m), and drawing that as "the baseline" flatters the result and
    #  wrecks the y scale. theta = 0 is the honest reference.
    T = float(pol.meta.get("T", 70.0))
    tracks = list(pol.meta.get("tracks") or [pol.meta.get("track", "arena")])
    if len(tracks) > 1:
        #  A multi-track fitness is already reward/baseline-reward per track,
        #  meaned, so the hand-written driver sits at exactly 1.0 and there is
        #  nothing to re-measure. Drawing it as metres would be a category
        #  error -- the two tracks do not have the same metres.
        #  keep this SHORT: at 7.5 in wide with two stacked axes a longer
        #  label is clipped by tight_layout rather than shrinking the axes
        base_r, y_lab = 1.0, f"fitness = mean(m / baseline m) in {T:.0f} s"
        b_lab = "hand-written baseline, 1.000 by construction"
    else:
        base_r = rollout(Policy(), str(tracks[0]),
                         dt=float(pol.meta.get("dt_train", DT_TRAIN)), T=T,
                         wing=str(pol.meta.get("wing", "plate"))).reward
        y_lab = f"reward = m advanced in {T:.0f} s"
        b_lab = f"hand-written baseline, {base_r:.0f} m"
    fig, ax = plt.subplots(2, 1, figsize=(7.5, 6.0), sharex=True,
                           gridspec_kw=dict(height_ratios=(2, 1)))
    ax[0].plot(it, [c["pop_best"] for c in curve], lw=0.9, color="#9aa0a6",
               label="population best")
    ax[0].plot(it, [c["pop_med"] for c in curve], lw=0.9, color="#d0d4d9",
               label="population median")
    ax[0].plot(it, [c["mean"] for c in curve], lw=1.8, color="#4fa3ff",
               label="mean policy (the curve)")
    ax[0].axhline(base_r, ls="--", lw=1.1, color="#ff8c2b", label=b_lab)
    #  clip the y range to the band that matters: the first iterate's crash
    #  would otherwise compress the whole run into the top 10 % of the axis
    fin = [c["mean"] for c in curve] + [base_r]
    lo = min(min(fin), base_r) - 0.02 * abs(base_r)
    ax[0].set_ylim(max(lo, 0.90 * min(base_r, min(fin))),
                   1.02 * max(max(fin), max(c["pop_best"] for c in curve)))
    ax[0].set_ylabel(y_lab)
    ax[0].legend(fontsize=8, loc="lower right")
    ax[0].grid(alpha=0.25)
    ax[0].set_title(f"drive.ml ES on {'+'.join(tracks)} / wing "
                    f"{pol.meta.get('wing')}  --  {pol.meta.get('iters')} iters "
                    f"x pop {pol.meta.get('pop')}, {pol.meta.get('secs')} s")
    ax[1].plot(it, [100 * c["wing_frac"] for c in curve], lw=1.4,
               color="#ff8c2b", label="flank panel deployed, % of steps")
    ax[1].plot(it, [100 * c["wing_outer"] for c in curve], lw=1.0,
               color="#5ac47d", label="... of those, on the OUTER flank, %")
    ax[1].set_ylim(0, 105)
    ax[1].set_xlabel("ES iteration")
    ax[1].set_ylabel("%")
    ax[1].legend(fontsize=8, loc="lower right")
    ax[1].grid(alpha=0.25)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="measure a drive.ml checkpoint at DT_PHYS")
    ap.add_argument("checkpoint", nargs="?", default=None,
                    help="omit it only with --car-matrix, which reads the "
                         "per-car checkpoints out of CAR_CKPTS")
    ap.add_argument("--track", default=None, help="default: the one it trained on")
    ap.add_argument("--wing", default=None, choices=("off", "fin", "plate"))
    ap.add_argument("--duration", type=float, default=280.0)
    ap.add_argument("--workers", type=int, default=None,
                    help="pool size for the grids; lower it to share the box "
                         "with a training run")
    ap.add_argument("--plot", default=None, help="write the learning curve here")
    ap.add_argument("--ablation", action="store_true", help="also run wing on/off")
    ap.add_argument("--transfer", action="store_true",
                    help="baseline vs learned on every (track, wing) cell -- "
                         "does the gain transfer off the cell it trained on?")
    ap.add_argument("--only-transfer", action="store_true",
                    help="the transfer grid and nothing else")
    ap.add_argument("--car", default=None,
                    help="cars.py key to MEASURE on: corsa | mx5 | 540i")
    ap.add_argument("--car-matrix", action="store_true",
                    help="the cross-car matrix: every (trained-on car x "
                         "evaluated-on car) cell, baseline vs learned")
    ap.add_argument("--wing-timing", action="store_true",
                    help="per-car flank-panel deployment BY CORNER PHASE, "
                         "anchor and learned, for every car")
    ap.add_argument("--safety", action="store_true",
                    help="the per-car residual-safety count: each per-car "
                         "policy over its OWN car's full (track x wing) grid")
    a = ap.parse_args(argv)
    if a.safety:
        grids = {}
        for cn in TRANSFER_CARS:
            pth = CAR_CKPTS.get(cn)
            if pth and os.path.exists(pth):
                grids[f"{cn}_policy/{cn}"] = transfer(
                    pth, T=a.duration, car=cn,
                    verbose=True, workers=a.workers)
        residual_safety(grids)
    if a.car_matrix or a.wing_timing or a.safety:
        if a.car_matrix:
            car_transfer(track=a.track or "arena", wing=a.wing or "plate",
                         T=a.duration, workers=a.workers)
        if a.wing_timing:
            print(f"\n  flank-panel timing by corner phase, "
                  f"{a.track or 'arena'} / {a.wing or 'plate'} / dt 1 ms")
            for cn in TRANSFER_CARS:
                for pth in (None, CAR_CKPTS.get(cn)):
                    if pth is None or os.path.exists(pth):
                        wing_timing(pth, track=a.track or "arena",
                                    wing=a.wing or "plate", car=cn)
        if not a.checkpoint:
            return 0
    meta = Policy.load(a.checkpoint).meta
    track = a.track or meta.get("track", "arena")
    wing = a.wing or meta.get("wing", "plate")
    print(f"  {a.checkpoint}\n  trained: { {k: v for k, v in meta.items() if k != 'curve'} }")
    if not a.only_transfer:
        compare(a.checkpoint, track, wing, a.duration, car=a.car)
    if a.transfer or a.only_transfer:
        transfer(a.checkpoint, T=a.duration, car=a.car, workers=a.workers)
    if a.ablation:
        aero_ablation(a.checkpoint, track, a.duration)
    if a.plot:
        p = plot_curve(a.checkpoint, a.plot)
        print(f"\n  curve -> {p}" if p else "\n  no curve in this checkpoint's metadata")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
