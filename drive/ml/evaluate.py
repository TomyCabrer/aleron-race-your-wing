"""Measure a checkpoint, and draw its learning curve.

    python3 -m drive.ml.evaluate drive/ml/checkpoints/arena_plate.json
    python3 -m drive.ml.evaluate <ckpt> --plot runs/ml_learning.png
    python3 -m drive.ml.evaluate <ckpt> --wing off --track open
    python3 -m drive.ml.evaluate <ckpt> --transfer

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

from .env import lap_time, rollout, DT_EVAL, DT_TRAIN
from .policy import Policy

#: The transfer grid: every track a lap time means something on, and every
#: flank device the car can be fitted with. `dragstrip` is DELIBERATELY absent.
#: It is a 1500 m straight, it is not closed, so `lap_time` cannot report a
#: lap on it at all, and a policy's cornering -- which is the whole question
#: -- leaves no trace on it. Putting it in the table would pad the table, not
#: test the policy.
TRANSFER_TRACKS = ("arena", "open", "skidpad")
TRANSFER_WINGS = ("plate", "off", "fin")


def compare(path: str, track: str = "arena", wing: str = "plate",
            T: float = 280.0, verbose: bool = True) -> dict:
    """Baseline vs learned, at DT_EVAL, same track and same aero."""
    learned = Policy.load(path)
    base = Policy()                     # theta = 0 -> the hand-written driver
    rows = {}
    for tag, pol in (("baseline", base), ("learned", learned)):
        rows[tag] = lap_time(pol, track, wing=wing, dt=DT_EVAL, T=T)
    if verbose:
        print(f"  {track} / wing {wing} / dt {DT_EVAL * 1e3:.0f} ms / {T:.0f} s")
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
    path, track, wing, T = arg
    base = lap_time(Policy(), track, wing=wing, dt=DT_EVAL, T=T)
    learn = lap_time(Policy.load(path), track, wing=wing, dt=DT_EVAL, T=T)
    return (track, wing, base, learn)


def transfer(path: str, tracks=TRANSFER_TRACKS, wings=TRANSFER_WINGS,
             T: float = 280.0, workers: int | None = None,
             verbose: bool = True) -> dict:
    """Does the gain transfer? Baseline vs learned on every (track, wing) cell.

    The policy in `drive/ml/checkpoints/arena_plate.json` was trained on ONE
    cell of this grid. Every other cell is a configuration it has never seen,
    evaluated UNCHANGED -- no re-tuning, no warm start, the same 308 numbers.
    The comparison that matters in each cell is not the learned lap time on
    its own but the learned lap time against the baseline in the SAME cell,
    because the baseline is what the residual degrades toward if the network's
    output is useless there.
    """
    jobs = [(path, tr, wg, T) for tr in tracks for wg in wings]
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
              f"trained on {trained[0]} / {trained[1]}")
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
        base_r, y_lab = 1.0, ("fitness = mean over tracks of "
                              f"(m advanced in {T:.0f} s / baseline's m)")
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
    ap.add_argument("checkpoint")
    ap.add_argument("--track", default=None, help="default: the one it trained on")
    ap.add_argument("--wing", default=None, choices=("off", "fin", "plate"))
    ap.add_argument("--duration", type=float, default=280.0)
    ap.add_argument("--plot", default=None, help="write the learning curve here")
    ap.add_argument("--ablation", action="store_true", help="also run wing on/off")
    ap.add_argument("--transfer", action="store_true",
                    help="baseline vs learned on every (track, wing) cell -- "
                         "does the gain transfer off the cell it trained on?")
    ap.add_argument("--only-transfer", action="store_true",
                    help="the transfer grid and nothing else")
    a = ap.parse_args(argv)
    meta = Policy.load(a.checkpoint).meta
    track = a.track or meta.get("track", "arena")
    wing = a.wing or meta.get("wing", "plate")
    print(f"  {a.checkpoint}\n  trained: { {k: v for k, v in meta.items() if k != 'curve'} }")
    if not a.only_transfer:
        compare(a.checkpoint, track, wing, a.duration)
    if a.transfer or a.only_transfer:
        transfer(a.checkpoint, T=a.duration)
    if a.ablation:
        aero_ablation(a.checkpoint, track, a.duration)
    if a.plot:
        p = plot_curve(a.checkpoint, a.plot)
        print(f"\n  curve -> {p}" if p else "\n  no curve in this checkpoint's metadata")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
