"""Measure a checkpoint, and draw its learning curve.

    python3 -m drive.ml.evaluate drive/ml/checkpoints/arena_plate.json
    python3 -m drive.ml.evaluate <ckpt> --plot runs/ml_learning.png
    python3 -m drive.ml.evaluate <ckpt> --wing off --track open

Every number printed here is measured at `env.DT_EVAL = 1 ms`, the contract's
`DT_PHYS`, not at the 2 ms the trainer uses -- so a quoted lap time is a lap
time and not a training artefact. The baseline row is `Policy()` with
`theta = 0`, which by construction IS `baseline.baseline_action`, so the
before/after comparison is the same code path with and without the learned
residual.
"""

from __future__ import annotations

import argparse
import os

from .env import lap_time, DT_EVAL
from .policy import Policy


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
    fig, ax = plt.subplots(2, 1, figsize=(7.5, 6.0), sharex=True,
                           gridspec_kw=dict(height_ratios=(2, 1)))
    ax[0].plot(it, [c["pop_best"] for c in curve], lw=0.9, color="#9aa0a6",
               label="population best")
    ax[0].plot(it, [c["pop_med"] for c in curve], lw=0.9, color="#d0d4d9",
               label="population median")
    ax[0].plot(it, [c["mean"] for c in curve], lw=1.8, color="#4fa3ff",
               label="mean policy (the curve)")
    ax[0].axhline(curve[0]["mean"], ls="--", lw=0.9, color="#ff8c2b",
                  label=f"baseline, {curve[0]['mean']:.0f} m")
    ax[0].set_ylabel(f"reward = m advanced in {pol.meta.get('T', 0):.0f} s")
    ax[0].legend(fontsize=8, loc="lower right")
    ax[0].grid(alpha=0.25)
    ax[0].set_title(f"drive.ml ES on {pol.meta.get('track')} / wing "
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
    a = ap.parse_args(argv)
    meta = Policy.load(a.checkpoint).meta
    track = a.track or meta.get("track", "arena")
    wing = a.wing or meta.get("wing", "plate")
    print(f"  {a.checkpoint}\n  trained: { {k: v for k, v in meta.items() if k != 'curve'} }")
    compare(a.checkpoint, track, wing, a.duration)
    if a.ablation:
        aero_ablation(a.checkpoint, track, a.duration)
    if a.plot:
        p = plot_curve(a.checkpoint, a.plot)
        print(f"\n  curve -> {p}" if p else "\n  no curve in this checkpoint's metadata")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
