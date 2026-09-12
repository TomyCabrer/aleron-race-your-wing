"""XFOIL as a subprocess, with a JSON cache: the same tool the AeroBO design
tool runs, driven the same way (PLOP G to kill the plot window, PANE to
re-panel, PACC to a polar file, a split sweep up from 0 and down from 0 so
a stall on one side never kills the other).

Nothing here is required: `available()` says whether the binary exists and
every caller falls back to polar.estimate_polar when it does not, or when
XFOIL fails to converge. Results are cached under the library's `polars/`
directory keyed on the shape fingerprint and the Reynolds number, so a
section costs its ~1.5 s once.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
import time

import numpy as np

from .polar import Polar

_CANDIDATES = ("/opt/homebrew/bin/xfoil", "/usr/local/bin/xfoil", "/usr/bin/xfoil")


def binary() -> str | None:
    env = os.environ.get("CARSIM_XFOIL")
    if env and os.path.isfile(env) and os.access(env, os.X_OK):
        return env
    w = shutil.which("xfoil")
    if w:
        return w
    for c in _CANDIDATES:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def available() -> bool:
    return binary() is not None and not os.environ.get("CARSIM_NO_XFOIL")


def cache_path(cache_dir: str, fingerprint: str, re: float, ncrit: float) -> str:
    return os.path.join(cache_dir, f"{fingerprint}_re{re:.3g}_n{ncrit:g}.json".replace("+", ""))


def _script(re: float, mach: float, ncrit: float, n_panel: int, a_hi: float,
            a_lo: float, step: float, n_iter: int) -> str:
    lines = [
        "PLOP", "G", "",
        "LOAD foil.dat", "foil",
        "PPAR", f"N {int(n_panel)}", "", "",
        "OPER",
        f"VISC {re:.6g}",
        f"MACH {mach:.4f}",
        "VPAR", f"N {ncrit:g}", "",
        f"ITER {int(n_iter)}",
        "PACC", "polar.txt", "",
        f"ASEQ 0 {a_hi:.2f} {step:.2f}",
        "INIT",
        f"ASEQ {-step:.2f} {a_lo:.2f} {-step:.2f}",
        "PACC", "",
        "QUIT", "",
    ]
    return "\n".join(lines)


def parse_polar_file(text: str):
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            continue
        if len(vals) >= 5:
            rows.append(vals[:5])
    if not rows:
        return None
    arr = np.asarray(rows, dtype=float)
    # duplicate alphas (the 0 of both sweeps) collapse to the first
    _, idx = np.unique(np.round(arr[:, 0], 3), return_index=True)
    arr = arr[np.sort(idx)]
    return arr


def run_polar(coords: np.ndarray, re: float, name: str = "section", *,
              mach: float = 0.0, ncrit: float = 9.0, n_panel: int = 200,
              a_lo: float = -14.0, a_hi: float = 22.0, step: float = 1.0,
              n_iter: int = 100, timeout_s: float = 90.0,
              cache_dir: str | None = None, fingerprint: str | None = None) -> Polar | None:
    """A converged XFOIL polar, or None (binary missing, timeout, nothing
    converged). Partial sweeps are accepted when they hold >= 8 rows."""
    xb = binary()
    if xb is None or os.environ.get("CARSIM_NO_XFOIL"):
        return None
    re = float(re)
    if cache_dir and fingerprint:
        cp = cache_path(cache_dir, fingerprint, re, ncrit)
        if os.path.isfile(cp):
            try:
                with open(cp) as f:
                    return Polar.from_json(json.load(f))
            except (OSError, ValueError, KeyError):
                pass
    coords = np.asarray(coords, dtype=float)
    tmp = tempfile.mkdtemp(prefix="carsim_xf_")
    try:
        with open(os.path.join(tmp, "foil.dat"), "w") as f:
            f.write("foil\n")
            for x, y in coords:
                f.write(f" {x:.6f} {y:.6f}\n")
        script = _script(re, mach, ncrit, n_panel, a_hi, a_lo, step, n_iter)
        t0 = time.perf_counter()
        try:
            subprocess.run([xb], input=script, capture_output=True, text=True,
                           timeout=timeout_s, cwd=tmp)
        except subprocess.TimeoutExpired:
            pass                                  # a partial polar is still read
        except (FileNotFoundError, PermissionError):
            return None
        wall = time.perf_counter() - t0
        pp = os.path.join(tmp, "polar.txt")
        if not os.path.isfile(pp):
            return None
        with open(pp) as f:
            arr = parse_polar_file(f.read())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if arr is None or arr.shape[0] < 8:
        return None
    pol = Polar(name=name, re=re, source="xfoil", alpha=arr[:, 0], cl=arr[:, 1],
                cd=arr[:, 2], cm=arr[:, 4])
    pol.wall_s = wall                              # type: ignore[attr-defined]
    if cache_dir and fingerprint:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            tmpf = cache_path(cache_dir, fingerprint, re, ncrit) + f".{os.getpid()}.tmp"
            with open(tmpf, "w") as f:
                json.dump(pol.to_json(), f)
            os.replace(tmpf, cache_path(cache_dir, fingerprint, re, ncrit))
        except OSError:
            pass
    return pol


# --------------------------------------------------------------------------- #
def self_check(verbose: bool = True) -> bool:
    from . import airfoil as af
    ok = True

    def rep(tag, passed, msg=""):
        nonlocal ok
        ok = ok and passed
        if verbose:
            print(f"  [{'ok' if passed else 'FAIL'}] {tag}: {msg}")

    txt = ("  alpha    CL        CD       CDp       CM     Top_Xtr  Bot_Xtr\n"
           " ------ -------- --------- --------- -------- ------- -------\n"
           "  0.000   0.2500   0.00600   0.00100  -0.0500  0.6000  0.9000\n"
           "  2.000   0.4700   0.00650   0.00120  -0.0510  0.5000  0.9500\n"
           "  0.000   0.2500   0.00600   0.00100  -0.0500  0.6000  0.9000\n"
           " -2.000   0.0300   0.00640   0.00110  -0.0490  0.7000  0.8000\n")
    arr = parse_polar_file(txt)
    rep("polar file parse drops the duplicate 0", arr is not None and arr.shape == (3, 5), str(arr.shape if arr is not None else None))
    if not available():
        rep("xfoil binary", True, "not on this machine: the estimate path is used (fine)")
    else:
        t0 = time.perf_counter()
        pol = run_polar(af.naca4_coords("2412"), 1e6, "naca2412", timeout_s=120.0)
        dt = time.perf_counter() - t0
        rep("xfoil naca2412 Re 1e6 converges", pol is not None and pol.n_rows >= 20,
            f"{0 if pol is None else pol.n_rows} rows in {dt:.1f} s")
        if pol is not None:
            rep("xfoil slope 5.6-6.6 /rad", 5.6 < pol.a_lin < 6.6, f"a {pol.a_lin:.3f}")
            rep("xfoil alpha_L0 -1.6..-2.6", -2.6 < pol.alpha_L0_deg < -1.6, f"{pol.alpha_L0_deg:.2f}")
            rep("xfoil cl_max 1.4-1.8", 1.4 < pol.cl_max < 1.8, f"{pol.cl_max:.3f} at {pol.alpha_clmax_deg:.0f}")
            rep("xfoil cd_min 0.004-0.008", 0.004 < pol.cd_min < 0.008, f"{pol.cd_min:.4f}")
    if verbose:
        print("  ALL PASS" if ok else "  FAILURES ABOVE")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_check() else 1)
