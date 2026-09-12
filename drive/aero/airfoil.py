"""Aerofoil coordinates: NACA-4 synthesis, the UIUC .dat reader, CST
(Kulfan) shapes, and the geometry numbers a designer reads off a section.

Every loop leaves here in XFOIL order -- trailing edge, along the UPPER
surface to the leading edge, back along the lower surface to the trailing
edge -- with a unit chord, the leading edge at x = 0 and a SHARP trailing
edge. panel2d and XFOIL both want exactly that; a blunt base was measured
(urop-bo-aero, session 63) to cost 7.6 % of cl in the panel method and to
get WORSE under refinement, so blunt loops are closed here, once, on entry.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, asdict

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "airfoils")

# --------------------------------------------------------------------------- #
#  NACA 4-digit                                                                #
# --------------------------------------------------------------------------- #
def naca4_coords(code: str, n: int = 81) -> np.ndarray:
    """Closed NACA-4 loop, `n` points per surface, cosine-clustered, sharp TE.

    The thickness polynomial uses a4 = -0.1036 (the closed-trailing-edge
    variant) so the loop needs no base panel."""
    code = str(code).strip()
    if len(code) != 4 or not code.isdigit():
        raise ValueError(f"NACA-4 code must be four digits, got {code!r}")
    m = int(code[0]) / 100.0
    p = int(code[1]) / 10.0
    t = int(code[2:]) / 100.0
    beta = np.linspace(0.0, math.pi, n)
    x = 0.5 * (1.0 - np.cos(beta))
    yt = 5.0 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2
                    + 0.2843 * x ** 3 - 0.1036 * x ** 4)
    if m > 0.0 and p > 0.0:
        yc = np.where(x < p, m / p ** 2 * (2 * p * x - x ** 2),
                      m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x ** 2))
        dyc = np.where(x < p, 2 * m / p ** 2 * (p - x),
                       2 * m / (1 - p) ** 2 * (p - x))
    else:
        yc = np.zeros_like(x)
        dyc = np.zeros_like(x)
    th = np.arctan(dyc)
    xu = x - yt * np.sin(th)
    yu = yc + yt * np.cos(th)
    xl = x + yt * np.sin(th)
    yl = yc - yt * np.cos(th)
    # TE -> LE along the upper, LE -> TE along the lower (XFOIL order)
    loop = np.concatenate([np.column_stack([xu, yu])[::-1],
                           np.column_stack([xl, yl])[1:]])
    return normalise_loop(loop)


# --------------------------------------------------------------------------- #
#  .dat reader (Selig and Lednicer) -- ported from urop-bo-aero               #
# --------------------------------------------------------------------------- #
def load_dat(path: str) -> tuple[str, np.ndarray]:
    """(name, closed loop). Selig files (TE-upper -> LE -> TE-lower) and
    Lednicer files (count line, upper LE->TE, blank, lower LE->TE) both
    come back as one XFOIL-ordered unit-chord loop."""
    with open(path, errors="replace") as f:
        lines = f.read().splitlines()
    name = os.path.splitext(os.path.basename(path))[0]
    for raw in lines:
        s = raw.strip()
        if s:
            try:
                [float(p) for p in s.replace(",", " ").split()]
            except ValueError:
                name = s
            break
    blocks: list[list[tuple[float, float]]] = [[]]
    for raw in lines:
        s = raw.strip()
        if not s:
            if blocks[-1]:
                blocks.append([])
            continue
        parts = s.replace(",", " ").split()
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            continue
        if len(vals) != 2:
            continue
        blocks[-1].append((vals[0], vals[1]))
    blocks = [b for b in blocks if b]
    if not blocks:
        raise ValueError(f"{os.path.basename(path)}: no coordinate pairs")
    loop = None
    b0 = blocks[0]
    if (len(blocks) >= 2 and b0[0][0] > 1.5 and float(b0[0][0]).is_integer()
            and float(b0[0][1]).is_integer()):
        pts = b0[1:]
        if pts:
            upper, lower = pts, [p for b in blocks[1:] for p in b]
        elif len(blocks) >= 3:
            upper, lower = blocks[1], [p for b in blocks[2:] for p in b]
        else:
            upper = lower = []
        if (len(upper) >= 2 and len(lower) >= 2 and upper[0][0] < upper[-1][0]
                and lower[0][0] < lower[-1][0]):
            loop = list(reversed(upper)) + lower[1:]
    if loop is None:
        loop = [p for b in blocks for p in b]
    c = np.asarray(loop, dtype=float)
    keep = np.ones(len(c), dtype=bool)
    keep[1:] = np.any(np.abs(np.diff(c, axis=0)) > 1e-12, axis=1)
    c = c[keep]
    if c.shape[0] < 20:
        raise ValueError(f"{os.path.basename(path)}: only {c.shape[0]} points")
    return name, normalise_loop(c)


# --------------------------------------------------------------------------- #
#  loop hygiene                                                                #
# --------------------------------------------------------------------------- #
def normalise_loop(c: np.ndarray) -> np.ndarray:
    """Unit chord, LE at x = 0, XFOIL order, sharp TE, no duplicate points."""
    c = np.asarray(c, dtype=float).copy()
    if c.ndim != 2 or c.shape[1] != 2 or c.shape[0] < 10:
        raise ValueError("a loop is an (n, 2) array with n >= 10")
    if not np.all(np.isfinite(c)):
        raise ValueError("non-finite coordinates")
    x_min, x_max = float(c[:, 0].min()), float(c[:, 0].max())
    chord = x_max - x_min
    if chord <= 1e-9:
        raise ValueError("zero chord")
    c = (c - [x_min, 0.0]) / chord
    i_le = int(np.argmin(c[:, 0]))
    if i_le < 3 or i_le > c.shape[0] - 4:
        raise ValueError("leading edge at the loop end: not a closed loop")
    # XFOIL order = the first half is the UPPER surface (mean y above the mean
    # of the second half); flip the traversal direction if it is not
    if c[:i_le, 1].mean() < c[i_le + 1:, 1].mean():
        c = c[::-1].copy()
        i_le = c.shape[0] - 1 - i_le
    # sharp trailing edge: both ends pulled onto their mean point, the base
    # thickness spread linearly over the aft 30 % (no kink, no base panel)
    te = 0.5 * (c[0] + c[-1])
    for sl, end in ((slice(0, i_le + 1), c[0]), (slice(i_le, None), c[-1])):
        seg = c[sl]
        w = np.clip((seg[:, 0] - 0.7) / 0.3, 0.0, 1.0)
        seg[:, 1] = seg[:, 1] - w * (end[1] - te[1])
        seg[:, 0] = seg[:, 0] - w * (end[0] - te[0])
    c[0] = te
    c[-1] = te
    if np.max(np.abs(c[:, 1])) > 0.5:
        raise ValueError("|y|/c > 0.5: not an aerofoil")
    keep = np.ones(len(c), dtype=bool)
    keep[1:] = np.any(np.abs(np.diff(c, axis=0)) > 1e-10, axis=1)
    c = c[keep]
    # re-anchor LE at x = 0 exactly after the TE move
    c[:, 0] -= c[:, 0].min()
    return c


def split_surfaces(c: np.ndarray, n_grid: int = 121):
    """(x, y_upper, y_lower) on a common cosine grid."""
    x, y = c[:, 0], c[:, 1]
    i_le = int(np.argmin(x))
    xu, yu = x[: i_le + 1][::-1], y[: i_le + 1][::-1]
    xl, yl = x[i_le:], y[i_le:]
    su, sl = np.argsort(xu), np.argsort(xl)
    xg = 0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, n_grid)))
    return xg, np.interp(xg, xu[su], yu[su]), np.interp(xg, xl[sl], yl[sl])


def resample(c: np.ndarray, n: int = 81) -> np.ndarray:
    """Cosine-resampled loop with `n` points per surface (2n-1 total)."""
    xg, yu, yl = split_surfaces(c, n)
    loop = np.concatenate([np.column_stack([xg, yu])[::-1],
                           np.column_stack([xg, yl])[1:]])
    return normalise_loop(loop)


def geometry(c: np.ndarray) -> dict:
    """Thickness, camber and their stations, LE radius estimate, TE angle."""
    xg, yu, yl = split_surfaces(c, 201)
    t = yu - yl
    cam = 0.5 * (yu + yl)
    i_t = int(np.argmax(t))
    i_c = int(np.argmax(np.abs(cam)))
    # LE radius from a circle fit through the first few upper/lower points
    pts = np.array([(x, y) for x, y in zip(xg[:6], yu[:6])] +
                   [(x, y) for x, y in zip(xg[1:6], yl[1:6])])
    A = np.column_stack([2 * pts[:, 0], 2 * pts[:, 1], np.ones(len(pts))])
    b = pts[:, 0] ** 2 + pts[:, 1] ** 2
    try:
        sol = np.linalg.lstsq(A, b, rcond=None)[0]
        r_le = float(math.sqrt(max(sol[2] + sol[0] ** 2 + sol[1] ** 2, 0.0)))
    except np.linalg.LinAlgError:
        r_le = 0.0
    te_ang = math.degrees(math.atan2(yu[-2] - yu[-1], xg[-1] - xg[-2])
                          - math.atan2(yl[-2] - yl[-1], xg[-1] - xg[-2]))
    return dict(tc=float(t[i_t]), x_tc=float(xg[i_t]), camber=float(cam[i_c]),
                x_camber=float(xg[i_c]), r_le=min(r_le, 0.2),
                te_angle_deg=abs(te_ang), area=float(np.trapezoid(t, xg)))


# --------------------------------------------------------------------------- #
#  CST (Kulfan)                                                                #
# --------------------------------------------------------------------------- #
def _bernstein(psi: np.ndarray, n_w: int) -> np.ndarray:
    n = n_w - 1
    cols = [math.comb(n, i) * psi ** i * (1.0 - psi) ** (n - i) for i in range(n_w)]
    return np.column_stack(cols)


def cst_coords(w_upper, w_lower, n: int = 81) -> np.ndarray:
    """Class-shape-transformation loop from two weight vectors (sharp TE)."""
    wu = np.asarray(w_upper, dtype=float)
    wl = np.asarray(w_lower, dtype=float)
    psi = 0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, n)))
    cls = np.sqrt(psi) * (1.0 - psi)
    yu = cls * (_bernstein(psi, wu.size) @ wu)
    yl = cls * (_bernstein(psi, wl.size) @ wl)
    loop = np.concatenate([np.column_stack([psi, yu])[::-1],
                           np.column_stack([psi, yl])[1:]])
    return normalise_loop(loop)


def fit_cst(c: np.ndarray, n_cst: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares CST weights (upper, lower) for a loop."""
    xg, yu, yl = split_surfaces(c, 201)
    psi = xg[1:-1]
    cls = np.sqrt(psi) * (1.0 - psi)
    B = _bernstein(psi, n_cst) * cls[:, None]
    wu = np.linalg.lstsq(B, yu[1:-1], rcond=None)[0]
    wl = np.linalg.lstsq(B, yl[1:-1], rcond=None)[0]
    return wu, wl


# --------------------------------------------------------------------------- #
#  the spec the library stores                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class AirfoilSpec:
    """A section the library knows by name. `source` says where the shape
    came from: 'naca' (regenerated from `code`), 'dat' (bundled file
    `file`), 'cst' (weights), 'coords' (an explicit loop, e.g. an import or
    an optimiser result)."""

    name: str
    source: str = "naca"
    code: str = "2412"
    file: str = ""
    w_upper: list = field(default_factory=list)
    w_lower: list = field(default_factory=list)
    points: list = field(default_factory=list)
    notes: str = ""
    builtin: bool = False

    _cache: np.ndarray | None = field(default=None, repr=False, compare=False)

    def coords(self) -> np.ndarray:
        if self._cache is None:
            if self.source == "naca":
                c = naca4_coords(self.code)
            elif self.source == "dat":
                path = self.file if os.path.isabs(self.file) else os.path.join(DATA_DIR, self.file)
                c = load_dat(path)[1]
            elif self.source == "cst":
                c = cst_coords(self.w_upper, self.w_lower)
            elif self.source == "coords":
                c = normalise_loop(np.asarray(self.points, dtype=float))
            else:
                raise ValueError(f"unknown airfoil source {self.source!r}")
            self._cache = resample(c, 81)
        return self._cache

    def geometry(self) -> dict:
        return geometry(self.coords())

    def to_json(self) -> dict:
        d = asdict(self)
        d.pop("_cache", None)
        if self.source != "coords":
            d["points"] = []
        else:
            d["points"] = [[round(float(x), 6), round(float(y), 6)] for x, y in self.points]
        return d

    @classmethod
    def from_json(cls, d: dict) -> "AirfoilSpec":
        return cls(name=str(d["name"]), source=str(d.get("source", "naca")),
                   code=str(d.get("code", "2412")), file=str(d.get("file", "")),
                   w_upper=list(d.get("w_upper", [])), w_lower=list(d.get("w_lower", [])),
                   points=list(d.get("points", [])), notes=str(d.get("notes", "")),
                   builtin=bool(d.get("builtin", False)))

    def fingerprint(self) -> str:
        """A short stable hash of the shape (the polar cache key)."""
        import hashlib
        c = self.coords()
        return hashlib.sha1(np.round(c, 6).tobytes()).hexdigest()[:16]


def bundled_dat_files() -> list[str]:
    try:
        return sorted(f for f in os.listdir(DATA_DIR) if f.lower().endswith(".dat"))
    except OSError:
        return []


# --------------------------------------------------------------------------- #
def self_check(verbose: bool = True) -> bool:
    ok = True

    def rep(tag, passed, msg=""):
        nonlocal ok
        ok = ok and passed
        if verbose:
            print(f"  [{'ok' if passed else 'FAIL'}] {tag}: {msg}")

    c = naca4_coords("0012")
    g = geometry(c)
    rep("naca0012 thickness", abs(g["tc"] - 0.12) < 0.002, f"t/c {g['tc']:.4f}")
    rep("naca0012 symmetric", abs(g["camber"]) < 1e-6, f"camber {g['camber']:.2e}")
    rep("sharp TE", np.allclose(c[0], c[-1]) and c[0, 0] > 0.999, f"TE {c[0]}")
    rep("XFOIL order (upper first)", c[len(c) // 4, 1] > 0.0, "")
    c2 = naca4_coords("4412")
    g2 = geometry(c2)
    rep("naca4412 camber 4 % at 40 %", abs(g2["camber"] - 0.04) < 0.002 and abs(g2["x_camber"] - 0.4) < 0.05,
        f"camber {g2['camber']:.4f} at x {g2['x_camber']:.2f}")
    n_ok = 0
    for f in bundled_dat_files():
        try:
            name, loop = load_dat(os.path.join(DATA_DIR, f))
            gg = geometry(loop)
            if 0.03 < gg["tc"] < 0.30:
                n_ok += 1
        except ValueError as exc:
            if verbose:
                print(f"     {f}: {exc}")
    rep("bundled .dat files parse", n_ok == len(bundled_dat_files()) and n_ok > 20,
        f"{n_ok}/{len(bundled_dat_files())}")
    wu, wl = fit_cst(c2, 5)
    c3 = cst_coords(wu, wl)
    xg, yu, yl = split_surfaces(c2)
    _, yu3, yl3 = split_surfaces(c3)
    err = max(np.max(np.abs(yu - yu3)), np.max(np.abs(yl - yl3)))
    rep("CST refit of 4412 within 5e-3 c (LE camber slope is outside the class fn)", err < 5e-3, f"max dev {err:.2e}")
    spec = AirfoilSpec("test", "cst", w_upper=list(wu), w_lower=list(wl))
    back = AirfoilSpec.from_json(json.loads(json.dumps(spec.to_json())))
    rep("spec json round-trip", np.allclose(back.coords(), spec.coords()), back.fingerprint())
    if verbose:
        print("  ALL PASS" if ok else "  FAILURES ABOVE")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_check() else 1)
