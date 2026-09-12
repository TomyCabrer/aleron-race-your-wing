"""Hess-Smith panel method for one section: constant-strength source panels
plus one vortex density shared by every panel, closed by the Kutta
condition. Inviscid, incompressible, ~1 ms for a 160-node loop.

What it is for here: the section's LIFT SLOPE `a_lin`, its zero-lift angle
`alpha_L0`, its quarter-chord moment and its pressure distribution -- the
numbers the vortex lattice needs when XFOIL is not on the machine, and the
Cp plot the designer looks at either way. It knows nothing about viscosity:
drag and stall come from polar.py (XFOIL or the labelled estimate).

Kernels are Katz & Plotkin's straight-segment integrals in the panel frame
(xi along the panel, zeta = xi rotated +90 deg):

    source  u_xi = (s/4pi) ln(r1^2/r2^2)     u_zeta = (s/2pi)(th2 - th1)
    vortex  u_xi = -(g/2pi)(th2 - th1)       u_zeta = (g/4pi) ln(r1^2/r2^2)

with r_k the distance to panel end k and th_k = atan2(zeta, xi - xi_k).
The XFOIL order (TE -> upper -> LE -> lower -> TE) is a COUNTER-clockwise
traversal in (x right, y up), so the outward normal is the right-rotated
tangent and the self-induced terms are taken on the -zeta side; either
orientation is accepted (the signed area decides). The Kutta condition is
v_t(first) + v_t(last) = 0: equal speed leaving the trailing edge on both
surfaces, since the first panel points away from it and the last towards it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TWO_PI = 2.0 * math.pi


@dataclass
class PanelResult:
    alpha_deg: float
    cl: float
    cm_c4: float            # nose-up positive, about the quarter chord
    cp: np.ndarray          # per panel
    x_mid: np.ndarray
    y_mid: np.ndarray
    cp_min: float
    kutta_residual: float
    gamma: float            # vortex density (circulation per unit length)


class Section2D:
    """Build once per loop (matrix + LU), solve for any alpha."""

    def __init__(self, coords: np.ndarray):
        c = np.asarray(coords, dtype=float)
        if c.shape[0] < 20:
            raise ValueError("need at least 20 nodes")
        if np.allclose(c[0], c[-1]):
            nodes = c[:-1]
        else:
            nodes = c
        self.nodes = nodes
        n = nodes.shape[0]
        a = nodes
        b = np.roll(nodes, -1, axis=0)
        d = b - a
        L = np.hypot(d[:, 0], d[:, 1])
        if np.any(L < 1e-12):
            raise ValueError("degenerate (zero-length) panel")
        t = d / L[:, None]
        # signed area: negative = clockwise (XFOIL order) -> outward is left
        area = 0.5 * float(np.sum(a[:, 0] * b[:, 1] - b[:, 0] * a[:, 1]))
        self.clockwise = area < 0.0
        if self.clockwise:
            nrm = np.column_stack([-t[:, 1], t[:, 0]])
        else:
            nrm = np.column_stack([t[:, 1], -t[:, 0]])
        mid = 0.5 * (a + b)
        self.n_panels = n
        self.L, self.t, self.nrm, self.mid = L, t, nrm, mid
        self.chord = float(np.ptp(nodes[:, 0]))
        self.x_le = float(nodes[:, 0].min())

        # ---- influence of panel j on the midpoint of panel i --------------
        # field points in panel j's frame
        rel = mid[:, None, :] - a[None, :, :]                 # (i, j, 2)
        xi = rel[..., 0] * t[None, :, 0] + rel[..., 1] * t[None, :, 1]
        ze = -rel[..., 0] * t[None, :, 1] + rel[..., 1] * t[None, :, 0]   # +90 deg rot
        s1 = xi
        s2 = xi - L[None, :]
        r1sq = s1 * s1 + ze * ze
        r2sq = s2 * s2 + ze * ze
        th1 = np.arctan2(ze, s1)
        th2 = np.arctan2(ze, s2)
        lnr = 0.5 * np.log(np.maximum(r1sq, 1e-30) / np.maximum(r2sq, 1e-30))
        dth = th2 - th1
        # self-influence: the midpoint sits on the panel (zeta = 0+): the
        # normal source velocity is s/2 and the tangential vortex velocity
        # -g/2 (limit of the subtended angle -> pi); ln term is exactly 0
        # ...on whichever side the OUTSIDE is: +zeta (left of travel) for a
        # clockwise loop, -zeta for the counter-clockwise XFOIL order
        ii = np.arange(n)
        lnr[ii, ii] = 0.0
        dth[ii, ii] = math.pi if self.clockwise else -math.pi
        us_xi = lnr / TWO_PI            # source, per unit density
        us_ze = dth / TWO_PI
        uv_xi = -dth / TWO_PI           # vortex, per unit density
        uv_ze = lnr / TWO_PI
        # back to global: u = u_xi * t_j + u_ze * (rot90 t_j)
        tj = t[None, :, :]
        rj = np.stack([-t[:, 1], t[:, 0]], axis=1)[None, :, :]
        Us = us_xi[..., None] * tj + us_ze[..., None] * rj        # (i, j, 2)
        Uv = uv_xi[..., None] * tj + uv_ze[..., None] * rj
        ni = nrm[:, None, :]
        ti = t[:, None, :]
        self.An = np.einsum("ijk,ijk->ij", Us, ni)            # source -> normal
        self.At = np.einsum("ijk,ijk->ij", Us, ti)            # source -> tangential
        self.Bn = np.einsum("ijk,ijk->ij", Uv, ni).sum(axis=1)   # vortex (all panels)
        self.Bt = np.einsum("ijk,ijk->ij", Uv, ti).sum(axis=1)
        A = np.zeros((n + 1, n + 1))
        A[:n, :n] = self.An
        A[:n, n] = self.Bn
        A[n, :n] = self.At[0] + self.At[n - 1]
        A[n, n] = self.Bt[0] + self.Bt[n - 1]
        self._A = A
        try:
            from scipy.linalg import lu_factor, lu_solve
            self._lu = lu_factor(A)
            self._solve = lambda rhs: lu_solve(self._lu, rhs)
        except Exception:                                    # pragma: no cover
            self._solve = lambda rhs: np.linalg.solve(A, rhs)

    def solve(self, alpha_deg: float, V: float = 1.0) -> PanelResult:
        n = self.n_panels
        al = math.radians(alpha_deg)
        Vinf = V * np.array([math.cos(al), math.sin(al)])
        rhs = np.empty(n + 1)
        rhs[:n] = -(self.nrm @ Vinf)
        rhs[n] = -((self.t[0] + self.t[n - 1]) @ Vinf)
        sol = self._solve(rhs)
        sigma, gamma = sol[:n], float(sol[n])
        vt = self.t @ Vinf + self.At @ sigma + self.Bt * gamma
        cp = 1.0 - (vt / V) ** 2
        # dF/q = -Cp n L (pressure acts against the outward normal)
        dF = -(cp * self.L)[:, None] * self.nrm
        F = dF.sum(axis=0)
        c = self.chord
        lift_dir = np.array([-math.sin(al), math.cos(al)])
        cl = float(F @ lift_dir) / c
        x0 = self.x_le + 0.25 * c
        Mz = float(np.sum((self.mid[:, 0] - x0) * dF[:, 1] - self.mid[:, 1] * dF[:, 0]))
        cm = -Mz / (c * c)
        return PanelResult(alpha_deg=float(alpha_deg), cl=cl, cm_c4=cm, cp=cp,
                           x_mid=self.mid[:, 0], y_mid=self.mid[:, 1],
                           cp_min=float(cp.min()),
                           kutta_residual=float(abs(vt[0] + vt[n - 1]) / V),
                           gamma=gamma)

    def linear(self) -> dict:
        """Lift slope a_lin [/rad], zero-lift angle [deg], cm about c/4 at
        zero lift, from two solves (cl is linear in sin(alpha) here)."""
        r0 = self.solve(0.0)
        r1 = self.solve(4.0)
        a = (r1.cl - r0.cl) / math.radians(4.0)
        al0 = -r0.cl / a if a > 1e-9 else 0.0
        return dict(a_lin=float(a), alpha_L0_deg=float(math.degrees(al0)),
                    cm0=float(r0.cm_c4 - (r1.cm_c4 - r0.cm_c4) * r0.cl / max(r1.cl - r0.cl, 1e-9)),
                    cl0=float(r0.cl))


def solve(coords: np.ndarray, alpha_deg: float) -> PanelResult:
    return Section2D(coords).solve(alpha_deg)


# --------------------------------------------------------------------------- #
def self_check(verbose: bool = True) -> bool:
    from . import airfoil as af
    ok = True

    def rep(tag, passed, msg=""):
        nonlocal ok
        ok = ok and passed
        if verbose:
            print(f"  [{'ok' if passed else 'FAIL'}] {tag}: {msg}")

    thin = Section2D(af.naca4_coords("0002", 121))
    r = thin.solve(4.0)
    exact = TWO_PI * math.sin(math.radians(4.0))
    rep("thin section ~ flat plate 2 pi sin(a)", abs(r.cl / exact - 1.0) < 0.04,
        f"cl {r.cl:.4f} vs {exact:.4f} ({100 * (r.cl / exact - 1):+.2f} %)")
    s12 = Section2D(af.naca4_coords("0012", 121))
    lin = s12.linear()
    rep("naca0012 lift slope 6.5-7.4 /rad (thickness raises 2 pi)",
        6.5 < lin["a_lin"] < 7.4, f"a {lin['a_lin']:.3f} /rad")
    rep("naca0012 alpha_L0 = 0", abs(lin["alpha_L0_deg"]) < 1e-6, f"{lin['alpha_L0_deg']:.2e} deg")
    r = s12.solve(6.0)
    rep("Kutta residual", r.kutta_residual < 1e-9, f"{r.kutta_residual:.1e}")
    rep("naca0012 cm_c4 ~ 0", abs(r.cm_c4) < 0.01, f"cm {r.cm_c4:+.4f}")
    rep("odd in alpha", abs(s12.solve(-6.0).cl + r.cl) < 1e-9, "")
    l44 = Section2D(af.naca4_coords("4412", 121)).linear()
    rep("naca4412 alpha_L0 -3.6..-4.5 deg", -4.5 < l44["alpha_L0_deg"] < -3.6,
        f"{l44['alpha_L0_deg']:.2f} deg (thin-airfoil -4.15)")
    rep("naca4412 cm0 -0.08..-0.12 nose-down", -0.12 < l44["cm0"] < -0.08, f"cm0 {l44['cm0']:+.4f}")
    import time
    t0 = time.perf_counter()
    for _ in range(5):
        Section2D(af.naca4_coords("2412", 81)).solve(3.0)
    ms = (time.perf_counter() - t0) * 1e3 / 5
    rep("cost", ms < 30.0, f"{ms:.1f} ms build+solve (161 nodes)")
    if verbose:
        print("  ALL PASS" if ok else "  FAILURES ABOVE")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_check() else 1)
