"""drive/aero -- the wing-design physics the garage runs, numpy only.

A lean port of the car-wing family of ~/dev/urop-bo-aero (the AeroBO design
tool): a section (NACA-4 / UIUC .dat / CST shape) is turned into a polar
(XFOIL when the binary is on this machine, a labelled ESTIMATE when it is
not), the polar feeds a horseshoe vortex lattice with endplates and the
track's rigid-wall image, and a small Gaussian-process Bayesian optimiser
searches the planform. What leaves this package is a handful of numbers the
1 kHz physics can afford per step (S, CL0, CL_alpha, CL_max, CD0, k), never
a solver call.

Modules
    airfoil   coordinates: NACA-4, .dat reader, CST fit/synthesis, geometry
    panel2d   Hess-Smith panel method (inviscid cl, cm, Cp, lift slope, alpha_L0)
    polar     the polar table + the viscous ESTIMATE fallback
    xfoil     the XFOIL subprocess wrapper with an on-disk cache
    vlm       the vortex lattice (imaged, endplated), Trefftz induced drag
    wing      WingSpec -> WingAero: what the vehicle reads
    optimize  GP-BO + random baseline on a WingSpec's planform
    library   airfoils / wings / builds on disk (runs/library)

No pygame anywhere in here: the garage draws, this computes.
"""

__all__ = ["airfoil", "panel2d", "polar", "xfoil", "vlm", "wing", "optimize", "library"]
