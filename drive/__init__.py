"""Real-time drivable simulator for the Corsa C flank-wing study.

The analysis suite one level up (corsa_c / crossover / ledger / qss) answers
"is the device worth it" in closed form and at quasi-steady equilibrium. This
package answers "what does it feel like", in the time domain, with the same
parameters and the same tyre, and it is held to the QSS answers by
drive/validate.py.

Read drive/CONTRACT.md before touching anything here. The subsystem specs the
contract was distilled from are in specs/.

Headless note: SDL_VIDEODRIVER has to be set BEFORE pygame is imported anywhere
in the package, not merely before pygame.display.init(). Setting it in a main()
is too late once pytest or another importer has already pulled the package in.
"""

import os

if os.environ.get("CARSIM_HEADLESS"):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

__all__ = [
    "tyre",
    "powertrain",
    "vehicle",
    "track",
    "input",
    "render",
    "telemetry",
    "plots",
    "drive",
    "validate",
]
