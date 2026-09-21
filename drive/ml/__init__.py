"""`drive.ml` — a driving agent that learns to drive the car AND its aero.

STRICTLY ADDITIVE. Nothing in `drive/` imports this package; the sim behaves
identically with it absent or disabled, and the 1 kHz physics path gains no
import, no allocation and no nondeterminism from it. The only hook is
`drive.drive`'s `--ml-drive PATH`, which wraps a loaded policy in the
`input.ScriptedInput` closure that already exists for the acceptance scripts
(CONTRACT section 6) — so the agent sits exactly where a scripted driver
sits, and `Controls` reaches `Vehicle.step` by the same path it always did.

Why an evolution strategy and not deep RL
-----------------------------------------
`torch` happens to be importable in this environment, but the repo uses none
of it and the brief is to add no dependency. An ES needs only numpy, has no
backward pass to get wrong, is trivially parallel over a process pool, and is
indifferent to the fact that our episode reward is a non-differentiable
function of a 1 kHz ODE integration. The binding constraint here is rollout
throughput, not sample efficiency.

Measured throughput, one core, arena, all three wings, `surface_at` every step:

    dt = 1 ms   16.8x real time      the contract's DT_PHYS
    dt = 2 ms   33.6x real time      0.128 m of drift over 20 s vs 1 ms
    dt = 4 ms   67.2x real time      2.618 m of drift  -- too much
    dt = 8 ms  133.1x real time     25.041 m of drift  -- useless

so **training runs at dt = 2 ms and every reported number is re-measured at
dt = 1 ms.** 2 ms buys a factor of two for 13 cm over 20 s; 4 ms does not.

Entry points
------------
    python3 -m drive.ml.train   --help      train, writes a checkpoint
    python3 -m drive.ml.evaluate --help     measure a checkpoint at DT_PHYS
    python3 -m drive.ml                     self-check (fast, deterministic)
    python3 -m drive.drive --ml-drive drive/ml/checkpoints/<name>.json
                                            watch it drive, with a window
    python3 -m drive.ml.swarm --help        the swarm: a GA, the best reproduce
    python3 -m drive.drive --swarm 32 [--swarm-seed latest]
                                            the same with a window; `latest` is
                                            the last seed lap the USER drove (K)

Checkpoints live in `drive/ml/checkpoints/` and NOT in `runs/`, because
`runs/` is gitignored and a trained policy is a deliverable.
"""

from .policy import Policy, OBS_NAMES, N_OBS, N_ACT
from .env import Episode, rollout, DT_TRAIN, DT_EVAL

__all__ = ["Policy", "OBS_NAMES", "N_OBS", "N_ACT",
           "Episode", "rollout", "DT_TRAIN", "DT_EVAL"]
