"""The policy: a 14 -> 16 -> 4 tanh net, and the observation it reads.

Deliberately small. 308 parameters is enough for a feedback controller on
physical states (this is not pixels), it keeps the ES population cheap, and it
is small enough to read out of a JSON file and check by eye.

Everything here is pure numpy and pure in its inputs. `act()` allocates one
(16,) and one (4,) array per call; at dt = 2 ms that is 500 calls a second of
simulated time, which measures as under 4 % of a physics step. It is never
called from `Vehicle.step` — it sits in the driver's seat, outside the
integrator, exactly where `ScriptedInput`'s closure sits.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

from .baseline import baseline_action

#: The observation, in order. Every entry is a read-only quantity the contract
#: already publishes (CONTRACT section 4's attribute list) or a pure function
#: of the track geometry (`track.project`) -- nothing is smuggled out of the
#: integrator, and nothing here is unavailable to a human driver looking out
#: of the windscreen and at the HUD.
OBS_NAMES = (
    "u_norm",        # forward speed / 40 m/s
    "n_norm",        # lateral offset from the centreline / half-width, +LEFT
    "psi_err",       # heading error against the centreline tangent, rad
    "kappa_0",       # centreline curvature here, 1/m * 50
    "kappa_1",       # ... 15 m ahead
    "kappa_2",       # ... 35 m ahead
    "kappa_3",       # ... 70 m ahead
    "beta",          # sideslip, rad
    "r_norm",        # yaw rate * 2
    "ay_norm",       # lateral acceleration / 9.81
    "util_f",        # front-axle grip utilisation (qss.fy_max), 1.0 = limit
    "util_r",        # rear-axle ditto
    "wing_dep",      # the flank panel's deploy fraction, 0..1
    "on_track",      # 1.0 on tarmac, 0.0 off it
)
N_OBS = len(OBS_NAMES)

#: The action, in order. `pedal` is ONE axis because a real driver has one
#: right foot: positive is throttle, negative is brake, and the policy cannot
#: physically ask for both at once (which a two-headed output does, and which
#: was the first thing an early version learned to do).
ACT_NAMES = ("steer", "pedal", "wing", "spare")
N_ACT = len(ACT_NAMES)

N_HID = 16

#: road-wheel lock, rad. `input.DELTA_LOCK_DEG = 32.625`; the policy's steer
#: output is tanh, so +-1 is full lock and it can never command more.
LOCK_RAD = math.radians(32.625)

#: The wing is armed when the `wing` output clears this. A threshold, not a
#: continuous deploy, because `Controls.wing_on` is the driver's BOOLEAN
#: toggle -- the 0.45 s actuator lag lives in `vehicle.py` and the policy must
#: learn to live with it rather than being handed a continuous actuator the
#: real car does not have.
WING_ON_THRESH = 0.0

#: The network is a RESIDUAL on `baseline.baseline_action`, and this is its
#: authority. `theta = 0` gives exactly the hand-written driver, so the ES
#: starts on the track at racing speed and spends its budget on the LINE and
#: the AERO rather than on rediscovering that grass is slow. 0.55 lets it
#: override the baseline's steering entirely at full output (the baseline
#: rarely uses more than 0.3 of lock) while keeping small outputs as trims.
RESID_GAIN = 0.55

#: Set False to evaluate the network on its own, with no baseline underneath --
#: used by the self-check to prove the residual is what is doing the work.
RESIDUAL = True


class Policy:
    """A flat parameter vector, viewed as two affine layers with a tanh.

    `theta` is the whole genome and the ES only ever sees it as a vector;
    `W1/b1/W2/b2` are views onto it, so there is no copy and no way for the
    two representations to disagree.
    """

    __slots__ = ("theta", "W1", "b1", "W2", "b2", "meta", "residual")

    N_PARAM = N_OBS * N_HID + N_HID + N_HID * N_ACT + N_ACT

    def __init__(self, theta=None, meta: dict | None = None,
                 residual: bool = RESIDUAL):
        if theta is None:
            theta = np.zeros(self.N_PARAM)
        theta = np.asarray(theta, dtype=np.float64).ravel()
        if theta.size != self.N_PARAM:
            raise ValueError(f"policy needs {self.N_PARAM} parameters, got {theta.size}")
        self.theta = theta
        i = 0
        self.W1 = theta[i:i + N_OBS * N_HID].reshape(N_HID, N_OBS); i += N_OBS * N_HID
        self.b1 = theta[i:i + N_HID]; i += N_HID
        self.W2 = theta[i:i + N_HID * N_ACT].reshape(N_ACT, N_HID); i += N_HID * N_ACT
        self.b2 = theta[i:i + N_ACT]
        self.meta = dict(meta or {})
        self.residual = bool(residual)

    # ---- the forward pass ------------------------------------------------
    def act(self, obs) -> np.ndarray:
        """(N_OBS,) -> (N_ACT,) in [-1, 1]. Pure; no state, so a replay of the
        same observations is bit-identical."""
        h = np.tanh(self.W1 @ obs + self.b1)
        return np.tanh(self.W2 @ h + self.b2)

    def action(self, obs) -> np.ndarray:
        """(steer, pedal, wing) in [-1, 1]: the baseline plus this net's trim.

        Clipped, so the composed action can never ask for more than full lock
        or more than a full pedal however large the network's output grows.
        """
        net = self.act(obs)
        if not self.residual:
            return net[:3]
        base = baseline_action(obs, LOCK_RAD)
        return np.clip(base + RESID_GAIN * net[:3], -1.0, 1.0)

    def controls(self, obs, Controls):
        """The action decoded into a `vehicle.Controls`.

        `Controls` is passed in rather than imported so this module has no
        import-time dependency on the physics at all -- `drive.ml.policy` is
        importable with `drive.vehicle` absent, which is what keeps the
        self-check cheap and the package honestly additive.
        """
        a = self.action(obs)
        pedal = float(a[1])
        return Controls(
            delta=float(a[0]) * LOCK_RAD,
            throttle=max(pedal, 0.0),
            brake=max(-pedal, 0.0),
            wing_on=bool(a[2] > WING_ON_THRESH),
            auto_gearbox=True, auto_clutch=True,
        )

    # ---- persistence -----------------------------------------------------
    def save(self, path: str) -> str:
        """JSON, not .npy: a checkpoint is a deliverable someone will read."""
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w") as fh:
            json.dump(dict(
                kind="drive.ml.Policy", n_obs=N_OBS, n_hid=N_HID, n_act=N_ACT,
                residual=self.residual,
                obs_names=list(OBS_NAMES), act_names=list(ACT_NAMES),
                meta=self.meta,
                theta=[float(v) for v in self.theta],
            ), fh, indent=1)
        return path

    @classmethod
    def load(cls, path: str) -> "Policy":
        with open(path) as fh:
            d = json.load(fh)
        if int(d.get("n_obs", N_OBS)) != N_OBS or int(d.get("n_hid", N_HID)) != N_HID:
            raise ValueError(f"{path}: built for {d.get('n_obs')}x{d.get('n_hid')} "
                             f"observations/hidden, this build is {N_OBS}x{N_HID}")
        return cls(np.asarray(d["theta"], float), d.get("meta"),
                   residual=bool(d.get("residual", True)))

    @classmethod
    def random(cls, seed: int = 0, scale: float = 0.5) -> "Policy":
        rng = np.random.default_rng(seed)
        return cls(rng.normal(0.0, scale, cls.N_PARAM))

    def __repr__(self):
        return (f"Policy({self.N_PARAM} params, |theta| "
                f"{float(np.linalg.norm(self.theta)):.3f}, meta {self.meta})")
