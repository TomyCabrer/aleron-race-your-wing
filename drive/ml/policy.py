"""The policy: a 17 -> 16 -> 4 (or 5) tanh net, and the observation it reads.

Deliberately small. 356 parameters (373 with the free-wings head) is enough
for a feedback controller on physical states (this is not pixels), it keeps
an ES population or a swarm cheap, and it is small enough to read out of a
JSON file and check by eye.

Everything here is pure numpy and pure in its inputs. `act()` allocates one
(16,) and one (4,) or (5,) array per call; at dt = 2 ms that is 500 calls a
second of simulated time, which measures as under 4 % of a physics step. It
is never called from `Vehicle.step` — it sits in the driver's seat, outside
the integrator, exactly where `ScriptedInput`'s closure sits.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

from .baseline import K_US_RAD_PER_G, AY_PLAN, WHEELBASE, baseline_action

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
    "kappa_4",       # ... 120 m ahead          <- wave 5
    "beta",          # sideslip, rad
    "r_norm",        # yaw rate * 2
    "ay_norm",       # lateral acceleration / 9.81
    "util_f",        # front-axle grip utilisation (qss.fy_max), 1.0 = limit
    "util_r",        # rear-axle ditto
    "wing_dep",      # the flank panel's deploy fraction, 0..1
    "on_track",      # 1.0 on tarmac, 0.0 off it
    "mu_here",       # surface scale under the car, 1.0 = dry tarmac  <- wave 5
    "mu_ahead",      # WORST surface scale in the next 90 m           <- wave 5
)
#: 14 through wave 4, 17 from wave 5. The three additions are the whole of
#: wave 5 and they are not cosmetic: `kappa_4` is the only thing that lets the
#: 540i brake for `open`'s corner from 41 m/s, and `mu_here` / `mu_ahead` are
#: the first surface information this package has ever had. Through wave 4 the
#: driver could not see the arena's `WET_T3` patch and a blanket
#: `baseline.MARGIN_FADE` paid for that everywhere, including where the track
#: was dry.
#:
#: Changing this INVALIDATES EVERY CHECKPOINT -- the parameter count moves with
#: it (308 -> 356) and there is no meaningful way to carry weights across. See
#: `Policy.load`, which refuses rather than reinterpreting.
N_OBS = len(OBS_NAMES)

#: The action, in order. `pedal` is ONE axis because a real driver has one
#: right foot: positive is throttle, negative is brake, and the policy cannot
#: physically ask for both at once (which a two-headed output does, and which
#: was the first thing an early version learned to do).
ACT_NAMES = ("steer", "pedal", "wing", "spare")
N_ACT = len(ACT_NAMES)

#: FREE WINGS (the swarm's head). Five outputs: the two the driver has, then
#: ONE PER WING -- left flank, right flank, top -- each a tanh threshold on
#: `Controls.wing_cmd`, so the car decides for itself which panel to run and
#: may run both flanks at once as an air brake, all three for a braking
#: zone, two for a corner, or none. `Policy` reads its head off
#: `theta.size`, so a 4-output checkpoint (every ES checkpoint in
#: `checkpoints/`) still loads and drives exactly as before; `Policy.widen`
#: turns one into a 5-output genome that starts by doing what it did.
ACT_NAMES_FREE = ("steer", "pedal", "wing_l", "wing_r", "wing_top")
N_ACT_FREE = len(ACT_NAMES_FREE)

#: With the free head the anchor's wing verdicts are PRIORS, not orders.
#: `expand_base` hands the residual +-WING_PRIOR per wing (the sign is the
#: anchor's choice) and the network's three wing outputs are composed at
#: WING_GAIN -- full authority -- rather than at RESID_GAIN, so an output
#: past -+WING_PRIOR overrules the anchor: deploys a wing it would have kept
#: in, or stows one it would have run. theta = 0 still drives exactly the
#: anchor's choice, and the swarm is free to learn any deployment pattern.
#:
#: Why: through task 17 the verdicts were +-1 and the wing outputs were
#: composed at RESID_GAIN = 0.55 like the steer and pedal, so the composed
#: wing entry sat in [-1, -0.45] or [0.45, 1] whatever the net said and
#: could never cross `WING_ON_THRESH`. The 'free-wings' head was free in
#: name only -- measured: a net forcing +1 or -1 on every wing output gave
#: the same wing_frac and the same reward as theta = 0, on BOTH heads -- and
#: every wing decision in every checkpoint was the anchor's rule. The
#: 4-output head is deliberately left as it was (its wing output is still
#: the anchor's), so every committed ES checkpoint's numbers are unmoved.
WING_PRIOR = 0.5
WING_GAIN = 1.0

#: `expand_base`: the anchor's ONE wing verdict spread over three wings so
#: that theta = 0 with a free head drives like the published car -- the
#: panel on the outer flank of the corner ahead (a left turn is `kappa > 0`,
#: its outer flank the RIGHT one), and the top wing under the 'active' law
#: (braking, or steering). Curvature in the observation's units (1/m * 50);
#: 0.15 is a 330 m radius.
KAPPA_SIDE = 0.15

N_HID = 16

#: The CORSA's road-wheel lock, rad, and the DEFAULT only. The policy's steer
#: output is tanh, so +-1 is full lock and it can never command more.
#:
#: It used to be the only lock this module knew, which was fine while the sim
#: had one car and silently wrong the moment it had three: `vehicle.py` takes
#: the lock from the car (`car_lock_rad`) and `input.py` from `lock_deg`, so a
#: policy driving an MX-5 would have been commanding 32.625 deg of a 31.2 deg
#: rack. `action` and `controls` take a `lock_rad` argument and `env.rollout`
#: passes `veh.lock_rad`; the default keeps every existing call and every
#: committed checkpoint's measured numbers identical, because for the Corsa
#: `car_lock_rad(car) == LOCK_RAD` exactly.
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

#: The per-output residual gain of the free head: steer and pedal at
#: `RESID_GAIN`, the three wings at `WING_GAIN` (see `WING_PRIOR`).
_GAIN_FREE = np.array([RESID_GAIN, RESID_GAIN, WING_GAIN, WING_GAIN, WING_GAIN])

#: Set False to evaluate the network on its own, with no baseline underneath --
#: used by the self-check to prove the residual is what is doing the work.
RESIDUAL = True


def expand_base(base, obs) -> np.ndarray:
    """(steer, pedal, wing) -> (steer, pedal, wing_l, wing_r, wing_top): the
    anchor's one verdict handed to the wing the published law would have
    used, as a +-WING_PRIOR prior the network can overrule. See
    `KAPPA_SIDE`, `WING_PRIOR`."""
    k_path = 0.6 * obs[3] + 0.4 * obs[4]
    w = WING_PRIOR if base[2] > 0.0 else -WING_PRIOR
    left = w if k_path < -KAPPA_SIDE else -WING_PRIOR   # a right turn: left is outer
    right = w if k_path > KAPPA_SIDE else -WING_PRIOR
    top = w if (base[1] < -0.05 or abs(base[0]) > 0.05) else -WING_PRIOR
    return np.array([base[0], base[1], left, right, top])


class Policy:
    """A flat parameter vector, viewed as two affine layers with a tanh.

    `theta` is the whole genome and the ES only ever sees it as a vector;
    `W1/b1/W2/b2` are views onto it, so there is no copy and no way for the
    two representations to disagree.
    """

    __slots__ = ("theta", "W1", "b1", "W2", "b2", "meta", "residual", "n_act")

    N_PARAM = N_OBS * N_HID + N_HID + N_HID * N_ACT + N_ACT      # the 4-output head

    @staticmethod
    def n_param(n_act: int = N_ACT) -> int:
        return N_OBS * N_HID + N_HID + N_HID * n_act + n_act

    @staticmethod
    def head_of(n_param: int) -> int:
        """The output count a parameter count implies, or a ValueError."""
        rest = n_param - (N_OBS * N_HID + N_HID)
        if rest <= 0 or rest % (N_HID + 1):
            raise ValueError(
                f"policy needs {Policy.N_PARAM} (4 outputs) or "
                f"{Policy.n_param(N_ACT_FREE)} (free wings) parameters, got {n_param}")
        return rest // (N_HID + 1)

    def __init__(self, theta=None, meta: dict | None = None,
                 residual: bool = RESIDUAL):
        if theta is None:
            theta = np.zeros(self.N_PARAM)
        theta = np.asarray(theta, dtype=np.float64).ravel()
        n_act = self.head_of(theta.size)
        if n_act not in (N_ACT, N_ACT_FREE):
            raise ValueError(f"policy has {n_act} outputs; 4 or {N_ACT_FREE} are the heads")
        self.n_act = n_act
        self.theta = theta
        i = 0
        self.W1 = theta[i:i + N_OBS * N_HID].reshape(N_HID, N_OBS); i += N_OBS * N_HID
        self.b1 = theta[i:i + N_HID]; i += N_HID
        self.W2 = theta[i:i + N_HID * n_act].reshape(n_act, N_HID); i += N_HID * n_act
        self.b2 = theta[i:i + n_act]
        self.meta = dict(meta or {})
        self.residual = bool(residual)

    @property
    def free_wings(self) -> bool:
        return self.n_act >= N_ACT_FREE

    @property
    def n_out(self) -> int:
        """The outputs the residual composes: 3 (steer, pedal, wing) or 5."""
        return N_ACT_FREE if self.free_wings else 3

    @staticmethod
    def widen(theta) -> np.ndarray:
        """A 4-output genome -> the 5-output one that DRIVES the same way:
        the steer and pedal rows carry over and the three wing rows are
        ZERO. The 4-output head's wing row never had any authority (see
        `WING_PRIOR`: its composed wing entry could not cross the
        threshold), so that genome's wing behaviour WAS the anchor's rule,
        and zero rows plus the free head's priors reproduce exactly that;
        the swarm then learns to split the wings. Copying the old row, as
        this used to, would have handed never-selected weights real
        authority. A 5-output genome is returned as is."""
        theta = np.asarray(theta, float).ravel()
        if Policy.head_of(theta.size) == N_ACT_FREE:
            return theta.copy()
        p = Policy(theta)
        z = np.zeros(N_HID)
        W2 = np.vstack([p.W2[0], p.W2[1], z, z, z])
        b2 = np.array([p.b2[0], p.b2[1], 0.0, 0.0, 0.0])
        return np.concatenate([p.W1.ravel(), p.b1, W2.ravel(), b2])

    # ---- the forward pass ------------------------------------------------
    def act(self, obs) -> np.ndarray:
        """(N_OBS,) -> (N_ACT,) in [-1, 1]. Pure; no state, so a replay of the
        same observations is bit-identical."""
        h = np.tanh(self.W1 @ obs + self.b1)
        return np.tanh(self.W2 @ h + self.b2)

    def action(self, obs, lock_rad: float = LOCK_RAD,
               wheelbase: float = WHEELBASE,
               ay_plan: float = AY_PLAN,
               k_us: float = K_US_RAD_PER_G) -> np.ndarray:
        """(steer, pedal, wing) in [-1, 1]: the baseline plus this net's trim
        -- or, with the free head, (steer, pedal, wing_l, wing_r, wing_top),
        the three wings composed on the anchor's priors at `WING_GAIN`.

        Clipped, so the composed action can never ask for more than full lock
        or more than a full pedal however large the network's output grows.

        The four per-car arguments go to the ANCHOR only -- the network reads
        none of them. That is deliberate: the parameters are a trim in the
        car's own actuator units (a fraction of ITS lock, a fraction of ITS
        pedal), so a checkpoint means the same thing on any car, and the
        cross-car matrix in `evaluate.car_transfer` is comparing the same
        policy rather than the same numbers meaning different angles. They
        come from `baseline.driver_trim(car)`, which `env.rollout` calls once
        per rollout; all four are exactly the Corsa's values for the Corsa.
        """
        net = self.act(obs)
        n = self.n_out
        if not self.residual:
            return net[:n]
        base = baseline_action(obs, lock_rad, wheelbase, ay_plan, k_us)
        if self.free_wings:
            #  the wings at full authority over the anchor's +-WING_PRIOR
            return np.clip(expand_base(base, obs) + _GAIN_FREE * net, -1.0, 1.0)
        return np.clip(base + RESID_GAIN * net[:n], -1.0, 1.0)

    def controls(self, obs, Controls, lock_rad: float = LOCK_RAD,
                 wheelbase: float = WHEELBASE, ay_plan: float = AY_PLAN,
                 k_us: float = K_US_RAD_PER_G):
        """The action decoded into a `vehicle.Controls`.

        `Controls` is passed in rather than imported so this module has no
        import-time dependency on the physics at all -- `drive.ml.policy` is
        importable with `drive.vehicle` absent, which is what keeps the
        self-check cheap and the package honestly additive.
        """
        a = self.action(obs, lock_rad, wheelbase, ay_plan, k_us)
        pedal = float(a[1])
        if self.free_wings:
            cmd = (bool(a[2] > WING_ON_THRESH), bool(a[3] > WING_ON_THRESH),
                   bool(a[4] > WING_ON_THRESH))
            return Controls(
                delta=float(a[0]) * lock_rad,
                throttle=max(pedal, 0.0),
                brake=max(-pedal, 0.0),
                wing_on=any(cmd), wing_cmd=cmd,
                auto_gearbox=True, auto_clutch=True,
            )
        return Controls(
            delta=float(a[0]) * lock_rad,
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
                kind="drive.ml.Policy", n_obs=N_OBS, n_hid=N_HID, n_act=self.n_act,
                residual=self.residual,
                obs_names=list(OBS_NAMES),
                act_names=list(ACT_NAMES_FREE if self.free_wings else ACT_NAMES),
                meta=self.meta,
                theta=[float(v) for v in self.theta],
            ), fh, indent=1)
        return path

    @classmethod
    def load(cls, path: str) -> "Policy":
        with open(path) as fh:
            d = json.load(fh)
        if int(d.get("n_obs", N_OBS)) != N_OBS or int(d.get("n_hid", N_HID)) != N_HID:
            raise ValueError(
                f"{path} was built for {d.get('n_obs')} observations x "
                f"{d.get('n_hid')} hidden units; THIS BUILD OF drive.ml IS "
                f"{N_OBS}x{N_HID}.\n"
                f"  The observation changed, so this checkpoint's {len(d.get('theta', []))} "
                f"parameters mean something else now and there is no honest way "
                f"to reinterpret them -- the weights are indexed by observation "
                f"and two of the new ones (mu_here, mu_ahead) did not exist when "
                f"it was trained.\n"
                f"  RETRAIN it:  python3 -m drive.ml.train --track "
                f"{d.get('meta', {}).get('track', 'arena')} "
                f"--car {d.get('meta', {}).get('car', 'corsa')} --iters 90 "
                f"--pop 32 --sigma 0.08 --lr 0.05 --duration 70 --out {path}\n"
                f"  (OBS_NAMES in drive/ml/policy.py records what changed and "
                f"when; .handoff/09-ml.md records why.)")
        return cls(np.asarray(d["theta"], float), d.get("meta"),
                   residual=bool(d.get("residual", True)))

    @classmethod
    def random(cls, seed: int = 0, scale: float = 0.5, n_act: int = N_ACT) -> "Policy":
        rng = np.random.default_rng(seed)
        return cls(rng.normal(0.0, scale, cls.n_param(n_act)))

    def __repr__(self):
        return (f"Policy({self.theta.size} params, {self.n_act} outputs, |theta| "
                f"{float(np.linalg.norm(self.theta)):.3f}, meta {self.meta})")
