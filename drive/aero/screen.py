"""The LIBRARY SCREEN: seven criteria, the user's weights, one score.

This is AeroBO's `airfoil_select.score_candidates` in carsim's units, and it
is the step the garage's AIRFOIL and ENDPLATE groups open on. The procedure it
implements is AeroBO's, in AeroBO's order:

1. the user states WHAT THE SECTION IS FOR, as seven criterion weights;
2. the whole library is scored on those weights and ranked;
3. the winner seeds a shape optimisation which maximises the SAME composite.

Step 1 is the one carsim did not have. The garage used to open the section
group on the eight CST weights -- the search's own design vector -- which is
not a question anyone can answer before the search has run. The weights a
designer actually has an opinion about are these: how much of the score is
cruise L/D, how much is cl_max, how much is the thickness a spar has to live
in. They belong to the screen, because the screen is what they decide.

WHY A FROZEN BAND
-----------------
A sub-score is 0-100 over a (lo, hi) band. Take that band from the live
population and the map moves with the population, so the composite is not a
fixed function of the shape and a search cannot maximise it -- AeroBO's own
argument, and the reason its screen writes a reference out. So `bands()` is
measured ONCE, over the screened library, and `SectionModel` keeps it: the
shortlist is chosen on the map it is judged on. Sub-scores are NOT clipped to
[0, 100]; a designed section that beats every library section scores past the
end, which is the point of running the search at all.

Percentiles rather than (min, max), as AeroBO has them, but the reason is
weaker here and is stated rather than inherited: AeroBO measures p2/p98 over
631 eligible sections, where a single outlier moves a raw span by up to 2.2x.
carsim ships 34. At n = 34 the 2nd percentile sits between the first and
second order statistic, so the band is very nearly the raw range -- the
robustness is nominal. It is kept because the two repositories should not
normalise differently for no reason.

WHAT A PLATE CANNOT BE RANKED ON
--------------------------------
`DEAD` is AeroBO's `DEAD_CRITERIA["plate"]`, shared there with the vertical
stabiliser's entry and shared here for the same reason: an end plate is a
vertical panel at nominally zero incidence. Both L/D criteria are read at a
lift it never carries, and a symmetric section's |cm| about the quarter chord
is zero identically. The rows stay on the form -- "nobody can rank this" and
"you set this to zero" are different sentences -- and a user who weights one
is told what they bought.
"""

from __future__ import annotations

import math

import numpy as np

#: criterion -> (label, what it means). The order is the form's order.
META = {
    "ldcr": ("L/D at the design cl",
             "the number this surface actually flies at, and the criterion "
             "WingLab's own presets put most of the score on"),
    "clmax": ("cl_max",
              "how much lift the section reaches before it stalls -- the "
              "headroom a slow corner asks for"),
    "cm": ("|cm| (lower better)",
           "the pitching moment: mount load, and the twist it drives into a "
           "wing the beam model has to carry"),
    "ldmax": ("(L/D) max",
              "the best point of the polar, wherever it sits. Not where this "
              "surface flies -- read it beside 'L/D at the design cl', not "
              "instead of it"),
    "cdcr": ("cd at the design cl (lower better)",
             "the drag this surface costs at the lift it flies. On a surface "
             "at ZERO lift -- an end plate -- this is the whole drag "
             "criterion, because both L/D criteria are dead there"),
    "thick": ("thickness t/c",
              "structural depth: how much spar fits inside the section"),
    "astall": ("stall angle",
               "how much incidence margin is left before the section lets go"),
}

#: the same seven, named short enough for a bar chart and a table heading.
SHORT = {"ldcr": "L/D @ cl", "clmax": "cl_max", "cm": "|cm|",
         "ldmax": "(L/D) max", "cdcr": "cd @ cl", "thick": "t/c",
         "astall": "stall angle"}

HIGHER_BETTER = ("thick", "clmax", "ldmax", "ldcr", "astall")
LOWER_BETTER = ("cm", "cdcr")
#: canonical order, AeroBO's: the lower-better pair is APPENDED, never
#: inserted, so a stored band's coordinates cannot silently re-order.
CRITERIA = HIGHER_BETTER + LOWER_BETTER

#: The weight sets the form offers.
#:
#: "AeroBO bulk sweep" is `airfoil_select.PRESETS["gdp-sweep"]` verbatim --
#: what AeroBO's `session.JOB_WEIGHTS` hands a surface whose job is "it
#: carries the design load" -- and it is kept so the shipped set is one
#: keypress away. It is NOT the default, and the reason is measured.
#:
#: THE |cm| CRITERION RANKS A CAR WING BACKWARDS. AeroBO already defines the
#: transformation for a surface whose moment something else carries: its
#: `wing-trimmed` job moves the |cm| weight to cruise L/D, because |cm| is
#: LOWER-better and so rewards REFLEX -- "which is what an aerofoil does
#: instead of having a tail". A wing bolted to a car has no tail either: the
#: MOUNT carries its moment, in bending, and the beam is sized for it. And
#: reflex is the opposite of what a downforce wing wants.
#:
#: MEASURED, over the 39 sections carsim ships, on all three circuits and
#: both roles -- Spearman's rho between the composite ranking and the LAP
#: ranking, and the lap the winner actually does against the best section
#: available:
#:
#:     flank, |cm| alone             rho -0.97     it ranks the library BACKWARDS
#:     flank, AeroBO bulk sweep      rho +0.96     mean +0.0059 s (worst +0.0108)
#:     flank, that set with cm = 0   rho +0.97     mean +0.0059 s
#:     flank, cm split over the
#:            three lift criteria    rho +0.98     mean +0.0031 s (worst +0.0054)
#:
#: So the default is the bulk-sweep preset with the |cm| weight removed and
#: its 0.20 redistributed across the three criteria that measure what a
#: downforce wing is for. It halves the lap cost of the section the screen
#: picks, and improves the rank correlation on every one of the six cases.
#:
#: `ldcr` stays the largest single weight, which is AeroBO's own ordering: a
#: heavier `ldmax` scores better on the flank still (+0.0019 s) but it is
#: read at a lift the surface does not fly, which AeroBO's own criterion note
#: warns about, and it is worse on the top slot.
PRESETS: dict[str, dict] = {
    "wing (downforce)": {"thick": 0.10, "clmax": 0.30, "ldmax": 0.25,
                         "ldcr": 0.35, "cm": 0.0, "astall": 0.0, "cdcr": 0.0},
    "WingLab bulk sweep": {"thick": 0.10, "clmax": 0.20, "ldmax": 0.15,
                          "ldcr": 0.35, "cm": 0.20, "astall": 0.0,
                          "cdcr": 0.0},
    #: A CARSIM END PLATE IS NOT AeroBO'S END PLATE, and the two presets
    #: below are that difference. AeroBO's plate is a FENCE at zero toe on a
    #: symmetric car: `CarWingEndplateProblem` refuses a cambered section
    #: outright, because the two plates' side loads cancel in CY and the
    #: camber is a trap. carsim's plates are lifting tip panels in the same
    #: lattice as the wing, and camber POINTS their load with the wing's --
    #: `WingSpec.plate_airfoil` records +9.0 % of CL0 for a plate at
    #: alpha_L0 = -8 deg.
    #:
    #: MEASURED, as the correlation between a plate section's |camber| and
    #: the lap it buys, over the same 39 sections:
    #:
    #:     FLANK plate   arena +0.96   open +0.52   skidpad +0.97
    #:     TOP plate     arena -0.96   open +0.97   skidpad -0.97
    #:
    #: The flank panel wants camber on every circuit. The top wing's plate
    #: changes SIGN with the circuit -- and the whole library is worth
    #: 0.0003 to 0.0119 s there, so there is no default that is right rather
    #: than merely neutral. So the flank gets a camber-seeking set and the
    #: top keeps AeroBO's drag-led one, which is the honest answer for a
    #: panel at zero design lift: the drag it costs, the depth it needs and
    #: the angle it still works at.
    "plate (cambered)": {"clmax": 0.45, "cdcr": 0.15, "thick": 0.20,
                         "astall": 0.20, "ldcr": 0.0, "ldmax": 0.0, "cm": 0.0},
    "plate (symmetric)": {"ldcr": 0.0, "clmax": 0.25, "cm": 0.0, "ldmax": 0.0,
                          "cdcr": 0.35, "thick": 0.20, "astall": 0.20},
    "high lift": {"thick": 0.10, "clmax": 0.25, "ldmax": 0.10, "ldcr": 0.40,
                  "cm": 0.10, "astall": 0.05, "cdcr": 0.0},
    #  AeroBO's gdp-fwd and gdp-rear presets, under names that say what they
    #  do rather than which surface of an aeroplane they were written for.
    #  gdp-rear's raw weights sum to 0.95 in AeroBO too; `normalised` divides
    #  by the sum, so the raw total is not a number anything reads.
    "stall margin": {"thick": 0.10, "clmax": 0.10, "ldmax": 0.05, "ldcr": 0.40,
                     "cm": 0.10, "astall": 0.20, "cdcr": 0.0},
}

#: what each (role, target) form opens on. Editing makes the weights the
#: user's and nothing moves them again.
RECOMMENDED = {("flank", "main"): "wing (downforce)",
               ("top", "main"): "wing (downforce)",
               ("flank", "plate"): "plate (cambered)",
               ("top", "plate"): "plate (symmetric)"}


def recommended(role: str = "flank", target: str = "main") -> str:
    return RECOMMENDED.get((str(role), str(target)),
                           RECOMMENDED[("flank", "main")])


#: criteria that cannot rank a target, and the sentence saying why.
DEAD = {
    "plate": {
        "ldcr": "an end plate is a vertical panel at nominally zero "
                "incidence, so cl/cd at the design lift is the same 0 for "
                "every candidate -- this weight ranks nothing.",
        "ldmax": "'(L/D) max' is read at whatever lift maximises cl/cd, "
                 "which is a lift this panel never carries. Weight 'cd at "
                 "the design cl' instead: at zero lift that IS the drag.",
        #  AeroBO retires |cm| here too -- "a symmetric section's |cm| about
        #  the quarter chord is zero identically, so this weight ranks
        #  noise". That argument holds for ITS plate, whose library is
        #  restricted to symmetric sections. carsim's is not: its plates are
        #  lifting panels and may be cambered, so |cm| is a real number that
        #  separates them. MEASURED: over the 39 sections it is the single
        #  most informative criterion a plate has -- rho +-0.97 against the
        #  lap on either role -- so it is weighted, not retired.
    },
}

#: criteria a target does not ASK at all, because another criterion already
#: asks exactly the same question there. Different from DEAD: a dead
#: criterion ranks nothing (every candidate scores the same) and is shown,
#: dimmed, with its reason, as AeroBO does; a redundant one ranks, but in
#: the same order as its twin, so offering both is two sliders for one
#: question. It is not offered: no weight row, no ranking column.
#:
#: On the WING the design cl is one number for every section, and at a fixed
#: cl, L/D = cl / cd -- so "cd at the design cl" orders the library exactly
#: as "L/D at the design cl" does, inverted. Every wing preset already
#: weights it 0. On the PLATE the pair swaps roles: cl = 0 makes L/D at cl
#: the dead one, and cd at cl is the plate's only drag criterion, so there
#: it stays.
REDUNDANT = {
    "main": {
        "cdcr": "at the design cl every section flies the same lift, so "
                "cl/cd and cd put the library in exactly the same order -- "
                "'cd at the design cl' repeats 'L/D at the design cl'.",
    },
}
#: ...and the criterion that asks the same question, which takes over a
#: redundant criterion's weight when a preset written for the other surface
#: is chosen (a plate preset on the wing).
REDUNDANT_TWIN = {"cdcr": "ldcr"}

#: THE LIFT A SECTION IS SCREENED AT, when nothing else states one.
#:
#: AeroBO's `gui/v3/session.REFERENCE_CL`, and its reason is carsim's word
#: for word:
#:
#:   "A family that declares no design lift -- the car rear wing MAXIMISES
#:    downforce under a drag budget -- still needs the mission form to open
#:    on a number. It opens on the load CL = 1.0 implies at that family's own
#:    track point: a plain reference chosen by this shell, labelled as such,
#:    and not a published target."
#:
#: carsim's wings are that family. They are not sized to carry a stated
#: weight; they are asked for as much downforce as the lap will pay for. So
#: there is no design lift to derive and the form opens on a plain reference
#: -- EDITABLE, and labelled as a reference on the page.
#:
#: The alternative was tried and is worse: reading the lift off the REFERENCE
#: WING at the incidence it currently sits at. Measured on the shipped top
#: slot, that is CL 1.718 (mean local strip cl 1.616), which is above the
#: 2-D cl_max of 30 of the 39 sections carsim ships -- so the screen threw
#: three quarters of the library away as right-censored, and it did so
#: because of a mount angle the WING page owns and is about to move. A screen
#: whose shortlist changes when another page's row moves is not a screen.
REFERENCE_CL = 1.0

#: what a gate is SENT as when it is switched off. The idiom the screen
#: already understands, so "no gate" needs no second code path.
GATE_OFF = {"tc_min": 0.0, "cm_max": 1.0e9}

#: THE GATES ARE OFF BY DEFAULT HERE, AND THAT IS NOT AeroBO'S CHOICE.
#:
#: AeroBO screens at `tc_min = 0.15`, `cm_max = 0.08`, over the 2120-section
#: UIUC database, for an AIRCRAFT wing: thick enough for a spar, and a moment
#: a tail has to trim. carsim ships 34 sections chosen for a race wing, and
#: MEASURED over them at Re 5e5, cl 0.9:
#:
#:     t/c    min 0.0427   p25 0.0956   median 0.1070   max 0.1610
#:     |cm|   min 0.0000   median 0.0684   p75 0.1856   max 0.3477
#:
#: AeroBO's pair admits ONE of the 34. Worse, the |cm| ceiling refuses exactly
#: the sections a downforce wing exists for -- S1223 (0.348), S1210 (0.306),
#: CH10 (0.275), FX74-CL5-140 (0.275), E423 (0.247) -- because a high-lift
#: section's moment is what it trades for its lift, and a wing bolted to a car
#: has no tail to trim it: the mount carries it. So a gate inherited unchanged
#: would not be a filter, it would be a deletion of the library.
#:
#: Off by default, then, and the values below are what a gate comes back ON
#: at: the library's own quartiles, so switching one on refuses about a
#: quarter of the library, which is what a gate is for.
GATE_DEFAULTS = {"tc_min": 0.0956, "cm_max": 0.1856}
#: ...and what the form opens on.
GATES_OFF_DEFAULT = dict(GATE_OFF)
#: the SHAPE OPTIMISER's own gates are looser than the screen's -- a search may
#: go thinner than the library it was seeded from. AeroBO's ratio of the two
#: (0.10 against 0.15) applied to carsim's t/c quartile.
OPT_GATE_DEFAULTS = {"tc_min": 0.0637, "cm_max": 0.1856}

#: extra minimum floors, on top of the two gates. Higher-is-better: a section
#: is dropped when the metric falls below. All default to off.
FLOOR_META = {"clmax": ("min cl_max", 0.05), "ldcr": ("min L/D at the cl", 1.0),
              "astall": ("min stall angle", 0.5)}
FLOORS_OFF = {"clmax": 0.0, "ldcr": 0.0, "astall": -90.0}

#: what a shortfall below the seed costs J, per point, per unit weight.
#: AeroBO's `GOAL_PENALTY`: 3 makes a point lost three times as expensive as
#: a point won, which is what stops a weighted sum selling one criterion to
#: buy another.
GOAL_PENALTY = 3.0
#: the augmentation on the Tchebycheff function. AeroBO's `RHO_ASF`.
RHO_ASF = 0.05
REF_P_LO, REF_P_HI = 2.0, 98.0

#: the three objectives this module defines, and the sentence the form shows.
#: They are AeroBO's `OBJECTIVE_CHOICES` minus `pareto` (which returns a front
#: rather than a winner, and the garage has nowhere to draw one).
OBJECTIVES = {
    "composite": "the composite score of your seven criteria",
    "composite, none below the seed":
        "the same score, with the SEED's own criteria as a floor -- a "
        "weighted sum is free to sell one criterion to buy another, and this "
        "prices what falling below the seed costs",
    "lift the weakest criterion":
        "an augmented Tchebycheff function: score the WORST of your criteria "
        "relative to the seed, so nothing can be sold at any exchange rate. "
        "Its number is not a composite and does not compare with the other two",
}


# --------------------------------------------------------------------------- #
def normalised(weights: dict) -> dict:
    """The weights as fractions of one score. A set that sums to zero is
    returned flat rather than raising: the form can be walked to all-zero on
    the way to somewhere else, and a screen that crashed on the way would be
    a worse answer than a screen that ranked nothing in particular."""
    w = {k: max(0.0, float(weights.get(k, 0.0))) for k in CRITERIA}
    s = sum(w.values())
    if s <= 0.0:
        return {k: 1.0 / len(CRITERIA) for k in CRITERIA}
    return {k: v / s for k, v in w.items()}


def metrics(res: dict, cl_design: float) -> dict | None:
    """The seven raw criterion values of one evaluated candidate.

    `res` is an `aero.section.evaluate_section` result; `cl_design` is the
    lift coefficient this surface flies, which is the map every candidate is
    read on and so is the PROBLEM's, never the candidate's.

    Returns None when there is no polar to read. `censored` is set when the
    design lift is above this section's own cl_max: `np.interp` clamps, so cd
    and cm would be read AT the stall and reported as if they were at the
    design point. AeroBO refuses such a candidate by default and so does the
    caller here -- a number produced by a clamp is not a measurement.
    """
    pol = res.get("polar")
    if pol is None or getattr(pol, "n_rows", 0) < 3:
        return None
    g = res.get("geometry") or {}
    cl = float(cl_design)
    censored = cl > float(pol.cl_max)
    cd = float(pol.cd_of_cl(cl))
    a = float(pol.alpha_of_cl(cl))
    return {
        "thick": float(g.get("tc", res.get("tc", 0.0))),
        "clmax": float(pol.cl_max),
        "ldmax": float(pol.ld_max),
        "ldcr": (cl / cd if cd > 1e-9 else 0.0),
        "astall": float(pol.alpha_clmax_deg),
        "cm": abs(float(pol.cm_at(a))),
        "cdcr": cd,
        "censored": bool(censored),
    }


def gate_reason(m: dict | None, gates: dict, floors: dict | None = None) -> str:
    """Why this candidate is not eligible, or "" when it is."""
    if m is None:
        return "no polar"
    if m.get("censored"):
        return "the design cl is above its cl_max"
    tc_min = float((gates or {}).get("tc_min", GATE_OFF["tc_min"]))
    cm_max = float((gates or {}).get("cm_max", GATE_OFF["cm_max"]))
    if m["thick"] < tc_min:
        return f"t/c {m['thick']:.3f} < {tc_min:.3f}"
    if m["cm"] > cm_max:
        return f"|cm| {m['cm']:.3f} > {cm_max:.3f}"
    for k, (label, _step) in FLOOR_META.items():
        lo = float((floors or {}).get(k, FLOORS_OFF[k]))
        if m[k] < lo:
            return f"{label} {m[k]:.3f} < {lo:.3f}"
    return ""


def bands(rows: list, p_lo: float = REF_P_LO, p_hi: float = REF_P_HI) -> dict:
    """`{criterion: (lo, hi)}` over an ELIGIBLE population of metric dicts.

    A criterion its population could not separate -- every candidate the same
    value, which is what both L/D criteria are on a set of plates at zero
    lift -- gets no band at all and is dropped from `sub_scores`. That is
    AeroBO's `covers()`: a weight on a criterion the band cannot score buys
    nothing, and the form says so rather than the score quietly ignoring it.
    """
    out = {}
    for k in CRITERIA:
        v = np.asarray([float(r[k]) for r in rows if r is not None], dtype=float)
        if v.size == 0:
            continue
        lo, hi = (float(np.percentile(v, p_lo)), float(np.percentile(v, p_hi)))
        if hi - lo <= 1e-12:
            continue                     # nothing to separate: not a criterion here
        out[k] = (lo, hi)
    return out


def covered(band: dict, weights: dict) -> list:
    """The WEIGHTED criteria this band cannot score. The form warns on it."""
    w = normalised(weights)
    return [k for k in CRITERIA if w[k] > 0.0 and k not in (band or {})]


def sub_scores(m: dict, band: dict) -> dict:
    """0-100 per criterion on the frozen band. Not clipped -- see the module
    docstring: clipping flattens the objective exactly where a designed
    section beats the library it was seeded from."""
    out = {}
    for k, (lo, hi) in (band or {}).items():
        span = max(hi - lo, 1e-12)
        v = float(m[k])
        out[k] = 100.0 * ((hi - v) / span if k in LOWER_BETTER else (v - lo) / span)
    return out


def composite(sub: dict, weights: dict) -> float:
    w = normalised(weights)
    return float(sum(w[k] * sub[k] for k in sub))


def points(sub: dict, weights: dict) -> dict:
    """What each criterion CONTRIBUTED to the score -- the number the ranking
    table prints beside the raw metric, so the table says where a score came
    from and not only what it was."""
    w = normalised(weights)
    return {k: float(w[k] * sub[k]) for k in sub}


def goal_composite(sub: dict, seed: dict, weights: dict,
                   penalty: float = GOAL_PENALTY) -> tuple:
    """`(J, rows)` -- the composite with the SEED's own sub-scores as a floor.

    A criterion at or above its seed value costs nothing; below it costs
    `penalty * w_k * shortfall_k`, at the SAME weight the composite pays a
    gain at. A criterion nobody weighted is not defended: it is not in J
    either, and protecting what the user weighted at nothing would be a
    preference nobody stated.
    """
    w = normalised(weights)
    j = composite(sub, weights)
    rows, total = {}, 0.0
    for k in sub:
        if w[k] <= 0.0 or k not in (seed or {}):
            continue
        short = max(0.0, float(seed[k]) - float(sub[k]))
        cost = float(penalty) * w[k] * short
        rows[k] = dict(goal=float(seed[k]), got=float(sub[k]),
                       shortfall=short, cost=cost)
        total += cost
    return j - total, rows


def asf_composite(sub: dict, seed: dict, weights: dict,
                  rho: float = RHO_ASF) -> tuple:
    """`(J_asf, rows)` -- an augmented Tchebycheff function on the seed.

    `min_k w_k (s_k - r_k) + rho * sum_k w_k (s_k - r_k)`. A weighted sum can
    only ever return a design on the convex hull of what the box reaches, so
    whole families of compromise sections are invisible to it at every weight
    a user could type; this scores the WORST criterion relative to the seed,
    which no exchange rate can buy off. A zero-weight criterion is DROPPED
    from both aggregates rather than entering at lambda = 0 -- it would pin
    the min at 0 for every candidate and turn the function off.
    """
    w = normalised(weights)
    terms, rows = [], {}
    for k in sub:
        if w[k] <= 0.0 or k not in (seed or {}):
            continue
        t = w[k] * (float(sub[k]) - float(seed[k]))
        terms.append(t)
        rows[k] = dict(ref=float(seed[k]), got=float(sub[k]), term=t)
    if not terms:
        return -math.inf, rows
    return float(min(terms) + float(rho) * sum(terms)), rows


def is_composite(objective) -> bool:
    """True for any objective built out of the screening composite. One name
    for the set, because every branch asking 'does this run need the frozen
    band, and does it need the lattice?' is asking about all three."""
    return str(objective) in OBJECTIVES


def score_objective(objective: str, sub: dict, seed: dict, weights: dict) -> float:
    o = str(objective)
    if o == "composite":
        return composite(sub, weights)
    if o == "composite, none below the seed":
        return goal_composite(sub, seed, weights)[0]
    if o == "lift the weakest criterion":
        return asf_composite(sub, seed, weights)[0]
    raise ValueError(f"not a composite objective: {objective!r}")


# --------------------------------------------------------------------------- #
def self_check(verbose: bool = True) -> bool:
    ok = True

    def rep(tag, passed, msg=""):
        nonlocal ok
        ok = ok and passed
        if verbose:
            print(f"  [{'PASS' if passed else 'FAIL'}] {tag}" + (f"  {msg}" if msg else ""))

    #  AeroBO's own sets are carried unchanged, to the digit, and are what
    #  the divergences below are measured AGAINST
    rep("WingLab's gdp-sweep is carried verbatim",
        PRESETS["WingLab bulk sweep"] == {"thick": 0.10, "clmax": 0.20,
                                         "ldmax": 0.15, "ldcr": 0.35,
                                         "cm": 0.20, "astall": 0.0, "cdcr": 0.0})
    rep("...and its FIN_WEIGHTS, which its own car endplate shares",
        PRESETS["plate (symmetric)"] == {"ldcr": 0.0, "clmax": 0.25, "cm": 0.0,
                                         "ldmax": 0.0, "cdcr": 0.35,
                                         "thick": 0.20, "astall": 0.20})
    #  the DEFAULT is that set with |cm| removed: a car wing's moment is
    #  carried by its mount, and |cm| lower-better rewards reflex
    d = PRESETS["wing (downforce)"]
    a = PRESETS["WingLab bulk sweep"]
    rep("the default wing set does not rank a car wing on |cm|",
        d["cm"] == 0.0 and a["cm"] == 0.20)
    rep("...and the weight it freed went to the three lift criteria",
        all(d[k] >= a[k] for k in ("clmax", "ldmax", "ldcr"))
        and d["thick"] == a["thick"] and d["cdcr"] == a["cdcr"],
        f"clmax {a['clmax']} -> {d['clmax']}, ldmax {a['ldmax']} -> {d['ldmax']}, "
        f"ldcr {a['ldcr']} -> {d['ldcr']}")
    rep("...and `ldcr` is still the largest single weight, as WingLab has it",
        max(d, key=lambda k: d[k]) == "ldcr")
    #  a preset does not have to sum to one -- AeroBO's own gdp-rear sums to
    #  0.95 -- because `normalised` divides by the sum. What must hold is
    #  that every one of them NORMALISES to one, which is what the score
    #  actually reads.
    rep("every preset normalises to one",
        all(abs(sum(normalised(v).values()) - 1.0) < 1e-9
            for v in PRESETS.values()),
        ", ".join(f"{k} raw {sum(v.values()):.2f}" for k, v in PRESETS.items()))
    #  a plate's preset depends on the ROLE, because the two roles' plates
    #  want opposite signs of camber -- measured, see PRESETS
    rep("a flank plate and a top plate open on different sets",
        recommended("flank", "plate") != recommended("top", "plate"),
        f"flank '{recommended('flank', 'plate')}', "
        f"top '{recommended('top', 'plate')}'")
    rep("the flank plate's set seeks camber, the top's avoids it",
        PRESETS[recommended("flank", "plate")]["clmax"]
        > PRESETS[recommended("top", "plate")]["clmax"]
        and PRESETS[recommended("top", "plate")]["cdcr"]
        > PRESETS[recommended("flank", "plate")]["cdcr"])
    rep("both roles' sections open on the same set",
        recommended("flank", "main") == recommended("top", "main")
        == "wing (downforce)")
    rep("the screening reference lift is WingLab's REFERENCE_CL",
        REFERENCE_CL == 1.0)
    w = normalised(PRESETS["wing (downforce)"])
    rep("the weights normalise to one", abs(sum(w.values()) - 1.0) < 1e-12,
        f"sum {sum(w.values()):.15f}")
    rep("an all-zero set does not raise", abs(sum(normalised({}).values()) - 1.0) < 1e-12)

    #  ...and it is ranked on |cm|, unlike AeroBO's, because carsim's plate
    #  is a lifting panel and may be cambered
    rep("a plate cannot be ranked on the two L/D criteria",
        tuple(sorted(DEAD["plate"])) == ("ldcr", "ldmax"),
        ", ".join(sorted(DEAD["plate"])))
    rep("...but |cm| is NOT retired on it", "cm" not in DEAD["plate"])
    wing_presets = {RECOMMENDED[k] for k in RECOMMENDED if k[1] == "main"}
    rep("on the WING cd at the design cl is not asked (L/D = cl/cd at one cl): "
        "every wing preset weights it 0, and its twin is L/D at the design cl",
        set(REDUNDANT["main"]) == {"cdcr"} and REDUNDANT_TWIN["cdcr"] == "ldcr"
        and all(PRESETS[p]["cdcr"] == 0.0 for p in wing_presets)
        and not set(REDUNDANT["main"]) & set(DEAD.get("main", {})))
    rep("...and on the PLATE it stays: cd at cl = 0 is the plate's only drag criterion",
        "plate" not in REDUNDANT and "cdcr" not in DEAD["plate"])

    #  sub-scores: the band ends are 0 and 100, and a lower-better criterion
    #  is mirrored
    band = {"ldcr": (10.0, 110.0), "cm": (0.0, 0.20)}
    s_lo = sub_scores({"ldcr": 10.0, "cm": 0.20}, band)
    s_hi = sub_scores({"ldcr": 110.0, "cm": 0.0}, band)
    rep("the band's ends score 0 and 100",
        abs(s_lo["ldcr"]) < 1e-9 and abs(s_hi["ldcr"] - 100.0) < 1e-9
        and abs(s_lo["cm"]) < 1e-9 and abs(s_hi["cm"] - 100.0) < 1e-9)
    beyond = sub_scores({"ldcr": 135.0, "cm": 0.0}, band)
    rep("a candidate past the band is NOT clipped", beyond["ldcr"] > 100.0,
        f"{beyond['ldcr']:.1f} points")

    #  a criterion the population cannot separate gets no band, and a weight
    #  on it is reported rather than silently ignored
    flat = [{k: 1.0 for k in CRITERIA} for _ in range(5)]
    b = bands(flat)
    rep("a criterion nothing separates gets no band", b == {}, f"{sorted(b)}")
    rep("a weight on an unscorable criterion is named",
        covered(b, {"ldcr": 1.0}) == ["ldcr"])

    #  the goal composite: at the seed it IS the composite, and below it pays
    seed = {"ldcr": 60.0, "clmax": 40.0}
    wts = {"ldcr": 0.5, "clmax": 0.5}
    j_at, _ = goal_composite(dict(seed), seed, wts)
    rep("at the seed the goal composite is the composite",
        abs(j_at - composite(seed, wts)) < 1e-12, f"{j_at:.6f}")
    worse = {"ldcr": 70.0, "clmax": 30.0}          # +10 bought with -10
    j_plain = composite(worse, wts)
    j_goal, rows = goal_composite(worse, seed, wts)
    rep("a trade the plain composite calls even is refused by the goal",
        abs(j_plain - composite(seed, wts)) < 1e-12 and j_goal < j_at,
        f"plain {j_plain:.3f} = {composite(seed, wts):.3f}, goal {j_goal:.3f} "
        f"(paid {rows['clmax']['cost']:.3f})")

    #  the ASF: same trade, and it refuses it too -- but by the WEAKEST term,
    #  not by a price
    a_at, _ = asf_composite(dict(seed), seed, wts)
    a_worse, arows = asf_composite(worse, seed, wts)
    rep("the ASF is zero at the seed and negative below it",
        abs(a_at) < 1e-12 and a_worse < 0.0,
        f"at seed {a_at:+.3f}, traded {a_worse:+.3f} "
        f"(worst {min(arows, key=lambda k: arows[k]['term'])})")
    #  ...and a zero-weight criterion is dropped, not admitted at lambda 0
    a_drop, drows = asf_composite(worse, seed, {"ldcr": 1.0, "clmax": 0.0})
    rep("a zero-weight criterion is dropped from the ASF",
        list(drows) == ["ldcr"] and a_drop > 0.0, f"{a_drop:+.4f}")

    #  points sum to the composite
    p = points(seed, wts)
    rep("the per-criterion points sum to the score",
        abs(sum(p.values()) - composite(seed, wts)) < 1e-12)

    if verbose:
        print(f"  screen: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if self_check() else 1)
