# Task 2 — "Optimiser should follow the same order as the app"

## What the two orders actually were

Established by reading the code that builds each, and by running it:

| | order |
|---|---|
| garage DESIGNER page rows (`Designer._build_params`) | SECTION: *airfoil*, browse · PLANFORM: **span, chord, taper, twist, end plates** · MOUNT: **incidence**, station x, height h, … |
| optimiser design vector (`Designer.optimise`, before) | **span, chord, taper, twist_deg, plate_h, inc_deg** |

**They already agreed.** The six-coordinate vector was in the page's row order
already. So the literal reading of the request — "make the optimiser follow the
app's order" — was already satisfied, and there was no reorder to perform.

What was wrong is that *nothing held them to it*. Four lists were written out
by hand and had to agree coordinate for coordinate:

1. the `bounds` list, 2. the `x0` start vector, 3. the decode inside the
objective `f(x)` (`x[0]`→span … `x[5]`→incidence), 4. the write-back after the
run. A single transposed index in any of them would have produced plausible,
wrong designs rather than an exception — the worst failure mode available here.

## What changed

* `drive/aero/wing.py` gains `DESIGN_VARS`, **one ordered table** of what a
  designer may move, in the page's row order, with each variable's owner
  (`spec` = a `WingSpec` field, `slot` = the mount angle, which belongs to the
  `CarBuild` slot and not to the wing), its label, its unit and its per-role
  band. `BOUNDS` is now *derived* from it — verified bit-identical to the
  hand-written dict it replaced.
* Five functions read that table and nothing else: `design_vars`,
  `design_labels`, `design_bounds` (with keyword band overrides that **raise**
  on an unknown name, so a typo cannot silently hand the optimiser the full
  packaging band), `design_x0` and `apply_design`. `design_x0` and
  `apply_design` are exact inverses, checked in the self-check, and a further
  check pushes each coordinate to its bound in turn and asserts that exactly
  the one variable it names moved — the transposed-index failure, tested.
* `WingSpec.clamp` now walks `design_vars(role, owner='spec')` instead of its
  own hand-written tuple.
* `drive/aero/optimize.py` carries `labels` through `maximise` / `random_search`
  into the result, raises if the label list has drifted out of step with
  `bounds`, and gains `describe()`. Coordinate order is the caller's throughout.
* `drive/garage.py` `Designer.optimise` now builds all four of those lists from
  the shared table (`design_bounds` / `design_x0` / `apply_design` /
  `design_labels`). The read-out prints the winning design named coordinate by
  coordinate in the page's row order (`format_design`).

## The real bug the exercise exposed — page and optimiser disagreed on the span band

The flank panel is a *vertical* extent centred on its mount, so how tall it may
be depends on where it is bolted: it must clear the sill and stay under the roof
rail. The optimiser applied that fit (two bare literals, `0.28` and `1.34`,
inline in `optimise()`); **the page's span row did not.** The page offered spans
the optimiser was forbidden to propose, at every slot height:

```
  h 0.40 m   page hi 1.05 m   optimiser hi 0.40 m   DISAGREE
  h 0.60 m   page hi 1.05 m   optimiser hi 0.64 m   DISAGREE
  h 0.90 m   page hi 1.05 m   optimiser hi 0.88 m   DISAGREE
  h 1.00 m   page hi 1.05 m   optimiser hi 0.68 m   DISAGREE
  h 1.15 m   page hi 1.05 m   optimiser hi 0.40 m   DISAGREE
  h 1.20 m   page hi 1.05 m   optimiser hi 0.40 m   DISAGREE
```

Fixed by lifting the fit into `wing.span_fit(role, h)` — verified to reproduce
the optimiser's two literals exactly, to 1e-12, at h = 0.40 … 1.20 — and having
**both** the page's span row and the optimiser's span band call it.
`Param.hi` is read once at build time and cannot express a band that moves with
the slot height, so the row's *setter* clamps dynamically (`Designer._set(...,
cap=self._span_band)`). A top wing is limited by the car's width, not its ride
height, so `span_fit` returns its band unconditionally — unchanged.

## Proof that what the optimiser optimises did not change

The garage self-check's optimiser group, on the pristine baseline tree
(`git archive edffa06`) and on this tree:

```
baseline : optimiser runs (16 evals) and reports BO vs random: BO 3.270 random 2.052 start 0.944
now      : optimiser runs (16 evals) and reports BO vs random: BO 3.270 random 2.052 start 0.944
```

**Bit-identical** — which is the expected result precisely *because* the order
already matched: same bounds, same start vector, same decode, same seed, same
Sobol coordinate assignment. Had the order genuinely differed, the trajectory
would have moved (a reordered vector reassigns which Sobol coordinate drives
which variable) even though the search *space* would not; `optimize.self_check`
now demonstrates that separately on Branin with its two variables swapped.

## Assumptions

* The page's order is taken as the truth and the optimiser follows it, per the
  owner's instruction — not the other way round.
* The section (airfoil) is the page's first row but is a **discrete library
  choice**, so it is not a vector coordinate: the optimiser designs the planform
  for whatever section the page is showing. Same for `station x` and `height h`,
  which are slot placement rather than wing design and which the optimiser has
  never moved. I did not change what is optimised, per the brief.

## Not verified

* Only exercised through the headless garage self-check, never by hand on a real
  window. The dynamic span cap in particular is a UI feel change: at a high or
  low slot height the span row now stops earlier than it used to, which is
  correct but will be noticeable.
