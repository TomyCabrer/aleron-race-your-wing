# AeroBO, vendored

This directory is AeroBO — the owner's wing-design engine (`~/dev/AeroBO`, MIT,
`LICENSE` here is its own) — copied into carsim unmodified, so the garage's
DESIGN page runs the same engine the AeroBO app runs: the same families, the
same screens, the same section optimiser, the same budgets, the same lattice.

<!-- provenance:begin (rewritten by aerobo/sync.sh; checked by row B2 of drive/aerobo_bridge.py) -->
```
source:         ~/dev/AeroBO
commit:         3f1b07d
describe:       v1.0.0-3-g3f1b07d
vendored:       2026-09-24
files:          src 68, data 2260, records 1, LICENSE 1
content sha256: 042c751b9ca7ea7b50403bffba4369952eb8c91a1d4fcfb40baee146c3fba3b7
seed sha256:    71403a4f4f1a147478ae221d6ab9d232cbcf22461c2e1cd096c857476d06913f  seed/airfoil_screen_checkpoint.json
```
<!-- provenance:end -->

## What was copied

Exactly what the release ships: the tracked files of the commit above, via
`git archive <rev> src data records LICENSE`.

| path | what |
|---|---|
| `src/aerobo/**` | the engine (every `.py` of the package; no `__pycache__`) |
| `data/**` | airfoil tables and the UIUC library (2174 `.dat` + polars), `search_budget.json` (the measured budget study), `screen_reference*.json`, `box_band*.json` |
| `records/airfoil_screen_branch_v2.json` | the library's branch sidecar (lets a cached polar be re-scored exactly at any design cl) |
| `LICENSE` | AeroBO's MIT licence |
| `seed/airfoil_screen_checkpoint.json` | NOT in AeroBO's git: its warm library-screen checkpoint, copied from `~/dev/AeroBO/results/`. Without it the first screen runs XFOIL over the whole library (AeroBO's own note: over an hour per Reynolds number). The bridge installs it into `results/` on first import. |

Not copied: `gui/`, `tests/`, `scripts/`, `docs/`, `installer/`, `launch.py`,
`.venv-app/`, `results/` (except the seed). `vsp.py` (OpenVSP) and two lazy
`matplotlib` imports are in the tree but never reached by the calls carsim
makes.

## How carsim uses it

* `drive/aerobo_bridge.py` is the ONLY importer. It puts `aerobo/src` first on
  `sys.path`, points `AEROBO_XFOIL_BIN` at the XFOIL carsim already uses, and
  asserts the imported package is this copy (an installed dev copy of AeroBO is
  never picked up silently). `aerobo/` itself has no `__init__.py`, so it never
  shadows the package in `src/`.
* The engine finds `data/`, `records/` and `results/` at
  `Path(api.__file__).parents[2]`, which is this directory.
* Runtime writes go to `results/` only (XFOIL cache, evaluation cache, screen
  checkpoints, the coordinate sidecar) — gitignored. Run records carsim keeps
  go to `runs/aerobo/<slot>/` (also ignored).
* Everything carsim adapts — its slot families (ride band, ground effect on/off,
  carsim's air), its operating point, its circuits for the lap objective, the
  law the car flies — is carsim-side, in the bridge. **Engine patches: none.**

## Re-sync

```
aerobo/sync.sh ~/dev/AeroBO <rev>
```

Archives `src/ data/ records/ LICENSE` at `<rev>` into a scratch directory and
syncs them in, so only the files that changed are touched (deleting the tree
and re-creating it in place on a Desktop that iCloud syncs came back as a
`name 2.py` duplicate of every file); stops, naming them, if any file on disk
is not in the release; refreshes the seed if the repo has a newer one, rewrites
the provenance block above, and runs `python3 -m drive.aerobo_bridge`. The bridge's identity rows fail loudly if
the new engine moved a private name the bridge relies on (`api._BuiltProblem`,
`api._apply_overrides`, the problem dataclass fields, `RIDE_HEIGHT_BOUNDS_M`,
`ALPHA_BOUNDS_DEG`). The captured UI fixtures (`drive/data/aerobo_fixtures/`)
name the commit they were recorded on; re-record them after a sync with
`python3 -m drive.aerobo_bridge --capture drive/data/aerobo_fixtures`.
