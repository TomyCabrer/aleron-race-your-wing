# Packaging the game for Steam

The game ships as a PyInstaller **one-folder** build per platform. Everything
here reads the name and version from `drive/branding.py`, so a rename is one
file (then delete `packaging/icons/` and rebuild so the icon follows).

| file | what |
|---|---|
| `../launch_game.py` | the entry point (source and packaged); `--probe` = smoke test |
| `game.spec` | the PyInstaller recipe (what ships, what doesn't) |
| `build_mac.sh` / `build_windows.ps1` / `build_linux.sh` | one-command builds into `dist/` |
| `requirements-build.txt` | the build venv (no torch) |
| `make_icons.py`, `icons/` | the app icon from the title logo (.png/.ico/.icns) |
| `licenses/` | shipped with every build: the LGPL-2.1 text, the LGPL source offer (`SOURCE_OFFER.md`, owner fills in the contact), pygame's native libraries and their licences (`BUNDLED_LIBRARIES.md`) |
| `PAD_TEST.md` | the controller test to run on Windows and a Steam Deck |
| `../.github/workflows/steam-build.yml` | CI: Windows + macOS arm64 (`macos-15`) + macOS x64 (`macos-15-intel`; `macos-13` was retired Dec 2025) + Linux builds, probed, uploaded as tar.gz artifacts; a `v*` tag also makes a draft GitHub Release (below) |

`dist/` and `build/` are build output: keep them out of git.

## Build (macOS, tested 2026-09-28)

```
bash packaging/build_mac.sh
CARSIM_DATA_DIR=/tmp/fresh SDL_VIDEODRIVER=dummy dist/Aleron.app/Contents/MacOS/Aleron --probe
open dist/Aleron.app
```

Measured on the M2 Pro (Python 3.13, PyInstaller 6.22, no torch): the build
takes ~45 s; `Aleron.app` is **152 MB**, arm64 only; `--probe` on an empty data
folder 3.5 s (first run copies AeroBO, 25 MB), 2.2 s after. The frozen app
drove a session offscreen, opened the garage (AeroBO loaded from the
per-user copy), and ran a 4-car swarm generation on 4 worker processes.
`--probe` also says whether a crash can show its alert ("crash alert via
SDL_ShowSimpleMessageBox"); CI runs it on every platform.

Windows and Linux can't be built from a Mac (PyInstaller does not
cross-compile). Either run `build_windows.ps1` on a Windows PC and
`build_linux.sh` in Valve's Steam Runtime "sniper" SDK container, or push the
repo to GitHub and run the `steam-build` workflow (Actions tab → Run workflow),
which builds all four and uploads each depot folder as an artifact.

## GitHub Release (downloads before Steam)

Pushing a tag `v*` (e.g. `git tag v0.9.0 && git push origin v0.9.0`; keep it
equal to `branding.VERSION`, the job warns otherwise) runs the same four
builds, then the `release` job creates a **draft** GitHub Release for the tag
with one download per system, named from `branding.ascii_name()`:
`Aleron-v0.9.0-windows.zip`, `-macos-arm64.zip`, `-macos-x64.zip` and
`-linux.tar.gz`. The macOS zips are made on the Mac runners with
`ditto -c -k --keepParent` (a Linux-made zip would break the .app's symlinks
and signature); the Linux build stays a tar.gz for its exec bit. The release
notes say the builds are unsigned (macOS: right-click > Open, or Privacy &
Security > Open Anyway on macOS 15+; Windows: SmartScreen > More info > Run
anyway). A tag with a `-` (`v1.0.0-beta1`) becomes a pre-release; re-running
the job replaces the files and keeps edited notes. It needs all four builds
green. A manual run makes no release: its macOS zips (`player-macos-*`,
named `dev-<sha>`) and the depots stay on the run's page.

The draft is visible only to the repo's owner: download and try it, then
`gh release edit v0.9.0 --draft=false` (or **Publish release** on its page)
makes the downloads public. Once published, anyone can play the game free
before it is on Steam; leave it a draft until you want that. On a public
repo, a run's artifacts can also be downloaded by any signed-in GitHub user
for the retention period (90 days by default).

## What a packaged build does differently (launch_game.py)

* **Saves** go to the per-user folder, not the install folder:
  macOS `~/Library/Application Support/<save id>/`, Windows `%APPDATA%\<save id>\`,
  Linux `~/.local/share/<save id>/` (`drive/userdata.py`; `<save id>` =
  `branding.save_id()`, pin it after the first public build). The working
  directory is that folder, so every `runs/...` path lands in `<save id>/runs/`.
  Steam Cloud should sync `runs/` there.
* **AeroBO** (vendored, finds its data and writes its caches beside its own
  files) is copied to `<save id>/aerobo/` once per version, and
  `drive/aerobo_bridge.py` imports it from there (`CARSIM_AEROBO_ROOT`).
* **Logs**: stdout/stderr go to `<save id>/logs/game.log` (the previous run's
  as `game.prev.log`); an uncaught error writes `logs/crash-<time>.txt` and a
  native alert tells the player where it is. If the saves folder itself can't
  be made (read-only, disk full) the report goes to the temp folder
  (`<save id>-crash-<time>.txt`, e.g. `%TEMP%` on Windows) with the same
  alert. Ask players for those files when they report a bug. The alert is
  skipped headless (SDL dummy/offscreen drivers, CI) and with
  `CARSIM_NO_MSGBOX=1`.
* **Launch arguments**: an argument the game doesn't know (a Steam launch
  option, a typo in a shortcut) is dropped and logged ("launch: ignoring
  unknown launch arguments ..."); an invalid value for a known one
  (`--car nope`) starts the game with no arguments, also logged. A source run
  keeps argparse strict.
* **Worker processes** (the swarm, the bot test) work: `freeze_support()` runs
  first.
* **Shipped bots**: `drive/ml/checkpoints/*.json` are copied into
  `<save id>/drive/ml/checkpoints/` (the game looks there, relative to the
  working directory, and saves bred bots there). An update refreshes a shipped
  bot only while the installed copy is untouched (hashes in
  `.installed.json`); a player's file is never overwritten.
* **Licences**: `licenses/` holds THIRD_PARTY_NOTICES.md, `packaging/licenses/*`,
  Python's, PyInstaller's, AeroBO's, the bundled fonts' and the licence files
  of every package the build collected (read from their wheels), plus a
  generated `README.txt` index with versions. macOS: `Aleron.app/Contents/
  Resources/licenses/`; Windows / Linux: `licenses/` next to the exe. pygame's
  wheel carries no licence files: `BUNDLED_LIBRARIES.md` and the LGPL text
  cover it.
* **Tyres**: only the `.tir` files the player's cars read (`cars.CARS`, today
  just `TNO_car205_60R15.tir`) ship, not the whole `tyre_data/` study folder.
* **Display** (`drive/display_mode.py`): fullscreen on first launch, F11 or
  Alt+Enter toggles (remembered in `runs/display.json`), `--fullscreen` /
  `--windowed` on the Steam launch line set it. The 1280x800 game is scaled
  and letterboxed (`SCALED | RESIZABLE`), DPI-aware on Windows. A source run
  gets it with `CARSIM_DISPLAY_MODE=1 python3 launch_game.py`;
  `python3 -m drive.drive` stays a plain 1280x800 window.
* **Not shipped**: `runs/` (your dev saves), `.handoff/`, `aerobo/results/`,
  torch/botorch/gpytorch (AeroBO's Bayesian optimisers stay unavailable; the
  DESIGN page offers the ones that need no torch), XFOIL (GPL, and optional).

## Before a public build

* `game.spec`: set `bundle_identifier` (now `com.example.<name>`).
* `licenses/SOURCE_OFFER.md`: put your contact address in; check the versions
  against the build's `licenses/README.txt`.
* `drive/branding.py`: final name; then pin `SAVE_ID_PINNED`.
* macOS: sign with a Developer ID and notarize (`codesign --deep --options
  runtime`, `xcrun notarytool submit`), or players may see Gatekeeper warnings
  outside Steam. The build is only ad-hoc signed now. Build once on Apple silicon
  and once on Intel (the CI does both); ship them as two depots or merge them with
  `lipo` into a universal app.
* Windows: code-signing is optional on Steam, but SmartScreen warns about
  unsigned exes run outside Steam.
