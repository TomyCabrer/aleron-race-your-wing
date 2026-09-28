# 49 -- Steam prep (2026-09-28, owner away 1 h)

Owner: "in 1h prepare the game to be published in steam. I have to leave. I
still have to change the name." Then: "If game in steam is going to be too
complicated I will want to have a browser version but with 1 car no bots and
no garage."

**Verdict on Steam vs browser:** Steam is NOT too complicated. The desktop
build works today on macOS (PyInstaller, 151 MB .app); the rest is Valve
paperwork and waiting periods. A browser build (pygbag) of one car / no bots
/ no garage is realistic but ~3-5 days: pygbag has numpy but no scipy (one
`PchipInterpolator` in drive/powertrain.py:49 pulls it in), all 12 frame
loops must become async, saves need IndexedDB, the 3-D chase view would run
~45 fps. Not started.

All uncommitted on main, over the other session's uncommitted tasks 46-48.

## Rename the game

Edit `drive/branding.py` only (GAME_NAME, GAME_SHORT, LOGO_WORD, SUBTITLE).
Then `python3 steam/store/make_store_assets.py` (all Steam art) and rebuild.
The logo draws any word (wing-A only if it starts with A, wing-N only if it
ends with N; accents become the white top wing). Before the first public
build set `SAVE_ID_PINNED` to what `save_id()` returns then.

## What exists now

| path | what |
|---|---|
| `drive/branding.py` | the name, version, save id -- one place |
| `drive/userdata.py` | per-user saves folder (macOS Application Support, %APPDATA%, XDG); packaged build chdirs there so every `runs/...` lands there |
| `launch_game.py` | the packaged entry: freeze_support, saves folder, logs (`logs/game.log`, `crash-*.txt`), copies AeroBO + shipped bots into the saves folder once, display mode, `--probe` smoke test. From source == `python3 -m drive.drive` |
| `drive/display_mode.py` | packaged build only: SCALED+RESIZABLE window, fullscreen by default, F11 / Alt+Enter toggle (remembered in `runs/display.json`), Windows DPI awareness |
| `packaging/` | `game.spec`, `build_mac.sh`, `build_windows.ps1`, `build_linux.sh`, icons, `README.md`, `PAD_TEST.md` |
| `.github/workflows/steam-build.yml` | Windows / macOS arm64 / macOS x64 / Linux builds + probe; artifacts are tar.gz (keeps exec bits, symlinks) |
| `steam/` | steamcmd upload kit (`upload.sh`, VDF templates, `collect.sh`), Steam Cloud rows, launch options, `README.md` |
| `steam/store/` | `text.md` (store copy, sysreqs, content survey), `make_store_assets.py` + `art/` (18 graphics + 6 screenshots, from the game's renderer) |
| `RELEASE_CHECKLIST.md` | owner steps in order with Valve's timelines (USD 100 fee, 30 days after paying, 2 weeks Coming Soon, 3-5 business-day reviews) |
| `THIRD_PARTY_NOTICES.md` | licences; owner decisions flagged |

Game-code edits (all exact-match patches): `drive/title.py` (name from
branding, logo takes any word), `drive/render.py` + `drive/garage.py`
(captions), `drive/tyre.py` (.tir paths resolve against the repo, not the
cwd -- the packaged build crashed without it), `drive/views_mission.py`
("carsim's" -> "the game's"), `drive/aerobo_bridge.py` (CARSIM_AEROBO_ROOT,
frozen runs/aerobo), `drive/drive.py` (3 x "(see the terminal)" -> "(see the
log)"), `drive/input.py` (Xbox / Steam Input pads through SDL's
GameController API, same button places as the DualSense, GUIDE no longer
bound; garage works with a non-Sony pad). `.gitignore` (+ /build/ /dist/
steam/output/).

## Gates run

- Frozen app (final code, fresh saves folder): `--probe` exit 0 in 3.4 s;
  no-arg launch reaches the live title; `--race best` loads a shipped bot;
  `--garage` and a 4-worker swarm ran clean in an earlier build.
- Real window from source: F11 fullscreen on/off, survives title -> drive.
- Self-checks: title 14/14, render 79/79, garage 195/195, vehicle 38/38,
  tyre, cars, input, menu, controls_page all pass. Source launch smoke clean.

## Owner must do (the order is in RELEASE_CHECKLIST.md)

1. Steamworks account, USD 100 fee, tax/bank (the 30-day clock starts at payment).
2. Decide: real car names (Opel/BMW/Ford/Renault: `cars.py:1136`,
   `drive/prerace.py:127` + ~350 strings), tyre file licence
   (`tyre_data/TNO_car205_60R15.tir`, TNO sample, no licence), UIUC airfoil
   data terms, LGPL source-offer hosting, macOS depot arm64 vs x64, bundle id
   (`com.example.aleron` in packaging/game.spec), price, AI disclosure answer.
3. Build + test on Windows (never run there) and, for Deck support, a Steam
   Deck; test an Xbox pad (`packaging/PAD_TEST.md`) -- pad code is tested
   offline only.
4. Apple Developer ID signing + notarisation for the macOS depot.

Left small: "carsim" text in drive/views_wing.py + views_results.py (other
session's files), Controls page doesn't mention F11 / Alt+Enter, no Display
row in Settings, `--self-check` paths that re-launch `sys.executable` don't
work inside a frozen build (dev-only).

## Review (one pass, no crash bug confirmed)

Fixed after it: an update now refreshes a shipped bot the player never
touched (`launch_game._install_bots`, hashes in `.installed.json`).
Risks for the hardware test: an unknown launch argument (e.g. a Steam launch
option) makes drive.main's strict argparse exit 2 silently (message only in
`logs/game.log`); an unwritable saves folder on the very first run exits with
no crash file; fullscreen tested title -> drive only (not garage switches);
the game always opens pad 0 and never closes a disconnected one (Deck with an
external pad); macOS takes F11 (Option+Enter works).
