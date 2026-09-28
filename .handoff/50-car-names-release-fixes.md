# 50 -- fictional car names + the remaining release fixes (2026-09-28 evening)

Owner: "rename the cars with the fictional names", then "Then do all of the
changes needed." Follows 49-steam-prep.md. All uncommitted on main.

## The cars

| key (unchanged) | was | shown now | short tag |
|---|---|---|---|
| corsa | Opel Corsa C 1.2 | Aurel Civetta 1.2 | Civetta |
| rally | Ford Escort rally | Halcón RS18 rally | Halcón |
| 540i | BMW 540i | Nordwerk N540 | N540 |
| express | Renault Express 1.4 | Rivière Courier 1.4 | Courier |

- Keys stay everywhere (settings, saves, records, leaderboards, medals, CLI
  `--car`, bot file names). Only text the player reads changed.
- `cars.DISPLAY_FIELDS = ("name",)`: a display name is not a medal input.
  medals.py's hash leaves it out; `medals.json` and `reference_laps.json`
  were re-stamped (324f20135bb7...) after proving the new payload with the
  old names put back reproduces the old stamp d549f20c... exactly. Medals
  are current; no rebuild.
- `cars.py` self-check: "the car names on screen are fictional".
- Knock-ons fixed: `bodies.style_of`, `audio.profile_key` and `fx` match the
  new names (the Halcón and the Courier drew as the hatch until then);
  default build names `my civetta` etc. (`garage.default_build_name`; an old
  save's `my corsa` is still treated as a default name when carried to
  another car); bot file stems shown via `drive.shown_stem`
  (`540i_arena_plate` -> `n540_arena_plate`, retired MX-5 -> `roadster_...`);
  `records.class_label`, swarm default names (`swarm.car_word`), seed-lap
  file names use the shown word.
- Verified: a rendered-text sweep (monkeypatched Font.render / set_caption /
  Menu.show over a scripted session in all 4 cars, 68 pages, + the
  page-drawing self-checks; 91,362 strings) -- zero real makes / models /
  codes on screen. `validate --modules` 134/134, `drive --self-check` ALL
  PASS, cars.py, medals 8/8 current, launch smoke clean. Store art re-run.
- Not changed: body shapes (`bodies.py:61-96` follow the real cars'
  proportions; no badges / grilles / logos drawn) -- owner's call;
  comments / docstrings / provenance strings; README; the owner's own saved
  builds and local leaderboard entries keep their old names.

## Release fixes (second round)

- `launch_game.py`: unknown launch arguments in a packaged build are dropped
  and logged (a bad value for a known flag -> start with none); setup
  failures (unwritable / full saves folder) write
  `<tmp>/<save id>-crash-*.txt` and show a native alert (SDL via ctypes;
  skipped headless or with CARSIM_NO_MSGBOX=1); crashes in game show it too.
- `packaging/game.spec`: `licenses/` in the bundle (74 files: every bundled
  package's licence, Python, PyInstaller, AeroBO, the font OFL,
  THIRD_PARTY_NOTICES, `packaging/licenses/LGPL-2.1.txt`,
  `BUNDLED_LIBRARIES.md`); only the tyre files the 4 cars read are shipped.
- `packaging/licenses/SOURCE_OFFER.md`: LGPL written offer -- OWNER fills in
  a contact address.
- CI: `macos-13` is retired -> `macos-15-intel` (Intel) and `macos-15`.
- On-screen "carsim" text reworded ("the game's", "car packaging") in
  views_wing / views_results / views_mission / aerobo_models / design_shell
  / cae.chrome; WingLab shows family names without the "carsim " prefix
  (`aerobo_bridge.shown_family`, display only -- the registered names stay,
  saved runs and fixtures match on them). Left: the WingLab code panel's
  literal `problem_name='carsim flank ...'` (the real identifier).
- Controls page: `MENU_HELP_KB` row "F11 / ALT+ENTER  fullscreen on / off"
  (packaged builds; drive/display_mode.py).
- Store art named by the shown car word (`screenshot_02_kestrel_n540.jpg`
  ...); old key-named files removed. `library_logo.png` is now 1280x360
  because the other session's new title-logo art is shorter.

Final gates after everything: `drive.garage` 195/195, `drive --self-check`
ALL PASS, rebuilt frozen app `--probe` exit 0. Frozen macOS build (152 MB) passes: `--probe`, title, `--race best`, an
unknown `--steam-thing 1` (dropped, title reached), `--car nope` (starts
with none), read-only saves folder (crash report in tmp, exit 1).

## Still the owner's

Valve account / fee / store page; Windows + Deck + Xbox-pad hardware tests
(`packaging/PAD_TEST.md`; the crash alert has never been seen on a real
screen); tyre-file + UIUC licences; the source-offer address; Apple
signing + notarisation + a real bundle id; arm64 vs x64 macOS depot; price;
`SAVE_ID_PINNED` once the name is final; optionally softer body shapes and a
Display row in Settings.

## Free + GitHub (2026-09-28 night)

Owner: "I have steamworks account. I will make it free." and "If I can't
upload now I will upload first on github so employees can see that I have
done it but still need time to publish on steam".

- Free: checked on Valve's docs -- the USD 100 app credit still applies (never
  recouped when free), the 30-day wait counts from that app's fee, create the
  app with "This is a free product", no trading cards, free<->paid only via
  Valve support. Builds can be uploaded as soon as the App ID exists.
  RELEASE_CHECKLIST.md section A + D4 and steam/store/text.md (price, tags)
  rewritten.
- GitHub showcase: README.md opens with an employer-facing section (header
  art, 3 screenshots, features, engineering highlights with code links, how to
  play); the old H1 became "Developer documentation". LICENSE (all rights
  reserved; aerobo/ MIT). .gitignore + .claude/, .venv*/, steam/content/ (the
  collected ~150 MB builds), the two sketch PDFs. Release job in
  steam-build.yml: a v* tag builds all OSes and makes a DRAFT GitHub Release
  (publish with `gh release edit <tag> --draft=false`). Hygiene audit: no
  secrets / IPs / other projects anywhere in files or history; commits carry
  bc1923@ic.ac.uk (owner's call) and Claude co-author trailers.
- Not done: commit / repo create / push -- waiting for the owner's OK.

## Published on GitHub (2026-09-28, owner approved: public + builds, keep emails)

- https://github.com/TomyCabrer/aleron-race-your-wing (public; only main
  pushed; branch t44 + stash stay local). Commits 8316661 (tasks 46-50) and
  4e08201 (README with the owner's saved builds + garage images, CI actions
  bumped to checkout v7 / setup-python v7 / upload-artifact v7 /
  download-artifact v8).
- Tag v0.9.0: CI built and probed Windows, macOS arm64, macOS x64 and Linux
  (all green -- first real Windows run) and made a DRAFT release with
  Aleron-v0.9.0-{windows.zip 87 MB, macos-arm64.zip 62 MB, macos-x64.zip
  67 MB, linux.tar.gz 109 MB}. Publish: `gh release edit v0.9.0 --draft=false`.
  (The first tag push raced the workflow's registration; the tag was
  re-pushed on the same commit.)
- README images: `python3 docs/make_readme_images.py` (race frames of saved
  builds) and `python3 steam/store/make_garage_shots.py` (garage, SAVED CARS,
  WingLab) -- both render from a scratch copy of runs/, renaming "my corsa" /
  "my express" builds to "my civetta" / "my courier" in the copy only.
- Garage pages at 1920x1200 overlap (garage_ui.key_hint_bar fixed pixel
  offsets) -- shots kept at 1280x800; a UI-scale bug worth fixing.
