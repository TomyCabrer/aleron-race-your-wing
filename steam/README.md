# steam/ -- getting a build onto Steam

Everything here is for the Steam release. Nothing in it runs inside the game.
The name used in file names comes from `drive/branding.py` (`ascii_name()`,
today `Aleron`), so a rename changes the paths below by itself.

| file | what |
|---|---|
| `upload.sh` | uploads the depots with steamcmd (SteamPipe) |
| `collect.sh` | copies this machine's packaged build into `steam/content/<os>/` |
| `templates/*.vdf.in` | the app build + one depot build per OS; `upload.sh` fills them into `steam/out/` |
| `dev/steam_appid.txt` | `480` (Valve's Spacewar test app). DEV ONLY, see below |
| `store/text.md` | store page texts, tags, system requirements, content survey |
| `store/make_store_assets.py` | the capsules, header, library art and screenshots (written by the store-art lane) |

The owner steps (account, fee, reviews, dates) are in `../RELEASE_CHECKLIST.md`;
the licences a build ships are in `../THIRD_PARTY_NOTICES.md`.

---

## 1. Build, per OS

PyInstaller does not cross-compile: a Windows build is made on Windows, a
macOS build on a Mac, a Linux build on Linux (the Steam Deck runs the Windows
build through Proton, or the Linux one).

```
packaging/build_mac.sh          # -> dist/Aleron.app
packaging/build_windows.ps1     # -> dist/Aleron/Aleron.exe + _internal/
packaging/build_linux.sh        # -> dist/Aleron/Aleron + _internal/
steam/collect.sh                # -> steam/content/<os>/   (reads dist/, or packaging/dist/)
```

(The build scripts, `packaging/game.spec` and the CI build
`.github/workflows/steam-build.yml` belong to the packaging lane; see
`packaging/`.) On Windows without Git Bash, `collect.sh` is just: copy the
CONTENTS of `dist\Aleron\` into `steam\content\windows\`.

**From GitHub Actions instead** (no Windows PC needed): the workflow uploads
one artifact per depot, `depot-windows`, `depot-macos-arm64`,
`depot-macos-x64`, `depot-linux`, each holding ONE file,
`depot-<os>.tar.gz`, with exactly the depot's root inside. It is a tar
because GitHub artifacts drop Unix permissions and symlinks, and a
PyInstaller .app and the Linux binary need both. Unpack them into the folders
`upload.sh` reads:

```
gh run download <run id> -n depot-windows     -D /tmp/d && mkdir -p steam/content/windows && tar -xzf /tmp/d/depot-windows.tar.gz -C steam/content/windows
gh run download <run id> -n depot-macos-arm64 -D /tmp/d && mkdir -p steam/content/macos   && tar -xzf /tmp/d/depot-macos-arm64.tar.gz -C steam/content/macos   # or macos-x64, see below
gh run download <run id> -n depot-linux       -D /tmp/d && mkdir -p steam/content/linux   && tar -xzf /tmp/d/depot-linux.tar.gz -C steam/content/linux
```

Check with `open steam/content/macos/Aleron.app` before uploading.

**GitHub Release.** A `v*` tag push also puts the builds in a **draft** GitHub
Release, public only once you publish it (player zips per system, unsigned-build notes; see
`packaging/README.md`, "GitHub Release"). That is separate from Steam: the
depots above are still the `depot-*` artifacts of the same run, and a
manual run makes no release. The macOS run also uploads `player-macos-*`
artifacts (the .app zipped with `ditto`); those are for players, not
`upload.sh`.

**One macOS depot, two CPUs.** Steam has no arm64/x64 filter for macOS
depots, so the macOS depot holds ONE .app. Choose:
- `depot-macos-x64` (Intel build): runs on every Mac, on Apple Silicon
  through Rosetta 2 -- the widest reach, a little slower on M-series Macs;
- `depot-macos-arm64`: native and fastest on M-series, **no Intel Macs**
  (say "Apple Silicon required" in the system requirements);
- a universal2 .app (both in one): needs a universal2 Python and universal2
  wheels for every package -- not set up.
Default recommendation: x64 for the first release if the Rosetta frame rate
is fine on an M-series Mac, else arm64 + "Apple Silicon required".

## 2. What goes in each depot

A depot's content folder is exactly what the player's install folder holds:

| depot | `steam/content/<os>/` holds | launch option (Steamworks > Installation > General) |
|---|---|---|
| Windows | `Aleron.exe`, `_internal/` | Executable `Aleron.exe`, OS Windows, 64-bit |
| macOS | `Aleron.app` | Executable `Aleron.app`, OS macOS |
| Linux | `Aleron`, `_internal/` | Executable `Aleron`, OS Linux + SteamOS |

Valve: on macOS launch the **.app**, not the binary inside it -- a bare binary
launched by Steam on Apple Silicon runs as x86_64
([uploading](https://partner.steamgames.com/doc/sdk/uploading)).

Create the three depots in Steamworks (App Admin > SteamPipe > Depots), set each
one's OS filter, and note their ids. Valve gives the first depot `APPID + 1`.

The game writes nothing into its install folder: saves, logs and AeroBO's
caches go to the per-user folder (`drive/userdata.py`, `launch_game.py`).
That is why the depots need no `userconfig` file properties.

## 3. Upload

Unzip the Steamworks SDK (partner site > Getting Started > SDK) to
`steam/sdk/` (gitignored by you: add `steam/sdk/`, `steam/content/` and
`steam/out/` to `.gitignore`), or install `steamcmd` on the PATH. Use a
dedicated **builder account** with only the "Edit App Metadata" and "Publish App
Changes To Steam" permissions, as Valve recommends. Then:

```
export STEAM_USER=<builder account> STEAM_APPID=<app id>
export STEAM_DEPOT_WINDOWS=<id> STEAM_DEPOT_MACOS=<id> STEAM_DEPOT_LINUX=<id>
steam/upload.sh --preview        # dry run: manifests + logs in steam/out/logs, nothing uploaded
steam/upload.sh --beta testing   # real upload, live on the 'testing' beta branch
steam/upload.sh                  # real upload; set it live on 'default' by hand
```

The password and the Steam Guard code are typed into steamcmd, never stored
here. A depot whose id is unset or whose folder is missing is skipped, so you
can upload the Mac depot alone. The build appears in Steamworks > SteamPipe >
Builds; the **default** branch can only be set live there, by hand
([uploading](https://partner.steamgames.com/doc/sdk/uploading)).

On Windows, run `upload.sh` from Git Bash, or copy `steam/content/windows/` to
the Mac and upload all three from there.

## 4. steam_appid.txt (dev only)

The game does not use the Steamworks API today (the leaderboards run on their
local backend, `drive/leaderboard.py`), so it needs no `steam_appid.txt` at all.
If a Steamworks binding is added later: during development put
`steam/dev/steam_appid.txt` (`480`, or the real app id) next to the executable
so the game can start outside the Steam client. **Never ship it**: the depot
templates exclude `steam_appid.txt` and `collect.sh` deletes it.

## 5. Steam Cloud (Auto-Cloud, no code)

Steamworks > App Admin > Steam Cloud: set a byte and file quota (e.g. 50 MB,
2000 files), then under **Auto-Cloud** add these root paths. `<id>` is
`drive/branding.py`'s `save_id()` (today `Aleron`); pin it before the first
public build (RELEASE_CHECKLIST.md).

| Root | Subdirectory | Pattern | OS | Recursive | what |
|---|---|---|---|---|---|
| WinAppDataRoaming | `<id>/runs` | `settings.json` | Windows | no | settings |
| WinAppDataRoaming | `<id>/runs` | `progress.json` | Windows | no | medals, tutorial, challenges |
| WinAppDataRoaming | `<id>/runs` | `garage_design.json` | Windows | no | the garage's current design |
| WinAppDataRoaming | `<id>/runs/records` | `*.json` | Windows | no | top-5 laps per class + ghosts |
| WinAppDataRoaming | `<id>/runs/leaderboard_local` | `*.json` | Windows | no | local leaderboards |
| WinAppDataRoaming | `<id>/runs/library/builds` | `*.json` | Windows | no | saved cars |
| WinAppDataRoaming | `<id>/runs/library/wings` | `*.json` | Windows | no | saved wings |
| WinAppDataRoaming | `<id>/runs/library/airfoils` | `*.json` | Windows | no | saved sections |

Then, instead of repeating every row for the other two systems, add **Root
Overrides** (Root OS for every row above must then be "All OSes"):

| Original root | OS | New root | Add/replace path |
|---|---|---|---|
| WinAppDataRoaming | macOS | MacAppSupport | (none) |
| WinAppDataRoaming | Linux | LinuxXdgDataHome | (none) |

which maps `%APPDATA%\<id>\runs\...` to `~/Library/Application Support/<id>/runs/...`
and `$XDG_DATA_HOME/<id>/runs/...` -- exactly where `drive/userdata.py` puts
them. Root names from [Steam Cloud](https://partner.steamgames.com/doc/features/cloud).

Deliberately NOT synced (big, regenerable or per machine): `runs/*.csv`
telemetry recordings (~8 MB each), `runs/aerobo/` (AeroBO run outputs),
`runs/library/polars/` (a cache), `runs/export/`, `runs/swarm/` (bot training
states, several MB each -- OWNER DECISION: add
`<id>/runs/swarm` `*_state.json` if trained bots should follow the player),
`<id>/aerobo/` (AeroBO's copied engine + caches) and `<id>/logs/`.

`*.bak` files next to records are the game's own backups; they are not synced
on purpose (pattern `*.json` skips them).

## 6. Steam features the game does not use (yet)

No Steamworks SDK calls are made: no achievements, no Steam leaderboards, no
overlay hooks, no Steam Input API. Nothing of that is required to sell on
Steam. The plan (`.handoff/PLAN-steam-engagement.md`, T28) keeps a
`SteamBackend` for the leaderboards as later work: it needs a Python binding
for the Steamworks SDK shipped next to the game, and then `steam_appid.txt` in
development (section 4).

Controllers work through SDL (the game reads the pad itself), so in Steamworks
> Steam Input set the default to "use the game's native controller support" /
"Steam Input off by default" unless you test Steam Input's translation.
