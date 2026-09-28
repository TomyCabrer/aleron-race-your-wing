#!/usr/bin/env bash
# steam/upload.sh -- upload a build to Steam with steamcmd (SteamPipe).
#
#   STEAM_USER=<builder account> STEAM_APPID=<app id> \
#   STEAM_DEPOT_WINDOWS=<id> STEAM_DEPOT_MACOS=<id> STEAM_DEPOT_LINUX=<id> \
#   steam/upload.sh [--preview] [--beta <branch>] [--desc "<text>"]
#
# No credentials live in this repo. steamcmd asks for the password (and the
# Steam Guard code) itself the first time, then caches the login.
#
# Content (each folder = exactly what lands in the player's install dir):
#   WIN_DIR    default steam/content/windows   (Aleron.exe + _internal/)
#   MAC_DIR    default steam/content/macos     (Aleron.app)
#   LINUX_DIR  default steam/content/linux     (Aleron + _internal/)
# A platform whose depot id is unset or whose folder is missing is skipped.
# The file names follow drive/branding.py (ascii_name), so they change with
# the game's name.
#
# --preview   Valve's Preview mode: logs + manifests only, nothing uploaded.
#             Run this first after any change to the depots.
# --beta X    set the build live on beta branch X (create it in Steamworks
#             first). The default branch can never be set live from a script:
#             do that by hand in Steamworks > SteamPipe > Builds.
#
# See steam/README.md.

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

PREVIEW=0
SETLIVE=""
DESC=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --preview) PREVIEW=1; shift ;;
    --beta) SETLIVE="${2:?--beta needs a branch name}"; shift 2 ;;
    --desc) DESC="${2:?--desc needs text}"; shift 2 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

: "${STEAM_USER:?set STEAM_USER to your Steamworks builder account name}"
: "${STEAM_APPID:?set STEAM_APPID to the app id of the game, see Steamworks App Admin}"

PY="${PYTHON:-python3}"
NAME="$("$PY" -c 'from drive import branding as b; print(b.ascii_name())')"
VERSION="$("$PY" -c 'from drive import branding as b; print(b.VERSION)')"
REV="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
DESC="${DESC:-$NAME $VERSION ($REV)}"

WIN_DIR="${WIN_DIR:-$ROOT/steam/content/windows}"
MAC_DIR="${MAC_DIR:-$ROOT/steam/content/macos}"
LINUX_DIR="${LINUX_DIR:-$ROOT/steam/content/linux}"

OUT="$ROOT/steam/out"
mkdir -p "$OUT/logs"

# steamcmd: $STEAMCMD, else on the PATH, else the SDK's ContentBuilder copy
if [[ -z "${STEAMCMD:-}" ]]; then
  if command -v steamcmd >/dev/null 2>&1; then
    STEAMCMD="$(command -v steamcmd)"
  elif [[ -x "$ROOT/steam/sdk/tools/ContentBuilder/builder_osx/steamcmd.sh" ]]; then
    STEAMCMD="$ROOT/steam/sdk/tools/ContentBuilder/builder_osx/steamcmd.sh"
  elif [[ -x "$ROOT/steam/sdk/tools/ContentBuilder/builder_linux/steamcmd.sh" ]]; then
    STEAMCMD="$ROOT/steam/sdk/tools/ContentBuilder/builder_linux/steamcmd.sh"
  else
    echo "steamcmd not found: set STEAMCMD, or unzip the Steamworks SDK to steam/sdk/" >&2
    exit 1
  fi
fi

render() {  # render <template> <output> KEY=VALUE...  (each KEY must occur)
  "$PY" - "$@" <<'PYEOF'
import sys
src, dst, pairs = sys.argv[1], sys.argv[2], sys.argv[3:]
text = open(src).read()
for kv in pairs:
    k, v = kv.split("=", 1)
    assert k in text, f"{k} not in {src}"
    text = text.replace(k, v)
open(dst, "w").write(text)
PYEOF
}

DEPOTS=""
add_depot() {  # add_depot <os> <depot id> <content dir> <check glob>
  local os="$1" id="$2" dir="$3" want="$4"
  local var
  var="STEAM_DEPOT_$(printf %s "$os" | tr a-z A-Z)"
  if [[ -z "$id" ]]; then echo "skip $os: no depot id in $var"; return; fi
  if [[ ! -d "$dir" ]]; then echo "skip $os: no content at $dir"; return; fi
  # shellcheck disable=SC2086
  if ! compgen -G "$dir/$want" >/dev/null; then
    echo "error: $os content at $dir has no $want -- wrong folder?" >&2; exit 1
  fi
  if [[ -e "$dir/steam_appid.txt" ]]; then
    echo "note: $dir/steam_appid.txt is excluded from the depot (dev-only file)"
  fi
  render "$ROOT/steam/templates/depot_build_$os.vdf.in" "$OUT/depot_build_$os.vdf" \
    "__DEPOTID__=$id" "__CONTENT_DIR__=$dir"
  DEPOTS+=$'\t\t'"\"$id\" \"$OUT/depot_build_$os.vdf\""$'\n'
  echo "depot $id <- $os  ($dir)"
}

add_depot windows "${STEAM_DEPOT_WINDOWS:-}" "$WIN_DIR" "$NAME.exe"
add_depot macos   "${STEAM_DEPOT_MACOS:-}"   "$MAC_DIR" "$NAME.app"
add_depot linux   "${STEAM_DEPOT_LINUX:-}"   "$LINUX_DIR" "$NAME"

if [[ -z "$DEPOTS" ]]; then echo "nothing to upload" >&2; exit 1; fi

render "$ROOT/steam/templates/app_build.vdf.in" "$OUT/app_build.vdf" \
  "__APPID__=$STEAM_APPID" "__DESC__=$DESC" "__CONTENT_ROOT__=$ROOT" \
  "__BUILD_OUTPUT__=$OUT/logs" "__PREVIEW__=$PREVIEW" "__SETLIVE__=$SETLIVE"
# the depot lines go in last: they hold tabs and newlines sed would mangle
"$PY" - "$OUT/app_build.vdf" "$DEPOTS" <<'PYEOF'
import sys
path, depots = sys.argv[1], sys.argv[2]
text = open(path).read()
assert text.count("__DEPOTS__") == 1
open(path, "w").write(text.replace("__DEPOTS__\n", depots))
PYEOF

echo "app build: $OUT/app_build.vdf  (preview=$PREVIEW, setlive='${SETLIVE}', desc='$DESC')"
"$STEAMCMD" +login "$STEAM_USER" +run_app_build "$OUT/app_build.vdf" +quit
echo "done. Logs: $OUT/logs. Builds appear in Steamworks > SteamPipe > Builds."
