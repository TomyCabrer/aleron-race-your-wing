#!/usr/bin/env bash
# steam/collect.sh -- copy THIS machine's packaged build into steam/content/<os>/,
# the folder steam/upload.sh uploads as that platform's depot.
#
#   packaging/build_*.sh (or .ps1)   ->  dist/<Name>[.app]   (or packaging/dist/)
#   steam/collect.sh                 ->  steam/content/<os>/
#
# Builds are per OS (PyInstaller does not cross-compile): run the build and
# this script on each OS, then bring the three steam/content/<os>/ folders to
# the machine that runs steam/upload.sh.

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
PY="${PYTHON:-python3}"
NAME="$("$PY" -c 'from drive import branding as b; print(b.ascii_name())')"

DIST=""
for d in "$ROOT/dist" "$ROOT/packaging/dist"; do
  if [[ -d "$d" ]]; then DIST="$d"; break; fi
done
if [[ -z "$DIST" ]]; then echo "no dist/ (or packaging/dist/) folder: build first" >&2; exit 1; fi

case "$(uname -s)" in
  Darwin)
    OS=macos; SRC="$DIST/$NAME.app"
    [[ -d "$SRC" ]] || { echo "no $SRC" >&2; exit 1; }
    ;;
  Linux)
    OS=linux; SRC="$DIST/$NAME"
    [[ -x "$SRC/$NAME" ]] || { echo "no $SRC/$NAME" >&2; exit 1; }
    ;;
  MINGW*|MSYS*|CYGWIN*)
    OS=windows; SRC="$DIST/$NAME"
    [[ -f "$SRC/$NAME.exe" ]] || { echo "no $SRC/$NAME.exe" >&2; exit 1; }
    ;;
  *) echo "unknown OS $(uname -s)" >&2; exit 1 ;;
esac

DST="$ROOT/steam/content/$OS"
rm -rf "$DST"
mkdir -p "$DST"
if [[ "$OS" == macos ]]; then
  # the depot's root holds the .app itself; ditto keeps the bundle's symlinks,
  # permissions and signature intact
  ditto "$SRC" "$DST/$NAME.app"
else
  # the depot's root holds the folder's CONTENTS: <Name>[.exe] + _internal/
  cp -R "$SRC/." "$DST/"
fi
rm -f "$DST/steam_appid.txt"
echo "$OS depot content: $DST"
du -sh "$DST"
