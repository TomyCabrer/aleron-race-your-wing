#!/usr/bin/env bash
# packaging/build_mac.sh -- build the macOS app (dist/<Name>.app) for Steam.
#
#   bash packaging/build_mac.sh            # build with a fresh venv in build/venv
#   SKIP_VENV=1 bash packaging/build_mac.sh  # use the python3 on PATH as it is
#
# The .app runs on the CPU it was built on: an Apple-silicon Mac builds arm64,
# an Intel Mac builds x86_64. Steam's macOS depot wants both -- build on each
# (or a universal2 Python) and see packaging/README.md.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
if [[ -z "${SKIP_VENV:-}" ]]; then
  "$PY" -m venv build/venv
  PY=build/venv/bin/python
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install -r packaging/requirements-build.txt
fi
[[ -f packaging/icons/game.icns ]] || "$PY" packaging/make_icons.py
"$PY" -m PyInstaller --noconfirm --clean --distpath dist --workpath build/pyinstaller packaging/game.spec
NAME="$("$PY" -c 'from drive import branding; print(branding.ascii_name())')"
echo
echo "built dist/$NAME.app ($(du -sh "dist/$NAME.app" | cut -f1))"
echo "smoke test: CARSIM_DATA_DIR=/tmp/fresh dist/$NAME.app/Contents/MacOS/$NAME"
