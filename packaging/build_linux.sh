#!/usr/bin/env bash
# packaging/build_linux.sh -- build the Linux folder (dist/<Name>/) for Steam.
#
# Build inside Valve's Steam Runtime SDK container ("sniper") so the binary runs
# on every Steam Linux client and on the Steam Deck:
#   docker run --rm -it -v "$PWD":/src -w /src \
#     registry.gitlab.steamos.cloud/steamrt/sniper/sdk bash packaging/build_linux.sh
# A plain Ubuntu build also works for testing.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
if [[ -z "${SKIP_VENV:-}" ]]; then
  "$PY" -m venv build/venv
  PY=build/venv/bin/python
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install -r packaging/requirements-build.txt
fi
"$PY" -m PyInstaller --noconfirm --clean --distpath dist --workpath build/pyinstaller packaging/game.spec
NAME="$("$PY" -c 'from drive import branding; print(branding.ascii_name())')"
echo "built dist/$NAME/ ($(du -sh "dist/$NAME" | cut -f1)); launch: dist/$NAME/$NAME"
