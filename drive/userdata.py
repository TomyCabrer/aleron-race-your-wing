"""drive/userdata.py -- where a packaged (Steam) build keeps the player's files.

The game writes everything it saves under a relative `runs/` (settings,
progress, records, builds, leaderboards, the section library, AeroBO runs).
From a source checkout that is the repo's `runs/`, exactly as before. A
packaged build lives in a folder the player must not write into (a signed
macOS .app, Steam's install dir), so its launcher (launch_game.py) calls
`enter_data_root()` first: the working directory becomes the per-user folder
below and every `runs/...` path lands there.

    macOS    ~/Library/Application Support/<save id>/runs/
    Windows  %APPDATA%\\<save id>\\runs\\
    Linux    $XDG_DATA_HOME/<save id>/runs/  (~/.local/share/...)

`CARSIM_DATA_DIR` overrides the folder (tests, a portable install). The save
id is drive/branding.py's `save_id()`. Steam Cloud syncs `runs/` under that
folder (steam/README.md).

Pure data: importing this pulls in no pygame.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from . import branding

ENV = "CARSIM_DATA_DIR"


def frozen() -> bool:
    """True inside a packaged build (PyInstaller sets `sys.frozen`)."""
    return bool(getattr(sys, "frozen", False))


def data_root() -> Path:
    """The per-user folder the saves live in (created by `enter_data_root`)."""
    env = os.environ.get(ENV)
    if env:
        return Path(env).expanduser().resolve()
    name = branding.save_id()
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif os.name == "nt":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / name


def enter_data_root() -> Path:
    """Create the folder and its `runs/`, make it the working directory and
    return it. Only a packaged build (or a run with CARSIM_DATA_DIR set) calls
    this; a source checkout keeps the repo as its working directory."""
    root = data_root()
    (root / "runs").mkdir(parents=True, exist_ok=True)
    os.chdir(root)
    return root


def logs_dir() -> Path:
    """Where a packaged build writes its log and crash reports."""
    d = data_root() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d
