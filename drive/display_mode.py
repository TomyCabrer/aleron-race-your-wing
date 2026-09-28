"""drive/display_mode.py -- window / fullscreen for the packaged (Steam) build.

The game draws at a fixed logical size (1280x800: `render.py`, `title.py`,
`garage.py` each call `pygame.display.set_mode(size)` with no flags). On a
player's machine that is a problem: no fullscreen, no resizing, and on a
Windows laptop at 150 % scaling a DPI-unaware SDL window is blown up past the
screen. `install()` fixes it without touching those three calls:

* `SDL_WINDOWS_DPI_AWARENESS=permonitorv2` (setdefault) before SDL starts, so
  Windows stops bitmap-stretching the window;
* `pygame.display.set_mode` is wrapped: every real (non-dummy) window gets
  `SCALED | RESIZABLE` -- SDL scales the game's logical surface to the window
  or screen and letterboxes it; mouse positions stay in logical pixels, so
  every click lands where it did -- plus `FULLSCREEN` while fullscreen is on;
* `pygame.event.get` / `poll` / `wait` are wrapped: F11 and Alt+Enter toggle
  fullscreen and are swallowed (the game never sees them);
* the choice is remembered in `runs/display.json` (the saves folder in a
  packaged build). First launch: fullscreen.

Why a wrapper and not three edited `set_mode` calls: those files are being
edited by other work; one module keeps the behaviour in one place; and a dev
run (`python3 -m drive.drive`) is untouched -- only launch_game.py installs
this (a packaged build, `CARSIM_DISPLAY_MODE=1`, or `--fullscreen` /
`--windowed`). Headless runs (SDL_VIDEODRIVER dummy / offscreen) get no flags.
"""

from __future__ import annotations

import json
import os

PATH = os.path.join("runs", "display.json")
ENV = "CARSIM_DISPLAY_MODE"
DEFAULT_FULLSCREEN = True

_state = {"fullscreen": DEFAULT_FULLSCREEN}
_orig = {}
_swallow_up = set()          # keys whose KEYUP is eaten with their KEYDOWN


def _load() -> None:
    try:
        with open(PATH, encoding="utf-8") as fh:
            d = json.load(fh)
        if isinstance(d, dict) and isinstance(d.get("fullscreen"), bool):
            _state["fullscreen"] = d["fullscreen"]
    except (OSError, ValueError):
        pass


def _save() -> None:
    try:
        os.makedirs(os.path.dirname(PATH), exist_ok=True)
        tmp = PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"fullscreen": _state["fullscreen"]}, fh)
        os.replace(tmp, PATH)
    except OSError:
        pass


def _headless() -> bool:
    return os.environ.get("SDL_VIDEODRIVER", "").lower() in ("dummy", "offscreen")


def fullscreen() -> bool:
    return _state["fullscreen"]


def _set_mode(size=(0, 0), flags=0, *args, **kwargs):
    import pygame
    orig = _orig["set_mode"]
    if _headless() or not size or not all(size):
        return orig(size, flags, *args, **kwargs)
    extra = pygame.SCALED | pygame.RESIZABLE
    if _state["fullscreen"]:
        extra |= pygame.FULLSCREEN
    try:
        return orig(size, flags | extra, *args, **kwargs)
    except pygame.error as exc:          # a driver without a renderer: plain window
        print(f"display: scaled window refused ({exc}); plain window")
        return orig(size, flags, *args, **kwargs)


def toggle() -> None:
    """Flip fullscreen on the live window and remember it."""
    import pygame
    _state["fullscreen"] = not _state["fullscreen"]
    _save()
    if _headless() or pygame.display.get_surface() is None:
        return
    try:
        pygame.display.toggle_fullscreen()
    except pygame.error as exc:
        print(f"display: fullscreen toggle failed ({exc})")


def _is_toggle(ev) -> bool:
    import pygame
    if ev.type != pygame.KEYDOWN:
        return False
    key = getattr(ev, "key", None)
    if key == pygame.K_F11:
        return True
    mod = getattr(ev, "mod", 0) or 0
    return key in (pygame.K_RETURN, pygame.K_KP_ENTER) and bool(mod & pygame.KMOD_ALT)


def _filter(events):
    import pygame
    out = []
    for ev in events:
        if _is_toggle(ev):
            _swallow_up.add(ev.key)
            toggle()
            continue
        if ev.type == pygame.KEYUP and getattr(ev, "key", None) in _swallow_up:
            _swallow_up.discard(ev.key)
            continue
        out.append(ev)
    return out


def _get(*args, **kwargs):
    return _filter(_orig["get"](*args, **kwargs))


def _poll():
    import pygame
    while True:
        ev = _orig["poll"]()
        if ev.type == pygame.NOEVENT or _filter([ev]):
            return ev


def _wait(*args, **kwargs):
    while True:
        ev = _orig["wait"](*args, **kwargs)
        if _filter([ev]):
            return ev


def wanted(argv) -> bool:
    """Whether a source run asked for this module (packaged builds always do)."""
    return bool(os.environ.get(ENV)) or "--fullscreen" in argv or "--windowed" in argv


def install(argv: list) -> list:
    """Install the wrappers (once) and return `argv` without --fullscreen /
    --windowed. Call before anything opens a window."""
    os.environ.setdefault("SDL_WINDOWS_DPI_AWARENESS", "permonitorv2")
    import pygame
    _load()
    if "--fullscreen" in argv or "--windowed" in argv:
        _state["fullscreen"] = "--fullscreen" in argv and "--windowed" not in argv
        _save()
    if not _orig:
        _orig.update(set_mode=pygame.display.set_mode, get=pygame.event.get,
                     poll=pygame.event.poll, wait=pygame.event.wait)
        pygame.display.set_mode = _set_mode
        pygame.event.get = _get
        pygame.event.poll = _poll
        pygame.event.wait = _wait
    return [a for a in argv if a not in ("--fullscreen", "--windowed")]
