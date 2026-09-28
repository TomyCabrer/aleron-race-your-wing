"""launch_game.py -- the game's entry point, from source and in a packaged build.

From a source checkout `python3 launch_game.py [flags]` is exactly
`python3 -m drive.drive [flags]`: same flags, the repo stays the working
directory, saves go to the repo's runs/.

In a packaged build (PyInstaller, packaging/game.spec; `sys.frozen` is set):

* `multiprocessing.freeze_support()` comes first, so a swarm or bot-test
  worker the game spawns runs its job instead of a second copy of the game;
* the working directory becomes the per-user data folder
  (drive/userdata.py), so every relative `runs/...` the game writes lands
  there and never inside the install folder;
* the vendored AeroBO tree is copied once per version next to it
  (`<data>/aerobo`): AeroBO keeps its caches under its own `results/`, and
  the install folder is not the player's to write into;
* stdout / stderr go to `<data>/logs/game.log` (the previous run's kept as
  game.prev.log): a windowed app has no terminal;
* an uncaught error is written to `<data>/logs/crash-<time>.txt` and a
  native alert says where; a failure before the saves folder exists (it is
  read-only, the disk is full) is written to the temp folder instead
  (`<save id>-crash-<time>.txt`), with the same alert;
* a launch argument the game does not know (a Steam launch option) is
  dropped and logged instead of quitting silently with argparse's usage
  error (`_lenient_args`);
* the shipped bots (drive/ml/checkpoints/*.json) are copied into
  `<data>/drive/ml/checkpoints/`: the game looks them up there, relative to
  the working directory, and saves the player's bred bots there too. An
  update refreshes a shipped bot only while the installed copy is untouched
  (a player's file is never overwritten);
* drive/display_mode.py is installed: fullscreen (remembered, F11 or
  Alt+Enter toggles), a scaled resizable window, DPI-aware on Windows.
  `--fullscreen` / `--windowed` set it; `CARSIM_DISPLAY_MODE=1` turns it on
  for a source run.

`CARSIM_DATA_DIR=<folder>` gives a source run the packaged layout too (tests,
a portable install).
"""

from __future__ import annotations

import os
import sys


def _log_to_file(logs) -> None:
    """stdout / stderr into logs/game.log, line-buffered; the last run's log
    is kept as game.prev.log."""
    log = logs / "game.log"
    try:
        if log.exists():
            os.replace(log, logs / "game.prev.log")
    except OSError:
        pass
    fh = open(log, "w", buffering=1, encoding="utf-8", errors="replace")
    sys.stdout = sys.stderr = fh


def _write_crash(folder, prefix: str = "crash") -> str:
    """The current exception's traceback into `<folder>/<prefix>-<time>.txt`;
    "" if even that cannot be written."""
    import time
    import traceback
    path = os.path.join(str(folder), f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}.txt")
    try:
        from drive import branding
        head = f"{branding.GAME_NAME} {branding.VERSION} on {sys.platform}\n"
    except Exception:
        head = ""
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(head + " ".join(sys.argv) + "\n\n")
            traceback.print_exc(file=fh)
    except OSError:
        return ""
    return str(path)


def _message_box(text: str) -> None:
    """A native one-line alert: a windowed build has no terminal, so a crash
    must say where its report is. Skipped headless (SDL's dummy / offscreen
    drivers, the tests, CI): the alert is modal and would wait for a click.
    pygame-ce has `pygame.display.message_box`; pygame (2.6) does not, so the
    SDL library pygame bundles is called directly -- SDL_ShowSimpleMessageBox
    works before SDL_Init and with no window."""
    if (os.environ.get("SDL_VIDEODRIVER") in ("dummy", "offscreen")
            or os.environ.get("CARSIM_NO_MSGBOX")):
        return
    try:
        from drive import branding
        title = branding.GAME_NAME
    except Exception:
        title = "Error"
    try:
        import pygame
        try:
            pygame.display.quit()           # out of fullscreen, so the alert is seen
        except Exception:
            pass
        box = getattr(pygame.display, "message_box", None)
        if box is not None:
            box(title, text, message_type="error")
            return
        fn = _sdl_function("SDL_ShowSimpleMessageBox")
        if fn is not None:
            import ctypes
            fn.argtypes = [ctypes.c_uint32, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p]
            fn(0x10, title.encode("utf-8"), text.encode("utf-8"), None)   # SDL_MESSAGEBOX_ERROR
    except Exception:
        pass


def _sdl_function(name: str):
    """`name` from the SDL2 library pygame ships (already loaded by pygame's
    import: loading it by path again returns the same library), or None."""
    import ctypes
    import glob
    import pygame
    base = os.path.dirname(os.path.abspath(pygame.__file__))
    patterns = ("SDL2.dll",                                          # Windows wheels
                os.path.join(".dylibs", "libSDL2-2.0*.dylib"),       # macOS wheels
                os.path.join(os.pardir, "pygame.libs", "libSDL2-2*.so*"))  # manylinux
    for pat in patterns:
        for path in sorted(glob.glob(os.path.join(base, pat))):
            try:
                return getattr(ctypes.CDLL(path), name)
            except (OSError, AttributeError):
                continue
    return None


def _lenient_args() -> None:
    """A packaged build does not quit on a launch argument it does not know
    (a Steam launch option, a typo in a shortcut): argparse would print its
    usage into the log and exit 2, and the player would see nothing at all.
    drive.main's parser is wrapped so an unknown argument is dropped and an
    invalid value for a known one falls back to no arguments -- both said in
    logs/game.log. `--help` still exits. A source run keeps argparse strict."""
    from drive import drive
    build = drive.build_parser

    def lenient():
        p = build()

        def parse_args(args=None, namespace=None):
            try:
                ns, extra = p.parse_known_args(args, namespace)
            except SystemExit as exc:
                if exc.code in (0, None):
                    raise
                print(f"launch: the launch arguments {args!r} are not valid; "
                      f"starting with none")
                return p.parse_known_args([], namespace)[0]
            if extra:
                print(f"launch: ignoring unknown launch arguments {extra!r}")
            return ns

        p.parse_args = parse_args
        return p

    drive.build_parser = lenient


def _install_aerobo(root) -> None:
    """Copy the bundle's vendored AeroBO next to the saves, once per version,
    and point drive/aerobo_bridge.py at the copy (CARSIM_AEROBO_ROOT). The
    copy's own results/ (AeroBO's caches) survives an update."""
    import shutil
    from pathlib import Path
    from drive import branding
    src = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)) / "aerobo"
    if not src.is_dir():
        return
    dst = Path(root) / "aerobo"
    stamp = dst / ".installed"
    try:
        want = branding.VERSION + "\n" + (src / "VENDORED.md").read_text(encoding="utf-8")
    except OSError:
        want = branding.VERSION
    have = stamp.read_text(encoding="utf-8") if stamp.is_file() else None
    if have != want:
        for part in ("src", "data", "records", "seed"):
            if (dst / part).exists():
                shutil.rmtree(dst / part)
            if (src / part).is_dir():
                shutil.copytree(src / part, dst / part,
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))
        for f in ("LICENSE", "VENDORED.md"):
            if (src / f).is_file():
                shutil.copy2(src / f, dst / f)
        stamp.write_text(want, encoding="utf-8")
    os.environ["CARSIM_AEROBO_ROOT"] = str(dst)


def _install_bots(root) -> None:
    """Copy the bundle's shipped bots into `<data>/drive/ml/checkpoints/`.
    A missing one is copied; one an update ships anew replaces the copy an
    earlier version installed, but only while that copy is untouched (its
    hash is the one recorded in `.installed.json`): a player's file, or a
    shipped one the game has since rewritten, is never overwritten."""
    import hashlib
    import json
    import shutil
    from pathlib import Path
    src = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)) / "drive" / "ml" / "checkpoints"
    if not src.is_dir():
        return
    dst = Path(root) / "drive" / "ml" / "checkpoints"
    dst.mkdir(parents=True, exist_ok=True)
    stamp = dst / ".installed.json"

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    try:
        installed = json.loads(stamp.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        installed = {}
    for f in sorted(src.glob("*.json")):
        new, target = sha(f), dst / f.name
        if target.exists():
            have = sha(target)
            if have == new or installed.get(f.name) != have:
                continue                    # current, or not ours to replace
        shutil.copy2(f, target)
        installed[f.name] = new
    try:
        stamp.write_text(json.dumps(installed, indent=1, sort_keys=True), encoding="utf-8")
    except OSError:
        pass


def probe() -> int:
    """`--probe`: import what a player can reach, report where AeroBO was
    loaded from and quit -- the packaged build's smoke test (packaging/, CI)."""
    import importlib
    import time
    t0 = time.time()
    for m in ("drive.drive", "drive.title", "drive.garage", "drive.aerobo_bridge",
              "drive.design_jobs", "drive.ml.swarm", "drive.ml.evaluate",
              "drive.leaderboard", "drive.challenges"):
        importlib.import_module(m)
        print(f"probe: import {m} ok")
    from drive import aerobo_bridge as ab
    print(f"probe: aerobo from {ab.api.__file__}")
    print(f"probe: screen checkpoint {ab.api.SCREEN_CHECKPOINT} "
          f"{'present' if ab.api.SCREEN_CHECKPOINT.is_file() else 'MISSING'}")
    import pygame
    from drive import title
    pygame.init()
    pygame.display.set_mode((64, 64))
    logo, _ = title.logo_surfaces(1.0)
    alert = ("pygame.display.message_box" if hasattr(pygame.display, "message_box")
             else "SDL_ShowSimpleMessageBox" if _sdl_function("SDL_ShowSimpleMessageBox")
             else "NONE (a crash is only in the logs)")
    print(f"probe: crash alert via {alert}")
    print(f"probe: logo {logo.get_size()}; cwd {os.getcwd()}; {time.time() - t0:.1f} s")
    return 0


def display_mode_wanted(argv) -> bool:
    from drive import display_mode
    return display_mode.wanted(argv)


def run(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # macOS Finder / older Steam clients may hand a .app a process serial number
    argv = [a for a in argv if not a.startswith("-psn_")]
    from drive import userdata
    frozen = userdata.frozen()
    logs = None
    if frozen or os.environ.get(userdata.ENV):
        try:
            root = userdata.enter_data_root()
            if frozen:
                logs = userdata.logs_dir()
                # the probe reports to its caller, if it has a console to report to
                if argv[:1] != ["--probe"] or sys.stdout is None:
                    _log_to_file(logs)
                _install_aerobo(root)
            _install_bots(root)
        except Exception as exc:
            # the saves folder cannot be made or filled (read-only, disk
            # full): no log exists yet, so the report goes to the temp folder
            if not frozen:
                raise
            import tempfile
            from drive import branding
            where = _write_crash(tempfile.gettempdir(), f"{branding.save_id()}-crash")
            print(f"setup failed ({type(exc).__name__}: {exc}); the report is {where}")
            _message_box(f"{branding.GAME_NAME} could not set up its saves folder "
                         f"{userdata.data_root()}: {exc}. The report is {where or 'unwritable'}.")
            return 1
    if frozen or display_mode_wanted(argv):
        from drive import display_mode
        argv = display_mode.install(argv)
    try:
        if argv[:1] == ["--probe"]:
            return probe()
        if frozen:
            _lenient_args()
        from drive import drive
        return drive.main(argv)
    except SystemExit:
        raise
    except BaseException:
        if logs is None:
            raise
        where = _write_crash(logs)
        print(f"crashed; the report is {where}")
        from drive import branding
        _message_box(f"{branding.GAME_NAME} stopped with an error. "
                     f"The report is {where or 'in ' + str(logs)}.")
        return 1


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    sys.exit(run())
