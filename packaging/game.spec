# -*- mode: python ; coding: utf-8 -*-
"""packaging/game.spec -- the PyInstaller build of the game (Steam prep).

    pyinstaller --noconfirm --clean packaging/game.spec      (see build_*.sh / .ps1)

One-folder build (Steam ships a folder; a one-file exe unpacks itself to a temp
dir on every launch). On macOS the folder is wrapped as <Name>.app. The name
and version come from drive/branding.py, so a rename is one file.

What ships, laid out as in the repo because the code finds its files from
its own `__file__`:
  drive/ code          -> in the archive; its data files (drive/data/**,
                          drive/aero/data/**, drive/ml/checkpoints/**) as files
  cars / corsa_c / qss -> in the archive (drive imports them)
  tyre_data/<.tir>     -> files: only the ones the player's cars (cars.CARS)
                          read (drive/tyre.py, drive/powertrain.py)
  licences             -> licenses/ (every shipped package's own licence
                          files + packaging/licenses/ + ours); macOS: inside
                          the .app's Resources, Windows/Linux: next to the exe
  aerobo/              -> REAL FILES, not the archive: AeroBO finds its data
                          and its caches from `Path(__file__).parents[2]`;
                          launch_game.py copies this tree to the player's data
                          folder and drive/aerobo_bridge.py imports it from there
Never shipped: runs/, .handoff/, .git, aerobo/results/, __pycache__, torch.
"""

import os
import sys
from pathlib import Path

ROOT = Path(SPECPATH).resolve().parent
sys.path.insert(0, str(ROOT))
from drive import branding  # noqa: E402

NAME = branding.ascii_name()
JUNK = {"__pycache__", ".DS_Store"}


def tree(src_rel, dst_rel=None, suffixes=None, skip_py=True):
    """(file, dest dir) pairs for every file under ROOT/src_rel."""
    out = []
    base = ROOT / src_rel
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in JUNK]
        for fn in filenames:
            if fn in JUNK or fn.endswith(".pyc"):
                continue
            if skip_py and fn.endswith(".py"):
                continue
            if suffixes and not fn.endswith(tuple(suffixes)):
                continue
            rel = Path(dirpath).relative_to(ROOT)
            if dst_rel is not None:
                rel = Path(dst_rel) / Path(dirpath).relative_to(base)
            out.append((str(Path(dirpath) / fn), str(rel)))
    return out


def modules(pkg_dir, prefix):
    """Every module under ROOT/pkg_dir, as dotted names (no imports run)."""
    names = []
    base = ROOT / pkg_dir
    for py in base.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        rel = py.relative_to(base).with_suffix("")
        parts = [p for p in rel.parts if p != "__init__"]
        names.append(".".join([prefix] + parts) if parts else prefix)
    return sorted(set(names))


datas = []
datas += tree("drive/data")
datas += tree("drive/aero/data")
datas += tree("drive/ml/checkpoints")
# the tyre files the player's cars read -- not the whole study folder
try:
    import cars as _cars
    TYRE_FILES = sorted({c.tyre_file for c in _cars.CARS.values()} | {_cars.TYRE_REF})
except Exception as exc:                       # noqa: BLE001
    print(f"game.spec: cars.py would not import ({exc}); shipping the default tyre only")
    TYRE_FILES = ["tyre_data/TNO_car205_60R15.tir"]
for rel in TYRE_FILES:
    datas.append((str(ROOT / rel), str(Path(rel).parent)))
# AeroBO as real files, sources included (see the docstring)
for part in ("src", "data", "records", "seed"):
    datas += tree(f"aerobo/{part}", skip_py=False)
datas += [(str(ROOT / "aerobo" / "LICENSE"), "aerobo"),
          (str(ROOT / "aerobo" / "VENDORED.md"), "aerobo")]
# (the licences are gathered after the Analysis, from what it collected)

AEROBO_MODULES = modules("aerobo/src/aerobo", "aerobo")
hiddenimports = (modules("drive", "drive")
                 + ["cars", "corsa_c", "qss", "crossover", "ledger"]
                 # traced so their third-party imports (pymoo, scipy ...) are
                 # collected; the aerobo modules themselves are dropped below
                 + AEROBO_MODULES)
# pymoo loads parts of itself by name; scipy and pymoo read data files of
# their own (scipy.stats' Sobol table is what AeroBO's samplers need)
try:
    from PyInstaller.utils.hooks import collect_submodules, collect_data_files
    datas += collect_data_files("scipy", excludes=["**/tests/**"])
    datas += collect_data_files("pymoo")
    hiddenimports += collect_submodules(
        "pymoo", filter=lambda m: not m.startswith("pymoo.gradient"))  # its gradient toolbox aliases numpy
except Exception:
    pass

EXCLUDES = ["torch", "torchvision", "torchaudio", "botorch", "gpytorch", "pyro",
            "tkinter", "_tkinter", "IPython", "jupyter", "notebook", "pytest",
            "sphinx", "PyQt5", "PyQt6", "PySide2", "PySide6", "wx"]

a = Analysis(
    [str(ROOT / "launch_game.py")],
    pathex=[str(ROOT), str(ROOT / "aerobo" / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
    optimize=0,
)
# AeroBO must import from its files (parents[2] lookups), never the archive
a.pure = [e for e in a.pure if not (e[0] == "aerobo" or e[0].startswith("aerobo."))]


def licence_files():
    """(source file, path under licenses/) for everything the build must
    carry: ours (THIRD_PARTY_NOTICES.md, AeroBO's, the bundled fonts'),
    packaging/licenses/ (LGPL-2.1 text, the source offer, pygame's native
    libraries), Python's own licence, PyInstaller's (its bootloader is the
    exe) and every distribution whose modules, binaries or data this build
    collected, with the licence files its wheel carries."""
    import importlib.metadata as md
    import re
    out = []
    ours = [(ROOT / "THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.md"),
            (ROOT / "aerobo" / "LICENSE", "AeroBO/LICENSE")]
    ours += [(f, f"fonts/{f.name}") for f in sorted((ROOT / "drive" / "data" / "fonts").glob("*"))
             if re.search(r"licen[cs]e|ofl", f.name, re.I)]
    ours += [(f, f.name) for f in sorted((ROOT / "packaging" / "licenses").glob("*")) if f.is_file()]
    for cand in (Path(sys.base_prefix) / "lib" / f"python{sys.version_info[0]}.{sys.version_info[1]}" / "LICENSE.txt",
                 Path(sys.base_prefix) / "LICENSE.txt", Path(sys.base_prefix) / "LICENSE"):
        if cand.is_file():
            ours.append((cand, "Python/LICENSE.txt"))
            break
    out += [(str(f), dst) for f, dst in ours if Path(f).is_file()]
    # the distributions this build collected, by their top-level names
    tops = {e[0].split(".")[0] for e in a.pure}
    tops |= {Path(e[0]).parts[0] for e in list(a.binaries) + list(a.datas) if Path(e[0]).parts}
    owners = md.packages_distributions()
    dists = {d for t in tops for d in owners.get(t, ())} | {"pyinstaller"}
    index = []
    for name in sorted(dists, key=str.lower):
        try:
            dist = md.distribution(name)
        except md.PackageNotFoundError:
            continue
        found = 0
        for f in dist.files or []:
            fn = Path(str(f))
            if fn.suffix in (".py", ".pyc", ".pyi"):
                continue
            if not (re.search(r"licen[cs]e|copying|notice|copyright|authors", fn.name, re.I)
                    or "licenses" in fn.parts[:2]):
                continue
            src = Path(dist.locate_file(f)).resolve()
            if not src.is_file():
                continue
            parts = [p for p in fn.parts if p not in ("..",)]
            if parts and parts[0].endswith(".dist-info"):
                parts = parts[1:]
            if parts and parts[0] == "licenses":
                parts = parts[1:]
            out.append((str(src), f"{dist.metadata['Name']}/{'__'.join(parts) or fn.name}"))
            found += 1
        lic = dist.metadata.get("License-Expression") or next(
            (ln.strip() for ln in (dist.metadata.get("License") or "").splitlines()
             if re.search(r"[A-Za-z]", ln)), "")[:60]
        index.append(f"{dist.metadata['Name']} {dist.version}: {lic or 'see its files'}"
                     f"{'' if found else '  (its wheel carries no licence file; see BUNDLED_LIBRARIES.md / THIRD_PARTY_NOTICES.md)'}")
    # an index written into the build's work folder, shipped as licenses/README.txt
    readme = Path(workpath) / "licenses-README.txt"
    readme.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text(
        f"{branding.GAME_NAME} {branding.VERSION} -- third-party licences\n\n"
        "THIRD_PARTY_NOTICES.md is the overview. Each folder holds a shipped\n"
        "package's own licence files; LGPL-2.1.txt and SOURCE_OFFER.md cover the\n"
        "LGPL libraries; BUNDLED_LIBRARIES.md lists pygame's native libraries.\n\n"
        "Packages in this build:\n" + "\n".join("  " + i for i in index) + "\n",
        encoding="utf-8")
    out.append((str(readme), "README.txt"))
    seen, uniq = set(), []
    for src, dst in out:
        if dst not in seen:
            seen.add(dst)
            uniq.append((src, dst))
    return uniq


LICENCES = licence_files()
if sys.platform == "darwin":
    # inside the .app (Contents/Resources/licenses)
    a.datas += [(f"licenses/{dst}", src, "DATA") for src, dst in LICENCES]
print(f"game.spec: {len(LICENCES)} licence files, tyre files {TYRE_FILES}")

pyz = PYZ(a.pure)

ICON_DIR = ROOT / "packaging" / "icons"
if sys.platform == "darwin":
    icon = ICON_DIR / "game.icns"
elif os.name == "nt":
    icon = ICON_DIR / "game.ico"
else:
    icon = None
icon = str(icon) if icon is not None and icon.is_file() else None

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,          # windowed: the log is <data>/logs/game.log
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=NAME,
)
if sys.platform != "darwin":
    # Windows / Linux: licenses/ next to the exe, where a player looks
    import shutil
    for src, dst in LICENCES:
        target = Path(DISTPATH) / NAME / "licenses" / dst
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name=f"{NAME}.app",
        icon=icon,
        bundle_identifier=f"com.example.{NAME.lower()}",   # TODO(owner): your reverse-DNS id
        version=branding.VERSION,
        info_plist={
            "CFBundleName": NAME,
            "CFBundleDisplayName": branding.GAME_NAME,
            "CFBundleShortVersionString": branding.VERSION,
            "CFBundleVersion": branding.VERSION,
            "NSHighResolutionCapable": True,
            "LSApplicationCategoryType": "public.app-category.racing-games",
            "LSMinimumSystemVersion": "11.0",
        },
    )
