"""packaging/make_icons.py -- the app icon, drawn from the title screen's logo.

    python3 packaging/make_icons.py

Writes packaging/icons/game.png (1024 px), game.ico (Windows, needs Pillow) and
game.icns (macOS, needs `iconutil`, i.e. a Mac). The logo is drive/title.py's
`logo_surfaces` -- the word in drive/branding.py -- on a dark rounded tile, so
after a rename run this again (the build scripts only make icons that are
missing: delete packaging/icons/ first).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pygame  # noqa: E402

OUT = os.path.join(ROOT, "packaging", "icons")
SIZE = 1024
TILE = (14, 22, 42)          # the logo's dark navy, as a tile


def draw(size: int = SIZE) -> pygame.Surface:
    from drive import title
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    r = int(size * 0.18)
    pygame.draw.rect(surf, TILE, surf.get_rect(), border_radius=r)
    logo, shadow = title.logo_surfaces(2.0, ss=2)
    w = int(size * 0.86)
    h = max(1, int(logo.get_height() * w / logo.get_width()))
    logo = pygame.transform.smoothscale(logo, (w, h))
    surf.blit(logo, ((size - w) // 2, (size - h) // 2))
    return surf


def main() -> int:
    pygame.init()
    pygame.display.set_mode((16, 16))
    os.makedirs(OUT, exist_ok=True)
    png = os.path.join(OUT, "game.png")
    pygame.image.save(draw(), png)
    print(f"wrote {png}")
    try:
        from PIL import Image
        Image.open(png).save(os.path.join(OUT, "game.ico"),
                             sizes=[(s, s) for s in (16, 24, 32, 48, 64, 128, 256)])
        print(f"wrote {os.path.join(OUT, 'game.ico')}")
    except ImportError:
        print("no Pillow: game.ico not written (pip install pillow)")
    if shutil.which("iconutil"):
        with tempfile.TemporaryDirectory() as tmp:
            iconset = os.path.join(tmp, "game.iconset")
            os.makedirs(iconset)
            big = pygame.image.load(png)
            for s in (16, 32, 128, 256, 512):
                for k, suffix in ((1, ""), (2, "@2x")):
                    px = s * k
                    pygame.image.save(pygame.transform.smoothscale(big, (px, px)),
                                      os.path.join(iconset, f"icon_{s}x{s}{suffix}.png"))
            subprocess.run(["iconutil", "-c", "icns", iconset, "-o",
                            os.path.join(OUT, "game.icns")], check=True)
        print(f"wrote {os.path.join(OUT, 'game.icns')}")
    else:
        print("no iconutil (not a Mac): game.icns not written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
