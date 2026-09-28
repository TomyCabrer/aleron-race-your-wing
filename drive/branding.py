"""drive/branding.py -- the game's name, in ONE place (Steam prep, 2026-09-28).

To rename the game, edit the four strings below and nothing else: the window
captions, the title screen's logo and subtitle, the packaged app's file name
(packaging/), the store texts and the store art (steam/) all read them.

`SAVE_ID_PINNED` names the folder the player's saves live in (see
drive/userdata.py). While it is None the folder follows `GAME_SHORT`. Once a
build is public, pin it to the value `save_id()` returns then and never change
it again: a later rename would otherwise strand every player's saves.

Pure data: importing this pulls in no pygame.
"""

from __future__ import annotations

import re
import unicodedata

#: the full title: window caption, store page, the app's display name
GAME_NAME = "Alerón: Race Your Wing"
#: the short name: menus, the app / exe file name (made ASCII by `ascii_name`)
GAME_SHORT = "Alerón"
#: the word the title screen's logo spells. The logo and the subtitle are the
#: owner's art (drive/data/art, drive/title.py): another word or line here is
#: set plainly in the menu's font until it has new art
LOGO_WORD = "ALERÓN"
#: the line under the logo
SUBTITLE = "Race Your Wing"

#: the build's version: the packaged app's metadata and the store build notes
VERSION = "0.9.0"

#: the saves folder's name, pinned after the first public build (see above)
SAVE_ID_PINNED: str | None = None


def ascii_name(text: str | None = None) -> str:
    """`text` (default GAME_SHORT) as a file-name-safe ASCII word:
    "Alerón" -> "Aleron". Accents dropped, anything else not [A-Za-z0-9] removed."""
    s = unicodedata.normalize("NFKD", GAME_SHORT if text is None else text)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^A-Za-z0-9]+", "", s)
    return s or "Game"


def save_id() -> str:
    """The saves folder's name: the pinned one, else `ascii_name()`."""
    return SAVE_ID_PINNED or ascii_name()


def caption(page: str | None = None) -> str:
    """A window caption: the full title, or "<short> - <page>" for a page
    that has its own window (the garage)."""
    return GAME_NAME if not page else f"{GAME_SHORT} - {page}"


if __name__ == "__main__":
    print(f"name {GAME_NAME!r}  short {GAME_SHORT!r}  logo {LOGO_WORD!r}  "
          f"subtitle {SUBTITLE!r}  version {VERSION}")
    print(f"file name {ascii_name()!r}  save id {save_id()!r}  "
          f"garage caption {caption('garage')!r}")
