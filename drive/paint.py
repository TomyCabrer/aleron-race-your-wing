"""drive/paint.py -- the player's car paint: the palette the Settings page offers.

The paint is COSMETIC and nothing else. It is chosen per car on the Settings
page and saved with the settings, and it never reaches the class key, a
ranking, a medal, a ghost, the CarSpec or the garage's CarBuild JSON (a lap
record's settings snapshot lists it, as it lists Graphics; nothing reads it
back): a red Corsa and a yellow one set the same times. `PAINT_DEFAULT` ('factory') draws
each car in its own colour (render.C_CAR_STYLE: the Corsa yellow, the rally
Escort red, the 540i blue, the Express white), so a session that never picks one -- and every headless
or scripted run -- draws exactly what it drew before paint existed.

Why a fixed palette and not an RGB picker: the self-checks COUNT exact
colours on screen (the delta's green and red, the flash purple, the PB
ghost's green, the tarmac tones, the HUD's text in the strip right of the
tutorial box), and the plan view draws the paint at four tones (0.72, 0.80,
1.0 and 1.10 of it; the chase view's sills are 0.58). A naive white
(232, 234, 238) IS the HUD text colour and a naive silver (174, 180, 191) has
the HUD's dim grey as its 0.80 tone: both were measured putting 53 and 12
stray pixels into the tutorial check. Every colour here was checked at every
tone against those, and kept clear of the race slots' colours
(race_grid.COLOURS), the PB ghost's green and the wing device's orange --
the paints a player would lose a bot or the wing against. That is why there
is no orange, sky blue, violet, rose, lime or mint, and no mid grey (the
stowed wing), light grey (the reference ghost), tarmac grey or gravel beige.
drive/render.py's self-check re-measures the tones on the real renderer
(its 'paint: ...' row); this module's self-check covers the palette itself.

Pure data, no pygame: drive.drive reads the names for its Settings row and
hands `rgb` to the row's swatch (drive.menu); drive.render and drive.garage
take the resolved RGB.
"""
from __future__ import annotations

#: the paint every car starts in: its own factory colour (render.C_CAR_STYLE)
PAINT_DEFAULT = "factory"

#: name -> RGB, in the Settings page's cycling order; 'factory' is None
#: because its colour depends on the car (render.factory_colour). Names are
#: short lowercase keys: they are what runs/settings.json stores.
PAINTS = {
    "factory": None,
    "yellow": (215, 195, 74),      # the Corsa's own yellow, for the other two
    "red": (176, 34, 42),          # the rally Escort's own red (task 46; the MX-5's before)
    "blue": (64, 92, 138),         # the 540i's own blue
    "white": (226, 226, 220),      # warm: a cool (232, 234, 238) is the HUD text
    "silver": (170, 174, 182),     # (174, 180, 191)'s 0.80 tone is the HUD's dim grey
    "green": (30, 92, 56),
    "purple": (74, 44, 112),       # dark: the flash's (176, 78, 224) stays unmistakable
    "cobalt": (40, 72, 186),
    "teal": (22, 128, 132),
    "burgundy": (112, 26, 44),
}

#: the cycling order (LEFT / RIGHT on the Paint row): factory first
PAINT_ORDER = tuple(PAINTS)

#: what the Settings row shows for each paint
PAINT_LABELS = {
    "factory": "factory",
    "yellow": "signal yellow",
    "red": "rosso red",
    "blue": "estoril blue",
    "white": "arctic white",
    "silver": "silver",
    "green": "racing green",
    "purple": "midnight purple",
    "cobalt": "cobalt blue",
    "teal": "teal",
    "burgundy": "burgundy",
}


def rgb(name) -> tuple | None:
    """The paint's RGB, or None for 'factory' and for any name this palette
    does not have (a hand-edited settings file draws the car's own colour
    rather than failing). The caller resolves None to the car's factory
    colour where it needs a concrete one (render.factory_colour, which takes
    the car's spec or its key)."""
    c = PAINTS.get(str(name)) if name is not None else None
    return tuple(c) if c is not None else None


# ==================================================================== #
#  SELF-CHECK                                                          #
# ==================================================================== #
def self_check(verbose: bool = True) -> bool:
    import math
    import os
    import subprocess
    import sys
    ok = True

    def rep(tag, passed, msg=""):
        nonlocal ok
        ok = ok and bool(passed)
        if verbose:
            print(f"  [{'ok' if passed else 'FAIL'}] {tag}" + (f": {msg}" if msg else ""))

    # in a fresh interpreter: in this one a caller may have loaded pygame already
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run([sys.executable, "-c", "import sys, drive.paint; "
                        "sys.exit(3 if 'pygame' in sys.modules else 0)"],
                       cwd=root, capture_output=True, text=True)
    rep("pure data: importing the palette pulls in no pygame", r.returncode == 0,
        f"fresh interpreter exit {r.returncode}")
    cols = [c for c in PAINTS.values() if c is not None]
    rep("factory first, the default, the only paint without a colour",
        PAINT_ORDER[0] == PAINT_DEFAULT == "factory" and PAINTS[PAINT_DEFAULT] is None
        and len(cols) == len(PAINTS) - 1, f"{len(PAINT_ORDER)} paints")
    rep("every colour is three ints in 0..255",
        all(isinstance(c, tuple) and len(c) == 3
            and all(isinstance(v, int) and 0 <= v <= 255 for v in c) for c in cols))
    rep("names are short lowercase keys a settings file can hold (no '_' ',' '|' or space)",
        all(n.isascii() and n.isalpha() and n.islower() and len(n) <= 12 for n in PAINT_ORDER),
        ", ".join(PAINT_ORDER))
    rep("the order is the palette, every name once", len(set(PAINT_ORDER)) == len(PAINT_ORDER)
        and set(PAINT_ORDER) == set(PAINTS))
    rep("a label per paint, unique, short enough for the Settings row",
        set(PAINT_LABELS) == set(PAINTS)
        and len(set(PAINT_LABELS.values())) == len(PAINT_LABELS)
        and all(0 < len(v) <= 16 for v in PAINT_LABELS.values()),
        f"longest {max(PAINT_LABELS.values(), key=len)!r}")
    near = min((math.dist(a, b), a, b) for i, a in enumerate(cols) for b in cols[i + 1:])
    rep("every colour its own: no two paints within 30 RGB units",
        len(set(cols)) == len(cols) and near[0] >= 30.0,
        f"closest pair {near[1]} / {near[2]}: {near[0]:.1f}")
    rep("rgb(): the colour, None for factory and for an unknown name",
        rgb("cobalt") == (40, 72, 186) and rgb("factory") is None and rgb("chartreuse") is None
        and rgb(None) is None and rgb(7) is None)
    # the colours a paint must not be mistaken for: a race slot (a bot), the
    # PB ghost, the wing device's orange (it must stand out from the body)
    # and, looser, the reference ghost's light grey (translucent and
    # labelled, so arctic white may sit 30 units from it)
    from . import race_grid, ghosts
    avoid = [(c, f"race slot {race_grid.COLOUR_NAMES[i]}", 40.0)
             for i, c in enumerate(race_grid.COLOURS)]
    avoid += [(ghosts.C_GHOST_PB, "PB ghost", 40.0), ((255, 140, 43), "wing orange", 40.0),
              (ghosts.C_GHOST_2, "REF ghost", 25.0)]
    worst = min((math.dist(p, c) - lim, n, what, math.dist(p, c))
                for n in PAINT_ORDER[1:] for p in (PAINTS[n],) for c, what, lim in avoid)
    rep("no paint mistaken for a bot, a ghost or the wing (>= 40 units; the REF ghost >= 25)",
        worst[0] >= 0.0, f"tightest: {worst[1]} vs the {worst[2]}, {worst[3]:.1f} units")
    if verbose:
        print(f"paint self-check: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_check() else 1)
