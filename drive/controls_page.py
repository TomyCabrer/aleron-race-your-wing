"""drive/controls_page.py -- the CONTROLS page: a DualSense drawn with what
every button does, and the keyboard's keys (task 37).

The owner (2026-09-24): "difficult to see what the ps controls do in
setting(s)". The pause page listed the pad in a second help column that ran
off the panel's right edge, and the settings page showed no pad at all. This
page draws the controller -- the shoulder buttons, the d-pad, the four face
buttons in their colours, the sticks, the touchpad, CREATE / OPTIONS -- with
a leader line from each control to what it does, driving and in a menu.

`draw_pad(screen, rect)` draws it into `rect` (the menu page's art hook,
drive/menu.Menu.show(art=)); `pad_rows()` / `kb_rows()` are the same
bindings as text rows. The bindings are drive/input.py's (`PS_PAD_BUTTONS`,
`MENU_HELP_PAD`): this module only lays them out. pygame is imported
lazily; the rows are pure.
"""
from __future__ import annotations

#: design space the drawing is laid out in (scaled to the page's rect)
DW, DH = 760.0, 330.0
CX, CY = 380.0, 170.0            # the controller's centre

C_BODY = (60, 64, 72)
C_BODY_EDGE = (150, 156, 166)
C_BTN = (34, 36, 42)
C_BTN_EDGE = (120, 126, 136)
C_LEAD = (110, 116, 126)
C_NAME = (217, 206, 85)          # the menu's key colour
C_TEXT = (232, 234, 238)
C_DIM = (139, 144, 153)
C_TRI = (64, 214, 160)           # the face buttons' own colours
C_CIR = (255, 102, 110)
C_CRS = (124, 178, 255)
C_SQR = (255, 143, 216)

#: (control, anchor in design space, side, label, what it does driving)
#: -- the bindings are drive/input.py's PS_PAD_BUTTONS / axes
CONTROLS = (
    ("L2", (CX - 118, CY - 104), "L", "L2", "brake"),
    ("L1", (CX - 118, CY - 86), "L", "L1", "shift down"),
    ("create", (CX - 76, CY - 58), "L", "CREATE", "reset to the sector line"),
    ("up", (CX - 104, CY - 40), "L", "d-pad UP", "HUD cycle"),
    ("left", (CX - 120, CY - 24), "L", "d-pad LEFT", "slow motion"),
    ("right", (CX - 88, CY - 24), "L", "d-pad RIGHT", "normal speed"),
    ("down", (CX - 104, CY - 8), "L", "d-pad DOWN", "force arrows"),
    ("lstick", (CX - 56, CY + 28), "L", "left stick", "steer; press: auto zoom"),
    ("R2", (CX + 118, CY - 104), "R", "R2", "throttle"),
    ("R1", (CX + 118, CY - 86), "R", "R1", "shift up"),
    ("options", (CX + 76, CY - 58), "R", "OPTIONS", "pause menu"),
    ("tri", (CX + 104, CY - 48), "R", "TRIANGLE", "wing mode (air brake, top)"),
    ("cir", (CX + 128, CY - 24), "R", "CIRCLE", "wings armed on / off"),
    ("sqr", (CX + 80, CY - 24), "R", "SQUARE", "clutch (hold)"),
    ("crs", (CX + 104, CY + 0), "R", "CROSS", "handbrake (hold)"),
    ("rstick", (CX + 56, CY + 28), "R", "right stick", "press: camera"),
    ("touch", (CX, CY - 60), "T", "touchpad", "garage (the wing designer)"),
)
#: in a menu the pad moves and picks (drive/input.py's menu mode)
MENU_PAD = (("d-pad / left stick", "move, LEFT / RIGHT change a value"),
            ("CROSS", "select"), ("CIRCLE / OPTIONS", "back"))


def pad_rows() -> list:
    """The DualSense's bindings as (control, what) rows, driving then menus."""
    return [(lbl, what) for _c, _a, _s, lbl, what in CONTROLS] + list(MENU_PAD)


def kb_rows() -> list:
    """The keyboard's, from drive/input.py (the one list the menus show)."""
    from .input import MENU_HELP_KB
    return list(MENU_HELP_KB)


_FONTS: dict = {}


def _font(size: int, bold: bool = False):
    import pygame
    key = (size, bold)
    f = _FONTS.get(key)
    if f is None:
        from .render import FONT_NAMES
        pygame.font.init()
        f = pygame.font.SysFont(",".join(FONT_NAMES), size, bold=bold)
        _FONTS[key] = f
    return f


def draw_pad(screen, rect) -> None:
    """The controller and its labels, fitted into `rect` (x, y, w, h)."""
    import pygame
    x0, y0, w, h = (float(v) for v in (rect[0], rect[1], rect[2], rect[3]))
    s = min(w / DW, h / DH)
    ox, oy = x0 + (w - DW * s) / 2.0, y0 + (h - DH * s) / 2.0

    def P(x, y):
        return (int(round(ox + x * s)), int(round(oy + y * s)))

    def R(x, y, ww, hh):
        return pygame.Rect(P(x, y), (max(1, int(round(ww * s))), max(1, int(round(hh * s)))))

    lw = max(1, int(round(1.5 * s)))
    # shoulders first (the body covers their lower edges)
    for sx in (-1, 1):
        pygame.draw.rect(screen, C_BTN, R(CX + sx * 118 - 34, CY - 112, 68, 18), border_radius=max(2, int(8 * s)))
        pygame.draw.rect(screen, C_BTN_EDGE, R(CX + sx * 118 - 34, CY - 112, 68, 18), lw, border_radius=max(2, int(8 * s)))
        pygame.draw.rect(screen, C_BTN, R(CX + sx * 118 - 38, CY - 93, 76, 12), border_radius=max(2, int(5 * s)))
        pygame.draw.rect(screen, C_BTN_EDGE, R(CX + sx * 118 - 38, CY - 93, 76, 12), lw, border_radius=max(2, int(5 * s)))
    # the body and its two grips
    for sx in (-1, 1):
        g = R(CX + sx * 112 - 52, CY - 20, 104, 150)
        pygame.draw.ellipse(screen, C_BODY, g)
        pygame.draw.ellipse(screen, C_BODY_EDGE, g, lw)
    body = R(CX - 165, CY - 84, 330, 124)
    pygame.draw.rect(screen, C_BODY, body, border_radius=max(3, int(34 * s)))
    pygame.draw.rect(screen, C_BODY_EDGE, body, lw, border_radius=max(3, int(34 * s)))
    for sx in (-1, 1):                      # repaint the grips' inner join
        pygame.draw.ellipse(screen, C_BODY, R(CX + sx * 112 - 50, CY - 16, 100, 60))
    # touchpad, CREATE / OPTIONS, the PS button
    pygame.draw.rect(screen, C_BTN, R(CX - 62, CY - 80, 124, 50), border_radius=max(2, int(9 * s)))
    pygame.draw.rect(screen, C_BTN_EDGE, R(CX - 62, CY - 80, 124, 50), lw, border_radius=max(2, int(9 * s)))
    for sx in (-1, 1):
        pygame.draw.rect(screen, C_BTN, R(CX + sx * 76 - 4, CY - 68, 8, 20), border_radius=max(1, int(4 * s)))
    pygame.draw.circle(screen, C_BTN, P(CX, CY + 14), max(2, int(8 * s)))
    pygame.draw.circle(screen, C_BTN_EDGE, P(CX, CY + 14), max(2, int(8 * s)), lw)
    # the d-pad
    for dx, dy, ww, hh in ((-7, -25, 14, 18), (-7, 7, 14, 18), (-25, -7, 18, 14), (7, -7, 18, 14)):
        rr = R(CX - 104 + dx, CY - 24 + dy, ww, hh)   # four arms, symmetric about
        pygame.draw.rect(screen, C_BTN, rr, border_radius=max(1, int(3 * s)))   # the hub
        pygame.draw.rect(screen, C_BTN_EDGE, rr, lw, border_radius=max(1, int(3 * s)))
    pygame.draw.rect(screen, C_BTN, R(CX - 104 - 7, CY - 24 - 7, 14, 14))
    # the face buttons, each with its symbol in its colour
    fr = max(3, int(round(11 * s)))
    for (bx, by), col, kind in (((CX + 104, CY - 48), C_TRI, "tri"), ((CX + 128, CY - 24), C_CIR, "cir"),
                                ((CX + 104, CY + 0), C_CRS, "crs"), ((CX + 80, CY - 24), C_SQR, "sqr")):
        c = P(bx, by)
        pygame.draw.circle(screen, C_BTN, c, fr)
        pygame.draw.circle(screen, C_BTN_EDGE, c, fr, lw)
        g = 0.52 * fr
        if kind == "tri":
            pygame.draw.polygon(screen, col, [(c[0], c[1] - g), (c[0] - g * 0.95, c[1] + g * 0.6),
                                              (c[0] + g * 0.95, c[1] + g * 0.6)], lw)
        elif kind == "cir":
            pygame.draw.circle(screen, col, c, max(2, int(g)), lw)
        elif kind == "crs":
            pygame.draw.line(screen, col, (c[0] - g * 0.8, c[1] - g * 0.8), (c[0] + g * 0.8, c[1] + g * 0.8), lw + 1)
            pygame.draw.line(screen, col, (c[0] - g * 0.8, c[1] + g * 0.8), (c[0] + g * 0.8, c[1] - g * 0.8), lw + 1)
        else:
            pygame.draw.rect(screen, col, pygame.Rect(int(c[0] - g * 0.75), int(c[1] - g * 0.75),
                                                      int(g * 1.5), int(g * 1.5)), lw)
    # the sticks
    for sx in (-1, 1):
        c = P(CX + sx * 56, CY + 28)
        pygame.draw.circle(screen, C_BTN, c, max(4, int(21 * s)))
        pygame.draw.circle(screen, C_BTN_EDGE, c, max(4, int(21 * s)), lw)
        pygame.draw.circle(screen, (48, 50, 58), c, max(3, int(13 * s)))
    # the labels -- the button's name over what it does -- and a leader
    # line from each to its control
    #  the label columns are (CX - 182) design px wide each side: the font
    #  shrinks until the widest label fits them (a small page)
    room = (CX - 182) * s - 2
    size = max(7, int(round(14 * s)))
    while size > 7:
        f_name, f_what = _font(size, bold=True), _font(size)
        if max(max(f_name.size(c[3])[0], f_what.size(c[4])[0])
               for c in CONTROLS if c[2] in "LR") <= room:
            break
        size -= 1
    f_name, f_what = _font(size, bold=True), _font(size)
    lh = f_what.get_linesize()
    lefts = [c for c in CONTROLS if c[2] == "L"]
    rights = [c for c in CONTROLS if c[2] == "R"]
    top = [c for c in CONTROLS if c[2] == "T"]
    y_first, y_last = 40.0, DH - 8.0 - 2 * lh / max(s, 1e-6)
    for group, side in ((lefts, "L"), (rights, "R")):
        n = len(group)
        for i, (_c, (ax, ay), _s, name, what) in enumerate(group):
            ly = y_first + (y_last - y_first) * i / max(n - 1, 1)
            nm = f_name.render(name, True, C_NAME)
            wt = f_what.render(what, True, C_TEXT)
            yy = P(0, ly)[1]
            if side == "L":
                xr = P(CX - 182, 0)[0]
                screen.blit(nm, (xr - nm.get_width(), yy))
                screen.blit(wt, (xr - wt.get_width(), yy + lh))
                end = (xr + 3, yy + lh // 2)
                mid = (P(CX - 172, 0)[0], end[1])
            else:
                xl = P(CX + 182, 0)[0]
                screen.blit(nm, (xl, yy))
                screen.blit(wt, (xl, yy + lh))
                end = (xl - 3, yy + lh // 2)
                mid = (P(CX + 172, 0)[0], end[1])
            a = P(ax, ay)
            pygame.draw.lines(screen, C_LEAD, False, [end, mid, a], 1)
            pygame.draw.circle(screen, C_NAME, a, max(2, int(2.5 * s)))
    for _c, (ax, ay), _s, name, what in top:
        nm = f_name.render(name, True, C_NAME)
        wt = f_what.render(what, True, C_TEXT)
        tw = nm.get_width() + 8 + wt.get_width()
        tx = P(CX, 0)[0] - tw // 2
        ty = P(0, 4)[1]                     # above the shoulders, clear of both columns
        screen.blit(nm, (tx, ty))
        screen.blit(wt, (tx + nm.get_width() + 8, ty))
        a = P(ax, ay)
        pygame.draw.line(screen, C_LEAD, (P(CX, 0)[0], ty + lh), a, 1)
        pygame.draw.circle(screen, C_NAME, a, max(2, int(2.5 * s)))


#: what each bound command must be called on the drawing: the self-check
#: holds the labels to drive/input.py's PS_PAD_BUTTONS (a rebinding fails it)
CMD_WORDS = {"handbrake": "handbrake", "wing": "wings armed", "clutch": "clutch",
             "wing_side": "wing mode", "reset": "reset", "menu": "menu",
             "zoom_auto": "auto zoom", "camera": "camera", "shift_down": "shift down",
             "shift_up": "shift up", "hud": "HUD", "vectors": "force arrows", "slowmo": "slow",
             "normal_speed": "normal speed", "garage": "garage"}
#: drive/input.py's PS button names -> the control on the drawing
BUTTON_CONTROL = {"cross": "crs", "circle": "cir", "square": "sqr", "triangle": "tri",
                  "create": "create", "options": "options", "l3": "lstick", "r3": "rstick",
                  "l1": "L1", "r1": "R1", "up": "up", "down": "down", "left": "left",
                  "right": "right", "touchpad": "touch"}


# ==================================================================== #
#  SELF-CHECK                                                          #
# ==================================================================== #
def self_check(verbose: bool = True) -> bool:
    import os
    ok = True

    def rep(tag, passed, msg=""):
        nonlocal ok
        ok = ok and bool(passed)
        if verbose:
            print(f"  [{'ok' if passed else 'FAIL'}] {tag}" + (f": {msg}" if msg else ""))

    from .input import PS_PAD_BUTTONS, PS_BUTTON_NAMES, MENU_HELP_KB
    ids = {c[0] for c in CONTROLS}
    missing = [PS_BUTTON_NAMES[i] for i in PS_PAD_BUTTONS
               if BUTTON_CONTROL.get(PS_BUTTON_NAMES.get(i, "")) not in ids]
    rep("every bound DualSense button is on the drawing, and the triggers",
        not missing and {"L2", "R2"} <= ids, f"missing {missing}")
    what = {c[0]: c[4] for c in CONTROLS}
    wrong = [f"{PS_BUTTON_NAMES[i]}={cmd}" for i, cmd in PS_PAD_BUTTONS.items()
             if CMD_WORDS.get(cmd, "?") not in what.get(BUTTON_CONTROL.get(PS_BUTTON_NAMES.get(i)), "")]
    rep("each button's label says what input.py binds it to", not wrong, f"wrong {wrong}")
    rep("the rows: the pad's (driving, then a menu) and the keyboard's",
        len(pad_rows()) == len(CONTROLS) + len(MENU_PAD) and kb_rows() == list(MENU_HELP_KB))
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import numpy as np
    import pygame
    pygame.init()
    inside, lit = True, []
    for rect in ((20, 20, 776, 330), (10, 10, 520, 225), (30, 40, 1100, 480)):
        sf = pygame.Surface((1200, 600))
        sf.fill((0, 0, 255))
        draw_pad(sf, rect)
        a = pygame.surfarray.pixels3d(sf)
        m = ~((a[..., 0] == 0) & (a[..., 1] == 0) & (a[..., 2] == 255))
        xs, ys = np.nonzero(m)
        del a
        lit.append(int(m.sum()))
        inside = inside and len(xs) > 0 and (xs.min() >= rect[0] and xs.max() < rect[0] + rect[2]
                                             and ys.min() >= rect[1] and ys.max() < rect[1] + rect[3])
    rep("the drawing stays inside its rect, at three sizes", inside, f"lit px {lit}")
    if verbose:
        print(f"controls_page self-check: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_check() else 1)
