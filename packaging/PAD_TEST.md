# Gamepad test before release (Windows PC + Steam Deck)

No real pad has run through the generic path on this Mac (`pygame.joystick.get_count() == 0`).
The DualSense path is unchanged and was checked on hardware earlier. Everything below covers
the **generic path**: Xbox pads, Steam Input's virtual pad, the Deck, 8BitDo and the like.

## What changed (2026-09-28, Steam prep)

- If a pad is not a Sony pad and SDL knows it, `drive/input.py` now opens it through
  **SDL's GameController API** (`pygame._sdl2.controller`, wrapped by `_ControllerPad`).
  - SDL numbers the buttons the same way on every OS: A B X Y BACK GUIDE START L3 R3 LB RB, then the d-pad.
  - The triggers are separate axes: LT 4, RT 5.
  - Raw joystick numbers used to differ by driver. Windows XInput reports `A B X Y LB RB BACK START L3 R3 GUIDE` with the triggers on axes 2/5. Linux xpad puts GUIDE at 8.
- Under Steam, the pad the game sees is Steam Input's virtual pad. Steam hands SDL its mapping (`SDL_GAMECONTROLLERCONFIG`), so this path is the one that works on the Deck too.
- The generic layout now sits where the DualSense layout sits. **GUIDE is not used**, because Steam and Windows Game Bar take it.

| pad | does |
|---|---|
| RT / LT | throttle / brake |
| left stick | steer |
| RB / LB | shift up / down |
| A / X | handbrake / clutch (hold) |
| B | wings armed on / off |
| BACK (View) | wing mode: auto, air brake, top ... |
| Y | back to the sector line (tap); restart the lap (hold 0.8 s) |
| d-pad up / down | HUD / force arrows |
| d-pad left / right | slow-mo / normal |
| R3 / L3 | camera / auto zoom |
| START (Menu) | pause menu / settings |
| menus | d-pad or stick to move, A select, B back, START closes, Y reset |

- The garage and WingLab ask for DualSense names ("cross", "triangle", "l1", "options"). On an Xbox pad those now read the button in the same place: cross = A, triangle = Y, L1 = LB, options = START.

## Quick probe (from source, any OS with the pad plugged in)

```
python3 - <<'EOF'
import time, pygame
pygame.init(); pygame.joystick.init()
from drive.input import GamepadInput
p = GamepadInput(0, steer_limit=False, user_config=False)
print(p.name, "| layout", p.layout, "| via", type(p.joy).__name__,
      "| throttle axis", p.map["throttle"], "brake axis", p.map["brake"])
t0 = time.time()
while time.time() - t0 < 20:
    pygame.event.pump()
    down = [p.button_name(i) for i in range(21) if p._button(i)]
    print(f"\rRT {p._trigger('throttle'):.2f} LT {p._trigger('brake'):.2f} "
          f"L {p.stick('left')} R {p.stick('right')} {down}      ", end="")
    time.sleep(0.05)
print()
EOF
```

- **Expect:** `layout generic`, `via _ControllerPad`, throttle axis 5, brake axis 4.
- **Expect:** both triggers read 0.00 at rest and 1.00 fully pressed.
- **Expect:** each button prints its own name. `lb` and `rb` are the bumpers; `back` and `start` are View and Menu.
- **If you see `via Joystick`:** SDL did not recognise the pad. Write the right numbers in `~/.carsim_pad.json` (the old fallback), or add the pad to SDL's database.

## Windows PC checklist (packaged build, `dist\<Name>\<Name>.exe`)

1. Xbox pad (USB or wireless), game started **outside** Steam:
   - [ ] Title screen: d-pad moves, A selects.
   - [ ] RT accelerates, LT brakes, left stick steers.
   - [ ] RB / LB shift in Manual.
   - [ ] START opens the pause menu, B goes back.
   - [ ] Hold Y restarts the lap.
2. The same pad, game started **from Steam** (Steam Input on):
   - [ ] Everything in 1 again.
   - [ ] The Xbox button opens the Steam overlay and does nothing in the game.
3. A DualSense through Steam:
   - [ ] Steam Input may turn it into an Xbox-style virtual pad. The game then shows the GAMEPAD help, not PS5. That is fine as long as the buttons work.
4. Unplug the pad while driving:
   - [ ] The game pauses. Plug it back in and it is picked up again.
5. Rumble:
   - [ ] Kerbs and a spin buzz the pad. You can switch it off with the env var `CARSIM_NO_RUMBLE=1`.

## Steam Deck checklist (Linux depot, Game Mode)

1. [ ] The game fills the 1280x800 screen, and the built-in controls work as in Windows step 1. The Deck appears as Steam's virtual Xbox-style pad.
2. [ ] The Steam button and `...` open Steam's menus, not the game's.
3. [ ] The trackpads and back grips do nothing unless you map them in Steam Input. Steam's default gamepad template is fine.
4. [ ] Suspend and resume in the middle of a lap; the pad still works afterwards.
5. [ ] On-screen text is readable at 7" (Steam's Deck review checks this).

## Known gaps

- The CONTROLS page (`drive/controls_page.py`) always draws a DualSense, even with an Xbox pad attached. The pause menu's text help does show the GAMEPAD rows.
- `python3 -m drive.drive --pad-calib` still reads the raw joystick, so its numbers are raw driver numbers, not the ones the game uses on the GameController path.
