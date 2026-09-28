# Third-party notices

What a packaged build of **Alerón: Race Your Wing** (name: `drive/branding.py`)
ships besides its own code, under which terms, and what those terms ask of us.
Drafted 2026-09-28 from the installed packages on the build Mac (pygame 2.5.2
/ SDL 2.28.3, numpy 2.4.4, scipy 1.17.1, pymoo 0.6.2); the packaging lane's
build venv may carry newer versions -- re-check versions against
`packaging/` before release.

**This file is not legal advice.** Items marked **OWNER DECISION** have terms
that are unclear or that impose real obligations; decide them before the
first public build.

How to ship it: copy this file, plus the licence texts listed below, into the
build (e.g. `licenses/` next to the executable / inside `Contents/Resources`)
and add a "Licenses" line to the store page's legal section or the in-game
credits. PyInstaller's one-FOLDER build (what `packaging/game.spec` makes)
keeps every dynamic library as a separate, replaceable file, which is what
the LGPL items below need; do not switch to a one-file build.

---

## 1. Runtime and libraries

| component | licence | obligations |
|---|---|---|
| CPython 3.11 (interpreter + stdlib) | PSF License 2.0 | include the licence text; its bundled libs: OpenSSL (Apache-2.0), libffi (MIT), expat (MIT), zlib, bzip2, xz (public domain / 0BSD), mpdecimal (BSD-2), SQLite (public domain) |
| PyInstaller bootloader | GPL-2.0 **with the bootloader exception** | none for the game: the exception lets the bundled app use any licence |
| pygame 2.x | **LGPL-2.1** | see 2 below |
| SDL2, SDL2_image, SDL2_mixer, SDL2_ttf | zlib | include the notice |
| pygame's bundled codecs/fonts libs (macOS wheel, `pygame/.dylibs`) | libpng (libpng), zlib-ng (zlib), libjpeg (IJG), libtiff (libtiff/BSD-like), libwebp + sharpyuv (BSD-3), FreeType (FTL, chosen over GPL-2), brotli (MIT), ogg / vorbis / opus / opusfile / FLAC (BSD-3), libmodplug (public domain), portmidi (MIT-like) | include each notice |
| same, LGPL ones: libfluidsynth, glib + gthread, libintl (gettext runtime), libmpg123, libsndfile | **LGPL-2.1+** | as pygame (2 below). The game uses none of them (sound is synthesised, no MIDI/MP3/fonts via glib) -- the packaging lane may be able to drop them; if they ship, they carry the LGPL duties |
| numpy | BSD-3 (+ 0BSD, MIT, Zlib, CC0 parts) | include the notice. Windows/Linux wheels bundle **OpenBLAS** (BSD-3) and the gfortran runtime (GPL-3 **with the GCC Runtime Library Exception**: no obligation beyond notice); macOS arm64 uses Apple's Accelerate |
| scipy | BSD-3 | include the notice; bundles libgfortran / libgcc_s (GCC RLE) and **libquadmath (LGPL-2.1)** on macOS, OpenBLAS on Windows/Linux |
| pymoo 0.6 | Apache-2.0 | include the licence + any NOTICE file |
| pymoo's deps that ship if imported: autograd (MIT), cma (BSD-3), alive-progress (MIT), about-time (MIT), Deprecated (MIT), wrapt (BSD-2), dill (BSD-3) | as listed | include notices |
| moocore (pymoo dep) | **LGPL-2.1+** | as pygame (2 below); if pymoo does not need it at runtime the packaging lane can exclude it |
| matplotlib (pymoo dep, if not excluded) + pillow (MIT-CMU), kiwisolver (BSD), fonttools (MIT), contourpy (BSD-3), cycler (BSD), pyparsing (MIT), python-dateutil (BSD/Apache), packaging (BSD/Apache); matplotlib's fonts: DejaVu (Bitstream Vera licence), STIX (OFL-1.1) | as listed | include notices. Better: exclude matplotlib from the build if nothing on the player's path imports it |

Not shipped, on purpose: **torch / botorch / gpytorch** (BSD-3 / MIT; excluded
for size -- the designer falls back to AeroBO's optimisers that need none),
**XFOIL** (GPL-2: the game only calls it if the player has it installed;
never bundle it), OpenVSP.

## 2. What the LGPL-2.1 items require (pygame and friends)

We distribute the LGPL libraries themselves, unmodified, as object code next
to our program, which only links to them dynamically. That is allowed
(LGPL-2.1 section 6) provided we:

1. **Say so and ship the licence**: a notice that the game uses pygame (and
   the other LGPL libs), that they are under the LGPL-2.1, and the full
   LGPL-2.1 text.
2. **Keep them replaceable**: the player must be able to swap in a modified
   version. A one-folder build with the `.dylib` / `.dll` / `.so` files and
   pygame's modules as separate files does this; one-file builds and static
   linking do not.
3. **Make the source available**: either ship the complete corresponding
   source of the exact versions we distribute, or include a **written offer,
   valid at least three years**, to provide it (section 4/6). The simplest
   route: archive the source tarballs of the exact pygame / SDL_mixer-deps /
   moocore / libquadmath versions in the build, host them (e.g. a GitHub
   release), and put the link + offer in the licences folder.
4. **Not forbid reverse engineering** for debugging such modifications in the
   game's EULA (Steam's default subscriber agreement is fine; a custom EULA
   must allow it).

**OWNER DECISION**: approve the written-offer route (item 3) and who hosts the
source archive. Everything else above is mechanical.

## 3. Bundled data and assets

| asset | where | terms | status |
|---|---|---|---|
| AeroBO wing-design engine | `aerobo/` | MIT, (c) 2026 Bartolome Cabrer -- the owner's own (`aerobo/LICENSE`) | fine; include the licence |
| **UIUC airfoil coordinates** (2174 `.dat` files) | `aerobo/data/airfoils/uiuc/` | The UIUC Airfoil Data Site distributes the data under the **GNU GPL** with conditions from the UIUC Low-Speed Airfoil Tests program: no extra charge for the data itself, products using it must state conspicuously that it comes from the UIUC LSAT program, recipients may copy and redistribute the data freely, and the licence + copyright notice + the LSAT "manifesto" must accompany each distribution (per [UIUC Airfoil Data Site](https://m-selig.ae.illinois.edu/ads.html) / [coordinate database](https://m-selig.ae.illinois.edu/ads/coord_database.html); the pages fetched today no longer show the text, so confirm with the site owner) | **OWNER DECISION**: (a) ship the `.dat` files as plain, separately readable data with the GPL text + UIUC attribution + manifesto in the licences folder, charging nothing for the data (arguably mere aggregation; most commercial tools that bundle it do this); or (b) confirm terms by e-mail with m-selig@illinois.edu; or (c) ship only AeroBO's derived library/cache and not the raw UIUC coordinates, if the designer can run without them |
| Material Icons | `drive/data/fonts/MaterialIcons-Regular.ttf` | Apache-2.0, Google (`LICENSE-MaterialIcons.txt`) | fine; ship the licence |
| **Tyre model: `TNO_car205_60R15.tir`** | `tyre_data/` (read at runtime by `drive/tyre.py`) | An MF-Tyre/MF-Swift 6.2 property file written by **MF-Tool 6.2 (TNO / Delft-Tyre, now Siemens)**, "Manufacturer Delft-Tyre", dated 7 Aug 2013 -- the example file that ships with MFeval / MF-Tool. **No licence in the file**; its redistribution terms are unknown | **OWNER DECISION -- release blocker until settled**: ask Siemens/Delft-Tyre, or replace it with a tyre whose parameters you own (fit your own MF6.2 set, or derive one from a clearly-licensed source), then re-validate (drive/validate.py pins the golden points of this file) |
| other `.tir` files in `tyre_data/` (Pac2002 Audi/Sedan, Siemens, TASS, Pacejka book defaults, generic `mf_185_80R14` ...) | `tyre_data/` | mixed vendor examples, no licences | the game does not read them: **do not ship them** (packaging lane: ship only `TNO_car205_60R15.tir`, or none after the item above) |
| `tyre_data/mf_eval.py` | `tyre_data/` | unknown origin (dev tool) | not needed at runtime: do not ship |
| Sound | -- | synthesised in code (`drive/audio.py`); no recorded samples | fine |
| Images / logo | -- | drawn in code (logo: `drive/title.py`) | fine (see the AI-disclosure note in `steam/store/text.md`) |
| Car shapes, cockpit/controller drawings | `drive/bodies.py`, `drive/render.py`, `drive/controls_page.py` | our own drawings, but modelled on real cars and on Sony's DualSense | see 4 |

## 4. Trademarks (real makes, models and products the player sees)

Real brand and model names are trademarks; using them in a sold game (and in
the store page) without a licence is a legal risk -- car makers do license
and do enforce. Recognisable body shapes can also count (trade dress).

**Done 2026-09-28: the cars are fictional on screen.** The keys stay
(`corsa` / `rally` / `540i` / `express`: settings, saves, records, medals,
bot file names); only what the player reads changed:

| key | was | shown now (`cars.CAR_TITLES`) | full name (`CarSpec.name`) | short tag (`prerace.CAR_SHORT`) |
|---|---|---|---|---|
| corsa | Opel Corsa C 1.2 16V | Aurel Civetta 1.2 | Aurel Civetta 1.2 16V (2003) | Civetta |
| rally | Ford Escort RS1800 Mk2 | Halcón RS18 rally | Halcón RS18 (Group 4 tarmac, 1979) | Halcón |
| 540i | BMW 540i E39 | Nordwerk N540 | Nordwerk N540 (1998) | N540 |
| express | Renault Express 1.4 | Rivière Courier 1.4 | Rivière Courier 1.4 (1995) | Courier |

Also routed through the shown names: bot file stems on the RACE page, ghost
labels and leaderboards (`540i_arena_plate` shows as `n540_arena_plate`, the
retired MX-5's as `roadster_...`), default build names (`my civetta`), class
labels, swarm names. A car's display name is not a medal input
(`cars.DISPLAY_FIELDS`), so the medal table stayed current. A sweep of every
string the game rendered (91k strings: a scripted session in every car over
68 pages, plus the page-drawing self-checks) found no real make, model or
code on screen. `cars.py`'s self-check now fails if a shown name is real.

Left, not on screen: comments, docstrings and self-check labels; the
`source=` provenance strings in `cars.py`; `corsa_c.py` (the study);
`README.md` and `.handoff/` (not shipped); the retired MX-5 / Citaro specs
(never shown).

Still to weigh (OWNER): the body SHAPES follow the real cars' proportions
(`drive/bodies.py:61-96`: Corsa C hatch, E39 saloon, a Renault-5 nose on the
van, the Mk2's three-box) -- low-poly, no badges, grilles, logos or scripts
drawn, plates blank. Softening them is optional.

Controller: "DualSense" / "PS5" button names and a drawn DualSense
(`drive/controls_page.py`, `drive/input.py`) -- naming the pad a player owns
is normal practice (nominative use); no Sony logo or wordmark styling.

## 5. Our own code

Everything else in the build (`drive/`, `cars.py`, `corsa_c.py`, `qss.py`,
`ledger.py`, `crossover.py`, `aerobo/` as the owner's own) is the developer's.
