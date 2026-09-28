# Store page texts -- drafts

Every text below uses the current name from `drive/branding.py`
(`GAME_NAME` = "Alerón: Race Your Wing", `GAME_SHORT` = "Alerón"). After a
rename, search this file for "Alerón" and replace it; nothing else here
depends on the name. The cars are fictional since 2026-09-28 (Aurel
Civetta 1.2, Halcón RS18 rally, Nordwerk N540, Rivière Courier 1.4 --
`cars.CAR_TITLES`); the texts below may name them, but never a real make.

---

## Short description (Steam limit: 300 characters)

> Design the wings, then drive them. A physics-first time-attack racer where
> every lap is decided in the garage: shape side and top wings with a real
> aerodynamic optimiser, bolt them onto your car, chase medals and ghosts on
> five circuits, and breed bots to beat you.

(265 characters. A shorter one for tight spots, 125 characters:
"Design real wings, bolt them on, and chase the lap. A physics-first
time-attack racer where the garage decides the stopwatch.")

---

## About This Game (~250 words)

**Alerón: Race Your Wing** is a time-attack racer about one idea: the wing.
Not a cosmetic spoiler -- movable aerodynamic devices that stand up on the
flanks of the car and above the roof, deploy under braking, lean into corners
and push the tyres into the road. You design them, and you drive them.

In the **garage** you shape each wing in 3-D: span, chord, section, twist and
how it is carried on the body. A real aerodynamic optimiser searches the
section library and the design space for you, inside each car's limits.
Save your wings, save whole cars, and take them to the track.

On the **track** the physics does the talking: a full tyre model, load
transfer, a real gearbox and clutch, wet patches that change the grip mid-lap.
Five circuits and an open proving ground, four cars with their own engines,
weights and driving aids. Deploy the wings as an air brake into a hairpin,
fix the top wing for the fast sweepers, or run clean for the straights.

Then chase the clock: **medals** on every map and car, your **top five laps**
per class with a **ghost** and a live delta, **eight challenges** about
stopping and grip, and **leaderboards** for every map, car and wing mode.
Or hand the car over: breed a **swarm of bots** that learns to drive your
build, race them, and see who is faster.

Features
- A wing designer built on a real optimisation engine, not a paint shop
- Side wings, a top wing, an air-brake mode, and per-car span limits
- Five circuits + an open proving ground and a skidpad, with wet patches
- Four cars: a small hatchback, a 1970s rally car, a big rear-drive saloon, a van
- Medals, ghosts, personal records, challenges and leaderboards
- Trainable bots: breed a swarm, race the best of it
- Driving and wing-design tutorials
- Keyboard or controller (DualSense and other SDL pads), manual, clutch or auto gearbox

---

## Tags (Steam lets you pick up to 20; the first ones weigh most)

Suggested set, in order: **Racing, Free to Play, Automobile Sim, Driving,
Simulation, Time Attack, Physics, Building, Realistic, Singleplayer, Indie,
Sandbox, Education, Controller**.

## Genre (store page "Genre" field)

Racing, Simulation, Indie.

---

## System requirements (TODO: confirm with a clean-machine test of each build)

Numbers are estimates until the packaged builds are measured: the game is
Python + pygame (software 3-D in SDL), single-threaded while driving; the
wing designer and the bot swarm use more cores and memory.

**Windows** (64-bit only)
- Minimum: Windows 10 64-bit; dual-core 2.5 GHz (Intel i3-6100 / Ryzen 3 1200);
  4 GB RAM; any GPU with a 1280x720 display; 500 MB disk (the macOS arm64
  build is 150 MB, plus ~25 MB of AeroBO data copied to the save folder on first run)
- Recommended: Windows 10/11 64-bit; quad-core 3.5 GHz (i5-8600 / Ryzen 5 3600);
  8 GB RAM; 1920x1080 display; 500 MB disk

**macOS**
- Minimum: macOS 11 Big Sur (the build's LSMinimumSystemVersion); Apple
  Silicon (M1) -- or any Mac if the Intel build ships (see
  steam/README.md section 1, "One macOS depot, two CPUs"); 4 GB RAM; 500 MB disk
- Recommended: Apple M1 Pro / M2 or better; 8 GB RAM

**Linux / SteamOS**
- Minimum: Ubuntu 22.04 64-bit or SteamOS 3; same CPU/RAM as Windows
  minimum; 500 MB disk. TODO: only list Linux if a Linux build is uploaded;
  otherwise leave Linux unticked and let the Deck run the Windows build
  through Proton.

Controller: "Partial Controller Support" until every page (garage included)
is checked with a pad only; "Full" if it is.

---

## Content survey answers (Steamworks > Store Presence > Content Survey)

- Frequent violence or gore: **No**
- Sexual content / nudity: **No**
- Mature / adult-only content: **No**
- Generally mature content (drugs, alcohol, gambling, strong language): **No**
- Real-money purchases / loot boxes / in-game purchases: **No**
- User-generated content shared with others: **No** (builds and wings are
  saved locally; nothing is uploaded)
- Online interaction / chat: **No** (leaderboards are local today)
- Data collection: **No** (no telemetry leaves the machine)
- Generative AI section (mandatory): **OWNER DECISION.** Since Valve's
  16 Jan 2026 clarification, AI coding assistants and other efficiency tools
  used only during development are exempt; disclosure covers AI-made content
  that "ships with your game, and is consumed by players" or is used in
  marketing (art, sound, voice, writing). Live-generated: **No** (nothing is
  generated at runtime). Pre-generated: answer **Yes** if any shipped
  player-facing text, store copy (these drafts were AI-drafted) or art/logo
  was AI-generated and kept; describe it honestly. Source:
  [PC Gamer](https://www.pcgamer.com/software/ai/steam-updates-ai-disclosure-form-to-specify-that-its-focused-on-ai-generated-content-that-is-consumed-by-players-not-efficiency-tools-used-behind-the-scenes/).

## Price: FREE (owner, 2026-09-28)

Created as a free product in Steamworks ("This is a free product"): no price,
no regional pricing, no discounts. Say it plainly on the page:
"100% free -- no microtransactions, no ads, no DLC." Genre: add **Free To
Play**. Free apps get no trading cards (Valve allows them only with in-app
purchases). See RELEASE_CHECKLIST.md section A.
