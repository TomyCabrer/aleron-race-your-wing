# Release checklist -- getting the game onto Steam

Drafted 2026-09-28 (Steam prep). Ordered: do the steps top to bottom; the
long waits (verification, the 30-day fee wait, 2 weeks of "Coming Soon",
reviews) run in parallel with the finishing work, so **start step A now**.

The technical kit lives in `steam/` (upload, depots, Steam Cloud) and
`packaging/` (the builds); licences in `THIRD_PARTY_NOTICES.md`; store texts in
`steam/store/text.md`.

Fastest realistic timeline from today, if nothing bounces:
**~4-5 weeks** (account already exists, game free) = app credit -> 30-day
wait, during which the store page goes up and sits in "Coming Soon" >= 2
weeks, the store page review (3-5 business days) and the build review (3-5
business days) happen -> release button. Earliest ~2026-10-28 if the credit
is bought on 2026-09-28.

---

## A. Account and money (owner only)

Decided 2026-09-28: **you already have a Steamworks account, and the game is
FREE** (no in-app purchases, no ads, no DLC). Checked against Valve's docs
that day:

1. **The USD 100 app credit still applies to a free game** -- one per app,
   no exemption. It is paid back only after USD 1,000 revenue, so for a free
   game it is effectively never recovered
   ([app fee](https://partner.steamgames.com/doc/gettingstarted/appfee)).
   First look in Steamworks for an **unused credit** on your account. Only
   an Admin can buy one, the person who paid activates it, Steam Wallet
   funds are not accepted.
2. **Waiting period**: counted from the day THAT app's fee was paid; Valve
   applies it to a partner's "first few titles". Its pages say 30 days
   ([Steam Direct](https://partner.steamgames.com/steamdirect)) and 21 days
   ([onboarding](https://partner.steamgames.com/doc/gettingstarted/onboarding))
   -- plan for 30. Buy the credit today and the earliest release is ~2026-10-28.
3. **Create the app with "This is a free product" ticked** (Options). Steam
   makes a hidden free package; there is no price or regional pricing to set
   ([free to play](https://partner.steamgames.com/doc/store/freetoplay)).
   Treat free as permanent: Valve can switch free <-> paid only through its
   support form with a week's notice
   ([pricing](https://partner.steamgames.com/doc/store/pricing)), and
   developers report reviews being wiped by a switch.
4. You get the **App ID**. Create the depots (Windows, macOS, and Linux if you
   ship one) and a builder account with only upload rights
   (`steam/README.md`). **You can upload builds from this moment**; only the
   public release waits for the reviews and the Coming Soon period.

Free-game differences: no trading cards (free apps only get them with
in-app purchases); no deals / discounts; achievements, wishlists, Steam
Cloud, the Free to Play hub, Next Fest (with a demo) and Steam Playtest all
work. Same reviews and content survey as a paid game.

Timeline: **Path A** (release this autumn): credit -> free app -> store page
submitted (review 3-5 business days; submit >= 7 business days ahead) ->
Coming Soon >= 2 weeks -> build review (plan 7 business days) -> release
>= 30 days after the fee, i.e. late October / early November 2026. **Path B**:
stay unreleased with a free demo for Steam Next Fest Feb 22 - Mar 1 2027
(register by Jan 10), then launch, e.g. in the Racing Fest (Apr 12-19 2027)
([Next Fest](https://partner.steamgames.com/doc/marketing/upcoming_events/nextfest)).
A public Coming Soon page is a link you can show employers within days either way.

## B. Rename the game (before any store page goes public)

The name lives in ONE file:

1. Edit `drive/branding.py`: `GAME_NAME`, `GAME_SHORT`, `LOGO_WORD`,
   `SUBTITLE`. (The title logo and the line under it are the owner's art,
   `drive/data/art/logo.png` and `subtitle.png`: a new word or line is set
   plainly in the menu's font until the art is redrawn -- look at the title
   screen.)
2. **Pin the save folder**: set `SAVE_ID_PINNED` to the value `save_id()`
   returns for the final name (`python3 -m drive.branding` prints it), before
   the first public build. After that, never change it (players' saves live
   in `<AppData|Application Support>/<save id>/`).
3. Re-run the store art: `python3 steam/store/make_store_assets.py` (the
   capsules and the library logo are drawn from the name).
4. Rebuild every platform (`packaging/build_*`), then `steam/collect.sh`.
5. Search `steam/store/text.md` for the old name.
6. Steamworks: enter the final name when you create the app. Renaming once
   the store page is public confuses wishlists and search -- settle it first.
7. Check the name is free: search Steam, and trademark databases (EUIPO,
   USPTO) for games/software in class 9/41.

## C. Blockers before release

### Known
- ~~Real car names~~ DONE 2026-09-28: the player sees Aurel Civetta 1.2,
  Halcón RS18 rally, Nordwerk N540 and Rivière Courier 1.4 (keys, saves,
  medals unchanged; a rendered-text sweep of 91k strings found no real make
  or model on screen). Left, lower risk: the body SHAPES follow the real
  cars' proportions (`drive/bodies.py:61-96`, no badges / grilles / logos
  drawn) -- `THIRD_PARTY_NOTICES.md` section 4.
- **Tyre data licence**: the only tyre model the game reads,
  `tyre_data/TNO_car205_60R15.tir`, is a Delft-Tyre/TNO MF-Tool sample file
  with no licence. OWNER DECISION: get permission or replace it
  (`THIRD_PARTY_NOTICES.md` section 3).
- **UIUC airfoil data** (GPL-style terms with attribution) and the **LGPL
  source offer** for pygame & co: decide and ship the licence folder
  (`THIRD_PARTY_NOTICES.md` sections 2-3). The build now ships a
  `licenses/` folder (every bundled package's licence, LGPL-2.1 text);
  fill your contact address into `packaging/licenses/SOURCE_OFFER.md`.
- **Windows build**: PyInstaller does not cross-compile. You need a Windows PC
  (`packaging/build_windows.ps1`), a Windows VM, or GitHub Actions: the repo
  has no remote today -- push it to GitHub (a public repo's Actions runners
  are free; a private repo spends your Actions minutes, macOS ones fastest) and
  `.github/workflows/steam-build.yml` builds Windows, macOS arm64 + x64 and
  Linux depots (`steam/README.md` section 1). Then test the result on a clean
  Windows machine. Most Steam players are on Windows: don't launch without it.
- **macOS**: Steam requires new macOS apps to be **64-bit and notarized by
  Apple** ([platforms](https://partner.steamgames.com/doc/store/application/platforms)).
  That needs an Apple Developer account (USD 99/year), a "Developer ID
  Application" certificate, `codesign --deep --options runtime` of the .app,
  then `xcrun notarytool submit --wait` and `xcrun stapler staple`. Without
  that, drop macOS from the first release. Set a real bundle id first
  (`packaging/game.spec` has the placeholder `com.example.aleron`). The macOS
  depot holds one .app: Intel (runs everywhere via Rosetta) or Apple Silicon
  only -- `steam/README.md` section 1.

### From the audit (2026-09-28)

Fixed during the prep (no action):
- Packaged build crashed on start: the tyre file was found relative to the
  working folder (fixed in `drive/tyre.py`).
- Shipped bots were not found from the saves folder (the launcher copies them).
- No fullscreen / HiDPI: fullscreen by default, F11 or Alt+Enter toggles,
  remembered in `runs/display.json`; DPI-aware on Windows
  (`drive/display_mode.py`).
- "(see the terminal)" messages now say "(see the log)"; stdout/stderr go to
  `<saves>/logs/game.log`, crashes to `<saves>/logs/crash-*.txt`.
- AeroBO's caches no longer write inside the install folder.
- The logo draws any word (rename-safe).

Still open (owner):
1. **Test on real hardware** before submitting the build: a Windows PC (the
   Windows build has never run), a Steam Deck if you claim Deck support, and
   an Xbox-style pad through Steam Input (`packaging/PAD_TEST.md`).
2. The tyre-file / UIUC licences and the LGPL source-offer address: see
   "Known" above and `THIRD_PARTY_NOTICES.md`. (Car names: done.)
3. Nice to have: a Display row in Settings (F11 / Alt+Enter work already).

Done 2026-09-28 (second round): fictional car names everywhere on screen,
player-visible "carsim" text, the F11 / Alt+Enter hint on the Controls page,
unknown Steam launch options (dropped and logged, the game still starts), a
crash report + alert when the saves folder can't be written, the `licenses/`
folder in the build, only the used tyre file shipped, CI on current GitHub
runners (`macos-15`, `macos-15-intel`).

## D. Store page (can start the day the App ID exists)

1. Store Presence > **Store page**: texts from `steam/store/text.md` (short
   description, About This Game, features), genre, tags, supported languages
   (English; the UI is English-only), controller support level, system
   requirements (measure the builds first).
2. **Graphics** (`steam/store/make_store_assets.py` makes them; check each by
   eye). Required store assets
   ([standard assets](https://partner.steamgames.com/doc/store/assets/standard)):
   header capsule 920x430, small capsule 462x174, main capsule 1232x706,
   vertical capsule 748x896, at least 5 screenshots 1920x1080 (16:9) of real
   gameplay; optional page background 1438x810. Library assets (Store
   Presence > Graphical Assets > Library): library capsule 600x900, library
   hero 3840x1240, library logo (transparent PNG, up to 1280x720), library
   header 920x430; community icon 184x184 and client icon (.ico) under App
   Admin. A trailer is optional but helps a lot.
3. **Content survey** (mandatory): answers drafted in `steam/store/text.md`,
   including the Generative-AI section (owner decision).
4. **Pricing**: nothing to set -- the app was created as a free product
   (step A3). Tick the **Free To Play** genre and keep "Free to Play" among
   the top tags; say "100% free: no microtransactions, no ads, no DLC" in
   the short description.
5. **Release date**: set it (or "Coming soon") -- it can move later.
6. **Submit the store page for review**: typically **3-5 business days**;
   submit at least 7 days before you want it live
   ([releasing](https://partner.steamgames.com/doc/store/releasing),
   [review process](https://partner.steamgames.com/doc/store/review_process)).
7. After approval, **publish it as "Coming Soon"**. It must be visible as
   Coming Soon for **at least 2 weeks** before release
   ([releasing](https://partner.steamgames.com/doc/store/releasing)).
   Wishlists start counting now: post the link.

## E. Build

1. Build on each OS (`packaging/`), test on a clean machine: first launch
   opens the title screen, saves land in the per-user folder, quitting and
   relaunching keeps settings and records, the log in `<data>/logs/` has no
   traceback.
2. `steam/collect.sh` on each, then `steam/upload.sh --preview` (dry run) and
   `steam/upload.sh` for real.
3. Steamworks > Installation > General: the three **launch options**
   (`steam/README.md` section 2).
4. Steam Cloud: the Auto-Cloud rows (`steam/README.md` section 5).
5. Install from Steam on each OS via the **default branch** (set the build
   live on 'default' by hand) or a password-protected beta branch; play a
   full session.
6. **Submit the build for review** (the "game build" checklist): **3-5
   business days**. Valve checks it launches on every OS the store page lists
   and that every feature the page promises is in it
   ([review process](https://partner.steamgames.com/doc/store/review_process)).

## F. Steam Deck (optional, recommended)

Request a review from the app landing page > Technical Tools > "Steam Hardware
Compatibility Review"; it works for unreleased games too. "Verified" needs:
a default controller config reaching all content without touching settings,
text readable at 1280x800 (min 9 px), 30 fps at 800p, text entry through
Steam's keyboard or your own, no "unsupported hardware" warnings. Valve tests
the Linux build if there is one, else the Windows build under Proton, and
keeps the better result
([Deck compatibility](https://partner.steamgames.com/doc/steamdeck/compat)).
The garage's typed fields (build names) need a pad-friendly path for Verified.

## G. Release

Both checklists (store + build) approved, 30 days past the fee, 2 weeks in
Coming Soon: press **Release** yourself in Steamworks -- an approved game does
not release itself ([releasing](https://partner.steamgames.com/doc/store/releasing)).
Needs the "Publish app changes to Steam" permission. A launch discount can be
set up beforehand.

## H. After release

- Watch the reviews and the crash logs players send (`<data>/logs/`).
- Updates: build, `steam/upload.sh --beta testing`, try it, then set live on
  default. No new review needed for updates.
- Before any SAVE-format change: keep old saves loading (the save id is
  pinned; see B.2).
