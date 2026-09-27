#!/bin/sh
# Re-vendor AeroBO into carsim/aerobo/ from a local AeroBO checkout.
#
#   aerobo/sync.sh <aerobo-repo> <rev>        e.g.  aerobo/sync.sh ~/dev/AeroBO 3f1b07d
#
# Copies EXACTLY what the release ships (git archive of the tracked src/, data/,
# records/ and LICENSE at <rev>), refreshes the warm screen checkpoint seed when
# the repo has one, rewrites the provenance block of VENDORED.md, then runs the
# bridge's self-check (python3 -m drive.aerobo_bridge). Engine patches: none --
# everything carsim needs lives in drive/aerobo_bridge.py, and its identity rows
# fail loudly if a re-sync moved a private helper the bridge relies on.
set -eu

REPO=${1:?usage: aerobo/sync.sh <aerobo-repo> <rev>}
REV=${2:?usage: aerobo/sync.sh <aerobo-repo> <rev>}
HERE=$(cd "$(dirname "$0")" && pwd)            # carsim/aerobo
ROOT=$(dirname "$HERE")                          # carsim

# /usr/bin/git can refuse to run until the Xcode licence is accepted; the
# command-line tools' own git never asks
GIT=git
if [ -x /Library/Developer/CommandLineTools/usr/bin/git ]; then
    GIT=/Library/Developer/CommandLineTools/usr/bin/git
fi

COMMIT=$("$GIT" -C "$REPO" rev-parse --short=7 "$REV^{commit}")
DESCRIBE=$("$GIT" -C "$REPO" describe --tags --always "$COMMIT")

# 1. the engine, data and records: tracked files only, nothing else. Unpacked
#    beside the tree and SYNCED into it, so only the files that changed are
#    touched: deleting ~2300 files and re-creating them in place on a Desktop
#    that iCloud syncs came back as a "name 2.py" duplicate of every one of
#    them (2026-09-24).
STAGE=$(mktemp -d "${TMPDIR:-/tmp}/aerobo-sync.XXXXXX")
trap 'rm -rf "$STAGE"' EXIT
"$GIT" -C "$REPO" archive "$COMMIT" src data records LICENSE | tar -x -C "$STAGE"
for top in src data records; do
    mkdir -p "$HERE/$top"
    rsync -rc --delete --exclude '__pycache__/' "$STAGE/$top/" "$HERE/$top/"
done
cp "$STAGE/LICENSE" "$HERE/LICENSE"

#    every file the release ships and nothing else: a stray (a sync client's
#    duplicate, a file left by hand) fails here, by name
WANT=$("$GIT" -C "$REPO" ls-tree -r --name-only "$COMMIT" src data records LICENSE | wc -l)
HAVE=$(cd "$HERE" && find src data records LICENSE -type f ! -path '*/__pycache__/*' \
       ! -name '.DS_Store' | wc -l)
if [ "$WANT" -ne "$HAVE" ]; then
    echo "aerobo/sync.sh: $HAVE files on disk, the release ships $WANT; not in it:" >&2
    "$GIT" -C "$REPO" ls-tree -r --name-only "$COMMIT" src data records LICENSE | sort > "$STAGE/want"
    (cd "$HERE" && find src data records LICENSE -type f ! -path '*/__pycache__/*' \
        ! -name '.DS_Store' | sort) | comm -13 "$STAGE/want" - | head -20 >&2
    exit 1
fi

# 2. the warm library checkpoint (results/ is gitignored in AeroBO, so it is
#    never in the archive; without it the first screen runs XFOIL over the
#    whole library). A refreshed seed also retires the installed copy, so the
#    bridge installs the new one on its next import.
mkdir -p "$HERE/seed"
if [ -f "$REPO/results/airfoil_screen_checkpoint.json" ]; then
    if ! cmp -s "$REPO/results/airfoil_screen_checkpoint.json" \
                "$HERE/seed/airfoil_screen_checkpoint.json"; then
        cp "$REPO/results/airfoil_screen_checkpoint.json" "$HERE/seed/"
        rm -f "$HERE/results/airfoil_screen_checkpoint.json"
    fi
fi

# 3. the provenance block (the bridge's row B2 re-derives every number in it)
python3 - "$HERE" "$REPO" "$COMMIT" "$DESCRIBE" <<'PY'
import datetime, hashlib, os, re, sys
here, repo, commit, describe = sys.argv[1:5]

def content_digest(root):
    """(counts per top entry, sha256 over 'path NUL sha256(file)' lines, sorted)."""
    rows, counts = [], {"src": 0, "data": 0, "records": 0, "LICENSE": 0}
    for top in counts:
        base = os.path.join(root, top)
        paths = [base] if os.path.isfile(base) else [
            os.path.join(d, f) for d, dirs, fs in os.walk(base)
            if "__pycache__" not in d.split(os.sep) for f in fs
            if f != ".DS_Store" and not f.endswith(".pyc")]
        for p in paths:
            rel = os.path.relpath(p, root).replace(os.sep, "/")
            with open(p, "rb") as fh:
                rows.append(f"{rel}\0{hashlib.sha256(fh.read()).hexdigest()}\n")
            counts[top] += 1
    rows.sort()
    return counts, hashlib.sha256("".join(rows).encode()).hexdigest()

counts, digest = content_digest(here)
seed = os.path.join(here, "seed", "airfoil_screen_checkpoint.json")
with open(seed, "rb") as fh:
    seed_sha = hashlib.sha256(fh.read()).hexdigest()
block = "\n".join([
    "<!-- provenance:begin (rewritten by aerobo/sync.sh; checked by row B2 of drive/aerobo_bridge.py) -->",
    "```",
    f"source:         {repo.replace(os.path.expanduser('~'), '~')}",
    f"commit:         {commit}",
    f"describe:       {describe}",
    f"vendored:       {datetime.date.today().isoformat()}",
    f"files:          src {counts['src']}, data {counts['data']}, "
    f"records {counts['records']}, LICENSE {counts['LICENSE']}",
    f"content sha256: {digest}",
    f"seed sha256:    {seed_sha}  seed/airfoil_screen_checkpoint.json",
    "```",
    "<!-- provenance:end -->"])
path = os.path.join(here, "VENDORED.md")
with open(path) as fh:
    text = fh.read()
new, n = re.subn(r"<!-- provenance:begin.*?<!-- provenance:end -->", lambda m: block,
                 text, flags=re.S)
if n != 1:
    sys.exit("VENDORED.md has no provenance block to rewrite")
with open(path, "w") as fh:
    fh.write(new)
print(f"vendored AeroBO {describe}: src {counts['src']}, data {counts['data']}, "
      f"records {counts['records']}; content {digest[:12]}; seed {seed_sha[:12]}")
PY

# 4. the bridge's self-check: identity rows, budgets, a tiny engine smoke
cd "$ROOT" && python3 -m drive.aerobo_bridge
