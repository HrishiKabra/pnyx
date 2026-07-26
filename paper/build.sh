#!/usr/bin/env bash
# Build the arXiv-style PDF of the pnyx writeup: docs/writeup.md -> paper/pnyx.pdf
#
# Usage: bash paper/build.sh   (run from anywhere; resolves the repo root
# itself). Idempotent (wipes its own scratch dir first) and offline (only
# invokes the local pandoc + xelatex toolchain, no network access).
#
# docs/writeup.md remains the single source of truth for the prose. The only
# preprocessing is paper/prepare.py, which lifts the title/abstract into
# front matter and inserts the three vector-PDF figures next to the
# paragraphs that already name them -- see that script's docstring for the
# exact, meaning-preserving transforms performed.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PANDOC="${PANDOC:-pandoc}"
XELATEX="${XELATEX:-/Library/TeX/texbin/xelatex}"

command -v "$PANDOC" >/dev/null 2>&1 || { echo "error: pandoc not found (set \$PANDOC)" >&2; exit 1; }
command -v "$XELATEX" >/dev/null 2>&1 || { echo "error: xelatex not found at $XELATEX (set \$XELATEX)" >&2; exit 1; }

BUILD_DIR="$REPO_ROOT/paper/build"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

python3 "$REPO_ROOT/paper/prepare.py" \
  "$REPO_ROOT/docs/writeup.md" \
  "$BUILD_DIR/writeup.pandoc.md"

"$PANDOC" "$BUILD_DIR/writeup.pandoc.md" \
  --from=markdown+raw_attribute \
  --include-in-header="$REPO_ROOT/paper/header.tex" \
  --shift-heading-level-by=-1 \
  -V geometry:margin=1in \
  -V fontsize=11pt \
  -V colorlinks=true -V linkcolor=blue -V urlcolor=blue -V citecolor=blue \
  --standalone \
  -o "$BUILD_DIR/pnyx.tex"

# Resolve analysis_out/*.pdf figure paths (relative to the repo root) from
# whatever directory xelatex runs in.
(
  cd "$REPO_ROOT"
  "$XELATEX" -interaction=nonstopmode -halt-on-error \
    -output-directory="$BUILD_DIR" "$BUILD_DIR/pnyx.tex" >"$BUILD_DIR/xelatex_pass1.log" 2>&1
  "$XELATEX" -interaction=nonstopmode -halt-on-error \
    -output-directory="$BUILD_DIR" "$BUILD_DIR/pnyx.tex" >"$BUILD_DIR/xelatex_pass2.log" 2>&1
)

if grep -qi "missing character" "$BUILD_DIR/pnyx.log"; then
  echo "warning: 'Missing character' found in $BUILD_DIR/pnyx.log -- inspect before shipping" >&2
fi

cp "$BUILD_DIR/pnyx.pdf" "$REPO_ROOT/paper/pnyx.pdf"

if command -v pdfinfo >/dev/null 2>&1; then
  pages="$(pdfinfo "$REPO_ROOT/paper/pnyx.pdf" | awk -F': *' '/^Pages/{print $2}')"
  echo "Built paper/pnyx.pdf ($pages pages)"
else
  echo "Built paper/pnyx.pdf"
fi
