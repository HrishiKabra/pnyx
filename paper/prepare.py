#!/usr/bin/env python3
"""Conversion-time preprocessing for the arXiv-style PDF build.

docs/writeup.md stays the single source of truth for the prose; this script
never edits it. It reads that file and writes ONE intermediate markdown file
(with a YAML front-matter block) for pandoc to convert, doing only mechanical,
meaning-preserving transforms:

  1. Lift the writeup's own H1 title and "## Abstract" section into a YAML
     front-matter block (title/author/date/abstract) so pandoc's default
     LaTeX template renders them as a proper title block + \\begin{abstract}.
     Both are then removed from the body so they are not duplicated.
  2. Insert a numbered LaTeX \\figure block (vector PDF, not PNG) directly
     after the paragraph that already names each figure by its
     analysis_out/*.png path -- resolving that path to its .pdf sibling. No
     prose is added, removed, or reworded; the figures were already described
     in the text, this only makes them visible in the PDF.

Usage: prepare.py <writeup.md> <output.md>
"""
import json
import re
import sys
from pathlib import Path

AUTHOR = "Hrishi Kabra"
DATE = "2026-07-26"

# (png path as it appears in backticks in the prose, pdf sibling to embed,
#  caption -- phrased from the writeup's own description of that figure)
FIGURES = [
    (
        "analysis_out/fig1.png",
        "analysis_out/fig1.pdf",
        "Posterior gap and Brier, market vs. static pools, computed over "
        "independently elicited beliefs from the same agents, with error bars.",
    ),
    (
        "analysis_out/v2/fig2_v2.png",
        "analysis_out/v2/fig2_v2.pdf",
        "Phase map: market deficit vs. measured team error correlation rho, "
        "with the pro and luna capability tiers added; manipulation panel of "
        "flip rate and recovery vs. adversary bankroll multiple k.",
    ),
    (
        "analysis_out/v2/fig3.png",
        "analysis_out/v2/fig3.pdf",
        "Per-tier market deficit (market gap minus pool gap) and the "
        "herding-regression price-weight b2, by capability tier and trading round.",
    ),
]


def latex_escape_light(text: str) -> str:
    """Minimal escaping for the hand-written captions above (defensive only;
    the captions as written contain none of these characters)."""
    for ch in ("\\", "%", "_", "&", "#", "{", "}"):
        text = text.replace(ch, "\\" + ch)
    return text


def figure_block(pdf_path: str, caption: str) -> str:
    return (
        "\n```{=latex}\n"
        "\\begin{figure}[htbp]\n"
        "\\centering\n"
        f"\\includegraphics[width=0.92\\textwidth]{{{pdf_path}}}\n"
        f"\\caption{{{latex_escape_light(caption)}}}\n"
        "\\end{figure}\n"
        "```\n"
    )


def yaml_block_scalar(text: str, indent: int = 2) -> str:
    pad = " " * indent
    return "\n".join(pad + line if line else pad.rstrip() for line in text.split("\n"))


def main() -> None:
    src_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    text = src_path.read_text(encoding="utf-8")
    lines = text.split("\n")

    if not lines[0].startswith("# "):
        raise SystemExit("expected the writeup to open with a single '# Title' line")
    title = lines[0][2:].strip()
    body = "\n".join(lines[1:])

    m = re.search(r"(?ms)^## Abstract\s*\n(.*?)(?=^## )", body)
    if not m:
        raise SystemExit("expected an '## Abstract' section before the next '## ' heading")
    abstract = m.group(1).strip()
    body = body[: m.start()] + body[m.end() :]

    # Insert figure blocks. Process in reverse order of first occurrence so
    # each insertion point (computed before any edits) stays valid for
    # markers to its left, and multiple markers sharing one paragraph (fig2_v2
    # and fig3 do) come out in the same left-to-right order they're named in.
    hits = []
    for png_path, pdf_path, caption in FIGURES:
        marker = f"`{png_path}`"
        idx = body.find(marker)
        if idx == -1:
            raise SystemExit(f"figure reference {marker} not found in writeup body")
        hits.append((idx, pdf_path, caption))
    hits.sort(key=lambda t: t[0], reverse=True)

    for idx, pdf_path, caption in hits:
        para_end = body.find("\n\n", idx)
        insertion_point = para_end + 2 if para_end != -1 else len(body)
        body = body[:insertion_point] + figure_block(pdf_path, caption) + body[insertion_point:]

    front_matter = (
        "---\n"
        f"title: {json.dumps(title)}\n"
        f"author: {json.dumps(AUTHOR)}\n"
        f"date: {json.dumps(DATE)}\n"
        "abstract: |\n"
        f"{yaml_block_scalar(abstract)}\n"
        "---\n"
    )

    out_path.write_text(front_matter + body, encoding="utf-8")


if __name__ == "__main__":
    main()
