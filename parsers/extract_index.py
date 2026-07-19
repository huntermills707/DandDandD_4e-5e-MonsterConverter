#!/usr/bin/env python3
"""Extract the 5e MM stat-block index using block-level column clustering.

The index pages use a three-column layout that defeats a fixed midpoint
split; PyMuPDF's text blocks, however, are already whole column runs.
Cluster blocks by x-position into columns, then emit each column's lines
top to bottom.

Usage: python3 -m parsers.extract_index "D&D 5E - Monster Manual.pdf" data/raw/mm5_index.txt
"""
import sys

import pymupdf

COL_GAP = 40  # points between column x-positions


def extract_page(page) -> list[str]:
    blocks = [b for b in page.get_text("blocks") if b[6] == 0]
    blocks.sort(key=lambda b: b[0])
    columns = []
    for b in blocks:
        if columns and abs(b[0] - columns[-1][0][0]) < COL_GAP:
            columns[-1].append(b)
        else:
            columns.append([b])
    out = []
    for col in columns:
        col.sort(key=lambda b: b[1])
        for b in col:
            out.extend(ln.strip() for ln in b[4].splitlines())
    return [ln for ln in out if ln]


def main(pdf_path, out_path):
    doc = pymupdf.open(pdf_path)
    # index is at the back; find it by its heading
    start = None
    for n in range(len(doc) - 1, max(len(doc) - 8, -1), -1):
        if "INDEX OF STAT BLOCKS" in doc[n].get_text():
            start = n
            break
    if start is None:
        sys.exit("index heading not found in the last pages")
    lines = []
    for n in range(start, len(doc)):
        lines.extend(extract_page(doc[n]))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {out_path} (pdf pages {start + 1}-{len(doc)})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
