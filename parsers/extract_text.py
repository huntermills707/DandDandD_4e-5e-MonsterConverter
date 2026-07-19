#!/usr/bin/env python3
"""Geometry-aware text extraction for two-column rule book PDFs.

pdftotext's reading-order heuristics scramble text when stat blocks span
page/column breaks (name lines float away from their headers), and
PyMuPDF's block-level extraction can merge an entire busy page into one
giant "line". This script rebuilds the text stream from word positions
instead: cluster words into visual lines by y-coordinate, split lines into
left/right columns at the page midpoint, emit left column then right.

Usage: python3 -m parsers.extract_text "Monster Manual 4e.pdf" data/raw/mm4.txt
"""
import sys

import pymupdf

Y_TOL = 3.0  # points; words within this vertical distance share a line


def extract_page(page) -> list[str]:
    mid = page.rect.width / 2
    words = page.get_text("words")  # (x0, y0, x1, y1, word, ...)
    columns = ([], [])
    for w in words:
        columns[(w[0] + w[2]) / 2 >= mid].append(w)

    out = []
    for col in columns:
        col.sort(key=lambda w: (w[1], w[0]))
        lines = []
        for w in col:
            if lines and abs(w[1] - lines[-1][0][1]) <= Y_TOL:
                lines[-1].append(w)
            else:
                lines.append([w])
        for line in lines:
            line.sort(key=lambda w: w[0])
            out.append(" ".join(w[4] for w in line))
    return out


def extract(pdf_path: str) -> str:
    doc = pymupdf.open(pdf_path)
    pages = []
    for n, page in enumerate(doc, start=1):
        pages.append(f"<<<PAGE {n}>>>")
        pages.append("\n".join(extract_page(page)))
    return "\n".join(pages)


def main(pdf_path, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(extract(pdf_path))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
