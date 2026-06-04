"""
grid.py — 3x3 presentation grids of the curated per-character face sets.

Groups the crops in a flat folder by person (the <char>-<scenario>-NN.jpg stem) and writes one titled
3x3 PNG per character. Built for the curated local/fetch set (9 each).

    local/.venv/bin/python tools/grid.py                          # local/fetch -> local/fetch_review/grids/
    local/.venv/bin/python tools/grid.py --in local/fetch --cell 480
"""
import argparse
import os
import re

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# slug -> presentation title (the actual Community character names; Klepper is the similar-face control)
NAMES = {"jeff": "Jeff Winger", "britta": "Britta Perry", "abed": "Abed Nadir", "troy": "Troy Barnes",
         "annie": "Annie Edison", "shirley": "Shirley Bennett", "pierce": "Pierce Hawthorne",
         "chang": "Ben Chang", "dean": "Dean Pelton", "klepper": "Jordan Klepper"}


def person_of(stem):
    return re.split(r"[ -]", stem.strip(), 1)[0].lower()           # '<char>-<scenario>-NN' -> '<char>'


def font(size):
    for p in ("/System/Library/Fonts/Helvetica.ttc",
              "/System/Library/Fonts/Supplemental/Arial.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def make_grid(char, files, out_path, cell, cols, gap=10, margin=18, title_h=60):
    rows = (len(files) + cols - 1) // cols
    W = margin * 2 + cols * cell + (cols - 1) * gap
    H = title_h + margin + rows * cell + (rows - 1) * gap + margin
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)

    title = NAMES.get(char, char.capitalize())
    f = font(34)
    tb = d.textbbox((0, 0), title, font=f)
    d.text(((W - (tb[2] - tb[0])) / 2, (title_h - (tb[3] - tb[1])) / 2 - tb[1]), title,
           fill=(20, 20, 20), font=f)

    for i, fp in enumerate(sorted(files)):
        r, c = divmod(i, cols)
        x = margin + c * (cell + gap)
        y = title_h + margin + r * (cell + gap)
        im = Image.open(fp).convert("RGB").resize((cell, cell), Image.LANCZOS)   # crops are already square
        canvas.paste(im, (x, y))
    canvas.save(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="3x3 presentation grids of the curated per-character faces.")
    ap.add_argument("--in", dest="in_dir", default=os.path.join(ROOT, "local", "fetch"), help="flat crops dir")
    ap.add_argument("--out", default=os.path.join(ROOT, "local", "fetch_review", "grids"), help="output dir")
    ap.add_argument("--cell", type=int, default=460, help="per-image cell size px [460]")
    ap.add_argument("--cols", type=int, default=3, help="columns [3]")
    a = ap.parse_args()

    groups = {}
    for fn in sorted(os.listdir(a.in_dir)):
        if os.path.splitext(fn)[1].lower() in EXTS:
            groups.setdefault(person_of(os.path.splitext(fn)[0]), []).append(os.path.join(a.in_dir, fn))

    os.makedirs(a.out, exist_ok=True)
    for char in sorted(groups):
        p = make_grid(char, groups[char], os.path.join(a.out, f"{char}.png"), a.cell, a.cols)
        print(f"{char:<8} {len(groups[char])} imgs -> {p}")


if __name__ == "__main__":
    main()
