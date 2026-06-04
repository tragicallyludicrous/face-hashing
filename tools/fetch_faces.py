"""
fetch_faces.py — build a face dataset for the ArcFace / MICA identity stress-test by scraping the
web, DETECTION-ONLY. Nothing is ever dropped by identity, so the costume / paintball / silly-face
outliers SURVIVE — they are the whole point of the experiment.

Why detection-only matters: if we curated by "cosine to a reference identity is high enough", we'd
delete exactly the hard shots (zombie makeup, paintball masks, Halloween costumes) where ArcFace
struggles — and conclude it's more robust than it is. That's circular. So here we keep every
resolvable face and instead MEASURE: build a clean centroid per person from the `normal` shots, then
record each shot's cosine to that centroid. The manifest column is the result — sort by it to see the
difficulty curve (a press headshot near 0.7, Abed-as-Batman maybe 0.3). A low cosine is a *flag*, not
a filter: it could mean a genuine outlier OR a wrong-person scrape — your call from the contact sheet.

What it does, per target person:
  1. scrape DuckDuckGo Images for each (scenario -> query)   [ddgs]
  2. detect EVERY face in each image and crop it (group/ensemble shots -> one crop per face)
  3. dedupe literal repeats by image aHash (NOT by identity)
  4. centroid = mean of the `normal`-scenario embeddings; record cosine(face, centroid) for all
  5. write flat crops  <out>/<char>-<scenario>-NN.jpg     <- ready for run.py (-i <out>)
     a manifest CSV + a per-person contact sheet (outliers first) into <review>/  <- triage by eye

Run (one-time, needs internet — `local/.venv/bin/pip install ddgs` once):
    local/.venv/bin/python tools/fetch_faces.py                 # all built-in targets
    local/.venv/bin/python tools/fetch_faces.py --only abed,klepper
    local/.venv/bin/python tools/fetch_faces.py --config my_targets.json

Then the pipeline, pointed at the flat crops:
    cd local && .venv/bin/python run.py -i fetch -o out_community --device mps
    .venv/bin/python ../tools/present.py --compare --dir out_community     # the actual separation metrics

The crops land flat and named <char>-<scenario>-NN.jpg so present.py's person_of() groups them by
<char> automatically. The contact sheets / manifest live in a SEPARATE dir so run.py (flat os.listdir)
never ingests them as input photos. Output is git-ignored (local/fetch*/ — scraped, copyrighted).

NOTE: the cosine here comes from antelopev2's recognizer (the detector already on disk, same one the
pipeline's fallback uses) — a triage signal, not the pipeline's MICA-internal ArcFace. The rigorous
number is present.py over the real run.py outputs.
"""
import argparse
import csv
import json
import os
import shutil
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "local", "fetch")              # flat crops -> run.py -i fetch
REVIEW_DIR = os.path.join(ROOT, "local", "fetch_review")    # manifest + contact sheets (NOT pipeline input)
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Built-in targets: <char slug> -> {<scenario>: <search query>}. A scenario named "normal"/"normal*"
# is the clean reference the centroid is built from (aim ~60% normal). The rest deliberately mine the
# weird tail. Character names (not actor names) — that's the experiment: how far ArcFace identity
# holds across the same character in costume/paint/silly faces. Edit freely or pass --config.
TARGETS = {
    "jeff":    {"normal": "Jeff Winger Community", "normal2": "Jeff Winger Greendale suit",
                "paintball": "Jeff Winger paintball Community", "halloween": "Jeff Winger Halloween Community"},
    "britta":  {"normal": "Britta Perry Community", "normal2": "Britta Perry Gillian Jacobs Community",
                "paintball": "Britta Perry paintball Community", "costume": "Britta Perry costume Community"},
    "abed":    {"normal": "Abed Nadir Community", "normal2": "Abed Nadir Danny Pudi Community",
                "batman": "Abed Nadir Batman Community", "paintball": "Abed paintball Community",
                "spacetime": "Abed Inspector Spacetime Community"},
    "troy":    {"normal": "Troy Barnes Community", "normal2": "Troy Barnes Donald Glover Community",
                "paintball": "Troy Barnes paintball Community", "halloween": "Troy Barnes Halloween Community"},
    "annie":   {"normal": "Annie Edison Community", "normal2": "Annie Edison Alison Brie Community",
                "paintball": "Annie Edison paintball Community", "costume": "Annie Edison costume Community"},
    "shirley": {"normal": "Shirley Bennett Community", "normal2": "Shirley Bennett Yvette Nicole Brown Community",
                "halloween": "Shirley Bennett Halloween Community", "paintball": "Shirley paintball Community"},
    "pierce":  {"normal": "Pierce Hawthorne Community", "normal2": "Pierce Hawthorne Chevy Chase Community",
                "costume": "Pierce Hawthorne costume Community", "laser": "Pierce Hawthorne hologram Community"},
    "chang":   {"normal": "Senor Chang Community", "normal2": "Ben Chang Ken Jeong Community",
                "changnesia": "Chang Changnesia Community", "guard": "Chang security guard Community",
                "costume": "Chang costume Community"},
    "dean":    {"normal": "Dean Pelton Community", "normal2": "Dean Pelton Jim Rash Community",
                "dalmatian": "Dean Pelton dalmatian costume Community", "costume": "Dean Pelton costume Community"},
    # Not a Community character — a similar-face stress test (looks like the user): keep it mostly clean.
    "klepper": {"normal": "Jordan Klepper portrait", "normal2": "Jordan Klepper Daily Show",
                "interview": "Jordan Klepper interview", "normal3": "Jordan Klepper headshot"},
}


def is_normal(scenario):
    return scenario.lower().startswith("normal")


# ---- scrape -------------------------------------------------------------------------------------

_UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"}      # some hosts 403 a bare UA


def scrape(query, n, raw_dir):
    """DuckDuckGo image search -> download up to n images into raw_dir (ddgs gives URLs; we fetch)."""
    import time

    import requests
    from ddgs import DDGS
    os.makedirs(raw_dir, exist_ok=True)
    try:
        with DDGS() as ddg:
            results = list(ddg.images(query, max_results=n * 2))    # over-fetch: some URLs 403 / are dead
    except Exception as e:                                          # ratelimit etc. — skip this query
        print(f"    ! ddgs '{query}': {e}"); return []
    sess = requests.Session()
    saved = []
    for r in results:
        if len(saved) >= n:
            break
        url = r.get("image")
        if not url:
            continue
        ext = os.path.splitext(url.split("?")[0])[1].lower()
        dest = os.path.join(raw_dir, f"{len(saved):03d}{ext if ext in EXTS else '.jpg'}")
        try:
            resp = sess.get(url, timeout=15, headers=_UA)
            resp.raise_for_status()
            with open(dest, "wb") as fh:
                fh.write(resp.content)
            saved.append(dest)
        except Exception:
            continue                                               # dead link / 403 / timeout -> next
    time.sleep(1.0)                                                # be polite to ddgs between queries
    return saved


# ---- dedupe (image aHash, NOT identity) ---------------------------------------------------------

def ahash(crop_bgr):
    """64-bit average hash of a crop — catches the same press photo returned by multiple queries."""
    import cv2
    g = cv2.resize(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY), (8, 8), interpolation=cv2.INTER_AREA)
    bits = (g > g.mean()).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def hamming(a, b):
    return bin(a ^ b).count("1")


# ---- detect + crop ------------------------------------------------------------------------------

def detector(providers=("CPUExecutionProvider",)):
    """antelopev2 with detection + recognition (recognition gives the ArcFace embedding for cosine)."""
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="antelopev2", allowed_modules=["detection", "recognition"], providers=list(providers))
    app.prepare(ctx_id=-1, det_size=(640, 640))             # big det_size -> catches small-in-frame faces
    return app


def crop_square(img, bbox, pad):
    """Square, in-bounds crop centered on the face (no stretch). Returns (crop, native_face_px)."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    face_px = max(x2 - x1, y2 - y1)
    side = min(face_px * pad, W, H)                          # keep it square AND inside the image
    half = side / 2.0
    X1 = int(round(min(max(cx - half, 0.0), W - side)))
    Y1 = int(round(min(max(cy - half, 0.0), H - side)))
    s = int(round(side))
    return img[Y1:Y1 + s, X1:X1 + s], int(round(face_px))


# ---- per-person processing ----------------------------------------------------------------------

def process_char(app, char, scenarios, args, raw_root):
    """Scrape + detect + crop + centroid for one person -> (rows, [(crop, label) for the sheet])."""
    import cv2
    cands = []                                               # {scenario, query, crop, emb, face_px, n_faces, w, h, score, hash}
    for scenario, query in scenarios.items():
        raw = tempfile.mkdtemp(prefix=f"{char}_{scenario}_", dir=raw_root)
        try:
            paths = scrape(query, args.per_scenario, raw)
        except Exception as e:                              # one bad query shouldn't sink the person
            print(f"    ! {scenario}: scrape failed ({e})"); paths = []
        kept = 0
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                continue
            H, W = img.shape[:2]
            faces = app.get(img)
            for fc in faces:
                crop, face_px = crop_square(img, fc.bbox, args.pad)
                if face_px < args.min_face or crop.size == 0:
                    continue                                # too small to be usable (a quality floor, not identity)
                cands.append({"scenario": scenario, "query": query, "crop": crop, "face_px": face_px,
                              "emb": fc.normed_embedding.astype(np.float64), "n_faces": len(faces),
                              "w": W, "h": H, "score": float(fc.det_score), "hash": ahash(crop)})
                kept += 1
        print(f"    {scenario:<10} {query[:42]:<42} {len(paths):>2} imgs -> {kept:>2} faces")
        shutil.rmtree(raw, ignore_errors=True)

    # dedupe literal repeats (image aHash) across the whole person; keep the higher-detection-score copy
    cands.sort(key=lambda c: -c["score"])
    uniq, hashes = [], []
    for c in cands:
        if any(hamming(c["hash"], h) <= args.dup_dist for h in hashes):
            continue
        hashes.append(c["hash"]); uniq.append(c)
    dropped = len(cands) - len(uniq)

    # centroid from the `normal` scenarios -> cosine for every face (the measurement; drops nothing)
    norm_embs = [c["emb"] for c in uniq if is_normal(c["scenario"])]
    centroid = None
    if norm_embs:
        m = np.mean(norm_embs, axis=0)
        centroid = m / (np.linalg.norm(m) + 1e-9)
    for c in uniq:
        c["cos"] = float(np.dot(c["emb"], centroid)) if centroid is not None else float("nan")

    # write crops (outliers first so NN encodes difficulty) + manifest rows + sheet tiles
    uniq.sort(key=lambda c: (np.inf if np.isnan(c["cos"]) else c["cos"]))   # hardest first; no-centroid last
    rows, tiles = [], []
    for i, c in enumerate(uniq):
        fname = f"{char}-{c['scenario']}-{i:02d}.jpg"
        cv2.imwrite(os.path.join(args.out, fname), c["crop"])
        rows.append({"file": fname, "char": char, "scenario": c["scenario"],
                     "centroid_cosine": f"{c['cos']:.4f}", "face_px": c["face_px"],
                     "n_faces_in_image": c["n_faces"], "det_score": f"{c['score']:.3f}",
                     "src_w": c["w"], "src_h": c["h"], "query": c["query"]})
        tiles.append((c["crop"], f"{c['scenario']} c={c['cos']:.2f} {c['face_px']}px"))
    print(f"    => {len(uniq)} faces kept ({dropped} dup), centroid={'yes' if centroid is not None else 'NONE'}")
    return rows, tiles


# ---- contact sheet ------------------------------------------------------------------------------

def contact_sheet(char, tiles, path, cell=168, cols=6):
    """Tile a person's crops (outliers first), captioned scenario + cosine + px, for 10-second triage."""
    import cv2
    if not tiles:
        return
    pad, head = 6, 26
    rows = (len(tiles) + cols - 1) // cols
    W = cols * (cell + pad) + pad
    Hgrid = rows * (cell + head + pad) + pad + head
    sheet = np.full((Hgrid, W, 3), 32, np.uint8)
    cv2.putText(sheet, f"{char}  ({len(tiles)} faces, hardest first)", (pad, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    for k, (crop, cap) in enumerate(tiles):
        r, cc = divmod(k, cols)
        x = pad + cc * (cell + pad)
        y = head + pad + r * (cell + head + pad)
        sheet[y:y + cell, x:x + cell] = cv2.resize(crop, (cell, cell), interpolation=cv2.INTER_AREA)
        cv2.putText(sheet, cap, (x + 2, y + cell + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (200, 230, 255), 1, cv2.LINE_AA)
    cv2.imwrite(path, sheet)


# ---- main ---------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Scrape a face dataset (detection-only) for the ArcFace/MICA stress test.")
    ap.add_argument("--out", default=OUT_DIR, help="flat crops dir (run.py -i this) [local/fetch]")
    ap.add_argument("--review", default=REVIEW_DIR, help="manifest + contact sheets [local/fetch_review]")
    ap.add_argument("--config", help="JSON {char: {scenario: query}} overriding the built-in targets")
    ap.add_argument("--only", help="comma list of chars to fetch (subset of the targets)")
    ap.add_argument("--per-scenario", type=int, default=10, help="images scraped per query [10]")
    ap.add_argument("--pad", type=float, default=2.2, help="crop = face box * pad, square [2.2]")
    ap.add_argument("--min-face", type=int, default=50, help="drop faces smaller than this (px) [50]")
    ap.add_argument("--dup-dist", type=int, default=5, help="aHash Hamming <= this = duplicate [5]")
    a = ap.parse_args()

    targets = json.load(open(a.config)) if a.config else TARGETS
    if a.only:
        want = {s.strip() for s in a.only.split(",")}
        missing = want - set(targets)
        if missing:
            ap.error(f"--only: unknown {sorted(missing)}; have {sorted(targets)}")
        targets = {k: v for k, v in targets.items() if k in want}

    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.review, exist_ok=True)
    app = detector()
    raw_root = tempfile.mkdtemp(prefix="fetch_raw_")

    all_rows = []
    try:
        for char, scenarios in targets.items():
            print(f"\n[{char}]")
            rows, tiles = process_char(app, char, scenarios, a, raw_root)
            all_rows += rows
            contact_sheet(char, tiles, os.path.join(a.review, f"{char}_contact.png"))
    finally:
        shutil.rmtree(raw_root, ignore_errors=True)

    cols = ["file", "char", "scenario", "centroid_cosine", "face_px", "n_faces_in_image",
            "det_score", "src_w", "src_h", "query"]
    with open(os.path.join(a.review, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(all_rows)

    rel = os.path.relpath(a.out, os.path.join(ROOT, "local"))
    shown = rel if not rel.startswith("..") else a.out          # clean for the default, absolute if elsewhere
    print(f"\n{len(all_rows)} crops -> {a.out}")
    print(f"review: {os.path.join(a.review, 'manifest.csv')} + <char>_contact.png  (outliers first; low cosine = costume OR wrong person — your call)")
    print(f"next:   cd local && .venv/bin/python run.py -i {shown} -o out_community --device mps")


if __name__ == "__main__":
    main()
