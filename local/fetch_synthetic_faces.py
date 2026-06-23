"""fetch_synthetic_faces.py — download synthetic faces (SFHQ) for the v3 whitening basis.

SFHQ (Synthetic Faces High Quality, SelfishGene) is *fully synthetic* (StyleGAN2 / text-to-image) —
no real identity, and the author states no privacy/license issues. We only need a couple thousand to
MEASURE the shape of the ArcFace identity manifold (`build_gallery.py` -> `build_basis.py`); v3's
output is a derived synthetic, not a copy of anyone.

The HF mirror (bitmind/SyntheticFacesHQ) stores the data as ZIP files (verified), so this downloads
the small sample zip(s) — ~750 faces each, 1024x1024 — and extracts them. Needs only huggingface_hub
+ pillow (no `datasets`).

    python fetch_synthetic_faces.py -o synthetic_faces/             # ~1500 faces (two small samples)
    python fetch_synthetic_faces.py --zips tiny-sample.zip -o ...   # quick 23 MB smoke test (~130 faces)
    python fetch_synthetic_faces.py --zips SFHQ-part1.zip  -o ... -n 5000   # from a full 90k part (4.7 GB)
"""
import argparse
import io
import os
import zipfile


def main():
    ap = argparse.ArgumentParser(description="Download + extract synthetic faces (SFHQ) for the v3 basis.")
    ap.add_argument("-o", "--out", default="synthetic_faces")
    ap.add_argument("--zips", nargs="+", default=["small-sample.zip", "small-sample-4.zip"],
                    help="which zip(s) from bitmind/SyntheticFacesHQ to fetch+extract")
    ap.add_argument("--dataset", default="bitmind/SyntheticFacesHQ", help="HF dataset id")
    ap.add_argument("-n", "--num", type=int, default=0, help="cap total images extracted (0 = all)")
    ap.add_argument("--max-size", type=int, default=512, help="downscale longest side (0 = keep full)")
    ap.add_argument("--pad", type=float, default=0.4, help="margin border as a fraction of the long side "
                    "(SFHQ are tight face crops; RetinaFace needs margin — 0.4 -> ~100%% detection)")
    a = ap.parse_args()

    import numpy as np
    import cv2
    from huggingface_hub import hf_hub_download
    from PIL import Image

    os.makedirs(a.out, exist_ok=True)
    n = 0
    for zp in a.zips:
        print(f"downloading {zp} from {a.dataset} ...")
        path = hf_hub_download(a.dataset, zp, repo_type="dataset")     # cached under ~/.cache/huggingface
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if name.startswith("__MACOSX") or not name.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                try:
                    img = Image.open(io.BytesIO(z.read(name))).convert("RGB")
                except Exception:                                     # noqa: BLE001 — skip bad entries
                    continue
                if a.max_size and max(img.size) > a.max_size:
                    img.thumbnail((a.max_size, a.max_size))
                arr = np.asarray(img)
                if a.pad > 0:                                         # give the detector margin
                    m = int(max(arr.shape[:2]) * a.pad)
                    arr = cv2.copyMakeBorder(arr, m, m, m, m, cv2.BORDER_REPLICATE)
                Image.fromarray(arr).save(os.path.join(a.out, f"sfhq_{n:06d}.jpg"), quality=92)
                n += 1
                if n % 250 == 0:
                    print(f"  {n}")
                if a.num and n >= a.num:
                    break
        if a.num and n >= a.num:
            break
    print(f"\n{n} synthetic faces -> {a.out}")


if __name__ == "__main__":
    main()
