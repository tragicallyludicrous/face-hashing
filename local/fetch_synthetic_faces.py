"""fetch_synthetic_faces.py — download N synthetic faces for the v3 whitening basis.

SFHQ (Synthetic Faces High Quality, SelfishGene): StyleGAN2 / text-to-image faces — *fully
synthetic*, the author states there are no privacy or license issues since no image depicts a real
person. We only need a few thousand to MEASURE the shape of the ArcFace identity manifold
(`build_gallery.py` -> `build_basis.py`); no real identity is ever involved, and v3's output is a
derived synthetic.

    pip install "datasets>=2.14" huggingface_hub pillow
    python fetch_synthetic_faces.py -n 3000 -o synthetic_faces/

Streams from the Hugging Face mirror, so it pulls only N, not the whole set. A few thousand is plenty
for a stable 512-d whitening basis. If the default dataset/column ever changes, pass --dataset.
"""
import argparse
import os


def main():
    ap = argparse.ArgumentParser(description="Download N synthetic faces (SFHQ) for the v3 basis.")
    ap.add_argument("-n", "--num", type=int, default=3000, help="how many faces (>= ~2000 recommended)")
    ap.add_argument("-o", "--out", default="synthetic_faces")
    ap.add_argument("--dataset", default="bitmind/SyntheticFacesHQ", help="HF dataset id")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-size", type=int, default=512, help="downscale longest side (0 = keep full)")
    a = ap.parse_args()

    from datasets import load_dataset
    from PIL import Image

    os.makedirs(a.out, exist_ok=True)
    print(f"streaming {a.dataset} [{a.split}] -> {a.out}  (target {a.num})")
    ds = load_dataset(a.dataset, split=a.split, streaming=True)

    n = 0
    for ex in ds:
        img = next((v for v in ex.values() if isinstance(v, Image.Image)), None)  # find the image col
        if img is None:
            continue
        img = img.convert("RGB")
        if a.max_size and max(img.size) > a.max_size:
            img.thumbnail((a.max_size, a.max_size))
        img.save(os.path.join(a.out, f"sfhq_{n:06d}.jpg"), quality=92)
        n += 1
        if n % 200 == 0:
            print(f"  {n}/{a.num}")
        if n >= a.num:
            break
    print(f"\n{n} synthetic faces -> {a.out}"
          + ("" if n >= a.num else "  (stream ended early — try --dataset or a larger split)"))


if __name__ == "__main__":
    main()
