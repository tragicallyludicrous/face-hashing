"""
render_cond.py — Stage-4 BRIDGE (work in progress): render a SMIRK-posed mesh aligned to the
ORIGINAL photo, to feed ControlNet / inpainting in SwarmUI/ComfyUI (see ../stage4-swarmui.md).

SMIRK predicts an orthographic camera in the 224 face-CROP space, so to land the mesh on the full
photo we: orthographic-project (its `cam` = [s, tx, ty]) into the crop, then map crop->full with the
inverse of the face-crop similarity transform.

`overlay` checks alignment; `--maps` then RASTERIZES the posed mesh into ControlNet conditioning
(depth + inpaint mask, optional normal) registered to the photo — the inputs Stage-4 M1 loads.

    python render_cond.py <image.jpg>                                  # overlay (alignment check)
    python render_cond.py <image_or_dir> --maps [--normal] [--shape shape.npy] [--out-dir DIR]

Geometry defaults to SMIRK's OWN reconstruction of the photo (self-aligned); pass --shape (a 300-d
MICA / hashed shape) to impose a different identity instead, exactly like smirk_local.compose.
"""
import argparse
import os

import numpy as np

import smirk_local as smirk          # reuse load(), crop_face(), the encoder+FLAME, the detector fallback
import torch                          # noqa: E402  (smirk_local sets the MPS/torch shims on import)


def encode_with_tform(h, image_path):
    """smirk._encode + also return the crop transform and the original BGR image. None if no face."""
    import cv2
    from skimage.transform import warp
    from utils.mediapipe_utils import run_mediapipe

    img = cv2.imread(image_path)
    if img is None:
        return None
    kpt = run_mediapipe(img)
    if kpt is None:
        kpt = smirk._retinaface_landmarks(h, img, run_mediapipe)   # small/profile faces mediapipe misses
    if kpt is None:
        return None
    tform = smirk.crop_face(img, kpt, scale=1.4, image_size=224)
    crop = warp(img, tform.inverse, output_shape=(224, 224), preserve_range=True).astype(np.uint8)
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    t = torch.tensor(crop).permute(2, 0, 1).unsqueeze(0).float().div(255.0).to(h.device)
    with torch.no_grad():
        out = h.encoder(t)
    return out, tform, img


def project_to_image(verts, cam, tform, image_size=224):
    """FLAME verts (N,3 m) + SMIRK cam [s,tx,ty] + crop tform -> full-image xy (N,2 px) and depth z (N,).

    Mirrors SMIRK's batch_orth_proj then its y/z negation, maps NDC[-1,1]->crop px, then crop->full.
    """
    s, tx, ty = (float(c) for c in cam)
    ndc = np.concatenate([verts[:, :2] + np.array([tx, ty]), verts[:, 2:3]], axis=1) * s
    ndc[:, 1:] *= -1.0                                              # SMIRK negates Y and Z
    crop_px = (ndc[:, 0] + 1.0) * (image_size / 2.0)
    crop_py = (ndc[:, 1] + 1.0) * (image_size / 2.0)
    full = tform.inverse(np.stack([crop_px, crop_py], axis=1))     # crop px -> full-image px
    return full, ndc[:, 2]


def _flame_verts(h, out):
    with torch.no_grad():
        return h.flame.forward(out)["vertices"][0].detach().cpu().numpy().astype(np.float64)


def _flame_faces(h):
    """FLAME triangle list (F,3) from the SMIRK model (faces_tensor or faces)."""
    f = getattr(h.flame, "faces_tensor", None)
    if f is None:
        f = getattr(h.flame, "faces", None)
    if f is None:
        raise SystemExit("could not find FLAME faces on the SMIRK model (no faces_tensor/faces)")
    return (f.detach().cpu().numpy() if hasattr(f, "detach") else np.asarray(f)).astype(np.int64)


def _vertex_normals(verts, faces):
    """Per-vertex normals (N,3, unit) = normalized sum of incident face normals."""
    vn = np.zeros_like(verts)
    tri = verts[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-9)


def rasterize(xy, z, faces, W, H, vnormals=None, cull=1):
    """Barycentric z-buffer raster -> (depth float HxW, coverage bool HxW, normal HxWx3 | None).

    Front surface only: triangles are backface-culled by screen-space winding (sign `cull`), then the
    nearest (largest z) fragment wins per pixel. Loops per kept triangle over its bbox (vectorized inside).
    """
    depth = np.full((H, W), -np.inf, np.float32)
    cover = np.zeros((H, W), bool)
    nrm = np.zeros((H, W, 3), np.float32) if vnormals is not None else None
    p, zf = xy[faces], z[faces]                                     # (F,3,2), (F,3)
    area2 = ((p[:, 1, 0]-p[:, 0, 0])*(p[:, 2, 1]-p[:, 0, 1])
             - (p[:, 2, 0]-p[:, 0, 0])*(p[:, 1, 1]-p[:, 0, 1]))     # screen-space signed area (winding)
    for f in np.nonzero(area2 * cull > 0)[0]:                       # front-facing only
        (x0, y0), (x1, y1), (x2, y2) = p[f]
        minx, maxx = max(int(np.floor(min(x0, x1, x2))), 0), min(int(np.ceil(max(x0, x1, x2))), W-1)
        miny, maxy = max(int(np.floor(min(y0, y1, y2))), 0), min(int(np.ceil(max(y0, y1, y2))), H-1)
        denom = (y1-y2)*(x0-x2) + (x2-x1)*(y0-y2)
        if maxx < minx or maxy < miny or abs(denom) < 1e-9:
            continue
        gx, gy = np.meshgrid(np.arange(minx, maxx+1)+0.5, np.arange(miny, maxy+1)+0.5)
        a = ((y1-y2)*(gx-x2) + (x2-x1)*(gy-y2)) / denom
        b = ((y2-y0)*(gx-x2) + (x0-x2)*(gy-y2)) / denom
        c = 1.0 - a - b
        zi = a*zf[f, 0] + b*zf[f, 1] + c*zf[f, 2]
        sd = depth[miny:maxy+1, minx:maxx+1]
        win = (a >= 0) & (b >= 0) & (c >= 0) & (zi > sd)            # inside triangle AND nearer
        if not win.any():
            continue
        sd[win] = zi[win]
        cover[miny:maxy+1, minx:maxx+1][win] = True
        if nrm is not None:
            ni = (a[..., None]*vnormals[faces[f, 0]] + b[..., None]*vnormals[faces[f, 1]]
                  + c[..., None]*vnormals[faces[f, 2]])
            nrm[miny:maxy+1, minx:maxx+1][win] = ni[win]
    return depth, cover, nrm


def render_maps(h, image_path, out_dir=None, shape=None, feather=0.0, want_normal=False, flip_cull=False):
    """Rasterize the SMIRK-posed mesh into ControlNet conditioning aligned to the photo:
    <stem>_depth.png + <stem>_mask.png (+ <stem>_normal.png) — the inputs Stage-4 M1 loads.

    Geometry defaults to SMIRK's own reconstruction of THIS photo (self-aligned). Pass `shape` (a 300-d
    MICA / hashed shape) to impose a DIFFERENT identity instead — that's the real bridge for the hash.
    """
    import cv2
    enc = encode_with_tform(h, image_path)
    if enc is None:
        print("no face:", image_path); return None
    out, tform, img = enc
    if shape is not None:                                          # swap identity into SMIRK's FLAME (like compose)
        K = out["shape_params"].shape[1]
        out = dict(out)
        out["shape_params"] = torch.tensor(np.asarray(shape, np.float32).reshape(1, -1)[:, :K], device=h.device)
    verts, faces = _flame_verts(h, out), _flame_faces(h)
    xy, z = project_to_image(verts, out["cam"][0].cpu().numpy(), tform)
    vn = _vertex_normals(verts, faces) if want_normal else None
    H, W = img.shape[:2]
    # project_to_image's z grows AWAY from the camera, so negate it: the FRONT face then wins the
    # z-buffer (nearest) and normalizes to white. cull=-1 keeps the front-facing (not back-of-head) triangles.
    depth, cover, nrm = rasterize(xy, -z, faces, W, H, vn, cull=(1 if flip_cull else -1))
    if not cover.any():
        print("mesh projected outside the image:", image_path); return None

    stem = os.path.splitext(os.path.basename(image_path))[0]
    d = os.path.abspath(out_dir or os.path.dirname(image_path) or ".")
    os.makedirs(d, exist_ok=True)
    zc = depth[cover]                                              # depth: front = white, background = black
    dn = np.zeros((H, W), np.float32); dn[cover] = (depth[cover] - zc.min()) / (np.ptp(zc) + 1e-9)
    cv2.imwrite(os.path.join(d, f"{stem}_depth.png"), cv2.cvtColor((dn*255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
    mask8 = (cover*255).astype(np.uint8)                          # silhouette = the inpaint region
    if feather > 0:
        mask8 = cv2.GaussianBlur(mask8, (0, 0), feather)
    cv2.imwrite(os.path.join(d, f"{stem}_mask.png"), mask8)
    outs = [f"{stem}_depth.png", f"{stem}_mask.png"]
    if nrm is not None:
        n = nrm.copy(); n[..., 1] *= -1; n[..., 2] *= -1          # to screen space (y,z negated like the projection)
        enc8 = np.zeros((H, W, 3), np.uint8)
        enc8[cover] = ((n[cover]*0.5 + 0.5)*255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(d, f"{stem}_normal.png"), enc8[..., ::-1])   # RGB -> BGR for cv2
        outs.append(f"{stem}_normal.png")
    print(f"{stem}: {int(cover.sum())} px -> " + ", ".join(outs) + f"  ({d})")
    return d


def overlay(h, image_path, out_png):
    """Project the posed mesh onto the photo and dump an alignment-check overlay."""
    import cv2
    enc = encode_with_tform(h, image_path)
    if enc is None:
        print("no face:", image_path); return None
    out, tform, img = enc
    verts = _flame_verts(h, out)
    xy, z = project_to_image(verts, out["cam"][0].cpu().numpy(), tform)

    vis = img.copy()
    H, W = vis.shape[:2]
    inb = (xy[:, 0] >= 0) & (xy[:, 0] < W) & (xy[:, 1] >= 0) & (xy[:, 1] < H)
    zc = z[inb]
    t = (zc - zc.min()) / (np.ptp(zc) + 1e-9)                       # depth -> color, just for the check
    pts = xy[inb].astype(int)
    for (x, y), tt in zip(pts, t):
        cv2.circle(vis, (x, y), 1, (int(255 * (1 - tt)), 80, int(255 * tt)), -1)
    # silhouette (convex hull of the projected verts) — a preview of the inpaint mask
    hull = cv2.convexHull(xy.astype(np.float32).reshape(-1, 1, 2))
    cv2.polylines(vis, [hull.astype(int)], True, (0, 255, 0), 2)

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    cv2.imwrite(out_png, vis)
    print(f"overlay -> {out_png}  ({len(verts)} verts, {int(inb.sum())} in frame; green = mesh silhouette)")
    return out_png


def cams_batch(h, in_dir, out_json, exts=(".jpg", ".jpeg", ".png", ".bmp", ".webp")):
    """For every photo in in_dir, write its SMIRK camera + crop->full transform to out_json, so the
    studio viewer can project the (hashed) mesh onto the photo client-side and rasterize conditioning.

    Per stem: {"cam": [s,tx,ty], "crop_to_full": 3x3 row-major (maps 224-crop px -> full px),
               "w": img_w, "h": img_h}.
    """
    import json
    cams, missed = {}, []
    for f in sorted(os.listdir(in_dir)):
        stem, ext = os.path.splitext(f)
        if ext.lower() not in exts:
            continue
        enc = encode_with_tform(h, os.path.join(in_dir, f))
        if enc is None:
            missed.append(stem); continue
        out, tform, img = enc
        H, W = img.shape[:2]
        cams[stem] = {"cam": [float(c) for c in out["cam"][0].cpu().numpy()],
                      "crop_to_full": np.linalg.inv(tform.params).tolist(),   # crop px -> full px (affine 3x3)
                      "w": int(W), "h": int(H)}
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    json.dump(cams, open(out_json, "w"), indent=0)
    print(f"wrote {out_json}: {len(cams)} camera(s)" + (f"; no face: {', '.join(missed)}" if missed else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Stage-4 bridge: project the SMIRK-posed mesh onto the photo.")
    ap.add_argument("image", nargs="?", help="single photo -> alignment-check overlay")
    ap.add_argument("-o", "--out", default=None, help="overlay PNG (default: alongside the image)")
    ap.add_argument("--cams", nargs="?", const="", metavar="IN_DIR",
                    help="batch: write cams.json for every photo in IN_DIR (default local/in)")
    ap.add_argument("--cams-out", default=os.path.join(os.path.dirname(__file__), "out", "cams.json"),
                    help="cams.json path (default local/out/cams.json)")
    ap.add_argument("--maps", action="store_true",
                    help="rasterize depth + inpaint mask (+ --normal) aligned to the photo (ControlNet inputs)")
    ap.add_argument("--shape", help="300-d shape .npy (MICA / hashed) to impose instead of SMIRK's own")
    ap.add_argument("--normal", action="store_true", help="also write a view-space normal map (best-effort)")
    ap.add_argument("--out-dir", help="where to write the maps (default: alongside the image)")
    ap.add_argument("--feather", type=float, default=0.0, help="Gaussian feather (sigma px) on the mask edge")
    ap.add_argument("--flip-cull", action="store_true", help="flip backface culling (if you get the back of the head)")
    ap.add_argument("--device", default="cpu", help="cpu (default) | mps | auto")
    a = ap.parse_args()

    if a.cams is not None:                                # batch camera export
        in_dir = os.path.abspath(a.cams or os.path.join(os.path.dirname(__file__), "in"))
        out_json = os.path.abspath(a.cams_out)
        h = smirk.load(device=a.device)
        cams_batch(h, in_dir, out_json)
    elif a.maps and a.image:                              # rasterize ControlNet maps (single image or a folder)
        path = os.path.abspath(a.image)
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        targets = ([os.path.join(path, f) for f in sorted(os.listdir(path))
                    if os.path.splitext(f)[1].lower() in exts] if os.path.isdir(path) else [path])
        shape = np.load(a.shape) if a.shape else None
        h = smirk.load(device=a.device)
        for t in targets:
            render_maps(h, t, out_dir=a.out_dir, shape=shape, feather=a.feather,
                        want_normal=a.normal, flip_cull=a.flip_cull)
    elif a.image:                                        # single overlay
        img = os.path.abspath(a.image)
        out_png = os.path.abspath(a.out) if a.out else os.path.splitext(img)[0] + "_overlay.png"
        h = smirk.load(device=a.device)
        overlay(h, img, out_png)
    else:
        ap.error("give a photo (overlay), --maps (ControlNet maps), or --cams [IN_DIR] (camera export)")
