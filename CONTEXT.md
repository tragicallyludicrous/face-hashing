# Face Hashing — Project Context

## What I'm building

See below a lightly-modified version of the original deep-research prompt:

I want to build a personal-project program for exploratory/creative purposes. I call it Face Hashing, and here's the long and short of it:

In: a photograph (eventually video, but let's start with stills) of a person (eventually multiple people).
Out: the same photograph, but the person's face is different.

Here's the twist: the out face should always be the same face given the same input face. Under the hood, the transform should be, as any good hash:

Deterministic
One-way

Here's one way I picture it working, with a limited knowledge of the tools involved:

1. Some sort of facial recognition algorithm (perhaps the kind that can identify a specific face in, say, Apple Photos or the like) outputs the specifics of what makes that face unique. Pupillary distance, jaw shape, whatnot. In my brain this outputs as a JSON object or something.

2. Some sort of mathematical transform that changes that object's values (but not structure) such that they would represent the output of a different face. Ideally we could hotswap this 'hash function' as the project mutates.

3. Some way to turn this object into the rough draft version of the face. I picture something like the Skyrim character generator, but as granular as the data it receives.

4. A diffusion model to make this face photorealistic and composite it back into the original image.

Each step seems like a unique and interesting challenge, and I'd love to know what libraries, approaches, etc I should dig into in order to make this possible. Also open to other approaches (I imagine simply generating a seed from the facial-recognition algo, transforming that, and putting it into a comfyUI workflow might get up and running faster but it doesn't sound as interesting), though this one seems very interesting on many technical levels that I'd like to dig into.

## Stage 1 goal (current)

Photo → DECA → FLAME parameters (the "JSON") → 3D mesh → browser viewer.
No transform yet. Just proving the pipeline.

## Architecture (eventual, 4 stages)

1. Face → structured representation (FLAME params via DECA)
2. Deterministic transform on those params (the "hash function", hot-swappable)
3. Reconstruct a rough face from transformed params
4. Diffusion model for photorealism + composite back into original image

## Environment

- macOS Apple Silicon, no NVIDIA GPU
- Running DECA in Google Colab (T4 free tier) to avoid PyTorch3D install pain on Mac
- Viewer is local: HTML + <model-viewer> served via `python3 -m http.server`

## Repo layout

```
face-hashing/
├── README.md
├── colab/
│   └── stage1_deca.ipynb          # The notebook (we'll build it in Colab and save here)
├── outputs/                        # Where downloaded results go
│   └── .gitkeep
├── viewer/
│   ├── index.html                  # The <model-viewer> page
│   └── models/                     # .glb files go here
└── .gitignore
```

## Where I am right now

- DECA cloned in Colab at /content/DECA
- chumpy patched on disk (sed replaced inspect.getargspec and the broken numpy import)
- face-alignment LandmarksType.\_2D patched to TWO_D via sed
- FLAME 2020 uploaded manually, generic_model.pkl in data/
- deca_model.tar downloaded via gdown, in data/
- Skipping pytorch3d entirely; using --saveVis False
- About to retry: !python demos/demo_reconstruct.py -i TestSamples/examples -s outputs/examples --saveObj True --saveMat True --saveVis False

## Patches applied to DECA's environment (need to re-run after Colab session resets)

- sed -i 's/inspect\.getargspec/inspect.getfullargspec/g' /usr/local/lib/python3.12/dist-packages/chumpy/ch.py
- sed -i 's/from numpy import bool, int, float, complex, object, unicode, str, nan, inf/from numpy import nan, inf/' /usr/local/lib/python3.12/dist-packages/chumpy/**init**.py
- sed -i 's/LandmarksType\.\_2D/LandmarksType.TWO_D/g' /content/DECA/decalib/datasets/detectors.py

## Known design decisions

- DECA over EMOCA/MICA for first pass (most docs, easiest)
- Untextured mesh (gray vertex color) for the Skyrim aesthetic
- <model-viewer> for the browser viewer (one HTML tag, no three.js code)
- .obj → .glb conversion via trimesh (model-viewer doesn't load .obj)

## Open questions / known footguns

- Each !python subprocess in Colab is a fresh interpreter and doesn't inherit
  in-process monkey-patches; patches must be on disk
- Colab is on Python 3.12, well ahead of what DECA expects (3.7-3.10)
- No prebuilt PyTorch3D wheel exists for current Colab (PyTorch 2.10/CUDA 12.8)
- Once Stage 1 works: implement Stage 2 (the transform) as a strategy pattern
  so the algorithm is hot-swappable

## Reference: the two setup-guide artifacts from the Claude.ai conversation

./face_hashing_research_report.md
./face-hashing-setup.md <!-- this is a bit off of the existing workflow as it hasn't been updated as we've worked through the project -->
