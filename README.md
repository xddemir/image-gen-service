# image-gen

Turns prompts into reproducible image files on disk.

Part of an M.Sc. thesis at DFKI, *Personality-Based Generation of VR Environments for Relaxation*.
The full pipeline is:

```
Persona (Big Five) → LLM → SceneSpec → image-gen → files on disk → Unity (HTC Vive Cosmos)
```

This package owns one link in that chain: **prompt + seed → image file + metadata.**

## What it does

For each requested asset it:

1. resolves parameters (request → kind defaults),
2. builds the final prompt, prepending the kind's trigger phrase if absent,
3. runs the model with the given seed,
4. writes the PNG, plus a sidecar JSON recording everything needed to regenerate it,
5. records the outcome — **including failures** — in a manifest.

One failed asset never aborts a batch.

## What it deliberately does not do

- run the LLM, or decide *what* to generate — prompts arrive already written
- talk to Unity, or send files anywhere
- place assets in a scene (that is `scene_gen`, a later phase in this same repo)

---

## Install

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[api,dev]"   # Windows
# .venv/bin/python -m pip install -e ".[api,dev]"     # Linux/macOS
```

`torch` is **not** a core dependency. The default `stub` backend runs anywhere with no GPU, which is
what makes the whole contract testable locally. GPU support installs separately:

| Extra | Contents | When |
|---|---|---|
| *(core)* | pydantic, pyyaml, pillow, typer | always |
| `[api]` | fastapi, uvicorn | to serve HTTP |
| `[local]` | torch, diffusers, transformers, accelerate | on the GPU machine |
| `[dev]` | pytest, httpx | to run the tests |

## Quickstart

```bash
# one asset, while iterating on a prompt
image-gen gen --kind skybox --prompt "a calm misty forest at dawn" --seed 1234 --scene-id p07
# -> out/p07/p07_skybox_1234.png

# a batch
image-gen run examples/queue.json
image-gen merge-manifest out
image-gen status out

# in a browser: type prompts into the Swagger form
uvicorn image_gen.api.app:app --reload     # -> http://localhost:8000/docs
```

With `backend: stub` no model runs. The placeholder is *drawn*, with the kind, scene, seed, resolved
parameters and final prompt rendered onto it at the correct resolution — so you can confirm the right
file landed in the right place, that a skybox really is 2:1, and that the trigger phrase was applied,
all without a GPU. Everything else — paths, metadata, manifests, failure records — is identical to a
real run.

---

# Contract reference

Everything below is the stable interface. A separate service can produce and consume these formats
without reading any of the Python.

## Output contract

```
out/<scene_id>/<scene_id>_<kind>_<seed>.png
out/<scene_id>/<scene_id>_<kind>_<seed>.json    # sidecar metadata
out/<scene_id>/<scene_id>_manifest.json         # per-scene
out/manifest.json                               # merged, Unity-facing
```

```
out/
  p07/
    p07_skybox_1234.png     p07_skybox_1234.json
    p07_image_5678.png      p07_image_5678.json
    p07_manifest.json
  p08/
    p08_skybox_9012.png     ...
  manifest.json
```

`scene_id` holds the **participant ID**. It names the folder *and* prefixes every file inside it.

The prefix duplicates the folder name on purpose: a file dragged into Unity, attached to an email or
opened in a viewer loses its folder context, and `skybox_1234.png` would then say nothing about who
it belongs to. `scene_id` stays a generic name so fallback, pilot and test scenes (`fallback_forest`,
`test01`) use the same structure without pretending to be participants.

One scene per participant. Regenerating overwrites; bump the seed to keep both.

**`scene_id` must match `^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$`** — it becomes a path, so it must not be
able to escape `out/`. Invalid ids are rejected at validation time, not at write time.

## Metadata sidecar

Written for **every** attempt, successful or not.

```json
{
  "scene_id": "p07",
  "kind": "skybox",
  "status": "ok",
  "prompt": "a calm misty forest clearing at dawn",
  "final_prompt": "equirectangular 360 view, a calm misty forest clearing at dawn",
  "negative_prompt": "people, text, watermark, buildings, distorted horizon",
  "seed": 1234,
  "model": {
    "base": "stabilityai/stable-diffusion-xl-base-1.0",
    "base_revision": "462165...",
    "unet_repo": "ProGamerGov/sdxl-360-diffusion",
    "unet_revision": "3565852..."
  },
  "width": 2048, "height": 1024, "steps": 30, "guidance": 7.0,
  "backend": "local_diffusers",
  "versions": { "image_gen": "0.1.0", "torch": "2.4.0", "diffusers": "0.31.0" },
  "file": "p07_skybox_1234.png",
  "timestamp": "2026-09-21T14:30:02Z",
  "duration_s": 42.1
}
```

| Field | Notes |
|---|---|
| `status` | `"ok"` or `"failed"` |
| `prompt` | what the caller supplied, unmodified |
| `final_prompt` | what the model actually received, after prefixing |
| `model` | `null` for kinds with no model configured |
| `width`…`guidance` | **resolved** values, not requested ones |
| `versions` | a seed only reproduces on matching library versions — see below |
| `file` | bare filename, or `null` when failed |
| `error`, `traceback` | present only when `status` is `"failed"` |

**Reproducibility caveat.** An identical seed reproduces an identical image only on the same GPU
model, driver and torch/diffusers versions. That is why they are recorded rather than assumed.

## Manifests

Two levels. Per-scene manifests are written **only by the job that owns the scene**, so a Slurm job
array needs no file locking.

**`out/<scene_id>/<scene_id>_manifest.json`** — bare filenames:

```json
{
  "scene_id": "p07",
  "created": "2026-09-21T14:30:02Z",
  "assets": [
    { "kind": "skybox", "seed": 1234, "file": "p07_skybox_1234.png", "status": "ok" },
    { "kind": "image",  "seed": 5678, "file": null, "status": "failed",
      "error": "RuntimeError: CUDA out of memory" }
  ]
}
```

**`out/manifest.json`** — merged, keyed by scene, paths relative to `out/` with forward slashes
(the file may be written on Linux and read on Windows):

```json
{
  "p07": [
    { "kind": "skybox", "file": "p07/p07_skybox_1234.png", "status": "ok" },
    { "kind": "image",  "file": null, "status": "failed", "error": "..." }
  ]
}
```

Seeds are not repeated in the merged manifest — read them from the filename or the sidecar.

**Failures are first-class.** A failed asset writes no PNG but still writes a sidecar and a manifest
entry with `"file": null`. A *missing* entry is ambiguous; a `failed` entry is not, and Unity needs
that difference to choose a fallback.

## `queue.json`

One entry per scene, each holding several assets — mirroring the manifest.

```json
{
  "version": 1,
  "defaults": { "steps": 30, "guidance": 7.0 },
  "scenes": [
    {
      "scene_id": "p07",
      "seed": 1234,
      "assets": [
        { "kind": "skybox", "prompt": "a calm misty forest clearing at dawn" },
        { "kind": "image",  "prompt": "a smooth river stone", "seed": 5678 }
      ]
    }
  ]
}
```

| Level | Required | Optional |
|---|---|---|
| top | `scenes` | `version`, `defaults` |
| scene | `scene_id`, `assets` | `seed`, and any parameter override |
| asset | `kind`, `prompt` | `seed`, and any parameter override |

Parameter overrides are `width`, `height`, `steps`, `guidance`, `negative_prompt`.

**Precedence:** `asset → scene → queue defaults → kind defaults`.

A seed must be resolvable for every asset, from the asset or its scene — generation has to be
reproducible, so there is no implicit random seed.

**Unknown fields are rejected.** A typo like `"promt"` fails validation rather than being silently
ignored and discovered after the cluster queue.

## Configuration

`configs/default.yaml`. Asset kinds are defined here and nowhere else — adding one is a config
change, with no code touched in the CLI, the API or the generators.

```yaml
backend: stub            # stub | local_diffusers | slurm_remote
out_dir: out

backend_options:
  dtype: float16
  max_loaded_pipelines: 1     # evict the previous pipeline when switching kinds
  enable_cpu_offload: false   # reserved, not implemented
  attention_slicing: false    # reserved, not implemented
  vae_tiling: false           # reserved, not implemented

kinds:
  skybox:
    model:
      base: stabilityai/stable-diffusion-xl-base-1.0
      base_revision: null                        # pinned by download-weights
      unet_repo: ProGamerGov/sdxl-360-diffusion
      unet_subfolder: unet
      unet_revision: 35658524f3...
    prompt_prefix: "equirectangular 360 view, "
    prompt_prefix_markers: ["equirectangular 360 view", "360 panorama"]
    aspect_ratio: 2.0
    defaults:
      width: 2048
      height: 1024
      steps: 30
      guidance: 7.0
      negative_prompt: "people, text, watermark, buildings, distorted horizon"
    # stub_file: assets/stub/skybox.png    # optional: real image instead of drawn
```

| Key | Meaning |
|---|---|
| `prompt_prefix` | trigger phrase, prepended **only if absent** |
| `prompt_prefix_markers` | phrases that count as "already present"; defaults to the prefix itself |
| `aspect_ratio` | enforced at load and per request; `2.0` means 2:1 |
| `stub_file` | optional real image for the stub, resized to the requested dimensions |

The three reserved `backend_options` raise a clear error if enabled, rather than being silently
ignored — watching a job OOM after setting `enable_cpu_offload` would be worse.

### Pinned revisions

`base_revision` is `null` in the committed config on purpose: **no revision is guessed.**
`image-gen download-weights` resolves the current commits and writes
`configs/revisions.lock.json`:

```json
{ "stabilityai/stable-diffusion-xl-base-1.0": "462165...",
  "ProGamerGov/sdxl-360-diffusion": "35658524f3..." }
```

The lock fills in only revisions left `null`; an explicit revision in the YAML always wins. It is a
separate file because rewriting the YAML would strip its comments, and those comments record *why*
the UNet is loaded the way it is.

### Adding an asset kind

Add a block under `kinds:`. Nothing else changes — `POST /generate/<newkind>` and
`--kind <newkind>` start working, and `texture` is already reserved as a stub entry.

---

## CLI

| Command | Purpose |
|---|---|
| `gen --kind K --prompt P --seed N [--scene-id ID]` | one asset; `--scene-id` defaults to a timestamp |
| `run queue.json` | a batch; never aborts on a single asset failure |
| `merge-manifest [out]` | combine per-scene manifests into `out/manifest.json` |
| `status [out]` | what is on disk, what succeeded, what failed |
| `validate queue.json` | check against the config, generate nothing |
| `download-weights` | pre-fetch weights and pin revisions (needs internet) |

Shared options: `--config/-c`, `--backend`, `--out`.

`run` also takes:

| Flag | Purpose |
|---|---|
| `--shard-index N --shard-count M` | partition scenes across a Slurm array; each task owns a disjoint set, which is what makes per-scene manifests lock-free |
| `--skip-existing` | resume a partially failed run without redoing GPU work |
| `--dry-run` | print what would be written |

### Exit codes

| Code | Meaning |
|---|---|
| `0` | everything succeeded |
| `1` | fatal — bad config, malformed queue, nothing ran |
| `2` | the run completed, but some assets failed |

`2` is distinguishable from `1` so a wrapper can tell "most of it worked" from "nothing did". On
Slurm the merge job runs with `afterany`, so a non-zero task never blocks the manifest being built.

---

## HTTP API

```bash
uvicorn image_gen.api.app:app            # /docs is the live reference
```

Config path comes from `$IMAGE_GEN_CONFIG`, else `configs/default.yaml`.

| Route | Purpose |
|---|---|
| `POST /generate/{kind}` | → `202` + job. `404` if the kind is not configured |
| `GET /jobs/{id}` | `queued` → `running` → `done` \| `failed` |
| `GET /files/{path}` | serve a generated file |
| `GET /health` | backend, configured/loaded kinds, queue depth, versions |

```jsonc
// POST /generate/skybox
{ "prompt": "a forest with blue trees",   // required
  "seed": 1234,                           // required -- reproducibility
  "scene_id": "p07",                      // optional, defaults to scene_<timestamp>
  "negative_prompt": null, "width": null, "height": null,
  "steps": null, "guidance": null }       // null -> kind default
```

One parameterised route, not one per kind: `/generate/skybox` and `/generate/image` both work, and a
new kind needs no API code.

Notes for a client:

- **`202`, not `200`** — generation is queued, not done. Poll `GET /jobs/{id}`.
- **`job.file` is relative to `out/`**, ready to append to `/files/`.
- **Job state is in memory and lost on restart.** Deliberate: the manifest on disk is the durable
  record. A client that needs history should read manifests, not jobs.
- **One job at a time.** A single worker thread, which is what serialises GPU access.
- `422` for an invalid `scene_id`, an aspect-ratio violation, or an unknown field — all caught before
  the job is queued, so you learn immediately rather than a minute later.
- `GET /files/` refuses any path escaping `out/`.

---

## Backends

One interface, swappable implementations. Callers never know which ran.

```python
class ImageGenerator(Protocol):
    def generate(self, req: GenerateRequest) -> GenerateResult: ...
    def loaded_kinds(self) -> list[str]: ...
```

| Backend | Runs where | Status |
|---|---|---|
| `stub` | anywhere, no GPU | **implemented** — the default |
| `local_diffusers` | on a GPU node | **implemented**, not yet verified on a GPU |
| `slurm_remote` | your machine; submits + fetches | planned |

`BaseGenerator` owns everything except the pixels — parameter resolution, prompt prefixing, paths,
metadata, failure handling. A backend implements only:

```python
class MyGenerator(BaseGenerator):
    backend_name = "mine"

    def _render(self, req: GenerateRequest, params: ResolvedParams) -> PIL.Image.Image:
        ...

    def _extra_metadata(self, req) -> dict:   # optional: scheduler, dtype, device
        return {}
```

then a branch in `build_generator`. This is why the stub is a useful test of the real thing: they
differ only in how pixels are produced.

### Why the skybox model loads the way it does

`ProGamerGov/sdxl-360-diffusion` ships **no `model_index.json`**, so
`StableDiffusionXLPipeline.from_pretrained()` on that repo ID **fails**. Only `unet/` is in diffusers
format. The pipeline is therefore built from base SDXL with the 360 UNet swapped in:

```python
unet = UNet2DConditionModel.from_pretrained(unet_repo, subfolder="unet", revision=...)
pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, revision=..., variant="fp16")
```

A useful consequence: `skybox` and `image` differ *only* in the UNet, sharing VAE, text encoders and
scheduler — fewer bytes to pre-download, and room to swap `pipe.unet` instead of holding two
pipelines. The model card publishes no diffusers example, so this is inferred from the repo layout
and **must be confirmed on the first GPU run**; `from_single_file` on
`sdxl_360_diffusion.safetensors` is the fallback.

---

## Library use

```python
from image_gen import load_config, build_generator, GenerateRequest

config = load_config("configs/default.yaml")
generator = build_generator(config)

result = generator.generate(
    GenerateRequest(kind="skybox", prompt="a calm forest", seed=1234, scene_id="p07")
)
if result.ok:
    print(result.path, result.metadata["final_prompt"])
```

`generate()` returns a `GenerateResult` rather than a bare `Path` because the sidecar must record
what *actually ran*. If it returned only a path, every caller would re-derive the resolved
parameters and the two would drift.

---

## Tests

```bash
.venv/Scripts/python -m pytest
```

71 tests, no GPU required. They cover parameter precedence, prompt-prefix idempotency, the filename
and folder contract, the failure path, manifest merging, queue validation, shard disjointness, and
path-traversal refusal.

## Status

| Step | | |
|---|---|---|
| 1 | core, kind config, stub backend, metadata | done |
| 2 | CLI: `gen`, `run`, `status`, `merge-manifest`, `validate` | done |
| 3 | FastAPI wrapper | done |
| 5 | `LocalDiffusersGenerator` | code done, awaiting first GPU run |
| 6 | sbatch scripts for Pegasus | next |
| 7 | `SlurmRemoteGenerator` + gateway | |
| 8 | `scene_gen` — SceneSpec → prompts + Unity payload | |
