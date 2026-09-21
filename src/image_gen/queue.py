"""queue.json: the batch input format.

One entry per scene, each holding several assets -- mirroring the manifest, so
the input and the output are shaped the same way.

Parameter precedence, applied here and nowhere else:

    per-asset  ->  per-scene  ->  queue defaults  ->  kind defaults

The first three are collapsed into the GenerateRequest; kind defaults are
applied later by `resolve_params`, so a request that reaches a backend already
carries every override the queue expressed.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .config import AppConfig
from .outputs import SCENE_ID_RE
from .types import GenerateRequest


class QueueError(Exception):
    """Raised for a malformed or internally inconsistent queue file."""


class ParamOverrides(BaseModel):
    """Optional generation parameters. `None` means "defer to the next level"."""

    # Typos in a queue file are expensive: the job runs, the asset is wrong,
    # and you find out after the cluster queue. Reject unknown keys instead.
    model_config = ConfigDict(extra="forbid")

    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    steps: int | None = Field(default=None, gt=0)
    guidance: float | None = Field(default=None, ge=0)
    negative_prompt: str | None = None


class QueueAsset(ParamOverrides):
    kind: str
    prompt: str
    seed: int | None = None


class QueueScene(ParamOverrides):
    scene_id: str
    seed: int | None = None
    assets: list[QueueAsset]


class Queue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    defaults: ParamOverrides = Field(default_factory=ParamOverrides)
    scenes: list[QueueScene]


def load_queue(path: Path | str) -> Queue:
    path = Path(path)
    if not path.is_file():
        raise QueueError(f"queue file not found: {path}")
    try:
        # utf-8-sig, not utf-8: Windows tooling (PowerShell, Notepad) writes a
        # UTF-8 BOM, and a queue file is exactly the kind of thing that gets
        # hand-edited or emitted by a script there. utf-8-sig reads both.
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise QueueError(f"{path}: invalid JSON: {exc}") from exc
    try:
        return Queue.model_validate(raw)
    except ValidationError as exc:
        raise QueueError(f"{path}: {exc}") from exc


def validate_queue(queue: Queue, config: AppConfig) -> list[str]:
    """Return every problem found, not just the first.

    Run before any GPU time is spent: a batch of 40 scenes should report all
    its bad kinds at once rather than one per failed cluster job.
    """
    errors: list[str] = []
    seen: set[str] = set()

    if not queue.scenes:
        errors.append("queue contains no scenes")

    for i, scene in enumerate(queue.scenes):
        where = f"scenes[{i}] ({scene.scene_id!r})"

        if not SCENE_ID_RE.match(scene.scene_id or ""):
            errors.append(
                f"{where}: invalid scene_id; use 1-64 characters of "
                "[A-Za-z0-9._-], starting with a letter or digit"
            )
        if scene.scene_id in seen:
            errors.append(
                f"{where}: duplicate scene_id -- two scenes would write to the "
                "same folder and overwrite each other"
            )
        seen.add(scene.scene_id)

        if not scene.assets:
            errors.append(f"{where}: has no assets")

        for j, asset in enumerate(scene.assets):
            aw = f"{where} assets[{j}]"
            if asset.kind not in config.kinds:
                known = ", ".join(sorted(config.kinds))
                errors.append(
                    f"{aw}: unknown kind {asset.kind!r}; configured kinds: {known}"
                )
            if not asset.prompt.strip():
                errors.append(f"{aw}: empty prompt")
            if asset.seed is None and scene.seed is None:
                errors.append(
                    f"{aw}: no seed -- set it on the asset or on the scene "
                    "(generation must be reproducible)"
                )

    return errors


def shard(scenes: list[QueueScene], index: int, count: int) -> list[QueueScene]:
    """Partition scenes across Slurm array tasks.

    Round-robin slicing: every scene lands in exactly one shard, no shard
    overlaps another, and it needs no coordination between tasks. That is what
    makes per-scene manifests safe without file locking.
    """
    if count < 1:
        raise QueueError("shard-count must be >= 1")
    if not 0 <= index < count:
        raise QueueError(f"shard-index must be in [0, {count}), got {index}")
    return scenes[index::count]


def _pick(*values):
    """First non-None value, or None. Encodes the precedence chain."""
    for value in values:
        if value is not None:
            return value
    return None


def build_request(
    scene: QueueScene, asset: QueueAsset, defaults: ParamOverrides
) -> GenerateRequest:
    """Collapse asset / scene / queue-default overrides into one request."""
    seed = _pick(asset.seed, scene.seed)
    if seed is None:
        raise QueueError(
            f"scene {scene.scene_id!r} asset {asset.kind!r}: no seed available"
        )

    return GenerateRequest(
        kind=asset.kind,
        prompt=asset.prompt,
        seed=seed,
        scene_id=scene.scene_id,
        negative_prompt=_pick(
            asset.negative_prompt, scene.negative_prompt, defaults.negative_prompt
        ),
        width=_pick(asset.width, scene.width, defaults.width),
        height=_pick(asset.height, scene.height, defaults.height),
        steps=_pick(asset.steps, scene.steps, defaults.steps),
        guidance=_pick(asset.guidance, scene.guidance, defaults.guidance),
    )
