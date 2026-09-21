"""Core value types passed across the generator boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class AssetStatus(str, Enum):
    """Outcome of a single asset. Recorded in the sidecar and the manifest.

    A failed asset is written down rather than omitted: a missing manifest
    entry is ambiguous, a `failed` entry is not, and Unity needs to tell the
    difference to pick a fallback.
    """

    OK = "ok"
    FAILED = "failed"


@dataclass(frozen=True)
class GenerateRequest:
    """What the caller asks for.

    Every dimension/step/guidance field is optional: `None` means "fall back to
    the kind's configured default". Resolution happens in exactly one place
    (`resolve_params`) so the CLI, the HTTP API and the batch runner cannot
    drift apart.
    """

    kind: str
    prompt: str
    seed: int
    scene_id: str
    negative_prompt: str | None = None
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    guidance: float | None = None


@dataclass(frozen=True)
class ResolvedParams:
    """Exactly what will be handed to the model, after defaults are applied.

    `final_prompt` is the prompt after the kind's trigger phrase has been
    prepended; `prompt` on the request stays the caller's original text. Both
    are recorded, because reproducing a run needs the former and understanding
    it needs the latter.
    """

    width: int
    height: int
    steps: int
    guidance: float
    negative_prompt: str
    final_prompt: str


@dataclass
class GenerateResult:
    """What came back.

    `path` is None when `status` is FAILED -- no PNG is written in that case,
    though the sidecar metadata still is.
    """

    status: AssetStatus
    metadata: dict[str, Any]
    path: Path | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status is AssetStatus.OK


@dataclass
class ManifestEntry:
    """One asset's line in a manifest."""

    kind: str
    seed: int
    status: AssetStatus
    file: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "kind": self.kind,
            "seed": self.seed,
            "file": self.file,
            "status": self.status.value,
        }
        if self.error:
            out["error"] = self.error
        return out


@dataclass
class SceneManifest:
    """Per-scene manifest. Only the job that owns the scene ever writes it,
    which is what makes Slurm job arrays safe without any file locking."""

    scene_id: str
    created: str
    assets: list[ManifestEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "created": self.created,
            "assets": [a.to_dict() for a in self.assets],
        }
