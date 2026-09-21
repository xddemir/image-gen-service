"""The generator interface, parameter resolution and the shared pipeline.

Every backend produces pixels differently, but they must all write identical
paths, identical metadata and identical failure records -- otherwise the stub
tests nothing useful about the real thing. So `BaseGenerator` owns everything
except the pixels, and backends implement only `_render`.
"""

from __future__ import annotations

import time
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .config import AppConfig, AssetKindConfig, ConfigError
from .outputs import (
    image_path,
    library_versions,
    sidecar_path,
    utc_now_iso,
    validate_scene_id,
    write_json,
)
from .types import AssetStatus, GenerateRequest, GenerateResult, ResolvedParams

if TYPE_CHECKING:  # pragma: no cover
    from PIL.Image import Image


@runtime_checkable
class ImageGenerator(Protocol):
    """One interface, swappable backends. Callers never know which one ran."""

    def generate(self, req: GenerateRequest) -> GenerateResult: ...

    def loaded_kinds(self) -> list[str]:
        """Kinds that can be served right now without paying a load cost."""
        ...


def apply_prompt_prefix(prompt: str, kind: AssetKindConfig) -> str:
    """Prepend the kind's trigger phrase, but only once.

    Idempotent because the prompt may arrive from an LLM that was already told
    to write "equirectangular 360 view". Prefixing unconditionally would
    produce "equirectangular 360 view, equirectangular 360 view, ..." and
    quietly degrade the result.
    """
    if not kind.prompt_prefix:
        return prompt
    lowered = prompt.lower()
    for marker in kind.markers():
        if marker and marker.lower() in lowered:
            return prompt
    return f"{kind.prompt_prefix}{prompt}"


def resolve_params(config: AppConfig, req: GenerateRequest) -> ResolvedParams:
    """Apply defaults. The single place precedence is decided.

    Per-request values win over the kind's defaults; the batch runner has
    already collapsed per-asset / per-scene / queue defaults into the request
    before it gets here.
    """
    kind = config.kind(req.kind)
    width = req.width if req.width is not None else kind.defaults.width
    height = req.height if req.height is not None else kind.defaults.height

    # `is not None`, not truthiness: an explicit empty negative prompt is a
    # legitimate choice and must not silently fall back to the default.
    negative = (
        req.negative_prompt
        if req.negative_prompt is not None
        else kind.defaults.negative_prompt
    )

    kind.check_size(width, height)

    return ResolvedParams(
        width=width,
        height=height,
        steps=req.steps if req.steps is not None else kind.defaults.steps,
        guidance=req.guidance if req.guidance is not None else kind.defaults.guidance,
        negative_prompt=negative,
        final_prompt=apply_prompt_prefix(req.prompt, kind),
    )


class BaseGenerator(ABC):
    """Shared machinery: resolve, render, save, record.

    Failures are caught here rather than raised, because one bad asset must not
    abort a batch that may represent hours of queued GPU time. The failure is
    written to disk -- sidecar and manifest entry -- so it stays visible.
    """

    backend_name = "base"

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.out_dir = Path(config.out_dir)

    # -- to implement in a backend ---------------------------------------

    @abstractmethod
    def _render(self, req: GenerateRequest, params: ResolvedParams) -> "Image":
        """Produce the actual image. The only backend-specific step."""

    def _extra_metadata(self, req: GenerateRequest) -> dict[str, Any]:
        """Backend-specific metadata fields (scheduler, dtype, device...)."""
        return {}

    def loaded_kinds(self) -> list[str]:
        return []

    # -- the contract ----------------------------------------------------

    def generate(self, req: GenerateRequest) -> GenerateResult:
        validate_scene_id(req.scene_id)
        kind = self.config.kind(req.kind)  # raises ConfigError on unknown kind
        params = resolve_params(self.config, req)

        png = image_path(self.out_dir, req.scene_id, req.kind, req.seed)
        json_path = sidecar_path(self.out_dir, req.scene_id, req.kind, req.seed)

        started = time.monotonic()
        try:
            image = self._render(req, params)
            png.parent.mkdir(parents=True, exist_ok=True)
            image.save(png)
        except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
            duration = time.monotonic() - started
            metadata = self._metadata(
                req, params, kind, AssetStatus.FAILED, None, duration
            )
            metadata["error"] = f"{type(exc).__name__}: {exc}"
            metadata["traceback"] = traceback.format_exc()
            write_json(json_path, metadata)
            return GenerateResult(
                status=AssetStatus.FAILED,
                metadata=metadata,
                path=None,
                error=metadata["error"],
            )

        duration = time.monotonic() - started
        metadata = self._metadata(req, params, kind, AssetStatus.OK, png, duration)
        write_json(json_path, metadata)
        return GenerateResult(status=AssetStatus.OK, metadata=metadata, path=png)

    def _metadata(
        self,
        req: GenerateRequest,
        params: ResolvedParams,
        kind: AssetKindConfig,
        status: AssetStatus,
        png: Path | None,
        duration: float,
    ) -> dict[str, Any]:
        """Everything needed to regenerate this exact asset, or to explain why
        it could not be produced."""
        metadata: dict[str, Any] = {
            "scene_id": req.scene_id,
            "kind": req.kind,
            "status": status.value,
            "prompt": req.prompt,
            "final_prompt": params.final_prompt,
            "negative_prompt": params.negative_prompt,
            "seed": req.seed,
            "model": kind.model.to_metadata() if kind.model else None,
            "width": params.width,
            "height": params.height,
            "steps": params.steps,
            "guidance": params.guidance,
            "backend": self.backend_name,
            "versions": library_versions(),
            "file": png.name if png else None,
            "timestamp": utc_now_iso(),
            "duration_s": round(duration, 3),
        }
        metadata.update(self._extra_metadata(req))
        return metadata


def build_generator(config: AppConfig) -> ImageGenerator:
    """Pick a backend from config. Imports are lazy so that the stub path never
    pulls in torch."""
    name = config.backend

    if name == "stub":
        from .backends.stub import StubGenerator

        return StubGenerator(config)

    if name == "local_diffusers":
        from .backends.diffusers_local import LocalDiffusersGenerator

        return LocalDiffusersGenerator(config)

    if name == "slurm_remote":
        raise ConfigError(
            "backend 'slurm_remote' is not implemented yet (build order step 6)"
        )

    raise ConfigError(
        f"unknown backend {name!r}; expected one of: stub, local_diffusers, slurm_remote"
    )
