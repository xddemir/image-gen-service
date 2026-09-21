"""Real generation with diffusers, on a GPU.

The load path for the 360 skybox model is the risky part of this file, so it is
worth stating plainly:

`ProGamerGov/sdxl-360-diffusion` ships **no `model_index.json`**. Calling
`StableDiffusionXLPipeline.from_pretrained()` on that repo id fails. What it
does ship is a diffusers-format `unet/` subfolder, so the pipeline is built
from base SDXL and has that UNet swapped in.

That also means `skybox` and `image` share everything except the UNet -- same
VAE, text encoders, tokenizers, scheduler.

The model card publishes no diffusers example, so this is inferred from the
repository layout rather than documented. `from_single_file` on
`sdxl_360_diffusion.safetensors` is the fallback if it turns out to be wrong.

`plan_pipeline` is separated out so the decision of *what* to load is testable
without torch installed; only `_build` needs a GPU.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..config import AppConfig, ConfigError
from ..generator import BaseGenerator
from ..types import GenerateRequest, ResolvedParams

if TYPE_CHECKING:  # pragma: no cover
    from PIL.Image import Image

DTYPES = ("float16", "bfloat16", "float32")


@dataclass
class PipelinePlan:
    """What would be loaded for a kind, decided without importing torch."""

    kind: str
    base_repo: str
    base_revision: str | None = None
    unet_repo: str | None = None
    unet_subfolder: str | None = None
    unet_revision: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def swaps_unet(self) -> bool:
        return self.unet_repo is not None


def plan_pipeline(config: AppConfig, kind_name: str) -> PipelinePlan:
    """Decide what to load for `kind_name`, and flag anything suspect."""
    kind = config.kind(kind_name)
    if kind.model is None:
        raise ConfigError(
            f"kind {kind_name!r} has no model configured, so it cannot be "
            "generated with the local_diffusers backend"
        )

    plan = PipelinePlan(
        kind=kind_name,
        base_repo=kind.model.base,
        base_revision=kind.model.base_revision,
        unet_repo=kind.model.unet_repo,
        unet_subfolder=kind.model.unet_subfolder if kind.model.unet_repo else None,
        unet_revision=kind.model.unet_revision,
    )

    # An unpinned revision still runs, but it silently tracks whatever the repo's
    # main branch points at today -- which quietly breaks the reproducibility
    # claim this whole service exists to support.
    if plan.base_revision is None:
        plan.warnings.append(
            f"{plan.base_repo} has no pinned revision; run `image-gen "
            "download-weights` so results stay reproducible"
        )
    if plan.swaps_unet and plan.unet_revision is None:
        plan.warnings.append(f"{plan.unet_repo} has no pinned revision")

    return plan


class LocalDiffusersGenerator(BaseGenerator):
    backend_name = "local_diffusers"

    def __init__(self, config: AppConfig) -> None:
        super().__init__(config)

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on install
            raise ConfigError(
                "backend 'local_diffusers' needs the GPU extra: "
                "pip install -e '.[local]'"
            ) from exc

        self._torch = torch
        self._pipelines: "OrderedDict[str, Any]" = OrderedDict()

        options = config.backend_options
        if options.dtype not in DTYPES:
            raise ConfigError(
                f"backend_options.dtype must be one of {', '.join(DTYPES)}, "
                f"got {options.dtype!r}"
            )
        self._dtype = getattr(torch, options.dtype)
        self._max_loaded = options.max_loaded_pipelines
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    # -- pipeline cache ---------------------------------------------------

    def loaded_kinds(self) -> list[str]:
        return list(self._pipelines)

    def _pipeline(self, kind_name: str) -> Any:
        """Lazily load, LRU-evict, and reuse pipelines.

        An SDXL pipeline is ~10-12 GB of VRAM, so holding one per kind will not
        fit on most cards. `max_loaded_pipelines` defaults to 1: switching kinds
        evicts the previous pipeline and pays a reload rather than an OOM.
        """
        if kind_name in self._pipelines:
            self._pipelines.move_to_end(kind_name)
            return self._pipelines[kind_name]

        pipeline = self._build(kind_name)
        self._pipelines[kind_name] = pipeline

        while len(self._pipelines) > self._max_loaded:
            _, evicted = self._pipelines.popitem(last=False)
            del evicted
            if self._device == "cuda":
                self._torch.cuda.empty_cache()

        return pipeline

    def _build(self, kind_name: str) -> Any:
        from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel

        plan = plan_pipeline(self.config, kind_name)
        for warning in plan.warnings:
            print(f"warning: {warning}")

        kwargs: dict[str, Any] = {"torch_dtype": self._dtype}

        if plan.swaps_unet:
            # The 360 repo has no model_index.json; only this subfolder is
            # loadable, and it is the entire difference from base SDXL.
            kwargs["unet"] = UNet2DConditionModel.from_pretrained(
                plan.unet_repo,
                subfolder=plan.unet_subfolder,
                revision=plan.unet_revision,
                torch_dtype=self._dtype,
            )

        # The fp16 variant halves VRAM and load time, but not every repo or
        # mirror carries it. Fall back rather than failing the whole run.
        try:
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                plan.base_repo,
                revision=plan.base_revision,
                variant="fp16" if self._dtype is self._torch.float16 else None,
                **kwargs,
            )
        except (OSError, ValueError):
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                plan.base_repo, revision=plan.base_revision, **kwargs
            )

        return pipeline.to(self._device)

    # -- generation -------------------------------------------------------

    def _render(self, req: GenerateRequest, params: ResolvedParams) -> "Image":
        pipeline = self._pipeline(req.kind)

        # Seeded on the generation device. Note this reproduces exactly only on
        # the same GPU model, driver and library versions -- which is why all
        # of those are written into the metadata sidecar.
        generator = self._torch.Generator(device=self._device).manual_seed(req.seed)

        output = pipeline(
            prompt=params.final_prompt,
            # "" and None mean the same thing to diffusers; pass None so the
            # unconditional embedding path is used rather than encoding "".
            negative_prompt=params.negative_prompt or None,
            width=params.width,
            height=params.height,
            num_inference_steps=params.steps,
            guidance_scale=params.guidance,
            generator=generator,
        )
        return output.images[0]

    def _extra_metadata(self, req: GenerateRequest) -> dict[str, Any]:
        pipeline = self._pipelines.get(req.kind)
        metadata: dict[str, Any] = {
            "dtype": str(self._dtype).replace("torch.", ""),
            "device": self._device,
        }

        if pipeline is not None:
            metadata["scheduler"] = type(pipeline.scheduler).__name__

        if self._device == "cuda":
            try:
                metadata["gpu"] = self._torch.cuda.get_device_name(0)
                metadata["driver_cuda"] = self._torch.version.cuda
            except Exception:  # noqa: BLE001 - metadata must never break a run
                pass

        return metadata
