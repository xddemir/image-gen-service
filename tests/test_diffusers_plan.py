"""What the local_diffusers backend would load, decided without torch.

The UNet swap is the one inference in this codebase that is not documented by
the model card, so the decision is separated from the loading and pinned down
here. These tests need no GPU and no torch.
"""

from __future__ import annotations

import pytest

from image_gen.backends.diffusers_local import plan_pipeline
from image_gen.config import ConfigError


def test_skybox_swaps_the_360_unet_into_base_sdxl(config):
    """ProGamerGov/sdxl-360-diffusion has no model_index.json, so it must be
    loaded as a UNet into a base SDXL pipeline, never as a pipeline itself."""
    plan = plan_pipeline(config, "skybox")

    assert plan.swaps_unet
    assert plan.base_repo == "stabilityai/stable-diffusion-xl-base-1.0"
    assert plan.unet_repo == "ProGamerGov/sdxl-360-diffusion"
    assert plan.unet_subfolder == "unet"


def test_flat_image_loads_base_sdxl_unmodified(config):
    plan = plan_pipeline(config, "image")

    assert not plan.swaps_unet
    assert plan.unet_repo is None
    assert plan.unet_subfolder is None


def test_both_kinds_share_the_same_base(config):
    """They differ only in the UNet. If that ever stops being true, the
    download planning and the VRAM story both need revisiting."""
    assert plan_pipeline(config, "skybox").base_repo == (
        plan_pipeline(config, "image").base_repo
    )


def test_unpinned_base_revision_is_warned_about(config):
    """Unpinned still runs, but it tracks whatever main points at today, which
    silently breaks the reproducibility claim."""
    config.kinds["skybox"].model.base_revision = None
    plan = plan_pipeline(config, "skybox")

    assert any("no pinned revision" in w for w in plan.warnings)
    assert any("download-weights" in w for w in plan.warnings)


def test_unpinned_unet_revision_is_warned_about(config):
    config.kinds["skybox"].model.unet_revision = None
    assert any(
        "sdxl-360-diffusion" in w for w in plan_pipeline(config, "skybox").warnings
    )


def test_fully_pinned_plan_is_silent(config):
    config.kinds["skybox"].model.base_revision = "462165984030d82259a11f4367a4eed129e94a7b"
    assert plan_pipeline(config, "skybox").warnings == []


def test_kind_without_a_model_is_rejected(config):
    """`texture` is reserved in config with no model; asking the GPU backend
    for it should say so rather than fail deep inside diffusers."""
    config.kinds["image"].model = None
    with pytest.raises(ConfigError, match="no model configured"):
        plan_pipeline(config, "image")


def test_unknown_kind_is_rejected(config):
    with pytest.raises(ConfigError, match="unknown asset kind"):
        plan_pipeline(config, "nope")


@pytest.mark.parametrize("dtype", ["float64", "int8", "fp16", ""])
def test_invalid_dtype_is_rejected(config, dtype):
    pytest.importorskip("torch")
    from image_gen.backends.diffusers_local import LocalDiffusersGenerator

    config.backend_options.dtype = dtype
    with pytest.raises(ConfigError, match="dtype"):
        LocalDiffusersGenerator(config)
