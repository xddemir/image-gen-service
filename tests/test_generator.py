"""Parameter resolution, prompt prefixing, and the output contract."""

from __future__ import annotations

import json

import pytest
from PIL import Image

from image_gen.config import ConfigError
from image_gen.generator import apply_prompt_prefix, build_generator, resolve_params
from image_gen.outputs import OutputError
from image_gen.types import AssetStatus, GenerateRequest


def req(**kwargs) -> GenerateRequest:
    base = dict(kind="skybox", prompt="a calm forest", seed=1234, scene_id="p07")
    base.update(kwargs)
    return GenerateRequest(**base)


# -- parameter precedence -------------------------------------------------


def test_kind_defaults_apply_when_request_omits_values(config):
    params = resolve_params(config, req())
    assert (params.width, params.height) == (256, 128)
    assert params.steps == 30
    assert params.guidance == 7.0
    assert params.negative_prompt == "people, text, watermark"


def test_request_values_override_kind_defaults(config):
    params = resolve_params(config, req(width=512, height=256, steps=7, guidance=2.5))
    assert (params.width, params.height) == (512, 256)
    assert params.steps == 7
    assert params.guidance == 2.5


def test_empty_negative_prompt_is_respected_not_replaced(config):
    """An explicit "" means "no negative prompt", not "use the default".

    Truthiness checks get this wrong, which would silently reintroduce the
    kind's negative prompt for a caller who deliberately cleared it.
    """
    params = resolve_params(config, req(negative_prompt=""))
    assert params.negative_prompt == ""


# -- prompt prefixing -----------------------------------------------------


def test_trigger_phrase_is_prepended(config):
    params = resolve_params(config, req(prompt="a calm forest"))
    assert params.final_prompt == "equirectangular 360 view, a calm forest"


@pytest.mark.parametrize(
    "prompt",
    [
        "equirectangular 360 view, a calm forest",
        "Equirectangular 360 View, a calm forest",
        "a calm forest, 360 panorama",
        "an equirectangular shot of a forest",
    ],
)
def test_trigger_phrase_is_not_doubled(config, prompt):
    params = resolve_params(config, req(prompt=prompt))
    assert params.final_prompt == prompt
    assert params.final_prompt.lower().count("equirectangular 360 view") <= 1


def test_kind_without_prefix_leaves_prompt_alone(config):
    params = resolve_params(config, req(kind="image", prompt="a river stone"))
    assert params.final_prompt == "a river stone"


def test_apply_prompt_prefix_is_idempotent(config):
    kind = config.kind("skybox")
    once = apply_prompt_prefix("a calm forest", kind)
    assert apply_prompt_prefix(once, kind) == once


# -- aspect ratio ---------------------------------------------------------


def test_non_2to1_skybox_is_rejected(config):
    with pytest.raises(ConfigError, match="aspect ratio"):
        resolve_params(config, req(width=512, height=512))


def test_aspect_ratio_unconstrained_for_flat_images(config):
    params = resolve_params(config, req(kind="image", width=200, height=100))
    assert (params.width, params.height) == (200, 100)


# -- the output contract --------------------------------------------------


def test_files_land_at_the_contracted_paths(config, out_dir):
    result = build_generator(config).generate(req())

    assert result.ok
    assert result.path == out_dir / "p07" / "p07_skybox_1234.png"
    assert result.path.is_file()

    sidecar = out_dir / "p07" / "p07_skybox_1234.json"
    assert sidecar.is_file()


def test_placeholder_has_the_configured_dimensions(config, out_dir):
    result = build_generator(config).generate(req())
    with Image.open(result.path) as image:
        assert image.size == (256, 128)


def test_regenerating_the_same_request_is_byte_identical(config, out_dir):
    generator = build_generator(config)
    first = generator.generate(req()).path.read_bytes()
    second = generator.generate(req()).path.read_bytes()
    assert first == second


def test_placeholder_colour_depends_only_on_kind_and_seed(config):
    """Derived with hashlib rather than hash(), which Python randomises per
    process. Stability is the whole point: the same seed should look the same
    every run, so you can spot a wrong seed by eye."""
    from image_gen.backends.stub import _palette

    assert _palette("skybox", 1234) == _palette("skybox", 1234)
    assert _palette("skybox", 1234) != _palette("skybox", 9999)
    assert _palette("skybox", 1234) != _palette("image", 1234)


def test_scene_id_is_visible_in_the_placeholder(config, out_dir):
    """Two scenes must not render identically -- being able to tell which scene
    loaded is the reason the placeholder is drawn rather than copied."""
    generator = build_generator(config)
    a = generator.generate(req(scene_id="a")).path.read_bytes()
    b = generator.generate(req(scene_id="b")).path.read_bytes()
    assert a != b


def test_metadata_records_what_actually_ran(config, out_dir):
    build_generator(config).generate(req(steps=11))
    meta = json.loads((out_dir / "p07" / "p07_skybox_1234.json").read_text())

    assert meta["status"] == "ok"
    assert meta["scene_id"] == "p07"
    assert meta["kind"] == "skybox"
    assert meta["seed"] == 1234
    assert meta["file"] == "p07_skybox_1234.png"
    assert meta["backend"] == "stub"
    # Both prompts: the caller's, and what the model was actually given.
    assert meta["prompt"] == "a calm forest"
    assert meta["final_prompt"] == "equirectangular 360 view, a calm forest"
    # Resolved, not requested: the override must be what is recorded.
    assert meta["steps"] == 11
    assert meta["model"]["unet_repo"] == "ProGamerGov/sdxl-360-diffusion"
    assert "image_gen" in meta["versions"]


# -- failures -------------------------------------------------------------


def _break_render(monkeypatch, message="CUDA out of memory"):
    from image_gen.backends.stub import StubGenerator

    def boom(self, request, params):
        raise RuntimeError(message)

    monkeypatch.setattr(StubGenerator, "_render", boom)


def test_failure_writes_no_png_but_still_writes_metadata(config, out_dir, monkeypatch):
    _break_render(monkeypatch)
    result = build_generator(config).generate(req())

    assert result.status is AssetStatus.FAILED
    assert result.path is None
    assert not (out_dir / "p07" / "p07_skybox_1234.png").exists()

    meta = json.loads((out_dir / "p07" / "p07_skybox_1234.json").read_text())
    assert meta["status"] == "failed"
    assert meta["file"] is None
    assert "CUDA out of memory" in meta["error"]


def test_failure_is_returned_not_raised(config, monkeypatch):
    """One bad asset must never abort a batch that cost hours of queue time."""
    _break_render(monkeypatch)
    result = build_generator(config).generate(req())
    assert result.error is not None


# -- identifier safety ----------------------------------------------------


@pytest.mark.parametrize(
    "scene_id", ["../escape", "a/b", "..", "", "with space", "x" * 65]
)
def test_unsafe_scene_ids_are_rejected(config, scene_id):
    with pytest.raises(OutputError):
        build_generator(config).generate(req(scene_id=scene_id))


def test_unknown_kind_is_rejected(config):
    with pytest.raises(ConfigError, match="unknown asset kind"):
        build_generator(config).generate(req(kind="texture"))
