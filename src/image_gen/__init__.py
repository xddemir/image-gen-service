"""image_gen -- prompt + seed -> image file + metadata.

The narrow middle of the thesis pipeline. It does not run the LLM, does not
decide what to generate, and does not talk to Unity; it turns already-written
prompts into reproducible files on disk.

    from image_gen import load_config, build_generator, GenerateRequest

    config = load_config("configs/default.yaml")
    generator = build_generator(config)
    result = generator.generate(
        GenerateRequest(kind="skybox", prompt="a calm forest", seed=1234,
                        scene_id="p07")
    )
"""

from .config import AppConfig, AssetKindConfig, ConfigError, load_config
from .generator import ImageGenerator, build_generator, resolve_params
from .types import (
    AssetStatus,
    GenerateRequest,
    GenerateResult,
    ManifestEntry,
    ResolvedParams,
)

__all__ = [
    "AppConfig",
    "AssetKindConfig",
    "AssetStatus",
    "ConfigError",
    "GenerateRequest",
    "GenerateResult",
    "ImageGenerator",
    "ManifestEntry",
    "ResolvedParams",
    "build_generator",
    "load_config",
    "resolve_params",
]

__version__ = "0.1.0"
