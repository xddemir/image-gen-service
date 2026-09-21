from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from image_gen.config import load_config

# Small resolutions: these tests exercise the contract, not the pixels, and a
# 2048x1024 draw per test adds up. The 2:1 skybox ratio is preserved because
# the aspect-ratio validation is part of what is under test.
CONFIG = {
    "backend": "stub",
    "kinds": {
        "skybox": {
            "model": {
                "base": "stabilityai/stable-diffusion-xl-base-1.0",
                "base_revision": None,
                "unet_repo": "ProGamerGov/sdxl-360-diffusion",
                "unet_subfolder": "unet",
                "unet_revision": "3565852",
            },
            "prompt_prefix": "equirectangular 360 view, ",
            "prompt_prefix_markers": [
                "equirectangular 360 view",
                "equirectangular",
                "360 panorama",
            ],
            "aspect_ratio": 2.0,
            "defaults": {
                "width": 256,
                "height": 128,
                "steps": 30,
                "guidance": 7.0,
                "negative_prompt": "people, text, watermark",
            },
        },
        "image": {
            "model": {
                "base": "stabilityai/stable-diffusion-xl-base-1.0",
                "base_revision": None,
            },
            "prompt_prefix": "",
            "defaults": {
                "width": 128,
                "height": 128,
                "steps": 30,
                "guidance": 7.0,
                "negative_prompt": "text, watermark",
            },
        },
    },
}


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    payload = dict(CONFIG)
    payload["out_dir"] = str(tmp_path / "out")
    path = cfg_dir / "default.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


@pytest.fixture
def config(config_path: Path):
    return load_config(config_path)


@pytest.fixture
def out_dir(config) -> Path:
    return Path(config.out_dir)


@pytest.fixture
def queue_file(tmp_path: Path) -> Path:
    path = tmp_path / "queue.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "defaults": {"steps": 12},
                "scenes": [
                    {
                        "scene_id": "p07",
                        "seed": 1234,
                        "assets": [
                            {"kind": "skybox", "prompt": "a calm misty forest"},
                            {
                                "kind": "image",
                                "prompt": "a river stone",
                                "seed": 5678,
                            },
                        ],
                    },
                    {
                        "scene_id": "p08",
                        "seed": 9012,
                        "assets": [{"kind": "skybox", "prompt": "an alpine lake"}],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path
