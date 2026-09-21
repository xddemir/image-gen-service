"""The on-disk output contract: naming, paths, metadata sidecars.

    out/<scene_id>/<scene_id>_<kind>_<seed>.png
    out/<scene_id>/<scene_id>_<kind>_<seed>.json
    out/<scene_id>/<scene_id>_manifest.json
    out/manifest.json

The `scene_id` prefix duplicates the folder name on purpose. A file that gets
dragged into Unity, attached to an email or opened in a viewer loses its folder
context, and `skybox_1234.png` then tells you nothing about who it belongs to.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Conservative on purpose: scene_id becomes a directory name and a filename
# prefix, so it must not be able to escape out_dir or collide with shell or
# Windows path semantics. Leading alphanumeric also rules out "." and "..".
SCENE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class OutputError(Exception):
    """Raised for invalid identifiers or unwritable output paths."""


def validate_scene_id(scene_id: str) -> str:
    if not SCENE_ID_RE.match(scene_id or ""):
        raise OutputError(
            f"invalid scene_id {scene_id!r}: use 1-64 characters of "
            "[A-Za-z0-9._-], starting with a letter or digit"
        )
    return scene_id


def auto_scene_id(now: datetime | None = None) -> str:
    """Fallback id when a caller does not supply one (bare API/CLI calls)."""
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d_%H%M%S")
    return f"scene_{stamp}"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def asset_stem(scene_id: str, kind: str, seed: int) -> str:
    return f"{scene_id}_{kind}_{seed}"


def scene_dir(out_dir: Path, scene_id: str) -> Path:
    return Path(out_dir) / validate_scene_id(scene_id)


def image_path(out_dir: Path, scene_id: str, kind: str, seed: int) -> Path:
    return scene_dir(out_dir, scene_id) / f"{asset_stem(scene_id, kind, seed)}.png"


def sidecar_path(out_dir: Path, scene_id: str, kind: str, seed: int) -> Path:
    return scene_dir(out_dir, scene_id) / f"{asset_stem(scene_id, kind, seed)}.json"


def scene_manifest_path(out_dir: Path, scene_id: str) -> Path:
    return scene_dir(out_dir, scene_id) / f"{scene_id}_manifest.json"


def merged_manifest_path(out_dir: Path) -> Path:
    return Path(out_dir) / "manifest.json"


def relative_to_out(out_dir: Path, path: Path) -> str:
    """Path as the merged manifest records it: relative to out/, forward slashes.

    Unity resolves against out/, and a Windows-style separator in a JSON file
    that may be read on Linux is a portability bug waiting to happen.
    """
    rel = Path(path).resolve().relative_to(Path(out_dir).resolve())
    return rel.as_posix()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically.

    Batch jobs get killed -- by the walltime limit, by the OOM killer, by a node
    failure. A half-written manifest is worse than no manifest, so the content
    lands in a temp file and is moved into place in one step.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> dict[str, Any]:
    # utf-8-sig tolerates a BOM from Windows tooling; files we write have none.
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def library_versions() -> dict[str, str]:
    """Versions that affect whether a seed reproduces the same pixels.

    An identical seed only reproduces an identical image on the same GPU,
    driver and library versions. Recording them is the difference between a
    reproducibility claim and a reproducibility hope.
    """
    from importlib.metadata import PackageNotFoundError, version

    versions: dict[str, str] = {}
    try:
        versions["image_gen"] = version("image-gen")
    except PackageNotFoundError:
        versions["image_gen"] = "0.0.0+dev"

    for name in ("torch", "diffusers", "transformers", "pillow"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            pass
    return versions
