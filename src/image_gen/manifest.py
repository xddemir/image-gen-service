"""Manifests: what was produced, and what failed trying.

Two levels, on purpose:

  out/<scene_id>/<scene_id>_manifest.json   written only by the job owning the
                                            scene -- no shared file, so a Slurm
                                            job array needs no locking
  out/manifest.json                         merged afterwards by `merge-manifest`

Per-scene manifests carry bare filenames; the merged one carries paths relative
to out/, because that is what Unity resolves against.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .outputs import (
    merged_manifest_path,
    read_json,
    scene_manifest_path,
    utc_now_iso,
    write_json,
)
from .types import AssetStatus, ManifestEntry, SceneManifest

SCENE_MANIFEST_GLOB = "*/*_manifest.json"


def write_scene_manifest(
    out_dir: Path, scene_id: str, entries: list[ManifestEntry]
) -> Path:
    manifest = SceneManifest(scene_id=scene_id, created=utc_now_iso(), assets=entries)
    path = scene_manifest_path(out_dir, scene_id)
    write_json(path, manifest.to_dict())
    return path


def upsert_scene_entry(out_dir: Path, scene_id: str, entry: ManifestEntry) -> Path:
    """Add or replace one asset's entry in a scene manifest, keeping the rest.

    Used by the one-shot `gen` command, where successive calls build a scene up
    an asset at a time and a plain overwrite would discard the earlier ones.
    The batch runner does not use this: there, a task owns whole scenes and
    writes each manifest exactly once, which is what keeps job arrays safe.
    """
    path = scene_manifest_path(out_dir, scene_id)

    assets: list[dict[str, Any]] = []
    if path.is_file():
        try:
            assets = read_json(path).get("assets", [])
        except (OSError, ValueError):
            assets = []

    replacement = entry.to_dict()
    assets = [
        a
        for a in assets
        if not (a.get("kind") == entry.kind and a.get("seed") == entry.seed)
    ]
    assets.append(replacement)

    write_json(
        path,
        {"scene_id": scene_id, "created": utc_now_iso(), "assets": assets},
    )
    return path


def iter_scene_manifests(out_dir: Path) -> list[tuple[str, dict[str, Any]]]:
    """Every per-scene manifest under out/, sorted by scene id.

    `out/manifest.json` sits at the top level so the glob cannot pick it up.
    """
    found: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(Path(out_dir).glob(SCENE_MANIFEST_GLOB)):
        try:
            data = read_json(path)
        except (OSError, ValueError):
            # A truncated manifest from a killed job shouldn't sink the merge;
            # the scene simply reports as missing rather than as succeeded.
            continue
        scene_id = data.get("scene_id") or path.parent.name
        found.append((scene_id, data))
    return sorted(found, key=lambda pair: pair[0])


def merge_manifests(out_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Combine per-scene manifests into the merged, Unity-facing manifest."""
    merged: dict[str, list[dict[str, Any]]] = {}

    for scene_id, data in iter_scene_manifests(out_dir):
        entries: list[dict[str, Any]] = []
        for asset in data.get("assets", []):
            file_name = asset.get("file")
            entry: dict[str, Any] = {
                "kind": asset.get("kind"),
                # Relative to out/, forward slashes, so the same manifest reads
                # correctly whether it was produced on Linux or Windows.
                "file": f"{scene_id}/{file_name}" if file_name else None,
                "status": asset.get("status", AssetStatus.FAILED.value),
            }
            if asset.get("error"):
                entry["error"] = asset["error"]
            entries.append(entry)
        merged[scene_id] = entries

    write_json(merged_manifest_path(out_dir), merged)
    return merged


def summarize(out_dir: Path) -> dict[str, Any]:
    """What is actually on disk, for `image-gen status`.

    Reads the per-scene manifests rather than the merged file, so it works
    before a merge has run and whatever produced the output.
    """
    scenes: list[dict[str, Any]] = []
    total = ok = failed = 0

    for scene_id, data in iter_scene_manifests(out_dir):
        assets = data.get("assets", [])
        rows = []
        for asset in assets:
            status = asset.get("status", AssetStatus.FAILED.value)
            total += 1
            if status == AssetStatus.OK.value:
                ok += 1
            else:
                failed += 1
            rows.append(
                {
                    "kind": asset.get("kind"),
                    "seed": asset.get("seed"),
                    "status": status,
                    "file": asset.get("file"),
                    "error": asset.get("error"),
                }
            )
        scenes.append({"scene_id": scene_id, "assets": rows})

    return {
        "scenes": scenes,
        "counts": {
            "scenes": len(scenes),
            "assets": total,
            "ok": ok,
            "failed": failed,
        },
    }
