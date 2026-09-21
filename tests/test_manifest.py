"""Manifest merging and the status summary."""

from __future__ import annotations

import json

from image_gen.generator import build_generator
from image_gen.manifest import merge_manifests, summarize, upsert_scene_entry
from image_gen.queue import load_queue
from image_gen.runner import run_queue
from image_gen.types import AssetStatus, ManifestEntry


def _run(config, queue_file, **kwargs):
    return run_queue(
        config,
        load_queue(queue_file),
        build_generator(config),
        log=lambda _: None,
        **kwargs,
    )


def test_merge_combines_per_scene_manifests(config, out_dir, queue_file):
    _run(config, queue_file)
    merged = merge_manifests(out_dir)

    assert set(merged) == {"p07", "p08"}
    assert len(merged["p07"]) == 2

    entry = next(e for e in merged["p07"] if e["kind"] == "skybox")
    # Relative to out/, forward slashes: Unity resolves against out/, and the
    # file may be written on Linux and read on Windows.
    assert entry["file"] == "p07/p07_skybox_1234.png"
    assert entry["status"] == "ok"
    assert "\\" not in entry["file"]


def test_merged_manifest_is_written_to_disk(config, out_dir, queue_file):
    _run(config, queue_file)
    merge_manifests(out_dir)

    on_disk = json.loads((out_dir / "manifest.json").read_text())
    assert set(on_disk) == {"p07", "p08"}


def test_failures_appear_in_the_merged_manifest(config, out_dir, queue_file, monkeypatch):
    """A missing entry is ambiguous; a `failed` entry is not. Unity needs the
    difference to pick a fallback scene."""
    from image_gen.backends.stub import StubGenerator

    monkeypatch.setattr(
        StubGenerator,
        "_render",
        lambda self, request, params: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    _run(config, queue_file)
    merged = merge_manifests(out_dir)

    entries = merged["p07"]
    assert all(e["status"] == "failed" for e in entries)
    assert all(e["file"] is None for e in entries)
    assert all("boom" in e["error"] for e in entries)


def test_merge_ignores_the_top_level_manifest(config, out_dir, queue_file):
    """out/manifest.json must not be picked up as if it were a scene."""
    _run(config, queue_file)
    merge_manifests(out_dir)
    again = merge_manifests(out_dir)
    assert set(again) == {"p07", "p08"}


def test_merge_survives_a_truncated_scene_manifest(config, out_dir, queue_file):
    """Jobs get killed mid-write. A corrupt manifest should cost one scene,
    not the whole merge."""
    _run(config, queue_file)
    (out_dir / "p08" / "p08_manifest.json").write_text("{ truncated", encoding="utf-8")

    merged = merge_manifests(out_dir)
    assert "p07" in merged
    assert "p08" not in merged


def test_upsert_adds_without_clobbering_earlier_assets(config, out_dir):
    for kind, seed in (("skybox", 1), ("image", 2)):
        upsert_scene_entry(
            out_dir,
            "p07",
            ManifestEntry(kind=kind, seed=seed, status=AssetStatus.OK, file=f"f{seed}"),
        )

    manifest = json.loads((out_dir / "p07" / "p07_manifest.json").read_text())
    assert {a["kind"] for a in manifest["assets"]} == {"skybox", "image"}


def test_upsert_replaces_a_rerun_of_the_same_asset(config, out_dir):
    for status in (AssetStatus.FAILED, AssetStatus.OK):
        upsert_scene_entry(
            out_dir,
            "p07",
            ManifestEntry(kind="skybox", seed=1, status=status, file="f"),
        )

    manifest = json.loads((out_dir / "p07" / "p07_manifest.json").read_text())
    assert len(manifest["assets"]) == 1
    assert manifest["assets"][0]["status"] == "ok"


def test_summarize_counts_outcomes(config, out_dir, queue_file):
    _run(config, queue_file)
    report = summarize(out_dir)

    assert report["counts"] == {"scenes": 2, "assets": 3, "ok": 3, "failed": 0}


def test_summarize_works_before_a_merge_has_run(config, out_dir, queue_file):
    _run(config, queue_file)
    assert not (out_dir / "manifest.json").exists()
    assert summarize(out_dir)["counts"]["assets"] == 3
