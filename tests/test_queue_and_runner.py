"""Queue validation, Slurm sharding, and batch execution."""

from __future__ import annotations

import json

import pytest

from image_gen.generator import build_generator
from image_gen.queue import (
    Queue,
    QueueAsset,
    QueueError,
    QueueScene,
    ParamOverrides,
    build_request,
    load_queue,
    shard,
    validate_queue,
)
from image_gen.runner import run_queue


def scene(scene_id="p07", seed=1, assets=None) -> QueueScene:
    return QueueScene(
        scene_id=scene_id,
        seed=seed,
        assets=assets or [QueueAsset(kind="skybox", prompt="a forest")],
    )


# -- validation -----------------------------------------------------------


def test_valid_queue_reports_no_problems(config, queue_file):
    assert validate_queue(load_queue(queue_file), config) == []


def test_unknown_kind_is_reported(config):
    queue = Queue(scenes=[scene(assets=[QueueAsset(kind="texture", prompt="x")])])
    problems = validate_queue(queue, config)
    assert any("unknown kind 'texture'" in p for p in problems)


def test_missing_seed_is_reported(config):
    queue = Queue(
        scenes=[
            QueueScene(
                scene_id="p07",
                seed=None,
                assets=[QueueAsset(kind="skybox", prompt="a forest")],
            )
        ]
    )
    problems = validate_queue(queue, config)
    assert any("no seed" in p for p in problems)


def test_duplicate_scene_ids_are_reported(config):
    """Two scenes with one id would write to the same folder and silently
    overwrite each other."""
    queue = Queue(scenes=[scene("p07"), scene("p07")])
    assert any("duplicate scene_id" in p for p in validate_queue(queue, config))


def test_unsafe_scene_id_is_reported(config):
    queue = Queue(scenes=[scene("../escape")])
    assert any("invalid scene_id" in p for p in validate_queue(queue, config))


def test_all_problems_are_reported_not_just_the_first(config):
    queue = Queue(
        scenes=[
            scene("p07", assets=[QueueAsset(kind="nope", prompt="x")]),
            scene("p07", assets=[QueueAsset(kind="also-nope", prompt="y")]),
        ]
    )
    assert len(validate_queue(queue, config)) >= 3


def test_queue_file_with_a_utf8_bom_loads(config, tmp_path):
    """PowerShell and Notepad write a UTF-8 BOM by default, and queue.json is
    exactly the kind of file that gets hand-edited or script-generated on
    Windows. Rejecting a BOM would be a papercut on every such file."""
    path = tmp_path / "bom.json"
    payload = {
        "scenes": [
            {
                "scene_id": "p07",
                "seed": 1,
                "assets": [{"kind": "skybox", "prompt": "a forest"}],
            }
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8-sig")

    queue = load_queue(path)
    assert validate_queue(queue, config) == []


def test_typo_in_a_field_name_is_rejected(tmp_path):
    """extra='forbid' turns a silently-ignored typo into a startup error."""
    path = tmp_path / "q.json"
    path.write_text(
        json.dumps(
            {
                "scenes": [
                    {
                        "scene_id": "p07",
                        "seed": 1,
                        "assets": [{"kind": "skybox", "promt": "typo"}],
                    }
                ]
            }
        )
    )
    with pytest.raises(QueueError):
        load_queue(path)


# -- precedence -----------------------------------------------------------


def test_precedence_asset_beats_scene_beats_queue_defaults(config):
    defaults = ParamOverrides(steps=1, guidance=1.0, width=2, height=1)
    s = QueueScene(
        scene_id="p07",
        seed=10,
        steps=2,
        guidance=2.0,
        assets=[QueueAsset(kind="skybox", prompt="x", steps=3, seed=99)],
    )
    request = build_request(s, s.assets[0], defaults)

    assert request.steps == 3  # asset
    assert request.guidance == 2.0  # scene
    assert request.width == 2  # queue defaults
    assert request.seed == 99  # asset seed beats scene seed


def test_scene_seed_is_used_when_the_asset_has_none(config):
    s = scene(seed=42)
    assert build_request(s, s.assets[0], ParamOverrides()).seed == 42


# -- sharding -------------------------------------------------------------


@pytest.mark.parametrize("count", [1, 2, 3, 5, 10])
def test_shards_cover_every_scene_exactly_once(count):
    scenes = [scene(f"p{i:02d}") for i in range(10)]
    seen = [s for i in range(count) for s in shard(scenes, i, count)]

    ids = [s.scene_id for s in seen]
    assert sorted(ids) == sorted(s.scene_id for s in scenes)
    assert len(ids) == len(set(ids)), "a scene appeared in two shards"


def test_shard_bounds_are_enforced():
    scenes = [scene("p01")]
    with pytest.raises(QueueError):
        shard(scenes, 2, 2)
    with pytest.raises(QueueError):
        shard(scenes, 0, 0)


# -- running --------------------------------------------------------------


def test_run_writes_assets_and_per_scene_manifests(config, out_dir, queue_file):
    report = run_queue(
        config, load_queue(queue_file), build_generator(config), log=lambda _: None
    )

    assert (report.ok, report.failed) == (3, 0)
    assert report.exit_code == 0

    assert (out_dir / "p07" / "p07_skybox_1234.png").is_file()
    assert (out_dir / "p07" / "p07_image_5678.png").is_file()
    assert (out_dir / "p08" / "p08_skybox_9012.png").is_file()

    manifest = json.loads((out_dir / "p07" / "p07_manifest.json").read_text())
    assert manifest["scene_id"] == "p07"
    assert {a["kind"] for a in manifest["assets"]} == {"skybox", "image"}


def test_one_failure_does_not_stop_the_batch(config, out_dir, queue_file, monkeypatch):
    from image_gen.backends.stub import StubGenerator

    original = StubGenerator._render
    calls = {"n": 0}

    def flaky(self, request, params):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("CUDA out of memory")
        return original(self, request, params)

    monkeypatch.setattr(StubGenerator, "_render", flaky)

    report = run_queue(
        config, load_queue(queue_file), build_generator(config), log=lambda _: None
    )

    assert report.failed == 1
    assert report.ok == 2, "the run continued past the failure"
    assert report.exit_code == 2

    manifest = json.loads((out_dir / "p07" / "p07_manifest.json").read_text())
    failed = [a for a in manifest["assets"] if a["status"] == "failed"]
    assert len(failed) == 1
    assert failed[0]["file"] is None
    assert "CUDA out of memory" in failed[0]["error"]


def test_skip_existing_does_not_regenerate(config, out_dir, queue_file):
    queue = load_queue(queue_file)
    generator = build_generator(config)
    run_queue(config, queue, generator, log=lambda _: None)

    second = run_queue(
        config, queue, generator, skip_existing=True, log=lambda _: None
    )
    assert second.skipped == 3
    assert second.ok == 0
    # Skipped assets still appear in the manifest -- a resumed run must produce
    # a complete record, not a partial one.
    manifest = json.loads((out_dir / "p07" / "p07_manifest.json").read_text())
    assert len(manifest["assets"]) == 2


def test_dry_run_writes_nothing(config, out_dir, queue_file):
    run_queue(
        config,
        load_queue(queue_file),
        build_generator(config),
        dry_run=True,
        log=lambda _: None,
    )
    assert not (out_dir / "p07").exists()


def test_sharded_runs_produce_disjoint_output(config, out_dir, queue_file):
    queue = load_queue(queue_file)
    generator = build_generator(config)
    for index in range(2):
        run_queue(
            config,
            queue,
            generator,
            shard_index=index,
            shard_count=2,
            log=lambda _: None,
        )
    assert (out_dir / "p07" / "p07_manifest.json").is_file()
    assert (out_dir / "p08" / "p08_manifest.json").is_file()
