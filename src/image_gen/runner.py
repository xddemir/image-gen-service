"""Batch execution: queue in, files and manifests out.

The contract this module upholds is that a single failed asset never aborts the
batch. On a cluster, a run may represent hours of queue wait; losing the other
39 participants because one prompt tripped an OOM would be indefensible. The
failure is recorded and the loop moves on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import AppConfig
from .generator import ImageGenerator
from .manifest import write_scene_manifest
from .outputs import image_path, read_json, sidecar_path
from .queue import Queue, build_request, shard
from .types import AssetStatus, GenerateRequest, ManifestEntry

Logger = Callable[[str], None]


@dataclass
class RunReport:
    scenes: int = 0
    ok: int = 0
    failed: int = 0
    skipped: int = 0

    @property
    def total(self) -> int:
        return self.ok + self.failed + self.skipped

    @property
    def exit_code(self) -> int:
        """0 all good, 2 completed with failures. Reserved: 1 means fatal.

        Distinguishable so a wrapper script can tell "nothing worked" from
        "most of it worked" without parsing output.
        """
        return 2 if self.failed else 0


def _already_done(out_dir: Path, req: GenerateRequest) -> bool:
    """True only if a previous run produced this asset successfully.

    Both the PNG and an `ok` sidecar must be present: a PNG without metadata
    cannot be reproduced, and a `failed` sidecar should be retried.
    """
    png = image_path(out_dir, req.scene_id, req.kind, req.seed)
    meta = sidecar_path(out_dir, req.scene_id, req.kind, req.seed)
    if not (png.is_file() and meta.is_file()):
        return False
    try:
        return read_json(meta).get("status") == AssetStatus.OK.value
    except (OSError, ValueError):
        return False


def run_queue(
    config: AppConfig,
    queue: Queue,
    generator: ImageGenerator,
    *,
    shard_index: int = 0,
    shard_count: int = 1,
    skip_existing: bool = False,
    dry_run: bool = False,
    log: Logger = print,
) -> RunReport:
    out_dir = Path(config.out_dir)
    scenes = shard(queue.scenes, shard_index, shard_count)
    report = RunReport(scenes=len(scenes))

    if shard_count > 1:
        log(
            f"shard {shard_index + 1}/{shard_count}: "
            f"{len(scenes)} of {len(queue.scenes)} scenes"
        )

    total = sum(len(scene.assets) for scene in scenes)
    index = 0

    for scene in scenes:
        entries: list[ManifestEntry] = []

        for asset in scene.assets:
            index += 1
            req = build_request(scene, asset, queue.defaults)
            prefix = f"[{index}/{total}] {req.scene_id}  {req.kind} seed={req.seed}"

            if dry_run:
                target = image_path(out_dir, req.scene_id, req.kind, req.seed)
                log(f"{prefix}  ... would write -> {target}")
                continue

            if skip_existing and _already_done(out_dir, req):
                png = image_path(out_dir, req.scene_id, req.kind, req.seed)
                log(f"{prefix}  ... skipped (exists)")
                report.skipped += 1
                entries.append(
                    ManifestEntry(
                        kind=req.kind,
                        seed=req.seed,
                        status=AssetStatus.OK,
                        file=png.name,
                    )
                )
                continue

            result = generator.generate(req)

            if result.ok and result.path is not None:
                report.ok += 1
                log(
                    f"{prefix}  ... ok      "
                    f"({result.metadata['duration_s']:.1f}s) -> {result.path}"
                )
                entries.append(
                    ManifestEntry(
                        kind=req.kind,
                        seed=req.seed,
                        status=AssetStatus.OK,
                        file=result.path.name,
                    )
                )
            else:
                report.failed += 1
                log(f"{prefix}  ... FAILED  {result.error}")
                entries.append(
                    ManifestEntry(
                        kind=req.kind,
                        seed=req.seed,
                        status=AssetStatus.FAILED,
                        file=None,
                        error=result.error,
                    )
                )

        if not dry_run:
            write_scene_manifest(out_dir, scene.scene_id, entries)

    return report
