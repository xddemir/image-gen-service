"""The `image-gen` command line.

This is the program that runs on Pegasus. The same commands run locally with
`backend: stub`, which is how the whole contract gets exercised without a GPU.

Exit codes:
    0  everything succeeded
    1  fatal -- bad config, malformed queue, nothing ran
    2  the run completed but some assets failed (see the manifest)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer

from .config import AppConfig, ConfigError, load_config
from .generator import build_generator
from .manifest import merge_manifests, summarize, upsert_scene_entry
from .outputs import OutputError, auto_scene_id, validate_scene_id
from .queue import QueueError, load_queue, validate_queue
from .runner import run_queue
from .types import AssetStatus, GenerateRequest, ManifestEntry

app = typer.Typer(
    add_completion=False,
    help="Reproducible generation of 360 skyboxes and flat images.",
)

EXIT_FATAL = 1


def _err(message: str) -> None:
    typer.secho(f"error: {message}", fg=typer.colors.RED, err=True)


def _load(
    config_path: Optional[Path],
    backend: Optional[str],
    out: Optional[Path],
) -> AppConfig:
    """Load config, applying the two overrides every command accepts."""
    config = load_config(config_path)
    if backend:
        config.backend = backend
    if out:
        config.out_dir = out
    return config


@app.command()
def gen(
    prompt: str = typer.Option(..., "--prompt", "-p", help="Prompt text."),
    kind: str = typer.Option("skybox", "--kind", "-k", help="Configured asset kind."),
    seed: int = typer.Option(..., "--seed", "-s", help="Seed. Required: output "
                             "must be reproducible."),
    scene_id: Optional[str] = typer.Option(
        None, "--scene-id", help="Participant/scene id. Defaults to a timestamp."
    ),
    negative_prompt: Optional[str] = typer.Option(None, "--negative-prompt"),
    width: Optional[int] = typer.Option(None, "--width"),
    height: Optional[int] = typer.Option(None, "--height"),
    steps: Optional[int] = typer.Option(None, "--steps"),
    guidance: Optional[float] = typer.Option(None, "--guidance"),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
    backend: Optional[str] = typer.Option(None, "--backend"),
    out: Optional[Path] = typer.Option(None, "--out"),
) -> None:
    """Generate a single asset. The quick path while iterating on a prompt."""
    try:
        config = _load(config_path, backend, out)
        if kind not in config.kinds:
            raise ConfigError(
                f"unknown asset kind {kind!r}; configured kinds: "
                + ", ".join(sorted(config.kinds))
            )
        resolved_scene = validate_scene_id(scene_id) if scene_id else auto_scene_id()
        generator = build_generator(config)
    except (ConfigError, OutputError) as exc:
        _err(str(exc))
        raise typer.Exit(EXIT_FATAL)

    result = generator.generate(
        GenerateRequest(
            kind=kind,
            prompt=prompt,
            seed=seed,
            scene_id=resolved_scene,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            guidance=guidance,
        )
    )

    # Keep the scene manifest current, merging rather than overwriting, so
    # successive `gen` calls build a scene up instead of clobbering each other.
    upsert_scene_entry(
        Path(config.out_dir),
        resolved_scene,
        ManifestEntry(
            kind=kind,
            seed=seed,
            status=result.status,
            file=result.path.name if result.path else None,
            error=result.error,
        ),
    )

    if result.ok and result.path:
        typer.echo(str(result.path))
        return

    _err(result.error or "generation failed")
    raise typer.Exit(2)


@app.command()
def run(
    queue_file: Path = typer.Argument(..., help="Path to queue.json."),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
    backend: Optional[str] = typer.Option(None, "--backend"),
    out: Optional[Path] = typer.Option(None, "--out"),
    shard_index: int = typer.Option(
        0, "--shard-index", help="This task's index. Use $SLURM_ARRAY_TASK_ID."
    ),
    shard_count: int = typer.Option(
        1, "--shard-count", help="Total tasks. Use $SLURM_ARRAY_TASK_COUNT."
    ),
    skip_existing: bool = typer.Option(
        False, "--skip-existing", help="Resume: leave already-generated assets alone."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print what would be written, generate nothing."
    ),
) -> None:
    """Generate a batch from queue.json. One failed asset never stops the run."""
    try:
        config = _load(config_path, backend, out)
        queue = load_queue(queue_file)
        problems = validate_queue(queue, config)
        if problems:
            for problem in problems:
                _err(problem)
            raise typer.Exit(EXIT_FATAL)
        generator = build_generator(config)
    except (ConfigError, QueueError, OutputError) as exc:
        _err(str(exc))
        raise typer.Exit(EXIT_FATAL)

    report = run_queue(
        config,
        queue,
        generator,
        shard_index=shard_index,
        shard_count=shard_count,
        skip_existing=skip_existing,
        dry_run=dry_run,
        log=typer.echo,
    )

    if dry_run:
        typer.echo(f"Dry run: {report.scenes} scenes, nothing written.")
        return

    summary = f"Done: {report.ok} ok, {report.failed} failed"
    if report.skipped:
        summary += f", {report.skipped} skipped"
    typer.echo(summary)
    raise typer.Exit(report.exit_code)


@app.command("merge-manifest")
def merge_manifest_cmd(
    out_dir: Path = typer.Argument(Path("out"), help="Output directory."),
) -> None:
    """Combine per-scene manifests into out/manifest.json."""
    if not out_dir.is_dir():
        _err(f"not a directory: {out_dir}")
        raise typer.Exit(EXIT_FATAL)

    merged = merge_manifests(out_dir)
    assets = sum(len(v) for v in merged.values())
    typer.echo(
        f"Merged {len(merged)} scenes ({assets} assets) -> {out_dir / 'manifest.json'}"
    )


@app.command()
def status(
    out_dir: Path = typer.Argument(Path("out"), help="Output directory."),
) -> None:
    """Show what is on disk: what succeeded, what failed."""
    if not out_dir.is_dir():
        _err(f"not a directory: {out_dir}")
        raise typer.Exit(EXIT_FATAL)

    report = summarize(out_dir)
    if not report["scenes"]:
        typer.echo(f"No scene manifests found under {out_dir}")
        return

    width = max(len(s["scene_id"]) for s in report["scenes"])
    for scene in report["scenes"]:
        parts = []
        for asset in scene["assets"]:
            mark = "ok" if asset["status"] == AssetStatus.OK.value else "FAILED"
            cell = f"{asset['kind']} {mark}"
            if asset["error"]:
                cell += f" ({asset['error']})"
            parts.append(cell)
        typer.echo(f"{scene['scene_id']:<{width}}    " + "       ".join(parts))

    counts = report["counts"]
    typer.echo(
        f"\n{counts['scenes']} scenes - {counts['assets']} assets - "
        f"{counts['ok']} ok - {counts['failed']} failed"
    )
    if counts["failed"]:
        raise typer.Exit(2)


@app.command()
def validate(
    queue_file: Path = typer.Argument(..., help="Path to queue.json."),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
) -> None:
    """Check a queue file against the config without generating anything."""
    try:
        config = load_config(config_path)
        queue = load_queue(queue_file)
    except (ConfigError, QueueError) as exc:
        _err(str(exc))
        raise typer.Exit(EXIT_FATAL)

    problems = validate_queue(queue, config)
    if problems:
        for problem in problems:
            _err(problem)
        raise typer.Exit(EXIT_FATAL)

    assets = sum(len(s.assets) for s in queue.scenes)
    typer.echo(f"OK: {len(queue.scenes)} scenes, {assets} assets.")


@app.command("download-weights")
def download_weights(
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
    resolve_only: bool = typer.Option(
        False, "--resolve-only", help="Pin revisions without downloading."
    ),
    max_workers: int = typer.Option(
        2,
        "--max-workers",
        help="Parallel file downloads. Low by default: login nodes cap "
        "per-user memory and a fast parallel fetch gets OOM-killed.",
    ),
) -> None:
    """Pre-fetch weights and pin their revisions. Run on a machine with internet.

    Compute nodes are offline, so this has to happen on the login node first.
    Resolved commit hashes are written to configs/revisions.lock.json rather
    than back into the YAML, which keeps the config's comments intact.
    """
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        _err(str(exc))
        raise typer.Exit(EXIT_FATAL)

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        _err(
            "huggingface_hub is required. Install the GPU extra: "
            "pip install -e '.[local]'"
        )
        raise typer.Exit(EXIT_FATAL)

    import json

    api = HfApi()
    lock: dict[str, str] = {}

    # These duplicate the safetensors weights in formats we never load.
    ignore = ["*.bin", "*.pth", "*.onnx*", "*.msgpack", "*non_ema*"]

    # Build one plan per repository before fetching anything.
    #
    # Kinds share weights -- skybox is base SDXL with a different UNet -- so a
    # repo is usually wanted by several kinds. Iterating kinds and downloading
    # the first time a repo appears would apply only that kind's settings and
    # silently ignore every other kind's, which is how the whole repo ends up
    # downloaded because the kind that happened to sort first had no excludes.
    plan: dict[str, dict] = {}

    def want(repo: str, pinned: Optional[str], allow: Optional[list[str]],
             extra_ignore: list[str], kind_name: str) -> None:
        entry = plan.setdefault(
            repo, {"pinned": None, "allow": allow, "ignore": set(ignore), "users": []}
        )
        if kind_name not in entry["users"]:
            entry["users"].append(kind_name)
        if pinned and not entry["pinned"]:
            entry["pinned"] = pinned
        # One kind needing the whole repo outranks another needing a subfolder.
        if allow is None:
            entry["allow"] = None
        entry["ignore"].update(extra_ignore)

    for name, kind in sorted(config.kinds.items()):
        if kind.model is None:
            typer.echo(f"{name}: no model configured, skipping")
            continue

        want(kind.model.base, kind.model.base_revision, None,
             kind.model.download_ignore, name)

        if kind.model.unet_repo:
            # Only the diffusers-format UNet subfolder is needed; the repo also
            # ships standalone checkpoints that from_pretrained never reads.
            want(kind.model.unet_repo, kind.model.unet_revision,
                 [f"{kind.model.unet_subfolder}/*"],
                 kind.model.download_ignore, name)

    for repo, entry in plan.items():
        revision = entry["pinned"] or api.model_info(repo).sha
        lock[repo] = revision
        allow = entry["allow"]
        skip = sorted(entry["ignore"])

        typer.echo(f"{repo} @ {revision}")
        typer.echo(f"    used by: {', '.join(entry['users'])}")
        if allow:
            typer.echo(f"    only:    {', '.join(allow)}")
        elif skip:
            typer.echo(f"    skipping: {', '.join(skip)}")

        if not resolve_only:
            snapshot_download(
                repo_id=repo,
                revision=revision,
                allow_patterns=allow,
                ignore_patterns=None if allow else skip,
                max_workers=max_workers,
            )

    lock_path = Path(config.source_dir) / "revisions.lock.json"
    lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    typer.echo(f"\nPinned {len(lock)} repositories -> {lock_path}")


def main() -> None:
    try:
        app()
    except KeyboardInterrupt:  # pragma: no cover
        _err("interrupted")
        sys.exit(EXIT_FATAL)


if __name__ == "__main__":  # pragma: no cover
    main()
