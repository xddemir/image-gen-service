"""Configuration loading and validation.

Asset kinds are defined entirely in YAML. Nothing in the CLI, the API or the
generators enumerates kinds by name, so adding one is a config change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")

# Resolved commit hashes live beside the config rather than inside it.
# Rewriting the YAML in place would strip its comments, and those comments are
# the only place the from_pretrained/UNet-swap reasoning is recorded.
REVISIONS_LOCK_NAME = "revisions.lock.json"

# Keys accepted in YAML so the schema is stable, but not implemented yet.
# Silently ignoring them would be worse than refusing: a user who sets
# enable_cpu_offload and watches the job OOM deserves to be told why.
RESERVED_BACKEND_OPTIONS = ("enable_cpu_offload", "attention_slicing", "vae_tiling")


class ConfigError(Exception):
    """Raised for any malformed or unusable configuration."""


class ModelConfig(BaseModel):
    """Which weights a kind uses.

    `unet_repo` exists because ProGamerGov/sdxl-360-diffusion ships no
    `model_index.json` -- only a diffusers-format `unet/` subfolder. So the
    pipeline is built from `base` and has this UNet swapped in, rather than
    being loaded from the 360 repo directly.
    """

    model_config = ConfigDict(protected_namespaces=())

    base: str
    base_revision: str | None = None
    unet_repo: str | None = None
    unet_subfolder: str = "unet"
    unet_revision: str | None = None

    # Extra glob patterns to skip when pre-fetching this repo, on top of the
    # built-in defaults. Model repos often ship standalone single-file
    # checkpoints alongside the diffusers-format subfolders; from_pretrained()
    # never reads them, and they can be most of the download.
    download_ignore: list[str] = Field(default_factory=list)

    def to_metadata(self) -> dict[str, Any]:
        out: dict[str, Any] = {"base": self.base, "base_revision": self.base_revision}
        if self.unet_repo:
            out["unet_repo"] = self.unet_repo
            out["unet_revision"] = self.unet_revision
        return out


class KindDefaults(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    steps: int = Field(default=30, gt=0)
    guidance: float = Field(default=7.0, ge=0)
    negative_prompt: str = ""


class AssetKindConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model: ModelConfig | None = None
    prompt_prefix: str = ""
    prompt_prefix_markers: list[str] = Field(default_factory=list)
    aspect_ratio: float | None = None
    defaults: KindDefaults
    stub_file: Path | None = None

    @field_validator("aspect_ratio")
    @classmethod
    def _positive_ratio(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError("aspect_ratio must be positive")
        return v

    def markers(self) -> list[str]:
        """Phrases that count as "the trigger phrase is already present".

        Defaults to the prefix itself with trailing punctuation stripped, so a
        kind only needs to list extras (e.g. "360 panorama") when the model
        accepts more than one wording.
        """
        if self.prompt_prefix_markers:
            return self.prompt_prefix_markers
        stripped = self.prompt_prefix.strip().rstrip(",").strip()
        return [stripped] if stripped else []

    def check_size(self, width: int, height: int) -> None:
        """Reject sizes that violate the kind's required aspect ratio."""
        if self.aspect_ratio is None:
            return
        actual = width / height
        # Tolerate rounding, not genuine mistakes.
        if abs(actual - self.aspect_ratio) > 0.01:
            raise ConfigError(
                f"{width}x{height} has aspect ratio {actual:.3f}, but this kind "
                f"requires {self.aspect_ratio:g}:1 "
                f"(e.g. {int(self.aspect_ratio * height)}x{height})"
            )


class BackendOptions(BaseModel):
    dtype: str = "float16"
    max_loaded_pipelines: int = Field(default=1, ge=1)

    # Reserved -- see RESERVED_BACKEND_OPTIONS.
    enable_cpu_offload: bool = False
    attention_slicing: bool = False
    vae_tiling: bool = False


class AppConfig(BaseModel):
    backend: str = "stub"
    out_dir: Path = Path("out")
    backend_options: BackendOptions = Field(default_factory=BackendOptions)
    kinds: dict[str, AssetKindConfig]

    # Where this config was loaded from; relative paths resolve against it.
    source_dir: Path = Field(default=Path("."), exclude=True)

    @field_validator("kinds")
    @classmethod
    def _at_least_one_kind(
        cls, v: dict[str, AssetKindConfig]
    ) -> dict[str, AssetKindConfig]:
        if not v:
            raise ValueError("at least one asset kind must be configured")
        return v

    def kind(self, name: str) -> AssetKindConfig:
        try:
            return self.kinds[name]
        except KeyError:
            known = ", ".join(sorted(self.kinds)) or "(none)"
            raise ConfigError(f"unknown asset kind {name!r}; configured kinds: {known}")

    def resolve_path(self, path: Path) -> Path:
        """Resolve a config-relative path against the config file's directory."""
        return path if path.is_absolute() else (self.source_dir / path)


def _validate_reserved(raw_backend_options: dict[str, Any]) -> None:
    enabled = [k for k in RESERVED_BACKEND_OPTIONS if raw_backend_options.get(k)]
    if enabled:
        raise ConfigError(
            "backend_options "
            + ", ".join(repr(k) for k in enabled)
            + " are reserved but not implemented yet. Remove them or set them to "
            "false; leaving them on would silently do nothing."
        )


def load_config(path: Path | str | None = None) -> AppConfig:
    """Load and validate a config file.

    Also validates each kind's configured default size against its declared
    aspect ratio, so a bad default fails at startup rather than after a job has
    been queued on the cluster.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.is_file():
        raise ConfigError(f"config file not found: {cfg_path}")

    try:
        # utf-8-sig: a config hand-edited on Windows may carry a UTF-8 BOM.
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8-sig")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{cfg_path}: invalid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"{cfg_path}: top level must be a mapping")

    _validate_reserved(raw.get("backend_options") or {})

    raw["source_dir"] = cfg_path.parent

    try:
        config = AppConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"{cfg_path}: {exc}") from exc

    for name, kind in config.kinds.items():
        try:
            kind.check_size(kind.defaults.width, kind.defaults.height)
        except ConfigError as exc:
            raise ConfigError(f"{cfg_path}: kind {name!r}: {exc}") from exc

    apply_revisions_lock(config)
    return config


def load_revisions_lock(source_dir: Path) -> dict[str, str]:
    path = Path(source_dir) / REVISIONS_LOCK_NAME
    if not path.is_file():
        return {}
    try:
        import json

        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise ConfigError(f"{path}: unreadable revisions lock: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected an object mapping repo -> commit sha")
    return {str(k): str(v) for k, v in data.items()}


def apply_revisions_lock(config: AppConfig) -> None:
    """Fill in any revision left null from the lock file written by
    `image-gen download-weights`.

    An explicit revision in the YAML always wins -- the lock only supplies what
    was never pinned, so hand-pinning a repo is never silently overridden.
    """
    lock = load_revisions_lock(config.source_dir)
    if not lock:
        return
    for kind in config.kinds.values():
        model = kind.model
        if model is None:
            continue
        if model.base_revision is None and model.base in lock:
            model.base_revision = lock[model.base]
        if (
            model.unet_repo
            and model.unet_revision is None
            and model.unet_repo in lock
        ):
            model.unet_revision = lock[model.unet_repo]
