"""Request and response bodies.

Every generation parameter is optional except prompt and seed: omitting one
means "use the kind's configured default", so the API and the CLI resolve
parameters identically.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class GenerateBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(..., min_length=1, examples=["a calm misty forest at dawn"])
    # Required: an unseeded generation cannot be reproduced, and reproducibility
    # is the point.
    seed: int = Field(..., examples=[1234])
    # Optional -- defaults to scene_<UTC timestamp> so a bare curl still works.
    scene_id: Optional[str] = Field(default=None, examples=["p07"])
    negative_prompt: Optional[str] = None
    width: Optional[int] = Field(default=None, gt=0)
    height: Optional[int] = Field(default=None, gt=0)
    steps: Optional[int] = Field(default=None, gt=0)
    guidance: Optional[float] = Field(default=None, ge=0)


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    kind: str
    scene_id: str
    seed: int
    file: Optional[str] = None
    error: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    backend: str
    out_dir: str
    kinds_configured: list[str]
    kinds_loaded: list[str]
    queued: int
    versions: dict[str, Any]
