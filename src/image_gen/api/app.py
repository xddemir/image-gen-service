"""FastAPI app.

One parameterised route rather than one per kind. `POST /generate/skybox` and
`POST /generate/image` both work exactly as sketched, but adding a kind is a
config change with no API code behind it -- which is what "config-driven kinds"
has to mean in practice.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse

from ..config import AppConfig, ConfigError, load_config
from ..generator import build_generator
from ..outputs import auto_scene_id, library_versions, validate_scene_id, OutputError
from ..types import GenerateRequest
from .jobs import Job, JobRunner
from .models import GenerateBody, HealthResponse, JobResponse

CONFIG_ENV_VAR = "IMAGE_GEN_CONFIG"


def _to_response(job: Job) -> JobResponse:
    return JobResponse(
        job_id=job.job_id,
        status=job.status,
        kind=job.request.kind,
        scene_id=job.request.scene_id,
        seed=job.request.seed,
        file=job.file,
        error=job.error,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


def create_app(config: Optional[AppConfig] = None) -> FastAPI:
    if config is None:
        config = load_config(os.environ.get(CONFIG_ENV_VAR) or None)

    generator = build_generator(config)
    out_dir = Path(config.out_dir)
    runner = JobRunner(generator, out_dir)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        runner.start()
        try:
            yield
        finally:
            runner.stop()

    app = FastAPI(
        title="image-gen",
        version="0.1.0",
        summary="prompt + seed -> reproducible image files on disk",
        lifespan=lifespan,
    )
    app.state.config = config
    app.state.runner = runner

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse("/docs")

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            backend=config.backend,
            out_dir=str(out_dir),
            kinds_configured=sorted(config.kinds),
            kinds_loaded=generator.loaded_kinds(),
            queued=runner.pending(),
            versions=library_versions(),
        )

    @app.post("/generate/{kind}", response_model=JobResponse, status_code=202)
    async def generate(kind: str, body: GenerateBody) -> JobResponse:
        if kind not in config.kinds:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"unknown asset kind {kind!r}; configured kinds: "
                    + ", ".join(sorted(config.kinds))
                ),
            )

        try:
            scene_id = (
                validate_scene_id(body.scene_id) if body.scene_id else auto_scene_id()
            )
        except OutputError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        try:
            # Catches aspect-ratio violations before the job is queued, so the
            # caller gets a 422 instead of a job that fails a minute later.
            from ..generator import resolve_params

            request = GenerateRequest(
                kind=kind,
                prompt=body.prompt,
                seed=body.seed,
                scene_id=scene_id,
                negative_prompt=body.negative_prompt,
                width=body.width,
                height=body.height,
                steps=body.steps,
                guidance=body.guidance,
            )
            resolve_params(config, request)
        except ConfigError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        return _to_response(runner.submit(request))

    @app.get("/jobs/{job_id}", response_model=JobResponse)
    async def job_status(job_id: str) -> JobResponse:
        job = runner.get(job_id)
        if job is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"unknown job {job_id!r}. Job state is in memory and is lost "
                    "on restart; the manifest on disk is the durable record."
                ),
            )
        return _to_response(job)

    @app.get("/files/{path:path}")
    async def get_file(path: str) -> FileResponse:
        base = out_dir.resolve()
        # Resolve first, then confirm containment: this is what stops
        # ../../etc/passwd, symlinks out of the tree, and absolute paths.
        target = (base / path).resolve()
        if not target.is_relative_to(base):
            raise HTTPException(status_code=403, detail="path escapes the output tree")
        if not target.is_file():
            raise HTTPException(status_code=404, detail=f"no such file: {path}")
        return FileResponse(target)

    return app


# Module-level app for `uvicorn image_gen.api.app:app`.
# Built lazily via __getattr__ so importing this module never reads config --
# tests build their own app with create_app().
def __getattr__(name: str):
    if name == "app":
        return create_app()
    raise AttributeError(name)
