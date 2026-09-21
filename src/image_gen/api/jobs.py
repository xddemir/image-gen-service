"""A single-worker background queue.

One worker thread, not an asyncio task: the diffusers call is blocking and CPU/
GPU bound, so running it on the event loop would stall every other request. One
worker also gives "one job at a time on the GPU" for free, with no semaphore.

Job state is in memory and is lost on restart. That is deliberate -- disk
(sidecars plus manifests) is the source of truth, which is exactly why the
manifest exists. The API is a convenience over the files, never the record.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Optional

from ..generator import ImageGenerator
from ..manifest import upsert_scene_entry
from ..outputs import utc_now_iso
from ..types import AssetStatus, GenerateRequest, ManifestEntry
from .models import JobStatus

_SHUTDOWN = object()


@dataclass
class Job:
    job_id: str
    request: GenerateRequest
    status: JobStatus = JobStatus.QUEUED
    file: Optional[str] = None
    error: Optional[str] = None
    created_at: str = field(default_factory=utc_now_iso)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None


class JobRunner:
    """Owns the worker thread and the job registry."""

    def __init__(self, generator: ImageGenerator, out_dir: Path) -> None:
        self._generator = generator
        self._out_dir = Path(out_dir)
        self._queue: Queue = Queue()
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    # -- lifecycle -------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._work, name="image-gen-worker", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        if self._thread is None:
            return
        self._queue.put(_SHUTDOWN)
        self._thread.join(timeout=timeout)
        self._thread = None

    # -- api -------------------------------------------------------------

    def submit(self, request: GenerateRequest) -> Job:
        job = Job(job_id=uuid.uuid4().hex[:12], request=request)
        with self._lock:
            self._jobs[job.job_id] = job
        self._queue.put(job)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def pending(self) -> int:
        return self._queue.qsize()

    def wait_idle(self, timeout: float = 30.0) -> bool:
        """Block until the queue drains. For tests and scripted runs."""
        import time

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                busy = any(
                    j.status in (JobStatus.QUEUED, JobStatus.RUNNING)
                    for j in self._jobs.values()
                )
            if not busy:
                return True
            time.sleep(0.02)
        return False

    # -- worker ----------------------------------------------------------

    def _work(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=0.2)
            except Empty:
                continue
            if item is _SHUTDOWN:
                return
            self._run_one(item)

    def _run_one(self, job: Job) -> None:
        with self._lock:
            job.status = JobStatus.RUNNING
            job.started_at = utc_now_iso()

        # generate() already converts render failures into a failed result and
        # writes the sidecar; anything escaping it is a bug worth surfacing on
        # the job rather than killing the worker thread.
        try:
            result = self._generator.generate(job.request)
            status = result.status
            file_name = result.path.name if result.path else None
            error = result.error
        except Exception as exc:  # noqa: BLE001
            status = AssetStatus.FAILED
            file_name = None
            error = f"{type(exc).__name__}: {exc}"

        try:
            upsert_scene_entry(
                self._out_dir,
                job.request.scene_id,
                ManifestEntry(
                    kind=job.request.kind,
                    seed=job.request.seed,
                    status=status,
                    file=file_name,
                    error=error,
                ),
            )
        except Exception as exc:  # noqa: BLE001 - never lose the job result
            error = error or f"manifest update failed: {exc}"

        with self._lock:
            if status is AssetStatus.OK:
                job.status = JobStatus.DONE
                # Relative to out/, which is what GET /files/ expects.
                job.file = f"{job.request.scene_id}/{file_name}"
            else:
                job.status = JobStatus.FAILED
                job.error = error
            job.finished_at = utc_now_iso()
