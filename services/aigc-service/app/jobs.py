from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from .backends.base import AigcBackend
from .config import Settings
from .schemas import JobRecord, JobStatus


class IdempotencyConflict(RuntimeError):
    pass


class JobStore:
    def __init__(self, backend: AigcBackend, cfg: Settings):
        self.backend = backend
        self.cfg = cfg
        self._jobs: dict[str, JobRecord] = {}
        self._keys: dict[tuple[str, str], tuple[str, str]] = {}
        self._lock = asyncio.Lock()
        # Model calls are synchronous and GPU-heavy. Offload them so polling
        # remains responsive, then serialize them in this worker to prevent two
        # requests from silently colliding on the configured GPU.
        self._generation_lock = asyncio.Lock()
        self._tasks: set[asyncio.Task[None]] = set()
        self._counter = 0

    @staticmethod
    def _fingerprint(payload: dict[str, Any]) -> str:
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()

    async def submit(self, kind: str, payload: dict[str, Any], idempotency_key: str | None) -> tuple[JobRecord, bool]:
        fingerprint = self._fingerprint(payload)
        async with self._lock:
            if idempotency_key:
                prior = self._keys.get((kind, idempotency_key))
                if prior:
                    prior_fingerprint, job_id = prior
                    if prior_fingerprint != fingerprint:
                        raise IdempotencyConflict("Idempotency-Key was already used with a different request")
                    return self._jobs[job_id].model_copy(deep=True), True
            self._counter += 1
            stable = hashlib.sha256(f"{kind}:{fingerprint}:{self._counter}".encode()).hexdigest()[:20]
            job_id = f"job_{stable}"
            now = datetime.now(timezone.utc)
            job = JobRecord(
                id=job_id, kind=kind, status=JobStatus.queued,
                created_at=now, updated_at=now,
                run=self.backend.generation_run(kind, payload),
            )
            self._jobs[job_id] = job
            if idempotency_key:
                self._keys[(kind, idempotency_key)] = (fingerprint, job_id)
            task = asyncio.create_task(self._execute(job_id, payload))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            return job.model_copy(deep=True), False

    async def _execute(self, job_id: str, payload: dict[str, Any]) -> None:
        await asyncio.sleep(self.cfg.fake_job_start_delay_ms / 1000)
        async with self._lock:
            job = self._jobs[job_id]
            job.status = JobStatus.running
            job.updated_at = datetime.now(timezone.utc)
        await asyncio.sleep(self.cfg.fake_job_run_delay_ms / 1000)
        try:
            async with self._generation_lock:
                result = await asyncio.to_thread(
                    self.backend.generate,
                    self._jobs[job_id].kind,
                    payload,
                    self._jobs[job_id].run,
                )
        except Exception as exc:
            async with self._lock:
                job = self._jobs[job_id]
                job.status = JobStatus.failed
                job.error = str(exc)
                job.updated_at = datetime.now(timezone.utc)
            return
        async with self._lock:
            job = self._jobs[job_id]
            job.status = JobStatus.succeeded
            job.result = result
            job.updated_at = datetime.now(timezone.utc)

    async def get(self, job_id: str) -> JobRecord | None:
        async with self._lock:
            job = self._jobs.get(job_id)
            return job.model_copy(deep=True) if job else None
