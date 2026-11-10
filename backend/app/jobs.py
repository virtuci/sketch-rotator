"""Job queue with replayable progress streams.

Two interchangeable backends behind one interface:

* InlineJobQueue - a thread pool inside the gateway, one thread per configured
  device. No Redis needed; this is what dev and the tests use.
* CeleryJobQueue - tasks go to Celery GPU workers; progress comes back through
  a Redis stream per job, so any gateway replica can serve any job's WebSocket.

Every job produces an ordered list of events:
  {"type": "queued"}
  {"type": "progress", "stage": str, "progress": float, "message": str}
  {"type": "result", "result": {...}}  |  {"type": "error", "message": str}
`stream()` always replays from the first event, so a client that connects
late (or reconnects) still sees the outcome.
"""
from __future__ import annotations

import asyncio
import json
import logging
import queue
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator, Protocol

from .config import Settings
from .pipeline import TASKS, PipelineError
from .storage import SessionNotFound

log = logging.getLogger("turnsketch.jobs")
TERMINAL = {"result", "error"}
JOB_TTL_SECONDS = 3600


class JobNotFound(KeyError):
    pass


def run_task(kind: str, payload: dict, emit, device: str) -> None:
    """Run one pipeline task, reporting through `emit(event_dict)`. Never raises."""
    def progress(stage: str, fraction: float, message: str) -> None:
        emit({"type": "progress", "stage": stage, "progress": round(float(fraction), 4), "message": message})

    try:
        emit({"type": "result", "result": TASKS[kind](payload, progress, device)})
    except (PipelineError, SessionNotFound, ValueError) as exc:
        emit({"type": "error", "message": str(exc), "retryable": False})
    except Exception:  # model or driver failure: log the detail, tell the artist it is on our side
        log.exception("job %s failed", kind)
        emit({"type": "error", "message": "The server failed while processing this job.", "retryable": True})


class JobQueue(Protocol):
    async def submit(self, kind: str, payload: dict, job_id: str | None = None) -> str: ...
    def stream(self, job_id: str) -> AsyncIterator[dict]: ...
    async def latest(self, job_id: str) -> dict: ...
    async def close(self) -> None: ...


async def wait_result(q: JobQueue, job_id: str) -> dict:
    """Consume a job's stream and return its terminal event."""
    async for event in q.stream(job_id):
        if event["type"] in TERMINAL:
            return event
    raise RuntimeError("job stream ended without a result")


class _Record:
    def __init__(self) -> None:
        self.events: list[dict] = [{"type": "queued"}]
        self.changed = asyncio.Condition()


class InlineJobQueue:
    def __init__(self, settings: Settings):
        self._devices: queue.Queue[str] = queue.Queue()
        for d in settings.devices or ("cpu",):
            self._devices.put(d)
        self._pool = ThreadPoolExecutor(max_workers=self._devices.qsize(), thread_name_prefix="ts-worker")
        self._jobs: dict[str, _Record] = {}

    async def submit(self, kind: str, payload: dict, job_id: str | None = None) -> str:
        job_id = job_id or uuid.uuid4().hex
        rec = self._jobs[job_id] = _Record()
        loop = asyncio.get_running_loop()

        async def append(event: dict) -> None:
            async with rec.changed:
                rec.events.append(event)
                rec.changed.notify_all()
            if event["type"] in TERMINAL:
                loop.call_later(JOB_TTL_SECONDS, self._jobs.pop, job_id, None)

        def emit(event: dict) -> None:
            asyncio.run_coroutine_threadsafe(append(event), loop)

        def work() -> None:
            device = self._devices.get()  # one job per device at a time
            try:
                run_task(kind, payload, emit, device)
            finally:
                self._devices.put(device)

        self._pool.submit(work)
        return job_id

    async def stream(self, job_id: str) -> AsyncIterator[dict]:
        rec = self._jobs.get(job_id)
        if rec is None:
            raise JobNotFound(job_id)
        sent = 0
        while True:
            async with rec.changed:
                await rec.changed.wait_for(lambda: len(rec.events) > sent)
                batch = rec.events[sent:]
            for event in batch:
                yield event
                if event["type"] in TERMINAL:
                    return
            sent += len(batch)

    async def latest(self, job_id: str) -> dict:
        rec = self._jobs.get(job_id)
        if rec is None:
            raise JobNotFound(job_id)
        return rec.events[-1]

    async def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


def stream_key(job_id: str) -> str:
    return f"turnsketch:job:{job_id}"


class CeleryJobQueue:
    def __init__(self, settings: Settings):
        import redis.asyncio as aioredis

        from .worker import celery_app

        self._celery = celery_app
        self._redis = aioredis.from_url(settings.redis_url, decode_responses=True)

    async def submit(self, kind: str, payload: dict, job_id: str | None = None) -> str:
        job_id = job_id or uuid.uuid4().hex
        key = stream_key(job_id)
        await self._redis.xadd(key, {"event": json.dumps({"type": "queued"})})
        await self._redis.expire(key, JOB_TTL_SECONDS)
        await asyncio.to_thread(
            self._celery.send_task, "turnsketch.run", args=[kind, payload, job_id], task_id=job_id
        )
        return job_id

    async def stream(self, job_id: str) -> AsyncIterator[dict]:
        key = stream_key(job_id)
        if not await self._redis.exists(key):
            raise JobNotFound(job_id)
        last = "0"
        while True:
            reply = await self._redis.xread({key: last}, block=15000, count=100)
            if not reply:
                if not await self._redis.exists(key):
                    raise JobNotFound(job_id)
                continue
            for last, fields in reply[0][1]:
                event = json.loads(fields["event"])
                yield event
                if event["type"] in TERMINAL:
                    return

    async def latest(self, job_id: str) -> dict:
        entries = await self._redis.xrevrange(stream_key(job_id), count=1)
        if not entries:
            raise JobNotFound(job_id)
        return json.loads(entries[0][1]["event"])

    async def close(self) -> None:
        await self._redis.aclose()


def make_queue(settings: Settings) -> JobQueue:
    if settings.queue_backend == "celery":
        return CeleryJobQueue(settings)
    if settings.queue_backend == "inline":
        return InlineJobQueue(settings)
    raise ValueError(f"unknown TS_QUEUE_BACKEND {settings.queue_backend!r}")
