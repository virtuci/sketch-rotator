"""Celery GPU worker.

Run one worker process per GPU, each started with its own CUDA_VISIBLE_DEVICES
so models load once and stay resident:

    CUDA_VISIBLE_DEVICES=0 TS_DEVICE=cuda celery -A app.worker worker --pool=solo -Q gpu

`--pool=solo` keeps a single process that owns the GPU; concurrency across
artists comes from running one worker per card.
"""
from __future__ import annotations

import json
import os

import redis
from celery import Celery

from .config import get_settings

settings = get_settings()
celery_app = Celery("turnsketch", broker=settings.redis_url)
celery_app.conf.update(
    task_default_queue="gpu",
    task_acks_late=True,  # a job survives a worker crash and is picked up by another card
    worker_prefetch_multiplier=1,  # do not hoard jobs while another GPU sits idle
    task_ignore_result=True,  # outcomes travel on the Redis stream instead
    broker_connection_retry_on_startup=True,
)
_redis = redis.Redis.from_url(settings.redis_url, decode_responses=True)


@celery_app.task(name="turnsketch.run")
def run(kind: str, payload: dict, job_id: str) -> None:
    from .jobs import JOB_TTL_SECONDS, run_task, stream_key

    key = stream_key(job_id)

    def emit(event: dict) -> None:
        _redis.xadd(key, {"event": json.dumps(event)})
        _redis.expire(key, JOB_TTL_SECONDS)

    run_task(kind, payload, emit, os.environ.get("TS_DEVICE", "cuda"))
