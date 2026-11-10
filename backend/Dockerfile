# Two images from one file:
#   --target gateway   CPU-only API gateway
#   --target worker    CUDA worker with the learned models

FROM python:3.12-slim AS gateway
WORKDIR /srv
COPY pyproject.toml ./
COPY app ./app
RUN pip install --no-cache-dir ".[queue]"
ENV TS_DATA_DIR=/data TS_QUEUE_BACKEND=celery
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime AS worker
WORKDIR /srv
COPY pyproject.toml ./
COPY app ./app
RUN pip install --no-cache-dir ".[queue,ml]"
# Optional models, needed only when the matching TS_* switch is on:
#   TS_RECONSTRUCTOR=hunyuan3d  ->  pip install git+https://github.com/Tencent/Hunyuan3D-2.git
#   TS_SEGMENTER=sam2           ->  pip install git+https://github.com/facebookresearch/sam2.git
ENV TS_DATA_DIR=/data TS_QUEUE_BACKEND=celery TS_DEVICE=cuda HF_HOME=/models
# One process that owns one GPU; scale by running one container per card.
CMD ["celery", "-A", "app.worker", "worker", "--pool=solo", "-Q", "gpu", "--loglevel=info"]
