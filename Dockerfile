# ClimateResilience-India: dashboards and advisory bot in one image.
#
#   docker build -t climateresilience .                                   # serving dependencies (requirements.txt)
#   docker build --build-arg REQUIREMENTS=requirements.lock -t climateresilience:full .   # everything, to run the tests
#   docker run --rm -p 8501:8501 climateresilience                        # dashboards on http://localhost:8501
#   docker run --rm -p 8000:8000 climateresilience uvicorn advisory.bot:app --host 0.0.0.0 --port 8000
#
# The image holds no data of its own: on first start the app copies the demo bundle (data/sample) into place.
# Mount real pipeline outputs over /app/data/processed and /app/models to use them instead.
FROM python:3.11-slim

ARG REQUIREMENTS=requirements.txt
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgomp1: OpenMP runtime used by xgboost and torch; curl: container health check.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app

WORKDIR /app

# Dependencies first so that code changes do not invalidate this layer. torch comes from the CPU wheel index
# (the default Linux wheel pulls gigabytes of CUDA libraries that a CPU-only service never uses).
COPY requirements.txt requirements.lock ./
# xgboost's Linux wheel pulls nvidia-nccl-cu12 (about 470 MB) for multi-GPU training; CPU inference never loads it.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r "${REQUIREMENTS}" \
    && pip uninstall -y nvidia-nccl-cu12 \
    && python -c "import xgboost, torch, shap, geopandas"

COPY --chown=app:app . .
RUN mkdir -p data/raw data/interim data/processed models reports dashboards/static && chown -R app:app /app

USER app
EXPOSE 8501 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl --fail --silent http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "dashboards/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
