# One image for the API and the Streamlit dashboard (different start commands in docker-compose.yml).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

# libgomp1: OpenMP runtime needed by XGBoost; curl: container health checks
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .
# Slow / flaky networks: long timeout, many retries, and a BuildKit cache so a failed build does not re-download
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 180 --retries 10 -r requirements-docker.txt

# Code + only the model files the deployed detector needs (.dockerignore keeps the 1.3 GB models/ folder out)
COPY api ./api
COPY dashboard ./dashboard
COPY src ./src
COPY models ./models

RUN useradd --create-home --uid 10001 soc && mkdir -p /app/data /app/reports/drift && chown -R soc /app
USER soc

EXPOSE 8000 8501
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
    CMD curl -fsS http://localhost:8000/health || exit 1
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
