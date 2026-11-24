# ---------- Base: pinned production Python image from the accessible MCR Docker Hub mirror ----------
ARG PYTHON_IMAGE=mcr.microsoft.com/mirror/docker/library/python@sha256:2c5bc243b1cc47985ee4fb768bb0bbd4490481c5d0897a62da31b7f30b7304a7

# ---------- Builder: compile the package wheel without development extras ----------
FROM ${PYTHON_IMAGE} AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN python -m pip install --no-cache-dir --upgrade \
    pip==24.0 \
    setuptools==83.0.0 \
    wheel==0.46.3

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip wheel \
    --no-deps \
    --no-build-isolation \
    --wheel-dir /wheels \
    .

# ---------- Runtime: serving dependencies, package wheel, and runtime config only ----------
FROM ${PYTHON_IMAGE} AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SMARTPREDICT_ENVIRONMENT=production \
    SMARTPREDICT_API_HOST=0.0.0.0 \
    SMARTPREDICT_API_PORT=8000 \
    SMARTPREDICT_API_PREFIX=/api/v1 \
    SMARTPREDICT_CONFIGS_DIR=/app/configs \
    SMARTPREDICT_ARTIFACTS_DIR=/app/artifacts \
    SMARTPREDICT_MODELS_DIR=/app/artifacts/models \
    SMARTPREDICT_LOG_FORMAT=json \
    SMARTPREDICT_LOG_LEVEL=INFO \
    SMARTPREDICT_LOG_TO_FILE=false

WORKDIR /app

# Explicit pins plus --no-deps keep development, plotting, notebook, Optuna,
# SHAP, and NVIDIA GPU packages out of the CPU inference image.
COPY requirements-runtime.txt ./
RUN python -m pip install --no-cache-dir --no-deps -r requirements-runtime.txt

COPY --from=builder /wheels/*.whl /tmp/
RUN python -m pip install --no-cache-dir --no-deps /tmp/*.whl \
    && rm -f /tmp/*.whl \
    && python -m pip uninstall -y pip setuptools wheel

COPY configs ./configs

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home --home-dir /app \
        --shell /usr/sbin/nologin app \
    && mkdir -p /app/artifacts/models \
    && chown -R app:app /app

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-m", "smartpredict.deployment.healthcheck"]

CMD ["python", "-m", "smartpredict.cli.serve"]
