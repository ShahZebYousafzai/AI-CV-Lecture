# syntax=docker/dockerfile:1

# ============================================================================
# Stage 1: builder
#
# Compilers, pip caches and build metadata all live here and none of it ships.
# The only thing that crosses into the final image is the finished virtualenv.
# ============================================================================
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# A self-contained venv is the cleanest thing to copy between stages: one
# directory, no site-packages archaeology.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Dependencies are copied and installed BEFORE the application code, so that
# editing a route does not invalidate the layer that took four minutes to
# build. Order your Dockerfile from least to most frequently changed.
COPY requirements-torch-cpu.txt requirements.txt ./

# CPU-only PyTorch first, so the CUDA wheels never enter the resolution.
RUN pip install --no-cache-dir -r requirements-torch-cpu.txt
RUN pip install --no-cache-dir -r requirements.txt

# Bake the weights into the image. The alternative, downloading at startup,
# means slow cold starts, a hard dependency on the network from inside your
# VPC, and a container that can fail long after it was deployed.
COPY scripts/download_model.py scripts/download_model.py
RUN python scripts/download_model.py && ls -lh models/

# Prune what inference never imports. Ultralytics and PyTorch declare a lot
# of dependencies that only training, plotting and dataset tooling touch, and
# pip has no idea which half of a library you actually use.
#
#   polars      Ultralytics dataset tooling      about 172 MB
#   matplotlib  plotting, only used by training  about  66 MB with fontTools
#   pip, setuptools, wheel  build tools, not needed at runtime
#   torch/test, torch/include  C++ headers and test fixtures
#
# Verified by importing the app and running one inference afterwards. If a
# future version of a library starts importing one of these, the test suite
# fails loudly rather than the container failing quietly in production.
RUN SP=/opt/venv/lib/python3.11/site-packages \
    && rm -rf $SP/polars $SP/_polars_runtime_* $SP/polars-*.dist-info \
    && rm -rf $SP/matplotlib $SP/matplotlib-*.dist-info $SP/mpl_toolkits \
              $SP/pylab.py $SP/fontTools $SP/fonttools-*.dist-info \
    && rm -rf $SP/pip $SP/pip-*.dist-info $SP/setuptools $SP/setuptools-*.dist-info \
              $SP/wheel $SP/wheel-*.dist-info $SP/pkg_resources \
    && rm -rf $SP/torch/test $SP/torch/include $SP/torch/utils/tensorboard \
    && find /opt/venv -name "__pycache__" -type d -prune -exec rm -rf {} + \
    && find /opt/venv -name "*.pyc" -delete \
    && du -sh /opt/venv


# ============================================================================
# Stage 2: runtime
#
# Same base, but only the venv, the weights and the source. No compilers,
# no pip cache, no build tooling to keep patched.
# ============================================================================
FROM python:3.11-slim AS runtime

# libgl1 and libglib2.0-0 are OpenCV's runtime shared libraries. Ultralytics
# imports cv2, and without these the container dies on import with
# "libGL.so.1: cannot open shared object file".
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    # Ultralytics and Matplotlib write settings on first import. Without a
    # writable location they warn on every start and fall back to /tmp.
    YOLO_CONFIG_DIR=/home/appuser/.config \
    MPLCONFIGDIR=/home/appuser/.cache/matplotlib \
    # Application configuration. Every one of these can be overridden with
    # `docker run -e`, which is how the same image serves dev and prod.
    APP_ENV=docker \
    APP_DEVICE=cpu \
    APP_TORCH_THREADS=2 \
    APP_INFERENCE_CONCURRENCY=2 \
    APP_LOG_LEVEL=INFO

# Run as a normal user. A container process that is root is root on the host
# kernel if anything escapes, and nothing in this image needs it.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /home/appuser/.config /home/appuser/.cache/matplotlib \
    && chown -R appuser:appuser /home/appuser

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=appuser:appuser /build/models /app/models
COPY --chown=appuser:appuser app/ /app/app/
COPY --chown=appuser:appuser scripts/ /app/scripts/

USER appuser

EXPOSE 8000

# Docker and ECS use this to decide whether the container is healthy. Note it
# hits readiness, not liveness: a container whose model has not finished
# loading should not be sent traffic.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl --fail --silent http://localhost:8000/health/ready || exit 1

# Exec form, so uvicorn is PID 1 and receives SIGTERM directly. With the
# shell form, the shell swallows the signal and Docker kills the container
# after the 10 second grace period instead of shutting down cleanly.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
