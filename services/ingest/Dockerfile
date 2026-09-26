FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# curl serves the compose healthcheck. No build toolchain is installed:
# every pinned dependency ships a manylinux wheel for cp313, and
# psycopg[binary] bundles libpq. Adding a compiler back means a
# dependency lost its wheel — check that before installing one.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Built with the repository root as context, so paths start there.
COPY services/ingest/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The shared packages go in after the pinned dependencies and without
# pulling their own: pyproject.toml carries ranges, requirements.txt the
# exact versions, and pip check fails the build if a pin falls outside a
# range. Editing shared code also leaves the layer above cached.
COPY packages /packages
RUN pip install --no-cache-dir --no-deps /packages/contracts /packages/common \
    && pip check

COPY services/ingest/app ./app

EXPOSE 8000
EXPOSE 9201

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
