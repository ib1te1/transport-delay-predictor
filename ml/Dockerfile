FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY busdelay ./busdelay

# default paths are ../data/..., so the repository's data/ is mounted at /data
ENTRYPOINT ["python", "-m", "busdelay"]
