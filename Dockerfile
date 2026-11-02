# CPU-only image for the text-label pipeline.
# Build from this repository root:
#   docker build -t text-label-cpu:latest .
FROM ubuntu:24.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv python3-pip ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY docker/requirements.txt /tmp/requirements.txt
# The build host's local proxy is not reliable for PyPI; use the reachable
# Tsinghua mirror only while building. Runtime has no dependency on this URL.
RUN python3 -m venv "$VIRTUAL_ENV" \
    && env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
       pip install --no-cache-dir --timeout 60 \
       -i https://pypi.tuna.tsinghua.edu.cn/simple -r /tmp/requirements.txt \
    && rm -f /tmp/requirements.txt

COPY api/api_seed_openai_easy_no_short_step/ /app/pipeline/

# No API key, data location, output location, or host-specific path is baked in.
# Supply ARK_API_KEY (or API_KEY) and paths at `docker run` time.
ENTRYPOINT ["python3", "/app/pipeline/pipeline.py"]
CMD ["--help"]
