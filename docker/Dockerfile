FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends     python3.12     python3-pip     ffmpeg     ca-certificates     && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# Environment only: no pipeline code is copied into this image.
# Mount code (api_seed_openai_easy_no_short_step/) and data volumes at run time.

# No secrets are baked into this image. Pass credentials/config at container start,
# e.g. `docker run -e ARK_API_KEY=... -e API_BASE_URL=... -e MODEL_NAME=...` or --env-file.
# Required: ARK_API_KEY (or API_KEY). Optional: API_BASE_URL, MODEL_NAME,
# ARK_REASONING_EFFORT, MAX_TOKENS, TEMPERATURE, VIDEO_UNDERSTANDING_FPS.
