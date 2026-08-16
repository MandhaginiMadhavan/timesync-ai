FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# FFmpeg supplies both ffmpeg and ffprobe. No compiler toolchain is needed by
# the pinned Python dependencies on the supported Linux/Python platform.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt ./

# Install the validated CPU-only PyTorch build from PyTorch's official CPU
# wheel index. Whisper is installed without dependencies so its Linux-only
# Triton dependency (a CUDA optimization unused on CPU) is not included.
RUN python -m pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.13.0+cpu \
    && python -m pip install --requirement requirements-docker.txt \
    && python -m pip install --no-deps openai-whisper==20250625

COPY app.py main.py ./
COPY src/ ./src/
COPY .streamlit/ ./.streamlit/

RUN mkdir -p /app/.tools/whisper /app/.tools/ui_uploads /app/output \
    && useradd --create-home --shell /usr/sbin/nologin timesync \
    && chown -R timesync:timesync /app

USER timesync

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os, urllib.request; port = os.environ.get('PORT', '8501'); urllib.request.urlopen(f'http://127.0.0.1:{port}/_stcore/health', timeout=3)" || exit 1

CMD ["sh", "-c", "exec streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8501} --server.headless=true --browser.gatherUsageStats=false"]
