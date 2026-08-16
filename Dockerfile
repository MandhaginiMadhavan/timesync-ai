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

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --requirement requirements.txt

COPY app.py main.py ./
COPY src/ ./src/
COPY .streamlit/ ./.streamlit/

RUN mkdir -p /app/.tools/whisper /app/.tools/ui_uploads /app/output \
    && useradd --create-home --shell /usr/sbin/nologin timesync \
    && chown -R timesync:timesync /app

USER timesync

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)" || exit 1

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]
