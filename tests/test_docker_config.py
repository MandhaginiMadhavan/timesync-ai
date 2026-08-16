"""Static validation for the production Docker configuration."""

from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_dockerfile_has_required_runtime_contract() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert dockerfile.startswith("FROM python:3.11-slim")
    assert "apt-get install --yes --no-install-recommends ffmpeg" in dockerfile
    assert "COPY requirements.txt ./" in dockerfile
    assert "EXPOSE 8501" in dockerfile
    assert 'USER timesync' in dockerfile
    assert '["streamlit", "run", "app.py"' in dockerfile
    assert "/_stcore/health" in dockerfile


def test_docker_context_excludes_local_and_generated_assets() -> None:
    ignored = {
        line.strip()
        for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert {".git", ".tools", ".conda", "data", "output", "tests"} <= ignored
    assert {"*.pt", "*.pth", "*.mp4", "*.wav"} <= ignored


def test_streamlit_container_settings_and_cache_volume() -> None:
    config = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert 'address = "0.0.0.0"' in config
    assert "port = 8501" in config
    assert "headless = true" in config
    assert "gatherUsageStats = false" in config
    assert "whisper-cache:/app/.tools/whisper" in compose
    assert "timesync-output:/app/output" in compose
