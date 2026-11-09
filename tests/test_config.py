"""config.py on start-up: outside development the Qdrant collection must be
named, and never a development one; development keeps its own default.

Each case runs config.py from a copy in an empty folder, in a fresh process,
so no local .env file can fill anything in."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

CONFIG = Path(__file__).resolve().parent.parent / "config.py"


def start(tmp_path: Path, **env: str) -> subprocess.CompletedProcess:
    shutil.copy(CONFIG, tmp_path / "config.py")
    base = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "SYSTEMROOT")}
    return subprocess.run(
        [sys.executable, "-c", "import config; print(config.QDRANT_COLLECTION)"],
        cwd=tmp_path, env={**base, **env}, capture_output=True, text=True, timeout=60,
    )


PRODUCTION = {"QDRANT_URL": "https://qdrant.example.cloud:6333"}


def test_production_with_its_collection_starts(tmp_path):
    run = start(tmp_path, **PRODUCTION, QDRANT_COLLECTION="socrate_docs")
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == "socrate_docs"


@pytest.mark.parametrize("collection", [None, "", "  "])
def test_production_without_a_collection_refuses_to_start(tmp_path, collection):
    env = dict(PRODUCTION) if collection is None else {**PRODUCTION, "QDRANT_COLLECTION": collection}
    run = start(tmp_path, **env)
    assert run.returncode != 0
    assert "QDRANT_COLLECTION is not set" in run.stderr


def test_production_with_a_development_collection_refuses_to_start(tmp_path):
    run = start(tmp_path, **PRODUCTION, QDRANT_COLLECTION="socrate_docs_dev")
    assert run.returncode != 0
    assert "a development collection" in run.stderr


def test_development_defaults_to_its_own_collection(tmp_path):
    (tmp_path / ".env.development").write_text("")
    run = start(tmp_path, SOCRATE_ENV="development", QDRANT_URL="http://localhost:6333", S3_ENDPOINT="http://localhost:9000")
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == "socrate_docs_dev"


def test_the_api_key_as_the_topic_model_refuses_to_start(tmp_path):
    run = start(tmp_path, **PRODUCTION, QDRANT_COLLECTION="socrate_docs", COHERE_API_KEY="k-123", TOPIC_AI_MODEL="k-123")
    assert run.returncode != 0
    assert "TOPIC_AI_MODEL is set to the Cohere API key" in run.stderr


def test_the_topic_model_is_off_unless_named(tmp_path):
    shutil.copy(CONFIG, tmp_path / "config.py")
    base = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "SYSTEMROOT")}
    run = subprocess.run([sys.executable, "-c", "import config; print(config.TOPIC_AI_MODEL)"], cwd=tmp_path,
                         env={**base, **PRODUCTION, "QDRANT_COLLECTION": "socrate_docs", "COHERE_API_KEY": "k-123"}, capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == "None"
