import os
import sys
from pathlib import Path

import psycopg
import pytest

FIXTURES = Path(__file__).parent / "fixtures"
sys.path.insert(0, str(FIXTURES))

TEST_DB = os.environ.get("TEST_DATABASE_URL", "postgresql://bra:bra@localhost:5433/bra_test")

# Everything in the test session uses the fake LLM, hash embeddings and fictional banks.
os.environ.update({
    "DATABASE_URL": TEST_DB,
    "EMBED_MODEL": "hash:256",
    "LLM_MODEL": "fake",
    "JUDGE_MODEL": "fake",
    "BRA_BANKS_FILE": str(FIXTURES / "banks.yaml"),
    "TRACING": "off",
})


def _db_up() -> bool:
    try:
        with psycopg.connect(TEST_DB, connect_timeout=2):
            return True
    except Exception:
        return False


DB_UP = _db_up()


def pytest_collection_modifyitems(items):
    if DB_UP:
        return
    skip = pytest.mark.skip(reason=f"no database at {TEST_DB}")
    for item in items:
        if "db" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def corpus(tmp_path_factory) -> list[Path]:
    from make_corpus import build

    return build(tmp_path_factory.mktemp("corpus"))


@pytest.fixture(scope="session")
def settings():
    from bra.config import get_settings

    get_settings.cache_clear()
    return get_settings()


@pytest.fixture(scope="session")
def store(corpus, settings):
    from bra.embed import get_embedder
    from bra.store import Store

    s = Store(settings.database_url, get_embedder(settings.embed_model), settings.fts_config)
    s.reset()
    s.init()
    for p in corpus:
        s.ingest(p)
    return s
