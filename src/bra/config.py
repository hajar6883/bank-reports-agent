from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# repo root when installed with `pip install -e .`; lets MCP clients launch the server from any cwd
ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(ROOT / ".env", ".env"), extra="ignore")

    database_url: str = "postgresql://bra:bra@localhost:5433/bra"
    llm_model: str = "openai:gpt-4.1-mini"
    judge_model: str = "openai:gpt-4.1-mini"
    embed_model: str = "local:intfloat/multilingual-e5-base"
    fts_config: str = "english"

    # chunking
    chunk_chars: int = 1200
    chunk_overlap: int = 200

    # retrieval
    candidates: int = 50  # per retriever, before fusion
    rrf_k: int = 60
    rag_top_k: int = 8  # baseline RAG context size
    agent_k_per_query: int = 5  # agent: hits kept per sub-query
    agent_max_context: int = 16
    agent_max_attempts: int = 2  # retrieve/answer/check loops

    price_input_per_m: float = 0.0
    price_output_per_m: float = 0.0
    tracing: str = "off"


@lru_cache
def get_settings() -> Settings:
    # Provider SDKs read their keys (OPENAI_API_KEY, ...) from os.environ, not from Settings.
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    load_dotenv(".env")
    return Settings()
