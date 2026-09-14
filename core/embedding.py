"""Embedding 工厂：local / ali / default 三源，统一对外提供
`embed_documents(texts)` 与 `embed_query(text)`（OpenAIEmbeddings 兼容接口）。
"""
from langchain_openai import OpenAIEmbeddings

from config import (
    EMBED_MODEL, EMBED_DIM, EMBED_BASE_URL, EMBED_API_KEY,
    EMBED_MODEL_ALI, EMBED_DIM_ALI, EMBED_BASE_URL_ALI, EMBED_API_KEY_ALI,
    EMBED_MODEL_LOCAL, EMBED_DIM_LOCAL,
    EMBED_SOURCE,
)


class LocalEmbeddings:
    """本地 sentence-transformers 模型的最小适配（无密钥、离线可用）。"""

    def __init__(self, model: str):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model)
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False,
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(
            text, normalize_embeddings=True, show_progress_bar=False,
        ).tolist()


def get_embed_config(source: str | None = None) -> dict:
    s = source or EMBED_SOURCE
    if s == "ali":
        return {
            "source": "ali", "model": EMBED_MODEL_ALI, "dim": EMBED_DIM_ALI,
            "base_url": EMBED_BASE_URL_ALI, "api_key": EMBED_API_KEY_ALI,
        }
    if s == "default":
        return {
            "source": "default", "model": EMBED_MODEL, "dim": EMBED_DIM,
            "base_url": EMBED_BASE_URL, "api_key": EMBED_API_KEY,
        }
    return {
        "source": "local", "model": EMBED_MODEL_LOCAL, "dim": EMBED_DIM_LOCAL,
    }


def get_embeddings(source: str | None = None):
    ec = get_embed_config(source)
    if ec["source"] == "local":
        return LocalEmbeddings(model=ec["model"])
    return OpenAIEmbeddings(
        model=ec["model"], base_url=ec["base_url"], api_key=ec["api_key"],
        tiktoken_enabled=False, check_embedding_ctx_length=False,
        chunk_size=10,
    )