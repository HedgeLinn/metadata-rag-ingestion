"""Embedding 工厂：local / ali / default 三源，统一对外提供
`embed_documents(texts)` 与 `embed_query(text)`（OpenAIEmbeddings 兼容接口）。
"""
from functools import lru_cache

from langchain_openai import OpenAIEmbeddings

from config import (
    EMBED_MODEL, EMBED_DIM, EMBED_BASE_URL, EMBED_API_KEY,
    EMBED_MODEL_ALI, EMBED_DIM_ALI, EMBED_BASE_URL_ALI, EMBED_API_KEY_ALI,
    EMBED_MODEL_LOCAL, EMBED_DIM_LOCAL,
    EMBED_SOURCE,
)


class LocalEmbeddings:
    """本地 sentence-transformers 模型的最小适配（无密钥、离线可用）。

    首次加载需联网下载权重（约 95MB），之后走 HF 本地缓存。
    """

    def __init__(self, model: str):
        from sentence_transformers import SentenceTransformer
        try:
            self._model = SentenceTransformer(model)
        except Exception as e:  # 模型缺失/下载失败时给出可操作的提示
            raise RuntimeError(
                f"本地 embedding 模型加载失败（{model}）：{e}\n"
                "首次运行需要联网下载模型权重；也可设置 HF_ENDPOINT 指向镜像。"
            ) from e
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


@lru_cache(maxsize=None)
def get_embeddings(source: str | None = None):
    """按 source 缓存的单例工厂：并发检索时所有线程共享同一模型实例，
    避免每线程重复初始化 SentenceTransformer（重复联网检查/重载会引发下载竞争失败）。"""
    ec = get_embed_config(source)
    if ec["source"] == "local":
        return LocalEmbeddings(model=ec["model"])
    return OpenAIEmbeddings(
        model=ec["model"], base_url=ec["base_url"], api_key=ec["api_key"],
        tiktoken_enabled=False, check_embedding_ctx_length=False,
        chunk_size=10,
    )