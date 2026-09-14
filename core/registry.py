"""YAML-driven component factory — builds LangChain TextSplitters, Retrievers, Compressors."""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import yaml
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class ComponentFactory:
    """从 strategies.yaml 构建 LangChain 组件。"""

    def __init__(self, config_path: Path | None = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "strategies.yaml"
        with open(config_path, encoding="utf-8") as f:
            self._cfg = yaml.safe_load(f)

    # ── 切分器 ──

    def get_splitter(self, name: str, embeddings: Embeddings | None = None):
        """获取 TextSplitter 实例。"""
        entry = self._cfg["chunkers"][name]
        cls = self._import_class(entry["class"])
        params = dict(entry.get("params", {}))
        if entry.get("requires_embeddings") and embeddings is not None:
            params["embeddings"] = embeddings
        return cls(**params)

    def list_chunkers(self) -> list[str]:
        return list(self._cfg["chunkers"].keys())

    def chunker_info(self, name: str) -> dict:
        return self._cfg["chunkers"].get(name, {})

    # ── 检索器 ──

    def get_retriever(self, name: str, vector_store=None, documents: list[Document] | None = None,
                      embeddings: Embeddings | None = None, top_k: int = 5):
        """构建检索器。

        vector: 从 QdrantVectorStore.as_retriever() 创建
        bm25: 从文档列表构建 BM25Retriever
        hybrid: vector + bm25 加权融合（自研 WeightedFusionRetriever）
        """
        if name == "vector":
            if vector_store is None:
                raise ValueError("vector retriever requires vector_store")
            return vector_store.as_retriever(search_kwargs={"k": top_k})

        elif name == "bm25":
            from langchain_community.retrievers import BM25Retriever
            if documents is None:
                raise ValueError("bm25 retriever requires documents")
            retriever = BM25Retriever.from_documents(documents)
            retriever.k = top_k
            return retriever

        elif name == "hybrid":
            if vector_store is None or documents is None:
                raise ValueError("hybrid retriever requires both vector_store and documents")
            from pipeline.retrievers import WeightedFusionRetriever
            from langchain_community.retrievers import BM25Retriever
            weights = self._cfg["retrievers"]["hybrid"].get("params", {}).get("weights", [0.3, 0.7])
            vec_ret = vector_store.as_retriever(search_kwargs={"k": top_k * 2})
            bm25_ret = BM25Retriever.from_documents(documents)
            bm25_ret.k = top_k * 2
            return WeightedFusionRetriever(retrievers=[vec_ret, bm25_ret], weights=weights, top_k=top_k)

        elif name == "hybrid_rrf":
            if vector_store is None or documents is None:
                raise ValueError("hybrid_rrf retriever requires both vector_store and documents")
            from pipeline.retrievers import RRFFusionRetriever
            from langchain_community.retrievers import BM25Retriever
            rrf_k = self._cfg["retrievers"]["hybrid_rrf"].get("params", {}).get("k", 60)
            vec_ret = vector_store.as_retriever(search_kwargs={"k": top_k * 2})
            bm25_ret = BM25Retriever.from_documents(documents)
            bm25_ret.k = top_k * 2
            return RRFFusionRetriever(retrievers=[vec_ret, bm25_ret], k=rrf_k, top_k=top_k)

        else:
            entry = self._cfg["retrievers"][name]
            cls = self._import_class(entry["class"])
            return cls(**entry.get("params", {}))

    def list_retrievers(self) -> list[str]:
        return list(self._cfg["retrievers"].keys())

    def retriever_info(self, name: str) -> dict:
        return self._cfg["retrievers"].get(name, {})

    # ── 重排器 ──

    def get_reranker(self, name: str):
        """获取 BaseDocumentCompressor 实例。"""
        entry = self._cfg["rerankers"][name]
        cls = self._import_class(entry["class"])
        return cls(**entry.get("params", {}))

    def list_rerankers(self) -> list[str]:
        return list(self._cfg["rerankers"].keys())

    # ── 工具 ──

    @staticmethod
    def _import_class(full_path: str):
        """动态导入类: 'langchain_text_splitters.RecursiveCharacterTextSplitter'"""
        parts = full_path.rsplit(".", 1)
        if len(parts) == 2:
            module = importlib.import_module(parts[0])
            return getattr(module, parts[1])
        raise ValueError(f"Invalid class path: {full_path}")


# 全局单例
factory = ComponentFactory()
