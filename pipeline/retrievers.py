"""Fusion retrievers — Weighted + RRF."""
from __future__ import annotations

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import Callbacks


class WeightedFusionRetriever(BaseRetriever):
    """加权融合多个检索器：各自召回 → 归一化分数 → 按权重合并 → top_k。"""

    retrievers: list[BaseRetriever]
    weights: list[float]
    top_k: int = 5

    def _get_relevant_documents(self, query: str, *, callbacks: Callbacks = None) -> list[Document]:
        all_doc_lists: list[list[Document]] = []
        for retriever in self.retrievers:
            try:
                all_doc_lists.append(retriever.invoke(query))
            except Exception:
                all_doc_lists.append([])

        # 加权归一化合并
        scored: dict[str, tuple[Document, float]] = {}
        for weight, docs in zip(self.weights, all_doc_lists):
            if not docs:
                continue
            raw_scores = [d.metadata.get("score", 0.0) for d in docs]
            min_s, max_s = min(raw_scores), max(raw_scores)
            for doc, raw in zip(docs, raw_scores):
                doc_id = doc.metadata.get("content_id", "") or doc.page_content[:80]
                norm = (raw - min_s) / (max_s - min_s + 1e-9) * weight
                if doc_id not in scored or norm > scored[doc_id][1]:
                    scored[doc_id] = (doc, norm)

        sorted_docs = sorted(scored.values(), key=lambda x: x[1], reverse=True)
        result = []
        for doc, fusion_score in sorted_docs[:self.top_k]:
            doc.metadata["fusion_score"] = round(fusion_score, 4)
            result.append(doc)
        return result

    async def _aget_relevant_documents(self, query: str, *, callbacks: Callbacks = None) -> list[Document]:
        return self._get_relevant_documents(query, callbacks=callbacks)


class RRFFusionRetriever(BaseRetriever):
    """RRF (Reciprocal Rank Fusion) 融合：Σ 1/(k + rank_i)，不依赖原始分数。"""

    retrievers: list[BaseRetriever]
    top_k: int = 5
    k: int = 60

    def _get_relevant_documents(self, query: str, *, callbacks: Callbacks = None) -> list[Document]:
        all_doc_lists: list[list[Document]] = []
        for retriever in self.retrievers:
            try:
                all_doc_lists.append(retriever.invoke(query))
            except Exception:
                all_doc_lists.append([])

        rrf: dict[str, tuple[Document, float]] = {}
        for docs in all_doc_lists:
            for rank, doc in enumerate(docs, start=1):
                doc_id = doc.metadata.get("content_id", "") or doc.page_content[:80]
                score = 1.0 / (self.k + rank)
                if doc_id not in rrf or score > rrf[doc_id][1]:
                    rrf[doc_id] = (doc, score)

        sorted_docs = sorted(rrf.values(), key=lambda x: x[1], reverse=True)
        result = []
        for doc, rrf_score in sorted_docs[:self.top_k]:
            doc.metadata["rrf_score"] = round(rrf_score, 4)
            result.append(doc)
        return result

    async def _aget_relevant_documents(self, query: str, *, callbacks: Callbacks = None) -> list[Document]:
        return self._get_relevant_documents(query, callbacks=callbacks)
