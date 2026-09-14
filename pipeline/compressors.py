"""Custom document compressors — NoOp, LLMRerank, BGERerank.

遵循 BaseDocumentCompressor 接口（compress_documents）。
"""
from __future__ import annotations

import re

from langchain_core.documents import Document, BaseDocumentCompressor
from langchain_core.callbacks import Callbacks
from langchain_openai import ChatOpenAI

from config import LLM_BASE_URL, LLM_API_KEY
from core.llm_cost import get_cost_callback


class NoOpCompressor(BaseDocumentCompressor):
    """不做重排，原样返回。"""

    def compress_documents(self, documents: list[Document], query: str,
                           callbacks: Callbacks = None) -> list[Document]:
        return documents


class LLMRerankCompressor(BaseDocumentCompressor):
    """LLM 逐个打分后按分数降序排列。"""

    def __init__(self, model: str = "deepseek-chat"):
        super().__init__()
        self._llm = ChatOpenAI(
            model=model,
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            temperature=0.1,
            extra_body={"thinking": {"type": "disabled"}},
            callbacks=[get_cost_callback()],
        )

    def compress_documents(self, documents: list[Document], query: str,
                           callbacks: Callbacks = None) -> list[Document]:
        if not documents:
            return documents

        scored = []
        for doc in documents:
            try:
                resp = self._llm.invoke(
                    f"评估以下文本块与用户查询的相关性。\n"
                    f"用户查询：{query}\n\n文本块：\n{doc.page_content}\n\n"
                    f"给出 0-10 的相关性评分，只输出一个数字。"
                )
                score = float(re.search(r"[-+]?\d*\.?\d+", resp.content).group())
                score = min(max(score, 0), 10)
            except Exception:
                score = 5.0
            doc.metadata["rerank_score"] = score
            scored.append(doc)

        scored.sort(key=lambda x: x.metadata.get("rerank_score", 0), reverse=True)
        return scored


class BGERerankCompressor(BaseDocumentCompressor):
    """BGE Cross-Encoder 重排。需 pip install sentence-transformers。"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", top_n: int = 5):
        super().__init__()
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(model_name)
        self.top_n = top_n

    def compress_documents(self, documents: list[Document], query: str,
                           callbacks: Callbacks = None) -> list[Document]:
        if not documents:
            return documents
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self._model.predict(pairs)
        for doc, score in zip(documents, scores):
            doc.metadata["rerank_score"] = float(score)
        documents.sort(key=lambda x: x.metadata.get("rerank_score", 0), reverse=True)
        return documents[:self.top_n]
