"""扁平检索：直接在 chunk 集合上搜索，无过滤。"""
from search.base import BaseStrategy
from core.qdrant import hybrid_search, _dense_vector_search, _sparse_vector_search


class FlatStrategy(BaseStrategy):
    name = "flat"
    description = "直接块搜索：在 chunk 集合上 dense+sparse hybrid 检索，无路由过滤"

    def search(self, query: str, kb_name: str, top_k: int = 5,
               chunker_name: str = "fixed_1100",
               retriever: str = "hybrid") -> list[dict]:
        if retriever == "vector":
            return _dense_vector_search(query, f"{kb_name}__{chunker_name}", top_k)
        elif retriever == "bm25":
            return _sparse_vector_search(query, f"{kb_name}__{chunker_name}", kb_name, chunker_name, top_k)
        else:
            return hybrid_search(query, kb_name, chunker_name, top_k)