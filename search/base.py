"""检索策略抽象基类。"""
from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    name: str = ""
    description: str = ""

    @abstractmethod
    def search(self, query: str, kb_name: str, top_k: int = 5,
               chunker_name: str = "fixed_1100",
               retriever: str = "hybrid") -> list[dict]:
        """执行检索，返回 list[dict] 统一格式。"""
        ...