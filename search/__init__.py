"""检索策略注册表。"""
from search.flat import FlatStrategy

STRATEGIES = {
    "flat": FlatStrategy(),
}


def get_strategy(name: str):
    if name not in STRATEGIES:
        raise ValueError(f"未知策略: {name}，可用: {list(STRATEGIES.keys())}")
    return STRATEGIES[name]


def list_strategies() -> list[str]:
    return list(STRATEGIES.keys())


def search(query: str, kb_name: str, chunker_name: str,
           strategy_name: str = "flat", top_k: int = 5,
           retriever: str = "hybrid") -> list[dict]:
    """统一检索入口。"""
    s = get_strategy(strategy_name)
    return s.search(query, kb_name, top_k=top_k, chunker_name=chunker_name,
                    retriever=retriever)