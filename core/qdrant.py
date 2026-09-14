"""Qdrant 客户端封装 — dense / sparse / hybrid 检索 + 集合管理。"""
from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams, SparseVector,
    PointStruct, OptimizersConfigDiff,
)

from config import QDRANT_URL, COLLECTION_SEP, RRF_K
from core.sparse_encoder import BM25SparseEncoder
from core.embedding import get_embeddings, get_embed_config


def get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, timeout=300, trust_env=False)


def get_collection_name(kb: str, suffix: str, with_meta: bool = False) -> str:
    name = f"{kb}{COLLECTION_SEP}{suffix}"
    if with_meta:
        name += "_meta"
    return name


def ensure_collection(client: QdrantClient, name: str, force: bool = False):
    if client.collection_exists(name):
        if force:
            client.delete_collection(name)
        else:
            return
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(
            size=get_embed_config()["dim"], distance=Distance.COSINE, on_disk=True,
        ),
        sparse_vectors_config={"sparse": SparseVectorParams()},
        on_disk_payload=True,
        optimizers_config=OptimizersConfigDiff(
            memmap_threshold=1048576,  # >1MB 的段用 mmap，OS 按需换页（单位：字节）
        ),
    )


def _dense_vector_search(
    query: str, collection_name: str, top_k: int = 5,
) -> list[dict]:
    """纯稠密向量检索。"""
    embeddings = get_embeddings()
    client = get_client()

    if not client.collection_exists(collection_name):
        client.close()
        return []

    query_vector = embeddings.embed_query(query)

    try:
        results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
        )
    except Exception:
        client.close()
        return []

    output = _parse_results(results.points)
    client.close()
    return output


def _sparse_vector_search(
    query: str, collection_name: str, kb_name: str, suffix: str,
    top_k: int = 5,
) -> list[dict]:
    """纯稀疏向量检索。"""
    from config import DATASETS_DIR

    vocab_path = DATASETS_DIR / kb_name / f"_sparse_{suffix}.json"
    if not vocab_path.exists():
        return []

    encoder = BM25SparseEncoder.load(vocab_path)
    indices, values = encoder.encode_query(query)
    if not indices:
        return []

    client = get_client()
    if not client.collection_exists(collection_name):
        client.close()
        return []

    try:
        results = client.query_points(
            collection_name=collection_name,
            query=SparseVector(indices=indices, values=values),
            using="sparse",
            limit=top_k,
        )
    except Exception:
        client.close()
        return []

    output = _parse_results(results.points)
    client.close()
    return output


def hybrid_search(
    query: str, kb_name: str, suffix: str, top_k: int = 5,
) -> list[dict]:
    """Qdrant 原生 dense + sparse RRF 混合检索。词表缺失时降级为纯稠密。"""
    from config import DATASETS_DIR

    collection_name = get_collection_name(kb_name, suffix)
    client = get_client()

    if not client.collection_exists(collection_name):
        client.close()
        return []

    # 尝试加载稀疏编码器
    vocab_path = DATASETS_DIR / kb_name / f"_sparse_{suffix}.json"
    sparse_available = vocab_path.exists()

    if not sparse_available:
        client.close()
        return _dense_vector_search(query, collection_name, top_k)

    encoder = BM25SparseEncoder.load(vocab_path)
    sparse_indices, sparse_values = encoder.encode_query(query)
    if not sparse_indices:
        client.close()
        return _dense_vector_search(query, collection_name, top_k)

    embeddings = get_embeddings()
    dense_vec = embeddings.embed_query(query)

    try:
        results = client.query_points(
            collection_name=collection_name,
            prefetch=[
                __import__("qdrant_client").models.Prefetch(
                    query=dense_vec, using="", limit=top_k * 2,
                ),
                __import__("qdrant_client").models.Prefetch(
                    query=SparseVector(indices=sparse_indices, values=sparse_values),
                    using="sparse", limit=top_k * 2,
                ),
            ],
            query=__import__("qdrant_client").models.FusionQuery(
                fusion=__import__("qdrant_client").models.Fusion.RRF
            ),
            limit=top_k,
        )
    except Exception:
        client.close()
        return _dense_vector_search(query, collection_name, top_k)

    output = _parse_results(results.points)
    client.close()
    return output


def _parse_results(points: list) -> list[dict]:
    """将 Qdrant 查询结果统一解析为 dict 列表。"""
    results = []
    for point in points:
        payload = point.payload or {}
        results.append({
            "id": point.id,
            "score": point.score,
            "content_text": str(payload.get("content_text", payload.get("page_content", ""))),
            "index_text": str(payload.get("page_content", payload.get("index_text", ""))),
            "source": str(payload.get("source", payload.get("file_id", ""))),
            "file_id": str(payload.get("file_id", "")),
            "section_id": str(payload.get("section_id", "")),
            "chunk_id": str(payload.get("chunk_id", "")),
            # 元数据字段（with_meta=True 构建时写入）
            "file_name": str(payload.get("file_name", "")),
            "file_type": str(payload.get("file_type", "")),
            "file_summary": str(payload.get("file_summary", "")),
            "section_summary": str(payload.get("section_summary", "")),
            "section_title": str(payload.get("section_title", "")),
        })
    return results


def scroll_all(collection_name: str) -> list[dict]:
    """全量 Scroll 一个 collection 的所有点。"""
    client = get_client()
    if not client.collection_exists(collection_name):
        client.close()
        return []

    points = []
    offset = None
    while True:
        page = client.scroll(
            collection_name=collection_name, limit=1000, offset=offset,
            with_payload=True, with_vectors=False,
        )
        points.extend(page[0])
        offset = page[1]
        if offset is None:
            break
    client.close()
    return _parse_results(points)


def count_vectors(collection_name: str) -> int:
    client = get_client()
    if not client.collection_exists(collection_name):
        client.close()
        return 0
    count = client.count(collection_name).count
    client.close()
    return count