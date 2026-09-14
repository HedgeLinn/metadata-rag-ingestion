"""入库编排器：所有切分器并行构建。"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from config import BUILD_WORKERS, DEFAULT_KB, COLLECTION_SEP
from core.registry import factory
from core.qdrant import get_client
from builders.chunk_index import build_chunk_index


def build_all(kb_name: str = DEFAULT_KB, chunker_name: str | None = None,
              force: bool = False, parallel: int = BUILD_WORKERS, pure: bool = False):
    """完整入库：所有切分器并行构建 chunk 索引。pure=True 时全部跳过元数据生成。"""
    t_start = time.time()

    chunker_names = [chunker_name] if chunker_name else factory.list_chunkers()
    if pure:
        _exclude = {"llm_index", "parent_child_v1", "parent_child_v2", "fixed_1100_ovl150_pure"}
        chunker_names = [cn for cn in chunker_names if cn not in _exclude]
    pure_tag = " [pure]" if pure else ""
    print(f"\n构建块索引 ({len(chunker_names)} 个切分器){pure_tag}")
    print(f"KB: {kb_name}  |  Chunkers: {len(chunker_names)}  |  Parallel: {parallel}")
    if parallel <= 1:
        for cn in chunker_names:
            with_meta = (not pure) and (not cn.endswith("_pure"))
            build_chunk_index(kb_name, cn, force=force, with_meta=with_meta)
    else:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {}
            for cn in chunker_names:
                with_meta = (not pure) and (not cn.endswith("_pure"))
                f = pool.submit(build_chunk_index, kb_name, cn, force, False, with_meta)
                futures[f] = cn
            with tqdm(total=len(futures), desc="Overall progress", unit="strat") as pbar:
                for f in as_completed(futures):
                    cn = futures[f]
                    try:
                        result = f.result()
                        tqdm.write(f"  [{cn}] {result} vectors -> done")
                    except Exception as e:
                        tqdm.write(f"  [{cn}] ERROR: {e}")
                    pbar.update(1)

    elapsed = time.time() - t_start
    print(f"\n{'─' * 50}")
    print(f"构建完成，耗时: {elapsed/60:.1f} 分钟")
    client = get_client()
    for c in client.get_collections().collections:
        if c.name.startswith(f"{kb_name}{COLLECTION_SEP}"):
            print(f"  {c.name}: {client.count(c.name).count} vectors")
    client.close()