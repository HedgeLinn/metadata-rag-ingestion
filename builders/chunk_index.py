"""块级索引构建 — 参数化切分器，关联 file_id + section_id。"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path

from langchain_core.documents import Document
from qdrant_client.models import PointStruct, SparseVector
from tqdm import tqdm

from config import (
    DATASETS_DIR, EMBED_BATCH, UPSERT_BATCH, META_WORKERS,
)
from core.embedding import get_embeddings
from core.qdrant import get_client, get_collection_name, ensure_collection
from core.sparse_encoder import BM25SparseEncoder
from core.registry import factory
from core.section_meta import (
    load_documents, detect_sections, _clean_markdown,
    _build_llm, generate_section_summary, generate_file_meta,
)
from core.llm_cost import require_budget


# ── 元数据生成（可选） ──

def _generate_metadata(file_contents: dict[str, str], kb_name: str,
                       chunker_name: str, show_progress: bool = True,
                       on_file_done: callable = None
                       ) -> dict[str, dict]:
    """为每个文件生成 LLM 元数据，带缓存 + 失败重试。
    返回 {file_id: {file_type, file_summary, section_summaries: {title: summary}}}。
    on_file_done(file_num, total) 在每个文件处理完后调用。
    """
    cache_path = DATASETS_DIR / kb_name / "_meta_cache.json"
    meta_cache = {}
    if cache_path.exists():
        try:
            meta_cache = json.loads(cache_path.read_text(encoding="utf-8"))
            if show_progress:
                tqdm.write(f"  [{chunker_name}] 加载元数据缓存: {len(meta_cache)} 条")
        except Exception:
            meta_cache = {}

    llm = _build_llm()
    result = {}
    cache_dirty = False

    file_list = sorted(file_contents.items())
    total = len(file_list)
    file_iter = tqdm(file_list, desc=f"  [{chunker_name}] LLM meta", unit="file",
                     leave=False) if show_progress else file_list

    # 收集失败的 section（需要重试）
    failed_sections: list[tuple[str, str, str]] = []  # (fid, title, text)
    failed_files: list[tuple[str, str]] = []  # (fid, summaries_text)

    for file_num, (fid, content) in enumerate(file_iter, 1):
        sections = detect_sections(content)
        if len(sections) <= 1:
            sections = [(0, len(content), "正文", 1)]

        valid_sections = [(s, e, t, l) for s, e, t, l in sections
                          if len(content[s:e].strip()) >= 50]

        # 生成 section_summary（带缓存，失败则跳过）
        section_summaries = {}
        cached_sections = [(s, e, t, l) for s, e, t, l in valid_sections
                           if f"{fid}|{t}" in meta_cache]
        for _, _, title, _ in cached_sections:
            section_summaries[title] = meta_cache[f"{fid}|{title}"]

        new_sections = [(s, e, t, l) for s, e, t, l in valid_sections
                        if f"{fid}|{t}" not in meta_cache]
        if new_sections:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            sec_data = [(content[s:e].strip(), title) for s, e, t, _ in new_sections]
            with ThreadPoolExecutor(max_workers=META_WORKERS) as pool:
                futures = {pool.submit(generate_section_summary, text, llm): (text, title)
                          for text, title in sec_data}
                sec_iter = tqdm(as_completed(futures), total=len(futures),
                               desc=f"  [{chunker_name}] sec_summary",
                               unit="sec", leave=False) if show_progress else as_completed(futures)
                for f in sec_iter:
                    sec_text, title = futures[f]
                    try:
                        summary = f.result()
                    except Exception:
                        summary = None
                    if summary:
                        section_summaries[title] = summary
                        meta_cache[f"{fid}|{title}"] = summary
                        cache_dirty = True
                    else:
                        failed_sections.append((fid, title, sec_text))

        # 生成 file_meta（带缓存）
        file_cache_key = f"__file__|{fid}"
        if file_cache_key in meta_cache:
            file_meta = meta_cache[file_cache_key]
        else:
            summaries_list = [section_summaries.get(t, "") for _, _, t, _ in valid_sections]
            file_meta = generate_file_meta(fid, summaries_list, llm)
            if file_meta:
                meta_cache[file_cache_key] = file_meta
                cache_dirty = True
            else:
                failed_files.append((fid, "\n".join(f"- {s}" for s in summaries_list if s)))

        if cache_dirty:
            cache_path.write_text(json.dumps(meta_cache, ensure_ascii=False, indent=2),
                                  encoding="utf-8")

        result[fid] = {
            "file_type": file_meta.get("file_type", "") if file_meta else "",
            "file_summary": file_meta.get("file_summary", "") if file_meta else "",
            "section_summaries": section_summaries,
        }

        if on_file_done:
            on_file_done(file_num, total)

    # ── 第二遍：重试失败项 ──
    if failed_sections:
        tqdm.write(f"  [{chunker_name}] 重试 {len(failed_sections)} 个失败 section...")
        retry_ok = 0
        for fid, title, sec_text in tqdm(failed_sections, desc=f"  [{chunker_name}] retry sec",
                                         unit="sec", leave=False):
            summary = generate_section_summary(sec_text, llm)
            if summary:
                result[fid]["section_summaries"][title] = summary
                meta_cache[f"{fid}|{title}"] = summary
                cache_dirty = True
                retry_ok += 1
            else:
                # 彻底失败，用原文截断
                result[fid]["section_summaries"][title] = sec_text[:80]
        tqdm.write(f"  [{chunker_name}] section 重试成功: {retry_ok}/{len(failed_sections)}")

    if failed_files:
        tqdm.write(f"  [{chunker_name}] 重试 {len(failed_files)} 个失败 file_meta...")
        retry_ok = 0
        for fid, summaries_text in tqdm(failed_files, desc=f"  [{chunker_name}] retry file",
                                        unit="file", leave=False):
            summaries_list = [s for s in summaries_text.split("\n- ") if s]
            file_meta = generate_file_meta(fid, summaries_list, llm)
            if file_meta:
                result[fid]["file_type"] = file_meta.get("file_type", "")
                result[fid]["file_summary"] = file_meta.get("file_summary", "")
                meta_cache[f"__file__|{fid}"] = file_meta
                cache_dirty = True
                retry_ok += 1
        tqdm.write(f"  [{chunker_name}] file_meta 重试成功: {retry_ok}/{len(failed_files)}")

    if cache_dirty:
        cache_path.write_text(json.dumps(meta_cache, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    return result


# ── 主构建函数 ──

def build_chunk_index(kb_name: str, chunker_name: str, force: bool = False,
                      show_progress: bool = True, with_meta: bool = False,
                      on_file_done: callable = None) -> int:
    """为指定切分器构建块级索引 {kb}__{chunker}。返回向量数。

    with_meta=True 时，使用 LLM 为每个文件/章节生成元数据并注入 payload。
    """
    t0 = time.time()
    collection_name = get_collection_name(kb_name, chunker_name, with_meta=with_meta)

    client = get_client()
    if client.collection_exists(collection_name):
        if force:
            client.delete_collection(collection_name)
        else:
            count = client.count(collection_name).count
            if count > 0:
                if show_progress:
                    tqdm.write(f"  [{chunker_name}] 跳过 ({count} vectors 已存在)")
                client.close()
                return count
            if show_progress:
                tqdm.write(f"  [{chunker_name}] collection 存在但为空，重新构建")
    client.close()

    # 加载文档
    all_docs = load_documents(kb_name)
    if not all_docs:
        return 0

    # 按 source 合并
    file_contents: dict[str, str] = {}
    for d in all_docs:
        fid = d.metadata.get("source", "unknown")
        if fid not in file_contents:
            file_contents[fid] = ""
        file_contents[fid] += d.page_content + "\n\n"

    # 获取切分器
    info = factory.chunker_info(chunker_name)
    splitter = factory.get_splitter(chunker_name,
                                    embeddings=get_embeddings() if info.get("requires_embeddings") else None)

    # 可选：生成 LLM 元数据
    metadata = {}
    if with_meta:
        require_budget(task=f"元数据构建 [{chunker_name}]")
        metadata = _generate_metadata(file_contents, kb_name, chunker_name,
                                      show_progress, on_file_done=on_file_done)

    # 按章节切分 + 应用切分器
    chunks = []

    file_list = sorted(file_contents.items())
    for fid, content in file_list:
        sections = detect_sections(content)
        if len(sections) <= 1:
            sections = [(0, len(content), "正文", 1)]

        meta = metadata.get(fid, {})
        section_summaries = meta.get("section_summaries", {})

        for start, end, title, level in sections:
            sec_text = content[start:end].strip()
            if len(sec_text) < 50:
                continue

            sec_doc = Document(page_content=sec_text, metadata={"source": fid})
            try:
                if info.get("type") == "custom" or hasattr(splitter, 'split_documents'):
                    sec_chunks = splitter.split_documents([sec_doc])
                else:
                    result = splitter.split_text(sec_text)
                    sec_chunks = []
                    for item in result:
                        if isinstance(item, Document):
                            item.metadata.update(sec_doc.metadata)
                            sec_chunks.append(item)
                        else:
                            sec_chunks.append(Document(
                                page_content=str(item),
                                metadata=sec_doc.metadata,
                            ))
            except Exception:
                try:
                    from langchain_text_splitters import RecursiveCharacterTextSplitter
                    pre_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=0)
                    sub_docs = pre_splitter.split_documents([sec_doc])
                    sec_chunks = []
                    for sub in sub_docs:
                        try:
                            sub_chunks = splitter.split_documents([sub])
                            sec_chunks.extend(sub_chunks)
                        except Exception:
                            sec_chunks.append(sub)
                except Exception:
                    sec_chunks = [sec_doc]

            for ch in sec_chunks:
                ch.metadata["source"] = fid
                if with_meta:
                    ch.metadata["file_name"] = fid
                    ch.metadata["file_type"] = meta.get("file_type", "")
                    ch.metadata["file_summary"] = meta.get("file_summary", "")
                    ch.metadata["section_summary"] = section_summaries.get(title, "")
                    ch.metadata["section_title"] = title
            chunks.extend(sec_chunks)

    if not chunks:
        tqdm.write(f"  [{chunker_name}] 0 chunks")
        return 0

    # header_aware 后处理
    if chunker_name.startswith("header_aware"):
        _HEADER_KEYS = ["h1", "h2", "h3", "h4", "h5", "h6"]
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        _sub_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=150,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
        )
        processed = []
        for ch in chunks:
            header_chain = []
            for key in _HEADER_KEYS:
                val = ch.metadata.get(key, "")
                if val:
                    header_chain.append(f"{'#' * int(key[1])} {val}")
            prefix = " > ".join(header_chain) if header_chain else ""
            if prefix:
                ch.page_content = prefix + "\n" + ch.page_content
            if len(ch.page_content) > 1500:
                sub_chunks = _sub_splitter.split_documents([ch])
                for sc in sub_chunks:
                    sc.metadata.update(ch.metadata)
                    if prefix:
                        sc.page_content = prefix + "\n" + sc.page_content
                processed.extend(sub_chunks)
            else:
                processed.append(ch)
        tqdm.write(f"  [{chunker_name}] header-aware: {len(chunks)} -> {len(processed)} after header inject + sub-split")
        chunks = processed

    # 元数据注入：拼入 page_content，参与 sparse + dense 向量化
    if with_meta:
        for ch in chunks:
            meta_parts = []
            ft = ch.metadata.get("file_type", "")
            if ft:
                meta_parts.append(f"文件类型：{ft}")
            st = ch.metadata.get("section_title", "")
            if st:
                meta_parts.append(f"章节：{st}")
            ss = ch.metadata.get("section_summary", "")
            if ss:
                meta_parts.append(f"章节摘要：{ss}")
            fs = ch.metadata.get("file_summary", "")
            if fs:
                meta_parts.append(f"文件摘要：{fs}")
            prefix = "；".join(meta_parts)
            if prefix:
                ch.page_content = f"{prefix}\n{ch.page_content}"

    # 稀疏向量
    texts_for_sparse = [ch.page_content for ch in chunks]
    sparse_encoder = BM25SparseEncoder()
    sparse_encoder.fit(texts_for_sparse)
    sparse_encoder.save(DATASETS_DIR / kb_name / f"_sparse_{chunker_name}{'_meta' if with_meta else ''}.json")
    if show_progress:
        tqdm.write(f"  [{chunker_name}] sparse vocab: {sparse_encoder.vocab_size} tokens")

    sparse_vecs = []
    for ch in chunks:
        indices, values = sparse_encoder.encode(ch.page_content)
        sparse_vecs.append((indices, values))

    # 创建集合
    ensure_collection(get_client(), collection_name)

    # 嵌入（切分器已控制块大小，无需截断）
    texts = [ch.page_content for ch in chunks]
    embeddings = get_embeddings()
    all_vectors = [None] * len(texts)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from config import BUILD_WORKERS

    def _embed_batch_with_retry(batch_texts, max_retries=30, wait_secs=60):
        import openai
        for attempt in range(1, max_retries + 1):
            try:
                return embeddings.embed_documents(batch_texts)
            except openai.RateLimitError:
                tqdm.write(f"  [{chunker_name}] 429 配额限制，等待 {wait_secs}s 后重试 ({attempt}/{max_retries})")
                time.sleep(wait_secs)
        raise RuntimeError(f"[{chunker_name}] embedding 429 重试 {max_retries} 次仍失败")

    with ThreadPoolExecutor(max_workers=BUILD_WORKERS) as pool:
        futures_map = {}
        for i in range(0, len(texts), EMBED_BATCH):
            batch_texts = texts[i:i + EMBED_BATCH]
            futures_map[pool.submit(_embed_batch_with_retry, batch_texts)] = i
        batch_iter = tqdm(as_completed(futures_map), total=len(futures_map),
                          desc=f"  [{chunker_name}] embed", unit="b", leave=False) if show_progress else as_completed(futures_map)
        for f in batch_iter:
            start = futures_map[f]
            for j, vec in enumerate(f.result()):
                all_vectors[start + j] = vec

    # Upsert
    client = get_client()
    for i in range(0, len(chunks), UPSERT_BATCH):
        batch_points = []
        for j in range(i, min(i + UPSERT_BATCH, len(chunks))):
            ch = chunks[j]
            vec = all_vectors[j]
            payload = {
                "page_content": ch.page_content,
                "content_text": ch.page_content,
                "source": ch.metadata.get("source", ""),
            }
            if with_meta:
                payload["file_name"] = ch.metadata.get("file_name", "")
                payload["file_type"] = ch.metadata.get("file_type", "")
                payload["file_summary"] = ch.metadata.get("file_summary", "")
                payload["section_summary"] = ch.metadata.get("section_summary", "")
                payload["section_title"] = ch.metadata.get("section_title", "")
            batch_points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "": vec,
                    "sparse": SparseVector(
                        indices=sparse_vecs[j][0],
                        values=sparse_vecs[j][1],
                    ),
                },
                payload=payload,
            ))
        client.upsert(collection_name=collection_name, points=batch_points, wait=True)
    client.close()

    elapsed = time.time() - t0
    meta_tag = " +meta" if with_meta else ""
    tqdm.write(f"  [{chunker_name}{meta_tag}] {len(chunks)} vectors ({elapsed/60:.0f}min)")
    return len(chunks)