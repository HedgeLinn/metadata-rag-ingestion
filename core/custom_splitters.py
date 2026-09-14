"""Custom text splitters — LLMIndexSplitter and ParentChildSplitter.

LangChain 不提供这两种切分模式，作为 TextSplitter 子类实现。
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from tqdm import tqdm

from config import CHUNK_LLM_BASE_URL, CHUNK_LLM_API_KEY, CHUNK_LLM_MODEL
from core.llm_cost import get_cost_callback, require_budget

LLM_INDEX_PROMPT = (
    "你是一个检索系统优化助手。针对以下知识片段，生成3个不同的搜索索引文本，"
    "帮助用户从不同角度检索到这段内容。只输出3行，不要编号、不要解释。\n\n"
    "知识片段：\n{content}\n\n"
    "3个索引文本："
)


class LLMIndexSplitter(TextSplitter):
    """段落切分后 LLM 为每段生成 3 条检索索引文本。

    split_documents() 返回的每个 Document:
      - page_content = index_text（用于 embedding）
      - metadata["content_text"] = 原始段落（检索返回时用）
    """

    def __init__(self):
        super().__init__()
        self._llm = ChatOpenAI(
            model=CHUNK_LLM_MODEL,
            base_url=CHUNK_LLM_BASE_URL,
            api_key=CHUNK_LLM_API_KEY,
            temperature=0.3,
            extra_body={"thinking": {"type": "disabled"}},
            callbacks=[get_cost_callback()],
        )

    def split_text(self, text: str, **kwargs) -> list[str]:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        return paragraphs

    def split_documents(self, documents: list[Document]) -> list[Document]:
        require_budget(task="llm_index 索引生成")
        # 收集所有段落
        tasks = []  # (doc_metadata, paragraph_index, paragraph_text)
        for doc in documents:
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", doc.page_content) if p.strip()]
            for pi, para in enumerate(paragraphs):
                tasks.append((doc.metadata, pi, para))

        # 并发 20 线程调 LLM 生成索引
        results = [None] * len(tasks)
        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = {pool.submit(_gen_indexes, para, self._llm): idx
                       for idx, (_, _, para) in enumerate(tasks)}
            for f in tqdm(as_completed(futures), total=len(futures),
                          desc="  [llm_index] LLM", unit="para", leave=False):
                idx = futures[f]
                results[idx] = f.result()

        # 构建 Documents（每段: 原文 + N 条索引）
        result = []
        for (meta, pi, para), indexes in zip(tasks, results):
            result.append(Document(
                page_content=para,
                metadata={**meta, "content_text": para, "paragraph_index": pi},
            ))
            for idx_text in (indexes or [para]):
                result.append(Document(
                    page_content=idx_text,
                    metadata={**meta, "content_text": para, "paragraph_index": pi},
                ))
        return result


def _gen_indexes(para: str, llm) -> list[str]:
    try:
        resp = llm.invoke(LLM_INDEX_PROMPT.format(content=para))
        indexes = [l.strip() for l in resp.content.split("\n")
                   if l.strip() and len(l.strip()) >= 3]
    except Exception:
        indexes = []
    return indexes or [para]


class ParentChildSplitter(TextSplitter):
    """父子分块：父块固定大小切分，子块按句子切分。

    embed 子块，检索时返回父块内容。
    """

    def __init__(self, parent_chunk_size: int = 1100, child_chunk_size: int = 150,
                 parent_overlap: int = 0):
        super().__init__()
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size, chunk_overlap=parent_overlap,
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size, chunk_overlap=0,
            separators=["。", "！", "？", "\n", " ", ""],
        )

    def split_text(self, text: str, **kwargs) -> list[str]:
        return self.child_splitter.split_text(text)

    def split_documents(self, documents: list[Document]) -> list[Document]:
        result = []
        for doc in documents:
            parent_chunks = self.parent_splitter.split_documents([doc])
            for pi, parent in enumerate(parent_chunks):
                children = self.child_splitter.split_documents([parent])
                for child in children:
                    child.metadata["content_text"] = parent.page_content
                    child.metadata["parent_index"] = pi
                    result.append(child)
        return result
