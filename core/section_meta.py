"""章节检测与 LLM 元数据生成 — chunk_index 构建时使用。"""
import json
import re
import time
from pathlib import Path

import openai
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from tqdm import tqdm

from config import (
    DATASETS_DIR, CHUNK_LLM_BASE_URL, CHUNK_LLM_API_KEY, CHUNK_LLM_MODEL,
)
from core.llm_cost import get_cost_callback

# ── Markdown 噪声清洗 ──
_IMG_RE = re.compile(r'!\[.*?\]\(.*?\)')
_LINK_RE = re.compile(r'\[.*?\]\(.*?\)')


def _clean_markdown(text: str) -> str:
    text = _IMG_RE.sub('', text)
    text = _LINK_RE.sub('', text)
    return text


def load_documents(kb_name: str) -> list[Document]:
    doc_dir = DATASETS_DIR / kb_name
    files = list(doc_dir.glob("*.md"))
    files = [f for f in files if f.name != "index_table.csv"]
    if not files:
        print(f"[section_meta] 未在 {doc_dir} 中找到 .md 文件")
        return []

    all_docs = []
    for fp in tqdm(files, desc="加载文件", unit="file"):
        loader = TextLoader(str(fp), encoding="utf-8")
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = fp.stem
            d.page_content = _clean_markdown(d.page_content)
        all_docs.extend(docs)
    print(f"[section_meta] 加载 {len(files)} 个文件，{len(all_docs)} 个页面")
    return all_docs


def detect_sections(text: str) -> list[tuple[int, int, str, int]]:
    header_re = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    point_re = re.compile(r'^([一二三四五六七八九十]+)、(.+)$', re.MULTILINE)

    matches = []
    for m in header_re.finditer(text):
        matches.append((m.start(), len(m.group(1)), m.group(2).strip()))

    if not matches:
        for m in point_re.finditer(text):
            matches.append((m.start(), 1, m.group(0).strip()))

    if not matches:
        return [(0, len(text), "正文", 1)]

    result = []
    for i, (start, level, title) in enumerate(matches):
        end = matches[i + 1][0] if i + 1 < len(matches) else len(text)
        result.append((start, end, title, level))
    return result


# ── LLM 元数据生成 ──

MAX_SECTION_LEN = 3000

SECTION_SUMMARY_PROMPT = (
    "用一句话概括以下文本片段（20-40字），包含关键数字、条件：\n\n"
)

FILE_META_PROMPT = """你是一个文档分析专家。根据以下文档的章节摘要，生成文件级元数据。

- file_type: 判断文件类型，必须是以下之一：通知、细则、办法、指南、制度、规范
- file_summary: 2-3句话概述（100-150字），综合所有章节内容，说明文件性质、主要内容和适用范围

## 输出格式
严格输出 JSON，不要 markdown 代码块：
{"file_type": "办法", "file_summary": "..."}

## 章节摘要
"""


def _build_llm(model: str = None) -> ChatOpenAI:
    return ChatOpenAI(
        model=model or CHUNK_LLM_MODEL, base_url=CHUNK_LLM_BASE_URL,
        api_key=CHUNK_LLM_API_KEY, temperature=0.1,
        timeout=60, max_retries=1,
        extra_body={"thinking": {"type": "disabled"}},
        callbacks=[get_cost_callback()],
    )


# 模型回退链：主模型失败后依次尝试（较新模型，不含 deepseek-v3.2）
FALLBACK_MODELS = ["kimi-k2.5", "qwen3.7-max", "glm-5.1", "kimi-k2.6", "MiniMax-M2.5"]


def _invoke_with_fallback(llm: ChatOpenAI, prompt: str) -> str | None:
    """调用 LLM，失败时依次尝试回退模型。返回响应文本或 None。"""
    # 先尝试主模型
    try:
        resp = llm.invoke(prompt)
        return resp.content.strip()
    except Exception:
        pass

    # 依次尝试回退模型
    for model in FALLBACK_MODELS:
        try:
            fb_llm = _build_llm(model)
            resp = fb_llm.invoke(prompt)
            return resp.content.strip()
        except Exception:
            continue

    return None


def _extract_json(text: str) -> dict | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def generate_section_summary(text: str, llm: ChatOpenAI) -> str | None:
    """生成章节摘要，失败返回 None（由调用方稍后重试）。"""
    if len(text) > MAX_SECTION_LEN:
        sub_texts = [text[i:i + 2000] for i in range(0, len(text), 2000)]
        mini = []
        for sub in sub_texts:
            result = _invoke_with_fallback(llm, SECTION_SUMMARY_PROMPT + sub)
            if result:
                mini.append(result)
            else:
                return None
        text = " ".join(mini)[:MAX_SECTION_LEN]

    return _invoke_with_fallback(llm, SECTION_SUMMARY_PROMPT + text[:MAX_SECTION_LEN])


def generate_file_meta(file_name: str, section_summaries: list[str],
                       llm: ChatOpenAI) -> dict | None:
    """生成文件元数据，失败返回 None（由调用方稍后重试）。"""
    if not section_summaries:
        return {"file_type": "未知", "file_summary": ""}

    valid_summaries = [s for s in section_summaries if s]
    if not valid_summaries:
        return {"file_type": "未知", "file_summary": ""}

    summaries_text = "\n".join(f"- {s}" for s in valid_summaries)
    prompt = FILE_META_PROMPT + f"\n文件: {file_name}\n\n{summaries_text}"

    resp_text = _invoke_with_fallback(llm, prompt)
    if resp_text:
        result = _extract_json(resp_text)
        if result:
            result.setdefault("file_type", "未知")
            result.setdefault("file_summary", "")
            return result

    return None