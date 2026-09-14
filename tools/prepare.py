"""Test set enrichment — LLM auto-extracts scoring criteria from QA pairs.

Input xlsx needs at least:
  - 问题 / question 列
  - 正确回答 / 标准答案 列

Output xlsx adds:
  - point:     评分要点（编号列表，分号分隔）
  - 必答点:     核心必答要点编号
  - 选答点:     加分选答要点编号
  - 参考文件:    引用依据要点编号
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pandas as pd
from langchain_openai import ChatOpenAI

from config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
from core.llm_cost import get_cost_callback, require_budget

# ── Prompts ──

SYSTEM_PROMPT = """你是一个文档测试集构建助手。你的任务是根据给定的问题和标准答案，提取评分要点并分类。

## 任务

对每个问题，完成以下工作：

### 1. 提取评分要点（point）
- 从"标准答案"中逐条提取关键得分点
- **必须精准保留原文数据**：数字、费率、公式、百分比、阈值、参考依据等一律不得修改或省略
- 每条要点用简洁的陈述句表达，但数据一个字都不能改
- 内容要点排前面，引用依据排在最后

### 2. 分类要点
- **必答点**：回答该问题时必须包含的核心要点编号（只标内容要点，不标引用依据）
- **选答点**：回答正确可加分但不是必须的要点编号（只标内容要点，不标引用依据）
- **参考文件索引**：那些引用依据类要点的编号（即排在最后面的引用条目）

### 3. 输出格式
严格输出 JSON，不要有任何额外文字：

```json
{
  "points": ["1. 第一条要点", "2. 第二条要点", "3. 参考依据：《xxx》"],
  "required": [1],
  "optional": [2],
  "ref_idx": [3]
}
```

## 重要规则
- points 是连续编号的完整列表，编号从 1 开始
- 内容要点在前，引用依据要点在最后
- required + optional 覆盖所有内容要点，不重叠
- ref_idx 必须指向 points 列表中确实引用文件/标准的那几条
- ref_idx 中的编号必须在 points 中真实存在且是引用类型的条目
- 如果标准答案中没有任何明确的参考文件引用，ref_idx 为空数组 []
"""

USER_MSG_TEMPLATE = """## 问题
{question}

## 标准答案
{answer}

## 涉及的文件/标准
{references}

请提取评分要点并分类，输出 JSON。"""

# ── Column detection ──

REQUIRED_SCORING_COLS = ["point", "必答点", "选答点", "参考文件"]

COLUMN_PATTERNS = {
    "question": ["问题", "question", "query"],
    "qid": ["问题ID", "问题id", "id", "编号", "序号"],
    "answer": ["正确回答", "标准答案", "answer", "参考答案", "答案"],
    "ref_docs": ["文件来源", "文件溯源", "参考文件", "涉及计价标准", "涉及标准", "reference", "ref"],
}


def detect_columns(df: pd.DataFrame) -> dict[str, str | None]:
    """Map canonical column names to actual column names in dataframe."""
    cols = [str(c).strip() for c in df.columns]
    mapping: dict[str, str | None] = {k: None for k in COLUMN_PATTERNS}

    for c in cols:
        for canonical, patterns in COLUMN_PATTERNS.items():
            if c in patterns:
                mapping[canonical] = c
                break

    if not mapping["question"]:
        for c in cols:
            if "问题" in c or "question" in c.lower():
                mapping["question"] = c; break
    if not mapping["answer"]:
        for c in cols:
            if "回答" in c or "答案" in c or "answer" in c.lower():
                mapping["answer"] = c; break
    if not mapping["ref_docs"]:
        for c in cols:
            if "文件" in c or "来源" in c or "溯源" in c:
                mapping["ref_docs"] = c; break
    if not mapping["ref_docs"]:
        for c in cols:
            if "计价" in c or "标准" in c or "ref" in c.lower() or "参考" in c:
                mapping["ref_docs"] = c; break

    return mapping


def detect_missing_columns(xlsx_path: str | Path) -> list[str]:
    """Return list of scoring column names missing from the test set."""
    df = pd.read_excel(xlsx_path)
    cols = [str(c).strip() for c in df.columns]
    return [c for c in REQUIRED_SCORING_COLS if c not in cols]


# ── LLM extraction ──

def _extract_json(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1:
        raw = raw[start:end + 1]
    return raw


def _validate_result(result: dict):
    if not isinstance(result, dict):
        raise ValueError("输出不是 JSON 对象")
    for key in ["points", "required", "optional", "ref_idx"]:
        if key not in result:
            raise ValueError(f"缺少字段: {key}")
    points = result["points"]
    if not isinstance(points, list) or len(points) == 0:
        raise ValueError("points 不能为空")
    n = len(points)
    for i, p in enumerate(points):
        expected = f"{i + 1}."
        if not p.strip().startswith(expected):
            result["points"][i] = f"{i + 1}. {p.strip()}"
    for key in ["required", "optional", "ref_idx"]:
        vals = result.get(key, [])
        if not isinstance(vals, list):
            result[key] = []
            vals = []
        result[key] = [int(v) for v in vals if isinstance(v, (int, float)) and 1 <= v <= n]
    req_set = set(result["required"])
    opt_set = set(result["optional"])
    overlap = req_set & opt_set
    if overlap:
        result["optional"] = [x for x in result["optional"] if x not in overlap]


def extract_points(llm: ChatOpenAI, question: str, answer: str,
                   references: str = "", max_retries: int = 3) -> dict:
    """Call LLM to extract scoring criteria from a single QA pair."""
    user_msg = USER_MSG_TEMPLATE.format(
        question=question,
        answer=answer[:8000],
        references=references if references else "（无）",
    )
    for attempt in range(max_retries):
        try:
            resp = llm.invoke([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ])
            json_str = _extract_json(resp.content)
            result = json.loads(json_str)
            _validate_result(result)
            return result
        except (json.JSONDecodeError, ValueError) as e:
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))
            else:
                raise
    return {"points": [], "required": [], "optional": [], "ref_idx": []}


# ── Main pipeline ──

def enrich_test_set(
    input_path: str | Path,
    output_path: str | Path | None = None,
    model_name: str | None = None,
    delay: float = 0.3,
) -> Path:
    """Auto-detect missing scoring columns and generate them via LLM.

    Args:
        input_path: xlsx with 问题 + 正确回答 columns
        output_path: output xlsx path (default: *_enhanced.xlsx)
        model_name: LLM model (default: from config)
        delay: seconds between LLM calls

    Returns:
        Path to enriched xlsx.
    """
    require_budget(task="评分要点提取")
    input_path = Path(input_path)
    df = pd.read_excel(input_path)
    col_map = detect_columns(df)

    q_col = col_map["question"]
    a_col = col_map["answer"]
    ref_col = col_map["ref_docs"]

    if not q_col:
        raise ValueError("未检测到问题列（问题/question）")
    if not a_col:
        raise ValueError("未检测到答案列（正确回答/标准答案）")

    llm = ChatOpenAI(
        model=model_name or LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=0.05,
        extra_body={"thinking": {"type": "disabled"}},
        callbacks=[get_cost_callback()],
    )

    rows = []
    total = len(df)
    print(f"测试集: {total} 题")
    print(f"问题列: '{q_col}'  答案列: '{a_col}'  参考列: '{ref_col or '(未检测到)'}'")
    print(f"模型: {model_name or LLM_MODEL}")
    print()

    for i in range(total):
        question = str(df.iloc[i][q_col]) if not pd.isna(df.iloc[i][q_col]) else ""
        answer = str(df.iloc[i][a_col]) if not pd.isna(df.iloc[i][a_col]) else ""
        ref_docs = str(df.iloc[i][ref_col]) if ref_col and not pd.isna(df.iloc[i][ref_col]) else ""

        print(f"[{i + 1}/{total}] {question[:60]}...")

        if not answer.strip():
            rows.append({"问题": question, "正确回答": answer,
                         "point": "", "必答点": "", "选答点": "", "参考文件": ""})
            print("  skip (无答案)")
            continue

        try:
            result = extract_points(llm, question, answer, ref_docs)
        except Exception as e:
            print(f"  [ERROR] {e}")
            result = {"points": [], "required": [], "optional": [], "ref_idx": []}

        points = result.get("points", [])
        ref_idx = result.get("ref_idx", [])
        if ref_docs.strip():
            points.append(f"{len(points) + 1}. 参考依据：{ref_docs.strip()}")
            ref_idx.append(len(points))

        rows.append({
            "问题": question,
            "正确回答": answer,
            "point": "; ".join(points),
            "必答点": ",".join(str(r) for r in result.get("required", [])),
            "选答点": ",".join(str(o) for o in result.get("optional", [])),
            "参考文件": ",".join(str(r) for r in ref_idx),
        })

        print(f"  points={len(result.get('points',[]))}  required={result.get('required',[])}  optional={result.get('optional',[])}  ref={result.get('ref_idx',[])}")

        if i < total - 1:
            time.sleep(delay)

    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_enhanced{input_path.suffix}"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_excel(str(output_path), index=False)
    print(f"\n增强测试集已保存: {output_path}")
    return output_path