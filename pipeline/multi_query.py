"""多查询扩展召回实验（Multi-Query Recall）。

对测试集每个问题，用 LLM 生成 3 个"类似问题"，4 条查询（原问题 + 3 生成）
各自走 hybrid 检索后做 RRF 融合，取最终 topN 再算召回四口径。

对照组：同 3 个 chunker、同 topN，仅原问题单查询（不生成 3 问）。
输出：output/{kb}/{exp_name}/top{topk}/ 下 recall_MQ.xlsx + recall_direct.xlsx
（各自含 综合召回 sheet 与 report html），外加 cross_对比.xlsx 十字对照。
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from langchain_openai import ChatOpenAI
from openpyxl import load_workbook, Workbook
from openpyxl.styles import Font

from config import (
    CHUNK_LLM_API_KEY, CHUNK_LLM_BASE_URL, CHUNK_LLM_MODEL,
    DEFAULT_KB, SEARCH_WORKERS, RRF_K,
)
from core.llm_cost import get_cost_callback, require_budget
from search import search as strategy_search
from pipeline.recall import calc_recall, calc_recall_merged, calc_recall_concat, calc_file_recall

# ── 相似问题生成提示词 ──

SIM_QUESTION_PROMPT = (
    "你是政策解读领域的检索优化助手。给定一个用户问题，请从不同角度改写生成 3 个"
    "「相似问题」（换措辞 / 换入口叫法 / 补充隐含条件），用于同义召回。要求：\n"
    "1. 只输出 3 行，每行一个问题，不要编号、不要序号、不要解释。\n"
    "2. 与原文是同义/近义关系，不能改变问题意图，不能新增题目没有的限定。\n"
    "3. 覆盖不同提问角度：例如一种直问、一种换个说法、一种带上会出现在资料里的专有名词。\n\n"
    "原问题：{question}\n\n"
    "3 个相似问题："
)


def _build_llm(model: str | None = None) -> ChatOpenAI:
    """生成相似问题用的 LLM（默认 deepseek-v4-flash，入库同款）。"""
    return ChatOpenAI(
        model=model or CHUNK_LLM_MODEL,
        base_url=CHUNK_LLM_BASE_URL,
        api_key=CHUNK_LLM_API_KEY,
        temperature=0.3,
        timeout=60,
        max_retries=1,
        extra_body={"thinking": {"type": "disabled"}},
        callbacks=[get_cost_callback()],
    )


def generate_similar_questions(llm: ChatOpenAI, question: str, n: int = 3) -> list[str]:
    """生成 n 个相似问题；失败或产出不足时返回已有部分（可为空，上层回退单查询）。"""
    try:
        resp = llm.invoke(SIM_QUESTION_PROMPT.format(question=question))
        lines = [l.strip() for l in resp.content.split("\n")
                 if l.strip() and len(l.strip()) >= 4]
        return lines[:n]
    except Exception:
        return []


# ── RRF 融合 ──

def rrf_fuse(list_of_results: list[list[dict]], k: int = RRF_K,
             top_n: int | None = None) -> list[dict]:
    """对多路检索结果做 RRF 融合（等权重）。同一 chunk 按原始 score 取高者。

    list_of_results: 每路一个 list[dict]（search() 返回的统一 dict，用 content_text 判重）。
    返回按融合分降序的 list[dict]，长度 ≤ top_n（None 时全量），并给每条标注 fusion_score。
    """
    scores: dict[str, float] = {}
    best: dict[str, dict] = {}
    for docs in list_of_results:
        for rank, d in enumerate(docs, start=1):
            key = d.get("content_text", "") or str(d.get("id", ""))
            if not key:
                continue
            s = 1.0 / (k + rank)
            if key in scores:
                scores[key] += s
                if (d.get("score") or 0) > (best[key].get("score") or 0):
                    best[key] = d
            else:
                scores[key] = s
                best[key] = d

    merged = sorted(best.items(), key=lambda kv: scores[kv[0]], reverse=True)
    result = [d for _, d in merged]
    if top_n is not None:
        result = result[:top_n]
    for d in result:
        d["fusion_score"] = round(scores[d.get("content_text", "") or str(d.get("id", ""))], 4)
    return result


def multi_query_retrieve(question: str, kb_name: str, chunker_name: str,
                         similar: list[str], top_n: int,
                         retriever: str = "hybrid") -> list[dict]:
    """原问题 + N 个相似问题各自 hybrid 检索，RRF 融合取 top_n。"""
    queries = [question] + similar
    routes = [
        strategy_search(query=q, kb_name=kb_name, chunker_name=chunker_name,
                        top_k=top_n, retriever=retriever)
        for q in queries
    ]
    return rrf_fuse(routes, top_n=top_n)


def direct_retrieve(question: str, kb_name: str, chunker_name: str,
                    top_n: int, retriever: str = "hybrid") -> list[dict]:
    """对照：仅原问题单查询。"""
    return strategy_search(query=question, kb_name=kb_name, chunker_name=chunker_name,
                           top_k=top_n, retriever=retriever)


# ── 测试集加载（与 experiment.py 同逻辑） ──

def load_test_set(path: str) -> list[dict]:
    wb = load_workbook(path)
    ws = wb.active
    rows = []
    for r in range(2, ws.max_row + 1):
        q = str(ws.cell(row=r, column=1).value or "").strip()
        if not q:
            continue
        item = {"id": f"Q{r-1}", "question": q}
        for ci in range(2, ws.max_column + 1):
            hdr = str(ws.cell(row=1, column=ci).value or "").strip()
            val = ws.cell(row=r, column=ci).value
            if hdr and val:
                item[hdr] = str(val)
        for src, dst in {"参考答案": "正确回答", "文件来源": "参考文件"}.items():
            if src in item and dst not in item:
                item[dst] = item[src]
        rows.append(item)
    return rows


# ── 评测执行 ──

def _evaluate_one_group(questions, kb, chunks, sim_map, topn, retriever, group,
                        tk_dir, workers=SEARCH_WORKERS):
    """对一组 chunkers × 一个 topn × 一个 group（MQ/direct）跑召回，落盘 recall_{group}.xlsx。

    返回 {chunker: (rate, merged_rate, concat_rate, file_rate, avg_pos_display)}。
    """
    wb = Workbook()
    ws = wb.active
    ws.title = "汇总"
    headers = ["问题ID", "问题", "正确回答", "参考文件"]
    for cn in chunks:
        pfx = f"{cn}_{group}"
        headers += [f"{pfx}_召回", f"{pfx}_召回位置", f"{pfx}_合并召回",
                    f"{pfx}_拼接召回", f"{pfx}_文件召回"]
    for ci, h in enumerate(headers, start=1):
        ws.cell(row=1, column=ci, value=h).font = Font(bold=True)

    stats = {cn: {"hits": 0, "merged": 0, "concat": 0, "file": 0, "pos": []} for cn in chunks}
    total = len(questions)

    for ci, cn in enumerate(chunks):
        results = [None] * total
        work = [(qi, it["question"], sim_map[qi] if group == "MQ" else [])
                for qi, it in enumerate(questions)]

        def _one(qi_q_sims):
            qi, question, sims = qi_q_sims
            if group == "MQ":
                return qi, multi_query_retrieve(question, kb, cn, sims, topn, retriever)
            return qi, direct_retrieve(question, kb, cn, topn, retriever)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_one, w): w for w in work}
            for f in as_completed(futures):
                qi, docs = f.result()
                results[qi] = docs

        for qi, it in enumerate(questions):
            correct = str(it.get("正确回答", ""))
            ref = str(it.get("参考文件", ""))
            docs = results[qi] or []
            recall, rpos = calc_recall(correct, docs)
            recall_m, _ = calc_recall_merged(correct, docs)
            recall_c, _ = calc_recall_concat(correct, docs)
            file_r, _ = calc_file_recall(ref, docs, topn)

            st = stats[cn]
            st["hits"] += 1 if recall else 0
            st["merged"] += 1 if recall_m else 0
            st["concat"] += 1 if recall_c else 0
            st["file"] += 1 if file_r else 0
            if recall:
                st["pos"].append(rpos)

            col = 5 + ci * 5
            row = qi + 2
            ws.cell(row=row, column=col, value=recall)
            ws.cell(row=row, column=col + 1, value=f"top{rpos}" if recall else 0)
            ws.cell(row=row, column=col + 2, value=recall_m)
            ws.cell(row=row, column=col + 3, value=recall_c)
            ws.cell(row=row, column=col + 4, value=file_r)
        print(f"  [{group}] {cn} top{topn} done")

    ws_score = wb.create_sheet("综合召回")
    for ci, h in enumerate(["策略", "召回率(%)", "合并召回率(%)", "拼接召回率(%)",
                            "平均命中位置", "文件召回率(%)"], start=1):
        ws_score.cell(row=1, column=ci, value=h).font = Font(bold=True)
    rates_map = {}
    for ri, cn in enumerate(chunks, start=2):
        st = stats[cn]
        rate = round(st["hits"] / total * 100, 1) if total else 0
        mrate = round(st["merged"] / total * 100, 1) if total else 0
        crate = round(st["concat"] / total * 100, 1) if total else 0
        frate = round(st["file"] / total * 100, 1) if total else 0
        avgpos = round(sum(st["pos"]) / len(st["pos"]), 2) if st["pos"] else 0
        label = f"{cn}_{group}"
        ws_score.cell(row=ri, column=1, value=label)
        ws_score.cell(row=ri, column=2, value=rate)
        ws_score.cell(row=ri, column=3, value=mrate)
        ws_score.cell(row=ri, column=4, value=crate)
        ws_score.cell(row=ri, column=5, value=f"top{avgpos}" if avgpos else "--")
        ws_score.cell(row=ri, column=6, value=frate)
        rates_map[cn] = (rate, mrate, crate, frate, f"top{avgpos}" if avgpos else "--")

    fn = tk_dir / f"recall_{group}.xlsx"
    wb.save(fn)
    return rates_map, fn


def _write_cross_comparison(exp_dir, topn, chunks, per_topn):
    """写 top{topn}/cross_对比.xlsx：每个 chunker MQ-vs-direct 十字对照。"""
    wb = Workbook()
    ws = wb.active
    ws.title = "MQ vs Direct"
    headers = ["chunker", "topk",
               "MQ_召回", "MQ_合并", "MQ_拼接", "MQ_文件", "MQ_位置",
               "direct_召回", "direct_合并", "direct_拼接", "direct_文件", "direct_位置",
               "召回差(pp)", "文件召回差(pp)"]
    for ci, h in enumerate(headers, start=1):
        ws.cell(row=1, column=ci, value=h).font = Font(bold=True)
    mq = per_topn["MQ"]
    di = per_topn["direct"]
    for ri, cn in enumerate(chunks, start=2):
        a, b = mq[cn], di[cn]
        ws.cell(row=ri, column=1, value=cn)
        ws.cell(row=ri, column=2, value=topn)
        vals = [a[0], a[1], a[2], a[3], a[4], b[0], b[1], b[2], b[3], b[4],
                round(a[0] - b[0], 1), round(a[3] - b[3], 1)]
        for ci, v in enumerate(vals, start=3):
            ws.cell(row=ri, column=ci, value=v)
    fn = exp_dir / f"top{topn}" / "cross_对比.xlsx"
    wb.save(fn)
    print(f"  十字对比: {fn}")


# ── 主入口 ──

def run_multi_query_experiment(
    test_set_path: str,
    kb_name: str = DEFAULT_KB,
    chunkers: list[str] | None = None,
    top_k_values: list[int] | None = None,
    retriever: str = "hybrid",
    gen_model: str = CHUNK_LLM_MODEL,
    exp_name: str = "multi_query_汇总",
    output_base: str = "output",
    workers: int = SEARCH_WORKERS,
):
    """入口。返回 (exp_dir, per_topk_cross)。"""
    require_budget(task="Multi-Query 相似问题生成")

    chunks = chunkers or [
        "fixed_1100_ovl150_raw_pure", "fixed_1500_raw_pure", "fixed_1500_ovl200_raw_pure",
    ]
    topks = top_k_values or [5, 10]

    exp_dir = Path(output_base) / kb_name / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "name": exp_name,
        "kb_name": kb_name,
        "test_set_path": test_set_path,
        "chunkers": chunks,
        "top_k_values": topks,
        "retriever": retriever,
        "gen_model": gen_model,
        "generated_per_question": 3,
        "fusion": f"rrf(k={RRF_K})",
        "reuse_generation_across_chunkers": True,
        "output_dir": str(exp_dir),
    }
    (exp_dir / "config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    questions = load_test_set(test_set_path)
    total = len(questions)
    print(f"测试集: {total} 题 | chunkers: {chunks} | topk: {topks}")
    print(f"生成模型: {gen_model}（每题 3 个相似问题，跨 chunker 复用）")
    print()

    # 1) 生成相似问题（串行 LLM）
    llm = _build_llm(gen_model)
    similar_map: dict[int, list[str]] = {}
    t0 = time.time()
    ok = 0
    for qi, it in enumerate(questions):
        sims = generate_similar_questions(llm, it["question"], n=3)
        similar_map[qi] = sims
        ok += 1 if sims else 0
        if (qi + 1) % 25 == 0 or qi == total - 1:
            print(f"  生成 {qi + 1}/{total} ... {time.time() - t0:.0f}s")
    print(f"相似问题生成完成：{ok}/{total} 题至少 1 条，耗时 {time.time() - t0:.1f}s")
    (exp_dir / "generated_queries.jsonl").write_text(
        "\n".join(json.dumps(
            {"id": questions[qi]["id"], "question": questions[qi]["question"],
             "similar": similar_map[qi]}, ensure_ascii=False)
            for qi in range(total)),
        encoding="utf-8")
    print(f"生成结果已保存: {exp_dir / 'generated_queries.jsonl'}")

    # 2) 按 topn × group 评测
    per_topk_cross = {}
    for topn in topks:
        tk_dir = exp_dir / f"top{topn}"
        tk_dir.mkdir(parents=True, exist_ok=True)
        per_topn = {}
        for group in ("MQ", "direct"):
            rates_map, fn = _evaluate_one_group(
                questions, kb_name, chunks, similar_map, topn, retriever, group,
                tk_dir, workers=workers)
            per_topn[group] = rates_map
            from pipeline.report import generate_recall_report
            generate_recall_report(str(fn), str(tk_dir / f"report_{group}.html"),
                                   title=f"Multi-Query 召回 [{group}] (Top-{topn})")
        _write_cross_comparison(exp_dir, topn, chunks, per_topn)
        per_topk_cross[topn] = per_topn

        print(f"\n=== Top-{topn} 十字对照（召回率%）===")
        mq = per_topn["MQ"]
        di = per_topn["direct"]
        for cn in chunks:
            print(f"  {cn}: MQ={mq[cn][0]} | direct={di[cn][0]} | 差={mq[cn][0] - di[cn][0]:+.1f}")

    print("\n实验完成。输出目录:", exp_dir)
    return exp_dir, per_topk_cross