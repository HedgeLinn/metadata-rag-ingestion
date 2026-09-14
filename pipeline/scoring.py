"""LLM 评分引擎。"""
import re
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Font

from config import PROMPTS, LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, SCORE_WORKERS
from langchain_openai import ChatOpenAI
from core.llm_cost import get_cost_callback, require_budget


def _call_score_api(prompt2: str, score_prompt: str, model_name: str = "") -> str:
    """调用评分 LLM 并返回原始文本，带超时和 429 重试。"""
    llm = ChatOpenAI(
        model=model_name or LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=0.1,
        timeout=120,
        max_retries=2,
        extra_body={"thinking": {"type": "disabled"}},
        callbacks=[get_cost_callback()],
    )
    for attempt in range(3):
        try:
            resp = llm.invoke([
                {"role": "system", "content": score_prompt},
                {"role": "user", "content": prompt2},
            ])
            return resp.content.replace("```json", "").replace("```", "").strip()
        except Exception as e:
            err = str(e)
            if "429" in err:
                wait = 5 * (attempt + 1)
                print(f"  [评分API] 429 限流, 等待{wait}秒重试...")
                time.sleep(wait)
            else:
                print(f"  [评分API] 失败: {e}")
                return ""
    print(f"  [评分API] 重试3次后仍失败")
    return ""


def _parse_score_reason(result: str) -> str:
    if not result:
        return ""
    try:
        obj = json.loads(result)
    except json.JSONDecodeError:
        try:
            import ast
            obj = json.loads(json.dumps(ast.literal_eval(result), ensure_ascii=False))
        except Exception:
            points = re.findall(r'\d+\..*?(?=\d+\.|$)', result)
            return '\n'.join(points)

    reason = obj.get('评分理由', [])
    if isinstance(reason, list):
        return '\n'.join([item.strip() for item in reason if isinstance(item, str)])
    elif isinstance(reason, str):
        points = re.findall(r'\d+\..*?(?=\d+\.|$)', reason)
        return '\n'.join(points)
    return ""


def _calculate_score(score_reason: str) -> dict:
    if not score_reason:
        return {"总分": 0, "必答点得分": 0, "选答点得分": 0, "参考文件得分": 0}

    req = re.findall(r'\[必答\](\d+):.*?(得分|0分)', score_reason)
    opt = re.findall(r'\[选答\](\d+):.*?(得分|0分)', score_reason)
    ref = re.findall(r'\[参考文件\](\d+):.*?(得分|0分)', score_reason)

    req_deduction = (70 / len(req) * sum(1 for _, r in req if r == '0分')) if req else 0
    opt_deduction = (20 / len(opt) * sum(1 for _, r in opt if r == '0分')) if opt else 0
    ref_deduction = (10 / len(ref) * sum(1 for _, r in ref if r == '0分')) if ref else 0

    req_score = 70 - req_deduction if req else 0
    opt_score = 20 - opt_deduction if opt else 0
    ref_score = 10 - ref_deduction if ref else 0

    core = 100 - (req_deduction + opt_deduction + ref_deduction)
    surplus = len(re.findall(r'\(surplus\)', score_reason)) * 5
    surplus = min(surplus, 30)

    if '[冲突点]' in score_reason:
        total = 0
    else:
        total = round(max(core - surplus, 0), 2)

    return {"总分": total, "必答点得分": round(req_score, 2), "选答点得分": round(opt_score, 2), "参考文件得分": round(ref_score, 2)}


def score_eval_output(input_xlsx: str, output_xlsx: str, strategy_names: list[str],
                      strict: bool = False, pre_scores_json: str = ""):
    """对 eval 输出的 xlsx 进行 LLM 评分。pre_scores_json 含多轮选优时的预评分，跳过重复 LLM 调用。"""
    require_budget(task="LLM 评分")
    prompt_key = "score_rubric_strict" if strict else "score_rubric"
    print(f"评分模式: {'严格版' if strict else '宽泛版'} ({prompt_key})")
    score_prompt = PROMPTS[prompt_key]

    # 加载预评分
    pre_scores: dict[str, dict[int, float]] = {}
    has_runs = False
    if pre_scores_json:
        try:
            raw = json.loads(Path(pre_scores_json).read_text(encoding="utf-8"))
            # JSON 会将 int key 转为 string，需恢复
            pre_scores = {sname: {int(k): v for k, v in qs.items()} for sname, qs in raw.items()}
            has_runs = True
            total_pre = sum(len(v) for v in pre_scores.values())
            print(f"预评分加载: {total_pre} 条 (跳过重复 LLM 调用)")
        except Exception:
            pass

    from openpyxl import load_workbook
    wb_in = load_workbook(input_xlsx)
    ws_hz = wb_in["汇总"]
    all_headers = [str(ws_hz.cell(row=1, column=c).value or "") for c in range(1, ws_hz.max_column + 1)]

    input_col_names = []
    for h in all_headers:
        if h in strategy_names:
            break
        input_col_names.append(h)

    sname_to_header = {}
    for s in strategy_names:
        for h in all_headers:
            if s in h:
                sname_to_header[s] = h
                break
    active_strategies = list(sname_to_header.keys())

    # 检索内容
    retrieval_data: dict[str, dict[int, list[tuple[str, str]]]] = {}
    retrieval_headers: list[str] = []
    for sname in active_strategies:
        retrieval_data[sname] = {}
        detail_sheet = sname[:31]
        if detail_sheet not in wb_in.sheetnames:
            continue
        ws_src = wb_in[detail_sheet]
        src_hdrs = [str(ws_src.cell(1, c).value or "") for c in range(1, ws_src.max_column + 1)]
        content_cols = [ci for ci, h in enumerate(src_hdrs) if "检索" in h and "内容" in h]
        source_cols = [ci for ci, h in enumerate(src_hdrs) if "来源" in h]
        pairs: list[tuple[int, int]] = []
        for i, ci in enumerate(content_cols):
            si = source_cols[i] if i < len(source_cols) else -1
            pairs.append((ci, si))
        if not retrieval_headers and pairs:
            for j in range(len(pairs)):
                retrieval_headers.append(f"检索内容{j+1}")
                retrieval_headers.append(f"来源{j+1}")
        for ri in range(2, ws_src.max_row + 1):
            docs = []
            for ci, si in pairs:
                content = str(ws_src.cell(ri, ci + 1).value or "")
                source = str(ws_src.cell(ri, si + 1).value or "") if si >= 0 else ""
                docs.append((content, source))
            retrieval_data[sname][ri - 2] = docs

    # 读取全量轮次得分 (从 _轮次 sheet)
    all_runs_scores: dict[str, list[float]] = {sname: [] for sname in active_strategies}
    if has_runs:
        for sname in active_strategies:
            runs_sheet = f"{sname}_轮次"[:31]
            if runs_sheet not in wb_in.sheetnames:
                continue
            ws_r = wb_in[runs_sheet]
            r_hdrs = [str(ws_r.cell(1, c).value or "") for c in range(1, ws_r.max_column + 1)]
            score_cols = [ci for ci, h in enumerate(r_hdrs) if h.startswith("得分_")]
            for ri in range(2, ws_r.max_row + 1):
                for ci in score_cols:
                    v = ws_r.cell(ri, ci + 1).value
                    if v is not None and str(v).strip():
                        try:
                            all_runs_scores[sname].append(float(v))
                        except ValueError:
                            pass

    rows_data = []
    for r in range(2, ws_hz.max_row + 1):
        row = {h: ws_hz.cell(row=r, column=c+1).value for c, h in enumerate(all_headers)}
        rows_data.append(row)

    # 收集评分任务 — 跳过已有预评分的
    DEFAULT_SCORE = {"总分": 0, "必答点得分": 0, "选答点得分": 0, "参考文件得分": 0}
    all_scores = {sname: [DEFAULT_SCORE.copy()] * len(rows_data) for sname in active_strategies}
    all_reasons = {sname: [""] * len(rows_data) for sname in active_strategies}
    tasks = []
    prescore_count = 0
    for sname in active_strategies:
        hdr = sname_to_header[sname]
        for idx, row in enumerate(rows_data):
            ai_answer = str(row.get(hdr, "") or "").strip()
            if not ai_answer:
                continue
            # 有预评分直接复用
            if sname in pre_scores and idx in pre_scores[sname]:
                all_scores[sname][idx] = {"总分": pre_scores[sname][idx], "必答点得分": 0,
                                           "选答点得分": 0, "参考文件得分": 0}
                all_reasons[sname][idx] = "（多轮选优，已在生成阶段评分）"
                prescore_count += 1
            else:
                tasks.append((sname, idx, str(row.get("问题", "")), str(row.get("point", "")),
                             str(row.get("必答点", "")), str(row.get("选答点", "")),
                             str(row.get("参考文件", "")), ai_answer))

    total_tasks = len(tasks)
    if prescore_count:
        print(f"预评分复用: {prescore_count} 条跳过")
    if total_tasks == 0:
        print("评分任务为空，全部使用预评分")
    else:
        completed = 0
        print(f"\n并发评分: {total_tasks} 任务, {SCORE_WORKERS} 线程")

        def _score_one(task):
            sname, idx, question, point, bidian, xuanda, ref_file, ai_answer = task
            prompt2 = json.dumps({
                "问题": question, "参考要点": point, "必答点": bidian,
                "选答点": xuanda, "参考文件": ref_file, "ai答案": ai_answer,
            }, ensure_ascii=False, indent=2)
            raw = _call_score_api(prompt2, score_prompt)
            reason = _parse_score_reason(raw)
            scores = _calculate_score(reason)
            return sname, idx, scores, reason

        with ThreadPoolExecutor(max_workers=SCORE_WORKERS) as executor:
            futures = {executor.submit(_score_one, t): t for t in tasks}
            for future in as_completed(futures):
                sname, idx, scores, reason = future.result()
                all_scores[sname][idx] = scores
                all_reasons[sname][idx] = reason
                completed += 1
                print(f"  [{completed}/{total_tasks}] [{sname}] Q{idx+1}: 总分={scores['总分']}")

    # 输出 xlsx
    wb_out = Workbook()
    wb_out.remove(wb_out.active)

    ws_summary = wb_out.create_sheet("汇总")
    summary_headers = list(input_col_names)
    for sname in active_strategies:
        summary_headers.append(f"{sname}_回答")
        summary_headers.append(f"{sname}_得分")
        summary_headers.append(f"{sname}_召回")
        summary_headers.append(f"{sname}_召回位置")
    for ci, h in enumerate(summary_headers, start=1):
        ws_summary.cell(row=1, column=ci, value=h).font = Font(bold=True)
    for ri, row in enumerate(rows_data, start=2):
        for ci, h in enumerate(input_col_names, start=1):
            val = row.get(h, "")
            ws_summary.cell(row=ri, column=ci, value=str(val)[:30000] if val is not None else "")
        col_offset = len(input_col_names) + 1
        for sname in active_strategies:
            ws_summary.cell(row=ri, column=col_offset, value=str(row.get(sname, ""))[:30000])
            ws_summary.cell(row=ri, column=col_offset + 1, value=all_scores[sname][ri - 2]["总分"])
            ws_summary.cell(row=ri, column=col_offset + 2, value=row.get(f"{sname}_召回", ""))
            ws_summary.cell(row=ri, column=col_offset + 3, value=row.get(f"{sname}_召回位置", ""))
            col_offset += 4

    for sname in active_strategies:
        ws = wb_out.create_sheet(sname)
        detail_headers = list(input_col_names) + ["最终回答", "总分", "必答点得分", "选答点得分", "参考文件得分", "评分理由"] + retrieval_headers + ["召回", "召回位置"]
        hdr_name = sname_to_header[sname]
        for ci, h in enumerate(detail_headers, start=1):
            ws.cell(row=1, column=ci, value=h).font = Font(bold=True)
        for ri, row in enumerate(rows_data, start=2):
            for ci, h in enumerate(input_col_names, start=1):
                val = row.get(h, "")
                ws.cell(row=ri, column=ci, value=str(val)[:30000] if val is not None else "")
            col = len(input_col_names) + 1
            ws.cell(row=ri, column=col, value=str(row.get(hdr_name, ""))[:30000]); col += 1
            scores = all_scores[sname][ri - 2]
            ws.cell(row=ri, column=col, value=scores["总分"]); col += 1
            ws.cell(row=ri, column=col, value=scores["必答点得分"]); col += 1
            ws.cell(row=ri, column=col, value=scores["选答点得分"]); col += 1
            ws.cell(row=ri, column=col, value=scores["参考文件得分"]); col += 1
            ws.cell(row=ri, column=col, value=str(all_reasons[sname][ri - 2])[:30000]); col += 1
            for content, source in retrieval_data.get(sname, {}).get(ri - 2, []):
                ws.cell(row=ri, column=col, value=content[:30000]); col += 1
                ws.cell(row=ri, column=col, value=source[:30000]); col += 1
            ws.cell(row=ri, column=col, value=row.get(f"{sname}_召回", "")); col += 1
            ws.cell(row=ri, column=col, value=row.get(f"{sname}_召回位置", ""))

    # 综合评分: 最优轮平均分 + 全量平均分 + 召回率
    ws_score = wb_out.create_sheet("综合评分")
    for ci, h in enumerate(["策略", "最优轮平均分", "全量平均分"], start=1):
        ws_score.cell(row=1, column=ci, value=h).font = Font(bold=True)
    for ri, sname in enumerate(active_strategies, start=2):
        totals = [s["总分"] for s in all_scores[sname]]
        best_avg = round(sum(totals) / len(totals), 2) if totals else 0
        runs_totals = all_runs_scores.get(sname, [])
        all_avg = round(sum(runs_totals) / len(runs_totals), 2) if runs_totals else best_avg
        ws_score.cell(row=ri, column=1, value=sname)
        ws_score.cell(row=ri, column=2, value=best_avg)
        ws_score.cell(row=ri, column=3, value=all_avg)

    wb_out.save(output_xlsx)
    print(f"\n评分完成，保存至: {output_xlsx}")
