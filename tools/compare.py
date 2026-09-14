"""
策略对比分析 — 两策略得分差异、召回交叉、分项拆解、大分差题目详情。
用法: python main.py compare <scored.xlsx> --a <策略A> --b <策略B> [--top N]
"""
from collections import defaultdict
from pathlib import Path

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter


# ══════════════════════════════════════════════════
# 数据加载
# ══════════════════════════════════════════════════

def _parse_summary(ws):
    """解析汇总 sheet，返回 (fixed_cols, strategy_map, rows_data)。"""
    headers = [str(ws.cell(row=1, column=c).value or "") for c in range(1, ws.max_column + 1)]

    fixed_cols = []
    strategy_map = {}
    for ci, h in enumerate(headers, start=1):
        if h.endswith("_召回位置"):
            continue
        elif h.endswith("_召回"):
            sname = h[:-3]
            strategy_map.setdefault(sname, {})["召回"] = ci
        elif h.endswith("_得分"):
            sname = h[:-3]
            strategy_map.setdefault(sname, {})["得分"] = ci
        elif h.endswith("_回答"):
            sname = h[:-3]
            strategy_map.setdefault(sname, {})["回答"] = ci
        elif "_召回位置" not in h and "_召回" not in h and "_得分" not in h and "_回答" not in h:
            fixed_cols.append((ci, h))

    for ci, h in enumerate(headers, start=1):
        if h.endswith("_召回位置"):
            sname = h[:-5]
            if sname in strategy_map:
                strategy_map[sname]["召回位置"] = ci

    rows = []
    for r in range(2, ws.max_row + 1):
        row = {}
        row["_row"] = r
        for ci, h in fixed_cols:
            row[h] = ws.cell(row=r, column=ci).value
        for sname, cols in strategy_map.items():
            for key, col in cols.items():
                row[f"{sname}__{key}"] = ws.cell(row=r, column=col).value
        rows.append(row)
    return fixed_cols, strategy_map, rows


def _parse_detail(ws):
    """解析详情 sheet，返回 (fixed_cols, retrieval_pairs, rows_data)。"""
    headers = [str(ws.cell(row=1, column=c).value or "") for c in range(1, ws.max_column + 1)]

    fixed_cols = []
    retrieval_pairs = []
    detail_keys = []
    i = 0
    while i < len(headers):
        h = headers[i]
        if h.startswith("检索内容"):
            content_col = i + 1
            source_col = i + 2 if i + 1 < len(headers) else None
            retrieval_pairs.append((content_col, source_col))
            i += 2
        elif h in ("召回", "召回位置"):
            detail_keys.append((i + 1, h))
            i += 1
        elif h in ("最终回答", "总分", "必答点得分", "选答点得分", "参考文件得分", "评分理由"):
            detail_keys.append((i + 1, h))
            i += 1
        else:
            fixed_cols.append((i + 1, h))
            i += 1

    rows = []
    for r in range(2, ws.max_row + 1):
        row = {}
        for ci, h in fixed_cols:
            row[h] = ws.cell(row=r, column=ci).value
        for ci, h in detail_keys:
            row[h] = ws.cell(row=r, column=ci).value
        docs = []
        for ci_content, ci_source in retrieval_pairs:
            content = str(ws.cell(row=r, column=ci_content).value or "")
            source = str(ws.cell(row=r, column=ci_source).value or "") if ci_source else ""
            docs.append({"content": content, "source": source})
        row["_docs"] = docs
        rows.append(row)
    return fixed_cols, detail_keys, retrieval_pairs, rows


# ══════════════════════════════════════════════════
# 分析函数
# ══════════════════════════════════════════════════

def win_lose_draw(rows, sname_a, sname_b):
    """赢/平/输统计。"""
    a_k = f"{sname_a}__得分"
    b_k = f"{sname_b}__得分"
    a_wins = b_wins = draws = 0
    a_win_margin = b_win_margin = 0.0
    for row in rows:
        sa = float(row.get(a_k, 0) or 0)
        sb = float(row.get(b_k, 0) or 0)
        if sa > sb:
            a_wins += 1
            a_win_margin += sa - sb
        elif sb > sa:
            b_wins += 1
            b_win_margin += sb - sa
        else:
            draws += 1
    return {
        "A赢": a_wins, "A平均领先": round(a_win_margin / a_wins, 1) if a_wins else 0,
        "平": draws,
        "B赢": b_wins, "B平均领先": round(b_win_margin / b_wins, 1) if b_wins else 0,
    }


def score_diff_table(rows, sname_a, sname_b, detail_a_rows, detail_b_rows):
    """逐题得分差，含完整回答和检索片段，按差值降序。"""
    a_score_k = f"{sname_a}__得分"
    b_score_k = f"{sname_b}__得分"
    a_ans_k = f"{sname_a}__回答"
    b_ans_k = f"{sname_b}__回答"
    a_recall_k = f"{sname_a}__召回"
    b_recall_k = f"{sname_b}__召回"
    a_pos_k = f"{sname_a}__召回位置"
    b_pos_k = f"{sname_b}__召回位置"

    diffs = []
    for i, row in enumerate(rows):
        sa = float(row.get(a_score_k, 0) or 0)
        sb = float(row.get(b_score_k, 0) or 0)
        qid = str(row.get("问题ID", row.get("_row", "")))
        question = str(row.get("问题", ""))
        a_ans = str(row.get(a_ans_k, ""))
        b_ans = str(row.get(b_ans_k, ""))
        a_recall = int(row.get(a_recall_k, 0) or 0)
        b_recall = int(row.get(b_recall_k, 0) or 0)
        a_pos = str(row.get(a_pos_k, ""))
        b_pos = str(row.get(b_pos_k, ""))

        a_docs = detail_a_rows[i].get("_docs", []) if i < len(detail_a_rows) else []
        b_docs = detail_b_rows[i].get("_docs", []) if i < len(detail_b_rows) else []

        diffs.append({
            "问题ID": qid, "问题": question,
            "A得分": sa, "B得分": sb, "差值": round(sa - sb, 2),
            "A回答": a_ans, "B回答": b_ans,
            "A召回": a_recall, "B召回": b_recall,
            "A召回位置": a_pos, "B召回位置": b_pos,
            "A检索": a_docs, "B检索": b_docs,
        })
    diffs.sort(key=lambda x: -x["差值"])
    return diffs


def recall_cross_table(rows, sname_a, sname_b):
    """召回交叉表。"""
    a_k = f"{sname_a}__召回"
    b_k = f"{sname_b}__召回"
    a_score_k = f"{sname_a}__得分"
    b_score_k = f"{sname_b}__得分"
    cells = {"双召": [], "A独召": [], "B独召": [], "双失": []}
    for row in rows:
        ra = int(row.get(a_k, 0) or 0)
        rb = int(row.get(b_k, 0) or 0)
        sa = float(row.get(a_score_k, 0) or 0)
        sb = float(row.get(b_score_k, 0) or 0)
        if ra and rb:
            cells["双召"].append((sa, sb))
        elif ra:
            cells["A独召"].append((sa, sb))
        elif rb:
            cells["B独召"].append((sa, sb))
        else:
            cells["双失"].append((sa, sb))

    result = {}
    for label, scores in cells.items():
        n = len(scores)
        if n:
            avg_a = round(sum(s[0] for s in scores) / n, 1)
            avg_b = round(sum(s[1] for s in scores) / n, 1)
        else:
            avg_a = avg_b = 0
        result[label] = {"题数": n, "A均分": avg_a, "B均分": avg_b}
    return result


def score_breakdown(rows, sname_a, sname_b, detail_a_rows, detail_b_rows):
    """分项得分对比 — 从详情 sheet 读取。"""
    def _avg_detail(detail_rows, key):
        vals = [float(r.get(key, 0) or 0) for r in detail_rows]
        return round(sum(vals) / len(vals), 1) if vals else 0

    return {
        sname_a: {
            "必答点均分": _avg_detail(detail_a_rows, "必答点得分"),
            "选答点均分": _avg_detail(detail_a_rows, "选答点得分"),
            "参考文件均分": _avg_detail(detail_a_rows, "参考文件得分"),
        },
        sname_b: {
            "必答点均分": _avg_detail(detail_b_rows, "必答点得分"),
            "选答点均分": _avg_detail(detail_b_rows, "选答点得分"),
            "参考文件均分": _avg_detail(detail_b_rows, "参考文件得分"),
        },
    }


def top_divergence(rows, sname_a, sname_b, detail_a_rows, detail_b_rows, n=5):
    """分差最大的 N 题，含回答和检索详情。"""
    a_score_k = f"{sname_a}__得分"
    b_score_k = f"{sname_b}__得分"
    a_ans_k = f"{sname_a}__回答"
    b_ans_k = f"{sname_b}__回答"

    diffs = []
    for i, row in enumerate(rows):
        sa = float(row.get(a_score_k, 0) or 0)
        sb = float(row.get(b_score_k, 0) or 0)
        correct = str(detail_a_rows[i].get("正确回答", "")) if i < len(detail_a_rows) else ""
        diffs.append({
            "idx": i,
            "问题ID": str(row.get("问题ID", "")),
            "问题": str(row.get("问题", ""))[:80],
            "正确回答": correct,
            "A得分": sa, "B得分": sb, "差值": round(sa - sb, 2),
            "A回答": str(row.get(a_ans_k, ""))[:500],
            "B回答": str(row.get(b_ans_k, ""))[:500],
            "A检索": detail_a_rows[i].get("_docs", []) if i < len(detail_a_rows) else [],
            "B检索": detail_b_rows[i].get("_docs", []) if i < len(detail_b_rows) else [],
        })
    diffs.sort(key=lambda x: -abs(x["差值"]))
    return diffs[:n]


# ══════════════════════════════════════════════════
# 输出
# ══════════════════════════════════════════════════

GREEN = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
RED = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
YELLOW = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")


def write_report(output_path, sname_a, sname_b, wld, diffs, cross, breakdown, top_n):
    wb = openpyxl.Workbook()
    bold = Font(bold=True)

    # Sheet 1: 概览
    ws = wb.active
    ws.title = "概览"
    ws.cell(row=1, column=1, value="对比概览").font = Font(bold=True, size=14)
    ws.cell(row=2, column=1, value=f"A: {sname_a}")
    ws.cell(row=3, column=1, value=f"B: {sname_b}")

    ws.cell(row=5, column=1, value="赢/平/输").font = bold
    ws.cell(row=6, column=1, value="A 赢"); ws.cell(row=6, column=2, value=wld["A赢"])
    ws.cell(row=6, column=3, value="平均领先"); ws.cell(row=6, column=4, value=wld["A平均领先"])
    ws.cell(row=7, column=1, value="平"); ws.cell(row=7, column=2, value=wld["平"])
    ws.cell(row=8, column=1, value="B 赢"); ws.cell(row=8, column=2, value=wld["B赢"])
    ws.cell(row=8, column=3, value="平均领先"); ws.cell(row=8, column=4, value=wld["B平均领先"])

    # 召回交叉表
    ws.cell(row=10, column=1, value="召回交叉表").font = bold
    headers = ["", "B召回=1", "B召回=0"]
    for ci, h in enumerate(headers, start=1):
        ws.cell(row=11, column=ci, value=h).font = bold
    ws.cell(row=12, column=1, value="A召回=1").font = bold
    ws.cell(row=12, column=2, value=f"{cross['双召']['题数']}题 (A均分{cross['双召']['A均分']}/B均分{cross['双召']['B均分']})")
    ws.cell(row=12, column=3, value=f"{cross['A独召']['题数']}题 (A均分{cross['A独召']['A均分']}/B均分{cross['A独召']['B均分']})")
    ws.cell(row=13, column=1, value="A召回=0").font = bold
    ws.cell(row=13, column=2, value=f"{cross['B独召']['题数']}题 (A均分{cross['B独召']['A均分']}/B均分{cross['B独召']['B均分']})")
    ws.cell(row=13, column=3, value=f"{cross['双失']['题数']}题 (A均分{cross['双失']['A均分']}/B均分{cross['双失']['B均分']})")

    # 分项得分
    ws.cell(row=15, column=1, value="得分分项对比").font = bold
    for ci, h in enumerate(["", "必答点均分(70)", "选答点均分(20)", "参考文件均分(10)"], start=1):
        ws.cell(row=16, column=ci, value=h).font = bold
    ws.cell(row=17, column=1, value=sname_a).font = bold
    ws.cell(row=17, column=2, value=breakdown[sname_a]["必答点均分"])
    ws.cell(row=17, column=3, value=breakdown[sname_a]["选答点均分"])
    ws.cell(row=17, column=4, value=breakdown[sname_a]["参考文件均分"])
    ws.cell(row=18, column=1, value=sname_b).font = bold
    ws.cell(row=18, column=2, value=breakdown[sname_b]["必答点均分"])
    ws.cell(row=18, column=3, value=breakdown[sname_b]["选答点均分"])
    ws.cell(row=18, column=4, value=breakdown[sname_b]["参考文件均分"])

    # Sheet 2: 逐题得分差
    ws = wb.create_sheet("逐题得分差")
    headers = [
        "问题ID", "问题", "A得分", "B得分", "差值(A-B)", "优势方",
        "A召回", "B召回", "A召回位置", "B召回位置",
        f"A回答({sname_a[:30]})", f"B回答({sname_b[:30]})",
    ]
    for ki in range(5):
        headers.append(f"A检索{ki+1}")
        headers.append(f"A来源{ki+1}")
    for ki in range(5):
        headers.append(f"B检索{ki+1}")
        headers.append(f"B来源{ki+1}")

    for ci, h in enumerate(headers, start=1):
        ws.cell(row=1, column=ci, value=h).font = bold
        ws.cell(row=1, column=ci).fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")

    for ri, d in enumerate(diffs, start=2):
        ws.cell(row=ri, column=1, value=d["问题ID"])
        ws.cell(row=ri, column=2, value=d["问题"][:100])
        ws.cell(row=ri, column=3, value=d["A得分"])
        ws.cell(row=ri, column=4, value=d["B得分"])
        ws.cell(row=ri, column=5, value=d["差值"])
        ws.cell(row=ri, column=6, value="A" if d["差值"] > 0 else ("B" if d["差值"] < 0 else "平"))
        ws.cell(row=ri, column=7, value=d["A召回"])
        ws.cell(row=ri, column=8, value=d["B召回"])
        ws.cell(row=ri, column=9, value=d["A召回位置"])
        ws.cell(row=ri, column=10, value=d["B召回位置"])
        ws.cell(row=ri, column=11, value=d["A回答"][:30000])
        ws.cell(row=ri, column=12, value=d["B回答"][:30000])

        for ki, doc in enumerate(d["A检索"][:5]):
            ws.cell(row=ri, column=13 + ki * 2, value=str(doc.get("content", ""))[:30000])
            ws.cell(row=ri, column=14 + ki * 2, value=str(doc.get("source", "")))
        for ki, doc in enumerate(d["B检索"][:5]):
            ws.cell(row=ri, column=23 + ki * 2, value=str(doc.get("content", ""))[:30000])
            ws.cell(row=ri, column=24 + ki * 2, value=str(doc.get("source", "")))

        if d["差值"] > 0:
            ws.cell(row=ri, column=5).fill = GREEN
        elif d["差值"] < 0:
            ws.cell(row=ri, column=5).fill = RED

    ws.freeze_panes = "C2"

    # Sheet 3: 分差最大题目
    ws = wb.create_sheet("大分差题目")
    n_retrieval = len(top_n[0]["A检索"]) if top_n else 5
    hdrs = ["排名", "问题ID", "问题", "正确回答", "A得分", "B得分", "差值", "优势方",
            f"A回答({sname_a[:20]})", f"B回答({sname_b[:20]})"]
    for ki in range(n_retrieval):
        hdrs.append(f"A检索{ki+1}")
        hdrs.append(f"A来源{ki+1}")
    for ki in range(n_retrieval):
        hdrs.append(f"B检索{ki+1}")
        hdrs.append(f"B来源{ki+1}")

    for ci, h in enumerate(hdrs, start=1):
        ws.cell(row=1, column=ci, value=h).font = bold
        ws.cell(row=1, column=ci).fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")

    for ri, item in enumerate(top_n, start=2):
        ws.cell(row=ri, column=1, value=ri - 1)
        ws.cell(row=ri, column=2, value=item["问题ID"])
        ws.cell(row=ri, column=3, value=item["问题"][:100])
        ws.cell(row=ri, column=4, value=str(item.get("正确回答", ""))[:30000])
        ws.cell(row=ri, column=5, value=item["A得分"])
        ws.cell(row=ri, column=6, value=item["B得分"])
        ws.cell(row=ri, column=7, value=item["差值"])
        winner = "A" if item["差值"] > 0 else "B"
        ws.cell(row=ri, column=8, value=winner)
        ws.cell(row=ri, column=9, value=item["A回答"][:30000])
        ws.cell(row=ri, column=10, value=item["B回答"][:30000])

        base = 11
        for ki, doc in enumerate(item.get("A检索", [])[:n_retrieval]):
            ws.cell(row=ri, column=base + ki * 2, value=str(doc.get("content", ""))[:30000])
            ws.cell(row=ri, column=base + ki * 2 + 1, value=str(doc.get("source", "")))
        base = 11 + n_retrieval * 2
        for ki, doc in enumerate(item.get("B检索", [])[:n_retrieval]):
            ws.cell(row=ri, column=base + ki * 2, value=str(doc.get("content", ""))[:30000])
            ws.cell(row=ri, column=base + ki * 2 + 1, value=str(doc.get("source", "")))

        if item["差值"] > 0:
            ws.cell(row=ri, column=7).fill = GREEN
        elif item["差值"] < 0:
            ws.cell(row=ri, column=7).fill = RED

    for ws in wb.worksheets:
        for col in ws.columns:
            max_len = max((len(str(cell.value or "")) for cell in col), default=10)
            ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_len + 4, 60)

    wb.save(output_path)
    print(f"对比报告已保存: {output_path}")


# ══════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════

def run_compare(scored_xlsx: str, strategy_a: str, strategy_b: str,
                output: str = "", top_n: int = 5):
    """策略对比分析入口。output 为空时自动生成文件名。"""
    scored_path = Path(scored_xlsx)
    if not scored_path.exists():
        print(f"文件不存在: {scored_path}")
        return

    wb = openpyxl.load_workbook(scored_path)

    if "汇总" not in wb.sheetnames:
        print("错误: 缺少汇总 sheet")
        return
    ws_hz = wb["汇总"]
    _, strategy_map, rows = _parse_summary(ws_hz)
    available = list(strategy_map.keys())
    if strategy_a not in available:
        print(f"策略 A '{strategy_a}' 不在文件中。可用: {available}")
        return
    if strategy_b not in available:
        print(f"策略 B '{strategy_b}' 不在文件中。可用: {available}")
        return

    detail_a = None
    detail_b = None
    for sname in wb.sheetnames:
        if sname in strategy_a[:31] or strategy_a[:31] in sname:
            detail_a = sname
        if sname in strategy_b[:31] or strategy_b[:31] in sname:
            detail_b = sname
    _, _, _, detail_a_rows = _parse_detail(wb[detail_a]) if detail_a else ([], [], [], [])
    _, _, _, detail_b_rows = _parse_detail(wb[detail_b]) if detail_b else ([], [], [], [])

    print(f"\n{'='*60}")
    print(f"策略对比: {strategy_a}  vs  {strategy_b}")
    print(f"{'='*60}")

    wld = win_lose_draw(rows, strategy_a, strategy_b)
    print(f"\n--- 赢/平/输 ---")
    print(f"  A赢: {wld['A赢']} 题 (平均领先 {wld['A平均领先']} 分)")
    print(f"  平:   {wld['平']} 题")
    print(f"  B赢: {wld['B赢']} 题 (平均领先 {wld['B平均领先']} 分)")

    diffs = score_diff_table(rows, strategy_a, strategy_b, detail_a_rows, detail_b_rows)
    a_avg = round(sum(d["A得分"] for d in diffs) / len(diffs), 1)
    b_avg = round(sum(d["B得分"] for d in diffs) / len(diffs), 1)
    print(f"\n  A均分: {a_avg}  |  B均分: {b_avg}  |  差值: {round(a_avg - b_avg, 1)}")

    cross = recall_cross_table(rows, strategy_a, strategy_b)
    print(f"\n--- 召回交叉表 ---")
    print(f"  双召: {cross['双召']['题数']} 题  A均分{cross['双召']['A均分']}  B均分{cross['双召']['B均分']}")
    print(f"  A独召: {cross['A独召']['题数']} 题  A均分{cross['A独召']['A均分']}  B均分{cross['A独召']['B均分']}")
    print(f"  B独召: {cross['B独召']['题数']} 题  A均分{cross['B独召']['A均分']}  B均分{cross['B独召']['B均分']}")
    print(f"  双失: {cross['双失']['题数']} 题  A均分{cross['双失']['A均分']}  B均分{cross['双失']['B均分']}")

    breakdown = score_breakdown(rows, strategy_a, strategy_b, detail_a_rows, detail_b_rows)
    print(f"\n--- 得分分项 ---")
    for sname, bd in breakdown.items():
        print(f"  {sname}: 必答{bd['必答点均分']}  选答{bd['选答点均分']}  参考文件{bd['参考文件均分']}")

    top_items = top_divergence(rows, strategy_a, strategy_b, detail_a_rows, detail_b_rows, n=top_n)
    print(f"\n--- 分差最大的 {len(top_items)} 题 ---")
    for item in top_items:
        winner = "A" if item["差值"] > 0 else "B"
        print(f"  {item['问题ID']}: A={item['A得分']} B={item['B得分']} 差={item['差值']} ({winner}优)")
        print(f"    {item['问题'][:80]}")

    out = output or str(scored_path.with_stem(scored_path.stem + f"_compare_{strategy_a[:20]}_vs_{strategy_b[:20]}"))
    write_report(Path(out), strategy_a, strategy_b, wld, diffs, cross, breakdown, top_items)
    return out