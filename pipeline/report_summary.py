"""数据源层统一汇总报告 — 扫描所有策略文件夹，生成 index.html 多策略多 top-k 对比。"""
from __future__ import annotations

import json
from pathlib import Path

import openpyxl

RANK_COLORS = ["#2e7d32", "#f9a825", "#d84315", "#546e7a", "#1565c0", "#6a1b9a", "#00838f", "#c62828"]

TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'Microsoft YaHei',sans-serif; background:#f0f2f5; padding:20px; }}
  .container {{ max-width:1300px; margin:0 auto; }}
  h1 {{ font-size:22px; color:#1a1a2e; margin-bottom:4px; }}
  .subtitle {{ color:#888; font-size:13px; margin-bottom:20px; }}
  .chart-row {{ display:flex; gap:16px; margin-bottom:16px; flex-wrap:wrap; }}
  .chart-card {{ flex:1; min-width:500px; background:#fff; border-radius:10px;
    box-shadow:0 2px 12px rgba(0,0,0,.06); padding:16px; }}
  .chart-card h3 {{ font-size:15px; color:#333; margin-bottom:8px; }}
  .chart-card.wide {{ flex:2; min-width:700px; }}
  table {{ width:100%; border-collapse:collapse; background:#fff; border-radius:10px;
    box-shadow:0 2px 12px rgba(0,0,0,.06); margin-top:16px; }}
  th {{ background:#2b579a; color:#fff; padding:10px 12px; text-align:center; font-size:13px; }}
  td {{ padding:8px 12px; text-align:center; border-bottom:1px solid #eee; font-size:13px; }}
  tr:hover td {{ background:#f0f4ff; }}
  .best {{ font-weight:bold; color:#2e7d32; }}
  .note {{ color:#999; font-size:12px; margin-top:16px; }}
</style>
</head>
<body>
<div class="container">
<h1>{title}</h1>
<div class="subtitle">{subtitle}</div>
<div class="chart-row">
  {charts}
</div>
{ranking_table}
<div class="note">{footer}</div>
</div>
<script>
  {plot_script}
</script>
</body>
</html>"""


def collect_results(output_dir: str) -> dict[str, dict[str, tuple[float, float, float, float, str]]]:
    """扫描 output_dir 下所有策略文件夹，返回 {strategy: {topk: (recall%, merged%, concat%, file_recall%, pos)}}。"""
    base = Path(output_dir)
    results: dict[str, dict[str, tuple[float, float, float, float, str]]] = {}

    for strategy_dir in sorted(base.iterdir()):
        if not strategy_dir.is_dir() or strategy_dir.name.startswith("_"):
            continue
        sname = strategy_dir.name
        results[sname] = {}

        for topk_dir in sorted(strategy_dir.iterdir()):
            if not topk_dir.is_dir():
                continue
            xlsx_path = topk_dir / "recall.xlsx"
            if not xlsx_path.exists():
                continue

            wb = openpyxl.load_workbook(xlsx_path)
            if "综合召回" not in wb.sheetnames:
                wb.close()
                continue

            ws = wb["综合召回"]
            for row in range(2, ws.max_row + 1):
                name = ws.cell(row=row, column=1).value
                if name and str(name) == sname:
                    rate = ws.cell(row=row, column=2).value
                    merged_rate = ws.cell(row=row, column=3).value
                    concat_rate = ws.cell(row=row, column=4).value
                    pos = ws.cell(row=row, column=5).value
                    file_rate = ws.cell(row=row, column=6).value
                    results[sname][topk_dir.name] = (
                        float(rate) if rate is not None else 0.0,
                        float(merged_rate) if merged_rate is not None else 0.0,
                        float(concat_rate) if concat_rate is not None else 0.0,
                        float(file_rate) if file_rate is not None else 0.0,
                        str(pos) if pos is not None else "--",
                    )
                    break
            wb.close()

    return results


def generate_summary_report(output_dir: str, title: str = "RAG 召回率汇总") -> str:
    """生成统一汇总报告，返回生成的 HTML 路径。"""
    results = collect_results(output_dir)

    if not results:
        print("未找到任何策略结果")
        return ""

    strategies = sorted(results.keys())
    topk_set = sorted(set(tk for s in strategies for tk in results[s]), key=lambda x: int(x.replace("top", "")))

    # 排名表：按第一个 top-k 的内容召回率降序
    first_topk = topk_set[0] if topk_set else ""
    ordered = sorted(strategies, key=lambda s: results[s].get(first_topk, (0, 0, 0, 0, ""))[0], reverse=True)

    # 构建排名表
    header_cols = ["排名", "策略"] + [f"Top-{tk.replace('top', '')} 内容召回" for tk in topk_set] \
                  + [f"Top-{tk.replace('top', '')} 合并召回" for tk in topk_set] \
                  + [f"Top-{tk.replace('top', '')} 拼接召回" for tk in topk_set] \
                  + [f"Top-{tk.replace('top', '')} 文件召回" for tk in topk_set] \
                  + [f"Top-{tk.replace('top', '')} 命中位置" for tk in topk_set]
    rows_html = ""
    for i, sname in enumerate(ordered):
        cells = f"<td>{i + 1}</td><td>{sname}</td>"
        for tk in topk_set:
            r = results[sname].get(tk, ("--", "--", "--", "--", "--"))
            cells += f"<td>{r[0]}%</td>"
        for tk in topk_set:
            r = results[sname].get(tk, ("--", "--", "--", "--", "--"))
            cells += f"<td>{r[1]}%</td>"
        for tk in topk_set:
            r = results[sname].get(tk, ("--", "--", "--", "--", "--"))
            cells += f"<td>{r[2]}%</td>"
        for tk in topk_set:
            r = results[sname].get(tk, ("--", "--", "--", "--", "--"))
            cells += f"<td>{r[3]}%</td>"
        for tk in topk_set:
            r = results[sname].get(tk, ("--", "--", "--", "--", "--"))
            cells += f"<td>{r[4]}</td>"
        rows_html += f"<tr>{cells}</tr>\n"

    ranking_table = f"<table><tr>{''.join(f'<th>{h}</th>' for h in header_cols)}</tr>{rows_html}</table>"

    # Plotly 数据
    plot_data = {
        "strategies": ordered,
        "topk": topk_set,
        "results": {s: {tk: list(results[s].get(tk, (0, 0, 0, 0, ""))) for tk in topk_set} for s in ordered},
    }

    plot_script = f"""
const data = {json.dumps(plot_data, ensure_ascii=False)};
const colors = {json.dumps(RANK_COLORS)};
const strategies = data.strategies;
const topk = data.topk;

// 内容召回率图
const recallTraces = topk.map((tk, ti) => {{
  const vals = strategies.map(s => data.results[s][tk][0]);
  return {{
    type: 'bar', x: vals, y: strategies, orientation: 'h',
    name: tk.replace('top', 'Top-'),
    marker: {{ color: colors[ti % colors.length] }},
    text: vals.map(v => v.toFixed(1) + '%'), textposition: 'outside',
  }};
}});

Plotly.newPlot('chart-recall', recallTraces, {{
  margin: {{ l: 180, r: 80, t: 10, b: 30 }},
  xaxis: {{ title: '内容召回率 (%)', range: [0, 100] }},
  yaxis: {{ automargin: true }},
  barmode: 'group',
}}, {{ responsive: true }});

// 文件召回率图
const fileTraces = topk.map((tk, ti) => {{
  const vals = strategies.map(s => data.results[s][tk][3]);
  return {{
    type: 'bar', x: vals, y: strategies, orientation: 'h',
    name: tk.replace('top', 'Top-'),
    marker: {{ color: colors[ti % colors.length] }},
    text: vals.map(v => v.toFixed(1) + '%'), textposition: 'outside',
  }};
}});

Plotly.newPlot('chart-file', fileTraces, {{
  margin: {{ l: 180, r: 80, t: 10, b: 30 }},
  xaxis: {{ title: '文件召回率 (%)', range: [0, 100] }},
  yaxis: {{ automargin: true }},
  barmode: 'group',
}}, {{ responsive: true }});
"""

    charts = f"""
  <div class="chart-card">
    <h3>内容召回率 — 按策略对比（降序）</h3>
    <div id="chart-recall" class="plot" style="height:{max(300, len(ordered) * 30 + 100)}px;"></div>
  </div>
  <div class="chart-card">
    <h3>文件召回率 — 按策略对比（降序）</h3>
    <div id="chart-file" class="plot" style="height:{max(300, len(ordered) * 30 + 100)}px;"></div>
  </div>"""

    kb_name = Path(output_dir).name
    subtitle = f"数据源: {kb_name}  |  {len(strategies)} 个策略  |  {len(topk_set)} 个 Top-K 范围"
    footer = f"Generated by RAG-Pro  |  更新策略后重新运行即可动态刷新"

    html = TEMPLATE.format(
        title=title, subtitle=subtitle,
        charts=charts, ranking_table=ranking_table, plot_script=plot_script,
        footer=footer,
    )

    out_path = Path(output_dir) / "index.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"汇总报告已生成: {out_path}")
    return str(out_path)


if __name__ == "__main__":
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "output/demo"
    title = sys.argv[2] if len(sys.argv) > 2 else "RAG 召回率汇总"
    generate_summary_report(output_dir, title)