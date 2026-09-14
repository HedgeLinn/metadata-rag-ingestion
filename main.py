"""RAG-Pro — 多策略切片 RAG 评测 CLI。"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from config import DEFAULT_KB, DEFAULT_TOP_K


def cmd_strategies(args):
    """列出已注册策略。"""
    from core.registry import factory

    if args.type in (None, "chunker"):
        print(f"\n切分器 ({len(factory.list_chunkers())}):")
        for name in factory.list_chunkers():
            info = factory.chunker_info(name)
            desc = info.get("description", "")
            ptype = info.get("type", "?")
            print(f"  {name:<25} [{ptype}] {desc}")


def cmd_build(args):
    """构建知识库。"""
    from builders.orchestrator import build_all
    build_all(args.kb or DEFAULT_KB, args.chunker, args.force, args.parallel,
              pure=args.pure)


def cmd_search(args):
    """搜索。"""
    from search import search
    results = search(
        query=args.query, kb_name=args.kb or DEFAULT_KB,
        chunker_name=args.chunker,
        top_k=args.top_k, retriever=args.retriever,
    )
    if not results:
        print("无结果。")
        return
    print(f"KB: {args.kb or DEFAULT_KB}  |  Chunker: {args.chunker}  |  Query: {args.query}")
    print("-" * 65)
    for i, r in enumerate(results):
        content = r.get('content_text', '')
        try:
            print(f"  #{i+1}  score={r['score']:.4f}  source={r.get('source', '?')}")
            preview = content[:120].replace('﻿', '')
            print(f"      {preview}...\n")
        except UnicodeEncodeError:
            print(f"      <content: {len(content)} chars>\n")


def cmd_experiment(args):
    """运行实验。"""
    if args.exp_action == "recall":
        return cmd_experiment_recall(args)

    from pipeline.experiment import ExperimentConfig, ExperimentRunner

    var_map = {
        "chunker": args.chunkers,
        "retriever": args.retrievers,
        "reranker": args.rerankers,
    }
    raw = var_map[args.variable]
    if isinstance(raw, list):
        variable_values = raw
    else:
        variable_values = [v.strip() for v in raw.split(",") if v.strip()]

    def first_of(s):
        return s.split(",")[0].strip()

    config = ExperimentConfig(
        name=args.name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        kb_name=args.kb or DEFAULT_KB,
        test_set_path=args.test_set,
        output_dir=args.output or "",
        variable=args.variable,
        variable_values=variable_values,
        chunker=first_of(args.chunkers) if args.variable != "chunker" else variable_values[0],
        retriever=args.retriever if args.variable != "retriever" else variable_values[0],
        reranker=first_of(args.rerankers) if args.variable != "reranker" else variable_values[0],
        top_k=args.top_k,
        score_model=args.score_model,
        score_mode=args.score_mode,
        runs=args.runs,
    )

    runner = ExperimentRunner(config)
    runner.run()


def cmd_experiment_recall(args):
    """纯召回率实验（无 LLM 调用）。"""
    from pipeline.experiment import RecallConfig, RecallRunner

    var_map = {
        "chunker": args.chunkers,
        "retriever": args.retrievers,
        "reranker": args.rerankers,
    }
    raw = var_map[args.variable]
    if isinstance(raw, list):
        variable_values = raw
    else:
        variable_values = [v.strip() for v in raw.split(",") if v.strip()]

    def first_of(s):
        return s.split(",")[0].strip()

    top_k_values = [int(v.strip()) for v in args.top_k.split(",")]

    config = RecallConfig(
        name=args.name or f"recall_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        kb_name=args.kb or DEFAULT_KB,
        test_set_path=args.test_set,
        output_dir=args.output or "",
        variable=args.variable,
        variable_values=variable_values,
        chunker=first_of(args.chunkers) if args.variable != "chunker" else variable_values[0],
        retriever=args.retriever if args.variable != "retriever" else variable_values[0],
        reranker=first_of(args.rerankers) if args.variable != "reranker" else variable_values[0],
        top_k_values=top_k_values,
    )

    runner = RecallRunner(config)
    runner.run()


def cmd_prepare(args):
    """从 QA 测试集自动提取评分要点。"""
    from tools.prepare import enrich_test_set
    enrich_test_set(args.input, args.output, args.model, delay=0.3)


def cmd_compare(args):
    """对比两个策略。"""
    from tools.compare import run_compare
    run_compare(args.scored_xlsx, args.a, args.b, output=args.output or "", top_n=args.top)


def cmd_report_summary(args):
    """生成数据源层统一汇总报告。"""
    from pipeline.report_summary import generate_summary_report
    generate_summary_report(args.output_dir, args.title or "RAG 召回率汇总")


def cmd_budget(args):
    """查看或重置 LLM 成本账本。"""
    from core import llm_cost
    if args.reset:
        llm_cost.reset()
        print("成本账本已重置。")
        return
    print(llm_cost.report())


def cmd_mq_recall(args):
    """多查询扩展召回实验：LLM 生成 3 个相似问题 → 4 路 hybrid → RRF 融合。"""
    from pipeline.multi_query import run_multi_query_experiment
    run_multi_query_experiment(
        test_set_path=args.test_set,
        kb_name=args.kb or DEFAULT_KB,
        chunkers=[c.strip() for c in (args.chunkers or "").split(",") if c.strip()] or None,
        top_k_values=[int(v) for v in (args.top_k or "5,10").split(",")] if args.top_k else None,
        retriever=args.retriever,
        gen_model=args.gen_model,
        exp_name=args.name,
        workers=args.workers,
    )


def main():
    parser = argparse.ArgumentParser(prog="rag-pro")
    sub = parser.add_subparsers(dest="command")

    # --- strategies ---
    p = sub.add_parser("strategies", help="列出已注册策略")
    p.add_argument("--type", default=None, choices=["chunker"])

    # --- build ---
    p = sub.add_parser("build", help="构建知识库向量")
    p.add_argument("--kb", default=None)
    p.add_argument("--chunker", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--parallel", type=int, default=3, help="并行策略数（默认3）")
    p.add_argument("--pure", action="store_true", help="跳过 LLM 元数据生成，纯文本切分入库")

    # --- search ---
    p = sub.add_parser("search", help="搜索知识库")
    p.add_argument("query")
    p.add_argument("--kb", default=None)
    p.add_argument("--chunker", default="fixed_1100")
    p.add_argument("--retriever", default="hybrid", choices=["vector", "bm25", "hybrid", "hybrid_rrf"])
    p.add_argument("--top-k", type=int, default=5)

    # --- experiment ---
    p = sub.add_parser("experiment", help="运行评测实验")
    p_sub = p.add_subparsers(dest="exp_action")

    p_run = p_sub.add_parser("run", help="完整实验（检索+生成+评分+报告）")
    p_run.add_argument("--kb", default=None)
    p_run.add_argument("--test-set", required=True, help="测试集 xlsx 路径")
    p_run.add_argument("--variable", required=True,
                       choices=["chunker", "retriever", "reranker"])
    p_run.add_argument("--chunkers", default="fixed_1100")
    p_run.add_argument("--retrievers", default="hybrid")
    p_run.add_argument("--rerankers", default="direct")
    p_run.add_argument("--retriever", default="hybrid", help="固定检索器（非 retriever 维度时使用）")
    p_run.add_argument("--top-k", type=int, default=5)
    p_run.add_argument("--score-model", default="deepseek-v4-pro")
    p_run.add_argument("--score-mode", default="rubric", choices=["rubric", "strict"])
    p_run.add_argument("--runs", type=int, default=1)
    p_run.add_argument("--name", default=None)
    p_run.add_argument("--output", default=None)

    p_recall = p_sub.add_parser("recall", help="纯召回率实验（无 LLM 调用）")
    p_recall.add_argument("--kb", default=None)
    p_recall.add_argument("--test-set", required=True)
    p_recall.add_argument("--variable", required=True,
                          choices=["chunker", "retriever", "reranker"])
    p_recall.add_argument("--chunkers", default="fixed_1100")
    p_recall.add_argument("--retrievers", default="hybrid")
    p_recall.add_argument("--rerankers", default="direct")
    p_recall.add_argument("--retriever", default="hybrid", help="固定检索器（非 retriever 维度时使用）")
    p_recall.add_argument("--top-k", type=str, default="5,10", help="Top-K 值，逗号分隔（默认5,10）")
    p_recall.add_argument("--name", default=None)
    p_recall.add_argument("--output", default=None)

    # --- prepare ---
    p = sub.add_parser("prepare", help="从 QA 测试集自动提取评分要点")
    p.add_argument("input", help="输入 xlsx（列: 问题, 正确回答）")
    p.add_argument("--output", default=None, help="输出 xlsx 路径")
    p.add_argument("--model", default=None, help="LLM 模型")

    # --- compare ---
    p = sub.add_parser("compare", help="对比两个策略（从 scored.xlsx）")
    p.add_argument("scored_xlsx", help="scored.xlsx 路径")
    p.add_argument("--a", required=True, help="策略 A 名称")
    p.add_argument("--b", required=True, help="策略 B 名称")
    p.add_argument("--top", type=int, default=5, help="分差最大的题目数（默认5）")
    p.add_argument("--output", "-o", default=None, help="输出 xlsx 路径")

    # --- report-summary ---
    p = sub.add_parser("report-summary", help="生成数据源层统一汇总报告（扫描所有策略）")
    p.add_argument("output_dir", nargs="?", default="output/demo", help="输出目录（默认 output/demo）")
    p.add_argument("--title", default=None, help="报告标题")

    # --- budget ---
    p = sub.add_parser("budget", help="查看 LLM 成本账本")
    p.add_argument("--reset", action="store_true", help="重置账本（清空累计数据）")

    # --- mq-recall（多查询扩展召回实验） ---
    p = sub.add_parser("mq-recall", help="Multi-Query 召回实验：LLM 生成3个相似问题→4路hybrid→RRF融合→召回")
    p.add_argument("--test-set", required=True, help="测试集 xlsx 路径")
    p.add_argument("--kb", default=None, help="知识库（默认 demo）")
    p.add_argument("--chunkers", default=None, help="逗号分隔 chunker（默认 3 个非 meta raw_pure）")
    p.add_argument("--top-k", default=None, help="逗号分隔 topk（默认 5,10）")
    p.add_argument("--retriever", default="hybrid", choices=["vector", "bm25", "hybrid", "hybrid_rrf"])
    p.add_argument("--gen-model", default="deepseek-v4-flash", help="生成相似问题的 LLM")
    p.add_argument("--name", default="multi_query_汇总", help="实验名/输出目录名")
    p.add_argument("--workers", type=int, default=5, help="检索并发（默认5）")

    args = parser.parse_args()

    dispatch = {
        "strategies": cmd_strategies,
        "build": cmd_build,
        "search": cmd_search,
        "experiment": cmd_experiment,
        "prepare": cmd_prepare,
        "compare": cmd_compare,
        "report-summary": cmd_report_summary,
        "budget": cmd_budget,
        "mq-recall": cmd_mq_recall,
    }
    fn = dispatch.get(args.command)
    if fn:
        fn(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()