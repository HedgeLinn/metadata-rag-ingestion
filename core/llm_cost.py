"""LLM 成本监控（可选）。

若检测到全局 llm_cost 技能（个人开发环境），则透传其记账能力；
否则全部降级为 no-op，不影响评测流程（开源环境无此技能）。
"""
import importlib.util
from pathlib import Path

_GLOBAL_MOD = Path.home() / ".claude" / "skills" / "llm_cost" / "llm_cost.py"

try:
    _spec = importlib.util.spec_from_file_location("llm_cost_global", str(_GLOBAL_MOD))
    _impl = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_impl)

    CostLedger = _impl.CostLedger
    get_cost_callback = _impl.get_cost_callback
    cost_of = _impl.cost_of
    total_cost = _impl.total_cost
    remaining_budget = _impl.remaining_budget
    check_budget = _impl.check_budget
    require_budget = _impl.require_budget
    report = _impl.report
    reset = _impl.reset
    set_budget = _impl.set_budget
except Exception:
    def CostLedger(*args, **kwargs):
        return None

    def get_cost_callback():
        return None

    def cost_of(*args, **kwargs):
        return 0.0

    def total_cost():
        return 0.0

    def remaining_budget():
        return None

    def check_budget(*args, **kwargs):
        return True

    def require_budget(task: str = ""):
        return None

    def report():
        return "成本监控未启用：未检测到全局 llm_cost 技能。"

    def reset():
        return None

    def set_budget(*args, **kwargs):
        return None