"""
[M7] 测试执行器

v1 的入口是「人工一条条去支付宝试，再把结果填进 Excel」—— 30 条花了很久。
v2 的入口是「把测试集丢给 Agent，让它自己跑，自动收轨迹」。

这就是「人工实测」和「自动回归」的差别：
前者一个月做一次，后者改一行 prompt 就能重跑一遍。

这个模块还负责算一批 **过程指标** —— 这是 v1 根本拿不到的东西：
    v1 只有「最终答案对不对」这一个维度。
    v2 还能回答：它调工具了吗？该查判例的时候查了吗？工具报错了它是怎么处理的？
                被安全闸门拦了几次？平均几轮收口？一共烧了多少 token？

    为什么过程指标重要：一个 Agent 可能「这次答对了，但过程完全是错的」——
    比如它没查判例、凭常识蒙对了答案。这种正确不可持续，换个问题就崩。
    只看结果指标，你发现不了这种「靠运气对」的情况。

运行方式（项目根目录）：
    python -m src.runner                # 跑全部 30 条
    python -m src.runner --limit 3      # 先跑 3 条试水（强烈建议第一次这么跑）
"""

from __future__ import annotations

import json
import re
import sys
import time

from src.agent.cs_agent import CSAgent
from src.agent.retrieval import build_index
from src.agent.tools import ToolRuntime
from src.agent.trace import Trace, save_traces
from src.config import (
    LLM_API_KEY,
    LLM_MODEL,
    OUTPUT_DIR,
    PRECEDENT_TOP_K,
    RECORDS_JSON,
    RULE_QUESTION_RE_PATTERN,
)

TRACES_JSON = OUTPUT_DIR / "traces.json"
METRICS_JSON = OUTPUT_DIR / "agent_metrics.json"

# 每条用例之间歇一下，别把接口打限流了
CASE_INTERVAL = 0.5

_RULE_Q_RE = re.compile(RULE_QUESTION_RE_PATTERN)


# ============================================================
# 一、单条执行
# ============================================================


def run_one(record: dict, index, model: str | None = None) -> Trace:
    """跑一条用例。

    关键在 exclude_id=record["id"]：
    判例库里有这条问题的「标准答法」，如果不在检索时排除掉它自己，
    Agent 就是直接把答案捞出来抄一遍 —— 评测数字会好看得离谱，而且毫无意义。
    """
    runtime = ToolRuntime(index=index, exclude_id=record["id"])
    agent = CSAgent(runtime=runtime, model=model)
    return agent.run(question=record["question"], case_id=record["id"])


# ============================================================
# 二、过程指标
# ============================================================


def classify_exit(trace: Trace, question: str) -> str:
    """给一次会话贴一个过程质量标签。

    这些标签是自动算的，不需要人工看 —— 所以一个晚上跑完 300 条，
    第二天早上能直接看到「哪一类失败在变多」。
    """
    if trace.stopped_reason in {"llm_error", "empty_response"}:
        return trace.stopped_reason
    if trace.stopped_reason == "human":
        return "escalated"
    if trace.failed_tools:
        return "had_tool_error"
    # v3：知识库和判例库都算「检索」。查了任一就不算跳过 ——
    # 之前只认 search_precedents，Agent 改用 search_knowledge 后会被全部误判成跳过检索。
    _RETRIEVAL_TOOLS = {"search_knowledge", "search_precedents"}
    if _RULE_Q_RE.search(question) and not (_RETRIEVAL_TOOLS & set(trace.tool_names)):
        # 最值得警惕的一类：该查却没查，凭常识答的
        return "skipped_retrieval"
    if trace.guard_events:
        return "guard_blocked"
    return "normal"


def compute_metrics(traces: list[Trace], records: list[dict]) -> dict:
    """汇总过程指标。"""
    n = len(traces)
    if n == 0:
        return {}

    by_id = {r["id"]: r for r in records}
    labels = [classify_exit(t, t.question) for t in traces]

    total_tools = sum(len(t.tool_calls) for t in traces)
    failed_tools = sum(len(t.failed_tools) for t in traces)
    guards = sum(len(t.guard_events) for t in traces)

    return {
        "cases": n,
        "avg_rounds": round(sum(t.rounds for t in traces) / n, 2),
        "avg_ms": int(sum(t.total_ms for t in traces) / n),
        "wall_ms": int(sum(t.total_ms for t in traces)),
        "tool_calls_total": total_tools,
        "avg_tool_calls": round(total_tools / n, 2),
        "tool_calls_failed": failed_tools,
        "tool_success_rate": (
            round(1 - failed_tools / total_tools, 4) if total_tools else 1.0
        ),
        "guard_blocks": guards,
        # v3：知识库和判例库分开统计。两者分工不同，混在一起看不出 Agent 到底在靠什么答。
        "knowledge_used": sum(1 for t in traces if "search_knowledge" in t.tool_names),
        "retrieval_used": sum(
            1
            for t in traces
            if {"search_knowledge", "search_precedents"} & set(t.tool_names)
        ),
        "precedents_used": sum(
            1 for t in traces if "search_precedents" in t.tool_names
        ),
        "escalated": sum(1 for t in traces if t.stopped_reason == "human"),
        "exit_dist": {label: labels.count(label) for label in sorted(set(labels))},
        "questions_with_reference": len(by_id),
    }


# ============================================================
# 三、入口
# ============================================================


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    print("=" * 62)
    print("[M7] 客服 Agent · 测试执行器")
    print("=" * 62)

    if not LLM_API_KEY:
        print("\n  [X] 没有读到 LLM_API_KEY")
        print("      复制 .env.example 为 .env 并填入 Key，然后重新运行。")
        return 1

    # 判例库必须用**全部** 30 条来建，不能用 --limit 截断后的那几条。
    # 否则「先跑 3 条试水」和「跑满 30 条」会得到不同的检索池，两次结果没法比。
    all_records = json.loads(RECORDS_JSON.read_text(encoding="utf-8"))
    records = all_records[:limit] if limit else all_records

    # 判例库只收「人工判定为答对」的案例 —— 从错误里学不到对的做法
    index = build_index(all_records, only_correct=True)

    print(f"\n模型：{LLM_MODEL}")
    print(f"判例库：{len(index)} 条（仅收答对样本）")
    print(f"待跑：{len(records)} 条" + ("（--limit 模式）" if limit else ""))
    print("\n开始执行 ...\n")

    traces: list[Trace] = []
    t_start = time.perf_counter()

    for i, rec in enumerate(records, 1):
        try:
            trace = run_one(rec, index)
            traces.append(trace)
            print(f"  [{i:>2}/{len(records)}] {trace.summary_line()}")
        except Exception as e:  # noqa: BLE001
            # 单条崩掉不能拖垮整轮回归 —— 记一笔，继续跑下一条
            print(f"  [{i:>2}/{len(records)}] #{rec['id']} 执行异常：{e}")
            broken = Trace(case_id=rec["id"], question=rec["question"])
            broken.stopped_reason = "error"
            traces.append(broken)

        time.sleep(CASE_INTERVAL)

    wall = time.perf_counter() - t_start
    metrics = compute_metrics(traces, records)
    metrics["wall_seconds"] = round(wall, 1)

    # 回填过程标签，方便看板按标签筛选
    for t in traces:
        t.exit_code = classify_exit(t, t.question)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_traces(traces, TRACES_JSON)
    METRICS_JSON.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "-" * 62)
    print("过程指标")
    print(f"  用例数              {metrics['cases']}")
    print(f"  平均轮数            {metrics['avg_rounds']}")
    print(f"  平均单条耗时        {metrics['avg_ms']} ms")
    print(f"  工具调用总数        {metrics['tool_calls_total']}（平均 {metrics['avg_tool_calls']} 次/条）")
    print(f"  工具成功率          {metrics['tool_success_rate']:.1%}")
    print(f"  安全闸门拦截        {metrics['guard_blocks']} 次")
    print(f"  调用了知识库        {metrics.get('knowledge_used', 0)} 条")
    print(f"  调用了判例库        {metrics.get('precedents_used', 0)} 条")
    print(f"  至少查过一次        {metrics['retrieval_used']} 条")
    print(f"  转人工              {metrics['escalated']} 条")
    print("\n  过程标签分布")
    for label, cnt in metrics["exit_dist"].items():
        print(f"    {label:<20}{cnt}")
    print("-" * 62)
    print(f"\n总耗时 {metrics['wall_seconds']} 秒")
    print(f"轨迹已写出：{TRACES_JSON.name}")
    print(f"指标已写出：{METRICS_JSON.name}")
    print("\n下一步：python -m src.eval.judge —— 让评测 Agent 读轨迹打分")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
