"""
[M8] 三维度指标

JD 里对 Agent 评测的要求写得很具体：从「任务完成率、工具调用准确性、用户满意度」
三个维度评估 Agent 能力。这份文件就是这三个维度。

三个维度各有各的坑，所以每个都要说清楚「怎么算」和「为什么这么算」：

一、任务完成率 —— 最容易做假的一个。
    如果只用「评测分 ≥ 4」，一个特别会说漂亮话的 Agent 也能刷高分。
    这里额外做了一层分层统计：把用例按「原始客服当时答对了还是答偏了」切开，
    分别算完成率。

    这一刀切下去，才看得出我们的 Agent 是不是真有进步：
    它在原始客服本来答对的题上应该是持平的；
    只有在原始客服答偏的那 9 条上做得更好，才叫真的改进 ——
    否则就只是「把简单的题又做对了一遍」。

二、工具调用准确性 —— 不能只看「调没调对」。
    还要看「调没调不该调的」。少调 = 该做的事没做；多调 = 白花钱，
    还可能触发不该有的副作用（比如多退了一笔款）。

    所以 precision 和 recall 必须分开报，不能捏成一个数糊过去 ——
    一个数好看，看不出它到底是漏了还是多了。

三、用户满意度 —— 这里没有真实用户，只能用评测 Agent 的打分做代理。
    **代理指标就必须叫代理指标**，在报告里写清楚。
    把代理说成真指标，是最容易被人一句话戳穿的地方。
"""

from __future__ import annotations

from src.eval.dataset import EvalCase


# ============================================================
# 一、任务完成率
# ============================================================

PASS_SCORE = 4  # 沿用 v1 量表：≥4 分算回答了核心诉求


def task_completion(cases: list[EvalCase], verdicts: dict[str, dict]) -> dict:
    """任务完成率。

    verdicts: {case_id: {"score": int, ...}}，来自评测 Agent。

    除了总完成率，还按「原始客服当时的表现」分层 —— 见文件头的说明。
    """
    judged = [(c, verdicts[c.case_id]) for c in cases if c.case_id in verdicts]
    if not judged:
        return {"total": 0, "completed": 0, "rate": 0.0, "by_human_label": {}}

    completed = sum(1 for _, v in judged if v["score"] >= PASS_SCORE)

    # 分层：原始客服答对的题 vs 原始客服答偏的题
    layers: dict[str, dict] = {}
    for label in ("答对", "答偏"):
        group = [(c, v) for c, v in judged if c.human_label == label]
        if not group:
            continue
        ok = sum(1 for _, v in group if v["score"] >= PASS_SCORE)
        layers[label] = {
            "cases": len(group),
            "completed": ok,
            "rate": round(ok / len(group), 4),
        }

    return {
        "total": len(judged),
        "completed": completed,
        "rate": round(completed / len(judged), 4),
        "by_human_label": layers,
        "by_scene": _by_scene(judged),
    }


def _by_scene(judged: list[tuple[EvalCase, dict]]) -> dict[str, dict]:
    """再按场景切一刀，看哪个业务域最弱。"""
    out: dict[str, dict] = {}
    for c, v in judged:
        bucket = out.setdefault(c.scene or "未分类", {"cases": 0, "completed": 0})
        bucket["cases"] += 1
        if v["score"] >= PASS_SCORE:
            bucket["completed"] += 1
    for bucket in out.values():
        bucket["rate"] = round(bucket["completed"] / bucket["cases"], 4)
    return out


# ============================================================
# 二、工具调用准确性
# ============================================================


# 只读工具：多调不扣分。客服场景下多查一次不是错，反而更稳妥。
READ_ONLY_TOOLS = {"search_precedents", "query_order"}


def tool_call_accuracy(case: EvalCase, actual_tools: list[str]) -> dict:
    """单条用例的工具行为评估。

    这里**刻意不用通用的 precision / recall**，因为它在这个场景里会误导人：

        用户给了订单号，Agent 除了查订单、还顺手查了一次判例库。
        按通用 precision 算，这是「调了一个不在期望里的工具」，要被扣分。
        可在客服场景里，多查一次判例恰恰是稳妥的做法，扣它没道理。

    所以按业务含义拆成三个数，各说各的事：

        required_recall       必需的工具调到了吗 —— 漏了就是该做的事没做
        forbidden_violations  禁止的工具调了吗 —— 这个必须是 0，是红线
        readonly_extra        顺带多调的只读工具 —— 记录，但不扣分
    """
    actual = set(actual_tools)
    expect = set(case.expect_tools)
    forbid = set(case.forbid_tools)

    missing = expect - actual
    forbidden_hit = actual & forbid
    extra = actual - expect - forbid

    required_recall = len(expect & actual) / len(expect) if expect else 1.0

    return {
        "case_id": case.case_id,
        "expect": sorted(expect),
        "actual": sorted(actual),
        "missing": sorted(missing),
        "forbidden_hits": sorted(forbidden_hit),
        "readonly_extra": sorted(extra & READ_ONLY_TOOLS),
        "write_extra": sorted(extra - READ_ONLY_TOOLS),
        "required_recall": round(required_recall, 4),
        "passed": not missing and not forbidden_hit,
    }


def aggregate_tool_accuracy(rows: list[dict]) -> dict:
    """把单条结果合起来。

    注意 required_recall 和 forbidden_violations 要分开报，不能合成一个数 ——
    合成之后，一个「该做的都没做、但也没乱调」的 Agent，看起来会和
    一个「该做的都做了、顺手多查了一次」的 Agent 得分差不多。那这个指标就废了。
    """
    if not rows:
        return {}
    n = len(rows)
    return {
        "cases": n,
        "required_recall": round(sum(r["required_recall"] for r in rows) / n, 4),
        "cases_all_required_hit": sum(1 for r in rows if not r["missing"]),
        "forbidden_violations": sum(len(r["forbidden_hits"]) for r in rows),
        "cases_with_missing_tool": sum(1 for r in rows if r["missing"]),
        "readonly_extra_total": sum(len(r["readonly_extra"]) for r in rows),
        "write_extra_total": sum(len(r["write_extra"]) for r in rows),
        "worst_cases": [
            r["case_id"]
            for r in sorted(rows, key=lambda r: r["required_recall"])[:5]
            if r["missing"]
        ],
    }


# ============================================================
# 三、用户满意度（代理）
# ============================================================


def satisfaction(verdicts: dict[str, dict]) -> dict:
    """用评测 Agent 的分值做满意度代理。

    一定要在报告里说明这是代理指标 —— 真实满意度得去问用户，
    我们手上没有真实用户，所以只能拿评审分当替身。
    """
    scores = [v["score"] for v in verdicts.values() if "score" in v]
    if not scores:
        return {"samples": 0}
    n = len(scores)
    return {
        "samples": n,
        "mean_score": round(sum(scores) / n, 3),
        "normalized": round(sum(scores) / n / 5, 4),
        "score_dist": {s: scores.count(s) for s in sorted(set(scores))},
        "note": "代理指标：真实满意度需向用户收集，此处用评测 Agent 的打分替代",
    }


# ============================================================
# 四、过程健康度
# ============================================================


def process_health(traces: list[dict]) -> dict:
    """过程侧的健康度。这些是 v1 完全抓不到的维度。"""
    if not traces:
        return {}
    n = len(traces)

    def has(trace: dict, name: str) -> bool:
        return name in (trace.get("tools") or [])

    skipped = [
        t["case_id"]
        for t in traces
        if t.get("stopped_reason") == "answered" and "search_precedents" not in (t.get("tools") or [])
    ]

    return {
        "cases": n,
        "avg_rounds": round(sum(t.get("rounds", 0) for t in traces) / n, 2),
        "retrieval_used": sum(1 for t in traces if has(t, "search_precedents")),
        "retrieval_skipped_cases": skipped,
        "escalated": sum(1 for t in traces if t.get("stopped_reason") == "human"),
        "llm_errors": sum(1 for t in traces if t.get("stopped_reason") in {"llm_error", "error"}),
        "avg_total_ms": int(sum(t.get("total_ms", 0) for t in traces) / n),
        "usage": _sum_usage(traces),
    }


def _sum_usage(traces: list[dict]) -> dict:
    calls = sum((t.get("usage") or {}).get("llm_calls", 0) for t in traces)
    total = sum((t.get("usage") or {}).get("total_tokens", 0) for t in traces)
    return {"llm_calls": calls, "total_tokens": total}


# ============================================================
# 五、汇总报告
# ============================================================


def build_report(
    cases: list[EvalCase],
    traces: list[dict],
    verdicts: dict[str, dict],
) -> dict:
    """把三个维度 + 过程健康度合成一份报告。"""
    by_id = {c.case_id: c for c in cases}

    tool_rows = [
        tool_call_accuracy(by_id[t["case_id"]], t.get("tools") or [])
        for t in traces
        if t["case_id"] in by_id
    ]

    return {
        "task_completion": task_completion(cases, verdicts),
        "tool_accuracy": aggregate_tool_accuracy(tool_rows),
        "satisfaction_proxy": satisfaction(verdicts),
        "process_health": process_health(traces),
        "tool_accuracy_detail": tool_rows,
    }
