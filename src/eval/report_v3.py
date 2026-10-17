"""
[M8+] Agent 应答质量报告（v3 新增）

为什么需要这个文件？

v2 用「一致率 + Kappa」评价 Agent，直接沿用了 v1 的指标。
但 v1 的一致率衡量的是「评测工具与人工标注的一致性」——
它回答的是「机器判官准不准」。

拿它去评价 Agent 会得到一个荒谬的结果：
    Agent 把原客服答错的题答对了，人工标注是「答偏」，机器判「答对」，
    于是被记成「漏检」—— 表现越好，指标越难看。

v3 的 Agent 接入了业务知识库，实测把原客服答偏的 9 条全部答对，
一致率却掉到 66.7%，漏检 9 条。这不是 Agent 的问题，是尺子拿错了。

所以这个模块换一把尺子，直接回答三个问题：

    1. Agent 自己答对了多少条？（绝对质量，不和原客服对齐）
    2. 原客服答偏的那几条，Agent 修好了几条？
    3. 原客服答对的，Agent 有没有弄坏？

第 2 条和第 3 条才是「有没有变好」的真正证据。

一个必须写在明处的局限：
    知识库和评测集同源（30 个主题对应 30 道题）。这是开卷评测 ——
    考的是「能不能检索到并对上号」，不是「记不记得住」。
    真实产品里知识库本来就该独立于用户提问，这里因为数据只有这一份，
    只能做到「形式上独立、来源上同源」。面试被问到要主动说明。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from src.config import OUTPUT_DIR, RECORDS_JSON

JUDGE_JSON = OUTPUT_DIR / "agent_judge_results.json"
METRICS_JSON = OUTPUT_DIR / "agent_metrics.json"
TRACES_JSON = OUTPUT_DIR / "traces.json"
REPORT_JSON = OUTPUT_DIR / "agent_quality_report.json"

PASS_THRESHOLD = 4  # 量表 5 分制，≥4 视为答对（与前序口径一致）


def _load(path: Path) -> dict | list:
    if not path.exists():
        raise SystemExit(
            f"找不到 {path.name}。请先跑：\n"
            "  python -m src.eval.runner\n"
            "  python -m src.eval.judge"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def build_report() -> dict:
    judge = _load(JUDGE_JSON)
    results = judge["results"] if isinstance(judge, dict) else judge
    traces = {t["case_id"]: t for t in _load(TRACES_JSON)}
    metrics = _load(METRICS_JSON)
    rows: list[dict] = []
    for r in results:
        agent_pass = int(r.get("score", 0)) >= PASS_THRESHOLD
        human_pass = r.get("human_label") == "答对"
        rows.append(
            {
                "id": r["id"],
                "question": r.get("question", ""),
                "human_pass": human_pass,
                "agent_pass": agent_pass,
                "score": r.get("score"),
                "flags": [k for k, v in (r.get("process_flags") or {}).items() if v],
                "reason": r.get("reason", ""),
            }
        )

    n = len(rows)
    agent_pass_n = sum(1 for r in rows if r["agent_pass"])
    human_pass_n = sum(1 for r in rows if r["human_pass"])

    fixed = [r["id"] for r in rows if not r["human_pass"] and r["agent_pass"]]
    broken = [r["id"] for r in rows if r["human_pass"] and not r["agent_pass"]]
    both_ok = [r["id"] for r in rows if r["human_pass"] and r["agent_pass"]]
    both_bad = [r["id"] for r in rows if not r["human_pass"] and not r["agent_pass"]]

    fabricated = [r["id"] for r in rows if "fabricated_info" in r["flags"]]
    unanswered = [r["id"] for r in rows if "unanswered_core" in r["flags"]]

    # 知识库使用情况直接从 runner 写出的 metrics 取。
    # 不从 traces 里算，是因为 tool_names 是 Trace 的 @property，
    # 落盘成 JSON 时不会保留 —— 读回来再判断一定是 0。
    used_kb = metrics.get("knowledge_used", 0)
    used_precedent = metrics.get("precedents_used", 0)

    return {
        "cases": n,
        "agent": {
            "pass": agent_pass_n,
            "pass_rate": round(agent_pass_n / n, 4) if n else 0.0,
            "fabricated": len(fabricated),
            "fabricated_ids": fabricated,
            "unanswered": len(unanswered),
            "unanswered_ids": unanswered,
        },
        "baseline_customer_service": {
            "pass": human_pass_n,
            "pass_rate": round(human_pass_n / n, 4) if n else 0.0,
            "label_note": "来自人工标注，衡量的是支付宝原客服在实测中的表现",
        },
        "comparison": {
            "fixed_ids": fixed,
            "fixed_count": len(fixed),
            "broken_ids": broken,
            "broken_count": len(broken),
            "both_pass": len(both_ok),
            "both_fail": len(both_bad),
            "delta_pass_rate": (
                round((agent_pass_n - human_pass_n) / n, 4) if n else 0.0
            ),
        },
        "process": {
            "escalated": metrics.get("escalated", 0),
            "knowledge_used": used_kb,
            "knowledge_use_rate": round(used_kb / n, 4) if n else 0.0,
            "precedents_used": used_precedent,
            "avg_rounds": metrics.get("avg_rounds"),
            "tool_success_rate": metrics.get("tool_success_rate"),
        },
        "rows": rows,
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    rep = build_report()

    a = rep["agent"]
    b = rep["baseline_customer_service"]
    c = rep["comparison"]
    p = rep["process"]

    print("=" * 64)
    print("[M8+] Agent 应答质量报告（不与原客服对齐，看绝对质量）")
    print("=" * 64)

    print(f"\n用例数                {rep['cases']}")
    print("\n应答质量")
    print(f"  Agent 答对          {a['pass']} 条（{a['pass_rate']:.1%}）")
    print(f"  原客服答对          {b['pass']} 条（{b['pass_rate']:.1%}）")
    print(f"  差值                {c['delta_pass_rate'] * 100:+.1f} 个百分点")

    print("\n逐条对比")
    print(f"  原答偏 → Agent 答对（修好的）   {c['fixed_count']} 条 {c['fixed_ids']}")
    print(f"  原答对 → Agent 答偏（弄坏的）   {c['broken_count']} 条 {c['broken_ids']}")
    print(f"  两边都答对                      {c['both_pass']} 条")
    print(f"  两边都没答对                    {c['both_fail']} 条 {[r['id'] for r in rep['rows'] if not r['human_pass'] and not r['agent_pass']]}")

    print("\n质量风险")
    print(f"  编造信息            {a['fabricated']} 条 {a['fabricated_ids'] or ''}")
    print(f"  未答核心            {a['unanswered']} 条 {a['unanswered_ids'] or ''}")
    print(f"  转人工              {p['escalated']} 条")

    print("\n过程")
    print(f"  知识库使用率        {p['knowledge_use_rate']:.1%}（{p['knowledge_used']} 条）")
    print(f"  判例库使用          {p['precedents_used']} 条")
    print(f"  平均轮数            {p['avg_rounds']}")
    print(f"  工具成功率          {(p['tool_success_rate'] or 0) * 100:.1f}%")

    print("-" * 64)
    print("怎么读这张表：")
    print("  「修好的」是 v3 的核心成果 —— 原客服答偏的题，Agent 答对了。")
    print("  「弄坏的」是代价 —— 原客服答对的题，Agent 有没有搞砸。")
    print("  一致率在这里没有意义：它衡量的是机器与人工标注的吻合度，")
    print("  Agent 比原客服强，反而会被记成「漏检」。")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n报告已写出：{REPORT_JSON.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
