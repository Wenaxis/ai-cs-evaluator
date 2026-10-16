"""
离线冒烟测试：用假的模型跑完整条链路，不花一分钱 API 额度。

为什么需要这个东西：

    配好 Key 之前，你没法知道「代码到底通不通」—— 是环境问题、Key 问题，
    还是代码本身写错了？三种情况在终端里长得差不多。

    这个脚本把模型换成一个确定性的假货，于是整条链路（测试执行器 → 轨迹落盘 →
    过程指标 → 评测 Agent → 对比基线）可以在不联网的情况下跑完。
    跑通了，说明你的代码和环境没问题，剩下唯一的不确定因素就是 Key 和模型。
    跑不通，报错信息一定指向代码。

    另外，它产出的 traces.json 是真实格式的 —— 你可以拿它先把看板调通，
    不用等 30 条真跑完。

运行方式（项目根目录）：
    python scripts/mock_run.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval import judge, runner  # noqa: E402
from src.agent import cs_agent  # noqa: E402
from src.agent.retrieval import build_index  # noqa: E402
from src.agent.trace import save_traces  # noqa: E402
from src.config import OUTPUT_DIR, RECORDS_JSON  # noqa: E402
from src.llm_client import LLMResponse  # noqa: E402

# 假模型的行为：先查一次判例，然后照参考案例说一段话。
# 刻意做得「像那么回事」，这样产出的轨迹格式和真跑的一模一样。


def fake_agent_chat(messages, tools=None, temperature=None, model=None, force_json=False):
    has_tool_result = any(m.get("role") == "tool" for m in messages)

    if tools and not has_tool_result:
        question = ""
        for m in messages:
            if m.get("role") == "user":
                question = m.get("content", "")
        return LLMResponse(
            content=None,
            tool_calls=[
                {"id": "mock_call_1", "name": "search_precedents",
                 "arguments": {"query": question}}
            ],
            usage={"prompt_tokens": 320, "completion_tokens": 24, "total_tokens": 344},
        )

    # 第二遍：假装读到了判例，给一段有结论的回复
    return LLMResponse(
        content="先说结论：这取决于你的具体情况。\n"
                "常规情形下可按以下路径处理：在我的 → 设置 → 客服中心提交申请，"
                "一般 1-3 个工作日会有结果。若涉及账户异常，建议直接走人工核验。",
        tool_calls=[],
        usage={"prompt_tokens": 480, "completion_tokens": 96, "total_tokens": 576},
    )


def fake_judge_chat(messages, tools=None, temperature=None, model=None, force_json=False):
    return LLMResponse(
        content=json.dumps(
            {
                "score": 4,
                "reason_tag": "不适用",
                "reason": "给了结论与路径，mock 评测",
                "process_flags": {
                    "fabricated_info": False,
                    "ignored_tool_error": False,
                    "skipped_retrieval": False,
                    "premature_action": False,
                    "unanswered_core": False,
                    "condition_missed": False,
                },
                "process_note": "无",
            },
            ensure_ascii=False,
        ),
        tool_calls=[],
        usage={"prompt_tokens": 900, "completion_tokens": 120, "total_tokens": 1020},
    )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 62)
    print("离线冒烟测试（假模型，不消耗 API 额度）")
    print("=" * 62)

    cs_agent.chat = fake_agent_chat        # 替换 Agent 看到的模型
    judge.chat = fake_judge_chat           # 替换评测 Agent 看到的模型

    records = json.loads(RECORDS_JSON.read_text(encoding="utf-8"))
    index = build_index(records, only_correct=True)
    print(f"\n判例库 {len(index)} 条 · 用例 {len(records)} 条\n")

    traces = []
    for rec in records:
        trace = runner.run_one(rec, index)
        trace.exit_code = runner.classify_exit(trace, trace.question)
        traces.append(trace)
        if rec["id"] <= 4:
            print(f"  {trace.summary_line()}")
    print(f"  ...（其余 {len(records) - 4} 条略）")

    metrics = runner.compute_metrics(traces, records)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_traces(traces, OUTPUT_DIR / "traces.json")
    (OUTPUT_DIR / "agent_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "-" * 62)
    print("链路自检")
    print(f"  轨迹条数            {len(traces)}")
    print(f"  平均轮数            {metrics['avg_rounds']}")
    print(f"  工具调用总数        {metrics['tool_calls_total']}")
    print(f"  工具成功率          {metrics['tool_success_rate']:.1%}")
    print(f"  调用了判例检索      {metrics['retrieval_used']} 条")
    print(f"  过程标签分布        {metrics['exit_dist']}")
    print("-" * 62)

    # 顺手验证评测 Agent 能读得懂自己产出的轨迹
    verdict = judge.normalize_verdict(
        json.loads(judge.chat([]).content)
    )
    print(f"  评测 Agent 解析自检  score={verdict['score']} "
          f"flags={sum(verdict['process_flags'].values())} 个")

    print("\n[OK] 整条链路可跑通。")
    print("     下一步：把真实 Key 填进 .env，然后跑 python -m src.eval.runner")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
