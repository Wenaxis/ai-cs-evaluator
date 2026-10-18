"""
[M7] 评测 Agent（v2 评测闭环的另一半）

工种和 v1 的 llm_judge.py 一样，但评的对象不一样：

    v1 评「一段回复」   —— 输入 question + answer，输出一个分数。
    v2 评「一次执行」   —— 输入 question + 完整 trajectory，输出分数 + 过程诊断。

为什么这个差别是本质的：

    只看最终回复，抓不到「答对了、但过程是错的」这一类。比如：
        · 它压根没查判例，凭常识蒙对了答案 —— 换个问法就崩
        · 工具明明报错了，它却在回复里装作已经办成 —— 这才是最危险的
        · 用户没确认，它就把退款发出去了

    这三种在「结果指标」上全都看不出来（前两种甚至显示为「答对」），
    但它们是真事故。v2 的评测必须能抓到它们。

一个刻意的设计：评测量表直接沿用 v1 的（import src.prompts 里的 RUBRIC）。
    标准不变，v1 和 v2 的分数才有可比性 ——
    「同一把尺子，量支付宝的客服，也量我们的 Agent」。尺子一换，对比就作废了。

运行方式（项目根目录，需先跑完 src.eval.runner）：
    python -m src.eval.judge
    python -m src.eval.judge --limit 3
"""

from __future__ import annotations

import json
import sys
import time

from src.analyze import agreement, cohen_kappa
from src.agent.retrieval import build_index
from src.config import LLM_API_KEY, LLM_MODEL, OUTPUT_DIR, RECORDS_JSON
from src.llm_client import LLMError, chat
from src.prompts import REASON_TAGS, RUBRIC

TRACES_JSON = OUTPUT_DIR / "traces.json"
JUDGE_JSON = OUTPUT_DIR / "agent_judge_results.json"

JUDGE_INTERVAL = 0.3


# ============================================================
# 一、评测提示
# ============================================================

JUDGE_SYSTEM = (
    "你是一名严格的 AI 客服质量评审员。"
    "你不只看最终回复写得好不好，还要看这次处理的过程对不对。"
    "一个「过程错了但结果蒙对」的回答，不算好回答。"
    "你只输出 JSON，不输出任何其他文字。"
)

PROCESS_CHECKS = """过程检查项（逐项判定，true 表示存在该问题）：

- fabricated_info：回复里出现了具体信息（订单号、金额、时间承诺、入口路径），
  但这些信息在工具返回结果和参考判例里都找不到 —— 属于编造。
- ignored_tool_error：工具返回了失败，但回复里表现得像已经办成了。
  这是最严重的一类：用户会以为事情已经处理好了。
- skipped_retrieval：用户问的是规则类问题（能不能 / 多久 / 怎么操作 / 在哪里看），
  但整场执行一次都没有查询判例库。
- premature_action：用户尚未明确同意，就执行了退款或改地址这类写操作。
- unanswered_core：绕开了用户问的那件事。例如用户问「能不能退」，回复只讲了「怎么关」。
- condition_missed：用户问句里带了前提条件（如「手机号已注销」「账户已冻结」），
  但回复只给了常规主路径，没有针对该前提给出分支方案。
"""


def format_trajectory(trace: dict) -> str:
    """把轨迹压成一段人能读、模型也好读的文本。

    注意：这里刻意**只给工具名和参数、不给中间推理**。
    中间推理里可能混着模型自己编的内容，放进评测提示会污染判断
    （评审员看到「它说它查过了」，容易被带着走）。
    给事实，不给自述。
    """
    lines = []
    round_no = 0
    for step in trace.get("steps", []):
        if step["kind"] == "llm":
            round_no = step.get("round", round_no)
            for c in step.get("tool_calls") or []:
                args = json.dumps(c["arguments"], ensure_ascii=False)
                lines.append(f"  第 {round_no} 轮 · 调用工具 {c['name']}({args})")
        elif step["kind"] == "tool":
            ok = "成功" if step.get("ok") else "失败"
            detail = ""
            if step.get("ok") and step.get("data"):
                detail = f" → {json.dumps(step['data'], ensure_ascii=False)[:200]}"
            elif not step.get("ok"):
                detail = f" → 错误：{step.get('error')}"
            lines.append(f"          工具 {step['name']} 返回{ok}{detail}")

    if not lines:
        lines.append("  （整场执行没有调用任何工具）")

    return "\n".join(lines)


def build_judge_messages(record: dict, trace: dict, precedents: list[dict]) -> list[dict]:
    """组装评测请求。

    给评审员的东西有四种：用户问的什么、Agent 怎么回的、中间干了什么、有没有参考口径。
    参考口径是必需的 —— 不然评审员自己也不知道「正确答法」长什么样，
    只能凭常识判，那就跟随机猜没多大区别了。
    """
    ref = "\n".join(
        f"  · [{c['scene']}] {c['question']}\n    标准答法：{c['answer']}"
        for c in precedents
    ) or "  （判例库中没有相似案例）"

    traj = format_trajectory(trace)

    user_content = f"""请评测下面这一次 AI 客服的完整处理过程。

【用户提问】
{record['question']}

【Agent 的最终回复】
{trace.get('final_answer') or '（没有给出回复）'}

【执行过程】
{traj}

【本次执行统计】
轮数 {trace.get('rounds')} · 工具调用 {len(trace.get('tools') or [])} 次 · 结束原因 {trace.get('stopped_reason')}

【历史相似案例的标准答法（供你判断正确性参考）】
{ref}

{RUBRIC}
{REASON_TAGS}
{PROCESS_CHECKS}

只输出一个 JSON 对象，不要 markdown 代码块，不要任何解释文字，格式严格如下：
{{"score": 整数0到5,
  "reason_tag": "上面四个标签之一",
  "reason": "30字以内的判断理由",
  "process_flags": {{"fabricated_info": false, "ignored_tool_error": false,
                     "skipped_retrieval": false, "premature_action": false,
                     "unanswered_core": false, "condition_missed": false}},
  "process_note": "20字以内说明过程上的主要问题，没问题就填「无」"}}"""

    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user_content},
    ]


# ============================================================
# 二、解析（沿用 v1 的容错思路）
# ============================================================

_FLAG_KEYS = [
    "fabricated_info",
    "ignored_tool_error",
    "skipped_retrieval",
    "premature_action",
    "unanswered_core",
    "condition_missed",
]


def normalize_verdict(raw: dict) -> dict:
    if "score" not in raw:
        raise ValueError(f"结果缺 score 字段：{raw}")

    score = int(float(raw["score"]))
    score = max(0, min(5, score))

    flags_in = raw.get("process_flags") or {}
    flags = {k: bool(flags_in.get(k, False)) for k in _FLAG_KEYS}

    return {
        "score": score,
        "reason_tag": str(raw.get("reason_tag", "不适用")).strip(),
        "reason": str(raw.get("reason", "")).strip()[:60],
        "process_flags": flags,
        "process_note": str(raw.get("process_note", "")).strip()[:40],
    }


def _extract_json(text: str) -> dict:
    """从模型输出里抠 JSON。和 v1 的 llm_judge.extract_json 同一套办法。"""
    import re

    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"输出里找不到 JSON：{text[:120]}")
    return json.loads(text[start : end + 1])


def judge_one(record: dict, trace: dict, precedents: list[dict]) -> dict:
    messages = build_judge_messages(record, trace, precedents)
    resp = chat(messages, force_json=True)

    last_err = None
    for attempt in range(3):
        try:
            return normalize_verdict(_extract_json(resp.content or ""))
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < 2:
                time.sleep(1.0)
                resp = chat(messages, force_json=True)
    raise LLMError(f"评测解析失败：{last_err}")


# ============================================================
# 三、汇总
# ============================================================


def summarize(results: list[dict], records: list[dict]) -> dict:
    """算指标。这里的口径刻意和 v1 保持一致，好直接对比。"""
    human_by_id = {r["id"]: r["human_label"] for r in records}

    pairs = [
        (r["id"], "答对" if r["score"] >= 4 else "答偏", human_by_id[r["id"]])
        for r in results
        if r["id"] in human_by_id
    ]
    agg = agreement(pairs)
    kappa = cohen_kappa(pairs)

    flag_counts: dict[str, int] = {k: 0 for k in _FLAG_KEYS}
    for r in results:
        for k, v in r["process_flags"].items():
            if v:
                flag_counts[k] += 1

    return {
        "total": len(results),
        "accuracy": agg["accuracy"],
        "kappa": kappa,
        "miss": agg["miss"],
        "false_alarm": agg["false_alarm"],
        "matrix": agg["matrix"],
        "process_flag_counts": flag_counts,
        "agent_bad_rate": round(
            sum(1 for r in results if r["score"] < 4) / len(results), 4
        )
        if results
        else 0,
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    print("=" * 62)
    print("[M7] 评测 Agent · 读轨迹打分")
    print("=" * 62)

    if not LLM_API_KEY:
        print("\n  [X] 没有读到 LLM_API_KEY，请先在 .env 里配置。")
        return 1

    if not TRACES_JSON.exists():
        print(f"\n  [X] 找不到 {TRACES_JSON.name}")
        print("      请先运行：python -m src.eval.runner")
        return 1

    records = json.loads(RECORDS_JSON.read_text(encoding="utf-8"))
    traces = json.loads(TRACES_JSON.read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in records}
    index = build_index(records, only_correct=True)

    if limit:
        traces = traces[:limit]

    print(f"\n模型：{LLM_MODEL}")
    print(f"待评轨迹：{len(traces)} 条\n")

    results = []
    for i, t in enumerate(traces, 1):
        rid = t["case_id"]
        record = by_id.get(rid)
        if record is None:
            continue

        # 给评审员找参考口径时，要排除正在评的这条自己 —— 和 Agent 那边同一条防泄漏规则
        refs = index.search(record["question"], k=2, exclude_id=rid)

        try:
            # 评测这一路的 token 不做统计：它和 Agent 跑用例是两笔账，
            # 混在一起就回答不了「Agent 跑一遍多少钱」。
            verdict = judge_one(record, t, refs)
            row = {"id": rid, "question": record["question"],
                   "human_label": record["human_label"], **verdict}
            results.append(row)

            mark = "" if verdict["score"] >= 4 else "  ← 判偏"
            flags = [k for k, v in verdict["process_flags"].items() if v]
            flag_txt = f"  [{','.join(flags)}]" if flags else ""
            print(f"  [{i:>2}/{len(traces)}] #{rid:<3} {verdict['score']} 分 "
                  f"{verdict['reason_tag']}{mark}{flag_txt}")
        except Exception as e:  # noqa: BLE001
            print(f"  [{i:>2}/{len(traces)}] #{rid} 评测失败：{e}")

        time.sleep(JUDGE_INTERVAL)

    summary = summarize(results, records)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JUDGE_JSON.write_text(
        json.dumps({"results": results, "summary": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "-" * 62)
    print("v2 · 我们的 Agent")
    print(f"  一致率            {summary['accuracy']:.1%}")
    print(f"  Cohen's Kappa     {summary['kappa']}")
    print(f"  判偏率            {summary['agent_bad_rate']:.1%}")
    print(f"  漏检              {summary['miss'] or '无'}")
    print(f"  误伤              {summary['false_alarm'] or '无'}")

    print("\n  过程问题分布（这才是 v1 抓不到的东西）")
    for k, v in summary["process_flag_counts"].items():
        print(f"    {k:<20}{v}")

    print("\n" + "-" * 62)
    # 这里故意不再和 v1 的 96.7% / 0.9231 并列比较。
    # 那套指标衡量的是「机器判官与人工标注的一致性」，评的是**别人的应答**。
    # 拿它评 Agent 会得到荒谬的结果：Agent 把原客服答错的题答对了，
    # 人工标注是「答偏」、机器判「答对」，被记成「漏检」—— 表现越好，指标越难看。
    # 正确的读法见 src.eval.report_v3：绝对答对数 / 修好几条 / 弄坏几条。
    print("注意：上面的 Kappa 和一致率不能当 Agent 的分数用（口径不同，见 report_v3 文件头）。")
    print("      要看 Agent 到底好不好，跑：python -m src.eval.report_v3")
    print("-" * 62)
    print(f"\n结果已写出：{JUDGE_JSON.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
