"""
看回答原文。分数是压缩过的，压缩会丢信息。

为什么需要这个东西：

    `agent_quality_report.json` 里每条只有一个 score 和一句 reason。
    「5 分」和「1 分」之间差的是什么，看分数永远看不出来。

    真正能判断「这个 Agent 到底行不行」的，是把它答的那段话读一遍 ——
    它有没有给出可执行路径、有没有说时限、有没有接住用户问句里的前提条件。
    这三样东西，任何量表都只能间接反映。

所以这个脚本只做一件事：把某几条的完整回答原样打到终端上，不做任何加工。

用法（项目根目录，先跑过 runner 和 judge）：

    .venv/Scripts/python.exe scripts/show_answer.py --id 7
    .venv/Scripts/python.exe scripts/show_answer.py --id 7,16,25
    .venv/Scripts/python.exe scripts/show_answer.py --fail      # 只看 Agent 答偏的
    .venv/Scripts/python.exe scripts/show_answer.py --diff      # 只看「修好/弄坏」的
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import OUTPUT_DIR  # noqa: E402

TRACES = OUTPUT_DIR / "traces.json"
REPORT = OUTPUT_DIR / "agent_quality_report.json"

W = 68


def _load() -> tuple[list[dict], dict[int, dict]]:
    if not TRACES.exists():
        raise SystemExit(f"找不到 {TRACES}\n先跑：.venv/Scripts/python.exe -m src.eval.runner")
    if not REPORT.exists():
        raise SystemExit(
            f"找不到 {REPORT}\n先跑：.venv/Scripts/python.exe -m src.eval.judge"
            " 然后 -m src.eval.report_v3"
        )
    traces = json.loads(TRACES.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    return traces, {r["id"]: r for r in report.get("rows", [])}


def _tools_of(trace: dict) -> str:
    """把工具调用链压成一行，失败的标 ❌ —— 一眼看出它查没查、失败没失败。"""
    parts = []
    for s in trace.get("steps", []):
        if s.get("kind") != "tool":
            continue
        mark = "" if s.get("ok") else " ❌"
        arg = s.get("arguments") or {}
        q = arg.get("query") or ""
        parts.append(f"{s.get('name')}{mark}" + (f"({q})" if q else ""))
    return " → ".join(parts) or "（没调工具）"


def show(trace: dict, row: dict | None) -> None:
    cid = trace.get("case_id")
    question = trace.get("question", "")

    print("=" * W)
    print(f"#{cid}  {question}")
    print("-" * W)

    if row:
        human = "答对" if row.get("human_pass") else "答偏"
        agent = "答对" if row.get("agent_pass") else "答偏"
        verdict = "✅ 修好了" if (not row.get("human_pass") and row.get("agent_pass")) else ""
        verdict = "⚠️ 弄坏了" if (row.get("human_pass") and not row.get("agent_pass")) else verdict
        print(f"原客服：{human}   Agent：{agent}   {verdict}")
        print(f"打分：{row.get('score')}    过程标签：{', '.join(row.get('flags') or []) or '无'}")
        print(f"判词：{row.get('reason', '')}")
    else:
        print("（这条没有判词 —— quality_report 里没有它）")

    print(f"退出：{trace.get('stopped_reason', '?')}   轮数：{trace.get('rounds', '?')}")
    print(f"工具：{_tools_of(trace)}")
    print("-" * W)
    print((trace.get("final_answer") or "（空回答）").strip())
    print()


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

    traces, rows = _load()
    by_id = {t["case_id"]: t for t in traces}

    if "--fail" in argv:
        ids = [i for i, r in rows.items() if not r.get("agent_pass")]
        ids.sort()
    elif "--diff" in argv:
        ids = [
            i
            for i, r in rows.items()
            if r.get("human_pass") != r.get("agent_pass")
        ]
        ids.sort()
    elif "--id" in argv:
        raw = argv[argv.index("--id") + 1] if len(argv) > argv.index("--id") + 1 else ""
        try:
            ids = [int(x) for x in raw.replace("，", ",").split(",") if x.strip()]
        except ValueError:
            raise SystemExit("--id 要写成 7 或 7,16,25")
    else:
        print(__doc__)
        return 0

    missing = [i for i in ids if i not in by_id]
    if missing:
        print(f"traces.json 里没有这几条：{missing}\n可用的 id：{sorted(by_id)}")
        return 1

    print(f"共 {len(traces)} 条轨迹，展示 {len(ids)} 条\n")
    for i in ids:
        show(by_id[i], rows.get(i))

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
