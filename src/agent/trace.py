"""
[M7] 执行轨迹（Agent 的黑匣子）

为什么要有这个东西 —— 这是 v2 相对 v1 最实在的一个升级：

    v1 能拿到的只有「最终回复」。一条回复答得不对，你只知道它不对，
    不知道为什么不对。而 Agent 的失败恰恰大多发生在中间环节：
    该调工具时没调、调错了工具、工具报错了却装作成功、第一轮就急着下结论。

    只评最终回复，等于只看体温计不看化验单。

所以 trace 记录的是**完整 trajectory**：
    每一步的模型输入输出、每一次工具调用的参数与返回、每一轮的 token 与耗时。

这份轨迹有两个消费者：
    1. 评测 Agent —— 它读轨迹来判「过程对不对」，而不只是「结果对不对」
    2. 你自己 —— 出问题时，翻轨迹比翻日志快得多
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Step:
    """一步动作。可能是「模型说话」「模型决定调工具」，也可能是「工具返回」。"""

    kind: str  # llm | tool | stop | guard
    round: int
    detail: dict = field(default_factory=dict)
    elapsed_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "round": self.round,
            "elapsed_ms": self.elapsed_ms,
            **self.detail,
        }


@dataclass
class Trace:
    """一次完整会话的轨迹。"""

    case_id: int
    question: str
    steps: list[Step] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    # 外部回填的汇总数据
    final_answer: str = ""
    stopped_reason: str = ""  # answered | human | max_rounds | error
    exit_code: str = ""       # normal | tool_error | no_tool | guard_blocked
    usage: dict = field(default_factory=dict)

    # ---------- 写入 ----------

    def add_llm(self, round_no: int, content: str | None, tool_calls: list[dict],
                usage: dict, elapsed_ms: int) -> None:
        self.steps.append(
            Step(
                kind="llm",
                round=round_no,
                elapsed_ms=elapsed_ms,
                detail={
                    "content": content,
                    "tool_calls": [
                        {"name": c["name"], "arguments": c["arguments"]}
                        for c in tool_calls
                    ],
                    "usage": usage,
                },
            )
        )

    def add_tool(self, round_no: int, name: str, args: dict, result: dict,
                 elapsed_ms: int) -> None:
        self.steps.append(
            Step(
                kind="tool",
                round=round_no,
                elapsed_ms=elapsed_ms,
                detail={
                    "name": name,
                    "arguments": args,
                    "ok": result.get("ok", False),
                    "data": result.get("data"),
                    "error": result.get("error"),
                },
            )
        )

    def add_mark(self, kind: str, round_no: int, **detail) -> None:
        self.steps.append(Step(kind=kind, round=round_no, detail=detail))

    # ---------- 读取 ----------

    @property
    def tool_calls(self) -> list[dict]:
        return [s.detail for s in self.steps if s.kind == "tool"]

    @property
    def tool_names(self) -> list[str]:
        return [t["name"] for t in self.tool_calls]

    @property
    def failed_tools(self) -> list[dict]:
        return [t for t in self.tool_calls if not t.get("ok")]

    @property
    def guard_events(self) -> list[dict]:
        """被确认闸门拦下来的次数。这个数字本身就是一个安全指标。"""
        return [
            s.detail
            for s in self.steps
            if s.kind == "tool" and "尚未确认" in str(s.detail.get("error") or "")
        ]

    @property
    def total_ms(self) -> int:
        return sum(s.elapsed_ms for s in self.steps)

    @property
    def rounds(self) -> int:
        return max((s.round for s in self.steps), default=0)

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "question": self.question,
            "final_answer": self.final_answer,
            "stopped_reason": self.stopped_reason,
            "exit_code": self.exit_code,
            "rounds": self.rounds,
            "total_ms": self.total_ms,
            "usage": self.usage,
            "tools": self.tool_names,
            "steps": [s.to_dict() for s in self.steps],
        }

    def summary_line(self) -> str:
        """终端里一行看完这次会话干了些啥。"""
        tools = " → ".join(self.tool_names) or "（没调工具）"
        return (
            f"#{self.case_id:<3} {self.rounds} 轮 {self.total_ms:>6}ms  "
            f"{self.stopped_reason:<10} {tools}"
        )


def save_traces(traces: list[Trace], path: Path) -> None:
    """批量落盘。评测 Agent 读的就是这个文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([t.to_dict() for t in traces], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
