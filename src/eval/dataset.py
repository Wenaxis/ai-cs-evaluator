"""
[M8] 评测集构建

一份能用的评测集，必须回答三个问题。少一个就没法自动跑：

    1. 考什么 ？   —— 任务本身（用户问的话，可能多轮）
    2. 怎么算对？  —— 判定标准（要点覆盖 + 人工标签）
    3. 怎么算过？  —— 预期行为（该调哪些工具、绝不该调哪些）

v1 只有第 1 和第 2。因为 v1 评的是「一段文本」，文本答没答到点上，看一眼就知道。

v2 评的是「一次执行」，所以必须补第 3 条：它该不该调工具、该调哪一个。
不把这条写清楚，评测就只能凭感觉判 —— 那绕了一圈，主观性又回来了。

两个数据来源：
    · 单轮 30 条 —— 复用 v1 的真实实测记录，人工标注直接当基准（不用重标，也不该重标）
    · 多轮若干条 —— 按 v1 人工复盘里发现的真实缺陷模式造（见 data/eval/multiturn_cases.json）

    为什么要专门造多轮用例：v1 的 30 条全是单轮，而真实客服里
    「用户补充一句前提条件」恰恰是最容易翻车的场景 —— 第 16 条那条
    「手机号注销了怎么换绑」的根因，就是它把带前提的多轮问题当成单轮处理了。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.config import BASE_DIR, RECORDS_JSON

MULTITURN_JSON = BASE_DIR / "data" / "eval" / "multiturn_cases.json"


@dataclass
class EvalCase:
    """一条可执行的评测用例。"""

    case_id: str
    scene: str
    turns: list[dict]                      # 完整对话，最后一轮是待测的那一句
    key_points: list[str] = field(default_factory=list)
    expect_tools: list[str] = field(default_factory=list)
    forbid_tools: list[str] = field(default_factory=list)
    human_label: str = ""                  # 答对 / 答偏；多轮用例由人工预先判定
    origin: str = "measured"               # measured（真实实测）| multiturn（构造）

    @property
    def question(self) -> str:
        """最后一轮用户说了什么 —— 这是 Agent 要回应的那一句。"""
        for t in reversed(self.turns):
            if t.get("role") == "user":
                return t.get("content", "")
        return ""

    @property
    def history(self) -> list[dict]:
        """最后一句之前的所有对话。"""
        turns = list(self.turns)
        for i in range(len(turns) - 1, -1, -1):
            if turns[i].get("role") == "user":
                return turns[:i]
        return []

    @property
    def numeric_id(self) -> int | None:
        """真实实测用例才有数字 id，用来和 v1 的标注对齐。"""
        try:
            return int(self.case_id)
        except (TypeError, ValueError):
            return None


# ============================================================
# 一、单轮用例（来自真实实测）
# ============================================================


def from_records(records: list[dict]) -> list[EvalCase]:
    """把 v1 的 30 条实测记录转成评测用例。

    刻意**不重新标注**：人工标签原样搬过来。这些标签是 v1 花了功夫逐条判定、
    还用 Kappa 验证过可信度的东西，是整份评测的基准。
    在 v2 里重新标一遍，等于把基准换掉，v1 和 v2 的数字就没法比了。

    expect_tools 统一要求先查判例库 —— 凡是规则类问题，检索是它该做的第一个动作。
    这本身就是一个可以量化的「过程正确性」指标。
    """
    cases: list[EvalCase] = []
    for r in records:
        cases.append(
            EvalCase(
                case_id=str(r["id"]),
                scene=r.get("scene", ""),
                turns=[{"role": "user", "content": r["question"]}],
                key_points=[r["human_note"]] if r.get("human_note") else [],
                expect_tools=["search_precedents"],
                forbid_tools=[],
                human_label=r.get("human_label", ""),
                origin="measured",
            )
        )
    return cases


# ============================================================
# 二、多轮用例（构造）
# ============================================================


def load_multiturn(path: Path = MULTITURN_JSON) -> list[EvalCase]:
    """读多轮用例。文件不存在就返回空列表 —— 不阻断整条链路。"""
    if not path.exists():
        return []

    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        EvalCase(
            case_id=c["case_id"],
            scene=c.get("scene", ""),
            turns=c["turns"],
            key_points=c.get("key_points", []),
            expect_tools=c.get("expect_tools", []),
            forbid_tools=c.get("forbid_tools", []),
            human_label=c.get("human_label", ""),
            origin="multiturn",
        )
        for c in raw
    ]


# ============================================================
# 三、组装
# ============================================================


def build_all(records_path: Path = RECORDS_JSON) -> list[EvalCase]:
    """完整评测集 = 单轮实测 30 条 + 多轮构造若干条。"""
    records = json.loads(records_path.read_text(encoding="utf-8"))
    return from_records(records) + load_multiturn()


def describe(cases: list[EvalCase]) -> dict:
    """评测集的元信息。报告里要写清楚这套评测集是怎么来的。"""
    by_origin: dict[str, int] = {}
    by_scene: dict[str, int] = {}
    for c in cases:
        by_origin[c.origin] = by_origin.get(c.origin, 0) + 1
        by_scene[c.scene] = by_scene.get(c.scene, 0) + 1
    return {
        "total": len(cases),
        "by_origin": by_origin,
        "by_scene": by_scene,
        "with_tool_expectation": sum(1 for c in cases if c.expect_tools),
        "forbidden_tool_cases": sum(1 for c in cases if c.forbid_tools),
    }
