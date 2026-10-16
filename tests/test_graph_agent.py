"""
LangGraph 版 Agent 的测试。

这一份测试有个额外目的：**验证两个版本真的可替换**。

    手写版和 LangGraph 版对外接口完全一致（run(question, case_id, history) → Trace），
    断言的行为也一项不差：同样的输入，同样的出口、同样的轨迹结构。
    这样换实现的时候，评测集不用改一行 —— 这正是「接口稳定」的实际价值。

    如果哪天有人改了其中一个版本的对外行为，这组测试会立刻红。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agent import graph_agent
from src.agent.graph_agent import GraphAgent
from src.agent.retrieval import build_index
from src.agent.tools import ToolRuntime
from src.llm_client import LLMError, LLMResponse

ORDER_REFUNDABLE = "20240815002"


@pytest.fixture(scope="module")
def records() -> list[dict]:
    path = Path(__file__).resolve().parent.parent / "data" / "processed" / "records.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture()
def runtime(records: list[dict]) -> ToolRuntime:
    return ToolRuntime(index=build_index(records), exclude_id=1)


def make_resp(content=None, tool_calls=None, usage=None) -> LLMResponse:
    return LLMResponse(
        content=content, tool_calls=tool_calls or [], usage=usage or {"total_tokens": 10}
    )


def install_fake_chat(monkeypatch, scripted: list[LLMResponse]):
    calls: list[dict] = []

    def fake_chat(messages, tools=None, temperature=None, model=None, force_json=False):
        calls.append({"messages": list(messages), "tools": tools})
        idx = min(len(calls) - 1, len(scripted) - 1)
        return scripted[idx]

    monkeypatch.setattr(graph_agent, "chat", fake_chat)
    return calls


# ============================================================
# 一、图和手写版行为一致
# ============================================================


def test_graph_builds(runtime):
    """图能编译出来，节点齐全。"""
    agent = GraphAgent(runtime)
    nodes = set(agent.graph.get_graph().nodes.keys())
    assert {"agent", "tools"} <= nodes


def test_answers_directly(runtime, monkeypatch):
    install_fake_chat(monkeypatch, [make_resp("能不能退取决于对方是否收款。")])
    trace = GraphAgent(runtime).run("转账转错了能退回来吗", case_id=1)

    assert trace.stopped_reason == "answered"
    assert trace.rounds == 1
    assert trace.tool_names == []


def test_tool_then_answer(runtime, monkeypatch):
    calls = install_fake_chat(
        monkeypatch,
        [
            make_resp(None, [{"id": "c1", "name": "search_precedents",
                              "arguments": {"query": "花呗退款退到哪里"}}]),
            make_resp("退回花呗额度。"),
        ],
    )
    trace = GraphAgent(runtime).run("花呗退款是退到哪里", case_id=3)

    assert trace.stopped_reason == "answered"
    assert trace.tool_names == ["search_precedents"]
    assert trace.rounds == 2

    # 和图上的消息顺序：assistant(tool_calls) 必须在 tool 结果之前
    msgs = calls[1]["messages"]
    roles = [m.get("role") for m in msgs]
    assert roles.index("assistant") < roles.index("tool")


def test_transfer_to_human_revokes_tools(runtime, monkeypatch):
    calls = install_fake_chat(
        monkeypatch,
        [
            make_resp(None, [{"id": "c1", "name": "transfer_to_human",
                              "arguments": {"reason": "超出范围", "summary": "账户被冻结"}}]),
            make_resp("已为您转接人工客服。"),
        ],
    )
    trace = GraphAgent(runtime).run("账户被冻结了怎么办", case_id=53)

    assert trace.stopped_reason == "human"
    assert calls[1]["tools"] is None, "转人工之后必须收回工具权限"


def test_max_rounds_is_enforced(runtime, monkeypatch):
    looping = make_resp(
        None, [{"id": "c1", "name": "search_precedents", "arguments": {"query": "一直查"}}]
    )
    install_fake_chat(monkeypatch, [looping])
    trace = GraphAgent(runtime, max_rounds=3).run("随便问", case_id=52)

    assert trace.rounds == 3
    assert trace.stopped_reason == "max_rounds"
    # 和手写版保持同一行为：上限轮不再执行工具
    assert len(trace.tool_calls) == 2


def test_llm_error_is_captured(runtime, monkeypatch):
    def boom(*args, **kwargs):
        raise LLMError("模拟接口故障")

    monkeypatch.setattr(graph_agent, "chat", boom)
    trace = GraphAgent(runtime).run("问点什么", case_id=55)

    assert trace.stopped_reason == "llm_error"
    assert any(s.kind == "error" for s in trace.steps)


# ============================================================
# 二、与手写版的一致性（核心）
# ============================================================


def test_guard_event_recorded(runtime, monkeypatch):
    """写操作未确认时必须被拦 —— 两个版本的行为必须一致。"""
    install_fake_chat(
        monkeypatch,
        [
            make_resp(None, [{"id": "c1", "name": "apply_refund",
                              "arguments": {"order_ref": ORDER_REFUNDABLE,
                                            "reason": "想退", "user_confirmed": False}}]),
            make_resp("请确认后再为您退款。"),
        ],
    )
    trace = GraphAgent(runtime).run("帮我退款", case_id=51)
    assert len(trace.guard_events) == 1


def test_multiturn_history_is_prepended(runtime, monkeypatch):
    calls = install_fake_chat(monkeypatch, [make_resp("好的。")])
    GraphAgent(runtime).run(
        "那要多久",
        case_id="MT-02",
        history=[
            {"role": "user", "content": "我的订单能退吗"},
            {"role": "assistant", "content": "请提供订单号"},
        ],
    )
    contents = [m.get("content") for m in calls[0]["messages"]]
    assert contents.index("我的订单能退吗") < contents.index("那要多久")


def test_trace_shape_matches_handwritten(runtime, monkeypatch):
    """两个版本产出的 Trace 结构必须一致 —— 否则评测集没法复用。"""
    install_fake_chat(
        monkeypatch,
        [
            make_resp(None, [{"id": "c1", "name": "search_precedents",
                              "arguments": {"query": "x"}}]),
            make_resp("答案。"),
        ],
    )
    trace = GraphAgent(runtime).run("问题", case_id=7)
    d = trace.to_dict()

    for key in ("case_id", "question", "final_answer", "stopped_reason",
                "rounds", "total_ms", "usage", "tools", "steps"):
        assert key in d, f"轨迹缺字段：{key}"
    assert isinstance(d["steps"], list)
    assert all("kind" in s and "round" in s for s in d["steps"])
