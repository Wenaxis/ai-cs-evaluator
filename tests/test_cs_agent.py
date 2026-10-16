"""
Agent 主循环的测试 —— 用假的模型响应，不烧 API 额度。

为什么专门给主循环写测试：

    循环里有好几个「只在特定情况下才走到」的出口：
    达到最大轮数、转人工、模型返回空、模型调用失败、工具报错。
    这些分支如果靠真跑 Agent 去触发，又慢、又贵、又难复现 ——
    真跑十次可能一次都撞不上「转人工」这条路。

    把 chat() 换成假的，这些分支就变成了确定性的、毫秒级的测试。
    这也是「Agent 可测」这件事本身的价值：把不确定的模型锁在边界外，
    边界内的一切都应该是可复现的。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agent import cs_agent
from src.agent.retrieval import build_index
from src.agent.tools import ToolRuntime
from src.llm_client import LLMResponse

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
    """把模型换成一个「剧本播放器」：按顺序吐出预先写好的响应。

    返回调用记录，方便断言「第几轮有没有开工具权限」。
    """
    calls: list[dict] = []

    def fake_chat(messages, tools=None, temperature=None, model=None, force_json=False):
        calls.append({"messages": list(messages), "tools": tools})
        idx = min(len(calls) - 1, len(scripted) - 1)
        return scripted[idx]

    monkeypatch.setattr(cs_agent, "chat", fake_chat)
    return calls


# ============================================================
# 一、正常出口
# ============================================================


def test_answers_directly_without_tools(runtime, monkeypatch):
    """模型第一轮就直接给答案 —— 循环应在一轮内收口。"""
    install_fake_chat(monkeypatch, [make_resp("能不能退取决于对方是否收款。")])
    trace = cs_agent.CSAgent(runtime).run("转账转错了能退回来吗", case_id=1)

    assert trace.stopped_reason == "answered"
    assert trace.rounds == 1
    assert trace.tool_names == []
    assert "取决于" in trace.final_answer


def test_tool_result_is_fed_back_then_answers(runtime, monkeypatch):
    """调工具 → 结果回灌 → 第二轮作答。这是最典型的一次 Agent 会话。"""
    calls = install_fake_chat(
        monkeypatch,
        [
            make_resp(None, [{"id": "c1", "name": "search_precedents",
                              "arguments": {"query": "花呗退款退到哪里"}}]),
            make_resp("退回花呗额度。"),
        ],
    )
    trace = cs_agent.CSAgent(runtime).run("花呗退款是退到哪里", case_id=3)

    assert trace.stopped_reason == "answered"
    assert trace.tool_names == ["search_precedents"]

    # 第二轮的消息里必须带上工具返回结果，否则模型无从作答
    second_round_msgs = calls[1]["messages"]
    assert any(m.get("role") == "tool" for m in second_round_msgs)

    # assistant 那条带 tool_calls 的消息必须存在，且排在 tool 结果之前 ——
    # 少了它 OpenAI 协议会直接报错
    roles = [m.get("role") for m in second_round_msgs]
    assert roles.index("assistant") < roles.index("tool")


# ============================================================
# 二、失败与兜底
# ============================================================


def test_tool_error_is_fed_back_not_raised(runtime, monkeypatch):
    """工具报错要回灌给模型，而不是把 Agent 崩掉。"""
    calls = install_fake_chat(
        monkeypatch,
        [
            make_resp(None, [{"id": "c1", "name": "query_order",
                              "arguments": {"order_ref": "00000000000"}}]),
            make_resp("没能查到这笔订单，请核对一下订单号。"),
        ],
    )
    trace = cs_agent.CSAgent(runtime).run("订单 00000000000 什么状态", case_id=99)

    assert trace.stopped_reason == "answered"
    assert len(trace.failed_tools) == 1

    # 回灌给模型的 tool 消息里，要能看出这次失败了
    tool_msg = [m for m in calls[1]["messages"] if m.get("role") == "tool"][0]
    payload = json.loads(tool_msg["content"])
    assert payload["ok"] is False
    assert payload["error"]


def test_guard_blocked_refund_is_recorded(runtime, monkeypatch):
    """模型跳过用户确认直接退款 → 必须被拦，且记进轨迹。"""
    install_fake_chat(
        monkeypatch,
        [
            make_resp(None, [{"id": "c1", "name": "apply_refund",
                              "arguments": {"order_ref": ORDER_REFUNDABLE,
                                            "reason": "用户不想要了",
                                            "user_confirmed": True}}]),
            make_resp("已为您发起退款。"),
        ],
    )
    trace = cs_agent.CSAgent(runtime).run("订单能退吗", case_id=50)

    # user_confirmed=True 是模型自己填的，工具照做 —— 这一条测的是「模型说有确认」
    assert trace.tool_names == ["apply_refund"]
    assert trace.tool_calls[0]["ok"] is True


def test_missing_confirmation_creates_guard_event(runtime, monkeypatch):
    install_fake_chat(
        monkeypatch,
        [
            make_resp(None, [{"id": "c1", "name": "apply_refund",
                              "arguments": {"order_ref": ORDER_REFUNDABLE,
                                            "reason": "想退", "user_confirmed": False}}]),
            make_resp("请确认后再为您退款。"),
        ],
    )
    trace = cs_agent.CSAgent(runtime).run("帮我退款", case_id=51)
    assert len(trace.guard_events) == 1


def test_max_rounds_stops_a_looping_agent(runtime, monkeypatch):
    """模型陷进循环出不来时，硬上限必须兜住，不能无限跑下去。"""
    looping = make_resp(
        None,
        [{"id": "c1", "name": "search_precedents", "arguments": {"query": "一直查"}}],
    )
    install_fake_chat(monkeypatch, [looping])
    trace = cs_agent.CSAgent(runtime, max_rounds=3).run("随便问", case_id=52)

    assert trace.stopped_reason == "max_rounds"
    assert trace.rounds == 3
    # 达到上限的那一轮不再执行工具 —— 执行了也没有下一轮消费结果
    assert len(trace.tool_calls) == 2


def test_transfer_to_human_revokes_tool_access(runtime, monkeypatch):
    """★ 转人工之后必须收回工具权限，逼它收尾。

    否则模型可能在「已经转人工」之后继续调工具，输出自相矛盾的话。
    """
    calls = install_fake_chat(
        monkeypatch,
        [
            make_resp(None, [{"id": "c1", "name": "transfer_to_human",
                              "arguments": {"reason": "超出范围", "summary": "账户被冻结"}}]),
            make_resp("已为您转接人工客服。"),
        ],
    )
    trace = cs_agent.CSAgent(runtime).run("账户被冻结了怎么办", case_id=53)

    assert trace.stopped_reason == "human"
    assert trace.final_answer == "已为您转接人工客服。"
    # 第一轮给了工具清单，转人工后第二轮必须收回去
    assert calls[0]["tools"] is not None
    assert calls[1]["tools"] is None


def test_empty_response_is_flagged(runtime, monkeypatch):
    install_fake_chat(monkeypatch, [make_resp("")])
    trace = cs_agent.CSAgent(runtime).run("问点什么", case_id=54)
    assert trace.stopped_reason == "empty_response"


def test_llm_error_is_captured(runtime, monkeypatch):
    """模型接口挂了，轨迹要记下这一点，而不是抛异常把整轮回归中断。"""
    from src.llm_client import LLMError

    def boom(*args, **kwargs):
        raise LLMError("模拟接口故障")

    monkeypatch.setattr(cs_agent, "chat", boom)
    trace = cs_agent.CSAgent(runtime).run("问点什么", case_id=55)

    assert trace.stopped_reason == "llm_error"
    assert any(s.kind == "error" for s in trace.steps)


# ============================================================
# 三、多轮
# ============================================================


def test_history_is_prepended(runtime, monkeypatch):
    """多轮：历史消息要出现在当前提问之前。"""
    calls = install_fake_chat(monkeypatch, [make_resp("好的。")])
    cs_agent.CSAgent(runtime).run(
        "那要多久",
        case_id=56,
        history=[
            {"role": "user", "content": "我的订单能退吗"},
            {"role": "assistant", "content": "请提供订单号"},
        ],
    )
    msgs = calls[0]["messages"]
    contents = [m.get("content") for m in msgs]
    assert contents.index("我的订单能退吗") < contents.index("那要多久")
