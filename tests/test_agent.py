"""
v2 的单元测试。

为什么值得花时间写测试 —— 三个理由，按重要性排：

1. 这套代码大部分跑在「不确定环境」里：模型可能返回烂 JSON、工具可能报错、
   检索可能一条都命不中。这些分支靠手点很难覆盖，而它们恰好最容易出问题。

2. 防泄漏那条规则（leave-one-out）如果被改坏了，评测数字会全线变好看，
   而且**不会报任何错**。这种静默失效只有测试拦得住 —— 这是写这份测试最主要的动机。

3. Agent 每跑一次都要烧 API 额度。能用测试验证的，就别用真跑验证。

运行方式（项目根目录）：
    .venv/Scripts/python -m pytest tests/ -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agent.retrieval import PrecedentIndex, build_index, tokenize
from src.agent.tools import TOOL_SCHEMAS, ToolRuntime
from src.agent.trace import Trace

# 几个 mock 订单号，和 tools.py 里的库对应
ORDER_REFUNDABLE = "20240815002"   # 待发货 → 可退
ORDER_SIGNED = "20240815001"       # 已签收 → 不可直接退
ORDER_REFUNDED = "20240815004"     # 已退款 → 重复请求


@pytest.fixture(scope="module")
def records() -> list[dict]:
    path = Path(__file__).resolve().parent.parent / "data" / "processed" / "records.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture()
def index(records: list[dict]) -> PrecedentIndex:
    return build_index(records, only_correct=True)


@pytest.fixture()
def runtime(index: PrecedentIndex) -> ToolRuntime:
    return ToolRuntime(index=index, exclude_id=1)


# ============================================================
# 一、分词与检索
# ============================================================


def test_tokenize_keeps_cjk_bigrams():
    tokens = tokenize("转账转错了能退回来吗")
    assert "转" in tokens
    assert "转账" in tokens          # 相邻汉字要拼成 bigram
    assert "退" in tokens


def test_tokenize_keeps_numbers_whole():
    # 订单号这类连续数字要整体保留，不能被切成单个数字
    assert "20240815002" in tokenize("订单 20240815002 怎么退")


def test_bm25_finds_same_topic(index: PrecedentIndex):
    """问「退款到哪」，检索结果里应该出现退款场景的案例。"""
    hits = index.search("花呗退款退到哪里", k=3, exclude_id=None)
    assert hits, "检索不应为空"
    assert any(h["scene"] == "退款" for h in hits)


def test_search_respects_exclude_id(index: PrecedentIndex):
    """★ 防泄漏：排除掉的 id 绝不能出现在结果里。

    这条测试守的是整个 v2 评测可信度的底线 ——
    一旦它红了，说明评测数字全是抄答案抄出来的。
    """
    target = index.cases[0]["id"]
    hits = index.search(index.cases[0]["question"], k=5, exclude_id=target)
    assert all(h["id"] != target for h in hits), "被判例检索泄漏了自己，评测结果不可信"


def test_search_never_returns_empty_after_scene_fallback(index: PrecedentIndex):
    """场景过滤过滤空了，应自动放宽，而不是返回空列表。"""
    hits = index.search("随便问点什么", k=3, exclude_id=None, scene="不存在的场景")
    assert hits, "场景过滤为空时应该放宽，而不是让 Agent 两手空空"


def test_index_only_keeps_correct_cases(records: list[dict], index: PrecedentIndex):
    """判例库默认只收「答对」的样本。"""
    assert len(index) == sum(1 for r in records if r["human_label"] == "答对")
    assert len(index) < len(records)


# ============================================================
# 二、工具层
# ============================================================


def test_schema_and_impl_are_in_sync(runtime: ToolRuntime):
    """schema 里公布的每个工具，运行时都得真的实现得了。

    这两边不同步是很典型的事故：模型看到工具清单里有它就调，
    结果这边没人接，Agent 就在一个永远失败的工具上转圈。
    """
    for spec in TOOL_SCHEMAS:
        name = spec["function"]["name"]
        assert hasattr(runtime, f"_tool_{name}"), f"schema 里有 {name}，但没有对应实现"


def test_dispatch_unknown_tool_is_handled(runtime: ToolRuntime):
    result = runtime.dispatch("不存在的工具", {})
    assert result["ok"] is False
    assert "不存在" in result["error"]


def test_dispatch_bad_arguments_gives_signature(runtime: ToolRuntime):
    """参数传错时，要把正确签名还给模型，它下一轮才能改对。"""
    result = runtime.dispatch("query_order", {"wrong_param": "x"})
    assert result["ok"] is False
    assert "order_ref" in result["hint"]


def test_query_order_success_and_failure(runtime: ToolRuntime):
    ok = runtime.dispatch("query_order", {"order_ref": ORDER_REFUNDABLE})
    assert ok["ok"] is True
    assert ok["data"]["status"] == "待发货"

    bad = runtime.dispatch("query_order", {"order_ref": "00000000000"})
    assert bad["ok"] is False
    assert bad["hint"]


# ============================================================
# 三、安全闸门（写操作）
# ============================================================


def test_refund_blocked_without_user_confirmation(runtime: ToolRuntime):
    """★ 没有用户确认，退款必须被拦下。

    这条守的是「模型不能自己决定把钱退出去」——
    闸门放在代码层而不是 prompt 层，就是因为 prompt 拦不住。
    """
    result = runtime.dispatch(
        "apply_refund", {"order_ref": ORDER_REFUNDABLE, "reason": "测试", "user_confirmed": False}
    )
    assert result["ok"] is False
    assert "确认" in result["error"]


def test_modify_delivery_blocked_without_user_confirmation(runtime: ToolRuntime):
    result = runtime.dispatch(
        "modify_delivery",
        {"order_ref": ORDER_REFUNDABLE, "new_address": "某地", "user_confirmed": False},
    )
    assert result["ok"] is False
    assert "确认" in result["error"]


def test_refund_succeeds_when_confirmed(runtime: ToolRuntime):
    result = runtime.dispatch(
        "apply_refund", {"order_ref": ORDER_REFUNDABLE, "reason": "不想要了", "user_confirmed": True}
    )
    assert result["ok"] is True
    assert result["data"]["refund_id"].startswith("RF")


def test_refund_rejects_signed_order(runtime: ToolRuntime):
    """已签收的订单不能直接退 —— Agent 必须撞上这个失败并处理它。"""
    result = runtime.dispatch(
        "apply_refund", {"order_ref": ORDER_SIGNED, "reason": "后悔了", "user_confirmed": True}
    )
    assert result["ok"] is False
    assert "已签收" in result["error"]


def test_refund_is_idempotent_on_already_refunded(runtime: ToolRuntime):
    """重复退款不算错误，但要告诉模型「别再操作了」。"""
    result = runtime.dispatch(
        "apply_refund", {"order_ref": ORDER_REFUNDED, "reason": "再退一次", "user_confirmed": True}
    )
    assert result["ok"] is True
    assert result["data"]["already_refunded"] is True


def test_modify_delivery_rejects_shipped_order(runtime: ToolRuntime):
    result = runtime.dispatch(
        "modify_delivery",
        {"order_ref": ORDER_SIGNED, "new_address": "新地址", "user_confirmed": True},
    )
    assert result["ok"] is False


# ============================================================
# 四、轨迹
# ============================================================


def test_trace_aggregates_correctly():
    t = Trace(case_id=7, question="测试问题")
    t.add_llm(1, None, [{"id": "c1", "name": "search_precedents", "arguments": {}}], {}, 120)
    t.add_tool(1, "search_precedents", {"query": "x"}, {"ok": True, "data": {"count": 3}}, 5)
    t.add_llm(2, "这是最终回复", [], {}, 200)

    assert t.tool_names == ["search_precedents"]
    assert t.failed_tools == []
    assert t.rounds == 2
    assert t.total_ms == 325
    d = t.to_dict()
    assert d["case_id"] == 7
    assert len(d["steps"]) == 3


def test_trace_detects_guard_events():
    """被确认闸门拦下的次数要能被单独统计出来 —— 这是个安全指标。"""
    t = Trace(case_id=8, question="测试问题")
    t.add_tool(
        1,
        "apply_refund",
        {"order_ref": ORDER_REFUNDABLE, "user_confirmed": False},
        {"ok": False, "error": "用户尚未确认，退款被拦截"},
        3,
    )
    assert len(t.guard_events) == 1
