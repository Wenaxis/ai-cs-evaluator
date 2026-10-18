"""
[M7] 客服 Agent 的工具集

一个工具 = 一份给模型看的说明书（schema）+ 一段真干活的代码（实现）。
两样缺一不可：只有实现没 schema，模型不知道有这个工具；
只有 schema 没实现，模型调了也没人接。

三个设计决定（都会被问）：

一、工具失败返回 {ok: False, error, hint}，而不是抛异常。
    模型看不懂 Python 的 traceback。它需要的是一句人话：哪儿错了、下一步能干什么。
    这就是 Agent 的失败兜底 —— 错误也是信息，不是崩溃。

二、写操作（退款 / 改地址）必须有 user_confirmed 闸门。
    模型不能自己决定把钱退出去。闸门放在工具层而不是写进 prompt，
    因为 prompt 是「建议」，代码才是「规则」——写进 prompt 的那种，模型心情不好就绕过去了。

三、订单库是 mock 的，而且故意留了几条会失败的记录。
    一个永远成功的工具集，测不出 Agent 的兜底能力。
    真实客服系统里 80% 的复杂度都在「事情办不成怎么办」上。
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import TYPE_CHECKING

from src.agent.retrieval import PrecedentIndex

if TYPE_CHECKING:
    # 只在类型标注里用，运行时不导入：knowledge 会去读 data/knowledge 下的数据文件，
    # 模块加载期就把它拉起来，等于给「工具集」绑了一个可选数据依赖。
    from src.agent.knowledge import KnowledgeBase


# ============================================================
# 一、给模型看的说明书（OpenAI function calling 格式）
# ============================================================
# description 的写法有讲究：不只说「这个工具干什么」，还要说「什么时候用它」。
# 模型是在几句 description 之间做选择，讲清适用场景比讲清功能更能减少误调用。

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": (
                "检索业务规则知识库，拿到回答的事实依据。"
                "这是回答规则类问题的第一选择，优先级高于 search_precedents。"
                "凡是「能不能 / 多久 / 怎么操作 / 在哪里看 / 需要什么」这类问题，先查这里。"
                "返回的是规则条款（含具体口径），可以直接作为答案的骨架。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索用问句，直接用用户的原话，不要改写成关键词",
                    },
                    "k": {
                        "type": "integer",
                        "description": "返回几条，默认 3，一般不用传",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_precedents",
            "description": (
                "在历史真实问答的标准答法库里检索相似案例，用于参考「这类问题通常怎么组织回答」。"
                "知识库没有覆盖时，可以查这里看看有没有可借鉴的表达。"
                "注意：判例是应答摘要，不保证含完整事实口径。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索用问句，直接用用户的原话，不要改写",
                    },
                    "scene": {
                        "type": "string",
                        "enum": ["退款", "改单", "物流", "催件"],
                        "description": "可选的场景过滤，拿不准就不传",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_order",
            "description": (
                "按订单号查询订单状态、金额和商品。"
                "只有当用户提供了具体订单号、且问题确实与该笔订单有关时才调用。"
                "查订单是一个「动作」，它本身不构成对用户问题的回答。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_ref": {
                        "type": "string",
                        "description": "订单号，通常是 11 位数字",
                    }
                },
                "required": ["order_ref"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_refund",
            "description": (
                "对指定订单发起退款。这是不可撤销的写操作。"
                "调用前必须在对话中向用户复述订单号和金额，并取得用户明确同意；"
                "用户没有明确说同意之前，user_confirmed 一律传 false。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_ref": {"type": "string", "description": "订单号"},
                    "reason": {"type": "string", "description": "退款原因，简短说明"},
                    "user_confirmed": {
                        "type": "boolean",
                        "description": "用户是否已在对话中明确同意退款",
                    },
                },
                "required": ["order_ref", "reason", "user_confirmed"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "modify_delivery",
            "description": (
                "修改待发货订单的收货地址。这是写操作。"
                "调用前必须向用户复述新地址并取得明确同意，否则 user_confirmed 传 false。"
                "订单一旦进入运输中或已签收，本工具会失败。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_ref": {"type": "string", "description": "订单号"},
                    "new_address": {"type": "string", "description": "新的收货地址全文"},
                    "user_confirmed": {
                        "type": "boolean",
                        "description": "用户是否已在对话中明确同意修改地址",
                    },
                },
                "required": ["order_ref", "new_address", "user_confirmed"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transfer_to_human",
            "description": (
                "转接人工客服。适用于三种情况：问题超出规则库范围、"
                "用户明确要求人工、或连续两轮都无法给出有效回答。"
                "转人工是兜底手段，不应作为第一个动作。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "转人工的原因"},
                    "summary": {
                        "type": "string",
                        "description": "给人工客服的问题摘要，一行说清用户诉求",
                    },
                },
                "required": ["reason", "summary"],
            },
        },
    },
]

# 写操作清单。runner 和评测模块会用这个集合来判断「这次执行有没有触碰副作用」。
WRITE_TOOLS = {"apply_refund", "modify_delivery"}


# ============================================================
# 二、mock 订单库
# ============================================================
# 状态刻意造得有差异：可退的、已退的、已签收不能退的、运输中改不了地址的。
# 这样 Agent 才有机会撞上真实世界里的「办不成」，也才测得出兜底能力。

_MOCK_ORDERS: dict[str, dict] = {
    "20240815001": {
        "order_ref": "20240815001",
        "item": "无线蓝牙耳机",
        "amount": 128.00,
        "status": "已签收",
        "refundable": False,
    },
    "20240815002": {
        "order_ref": "20240815002",
        "item": "洗发水套装",
        "amount": 59.90,
        "status": "待发货",
        "refundable": True,
    },
    "20240815003": {
        "order_ref": "20240815003",
        "item": "机械键盘",
        "amount": 899.00,
        "status": "运输中",
        "refundable": False,
    },
    "20240815004": {
        "order_ref": "20240815004",
        "item": "手机保护壳",
        "amount": 45.00,
        "status": "已退款",
        "refundable": False,
    },
}


# ============================================================
# 三、运行时
# ============================================================


class ToolRuntime:
    """工具的执行环境。

    exclude_id 在构造时就绑死，而不是每次调用时传 —— 判例检索的防泄漏
    （评第 N 条时不能检索到第 N 条自己）因此变成一件「忘不掉」的事。

    kb 是 v3 新增的业务知识库。不传就自动加载 data/knowledge/cs_rules.json；
    加载失败也不让整个 Agent 崩掉，只是 search_knowledge 会返回一条明确的错误 ——
    单点数据缺失不该拖垮会话。
    """

    def __init__(
        self,
        index: PrecedentIndex,
        exclude_id: int | None = None,
        kb: KnowledgeBase | None = None,
    ) -> None:
        self.index = index
        self.exclude_id = exclude_id
        if kb is None:
            try:
                from src.agent.knowledge import KnowledgeBase

                kb = KnowledgeBase()
            except Exception:  # noqa: BLE001 - 知识库缺失不该阻断主流程
                kb = None
        self.kb = kb
        self.call_log: list[dict] = []  # 本次会话调过哪些工具，供 trace 和评测使用

    # ---------- 统一入口 ----------

    def dispatch(self, name: str, args: dict) -> dict:
        """执行一个工具，永远返回结构化结果，从不抛异常。"""
        impl: Callable[..., dict] | None = getattr(self, f"_tool_{name}", None)
        if impl is None:
            return {
                "ok": False,
                "error": f"不存在名为 {name} 的工具",
                "hint": "请从工具清单里重新选择",
            }

        try:
            result = impl(**(args or {}))
        except TypeError as e:
            # 参数名或个数不对。把正确签名还给它，让模型下一轮能改对。
            result = {
                "ok": False,
                "error": f"参数不合法：{e}",
                "hint": f"该工具接受的参数是：{self._signature(name)}",
            }
        except Exception as e:  # noqa: BLE001
            result = {
                "ok": False,
                "error": f"工具执行异常：{type(e).__name__}: {e}",
                "hint": "换一种方式处理，或考虑转人工",
            }

        self.call_log.append(
            {"tool": name, "arguments": args, "ok": result.get("ok", False)}
        )
        return result

    @staticmethod
    def _signature(name: str) -> str:
        for spec in TOOL_SCHEMAS:
            fn = spec["function"]
            if fn["name"] == name:
                return ", ".join(fn["parameters"].get("properties", {}).keys())
        return "(未知)"

    # ---------- 各工具实现 ----------

    def _tool_search_knowledge(self, query: str, k: int = 3) -> dict:
        """查业务规则知识库。这是回答规则类问题的第一依据。"""
        if self.kb is None:
            return {
                "ok": False,
                "error": "知识库未加载",
                "hint": "data/knowledge/cs_rules.json 缺失或格式错误。"
                "这种情况下请改用 search_precedents，或直接转人工，不要凭空编规则。",
            }

        try:
            k = max(1, min(int(k), 5))
        except (TypeError, ValueError):
            k = 3

        hits = self.kb.search(query, k=k)
        if not hits:
            return {
                "ok": True,
                "data": {"rules": [], "count": 0},
                "hint": "知识库没有覆盖这个问题。可以再用 search_precedents 找找参考表达；"
                "两者都没有的话，转人工比编一个听起来合理的规则更合适。",
            }

        return {
            "ok": True,
            "data": {"rules": hits, "count": len(hits)},
            "hint": "以上是业务规则条款，按从上到下的相关度排列。"
            "请先用第一条作答：结论在前，路径在后，具体数值照条款里的口径说，不要自行加码。"
            "如果这些条款与用户问题对不上（问的不是同一件事），就当作没查到，不要硬套。",
        }

    def _tool_search_precedents(self, query: str, scene: str | None = None) -> dict:
        hits = self.index.search(
            query=query, k=3, exclude_id=self.exclude_id, scene=scene
        )
        if not hits:
            return {
                "ok": True,
                "data": {"cases": [], "count": 0},
                "hint": "判例库里没有相似案例。请基于通用常识谨慎作答，"
                "并明确告诉用户这一点的确定性有限；如涉及具体账户或资金操作，考虑转人工。",
            }
        return {
            "ok": True,
            "data": {"cases": hits, "count": len(hits)},
            "hint": "以上是历史真实案例中的标准答法，可作为口径参考；"
            "注意结合用户问句里的具体前提条件，不要直接照抄。",
        }

    def _tool_query_order(self, order_ref: str) -> dict:
        order = _MOCK_ORDERS.get(str(order_ref).strip())
        if order is None:
            return {
                "ok": False,
                "error": f"查不到订单 {order_ref}",
                "hint": "请让用户核对订单号；若用户无法提供，按通用规则作答，不要臆造订单信息。",
            }
        return {"ok": True, "data": order, "hint": "查询成功，可直接引用其中的状态与金额。"}

    def _tool_apply_refund(
        self, order_ref: str, reason: str, user_confirmed: bool = False
    ) -> dict:
        if not user_confirmed:
            return {
                "ok": False,
                "error": "用户尚未确认，退款被拦截",
                "hint": "请先向用户复述订单号与金额，取得明确同意后再调用一次，"
                "并把 user_confirmed 传 true。",
            }

        order = _MOCK_ORDERS.get(str(order_ref).strip())
        if order is None:
            return {
                "ok": False,
                "error": f"查不到订单 {order_ref}，无法发起退款",
                "hint": "请让用户核对订单号。",
            }
        if order["status"] == "已退款":
            return {
                "ok": True,
                "data": {"order_ref": order_ref, "already_refunded": True},
                "hint": "该订单此前已完成退款，属于重复请求，不必再操作，直接告知用户即可。",
            }
        if not order["refundable"]:
            return {
                "ok": False,
                "error": f"订单当前状态为「{order['status']}」，不符合直接退款条件",
                "hint": "请告知用户需要走退货退款流程，并给出申请入口；"
                "如用户不接受，可转人工。",
            }

        return {
            "ok": True,
            "data": {
                "order_ref": order_ref,
                "refund_id": f"RF{random.randint(100000, 999999)}",
                "amount": order["amount"],
                "eta": "1-3 个工作日",
                "reason": reason,
            },
            "hint": "退款已受理，请把退款单号、金额和到账时效一并告知用户。",
        }

    def _tool_modify_delivery(
        self, order_ref: str, new_address: str, user_confirmed: bool = False
    ) -> dict:
        if not user_confirmed:
            return {
                "ok": False,
                "error": "用户尚未确认，修改地址被拦截",
                "hint": "请先向用户复述新地址全文，取得明确同意后再调用，并传 user_confirmed=true。",
            }

        order = _MOCK_ORDERS.get(str(order_ref).strip())
        if order is None:
            return {
                "ok": False,
                "error": f"查不到订单 {order_ref}",
                "hint": "请让用户核对订单号。",
            }
        if order["status"] != "待发货":
            return {
                "ok": False,
                "error": f"订单已处于「{order['status']}」，无法修改收货地址",
                "hint": "请说明原因，并建议用户联系承运方或转人工处理。",
            }

        return {
            "ok": True,
            "data": {"order_ref": order_ref, "new_address": new_address, "status": "已更新"},
            "hint": "地址已更新，请把新地址复述一遍让用户确认。",
        }

    def _tool_transfer_to_human(self, reason: str, summary: str) -> dict:
        return {
            "ok": True,
            "data": {
                "ticket_id": f"T{random.randint(10000, 99999)}",
                "reason": reason,
                "summary": summary,
                "eta": "30 分钟内",
            },
            "hint": "人工工单已创建。请把工单号和预计响应时间告知用户，"
            "并简要说明已为ta保留了问题背景。",
        }
