"""
[M13] MCP Server —— 把工具集暴露给任何支持 MCP 的客户端

前面做的 5 个工具，原本是「只给自己的 Agent 用」的私有能力。
这一层把它们暴露成标准的 MCP 工具，于是 Claude Desktop、Cursor、
或者任何一个支持 MCP 的 Agent，都能直接调它们。

**这个差别的分量在哪：**

    私有工具 → 只有我这一个 Agent 能用
    MCP 工具 → 它变成了一个可被复用的能力单元，换框架、换客户端照样能用

    这就是「做了个 demo」和「做了个别人能用起来的东西」之间的差别。

实现上有两点值得说明：

    1. **复用同一个 ToolRuntime，不重写工具逻辑。**
       MCP 这层只做协议转换 —— 把返回值转成 MCP 能传的格式。
       工具实现只有一份，两个入口。这样就不会出现
       「MCP 版和内置版行为不一致」这种最难查的问题。

    2. **工具描述直接沿用 TOOL_SCHEMAS 里那套文案。**
       写两遍描述，早晚会有一边忘了改。

启动方式（项目根目录）：
    python -m src.mcp_server              # stdio 模式，供客户端拉起

接到 Claude Desktop / Cursor 的配置示例（mcpServers 里加一条）：

    "cs-evaluator-tools": {
      "command": "python",
      "args": ["-m", "src.mcp_server"],
      "cwd": "<项目根目录>"
    }
"""

from __future__ import annotations

import json

# 注意：MCP SDK 从 2.0 起把 FastMCP 改名成了 MCPServer。
# 这里直接用新名字 —— 老名字在 2.x 里会直接抛 ModuleNotFoundError，
# 是个故意做出来的硬切换，没有兼容层。写代码时踩过这一下，记在这。
from mcp.server.mcpserver import MCPServer

from src.agent.retrieval import build_index
from src.agent.tools import ToolRuntime
from src.config import RECORDS_JSON

mcp = MCPServer("cs-evaluator-tools")

# 进程内只建一次判例索引
_runtime: ToolRuntime | None = None


def _rt() -> ToolRuntime:
    global _runtime
    if _runtime is None:
        records = json.loads(RECORDS_JSON.read_text(encoding="utf-8"))
        # 在线调用没有「正在被评测的那条用例」，所以 exclude_id 传 None。
        # 评测场景下必须传 case id（见 eval 模块），否则会泄漏答案。
        _runtime = ToolRuntime(index=build_index(records), exclude_id=None)
    return _runtime


def _dump(result: dict) -> str:
    """统一把工具结果转成 JSON 字符串。

    MCP 的返回值需要是可序列化的内容。这里刻意把 ok / error / hint 都留着 ——
    调用方（可能是别的 Agent）需要知道「这次到底成没成、下一步能干什么」，
    而不只是拿到一段文本。
    """
    return json.dumps(result, ensure_ascii=False, indent=2)


# ============================================================
# 工具（描述与 TOOL_SCHEMAS 保持一致）
# ============================================================


@mcp.tool()
def search_precedents(query: str, scene: str | None = None) -> str:
    """在历史真实问答的标准答法库里检索相似案例。

    凡是「通用规则类」问题（能不能退 / 要多久 / 怎么操作 / 在哪里看），
    都应先调这个工具拿到参考口径，再组织回复。
    """
    return _dump(_rt().dispatch("search_precedents", {"query": query, "scene": scene}))


@mcp.tool()
def query_order(order_ref: str) -> str:
    """按订单号查询订单状态、金额和商品。

    只有当用户提供了具体订单号、且问题确实与该笔订单有关时才调用。
    查订单是一个「动作」，它本身不构成对用户问题的回答。
    """
    return _dump(_rt().dispatch("query_order", {"order_ref": order_ref}))


@mcp.tool()
def apply_refund(order_ref: str, reason: str, user_confirmed: bool = False) -> str:
    """对指定订单发起退款。这是不可撤销的写操作。

    调用前必须在对话中向用户复述订单号和金额并取得明确同意；
    用户没有明确说同意之前，user_confirmed 一律传 false。
    """
    return _dump(
        _rt().dispatch(
            "apply_refund",
            {"order_ref": order_ref, "reason": reason, "user_confirmed": user_confirmed},
        )
    )


@mcp.tool()
def modify_delivery(order_ref: str, new_address: str, user_confirmed: bool = False) -> str:
    """修改待发货订单的收货地址。这是写操作，需用户明确同意。

    订单一旦进入运输中或已签收，本工具会失败。
    """
    return _dump(
        _rt().dispatch(
            "modify_delivery",
            {
                "order_ref": order_ref,
                "new_address": new_address,
                "user_confirmed": user_confirmed,
            },
        )
    )


@mcp.tool()
def transfer_to_human(reason: str, summary: str) -> str:
    """转接人工客服。

    适用于：问题超出规则库范围、用户明确要求人工、或连续两轮都无法给出有效回答。
    转人工是兜底手段，不应作为第一个动作。
    """
    return _dump(_rt().dispatch("transfer_to_human", {"reason": reason, "summary": summary}))


if __name__ == "__main__":
    mcp.run()
