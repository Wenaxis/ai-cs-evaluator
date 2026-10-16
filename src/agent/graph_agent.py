"""
[M11] LangGraph 版客服 Agent

同一个 Agent，用 LangGraph 重写一遍。手写版（cs_agent.py）**保留不删**。

为什么要留两个版本？这是刻意的，不是重复劳动：

    · 手写版回答的是「你懂不懂原理」—— ReAct 循环怎么转、消息协议怎么拼、
      转人工为什么是终态、失败信息怎么回灌，每一行都在自己手里。
    · LangGraph 版回答的是「你会不会用工业界的工具」—— 状态机、条件边、
      检查点、可视图。这是真实团队协作里的形态。

    面试里这两个问题都会问，而且是两个不同的问题：

        只写手写版 → 会被问「为什么不用现成框架」
        只写框架版 → 会被问「框架帮你做了什么、你自己又做了什么」

    两个都在，这两个问题就都有答案。而且第二问的答案会很有说服力 ——
    因为我们不是照着教程抄的，是先手写过一遍，再用框架重写。

**LangGraph 在这里的真实价值是什么？**

    不是「少写代码」—— 说实话，声明式状态机比手写循环还长一点。
    它真正的价值是另外三条：

    1. 流程变成了显式的图，而不是埋在 while 循环的控制流里。
       新人接代码时看的是节点和边，不用跟着跳转一路读到底。
    2. 可以挂 checkpointer 做断点续跑。长任务跑挂了能从中间接着跑，
       手写版要自己实现这一整套。
    3. 可以 print_ascii() 直接打出流程图，评审和调试时能看见流程长什么样。

    所以结论是：**框架解决的是「流程可见性和可恢复性」，不是「写起来更短」**。
    能说清这一点，比会用框架本身更重要。
"""

from __future__ import annotations

import json
import time
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from src.agent.cs_agent import SYSTEM_PROMPT
from src.agent.tools import TOOL_SCHEMAS, ToolRuntime
from src.agent.trace import Step, Trace
from src.config import AGENT_MAX_ROUNDS, AGENT_TEMPERATURE
from src.llm_client import LLMError, UsageMeter, chat


class AgentState(TypedDict, total=False):
    """在图上流动的状态。

    这里刻意**不用** Annotated[list, operator.add] 那种 reducer 写法。
    看起来更「框架原生」，但状态一旦被隐式追加，调试时你就不知道
    某一轮消息到底是谁加的 —— 对 Agent 这种每轮都要复盘的东西，
    「显式覆盖」比「自动追加」好排查得多。
    """

    messages: list[dict]
    rounds: int
    steps: list[dict]
    pending_tool_calls: list[dict]
    final_answer: str
    stopped_reason: str
    tools_enabled: bool
    escalated: bool


class GraphAgent:
    """LangGraph 编排的客服 Agent。

    对外接口和手写版 CSAgent 完全一致（run(question, case_id, history) → Trace），
    所以两者可以直接替换、跑同一份评测集 —— 这正是把评测做成独立体系的好处。
    """

    def __init__(
        self,
        runtime: ToolRuntime,
        max_rounds: int | None = None,
        meter: UsageMeter | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> None:
        self.runtime = runtime
        self.max_rounds = max_rounds or AGENT_MAX_ROUNDS
        self.meter = meter or UsageMeter()
        self.model = model
        self.temperature = AGENT_TEMPERATURE if temperature is None else temperature
        self.graph = self._build_graph()

    # ============================================================
    # 图定义
    # ============================================================

    def _build_graph(self):
        builder = StateGraph(AgentState)

        builder.add_node("agent", self._agent_node)
        builder.add_node("tools", self._tools_node)

        builder.add_edge(START, "agent")
        builder.add_conditional_edges(
            "agent",
            self._route_after_agent,
            {"tools": "tools", "end": END},
        )
        builder.add_conditional_edges(
            "tools",
            self._route_after_tools,
            {"agent": "agent", "end": END},
        )

        return builder.compile()

    # ---------- 节点 ----------

    def _agent_node(self, state: AgentState) -> dict:
        """问一次模型。这是图上唯一会说话的地方。"""
        rounds = state.get("rounds", 0) + 1
        steps = list(state.get("steps") or [])
        messages = list(state["messages"])

        t0 = time.perf_counter()
        try:
            resp = chat(
                messages,
                tools=TOOL_SCHEMAS if state.get("tools_enabled", True) else None,
                temperature=self.temperature,
                model=self.model,
            )
        except LLMError as e:
            steps.append(
                {"kind": "error", "round": rounds, "detail": {"error": str(e)}, "elapsed_ms": 0}
            )
            return {
                "steps": steps,
                "stopped_reason": "llm_error",
                "pending_tool_calls": [],
            }

        elapsed = int((time.perf_counter() - t0) * 1000)
        self.meter.add(resp.usage)
        steps.append(
            {
                "kind": "llm",
                "round": rounds,
                "elapsed_ms": elapsed,
                "detail": {
                    "content": resp.content,
                    "tool_calls": [
                        {"name": c["name"], "arguments": c["arguments"]} for c in resp.tool_calls
                    ],
                    "usage": resp.usage,
                },
            }
        )

        update: dict[str, Any] = {"rounds": rounds, "steps": steps}

        if resp.wants_tool:
            messages.append(self._assistant_tool_msg(resp))
            update["messages"] = messages
            update["pending_tool_calls"] = resp.tool_calls
            # 轮数用尽还在调工具：就地收口，别让它无限转下去
            if rounds >= self.max_rounds:
                update["stopped_reason"] = "max_rounds"
                update["pending_tool_calls"] = []
        else:
            messages.append({"role": "assistant", "content": resp.content or ""})
            update["messages"] = messages
            update["pending_tool_calls"] = []
            answer = (resp.content or "").strip()
            update["final_answer"] = answer
            if state.get("escalated"):
                # 已经转过人工了：这句只是收尾话术，不改终态的性质
                update["stopped_reason"] = "human"
            else:
                update["stopped_reason"] = "answered" if answer else "empty_response"

        return update

    def _tools_node(self, state: AgentState) -> dict:
        """执行模型要调的每一个工具。"""
        messages = list(state["messages"])
        steps = list(state.get("steps") or [])
        round_no = state.get("rounds", 1)
        tools_enabled = state.get("tools_enabled", True)
        escalated = state.get("escalated", False)

        for call in state.get("pending_tool_calls") or []:
            t1 = time.perf_counter()
            result = self.runtime.dispatch(call["name"], call["arguments"])
            tool_ms = int((time.perf_counter() - t1) * 1000)

            steps.append(
                {
                    "kind": "tool",
                    "round": round_no,
                    "elapsed_ms": tool_ms,
                    "detail": {
                        "name": call["name"],
                        "arguments": call["arguments"],
                        "ok": result.get("ok", False),
                        "data": result.get("data"),
                        "error": result.get("error"),
                    },
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(
                        {
                            "ok": result.get("ok"),
                            "data": result.get("data"),
                            "error": result.get("error"),
                            "hint": result.get("hint"),
                        },
                        ensure_ascii=False,
                    ),
                }
            )

            if call["name"] == "transfer_to_human" and result.get("ok"):
                # 转人工即终态：收回工具权限，逼它下一轮只能说话
                tools_enabled = False
                escalated = True

        return {
            "messages": messages,
            "steps": steps,
            "pending_tool_calls": [],
            "tools_enabled": tools_enabled,
            "escalated": escalated,
        }

    # ---------- 路由 ----------

    def _route_after_agent(self, state: AgentState) -> str:
        """模型说完话之后：还有工具要调就去 tools，否则结束。"""
        if state.get("stopped_reason"):
            return "end"
        if state.get("pending_tool_calls"):
            return "tools"
        return "end"

    def _route_after_tools(self, state: AgentState) -> str:
        """工具跑完之后：回模型继续，除非轮数已经用尽。"""
        if state.get("rounds", 0) >= self.max_rounds and not state.get("escalated"):
            return "end"
        return "agent"

    # ---------- 消息组装 ----------

    @staticmethod
    def _assistant_tool_msg(resp) -> dict:
        """OpenAI 协议要求：带 tool_calls 的 assistant 消息必须排在 tool 结果之前。"""
        return {
            "role": "assistant",
            "content": resp.content or "",
            "tool_calls": [
                {
                    "id": c["id"],
                    "type": "function",
                    "function": {
                        "name": c["name"],
                        "arguments": json.dumps(c["arguments"], ensure_ascii=False),
                    },
                }
                for c in resp.tool_calls
            ],
        }

    # ============================================================
    # 对外接口（和手写版保持一致）
    # ============================================================

    def run(
        self,
        question: str,
        case_id: int | str,
        history: list[dict] | None = None,
    ) -> Trace:
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for h in history or []:
            messages.append(h)
        messages.append({"role": "user", "content": question})

        init: AgentState = {
            "messages": messages,
            "rounds": 0,
            "steps": [],
            "pending_tool_calls": [],
            "final_answer": "",
            "stopped_reason": "",
            "tools_enabled": True,
            "escalated": False,
        }

        final = self.graph.invoke(init, config={"recursion_limit": self.max_rounds * 3 + 5})

        trace = Trace(case_id=case_id, question=question)
        trace.steps = [
            Step(
                kind=s.get("kind", "llm"),
                round=s.get("round", 0),
                detail=s.get("detail") or {},
                elapsed_ms=s.get("elapsed_ms", 0),
            )
            for s in final.get("steps") or []
        ]
        trace.final_answer = final.get("final_answer", "")
        trace.stopped_reason = final.get("stopped_reason") or "max_rounds"
        trace.usage = self.meter.summary()
        return trace

    def print_flow(self) -> None:
        """打出流程图。调试和评审时，这个比读代码快得多。"""
        try:
            print(self.graph.get_graph().draw_ascii())
        except Exception as e:  # noqa: BLE001
            print(f"（当前环境无法绘制流程图：{e}）")
