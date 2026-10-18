"""
[M7] 客服 Agent 主循环（ReAct）

这是 v2 的心脏。一次会话长这样：

    用户提问
      └─► 模型：我先查一下判例库
            └─► 工具：返回 3 条相似案例
                  └─► 模型：还不够，这条涉及账户状态，再查一下订单
                        └─► 工具：返回订单详情
                              └─► 模型：够了，给出回复          ← 循环结束

和 v1 最本质的差别就在这个「循环」上：
    v1 是「读数据 → 打分 → 输出」，一条道走到黑，中间不回头。
    v2 的 Agent 每走一步都要重新判断「我现在够了没有」，
    而这个判断权在模型手里，不在代码手里 —— 这就是 agent 和流水线的分界。

四层护栏（没有护栏的 Agent 是 demo，不是工程）：

    1. 最大轮数。模型偶尔会陷进「调工具 → 觉得不够 → 再调同一个工具」的圈里。
       人和模型都会这样，所以要有个硬上限兜住。
    2. 工具失败不中断。失败信息回灌给模型，让它自己换路子。
       把异常抛出去 = Agent 死在这一步；把异常喂回去 = Agent 有机会自救。
    3. 转人工即终态。一旦转了人工，就收回它的工具权限，逼它收尾。
       否则它可能在「已经转人工」之后还继续折腾工具，输出一段自相矛盾的话。
    4. 规则类问题不许裸答（v3.1 加）。前三层管「怎么收尾」，这一层管
       「不许不查资料就凭记忆答」—— 它是唯一一层在「模型想说话」时主动打断的护栏，
       实现和踩坑记录见下方第 93 行起。

    注意：第 4 层目前只在手写版（本文件）里。LangGraph 版
    （graph_agent.py）只实现了前 3 层，属于对照实现，未补齐 ——
    所以 API 默认走 loop，别默默换成 graph。
"""

from __future__ import annotations

import json
import re
import time

from src.agent.tools import TOOL_SCHEMAS, ToolRuntime
from src.agent.trace import Trace
from src.config import (
    AGENT_MAX_ROUNDS,
    AGENT_TEMPERATURE,
    RULE_QUESTION_RE_PATTERN,
)
from src.llm_client import LLMError, UsageMeter, chat


# ============================================================
# 一、行为准则
# ============================================================
# 注意：这八条不是凭空写的，每一条都对应 v1 那份评分量表里的一条判据。
# v1 用它们来「评」别人的客服用它们来「做」。
# 一个评测标准能指导生产，才算真的想明白了 —— 这是 v1 和 v2 之间的逻辑闭环。

SYSTEM_PROMPT = """你是某第三方支付平台的在线客服的智能客服，负责解答用户在资金、订单、账户方面的疑问。

回答时必须遵守以下准则：

1. 先给结论，再给路径。
   用户问「能不能」，第一句就说清能还是不能；问「多久」，就给出具体时间或分档时长。
   把结论放最后、或者绕开结论只讲操作步骤，都算没回答这个问题。

2. 规则类问题先查知识库。
   凡是「能不能 / 多久 / 怎么操作 / 在哪里看 / 需要什么」这类通用规则问题，
   先调用 search_knowledge 拿到规则条款，按条款里的口径作答。
   知识库没覆盖时，再用 search_precedents 看有没有可借鉴的表达方式。
   两者都没有，才考虑转人工 —— 具体数值不要凭常识编。
   注意：查了订单、翻了账户，都只是动作，不是回答。用户问的那件事没被正面回答，就等于没回答。

3. 看住用户问句里的前提条件。
   「手机号已注销」「账户已冻结」这类前提必须单独给出分支方案，
   不能套用常规主路径。这是最容易出错的地方 —— 模板复用的典型症状就是这个。

4. 时效问题必须给时间。
   只说「请查看进度」「以实际为准」而不给任何具体时长，等于没回答。

5. 写操作必须先确认。
   退款、改地址之前，向用户复述订单号和金额，取得明确同意后才调用工具。
   没有明确同意就传 user_confirmed=false。

6. 转人工是兜底，不是首选。
   知识库和判例库都查不到、工具连续失败、或问题明确超出客服业务范围时，调用 transfer_to_human。
   但只要知识库给了口径，就照着答 —— 那种情况下转人工等于没干活。
   确实查不到依据时，宁可转人工，也不要编一个听起来很合理的数字。

7. 长度不是优点。
   用户要的是解决问题，不是一段客气的废话。不要为了显得周全而堆砌无关信息。

8. 分清「问怎么做」和「让你去做」。
   用户问「怎么转人工」「怎么申请退款」，问的是**路径**，不是让你立刻执行这个动作。
   先把路径讲清楚；只有用户明确说「帮我转」「帮我退」，才算进入执行流程。
   这类判断最容易错在「你手上正好有对应工具」的动作上 ——
   看到那几个字就去调工具，等于上来就动手，而用户真正问的（在哪里操作、需要什么）
   反而一个字没答。手上的工具越顺手，越要先确认这是不是用户要的。
"""

# ---- 护栏 4：规则类问题不许裸答 ----
#
# 准则 2 和 8 是软约束，模型可以不听。v3 跑完发现它真的会不听：
# #25「转人工客服要怎么操作」第一轮 0 次工具调用，直接凭记忆答了。
#
# 答得还挺像回事 —— 给了电话、给了 App 路径、还问了「要不要我帮你转」，
# judge 给了 5 分。但知识库里写的是「对话页点转人工按钮，服务时间 9:00-24:00」，
# 它答的是「电话热线，8:00-24:00」。时间错了。
#
# 这就是把软约束变成硬约束的理由：**凭记忆答，错得不明显，分数还好看。**
# 代码兜住这一层，只花一次多余的检索，比在客户面前把时间说错便宜得多。
#
# 只打回一次。第二次还不查，就让它答 —— 说明它确实判断不需要，
# 硬拧会把它拧成复读机。这时 skipped_retrieval 标签照打，账要记着。

_RULE_Q_RE = re.compile(RULE_QUESTION_RE_PATTERN)

_NUDGE_TEXT = (
    "这条问题是在问规则或操作路径，但你刚才没有查任何资料就作答了。\n"
    "请先调用 search_knowledge 查相关条款，按条款里的口径回答；\n"
    "知识库没覆盖，再调用 search_precedents 看看有没有可借鉴的表达。\n"
    "具体的时间、金额、入口路径不要凭记忆给 —— 记忆里的口径经常是过时的版本。"
)


class CSAgent:
    """一个客服 Agent 实例。

    刻意做成「一个实例 = 一次会话」的形态，而不是全局单例：
    这样并发跑测试集时每个用例各跑各的，工具调用日志不会互相污染。
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

    # ---------- 主循环 ----------

    def run(
        self,
        question: str,
        case_id: int,
        history: list[dict] | None = None,
    ) -> Trace:
        """回答一个问题，返回完整执行轨迹。

        history 用来支持多轮：传进来的是之前的 [{"role": ..., "content": ...}]。
        单轮场景传 None 即可（v1 那 30 条数据都是单轮）。
        """
        trace = Trace(case_id=case_id, question=question)

        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for h in history or []:
            messages.append(h)
        messages.append({"role": "user", "content": question})

        tools_enabled = True
        escalated = False          # 是否已经转人工。转人工是终态，不能被后续回答冲掉
        nudged = False             # 护栏 4 是否已经打回过一次
        retrieved = False          # 是否**成功**查过知识库或判例库
        final_answer = ""
        stopped_reason = "max_rounds"

        for round_no in range(1, self.max_rounds + 1):
            # ---- 1. 问模型 ----
            t0 = time.perf_counter()
            try:
                resp = chat(
                    messages,
                    tools=TOOL_SCHEMAS if tools_enabled else None,
                    temperature=self.temperature,
                    model=self.model,
                )
            except LLMError as e:
                trace.add_mark("error", round_no, error=str(e))
                trace.final_answer = final_answer
                trace.stopped_reason = "llm_error"
                trace.usage = self.meter.summary()
                return trace

            elapsed = int((time.perf_counter() - t0) * 1000)
            self.meter.add(resp.usage)
            trace.add_llm(round_no, resp.content, resp.tool_calls, resp.usage, elapsed)

            # ---- 2. 模型没有要调工具 = 它认为可以答了 ----
            if not resp.wants_tool:
                # 护栏 4 打回：规则类问题，一次资料都没查就想直接答。
                # 留一轮余地，不然最后一轮打回等于把它逼死在 max_rounds 上。
                if (
                    not escalated
                    and not nudged
                    and not retrieved
                    and round_no < self.max_rounds
                    and _RULE_Q_RE.search(question)
                ):
                    nudged = True
                    trace.add_mark(
                        "nudge",
                        round_no,
                        reason="规则类问题未检索就想直接回答",
                        drafted=resp.content,
                    )
                    messages.append({"role": "assistant", "content": resp.content or ""})
                    messages.append({"role": "user", "content": _NUDGE_TEXT})
                    continue

                final_answer = (resp.content or "").strip()
                if escalated:
                    # 已经转人工了：这一句只是收尾话术，不改「终态」的性质。
                    # 早期版本这里会被覆盖成 answered，导致转人工在指标里消失 —— 已修。
                    stopped_reason = "human"
                else:
                    stopped_reason = "answered" if final_answer else "empty_response"
                break

            # ---- 3. 轮数用尽却还想调工具：不再执行 ----
            # 执行了也没有下一轮去消费结果，纯属白花钱。
            # 这一条必须和 LangGraph 版保持一致 —— 两版行为可对齐，
            # 「换个编排方式」才不至于变成「换了个 Agent」。
            if round_no >= self.max_rounds:
                stopped_reason = "max_rounds"
                break

            messages.append(self._assistant_tool_call_message(resp))

            for call in resp.tool_calls:
                t1 = time.perf_counter()
                result = self.runtime.dispatch(call["name"], call["arguments"])
                tool_ms = int((time.perf_counter() - t1) * 1000)

                trace.add_tool(round_no, call["name"], call["arguments"], result, tool_ms)
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

                # 转人工成功 → 收回工具权限，下一轮逼它输出收尾话术
                if call["name"] == "transfer_to_human" and result.get("ok"):
                    tools_enabled = False
                    escalated = True
                    stopped_reason = "human"

                # 记「真的查到东西了」。护栏 4 只认成功，不认尝试 ——
                # 工具报错也算「查过」的话，知识库没加载时这条护栏就废了。
                if (
                    call["name"] in {"search_knowledge", "search_precedents"}
                    and result.get("ok")
                ):
                    retrieved = True

        trace.final_answer = final_answer
        trace.stopped_reason = stopped_reason
        trace.usage = self.meter.summary()
        return trace

    # ---------- 消息组装 ----------

    @staticmethod
    def _assistant_tool_call_message(resp) -> dict:
        """把模型的工具调用原样放回消息历史。

        OpenAI 的协议要求：assistant 那条带 tool_calls 的消息必须出现在
        对应的 tool 结果消息之前，否则接口会拒绝。少了这一条，整个会话就断了。
        """
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
