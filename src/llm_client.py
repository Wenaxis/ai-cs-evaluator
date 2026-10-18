"""
[M7] 统一大模型调用层（v2 新增）

v1 里 llm_judge.py 自己带了一份 requests 调用 —— 能用，但只够「发一段文本、收一段文本」。
v2 的客服 Agent 需要两件 v1 没有的能力：

    1. function calling —— 把工具清单连同对话一起发给模型，让它自己决定调哪个、传什么参数。
       这是「agent」和「套了 prompt 的脚本」之间的分界线。
    2. 用量统计 —— Agent 一轮要调好几次模型，上下文一轮轮变长。
       不记 token，就没法回答「你这个 Agent 跑一遍多少钱」。

把「怎么调模型」抽成一层，业务代码里就不再出现 requests。
llm_judge.py 保持原样不动 —— 它是 v1 的基线，基线不重构。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import requests

from src.config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MAX_RETRY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
)


class LLMError(RuntimeError):
    """模型调用彻底失败（重试到上限仍然不成）。"""


@dataclass
class LLMResponse:
    """一次模型调用的统一产物。

    注意 content 允许是 None：模型决定调工具时，content 往往就是空的，
    这时候唯一有意义的字段是 tool_calls。
    """

    content: str | None = None
    tool_calls: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    finish_reason: str = ""

    @property
    def wants_tool(self) -> bool:
        return bool(self.tool_calls)


# ============================================================
# 一、调用
# ============================================================


def chat(
    messages: list[dict],
    tools: list[dict] | None = None,
    temperature: float | None = None,
    force_json: bool = False,
    model: str | None = None,
) -> LLMResponse:
    """发一次对话请求。

    参数：
        messages      OpenAI 格式的消息列表
        tools         工具 schema 列表；传了就开启 function calling
        force_json    要求接口层返回合法 JSON（评审类调用用得上）
        model         临时指定别的模型（做「换模型复跑」的消融实验）
    """
    if not LLM_API_KEY:
        raise LLMError("没有读到 LLM_API_KEY，请先在项目根目录的 .env 里配置")

    payload = {
        "model": model or LLM_MODEL,
        "messages": messages,
        # 温度默认 0：评分和工具选择都要可复现，不要「创造力」
        "temperature": LLM_TEMPERATURE if temperature is None else temperature,
    }
    if tools:
        payload["tools"] = tools
        # auto = 先调工具还是先说话，交给模型自己判断。
        # 这一步就是「没有工具选择」和「有工具选择」的差别 —— v1 是没有的。
        payload["tool_choice"] = "auto"
    if force_json:
        payload["response_format"] = {"type": "json_object"}

    data = _post_with_retry(payload)
    choice = data["choices"][0]
    message = choice.get("message") or {}

    return LLMResponse(
        content=message.get("content"),
        tool_calls=normalize_tool_calls(message.get("tool_calls") or []),
        usage=data.get("usage") or {},
        finish_reason=choice.get("finish_reason", ""),
    )


def _post_with_retry(payload: dict, max_retry: int | None = None) -> dict:
    """POST 到 chat/completions，带退避重试。"""
    max_retry = max_retry or LLM_MAX_RETRY
    url = f"{LLM_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }

    last_err: Exception | None = None
    for attempt in range(1, max_retry + 1):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=LLM_TIMEOUT)

            # 429 是限流、5xx 是服务端抖动 —— 这两种等一下再来是有意义的。
            # 其余 4xx 是我们自己请求写错了（比如模型名不存在），重试一百次也一样。
            if resp.status_code == 429 or resp.status_code >= 500:
                raise requests.HTTPError(f"HTTP {resp.status_code}: {resp.text[:200]}")

            resp.raise_for_status()
            return resp.json()
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < max_retry:
                time.sleep(1.5 * attempt)  # 退避：1.5s、3s、4.5s

    raise LLMError(f"重试 {max_retry} 次仍失败：{last_err}")


# ============================================================
# 二、工具调用结果的规整
# ============================================================


def normalize_tool_calls(raw_calls: list[dict]) -> list[dict]:
    """把模型的工具调用规整成统一形状，并顺手把 arguments 解析掉。

    模型给回来的是字符串 '{"order_ref": "2024..."}'，不是 dict。
    留着字符串的话，下游每个工具都要自己 json.loads 一遍，还各自处理一次解析失败。
    不如在这儿一次做完。

    解析失败不抛异常 —— 把原文塞进 _raw 还给它。这样 Agent 下一轮能「看见」
    自己刚才传了什么烂参数，有机会自己纠回来。这就是兜底，不是崩溃。
    """
    calls = []
    for c in raw_calls:
        fn = c.get("function") or {}
        args_text = fn.get("arguments") or "{}"
        try:
            args = json.loads(args_text)
        except json.JSONDecodeError:
            args = {"_raw": args_text, "_parse_error": "arguments 不是合法 JSON"}

        calls.append(
            {
                "id": c.get("id", ""),
                "name": fn.get("name", ""),
                "arguments": args if isinstance(args, dict) else {"_value": args},
            }
        )
    return calls


# ============================================================
# 三、用量统计
# ============================================================


class UsageMeter:
    """累计一次完整运行里的 token 消耗。

    Agent 跑一轮要调好几次模型，单次调用的 usage 散在各处没有意义。
    被问「跑一遍 30 条多少钱」时，需要一个能直接报总数的东西。
    """

    def __init__(self) -> None:
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def add(self, usage: dict) -> None:
        self.calls += 1
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def summary(self) -> dict:
        return {
            "llm_calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }
