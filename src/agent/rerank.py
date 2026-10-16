"""
[M9] 重排（Rerank）

双路召回之后会拿到十几条候选，里面一定有噪声。

向量检索尤其容易捞进「话题相关、但答的不是用户问的那件事」的候选。举个例子：
用户问「自动续费扣的钱能退吗」，向量很可能把「怎么关闭自动续费」排到前面 ——
两者词面高度相似，但回答的重点完全不是一回事。**这正是原始客服第 5 条答偏的原因，
也是判例检索最危险的地方：它可能把模板答案送到 Agent 面前，诱导它犯同样的错。**

重排就是在这批候选上再精筛一遍。这里用 LLM 做（cross-encoder 的思路），理由：

    · 本地跑 cross-encoder 要再装一套模型权重，而候选只有十几条
    · 一次 API 调用就能排完，成本可控
    · 顺带拿到「为什么这条排前面」的理由 —— 调检索的时候，这一点极其有用

代价是慢和贵。所以**只在候选集上跑一次，绝不拿它扫全库**。
这正是「召回 + 重排」两级架构存在的理由：粗筛用便宜的，精排用贵的。

降级策略：没配 Key、或解析失败，就退回原顺序 —— 重排是加分项，不是必要条件。
"""

from __future__ import annotations

import json
import re

from src.config import RERANK_TOP_K
from src.llm_client import LLMError, chat

RERANK_SYSTEM = (
    "你是一个检索结果排序器。给你一个用户问题和一批候选参考案例，"
    "按「这条案例对回答该问题有多大帮助」从高到低排序。"
    "注意区分「话题相近」和「真的能回答这个问题」—— 前者不算相关。"
    "你只输出 JSON。"
)


def _build_prompt(query: str, candidates: list[dict]) -> str:
    listing = "\n".join(
        f"[{c['case_id']}] 问：{c.get('question', '')}\n"
        f"      标准答法：{(c.get('answer') or '')[:120]}"
        for c in candidates
    )
    return f"""用户问题：{query}

候选案例：
{listing}

请按相关性从高到低排序，只输出 JSON，不要解释：
{{"ranking": ["最相关的 case_id", "次相关", "..."]}}

要求：
1. 只出现上面列表里的 case_id，不要编造。
2. 全部候选都要出现在排序里。
3. 判断标准是「这条案例能不能帮我回答用户问的那件事」，
   而不是「话题像不像」。比如用户问「能不能退」，一条讲「怎么关闭」的案例
   尽管话题相近，也应该排在后面。"""


def _parse_ranking(text: str, valid_ids: list[str]) -> list[str] | None:
    """从模型输出里抠出排序列表，并过滤掉它编出来的 id。"""
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None

    try:
        raw = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None

    ranking = raw.get("ranking") or []
    valid = [cid for cid in ranking if cid in valid_ids]

    # 模型漏掉的候选，按原顺序补到后面 ——
    # 宁可保留一条可能不相关的，也不要因为模型漏写就整条丢掉
    for cid in valid_ids:
        if cid not in valid:
            valid.append(cid)
    return valid


def llm_rerank(
    query: str,
    candidates: list[dict],
    top_k: int = RERANK_TOP_K,
    model: str | None = None,
) -> list[dict]:
    """用 LLM 对候选重排，返回前 top_k 条。

    candidates 每项需含 case_id；会原样返回并附加 rank_after 字段。
    """
    if len(candidates) <= 1:
        return candidates[:top_k]

    valid_ids = [c["case_id"] for c in candidates]
    by_id = {c["case_id"]: c for c in candidates}

    try:
        resp = chat(
            [
                {"role": "system", "content": RERANK_SYSTEM},
                {"role": "user", "content": _build_prompt(query, candidates)},
            ],
            force_json=True,
            model=model,
        )
        ranking = _parse_ranking(resp.content or "", valid_ids)
    except (LLMError, Exception):  # noqa: BLE001
        # 重排失败不影响主流程 —— 退回原顺序继续
        ranking = None

    if not ranking:
        return candidates[:top_k]

    out = []
    for i, cid in enumerate(ranking[:top_k]):
        item = dict(by_id[cid])
        item["rank_after"] = i + 1
        out.append(item)
    return out
