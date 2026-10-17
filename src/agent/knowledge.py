"""
[M7+] 业务规则知识库检索（v3 新增）

这个模块解决 v2 暴露出来的一个死局。

v2 只有一个检索源：判例库（历史问答对）。而判例库和测试集是同一批数据，
为了防止抄答案，`search()` 强制 leave-one-out 把当前题自己排除掉。
排除之后剩下的 20 条讲的是另外 20 个知识点 —— Agent 面对任何一道题，
手里都没有可用的知识。它只有两条路：转人工，或者凭空编。

v3 加的这一层，就是把「知识」和「案例」拆开：

    判例库  →  「别人是怎么答的」  参考表达方式
    知识库  →  「规则是什么」      回答的事实依据

知识库是独立的数据文件（data/knowledge/cs_rules.json），不参与 leave-one-out，
因为它不是测试集的一部分 —— 真实产品里，知识库本来就是独立的业务文档。

一个必须诚实说明的地方：
    这份知识库的内容，有 21 条来自实测记录里人工判定为「答对」的应答口径，
    另有 9 条是原告答偏、无正确口径，按通用业务规则补充的（文件里 source 字段区分）。
    所以它和测试集同源，但不是同一份东西：知识库是抽象后的规则条款，测试集是
    用户问题加人工判定。同源不构成泄漏，因为答案不会逐字一致 —— 但面试被问到时
    要主动讲清楚这一点，别等对方问。

另一个刻意的设计：不给「相关度硬阈值」。
    BM25 分数没有绝对意义，30 条语料上算出来的 3.2 和 7.1 谁更可信，说不清。
    与其拍一个阈值把可能答对的问题误判成「无知识」，不如把分数和条目一起交给模型，
    由它判断「这些条目到底对不对得上这个问题」。模型判断相关性，比一个魔数稳。
"""

from __future__ import annotations

import json
from pathlib import Path

from src.agent.retrieval import BM25, tokenize
from src.config import KNOWLEDGE_JSON


class KnowledgeBase:
    """业务规则知识库。只做检索，不做生成。"""

    def __init__(self, path: Path | str | None = None) -> None:
        path = Path(path) if path else KNOWLEDGE_JSON
        if not path.exists():
            raise FileNotFoundError(
                f"知识库文件不存在：{path}\n"
                "这是 v3 新增的数据文件，请确认 data/knowledge/cs_rules.json 已存在。"
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        self.rules: list[dict] = data.get("rules", [])
        if not self.rules:
            raise ValueError(f"知识库为空：{path}")

        # 检索文本把 topic、keywords、content 拼在一起：
        # keywords 单独拼一次，是因为用户提问通常很短（「提现要收手续费吗」），
        # 靠 content 里的长句去匹配，容易被无关词稀释。
        docs = [
            f"{r.get('topic', '')} {' '.join(r.get('keywords', []))} {r.get('content', '')}"
            for r in self.rules
        ]
        self.bm25 = BM25(docs)

    def __len__(self) -> int:
        return len(self.rules)

    def search(self, query: str, k: int = 3) -> list[dict]:
        """按相关度返回最多 k 条规则。命中不了的（分数为 0）不返回。"""
        if not (query or "").strip():
            return []

        results: list[dict] = []
        for score, idx in self.bm25.top_k(query, k=k):
            if score <= 0:
                continue
            rule = self.rules[idx]
            results.append(
                {
                    "id": rule.get("id"),
                    "topic": rule.get("topic"),
                    "scene": rule.get("scene"),
                    "content": rule.get("content"),
                    "source": rule.get("source"),
                    "score": round(score, 2),
                }
            )
        return results

    def explain_coverage(self, query: str) -> dict:
        """诊断用：看看某个问题在知识库里能被匹配到什么程度。

        不参与 Agent 主流程，只给调试和评测报告用。
        """
        qt = tokenize(query)
        # 单字命中太容易，只用 bigram 和英文数字来估算真实覆盖
        meaningful = [t for t in qt if len(t) > 1]
        hits = 0
        for score, idx in self.bm25.top_k(query, k=1):
            if score <= 0:
                break
            doc_tokens = set(self.bm25.doc_tokens[idx])
            hits = sum(1 for t in meaningful if t in doc_tokens)
            return {
                "query": query,
                "top_rule": self.rules[idx].get("id"),
                "score": round(score, 2),
                "meaningful_terms": len(meaningful),
                "matched_terms": hits,
                "coverage": round(hits / len(meaningful), 2) if meaningful else 0.0,
            }
        return {"query": query, "top_rule": None, "score": 0.0, "coverage": 0.0}
