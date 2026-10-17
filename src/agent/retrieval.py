"""
[M7] 判例检索（BM25 + 中文 bigram）

Agent 需要一个「记得住历史标准答法」的能力：给它用户的问题，它能把最像的
几条历史案例捞出来当参考。这就是 RAG —— 只不过语料不是产品文档，
而是你自己实测攒下来的问答对。

两个必须讲清楚的设计点：

一、为什么用 BM25 而不是向量检索？
    语料只有 30 条，全是中文短问句。向量检索的强项是「意思相近但用词完全不同」，
    但要在 30 条上验证这个强项，样本量根本不够；还要多背一个 embedding 模型和
    每次检索的 API 成本。BM25 在这个规模上是更诚实的选择。
    真要换向量，替换点只有 PrecedentIndex 一个类 —— 这就是把检索单独成模块的意义。

二、为什么必须有 leave-one-out？
    判例库来自「答对」的样本，而测试集用的是同一批问题。评第 N 条时如果不把
    第 N 条自己从检索池里排除掉，Agent 就是在抄答案，评测数字全是假的。
    所以 search() 强制带 exclude_id，且不给默认值 —— 想让调用方忘都忘不掉。
"""

from __future__ import annotations

import math
import re
from collections import Counter

# 英文 / 数字整体保留，汉字逐字切（bigram 在 tokenize 里拼）
_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]")


def _is_cjk(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def tokenize(text: str) -> list[str]:
    """中文按「单字 + 相邻 bigram」切，英文数字整体保留。

    不上 jieba：多一个依赖、多一份词典，而这里只有 30 条短问句。
    单字 + bigram 已经足够把「退款 / 退回」「多久 / 多长时间」这类表达匹配上，
    而且不会因为分词器的词典没覆盖到金融术语而漏掉。
    """
    units = _TOKEN_RE.findall((text or "").lower())
    tokens: list[str] = []
    for i, u in enumerate(units):
        tokens.append(u)
        # 只把相邻的两个汉字拼成 bigram，避免出现「款2」这种跨类型组合
        if i + 1 < len(units) and _is_cjk(u) and _is_cjk(units[i + 1]):
            tokens.append(u + units[i + 1])
    return tokens


# ============================================================
# 一、BM25
# ============================================================


class BM25:
    """标准 BM25（k1=1.5, b=0.75），自己实现，不引第三方库。"""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.n = len(docs)
        self.doc_tokens = [tokenize(d) for d in docs]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avg_len = sum(self.doc_len) / self.n if self.n else 0.0
        self.tf = [Counter(t) for t in self.doc_tokens]

        self.df: Counter = Counter()
        for toks in self.doc_tokens:
            for t in set(toks):
                self.df[t] += 1

    def idf(self, term: str) -> float:
        """加 0.5 平滑的 IDF，保证任何词都不会算出负值。"""
        df = self.df.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def score(self, query_tokens: list[str], idx: int) -> float:
        dl = self.doc_len[idx] or 1
        norm = self.k1 * (1 - self.b + self.b * dl / (self.avg_len or 1))
        s = 0.0
        for t in query_tokens:
            f = self.tf[idx].get(t, 0)
            if not f:
                continue
            s += self.idf(t) * f * (self.k1 + 1) / (f + norm)
        return s

    def top_k(
        self, query: str, k: int = 3, allowed: set[int] | None = None
    ) -> list[tuple[float, int]]:
        """返回 [(得分, 文档下标)]，按得分降序。allowed 为 None 表示不限制。"""
        qt = tokenize(query)
        scored = [
            (self.score(qt, i), i)
            for i in range(self.n)
            if allowed is None or i in allowed
        ]
        scored.sort(key=lambda x: -x[0])
        return scored[:k]


# ============================================================
# 二、判例库
# ============================================================


class PrecedentIndex:
    """把历史问答对做成可检索的判例库。

    默认只收「人工判定为答对」的案例 —— 判例库的价值在于「这是对的做法」，
    把错例也放进去，等于让 Agent 从错误里找参考。
    """

    def __init__(self, records: list[dict], only_correct: bool = True) -> None:
        self.only_correct = only_correct
        self.cases = [
            r for r in records if not only_correct or r.get("human_label") == "答对"
        ]
        # 检索文本用「场景 + 问句」拼——场景是很有判别力的一个词，
        # 比如「改单」能帮模型避开退款类判例
        docs = [f"{c.get('scene', '')} {c.get('question', '')}" for c in self.cases]
        self.bm25 = BM25(docs)

    def search(
        self,
        query: str,
        exclude_id: int | None,
        k: int = 3,
        scene: str | None = None,
        min_score: float = 0.0,
    ) -> list[dict]:
        """检索最像的 k 条判例。

        exclude_id 必须由调用方显式传入（评测本轮问题时要排除它自己）。
        这里不给默认值，就是为了让「忘了防泄漏」这件事在代码上不可能发生。

        注：这个参数原来带 `= None`，但注释写着「不给默认值」—— 注释和行为对不上。
        是 `scripts/selfcheck.py` 把这条不一致抓出来的，后来把参数顺序调到 k 前面去掉了默认值。
        现在真忘了传会直接 TypeError，而不是悄悄泄漏答案。

        scene 过滤如果过滤到一条都不剩，就自动放宽 —— 宁可返回跨场景的参考，
        也不要让 Agent 两手空空。
        """
        allowed = {
            i
            for i, c in enumerate(self.cases)
            if c.get("id") != exclude_id
            and (scene is None or c.get("scene") == scene)
        }
        if not allowed:
            allowed = {
                i for i, c in enumerate(self.cases) if c.get("id") != exclude_id
            }

        hits = []
        for score, idx in self.bm25.top_k(query, k=k, allowed=allowed):
            if score <= min_score:
                continue
            case = self.cases[idx]
            hits.append(
                {
                    "id": case.get("id"),
                    "scene": case.get("scene"),
                    "question": case.get("question"),
                    "answer": case.get("answer"),
                    "score": round(score, 3),
                }
            )
        return hits

    def __len__(self) -> int:
        return len(self.cases)


def build_index(records: list[dict], only_correct: bool = True) -> PrecedentIndex:
    """从 records.json 直接建索引。"""
    return PrecedentIndex(records, only_correct=only_correct)
