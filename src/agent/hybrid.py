"""
[M10] 双路检索 + RRF 融合

一条查询同时走两条路召回，再把两边排名融合，最后重排：

    查询 ──┬─► BM25 召回（词面匹配）──┐
           └─► 向量召回（语义匹配）──┴─► RRF 融合 ──► LLM 重排 ──► top-k

**为什么用 RRF（Reciprocal Rank Fusion）融合，而不是把两边的分数加权平均？**

    BM25 的分数和余弦相似度的量纲完全不是一回事。直接加权得先做归一化，
    而归一化的方式本身又会影响结果 —— 调半天，收益还不确定。

    RRF 只看「排在第几名」，不看分数绝对值，天生免掉了跨量纲的问题：

        score(d) = Σ  1 / (k + rank_i(d))          k 取 60（业界常用值）

    代价是丢掉了分数里的一部分信息。所以它适合「候选不多、质量接近」的场景 ——
    我们这个场合正好是。这个取舍本身也是要能讲清楚的。

**为什么要有降级路径？**

    向量检索依赖外部 embedding 接口。接口挂了、没配额、没配 Key ——
    任何一种情况都不该让 Agent 停止工作。
    检索是增强能力，不是单点故障。所以没配 embedding 时就退回纯 BM25，
    并在返回结果里标明这次用的是哪条路。
"""

from __future__ import annotations

from src.agent.rerank import llm_rerank
from src.agent.retrieval import PrecedentIndex
from src.agent.vector_store import VectorStore
from src.config import RECALL_TOP_N, RERANK_TOP_K

RRF_K = 60


def rrf_fuse(rank_lists: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """把多路排名融合成一个分数表，返回按分数降序的 [(id, score)]。

    只吃排名、不吃原始分数 —— 这就是它能跨量纲工作的原因。
    """
    scores: dict[str, float] = {}
    for ranks in rank_lists:
        for rank, doc_id in enumerate(ranks):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


class HybridRetriever:
    """对外只需要一个 search()，内部自己决定走几条路。

    这一点很重要：上层（工具层、Agent）不应该知道下面用了 BM25 还是向量，
    更不该关心降级。换检索方案时，改动范围圈在这一个类里。
    """

    def __init__(
        self,
        records: list[dict],
        vector_store: VectorStore | None = None,
        use_rerank: bool = True,
        model: str | None = None,
    ) -> None:
        self.index = PrecedentIndex(records, only_correct=True)
        self.cases = self.index.cases
        self.by_id = {str(c["id"]): c for c in self.cases}

        self.vector_store = vector_store or VectorStore()
        self.use_rerank = use_rerank
        self.model = model

        self._vector_ready = False
        self.last_route = ""      # 记录这次实际走了哪条路，供 trace / 调试查看

    # ---------- 准备 ----------

    def prepare(self, rebuild: bool = False) -> dict:
        """建向量索引（如果配了 embedding）。

        返回一份状态说明 —— 调用方要能知道「向量那一路到底有没有生效」，
        否则评测时会分不清「检索差」是方法问题还是配置问题。
        """
        status = {
            "bm25_cases": len(self.cases),
            "vector_enabled": False,
            "vector_cases": 0,
            "reason": "",
        }

        if not self.vector_store.configured:
            status["reason"] = "未配置 EMBED_API_KEY，本次只走 BM25 单路召回"
            return status

        try:
            need_build = rebuild or self.vector_store.count() != len(self.cases)
            if need_build:
                n = self.vector_store.build(self.cases, reset=True)
            else:
                n = self.vector_store.count()
            self._vector_ready = True
            status["vector_enabled"] = True
            status["vector_cases"] = n
        except Exception as e:  # noqa: BLE001
            status["reason"] = f"向量索引构建失败，退回 BM25：{e}"

        return status

    @property
    def vector_enabled(self) -> bool:
        return self._vector_ready and self.vector_store.configured

    # ---------- 检索 ----------

    def search(
        self,
        query: str,
        k: int = RERANK_TOP_K,
        exclude_id: int | str | None = None,
        scene: str | None = None,
    ) -> list[dict]:
        """检索。签名和 PrecedentIndex.search 保持一致，两者可互换。"""
        bm25_hits = self.index.search(
            query=query, k=RECALL_TOP_N, exclude_id=exclude_id, scene=scene
        )
        bm25_ids = [str(h["id"]) for h in bm25_hits]

        vector_ids: list[str] = []
        if self.vector_enabled:
            vector_hits = self.vector_store.search(
                query=query, k=RECALL_TOP_N, exclude_id=exclude_id, scene=scene
            )
            vector_ids = [h["case_id"] for h in vector_hits]

        if not bm25_ids and not vector_ids:
            self.last_route = "none"
            return []

        # 只有一路有结果时，不做融合 —— RRF 是为了「合并两种意见」，
        # 只有一种意见时套公式纯属多余
        if not vector_ids:
            self.last_route = "bm25"
            fused_ids = bm25_ids
        elif not bm25_ids:
            self.last_route = "vector"
            fused_ids = vector_ids
        else:
            self.last_route = "hybrid"
            fused_ids = [cid for cid, _ in rrf_fuse([bm25_ids, vector_ids])]

        candidates = []
        for cid in fused_ids:
            case = self.by_id.get(cid)
            if case is None:
                continue
            candidates.append(
                {
                    "case_id": cid,
                    "id": case["id"],
                    "scene": case.get("scene", ""),
                    "question": case.get("question", ""),
                    "answer": case.get("answer", ""),
                }
            )

        if not candidates:
            return []

        if self.use_rerank and len(candidates) > 1:
            ranked = llm_rerank(query, candidates, top_k=k, model=self.model)
            self.last_route += "+rerank"
            return ranked

        return candidates[:k]

    def __len__(self) -> int:
        return len(self.cases)
