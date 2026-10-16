"""
[M8] 向量检索（Chroma）

先说一句实话：**在 21 条语料上，向量检索的效果和 BM25 差不多，甚至可能更差。**

那为什么还要上？两个理由，都不是「因为流行」：

    1. 判例库会长大。语料到几百上千条时，「意思相近但用词完全不同」的情况会变多
       （比如「钱退到哪」vs「款项原路返回的渠道」），那时候 BM25 的短板才真正暴露。
       现在把这一层留出来，将来换实现不用动上面的代码。

    2. 双路召回 + 重排是工业界检索的标配架构。能把这个讲清楚，
       比「我会调 Chroma 的 API」值钱得多。

所以这里的重点不是「我用了向量库」，而是「检索被抽象成了一层可替换的东西，
而且我知道什么时候该换」。

一个实现细节值得说明：这里**自己算 embedding 再传给 Chroma**，
用的不是 Chroma 的 EmbeddingFunction 回调。原因很实际 ——
那套回调接口在不同版本间改过签名，写死了容易被版本升级搞崩。
自己算完直接传向量，接口反而最稳。
"""

from __future__ import annotations

import requests

from src.config import (
    CHROMA_COLLECTION,
    EMBED_API_KEY,
    EMBED_BASE_URL,
    EMBED_MODEL,
    VECTOR_DIR,
)


class EmbeddingUnavailable(RuntimeError):
    """没有配 embedding 接口。

    调用方应该据此降级到纯 BM25 继续跑，而不是把整条链路崩掉 ——
    检索是一种增强，不该成为单点故障。
    """


def embed_texts(texts: list[str]) -> list[list[float]]:
    """调 OpenAI 兼容的 embeddings 接口。

    智谱、OpenAI、以及大多数国内厂商都遵循这个格式：
        POST {base}/embeddings  {"model": ..., "input": [...]}
    """
    if not EMBED_API_KEY:
        raise EmbeddingUnavailable("未配置 EMBED_API_KEY，向量检索不可用")

    url = f"{EMBED_BASE_URL.rstrip('/')}/embeddings"
    resp = requests.post(
        url,
        json={"model": EMBED_MODEL, "input": texts},
        headers={
            "Authorization": f"Bearer {EMBED_API_KEY}",
            "Content-Type": "application/json",
        },
        timeout=30,
    )
    resp.raise_for_status()

    data = resp.json()["data"]
    # 按 index 排一遍：接口不保证返回顺序和输入顺序一致，
    # 不排的话会出现「文档和向量错位」这种极难排查的问题
    data.sort(key=lambda d: d.get("index", 0))
    return [d["embedding"] for d in data]


class VectorStore:
    """判例库的向量索引，本地持久化。"""

    def __init__(
        self,
        persist_dir=None,
        collection_name: str | None = None,
    ) -> None:
        self.persist_dir = persist_dir or VECTOR_DIR
        self.collection_name = collection_name or CHROMA_COLLECTION
        self._client = None
        self._collection = None

    # ---------- 状态 ----------

    @property
    def configured(self) -> bool:
        """有没有配好 embedding 接口。"""
        return bool(EMBED_API_KEY)

    def _get_collection(self):
        if self._collection is None:
            import chromadb

            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                # 用余弦距离：我们的向量没做归一化，余弦比欧氏更适合文本相似度
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def count(self) -> int:
        try:
            return self._get_collection().count()
        except Exception:  # noqa: BLE001
            return 0

    # ---------- 建索引 ----------

    def build(self, records: list[dict], reset: bool = False) -> int:
        """把判例写进向量库，返回写入条数。

        reset=True 时先清空重建 —— 语料很小，重建比增量更新更省心，
        也更不容易出现「库存和代码不一致」的脏数据。
        """
        if not self.configured:
            raise EmbeddingUnavailable("未配置 EMBED_API_KEY")

        collection = self._get_collection()

        if reset:
            try:
                self._client.delete_collection(self.collection_name)
                self._collection = None
                collection = self._get_collection()
            except Exception:  # noqa: BLE001
                pass

        # 检索文本用「场景 + 问句」拼 —— 场景词的判别力很强，
        # 能帮向量把「改单」和「退款」这类相邻话题分开
        docs = [f"{r.get('scene', '')} {r.get('question', '')}" for r in records]
        vectors = embed_texts(docs)

        collection.add(
            ids=[str(r["id"]) for r in records],
            documents=docs,
            embeddings=vectors,
            metadatas=[
                {
                    "case_id": str(r["id"]),
                    "scene": r.get("scene", ""),
                    "question": r.get("question", ""),
                }
                for r in records
            ],
        )
        return len(records)

    # ---------- 检索 ----------

    def search(
        self,
        query: str,
        k: int = 8,
        exclude_id: str | int | None = None,
        scene: str | None = None,
    ) -> list[dict]:
        """语义召回。

        排除自身的规则和 BM25 那边完全一致 —— 判例库里有这条问题的标准答法，
        不排除掉就是在抄答案。防泄漏这件事，两条检索路径都要做，漏一条就白防。
        """
        if not self.configured or self.count() == 0:
            return []

        collection = self._get_collection()

        where: dict | None = None
        clauses = []
        if exclude_id is not None:
            clauses.append({"case_id": {"$ne": str(exclude_id)}})
        if scene:
            clauses.append({"scene": scene})
        if len(clauses) == 1:
            where = clauses[0]
        elif len(clauses) > 1:
            where = {"$and": clauses}

        try:
            res = collection.query(
                query_embeddings=embed_texts([query]),
                n_results=k,
                where=where,
            )
        except Exception:  # noqa: BLE001
            # 过滤条件把结果过滤空了、或向量库出问题 —— 都不该让 Agent 崩掉
            return []

        out = []
        ids = (res.get("ids") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        for i, cid in enumerate(ids):
            meta = metas[i] if i < len(metas) else {}
            dist = dists[i] if i < len(dists) else None
            out.append(
                {
                    "case_id": str(cid),
                    # 余弦距离 → 相似度。分数只在同一次查询里可比，不要跨查询比大小
                    "score": round(1 - dist, 4) if dist is not None else None,
                    "scene": (meta or {}).get("scene", ""),
                    "source": "vector",
                }
            )
        return out
