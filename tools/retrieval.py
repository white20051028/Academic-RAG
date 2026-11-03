"""混合检索：向量 + BM25 + RRF。

为什么要加 BM25：
  纯向量检索会把"关键词"的权重视作语义的一部分，容易被长文档稀释。
  实测失败案例：「补考没过会不会影响拿学位」→ 召回到辅修管理办法
  （"学位"这个词的引力压过了"补考"）。BM25 靠字面匹配，会给"补考"更高的权重。

为什么用 RRF 而不是加权求和：
  两路的分数根本不可比 —— 向量是 cosine 相似度（0~1），
  BM25 是无上界的对数分数。直接加权是错的，得先归一化；
  而 RRF 只用**排名**，天然绕开量纲问题：score = Σ 1/(k + rank)。
"""

import jieba
import numpy as np

RRF_K = 60          # RRF 的标准取值，一般不用调
POOL = 20           # 每路先各取 20 个候选，再融合


class HybridRetriever:
    """向量 + BM25 双路召回，RRF 融合。"""

    def __init__(self, collection, embed_client, embed_model="BAAI/bge-m3"):
        from rank_bm25 import BM25Okapi

        self.col = collection
        self.client = embed_client
        self.embed_model = embed_model

        data = collection.get(include=["documents", "metadatas"])
        self.ids: list[str] = data["ids"]
        self.docs: list[str] = data["documents"]
        self.metas: list[dict] = data["metadatas"]
        self.id2doc = dict(zip(self.ids, self.docs))
        self.id2meta = dict(zip(self.ids, self.metas))

        # BM25 索引：jieba 分词（词典首次加载约 1 秒）
        self.tokens = [list(jieba.cut(d)) for d in self.docs]
        self.bm25 = BM25Okapi(self.tokens)

    @property
    def size(self) -> int:
        return len(self.ids)

    def _vector_ids(self, query: str, pool: int) -> tuple[list[str], float]:
        """向量路。返回 (候选 id 列表, top-1 的相似度)。"""
        vec = self.client.embeddings.create(
            model=self.embed_model, input=[query]
        ).data[0].embedding
        r = self.col.query(
            query_embeddings=[vec], n_results=min(pool, len(self.ids)),
            include=["distances"],
        )
        ids = r["ids"][0]
        best_sim = 1 - r["distances"][0][0] if r["distances"][0] else 0.0
        return ids, best_sim

    def _bm25_ids(self, query: str, pool: int) -> list[str]:
        """BM25 路。"""
        qtokens = list(jieba.cut(query))
        scores = self.bm25.get_scores(qtokens)
        top_idx = np.argsort(scores)[::-1][:pool]
        return [self.ids[i] for i in top_idx if scores[i] > 0]

    def search(self, query: str, k: int = 5, pool: int = POOL,
               rewrite: bool = False) -> tuple[list[dict], float]:
        """返回 (融合后的前 k 条, 向量 top-1 相似度)。

        相似度单独返回，因为拒答判断要用它 —— RRF 分数没有绝对尺度，
        不能说"RRF > 0.02 就可信"。

        rewrite=True 时：向量路用原查询（保语义），BM25 路用改写后的书面词
        （保字面匹配）。两路各用各的长处。
        """
        vec_ids, best_sim = self._vector_ids(query, pool)

        bm25_query = query
        if rewrite:
            from rewrite import rewrite_query
            bm25_query = rewrite_query(query, self.client)
        bm25_ids = self._bm25_ids(bm25_query, pool)

        scores: dict[str, float] = {}
        for rank, did in enumerate(vec_ids, 1):
            scores[did] = scores.get(did, 0.0) + 1.0 / (RRF_K + rank)
        for rank, did in enumerate(bm25_ids, 1):
            scores[did] = scores.get(did, 0.0) + 1.0 / (RRF_K + rank)

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
        results = [
            {
                "id": did,
                "doc": self.id2doc[did],
                "meta": self.id2meta[did],
                "rrf": s,
            }
            for did, s in ranked
        ]
        return results, best_sim
