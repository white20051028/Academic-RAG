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

import json
import re

import jieba
import numpy as np

RRF_K = 60          # RRF 的标准取值，一般不用调
POOL = 20           # 每路先各取 20 个候选，再融合


class HybridRetriever:
    """向量 + BM25 双路召回，RRF 融合。

    数据来源：data/chunks.json（文本+meta）+ data/embeddings.npy（归一化向量）。
    **不用向量数据库。**

    为什么：
      1070 条 × 1024 维，numpy 点积一次约 1ms，完全够用；
      而 Chroma 1.5.9 在这个量级会把 HNSW 索引写坏
      （InternalError: Error loading hnsw index，分批写和重建都修不好）。
      **工具要匹配规模** —— 1000 条的数据不需要向量数据库。
    """

    def __init__(self, embed_client, embed_model="BAAI/bge-m3", data_dir=None):
        from pathlib import Path

        import numpy as np
        from rank_bm25 import BM25Okapi

        self.client = embed_client
        self.embed_model = embed_model

        if data_dir is None:
            data_dir = Path(__file__).resolve().parent.parent / "data"
        data_dir = Path(data_dir)

        rows = json.loads((data_dir / "chunks.json").read_text(encoding="utf-8"))
        self.ids = [r["id"] for r in rows]
        self.docs = [r["doc"] for r in rows]
        self.metas = [r["meta"] for r in rows]

        # 已归一化，点积即余弦相似度
        self.vectors = np.load(data_dir / "embeddings.npy")
        if len(self.vectors) != len(self.ids):
            raise RuntimeError(
                f"向量数 {len(self.vectors)} 和 chunk 数 {len(self.ids)} 不一致，"
                f"请重跑 tools/build_index.py"
            )

        self.id2doc = dict(zip(self.ids, self.docs))
        self.id2meta = dict(zip(self.ids, self.metas))

        # BM25 索引：jieba 分词（词典首次加载约 1 秒）
        self.tokens = [list(jieba.cut(d)) for d in self.docs]
        self.bm25 = BM25Okapi(self.tokens)

    @property
    def size(self) -> int:
        return len(self.ids)

    def _vector_ids(self, query: str, pool: int) -> tuple[list[int], float]:
        """向量路：numpy 点积。返回 (候选下标列表, top-1 的相似度)。

        embeddings.npy 里的向量已经归一化，所以点积 = 余弦相似度。
        """
        import numpy as np

        vec = self.client.embeddings.create(
            model=self.embed_model, input=[query]
        ).data[0].embedding
        q = np.asarray(vec, dtype="float32")
        q /= np.linalg.norm(q) + 1e-9

        sims = self.vectors @ q                       # (N,)
        top_idx = np.argsort(sims)[::-1][:pool]
        best_sim = float(sims[top_idx[0]]) if len(top_idx) else 0.0
        return [int(i) for i in top_idx], best_sim

    def _bm25_ids(self, query: str, pool: int) -> list[int]:
        """BM25 路。返回下标。"""
        import numpy as np

        qtokens = list(jieba.cut(query))
        scores = self.bm25.get_scores(qtokens)
        top_idx = np.argsort(scores)[::-1][:pool]
        return [int(i) for i in top_idx if scores[i] > 0]

    def search(self, query: str, k: int = 5, pool: int = POOL,
               rewrite: bool = False,
               vec_mode: str = "orig") -> tuple[list[dict], float]:
        """返回 (融合后的前 k 条, 向量 top-1 相似度)。

        相似度单独返回，因为拒答判断要用它 —— RRF 分数没有绝对尺度，
        不能说"RRF > 0.02 就可信"。

        rewrite=True：先做一次查询改写（口语 → 书面），改写结果给 BM25 路用。

        vec_mode 控制**向量路**用什么查询：
          "orig"    原查询（默认，保语义）
          "rewrite" 改写后的查询（保字面命中）
          "both"    两路都跑，合并候选

        为什么要这个参数：文档里"保研"出现 0 次、"推免"2 次。
        向量路用原查询"保研要求"时，压根匹配不上推免方案；
        而 BM25 路用改写后的"推荐免试研究生"能匹配上。
        结果是向量路拖后腿——所以得实测三种模式哪种最好。
        """
        need_rewrite = rewrite or vec_mode in ("rewrite", "both")
        rq = None
        if need_rewrite:
            from rewrite import rewrite_query
            rq = rewrite_query(query, self.client)

        # ---- 向量路 ----
        if vec_mode == "rewrite" and rq:
            vec_ids, best_sim = self._vector_ids(rq, pool)
        elif vec_mode == "both" and rq:
            v1, sim1 = self._vector_ids(query, pool)
            v2, sim2 = self._vector_ids(rq, pool)
            best_sim = max(sim1, sim2)
            vec_ids = list(dict.fromkeys(list(v1) + list(v2)))   # 去重保序
        else:
            vec_ids, best_sim = self._vector_ids(query, pool)

        # ---- BM25 路 ----
        bm25_query = rq or query
        bm25_ids = self._bm25_ids(bm25_query, pool)

        # 加权 RRF。**不是等权** —— 实测等权会输。
        #
        # 原因：数据量从 535 涨到 1084 后，BM25 的噪音变多。
        # 一个无关 chunk 只要在 BM25 路排第 1、向量路排第 2，
        # 就能拿到 1/61 + 1/62 = 0.0325；
        # 而正确答案只有向量路的 1/61 = 0.0164。
        # 噪音一路累加就把它挤掉了 —— 实测 Recall@5 从 96% 掉到 3.64%。
        #
        # 向量路用的是原查询（语义准），BM25 路用的是改写后关键词（字面召回）。
        # 语义比字面可信，所以给向量更高的权重。
        VEC_W, BM25_W = 1.0, 0.4

        scores: dict[int, float] = {}
        for rank, idx in enumerate(vec_ids, 1):
            scores[idx] = scores.get(idx, 0.0) + VEC_W / (RRF_K + rank)
        for rank, idx in enumerate(bm25_ids, 1):
            scores[idx] = scores.get(idx, 0.0) + BM25_W / (RRF_K + rank)

        # 按文件去重：每个来源文件最多留 MAX_PER_FILE 个 chunk。
        #
        # 为什么必须去重：RRF 奖励的是「数量」不是「质量」。
        # 「智能开采特色班招生通知」被切成 20 个 chunk，
        # BM25 路召回其中 10 个 → 这 10 个各拿一份分，全挤在前面，
        # 把真正的答案（向量 rank 1，只占一个 chunk）顶掉。
        # 实测：不去重 Recall@5 = 3.64%，去重后应回到 90%+。
        MAX_PER_FILE = 2

        # ---- 年份降权 ----
        #
        # 同一主题常有多个年份的版本：2021 / 2023 / 2025 / 2026 四份《转专业通知》。
        # 它们在语义上几乎一样，会互相稀释 top-5，把最新版本的关键段落挤出去。
        #
        # 实测：「我大二，绩点2.8，能转专业吗」
        #   top-5 被四个年份各占一条，全是不同段落，
        #   而「平均学分绩点不低于 3.00」这条就在 2026 版里，没进来。
        #   结果模型回答「资料里没有明确规定绩点的具体要求」——资料其实有。
        #
        # 修法：同主题只让最新年份拿全分，旧版本乘 0.5。
        # **降权不删除**：万一新版本没覆盖某个条款，旧版本还有机会被召回。
        best_year: dict[str, int] = {}
        for idx in scores:
            src = self.metas[idx].get("source", "")
            t = _topic_key(src)
            best_year[t] = max(best_year.get(t, 0), _year_of(src))

        ranked = []
        for idx, s in scores.items():
            src = self.metas[idx].get("source", "")
            y = _year_of(src)
            if y and y < best_year.get(_topic_key(src), y):
                s = s * 0.5
            ranked.append((idx, s))
        ranked.sort(key=lambda kv: -kv[1])

        picked, per_file = [], {}
        for idx, s in ranked:
            src = self.metas[idx].get("source", "")
            if per_file.get(src, 0) >= MAX_PER_FILE:
                continue
            per_file[src] = per_file.get(src, 0) + 1
            picked.append((idx, s))
            if len(picked) >= k:
                break

        results = [
            {
                "id": self.ids[idx],
                "doc": self.docs[idx],
                "meta": self.metas[idx],
                "rrf": s,
            }
            for idx, s in picked
        ]
        return results, best_sim

def _year_of(source: str) -> int:
    """从文件名里抠出年份。没有年份返回 0。"""
    m = re.search(r"(20\d{2})", source)
    return int(m.group(1)) if m else 0


def _topic_key(source: str) -> str:
    """去掉编号和年份，得到主题。

    「6041_关于开展2025年度本科生转专业工作的通知.txt」
    「5272_关于开展2023年度本科生转专业工作的通知.txt」
    都变成「_关于开展年度本科生转专业工作的通知.txt」，视为同一主题的不同版本。
    """
    s = re.sub(r"^\d+_", "", source)
    s = re.sub(r"20\d{2}", "", s)
    return s
