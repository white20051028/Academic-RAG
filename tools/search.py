"""检索入口：从「用户原话」到「一批资料」。

**为什么单独一个模块**

2026-11-08 第二次踩同一个坑：多查询只在 `app.py` 里实现，
评测脚本（`eval_judge.py` / `eval_compound.py`）还在用单查询。
于是「测的」和「跑的」又不是一回事。

第一次是 `eval.py` 写死 `vec_mode`，导致半个月的数字都是错的。

所以检索策略必须只有一份：线上和评测都从这里进。

## 这一层做了什么

1. **意图识别**（`intake`）：看出用户在问哪类事、已经给了哪些信息
2. **多查询构造**：用户带了个人信息时，用「意图 + 槽位」重新构造几个查询——
   因为「我大二，绩点2.8，能转专业吗」召回不到「绩点不低于3.00」那条，
   而「转专业 绩点 要求」能召回
3. **复合问题拆分**（`planner`）：一句话问两件事时拆成两个查询
4. **轮流合并**：保证每个查询都有代表进 top-k
"""

import intake
import planner

# 多查询并发用的线程池。
# 每个查询是一次独立的 embedding 调用（网络 IO），彼此不依赖。
from concurrent.futures import ThreadPoolExecutor


def build_queries(query: str, client=None, model: str = "Qwen/Qwen2.5-7B-Instruct") -> list[str]:
    """决定这次要查几遍、分别查什么。

    优先级：意图 > 复合问题 > 原话。
    """
    intent = intake.detect_intent(query)
    if intent:
        slots = intake.extract_slots(query, intent)
        return intake.search_queries(query, intent, slots)

    if client is not None and planner.is_compound(query):
        return planner.split(query, client, model)

    return [query]


def merge_round_robin(buckets: list[list[dict]], width: int) -> list[dict]:
    """轮流从每个查询的结果里取，保证每个查询都有代表。

    不这么做：第一个查询的 5 条分数普遍更高，直接排序会把后面的查询全挤掉。
    """
    res, seen = [], set()
    buckets = [list(b) for b in buckets]
    while len(res) < width and any(buckets):
        for bucket in buckets:
            if len(res) >= width:
                break
            while bucket:
                rec = bucket.pop(0)
                if rec["id"] in seen:
                    continue
                seen.add(rec["id"])
                res.append(rec)
                break
    return res


def search(retriever, query: str, k: int, *, rewrite: bool = True, client=None,
           model: str = "Qwen/Qwen2.5-7B-Instruct"):
    """返回 (资料列表, 最高相似度)。这是线上和评测共用的唯一入口。

    k 是用户看到的条数。多查询时会自动放大到 min(2k, 10)——
    实测不放大反而更差：2 个查询只能分 5 条，每个问题拿到 2-3 条。

    **多查询并发跑，不串行**

    每个查询是一次独立的网络调用（embedding + 检索），彼此不依赖。
    串行 4 个查询实测 3.7 秒，并发后应该在 1 秒左右——
    总耗时 = 最慢的那个，不是四个相加。

    这也是「API 慢的时候多开几个」的朴素想法能成立的地方：
    它不减少总工作量，但把等待重叠起来了。
    """
    queries = build_queries(query, client=client, model=model)
    width = min(k * 2, 10) if len(queries) > 1 else k

    def one(q: str):
        return retriever.search(q, k=width, rewrite=rewrite, vec_mode="orig")

    if len(queries) == 1:
        return one(queries[0])

    # 并发。上限 5：再多人家的 API 会限流，收益也到顶了。
    with ThreadPoolExecutor(max_workers=min(len(queries), 5)) as pool:
        results = list(pool.map(one, queries))

    buckets = [res for res, _ in results]
    best = max(sim for _, sim in results)
    return merge_round_robin(buckets, width), best
