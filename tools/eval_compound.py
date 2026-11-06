"""复合问题评测：拆分检索 vs 单次检索，到底差多少。

判分规则：**每个子答案都要命中**才算答对。

为什么不用「任意一个命中」：
「辅修能抵学分吗，最多抵多少」如果只答出"能抵"，用户还是不知道上限。
只算一半等于答错。

跑法：
  python tools/eval_compound.py            # 拆分后检索（新逻辑）
  python tools/eval_compound.py --single   # 单次检索（模拟改之前）
"""

import json
import sys
from pathlib import Path

import paths
from paths import ROOT

SET_FILE = ROOT / "data" / "eval_set_compound.jsonl"
EMBED_MODEL = "BAAI/bge-m3"
CHAT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
K = 8

sys.path.insert(0, str(ROOT / "tools"))
import judge  # noqa: E402
import planner  # noqa: E402


def read_key() -> str:
    return paths.read_env("SILICONFLOW_API_KEY")


def load_set(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def group_hit(group, results) -> bool:
    """一组期望（来源 + 关键词）有没有被满足。"""
    src_kw, kws = group
    for r in results:
        if src_kw not in r["meta"].get("source", ""):
            continue
        if any(kw in r["doc"] for kw in kws):
            return True
    return False


def merge_round_robin(buckets: list[list[dict]], k: int) -> list[dict]:
    """轮流从每个子查询的结果里取，保证每个子问题都有代表。

    不这么做的话：第一个子查询的 5 条分数普遍更高，直接排序取 top-5
    会把第二个子查询的结果全挤掉——拆了等于没拆。
    """
    res, seen = [], set()
    buckets = [list(b) for b in buckets]
    while len(res) < k and any(buckets):
        for bucket in buckets:
            if len(res) >= k:
                break
            while bucket:
                rec = bucket.pop(0)
                if rec["id"] in seen:
                    continue
                seen.add(rec["id"])
                res.append(rec)
                break
    return res


def main():
    from openai import OpenAI

    import retrieval as R

    single = "--single" in sys.argv

    client = OpenAI(api_key=read_key(), base_url="https://api.siliconflow.cn/v1")
    r = R.HybridRetriever(client, EMBED_MODEL)

    items = load_set(SET_FILE)
    mode = "单次检索（改前）" if single else "拆分检索（改后）"
    print(f"复合问题 {len(items)} 条 ｜ {mode}\n")
    print("=" * 64)

    full_ok = 0
    half_ok = 0
    sub_hits = 0
    sub_total = 0

    for i, item in enumerate(items, 1):
        q = item["q"]
        if single:
            res, _ = r.search(q, k=K, rewrite=True, vec_mode="orig")
            subs = [q]
        else:
            subs = planner.split(q, client, CHAT_MODEL)
            buckets = []
            for s in subs:
                rr, _ = r.search(s, k=K, rewrite=True, vec_mode="orig")
                buckets.append(rr)
            res = merge_round_robin(buckets, K)

        hits = [group_hit(g, res) for g in item["expect_all"]]
        n_hit, n_all = sum(hits), len(hits)
        sub_hits += n_hit
        sub_total += n_all
        full = n_hit == n_all
        full_ok += full
        half_ok += n_hit > 0

        mark = "✅" if full else ("⚠️一半" if n_hit else "❌")
        print(f"  [{i:>2}/{len(items)}] {mark:<6} {n_hit}/{n_all}  {q}")
        sys.stdout.flush()

    print("\n" + "=" * 64)
    print(f"完整答对    : {full_ok}/{len(items)} = {full_ok/len(items):.1%}")
    print(f"至少答出一半: {half_ok}/{len(items)} = {half_ok/len(items):.1%}")
    print(f"子答案命中率: {sub_hits}/{sub_total} = {sub_hits/sub_total:.1%}")


if __name__ == "__main__":
    main()
