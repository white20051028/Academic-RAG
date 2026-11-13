"""跑 57 条端到端评测集，统计判断层 A/B/C 分布。

目的：决定 Agent 兜底值不值得接。
如果 C 的比例很低（比如 < 8%），Agent 就是白搭；
如果 C 有一两成，说明还有救。

跑法：
  E:\\anaconda3\\envs\\llm-rag\\python.exe tools/count_abc.py
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from openai import OpenAI

import judge
import paths
import retrieval as R
import search as S

BATCH = 4


def load_questions() -> list[tuple[str, str]]:
    """读两份评测集，返回 [(问题, 来源文件)]。"""
    out = []
    for name in ("eval_set.jsonl", "eval_set_colloquial.jsonl"):
        p = paths.DATA / name
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            q = row.get("question") or row.get("q")
            if q:
                out.append((q, name))
    return out


def judge_one(item, sf, ds, retriever):
    q, src = item
    try:
        res, sim = S.search(retriever, q, 5, client=sf)
        ctx = "\n\n".join(
            f"[{i}] {x['meta']['source']}\n{x['doc']}" for i, x in enumerate(res, 1)
        )
        kind, reason = judge.classify(q, ctx, ds, judge.JUDGE_MODEL)
        return (q, src, kind, round(sim, 3), reason[:40])
    except Exception as e:  # noqa: BLE001
        return (q, src, "ERR", 0.0, f"{type(e).__name__}: {str(e)[:40]}")


def main():
    sf = OpenAI(api_key=paths.read_env("SILICONFLOW_API_KEY"),
                base_url="https://api.siliconflow.cn/v1")
    ds = OpenAI(api_key=paths.read_env("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com")
    retriever = R.HybridRetriever(sf)

    items = load_questions()
    print(f"共 {len(items)} 条问题\n")

    rows = []
    with ThreadPoolExecutor(max_workers=BATCH) as pool:
        for r in pool.map(lambda it: judge_one(it, sf, ds, retriever), items):
            rows.append(r)
            print(f"  {r[2]:>3}  {r[0][:34]}")

    from collections import Counter

    dist = Counter(r[2] for r in rows)
    total = len(rows)
    print("\n" + "=" * 50)
    print("分类分布：")
    for k in ("A", "B", "C", "ERR"):
        if dist.get(k):
            print(f"  {k}: {dist[k]:>2} 条  ({dist[k]/total*100:.0f}%)")

    c_like = [r for r in rows if r[2] == "C"]
    if c_like:
        print(f"\nC 类的 {len(c_like)} 条（Agent 会接手的）：")
        for r in c_like:
            print(f"  · {r[0][:36]}")
            print(f"      {r[4]}")

    out = paths.DATA / "abc_count.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n明细已写到 {out}")


if __name__ == "__main__":
    main()
