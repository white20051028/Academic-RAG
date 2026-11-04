"""评测检索质量：Recall@5 / MRR，支持向量 vs 混合两种模式。

用法：
  python eval.py                              # 默认：混合检索 + 书面评测集
  python eval.py --vector                     # 纯向量（v0 基线）
  python eval.py data/eval_set_colloquial.jsonl
  python eval.py --vector data/eval_set_colloquial.jsonl
  python eval.py data/eval_set_refusal.jsonl  # 拒答专项

评测集会自动识别两类条目：
  - 有 expect_source / expect_keywords  → 测「召回命中」
  - 有 expect_no_answer: true          → 测「是否正确拒答」

三个指标：
  Recall@5     召回的 5 条里有没有正确答案（只统计应该回答的）
  拒答准确率    应该拒答的里面，正确拒了的比例
  误拒率        应该回答的里面，因为相似度低于阈值而被拒的比例
"""

import json
import sys
from pathlib import Path


ROOT = Path(r"D:\桌面\教务RAG")
CHROMA = ROOT / "data" / "chroma"
ENV_FILE = Path(r"D:\桌面\多轮对话机器人项目\.env")
DEFAULT_SET = ROOT / "data" / "eval_set.jsonl"
K = 5
EMBED_MODEL = "BAAI/bge-m3"
REFUSE_THRESHOLD = 0.60      # 用 50 条样本标定，见 docs/v0-基线.md


def read_key() -> str:
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        if line.startswith("SILICONFLOW_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("没找到 SILICONFLOW_API_KEY")


def load_set(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("//")
    ]


def is_hit(item: dict, results: list[dict]) -> tuple[bool, int]:
    """命中 = 来源对 + 含关键词。返回 (是否命中, 名次)。"""
    for rank, r in enumerate(results, 1):
        src_need = item.get("expect_source")
        if src_need and src_need not in r["meta"].get("source", ""):
            continue
        kws = item.get("expect_keywords") or []
        if kws and not any(kw in r["doc"] for kw in kws):
            continue
        return True, rank
    return False, 0


def search_vector(client, col, q: str) -> tuple[list[dict], float]:
    vec = client.embeddings.create(model=EMBED_MODEL, input=[q]).data[0].embedding
    r = col.query(query_embeddings=[vec], n_results=K,
                  include=["documents", "metadatas", "distances"])
    results = [
        {"doc": d, "meta": m, "score": 1 - dist}
        for d, m, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])
    ]
    return results, results[0]["score"]


def main():
    from openai import OpenAI

    import retrieval as R

    use_vector = "--vector" in sys.argv
    use_rewrite = "--rewrite" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    eval_path = Path(args[0]) if args else DEFAULT_SET
    if not eval_path.exists():
        print(f"❌ 评测集不存在: {eval_path}")
        sys.exit(1)

    items = load_set(eval_path)
    for item in items:
        if not item.get("expect_source") and not item.get("expect_keywords") \
                and not item.get("expect_no_answer"):
            print(f"❌ 这条没有任何判定条件，会让结果虚高：{item.get('q')}")
            sys.exit(1)

    client = OpenAI(api_key=read_key(), base_url="https://api.siliconflow.cn/v1")
    retriever = R.HybridRetriever(client, EMBED_MODEL)

    mode = "纯向量" if use_vector else ("混合+改写" if use_rewrite else "混合(BM25+向量+RRF)")
    print(f"索引 {retriever.size} 个 chunk ｜ 模式：{mode}")
    print(f"评测集：{eval_path.name}（{len(items)} 条）\n")

    recall_items = [i for i in items if not i.get("expect_no_answer")]
    refuse_items = [i for i in items if i.get("expect_no_answer")]

    hits = 0
    rr_sum = 0.0
    false_refuse = 0
    failures: list[tuple] = []
    by_cat: dict[str, list[int]] = {}     # cat -> [ok, bad]

    def do_search(q: str):
        if use_vector:
            return search_vector(client, col, q)
        res, sim = retriever.search(q, k=K, rewrite=use_rewrite)
        return [{"doc": r["doc"], "meta": r["meta"]} for r in res], sim

    for idx, item in enumerate(recall_items, 1):
        q = item["q"]
        results, best_sim = do_search(q)
        ok, rank = is_hit(item, results)
        cat = item.get("category", "未分类")
        by_cat.setdefault(cat, [0, 0])

        flags = []
        if ok:
            hits += 1
            rr_sum += 1.0 / rank
            by_cat[cat][0] += 1
            flags.append(f"✅ rank={rank}")
        else:
            by_cat[cat][1] += 1
            flags.append("❌")
            failures.append((q, results[0]["meta"].get("source", "?") if results else "?", best_sim))
        if best_sim < REFUSE_THRESHOLD:
            false_refuse += 1
            flags.append("[会被拒答]")

        print(f"  [{idx:>2}/{len(recall_items)}] {' '.join(flags):<20} {q[:34]}")

    refused_ok = 0
    refuse_fail: list[tuple] = []
    if refuse_items:
        print()
        for idx, item in enumerate(refuse_items, 1):
            q = item["q"]
            results, best_sim = do_search(q)
            got = results[0]["meta"].get("source", "?") if results else "?"
            ok = best_sim < REFUSE_THRESHOLD
            if ok:
                refused_ok += 1
            else:
                refuse_fail.append((q, got, best_sim))
            print(f"  [拒 {idx:>2}/{len(refuse_items)}] "
                  f"{'✅ 拒答' if ok else '❌ 放行':<12} 相似度 {best_sim:.3f}  {q[:32]}")

    n = len(recall_items)
    print()
    print("=" * 66)
    if n:
        print(f"  Recall@{K}  : {hits}/{n} = {hits/n:.2%}")
        print(f"  MRR        : {rr_sum/n:.3f}")
        print(f"  误拒率     : {false_refuse}/{n} = {false_refuse/n:.2%}")
    if refuse_items:
        print(f"  拒答准确率 : {refused_ok}/{len(refuse_items)} = {refused_ok/len(refuse_items):.2%}")

    if by_cat:
        print("\n=== 按类别 ===")
        for cat, (ok_n, bad_n) in sorted(by_cat.items()):
            tot = ok_n + bad_n
            print(f"  {cat:<14} {ok_n:>2}/{tot:<2} = {ok_n/max(tot, 1):.0%}")

    if failures:
        print(f"\n=== 召回失败的 {len(failures)} 条 ===")
        for q, got, sim in failures:
            print(f"  「{q[:32]}」 相似度 {sim:.3f}  实际 {got[:42]}")
    if refuse_fail:
        print(f"\n=== 该拒没拒的 {len(refuse_fail)} 条 ===")
        for q, got, sim in refuse_fail:
            print(f"  「{q[:32]}」 相似度 {sim:.3f}  召回 {got[:42]}")

    tag = ("vector" if use_vector else ("hybrid_rewrite" if use_rewrite else "hybrid")) + "_" + eval_path.stem
    out = ROOT / "data" / f"eval_result_{tag}.json"
    out.write_text(json.dumps({
        "eval_set": eval_path.name, "mode": mode, "k": K,
        "recall@5": hits / n if n else None,
        "mrr": rr_sum / n if n else None,
        "refusal_accuracy": refused_ok / len(refuse_items) if refuse_items else None,
        "false_refusal_rate": false_refuse / n if n else None,
        "n_recall": n, "n_refuse": len(refuse_items),
        "by_category": {c: {"ok": v[0], "bad": v[1]} for c, v in by_cat.items()},
        "failures": [{"q": q, "got": g, "sim": s} for q, g, s in failures],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已存: {out.name}")


if __name__ == "__main__":
    main()
