"""检查评测集标注本身有没有错。

跑评测之前必须先跑这个。为什么：
  如果 expect_keywords 里的词在文档里根本不存在（多了个空格、写错字），
  那条会被判为"检索失败"，但实际是标注错了。
  ——你会以为是检索系统不行，其实是自己写错了。

检查三件事：
  1. expect_source 能不能匹配到某个保留的文件
  2. expect_keywords 每个都能在该文件的 chunk 里找到
  3. 有没有重复问题
"""

import json
from pathlib import Path

import chromadb

ROOT = Path(r"D:\桌面\教务RAG")
CHROMA = ROOT / "data" / "chroma"
EVAL_SET = ROOT / "data" / "eval_set.jsonl"


def main():
    items = [
        json.loads(line)
        for line in EVAL_SET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"评测集 {len(items)} 条\n")

    db = chromadb.PersistentClient(path=str(CHROMA))
    col = db.get_collection("xust")

    # 把所有 chunk 按来源文件分组
    data = col.get(include=["documents", "metadatas"])
    by_source: dict[str, list[str]] = {}
    for doc, meta in zip(data["documents"], data["metadatas"]):
        by_source.setdefault(meta["source"], []).append(doc)

    print(f"索引里 {len(data['documents'])} 个 chunk，来自 {len(by_source)} 个文件\n")

    problems = []
    seen_q = {}

    for i, item in enumerate(items, 1):
        q = item["q"]
        src_need = item.get("expect_source", "")
        kws = item.get("expect_keywords", [])

        if q in seen_q:
            problems.append((i, q, f"问题重复（第 {seen_q[q]} 条已出现）"))
        seen_q[q] = i

        # 1. 找匹配的文件
        matched = [f for f in by_source if src_need in f]
        if not matched:
            problems.append((i, q, f"⚠️ 没有文件包含「{src_need}」"))
            continue

        # 2. 关键词是否真的存在
        texts = [t for f in matched for t in by_source[f]]
        missing = [kw for kw in kws if not any(kw in t for t in texts)]
        if missing:
            problems.append((i, q, f"⚠️ 关键词不存在: {missing}（文件: {matched[0][:40]}）"))

    if problems:
        print("=== 标注问题 ===")
        for i, q, msg in problems:
            print(f"  [{i:>2}] {q[:32]}")
            print(f"       {msg}")
        print(f"\n共 {len(problems)} 处问题，建议先修再评测")
    else:
        print("✅ 所有标注都能在文档里对上，可以跑评测了")

    # 顺带统计每个来源被多少条引用
    print("\n=== 来源分布 ===")
    from collections import Counter
    cnt = Counter()
    for item in items:
        for f in by_source:
            if item["expect_source"] in f:
                cnt[f.replace(".txt", "")[:52]] += 1
                break
    for f, n in cnt.most_common():
        print(f"  {n:>2} 条  {f}")


if __name__ == "__main__":
    main()
