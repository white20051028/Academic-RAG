"""建立 v0 检索索引。

流程：加载 txt → 剔除噪音 → 切分 → embedding → 存 Chroma

**v0 的切分故意用最朴素的固定窗口**（500 字 + 50 字重叠）。
后面 v1 做标题感知切分时，要跟这个基线对比——所以它必须朴实。

剔除的噪音：
  - 名单 / 考试安排 / 申请表（无问答价值，且名单含隐私）
  - 文件名乱码的（rar 解压时 GBK 被误解码，无法做引用来源）

输出：
  data/chroma/          向量库
  data/index_stats.json 统计（有效文件/chunk 数/剔除清单）
"""

import json
import hashlib
import os
import re
from pathlib import Path

import chunking

from paths import DATA, PROC, ROOT

STATS = ROOT / "data" / "index_stats.json"
CHROMA = ROOT / "data" / "chroma"          # 已弃用，保留常量防止旧代码引用报错

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MIN_CHUNK = 100
EMBED_MODEL = "BAAI/bge-m3"
BATCH = 16

# 制度类文件名永不剔除 —— 否则会误杀。
# 真实事故：噪音词里有「重修」（本意过滤"重修名单"），
# 结果把《本科生课程重修管理办法》也剔掉了，
# 导致「挂科了要重修吗」永远答不出来，而且从外面看不出问题在哪。
POLICY_WORDS = ("管理办法", "实施细则", "管理规定", "实施办法",
                "暂行办法", "暂行规定", "工作细则", "查处细则")

NOISE_WORDS = (
    "重修", "考试安排", "补考", "期中", "期末", "名单", "申请表",
    "申报书", "汇总表", "汇总", "考场", "成绩", "课表", "安排表",
    # 实践教学栏目里大量「毕设模板」，是纯表格，没有问答价值
    "模板", "简报",
)


def read_key() -> str:
    import paths

    return paths.read_env("SILICONFLOW_API_KEY")


def noise_reason(name: str, text: str) -> str | None:
    """判断一个文件是不是噪音；是的话返回原因。"""
    # 制度类永不剔除（否则「重修管理办法」会被「重修」误杀）
    if any(w in name for w in POLICY_WORDS):
        return None
    if any(w in name for w in NOISE_WORDS):
        return "文件名含噪音词"
    # 乱码文件名：非中文字节被当成别的编码读出来的典型特征
    weird = sum(
        1 for c in name
        if ord(c) > 127 and not ("\u4e00" <= c <= "\u9fff")
    )
    if len(name) > 6 and weird / len(name) > 0.2:
        return "文件名乱码"
    # 内容像名单：学号/姓名高频出现
    if text.count("学号") > 20 or text.count("姓名") > 20:
        return "内容像名单"
    return None


def content_fingerprint(text: str) -> str:
    """内容指纹：取前 1000 字的哈希。

    为什么不用全文哈希：同一份《专业目录》被三个栏目各自引用，
    提取出来的字数有细微差异（21,341 / 21,363 / 23,624），
    全文哈希命中不了，但它们其实是同一份文件。
    用前 1000 字做指纹就能抓住。
    """
    head = re.sub(r"\s+", "", text[:1000])
    return hashlib.sha256(head.encode("utf-8")).hexdigest()[:16]


def split_text(text: str) -> list[str]:
    """固定窗口切分。v0 基线，故意不智能。"""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for i in range(0, len(text), step):
        piece = text[i:i + CHUNK_SIZE].strip()
        if len(piece) >= MIN_CHUNK:
            chunks.append(piece)
    return chunks


def embed_all(texts: list[str], client) -> list[list[float]]:
    out = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        out.extend([d.embedding for d in resp.data])
        print(f"    已向量化 {min(i+BATCH, len(texts))}/{len(texts)}", end="\r")
    print()
    return out


def main():
    from openai import OpenAI

    key = read_key()
    client = OpenAI(api_key=key, base_url="https://api.siliconflow.cn/v1")

    files = sorted(PROC.glob("*.txt"))
    print(f"扫到 {len(files)} 个文本文件\n")

    kept, dropped = [], []
    seen_fp: dict[str, str] = {}
    for p in files:
        text = p.read_text(encoding="utf-8")
        reason = noise_reason(p.name, text)
        if reason:
            dropped.append({"file": p.name, "reason": reason, "chars": len(text)})
            continue
        fp = content_fingerprint(text)
        if fp in seen_fp:
            dropped.append({
                "file": p.name,
                "reason": f"内容重复-同 {seen_fp[fp]}",
                "chars": len(text),
            })
            continue
        seen_fp[fp] = p.name
        kept.append((p, text))

    print(f"保留 {len(kept)} 个，剔除 {len(dropped)} 个\n")
    print("=== 剔除的（按原因统计）===")
    from collections import Counter
    for r, n in Counter(d["reason"] for d in dropped).most_common():
        print(f"  {r}: {n} 个")

    print("\n=== 保留的 ===")
    for p, text in sorted(kept, key=lambda x: -len(x[1])):
        print(f"  {len(text):>7,} 字  {p.name[:60]}")

    # 切分（v3：培养方案那种"每行一门课"的按行切，其余固定窗口）
    # 为什么单独处理：培养方案是 83 行课程，每行一门课。
    # 用 500 字固定窗口会把 5-6 门课挤进一个 chunk，
    # 问「高等数学多少学分」会召回到一个包含别人的块，答不准。
    print("\n=== 切分 ===")
    docs, metas, ids = [], [], []
    by_kind = {"article": 0, "window": 0, "line": 0}
    for p, text in kept:
        if "培养方案" in p.name and "课程列表" in text:
            # 每行一门课，行长的才要（短行是标题/空行）
            pieces = [{"text": ln.strip(), "kind": "line", "chapter": "", "article": ""}
                      for ln in text.split("\n") if len(ln.strip()) >= 15]
            how = "line"
        else:
            pieces, how = chunking.split(text, CHUNK_SIZE, CHUNK_OVERLAP)
        for j, piece in enumerate(pieces):
            docs.append(piece["text"])
            metas.append({
                "source": p.name,
                "chunk": j,
                "chars": len(piece["text"]),
                "kind": piece["kind"],
                "chapter": piece["chapter"],
                "article": piece["article"],
            })
            ids.append(f"{p.stem}__{j}")
            by_kind[piece["kind"]] += 1
    print(f"  总 chunk: {len(docs)}")
    print(f"    按条切: {by_kind['article']}  固定窗口: {by_kind['window']}")
    print(f"  平均长度: {sum(len(d) for d in docs) // max(len(docs), 1)} 字")
    print(f"  有效字数: {sum(len(t) for _, t in kept):,}")

    if not docs:
        print("⚠️ 没有 chunk，退出")
        return

    # 向量化
    print(f"\n=== 向量化（{EMBED_MODEL}）===")
    vectors = embed_all(docs, client)

    # 导出索引：文本 + metadata（JSON）+ 向量矩阵（npy）。
    #
    # 为什么不用向量数据库：
    #   1070 条 × 1024 维，numpy 点积一次约 1ms，完全够用。
    #   而 Chroma 1.5.9 在这个量级会把 HNSW 索引写坏
    #   （InternalError: Error loading hnsw index）——
    #   分批写没用、删库重建也没用。535 条时是好的，1070 条就崩。
    #   **工具要匹配规模**，不是越复杂越好。
    print("\n=== 导出索引 ===")
    import numpy as np

    DATA.mkdir(parents=True, exist_ok=True)

    (DATA / "chunks.json").write_text(
        json.dumps(
            [{"id": i, "doc": d, "meta": m} for i, d, m in zip(ids, docs, metas)],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # 预先归一化：之后点积就等于余弦相似度，检索时不用再算范数
    vec = np.asarray(vectors, dtype="float32")
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    np.save(DATA / "embeddings.npy", vec)

    print(f"  chunks.json     {(DATA / 'chunks.json').stat().st_size // 1024} KB")
    print(f"  embeddings.npy  {vec.shape}  {(DATA / 'embeddings.npy').stat().st_size // 1024} KB")

    stats = {
        "files_total": len(files),
        "files_kept": len(kept),
        "files_dropped": len(dropped),
        "chunks": len(docs),
        "valid_chars": sum(len(t) for _, t in kept),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embed_model": EMBED_MODEL,
        "dropped": dropped,
        "kept": [p.name for p, _ in kept],
    }
    STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n统计已写入 {STATS}")


if __name__ == "__main__":
    main()
