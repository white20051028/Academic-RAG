"""快速验证：只跑评测集的前 N 条，看 Recall 是否恢复。

为什么单独写：完整 55 条在 API 慢的时候要跑十几分钟。
凌晨调试只需要看"修复有没有效"，10 条足够。

用法：python quick_eval.py [N]
"""

import json
import sys
from pathlib import Path

ROOT = Path(r"D:\桌面\教务RAG")
ENV_FILE = Path(r"D:\桌面\多轮对话机器人项目\.env")
EVAL = ROOT / "data" / "eval_set.jsonl"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10


def read_key():
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        if line.startswith("SILICONFLOW_API_KEY="):
            return line.split("=", 1)[1].strip()


def main():
    from openai import OpenAI

    import retrieval as R

    client = OpenAI(api_key=read_key(), base_url="https://api.siliconflow.cn/v1")
    r = R.HybridRetriever(client)

    items = [
        json.loads(line)
        for line in EVAL.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][:N]

    hits = 0
    for item in items:
        q = item["q"]
        res, sim = r.search(q, k=5, rewrite=True)
        ok = False
        for rank, x in enumerate(res, 1):
            if item["expect_source"] not in x["meta"]["source"]:
                continue
            if not any(kw in x["doc"] for kw in item["expect_keywords"]):
                continue
            ok = True
            break
        hits += ok
        print(f"  {'OK' if ok else 'X '} {sim:.3f}  {q[:34]}")

    print(f"\n  Recall@5: {hits}/{len(items)} = {hits/len(items):.0%}")


if __name__ == "__main__":
    main()
