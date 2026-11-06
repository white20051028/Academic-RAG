"""测量判断层的稳定性：同一个问题，同一份资料，判定会不会变。

为什么要单独一个脚本：

2026-11-06 发现「本硕连读是不是直接读研」两次跑结果不同（一次说能、一次说不能），
temperature=0 也没保证确定性。如果判断层本身会抖，那后面所有评测数字都不可信——
39 条样本里翻 2 条，指标就差 5 个百分点。

测量方法：
  - 检索只跑一次（向量/BM25/融合都是确定的，随机性只在 LLM 调用）
  - 判断层跑 N 次，统计「全票通过 / 全票否决 / 摇摆」
  - 摇摆的题单独列出来，看它们卡在什么位置

用法：
  python tools/stability.py            # 默认 N=3
  python tools/stability.py --n=5      # 跑 5 次
  python tools/stability.py --limit=10 # 先跑 10 条冒烟
"""

import json
import sys
from pathlib import Path

import paths
from paths import ROOT

MODEL = "Qwen/Qwen2.5-7B-Instruct"

sys.path.insert(0, str(ROOT / "tools"))
import judge  # noqa: E402


def read_key() -> str:
    return paths.read_env("SILICONFLOW_API_KEY")


def load_set(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("//")
    ]


def build_context(results) -> str:
    return "\n\n".join(
        f"[{i}] 来源：{r['meta']['source']}\n{r['doc']}" for i, r in enumerate(results, 1)
    )


def main():
    from openai import OpenAI

    import retrieval as R

    n = next((int(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--n=")), 3)
    limit = next((int(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--limit=")), None)

    client = OpenAI(api_key=read_key(), base_url="https://api.siliconflow.cn/v1")
    retriever = R.HybridRetriever(client, "BAAI/bge-m3")

    items = load_set(ROOT / "data" / "eval_set_colloquial.jsonl") + \
            load_set(ROOT / "data" / "eval_set_refusal.jsonl")
    if limit:
        items = items[:limit]

    print(f"判断层稳定性：{len(items)} 条 × {n} 次 = {len(items) * n} 次调用")
    print(f"模型 {MODEL} ｜ temperature=0\n")
    print("=" * 64)

    stable_yes = 0
    stable_no = 0
    flaky = []

    for idx, item in enumerate(items, 1):
        q = item["q"]
        res, sim = retriever.search(q, k=5, rewrite=True, vec_mode="orig")
        ctx = build_context(res)

        votes = [judge.check_answerable(q, ctx, client, MODEL) for _ in range(n)]
        yes = sum(votes)

        if yes == n:
            stable_yes += 1
            mark = "ALL-能"
        elif yes == 0:
            stable_no += 1
            mark = "ALL-不能"
        else:
            flaky.append((q, yes, n, sim))
            mark = f"摇摆 {yes}/{n}"

        print(f"  [{idx:>2}/{len(items)}] {mark:<10} {sim:.3f}  {q}")
        sys.stdout.flush()

    total = len(items)
    # 只统计「不稳定」：同一题既有能又有不能
    print("\n" + "=" * 64)
    print(f"稳定判「能」   {stable_yes}/{total} = {stable_yes/total:.1%}")
    print(f"稳定判「不能」 {stable_no}/{total} = {stable_no/total:.1%}")
    print(f"**摇摆**       {len(flaky)}/{total} = {len(flaky)/total:.1%}")

    if flaky:
        print("\n摇摆的题（同一份资料、同一个问题，判定会变）：")
        for q, yes, n_, sim in flaky:
            print(f"  {yes}/{n_} 次说能  ·  相似度 {sim:.3f}  ·  {q}")

    print("\n" + "=" * 64)
    if flaky:
        print("结论：判断层不稳定。单次评测的数字有噪声，不能当精确值。")
    else:
        print(f"结论：{n} 次全部一致。要么确实稳定，要么这个样本量看不出问题。")
        print("      要排除「样本太小」的可能，可以加大 --n 或换成边界样本再测。")


if __name__ == "__main__":
    main()
