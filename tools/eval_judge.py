"""评测判断层：资料够不够、该不该拒答、还是闲聊。

**为什么单独写这个脚本**

`tools/eval.py` 只量检索和阈值，量不到 `app.py` 里那层模型判断
（`judge.check_answerable`）。可那层决定了用户最终有没有拿到答案。

所以之前的「拒答 80%」是**纯阈值**算的，不是线上真实表现。
这个脚本第一次把整条判定链（阈值 → 判断层）一起量出来。

**为什么和 app.py 用同一个 Judge**

2026-11-06 的教训：`eval.py` 写死 `vec_mode`，和 `app.py` 分叉，
「测的」和「跑的」不是一回事，半个月没人发现。
所以这里 import 的是 `judge.py` 本身，阈值也来自 `judge.DEFAULT_MIN_SCORE`。

跑法：
  python tools/eval_judge.py
  python tools/eval_judge.py --limit=5     # 先跑 5 条冒烟
"""

import json
import sys
from pathlib import Path

import paths
from paths import ROOT

ANSWER_SET = ROOT / "data" / "eval_set_colloquial.jsonl"
REFUSE_SET = ROOT / "data" / "eval_set_refusal.jsonl"
EMBED_MODEL = "BAAI/bge-m3"
CHAT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

sys.path.insert(0, str(ROOT / "tools"))
import judge  # noqa: E402
import search as search_mod  # noqa: E402

# 闲聊测试：规则判断，不调 API。期望全部被 is_smalltalk 认出来。
SMALLTALK_CASES = ["你好", "在吗", "谢谢", "你是谁", "你能干什么", "您好呀", "hello", "好的"]
# 期望**不**被当成闲聊（含问候但有实质问题）
NOT_SMALLTALK_CASES = ["你好，转专业要什么条件", "我想问问绩点2.1能不能拿学位", "谢谢，那挂科了要重修吗"]


def read_key() -> str:
    return paths.read_env("SILICONFLOW_API_KEY")


def load_set(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("//")
    ]


def build_context(results) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        src = r["meta"]["source"].replace(".txt", "")
        parts.append(f"[{i}] 来源：{src}\n{r['doc']}")
    return "\n\n".join(parts)


def is_hit(item: dict, results) -> bool:
    """检索结果里有没有「真的能回答问题」的那一条。

    **为什么必须有这个判断，不能只看「答没答」**

    有些题检索本身就错了（问毕业证，召回的全是学生证通知）。
    如果只看「最终答了没有」，这种题会被算成"答对"——
    但它答的是别的内容，是**答非所问**。

    2026-11-06 第一次评测就踩了这个坑：把「答出 26 条」当成「答对 26 条」，
    得出「判断层净亏 7 条」的错误结论。加上这个判定后，
    真相是：判断层答出的 20 条全部命中，一条错都没有。
    """
    for r in results:
        if item.get("expect_source") and item["expect_source"] not in r["meta"]["source"]:
            continue
        kws = item.get("expect_keywords") or []
        if kws and not any(kw in r["doc"] for kw in kws):
            continue
        return True
    return False


def decide(question: str, results, sim: float, client) -> tuple[str, str]:
    """复现 app.py 的判定链。返回 (最终结论, 说明)。

    结论取值：
      answer        最终答了
      refuse_judge  判断层判资料不够
      smalltalk     闲聊

    2026-11-06 拆掉了「低于 0.60 直接拒」那道闸门：
    问「我想服兵役」向量分只有 0.544，但资料里白纸黑字有答案。
    现在相似度只用于「≥ FAST_PATH 直接答」这条快速通道，不再决定拒答。
    """
    if judge.is_smalltalk(question):
        return "smalltalk", "规则判为闲聊"

    if sim >= judge.FAST_PATH:
        return "answer", f"相似度 {sim:.3f} ≥ {judge.FAST_PATH}，直接答"

    ok = judge.check_answerable(question, build_context(results), client, CHAT_MODEL)
    return ("answer", f"判断层判能答（相似度 {sim:.3f}）") if ok else \
           ("refuse_judge", f"判断层判资料不够（相似度 {sim:.3f}）")


def main():
    from openai import OpenAI

    import retrieval as R

    limit = next((int(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--limit=")), None)

    client = OpenAI(api_key=read_key(), base_url="https://api.siliconflow.cn/v1")
    r = R.HybridRetriever(client, EMBED_MODEL)

    answer_items = load_set(ANSWER_SET)
    refuse_items = load_set(REFUSE_SET)
    if limit:
        answer_items = answer_items[:limit]
        refuse_items = refuse_items[:limit]

    print(f"索引 {r.size} 个 chunk ｜ 判断层：{CHAT_MODEL}")
    print(f"快速通道 ≥{judge.FAST_PATH} 直接答 ｜ 其余全走判断层\n")

    # ---------- 一、闲聊规则（不调 API） ----------
    print("=" * 60)
    print("① 闲聊识别（规则，期望 8 条全部认出 / 3 条全部不误伤）")
    print("=" * 60)
    smalltalk_ok = sum(judge.is_smalltalk(q) for q in SMALLTALK_CASES)
    not_ok = sum(not judge.is_smalltalk(q) for q in NOT_SMALLTALK_CASES)
    for q in SMALLTALK_CASES:
        print(f"  {'✅' if judge.is_smalltalk(q) else '❌'} 认出  {q}")
    for q in NOT_SMALLTALK_CASES:
        print(f"  {'✅' if not judge.is_smalltalk(q) else '❌'} 不误伤 {q}")
    print(f"\n  认出 {smalltalk_ok}/{len(SMALLTALK_CASES)}   不误伤 {not_ok}/{len(NOT_SMALLTALK_CASES)}\n")

    # ---------- 二、该答的（期望最终 answer） ----------
    print("=" * 60)
    print(f"② 该答的 {len(answer_items)} 条（期望最终 answer）")
    print("=" * 60)
    answered = 0
    gray_used = 0
    hit_answered = 0     # 答了 + 检索命中 = 真答对
    miss_answered = 0    # 答了 + 检索没命中 = 答非所问
    for idx, item in enumerate(answer_items, 1):
        q = item["q"]
        res, sim = search_mod.search(r, q, 5, rewrite=True, client=client, model=CHAT_MODEL)
        verdict, why = decide(q, res, sim, client)
        if sim < judge.FAST_PATH:
            gray_used += 1
        ok = verdict == "answer"
        answered += ok
        hit = is_hit(item, res)
        if ok and hit:
            hit_answered += 1
        elif ok and not hit:
            miss_answered += 1
        # 对错按「最终答对」算：答了但不命中，等于答错
        mark = "✅" if (ok and hit) else ("⚠️答非所问" if ok else "❌")
        print(f"  [{idx:>2}/{len(answer_items)}] {mark:<8} {verdict:<12} {q}")
        if not ok:
            print(f"         └─ {why}")
        elif not hit:
            print(f"         └─ 检索没命中正确答案，答了也是错的")
        sys.stdout.flush()

    print(f"\n  最终答出 {answered}/{len(answer_items)} = {answered/len(answer_items):.1%}")
    print(f"  ├─ 其中真答对（检索命中）  {hit_answered}")
    print(f"  └─ 其中答非所问（没命中）  {miss_answered}   ← 这个数越小越好")
    print(f"  进入判断层的 {gray_used} 条（{gray_used/len(answer_items):.0%}）\n")

    # ---------- 三、该拒的（期望最终 refuse_*） ----------
    print("=" * 60)
    print(f"③ 该拒的 {len(refuse_items)} 条（期望 refuse_judge）")
    print("=" * 60)
    refused = 0
    saved_by_judge = 0
    for idx, item in enumerate(refuse_items, 1):
        q = item["q"]
        res, sim = search_mod.search(r, q, 5, rewrite=True, client=client, model=CHAT_MODEL)
        verdict, why = decide(q, res, sim, client)
        ok = verdict == "refuse_judge"
        refused += ok
        if verdict == "refuse_judge":
            saved_by_judge += 1
        print(f"  [{idx:>2}/{len(refuse_items)}] {'✅' if ok else '❌'} {verdict:<12} {q}")
        if not ok:
            print(f"         └─ {why}")
        sys.stdout.flush()

    print(f"\n  正确拒答 {refused}/{len(refuse_items)} = {refused/len(refuse_items):.1%}")
    print(f"  其中靠判断层拦下 {saved_by_judge} 条（纯阈值拦不住的）\n")

    # ---------- 四、总账（按「答对」算，不按「答出」算） ----------
    print("=" * 60)
    total = len(answer_items) + len(refuse_items)
    correct = hit_answered + refused
    print(f"总账（按「答对」算）：{correct}/{total} = {correct/total:.1%}")
    print(f"  真答对 {hit_answered} ｜ 答非所问 {miss_answered} ｜ 正确拒答 {refused}")
    print(f"  答出的 {answered} 条里，答非所问 {miss_answered} 条（{miss_answered/max(answered,1):.0%}）")
    print("=" * 60)


if __name__ == "__main__":
    main()
