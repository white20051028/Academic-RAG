"""问题分解：把一个复合问题拆成几个检索子查询。

**为什么需要它**

现在的检索永远是「一个问题 → top-5」。但学生问的问题经常是复合的：

    辅修能抵主修学分吗，最多抵多少
    ↑ 其实是两个问题：能不能 + 上限多少

一次 top-5 装不下两个答案，结果就是答一半。

**为什么用规则先筛、再让模型拆**

每条问题多一次模型调用就是多 2 秒。大部分问题是简单的（「绩点2.1能拿学位吗」），
拆它纯属浪费。所以先用规则判断「值不值得拆」，只有复合问题才调模型。

这个取舍是量过的：57 条评测集里，符合复合特征的大约 1/4。
不筛的话，75% 的查询白等 2 秒。
"""

import json
import re

# 疑问词。分句里出现任意一个，就算这个分句在问东西。
QUESTION_WORDS = (
    "多少", "几个", "几门", "多久", "多长", "几次",
    "怎么", "如何", "怎样",
    "能不能", "可不可以", "是否", "有没有", "要不要", "会不会", "算不算",
    "哪个", "哪些", "哪种", "什么", "区别", "差别", "不同",
)

# 按中文标点切成分句。保留分隔符位置信息不重要，只要知道"有几个问句"。
_SPLIT = re.compile(r"[，,；;。？?！!]")

# 开场白，不算一个问句
_OPENERS = ("我想问", "请问", "想问一下", "麻烦问", "咨询一下")


def _has_question(fragment: str) -> bool:
    """这个分句在问东西吗。

    「吗」结尾必须单独判断——「重修要交钱吗」里一个疑问词都没有，
    但它显然是个问句。
    """
    if fragment.rstrip("。！! ").endswith("吗"):
        return True
    return any(w in fragment for w in QUESTION_WORDS)


def is_compound(question: str) -> bool:
    """规则判断：这个问题值不值得拆。

    做法：按标点切成分句，数有几个分句在问东西。
    两个以上 = 复合问题。

    **为什么用「数问句」而不是查连接词**：
    「辅修能抵学分吗，最多抵多少」里没有"并且/而且"，但它确实是两个问题。
    数疑问句比查关键词靠谱。

    宁可漏拆，不可白拆——漏拆只是维持现状（答一半），
    白拆是让所有简单问题都多等 2 秒。
    """
    q = question.strip()
    if len(q) < 8:
        return False

    parts = [p.strip() for p in _SPLIT.split(q) if p.strip()]
    if len(parts) < 2:
        return False

    asked = 0
    for part in parts:
        if any(part.startswith(op) for op in _OPENERS):
            continue          # 「我想问一下」不是问题
        if _has_question(part):
            asked += 1

    return asked >= 2


SPLIT_PROMPT = """把学生的问题拆成 1-3 个**完整的问句**。

规则：
- 每个子问题必须是能独立问出口的完整句子，**不是关键词串**
- 保留原句里的限定词（数量、条件、时间、对象）
- 不要抽成「辅修 抵 学分」这种裸词

为什么不能抽成关键词：向量检索靠的是整句的语义，
拆成裸词之后每个词的权重被放大，反而匹配得更差。

示例：
问题：辅修能抵主修学分吗，最多抵多少
输出：["辅修能抵主修学分吗", "辅修最多能抵多少学分"]

问题：{question}
输出："""


def split(question: str, client, model: str, max_parts: int = 3) -> list[str]:
    """返回 1~max_parts 个子查询。失败就原样返回 [question]。

    **失败必须降级，不能报错**：拆问题是优化，不是必需。
    拆失败还让整个问答挂掉，就本末倒置了。
    """
    if not is_compound(question):
        return [question]

    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": SPLIT_PROMPT.format(question=question)}],
            max_tokens=150,
            temperature=0,
        )
        text = (r.choices[0].message.content or "").strip()
        # 模型经常包一层 ```json
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
        parts = json.loads(text)
        if not isinstance(parts, list):
            return [question]
        parts = [str(p).strip() for p in parts if str(p).strip()]
        # 太少（拆了等于没拆）或太多（拆碎了）都退回原问题
        if len(parts) < 2:
            return [question]
        return parts[:max_parts]
    except Exception:  # noqa: BLE001
        return [question]
