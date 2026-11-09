"""意图识别 + 槽位抽取：从用户那句话里，看出他问的是什么、已经给了哪些信息。

**为什么需要它**

用户不会一次把信息说全。实测下来大概分三种：

    完全笼统  「转专业有啥要求」                → 什么信息都没有
    部分完整  「我大二，能转专业吗」             → 有年级，没绩点
    完全完整  「我大二，绩点2.8，能转专业吗」    → 都有

三种要给不同的回答（见 docs/产品判断-先给规则再引导.md）：
槽位齐 → 直接判断；槽位缺 → 先列规则，再引导补。

**为什么用正则抽，不用模型**

抽取的是「大二」「2.8」这类格式固定的东西，正则又快又准，还免费。
用模型抽一次要多 2 秒和一次调用，为了一个数字不值得。

抽不到的情况（「我分数不太高」）就当作没提供——让用户补一句比猜错强。
"""

import re

# ---------------- 意图 ----------------

# 一个意图的关键词。命中任意一个就算。
INTENTS = {
    "转专业": ("转专业", "换专业", "转系"),
    "毕业学分": ("毕业", "学分够", "差多少学分", "够不够毕业"),
    "辅修": ("辅修", "第二专业"),
    "重修": ("重修", "挂科", "补考"),
    "学位": ("学位证", "学位", "绩点"),
}

# 每个意图需要哪些信息才能「给结论」而不是「只列规则」
REQUIRED_SLOTS = {
    "转专业": ("年级", "绩点"),
    "毕业学分": ("专业", "已修学分"),
    "辅修": ("绩点",),
    "重修": (),
    "学位": ("绩点",),
}

# 槽位的中文名，用来拼提示语
SLOT_LABELS = {
    "年级": "年级（大几）",
    "绩点": "平均学分绩点",
    "专业": "专业",
    "已修学分": "已修学分",
}


def detect_intent(question: str) -> str | None:
    """这句话在问哪一类事。多个命中时取第一个（INTENTS 是按优先级排的）。"""
    for intent, words in INTENTS.items():
        if any(w in question for w in words):
            return intent
    return None


# ---------------- 槽位抽取 ----------------

# 每个意图的抽取规则。键是槽位名，值是正则列表（任一命中即可）
EXTRACTORS = {
    "年级": [r"大[一二三四五]", r"([一二三四五])年级"],
    "绩点": [
        r"绩点\s*(\d(?:\.\d{1,2})?)",
        r"(\d\.\d{1,2})\s*的?绩点",
        r"平均学分绩点\s*(\d(?:\.\d{1,2})?)",
    ],
    "已修学分": [
        r"(?:修了|已经修|修够|得到)\s*(\d{2,3})\s*学分",
        r"(\d{2,3})\s*学分",
    ],
    "专业": [r"计算机|软件工程|通信|自动化|电子|机械|会计|英语"],
}


def extract_slots(question: str, intent: str) -> dict:
    """从问题里抽出已知的槽位值。抽不到就不放进结果。"""
    need = REQUIRED_SLOTS.get(intent, ())
    found = {}
    for slot in need:
        for pattern in EXTRACTORS.get(slot, ()):
            m = re.search(pattern, question)
            if not m:
                continue
            # 有的正则带捕获组，有的不带
            value = m.group(1) if m.groups() else m.group(0)
            found[slot] = value
            break
    return found


def missing_slots(intent: str, found: dict) -> list[str]:
    """还缺哪些槽位。"""
    return [s for s in REQUIRED_SLOTS.get(intent, ()) if s not in found]


def ask_text(intent: str, missing: list[str]) -> str:
    """缺信息时，结尾该引导用户补什么。

    措辞很重要：要说「我可以帮你判断」，不是「请补充信息」——
    前者是提供价值，后者是提要求。
    """
    if not missing:
        return ""
    names = "、".join(SLOT_LABELS.get(s, s) for s in missing)
    if intent == "转专业":
        return (
            f"\n\n---\n\n**想知道你自己能不能转？** 把 {names} 告诉我，"
            f"我直接帮你对照条件判断，并告诉你差在哪。"
        )
    if intent == "毕业学分":
        return (
            f"\n\n---\n\n**想知道你还差多少学分？** 把 {names} 告诉我，"
            f"我算给你看。"
        )
    if intent == "学位":
        return (
            f"\n\n---\n\n**想知道你能不能拿到学位证？** 把 {names} 告诉我，"
            f"我对照条件帮你判断。"
        )
    return f"\n\n---\n\n**想让我帮你判断？** 告诉我你的 {names} 就行。"


def needs_context(question: str, has_history: bool) -> bool:
    """**规则版**判断：这句话是不是在承接上一轮。

    只处理「明确」的情况，剩下的返回 None 交给模型判断（见 search.py）。
    规则能确定的两头：

        明确承接  「那辅修呢」「这个要多久」     → True
        明确独立  「转专业要什么条件？」          → False
        不确定    「大三，计科，剩下的都是无」    → None

    **为什么不再试图用规则覆盖全部**：
    这个函数已经被真实用户打脸三次了——
      「学分180 绩点2.8」     规则要求带数字 → 漏了汉字数字
      「大三，计科，都是无」   规则要求 \d     → 漏了「三」
    用户怎么说话是不可穷举的，规则只该处理最明确的两头。
    """
    if not has_history:
        return False

    q = question.strip()
    if not q:
        return False

    # 明确承接：指代词开头
    if any(q.startswith(w) for w in _REFER):
        return True

    # 明确独立：有问号 + 有疑问词 + 不是指代
    has_qmark = "？" in q or "?" in q
    has_qword = q.rstrip("。！! ").endswith("吗") or any(w in q for w in QUESTION_WORDS)
    if has_qmark and has_qword:
        return False

    # 其余交给模型
    return None


CONTEXT_PROMPT = """判断这一轮用户的话是不是在接上一轮。

上一轮用户说的是：{prev}
这一轮用户说的是：{cur}

如果这一轮是在回答上一轮的问题、补充信息、或者继续聊同一个话题 → 输出 YES
如果这一轮是一个全新的话题、和上一轮无关 → 输出 NO

只输出 YES 或 NO，不要解释。

输出："""


def llm_is_followup(prev: str, cur: str, client, model: str) -> bool:
    """让模型判断「这句话是不是在接上一轮」。

    真实场景里用户怎么说话是猜不完的：
        「大三，计科，剩下的问题全都是无（否）」
    这句话没有问号、没有疑问词、没有阿拉伯数字——规则怎么写都会漏。
    让模型看一眼上下文，比堆正则可靠。

    失败时返回 False（当成独立问题）——因为判断错了最多是少拼一条，
    而误拼会让无关资料混进检索。
    """
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user",
                       "content": CONTEXT_PROMPT.format(prev=prev, cur=cur)}],
            max_tokens=8,
            temperature=0,
        )
        ans = (r.choices[0].message.content or "").strip().upper()
        return ans.startswith("YES") or ans.startswith("是")
    except Exception:  # noqa: BLE001
        return False


def search_queries(question: str, intent: str | None, slots: dict,
                   max_q: int = 4) -> list[str]:
    """构造多个检索查询，而不是只用一个。

    **为什么不只用原话**

    用户提供的个人信息会污染检索：
        「我大二，绩点2.8，能转专业吗」    → 召回不到「绩点不低于3.00」那条
        「转专业 绩点要求」                → 能召回

    「大二」「2.8」这些数字抢走了「转专业」的语义权重。

    **为什么不能只用一个模板**

    实测同一个意图，措辞差一个字结果就翻：
        「转专业 绩点要求」   ✅
        「转专业 绩点 要求」  ✅
        「转专业 绩点」       ❌
        「转专业申请条件」    ❌

    与其猜哪个措辞对，不如都查一遍——总有一个能召回。
    这是用一点延迟换鲁棒性：多查 2-3 次，每次不到 1 秒。
    """
    qs = [question]          # 原话保底：有时候原话反而准

    if not intent:
        return qs

    # **每个槽位单独配一个查询，不要拼在一起。**
    #
    # 实测：「转专业 绩点 要求」能召回「绩点不低于3.00」那条，
    #       但「转专业 年级 绩点 要求」就不行了——「年级」这个词把「绩点」稀释了。
    #
    # 所以不是「把所有已知信息拼成一个查询」，而是「一个槽位一个查询，都查一遍」。
    for slot in slots:
        qs.append(f"{intent} {slot} 要求")

    qs.append(f"{intent} 要求")
    qs.append(f"{intent} 条件")

    # 去重保序
    return list(dict.fromkeys(qs))[:max_q]


# ---------------- 判断要不要拼接上下文 ----------------

# 指代词开头，说明在接上一轮
_REFER = ("那", "这个", "那个", "它", "上面", "刚才", "还有", "另外")

# 疑问词。分句里出现任意一个，就算这个分句在问东西。
# 复合问题的判断在 planner.is_compound 里，这里用来判断
# 「当前这句是独立问题 / 还是在补充上一轮的答案」。
QUESTION_WORDS = (
    "多少", "几个", "几门", "几学期", "几点", "多久", "多长", "几次",
    "怎么", "如何", "怎样",
    "能不能", "可不可以", "是否", "有没有", "要不要", "会不会", "算不算",
    "哪个", "哪些", "哪种", "什么", "区别", "差别", "不同", "要求", "条件",
)
