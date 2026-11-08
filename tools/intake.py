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
