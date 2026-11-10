"""教务问答 Web 界面（Streamlit）。

设计要点：
  1. **检索和生成分开显示** —— 答案不准时，一眼看出是"没检索到"还是"模型没用好"
  2. **相似度可见** —— 分数低就说明这条不该信，这是防幻觉的第一层
  3. **两个生成后端** —— 优先走 P1 网关（复用降级/限流/计量），没启动就直接调模型

启动：
  E:\\anaconda3\\envs\\llm-rag\\python.exe -m streamlit run app.py
"""

import json
import re
import sys
from pathlib import Path

import httpx
import streamlit as st

ROOT = Path(__file__).resolve().parent
CHROMA = ROOT / "data" / "chroma"
GATEWAY = "http://127.0.0.1:6066"
EMBED_MODEL = "BAAI/bge-m3"
CHAT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# 判断层单独一个模块：线上和评测共用同一份逻辑，避免"测的"和"跑的"分叉
sys.path.insert(0, str(ROOT / "tools"))
import intake  # noqa: E402
import judge  # noqa: E402
import paths  # noqa: E402
import planner  # noqa: E402
import search as search_mod  # noqa: E402

PROMPT = """你是西安科技大学教务处的问答助手。只根据下面的资料回答问题。

规则：
1. 只用资料里的内容，不要编造。资料里没有的，直接说"文件里没有相关规定"。
2. 回答要具体，把条件、数字、期限说清楚。
3. 在句末用 [1] [2] 标出你是根据第几段资料答的。

资料：
{context}

问题：{question}

回答："""


def read_env(name: str) -> str:
    """读一个配置项。优先项目自己的 .env，没有再用环境变量/旧位置。

    注意这里有两类 key，不能混：
      SILICONFLOW_API_KEY —— 上游模型服务的凭据（调 Qwen 用）
      GATEWAY_API_KEY     —— P1 网关自己的鉴权凭据（走 P1 时用这个）
    用错会直接 401。
    """
    return paths.read_env(name)


def read_key() -> str:
    """当前该用哪个 key。

    优先用**用户在界面上填的**，没有才用 .env 里的。
    这样同学老师用的时候，花的是他们自己的额度，不需要改任何文件。
    """
    ui_key = st.session_state.get("user_key")
    if ui_key:
        return ui_key
    return read_env("SILICONFLOW_API_KEY")


def read_gateway_key() -> str:
    return read_env("GATEWAY_API_KEY")


# 拒答时给"去哪里找"的建议。
# 只写真实存在的信息（从教务处官网抓的），不编造机构名——
# 编一个不存在的"实践教学科"比拒答更糟。
# 拒答时的指路。
# 设计原则：**给"点哪里"，不给"打电话"**。
# 电话只是最后兜底 —— 用户如果愿意打电话，他根本不需要这个工具。
# 每一条都要能让人立刻动手，而不是"去咨询一下"。
#
# 格式：(触发关键词, 标题, [具体操作步骤])
WHERE_TO_LOOK = [
    (("保研", "推免", "免试", "研究生", "读研"),
     "保研 / 推免",
     [
         "查自己够不够格：**教务系统 → 信息查询 → 学生学业情况查询**（看绩点和专业排名）",
         "看学校政策：搜「西安科技大学 推免工作实施办法」（每年 6–9 月教务处发布）",
         "看学院细则：**学院官网 → 研究生教育 / 通知公告**",
         "问名额和流程：**学院教务办**（比教务处更直接，他们管名额分配）",
     ]),
    (("老师", "教师", "师资", "教授", "任课", "谁教"),
     "任课教师",
     [
         "**教务系统 → 信息查询 → 个人课表查询** → 课表里每门课都带任课教师名",
         "**学院官网 → 师资队伍**（看全院的老师名单和研究方向）",
         "想查某位老师的联系方式：学院官网的教师主页",
     ]),
    (("课表", "上课时间", "哪个教室", "几号楼", "在哪上课"),
     "课表 / 教室",
     [
         "**教务系统 → 信息查询 → 个人课表查询**",
         "**教务系统 → 信息查询 → 班级课表查询**（看整个班的）",
     ]),
    (("成绩", "绩点", "考了多少分", "分数"),
     "成绩 / 绩点",
     [
         "**教务系统 → 信息查询 → 学生成绩查询**",
         "**教务系统 → 信息查询 → 学生成绩总表打印**（可以导出成 PDF）",
         "想算 GPA：教务系统 → 信息查询 → 学生学业情况查询",
     ]),
    (("实习", "实践", "毕设", "毕业设计", "实验", "实训"),
     "实习 / 实践环节",
     [
         "**教务系统 → 信息查询 → 教学执行计划查看**（看本学期的实践环节安排）",
         "**学院教务办**问实习的时间安排和学分认定（他们管这块）",
         "学校层面：教务处「实践教学」栏目",
     ]),
    (("宿舍", "食堂", "奖学金", "助学金", "贷款", "勤工", "评优", "处分", "违纪"),
     "学生事务（不归教务处）",
     [
         "**学工部（学生处）官网** → 通知公告 / 规章制度",
         "奖助学金：**学工部 → 学生资助** 栏目",
         "宿舍问题：找**楼管 / 学院辅导员**",
     ]),
    (("补考", "重修", "挂科", "不及格"),
     "补考 / 重修",
     [
         "**教务系统 → 信息查询 → 考试信息查询**（看补考安排）",
         "**教务系统 → 报名申请 → 重修报名 / 补考确认**",
         "规则看《本科生课程重修管理办法》（本系统已收录，可以直接问「重修怎么算」）",
     ]),
    (("转专业", "转学", "换专业"),
     "转专业",
     [
         "**教务系统 → 报名申请 → 学生转专业申请**",
         "政策看本系统里的《转专业工作通知》（可以直接问「转专业要什么条件」）",
         "各学院接收细则：**学院官网 → 通知公告**",
     ]),
    (("辅修", "双学位", "第二专业"),
     "辅修 / 双学位",
     [
         "**教务系统 → 报名申请 → 辅修报名**",
         "政策看本系统里的《辅修专业管理办法》和《辅修专业培养方案》",
     ]),
    (("学位证", "毕业证", "证书遗失", "补办"),
     "证书补办",
     [
         "**教务系统 → 报名申请 → 学生证补办申请**（学生证）",
         "毕业证 / 学位证遗失：**学信网**申请学历学位认证报告，学校不补发原件",
     ]),
]

# 兜底：什么关键词都没匹配上时才用它。**放最后**，因为打电话成本最高。
#
# **每行不要以空格开头。**
# Markdown 里行首 4 个空格 = 代码块，会把 **加粗** 原样显示成星号，
# 还套一个灰底方块。2026-11-09 用户截图里那块看不懂的灰底就是这么来的。
DEFAULT_WHERE = (
    "**教务系统**（59.74.174.150/jwglxt）→ 信息查询，看有没有对应入口\n"
    "**学校官网 / 学院官网** → 搜关键词 + 「西安科技大学」\n"
    "最后才考虑：教务处 029-83858041 · jwc@xust.edu.cn"
)


def suggest_where(question: str) -> list[tuple[str, list[str]]]:
    """按问题关键词给出「去哪查、点哪里」。返回 [(标题, [步骤])]。"""
    hits = [(title, steps) for words, title, steps in WHERE_TO_LOOK
            if any(w in question for w in words)]
    return hits if hits else [("下一步", DEFAULT_WHERE.split("\n"))]


# ---------------- 拒答分三档 ----------------
#
# 用户实测：同学问「吃什么」「赤犬和白胡子谁更强」，系统一本正经地说
# 「答案不在我收录的教务文件里，你可以去教务处查」——很扫兴。
#
# 不是所有拒答都该用同一套话术：
#   教务问题查不到  → 详细指路（这才是真的「没收录」）
#   校园事务不归教务处 → 指路到学工部/后勤
#   完全无关（动漫/闲聊）→ 轻松一句带过，别摆出一副公事公办的脸

# 教务相关：出现任意一个，都按「教务问题」处理
ACADEMIC_WORDS = (
    "学籍", "学分", "绩点", "转专业", "辅修", "微专业", "重修", "补考", "挂科",
    "考试", "成绩", "毕业", "学位", "论文", "毕设", "保研", "推免", "研究生",
    "培养方案", "选课", "课程", "教材", "退学", "休学", "复学", "学制", "延期",
    "本硕连读", "卓越班", "菁英班", "转学", "双学位", "学位证", "毕业证",
    "四六级", "体测", "军训", "实习", "创新创业", "学信网", "学号", "班级",
    # 「校规」类：真实反馈里「西安科技大学校规」被误判成无关问题秒拒了。
    # 《学生管理规定》里明明有，只是这个词不在词表里。
    "校规", "校纪", "规章制度", "学生管理", "处分规定", "考勤", "请假",
    "缺课", "旷课", "作弊", "违纪", "申诉",
)

# 校园相关但不归教务处
CAMPUS_WORDS = (
    "食堂", "宿舍", "寝室", "奖学金", "助学金", "贷款", "勤工", "评优", "处分",
    "违纪", "学生证", "校园卡", "图书馆", "操场", "体育馆", "后勤", "辅导员",
    "社团", "学生会", "校车", "快递", "洗澡", "热水", "网费",
)


def refusal_kind(question: str) -> str:
    """这条拒答该用哪种话术。"""
    if any(w in question for w in ACADEMIC_WORDS):
        return "academic"
    if any(w in question for w in CAMPUS_WORDS):
        return "campus"
    return "off_topic"


# ---------------- 闲聊彩蛋 ----------------
#
# 用户说：问「赤犬和白胡子谁更强」回一句「白胡子」、问「吃什么」回「大饼卷肉」，
# 会比一句正儿八经的「我不懂」有意思。
#
# **为什么是固定表，不是让模型随便聊**：
#   1. 调模型每次多 2-5 秒 + 花钱，为了逗一句不值得
#   2. 让它自由发挥，它可能开始编「食堂二楼有大饼卷肉」这种它并不知道的事
#   3. 这个系统的定位是教务助手，能聊两句是加分，聊得好是越界
#
# 所以每一条都是人写的、**不含任何事实断言**（不说哪里有卖、不说具体价格）。
FUN_REPLIES = [
    # 真实反馈来的：有同学问「校训是什么」被拒答了，但这是常识，答一句更好
    (("校训",),
     "「祖国利益高于一切」。这个我记得住。"),
    (("校史", "建校", "哪年成立"),
     "西科大 1958 年建校，前身是西安矿业学院。这些细节去学校官网更准。"),
    (("短发假小子", "假小子", "黑长直"),
     "短发假小子。这个也没有争议。"),
    # 这两条来自真实同学反馈的梗。做成彩蛋是因为：
    # 1. 是他们自己提的，不是我们编的
    # 2. 产品里留下用户的梗，比任何"欢迎反馈"的标语都有说服力
    # 3. 面试时可以讲「我的用户在我产品里留了个梗」
    (("西科cs", "cs的top1", "cs 的 top1", "计科第一", "cs第一"),
     "张圣泽。这个没有争议。"),
    (("赤犬", "白胡子"),
     "白胡子。顶上战争他输了命，赤犬输了格局。"),
    (("赤犬",),
     "赤犬这人……能力满分，做人零分。"),
    (("白胡子",),
     "白胡子。这个没啥好争的。"),
    (("吃什么", "吃啥", "吃点什么"),
     "大饼卷肉。不接受反驳。"),
    (("海贼", "路飞", "onepiece", "ONE PIECE"),
     "我只知道「我是要成为教务王的男人」这种程度的梗。"),
    (("天气",),
     "这个我真不知道——我在你电脑里，看不见窗外。"),
    (("你是谁", "你是什么"),
     "我是教务政策问答。一个只会背教务处文件、不会聊天的工具。"),
]


def fun_reply(question: str) -> str | None:
    """在固定表里找一句俏皮话。找不到返回 None。"""
    q = question.lower()
    for words, reply in FUN_REPLIES:
        if any(w.lower() in q for w in words):
            return reply
    return None


# ---------------- 二选一：随机站一边 ----------------
#
# 用户的想法：打趣的问题经常是「A 还是 B」（梅西还是 C罗、张元英还是柳志敏），
# 检测到明显是二选一、又跟教务无关，就随机挑一个回答。
#
# **为什么必须明说「瞎选的」**：
# 不说的话，用户会以为系统真有观点，甚至拿去当依据。
# 加一句「别当真」，既保留趣味，又不会误导——这是这个功能能存在的前提。

import random

# 「A 还是 B」：A/B 各允许 2-20 个字符，允许中间带「更」「比较」「好」这类尾巴
_CHOICE_RE = re.compile(
    r"^\s*(.{2,20}?)\s*(?:还是|or|和|vs\.?|VS)\s*(.{2,20}?)\s*"
    r"(?:更|比较|哪个|谁)?[^\w\s]{0,3}\s*[呢吗啊]?[？?。]?\s*$",
    re.IGNORECASE,
)

# 这些尾巴要剪掉，避免把「哪个好」当成选项的一部分
_TAIL_WORDS = ("更好", "更强", "更厉害", "更值得", "哪个好", "哪个强", "谁更强",
               "谁更厉害", "好", "强", "厉害", "值得")


def pick_choice(question: str) -> str | None:
    """如果是「A 还是 B」的二选一，随机选一个，并说明是瞎选的。"""
    m = _CHOICE_RE.match(question.strip())
    if not m:
        return None

    a = m.group(1).strip(" ，,。")
    b = m.group(2).strip(" ，,。")
    for w in _TAIL_WORDS:
        if b.endswith(w):
            b = b[: -len(w)].strip(" ，,") or b
    if not a or not b or a == b:
        return None
    # 太短的当语气词，不算选项
    if len(a) < 2 or len(b) < 2:
        return None

    pick = random.choice([a, b])
    return f"{pick}。\n\n（我瞎选的，别当真 😅）"


@st.cache_resource
def get_embed_client():
    from openai import OpenAI

    return OpenAI(api_key=read_key(), base_url="https://api.siliconflow.cn/v1")


@st.cache_resource
def get_judge_client():
    """判断层专用的 client（DeepSeek 官方）。

    为什么要单独一个：判断层用 deepseek-chat（判得准），生成端用免费的
    Qwen2.5-7B。两个服务商的 base_url 和 key 都不同，不能共用 client。

    成本：每次判断约 2400 tokens，折合 0.003 元左右。生成端不花钱。
    """
    from openai import OpenAI

    return OpenAI(api_key=read_env(judge.JUDGE_KEY_ENV), base_url=judge.JUDGE_BASE_URL)


def gateway_available() -> bool:
    try:
        return httpx.get(f"{GATEWAY}/health", timeout=2).status_code == 200
    except Exception:  # noqa: BLE001
        return False


@st.cache_resource
def get_retriever():
    """混合检索器（向量 + BM25 + RRF）。缓存的，只建一次。

    不再用 Chroma —— 1070 条数据用 numpy 点积就够（一次约 1ms），
    而 Chroma 1.5.9 在这个量级会把 HNSW 索引写坏。
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))
    import retrieval as R

    return R.HybridRetriever(get_embed_client(), EMBED_MODEL)


def retrieve(query: str, k: int, rewrite: bool = True, history: list | None = None):
    """检索。策略全在 tools/search.py 里，这里只负责把它接进 Streamlit。

    复合问题拆成几个子查询、带个人信息的问题用意图重新构造查询——
    这些都是 search.py 的事，因为评测脚本也要用同一套逻辑。

    **注意：这个函数里只能有一个 docstring。**
    2026-11-09 出过一次事故：编辑时把新 docstring 追加在旧 docstring 后面，
    函数体里出现了两个字符串字面量。第二个不再是 docstring，而是裸表达式——
    Streamlit 的 magic 功能会把它自动 st.write() 到页面上，
    于是这段开发注释以代码块的形式出现在了页面最显眼的位置。
    """
    return search_mod.search(
        get_retriever(), query, k, rewrite=rewrite,
        client=get_embed_client(), model=CHAT_MODEL,
        # 传进来什么就用什么。调用方会传「不含当前问题」的历史——
        # 否则 search 内部取 history[-1] 会取到当前这句自己。
        history=history if history is not None else st.session_state.get("history", []),
    )


FEEDBACK_FILE = ROOT / "data" / "feedback.jsonl"


def log_feedback(question: str, best_sim: float, verdict: str, note: str = "",
                 kind: str = "") -> None:
    """记一条用户反馈，追加到 data/feedback.jsonl。

    为什么落盘而不是只打日志：
    这是**唯一能拿到真实用户问了什么**的渠道。评测集是我自己编的题，
    真实提问才是下一批评测题的来源——也是「有真实用户」这件事的证据。

    不记任何个人信息，只记问题和判定。
    """
    from datetime import datetime

    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "best_sim": round(best_sim, 4),
        "verdict": verdict,
        "note": note.strip(),
        "kind": kind,
        "from": "qa_page",
    }
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with FEEDBACK_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_prompt(question: str, context: str, intent: str | None,
                 slots: dict, missing: list) -> str:
    """按「用户提供了多少信息」拼不同的 prompt。

    三档（见 docs/产品判断-先给规则再引导.md）：

      信息完整  → 直接对照条件判断，说能/不能、差在哪
      信息不全  → 先完整列要求，结尾引导补信息
      没有意图  → 通用问答（和以前一样）

    **为什么不能只用一个 prompt**：
    「转专业有啥要求」如果按"判断类"prompt 喂给模型，它会硬编一个
    "你不能转"——用户什么信息都没给，凭什么判断。反过来
    「我大二绩点2.8能转吗」如果按"列规则"prompt 喂，它会把整篇制度
    抄一遍，用户还得自己对照。两种都要浪费用户时间。
    """
    base = PROMPT.format(context=context, question=question)

    if not intent:
        return base

    if slots and not missing:
        # 信息齐：直接判断
        info = "、".join(f"{k}={v}" for k, v in slots.items())
        return base + (
            f"\n\n【用户自己的情况】{info}\n\n"
            f"请**对照资料里的条件逐条判断**，明确回答：\n"
            f"1. 能不能（能/不能/需要补充其他条件）\n"
            f"2. 依据是资料里哪一条（标 [n]）\n"
            f"3. 如果不能或差一点，说明差在哪、有没有补救办法\n"
            f"不要只把制度抄一遍，用户要的是结论。\n\n"
            f"**⚠️ 必须先核对「适用范围」，这比条件本身更重要：**\n"
            f"- 资料里如果写了「限定 / 仅限 / 只面向 / 适用于」某个年级、专业类、学制，"
            f"必须先把它写出来\n"
            f"- 用户如果不在这个范围内，**条件再满足也要答「不能」**，并说明是范围不符\n"
            f"- 真实案例：学生大三、绩点 4.0 想转专业。资料写「绩点不低于3.00」，"
            f"同时写「第一～二款限定本科一年级」。只看绩点会答「能」，"
            f"但正确答案是「不能——绩点那条只对大一开放；大三只能走第八、九款"
            f"（兴趣专长 / 学习困难）」\n"
            f"- **不要只引用「条件」那一条，一定要找到「谁适用」那一条**"
        )

    if slots and missing:
        # 信息缺一部分：**先给能确定的判断，再列规则，最后引导**
        #
        # 2026-11-10 改：原来是「完整列规则 + 说缺什么 + 引导」。
        # 用户实测第二轮说「大三，计科专业」，系统列了 20 条规则（9 条基本条件
        # + 11 条不予转专业），末尾的「还缺绩点」引导直接被 max_tokens 截掉了。
        # 用户要的是判断，不是一份目录。
        info = "、".join(f"{k}={v}" for k, v in slots.items())
        need = "、".join(missing)
        return base + (
            f"\n\n【用户自己的情况】{info}（还不全，缺 {need}）\n\n"
            f"请**按这个顺序**回答，总共不超过 500 字：\n"
            f"1. **先用已知信息给出能确定的部分**。例如用户给了年级，就先说清"
            f"「你所在的年级能走哪几条、不能走哪几条」——这是资料里能确定的，不要留给用户自己比。\n"
            f"2. 再说：要完整判断还缺 {need}，以及为什么缺了这个就没法给结论。\n"
            f"3. 结尾一句话引导他补充。\n"
            f"**只挑和用户情况相关的条款说，不要把所有条款抄一遍。**\n"
            f"不要凭空判断「能不能」——信息不全时硬给结论比不回答更糟。"
        )

    # 完全没提供信息：列完整规则 + 引导
    return base + (
        f"\n\n用户没有提供自己的情况。请：\n"
        f"1. 完整、清楚地列出资料里关于这件事的**全部要求和限制**"
        f"（绩点门槛、年级限制、时间节点、需要提交的材料、流程）\n"
        f"2. 不要猜测用户的情况，也不要说「你可以/不可以」\n"
        f"3. 结尾引导：告诉他补充哪些信息，你就能帮他判断"
    )


def build_context(results) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        src = r["meta"]["source"].replace(".txt", "")
        parts.append(f"[{i}] 来源：{src}\n{r['doc']}")
    return "\n\n".join(parts)


# ---------------- 引用的三个后处理 ----------------
# Prompt 里要求模型自己标 [1][2]，但它经常忘。这三步是保底。

CITE_RE = re.compile(r"\[(\d+)\]")

# 只校验两类数字：小数（2.30）和"两位以上的整数"（170 / 72 / 2024）。
# 为什么跳过个位数：1、2、3 多是"第 1 学期""3 门"里的序号，校验它们全是误报。
# 为什么不连单位一起比对：资料写「170，」答案写「170 学分」，
# 整体匹配会对正确答案误报 —— 这是实测踩到的。
NUM_PAT = re.compile(r"\d+\.\d+|\d{2,}")


def has_citation(answer: str) -> bool:
    """答案里有没有 [n] 形式的引用标记。"""
    return bool(CITE_RE.search(answer))


def append_sources(answer: str, results) -> str:
    """模型没标引用时的兜底：末尾列出所有来源。

    局限：这是「整体来源列表」，不能精确到「哪句话来自哪一条」。
    要精确溯源得让模型逐句标，而模型不可靠 —— 所以这是次优但可用的方案。
    """
    if has_citation(answer):
        return answer
    lines = [answer.rstrip(), "", "---", "**参考来源**（按相关度排序）："]
    for i, r in enumerate(results, 1):
        src = r["meta"]["source"].replace(".txt", "")
        head = r["doc"].replace("\n", " ").strip()[:36]
        lines.append(f"{i}. 《{src}》—— {head}…")
    return "\n".join(lines)


def verify_numbers(answer: str, results) -> list[str]:
    """轻量校验：答案里的小数和两位数以上整数，是否都在资料里出现过。

    返回「资料里找不到的数字」列表。空列表 = 没发现问题。

    局限（必须知道）：
      1. 只能查「字面出现」，查不出「数字对但逻辑反了」
         （资料写"2.30 以上"，答案写"2.30 以下" → 校验通过）
      2. 同一个数字可能有多种含义（"170 学分"和"170 学时"都含 170），
         所以它只能提示"这个数字可疑"，不能断定错
    """
    # 校验池 = 正文 + 文件名。
    # 文件名也要算进来，因为「2024 级」这种年份在文件名里，正文里未必出现 ——
    # 不带上文件名就会把正确答案误报成"资料里没有"。
    pool = re.sub(
        r"\s+", "",
        "\n".join(r["doc"] + " " + r["meta"].get("source", "") for r in results),
    )
    missing = []
    for m in NUM_PAT.finditer(answer):
        token = m.group(0)
        if re.sub(r"\s+", "", token) not in pool:
            missing.append(token)
    # 去重且保持顺序
    return list(dict.fromkeys(missing))


def data_version_note(results) -> str | None:
    """从文件名里认出「2024 级」「2026 版」这种版本标记，给用户提示。"""
    for r in results:
        src = r["meta"]["source"]
        m = re.search(r"(20\d{2})\s*级", src) or re.search(r"(20\d{2})\s*版", src)
        if m:
            return f"数据版本：**{m.group(1)}** 的方案。其他年份的版本可能不同。"
    return None


def stream_via_gateway(prompt: str):
    """走 P1 网关（有降级、限流、计量）。

    鉴权用 GATEWAY_API_KEY，不是 SILICONFLOW_API_KEY —— 用错就是 401。
    """
    with httpx.stream(
        "POST",
        f"{GATEWAY}/chat",
        json={"query": prompt, "history_len": 0, "stream_format": "text"},
        headers={"X-API-Key": read_gateway_key()},
        timeout=90,
    ) as r:
        r.raise_for_status()
        for piece in r.iter_text():
            yield piece


def stream_direct(prompt: str):
    """直接调模型（网关没启动时的退路）。

    **用 DeepSeek 而不是免费的 Qwen2.5-7B。**
    2026-11-10 真实翻车：用户问「大三绩点4.0能转人工智能吗」，
    7B 答「不能，因为属于毕业班」——**结论碰巧对，理由是编的**（大三不是毕业班）。
    同一个 prompt 换 DeepSeek，回答准确到能写进法律文书。

    编理由比答错更危险：用户会拿着错理由去教务处，然后被当场驳回。
    成本约 0.004 元/次（输入 3000 + 输出 600 tokens），值得。
    """
    from openai import OpenAI

    client = OpenAI(api_key=read_env(judge.JUDGE_KEY_ENV), base_url=judge.JUDGE_BASE_URL)
    stream = client.chat.completions.create(
        model=judge.JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        max_tokens=800,
        temperature=0.2,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


# ---------------- 界面 ----------------


# ==================== 界面（对话流） ====================
#
# 2026-11-10：从「一次一问」改成聊天流。
# 为什么：多轮上下文逻辑早就做好了（用户回答追问时能接上），
# 但界面是单问单答的——用户看不到自己说过什么，也看不出系统记住了。
# 逻辑做了但用户感知不到，等于白做。

st.set_page_config(page_title="教务问答", page_icon="📚", layout="centered")
st.title("📚 教务问答")
st.caption("查西科大教务规定 · 每个答案都标出处")

gw_ok = gateway_available()

with st.sidebar:
    st.subheader("🔑 用自己的 Key")
    st.caption("不填也能用。填了你问多少都不会消耗站长的额度。")
    user_key = st.text_input(
        "硅基流动 API Key", type="password", placeholder="sk-...",
        label_visibility="collapsed",
        help="免费申请：https://cloud.siliconflow.cn/",
    )
    if user_key and user_key != st.session_state.get("_key_applied"):
        st.session_state["user_key"] = user_key.strip()
        st.session_state["_key_applied"] = user_key
        st.cache_resource.clear()
        st.rerun()
    if st.session_state.get("user_key"):
        st.success("正在用你自己的 Key ✅")
    else:
        st.caption("当前用站点配置的 Key。")

    k = 5
    backend = "直接调模型"
    with st.expander("⚙️ 高级设置"):
        k = st.slider("检索条数", 1, 10, 5)
        if gw_ok:
            backend = st.radio("生成后端", ["P1 网关", "直接调模型"], index=0)
        else:
            st.caption("当前直连模型服务。")
        st.caption(f"索引 {get_retriever().size} 个片段 · 模型 {EMBED_MODEL}")

    if st.button("🗑 清空对话", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["history"] = []
        st.session_state["_last_q"] = None
        st.rerun()


def show_feedback_block(question: str, best_sim: float = 0.0, context: str = "answer"):
    """反馈区。所有出口都挂：正常答、拒答、闲聊、无关问题。"""
    st.divider()
    st.markdown("### 这个答案帮到你了？")
    st.caption("**你的反馈对我真的很重要。**")
    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("👍 有用", key=f"fb_good_{context}", use_container_width=True):
        log_feedback(question, best_sim, "good", kind="答得挺好")
        st.success("谢谢！已记录。")
    if c2.button("👎 不对 / 没回答", key=f"fb_bad_{context}", use_container_width=True):
        st.session_state["show_feedback"] = True

    if st.session_state.get("show_feedback"):
        with st.form(f"detail_{context}", clear_on_submit=False):
            st.caption(f"你问的是：**{question}**（已自动带上，不用重打）")
            note = st.text_area(
                "哪里不对 / 正确答案是什么",
                placeholder="例如：它说资料里没有，但《2026 转专业通知》写着绩点不低于 3.00",
                height=90,
            )
            kind = st.radio(
                "这条属于",
                ["答错了", "答得不完整", "该答的没答", "它不该拒答", "其他"],
                horizontal=True,
            )
            if st.form_submit_button("提交反馈", type="primary"):
                log_feedback(question, best_sim, "bad", note=note, kind=kind)
                st.session_state["show_feedback"] = False
                st.success("收到，谢谢！站长会看。")


# ---------------- 对话历史 ----------------

if "messages" not in st.session_state:
    st.session_state["messages"] = []      # [{"role","content"}] 用于渲染
if "history" not in st.session_state:
    st.session_state["history"] = []       # [{"user"}] 用于检索拼接

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("materials"):
            with st.expander(f"📎 相关材料（{len(msg['materials'])} 条）", expanded=False):
                for i, (src, head) in enumerate(msg["materials"], 1):
                    st.markdown(f"**[{i}]** {src}")
                    st.caption(head)

prompt = st.chat_input("问点什么，比如：转专业需要什么条件")

if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 检索拼接用的历史：只记用户说了什么
    st.session_state["history"].append({"user": prompt})
    st.session_state["history"] = st.session_state["history"][-5:]

    with st.chat_message("assistant"):
        # ---------- 1. 闲聊 ----------
        if judge.is_smalltalk(prompt):
            st.markdown(judge.SMALLTALK_REPLY)
            st.session_state["messages"].append(
                {"role": "assistant", "content": judge.SMALLTALK_REPLY}
            )
            show_feedback_block(prompt, context="chat")
            st.stop()

        # ---------- 2. 先判断这轮是不是在承接上一轮 ----------
        #
        # **必须放在 off_topic 判断之前。**
        # 真实翻车：用户第一轮问「转专业条件」，第二轮答「大三，计科专业」。
        # 这句话里没有任何教务关键词，refusal_kind 判成「完全无关」直接秒拒——
        # 承接判断根本没机会跑。而它明明是在回答上一轮的追问。
        #
        # 所以顺序必须是：先拼接 → 再用拼接后的问题判断「是不是无关」。
        _hist_all = st.session_state["history"]
        followup = False
        prev_q = ""
        if len(_hist_all) >= 2:
            prev_q = _hist_all[-2].get("user", "")
            _ck = (prev_q, prompt)
            if st.session_state.get("_fu_key") != _ck:
                st.session_state["_fu_key"] = _ck
                st.session_state["_fu_val"] = search_mod.is_followup(
                    prompt, _hist_all[:-1], get_embed_client(), CHAT_MODEL
                )
            followup = bool(st.session_state["_fu_val"])

        effective = f"{prev_q} {prompt}" if followup else prompt

        # ---------- 3. 完全无关（用 effective 判断，不是原句） ----------
        if refusal_kind(effective) == "off_topic":
            joke = fun_reply(prompt) or pick_choice(prompt)
            if joke:
                st.markdown(f"### {joke}")
                st.caption("（这个真不归我管，纯粹忍不住插一句 😅）")
                out = joke
            else:
                out = ("这个我真答不了 😅\n\n"
                       "我只懂西安科技大学的教务政策——学籍、转专业、学分、毕业这些。")
                st.info(out)
            st.markdown("**正经问题可以问我：** 转专业要什么条件 · 绩点2.1能拿学位吗 · 挂科了怎么办")
            st.session_state["messages"].append({"role": "assistant", "content": out})
            show_feedback_block(prompt, context="offtopic")
            st.stop()

        # ---------- 4. 正常流程 ----------
        # intent 也要基于 effective 算。
        # 踩过的坑：第二轮「大三，绩点2.8」含「绩点」会被识别成「学位」问题，
        # 用户问转专业，它答能不能拿学位证。
        intent = intake.detect_intent(effective)
        slots = intake.extract_slots(effective, intent) if intent else {}
        missing = intake.missing_slots(intent, slots) if intent else []

        with st.spinner("检索中…"):
            results, best = retrieve(prompt, k, history=_hist_all[:-1])

        materials = [
            (r["meta"]["source"].replace(".txt", ""),
             r["doc"].replace("\n", " ").strip()[:80] + "…")
            for r in results
        ]

        if intent:
            if slots and not missing:
                st.caption(f"识别到你在问「{intent}」，信息齐全 → 直接给判断")
            elif slots:
                st.caption(f"识别到你在问「{intent}」，还差「{'、'.join(missing)}」→ 先列要求")
            else:
                st.caption(f"识别到你在问「{intent}」→ 先列要求，需要时再补信息")

        # 判断层
        can_answer = True
        if best < judge.FAST_PATH:
            judge_q = prompt
            _hist = st.session_state["history"]
            if len(_hist) >= 2:
                _ck = (_hist[-2].get("user", ""), prompt)
                if st.session_state.get("_fu_key") != _ck:
                    st.session_state["_fu_key"] = _ck
                    st.session_state["_fu_val"] = search_mod.is_followup(
                        prompt, _hist[:-1], get_embed_client(), CHAT_MODEL
                    )
                if st.session_state["_fu_val"]:
                    judge_q = f"{_hist[-2].get('user', '')} {prompt}"
            with st.spinner("确认资料是否够用…"):
                ok = judge.check_answerable(
                    judge_q, build_context(results), get_judge_client(), judge.JUDGE_MODEL
                )
            if not ok:
                can_answer = False

        if not can_answer:
            kind = refusal_kind(prompt)
            if kind == "campus":
                st.warning("**这类事不归教务处管**，所以我的资料里没有。\n\n但我可以告诉你该找谁。")
            else:
                st.warning(f"**这个问题不在我收录的教务文件里。**\n\n"
                           f"检索到的内容和问题相关（相似度 {best:.3f}），但资料里没有能直接回答的具体内容。")
            st.markdown("#### 你可以去这些地方查")
            for title, steps in suggest_where(prompt):
                st.markdown(f"**{title}**")
                for s in steps:
                    if s.strip():
                        st.markdown(f"- {s}")
            with st.expander("我收录了哪些文件", expanded=False):
                st.markdown(
                    "学籍 · 学位 · 学分认定 · 辅修 / 微专业 · 转专业 · 重修 · "
                    "本硕连读 · 课程设置\n\n共 146 份文件、约 46.7 万字，全部来自"
                    "西安科技大学教务处公开文件，不含任何个人信息。"
                )
            head = "这个问题的答案不在我收录的教务文件里。"
            st.session_state["messages"].append({"role": "assistant", "content": head})
            show_feedback_block(prompt, best, context="refuse")
            st.stop()

        # 生成
        # **传 effective 而不是 prompt。**
        # 用户第二轮说「大三，绩点2.8」，如果只把这个当成"问题"，
        # 模型不知道他在问什么，会从 10 条资料里随便挑一个（比如学位）。
        # 拼上上一轮的「转专业要什么条件」它才明白这是在补充信息。
        prompt_text = build_prompt(effective, build_context(results), intent, slots, missing)
        placeholder = st.empty()
        acc = ""
        try:
            gen = stream_via_gateway(prompt_text) if "网关" in backend else stream_direct(prompt_text)
            for piece in gen:
                acc += piece
                placeholder.markdown(acc + "▌")

            final = append_sources(acc, results)
            placeholder.markdown(final)

            miss = verify_numbers(final, results)
            if miss:
                st.warning(f"⚠️ 答案里这些数字在资料里没找到，**可能不准确**：{'、'.join(miss[:6])}")
            else:
                st.caption("✅ 答案里的关键数字都能在资料里对上")
        except Exception as e:  # noqa: BLE001
            placeholder.empty()
            st.error(f"生成失败：{type(e).__name__}: {str(e)[:200]}")
            final = "生成失败，请重试。"
            st.stop()

        with st.expander(f"📎 相关材料（{len(materials)} 条）", expanded=False):
            for i, (src, head) in enumerate(materials, 1):
                st.markdown(f"**[{i}]** {src}")
                st.caption(head)

        st.session_state["messages"].append(
            {"role": "assistant", "content": final, "materials": materials}
        )
        show_feedback_block(prompt, best, context="answer")

st.divider()
st.caption(f"数据来源：西安科技大学教务处公开文件 · 共 {get_retriever().size} 个检索片段")
st.caption("⚠️ 本系统只覆盖制度类文件，**不含**师资名录、课表、成绩、宿舍/食堂信息")
