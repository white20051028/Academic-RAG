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
DEFAULT_WHERE = (
    "这类问题不在教务处职责范围内。试这三条路：\n"
    "        · **教务系统**（59.74.174.150/jwglxt）→ 信息查询，看有没有对应入口\n"
    "        · **学校官网 / 学院官网** → 搜关键词 + 「西安科技大学」\n"
    "        · 最后才考虑：教务处 029-83858041 · jwc@xust.edu.cn"
)


def suggest_where(question: str) -> list[tuple[str, list[str]]]:
    """按问题关键词给出「去哪查、点哪里」。返回 [(标题, [步骤])]。"""
    hits = [(title, steps) for words, title, steps in WHERE_TO_LOOK
            if any(w in question for w in words)]
    return hits if hits else [("下一步", DEFAULT_WHERE.split("\n"))]


@st.cache_resource
def get_embed_client():
    from openai import OpenAI

    return OpenAI(api_key=read_key(), base_url="https://api.siliconflow.cn/v1")


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


def retrieve(query: str, k: int, rewrite: bool = True):
    """检索。复合问题会拆成几个子查询，分别检索后合并。

    vec_mode="orig" 是量出来的，不是拍的：向量路也用改写后，口语集
    Recall@5 从 83% 掉到 79%。改写把原句的语序和限定词抹平了，
    而向量路恰恰靠这些区分"保研"和"推免材料归档"。

    **为什么合并要"轮流取"而不是按分数排序**：
    「辅修能抵学分吗，最多抵多少」拆成两个子查询，第一个子查询的 5 条
    通常分数都比第二个高。如果直接按 RRF 排序取 top-5，第二个子查询的
    结果会被挤光——等于拆了也没用。轮流取保证每个子问题都有代表。
    """
    """检索。策略全在 tools/search.py 里，这里只负责把它接进 Streamlit。

    复合问题拆成几个子查询、带个人信息的问题用意图重新构造查询——
    这些都是 search.py 的事，因为评测脚本也要用同一套逻辑。
    """
    return search_mod.search(
        get_retriever(), query, k, rewrite=rewrite,
        client=get_embed_client(), model=CHAT_MODEL,
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
            f"不要只把制度抄一遍，用户要的是结论。"
        )

    if slots and missing:
        # 信息缺一部分：列规则 + 指出缺什么
        info = "、".join(f"{k}={v}" for k, v in slots.items())
        need = "、".join(missing)
        return base + (
            f"\n\n【用户自己的情况】{info}（还不全）\n\n"
            f"请：\n"
            f"1. 完整列出资料里关于这件事的**要求和限制**（条件、数字、流程）\n"
            f"2. 告诉用户：按已知信息，目前还缺 {need} 才能判断\n"
            f"3. 结尾用一句话引导他补充这些信息\n"
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
    """直接调模型（网关没启动时的退路）。"""
    from openai import OpenAI

    client = OpenAI(api_key=read_key(), base_url="https://api.siliconflow.cn/v1")
    stream = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        max_tokens=800,
        temperature=0.2,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


# ---------------- 界面 ----------------

st.set_page_config(page_title="教务问答", page_icon="📚", layout="wide")
st.title("📚 教务问答")
st.caption("基于教务处公开文件的检索问答 · 回答会标注出自哪一条资料")

gw_ok = gateway_available()

with st.sidebar:
    st.subheader("参数")
    k = st.slider("召回条数", 1, 10, 5)
    st.caption(
        "拒答不再看相似度阈值——由判断层读资料内容后决定。"
        "相似度仍在「检索结果」里显示，用来判断检索质量。"
    )

    st.divider()
    st.subheader("🔑 用你自己的 Key")
    user_key = st.text_input(
        "硅基流动 API Key",
        type="password",
        placeholder="sk-...",
        help="填你自己的 key，问答就不消耗站长的额度。不填则用服务器上配置的那个。"
             "免费申请：https://cloud.siliconflow.cn/",
    )
    if user_key and user_key != st.session_state.get("_key_applied"):
        st.session_state["user_key"] = user_key.strip()
        st.session_state["_key_applied"] = user_key
        # 换了 key 必须清缓存，否则检索器还在用旧 client
        st.cache_resource.clear()
        st.rerun()
    if st.session_state.get("user_key"):
        st.success("正在用你自己的 Key ✅")
    else:
        st.caption("当前用的是站点配置的 Key。")

    st.subheader("生成后端")
    if gw_ok:
        st.success("P1 网关：在线 ✅")
        backend = st.radio("用哪个", ["P1 网关（降级/限流/计量）", "直接调模型"], index=0)
    else:
        st.warning("P1 网关：离线")
        st.caption("启动方式（在 P1 项目目录）：\n```\npython -m uvicorn fastapi_bot:app --port 6066\n```")
        backend = "直接调模型"

    st.divider()
    st.caption("索引：{} 个片段 · 模型：{}".format(get_retriever().size, EMBED_MODEL))

question = st.text_input("你的问题", placeholder="例如：转专业需要什么条件？")

if question:
    # 闲聊在检索之前拦住。这类问题检索出来的是垃圾，走拒答流程会回一段
    # "教务处电话"，比不回答更糟。规则判断，不调模型，零延迟。
    if judge.is_smalltalk(question):
        st.info(judge.SMALLTALK_REPLY)
        st.stop()

    # 看用户提供了多少信息 —— 决定后面用哪种回答方式
    intent = intake.detect_intent(question)
    slots = intake.extract_slots(question, intent) if intent else {}
    missing = intake.missing_slots(intent, slots) if intent else []

    with st.spinner("检索中…"):
        results, best = retrieve(question, k)

    st.subheader("① 检索结果")
    cols = st.columns([1, 1, 1])
    cols[0].metric("最高相似度", f"{best:.3f}")
    cols[1].metric("召回条数", len(results))
    cols[2].metric("来源文件", len({r["meta"]["source"] for r in results}))

    for i, r in enumerate(results, 1):
        meta, doc = r["meta"], r["doc"]
        title = f"[{i}] {meta['source'].replace('.txt', '')}"
        with st.expander(title, expanded=(i <= 2)):
            st.text(doc)

    st.subheader("② 回答")

    # 让用户知道系统是怎么理解他的问题的（透明度）。
    # 这也是产品判断的落地：用户不提供信息时，系统不硬编结论，而是说明缺什么。
    if intent:
        if slots and not missing:
            st.caption(f"识别到你在问「{intent}」，信息齐全 → 直接给判断")
        elif slots:
            st.caption(f"识别到你在问「{intent}」，还差「{'、'.join(missing)}」→ 先列要求")
        else:
            st.caption(f"识别到你在问「{intent}」→ 先列要求，需要时再补信息")

    def show_refusal(reason: str):
        """拒答时的完整交代：说不知道 → 说为什么 → **给能立刻动手的下一步**。"""
        st.warning(f"**这个问题的答案不在我收录的教务文件里。**\n\n{reason}")
        st.markdown(
            "**我收录的是**：学籍、学位、学分认定、辅修 / 微专业、转专业、"
            "重修、本硕连读、课程设置（73 份，20 万字）。"
        )
        st.markdown("**你可以在这些地方查到：**")
        for title, steps in suggest_where(question):
            st.markdown(f"**{title}**")
            for s in steps:
                if s.strip():
                    st.markdown(f"- {s}")

    # 判定只看「资料够不够」，不看相似度分数。
    #
    # 0.60 那道闸门 2026-11-06 拆掉了。起因：问「我想服兵役」，向量分只有 0.544，
    # 被闸门直接拒——但《学籍管理规定》里写着「应征参加解放军，保留学籍至退役后2年」，
    # 判断层也能一字不差地摘出这句。分数低不等于没答案，只说明问法和文档用词不一样。
    # 实测全量走判断层：答非所问 3→0，该拒没拒 2→0，总分不降。见 docs/v8。
    if best >= judge.FAST_PATH:
        can_answer = True
    else:
        with st.spinner("确认资料是否够用…"):
            ok = judge.check_answerable(
                question, build_context(results), get_embed_client(), CHAT_MODEL
            )
        if ok:
            can_answer = True
            if best < judge.DEFAULT_MIN_SCORE:
                st.info(
                    f"相似度只有 {best:.3f}（低于旧阈值 {judge.DEFAULT_MIN_SCORE}），"
                    f"但模型在资料里找到了能回答的依据。"
                )
        else:
            show_refusal(
                f"检索到的内容和问题相关（相似度 {best:.3f}），"
                f"但资料里没有能直接回答的具体内容。"
            )
            can_answer = False

    if can_answer:
        # 引用区先显示：检索完就有，不用等生成
        st.markdown("**📎 参考资料**（答案里的 `[1][2]` 对应这里）")
        for i, r in enumerate(results, 1):
            src = r["meta"]["source"].replace(".txt", "")
            head = r["doc"].replace("\n", " ").strip()[:90]
            st.markdown(f"**[{i}]** 《{src}》  \n<span style='color:gray'>{head}…</span>",
                        unsafe_allow_html=True)

        # 数据版本提示（防误导：2024 级方案不等于菁英班方案）
        note = data_version_note(results)
        if note:
            st.caption(f"⚠️ {note}")

        st.divider()

        if st.button("生成回答", type="primary"):
            prompt = build_prompt(question, build_context(results), intent, slots, missing)
            placeholder = st.empty()
            acc = ""
            try:
                gen = stream_via_gateway(prompt) if "网关" in backend else stream_direct(prompt)
                for piece in gen:
                    acc += piece
                    placeholder.markdown(acc + "▌")

                # 后处理 1：模型没标 [n] 就在末尾列出来源
                final = append_sources(acc, results)
                placeholder.markdown(final)

                # 后处理 2：轻量校验（小数 + 带单位数字）
                missing = verify_numbers(final, results)
                if missing:
                    st.warning(
                        f"⚠️ 答案里这些数字在资料里没找到，**可能不准确**："
                        f"{'、'.join(missing[:6])}"
                    )
                else:
                    st.caption("✅ 答案里的关键数字都能在资料里对上")

                st.caption(f"生成后端：{backend} · 回答 {len(final)} 字")
            except Exception as e:  # noqa: BLE001
                placeholder.empty()
                st.error(f"生成失败：{type(e).__name__}: {str(e)[:200]}")
                st.info("如果选了 P1 网关，确认它正在 127.0.0.1:6066 运行。")

    # ---------------- 反馈 ----------------
    # 为什么放在页面内展开，而不是跳到「反馈」页面：
    # 多页面切换会清空问答页的状态（问题、答案、检索结果全没）。
    # 用户为了反馈要重新输入、重新等待、还要担心刷新丢失——成本翻倍，直接劝退。
    # 所以反馈必须**不离开当前页面**，并且自动带上他问的问题。
    st.divider()
    st.markdown("**这个答案对你有用吗？**")
    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("👍 有用", key="fb_good"):
        log_feedback(question, best, "good", kind="答得挺好")
        st.success("谢谢！已记录。")
    if c2.button("👎 不对 / 没回答", key="fb_bad"):
        st.session_state["show_feedback"] = True

    if st.session_state.get("show_feedback"):
        with st.form("detail_feedback", clear_on_submit=False):
            st.caption(f"你问的是：**{question}**（已自动带上，不用重打）")
            note = st.text_area(
                "哪里不对 / 正确答案是什么",
                placeholder="例如：它说资料里没有，但《2026 转专业通知》写着绩点不低于 3.00",
                height=90,
            )
            kind = st.radio(
                "这条属于", ["答错了", "答得不完整", "该答的没答", "其他"],
                horizontal=True,
            )
            if st.form_submit_button("提交反馈", type="primary"):
                log_feedback(question, best, "bad", note=note, kind=kind)
                st.session_state["show_feedback"] = False
                st.success("收到，谢谢！站长会看。")

st.divider()
st.caption(f"数据来源：西安科技大学教务处公开文件 · 共 {get_retriever().size} 个检索片段")
st.caption("⚠️ 本系统只覆盖制度类文件，**不含**师资名录、课表、成绩、宿舍/食堂信息")
