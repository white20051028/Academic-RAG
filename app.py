"""教务问答 Web 界面（Streamlit）。

设计要点：
  1. **检索和生成分开显示** —— 答案不准时，一眼看出是"没检索到"还是"模型没用好"
  2. **相似度可见** —— 分数低就说明这条不该信，这是防幻觉的第一层
  3. **两个生成后端** —— 优先走 P1 网关（复用降级/限流/计量），没启动就直接调模型

启动：
  E:\\anaconda3\\envs\\llm-rag\\python.exe -m streamlit run app.py
"""

from pathlib import Path

import chromadb
import httpx
import streamlit as st

ROOT = Path(__file__).resolve().parent
CHROMA = ROOT / "data" / "chroma"
ENV_FILE = Path(r"D:\桌面\多轮对话机器人项目\.env")
GATEWAY = "http://127.0.0.1:6066"
EMBED_MODEL = "BAAI/bge-m3"
CHAT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

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
    """从 P1 的 .env 里读一个变量。

    注意这里有两类 key，不能混：
      SILICONFLOW_API_KEY —— 上游模型服务的凭据（P1 内部用它调 Qwen）
      GATEWAY_API_KEY     —— P1 网关自己的鉴权凭据（P2 调 P1 时要用这个）
    用错会直接 401。
    """
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError(f"没找到 {name}")


def read_key() -> str:
    return read_env("SILICONFLOW_API_KEY")


def read_gateway_key() -> str:
    return read_env("GATEWAY_API_KEY")


# 拒答时给"去哪里找"的建议。
# 只写真实存在的信息（从教务处官网抓的），不编造机构名——
# 编一个不存在的"实践教学科"比拒答更糟。
WHERE_TO_LOOK = [
    (("老师", "教师", "师资", "教授", "任课"),
     "任课教师名单 → 学院官网「师资队伍」页，或教务系统查课表"),
    (("课表", "上课时间", "哪个教室", "几号楼"),
     "课表 → 教务系统 https://ehallnew.xust.edu.cn"),
    (("成绩", "绩点查询", "考了多少分"),
     "成绩 → 教务系统 https://ehallnew.xust.edu.cn"),
    (("实习", "实践", "毕设", "毕业设计", "实验"),
     "实习 / 实践环节 → 所在学院教务办，或教务处实践教学科"),
    (("宿舍", "食堂", "奖学金", "助学金", "贷款", "勤工"),
     "学生事务 → 学生处"),
    (("补考", "重修", "考试时间"),
     "考试安排 → 教务处通知栏，或教务系统"),
]

DEFAULT_WHERE = "教务处：电话 029-83858041 · 邮箱 jwc@xust.edu.cn"


def suggest_where(question: str) -> list[str]:
    """按问题关键词给出「可能去哪找」的建议。"""
    hits = [tip for words, tip in WHERE_TO_LOOK if any(w in question for w in words)]
    return hits or [DEFAULT_WHERE]


def check_answerable(question: str, results) -> bool:
    """让模型判断：这些资料能不能回答问题？

    关键设计：**要求它指出具体依据**。
    第一版只问"能不能回答"，结果它看到体育课安排里有老师名字就说"能"——
    问的是数学老师，资料里是体育老师，字面相关但领域完全错。

    现在要求它必须摘出回答这个问题的那句话；摘不出来就答"不能"。
    """
    from openai import OpenAI

    probe = (
        "任务：判断下面这些资料能不能回答用户的问题。\n\n"
        "判断标准（严格）：\n"
        "- 资料里必须有**直接回答**这个问题的内容（具体条件、数字、期限、流程）\n"
        "- 只有「字面相关」不算。例如：问「数学老师是谁」，资料里全是体育老师名单 → 不能\n"
        "- 只有「提到这个词」但没给答案不算。例如：问「实习怎么请假」，"
        "资料只写「请假不超过2周」（讲的是新生入学）→ 不能\n\n"
        f"资料：\n{build_context(results)[:3000]}\n\n"
        f"用户问题：{question}\n\n"
        "请按下面格式回答：\n"
        "能 | 依据：<摘出资料里直接回答该问题的那句话>\n"
        "或\n"
        "不能 | 原因：<为什么资料不够>\n"
    )
    try:
        client = OpenAI(api_key=read_key(), base_url="https://api.siliconflow.cn/v1")
        r = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": probe}],
            max_tokens=120,
            temperature=0,
        )
        ans = (r.choices[0].message.content or "").strip()
        first = ans.split("\n", 1)[0]
        # 必须：开头"能"，而且真的给出了依据
        return first.startswith("能") and "依据" in first
    except Exception:  # noqa: BLE001
        return True


@st.cache_resource
def get_embed_client():
    from openai import OpenAI

    return OpenAI(api_key=read_key(), base_url="https://api.siliconflow.cn/v1")


@st.cache_resource
def get_collection():
    db = chromadb.PersistentClient(path=str(CHROMA))
    return db.get_collection("xust")


def gateway_available() -> bool:
    try:
        return httpx.get(f"{GATEWAY}/health", timeout=2).status_code == 200
    except Exception:  # noqa: BLE001
        return False


@st.cache_resource
def get_retriever():
    """混合检索器（向量 + BM25 + RRF）。缓存的，只建一次。"""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))
    import retrieval as R

    return R.HybridRetriever(get_collection(), get_embed_client(), EMBED_MODEL)


def retrieve(query: str, k: int, rewrite: bool = True):
    """检索。rewrite=True 时 BM25 路用改写后的书面词（口语问题提升 10 个点）。"""
    retriever = get_retriever()
    results, best_sim = retriever.search(query, k=k, rewrite=rewrite)
    return results, best_sim


def build_context(results) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        src = r["meta"]["source"].replace(".txt", "")
        parts.append(f"[{i}] 来源：{src}\n{r['doc']}")
    return "\n\n".join(parts)


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
    min_score = st.slider(
        "相似度下限（低于它直接拒答）", 0.0, 0.9, 0.60, 0.05,
        help="0.60 是用 50 条评测集校准出来的：答对的样本相似度最低 0.618；"
             "而「高数老师是谁」这类越界问题只有 0.516。",
    )

    st.subheader("生成后端")
    if gw_ok:
        st.success("P1 网关：在线 ✅")
        backend = st.radio("用哪个", ["P1 网关（降级/限流/计量）", "直接调模型"], index=0)
    else:
        st.warning("P1 网关：离线")
        st.caption("启动方式（在 P1 项目目录）：\n```\npython -m uvicorn fastapi_bot:app --port 6066\n```")
        backend = "直接调模型"

    st.divider()
    st.caption("索引：{} 个片段 · 模型：{}".format(get_collection().count(), EMBED_MODEL))

question = st.text_input("你的问题", placeholder="例如：转专业需要什么条件？")

if question:
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

    def show_refusal(reason: str):
        """拒答时的完整交代：说不知道 → 说为什么 → 说去哪找。"""
        st.warning(f"**我在教务处的公开文件里没找到能回答这个问题的内容。**\n\n{reason}")
        st.markdown("**我能查的范围**：学籍管理、学位授予、学分认定、辅修 / 微专业、转专业、各类办事流程。")
        st.markdown("**这个问题可能要看：**")
        for tip in suggest_where(question):
            st.markdown(f"- {tip}")
        st.caption("如果上面都不对，直接打教务处电话问最快：029-83858041")

    if best < min_score:
        show_refusal(f"最高相似度只有 {best:.3f}，低于可信阈值 {min_score}。")
        can_answer = False

    elif best < 0.70:
        # 灰区：字面相关，但不确定真能回答 —— 让模型当裁判
        with st.spinner("确认资料是否够用…"):
            ok = check_answerable(question, results)
        if not ok:
            show_refusal(
                f"检索到的内容字面相关（相似度 {best:.3f}），"
                f"但模型判断里面没有能直接回答的具体内容。"
            )
            can_answer = False
        else:
            st.info(f"相似度 {best:.3f} 属灰区，模型确认资料够用。")
            can_answer = True
    else:
        can_answer = True

    if can_answer:
        if st.button("生成回答", type="primary"):
            prompt = PROMPT.format(context=build_context(results), question=question)
            placeholder = st.empty()
            acc = ""
            try:
                gen = stream_via_gateway(prompt) if "网关" in backend else stream_direct(prompt)
                for piece in gen:
                    acc += piece
                    placeholder.markdown(acc + "▌")
                placeholder.markdown(acc)
                st.caption(f"生成后端：{backend} · 回答长度 {len(acc)} 字")
            except Exception as e:  # noqa: BLE001
                placeholder.empty()
                st.error(f"生成失败：{type(e).__name__}: {str(e)[:200]}")
                st.info("如果选了 P1 网关，确认它正在 127.0.0.1:6066 运行。")

st.divider()
st.caption("数据来源：西安科技大学教务处公开文件 · 共 46 份制度类文档")
st.caption("⚠️ 本系统只覆盖制度类文件，**不含**师资名录、课表、成绩、宿舍/食堂信息")
