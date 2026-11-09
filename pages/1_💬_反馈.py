"""看看大家提了什么。

**这里不能提交反馈，只能看。**

原因：Streamlit 多页面切换会清空问答页的状态（问题、答案、检索结果全没）。
如果让用户在这个页面填反馈，他得切回来重新问一遍才能对照——成本翻倍，直接劝退。

所以提交入口在**问答页**：答案下面点「👎 不对」，就地展开，问题自动带上。

这个页面只负责让你知道「别人也在用、别人提了什么」。
"""

import json
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
FEEDBACK = ROOT / "data" / "feedback.jsonl"

st.set_page_config(page_title="大家的反馈", page_icon="💬")
st.title("💬 大家的反馈")

st.info(
    "**想提反馈？** 回到「问答」页，在答案下面点「👎 不对 / 没回答」，"
    "就在原地展开填写，你问的问题会自动带上，不用重打。"
)

if not FEEDBACK.exists():
    st.caption("还没有反馈。")
    st.stop()

rows = [
    json.loads(line)
    for line in FEEDBACK.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

good = sum(1 for r in rows if r.get("verdict") == "good")
c1, c2 = st.columns(2)
c1.metric("累计反馈", len(rows))
c2.metric("好评", good)

st.divider()

# 只展示有文字的（👍/👎 没内容，展示出来没意义）
text_rows = [r for r in rows if r.get("note")]
if not text_rows:
    st.caption("还没有写文字的反馈。")
else:
    for r in reversed(text_rows[-10:]):
        with st.container(border=True):
            st.markdown(f"**{r.get('question', '（没写问题）')}**")
            st.markdown(f"　{r['note']}")
            st.caption(f"{r.get('time', '')} · {r.get('kind', '')}")
