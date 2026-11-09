"""反馈：用户提交 + 看最近大家提了什么。

**为什么要这个页面**

原来的 👍/👎 只记了「好/不好」，站长看不到**哪里不好**。
这个页面让人能写一句：「转专业那条，绩点要求是 3.0，它答成 2.8 了。」

**为什么反馈要落盘而不是发微信**

1. 微信里聊完就散了，没法统计
2. 攒够 20 条就是「有真实用户反馈」的证据，简历能写
3. 反馈里的问题可以直接变成下一批评测题

数据存在 data/feedback.jsonl，不进 git（可能含个人信息）。
"""

import json
from datetime import datetime
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
FEEDBACK = ROOT / "data" / "feedback.jsonl"

st.set_page_config(page_title="反馈", page_icon="💬")
st.title("💬 反馈")
st.caption("哪里答错了、哪里没说清楚，都可以写在这里。站长会看。")

# ---------------- 表单 ----------------

with st.form("feedback", clear_on_submit=True):
    question = st.text_input(
        "你问的问题（必填）",
        placeholder="例如：转专业要多少绩点",
    )
    note = st.text_area(
        "哪里不对 / 正确答案是什么（选填，但写了最有帮助）",
        placeholder="例如：它说资料里没有，但通知里写着绩点不低于 3.00",
        height=100,
    )
    kind = st.radio(
        "这条反馈是",
        ["答错了", "答得不完整", "答得挺好", "其他"],
        horizontal=True,
    )
    submitted = st.form_submit_button("提交", type="primary")

if submitted:
    if not question.strip():
        st.error("至少写一下你问的是什么")
    else:
        record = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "question": question.strip(),
            "note": note.strip(),
            "verdict": {"答得挺好": "good"}.get(kind, "bad"),
            "kind": kind,
            "from": "feedback_page",
        }
        FEEDBACK.parent.mkdir(parents=True, exist_ok=True)
        with FEEDBACK.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        st.success("收到，谢谢！站长会看。")

# ---------------- 最近反馈 ----------------

st.divider()
st.subheader("最近大家提的")

if not FEEDBACK.exists():
    st.info("还没有反馈。你可以是第一个。")
else:
    rows = [
        json.loads(line)
        for line in FEEDBACK.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # 只展示来自反馈页的（👍/👎 那类没有文字，展示出来没意义）
    shown = [r for r in rows if r.get("from") == "feedback_page"][-10:]
    if not shown:
        st.info("还没有文字反馈。")
    else:
        for r in reversed(shown):
            with st.container(border=True):
                st.markdown(f"**{r['question']}**")
                if r.get("note"):
                    st.markdown(f"　{r['note']}")
                st.caption(f"{r['time']} · {r.get('kind', '')}")
