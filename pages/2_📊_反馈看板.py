"""反馈看板：站长用。看统计、看全部反馈、导出。

**为什么要密码**

这个页面能看到所有人提的问题（可能含学号、姓名这类信息）。
不加密码等于把别人的隐私挂在公网上。

密码从 `.env` 的 `ADMIN_PASSWORD` 读；没配就用默认值并在页面顶部提醒改。
"""

import json
import sys
from collections import Counter
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import paths  # noqa: E402

FEEDBACK = ROOT / "data" / "feedback.jsonl"

st.set_page_config(page_title="反馈看板", page_icon="📊")


def admin_password() -> tuple[str, bool]:
    """返回 (密码, 是不是用的默认值)。"""
    try:
        return paths.read_env("ADMIN_PASSWORD"), False
    except Exception:  # noqa: BLE001
        return "admin", True


st.title("📊 反馈看板")

real, is_default = admin_password()

if is_default:
    st.warning(
        "⚠️ 正在用默认密码 `admin`。在 `.env` 里加一行 `ADMIN_PASSWORD=你的密码` 就能改掉。"
    )

pwd = st.text_input("密码", type="password")
if pwd != real:
    st.info("输入密码后查看。")
    st.stop()

if not FEEDBACK.exists():
    st.info("还没有任何反馈。")
    st.stop()

rows = [
    json.loads(line)
    for line in FEEDBACK.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

# ---------------- 统计 ----------------

c1, c2, c3, c4 = st.columns(4)
c1.metric("反馈总数", len(rows))
good = sum(1 for r in rows if r.get("verdict") == "good")
c2.metric("👍 有用", good)
c3.metric("👎 有问题", len(rows) - good)
if good:
    c4.metric("好评率", f"{good/len(rows):.0%}")

kinds = Counter(r.get("kind", "未分类") for r in rows)
if kinds:
    st.caption("按类型：" + " · ".join(f"{k} {v}" for k, v in kinds.most_common()))

# ---------------- 列表 ----------------

st.divider()
st.subheader(f"全部反馈（{len(rows)} 条）")

# 只看有文字的，👍/👎 那类没内容
only_text = st.checkbox("只看写了文字的", value=False)
shown = [r for r in rows if r.get("note")] if only_text else rows

for r in reversed(shown):
    icon = "👍" if r.get("verdict") == "good" else "👎"
    with st.container(border=True):
        st.markdown(f"{icon} **{r.get('question', '（没写问题）')}**")
        if r.get("note"):
            st.markdown(f"　{r['note']}")
        sim = r.get("best_sim")
        meta = f"{r.get('time', '')} · {r.get('kind', '未分类')}"
        if sim is not None:
            meta += f" · 相似度 {sim}"
        st.caption(meta)

# ---------------- 导出 ----------------

st.divider()
csv = "time,question,verdict,kind,note\n" + "\n".join(
    '"{}","{}","{}","{}","{}"'.format(
        r.get("time", ""),
        r.get("question", "").replace('"', "'"),
        r.get("verdict", ""),
        r.get("kind", ""),
        r.get("note", "").replace('"', "'"),
    )
    for r in rows
)
st.download_button(
    "⬇️ 导出 CSV",
    data=csv.encode("utf-8-sig"),   # utf-8-sig：Excel 打开不乱码
    file_name="feedback.csv",
    mime="text/csv",
)
