"""切分策略（v1）：条文式文档按「第X条」切，其余按固定窗口。

为什么要两种：
  70 份文档里只有 8 份有「第X条」结构，但这 8 份是最核心的制度文件
  （学籍、学位、学分、辅修、微专业、本硕连读）。

  v0 用固定 500 字窗口，实测出现了一个典型失败：
    「转专业的绩点要求是多少」→ 文件召回对了，但 top-5 的 chunk 里没有 "3.00"
    —— 因为一条完整的条款被切成两半，条件在前半、数字在后半。

  按条切就不会有这个问题：一条 = 一个 chunk。

带章标题的理由：
  「第七章 转专业与转学」本身就告诉检索器这一章讲什么，
  是免费增加的语义线索（不用额外调用模型）。
"""

import re

CN = "一二三四五六七八九十百零"
CHAPTER_RE = re.compile(rf"^第([{CN}\d]+)章\s*(.*)$")
ARTICLE_RE = re.compile(rf"^第([{CN}\d]+)条")

MIN_ARTICLE_CHARS = 40      # 太短的"条"（比如只有标题）不算
MIN_ARTICLES_FOR_ARTICLE_MODE = 3   # 少于 3 条就认为不是条文式文档


def split_by_article(text: str, ctx: int = 100) -> list[dict]:
    """按「第X章 / 第X条」切分，每条带上所属章标题和前后文。

    为什么加前后文（ctx）：
      实测过"一条一块"的纯条文切分，召回反而从 96% 掉到 90%。
      原因是单条太"纯"——问「辅修的绩点要求」时，单条里只有"绩点3.0"，
      而相邻条文里才有"辅修报名条件"这种更贴近问法的词。
      加上前一条末尾 / 后一条开头各 100 字，就同时保住精度和上下文。

    返回 [{"text": ..., "chapter": ..., "article": ...}, ...]
    """
    raw = _split_articles_raw(text)
    if not raw:
        return []

    out = []
    for i, item in enumerate(raw):
        prev_tail = raw[i - 1]["text"][-ctx:] if i > 0 else ""
        next_head = raw[i + 1]["text"][:ctx] if i + 1 < len(raw) else ""
        body = "\n".join(p for p in (prev_tail, item["text"], next_head) if p)
        out.append({**item, "text": body})
    return out


def _split_articles_raw(text: str) -> list[dict]:
    """纯条文切分（不带前后文），内部用。"""
    lines = [ln.strip() for ln in text.split("\n")]
    chunks: list[dict] = []
    chapter = ""
    buf: list[str] = []
    art_no = ""

    def flush():
        nonlocal buf, art_no
        if buf and art_no:
            body = "\n".join(buf).strip()
            if len(body) >= MIN_ARTICLE_CHARS:
                chunks.append({
                    "text": f"[{chapter}] {body}" if chapter else body,
                    "chapter": chapter,
                    "article": art_no,
                })
        buf = []
        art_no = ""

    i = 0
    while i < len(lines):
        line = lines[i]

        m_ch = CHAPTER_RE.match(line)
        if m_ch:
            flush()
            title = m_ch.group(2).strip().replace("\u3000", "")
            # 章标题常常在下一行：「第七章」换行「转专业与转学」
            if not title and i + 1 < len(lines):
                nxt = lines[i + 1]
                if nxt and not ARTICLE_RE.match(nxt) and len(nxt) <= 24:
                    title = nxt.replace("\u3000", "").strip()
                    i += 1
            chapter = f"第{m_ch.group(1)}章 {title}".strip()
            i += 1
            continue

        m_art = ARTICLE_RE.match(line)
        if m_art:
            flush()
            art_no = f"第{m_art.group(1)}条"
            buf = [line]
            i += 1
            continue

        if buf:
            buf.append(line)
        i += 1

    flush()
    return chunks


def fixed_window(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    """固定窗口切分（v0 的做法，留作非条文式文档的兜底）。"""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    out = []
    step = size - overlap
    for i in range(0, len(text), step):
        piece = text[i:i + size].strip()
        if len(piece) >= 100:
            out.append(piece)
    return out


def split(text: str, size: int = 500, overlap: int = 50, mode: str = "window"):
    """统一入口。返回 (chunks, 用的哪种策略)。

    ⚠️ 实测结论（2026-11-03，497 chunk 索引，同一套评测集）：

        切分方式              chunk 平均长   书面 50 条   口语 19 条
        固定窗口（默认）        441 字        96.00%       89.47%
        纯条文（一条一块）      125 字        90.00%       84.21%
        条文 + 前后各 100 字    314 字        88.00%       73.68%

    **固定窗口最好，条文切分越做越差。** 两个原因：
      1. 单条太短（125 字），语义不完整 —— 问题往往跨条，单条匹配不上
      2. 加前后文补上下文，却制造了大面积重叠 —— 同一段话出现在 3 个 chunk 里，
         检索时好几个都"看起来对"，排名被打乱，反而挤掉真答案

    所以默认走固定窗口。article 模式保留着，作为一种"试过且失败"的记录。
    """
    if mode == "article":
        arts = split_by_article(text)
        if len(arts) >= MIN_ARTICLES_FOR_ARTICLE_MODE:
            return [
                {"text": a["text"], "kind": "article",
                 "chapter": a["chapter"], "article": a["article"]}
                for a in arts
            ], "article"

    windows = fixed_window(text, size, overlap)
    return [
        {"text": w, "kind": "window", "chapter": "", "article": ""}
        for w in windows
    ], "window"
