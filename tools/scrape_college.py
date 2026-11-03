"""抓各学院/部门网站上转发的校级制度（教务处没挂的那些）。

背景：扫「引用空洞」时发现 12 份被引用但缺失的文件，其中：
  《本科生课程重修管理办法（试行）》 → 地质与环境学院转发
  《学生管理规定》                 → 学工部转发
  《学术不端行为查处细则》          → 艺术学院转发

这些网站用的是博达 CMS，正文在 <div class="v_news_content"> 一类容器里。
抓下来直接存成 txt，和教务处抓的 PDF 一起进索引。

用法：
  python scrape_college.py            # 抓内置清单
  python scrape_college.py <url>      # 抓指定页面
"""

import re
import sys
import time
import urllib.request
from pathlib import Path

OUT = Path(r"D:\桌面\教务RAG\data\processed")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120 Safari/537.36")

# 内置清单：从「引用空洞」分析里得来的
TARGETS = [
    ("本科生课程重修管理办法（试行）",
     "https://dhxy.xust.edu.cn/info/1227/11194.htm"),
    ("西安科技大学学生管理规定",
     "https://xgb.xust.edu.cn/info/1053/6502.htm"),
    ("西安科技大学学术不端行为查处细则",
     "https://ysxy.xust.edu.cn/info/1039/2188.htm"),
]


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=40) as r:
        raw = r.read()
    # 这些站点是 UTF-8（前面看到 BOM）
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace")


def extract_body(html: str) -> str:
    """抽正文：优先找 CMS 的正文容器，否则退回"去标签取最长段"。"""
    # 博达 CMS 常见容器
    for pat in (
        r'<div[^>]+class="[^"]*v_news_content[^"]*"[^>]*>(.*?)</div>\s*(?:</div>|<div)',
        r'<div[^>]+id="vsb_content[^"]*"[^>]*>(.*?)</div>',
        r'<div[^>]+class="[^"]*content[^"]*"[^>]*>(.*?)</div>\s*</div>',
    ):
        m = re.search(pat, html, re.S | re.I)
        if m:
            html = m.group(1)
            break

    # 去脚本/样式/注释
    html = re.sub(r"<script.*?</script>", "", html, flags=re.S | re.I)
    html = re.sub(r"<style.*?</style>", "", html, flags=re.S | re.I)
    html = re.sub(r"<!--.*?-->", "", html, flags=re.S)
    # 块级标签转换行
    html = re.sub(r"</(p|div|tr|h\d|li)>", "\n", html, flags=re.I)
    html = re.sub(r"<br\s*/?>", "\n", html, flags=re.I)
    # 去剩余标签
    text = re.sub(r"<[^>]+>", "", html)
    # HTML 实体
    for a, b in (("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"),
                 ("&gt;", ">"), ("&quot;", '"'), ("&#39;", "'")):
        text = text.replace(a, b)
    text = re.sub(r"[ \t\u3000]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main():
    targets = TARGETS
    if len(sys.argv) > 1:
        url = sys.argv[1]
        targets = [(url.rstrip("/").rsplit("/", 1)[-1].replace(".htm", ""), url)]

    for name, url in targets:
        print(f"=== {name}")
        print(f"    {url}")
        try:
            html = fetch(url)
            body = extract_body(html)
            if len(body) < 300:
                print(f"    ⚠️ 正文太短（{len(body)} 字），可能没抓对区域")
                continue
            # 文件名：加前缀标明来源，避免和教务处文件重名
            safe = re.sub(r'[\\/:*?"<>|]', "_", name)
            out = OUT / f"学院转发_{safe}.txt"
            out.write_text(body, encoding="utf-8")
            print(f"    ✅ {len(body):,} 字 → {out.name}")
            print(f"    开头：{body[:70].replace(chr(10), ' ')}")
            time.sleep(0.5)
        except Exception as e:  # noqa: BLE001
            print(f"    ❌ {type(e).__name__}: {str(e)[:80]}")
        print()


if __name__ == "__main__":
    main()
