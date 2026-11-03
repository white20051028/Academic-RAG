"""抓取西安科技大学教务处的制度类附件。

来源：教务处 4 个栏目（管理制度 / 学籍与学位 / 辅修专业 / 微专业）
排除：文件名含「公示」「名单」的 —— 那些含学生真实姓名学号，不能进数据集
输出：
  data/raw/          原始文件（保持原扩展名）
  data/manifest.csv  清单：栏目 / 文章ID / 附件名 / 本地文件名 / 状态

礼貌爬取：每个请求间隔 0.4 秒，带正常 UA。
"""

import csv
import os
import re
import time
import urllib.parse
import urllib.request

BASE = "https://jwc.xust.edu.cn"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120 Safari/537.36")

COLUMNS = {
    "管理制度": 1024,
    "学籍与学位": 1004,
    "辅修专业": 1085,
    "微专业": 1086,
    "实践教学": 1006,
}

PRIVACY_WORDS = ("公示", "名单")
ROOT = r"D:\桌面\教务RAG"
OUT_RAW = os.path.join(ROOT, "data", "raw")
OUT_MANIFEST = os.path.join(ROOT, "data", "manifest.csv")
DELAY = 0.4


def fetch_bytes(url: str, timeout: int = 40) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_text(url: str) -> str:
    raw = fetch_bytes(url)
    for enc in ("utf-8", "gbk", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace")


def article_links(col_id: int) -> list[str]:
    html = fetch_text(f"{BASE}/News_list.jsp?urltype=tree.TreeTempUrl&wbtreeid={col_id}")
    seen, out = set(), []
    for l in re.findall(r'href="(/?info/\d+/\d+\.htm)"', html):
        l = l.lstrip("/")
        if l not in seen:
            seen.add(l)
            out.append(f"{BASE}/{l}")
    return out


def attachments(article_url: str) -> list[tuple[str, str]]:
    try:
        html = fetch_text(article_url)
    except Exception as e:  # noqa: BLE001
        print(f"      ⚠️ 抓取失败 {type(e).__name__}")
        return []

    found = []
    for m in re.finditer(
        r'href="([^"]*download\.jsp[^"]*)"[^>]*>([^<]{0,150})<', html
    ):
        href, name = m.group(1), m.group(2).strip()
        if not href.startswith("http"):
            href = BASE + "/" + href.lstrip("/")
        found.append((name or "(无名附件)", href))
    return found


def safe_name(name: str, fallback: str) -> str:
    """把附件名变成安全文件名，保留扩展名。"""
    name = name.strip().replace("\u3000", " ")
    name = re.sub(r'[\\/:*?"<>|\r\n\t]', "_", name)
    name = re.sub(r"\s+", " ", name).strip(" .")
    if not name:
        name = fallback
    # 太长的截断，但保住扩展名
    if len(name) > 90:
        stem, dot, ext = name.rpartition(".")
        if dot and len(ext) <= 5:
            name = stem[: 90 - len(ext) - 1] + "." + ext
        else:
            name = name[:90]
    return name


def main():
    os.makedirs(OUT_RAW, exist_ok=True)
    rows = []
    seen_urls = set()
    ok = skipped = failed = 0

    for col, cid in COLUMNS.items():
        print("=" * 66)
        print(f"【{col}】")
        try:
            arts = article_links(cid)
        except Exception as e:  # noqa: BLE001
            print(f"  ❌ 列表页失败: {type(e).__name__}")
            continue
        print(f"  文章 {len(arts)} 篇")
        time.sleep(DELAY)

        for i, art in enumerate(arts, 1):
            art_id = art.rsplit("/", 1)[-1].replace(".htm", "")
            atts = attachments(art)
            for name, url in atts:
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                if any(w in name for w in PRIVACY_WORDS):
                    print(f"    [{i}/{len(arts)}] 跳过（隐私）: {name[:50]}")
                    rows.append({
                        "栏目": col, "文章ID": art_id, "附件名": name,
                        "本地文件名": "", "状态": "跳过-隐私",
                    })
                    skipped += 1
                    continue

                fname = f"{art_id}_{safe_name(name, 'attachment')}"
                dest = os.path.join(OUT_RAW, fname)

                # 已经下过就跳过 —— 重跑时只补新增的
                if os.path.exists(dest) and os.path.getsize(dest) > 0:
                    rows.append({
                        "栏目": col, "文章ID": art_id, "附件名": name,
                        "本地文件名": fname, "状态": "跳过-已存在",
                    })
                    skipped += 1
                    continue

                try:
                    data = fetch_bytes(url)
                    # 有些 download.jsp 返回的是 HTML 错误页
                    if data[:5].lstrip().lower().startswith(b"<html") or b"<html" in data[:200].lower():
                        print(f"    [{i}/{len(arts)}] 跳过（返回 HTML）: {name[:40]}")
                        rows.append({
                            "栏目": col, "文章ID": art_id, "附件名": name,
                            "本地文件名": "", "状态": "跳过-非文件",
                        })
                        skipped += 1
                        continue
                    with open(dest, "wb") as f:
                        f.write(data)
                    print(f"    [{i}/{len(arts)}] ✅ {fname[:60]}  ({len(data):,} 字节)")
                    rows.append({
                        "栏目": col, "文章ID": art_id, "附件名": name,
                        "本地文件名": fname, "状态": f"ok-{len(data)}",
                    })
                    ok += 1
                except Exception as e:  # noqa: BLE001
                    print(f"    [{i}/{len(arts)}] ❌ {name[:40]}  {type(e).__name__}")
                    rows.append({
                        "栏目": col, "文章ID": art_id, "附件名": name,
                        "本地文件名": "", "状态": f"失败-{type(e).__name__}",
                    })
                    failed += 1
                time.sleep(DELAY)

    with open(OUT_MANIFEST, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["栏目", "文章ID", "附件名", "本地文件名", "状态"])
        w.writeheader()
        w.writerows(rows)

    print()
    print("=" * 66)
    print(f"下载成功 {ok} 个 / 跳过 {skipped} 个 / 失败 {failed} 个")
    print(f"文件目录: {OUT_RAW}")
    print(f"清单文件: {OUT_MANIFEST}")


if __name__ == "__main__":
    main()
