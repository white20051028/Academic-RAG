"""把 data/raw 下各种格式的文档转成纯文本。

支持的格式和对应工具：
  .pdf   PyMuPDF（本地已确认 22 个全部有文字层）
  .docx  python-docx（含表格，培养方案里全是表格）
  .doc   Word COM —— **复用同一个 Word 实例**，每个都新建要 5 秒
  .xlsx  openpyxl（所有 sheet）
  .rar   7-Zip 解压后递归处理
  .zip   zipfile 解压后递归处理

输出：
  data/processed/<原名>.txt
  data/extract_manifest.csv   清单（源文件 / 输出 / 字数 / 状态）

去重：按全文 sha256。同一份文件在多个文章里出现时只留一份。
"""

import csv
import hashlib
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import pymupdf
from docx import Document

ROOT = Path(r"D:\桌面\教务RAG")
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
MANIFEST = ROOT / "data" / "extract_manifest.csv"
SEVENZIP = Path(r"C:\Program Files\7-Zip\7z.exe")


def from_pdf(p: Path) -> str:
    doc = pymupdf.open(p)
    try:
        return "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()


def from_docx(p: Path) -> str:
    d = Document(str(p))
    parts = [para.text for para in d.paragraphs]
    for table in d.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def from_xlsx(p: Path) -> str:
    import openpyxl

    wb = openpyxl.load_workbook(p, data_only=True, read_only=True)
    parts = []
    try:
        for ws in wb.worksheets:
            parts.append(f"[工作表] {ws.title}")
            for row in ws.iter_rows(values_only=True):
                cells = [str(c) for c in row if c is not None]
                if cells:
                    parts.append(" | ".join(cells))
    finally:
        wb.close()
    return "\n".join(parts)


class WordReader:
    """复用同一个 Word 实例读 .doc。"""

    def __init__(self):
        import win32com.client

        self.word = win32com.client.Dispatch("Word.Application")
        self.word.Visible = False
        self.word.DisplayAlerts = 0

    def read(self, p: Path) -> str:
        doc = self.word.Documents.Open(
            str(p), ReadOnly=True, AddToRecentFiles=False, Visible=False
        )
        try:
            return doc.Content.Text
        finally:
            doc.Close(False)

    def close(self):
        try:
            self.word.Quit()
        except Exception:  # noqa: BLE001
            pass


def unzip_rar(p: Path, dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(SEVENZIP), "x", "-y", f"-o{dest}", str(p)],
        capture_output=True,
        timeout=180,
    )
    return [f for f in dest.rglob("*") if f.is_file()]


def unzip_zip(p: Path, dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(p) as z:
        z.extractall(dest)
    return [f for f in dest.rglob("*") if f.is_file()]


HANDLERS = {".pdf": from_pdf, ".docx": from_docx, ".xlsx": from_xlsx}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    seen_hash: dict[str, str] = {}
    word: WordReader | None = None

    def process(p: Path, origin: str):
        nonlocal word
        ext = p.suffix.lower()
        try:
            if ext == ".doc":
                if word is None:
                    word = WordReader()
                text = word.read(p)
            elif ext in HANDLERS:
                text = HANDLERS[ext](p)
            else:
                return

            text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
            if len(text) < 50:
                rows.append({"源文件": origin, "输出": "", "字数": len(text), "状态": "跳过-内容过短"})
                return

            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
            if digest in seen_hash:
                rows.append({
                    "源文件": origin, "输出": "", "字数": len(text),
                    "状态": f"重复-同 {seen_hash[digest]}",
                })
                print(f"  ♻️ 重复            {Path(origin).name[:50]}")
                return
            seen_hash[digest] = Path(origin).name

            out = OUT / f"{Path(origin).stem}.txt"
            out.write_text(text, encoding="utf-8")
            rows.append({"源文件": origin, "输出": out.name, "字数": len(text), "状态": "ok"})
            print(f"  ✅ {len(text):>6,} 字  {Path(origin).name[:58]}")
        except Exception as e:  # noqa: BLE001
            rows.append({
                "源文件": origin, "输出": "", "字数": 0,
                "状态": f"失败-{type(e).__name__}",
            })
            print(f"  ❌ {Path(origin).name[:46]}  {type(e).__name__}")

    files = sorted(f for f in RAW.iterdir() if f.is_file())
    print(f"共 {len(files)} 个文件\n")

    for p in files:
        ext = p.suffix.lower()
        if ext in (".rar", ".zip"):
            print(f"📦 解压 {p.name[:52]}")
            tmp = Path(tempfile.mkdtemp(prefix="xust_"))
            try:
                inner = (unzip_rar if ext == ".rar" else unzip_zip)(p, tmp)
                print(f"   解出 {len(inner)} 个文件")
                for f in sorted(inner):
                    process(f, f"{p.stem}/{f.name}")
            except Exception as e:  # noqa: BLE001
                print(f"   ❌ 解压失败: {type(e).__name__}")
                rows.append({"源文件": p.name, "输出": "", "字数": 0, "状态": f"解压失败-{type(e).__name__}"})
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        else:
            process(p, p.name)

    if word:
        word.close()

    with open(MANIFEST, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["源文件", "输出", "字数", "状态"])
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r["状态"] == "ok"]
    dup = [r for r in rows if r["状态"].startswith("重复")]
    fail = [r for r in rows if r["状态"].startswith(("失败", "解压失败"))]
    short = [r for r in rows if r["状态"] == "跳过-内容过短"]
    total = sum(r["字数"] for r in ok)

    print()
    print("=" * 62)
    print(f"成功 {len(ok)}  重复 {len(dup)}  过短 {len(short)}  失败 {len(fail)}")
    print(f"总字数: {total:,}")
    print(f"输出目录: {OUT}")


if __name__ == "__main__":
    main()
