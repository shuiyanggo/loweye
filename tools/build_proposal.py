"""把 docs/proposal/*.md 汇编成一份可直接提交的 PDF 方案书。

链路：Markdown → HTML（markdown-it-py）→ 无头 Edge 打印成 PDF。
选浏览器打印而不是 PyMuPDF Story，是因为方案书里有二十多张表格，
Story 的 HTML/CSS 子集对表格支持有限，浏览器的排版结果才是所见即所得。

用法：
    python -m tools.build_proposal                 # 出 out/低空之眼-参赛方案书.pdf
    python -m tools.build_proposal --html-only     # 只出 HTML，便于先看排版
"""

from __future__ import annotations

import argparse
import html
import re
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DOCS = PROJECT / "docs" / "proposal"
EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]

# 章节顺序即提交顺序；(文件名, 章标题)
CHAPTERS = [
    ("01_overview.md", "第一章　项目概述"),
    ("02_market.md", "第二章　市场与竞品"),
    ("04_tech.md", "第三章　技术方案"),
    ("05_prototype.md", "第四章　原型与演示"),
    ("06_innovation.md", "第五章　创新点与对照实验"),
    ("07_business.md", "第六章　商业模式与投资回报"),
    ("08_team.md", "第七章　团队分工"),
    ("09_risk.md", "第八章　里程碑与风险"),
    ("10_compliance.md", "第九章　合规与伦理"),
]

# 03_* 若单独存在（TAM/SAM/SOM 独立成章）就插到市场章之后
OPTIONAL_AFTER = {
    "02_market.md": [("03_market_size.md", "第二章附　TAM / SAM / SOM 测算")],
    "06_innovation.md": [
        ("06b_bge.md", "第五章附一　换 BGE 后端后的复测（ML-03 里程碑结果）"),
        ("06c_detection.md", "第五章附二　检测环节：真实权重通道与逐类漏检率"),
        # 由 tools/write_model_report.py 在制品回收并独立评测后生成；文件不存在时
        # build_html 会跳过该章（第 141-142 行），因此这一行在报告生成前不影响任何产物。
        ("06d_trained_detectors.md", "第五章附三　扩展检测器：烟火与农作物害虫的训练与独立评测"),
    ],
}

CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
* { box-sizing: border-box; }
body { font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", sans-serif;
       font-size: 10.5pt; line-height: 1.65; color: #14181d; margin: 0; }
.cover { height: 247mm; display: flex; flex-direction: column; justify-content: center;
         page-break-after: always; text-align: left; }
.cover h1 { font-size: 30pt; margin: 0 0 6mm 0; letter-spacing: 1px; }
.cover .sub { font-size: 13pt; color: #1b4fd8; margin-bottom: 18mm; }
.cover dl { font-size: 11pt; line-height: 2; }
.cover dt { color: #5b6572; display: inline-block; width: 30mm; }
.cover dd { display: inline; margin: 0; }
.toc { page-break-after: always; }
.toc h2 { font-size: 16pt; border: 0; margin-bottom: 4mm; }
.toc ol { padding-left: 6mm; }
.toc li { margin: 2.2mm 0; font-size: 11pt; }
.toc .lv2 { color: #5b6572; font-size: 10pt; margin-left: 6mm; }
h1 { font-size: 19pt; margin: 0 0 5mm 0; padding-bottom: 2mm; border-bottom: 2px solid #1b4fd8; }
h2 { font-size: 13.5pt; color: #1b4fd8; margin: 7mm 0 2.5mm 0; }
h3 { font-size: 11.5pt; margin: 5mm 0 2mm 0; }
h4 { font-size: 10.5pt; color: #3d4550; margin: 4mm 0 1.5mm 0; }
p { margin: 0 0 2.4mm 0; text-align: justify; }
ul, ol { margin: 0 0 2.6mm 0; padding-left: 6mm; }
li { margin: 0.8mm 0; }
table { width: 100%; border-collapse: collapse; margin: 2.5mm 0 4mm 0; font-size: 9pt; }
th, td { border: 1px solid #cfd6de; padding: 1.6mm 2mm; text-align: left; vertical-align: top; }
th { background: #f2f5f9; font-weight: 600; }
tr { page-break-inside: avoid; }
code { font-family: Consolas, "Courier New", monospace; font-size: 9pt;
       background: #f2f5f9; padding: 0.3mm 1mm; border-radius: 1mm; }
pre { background: #f7f9fc; border: 1px solid #e3e8ef; border-radius: 1.5mm;
      padding: 3mm; overflow-wrap: anywhere; white-space: pre-wrap; font-size: 8.8pt; }
blockquote { margin: 2mm 0; padding: 1mm 0 1mm 4mm; border-left: 3px solid #cfd6de; color: #3d4550; }
.chapter { page-break-before: always; }
a { color: #1b4fd8; text-decoration: none; word-break: break-all; }
.footnote { color: #8a929c; font-size: 8.5pt; }
"""


def find_edge() -> str:
    for cand in EDGE_CANDIDATES:
        if Path(cand).exists():
            return cand
    raise SystemExit("没找到 msedge.exe（打印 PDF 需要它）")


def md_to_html(text: str) -> tuple[str, list[tuple[int, str]]]:
    """Markdown → HTML，并顺手取出标题清单用于生成目录。"""
    from markdown_it import MarkdownIt

    md = MarkdownIt("js-default", {"html": False, "linkify": False, "typographer": False})
    # 章文件通常自带一级标题，这里统一降级，避免与章标题重复
    body = md.render(text)

    headings: list[tuple[int, str]] = []
    for level, raw in re.findall(r"<h([1-4])>(.*?)</h\1>", body, flags=re.S):
        plain = html.unescape(re.sub(r"<[^>]+>", "", raw)).strip()
        headings.append((int(level), plain))
    return body, headings


def strip_first_h1(body: str) -> str:
    return re.sub(r"<h1>.*?</h1>", "", body, count=1, flags=re.S)


def build_html(*, title: str, subtitle: str, meta: dict[str, str]) -> tuple[str, list[str]]:
    parts: list[str] = []
    toc_entries: list[str] = []
    missing: list[str] = []

    for filename, chapter_title in CHAPTERS:
        path = DOCS / filename
        if not path.exists():
            missing.append(filename)
            continue
        body, headings = md_to_html(path.read_text(encoding="utf-8"))
        body = strip_first_h1(body)
        anchor = filename.replace(".md", "")
        parts.append(f'<section class="chapter" id="{anchor}"><h1>{html.escape(chapter_title)}</h1>{body}</section>')
        toc_entries.append(f"<li>{html.escape(chapter_title)}")
        subs = [h for h in headings if h[0] in (2, 3)][:12]
        if subs:
            toc_entries.append("<div>")
            for level, text in subs:
                css = "lv2" if level == 2 else "lv2"
                toc_entries.append(f'<div class="{css}">{html.escape(text)}</div>')
            toc_entries.append("</div>")
        toc_entries.append("</li>")

        for extra_file, extra_title in OPTIONAL_AFTER.get(filename, []):
            extra = DOCS / extra_file
            if not extra.exists():
                continue
            ebody, _ = md_to_html(extra.read_text(encoding="utf-8"))
            parts.append(
                f'<section class="chapter" id="{extra_file.replace(".md", "")}">'
                f"<h1>{html.escape(extra_title)}</h1>{strip_first_h1(ebody)}</section>"
            )
            toc_entries.append(f"<li>{html.escape(extra_title)}</li>")

    meta_html = "".join(
        f"<dt>{html.escape(k)}</dt><dd>{html.escape(v)}</dd>" for k, v in meta.items()
    )
    doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8"><title>{html.escape(title)}</title>
<style>{CSS}</style></head><body>
<section class="cover">
  <h1>{html.escape(title)}</h1>
  <div class="sub">{html.escape(subtitle)}</div>
  <dl>{meta_html}</dl>
</section>
<section class="toc"><h2>目录</h2><ol>{''.join(toc_entries)}</ol></section>
{''.join(parts)}
</body></html>"""
    return doc, missing


def resolve_pdf_out(value: str) -> Path:
    """把 --out 归一成绝对路径。

    Edge 的 --print-to-pdf 是**按它自己的进程工作目录**解析相对路径的，不是按调用者的
    工作目录。实测传 `out\\_rehearsal_proposal.pdf`（相对）时 Edge 报
    `Failed to write file out\\_rehearsal_proposal.pdf: 系统找不到指定的路径。 (0x3)`，
    构建以「打印 PDF 失败」退出 1；而默认值是绝对路径，所以这条只在显式传相对路径时暴露。
    """
    return Path(value).expanduser().resolve()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PROJECT / "out" / "低空之眼-参赛方案书.pdf"))
    ap.add_argument("--html-only", action="store_true")
    ap.add_argument("--title", default="低空之眼")
    ap.add_argument("--subtitle", default="低空巡检决策层软件 · 参赛方案书")
    args = ap.parse_args()

    meta = {
        "参赛赛道": "蓝桥杯数字科技创新赛（非命题赛）",
        "作品形态": "Web 软件（B/S，免构建静态前端 + Python 后端）",
        "文档说明": "正文数字均来自仓库内脚本实测或标注来源的公开数据；标注【假设】的取值可替换重算",
        "生成方式": "python -m tools.build_proposal（Markdown → HTML → 无头浏览器打印）",
    }
    doc, missing = build_html(title=args.title, subtitle=args.subtitle, meta=meta)

    html_path = PROJECT / "out" / "proposal.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(doc, encoding="utf-8")
    print(f"HTML 已写出：{html_path}（{len(doc)} 字符）")
    if missing:
        print(f"缺少章文件（这些章不会出现在 PDF 里）：{', '.join(missing)}")

    if args.html_only:
        return 0

    out = resolve_pdf_out(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    edge = find_edge()
    proc = subprocess.run(
        [
            edge, "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
            "--no-pdf-header-footer", "--print-to-pdf-no-header",
            f"--user-data-dir={PROJECT / 'out' / 'edge-print'}",
            f"--print-to-pdf={out}", html_path.as_uri(),
        ],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300,
    )
    if not out.exists():
        print(proc.stderr[-1500:], file=sys.stderr)
        raise SystemExit("打印 PDF 失败")

    try:
        import pymupdf

        with pymupdf.open(out) as pdf:
            pages = pdf.page_count
            text = pdf[0].get_text().strip().replace("\n", " ")[:80]
    except Exception as exc:  # 校验失败不算致命，但要显式说出来
        pages, text = -1, f"（校验失败：{type(exc).__name__}: {exc}）"
    size_mb = out.stat().st_size / 1024 / 1024
    print(f"PDF 已写出：{out}（{size_mb:.1f} MB，{pages} 页，首页开头：{text}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
