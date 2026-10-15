"""中文文本处理：分词、条款切分、文本规范化。

两件事在这里做，都是确定性纯函数，可单测：
1. tokenize —— 面向中文规程文本的分词（BM25 与向量检索共用同一套词表）
2. split_clauses —— 按「章-节-条」语义切分，而不是按固定字符数切
   （方案 v2 第 4 章 M2：切分是否合理是评分点，切错了引用就没有意义）
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

import jieba

# ---------------------------------------------------------------- 规范化

_WS_RE = re.compile(r"[ \t\u3000]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")
_PAGE_NOISE_RE = re.compile(r"^\s*(?:-\s*)?\d{1,4}\s*(?:-\s*)?$")  # 孤立的页码行
_HEADER_FOOTER_RE = re.compile(r"^\s*(?:第\s*\d+\s*页|共\s*\d+\s*页)\s*$")

# Markdown 标题前缀（本项目知识文档以 md 交付，必须先剥掉井号再判结构）
_MD_HEADING_RE = re.compile(r"^(?P<hashes>#{1,6})\s*(?P<content>.*)$")
_BLOCKQUOTE_RE = re.compile(r"^>\s?(?P<content>.*)$")

# 章节标题：第X章 / 第X节 / 附录X
_CN_CHAPTER_RE = re.compile(r"^第\s*(?P<cn>[一二三四五六七八九十百零〇\d]+)\s*章\s*(?P<title>.*)$")
_CN_SECTION_RE = re.compile(r"^第\s*(?P<cn>[一二三四五六七八九十百零〇\d]+)\s*节\s*(?P<title>.*)$")
_APPENDIX_RE = re.compile(r"^附\s*录\s*(?P<tag>[A-Za-zＡ-Ｚ一二三四五六七八九十\d]+)\s*(?P<title>.*)$")

# 条款起始行：1.1.1 正文 / 第五条 正文 / （1）正文
_NUM_CLAUSE_RE = re.compile(r"^(?P<no>\d+(?:\.\d+){0,3})\s*(?P<body>\S.*)$")
_CN_ARTICLE_LINE_RE = re.compile(r"^第\s*(?P<cn>[一二三四五六七八九十百零〇\d]+)\s*条\s*(?P<body>.+)$")
_PAREN_CLAUSE_RE = re.compile(r"^[（(]\s*(?P<no>[0-9一二三四五六七八九十]+)\s*[）)]\s*(?P<body>\S.*)$")

# 标题行不允许以句末标点结尾，且长度有限 —— 这是区分「小节标题」和「条款正文」的关键
_TITLE_MAX_CHARS = 30
_TITLE_FORBIDDEN_TAIL = ("。", "；", "，", "：", "！", "？", ".", ";", ",", ":", "!", "?")

# 合格条款号：2.1.1 / 5 / 第五条。切分时为无编号内容生成的「段落N」不算条款号
_NUMBERED_CLAUSE_RE = re.compile(r"^(?:\d+(?:\.\d+){0,3}|第[一二三四五六七八九十百零〇\d]+条)$")


def is_numbered_clause(clause_no: str) -> bool:
    """判断一个 clause_no 是不是真正的条款编号。

    「段落12」「（无编号正文）」这类是切分器给无编号内容起的名字，不是条款号。
    开启 require_clause_no 时它们不得作为处置动作的依据 —— 否则文档开头的
    免责声明也能被当成条款引用出来。
    """
    return bool(_NUMBERED_CLAUSE_RE.match((clause_no or "").strip()))


# 范围排除性表述：文档自己声明「这件事不在我管」。这类条款命中了也不构成
# 处置依据 —— 恰恰相反，它说明该问题超出知识库覆盖范围。
# 例：「本导则不适用于带电作业现场操作、直升机巡检与红外测温定量判读」。
_EXCLUSION_RE = re.compile(
    r"(不适用于|不适用|不涵盖|不包含|不包括|不涉及|不替代|不予适用|超出本(?:导则|指引|预案|文档))"
)


def is_exclusion_text(text: str) -> bool:
    """判断一段条款正文是不是「范围排除」声明。

    为什么需要它：检索是按词面相似度工作的，「带电作业」「红外测温」这类词
    本来就出现在排除条款里，于是最相似的那条恰好是「本导则不适用于……」。
    如果把它当成有依据，系统就会一本正经地为知识库明确排除的问题编出处置建议。
    """
    return bool(_EXCLUSION_RE.search(text or ""))


# 「声明自己管什么 / 解释名词」的条款：适用范围、编制目的、术语和定义。
# 它们同样是元信息，不是可执行要求 —— 把「本指引适用于……的巡检与缺陷判定」
# 当成处置依据引用出来，等于用一句自我介绍回答用户的问题。
_SCOPE_STATEMENT_RE = re.compile(
    r"(本(?:导则|指引|预案|文档|文件|规范)(?:适用|规定|用于|所称)|适用范围|术语和定义|编制目的)"
)


def is_scope_statement(text: str) -> bool:
    """判断一段条款正文是不是「适用范围/术语/目的」这类元信息条款。

    与 is_exclusion_text 的区别：排除声明（不适用于……）是**拒答信号**，必须
    保留在候选里用于判定超出范围；而适用范围/术语定义只是自我介绍，不该被
    当成处置依据排在候选前面。
    """
    return bool(_SCOPE_STATEMENT_RE.search(text or ""))


def normalize_text(raw: str) -> str:
    """统一换行、去零宽字符、压掉多余空白，但保留换行结构（切分依赖它）。"""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u200b", "").replace("\ufeff", "").replace("\xa0", " ")
    text = _WS_RE.sub(" ", text)
    text = _MULTI_NL_RE.sub("\n\n", text)
    return text.strip()


def clean_lines(text: str) -> list[str]:
    """去掉页码与页眉页脚噪声，返回有效行。"""
    out: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            out.append("")
            continue
        if _PAGE_NOISE_RE.match(stripped) or _HEADER_FOOTER_RE.match(stripped):
            continue
        out.append(stripped)
    return out


# ---------------------------------------------------------------- 分词

_STOPWORDS = {
    "的", "了", "和", "与", "及", "或", "在", "是", "为", "对", "按", "应", "宜", "可", "并",
    "本", "该", "其", "以", "等", "由", "于", "中", "上", "下", "内", "外", "时", "后", "前",
    "一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "不", "有", "无", "如", "则",
}
_TOKEN_KEEP_RE = re.compile(r"^[0-9a-zA-Z][0-9a-zA-Z.\-_/]*$")
_CN_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """分词：中文用 jieba 搜索引擎模式，编号/英文/单位整体保留。

    编号（如 5.2.3）必须整体保留 —— 巡检问答里「5.2.3 条怎么规定的」
    是最常见的提问方式，切碎了就检索不到。
    """
    tokens: list[str] = []
    for raw in jieba.cut_for_search(text):
        tok = raw.strip().lower()
        if not tok:
            continue
        if _TOKEN_KEEP_RE.match(tok):
            tokens.append(tok)
            continue
        if _CN_CHAR_RE.search(tok):
            if len(tok) == 1:
                continue
            if tok in _STOPWORDS:
                continue
            tokens.append(tok)
    return tokens


def char_ngrams(text: str, n_min: int = 2, n_max: int = 3) -> list[str]:
    """中文字符 n-gram，作为向量检索的兜底特征（分词切错时仍有召回）。"""
    grams: list[str] = []
    for seg in re.findall(r"[\u4e00-\u9fff]+", text):
        for n in range(n_min, n_max + 1):
            if len(seg) < n:
                continue
            grams.extend(seg[i : i + n] for i in range(len(seg) - n + 1))
    return grams


# ---------------------------------------------------------------- 条款切分


@dataclass
class RawClause:
    clause_no: str
    hierarchy_path: str
    text: str
    page: int | None = None
    level: int = 0


@dataclass
class _Section:
    no: str
    title: str
    level: int


def _is_title_like(body: str) -> bool:
    """判断一段文字是「小节标题」还是「条款正文」。

    判据只有两条，都可解释、可单测：长度不超过 30 字，且不以句末标点结尾。
    这条判据直接决定切分质量：判错会把整节内容并成一坨，或把正文当标题丢掉。
    """
    b = body.strip()
    if not b:
        return True
    if len(b) > _TITLE_MAX_CHARS:
        return False
    return not b.endswith(_TITLE_FORBIDDEN_TAIL)


def _strip_markup(line: str) -> tuple[str, int]:
    """返回 (内容, markdown 标题层级)，非标题层级为 0。"""
    m = _MD_HEADING_RE.match(line)
    if m:
        return m.group("content").strip(), len(m.group("hashes"))
    bq = _BLOCKQUOTE_RE.match(line)
    if bq:
        return bq.group("content").strip(), 0
    return line.strip(), 0


def _detect_section(content: str, md_level: int) -> _Section | None:
    """识别章节标题（章 / 节 / 附录 / 数字小节标题）。"""
    m = _CN_CHAPTER_RE.match(content)
    if m:
        return _Section(no=f"第{m.group('cn')}章", title=m.group("title").strip(), level=1)
    m = _CN_SECTION_RE.match(content)
    if m:
        return _Section(no=f"第{m.group('cn')}节", title=m.group("title").strip(), level=2)
    m = _APPENDIX_RE.match(content)
    if m:
        return _Section(no=f"附录{m.group('tag')}", title=m.group("title").strip(), level=1)

    m = _NUM_CLAUSE_RE.match(content)
    if m and _is_title_like(m.group("body")):
        no = m.group("no")
        return _Section(no=no, title=(m.group("body") or "").strip(), level=min(no.count(".") + 1, 3))
    return None


def _detect_clause_start(content: str) -> tuple[str, str] | None:
    """识别条款起始行，返回 (条款号, 该行正文)。"""
    m = _CN_ARTICLE_LINE_RE.match(content)
    if m:
        return f"第{m.group('cn')}条", m.group("body").strip()

    m = _PAREN_CLAUSE_RE.match(content)
    if m:
        return f"（{m.group('no')}）", m.group("body").strip()

    m = _NUM_CLAUSE_RE.match(content)
    if m and not _is_title_like(m.group("body")):
        return m.group("no"), m.group("body").strip()
    return None


def split_clauses(
    text: str,
    *,
    page_map: dict[int, tuple[int, int]] | None = None,
    min_chars: int = 12,
    max_chars: int = 1200,
) -> list[RawClause]:
    """把一篇规程切分成条款列表。

    切分规则（对应评分点「文档切分是否合理」）：
      章 / 节 / 附录 / 数字小节标题 → 维护层级栈，构成条款的 hierarchy_path
      1.1.1 正文 / 第五条 正文 / （1）正文 → 一条独立条款
      其余行 → 追加到当前条款正文（条款正文跨行时不会被拆散）
    无编号的段落退化为「段落N」，仍然可检索、可引用，不会被丢弃。

    page_map：{行号: (页码, 页内行号)}，用于给条款标页码（PDF 场景）。
    """
    lines = clean_lines(normalize_text(text))
    clauses: list[RawClause] = []
    stack: list[_Section] = []
    buf: list[str] = []
    buf_page: int | None = None
    current_no: str | None = None
    para_index = 0

    def section_path() -> str:
        return " › ".join(s.no + (f" {s.title}" if s.title else "") for s in stack)

    def emit(body: str, clause_no: str, page: int | None) -> None:
        clauses.append(
            RawClause(
                clause_no=clause_no,
                hierarchy_path=section_path() or "（无编号正文）",
                text=body,
                page=page,
                level=len(stack) + 1,
            )
        )

    def flush() -> None:
        nonlocal buf, buf_page, current_no, para_index
        body = "\n".join(buf).strip()
        page = buf_page
        buf, buf_page = [], None
        no = current_no
        current_no = None
        if len(body) < min_chars:
            return
        if no is None:
            para_index += 1
            emit(body, f"段落{para_index}", page)
        else:
            emit(body, no, page)

    for idx, line in enumerate(lines):
        if page_map and idx in page_map:
            buf_page = page_map[idx][0]
        if not line:
            if buf:
                buf.append("")
            continue

        content, md_level = _strip_markup(line)

        section = _detect_section(content, md_level)
        if section is not None:
            flush()
            if md_level and not stack:
                section.level = min(md_level, 3)
            while stack and stack[-1].level >= section.level:
                stack.pop()
            stack.append(section)
            continue

        clause_start = _detect_clause_start(content)
        if clause_start is not None:
            flush()
            current_no, first_line = clause_start
            buf = [first_line]
            continue

        buf.append(content)
        joined = "\n".join(buf)
        if len(joined) > max_chars:
            # 超长条款按长度硬切，编号加「（续）」后缀，引用时仍然指得回原条款
            head, tail = joined[:max_chars], joined[max_chars:]
            emit(head, (current_no or f"段落{para_index + 1}") + "（续）", buf_page)
            buf = [tail]
            current_no = current_no or f"段落{para_index + 1}"

    flush()
    return clauses


def dedupe_clauses(clauses: Iterable[RawClause]) -> list[RawClause]:
    """去掉重复条款（同一编号同一正文只留一条），保留首次出现顺序。"""
    seen: set[tuple[str, str]] = set()
    out: list[RawClause] = []
    for c in clauses:
        key = (c.clause_no, c.text[:60])
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


@dataclass
class SplitStats:
    total: int = 0
    numbered: int = 0
    fallback: int = 0
    docs: dict[str, int] = field(default_factory=dict)

    @property
    def numbered_ratio(self) -> float:
        return (self.numbered / self.total) if self.total else 0.0
