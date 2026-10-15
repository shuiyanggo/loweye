"""临时诊断：真实的排除条款（1.1.2）是否同时被两个判据命中。——用完即可删。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loweye.config import load_config  # noqa: E402
from loweye.retrieval.index import KnowledgeBase  # noqa: E402
from loweye.text import is_exclusion_text, is_scope_statement  # noqa: E402

cfg = load_config()
kb = KnowledgeBase.load(cfg.path("index_dir"), cfg)
for c in kb.clauses:
    if c.clause_no.endswith("1.1.1") or c.clause_no.endswith("1.1.2"):
        text = c.text.replace("\n", " ")
        print(
            f"{c.clause_no:<8} {c.doc_id:<22} excl={is_exclusion_text(c.text)!s:<5} "
            f"scope={is_scope_statement(c.text)!s:<5} {text[:64]}"
        )
