"""临时诊断：术语表是否含重复 canonical、expand 是否重复追加。——用完即可删。"""

from __future__ import annotations

from collections import Counter

from loweye.config import load_config
from loweye.retrieval.lexicon import build_lexicon

cfg = load_config()
lex = build_lexicon(cfg)
print("entries", len(lex.entries), "aliases", lex.alias_count)
c = Counter(e.canonical for e in lex.entries)
print("dup canonicals:", dict((k, v) for k, v in c.items() if v > 1))
print("canonicals_in:", lex.canonicals_in("瓷瓶碎了一片"))
print("expand:", lex.expand("瓷瓶碎了一片"))
print("describe:", lex.describe("瓷瓶碎了一片"))
