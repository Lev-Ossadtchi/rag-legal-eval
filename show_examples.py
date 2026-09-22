#!/usr/bin/env python3
"""Примеры выдачи для README: как система отвечает на живые вопросы."""
import json
from pathlib import Path
from hybrid import Hybrid

QUERIES = [
    "когда гражданина перестают заставлять платить долги",
    "кто платит вознаграждение арбитражному управляющему",
    "что попадает в конкурсную массу",
    "может ли директор распоряжаться имуществом после введения наблюдения",
    "сроки подачи требований кредиторов",
]

arts = json.loads(Path("corpus.json").read_text(encoding="utf-8"))
idx = Hybrid(arts, alpha=0.7)
out = []
for q in QUERIES:
    hits = idx.search(q, k=3)
    out.append((q, [(a["article"], a["title"], round(s, 2)) for a, s in hits]))

lines = []
for q, hits in out:
    lines.append(f"**«{q}»**\n")
    for art, title, score in hits:
        lines.append(f"- ст. {art} — {title} · {score}")
    lines.append("")
Path("examples.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
