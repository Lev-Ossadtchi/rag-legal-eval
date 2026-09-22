#!/usr/bin/env python3
"""Замер качества поиска: находит ли система нужную статью.

Метрики выбраны под задачу юриста:
* Recall@1 — ответ сразу правильный;
* Recall@3 — нужная статья в трёх первых (юрист пролистает три, не тридцать);
* Recall@5 и MRR — насколько глубоко приходится копать.
MRR = средняя величина 1/позиция: 1.0 значит «всегда первая», 0.5 — «в среднем вторая».

  python3 evaluate.py
"""
import json
from pathlib import Path

from search import Index


def run(index, questions, k=5):
    ranks = []
    for q in questions:
        hits = index.search(q["question"], k=k)
        pos = next((i + 1 for i, (a, _) in enumerate(hits) if a["article"] == q["article"]), None)
        ranks.append((q, pos, hits))
    return ranks


def report(ranks, k=5):
    n = len(ranks)
    r1 = sum(1 for _, p, _ in ranks if p == 1)
    r3 = sum(1 for _, p, _ in ranks if p and p <= 3)
    r5 = sum(1 for _, p, _ in ranks if p and p <= k)
    mrr = sum(1 / p for _, p, _ in ranks if p) / n
    print(f"вопросов: {n}")
    print(f"Recall@1: {r1}/{n} ({100*r1/n:.0f}%)   нужная статья сразу первой")
    print(f"Recall@3: {r3}/{n} ({100*r3/n:.0f}%)   в первых трёх")
    print(f"Recall@5: {r5}/{n} ({100*r5/n:.0f}%)   в первых пяти")
    print(f"MRR:      {mrr:.2f}")
    misses = [(q, p, h) for q, p, h in ranks if not p]
    if misses:
        print(f"\nне найдено вовсе — {len(misses)}:")
        for q, _, h in misses[:6]:
            got = ", ".join(f"ст.{a['article']}" for a, _ in h[:3])
            print(f"  «{q['question'][:70]}»")
            print(f"     ждали ст.{q['article']} ({q['title'][:45]}), выдало: {got}")
    return {"n": n, "r1": r1, "r3": r3, "r5": r5, "mrr": round(mrr, 3)}


if __name__ == "__main__":
    idx = Index()
    qs = json.loads(Path("questions.json").read_text(encoding="utf-8"))
    ranks = run(idx, qs)
    stats = report(ranks)
    Path("results.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
