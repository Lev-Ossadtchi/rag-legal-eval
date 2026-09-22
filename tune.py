#!/usr/bin/env python3
"""Подбор настроек поиска: меряем каждую конфигурацию на одном наборе вопросов.

Смысл не в «покрутить параметры», а в том, чтобы показать цифрами, что именно
влияет на качество в юридическом поиске:
* вес заголовка — заголовок статьи прямо называет её предмет;
* b в BM25 — насколько сильно штрафуем длинные статьи (в законе они длинные
  и «богаты» общей лексикой, поэтому без штрафа они забивают короткие точные);
* отдельный балл за совпадение заголовка целиком.

  python3 tune.py
"""
import json, re
from pathlib import Path

from rank_bm25 import BM25Okapi

from search import tokens
from evaluate import run, report


class Tuned:
    def __init__(self, articles, title_weight=3, k1=1.5, b=0.75, title_bonus=0.0):
        self.articles = articles
        self.title_bonus = title_bonus
        self.title_tokens = [set(tokens(a["title"])) for a in articles]
        docs = [tokens(a["title"]) * title_weight + tokens(a["text"]) for a in articles]
        self.bm25 = BM25Okapi(docs, k1=k1, b=b)

    def search(self, query, k=5):
        qt = tokens(query)
        scores = self.bm25.get_scores(qt)
        if self.title_bonus:
            qs = set(qt)
            for i, tt in enumerate(self.title_tokens):
                if tt:
                    scores[i] += self.title_bonus * len(qs & tt) / len(tt)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self.articles[i], float(scores[i])) for i in order]


CONFIGS = [
    ("базовая: вес заголовка 3, b=0.75", dict(title_weight=3, b=0.75)),
    ("вес заголовка 6", dict(title_weight=6, b=0.75)),
    ("вес заголовка 10", dict(title_weight=10, b=0.75)),
    ("b=0.4 (слабее штраф за длину)", dict(title_weight=6, b=0.4)),
    ("b=0.9 (сильнее штраф за длину)", dict(title_weight=6, b=0.9)),
    ("вес 6 + бонус за совпадение заголовка", dict(title_weight=6, b=0.75, title_bonus=4.0)),
    ("вес 10 + бонус 6", dict(title_weight=10, b=0.75, title_bonus=6.0)),
]

if __name__ == "__main__":
    arts = json.loads(Path("corpus.json").read_text(encoding="utf-8"))
    qs = json.loads(Path("questions.json").read_text(encoding="utf-8"))
    rows = []
    for name, kw in CONFIGS:
        idx = Tuned(arts, **kw)
        ranks = run(idx, qs)
        n = len(ranks)
        r1 = sum(1 for _, p, _ in ranks if p == 1)
        r3 = sum(1 for _, p, _ in ranks if p and p <= 3)
        mrr = sum(1 / p for _, p, _ in ranks if p) / n
        rows.append((name, r1, r3, mrr, kw))
        print(f"{name:45s} R@1 {r1:2d}/{n}  R@3 {r3:2d}/{n}  MRR {mrr:.2f}")
    best = max(rows, key=lambda r: (r[3], r[1]))
    print(f"\nлучшая: {best[0]} → {best[4]}")
    Path("best_config.json").write_text(json.dumps(best[4], ensure_ascii=False), encoding="utf-8")
