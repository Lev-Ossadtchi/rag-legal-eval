#!/usr/bin/env python3
"""Гибридный поиск: лексический BM25 плюс смысловой по эмбеддингам.

Зачем гибрид, а не что-то одно — видно по замеру:
* BM25 находит, когда в вопросе есть термин из нормы («конкурсная масса»);
* эмбеддинги находят, когда человек спрашивает бытовыми словами
  («когда гражданина перестают заставлять платить долги» → ст. 213.28);
* по отдельности каждый промахивается на своей половине вопросов.

Модель локальная (multilingual-e5), работает офлайн — для юридических документов
это важно: текст никуда не уходит.

  python3 hybrid.py
"""
import json
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from search import tokens

MODEL = "intfloat/multilingual-e5-small"
CACHE = Path("embeddings.npy")


class Hybrid:
    def __init__(self, articles, title_weight=6, b=0.75, alpha=0.5):
        self.articles = articles
        self.alpha = alpha                     # вес смысловой части
        docs = [tokens(a["title"]) * title_weight + tokens(a["text"]) for a in articles]
        self.bm25 = BM25Okapi(docs, b=b)
        self.model = SentenceTransformer(MODEL)
        if CACHE.exists():
            self.emb = np.load(CACHE)
        else:
            # e5 требует префиксы: passage для документов, query для запросов
            texts = [f"passage: {a['title']}. {a['text'][:1200]}" for a in articles]
            self.emb = self.model.encode(texts, batch_size=32, normalize_embeddings=True,
                                         show_progress_bar=True)
            np.save(CACHE, self.emb)

    @staticmethod
    def _norm(x):
        lo, hi = float(np.min(x)), float(np.max(x))
        return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)

    def search(self, query, k=5):
        lex = self._norm(np.asarray(self.bm25.get_scores(tokens(query))))
        qv = self.model.encode([f"query: {query}"], normalize_embeddings=True)[0]
        sem = self._norm(self.emb @ qv)
        total = (1 - self.alpha) * lex + self.alpha * sem
        order = np.argsort(-total)[:k]
        return [(self.articles[i], float(total[i])) for i in order]


if __name__ == "__main__":
    from evaluate import run, report
    arts = json.loads(Path("corpus.json").read_text(encoding="utf-8"))
    qs = json.loads(Path("questions.json").read_text(encoding="utf-8"))
    for alpha in (0.0, 0.3, 0.5, 0.7, 1.0):
        idx = Hybrid(arts, alpha=alpha)
        ranks = run(idx, qs)
        n = len(ranks)
        r1 = sum(1 for _, p, _ in ranks if p == 1)
        r3 = sum(1 for _, p, _ in ranks if p and p <= 3)
        mrr = sum(1 / p for _, p, _ in ranks if p) / n
        label = {0.0: "только BM25", 1.0: "только смысл"}.get(alpha, f"гибрид α={alpha}")
        print(f"{label:16s}  R@1 {r1:2d}/{n} ({100*r1/n:3.0f}%)  R@3 {r3:2d}/{n} ({100*r3/n:3.0f}%)  MRR {mrr:.2f}")
