#!/usr/bin/env python3
"""Поиск по статьям закона: BM25 с приведением слов к начальной форме.

Почему не «эмбеддинги и косинус»: в праве запрос почти всегда содержит те же
термины, что и норма («наблюдение», «конкурсный управляющий», «мораторий»),
а точность важнее пересказа. Лексический поиск здесь и быстрее, и проверяем:
видно, по каким словам нашлось. Морфология обязательна — без неё «должника»
и «должник» считаются разными словами.
"""
import functools, json, re
from pathlib import Path

import pymorphy3
from rank_bm25 import BM25Okapi

MORPH = pymorphy3.MorphAnalyzer()
WORD = re.compile(r"[а-яёa-z0-9]+", re.I)
STOP = {"и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то", "все",
        "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за", "бы", "по",
        "только", "ее", "мне", "было", "вот", "от", "меня", "еще", "нет", "о", "из",
        "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли", "если", "или", "быть",
        "для", "при", "этом", "этого", "который", "которые", "которых"}


@functools.lru_cache(maxsize=200_000)
def norm(word):
    return MORPH.parse(word)[0].normal_form


def tokens(text):
    return [norm(w) for w in WORD.findall(text.lower()) if w not in STOP and len(w) > 2]


class Index:
    def __init__(self, path="corpus.json"):
        self.articles = json.loads(Path(path).read_text(encoding="utf-8"))
        # Заголовок весит больше: он называет предмет статьи прямо.
        docs = [tokens(a["title"]) * 3 + tokens(a["text"]) for a in self.articles]
        self.bm25 = BM25Okapi(docs)

    def search(self, query, k=5):
        scores = self.bm25.get_scores(tokens(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self.articles[i], float(scores[i])) for i in order]


if __name__ == "__main__":
    import sys
    idx = Index()
    q = " ".join(sys.argv[1:]) or "что такое наблюдение в деле о банкротстве"
    print(f"Запрос: {q}\n")
    for art, score in idx.search(q):
        print(f"  ст. {art['article']} ({score:.1f}) — {art['title'][:80]}")
