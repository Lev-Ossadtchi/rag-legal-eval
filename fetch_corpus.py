#!/usr/bin/env python3
"""Корпус: 127-ФЗ «О несостоятельности (банкротстве)» из Викитеки.

Почему так:
* Источник открытый (лицензия RusGov), ссылку можно дать заказчику — в отличие
  от текстов за платной стеной.
* Берём вики-разметку через action=raw: API extracts на этих страницах отдаёт
  пустой текст, потому что содержимое собрано шаблонами.
* Режем по статьям, а не по абзацам фиксированной длины: в праве ответ обязан
  ссылаться на статью, и статья же — естественная единица поиска.

  python3 fetch_corpus.py
"""
import json, re, time, urllib.error, urllib.parse, urllib.request
from pathlib import Path

BASE = "Федеральный закон от 26.10.2002 № 127-ФЗ"
API = "https://ru.wikisource.org/w/api.php"
UA = {"User-Agent": "rag-legal-eval/1.0 (research demo)"}


def get(url, tries=5):
    """Викитека ограничивает частоту — ждём и повторяем, а не давим запросами."""
    req = urllib.request.Request(url, headers=UA)
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == tries - 1:
                raise
            time.sleep(8 * (attempt + 1))


def chapter_titles():
    q = urllib.parse.urlencode({"action": "query", "generator": "prefixsearch",
                                "gpssearch": BASE, "gpslimit": 40, "format": "json"})
    data = json.loads(get(API + "?" + q))
    titles = [p["title"] for p in data.get("query", {}).get("pages", {}).values()]
    return sorted(t for t in titles if t.startswith(BASE + "/Глава"))


def raw(title):
    return get("https://ru.wikisource.org/wiki/"
               + urllib.parse.quote(title.replace(" ", "_")) + "?action=raw")


def clean(wiki):
    """Убираем разметку, оставляем читаемый текст статьи."""
    t = re.sub(r"\{\{якорь\|[^}]*\}\}", "", wiki)
    t = re.sub(r"\{\{[^{}]*\}\}", " ", t)
    t = re.sub(r"\[\[([^\]|]*\|)?([^\]]*)\]\]", r"\2", t)
    t = re.sub(r"'{2,}", "", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"[ \t]+", " ", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def split_articles(wiki, chapter, source):
    out = []
    parts = re.split(r"\n=+\s*(?:\{\{якорь\|[^}]*\}\})?\s*Статья\s+", wiki)
    for part in parts[1:]:
        m = re.match(r"(\d+(?:\.\d+)*)\.\s*([^=\n]{0,220})", part)
        if not m:
            continue
        body = clean(part[m.end():])
        if len(body) < 150:          # утратившие силу и чистые отсылки
            continue
        out.append({"article": m.group(1), "title": clean(m.group(2)).strip(),
                    "chapter": chapter, "source": source, "text": body})
    return out


if __name__ == "__main__":
    arts, seen = [], set()
    for title in chapter_titles():
        chapter = title.split("/")[-1]
        src = "https://ru.wikisource.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
        for a in split_articles(raw(title), chapter, src):
            if a["article"] in seen:
                continue
            seen.add(a["article"])
            arts.append(a)
        time.sleep(1)
    arts.sort(key=lambda a: [int(x) for x in a["article"].split(".")])
    Path("corpus.json").write_text(json.dumps(arts, ensure_ascii=False, indent=1), encoding="utf-8")
    sizes = sorted(len(a["text"]) for a in arts)
    print(f"статей: {len(arts)}")
    print(f"символов: {sum(sizes):,}".replace(",", " "))
    print(f"медиана статьи: {sizes[len(sizes)//2]:,}".replace(",", " "))
    print("примеры:", ", ".join(f'ст. {a["article"]}' for a in arts[:6]))
