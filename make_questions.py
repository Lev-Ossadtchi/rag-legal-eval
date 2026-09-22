#!/usr/bin/env python3
"""Набор вопросов для замера: по одному на статью, сформулирован своими словами.

Вопросы генерирует дешёвая модель по ТЕКСТУ статьи — так они звучат как настоящий
запрос юриста, а не как её заголовок. Эталон известен по построению: статья-источник.
Набор синтетический, и это честно сказано в README: он проверяет поиск,
а не заменяет вопросы реальных пользователей.

  python3 make_questions.py 60 > questions.json
"""
import json, os, random, re, sys, time, urllib.error, urllib.request
from pathlib import Path

KEY = (os.environ.get("OPENROUTER_API_KEY")
       or Path(os.path.expanduser("~/.config/openrouter/key")).read_text().strip())
MODEL = "qwen/qwen3.7-flash"
URL = "https://openrouter.ai/api/v1/chat/completions"

PROMPT = """Ниже статья закона о банкротстве. Сформулируй ОДИН вопрос, который юрист
задал бы своими словами, чтобы найти именно эту норму.

Требования:
- вопрос на русском, одно предложение, до 15 слов;
- НЕ повторяй заголовок статьи дословно и не упоминай её номер;
- используй бытовые формулировки, а не цитату из текста.

Ответ — только сам вопрос, без кавычек и пояснений.

Статья {num}. {title}
{body}"""


def ask(art, tries=4):
    """OpenRouter отвечает 429 при частых запросах — ждём и повторяем."""
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT.format(
            num=art["article"], title=art["title"], body=art["text"][:1800])}],
        "max_tokens": 900, "temperature": 0.3}).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Content-Type": "application/json", "Authorization": "Bearer " + KEY,
        "HTTP-Referer": "https://github.com/Lev-Ossadtchi", "X-Title": "rag-legal-eval"})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.load(r)
            break
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == tries - 1:
                raise
            time.sleep(6 * (attempt + 1))
    msg = data["choices"][0]["message"]
    text = ((msg.get("content") or "") + "\n" + (msg.get("reasoning") or "")).strip()
    return pick_question(text), data.get("usage", {})


CYR = re.compile(r"[а-яё]", re.I)


def pick_question(text):
    """Модель часто рассуждает вслух. Берём строку, которая реально является
    вопросом: кириллица, знак вопроса, без английских слов и без кавычек-цитат."""
    best = ""
    for line in text.splitlines():
        line = re.sub(r"^[-•\d.\s]+", "", line).strip(' "«»')
        if not line.endswith("?") or not CYR.search(line):
            continue
        if re.search(r"[a-z]{3,}", line, re.I):      # обрывки рассуждений на английском
            continue
        if not 15 < len(line) < 200:
            continue
        words = len(line.split())
        if 4 <= words <= 20 and (not best or words < len(best.split())):
            best = line
    return best


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    arts = json.loads(Path("corpus.json").read_text(encoding="utf-8"))
    arts = [a for a in arts if len(a["text"]) > 900]
    random.seed(42)
    sample = random.sample(arts, min(n, len(arts)))
    out, tin, tout = [], 0, 0
    for i, a in enumerate(sample, 1):
        try:
            q, usage = ask(a)
        except Exception as e:
            print(f"# пропуск ст. {a['article']}: {e}", file=sys.stderr)
            continue
        tin += usage.get("prompt_tokens", 0); tout += usage.get("completion_tokens", 0)
        if q:
            out.append({"question": q, "article": a["article"], "title": a["title"]})
        if i % 20 == 0:
            print(f"# {i}/{len(sample)}", file=sys.stderr, flush=True)
        time.sleep(1.5)
    Path("questions.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    cost = tin * 0.00000003 + tout * 0.00000013
    print(f"вопросов: {len(out)}; токенов {tin}+{tout}; стоимость ≈ ${cost:.4f}", file=sys.stderr)
