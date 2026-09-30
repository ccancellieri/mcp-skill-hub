"""Local, deterministic ranking of skill names and descriptions."""
from __future__ import annotations

import re
import unicodedata

_STOP = frozenset({
    "a", "about", "an", "and", "are", "behavior", "build", "can", "could", "create",
    "do", "existing", "for", "from", "help", "how", "i", "in", "is", "keep", "me", "my",
    "of", "on", "optimize", "or", "please", "should", "skill", "skills", "task", "the",
    "this", "to", "use", "using", "what", "which", "with", "would", "you",
    "your", "ad", "ai", "al", "alla", "anche", "che", "come", "con",
    "crea", "creare", "di", "e", "il", "la", "le", "lo", "mi", "per",
    "posso", "quale", "quali", "questo", "su", "un", "una", "usare",
})

# A few common artifact words bridge English descriptions and Italian requests.
_ITALIAN_ARTIFACTS = {
    "presentazione": "presentation", "presentazioni": "presentation",
    "diapositiva": "slide", "diapositive": "slide",
    "foglio": "spreadsheet", "fogli": "spreadsheet",
    "documento": "document", "documenti": "document",
    "immagine": "image", "immagini": "image",
    "grafico": "chart", "grafici": "chart",
    "sito": "website", "siti": "website",
}


def _normalize(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", text.casefold())
                   if not unicodedata.combining(ch))


def terms(text: str) -> set[str]:
    result = set()
    for word in re.findall(r"[a-z0-9]+", _normalize(text)):
        if word in _STOP or len(word) < 3:
            continue
        word = _ITALIAN_ARTIFACTS.get(word, word)
        if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        result.add(word)
    return result


def explicit_skill_ids(prompt: str, rows: list[dict]) -> set[str]:
    """Return IDs explicitly named with ``$name`` or ``$plugin:name``."""
    requested = {_normalize(match).strip("-_:") for match in
                 re.findall(r"\$([\w:-]+)", prompt)}
    if not requested:
        return set()
    return {
        row["id"] for row in rows
        if requested & {_normalize(row.get("name") or ""), _normalize(row["id"]),
                        _normalize(row["id"].split(":")[-1])}
    }


def rank_skills(prompt: str, rows: list[dict], limit: int) -> list[tuple[dict, int]]:
    """Return metadata matches in stable order, abstaining on weak generic hits."""
    limit = max(0, min(limit, 20))
    if not limit:
        return []
    query = terms(prompt)
    explicit = explicit_skill_ids(prompt, rows)
    if not query and not explicit:
        return []

    ranked: list[tuple[dict, int]] = []
    for row in rows:
        description = row.get("description") or ""
        invoked = row["id"] in explicit
        if not terms(description) and not invoked:
            continue
        name = row.get("name") or ""
        name_hits = query & terms(name)
        description_hits = query & terms(description)
        hits = name_hits | description_hits
        if not invoked and not hits:
            continue
        score = 5_000 + 1_000 * len(name_hits) + 100 * len(description_hits)
        score += 10 * len(hits)
        if invoked:
            score += 10_000
        ranked.append((row, score))
    ranked.sort(key=lambda item: (-item[1], item[0]["id"]))
    return ranked[:limit]
