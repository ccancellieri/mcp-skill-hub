"""Shared measurement helpers for the isolated context comparison fixtures.

These helpers do not run the historical router comparison described in README.
"""
import hashlib
import os
from pathlib import Path
import re


_O200K_BPE_URL = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
_O200K_BPE_SHA256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"


def load_tokenizer():
    cache_dir = os.environ.get("TIKTOKEN_CACHE_DIR")
    if not cache_dir:
        raise RuntimeError("TIKTOKEN_CACHE_DIR must name a pre-provisioned offline tokenizer cache")
    cache_key = hashlib.sha1(_O200K_BPE_URL.encode()).hexdigest()  # noqa: S324 - cache filename
    cache_path = Path(cache_dir) / cache_key
    if not cache_path.is_file():
        raise RuntimeError("o200k_base is not cached; provision it before running the offline benchmark")
    if hashlib.sha256(cache_path.read_bytes()).hexdigest() != _O200K_BPE_SHA256:
        raise RuntimeError("o200k_base cache checksum is invalid; provision a verified cache artifact")
    import tiktoken
    return tiktoken.get_encoding("o200k_base")


def source_catalog(corpus):
    catalog = {}
    for kind, section in (("skill", "skills"), ("task", "tasks"),
                          ("memory", "memories"), ("wiki", "wiki")):
        for item in corpus.get(section, []):
            key = item.get("id") if kind == "skill" else item["key"]
            text = " ".join(str(item.get(k, "")) for k in ("content", "context", "text"))
            markers = re.findall(r"EVIDENCE_[A-Z_0-9]+", text)
            if kind == "skill":
                markers.append(item["name"])
            catalog[f"{kind}:{key}"] = markers
    return catalog


def validate_corpus(corpus):
    sources = source_catalog(corpus)
    for case in corpus["cases"]:
        for key in ("expected_sources", "forbidden_sources"):
            if any(source not in sources for source in case.get(key, [])):
                raise ValueError("unknown source in corpus labels")


def extract_sources(text, prompt, catalog):
    if text.startswith(prompt):
        text = text[len(prompt):]
    return sorted(source for source, markers in catalog.items()
                  if any(marker in text for marker in markers))


def measure_output(prompt, output, tokenizer):
    context = output.get("hookSpecificOutput", {}).get("additionalContext", "")
    original = len(tokenizer.encode(prompt))
    additional = len(tokenizer.encode(context))
    return {"original_tokens": original, "additional_context_tokens": additional,
            "primary_total_tokens": original + additional,
            "ui_warning_tokens": len(tokenizer.encode(output.get("systemMessage", "")))}


def score_sources(*, expected, forbidden, retrieved):
    expected, retrieved = set(expected), set(retrieved)
    hits = len(expected & retrieved)
    precision = hits / len(retrieved) if retrieved else 0.0
    recall = hits / len(expected) if expected else 0.0
    return {"precision": precision if expected else None,
            "recall": recall if expected else None,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.,
            "no_answer_correct": not retrieved if not expected else None,
            "foreign_source_leakage": bool(retrieved & set(forbidden))}


def claim_gate(comparisons):
    return bool(comparisons) and all(row.get("lower_tokens") and row.get("higher_quality")
                                     for row in comparisons.values())
