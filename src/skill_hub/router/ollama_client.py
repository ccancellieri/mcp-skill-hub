"""Tier 2 — local Ollama classifier.

Calls a lightweight local model (default: qwen2.5:3b) to produce a structured
JSON classification of the user's prompt. Falls back silently if Ollama is
unavailable or times out.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from .. import config as _cfg


_CLASSIFY_PROMPT = """\
You are a prompt complexity classifier. Given a user message sent to an AI coding assistant, \
reply with ONLY a JSON object — no prose, no markdown fences.

Required fields:
  complexity  : float 0.0-1.0  (0=trivial rename, 1=major architecture)
  ambiguity   : float 0.0-1.0  (0=crystal clear, 1=multiple valid interpretations)
  scope       : "single" | "multi" | "cross-repo"
  domain_hints: list[str]  (pick from: debugging, architecture, testing, frontend,
                             database, security, devops, api — empty list if none)
  confidence  : float 0.0-1.0  (your confidence in this classification)

Examples:
  "rename foo to bar in utils.py" → {{"complexity":0.1,"ambiguity":0.05,"scope":"single","domain_hints":[],"confidence":0.95}}
  "should we refactor auth or just patch it?" → {{"complexity":0.7,"ambiguity":0.8,"scope":"multi","domain_hints":["architecture","security"],"confidence":0.88}}

User message:
{prompt}

JSON:"""


@dataclass
class ClassifierResult:
    complexity: float = 0.5
    ambiguity: float = 0.3
    scope: str = "single"
    domain_hints: list[str] = field(default_factory=list)
    confidence: float = 0.5


def classify(
    prompt: str,
    cfg: dict[str, Any] | None = None,
    cwd: str = "",
) -> ClassifierResult | None:
    """Call local Ollama to classify *prompt*.

    Returns ``None`` on any error so the caller can fall back gracefully.
    """
    if cfg is None:
        cfg = _cfg.load_config()

    ollama_base: str = cfg.get("ollama_base", "http://localhost:11434")
    model: str = ((cfg.get("services") or {}).get("ollama_router") or {}).get("model") or "qwen2.5:3b"
    timeout: float = float(cfg.get("router_tier2_timeout", 10.0))

    # Prepend project context so the LLM knows which codebase it's routing for
    project_prefix = ""
    if cwd:
        project_name = os.path.basename(cwd.rstrip("/"))
        project_prefix = f"Project context: {project_name} ({cwd})\n\n"

    full_prompt = project_prefix + _CLASSIFY_PROMPT.format(prompt=prompt[:1500])

    try:
        resp = httpx.post(
            f"{ollama_base}/api/generate",
            json={
                "model": model,
                "prompt": full_prompt,
                "stream": False,
                "options": {"temperature": 0.05, "num_predict": 150},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "{}")
        # Strip any chain-of-thought blocks (e.g. deepseek-r1 <think>…</think>)
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        # Extract JSON even if the model wraps it in extra text
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return None
        data: dict[str, Any] = json.loads(m.group())
    except Exception:
        return None

    try:
        return ClassifierResult(
            complexity=float(data.get("complexity", 0.5)),
            ambiguity=float(data.get("ambiguity", 0.3)),
            scope=str(data.get("scope", "single")),
            domain_hints=[str(d) for d in data.get("domain_hints", [])],
            confidence=float(data.get("confidence", 0.5)),
        )
    except (TypeError, ValueError):
        return None
