"""Async tailer for skill-hub log files.

Emits JSON dicts {"ts","level","source","text"} for each new line. Portable
polling loop (200 ms) via ``aiofiles`` — no inotify dependency.
"""
from __future__ import annotations

import asyncio
import re
from collections import deque
from pathlib import Path
from typing import AsyncIterator, Callable, Iterable

import aiofiles

LOG_DIR = Path.home() / ".claude" / "mcp-skill-hub" / "logs"
HOOK_LOG = LOG_DIR / "hook-debug.log"
ACTIVITY_LOG = LOG_DIR / "activity.log"

_TS_RE = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\]\s*(.*)$")
_TAG_RE = re.compile(
    r"\b(HOOK|AUTO_APPROVE|AUTO_PROCEED|INTERCEPT|ROUTER|STOP|ERROR|WARN)\b"
)


def parse_line(raw: str, default_source: str = "") -> dict:
    """Extract ts/level/source/text from a log line (best-effort)."""
    text = raw.rstrip("\n")
    ts = ""
    m = _TS_RE.match(text)
    if m:
        ts = m.group(1)
        body = m.group(2)
    else:
        body = text
    level = "info"
    source = default_source
    tag_m = _TAG_RE.search(body)
    if tag_m:
        tag = tag_m.group(1)
        source = tag.lower()
        if tag in {"ERROR"}:
            level = "error"
        elif tag in {"WARN"}:
            level = "warn"
        elif tag == "HOOK":
            level = "hook"
        elif tag == "ROUTER":
            level = "router"
        elif tag in {"INTERCEPT"}:
            level = "intercept"
        elif tag == "STOP":
            level = "stop"
        elif tag.startswith("AUTO"):
            level = "hook"
    if "error" in body.lower() or "traceback" in body.lower():
        level = "error"
    return {"ts": ts, "level": level, "source": source, "text": body}


async def _seed_tail(path: Path, n: int) -> tuple[list[str], int]:
    """Return last ``n`` lines of ``path`` plus the resulting file offset."""
    if not path.exists():
        return [], 0
    async with aiofiles.open(path, "r", errors="replace") as f:
        buf: deque[str] = deque(maxlen=n)
        async for line in f:
            buf.append(line)
        pos = await f.tell()
    return list(buf), pos


async def tail_files(
    paths: Iterable[Path],
    seed_lines: int = 200,
    poll_interval: float = 0.2,
    predicate: Callable[[str], bool] | None = None,
) -> AsyncIterator[dict]:
    """Yield parsed log entries from ``paths`` forever.

    Starts with the last ``seed_lines`` lines of each file for context, then
    polls for appended data. New lines only; truncation resets the offset.
    """
    paths = [Path(p) for p in paths]
    offsets: dict[Path, int] = {}
    for p in paths:
        seed, pos = await _seed_tail(p, seed_lines)
        offsets[p] = pos
        src = p.stem
        for ln in seed:
            if predicate is not None and not predicate(ln):
                continue
            yield parse_line(ln, default_source=src)

    while True:
        any_new = False
        for p in paths:
            if not p.exists():
                continue
            try:
                size = p.stat().st_size
            except OSError:
                continue
            pos = offsets.get(p, 0)
            if size < pos:
                # rotated/truncated
                pos = 0
            if size == pos:
                continue
            try:
                async with aiofiles.open(p, "r", errors="replace") as f:
                    await f.seek(pos)
                    chunk = await f.read()
                    offsets[p] = await f.tell()
            except OSError:
                continue
            src = p.stem
            for ln in chunk.splitlines():
                any_new = True
                if predicate is not None and not predicate(ln):
                    continue
                yield parse_line(ln, default_source=src)
        if not any_new:
            await asyncio.sleep(poll_interval)


def tail_file_sync(path: Path, max_lines: int) -> list[str]:
    """Return the last ``max_lines`` lines of ``path`` (stdlib only)."""
    if not path.exists():
        return []
    buf: deque[str] = deque(maxlen=max_lines)
    with path.open("r", errors="replace") as f:
        for line in f:
            buf.append(line)
    return list(buf)
