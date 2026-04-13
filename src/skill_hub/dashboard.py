"""Benefit/cost dashboard renderer.

Reads aggregates from the existing SQLite store plus a tail-parse of
``~/.claude/mcp-skill-hub/logs/hook-debug.log``, then writes a single
self-contained HTML file (no CDNs) to
``~/.claude/mcp-skill-hub/reports/dashboard.html``.

Triggered from ``close_task`` so every task closure refreshes the page.
"""
from __future__ import annotations

import html
import logging
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

LOG_PATH = Path.home() / ".claude" / "mcp-skill-hub" / "logs" / "hook-debug.log"
OUT_PATH = Path.home() / ".claude" / "mcp-skill-hub" / "reports" / "dashboard.html"
VERDICTS_DB = Path.home() / ".claude" / "mcp-skill-hub" / "command_verdicts.db"
LOG_TAIL_BYTES = 2_000_000  # ~last 2 MB is plenty for a rolling view

# Conservative rule-of-thumb: 1s of local-LLM wall time ≈ 50 Sonnet-equivalent
# output tokens. Tuneable by the user; surfaced as a tooltip so the net figure
# is not silently deceptive.
TOKENS_PER_LLM_SECOND = 50


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

_RE_TIME_MS = re.compile(r"(?:cli_time|cli|time|total_time)=(\d+)ms")
_RE_AUTO_APPROVE = re.compile(
    r"AUTO_APPROVE\s+(\S+)\s+decision=(allow|deny|pass)", re.IGNORECASE
)
_RE_AUTO_PROCEED = re.compile(r"AUTO_PROCEED\s+PROCEED", re.IGNORECASE)
_RE_RESUME_CONSUME = re.compile(r"RESUME marker consumed")
_RE_INTERCEPT_ERROR = re.compile(r"INTERCEPT.*(error|cli_failed_or_empty)")


def _parse_log(path: Path = LOG_PATH, max_bytes: int = LOG_TAIL_BYTES) -> dict[str, Any]:
    """Scan the tail of the hook debug log for hook-derived metrics."""
    out: dict[str, Any] = {
        "auto_approve": Counter(),          # {"allow": n, "deny": n, "pass": n}
        "auto_approve_tool": Counter(),     # by tool_name
        "auto_proceed_fires": 0,
        "resume_consumed": 0,
        "intercept_errors": 0,
        "llm_ms": [],                       # list[int]  (for p50/p95/sum)
        "by_day_approve": defaultdict(lambda: Counter()),  # date -> Counter
        "by_day_llm_ms": defaultdict(int),  # date -> total ms
        "log_missing": False,
    }
    if not path.exists():
        out["log_missing"] = True
        return out
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
                f.readline()  # discard partial line
            data = f.read().decode("utf-8", errors="replace")
    except OSError as e:
        _log.warning("log read failed: %s", e)
        out["log_missing"] = True
        return out

    # Line format example:
    # [22:15:03] [  0.4s] AUTO_APPROVE Bash  decision=allow  cmd="git status"  ...
    ts_today = date.today()
    for line in data.splitlines():
        m_day = re.match(r"\[(\d{2}:\d{2}:\d{2})\]", line)
        # We don't know the date from the log line — bucket everything by
        # file-level scan order against "today" (best effort). For a true
        # daily trend the upstream logger should add YYYY-MM-DD, but this
        # approximation is fine for a live dashboard.
        day_key = ts_today.isoformat()

        m = _RE_AUTO_APPROVE.search(line)
        if m:
            tool_name, decision = m.group(1), m.group(2).lower()
            out["auto_approve"][decision] += 1
            out["auto_approve_tool"][tool_name] += 1
            out["by_day_approve"][day_key][decision] += 1
            continue
        if _RE_AUTO_PROCEED.search(line):
            out["auto_proceed_fires"] += 1
            continue
        if _RE_RESUME_CONSUME.search(line):
            out["resume_consumed"] += 1
            continue
        if _RE_INTERCEPT_ERROR.search(line):
            out["intercept_errors"] += 1
            continue
        m = _RE_TIME_MS.search(line)
        if m:
            ms = int(m.group(1))
            if ms > 0:
                out["llm_ms"].append(ms)
                out["by_day_llm_ms"][day_key] += ms
    return out


def _percentile(xs: list[int], p: float) -> int:
    if not xs:
        return 0
    xs2 = sorted(xs)
    k = max(0, min(len(xs2) - 1, int(round((p / 100.0) * (len(xs2) - 1)))))
    return xs2[k]


# ---------------------------------------------------------------------------
# DB aggregation
# ---------------------------------------------------------------------------

def _db_metrics(store: Any) -> dict[str, Any]:
    """Pull pre-computed counters from the store."""
    conn: sqlite3.Connection = store._conn  # intentional: reuse the shared conn
    out: dict[str, Any] = {}

    # Cheap counters (already exposed).
    out["skills"] = store.count_skills()
    out["tasks"] = store.count_tasks()
    out["teachings"] = store.count_teachings()
    out["interceptions"] = store.count_interceptions()
    out["tokens_saved"] = store.total_tokens_saved()

    # Embeddings health.
    rows = conn.execute(
        "SELECT model, COUNT(*) as n FROM embeddings GROUP BY model"
    ).fetchall()
    out["embeddings"] = [{"model": r["model"], "n": r["n"]} for r in rows]

    # Feedback ratio.
    rows = conn.execute(
        "SELECT helpful, COUNT(*) as n FROM feedback GROUP BY helpful"
    ).fetchall()
    fb = {int(r["helpful"]): r["n"] for r in rows}
    out["feedback_helpful"] = fb.get(1, 0)
    out["feedback_unhelpful"] = fb.get(0, 0)

    # Triage breakdown (if table exists — defensive).
    try:
        rows = conn.execute(
            "SELECT action, COUNT(*) as n, "
            "COALESCE(SUM(estimated_tokens_saved), 0) as tok "
            "FROM triage_log GROUP BY action ORDER BY n DESC"
        ).fetchall()
        out["triage"] = [
            {"action": r["action"], "n": r["n"], "tokens": r["tok"]} for r in rows
        ]
    except sqlite3.OperationalError:
        out["triage"] = []

    # Interception breakdown.
    try:
        rows = store.get_interception_stats()
        out["intercept_by_type"] = [
            {"type": r["command_type"], "n": r["intercept_count"],
             "tokens": r["total_tokens_saved"] or 0}
            for r in rows
        ]
    except Exception:  # noqa: BLE001
        out["intercept_by_type"] = []

    # Context injection impact.
    try:
        row = conn.execute(
            "SELECT COUNT(*) as n, "
            "COALESCE(SUM(chars_injected), 0) as chars, "
            "COALESCE(AVG(skills_found), 0) as avg_skills "
            "FROM context_injections"
        ).fetchone()
        out["context_injections"] = {
            "n": row["n"], "chars": row["chars"],
            "avg_skills": round(row["avg_skills"] or 0, 2),
        }
    except sqlite3.OperationalError:
        out["context_injections"] = {"n": 0, "chars": 0, "avg_skills": 0}

    # Tasks closed over time (last 30 days).
    try:
        rows = conn.execute(
            "SELECT DATE(closed_at) as day, COUNT(*) as n "
            "FROM tasks WHERE status='closed' AND closed_at IS NOT NULL "
            "AND closed_at >= DATE('now','-30 day') "
            "GROUP BY day ORDER BY day"
        ).fetchall()
        out["closed_by_day"] = [(r["day"], r["n"]) for r in rows]
    except sqlite3.OperationalError:
        out["closed_by_day"] = []

    # Interceptions over time.
    try:
        rows = conn.execute(
            "SELECT DATE(created_at) as day, "
            "COUNT(*) as n, COALESCE(SUM(estimated_tokens),0) as tok "
            "FROM interceptions WHERE created_at >= DATE('now','-30 day') "
            "GROUP BY day ORDER BY day"
        ).fetchall()
        out["intercept_by_day"] = [
            {"day": r["day"], "n": r["n"], "tok": r["tok"]} for r in rows
        ]
    except sqlite3.OperationalError:
        out["intercept_by_day"] = []

    return out


def _verdict_metrics(path: Path = VERDICTS_DB) -> dict[str, Any]:
    """Read the auto-approve verdict cache (separate tiny DB)."""
    out: dict[str, Any] = {"exists": path.exists(), "by_source": [], "top": [],
                           "total": 0, "hits_total": 0}
    if not path.exists():
        return out
    try:
        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        out["total"] = conn.execute(
            "SELECT COUNT(*) as n FROM command_verdicts"
        ).fetchone()["n"]
        out["hits_total"] = conn.execute(
            "SELECT COALESCE(SUM(hit_count),0) as n FROM command_verdicts"
        ).fetchone()["n"]
        rows = conn.execute(
            "SELECT source, decision, COUNT(*) as n, SUM(hit_count) as hits "
            "FROM command_verdicts GROUP BY source, decision "
            "ORDER BY hits DESC"
        ).fetchall()
        out["by_source"] = [
            {"source": r["source"], "decision": r["decision"],
             "n": r["n"], "hits": r["hits"] or 0}
            for r in rows
        ]
        rows = conn.execute(
            "SELECT command, source, hit_count FROM command_verdicts "
            "WHERE decision='allow' ORDER BY hit_count DESC LIMIT 10"
        ).fetchall()
        out["top"] = [
            {"command": r["command"], "source": r["source"],
             "hits": r["hit_count"]}
            for r in rows
        ]
        conn.close()
    except sqlite3.Error:
        pass
    return out


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

def _sparkline_svg(values: list[float], width: int = 180, height: int = 36,
                   color: str = "#2b7bd6") -> str:
    """Render a stand-alone SVG polyline sparkline."""
    if not values:
        return ('<svg class="spark" width="{w}" height="{h}"></svg>'
                .format(w=width, h=height))
    vmax = max(values) or 1.0
    n = len(values)
    if n == 1:
        values = [values[0], values[0]]
        n = 2
    pts = []
    for i, v in enumerate(values):
        x = i * (width - 4) / (n - 1) + 2
        y = height - 2 - (v / vmax) * (height - 4)
        pts.append(f"{x:.1f},{y:.1f}")
    poly = " ".join(pts)
    return (
        f'<svg class="spark" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        f'<polyline fill="none" stroke="{color}" stroke-width="1.8" '
        f'points="{poly}"/>'
        '</svg>'
    )


def _esc(s: Any) -> str:
    return html.escape(str(s))


def _kpi(label: str, value: str, sub: str = "") -> str:
    return (
        f'<div class="kpi"><div class="kpi-label">{_esc(label)}</div>'
        f'<div class="kpi-value">{_esc(value)}</div>'
        f'<div class="kpi-sub">{_esc(sub)}</div></div>'
    )


def _render(db: dict[str, Any], logm: dict[str, Any],
            vcache: dict[str, Any]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tokens_saved = int(db["tokens_saved"] or 0)
    llm_seconds = sum(logm["llm_ms"]) / 1000.0
    llm_cost_eq = int(llm_seconds * TOKENS_PER_LLM_SECOND)
    net = tokens_saved - llm_cost_eq
    helpful = db["feedback_helpful"]
    unhelpful = db["feedback_unhelpful"]
    total_fb = helpful + unhelpful
    helpful_pct = (helpful / total_fb * 100) if total_fb else 0
    tasks_open = db["tasks"].get("open", 0)
    tasks_closed = db["tasks"].get("closed", 0)

    approve = logm["auto_approve"]["allow"]
    deny = logm["auto_approve"]["deny"]
    pass_through = logm["auto_approve"]["pass"]

    spark_intercept = _sparkline_svg([x["n"] for x in db["intercept_by_day"]])
    spark_closed = _sparkline_svg([n for _, n in db["closed_by_day"]], color="#2b9d5e")
    spark_llm = _sparkline_svg(
        sorted(logm["by_day_llm_ms"].values()), color="#d6732b"
    )

    p50 = _percentile(logm["llm_ms"], 50)
    p95 = _percentile(logm["llm_ms"], 95)

    emb_rows = "".join(
        f"<tr><td>{_esc(e['model'])}</td><td>{e['n']:,}</td></tr>"
        for e in db["embeddings"]
    ) or "<tr><td colspan=2>no embeddings yet</td></tr>"

    triage_rows = "".join(
        f"<tr><td>{_esc(t['action'])}</td><td>{t['n']:,}</td>"
        f"<td>{t['tokens']:,}</td></tr>"
        for t in db["triage"]
    ) or "<tr><td colspan=3>no triage events yet</td></tr>"

    intercept_rows = "".join(
        f"<tr><td>{_esc(i['type'])}</td><td>{i['n']:,}</td>"
        f"<td>{i['tokens']:,}</td></tr>"
        for i in db["intercept_by_type"]
    ) or "<tr><td colspan=3>no interceptions yet</td></tr>"

    # Auto-approve breakdown by tool.
    aat_rows = "".join(
        f"<tr><td>{_esc(k)}</td><td>{v:,}</td></tr>"
        for k, v in logm["auto_approve_tool"].most_common(10)
    ) or "<tr><td colspan=2>no auto-approve events yet</td></tr>"

    vcache_rows = "".join(
        f"<tr><td>{_esc(r['source'])}</td><td>{_esc(r['decision'])}</td>"
        f"<td>{r['n']:,}</td><td>{r['hits']:,}</td></tr>"
        for r in vcache["by_source"]
    ) or "<tr><td colspan=4>no verdicts cached yet</td></tr>"

    vcache_top = "".join(
        f"<tr><td><code>{_esc(r['command'][:70])}</code></td>"
        f"<td>{_esc(r['source'])}</td><td>{r['hits']:,}</td></tr>"
        for r in vcache["top"]
    ) or "<tr><td colspan=3>—</td></tr>"

    log_note = ""
    if logm["log_missing"]:
        log_note = ('<p class="warn">hook-debug.log not found — hook metrics '
                    'unavailable until a session runs.</p>')

    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<title>skill-hub — benefit / cost</title>
<style>
 :root {{
   --fg:#1b2430; --muted:#6b7280; --line:#e5e7eb;
   --good:#2b9d5e; --bad:#c0392b; --warn:#d6732b; --accent:#2b7bd6;
 }}
 body {{ font-family:-apple-system,BlinkMacSystemFont,sans-serif;
         margin:0; padding:24px; color:var(--fg); background:#fafbfc; }}
 h1 {{ margin:0 0 4px; font-size:20px; }}
 .sub {{ color:var(--muted); font-size:13px; margin-bottom:20px; }}
 .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
          gap:12px; margin-bottom:24px; }}
 .kpi {{ background:#fff; border:1px solid var(--line); border-radius:8px;
         padding:14px 16px; }}
 .kpi-label {{ font-size:11px; color:var(--muted); text-transform:uppercase;
               letter-spacing:.04em; }}
 .kpi-value {{ font-size:22px; font-weight:600; margin-top:4px; }}
 .kpi-sub {{ font-size:11px; color:var(--muted); margin-top:4px; }}
 .kpi.good .kpi-value {{ color:var(--good); }}
 .kpi.bad .kpi-value {{ color:var(--bad); }}
 .kpi.warn .kpi-value {{ color:var(--warn); }}
 .row {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr));
         gap:16px; margin-bottom:20px; }}
 .card {{ background:#fff; border:1px solid var(--line); border-radius:8px;
          padding:14px 16px; }}
 .card h2 {{ font-size:13px; margin:0 0 10px; text-transform:uppercase;
             letter-spacing:.04em; color:var(--muted); }}
 table {{ width:100%; border-collapse:collapse; font-size:13px; }}
 th, td {{ padding:6px 8px; text-align:left; border-bottom:1px solid var(--line); }}
 th {{ font-weight:500; color:var(--muted); font-size:11px;
       text-transform:uppercase; letter-spacing:.04em; }}
 tr:last-child td {{ border-bottom:none; }}
 td:nth-child(n+2) {{ text-align:right; font-variant-numeric:tabular-nums; }}
 .spark {{ display:block; margin-top:6px; }}
 .warn {{ color:var(--warn); font-size:12px; margin-top:6px; }}
 .footer {{ color:var(--muted); font-size:11px; margin-top:24px; }}
 .tooltip {{ border-bottom:1px dotted var(--muted); cursor:help; }}
</style>
</head><body>
<h1>skill-hub — benefit / cost</h1>
<div class="sub">Regenerated at {now} · source:
 <code>~/.claude/mcp-skill-hub/skill_hub.db</code> +
 <code>hook-debug.log</code></div>

<div class="grid">
 {_kpi("Tokens saved (est.)", f"{tokens_saved:,}", "sum of interception estimates")}
 {_kpi("Net savings (est.)", f"{net:+,}",
       f"minus ≈ {llm_cost_eq:,} tok local-LLM equiv.")}
 {_kpi("Tasks closed / open", f"{tasks_closed} / {tasks_open}", "")}
 {_kpi("Skills indexed", f"{db['skills']:,}", "")}
 {_kpi("Teachings", f"{db['teachings']:,}", "explicit rules in play")}
 {_kpi("Feedback helpful",
       f"{helpful_pct:.0f}%" if total_fb else "—",
       f"{helpful} ↑ · {unhelpful} ↓")}
</div>

<div class="row">
 <div class="card"><h2>Auto-approve hook</h2>
  <table>
   <tr><th>decision</th><th>count</th></tr>
   <tr><td>allow</td><td>{approve:,}</td></tr>
   <tr><td>deny</td><td>{deny:,}</td></tr>
   <tr><td>pass-through</td><td>{pass_through:,}</td></tr>
  </table>
  <table style="margin-top:10px">
   <tr><th>by tool</th><th>count</th></tr>
   {aat_rows}
  </table>
 </div>

 <div class="card"><h2>Auto-proceed &amp; resume</h2>
  <table>
   <tr><td>auto-proceed fires</td><td>{logm['auto_proceed_fires']:,}</td></tr>
   <tr><td>resume markers consumed</td><td>{logm['resume_consumed']:,}</td></tr>
   <tr><td>intercept errors</td><td>{logm['intercept_errors']:,}</td></tr>
  </table>
 </div>

 <div class="card"><h2>Local-LLM cost
  <span class="tooltip" title="1s ≈ {TOKENS_PER_LLM_SECOND} Sonnet-equivalent output tokens (configurable).">(?)</span>
  </h2>
  <table>
   <tr><td>total wall time</td><td>{llm_seconds:,.1f} s</td></tr>
   <tr><td>p50 / p95 latency</td><td>{p50} / {p95} ms</td></tr>
   <tr><td>samples</td><td>{len(logm['llm_ms']):,}</td></tr>
  </table>
  {spark_llm}
 </div>
</div>

<div class="row">
 <div class="card"><h2>Interceptions by type</h2>
  <table>
   <tr><th>type</th><th>count</th><th>tokens saved</th></tr>
   {intercept_rows}
  </table>
  <div style="margin-top:6px">{spark_intercept}</div>
 </div>

 <div class="card"><h2>Triage log</h2>
  <table>
   <tr><th>action</th><th>count</th><th>tokens saved</th></tr>
   {triage_rows}
  </table>
 </div>

 <div class="card"><h2>Vector index</h2>
  <table>
   <tr><th>embedding model</th><th>vectors</th></tr>
   {emb_rows}
  </table>
  <table style="margin-top:10px">
   <tr><td>context injections</td><td>{db['context_injections']['n']:,}</td></tr>
   <tr><td>chars injected</td><td>{db['context_injections']['chars']:,}</td></tr>
   <tr><td>avg skills per inject</td><td>{db['context_injections']['avg_skills']}</td></tr>
  </table>
 </div>
</div>

<div class="row">
 <div class="card"><h2>Tasks closed (last 30 days)</h2>
  {spark_closed}
 </div>

 <div class="card"><h2>Adaptive auto-approve cache</h2>
  <table>
   <tr><th>source</th><th>decision</th><th>entries</th><th>hits</th></tr>
   {vcache_rows}
  </table>
  <div style="margin-top:8px; color:var(--muted); font-size:11px">
   Sources: <b>user_approved</b> = learned from your approvals · <b>llm</b> =
   local classifier · <b>static</b> = allow-list YAML.
  </div>
 </div>

 <div class="card"><h2>Top auto-approved commands</h2>
  <table>
   <tr><th>command</th><th>source</th><th>hits</th></tr>
   {vcache_top}
  </table>
 </div>
</div>

{log_note}

<div class="footer">
 To regenerate manually, call the <code>render_dashboard</code> MCP tool.
 Source: <code>src/skill_hub/dashboard.py</code>.
</div>
</body></html>
"""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

_WEBAPP_SINGLETON: dict[str, Any] = {"url": None, "thread": None, "server": None}


def render_interactive(store: Any) -> str | None:
    """Boot the FastAPI webapp in a daemon thread and return its URL.

    Singleton: subsequent calls return the already-running URL.
    Returns None if config disables the server or the port bind fails.
    """
    if _WEBAPP_SINGLETON["url"]:
        return _WEBAPP_SINGLETON["url"]

    cfg_path = Path.home() / ".claude" / "mcp-skill-hub" / "config.json"
    cfg: dict[str, Any] = {}
    try:
        import json as _json
        cfg = _json.loads(cfg_path.read_text())
    except Exception:  # noqa: BLE001
        pass
    if cfg.get("dashboard_server_enabled") is False:
        return None

    host = cfg.get("dashboard_server_host", "127.0.0.1")
    port = int(cfg.get("dashboard_server_port", 8765))

    try:
        import socket
        import threading
        import uvicorn
        from .webapp import create_app

        # Pre-bind check: fail fast and return None if port busy.
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError as e:
            _log.warning("dashboard port %s busy: %s", port, e)
            probe.close()
            return None
        probe.close()

        app = create_app(store)
        config = uvicorn.Config(
            app, host=host, port=port, log_level="warning",
            access_log=False, workers=1,
        )
        server = uvicorn.Server(config)
        # Disable uvicorn's signal handlers: they can only install on the
        # main thread, and we run in a daemon thread.
        server.install_signal_handlers = lambda: None  # type: ignore[assignment]

        def _run() -> None:
            try:
                server.run()
            except Exception as e:  # noqa: BLE001
                _log.warning("uvicorn crashed: %s", e)

        t = threading.Thread(target=_run, name="skill-hub-webapp", daemon=True)
        t.start()

        url = f"http://{host}:{port}/"
        _WEBAPP_SINGLETON["url"] = url
        _WEBAPP_SINGLETON["thread"] = t
        _WEBAPP_SINGLETON["server"] = server

        if cfg.get("dashboard_auto_open_browser"):
            try:
                import webbrowser
                webbrowser.open(url)
            except Exception:  # noqa: BLE001
                pass

        return url
    except Exception as e:  # noqa: BLE001
        _log.warning("render_interactive failed: %s", e)
        return None


def render(store: Any, out_path: Path = OUT_PATH,
           log_path: Path = LOG_PATH) -> Path:
    """Aggregate metrics and write the HTML file. Returns the path on success."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    db = _db_metrics(store)
    logm = _parse_log(log_path)
    vcache = _verdict_metrics()
    html_text = _render(db, logm, vcache)
    out_path.write_text(html_text, encoding="utf-8")
    return out_path
