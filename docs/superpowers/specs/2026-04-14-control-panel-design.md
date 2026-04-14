# Control Panel — Service Lifecycle Management

**Date:** 2026-04-14
**Status:** Design approved, awaiting implementation plan
**Motivation:** mcp-skill-hub's background services (Ollama daemon, model loads, SearXNG container, in-process tasks) consume non-trivial CPU/RAM on the developer machine. The existing `/settings` page exposes config toggles but does not start, stop, or unload the underlying processes — so disabling a feature in the UI does not reclaim resources. This design adds a Control Panel that owns full service lifecycle.

## Goals

1. Single page (`/control`) where every controllable service can be started, stopped, and configured.
2. Toggling a service in the UI **physically** starts/stops the underlying process or container — RAM and CPU are reclaimed, not just bypassed.
3. State is persistent: a service toggled off stays off across hub restarts and machine reboots.
4. Live effect on running sessions: the in-process MCP server reconciles to match config within ~2 seconds; hooks pick up changes on next prompt.
5. Failures never block hooks — disabled services are skipped silently in the request path; a sticky banner reminds the user what is currently disabled.

## Non-goals

- Auto-creating the SearXNG Docker container if missing — `install.sh` handles that.
- Installing Ollama or Docker if missing — surfaced as `unavailable` with PATH hint.
- Multi-host orchestration — single-machine only.
- RAM/CPU telemetry beyond what `ollama ps` and `docker stats` already expose.

## Architecture

```
src/skill_hub/services/                    NEW
  base.py        Service ABC: status() start() stop() is_available()
  ollama.py      OllamaDaemon, OllamaModel(name)
  searxng.py     SearxngContainer
  inproc.py      WatcherTask, SkillEvolutionTask
  registry.py    name -> Service singleton; reconcile(config)

src/skill_hub/webapp/routes/control.py     NEW
  GET  /control                           full panel
  GET  /control/{svc}/card                single card refresh
  POST /control/{svc}/start               start + return updated card
  POST /control/{svc}/stop                stop  + return updated card
  POST /control/{svc}/toggle              convenience
  POST /control/{svc}/config              update mutable fields (model name, container name)

src/skill_hub/webapp/templates/
  control.html                             NEW: card grid
  _service_card.html                       NEW: HTMX-swappable partial
  base.html                                MODIFIED: sticky banner

src/skill_hub/webapp/middleware/
  banner.py                                NEW: inject disabled_services into all template responses
```

### Single source of truth

A new `services` dict in `config.json`:

```json
"services": {
  "auto_reconcile": true,
  "ollama_daemon":   {"enabled": true, "auto_start": true},
  "ollama_router":   {"enabled": true, "auto_start": true, "model": "qwen2.5:3b"},
  "ollama_embed":    {"enabled": true, "auto_start": true, "model": "nomic-embed-text"},
  "searxng":         {"enabled": true, "auto_start": true, "container": "skill-hub-searxng"},
  "watcher":         {"enabled": true, "auto_start": true},
  "skill_evolution": {"enabled": true, "auto_start": true},
  "haiku_router":    {"enabled": false}
}
```

- `enabled` — persisted desired state. The reconciler aligns OS state to this value.
- `auto_start` — controls hub-launch behavior. `true` = reconciler aligns OS state to `enabled` on startup. `false` = hub leaves whatever is currently running alone (manual-only).
- `services.auto_reconcile` — global kill-switch for the 2-second reconciliation loop. Off = changes only apply on next hub restart.

## Per-service lifecycle

| Service | `status()` | `start()` | `stop()` | Notes |
|---|---|---|---|---|
| **ollama_daemon** | `pgrep -f "ollama serve"` | `subprocess.Popen(["ollama","serve"], start_new_session=True)` detached | `pkill -f "ollama serve"` (SIGTERM, 5s grace, then SIGKILL) | macOS: prefer `brew services` if user installed via Homebrew (detect via `brew services list \| grep ollama`). |
| **ollama_router** | `ollama ps` parse — loaded if model name appears | `ollama run <model> ""` (loads into VRAM, returns immediately) | `ollama stop <model>` | Depends on `ollama_daemon`. Start fails with helpful message if daemon down. |
| **ollama_embed** | same as router | same | same | same |
| **searxng** | `docker inspect -f '{{.State.Running}}' <container>` | `docker start <container>` | `docker stop -t 5 <container>` | If container doesn't exist: `start()` returns `unavailable` with "run install.sh first" hint. Container name configurable. |
| **watcher** | `app.state.watcher_task` not done | spawn task, store in `app.state` | `task.cancel(); await task` | in-process; no subprocess |
| **skill_evolution** | `app.state.skill_evolution_task` not done | spawn task | cancel + await | in-process |
| **haiku_router** | always available (cloud API) | flip `enabled` only | flip `enabled` only | no process; pure config gate |

### Error handling rules

- `is_available()` returns `(False, reason)` when prerequisite missing (`ollama` binary not in PATH, `docker` daemon not running). Card renders disabled with reason in tooltip.
- `start()` and `stop()` never raise to the route handler — they return `(ok: bool, message: str)`. Route returns 200 + updated card HTML always; failures show inline as red text under the action button for 5 seconds.
- Hooks never call `start()` or `stop()` — only `is_enabled()`. If disabled, the hook skips that signal silently. Aligns with the project's "hooks never block" principle.

### Reconciliation loop

`registry.reconcile(cfg)` runs every 2 seconds in a background asyncio task spawned by `server.py` startup:

1. For each service, compare `desired = cfg.services[name].enabled` vs `actual = service.status() == running`.
2. If mismatch, call `start()` or `stop()`. Idempotent.
3. Update `app.state.disabled_services` (set of display names) for the banner middleware.

If `services.auto_reconcile == false`, the loop is not started; changes only apply at next hub launch (subject to `auto_start`).

## Data flow

**User toggles SearXNG off:**

1. Browser POSTs `/control/searxng/toggle`.
2. Route handler reads current `services.searxng.enabled`, flips it, saves config.
3. Handler calls `registry.get("searxng").stop()` directly (don't wait for the 2s loop).
4. Handler returns the updated card HTML; HTMX swaps it in.
5. Within 2 seconds, the reconciler loop notices the config delta on its next pass (no-op, already stopped) and updates `app.state.disabled_services` to include "SearXNG".
6. The banner middleware reads `app.state.disabled_services` for the next request and renders the sticky warning on every page.
7. Hooks running in separate processes re-read `config.json` per invocation (they already do this) and skip the SearXNG signal on the next prompt.

## UI

```
┌─ Control Panel ────────────────────────────────────────┐
│ [✓] Auto-reconcile every 2s   [✓] Auto-start on launch│
└────────────────────────────────────────────────────────┘

┌─ Ollama daemon ──────────────── ● running ────────────┐
│ Local LLM host. Required by router/embed models.      │
│ [ Stop ]   auto_start: [✓]                            │
└────────────────────────────────────────────────────────┘

┌─ Ollama router model ────────── ● loaded (1.8 GB) ────┐
│ qwen2.5:3b — used by router T2.                       │
│ [ Unload ]   model: [qwen2.5:3b____]   auto_start:[✓] │
│   Depends on: Ollama daemon ✓                         │
└────────────────────────────────────────────────────────┘

┌─ SearXNG ───────────────────── ○ stopped ─────────────┐
│ Web-search backend (Docker container).                │
│ [ Start ]   container: [skill-hub-searxng__]          │
│   docker available ✓                                  │
└────────────────────────────────────────────────────────┘

┌─ Haiku router (T3) ─────────── ○ disabled ────────────┐
│ Cloud API tier — costs $. No process to manage.       │
│ [ Enable ]                                             │
└────────────────────────────────────────────────────────┘
```

- Status dot: green=running, gray=stopped, red=unavailable, yellow=transitioning.
- Action button text flips Start↔Stop based on current state.
- Mutable fields (model name, container name) are inline `<input>` tags with auto-save-on-blur posting to `/control/{svc}/config`.
- Each card self-refreshes every 5 seconds via `hx-trigger="every 5s"` on the card root → GET `/control/{svc}/card` returns just the partial.

### Sticky banner (`base.html`, every page)

```jinja
{% if disabled_services %}
<div class="banner-warning">
  ⚠ Disabled: {{ disabled_services|join(", ") }}
  <a href="/control">→ control panel</a>
</div>
{% endif %}
```

`disabled_services` is injected via FastAPI middleware reading `app.state.disabled_services` (updated by the reconciler).

## Config migration

One-shot, on first hub launch after upgrade. `config.py::load_config()` detects legacy keys, folds them into the new `services` dict, and deletes them. Writes the migrated config back. Idempotent (no-op on second run).

| Legacy key | New location |
|---|---|
| `router_enabled` | implicit: router T2 disabled if `services.ollama_router.enabled == false` |
| `router_ollama_model` | `services.ollama_router.model` |
| `router_haiku_enabled` | `services.haiku_router.enabled` |
| `searxng_enabled` | `services.searxng.enabled` |
| `searxng_url` | stays top-level (connection target, not service) |
| `skill_evolution_enabled` | `services.skill_evolution.enabled` |
| `embedding_model` | `services.ollama_embed.model` |

All consumer modules (`router/ollama_client.py`, `router/haiku_client.py`, `searxng.py`, `embeddings.py`, hooks) updated to read from `services.*`. No coexistence shim — atomic rename across call-sites per the project's no-legacy-coexistence rule.

## Testing

| File | Coverage |
|---|---|
| `tests/test_services_base.py` | Service ABC contract |
| `tests/test_services_ollama.py` | mock `subprocess.run`/`subprocess.Popen`; verify argv for daemon/model start/stop; status parsing of captured `ollama ps` fixtures |
| `tests/test_services_searxng.py` | mock docker calls; container-missing path returns `unavailable` |
| `tests/test_services_inproc.py` | real asyncio task lifecycle (no mocks); cancel + restart |
| `tests/test_services_registry.py` | `reconcile()` drives correct start/stop given config delta |
| `tests/test_routes_control.py` | HTMX endpoints return updated card HTML; toggle persists to config |
| `tests/test_config_migration.py` | legacy keys folded correctly; idempotent on second run |
| `tests/test_banner.py` | middleware injects `disabled_services` into all template responses |

## File inventory

**New:**
- `src/skill_hub/services/__init__.py`
- `src/skill_hub/services/base.py`
- `src/skill_hub/services/ollama.py`
- `src/skill_hub/services/searxng.py`
- `src/skill_hub/services/inproc.py`
- `src/skill_hub/services/registry.py`
- `src/skill_hub/webapp/routes/control.py`
- `src/skill_hub/webapp/templates/control.html`
- `src/skill_hub/webapp/templates/_service_card.html`
- `src/skill_hub/webapp/middleware/banner.py`
- `tests/test_services_*.py` (5 files)
- `tests/test_routes_control.py`
- `tests/test_config_migration.py`
- `tests/test_banner.py`

**Modified:**
- `src/skill_hub/config.py` — `services` defaults + migration
- `src/skill_hub/server.py` — reconciler task on startup + config-watcher
- `src/skill_hub/webapp/main.py` — register control router + banner middleware
- `src/skill_hub/webapp/templates/base.html` — banner block + nav link
- `src/skill_hub/router/ollama_client.py` — read `services.ollama_router.model`
- `src/skill_hub/router/haiku_client.py` — read `services.haiku_router.enabled`
- `src/skill_hub/searxng.py` — read `services.searxng.enabled`
- `src/skill_hub/embeddings.py` — read `services.ollama_embed.model`
- `src/skill_hub/watcher.py` — spawned by registry, not directly
- `hooks/*.sh` and `hooks/*.py` — update legacy key reads
