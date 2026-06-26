-- Cost tracking tables for cost-aware-router plugin
-- All tables use the plugin_cost_router_* prefix for isolation

CREATE TABLE IF NOT EXISTS plugin_cost_router_cost_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    project TEXT,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0.0,
    tool_name TEXT,
    tier TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_cost_log_session ON plugin_cost_router_cost_log(session_id);
CREATE INDEX IF NOT EXISTS idx_cost_log_created ON plugin_cost_router_cost_log(created_at);
CREATE INDEX IF NOT EXISTS idx_cost_log_model ON plugin_cost_router_cost_log(model);
CREATE INDEX IF NOT EXISTS idx_cost_log_project ON plugin_cost_router_cost_log(project);

CREATE TABLE IF NOT EXISTS plugin_cost_router_budget_limits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scope TEXT NOT NULL UNIQUE,
    scope_type TEXT NOT NULL CHECK(scope_type IN ('session', 'daily', 'project', 'global')),
    budget_usd REAL NOT NULL,
    spent_usd REAL NOT NULL DEFAULT 0.0,
    alert_sent INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS plugin_cost_router_daily_summary (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL UNIQUE,
    total_cost_usd REAL NOT NULL DEFAULT 0.0,
    total_input_tokens INTEGER NOT NULL DEFAULT 0,
    total_output_tokens INTEGER NOT NULL DEFAULT 0,
    model_breakdown TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_daily_summary_date ON plugin_cost_router_daily_summary(date);
