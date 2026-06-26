-- feedback-amplifier plugin schema
-- Tracks context when skills are suggested/injected for implicit feedback scoring

CREATE TABLE IF NOT EXISTS plugin_fbamp_feedback_context (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    skill_id        TEXT NOT NULL,
    session_id      TEXT NOT NULL,
    query           TEXT,
    domain_hints    TEXT,           -- JSON array of domain hints
    injection_id    INTEGER,        -- reference to skill_injections.id
    was_used        INTEGER DEFAULT 0,  -- 0=pending, 1=used, -1=ignored
    ts              REAL NOT NULL,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_plugin_fbamp_feedback_context_skill
    ON plugin_fbamp_feedback_context(skill_id);

CREATE INDEX IF NOT EXISTS idx_plugin_fbamp_feedback_context_session
    ON plugin_fbamp_feedback_context(session_id);

CREATE INDEX IF NOT EXISTS idx_plugin_fbamp_feedback_context_used
    ON plugin_fbamp_feedback_context(was_used, ts);


-- Per-skill aggregated scores with decay tracking
CREATE TABLE IF NOT EXISTS plugin_fbamp_skill_scores (
    skill_id        TEXT PRIMARY KEY,
    ema_score       REAL NOT NULL DEFAULT 1.0,
    last_used_at    TEXT,
    injection_count INTEGER DEFAULT 0,
    used_count      INTEGER DEFAULT 0,
    decay_applied_at TEXT,
    updated_at      TEXT DEFAULT (datetime('now'))
);


-- Per-domain skill performance (for domain-specific recommendations)
CREATE TABLE IF NOT EXISTS plugin_fbamp_domain_performance (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    skill_id        TEXT NOT NULL,
    domain          TEXT NOT NULL,
    success_count   INTEGER DEFAULT 0,
    total_count     INTEGER DEFAULT 0,
    last_at         TEXT DEFAULT (datetime('now')),
    UNIQUE(skill_id, domain)
);

CREATE INDEX IF NOT EXISTS idx_plugin_fbamp_domain_performance_skill
    ON plugin_fbamp_domain_performance(skill_id);

CREATE INDEX IF NOT EXISTS idx_plugin_fbamp_domain_performance_domain
    ON plugin_fbamp_domain_performance(domain);
