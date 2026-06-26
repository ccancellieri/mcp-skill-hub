-- Contradiction detection results
-- Table names must be prefixed with plugin_{namespace}_

CREATE TABLE IF NOT EXISTS plugin_contradiction_findings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    page_a TEXT NOT NULL,
    page_b TEXT NOT NULL,
    claim_a TEXT NOT NULL,
    claim_b TEXT NOT NULL,
    confidence REAL NOT NULL,
    resolution_status TEXT NOT NULL DEFAULT 'pending',
    resolution TEXT,
    resolved_by TEXT,
    resolved_at TEXT,
    detected_at TEXT NOT NULL DEFAULT (datetime('now')),
    detection_run TEXT NOT NULL,
    UNIQUE(page_a, page_b, claim_a, claim_b)
);

CREATE INDEX IF NOT EXISTS idx_contradiction_status
    ON plugin_contradiction_findings(resolution_status);

CREATE INDEX IF NOT EXISTS idx_contradiction_pages
    ON plugin_contradiction_findings(page_a, page_b);

CREATE INDEX IF NOT EXISTS idx_contradiction_run
    ON plugin_contradiction_findings(detection_run);

-- Detection run history
CREATE TABLE IF NOT EXISTS plugin_contradiction_runs (
    id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    pages_scanned INTEGER DEFAULT 0,
    pairs_analyzed INTEGER DEFAULT 0,
    contradictions_found INTEGER DEFAULT 0,
    error_message TEXT
);
