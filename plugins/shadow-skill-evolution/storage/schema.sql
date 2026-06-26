-- A7 — Plugin storage schema for shadow-skill-evolution
-- Tracks tool chains and skill proposals.

-- Tool chains extracted from session events
CREATE TABLE IF NOT EXISTS tool_chains (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    chain_hash      TEXT NOT NULL,       -- SHA1 of the tool sequence
    tool_sequence   TEXT NOT NULL,       -- JSON array of [tool_name, args_hash] tuples
    embedding       TEXT,                -- JSON array of floats (embedded sequence)
    occurrence_count INTEGER NOT NULL DEFAULT 1,
    first_seen_at   TEXT DEFAULT (datetime('now')),
    last_seen_at    TEXT DEFAULT (datetime('now')),
    metadata        TEXT,                -- JSON blob (topic, project hints)
    UNIQUE(chain_hash, session_id)
);

CREATE INDEX IF NOT EXISTS idx_tool_chains_hash ON tool_chains (chain_hash);
CREATE INDEX IF NOT EXISTS idx_tool_chains_session ON tool_chains (session_id);

-- Skill proposals generated from clustered patterns
CREATE TABLE IF NOT EXISTS skill_proposals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,           -- Proposed skill name (slug)
    title           TEXT NOT NULL,           -- Human-readable title
    description     TEXT NOT NULL,
    triggers        TEXT NOT NULL,           -- JSON array of trigger phrases
    steps           TEXT NOT NULL,           -- JSON array of skill steps
    source_chains   TEXT NOT NULL,           -- JSON array of chain_hash values
    cluster_size    INTEGER NOT NULL DEFAULT 1,
    similarity_score REAL NOT NULL DEFAULT 0.0,
    status          TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected', 'expired')),
    approved_at     TEXT,
    rejected_at     TEXT,
    expires_at      TEXT,
    created_at      TEXT DEFAULT (datetime('now')),
    updated_at      TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_skill_proposals_status ON skill_proposals (status);
CREATE INDEX IF NOT EXISTS idx_skill_proposals_name ON skill_proposals (name);

-- Track which local skills were generated from this plugin
CREATE TABLE IF NOT EXISTS generated_skills (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    proposal_id     INTEGER NOT NULL REFERENCES skill_proposals(id),
    skill_path      TEXT NOT NULL,           -- Path to the generated skill JSON
    skill_name      TEXT NOT NULL,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_generated_skills_proposal ON generated_skills (proposal_id);
