-- Cross-project knowledge bridge schema
-- Tracks relationships and sync state between projects

-- Projects registered for knowledge federation
CREATE TABLE IF NOT EXISTS cpkb_projects (
    name        TEXT PRIMARY KEY,
    path        TEXT NOT NULL,
    tags        TEXT,               -- JSON array of tag strings
    last_sync   TEXT,               -- ISO timestamp of last successful sync
    enabled     INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT DEFAULT (datetime('now'))
);

-- Cross-project entity mappings (source truth -> wiki representation)
CREATE TABLE IF NOT EXISTS cpkb_entities (
    id              TEXT PRIMARY KEY,
    slug            TEXT NOT NULL UNIQUE,
    title           TEXT NOT NULL,
    type            TEXT NOT NULL DEFAULT 'entity',
    source_project  TEXT NOT NULL,
    source_path     TEXT NOT NULL,
    wiki_slug       TEXT,           -- Corresponding wiki page slug
    hash            TEXT,           -- Content hash for change detection
    created_at      TEXT DEFAULT (datetime('now')),
    updated_at      TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (source_project) REFERENCES cpkb_projects(name)
);

CREATE INDEX IF NOT EXISTS idx_cpkb_entities_project ON cpkb_entities(source_project);
CREATE INDEX IF NOT EXISTS idx_cpkb_entities_type ON cpkb_entities(type);

-- Cross-project relationships (derived from shared tags or explicit links)
CREATE TABLE IF NOT EXISTS cpkb_relations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    from_project    TEXT NOT NULL,
    to_project      TEXT NOT NULL,
    relation_type   TEXT NOT NULL,  -- 'shared_tag', 'depends_on', 'integrates_with'
    tag             TEXT,           -- For shared_tag relations
    strength        REAL DEFAULT 1.0,
    created_at      TEXT DEFAULT (datetime('now')),
    UNIQUE(from_project, to_project, relation_type, tag),
    FOREIGN KEY (from_project) REFERENCES cpkb_projects(name),
    FOREIGN KEY (to_project) REFERENCES cpkb_projects(name)
);

CREATE INDEX IF NOT EXISTS idx_cpkb_relations_from ON cpkb_relations(from_project);
CREATE INDEX IF NOT EXISTS idx_cpkb_relations_to ON cpkb_relations(to_project);
CREATE INDEX IF NOT EXISTS idx_cpkb_relations_type ON cpkb_relations(relation_type);

-- Sync history log
CREATE TABLE IF NOT EXISTS cpkb_sync_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    project         TEXT NOT NULL,
    started_at      TEXT NOT NULL,
    completed_at    TEXT,
    status          TEXT NOT NULL DEFAULT 'pending',  -- pending, success, failed
    pages_created   INTEGER DEFAULT 0,
    pages_updated   INTEGER DEFAULT 0,
    errors          TEXT,           -- JSON array of error messages
    FOREIGN KEY (project) REFERENCES cpkb_projects(name)
);

CREATE INDEX IF NOT EXISTS idx_cpkb_sync_log_project ON cpkb_sync_log(project);
CREATE INDEX IF NOT EXISTS idx_cpkb_sync_log_status ON cpkb_sync_log(status);

-- Knowledge graph edge cache (precomputed for visualization)
CREATE TABLE IF NOT EXISTS cpkb_graph_edges (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source_node     TEXT NOT NULL,  -- node_id format: 'project:name' or 'entity:project:slug'
    target_node     TEXT NOT NULL,
    edge_type       TEXT NOT NULL,  -- 'contains', 'has_tag', 'related_to'
    weight          REAL DEFAULT 1.0,
    updated_at      TEXT DEFAULT (datetime('now')),
    UNIQUE(source_node, target_node, edge_type)
);

CREATE INDEX IF NOT EXISTS idx_cpkb_graph_edges_source ON cpkb_graph_edges(source_node);
CREATE INDEX IF NOT EXISTS idx_cpkb_graph_edges_target ON cpkb_graph_edges(target_node);
