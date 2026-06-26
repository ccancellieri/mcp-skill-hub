-- A7 — Plugin-scoped storage for wiki-derived skill tracking.
-- Tracks wiki_page -> skill_id mappings and generation metadata.

CREATE TABLE IF NOT EXISTS plugin_wiki_skills (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    wiki_slug     TEXT NOT NULL UNIQUE,      -- wiki page slug (source)
    wiki_title    TEXT NOT NULL,             -- wiki page title
    wiki_type     TEXT NOT NULL,             -- entity|concept|source|overview
    skill_path    TEXT NOT NULL,             -- relative path to generated SKILL.md
    skill_id      TEXT,                      -- skill ID from frontmatter
    access_count  INTEGER NOT NULL DEFAULT 0,-- snapshot of wiki access at generation
    generated_at  TEXT NOT NULL DEFAULT (datetime('now')),
    last_used     TEXT,                      -- last session where skill was suggested
    use_count     INTEGER NOT NULL DEFAULT 0 -- times skill was suggested
);

CREATE INDEX IF NOT EXISTS idx_plugin_wiki_skills_slug ON plugin_wiki_skills (wiki_slug);
CREATE INDEX IF NOT EXISTS idx_plugin_wiki_skills_type ON plugin_wiki_skills (wiki_type);
CREATE INDEX IF NOT EXISTS idx_plugin_wiki_skills_access ON plugin_wiki_skills (access_count DESC);
