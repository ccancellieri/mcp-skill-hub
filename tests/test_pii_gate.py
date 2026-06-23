"""M1 — PII gate: regex scan before save_task / teach when repo is marked public."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from skill_hub.pii_gate import (
    GateResult,
    POLICY_REL,
    check,
    is_public,
    log_override,
    scan,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _mkpolicy(repo_root: Path, content: str) -> Path:
    p = repo_root / POLICY_REL
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# scan() — pattern matching
# ---------------------------------------------------------------------------


def test_scan_catches_anthropic_key():
    hits = scan("API: sk-ant-api03-AbCdEfGhIjKlMnOpQrStUv")
    assert any(h.pattern == "sk_ant_key" for h in hits)


def test_scan_catches_github_token():
    hits = scan("token=ghp_abcdefghijklmnopqrstuvwxyz12")
    assert any(h.pattern == "github_token" for h in hits)


def test_scan_catches_github_fine_grained_token():
    hits = scan("token=github_pat_11AAAAAA0abcdefghijklmnop")
    assert any(h.pattern == "github_token" for h in hits)


def test_scan_catches_ipv4():
    hits = scan("the server is at 192.168.1.42 right now")
    assert any(h.pattern == "ipv4" and h.match == "192.168.1.42" for h in hits)


def test_scan_catches_public_ipv4():
    hits = scan("nslookup says 8.8.8.8")
    assert any(h.pattern == "ipv4" for h in hits)


def test_scan_catches_cloud_run_revision():
    hits = scan("Revision: api-prod-00042-abc rolled out")
    assert any(h.pattern == "cloud_run_revision" for h in hits)


def test_scan_catches_gcp_project_id():
    hits = scan("Use project my-project-12345 for billing.")
    labels = {h.pattern for h in hits}
    assert "gcp_project_id" in labels


def test_scan_catches_email():
    hits = scan("reach me at ccancellieri@hotmail.com any time")
    assert any(h.pattern == "email" and h.match == "ccancellieri@hotmail.com"
               for h in hits)


def test_scan_catches_phone():
    # The exact format that leaked via a public career-profile reference.
    hits = scan("Phone: +39 338 200 3690")
    assert any(h.pattern == "phone" for h in hits)


def test_scan_phone_no_false_positive_on_short_numbers():
    # Years, small counts, and dotted version strings must not match phone.
    hits = scan("Released in 2026 after 3 fixes; build 1.2.3.4 today.")
    assert not any(h.pattern == "phone" for h in hits)


def test_scan_clean_text_no_hits():
    """Ordinary prose, kebab-case slugs without digits, and version strings
    must not trigger the gate — those are the high-volume false positives.
    """
    hits = scan(
        "Refactor the catalog pipeline. See feedback-no-intermodule-deps.md. "
        "Version 1.2.3-rc.1 ships next week."
    )
    assert hits == [], f"unexpected hits on clean text: {hits}"


def test_scan_semver_with_more_than_four_octets_no_match():
    # Semver "1.2.3.4.5" must not match — the trailing ".5" extends the
    # candidate past the dotted-quad shape via the `(?![\w.])` lookahead.
    hits = scan("build 1.2.3.4.5 ships next quarter")
    assert not any(h.pattern == "ipv4" for h in hits)


def test_scan_overrides_suppress_hit():
    overrides = ["10.0.0.1"]
    hits = scan("server at 10.0.0.1", overrides=overrides)
    assert hits == []


def test_scan_overrides_only_match_exact():
    overrides = ["10.0.0.1"]
    hits = scan("but 10.0.0.2 still leaks", overrides=overrides)
    assert any(h.match == "10.0.0.2" for h in hits)


def test_scan_empty_input():
    assert scan("") == []
    assert scan(None) == []  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _load_policy / is_public
# ---------------------------------------------------------------------------


def test_is_public_true(tmp_path):
    _mkpolicy(tmp_path, "public: true\n")
    assert is_public(tmp_path) is True


def test_is_public_false_when_flag_missing(tmp_path):
    _mkpolicy(tmp_path, "other: value\n")
    assert is_public(tmp_path) is False


def test_is_public_false_when_policy_missing(tmp_path):
    assert is_public(tmp_path) is False


def test_is_public_false_when_explicit_false(tmp_path):
    _mkpolicy(tmp_path, "public: false\n")
    assert is_public(tmp_path) is False


def test_malformed_yaml_treated_as_private(tmp_path):
    _mkpolicy(tmp_path, "public: [unclosed\n")
    assert is_public(tmp_path) is False


# ---------------------------------------------------------------------------
# check() — the gate
# ---------------------------------------------------------------------------


def test_check_noop_without_repo_root():
    r = check("server at 10.0.0.1", None)
    assert r.allowed is True
    assert r.reason == "no repo context"


def test_check_noop_when_repo_is_private(tmp_path):
    # No policy.yml at all → private by default.
    r = check("server at 10.0.0.1 sk-ant-api03-AbCdEfGhIjKlMnOp", tmp_path)
    assert r.allowed is True
    assert r.public is False


def test_check_blocks_pii_in_public_repo(tmp_path):
    _mkpolicy(tmp_path, "public: true\n")
    r = check("ssh root@192.168.1.42", tmp_path)
    assert r.allowed is False
    assert r.public is True
    assert r.hits, "must report offending substrings"
    assert any("192.168.1.42" in h.match for h in r.hits)


def test_check_blocks_leaked_pii_in_public_repo(tmp_path):
    """Regression for the actual leak: phone + email + internal id in a
    career-profile reference reaching a public repo must be blocked."""
    _mkpolicy(tmp_path, "public: true\n")
    r = check(
        "Phone: +39 338 200 3690\nEmail: ccancellieri@hotmail.com\n",
        tmp_path,
    )
    assert r.allowed is False
    labels = {h.pattern for h in r.hits}
    assert "phone" in labels
    assert "email" in labels


def test_check_allows_clean_content_in_public_repo(tmp_path):
    _mkpolicy(tmp_path, "public: true\n")
    r = check("Just a normal task summary about refactoring.", tmp_path)
    assert r.allowed is True
    assert r.hits == []


def test_check_override_allows_with_log(tmp_path):
    _mkpolicy(tmp_path, "public: true\n")
    r = check("server at 192.168.1.42", tmp_path, override=True)
    assert r.allowed is True
    assert r.override_used is True
    assert r.hits, "override must still surface what was matched"


def test_check_respects_pii_overrides_yaml(tmp_path):
    _mkpolicy(
        tmp_path,
        "public: true\npii_overrides:\n  - '10.0.0.1'\n",
    )
    r = check("server at 10.0.0.1", tmp_path)
    assert r.allowed is True
    assert r.hits == []


def test_check_format_block_message_lists_hits(tmp_path):
    _mkpolicy(tmp_path, "public: true\n")
    r = check("server at 192.168.1.42 with ghp_aaaaaaaaaaaaaaaaaaaa12", tmp_path)
    msg = r.format_block_message()
    assert "Refused" in msg
    assert "192.168.1.42" in msg
    assert "ghp_" in msg
    assert ".skill-hub/policy.yml" in msg


def test_check_repo_root_not_a_directory(tmp_path):
    bogus = tmp_path / "does-not-exist"
    r = check("anything", bogus)
    assert r.allowed is True


# ---------------------------------------------------------------------------
# log_override()
# ---------------------------------------------------------------------------


def test_log_override_writes_per_repo_log(tmp_path):
    from skill_hub.pii_gate import Hit
    hits = [
        Hit(pattern="ipv4", match="192.168.1.42", start=0, end=12),
        Hit(pattern="github_token", match="ghp_xxx", start=20, end=27),
    ]
    log_path = log_override(tmp_path, "save_task", hits)
    text = log_path.read_text(encoding="utf-8")
    assert "save_task" in text
    assert "ipv4" in text
    assert "github_token" in text
    # The matched substrings must NOT be in the log — only the labels.
    assert "192.168.1.42" not in text
    assert "ghp_xxx" not in text


def test_log_override_appends(tmp_path):
    from skill_hub.pii_gate import Hit
    hit = [Hit(pattern="ipv4", match="x", start=0, end=1)]
    log_override(tmp_path, "teach", hit)
    log_override(tmp_path, "save_task", hit)
    log_path = tmp_path / ".skill-hub" / "pii_overrides.log"
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert "teach" in lines[0]
    assert "save_task" in lines[1]


# ---------------------------------------------------------------------------
# Integration with server-level helpers
# ---------------------------------------------------------------------------


def test_resolve_repo_root_walks_up_to_git(tmp_path):
    """resolve_repo_root walks up cwd to the nearest .git dir."""
    from skill_hub.pii_gate import resolve_repo_root
    (tmp_path / ".git").mkdir()
    sub = tmp_path / "src" / "deep"
    sub.mkdir(parents=True)
    root = resolve_repo_root(cwd=str(sub))
    assert root == tmp_path.resolve()


def test_resolve_repo_root_returns_none_when_no_git(tmp_path):
    from skill_hub.pii_gate import resolve_repo_root
    sub = tmp_path / "dir"
    sub.mkdir()
    root = resolve_repo_root(cwd=str(sub))
    assert root is None


def test_enforce_blocks_save_task_message(tmp_path):
    """enforce() returns (False, refusal) for PII in a public repo."""
    from skill_hub.pii_gate import enforce
    (tmp_path / ".git").mkdir()
    _mkpolicy(tmp_path, "public: true\n")
    allowed, msg = enforce(
        tool="save_task",
        content="ssh root@192.168.1.42",
        cwd=str(tmp_path),
    )
    assert allowed is False
    assert "192.168.1.42" in msg


def test_enforce_allows_clean_content(tmp_path):
    from skill_hub.pii_gate import enforce
    (tmp_path / ".git").mkdir()
    _mkpolicy(tmp_path, "public: true\n")
    allowed, msg = enforce(
        tool="save_task",
        content="ordinary task content",
        cwd=str(tmp_path),
    )
    assert allowed is True
    assert msg == ""


def test_enforce_override_logs(tmp_path):
    from skill_hub.pii_gate import enforce
    (tmp_path / ".git").mkdir()
    _mkpolicy(tmp_path, "public: true\n")
    allowed, msg = enforce(
        tool="save_task",
        content="ssh root@192.168.1.42",
        cwd=str(tmp_path),
        override=True,
    )
    assert allowed is True
    log = (tmp_path / ".skill-hub" / "pii_overrides.log")
    assert log.exists()
    txt = log.read_text(encoding="utf-8")
    assert "save_task" in txt
    assert "ipv4" in txt


def test_enforce_private_repo_passthrough(tmp_path):
    """No policy.yml = private repo = gate is a no-op even with obvious PII."""
    from skill_hub.pii_gate import enforce
    (tmp_path / ".git").mkdir()
    allowed, msg = enforce(
        tool="save_task",
        content="ssh root@192.168.1.42 sk-ant-api03-aaaaaaaaaaaaaaaaaa",
        cwd=str(tmp_path),
    )
    assert allowed is True
    assert msg == ""
