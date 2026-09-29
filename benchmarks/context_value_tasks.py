"""Frozen synthetic-development tasks and isolated fixture materialization.

These are independent repositories shaped like recurring Skill Hub and
Tellurion problems. They are not results from either real project.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PY_TASKS = (
    ("preserve-prompt", "prompt_preserve", "Preserve a multiline prompt while appending evidence", "def enrich(prompt, evidence):\n    return evidence\n", "self.assertEqual(enrich('a\\nb', 'fact'), 'a\\nb\\n\\nfact')"),
    ("scope-rejection", "session_scope", "Reject memory from a mismatched canonical project root", "def scoped(records, root):\n    return list(records)\n", "self.assertEqual(scoped([('/a','own'),('/b','secret')], '/a'), [('/a','own')])"),
    ("paused-task", "task_pause", "Require an explicit identity before retrieving a paused task", "def include_task(paused, requested_id, task_id):\n    return True\n", "self.assertFalse(include_task(True, None, 7)); self.assertTrue(include_task(True, 7, 7))"),
    ("cached-usage", "native_usage", "Parse inclusive cached input without double counting", "def total_usage(input_tokens, cached, output, reasoning):\n    return input_tokens + cached + output + reasoning\n", "self.assertEqual(total_usage(100, 80, 20, 5), 125)"),
    ("provider-pin", "provider_pin", "Keep an explicit provider pin through capability refresh", "def resolve(explicit, available, fallback):\n    return fallback\n", "self.assertEqual(resolve('local-a', {'local-a'}, 'local-b'), 'local-a')"),
    ("fts-punctuation", "fts_escape", "Escape punctuation in deterministic FTS lookup", "def fts_term(value):\n    return value\n", "self.assertEqual(fts_term('cache:key'), '\"cache:key\"')"),
    ("digest-hash", "digest_refresh", "Skip digest replacement when the content hash is unchanged", "def should_refresh(old_hash, content):\n    return True\n", "self.assertFalse(should_refresh(hashlib.sha256(b'same').hexdigest(), 'same'))"),
    ("manifest-root", "plugin_manifest", "Reject a plugin component outside its manifest root", "def within_root(root, component):\n    return True\n", "self.assertFalse(within_root('/plugins/a', '/plugins/b/tool.py')); self.assertTrue(within_root('/plugins/a', '/plugins/a/tool.py'))"),
    ("event-tiebreak", "event_order", "Order equal-timestamp events by sequence", "def order_events(events):\n    return list(events)\n", "self.assertEqual([x['seq'] for x in order_events([{'time':1,'seq':2},{'time':1,'seq':1}])], [1,2])"),
    ("redact-secret", "secret_redact", "Redact a credential value while preserving its field name", "def redact(fields):\n    return dict(fields)\n", "self.assertEqual(redact({'token':'abc','host':'local'}), {'token':'[REDACTED]','host':'local'})"),
    ("vector-migrate", "vector_dimension", "Reject mixed vector dimensions until rebuild", "def accepts_dimension(active, incoming):\n    return True\n", "self.assertFalse(accepts_dimension(768, 1024)); self.assertTrue(accepts_dimension(768, 768))"),
    ("hook-timeout", "hook_deadline", "Return pass-through output after a bounded hook timeout", "def timeout_result(prompt):\n    return {'prompt': '', 'context': 'stale'}\n", "self.assertEqual(timeout_result('keep me'), {'prompt':'keep me','context':''})"),
)

RS_TASKS = (
    ("cache-key", "tile_cache", "Include source revision style filter and scope in a tile cache key", "pub fn value(a:&str,_b:&str,_c:&str,_d:&str)->String { a.to_string() }", 'assert_eq!(value("r1","s2","f3","tenant"), "r1|s2|f3|tenant");'),
    ("keyset-tie", "keyset_page", "Add a stable feature identifier tie breaker to keyset paging", "pub fn value(rows:Vec<(i32,&str)>)->Vec<(i32,&str)> { rows }", 'assert_eq!(value(vec![(1,"b"),(1,"a")]), vec![(1,"a"),(1,"b")]);'),
    ("unsupported-filter", "driver_capability", "Refuse an unsupported driver filter without scanning", "pub fn value(_supported:bool)->Result<(), &'static str> { Ok(()) }", 'assert_eq!(value(false), Err("unsupported filter"));'),
    ("cancel-batch", "stream_cancel", "Stop scheduling bounded stream batches after cancellation", "pub fn value(_cancelled:bool, queued:usize)->usize { queued + 1 }", 'assert_eq!(value(true, 3), 3); assert_eq!(value(false, 3), 4);'),
    ("range-limit", "range_read", "Reject a remote range response above the decoded-byte cap", "pub fn value(_size:usize, _cap:usize)->bool { true }", 'assert!(!value(11,10)); assert!(value(10,10));'),
    ("retry-incarnation", "job_retry", "Bind ingestion retry to an immutable resource incarnation", "pub fn value(_job:&str, _current:&str)->bool { true }", 'assert!(!value("v1","v2")); assert!(value("v2","v2"));'),
    ("style-invalidate", "style_revision", "Invalidate rendered tiles after a style revision changes", "pub fn value(_cached:u64, _current:u64)->bool { false }", 'assert!(value(1,2)); assert!(!value(2,2));'),
    ("axis-order", "crs_axis", "Apply declared wire CRS axis order at protocol adaptation", "pub fn value(x:f64,y:f64,_lat_first:bool)->(f64,f64) { (x,y) }", 'assert_eq!(value(12.0,45.0,true), (45.0,12.0)); assert_eq!(value(12.0,45.0,false),(12.0,45.0));'),
    ("open-interval", "temporal_extent", "Represent open temporal intervals without synthetic dates", "pub fn value(start:Option<i64>, end:Option<i64>)->(i64,i64) { (start.unwrap_or(0),end.unwrap_or(0)) }", 'assert_eq!(value(None,Some(9)), (None,Some(9)));'),
    ("projection-id", "feature_projection", "Preserve feature identifiers under property projection", "pub fn value(_id:&str, _fields:&[&str])->String { String::new() }", 'assert_eq!(value("feature-7", &["name"]), "feature-7");'),
    ("overload-reject", "overload_bound", "Reject overload before allocating an unbounded buffer", "pub fn value(_active:usize, _limit:usize)->Result<(), &'static str> { Ok(()) }", 'assert_eq!(value(10,10), Err("overloaded")); assert_eq!(value(9,10), Ok(()));'),
    ("schema-expand", "schema_expand", "Keep old readers compatible during an additive schema rollout", "pub fn value(name:&str, extra:Option<&str>)->String { format!(\"{}:{}\",name,extra.unwrap_or(\"\")) }", 'assert_eq!(value("old",None), "old"); assert_eq!(value("new",Some("x")), "new:x");'),
)


def build_task_specs() -> list[dict]:
    tasks = []
    for project, language, rows in (("skill-hub", "python", PY_TASKS), ("tellurion", "rust", RS_TASKS)):
        for slug, topic, objective, initial_source, assertion in rows:
            tasks.append({"id": f"{project}-{slug}", "project": project, "language": language,
                          "topic": topic, "objective": objective,
                          "test_argv": (["python3", "-m", "unittest", "-q", "test_task.py"] if language == "python" else ["cargo", "test", "-q"]),
                          "initial_source": initial_source, "assertion": assertion,
                          "fixture": "synthetic-development isolated repository generated by the harness",
                          "success": "baseline test fails, agent edits implementation, then the same test passes",
                          "source": "synthetic-development; independently written; no real project or GeoID material"})
    return tasks


def materialize_fixture(task: dict, root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    if task["language"] == "python":
        (root / "fixture.py").write_text("import hashlib\n\n" + task["initial_source"] + "\n")
        (root / "test_task.py").write_text("import hashlib\nimport unittest\nfrom fixture import *\n\nclass ContractTest(unittest.TestCase):\n    def test_contract(self):\n        " + task["assertion"] + "\n\nif __name__ == '__main__': unittest.main()\n")
        editable = ["fixture.py"]
    else:
        (root / "src").mkdir()
        (root / "tests").mkdir()
        (root / "Cargo.toml").write_text('[package]\nname = "context_value_fixture"\nversion = "0.1.0"\nedition = "2021"\n')
        (root / "src" / "lib.rs").write_text(task["initial_source"] + "\n")
        (root / "tests" / "contract.rs").write_text(
            "use context_value_fixture::*;\n\n#[test]\nfn contract() {\n    "
            + task["assertion"] + "\n}\n"
        )
        editable = ["src/lib.rs"]
    return {"editable_files": editable, "initial_snapshot": fixture_snapshot(root)}


def fixture_snapshot(root: Path) -> dict:
    hashes = {}
    excluded_dirs = {".git", "target", "__pycache__", ".pytest_cache"}
    excluded_files = {"Cargo.lock"}
    for path in sorted(file for file in root.rglob("*") if file.is_file()
                       and not (set(file.parts) & excluded_dirs)
                       and file.name not in excluded_files and file.suffix != ".pyc"):
        hashes[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    encoded = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    return {"files": hashes, "sha256": hashlib.sha256(encoded).hexdigest()}


def counterbalanced_slots(tasks: list[dict]) -> list[dict]:
    conditions = ("A_no_hub", "B_build_context", "C_context_composer")
    slots = []
    for task_index, task in enumerate(tasks):
        for repeat in (1, 2):
            offset = (task_index + repeat - 1) % len(conditions)
            order = conditions[offset:] + conditions[:offset]
            for position, condition in enumerate(order, 1):
                slots.append({"task_id": task["id"], "project": task["project"],
                              "repeat": repeat, "condition": condition,
                              "condition_role": {"A_no_hub": "baseline",
                                                 "B_build_context": "control",
                                                 "C_context_composer": "candidate"}[condition],
                              "order": position})
    return slots
