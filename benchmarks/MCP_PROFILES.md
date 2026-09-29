# Minimal MCP profile measurement

The 2026-09-29 local candidate exposes 87 full-profile tools or eight minimal
profile tools. Four unshipped selector-learning endpoints were removed from
both profiles. The full profile remains the default.

## Method and result

Capture `tools/list` and server instructions through the MCP protocol, serialize
that pair once as compact JSON (`ensure_ascii=False`, separators `,` and `:`),
and count with `o200k_base`. Include output schemas and protocol metadata in
both profiles. This measures serialized payload size, not native client billing:
clients may cache, defer, strip or transform tool definitions.

| Profile | Tools | UTF-8 bytes | Serialized tokens |
| --- | ---: | ---: | ---: |
| Full | 87 | 69,243 | 16,362 |
| Minimal | 8 | 5,534 | 1,242 |

The reduction is 92.41%. Whole-task savings and real-client tool selection remain
unqualified; see issues #156, #129 and #152. Neither model backends nor live
client configurations were changed for this measurement.

## Reproduction

Use the test environment with FastMCP and the benchmark tokenizer installed.
The protocol probe creates an isolated database and disables full-profile
background services. No live memory or model endpoint is needed.

```sh
PYTHONPATH=tests:src python - <<'PY'
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import tiktoken
from test_mcp_profiles import _probe

encoding = tiktoken.get_encoding('o200k_base')
with TemporaryDirectory() as root:
    for profile in ('full', 'minimal'):
        result = _probe(Path(root) / profile, '--profile', profile)
        payload = json.dumps(
            {'tools': result['schemas'], 'instructions': result['instructions']},
            ensure_ascii=False, separators=(',', ':'),
        )
        print(profile, len(result['names']), len(payload.encode()),
              len(encoding.encode(payload)))
PY
```

Run `pytest tests/test_mcp_profiles.py` to verify excluded tools cannot be called,
manual-only composition, description-first retrieval, rejected model options,
invalid arguments and startup isolation. Exact counts depend on frozen code
and FastMCP versions; archive the protocol payload with every future comparison.
