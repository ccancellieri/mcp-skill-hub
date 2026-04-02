#!/bin/bash
set -e

# MCP Skill Hub — one-command installer for Claude Code
# Usage: ./install.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MCP_JSON="$HOME/.mcp.json"
SETTINGS="$HOME/.claude/settings.json"

echo "=== MCP Skill Hub Installer ==="

# 1. Create venv and install
echo "[1/4] Installing Python package..."
cd "$SCRIPT_DIR"
python3 -m venv .venv
.venv/bin/pip install -e . -q

# 2. Check Ollama and pull models
echo "[2/4] Checking Ollama and models..."
if ! command -v ollama &>/dev/null; then
    echo "  ⚠ Ollama not found. Install from https://ollama.ai then run:"
    echo "    ollama pull nomic-embed-text"
else
    INSTALLED=$(ollama list 2>/dev/null || echo "")

    # Required: embedding model
    if echo "$INSTALLED" | grep -q "nomic-embed-text"; then
        echo "  nomic-embed-text already available."
    else
        echo "  Pulling nomic-embed-text (274 MB, required for embeddings)..."
        ollama pull nomic-embed-text
    fi

    # Detect RAM and recommend models
    RAM_GB=0
    if [ "$(uname)" = "Darwin" ]; then
        RAM_GB=$(($(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824))
    elif [ -f /proc/meminfo ]; then
        RAM_GB=$(($(grep MemTotal /proc/meminfo | awk '{print $2}') / 1048576))
    fi

    if [ "$RAM_GB" -gt 0 ]; then
        echo "  Detected RAM: ${RAM_GB} GB"
    else
        RAM_GB=16
        echo "  Could not detect RAM, assuming 16 GB"
    fi

    # Select models by RAM tier
    if [ "$RAM_GB" -ge 64 ]; then
        REASON_MODEL="qwen2.5-coder:14b"
        L4_MODEL="qwen2.5-coder:32b"
        echo "  Tier: 64GB+ — maximum quality"
    elif [ "$RAM_GB" -ge 32 ]; then
        REASON_MODEL="qwen2.5-coder:7b-instruct-q4_k_m"
        L4_MODEL="qwen2.5-coder:14b"
        echo "  Tier: 32GB — high quality"
    elif [ "$RAM_GB" -ge 16 ]; then
        REASON_MODEL="qwen2.5-coder:7b-instruct-q4_k_m"
        L4_MODEL="qwen2.5-coder:7b-instruct-q4_k_m"
        echo "  Tier: 16GB — recommended"
    else
        REASON_MODEL="deepseek-r1:1.5b"
        L4_MODEL="qwen2.5-coder:3b"
        echo "  Tier: 8GB — minimal"
    fi

    echo "  Recommended: reason=$REASON_MODEL  L4=$L4_MODEL"

    # Pull models that aren't installed yet
    for MODEL in $REASON_MODEL $L4_MODEL; do
        if ! echo "$INSTALLED" | grep -q "$MODEL"; then
            echo "  Pulling $MODEL..."
            ollama pull "$MODEL"
        fi
    done

    # Write config
    python3 -c "
import json, os
cfg_path = os.path.expanduser('~/.claude/mcp-skill-hub/config.json')
os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
cfg = {}
if os.path.exists(cfg_path):
    with open(cfg_path) as f:
        cfg = json.load(f)
cfg['reason_model'] = '$REASON_MODEL'
cfg['local_models'] = {
    'level_1': 'qwen2.5-coder:3b',
    'level_2': 'qwen2.5-coder:7b-instruct-q4_k_m' if $RAM_GB >= 16 else 'qwen2.5-coder:3b',
    'level_3': '$REASON_MODEL',
    'level_4': '$L4_MODEL',
}
with open(cfg_path, 'w') as f:
    json.dump(cfg, f, indent=2)
print(f'  Config updated: reason_model=$REASON_MODEL')
"
fi

# 3. Register MCP server in ~/.mcp.json
echo "[3/4] Registering MCP server..."
SKILL_HUB_BIN="$SCRIPT_DIR/.venv/bin/skill-hub"

if [ -f "$MCP_JSON" ]; then
    # Check if already registered
    if python3 -c "import json; d=json.load(open('$MCP_JSON')); exit(0 if 'skill-hub' in d.get('mcpServers',{}) else 1)" 2>/dev/null; then
        echo "  Already registered in $MCP_JSON"
    else
        # Add to existing file
        python3 -c "
import json
with open('$MCP_JSON') as f:
    d = json.load(f)
d.setdefault('mcpServers', {})['skill-hub'] = {
    'type': 'stdio',
    'command': '$SKILL_HUB_BIN'
}
with open('$MCP_JSON', 'w') as f:
    json.dump(d, f, indent=2)
print('  Added skill-hub to $MCP_JSON')
"
    fi
else
    cat > "$MCP_JSON" << EOF
{
  "mcpServers": {
    "skill-hub": {
      "type": "stdio",
      "command": "$SKILL_HUB_BIN"
    }
  }
}
EOF
    echo "  Created $MCP_JSON"
fi

# 4. Install hooks
echo "[4/5] Installing hooks..."
HOOKS_DIR="$SCRIPT_DIR/hooks"
chmod +x "$HOOKS_DIR"/*.sh 2>/dev/null

# Merge hooks into settings.json (idempotent — skips if already present)
python3 -c "
import json, os, sys

settings_path = os.path.expanduser('$SETTINGS')
hooks_dir = '$HOOKS_DIR'

# Define the hooks we need registered
required_hooks = [
    {
        'type': 'command',
        'command': f'{hooks_dir}/session-start-enforcer.sh',
        'timeout': 5,
        'statusMessage': 'Checking session start protocol...'
    },
    {
        'type': 'command',
        'command': f'{hooks_dir}/intercept-task-commands.sh',
        'timeout': 45,
        'statusMessage': 'Checking for task commands...'
    },
]
stop_hooks = [
    {
        'type': 'command',
        'command': f'{hooks_dir}/session-end.sh',
        'timeout': 45,
        'statusMessage': 'Saving session memory...'
    },
]

# Load or create settings
if os.path.exists(settings_path):
    with open(settings_path) as f:
        settings = json.load(f)
else:
    os.makedirs(os.path.dirname(settings_path), exist_ok=True)
    settings = {}

hooks = settings.setdefault('hooks', {})
existing_json = json.dumps(hooks)

changed = False

# UserPromptSubmit — merge required hooks (order matters: enforcer before interceptor)
ups = hooks.setdefault('UserPromptSubmit', [{'hooks': []}])
existing_cmds = {h.get('command', '') for entry in ups for h in entry.get('hooks', [])}
for hook_def in required_hooks:
    if hook_def['command'] not in existing_cmds:
        ups[0].setdefault('hooks', []).append(hook_def)
        changed = True
        print(f'  + Added {os.path.basename(hook_def[\"command\"])}')

# Stop — merge session-end hook
stop = hooks.setdefault('Stop', [{'hooks': []}])
existing_stop_cmds = {h.get('command', '') for entry in stop for h in entry.get('hooks', [])}
for hook_def in stop_hooks:
    if hook_def['command'] not in existing_stop_cmds:
        stop[0].setdefault('hooks', []).append(hook_def)
        changed = True
        print(f'  + Added {os.path.basename(hook_def[\"command\"])}')

if changed:
    with open(settings_path, 'w') as f:
        json.dump(settings, f, indent=2)
    print(f'  Updated {settings_path}')
else:
    print('  Hooks already registered.')
"

# 5. Summary
echo "[5/5] Done!"
echo ""
echo "Next steps:"
echo "  1. Restart Claude Code"
echo "  2. In Claude Code, run:  index_skills()  then  index_plugins()"
echo "  3. Try:  search_skills(\"your task description\")"
echo "  4. Teach it:  teach(rule=\"when I give a URL\", suggest=\"chrome-devtools-mcp\")"
echo ""
echo "Monitor hook activity:"
echo "  tail -f ~/.claude/mcp-skill-hub/logs/hook-debug.log"
echo ""
echo "For SearXNG web search (optional):"
echo "  python install.py --searxng"
echo ""
echo "For remote VPS Ollama (optional):"
echo "  python install.py --vps http://myserver:11434"
