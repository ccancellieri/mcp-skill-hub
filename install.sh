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

# 2. Check Ollama
echo "[2/4] Checking Ollama..."
if ! command -v ollama &>/dev/null; then
    echo "  ⚠ Ollama not found. Install from https://ollama.ai then run:"
    echo "    ollama pull nomic-embed-text"
else
    if ! ollama list 2>/dev/null | grep -q "nomic-embed-text"; then
        echo "  Pulling nomic-embed-text (274 MB)..."
        ollama pull nomic-embed-text
    else
        echo "  nomic-embed-text already available."
    fi
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

# 4. Summary
echo "[4/4] Done!"
echo ""
echo "Next steps:"
echo "  1. Restart Claude Code"
echo "  2. In Claude Code, run:  index_skills()  then  index_plugins()"
echo "  3. Try:  search_skills(\"your task description\")"
echo "  4. Teach it:  teach(rule=\"when I give a URL\", suggest=\"chrome-devtools-mcp\")"
echo ""
echo "Optional: pull deepseek-r1:1.5b for smarter re-ranking:"
echo "  ollama pull deepseek-r1:1.5b"
