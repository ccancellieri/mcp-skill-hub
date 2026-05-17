---
name: swarm-init
description: Initialize a multi-agent swarm with anti-drift configuration
argument-hint: "[--topology hierarchical|mesh|ring]"
allowed-tools: Bash(npx *) mcp__claude-flow__swarm_init mcp__claude-flow__swarm_status Agent
---
Initialize a hierarchical swarm for coordinated multi-agent work.

Via MCP: `mcp__claude-flow__swarm_init({ topology: "hierarchical", maxAgents: 8 })`
