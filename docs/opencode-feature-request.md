# Feature Request: `prompt.submit.before` Hook Event

## Summary

Add a new plugin hook event `prompt.submit.before` that allows plugins to inspect, modify, or block user prompts **before** they reach the LLM.

## Motivation

Currently, opencode provides excellent tool hooks (`tool.execute.before`/`after`) and session events, but there's no way to intercept user messages before they're sent to the LLM. This capability is essential for:

1. **Zero-token command interception** - Handle simple commands locally without spending API tokens
2. **Prompt enrichment** - Inject context, RAG results, or hints into prompts
3. **Intent classification** - Route prompts to different models or agents based on complexity
4. **Prompt modification** - Sanitize, format, or augment user input
5. **Local-first features** - Answer simple queries locally, forward complex ones

This is the key feature that enables the skill-hub's zero-token task interception in Claude Code (via `UserPromptSubmit` hook).

## Proposed API

### Event Name
`prompt.submit.before`

### Hook Input
```typescript
interface PromptSubmitInput {
  prompt: string           // The user's message
  sessionId: string        // Current session ID
  agentName?: string       // Target agent (if specified)
  attachments?: Array<{    // Any attached files/images
    type: "image" | "file"
    path?: string
    data?: string
  }>
}
```

### Hook Output
```typescript
interface PromptSubmitOutput {
  // Decision: allow the prompt through, or block it
  decision: "allow" | "block"
  
  // If blocking, show this message to user (required when decision="block")
  message?: string
  
  // If allowing, optionally inject a system message (invisible to user)
  systemMessage?: string
  
  // If allowing, optionally modify the user's prompt
  modifiedPrompt?: string
  
  // If allowing, optionally attach additional context
  additionalContext?: string
}
```

### Plugin Example

```typescript
import type { Plugin } from "@opencode-ai/plugin"

export const PromptInterceptorPlugin: Plugin = async ({ client }) => {
  return {
    "prompt.submit.before": async (input, output) => {
      const { prompt } = input
      
      // Example 1: Local command handling (zero tokens)
      if (/^(list|show|status|help)$/i.test(prompt.trim())) {
        output.decision = "block"
        output.message = await handleLocalCommand(prompt)
        return
      }
      
      // Example 2: Context injection (RAG)
      const relevantSkills = await searchSkills(prompt)
      if (relevantSkills.length > 0) {
        output.decision = "allow"
        output.systemMessage = `Relevant context:\n${relevantSkills.join("\n")}`
        return
      }
      
      // Example 3: Prompt enrichment
      if (prompt.includes("refactor")) {
        output.decision = "allow"
        output.systemMessage = "[Model hint: Consider using Sonnet + high effort for refactoring]"
        return
      }
      
      // Default: pass through unchanged
      output.decision = "allow"
    },
  }
}
```

## Use Cases

### 1. Skill Hub Integration (Primary)

The mcp-skill-hub uses this pattern in Claude Code to intercept task commands:

```
User: "save this to memory and close"
  → Hook detects task command
  → Executes locally via skill-hub-cli
  → Returns result directly to user
  → 0 Claude tokens used
```

Without this hook, every interaction goes to the LLM, costing tokens even for simple commands.

### 2. Complexity-Based Model Routing

```typescript
if (isSimple(prompt)) {
  output.decision = "allow"
  output.systemMessage = "[Route to: haiku]"
} else if (isComplex(prompt)) {
  output.decision = "allow"
  output.systemMessage = "[Route to: opus]"
}
```

### 3. Local-First Responses

```typescript
if (isGreeting(prompt)) {
  output.decision = "block"
  output.message = "Hello! How can I help you today?"
}
```

### 4. Context Injection

```typescript
const docs = await searchDocumentation(prompt)
if (docs) {
  output.decision = "allow"
  output.systemMessage = docs
}
```

## Implementation Notes

1. **Performance**: The hook should run quickly (< 100ms typical). Plugins can use `async: true` for longer operations.

2. **Multiple plugins**: Run all registered hooks in sequence. If any returns `decision: "block"`, stop and return that block message.

3. **Error handling**: Plugin errors should NOT block the prompt. Log and continue with `decision: "allow"`.

4. **Visibility**: `systemMessage` is injected into the LLM context but NOT shown to the user (like Claude's `systemMessage` in hooks).

5. **Backward compatibility**: Existing plugins work unchanged. This is a new event type.

## Comparison with Claude Code

Claude Code already has this capability via `UserPromptSubmit` hook in `settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": [{
      "hooks": [{
        "type": "command",
        "command": "/path/to/hook.sh",
        "timeout": 45
      }]
    }]
  }
}
```

The hook receives JSON on stdin with `prompt` and `session_id`, and can output:
- `{"continue": false, "stopReason": "..."}` to block
- `{"systemMessage": "..."}` to inject context

This proposal adapts that proven pattern to opencode's plugin architecture.

## Related

- MCP tools are great for LLM-initiated actions
- But they can't intercept user messages before the LLM sees them
- This hook fills that gap, enabling zero-token local execution

---

**Would the opencode team be open to this feature?** Happy to contribute implementation if the design is approved.
