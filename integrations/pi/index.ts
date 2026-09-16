import { createPiExtension } from "@mcp-skill-hub/context-adapter";

export default function skillHubContext(pi: unknown) {
  createPiExtension(pi as { on: (event: string, handler: unknown) => void }, {
    pythonCommand: process.env.SKILL_HUB_CONTEXT_PYTHON || "python3",
    contextCommand: process.env.SKILL_HUB_CONTEXT_COMMAND,
  });
}
