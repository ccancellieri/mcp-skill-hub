import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";
import { registerOpenClawContext } from "@mcp-skill-hub/context-adapter";

export default definePluginEntry({
  id: "skill-hub-context",
  name: "Skill Hub Context",
  description: "Adds deterministic project context before an agent turn.",
  register(api) {
    const config = api.pluginConfig || {};
    registerOpenClawContext(api, {
      pythonCommand: typeof config.pythonCommand === "string"
        ? config.pythonCommand : "python3",
      contextCommand: typeof config.contextCommand === "string"
        ? config.contextCommand : undefined,
      projectRoot: typeof config.projectRoot === "string"
        ? config.projectRoot : undefined,
    });
  },
});
