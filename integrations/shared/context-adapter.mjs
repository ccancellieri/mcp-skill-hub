import { isAbsolute } from "node:path";
import { spawn as nodeSpawn } from "node:child_process";

const TIMEOUT_MS = 2_000;
const MAX_BUFFER = 128 * 1024;

function commandSpec(config) {
  if (typeof config.contextCommand === "string" && config.contextCommand) {
    return { command: config.contextCommand, args: [] };
  }
  return {
    command: config.pythonCommand || "python3",
    args: ["-m", "skill_hub.context_cli"],
  };
}

export function runContextCli(payload, config = {}) {
  const { command, args } = commandSpec(config);
  const spawn = config.spawn || nodeSpawn;

  return new Promise((resolve) => {
    let child;
    let stdout = "";
    let settled = false;
    const settle = (result = null) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve(result);
    };
    const timer = setTimeout(() => {
      child?.kill("SIGKILL");
      settle();
    }, TIMEOUT_MS);

    try {
      child = spawn(command, args, {
        shell: false,
        stdio: ["pipe", "pipe", "ignore"],
        maxBuffer: MAX_BUFFER,
      });
    } catch {
      settle();
      return;
    }

    child.stdout?.on("data", (chunk) => {
      stdout += chunk.toString();
      if (Buffer.byteLength(stdout) > MAX_BUFFER) {
        child.kill("SIGKILL");
        settle();
      }
    });
    child.on("error", () => settle());
    child.once("close", (code) => {
      if (code !== 0 || stdout.length > MAX_BUFFER) {
        settle();
        return;
      }
      try {
        const output = JSON.parse(stdout);
        settle(output && typeof output === "object" ? output : null);
      } catch {
        settle();
      }
    });
    child.stdin?.on("error", () => settle());
    try {
      child.stdin?.end(JSON.stringify(payload));
    } catch {
      child.kill("SIGKILL");
      settle();
    }
  });
}

function contextText(result) {
  return typeof result?.context === "string" && result.context.trim()
    ? result.context
    : "";
}

export function createPiExtension(pi, config = {}) {
  pi.on("context", (event) => {
    const messages = Array.isArray(event?.messages) ? event.messages : null;
    if (!messages) return;

    let newestContext = -1;
    for (let index = messages.length - 1; index >= 0; index -= 1) {
      if (messages[index]?.customType === "skill-hub-context") {
        newestContext = index;
        break;
      }
    }
    if (newestContext < 0) return;

    const filtered = messages.filter(
      (message, index) => message?.customType !== "skill-hub-context" || index === newestContext,
    );
    if (filtered.length !== messages.length) return { messages: filtered };
  });

  pi.on("before_agent_start", async (event, ctx) => {
    const prompt = typeof event?.prompt === "string" ? event.prompt : "";
    if (!prompt.trim()) return;
    const sessionId = ctx?.sessionManager?.getSessionId?.() || "";
    let result;
    try {
      result = await (config.runContext || runContextCli)({
        prompt,
        cwd: ctx?.cwd || "",
        session_id: sessionId,
        task_id: null,
      }, config);
    } catch {
      return;
    }
    const context = contextText(result);
    if (!context) return;
    return {
      message: {
        customType: "skill-hub-context",
        content: context,
        display: false,
      },
    };
  });
}

function openClawCwd(ctx, config) {
  if (typeof config.projectRoot === "string" && isAbsolute(config.projectRoot)) {
    return config.projectRoot;
  }
  if (typeof ctx?.workspaceDir === "string" && isAbsolute(ctx.workspaceDir)) {
    return ctx.workspaceDir;
  }
  return "";
}

export function registerOpenClawContext(api, config = {}) {
  api.on("before_prompt_build", async (event, ctx) => {
    const prompt = typeof event?.currentUserMessage === "string"
      ? event.currentUserMessage
      : typeof event?.prompt === "string" ? event.prompt : "";
    const cwd = openClawCwd(ctx, config);
    if (!prompt.trim() || !cwd) return;
    let result;
    try {
      result = await (config.runContext || runContextCli)({
        prompt,
        cwd,
        session_id: ctx?.sessionKey || "",
        task_id: null,
      }, config);
    } catch {
      return;
    }
    const context = contextText(result);
    if (!context) return;
    try {
      ctx?.hookInvocation?.assertActive?.();
    } catch {
      return;
    }
    return { prependContext: context };
  });
}
