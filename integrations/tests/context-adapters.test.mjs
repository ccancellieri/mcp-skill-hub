import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { PassThrough } from "node:stream";
import test from "node:test";

import {
  createPiExtension,
  registerOpenClawContext,
  runContextCli,
} from "../shared/context-adapter.mjs";

function successfulSpawn(response, calls) {
  return (command, args, options) => {
    calls.push({ command, args, options });
    const child = new EventEmitter();
    child.stdin = new PassThrough();
    child.stdout = new PassThrough();
    child.kill = () => child.emit("close", null, "SIGTERM");
    process.nextTick(() => {
      child.stdout.end(JSON.stringify(response));
      child.emit("close", 0);
    });
    return child;
  };
}

test("runner spawns configured Python without a shell and parses context", async () => {
  const calls = [];
  const result = await runContextCli(
    { prompt: "find cache policy", cwd: "/repo", session_id: "s-1", task_id: 7 },
    { pythonCommand: "/opt/python", spawn: successfulSpawn({ context: "cached policy" }, calls) },
  );

  assert.deepEqual(result, { context: "cached policy" });
  assert.equal(calls[0].command, "/opt/python");
  assert.deepEqual(calls[0].args, ["-m", "skill_hub.context_cli"]);
  assert.equal(calls[0].options.shell, false);
  assert.equal(calls[0].options.maxBuffer, 128 * 1024);
});

test("runner selects adapter-reported CLI provenance outside the JSON payload", async () => {
  const calls = [];
  await runContextCli(
    { prompt: "find cache policy", cwd: "/repo", session_id: "s-1" },
    {
      adapterSource: "pi",
      pythonCommand: "/opt/python",
      spawn: successfulSpawn({ context: "" }, calls),
    },
  );

  assert.deepEqual(calls[0].args, [
    "-m", "skill_hub.context_cli", "--adapter-source", "pi",
  ]);
});

test("runner kills a process that exceeds the two-second budget", async () => {
  let killed = false;
  const result = await runContextCli(
    { prompt: "slow lookup", cwd: "/repo", session_id: "s-1" },
    {
      spawn: () => {
        const child = new EventEmitter();
        child.stdin = new PassThrough();
        child.stdout = new PassThrough();
        child.kill = () => { killed = true; };
        return child;
      },
    },
  );

  assert.equal(result, null);
  assert.equal(killed, true);
});

test("runner resolves without context when stdin emits EPIPE", async () => {
  const result = await runContextCli(
    { prompt: "broken pipe", cwd: "/repo", session_id: "s-1" },
    {
      spawn: () => {
        const child = new EventEmitter();
        child.stdin = new PassThrough();
        child.stdout = new PassThrough();
        child.kill = () => {};
        process.nextTick(() => child.stdin.emit("error", new Error("EPIPE")));
        return child;
      },
    },
  );
  assert.equal(result, null);
});

test("Pi adapter injects a custom context message without rewriting system prompt", async () => {
  const handlers = new Map();
  const pi = { on(event, callback) { handlers.set(event, callback); } };
  createPiExtension(pi, { runContext: async () => ({ context: "project evidence" }) });

  const output = await handlers.get("before_agent_start")(
    { prompt: "implement it" },
    { cwd: "/repo", sessionManager: { getSessionId: () => "pi-session" } },
  );

  assert.deepEqual(output, {
    message: { customType: "skill-hub-context", content: "project evidence", display: false },
  });
  assert.equal("systemPrompt" in output, false);
});

test("Pi adapter reports native model and thinking level for its session", async () => {
  const handlers = new Map();
  const pi = { on(event, callback) { handlers.set(event, callback); } };
  let payload;
  createPiExtension(pi, { runContext: async (value) => { payload = value; return {}; } });

  await handlers.get("before_agent_start")(
    { prompt: "implement it", thinkingLevel: "high" },
    {
      cwd: "/repo",
      model: { id: "claude-sonnet", provider: "anthropic", name: "Claude Sonnet" },
      sessionManager: { getSessionId: () => "pi-session" },
    },
  );

  assert.deepEqual(payload.runtime, {
    client: { id: "pi" },
    model: { id: "claude-sonnet", provider: "anthropic", display_name: "Claude Sonnet" },
    effort: { value: "high", scheme: "thinking_level" },
    session: { id: "pi-session" },
    provenance: {
      client_id: "native_event", model_id: "native_event",
      model_provider: "native_event", model_display_name: "native_event",
      effort_value: "native_event", effort_scheme: "native_event",
      session_id: "native_event",
    },
  });
});

test("Pi context filtering keeps only the newest injected context without mutating history", () => {
  const handlers = new Map();
  const pi = { on(event, callback) { handlers.set(event, callback); } };
  createPiExtension(pi);

  const messages = [
    { role: "user", content: "keep me" },
    { customType: "skill-hub-context", content: "old context" },
    { customType: "another-extension", content: "keep this" },
    { customType: "skill-hub-context", content: "latest context" },
  ];
  const event = Object.freeze({ messages: Object.freeze(messages) });
  const output = handlers.get("context")(event, {});

  assert.deepEqual(output, {
    messages: [messages[0], messages[2], messages[3]],
  });
  assert.deepEqual(event.messages, messages);
});

test("host callbacks leave input untouched and turn context failures into no context", async () => {
  let piHandler;
  const pi = { on(_event, callback) { piHandler = callback; } };
  createPiExtension(pi, { runContext: async () => { throw new Error("offline"); } });
  const piEvent = Object.freeze({ prompt: "safe", messages: Object.freeze([{ content: "unchanged" }]) });
  assert.equal(await piHandler(piEvent, { cwd: "/repo" }), undefined);
  assert.equal(piEvent.messages[0].content, "unchanged");

  let openClawHandler;
  const api = { on(_event, callback) { openClawHandler = callback; } };
  registerOpenClawContext(api, { runContext: async () => { throw new Error("offline"); } });
  assert.equal(
    await openClawHandler(Object.freeze({ prompt: "safe" }), { workspaceDir: "/repo" }),
    undefined,
  );
});

test("OpenClaw adapter uses prependContext and never falls back to process cwd", async () => {
  let handler;
  const api = { on(event, callback) { assert.equal(event, "before_prompt_build"); handler = callback; } };
  let calls = 0;
  registerOpenClawContext(api, {
    runContext: async (payload) => {
      calls += 1;
      assert.equal(payload.cwd, "/verified-workspace");
      assert.equal(payload.task_id, null);
      return { context: "retrieved evidence" };
    },
  });

  const output = await handler(
    { currentUserMessage: "inspect this", prompt: "reconstructed history" },
    {
      workspaceDir: "/verified-workspace",
      sessionKey: "oc-session",
      hookInvocation: { assertActive() { calls += 1; } },
    },
  );
  assert.deepEqual(output, { prependContext: "retrieved evidence" });
  assert.equal(calls, 2);

  const missingCwd = await handler({ prompt: "no scope" }, { sessionKey: "oc-session" });
  assert.equal(missingCwd, undefined);
  assert.equal(calls, 2);
});

test("OpenClaw adapter prefers an absolute configured project root", async () => {
  let handler;
  const api = { on(_event, callback) { handler = callback; } };
  registerOpenClawContext(api, {
    projectRoot: "/configured-project",
    runContext: async (payload) => ({ context: payload.cwd }),
  });

  const output = await handler(
    { prompt: "scope it" },
    { workspaceDir: "/other-project", sessionKey: "oc-session" },
  );
  assert.deepEqual(output, { prependContext: "/configured-project" });
});

test("OpenClaw reports only documented prompt-hook identity fields", async () => {
  let handler;
  const api = { on(_event, callback) { handler = callback; } };
  let payload;
  registerOpenClawContext(api, {
    runContext: async (value) => { payload = value; return {}; },
  });

  await handler(
    { prompt: "scope it", model: "must-not-be-trusted" },
    {
      workspaceDir: "/repo", sessionKey: "routing-key", sessionId: "oc-session-2",
      runId: "run-7", agentId: "main", modelId: "gpt-5", modelProviderId: "openai",
    },
  );

  assert.equal(payload.runtime.client.id, "openclaw");
  assert.equal(payload.session_id, "routing-key");
  assert.equal(payload.runtime.session.id, "oc-session-2");
  assert.equal(payload.runtime.session.turn_id, "run-7");
  assert.deepEqual(payload.runtime.model, { id: "gpt-5", provider: "openai" });
});

test("OpenClaw routing key alone is not persisted as native session identity", async () => {
  let handler;
  const api = { on(_event, callback) { handler = callback; } };
  let payload;
  registerOpenClawContext(api, {
    runContext: async (value) => { payload = value; return {}; },
  });

  await handler(
    { prompt: "scope it" },
    { workspaceDir: "/repo", sessionKey: "stable-routing-key", modelId: "gpt-5" },
  );

  assert.equal(payload.session_id, "stable-routing-key");
  assert.equal("id" in payload.runtime.session, false);
});
