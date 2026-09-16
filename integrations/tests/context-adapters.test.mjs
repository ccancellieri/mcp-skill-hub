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
