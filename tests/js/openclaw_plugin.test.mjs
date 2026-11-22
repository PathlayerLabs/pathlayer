// Tests of the reference OpenClaw plugin's `before_tool_call` gate, against a
// stub Pathlayer listener. Run with `node --test tests/js/openclaw_plugin.test.mjs`
// (tests/test_openclaw_plugin_js.py runs it under pytest when Node is available).
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { after, before, beforeEach, test } from "node:test";

// The plugin reads PATHLAYER_URL once, when it is imported: start the stub
// listener first, then import the plugin pointed at it.
let respond = (_payload, res) => res.end(JSON.stringify({ ok: true }));
const received = [];
const server = createServer((req, res) => {
  let body = "";
  req.on("data", (chunk) => {
    body += chunk;
  });
  req.on("end", () => {
    const payload = JSON.parse(body).payload;
    received.push(payload);
    res.setHeader("Content-Type", "application/json");
    respond(payload, res);
  });
});
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
process.env.PATHLAYER_URL = `http://127.0.0.1:${server.address().port}/ingest`;
// A Node process's first fetch also starts its HTTP client, and once on a busy
// CI runner that took longer than the gate's 500 ms budget: the first test's
// block was abandoned and the call allowed. In OpenClaw a run's lifecycle
// events are forwarded before its first tool call, so a decision query is
// never the process's first request. Warm the client up the same way.
await fetch(process.env.PATHLAYER_URL, {
  method: "POST",
  body: JSON.stringify({ payload: { hook: "warm-up" } }),
});
const { default: plugin, _internals } = await import(
  "../../pathlayer/adapters/openclaw_plugin/index.js"
);

const hooks = new Map();
const logs = [];
const api = {
  on(name, handler, opts) {
    hooks.set(name, { handler, opts });
  },
  logger: {
    info: (message) => logs.push(["info", message]),
    warn: (message) => logs.push(["warn", message]),
    error: (message) => logs.push(["error", message]),
    debug: (message) => logs.push(["debug", message]),
  },
};

before(() => plugin.register(api));
beforeEach(() => {
  received.length = 0;
  respond = (_payload, res) => res.end(JSON.stringify({ ok: true }));
});
after(() => {
  server.closeAllConnections();
  server.close();
});

const toolEvent = { toolName: "exec", toolCallId: "call-1", params: { command: "rm -rf build" } };
const ctx = { runId: "run-a", sessionKey: "s", agentId: "a" };
const gate = (event = toolEvent, context = ctx) => hooks.get("before_tool_call").handler(event, context);

test("a block decision blocks the tool call with its reason", async () => {
  respond = (_payload, res) =>
    res.end(JSON.stringify({ ok: true, decision: { block: true, blockReason: "stopped: loop" } }));
  assert.deepEqual(await gate(), { block: true, blockReason: "stopped: loop" });
});

test("a block decision without a usable reason still names Pathlayer", async () => {
  respond = (_payload, res) => res.end(JSON.stringify({ ok: true, decision: { block: true } }));
  assert.deepEqual(await gate(), { block: true, blockReason: "Pathlayer stopped this run." });
});

test("no decision, or anything but block: true, allows the tool call", async () => {
  for (const body of [{ ok: true }, { ok: true, decision: { block: "yes" } }, { decision: null }]) {
    respond = (_payload, res) => res.end(JSON.stringify(body));
    assert.equal(await gate(), undefined);
  }
});

test("an HTTP error or a body that is not JSON allows the tool call", async () => {
  respond = (_payload, res) => {
    res.statusCode = 500;
    res.end(JSON.stringify({ decision: { block: true } }));
  };
  assert.equal(await gate(), undefined);
  respond = (_payload, res) => res.end("<html>not json</html>");
  assert.equal(await gate(), undefined);
});

test("a dropped connection allows the tool call", async () => {
  respond = (_payload, res) => res.socket.destroy();
  assert.equal(await gate(), undefined);
});

test("a listener that never answers is abandoned before OpenClaw's hook timeout", async () => {
  respond = () => {}; // never answers
  const started = performance.now();
  assert.equal(await gate(), undefined);
  const elapsed = performance.now() - started;
  assert.ok(elapsed >= _internals.DECISION_TIMEOUT_MS - 50, `gave up too early: ${elapsed} ms`);
  assert.ok(elapsed < _internals.DECISION_HOOK_TIMEOUT_MS / 2, `waited too long: ${elapsed} ms`);
});

test("the gate's own budget stays far below the hook timeout that would block", () => {
  const { opts } = hooks.get("before_tool_call");
  assert.equal(opts.timeoutMs, _internals.DECISION_HOOK_TIMEOUT_MS);
  const worstCase = _internals.ORDERING_WAIT_MS + _internals.DECISION_TIMEOUT_MS;
  assert.ok(worstCase * 4 <= opts.timeoutMs, `${worstCase} ms against ${opts.timeoutMs} ms`);
});

test("the query carries identifiers and a fingerprint, never tool arguments", async () => {
  await gate();
  const digest = _internals.paramsDigest(toolEvent);
  assert.match(digest, /^[0-9a-f]{64}$/);
  assert.deepEqual(received, [
    {
      hook: "before_tool_call",
      event: { toolName: "exec", toolCallId: "call-1", paramsDigest: digest },
      ctx: { runId: "run-a", sessionKey: "s", agentId: "a" },
    },
  ]);
  assert.ok(!JSON.stringify(received).includes("rm -rf build"));
});

test("the fingerprint is equal for identical calls and only for them", () => {
  const digest = _internals.paramsDigest;
  const a = { toolName: "exec", toolCallId: "c1", params: { command: "ls", cwd: "/w" } };
  // Another call id and another key order: the same call.
  const same = { toolName: "exec", toolCallId: "c2", params: { cwd: "/w", command: "ls" } };
  assert.equal(digest(a), digest(same));
  assert.notEqual(digest(a), digest({ ...a, params: { command: "ls -a", cwd: "/w" } }));
  assert.notEqual(digest(a), digest({ ...a, toolName: "read" }));
  assert.equal(digest({ toolName: "x" }), digest({ toolName: "x", params: undefined }));
  const cycle = {};
  cycle.self = cycle;
  assert.equal(digest({ toolName: "x", params: cycle }), undefined);
  assert.equal(digest({ toolName: "x", params: { n: 1n } }), undefined);
  // Values JSON cannot hold are compared as JSON would send them.
  const day = (iso) => ({ toolName: "x", params: { at: new Date(iso) } });
  assert.notEqual(digest(day("2026-01-01")), digest(day("2026-01-02")));
  assert.equal(digest(day("2026-01-01")), digest({ toolName: "x", params: { at: "2026-01-01T00:00:00.000Z" } }));
});

test("the query waits until the run's in-flight forward is answered", async () => {
  // The listener answers a forward once it has processed the event, so a query
  // sent after that answer sees the STOP the event may have triggered.
  let forwardAnswered = false;
  let answeredWhenAsked;
  respond = (payload, res) => {
    if (payload.hook === "after_tool_call") {
      setTimeout(() => {
        forwardAnswered = true;
        res.end(JSON.stringify({ ok: true }));
      }, 100);
    } else {
      answeredWhenAsked = forwardAnswered;
      res.end(JSON.stringify({ ok: true }));
    }
  };
  hooks.get("after_tool_call").handler({ toolName: "read", result: "ok" }, ctx);
  await gate();
  assert.equal(answeredWhenAsked, true);
});

test("the wait for an in-flight forward is bounded", async () => {
  respond = (payload, res) => {
    if (payload.hook === "before_tool_call") {
      res.end(JSON.stringify({ ok: true }));
    }
    // the forward is never answered: it times out on its own
  };
  hooks.get("after_tool_call").handler({ toolName: "read" }, { ...ctx, runId: "run-slow" });
  const started = performance.now();
  assert.equal(await gate(toolEvent, { ...ctx, runId: "run-slow" }), undefined);
  assert.ok(performance.now() - started < _internals.ORDERING_WAIT_MS + 250);
});

test("a throwing logger never turns into a thrown gate", async () => {
  const warn = api.logger.warn;
  api.logger.warn = () => {
    throw new Error("logger down");
  };
  try {
    respond = (_payload, res) =>
      res.end(JSON.stringify({ ok: true, decision: { block: true, blockReason: "stop" } }));
    assert.deepEqual(await gate(), { block: true, blockReason: "stop" });
    respond = (_payload, res) => res.socket.destroy();
    assert.equal(await gate(), undefined);
  } finally {
    api.logger.warn = warn;
  }
});

test("an unreachable listener is reported once, and recovery once", async () => {
  _internals.resetReachability();
  logs.length = 0;
  const levels = (level) => logs.filter(([l]) => l === level).map(([, m]) => m);
  respond = (_payload, res) => res.socket.destroy();
  assert.equal(await gate(), undefined);
  assert.equal(await gate(), undefined);
  assert.equal(levels("error").length, 1);
  assert.match(levels("error")[0], /listener unreachable at http:\/\/127\.0\.0\.1:\d+\/ingest/);
  assert.match(levels("error")[0], /NOT checked.*STOP not enforced.*NOT recorded/);
  // Per-call detail is demoted to debug.
  assert.equal(levels("warn").length, 0);
  assert.equal(levels("debug").length, 2);
  assert.equal(levels("info").length, 0);

  // An HTTP error status is an answer, not an outage: it recovers.
  respond = (_payload, res) => {
    res.statusCode = 500;
    res.end("{}");
  };
  assert.equal(await gate(), undefined);
  assert.equal(await gate(), undefined);
  assert.equal(levels("info").length, 1);
  assert.match(levels("info")[0], /listener reachable again at /);
  assert.equal(levels("error").length, 1);

  // A second outage is reported again.
  respond = (_payload, res) => res.socket.destroy();
  assert.equal(await gate(), undefined);
  assert.equal(levels("error").length, 2);
  _internals.resetReachability();
});

test("a first success, or an HTTP error while unknown, logs nothing about reachability", async () => {
  _internals.resetReachability();
  logs.length = 0;
  respond = (_payload, res) => {
    res.statusCode = 500;
    res.end("{}");
  };
  assert.equal(await gate(), undefined);
  assert.deepEqual(
    logs.filter(([l]) => l === "error" || l === "info"),
    [],
  );
});

test("without a debug logger, per-call messages stay at warn", async () => {
  _internals.resetReachability();
  logs.length = 0;
  const { debug } = api.logger;
  api.logger.debug = undefined;
  try {
    respond = (_payload, res) => res.socket.destroy();
    assert.equal(await gate(), undefined);
    assert.ok(logs.some(([l, m]) => l === "warn" && m.includes("no decision for exec")));
  } finally {
    api.logger.debug = debug;
    _internals.resetReachability();
  }
});

test("compaction hooks forward their counters, never the session history", async () => {
  for (const hook of _internals.COMPACTION_HOOKS) {
    received.length = 0;
    hooks.get(hook).handler(
      {
        messageCount: 63,
        compactedCount: 2,
        tokenCount: 31000,
        messages: [{ role: "user", content: "secret prompt" }],
        sessionFile: "/home/user/.openclaw/agents/main/sessions/s.jsonl",
      },
      { sessionKey: "s" },
    );
    const deadline = performance.now() + 1000;
    while (received.length === 0 && performance.now() < deadline) {
      await new Promise((resolve) => setTimeout(resolve, 5));
    }
    assert.equal(received.length, 1);
    assert.equal(received[0].hook, hook);
    assert.deepEqual(received[0].event, { messageCount: 63, tokenCount: 31000, compactedCount: 2 });
    assert.deepEqual(received[0].ctx, { sessionKey: "s" });
  }
});
