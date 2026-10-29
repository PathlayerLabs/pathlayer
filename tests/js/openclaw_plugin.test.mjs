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
  "../../examples/openclaw_plugin/pathlayer-plugin/index.js"
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

test("the query carries identifiers, never tool arguments", async () => {
  await gate();
  assert.deepEqual(received, [
    {
      hook: "before_tool_call",
      event: { toolName: "exec", toolCallId: "call-1" },
      ctx: { runId: "run-a", sessionKey: "s", agentId: "a" },
    },
  ]);
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
