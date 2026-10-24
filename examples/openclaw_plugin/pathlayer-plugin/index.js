/**
 * Pathlayer plugin (reference implementation for the OpenClaw adapter).
 *
 * Forwards OpenClaw lifecycle hooks to a local Pathlayer HTTP ingest endpoint,
 * and asks it before each tool call whether the run has been stopped.
 * Translation, recording and decisions live in Python (`pathlayer.adapters.openclaw*`);
 * this file stays as dumb as possible.
 *
 * Required OpenClaw config (conversation hooks need explicit opt-in):
 *
 * {
 *   "plugins": {
 *     "entries": {
 *       "pathlayer-plugin": {
 *         "hooks": {
 *           "allowConversationAccess": true
 *         }
 *       }
 *     }
 *   }
 * }
 *
 * Observation hooks are best-effort: every send is fire-and-forget with a short
 * timeout. The `before_tool_call` gate fails open. OpenClaw blocks a tool call
 * when a `before_tool_call` handler throws or times out, so this handler never
 * throws and bounds its own wait far below its hook timeout: when Pathlayer is
 * down, slow, or answers anything but a block, the tool call runs.
 */

// Keep in sync with DEFAULT_PORT in pathlayer/adapters/openclaw.py
const DEFAULT_PORT = 8787;
const PATHLAYER_URL = process.env.PATHLAYER_URL ?? `http://127.0.0.1:${DEFAULT_PORT}/ingest`;
const SEND_TIMEOUT_MS = 300;

// Before asking, wait at most this long for the run's previous forward, so the
// event that triggered a STOP has reached Pathlayer before the next tool call.
const ORDERING_WAIT_MS = SEND_TIMEOUT_MS;
// Budget of the decision request itself, response body included.
const DECISION_TIMEOUT_MS = 500;
// OpenClaw's budget for the gate. Expiry would block the tool call, so it stays
// far above the most this handler can wait: ORDERING_WAIT_MS + DECISION_TIMEOUT_MS.
const DECISION_HOOK_TIMEOUT_MS = 5000;

// `llm_output` is required for the bounded total-token cost estimate. The
// Python side will only price it when a non-negative `usage.total` is present;
// it never infers a model call from `after_tool_call`.
const FORWARDED_HOOKS = ["before_agent_run", "llm_output", "after_tool_call", "agent_end"];
const DECISION_HOOK = "before_tool_call";

// The latest in-flight forward of each run.
const inflight = new Map();

function log(api, level, message) {
  try {
    api.logger[level](message);
  } catch {
    // A logging failure must never turn into a blocked tool call.
  }
}

function runIdOf(event, ctx) {
  const runId = ctx?.runId ?? event?.runId;
  return typeof runId === "string" && runId.trim() ? runId : undefined;
}

function post(payload, timeoutMs) {
  return fetch(PATHLAYER_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source: "openclaw", payload }),
    signal: AbortSignal.timeout(timeoutMs),
  });
}

async function forward(api, hook, event, ctx) {
  try {
    const response = await post({ hook, event, ctx }, SEND_TIMEOUT_MS);
    if (response.ok) {
      log(api, "info", `[pathlayer] forwarded ${hook}`);
    } else {
      log(api, "warn", `[pathlayer] ${hook} rejected by Pathlayer: HTTP ${response.status}`);
    }
  } catch (err) {
    log(api, "warn", `[pathlayer] failed to forward ${hook}: ${err}`);
  }
}

function track(runId, sending) {
  if (runId === undefined) {
    return;
  }
  inflight.set(runId, sending);
  // forward() never rejects, so this chain cannot either.
  sending.finally(() => {
    if (inflight.get(runId) === sending) {
      inflight.delete(runId);
    }
  });
}

function registerHook(api, hook) {
  api.on(
    hook,
    (event, ctx) => {
      track(runIdOf(event, ctx), forward(api, hook, event, ctx));
    },
    { timeoutMs: SEND_TIMEOUT_MS },
  );
}

function delay(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms).unref?.();
  });
}

// Identifiers only: tool arguments never leave OpenClaw for a decision.
function decisionQuery(event, ctx) {
  return {
    hook: DECISION_HOOK,
    event: { toolName: event?.toolName, toolCallId: event?.toolCallId, runId: event?.runId },
    ctx: { runId: ctx?.runId, sessionKey: ctx?.sessionKey, agentId: ctx?.agentId },
  };
}

async function decide(api, event, ctx) {
  try {
    const runId = runIdOf(event, ctx);
    const pending = runId === undefined ? undefined : inflight.get(runId);
    if (pending !== undefined) {
      await Promise.race([pending, delay(ORDERING_WAIT_MS)]);
    }
    const response = await post(decisionQuery(event, ctx), DECISION_TIMEOUT_MS);
    if (!response.ok) {
      return undefined;
    }
    const decision = (await response.json())?.decision;
    if (decision?.block === true) {
      const reason =
        typeof decision.blockReason === "string" && decision.blockReason
          ? decision.blockReason
          : "Pathlayer stopped this run.";
      log(api, "warn", `[pathlayer] blocked ${event?.toolName}: ${reason}`);
      return { block: true, blockReason: reason };
    }
  } catch (err) {
    log(api, "warn", `[pathlayer] no decision for ${event?.toolName}, allowing it: ${err}`);
  }
  return undefined;
}

const plugin = {
  id: "pathlayer-plugin",
  name: "Pathlayer capture",
  description:
    "Forwards OpenClaw run lifecycle and tool events to a local Pathlayer recorder, " +
    "and blocks the tool calls of a run it stopped when the recorder enforces STOP.",
  register(api) {
    for (const hook of FORWARDED_HOOKS) {
      registerHook(api, hook);
    }
    api.on(DECISION_HOOK, (event, ctx) => decide(api, event, ctx), {
      timeoutMs: DECISION_HOOK_TIMEOUT_MS,
    });
    log(
      api,
      "info",
      `[pathlayer] plugin registered (${[...FORWARDED_HOOKS, DECISION_HOOK].join(", ")}) -> ${PATHLAYER_URL}`,
    );
  },
};

// Read by the plugin's tests only.
export const _internals = {
  DECISION_TIMEOUT_MS,
  DECISION_HOOK_TIMEOUT_MS,
  ORDERING_WAIT_MS,
  SEND_TIMEOUT_MS,
};

export default plugin;
