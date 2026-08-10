/**
 * Pathlayer capture plugin (reference implementation for ADP-1).
 *
 * Forwards OpenClaw lifecycle hooks to a local Pathlayer HTTP ingest endpoint.
 * Translation and per-run recording live in Python (`pathlayer.adapters.openclaw*`);
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
 * Best-effort / non-blocking: every send is fire-and-forget with a short timeout.
 */

// Keep in sync with DEFAULT_PORT in pathlayer/adapters/openclaw.py
const DEFAULT_PORT = 8787;
const PATHLAYER_URL = process.env.PATHLAYER_URL ?? `http://127.0.0.1:${DEFAULT_PORT}/ingest`;
const SEND_TIMEOUT_MS = 300;

const FORWARDED_HOOKS = ["before_agent_run", "after_tool_call", "agent_end"];

async function forward(api, hook, event, ctx) {
  try {
    const response = await fetch(PATHLAYER_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source: "openclaw", payload: { hook, event, ctx } }),
      signal: AbortSignal.timeout(SEND_TIMEOUT_MS),
    });
    if (response.ok) {
      api.logger.info(`[pathlayer] forwarded ${hook}`);
    } else {
      api.logger.warn(`[pathlayer] ${hook} rejected by Pathlayer: HTTP ${response.status}`);
    }
  } catch (err) {
    api.logger.warn(`[pathlayer] failed to forward ${hook}: ${err}`);
  }
}

function registerHook(api, hook) {
  api.on(
    hook,
    (event, ctx) => {
      void forward(api, hook, event, ctx);
    },
    { timeoutMs: SEND_TIMEOUT_MS },
  );
}

const plugin = {
  id: "pathlayer-plugin",
  name: "Pathlayer capture",
  description:
    "Forwards OpenClaw run lifecycle and tool events to a local Pathlayer recorder.",
  register(api) {
    for (const hook of FORWARDED_HOOKS) {
      registerHook(api, hook);
    }
    api.logger.info(
      `[pathlayer] plugin registered (${FORWARDED_HOOKS.join(", ")}) -> ${PATHLAYER_URL}`,
    );
  },
};

export default plugin;
