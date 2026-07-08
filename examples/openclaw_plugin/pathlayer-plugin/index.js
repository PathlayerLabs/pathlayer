/**
 * Pathlayer capture plugin (reference implementation for ADP-1).
 *
 * Deliberately dumb: reads the `after_tool_call` hook payload and forwards it
 * verbatim to a local Pathlayer instance over HTTP. All translation logic
 * lives in Python (`pathlayer.adapters.openclaw.translate_openclaw_event`),
 * reviewed and unit-tested there -- this file has as little logic as
 * possible on purpose, since it cannot be tested against a real gateway from
 * this repo.
 *
 * Best-effort / not CI-tested: written against the public plugin hook
 * reference (https://docs.openclaw.ai/plugins/hooks), never run against a
 * real OpenClaw gateway from this repo. See ../README.md for the runbook
 * used to validate it manually against a live instance.
 *
 * Never blocks or crashes the agent run: every send is fire-and-forget
 * (the hook handler does not await it), wrapped in try/catch, with a short
 * timeout -- if Pathlayer is slow, down, or unreachable, the agent's tool
 * call is never delayed or affected. Every attempt (success or failure) is
 * logged via `api.logger`, so a silent failure is still visible somewhere
 * even if nobody is watching Pathlayer's own output.
 *
 * Plain JavaScript on purpose (no TypeScript, no build step): copy this
 * folder into OpenClaw's extensions directory and it runs as-is.
 */

// Keep this literal in sync by hand with DEFAULT_PORT in
// pathlayer/adapters/openclaw.py -- there is no shared source of truth
// across the Python/JavaScript boundary, so a change on one side must be
// mirrored on the other.
const DEFAULT_PORT = 8787;
const PATHLAYER_URL = process.env.PATHLAYER_URL ?? `http://127.0.0.1:${DEFAULT_PORT}/ingest`;
// Short on purpose: this runs on the agent's hot path (after_tool_call), so a
// hung Pathlayer must never noticeably delay the agent.
const SEND_TIMEOUT_MS = 300;

async function forward(api, hook, event, ctx) {
  try {
    const response = await fetch(PATHLAYER_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source: "openclaw", payload: { hook, event, ctx } }),
      signal: AbortSignal.timeout(SEND_TIMEOUT_MS),
    });
    if (response.ok) {
      api.logger.info(`[pathlayer] forwarded ${hook} for tool=${event?.toolName ?? "?"}`);
    } else {
      api.logger.warn(`[pathlayer] ${hook} rejected by Pathlayer: HTTP ${response.status}`);
    }
  } catch (err) {
    // Pathlayer is optional tooling: a failed/slow send must never affect the
    // agent run. Logged so a silent failure is still visible somewhere (e.g.
    // Pathlayer's listener isn't running, or the port is wrong).
    api.logger.warn(`[pathlayer] failed to forward ${hook}: ${err}`);
  }
}

const plugin = {
  id: "pathlayer-plugin",
  name: "Pathlayer capture",
  description:
    "Forwards tool-call events to a local Pathlayer instance for loop/no-progress detection.",
  register(api) {
    api.registerHook("after_tool_call", (event, ctx) => {
      // Fire-and-forget: do NOT await here, so a slow Pathlayer never delays
      // the hook's own return (and therefore the agent's next step).
      void forward(api, "after_tool_call", event, ctx);
    });
    api.logger.info(`[pathlayer] plugin registered, forwarding to ${PATHLAYER_URL}`);
  },
};

export default plugin;
