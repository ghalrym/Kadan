import assert from "node:assert/strict";
import { test } from "node:test";
import { activeLabel, vncUrl } from "./presence.ts";

const now = Date.parse("2026-10-19T12:00:00Z");
const ago = (seconds) => new Date(now - seconds * 1000).toISOString();

test("presence follows the agent's latest message, not the user's", () => {
  assert.equal(activeLabel([], now), "Not active yet");
  assert.equal(activeLabel([{ role: "user", created_at: ago(5) }], now), "Not active yet");
  assert.equal(activeLabel([
    { role: "assistant", created_at: ago(3 * 60) },
    { role: "user", created_at: ago(10) },
  ], now), "Active 3 minutes ago");
});

test("presence rounds down to the largest whole unit", () => {
  assert.equal(activeLabel([{ role: "assistant", created_at: ago(20) }], now), "Active just now");
  assert.equal(activeLabel([{ role: "assistant", created_at: ago(60) }], now), "Active 1 minute ago");
  assert.equal(activeLabel([{ role: "assistant", created_at: ago(2 * 3600 + 59 * 60) }], now), "Active 2 hours ago");
  assert.equal(activeLabel([{ role: "assistant", created_at: ago(3 * 86400) }], now), "Active 3 days ago");
  assert.equal(activeLabel([{ role: "assistant", created_at: ago(400 * 86400) }], now), "Active 1 year ago");
});

test("the VNC relay uses the page's own origin and a matching WebSocket scheme", () => {
  assert.equal(vncUrl({ protocol: "http:", host: "127.0.0.1:8630" }), "ws://127.0.0.1:8630/api/computer/vnc");
  assert.equal(vncUrl({ protocol: "https:", host: "kadan.lan" }), "wss://kadan.lan/api/computer/vnc");
});
