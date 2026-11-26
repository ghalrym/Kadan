import assert from "node:assert/strict";
import { test } from "node:test";
import { activeLabel } from "./presence.ts";

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
