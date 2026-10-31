import assert from "node:assert/strict";
import { test } from "node:test";
import { stopComputerTask, workLabel } from "./computerWork.ts";

const now = Date.parse("2026-10-31T12:00:00Z");
const task = (status, summary = null, secondsAgo = 0) => ({
  id: "t", instructions: "Open", origin: null, status, summary,
  updated_at: new Date(now - secondsAgo * 1000).toISOString(),
});
const idle = (last) => ({ task: null, queued: 0, pending_steering: 0, last });

test("the panel status names the computer agent's current work", () => {
  assert.equal(workLabel(null), "Idle");
  assert.equal(workLabel({ task: null, queued: 0, pending_steering: 0 }), "Idle");
  assert.equal(workLabel({ task: task("queued"), queued: 0, pending_steering: 0 }), "Starting");
  assert.equal(workLabel({ task: task("running"), queued: 2, pending_steering: 0 }), "Working · 2 more queued");
});

test("unread steering shows in the status until the assistant reads it", () => {
  assert.equal(workLabel({ task: task("running"), queued: 0, pending_steering: 1 }), "Working · reading your request");
});

test("an idle computer shows its last result briefly, then Idle", () => {
  assert.equal(workLabel(idle(task("failed", "No network.", 5)), now), "Failed: No network.");
  assert.equal(workLabel(idle(task("completed", "YouTube is open.", 60)), now), "Done: YouTube is open.");
  assert.equal(workLabel(idle(task("cancelled", null, 5)), now), "Stopped");
  assert.equal(workLabel(idle(task("failed", "No network.", 121)), now), "Idle");
});


test("stopping a computer task sends a POST", async () => {
  const original = globalThis.fetch;
  let request;
  globalThis.fetch = async (url, options) => {
    request = { url, options };
    return { ok: true, status: 204 };
  };
  try {
    await stopComputerTask("/agents/test/computer", "task/1");
    assert.equal(request.url, "/api/agents/test/computer/tasks/task%2F1/cancel");
    assert.equal(request.options.method, "POST");
  } finally {
    globalThis.fetch = original;
  }
});
