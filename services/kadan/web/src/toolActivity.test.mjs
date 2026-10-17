import assert from "node:assert/strict";
import { test } from "node:test";
import { conversationActivity, toolStatus } from "./toolActivity.ts";

const execution = {
  id: "call-1",
  name: "memory_create",
  arguments: { title: "Remember this" },
  status: "started",
};
function toolMessage(id, value) {
  return { id, role: "tool", content: JSON.stringify(value) };
}

test("a saved outcome replaces its intent without duplicating or reordering tool activity", () => {
  const user = { id: "user", role: "user", content: "Remember this" };
  const second = { ...execution, id: "call-2" };
  const completed = {
    ...execution,
    status: "completed",
    result: { id: "memory" },
  };
  const activities = conversationActivity([
    user,
    toolMessage("start", execution),
    toolMessage("second", second),
    toolMessage("finish", completed),
  ]);
  assert.equal(activities.length, 3);
  assert.deepEqual(activities[0], { kind: "message", message: user });
  assert.deepEqual(activities[1].execution, completed);
  assert.equal(activities[2].execution.id, "call-2");
});

test("an interrupted stored intent is unknown unless it is observed running in this session", () => {
  assert.equal(toolStatus(execution, false), "Outcome unknown");
  assert.equal(toolStatus(execution, true), "Running…");
});

test("saved outcomes keep their final status even if an intent is replayed later", () => {
  const failed = {
    ...execution,
    status: "error",
    result: { message: "Version conflict" },
  };
  const activities = conversationActivity([
    toolMessage("finish", failed),
    toolMessage("start", execution),
  ]);
  assert.equal(activities.length, 1);
  assert.equal(toolStatus(activities[0].execution, true), "Failed");
  assert.equal(
    toolStatus({ ...execution, status: "completed" }, false),
    "Completed",
  );
});

test("unreadable tool messages stay separate from Markdown messages", () => {
  for (const content of [
    "broken",
    "null",
    JSON.stringify({ ...execution, arguments: [] }),
    JSON.stringify({ ...execution, result: "broken" }),
  ]) {
    const [activity] = conversationActivity([
      { id: "tool", role: "tool", content },
    ]);
    assert.equal(activity.kind, "tool");
    assert.equal(activity.execution, null);
  }
});
