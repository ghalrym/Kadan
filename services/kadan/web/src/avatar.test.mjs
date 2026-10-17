import assert from "node:assert/strict";
import { test } from "node:test";
import { avatarOffset, avatarInitials, chatActivity } from "./avatar.ts";

test("idle movement follows the pointer direction and is bounded radially", () => {
  assert.deepEqual(avatarOffset(50, 60, 50, 60), { x: 0, y: 0 });
  assert.deepEqual(avatarOffset(170, 60, 50, 60), { x: 2, y: 0 });
  const far = avatarOffset(-10000, 10000, 50, 60);
  assert.ok(far.x < 0 && far.y > 0);
  assert.ok(Math.hypot(far.x, far.y) <= 4.00001);
});
test("real chat events distinguish waiting, response output and completion", () => {
  assert.equal(chatActivity({ event: "user_saved", data: {} }), "waiting");
  assert.equal(
    chatActivity({ event: "delta", data: { content: "Hello" } }),
    "responding",
  );
  assert.equal(chatActivity({ event: "done", data: {} }), "idle");
  assert.equal(chatActivity({ event: "error", data: {} }), "idle");
});
test("initials keep names short and handle empty or unicode names", () => {
  assert.equal(avatarInitials("  Research Agent Three "), "RA");
  assert.equal(avatarInitials(""), "");
  assert.equal(avatarInitials("🌱 Garden"), "🌱G");
});
