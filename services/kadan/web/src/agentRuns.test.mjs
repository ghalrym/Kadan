import assert from "node:assert/strict";
import { test } from "node:test";
import { activeRun, cancellableImage, imageStatus, isImageAsset, latestConversation } from "./agentRuns.ts";

test("only queued and running responses offer cancellation", () => {
  for (const status of ["queued", "running"]) assert.equal(activeRun({ status }), true);
  for (const status of ["completed", "failed", "cancelled"]) assert.equal(activeRun({ status }), false);
});

test("a completed image stays pending delivery until its message is saved", () => {
  assert.equal(imageStatus({ status: "completed", delivered_message_id: null }), "Image ready · awaiting delivery");
  assert.equal(imageStatus({ status: "completed", delivered_message_id: "message" }), null);
});

test("image failure and cancellation remain visible without offering another cancellation", () => {
  assert.equal(imageStatus({ status: "failed", error: "Worker unavailable" }), "Image failed: Worker unavailable");
  assert.equal(imageStatus({ status: "cancelling" }), "Stopping image…");
  assert.equal(imageStatus({ status: "cancelled" }), "Image cancelled");
  for (const status of ["cancelling", "cancelled", "failed", "completed"]) {
    assert.equal(cancellableImage({ status }), false);
  }
  for (const status of ["queued", "running"]) assert.equal(cancellableImage({ status }), true);
});

test("attachments need both image metadata fields before rendering inline", () => {
  assert.equal(isImageAsset({ is_image: true, media_type: "image/png" }), true);
  assert.equal(isImageAsset({ is_image: false, media_type: "image/png" }), false);
  assert.equal(isImageAsset({ is_image: true, media_type: "text/html" }), false);
});

test("late and duplicate polls cannot replace newer confirmed messages", () => {
  const saved = { id: "conversation", version: 4, messages: [{ id: "latest" }] };
  assert.equal(latestConversation(saved, { ...saved, version: 3, messages: [] }), saved);
  assert.equal(latestConversation(saved, { ...saved }), saved);
  assert.equal(latestConversation(saved, { ...saved, id: "other", version: 8 }), saved);
  const next = { ...saved, version: 5, messages: [...saved.messages, { id: "image" }] };
  assert.equal(latestConversation(saved, next), next);
});
