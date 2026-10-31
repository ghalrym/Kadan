import assert from "node:assert/strict";
import { test } from "node:test";
import { activeRun, cancellableImage, currentResponseOutcome, imageStatus, isImageAsset, latestConversation, responseProgress } from "./agentRuns.ts";

test("earlier failure remains historical when a newer response starts or succeeds", () => {
  const conversation = { id: "chat", messages: [] };
  const failed = { id: "failed", conversation_id: "chat", status: "failed" };
  assert.equal(currentResponseOutcome(conversation, [failed]), failed);
  for (const status of ["queued", "running", "completed"]) {
    assert.equal(currentResponseOutcome(conversation, [failed, { ...failed, id: "new", status }]), undefined);
  }
  assert.equal(currentResponseOutcome({ ...conversation, id: "another" }, [failed]), undefined);
});

test("a later saved user message makes a prior failure historical before its run poll arrives", () => {
  const conversation = { id: "chat", messages: [{ id: "new", role: "user", created_at: "2026-10-30T12:00:00Z" }] };
  const failed = { id: "failed", conversation_id: "chat", status: "failed", trigger_message_id: "old", created_at: "2026-10-30T11:00:00Z" };
  assert.equal(currentResponseOutcome(conversation, [failed]), undefined);
  assert.equal(currentResponseOutcome(conversation, [{ ...failed, trigger_message_id: "new" }])?.id, "failed");
});

test("only queued and running responses offer cancellation", () => {
  for (const status of ["queued", "running"]) assert.equal(activeRun({ status }), true);
  for (const status of ["completed", "failed", "cancelled"]) assert.equal(activeRun({ status }), false);
});

test("response progress clears when work completes, fails or stops", () => {
  for (const status of ["completed", "failed", "cancelled"]) {
    assert.equal(responseProgress([{ status }], { state: "ready", model_state: "loading" }), null);
  }
  assert.equal(responseProgress([], { state: "model_required", download_state: "downloading" }), null);
});

test("queued work is distinguishable from a running response", () => {
  assert.equal(responseProgress([{ status: "queued" }], undefined), "Response queued…");
  assert.equal(responseProgress([{ status: "queued" }], { state: "ready", model_state: "loading" }), "Response queued…");
  assert.equal(responseProgress([{ status: "completed" }, { status: "running" }], undefined), "Waiting for response…");
});

test("downloading is shown only for a reported active language model download", () => {
  const runs = [{ status: "running" }];
  assert.equal(responseProgress(runs, { state: "model_required", download_state: "downloading" }), "Hikari is downloading the language model…");
  for (const download_state of [undefined, null, "not_downloaded", "cancelling", "cancelled", "failed", "complete"]) {
    assert.equal(responseProgress(runs, { state: "model_required", download_state }), "Waiting for response…");
  }
  assert.equal(responseProgress(runs, { state: "unavailable", download_state: "downloading" }), "Waiting for response…");
});

test("loading clears without treating a resident model as proof of prompt processing", () => {
  const runs = [{ status: "running" }];
  assert.equal(responseProgress(runs, { state: "ready", model_state: "loading" }), "Hikari is loading the language model…");
  for (const model_state of [undefined, null, "unloaded", "ready", "offloaded", "unloading", "error"]) {
    assert.equal(responseProgress(runs, { state: "ready", model_state }), "Waiting for response…");
  }
  assert.equal(responseProgress(runs, { state: "unavailable", model_state: "loading" }), "Waiting for response…");
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
