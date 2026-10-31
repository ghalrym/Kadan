import assert from "node:assert/strict";
import { test } from "node:test";
import { conversationActivity } from "./activity.ts";

const conversation = { id: "chat", messages: [{ role: "user", id: "question", content: "Research a weekend trip" }] };
const run = (id, status, extra = {}) => ({
  id, conversation_id: "chat", trigger_message_id: "question", trigger_job_id: null,
  delivered_message_ids: [], status, error: null, updated_at: "2026-10-25T10:00:00Z", ...extra,
});
const image = (id, status, extra = {}) => ({
  id, conversation_id: "chat", prompt: "A garden", status, error: null,
  delivered_message_id: null, updated_at: "2026-10-25T10:00:00Z", ...extra,
});

test("failed responses retain their recorded error and remain distinct from deliberate stops", () => {
  const [review, progressing, finished, earlier] = conversationActivity(conversation, [
    run("failed", "failed", { error: "Context allowance exceeded" }), run("stop", "cancelled"),
  ], []);
  assert.equal(earlier.items[0].state, "Response failed");
  assert.equal(earlier.items[0].detail, "Context allowance exceeded");
  assert.equal(review.items[0].state, "Response stopped");
  assert.equal(review.items[0].detail, "");
  assert.deepEqual(progressing.items, []);
  assert.deepEqual(finished.items, []);
});

test("response completion reports message delivery evidence without claiming task success", () => {
  const [, , finished] = conversationActivity(conversation, [
    run("silent", "completed"), run("spoken", "completed", { delivered_message_ids: ["reply"] }),
  ], []);
  assert.equal(finished.items[0].detail, "No message sent");
  assert.equal(finished.items[1].detail, "");
  assert.ok(finished.items.every(item => item.state === "Response finished"));
});

test("queued and running responses stay in progress with their triggering request", () => {
  const [, progressing] = conversationActivity(conversation, [run("queued", "queued"), run("running", "running")], []);
  assert.deepEqual(progressing.items.map(item => item.state), ["Response queued", "Responding"]);
  assert.ok(progressing.items.every(item => item.title === "Research a weekend trip"));
});

test("image generation waits for recorded delivery before entering finished", () => {
  const [, progressing, finished] = conversationActivity(conversation, [], [
    image("ready", "completed"), image("delivered", "completed", { delivered_message_id: "saved-image" }),
    image("stopping", "cancelling"),
  ]);
  assert.deepEqual(progressing.items.map(item => item.state), ["Image ready, not sent yet", "Stopping image"]);
  assert.equal(finished.items[0].state, "Image sent");
});

test("images fail and cancel independently from response records", () => {
  const [review] = conversationActivity(conversation, [run("same-id", "cancelled")], [
    image("failed", "failed", { error: "Image worker unavailable" }), image("same-id", "cancelled"),
  ]);
  assert.equal(review.items.length, 3);
  assert.equal(review.items.find(item => item.state === "Image failed").detail, "Image worker unavailable");
  assert.equal(review.items.find(item => item.state === "Image cancelled").kind, "image");
});

test("activity excludes other conversations while a replacement snapshot is pending", () => {
  const groups = conversationActivity(conversation, [run("other", "failed", { conversation_id: "old" })], [
    image("old-image", "running", { conversation_id: "old" }),
  ]);
  assert.ok(groups.every(group => group.items.length === 0));
});

test("recent updates sort first with a deterministic tie break without mutating records", () => {
  const runs = [run("older", "completed"), run("newer", "completed", { updated_at: "2026-10-25T11:00:00Z" })];
  const [, , finished] = conversationActivity(conversation, runs, []);
  assert.deepEqual(finished.items.map(item => item.id), ["newer", "older"]);
  assert.deepEqual(runs.map(item => item.id), ["older", "newer"]);
});

test("image follow-up responses and missing triggering messages have honest fallback titles", () => {
  const [, progressing] = conversationActivity(conversation, [
    run("follow-up", "queued", { trigger_message_id: null, trigger_job_id: "image" }),
    run("unknown", "running", { trigger_message_id: "missing" }),
  ], []);
  assert.deepEqual(progressing.items.map(item => item.title), ["Image follow-up response", "Response"]);
});


test("a newer response preserves all previous failures without counting them as current review", () => {
  const failed = run("failed", "failed", { error: "Command rejected" });
  const stopped = run("stopped", "cancelled");
  const groups = conversationActivity(conversation, [failed, stopped, run("new", "running")], []);
  assert.equal(groups[0].items.length, 0);
  assert.equal(groups[1].items.length, 1);
  assert.equal(groups[3].historical, true);
  assert.equal(groups[3].items.length, 2);
  assert.equal(groups[3].items.find(item => item.id === "failed").detail, "Command rejected");
});

test("the latest response failure stays current without labelling older failures resolved", () => {
  const groups = conversationActivity(conversation, [run("old", "failed"), run("new", "failed", { error: "Still blocked" })], []);
  assert.equal(groups[0].items[0].detail, "Still blocked");
  assert.equal(groups[3].items[0].id, "old");
  assert.equal(groups[3].items[0].state, "Response failed");
});
