import assert from "node:assert/strict";
import { test } from "node:test";
import { chatStorageKeys, confirmSend, copyPendingToDraft, discardUnassignedPending, queueSend, readPending } from "./outbox.ts";

const first = { id: "a", content: "First", generate: true, persona: "persona-a" };
const next = { id: "b", content: "Next draft", generate: false, persona: "" };

test("copying an unassigned submission preserves newer draft text and the pending submission", () => {
  const pending = Object.freeze({ ...first });
  assert.equal(copyPendingToDraft("Newer draft", pending), "Newer draft\nFirst");
  assert.equal(copyPendingToDraft("", pending), "First");
  assert.deepEqual(pending, first);
  assert.throws(() => copyPendingToDraft("x".repeat(16000), pending), /Shorten the draft/);
});

test("discarding an unassigned submission leaves scoped pending messages and drafts intact", () => {
  const keys = chatStorageKeys("agent", "conversation");
  const storage = new Map([
    ["kadan.draft.agent.pending", JSON.stringify(first)],
    [keys.pending, JSON.stringify(next)],
    [keys.draft, "Newer draft"],
    ["kadan.draft.another-agent.pending", JSON.stringify(first)],
  ]);
  const unassigned = readPending(storage.get("kadan.draft.agent.pending"));
  assert.equal(copyPendingToDraft(storage.get(keys.draft), unassigned), "Newer draft\nFirst");
  assert.deepEqual(readPending(storage.get(keys.pending)), next);
  discardUnassignedPending({ removeItem: (key) => storage.delete(key) }, "agent");
  assert.equal(storage.has("kadan.draft.agent.pending"), false);
  assert.deepEqual(readPending(storage.get(keys.pending)), next);
  assert.equal(storage.get(keys.draft), "Newer draft");
  assert.deepEqual(readPending(storage.get("kadan.draft.another-agent.pending")), first);
});

test("changing the active conversation preserves each conversation's unconfirmed send", () => {
  const storage = new Map();
  const original = chatStorageKeys("agent", "original");
  const replacement = chatStorageKeys("agent", "replacement");
  storage.set(original.pending, JSON.stringify(first));
  storage.set(original.draft, "Still writing");

  assert.equal(readPending(storage.get(replacement.pending)), null);
  assert.equal(storage.get(replacement.draft), "Still writing");
  storage.set(replacement.pending, JSON.stringify(next));
  assert.deepEqual(readPending(storage.get(original.pending)), first);

  assert.equal(confirmSend(readPending(storage.get(replacement.pending)), [next]), null);
  storage.delete(replacement.pending);
  assert.deepEqual(readPending(storage.get(original.pending)), first);
  assert.equal(readPending(storage.get(chatStorageKeys("another-agent", "original").pending)), null);
});

test("a failed send cannot be overwritten by the next draft", () => {
  const pending = queueSend(null, first);
  assert.throws(() => queueSend(pending, next), /Retry or discard/);
  assert.deepEqual(pending, first);
  assert.deepEqual(queueSend(pending, first), first);
});

test("reload preserves the original ID, text, send mode and persona until confirmed", () => {
  const pending = readPending(JSON.stringify(first));
  assert.deepEqual(confirmSend(pending, [{ id: "older" }]), first);
  assert.equal(confirmSend(pending, [{ id: first.id }]), null);
});

test("a confirmed save removes recovery text rather than restoring it as a draft", () => {
  const pending = confirmSend(readPending(JSON.stringify(first)), [first]);
  assert.equal(pending, null);
  assert.deepEqual(queueSend(pending, next), next);
});

test("malformed outbox records do not become sends", () => {
  for (const raw of [null, "broken", "{}", JSON.stringify({ ...first, generate: "yes" })]) {
    assert.equal(readPending(raw), null);
  }
});

test("explicit coding continuation keeps its original job on reconnect and retry", () => {
  const pending = { ...first, resume_coding_job_id: "saved-job" };
  assert.deepEqual(readPending(JSON.stringify(pending)), pending);
  assert.deepEqual(queueSend(pending, { ...pending, content: "Changed prompt", resume_coding_job_id: "other-job" }), pending);
  assert.equal(readPending(JSON.stringify({ ...pending, generate: false })), null);
  assert.equal(readPending(JSON.stringify({ ...pending, resume_coding_job_id: 42 })), null);
});
