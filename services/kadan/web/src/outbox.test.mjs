import assert from "node:assert/strict";
import { test } from "node:test";
import { chatStorageKeys, confirmSend, queueSend, readPending } from "./outbox.ts";

const first = { id: "a", content: "First", generate: true, persona: "persona-a" };
const next = { id: "b", content: "Next draft", generate: false, persona: "" };

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
