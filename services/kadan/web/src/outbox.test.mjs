import assert from "node:assert/strict";
import { test } from "node:test";
import { confirmSend, queueSend, readPending } from "./outbox.ts";

const first = { id: "a", content: "First", generate: true, persona: "persona-a" };
const next = { id: "b", content: "Next draft", generate: false, persona: "" };

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
