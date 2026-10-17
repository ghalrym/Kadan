import assert from "node:assert/strict";
import { test } from "node:test";
import { readEvents } from "./api.ts";

function stream(parts) {
  const bytes = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const part of parts) controller.enqueue(bytes.encode(part));
      controller.close();
    },
  });
}
test("SSE preserves deltas split across network chunks and commits only at done", async () => {
  const events = [];
  await readEvents(
    stream([
      'event: delta\r\ndata: {"content":"Hel',
      'lo"}\r\n\r',
      '\nevent: done\ndata: {"conversation":{"id":"saved"},"finish_reason":"stop"}\n\n',
    ]),
    (event) => events.push(event),
  );
  assert.equal(events[0].data.content, "Hello");
  assert.equal(events[1].data.conversation.id, "saved");
});
test("SSE reports an upstream error after showing the received text", async () => {
  const events = [];
  await assert.rejects(
    readEvents(
      stream([
        'event: delta\ndata: {"content":"Partial"}\n\nevent: error\ndata: {"message":"Storage conflict"}\n\n',
      ]),
      (event) => events.push(event),
    ),
    /Storage conflict/,
  );
  assert.deepEqual(
    events.map((event) => event.event),
    ["delta", "error"],
  );
});
test("an EOF without the persisted done event is an interruption", async () => {
  await assert.rejects(
    readEvents(
      stream(['event: delta\ndata: {"content":"Partial"}\n\n']),
      () => {},
    ),
    /interrupted/,
  );
});
test("malformed JSON cannot be accepted as a completed answer", async () => {
  await assert.rejects(
    readEvents(stream(["event: done\ndata: broken\n\n"]), () => {}),
    SyntaxError,
  );
});
test("heartbeat comments do not produce message events", async () => {
  const events = [];
  await readEvents(
    stream([
      ': keepalive\n\nevent: done\ndata: {"conversation":{"id":"saved"},"finish_reason":"length"}\n\n',
    ]),
    (event) => events.push(event),
  );
  assert.equal(events.length, 1);
  assert.equal(events[0].data.finish_reason, "length");
});
