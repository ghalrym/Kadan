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
test("SSE delivers persisted tool progress and status before the final response", async () => {
  const execution = {
    id: "call-1",
    name: "memory_search",
    arguments: { query: "project" },
    status: "started",
  };
  const frames = [
    { event: "status", data: { message: "Discovering tools…" } },
    {
      event: "tool_started",
      data: { conversation: { id: "saved", version: 2 }, execution },
    },
    {
      event: "tool_finished",
      data: {
        conversation: { id: "saved", version: 3 },
        execution: {
          ...execution,
          status: "completed",
          result: { content: [] },
        },
      },
    },
    { event: "delta", data: { content: "Found it." } },
    {
      event: "done",
      data: {
        conversation: { id: "saved", version: 4 },
        finish_reason: "stop",
      },
    },
  ];
  const events = [];
  await readEvents(
    stream(
      frames.map(
        ({ event, data }) =>
          `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`,
      ),
    ),
    (event) => events.push(event),
  );
  assert.deepEqual(events, frames);
});
test("a saved tool start followed by interruption is still delivered", async () => {
  const events = [];
  await assert.rejects(
    readEvents(
      stream([
        'event: tool_started\ndata: {"conversation":{"version":2},"execution":{"id":"call-1","status":"started"}}\n\n',
      ]),
      (event) => events.push(event),
    ),
    /interrupted/,
  );
  assert.equal(events[0].event, "tool_started");
  assert.equal(events[0].data.conversation.version, 2);
});
