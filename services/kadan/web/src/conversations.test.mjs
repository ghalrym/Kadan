import assert from "node:assert/strict";
import { test } from "node:test";
import {
  emptyConversation,
  listedConversations,
  pickConversation,
  relativeTime,
  summarize,
  upsertConversation,
} from "./conversations.ts";

const row = (id, updated_at, extra = {}) => ({
  id, agent_key: "guide", title: id, message_count: 1, preview: "",
  version: 1, created_at: updated_at, updated_at, ...extra,
});

test("opens the preferred chat while it exists, otherwise the latest", () => {
  const rows = [row("old", "2026-01-01T00:00:00Z"), row("recent", "2026-01-02T00:00:00Z")];
  assert.equal(pickConversation(rows, "old"), "old");
  assert.equal(pickConversation(rows, "gone"), "recent");
  assert.equal(pickConversation([], null), null);
});

test("lists started chats plus the open empty chat", () => {
  const rows = [
    row("a", "2026-01-01T00:00:00Z"),
    row("b", "2026-01-02T00:00:00Z", { message_count: 0 }),
    row("c", "2026-01-03T00:00:00Z", { message_count: 0 }),
  ];
  assert.deepEqual(listedConversations(rows, "b").map((item) => item.id), ["b", "a"]);
  assert.deepEqual(listedConversations(rows, null).map((item) => item.id), ["a"]);
  assert.equal(emptyConversation(rows)?.id, "c");
  assert.equal(emptyConversation([rows[0]]), null);
});

test("summaries name unnamed chats from their first message and ignore stale versions", () => {
  const conversation = {
    id: "x", agent_key: "guide", kind: "conversation", title: "New chat", version: 4,
    created_at: "2026-01-01T00:00:00Z", updated_at: "2026-01-02T00:00:00Z",
    messages: [{ id: "m", role: "user", content: "  Plan   a trip\nto Kyoto " }],
  };
  const summary = summarize(conversation);
  assert.equal(summary.title, "Plan a trip to Kyoto");
  assert.equal(summary.message_count, 1);
  assert.equal(summarize({ ...conversation, title: "Kyoto" }).title, "Kyoto");
  assert.equal(summarize({ ...conversation, messages: [] }).title, "New chat");
  const rows = [{ ...summary, version: 5, title: "Newer" }];
  assert.equal(upsertConversation(rows, summary)[0].title, "Newer");
  assert.equal(upsertConversation(rows, { ...summary, version: 6, title: "Newest" })[0].title, "Newest");
});

test("relative times stay compact", () => {
  const now = Date.parse("2026-03-10T12:00:00Z");
  assert.equal(relativeTime("2026-03-10T11:59:30Z", now), "now");
  assert.equal(relativeTime("2026-03-10T11:55:00Z", now), "5m");
  assert.equal(relativeTime("2026-03-10T09:00:00Z", now), "3h");
  assert.equal(relativeTime("2026-03-08T12:00:00Z", now), "2d");
});
