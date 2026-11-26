import assert from "node:assert/strict";
import { test } from "node:test";
import { folderTrail, formatSize, itemMeta, moveTargets, recentItems, visibleItems } from "./spaceItems.ts";

const item = (id, type, extra = {}) => ({
  id, type, title: id, parent_id: null, pinned: false, filename: null, media_type: null,
  size_bytes: null, version: 1, created_at: "2026-01-01T00:00:00Z", updated_at: "2026-01-01T00:00:00Z",
  deleted_at: null, ...extra,
});
const items = [
  item("docs", "folder"),
  item("inner", "folder", { parent_id: "docs" }),
  item("plan", "page", { updated_at: "2026-01-05T00:00:00Z", pinned: true }),
  item("deck", "slides", { parent_id: "docs", updated_at: "2026-01-04T00:00:00Z" }),
  item("photo", "image", { parent_id: "inner", filename: "beach.png", updated_at: "2026-01-03T00:00:00Z" }),
];

test("All browses one folder level with folders first", () => {
  assert.deepEqual(visibleItems(items, "all", "", null).map((row) => row.id), ["docs", "plan"]);
  assert.deepEqual(visibleItems(items, "all", "", "docs").map((row) => row.id), ["inner", "deck"]);
});

test("type tabs, Pinned and search list matches from every folder", () => {
  assert.deepEqual(visibleItems(items, "image", "", null).map((row) => row.id), ["photo"]);
  assert.deepEqual(visibleItems(items, "pinned", "", "docs").map((row) => row.id), ["plan"]);
  assert.deepEqual(visibleItems(items, "all", "BEACH", null).map((row) => row.id), ["photo"]);
});

test("breadcrumbs follow parents and moves never target the item's own subtree", () => {
  assert.deepEqual(folderTrail(items, "inner").map((row) => row.id), ["docs", "inner"]);
  assert.deepEqual(folderTrail(items, null), []);
  assert.deepEqual(moveTargets(items, items[0]).map((row) => row.id), []);
  assert.deepEqual(moveTargets(items, items[4]).map((row) => row.id), ["docs", "inner"]);
});

test("recents skip folders and labels stay compact", () => {
  assert.deepEqual(recentItems(items, 2).map((row) => row.id), ["plan", "deck"]);
  assert.equal(formatSize(51814), "51 KB");
  assert.equal(formatSize(2348810), "2.2 MB");
  assert.equal(formatSize(900), "900 B");
  const now = new Date("2026-01-05T15:00:00");
  assert.match(itemMeta(item("x", "file", { size_bytes: 2048, updated_at: "2026-01-04T10:00:00" }), now), /^Yesterday · 2.0 KB$/);
  assert.equal(itemMeta(item("f", "folder", { size_bytes: 10 }), now).includes("·"), false);
});
