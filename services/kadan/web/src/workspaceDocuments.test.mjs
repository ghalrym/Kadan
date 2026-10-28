import assert from "node:assert/strict";
import { test } from "node:test";
import { documentOutline, documentTrail, workspaceDocuments } from "./workspaceDocuments.ts";

const document = (id, parent_id = null, kind = "space") => ({ id, parent_id, kind, title: id });

test("only Spaces and decks participate in document navigation", () => {
  const rows = [document("Project"), document("Slides", "Project", "deck"),
    document("Memory", "Project", "memory"), document("Persona", null, "persona"),
    document("Chat", "Project", "conversation")];
  assert.deepEqual(workspaceDocuments(rows).map(row => row.id), ["Project", "Slides"]);
});

test("outline orders children under their parents while preserving sibling order and content", () => {
  const rows = [document("Plan", "Project"), document("Other"), document("Project"),
    document("Research", "Project"), document("Details", "Plan")];
  const snapshot = structuredClone(rows);
  assert.deepEqual(documentOutline(rows).map(row => [row.document.id, row.depth]), [
    ["Other", 0], ["Project", 0], ["Plan", 1], ["Details", 2], ["Research", 1],
  ]);
  assert.deepEqual(rows, snapshot);
  assert.equal(documentOutline(rows)[2].document, rows[0]);
});

test("missing or excluded parents leave documents reachable at the root", () => {
  const rows = [document("Orphan", "missing"), document("Nested", "Orphan")];
  assert.deepEqual(documentOutline(rows).map(row => [row.document.id, row.depth]), [
    ["Orphan", 0], ["Nested", 1],
  ]);
  assert.deepEqual(documentTrail(rows[1], rows).map(row => row.id), ["Orphan", "Nested"]);
});

test("breadcrumb follows stored parent links from root to current document", () => {
  const rows = [document("Project"), document("Plan", "Project"), document("Details", "Plan")];
  assert.deepEqual(documentTrail(rows[2], rows).map(row => row.id), ["Project", "Plan", "Details"]);
});

test("incomplete cyclic data stays reachable once and navigation terminates", () => {
  const rows = [document("A", "B"), document("B", "A"), document("Self", "Self")];
  assert.deepEqual(documentOutline(rows).map(row => row.document.id), ["A", "B", "Self"]);
  assert.deepEqual(documentTrail(rows[0], rows).map(row => row.id), ["B", "A"]);
  assert.deepEqual(documentTrail(rows[2], rows).map(row => row.id), ["Self"]);
});
