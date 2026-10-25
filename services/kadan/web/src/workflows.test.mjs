import assert from "node:assert/strict";
import { test } from "node:test";
import { parseWorkflow, exportWorkflow } from "./workflows.ts";

const valid = { schema_version: 1, title: "Review", instructions: "Inspect the current task." };

test("portable export contains only version, title and instructions", () => {
  const exported = exportWorkflow({ title: valid.title, markdown: valid.instructions, id: "private", version: 8, credentials: "secret" });
  assert.deepEqual(parseWorkflow(exported), valid);
});

test("imports reject unsupported fields, types and instruction budgets", () => {
  for (const extra of [{ id: "remote" }, { credentials: "secret" }, { permissions: [] }, { tools: [] }, { schedule: "daily" }]) {
    assert.throws(() => parseWorkflow(JSON.stringify({ ...valid, ...extra })));
  }
  for (const change of [{ schema_version: true }, { schema_version: 2 }, { title: " " }, { title: "x".repeat(501) },
    { instructions: " " }, { instructions: "x".repeat(12001) }]) {
    assert.throws(() => parseWorkflow(JSON.stringify({ ...valid, ...change })));
  }
  assert.throws(() => parseWorkflow("x".repeat(80001)), /too large/);
});

test("import parser returns data without execution", () => {
  const instructions = "Send a message and run commands.";
  assert.equal(parseWorkflow(JSON.stringify({ ...valid, instructions })).instructions, instructions);
});
