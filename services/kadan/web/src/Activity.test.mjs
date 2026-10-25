import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { runInNewContext } from "node:vm";
import ts from "typescript";
import { conversationActivity } from "./activity.ts";

// Inspect component output with local JSX records; no renderer or browser.
const source = ts.transpileModule(readFileSync(new URL("./Activity.tsx", import.meta.url), "utf8"), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX },
}).outputText;
const exports = {};
const jsx = (type, props) => ({ type, props });
runInNewContext(source, {
  exports,
  require: name => {
    if (name === "./activity") return { conversationActivity };
    assert.equal(name, "react/jsx-runtime");
    return { jsx, jsxs: jsx };
  },
});
const conversation = { id: "chat", messages: [] };
const run = (id, status) => ({
  id, status, conversation_id: "chat", trigger_message_id: null, trigger_job_id: null,
  delivered_message_ids: [], updated_at: "2026-10-25T10:00:00Z", error: null,
});
function elements(node) {
  if (!node || typeof node !== "object") return [];
  return [node, ...[node.props?.children].flat(Infinity).flatMap(child => elements(child))];
}
function text(node) {
  if (node == null || typeof node === "boolean") return "";
  if (Array.isArray(node)) return node.map(text).join("");
  return typeof node === "object" ? text(node.props?.children) : String(node);
}
const view = extra => exports.default({ conversation, runs: [], images: [], loading: false, error: "", ...extra });

test("a new conversation shows loading rather than an empty or old activity history", () => {
  const tree = view({ loading: true, runs: [run("old", "failed")] });
  assert.match(text(tree), /Loading activity/);
  assert.doesNotMatch(text(tree), /Response failed|No response or image/);
  assert.equal(tree.type, "details");
  assert.equal(tree.props.open, undefined);
});

test("an initial failure is unavailable while a refresh failure labels the retained snapshot", () => {
  const unavailable = view({ loading: true, error: "Storage offline" });
  assert.match(text(unavailable), /Unavailable/);
  assert.match(text(unavailable), /Activity has not loaded/);
  const stale = view({ error: "Storage offline", runs: [run("done", "completed")] });
  assert.match(text(stale), /Showing the last received activity/);
  assert.match(text(stale), /Response finished/);
});

test("long groups show an explicit bound and total without hiding other groups", () => {
  const tree = view({ runs: [
    ...Array.from({ length: 12 }, (_, i) => run(`finished-${i}`, "completed")), run("active", "queued"),
  ] });
  assert.equal(elements(tree).filter(node => node.type === "li").length, 11);
  assert.match(text(tree), /Showing the 10 most recently updated of 12/);
  assert.match(text(tree), /Response queued/);
});

test("routine polls do not open the panel or introduce live announcements and retry actions", () => {
  const tree = view({ runs: [run("failed", "failed"), run("stopped", "cancelled")] });
  assert.equal(tree.props.open, undefined);
  assert.equal(elements(tree).some(node => node.props?.["aria-live"]), false);
  assert.equal(elements(tree).some(node => node.type === "button"), false);
  assert.match(text(tree), /Response failed/);
  assert.match(text(tree), /Response stopped/);
  assert.match(text(tree), /not retried automatically/);
});
