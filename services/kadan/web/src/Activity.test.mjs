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
    if (name === "lucide-react") return new Proxy({}, { get: (_, icon) => icon });
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
  assert.doesNotMatch(text(tree), /Response failed|Nothing yet/);
});

test("an initial failure is unavailable while a refresh failure labels the retained snapshot", () => {
  const unavailable = view({ loading: true, error: "Storage offline" });
  assert.match(text(unavailable), /Couldn’t load activity/);
  const stale = view({ error: "Storage offline", runs: [run("done", "completed")] });
  assert.match(text(stale), /Showing the last update/);
  assert.match(text(stale), /Response finished/);
});

test("long groups show an explicit bound and total without hiding other groups", () => {
  const tree = view({ runs: [
    ...Array.from({ length: 12 }, (_, i) => run(`finished-${i}`, "completed")), run("active", "queued"),
  ] });
  assert.equal(elements(tree).filter(node => node.type === "li").length, 11);
  assert.match(text(tree), /Showing 10 of 12/);
  assert.match(text(tree), /Response queued/);
});

test("retained image failures cannot displace queued or running responses and images", () => {
  const tree = view({
    runs: [run("active", "running"), run("waiting", "queued")],
    images: [
      ...Array.from({ length: 12 }, (_, i) => ({
        id: `failed-${i}`, status: "failed", conversation_id: "chat", prompt: "Old image",
        error: "Worker unavailable", updated_at: "2026-10-24T10:00:00Z",
      })),
      {
        id: "image", status: "running", conversation_id: "chat", prompt: "Current image",
        updated_at: "2026-10-25T10:00:00Z",
      },
    ],
  });
  const rows = elements(tree).filter(node => node.type === "li");
  assert.equal(rows.length, 13);
  assert.match(text(tree), /3 in progress/);
  assert.match(text(tree), /12 to review/);
  assert.match(text(tree), /Showing 10 of 12/);
  assert.match(text(rows[0]), /Responding|Response queued|Generating image/);
  assert.ok(rows.some(row => text(row).includes("Responding")));
  assert.ok(rows.some(row => text(row).includes("Response queued")));
  assert.ok(rows.some(row => text(row).includes("Generating image")));
});

test("current and earlier failures expose their full error through a disclosure", () => {
  const detail = `${"Worker failed during preparation. ".repeat(30)}Free memory and retry.`;
  const tree = view({
    runs: [{ ...run("failed", "failed"), error: detail }, run("newer", "running")],
    images: [{
      id: "failed-image", status: "failed", conversation_id: "chat", prompt: "Garden",
      error: detail, updated_at: "2026-10-25T10:00:00Z",
    }],
  });
  const disclosures = elements(tree).filter(node => node.type === "details" && node.props.className === "activity-error-details");
  assert.equal(disclosures.length, 2);
  for (const disclosure of disclosures) {
    assert.equal(disclosure.props.open, undefined);
    const summary = disclosure.props.children.find(node => node.type === "summary");
    assert.match(text(summary), /Error details/);
    assert.equal(text(disclosure.props.children.find(node => node.type === "p")), detail);
  }
  const earlier = elements(tree).find(node => node.type === "details" && text(node.props.children[0]).startsWith("Earlier"));
  assert.equal(elements(earlier).filter(node => node.props?.className === "activity-error-details").length, 1);
});

test("routine polls do not open the panel or introduce live announcements and retry actions", () => {
  const tree = view({ runs: [run("failed", "failed"), run("stopped", "cancelled")] });
  assert.ok(elements(tree).filter(node => node.type === "details").every(node => node.props.open === undefined));
  assert.equal(elements(tree).some(node => node.props?.["aria-live"]), false);
  assert.equal(elements(tree).some(node => node.type === "button"), false);
  assert.match(text(tree), /Response failed/);
  assert.match(text(tree), /Response stopped/);
});
