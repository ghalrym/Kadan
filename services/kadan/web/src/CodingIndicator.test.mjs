import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { runInNewContext } from "node:vm";
import ts from "typescript";
import * as coding from "./coding.ts";

const source = ts.transpileModule(readFileSync(new URL("./CodingIndicator.tsx", import.meta.url), "utf8"), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX },
}).outputText;
const settle = () => new Promise(resolve => setImmediate(resolve));
const job = (status, id = "job") => ({ id, status, cwd: "/project", created_at: "2026-10-30T10:00:00Z" });
const status = jobs => ({ setup: { state: "ready" }, jobs });

// Local hooks and a fake API exercise polling state without a browser or computer.
async function indicator(reply) {
  const slots = [], timers = [], calls = [];
  let cursor = 0, cleanup, opened = 0;
  const jsx = (type, props) => ({ type, props });
  const exports = {};
  runInNewContext(source, {
    exports,
    require: name => ({
      react: {
        useState(initial) {
          const index = cursor++;
          if (!(index in slots)) slots[index] = initial;
          return [slots[index], value => { slots[index] = typeof value === "function" ? value(slots[index]) : value; }];
        },
        useEffect(effect) { if (!cleanup) cleanup = effect(); },
      },
      "react/jsx-runtime": { jsx, jsxs: jsx },
      "./coding.ts": coding,
      "./api": { api: path => { calls.push(path); return reply(); } },
    })[name],
    setTimeout: fn => timers.push(fn), clearTimeout: () => { timers.length = 0; },
  });
  const render = () => { cursor = 0; return exports.default({ onOpen: () => opened++ }); };
  function text(node) {
    if (typeof node === "string" || typeof node === "number") return String(node);
    return node?.props ? [node.props.children].flat(Infinity).map(text).join("") : "";
  }
  const initial = text(render());
  await settle();
  return {
    initial, calls, timers, render, text: () => text(render()), close: () => cleanup(),
    opened: () => opened,
    async poll(next) { reply = next; await timers.shift()(); },
  };
}

test("checking, empty evidence and completed evidence are distinct from running Codex", async () => {
  const view = await indicator(async () => status([]));
  assert.match(view.initial, /Checking status/);
  assert.match(view.text(), /No active job/);
  await view.poll(async () => status([job("completed")]));
  assert.match(view.text(), /No active job.*Latest job: Completed/s);
  assert.match(view.text(), /started/);
  await view.poll(async () => status([job("completed"), job("running", "active")]));
  assert.match(view.text(), /shared computer · Working/);
  assert.doesNotMatch(view.text(), /No active job|Latest job/);
  assert.ok(view.calls.every(path => path === "/computer/coding"));
});

test("stopping and unconfirmed cleanup never look idle or completed", async () => {
  const view = await indicator(async () => status([job("stopping")]));
  assert.match(view.text(), /Stopping and cleaning up/);
  await view.poll(async () => status([job("cleanup_required")]));
  assert.match(view.text(), /Cleanup needs attention/);
  assert.doesNotMatch(view.text(), /No active job/);
});

test("failed polls label retained job evidence as last known and recover", async () => {
  const view = await indicator(async () => status([job("running")]));
  await view.poll(async () => { throw new Error("Computer offline"); });
  assert.match(view.text(), /Status unavailable.*Last known: Working/s);
  assert.match(view.text(), /may still be running/);
  await view.poll(async () => status([job("failed")]));
  assert.match(view.text(), /No active job.*Latest job: Failed/s);
  assert.doesNotMatch(view.text(), /Last known/);
});

test("initial failure and disconnected setup never imply an empty or idle computer", async () => {
  const view = await indicator(async () => { throw new Error("Computer offline"); });
  assert.match(view.text(), /Status unavailable/);
  assert.doesNotMatch(view.text(), /No active job/);
  await view.poll(async () => ({ setup: { state: "unavailable" }, jobs: [] }));
  assert.doesNotMatch(view.text(), /No active job/);
});

test("an unavailable setup with no job read preserves active and cleanup evidence as last known", async () => {
  for (const state of ["running", "cleanup_required"]) {
    const view = await indicator(async () => status([job(state)]));
    await view.poll(async () => ({ setup: { state: "unavailable" }, jobs: [] }));
    assert.match(view.text(), /Status unavailable.*Last known:/s);
    assert.match(view.text(), new RegExp(coding.codingLabels[state]));
    assert.doesNotMatch(view.text(), /No active job/);
    await view.poll(async () => status([]));
    assert.match(view.text(), /No active job/);
    assert.doesNotMatch(view.text(), /Last known/);
  }
});

test("closing ignores a pending status read and schedules no further polling", async () => {
  let resolve;
  const view = await indicator(() => new Promise(done => { resolve = done; }));
  view.close();
  resolve(status([job("running")]));
  await settle();
  assert.equal(view.timers.length, 0);
  assert.match(view.text(), /Checking status/);
});

test("Open Coding uses the existing panel action", async () => {
  const view = await indicator(async () => status([]));
  const button = view.render().props.children[1];
  button.props.onClick();
  assert.equal(view.opened(), 1);
});
