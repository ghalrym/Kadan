import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { registerHooks } from "node:module";
import { beforeEach, test } from "node:test";
import ts from "typescript";

// Exercise handlers with local hooks and a mocked API, without mounting a UI.
const hooksUrl = `data:text/javascript,${encodeURIComponent(`
  export const slots = [];
  export const cleanups = [];
  let index = 0;
  export function render() { index = 0; }
  export function reset() { slots.length = 0; cleanups.length = 0; index = 0; }
  export function useState(initial) {
    const slot = index++;
    if (!(slot in slots)) slots[slot] = initial;
    return [slots[slot], value => { slots[slot] = value; }];
  }
  export function useRef(current) {
    const slot = index++;
    if (!(slot in slots)) slots[slot] = { current };
    return slots[slot];
  }
  export function useEffect(effect) {
    const slot = index++;
    if (!(slot in slots)) { slots[slot] = true; cleanups.push(effect()); }
  }
`)}`;
const apiUrl = `data:text/javascript,${encodeURIComponent(`
  export const calls = [];
  export const replies = [];
  export const agentPath = key => '/agents/' + encodeURIComponent(key);
  export const api = (...args) => { calls.push(args); return replies.shift()(); };
`)}`;
const componentUrl = new URL("./RenameAgent.tsx", import.meta.url).href;
const loader = registerHooks({
  resolve(specifier, context, nextResolve) {
    if (context.parentURL === componentUrl) {
      if (specifier === "react") return { url: hooksUrl, shortCircuit: true };
      if (specifier === "./api") return { url: apiUrl, shortCircuit: true };
    }
    return nextResolve(specifier, context);
  },
  load(url, context, nextLoad) {
    if (url !== componentUrl) return nextLoad(url, context);
    const { outputText } = ts.transpileModule(readFileSync(new URL(url), "utf8"), {
      compilerOptions: { module: ts.ModuleKind.ESNext, jsx: ts.JsxEmit.ReactJSX },
    });
    return { format: "module", source: outputText, shortCircuit: true };
  },
});
const hooks = await import(hooksUrl);
const { calls, replies } = await import(apiUrl);
const { default: RenameAgent } = await import(componentUrl);
loader.deregister();

const agent = { key: "ember", name: "Old", created_at: "2026-01-01T00:00:00Z" };
let updates;
let closed;
function render() {
  hooks.render();
  return RenameAgent({ agent, onUpdated: saved => updates.push(saved), onClose: () => closed++ });
}
function nodes(node, type) {
  if (!node || typeof node !== "object") return [];
  return [ ...(node.type === type ? [node] : []),
    ...[node.props?.children].flat(Infinity).flatMap(child => nodes(child, type)) ];
}
function draft(name) {
  nodes(render(), "input")[0].props.onChange({ target: { value: name } });
}
function save() { return nodes(render(), "form")[0].props.onSubmit({ preventDefault() {} }); }
function reload() { return nodes(render(), "button").find(node => node.props.children === "Reload current name").props.onClick(); }
async function settle() { await new Promise(resolve => setImmediate(resolve)); }
beforeEach(() => { hooks.reset(); calls.length = replies.length = 0; updates = []; closed = 0; });

test("save sends trimmed display name with exact prior name and returns saved agent", async () => {
  draft("  Ember  ");
  const saved = { ...agent, name: "Ember" };
  replies.push(() => Promise.resolve(saved));
  await save();
  assert.deepEqual(calls, [["/agents/ember", { name: "Ember", expected_name: "Old" }, "PUT"]]);
  assert.deepEqual(updates, [saved]);
  assert.equal(closed, 1);
});

test("cancel leaves the stored name alone", () => {
  draft("Ember");
  nodes(render(), "button").find(node => node.props.children === "Cancel").props.onClick();
  assert.equal(closed, 1);
  assert.deepEqual(calls, []);
});

test("blank draft cannot be submitted", async () => {
  draft("   ");
  await save();
  assert.deepEqual(calls, []);
  assert.equal(nodes(render(), "button").at(-1).props.disabled, true);
});

test("failed save retains draft and requires a reload before retrying against a concurrent name", async () => {
  draft("My choice");
  replies.push(() => Promise.reject(new Error("Agent name changed")));
  await save();
  assert.equal(nodes(render(), "input")[0].props.value, "My choice");
  assert.equal(closed, 0);
  await save();
  assert.equal(calls.length, 1);
  replies.push(() => Promise.resolve({ ...agent, name: "Concurrent choice" }));
  reload();
  await settle();
  assert.equal(nodes(render(), "input")[0].props.value, "My choice");
  assert.equal(nodes(render(), "p")[0].props.children.join(""), "Saved name: Concurrent choice");
  replies.push(() => Promise.resolve({ ...agent, name: "My choice" }));
  await save();
  assert.deepEqual(calls.at(-1), ["/agents/ember", { name: "My choice", expected_name: "Concurrent choice" }, "PUT"]);
});

test("uncertain save reloads its committed name without discarding the draft", async () => {
  draft("Ember");
  replies.push(() => Promise.reject(new Error("Connection lost")));
  await save();
  replies.push(() => Promise.resolve({ ...agent, name: "Ember" }));
  reload();
  await settle();
  assert.equal(nodes(render(), "input")[0].props.value, "Ember");
  assert.deepEqual(updates, [{ ...agent, name: "Ember" }]);
  assert.equal(nodes(render(), "button").at(-1).props.disabled, false);
});

test("failed reload keeps save blocked and draft available", async () => {
  draft("Ember");
  replies.push(() => Promise.reject(new Error("Connection lost")));
  await save();
  replies.push(() => Promise.reject(new Error("Still unavailable")));
  reload();
  await settle();
  assert.equal(nodes(render(), "input")[0].props.value, "Ember");
  assert.equal(nodes(render(), "button").at(-1).props.disabled, true);
  assert.deepEqual(updates, []);
});

test("in-flight save cannot submit twice or be cancelled and ignores completion after unmount", async () => {
  draft("Ember");
  let finish;
  replies.push(() => new Promise(resolve => { finish = resolve; }));
  const saving = save();
  await save();
  nodes(render(), "button").find(node => node.props.children === "Cancel").props.onClick();
  assert.equal(closed, 0);
  assert.equal(calls.length, 1);
  for (const cleanup of hooks.cleanups) cleanup?.();
  finish({ ...agent, name: "Ember" });
  await saving;
  assert.deepEqual(updates, []);
  assert.equal(closed, 0);
});
