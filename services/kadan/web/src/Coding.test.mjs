import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { registerHooks } from "node:module";
import { afterEach, beforeEach, test } from "node:test";
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
    if (!(slot in slots)) slots[slot] = typeof initial === 'function' ? initial() : initial;
    return [slots[slot], value => { slots[slot] = typeof value === 'function' ? value(slots[slot]) : value; }];
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
  export const api = (...args) => { calls.push(args); return replies.shift()(); };
`)}`;
const componentUrl = new URL("./Coding.tsx", import.meta.url).href;
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
const { default: Coding } = await import(componentUrl);
loader.deregister();


let closed;
let interval;
const originalInterval = globalThis.setInterval;
const originalClear = globalThis.clearInterval;
function render() { hooks.render(); return Coding({ onClose: () => closed++ }); }
function nodes(node, type) {
  if (!node || typeof node !== "object") return [];
  return [ ...(node.type === type ? [node] : []),
    ...[node.props?.children].flat(Infinity).flatMap(child => nodes(child, type)) ];
}
function text(node) {
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (!node || typeof node !== "object") return "";
  return [node.props?.children].flat(Infinity).map(text).join("");
}
function button(label) {
  return nodes(render(), "button").find(node => text(node) === label || text(node).startsWith(label));
}
async function settle() { for (let i = 0; i < 5; i++) await new Promise(resolve => setImmediate(resolve)); }
const thread = { id: "thread-a", title: "Fix the parser", cwd: "/home/gem/project", created_at: null, updated_at: "2026-10-30T12:00:00Z", running: true };
const signedIn = { state: "signed_in", message: "Signed in with ChatGPT." };
function status(threads = [thread], setup = signedIn) { return { setup, threads }; }
function reply(value) { replies.push(() => Promise.resolve(value)); }
async function open(result = status()) { reply(result); render(); await settle(); }
beforeEach(() => {
  hooks.reset(); calls.length = replies.length = 0; closed = 0;
  globalThis.setInterval = fn => { interval = fn; return 1; };
  globalThis.clearInterval = () => { interval = undefined; };
});
afterEach(() => {
  for (const cleanup of hooks.cleanups) cleanup?.();
  globalThis.setInterval = originalInterval;
  globalThis.clearInterval = originalClear;
});

test("signed-out computer offers one sign-in button that shows the device code", async () => {
  await open(status([], { state: "signed_out", message: "Sign in with ChatGPT to let your agent use Codex." }));
  assert.match(text(render()), /No Codex threads yet/);
  const login = { state: "signing_in", message: "Open the sign-in page and enter the code.", login: { url: "https://auth.example/device", code: "ABCD-1234" } };
  reply(login);
  reply(status([], login));
  button("Sign in with ChatGPT").props.onClick();
  await settle();
  assert.deepEqual(calls.slice(-2).map(call => call[0]), ["/computer/coding/login", "/computer/coding"]);
  const view = render();
  assert.match(text(view), /ABCD-1234/);
  assert.equal(nodes(view, "a")[0].props.href, "https://auth.example/device");
});

test("signed-in computer can sign out", async () => {
  await open(status([]));
  reply({ state: "signed_out", message: "Signed out." });
  reply(status([], { state: "signed_out", message: "Signed out." }));
  button("Sign out").props.onClick();
  await settle();
  assert.equal(calls.at(-2)[0], "/computer/coding/logout");
  assert.ok(button("Sign in with ChatGPT"));
});

test("threads list opens a thread transcript and polls it", async () => {
  await open();
  assert.match(text(render()), /Fix the parser/);
  assert.match(text(render()), /Working/);
  reply(status());
  reply({ ...thread, items: [{ kind: "user", text: "Fix it" }, { kind: "command", text: "Ran pytest" }, { kind: "assistant", text: "Done." }], items_truncated: false });
  nodes(render(), "button").find(node => /Fix the parser/.test(text(node))).props.onClick();
  await settle();
  assert.equal(calls.at(-1)[0], "/computer/coding/threads/thread-a");
  const items = nodes(render(), "li").map(text);
  assert.deepEqual(items, ["Fix it", "Ran pytest", "Done."]);
  reply(status([{ ...thread, running: false }]));
  reply({ ...thread, running: false, items: [], items_truncated: false });
  await interval();
  assert.equal(calls.at(-1)[0], "/computer/coding/threads/thread-a");
  assert.equal(button("Stop"), undefined);
  button("All threads").props.onClick();
  assert.match(text(render()), /Fix the parser/);
});

test("a running thread can be stopped from the tracker", async () => {
  await open();
  reply(status());
  reply({ ...thread, items: [], items_truncated: false });
  nodes(render(), "button").find(node => /Fix the parser/.test(text(node))).props.onClick();
  await settle();
  reply({ thread_id: "thread-a", running: false });
  reply(status([{ ...thread, running: false }]));
  reply({ ...thread, running: false, items: [], items_truncated: false });
  button("Stop").props.onClick();
  await settle();
  assert.deepEqual(calls.find(call => call[0].endsWith("/stop")), ["/computer/coding/threads/thread-a/stop", {}]);
  assert.match(text(render()), /Idle/);
});

test("a failed refresh reports the error and Close closes", async () => {
  replies.push(() => Promise.reject(new Error("Computer unavailable")));
  render();
  await settle();
  assert.match(text(render()), /Codex status unavailable/);
  assert.match(text(render()), /Computer unavailable/);
  button("Close").props.onClick();
  assert.equal(closed, 1);
});
