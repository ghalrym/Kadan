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
    if (!(slot in slots)) slots[slot] = initial;
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
  export class ApiError extends Error { constructor(message, status) { super(message); this.status = status; } }
  export const calls = [];
  export const replies = [];
  export const agentPath = key => '/agents/' + encodeURIComponent(key);
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
const { calls, replies, ApiError } = await import(apiUrl);
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
function button(label) { return nodes(render(), "button").find(node => node.props.children === label); }
async function settle() { await new Promise(resolve => setImmediate(resolve)); }
const first = { id: "job-a", status: "running", cwd: "/home/gem/project", session_id: "session-a", created_at: "2026-10-30T12:00:00Z", message: "Working.", events: [] };
function status(jobs = [first], setup = { state: "ready", message: "Credentials present; execution unverified." }) { return { setup, jobs }; }
async function open(result = status()) { replies.push(() => Promise.resolve(result)); render(); await settle(); }
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

test("setup failure is not an empty job history and refresh can recover", async () => {
  replies.push(() => Promise.reject(new Error("Computer unavailable")));
  assert.match(text(render()), /Checking computer/);
  await settle();
  assert.match(text(render()), /Coding status unavailable/);
  assert.doesNotMatch(text(render()), /No coding jobs/);
  replies.push(() => Promise.resolve(status([], { state: "auth_required", message: "Sign in inside the computer." })));
  button("Refresh coding jobs").props.onClick();
  await settle();
  assert.match(text(render()), /codex login --device-auth/);
  assert.match(text(render()), /No coding jobs/);
  assert.doesNotMatch(text(render()), /Could not refresh/);
});

test("failed reconnect retains clearly stale job state without resubmission", async () => {
  await open();
  replies.push(() => Promise.reject(new Error("Disconnected")));
  interval();
  await settle();
  assert.match(text(render()), /Last known state: Working/);
  assert.match(text(render()), /Could not refresh coding jobs/);
  assert.ok(calls.every(call => call[1] === undefined));
  replies.push(() => Promise.resolve(status([{ ...first, status: "completed", message: "Done." }])));
  interval();
  await settle();
  assert.match(text(render()), /Completed/);
  assert.doesNotMatch(text(render()), /Last known state/);
});

test("stopping remains pending until a later cleanup confirmation", async () => {
  await open();
  replies.push(() => Promise.resolve({ ...first, status: "stopping", message: "Cleaning up." }));
  button("Stop coding").props.onClick();
  await settle();
  assert.deepEqual(calls[1], ["/computer/coding/job-a/cancel", {}]);
  assert.match(text(render()), /Stopping and cleaning up/);
  assert.equal(button("Waiting for cleanup…").props.disabled, true);
  replies.push(() => Promise.resolve(status([{ ...first, status: "cancelled", message: "Cancelled; changes remain." }])));
  interval();
  await settle();
  assert.match(text(render()), /Cancelled; changes remain/);
  assert.equal(button("Stop coding"), undefined);
});

test("a late polling response cannot overwrite cancellation and does not delay stop", async () => {
  await open();
  let finish;
  replies.push(() => new Promise(resolve => { finish = resolve; }));
  interval();
  replies.push(() => Promise.resolve({ ...first, status: "stopping", message: "Stopping." }));
  button("Stop coding").props.onClick();
  button("Requesting stop…").props.onClick();
  await settle();
  finish(status());
  await settle();
  assert.match(text(render()), /Stopping and cleaning up/);
  assert.equal(calls.filter(call => call[1]).length, 1);
});

test("an uncertain cancellation is read back without replaying the write", async () => {
  await open();
  replies.push(() => Promise.reject(new Error("Response lost")));
  button("Stop coding").props.onClick();
  await settle();
  assert.match(text(render()), /Stop coding is unconfirmed/);
  replies.push(() => Promise.resolve(status([{ ...first, status: "cancelled", message: "Cancelled." }])));
  interval();
  await settle();
  assert.match(text(render()), /Cancelled/);
  assert.equal(calls.filter(call => call[1]).length, 1);
});

test("closing while a request is pending neither cancels work nor applies a late response", async () => {
  await open();
  let finish;
  replies.push(() => new Promise(resolve => { finish = resolve; }));
  interval();
  button("Close").props.onClick();
  assert.equal(closed, 1);
  finish(status([{ ...first, status: "completed" }]));
  await settle();
  assert.match(text(render()), /Working/);
  assert.equal(calls.filter(call => call[1]).length, 0);
});

test("cleanup failure remains attention-needed and explicit session identity is visible", async () => {
  await open(status([{ ...first, status: "cleanup_required", message: "Restart the computer to confirm cleanup.", events_truncated: true } ]));
  assert.match(text(render()), /Cleanup needs attention/);
  assert.match(text(render()), /session-a/);
  assert.match(text(render()), /Earlier events have been omitted/);
  assert.equal(button("Stop coding"), undefined);
});


test("ready first use suggests an existing project and a read-only task without submitting", async () => {
  await open(status([]));
  assert.match(text(render()), /Choose an existing project folder/);
  assert.match(text(render()), /Ask Codex to look through my project/);
  assert.match(text(render()), /without changing any files/);
  assert.match(text(render()), /Replace \[folder\] with its path/);
  assert.ok(calls.every(call => call[1] === undefined));
});


test("sign-in guidance locates the terminal and does not treat ChatGPT browser login as Codex access", async () => {
  await open(status([], { state: "auth_required", message: "Sign in on this computer." }));
  assert.match(text(render()), /open your agent’s card, open its computer, and choose Take control/);
  assert.match(text(render()), /CODEX_HOME=\/home\/gem\/\.codex codex login --device-auth/);
  assert.match(text(render()), /Keep passwords, codes and API keys out of chat/);
  assert.match(text(render()), /Finish the setup above/);
  assert.doesNotMatch(text(render()), /For a first try/);
  const help = nodes(render(), "a").find(link => link.props.href === "https://developers.openai.com/codex/auth");
  assert.equal(help.props.target, "_blank");
  assert.match(text(help), /opens in a new tab/);
  assert.match(text(render()), /Sign-in does not confirm access to GPT-6.1 Sol or remaining usage/);
  assert.match(text(render()), /API-key use can incur charges/);
  assert.ok(calls.every(call => call[1] === undefined));
});


test("update setup does not offer sign-in as a fix for a missing or incompatible package", async () => {
  await open(status([], { state: "unavailable", message: "Ask the person who runs Kadan to update its computer." }));
  assert.match(text(render()), /update its computer/);
  assert.doesNotMatch(text(render()), /codex login --device-auth/);
  assert.doesNotMatch(text(render()), /For a first try/);
});


test("the project is visible before stopping and state changes have a polite announcement", async () => {
  await open();
  assert.match(text(render()), /Project folder: \/home\/gem\/project/);
  const state = nodes(render(), "p").find(node => node.props.className === "coding-state");
  assert.equal(state.props.role, "status");
  assert.match(text(render()), /Stop response in Messages prevents that reply and leaves coding running/);
  assert.match(text(render()), /Stop coding ends the coding task; file changes remain/);
});
