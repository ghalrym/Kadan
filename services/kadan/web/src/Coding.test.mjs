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
  export class ApiError extends Error { constructor(message, status) { super(message); this.status = status; } }
  export const calls = [];
  export const replies = [];
  export const authReplies = [];
  export const agentPath = key => '/agents/' + encodeURIComponent(key);
  export const api = (...args) => {
    calls.push(args);
    if (['/computer/coding/login/status', '/computer/coding/login', '/computer/coding/logout'].includes(args[0])) {
      return authReplies.length ? authReplies.shift()() : Promise.resolve({ state: 'signed_out', message: 'Sign in with ChatGPT.' });
    }
    return replies.shift()();
  };
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
const { calls, replies, authReplies, ApiError } = await import(apiUrl);
const { default: Coding } = await import(componentUrl);
loader.deregister();


let closed;
let queued;
const conversation = { id: "conversation-a", agent_key: "agent-a", version: 1, messages: [] };
const storage = new Map();
const originalStorage = globalThis.sessionStorage;
let interval;
const originalInterval = globalThis.setInterval;
const originalClear = globalThis.clearInterval;
let currentConversation;
function render() { hooks.render(); return Coding({ agentKey: "agent-a", conversation: currentConversation, onQueued: value => queued.push(value), onClose: () => closed++ }); }
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
  hooks.reset(); calls.length = replies.length = authReplies.length = 0; closed = 0; queued = []; storage.clear();
  currentConversation = conversation;
  globalThis.sessionStorage = { getItem: key => storage.get(key) ?? null, setItem: (key, value) => storage.set(key, value), removeItem: key => storage.delete(key) };
  globalThis.setInterval = fn => { interval = fn; return 1; };
  globalThis.clearInterval = () => { interval = undefined; };
});
afterEach(() => {
  for (const cleanup of hooks.cleanups) cleanup?.();
  globalThis.setInterval = originalInterval;
  globalThis.clearInterval = originalClear;
  globalThis.sessionStorage = originalStorage;
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
  assert.ok(button("Sign in with ChatGPT"));
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
  assert.deepEqual(calls.find(([path]) => path.endsWith("/cancel")), ["/computer/coding/job-a/cancel", {}]);
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


test("device sign-in is available without implying model access", async () => {
  await open(status([], { state: "auth_required", message: "Sign in on this computer." }));
  assert.ok(button("Sign in with ChatGPT"));
  assert.match(text(render()), /Keep passwords, codes and API keys out of chat/);
  assert.match(text(render()), /Finish the setup above/);
  assert.doesNotMatch(text(render()), /For a first try/);
  assert.match(text(render()), /Sign-in does not confirm access to GPT-6.1 Sol or remaining usage/);
  assert.ok(calls.every(call => call[1] === undefined));
});


test("update setup does not offer sign-in as a fix for a missing or incompatible package", async () => {
  authReplies.push(() => Promise.resolve({ state: "missing", message: "Update the computer." }));
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

const completed = { ...first, status: "completed", message: "Earlier turn completed." };
const pendingKey = "kadan.draft.agent-a.conversation-a.pending";
function followup(value) { nodes(render(), "textarea")[0].props.onChange({ target: { value } }); }
function submit() { nodes(render(), "form")[0].props.onSubmit({ preventDefault() {} }); }

test("explicit continuation queues a scoped durable request and shows its allowance impact", async () => {
  await open(status([completed]));
  followup("Rerun the unit tests");
  assert.match(text(render()), /queues this task directly in Messages/);
  assert.match(text(render()), /uses your Codex allowance/);
  replies.push(() => Promise.resolve(conversation), () => Promise.resolve({ conversation: { ...conversation, version: 2 } }));
  submit(); submit();
  await settle();
  assert.ok(calls.some(([path]) => path === "/agents/agent-a/resources/conversation-a"));
  const posts = calls.filter(call => call[1]);
  assert.equal(posts.length, 1);
  assert.equal(posts[0][0], "/agents/agent-a/conversations/conversation-a/runs");
  assert.equal(posts[0][1].resume_coding_job_id, completed.id);
  assert.equal(posts[0][1].content, "Rerun the unit tests");
  assert.equal(posts[0][1].expected_version, 1);
  assert.equal(posts[0][1].persona_id, null);
  assert.equal(closed, 1); assert.equal(queued.length, 1);
  assert.equal(storage.has(pendingKey), false);
});

test("uncertain continuation retries the frozen ID, job and prompt rather than submitting new work", async () => {
  await open(status([completed])); followup("Check the fix");
  replies.push(() => Promise.resolve(conversation), () => Promise.reject(new Error("Response lost")));
  submit(); await settle();
  const firstPost = calls.find(call => call[1])[1];
  assert.match(text(render()), /Continuation is unconfirmed/);
  assert.equal(nodes(render(), "textarea")[0].props.disabled, true);
  assert.equal(nodes(render(), "select")[0].props.disabled, true);
  assert.equal(JSON.parse(storage.get(pendingKey)).id, firstPost.message_id);
  replies.push(() => Promise.resolve({ ...conversation, version: 2 }), () => Promise.resolve({ conversation }));
  submit(); await settle();
  const lastPost = calls.filter(call => call[1]).at(-1)[1];
  assert.deepEqual(lastPost, { ...firstPost, expected_version: 2 });
  assert.equal(storage.has(pendingKey), false);
});

test("reopening restores only this conversation's unconfirmed continuation", async () => {
  const pending = { id: "same-request", content: "Check only", generate: true, persona: "", resume_coding_job_id: completed.id };
  storage.set(pendingKey, JSON.stringify(pending));
  await open(status([completed]));
  assert.equal(nodes(render(), "textarea")[0].props.value, pending.content);
  assert.equal(button("Retry same continuation").props.disabled, false);
  replies.push(() => Promise.resolve(conversation), () => Promise.resolve({ conversation }));
  submit(); await settle();
  assert.equal(calls.find(call => call[1])[1].message_id, pending.id);
});

test("closing during preflight prevents a late write or a callback into another conversation", async () => {
  await open(status([completed])); followup("Check only");
  let finish;
  replies.push(() => new Promise(resolve => { finish = resolve; }));
  submit(); button("Close").props.onClick();
  finish(conversation); await settle();
  assert.equal(calls.filter(call => call[1]).length, 0);
  assert.equal(queued.length, 0); assert.equal(closed, 1);
  assert.equal(JSON.parse(storage.get(pendingKey)).resume_coding_job_id, completed.id);
});

test("a mismatched conversation cannot accept a continuation or clear its recovery", async () => {
  await open(status([completed])); followup("Check only");
  replies.push(() => Promise.resolve({ ...conversation, id: "other" }));
  submit(); await settle();
  assert.equal(calls.filter(call => call[1]).length, 0);
  assert.equal(queued.length, 0); assert.ok(storage.has(pendingKey));
  assert.match(text(render()), /conversation changed/);
});

test("unrelated pending chat is preserved and blocks a continuation", async () => {
  const pending = { id: "ordinary-chat", content: "Existing work", generate: true, persona: "" };
  storage.set(pendingKey, JSON.stringify(pending));
  await open(status([completed])); followup("Check only"); submit(); await settle();
  assert.deepEqual(calls.map(([path]) => path), ["/computer/coding", "/computer/coding/login/status"]);
  assert.deepEqual(JSON.parse(storage.get(pendingKey)), pending);
  assert.equal(queued.length, 0);
});

test("a new continuation is disabled while another job or unconfirmed cleanup owns the computer", async () => {
  await open(status([completed, { ...first, id: "other", status: "cleanup_required" }]));
  followup("Check only");
  assert.equal(button("Continue with Codex").props.disabled, true);
  assert.equal(calls.filter(call => call[1]).length, 0);
});

test("jobs without a recorded session do not offer a fresh-session fallback", async () => {
  await open(status([{ ...completed, session_id: null }]));
  assert.equal(nodes(render(), "form").length, 0);
  assert.equal(button("Continue with Codex"), undefined);
});

test("unsent follow-ups survive closing, Escape and reopening without crossing conversations", async () => {
  await open(status([completed])); followup("Keep my draft");
  nodes(render(), "dialog")[0].props.onCancel({ preventDefault() {} });
  for (const cleanup of hooks.cleanups) cleanup?.();
  hooks.reset();
  await open(status([completed]));
  assert.equal(nodes(render(), "textarea")[0].props.value, "Keep my draft");
  button("Close").props.onClick();
  for (const cleanup of hooks.cleanups) cleanup?.();
  hooks.reset(); currentConversation = { ...conversation, id: "conversation-b" };
  await open(status([completed]));
  assert.equal(nodes(render(), "textarea")[0].props.value, "");
  assert.equal(calls.filter(call => call[1]).length, 0);
});

test("switching jobs preserves separate drafts and queuing moves only the selected draft to recovery", async () => {
  const other = { ...completed, id: "job-b" };
  await open(status([completed, other])); followup("First draft");
  nodes(render(), "select")[0].props.onChange({ target: { value: other.id } });
  assert.equal(nodes(render(), "textarea")[0].props.value, "");
  followup("Second draft");
  nodes(render(), "select")[0].props.onChange({ target: { value: completed.id } });
  assert.equal(nodes(render(), "textarea")[0].props.value, "First draft");
  replies.push(() => Promise.resolve(conversation), () => Promise.reject(new Error("Response lost")));
  submit(); await settle();
  assert.equal(JSON.parse(storage.get(pendingKey)).content, "First draft");
  assert.deepEqual(JSON.parse(storage.get(`${pendingKey}.coding-drafts`)), { "job-b": "Second draft" });
});


test("unavailable login helper preserves job refresh and stop controls", async () => {
  authReplies.push(() => Promise.reject(new Error("Missing helper")));
  await open();
  assert.match(text(render()), /Sign-in status unavailable/);
  assert.match(text(render()), /Working/);
  assert.equal(button("Stop coding").props.disabled, false);
  replies.push(() => Promise.resolve(status([{ ...first, status: "completed", message: "Done." }])));
  authReplies.push(() => Promise.reject(new Error("Missing helper")));
  interval();
  await settle();
  assert.match(text(render()), /Done/);
  assert.doesNotMatch(text(render()), /Last known state/);
});

test("pending sign-in can be cancelled without cancelling a coding job", async () => {
  authReplies.push(() => Promise.resolve({ state: "signing_in", message: "Enter code", login: {url: "https://example.com", code: "ABCD-EFGH"} }));
  await open();
  authReplies.push(() => Promise.resolve({ state: "signed_out", message: "Signed out" }));
  replies.push(() => Promise.resolve(status()));
  button("Cancel sign-in").props.onClick();
  await settle();
  assert.ok(calls.some(([path, body]) => path === "/computer/coding/logout" && body));
  assert.ok(!calls.some(([path]) => path.endsWith("/cancel")));
  assert.ok(button("Sign in with ChatGPT"));
});
