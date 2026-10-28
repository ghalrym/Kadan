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
const componentUrl = new URL("./RunnerAssignment.tsx", import.meta.url).href;
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
const { default: RunnerAssignment } = await import(componentUrl);
loader.deregister();

const agent = { key: "ember", name: "Ember" };
const first = { runner_id: "runner-1", capacity: 5, agent_keys: ["ember"], lease_expires_at: "2026-10-22T12:01:00Z" };
const second = { runner_id: "runner-2", capacity: 5, agent_keys: [], lease_expires_at: first.lease_expires_at };
const originalNow = Date.now;
let now;
let closed;
function render() {
  hooks.render();
  return RunnerAssignment({ agent, onClose: () => closed++ });
}
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
function choose(id) { nodes(render(), "select")[0].props.onChange({ target: { value: id } }); }
function save() { nodes(render(), "form")[0].props.onSubmit({ preventDefault() {} }); }
async function settle() { await new Promise(resolve => setImmediate(resolve)); }
async function open(rows) {
  replies.push(() => Promise.resolve(rows));
  render();
  await settle();
}
beforeEach(() => {
  hooks.reset(); calls.length = replies.length = 0; closed = 0;
  now = Date.parse("2026-10-22T12:00:00Z");
  Date.now = () => now;
});
afterEach(() => { Date.now = originalNow; });

test("loading and failed reads do not imply the agent is unassigned", async () => {
  let fail;
  replies.push(() => new Promise((resolve, reject) => { fail = reject; }));
  assert.match(text(render()), /Loading…/);
  assert.doesNotMatch(text(render()), /Unassigned/);
  fail(new Error("Storage unavailable"));
  await settle();
  assert.match(text(render()), /Unavailable/);
  assert.doesNotMatch(text(render()), /Unassigned/);
  replies.push(() => Promise.resolve([]));
  button("Refresh assignments").props.onClick();
  await settle();
  assert.match(text(render()), /Unassigned/);
  assert.match(text(render()), /No runners registered/);
});

test("current owner and occupancy remain visible after heartbeat expiry, and unavailable destinations cannot be selected", async () => {
  await open([
    { ...first, lease_expires_at: "2026-10-22T11:59:00Z" },
    { ...second, capacity: 1, agent_keys: ["another"] },
    { ...second, runner_id: "expired", lease_expires_at: "2026-10-22T11:59:00Z" },
    { ...second, runner_id: "available" },
  ]);
  assert.match(text(render()), /runner-1Heartbeat at last refreshExpiredAssigned agents1 \/ 5/);
  const options = nodes(render(), "option");
  for (const id of ["runner-1", "runner-2", "expired"]) {
    assert.equal(options.find(option => option.props.value === id).props.disabled, true);
  }
  assert.equal(options.find(option => option.props.value === "available").props.disabled, false);
});

test("assignment compares the displayed owner and refreshes confirmed state", async () => {
  await open([first, second]);
  choose("runner-2");
  replies.push(() => Promise.resolve(undefined), () => Promise.resolve([
    { ...first, agent_keys: [] }, { ...second, agent_keys: ["ember"] },
  ]));
  save();
  await settle();
  assert.deepEqual(calls[1], ["/agents/ember/runner", { runner_id: "runner-2", expected_runner_id: "runner-1" }]);
  assert.deepEqual(calls[2], ["/runners"]);
  assert.match(text(render()), /Assigned runnerrunner-2/);
  assert.match(text(render()), /Assignment saved/);
  assert.equal(button("Assign").props.disabled, true);
});

test("assigning an unassigned agent sends an explicit null expected owner", async () => {
  await open([second]);
  choose("runner-2");
  replies.push(() => Promise.resolve(undefined), () => Promise.resolve([{ ...second, agent_keys: ["ember"] }]));
  save();
  await settle();
  assert.deepEqual(calls[1][1], { runner_id: "runner-2", expected_runner_id: null });
});

test("release shows a fresh automatic assignment instead of pretending the agent remains unassigned", async () => {
  await open([first, second]);
  replies.push(() => Promise.resolve(undefined), () => Promise.resolve([first, second]));
  button("Release assignment").props.onClick();
  await settle();
  assert.deepEqual(calls[1][1], { runner_id: null, expected_runner_id: "runner-1" });
  assert.match(text(render()), /Assigned runnerrunner-1/);
  assert.match(text(render()), /Assignment released. An available runner may assign this agent again/);
});

test("conflicts retain the destination and require refresh before retrying with the new owner", async () => {
  await open([first, second]);
  choose("runner-2");
  replies.push(() => Promise.reject(new ApiError("Agent assignment changed", 409)));
  save();
  await settle();
  assert.equal(nodes(render(), "select")[0].props.value, "runner-2");
  assert.equal(button("Assign").props.disabled, true);
  save();
  assert.equal(calls.length, 2);
  assert.match(text(render()), /Last known assignment/);
  replies.push(() => Promise.resolve([{ ...first, agent_keys: [] }, second]));
  button("Refresh assignments").props.onClick();
  await settle();
  replies.push(() => Promise.resolve(undefined), () => Promise.resolve([{ ...second, agent_keys: ["ember"] }]));
  save();
  await settle();
  assert.deepEqual(calls[3][1], { runner_id: "runner-2", expected_runner_id: null });
});

test("uncertain saves can be recovered by reading the already committed assignment", async () => {
  await open([first, second]);
  choose("runner-2");
  replies.push(() => Promise.reject(new Error("Response lost")));
  save();
  await settle();
  assert.match(text(render()), /may have been saved/);
  replies.push(() => Promise.resolve([{ ...second, agent_keys: ["ember"] }]));
  button("Refresh assignments").props.onClick();
  await settle();
  assert.match(text(render()), /Assigned runnerrunner-2/);
  assert.equal(button("Assign").props.disabled, true);
  assert.equal(calls.filter(call => call[1]).length, 1);
});

test("a failed read after confirmed mutation preserves success and blocks further writes", async () => {
  await open([first, second]);
  choose("runner-2");
  replies.push(() => Promise.resolve(undefined), () => Promise.reject(new Error("Offline")));
  save();
  await settle();
  assert.match(text(render()), /Assignment saved/);
  assert.match(text(render()), /Could not refresh the saved assignment/);
  assert.equal(button("Assign").props.disabled, true);
  assert.equal(button("Release assignment").props.disabled, true);
});

test("lease expiry since the last refresh prevents submitting a stale destination", async () => {
  await open([first, second]);
  choose("runner-2");
  now += 61000;
  save();
  assert.equal(calls.length, 1);
  assert.match(text(render()), /current heartbeat and a free slot/);
});

test("in-flight mutations block duplicate requests and closing, and late unmounted responses do not refresh", async () => {
  await open([first, second]);
  choose("runner-2");
  let finish;
  replies.push(() => new Promise(resolve => { finish = resolve; }));
  save();
  save();
  button("Close").props.onClick();
  assert.equal(calls.length, 2);
  assert.equal(closed, 0);
  for (const cleanup of hooks.cleanups) cleanup?.();
  finish(undefined);
  await settle();
  assert.equal(calls.length, 2);
  assert.doesNotMatch(text(render()), /Assignment saved/);
});

test("closing during an initial read ignores its late result", async () => {
  let finish;
  replies.push(() => new Promise(resolve => { finish = resolve; }));
  render();
  button("Close").props.onClick();
  assert.equal(closed, 1);
  finish([first]);
  await settle();
  assert.doesNotMatch(text(render()), /runner-1/);
});
