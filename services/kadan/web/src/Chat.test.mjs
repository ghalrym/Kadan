import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import ts from "typescript";
import * as outbox from "./outbox.ts";
import * as agentRuns from "./agentRuns.ts";

// Execute only Chat's function and event handlers. Child components, effects and
// service requests are boundaries; this never mounts React or starts polling.
const source = ts.transpileModule(readFileSync(new URL("./Chat.tsx", import.meta.url), "utf8"), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, target: ts.ScriptTarget.ES2022 },
}).outputText;
const tick = () => new Promise(resolve => setImmediate(resolve));
function harness({ pending, unresolved = false, offline = false } = {}) {
  const states = [], refs = [], elements = [], requests = [], changes = [];
  let stateIndex = 0, refIndex = 0;
  const data = new Map([["kadan.draft.agent", "Review this file"]]);
  if (pending) data.set("kadan.draft.agent.chat.pending", JSON.stringify(pending));
  const storage = { getItem: key => data.get(key) ?? null, setItem: (key, value) => data.set(key, value), removeItem: key => data.delete(key) };
  const react = {
    Fragment: "Fragment", useEffect() {}, useLayoutEffect() {}, useCallback: fn => fn,
    useState(initial) {
      const index = stateIndex++;
      if (!(index in states)) states[index] = typeof initial === "function" ? initial() : initial;
      return [states[index], next => { states[index] = typeof next === "function" ? next(states[index]) : next; }];
    },
    useRef(initial) { const index = refIndex++; return refs[index] ??= { current: initial }; },
  };
  const jsx = (type, props, key) => { const element = { type, props, key }; elements.push(element); return element; };
  const selected = { ready: !unresolved, entries: [{ id: "file-b" }, { id: "file-a" }],
    ids() { if (!this.ready) throw new Error("Unresolved upload"); return this.entries.map(item => item.id); },
    removeMany(ids) { this.entries = this.entries.filter(item => !ids.includes(item.id)); },
  };
  const mocks = {
    react, "react/jsx-runtime": { jsx, jsxs: jsx }, "lucide-react": { ArrowUpRight: "Icon", RotateCw: "Icon" },
    "./outbox": outbox, "./agentRuns": agentRuns,
    "./api": { agentPath: key => `/agents/${key}`, api: (path, body) => new Promise((resolve, reject) => requests.push({ path, body, resolve, reject })) },
    "./uuid": { randomUuid: () => "new-message" },
    "./attachments": { ChatAttachments: class { constructor() { return selected; } } },
    "./Composer": { default: "Composer", readiness: status => ({ storage: status.tsuchi.state === "ready", chat: status.hikari.state === "ready" }) },
    "./Pet": { default: "Pet", useMood: () => ["idle", () => {}] },
    "./VoiceCallControl": { useVoiceCall: () => ({ active: false }) },
    "./presence": { activeLabel: () => "Here" }, "./PetContext": { useAgentPet: () => null },
    ...Object.fromEntries(["Markdown", "AgentCard", "ComputerViewer"].map(name => [`./${name}`, { default: name }])),
  };
  const exports = {};
  new Function("require", "exports", "sessionStorage", source)(name => {
    assert.ok(name in mocks, `Unexpected dependency ${name}`);
    return mocks[name];
  }, exports, storage);
  const chain = { id: "chat", agent_key: "agent", version: 1, messages: [] };
  const props = { agentKey: "agent", agentName: "Agent", chain, personas: [], entrance: "idle", blocked: false,
    status: { tsuchi: { state: "ready" }, hikari: { state: offline ? "offline" : "ready" } },
    onChange: value => changes.push(value), onBusy() {}, onEditPet() {}, refreshStatus: async () => {},
  };
  function render() {
    stateIndex = refIndex = 0; elements.length = 0;
    exports.default(props);
    return elements.find(item => item.type === "Composer").props;
  }
  return { render, requests, data, selected, states, elements, chain, changes };
}

test("notes submit ordered attachment IDs without Hikari and show them optimistically", async () => {
  const h = harness({ offline: true });
  h.render().onSend(false);
  assert.equal(h.requests[0].path, "/agents/agent/conversations/chat/notes");
  assert.deepEqual(h.requests[0].body.attachment_ids, ["file-b", "file-a"]);
  assert.deepEqual(h.states.find(value => value?.role === "user").attachment_ids, ["file-b", "file-a"]);
  assert.deepEqual(JSON.parse(h.data.get("kadan.draft.agent.chat.pending")).attachment_ids, ["file-b", "file-a"]);
  assert.deepEqual(h.selected.entries, []);
  h.requests[0].resolve({ ...h.chain, version: 2, messages: [{ id: "new-message" }] });
  await tick();
  assert.equal(h.data.has("kadan.draft.agent.chat.pending"), false);
});

test("run submission preserves attachments after failure and retry ignores newer draft attachments", async () => {
  const h = harness();
  h.render().onSend(true);
  assert.equal(h.requests[0].path, "/agents/agent/conversations/chat/runs");
  assert.deepEqual(h.requests[0].body.attachment_ids, ["file-b", "file-a"]);
  h.requests[0].reject(new Error("Lost response"));
  await tick();
  h.requests[1].resolve(h.chain);
  await tick();
  h.selected.entries = [{ id: "new-file" }];
  h.selected.ready = false;
  h.render().onChange("New draft");
  h.render();
  assert.equal(h.elements.filter(item => typeof item.type === "function" && item.type.name === "Attachment").length, 2);
  h.elements.find(item => item.type === "button" && item.props.children === "Retry").props.onClick();
  assert.deepEqual(h.requests[2].body, h.requests[0].body);
  assert.deepEqual(h.selected.entries, [{ id: "new-file" }]);
  assert.equal(h.render().value, "New draft");
});

test("reload retries the pending attachment snapshot and still supports old text-only submissions", () => {
  for (const attachment_ids of [undefined, ["saved-b", "saved-a"]]) {
    const pending = { id: "saved-message", content: "Earlier message", generate: false, persona: "", ...(attachment_ids && { attachment_ids }) };
    const h = harness({ pending });
    h.render();
    h.elements.find(item => item.type === "button" && item.props.children === "Retry").props.onClick();
    assert.equal(h.requests[0].body.message_id, "saved-message");
    assert.deepEqual(h.requests[0].body.attachment_ids, attachment_ids ?? []);
    assert.equal(h.render().value, "Review this file");
    assert.deepEqual(h.selected.entries, [{ id: "file-b" }, { id: "file-a" }]);
  }
});

test("new sends with unresolved attachments or blank text cannot submit", () => {
  const h = harness({ unresolved: true });
  h.render().onSend(false);
  assert.equal(h.requests.length, 0);
  assert.match(h.states.find(value => typeof value === "string" && value.startsWith("Finish uploading")), /unresolved attachments/);
  h.selected.ready = true;
  h.render().onChange("   ");
  h.render().onSend(true);
  assert.equal(h.requests.length, 0);
});
