import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { runInNewContext } from "node:vm";
import ts from "typescript";
import * as agentRuns from "./agentRuns.ts";
import * as outbox from "./outbox.ts";

const source = ts.transpileModule(
  readFileSync(new URL("./Chat.tsx", import.meta.url), "utf8"),
  { compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX } },
).outputText;
const chain = { id: "conversation", version: 1, messages: [] };
const message = (id, role, content) => ({
  id, role, content, created_at: "2026-10-23T10:00:00Z", resource_ids: [], attachment_ids: [], sources: [],
});
const run = (id, status, error = null) => ({
  id, status, error, conversation_id: chain.id, created_at: "2026-10-23T10:00:00Z",
});
const settle = () => new Promise(resolve => setImmediate(resolve));

// Exercise Chat's poll, send and Stop handlers with in-memory hooks and API boundaries.
async function chat(initialRuns, initialConversation = chain, storedPending = null, callOptions = {}) {
  const slots = [], effects = [], timers = [], requests = [], callStarts = [];
  let callDisabled;
  const storage = new Map();
  if (storedPending) storage.set(outbox.chatStorageKeys("guide", initialConversation.id).pending, JSON.stringify(storedPending));
  const attachments = { ready: true, entries: [], start() {}, stop() {},
    ids() { return this.entries.map(item => item.id); },
    removeMany(ids) { this.entries = this.entries.filter(item => !ids.includes(item.id)); },
  };
  let cursor = 0, tree, snapshotRuns = initialRuns, sendResponse, replacementResponse;
  let selectedChain = initialConversation;
  const react = {
    Fragment: "Fragment",
    useState(initial) {
      const index = cursor++;
      if (!(index in slots)) slots[index] = typeof initial === "function" ? initial() : initial;
      return [slots[index], value => {
        slots[index] = typeof value === "function" ? value(slots[index]) : value;
      }];
    },
    useRef(initial) {
      const index = cursor++;
      return slots[index] ??= { current: initial };
    },
    useCallback: callback => callback,
    useEffect(effect, dependencies) {
      const index = cursor++;
      const previous = slots[index];
      if (!previous || dependencies.some((value, i) => value !== previous.dependencies[i])) {
        const slot = slots[index] = { dependencies };
        effects.push(() => {
          previous?.cleanup?.();
          slot.cleanup = effect();
        });
      }
    },
    useLayoutEffect() {},
  };
  const jsx = (type, props) => ({ type, props });
  const modules = {
    react,
    "react/jsx-runtime": { jsx, jsxs: jsx },
    "lucide-react": {},
    "./RequestReferences": { default: "RequestReferences" },
    "./Markdown": { default: "Markdown" },
    "./Composer": { default: "Composer", readiness: () => ({ chat: true, storage: true, ...callOptions.ready }) },
    "./Pet": { default: "Pet", useMood: () => ["idle", () => {}] },
    "./AgentCard": { default: "AgentCard" },
    "./ComputerViewer": { default: "ComputerViewer" },
    "./ChatExport": { default: "ChatExport" },
    "./Activity": { default: "Activity" },
    "./VoiceCallControl": { useVoiceCall: (_agent, _persona, disabled) => {
      callDisabled = disabled;
      return { active: false, unavailable: null, ...callOptions.voiceCall, start: (...args) => callStarts.push(args) };
    } },
    "./PetContext": { useAgentPet: () => null },
    "./presence": { activeLabel: () => "Not active yet" },
    "./agentRuns": agentRuns,
    "./outbox": outbox,
    "./attachments": { ChatAttachments: class { constructor() { return attachments; } } },
    "./uuid": { randomUuid: () => "sent-message" },
    "./SaveChatResult": { default: "SaveChatResult" },
    "./api": {
      agentPath: key => `/agents/${key}`,
      async api(path, body) {
        requests.push({ path, body });
        if (path === `/agents/guide/conversations/${selectedChain.id}/runs`) {
          if (body) return sendResponse;
          if (replacementResponse) return replacementResponse;
          return { runs: snapshotRuns, images: [], conversation: selectedChain };
        }
        if (path === `/agents/guide/conversations/${selectedChain.id}/notes`) return sendResponse;
        if (path.endsWith("/cancel")) {
          const id = path.split("/").at(-2);
          return { ...snapshotRuns.find(item => item.id === id), status: "cancelled" };
        }
        assert.fail(`Unexpected API call: ${path}`);
      },
    },
  };
  const exports = {};
  runInNewContext(source, {
    exports,
    ResizeObserver: class { observe() {} disconnect() {} },
    require: name => { assert.ok(name in modules, name); return modules[name]; },
    sessionStorage: {
      getItem: key => storage.get(key) ?? null,
      setItem: (key, value) => storage.set(key, value),
      removeItem: key => storage.delete(key),
    },
    setTimeout: callback => timers.push(callback), clearTimeout() {},
    setInterval: () => 0, clearInterval() {},
  });
  function render() {
    cursor = 0;
    tree = exports.default({
      agentKey: "guide", agentName: "Guide", chain: selectedChain, personas: [], status: null,
      entrance: "idle", onChange(value) { selectedChain = value; }, onBusy() {}, onEditPet() {}, blocked: false,
      refreshStatus: async () => {},
    });
    effects.splice(0).forEach(effect => effect());
  }
  function elements(node = tree) {
    if (!node || typeof node !== "object") return [];
    return [node, ...[node.props?.children].flat(Infinity).flatMap(child => elements(child ?? null))];
  }
  render();
  await settle();
  render();
  return {
    render, requests, storage, attachments, callStarts,
    callDisabled: () => callDisabled,
    card() {
      elements().find(node => node.props?.["aria-label"] === "Open Guide’s card").props.onClick();
      render();
      return elements().find(node => node.type === "AgentCard").props;
    },
    attachmentIds: () => elements().filter(node => typeof node.type === "function" && node.type.name === "Attachment").map(node => node.props.id),
    savedResults: () => elements().filter(node => node.type === "SaveChatResult").map(node => node.props),
    retry: () => elements().find(node => node.type === "button" && node.props.children === "Retry").props.onClick(),
    notices: () => elements().filter(node => node.props?.className === "notice").map(node => node.props.children),
    alerts: () => elements().filter(node => node.props?.role === "alert").map(node => node.props.children[0]),
    receipts: () => elements().filter(node => node.props?.className === "receipt").map(node => node.props.children),
    references: () => elements().find(node => node.type === "RequestReferences").props,
    composer: () => elements().find(node => node.type === "Composer").props,
    activity() {
      if (!elements().some(node => node.type === "AgentCard")) {
        elements().find(node => node.props?.["aria-label"] === "Open Guide’s card").props.onClick();
        render();
      }
      return elements().find(node => node.type === "AgentCard").props.activity.props;
    },
    async poll(runs, conversation = selectedChain) {
      snapshotRuns = runs;
      selectedChain = conversation;
      assert.ok(timers.length, "poll schedules its next snapshot");
      await timers.shift()();
      render();
    },
    holdSend() {
      let resolve;
      sendResponse = new Promise((done, fail) => { resolve = done; resolve.reject = fail; });
      return resolve;
    },
    replaceConversation(value) {
      let reject;
      replacementResponse = new Promise((_, fail) => { reject = fail; });
      selectedChain = value;
      render();
      return reject;
    },
  };
}

for (const status of ["failed", "cancelled"]) {
  test(`${status} outcomes move to Activity behind newer queued, running and successful responses`, async () => {
    const previous = run("previous", status, "Context limit reached");
    const view = await chat([previous]);
    const notice = status === "failed" ? "Response failed: Context limit reached" : "Response stopped";
    assert.deepEqual(view.notices(), [notice]);
    for (const next of ["queued", "running", "completed"]) {
      await view.poll([previous, run("newer", next)]);
      assert.ok(!view.notices().some(text => /failed|stopped/.test(text)));
      assert.ok(view.activity().runs.includes(previous));
    }
  });
}

for (const status of ["failed", "cancelled"]) {
  test(`sending another message retains a ${status} response in Activity`, async () => {
    const view = await chat([run("previous", status, "Command limit reached")]);
    const finishSend = view.holdSend();
    view.composer().onChange("Try another question");
    view.render();
    view.composer().onSend(true);
    view.render();
    assert.equal(view.composer().sending, true);
    assert.ok(!view.notices().some(text => /failed|stopped/.test(text)));
    assert.equal(view.activity().runs[0].status, status);
    finishSend({
      run: run("newer", "queued"),
      conversation: { ...chain, version: 2, messages: [message("sent-message", "user", "Try another question")] },
    });
    await settle();
    view.render();
    assert.equal(view.composer().sending, false);
    assert.ok(!view.notices().some(text => /failed|stopped/.test(text)));
    assert.equal(view.activity().runs[0].status, status);
  });
}

test("only the current outcome is repeated beside Messages; all history stays in Activity", async () => {
  const view = await chat([
    run("old-failure", "failed", "Old error"), run("old-stop", "cancelled"),
    run("latest-failure", "failed", "Current error"), run("latest-stop", "cancelled"),
  ]);
  assert.deepEqual(view.notices(), ["Response stopped"]);
  await view.poll([
    run("latest-stop", "cancelled"), run("latest-failure", "failed"),
  ]);
  assert.deepEqual(view.notices(), ["Response failed: Please try again."]);
});

test("Stop updates queued and running responses while retaining earlier failures in Activity", async () => {
  const view = await chat([
    run("failed", "failed", "Context limit reached"), run("running", "running"), run("queued", "queued"),
  ]);
  view.composer().onStop();
  await settle();
  view.render();
  assert.deepEqual(view.notices(), ["Response stopped"]);
  assert.equal(view.composer().onStop, undefined);
  assert.deepEqual(view.requests.filter(item => item.path.endsWith("/cancel")).map(item => item.path), [
    "/agents/guide/runs/running/cancel", "/agents/guide/runs/queued/cancel",
  ]);
});

test("active and successful histories show no failure or stopped outcome", async () => {
  const view = await chat([run("done", "completed"), run("queued", "queued")]);
  assert.deepEqual(view.notices(), ["Response queued…"]);
  await view.poll([run("done", "completed")]);
  assert.deepEqual(view.notices(), []);
});

test("replacing the conversation hides its old outcomes while the new poll waits or fails", async () => {
  const view = await chat([run("failed", "failed", "Old error"), run("stopped", "cancelled")]);
  assert.equal(view.notices().length, 1);
  const failPoll = view.replaceConversation({ ...chain, id: "replacement" });
  assert.equal(view.requests.at(-1).path, "/agents/guide/conversations/replacement/runs");
  assert.deepEqual(view.notices(), []);
  assert.equal(view.activity().loading, true);
  failPoll(new Error("Storage offline"));
  await settle();
  view.render();
  assert.deepEqual(view.notices(), []);
  assert.deepEqual(view.alerts(), ["Updates unavailable: Storage offline"]);
  assert.equal(view.activity().loading, true);
  assert.equal(view.activity().error, "Updates unavailable: Storage offline");
});

test("activity reuses chat's snapshot without adding a polling request", async () => {
  const records = [run("queued", "queued")];
  const view = await chat(records);
  assert.equal(view.requests.length, 1);
  assert.equal(view.activity().loading, false);
  assert.equal(view.activity().runs, records);
  await view.poll([run("queued", "completed")]);
  assert.equal(view.requests.length, 2);
  assert.equal(view.activity().runs[0].status, "completed");
});

for (const generate of [true, false]) {
  test(`${generate ? "queued message B" : "saved note"} stays Delivered when an older response A arrives later`, async () => {
    const first = message("first", "user", "Question A");
    const view = await chat([run("A", "running")], { ...chain, messages: [first] });
    const finishSend = view.holdSend();
    const content = generate ? "Question B" : "A note for later";
    view.composer().onChange(content);
    view.render();
    view.composer().onSend(generate);
    view.render();
    assert.deepEqual(view.receipts(), ["Sending…"]);
    assert.equal(view.requests.at(-1).path, `/agents/guide/conversations/conversation/${generate ? "runs" : "notes"}`);
    const saved = { ...chain, version: 2, messages: [first, message("sent-message", "user", content)] };
    const queued = run("B", "queued");
    finishSend(generate ? { run: queued, conversation: saved } : saved);
    await settle();
    view.render();
    assert.deepEqual(view.receipts(), ["Delivered"]);
    await view.poll(generate ? [run("A", "completed"), queued] : [run("A", "completed")], {
      ...saved, version: 3, messages: [...saved.messages, message("reply-A", "assistant", "Reply to question A")],
    });
    assert.deepEqual(view.receipts(), ["Delivered"]);
    assert.equal(view.composer().sending, false);
  });
}

test("confirmed messages are Delivered before and after a normal reply", async () => {
  const question = message("question", "user", "A saved question");
  const view = await chat([run("answer", "queued")], { ...chain, messages: [question] });
  assert.deepEqual(view.receipts(), ["Delivered"]);
  await view.poll([run("answer", "completed")], {
    ...chain, version: 2, messages: [question, message("reply", "assistant", "An answer")],
  });
  assert.deepEqual(view.receipts(), ["Delivered"]);
});

test("assistant-only history has no user delivery receipt", async () => {
  const view = await chat([], { ...chain, messages: [message("reply", "assistant", "Hello")] });
  assert.deepEqual(view.receipts(), []);
});

test("note submission snapshots ordered attachments separately from upload confirmation", async () => {
  const view = await chat([]);
  view.attachments.entries = [{ id: "file-b" }, { id: "file-a" }];
  const finish = view.holdSend();
  view.composer().onChange("Keep these files");
  view.render();
  view.composer().onSend(false);
  view.render();
  const request = view.requests.find(request => request.body);
  assert.equal(request.path, "/agents/guide/conversations/conversation/notes");
  assert.deepEqual(Array.from(request.body.attachment_ids), ["file-b", "file-a"]);
  assert.deepEqual(view.attachmentIds(), ["file-b", "file-a"]);
  const key = outbox.chatStorageKeys("guide", chain.id).pending;
  assert.deepEqual(JSON.parse(view.storage.get(key)).attachment_ids, ["file-b", "file-a"]);
  assert.deepEqual(view.attachments.entries, []);
  finish({ ...chain, version: 2, messages: [message("sent-message", "user", "Keep these files")] });
  await settle();
  view.render();
  assert.equal(view.storage.has(key), false);
});

test("failed run retry keeps original references and attachments while newer uploads remain unresolved", async () => {
  const view = await chat([]);
  view.references().onChange([{ resource_id: "original-space", expected_version: 4, title: "Plan" }]);
  view.render();
  view.attachments.entries = [{ id: "file-b" }, { id: "file-a" }];
  const fail = view.holdSend();
  view.composer().onChange("Read these files");
  view.render();
  view.composer().onSend(true);
  const original = view.requests.find(request => request.body).body;
  assert.deepEqual(Array.from(original.attachment_ids), ["file-b", "file-a"]);
  assert.deepEqual(JSON.parse(JSON.stringify(original.references)), [{ resource_id: "original-space", expected_version: 4 }]);
  fail.reject(new Error("Lost send response"));
  await settle();
  view.attachments.entries = [{ id: "new-file" }];
  view.attachments.ready = false;
  view.render();
  view.references().onChange([{ resource_id: "new-memory", expected_version: 6, title: "New" }]);
  view.composer().onChange("New draft");
  view.render();
  const finish = view.holdSend();
  view.retry();
  const retry = view.requests.filter(request => request.body).at(-1).body;
  assert.deepEqual(JSON.parse(JSON.stringify(retry)), JSON.parse(JSON.stringify(original)));
  assert.deepEqual(view.attachments.entries, [{ id: "new-file" }]);
  view.render();
  assert.equal(view.composer().value, "New draft");
  finish({ run: run("queued", "queued"), conversation: { ...chain, version: 2, messages: [message("sent-message", "user", "Read these files")] } });
  await settle();
});

test("new sends with unresolved attachments cannot submit", async () => {
  const view = await chat([]);
  view.attachments.ready = false;
  view.composer().onChange("Wait for upload");
  view.render();
  view.composer().onSend(false);
  view.render();
  assert.equal(view.requests.filter(request => request.body).length, 0);
  assert.ok(view.alerts().some(text => /Finish uploading/.test(text)));
});

test("reloaded retry preserves attachments independently of the draft", async () => {
  for (const attachment_ids of [undefined, ["saved-b", "saved-a"]]) {
    const pending = { id: "earlier", content: "Continue", generate: true, persona: "original", ...(attachment_ids && { attachment_ids }) };
    const view = await chat([], chain, pending);
    view.attachments.entries = [{ id: "draft-file" }];
    const finish = view.holdSend();
    view.retry();
    const request = view.requests.find(request => request.body).body;
    assert.equal(request.message_id, "earlier");
    assert.equal(request.persona_id, "original");
    assert.deepEqual(Array.from(request.attachment_ids), attachment_ids ?? []);
    assert.deepEqual(view.attachments.entries, [{ id: "draft-file" }]);
    finish({ run: run("queued", "queued"), conversation: { ...chain, version: 2, messages: [message("earlier", "user", "Continue")] } });
    await settle();
  }
});

test("only saved assistant replies with text offer saving in the current agent and conversation", async () => {
  const reply = message("reply", "assistant", "Useful reply");
  const view = await chat([], { ...chain, messages: [
    message("user", "user", "Question"), reply,
    message("empty", "assistant", " "), message("tool", "tool", "Tool result"),
  ] });
  const controls = view.savedResults();
  assert.equal(controls.length, 1);
  assert.equal(controls[0].agentKey, "guide");
  assert.equal(controls[0].conversationId, chain.id);
  assert.equal(controls[0].message, reply);
});


test("normal Call remains available when text language is unavailable", async () => {
  const view = await chat([], chain, null, { ready: { storage: true, chat: false } });
  const card = view.card();
  assert.equal(card.callDisabled, false);
  assert.equal(view.callDisabled(), false);
  assert.equal(card.callTitle, "Start a voice call");
  card.onCall();
  assert.deepEqual(view.callStarts, [[]]);
});

test("Call needs Tsuchi for persona instructions", async () => {
  const view = await chat([], chain, null, { ready: { storage: false, chat: false } });
  const card = view.card();
  assert.equal(card.callDisabled, true);
  assert.equal(view.callDisabled(), true);
  assert.match(card.callTitle, /storage.*persona/);
});

test("Call reports missing browser support before starting", async () => {
  const view = await chat([], chain, null, { voiceCall: { unavailable: "Calls require HTTPS or localhost." } });
  const card = view.card();
  assert.equal(card.callDisabled, true);
  assert.equal(card.callTitle, "Calls require HTTPS or localhost.");
});


test("selected reference versions travel separately and clear on confirmed send", async () => {
  const view = await chat([]);
  view.references().onChange([{ resource_id: "space", expected_version: 4, title: "Plan" }]);
  view.composer().onChange("Use the plan");
  view.render();
  const resolve = view.holdSend();
  view.composer().onSend(true);
  view.render();
  assert.equal(view.references().selected.length, 1);
  const body = view.requests.find(item => item.body?.content === "Use the plan").body;
  assert.deepEqual(JSON.parse(JSON.stringify(body.references)), [{ resource_id: "space", expected_version: 4 }]);
  assert.equal(body.content, "Use the plan");
  resolve({ run: run("new", "queued"), conversation: { ...chain, version: 2,
    messages: [message("sent-message", "user", "Use the plan")] } });
  await settle();
  view.render();
  assert.equal(view.references().selected.length, 0);
});

test("saving a note retains selections and sends no references", async () => {
  const view = await chat([]);
  view.references().onChange([{ resource_id: "memory", expected_version: 2, title: "Fact" }]);
  view.composer().onChange("A note");
  view.render();
  const resolve = view.holdSend();
  view.composer().onSend(false);
  resolve({ ...chain, version: 2, messages: [message("sent-message", "user", "A note")] });
  await settle();
  view.render();
  assert.equal(view.references().selected.length, 1);
  assert.equal(view.requests.find(item => item.body?.content === "A note").body.references, undefined);
});
