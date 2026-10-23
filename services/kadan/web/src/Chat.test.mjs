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
const run = (id, status, error = null) => ({
  id, status, error, conversation_id: chain.id, created_at: "2026-10-23T10:00:00Z",
});
const settle = () => new Promise(resolve => setImmediate(resolve));

// Exercise Chat's poll, send and Stop handlers with in-memory hooks and API boundaries.
async function chat(initialRuns) {
  const slots = [], effects = [], timers = [], requests = [];
  const storage = new Map();
  let cursor = 0, tree, snapshotRuns = initialRuns, sendResponse, replacementResponse;
  let selectedChain = chain;
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
    "./Markdown": { default: "Markdown" },
    "./Composer": { default: "Composer", readiness: () => ({ chat: true, storage: true }) },
    "./Pet": { default: "Pet", useMood: () => ["idle", () => {}] },
    "./AgentCard": { default: "AgentCard" },
    "./ComputerViewer": { default: "ComputerViewer" },
    "./VoiceCallControl": { useVoiceCall: () => ({ active: false }) },
    "./PetContext": { useAgentPet: () => null },
    "./presence": { activeLabel: () => "Not active yet" },
    "./agentRuns": agentRuns,
    "./outbox": outbox,
    "./uuid": { randomUuid: () => "sent-message" },
    "./api": {
      agentPath: key => `/agents/${key}`,
      async api(path, body) {
        requests.push({ path, body });
        if (path === `/agents/guide/conversations/${selectedChain.id}/runs`) {
          if (body) return sendResponse;
          if (replacementResponse) return replacementResponse;
          return { runs: snapshotRuns, images: [], conversation: selectedChain };
        }
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
      entrance: "idle", onChange() {}, onBusy() {}, onEditPet() {}, blocked: false,
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
    render, requests,
    notices: () => elements().filter(node => node.props?.className === "notice").map(node => node.props.children),
    alerts: () => elements().filter(node => node.props?.role === "alert").map(node => node.props.children[0]),
    composer: () => elements().find(node => node.type === "Composer").props,
    async poll(runs) {
      snapshotRuns = runs;
      assert.ok(timers.length, "poll schedules its next snapshot");
      await timers.shift()();
      render();
    },
    holdSend() {
      let resolve;
      sendResponse = new Promise(done => { resolve = done; });
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
  test(`${status} outcomes remain visible behind newer queued, running and successful responses`, async () => {
    const previous = run("previous", status, "Context limit reached");
    const view = await chat([previous]);
    const notice = status === "failed" ? "Response failed: Context limit reached" : "Response stopped";
    assert.deepEqual(view.notices(), [notice]);
    for (const next of ["queued", "running", "completed"]) {
      await view.poll([previous, run("newer", next)]);
      const earlier = status === "failed" ? "Earlier response failed: Context limit reached" : "Earlier response stopped";
      assert.ok(view.notices().includes(earlier), `${status} remains visible during ${next}`);
    }
  });
}

for (const status of ["failed", "cancelled"]) {
  test(`sending another message does not temporarily hide a ${status} response`, async () => {
    const view = await chat([run("previous", status, "Command limit reached")]);
    const notice = status === "failed" ? "Earlier response failed: Command limit reached" : "Earlier response stopped";
    const finishSend = view.holdSend();
    view.composer().onChange("Try another question");
    view.render();
    view.composer().onSend(true);
    view.render();
    assert.equal(view.composer().sending, true);
    assert.ok(view.notices().includes(notice));
    finishSend({
      run: run("newer", "queued"),
      conversation: { ...chain, version: 2, messages: [{ id: "sent-message", role: "user" }] },
    });
    await settle();
    view.render();
    assert.equal(view.composer().sending, false);
    assert.ok(view.notices().includes(notice));
  });
}

test("only the latest failure and stop are shown, with the current outcome distinguished", async () => {
  const view = await chat([
    run("old-failure", "failed", "Old error"), run("old-stop", "cancelled"),
    run("latest-failure", "failed", "Current error"), run("latest-stop", "cancelled"),
  ]);
  assert.deepEqual(view.notices(), ["Earlier response failed: Current error", "Response stopped"]);
  await view.poll([
    run("latest-stop", "cancelled"), run("latest-failure", "failed"),
  ]);
  assert.deepEqual(view.notices(), ["Response failed: Please try again.", "Earlier response stopped"]);
});

test("Stop updates queued and running responses without hiding the earlier failure", async () => {
  const view = await chat([
    run("failed", "failed", "Context limit reached"), run("running", "running"), run("queued", "queued"),
  ]);
  view.composer().onStop();
  await settle();
  view.render();
  assert.deepEqual(view.notices(), ["Earlier response failed: Context limit reached", "Response stopped"]);
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
  assert.equal(view.notices().length, 2);
  const failPoll = view.replaceConversation({ ...chain, id: "replacement" });
  assert.equal(view.requests.at(-1).path, "/agents/guide/conversations/replacement/runs");
  assert.deepEqual(view.notices(), []);
  failPoll(new Error("Storage offline"));
  await settle();
  view.render();
  assert.deepEqual(view.notices(), []);
  assert.deepEqual(view.alerts(), ["Updates unavailable: Storage offline"]);
});
